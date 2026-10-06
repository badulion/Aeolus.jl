# Nesting: the NestBC halo marker and the NestedGrid geometry (moved verbatim
# from AdaptiveMesh.jl, N-nest-a #250).

# =========================================================================
# Nest boundary condition: a no-op `fill_halos!` so nest halo data survives
# =========================================================================

"""
    NestBC()

Boundary condition marking a face as a *nest* boundary. `fill_halos!`
deliberately does nothing on such a face — the halo is owned by
[`interpolate_to_nest_boundary!`](@ref), which writes the parent's
interpolated state there. Without this, `step!`'s internal
`fill_halos!` would overwrite the nest halo with a periodic wrap.
"""
struct NestBC <: AbstractBC end

_fill_x!(d, ::NestBC, ::NestBC, Hx, Nx, ::Location) = d
_fill_y!(d, ::NestBC, ::NestBC, Hy, Ny, ::Location) = d
_fill_z!(d, ::NestBC, ::NestBC, Hz, Nz, ::Location) = d

_bck(s::Symbol) = s === :nest     ? NestBC()          :
                  s === :periodic ? Periodic()        :
                  s === :noflux   ? NoFlux()          :
                  s === :wall     ? ImpermeableWall() :
                  s === :noslip   ? NoSlipWall()      :
                  error("nest_bcs: unknown BC kind $s")

"""
    nest_bcs(x=:nest, y=:nest, z=:nest)

`FieldBoundaryConditions` for a nested child. `:nest` faces are left
alone by `fill_halos!`; `:periodic`, `:noflux`, `:wall`, `:noslip` behave as usual.
A child that spans the parent's full extent in a direction should use the
parent's BC there rather than `:nest`.
"""
function nest_bcs(x::Symbol=:nest, y::Symbol=x, z::Symbol=y)
    bx = _bck(x); by = _bck(y); bz = _bck(z)
    return FieldBoundaryConditions(bx, bx, by, by, bz, bz)
end

# =========================================================================
# NestedGrid
# =========================================================================

"""
    NestedGrid(parent_mesh, child_mesh; refinement, offset)

Geometric description of a child domain embedded in a parent.

  * `parent_mesh` — a [`RegularGrid`](@ref) or a z-stretched
    [`StretchedGrid`](@ref) (N-nest-b, #256); `child_mesh` is a
    `RegularGrid`.
  * `refinement` — integer ratio per dimension, `Int` or `NTuple{3,Int}`
    (`rx, ry, rz`; commonly 2 or 3).
  * `offset` — `(ix, iy, iz)`, the 1-based *interior* index of the first
    parent cell the child covers.

The constructor validates that the child aligns to parent cell
boundaries, that `child.N` is divisible by the refinement, that
`child.d * r == parent.d`, and that the covered box fits inside the
parent's interior. Any violation raises a descriptive error.

On a stretched parent the vertical checks use the parent's actual face
heights ([`zfaces`](@ref)) and cell thicknesses
([`cell_thickness`](@ref)): the child's bottom and top must sit on
parent faces and **every covered parent layer must be exactly `rz` child
layers thick** — an integer number of child layers per parent layer in
the overlap, so the overlap is uniform in the parent. The parent may
stretch freely above (or below) the overlap; the halo stencils that
reach into those layers interpolate on the actual face heights.

Use [`nested_grid`](@ref) to build the child mesh from a physical extent.
"""
struct NestedGrid{T, MP<:AbstractMesh{T}, MC<:RegularGrid{T}}
    parent_mesh::MP
    child_mesh::MC
    rx::Int; ry::Int; rz::Int
    ix::Int; iy::Int; iz::Int
    nx::Int; ny::Int; nz::Int
    ledger::Dict{Symbol,Float64}
    plans::Dict{Tuple{Symbol,Int,Int},Vector{Tuple{Int,Vector{Float64}}}}
    measures::Dict{Any,Any}          # cached NestMeasure objects (volume / mass)
end

"Parent meshes a [`NestedGrid`](@ref) accepts."
const NestParentMesh{T} = Union{RegularGrid{T}, StretchedGrid{T}}

_triple(r::Integer) = (Int(r), Int(r), Int(r))
_triple(r::NTuple{3,Integer}) = (Int(r[1]), Int(r[2]), Int(r[3]))

function _near(a::Real, b::Real; rtol=1e-9)
    return abs(a - b) <= rtol * max(one(Float64), abs(Float64(a)), abs(Float64(b)))
end

# --- vertical metrics of a parent mesh: one source of truth ---------------
# A RegularGrid has the affine faces z0 + (k-Hz-1)·dz; a StretchedGrid has
# its own `zfaces` / `cell_thickness` (StretchedGrids, #151). Everything in
# Nesting that needs a parent layer height goes through these two.

"Storage-indexed bottom-face heights of `m` (length `Nz + 2Hz + 1`)."
_zfaces_storage(m::RegularGrid) =
    Float64[m.z0 + (k - m.Hz - 1) * m.dz for k in 1:m.Nz+2m.Hz+1]
_zfaces_storage(m::StretchedGrid) = Float64.(zfaces(m))

"Thickness of the parent cell at *storage* level `s`."
_dz_storage(m::RegularGrid, s::Integer) = m.dz
_dz_storage(m::StretchedGrid, s::Integer) = cell_thickness(m)[s]

"Thickness of parent *interior* level `K`."
_dzp(m, K::Integer) = _dz_storage(m, m.Hz + K)

"Parent cell volume at interior level `K`."
_vol_p(m, K::Integer) = m.dx * m.dy * _dzp(m, K)

function _check_z(parent::RegularGrid, child::RegularGrid, rz, iz, nz)
    _near(child.dz * rz, parent.dz) ||
        error("NestedGrid: child dz * rz = $(child.dz*rz) does not match parent " *
              "dz = $(parent.dz) (refinement mismatch).")
    f = (child.z0 - parent.z0) / parent.dz
    _near(f, iz - 1) || error("NestedGrid: child origin z = $(child.z0) sits $f parent " *
                              "cells from the parent origin, which is not consistent " *
                              "with offset index $iz (expected $(iz-1)). The child must " *
                              "align to parent cell boundaries.")
    return nothing
end

function _check_z(parent::StretchedGrid, child::RegularGrid, rz, iz, nz)
    zf = zfaces(parent); Hz = parent.Hz
    (1 <= iz && iz + nz - 1 <= parent.Nz) ||
        error("NestedGrid: child covers parent z-cells $iz:$(iz+nz-1), outside the " *
              "parent interior 1:$(parent.Nz).")
    _near(child.z0, zf[Hz + iz]) ||
        error("NestedGrid: child origin z = $(child.z0) is not the parent face " *
              "z = $(zf[Hz+iz]) at the bottom of parent layer $iz. The child must " *
              "align to parent faces.")
    for K in iz:iz+nz-1
        dzK = _dzp(parent, K)
        _near(child.dz * rz, dzK) ||
            error("NestedGrid: parent layer $K is $dzK thick but rz · child dz = " *
                  "$(child.dz*rz): every parent layer in the overlap must hold exactly " *
                  "rz = $rz child layers (keep the overlap in the parent's uniform region).")
    end
    return nothing
end

function NestedGrid(parent::NestParentMesh{T}, child::RegularGrid{T};
                    refinement, offset::NTuple{3,Integer}) where {T}
    rx, ry, rz = _triple(refinement)
    ix, iy, iz = Int.(offset)
    (rx >= 1 && ry >= 1 && rz >= 1) ||
        error("NestedGrid: refinement must be >= 1 per dimension, got ($rx,$ry,$rz)")

    for (nm, Nc, r) in (("x", child.Nx, rx), ("y", child.Ny, ry), ("z", child.Nz, rz))
        Nc % r == 0 || error("NestedGrid: child N$nm = $Nc is not divisible by " *
                             "the refinement ratio r$nm = $r — the child must align " *
                             "to parent cell boundaries.")
    end
    nx = child.Nx ÷ rx; ny = child.Ny ÷ ry; nz = child.Nz ÷ rz

    for (nm, dc, dp, r) in (("x", child.dx, parent.dx, rx),
                            ("y", child.dy, parent.dy, ry))
        _near(dc * r, dp) || error("NestedGrid: child d$nm * r$nm = $(dc*r) does not " *
                                   "match parent d$nm = $dp (refinement mismatch).")
    end

    for (nm, c0, p0, dp, i) in (("x", child.x0, parent.x0, parent.dx, ix),
                                ("y", child.y0, parent.y0, parent.dy, iy))
        f = (c0 - p0) / dp
        _near(f, i - 1) || error("NestedGrid: child origin $nm = $c0 sits $f parent " *
                                 "cells from the parent origin, which is not consistent " *
                                 "with offset index $i (expected $(i-1)). The child must " *
                                 "align to parent cell boundaries.")
    end

    for (nm, i, n, Np) in (("x", ix, nx, parent.Nx), ("y", iy, ny, parent.Ny),
                           ("z", iz, nz, parent.Nz))
        (1 <= i && i + n - 1 <= Np) ||
            error("NestedGrid: child covers parent $nm-cells $i:$(i+n-1), outside the " *
                  "parent interior 1:$Np.")
    end
    _check_z(parent, child, rz, iz, nz)

    return NestedGrid{T, typeof(parent), typeof(child)}(
        parent, child, rx, ry, rz, ix, iy, iz, nx, ny, nz,
        Dict{Symbol,Float64}(),
        Dict{Tuple{Symbol,Int,Int},Vector{Tuple{Int,Vector{Float64}}}}(),
        Dict{Any,Any}())
end

"""
    nested_grid(parent_mesh, child_extent, refinement; halo=parent halo)

Build the child mesh and the [`NestedGrid`](@ref) that links it to
`parent_mesh` (a `RegularGrid` or a z-stretched `StretchedGrid`).
`child_extent` is `((x₁,x₂), (y₁,y₂), (z₁,z₂))` in physical units; each
endpoint must land on a parent cell boundary (an error names the
offending direction otherwise). On a stretched parent `z₁`, `z₂` must be
parent face heights and the parent layers between them must all have
the same thickness (the child is uniform, `dz = dz_parent / rz`).
`refinement` is an `Int` or `NTuple{3,Int}`.
"""
function nested_grid(parent::NestParentMesh{T}, child_extent, refinement;
                     halo::NTuple{3,Int}=(parent.Hx, parent.Hy, parent.Hz)) where {T}
    rx, ry, rz = _triple(refinement)
    (ax, bx), (ay, by), (az, bz) = child_extent
    ix, nx = _align_extent(ax, bx, parent.x0, parent.dx, parent.Nx, "x")
    iy, ny = _align_extent(ay, by, parent.y0, parent.dy, parent.Ny, "y")
    iz, nz = _align_extent_z(parent, az, bz)
    dzp = _dzp(parent, iz)
    zf  = _zfaces_storage(parent)
    child = RegularGrid(T;
        size   = (nx * rx, ny * ry, nz * rz),
        extent = (nx * parent.dx, ny * parent.dy, nz * dzp),
        halo   = halo,
        origin = (parent.x0 + (ix - 1) * parent.dx,
                  parent.y0 + (iy - 1) * parent.dy,
                  zf[parent.Hz + iz]))
    return NestedGrid(parent, child; refinement=(rx, ry, rz), offset=(ix, iy, iz))
end

_align_extent_z(p::RegularGrid, a, b) = _align_extent(a, b, p.z0, p.dz, p.Nz, "z")

function _align_extent_z(p::StretchedGrid, a, b)
    zf = zfaces(p); Hz = p.Hz
    fi = Hz+1:Hz+p.Nz+1
    ia = findfirst(k -> _near(zf[k], a), fi)
    ib = findfirst(k -> _near(zf[k], b), fi)
    ia === nothing && error("nested_grid: child z-start $a is not a face of the " *
                            "stretched parent (faces $(zf[first(fi)]) … $(zf[last(fi)])).")
    ib === nothing && error("nested_grid: child z-end $b is not a face of the " *
                            "stretched parent (faces $(zf[first(fi)]) … $(zf[last(fi)])).")
    n = ib - ia
    n >= 1 || error("nested_grid: child z extent must span at least one parent cell.")
    return (ia, n)
end

function _align_extent(a, b, o, d, N, nm)
    fa = (a - o) / d; fb = (b - o) / d
    ia = round(Int, fa); ib = round(Int, fb)
    _near(fa, ia) || error("nested_grid: child $nm-start $a is $fa parent cells from " *
                           "the parent origin — not a parent cell boundary.")
    _near(fb, ib) || error("nested_grid: child $nm-end $b is $fb parent cells from " *
                           "the parent origin — not a parent cell boundary.")
    n = ib - ia
    n >= 1 || error("nested_grid: child $nm extent must span at least one parent cell.")
    (0 <= ia && ib <= N) || error("nested_grid: child $nm extent [$a,$b] leaves the " *
                                  "parent domain (parent cells 1:$N).")
    return (ia + 1, n)
end

refinement_ratio(n::NestedGrid) = (n.rx, n.ry, n.rz)
nest_offset(n::NestedGrid)      = (n.ix, n.iy, n.iz)
nest_parent_mesh(n::NestedGrid) = n.parent_mesh
nest_child_mesh(n::NestedGrid)  = n.child_mesh

"Parent interior index ranges covered by the nest."
nest_covered_range(n::NestedGrid) = (n.ix : n.ix + n.nx - 1,
                                     n.iy : n.iy + n.ny - 1,
                                     n.iz : n.iz + n.nz - 1)

function Base.show(io::IO, n::NestedGrid)
    print(io, "NestedGrid(r=", refinement_ratio(n), ", parent cells ",
          nest_covered_range(n), ", child size=(",
          n.child_mesh.Nx, ",", n.child_mesh.Ny, ",", n.child_mesh.Nz, "))")
end
