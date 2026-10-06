# Nesting: the child's boundary conditions (nest_bcs) and the NestedGrid
# geometry. The NestBC no-op halo of N-nest-a was retired by N-nest-d
# (#261): a child's nest faces are N-open-b OpenBoundary faces.

# =========================================================================
# The child's boundary conditions: OpenBoundary faces on the nest sides
# =========================================================================

_nest_side(s::Symbol, what) = s === :nest ? :open :
    s in (:periodic, :wall, :noslip, :noflux) ? s :
    throw(ArgumentError("nest_bcs: unknown $what kind :$s (use :nest, :periodic, " *
                        ":wall, :noslip)"))

"""
    nest_bcs(child_mesh, x = :nest, y = x, z = y; bottom = z, top = z,
             moisture = false, scalars = (), correction = :uniform,
             normal_halo = :value, scalar_halo = :value, record = false)
        -> FieldBoundaryConditions

Boundary conditions of a nested child. Every `:nest` face is an
[`OpenBoundary`](@ref Aeolus.BoundaryConditions.OpenBoundary) (N-open-b,
#255) whose data come from the parent through a
[`NestBoundaryData`](@ref), attached when the child joins a
[`NestedRun`](@ref) (or by [`apply_nest_boundary!`](@ref)): the normal
velocity on the boundary face and `H` halo layers beyond it, Dirichlet
tangential velocities and scalars, and a Neumann pressure correction, so
the child's projection (the all-Neumann `CosineFFTPoissonSolver` picked
by default) sees the parent's boundary flux. The single global mass-flux
correction
([`correct_boundary_mass_flux!`](@ref Aeolus.BoundaryConditions.correct_boundary_mass_flux!))
makes that flux compatible with the solve.

`x`, `y` — `:nest` or `:periodic` (a child spanning its parent's
periodic extent). `bottom`, `top` — `:nest` on both (a child inside its
parent), `:periodic` on both, or a wall below (`:wall`, `:noslip`) and
`:nest` above: the urban child on the ground. `fields` of the open faces:
`u, v, w, θ`, `qv` when `moisture = true` (the simulation must then be
moist too), and the extra cell-centred `scalars` (names, e.g. `(:co2,)`,
matching the `scalars` of the run's [`NestInterpState`](@ref)). Pass the
result as the child's `bcs` (it is also the default `θ_bcs`).

This replaces the `NestBC` no-op halo of the Phase 16/30 nests, whose
child kept a periodic projection that never saw the parent's flux.
"""
function nest_bcs(m::RegularGrid, x::Symbol = :nest, y::Symbol = x, z::Symbol = y;
                  bottom::Symbol = z, top::Symbol = z,
                  moisture::Bool = false, scalars = (),
                  correction::Symbol = :uniform, normal_halo::Symbol = :value,
                  scalar_halo::Symbol = :value, record::Bool = false)
    xs = _nest_side(x, "x"); ys = _nest_side(y, "y")
    bs = _nest_side(bottom, "bottom"); ts = _nest_side(top, "top")
    (xs in (:open, :periodic) && ys in (:open, :periodic)) ||
        throw(ArgumentError("nest_bcs: the lateral faces must be :nest or :periodic"))
    names = Symbol[:u, :v, :w, :θ]
    moisture && push!(names, :qv)
    for sname in scalars
        sname isa Symbol || throw(ArgumentError("nest_bcs: scalars are given by name, e.g. (:co2,)"))
        sname in names && throw(ArgumentError("nest_bcs: scalar :$sname given twice or reserved"))
        push!(names, sname)
    end
    if xs === :periodic && ys === :periodic && bs !== :open && ts !== :open
        # no nest face at all: an ordinary closed / periodic box
        return FieldBoundaryConditions(:periodic, :periodic; bottom = bs, top = ts)
    end
    return open_boundary_conditions(m, nothing; x = xs, y = ys, bottom = bs, top = ts,
                                    fields = Tuple(names), correction, normal_halo,
                                    scalar_halo, record)
end

nest_bcs(x::Symbol, args...; kw...) =
    throw(ArgumentError("nest_bcs needs the child mesh: nest_bcs(child_mesh, :$x, …). " *
                        "A nest face is an OpenBoundary, which holds per-face buffers " *
                        "on that mesh (N-nest-d, #261)."))

# =========================================================================
# NestedGrid
# =========================================================================

"""
    NestedGrid(parent_mesh, child_mesh; refinement, offset)

Geometric description of a child domain embedded in a parent.

  * `refinement` — integer ratio per dimension, `Int` or `NTuple{3,Int}`
    (`rx, ry, rz`; commonly 2 or 3).
  * `offset` — `(ix, iy, iz)`, the 1-based *interior* index of the first
    parent cell the child covers.

The constructor validates that the child aligns to parent cell
boundaries, that `child.N` is divisible by the refinement, that
`child.d * r == parent.d`, and that the covered box fits inside the
parent's interior. Any violation raises a descriptive error.

Use [`nested_grid`](@ref) to build the child mesh from a physical extent.
"""
struct NestedGrid{T, MP<:RegularGrid{T}, MC<:RegularGrid{T}}
    parent_mesh::MP
    child_mesh::MC
    rx::Int; ry::Int; rz::Int
    ix::Int; iy::Int; iz::Int
    nx::Int; ny::Int; nz::Int
    ledger::Dict{Symbol,Float64}
    plans::Dict{Tuple{Symbol,Int,Int},Vector{Tuple{Int,Vector{Float64}}}}
end

_triple(r::Integer) = (Int(r), Int(r), Int(r))
_triple(r::NTuple{3,Integer}) = (Int(r[1]), Int(r[2]), Int(r[3]))

function _near(a::Real, b::Real; rtol=1e-9)
    return abs(a - b) <= rtol * max(one(Float64), abs(Float64(a)), abs(Float64(b)))
end

function NestedGrid(parent::RegularGrid{T}, child::RegularGrid{T};
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
                            ("y", child.dy, parent.dy, ry),
                            ("z", child.dz, parent.dz, rz))
        _near(dc * r, dp) || error("NestedGrid: child d$nm * r$nm = $(dc*r) does not " *
                                   "match parent d$nm = $dp (refinement mismatch).")
    end

    for (nm, c0, p0, dp, i) in (("x", child.x0, parent.x0, parent.dx, ix),
                                ("y", child.y0, parent.y0, parent.dy, iy),
                                ("z", child.z0, parent.z0, parent.dz, iz))
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

    return NestedGrid{T, typeof(parent), typeof(child)}(
        parent, child, rx, ry, rz, ix, iy, iz, nx, ny, nz,
        Dict{Symbol,Float64}(),
        Dict{Tuple{Symbol,Int,Int},Vector{Tuple{Int,Vector{Float64}}}}())
end

"""
    nested_grid(parent_mesh, child_extent, refinement; halo=parent halo)

Build the child mesh and the [`NestedGrid`](@ref) that links it to
`parent_mesh`. `child_extent` is `((x₁,x₂), (y₁,y₂), (z₁,z₂))` in
physical units; each endpoint must land on a parent cell boundary (an
error names the offending direction otherwise). `refinement` is an `Int`
or `NTuple{3,Int}`.
"""
function nested_grid(parent::RegularGrid{T}, child_extent, refinement;
                     halo::NTuple{3,Int}=(parent.Hx, parent.Hy, parent.Hz)) where {T}
    rx, ry, rz = _triple(refinement)
    (ax, bx), (ay, by), (az, bz) = child_extent
    ix, nx = _align_extent(ax, bx, parent.x0, parent.dx, parent.Nx, "x")
    iy, ny = _align_extent(ay, by, parent.y0, parent.dy, parent.Ny, "y")
    iz, nz = _align_extent(az, bz, parent.z0, parent.dz, parent.Nz, "z")
    child = RegularGrid(T;
        size   = (nx * rx, ny * ry, nz * rz),
        extent = (nx * parent.dx, ny * parent.dy, nz * parent.dz),
        halo   = halo,
        origin = (parent.x0 + (ix - 1) * parent.dx,
                  parent.y0 + (iy - 1) * parent.dy,
                  parent.z0 + (iz - 1) * parent.dz))
    return NestedGrid(parent, child; refinement=(rx, ry, rz), offset=(ix, iy, iz))
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
