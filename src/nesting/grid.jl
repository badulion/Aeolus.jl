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
             normal_halo = :value, scalar_halo = :value, record = false,
             outflow = :dirichlet, halo = 1)
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

`outflow` and `halo` are those of
[`open_boundary_conditions`](@ref Aeolus.BoundaryConditions.open_boundary_conditions):
the outflow condition of the nest faces (`:dirichlet`, PALM's nested
child and the default, or a `:convective` / `:orlanski` radiation
outflow) and the halo depth the child's advection scheme needs there.

This replaces the `NestBC` no-op halo of the Phase 16/30 nests, whose
child kept a periodic projection that never saw the parent's flux.
"""
function nest_bcs(m::NestChildMesh, x::Symbol = :nest, y::Symbol = x, z::Symbol = y;
                  bottom::Symbol = z, top::Symbol = z,
                  moisture::Bool = false, scalars = (),
                  correction::Symbol = :uniform, normal_halo::Symbol = :value,
                  scalar_halo::Symbol = :value, record::Bool = false,
                  outflow = :dirichlet, halo::Integer = 1)
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
                                    scalar_halo, record, outflow, halo)
end

nest_bcs(x::Symbol, args...; kw...) =
    throw(ArgumentError("nest_bcs needs the child mesh: nest_bcs(child_mesh, :$x, …). " *
                        "A nest face is an OpenBoundary, which holds per-face buffers " *
                        "on that mesh (N-nest-d, #261)."))

# =========================================================================
# 1-D prolongation plans and the vertical measure (types; the builders are
# in operators.jl and measure.jl)
# =========================================================================

# For every child storage index along one dimension: the parent storage
# index where the stencil starts, the `ns` stencil weights (column `ic` of
# `w`) and, for a cell-average plan, the parent storage cell holding the
# child cell (the limiter groups children by it; 0 for a nodal plan). Flat
# arrays rather than a vector of (start, weights) tuples (#329): one
# allocation per plan and a concrete, GPU-friendly layout.
#
# `clo[q]:chi[q]` are the child indices whose parent cell is `c1 + q - 1`
# (contiguous, since `cell` is non-decreasing): the inverse of `cell`, so the
# limiter's parent cells gather their children (KA-f, #297). Empty for a
# nodal plan.
struct _Plan1D
    start::Vector{Int}
    w::Matrix{Float64}
    cell::Vector{Int}
    clo::Vector{Int}
    chi::Vector{Int}
    c1::Int
end

function _Plan1D(start::Vector{Int}, w::Matrix{Float64}, cell::Vector{Int})
    (isempty(cell) || first(cell) == 0) && return _Plan1D(start, w, cell, Int[], Int[], 1)
    issorted(cell) || error("_Plan1D: the parent cells of a cell-average plan must be non-decreasing")
    c1 = first(cell); n = last(cell) - c1 + 1
    clo = fill(1, n); chi = fill(0, n)            # empty ranges by default
    for ic in length(cell):-1:1
        clo[cell[ic] - c1 + 1] = ic
    end
    for ic in eachindex(cell)
        chi[cell[ic] - c1 + 1] = ic
    end
    return _Plan1D(start, w, cell, clo, chi, c1)
end

Base.length(p::_Plan1D) = length(p.start)
Base.getindex(p::_Plan1D, ic::Integer) = (p.start[ic], view(p.w, :, ic))
_stencil(p::_Plan1D) = size(p.w, 1)

"""
    NestMeasure

Vertical measure of the nesting operators of one [`NestedGrid`](@ref):
the coordinate in which the conservative prolongation reconstructs
parent averages and the weights with which restriction averages child
cells. Build with [`nest_measure`](@ref).

  * `kind = :volume` — volume (height) measure. On a uniform parent the
    operators are exactly the N-nest-a (#250) operators; on a z-stretched
    parent the reconstruction uses the actual face heights.
  * `kind = :mass` — the reference-mass measure `dμ = W(z) dz` of a
    [`MassWeightedSimulation`](@ref) pair (`W = ρ̄` anelastic, `ρ̄θ̄`
    pseudo-incompressible, from `constraint_weights`). Restriction is
    `W`-weighted (scalars) and `W`-area-weighted (horizontal-normal
    faces); prolongation conserves `∫ W φ dV` exactly.

`zlen` are the overlaps of every conservation-box parent layer with the
child layers in the measure coordinate (the rows of the vertical
[`NestAxis`](@ref)): the weights of the general (non-aligned, stretched
child) restriction.

Fields (all storage-indexed, halos included): `pc`/`cc` the parent/child
face coordinates in the measure, `ph`/`ch` the parent/child face heights
(child faces that coincide with parent faces are bit-identical copies),
`cw` the child cell measure per level (`Δz` or `W Δz`), `pw` the parent
cell measure per level from the parent's *own* weight (`Δz` or
`W_p Δz_p`), and the plan cache.

Two parent cell masses exist in the mass measure. Inside the covered
layers the coordinate `pc` takes a parent layer's mass to be the sum of
its children's (`Σ W_c Δz_c`), which is what makes prolongation and
restriction exactly conservative and the Lagrange basis exactly 0/1 on
coincident faces. The parent simulation itself conserves `W_p Δz_p`,
which differs from that sum by the midpoint-rule error of sampling one
profile on two grids ([`nest_reference_mismatch`](@ref), `~4e-7` for a
25 m layer). Cell integrals over *uncovered* parent cells
([`composite_integral`](@ref), the refluxing correction) therefore use
`pw`, so the composite integral is exactly what the two simulations
conserve; covered parent cells only ever exchange with their own
children and use `pc`.
"""
struct NestMeasure
    kind::Symbol
    pc::Vector{Float64}
    cc::Vector{Float64}
    ph::Vector{Float64}
    ch::Vector{Float64}
    cw::Vector{Float64}
    pw::Vector{Float64}
    zlen::Vector{Float64}       # overlaps of the box layers with the child layers, in the measure (#328)
    plans::Dict{Tuple{Symbol,Int},_Plan1D}
end

# =========================================================================
# NestedGrid
# =========================================================================

# (:volume, 0, 0, :none), or (:mass, objectid(parent ref), objectid(child ref), weighting)
const _MeasureKey = Tuple{Symbol,UInt,UInt,Symbol}
const _VOLUME_KEY = (:volume, UInt(0), UInt(0), :none)

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
Without `refinement` and `offset`, `NestedGrid(parent_mesh, child_mesh)`
infers the geometry and also accepts rational ratios, children off the
parent faces and z-stretched children (#328); see that method. The
geometry along each axis is a [`NestAxis`](@ref) (`nest.axes`); `nx, ny,
nz` and `ix, iy, iz` are the size and offset of the conservation box (the
whole child here), `rx, ry, rz` the integer ratios (0 along an axis that
is not integer-aligned), and `aligned` is true when every axis is
integer-aligned with a uniform child (the N-nest-a operators).
"""
struct NestedGrid{T, MP<:AbstractMesh{T}, MC<:AbstractMesh{T}}
    parent_mesh::MP
    child_mesh::MC
    rx::Int; ry::Int; rz::Int      # integer ratio per axis, 0 where the axis is not integer-aligned
    ix::Int; iy::Int; iz::Int      # first parent interior cell of the conservation box
    nx::Int; ny::Int; nz::Int      # conservation box size, in parent cells
    axes::NTuple{3,NestAxis}       # the per-axis geometry (#328)
    aligned::Bool                  # integer-aligned on every axis, uniform child: N-nest-a operators
    ledger::Dict{Symbol,Float64}
    plans::Dict{Tuple{Symbol,Int,Int},_Plan1D}
    measures::Dict{_MeasureKey,NestMeasure}   # cached measures (volume / mass)
    limscratch::Dict{NTuple{3,Int},NTuple{3,Array{Float64,3}}}   # prolongation limiter
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

function _nested_grid_aligned(parent::NestParentMesh{T}, child::RegularGrid{T},
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

    axs = _nest_axes(parent, child, true)
    for (d, r, o) in ((1, rx, ix), (2, ry, iy), (3, rz, iz))
        axs[d].integer == (r, o) ||
            error("NestedGrid: internal: axis $d geometry $(axs[d]) does not match the " *
                  "refinement $r and offset $o")
    end
    return _make_nest(parent, child, (rx, ry, rz), (ix, iy, iz), (nx, ny, nz), axs, true)
end

_make_nest(parent::AbstractMesh{T}, child::AbstractMesh{T}, r, o, n, axs, aligned) where {T} =
    NestedGrid{T, typeof(parent), typeof(child)}(
        parent, child, r..., o..., n..., axs, aligned,
        Dict{Symbol,Float64}(), Dict{Tuple{Symbol,Int,Int},_Plan1D}(),
        Dict{_MeasureKey,NestMeasure}(), Dict{NTuple{3,Int},NTuple{3,Array{Float64,3}}}())

# The three axis geometries of a parent–child pair.
_nest_axes(parent, child, integer::Bool) =
    ntuple(d -> NestAxis(_mesh_faces(parent, d), _mesh_HN(parent, d)...,
                         _mesh_faces(child, d), _mesh_HN(child, d)...;
                         uniform_child = d < 3 || child isa RegularGrid, integer), 3)

"""
    NestedGrid(parent_mesh, child_mesh; plan = :auto)

The [`NestedGrid`](@ref) of a child placed anywhere inside its parent,
with the geometry inferred from the two meshes (#328): the refinement may
be rational (`Δ_p/Δ_c = 5/2`) or 1, the child's origin and extent need
not lie on parent faces, and the child may be stretched in z (a
[`StretchedGrid`](@ref) child under a uniform or stretched parent).

Requirements (an error names the axis otherwise):

  * the child interior lies inside the parent interior;
  * the child is at least as fine as the parent wherever they overlap;
  * the child's storage (halos included) lies inside the parent's, so
    every child halo stencil reads parent data.

The *conservation box* is, along each axis, the parent cells between the
first and the last parent face inside the child that is also a child face
(a common face). Two-way transfers (anterpolation, refluxing, interface
flux correction, composite integrals) act on the box; the strip between
the box and the child's edge is computed by the child but owned by the
parent. When the child's edges lie on parent faces the box is the whole
child. A child with no box along some axis can only be nested one-way.
`nest_covered_range` returns the box, `nx, ny, nz` and `ix, iy, iz` are
its size and offset, and `refinement_ratio` gives the per-axis ratio
(`nothing` along a stretched z).

When every axis is aligned with an integer ratio and the child is uniform,
the result is identical to `NestedGrid(parent, child; refinement, offset)`
and every operator takes the N-nest-a integer path. `plan = :overlap`
forces the general overlap operators even then (they reproduce the integer
operators to round-off, which the tests check).
"""
function NestedGrid(parent::NestParentMesh{T}, child::NestChildMesh{T};
                    refinement = nothing, offset = nothing, plan::Symbol = :auto) where {T}
    if refinement !== nothing || offset !== nothing
        (refinement === nothing || offset === nothing) &&
            throw(ArgumentError("NestedGrid: give both refinement and offset, or neither " *
                                "(the geometry is then inferred)"))
        child isa RegularGrid ||
            throw(ArgumentError("NestedGrid: refinement/offset describe a uniform child; build " *
                                "a stretched child with NestedGrid(parent, child)"))
        plan === :auto ||
            throw(ArgumentError("NestedGrid: plan = :$plan needs the inferred geometry, " *
                                "NestedGrid(parent, child; plan)"))
        return _nested_grid_aligned(parent, child, refinement, offset)
    end
    plan in (:auto, :overlap) ||
        throw(ArgumentError("NestedGrid: plan must be :auto or :overlap, got :$plan"))
    axs = _nest_axes(parent, child, plan === :auto)
    names = ("x", "y", "z")
    for d in 1:3
        a = axs[d]
        tol = 1e-9 * (a.cf[end] - a.cf[1])
        a.cf[a.Hc+1] >= a.pf[a.Hp+1] - tol && a.cf[a.Hc+a.Nc+1] <= a.pf[a.Hp+a.Np+1] + tol ||
            error("NestedGrid: the child's $(names[d]) extent [$(a.cf[a.Hc+1]), " *
                  "$(a.cf[a.Hc+a.Nc+1])] leaves the parent interior [$(a.pf[a.Hp+1]), " *
                  "$(a.pf[a.Hp+a.Np+1])].")
        a.cf[1] >= a.pf[1] - tol && a.cf[end] <= a.pf[end] + tol ||
            error("NestedGrid: the child's $(names[d]) storage (halos included) " *
                  "[$(a.cf[1]), $(a.cf[end])] reaches beyond the parent's storage " *
                  "[$(a.pf[1]), $(a.pf[end])]: the child halo stencils would read no parent " *
                  "data. Move the child away from the parent's edge or deepen the parent halo.")
        for c in a.Hc+1:a.Hc+a.Nc
            h = a.cf[c+1] - a.cf[c]
            P = clamp(searchsortedlast(a.pf, (a.cf[c] + a.cf[c+1]) / 2), 1, length(a.pf) - 1)
            h <= (1 + 1e-9) * (a.pf[P+1] - a.pf[P]) ||
                error("NestedGrid: the child is coarser than its parent along $(names[d]) " *
                      "(child cell $(c - a.Hc) is $h thick, the parent cell around it " *
                      "$(a.pf[P+1] - a.pf[P])). A nest must refine its parent.")
        end
    end
    r = ntuple(d -> axs[d].integer === nothing ? 0 : axs[d].integer[1], 3)
    o = ntuple(d -> isempty(axs[d].pbox) ? 1 : first(axs[d].pbox), 3)
    n = ntuple(d -> length(axs[d].pbox), 3)
    aligned = all(>(0), r) && child isa RegularGrid
    return _make_nest(parent, child, r, o, n, axs, aligned)
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

**Non-aligned children** (#328). On a uniform parent `refinement` may
also be rational (`5//2`: five child cells in two parent cells) and the
endpoints need not lie on parent faces: the child then gets
`Δ_c = Δ_p / r`, its origin at `(x₁, y₁, z₁)`, and each extent must be a
whole number of child cells. The [`NestedGrid`](@ref) is built from the
geometry (`NestedGrid(parent, child)`), with its conservation box between
the first and last common faces.
"""
function nested_grid(parent::NestParentMesh{T}, child_extent, refinement;
                     halo::NTuple{3,Int}=(parent.Hx, parent.Hy, parent.Hz)) where {T}
    rr = refinement isa Tuple ? refinement : (refinement, refinement, refinement)
    length(rr) == 3 || throw(ArgumentError("nested_grid: refinement is a number or a 3-tuple"))
    if !(all(r -> r isa Integer, rr) && _extent_on_faces(parent, child_extent))
        return _nested_grid_general(parent, child_extent, rr, halo)
    end
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

# Whether every endpoint of the extent is a parent face (the N-nest-a case).
function _extent_on_faces(p, ext)
    (ax, bx), (ay, by), (az, bz) = ext
    onf(a, o, d) = _near((a - o) / d, round((a - o) / d))
    zf = _zfaces_storage(p)
    onz(a) = any(z -> _near(z, a), zf)
    return onf(ax, p.x0, p.dx) && onf(bx, p.x0, p.dx) && onf(ay, p.y0, p.dy) &&
           onf(by, p.y0, p.dy) && onz(az) && onz(bz)
end

function _nested_grid_general(parent::NestParentMesh{T}, ext, rr, halo) where {T}
    parent isa RegularGrid ||
        error("nested_grid: a rational refinement or an extent off the parent faces needs a " *
              "uniform (RegularGrid) parent; on a stretched parent build the child mesh and " *
              "call NestedGrid(parent, child)")
    Δp = (parent.dx, parent.dy, parent.dz)
    N = zeros(Int, 3); L = zeros(Float64, 3); o = zeros(Float64, 3)
    for d in 1:3
        r = rr[d]
        (r isa Real && r >= 1) ||
            error("nested_grid: refinement must be ≥ 1 per axis, got $r")
        a, b = ext[d]
        b > a || error("nested_grid: empty child extent along axis $d")
        Δc = Δp[d] / r
        n = round(Int, (b - a) / Δc)
        abs(n * Δc - (b - a)) <= 1e-9 * (b - a) ||
            error("nested_grid: the child extent $(b - a) along axis $d is not a whole " *
                  "number of child cells Δ_p/r = $Δc")
        N[d] = n; L[d] = n * Δc; o[d] = a
    end
    child = RegularGrid(T; size = Tuple(N), extent = Tuple(L), halo = halo, origin = Tuple(o))
    return NestedGrid(parent, child)
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

"""
    refinement_ratio(nest) -> NTuple{3}

The refinement per axis: the integer ratio of an integer-aligned axis, the
rational `Δ_p/Δ_c` (to 1e-9) of a uniform non-integer axis, `nothing`
along a stretched or non-uniform z (#328).
"""
refinement_ratio(n::NestedGrid) = ntuple(d -> _axis_ratio(n, d), 3)

function _axis_ratio(n::NestedGrid, d::Int)
    r = (n.rx, n.ry, n.rz)[d]
    r > 0 && return r
    d == 3 && !(n.child_mesh isa RegularGrid && n.parent_mesh isa RegularGrid) && return nothing
    Δp = d == 1 ? n.parent_mesh.dx : d == 2 ? n.parent_mesh.dy : n.parent_mesh.dz
    Δc = d == 1 ? n.child_mesh.dx : d == 2 ? n.child_mesh.dy : n.child_mesh.dz
    return rationalize(Int, Δp / Δc; tol = 1e-9 * Δp / Δc)
end

# The largest refinement over the three axes (the default number of child
# substeps of a fixed-step run), from the cell sizes where no ratio is integer.
function _max_ratio(n::NestedGrid)
    r = 1.0
    for d in 1:3
        a = n.axes[d]
        rd = (n.rx, n.ry, n.rz)[d]
        if rd > 0
            r = max(r, Float64(rd))
        else
            hp = minimum(diff(a.pf[a.Hp+1:a.Hp+a.Np+1]))
            hc = minimum(diff(a.cf[a.Hc+1:a.Hc+a.Nc+1]))
            r = max(r, hp / hc)
        end
    end
    return r
end
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
