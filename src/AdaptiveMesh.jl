module AdaptiveMesh

# =========================================================================
# Phases 30-31 — production grid nesting + block-structured AMR.
# =========================================================================
#
# This module supersedes `src/Nesting.jl` (a refine=1 direct-copy
# scaffold). Everything here is built around three primitives:
#
#   1. A *conservative, high-order* prolongation (parent -> child).
#      Implemented via the primitive-function (cumulative-integral)
#      construction: the parent cell averages define the primitive `F`
#      exactly at parent faces; a degree-`ns` polynomial interpolates `F`
#      there; the child cell average is the polynomial's difference
#      quotient across the child cell. Because `F` is matched *exactly* at
#      the parent faces, the r sub-cell averages inside a parent cell sum
#      back to the parent average to machine precision — conservation is
#      structural, not approximate. `ns = order` parent cells in the
#      stencil gives a formal order-`order` reconstruction.
#
#   2. A *conservative* restriction (child -> parent): volume-weighted
#      average for cell-centred fields, area-weighted average for
#      face-located (flux) fields.
#
#   3. A *flux register / refluxing* correction at the coarse-fine
#      interface (Berger-Colella). Without it two-way nesting leaks: the
#      coarse cells outside the nest were advanced with the coarse flux
#      while the fine grid used the fine fluxes, so the composite
#      integral drifts.
#
# Staggering is respected throughout: a `FaceX` field is a *point value*
# in x (interpolated with a nodal Lagrange stencil, so a child face that
# coincides with a parent face reproduces the parent value bit-for-bit)
# and a *face-area average* in y and z (interpolated conservatively, so
# the summed fine fluxes through a parent face equal the parent flux).
# That combination is what makes the prolongation flux-preserving.

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ,
               Location, interior, totalsize
using ..Fields
using ..Fields: Field, location, interior_view, set!
using ..BoundaryConditions
using ..BoundaryConditions: AbstractBC, FieldBoundaryConditions, fill_halos!,
                            Periodic, NoFlux, ImpermeableWall, NoSlipWall
using ..Simulations
using ..Simulations: BoussinesqSimulation
using ..Operators: divergence
using ..TimeSteppers: step!

import ..BoundaryConditions: _fill_x!, _fill_y!, _fill_z!

export # --- Phase 30: nesting -------------------------------------------
       NestedGrid, nested_grid, refinement_ratio, nest_offset,
       nest_covered_range, nest_child_mesh, nest_parent_mesh,
       NestBC, nest_bcs,
       interpolate_to_nest_boundary!, interpolate_parent_to_child!,
       restrict_to_parent!,
       NestFluxMismatch, compute_nest_flux_mismatch, apply_flux_correction!,
       mismatch_l1, mismatch_area,
       NestSponge, apply_nest_sponge!,
       NestInterpState, nest_interp_state, interpolate_parent_state!,
       step_nested!, nest_conservation_error, composite_integral,
       # --- Phase 31: AMR ----------------------------------------------
       RefinementBlock, AMRHierarchy, amr_hierarchy,
       refinement_criterion_gradient, refinement_criterion_vorticity,
       refinement_criterion_richardson, combine_criteria,
       cluster_flagged_cells, box_efficiency,
       regrid!, check_proper_nesting, step_amr!,
       amr_total_mass, amr_error, total_cells, level_cells, block_bounds

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

# =========================================================================
# 1-D interpolation weights
# =========================================================================

"Lagrange basis values at `x` for the `n` integer nodes `0, 1, …, n-1`."
function _lagrange_basis(n::Int, x::Float64)
    l = Vector{Float64}(undef, n)
    @inbounds for q in 0:n-1
        p = 1.0
        for s in 0:n-1
            s == q && continue
            p *= (x - s) / (q - s)
        end
        l[q+1] = p
    end
    return l
end

"""
    _cons_weights(ns, r, m, L) -> Vector{Float64}

Weights of the *conservative* prolongation: given `ns` consecutive parent
cell averages (stencil-local cells `1…ns`, cell `q` spanning `[q-1, q]`
in units of the parent spacing), return the weights that produce the
average over sub-cell `m` of `r` inside stencil-local parent cell `L`.

Built from the primitive function: `F` is known exactly at the `ns+1`
parent faces, a degree-`ns` polynomial interpolates it, and the sub-cell
average is `r·ΔP`. Because `P` matches `F` *at the nodes*, summing the
`r` sub-cell weights recovers `δ(a, L)` exactly — machine-precision
conservation.
"""
function _cons_weights(ns::Int, r::Int, m::Int, L::Int)
    a = (L - 1) + (m - 1) / r
    b = (L - 1) + m / r
    la = _lagrange_basis(ns + 1, a)
    lb = _lagrange_basis(ns + 1, b)
    w = Vector{Float64}(undef, ns)
    acc = 0.0
    @inbounds for q in ns:-1:1
        acc += lb[q+1] - la[q+1]
        w[q] = r * acc
    end
    return w
end

"Nodal (point-value) Lagrange weights at `ξ` over `ns` unit-spaced nodes."
_nodal_weights(ns::Int, ξ::Float64) = _lagrange_basis(ns, ξ)

# --- per-dimension plan ---------------------------------------------------
# For every child array index along one dimension, record the parent array
# index where the stencil starts and the stencil weights.

function _build_plan(kind::Symbol, order::Int, r::Int, off::Int,
                     Hp::Int, Np::Int, Hc::Int, Nc::Int)
    ns  = max(order, 1)
    lo  = 1 - Hp                       # lowest representable parent cell/face number
    hi  = max(lo, Np + Hp - ns + 1)    # highest legal stencil start
    plan = Vector{Tuple{Int,Vector{Float64}}}(undef, Nc + 2Hc)
    @inbounds for ic in 1:(Nc + 2Hc)
        n = ic - Hc                    # child cell / face number (may be <=0)
        if kind === :avg
            q = n - 1
            I = off + fld(q, r)
            m = mod(q, r) + 1
            s0 = clamp(I - ((ns - 1) ÷ 2), lo, hi)
            plan[ic] = (Hp + s0, _cons_weights(ns, r, m, I - s0 + 1))
        else # :nodal — child face position in parent-face-number units
            qf = off + (n - 1) / r
            s0 = clamp(Int(floor(qf)) - ((ns - 1) ÷ 2), lo, hi)
            plan[ic] = (Hp + s0, _nodal_weights(ns, qf - s0))
        end
    end
    return plan
end

function _plan(nest::NestedGrid, kind::Symbol, d::Int, order::Int)
    key = (kind, d, order)
    p = get(nest.plans, key, nothing)
    p === nothing || return p
    pm = nest.parent_mesh; cm = nest.child_mesh
    Hp, Np = d == 1 ? (pm.Hx, pm.Nx) : d == 2 ? (pm.Hy, pm.Ny) : (pm.Hz, pm.Nz)
    Hc, Nc = d == 1 ? (cm.Hx, cm.Nx) : d == 2 ? (cm.Hy, cm.Ny) : (cm.Hz, cm.Nz)
    r      = d == 1 ? nest.rx : d == 2 ? nest.ry : nest.rz
    off    = d == 1 ? nest.ix : d == 2 ? nest.iy : nest.iz
    p = _build_plan(kind, order, r, off, Hp, Np, Hc, Nc)
    nest.plans[key] = p
    return p
end

# Which 1-D rule applies in each direction, per staggered location:
# `:avg` = the field is a cell/face-area average there (conservative rule);
# `:nodal` = the field is a point value there (Lagrange rule).
_dim_kinds(::CellCenter) = (:avg,   :avg,   :avg)
_dim_kinds(::FaceX)      = (:nodal, :avg,   :avg)
_dim_kinds(::FaceY)      = (:avg,   :nodal, :avg)
_dim_kinds(::FaceZ)      = (:avg,   :avg,   :nodal)

_interface_plane(::CellCenter, cm, i, j, k) = false
_interface_plane(::FaceX, cm, i, j, k) = (i == cm.Hx + 1)
_interface_plane(::FaceY, cm, i, j, k) = (j == cm.Hy + 1)
_interface_plane(::FaceZ, cm, i, j, k) = (k == cm.Hz + 1)

# =========================================================================
# Prolongation: parent -> child
# =========================================================================

function _interp_child!(cf::Field, pf::Field, nest::NestedGrid;
                        order::Int=3, region::Symbol=:halo)
    cm = nest.child_mesh
    cf.mesh === cm || size(cf.data) == totalsize(cm) ||
        error("_interp_child!: child field is not on the nest's child mesh")
    size(pf.data) == totalsize(nest.parent_mesh) ||
        error("_interp_child!: parent field is not on the nest's parent mesh")
    loc = location(cf)
    kinds = _dim_kinds(loc)
    px = _plan(nest, kinds[1], 1, order)
    py = _plan(nest, kinds[2], 2, order)
    pz = _plan(nest, kinds[3], 3, order)
    cd = cf.data; pd = pf.data
    ii, jj, kk = interior(cm)
    T = eltype(cd)
    all_region = region === :all
    iface = region === :halo_interface
    @inbounds for k in axes(cd, 3)
        bk, wk = pz[k]
        kin = k in kk
        for j in axes(cd, 2)
            bj, wj = py[j]
            jin = kin && (j in jj)
            for i in axes(cd, 1)
                if !all_region && jin && (i in ii)
                    (iface && _interface_plane(loc, cm, i, j, k)) || continue
                end
                bi, wi = px[i]
                s = 0.0
                for c in eachindex(wk)
                    wc = wk[c]; kp = bk + c - 1
                    for b in eachindex(wj)
                        wbc = wc * wj[b]; jp = bj + b - 1
                        for a in eachindex(wi)
                            s += wbc * wi[a] * pd[bi + a - 1, jp, kp]
                        end
                    end
                end
                cd[i, j, k] = T(s)
            end
        end
    end
    return cf
end

"""
    interpolate_to_nest_boundary!(child_field, parent_field, nest; order=3,
                                  impose_interface=false)

Fill *every* halo cell of `child_field` (faces, edges and corners) by
conservative order-`order` interpolation of `parent_field`. The rule used
in each direction follows the field's [`Location`](@ref):

  * cell-centred scalars — conservative in all three directions, so the
    child sub-cell averages sum back to the parent average exactly;
  * `FaceX`/`FaceY`/`FaceZ` velocities — *nodal* (point-value) in their
    own direction and conservative in the two tangential directions. That
    is what preserves the **flux**: the summed fine-face fluxes through a
    parent face equal the parent flux, and a child face coincident with a
    parent face reproduces the parent value exactly.

`parent_field`'s halos must already be filled. With
`impose_interface=true` the child's boundary-normal interior face plane
(e.g. `i = Hx+1` for `FaceX`) is also overwritten with the parent value,
i.e. a Dirichlet inflow face.
"""
interpolate_to_nest_boundary!(cf::Field, pf::Field, nest::NestedGrid;
                              order::Int=3, impose_interface::Bool=false) =
    _interp_child!(cf, pf, nest; order,
                   region = impose_interface ? :halo_interface : :halo)

"""
    interpolate_parent_to_child!(child_field, parent_field, nest; order=3)

As [`interpolate_to_nest_boundary!`](@ref) but fills the child field
*everywhere* (interior included). This is the AMR prolongation operator —
used to initialize newly refined blocks and to build the reference state
for [`apply_nest_sponge!`](@ref).
"""
interpolate_parent_to_child!(cf::Field, pf::Field, nest::NestedGrid; order::Int=3) =
    _interp_child!(cf, pf, nest; order, region=:all)

# =========================================================================
# Restriction: child -> parent
# =========================================================================

"""
    restrict_to_parent!(parent_field, child_field, nest)

Conservative restriction.

  * `CellCenter` — volume-weighted average of the `rx·ry·rz` child cells
    inside each covered parent cell. Since `V_parent = rx·ry·rz · V_child`
    on a uniform mesh, `∫` over the overlap is preserved to machine
    precision.
  * `FaceX`/`FaceY`/`FaceZ` — area-weighted average of the `r·r` child
    faces covering each aligned parent face, i.e. flux-preserving. By
    default only faces *strictly inside* the nest are overwritten; the two
    interface planes are owned by the flux correction
    ([`apply_flux_correction!`](@ref)). Pass `include_boundary=true` to
    overwrite them as well.
"""
function restrict_to_parent!(pf::Field{CellCenter}, cf::Field{CellCenter},
                             nest::NestedGrid)
    pm = nest.parent_mesh; cm = nest.child_mesh
    rx, ry, rz = nest.rx, nest.ry, nest.rz
    inv = 1 / (rx * ry * rz)
    pd = pf.data; cd = cf.data
    @inbounds for K in 1:nest.nz, J in 1:nest.ny, I in 1:nest.nx
        s = 0.0
        for kc in 1:rz, jc in 1:ry, ic in 1:rx
            s += cd[cm.Hx + (I-1)*rx + ic, cm.Hy + (J-1)*ry + jc, cm.Hz + (K-1)*rz + kc]
        end
        pd[pm.Hx + nest.ix + I - 1, pm.Hy + nest.iy + J - 1, pm.Hz + nest.iz + K - 1] =
            s * inv
    end
    return pf
end

restrict_to_parent!(pf::Field{L}, cf::Field{L}, nest::NestedGrid;
                    include_boundary::Bool=false) where {L<:Union{FaceX,FaceY,FaceZ}} =
    _restrict_face!(pf, cf, nest, _axis_of(L), include_boundary)

_axis_of(::Type{FaceX}) = 1
_axis_of(::Type{FaceY}) = 2
_axis_of(::Type{FaceZ}) = 3

function _restrict_face!(pf::Field, cf::Field, nest::NestedGrid, ax::Int,
                         include_boundary::Bool)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    nn = (nest.nx, nest.ny, nest.nz)
    oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz); Hc = (cm.Hx, cm.Hy, cm.Hz)
    ta = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    t1, t2 = ta
    inv = 1 / (r[t1] * r[t2])
    q0 = include_boundary ? 0 : 1
    q1 = include_boundary ? nn[ax] : nn[ax] - 1
    pd = pf.data; cd = cf.data
    pidx = [0, 0, 0]; cidx = [0, 0, 0]
    @inbounds for q in q0:q1
        pidx[ax] = Hp[ax] + oo[ax] + q
        cidx[ax] = Hc[ax] + q * r[ax] + 1
        for B in 1:nn[t2], A in 1:nn[t1]
            pidx[t1] = Hp[t1] + oo[t1] + A - 1
            pidx[t2] = Hp[t2] + oo[t2] + B - 1
            s = 0.0
            for b in 1:r[t2], a in 1:r[t1]
                cidx[t1] = Hc[t1] + (A-1)*r[t1] + a
                cidx[t2] = Hc[t2] + (B-1)*r[t2] + b
                s += cd[cidx[1], cidx[2], cidx[3]]
            end
            pd[pidx[1], pidx[2], pidx[3]] = s * inv
        end
    end
    return pf
end

# =========================================================================
# Flux mismatch + correction (refluxing)
# =========================================================================

"""
    NestFluxMismatch

Per-face-patch difference between the parent's coarse flux and the summed
fine child fluxes on the six nest-boundary planes, in *flux* units
(field value × area). Populated pairs depend on the field's
[`Location`](@ref): `FaceX` fills `west`/`east` (size `ny × nz`),
`FaceY` fills `south`/`north` (`nx × nz`), `FaceZ` fills `bottom`/`top`
(`nx × ny`). Unused planes are `0 × 0`.
"""
struct NestFluxMismatch{T}
    west::Matrix{T};   east::Matrix{T}
    south::Matrix{T};  north::Matrix{T}
    bottom::Matrix{T}; top::Matrix{T}
end

function NestFluxMismatch{T}() where {T}
    z = zeros(T, 0, 0)
    NestFluxMismatch{T}(copy(z), copy(z), copy(z), copy(z), copy(z), copy(z))
end

Base.:+(a::NestFluxMismatch{T}, b::NestFluxMismatch{T}) where {T} =
    NestFluxMismatch{T}(_addm(a.west, b.west), _addm(a.east, b.east),
                        _addm(a.south, b.south), _addm(a.north, b.north),
                        _addm(a.bottom, b.bottom), _addm(a.top, b.top))
_addm(a, b) = isempty(a) ? copy(b) : isempty(b) ? copy(a) : a .+ b

"Sum of `|mismatch|` over all six planes (flux units)."
mismatch_l1(m::NestFluxMismatch) =
    sum(x -> isempty(x) ? 0.0 : sum(abs, x),
        (m.west, m.east, m.south, m.north, m.bottom, m.top))

"Total area of the six nest-interface planes."
function mismatch_area(nest::NestedGrid)
    pm = nest.parent_mesh
    return 2 * (nest.ny * nest.nz * pm.dy * pm.dz +
                nest.nx * nest.nz * pm.dx * pm.dz +
                nest.nx * nest.ny * pm.dx * pm.dy)
end

"""
    compute_nest_flux_mismatch(parent_field, child_field, nest)

`coarse flux − Σ fine fluxes` across each nest-boundary face patch, for a
face-located field (a velocity, or a time-accumulated flux register
`Φ = ∫F dt`). Returns a [`NestFluxMismatch`](@ref).

A nonzero value is exactly the amount by which the two grids disagree
about the transport through the interface — i.e. the leak that
[`apply_flux_correction!`](@ref) removes.
"""
compute_nest_flux_mismatch(pf::Field{L}, cf::Field{L},
                           nest::NestedGrid) where {L<:Union{FaceX,FaceY,FaceZ}} =
    _face_mismatch(pf, cf, nest, _axis_of(L))

function _face_mismatch(pf::Field, cf::Field, nest::NestedGrid, ax::Int)
    T  = eltype(pf.data)
    pm = nest.parent_mesh; cm = nest.child_mesh
    r  = (nest.rx, nest.ry, nest.rz)
    nn = (nest.nx, nest.ny, nest.nz)
    oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz); Hc = (cm.Hx, cm.Hy, cm.Hz)
    dp = (pm.dx, pm.dy, pm.dz)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    Ap  = dp[t1] * dp[t2]
    inv = 1 / (r[t1] * r[t2])
    pd = pf.data; cd = cf.data
    lowm  = zeros(T, nn[t1], nn[t2])
    highm = zeros(T, nn[t1], nn[t2])
    pidx = [0, 0, 0]; cidx = [0, 0, 0]
    for (q, M) in ((0, lowm), (nn[ax], highm))
        pidx[ax] = Hp[ax] + oo[ax] + q
        cidx[ax] = Hc[ax] + q * r[ax] + 1
        @inbounds for B in 1:nn[t2], A in 1:nn[t1]
            pidx[t1] = Hp[t1] + oo[t1] + A - 1
            pidx[t2] = Hp[t2] + oo[t2] + B - 1
            s = 0.0
            for b in 1:r[t2], a in 1:r[t1]
                cidx[t1] = Hc[t1] + (A-1)*r[t1] + a
                cidx[t2] = Hc[t2] + (B-1)*r[t2] + b
                s += cd[cidx[1], cidx[2], cidx[3]]
            end
            M[A, B] = (pd[pidx[1], pidx[2], pidx[3]] - s * inv) * Ap
        end
    end
    e = zeros(T, 0, 0)
    ax == 1 && return NestFluxMismatch{T}(lowm, highm, copy(e), copy(e), copy(e), copy(e))
    ax == 2 && return NestFluxMismatch{T}(copy(e), copy(e), lowm, highm, copy(e), copy(e))
    return NestFluxMismatch{T}(copy(e), copy(e), copy(e), copy(e), lowm, highm)
end

"""
    apply_flux_correction!(parent_field, mismatch, nest)

Remove the coarse-fine flux inconsistency recorded in `mismatch`.

  * **face-located `parent_field`** — replace the parent's interface face
    value with the summed fine flux (`u_p -= δΦ / A_p`). The parent's
    transport through the interface then matches the child's exactly.
  * **cell-centred `parent_field`** (the Berger-Colella *refluxing* step,
    with `mismatch` built from time-accumulated flux registers) — adjust
    the parent cells immediately *outside* the nest:
    `θ ± δΦ·A / V_p`. This is what makes the composite integral of a
    conserved scalar exactly invariant across a nested step. Indices wrap
    periodically, so a nest touching the parent boundary still works.

Without this correction two-way nesting leaks: the parent kept the coarse
flux while the child used the fine ones.
"""
function apply_flux_correction!(pf::Field{L}, m::NestFluxMismatch,
                                nest::NestedGrid) where {L<:Union{FaceX,FaceY,FaceZ}}
    ax = _axis_of(L)
    pm = nest.parent_mesh
    dp = (pm.dx, pm.dy, pm.dz)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    Ap = dp[t1] * dp[t2]
    lowm, highm = ax == 1 ? (m.west, m.east) : ax == 2 ? (m.south, m.north) :
                                               (m.bottom, m.top)
    (isempty(lowm) || isempty(highm)) && return pf
    nn = (nest.nx, nest.ny, nest.nz); oo = (nest.ix, nest.iy, nest.iz)
    Hp = (pm.Hx, pm.Hy, pm.Hz)
    pd = pf.data; pidx = [0, 0, 0]
    for (q, M) in ((0, lowm), (nn[ax], highm))
        pidx[ax] = Hp[ax] + oo[ax] + q
        @inbounds for B in 1:nn[t2], A in 1:nn[t1]
            pidx[t1] = Hp[t1] + oo[t1] + A - 1
            pidx[t2] = Hp[t2] + oo[t2] + B - 1
            pd[pidx[1], pidx[2], pidx[3]] -= M[A, B] / Ap
        end
    end
    return pf
end

function apply_flux_correction!(pf::Field{CellCenter}, m::NestFluxMismatch,
                                nest::NestedGrid)
    pm = nest.parent_mesh
    Vp = pm.dx * pm.dy * pm.dz
    Np = (pm.Nx, pm.Ny, pm.Nz); Hp = (pm.Hx, pm.Hy, pm.Hz)
    nn = (nest.nx, nest.ny, nest.nz); oo = (nest.ix, nest.iy, nest.iz)
    pd = pf.data
    planes = ((1, m.west, m.east), (2, m.south, m.north), (3, m.bottom, m.top))
    for (ax, lowm, highm) in planes
        (isempty(lowm) || isempty(highm)) && continue
        t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
        # cell just outside the low face:  Δθ = +δΦ·A / V
        # cell just outside the high face: Δθ = -δΦ·A / V
        for (cell, M, sgn) in ((oo[ax] - 1, lowm, +1.0),
                               (oo[ax] + nn[ax], highm, -1.0))
            c = mod1(cell, Np[ax])
            idx = [0, 0, 0]; idx[ax] = Hp[ax] + c
            @inbounds for B in 1:nn[t2], A in 1:nn[t1]
                idx[t1] = Hp[t1] + oo[t1] + A - 1
                idx[t2] = Hp[t2] + oo[t2] + B - 1
                pd[idx[1], idx[2], idx[3]] += sgn * M[A, B] / Vp
            end
        end
    end
    return pf
end

# =========================================================================
# Nest sponge (reflection control)
# =========================================================================

"""
    NestSponge(; width=4, α_max=1.0, profile=:cos2, sides=ntuple(_->true, 6))

Relaxation zone `width` child cells deep inside the child, adjacent to
its nest boundaries. [`apply_nest_sponge!`](@ref) blends the child
solution toward the interpolated parent solution with rate `α_max`
(1/time), ramped by `profile` (`:cos2` — smooth `sin²` taper — or
`:linear`) from 0 at the inner edge of the zone to `α_max` at the
boundary. `sides` is `(west, east, south, north, bottom, top)`.

The smooth taper is the point: a hard relaxation edge is itself a
reflector. Without a sponge, the halo/interior mismatch at a nest
boundary reflects outgoing waves back into the child.
"""
struct NestSponge{T}
    width::Int
    α_max::T
    profile::Symbol
    sides::NTuple{6,Bool}
end

function NestSponge(; width::Int=4, α_max::Real=1.0, profile::Symbol=:cos2,
                    sides::NTuple{6,Bool}=(true, true, true, true, true, true))
    width >= 1 || error("NestSponge: width must be >= 1")
    profile in (:cos2, :linear) || error("NestSponge: profile must be :cos2 or :linear")
    return NestSponge{Float64}(width, Float64(α_max), profile, sides)
end

_sponge_prof(sp::NestSponge, s::Float64) =
    sp.profile === :cos2 ? sin(0.5π * s)^2 : s

function _sponge_ramp(sp::NestSponge, d::Int)
    d < 1 && return 1.0                 # halo side of the boundary
    d > sp.width && return 0.0
    return _sponge_prof(sp, (sp.width + 1 - d) / sp.width)
end

function _sponge_weight(sp::NestSponge, cm::RegularGrid, i::Int, j::Int, k::Int)
    n1 = i - cm.Hx; n2 = j - cm.Hy; n3 = k - cm.Hz
    w = 0.0
    sp.sides[1] && (w = max(w, _sponge_ramp(sp, n1)))
    sp.sides[2] && (w = max(w, _sponge_ramp(sp, cm.Nx + 1 - n1)))
    sp.sides[3] && (w = max(w, _sponge_ramp(sp, n2)))
    sp.sides[4] && (w = max(w, _sponge_ramp(sp, cm.Ny + 1 - n2)))
    sp.sides[5] && (w = max(w, _sponge_ramp(sp, n3)))
    sp.sides[6] && (w = max(w, _sponge_ramp(sp, cm.Nz + 1 - n3)))
    return w
end

# =========================================================================
# Parent state interpolated onto the child mesh
# =========================================================================

"""
    NestInterpState

The parent solution (`u, v, w, θ`) interpolated onto the child mesh —
boundary data for the child and the relaxation target for
[`apply_nest_sponge!`](@ref). Build with [`nest_interp_state`](@ref),
fill with [`interpolate_parent_state!`](@ref).
"""
struct NestInterpState{T, M<:RegularGrid{T}}
    u::Field{FaceX,      T, Array{T,3}, M}
    v::Field{FaceY,      T, Array{T,3}, M}
    w::Field{FaceZ,      T, Array{T,3}, M}
    θ::Field{CellCenter, T, Array{T,3}, M}
end

function nest_interp_state(nest::NestedGrid{T}) where {T}
    cm = nest.child_mesh
    return NestInterpState(Field(FaceX(), cm), Field(FaceY(), cm),
                           Field(FaceZ(), cm), Field(CellCenter(), cm))
end

"""
    interpolate_parent_state!(state, parent_sim, nest; order=3, region=:all)

Interpolate `parent_sim`'s `u, v, w, θ` onto the child mesh into `state`.
`region=:all` fills the whole child field (needed for the sponge);
`region=:halo` fills only halos. Parent halos are filled first.
"""
function interpolate_parent_state!(s::NestInterpState, psim::BoussinesqSimulation,
                                   nest::NestedGrid; order::Int=3,
                                   region::Symbol=:all)
    fill_halos!(psim.u, psim.bcs); fill_halos!(psim.v, psim.bcs)
    fill_halos!(psim.w, psim.bcs); fill_halos!(psim.θ, psim.θ_bcs)
    _interp_child!(s.u, psim.u, nest; order, region)
    _interp_child!(s.v, psim.v, nest; order, region)
    _interp_child!(s.w, psim.w, nest; order, region)
    _interp_child!(s.θ, psim.θ, nest; order, region)
    return s
end

function _blend_state!(dst::NestInterpState, a::NestInterpState,
                       b::NestInterpState, τ::Real)
    for f in (:u, :v, :w, :θ)
        getfield(dst, f).data .= (1 - τ) .* getfield(a, f).data .+ τ .* getfield(b, f).data
    end
    return dst
end

function _copy_halo!(dst::Field, src::Field; interface::Bool=false)
    m = dst.mesh
    ii, jj, kk = interior(m)
    loc = location(dst)
    dd = dst.data; sd = src.data
    @inbounds for k in axes(dd, 3), j in axes(dd, 2), i in axes(dd, 1)
        if (i in ii) && (j in jj) && (k in kk)
            (interface && _interface_plane(loc, m, i, j, k)) || continue
        end
        dd[i, j, k] = sd[i, j, k]
    end
    return dst
end

"""
    apply_nest_boundary!(child_sim, state; impose_interface=false)

Write the interpolated parent `state` into `child_sim`'s halo cells (and,
optionally, the boundary-normal interior face planes).
"""
function apply_nest_boundary!(csim::BoussinesqSimulation, s::NestInterpState;
                              impose_interface::Bool=false)
    _copy_halo!(csim.u, s.u; interface=impose_interface)
    _copy_halo!(csim.v, s.v; interface=impose_interface)
    _copy_halo!(csim.w, s.w; interface=impose_interface)
    _copy_halo!(csim.θ, s.θ)
    return csim
end

"""
    apply_nest_sponge!(child_sim, parent_interp, sponge; dt=1.0)

Relax the child solution toward `parent_interp` inside the sponge zone:
`f ← (1-β) f + β f_parent` with `β = clamp(α_max·dt·ramp, 0, 1)`.
Applied to `u, v, w, θ`. Call once per child substep, after `step!`.
"""
function apply_nest_sponge!(csim::BoussinesqSimulation, s::NestInterpState,
                            sp::NestSponge; dt::Real=1.0)
    cm = csim.mesh
    ii, jj, kk = interior(cm)
    αdt = sp.α_max * dt
    for (cf, gf) in ((csim.u, s.u), (csim.v, s.v), (csim.w, s.w), (csim.θ, s.θ))
        cd = cf.data; gd = gf.data
        @inbounds for k in kk, j in jj, i in ii
            wgt = _sponge_weight(sp, cm, i, j, k)
            wgt == 0.0 && continue
            β = clamp(αdt * wgt, 0.0, 1.0)
            cd[i, j, k] = (1 - β) * cd[i, j, k] + β * gd[i, j, k]
        end
    end
    return csim
end

# =========================================================================
# Composite diagnostics
# =========================================================================

"""
    composite_integral(parent_sim, child_sim, nest, quantity) -> Float64

Volume integral of `quantity` over the *composite* grid: parent cells not
covered by the nest plus all child cells. `quantity ∈ (:mass, :theta,
:momentum_x)`. For the Boussinesq system `:mass` is `ρ₀ ∫dV`, a constant —
mass conservation across a nest interface is therefore diagnosed by
[`nest_conservation_error`](@ref) from the interface flux imbalance
instead.
"""
function composite_integral(psim::BoussinesqSimulation, csim::BoussinesqSimulation,
                            nest::NestedGrid, quantity::Symbol)
    pm = nest.parent_mesh; cm = nest.child_mesh
    Vp = pm.dx * pm.dy * pm.dz; Vc = cm.dx * cm.dy * cm.dz
    quantity === :mass && return (pm.Nx*pm.Ny*pm.Nz - nest.nx*nest.ny*nest.nz) * Vp +
                                 cm.Nx*cm.Ny*cm.Nz * Vc
    cx, cy, cz = nest_covered_range(nest)
    total = 0.0
    if quantity === :theta
        pd = psim.θ.data
        @inbounds for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
            (i in cx) && (j in cy) && (k in cz) && continue
            total += pd[pm.Hx+i, pm.Hy+j, pm.Hz+k] * Vp
        end
        cd = csim.θ.data
        @inbounds for k in 1:cm.Nz, j in 1:cm.Ny, i in 1:cm.Nx
            total += cd[cm.Hx+i, cm.Hy+j, cm.Hz+k] * Vc
        end
    elseif quantity === :momentum_x
        pd = psim.u.data
        @inbounds for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
            (i in cx) && (j in cy) && (k in cz) && continue
            total += 0.5 * (pd[pm.Hx+i, pm.Hy+j, pm.Hz+k] +
                            pd[pm.Hx+i+1, pm.Hy+j, pm.Hz+k]) * Vp
        end
        cd = csim.u.data
        @inbounds for k in 1:cm.Nz, j in 1:cm.Ny, i in 1:cm.Nx
            total += 0.5 * (cd[cm.Hx+i, cm.Hy+j, cm.Hz+k] +
                            cd[cm.Hx+i+1, cm.Hy+j, cm.Hz+k]) * Vc
        end
    else
        error("composite_integral: unknown quantity $quantity " *
              "(:mass, :theta, :momentum_x)")
    end
    return total
end

"Interface mass-flux imbalance, normalized by (velocity scale × interface area)."
function _interface_imbalance(psim::BoussinesqSimulation, csim::BoussinesqSimulation,
                              nest::NestedGrid)
    m = compute_nest_flux_mismatch(psim.u, csim.u, nest) +
        compute_nest_flux_mismatch(psim.v, csim.v, nest) +
        compute_nest_flux_mismatch(psim.w, csim.w, nest)
    U = max(maximum(abs, interior_view(psim.u)),
            maximum(abs, interior_view(psim.v)),
            maximum(abs, interior_view(psim.w)),
            maximum(abs, interior_view(csim.u)))
    U = U > 0 ? U : 1.0
    return mismatch_l1(m) / (U * mismatch_area(nest))
end

"""
    nest_conservation_error(parent_sim, child_sim, nest, quantity) -> Float64

Relative error in the globally integrated `quantity` across a nested
step, as recorded by the most recent [`step_nested!`](@ref).

  * `:mass` — the L1 coarse-fine **mass-flux imbalance** across the six
    interface planes, normalized by (velocity scale × interface area).
    This is the discrete statement that the parent and the child agree on
    the volume flux through the interface; `apply_flux_correction!` drives
    it to machine zero, and skipping the correction leaves it at the size
    of the genuine coarse/fine disagreement.
  * `:theta`, `:momentum_x` — relative change of
    [`composite_integral`](@ref) over the step.

If no step has been recorded, `:mass` is evaluated instantaneously and the
other quantities raise an error.
"""
function nest_conservation_error(psim::BoussinesqSimulation, csim::BoussinesqSimulation,
                                 nest::NestedGrid, quantity::Symbol)
    haskey(nest.ledger, quantity) && return nest.ledger[quantity]
    quantity === :mass && return _interface_imbalance(psim, csim, nest)
    error("nest_conservation_error: no recorded value for :$quantity — " *
          "run step_nested! first.")
end

# =========================================================================
# Time-stepping coordination
# =========================================================================

"""
    step_nested!(parent_sim, child_sim, nest, dt; substeps=nothing, order=3,
                 flux_correction=true, sponge=nothing, impose_interface=false,
                 state=nothing)

One parent step of size `dt` plus `substeps` child steps of size
`dt/substeps` (default `max(rx,ry,rz)`, which keeps the child CFL equal to
the parent's).

Sequence:

 1. interpolate the parent state at time `t` onto the child mesh;
 2. `step!(parent_sim, dt)`; interpolate the new parent state;
 3. for each child substep, **linearly interpolate the parent state in
    time** to the substep midpoint, write it into the child halos,
    `step!` the child, then apply the sponge (if any);
 4. conservatively restrict the child interior back onto the parent;
 5. compute the interface flux mismatch and apply the flux correction.

Conservation diagnostics for `:mass`, `:theta` and `:momentum_x` are
recorded in `nest.ledger`; read them with
[`nest_conservation_error`](@ref). Returns `(parent_sim, child_sim)`.

`state` may be a pre-allocated `NTuple{3,NestInterpState}` to avoid
re-allocating the boundary buffers every step.
"""
function step_nested!(psim::BoussinesqSimulation, csim::BoussinesqSimulation,
                      nest::NestedGrid, dt::Real;
                      substeps::Union{Nothing,Integer}=nothing,
                      order::Int=3,
                      flux_correction::Bool=true,
                      sponge::Union{Nothing,NestSponge}=nothing,
                      impose_interface::Bool=false,
                      state=nothing,
                      record::Bool=true)
    ns = substeps === nothing ? max(nest.rx, nest.ry, nest.rz) : Int(substeps)
    ns >= 1 || error("step_nested!: substeps must be >= 1")

    θ0 = composite_integral(psim, csim, nest, :theta)
    m0 = composite_integral(psim, csim, nest, :momentum_x)

    s_old, s_new, s_bc = state === nothing ?
        (nest_interp_state(nest), nest_interp_state(nest), nest_interp_state(nest)) :
        state

    interpolate_parent_state!(s_old, psim, nest; order, region=:all)
    step!(psim, dt)
    interpolate_parent_state!(s_new, psim, nest; order, region=:all)

    dtc = dt / ns
    for m in 1:ns
        _blend_state!(s_bc, s_old, s_new, (m - 0.5) / ns)
        apply_nest_boundary!(csim, s_bc; impose_interface)
        step!(csim, dtc)
        sponge === nothing || apply_nest_sponge!(csim, s_bc, sponge; dt=dtc)
    end

    # --- conservative restriction (interior faces / covered cells) --------
    restrict_to_parent!(psim.θ, csim.θ, nest)
    restrict_to_parent!(psim.u, csim.u, nest)
    restrict_to_parent!(psim.v, csim.v, nest)
    restrict_to_parent!(psim.w, csim.w, nest)

    e_before = _interface_imbalance(psim, csim, nest)
    if flux_correction
        apply_flux_correction!(psim.u, compute_nest_flux_mismatch(psim.u, csim.u, nest), nest)
        apply_flux_correction!(psim.v, compute_nest_flux_mismatch(psim.v, csim.v, nest), nest)
        apply_flux_correction!(psim.w, compute_nest_flux_mismatch(psim.w, csim.w, nest), nest)
    end
    e_after = _interface_imbalance(psim, csim, nest)

    if record
        nest.ledger[:mass] = e_after
        nest.ledger[:mass_before_correction] = e_before
        θ1 = composite_integral(psim, csim, nest, :theta)
        m1 = composite_integral(psim, csim, nest, :momentum_x)
        nest.ledger[:theta] = abs(θ1 - θ0) / max(abs(θ0), 1e-300)
        nest.ledger[:momentum_x] = abs(m1 - m0) / max(abs(m0), 1e-300)
    end
    return (psim, csim)
end


# =========================================================================
# Phase 31 — block-structured AMR
# =========================================================================

"""
    RefinementBlock

A rectangular patch of refined mesh.

  * `level` — 1 for the base (root) block, 2 for the first refined level, …
  * `bounds` — index range `(ix, iy, iz)` in **this level's** global index
    space (`base.N .* r.^(level-1)` cells). Always aligned to the parent
    level's cell boundaries.
  * `mesh` — the block's own [`RegularGrid`](@ref) (with halos).
  * `state` — `Dict{Symbol,Field}`: the advected scalars, the velocity
    `:u, :v, :w`, and the time-accumulated flux registers
    `:Fx_θ, :Fy_θ, :Fz_θ` used for refluxing.
  * `parent` / `children` — the level links.
  * `nest` — the [`NestedGrid`](@ref) relating this block to its parent
    (prolongation / restriction / refluxing operators).
"""
mutable struct RefinementBlock{T}
    level::Int
    bounds::NTuple{3,UnitRange{Int}}
    mesh::RegularGrid{T}
    state::Dict{Symbol,Field}
    parent::Union{Nothing,RefinementBlock{T}}
    children::Vector{RefinementBlock{T}}
    nest::Union{Nothing,NestedGrid}
end

block_bounds(b::RefinementBlock) = b.bounds
Base.show(io::IO, b::RefinementBlock) =
    print(io, "RefinementBlock(level=", b.level, ", bounds=", b.bounds, ", cells=",
          b.mesh.Nx * b.mesh.Ny * b.mesh.Nz, ")")

"""
    AMRHierarchy

The level structure: `levels[ℓ]` is the vector of [`RefinementBlock`](@ref)s
at level `ℓ` (`levels[1]` always holds the single root block covering the
base grid). Blocks at one level never overlap; levels deeper than 1 are
built by [`regrid!`](@ref), which enforces proper nesting with a buffer.

Fields of interest: `base`, `refinement` (`(rx,ry,rz)`), `max_level`,
`vars` (advected scalars), `velocity` (a prescribed
`(x,y,z,t) -> (u,v,w)`), `advance!` (per-block update), `order`
(prolongation order), `buffer`, `t`, `nstep`.
"""
mutable struct AMRHierarchy{T}
    base::RegularGrid{T}
    refinement::NTuple{3,Int}
    max_level::Int
    levels::Vector{Vector{RefinementBlock{T}}}
    vars::Vector{Symbol}
    velocity::Any
    advance!::Any
    order::Int
    bcs::FieldBoundaryConditions
    buffer::Int
    t::T
    nstep::Int
end

_reg(v::Symbol, ax::Symbol) = Symbol("F", ax, "_", v)

_level_spacing(h::AMRHierarchy, ℓ::Int) =
    (h.base.dx / h.refinement[1]^(ℓ-1),
     h.base.dy / h.refinement[2]^(ℓ-1),
     h.base.dz / h.refinement[3]^(ℓ-1))

_level_size(h::AMRHierarchy, ℓ::Int) =
    (h.base.Nx * h.refinement[1]^(ℓ-1),
     h.base.Ny * h.refinement[2]^(ℓ-1),
     h.base.Nz * h.refinement[3]^(ℓ-1))

# physical coordinates of an array index (halos included)
_xf(m::RegularGrid, i) = m.x0 + (i - m.Hx - 1) * m.dx
_yf(m::RegularGrid, j) = m.y0 + (j - m.Hy - 1) * m.dy
_zf(m::RegularGrid, k) = m.z0 + (k - m.Hz - 1) * m.dz
_xc(m::RegularGrid, i) = m.x0 + (i - m.Hx - 0.5) * m.dx
_yc(m::RegularGrid, j) = m.y0 + (j - m.Hy - 0.5) * m.dy
_zc(m::RegularGrid, k) = m.z0 + (k - m.Hz - 0.5) * m.dz

function _alloc_state!(h::AMRHierarchy, b::RefinementBlock)
    m = b.mesh
    b.state[:u] = Field(FaceX(), m)
    b.state[:v] = Field(FaceY(), m)
    b.state[:w] = Field(FaceZ(), m)
    b.state[:fx] = Field(FaceX(), m)   # instantaneous flux scratch
    b.state[:fy] = Field(FaceY(), m)
    b.state[:fz] = Field(FaceZ(), m)
    for v in h.vars
        b.state[v] = Field(CellCenter(), m)
        b.state[_reg(v, :x)] = Field(FaceX(), m)
        b.state[_reg(v, :y)] = Field(FaceY(), m)
        b.state[_reg(v, :z)] = Field(FaceZ(), m)
    end
    return b
end

_coarse_bounds(h::AMRHierarchy, b::RefinementBlock) =
    ntuple(d -> ((first(b.bounds[d]) - 1) ÷ h.refinement[d] + 1) :
                (last(b.bounds[d]) ÷ h.refinement[d]), 3)

function _make_block(h::AMRHierarchy{T}, ℓ::Int, bounds::NTuple{3,UnitRange{Int}},
                     parent::Union{Nothing,RefinementBlock{T}}) where {T}
    d = _level_spacing(h, ℓ)
    n = (length(bounds[1]), length(bounds[2]), length(bounds[3]))
    mesh = RegularGrid(T; size=n,
                       extent=(n[1]*d[1], n[2]*d[2], n[3]*d[3]),
                       halo=(h.base.Hx, h.base.Hy, h.base.Hz),
                       origin=(h.base.x0 + (first(bounds[1]) - 1) * d[1],
                               h.base.y0 + (first(bounds[2]) - 1) * d[2],
                               h.base.z0 + (first(bounds[3]) - 1) * d[3]))
    b = RefinementBlock{T}(ℓ, bounds, mesh, Dict{Symbol,Field}(), parent,
                           RefinementBlock{T}[], nothing)
    _alloc_state!(h, b)
    if parent !== nothing
        cb = _coarse_bounds(h, b)
        off = ntuple(dd -> first(cb[dd]) - first(parent.bounds[dd]) + 1, 3)
        b.nest = NestedGrid(parent.mesh, mesh; refinement=h.refinement, offset=off)
    end
    _set_block_velocity!(h, b, h.t)
    return b
end

function _set_block_velocity!(h::AMRHierarchy, b::RefinementBlock, t)
    h.velocity === nothing && return b
    m = b.mesh
    ud = b.state[:u].data; vd = b.state[:v].data; wd = b.state[:w].data
    @inbounds for k in axes(ud, 3), j in axes(ud, 2), i in axes(ud, 1)
        ud[i,j,k] = h.velocity(_xf(m,i), _yc(m,j), _zc(m,k), t)[1]
        vd[i,j,k] = h.velocity(_xc(m,i), _yf(m,j), _zc(m,k), t)[2]
        wd[i,j,k] = h.velocity(_xc(m,i), _yc(m,j), _zf(m,k), t)[3]
    end
    return b
end

function _zero_registers!(h::AMRHierarchy, b::RefinementBlock)
    for v in h.vars, ax in (:x, :y, :z)
        fill!(b.state[_reg(v, ax)].data, 0)
    end
    return b
end

"""
    _amr_advect!(hierarchy, block, dt, t)

Default per-block update: flux-form first-order-upwind advection of every
scalar in `hierarchy.vars` by the block's velocity. The face fluxes are
accumulated (×`dt`) into the block's flux registers, which is what makes
Berger-Colella refluxing — and therefore exact composite conservation —
possible. Replace via the `advance!` keyword of [`amr_hierarchy`](@ref).
"""
function _amr_advect!(h::AMRHierarchy, b::RefinementBlock, dt::Real, t)
    m = b.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    idx, idy, idz = 1/m.dx, 1/m.dy, 1/m.dz
    ud = b.state[:u].data; vd = b.state[:v].data; wd = b.state[:w].data
    fxd = b.state[:fx].data; fyd = b.state[:fy].data; fzd = b.state[:fz].data
    for v in h.vars
        θd = b.state[v].data
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx+1
            uu = ud[i,j,k]
            fxd[i,j,k] = uu * (uu >= 0 ? θd[i-1,j,k] : θd[i,j,k])
        end
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny+1, i in Hx+1:Hx+Nx
            vv = vd[i,j,k]
            fyd[i,j,k] = vv * (vv >= 0 ? θd[i,j-1,k] : θd[i,j,k])
        end
        @inbounds for k in Hz+1:Hz+Nz+1, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            ww = wd[i,j,k]
            fzd[i,j,k] = ww * (ww >= 0 ? θd[i,j,k-1] : θd[i,j,k])
        end
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            θd[i,j,k] -= dt * ((fxd[i+1,j,k] - fxd[i,j,k]) * idx +
                               (fyd[i,j+1,k] - fyd[i,j,k]) * idy +
                               (fzd[i,j,k+1] - fzd[i,j,k]) * idz)
        end
        Fx = b.state[_reg(v,:x)].data
        Fy = b.state[_reg(v,:y)].data
        Fz = b.state[_reg(v,:z)].data
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx+1
            Fx[i,j,k] += dt * fxd[i,j,k]
        end
        @inbounds for k in Hz+1:Hz+Nz, j in Hy+1:Hy+Ny+1, i in Hx+1:Hx+Nx
            Fy[i,j,k] += dt * fyd[i,j,k]
        end
        @inbounds for k in Hz+1:Hz+Nz+1, j in Hy+1:Hy+Ny, i in Hx+1:Hx+Nx
            Fz[i,j,k] += dt * fzd[i,j,k]
        end
    end
    return b
end

"""
    amr_hierarchy(base_mesh; refinement=(2,2,2), max_level=2, vars=[:θ],
                  velocity=nothing, advance! = _amr_advect!, order=3,
                  bcs=FieldBoundaryConditions(:periodic), buffer=2)

Build a one-level hierarchy whose root block covers `base_mesh`. Call
[`regrid!`](@ref) (or [`step_amr!`](@ref) with a criterion) to create the
refined levels.
"""
function amr_hierarchy(base::RegularGrid{T};
                       refinement=(2,2,2), max_level::Int=2,
                       vars::Vector{Symbol}=[:θ],
                       velocity=nothing,
                       advance! = _amr_advect!,
                       order::Int=3,
                       bcs::FieldBoundaryConditions=FieldBoundaryConditions(:periodic),
                       buffer::Int=2) where {T}
    r = _triple(refinement)
    max_level >= 1 || error("amr_hierarchy: max_level must be >= 1")
    h = AMRHierarchy{T}(base, r, max_level, Vector{Vector{RefinementBlock{T}}}(),
                        copy(vars), velocity, advance!, order, bcs, buffer,
                        zero(T), 0)
    root = _make_block(h, 1, (1:base.Nx, 1:base.Ny, 1:base.Nz), nothing)
    push!(h.levels, RefinementBlock{T}[root])
    return h
end

total_cells(h::AMRHierarchy) =
    sum(b -> b.mesh.Nx * b.mesh.Ny * b.mesh.Nz, Iterators.flatten(h.levels); init=0)

level_cells(h::AMRHierarchy, ℓ::Int) =
    ℓ > length(h.levels) ? 0 :
    sum(b -> b.mesh.Nx * b.mesh.Ny * b.mesh.Nz, h.levels[ℓ]; init=0)

# =========================================================================
# Refinement criteria
# =========================================================================

_field_of(x::BoussinesqSimulation, name::Symbol) = getproperty(x, name)
_field_of(b::RefinementBlock, name::Symbol) = b.state[name]
_field_of(f::Field, ::Symbol) = f

function _lohner(fm::Float64, f0::Float64, fp::Float64, ε::Float64)
    num = abs(fp - 2 * f0 + fm)
    den = abs(fp - f0) + abs(f0 - fm) + ε * (abs(fp) + 2abs(f0) + abs(fm))
    den <= 0 && return 0.0
    return num / den
end

"""
    refinement_criterion_gradient(x; field=:θ, threshold=0.1, ε=0.01) -> BitArray{3}

Löhner-style normalized second-derivative indicator on `field`
(`x` may be a `BoussinesqSimulation`, a [`RefinementBlock`](@ref) or a
`Field`). For each direction

    e = |fᵢ₊₁ - 2fᵢ + fᵢ₋₁| / (|fᵢ₊₁-fᵢ| + |fᵢ-fᵢ₋₁| + ε(|fᵢ₊₁| + 2|fᵢ| + |fᵢ₋₁|))

and the mask flags cells where `max(eₓ, e_y, e_z) > threshold`. The `ε`
term is what keeps smooth *and* flat regions unflagged — without it the
indicator saturates on round-off. The indicator is dimensionless, so
`threshold` is scale-free. Domain-boundary cells contribute `0` in the
affected direction (no reliance on halo contents).
"""
function refinement_criterion_gradient(x; field::Symbol=:θ, threshold::Real=0.1,
                                       ε::Real=0.01)
    f = _field_of(x, field); m = f.mesh
    d = f.data; εf = Float64(ε)
    mask = falses(m.Nx, m.Ny, m.Nz)
    @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
        ia = m.Hx + i; ja = m.Hy + j; ka = m.Hz + k
        f0 = Float64(d[ia, ja, ka])
        e = 0.0
        if 1 < i < m.Nx
            e = max(e, _lohner(Float64(d[ia-1,ja,ka]), f0, Float64(d[ia+1,ja,ka]), εf))
        end
        if 1 < j < m.Ny
            e = max(e, _lohner(Float64(d[ia,ja-1,ka]), f0, Float64(d[ia,ja+1,ka]), εf))
        end
        if 1 < k < m.Nz
            e = max(e, _lohner(Float64(d[ia,ja,ka-1]), f0, Float64(d[ia,ja,ka+1]), εf))
        end
        mask[i,j,k] = e > threshold
    end
    return mask
end

"""
    refinement_criterion_vorticity(x; threshold=0.2, relative=true) -> BitArray{3}

Flag cells by vorticity magnitude `|ω|` — the natural LES criterion, since
`|ω|` marks the shear layers and coherent structures that carry the SGS
burden. `u, v, w` are averaged to cell centres and differentiated with
centred (one-sided at the domain edge) differences. With `relative=true`
(default) `threshold` is a fraction of `max|ω|`; otherwise it is absolute.
"""
function refinement_criterion_vorticity(x; threshold::Real=0.2, relative::Bool=true)
    u = _field_of(x, :u); v = _field_of(x, :v); w = _field_of(x, :w)
    m = u.mesh
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    uc = zeros(Float64, Nx, Ny, Nz); vc = similar(uc); wc = similar(uc)
    ud = u.data; vd = v.data; wd = w.data
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ia = m.Hx + i; ja = m.Hy + j; ka = m.Hz + k
        uc[i,j,k] = 0.5 * (ud[ia,ja,ka] + ud[ia+1,ja,ka])
        vc[i,j,k] = 0.5 * (vd[ia,ja,ka] + vd[ia,ja+1,ka])
        wc[i,j,k] = 0.5 * (wd[ia,ja,ka] + wd[ia,ja,ka+1])
    end
    ω = zeros(Float64, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = min(i+1, Nx); im = max(i-1, 1); hx = (ip - im) * m.dx
        jp = min(j+1, Ny); jm = max(j-1, 1); hy = (jp - jm) * m.dy
        kp = min(k+1, Nz); km = max(k-1, 1); hz = (kp - km) * m.dz
        dwdy = hy > 0 ? (wc[i,jp,k] - wc[i,jm,k]) / hy : 0.0
        dvdz = hz > 0 ? (vc[i,j,kp] - vc[i,j,km]) / hz : 0.0
        dudz = hz > 0 ? (uc[i,j,kp] - uc[i,j,km]) / hz : 0.0
        dwdx = hx > 0 ? (wc[ip,j,k] - wc[im,j,k]) / hx : 0.0
        dvdx = hx > 0 ? (vc[ip,j,k] - vc[im,j,k]) / hx : 0.0
        dudy = hy > 0 ? (uc[i,jp,k] - uc[i,jm,k]) / hy : 0.0
        ω[i,j,k] = sqrt((dwdy - dvdz)^2 + (dudz - dwdx)^2 + (dvdx - dudy)^2)
    end
    ωmax = maximum(ω)
    thr = relative ? Float64(threshold) * ωmax : Float64(threshold)
    (relative && ωmax <= 0) && return falses(Nx, Ny, Nz)
    return ω .> thr
end

"""
    refinement_criterion_richardson(x; field=:θ, threshold=0.05, p=2) -> BitArray{3}

Richardson-extrapolation error estimate — the most defensible criterion
because it estimates the *discretization error itself* rather than a proxy
for it. The field is restricted to a 2× coarser mesh (in every direction
whose extent is even), prolonged back trilinearly, and the difference
divided by `2^p - 1` (with `p` the scheme's formal order) gives the local
error estimate. Normalized by the field's range; cells exceeding
`threshold` are flagged.
"""
function refinement_criterion_richardson(x; field::Symbol=:θ, threshold::Real=0.05,
                                         p::Int=2)
    f = _field_of(x, field)
    a = Float64.(Array(interior_view(f)))
    Nx, Ny, Nz = size(a)
    fx = iseven(Nx) ? 2 : 1; fy = iseven(Ny) ? 2 : 1; fz = iseven(Nz) ? 2 : 1
    Cx, Cy, Cz = Nx ÷ fx, Ny ÷ fy, Nz ÷ fz
    c = zeros(Float64, Cx, Cy, Cz)
    @inbounds for k in 1:Cz, j in 1:Cy, i in 1:Cx
        s = 0.0
        for kk in 1:fz, jj in 1:fy, ii in 1:fx
            s += a[(i-1)*fx+ii, (j-1)*fy+jj, (k-1)*fz+kk]
        end
        c[i,j,k] = s / (fx*fy*fz)
    end
    # trilinear prolongation back to the fine mesh
    b = zeros(Float64, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        i0, ti = _prolong_1d(i, fx, Cx)
        j0, tj = _prolong_1d(j, fy, Cy)
        k0, tk = _prolong_1d(k, fz, Cz)
        i1 = min(i0+1, Cx); j1 = min(j0+1, Cy); k1 = min(k0+1, Cz)
        b[i,j,k] =
            (1-ti)*(1-tj)*(1-tk)*c[i0,j0,k0] + ti*(1-tj)*(1-tk)*c[i1,j0,k0] +
            (1-ti)*tj*(1-tk)*c[i0,j1,k0]     + ti*tj*(1-tk)*c[i1,j1,k0] +
            (1-ti)*(1-tj)*tk*c[i0,j0,k1]     + ti*(1-tj)*tk*c[i1,j0,k1] +
            (1-ti)*tj*tk*c[i0,j1,k1]         + ti*tj*tk*c[i1,j1,k1]
    end
    err = abs.(a .- b) ./ (2^p - 1)
    rng = maximum(a) - minimum(a)
    scale = rng > 0 ? rng : max(maximum(abs, a), eps())
    return (err ./ scale) .> threshold
end

function _prolong_1d(i::Int, f::Int, Nc::Int)
    f == 1 && return (i, 0.0)
    ci = (i - 0.5) / f + 0.5            # coarse-index coordinate of the fine centre
    i0 = clamp(floor(Int, ci), 1, max(1, Nc - 1))
    return (i0, ci - i0)                # may leave [0,1] → linear extrapolation at
end                                     #   the edges, so a linear field is exact

"""
    combine_criteria(masks...; mode=:union) -> BitArray{3}

Combine refinement masks. `mode`: `:union` (flag if any), `:intersection`
(flag only if all), `:majority` (flag if at least half).
"""
function combine_criteria(masks...; mode::Symbol=:union)
    isempty(masks) && error("combine_criteria: need at least one mask")
    sz = size(first(masks))
    all(m -> size(m) == sz, masks) ||
        error("combine_criteria: masks have mismatched sizes")
    if mode === :union
        out = falses(sz); for m in masks; out .|= m; end; return out
    elseif mode === :intersection
        out = trues(sz); for m in masks; out .&= m; end; return out
    elseif mode === :majority
        cnt = zeros(Int, sz); for m in masks; cnt .+= m; end
        return cnt .>= ceil(Int, length(masks) / 2)
    end
    error("combine_criteria: unknown mode $mode (:union, :intersection, :majority)")
end

# =========================================================================
# Berger-Rigoutsos clustering
# =========================================================================

const Box3 = NTuple{3,UnitRange{Int}}

function _bbox(mask::AbstractArray{Bool,3}, box::Box3)
    lo = [typemax(Int), typemax(Int), typemax(Int)]; hi = [0, 0, 0]
    found = false
    @inbounds for k in box[3], j in box[2], i in box[1]
        if mask[i,j,k]
            found = true
            lo[1] = min(lo[1], i); hi[1] = max(hi[1], i)
            lo[2] = min(lo[2], j); hi[2] = max(hi[2], j)
            lo[3] = min(lo[3], k); hi[3] = max(hi[3], k)
        end
    end
    found || return nothing
    return (lo[1]:hi[1], lo[2]:hi[2], lo[3]:hi[3])
end

function _count_flagged(mask::AbstractArray{Bool,3}, box::Box3)
    n = 0
    @inbounds for k in box[3], j in box[2], i in box[1]
        mask[i,j,k] && (n += 1)
    end
    return n
end

_box_volume(box::Box3) = length(box[1]) * length(box[2]) * length(box[3])

function _signature(mask::AbstractArray{Bool,3}, box::Box3, d::Int)
    n = length(box[d])
    sig = zeros(Int, n)
    @inbounds for k in box[3], j in box[2], i in box[1]
        mask[i,j,k] || continue
        t = d == 1 ? i - first(box[1]) + 1 :
            d == 2 ? j - first(box[2]) + 1 : k - first(box[3]) + 1
        sig[t] += 1
    end
    return sig
end

function _split_box(box::Box3, d::Int, s::Int)
    lo = first(box[d]); hi = last(box[d])
    b1 = ntuple(e -> e == d ? (lo : lo + s - 1) : box[e], 3)
    b2 = ntuple(e -> e == d ? (lo + s : hi) : box[e], 3)
    return (b1, b2)
end

"""
    _try_split(mask, box, min_size) -> ((box1, box2) | nothing)

Berger-Rigoutsos split selection. Longest dimension first: prefer a *hole*
in the signature (a slice with no flagged cells), otherwise the strongest
*inflection* of the signature's second difference (a sign change in
`Σᵢ₊₁ - 2Σᵢ + Σᵢ₋₁`). Falls back to a midpoint bisection so progress —
and therefore termination — is guaranteed.
"""
function _try_split(mask::AbstractArray{Bool,3}, box::Box3, min_size::Int)
    lens = (length(box[1]), length(box[2]), length(box[3]))
    dorder = sortperm(collect(lens); rev=true)
    # 1. holes in the signature
    for d in dorder
        n = lens[d]
        n >= 2 * min_size + 1 || continue
        sig = _signature(mask, box, d)
        cand = [z for z in min_size:(n - min_size) if sig[z] == 0]
        isempty(cand) && continue
        mid = (n + 1) / 2
        s = cand[argmin(abs.(cand .- mid))]
        return _split_box(box, d, s)
    end
    # 2. inflection point of the signature Laplacian
    bestd = 0; bests = 0; bestval = -1.0; signchange = false
    for d in dorder
        n = lens[d]
        n >= 2 * min_size + 2 || continue
        sig = _signature(mask, box, d)
        Δ = [sig[i+1] - 2sig[i] + sig[i-1] for i in 2:n-1]
        for t in 1:length(Δ)-1
            s = t + 1                       # split after slice s
            (min_size <= s <= n - min_size) || continue
            sc = Δ[t] * Δ[t+1] < 0
            val = abs(Δ[t] - Δ[t+1])
            if (sc && !signchange) || (sc == signchange && val > bestval)
                bestd = d; bests = s; bestval = val; signchange = sc
            end
        end
        signchange && break                 # longest dim with a real inflection wins
    end
    bestd != 0 && return _split_box(box, bestd, bests)
    # 3. midpoint bisection of the longest splittable dimension
    for d in dorder
        lens[d] >= 2 * min_size && return _split_box(box, d, lens[d] ÷ 2)
    end
    return nothing
end

"""
    cluster_flagged_cells(mask; min_efficiency=0.7, max_blocks=64, min_size=1)
        -> Vector{NTuple{3,UnitRange{Int}}}

Berger-Rigoutsos signature-based clustering: turn a scattered Bool flag
mask into a small set of efficient, non-overlapping rectangular boxes.

A box is accepted when its *efficiency* (flagged cells / box cells) reaches
`min_efficiency`, when it can no longer be split, or when the `max_blocks`
budget is exhausted; otherwise it is split at a signature hole or
inflection (see `_try_split`) and each half is re-tightened to its bounding
box. Report the achieved efficiency with [`box_efficiency`](@ref).
"""
function cluster_flagged_cells(mask::AbstractArray{Bool,3};
                               min_efficiency::Real=0.7, max_blocks::Int=64,
                               min_size::Int=1)
    root = _bbox(mask, (1:size(mask,1), 1:size(mask,2), 1:size(mask,3)))
    root === nothing && return Box3[]
    accepted = Box3[]
    queue = Box3[root]
    while !isempty(queue)
        box = popfirst!(queue)
        nf = _count_flagged(mask, box)
        if nf == 0
            continue
        end
        eff = nf / _box_volume(box)
        if eff >= min_efficiency || (length(accepted) + length(queue) + 1) >= max_blocks
            push!(accepted, box); continue
        end
        sp = _try_split(mask, box, min_size)
        if sp === nothing
            push!(accepted, box); continue
        end
        for hb in sp
            t = _bbox(mask, hb)
            t === nothing || push!(queue, t)
        end
    end
    return accepted
end

"""
    box_efficiency(mask, boxes) -> Float64

Achieved clustering efficiency: flagged cells inside `boxes` divided by the
total cell count of `boxes`.
"""
function box_efficiency(mask::AbstractArray{Bool,3}, boxes::AbstractVector)
    isempty(boxes) && return 1.0
    nf = sum(b -> _count_flagged(mask, b), boxes)
    nv = sum(_box_volume, boxes)
    return nv == 0 ? 1.0 : nf / nv
end

# =========================================================================
# Regridding
# =========================================================================

function _dilate(mask::AbstractArray{Bool,3}, b::Int)
    b <= 0 && return BitArray(mask)
    nx, ny, nz = size(mask)
    out = falses(nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        mask[i,j,k] || continue
        for kk in max(1,k-b):min(nz,k+b), jj in max(1,j-b):min(ny,j+b),
            ii in max(1,i-b):min(nx,i+b)
            out[ii,jj,kk] = true
        end
    end
    return out
end

"Zero the mask within `buffer` cells of `pb`'s sides that are not domain edges."
function _restrict_legal!(mask::AbstractArray{Bool,3}, h::AMRHierarchy,
                          pb::RefinementBlock, ℓ::Int, buffer::Int)
    buffer <= 0 && return mask
    N = _level_size(h, ℓ)
    n = size(mask)
    for d in 1:3
        w = min(buffer, n[d])
        if first(pb.bounds[d]) != 1
            d == 1 ? (mask[1:w, :, :] .= false) :
            d == 2 ? (mask[:, 1:w, :] .= false) : (mask[:, :, 1:w] .= false)
        end
        if last(pb.bounds[d]) != N[d]
            lo = n[d] - w + 1
            d == 1 ? (mask[lo:n[1], :, :] .= false) :
            d == 2 ? (mask[:, lo:n[2], :] .= false) : (mask[:, :, lo:n[3]] .= false)
        end
    end
    return mask
end

_eval_criterion(c, h::AMRHierarchy, pb::RefinementBlock) = c(pb)
function _eval_criterion(c::AbstractArray{Bool,3}, h::AMRHierarchy,
                         pb::RefinementBlock)
    pb.level == 1 || return falses(pb.mesh.Nx, pb.mesh.Ny, pb.mesh.Nz)
    size(c) == (pb.mesh.Nx, pb.mesh.Ny, pb.mesh.Nz) ||
        error("regrid!: mask size $(size(c)) does not match the base grid")
    return c
end

function _copy_overlap!(dst::Field, db::Box3, src::Field, sb::Box3, ov::Box3)
    dm = dst.mesh; sm = src.mesh
    dd = dst.data; sd = src.data
    @inbounds for k in ov[3], j in ov[2], i in ov[1]
        dd[dm.Hx + i - first(db[1]) + 1,
           dm.Hy + j - first(db[2]) + 1,
           dm.Hz + k - first(db[3]) + 1] =
        sd[sm.Hx + i - first(sb[1]) + 1,
           sm.Hy + j - first(sb[2]) + 1,
           sm.Hz + k - first(sb[3]) + 1]
    end
    return dst
end

function _init_block_state!(h::AMRHierarchy, nb::RefinementBlock,
                            old::Vector{<:RefinementBlock})
    # (a) conservative prolongation from the parent everywhere: the r³
    #     sub-cell averages sum back to the parent average, so the composite
    #     integral is untouched by creating new refinement.
    for v in h.vars
        interpolate_parent_to_child!(nb.state[v], nb.parent.state[v], nb.nest;
                                     order=h.order)
    end
    # (b) where an old block of the same level overlaps, keep its (more
    #     accurate) data. Both blocks are parent-cell aligned, so the overlap
    #     carries the same mass either way — still conservative.
    for ob in old
        ob.level == nb.level || continue
        ov = ntuple(d -> intersect(nb.bounds[d], ob.bounds[d]), 3)
        any(isempty, ov) && continue
        for v in h.vars
            _copy_overlap!(nb.state[v], nb.bounds, ob.state[v], ob.bounds, ov)
        end
    end
    _set_block_velocity!(h, nb, h.t)
    _zero_registers!(h, nb)
    return nb
end

"""
    regrid!(hierarchy, criteria; buffer=hierarchy.buffer, min_efficiency=0.7,
            max_blocks=64, min_size=1)

Rebuild the refined levels. `criteria` is either a function
`block -> Bool mask` (evaluated on every block of the level being refined)
or a Bool array over the base grid.

Per level: evaluate the criterion, dilate the flags by `buffer` cells,
clip to the legally nestable region (`buffer` cells inside the parent
block, except at domain boundaries), cluster with
[`cluster_flagged_cells`](@ref), then transfer state — conservative
prolongation where refinement is new, direct copy where an old block
overlapped, and the parent already holds the restricted averages where
refinement disappeared. Proper nesting is guaranteed by construction; see
[`check_proper_nesting`](@ref).
"""
function regrid!(h::AMRHierarchy{T}, criteria;
                 buffer::Int=h.buffer, min_efficiency::Real=0.7,
                 max_blocks::Int=64, min_size::Int=1) where {T}
    # parent halos must be valid before prolongation
    for ℓ in 1:length(h.levels), b in h.levels[ℓ]
        _fill_block_halos!(h, b)
    end
    newlevels = Vector{Vector{RefinementBlock{T}}}()
    push!(newlevels, h.levels[1])
    for ℓ in 1:h.max_level-1
        parents = newlevels[ℓ]
        isempty(parents) && break
        fresh = RefinementBlock{T}[]
        for pb in parents
            mask = _eval_criterion(criteria, h, pb)
            m2 = _dilate(mask, buffer)
            _restrict_legal!(m2, h, pb, ℓ, buffer)
            any(m2) || continue
            for bx in cluster_flagged_cells(m2; min_efficiency, max_blocks, min_size)
                gcb = ntuple(d -> (first(pb.bounds[d]) + first(bx[d]) - 1) :
                                  (first(pb.bounds[d]) + last(bx[d]) - 1), 3)
                fb = ntuple(d -> ((first(gcb[d]) - 1) * h.refinement[d] + 1) :
                                 (last(gcb[d]) * h.refinement[d]), 3)
                push!(fresh, _make_block(h, ℓ+1, fb, pb))
            end
        end
        old = length(h.levels) >= ℓ+1 ? h.levels[ℓ+1] : RefinementBlock{T}[]
        for nb in fresh
            _init_block_state!(h, nb, old)
        end
        isempty(fresh) && break
        push!(newlevels, fresh)
    end
    h.levels = newlevels
    for lv in h.levels, b in lv
        empty!(b.children)
    end
    for ℓ in 2:length(h.levels), b in h.levels[ℓ]
        push!(b.parent.children, b)
    end
    # bring the coarse levels back in sync with the (possibly new) fine data
    for ℓ in length(h.levels):-1:2, b in h.levels[ℓ]
        _restrict_block!(h, b)
    end
    return h
end

"""
    check_proper_nesting(hierarchy; buffer=hierarchy.buffer) -> Bool

`true` when every level-`n+1` block, coarsened to level-`n` index space,
lies inside a level-`n` block with at least `buffer` level-`n` cells of
margin on every side — except on sides where the containing level-`n`
block itself abuts the domain boundary, where the margin is unnecessary
(and impossible).
"""
function check_proper_nesting(h::AMRHierarchy; buffer::Int=h.buffer)
    for ℓ in 2:length(h.levels)
        N = _level_size(h, ℓ-1)
        for b in h.levels[ℓ]
            pb = b.parent
            pb === nothing && return false
            pb.level == ℓ-1 || return false
            cb = _coarse_bounds(h, b)
            for d in 1:3
                # coarsening must be exact (block aligned to parent cells)
                (first(b.bounds[d]) - 1) % h.refinement[d] == 0 || return false
                last(b.bounds[d]) % h.refinement[d] == 0 || return false
                first(cb[d]) >= first(pb.bounds[d]) || return false
                last(cb[d])  <= last(pb.bounds[d])  || return false
                if first(pb.bounds[d]) != 1
                    first(cb[d]) - first(pb.bounds[d]) >= buffer || return false
                end
                if last(pb.bounds[d]) != N[d]
                    last(pb.bounds[d]) - last(cb[d]) >= buffer || return false
                end
            end
        end
    end
    return true
end

# =========================================================================
# Subcycled AMR time stepping
# =========================================================================

"""
`true` when `b` spans the whole domain along dimension `d` at its own level —
then there is no coarse-fine interface in `d`, only a periodic seam.
"""
_full_span(h::AMRHierarchy, b::RefinementBlock, d::Int) =
    length(b.bounds[d]) == _level_size(h, b.level)[d]

_periodic_dim(h::AMRHierarchy, d::Int) =
    d == 1 ? (h.bcs.west isa Periodic) :
    d == 2 ? (h.bcs.south isa Periodic) : (h.bcs.bottom isa Periodic)

# In a direction the block spans completely, the "boundary" is the domain's own
# periodic seam: wrap within the block instead of prolonging the coarse data, so
# the fine fluxes on the two sides of the seam agree exactly.
function _wrap_full_span!(h::AMRHierarchy, b::RefinementBlock)
    (b.level == 1 || b.parent === nothing) && return b
    m = b.mesh
    for v in h.vars
        f = b.state[v]; loc = location(f)
        if _full_span(h, b, 1) && _periodic_dim(h, 1)
            _fill_x!(f.data, Periodic(), Periodic(), m.Hx, m.Nx, loc)
        end
        if _full_span(h, b, 2) && _periodic_dim(h, 2)
            _fill_y!(f.data, Periodic(), Periodic(), m.Hy, m.Ny, loc)
        end
        if _full_span(h, b, 3) && _periodic_dim(h, 3)
            _fill_z!(f.data, Periodic(), Periodic(), m.Hz, m.Nz, loc)
        end
    end
    return b
end

function _fill_block_halos!(h::AMRHierarchy, b::RefinementBlock)
    h.velocity === nothing || _set_block_velocity!(h, b, h.t)
    if b.level == 1 || b.parent === nothing
        for v in h.vars
            fill_halos!(b.state[v], h.bcs)
        end
    else
        for v in h.vars
            interpolate_to_nest_boundary!(b.state[v], b.parent.state[v], b.nest;
                                         order=h.order)
        end
        _wrap_full_span!(h, b)
    end
    return b
end

_restrict_block!(h::AMRHierarchy, b::RefinementBlock) = begin
    b.parent === nothing && return b
    for v in h.vars
        restrict_to_parent!(b.parent.state[v], b.state[v], b.nest)
    end
    b
end

"""
Berger-Colella refluxing of one block into its parent: the coarse cells just
outside the block absorb `coarse flux − Σ fine fluxes` on every coarse-fine
interface plane. Directions in which the block spans the whole (periodic)
domain carry no interface and are skipped.
"""
function _reflux_block!(h::AMRHierarchy, b::RefinementBlock)
    b.parent === nothing && return b
    T = eltype(b.mesh)
    for v in h.vars
        mm = NestFluxMismatch{T}()
        for (ax, sym) in ((1, :x), (2, :y), (3, :z))
            (_full_span(h, b, ax) && _periodic_dim(h, ax)) && continue
            mm = mm + compute_nest_flux_mismatch(b.parent.state[_reg(v, sym)],
                                                 b.state[_reg(v, sym)], b.nest)
        end
        apply_flux_correction!(b.parent.state[v], mm, b.nest)
    end
    return b
end

function _advance_level!(h::AMRHierarchy, ℓ::Int, dt::Real)
    for b in h.levels[ℓ]
        _fill_block_halos!(h, b)
        h.advance!(h, b, dt, h.t)
    end
    if ℓ < length(h.levels) && !isempty(h.levels[ℓ+1])
        r = maximum(h.refinement)
        for b in h.levels[ℓ+1]
            _zero_registers!(h, b)
        end
        for _ in 1:r
            _advance_level!(h, ℓ+1, dt / r)
        end
        for b in h.levels[ℓ+1]
            _restrict_block!(h, b)
            _reflux_block!(h, b)
        end
    end
    return h
end

"""
    step_amr!(hierarchy, dt; regrid_interval=10, criterion=nothing)

One coarse step of the recursive, subcycled AMR advance:

  * advance every level-`ℓ` block by `dt`, accumulating face fluxes into
    the flux registers;
  * take `r = max(refinement)` substeps of level `ℓ+1` with `dt/r`, filling
    the fine halos from the coarse level by conservative prolongation;
  * **restrict** the fine interior onto the coarse cells and **reflux** the
    coarse-fine flux mismatch into the coarse cells just outside each fine
    block.

That last pair is what makes the composite integral exactly invariant:
the coarse level was advanced with coarse fluxes across the interface
while the fine level used the fine ones, and refluxing removes the
difference.

Regrids every `regrid_interval` coarse steps when `criterion !== nothing`
(pass `regrid_interval=0` to disable).
"""
function step_amr!(h::AMRHierarchy, dt::Real; regrid_interval::Integer=10,
                   criterion=nothing)
    if criterion !== nothing && regrid_interval > 0 && h.nstep % regrid_interval == 0
        regrid!(h, criterion)
    end
    for b in h.levels[1]
        _zero_registers!(h, b)
    end
    _advance_level!(h, 1, dt)
    h.t += dt
    h.nstep += 1
    return h
end

# =========================================================================
# AMR diagnostics
# =========================================================================

function _block_covered(h::AMRHierarchy, b::RefinementBlock)
    m = b.mesh
    cov = falses(m.Nx, m.Ny, m.Nz)
    for c in b.children
        cb = _coarse_bounds(h, c)
        lr = ntuple(d -> (first(cb[d]) - first(b.bounds[d]) + 1) :
                         (last(cb[d]) - first(b.bounds[d]) + 1), 3)
        cov[lr[1], lr[2], lr[3]] .= true
    end
    return cov
end

"""
    amr_total_mass(hierarchy; var=:θ) -> Float64

Composite volume integral of `var`: every cell of every block that is not
covered by a finer block, weighted by that level's cell volume. This is the
quantity [`step_amr!`](@ref) conserves exactly.
"""
function amr_total_mass(h::AMRHierarchy; var::Symbol=:θ)
    tot = 0.0
    for ℓ in 1:length(h.levels)
        d = _level_spacing(h, ℓ); V = d[1] * d[2] * d[3]
        for b in h.levels[ℓ]
            cov = _block_covered(h, b)
            m = b.mesh; θd = b.state[var].data
            @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
                cov[i,j,k] && continue
                tot += θd[m.Hx+i, m.Hy+j, m.Hz+k] * V
            end
        end
    end
    return tot
end

"""
    amr_error(hierarchy, exact; var=:θ) -> Float64

Volume-weighted composite L1 error of `var` against
`exact(x, y, z)`, evaluated on each block's uncovered cells at that
level's resolution.
"""
function amr_error(h::AMRHierarchy, exact; var::Symbol=:θ)
    num = 0.0; vol = 0.0
    for ℓ in 1:length(h.levels)
        d = _level_spacing(h, ℓ); V = d[1] * d[2] * d[3]
        for b in h.levels[ℓ]
            cov = _block_covered(h, b)
            m = b.mesh; θd = b.state[var].data
            @inbounds for k in 1:m.Nz, j in 1:m.Ny, i in 1:m.Nx
                cov[i,j,k] && continue
                e = θd[m.Hx+i, m.Hy+j, m.Hz+k] - exact(_xc(m,m.Hx+i), _yc(m,m.Hy+j),
                                                       _zc(m,m.Hz+k))
                num += abs(e) * V; vol += V
            end
        end
    end
    return vol > 0 ? num / vol : 0.0
end

end # module
