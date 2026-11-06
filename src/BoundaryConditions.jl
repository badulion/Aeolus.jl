module BoundaryConditions

using ..Grids
using ..Grids: CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields
using ..Fields: location, Field
using ..Backends: Backends, launch!, launch_weighted!, runs_serially

export AbstractBC, Periodic, NoFlux, ImpermeableWall, ValueBC, FluxBC,
       FieldBoundaryConditions, fill_halos!
export NoSlipWall, is_wall, homogeneous_bcs     # P59-a (#172)
export passive_scalar_bcs                        # qv faces (#254 + #261)
export FreeStreamBoundary                        # P59-d (#185)
export axis_poisson_bc                           # N-open-a (#251)
export neighbor_index, mark_open_boundary_solids!   # N-open-d (#262)
export AbstractOpenBoundary, OpenBoundary, OpenPlane, OpenBoundaryState,   # N-open-b (#255)
       AbstractBoundaryData, ProfileBoundaryData, FunctionBoundaryData,
       boundary_values!, open_boundary_conditions, set_boundary_data!, add_open_scalars!,
       has_open_boundaries, open_boundary_states, update_open_boundaries!,
       prepare_open_boundaries!, correct_boundary_mass_flux!, open_boundary_flux,
       boundary_flux_weights, open_boundary_ledger, pinned_face_indices,
       open_boundary_snapshot, restore_open_boundaries!, combine_open_boundaries!,
       open_boundary_checkpoint, restore_open_boundary_checkpoint!,   # #528
       assert_open_face_correction, open_intermediate_velocity!
export WindowedBoundaryData, snapshot_values, window_fraction, blend_window!   # #329
export synthetic_normal_velocity                                             # #418
export RadiationPlane, OUTFLOW_MODES, has_radiating_outflow,      # N-open-c (#260)
       rewind_open_boundaries!, mark_open_boundary_stage!

"""
    AbstractBC

Root type for per-face boundary conditions. Six BCs are grouped by face
into a [`FieldBoundaryConditions`](@ref) container.

Concrete types:
  * [`Periodic`](@ref) — copy from the opposite face.
  * [`NoFlux`](@ref) — homogeneous Neumann (ghost = mirror interior).
  * [`ImpermeableWall`](@ref) — free-slip rigid wall for velocity, mirror
    for scalars.
  * [`NoSlipWall`](@ref) — rigid wall with a prescribed tangential
    velocity (no slip, or a moving wall).
  * [`FreeStreamBoundary`](@ref) — top face with zero shear and a
    prescribed normal (transpiration) velocity.
  * [`OpenBoundary`](@ref) — lateral or top face whose velocities and
    scalars come from boundary data (Dirichlet), with a Neumann `φ`, and
    optionally a convective or Orlanski radiation outflow.
  * [`ValueBC`](@ref) — Dirichlet with prescribed value.
  * [`FluxBC`](@ref) — Neumann with prescribed flux `∂/∂n = f`.
"""
abstract type AbstractBC end

"Periodic wrap on this face."
struct Periodic <: AbstractBC end

"Zero-gradient (no-flux) reflecting boundary."
struct NoFlux <: AbstractBC end

"""
    ImpermeableWall

Free-slip, impermeable rigid wall. Applied per-face location:

  * On `w @ FaceZ` at a z-wall: sets the wall-face value to zero every
    halo fill, and uses anti-symmetric halo extension so the wall face
    stays exactly zero.
  * On `u, v @ FaceX/FaceY` at a z-wall: mirror reflection (free-slip).
  * On cell-centered fields (`p, θ`) at a z-wall: mirror reflection
    (Neumann).

Together these are the discrete BCs consistent with
[`CosineFFTPoissonSolver`](@ref Aeolus.PressureSolvers.CosineFFTPoissonSolver).
"""
struct ImpermeableWall <: AbstractBC end

"""
    ValueBC(v)

Dirichlet boundary condition: field takes value `v` at the boundary.
(Applied by `fill_halos!` for scalar cell-centered fields at z-boundaries.)
"""
struct ValueBC{T} <: AbstractBC
    v::T
end

"""
    FluxBC(Γ)

Neumann boundary condition: a prescribed outward normal gradient
`∂φ/∂n = Γ` of a cell-centred scalar. Supported as the **top** (z) face
of a scalar's boundary conditions (N-parent-b, #254), above a wall or a
no-flux bottom: the fixed-gradient top of a free atmosphere (PALM; DALES,
Heus et al. 2010, Eq. 39), with `Γ = ∂θ/∂z`, usually the initial lapse
rate. `fill_halos!` sets the ghost cells to

    φ(N + h) = φ(N + 1 − h) + Γ · (z_c(N + h) − z_c(N + 1 − h)),     h = 1 … Hz,

so the first ghost is `φ_N + Γ·Δz_f` with `Δz_f` the centre spacing
across the top face (`dz` on a `RegularGrid`, the face spacing on a
`StretchedGrid`), a linear profile of slope `Γ` is continued exactly
into every halo layer, and a diffusion operator with diffusivity `κ`
carries the flux `κΓ` (per unit area, into the domain for `Γ > 0` and
warmer air above) through the top face.

`FieldBoundaryConditions(:periodic, :periodic; bottom = :noflux, top = FluxBC(Γ))`.
Velocity fields (`FaceX/Y/Z`) cannot take it. The homogeneous part
(the operator an implicit solver iterates with, see
[`homogeneous_bcs`](@ref)) is `FluxBC(0)`, the mirror (no-flux) halo.
"""
struct FluxBC{T} <: AbstractBC
    f::T
end

"""
    FieldBoundaryConditions(west, east, south, north, bottom, top)
    FieldBoundaryConditions(x = :periodic, y = x, z = y; bottom = z, top = z)

Six BCs, one per face. Convenience constructor
`FieldBoundaryConditions(:periodic)` gives triply-periodic BCs;
`FieldBoundaryConditions(:periodic, :periodic, :wall)` gives periodic
in `x, y` and impermeable walls at `z=0` and `z=Lz`. Recognized symbols:
`:periodic`, `:noflux`, `:wall` (free slip) and, in `z` only, `:noslip`
(a stationary [`NoSlipWall`](@ref)).

The `bottom` and `top` keywords set the two z faces separately, as a
symbol or as a BC object — e.g. an open channel
`FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)`
or a Couette lid
`FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = NoSlipWall(U, 0))`.
A z-periodic face cannot be mixed with a non-periodic one, and a mixed
bottom/top pair must be two walls (`:wall`, `:noslip`, `NoSlipWall`) or a
wall below a [`FreeStreamBoundary`](@ref) or [`OpenBoundary`](@ref) top
(open boxes are built with [`open_boundary_conditions`](@ref));
anything else, e.g. `bottom = :noslip, top = :noflux`, throws an
`ArgumentError` rather than failing at the first halo fill.
"""
struct FieldBoundaryConditions{W,E,S,N,B,T}
    west::W; east::E
    south::S; north::N
    bottom::B; top::T
end

function FieldBoundaryConditions(hkind::Symbol=:periodic,
                                 vkind::Symbol=hkind,
                                 zkind::Symbol=vkind;
                                 bottom = zkind, top = zkind)
    hx = _bc(hkind); hy = _bc(vkind)
    (_is_noslip(hx) || _is_noslip(hy)) &&
        throw(ArgumentError(":noslip is a z-wall condition; use it for bottom/top only"))
    hb = _bc(bottom); ht = _bc(top)
    (hb isa Periodic) == (ht isa Periodic) ||
        throw(ArgumentError("a z-periodic face cannot be mixed with a non-periodic one " *
                            "(bottom = $(nameof(typeof(hb))), top = $(nameof(typeof(ht))))"))
    _z_pair_supported(hb, ht) ||
        throw(ArgumentError("unsupported z boundary pair bottom = $(nameof(typeof(hb))), " *
                            "top = $(nameof(typeof(ht))): the two z faces must be the same " *
                            "kind, or both walls (:wall / :noslip / NoSlipWall)"))
    return FieldBoundaryConditions(hx, hx, hy, hy, hb, ht)
end

# The z pairs `fill_halos!` has methods for: two faces of the same kind,
# or any two impermeable walls (free slip and/or no slip, mixed allowed),
# or a fixed-gradient (FluxBC) top above a wall or a no-flux bottom (#254).
_z_pair_supported(b, t) = nameof(typeof(b)) === nameof(typeof(t)) ||
                          (is_wall(b) && is_wall(t))
_z_pair_supported(b, t::FluxBC) = is_wall(b) || b isa NoFlux
_z_pair_supported(b::FluxBC, t::FluxBC) = false

_bc(bc::AbstractBC) = bc
_bc(s::Symbol) = s === :periodic ? Periodic()        :
                 s === :noflux   ? NoFlux()          :
                 s === :wall     ? ImpermeableWall() :
                 s === :noslip   ? _noslip_default() :
                 error("Unknown BC kind $s (use :periodic, :noflux, :wall, :noslip)")

# =========================================================================
# Halo filling
# =========================================================================

include("boundary_conditions/halo_kernels.jl")  # the per-axis kernels (KA-f, #297)

"""
    fill_halos!(field, bcs)

Populate halo cells of `field` according to `bcs`
(a [`FieldBoundaryConditions`](@ref)). Multi-dispatches on the field's
[`Location`](@ref Aeolus.Grids.Location) so that e.g. an impermeable wall
zeros `w` at the wall face but mirrors `u, v`.

The axes are filled x, then y, then z, over the whole face slab, so an
edge or corner holds the value of the last axis that reaches it. Each
fill phase is one [`launch!`](@ref Aeolus.Backends.launch!) of a per-axis
halo kernel (KA-f, #297): threaded for large faces, serial below the
backend's threshold, and identical bits at every thread count. When the
halo of the field is large enough to thread, the whole fill is the two
launches of the set fill, `fill_halos!((f,), bcs)` (SC-b, #402).
"""
fill_halos!(f::Field, bcs::FieldBoundaryConditions; key = f.data) =
    (fill_halos!((f,), bcs; keys = (key,)); f)

# The fill of one field, phase by phase (one launch each).
function _fill_one!(f::Field, bcs::FieldBoundaryConditions, key)
    g   = f.mesh
    d   = f.data
    loc = location(f)
    _fill_x!(d, bcs.west,   bcs.east,   g.Hx, g.Nx, loc, key)
    _fill_y!(d, bcs.south,  bcs.north,  g.Hy, g.Ny, loc, key)
    _fill_z!(d, bcs.bottom, bcs.top,    g.Hz, g.Nz, loc, g, key)
    return f
end

# `key` identifies the field to boundary conditions that hold per-field
# data ([`OpenBoundary`](@ref)): a scratch field filled on behalf of
# another (an implicit solver's search direction) passes that field's
# array. Every other BC ignores it.
_fill_x!(d, a, b, H, N, loc::Location, key) = _fill_x!(d, a, b, H, N, loc)
_fill_y!(d, a, b, H, N, loc::Location, key) = _fill_y!(d, a, b, H, N, loc)
_fill_z!(d, a, b, H, N, loc::Location, key) = _fill_z!(d, a, b, H, N, loc)

# The z fill with the mesh `g` and the key: only the fixed-gradient top
# (FluxBC, #254) needs the level heights; every other pair is the
# mesh-free method (with the key, for an open top).
_fill_z!(d, bot, top, Hz, Nz, loc::Location, g, key) = _fill_z!(d, bot, top, Hz, Nz, loc, key)

# Every fill below is a sequence of `launch!`es of the per-axis halo
# kernels (boundary_conditions/halo_kernels.jl, KA-f #297), one per fill
# phase of the face slab, in the order the pre-port loops ran.

# --- X direction ---------------------------------------------------------

_fill_x!(d, ::Periodic, ::Periodic, Hx, Nx, ::Location) = _halo_periodic!(d, 1, Hx, Nx)
_fill_x!(d, ::NoFlux, ::NoFlux, Hx, Nx, ::Location) = _halo_mirror!(d, 1, Hx, Nx)

# --- Y direction ---------------------------------------------------------

_fill_y!(d, ::Periodic, ::Periodic, Hy, Ny, ::Location) = _halo_periodic!(d, 2, Hy, Ny)
_fill_y!(d, ::NoFlux, ::NoFlux, Hy, Ny, ::Location) = _halo_mirror!(d, 2, Hy, Ny)

# --- Z direction ---------------------------------------------------------

_fill_z!(d, ::Periodic, ::Periodic, Hz, Nz, ::Location) = _halo_periodic!(d, 3, Hz, Nz)
_fill_z!(d, ::NoFlux, ::NoFlux, Hz, Nz, ::Location) = _halo_mirror!(d, 3, Hz, Nz)

# Impermeable wall — for FaceX/FaceY (u, v) and CellCenter (p, θ), behave
# as free-slip / Neumann → same as NoFlux (mirror).
_fill_z!(d, ::ImpermeableWall, ::ImpermeableWall, Hz, Nz, ::Union{CellCenter,FaceX,FaceY}) =
    _halo_mirror!(d, 3, Hz, Nz)

# Impermeable wall — for FaceZ (w): zero the two wall-face slots and
# extend the halo anti-symmetrically about each wall so the wall face
# stays exactly zero under stencil evaluation.
function _fill_z!(d::AbstractArray{T,3}, ::ImpermeableWall, ::ImpermeableWall,
                  Hz, Nz, ::FaceZ) where {T}
    _halo_set!(d, 3, Hz + 1, zero(T))                    # bottom wall face
    # top wall face lives at Hz+Nz+1 (in halo region for FaceZ storage)
    Hz + Nz + 1 <= size(d, 3) && _halo_set!(d, 3, Hz + Nz + 1, zero(T))
    # anti-symmetric about the walls at Hz+1 and Hz+Nz+1
    return _halo_negate!(d, 3, Hz, Nz, true, true)
end

include("boundary_conditions/noslip_wall.jl")   # P59-a (#172)
include("boundary_conditions/open_boundary.jl") # N-open-b (#255)
include("boundary_conditions/outflow.jl")       # N-open-c (#260)
include("boundary_conditions/free_stream.jl")   # P59-d (#185)
include("boundary_conditions/gradient_top.jl")  # N-parent-b (#254)
include("boundary_conditions/halo_sets.jl")     # SC-b (#402): threaded fills of field sets

# =========================================================================
# Pressure-Poisson boundary kind of a velocity face (N-open-a, #251)
# =========================================================================

"""
    axis_poisson_bc(face::AbstractBC) -> Symbol

The boundary condition the pressure correction `φ` of a projection method
obeys on a face whose *velocity* boundary condition is `face`:

* `:periodic` for [`Periodic`](@ref);
* `:neumann` (`∂φ/∂n = 0`) for every face on which the normal velocity is
  prescribed and the projection must leave it unchanged —
  [`ImpermeableWall`](@ref), [`NoSlipWall`](@ref), [`NoFlux`](@ref),
  and every [`AbstractOpenBoundary`](@ref) — [`FreeStreamBoundary`](@ref)
  (prescribed transpiration) and [`OpenBoundary`](@ref), whose normal
  velocity comes from boundary data. Because
  `u^{n+1}·n = u^*·n − Δt ∂φ/∂n`, the face keeps its prescribed value
  exactly when `∂φ/∂n = 0`.

A new face type that prescribes the normal velocity extends this with
`BoundaryConditions.axis_poisson_bc(::MyBC) = :neumann`. Types with no
method (e.g. [`ValueBC`](@ref), [`FluxBC`](@ref), which are scalar
conditions and say nothing about the normal velocity) throw an
`ArgumentError`, so a pressure solver is never silently built for the
wrong problem. `Simulations._default_solver` and the
`poisson_axis_bcs(bcs)` of `PressureSolvers` read the six faces through
this function.
"""
axis_poisson_bc(::Periodic)           = :periodic
axis_poisson_bc(::ImpermeableWall)    = :neumann
axis_poisson_bc(::NoSlipWall)         = :neumann
axis_poisson_bc(::NoFlux)             = :neumann
axis_poisson_bc(::AbstractOpenBoundary) = :neumann   # OpenBoundary, FreeStreamBoundary
axis_poisson_bc(bc::AbstractBC) =
    throw(ArgumentError("no pressure-Poisson boundary kind is defined for a " *
                        "$(nameof(typeof(bc))) velocity face; define " *
                        "BoundaryConditions.axis_poisson_bc(::$(nameof(typeof(bc))))"))

# =========================================================================
# The one BC-aware neighbour helper (N-open-d, #262)
# =========================================================================

"""
    neighbor_index(mesh, bcs, i, axis; face = false, outside = :mirror) -> Int

The interior index (`1…N` along `axis`, `1` = x, `2` = y, `3` = z) that
stands for the neighbour index `i` of a stencil, according to the
boundary conditions `bcs` of that axis:

* inside the domain (`1 ≤ i ≤ N`; with `face = true`, a field staggered
  along `axis`, `1 ≤ i ≤ N + 1`, whose `N + 1` is the far boundary face)
  `i` itself;
* across a [`Periodic`](@ref) axis, the wrapped index `mod1(i, N)`;
* across any other face (a wall, [`NoFlux`](@ref), an
  [`OpenBoundary`](@ref)): with `outside = :mirror` the mirror image
  about the face — the cell-centred zero-gradient neighbour (`0 → 1`,
  `N + 1 → N`), or for `face = true` the mirror about the boundary face
  (`0 → 2`, `N + 2 → N`); with `outside = :none` the sentinel `0`
  ("outside the domain: no neighbour").

This replaces every hard-coded periodic wrap (`mod1(i ± 1, N)`) of code
that must also run in an open (nested) domain: on a periodic axis it is
`mod1` bit for bit, and on an open axis a stencil no longer reaches
across the domain to the opposite face. With `face = true` the index
`N + 1` of an open face is the boundary-face slot of the normal velocity
(storage index `H + N + 1`, written by the open boundary), so
`(u[i] + u[i+1])/2` at `i = N` reads the prescribed boundary value. A
caller indexing the storage array adds the halo width `H`.
"""
function neighbor_index(m, bcs::FieldBoundaryConditions, i::Integer, axis::Integer;
                        face::Bool = false, outside::Symbol = :mirror)
    N, lo = axis == 1 ? (m.Nx, bcs.west) : axis == 2 ? (m.Ny, bcs.south) :
            axis == 3 ? (m.Nz, bcs.bottom) :
            throw(ArgumentError("neighbor_index: axis must be 1, 2 or 3, got $axis"))
    lo isa Periodic && return mod1(i, N)
    hi = face ? N + 1 : N
    1 <= i <= hi && return Int(i)
    outside === :none && return 0
    outside === :mirror ||
        throw(ArgumentError("neighbor_index: outside must be :mirror or :none, got :$outside"))
    j = face ? (i < 1 ? 2 - i : 2hi - i) : (i < 1 ? 1 - i : 2N + 1 - i)
    return clamp(j, 1, hi)
end

end # module
