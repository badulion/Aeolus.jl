module BoundaryConditions

using ..Grids
using ..Grids: CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields
using ..Fields: location, Field

export AbstractBC, Periodic, NoFlux, ImpermeableWall, ValueBC, FluxBC,
       FieldBoundaryConditions, fill_halos!
export NoSlipWall, is_wall, homogeneous_bcs     # P59-a (#172)
export FreeStreamBoundary                        # P59-d (#185)
export axis_poisson_bc                           # N-open-a (#251)
export AbstractOpenBoundary, OpenBoundary, OpenPlane, OpenBoundaryState,   # N-open-b (#255)
       AbstractBoundaryData, ProfileBoundaryData, FunctionBoundaryData,
       boundary_values!, open_boundary_conditions, set_boundary_data!,
       has_open_boundaries, open_boundary_states, update_open_boundaries!,
       prepare_open_boundaries!, correct_boundary_mass_flux!, open_boundary_flux,
       boundary_flux_weights, open_boundary_ledger, pinned_face_indices,
       open_boundary_snapshot, restore_open_boundaries!, combine_open_boundaries!,
       assert_open_face_correction

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
    scalars come from boundary data (Dirichlet), with a Neumann `φ`.
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
    FluxBC(f)

Neumann boundary condition: prescribed flux `∂/∂n = f` across the
boundary. Currently applied only in the z-direction on cell-centered
scalars — used for prescribed surface heat flux `w'θ'|₀`.
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
# or any two impermeable walls (free slip and/or no slip, mixed allowed).
_z_pair_supported(b, t) = nameof(typeof(b)) === nameof(typeof(t)) ||
                          (is_wall(b) && is_wall(t))

_bc(bc::AbstractBC) = bc
_bc(s::Symbol) = s === :periodic ? Periodic()        :
                 s === :noflux   ? NoFlux()          :
                 s === :wall     ? ImpermeableWall() :
                 s === :noslip   ? _noslip_default() :
                 error("Unknown BC kind $s (use :periodic, :noflux, :wall, :noslip)")

# =========================================================================
# Halo filling
# =========================================================================

"""
    fill_halos!(field, bcs)

Populate halo cells of `field` according to `bcs`
(a [`FieldBoundaryConditions`](@ref)). Multi-dispatches on the field's
[`Location`](@ref Aeolus.Grids.Location) so that e.g. an impermeable wall
zeros `w` at the wall face but mirrors `u, v`.
"""
function fill_halos!(f::Field, bcs::FieldBoundaryConditions; key = f.data)
    g   = f.mesh
    d   = f.data
    loc = location(f)
    _fill_x!(d, bcs.west,   bcs.east,   g.Hx, g.Nx, loc, key)
    _fill_y!(d, bcs.south,  bcs.north,  g.Hy, g.Ny, loc, key)
    _fill_z!(d, bcs.bottom, bcs.top,    g.Hz, g.Nz, loc, key)
    return f
end

# `key` identifies the field to boundary conditions that hold per-field
# data ([`OpenBoundary`](@ref)): a scratch field filled on behalf of
# another (an implicit solver's search direction) passes that field's
# array. Every other BC ignores it.
_fill_x!(d, a, b, H, N, loc::Location, key) = _fill_x!(d, a, b, H, N, loc)
_fill_y!(d, a, b, H, N, loc::Location, key) = _fill_y!(d, a, b, H, N, loc)
_fill_z!(d, a, b, H, N, loc::Location, key) = _fill_z!(d, a, b, H, N, loc)

# --- X direction ---------------------------------------------------------

function _fill_x!(d, ::Periodic, ::Periodic, Hx, Nx, ::Location)
    @inbounds for h in 1:Hx
        d[h,               :, :] .= d[Nx + h,      :, :]
        d[Hx + Nx + h,     :, :] .= d[Hx + h,      :, :]
    end
    return d
end

function _fill_x!(d, ::NoFlux, ::NoFlux, Hx, Nx, ::Location)
    @inbounds for h in 1:Hx
        d[Hx + 1 - h,      :, :] .= d[Hx + h,          :, :]
        d[Hx + Nx + h,     :, :] .= d[Hx + Nx + 1 - h, :, :]
    end
    return d
end

# --- Y direction ---------------------------------------------------------

function _fill_y!(d, ::Periodic, ::Periodic, Hy, Ny, ::Location)
    @inbounds for h in 1:Hy
        d[:, h,              :] .= d[:, Ny + h,      :]
        d[:, Hy + Ny + h,    :] .= d[:, Hy + h,      :]
    end
    return d
end

function _fill_y!(d, ::NoFlux, ::NoFlux, Hy, Ny, ::Location)
    @inbounds for h in 1:Hy
        d[:, Hy + 1 - h,     :] .= d[:, Hy + h,          :]
        d[:, Hy + Ny + h,    :] .= d[:, Hy + Ny + 1 - h, :]
    end
    return d
end

# --- Z direction ---------------------------------------------------------

function _fill_z!(d, ::Periodic, ::Periodic, Hz, Nz, ::Location)
    @inbounds for h in 1:Hz
        d[:, :, h]           .= d[:, :, Nz + h]
        d[:, :, Hz + Nz + h] .= d[:, :, Hz + h]
    end
    return d
end

function _fill_z!(d, ::NoFlux, ::NoFlux, Hz, Nz, ::Location)
    @inbounds for h in 1:Hz
        d[:, :, Hz + 1 - h]  .= d[:, :, Hz + h]
        d[:, :, Hz + Nz + h] .= d[:, :, Hz + Nz + 1 - h]
    end
    return d
end

# Impermeable wall — for FaceX/FaceY (u, v) and CellCenter (p, θ), behave
# as free-slip / Neumann → same as NoFlux (mirror).
function _fill_z!(d, ::ImpermeableWall, ::ImpermeableWall, Hz, Nz,
                  ::Union{CellCenter,FaceX,FaceY})
    @inbounds for h in 1:Hz
        d[:, :, Hz + 1 - h]  .= d[:, :, Hz + h]
        d[:, :, Hz + Nz + h] .= d[:, :, Hz + Nz + 1 - h]
    end
    return d
end

# Impermeable wall — for FaceZ (w): zero the two wall-face slots and
# extend the halo anti-symmetrically about each wall so the wall face
# stays exactly zero under stencil evaluation.
function _fill_z!(d::AbstractArray{T,3}, ::ImpermeableWall, ::ImpermeableWall,
                  Hz, Nz, ::FaceZ) where {T}
    zval = zero(T)
    @inbounds begin
        d[:, :, Hz + 1]      .= zval    # bottom wall face
        # top wall face lives at Hz+Nz+1 (in halo region for FaceZ storage)
        if Hz + Nz + 1 <= size(d, 3)
            d[:, :, Hz + Nz + 1] .= zval
        end

        for h in 1:Hz
            # below bottom wall: anti-symmetric about the wall at Hz+1
            d[:, :, Hz + 1 - h] .= .-d[:, :, Hz + 1 + h]
            # above top wall: anti-symmetric about the wall at Hz+Nz+1
            src = Hz + Nz + 1 - h
            dst = Hz + Nz + 1 + h
            if dst <= size(d, 3)
                d[:, :, dst] .= .-d[:, :, src]
            end
        end
    end
    return d
end

include("boundary_conditions/noslip_wall.jl")   # P59-a (#172)
include("boundary_conditions/open_boundary.jl") # N-open-b (#255)
include("boundary_conditions/free_stream.jl")   # P59-d (#185)

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

end # module
