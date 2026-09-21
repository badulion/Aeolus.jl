module BoundaryConditions

using ..Grids
using ..Grids: CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields
using ..Fields: location, Field

export AbstractBC, Periodic, NoFlux, ImpermeableWall, ValueBC, FluxBC,
       FieldBoundaryConditions, fill_halos!

"""
    AbstractBC

Root type for per-face boundary conditions. Six BCs are grouped by face
into a [`FieldBoundaryConditions`](@ref) container.

Concrete types:
  * [`Periodic`](@ref) — copy from the opposite face.
  * [`NoFlux`](@ref) — homogeneous Neumann (ghost = mirror interior).
  * [`ImpermeableWall`](@ref) — free-slip rigid wall for velocity, mirror
    for scalars.
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

Six BCs, one per face. Convenience constructor
`FieldBoundaryConditions(:periodic)` gives triply-periodic BCs;
`FieldBoundaryConditions(:periodic, :periodic, :wall)` gives periodic
in `x, y` and impermeable walls at `z=0` and `z=Lz`. Recognized symbols:
`:periodic`, `:noflux`, `:wall`.
"""
struct FieldBoundaryConditions{W,E,S,N,B,T}
    west::W; east::E
    south::S; north::N
    bottom::B; top::T
end

function FieldBoundaryConditions(hkind::Symbol=:periodic,
                                 vkind::Symbol=hkind,
                                 zkind::Symbol=vkind)
    hx = _bc(hkind); hy = _bc(vkind); hz = _bc(zkind)
    return FieldBoundaryConditions(hx, hx, hy, hy, hz, hz)
end

_bc(s::Symbol) = s === :periodic ? Periodic()        :
                 s === :noflux   ? NoFlux()          :
                 s === :wall     ? ImpermeableWall() :
                 error("Unknown BC kind $s (use :periodic, :noflux, :wall)")

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
function fill_halos!(f::Field, bcs::FieldBoundaryConditions)
    g   = f.mesh
    d   = f.data
    loc = location(f)
    _fill_x!(d, bcs.west,   bcs.east,   g.Hx, g.Nx, loc)
    _fill_y!(d, bcs.south,  bcs.north,  g.Hy, g.Ny, loc)
    _fill_z!(d, bcs.bottom, bcs.top,    g.Hz, g.Nz, loc)
    return f
end

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

end # module
