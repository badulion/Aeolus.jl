# P59-a (#172) — the no-slip / moving-wall boundary condition.
#
# Included inside `module BoundaryConditions`, after `ImpermeableWall` and
# its halo fill are defined. `ImpermeableWall` (free slip) is untouched:
# its two `_fill_z!` methods are more specific than the generic wall-pair
# method below, so `:wall` runs take exactly the code path they did before.

"""
    NoSlipWall(u_wall = 0, v_wall = 0)
    NoSlipWall(; u_wall = 0, v_wall = 0)

Impermeable rigid wall with a prescribed tangential velocity
`(u_wall, v_wall)` — a stationary no-slip wall when both are zero, a
moving wall (Couette lid, Stokes plate) otherwise. Valid on the `bottom`
and `top` (z) faces. Applied per location:

  * `u @ FaceX`, `v @ FaceY` are cell-centred in `z`, so the wall lies
    half a cell below the first interior value. The halo is filled
    *antisymmetrically about the wall value*,
    `u_ghost(h) = 2 u_wall − u_interior(h)`, so the average of the first
    interior value and its ghost — the wall-face value seen by every
    stencil — is exactly `u_wall`, and the one-sided wall gradient
    `(u₁ − u_ghost)/Δz = 2(u₁ − u_wall)/Δz` is second-order accurate.
  * `w @ FaceZ` and cell-centred fields (`p, θ, νₜ`) are treated exactly
    as by [`ImpermeableWall`](@ref): `w = 0` on the wall face with an
    antisymmetric halo, mirror (Neumann) for scalars. The pressure
    problem is therefore unchanged and the
    [`CosineFFTPoissonSolver`](@ref Aeolus.PressureSolvers.CosineFFTPoissonSolver)
    remains the exact solver.

A `NoSlipWall` may face an `ImpermeableWall` on the opposite side (an
open channel, `bottom = :noslip, top = :wall`); see
[`FieldBoundaryConditions`](@ref).

The plane-mean wall stress `ν ∂u/∂z` is returned by
`wall_shear_stress(sim; side)` (a method of the `CutCell` generic).

References: Ferziger & Perić (2002), *Computational Methods for Fluid
Dynamics*, 3rd ed., §8.10 (ghost-cell no-slip walls on a staggered grid);
Pope (2000), *Turbulent Flows*, §7.1 (laminar Couette and Poiseuille
flow).
"""
struct NoSlipWall{T<:Real} <: AbstractBC
    u_wall::T
    v_wall::T
end

function NoSlipWall(u::Real = 0.0, v::Real = 0.0; u_wall::Real = u, v_wall::Real = v)
    a, b = promote(float(u_wall), float(v_wall))
    return NoSlipWall{typeof(a)}(a, b)
end

_noslip_default() = NoSlipWall()
_is_noslip(bc) = bc isa NoSlipWall

"The tangential wall velocity a wall imposes on a field at `loc` (zero for free slip)."
_wall_velocity(::ImpermeableWall, ::Location, ::Type{T}) where {T} = zero(T)
_wall_velocity(bc::NoSlipWall, ::FaceX, ::Type{T}) where {T} = T(bc.u_wall)
_wall_velocity(bc::NoSlipWall, ::FaceY, ::Type{T}) where {T} = T(bc.v_wall)
_wall_velocity(::NoSlipWall, ::Location, ::Type{T}) where {T} = zero(T)

"""
    is_wall(bc) -> Bool

`true` for the impermeable z-wall types ([`ImpermeableWall`](@ref),
[`NoSlipWall`](@ref)): the face carries `w = 0` and a Neumann pressure
condition. Every solver that special-cases a wall asks this instead of
`bc isa ImpermeableWall`, so no-slip walls cannot fall back to free slip
at a missed dispatch site.
"""
is_wall(::AbstractBC) = false
is_wall(::ImpermeableWall) = true
is_wall(::NoSlipWall) = true

"""
    homogeneous_bcs(bcs::FieldBoundaryConditions) -> FieldBoundaryConditions

`bcs` with every moving [`NoSlipWall`](@ref) replaced by a stationary
one: the homogeneous part of an affine boundary operator. Implicit
solvers apply the operator to search directions and increments with
these, and move the wall-velocity contribution to the right-hand side.
Returns `bcs` itself (`===`) when nothing moves, so the existing BC types
take their usual code path.
"""
function homogeneous_bcs(bcs::FieldBoundaryConditions)
    _moving(bcs.bottom) || _moving(bcs.top) || return bcs
    return FieldBoundaryConditions(bcs.west, bcs.east, bcs.south, bcs.north,
                                   _homogeneous(bcs.bottom), _homogeneous(bcs.top))
end
_moving(::AbstractBC) = false
_moving(bc::NoSlipWall) = !(iszero(bc.u_wall) && iszero(bc.v_wall))
_homogeneous(bc::AbstractBC) = bc
_homogeneous(bc::NoSlipWall{T}) where {T} = NoSlipWall{T}(zero(T), zero(T))

# --- halo fill: any pair of z-walls where at least one is NoSlipWall -----
# (`ImpermeableWall, ImpermeableWall` keeps its own, more specific methods.)

const _ZWall = Union{ImpermeableWall, NoSlipWall}

function _fill_z!(d, bot::_ZWall, top::_ZWall, Hz, Nz, loc::Location)
    _fill_wall_bottom!(d, bot, Hz, loc)
    _fill_wall_top!(d, top, Hz, Nz, loc)
    return d
end

# u, v (FaceX/FaceY): antisymmetric about the wall velocity (mirror when
# it is a free-slip wall — ImpermeableWall keeps its free-slip meaning).
function _fill_wall_bottom!(d::AbstractArray{T,3}, bc::NoSlipWall,
                            Hz, loc::Union{FaceX,FaceY}) where {T}
    U2 = 2 * _wall_velocity(bc, loc, T)
    @inbounds for h in 1:Hz
        @views d[:, :, Hz + 1 - h] .= U2 .- d[:, :, Hz + h]
    end
    return d
end
function _fill_wall_top!(d::AbstractArray{T,3}, bc::NoSlipWall,
                         Hz, Nz, loc::Union{FaceX,FaceY}) where {T}
    U2 = 2 * _wall_velocity(bc, loc, T)
    @inbounds for h in 1:Hz
        @views d[:, :, Hz + Nz + h] .= U2 .- d[:, :, Hz + Nz + 1 - h]
    end
    return d
end

# Free-slip side of a mixed pair, and cell-centred fields at any wall:
# mirror (identical to the ImpermeableWall fill).
function _fill_wall_bottom!(d, bc::_ZWall, Hz, loc::Union{CellCenter,FaceX,FaceY})
    @inbounds for h in 1:Hz
        @views d[:, :, Hz + 1 - h] .= d[:, :, Hz + h]
    end
    return d
end
function _fill_wall_top!(d, bc::_ZWall, Hz, Nz, loc::Union{CellCenter,FaceX,FaceY})
    @inbounds for h in 1:Hz
        @views d[:, :, Hz + Nz + h] .= d[:, :, Hz + Nz + 1 - h]
    end
    return d
end

# w (FaceZ): zero on the wall face, antisymmetric halo — same as
# ImpermeableWall, so the pressure problem is unchanged.
function _fill_wall_bottom!(d::AbstractArray{T,3}, ::_ZWall, Hz, ::FaceZ) where {T}
    @inbounds begin
        @views d[:, :, Hz + 1] .= zero(T)
        for h in 1:Hz
            @views d[:, :, Hz + 1 - h] .= .-d[:, :, Hz + 1 + h]
        end
    end
    return d
end
function _fill_wall_top!(d::AbstractArray{T,3}, ::_ZWall, Hz, Nz, ::FaceZ) where {T}
    kw = Hz + Nz + 1
    @inbounds begin
        kw <= size(d, 3) && (@views d[:, :, kw] .= zero(T))
        for h in 1:Hz
            dst = kw + h
            dst <= size(d, 3) && (@views d[:, :, dst] .= .-d[:, :, kw - h])
        end
    end
    return d
end
