module TimeSteppers

using ..Grids
using ..Fields

export RK3, step!

"""
    RK3(state)

Shu–Osher strong-stability-preserving RK3 stepper.

Given a `state::NamedTuple{names, NTuple{N,Field}}`, `RK3(state)` allocates
scratch fields for the two intermediate stages plus a tendency buffer.

Advance one step with

    step!(stepper, state, tendency!, dt, t)

where `tendency!(k, state, t)` fills the tendency `k::NamedTuple` (matching
`state`) with ∂/∂t contributions. The caller is responsible for filling
halos of `state` — usually inside `tendency!` before it computes stencils.
"""
struct RK3{S}
    stage1::S
    stage2::S
    tendency::S
end

function RK3(state::NamedTuple)
    stage1   = _allocate_like(state)
    stage2   = _allocate_like(state)
    tendency = _allocate_like(state)
    return RK3(stage1, stage2, tendency)
end

function _allocate_like(state::NamedTuple)
    pairs = map(state) do f
        Field(Fields.location(f), f.mesh)
    end
    return pairs
end

"""
    step!(stepper, state, tendency!, dt, t = 0)
    step!(sim::BoussinesqSimulation, dt)

Advance one time step. The first method is the generic Shu–Osher SSP-RK3
driver (see [`RK3`](@ref)); the second is the fractional-step / pressure-
projection driver for [`BoussinesqSimulation`](@ref). Both add methods
to the same generic function.
"""
function step!(rk::RK3, state::NamedTuple, tendency!::Function, dt::Real, t::Real=0.0)
    k  = rk.tendency
    s1 = rk.stage1
    s2 = rk.stage2

    # Stage 1: s1 = u + dt * L(u)
    _zero!(k)
    tendency!(k, state, t)
    _combine!(s1, 1.0, state, dt, k)

    # Stage 2: s2 = 3/4 u + 1/4 (s1 + dt * L(s1))
    _zero!(k)
    tendency!(k, s1, t + dt)
    _combine_3!(s2, 0.75, state, 0.25, s1, 0.25*dt, k)

    # Stage 3: u = 1/3 u + 2/3 (s2 + dt * L(s2))
    _zero!(k)
    tendency!(k, s2, t + 0.5*dt)
    _combine_3!(state, 1/3, state, 2/3, s2, (2/3)*dt, k)

    return state
end

# out = a1 * A + a2 * K
function _combine!(out::NamedTuple, a1, A::NamedTuple, a2, K::NamedTuple)
    for name in keys(out)
        @. out[name].data = a1 * A[name].data + a2 * K[name].data
    end
end

# out = a1 * A + a2 * B + a3 * K
function _combine_3!(out::NamedTuple, a1, A::NamedTuple, a2, B::NamedTuple, a3, K::NamedTuple)
    for name in keys(out)
        @. out[name].data = a1 * A[name].data + a2 * B[name].data + a3 * K[name].data
    end
end

function _zero!(nt::NamedTuple)
    for name in keys(nt)
        fill!(nt[name].data, zero(eltype(nt[name])))
    end
end

end # module
