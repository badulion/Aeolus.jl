module DataAssimilation

# Phase 20 (component) — Minimal data-assimilation hooks.
#
# Two functions cover the essentials at the level of "the model exposes
# the right seams" — the actual optimizer / EnKF loop lives above the
# model (typical DA drivers are their own systems):
#
#   * `Observation(field_symbol, i, j, k, value, sigma)` — a scalar
#     observation of a prognostic field at a grid cell with an error
#     standard deviation.
#   * `assimilate_nudging!(sim, obs, γ, dt)` — simple Newtonian nudging
#     toward the observation, `∂φ/∂t += γ · (obs - φ)` at the cell.
#
# The 4D-Var / EnKF drivers that actually use these seams live outside
# the simulation core; providing the seams here makes the model
# "assimilable" without committing to a specific DA framework.

using ..Grids: interior
using ..Fields: interior_view
using ..Simulations: BoussinesqSimulation

export Observation, assimilate_nudging!

"""
    Observation(field, i, j, k, value; sigma = 1.0)

A single scalar observation of the interior cell `(i, j, k)` of the
prognostic field named by the `Symbol` `field` — one of `:u`, `:v`,
`:w`, `:θ`, `:p`, `:qv`. `sigma` records the observational-error
standard deviation for use by an outer optimizer.
"""
struct Observation{T<:AbstractFloat}
    field::Symbol
    i::Int; j::Int; k::Int
    value::T
    sigma::T
end

Observation(field::Symbol, i, j, k, value; sigma = 1.0) =
    Observation{Float64}(field, i, j, k, Float64(value), Float64(sigma))

"""
    assimilate_nudging!(sim, obs, γ, dt)

Apply Newtonian nudging of the observed field toward `obs.value` at the
observation location: `field[i,j,k] += γ · (obs.value - field[i,j,k]) · dt`.
`γ` [1/s] is the relaxation rate — inverse of the nudging time scale.

This is the simplest DA method — a first-order low-pass toward the
observation. More sophisticated schemes (3D-Var, ensemble Kalman)
consume the same `Observation` list and can be layered on top.
"""
function assimilate_nudging!(sim::BoussinesqSimulation, obs::Observation,
                             γ::Real, dt::Real)
    field = _get_field(sim, obs.field)
    m = sim.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    i, j, k = obs.i + Hx, obs.j + Hy, obs.k + Hz
    field.data[i, j, k] += γ * dt * (obs.value - field.data[i, j, k])
    return sim
end

_get_field(sim::BoussinesqSimulation, s::Symbol) =
    s === :u ? sim.u : s === :v ? sim.v : s === :w ? sim.w :
    s === :θ ? sim.θ : s === :p ? sim.p : s === :qv ? sim.qv :
    error("DataAssimilation: unknown field symbol :$s")

end # module
