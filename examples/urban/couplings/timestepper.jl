# Time integration of the dynamics: slot `:timestepper`.
#
#   :fixed     one `step!` (forward-Euler projection) of cfg.dt per step
#              (the step every core provides; required by the sound-proof
#              cores).
#   :adaptive  (default) TimeIntegration.step_adaptive!: the step length is the
#              CFL / diffusion / buoyancy limit of `adaptive_dt`, growth-
#              limited; a step that fails (non-finite, divergence above
#              tolerance, velocity above a ceiling) is rolled back and
#              retried at half the step. With `imex = true` the dynamics
#              is ARK2 (below) and only the explicit limits bind.
#   :imex      TimeIntegration.step_imex! with ARK2 (ARS(2,2,2)) at cfg.dt:
#              implicit diffusion, explicit advection / buoyancy /
#              Coriolis / hooks, a projection per stage.
#
# The step length is chosen at the START of a case step (`next_dt`), so
# every stage of that step — surface, transport, microphysics — uses the
# same dt the dynamics covers. If step_adaptive! has to retry, the
# dynamics covers that dt in two or more shorter pieces (each piece gets
# its own `max_retries`), so the case clock and the dynamics never
# disagree. The dynamics hooks (IBM, canopy drag, geostrophic forcing)
# are pure functions of the velocity and its tendency, so re-running them
# on a retry or an IMEX stage is safe.
#
# Every provider passes the numerics slots' step keywords
# (`case.step_kwargs`: `momentum_advection` (#132), `sgs` (#167)) to the
# step it takes — `step!`, or `step_imex!`, which takes the same keywords —
# so the momentum-advection and SGS choices hold under every time stepper.
# The SGS closure commits its memory once per accepted step: the adaptive
# provider also hands it to `step_adaptive!`, which rolls it back with the
# state when a step is rejected, and estimates the diffusive limit with the
# closure's pure `compute_sgs!` (νt and κt of the current flow; no commit).
#
# `adaptive_dt`, `step_adaptive!` and `step_imex!` are typed on
# BoussinesqSimulation, so both providers are incompatible with the
# anelastic and pseudo-incompressible cores (couplings/core_select.jl).

const _SOUNDPROOF_CORES = (:core => :anelastic, :core => :pseudo_incompressible)
const _SOUNDPROOF_REASON =
    "TimeIntegration's adaptive_dt / step_adaptive! / step_imex! are defined for the " *
    "Boussinesq core only (they are typed on BoussinesqSimulation); use timestepper = :fixed " *
    "with the anelastic or pseudo-incompressible core"

"""
    FixedStep

Slot `:timestepper`, provider `:fixed`: one `step!(sim, dt)` of
the dynamical core per case step, `dt = cfg.dt`, with the numerics
slots' step keywords ([`dynamics_step!`](@ref), the contract default of
`advance_dynamics!`). Works with every core (`step!` dispatches on the
simulation type), so it is the stepper of the anelastic and
pseudo-incompressible cores. Not the default: at the flagship's
`cfg.dt = 2 s` the winter case is unstable (θ spans −95…587 K by
step 141), and in summer the step exceeds the advective limit the
adaptive stepper keeps to (it takes 1.67 s on average, 0.76 s by 13:00).
"""
struct FixedStep <: AbstractCoupling end

coupling_name(::Type{FixedStep}) = :timestep_fixed
slot(::Type{FixedStep}) = :timestepper
provider(::Type{FixedStep}) = :fixed
is_default(::Type{FixedStep}) = false
modules_used(::Type{FixedStep}) = (:TimeSteppers,)

"""
    StepStats

Statistics of the adaptive / IMEX steps taken: `n` case steps, `pieces`
(dynamics sub-steps; > `n` only after a retry), `retries` (total) and
`max_retries_step` (most in one case step), the smallest / largest /
summed `dt`, the largest advective CFL number `dt · Σ|uᵢ|/Δxᵢ` measured
at the start of a step, how often each process limited the step, and the implicit-solver statistics of the last IMEX step.
"""
mutable struct StepStats
    n::Int
    pieces::Int
    retries::Int
    max_retries_step::Int
    dt_min::Float64
    dt_max::Float64
    dt_sum::Float64
    cfl_max::Float64
    limiters::Dict{Symbol,Int}
    imex_iterations::Int
    imex_residual::Float64
end
StepStats() = StepStats(0, 0, 0, 0, Inf, 0.0, 0.0, 0.0, Dict{Symbol,Int}(), 0, 0.0)

function _record_dt!(s::StepStats, dt, cfl)
    s.n += 1
    s.dt_min = min(s.dt_min, dt)
    s.dt_max = max(s.dt_max, dt)
    s.dt_sum += dt
    s.cfl_max = max(s.cfl_max, cfl)
    return s
end

"""
    AdaptiveStep

Slot `:timestepper`, provider `:adaptive` (default): `TimeIntegration.adaptive_dt`
picks each step (`CFLConstraints` from the options `C_advective`,
`C_diffusive`, `C_buoyancy`, `C_coriolis`, `dt_min`, `dt_max`,
`growth_max`) and `step_adaptive!` takes it, rolling back and halving on
failure (`max_retries`, `divergence_tol`, `max_velocity`). With
`imex = true` the dynamics is `step_imex!(…, ARK2())` and the step is
limited only by the explicitly treated processes.

Each step (and each retried piece) is `step!` — or `step_imex!` — with
the numerics slots' `case.step_kwargs`. The diffusive limit is computed
from the selected SGS closure's ν_t and κ_t of the current flow, and the
closure's memory is snapshot / restored by `step_adaptive!` around a
rejected step, so it commits exactly once per accepted piece.

`dt_max` is further capped at the shortest period in `cfg.cadence`, so a
step never skips a firing of a cadence (the soil and building couplings
integrate over their period). `run_case!` runs to the simulated time
`nsteps · cfg.dt` rather than a number of steps; the first step grows
from `cfg.dt` (at most `growth_max · cfg.dt`).
"""
mutable struct AdaptiveStep{S} <: AbstractCoupling
    constraints::CFLConstraints{Float64}
    scheme::S                      # nothing (explicit step!) or ARK2()
    max_retries::Int
    divergence_tol::Float64
    max_velocity::Float64
    dt_prev::Float64
    rate::Float64                  # advective rate at the start of the step (s⁻¹)
    stats::StepStats
end

coupling_name(::Type{<:AdaptiveStep}) = :timestep_adaptive
slot(::Type{<:AdaptiveStep}) = :timestepper
provider(::Type{<:AdaptiveStep}) = :adaptive
modules_used(::Type{<:AdaptiveStep}) = (:TimeIntegration, :TimeSteppers)   # step_adaptive! drives the generic step!
incompatible(::Type{<:AdaptiveStep}) = _SOUNDPROOF_CORES
incompatible_reason(::Type{<:AdaptiveStep}) = _SOUNDPROOF_REASON
variable_dt(::Type{<:AdaptiveStep}) = true
default_options(::Type{<:AdaptiveStep}) =
    (C_advective = 0.5, C_diffusive = 0.25, C_buoyancy = 0.5, C_coriolis = 0.5,
     dt_min = 0.01, dt_max = 10.0, growth_max = 1.1,
     max_retries = 3, divergence_tol = 1e-6, max_velocity = 50.0, imex = false)
smoke_variants(::Type{<:AdaptiveStep}) =
    ((label = "imex = true", physics = Dict{Symbol,Any}(:timestepper => :adaptive),
      options = Dict{Symbol,Any}(:timestep_adaptive => (imex = true,))),)

function build(::Type{<:AdaptiveStep}, case, o)
    # A step never spans a whole cadence period: couplings that integrate
    # over their period (soil, the building model) would lose time.
    dt_max = min(o.dt_max, minimum(p for p in values(case.cfg.cadence) if p > 0; init = Inf))
    c = CFLConstraints(; C_advective = o.C_advective, C_diffusive = o.C_diffusive,
                       C_buoyancy = o.C_buoyancy, C_coriolis = o.C_coriolis,
                       dt_min = o.dt_min, dt_max = dt_max, growth_max = o.growth_max)
    return AdaptiveStep(c, o.imex ? ARK2() : nothing, o.max_retries, o.divergence_tol,
                        o.max_velocity, case.cfg.dt, 0.0, StepStats())
end

"""
    sgs_cfl_diagnostics(case, constraints)

`cfl_diagnostics` of the case's core with the SGS closure the next step
will run (the `sgs` of `case.step_kwargs`; none when `C_s = 0` and no
other closure is selected). The closure's coefficients are recomputed
from the current flow with the pure `compute_sgs!`: no closure memory
advances, however often the limit is estimated.
"""
sgs_cfl_diagnostics(case, k) = cfl_diagnostics(case.sim, k; sgs = get(case.step_kwargs, :sgs, nothing))

function next_dt(c::AdaptiveStep, case)
    sim, k = case.sim, c.constraints
    if c.scheme === nothing
        d = sgs_cfl_diagnostics(case, k)
        lim = d.limiter
    else
        l = imex_stability_limit(sim, c.scheme; constraints = k,
                                 sgs = get(case.step_kwargs, :sgs, nothing))
        d, lim = l.diagnostics, l.limiter
    end
    dt = adaptive_dt(sim, k; dt_prev = c.dt_prev, diagnostics = d, scheme = c.scheme)
    c.rate = d.advective_rate
    c.stats.limiters[lim] = get(c.stats.limiters, lim, 0) + 1
    return dt
end

function advance_dynamics!(c::AdaptiveStep, case, dt)
    sim, s = case.sim, c.stats
    skw, scheme = case.step_kwargs, c.scheme
    sgs = get(skw, :sgs, nothing)      # step_adaptive! snapshots its memory
    # the step (and every retry) with the numerics slots' keywords
    stepper = scheme === nothing ? ((m, h; kw...) -> step!(m, h; kw..., skw...)) :
                                   ((m, h; kw...) -> step_imex!(m, h, scheme; kw..., skw...))
    remaining, nretry, piece_min = Float64(dt), 0, Inf
    while remaining > 1e-12 * dt
        _, used, r = step_adaptive!(sim, c.constraints; dt = remaining,
                                    body_force! = case.body_force,
                                    post_predictor! = case.post_predictor,
                                    stepper = stepper,
                                    max_retries = c.max_retries, scheme = c.scheme,
                                    divergence_tol = c.divergence_tol,
                                    max_velocity = c.max_velocity, sgs = sgs)
        remaining -= used
        nretry += r
        piece_min = min(piece_min, used)
        s.pieces += 1
    end
    s.retries += nretry
    s.max_retries_step = max(s.max_retries_step, nretry)
    _record_dt!(s, dt, dt * c.rate)
    # growth from the step that succeeded: after a retry, the shortest piece
    c.dt_prev = nretry == 0 ? Float64(dt) : piece_min
    if c.scheme !== nothing
        ws = imex_workspace(sim)
        s.imex_iterations, s.imex_residual = ws.total_iterations, ws.max_residual
    end
    return nothing
end

"The advective rate max Σ|uᵢ|/Δxᵢ of `cfl_diagnostics` (s⁻¹); dt × this is the CFL number."
advective_rate(sim) = cfl_diagnostics(sim; update_νt = false).advective_rate

report(c::AdaptiveStep, case) = step_report(c.stats, "adaptive" * (c.scheme === nothing ? "" : " (ARK2)"))

"""
    step_report(stats, label) -> String

One log line of [`StepStats`](@ref): steps, dt range and mean, retries,
largest CFL number and the limiting processes.
"""
function step_report(s::StepStats, label)
    s.n == 0 && return "  $label: no steps yet"
    lims = join(("$k $v" for (k, v) in sort!(collect(s.limiters); by = last, rev = true)), ", ")
    return @sprintf("  %s: %d steps, dt %.3f–%.3f s (mean %.3f), %d retries (max %d in a step), max CFL %.2f; limited by %s",
                    label, s.n, s.dt_min, s.dt_max, s.dt_sum / s.n, s.retries, s.max_retries_step,
                    s.cfl_max, isempty(lims) ? "-" : lims)
end

function smoke_check(c::AdaptiveStep, case)
    s, k = c.stats, c.constraints
    return ["steps were taken" => s.n > 0,
            "dt within [dt_min, dt_max]" => k.dt_min <= s.dt_min && s.dt_max <= k.dt_max,
            "the CFL-limited steps needed no retry" => s.retries == 0,
            # the explicit advective limit C_advective holds at the step start
            # (the IMEX pick uses the same advective limit)
            "advective CFL number ≤ C_advective (+1e-9)" => s.cfl_max <= k.C_advective + 1e-9,
            "the case clock is the dynamics clock" => isapprox(case.sim.t, s.dt_sum; rtol = 1e-12)]
end

"""
    IMEXStep

Slot `:timestepper`, provider `:imex`: `step_imex!(sim, cfg.dt, ARK2())`
— the diffusion (molecular + SGS, frozen ν_eff) implicit, the rest
explicit, a projection after each of the two stages. The numerics slots'
step keywords are passed on: the momentum-advection operator is the
explicit one, and the `sgs` closure supplies the frozen ν_t and κ_t. Options `tol`
and `maxit` of the implicit CG solves. Buys nothing at a dt the explicit
step is stable at (it costs two projections and four implicit solves per
step); it is what lets `cfg.dt` exceed the diffusive limit Δ²/ν_eff.
"""
struct IMEXStep{S} <: AbstractCoupling
    scheme::S
    tol::Float64
    maxit::Int
    stats::StepStats
end

coupling_name(::Type{<:IMEXStep}) = :timestep_imex
slot(::Type{<:IMEXStep}) = :timestepper
provider(::Type{<:IMEXStep}) = :imex
is_default(::Type{<:IMEXStep}) = false
modules_used(::Type{<:IMEXStep}) = (:TimeIntegration,)
incompatible(::Type{<:IMEXStep}) = _SOUNDPROOF_CORES
incompatible_reason(::Type{<:IMEXStep}) = _SOUNDPROOF_REASON
default_options(::Type{<:IMEXStep}) = (tol = 1e-8, maxit = 200)
build(::Type{<:IMEXStep}, case, o) = IMEXStep(ARK2(), o.tol, o.maxit, StepStats())

function advance_dynamics!(c::IMEXStep, case, dt)
    sim = case.sim
    cfl = dt * advective_rate(sim)
    ws = imex_workspace(sim)
    step_imex!(sim, dt, c.scheme; body_force! = case.body_force,
               post_predictor! = case.post_predictor, workspace = ws,
               tol = c.tol, maxit = c.maxit, case.step_kwargs...)
    c.stats.pieces += 1
    _record_dt!(c.stats, dt, cfl)
    c.stats.imex_iterations, c.stats.imex_residual = ws.total_iterations, ws.max_residual
    return nothing
end

report(c::IMEXStep, case) =
    step_report(c.stats, "imex (ARK2)") *
    @sprintf("; last step %d CG iterations, max relative residual %.1e",
             c.stats.imex_iterations, c.stats.imex_residual)

function smoke_check(c::IMEXStep, case)
    s = c.stats
    return ["steps were taken" => s.n > 0,
            "implicit diffusion solves converged (residual ≤ tol)" => s.imex_residual <= c.tol,
            "∇·u at round-off after the stage projections (< 1e-10 s⁻¹)" => max_divergence(case.sim) < 1e-10]
end

register!(FixedStep)
register!(AdaptiveStep)
register!(IMEXStep)
