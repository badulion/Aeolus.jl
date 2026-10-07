# The per-step record and the diagnostics hook of the step cores (#336,
# epic #293). Included into `module Simulations`, before `StepWorkspaces`.
#
# Every step core (`step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`, the
# sound-proof cores, and per domain inside `step_nested!`) brackets its work
# with `_step_begin!` / `_step_end!`. Only the outermost bracket of a call
# tree counts as a step: `step_adaptive!` runs `step!` inside its own
# bracket, so one adaptive step is one step with its retries, not one per
# attempt. At the end of an outermost step the record holds the cheap
# quantities of that step and the record's hook, if one is installed, is
# called. The cheap quantities cost a few `time_ns()` calls and integer
# additions per step; the CFL numbers are those of the dt pick the step was
# taken with (`cfl_diagnostics`), never a sweep of their own.

"""
    StepRecord

What the step cores record about the steps of one simulation (#336): the
cheap quantities of every step, for the diagnostics registry
(`RunLog.diagnostics`) and the run monitor, and the **diagnostics hook**
of the simulation. One per simulation, in its [`StepWorkspaces`](@ref)
([`step_record`](@ref)); a [`NestedRun`](@ref Aeolus.Nesting.NestedRun)
has its own.

**The last step** (set at the end of every step):

| field | meaning |
| :---- | :------ |
| `steps` | steps completed since construction |
| `dt` | the step taken (s) |
| `limiter` | why it has this length: the binding limit of the dt pick (`:advective`, `:diffusive`, `:buoyancy`, `:coriolis`, `:sgs`; `:growth`, `:dt_max`, `:dt_min` when a clamp of [`adaptive_dt`](@ref Aeolus.TimeIntegration.adaptive_dt) bound), `:retry` (shortened by rejected attempts), `:prescribed` (a dt pick was made but the step taken differs, e.g. shortened to land on an output time), `:fixed` (no dt pick: a fixed-step run) |
| `retries` | rejected attempts of the step |

**The interval** since the last [`reset_interval!`](@ref) (the run
monitor resets it after each diagnostics row, so a row covers the steps
since the previous one, one or several):

| field | meaning |
| :---- | :------ |
| `n` | steps |
| `dt_sum` | the simulated time they covered (s) |
| `retries_sum` | rejected attempts |
| `cfl_adv` | largest advective Courant number `dt · max_cell Σ|uᵢ|/Δxᵢ` of a step, from the dt pick (`NaN` when no step of the interval had one) |
| `cfl_diff` | largest diffusive number `dt · max(ν + νₜ, κₜ) · (Δx⁻² + Δy⁻² + Δz_min⁻²)`, comparable with `C_diffusive` (`NaN` likewise) |
| `solves` | pressure solves |
| `solver_iters` | their iterations, summed ([`solver_stats`](@ref Aeolus.PressureSolvers.solver_stats); 0 for a direct solver) |
| `solver_residual` | the largest final residual (`NaN` for a direct solver) |
| `implicit_iters`, `implicit_residual` | the implicit-diffusion CG of `step_imex!`: iterations summed, largest residual |

**Wall time** (s, cumulative since construction): `wall_step` (inside
the outermost step cores), `wall_pressure` (the pressure solves),
`wall_sgs` (the SGS closure: coefficients and tendencies), `wall_cfl`
(the CFL sweeps of `cfl_diagnostics` and the sound-proof `adaptive_dt`,
also when a driver calls them between steps), `wall_guard`
(`step_adaptive!`'s snapshot, post-step guard and rollbacks).

The hook ([`set_step_hook!`](@ref)) is called as `hook(sim, record)` at
the end of every step, after the record is complete; the run monitor's
`attach` mode installs one (`Aeolus.IO.with_monitor`).
"""
mutable struct StepRecord
    hook::Any
    depth::Int
    steps::Int
    t_wall0::UInt64            # time_ns() at the outermost begin
    t_sim0::Float64            # simulated time at the outermost begin
    # the last step
    dt::Float64
    limiter::Symbol
    retries::Int
    # the dt pick of a state (cfl_diagnostics / adaptive_dt)
    pick_t::Float64            # simulated time of the state it was made from (NaN: none)
    pick_dt::Float64
    pick_rate::Float64         # max_cell Σ|uᵢ|/Δxᵢ (s⁻¹)
    pick_diff::Float64         # max(ν+νt, κt)(Δx⁻²+Δy⁻²+Δz⁻²) (s⁻¹)
    pick_limiter::Symbol
    # the interval
    n::Int
    dt_sum::Float64
    retries_sum::Int
    cfl_adv::Float64
    cfl_diff::Float64
    solves::Int
    solver_iters::Int
    solver_residual::Float64
    implicit_iters::Int
    implicit_residual::Float64
    # cumulative wall time
    wall_step::Float64
    wall_pressure::Float64
    wall_sgs::Float64
    wall_cfl::Float64
    wall_guard::Float64
end

StepRecord() = StepRecord(nothing, 0, 0, UInt64(0), NaN, NaN, :fixed, 0,
                          NaN, NaN, NaN, NaN, :fixed,
                          0, 0.0, 0, NaN, NaN, 0, 0, NaN, 0, NaN,
                          0.0, 0.0, 0.0, 0.0, 0.0)

"""
    reset_interval!(record::StepRecord) -> record

Start a new interval: zero the interval fields of `record` (`n`,
`retries_sum`, `solves`, the iteration sums; `NaN` for the maxima). The
run monitor calls it after each diagnostics row.
"""
function reset_interval!(r::StepRecord)
    r.n = 0; r.dt_sum = 0.0; r.retries_sum = 0
    r.cfl_adv = NaN; r.cfl_diff = NaN
    r.solves = 0; r.solver_iters = 0; r.solver_residual = NaN
    r.implicit_iters = 0; r.implicit_residual = NaN
    return r
end

# a NaN-ignoring maximum for the interval maxima (NaN = "none yet")
_nmax(a::Float64, b::Real) = isnan(b) ? a : isnan(a) ? Float64(b) : max(a, Float64(b))

_wall(t0::UInt64) = (time_ns() - t0) / 1e9

# Open a step bracket; returns whether it is the outermost one.
@inline function _step_begin!(r::StepRecord, sim)
    r.depth += 1
    r.depth == 1 || return false
    r.t_wall0 = time_ns()
    r.t_sim0 = Float64(sim.t)
    return true
end

# A step that throws: close its bracket without recording a step.
@inline _step_abort!(r::StepRecord) = (r.depth = max(r.depth - 1, 0); nothing)

# Close a step bracket. For the outermost one: record the step `dt` with its
# `retries`, the CFL numbers and the limiter of the dt pick made from the
# step's start state (if any), and call the hook.
function _step_end!(r::StepRecord, sim, dt::Real, outer::Bool; retries::Integer = 0)
    r.depth = max(r.depth - 1, 0)
    outer || return sim
    h = Float64(dt)
    r.wall_step += _wall(r.t_wall0)
    r.steps += 1
    r.n += 1
    r.dt_sum += h
    r.dt = h
    r.retries = retries
    r.retries_sum += retries
    if r.pick_t == r.t_sim0                                  # a pick from this step's start state
        r.cfl_adv = _nmax(r.cfl_adv, h * r.pick_rate)
        r.cfl_diff = _nmax(r.cfl_diff, h * r.pick_diff)
        r.limiter = h == r.pick_dt ? r.pick_limiter : retries > 0 ? :retry : :prescribed
    else
        r.limiter = retries > 0 ? :retry : :fixed
    end
    r.pick_t = NaN                                           # a pick is used once
    hk = r.hook
    hk === nothing || hk(sim, r)
    return sim
end

# Record the dt pick made from the current state of `sim`: `cfl_diagnostics`
# with the rates, `adaptive_dt` with the step it picked and its limiter.
function _note_pick!(r::StepRecord, sim, rate, diff, dt, limiter::Symbol)
    r.pick_t = Float64(sim.t)
    r.pick_rate = Float64(rate)
    r.pick_diff = Float64(diff)
    r.pick_dt = Float64(dt)
    r.pick_limiter = limiter
    return r
end
function _note_pick!(r::StepRecord, sim, dt, limiter::Symbol)   # adaptive_dt: the final pick
    Float64(sim.t) == r.pick_t || return r
    r.pick_dt = Float64(dt)
    r.pick_limiter = limiter
    return r
end

# After a pressure solve that started at `t0` (time_ns): its iterations,
# residual and wall time.
function _note_solve!(r::StepRecord, solver, t0::UInt64)
    r.wall_pressure += _wall(t0)
    it, res = solver_stats(solver)
    r.solves += 1
    r.solver_iters += it
    r.solver_residual = _nmax(r.solver_residual, res)
    return r
end

"""
    set_step_hook!(sim, hook) -> sim

Install `hook` (a function `hook(sim, record::StepRecord)`, or `nothing`
to remove it) as the diagnostics hook of `sim`: every step core calls it
at the end of each step of `sim` (#336). One hook per simulation; the run
monitor installs one in `attach` mode. See [`StepRecord`](@ref).
"""
function set_step_hook!(sim, hook)
    r = step_record(sim)
    r === nothing && throw(ArgumentError("set_step_hook!: a $(typeof(sim)) has no step record"))
    r.hook = hook
    return sim
end

"""
    step_hook(sim) -> hook or nothing

The diagnostics hook installed with [`set_step_hook!`](@ref) (`nothing`
when none, or when `sim` has no [`StepRecord`](@ref)).
"""
function step_hook(sim)
    r = step_record(sim)
    return r === nothing ? nothing : r.hook
end

"""
    step_record(sim) -> StepRecord or nothing

The [`StepRecord`](@ref) of `sim`: that of its [`StepWorkspaces`](@ref)
for the dynamical cores, the run's own for a `NestedRun`, `nothing` for
anything else.
"""
step_record(sim) = hasproperty(sim, :workspaces) && sim.workspaces isa StepWorkspaces ?
                   sim.workspaces.rec : nothing
