# The single ordered step loop.
#
# One step of the Berlin case is this fixed sequence of stages. Within a
# stage the entries run in the integer `order` their couplings declare
# (ties: registry order). The order reproduces the pre-refactor script
# exactly — floating-point order included — so the refactor is
# bit-for-bit.
#
#   :clock           local hour of day (core)
#   :nest_parent     the parent run advanced to cover the child step, the
#                    child's nest faces pointed at the new parent window (#265)
#   :radiation       sun, radiosity, buildings over the radiation interval
#   :sw_modifiers    tree shade, cloud shadow
#   :surface_flux    surface → air fluxes; builds bus.sensible
#   :anthropogenic   buildings, traffic, glazing
#   :chemistry       reactive species, aerosol
#   :transport       advection of the registered tracers
#   :microphysics    cloud / rain
#   :tracers         passive tracers with their own schemes
#   :fog_dew         dew on surfaces, fog in the air
#   -- dynamics: step!(sim, dt) with the :body_force and :post_predictor
#      entries composed into its body_force! / post_predictor! hooks --
#   :surface_update  surface and soil prognostics
#   :column_tendency column (radiative) tendencies of θ
#   :nest_feedback   the parent window closed when the child reached its end
#                    (one-way: clock and statistics; two-way, #267: the feedback)
#   :sanity          guards (e.g. clamps), then the core blow-up check
#   :output          NetCDF record and log line (core)

"Stages before the dynamics, in order."
const PRE_DYNAMICS_STAGES = (:clock, :nest_parent, :radiation, :sw_modifiers, :surface_flux,
                             :anthropogenic, :chemistry, :transport, :microphysics, :tracers,
                             :fog_dew)
"Stages composed into the hooks of `step!`."
const DYNAMICS_STAGES = (:body_force, :post_predictor)
"Stages after the dynamics, in order."
const POST_DYNAMICS_STAGES = (:surface_update, :column_tendency, :nest_feedback, :sanity, :output)
"Every stage name, in execution order."
const ALL_STAGES = (PRE_DYNAMICS_STAGES..., DYNAMICS_STAGES..., POST_DYNAMICS_STAGES...)

"Assemble `case.schedule` from the enabled couplings' `stages`, and the dynamics hooks."
function build_schedule!(case)
    empty!(case.schedule)
    for S in ALL_STAGES
        case.schedule[S] = ScheduleEntry[]
    end
    for c in case.couplings, (S, order, cad) in stages(typeof(c))
        push!(case.schedule[S], ScheduleEntry(c, order, resolve(cad, case.cfg)))
    end
    for S in ALL_STAGES
        sort!(case.schedule[S]; by = e -> e.order)   # stable: ties keep registry order
    end
    for S in DYNAMICS_STAGES, e in case.schedule[S]
        e.cadence.period > 0 && error("coupling :$(coupling_name(e.coupling)): stage :$S runs inside step! and must use every-step cadence")
    end
    bf = case.schedule[:body_force]
    pp = case.schedule[:post_predictor]
    # a fixed-gradient θ top: capture the top-face flux of every dynamics
    # stage (core/top_boundary.jl) before the body-force entries run
    # an open case: the open faces' heat and the θ relaxation toward the
    # boundary data (core/open_boundary.jl, #262), likewise
    tf = case.top
    ob = case.open
    case.body_force = (isempty(bf) && tf === nothing && ob === nothing) ? nothing :
        (s, Δt) -> (tf === nothing || capture_top_flux!(tf, s, get(case.step_kwargs, :sgs, nothing));
                    ob === nothing || capture_boundary_heat!(ob, s, get(case.step_kwargs, :sgs, nothing));
                    for e in bf; timed_run!(case, e, :body_force, Δt); end; nothing)
    case.post_predictor = isempty(pp) ? nothing :
        (s, Δt) -> (for e in pp; timed_run!(case, e, :post_predictor, Δt); end; nothing)
    case.step_kwargs = merge_keywords("step!", (coupling_name(c) => step_options(c, case)
                                                for c in case.couplings))
    empty!(case.timing)
    return case
end

# Every entry is an air-heat attribution context (core/air_heat_ledger.jl):
# the ledger measures the heat of the air around it and books the change
# under (stage, coupling). The measuring is timed as (:ledger, :air_heat),
# not as the coupling's.
function timed_run!(case, e::ScheduleEntry, S::Symbol, dt)
    name = coupling_name(e.coupling)
    hl = case.ledger.heat
    th = hl.active ? time_ns() : UInt64(0)
    hl.active && heat_open!(hl, S, name, case.sim.θ)
    t0 = time_ns()
    run_stage!(e.coupling, case, Val(S), dt)
    t1 = time_ns()
    key = (S, name)
    case.timing[key] = get(case.timing, key, 0.0) + (t1 - t0) * 1e-9
    if hl.active
        heat_close!(hl, case.sim.θ)
        _time_air_heat!(case, (t0 - th) + (time_ns() - t1))
    end
    return nothing
end

_time_air_heat!(case, ns) =
    (case.timing[(:ledger, :air_heat)] = get(case.timing, (:ledger, :air_heat), 0.0) + ns * 1e-9)

function run_entries!(case, S::Symbol, dt, n, t_old, t_new)
    for e in case.schedule[S]
        fires(e.cadence, n, t_old, t_new) && timed_run!(case, e, S, dt)
    end
    return nothing
end

"The enabled provider of the `:timestepper` slot, or `nothing` (plain `step!` at `cfg.dt`)."
function timestepper(case)
    for c in case.couplings
        slot(typeof(c)) === :timestepper && return c
    end
    return nothing
end

"""
    dynamics_step!(case, dt)

One `step!` of the dynamical core over `dt` with the composed
`:body_force` / `:post_predictor` hooks and the numerics slots'
`case.step_kwargs`. The fixed time stepper (and the contract default of
`advance_dynamics!`). Books the heat a fixed-gradient θ top and the open
faces of a `lateral = :open` case carried ([`book_dynamics_heat!`](@ref));
a stepper that takes the step in pieces books each accepted piece itself.
"""
dynamics_step!(case, dt) =
    (step!(case.sim, dt; body_force! = case.body_force, post_predictor! = case.post_predictor,
           case.step_kwargs...); book_dynamics_heat!(case, dt); nothing)

"""
    step_dt(case) -> dt

The length (s) of the next step: `next_dt` of the `:timestepper`
provider (`cfg.dt` for a fixed step; the CFL-limited step of the adaptive
provider). A variable step is then shortened by every coupling's
[`limit_dt`](@ref) (the nest: the child lands on the end of the parent's
step, #265).
"""
function step_dt(case)
    ts = timestepper(case)
    ts === nothing && return case.cfg.dt
    dt = next_dt(ts, case)
    variable_dt(typeof(ts)) || return dt
    for c in case.couplings
        d = limit_dt(c, case, dt)
        # the coupling that shortened the step is what limited it
        d < dt && hasproperty(ts, :stats) && (ts.stats.last_limiter = coupling_name(c))
        dt = d
    end
    return dt
end

"""
    step_case!(case) -> Bool
    step_case!(case, dt) -> Bool

Advance the case by one step through every stage: of `step_dt(case)`
seconds, or of the `dt` given. Returns `false` if the `:sanity` stage
found non-finite state (the step is complete, but the run should stop).
The dynamics is advanced by the `:timestepper` provider's
`advance_dynamics!` (default: one `step!`).

With a variable-step provider (`variable_dt`, the default `:adaptive`)
a `dt` above the provider's [`stable_dt`](@ref) is an `ArgumentError`,
thrown before any stage runs. The transport, microphysics and surface
stages take the whole `dt` in one go, and the dynamics' retry halves a
step at most `max_retries` times, so such a step is not made safe by
splitting it: `step_case!(case, cfg.dt)` on the default case took
2 s steps while the stable step fell from 0.47 s to 0.09 s; the guard
of `step_adaptive!` (finite, divergence-free, |u| ≤ 50 m/s) passed the
first nine without a retry while |v| grew from 7.8 to 39 m/s, and the
run failed at step 16 with `θ` spanning 229–360 K (#392). Call `step_case!(case)`, or `run_case!`, which runs to a time.
"""
step_case!(case) = _step_case!(case, step_dt(case))

function step_case!(case, dt::Real)
    ts = timestepper(case)
    if ts !== nothing && variable_dt(typeof(ts))
        lim = stable_dt(ts, case)
        dt <= lim * (1 + 1e-9) || throw(ArgumentError(
            @sprintf("step_case!: dt = %.4g s exceeds the stable step %.4g s of the :%s time stepper", dt, lim,
                     provider(typeof(ts))) *
            " (CFL / diffusion / buoyancy limit of the current flow); call step_case!(case) " *
            "to take the step it picks, or run_case!, which runs to a time (#392)"))
    end
    return _step_case!(case, dt)
end

# One step of `dt`, unchecked: `run_case!` passes the step `step_dt` picked.
function _step_case!(case, dt::Real)
    sim, clk = case.sim, case.clock
    clk.n += 1
    n = clk.n
    t_old = sim.t
    t_new = t_old + dt
    ledger_begin_step!(case)
    # :clock (core)
    clk.hour = hour_of_day(case.cfg, sim.t)
    run_entries!(case, :clock, dt, n, t_old, t_new)
    for S in PRE_DYNAMICS_STAGES[2:end]
        run_entries!(case, S, dt, n, t_old, t_new)
    end
    # the dynamics: one air-heat context, split into the terms booked
    # inside it (add_heat!), (:dynamics, :transport) and (:dynamics, :redistribution)
    hl = case.ledger.heat
    th = time_ns()
    heat_open!(hl, :dynamics, :step!, sim.θ; split = true)
    t0 = time_ns()
    ts = timestepper(case)
    ts === nothing ? dynamics_step!(case, dt) : advance_dynamics!(ts, case, dt)
    t1 = time_ns()
    case.timing[(:dynamics, :step!)] = get(case.timing, (:dynamics, :step!), 0.0) + (t1 - t0) * 1e-9
    heat_close!(hl, sim.θ)
    _time_air_heat!(case, (t0 - th) + (time_ns() - t1))
    run_entries!(case, :surface_update, dt, n, t_old, t_new)
    run_entries!(case, :column_tendency, dt, n, t_old, t_new)
    run_entries!(case, :nest_feedback, dt, n, t_old, t_new)
    # :sanity (coupling guards first, then the core finiteness check)
    run_entries!(case, :sanity, dt, n, t_old, t_new)
    if clk.blowup || any(!isfinite, interior_view(sim.u)) || any(!isfinite, case.bus.Ts)
        @error("blowup at step $n", subsystem = :berlin, domain = current_domain(),
               event = :blowup, step = n)
        clk.blowup = true
        return false
    end
    ledger_end_step!(case, dt)
    # :output (core record + log line, then coupling entries)
    fires(resolve(Cadence(:output), case.cfg), n, t_old, t_new) && write_output!(case)
    run_entries!(case, :output, dt, n, t_old, t_new)
    return true
end

"""
    last_step_dt(dt, remaining) -> dt

The step to take when `remaining` seconds are left: `dt`, or all of
`remaining` when it is no longer than `dt`, or half of it when it is
shorter than `2dt` (so the run does not end on a sliver of a step).
"""
last_step_dt(dt, remaining) = remaining <= dt ? remaining : remaining < 2dt ? remaining / 2 : dt

"""
    run_case!(case; nsteps = case.cfg.nsteps, monitor = nothing, callback = nothing,
              t_end = case.sim.t + nsteps * case.cfg.dt, resume = false) -> case

Open the output (unless `cfg.output_path === nothing`), run `nsteps`
steps (stopping early on a blow-up), close the output and log the
summary: the run's start (`event = :run_start`) and
[`log_run_end`](@ref) (the run, the stage timing and the ledgers), all
structured `Info` events (`subsystem = :berlin`). With a `monitor` (an
`Aeolus.IO.RunMonitor`, the `berlin` run driver's) every step runs under
it (`monitored_step!`: the diagnostics row with the step's dt, its
[`step_limiter`](@ref) and retries, the progress line, the blowup guard);
`callback(case, n)` is called after step `n`.

The loop sets the run's horizon on the clock (`t_stop`, `stop_margin`)
and calls every coupling's [`end_run!`](@ref) after it. With it a
one-way nest steps the parent's next window on its own task while the
child steps (SC-c, #403): between steps the nest parent may be busy, so
a `callback` must not read it (the child is the case's own); after
`run_case!` returns it is free.

With a variable-step `:timestepper` provider (`variable_dt`, e.g.
`:adaptive`) `nsteps · cfg.dt` is instead the simulated time to reach:
each step takes `step_dt(case)`, the last one shortened to land on it.

A run resumed from a checkpoint (#269) passes the remaining `nsteps`,
the original run's end `t_end` (the horizon the nest's parent steps
ahead to, so the resumed run starts them exactly where the uninterrupted
run did) and `resume = true` (the NetCDF output is reopened, not
recreated).
"""
function run_case!(case; nsteps::Integer = case.cfg.nsteps, monitor = nothing, callback = nothing,
                   t_end::Real = case.sim.t + nsteps * case.cfg.dt, resume::Bool = false)
    cfg = case.cfg
    cfg.output_path === nothing || open_output!(case; resume)
    P = cadence_period(cfg, :output)
    ts = timestepper(case)
    variable = ts !== nothing && variable_dt(typeof(ts))
    @info(variable ? "run: $(nsteps * cfg.dt) s with a variable step (:$(provider(typeof(ts))))" :
                     "run: $nsteps steps of $(cfg.dt) s",
          subsystem = :berlin, domain = current_domain(), event = :run_start,
          nsteps, t_end = Float64(t_end),
          dt = cfg.dt, variable_dt = variable,
          records = floor(Int, nsteps * cfg.dt / P))
    case.clock.t_wall0 = time()
    n0 = case.clock.n
    function one_step(dt)
        if monitor === nothing
            ok = _step_case!(case, dt)
        else
            # the row gets the step's dt, what limited it and its retries
            r0 = step_retries(case)
            ok, _, _ = Aeolus.IO.monitored_step!(monitor, case.sim; step = case.clock.n + 1,
                                                 limiter = step_limiter(case)) do
                (_step_case!(case, dt), Float64(dt), step_retries(case) - r0)
            end
        end
        ok && callback !== nothing && callback(case, case.clock.n)
        return ok
    end
    # the horizon: a step starts at t exactly when t_stop - t > stop_margin
    # (the fixed loop's steps start on multiples of dt, so half a step is margin)
    clk = case.clock
    clk.t_stop = Float64(t_end)
    clk.stop_margin = variable ? 1e-9 * cfg.dt : 0.5 * cfg.dt
    try
        if variable
            t_end = clk.t_stop
            while t_end - case.sim.t > 1e-9 * cfg.dt
                one_step(last_step_dt(step_dt(case), t_end - case.sim.t)) || break
            end
        else
            for _ in 1:nsteps
                one_step(cfg.dt) || break
            end
        end
    finally
        clk.t_stop = NaN
        foreach(c -> end_run!(c, case), case.couplings)
    end
    nsteps = case.clock.n - n0
    t_wall = time() - case.clock.t_wall0
    close_output!(case)
    log_run_end(case; nsteps, t_wall)
    return case
end
