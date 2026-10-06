# The single ordered step loop.
#
# One step of the Berlin case is this fixed sequence of stages. Within a
# stage the entries run in the integer `order` their couplings declare
# (ties: registry order). The order reproduces the pre-refactor script
# exactly — floating-point order included — so the refactor is
# bit-for-bit.
#
#   :clock           local hour of day (core)
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
#   :sanity          guards (e.g. clamps), then the core blow-up check
#   :output          NetCDF record and log line (core)

"Stages before the dynamics, in order."
const PRE_DYNAMICS_STAGES = (:clock, :radiation, :sw_modifiers, :surface_flux, :anthropogenic,
                             :chemistry, :transport, :microphysics, :tracers, :fog_dew)
"Stages composed into the hooks of `step!`."
const DYNAMICS_STAGES = (:body_force, :post_predictor)
"Stages after the dynamics, in order."
const POST_DYNAMICS_STAGES = (:surface_update, :column_tendency, :sanity, :output)
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
    tf = case.top
    case.body_force = (isempty(bf) && tf === nothing) ? nothing :
        (s, Δt) -> (tf === nothing || capture_top_flux!(tf, s, get(case.step_kwargs, :sgs, nothing));
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

"""
    timing_report(case) -> String

Wall time spent in each (stage, coupling) entry and in `step!` (the
dynamics, which includes the `:body_force` / `:post_predictor` entries),
in seconds and ms per step, largest first.
"""
function timing_report(case)
    n = max(case.clock.n, 1)
    io = IOBuffer()
    rows = sort!(collect(case.timing); by = last, rev = true)
    @printf(io, "Wall time by stage entry over %d steps (s, ms/step):\n", n)
    for ((S, name), t) in rows
        @printf(io, "  %-16s %-20s %8.2f %8.2f\n", S, name, t, 1e3 * t / n)
    end
    return String(take!(io))
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
`advance_dynamics!`). Books the heat a fixed-gradient θ top carried
([`book_top_flux!`](@ref)); a stepper that takes the step in pieces books
each accepted piece itself.
"""
dynamics_step!(case, dt) =
    (step!(case.sim, dt; body_force! = case.body_force, post_predictor! = case.post_predictor,
           case.step_kwargs...); book_top_flux!(case, dt); nothing)

"""
    step_dt(case) -> dt

The length (s) of the next step: `next_dt` of the `:timestepper`
provider (`cfg.dt` for a fixed step; the CFL-limited step of the adaptive
provider).
"""
step_dt(case) = (ts = timestepper(case); ts === nothing ? case.cfg.dt : next_dt(ts, case))

"""
    step_case!(case, dt = step_dt(case)) -> Bool

Advance the case by one step of `dt` seconds through every stage.
Returns `false` if the `:sanity` stage found non-finite state (the step
is complete, but the run should stop). The dynamics is advanced by the
`:timestepper` provider's `advance_dynamics!` (default: one `step!`).
"""
function step_case!(case, dt::Real = step_dt(case))
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
    # :sanity (coupling guards first, then the core finiteness check)
    run_entries!(case, :sanity, dt, n, t_old, t_new)
    if clk.blowup || any(!isfinite, interior_view(sim.u)) || any(!isfinite, case.bus.Ts)
        case.cfg.verbose && @printf(case.io, "!! blowup at step %d\n", n)
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
    run_case!(case; nsteps = case.cfg.nsteps) -> case

Open the output (unless `cfg.output_path === nothing`), run `nsteps`
steps (stopping early on a blow-up), close the output and print the
summary.

With a variable-step `:timestepper` provider (`variable_dt`, e.g.
`:adaptive`) `nsteps · cfg.dt` is instead the simulated time to reach:
each step takes `step_dt(case)`, the last one shortened to land on it.
"""
function run_case!(case; nsteps::Integer = case.cfg.nsteps)
    cfg, io = case.cfg, case.io
    cfg.output_path === nothing || open_output!(case)
    P = cadence_period(cfg, :output)
    ts = timestepper(case)
    variable = ts !== nothing && variable_dt(typeof(ts))
    if cfg.verbose
        variable ? @printf(io, "Run: %.0f s with a variable step (:%s), %d output records\n",
                           nsteps * cfg.dt, provider(typeof(ts)), floor(Int, nsteps * cfg.dt / P)) :
                   @printf(io, "Run: %d steps, %d output records, dt=%.0fs\n",
                           nsteps, floor(Int, nsteps * cfg.dt / P), cfg.dt)
    end
    case.clock.t_wall0 = time()
    n0 = case.clock.n
    if variable
        t_end = case.sim.t + nsteps * cfg.dt
        while t_end - case.sim.t > 1e-9 * cfg.dt
            step_case!(case, last_step_dt(step_dt(case), t_end - case.sim.t)) || break
        end
    else
        for _ in 1:nsteps
            step_case!(case, cfg.dt) || break
        end
    end
    nsteps = case.clock.n - n0
    t_wall = time() - case.clock.t_wall0
    close_output!(case)
    if cfg.verbose
        if cfg.output_path === nothing
            @printf(io, "Ran %d steps in %.1f s (no output file).\n", nsteps, t_wall)
        else
            @printf(io, "Ran %d steps in %.1f s. Wrote %d time levels to %s (%.1f MB)\n",
                    nsteps, t_wall, case.clock.nrecord, cfg.output_path,
                    filesize(cfg.output_path) / 1e6)
        end
        @printf(io, "Peak ΔT (heat island): %.1f K   max|∇·u| = %.2e\n",
                maximum(case.bus.Ts) - minimum(case.bus.Ts), max_divergence(case.sim))
        print(io, timing_report(case))
        print(io, ledger_report(case))
    end
    return case
end
