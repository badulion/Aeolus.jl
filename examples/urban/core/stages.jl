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
    case.body_force = isempty(bf) ? nothing :
        (s, Δt) -> (for e in bf; run_stage!(e.coupling, case, Val(:body_force), Δt); end; nothing)
    case.post_predictor = isempty(pp) ? nothing :
        (s, Δt) -> (for e in pp; run_stage!(e.coupling, case, Val(:post_predictor), Δt); end; nothing)
    return case
end

function run_entries!(case, S::Symbol, dt, n, t_old, t_new)
    for e in case.schedule[S]
        fires(e.cadence, n, t_old, t_new) && run_stage!(e.coupling, case, Val(S), dt)
    end
    return nothing
end

"""
    step_case!(case, dt = case.cfg.dt) -> Bool

Advance the case by one step of `dt` seconds through every stage.
Returns `false` if the `:sanity` stage found non-finite state (the step
is complete, but the run should stop).
"""
function step_case!(case, dt::Real = case.cfg.dt)
    sim, clk = case.sim, case.clock
    clk.n += 1
    n = clk.n
    t_old = sim.t
    t_new = t_old + dt
    # :clock (core)
    clk.hour = hour_of_day(case.cfg, sim.t)
    run_entries!(case, :clock, dt, n, t_old, t_new)
    for S in PRE_DYNAMICS_STAGES[2:end]
        run_entries!(case, S, dt, n, t_old, t_new)
    end
    step!(sim, dt; body_force! = case.body_force, post_predictor! = case.post_predictor)
    run_entries!(case, :surface_update, dt, n, t_old, t_new)
    run_entries!(case, :column_tendency, dt, n, t_old, t_new)
    # :sanity (coupling guards first, then the core finiteness check)
    run_entries!(case, :sanity, dt, n, t_old, t_new)
    if clk.blowup || any(!isfinite, interior_view(sim.u)) || any(!isfinite, case.bus.Ts)
        case.cfg.verbose && @printf(case.io, "!! blowup at step %d\n", n)
        clk.blowup = true
        return false
    end
    # :output (core record + log line, then coupling entries)
    fires(resolve(Cadence(:output), case.cfg), n, t_old, t_new) && write_output!(case)
    run_entries!(case, :output, dt, n, t_old, t_new)
    return true
end

"""
    run_case!(case; nsteps = case.cfg.nsteps) -> case

Open the output (unless `cfg.output_path === nothing`), run `nsteps`
steps (stopping early on a blow-up), close the output and print the
summary.
"""
function run_case!(case; nsteps::Integer = case.cfg.nsteps)
    cfg, io = case.cfg, case.io
    cfg.output_path === nothing || open_output!(case)
    P = cadence_period(cfg, :output)
    cfg.verbose && @printf(io, "Run: %d steps, %d output records, dt=%.0fs\n",
                           nsteps, floor(Int, nsteps * cfg.dt / P), cfg.dt)
    case.clock.t_wall0 = time()
    for _ in 1:nsteps
        step_case!(case, cfg.dt) || break
    end
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
    end
    return case
end
