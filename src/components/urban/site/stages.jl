# One step of the urban site: the core's stage pipeline
# (`Aeolus.Components`, ONE-a #450; ONE-e #454; ONE-k #474), taken by
# `step!(model)`.
#
# The site is a `Domain` of the pipeline: its components are the site
# itself (`UrbanSite`: its shared state, the clock and the captures of the
# fixed-gradient top and the open faces; it takes the core step) and every
# enabled coupling (`UrbanEntry`). A
# coupling's stages are the pipeline's (`Components.PIPELINE`), each with the
# cadence and the place in its stage (`order`) the coupling declares, so
# the pipeline runs them in the order of the urban stage loop it replaces,
# bit for bit:
#
#   :clock           local hour of day (the case)
#   :nest_sync       the parent run advanced to cover the child step, the
#                    child's nest faces pointed at the new parent window (#265)
#   :radiation       sun, radiosity, buildings over the radiation interval
#   :sw_modifiers    tree shade, cloud shadow
#   :surface_flux    surface → air fluxes; builds bus.sensible
#   :sources         buildings, traffic, glazing (anthropogenic heat)
#   :chemistry       reactive species
#   :transport       advection of the registered tracers
#   :microphysics    cloud / rain, then dew on surfaces and fog in the air
#   :tracers         passive tracers with their own schemes
#   -- the core's step (the :timestepper provider) with the :tendencies,
#      :body_force and :post_predictor entries called from its hooks --
#   :surface_update  surface and soil prognostics
#   :column_tendency column (radiative) tendencies of θ
#   :nest_feedback   the parent window closed when the child reached its end
#                    (one-way: clock and statistics; two-way, #267: the feedback)
#   :budgets         (the blow-up check, then the budgets close)
#   :checks, :output NetCDF record and log line (the case), then the entries

"""
    pipeline_stages(C, cfg) -> Tuple{Vararg{Stage}}

The pipeline stages of coupling type `C` under `cfg`: one
`Components.Stage` per entry of `stages(C)`, its `Cadence` resolved to
seconds (`every`, `first`) and its integer order its `order`.
"""
function pipeline_stages(C, cfg)
    return Tuple(begin
                     cad = resolve(c, cfg)
                     S in ALL_STAGES || error("coupling :$(component_name(C)): unknown stage :$S")
                     Stage(S; every = max(cad.period, 0.0), first = cad.first, order = order)
                 end for (S, order, c) in stages(C))
end

"""
    UrbanEntry

A coupling of the case as a component of its `Domain` (`Aeolus.Components`):
the coupling `c`, its name and its [`pipeline_stages`](@ref); every stage
entry is timed into `case.timing[(stage, coupling)]`. One concrete type for
every coupling (the call into the coupling is dynamic, as in the urban stage
loop it replaced), so a case's domain is not compiled anew for every
selection of couplings.
"""
struct UrbanEntry <: AbstractComponent
    c::UrbanComponent
    name::Symbol
    stages::Tuple{Vararg{Stage}}
end
UrbanEntry(c::UrbanComponent, st) = UrbanEntry(c, component_name(typeof(c)), Tuple(st))
Components.component_name(e::UrbanEntry) = e.name
Components.section(e::UrbanEntry) = string(e.name)
Components.stages(e::UrbanEntry) = e.stages
Components.budget!(e::UrbanEntry, d::Domain, b) = budget!(e.c, urban_case(d), b)
function Components.run_stage!(e::UrbanEntry, d::Domain, v::Val{S}, dt) where {S}
    case = urban_case(d)
    t0 = time_ns()
    run_stage!(e.c, case, v, dt)
    key = (S, e.name)
    case.timing[key] = get(case.timing, key, 0.0) + (time_ns() - t0) * 1e-9
    return nothing
end

# The site's own component of its `Domain` is the [`UrbanSite`](@ref)
# itself (site/case.jl): in `:clock` (first) the step count, the local
# hour and the snapshot of the surface temperature and the water in the
# air the budgets close against; in `:tendencies` (every dynamics stage,
# before the `:body_force` entries) the heat of a fixed-gradient θ top and
# of the open faces, and the θ relaxation of the open faces
# (`site/top_boundary.jl`, `site/open_boundary.jl`); in `:budgets` the
# sanity check; in `:output` (first) the site's record. It takes the core
# step (its `:timestepper` component's).
Components.component_name(::Type{<:UrbanSite}) = :urban
Components.section(::Type{<:UrbanSite}) = "urban"
Components.stages(case::UrbanSite) =
    (Stage(:clock; order = typemin(Int)), Stage(:tendencies), Stage(:budgets),
     Stage(:output; every = max(cadence_period(case.cfg, :output), 0.0), order = typemin(Int)))
function Components.run_stage!(case::UrbanSite, d::Domain, ::Val{:clock}, dt)
    case.clock.n += 1
    l = case.budgets
    size(l.Ts_start) == size(case.bus.Ts) ? copyto!(l.Ts_start, case.bus.Ts) : (l.Ts_start = copy(case.bus.Ts))
    l.W_start = air_water(case)
    case.clock.hour = hour_of_day(case.cfg, case.sim.t)
    return nothing
end
# the open faces' θ relaxation is booked as (:dynamics, :relaxation) after
# the step (book_dynamics_heat!), not measured as an entry of its own
Components.attributed(::UrbanSite, ::Val{:tendencies}) = false
function Components.run_stage!(case::UrbanSite, d::Domain, ::Val{:tendencies}, dt)
    sgs = get(case.step_kwargs, :sgs, nothing)
    case.top === nothing || capture_top_flux!(case.top, case.sim, sgs)
    case.open === nothing || capture_boundary_heat!(case.open, case.sim, sgs)
    return nothing
end
# The sanity check, after the stages that follow the core (and before the
# budgets close): a non-finite wind or surface temperature, or a blowup a
# coupling flagged, marks the site blown up and ends the step with an
# error (the run monitor diagnoses it and writes the crash checkpoint).
function Components.run_stage!(case::UrbanSite, d::Domain, ::Val{:budgets}, dt)
    clk = case.clock
    if clk.blowup || any(!isfinite, interior_view(case.sim.u)) || any(!isfinite, case.bus.Ts)
        clk.blowup = true
        @error("blowup at step $(clk.n)", subsystem = :urban, domain = current_domain(),
               event = :blowup, step = clk.n)
        error("the urban site blew up at step $(clk.n) (t = $(case.sim.t) s)")
    end
    isnan(clk.t_first) && !isnan(clk.t_stop) && (clk.t_first = time())
    return nothing
end
# an output frame: its log events (the record is the run's writer's,
# which writes the site's catalogue, Components.outputs)
Components.run_stage!(case::UrbanSite, d::Domain, ::Val{:output}, dt) =
    (case.clock.nrecord += 1; log_frame(case); nothing)
# reading the state only: no term of the budgets
Components.attributed(::UrbanSite, ::Val{:budgets}) = false
Components.attributed(::UrbanSite, ::Val{:output}) = false

# The case's time stepper takes the core step (ONE-k, #474): the
# `:timestepper` provider's `advance_dynamics!` (default one `step!`), with
# its step length ([`step_dt`](@ref)) and its check of an explicit step.
Components.takes_core_step(::UrbanSite) = true
function Components.advance_core!(case::UrbanSite, d::Domain, dt)
    t0 = time_ns()
    ts = timestepper(case)
    ts === nothing ? dynamics_step!(case, dt) : advance_dynamics!(ts, case, dt)
    case.timing[(:dynamics, :step!)] = get(case.timing, (:dynamics, :step!), 0.0) + (time_ns() - t0) * 1e-9
    return nothing
end
# the step the time stepper picks; within a run (its horizon set) a
# variable step lands on the run's end
function Components.next_dt(case::UrbanSite, d::Domain)
    dt = step_dt(case)
    clk = case.clock
    isnan(clk.t_stop) && return dt
    ts = timestepper(case)
    return ts !== nothing && variable_dt(typeof(ts)) ? Components.last_step_dt(dt, clk.t_stop - case.sim.t) : dt
end
Components.check_dt(case::UrbanSite, d::Domain, dt) = check_step_dt(case, dt)

# the water the air gained over the step (the budgets close after :budgets)
Components.budget!(case::UrbanSite, d::Domain, b) = (b.water_air += air_water(case) - b.W_start; nothing)

"""
    urban_check_names() -> Vector{String}

The checks an urban site publishes (ONE-d, #453; `[checks]` of the
model enables them): `urban_budgets`, the closure of the case's budgets
against [`BUDGET_TOLERANCES`](@ref) (the verdict of [`run_results`](@ref)),
and `<coupling>_checks` per urban component, its `smoke_check`s.
"""
urban_check_names() = vcat(["urban_budgets"], ["$(component_name(C))_checks" for C in URBAN_COMPONENTS])

# The checks of the case (see urban_check_names): `urban_budgets` is the
# largest closure measure relative to its tolerance (≤ 1 passes); a
# coupling's is the number of its smoke checks that fail.
function Components.checks(case::UrbanSite)
    budgets = Check("urban_budgets";
        doc = "the case's budgets closed: max(surface closure / $(BUDGET_TOLERANCES.surface) W m-2, " *
              "|water residual| / $(BUDGET_TOLERANCES.water) kg m-2, |air-heat closure| / $(BUDGET_TOLERANCES.air_heat) W m-2), " *
              "and nothing changed θ unattributed (Inf)",
        tolerance = 1.0,
        evaluate = _ -> begin
            r = run_results(case)["budgets"]
            tol = BUDGET_TOLERANCES
            v = max(r["surface_max_abs_closure"] / tol.surface, abs(r["water_residual"]) / tol.water,
                    abs(r["air_heat_closure"]) / tol.air_heat)
            (r["unattributed"] ? Inf : v,
             Dict{String,Any}("surface_max_abs_closure" => r["surface_max_abs_closure"],
                              "water_residual" => r["water_residual"], "air_heat_closure" => r["air_heat_closure"]))
        end)
    cs = Check[budgets]
    for c in case.couplings
        n = component_name(typeof(c))
        push!(cs, Check("$(n)_checks"; doc = "the smoke checks of the urban component $n that fail",
                        tolerance = 0.0,
                        evaluate = _ -> begin
                            r = smoke_check(c, case)
                            failed = String[k for (k, ok) in r if !ok]
                            (Float64(length(failed)), Dict{String,Any}("checks" => length(r), "failed" => failed))
                        end))
    end
    return cs
end

_urban_site(::Tuple{}) = error("the domain has no urban site")
_urban_site(cs::Tuple) = first(cs) isa UrbanSite ? first(cs) : _urban_site(Base.tail(cs))
"The [`UrbanSite`](@ref) of a `Domain` built by [`build_case`](@ref)."
urban_case(d::Domain) = _urban_site(d.components)

"""
    build_domain!(case) -> case

The case as a `Domain` of the core's pipeline (`case.domain`): its
[`UrbanSite`](@ref) itself and an [`UrbanEntry`](@ref) per enabled coupling, in
the pipeline's order (`Components.order_components`), with the case's
budgets as the domain's (the heat of the air, attributed to every stage
entry); `case.body_force` and `case.post_predictor` are the pipeline's
hooks of the core step, and `case.step_kwargs` the numerics slots' step
keywords.
"""
function build_domain!(case)
    cfg = case.cfg
    comps = (case, (UrbanEntry(c, pipeline_stages(typeof(c), cfg)) for c in case.couplings)...)
    d = Domain(case.sim; id = string(cfg.name), components = comps, validate = false)
    db = DomainBudgets(case.budgets, (:energy,), Dict(:energy => case.sim.θ), Dict(:energy => case.sim.tθ);
                       nsub = 1, hour = t -> hour_of_day(cfg, t))
    case.domain = Domain(d.id, d.state, d.scheme, d.cfl, d.drivers, d.components, db)
    case.model = Model(case.domain)
    h = Components.core_hooks(case.domain)
    case.body_force, case.post_predictor = h.body_force!, h.post_predictor!
    case.step_kwargs = merge_keywords("step!", (component_name(c) => step_options(c, case)
                                                for c in case.couplings))
    empty!(case.timing)
    return case
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
        d < dt && hasproperty(ts, :stats) && (ts.stats.last_limiter = component_name(c))
        dt = d
    end
    return dt
end

"""
    check_step_dt(case, dt)

An explicit step `dt` of the case (`step!(model, dt)`) with a
variable-step `:timestepper` provider (`variable_dt`, the default
`:adaptive`) above the provider's [`stable_dt`](@ref) is an
`ArgumentError`, thrown before any stage runs. The transport,
microphysics and surface stages take the whole `dt` in one go, and the
dynamics' retry halves a step at most `max_retries` times, so such a step
is not made safe by splitting it: steps of `cfg.dt` on the default case
were 2 s while the stable step fell from 0.47 s to 0.09 s; the guard of
`step_adaptive!` (finite, divergence-free, |u| ≤ 50 m/s) passed the first
nine without a retry while |v| grew from 7.8 to 39 m/s, and the run
failed at step 16 with `θ` spanning 229–360 K (#392). `step!(model)`
takes the step the provider picks.
"""
function check_step_dt(case, dt::Real)
    ts = timestepper(case)
    if ts !== nothing && variable_dt(typeof(ts))
        lim = stable_dt(ts, case)
        dt <= lim * (1 + 1e-9) || throw(ArgumentError(
            @sprintf("step!: dt = %.4g s exceeds the stable step %.4g s of the :%s time stepper", dt, lim,
                     provider(typeof(ts))) *
            " (CFL / diffusion / buoyancy limit of the current flow); call step!(model) " *
            "to take the step it picks, which runs to a time (#392)"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# the run (ONE-k, #474): the run loop of `aeolus run` (the core's) asks the
# site for its horizon, its steps and its results
# ---------------------------------------------------------------------------

"""
    Components.run_begin!(case::UrbanSite, d, t_end)

The run's horizon on the clock (`t_stop = t_end`; a step starts at `t`
exactly when `t_stop - t > stop_margin`, half a step for a fixed step,
`1e-9 dt` for a variable one, which lands on `t_end`), the run's start
for [`run_results`](@ref) and its `:run_start` event. With the horizon a
one-way nest steps the parent's next window on its own task while the
child steps (SC-c, #403).

With a variable-step `:timestepper` provider (`variable_dt`, e.g.
`:adaptive`) `nsteps · cfg.dt` is the simulated time to reach: each step
takes `step_dt(case)`, the last one shortened to land on it.
"""
function Components.run_begin!(case::UrbanSite, d::Domain, t_end)
    cfg, clk = case.cfg, case.clock
    ts = timestepper(case)
    variable = ts !== nothing && variable_dt(typeof(ts))
    clk.t_stop = Float64(t_end)
    clk.stop_margin = variable ? 1e-9 * cfg.dt : 0.5 * cfg.dt
    clk.t_wall0 = time()
    clk.n0, clk.t0, clk.t_first = clk.n, case.sim.t, NaN
    clk.θ_start = mean_θ_below(case, RESULTS_LAYER)
    P = cadence_period(cfg, :output)
    span = t_end - case.sim.t
    @info(variable ? "run: $(span) s with a variable step (:$(provider(typeof(ts))))" :
                     "run: $(round(Int, span / cfg.dt)) steps of $(cfg.dt) s",
          subsystem = :urban, domain = current_domain(), event = :run_start,
          t_end = Float64(t_end), dt = cfg.dt, variable_dt = variable,
          records = floor(Int, span / P))
    return nothing
end

"Whether the run loop takes another step: [`step_will_run`](@ref) at the site's time."
Components.run_continues(case::UrbanSite, d::Domain) = step_will_run(case.clock, case.sim.t)

"What limited the step the site picked last, and the time stepper's retries so far."
Components.step_report(case::UrbanSite, d::Domain) = (step_limiter(case), step_retries(case))

"""
    Components.run_end!(case::UrbanSite, d)

After the run's last step (also when it stops with an error): the
horizon is cleared, every coupling's [`end_run!`](@ref) runs (the nest
waits for a parent step it started ahead, SC-c #403), and the end-of-run
events are logged ([`log_run_end`](@ref)).
"""
function Components.run_end!(case::UrbanSite, d::Domain)
    clk = case.clock
    clk.t_stop = NaN
    foreach(c -> end_run!(c, case), case.couplings)
    log_run_end(case; nsteps = clk.n - clk.n0, t_wall = time() - clk.t_wall0)
    return nothing
end

"The run's diagnostics: [`case_diagnostics`](@ref) and [`case_profiles`](@ref)."
Components.run_diagnostics(case::UrbanSite, d::Domain) = (case_diagnostics(case), case_profiles(case))

"""
    Components.run_results(case::UrbanSite, d) -> Dict

What a finished run of the site writes to `results.toml`:
[`run_results`](@ref) of the run (the budgets' closure, the mean θ below
[`RESULTS_LAYER`](@ref) at its start and end, its timing), with the
`:results` event and a warning when the budgets do not close.
"""
function Components.run_results(case::UrbanSite, d::Domain)
    clk = case.clock
    wall = time() - clk.t_wall0
    res = run_results(case; θ_start = clk.θ_start, wall = wall, wall_first_step = clk.t_first - clk.t_wall0,
                      simulated = case.sim.t - clk.t0, steps = clk.n - clk.n0)
    led = res["budgets"]
    @info("budgets " * (led["closed"] ? "closed" : "NOT closed"), subsystem = :urban, domain = current_domain(),
          event = :results, surface = led["surface_max_abs_closure"], water = led["water_residual"],
          air_heat = led["air_heat_closure"])
    led["closed"] || @warn("the site's budgets do not close within their tolerances (results.toml)",
                           subsystem = :urban, domain = current_domain(), event = :budgets_open)
    return res
end

"""
    Components.state_arrays(case::UrbanSite, d) -> NamedTuple

The site's prognostic arrays besides the core's fields: every tracer,
the surface temperature `Ts` and, with the Aeolus nest parent, the
parent's fields ([`case_state`](@ref) without the child's `u, v, w, p, θ`).
"""
Components.state_arrays(case::UrbanSite, d::Domain) =
    Base.structdiff(case_state(case), NamedTuple{(:u, :v, :w, :p, :θ)})
