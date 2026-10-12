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
# itself (core/case.jl): in `:clock` (first) the step count, the local
# hour and the snapshot of the surface temperature and the water in the
# air the budgets close against; in `:tendencies` (every dynamics stage,
# before the `:body_force` entries) the heat of a fixed-gradient θ top and
# of the open faces, and the θ relaxation of the open faces
# (`core/top_boundary.jl`, `core/open_boundary.jl`); in `:budgets` the
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
# coupling flagged, marks the case blown up; the step's record is then not
# written, and the run stops after the step.
function Components.run_stage!(case::UrbanSite, d::Domain, ::Val{:budgets}, dt)
    clk = case.clock
    if clk.blowup || any(!isfinite, interior_view(case.sim.u)) || any(!isfinite, case.bus.Ts)
        clk.blowup || @error("blowup at step $(clk.n)", subsystem = :urban, domain = current_domain(),
                             event = :blowup, step = clk.n)
        clk.blowup = true
    end
    return nothing
end
# the case's NetCDF record and its log line, before the entries' :output
Components.run_stage!(case::UrbanSite, d::Domain, ::Val{:output}, dt) =
    (case.clock.blowup || write_output!(case); nothing)
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
Components.next_dt(case::UrbanSite, d::Domain) = step_dt(case)
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

"""
    run_case!(case; nsteps = case.cfg.nsteps, monitor = nothing, callback = nothing,
              t_end = case.sim.t + nsteps * case.cfg.dt, resume = false) -> case

Open the output (unless `cfg.output_path === nothing`), run `nsteps`
steps (stopping early on a blow-up), close the output and log the
summary: the run's start (`event = :run_start`) and
[`log_run_end`](@ref) (the run, the stage timing and the ledgers), all
structured `Info` events (`subsystem = :urban`). With a `monitor` (an
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
          subsystem = :urban, domain = current_domain(), event = :run_start,
          nsteps, t_end = Float64(t_end),
          dt = cfg.dt, variable_dt = variable,
          records = floor(Int, nsteps * cfg.dt / P))
    case.clock.t_wall0 = time()
    n0 = case.clock.n
    model = case.model
    # one step of the model (its time stepper takes the core step, ONE-k #474)
    stepped!(dt) = (step!(model, dt; checked = false); !case.clock.blowup)
    function one_step(dt)
        if monitor === nothing
            ok = stepped!(dt)
        else
            # the row gets the step's dt, what limited it and its retries
            r0 = step_retries(case)
            ok, _, _ = Aeolus.IO.monitored_step!(monitor, case.sim; step = case.clock.n + 1,
                                                 limiter = step_limiter(case)) do
                (stepped!(dt), Float64(dt), step_retries(case) - r0)
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
                one_step(Components.pick_dt(model; t_end)) || break
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
