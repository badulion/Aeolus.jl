# The urban site: one configured, built urban domain (formerly the Case).

"""
    Clock

Step counter and time state of a site: `n` (steps taken), `hour`
(local hour of day at the start of the current step), `sun` (the solar
position of the last radiation update), `blowup` (set by the sanity
check of the `:budgets` stage), `nrecord` (output frames logged: the
`:output` cadence), `t_wall0` (wall-clock start of the run) and the
run's horizon, set by the run loop for its length
(`Components.run_begin!`, ONE-k #474): a step starts at time `t` exactly
when `t_stop - t > stop_margin` (`t_stop = NaN`: unknown, a caller steps
the site itself). A coupling that works ahead of the child (the nest's
prefetched parent step, SC-c #403) does so only for steps that will run.
`n0`, `t0`, `θ_start` and `t_first` are the run's start (its step count,
model time, mean θ below [`RESULTS_LAYER`](@ref)) and the wall clock at
the end of its first step, for [`run_results`](@ref).
"""
mutable struct Clock
    n::Int
    hour::Float64
    sun::Any
    blowup::Bool
    nrecord::Int
    t_wall0::Float64
    t_stop::Float64
    stop_margin::Float64
    n0::Int
    t0::Float64
    θ_start::Float64
    t_first::Float64
end

"Whether the run loop will start a step at time `t` (false when it does not know)."
step_will_run(clk::Clock, t::Real) = clk.t_stop - t > clk.stop_margin

"""
    UrbanNest

What the model gives an urban site that is a nested domain (NEST-p3,
#483; `build_case(cfg; nest)`): its parent's `Components.Domain`
`parent`, the nest hierarchy `hierarchy` it joins (`Components.nest_hierarchy`
of the root, completed by `Components.nested_model`), the site's own id
and its parent's (`id`, `parent_id`, Symbols), its `origin` (m, from the
parent's corner) and `refinement`, and its `feedback` (`:one_way` or
`:two_way`).
"""
struct UrbanNest
    parent::Domain
    hierarchy::Any
    id::Symbol
    parent_id::Symbol
    origin::NTuple{2,Float64}
    refinement::NTuple{3,Int}
    feedback::Symbol
end

"""
    UrbanSite <: AbstractComponent

The urban site, a component of its `Domain` (ONE-k, #474; formerly the
`Case`, which stepped the core in its own loop): `cfg`, `mesh`, `sim` (the dynamical core: a
`BoussinesqSimulation`, or the `MassWeightedSimulation` of the `:core`
slot's anelastic / pseudo-incompressible provider), `geom`
([`Geometry`](@ref)), `bus` ([`SurfaceBus`](@ref)), `tracers`
([`TracerRegistry`](@ref)), `budgets` ([`Budgets`](@ref Aeolus.BudgetSystem.Budgets)), the hydrostatic reference state
`ref` ([`CaseReference`](@ref)), the enabled `couplings` (registry
order), the `domain` (the case as a `Domain` of the core's stage pipeline,
[`build_domain!`](@ref)), the `clock`, the pipeline's
`step!` hooks, `timing` (wall seconds per `(stage, coupling)`, the
`wall.*` diagnostics and the `:timing` events of [`log_run_end`](@ref)), `top` — the heat flux through a
fixed-gradient θ top ([`TopHeatFlux`](@ref), `nothing` for a closed
lid) — `open`, the heat of the open faces of a `lateral = :open`
case ([`OpenBoundaryHeat`](@ref), `nothing` for a periodic one), and
`nest`, what the model gives a site that is a nested domain
([`UrbanNest`](@ref): its parent domain and the nest it joins), or
`nothing`.

It is the first component of its domain (section `[urban]`, component
name `:urban`): it runs the site's own work in the pipeline's stages and
takes the domain's core step with its `:timestepper` component
(site/stages.jl). `model` is the `Model` of its domain; `step!(model)`
advances the site.
"""
mutable struct UrbanSite{M, S, G<:Geometry, O<:NamedTuple} <: AbstractComponent
    cfg::O
    mesh::M
    sim::S
    geom::G
    bus::SurfaceBus
    tracers::TracerRegistry
    budgets::Budgets
    ref::CaseReference
    couplings::Vector{UrbanComponent}
    domain::Any
    model::Any                   # Model(domain): what steps the case (ONE-k, #474)
    clock::Clock
    body_force::Union{Nothing,Function}
    post_predictor::Union{Nothing,Function}
    timing::Dict{Tuple{Symbol,Symbol},Float64}
    step_kwargs::NamedTuple      # extra step! keywords from the couplings' step_options
    top::Union{Nothing,TopHeatFlux}   # the fixed-gradient θ top's heat flux (#254)
    open::Union{Nothing,OpenBoundaryHeat}   # the open faces' heat (#262)
    nest::Union{Nothing,UrbanNest}   # the parent domain of a nested site (NEST-p3, #483), or nothing
end

"The enabled coupling called `name`, or `nothing`."
function find_coupling(case::UrbanSite, name::Symbol)
    for c in case.couplings
        component_name(c) === name && return c
    end
    return nothing
end

"""
Local hour of day in [0, 24) at simulation time `t` (s), also before the
start (`t < 0`: the surface spin-up, #519).
"""
hour_of_day(cfg::NamedTuple, t::Real) = mod(cfg.start_h + t / 3600, 24.0)

"""
    case_weekday(cfg) -> 1 … 7

Day of the week of `cfg.date` (1 = Monday … 7 = Sunday, the convention
of `BuildingEnergy.WeeklySchedule`): 29 June 2019 is a Saturday (6).
"""
case_weekday(cfg::NamedTuple) =
    mod(Aeolus.LargeScaleForcing._days_from_civil(cfg.date...) + 3, 7) + 1   # 1970-01-01: Thursday

"""
    solar_position_at(cfg, hour) -> SolarPosition

The sun at local clock `hour` of `cfg.date` over the case anchor. The
solar-position algorithm (`SolarGeometry.solar_time`) takes UTC, so the
local clock is converted with `cfg.utc_offset` (local = UTC + offset; 2 h
in summer, CEST). Before #259 the local hour was passed as UTC and the sun
of the case ran `utc_offset` hours early (solar noon at 11:07 CEST instead
of 13:07 on 29 June 2019). A local hour before `utc_offset` is a UTC hour
of the previous day (a negative hour of `cfg.date`; the Julian day handles
it).
"""
function solar_position_at(cfg::NamedTuple, hour::Real)
    hour = hour - cfg.utc_offset
    hh = floor(Int, hour)
    mi = floor(Int, (hour - hh) * 60)
    se = (hour - hh - mi / 60) * 3600
    y, m, d = cfg.date
    return solar_position(solar_time(y, m, d, hh, mi, se;
                                     latitude = cfg.lat0, longitude = cfg.lon0,
                                     elevation = cfg.elevation))
end

"""
    core_parameters(cfg, mesh) -> NamedTuple

The keyword arguments every dynamical core of the case shares: viscosity,
gravity, Coriolis, Smagorinsky, sponge and the boundary conditions of
[`case_boundary_conditions`](@ref) (none for a nested case, whose nest
faces the `:nesting` coupling's `sim_options` set, #265; otherwise
`cfg.lateral`, #262): for the
default periodic case periodic laterally, walls at the bottom and the
top for the velocity; for θ no flux at the ground (the surface fluxes
enter as tendencies) and [`θ_top_bc`](@ref)`(cfg)` at the top
(`cfg.θ_top`; default the fixed-gradient top, #254). For `:open`, the
open boundary set and the sponge target of its open top.
"""
core_parameters(cfg::NamedTuple, mesh) = (
    ν              = cfg.ν,
    g_accel        = cfg.g,
    f_coriolis     = cfg.f_coriolis,
    C_s            = cfg.C_s,
    Pr_t           = cfg.Pr_t,
    sponge_α_max   = cfg.sponge_α_max,
    sponge_z_start = cfg.sponge_frac * cfg.extent[3],
    case_boundary_conditions(cfg, mesh)...)

"The dry Boussinesq core of the case (buoyancy about the constant `cfg.θ₀`)."
boussinesq_core(cfg::NamedTuple, mesh; kw...) =
    BoussinesqSimulation(mesh; core_parameters(cfg, mesh)..., θ₀ = cfg.θ₀, moisture = false, kw...)

"""
    build_case(cfg; nest = nothing) -> UrbanSite

Build the mesh, the geometry, the dynamical core (`build_core` of the
provider of the `:core` slot) and its initial state,
the surface bus and the core tracers (`qv`, `ql`, `qr`), then the
components `cfg` enables (see [`selected_components`](@ref)): each is
constructed with `build(C, case, opts)`, then every `init!` runs
(in `init_priority` order, then registry order: the nest first), and
finally the stage schedule is assembled. The setup summary (buildings,
surface classes, couplings) is a `Debug` record (`subsystem = :urban`).
`nest` ([`UrbanNest`](@ref)): the site is a nested domain of the model,
the child of `nest.parent` (its `:nesting` component links it in).
"""
function build_case(cfg::NamedTuple; nest::Union{Nothing,UrbanNest} = nothing)
    # The run's compute backend (KA-a, #294): the default for every field
    # allocated below, BLAS pinned to one thread when Julia is threaded.
    init_threads!(cfg.backend)
    Nx, Ny, Nz = cfg.size
    mesh = RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
    geom = build_geometry(cfg, mesh)

    # Hydrostatic reference p̄, ρ̄ (site/reference_state.jl), built before
    # the core: the sound-proof cores use it as their base state, the
    # ledger integrates the water in the air with the same ρ̄ the couplings
    # use. One reference state per case.
    ref = build_reference(cfg, mesh)

    selected = selected_components(cfg)
    simkw = merge_keywords("BoussinesqSimulation",
                           (component_name(C) => sim_options(C, cfg, mesh, component_options(C, cfg))
                            for C in selected))
    icore = findfirst(C -> slot(C) === :core, selected)
    sim = icore === nothing ? boussinesq_core(cfg, mesh; simkw...) :
                              build_core(selected[icore], cfg, mesh, ref, simkw)
    set!(sim.θ, initial_θ(cfg))
    u0 = cfg.u0
    set!(sim.u, (x, y, z) -> u0)

    bus = SurfaceBus(Nx, Ny; Ts0 = cfg.Ts0)
    Ts0, ΔTw = cfg.Ts0, cfg.Ts_water_offset
    @. bus.Ts = Ts0 + ΔTw * (geom.sfc == SURFACE_WATER)

    budgets = Budgets(; order = ALL_STAGES)
    budgets.ρ = copy(ref.ρ̄)

    clock = Clock(0, hour_of_day(cfg, 0.0), solar_position_at(cfg, cfg.start_h), false, 0, time(), NaN, 0.0,
                  0, 0.0, NaN, NaN)
    case = UrbanSite(cfg, mesh, sim, geom, bus, TracerRegistry(), budgets, ref,
                UrbanComponent[], nothing, nothing, clock,
                nothing, nothing, Dict{Tuple{Symbol,Symbol},Float64}(), (;),
                top_heat_flux(cfg, sim, ref), open_boundary_heat(cfg, sim, ref), nest)
    # a building the box edge cuts is solid on the open faces (#262): zero
    # velocity there, and no mass-flux correction through it; then the open
    # faces get their data before any stage reads them (the transport runs
    # before the first dynamics step, whose prepare_open_boundaries! would
    # come too late: with empty buffers the first transport saw u = 0 on the
    # open faces and moved 14 % of the edge cells' vapour across them)
    # A nested case's faces get their data from the parent when the
    # `:nesting` coupling builds the nest, before the first stage reads them
    # (components/urban/nesting.jl, #265).
    if has_open_boundaries(sim)
        mark_open_boundary_solids!(sim.bcs, geom.sm.mask)
        if all(st -> st.data !== nothing, open_boundary_states(sim))
            prepare_open_boundaries!(sim)
            fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
            fill_halos!(sim.θ, sim.θ_bcs)
        end
    end

    # Core moisture tracers. qv and ql are advected by the :transport
    # stage; KNOWN: qr is sedimented by the microphysics but never
    # advected (kept for bit-for-bit; the microphysics-slot issue fixes it).
    qv = ensure_tracer!(case, :qv; advect = true, positive = true,
                        units = "kg kg-1", long_name = "water vapour specific humidity")
    set!(qv, initial_qv(cfg))                        # the sounding, or the idealised profile
    ensure_tracer!(case, :ql; advect = true, positive = true,
                   units = "kg kg-1", long_name = "cloud liquid water")
    ensure_tracer!(case, :qr; units = "kg kg-1", long_name = "rain water")

    for C in selected
        push!(case.couplings, build(C, case, component_options(C, cfg)))
    end
    for c in sort(case.couplings; by = c -> init_priority(typeof(c)))   # stable: registry order
        init!(c, case)
    end
    reset_budgets!(case)
    build_domain!(case)

    sfc = geom.sfc
    @debug("$(cfg.name): built", subsystem = :urban, domain = current_domain(), event = :built,
           case = cfg.name, grid = (case.mesh.Nx, case.mesh.Ny, case.mesh.Nz),
           buildings = length(geom.city.buildings), solid = count(geom.sm.mask),
           wall = count(==(SURFACE_WALL), sfc), glass = count(==(UInt8(6)), sfc),
           couplings = join(string.(component_name.(case.couplings)), ","))
    return case
end

"""
    describe(case) -> String

The enabled couplings with their slot and provider, and the stage
schedule of the case's `Domain`, one line per stage.
"""
function describe(case::UrbanSite)
    io = IOBuffer()
    println(io, "Couplings:")
    for c in case.couplings
        C = typeof(c)
        println(io, "  ", rpad(string(component_name(C)), 20), " slot=", something(slot(C), :addon),
                slot(C) === nothing ? "" : " provider=$(provider(C))")
    end
    println(io, "Schedule:")
    for S in ALL_STAGES
        es = [(Components.component_name(c), st) for c in case.domain.components
              for st in Components.stages(c) if st.name === S]
        isempty(es) && continue
        println(io, "  ", rpad(string(S), 16),
                join(("$(n)@$(st.order)" * (st.every > 0 ? "/$(st.every)s" : "") for (n, st) in es), "  "))
    end
    return String(take!(io))
end
