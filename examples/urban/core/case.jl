# The Case: one configured, built, runnable Berlin simulation.

"""
    Clock

Step counter and time state of a case: `n` (steps taken), `hour`
(local hour of day at the start of the current step), `sun` (the solar
position of the last radiation update), `blowup` (set by the `:sanity`
stage), `nrecord` (output records written) and `t_wall0` (wall-clock
start of `run_case!`).
"""
mutable struct Clock
    n::Int
    hour::Float64
    sun::Any
    blowup::Bool
    nrecord::Int
    t_wall0::Float64
end

"One scheduled call: `coupling` runs stage `S` at `order` with `cadence`."
struct ScheduleEntry
    coupling::AbstractCoupling
    order::Int
    cadence::Cadence{Float64}
end

"""
    Case

A built case: `cfg`, `mesh`, `sim` (the dynamical core: a
`BoussinesqSimulation`, or the `MassWeightedSimulation` of the `:core`
slot's anelastic / pseudo-incompressible provider), `geom`
([`Geometry`](@ref)), `bus` ([`SurfaceBus`](@ref)), `tracers`
([`TracerRegistry`](@ref)), `ledger`, the hydrostatic reference state
`ref` ([`CaseReference`](@ref)), the enabled `couplings` (registry
order), the `schedule` (stage ⇒ ordered entries), the `clock`, the
NetCDF `output` (or `nothing`), the `io` for log lines, the composed
`step!` hooks, and `timing` (wall seconds per `(stage, coupling)`, see
[`timing_report`](@ref)).
"""
mutable struct Case{M, S, G<:Geometry}
    cfg::CaseConfig
    mesh::M
    sim::S
    geom::G
    bus::SurfaceBus
    tracers::TracerRegistry
    ledger::Ledger
    ref::CaseReference
    couplings::Vector{AbstractCoupling}
    schedule::Dict{Symbol,Vector{ScheduleEntry}}
    clock::Clock
    output::Any
    io::IO
    body_force::Union{Nothing,Function}
    post_predictor::Union{Nothing,Function}
    timing::Dict{Tuple{Symbol,Symbol},Float64}
    step_kwargs::NamedTuple      # extra step! keywords from the couplings' step_options
end

"The enabled coupling called `name`, or `nothing`."
function find_coupling(case::Case, name::Symbol)
    for c in case.couplings
        coupling_name(c) === name && return c
    end
    return nothing
end

"Local hour of day at simulation time `t` (s)."
hour_of_day(cfg::CaseConfig, t::Real) = (cfg.start_h + t / 3600) % 24.0

"""
    case_weekday(cfg) -> 1 … 7

Day of the week of `cfg.date` (1 = Monday … 7 = Sunday, the convention
of `BuildingEnergy.WeeklySchedule`): 29 June 2019 is a Saturday (6).
"""
case_weekday(cfg::CaseConfig) =
    mod(Aeolus.LargeScaleForcing._days_from_civil(cfg.date...) + 3, 7) + 1   # 1970-01-01: Thursday

"""
    solar_position_at(cfg, hour) -> SolarPosition

The sun at local clock `hour` of `cfg.date` over the case anchor.
"""
function solar_position_at(cfg::CaseConfig, hour::Real)
    hh = floor(Int, hour)
    mi = floor(Int, (hour - hh) * 60)
    se = (hour - hh - mi / 60) * 3600
    y, m, d = cfg.date
    return solar_position(solar_time(y, m, d, hh, mi, se;
                                     latitude = cfg.lat0, longitude = cfg.lon0,
                                     elevation = cfg.elevation))
end

"""
    core_parameters(cfg) -> NamedTuple

The keyword arguments every dynamical core of the case shares: viscosity,
gravity, Coriolis, Smagorinsky, sponge and the boundary conditions
(periodic laterally, wall / no-flux at top and bottom).
"""
core_parameters(cfg::CaseConfig) = (
    ν              = cfg.ν,
    g_accel        = cfg.g,
    f_coriolis     = cfg.f_coriolis,
    C_s            = cfg.C_s,
    Pr_t           = cfg.Pr_t,
    sponge_α_max   = cfg.sponge_α_max,
    sponge_z_start = cfg.sponge_frac * cfg.extent[3],
    bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
    θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

"The dry Boussinesq core of the case (buoyancy about the constant `cfg.θ₀`)."
boussinesq_core(cfg::CaseConfig, mesh; kw...) =
    BoussinesqSimulation(mesh; core_parameters(cfg)..., θ₀ = cfg.θ₀, moisture = false, kw...)

"""
    build_case(cfg; io = stdout) -> Case

Build the mesh, the geometry, the dynamical core (`build_core` of the
provider of the `:core` slot) and its initial state,
the surface bus and the core tracers (`qv`, `ql`, `qr`), then the
couplings `cfg` enables (see [`selected_couplings`](@ref)): each is
constructed with `build(C, case, opts)`, then every `init!` runs in
registry order, and finally the stage schedule is assembled.
"""
function build_case(cfg::CaseConfig; io::IO = stdout)
    Nx, Ny, Nz = cfg.size
    mesh = RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
    geom = build_geometry(cfg, mesh; io = io)

    # Hydrostatic reference p̄, ρ̄ (core/reference_state.jl), built before
    # the core: the sound-proof cores use it as their base state, the
    # ledger integrates the water in the air with the same ρ̄ the couplings
    # use. One reference state per case.
    ref = build_reference(cfg, mesh)

    selected = selected_couplings(cfg)
    simkw = merge_keywords("BoussinesqSimulation",
                           (coupling_name(C) => sim_options(C, cfg, mesh, coupling_options(C, cfg))
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

    ledger = Ledger()
    ledger.ρ = copy(ref.ρ̄)

    clock = Clock(0, hour_of_day(cfg, 0.0), solar_position_at(cfg, cfg.start_h), false, 0, time())
    case = Case(cfg, mesh, sim, geom, bus, TracerRegistry(), ledger, ref,
                AbstractCoupling[], Dict{Symbol,Vector{ScheduleEntry}}(), clock,
                nothing, io, nothing, nothing, Dict{Tuple{Symbol,Symbol},Float64}(), (;))

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
        push!(case.couplings, build(C, case, coupling_options(C, cfg)))
    end
    for c in case.couplings
        init!(c, case)
    end
    build_schedule!(case)

    if cfg.verbose
        sfc = geom.sfc
        @printf(io, "%s OSM: %d buildings; solid=%d wall=%d glass=%d\n",
                cfg.name, length(geom.city.buildings), count(geom.sm.mask),
                count(==(SURFACE_WALL), sfc), count(==(UInt8(6)), sfc))
        @printf(io, "Couplings: %s\n", join(string.(coupling_name.(case.couplings)), ", "))
    end
    return case
end

"""
    describe(case) -> String

The enabled couplings with their slot and provider, and the stage
schedule, one line each.
"""
function describe(case::Case)
    io = IOBuffer()
    println(io, "Couplings:")
    for c in case.couplings
        C = typeof(c)
        println(io, "  ", rpad(string(coupling_name(C)), 20), " slot=", slot(C),
                slot(C) === :addon ? "" : " provider=$(provider(C))")
    end
    println(io, "Schedule:")
    for S in ALL_STAGES
        es = get(case.schedule, S, ScheduleEntry[])
        isempty(es) && continue
        println(io, "  ", rpad(string(S), 16),
                join(("$(coupling_name(e.coupling))@$(e.order)" *
                      (e.cadence.period > 0 ? "/$(e.cadence.period)s" : "") for e in es), "  "))
    end
    return String(take!(io))
end
