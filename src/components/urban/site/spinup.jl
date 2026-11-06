# The surface spin-up of the urban site (#519): PALM's `spinup_time`
# (time_integration_spinup.f90, PALM 25.10.1) on the site's pipeline.
#
# Before the first step of a fresh run the site's slow surfaces — the
# soil columns of the ground, the slabs of roofs and water, the city-mean
# building's zone and envelope — advance alone over `[spinup] time`
# seconds, at negative model times ending at the start:
#
#   * only the stages `Components.SPINUP_STAGES` (clock, radiation,
#     shortwave modifiers, surface fluxes, sources, surface update) of the
#     couplings that take part (`in_spinup`); no transport, microphysics,
#     chemistry or core step;
#   * the air the surfaces see is the initial state, except the potential
#     temperature of the air cell above every surface, which is the
#     forcing: PALM's θ = pt_mean + pt_amplitude cos(solar zenith), the
#     sun 1 h earlier, or the dynamic driver's sfc_temperature series. The
#     wind and the humidity stay at the initial state (what a step
#     delivered into θ and qv is put back before the next);
#   * the soil water is frozen unless `soil_moisture` (PALM's
#     calc_soil_moisture_during_spinup); windows (`:glass`) and the
#     traffic heat do not take part (PALM: no windows in the spin-up);
#   * afterwards the atmosphere is the initial state again, the clock at
#     the start, the surface exchange's schedule and ledgers fresh, and the
#     first step's radiation runs (the cadence's `first`), as PALM forces a
#     radiation call at the coupling.
#
# The convergence of the spin-up is reported (`results.toml`, [spinup]):
# per surface class the change of the daily-mean surface temperature over
# the last day, K d⁻¹, when the spin-up is at least two days long.

"""
    SpinupState

The site's surface spin-up (#519): `stop`, the model time it ends at
(`Inf` outside it: the surface exchange takes no surface step there, which
would belong to the run), `saved`, what the couplings put aside in
[`spinup_begin!`](@ref) to restore in [`spinup_end!`](@ref), and `report`,
what it writes to `results.toml`.
"""
mutable struct SpinupState
    stop::Float64
    saved::Dict{Symbol,Any}
    report::Dict{String,Any}
end
SpinupState() = SpinupState(Inf, Dict{Symbol,Any}(), Dict{String,Any}())

"""
    in_spinup(c::UrbanComponent) -> Bool

Whether the coupling runs its stages in the surface spin-up. Default
`false`; the couplings of the surfaces and of the radiation that reaches
them say `true`.
"""
in_spinup(c::UrbanComponent) = in_spinup(typeof(c))
in_spinup(::Type{<:UrbanComponent}) = false

"""
    spinup_begin!(c, case, sp, t0, dt)
    spinup_end!(c, case, t_end)

A coupling's preparation for the surface spin-up that starts at `t0` with
the surface step `dt` (`sp`: the `[spinup]` options), and its return to the
run's state at the start `t_end`. Default: nothing.
"""
spinup_begin!(::UrbanComponent, case, sp, t0, dt) = nothing
spinup_end!(::UrbanComponent, case, t_end) = nothing

Components.spins_up(::UrbanSite) = true
Components.spins_up(e::UrbanEntry) = in_spinup(e.c)

"""
    spinup_step(case, sp) -> dt

The surface step of the spin-up: `sp.dt`, or the site's surface step
(`surface_period`, else `soil_period`).
"""
spinup_step(case, sp) = sp.dt !== nothing ? Float64(sp.dt) :
                        Float64(get(case.cfg.cadence, :surface, case.cfg.cadence[:soil]))

"The name of surface class `c` (`InputDrivers.SURFACE_CLASSES`)."
surface_class_name(c) = something(get(Dict(SURFACE_CLASSES), UInt8(c), nothing), string(Int(c)))

"The columns with a surface: their (i, j, k_air)."
_surface_cells(case) = [(I[1], I[2], case.bus.k_air[I]) for I in CartesianIndices(case.bus.k_air)
                        if case.bus.k_air[I] <= case.mesh.Nz]

"""
    spinup_air(case, sp, cells, t0, t_end) -> (t -> θ)

The potential temperature (K) of the air cells above the surfaces during
the spin-up, as a function of the model time:

  * `forcing = :solar` (PALM): ``θ = θ_m + A \\cos ζ(t - 1\\,\\mathrm h)``, with
    ``ζ`` the solar zenith of the site's date and place (the case date's sun
    on every day of the spin-up), `pt_mean` ``θ_m`` (default: the initial mean
    of `cells`) and `pt_amplitude` ``A``;
  * `forcing = :driver`: the dynamic driver's `sfc_temperature` series,
    linear in time, on the model's time axis (the driver's `init_time`
    against the case's); it must cover `[t0, t_end]`.
"""
function spinup_air(case, sp, cells, t0, t_end)
    cfg = case.cfg
    if sp.forcing === :driver
        cfg.dynamic === nothing && throw(ArgumentError("spin-up: forcing = \"driver\" needs the domain's dynamic_driver"))
        d = read_dynamic_driver(cfg.dynamic; planes = false)
        s = d.surface
        (s === nothing || s.temperature === nothing) &&
            throw(ArgumentError("spin-up: the dynamic driver $(cfg.dynamic) has no sfc_temperature series"))
        off = isempty(d.init_time) ? 0.0 :
              Dates.value(_utc(case_init_time(cfg)) - _utc(d.init_time)) / 1000
        ts = Float64.(s.t) .- off
        θs = Float64.(s.temperature)
        (ts[1] <= t0 + 1e-6 && ts[end] >= t_end - 1e-6) ||
            throw(ArgumentError(@sprintf("spin-up: the driver's sfc_temperature covers model times [%.0f, %.0f] s, the spin-up needs [%.0f, %.0f] s",
                                         ts[1], ts[end], t0, t_end)))
        return t -> _interp_series(ts, θs, t)
    end
    θ = interior_view(case.sim.θ)
    θm = sp.pt_mean !== nothing ? Float64(sp.pt_mean) : mean(θ[i, j, k] for (i, j, k) in cells)
    A = sp.pt_amplitude
    return t -> θm + A * cosd(solar_position_at(cfg, hour_of_day(cfg, t - 3600.0)).zenith)
end

function _interp_series(ts, vs, t)
    n = searchsortedlast(ts, t)
    n == 0 && return vs[1]
    n == length(ts) && return vs[end]
    w = (t - ts[n]) / (ts[n+1] - ts[n])
    return (1 - w) * vs[n] + w * vs[n+1]
end

# the prognostic air the spin-up stages change (θ and the tracers: the
# surface fluxes, the building release) and puts back after every step
_air_fields(case) = (case.sim.θ, (t.field for t in case.tracers.list)...)

"""
    Components.spinup!(case::UrbanSite, d, sp) -> Bool

The site's surface spin-up (see the file's header): from `sim.t - sp.time`
to `sim.t` in steps of [`spinup_step`](@ref), the
`Components.SPINUP_STAGES` of the couplings that are [`in_spinup`](@ref),
the air above the surfaces at [`spinup_air`](@ref); then the atmosphere,
the clock and the couplings are back at the start. Records the spin-up in
`case.spin.report` (`results.toml`).
"""
function Components.spinup!(case::UrbanSite, d::Domain, sp)
    sp.time > 0 || return false
    sim, clk, cfg = case.sim, case.clock, case.cfg
    t_end = Float64(sim.t)
    t0 = t_end - sp.time
    dt = spinup_step(case, sp)
    cells = _surface_cells(case)
    air = spinup_air(case, sp, cells, t0, t_end)
    saved = [copy(f.data) for f in _air_fields(case)]
    n0 = clk.n
    case.spin.stop = t_end
    foreach(c -> spinup_begin!(c, case, sp, t0, dt), case.couplings)
    @info(@sprintf("surface spin-up: %.1f h in steps of %.0f s, forcing %s", sp.time / 3600, dt, sp.forcing),
          subsystem = :urban, domain = current_domain(), event = :spinup_start, time = sp.time, dt = dt)
    # the daily-mean surface temperature per class over the last two days
    cls = sort!(unique(case.geom.sfc[i, j] for (i, j, _) in cells))
    day = Dict(c => zeros(2) for c in cls)            # [the day before, the last day] ∫Ts dt
    ncls = Dict(c => count(((i, j, _),) -> case.geom.sfc[i, j] == c, cells) for c in cls)
    wall0 = time()
    sim.t = t0
    nstep = 0
    θi = interior_view(sim.θ)
    try
        while t_end - sim.t > 1e-9 * dt
            h = min(dt, t_end - sim.t)
            ta = Float64(sim.t)
            θa = air(ta)
            for (i, j, k) in cells
                θi[i, j, k] = θa
            end
            Components.run_spinup_stages!(d, h, ta, ta + h; first = nstep == 0)
            sim.t = ta + h
            nstep += 1
            for (f, s) in zip(_air_fields(case), saved)
                copyto!(f.data, s)
            end
            # the two last days, by the step's end
            slot = sim.t > t_end - 86400 + 1e-9 ? 2 : sim.t > t_end - 2 * 86400 + 1e-9 ? 1 : 0
            if slot > 0
                for (i, j, _) in cells
                    day[case.geom.sfc[i, j]][slot] += case.bus.Ts[i, j] * h
                end
            end
            any(!isfinite, case.bus.Ts) &&
                error(@sprintf("the surface spin-up blew up at model time %.0f s (a non-finite surface temperature)", sim.t))
        end
    finally
        for (f, s) in zip(_air_fields(case), saved)
            copyto!(f.data, s)
        end
        sim.t = t_end
        clk.n = n0
        clk.hour = hour_of_day(cfg, t_end)
        case.spin.stop = Inf
        foreach(c -> spinup_end!(c, case, t_end), case.couplings)
    end
    rep = Dict{String,Any}("time_s" => sp.time, "dt_s" => dt, "steps" => nstep, "forcing" => string(sp.forcing),
                           "wall_s" => time() - wall0, "soil_moisture" => sp.soil_moisture,
                           "buildings" => sp.buildings)
    if sp.time >= 2 * 86400 - 1e-6
        drift = Dict{String,Any}(surface_class_name(c) => (day[c][2] - day[c][1]) / 86400 / ncls[c] for c in cls)
        rep["drift_K_per_day"] = drift
        worst = maximum(abs, values(drift))
        rep["drift_max_K_per_day"] = worst
        worst > SPINUP_DRIFT_WARN &&
            @warn(@sprintf("surface spin-up: the daily-mean surface temperature still changes by %.2f K d⁻¹ (> %.1f): spin up longer, or initialise the slow state from data (#521)",
                           worst, SPINUP_DRIFT_WARN),
                  subsystem = :urban, domain = current_domain(), event = :spinup_drift, drift = worst)
    end
    case.spin.report = rep
    @info(@sprintf("surface spin-up done: %d steps, %.1f s wall", nstep, rep["wall_s"]),
          subsystem = :urban, domain = current_domain(), event = :spinup_end, steps = nstep)
    return true
end

"The drift of a class's daily-mean surface temperature over the spin-up's last day above which the run warns (K d⁻¹)."
const SPINUP_DRIFT_WARN = 0.5
