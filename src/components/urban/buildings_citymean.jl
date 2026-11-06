# The city-mean building (Aeolus.BuildingEnergy): one BuildingZone with
# the scene's total volume, footprint and wall area, stepped over each
# radiation interval against the canyon air; its release is put into the
# first air layer of every column. The building model, its mapping onto
# the grid and the release into the air are `BuildingEnergy.city_buildings`,
# `step_city_buildings!` and `apply_building_heat!` (moved there in ONE-e,
# #454); this file wires them into the case.
#
# Buildings: the output of the Phase 50 building energy model (issue #29),
# replacing the prescribed two-peak diurnal profile this file used to carry
# (class-dependent 2-40 W/m², with no source for the numbers). Traffic: a
# constant urban-area mean, see `add_traffic_heat!`.
#
# A BUG THE OLD CODE HAD: `add_anthropogenic_heat!` added Q_F to the
# `sensible` matrix, which `step_surfaces!` subtracts from the SURFACE
# budget, after `apply_surface_flux!` had already given the air its heat. So
# the prescribed anthropogenic heat never reached the air and instead cooled
# the surfaces by up to 40 W/m². The building release is now put into the
# first air layer directly (`apply_building_heat!`), and no longer touches
# `sensible`.

"""
    CityMeanBuildings

Slot `:buildings`, provider `:citymean`. The [`CityBuildings`](@ref)
state is built from the scene at `init!`; at the `:radiation` cadence
(after the radiosity, which it needs for the wall irradiance) the zone
is advanced over the radiation interval against the mean first-layer air
of the open columns; every step, in `:sources`, its release goes
into θ and qv and the window sun on the walls leaves the surface budget
([`apply_building_heat!`](@ref)).
"""
mutable struct CityMeanBuildings <: UrbanComponent
    bem::Union{Nothing,CityBuildings}
    spin::Bool                       # takes part in the surface spin-up ([spinup] buildings, #519)
end

component_name(::Type{CityMeanBuildings}) = :buildings_citymean
slot(::Type{CityMeanBuildings}) = :buildings
provider(::Type{CityMeanBuildings}) = :citymean
modules_used(::Type{CityMeanBuildings}) = (:BuildingEnergy, :UrbanSurface)
stages(::Type{CityMeanBuildings}) = ((:radiation, 20, Cadence(:radiation; first = true)),
                                     (:sources, 10, EVERY_STEP))
build(::Type{CityMeanBuildings}, case, o) = CityMeanBuildings(nothing, true)
# the zone and its envelope in the surface spin-up, unless [spinup] buildings = false (#519)
in_spinup(c::CityMeanBuildings) = c.spin
spinup_begin!(c::CityMeanBuildings, case, sp, t0, dt) = (c.spin = sp.buildings; nothing)
spinup_end!(c::CityMeanBuildings, case, t_end) = (c.spin = true; nothing)

function init!(c::CityMeanBuildings, case)
    g, mesh = case.geom, case.mesh
    scene = g.scene
    c.bem = city_buildings(g.heightmap, [I for I in CartesianIndices(g.sfc) if g.sfc[I] == SURFACE_WALL],
                           mesh.dx, mesh.dy, mesh.dz,
                           sum(scene.facets[i].area for i in g.wall_ids);
                           start_weekday = case_weekday(case.cfg),
                           # the envelope starts in its steady state against the
                           # initial canyon air (#519), not uniform at the zone air
                           T_outdoor = mean(view(interior_view(case.sim.θ), :, :, 1)[g.open_cols]) * ref_exner(case, 1))
    bem = c.bem
    @debug("city-mean building", subsystem = :urban, domain = current_domain(), event = :init,
           coupling = :buildings_citymean, floor_area = bem.zone.floor_area,
           wall_area = bem.zone.envelope[1].area / (1 - 0.33), roof_area = bem.zone.envelope[2].area)
    return nothing
end

function run_stage!(c::CityMeanBuildings, case, ::Val{:radiation}, dt)
    # Advance the buildings over the next radiation interval against the
    # canyon air; the irradiance on the walls sets the window sun.
    g, swsol, sim = case.geom, case.bus.swsol, case.sim
    scene, ids = g.scene, g.wall_ids
    aw = swsol === nothing ? 0.0 :
         sum(swsol.incident[i] * scene.facets[i].area for i in ids) /
         sum(scene.facets[i].area for i in ids)
    θ1 = view(interior_view(sim.θ), :, :, 1)
    q1 = view(interior_view(tracer(case, :qv)), :, :, 1)
    step_city_buildings!(c.bem, case.cfg.start_h * 3600 + sim.t, cadence_period(case.cfg, :radiation),
                         mean(θ1[g.open_cols]), mean(q1[g.open_cols]); sw_wall = aw)
    return nothing
end

run_stage!(c::CityMeanBuildings, case, ::Val{:sources}, dt) =
    apply_building_heat!(case.sim, tracer(case, :qv), case.bus.sensible, c.bem, dt;
                         ρ = ref_density(case), cp = case.ref.cp, π = case.ref.π̄)

"""
    diagnostics(c::CityMeanBuildings, case)

The city-mean building of the last building step (`NaN` before it): the
zone air temperature `T_in` (K); per unit ground area of the domain
(W m⁻²) the heat to the air `to_air` and its parts, the HVAC exhaust
`exhaust` and what leaves through the canyon facades `canyon`
(ventilation, windows, losses, vented latent heat); the heat the roof
and wall envelopes take from outside, `roof_envelope` and
`wall_envelope`; the sun through the windows `window_sun`; and the HVAC
`heating` and `cooling` (W).
"""
function diagnostics(c::CityMeanBuildings, case)
    bem = c.bem
    A = case.mesh.Lx * case.mesh.Ly
    N = case.mesh.Nx * case.mesh.Ny
    rel(x) = () -> bem.release === nothing ? NaN : x(bem.release)
    env(i) = () -> bem.release === nothing ? NaN :
                   (z = bem.zone.envelope[i]; -z.area * z.F_outer / A)
    return Diagnostic[
        Diagnostic("T_in", "K", () -> bem.release === nothing ? NaN : zone_air_temperature(bem.zone);
                   description = "zone air temperature"),
        Diagnostic("to_air", "W m-2", () -> bem.release === nothing ? NaN : sum(bem.H .+ bem.LE) / N),
        Diagnostic("exhaust", "W m-2", rel(r -> (r.Q_exhaust_sensible + r.Q_exhaust_latent) / A)),
        Diagnostic("canyon", "W m-2",
                   rel(r -> (r.Q_ventilation_sensible + r.Q_ventilation_latent + r.Q_windows +
                             r.Q_lost + r.Q_latent_vented) / A)),
        Diagnostic("roof_envelope", "W m-2", env(2)),
        Diagnostic("wall_envelope", "W m-2", env(1)),
        Diagnostic("window_sun", "W m-2", rel(r -> r.Q_solar / A)),
        Diagnostic("heating", "W", () -> bem.release === nothing ? NaN : bem.hvac_budget.Q_heating),
        Diagnostic("cooling", "W", () -> bem.release === nothing ? NaN : bem.hvac_budget.Q_cooling)]
end

# Ledger: the window sun on the walls leaves the surface budget (S ≤ 0 is
# the flux into the surface); the building's latent release enters the
# air as vapour, LE·dt/L_v per column (apply_building_heat! converts with
# the reference density ρ̄_k, the one the ledger integrates with).
function budget!(c::CityMeanBuildings, case, l)
    dt = l.dt
    add_surface!(l, :window_sun, c.bem.S, case.geom.sfc, dt)
    add_water!(l, :building_vapour, sum(c.bem.LE) * dt / CITY_BUILDING_L_V / length(c.bem.LE))
    return nothing
end

function smoke_check(c::CityMeanBuildings, case)
    bem = c.bem
    T_in = zone_air_temperature(bem.zone) - 273.15
    b = bem.hvac_budget
    return ["the zone was stepped" => bem.release !== nothing,
            "indoor air within the thermostat band ± 5 K (15–31 °C)" => 15 <= T_in <= 31,
            "HVAC heating and cooling ≥ 0" => b !== nothing && b.Q_heating >= 0 && b.Q_cooling >= 0,
            "window sun only leaves the surface (S ≤ 0)" => all(<=(0), bem.S),
            "building release into the air ≥ 0 in summer" =>
                case.cfg.scenario !== :summer || sum(bem.H .+ bem.LE) >= 0]
end

