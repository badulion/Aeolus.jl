# The city-mean building (Aeolus.BuildingEnergy): one BuildingZone with
# the scene's total volume, footprint and wall area, stepped over each
# radiation interval against the canyon air; its release is put into the
# first air layer of every column.

# --- L4. Anthropogenic heat ---------------------------------------------
#
# Buildings: the output of the Phase 50 building energy model (issue #29),
# replacing the prescribed two-peak diurnal profile this file used to carry
# (class-dependent 2-40 W/m², with no source for the numbers). Traffic: a
# constant urban-area mean, see `add_traffic_heat!`.
#
# The slab surface scheme of this example has no building interior, so the
# city's buildings are represented by ONE city-mean `BuildingZone` whose
# volume, footprint and wall area are the scene's totals, run with the DOE
# medium-office schedules (#33). Its walls and roof are conduction columns
# that exchange with the canyon air through the ISO 6946 outside
# coefficient but absorb no sunshine: the slabs already absorb it, and a
# first version with sol-air walls counted it twice (-8.7 W/m² returned to
# the air in a one-hour run). A version that held the envelope's outer face
# at the slab skin temperature was tried next and dropped: that stiff
# interface, exchanged explicitly every 10 min, swung the slab budget by
# -64 then +27 W/m² (domain mean) between two radiation updates.
# The release is mapped onto the grid: HVAC exhaust and the roof envelope
# at the roofs (exhaust weighted by the floors above each roof cell); air
# exchange, glazing, lost gains and the wall envelope into the canyon (the
# SURFACE_WALL cells next to buildings). Sunshine transmitted by the
# windows is taken out of the wall cells' surface budget (`S`), which would
# otherwise absorb it too. Totals are conserved: Σ (H + LE) dx dy = the
# zone's direct release, Σ S dx dy = -Q_solar.
# `UrbanFacetScene` + `step_buildings!` resolve each building behind its own
# facets; switching this example to them is #18 / #30 scope.
#
# A BUG THE OLD CODE HAD: `add_anthropogenic_heat!` added Q_F to the
# `sensible` matrix, which `step_surfaces!` subtracts from the SURFACE
# budget, after `apply_surface_flux!` had already given the air its heat. So
# the prescribed anthropogenic heat never reached the air and instead cooled
# the surfaces by up to 40 W/m². The building release is now put into the
# first air layer directly (`apply_building_heat!`), and no longer touches
# `sensible`.

"""
    CityBuildings

State of the city-mean building of the example: the `BuildingZone`, its
`IdealLoadsHVAC`, `ThermostatSchedule` and `InternalGains`, and the maps
`H`, `LE` (W m⁻² of plan area) of the heat it put directly into the air
over the last building step, and `S` (≤ 0), the window-transmitted sun
taken from the wall cells' surface budget. See [`city_buildings`](@ref).
"""
mutable struct CityBuildings
    zone::BuildingZone{Float64}
    hvac::IdealLoadsHVAC
    thermostat::ThermostatSchedule
    gains::InternalGains
    floors::Matrix{Float64}          # floors above each column (0 off buildings)
    kair::Matrix{Int}                # first air layer above each column
    wall_cells::Vector{CartesianIndex{2}}
    H::Matrix{Float64}
    LE::Matrix{Float64}
    S::Matrix{Float64}
    release::Union{Nothing, BuildingHeatRelease{Float64}}
    hvac_budget::Union{Nothing, HVACBudget{Float64}}
end

"""
    city_buildings(heightmap, sfc, dx, dy, dz, wall_area; floor_height = 3.96,
                   glazing = 0.33, start_weekday = 4, T_air = 297.15) -> CityBuildings

The city-mean office building of the example. `wall_area` is the total
wall area (m², e.g. the sum over the radiosity scene's wall facets).

Assumptions, none of them a survey of Alexanderplatz: DOE medium-office
schedules, internal gains and thermostat (`office_internal_gains`,
`office_thermostat`); floor-to-floor height 3.96 m and window-to-wall
ratio 0.33, chosen to be close to the DOE medium-office reference
building (Deru et al. 2011) but not checked against it here; the illustrative
`UrbanSurface` example wall and roof; window U = 2.8 W m⁻² K⁻¹;
infiltration 0.3 h⁻¹; a packaged DX unit (COP 3.0) and a gas boiler.
`start_weekday` (1 = Monday): the weekday of day 1; the coupling passes
`case_weekday(cfg)` (29 June 2019, the case day, is a Saturday).
"""
function city_buildings(heightmap::AbstractMatrix, sfc::AbstractMatrix{UInt8},
                        dx::Real, dy::Real, dz::Real, wall_area::Real;
                        floor_height::Real = 3.96, glazing::Real = 0.33,
                        start_weekday::Integer = 4, T_air::Real = 297.15)
    cell = dx * dy
    built = heightmap .> 0
    A_roof = count(built) * cell
    V = sum(heightmap[built]) * cell
    floor_area = V / floor_height
    # outdoor air through 1/R_se, no absorbed sun (the slabs take it)
    walls = EnvelopeElement(example_wall_layers(), (1 - glazing) * wall_area;
                            absorptance = 0.0, T = T_air, name = :walls)
    roof = EnvelopeElement(example_roof_layers(), A_roof; h_inside = 1 / 0.10,
                           absorptance = 0.0, T = T_air, name = :roof)
    slab = EnvelopeElement([WallLayer(:xps_insulation, 0.05), WallLayer(:concrete, 0.20)], A_roof;
                           exterior = :ground, T_ground = 285.15, h_inside = 1 / 0.17,
                           T = T_air, name = :slab)
    n_floors = V / A_roof / floor_height
    floors = InternalMass([WallLayer(:concrete, 0.20)], max(n_floors - 1, 0.01) * A_roof;
                          sides = 2, T = T_air, name = :floors)
    zone = BuildingZone(; volume = V, floor_area = floor_area,
                        envelope = AbstractZoneElement[walls, roof, slab],
                        internal_mass = [floors], window_area = glazing * wall_area,
                        window_U = 2.8, infiltration_ach = 0.3, T_air = T_air)
    hvac = IdealLoadsHVAC(cop_rated = 3.0, curves = DX_CURVES_PACKAGED_AC, heating = :boiler)
    fl = [built[I] ? heightmap[I] / floor_height : 0.0 for I in CartesianIndices(heightmap)]
    kair = [clamp(round(Int, heightmap[I] / dz) + 1, 1, typemax(Int)) for I in CartesianIndices(heightmap)]
    wc = [I for I in CartesianIndices(sfc) if sfc[I] == SURFACE_WALL]
    Nx, Ny = size(heightmap)
    return CityBuildings(zone, hvac, office_thermostat(; start_weekday),
                         office_internal_gains(floor_area; start_weekday), fl, kair, wc,
                         zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny), nothing, nothing)
end

"""
    step_city_buildings!(cb, t, dt, T_out, q_out; sw_wall = 0, shgc = 0.4)

Advance the city-mean building by `dt` seconds starting `t` seconds after
00:00 of day 1, against canyon air at `T_out` (K) and `q_out` (kg/kg),
with `sw_wall` (W m⁻²) the mean irradiance on the walls (for the
window-transmitted sun), and refresh `cb.H`, `cb.LE`, `cb.S`. `shgc` of
the glazing is an assumption.
"""
function step_city_buildings!(cb::CityBuildings, t::Real, dt::Real, T_out::Real, q_out::Real;
                              sw_wall::Real = 0.0, shgc::Real = 0.4)
    z = cb.zone
    # start from the outdoor humidity, so the first hours are not a
    # vapour exchange with an arbitrary initial indoor value
    cb.release === nothing && (z.q_air = q_out)
    r = internal_gains(cb.gains, t, dt; T_air = zone_air_temperature(z))
    f = ZoneForcing(r; T_out = T_out, q_out = q_out, Q_solar = shgc * z.window_area * sw_wall)
    hv = scheduled_hvac(cb.hvac, cb.thermostat, t)
    q0 = zone_specific_humidity(z)
    b, hb = step_zone!(z, f, dt, hv)
    rel = building_heat_release(z, f, b; dt = dt, gains = r, hvac = hv, hvac_budget = hb,
                                q_before = q0)
    cb.release = rel; cb.hvac_budget = hb
    # what the walls and roof give to the air (`Q_envelope_air` split)
    walls, roof = z.envelope[1], z.envelope[2]
    Q_wall = -walls.area * walls.F_outer
    Q_roof = -roof.area * roof.F_outer
    fill!(cb.H, 0.0); fill!(cb.LE, 0.0); fill!(cb.S, 0.0)
    cell_floors = sum(cb.floors)
    nroof = count(>(0), cb.floors)
    Nx, Ny = size(cb.H)
    cell_area = roof.area / nroof
    for I in CartesianIndices(cb.floors)
        cb.floors[I] > 0 || continue
        w = cb.floors[I] / cell_floors
        cb.H[I] += (w * rel.Q_exhaust_sensible + Q_roof / nroof) / cell_area
        cb.LE[I] += w * rel.Q_exhaust_latent / cell_area
    end
    canyon_H = rel.Q_ventilation_sensible + rel.Q_windows + rel.Q_lost + Q_wall
    canyon_LE = rel.Q_ventilation_latent + rel.Q_latent_vented
    nw = length(cb.wall_cells)
    for I in cb.wall_cells
        cb.H[I] += canyon_H / nw / cell_area
        cb.LE[I] += canyon_LE / nw / cell_area
        cb.S[I] -= rel.Q_solar / nw / cell_area
    end
    return cb
end

"""
    apply_building_heat!(sim, qv_field, sensible, cb, dt; ρ = 1.0, cp = 1005.0, π = 1.0)

Put the city-mean building's direct release into the first air layer of
each column (above the roof for building columns): θ gets `H/(ρ_k cp Π_k)`
(`π` the Exner function, scalar or per level; T = θΠ),
the vapour field `LE/(ρ_k L_v)`, per `dz`, over `dt`. `ρ` is a scalar or
a per-level vector (the case passes the reference density `ρ̄`). `S` goes to the slab
surfaces: `sensible` is their loss term, so it is reduced by `S`.
"""
function apply_building_heat!(sim, qv_field, sensible::AbstractMatrix, cb::CityBuildings, dt::Real;
                              ρ = 1.0, cp::Real = 1005.0, π = 1.0)
    sensible .-= cb.S
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    dz = sim.mesh.dz; Nz = sim.mesh.Nz
    qv = Aeolus.Fields.interior_view(qv_field)
    @inbounds for I in CartesianIndices(cb.H)
        i, j = Tuple(I)
        k = min(cb.kair[I], Nz)
        ρk = ρ isa AbstractVector ? ρ[k] : ρ
        πk = π isa AbstractVector ? π[k] : π
        sim.θ.data[Hx+i, Hy+j, Hz+k] += cb.H[I] / (ρk * cp * πk) * dt / dz
        qv[i, j, k] += cb.LE[I] / (ρk * _L_v) * dt / dz
    end
    return sim
end


"""
    CityMeanBuildings

Slot `:buildings`, provider `:citymean`. The [`CityBuildings`](@ref)
state is built from the scene at `init!`; at the `:radiation` cadence
(after the radiosity, which it needs for the wall irradiance) the zone
is advanced over the radiation interval against the mean first-layer air
of the open columns; every step, in `:anthropogenic`, its release goes
into θ and qv and the window sun on the walls leaves the surface budget
([`apply_building_heat!`](@ref)).
"""
mutable struct CityMeanBuildings <: AbstractCoupling
    bem::Union{Nothing,CityBuildings}
end

coupling_name(::Type{CityMeanBuildings}) = :buildings_citymean
slot(::Type{CityMeanBuildings}) = :buildings
provider(::Type{CityMeanBuildings}) = :citymean
modules_used(::Type{CityMeanBuildings}) = (:BuildingEnergy, :UrbanSurface)
stages(::Type{CityMeanBuildings}) = ((:radiation, 20, Cadence(:radiation; first = true)),
                                     (:anthropogenic, 10, EVERY_STEP))
build(::Type{CityMeanBuildings}, case, o) = CityMeanBuildings(nothing)

function init!(c::CityMeanBuildings, case)
    g, mesh = case.geom, case.mesh
    scene = g.scene
    c.bem = city_buildings(g.heightmap, g.sfc, mesh.dx, mesh.dy, mesh.dz,
                           sum(scene.facets[i].area for i in g.wall_ids);
                           start_weekday = case_weekday(case.cfg))
    bem = c.bem
    @debug(@sprintf("City-mean building: %.0f m² floor area, %.0f m² walls, %.0f m² roofs",
                    bem.zone.floor_area, bem.zone.envelope[1].area / (1 - 0.33), bem.zone.envelope[2].area),
           subsystem = :berlin, domain = current_domain(), coupling = :buildings)
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

run_stage!(c::CityMeanBuildings, case, ::Val{:anthropogenic}, dt) =
    apply_building_heat!(case.sim, tracer(case, :qv), case.bus.sensible, c.bem, dt;
                         ρ = ref_density(case), cp = case.ref.cp, π = case.ref.π̄)

function report(c::CityMeanBuildings, case)
    bem = c.bem
    bem.release === nothing && return nothing
    Lx, Ly = case.mesh.Lx, case.mesh.Ly
    Nx, Ny = case.mesh.Nx, case.mesh.Ny
    r = bem.release
    zr, zw = bem.zone.envelope[2], bem.zone.envelope[1]
    return @sprintf("           buildings: T_in=%5.2f°C  to air %5.1f W/m² (exhaust %5.1f, canyon %5.1f)  (roof env %5.1f, wall env %5.1f)  window sun %5.1f W/m²  heat %6.0f kW  cool %6.0f kW",
                    zone_air_temperature(bem.zone) - 273.15,
                    sum(bem.H .+ bem.LE) / (Nx * Ny),
                    (r.Q_exhaust_sensible + r.Q_exhaust_latent) / (Lx * Ly),
                    (r.Q_ventilation_sensible + r.Q_ventilation_latent + r.Q_windows + r.Q_lost +
                     r.Q_latent_vented) / (Lx * Ly),
                    -zr.area * zr.F_outer / (Lx * Ly), -zw.area * zw.F_outer / (Lx * Ly),
                    r.Q_solar / (Lx * Ly),
                    bem.hvac_budget.Q_heating / 1e3, bem.hvac_budget.Q_cooling / 1e3)
end

# Ledger: the window sun on the walls leaves the surface budget (S ≤ 0 is
# the flux into the surface); the building's latent release enters the
# air as vapour, LE·dt/L_v per column (apply_building_heat! converts with
# the reference density ρ̄_k, the one the ledger integrates with).
function budget!(c::CityMeanBuildings, case, l)
    dt = l.dt
    add_surface!(l, :window_sun, c.bem.S, case.geom.sfc, dt)
    add_water!(l, :building_vapour, sum(c.bem.LE) * dt / _L_v / length(c.bem.LE))
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

register!(CityMeanBuildings)
