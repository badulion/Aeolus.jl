# Phase 49 — anthropogenic heat from the building energy model (#29).
#
# The building part of the anthropogenic heat flux Q_F is what the
# BuildingEnergy zone and its HVAC plant give to the outdoors. Validation:
#
#   * the first law against the zone's own state: heat released + change in
#     stored (sensible + latent) heat = energy consumed + transmitted sun,
#     every step, with the storage measured independently from
#     `zone_heat_content` and the humidity;
#   * over a periodic steady state, energy consumed = heat released (the
#     premise of the inventory method, Sailor 2011);
#   * closed forms for steady heating by boiler and heat pump and for
#     steady cooling, where the net release is the purchased energy alone;
#   * the coupled zone + facet-scene step (step_buildings!) against step_zone!
#     and against its own budgets;
#   * magnitude (a validation run, validation/era5_anthropogenic_heat.jl):
#     the dense residential centre of Toulouse in winter against the
#     CAPITOUL inventory of building electricity and gas use (Pigeon et
#     al. 2007, as reported by Bueno et al. 2012, Table 2), and the range of
#     published city Q_F values.

using Test, Printf
using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface
using Aeolus.SurfaceExchange
using Aeolus.MoistThermodynamics
using Aeolus.Radiosity

const AHB = Aeolus.BuildingEnergy

# Massless box: air node + glazing + air exchange only, so steady states are
# scalar closed forms.
ah_air_box(; T = 293.15, ach = 0.5, UA = 150.0, q = nothing) =
    BuildingZone(; volume = 300.0, floor_area = 100.0, window_area = 1.0, window_U = UA,
                 infiltration_ach = ach, T_air = T, q_air = q)

# Every kind of element at once: an `:air` wall, a FacetElement roof, a
# ground slab, an adiabatic party wall, internal mass, glazing, ventilation
# with heat recovery.
function ah_full_zone(; T = 295.15)
    wl = example_wall_layers()
    walls = EnvelopeElement(wl, 60.0; absorptance = 0.6, T = T, name = :walls)
    roof = FacetElement(UrbanFacet(:roof, example_roof_layers(); T = T, T_interior = T), 48.0)
    slab = EnvelopeElement([WallLayer(:xps_insulation, 0.08), WallLayer(:concrete, 0.15)], 48.0;
                           exterior = :ground, T_ground = 285.15, h_inside = 1 / 0.17, T = T,
                           name = :slab)
    party = EnvelopeElement(wl, 16.2; exterior = :adiabatic, T = T, name = :party)
    im = [InternalMass([WallLayer(:brick, 0.12)], 20.0; sides = 2, T = T)]
    return BuildingZone(; volume = 8 * 6 * 2.7, floor_area = 48.0,
                        envelope = [walls, roof, slab, party], internal_mass = im,
                        window_area = 10.0, window_U = 2.8, infiltration_ach = 0.4,
                        ventilation_ach = 1.0, heat_recovery = 0.6, T_air = T)
end

# Two 20 m slab blocks either side of a 10 m street (H/W = 2), 60 m long,
# as in the #28 tests; one zone per block.
function ah_canyon(; T0 = 291.0, Tin = 293.15, subdiv = 2)
    hm = zeros(8, 12); hm[1:3, :] .= 20; hm[6:8, :] .= 20
    sc = build_scene(hm, 5.0, 5.0; wall_subdivisions = subdiv)
    vf = view_factors(sc)
    opts(f) = f.kind === :wall ? (h_convective = 10.0, T_interior = Tin) :
              f.kind === :roof ? (T_interior = Tin,) : (T_interior = 288.0,)
    u = UrbanFacetScene(sc, vf; T = T0, options = opts)
    west = falses(8, 12); west[1:3, :] .= true
    east = falses(8, 12); east[6:8, :] .= true
    return u, west, east
end

# A single horizontal roof facet that sees only the sky (F_sky = 1).
function ah_isolated_roof(area)
    mats = UrbanMaterials([0.15], [0.95])
    fs = [Facet((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                area, 1.0, :roof, 1, 293.15, (1, 1))]
    sc = RadiosityScene{Float64}(fs, zeros(1, 1), 1.0, 1.0, 0.0, 0.0, 0.0, mats)
    return sc, ViewFactors{Float64}(zeros(1, 1), [1.0], [area])
end

# Everything a zone has besides its roof: walls to outdoor air, a ground
# slab, internal mass, glazing, ventilation.
function ah_zone_with(roof_el; T = 293.15)
    walls = EnvelopeElement(example_wall_layers(), 60.0; absorptance = 0.6, T = T, name = :walls)
    slab = EnvelopeElement([WallLayer(:xps_insulation, 0.08), WallLayer(:concrete, 0.15)], 48.0;
                           exterior = :ground, T_ground = 285.15, h_inside = 1 / 0.17, T = T)
    im = [InternalMass([WallLayer(:brick, 0.12)], 20.0; sides = 2, T = T)]
    return BuildingZone(; volume = 8 * 6 * 2.7, floor_area = 48.0,
                        envelope = AbstractZoneElement[walls, roof_el, slab], internal_mass = im,
                        window_area = 10.0, window_U = 2.8, infiltration_ach = 0.5, T_air = T)
end

# Summer-like and winter-like days, 15-min steps.
ah_T_out(t, mean) = mean + 7 * sin(2π * (t / 86400 - 0.375))
ah_sun(t) = max(0.0, 700 * sin(2π * (t / 86400 - 0.25)))

@testset "Era V — anthropogenic heat from the building energy model (#29)" begin

    @testset "First law every step: release + storage = consumption + sun" begin
        # office gains with every split (lost lighting heat, latent equipment),
        # boiler + evaporative condenser + dehumidification, a prescribed
        # extra heat input, and a FacetElement roof in the open
        g = office_internal_gains(48.0; equipment_latent_fraction = 0.1,
                                  equipment_lost_fraction = 0.1)
        hv = IdealLoadsHVAC(T_heat = 294.15, T_cool = 297.15, condenser = :evaporative,
                            dehumidify = true)
        dt = 900.0
        worst = 0.0; worst_E = 0.0; gross_max = 0.0; n_cons_bad = 0
        seen = Dict(:heat => false, :cool => false, :lost => false, :vent_lat => false)
        for (season, Tm, q_out) in ((:winter, 276.15, 0.004), (:summer, 301.15, 0.014))
            z = ah_full_zone(T = season === :winter ? 294.15 : 297.15)
            for k in 1:(4 * 96)
                t = k * dt
                sun = ah_sun(t)
                T_out = ah_T_out(t, Tm)
                air = SurfaceForcing(sw_down = sun, lw_down = 300.0 + 0.5 * (T_out - 273.15),
                                     T_air = T_out, q_air = q_out, U = 3.0, z_ref = 10.0)
                r = internal_gains(g, t, dt; T_air = zone_air_temperature(z))
                f = ZoneForcing(r; T_out = T_out, q_out = q_out, Q_solar = 0.4 * sun,
                                irradiance = sun, surface = air, Q_hvac = 50.0)
                E0 = zone_heat_content(z); q0 = zone_specific_humidity(z)
                b, hb = step_zone!(z, f, dt, hv)
                rel = building_heat_release(z, f, b; dt = dt, gains = r, hvac = hv,
                                            hvac_budget = hb, q_before = q0)
                # storage measured independently of the budget
                dE = (zone_heat_content(z) - E0) / dt
                dL = latent_heat_vapor(b.T_air) * z.ρ_air * z.volume *
                     (zone_specific_humidity(z) - q0) / dt
                lhs = total_heat_release(rel) + dE + dL
                rhs = energy_consumption(rel) + rel.Q_solar + rel.Q_prescribed
                gross = abs(total_heat_release(rel)) + abs(dE) + abs(dL) + rhs
                gross_max = max(gross_max, gross)
                worst = max(worst, abs(lhs - rhs) / gross)
                worst_E = max(worst_E, abs(rel.storage_sensible - dE) / gross)
                # consumption is what the gains and plant say it is
                n_cons_bad += !(energy_consumption(rel) ≈ r.people_sensible + r.people_latent +
                                r.lighting + r.equipment + hb.P_heating + hb.P_cooling)
                seen[:heat] |= hb.Q_heating > 0
                seen[:cool] |= hb.Q_cooling > 0
                seen[:lost] |= rel.Q_lost > 0
                seen[:vent_lat] |= rel.Q_ventilation_latent != 0
            end
        end
        @printf("  first law, 8 days of 15-min steps: max |release + storage - consumption - sun| = %.2e (rel. to gross, max gross %.0f W); storage term vs zone_heat_content %.2e\n",
                worst, gross_max, worst_E)
        @test all(values(seen))
        @test n_cons_bad == 0
        @test worst < 1e-10
        @test worst_E < 1e-10
    end

    @testset "Periodic steady state: energy consumed = heat released" begin
        # Constant-schedule gains and a repeating day: once the building is
        # back in the same state every midnight, a day's release must equal
        # a day's consumption plus the transmitted sun.
        g = InternalGains(; floor_area = 48.0, floor_area_per_person = 12.0,
                          lighting_power_density = 8.0, equipment_power_density = 10.0,
                          lighting_return_air_fraction = 0.2,
                          occupancy = WeeklySchedule(DayProfile(8 => 0.0, 18 => 1.0, 24 => 0.0)),
                          lighting = WeeklySchedule(DayProfile(7 => 0.1, 19 => 1.0, 24 => 0.1)),
                          equipment = WeeklySchedule(DayProfile(8 => 0.3, 18 => 1.0, 24 => 0.3)))
        dt = 900.0
        for (label, Tm, hv) in (("winter, boiler", 278.15, IdealLoadsHVAC(T_heat = 294.15, T_cool = 299.15)),
                                ("summer, air-cooled AC", 300.15, IdealLoadsHVAC(T_heat = 292.15, T_cool = 297.15)))
            z = ah_full_zone(T = 295.15)
            day_rel = 0.0; day_cons = 0.0; day_sun = 0.0; day_direct = 0.0; day_fac = 0.0
            E_mid = zone_heat_content(z)
            ndays = 30
            for d in 1:ndays, k in 1:96
                t = (k - 1) * dt           # the same day every day
                T_out = ah_T_out(t, Tm); sun = ah_sun(t)
                air = SurfaceForcing(sw_down = sun, lw_down = 300.0 + 0.5 * (T_out - 273.15),
                                     T_air = T_out, U = 3.0, z_ref = 10.0)
                r = internal_gains(g, t, dt; T_air = zone_air_temperature(z))
                f = ZoneForcing(r; include_latent = false, T_out = T_out, Q_solar = 0.4 * sun,
                                irradiance = sun, surface = air)
                d == ndays && k == 1 && (E_mid = zone_heat_content(z))
                b, hb = step_zone!(z, f, dt, hv)
                if d == ndays
                    rel = building_heat_release(z, f, b; dt = dt, gains = r, hvac = hv,
                                                hvac_budget = hb)
                    day_rel += dt * total_heat_release(rel)
                    day_cons += dt * energy_consumption(rel)
                    day_sun += dt * rel.Q_solar
                    day_direct += dt * (anthropogenic_sensible_heat(rel) + anthropogenic_latent_heat(rel))
                    day_fac += dt * rel.Q_envelope_facets
                end
            end
            ΔE = zone_heat_content(z) - E_mid
            @printf("  %s, day %d: consumed %.3f kWh + sun %.3f kWh, released %.3f kWh (direct %.3f, via roof facet %.3f); ΔE_day = %.1f J; imbalance %.2e\n",
                    label, ndays, day_cons / 3.6e6, day_sun / 3.6e6, day_rel / 3.6e6,
                    day_direct / 3.6e6, day_fac / 3.6e6, ΔE, (day_rel - day_cons - day_sun) / day_cons)
            @test abs(ΔE) < 1e-4 * day_cons                      # periodic
            @test day_rel ≈ day_cons + day_sun rtol = 1e-4
        end
    end

    @testset "Steady heating: the release is the purchased energy" begin
        # boiler: windows + air exchange lose H ΔT, the flue Q(1/η - 1);
        # together Q/η, the fuel
        for (kind, η) in ((:boiler, 0.9), (:heat_pump, 3.0), (:electric, 1.0))
            z = ah_air_box(T = 293.15)
            H = zone_heat_loss_coefficient(z)
            hv = IdealLoadsHVAC(T_heat = 293.15, heating = kind)
            f = ZoneForcing(T_out = 268.15)
            b, hb = step_zone!(z, f, 600.0, hv)
            rel = building_heat_release(z, f, b; dt = 600.0, hvac = hv, hvac_budget = hb)
            Q = H * 25.0
            @test hb.Q_heating ≈ Q rtol = 1e-12
            @test rel.Q_windows + rel.Q_ventilation_sensible ≈ Q rtol = 1e-12
            purchased = kind === :boiler ? rel.P_fuel : rel.P_electric
            @test purchased ≈ Q / η rtol = 1e-12
            # the heat pump's negative exhaust: it takes Q(1 - 1/COP) out of the
            # outdoor air, so the city receives only what it paid for
            @test anthropogenic_sensible_heat(rel) ≈ Q / η rtol = 1e-12
            @test total_heat_release(rel) ≈ energy_consumption(rel) rtol = 1e-12
        end
    end

    @testset "Steady cooling: an AC adds only its power to the city" begin
        # T_out > T_set: the air exchange brings H ΔT in (release < 0), the
        # condenser rejects H ΔT (1 + 1/COP): net release = P_cooling
        z = ah_air_box(T = 297.15)
        H = zone_heat_loss_coefficient(z)
        hv = IdealLoadsHVAC(T_cool = 297.15, curves = nothing, cop_rated = 3.2)
        f = ZoneForcing(T_out = 307.15)
        b, hb = step_zone!(z, f, 600.0, hv)
        rel = building_heat_release(z, f, b; dt = 600.0, hvac = hv, hvac_budget = hb)
        @test rel.Q_windows + rel.Q_ventilation_sensible ≈ -H * 10 rtol = 1e-12
        @test rel.Q_exhaust_sensible ≈ H * 10 * (1 + 1 / 3.2) rtol = 1e-12
        @test anthropogenic_sensible_heat(rel) ≈ H * 10 / 3.2 rtol = 1e-10
        @test anthropogenic_sensible_heat(rel) ≈ rel.P_electric rtol = 1e-10
        @printf("  steady cooling at ΔT = 10 K: exhaust %.1f W, air exchange + glazing %.1f W, net %.1f W = P_cool %.1f W\n",
                rel.Q_exhaust_sensible, rel.Q_windows + rel.Q_ventilation_sensible,
                anthropogenic_sensible_heat(rel), rel.P_electric)
    end

    @testset "Latent gains: vented when the humidity is not advanced" begin
        g = InternalGains(; floor_area = 100.0, floor_area_per_person = 10.0,
                          equipment_power_density = 5.0, equipment_latent_fraction = 0.2)
        r = internal_gains(g, 0.0, 600.0; T_air = 296.15)
        @test r.latent > 0
        # no humidity: all latent gains leave with the exhaust air, at once
        z = ah_air_box(T = 296.15)
        f = ZoneForcing(r; include_latent = false, T_out = 290.15)
        b = step_zone!(z, f, 600.0)
        rel = building_heat_release(z, f, b; dt = 600.0, gains = r)
        @test rel.Q_latent_vented ≈ r.latent rtol = 1e-14
        @test rel.Q_ventilation_latent == 0 && rel.storage_latent == 0
        # with humidity: they enter the zone and leave by ventilation; at steady
        # state the ventilated latent heat equals the latent gain
        z = ah_air_box(T = 296.15, ach = 2.0, q = 0.008)
        local rel2
        for k in 1:400
            q0 = zone_specific_humidity(z)
            f2 = ZoneForcing(r; T_out = 296.15, q_out = 0.008)
            b2 = step_zone!(z, f2, 600.0)
            rel2 = building_heat_release(z, f2, b2; dt = 600.0, gains = r, q_before = q0)
        end
        @test rel2.Q_latent_vented ≈ 0 atol = 1e-12
        @test rel2.Q_ventilation_latent ≈ r.latent rtol = 1e-6
    end

    @testset "Facet envelope is released through the facet, not to the air" begin
        z = ah_full_zone(T = 294.15)
        air = SurfaceForcing(sw_down = 0.0, lw_down = 280.0, T_air = 275.15, U = 3.0, z_ref = 10.0)
        f = ZoneForcing(T_out = 275.15, surface = air, Q_internal = 500.0)
        b = step_zone!(z, f, 900.0)
        rel = building_heat_release(z, f, b; dt = 900.0)
        roof = z.envelope[2]
        @test rel.Q_envelope_facets ≈ -roof.area * facet_fluxes(roof).G rtol = 1e-14
        walls, slab = z.envelope[1], z.envelope[3]
        @test rel.Q_envelope_air ≈ -walls.area * walls.F_outer rtol = 1e-14
        @test rel.Q_envelope_ground ≈ -slab.area * slab.F_outer rtol = 1e-14
        @test anthropogenic_sensible_heat(rel) ≈ rel.Q_exhaust_sensible + rel.Q_ventilation_sensible +
              rel.Q_windows + rel.Q_lost + rel.Q_envelope_air rtol = 1e-14
        @test total_heat_release(rel) ≈ anthropogenic_sensible_heat(rel) +
              anthropogenic_latent_heat(rel) + rel.Q_envelope_facets + rel.Q_envelope_ground rtol = 1e-14
        @test_throws ArgumentError building_heat_release(z, f, b; dt = 900.0,
                                                         hvac = IdealLoadsHVAC())
    end

    # =====================================================================
    # Zones behind the facets of an urban scene: step_buildings!
    # =====================================================================

    @testset "step_buildings!: a zone with no scene facet is step_zone!" begin
        u, _, _ = ah_canyon()
        u2 = deepcopy(u)
        z = ah_full_zone(T = 293.15); z2 = deepcopy(z)
        hv = IdealLoadsHVAC(T_heat = 293.15, T_cool = 297.15)
        air = SurfaceForcing(lw_down = 300.0, T_air = 281.15, U = 2.0, z_ref = 30.0)
        worst_T = 0.0; worst_Q = 0.0; worst_s = 0.0
        for k in 1:12
            f = ZoneForcing(T_out = 281.15, Q_internal = 400.0, surface = air)
            fl, bs, hbs = step_buildings!(u, [z], [f], air, 900.0; hvac = [hv])
            b2, hb2 = step_zone!(z2, f, 900.0, hv)
            fl2 = step_urban_facets!(u2, air, 900.0)
            worst_T = max(worst_T, abs(bs[1].T_air - b2.T_air))
            worst_Q = max(worst_Q, abs(hbs[1].Q_heating - hb2.Q_heating))
            worst_s = max(worst_s, maximum(abs.(facet_temperatures(u) .- facet_temperatures(u2))))
        end
        @test worst_T < 1e-10 && worst_Q < 1e-8 && worst_s < 1e-10
    end

    @testset "step_buildings!: an isolated roof over a zone is step_zone! with a FacetElement" begin
        # Same zone, same roof: once as a scene facet stepped with the zone,
        # once as a FacetElement stepped by step_zone!. The only difference is
        # dRn/dT (analytic -4εσT³ in the scene, a δT = 0.01 K difference in
        # step_surface!), as in the #28 isolated-facet test.
        sc, vf = ah_isolated_roof(48.0)
        u = UrbanFacetScene(sc, vf; T = 290.0, options = f -> (T_interior = 293.15,))
        z = ah_zone_with(FacetElement(u.facets[1], 48.0))
        rf = UrbanFacet(:roof, example_roof_layers(); T = 290.0, T_interior = 293.15,
                        albedo = 0.15, emissivity = 0.95)
        zr = ah_zone_with(FacetElement(rf, 48.0))
        hv = IdealLoadsHVAC(T_heat = 290.15, T_cool = 297.15)
        dt = 900.0
        worst_T = 0.0; worst_Ts = 0.0; worst_Q = 0.0; Qsum = 0.0; n_on = 0
        for k in 1:96
            t = k * dt
            T_out = ah_T_out(t, 276.15)
            air = SurfaceForcing(lw_down = 290.0, T_air = T_out, U = 3.0, z_ref = 10.0)
            f = ZoneForcing(T_out = T_out, Q_internal = 300.0 + 2500 * (8 <= mod(t / 3600, 24) <= 18),
                            surface = air)
            _, bs, hbs = step_buildings!(u, [z], [f], air, dt; hvac = [hv])
            br, hbr = step_zone!(zr, f, dt, hv)
            worst_T = max(worst_T, abs(bs[1].T_air - br.T_air))
            worst_Ts = max(worst_Ts, abs(u.facets[1].T_skin - rf.T_skin))
            worst_Q = max(worst_Q, abs(hbs[1].Q_heating - hbr.Q_heating))
            Qsum += hbr.Q_heating; n_on += hbr.Q_heating > 0
        end
        @printf("  isolated roof + zone, 24 h: max |ΔT_air| = %.2e K, |ΔT_skin| = %.2e K, |ΔQ_heat| = %.2e W (mean load %.1f W, %d heating steps)\n",
                worst_T, worst_Ts, worst_Q, Qsum / 96, n_on)
        @test 0 < n_on < 96
        @test worst_Ts < 2e-4
        @test worst_T < 2e-4
        @test worst_Q < 1e-3 * Qsum / 96
    end

    @testset "step_buildings!: two buildings across a street — conservation, setpoints, maps" begin
        u, west, east = ah_canyon()
        # one heated office block, one free-floating block
        zw = scene_building_zone(u, west; window_area = 120.0, window_U = 2.8, infiltration_ach = 0.5)
        ze = scene_building_zone(u, east; window_area = 120.0, window_U = 2.8, infiltration_ach = 0.5)
        @test length(building_facets(u, west)) + length(building_facets(u, east)) ==
              count(f -> f.kind !== :ground, u.facets)
        # footprint 15 m × 60 m, 20 m tall: 6 floors of 3 m... 6.67 → 7 by volume/height
        @test zw.volume ≈ 15 * 60 * 20
        @test zw.floor_area ≈ zw.volume / 3
        g = office_internal_gains(zw.floor_area; lighting_return_air_fraction = 0.0)
        hv = [IdealLoadsHVAC(T_heat = 293.15, T_cool = 298.15), nothing]
        dt = 900.0
        ground = [i for (i, f) in enumerate(u.facets) if f.kind === :ground]
        worst_zone = 0.0; worst_col = 0.0; worst_set = 0.0; worst_map = 0.0; worst_law = 0.0
        n_heat = 0
        for k in 1:48
            t = 6 * 3600 + k * dt                 # Monday 06:00 onwards, a cold day
            T_out = ah_T_out(t, 275.15)
            air = SurfaceForcing(lw_down = 280.0, T_air = T_out, U = 2.0, z_ref = 30.0)
            E0 = zone_heat_content.((zw, ze))
            G0 = [column_heat_content(u.facets[i].column) for i in ground]
            rs = [internal_gains(g, t, dt; T_air = zone_air_temperature(z)) for z in (zw, ze)]
            fs = [ZoneForcing(r; include_latent = false, T_out = T_out) for r in rs]
            fl, bs, hbs = step_buildings!(u, [zw, ze], fs, air, dt; hvac = hv)
            for (z, b, e0) in zip((zw, ze), bs, E0)
                src = b.Q_exterior + b.Q_windows + b.Q_ventilation + b.Q_internal + b.Q_solar + b.Q_hvac
                gross = abs(b.Q_exterior) + abs(b.Q_windows) + abs(b.Q_ventilation) + b.Q_internal +
                        abs(b.Q_hvac) + abs(b.Q_surfaces)
                worst_zone = max(worst_zone, abs((zone_heat_content(z) - e0) / dt - src) / gross)
            end
            for (j, i) in enumerate(ground)
                f = u.facets[i]
                worst_col = max(worst_col, abs((column_heat_content(f.column) - G0[j]) / dt -
                                               (f.G - f.Q_interior)) / (abs(f.G) + abs(f.Q_interior) + 1))
            end
            if hbs[1].Q_heating > 0
                n_heat += 1; worst_set = max(worst_set, abs(bs[1].T_air - 293.15))
            end
            @test hbs[2] === nothing
            rel = [building_heat_release(z, f, b; dt = dt, gains = r,
                                         hvac = h, hvac_budget = hb)
                   for (z, f, b, r, h, hb) in zip((zw, ze), fs, bs, rs, hv, hbs)]
            for (z, rr, b, e0) in zip((zw, ze), rel, bs, E0)
                lhs = total_heat_release(rr) + (zone_heat_content(z) - e0) / dt
                rhs = energy_consumption(rr) + rr.Q_solar
                worst_law = max(worst_law, abs(lhs - rhs) / (abs(total_heat_release(rr)) + rhs))
            end
            maps = anthropogenic_heat_maps(u, [zw, ze], rel)
            direct = sum(anthropogenic_sensible_heat(r) + anthropogenic_latent_heat(r) for r in rel)
            worst_map = max(worst_map, abs(sum(maps.H .+ maps.LE) * 25.0 - direct) / abs(direct))
            @test all(maps.H[4:5, :] .!= 0)       # released into the street columns
        end
        @printf("  canyon + 2 zones, 12 h: zone budget %.2e, ground columns %.2e, first law %.2e, map sum %.2e (rel.); heating steps %d, max |T - T_heat| %.2e K; T_air east (free) %.2f K\n",
                worst_zone, worst_col, worst_law, worst_map, n_heat, worst_set, zone_air_temperature(ze))
        @test n_heat > 0
        @test worst_zone < 1e-10 && worst_col < 1e-10 && worst_law < 1e-10 && worst_map < 1e-12
        @test worst_set < 1e-9
        @test_throws ArgumentError step_buildings!(u, [zw, zw], [ZoneForcing(T_out = 280.0), ZoneForcing(T_out = 280.0)],
                                                   SurfaceForcing(lw_down = 280.0, T_air = 280.0), dt)
    end

    @testset "step_buildings!: implicit across the envelope — stable, converged" begin
        # A free-floating block through a cold night: the zone-air time
        # constant (C_air / Σ h A ≈ minutes) is far below the step, so the
        # coupling must be implicit. Reference: dt = 60 s.
        function night(dt)
            u, west, _ = ah_canyon(T0 = 290.0, Tin = 293.15)
            z = scene_building_zone(u, west; window_area = 120.0, window_U = 2.8)
            air = SurfaceForcing(lw_down = 270.0, T_air = 278.15, U = 2.0, z_ref = 30.0)
            Ts = Float64[]
            for k in 1:round(Int, 6 * 3600 / dt)
                _, bs, _ = step_buildings!(u, [z], [ZoneForcing(T_out = 278.15, Q_internal = 2000.0)],
                                           air, dt)
                push!(Ts, bs[1].T_air)
            end
            return Ts
        end
        ref = night(60.0); a = night(900.0); b = night(1800.0)
        e900 = abs(a[end] - ref[end]); e1800 = abs(b[end] - ref[end])
        mono = all(diff(a) .< 0)
        @printf("  free-floating zone, 6 h night: T_air(6 h) = %.4f K (dt = 60 s); error %.4f K at 900 s, %.4f K at 1800 s; monotone cooling: %s\n",
                ref[end], e900, e1800, mono)
        @test mono
        @test e900 < e1800 < 0.05
    end
end
