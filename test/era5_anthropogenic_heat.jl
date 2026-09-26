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
#     steady cooling, where the net release is the purchased energy alone.

using Test, Printf
using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface
using Aeolus.SurfaceExchange
using Aeolus.MoistThermodynamics

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
        worst = 0.0; worst_E = 0.0; gross_max = 0.0
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
                @assert energy_consumption(rel) ≈ r.people_sensible + r.people_latent +
                        r.lighting + r.equipment + hb.P_heating + hb.P_cooling
                seen[:heat] |= hb.Q_heating > 0
                seen[:cool] |= hb.Q_cooling > 0
                seen[:lost] |= rel.Q_lost > 0
                seen[:vent_lat] |= rel.Q_ventilation_latent != 0
            end
        end
        @printf("  first law, 8 days of 15-min steps: max |release + storage - consumption - sun| = %.2e (rel. to gross, max gross %.0f W); storage term vs zone_heat_content %.2e\n",
                worst, gross_max, worst_E)
        @test all(values(seen))
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
end
