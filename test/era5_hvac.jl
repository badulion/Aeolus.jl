# Phase 50 — BuildingEnergy: HVAC demand and exhaust heat (#32).
#
# Validation is against solutions the code does not contain: the implicit
# vapour balance (discrete relaxation factor and steady state), the steady
# heat-loss network for the ideal load, exact setpoint holding, the first
# law of the plant, the published EnergyPlus DX performance curves at their
# AHRI rating point, the EnergyPlus evaporative-condenser inlet temperature,
# and the linear AC–canyon feedback loop.

using Test, Printf
using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface
using Aeolus.MoistThermodynamics

const BEH = Aeolus.BuildingEnergy

# A massless box: air node + glazing + air exchange only (no columns), so the
# steady and step-by-step behaviour are scalar and known in closed form.
hvac_air_box(; T = 299.15, q = nothing, ach = 0.5, UA = 150.0) =
    BuildingZone(; volume = 300.0, floor_area = 100.0, window_area = 1.0, window_U = UA,
                 infiltration_ach = ach, T_air = T, q_air = q)

# A massive 8 m × 6 m × 2.7 m box (brick walls, roof, insulated slab, a brick
# partition), as in the #31 tests.
function hvac_mass_box(; T = 297.15)
    wl = [WallLayer(:brick, 0.24), WallLayer(:gypsum_plaster, 0.015)]
    walls = EnvelopeElement(wl, 2 * (8 + 6) * 2.7 - 12; absorptance = 0.6, T = T, name = :walls)
    roof = EnvelopeElement(example_roof_layers(), 48.0; h_inside = 10.0, absorptance = 0.7, T = T,
                           name = :roof)
    floor = EnvelopeElement([WallLayer(:xps_insulation, 0.1), WallLayer(:concrete, 0.15)], 48.0;
                            exterior = :adiabatic, h_inside = 1 / 0.17, T = T, name = :floor)
    im = [InternalMass([WallLayer(:brick, 0.12)], 20.0; sides = 2, T = T)]
    return BuildingZone(; volume = 8 * 6 * 2.7, floor_area = 48.0, envelope = [walls, roof, floor],
                        internal_mass = im, window_area = 12.0, window_U = 2.8,
                        infiltration_ach = 0.5, T_air = T)
end

@testset "Era V — BuildingEnergy: HVAC demand and exhaust heat (#32)" begin

    @testset "Indoor humidity: implicit vapour balance" begin
        p = 101325.0
        # default: 50 % relative humidity at the initial temperature
        z = hvac_air_box(T = 297.15)
        @test zone_relative_humidity(z) ≈ 0.5 atol = 1e-12
        # without an outdoor humidity the vapour is not advanced
        q0 = zone_specific_humidity(z)
        b = step_zone!(z, ZoneForcing(T_out = 300.0), 600.0)
        @test zone_specific_humidity(z) == q0 && b.q_air == q0 && b.W_ventilation == 0
        @test_throws ArgumentError ZoneForcing(T_out = 300.0, Q_latent = 100.0)

        # relaxation to outdoor humidity: implicit Euler factor 1/(1 + n dt) per step
        n = 2.0 / 3600; dt = 300.0; q_out = 0.015
        z = hvac_air_box(T = 297.15, q = 0.008, ach = 2.0)
        for k in 1:10
            step_zone!(z, ZoneForcing(T_out = 297.15, q_out = q_out), dt)
        end
        @test zone_specific_humidity(z) ≈ q_out + (0.008 - q_out) / (1 + n * dt)^10 rtol = 1e-12

        # steady state with a latent source: q = q_out + (Φ_lat/L_v)/(ρ V n)
        Φ_lat = 400.0
        z = hvac_air_box(T = 297.15, q = 0.010, ach = 0.5)
        local b
        for k in 1:2000
            b = step_zone!(z, ZoneForcing(T_out = 297.15, q_out = 0.010, Q_latent = Φ_lat), 600.0)
        end
        M = z.ρ_air * z.volume
        q_ss = 0.010 + Φ_lat / latent_heat_vapor(297.15) / (M * 0.5 / 3600)
        @printf("  humidity: steady q = %.6f, analytic %.6f kg/kg (ρV = %.1f kg)\n",
                zone_specific_humidity(z), q_ss, M)
        @test zone_specific_humidity(z) ≈ q_ss rtol = 1e-9
        # vapour budget closes: ρV Δq/Δt = W_ventilation + W_internal + W_hvac
        qa = zone_specific_humidity(z)
        b = step_zone!(z, ZoneForcing(T_out = 297.15, q_out = 0.012, Q_latent = Φ_lat), 600.0)
        @test M * (b.q_air - qa) / 600 ≈ b.W_ventilation + b.W_internal + b.W_hvac rtol = 1e-12
        @test b.W_internal ≈ Φ_lat / latent_heat_vapor(297.15) rtol = 1e-12
    end
end
