# Phase 50 — BuildingEnergy: HVAC demand and exhaust heat (#32).
#
# Validation is against solutions the code does not contain: the implicit
# vapour balance (discrete relaxation factor and steady state), the steady
# heat-loss network for the ideal load, exact setpoint holding, the first
# law of the plant, the published EnergyPlus DX performance curves at their
# AHRI rating point, the EnergyPlus evaporative-condenser inlet temperature,
# and the linear AC–canyon feedback loop.

using Test
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

isdefined(@__MODULE__, :hvac_mass_box) || include(joinpath(@__DIR__, "support", "building_box.jl"))

@testset "Era V — BuildingEnergy: HVAC demand and exhaust heat (#32)" begin

    @testset "Indoor humidity: implicit vapour balance" begin
        # relaxation to outdoor humidity: implicit Euler factor 1/(1 + n dt) per step
        n = 2.0 / 3600; dt = 300.0; q_out = 0.015
        z = hvac_air_box(T = 297.15, q = 0.008, ach = 2.0)
        for k in 1:10
            step_zone!(z, ZoneForcing(T_out = 297.15, q_out = q_out), dt)
        end
        @test zone_specific_humidity(z) ≈ q_out + (0.008 - q_out) / (1 + n * dt)^10 rtol = 1e-12
        # vapour budget closes: ρV Δq/Δt = W_ventilation + W_internal + W_hvac
        Φ_lat = 400.0
        M = z.ρ_air * z.volume
        qa = zone_specific_humidity(z)
        b = step_zone!(z, ZoneForcing(T_out = 297.15, q_out = 0.012, Q_latent = Φ_lat), 600.0)
        @test M * (b.q_air - qa) / 600 ≈ b.W_ventilation + b.W_internal + b.W_hvac rtol = 1e-12
        @test b.W_internal ≈ Φ_lat / latent_heat_vapor(297.15) rtol = 1e-12
    end

    @testset "Published DX curves and evaporative condenser" begin
        x0, y0 = DX_RATING_POINT.T_wb_indoor, DX_RATING_POINT.T_db_outdoor
        # both EnergyPlus curve sets are normalised to 1 at the AHRI A point;
        # checking it guards the transcription of all twelve coefficients
        @test all(c -> abs(c.capacity(x0, y0) - 1) < 1e-3 && abs(c.eir(x0, y0) - 1) < 1e-3,
                  (DX_CURVES_WINDOW_AC, DX_CURVES_PACKAGED_AC))
        w, pk = DX_CURVES_WINDOW_AC, DX_CURVES_PACKAGED_AC
        # two independently fitted units agree on the outdoor sensitivity
        r46w, r46p = w.eir(x0, y0) / w.eir(x0, 46.11111), pk.eir(x0, y0) / pk.eir(x0, 46.11111)
        @test abs(r46w - r46p) < 0.01 && 0.70 < r46w < 0.85
        h = IdealLoadsHVAC(cop_rated = 3.2)
        @test cooling_cop(h, x0 + 273.15, y0 + 273.15) ≈ 3.2 / w.eir(x0, y0) rtol = 1e-14
        # evaporative condenser: T_wb + (1 − ε)(T − T_wb), ε = 0.9
        T = 308.15; q = 0.010
        Twb = wet_bulb_temperature(T, 101325.0, q)
        he = IdealLoadsHVAC(condenser = :evaporative)
        @test condenser_inlet_temperature(he, T, q) ≈ Twb + 0.1 * (T - Twb) rtol = 1e-14
    end

    @testset "Ideal load equals the steady heat-loss network" begin
        # massless box: H (T_out - T_set) + Φ_conv is removed exactly in one step
        z = hvac_air_box(T = 299.15)
        H = zone_heat_loss_coefficient(z)
        h = IdealLoadsHVAC(T_heat = 293.15, T_cool = 299.15)
        b, hb = step_zone!(z, ZoneForcing(T_out = 308.15, Q_internal = 800.0), 600.0, h)
        @test b.T_air ≈ 299.15 atol = 1e-10
        @test hb.Q_cooling_sensible ≈ H * (308.15 - 299.15) + 800.0 rtol = 1e-12
        z = hvac_air_box(T = 293.15)
        H = zone_heat_loss_coefficient(z)
        b, hb = step_zone!(z, ZoneForcing(T_out = 268.15, Q_internal = 300.0), 600.0, h)
        @test hb.Q_heating ≈ H * (293.15 - 268.15) - 300.0 rtol = 1e-12
    end

    @testset "Setpoint holding and conservation over a hot and a cold day" begin
        h = IdealLoadsHVAC(T_heat = 293.15, T_cool = 299.15)
        z = hvac_mass_box(T = 296.15)
        dt = 1800.0
        E0 = zone_heat_content(z); src = 0.0; gross = 0.0
        on_err = 0.0; band_ok = true; n_heat = n_cool = 0
        for k in 1:96
            day = sin(2π * (k * dt / 86400 - 0.25))
            # one hot day, then one cold one (−7 °C mean, so the massive box
            # reaches the heating setpoint within a day; the 6-day original used 3 °C)
            T_out = (k <= 48 ? 301.15 : 266.15) + 7 * day
            f = ZoneForcing(T_out = T_out, Q_internal = 400.0 + 300 * max(day, 0),
                            Q_solar = max(0, 700 * day), irradiance = max(0, 600 * day))
            b, hb = step_zone!(z, f, dt, h)
            src += dt * (b.Q_exterior + b.Q_windows + b.Q_ventilation + b.Q_internal +
                         b.Q_solar + b.Q_hvac)
            gross += dt * (abs(b.Q_exterior) + abs(b.Q_windows) + abs(b.Q_ventilation) +
                           b.Q_internal + b.Q_solar + abs(b.Q_hvac))
            band_ok &= (293.15 - 1e-9 <= b.T_air <= 299.15 + 1e-9)
            hb.Q_heating > 0 && (n_heat += 1; on_err = max(on_err, abs(b.T_air - 293.15)))
            hb.Q_cooling > 0 && (n_cool += 1; on_err = max(on_err, abs(b.T_air - 299.15)))
        end
        @test band_ok && on_err < 1e-9 && n_heat > 0 && n_cool > 0
        @test abs(zone_heat_content(z) - E0 - src) < 1e-12 * gross
    end

    @testset "Capacity limit: unmet load" begin
        z = hvac_air_box(T = 299.15)
        h = IdealLoadsHVAC(T_cool = 299.15, cooling_capacity = 1500.0)
        T_wb = wet_bulb_temperature(299.15, z.p_ref, zone_specific_humidity(z))
        b, hb = step_zone!(z, ZoneForcing(T_out = 313.15, Q_internal = 1000.0), 600.0, h)
        @test hb.Q_cooling_sensible ≈ 1500 * DX_CURVES_WINDOW_AC.capacity(T_wb - 273.15, 40.0) rtol = 1e-12
        @test hb.Q_unmet < 0 && b.T_air > 299.15
        # the unmet load is what would have held the setpoint
        _, hb2 = step_zone!(hvac_air_box(T = 299.15), ZoneForcing(T_out = 313.15, Q_internal = 1000.0),
                            600.0, IdealLoadsHVAC(T_cool = 299.15))
        @test hb2.Q_cooling_sensible ≈ hb.Q_cooling_sensible - hb.Q_unmet rtol = 1e-12
    end

    @testset "Exhaust heat: first law of the plant, Q_exh = Q_cool (1 + 1/COP)" begin
        f = ZoneForcing(T_out = 308.15, Q_internal = 1200.0, q_out = 0.014)
        _, hb = step_zone!(hvac_air_box(T = 299.15), f, 600.0, IdealLoadsHVAC(T_cool = 299.15))
        @test hb.Q_cooling > 0
        @test hb.Q_exhaust_sensible ≈ hb.Q_cooling * (1 + 1 / hb.COP) rtol = 1e-14
        # evaporative: 85 % latent, as evaporated water
        _, hbe = step_zone!(hvac_air_box(T = 299.15), f, 600.0,
                            IdealLoadsHVAC(T_cool = 299.15, condenser = :evaporative))
        Q_rej = hbe.Q_cooling * (1 + 1 / hbe.COP)
        @test hbe.Q_exhaust_latent ≈ 0.85 * Q_rej rtol = 1e-14
        @test hbe.W_evaporated ≈ hbe.Q_exhaust_latent / latent_heat_vapor(hbe.T_condenser) rtol = 1e-14
        # the general identity, over a mixed day with dehumidification and heating
        z = hvac_mass_box(T = 297.15)
        h = IdealLoadsHVAC(T_heat = 294.15, T_cool = 297.15, dehumidify = true, condenser = :evaporative)
        worst = 0.0
        for k in 1:48
            day = sin(2π * (k / 48 - 0.25))
            _, hk = step_zone!(z, ZoneForcing(T_out = 293.15 + 10day, Q_internal = 500.0,
                                              Q_latent = 150.0, q_out = 0.013), 1800.0, h)
            lhs = hk.Q_exhaust_sensible + hk.Q_exhaust_latent
            rhs = hk.P_heating + hk.P_cooling + hk.Q_cooling - hk.Q_heating
            worst = max(worst, abs(lhs - rhs) / max(1.0, abs(rhs)))
        end
        @test worst < 1e-13
    end

    @testset "Dehumidification to RH_max; latent coil load" begin
        f = ZoneForcing(T_out = 303.15, q_out = 0.018, Q_internal = 500.0, Q_latent = 300.0)
        h = IdealLoadsHVAC(T_cool = 297.15, dehumidify = true, RH_max = 0.6)
        z = hvac_air_box(T = 297.15, ach = 1.0)
        local b, hb
        for k in 1:60      # ρV n ≈ 1/h: 10 h is ≈ 10 e-folds, steady to the tolerance
            b, hb = step_zone!(z, f, 600.0, h)
        end
        M, n = z.ρ_air * z.volume, 1.0 / 3600
        q_max = specific_humidity_from_relative_humidity(297.15, z.p_ref, 0.6)
        @test zone_relative_humidity(z) ≈ 0.6 atol = 1e-12
        # steady vapour balance: the coil condenses what ventilation and people bring
        W_ss = M * n * (0.018 - q_max) + 300.0 / latent_heat_vapor(297.15)
        @test hb.W_condensed ≈ W_ss rtol = 1e-9
    end

    @testset "AC feedback: exhaust heat amplifies the canyon heat island" begin
        # Canyon air (C_c) exchanging with rural air T_r through K, heated by Q0
        # and by the exhaust of an air-cooled AC whose zone sees T_out = T_c.
        # Constant COP: loop gain g = (1 + 1/COP) H / K and
        #   T_c = [T_r + (Q0 + (1 + 1/COP)(G - H T_set))/K] / (1 - g).
        T_r, T_set, Q0, K, G, C_c = 303.15, 299.15, 3000.0, 1000.0, 2000.0, 2e6
        z = hvac_air_box(T = T_set)
        H = zone_heat_loss_coefficient(z)
        h = IdealLoadsHVAC(T_cool = T_set, curves = nothing, cop_rated = 3.0)
        Tcan = T_r; dt = 6000.0     # canyon relaxation time C_c/(K(1-g)) ≈ 40 min
        for k in 1:60
            _, hbk = step_zone!(z, ZoneForcing(T_out = Tcan, Q_internal = G), dt, h)
            Tcan = (C_c / dt * Tcan + K * T_r + Q0 + hbk.Q_exhaust_sensible) / (C_c / dt + K)
        end
        a = 1 + 1 / 3.0
        @test Tcan ≈ (T_r + (Q0 + a * (G - H * T_set)) / K) / (1 - a * H / K) rtol = 1e-10
    end

    @testset "ISO 13790 §C.4.2 ideal load holds the 5R1C air node on the setpoint" begin
        iso2 = iso13790_zone(hvac_mass_box(T = 299.15))
        Φ, θa, _, _ = iso13790_ideal_load!(iso2, 310.0, 3600.0; θ_heat = 293.15, θ_cool = 299.15,
                                          Φ_int = 500.0)
        @test θa ≈ 299.15 atol = 1e-9
        @test Φ < 0
    end
end
