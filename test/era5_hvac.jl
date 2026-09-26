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

    @testset "Published DX curves: rating point, sensitivity, clamping" begin
        x0, y0 = DX_RATING_POINT.T_wb_indoor, DX_RATING_POINT.T_db_outdoor
        # both EnergyPlus curve sets are normalised to 1 at the AHRI A point;
        # checking it guards the transcription of all twelve coefficients
        for c in (DX_CURVES_WINDOW_AC, DX_CURVES_PACKAGED_AC)
            @test c.capacity(x0, y0) ≈ 1 atol = 1e-3
            @test c.eir(x0, y0) ≈ 1 atol = 1e-3
            # COP falls and capacity falls as the condenser air warms
            ys = range(18.0, 46.11, length = 30)
            @test all(diff([c.eir(x0, y) for y in ys]) .> 0)
            # capacity falls above ~20 °C (the packaged-unit fit peaks at
            # 19.6 °C: ∂CAP/∂y = c₄ + 2c₅y + c₆x = 0 there, a property of the
            # published fit)
            @test all(diff([c.capacity(x0, y) for y in range(20.0, 46.11, length = 30)]) .< 0)
            # outside the fitted domain the inputs are clamped
            @test c.eir(x0, 60.0) == c.eir(x0, 46.11111)
            @test c.capacity(5.0, y0) == c.capacity(12.77778, y0)
        end
        w, pk = DX_CURVES_WINDOW_AC, DX_CURVES_PACKAGED_AC
        r46w, r46p = w.eir(x0, y0) / w.eir(x0, 46.11111), pk.eir(x0, y0) / pk.eir(x0, 46.11111)
        rBw, rBp = w.eir(x0, y0) / w.eir(x0, 27.78), pk.eir(x0, y0) / pk.eir(x0, 27.78)
        @printf("  DX COP(46.1 °C)/COP(35 °C): window %.4f, packaged %.4f\n", r46w, r46p)
        @printf("  DX COP(27.8 °C)/COP(35 °C) (AHRI B/A): window %.4f, packaged %.4f\n", rBw, rBp)
        # two independently fitted units agree on the outdoor sensitivity
        @test abs(r46w - r46p) < 0.01
        @test 0.70 < r46w < 0.85
        # COP through the HVAC object, in K
        h = IdealLoadsHVAC(cop_rated = 3.2)
        @test cooling_cop(h, x0 + 273.15, y0 + 273.15) ≈ 3.2 / w.eir(x0, y0) rtol = 1e-14
        @test cooling_cop(h, x0 + 273.15, 46.11111 + 273.15) ≈ 3.2 / w.eir(x0, 46.11111) rtol = 1e-12
        @test cooling_cop(IdealLoadsHVAC(curves = nothing), 290.0, 320.0) == 3.0
        h = IdealLoadsHVAC(cooling_capacity = 5000.0, curves = DX_CURVES_PACKAGED_AC)
        @test available_cooling_capacity(h, x0 + 273.15, 40.0 + 273.15) ≈ 5000 * pk.capacity(x0, 40.0)
        @test available_cooling_capacity(IdealLoadsHVAC(), 290.0, 310.0) == Inf
        @test_throws ArgumentError IdealLoadsHVAC(T_heat = 300.0, T_cool = 295.0)
        @test_throws ArgumentError IdealLoadsHVAC(heating = :boiler, heating_efficiency = 1.2)
        @test_throws ArgumentError IdealLoadsHVAC(condenser = :river)
    end

    @testset "Condenser inlet temperature (EnergyPlus evaporative condenser)" begin
        p = 101325.0; T = 308.15; q = 0.010       # hot and fairly dry: 35 °C, 28 % RH
        Twb = wet_bulb_temperature(T, p, q)
        @test condenser_inlet_temperature(IdealLoadsHVAC(), T, nothing) == T
        @test condenser_inlet_temperature(IdealLoadsHVAC(condenser = :evaporative,
                                                         evaporative_effectiveness = 1.0), T, q) ≈ Twb
        @test condenser_inlet_temperature(IdealLoadsHVAC(condenser = :evaporative,
                                                         evaporative_effectiveness = 0.0), T, q) ≈ T
        he = IdealLoadsHVAC(condenser = :evaporative)
        Tc = condenser_inlet_temperature(he, T, q)
        @test Tc ≈ Twb + 0.1 * (T - Twb) rtol = 1e-14
        @test_throws ArgumentError condenser_inlet_temperature(he, T, nothing)
        # the evaporatively cooled condenser runs at a higher COP in dry heat
        Twb_in = 19.44444 + 273.15
        ga = cooling_cop(IdealLoadsHVAC(), Twb_in, T)
        ge = cooling_cop(he, Twb_in, Tc)
        @printf("  35 °C / q = 0.010: T_wb = %.2f °C, evaporative condenser inlet %.2f °C, COP %.3f vs air-cooled %.3f\n",
                Twb - 273.15, Tc - 273.15, ge, ga)
        @test ge > ga
    end

    @testset "Ideal load equals the steady heat-loss network" begin
        # massless box: H (T_out - T_set) + Φ_conv is removed exactly in one step
        z = hvac_air_box(T = 299.15)
        H = zone_heat_loss_coefficient(z)
        h = IdealLoadsHVAC(T_heat = 293.15, T_cool = 299.15)
        f = ZoneForcing(T_out = 308.15, Q_internal = 800.0)
        # (no interior surface: the radiative part of the gain also goes to the air)
        b, hb = step_zone!(z, f, 600.0, h)
        @test b.T_air ≈ 299.15 atol = 1e-10
        @test hb.Q_cooling_sensible ≈ H * (308.15 - 299.15) + 800.0 rtol = 1e-12
        @test b.Q_hvac ≈ -hb.Q_cooling_sensible rtol = 1e-14
        # heating side
        z = hvac_air_box(T = 293.15)
        H = zone_heat_loss_coefficient(z)      # ρ c_p of this zone (colder air, denser)
        b, hb = step_zone!(z, ZoneForcing(T_out = 268.15, Q_internal = 300.0), 600.0, h)
        @test b.T_air ≈ 293.15 atol = 1e-10
        @test hb.Q_heating ≈ H * (293.15 - 268.15) - 300.0 rtol = 1e-12
        # massive box under constant forcing: converges to U·A network load
        z = hvac_mass_box(T = 299.15)
        H = zone_heat_loss_coefficient(z)
        local b, hb
        for k in 1:6000     # 25 days of 6-minute steps
            b, hb = step_zone!(z, ZoneForcing(T_out = 306.15), 360.0, h)
        end
        Q_net = H * (306.15 - 299.15)
        @printf("  steady cooling load massive box: %.3f W, network H ΔT = %.3f W (H = %.2f W/K)\n",
                hb.Q_cooling_sensible, Q_net, H)
        @test hb.Q_cooling_sensible ≈ Q_net rtol = 1e-6
    end

    @testset "Setpoint holding and conservation over hot and cold days" begin
        h = IdealLoadsHVAC(T_heat = 293.15, T_cool = 299.15)
        z = hvac_mass_box(T = 296.15)
        dt = 600.0
        E0 = zone_heat_content(z); src = 0.0; gross = 0.0
        on_err = 0.0; band_ok = true; n_heat = n_cool = n_off = 0; off_ok = true
        for k in 1:(6 * 144)
            t = k * dt
            day = sin(2π * (t / 86400 - 0.25))
            # three hot days, then three cold ones
            T_out = (k <= 3 * 144 ? 301.15 : 276.15) + 7 * day
            f = ZoneForcing(T_out = T_out, Q_internal = 400.0 + 300 * max(day, 0),
                            Q_solar = max(0, 700 * day), irradiance = max(0, 600 * day))
            b, hb = step_zone!(z, f, dt, h)
            src += dt * (b.Q_exterior + b.Q_windows + b.Q_ventilation + b.Q_internal +
                         b.Q_solar + b.Q_hvac)
            gross += dt * (abs(b.Q_exterior) + abs(b.Q_windows) + abs(b.Q_ventilation) +
                           b.Q_internal + b.Q_solar + abs(b.Q_hvac))
            band_ok &= (293.15 - 1e-9 <= b.T_air <= 299.15 + 1e-9)
            if hb.Q_heating > 0
                n_heat += 1; on_err = max(on_err, abs(b.T_air - 293.15))
            elseif hb.Q_cooling > 0
                n_cool += 1; on_err = max(on_err, abs(b.T_air - 299.15))
            else
                n_off += 1
                off_ok &= (b.Q_hvac == 0)
            end
            @assert hb.Q_unmet == 0
        end
        cons = abs(zone_heat_content(z) - E0 - src)
        @printf("  setpoint holding: %d heating, %d cooling, %d dead-band steps; max |T - T_set| when on = %.2e K; |ΔE - ∫sources| = %.2e J (gross %.2e J)\n",
                n_heat, n_cool, n_off, on_err, cons, gross)
        @test band_ok && off_ok
        @test on_err < 1e-9
        @test n_heat > 0 && n_cool > 0 && n_off > 0
        @test cons < 1e-12 * gross
    end
end
