# Phase 50 — BuildingEnergy: occupancy, lighting and equipment schedules,
# internal gains and scheduled thermostats (#33).
#
# What is checked against something the code does not contain:
#   * the DOE medium-office schedules against hand-summed full-load hours
#     of the IDF transcription, and the exact step-mean against quadrature;
#   * the occupant sensible-heat fit against the two anchor points the
#     EnergyPlus Engineering Reference says it was constrained to;
#   * the first law of the gain split and of a zone driven by it for a week;
#   * the periodic-mean theorem of a linear time-invariant system: the week
#     mean of the periodic free-float response to scheduled gains equals the
#     steady response to the week-mean gains (to round-off);
#   * setpoint schedules held exactly, and free float while unavailable.

using Test, Printf
using Aeolus
using Aeolus.BuildingEnergy
using Aeolus.UrbanSurface

# massive 8 × 6 × 2.7 m box, as in the #31/#32 tests
function occ_mass_box(; T = 294.15, ach = 0.5)
    wl = [WallLayer(:brick, 0.24), WallLayer(:gypsum_plaster, 0.015)]
    walls = EnvelopeElement(wl, 2 * (8 + 6) * 2.7 - 12; absorptance = 0.6, T = T, name = :walls)
    roof = EnvelopeElement(example_roof_layers(), 48.0; h_inside = 10.0, absorptance = 0.7, T = T,
                           name = :roof)
    floor = EnvelopeElement([WallLayer(:xps_insulation, 0.1), WallLayer(:concrete, 0.15)], 48.0;
                            exterior = :adiabatic, h_inside = 1 / 0.17, T = T, name = :floor)
    im = [InternalMass([WallLayer(:brick, 0.12)], 20.0; sides = 2, T = T)]
    return BuildingZone(; volume = 8 * 6 * 2.7, floor_area = 48.0, envelope = [walls, roof, floor],
                        internal_mass = im, window_area = 12.0, window_U = 2.8,
                        infiltration_ach = ach, T_air = T)
end

@testset "Era V — BuildingEnergy: occupancy schedules and internal gains (#33)" begin

    @testset "Day profiles and weekly schedules" begin
        p = DayProfile(6 => 0.0, 7 => 0.1, 12 => 0.9, 24 => 0.2)
        @test p(0.0) == 0.0 && p(5.999) == 0.0
        @test p(6.0) == 0.1                   # a breakpoint starts the next interval
        @test p(11.5) == 0.9 && p(23.99) == 0.2 && p(24.0) == 0.0   # modulo 24
        @test full_load_hours(p) ≈ 0.1 + 5 * 0.9 + 12 * 0.2
        @test_throws ArgumentError DayProfile(6 => 0.0, 20 => 1.0)      # must end at 24
        @test_throws ArgumentError DayProfile(12 => 0.0, 6 => 1.0, 24 => 0.0)

        S = DOE_MEDIUM_OFFICE_SCHEDULES
        # full-load hours summed by hand from the IDF Schedule:Compact fields
        occ_wd = 0.1 + 0.2 + 4 * 0.95 + 0.5 + 4 * 0.95 + 0.7 + 2 * 0.4 + 2 * 0.1 + 2 * 0.05
        occ_sa = 2 * 0.1 + 6 * 0.5 + 3 * 0.1
        light_wd = 5 * 0.05 + 2 * 0.1 + 0.3 + 9 * 0.9 + 0.7 + 2 * 0.5 + 2 * 0.3 + 0.1 + 0.05
        light_sa = 6 * 0.05 + 2 * 0.1 + 6 * 0.5 + 3 * 0.15 + 7 * 0.05
        equip_wd = 8 * 0.4 + 4 * 0.9 + 0.8 + 4 * 0.9 + 0.8 + 2 * 0.6 + 2 * 0.5 + 2 * 0.4
        equip_sa = 6 * 0.3 + 2 * 0.4 + 6 * 0.5 + 3 * 0.35 + 7 * 0.3
        @test full_load_hours(S.occupancy.weekday) ≈ occ_wd
        @test full_load_hours(S.occupancy) ≈ 5occ_wd + occ_sa
        @test full_load_hours(S.lighting) ≈ 5light_wd + light_sa + 24 * 0.05
        @test full_load_hours(S.equipment) ≈ 5equip_wd + equip_sa + 24 * 0.3
        @info @sprintf("DOE medium office, full-load hours per week: occupancy %.2f, lighting %.2f, equipment %.2f h",
                       full_load_hours(S.occupancy), full_load_hours(S.lighting),
                       full_load_hours(S.equipment))

        # calendar: day 1 a Thursday (start_weekday = 4) → day 3 is Saturday, day 4 Sunday
        s = WeeklySchedule(DayProfile(1.0), DayProfile(0.5), DayProfile(0.0);
                           start_weekday = 4, holidays = [2])
        @test schedule_value(s, 0.5 * 86400) == 1.0      # Thursday
        @test schedule_value(s, 1.5 * 86400) == 0.0      # Friday, but a holiday
        @test schedule_value(s, 2.5 * 86400) == 0.5      # Saturday
        @test schedule_value(s, 3.5 * 86400) == 0.0      # Sunday
        @test schedule_value(s, 4.5 * 86400) == 1.0      # Monday

        # exact step mean vs midpoint quadrature over awkward intervals
        occ = S.occupancy
        for (t0, t1) in ((5.3 * 3600, 7.9 * 3600), (86400 * 4 + 1000.0, 86400 * 6 + 5000.0),
                         (11.99 * 3600, 12.01 * 3600))
            N = 200_000; h = (t1 - t0) / N
            quad = sum(schedule_value(occ, t0 + (k - 0.5) * h) for k in 1:N) / N
            @test schedule_mean(occ, t0, t1) ≈ quad atol = 2e-5
        end
        # the mean over any whole week is full_load_hours / 168
        for t0 in (0.0, 3.7 * 86400 + 1234.0)
            @test schedule_mean(occ, t0, t0 + 7 * 86400) ≈ full_load_hours(occ) / 168 rtol = 1e-12
        end
    end

    @testset "Occupant sensible heat (EnergyPlus fit)" begin
        # the Engineering Reference: the fit was constrained to S = 0 at 96 °F
        # and S = M at 30 °F, over average adjusted metabolic rates of
        # 350-1450 Btu/h (103-425 W)
        F(°F) = (°F - 32) * 5 / 9 + 273.15
        worst0 = 0.0; worstM = 0.0
        for M in range(350, 1450; length = 12) .* 0.29307107
            worst0 = max(worst0, occupant_sensible_heat(M, F(96)) / M)
            worstM = max(worstM, abs(occupant_sensible_heat(M, F(30)) - M) / M)
            # physics: more evaporative loss, less sensible heat, as the room warms
            S = [occupant_sensible_heat(M, T) for T in 288.15:1.0:306.15]
            @test all(diff(S) .< 0)
            @test all(0 .<= S .<= M)
        end
        @info @sprintf("occupant sensible heat: S(96 °F)/M ≤ %.3f, |S(30 °F) − M|/M ≤ %.3f; S(120 W, 24 °C) = %.1f W",
                       worst0, worstM, occupant_sensible_heat(120, 297.15))
        @test worst0 < 0.025
        @test worstM < 0.01
        @test occupant_sensible_heat(0.0, 297.15) == 0.0
    end

    @testset "Internal-gain split conserves energy" begin
        g = office_internal_gains(1000.0)
        for t in (3.0, 9.5, 12.5, 19.0) .* 3600, T in (293.15, 299.15)
            r = internal_gains(g, t; T_air = T)
            total = r.people_sensible + r.people_latent + r.lighting + r.equipment
            @test r.convective + r.radiative + r.latent + r.lost ≈ total rtol = 1e-13
            @test r.people_sensible + r.people_latent ≈ 120 * r.occupants rtol = 1e-13
            @test r.lost ≈ 0.4 * r.lighting rtol = 1e-13
            @test r.convective >= 0 && r.radiative >= 0
        end
        # design level at 10:00 on a weekday: 1000/18.58 people × 0.95
        r = internal_gains(g, 10 * 3600.0)
        @test r.occupants ≈ 1000 / 18.58 * 0.95
        @test r.lighting ≈ 10.76e3 * 0.9 && r.equipment ≈ 10.76e3 * 0.9
        # step mean = exact integral / dt
        r1 = internal_gains(g, 7.5 * 3600, 3600.0)
        @test r1.lighting ≈ 10.76e3 * (0.5 * 0.3 + 0.5 * 0.9)
        @test_throws ArgumentError InternalGains(floor_area = 10.0, lighting_radiant_fraction = 0.7,
                                                 lighting_visible_fraction = 0.4)

        # ZoneForcing(r) = the same split through the zone's radiative fraction
        rr = InternalGainRates(0, 0, 0, 0, 0, 300.0, 700.0, 0.0, 0.0)
        z1 = occ_mass_box(); z2 = occ_mass_box()
        z2.radiative_fraction = 0.7
        b1 = step_zone!(z1, ZoneForcing(rr; T_out = 280.0), 3600.0)
        b2 = step_zone!(z2, ZoneForcing(T_out = 280.0, Q_internal = 1000.0), 3600.0)
        @test b1.T_air == b2.T_air && b1.Q_surfaces == b2.Q_surfaces
        @test_throws ArgumentError ZoneForcing(r; T_out = 280.0)   # latent without q_out
    end

    @testset "A scheduled week conserves the zone's heat" begin
        z = occ_mass_box()
        g = office_internal_gains(48.0; lighting_return_air_fraction = 0.0)
        E0 = zone_heat_content(z); src = 0.0; dt = 900.0
        for n in 0:(7 * 96 - 1)
            t = n * dt
            r = internal_gains(g, t, dt; T_air = zone_air_temperature(z))
            T_out = 278.15 + 6 * sin(2π * (t / 86400 - 0.3))
            b = step_zone!(z, ZoneForcing(r; include_latent = false, T_out = T_out,
                                          irradiance = max(0.0, 500 * sin(2π * (t / 86400 - 0.25)))), dt)
            src += dt * (b.Q_exterior + b.Q_windows + b.Q_ventilation + b.Q_internal + b.Q_solar + b.Q_hvac)
        end
        ΔE = zone_heat_content(z) - E0
        @info @sprintf("scheduled office week: ΔE = %.6e J, Σ sources·dt = %.6e J, rel. error %.1e",
                       ΔE, src, abs(ΔE - src) / abs(src))
        @test ΔE ≈ src rtol = 1e-9
    end

    @testset "Periodic-mean theorem: week mean = steady response to the mean gains" begin
        # With T_out constant and the occupant split frozen at a fixed air
        # temperature, the zone is linear and time-invariant; its periodic
        # response to weekly gains has the mean of the steady response to the
        # week-mean (convective, radiative) gains. Exact for the discrete
        # θ-scheme as well, so the tolerance is convergence to periodicity.
        g = office_internal_gains(48.0; lighting_return_air_fraction = 0.0)
        dt = 3600.0; T_out = 283.15; week = 168
        rates = [internal_gains(g, n * dt, dt; T_air = 296.15) for n in 0:(week - 1)]
        conv = sum(r -> r.convective, rates) / week
        rad = sum(r -> r.radiative, rates) / week

        z = occ_mass_box(T = 290.0)
        means = Float64[]
        for w in 1:40
            s = 0.0
            for n in 1:week
                b = step_zone!(z, ZoneForcing(rates[n]; include_latent = false, T_out = T_out), dt)
                s += b.T_air
            end
            push!(means, s / week)
            w > 3 && abs(means[end] - means[end - 1]) < 1e-11 && break
        end
        zs = occ_mass_box(T = 290.0)
        Ts = 0.0
        for n in 1:(40 * week)
            Ts = step_zone!(zs, ZoneForcing(; T_out = T_out, Q_internal = conv + rad,
                                            radiative_fraction = rad / (conv + rad)), dt).T_air
        end
        @info @sprintf("periodic week-mean T_air = %.9f K, steady T_air (mean gains) = %.9f K, Δ = %.1e K; mean gain %.1f W",
                       means[end], Ts, means[end] - Ts, conv + rad)
        @test abs(means[end] - Ts) < 1e-8
        # and the steady state is the heat-loss network: all gains leave by
        # the envelope, windows and air exchange (adiabatic floor)
        b = step_zone!(zs, ZoneForcing(; T_out = T_out, Q_internal = conv + rad,
                                       radiative_fraction = rad / (conv + rad)), dt)
        @test b.Q_exterior + b.Q_windows + b.Q_ventilation + b.Q_internal ≈ 0 atol = 1e-6
    end

    @testset "Scheduled thermostat: setback, availability, exact setpoints" begin
        ts = office_thermostat(start_weekday = 1)
        hvac = IdealLoadsHVAC(heating = :electric, curves = nothing)
        # Monday 03:00: off; 10:00 on at 21 °C / 24 °C; Saturday 20:00 off; Sunday off
        h = scheduled_hvac(hvac, ts, 3 * 3600.0)
        @test h.heating_capacity == 0 && h.cooling_capacity == 0 && h.T_heat ≈ 288.75
        h = scheduled_hvac(hvac, ts, 10 * 3600.0)
        @test h.heating_capacity == Inf && h.T_heat ≈ 294.15 && h.T_cool ≈ 297.15
        @test scheduled_hvac(hvac, ts, (5 * 24 + 20) * 3600.0).heating_capacity == 0
        @test scheduled_hvac(hvac, ts, (6 * 24 + 12) * 3600.0).heating_capacity == 0

        # a winter week: the setpoint is held exactly while available, and the
        # zone floats (no HVAC energy) while it is not
        z = occ_mass_box(T = 290.0)
        g = office_internal_gains(48.0; lighting_return_air_fraction = 0.0)
        dt = 900.0; E_heat = 0.0; worst = 0.0; off_energy = 0.0; below = 0
        for n in 0:(7 * 96 - 1)
            t = n * dt
            hv = scheduled_hvac(hvac, ts, t)
            r = internal_gains(g, t, dt; T_air = zone_air_temperature(z))
            b, hb = step_zone!(z, ZoneForcing(r; include_latent = false, T_out = 268.15), dt, hv)
            E_heat += hb.Q_heating * dt
            if hv.heating_capacity > 0
                worst = max(worst, hv.T_heat - b.T_air)
            else
                off_energy += (hb.Q_heating + hb.Q_cooling) * dt
                b.T_air < 294.15 - 0.5 && (below += 1)
            end
        end
        @info @sprintf("office winter week at −5 °C: heating %.1f kWh; worst undershoot while on %.2e K; %d off-hours quarter-steps below 20.65 °C",
                       E_heat / 3.6e6, worst, below)
        @test worst < 1e-9
        @test off_energy == 0
        @test below > 0                       # the building cools at night/weekend
        @test E_heat > 0
    end
end
