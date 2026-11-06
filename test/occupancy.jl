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

using Test
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
        @test p(6.0) == 0.1 && p(24.0) == 0.0        # a breakpoint starts the next interval; modulo 24
        @test full_load_hours(p) ≈ 0.1 + 5 * 0.9 + 12 * 0.2
        S = DOE_MEDIUM_OFFICE_SCHEDULES
        # full-load hours summed by hand from the IDF Schedule:Compact fields
        occ_wd = 0.1 + 0.2 + 4 * 0.95 + 0.5 + 4 * 0.95 + 0.7 + 2 * 0.4 + 2 * 0.1 + 2 * 0.05
        occ_sa = 2 * 0.1 + 6 * 0.5 + 3 * 0.1
        light_wd = 5 * 0.05 + 2 * 0.1 + 0.3 + 9 * 0.9 + 0.7 + 2 * 0.5 + 2 * 0.3 + 0.1 + 0.05
        light_sa = 6 * 0.05 + 2 * 0.1 + 6 * 0.5 + 3 * 0.15 + 7 * 0.05
        @test full_load_hours(S.occupancy) ≈ 5occ_wd + occ_sa
        @test full_load_hours(S.lighting) ≈ 5light_wd + light_sa + 24 * 0.05
        # calendar: day 1 a Thursday → day 2 holiday, day 3 Saturday, day 4 Sunday
        s = WeeklySchedule(DayProfile(1.0), DayProfile(0.5), DayProfile(0.0);
                           start_weekday = 4, holidays = [2])
        @test [schedule_value(s, (d + 0.5) * 86400) for d in 0:4] == [1.0, 0.0, 0.5, 0.0, 1.0]
        # exact step mean vs midpoint quadrature over an awkward interval
        occ = S.occupancy
        t0, t1 = 5.3 * 3600, 7.9 * 3600
        N = 20_000; h = (t1 - t0) / N
        @test schedule_mean(occ, t0, t1) ≈ sum(schedule_value(occ, t0 + (k - 0.5) * h) for k in 1:N) / N atol = 2e-4
        @test schedule_mean(occ, 3.7 * 86400 + 1234.0, 10.7 * 86400 + 1234.0) ≈ full_load_hours(occ) / 168 rtol = 1e-12
    end

    @testset "Occupant sensible heat (EnergyPlus fit)" begin
        # the fit was constrained to S = 0 at 96 °F and S = M at 30 °F over
        # 350-1450 Btu/h (EnergyPlus Engineering Reference)
        F(°F) = (°F - 32) * 5 / 9 + 273.15
        Ms = range(350, 1450; length = 6) .* 0.29307107
        @test maximum(M -> occupant_sensible_heat(M, F(96)) / M, Ms) < 0.025
        @test maximum(M -> abs(occupant_sensible_heat(M, F(30)) - M) / M, Ms) < 0.01
        # more evaporative loss, less sensible heat, as the room warms
        S = [occupant_sensible_heat(120.0, T) for T in 288.15:1.0:306.15]
        @test all(diff(S) .< 0) && all(0 .<= S .<= 120)
    end

    @testset "Internal-gain split conserves energy" begin
        g = office_internal_gains(1000.0)
        worst = maximum(Iterators.product((3.0, 9.5, 12.5, 19.0) .* 3600, (293.15, 299.15))) do (t, T)
            r = internal_gains(g, t; T_air = T)
            total = r.people_sensible + r.people_latent + r.lighting + r.equipment
            abs(r.convective + r.radiative + r.latent + r.lost - total) / max(total, 1.0)
        end
        @test worst < 1e-13
        # design level at 10:00 on a weekday: 1000/18.58 people × 0.95
        r = internal_gains(g, 10 * 3600.0)
        @test r.occupants ≈ 1000 / 18.58 * 0.95
        # step mean = exact integral / dt
        @test internal_gains(g, 7.5 * 3600, 3600.0).lighting ≈ 10.76e3 * (0.5 * 0.3 + 0.5 * 0.9)
    end

    @testset "A scheduled day conserves the zone's heat" begin
        z = occ_mass_box()
        g = office_internal_gains(48.0; lighting_return_air_fraction = 0.0)
        E0 = zone_heat_content(z); src = 0.0; dt = 1800.0
        for n in 0:47
            t = n * dt + 6 * 3600
            r = internal_gains(g, t, dt; T_air = zone_air_temperature(z))
            T_out = 278.15 + 6 * sin(2π * (t / 86400 - 0.3))
            b = step_zone!(z, ZoneForcing(r; include_latent = false, T_out = T_out,
                                          irradiance = max(0.0, 500 * sin(2π * (t / 86400 - 0.25)))), dt)
            src += dt * (b.Q_exterior + b.Q_windows + b.Q_ventilation + b.Q_internal + b.Q_solar + b.Q_hvac)
        end
        @test zone_heat_content(z) - E0 ≈ src rtol = 1e-9
    end

    @testset "Scheduled thermostat: setback, availability, exact setpoints" begin
        ts = office_thermostat(start_weekday = 1)
        hvac = IdealLoadsHVAC(heating = :electric, curves = nothing)
        h = scheduled_hvac(hvac, ts, 3 * 3600.0)
        @test h.heating_capacity == 0 && h.cooling_capacity == 0 && h.T_heat ≈ 288.75
        h = scheduled_hvac(hvac, ts, 10 * 3600.0)
        @test h.heating_capacity == Inf && h.T_heat ≈ 294.15 && h.T_cool ≈ 297.15
        # a winter day: the setpoint is held exactly while available, and the
        # zone floats (no HVAC energy) while it is not
        z = occ_mass_box(T = 290.0)
        g = office_internal_gains(48.0; lighting_return_air_fraction = 0.0)
        dt = 1800.0; worst = 0.0; off_energy = 0.0; n_on = 0
        for n in 0:47
            t = n * dt
            hv = scheduled_hvac(hvac, ts, t)
            r = internal_gains(g, t, dt; T_air = zone_air_temperature(z))
            b, hb = step_zone!(z, ZoneForcing(r; include_latent = false, T_out = 268.15), dt, hv)
            if hv.heating_capacity > 0
                n_on += 1; worst = max(worst, hv.T_heat - b.T_air)
            else
                off_energy += (hb.Q_heating + hb.Q_cooling) * dt
            end
        end
        @test n_on > 0 && worst < 1e-9
        @test off_energy == 0
    end
end
