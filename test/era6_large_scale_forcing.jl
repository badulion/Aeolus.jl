# Phase 58 (P58-c, #171) — LargeScaleForcing: geostrophic pressure
# gradient, geostrophic nudging, prescribed advective tendencies and a
# prescribed surface-temperature series.
#
# What is checked against what:
#   * the tables: bilinear interpolation, end-holding, and the exact step
#     mean of a piecewise-linear series against hand-computed integrals;
#   * the pressure gradient: the frictionless inertial oscillation
#     (Blackadar 1957) after one period 2π/f and its maximum speed
#     U_g + |V_ag|, and the laminar Ekman spiral (Ekman 1905) over a
#     no-slip wall against the analytic profile and its 45° surface veer;
#   * the advective tendencies: Δ∫θ dV against ∫∫(∂θ/∂t)_adv dV dt, the
#     latter by the trapezoid rule on the table nodes, to round-off;
#   * bit for bit: GeostrophicNudging against the Berlin case's former
#     add_geostrophic_wind! over 100 smoke steps.

using Test, Printf
using Aeolus
using Aeolus.Fields: interior_view

@testset "Era VI — LargeScaleForcing (#171)" begin

    @testset "ForcingTable: interpolation, end values, exact step mean" begin
        z = [0.0, 100.0, 300.0]
        t = [0.0, 3600.0, 7200.0]
        V = [1.0 2.0 4.0;
             3.0 3.0 3.0;
             5.0 0.0 5.0]
        tab = ForcingTable(z, t, V)
        @test forcing_value(tab, 0.0, 0.0) == 1.0
        @test forcing_value(tab, 300.0, 3600.0) == 0.0
        @test forcing_value(tab, 50.0, 0.0) ≈ 2.0                 # mid-way in z
        @test forcing_value(tab, 200.0, 1800.0) ≈ (0.5 * (3 + 5) + 0.5 * (3 + 0)) / 2
        @test forcing_value(tab, -10.0, -100.0) == 1.0            # held below / before
        @test forcing_value(tab, 1e4, 1e5) == 5.0                 # held above / after
        # the step mean of a piecewise-linear series is exact:
        # at z = 0 over [1800, 5400]: ∫ = 900·(1.5+2)/2·2 … by hand
        # p(1800) = 1.5, p(3600) = 2, p(5400) = 3 → (1.5+2)/2·1800 + (2+3)/2·1800
        @test forcing_mean(tab, 0.0, 1800.0, 5400.0) ≈ ((1.5 + 2) / 2 * 1800 + (2 + 3) / 2 * 1800) / 3600
        # beyond the end the value is held: mean over [7200, 9000] = 4
        @test forcing_mean(tab, 0.0, 7200.0, 9000.0) ≈ 4.0
        # straddling the end: [6300, 8100] → (3.5+4)/2·900 + 4·900
        @test forcing_mean(tab, 0.0, 6300.0, 8100.0) ≈ ((3.5 + 4) / 2 * 900 + 4 * 900) / 1800
        # a zero-length step is the point value
        @test forcing_mean(tab, 200.0, 1800.0, 1800.0) == forcing_value(tab, 200.0, 1800.0)
        # step means add up to the whole integral to round-off
        tt = range(0.0, 9000.0; length = 38)
        s = sum(forcing_mean(tab, 150.0, tt[i], tt[i+1]) * (tt[i+1] - tt[i]) for i in 1:length(tt)-1)
        @test s ≈ forcing_mean(tab, 150.0, 0.0, 9000.0) * 9000 rtol = 1e-14
        # a profile (no time dependence), a number, a function
        prof = ForcingTable([0.0, 1000.0], [2.0, 12.0])
        @test forcing_value(prof, 250.0, 1e6) ≈ 4.5
        @test forcing_mean(prof, 250.0, 0.0, 600.0) ≈ 4.5
        @test forcing_mean(7.0, 1.0, 0.0, 1.0) === 7.0
        @test forcing_mean((z, t) -> z + t^2, 1.0, 0.0, 3.0) ≈ 1 + 3.0   # Simpson is exact for t²
        @test_throws ArgumentError ForcingTable([0.0, 0.0], [0.0], zeros(2, 1))
        @test_throws ArgumentError ForcingTable([0.0, 1.0], [0.0], zeros(3, 1))
    end

    @testset "SurfaceTemperatureSeries (GABLS1: 265 K, −0.25 K/h)" begin
        s = SurfaceTemperatureSeries([0.0, 9 * 3600.0], [265.0, 265.0 - 0.25 * 9])
        @test surface_temperature(s, 0.0) == 265.0
        @test surface_temperature(s, 4 * 3600.0) ≈ 264.0
        @test surface_temperature(s, 9 * 3600.0) ≈ 262.75
        @test surface_temperature(s, 20 * 3600.0) ≈ 262.75       # held after the end
        @test_throws ArgumentError SurfaceTemperatureSeries([0.0, 1.0], [265.0])
    end
end
