# P3 part-file tests: Hallett-Mossop rime splintering (src/p3/secondary.jl,
# issue #7). Asserted: the -8…-3 °C window peaking at -5 °C, 350
# splinters per mg of rime at the peak, conservation (only N_i and the
# rime split change), and the rimer-size condition of the P3 code.

using Aeolus: MoistThermodynamics

@testset "P3 Hallett-Mossop rime splintering" begin
    hm = P3_DEFAULT_HM
    C2K = 273.15
    f(Tc) = p3_hm_temperature_factor(Tc + C2K)
    riming(Tc) = P3ParcelState(T = Tc + C2K, p = 8.5e4,
        q_v = MoistThermodynamics.saturation_specific_humidity(Tc + C2K, 8.5e4),
        q_c = 5e-4, N_c = 1e8, q_r = 2e-4, N_r = 2e4,
        q_i = 3e-4, N_i = 5e3, q_rim = 1.5e-4, B_rim = 1.5e-4 / 500)

    @testset "Temperature window" begin
        @test f(-5.0) == 1.0
        @test all(f(Tc) == 0 for Tc in (-8.0, -3.0, -15.0, 0.0))
        @test abs(f(-4.0) - 0.5) < 1e-12 && abs(f(-6.5) - 0.5) < 1e-12
    end

    @testset "350 splinters per mg of rime; conservation" begin
        s = riming(-5.0)
        r = p3_process_rates(s)
        n, q = p3_hallett_mossop(s, r)
        @test n / ((r.rime_c + r.rime_r) * 1e6) ≈ 350 rtol = 1e-12
        @test q ≈ n * P3_HM_SPLINTER_MASS rtol = 1e-14
        @test p3_hallett_mossop(riming(-12.0), p3_process_rates(riming(-12.0))) == (0.0, 0.0)
        d = p3_hm_tendencies(s, r)
        @test d.N_i == n && d.q_rim == -q
        @test d.q_i == 0 && d.q_v == 0 && d.q_c == 0 && d.q_r == 0 && d.T == 0
        b, rb = p3_step_with_rates(s, 10.0; hallett_mossop = hm)
        a = p3_step(s, 10.0)
        @test p3_total_water(b) ≈ p3_total_water(s) rtol = 1e-14
        @test b.N_i - a.N_i ≈ first(p3_hallett_mossop(s, rb)) * 10.0 rtol = 1e-6
        # rimer below the 250 μm threshold (P3 code): rimes, no splinters
        D = p3_hm_rimer_diameter(s)
        @test p3_hallett_mossop(s, r, P3HallettMossop(D_ice_min = 1.01D)) == (0.0, 0.0)
    end
end
