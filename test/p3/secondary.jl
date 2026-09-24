# P3 part-file tests: Hallett-Mossop rime splintering (src/p3/secondary.jl,
# issue #7).
#
# Asserted: the -8…-3 °C window with its peak at -5 °C, exactly 350
# splinters per mg of rime at the peak computed from the P3 riming rates
# themselves, conservation (splintering changes no species mass and no
# temperature, only N_i and the rime split), unchanged density of the new
# rime, the large-drop fraction against direct quadrature, and ice
# multiplication in a riming parcel held at -5 °C.

using Aeolus: MoistThermodynamics

@testset "P3 Hallett-Mossop rime splintering" begin
    hm = P3_DEFAULT_HM
    pp = P3_DEFAULT_PROCESS
    C2K = 273.15

    @testset "Temperature window" begin
        f(Tc) = p3_hm_temperature_factor(Tc + C2K)
        @test f(-5.0) == 1.0
        for Tc in (-8.0, -3.0, -8.5, -2.5, -15.0, 0.0, 5.0, -30.0)
            @test f(Tc) == 0.0
        end
        @test f(-4.0) ≈ 0.5 atol = 1e-12          # warm limb, 2 K wide
        @test f(-6.5) ≈ 0.5 atol = 1e-12          # cold limb, 3 K wide
        grid = range(-10.0, 0.0; length = 2001)
        vals = f.(grid)
        @test grid[argmax(vals)] ≈ -5.0 atol = 1e-12
        @test all(0 .<= vals .<= 1)
        @test all(vals[(grid .<= -8) .| (grid .>= -3)] .== 0)
        @test all(vals[(grid .> -8) .& (grid .< -3)] .> 0)
    end

    # A riming parcel: cloud + rain + rimed ice, cold enough for riming.
    riming(Tc) = P3ParcelState(T = Tc + C2K, p = 8.5e4,
        q_v = MoistThermodynamics.saturation_specific_humidity(Tc + C2K, 8.5e4),
        q_c = 5e-4, N_c = 1e8, q_r = 2e-4, N_r = 2e4,
        q_i = 3e-4, N_i = 5e3, q_rim = 1.5e-4, B_rim = 1.5e-4 / 500)

    @testset "350 splinters per mg of rime at -5 °C" begin
        s = riming(-5.0)
        r = p3_process_rates(s)
        @test r.rime_c > 0 && r.rime_r > 0
        n, q = p3_hallett_mossop(s, r)
        per_mg = n / ((r.rime_c + r.rime_r) * 1e6)
        @printf("    HM at -5 °C: rime %.3e kg/kg/s → %.4e splinters/kg/s = %.6f per mg\n",
                r.rime_c + r.rime_r, n, per_mg)
        @test per_mg ≈ 350 rtol = 1e-12
        @test q ≈ n * P3_HM_SPLINTER_MASS rtol = 1e-14
        # the rimer is above P3's 250 μm threshold here
        @test p3_hm_rimer_diameter(s) >= P3_DEFAULT_HM.D_ice_min
        # the splinter source scales with the riming rates it is given
        r2 = merge(r, (rime_c = 2r.rime_c, rime_r = 2r.rime_r))
        @test first(p3_hallett_mossop(s, r2)) ≈ 2n rtol = 1e-12
        # outside the window: riming but no splinters
        for Tc in (-2.0, -9.0, -15.0)
            sc = riming(Tc); rc = p3_process_rates(sc)
            @test rc.rime_c > 0
            @test p3_hallett_mossop(sc, rc) == (0.0, 0.0)
        end
        # in the window but no riming: no splinters
        s0 = P3ParcelState(T = 268.15, p = 8.5e4, q_v = 2e-3, q_i = 1e-4, N_i = 1e4)
        @test p3_hallett_mossop(s0, p3_process_rates(s0)) == (0.0, 0.0)
    end

    @testset "Bookkeeping: conservation and rime density" begin
        s = riming(-5.0)
        r = p3_process_rates(s)
        d = p3_hm_tendencies(s, r)
        n, q = p3_hallett_mossop(s, r)
        @test d.N_i == n
        @test d.q_i == 0 && d.q_v == 0 && d.q_c == 0 && d.q_r == 0 && d.T == 0
        @test d.q_rim == -q
        # the density of the rime formed this step is unchanged by splintering
        base = p3_tendencies(s, r)
        ρ_new(dq, dB) = dq / dB
        @test ρ_new(base.q_rim + d.q_rim, base.B_rim + d.B_rim) ≈
              ρ_new(r.rime_c + r.rime_r, r.rime_c / r.ρ_rime_c + r.rime_r / pp.ρ_rim_rain) rtol = 1e-12
        # splinter mass is a tiny fraction of the rime
        # splinter mass is a small fraction of the rime: 350 mg⁻¹ of
        # 10 μm spheres at 900 kg m⁻³ (P3 code) = 1.65e-4
        @test P3_HM_SPLINTER_MASS ≈ π / 6 * 900 * (10e-6)^3 rtol = 1e-15
        @test q / (r.rime_c + r.rime_r) ≈ 3.5e8 * P3_HM_SPLINTER_MASS rtol = 1e-12
        @test q / (r.rime_c + r.rime_r) ≈ 1.649e-4 rtol = 1e-3

        # a full step with splintering: water conserved to rounding, only
        # N_i raised (to within what the size limiter allows)
        dt = 10.0
        a = p3_step(s, dt)
        b, rb = p3_step_with_rates(s, dt; hallett_mossop = hm)
        @test p3_total_water(b) ≈ p3_total_water(s) rtol = 1e-14
        @test b.q_i ≈ a.q_i rtol = 1e-12
        @test b.T ≈ a.T rtol = 1e-14
        nb, _ = p3_hallett_mossop(s, rb)
        @test b.N_i - a.N_i ≈ nb * dt rtol = 1e-6
        @test 0 <= b.q_rim <= b.q_i
        @test p3_step(s, dt; hallett_mossop = hm) == b
    end

    @testset "Rimer conditions of the P3 code: size and existing rime" begin
        s = riming(-5.0)
        r = p3_process_rates(s)
        D = p3_hm_rimer_diameter(s)
        # the mean diameter reproduces the mean mass with the bulk density
        md = p3_mass_dimension(s.q_i, s.q_rim, s.B_rim)
        λ = p3_slope(s.q_i, s.N_i, md; λ_min = Aeolus.P3Microphysics._p3_ice_λ_limits(0, md.F_rim)[1],
                     λ_max = Aeolus.P3Microphysics._p3_ice_λ_limits(0, md.F_rim)[2])
        @test π / 6 * D^3 * p3_bulk_density(0, λ, md) ≈ s.q_i / s.N_i rtol = 1e-10
        @printf("    rimer mean diameter %.0f μm (threshold 250 μm)\n", 1e6 * D)
        n, _ = p3_hallett_mossop(s, r)
        @test n > 0
        # raise the threshold above D: no splinters, all else equal
        @test p3_hallett_mossop(s, r, P3HallettMossop(D_ice_min = 1.01D)) == (0.0, 0.0)
        @test first(p3_hallett_mossop(s, r, P3HallettMossop(D_ice_min = 0.99D))) == n
        # the same ice mass split into many small crystals (mean D well
        # below 250 μm) rimes but does not splinter
        small = P3ParcelState(T = s.T, p = s.p, q_v = s.q_v, q_c = s.q_c, N_c = s.N_c,
                              q_r = s.q_r, N_r = s.N_r, q_i = s.q_i, N_i = 1e7,
                              q_rim = s.q_rim, B_rim = s.B_rim)
        @test p3_hm_rimer_diameter(small) < 250e-6
        rs = p3_process_rates(small)
        @test rs.rime_c > 0
        @test p3_hallett_mossop(small, rs) == (0.0, 0.0)
        # unrimed ice (no rime yet): no splinters, as in the P3 code
        bare = P3ParcelState(T = s.T, p = s.p, q_v = s.q_v, q_c = s.q_c, N_c = s.N_c,
                             q_r = s.q_r, N_r = s.N_r, q_i = s.q_i, N_i = s.N_i)
        @test p3_hallett_mossop(bare, p3_process_rates(bare)) == (0.0, 0.0)
        # no ice
        @test p3_hm_rimer_diameter(P3ParcelState(T = 268.0, p = 8e4)) == 0.0
    end

    @testset "Large-drop fraction" begin
        ρa = 1.1
        qc, Nc = 3e-4, 1e8          # mean-volume diameter ≈ 18 μm
        @test p3_hm_large_drop_fraction(qc, Nc, ρa, 0.0) == 1.0
        @test p3_hm_large_drop_fraction(0.0, 0.0, ρa, 25e-6) == 0.0
        # direct quadrature of the mass above D_min for the same gamma PSD
        μ = Aeolus.P3Microphysics._p3_cloud_μ(Nc * ρa)
        λ = gamma_psd_slope(Nc, 0, qc / (π / 6 * 1000.0), 3, μ)
        mass(D1, D2) = begin
            h = (D2 - D1) / 200_000
            sum(D -> D^(μ + 3) * exp(-λ * D), range(D1 + h / 2, D2 - h / 2; step = h)) * h
        end
        total = mass(0.0, 40 / λ * (μ + 4))
        fs = [p3_hm_large_drop_fraction(qc, Nc, ρa, D) for D in (10e-6, 20e-6, 25e-6, 40e-6)]
        for (D, fD) in zip((10e-6, 20e-6, 25e-6, 40e-6), fs)
            @test fD ≈ mass(D, 40 / λ * (μ + 4)) / total rtol = 1e-6
        end
        @test issorted(fs; rev = true)
        @printf("    cloud (q_c 0.3 g/kg, 100 /mg, μ_c=%.2f): mass fraction in drops > 25 μm = %.3f\n",
                μ, fs[3])
        # with the restriction, a small-droplet cloud produces fewer splinters
        s = P3ParcelState(T = 268.15, p = 8.5e4,
            q_v = MoistThermodynamics.saturation_specific_humidity(268.15, 8.5e4),
            q_c = 1e-4, N_c = 3e8, q_i = 3e-4, N_i = 5e3, q_rim = 1e-4, B_rim = 1e-4 / 500)
        r = p3_process_rates(s)
        n_all, _ = p3_hallett_mossop(s, r)
        n_big, _ = p3_hallett_mossop(s, r, P3HallettMossop(D_drop_min = 25e-6))
        # D_mv ≈ 8.6 μm with the broad Martin et al. shape (μ_c ≈ 3.7): the
        # tail beyond 25 μm carries well under 1 % of the cloud water
        @printf("    D_mv 8.6 μm cloud: splinters with 25 μm restriction / without = %.4f\n",
                n_big / n_all)
        @test n_big / n_all ≈ p3_hm_large_drop_fraction(s.q_c, s.N_c,
            MoistThermodynamics.air_density(s.T, s.p,
                MoistThermodynamics.PhasePartition(p3_total_water(s), s.q_c, s.q_i)), 25e-6) rtol = 1e-10
        @test 0 < n_big < 0.01 * n_all
    end

    @testset "Ice multiplication in a riming parcel at -5 °C" begin
        # Hold a riming parcel in the HM window for 20 min: splintering
        # must multiply the ice number, and do nothing at -12 °C.
        function run(Tc, hmopt; nsteps = 120, dt = 10.0)
            s = riming(Tc)
            s = P3ParcelState(T = s.T, p = s.p, q_v = s.q_v, q_c = s.q_c, N_c = s.N_c,
                              q_i = 1e-6, N_i = 100.0, q_rim = 5e-7, B_rim = 5e-7 / 500)
            w = p3_total_water(s)
            for _ in 1:nsteps
                # replenish cloud water as an updraft would
                s = P3ParcelState(T = s.T, p = s.p, q_v = s.q_v, q_c = 5e-4, N_c = 1e8,
                                  q_r = s.q_r, N_r = s.N_r, q_i = s.q_i, N_i = s.N_i,
                                  q_rim = s.q_rim, B_rim = s.B_rim)
                s = p3_step(s, dt; hallett_mossop = hmopt, adjust = false)
            end
            return s
        end
        with = run(-5.0, hm); without = run(-5.0, nothing)
        cold_with = run(-12.0, hm); cold_without = run(-12.0, nothing)
        @printf("    20 min at -5 °C: N_i %.3g /kg with HM, %.3g /kg without (×%.0f)\n",
                with.N_i, without.N_i, with.N_i / without.N_i)
        @test with.N_i > 10 * without.N_i
        @test cold_with.N_i == cold_without.N_i
    end
end
