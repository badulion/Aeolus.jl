# Phase 44 (issue #8) tests: sub-grid precipitation fraction
# (src/p3/precip_fraction.jl).
#
# The physics asserted: the option is off by default and then bit-
# identical to the grid-mean scheme (checked against values recorded on
# the base branch); full fractions recover the grid mean exactly; with
# f < 1 the nonlinear KK2000 rates change by the analytic factors their
# exponents imply; rain under cloud does not evaporate while rain in
# clear air evaporates at the clear-sky deficit; the vertical overlap
# assumptions are ordered max ≤ max-random ≤ random down a column with
# the analytic two-layer values; and water is conserved.

@testset "P3 sub-grid precipitation fraction (Phase 44)" begin
    qsat(T, p) = Aeolus.MoistThermodynamics.saturation_specific_humidity(
        T, p, Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS)
    fields(s) = [getfield(s, f) for f in fieldnames(typeof(s))]
    P3 = Aeolus.P3Microphysics
    ρair(s) = Aeolus.MoistThermodynamics.air_density(s.T, s.p, P3._p3_partition(s),
                                                     P3_DEFAULT_PROCESS.thermo)

    warm = P3ParcelState(T = 285.0, p = 9.0e4, q_v = 0.8 * qsat(285.0, 9e4),
                         q_c = 5e-4, N_c = 1e8, q_r = 4e-4, N_r = 2e4)
    cold = P3ParcelState(T = 263.15, p = 7.0e4, q_v = qsat(263.15, 7e4),
                         q_c = 3e-4, N_c = 1e8, q_r = 2e-4, N_r = 1e4,
                         q_i = 4e-4, N_i = 2e5, q_rim = 1e-4, B_rim = 1e-4 / 400)
    melt = P3ParcelState(T = 276.15, p = 8.0e4, q_v = 0.95 * qsat(276.15, 8e4),
                         q_c = 2e-4, N_c = 5e7, q_r = 1e-3, N_r = 5e3,
                         q_i = 3e-4, N_i = 1e5)

    @testset "Default off: bit-identical to the grid-mean scheme" begin
        # p3_step(s, 10 s) recorded on origin/integration/p3-icenuc before
        # this change (shortest round-trip repr, so == is bitwise), and
        # re-recorded for issue #82, when p3_condensation (step 5) moved
        # from the θ_li diagnosis to the enthalpy-conserving process
        # adjustment. That shifted only T, q_v and q_c of step 5: T by
        # +0.0138 K (warm), +0.0006 K (cold), +0.0045 K (melt); the
        # sub-grid-fraction code under test here was not touched.
        ref = [
            [283.7847623122012, 90000.0, 0.008217628079583062, 0.0, 0.0,
             0.00040232076000317564, 19268.96044637655, 0.0, 0.0, 0.0, 0.0],
            [263.15825174821794, 70000.0, 0.002551991706534512, 0.00028275205389503854,
             9.725197277522258e7, 0.00017553639040846392, 9929.506893599479,
             0.00044004646557073787, 200000.0, 0.0001327076912658683, 3.117083421982849e-7],
            [275.74035972657447, 80000.0, 0.005743990840089086, 6.0209381522781445e-5,
             4.597795875836948e7, 0.001314154405572437, 104931.78433446621,
             0.0, 0.0, 0.0, 0.0]]
        for (s, r) in zip((warm, cold, melt), ref)
            @test fields(p3_step(s, 10.0)) == r
            # fractions of one: the same, bit for bit
            @test fields(p3_step(s, 10.0; cloud_fraction = 1.0, precip_fraction = 1.0)) == r
            @test p3_fractional_process_rates(s, 1.0, 1.0) == p3_process_rates(s)
        end
    end

    @testset "Accretion and autoconversion: analytic nonlinearity factors" begin
        pp = P3_DEFAULT_PROCESS
        g = p3_process_rates(warm)
        e = pp.kk_acc_exp                                  # 1.15
        for f in (0.1, 0.25, 0.5, 0.8)
            # rain over a fraction f of a fully cloudy cell: overlap area f,
            # in-rain q_r/f → factor f · f^-1.15 = f^-0.15
            r = p3_fractional_process_rates(warm, 1.0, f)
            @test r.accr / g.accr ≈ f^(1 - e) rtol = 1e-12
            @test r.accr > g.accr                            # sign: enhanced
            # cloud and rain both on the same fraction f (max overlap):
            # f (q_c q_r / f²)^1.15 → f^(1 - 2·1.15)
            r2 = p3_fractional_process_rates(warm, f, f)
            @test r2.accr / g.accr ≈ f^(1 - 2e) rtol = 1e-12
            # autoconversion 1350 q_c^2.47 N_c^-1.79 on the cloud fraction:
            # f · f^-2.47 · f^+1.79 = f^0.32 (reduced). Exact once the air
            # density of the cloudy region (which counts its in-cloud
            # condensate, and enters through N_c in cm⁻³) is included; the
            # bare f^0.32 is off by 0.6 % at f = 0.1.
            ra = p3_fractional_process_rates(warm, f, 1.0)
            cloudy = P3._p3_with(warm; q_c = warm.q_c / f, N_c = warm.N_c / f,
                                 q_v = min(qsat(warm.T, warm.p), warm.q_v / f))
            ρr = ρair(cloudy) / ρair(warm)
            @test ra.auto / g.auto ≈ f^(1 - pp.kk_a + pp.kk_b) * ρr^(-pp.kk_b) rtol = 1e-12
            @test ra.auto / g.auto ≈ f^(1 - pp.kk_a + pp.kk_b) rtol = 1e-2
            @test ra.auto < g.auto
        end
        r = p3_fractional_process_rates(warm, 1.0, 0.25)
        @printf("    accretion, rain on 25%% of a cloudy cell: ×%.4f (analytic 0.25^-0.15 = %.4f)\n",
                r.accr / g.accr, 0.25^-0.15)
        r = p3_fractional_process_rates(warm, 0.25, 1.0)
        @printf("    autoconversion, 25%% cloud cover: ×%.4f (analytic 0.25^0.32 = %.4f)\n",
                r.auto / g.auto, 0.25^0.32)
        # Riming of cloud by (uniform) ice is linear in q_c: unchanged by f_c.
        rc = p3_fractional_process_rates(cold, 0.4, 1.0)
        gc = p3_process_rates(cold)
        @test rc.rime_c ≈ gc.rime_c rtol = 1e-3
        # rain riming is linear in (q_r, N_r) at fixed λ_r: unchanged by f_p
        rp = p3_fractional_process_rates(cold, 1.0, 0.3)
        @test rp.rime_r ≈ gc.rime_r rtol = 1e-3
        @test rp.accr / gc.accr ≈ 0.3^(1 - e) rtol = 1e-12
    end

    @testset "Evaporation: clear air versus cloudy air" begin
        T, p = 285.0, 9.0e4
        qs = qsat(T, p)
        s = P3ParcelState(T = T, p = p, q_v = 0.8qs, q_c = 3e-4, N_c = 1e8,
                          q_r = 4e-4, N_r = 2e4)
        grid = p3_process_rates(s).evap_r
        # rain entirely under the cloud: saturated air, no evaporation
        under = p3_fractional_process_rates(s, 0.5, 0.5).evap_r
        # rain over the whole cell, half of it in the clear part
        clear = p3_fractional_process_rates(s, 0.5, 1.0).evap_r
        # the clear air holds the vapour the saturated cloud does not:
        qclr = (0.8qs - 0.5qs) / 0.5
        byhand = 0.5 * p3_process_rates(P3ParcelState(T = T, p = p, q_v = qclr,
                                                      q_r = 4e-4, N_r = 2e4)).evap_r
        @printf("    rain evaporation: grid-mean %.3e, under cloud %.3e, half in clear air %.3e kg/kg/s\n",
                grid, under, clear)
        @test under == 0
        @test clear > under
        @test clear ≈ byhand rtol = 1e-12
        # in a cloud-free cell the in-rain rate is linear in (q_r, N_r) at
        # fixed λ_r, so shrinking the rain area changes it only through the
        # air density of the rainy region, which counts its rain (q_r/0.3):
        # measured 6.3e-5 relative.
        dry = P3ParcelState(T = T, p = p, q_v = 0.8qs, q_r = 4e-4, N_r = 2e4)
        @test p3_fractional_process_rates(dry, 1.0, 0.3).evap_r ≈
              p3_process_rates(dry).evap_r rtol = 2e-4
        # drier clear air → more evaporation (cloud holds q_s, clear the rest)
        wetter = p3_fractional_process_rates(
            P3ParcelState(T = T, p = p, q_v = 0.9qs, q_c = 3e-4, N_c = 1e8,
                          q_r = 4e-4, N_r = 2e4), 0.5, 1.0).evap_r
        @test clear > wetter > 0
    end

    @testset "Large steps: each region is limited by its own contents" begin
        # Review fix. A grid-mean limiter lets rain on 10 % of a fully
        # cloudy cell collect, in one step, cloud water from the 90 % it
        # does not cover (64 % of q_c gone in 60 s before the fix).
        T, p = 285.0, 9.0e4
        s = P3ParcelState(T = T, p = p, q_v = qsat(T, p), q_c = 5e-4, N_c = 1e8,
                          q_r = 1e-3, N_r = 5e3)
        # cloud outside the rain only autoconverts: its own rate bounds it
        only = p3_process_rates(P3._p3_with(s; q_r = 0.0, N_r = 0.0))
        for dt in (60.0, 300.0, 3000.0)
            n = p3_step(s, dt; adjust = false, cloud_fraction = 1.0, precip_fraction = 0.1)
            removed = 1 - n.q_c / s.q_c
            bound = 0.1 + 0.9 * min(1.0, dt * only.auto / s.q_c)
            @test removed <= bound * (1 + 1e-12)
            dt == 3000.0 && @test removed >= 0.1 * (1 - 1e-12)   # all rain-area cloud taken
        end
        # Rain under saturated cloud cannot evaporate however long the step:
        # with half the rain under cloud at least half of it survives.
        d = P3ParcelState(T = T, p = p, q_v = 0.5qsat(T, p), q_c = 1e-4, N_c = 1e8,
                          q_r = 1e-5, N_r = 1e3)
        for dt in (300.0, 3000.0)
            r = p3_fractional_process_rates(d, 0.5, 1.0; dt = dt)
            @test dt * r.evap_r <= 0.5 * d.q_r * (1 + 1e-12)
        end
        # Where the grid mean is ice-subsaturated but the cloudy part is at
        # water saturation, ice grows in the cloud (Wegener-Bergeron-
        # Findeisen) and sublimates in the clear air; a grid-mean limiter
        # would switch the in-cloud deposition off.
        Tc, pc = 258.0, 7.0e4
        c = P3ParcelState(T = Tc, p = pc, q_v = 0.8qsat(Tc, pc), q_c = 2e-4, N_c = 1e8,
                          q_i = 1e-4, N_i = 1e5)
        @test c.q_v < Aeolus.MoistThermodynamics.saturation_specific_humidity(
            Tc, pc, Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS; over = :ice)
        rl = p3_fractional_process_rates(c, 0.3, 1.0; dt = 10.0)
        ru = p3_fractional_process_rates(c, 0.3, 1.0)
        @test rl.dep ≈ ru.dep rtol = 1e-12      # not limited: far from the region's content
        @test rl.dep > 0 && rl.sub > 0
        # fractions of one: the grid-mean limited rates, bit for bit
        for st in (warm, cold, melt)
            @test p3_fractional_process_rates(st, 1.0, 1.0; dt = 10.0) ==
                  p3_limit_rates(st, p3_process_rates(st), 10.0)
        end
    end

    @testset "Vertical overlap: analytic values and ordering" begin
        ov = p3_precipitation_overlap
        # two adjacent cloudy layers, 0.3 above 0.5
        @test ov(0.3, 0.3, 0.5, :maximum) == 0.5
        @test ov(0.3, 0.3, 0.5, :random) ≈ 0.65
        @test ov(0.3, 0.3, 0.5, :maximum_random) ≈ 0.5       # adjacent → maximum
        # cloud 0.5 below a clear gap that rain crossed (c_above = 0) → random
        @test ov(0.3, 0.0, 0.5, :maximum_random) ≈ 0.65
        # no new cloud: the rain area is carried unchanged
        for o in (:maximum, :random, :maximum_random)
            @test ov(0.37, 0.2, 0.0, o) ≈ 0.37
        end
        @test_throws ArgumentError ov(0.1, 0.1, 0.1, :minimum)

        # Column: cloud layers separated by gaps, rain everywhere below the top.
        nz = 30
        cf = [k in 25:27 ? 0.3 : k in 18:20 ? 0.4 : k in 8:10 ? 0.5 : k in 4:5 ? 0.2 : 0.0
              for k in 1:nz]
        qc = [c > 0 ? 2e-4 : 0.0 for c in cf]
        qr = [k <= 27 ? 1e-4 : 0.0 for k in 1:nz]
        Pm = p3_precipitation_fraction(cf, qc, qr; overlap = :maximum)
        Pmr = p3_precipitation_fraction(cf, qc, qr; overlap = :maximum_random)
        Pr = p3_precipitation_fraction(cf, qc, qr; overlap = :random)
        @test all(0 .<= Pm .<= Pmr .<= Pr .<= 1)
        @test all(diff(reverse(Pmr[1:27])) .>= 0)             # grows downward
        # surface values: max 0.5; max-random = random over the four
        # gap-separated decks: 1 - 0.7·0.6·0.5·0.8
        @test Pm[1] ≈ 0.5
        @test Pmr[1] ≈ 1 - 0.7 * 0.6 * 0.5 * 0.8
        @test Pr[1] ≈ 1 - 0.7^3 * 0.6^3 * 0.5^3 * 0.8^2
        @printf("    surface precipitation fraction: max %.3f ≤ max-random %.3f ≤ random %.3f\n",
                Pm[1], Pmr[1], Pr[1])
        # rain that evaporated completely resets the area below it
        qr2 = copy(qr); qr2[15] = 0.0
        P2 = p3_precipitation_fraction(cf, qc, qr2)
        @test P2[14] == 1.0                                   # orphan rain → grid mean
        @test P2[11] == 1.0
        @test P2[8] ≈ 1.0                                     # union with 1 stays 1
        qr3 = copy(qr); qr3[1:15] .= 0.0; qr3[11:12] .= 1e-4   # rain only beneath a gap
        P3v = p3_precipitation_fraction(cf, qc, qr3)
        @test P3v[12] == 1.0
    end

    @testset "Column step: water conservation and fraction-weighted rates" begin
        T0, p0 = 283.0, 9.0e4
        nz = 12
        col = [P3ParcelState(T = T0 - 0.6k, p = p0 - 800k, q_v = 0.85 * qsat(T0 - 0.6k, p0 - 800k),
                             q_c = k >= 8 ? 4e-4 : 0.0, N_c = k >= 8 ? 1e8 : 0.0,
                             q_r = 3e-4, N_r = 1.5e4) for k in 1:nz]
        cf = [k >= 8 ? 0.4 : 0.0 for k in 1:nz]
        W(c) = sum(p3_total_water, c)
        for overlap in (:maximum, :maximum_random, :random)
            new, fp = p3_step_column(col, 30.0; cloud_fraction = cf, overlap = overlap)
            @test W(new) ≈ W(col) rtol = 1e-14
            @test all(p3_total_water(new[k]) ≈ p3_total_water(col[k]) for k in 1:nz)
            @test all(s -> s.q_r >= 0 && s.q_c >= 0 && s.q_v >= 0, new)
            # one contiguous 5-level deck of 0.4: maximum and maximum-random
            # overlap give 0.4, random 1 - 0.6^5
            @test fp[1] ≈ (overlap === :random ? 1 - 0.6^5 : 0.4)
        end
        # Below cloud base (no cloud) the in-rain rates are linear in
        # (q_r, N_r) at fixed λ_r, so the rain mass there evolves as in the
        # grid-mean step. Inside the deck the rain sits under the cloud
        # (f_p = f_c = 0.4, maximum overlap): it does not evaporate and
        # accretes at f^-1.3 the grid-mean rate, so it gains more rain.
        new, _ = p3_step_column(col, 30.0; cloud_fraction = cf, adjust = false)
        grid = [p3_step(s, 30.0; adjust = false) for s in col]
        below, deck = 1:7, 8:12
        # (to the rain-loading air-density effect, measured 1.7e-6)
        @test sum(s.q_r for s in new[below]) ≈ sum(s.q_r for s in grid[below]) rtol = 1e-5
        @test sum(s.q_r for s in new[deck]) > sum(s.q_r for s in grid[deck])
        @printf("    deck rain after 30 s: sub-grid %.4e vs grid-mean %.4e kg/kg\n",
                sum(s.q_r for s in new[deck]), sum(s.q_r for s in grid[deck]))
        # plus sedimentation: column mass + surface precipitation conserved
        qr = [s.q_r for s in new]; Nr = [s.N_r for s in new]
        ρ = fill(1.1, nz); dz = 250.0
        M0 = sum(ρ .* qr .* dz)
        Pq, _, _ = p3_sediment_rain!(qr, Nr, ρ, dz, 60.0)
        @test sum(ρ .* qr .* dz) + Pq ≈ M0 rtol = 1e-13
    end

    @testset "combined with Hallett-Mossop (issues #7 + #8 merge)" begin
        # Mixed-phase riming state at -5 °C, where rime splintering peaks.
        s = P3ParcelState{Float64}(268.15, 7e4, 3e-3, 5e-4, 1e8, 1e-3, 1e4,
                                   1e-3, 1e5, 5e-4, 5e-4 / 400)
        tw(x) = x.q_v + x.q_c + x.q_r + x.q_i
        grid, _ = p3_step_with_rates(s, 10.0; hallett_mossop = P3_DEFAULT_HM)
        full, _ = p3_step_with_rates(s, 10.0; hallett_mossop = P3_DEFAULT_HM,
                                     cloud_fraction = 1.0, precip_fraction = 1.0)
        frac, _ = p3_step_with_rates(s, 10.0; hallett_mossop = P3_DEFAULT_HM,
                                     cloud_fraction = 0.4, precip_fraction = 0.6)
        nohm = p3_step(s, 10.0; cloud_fraction = 0.4, precip_fraction = 0.6)
        # fractions of one are the grid-mean step, splinters included
        @test full == grid
        # splinters are still produced when the rates come from the regions
        @test frac.N_i > nohm.N_i
        # p3_step forwards both options to p3_step_with_rates
        @test p3_step(s, 10.0; hallett_mossop = P3_DEFAULT_HM,
                      cloud_fraction = 0.4, precip_fraction = 0.6) == frac
        @test tw(frac) ≈ tw(s) rtol = 1e-14
    end
end
