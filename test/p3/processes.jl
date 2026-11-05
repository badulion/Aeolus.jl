# P3 part-file tests: microphysical process rates (src/p3/processes.jl,
# issue #3).
#
# Asserted: water conservation of every process and of a full step,
# latent heating consistent with the phase change, the KK2000 scaling
# exponents, thresholds (melting only above 0 °C, sublimation only in
# subsaturation), the Cober & List rime density, and the size limiter
# shared with sedimentation.

using Aeolus: MoistThermodynamics, AerosolCloud, Superdroplet

@testset "P3 process rates" begin
    MT = MoistThermodynamics
    pp = P3_DEFAULT_PROCESS
    c = pp.thermo
    P3 = Aeolus.P3Microphysics
    relerr(x, ref) = abs(x - ref) / abs(ref)
    qsat(T, p) = MT.saturation_specific_humidity(T, p, c)
    qsati(T, p) = MT.saturation_specific_humidity(T, p, c; over = :ice)
    part(x) = MT.PhasePartition(x.q_v + x.q_c + x.q_r + x.q_i, x.q_c + x.q_r, x.q_i)

    @testset "Condensation: enthalpy-conserving saturation adjustment (#82)" begin
        s = P3ParcelState(T = 285.0, p = 6.0e4, q_v = 1.1 * qsat(285.0, 6.0e4))
        a = p3_condensation(s)
        @test a.q_v + a.q_c ≈ s.q_v + s.q_c rtol = 1e-15
        @test relerr(a.q_v, qsat(a.T, a.p)) < 1e-8
        h(x) = MT.moist_enthalpy(x.T, part(x), c)
        @test abs(h(a) - h(s)) / h(s) < 1e-14
        # first law at constant h: c_pm ΔT = L_v(T₀) Δq_c (θ_li gave 1.064 at 600 hPa)
        @test MT.heat_capacity_moist(part(a), c) * (a.T - s.T) /
              (MT.latent_heat_vapor(s.T, c) * a.q_c) ≈ 1 atol = 1e-12
    end

    @testset "KK2000 warm rain, self-collection, Cober-List" begin
        ρa, qc, Nc = 1.1, 5e-4, 1e8
        a, _, _ = p3_autoconversion(qc, Nc, ρa)
        # one source of truth: AerosolCloud's KK2000
        @test a == AerosolCloud.autoconversion_rate(qc, Nc * ρa * 1e-6)
        @test log(p3_autoconversion(2qc, Nc, ρa)[1] / a) / log(2) ≈ 2.47 atol = 1e-12
        @test log(p3_autoconversion(qc, 2Nc, ρa)[1] / a) / log(2) ≈ -1.79 atol = 1e-12
        @test p3_autoconversion(0.99 * pp.q_c_auto_min, Nc, ρa) == (0.0, 0.0, 0.0)
        acc, _ = p3_accretion(3e-4, Nc, 1e-4)
        @test acc ≈ 67 * (3e-4 * 1e-4)^1.15 rtol = 1e-14
        # self-collection balances breakup at D_break + ln2/k_break ≈ 581.5 μm
        Deq = pp.D_break + log(2) / pp.k_break
        @test 581 < 1e6 * Deq < 582
        @test abs(p3_rain_self_collection(1e-3, 1e4, 1 / Deq, ρa)) <
              1e-12 * p3_rain_self_collection(1e-3, 1e4, 1 / 100e-6, ρa)
        # Cober & List (1993) rime density: 159.5 at R_i = 1, 611 at 8, 900 at 12
        @test [p3_cober_list_density(R) for R in (1.0, 8.0, 12.0)] ≈ [159.5, 611.0, 900.0] rtol = 1e-12
        @test all(diff(p3_cober_list_density.(range(0, 15, length = 301))) .>= 0)
    end

    # mixed-phase cold state, warm melting state, subsaturated cold state
    cold = P3ParcelState(T = 263.15, p = 7.0e4, q_v = qsat(263.15, 7.0e4),
                         q_c = 3e-4, N_c = 1e8, q_r = 2e-4, N_r = 2e4,
                         q_i = 3e-4, N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)
    warm = P3ParcelState(T = 276.15, p = 8.0e4, q_v = 0.95 * qsat(276.15, 8.0e4),
                         q_c = 3e-4, N_c = 1e8, q_r = 2e-4, N_r = 2e4,
                         q_i = 3e-4, N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)
    dry = P3ParcelState(T = 258.15, p = 6.0e4, q_v = 0.7 * qsati(258.15, 6.0e4),
                        q_r = 2e-4, N_r = 2e4,
                        q_i = 3e-4, N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)
    allfields(x) = (x.q_v, x.q_c, x.N_c, x.q_r, x.N_r, x.q_i, x.N_i, x.q_rim, x.B_rim)

    @testset "Each process conserves water and heats with its latent heat" begin
        keys_mass = (:evap_r, :dep, :sub, :auto, :accr, :rime_c, :rime_r, :shed, :melt)
        companions = Dict(:evap_r => (:n_evap_r,), :dep => (), :sub => (:n_sub,),
                          :auto => (:n_auto_r, :n_auto_c), :accr => (:n_accr_c,),
                          :rime_c => (:n_rime_c,), :rime_r => (:n_rime_r,),
                          :shed => (:n_shed_c, :n_shed_r), :melt => (:n_melt,))
        only(r, k) = merge(merge(map(zero, r), (ρ_rime_c = r.ρ_rime_c,)),
                           NamedTuple{(k, companions[k]...)}(map(x -> r[x], (k, companions[k]...))))
        nact = 0; worst_w = 0.0; worst_h = 0.0
        for st in (cold, warm, dry)
            r = p3_process_rates(st)
            cpm = MT.heat_capacity_moist(part(st), c)
            Lv, Lf, Ls = MT.latent_heat_vapor(st.T, c), MT.latent_heat_fusion(st.T, c),
                         MT.latent_heat_sublim(st.T, c)
            Lexp = Dict(:evap_r => -Lv, :dep => Ls, :sub => -Ls, :auto => 0.0, :accr => 0.0,
                        :rime_c => Lf, :rime_r => Lf, :shed => 0.0, :melt => -Lf)
            for k in keys_mass
                r[k] > 0 || continue
                nact += 1
                d = p3_tendencies(st, only(r, k))
                worst_w = max(worst_w, abs(d.q_v + d.q_c + d.q_r + d.q_i) / r[k])
                worst_h = max(worst_h, abs(cpm * d.T - Lexp[k] * r[k]) / (Ls * r[k]))
            end
        end
        @test nact >= 12          # every process active in at least one state
        @test worst_w <= 1e-15
        @test worst_h <= 1e-12
    end

    @testset "Thresholds: evaporation, sublimation, melting" begin
        ev(S) = p3_process_rates(P3ParcelState(T = 280.0, p = 9.0e4, q_r = 5e-4, N_r = 5e3,
                                               q_v = (1 + S) * qsat(280.0, 9.0e4))).evap_r
        @test ev(0.05) == 0 && ev(-0.1) > 0
        rr(Si) = p3_process_rates(P3ParcelState(T = 258.15, p = 6.0e4, q_i = 1e-4, N_i = 1e5,
                                                q_v = (1 + Si) * qsati(258.15, 6.0e4)))
        @test rr(0.1).dep > 0 && rr(0.1).sub == 0
        @test rr(-0.1).sub > 0 && rr(-0.1).dep == 0
        mlt(T) = p3_process_rates(P3ParcelState(T = T, p = 8e4, q_v = qsat(T, 8e4), q_i = 3e-4,
                                                N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)).melt
        @test mlt(273.15) == 0 && 0 < mlt(274.15) < mlt(278.15)
        # monodisperse small drops (μ_r = 400): bulk evaporation → N 2πDS/(A+B)
        # with Superdroplet's F_k + F_d (one source of truth)
        Dm, Nr = 40e-6, 1e8
        st = P3ParcelState(T = 280.0, p = 9.0e4, q_v = 0.8 * qsat(280.0, 9.0e4),
                           q_r = Nr * π / 6 * 1000 * Dm^3, N_r = Nr)
        e = p3_process_rates(st, P3ProcessParameters{Float64}(μ_r = 400.0)).evap_r
        AB = (Superdroplet.fk_thermal(280.0) + Superdroplet.fd_vapor(280.0, 9.0e4)) / 1000
        @test e / (Nr * 2π * Dm * 0.2 / AB) ≈ 1 atol = 0.02
    end

    @testset "Riming: rime mass and volume at the Cober-List density" begin
        r = p3_process_rates(cold)
        d = p3_tendencies(cold, r)
        @test r.rime_c > 0 && r.rime_r > 0
        @test d.q_rim ≈ r.rime_c + r.rime_r rtol = 1e-14
        @test d.B_rim ≈ r.rime_c / r.ρ_rime_c + r.rime_r / pp.ρ_rim_rain rtol = 1e-12
        # above 0 °C collected cloud is shed to rain instead
        rw = p3_process_rates(warm)
        @test rw.rime_c == 0 && rw.shed > 0
    end

    @testset "Full step: conservation, energy, positivity" begin
        worst_e = 0.0; worst_w = 0.0
        for st in (cold, warm, dry)
            s1 = p3_step(st, 1.0; adjust = false)
            lhs = MT.heat_capacity_moist(part(st), c) * (s1.T - st.T)
            rhs = -MT.latent_heat_vapor(st.T, c) * (s1.q_v - st.q_v) +
                   MT.latent_heat_fusion(st.T, c) * (s1.q_i - st.q_i)
            worst_e = max(worst_e, relerr(lhs, rhs))
            # a few large steps with the adjustment
            x = st
            for _ in 1:5
                x = p3_step(x, 60.0)
            end
            worst_w = max(worst_w, relerr(p3_total_water(x), p3_total_water(st)))
            @test all(>=(0), allfields(x)) && x.q_rim <= x.q_i
        end
        @test worst_e < 1e-10
        @test worst_w < 1e-14
        # limiter: collected rain drops never removed faster than at the
        # mean drop mass (review fix: mass scaled to 0.242, number to 0.805)
        rc = p3_process_rates(cold)
        l = p3_limit_rates(cold, rc, 600.0)
        @test l.n_rime_r / rc.n_rime_r <= l.rime_r / rc.rime_r * (1 + 1e-12)
        # sublimation + rain evaporation over 1 h must not overshoot
        # saturation (review fix: q_v/q_si = 1.193 before)
        sd = P3ParcelState(T = 265.0, p = 7.0e4, q_v = 0.5 * qsati(265.0, 7.0e4),
                           q_r = 3e-3, N_r = 1e5, q_i = 3e-3, N_i = 1e4)
        e = p3_step(sd, 3600.0; adjust = false)
        @test e.q_v <= qsati(e.T, e.p) * (1 + 1e-3)
        @test relerr(p3_total_water(e), p3_total_water(sd)) <= 4eps()
    end

    @testset "Size limits: one definition shared with sedimentation" begin
        @test !isdefined(P3, :_p3_ice_lambda_bounds)
        md = p3_mass_dimension(1e-4, 4e-5, 4e-5 / 400, pp.ice)
        λmin, _ = P3._p3_ice_λ_limits(pp.μ_i, md.F_rim)
        @test λmin ≈ (pp.μ_i + 1) / (5e-3 + 2e-2 * md.F_rim^2) rtol = 1e-15
        s = P3ParcelState(T = 260.0, p = 7e4, q_v = 1e-3, q_i = 1e-4, N_i = 1e-300,
                          q_rim = 4e-5, B_rim = 4e-5 / 400)
        @test P3._p3_cleanup(s, pp).N_i ≈ 1e-4 / p3_mean_mass(pp.μ_i, λmin, md) rtol = 1e-12
        # rain: process rates and sedimentation see the same limited PSD
        # (P3 get_rain_dsd2: λ = (μ+1)/2 mm for mass without number)
        q, N = 1e-3, 10.0
        λp, Np = P3._p3_rain_slope(q, N, pp)
        Ns = p3_rain_limit_number(q, N; μ_r = pp.μ_r)
        @test Np ≈ Ns rtol = 1e-14
        @test λp ≈ p3_rain_size_distribution(q, Ns; μ_r = pp.μ_r)[3] rtol = 1e-14
        @test λp ≈ (pp.μ_r + 1) / 0.002 rtol = 1e-12
    end
end
