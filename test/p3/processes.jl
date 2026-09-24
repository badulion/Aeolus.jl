# P3 part-file tests: microphysical process rates (src/p3/processes.jl,
# issue #3).
#
# What is asserted is physics: water conservation of every process and of
# a full step, latent heating consistent with the phase change it
# accompanies, the KK2000 scaling exponents, thresholds (autoconversion
# below q_c_auto_min, melting below 0 °C, sublimation in supersaturation),
# and the Cober & List rime-density bounds.

using Aeolus: MoistThermodynamics, AerosolCloud, Superdroplet

@testset "P3 process rates" begin
    MT = MoistThermodynamics
    pp = P3_DEFAULT_PROCESS
    c = pp.thermo
    relerr(x, ref) = abs(x - ref) / abs(ref)
    qsat(T, p) = MT.saturation_specific_humidity(T, p, c)
    qsati(T, p) = MT.saturation_specific_humidity(T, p, c; over = :ice)

    @testset "Condensation by saturation adjustment" begin
        # Supersaturated warm parcel: condenses to exact saturation.
        s = P3ParcelState(T = 285.0, p = 9.0e4, q_v = 0.012)
        a = p3_condensation(s)
        @test a.q_c > 0
        @test a.q_v + a.q_c == s.q_v + s.q_c                     # exact
        @test relerr(a.q_v, qsat(a.T, a.p)) < 1e-8
        # θ_li conserved (the adjustment's invariant)
        part(x) = MT.PhasePartition(x.q_v + x.q_c, x.q_c, 0.0)
        θ0 = MT.liquid_ice_potential_temperature(s.T, s.p, part(s), c)
        θ1 = MT.liquid_ice_potential_temperature(a.T, a.p, part(a), c)
        @test abs(θ1 - θ0) < 1e-8
        # Latent heating vs condensate: c_pm ΔT ≈ L_v Δq_c. Not exact: the
        # adjustment conserves Bryan-Fritsch θ_li, not isobaric enthalpy,
        # and the moisture dependence of its Exner exponent R_m/c_pm alone
        # shifts ΔT by ~0.5 % at 900 hPa. Measured 1.0100 here and 1.0118
        # for the evaporation case below (a first 1 % bound was too tight
        # for that reason, not because of a code error).
        cpm = MT.heat_capacity_moist(part(a), c)
        ratio = cpm * (a.T - s.T) / (MT.latent_heat_vapor(a.T, c) * a.q_c)
        @printf("    adjustment: Δq_c = %.4f g/kg, ΔT = %.4f K, c_pm ΔT/(L_v Δq_c) = %.5f\n",
                1e3 * a.q_c, a.T - s.T, ratio)
        @test abs(ratio - 1) < 0.015
        # The deviation is the contract's θ_li, not P3: it vanishes at
        # p = p_ref (where the Exner factor is 1) and grows with
        # ln(p_ref/p). Measured: 0.9985 at 1000 hPa, 1.0121 at 900 hPa,
        # 1.0644 at 600 hPa (10 % supersaturation at 285 K).
        for (pz, tol) in ((1.0e5, 0.003), (6.0e4, 0.07))
            sz = P3ParcelState(T = 285.0, p = pz, q_v = 1.1 * qsat(285.0, pz))
            az = p3_condensation(sz)
            rz = MT.heat_capacity_moist(part(az), c) * (az.T - sz.T) /
                 (MT.latent_heat_vapor(az.T, c) * az.q_c)
            @printf("    p = %4.0f hPa: c_pm ΔT/(L_v Δq_c) = %.5f\n", pz / 100, rz)
            @test abs(rz - 1) < tol
        end
        # New cloud gets the prescribed droplet number.
        ρa = MT.air_density(a.T, a.p, MT.PhasePartition(a.q_v + a.q_c, a.q_c, 0.0), c)
        @test a.N_c ≈ pp.N_c_prescribed / ρa rtol = 1e-12

        # Supercooled: condensate stays LIQUID (P3 cloud is never ice by
        # adjustment), even at 250 K where MoistThermodynamics' default
        # ramp would make most of it ice.
        s2 = P3ParcelState(T = 250.0, p = 6.0e4, q_v = 1.2 * qsat(250.0, 6.0e4))
        a2 = p3_condensation(s2)
        @test a2.q_c > 0 && a2.q_i == 0
        @test relerr(a2.q_v, qsat(a2.T, a2.p)) < 1e-8

        # Subsaturated with cloud: the cloud evaporates completely.
        s3 = P3ParcelState(T = 285.0, p = 9.0e4, q_v = 0.5 * qsat(285.0, 9.0e4),
                           q_c = 2e-4, N_c = 1e8)
        a3 = p3_condensation(s3)
        @test a3.q_c == 0 && a3.N_c == 0
        @test a3.T < s3.T                                      # evaporative cooling
        @test a3.q_v ≈ s3.q_v + s3.q_c rtol = 1e-15
        cp3 = MT.heat_capacity_moist(MT.PhasePartition(a3.q_v, 0.0, 0.0), c)
        r3 = cp3 * (s3.T - a3.T) / (MT.latent_heat_vapor(a3.T, c) * s3.q_c)
        @printf("    evaporation: c_pm ΔT/(L_v Δq_c) = %.5f\n", r3)
        @test abs(r3 - 1) < 0.015
        # Rain and ice are untouched by the cloud adjustment.
        s4 = P3ParcelState(T = 270.0, p = 8e4, q_v = 0.004, q_r = 1e-3, N_r = 1e4,
                           q_i = 1e-4, N_i = 1e5)
        a4 = p3_condensation(s4)
        @test (a4.q_r, a4.N_r, a4.q_i, a4.N_i) == (s4.q_r, s4.N_r, s4.q_i, s4.N_i)
    end

    @testset "KK2000 autoconversion" begin
        ρa = 1.1
        # Delegates to AerosolCloud (one source of truth): identical number.
        qc, Nc = 5e-4, 1e8                     # 0.5 g/kg, 110 cm⁻³
        a, nr, nc = p3_autoconversion(qc, Nc, ρa)
        @test a == AerosolCloud.autoconversion_rate(qc, Nc * ρa * 1e-6)
        @test a ≈ 1350 * qc^2.47 * (Nc * ρa * 1e-6)^(-1.79) rtol = 1e-14
        @printf("    KK2000: q_c = 0.5 g/kg, N_c = 110 cm⁻³ → %.4e kg/kg/s\n", a)
        # Scaling exponents measured from finite ratios.
        e_q = log(p3_autoconversion(2qc, Nc, ρa)[1] / a) / log(2)
        e_N = log(p3_autoconversion(qc, 2Nc, ρa)[1] / a) / log(2)
        @printf("    KK2000 exponents: q_c %.12f, N_c %.12f\n", e_q, e_N)
        @test e_q ≈ 2.47 atol = 1e-12
        @test e_N ≈ -1.79 atol = 1e-12
        # Number bookkeeping: drizzle drops of 25 μm radius; droplets
        # removed at the mean droplet mass.
        @test nr ≈ a / (4 / 3 * π * 1000 * (25e-6)^3) rtol = 1e-14
        @test nc ≈ a * Nc / qc rtol = 1e-14
        # Threshold: vanishes below q_c_auto_min and without droplets.
        @test p3_autoconversion(0.99 * pp.q_c_auto_min, Nc, ρa) == (0.0, 0.0, 0.0)
        @test p3_autoconversion(pp.q_c_auto_min, Nc, ρa)[1] > 0
        @test p3_autoconversion(qc, 0.0, ρa) == (0.0, 0.0, 0.0)
        # Second indirect effect: doubling N_c suppresses to 2^-1.79 = 0.289
        @test p3_autoconversion(qc, 2Nc, ρa)[1] / a ≈ 2^-1.79 rtol = 1e-12
    end

    @testset "KK2000 accretion" begin
        qc, Nc, qr = 3e-4, 1e8, 1e-4
        a, nc = p3_accretion(qc, Nc, qr)
        @test a ≈ 67 * (qc * qr)^1.15 rtol = 1e-14
        e = log(p3_accretion(qc, Nc, 2qr)[1] / a) / log(2)
        @test e ≈ 1.15 atol = 1e-12
        @test nc ≈ a * Nc / qc rtol = 1e-14
        @test p3_accretion(qc, Nc, 0.0) == (0.0, 0.0)
        @test p3_accretion(0.0, Nc, qr) == (0.0, 0.0)
        # accretion beats autoconversion once drizzle exists (KK2000's point)
        @test a > p3_autoconversion(qc, Nc, 1.1)[1]
    end

    @testset "Rain self-collection and breakup" begin
        ρa, qr, Nr = 1.1, 1e-3, 1e4
        # Small drops: pure self-collection, φ = 1, SB2001 k_rr = 5.78.
        s_small = p3_rain_self_collection(qr, Nr, 1 / 100e-6, ρa)
        @test s_small ≈ 5.78 * ρa * qr * Nr rtol = 1e-14
        # φ is continuous at D_break and hits 0 at D_break + ln2/k_break.
        @test p3_rain_self_collection(qr, Nr, 1 / 280e-6, ρa) ≈ s_small rtol = 1e-12
        Deq = pp.D_break + log(2) / pp.k_break
        @printf("    breakup balances self-collection at D_m = %.2f μm\n", 1e6 * Deq)
        @test abs(p3_rain_self_collection(qr, Nr, 1 / Deq, ρa)) < 1e-12 * s_small
        @test 581 < 1e6 * Deq < 582
        # Beyond it breakup is a net number SOURCE (negative sink).
        @test p3_rain_self_collection(qr, Nr, 1 / 1e-3, ρa) < 0
        # Monotone: number loss decreases with mean size above D_break.
        rates = [p3_rain_self_collection(qr, Nr, 1 / D, ρa) for D in 300e-6:50e-6:1e-3]
        @test all(diff(rates) .< 0)
        @test p3_rain_self_collection(0.0, Nr, 1e4, ρa) == 0
    end

    @testset "Cober & List rime density" begin
        @test p3_cober_list_density(1.0) ≈ 159.5 rtol = 1e-12
        @test p3_cober_list_density(8.0) ≈ 611.0 rtol = 1e-12
        @test p3_cober_list_density(8.0 + 1e-9) ≈ 611.0 rtol = 1e-9   # continuous
        @test p3_cober_list_density(12.0) ≈ 900.0 rtol = 1e-12
        # clamped outside [1, 12]
        @test p3_cober_list_density(0.1) == p3_cober_list_density(1.0)
        @test p3_cober_list_density(50.0) == p3_cober_list_density(12.0)
        # Monotone and inside the P3 bounds [50, 900] for every input.
        Ris = range(0, 15, length = 3001)
        ρs = p3_cober_list_density.(Ris)
        @test all(diff(ρs) .>= 0)
        @test all(pp.ice.ρ_rim_min .<= ρs .<= pp.ice.ρ_rim_max)
        # Physical dependences through R_i = -r[μm] V / T_c:
        # denser for larger drops, faster impact, warmer surface.
        ρ(D, V, T) = p3_rime_density(D, V, T)
        # base case R_i = 10 μm × 2 m/s / 5 K = 4
        ρb = ρ(20e-6, 2.0, 268.15)
        @test ρb ≈ p3_cober_list_density(4.0) rtol = 1e-9
        @test ρ(30e-6, 2.0, 268.15) > ρb       # R_i = 6
        @test ρ(20e-6, 3.0, 268.15) > ρb       # R_i = 6
        @test ρ(20e-6, 2.0, 270.15) > ρb       # R_i = 6.67
        @test ρ(20e-6, 2.0, 265.15) < ρb       # R_i = 2.5
        # R_i = 5 μm × 1 m/s / 10 K = 0.5 → clamped to 1 → 159.5
        @test ρ(10e-6, 1.0, 263.15) ≈ 159.5 rtol = 1e-12
        # R_i = 10 μm × 4 m/s / 5 K = 8 → 611
        @test ρ(20e-6, 4.0, 268.15) ≈ 611.0 rtol = 1e-9
    end

    # ---------------------------------------------------------------------
    # Size-distribution-integrated processes
    # ---------------------------------------------------------------------
    keys_mass = (:evap_r, :dep, :sub, :auto, :accr, :rime_c, :rime_r, :shed, :melt)
    # number fields that belong to each mass process
    companions = Dict(:evap_r => (:n_evap_r,), :dep => (), :sub => (:n_sub,),
                      :auto => (:n_auto_r, :n_auto_c), :accr => (:n_accr_c,),
                      :rime_c => (:n_rime_c,), :rime_r => (:n_rime_r,),
                      :shed => (:n_shed_c, :n_shed_r), :melt => (:n_melt,))
    function only(r, k)
        z = map(v -> zero(v), r)
        z = merge(z, (ρ_rime_c = r.ρ_rime_c,))
        merge(z, NamedTuple{(k, companions[k]...)}(map(x -> r[x], (k, companions[k]...))))
    end
    # mixed-phase cold state (riming, deposition), warm melting state,
    # subsaturated cold state (sublimation, evaporation)
    cold = P3ParcelState(T = 263.15, p = 7.0e4, q_v = qsat(263.15, 7.0e4),
                         q_c = 3e-4, N_c = 1e8, q_r = 2e-4, N_r = 2e4,
                         q_i = 3e-4, N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)
    warm = P3ParcelState(T = 276.15, p = 8.0e4, q_v = 0.95 * qsat(276.15, 8.0e4),
                         q_c = 3e-4, N_c = 1e8, q_r = 2e-4, N_r = 2e4,
                         q_i = 3e-4, N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)
    dry = P3ParcelState(T = 258.15, p = 6.0e4, q_v = 0.7 * qsati(258.15, 6.0e4),
                        q_r = 2e-4, N_r = 2e4,
                        q_i = 3e-4, N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)

    @testset "Each process conserves water and heats with its latent heat" begin
        nact = 0
        for st in (cold, warm, dry)
            r = p3_process_rates(st)
            cpm = MT.heat_capacity_moist(MT.PhasePartition(p3_total_water(st),
                                         st.q_c + st.q_r, st.q_i), c)
            Lv, Lf = MT.latent_heat_vapor(st.T, c), MT.latent_heat_fusion(st.T, c)
            Ls = MT.latent_heat_sublim(st.T, c)
            Lexp = Dict(:evap_r => -Lv, :dep => Ls, :sub => -Ls, :auto => 0.0,
                        :accr => 0.0, :rime_c => Lf, :rime_r => Lf, :shed => 0.0,
                        :melt => -Lf)
            for k in keys_mass
                r[k] > 0 || continue
                nact += 1
                d = p3_tendencies(st, only(r, k))
                @test abs(d.q_v + d.q_c + d.q_r + d.q_i) <= 1e-15 * r[k]
                @test cpm * d.T ≈ Lexp[k] * r[k] atol = 1e-12 * Ls * r[k]
            end
            # and the sum of all of them
            d = p3_tendencies(st, r)
            tot = sum(r[k] for k in keys_mass)
            @test abs(d.q_v + d.q_c + d.q_r + d.q_i) <= 1e-15 * tot
            @test cpm * d.T ≈ -Lv * d.q_v + Lf * d.q_i rtol = 1e-12
        end
        @printf("    %d active (state, process) pairs checked\n", nact)
        @test nact >= 12          # every process appears in at least one state
    end

    @testset "Rain evaporation: only in subsaturation, ventilated" begin
        base = (T = 280.0, p = 9.0e4, q_r = 5e-4, N_r = 5e3)
        ev(S) = p3_process_rates(P3ParcelState(; base...,
                    q_v = (1 + S) * qsat(280.0, 9.0e4))).evap_r
        @test ev(0.0) == 0
        @test ev(0.05) == 0                 # no evaporation when supersaturated
        e1, e2 = ev(-0.1), ev(-0.2)
        @test e1 > 0
        @printf("    evap(S=-0.2)/evap(S=-0.1) = %.6f (linear in S)\n", e2 / e1)
        @test e2 / e1 ≈ 2 rtol = 2e-3      # ρ_air changes slightly with q_v
        # near-monodisperse small drops (μ_r = 400): f_v → 1 and the bulk rate
        # reduces to N 2π D S/(A+B) with Superdroplet's F_k + F_d.
        ppm = P3ProcessParameters{Float64}(μ_r = 400.0)
        Dm, Nr = 40e-6, 1e8
        qr = Nr * π / 6 * 1000 * Dm^3
        st = P3ParcelState(T = 280.0, p = 9.0e4, q_v = 0.8 * qsat(280.0, 9.0e4),
                           q_r = qr, N_r = Nr)
        e = p3_process_rates(st, ppm).evap_r
        AB = (Superdroplet.fk_thermal(280.0) + Superdroplet.fd_vapor(280.0, 9.0e4)) / 1000
        eref = Nr * 2π * Dm * 0.2 / AB
        @printf("    monodisperse 40 μm drops: evap / N 2πDS/(A+B) = %.5f\n", e / eref)
        @test e / eref ≈ 1 atol = 0.02
        # number removed in proportion to mass
        r = p3_process_rates(P3ParcelState(; base..., q_v = 0.8 * qsat(280.0, 9.0e4)))
        @test r.n_evap_r ≈ r.evap_r * base.N_r / base.q_r rtol = 1e-14
    end

    @testset "Ventilation coefficients" begin
        for k in (:ice, :rain)
            @test p3_ventilation(0.0, k) == 1
            @test all(p3_ventilation(x, k) >= 1 for x in 0:0.01:20)
        end
        @test p3_ventilation(1 - 1e-12, :ice) ≈ p3_ventilation(1.0, :ice) atol = 1e-10
        @test abs(p3_ventilation(1.4 - 1e-12, :rain) - p3_ventilation(1.4, :rain)) < 6e-4
        @test p3_ventilation(10.0, :ice) ≈ 0.86 + 2.8
        @test p3_ventilation(10.0, :rain) ≈ 0.78 + 3.08
    end

    @testset "Ice deposition and sublimation" begin
        T, p = 258.15, 6.0e4
        base = (T = T, p = p, q_i = 1e-4, N_i = 1e5)
        rr(Si) = p3_process_rates(P3ParcelState(; base..., q_v = (1 + Si) * qsati(T, p)))
        @test rr(0.0).dep == 0 && rr(0.0).sub == 0
        @test rr(0.1).dep > 0 && rr(0.1).sub == 0
        @test rr(-0.1).sub > 0 && rr(-0.1).dep == 0          # sublimation only if S_i < 0
        @test rr(-0.1).n_sub ≈ rr(-0.1).sub * base.N_i / base.q_i rtol = 1e-14
        @printf("    dep(S_i=+0.1) = %.4e, sub(S_i=-0.1) = %.4e kg/kg/s\n",
                rr(0.1).dep, rr(-0.1).sub)
        # Wegener-Bergeron-Findeisen: at water saturation below 0 °C the
        # ice grows while rain/cloud do not evaporate.
        wbf = p3_process_rates(P3ParcelState(; base..., q_v = qsat(T, p),
                                             q_r = 1e-4, N_r = 1e4))
        @test wbf.dep > 0 && wbf.evap_r == 0
        # No vapour growth of ice at or above 0 °C (melting takes over).
        hot = p3_process_rates(P3ParcelState(T = 275.0, p = 8e4, q_v = 1.1 * qsati(275.0, 8e4),
                                             q_i = 1e-4, N_i = 1e5))
        @test hot.dep == 0 && hot.sub == 0

        # Benchmark: small ice spheres (mean size ~3 μm, regime 1, f_v ≈ 1):
        # dep = N 4π E[D/2] S/(A+B) = N 2π (μ+1)/λ S/(A+B), A+B for ice
        # built from the MoistThermodynamics/Superdroplet ingredients.
        qi, Ni = 1e-6, 1e7            # mean size ~3 μm, inside P3's 2 μm limiter
        st = P3ParcelState(T = T, p = p, q_v = 1.05 * qsati(T, p), q_i = qi, N_i = Ni)
        dep = p3_process_rates(st).dep
        md = p3_mass_dimension(0.0, 400.0)
        λ = p3_slope(qi, Ni, md; λ_min = 1 / 5e-3, λ_max = 1 / 2e-6)
        Ls = MT.latent_heat_sublim(T, c); Rv = c.gas_constant / c.molmass_water
        Ka = Superdroplet.thermal_conductivity(T); Dv = Superdroplet.vapor_diffusivity(T, p)
        AB = (Ls / (Rv * T) - 1) * Ls / (Ka * T) +
             Rv * T / (Dv * MT.saturation_vapor_pressure_ice(T, c))
        dref = Ni * 2π / λ * 0.05 / AB
        @printf("    small-sphere deposition / analytic = %.5f (mean D = %.2f μm)\n",
                dep / dref, 1e6 / λ)
        @test dep / dref ≈ 1 atol = 2e-3

        # Ventilation raises the growth of large snow: E[C f_v]/E[C] > 1.
        pp0 = pp
        snow = P3ParcelState(T = T, p = p, q_v = 1.05 * qsati(T, p), q_i = 1e-3, N_i = 1e5)
        d_vent = p3_process_rates(snow).dep
        md2 = p3_mass_dimension(0.0, 400.0)
        # unventilated reference: same PSD, f_v ≡ 1
        λ2 = p3_slope(1e-3, 1e5, md2; λ_min = 1 / 5e-3, λ_max = 1 / 2e-6)
        EC = Aeolus.P3Microphysics._p3_psd_expectation(0.0, λ2, md2) do D
            reg = p3_regime(D, md2)
            ((reg == 1 || reg == 3 ? 1.0 : 0.48) * D / 2,)
        end[1]
        d_novent = 1e5 * 4π * EC * 0.05 / AB
        @printf("    snow (mean D = %.2f mm): ventilation factor %.3f\n",
                1e3 / λ2, d_vent / d_novent)
        @test d_vent > d_novent
        @test 1.1 < d_vent / d_novent < 3

        # Sublimation preserves rime density and rime fraction; deposition
        # adds unrimed mass, so F_rim falls.
        r = rr(-0.1)
        st2 = P3ParcelState(; base..., q_v = 0.9 * qsati(T, p), q_rim = 4e-5, B_rim = 4e-5 / 300)
        r2 = p3_process_rates(st2)
        d = p3_tendencies(st2, only(r2, :sub))
        @test d.q_rim / d.q_i ≈ st2.q_rim / st2.q_i rtol = 1e-14
        @test d.B_rim / d.q_i ≈ st2.B_rim / st2.q_i rtol = 1e-14
        st3 = P3ParcelState(; base..., q_v = 1.1 * qsati(T, p), q_rim = 4e-5, B_rim = 4e-5 / 300)
        d3 = p3_tendencies(st3)
        @test d3.q_rim == 0 && d3.q_i > 0                     # F_rim decreases
    end

    @testset "Riming: rime mass, volume and Cober-List density" begin
        r = p3_process_rates(cold)
        @test r.rime_c > 0 && r.rime_r > 0 && r.shed == 0
        d = p3_tendencies(cold, r)
        @test d.q_rim > 0 && d.B_rim > 0
        # B_rim budget: new rime at its own density (dep/sub contribute 0 here
        # since the cold state is at water saturation → only deposition)
        @test d.B_rim ≈ r.rime_c / r.ρ_rime_c + r.rime_r / pp.ρ_rim_rain rtol = 1e-12
        @test d.q_rim ≈ r.rime_c + r.rime_r rtol = 1e-14
        # rime fraction grows over a short step (riming ≫ deposition here)
        dt = 1.0
        F0 = cold.q_rim / cold.q_i
        F1 = (cold.q_rim + dt * d.q_rim) / (cold.q_i + dt * d.q_i)
        @printf("    riming: rime_c = %.3e, rime_r = %.3e kg/kg/s, ρ_rime_c = %.1f, F_rim %.4f → %.4f in 1 s\n",
                r.rime_c, r.rime_r, r.ρ_rime_c, F0, F1)
        @test F1 > F0
        # Collected-drop number: droplets removed at the mean droplet mass
        @test r.n_rime_c ≈ r.rime_c * cold.N_c / cold.q_c rtol = 1e-14
        # Density within Cober & List / P3 bounds over a sweep of conditions
        ρmin, ρmax = Inf, -Inf
        for T in 240.0:4.0:272.0, qc in (1e-5, 3e-4, 2e-3), Nc in (3e7, 3e8),
            qi in (1e-5, 1e-3)
            st = P3ParcelState(T = T, p = 7e4, q_v = qsat(T, 7e4), q_c = qc, N_c = Nc,
                               q_i = qi, N_i = 1e4, q_rim = 0.5qi, B_rim = 0.5qi / 500)
            ρr = p3_process_rates(st).ρ_rime_c
            ρmin, ρmax = min(ρmin, ρr), max(ρmax, ρr)
        end
        @printf("    Cober-List rime density over sweep: %.1f – %.1f kg/m³\n", ρmin, ρmax)
        @test 50 <= ρmin && ρmax <= 900
        @test ρmax > ρmin                                     # it does vary
        # No riming at or above 0 °C: collected cloud is shed to rain as 1 mm drops.
        rw = p3_process_rates(warm)
        @test rw.rime_c == 0 && rw.rime_r == 0 && rw.shed > 0
        @test rw.n_shed_r ≈ rw.shed / (π / 6 * 1000 * 1e-3^3) rtol = 1e-14
        dw = p3_tendencies(warm, only(rw, :shed))
        @test dw.q_r ≈ -dw.q_c rtol = 1e-15
        @test dw.q_i == 0 && dw.T == 0
        # riming is proportional to cloud water (continuous collection)
        c2 = P3ParcelState(cold.T, cold.p, cold.q_v, 2cold.q_c, 2cold.N_c, cold.q_r,
                           cold.N_r, cold.q_i, cold.N_i, cold.q_rim, cold.B_rim)
        @test p3_process_rates(c2).rime_c / r.rime_c ≈ 2 rtol = 1e-3
    end

    @testset "Melting only above 0 °C, conserving mass" begin
        base = (p = 8.0e4, q_i = 3e-4, N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)
        mlt(T, rh) = p3_process_rates(P3ParcelState(; base..., T = T,
                                      q_v = rh * qsat(T, 8.0e4))).melt
        for T in (250.0, 265.0, 273.0, 273.15), rh in (0.5, 1.0)
            @test mlt(T, rh) == 0
        end
        ms = [mlt(T, 1.0) for T in 273.65:0.5:283.15]
        @test all(ms .> 0)
        @test all(diff(ms) .> 0)                              # faster when warmer
        @printf("    melt rate at +1 K, +5 K, +10 K (RH 100%%): %.3e %.3e %.3e kg/kg/s\n",
                mlt(274.15, 1.0), mlt(278.15, 1.0), mlt(283.15, 1.0))
        # Evaporative cooling can hold the particle below 0 °C: dry air at
        # +0.5 °C gives no melting.
        @test mlt(273.65, 0.3) == 0
        # conservation: ice → rain, one drop per particle, rime melts in proportion
        st = P3ParcelState(; base..., T = 278.15, q_v = qsat(278.15, 8e4))
        r = p3_process_rates(st)
        d = p3_tendencies(st, only(r, :melt))
        @test d.q_r ≈ -d.q_i rtol = 1e-15
        @test d.q_v == 0 && d.q_c == 0
        @test d.N_r ≈ -d.N_i rtol = 1e-15
        @test d.N_i ≈ -r.melt * st.N_i / st.q_i rtol = 1e-14
        @test d.q_rim / d.q_i ≈ st.q_rim / st.q_i rtol = 1e-14
        @test d.B_rim / d.q_i ≈ st.B_rim / st.q_i rtol = 1e-14
        cpm = MT.heat_capacity_moist(MT.PhasePartition(p3_total_water(st), 0.0, st.q_i), c)
        @test cpm * d.T ≈ -MT.latent_heat_fusion(st.T, c) * r.melt rtol = 1e-12
    end

    # ---------------------------------------------------------------------
    # Full source-term step
    # ---------------------------------------------------------------------
    allfields(x) = (x.q_v, x.q_c, x.N_c, x.q_r, x.N_r, x.q_i, x.N_i, x.q_rim, x.B_rim)

    @testset "Full step: conservation, energy, positivity" begin
        # Energy identity of the process part (no adjustment, no clean-up
        # triggered): c_pm ΔT = -L_v Δq_v + L_f Δq_i exactly.
        for st in (cold, warm, dry)
            s1 = p3_step(st, 1.0; adjust = false)
            cpm = MT.heat_capacity_moist(MT.PhasePartition(p3_total_water(st),
                                         st.q_c + st.q_r, st.q_i), c)
            lhs = cpm * (s1.T - st.T)
            rhs = -MT.latent_heat_vapor(st.T, c) * (s1.q_v - st.q_v) +
                   MT.latent_heat_fusion(st.T, c) * (s1.q_i - st.q_i)
            @test lhs ≈ rhs rtol = 1e-10
            @test abs(p3_total_water(s1) - p3_total_water(st)) <= 4eps() * p3_total_water(st)
        end

        # Long integrations with a large step in three regimes.
        maxerr = 0.0
        for (st, dt, n) in ((cold, 60.0, 120), (warm, 60.0, 120), (dry, 60.0, 120))
            W0 = p3_total_water(st)
            x = st
            for _ in 1:n
                x = p3_step(x, dt)
                maxerr = max(maxerr, abs(p3_total_water(x) - W0) / W0)
                @test all(>=(0), allfields(x))
                @test x.q_rim <= x.q_i
                x.q_rim > 0 && @test 50 - 1e-9 <= x.q_rim / x.B_rim <= 900 + 1e-9
            end
        end
        @printf("    total water over 3 × 120 steps of 60 s: max rel. drift %.2e\n", maxerr)
        @test maxerr < 1e-14

        # An absurd step (1 day) still gives a non-negative, conserving state
        # that does not overshoot saturation.
        big = p3_step(cold, 86400.0; adjust = false)
        @test all(>=(0), allfields(big))
        @test abs(p3_total_water(big) - p3_total_water(cold)) <= 4eps() * p3_total_water(cold)
        @test big.q_v >= qsati(big.T, big.p) * (1 - 1e-3)     # no deposition overshoot
        dd = p3_step(dry, 86400.0; adjust = false)
        @printf("    1-day step: cold q_v/q_si = %.5f, dry q_v/q_s = %.5f\n",
                big.q_v / qsati(big.T, big.p), dd.q_v / qsat(dd.T, dd.p))
        @test dd.q_v <= qsat(dd.T, dd.p) * (1 + 1e-3)          # no evaporation overshoot
        @test all(>=(0), allfields(dd))
    end

    @testset "Parcel regimes" begin
        # Glaciation (WBF + riming): a mixed-phase parcel at -10 °C loses
        # all its liquid and relaxes to ice saturation.
        x = cold
        for _ in 1:120
            x = p3_step(x, 30.0)
        end
        @printf("    glaciation: q_c %.2e, q_r %.2e, q_i %.3e, q_v/q_si %.5f, F_rim %.3f, ρ_rim %.1f\n",
                x.q_c, x.q_r, x.q_i, x.q_v / qsati(x.T, x.p), x.q_rim / x.q_i,
                x.q_rim / x.B_rim)
        @test x.q_c == 0
        @test x.q_r < 1e-9
        @test x.q_v / qsati(x.T, x.p) ≈ 1 atol = 1e-3
        @test x.T > cold.T                        # net freezing/deposition warms
        @test x.q_rim > cold.q_rim                # rimed on the way

        # Melting: all ice ends up as rain above 0 °C, cooling the air.
        y = P3ParcelState(T = 278.15, p = 8e4, q_v = qsat(278.15, 8e4),
                          q_i = 3e-4, N_i = 5e4, q_rim = 1e-4, B_rim = 1e-4 / 400)
        for _ in 1:60
            y = p3_step(y, 10.0)
        end
        @printf("    melting: q_i %.2e → %.2e, q_r %.3e, N_r %.3e, ΔT %.3f K\n",
                3e-4, y.q_i, y.q_r, y.N_r, y.T - 278.15)
        @test y.q_i == 0 && y.N_i == 0 && y.q_rim == 0
        @test y.q_r > 2.5e-4
        @test y.T < 278.15

        # Warm rain: cloud water turns into rain monotonically
        # (autoconversion then accretion), total water conserved.
        z = P3ParcelState(T = 290.0, p = 9e4, q_v = qsat(290.0, 9e4),
                          q_c = 1.5e-3, N_c = 5e7)
        qcs, qrs = [z.q_c], [z.q_r]
        for _ in 1:60
            z = p3_step(z, 30.0)
            push!(qcs, z.q_c); push!(qrs, z.q_r)
        end
        @printf("    warm rain: q_c %.3e → %.3e, q_r 0 → %.3e in 30 min\n",
                qcs[1], qcs[end], qrs[end])
        @test all(diff(qcs) .<= 1e-12) && all(diff(qrs) .>= -1e-12)
        @test qrs[end] > 0.5e-3
    end
end
