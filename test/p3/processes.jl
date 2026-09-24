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
end
