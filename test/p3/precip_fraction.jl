# Phase 44 (issue #8): sub-grid precipitation fraction
# (src/p3/precip_fraction.jl).
#
# Asserted: fractions of one recover the grid-mean scheme; with f < 1 the
# KK2000 rates change by the analytic factors their exponents imply; rain
# under cloud does not evaporate; the overlap assumptions give their
# analytic two-layer values; water is conserved.

@testset "P3 sub-grid precipitation fraction (Phase 44)" begin
    qsat(T, p) = Aeolus.MoistThermodynamics.saturation_specific_humidity(
        T, p, Aeolus.MoistThermodynamics.DEFAULT_CONSTANTS)
    fields(s) = [getfield(s, f) for f in fieldnames(typeof(s))]
    warm = P3ParcelState(T = 285.0, p = 9.0e4, q_v = 0.8 * qsat(285.0, 9e4),
                         q_c = 5e-4, N_c = 1e8, q_r = 4e-4, N_r = 2e4)
    cold = P3ParcelState(T = 263.15, p = 7.0e4, q_v = qsat(263.15, 7e4),
                         q_c = 3e-4, N_c = 1e8, q_r = 2e-4, N_r = 1e4,
                         q_i = 4e-4, N_i = 2e5, q_rim = 1e-4, B_rim = 1e-4 / 400)

    @testset "Fractions of one: the grid-mean scheme" begin
        @test all(values_agree(fields(p3_step(s, 10.0; cloud_fraction = 1.0, precip_fraction = 1.0)),
                               fields(p3_step(s, 10.0))) for s in (warm, cold))
    end

    @testset "Analytic nonlinearity factors and clear-air evaporation" begin
        g = p3_process_rates(warm)
        e = P3_DEFAULT_PROCESS.kk_acc_exp                  # 1.15
        f = 0.25
        # rain on f of a cloudy cell: f · f^-1.15; cloud and rain both on f: f^(1-2·1.15)
        @test p3_fractional_process_rates(warm, 1.0, f).accr / g.accr ≈ f^(1 - e) rtol = 1e-12
        @test p3_fractional_process_rates(warm, f, f).accr / g.accr ≈ f^(1 - 2e) rtol = 1e-12
        # rain entirely under saturated cloud does not evaporate; half in
        # clear air evaporates at the clear-air deficit
        T, p = 285.0, 9.0e4
        qs = qsat(T, p)
        s = P3ParcelState(T = T, p = p, q_v = 0.8qs, q_c = 3e-4, N_c = 1e8, q_r = 4e-4, N_r = 2e4)
        @test p3_fractional_process_rates(s, 0.5, 0.5).evap_r == 0
        byhand = 0.5 * p3_process_rates(P3ParcelState(T = T, p = p, q_v = (0.8qs - 0.5qs) / 0.5,
                                                      q_r = 4e-4, N_r = 2e4)).evap_r
        @test p3_fractional_process_rates(s, 0.5, 1.0).evap_r ≈ byhand rtol = 1e-12
    end

    @testset "Vertical overlap: analytic values" begin
        ov = p3_precipitation_overlap
        @test ov(0.3, 0.3, 0.5, :maximum) == 0.5
        @test ov(0.3, 0.3, 0.5, :random) ≈ 0.65
        @test ov(0.3, 0.0, 0.5, :maximum_random) ≈ 0.65    # across a gap → random
        nz = 30
        cf = [k in 25:27 ? 0.3 : k in 18:20 ? 0.4 : k in 8:10 ? 0.5 : k in 4:5 ? 0.2 : 0.0
              for k in 1:nz]
        qc = [x > 0 ? 2e-4 : 0.0 for x in cf]
        qr = [k <= 27 ? 1e-4 : 0.0 for k in 1:nz]
        Pm = p3_precipitation_fraction(cf, qc, qr; overlap = :maximum)
        Pmr = p3_precipitation_fraction(cf, qc, qr; overlap = :maximum_random)
        @test Pm[1] ≈ 0.5
        @test Pmr[1] ≈ 1 - 0.7 * 0.6 * 0.5 * 0.8
    end

    @testset "Regional rates with Hallett-Mossop conserve water" begin
        # column step (random overlap of a 2-level deck of 0.4): water conserved
        col = [P3ParcelState(T = 283.0 - 0.6k, p = 9e4 - 800k, q_v = 0.85 * qsat(283.0 - 0.6k, 9e4 - 800k),
                             q_c = k >= 3 ? 4e-4 : 0.0, N_c = k >= 3 ? 1e8 : 0.0,
                             q_r = 3e-4, N_r = 1.5e4) for k in 1:4]
        new, fp = p3_step_column(col, 30.0; cloud_fraction = [0, 0, 0.4, 0.4], overlap = :random)
        @test sum(p3_total_water, new) ≈ sum(p3_total_water, col) rtol = 1e-14
        @test fp[1] ≈ 1 - 0.6^2
        # splinters still produced when the rates come from the regions (#7 + #8)
        s = P3ParcelState{Float64}(268.15, 7e4, 3e-3, 5e-4, 1e8, 1e-3, 1e4,
                                   1e-3, 1e5, 5e-4, 5e-4 / 400)
        frac, _ = p3_step_with_rates(s, 10.0; hallett_mossop = P3_DEFAULT_HM,
                                     cloud_fraction = 0.4, precip_fraction = 0.6)
        @test frac.N_i > p3_step(s, 10.0; cloud_fraction = 0.4, precip_fraction = 0.6).N_i
        @test p3_total_water(frac) ≈ p3_total_water(s) rtol = 1e-14
    end
end
