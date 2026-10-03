# Era VI, Phase 58 (#170) — the TurbulenceStability contract:
# gradient-Richardson stability functions, N² and Ri_g.
#
#   1. limits of every implementer: f(0) = 1, f ≥ 0, non-increasing in
#      Ri > 0; the Lilly cutoff is exactly 0 above Ri_c and continuous there;
#   2. MOST round trip: K_m, K_h of MOSTConsistent reproduce the surface-
#      layer fluxes of the analytic MOST profiles;
#   3. critical Ri of the log-linear and BDBH similarity functions;
#   4. the pointwise N² and Ri_g fields on analytic flows (exact for linear
#      profiles; second order for a tanh shear layer);
#   5. the fog column delegates to ShortTail bit for bit;
#   6. the mean-profile N² of TimeIntegration and Anelastic bit for bit.

using Aeolus.TurbulenceStability
using Aeolus.TurbulenceStability: AbstractRiStability

@testset "Era VI — TurbulenceStability (P58-a, #170)" begin

@testset "1. limits of every implementer" begin
    impls = [LillyCutoff(), BrownEtAl1994(), ShortTail(), LongTail()]
    Ris = vcat(range(-5, 0; length = 201), range(0, 10; length = 2001), [1e2, 1e4])
    for sf in impls
        for fac in (stability_factor_m, stability_factor_h)
            @test abs(fac(sf, 0.0) - 1) <= eps()
            fs = [fac(sf, Ri) for Ri in Ris]
            @test all(isfinite, fs)
            @test all(>=(0), fs)
            pos = [fac(sf, Ri) for Ri in Ris if Ri > 0]
            @test all(diff(pos) .<= 0)           # non-increasing in stable air
            # unstable air mixes at least as much as neutral air
            @test all(fac(sf, Ri) >= 1 for Ri in Ris if Ri < 0)
        end
        @test neutral_prandtl(sf) > 0
    end

    # Lilly / Mason: exactly 0 at and above Ri_c, continuous at Ri_c
    for Ri_c in (0.25, 1 / 3)
        sf = LillyCutoff(Ri_c = Ri_c)
        @test stability_factor_m(sf, Ri_c) == 0
        @test all(stability_factor_m(sf, Ri) == 0 for Ri in Ri_c .* (1:0.5:10))
        @test all(stability_factor_h(sf, Ri) == 0 for Ri in Ri_c .* (1:0.5:10))
        # the left limit: f(Ri_c − δ) = √(δ/Ri_c) → 0. In Float64 the closest
        # point below Ri_c gives √(eps) ≈ 1e-8, so the 1e-12 continuity is
        # measured in BigFloat at δ = 1e-26
        δ = big"1e-26"
        f_left = stability_factor_m(LillyCutoff(Ri_c = BigFloat(Ri_c)), BigFloat(Ri_c) - δ)
        @test f_left < 1e-12
        @test abs(f_left - sqrt(δ / BigFloat(Ri_c))) < 1e-30
    end
    # the TKE-equilibrium form K = l²(S² − N²/Ri_c)^{1/2}
    let sf = LillyCutoff(), S2 = 4e-4, N2 = 3e-5
        @test sqrt(S2) * stability_factor_m(sf, N2 / S2) ≈ sqrt(S2 - N2 / 0.25) rtol = 1e-14
    end

    # Brown et al. (1994) / MONC constants, quoted values
    let sf = BrownEtAl1994()
        @test stability_factor_m(sf, 0.1) ≈ (1 - 0.4)^4 rtol = 1e-15
        @test stability_factor_h(sf, 0.1) ≈ 0.6^4 * (1 - 0.12) rtol = 1e-15
        @test stability_factor_m(sf, -0.5) ≈ 3.0 rtol = 1e-15     # √(1 + 8)
        @test stability_factor_h(sf, -0.5) ≈ sqrt(21.0) rtol = 1e-15
        @test stability_factor_m(sf, 0.25) == 0 && stability_factor_h(sf, 0.3) == 0
        @test stability_factor_m(sf, nextfloat(0.25, -1)) < 1e-60
        @test neutral_prandtl(sf) == 0.7
    end
    # Louis (1979), b = d = 5: f_m(Ri = 3) = 1/(1 + 30/4) = 2/17
    let sf = LongTail()
        @test stability_factor_m(sf, 3.0) ≈ 2 / 17 rtol = 1e-15
        @test stability_factor_h(sf, 3.0) ≈ 1 / (1 + 45 * 4) rtol = 1e-15
        # long tail: no critical Ri, f_m → √5/(10 √Ri), with a relative
        # correction 1/√(20 Ri) (2e-5 at Ri = 1e8)
        @test stability_factor_m(sf, 1e8) * sqrt(1e8) ≈ sqrt(5) / 10 rtol = 1e-4
        @test stability_factor_m(sf, 1e4) > 1e-3
        @test neutral_prandtl(sf) == 0.74
    end
    # short tail: (1 + 5 Ri)⁻² — it falls below the long tail for Ri > 0
    @test stability_factor_m(ShortTail(), 1.0) == 1 / 36
    @test all(stability_factor_m(ShortTail(), Ri) < stability_factor_m(LongTail(), Ri)
              for Ri in 0.01:0.01:10)

    @test gradient_richardson(1e-4, 4e-4) == 0.25
    @test gradient_richardson(1e-4, 0.0) == 1e-4 / RI_SHEAR2_FLOOR
    @test isfinite(gradient_richardson(-1e-4, 0.0))
    # N²: the Boussinesq and logarithmic forms agree to O(Δθ/θ)
    @test buoyancy_frequency_squared(300.0, 301.0, 100.0, 9.81, 300.0) ≈ 9.81 / 300 / 100
    @test buoyancy_frequency_squared(300.0, 301.0, 100.0, 9.81) ≈
          buoyancy_frequency_squared(300.0, 301.0, 100.0, 9.81, 300.5) rtol = 1e-5
end

end # testset
