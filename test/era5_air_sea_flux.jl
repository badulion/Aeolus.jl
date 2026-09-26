# Phase 52 — wave-dependent sea-surface roughness (issue #39).
#
# COARE stability functions and gustiness, Charnock / COARE / wave-age /
# wave-steepness momentum roughness, scalar roughness, the COARE-style
# bulk algorithm, and the SeaSurface implementer of the Era V contract.
#
# Reference values come from (i) the NOAA-PSL COARE reference
# implementation (github.com/NOAA-PSL/COARE-algorithm, commit bd1cac8),
# whose own test data and output are in test/data/coare35_*.txt, and
# (ii) numbers printed in Fairall et al. (2003, J. Climate 16) and
# Edson et al. (2013, J. Phys. Oceanogr. 43). See test/data/README_coare.md.

using Test, Printf
using Aeolus
using Aeolus.SurfaceExchange
using Aeolus.AirSeaFlux

const ASF = Aeolus.AirSeaFlux

@testset "Phase 52 — AirSeaFlux: wave-dependent roughness (#39)" begin

    @testset "COARE stability: normalisation, φ–ψ consistency, reference code" begin
        for v in (:coare30, :coare35)
            sf = COAREStability(v)
            @test sf isa AbstractStabilityFunctions
            for f in (psi_m, psi_h)
                @test f(sf, 0.0) == 0
                @test abs(f(sf, -1e-9)) < 1e-8 && abs(f(sf, 1e-9)) < 1e-8
            end
            for f in (phi_m, phi_h)
                @test f(sf, 0.0) == 1
                @test f(sf, -1e-9) ≈ f(sf, 1e-9) atol = 1e-7
            end
            # dψ/dζ = (1 - φ)/ζ across the blended unstable branch — this
            # checks the analytic φ of the blend, including the f′ term.
            worst = 0.0
            for ζ in (-50.0, -10.0, -3.0, -1.0, -0.3, -0.05, 0.05, 0.3, 1.0, 3.0, 10.0),
                (ψ, φ) in ((psi_m, phi_m), (psi_h, phi_h))
                h = 1e-4 * max(abs(ζ), 0.1)
                dψ = (-ψ(sf, ζ + 2h) + 8ψ(sf, ζ + h) - 8ψ(sf, ζ - h) + ψ(sf, ζ - 2h)) / (12h)
                worst = max(worst, abs(dψ - (1 - φ(sf, ζ)) / ζ))
            end
            @test worst < 1e-7
            @info "COAREStability($v) φ–ψ consistency" max_err = worst
        end

        # Against the NOAA-PSL COARE 3.5 reference (Python meteo.psiu_26 /
        # psit_26, commit bd1cac8), evaluated at these ζ. The reference
        # uses x^0.3333 for the cube root and the rounded 14.28 / 8.525 in
        # the stable heat function, so agreement is to ~1e-3, not machine.
        ζs = (-10.0, -3.0, -1.0, -0.3, -0.05, 0.05, 0.3, 1.0, 3.0, 10.0)
        ref_u = (2.7054530992, 1.80713472525, 1.11042086012, 0.572291193057, 0.154980480191,
                 -0.257718362426, -1.4805245613, -4.39257224887, -9.85231262359, -17.6172226961)
        ref_t = (3.70791670101, 2.71506638768, 1.86535940766, 1.03813567997, 0.299540089787,
                 -0.25285660792, -1.44810646799, -4.43757146837, -11.0897487826, -29.6682557043)
        sf = COAREStability(:coare35)
        err_u = maximum(abs(psi_m(sf, ζ) - r) for (ζ, r) in zip(ζs, ref_u))
        err_t = maximum(abs(psi_h(sf, ζ) - r) for (ζ, r) in zip(ζs, ref_t))
        @test err_u < 2e-3
        @test err_t < 6e-3
        @info "COARE 3.5 ψ vs NOAA reference code" max_abs_err_psi_m = err_u max_abs_err_psi_h = err_t

        # Free-convection limit: φ → (1 - γ_c ζ)^{-1/3}, i.e. the local
        # slope d ln φ / d ln(-ζ) → -1/3 (Kansas would give -1/4, -1/2).
        for (φ, γc) in ((phi_m, sf.γc_m), (phi_h, sf.γc_h))
            ζ1, ζ2 = -1e3, -1e4
            slope = log(φ(sf, ζ2) / φ(sf, ζ1)) / log(ζ2 / ζ1)
            @test slope ≈ -1 / 3 atol = 2e-3
            @test φ(sf, -1e4) ≈ (1 + γc * 1e4)^(-1 / 3) rtol = 1e-6
        end
        bd = BusingerDyerBeljaarsHoltslag()
        kansas_slope = log(phi_h(bd, -1e4) / phi_h(bd, -1e3)) / log(10)
        @test kansas_slope ≈ -1 / 2 atol = 1e-3
        # Near neutral the blend is Kansas: f = ζ² ⇒ deviation O(ζ²).
        @test abs(psi_h(sf, -0.01) - psi_h(sf.heat, -0.01)) < 1e-4 * abs(psi_h(sf.heat, -0.01)) + 1e-6
        # Stable side is exactly Beljaars–Holtslag with COARE coefficients.
        @test psi_m(sf, 2.0) == psi_m(sf.momentum, 2.0)
        @test psi_h(COAREStability(:coare30), 2.0) == psi_h(bd, 2.0)
        @test_throws ArgumentError COAREStability(:coare40)
    end

    @testset "Convective gustiness" begin
        g = ConvectiveGustiness()
        @test gustiness_velocity(g, -1e-3) == g.w_min
        @test gustiness_velocity(g, 0.0) == g.w_min
        B = 5e-3                                     # m² s⁻³, a warm-pool afternoon
        @test convective_velocity_scale(B, 600.0) ≈ cbrt(3.0)
        @test gustiness_velocity(g, B) ≈ 1.2 * cbrt(B * 600)
        # w_g ∝ B^{1/3} z_i^{1/3}
        @test gustiness_velocity(g, 8B) ≈ 2 * gustiness_velocity(g, B)
        @test gustiness_velocity(ConvectiveGustiness(z_i = 4800.0), B) ≈ 2 * gustiness_velocity(g, B)
    end
end
