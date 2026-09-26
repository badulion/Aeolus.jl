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
    @testset "Air viscosity delegates to Sutherland / MoistThermodynamics" begin
        # COARE reference polynomial (Fairall et al. 2003 code), T in °C.
        coare_ν(t) = 1.326e-5 * (1 + 6.542e-3t + 8.301e-6t^2 - 4.84e-9t^3)
        worst = maximum(abs(ASF.air_kinematic_viscosity(273.15 + t, 101325.0) / coare_ν(t) - 1)
                        for t in 0:5:30)
        @test worst < 0.005
        @test ASF.air_kinematic_viscosity(293.15, 101325.0) ≈
              Aeolus.P3Microphysics.p3_air_viscosity(293.15) /
              Aeolus.MoistThermodynamics.air_density(293.15, 101325.0)
        @info "ν(T) vs COARE polynomial, 0–30 °C" max_rel_diff = worst
    end

    @testset "Sea state: deep-water dispersion, fully developed sea" begin
        g = Aeolus.SurfaceExchange.grav()
        s = deep_water_sea_state(8.0, 2.0)
        @test s.λ_p ≈ g * 64 / 2π
        @test s.c_p ≈ s.λ_p / 8.0                  # c = λ/T
        @test s.c_p^2 ≈ g * s.λ_p / 2π             # deep-water dispersion ω² = g k
        # Fairall et al. (2003) eq. 27: "equivalent to C_p/U10n = 1.14".
        for U in (5.0, 10.0, 20.0)
            f = fully_developed_sea_state(U)
            @test f.c_p / U ≈ 1.14 atol = 0.005
            @test f.H_s ≈ 0.0248U^2
        end
        @test wave_age(s, 0.4) == s.c_p / 0.4
    end

    @testset "Momentum roughness: Charnock, smooth flow, COARE 3.0 / 3.5" begin
        g, ν = Aeolus.SurfaceExchange.grav(), 1.5e-5
        # Smooth-flow limit (Smith 1988): R_r = z0 u*/ν → 0.11 as u* → 0.
        for r in (CharnockRoughness(), COARE35Charnock())
            z = sea_roughness_length(r, 1e-3; ν = ν, g = g, U10N = 1.0)
            @test roughness_reynolds(z, 1e-3, ν) ≈ 0.11 rtol = 1e-3
        end
        # Fully rough: z0 g/u*² → α.
        z = sea_roughness_length(CharnockRoughness(), 2.0; ν = ν, g = g)
        @test z * g / 4 ≈ 0.011 rtol = 1e-3
        # COARE 3.0 α: 0.011 → 0.018 linearly between 10 and 18 m/s.
        r30 = COARE30Charnock()
        @test charnock_parameter(r30, 0.3; U10N = 5.0) == 0.011
        @test charnock_parameter(r30, 0.3; U10N = 14.0) ≈ 0.0145
        @test charnock_parameter(r30, 0.3; U10N = 25.0) == 0.018
        # COARE 3.5 α (Edson et al. 2013 + corrigendum): 0.0017 U − 0.005,
        # levelling off at "α ≈ 0.028 around 19 m/s".
        r35 = COARE35Charnock()
        @test charnock_parameter(r35, 0.3; U10N = 10.0) ≈ 0.012
        @test charnock_parameter(r35, 0.3; U10N = 30.0) ≈ 0.0273
        @test abs(charnock_parameter(r35, 0.3; U10N = 30.0) - 0.028) < 1e-3
        @test charnock_parameter(r35, 0.05; U10N = 1.0) == 0      # floored
        @test_throws ArgumentError sea_roughness_length(r35, 0.3; ν = ν)
    end

    @testset "Neutral C_D10N / C_E10N against published COARE numbers" begin
        c30 = (COARE30Charnock(), COARE30ScalarRoughness())
        c35 = (COARE35Charnock(), COARE35ScalarRoughness())
        ntc(r, U) = neutral_transfer_coefficients(r..., U)
        # Self-consistency of the neutral solve: u* = κU/ln(10/z0).
        n = ntc(c35, 12.0)
        @test n.u_star ≈ 0.4 * 12 / log(10 / n.z0m) rtol = 1e-10
        @test n.C_D ≈ (n.u_star / 12)^2 rtol = 1e-10

        # Fairall et al. (2003), section 5: "The COARE 3.0 value [of
        # 10³ C_d10n] at 20 m s⁻¹ is 2.06."
        cd20 = 1e3 * ntc(c30, 20.0).C_D
        @test cd20 ≈ 2.06 rtol = 0.01
        # Fairall et al. (2003): measured C_d10n "about 1.0e-3 at 3 m s⁻¹",
        # C_e10n "from about 1.08e-3 at 5 m s⁻¹ to 1.2e-3 at 18 m s⁻¹",
        # which COARE 3.0 fits "with an rms deviation of 4.0 %".
        cd3 = 1e3 * ntc(c30, 3.0).C_D
        ce5, ce18 = 1e3 * ntc(c30, 5.0).C_E, 1e3 * ntc(c30, 18.0).C_E
        @test cd3 ≈ 1.0 rtol = 0.05
        @test ce5 ≈ 1.08 rtol = 0.04
        @test ce18 ≈ 1.20 rtol = 0.04
        @test ce18 > ce5                         # C_e10n rises with wind
        # C_d10n has its minimum between 2 and 5 m/s and rises both ways
        # (smooth flow below, Charnock above).
        cds = [ntc(c30, U).C_D for U in 1.0:0.5:20.0]
        imin = argmin(cds)
        @test 2.0 <= (1.0:0.5:20.0)[imin] <= 5.0
        @test all(diff(cds[imin:end]) .> 0) && all(diff(cds[1:imin]) .< 0)

        # Edson et al. (2013) eq. 22: fully rough data (U10N > 8.5 m/s)
        # fit u* = 0.062 U10N − 0.28. COARE 3.5 against that fit:
        worst = 0.0
        for U in 10.0:2.5:25.0
            u = ntc(c35, U).u_star
            worst = max(worst, abs(u / (0.062U - 0.28) - 1))
        end
        @test worst < 0.08
        # COARE 3.5 is rougher than 3.0 at high wind, smoother at moderate.
        @test ntc(c35, 20.0).C_D > ntc(c30, 20.0).C_D
        @test ntc(c35, 6.0).C_D < ntc(c30, 6.0).C_D
        # The 3.5 scalar refit was chosen to keep the Dalton number close
        # to COARE 3.0 (reference-code comment).
        ce_dev = maximum(abs(ntc(c35, U).C_E / ntc(c30, U).C_E - 1) for U in 1.0:1.0:25.0)
        @test ce_dev < 0.04
        @info "Neutral transfer coefficients" CD10N_coare30_20ms = cd20 CD10N_coare30_3ms = cd3 CE10N_coare30_5ms = ce5 CE10N_coare30_18ms = ce18 u_star_coare35_vs_Edson_eq22_max_rel = worst CE10N_35_vs_30_max_rel = ce_dev
    end

    @testset "Wave-dependent roughness: wave age and steepness" begin
        g, ν = Aeolus.SurfaceExchange.grav(), 1.5e-5
        wa = WaveAgeCharnock()
        # Edson et al. (2013): "α = 0.013 at the fully developed value of
        # c_p/u* = 32" and "spans 0.011–0.018 for 43 > c_p/u* > 20".
        sea32 = SeaState(c_p = 32 * 0.4, H_s = 2.0, λ_p = 100.0)
        α32 = charnock_parameter(wa, 0.4; sea = sea32)
        @test round(α32, digits = 3) == 0.013
        α43 = charnock_parameter(wa, 0.4; sea = SeaState(c_p = 43 * 0.4, H_s = 2.0, λ_p = 100.0))
        α20 = charnock_parameter(wa, 0.4; sea = SeaState(c_p = 20 * 0.4, H_s = 2.0, λ_p = 100.0))
        @test round(α43, digits = 3) == 0.011
        @test round(α20, digits = 3) == 0.018
        @info "COARE 3.5 wave-age Charnock" α_cp_u32 = α32 α_cp_u43 = α43 α_cp_u20 = α20

        # Oost et al. (2002) α = 50 (c_p/u*)^(-2.5) (Fairall 2003 eq. 24) is
        # recovered exactly from the roughness form (25b) for deep-water
        # waves, since g λ_p/2π = c_p².
        s = deep_water_sea_state(6.0, 1.5)
        for u in (0.2, 0.5, 1.0)
            @test charnock_parameter(OostRoughness(), u; g = g, ν = ν, sea = s) ≈
                  50 * (s.c_p / u)^(-2.5) rtol = 1e-10
        end
        # Sea-state law: rough z0/H_s = D (u*/c_p)².
        ss = SeaStateRoughness()
        u = 0.5
        zr = sea_roughness_length(ss, u; ν = ν, sea = s) - smooth_flow_roughness(u, ν)
        @test zr / s.H_s ≈ 0.091 * (u / s.c_p)^2
        # Taylor–Yelland: rough part independent of u*, grows with steepness.
        ty = TaylorYellandRoughness()
        z1 = sea_roughness_length(ty, 0.3; ν = 0.0, sea = s)
        @test z1 == sea_roughness_length(ty, 0.9; ν = 0.0, sea = s)
        @test z1 ≈ 1200 * 1.5 * (1.5 / s.λ_p)^4.5
        steeper = SeaState(c_p = s.c_p, H_s = 2 * s.H_s, λ_p = s.λ_p)
        @test sea_roughness_length(ty, 0.3; ν = 0.0, sea = steeper) ≈ 2^5.5 * z1

        # Younger seas are rougher at the same wind: halve the peak period
        # (c_p halves) at fixed U10N = 12 m/s and H_s.
        U = 12.0
        old, young = deep_water_sea_state(8.0, 2.0), deep_water_sea_state(4.0, 2.0)
        for r in (WaveAgeCharnock(), OostRoughness(), SeaStateRoughness())
            cd_old = neutral_transfer_coefficients(r, COARE35ScalarRoughness(), U; sea = old).C_D
            cd_young = neutral_transfer_coefficients(r, COARE35ScalarRoughness(), U; sea = young).C_D
            @test cd_young > 1.1 * cd_old
        end
        # Fully developed sea (Taylor & Yelland 2001 relations) under the
        # COARE 3.5 wave-age law lands at the Donelan (1990) fully
        # developed wave age c_p/u* ≈ 32 at moderate winds — two
        # independent definitions of "fully developed" agree.
        for U in (6.0, 8.0, 10.0, 12.0)
            f = fully_developed_sea_state(U)
            n = neutral_transfer_coefficients(wa, COARE35ScalarRoughness(), U; sea = f)
            @test 28 < wave_age(f, n.u_star) < 37
        end
        # A steep wave law with a large, slow sea still has a (lower,
        # stable) neutral root.
        big = fully_developed_sea_state(30.0)
        n = neutral_transfer_coefficients(OostRoughness(), COARE35ScalarRoughness(), 30.0; sea = big)
        @test n.u_star ≈ 0.4 * 30 / log(10 / n.z0m) rtol = 1e-10
    end

    @testset "Scalar roughness vs roughness Reynolds number" begin
        ν = 1.5e-5
        s30, s35, bg = COARE30ScalarRoughness(), COARE35ScalarRoughness(), BrutsaertGarrattRoughness()
        # Smooth flow: capped at the published maxima.
        @test sea_scalar_roughness(s30, 1e-5, 0.05; ν = ν).z0q == 1.1e-4
        @test sea_scalar_roughness(s35, 1e-5, 0.05; ν = ν).z0q == 1.6e-4
        # Rough flow: z0q = A R_r^(−B) exactly, and z0h = z0q.
        z0, u = 2e-3, 0.8
        Rr = roughness_reynolds(z0, u, ν)
        r = sea_scalar_roughness(s30, z0, u; ν = ν)
        @test r.z0q ≈ 5.5e-5 * Rr^-0.6 && r.z0h == r.z0q
        @test sea_scalar_roughness(s35, z0, u; ν = ν).z0q ≈ 5.8e-5 * Rr^-0.72
        # Scalar roughness falls as the surface roughens (surface renewal):
        # z0q decreases monotonically with R_r in the rough regime.
        zs = [sea_scalar_roughness(s35, zz, 0.5; ν = ν).z0q for zz in (1e-4, 3e-4, 1e-3, 3e-3)]
        @test issorted(zs, rev = true)
        # Brutsaert–Garratt: ln(z0/z0q) = 2.28 R_r^(1/4) − 2.
        rb = sea_scalar_roughness(bg, z0, u; ν = ν)
        @test log(z0 / rb.z0q) ≈ 2.28 * Rr^0.25 - 2
    end

end
