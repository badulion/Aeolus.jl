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
import Aeolus.SurfaceExchange: phi_m, phi_h, psi_m, psi_h

# The log-linear similarity functions, φ = 1 + αζ in stable air (Webb 1970;
# Businger et al. 1971 with α ≈ 5), Businger–Dyer (γ = 16) in unstable air.
struct LogLinearMOST <: AbstractStabilityFunctions
    α::Float64
end
const _BDBH = BusingerDyerBeljaarsHoltslag()
phi_m(s::LogLinearMOST, ζ::Real) = ζ < 0 ? phi_m(_BDBH, ζ) : 1 + s.α * ζ
phi_h(s::LogLinearMOST, ζ::Real) = ζ < 0 ? phi_h(_BDBH, ζ) : 1 + s.α * ζ
psi_m(s::LogLinearMOST, ζ::Real) = ζ < 0 ? psi_m(_BDBH, ζ) : -s.α * ζ
psi_h(s::LogLinearMOST, ζ::Real) = ζ < 0 ? psi_h(_BDBH, ζ) : -s.α * ζ

# 4th-order centred first derivative
_d4(f, z, h) = (8 * (f(z + h) - f(z - h)) - (f(z + 2h) - f(z - 2h))) / (12h)

@testset "Era VI — TurbulenceStability (P58-a, #170)" begin

@testset "1. limits of every implementer" begin
    impls = [LillyCutoff(), BrownEtAl1994(), ShortTail(), LongTail(),
             MOSTConsistent(), MOSTConsistent(LogLinearMOST(5.0))]
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

@testset "2. MOST round trip (MOSTConsistent)" begin
    κ, g, θ₀, u_star, z0, z = 0.4, 9.81, 290.0, 0.3, 0.1, 10.0
    worst_m = 0.0; worst_h = 0.0
    for sf in (BusingerDyerBeljaarsHoltslag(), LogLinearMOST(5.0))
        most = MOSTConsistent(sf)
        @test stability_factor_m(most, 0.0) == 1 && stability_factor_h(most, 0.0) == 1
        for ζ in range(-2, 10; length = 49)
            L = ζ == 0 ? Inf : z / ζ
            θ_star = ζ == 0 ? 0.0 : u_star^2 * θ₀ / (κ * g * L)
            u(zz) = u_star / κ * (log(zz / z0) - psi_m(sf, zz / L) + psi_m(sf, z0 / L))
            θ(zz) = θ₀ + θ_star / κ * (log(zz / z0) - psi_h(sf, zz / L) + psi_h(sf, z0 / L))
            S  = _d4(u, z, 1e-2)
            θz = _d4(θ, z, 1e-2)
            Ri = gradient_richardson(g / θ₀ * θz, S^2)
            # the inversion recovers ζ
            @test most_stability_parameter(most, Ri) ≈ ζ atol = 1e-7
            Km = (κ * z)^2 * S * stability_factor_m(most, Ri)
            Kh = (κ * z)^2 * S * stability_factor_h(most, Ri)
            em = abs(Km - u_star^2 / S) / (u_star^2 / S)
            eh = ζ == 0 ? abs(Kh * θz) : abs(Kh * θz - u_star * θ_star) / abs(u_star * θ_star)
            worst_m = max(worst_m, em); worst_h = max(worst_h, eh)
            @test em < 1e-6
            @test eh < 1e-6
        end
    end
    @info "P58-a MOST round trip, ζ ∈ [−2, 10], BDBH and log-linear α = 5" max_rel_err_Km = worst_m max_rel_err_KhΘz = worst_h
end

@testset "3. critical Richardson number" begin
    ll = LogLinearMOST(5.0)
    Ri_ll = most_gradient_richardson(ll, 1e4)
    @test abs(Ri_ll - 0.2) < 1e-3
    @test Ri_ll < 0.2                                   # approached from below
    # beyond the critical value the MOST-consistent factors vanish
    @test stability_factor_m(MOSTConsistent(ll), 0.2) == 0
    @test stability_factor_h(MOSTConsistent(ll), 0.25) == 0
    @test stability_factor_m(MOSTConsistent(ll), 0.19) > 0
    # Beljaars–Holtslag has no critical value
    Ri_bh = most_gradient_richardson(BusingerDyerBeljaarsHoltslag(), 100.0)
    @test Ri_bh > 1
    @test stability_factor_m(MOSTConsistent(), 5.0) > 0
    # unstable Businger–Dyer with γ_m = γ_h: Ri = ζ exactly
    @test most_gradient_richardson(BusingerDyerBeljaarsHoltslag(), -1.5) ≈ -1.5 rtol = 1e-14
    @info "P58-a critical Ri" log_linear_Ri_at_ζ_1e4 = Ri_ll BDBH_Ri_at_ζ_100 = Ri_bh
end

end # testset
