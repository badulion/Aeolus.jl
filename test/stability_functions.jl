# Era VI, Phase 58 (#170) — the TurbulenceStability contract:
# gradient-Richardson stability functions, N² and Ri_g.
#
#   1. limits of every implementer: f(0) = 1, f ≥ 0, non-increasing in
#      Ri > 0; the Lilly cutoff is exactly 0 above Ri_c and continuous there;
#   2. MOST round trip: K_m, K_h of MOSTConsistent reproduce the surface-
#      layer fluxes of the analytic MOST profiles;
#   3. critical Ri of the log-linear and BDBH similarity functions;
#
# The pointwise N²/Ri_g fields and the cloudy-air N² (#239) were cut from the
# fast suite (2026-11, suite reduction).

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
    ok = true
    for sf in impls, fac in (stability_factor_m, stability_factor_h)
        fs = [fac(sf, Ri) for Ri in Ris]
        pos = [fac(sf, Ri) for Ri in Ris if Ri > 0]
        ok &= abs(fac(sf, 0.0) - 1) <= eps() && all(isfinite, fs) && all(>=(0), fs) &&
              all(diff(pos) .<= 0) &&                        # non-increasing in stable air
              all(fac(sf, Ri) >= 1 for Ri in Ris if Ri < 0) && # unstable air mixes more
              neutral_prandtl(sf) > 0
    end
    @test ok

    # Lilly / Mason: exactly 0 at and above Ri_c, continuous at Ri_c (the
    # left limit √(δ/Ri_c), measured in BigFloat at δ = 1e-26)
    sf = LillyCutoff(Ri_c = 0.25)
    @test all(stability_factor_m(sf, Ri) == 0 && stability_factor_h(sf, Ri) == 0
              for Ri in 0.25 .* (1:0.5:10))
    δ = big"1e-26"
    f_left = stability_factor_m(LillyCutoff(Ri_c = BigFloat(0.25)), BigFloat(0.25) - δ)
    @test abs(f_left - sqrt(δ / BigFloat(0.25))) < 1e-30
    # the TKE-equilibrium form K = l²(S² − N²/Ri_c)^{1/2}
    let S2 = 4e-4, N2 = 3e-5
        @test sqrt(S2) * stability_factor_m(LillyCutoff(), N2 / S2) ≈ sqrt(S2 - N2 / 0.25) rtol = 1e-14
    end
    # Brown et al. (1994) / MONC constants, quoted values
    let sf = BrownEtAl1994()
        @test stability_factor_m(sf, 0.1) ≈ (1 - 0.4)^4 rtol = 1e-15
        @test stability_factor_h(sf, 0.1) ≈ 0.6^4 * (1 - 0.12) rtol = 1e-15
        @test stability_factor_m(sf, -0.5) ≈ 3.0 rtol = 1e-15     # √(1 + 8)
    end
    # Louis (1979), b = d = 5: f_m(Ri = 3) = 1/(1 + 30/4) = 2/17; long tail f_m → √5/(10 √Ri)
    @test stability_factor_m(LongTail(), 3.0) ≈ 2 / 17 rtol = 1e-15
    @test stability_factor_m(LongTail(), 1e8) * sqrt(1e8) ≈ sqrt(5) / 10 rtol = 1e-4
    # short tail: (1 + 5 Ri)⁻²
    @test stability_factor_m(ShortTail(), 1.0) ≈ 1 / 36

    @test gradient_richardson(1e-4, 4e-4) ≈ 0.25
    @test gradient_richardson(1e-4, 0.0) ≈ 1e-4 / RI_SHEAR2_FLOOR
    @test buoyancy_frequency_squared(300.0, 301.0, 100.0, 9.81, 300.0) ≈ 9.81 / 300 / 100
end

@testset "2. MOST round trip (MOSTConsistent)" begin
    κ, g, θ₀, u_star, z0, z = 0.4, 9.81, 290.0, 0.3, 0.1, 10.0
    worst_ζ = 0.0; worst_m = 0.0; worst_h = 0.0
    for sf in (BusingerDyerBeljaarsHoltslag(), LogLinearMOST(5.0))
        most = MOSTConsistent(sf)
        for ζ in range(-2, 10; length = 25)
            L = ζ == 0 ? Inf : z / ζ
            θ_star = ζ == 0 ? 0.0 : u_star^2 * θ₀ / (κ * g * L)
            u(zz) = u_star / κ * (log(zz / z0) - psi_m(sf, zz / L) + psi_m(sf, z0 / L))
            θ(zz) = θ₀ + θ_star / κ * (log(zz / z0) - psi_h(sf, zz / L) + psi_h(sf, z0 / L))
            S  = _d4(u, z, 1e-2)
            θz = _d4(θ, z, 1e-2)
            Ri = gradient_richardson(g / θ₀ * θz, S^2)
            worst_ζ = max(worst_ζ, abs(most_stability_parameter(most, Ri) - ζ))   # the inversion
            Km = (κ * z)^2 * S * stability_factor_m(most, Ri)
            Kh = (κ * z)^2 * S * stability_factor_h(most, Ri)
            worst_m = max(worst_m, abs(Km - u_star^2 / S) / (u_star^2 / S))
            worst_h = max(worst_h, ζ == 0 ? abs(Kh * θz) : abs(Kh * θz - u_star * θ_star) / abs(u_star * θ_star))
        end
    end
    @test worst_ζ < 1e-7
    @test worst_m < 1e-6
    @test worst_h < 1e-6
end

@testset "3. critical Richardson number" begin
    ll = LogLinearMOST(5.0)
    Ri_ll = most_gradient_richardson(ll, 1e4)
    @test abs(Ri_ll - 0.2) < 1e-3 && Ri_ll < 0.2          # approached from below
    @test stability_factor_m(MOSTConsistent(ll), 0.2) == 0
    @test stability_factor_m(MOSTConsistent(ll), 0.19) > 0
    # Beljaars–Holtslag has no critical value
    @test most_gradient_richardson(BusingerDyerBeljaarsHoltslag(), 100.0) > 1
    # unstable Businger–Dyer with γ_m = γ_h: Ri = ζ exactly
    @test most_gradient_richardson(BusingerDyerBeljaarsHoltslag(), -1.5) ≈ -1.5 rtol = 1e-14
end
end # testset
