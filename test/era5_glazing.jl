# Phase 49 — UrbanSurface glazing: angle-dependent window transmission (#27).
#
# Validation is against things the code does not contain:
#   * the Fresnel equations' analytic limits (normal incidence, Brewster,
#     grazing) and the closed-form transmittance of a stack of
#     non-absorbing plates (Duffie & Beckman 2013, Eq. 5.1.6);
#   * published normal-incidence solar properties of clear float glass:
#     Vitro Architectural Glass, "Clear Technical Product Data" (April
#     2020), Solar Performance Values table, ISO 9050:2003; and the
#     "Clear Glass" row of the LLumar architectural window-film
#     performance data sheet (NFRC methodology, 3 mm single clear pane);
#   * the clear-glass angular regression of LBNL WINDOW (WINDOW Technical
#     Documentation, April 2018, Eqs. 7-30/7-31 and Table 7-1), which
#     WINDOW and EnergyPlus use as their reference angular behaviour;
#   * NFRC SHGC and U-values of a clear/clear 6-13-6 insulating unit
#     (Vitro, "Performance Data for One-Inch Insulating Glass Units",
#     LBNL WINDOW 7.3, NFRC methodology);
#   * exact energy bookkeeping of the window facet, and its steady state
#     against the steady SHGC / U-value of the same glazing.

using Test, Printf, LinearAlgebra
using Aeolus
using Aeolus.UrbanSurface
using Aeolus.SurfaceExchange
import Aeolus.SurfaceOptics

const USG = Aeolus.UrbanSurface

# Test-side reference: Fresnel per polarisation, written out independently.
function ref_fresnel(θ, n)
    c = cos(θ); ct = sqrt(1 - (sin(θ) / n)^2)
    return (((c - n * ct) / (c + n * ct))^2, ((n * c - ct) / (n * c + ct))^2)
end

# WINDOW clear-glass angular regression (Technical Documentation 2018,
# Table 7-1): τ̄(Φ) = Σ τ_i cos^i Φ, ρ̄(Φ) = Σ ρ_i cos^i Φ - τ̄(Φ), applied as
# T(Φ) = T(0) τ̄(Φ), R(Φ) = R(0) (1 - ρ̄(Φ)) + ρ̄(Φ).
const TAU_CLR = (-0.0015, 3.355, -3.840, 1.460, 0.0288)
const RHO_CLR = (0.999, -0.563, 2.043, -2.532, 1.054)
poly4(a, x) = sum(a[i+1] * x^i for i in 0:4)
window_tau_clr(Φ) = poly4(TAU_CLR, cos(Φ))
window_rho_clr(Φ) = poly4(RHO_CLR, cos(Φ)) - window_tau_clr(Φ)

@testset "Era V — UrbanSurface glazing: window transmission (#27)" begin

    @testset "Fresnel (SurfaceOptics): polarised components and limits" begin
        n = 1.526
        R0 = ((n - 1) / (n + 1))^2
        Rs, Rp, ct = SurfaceOptics.fresnel_polarized_reflectance(0.0, n)
        @test Rs ≈ R0 atol = 1e-14
        @test Rp ≈ R0 atol = 1e-14
        @test ct ≈ 1.0
        @test R0 ≈ 0.04336 atol = 1e-5
        θB = atan(n)                                   # Brewster
        Rs, Rp, _ = SurfaceOptics.fresnel_polarized_reflectance(θB, n)
        @test Rp < 1e-15
        @test Rs > 0.1
        for θd in (10, 30, 45, 60, 75, 85)
            θ = deg2rad(θd)
            rs, rp = ref_fresnel(θ, n)
            Rs, Rp, ct = SurfaceOptics.fresnel_polarized_reflectance(θ, n)
            @test Rs ≈ rs atol = 1e-13
            @test Rp ≈ rp atol = 1e-13
            @test ct ≈ sqrt(1 - (sin(θ) / n)^2) atol = 1e-14   # Snell
            @test fresnel_reflectance(θ, n) ≈ (rs + rp) / 2 atol = 1e-13
        end
        Rs, Rp, _ = SurfaceOptics.fresnel_polarized_reflectance(π / 2, n)
        @test Rs ≈ 1 atol = 1e-9
        @test Rp ≈ 1 atol = 1e-9
        @test_throws ArgumentError SurfaceOptics.fresnel_polarized_reflectance(0.3, 1.0)
    end

    @testset "Single pane: slab optics, conservation, analytic limits" begin
        # Non-absorbing slab: T = (1-ρ)/(1+ρ) per polarisation (D&B Eq. 5.1.4)
        p0 = GlassPane(0.004; K = 0.0)
        for θd in (0, 20, 45, 60, 80)
            θ = deg2rad(θd)
            rs, rp = ref_fresnel(θ, p0.n)
            Tex = ((1 - rs) / (1 + rs) + (1 - rp) / (1 + rp)) / 2
            o = pane_optics(p0, θ)
            @test o.T ≈ Tex atol = 1e-14
            @test o.A ≈ 0 atol = 1e-14
            @test o.R + o.T ≈ 1 atol = 1e-14
        end
        n = p0.n
        @test pane_optics(p0, 0.0).T ≈ 2n / (n^2 + 1) atol = 1e-14

        # Absorbing pane: conservation at every angle, grazing limits
        p = GlassPane(0.006)
        for θd in 0:5:89
            o = pane_optics(p, deg2rad(θd))
            @test o.R + o.T + o.A ≈ 1 atol = 1e-14
            @test all(x -> 0 <= x <= 1, (o.R, o.T, o.A))
        end
        og = pane_optics(p, π / 2)
        @test og.T ≈ 0 atol = 1e-9
        @test og.R ≈ 1 atol = 1e-9
        # Opaque limit: only the first interface reflects
        pthick = GlassPane(0.5; K = 1e3)
        o = pane_optics(pthick, deg2rad(30))
        rs, rp = ref_fresnel(deg2rad(30), pthick.n)
        @test o.T < 1e-100
        @test o.R ≈ (rs + rp) / 2 atol = 1e-12
        # Beer–Lambert: at normal incidence, with multiple reflections removed
        # (T(1-ρ²τ²)/(1-ρ)² = τ), the internal transmittance is exp(-K d).
        ρ = ((n - 1) / (n + 1))^2
        for d in (0.003, 0.006, 0.012)
            q = GlassPane(d)
            o = pane_optics(q, 0.0)
            τ = exp(-q.K * d)
            @test o.T ≈ (1 - ρ)^2 * τ / (1 - ρ^2 * τ^2) atol = 1e-14
        end
        # Thicker glass absorbs more and transmits less
        Ts = [pane_optics(GlassPane(d), 0.0).T for d in (0.003, 0.006, 0.010)]
        @test issorted(Ts; rev = true)
        @test_throws ArgumentError GlassPane(-0.004)
        @test_throws ArgumentError GlassPane(0.004; n = 0.9)
    end

    @testset "Data-sheet inversion (WINDOW Eqs. 7-26…7-29): exact round trip" begin
        # A pane built from (n, K) is recovered from its own normal (T, R)
        for (d, n, K) in ((0.003, 1.526, 32.0), (0.006, 1.50, 10.0), (0.010, 1.55, 60.0))
            q = GlassPane(d; n = n, K = K)
            o = pane_optics(q, 0.0)
            r = glass_pane_from_normal(d, o.T, o.R)
            @test r.n ≈ n rtol = 1e-10
            @test r.K ≈ K rtol = 1e-8
        end
        # and a data-sheet pair is reproduced exactly
        r = glass_pane_from_normal(0.006, 0.77, 0.08)
        o = pane_optics(r, 0.0)
        @test o.T ≈ 0.77 atol = 1e-12
        @test o.R ≈ 0.08 atol = 1e-12
        # Inverting the Vitro 3 mm row (T 0.83, R 0.08) lands on the book
        # values the default pane uses: Duffie & Beckman K ≈ 32 m⁻¹, n ≈ 1.526.
        v3 = glass_pane_from_normal(0.003, 0.83, 0.08)
        @info @sprintf("Vitro 3 mm inverted: n = %.4f (D&B 1.526), K = %.2f m⁻¹ (D&B 32)", v3.n, v3.K)
        @test abs(v3.K - CLEAR_FLOAT_EXTINCTION) / CLEAR_FLOAT_EXTINCTION < 0.05
        @test abs(v3.n - CLEAR_FLOAT_REFRACTIVE_INDEX) < 0.03
        @test_throws ArgumentError glass_pane_from_normal(0.004, 0.9, 0.2)
    end

    @testset "Normal incidence vs published clear float (Vitro, ISO 9050)" begin
        # Vitro Clear Technical Product Data (2020), Solar Performance Values:
        # thickness (mm), total solar transmittance, total solar reflectance.
        vitro = [(3.0, 0.83, 0.08), (4.0, 0.81, 0.07), (5.0, 0.79, 0.07),
                 (6.0, 0.77, 0.08), (8.0, 0.72, 0.07), (10.0, 0.69, 0.07),
                 (12.0, 0.64, 0.06), (16.0, 0.59, 0.06), (19.0, 0.55, 0.06)]
        rows = String[]
        for (dmm, Tv, Rv) in vitro
            o = pane_optics(GlassPane(dmm / 1000), 0.0)
            push!(rows, @sprintf("%4.0f mm: T %.3f (%.2f)  R %.3f (%.2f)", dmm, o.T, Tv, o.R, Rv))
            # Published values are rounded to 0.01; the reflectance is
            # matched to that rounding at every thickness.
            @test abs(o.R - Rv) <= 0.011
            if dmm <= 8
                # building-glazing thicknesses: within 0.015
                @test abs(o.T - Tv) <= 0.015
            else
                # Thick glass: the grey (single-K) model under-predicts.
                # Real float glass absorbs mainly in the Fe²⁺ band near 1 μm;
                # once that band is saturated extra thickness removes little,
                # which a broadband Beer–Lambert law cannot represent. The
                # error is one-signed and bounded, not noise.
                @test -0.06 <= o.T - Tv < 0
            end
        end
        @info "Default clear float pane vs Vitro (published in brackets)\n" * join(rows, "\n")

        # LLumar / NFRC single 3 mm clear: T 0.83, R 0.08, A 0.09
        o = pane_optics(GlassPane(0.003), 0.0)
        @test abs(o.T - 0.83) <= 0.01
        @test abs(o.R - 0.08) <= 0.01
        @test abs(o.A - 0.09) <= 0.01
    end

    @testset "Angular dependence vs the WINDOW clear-glass regression" begin
        # WINDOW Table 7-1 regression for clear glass, normalised to the
        # normal-incidence values, compared with the first-principles
        # Fresnel + Beer–Lambert result. The regression is ONE curve for
        # all thicknesses; the physics is not — the refracted path grows
        # as 1/cos θ_t, so thicker glass loses relatively more at oblique
        # incidence and has a weaker back-surface reflection. The 3 mm
        # pane (the NFRC reference clear glass) is held to 0.01 / 0.015;
        # for 6 mm the gap opens towards grazing incidence, in the
        # direction the path-length argument predicts (model below the
        # regression), and is bounded rather than matched.
        rows = String[]
        for d in (0.003, 0.006)
            p = GlassPane(d)
            o0 = pane_optics(p, 0.0)
            for θd in 0:10:80
                Φ = deg2rad(θd)
                o = pane_optics(p, Φ)
                Tw = o0.T * window_tau_clr(Φ)
                Rw = o0.R * (1 - window_rho_clr(Φ)) + window_rho_clr(Φ)
                push!(rows, @sprintf("%2.0f mm %2d°: T %.3f (WINDOW %.3f)  R %.3f (WINDOW %.3f)",
                                     1000d, θd, o.T, Tw, o.R, Rw))
                if d == 0.003
                    @test abs(o.T - Tw) <= 0.01
                    @test abs(o.R - Rw) <= 0.015
                else
                    @test abs(o.T - Tw) <= (θd <= 60 ? 0.02 : 0.03)
                    @test abs(o.R - Rw) <= (θd <= 60 ? 0.01 : 0.04)
                    θd >= 50 && @test o.T < Tw && o.R < Rw
                end
            end
        end
        @info "Pane angular optics vs WINDOW clear-glass regression (published regression in brackets)\n" * join(rows, "\n")
        # Transmittance falls monotonically with angle. The reflectance is
        # flat (to 2e-5) up to ~15° — the unpolarised Fresnel reflectance is
        # stationary at normal incidence and the longer internal path trims
        # the back-surface reflection — and rises monotonically beyond.
        p = GlassPane(0.004)
        Ts = [pane_optics(p, deg2rad(θd)).T for θd in 0:5:90]
        Rs = [pane_optics(p, deg2rad(θd)).R for θd in 15:5:90]
        @test issorted(Ts; rev = true)
        @test issorted(Rs)
        @test all(θd -> abs(pane_optics(p, deg2rad(θd)).R - pane_optics(p, 0.0).R) < 5e-5, 0:5:15)
    end

    @testset "Multi-pane stacks: closed form, conservation, hemispherical" begin
        # N non-absorbing plates = 2N interfaces: per polarisation
        # T = (1-ρ)/(1 + (2N-1)ρ) (Duffie & Beckman Eq. 5.1.6)
        for N in 1:3, θd in (0, 45, 70)
            θ = deg2rad(θd)
            g = Glazing([GlassPane(0.004; K = 0.0) for _ in 1:N],
                        [GlazingGap(0.012) for _ in 1:N-1])
            rs, rp = ref_fresnel(θ, CLEAR_FLOAT_REFRACTIVE_INDEX)
            Tex = ((1 - rs) / (1 + (2N - 1) * rs) + (1 - rp) / (1 + (2N - 1) * rp)) / 2
            o = glazing_optics(g, θ)
            @test o.T ≈ Tex atol = 1e-13
            @test o.R ≈ 1 - Tex atol = 1e-13
        end
        # Absorbing: R + T + ΣA = 1; the outer pane absorbs more
        for g in (single_glazing(), double_glazing(), triple_glazing()), θd in (0, 40, 75)
            o = glazing_optics(g, deg2rad(θd))
            @test o.R + o.T + sum(o.A) ≈ 1 atol = 1e-13
            length(g) > 1 && @test issorted(o.A; rev = true)
        end
        # Two panes: the classic two-layer formula, per polarisation
        #   T12 = T1 T2 / (1 - R1 R2),   R12 = R1 + T1² R2 / (1 - R1 R2)
        g = double_glazing(thickness = 0.006)
        θ = deg2rad(35)
        s, pp = USG._pane_sp(g.panes[1], θ)
        T12 = sum(x -> x[2]^2 / (1 - x[1]^2), (s, pp)) / 2
        R12 = sum(x -> x[1] + x[2]^2 * x[1] / (1 - x[1]^2), (s, pp)) / 2
        o = glazing_optics(g, θ)
        @test o.T ≈ T12 atol = 1e-13
        @test o.R ≈ R12 atol = 1e-13
        # Grazing incidence (the sun on the horizon of the facet plane):
        # everything is reflected. A sunrise step at cos θ ~ 1e-17 once made
        # the inter-pane linear system singular; the limit is now exact.
        for g in (double_glazing(), triple_glazing())
            o = glazing_optics(g, π / 2)
            @test o.R == 1.0 && o.T == 0.0 && all(iszero, o.A)
            o = glazing_optics(g, acos(1e-17))
            @test o.R ≈ 1 atol = 1e-9
            @test o.R + o.T + sum(o.A) ≈ 1 atol = 1e-12
        end
        # Single-pane glazing is the pane
        @test glazing_optics(single_glazing(0.005), 0.4).T ≈ pane_optics(GlassPane(0.005), 0.4).T

        # Hemispherical averages: converged, conservative, below normal
        for g in (single_glazing(0.003), double_glazing())
            h = glazing_optics_hemispherical(g)
            h2 = glazing_optics_hemispherical(g; nquad = 96)
            @test h.T ≈ h2.T atol = 1e-6
            @test h.R ≈ h2.R atol = 1e-6
            @test h.R + h.T + sum(h.A) ≈ 1 atol = 1e-12
            o0 = glazing_optics(g, 0.0)
            @test h.T < o0.T
            @test h.R > o0.R
        end
        # Non-absorbing single interface pair: the hemispherical average
        # obeys R + T = 1 exactly as well
        h = glazing_optics_hemispherical(Glazing(GlassPane(0.004; K = 0.0)))
        @test h.R + h.T ≈ 1 atol = 1e-13
        h3 = glazing_optics_hemispherical(single_glazing(0.003))
        @info @sprintf("3 mm clear, hemispherical: T %.3f R %.3f A %.3f", h3.T, h3.R, sum(h3.A))
    end
    @testset "Gas gaps (ISO 15099): conduction, convection, radiation limits" begin
        c = DEFAULT_SURFACE_CONSTANTS
        # One source of truth: the ISO 15099 air fit reproduces the contract's
        # still-air conductivity at 0 °C.
        @test USG.gas_conductivity(GLAZING_AIR, 273.15) ≈ c.k_air rtol = 5e-3
        @test USG.gas_conductivity(GLAZING_ARGON, 283.15) < USG.gas_conductivity(GLAZING_AIR, 283.15)
        # Nusselt: pure conduction at Ra → 0, near-continuous across branches
        @test gap_nusselt(0.0, 100.0) == 1.0
        @test gap_nusselt(100.0, 100.0) ≈ 1 atol = 0.02
        for Ra in (1e4, 5e4)
            @test gap_nusselt(Ra * (1 - 1e-9), 1.0) ≈ gap_nusselt(Ra * (1 + 1e-9), 1.0) rtol = 0.02
        end
        @test gap_nusselt(1e6, 50.0) > gap_nusselt(1e5, 50.0) > gap_nusselt(1e3, 50.0)
        # Narrow gap, tiny ΔT: h_c = k/d (conduction); h_r → 4σT³/(2/ε-1)
        gap = GlazingGap(0.006)
        T = 283.15; ε = UNCOATED_GLASS_EMISSIVITY
        h = gap_conductance(gap, T + 0.005, T - 0.005, ε, ε)
        @test h.h_c ≈ USG.gas_conductivity(GLAZING_AIR, T) / 0.006 rtol = 1e-3
        @test h.h_r ≈ 4c.σ_SB * T^3 / (2 / ε - 1) rtol = 1e-6
        # the radiative coefficient is exact, not linearised
        T1, T2, ε1, ε2 = 300.0, 270.0, 0.84, 0.10
        h = gap_conductance(gap, T1, T2, ε1, ε2)
        @test h.h_r * (T1 - T2) ≈ c.σ_SB * (T1^4 - T2^4) / (1 / ε1 + 1 / ε2 - 1) rtol = 1e-12
        # A wide gap with a large ΔT convects (Nu > 1)
        hw = gap_conductance(GlazingGap(0.03), 300.0, 270.0, ε, ε)
        @test hw.h_c > USG.gas_conductivity(GLAZING_AIR, 285.0) / 0.03 * 1.2
    end

    @testset "U-value and SHGC: analytic single pane, published insulating units" begin
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        # Single pane: U = 1/(1/h_o + d/k + 1/h_i) exactly (series resistances)
        p = GlassPane(0.004)
        k = p.material.conductivity
        for (ho, hi) in ((25.0, 7.7), (15.0, 3.0))
            @test glazing_u_value(single_glazing(0.004); h_out = ho, h_in = hi) ≈
                  1 / (1 / ho + 0.004 / k + 1 / hi) rtol = 1e-12
        end
        # Inward-flowing fraction of one pane: the conductance divider
        ho, hi = 23.0, 8.0                               # EN 410 h_e, h_i
        Go = 1 / (1 / ho + 0.002 / k); Gi = 1 / (1 / hi + 0.002 / k)
        N1 = inward_flowing_fractions(single_glazing(0.004); h_out = ho, h_in = hi)
        @test inward_flowing_fractions(single_glazing(0.004)) == N1     # EN 410 defaults
        @test N1[1] ≈ Gi / (Gi + Go) rtol = 1e-12
        # EN 410's single-pane formula h_i/(h_i + h_e) neglects the pane
        # resistance; it differs by < 0.005
        @test abs(N1[1] - hi / (hi + ho)) < 0.005
        # Ordering: argon < air, triple < double < single; outer pane's
        # absorbed heat mostly goes out, the inner pane's mostly in
        Ua = glazing_u_value(double_glazing())
        Ur = glazing_u_value(double_glazing(gas = GLAZING_ARGON))
        @test Ur < Ua < glazing_u_value(single_glazing())
        @test glazing_u_value(triple_glazing()) < Ua
        @test issorted([glazing_u_value(double_glazing(gap = d)) for d in (0.006, 0.009, 0.012)]; rev = true)
        Nd = inward_flowing_fractions(double_glazing())
        @test Nd[1] < 0.5 < Nd[2] < 1
        @test issorted(inward_flowing_fractions(triple_glazing()))

        # --- Published centre-of-glass U, clear/clear 6-13-6 (Vitro IGU data,
        # NFRC 100 winter: T_out = -18 °C, T_in = 21 °C, V = 5.5 m/s).
        # Exterior: h_c = 4 + 4V (ISO 15099 Eq. 131); interior: laminar
        # natural convection on a 1 m vertical plate, Nu = 0.56 Ra^(1/4)
        # (ISO 15099 §8.3.2). Radiation on both sides to surroundings at
        # the air temperature, h_r = εσ(T_s² + T²)(T_s + T) — the exact
        # grey-surface coefficient. Everything is evaluated test-side and
        # iterated to the surface temperatures the U-value implies.
        # 1 Btu/(h ft² °F) = 5.678263 W/(m² K).
        ε = UNCOATED_GLASS_EMISSIVITY
        hrad(Ts, T) = ε * σ * (Ts^2 + T^2) * (Ts + T)
        function nfrc_winter_u(g)
            T_out, T_in = 255.15, 294.15
            U = 2.7; hc_in = 2.5; Tsi = T_in - 5.0; Tso = T_out + 1.0
            for _ in 1:30
                ho = 4 + 4 * 5.5 + hrad(Tso, T_out)
                hi = hc_in + hrad(Tsi, T_in)
                U = glazing_u_value(g; T_out = T_out, T_in = T_in, h_out = ho, h_in = hi)
                Tsi = T_in - U * (T_in - T_out) / hi
                Tso = T_out + U * (T_in - T_out) / ho
                Tf = (Tsi + T_in) / 2
                kk = USG.gas_conductivity(GLAZING_AIR, Tf); μ = USG.gas_viscosity(GLAZING_AIR, Tf)
                cp = USG.gas_heat_capacity(GLAZING_AIR, Tf)
                ρ = 101325 * 0.02897 / (8.314462618 * Tf)
                Ra = ρ^2 * 9.80665 * cp * (T_in - Tsi) / (Tf * μ * kk)          # H = 1 m
                hc_in = 0.56 * Ra^0.25 * kk
            end
            return U, hc_in
        end
        U_air, hci = nfrc_winter_u(double_glazing(thickness = 0.006, gap = 0.0127))
        U_ar, _ = nfrc_winter_u(double_glazing(thickness = 0.006, gap = 0.0127, gas = GLAZING_ARGON))
        # Vitro's "winter argon" unit is 90 % argon / 10 % air; the model gap
        # is pure argon (no ISO 15099 mixing rule). A 90/10 blend of the
        # property fits raises the model U by ~0.017 W m⁻² K⁻¹ (~0.6 %),
        # well inside the 3 % tolerance, so the comparison stands.
        U_pub_air, U_pub_ar = 0.47 * 5.678263, 0.45 * 5.678263
        @info @sprintf("6-13-6 clear, NFRC winter U: air %.3f (Vitro %.3f), argon %.3f (Vitro %.3f) W m⁻² K⁻¹; interior h_c %.2f",
                       U_air, U_pub_air, U_ar, U_pub_ar, hci)
        @test abs(U_air - U_pub_air) / U_pub_air < 0.03
        @test abs(U_ar - U_pub_ar) / U_pub_ar < 0.03

        # --- Published SHGC. NFRC 100 summer: T_out = 32 °C, T_in = 24 °C,
        # V = 2.75 m/s (h_c = 4 + 4V), interior h_c = 2.5 + radiation.
        nfrc_summer(g, θ) = glazing_shgc(g, θ; T_out = 305.15, T_in = 297.15,
                                         h_out = 4 + 4 * 2.75 + 4ε * σ * 305.15^3,
                                         h_in = 2.5 + 4ε * σ * 297.15^3)
        s3 = nfrc_summer(single_glazing(0.003), 0.0)
        @test abs(s3 - 0.86) <= 0.01                      # LLumar/NFRC single 3 mm clear
        g6 = double_glazing(thickness = 0.006, gap = 0.0127)
        s6 = nfrc_summer(g6, 0.0)
        v6 = glass_pane_from_normal(0.006, 0.77, 0.08)      # Vitro 6 mm row
        g6v = Glazing([v6, v6], [GlazingGap(0.0127)])
        s6v = nfrc_summer(g6v, 0.0)
        @info @sprintf("SHGC: single 3 mm %.3f (NFRC 0.86); 6-13-6 default pane %.3f, Vitro-inverted pane %.3f (Vitro NFRC 0.70)",
                       s3, s6, s6v)
        @test abs(s6 - 0.70) <= 0.02
        @test abs(s6v - 0.70) <= 0.01
        # EN 410 conventions give nearly the same g for these units
        @test abs(glazing_shgc(g6, 0.0) - s6) < 0.01
        # Angular: SHGC falls with incidence angle; diffuse below normal
        shgcs = [glazing_shgc(g6, deg2rad(θd)) for θd in 0:10:80]
        @test issorted(shgcs; rev = true)
        @test glazing_shgc(g6, :hemispherical) < shgcs[1]
        # SHGC = T + Σ N A, and lies between T and T + ΣA
        o = glazing_optics(g6, 0.0)
        @test o.T < shgcs[1] < o.T + sum(o.A)
    end
    @testset "WindowFacet: shortwave partition, energy bookkeeping, interface" begin
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        g = double_glazing()
        w = WindowFacet(g; T = 293.15, T_interior = 295.15)
        @test w isa AbstractSurface
        @test surface_temperature(w) == 293.15
        @test surface_emissivity(w) == UNCOATED_GLASS_EMISSIVITY
        @test roughness_lengths(w).z0m == 0.01
        # Default geometry is diffuse: the albedo is the hemispherical reflectance
        h = glazing_optics_hemispherical(g)
        @test surface_albedo(w) ≈ h.R
        # Direct beam: the facet sees the angular optics
        set_solar_incidence!(w, cosd(30); diffuse_fraction = 0.0)
        @test surface_albedo(w) ≈ glazing_optics(g, deg2rad(30)).R
        set_solar_incidence!(w, cosd(80); diffuse_fraction = 0.0)
        @test surface_albedo(w) > glazing_optics(g, deg2rad(30)).R
        # Sun behind the facet: no direct beam
        set_solar_incidence!(w, -0.3; diffuse_fraction = 0.2)
        @test w.diffuse_fraction == 1.0
        @test_throws ArgumentError set_solar_incidence!(w, 0.5; diffuse_fraction = 1.5)
        @test_throws ArgumentError WindowFacet(g; interior = fixed_temperature_boundary(290.0))

        # A sunny day with Monin–Obukhov exchange: every step conserves energy
        # exactly and partitions the shortwave exactly.
        w = WindowFacet(g; T = 290.15, T_interior = 295.15)
        dt = 120.0
        maxres = 0.0; maxsw = 0.0; maxbal = 0.0
        for step in 1:720                       # 24 h
            t = step * dt
            μ = max(0.0, sin(2π * (t / 86400 - 0.25)))
            set_solar_incidence!(w, μ; diffuse_fraction = 0.3)
            f = SurfaceForcing(sw_down = 700μ, lw_down = 320.0,
                               T_air = 290.0 + 5μ, U = 3.0)
            opt = window_optics(w)
            E0 = window_heat_content(w)
            fl = step_surface!(w, f, dt)
            dE = window_heat_content(w) - E0
            maxbal = max(maxbal, abs(fl.Rn - fl.H - fl.LE - fl.G))
            maxres = max(maxres, abs(dE - dt * (fl.G - interior_heat_flux(w) - transmitted_shortwave(w))))
            maxsw = max(maxsw, abs(transmitted_shortwave(w) - opt.T * f.sw_down) +
                               abs(w.sw_absorbed - sum(opt.A) * f.sw_down))
        end
        @info @sprintf("WindowFacet diurnal: max |Rn-H-LE-G| %.2e, max storage residual %.2e J m⁻², max SW partition error %.2e",
                       maxbal, maxres, maxsw)
        @test maxbal < 1e-9
        @test maxres < 1e-13 * window_heat_content(w)     # round-off of a ~4e6 J m⁻² quantity
        @test maxsw < 1e-10
        @test w.T_panes[1] > 250 && w.T_panes[1] < 330

        # Interior hook
        set_interior_temperature!(w, 300.0)
        @test w.interior.T_env == 300.0
        set_interior_boundary!(w, convective_boundary(296.0, 3.0))
        @test w.interior.h == 3.0
        @test_throws ArgumentError set_interior_boundary!(w, fixed_temperature_boundary(296.0))
    end

    @testset "WindowFacet steady states reproduce the steady SHGC and U-value" begin
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        ε = UNCOATED_GLASS_EMISSIVITY
        hc = 15.0
        hi = 1 / ISO6946_SURFACE_RESISTANCE.inside_horizontal
        rows = String[]
        for g in (single_glazing(0.004), double_glazing(), triple_glazing())
            # Sun at 40° incidence, outdoor air = room air = 298.15 K, sky at
            # the air temperature: the steady room gain per unit irradiance is
            # the SHGC by definition.
            T0 = 298.15; θi = deg2rad(40); I = 600.0
            w = WindowFacet(g; T = T0, T_interior = T0, h_convective = hc)
            set_solar_incidence!(w, cos(θi); diffuse_fraction = 0.0)
            f = SurfaceForcing(sw_down = I, lw_down = σ * T0^4, T_air = T0)
            for _ in 1:2000; step_surface!(w, f, 60.0); end
            gain = (interior_heat_flux(w) + transmitted_shortwave(w)) / I
            shgc = glazing_shgc(g, θi; h_out = hc + 4ε * σ * T0^3, h_in = hi,
                                T_out = T0, T_in = T0)
            # Night: no sun, cold outside, warm room — steady loss is U ΔT
            To, Ti = 268.15, 293.15
            w2 = WindowFacet(g; T = 280.0, T_interior = Ti, h_convective = hc)
            f2 = SurfaceForcing(sw_down = 0.0, lw_down = σ * To^4, T_air = To)
            for _ in 1:2000; step_surface!(w2, f2, 60.0); end
            U_dyn = -interior_heat_flux(w2) / (Ti - To)
            Ts = w2.T_skin
            U_st = glazing_u_value(g; T_out = To, T_in = Ti,
                                   h_out = hc + ε * σ * (Ts^2 + To^2) * (Ts + To), h_in = hi)
            push!(rows, @sprintf("%d pane(s): SHGC facet %.4f steady %.4f | U facet %.4f steady %.4f",
                                 length(g), gain, shgc, U_dyn, U_st))
            # Differences come only from nonlinearity the steady formulas
            # linearise: pane-to-air radiation and the gap coefficients.
            @test abs(gain - shgc) < 0.005
            @test abs(U_dyn - U_st) / U_st < 0.01
        end
        @info "WindowFacet steady state vs steady formulas\n" * join(rows, "\n")
    end
    @testset "examples/urban/glass_material.jl delegates to UrbanSurface" begin
        # The flagship example's glass class now carries no numbers of its
        # own; load it (its includes re-define docs, hence the null logger)
        ex = Module(:GlassExample)
        Core.eval(ex, :(include(p) = Base.include($ex, p)))
        Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            Base.include(ex, joinpath(@__DIR__, "..", "examples", "urban", "glass_material.jl"))
        end
        g = Base.invokelatest(getfield(ex, :example_glazing))
        params = Base.invokelatest(getfield(ex, :glass_surface_params))
        h = glazing_optics_hemispherical(g)
        cls = Int(getfield(ex, :SURFACE_GLASS))
        @test params.albedo[cls] ≈ h.R
        @test params.emissivity[cls] == g.panes[1].emissivity
        @test params.C_s[cls] ≈ sum(p -> USG.volumetric_heat_capacity(p.material) * p.thickness, g.panes)
        # One step of the example's surface scheme on a glass cell, with the
        # long-wave balance neutral: the cell absorbs exactly ΣA·SW — the
        # transmitted shortwave has left its budget. (Before #27 the add-on
        # subtracted the transmitted part from the loss and so heated the
        # glass with it.)
        sfc = fill(UInt8(cls), 1, 1)
        T0 = 300.0; sw = 600.0; dt = 10.0
        Ts = fill(T0, 1, 1)
        loss = zeros(1, 1)
        Base.invokelatest(getfield(ex, :apply_glass_sw_transmission!), loss, sfc, fill(sw, 1, 1))
        @test loss[1, 1] ≈ h.T * sw
        lw = getfield(ex, :STEFAN) * T0^4
        Base.invokelatest(getfield(ex, :step_surfaces!), Ts, sfc, params, fill(sw, 1, 1), lw, loss, dt)
        @test (Ts[1, 1] - T0) * params.C_s[cls] / dt ≈ sum(h.A) * sw rtol = 1e-10
        # direct-beam variant uses the angular transmittance
        loss .= 0
        Base.invokelatest(getfield(ex, :apply_glass_sw_transmission!), loss, sfc, fill(sw, 1, 1);
                          cos_incidence = cosd(60))
        @test loss[1, 1] ≈ glazing_optics(g, deg2rad(60)).T * sw
    end
end
