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
end
