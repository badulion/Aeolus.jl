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

using Test, LinearAlgebra
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

@testset "UrbanSurface glazing: window transmission (#27)" begin

    @testset "Fresnel (SurfaceOptics) and single-pane optics" begin
        n = 1.526
        Rs, Rp, _ = SurfaceOptics.fresnel_polarized_reflectance(0.0, n)
        @test Rs ≈ ((n - 1) / (n + 1))^2 atol = 1e-14
        @test SurfaceOptics.fresnel_polarized_reflectance(atan(n), n)[2] < 1e-15     # Brewster
        worst = maximum(θd -> maximum(abs.(SurfaceOptics.fresnel_polarized_reflectance(deg2rad(θd), n)[1:2] .-
                                           ref_fresnel(deg2rad(θd), n))), (10, 45, 85))
        @test worst < 1e-13
        # non-absorbing slab: T = (1−ρ)/(1+ρ) per polarisation (D&B Eq. 5.1.4)
        p0 = GlassPane(0.004; K = 0.0)
        rs, rp = ref_fresnel(deg2rad(45), p0.n)
        @test pane_optics(p0, deg2rad(45)).T ≈ ((1 - rs) / (1 + rs) + (1 - rp) / (1 + rp)) / 2 atol = 1e-14
        # absorbing pane: conservation at every angle; Beer–Lambert internally
        p = GlassPane(0.006)
        @test maximum(θd -> (o = pane_optics(p, deg2rad(θd)); abs(o.R + o.T + o.A - 1)), 0:10:89) < 1e-14
        ρ = ((p.n - 1) / (p.n + 1))^2; τ = exp(-p.K * 0.006)
        @test pane_optics(p, 0.0).T ≈ (1 - ρ)^2 * τ / (1 - ρ^2 * τ^2) atol = 1e-14
        # data-sheet inversion (WINDOW Eqs. 7-26…7-29): exact round trip
        r = glass_pane_from_normal(0.006, 0.77, 0.08)
        o = pane_optics(r, 0.0)
        @test o.T ≈ 0.77 atol = 1e-12
        @test o.R ≈ 0.08 atol = 1e-12
    end

    @testset "Published clear float: Vitro (ISO 9050), LLumar/NFRC, WINDOW angular regression" begin
        # Vitro Clear Technical Product Data (2020), building thicknesses
        vitro = [(3.0, 0.83, 0.08), (6.0, 0.77, 0.08), (8.0, 0.72, 0.07)]
        @test maximum(((dmm, Tv, Rv),) -> (o = pane_optics(GlassPane(dmm / 1000), 0.0);
                                           max(abs(o.T - Tv) / 0.015, abs(o.R - Rv) / 0.011)), vitro) <= 1
        # LLumar / NFRC single 3 mm clear: T 0.83, R 0.08, A 0.09
        o = pane_optics(GlassPane(0.003), 0.0)
        @test maximum(abs.((o.T, o.R, o.A) .- (0.83, 0.08, 0.09))) <= 0.01
        # WINDOW Table 7-1 clear-glass regression, 3 mm (NFRC reference pane)
        p = GlassPane(0.003); o0 = pane_optics(p, 0.0)
        worst_T = 0.0; worst_R = 0.0
        for θd in 0:20:80
            Φ = deg2rad(θd); o = pane_optics(p, Φ)
            worst_T = max(worst_T, abs(o.T - o0.T * window_tau_clr(Φ)))
            worst_R = max(worst_R, abs(o.R - (o0.R * (1 - window_rho_clr(Φ)) + window_rho_clr(Φ))))
        end
        @test worst_T <= 0.01
        @test worst_R <= 0.015
    end

    @testset "Multi-pane stacks: closed form, conservation, grazing" begin
        # N non-absorbing plates: T = (1−ρ)/(1 + (2N−1)ρ) (Duffie & Beckman Eq. 5.1.6)
        N, θ = 3, deg2rad(45)
        g = Glazing([GlassPane(0.004; K = 0.0) for _ in 1:N], [GlazingGap(0.012) for _ in 1:N-1])
        rs, rp = ref_fresnel(θ, CLEAR_FLOAT_REFRACTIVE_INDEX)
        @test glazing_optics(g, θ).T ≈ ((1 - rs) / (1 + (2N - 1) * rs) + (1 - rp) / (1 + (2N - 1) * rp)) / 2 atol = 1e-13
        # absorbing triple: R + T + ΣA = 1, outer pane absorbs most
        o = glazing_optics(triple_glazing(), deg2rad(40))
        @test o.R + o.T + sum(o.A) ≈ 1 atol = 1e-13
        @test issorted(o.A; rev = true)
        # near-grazing sunrise once made the inter-pane system singular
        o = glazing_optics(double_glazing(), acos(1e-17))
        @test o.R ≈ 1 atol = 1e-9
        h = glazing_optics_hemispherical(double_glazing())
        @test h.R + h.T + sum(h.A) ≈ 1 atol = 1e-12
    end

    @testset "Gas gaps (ISO 15099), U-value and SHGC vs published units" begin
        c = DEFAULT_SURFACE_CONSTANTS
        σ = c.σ_SB
        # ISO 15099 air fit reproduces the contract's still-air conductivity
        @test USG.gas_conductivity(GLAZING_AIR, 273.15) ≈ c.k_air rtol = 5e-3
        # exact (not linearised) radiative gap coefficient
        T1, T2, ε1, ε2 = 300.0, 270.0, 0.84, 0.10
        h = gap_conductance(GlazingGap(0.006), T1, T2, ε1, ε2)
        @test h.h_r * (T1 - T2) ≈ σ * (T1^4 - T2^4) / (1 / ε1 + 1 / ε2 - 1) rtol = 1e-12
        # single pane: series resistances
        k = GlassPane(0.004).material.conductivity
        @test glazing_u_value(single_glazing(0.004); h_out = 25.0, h_in = 7.7) ≈
              1 / (1 / 25.0 + 0.004 / k + 1 / 7.7) rtol = 1e-12
        # Vitro clear/clear 6-13-6, NFRC 100 winter (T_out −18 °C, T_in 21 °C,
        # 5.5 m/s): published centre-of-glass U 0.47 Btu/(h ft² °F); film
        # coefficients evaluated test-side (ISO 15099 Eq. 131, §8.3.2)
        ε = UNCOATED_GLASS_EMISSIVITY
        hrad(Ts, T) = ε * σ * (Ts^2 + T^2) * (Ts + T)
        g = double_glazing(thickness = 0.006, gap = 0.0127)
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
            hc_in = 0.56 * (ρ^2 * 9.80665 * cp * (T_in - Tsi) / (Tf * μ * kk))^0.25 * kk
        end
        @test abs(U - 0.47 * 5.678263) / (0.47 * 5.678263) < 0.03
        # NFRC summer SHGC: single 3 mm 0.86 (LLumar), 6-13-6 0.70 (Vitro)
        nfrc_summer(g) = glazing_shgc(g, 0.0; T_out = 305.15, T_in = 297.15,
                                      h_out = 4 + 4 * 2.75 + 4ε * σ * 305.15^3,
                                      h_in = 2.5 + 4ε * σ * 297.15^3)
        @test abs(nfrc_summer(single_glazing(0.003)) - 0.86) <= 0.01
        @test abs(nfrc_summer(g) - 0.70) <= 0.02
    end

    @testset "WindowFacet: energy bookkeeping, steady SHGC and U-value" begin
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        ε = UNCOATED_GLASS_EMISSIVITY
        g = double_glazing()
        w = WindowFacet(g; T = 290.15, T_interior = 295.15)
        @test w isa AbstractSurface
        # half a sunny day with Monin–Obukhov exchange: exact storage and SW partition
        dt = 600.0
        maxres = 0.0; maxsw = 0.0; maxbal = 0.0
        for step in 1:72
            t = step * dt + 6 * 3600
            μ = max(0.0, sin(2π * (t / 86400 - 0.25)))
            set_solar_incidence!(w, μ; diffuse_fraction = 0.3)
            f = SurfaceForcing(sw_down = 700μ, lw_down = 320.0, T_air = 290.0 + 5μ, U = 3.0)
            opt = window_optics(w)
            E0 = window_heat_content(w)
            fl = step_surface!(w, f, dt)
            maxbal = max(maxbal, abs(fl.Rn - fl.H - fl.LE - fl.G))
            maxres = max(maxres, abs(window_heat_content(w) - E0 -
                                     dt * (fl.G - interior_heat_flux(w) - transmitted_shortwave(w))))
            maxsw = max(maxsw, abs(transmitted_shortwave(w) - opt.T * f.sw_down) +
                               abs(w.sw_absorbed - sum(opt.A) * f.sw_down))
        end
        @test maxbal < 1e-9
        @test maxres < 1e-13 * window_heat_content(w)
        @test maxsw < 1e-10
        # steady states reproduce the steady SHGC (sun at 40°) and U-value
        hc = 15.0
        hi = 1 / ISO6946_SURFACE_RESISTANCE.inside_horizontal
        T0 = 298.15; θi = deg2rad(40); I = 600.0
        w = WindowFacet(g; T = T0, T_interior = T0, h_convective = hc)
        set_solar_incidence!(w, cos(θi); diffuse_fraction = 0.0)
        f = SurfaceForcing(sw_down = I, lw_down = σ * T0^4, T_air = T0)
        for _ in 1:200; step_surface!(w, f, 600.0); end
        gain = (interior_heat_flux(w) + transmitted_shortwave(w)) / I
        @test abs(gain - glazing_shgc(g, θi; h_out = hc + 4ε * σ * T0^3, h_in = hi,
                                      T_out = T0, T_in = T0)) < 0.005
        To, Ti = 268.15, 293.15
        w2 = WindowFacet(g; T = 280.0, T_interior = Ti, h_convective = hc)
        f2 = SurfaceForcing(sw_down = 0.0, lw_down = σ * To^4, T_air = To)
        for _ in 1:200; step_surface!(w2, f2, 600.0); end
        Ts = w2.T_skin
        U_st = glazing_u_value(g; T_out = To, T_in = Ti,
                               h_out = hc + ε * σ * (Ts^2 + To^2) * (Ts + To), h_in = hi)
        @test abs(-interior_heat_flux(w2) / (Ti - To) - U_st) / U_st < 0.01
    end

    @testset "src/components/urban/glass.jl delegates to UrbanSurface" begin
        # the urban case's glass class carries no numbers of its own, and the
        # transmitted shortwave leaves the cell's budget (before #27 it heated it)
        ex = Aeolus.Urban
        g = Base.invokelatest(getfield(ex, :example_glazing))
        params = Base.invokelatest(getfield(ex, :glass_surface_params))
        h = glazing_optics_hemispherical(g)
        cls = Int(getfield(ex, :SURFACE_GLASS))
        @test params.albedo[cls] ≈ h.R
        sfc = fill(UInt8(cls), 1, 1)
        loss = zeros(1, 1)
        Base.invokelatest(getfield(ex, :apply_glass_sw_transmission!), loss, sfc, fill(600.0, 1, 1))
        @test loss[1, 1] ≈ h.T * 600.0
    end
end
