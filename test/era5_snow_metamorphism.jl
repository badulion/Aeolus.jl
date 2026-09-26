# Phase 51 (#35) — snow grain metamorphism and albedo ageing.
#
# Grain growth: the Flanner & Zender (2006) dry-ageing fits (the CLM /
# SNICAR look-up table, shipped unchanged) and Brun (1989) wet growth,
# checked against their closed-form solutions, against a literal port of
# CLM's forward-Euler update, against the trends Flanner & Zender report,
# and against the SSA decrease observed at Summit, Greenland (Carmagnola
# et al. 2014). Albedo: delegated to SurfaceOptics.SnowSurface
# (Wiscombe-Warren) — checked for exact delegation, for the spectral
# signature of grain growth, and against the fresh-snow values and 30-day
# decay curves of four reference land-surface schemes.

using Test, Printf
using Aeolus
using Aeolus.Snowpack
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS
using Aeolus.MoistThermodynamics: saturation_specific_humidity

const SPm = Aeolus.Snowpack
const SOm = Aeolus.SurfaceOptics
const Tf_m = DEFAULT_SURFACE_CONSTANTS.thermo.T_freeze

# A pack of explicit layers (top first), bypassing the layering rules.
function layered(dz, ρ, T; grain = fill(54.5, length(dz)), liq = zeros(length(dz)), kwargs...)
    p = LayeredSnowpack(; kwargs...)
    for i in eachindex(dz)
        SPm._push_layer!(p, dz[i], ρ[i] * dz[i] - liq[i], liq[i], T[i], grain[i], 0.0)
    end
    return p
end

# Reference-model broadband albedo curves (days since snowfall), used only
# as comparison data — none of them is part of the model.
#  * US Army Corps of Engineers (1956), accumulation season, as coded in
#    VIC 5 (`snow_albedo`, SNOW_NEW_SNOW_ALB = 0.85, ACCUM_A = 0.94, B = 0.58).
usace_accum(d) = 0.85 * 0.94^(d^0.58)
#  * ECMWF (Douville et al. 1995; IFS documentation Part IV): dry snow,
#    linear decrease 0.008 per day from 0.85.
ecmwf_dry(d) = max(0.85 - 0.008d, 0.5)
#  * CLASS (Verseghy 1991) as in Noah-MP (SnowAlbedoClassMod, CLASS_ALB_NEW
#    = 0.84, CLASS_ALB_REF = 0.55, e-folding 100 h).
class_alb(d) = 0.55 + (0.84 - 0.55) * exp(-0.01 * d * 24)
#  * BATS (Dickinson et al. 1993; Yang et al. 1997) as in Noah-MP
#    (SnowAgingBatsMod / SnowAlbedoBatsMod, NoahmpTable.TBL): non-dimensional
#    age at ground temperature T, diffuse VIS/NIR albedos, and their mean.
function bats_alb(d, T)
    r1 = exp(5000 * (1 / 273.16 - 1 / T))
    τ = d * 86400 / 1e6 * (r1 + min(r1^10, 1.0) + 0.3)
    F = τ / (1 + τ)
    return (vis = 0.95 * (1 - 0.2F), nir = 0.65 * (1 - 0.5F))
end

@testset "Era V — Snow metamorphism and albedo ageing (#35)" begin

    tab = snicar_aging_table()

    @testset "SNICAR ageing table: provenance, structure, SSA ↔ radius" begin
        @test size(tab.drdt0) == size(tab.τ) == size(tab.κ) == (8, 31, 11)
        @test tab.T[1] ≈ 223.15 && tab.T[end] ≈ 273.15
        @test tab.dTdz[1] == 0 && tab.dTdz[end] == 300
        @test tab.ρ[1] == 50 && tab.ρ[end] == 400
        @test occursin("r_e0= 54.5", tab.description)
        @test tab.r0 == 54.5
        @test all(tab.drdt0 .> 0) && all(tab.τ .> 0) && all(tab.κ .> 0)
        # cached: the default table is read once
        @test snicar_aging_table() === tab
        # r = 3/(ρ_ice SSA): 54.5 μm ⇔ 60 m² kg⁻¹ with ρ_ice = 917 (CLM5 TN §2.3.2.3)
        @test snow_specific_surface_area(54.5) ≈ 60.03 atol = 0.01
        @test snow_optical_radius(snow_specific_surface_area(123.4)) ≈ 123.4 rtol = 1e-14
    end

    @testset "Flanner & Zender closed form: ODE, composition, initial rate" begin
        for (i, j, k) in ((2, 1, 11), (5, 3, 7), (1, 11, 9), (8, 31, 1))
            a, τ, κ = tab.drdt0[i, j, k], tab.τ[i, j, k], tab.κ[i, j, k]
            rate(r) = a * (τ / (r - 54.5 + τ))^(1 / κ)
            # initial slope at r = r0 is (dr/dt)_0
            h = 1e-6
            @test (flanner_grain_radius(54.5, h, a, τ, κ) - 54.5) / h ≈ a rtol = 1e-4
            # it solves the ODE at an arbitrary later time
            r5 = flanner_grain_radius(54.5, 120.0, a, τ, κ)
            d = (flanner_grain_radius(54.5, 120.0 + 1e-3, a, τ, κ) -
                 flanner_grain_radius(54.5, 120.0 - 1e-3, a, τ, κ)) / 2e-3
            @test d ≈ rate(r5) rtol = 1e-6
            # composition: ageing t1 then t2 is ageing t1 + t2 (time-step independence)
            @test flanner_grain_radius(r5, 600.0, a, τ, κ) ≈
                  flanner_grain_radius(54.5, 720.0, a, τ, κ) rtol = 1e-12
            @test flanner_grain_radius(r5, 0.0, a, τ, κ) ≈ r5 rtol = 1e-14
        end
    end

    @testset "Table trends reported by Flanner & Zender (2006)" begin
        # "warm snow, large temperature gradient and low density produce the
        # most rapid ageing" — 30-day growth from r0, along each axis.
        r30(i, j, k) = flanner_grain_radius(54.5, 720.0, tab.drdt0[i, j, k], tab.τ[i, j, k], tab.κ[i, j, k])
        up_T = [r30(i, j, k + 1) > r30(i, j, k) for i in 1:8, j in 1:31, k in 1:10]
        up_G = [r30(i, j + 1, k) >= r30(i, j, k) for i in 1:8, j in 1:30, k in 1:11]
        dn_ρ = [r30(i + 1, j, k) <= r30(i, j, k) for i in 1:7, j in 1:31, k in 1:11]
        @printf("  F06 table monotone cells: T %.1f %%, ∇T %.1f %%, ρ %.1f %%\n",
                100 * sum(up_T) / length(up_T), 100 * sum(up_G) / length(up_G),
                100 * sum(dn_ρ) / length(dn_ρ))
        @test sum(up_T) / length(up_T) > 0.97      # measured 98.1 %
        @test sum(up_G) / length(up_G) > 0.90      # measured 93.0 %
        @test sum(dn_ρ) / length(dn_ρ) > 0.92      # measured 94.2 %
        # the representative column: ρ = 100, ∇T = 0, −50 → 0 °C strictly warmer-faster
        @test issorted([r30(2, 1, k) for k in 1:11]) && r30(2, 1, 11) > 1.9 * r30(2, 1, 1)
        # at −10 °C, ρ = 100: 30-day radius 121 μm at 0 K m⁻¹, 296 μm at 20 K m⁻¹
        @test r30(2, 1, 9) ≈ 120.6 atol = 0.1
        @test r30(2, 3, 9) ≈ 295.5 atol = 0.1
    end

    @testset "Interpolation blends outcomes, never parameters" begin
        # At the nodes, :linear is the node itself.
        for (i, j, k) in ((3, 4, 5), (8, 1, 11), (1, 31, 1))
            @test snicar_grain_radius(tab, 80.0, 500.0, tab.T[k], tab.dTdz[j], tab.ρ[i]) ≈
                  snicar_grain_radius(tab, 80.0, 500.0, tab.T[k], tab.dTdz[j], tab.ρ[i];
                                      interpolation = :nearest) rtol = 1e-12
        end
        # The documented pathology of interpolating the fitted parameters:
        # −25 °C, 50 K m⁻¹, 37 days from 88.4 μm; nodes at ρ = 300 and 350.
        r_a = snicar_grain_radius(tab, 88.42, 888.0, 248.15, 50.0, 300.0)
        r_b = snicar_grain_radius(tab, 88.42, 888.0, 248.15, 50.0, 350.0)
        r_m = snicar_grain_radius(tab, 88.42, 888.0, 248.15, 50.0, 325.0)
        @test r_a ≈ 181.1 atol = 0.1
        @test r_b ≈ 122.7 atol = 0.1
        @test r_m ≈ (r_a + r_b) / 2 rtol = 1e-12          # (parameter interpolation: 484 μm)
        # Bounded by the corner outcomes everywhere, and continuous across a
        # cell boundary where nearest-node look-up jumps.
        for T in (230.0, 251.3, 266.6, 272.9), G in (3.0, 17.0, 44.0), ρ in (70.0, 180.0, 333.0)
            rs = [flanner_grain_radius(60.0, 240.0, tab.drdt0[i, j, k], tab.τ[i, j, k], tab.κ[i, j, k])
                  for (i, j, k, w) in SPm._snicar_corners(tab, T, G, ρ) if w > 0]
            r = snicar_grain_radius(tab, 60.0, 240.0, T, G, ρ)
            @test minimum(rs) - 1e-9 <= r <= maximum(rs) + 1e-9
        end
        jump_near = snicar_grain_radius(tab, 60.0, 240.0, 263.15, 15.01, 150.0; interpolation = :nearest) -
                    snicar_grain_radius(tab, 60.0, 240.0, 263.15, 14.99, 150.0; interpolation = :nearest)
        jump_lin = snicar_grain_radius(tab, 60.0, 240.0, 263.15, 15.01, 150.0) -
                   snicar_grain_radius(tab, 60.0, 240.0, 263.15, 14.99, 150.0)
        @test abs(jump_near) > 1.0
        @test abs(jump_lin) < 0.01 * abs(jump_near)
        @test_throws ArgumentError snicar_grain_radius(tab, 60.0, 1.0, 260.0, 0.0, 100.0;
                                                       interpolation = :cubic)
    end

    @testset "Against a literal port of CLM's SnowAge_grain (forward Euler, nearest node)" begin
        # CTSM src/biogeophys/SnowSnicarMod.F90: dr = drdt0 (τ/(r - r_min + τ))^(1/κ) dt/3600,
        # nearest-index look-up, 30-min steps. Same table, same law: the only
        # difference is our exact integration within each step.
        m = Flanner2006Metamorphism(interpolation = :nearest)
        for (T, G, ρ) in ((268.15, 20.0, 100.0), (258.15, 0.0, 250.0), (272.15, 60.0, 150.0))
            q = snicar_aging_parameters(tab, T, G, ρ)
            r_clm = 54.5
            for _ in 1:(30 * 48)
                r_clm += q.drdt0 * (q.τ / (r_clm - 54.5 + q.τ))^(1 / q.κ) * 0.5
            end
            r_ours = snicar_grain_radius(tab, 54.5, 720.0, T, G, ρ; interpolation = :nearest)
            @printf("  CLM port vs exact, T=%.2f ∇T=%.0f ρ=%.0f: %.2f vs %.2f μm\n", T, G, ρ, r_clm, r_ours)
            @test abs(r_clm - r_ours) / r_ours < 5e-3
            # and the growth rate the scheme reports is CLM's
            @test snow_dry_grain_growth_rate(m, 100.0, T, G, ρ) * 3600 ≈
                  q.drdt0 * (q.τ / (100.0 - 54.5 + q.τ))^(1 / q.κ) rtol = 1e-12
        end
    end

    @testset "Wet growth (Brun 1989, as in CLM)" begin
        m = Flanner2006Metamorphism()
        # dr/dt = 10¹⁸ C₁ f³ / (4π r²), C₁ = 4.22e-13: at f = 0.05, r = 100 μm
        @test snow_wet_grain_growth_rate(m, 100.0, 0.05) ≈ 1e18 * 4.22e-13 * 0.05^3 / (4π * 1e4) rtol = 1e-14
        @test snow_wet_grain_growth_rate(m, 100.0, 0.05) * 86400 ≈ 36.26 atol = 0.01   # μm per day
        # f_liq capped at 0.1 (CTSM), zero for dry snow
        @test snow_wet_grain_growth_rate(m, 100.0, 0.3) == snow_wet_grain_growth_rate(m, 100.0, 0.1)
        @test snow_wet_grain_growth_rate(m, 100.0, 0.0) == 0
        # In a wet, isothermal (∇T = 0) layer at 0 °C, dry and wet growth add;
        # the wet part is exact in r³, checked against fine Euler steps.
        mdry = Flanner2006Metamorphism(C_wet = 0.0)
        pw = layered([0.1], [300.0], [Tf_m]; liq = [0.05 * 30.0], metamorphism = m,
                     bottom = SnowGroundTemperature(Tf_m))
        pd = layered([0.1], [300.0], [Tf_m]; metamorphism = mdry, bottom = SnowGroundTemperature(Tf_m))
        pw_split = layered([0.1], [300.0], [Tf_m]; liq = [0.05 * 30.0], metamorphism = mdry,
                           bottom = SnowGroundTemperature(Tf_m))
        dt = 86400.0
        snow_metamorphism!(m, pw, dt); snow_metamorphism!(mdry, pd, dt)
        # wet-only integral from the dry result, by 10⁴ Euler steps
        r = pd.grain[1]
        for _ in 1:10_000
            r += snow_wet_grain_growth_rate(m, r, 0.05) * dt / 10_000
        end
        @test pw.grain[1] ≈ r rtol = 1e-4
        @printf("  one day at 0 °C, ρ = 300, ∇T = 0: dry %.1f μm, wet (f = 0.05) %.1f μm\n",
                pd.grain[1], pw.grain[1])
        @test pw.grain[1] > pd.grain[1] + 30
        # ten days of wet ageing at f = 0.1 from 54.5 μm: r³ grows by 3e18 C₁ f³ t/(4π),
        # (54.5³ + 100.74 μm³ s⁻¹ × 864000 s)^(1/3) = 443.45 μm — wet snow
        # coarsens eight-fold in ten days; dry snow at 0 °C, ∇T = 0, ρ = 100 only to 108 μm.
        @test cbrt(54.5^3 + 3e18 * 4.22e-13 * 0.1^3 / (4π) * 10 * 86400) ≈ 443.45 atol = 0.01
    end

    @testset "Layer temperature gradient (CLM5 TN Eq. 2.3.74)" begin
        # Linear profile T = 260 + 20 z (K, z m below the surface) in 4 equal
        # layers on ground held at the extrapolated temperature.
        dz = fill(0.1, 4)
        zc = [0.05, 0.15, 0.25, 0.35]
        p = layered(dz, fill(200.0, 4), 260 .+ 20 .* zc; bottom = SnowGroundTemperature(260 + 20 * 0.4))
        g = snow_temperature_gradient(p)
        @test g[2] ≈ 20 rtol = 1e-12
        @test g[3] ≈ 20 rtol = 1e-12
        @test g[4] ≈ 20 rtol = 1e-12
        @test g[1] ≈ 10 rtol = 1e-12          # CLM: top interface at the top-layer temperature
        # Flux base: the lower interface temperature carries F across Δz/2.
        pf = layered([0.1], [200.0], [265.0]; bottom = SnowBottomFlux(-2.0))
        k = snow_thermal_conductivity(pf.conductivity, 200.0)
        @test snow_temperature_gradient(pf)[1] ≈ 2.0 * 0.05 / k / 0.1 rtol = 1e-12
        @test snow_temperature_gradient(layered([0.1], [200.0], [265.0]))[1] == 0
    end

    @testset "Scheme on a pack: bounds, ages, step independence, fresh and refrozen grains" begin
        m = Flanner2006Metamorphism()
        prof(z) = 262.0 + 15z
        mk() = layered([0.03, 0.06, 0.12], [120.0, 200.0, 280.0], prof.([0.015, 0.06, 0.15]);
                       metamorphism = m, bottom = SnowGroundTemperature(prof(0.21)))
        p1, pN = mk(), mk()
        snow_metamorphism!(m, p1, 10 * 86400.0)
        for _ in 1:480
            snow_metamorphism!(m, pN, 1800.0)
        end
        @printf("  10 days: one step %s μm, 480 steps %s μm\n", round.(p1.grain; digits = 2),
                round.(pN.grain; digits = 2))
        @test maximum(abs.(p1.grain .- pN.grain) ./ pN.grain) < 0.02
        @test all(pN.age .≈ 10 * 86400)
        @test all(54.5 .< pN.grain .< 1500)
        # nearest-node look-up composes exactly at fixed state
        mn = Flanner2006Metamorphism(interpolation = :nearest)
        q1 = layered([0.1], [150.0], [263.0]; metamorphism = mn, bottom = SnowGroundTemperature(265.0))
        qN = layered([0.1], [150.0], [263.0]; metamorphism = mn, bottom = SnowGroundTemperature(265.0))
        snow_metamorphism!(mn, q1, 5 * 86400.0)
        for _ in 1:240
            snow_metamorphism!(mn, qN, 1800.0)
        end
        @test q1.grain[1] ≈ qN.grain[1] rtol = 1e-10
        # bounds
        pb = layered([0.1], [100.0], [Tf_m]; grain = [1490.0], liq = [1.0], metamorphism = m)
        snow_metamorphism!(m, pb, 30 * 86400.0)
        @test pb.grain[1] == 1500.0
        pl = layered([0.1], [100.0], [250.0]; grain = [40.0], metamorphism = m)
        snow_metamorphism!(m, pl, 1.0)
        @test pl.grain[1] >= 54.5
        # fresh snow: r_min by default; CLM5 ramp when enabled
        pf = LayeredSnowpack(metamorphism = m)
        add_snowfall!(pf, 5.0, 263.15, 100.0)
        @test pf.grain[1] == 54.5
        mr = Flanner2006Metamorphism(fresh_radius_max = 204.5)
        @test fresh_grain_radius(mr, pf, Tf_m - 15) ≈ 129.5
        @test fresh_grain_radius(mr, pf, Tf_m - 40) == 54.5
        @test fresh_grain_radius(mr, pf, Tf_m + 3) == 204.5
        # snowfall onto old snow: grain mass-weighted (CLM5 TN Eq. 2.3.72)
        po = layered([0.05], [200.0], [263.0]; grain = [300.0], metamorphism = m)
        add_snowfall!(po, 10.0, 263.0, 100.0)
        @test po.grain[1] ≈ (10 * 300 + 10 * 54.5) / 20 rtol = 1e-12
        # refreezing (#36's entry point): mass-weighted towards 1000 μm
        pr = layered([0.05], [200.0], [263.0]; grain = [100.0], metamorphism = m)
        mix_refrozen_grain!(pr, 1, 1.0)
        @test pr.grain[1] ≈ 0.9 * 100 + 0.1 * 1000 rtol = 1e-12
        p0 = layered([0.05], [200.0], [263.0]; grain = [100.0])
        mix_refrozen_grain!(p0, 1, 1.0)
        @test p0.grain[1] == 100.0                 # NoSnowMetamorphism: unchanged
        # merging (and re-splitting) layers conserves Σ m r
        pm = layered([0.004, 0.05], [100.0, 200.0], [263.0, 263.0]; grain = [60.0, 400.0])
        s0 = sum((pm.m_ice .+ pm.m_liq) .* pm.grain)
        SPm._merge_layers!(pm, 1, 2)
        @test pm.grain[1] ≈ (0.4 * 60 + 10 * 400) / 10.4 rtol = 1e-12
        @test sum((pm.m_ice .+ pm.m_liq) .* pm.grain) ≈ s0 rtol = 1e-12
        relayer_snowpack!(pm)
        @test sum((pm.m_ice .+ pm.m_liq) .* pm.grain) ≈ s0 rtol = 1e-12
    end

    @testset "Parabolic growth is SurfaceOptics' own ageing, made prognostic" begin
        pg = LayeredSnowpack(depth = 0.3, density = 200.0, metamorphism = ParabolicGrainGrowth(),
                             albedo = SnowOpticsAlbedo())
        for _ in 1:(7 * 24)
            snow_metamorphism!(pg.metamorphism, pg, 3600.0)
        end
        @test all(pg.grain .≈ snow_grain_radius(SNOW_REFERENCE_RADIUS, 7.0))
        @test surface_albedo(pg) ≈ broadband_albedo(SnowSurface(age_days = 7.0)) rtol = 1e-12
    end

    @testset "SnowOpticsAlbedo delegates to SurfaceOptics.SnowSurface" begin
        a = SnowOpticsAlbedo()
        p = layered([0.01, 0.01, 0.2], [150.0, 200.0, 250.0], fill(263.0, 3);
                    grain = [100.0, 300.0, 800.0], albedo = a)
        @test snowpack_optical_radius(p) ≈ 200.0 rtol = 1e-12            # top 2 cm, thickness-weighted
        @test snowpack_optical_radius(p; sensing_depth = 0.01) ≈ 100.0
        @test surface_albedo(p) == broadband_albedo(SnowSurface(grain_radius = 200.0))
        s = snowpack_optics(a, p)
        @test s isa SnowSurface && s.effective_radius ≈ 200.0
        # soot darkens; direct beam at high sun below low sun
        ps = layered([0.01, 0.01, 0.2], [150.0, 200.0, 250.0], fill(263.0, 3);
                     grain = [100.0, 300.0, 800.0], albedo = SnowOpticsAlbedo(soot_ppmw = 0.5))
        @test surface_albedo(ps) == broadband_albedo(SnowSurface(grain_radius = 200.0, soot_ppmw = 0.5))
        @test surface_albedo(ps) < surface_albedo(p) - 0.03
        hi = SnowOpticsAlbedo(μ₀ = 1.0); lo = SnowOpticsAlbedo(μ₀ = 0.2)
        @test snowpack_albedo(hi, p) < snowpack_albedo(a, p) < snowpack_albedo(lo, p)
        # monotone in grain size
        αs = [broadband_albedo(SnowSurface(grain_radius = r)) for r in (54.5, 100, 200, 500, 1000, 1500)]
        @test issorted(αs; rev = true)
        # bare ground: the fresh-snow value
        @test surface_albedo(LayeredSnowpack(albedo = a)) ≈ broadband_albedo(SnowSurface(grain_radius = 50.0))
        # the energy balance sees it: older snow absorbs more sunshine
        f = SurfaceForcing(sw_down = 600.0, lw_down = 250.0, T_air = 265.0, q_air = 0.001,
                           U = 3.0, p = 90000.0, z_ref = 2.0, rain = 0.0, snow = 0.0)
        pf = layered([0.02], [150.0], [265.0]; grain = [54.5], albedo = a)
        pa = layered([0.02], [150.0], [265.0]; grain = [500.0], albedo = a)
        ΔRn = net_radiation(pa, f) - net_radiation(pf, f)
        @test ΔRn ≈ 600 * (surface_albedo(pf) - surface_albedo(pa)) rtol = 1e-12
        @printf("  54.5 → 500 μm: albedo %.3f → %.3f, +%.1f W m⁻² absorbed at 600 W m⁻²\n",
                surface_albedo(pf), surface_albedo(pa), ΔRn)
        @test ΔRn > 60
        @test_throws ArgumentError SnowOpticsAlbedo(soot_ppmw = -1)
    end

    @testset "Fresh-snow albedo and the spectral signature of grain growth" begin
        spec = shortwave_bands()
        w = SOm._band_solar_weights(Float64, spec)
        vis = [b for b in 1:SOm.nbands(spec) if SOm.band_center_wavelength(spec, b) < 0.7]
        nir = setdiff(1:SOm.nbands(spec), vis)
        split(s) = (sum(w[vis] .* s.diffuse[vis]) / sum(w[vis]), sum(w[nir] .* s.diffuse[nir]) / sum(w[nir]))
        s0 = SnowSurface(grain_radius = 54.5)
        α0 = broadband_albedo(s0)
        v0, n0 = split(s0)
        @printf("  fresh (54.5 μm): broadband %.3f, VIS %.3f, NIR %.3f\n", α0, v0, n0)
        # Reference fresh-snow broadband albedos: CLASS 0.84 (Verseghy 1991;
        # Noah-MP), US Army Corps 0.85 (VIC), ECMWF 0.85 (Douville et al. 1995).
        @test abs(α0 - 0.84) < 0.015 && abs(α0 - 0.85) < 0.015
        # BATS fresh snow: NIR 0.65, VIS 0.95. The NIR agrees; the clean
        # Wiscombe-Warren VIS is higher because BATS's 0.95 carries impurities.
        @test abs(n0 - 0.65) < 0.04
        @test 0.95 < v0 < 0.99
        # Col de Porte observations: daily-integrated albedo between < 0.5 (old,
        # dirty, melting) and 0.95 (fresh) (Carmagnola et al. 2014, Sect. 4.1).
        @test 0.5 < α0 < 0.95
        # Grain growth is a near-infrared process (Wiscombe & Warren 1980):
        # 54.5 → 300 μm cuts NIR by ≥ 5× the relative cut in VIS. BATS's
        # age factors (0.5 NIR : 0.2 VIS) give 2.5×, the rest being impurity darkening.
        v3, n3 = split(SnowSurface(grain_radius = 300.0))
        ratio = ((n0 - n3) / n0) / ((v0 - v3) / v0)
        @printf("  54.5 → 300 μm: VIS −%.1f %%, NIR −%.1f %% (ratio %.1f; BATS 2.5)\n",
                100 * (v0 - v3) / v0, 100 * (n0 - n3) / n0, ratio)
        @test ratio > 5
    end

    @testset "Observed SSA decrease at Summit, Greenland (Carmagnola et al. 2014)" begin
        # Carmagnola et al. (2014, Sect. 4.1): at 2-15 cm depth, 5-19 May 2011
        # SSA = 37 ± 3 m² kg⁻¹, ρ = 315 ± 13 kg m⁻³; 11-25 June SSA = 23 ± 3,
        # ρ = 335 ± 20; i.e. 37 days (centre to centre) at ≈ 325 kg m⁻³. The
        # snow temperature and gradient are NOT given there; we scan an
        # assumed range for a cold ice-sheet spring (−20 to −10 °C,
        # 10-30 K m⁻¹) and require the observation to lie inside it.
        r0 = snow_optical_radius(37.0)
        ssa = [snow_specific_surface_area(snicar_grain_radius(tab, r0, 37 * 24.0, Tf_m + Tc, G, 325.0))
               for Tc in (-20, -15, -10), G in (10, 20, 30)]
        println("  Summit, SSA after 37 d (rows −20/−15/−10 °C, cols 10/20/30 K m⁻¹):")
        for i in 1:3
            println("    ", join((@sprintf("%.1f", x) for x in ssa[i, :]), "  "))
        end
        @test minimum(ssa) < 23 < maximum(ssa)
        @test abs(ssa[2, 2] - 23) < 3                  # −15 °C, 20 K m⁻¹: within the observed ±3
        # without a gradient, even the warm end is only marginally inside
        # the observed range: the decrease needs temperature-gradient metamorphism
        g0 = snow_specific_surface_area(snicar_grain_radius(tab, r0, 37 * 24.0, Tf_m - 15, 0.0, 325.0))
        @test g0 > 26
    end

    @testset "30-day albedo decay of a clear-sky pack vs reference-model curves" begin
        # A 50 cm, 150 kg m⁻³ pack under a clear-sky diurnal cycle at −10 °C
        # (SW peak 500 W m⁻², LW 230 W m⁻², RH 80 %, U = 3 m s⁻¹), 30-min steps,
        # no new snow, clean snow and 0.05 ppmw of black carbon.
        function season(soot)
            p = LayeredSnowpack(depth = 0.5, density = 150.0, T = 263.15,
                                metamorphism = Flanner2006Metamorphism(),
                                albedo = SnowOpticsAlbedo(soot_ppmw = soot),
                                bottom = SnowGroundTemperature(272.0))
            αd = [surface_albedo(p)]
            dt = 1800.0
            for n in 1:(30 * 48)
                h = mod(n * dt / 3600, 24)
                Ta = 263.15 + 4 * sin(π * (h - 9) / 12)
                qa = 0.8 * saturation_specific_humidity(Ta, 85000.0, DEFAULT_SURFACE_CONSTANTS.thermo; over = :ice)
                f = SurfaceForcing(sw_down = max(0.0, 500 * sin(π * (h - 6) / 12)), lw_down = 230.0,
                                   T_air = Ta, q_air = qa, U = 3.0, p = 85000.0, z_ref = 2.0,
                                   rain = 0.0, snow = 0.0)
                step_surface!(p, f, dt)
                n % 48 == 0 && push!(αd, surface_albedo(p))
            end
            return αd, p
        end
        αc, pc = season(0.0)
        αs, _ = season(0.05)
        Tg = pc.T[1]
        println("  day   clean  BC0.05  USACE  ECMWF  CLASS  BATS(VIS,NIR)")
        for d in (0, 1, 5, 10, 20, 30)
            b = bats_alb(d, Tg)
            @printf("  %3d   %.3f  %.3f   %.3f  %.3f  %.3f  %.3f, %.3f\n", d, αc[d+1], αs[d+1],
                    usace_accum(d), ecmwf_dry(d), class_alb(d), b.vis, b.nir)
        end
        # Physics: albedo only falls without new snow, and falls more with soot.
        @test all(diff(αc) .<= 1e-12)
        @test all(αs .< αc)
        # Fresh start agrees with the reference fresh values (above).
        @test abs(αc[1] - 0.85) < 0.01
        # Clean-snow grain growth alone darkens *more slowly* than every
        # empirical curve, which are fitted to real snow and so include
        # impurity deposition (BATS's explicit "dirt and soot" term 0.3 is
        # most of its cold-snow ageing). This is a physical bound, stated as one:
        for d in (5, 10, 30)
            b = bats_alb(d, Tg)
            @test αc[d+1] > max(usace_accum(d), ecmwf_dry(d), class_alb(d), (b.vis + b.nir) / 2)
        end
        # ... and it is not negligible: ≥ 0.06 in 30 days, from grains alone.
        @test αc[1] - αc[end] > 0.06
        # CLASS/USACE/ECMWF/BATS span 0.55-0.64 at day 30; with 0.05 ppmw BC
        # (mid-latitude seasonal snow) the pack reaches ≤ 0.74.
        @test αs[end] < 0.74
    end
end
