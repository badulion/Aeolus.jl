# Phase 51 (#35) — snow grain metamorphism and albedo ageing.
#
# Grain growth: the Flanner & Zender (2006) dry-ageing fits (the CLM /
# SNICAR look-up table, shipped unchanged) and Brun (1989) wet growth,
# checked against their closed-form solutions, against a literal port of
# CLM's forward-Euler update, against the trends Flanner & Zender report,
# and against the SSA decrease observed at Summit, Greenland (Carmagnola
# et al. 2014). Albedo: delegated to SurfaceOptics.SnowSurface
# (Wiscombe-Warren) — checked for exact delegation, for the spectral
# signature of grain growth, and against the fresh-snow values of
# reference land-surface schemes. The 30-day albedo decay of a clear-sky
# pack against the curves of four reference schemes is a validation run:
# validation/snow_metamorphism.jl.

using Test
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

@testset "Era V — Snow metamorphism and albedo ageing (#35)" begin

    tab = snicar_aging_table()

    @testset "SNICAR ageing table and the Flanner & Zender closed form" begin
        @test size(tab.drdt0) == (8, 31, 11) && tab.r0 == 54.5
        # r = 3/(ρ_ice SSA): 54.5 μm ⇔ 60 m² kg⁻¹ (CLM5 TN §2.3.2.3)
        @test snow_specific_surface_area(54.5) ≈ 60.03 atol = 0.01
        a, τ, κ = tab.drdt0[5, 3, 7], tab.τ[5, 3, 7], tab.κ[5, 3, 7]
        rate(r) = a * (τ / (r - 54.5 + τ))^(1 / κ)
        # it solves the ODE, and composes exactly (time-step independence)
        r5 = flanner_grain_radius(54.5, 120.0, a, τ, κ)
        d = (flanner_grain_radius(54.5, 120.0 + 1e-3, a, τ, κ) -
             flanner_grain_radius(54.5, 120.0 - 1e-3, a, τ, κ)) / 2e-3
        @test d ≈ rate(r5) rtol = 1e-6
        @test flanner_grain_radius(r5, 600.0, a, τ, κ) ≈ flanner_grain_radius(54.5, 720.0, a, τ, κ) rtol = 1e-12
    end

    @testset "Table trends reported by Flanner & Zender (2006)" begin
        # warm snow, large gradient and low density age fastest (30 days)
        r30(i, j, k) = flanner_grain_radius(54.5, 720.0, tab.drdt0[i, j, k], tab.τ[i, j, k], tab.κ[i, j, k])
        up_T = [r30(i, j, k + 1) > r30(i, j, k) for i in 1:8, j in 1:31, k in 1:10]
        dn_ρ = [r30(i + 1, j, k) <= r30(i, j, k) for i in 1:7, j in 1:31, k in 1:11]
        @test sum(up_T) / length(up_T) > 0.97      # measured 98.1 %
        @test sum(dn_ρ) / length(dn_ρ) > 0.92      # measured 94.2 %
        # −10 °C, ρ = 100: 121 μm at 0 K m⁻¹, 296 μm at 20 K m⁻¹
        @test r30(2, 1, 9) ≈ 120.6 atol = 0.1
        @test r30(2, 3, 9) ≈ 295.5 atol = 0.1
    end

    @testset "Interpolation blends outcomes, never parameters" begin
        # parameter interpolation would give 484 μm here
        r_a = snicar_grain_radius(tab, 88.42, 888.0, 248.15, 50.0, 300.0)
        r_b = snicar_grain_radius(tab, 88.42, 888.0, 248.15, 50.0, 350.0)
        @test snicar_grain_radius(tab, 88.42, 888.0, 248.15, 50.0, 325.0) ≈ (r_a + r_b) / 2 rtol = 1e-12
        ok = true
        for T in (230.0, 266.6), G in (3.0, 44.0), ρ in (70.0, 333.0)
            rs = [flanner_grain_radius(60.0, 240.0, tab.drdt0[i, j, k], tab.τ[i, j, k], tab.κ[i, j, k])
                  for (i, j, k, w) in SPm._snicar_corners(tab, T, G, ρ) if w > 0]
            r = snicar_grain_radius(tab, 60.0, 240.0, T, G, ρ)
            ok &= minimum(rs) - 1e-9 <= r <= maximum(rs) + 1e-9
        end
        @test ok
    end

    @testset "Against a literal port of CLM's SnowAge_grain (forward Euler, nearest node)" begin
        # CTSM SnowSnicarMod.F90, 30-min steps; we integrate each step exactly
        m = Flanner2006Metamorphism(interpolation = :nearest)
        T, G, ρ = 268.15, 20.0, 100.0
        q = snicar_aging_parameters(tab, T, G, ρ)
        r_clm = 54.5
        for _ in 1:(30 * 48)
            r_clm += q.drdt0 * (q.τ / (r_clm - 54.5 + q.τ))^(1 / q.κ) * 0.5
        end
        r_ours = snicar_grain_radius(tab, 54.5, 720.0, T, G, ρ; interpolation = :nearest)
        @test abs(r_clm - r_ours) / r_ours < 2e-3     # measured ≤ 0.095 %
        @test snow_dry_grain_growth_rate(m, 100.0, T, G, ρ) * 3600 ≈
              q.drdt0 * (q.τ / (100.0 - 54.5 + q.τ))^(1 / q.κ) rtol = 1e-12
    end

    @testset "Wet growth (Brun 1989) and the layer temperature gradient" begin
        m = Flanner2006Metamorphism()
        @test snow_wet_grain_growth_rate(m, 100.0, 0.05) ≈ 1e18 * 4.22e-13 * 0.05^3 / (4π * 1e4) rtol = 1e-14
        @test snow_wet_grain_growth_rate(m, 100.0, 0.3) == snow_wet_grain_growth_rate(m, 100.0, 0.1)
        # wet part exact in r³: against fine Euler steps from the dry result
        mdry = Flanner2006Metamorphism(C_wet = 0.0)
        pw = layered([0.1], [300.0], [Tf_m]; liq = [0.05 * 30.0], metamorphism = m,
                     bottom = SnowGroundTemperature(Tf_m))
        pd = layered([0.1], [300.0], [Tf_m]; metamorphism = mdry, bottom = SnowGroundTemperature(Tf_m))
        dt = 86400.0
        snow_metamorphism!(m, pw, dt); snow_metamorphism!(mdry, pd, dt)
        r = pd.grain[1]
        for _ in 1:10_000
            r += snow_wet_grain_growth_rate(m, r, 0.05) * dt / 10_000
        end
        @test pw.grain[1] ≈ r rtol = 1e-4
        # CLM5 TN Eq. 2.3.74 on a linear profile: 20 K/m inside, 10 K/m at the top
        zc = [0.05, 0.15, 0.25, 0.35]
        p = layered(fill(0.1, 4), fill(200.0, 4), 260 .+ 20 .* zc; bottom = SnowGroundTemperature(260 + 20 * 0.4))
        @test snow_temperature_gradient(p) ≈ [10.0, 20.0, 20.0, 20.0] rtol = 1e-12
    end

    @testset "Scheme on a pack: step independence, refrozen grains, merging" begin
        m = Flanner2006Metamorphism()
        prof(z) = 262.0 + 15z
        mk() = layered([0.03, 0.06, 0.12], [120.0, 200.0, 280.0], prof.([0.015, 0.06, 0.15]);
                       metamorphism = m, bottom = SnowGroundTemperature(prof(0.21)))
        p1, pN = mk(), mk()
        snow_metamorphism!(m, p1, 10 * 86400.0)
        for _ in 1:40
            snow_metamorphism!(m, pN, 6 * 3600.0)
        end
        @test maximum(abs.(p1.grain .- pN.grain) ./ pN.grain) < 0.02
        # snowfall onto old snow: mass-weighted (CLM5 TN Eq. 2.3.72)
        po = layered([0.05], [200.0], [263.0]; grain = [300.0], metamorphism = m)
        add_snowfall!(po, 10.0, 263.0, 100.0)
        @test po.grain[1] ≈ (10 * 300 + 10 * 54.5) / 20 rtol = 1e-12
        # refreezing (#36's entry point): mass-weighted towards 1000 μm
        pr = layered([0.05], [200.0], [263.0]; grain = [100.0], metamorphism = m)
        mix_refrozen_grain!(pr, 1, 1.0)
        @test pr.grain[1] ≈ 0.9 * 100 + 0.1 * 1000 rtol = 1e-12
        # merging and re-splitting layers conserves Σ m r
        pm = layered([0.004, 0.05], [100.0, 200.0], [263.0, 263.0]; grain = [60.0, 400.0])
        s0 = sum((pm.m_ice .+ pm.m_liq) .* pm.grain)
        SPm._merge_layers!(pm, 1, 2)
        relayer_snowpack!(pm)
        @test sum((pm.m_ice .+ pm.m_liq) .* pm.grain) ≈ s0 rtol = 1e-12
    end

    @testset "SnowOpticsAlbedo delegates to SurfaceOptics.SnowSurface" begin
        a = SnowOpticsAlbedo()
        p = layered([0.01, 0.01, 0.2], [150.0, 200.0, 250.0], fill(263.0, 3);
                    grain = [100.0, 300.0, 800.0], albedo = a)
        @test snowpack_optical_radius(p) ≈ 200.0 rtol = 1e-12            # top 2 cm, thickness-weighted
        @test surface_albedo(p) == broadband_albedo(SnowSurface(grain_radius = 200.0))
        # parabolic growth is SurfaceOptics' own ageing, made prognostic
        pg = LayeredSnowpack(depth = 0.3, density = 200.0, metamorphism = ParabolicGrainGrowth(),
                             albedo = SnowOpticsAlbedo())
        for _ in 1:7
            snow_metamorphism!(pg.metamorphism, pg, 86400.0)
        end
        @test surface_albedo(pg) ≈ broadband_albedo(SnowSurface(age_days = 7.0)) rtol = 1e-12
    end

    @testset "Fresh-snow albedo and the spectral signature of grain growth" begin
        spec = shortwave_bands()
        w = SOm._band_solar_weights(Float64, spec)
        vis = [b for b in 1:SOm.nbands(spec) if SOm.band_center_wavelength(spec, b) < 0.7]
        nir = setdiff(1:SOm.nbands(spec), vis)
        split(s) = (sum(w[vis] .* s.diffuse[vis]) / sum(w[vis]), sum(w[nir] .* s.diffuse[nir]) / sum(w[nir]))
        s0 = SnowSurface(grain_radius = 54.5)
        v0, n0 = split(s0)
        # reference fresh-snow broadband albedos: CLASS 0.84, VIC/ECMWF 0.85
        @test abs(broadband_albedo(s0) - 0.845) < 0.01
        # grain growth is a near-infrared process (Wiscombe & Warren 1980)
        v3, n3 = split(SnowSurface(grain_radius = 300.0))
        @test ((n0 - n3) / n0) / ((v0 - v3) / v0) > 5
    end

    @testset "Observed SSA decrease at Summit, Greenland (Carmagnola et al. 2014)" begin
        # 37 ± 3 → 23 ± 3 m² kg⁻¹ over 37 days at ≈ 325 kg m⁻³; temperature
        # and gradient assumed (−15 °C, 20 K m⁻¹)
        r0 = snow_optical_radius(37.0)
        ssa = snow_specific_surface_area(snicar_grain_radius(tab, r0, 37 * 24.0, Tf_m - 15, 20.0, 325.0))
        @test abs(ssa - 23) < 3
    end
end
