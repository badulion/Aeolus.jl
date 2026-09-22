using Test
using Aeolus
using Aeolus.RadiationCore
using Aeolus.SurfaceOptics
using Random

# Index of the band containing wavelength λ (μm).
function band_containing(spec, λ)
    ν = 1e4 / λ
    for b in 1:nbands(spec)
        lo, hi = band_limits(spec, b)
        lo <= ν <= hi && return b
    end
    error("no band contains λ = $λ μm")
end

@testset "Era III — Spectral albedo and BRDF (Phase 36)" begin

    sw = shortwave_bands()
    lw = longwave_bands()
    VIS = band_containing(sw, 0.55)    # green
    NIR = band_containing(sw, 0.90)    # NIR plateau
    SWIR = band_containing(sw, 1.50)   # ice absorption
    UV  = band_containing(sw, 0.30)

    # ---------------------------------------------------------------------
    # Band bookkeeping. Everything downstream is a weighted integral over
    # these bands, so if the wavelength mapping is wrong every number in
    # the module is wrong in a way no physical test would catch.
    # ---------------------------------------------------------------------
    @testset "Band bookkeeping" begin
        @test VIS == 10
        @test NIR == 8
        @test SWIR == 6
        for b in 1:nbands(sw)
            λlo, λhi = band_wavelength_limits(sw, b)
            νlo, νhi = band_limits(sw, b)
            @test λlo < λhi
            @test isapprox(λlo, 1e4 / νhi; rtol = 1e-12)
            @test isapprox(λhi, 1e4 / νlo; rtol = 1e-12)
            λc = band_center_wavelength(sw, b)
            @test λlo <= λc <= λhi
        end
        # the solar-weighted centre is pulled towards the short-wavelength
        # (high-irradiance) end of a wide band, not the arithmetic middle
        λlo, λhi = band_wavelength_limits(sw, NIR)
        @test band_center_wavelength(sw, NIR) < (λlo + λhi) / 2
        @info "Band 8 (NIR)" λlo λhi λc=band_center_wavelength(sw, NIR)
    end

    # ---------------------------------------------------------------------
    # Spectral reflectance interpolation.
    # ---------------------------------------------------------------------
    @testset "Spectral reflectance interpolation" begin
        @test spectral_reflectance(:grass, 0.55) ≈ 0.12
        @test spectral_reflectance(:grass, 0.68) ≈ 0.035
        @test spectral_reflectance(:grass, 0.90) ≈ 0.50
        # midpoint of a linear segment
        @test spectral_reflectance(:grass, 0.85) ≈ (0.48 + 0.50) / 2
        # clamped outside the tabulated range
        @test spectral_reflectance(:grass, 0.05) == spectral_reflectance(:grass, 0.30)
        @test spectral_reflectance(:grass, 100.0) == spectral_reflectance(:grass, 12.0)
        @test_throws ErrorException spectral_reflectance(:unobtainium, 0.5)
        # every material stays inside [0,1] everywhere
        for m in material_names(), λ in 0.2:0.05:3.0
            ρ = spectral_reflectance(m, λ)
            @test 0 <= ρ <= 1
        end
    end

    # ---------------------------------------------------------------------
    # The material library: broadband values against the literature.
    # ---------------------------------------------------------------------
    @testset "Material broadband albedos" begin
        bb = Dict(m => broadband_albedo(material_albedo(m, sw))
                  for m in material_names())
        @info "Broadband white-sky albedo" sort(collect(bb); by = last)
        # Oke (1987) Table 1.1 ranges
        @test 0.05 <= bb[:asphalt]         <= 0.15
        @test 0.20 <= bb[:concrete]        <= 0.40
        @test 0.20 <= bb[:brick]           <= 0.35
        @test 0.10 <= bb[:roof_tile_grey]  <= 0.25
        @test 0.45 <= bb[:metal_roof]      <= 0.70
        @test 0.15 <= bb[:soil_dry]        <= 0.35
        @test 0.30 <= bb[:sand]            <= 0.45
        @test 0.15 <= bb[:grass]           <= 0.26
        @test 0.10 <= bb[:tree_deciduous]  <= 0.25
        @test 0.05 <= bb[:tree_coniferous] <= 0.18
        @test 0.02 <= bb[:water]           <= 0.10
        @test 0.75 <= bb[:snow_fresh]      <= 0.95
        @test 0.50 <= bb[:snow_old]        <= 0.75
        # ordering relations that must hold whatever the exact numbers
        @test bb[:soil_wet] < bb[:soil_dry]          # wetting darkens soil
        @test bb[:snow_old] < bb[:snow_fresh]
        @test bb[:tree_coniferous] < bb[:tree_deciduous] < bb[:grass]
        @test bb[:asphalt] < bb[:concrete]
        @test bb[:water] == minimum(values(bb))
        @test bb[:snow_fresh] == maximum(values(bb))
        @test_throws ErrorException material_albedo(:unobtainium, sw)
    end

    # ---------------------------------------------------------------------
    # The vegetation red edge. This is the single most distinctive feature
    # of the terrestrial land surface: chlorophyll absorbs almost all
    # visible light and the leaf's internal air-water interfaces scatter
    # almost all near-infrared light, so the reflectance jumps by an order
    # of magnitude across 40 nm. A grey albedo destroys it entirely, and
    # with it any hope of a correct photosynthetically-active-radiation
    # budget or a correct NIR heating profile in the canopy.
    # ---------------------------------------------------------------------
    @testset "Vegetation red edge" begin
        for m in (:grass, :tree_deciduous, :tree_coniferous)
            a = material_albedo(m, sw)
            ratio = a.diffuse[NIR] / a.diffuse[VIS]
            @info "Red edge" material=m vis=a.diffuse[VIS] nir=a.diffuse[NIR] ratio
            @test ratio >= 5.0
            @test a.diffuse[VIS] < 0.10          # dark in the visible
            @test a.diffuse[NIR] > 0.25          # bright in the NIR
        end
        # the edge itself: reflectance climbs steeply between 0.68 and 0.78
        for m in (:grass, :tree_deciduous, :tree_coniferous)
            @test spectral_reflectance(m, 0.78) > 6 * spectral_reflectance(m, 0.68)
        end
        # the green bump: grass reflects more at 0.55 than at 0.65 μm,
        # which is why it looks green
        @test spectral_reflectance(:grass, 0.55) > spectral_reflectance(:grass, 0.65)
        # non-vegetated surfaces have no red edge
        for m in (:asphalt, :concrete, :metal_roof)
            a = material_albedo(m, sw)
            @test a.diffuse[NIR] / a.diffuse[VIS] < 2.0
        end
    end

    # ---------------------------------------------------------------------
    # Snow spectrum: bright in the visible, dark in the shortwave infrared.
    # ---------------------------------------------------------------------
    @testset "Snow spectral shape" begin
        sn = material_albedo(:snow_fresh, sw)
        @info "Fresh snow spectral albedo" uv=sn.diffuse[UV] vis=sn.diffuse[VIS] nir=sn.diffuse[NIR] swir_1p5=sn.diffuse[SWIR] band4_2p0=sn.diffuse[4] band3_2p3=sn.diffuse[3]
        @test sn.diffuse[VIS] > 0.90
        @test sn.diffuse[UV]  > 0.90
        @test 0.55 <= sn.diffuse[SWIR] <= 0.68      # ~0.6 at 1.5 μm
        @test sn.diffuse[4] < 0.30                  # 1.94-2.15 μm
        @test sn.diffuse[3] < sn.diffuse[4]         # still darker beyond
        # monotone decline from the visible into the infrared
        @test sn.diffuse[VIS] > sn.diffuse[NIR] > sn.diffuse[SWIR] >
              sn.diffuse[5] > sn.diffuse[4] > sn.diffuse[3]
        old = material_albedo(:snow_old, sw)
        @test old.diffuse[VIS] < sn.diffuse[VIS]
        @test old.diffuse[SWIR] < sn.diffuse[SWIR]
        # ageing costs far more in the NIR than in the visible — the
        # signature of grain growth rather than of contamination
        @test (sn.diffuse[VIS] - old.diffuse[VIS]) <
              (sn.diffuse[SWIR] - old.diffuse[SWIR])
    end

    # ---------------------------------------------------------------------
    # Longwave emissivity, and the metal-roof outlier.
    # ---------------------------------------------------------------------
    @testset "Longwave emissivity" begin
        ε = Dict(m => material_emissivity(m) for m in material_names())
        @info "Broadband emissivity" sort(collect(ε); by = last)
        @test ε[:asphalt]         ≈ 0.95
        @test ε[:concrete]        ≈ 0.91
        @test ε[:glass]           ≈ 0.94
        @test ε[:water]           ≈ 0.96
        @test ε[:snow_fresh]      ≈ 0.99
        @test ε[:grass]           ≈ 0.98
        @test ε[:tree_deciduous]  ≈ 0.98
        @test ε[:tree_coniferous] ≈ 0.98
        @test ε[:metal_roof]      ≈ 0.25
        # the outlier: bare metal radiates at a quarter of the efficiency
        # of everything else in a city, which is why a metal roof runs hot
        others = [v for (k, v) in ε if k !== :metal_roof]
        @test ε[:metal_roof] < minimum(others) / 3
        @test all(0.85 .<= others .<= 1.0)
        # ... and that shows up as a huge difference in equilibrium
        # radiating temperature for the same net absorbed flux
        F = 600.0                                   # W m⁻² absorbed
        T_metal = (F / (ε[:metal_roof] * STEFAN_BOLTZMANN))^0.25
        T_asph  = (F / (ε[:asphalt]    * STEFAN_BOLTZMANN))^0.25
        @info "Radiative equilibrium temperature at 600 W/m²" T_metal T_asph
        @test T_metal - T_asph > 100.0

        # quartz sand carries the silicate reststrahlen band: emissivity
        # dips near 9 μm, so its band emissivities are not grey
        s = material_albedo(:sand, sw)
        εb = emissivity_vector(s)
        @test length(εb) == nbands(lw)
        @test minimum(εb) < 0.85
        @test maximum(εb) > 0.92
        b9 = band_containing(lw, 9.0)
        @test εb[b9] == minimum(εb)
        @info "Sand band emissivity (reststrahlen)" min=minimum(εb) max=maximum(εb) band9=εb[b9]
        # grey materials really are grey
        εa = emissivity_vector(material_albedo(:asphalt, sw))
        @test all(εa .≈ 0.95)
    end

    # ---------------------------------------------------------------------
    # Solar-zenith dependence of the direct-beam albedo, and the
    # consistency constraint between direct and diffuse.
    # ---------------------------------------------------------------------
    @testset "Direct-beam zenith dependence" begin
        for m in (:grass, :soil_dry, :asphalt, :snow_fresh)
            a = material_albedo(m, sw)
            αs = [surface_albedo(a, VIS, μ) for μ in (1.0, 0.8, 0.5, 0.3, 0.2)]
            @info "Albedo vs μ₀" material=m αs
            @test issorted(αs)                   # rises as the sun drops
            @test all(0 .<= αs .<= 1)
            @test surface_albedo(a, VIS, 1.0) < surface_albedo(a, VIS)
            # white-sky albedo is the cosine-weighted mean of black-sky:
            # this is a definition, and enforcing it is what keeps a model
            # from jumping when the sky goes from clear to overcast
            acc = 0.0
            n = 200
            for i in 1:n
                μ = (i - 0.5) / n
                acc += 2 * μ * surface_albedo(a, VIS, μ) / n
            end
            @test isapprox(acc, surface_albedo(a, VIS); rtol = 2e-3)
        end
        # vegetation has a stronger zenith dependence than asphalt
        g = material_albedo(:grass, sw); p = material_albedo(:asphalt, sw)
        rg = surface_albedo(g, VIS, 0.2) / surface_albedo(g, VIS, 1.0)
        rp = surface_albedo(p, VIS, 0.2) / surface_albedo(p, VIS, 1.0)
        @info "Zenith sensitivity ratio α(μ=0.2)/α(μ=1)" grass=rg asphalt=rp
        @test rg > rp > 1.0
    end

    # =====================================================================
    # BRDF models
    # =====================================================================

    # Every model and every shipped parameter set, in one list, so the
    # structural properties below are checked exhaustively rather than on
    # a convenient sample.
    all_models = Any[("Lambertian-0.05", LambertianBRDF(0.05)),
                     ("Lambertian-0.30", LambertianBRDF(0.30)),
                     ("Lambertian-0.90", LambertianBRDF(0.90))]
    for p in (:grass, :forest, :soil, :sand, :snow, :urban)
        push!(all_models, ("RPV-$p", rpv_preset(p)))
    end
    for p in (:grass, :forest, :soil, :sand, :snow, :urban)
        push!(all_models, ("RossLi-$p", rossli_preset(p)))
    end
    for p in (:glass, :metal_roof, :polished_metal, :wet_asphalt)
        push!(all_models, ("CookTorrance-$p", cook_torrance_preset(p)))
    end

    # ---------------------------------------------------------------------
    # Reciprocity. Helmholtz reciprocity follows from time-reversal
    # symmetry of Maxwell's equations and holds for any passive surface;
    # a BRDF that violates it is not merely inaccurate, it is
    # thermodynamically impossible, and it will pump energy around a
    # multiple-reflection solver such as the urban radiosity scheme.
    #
    # The Li-sparse-*reciprocal* kernel is the variant that satisfies it;
    # the non-reciprocal variant still in circulation does not, and the
    # difference is a single `sec θ_i` factor in the last term.
    # ---------------------------------------------------------------------
    @testset "Reciprocity" begin
        Random.seed!(20240936)
        worst_overall = 0.0
        for (nm, m) in all_models
            worst = 0.0
            for _ in 1:2000
                θi = rand() * 1.45; θv = rand() * 1.45
                φi = rand() * 2π;   φv = rand() * 2π
                f1 = brdf(m, θi, φi, θv, φv)
                f2 = brdf(m, θv, φv, θi, φi)
                worst = max(worst, abs(f1 - f2) / max(abs(f1), 1e-12))
            end
            @test worst < 1e-10
            worst_overall = max(worst_overall, worst)
        end
        @info "Reciprocity: worst relative asymmetry over 17 models × 2000 geometries" worst_overall
        # exact-geometry spot checks, including the principal plane
        rl = rossli_preset(:forest)
        @test brdf(rl, 0.4, 0.0, 0.9, π) == brdf(rl, 0.9, π, 0.4, 0.0)
        ct = cook_torrance_preset(:glass)
        @test brdf(ct, 0.2, 1.0, 1.3, 2.0) ≈ brdf(ct, 1.3, 2.0, 0.2, 1.0)
        rp = rpv_preset(:soil)
        @test brdf(rp, 0.7, 0.3, 0.1, 5.0) ≈ brdf(rp, 0.1, 5.0, 0.7, 0.3)
    end

    # ---------------------------------------------------------------------
    # The Lambertian limit. This is the calibration of the quadrature:
    # a Lambertian BRDF of albedo a has BRF = a in every direction, so
    # both hemispherical integrals must return exactly a. Any quadrature
    # error shows up here undisguised, and no other kernel's albedo can be
    # trusted further than this test passes.
    # ---------------------------------------------------------------------
    @testset "Lambertian limit (quadrature calibration)" begin
        for a in (0.05, 0.3, 0.9)
            L = LambertianBRDF(a)
            for θ in (0.0, 0.3, 0.7, 1.1, 1.4)
                # the generic hemisphere quadrature, not the analytic
                # shortcut — this is what is being calibrated
                @test isapprox(hemisphere_quadrature_albedo(L, θ; nμ = 40, nφ = 72),
                               a; atol = 1e-12)
            end
            @test isapprox(black_sky_albedo(L, 0.5), a; atol = 1e-12)
            @test isapprox(white_sky_albedo(L), a; atol = 1e-12)
            @test isapprox(blue_sky_albedo(L, 0.5, 0.3), a; atol = 1e-12)
        end
        # ... and a Lambertian result built out of the generic path, so
        # the double integral itself is exercised
        L = LambertianBRDF(0.42)
        gen = 2 * sum(begin
            x = (i - 0.5) / 400
            2 * x * hemisphere_quadrature_albedo(L, acos(x); nμ = 24, nφ = 32) / 400
        end for i in 1:400) / 2
        @info "Lambertian double integral" nominal=0.42 measured=gen
        @test isapprox(gen, 0.42; atol = 1e-6)
    end

    # ---------------------------------------------------------------------
    # Independent validation against MODIS. The MCD43 algorithm's kernel
    # integrals are published constants; reproducing them from nothing but
    # the kernel formulae and a quadrature is a far stronger check than
    # any self-consistency test, because it validates the kernels and the
    # integrator against an operational implementation written by other
    # people from the same equations.
    # ---------------------------------------------------------------------
    @testset "MODIS kernel integrals" begin
        kvol = RossLiBRDF(0.0, 1.0, 0.0; clamp_negative = false)
        kgeo = RossLiBRDF(0.0, 0.0, 1.0; clamp_negative = false)
        wv = white_sky_albedo(kvol; nμi = 64)
        wg = white_sky_albedo(kgeo; nμi = 64)
        @info "White-sky kernel integrals" ross_thick=wv published_vol=0.189184 li_sparse_r=wg published_geo=-1.377622
        @test isapprox(wv,  0.189184; atol = 1e-4)
        @test isapprox(wg, -1.377622; atol = 1e-3)
        # the kernels' defining normalisations
        @test isapprox(ross_thick_kernel(0.0, 0.0, 0.0), 0.0; atol = 1e-12)
        @test li_sparse_kernel(0.0, 0.0, 0.0) ≈ 0.0 atol = 1e-12
        # Ross-thick peaks at backscatter for any fixed pair of angles
        θ = 0.6
        @test ross_thick_kernel(θ, θ, 0.0) > ross_thick_kernel(θ, θ, π / 2)
        @test ross_thick_kernel(θ, θ, π / 2) > ross_thick_kernel(θ, θ, π)
        # Li-sparse is strongly negative in the forward direction — the
        # reason the linear model can extrapolate below zero
        @test li_sparse_kernel(θ, θ, π) < -0.5
        @test li_sparse_kernel(θ, θ, 0.0) > 0.0
        # the crown-shape parameters actually enter
        @test li_sparse_kernel(0.5, 0.8, 1.0; br = 2.0) !=
              li_sparse_kernel(0.5, 0.8, 1.0; br = 1.0)
        # with f_iso alone the model is Lambertian
        iso = RossLiBRDF(0.3, 0.0, 0.0)
        @test isapprox(black_sky_albedo(iso, 0.7), 0.3; atol = 1e-10)
        @test isapprox(white_sky_albedo(iso), 0.3; atol = 1e-10)
    end

    # ---------------------------------------------------------------------
    # Energy conservation. A surface may not reflect more than it
    # receives. The semi-empirical kernels are not constructed to
    # guarantee this — RPV's Minnaert factor and Li-sparse's sec θ both
    # diverge at the horizon — so it is a property of the *parameter sets*
    # shipped here and has to be measured, not assumed.
    # ---------------------------------------------------------------------
    @testset "Energy conservation" begin
        worst = (0.0, "", 0.0)
        for (nm, m) in all_models
            for d in 0.0:2.5:85.0
                α = black_sky_albedo(m, deg2rad(d))
                @test 0.0 <= α <= 1.0
                α > worst[1] && (worst = (α, nm, d))
            end
            w = white_sky_albedo(m)
            @test 0.0 <= w <= 1.0
        end
        @info "Energy conservation: largest black-sky albedo over 0-85° incidence" max_alpha=worst[1] model=worst[2] zenith_deg=worst[3]
        # Beyond ~87° the published kernels genuinely diverge. Recorded
        # here so that the limitation is documented by measurement rather
        # than by assertion — this is the model's property, not a bug.
        div_rpv = black_sky_albedo(rpv_preset(:forest), deg2rad(89.5))
        div_rl  = black_sky_albedo(rossli_preset(:urban), deg2rad(89.5))
        @info "Kernel divergence at 89.5° incidence (documented limitation)" rpv_forest=div_rpv rossli_urban=div_rl
        @test div_rpv > 1.0
        @test div_rl  > 1.0
        # the surface-optics wrappers, which is what a solver sees, clamp
        for p in (:grass, :urban)
            s = BRDFSurface(rossli_preset(p), sw; material = :grass)
            for μ in (1.0, 0.5, 0.1, 0.0)
                @test 0.0 <= surface_albedo(s, VIS, μ) <= 1.0
            end
        end
    end

    # ---------------------------------------------------------------------
    # The hot spot. Looking exactly down the illumination direction, every
    # shadow is hidden behind the object that casts it, so the surface
    # appears brighter than in any neighbouring direction. It is the
    # sharpest feature of a real reflectance field and the one that most
    # distinguishes a physical BRDF from a fitted lobe.
    # ---------------------------------------------------------------------
    @testset "Hot spot" begin
        θi = deg2rad(35)
        for p in (:grass, :forest, :soil, :sand, :snow, :urban)
            for (nm, m) in (("RPV", rpv_preset(p)), ("RossLi", rossli_preset(p)))
                hs = brdf(m, θi, 0.0, θi, 0.0)
                # step away in zenith, and in azimuth, along the principal
                # plane and out of it
                off = [brdf(m, θi, 0.0, θi + deg2rad(d), 0.0) for d in (2, 5, 10, 20)]
                azi = [brdf(m, θi, 0.0, θi, deg2rad(d))       for d in (2, 5, 10, 20)]
                @test all(hs .> off)
                @test all(hs .> azi)
                @test issorted(off; rev = true)
                @test issorted(azi; rev = true)
                if p === :grass
                    @info "Hot spot" model=nm preset=p backscatter=hs off_zenith=off off_azimuth=azi
                end
            end
        end
        # the enhancement is a real percentage, not numerical noise
        m = rpv_preset(:soil)
        hs = brdf(m, θi, 0.0, θi, 0.0)
        far = brdf(m, θi, 0.0, θi, π)
        @info "Hot-spot enhancement, RPV soil" backscatter=hs forward=far ratio=hs/far
        @test hs / far > 1.15
        # the RPV hot-spot factor has the documented amplitude 2 - ρ0 at
        # G = 0 relative to its G → ∞ limit, so a brighter surface has a
        # weaker opposition effect (fewer visible shadows to hide)
        dark   = RPVBRDF(0.05, 1.0, 0.0)
        bright = RPVBRDF(0.80, 1.0, 0.0)
        rdark   = brdf(dark, θi, 0.0, θi, 0.0)   / dark.ρ0
        rbright = brdf(bright, θi, 0.0, θi, 0.0) / bright.ρ0
        @test rdark > rbright
        @test isapprox(rdark, 2 - 0.05; atol = 1e-12)
        @test isapprox(rbright, 2 - 0.80; atol = 1e-12)
    end

    # ---------------------------------------------------------------------
    # Cook-Torrance. The microfacet model is the one with a sharp lobe, so
    # it is also the one whose integration is easiest to get quietly
    # wrong: a hemisphere quadrature that cannot resolve a lobe of width α
    # *under*-estimates the albedo and still passes an energy test.
    # ---------------------------------------------------------------------
    @testset "Cook-Torrance microfacet" begin
        # specialised half-vector quadrature against brute force, at
        # roughnesses where the brute-force rule is still trustworthy
        for α in (0.5, 0.3, 0.2), d in (0.0, 40.0, 70.0)
            m = CookTorranceBRDF(0.05, 0.3, α)
            θ = deg2rad(d)
            a = black_sky_albedo(m, θ)
            b = hemisphere_quadrature_albedo(m, θ; nμ = 400, nφ = 800)
            @test isapprox(a, b; rtol = 3e-3)
        end
        m = CookTorranceBRDF(0.05, 0.3, 0.3)
        @info "Cook-Torrance α=0.3: specialised vs brute-force BSA" specialised=black_sky_albedo(m, 0.7) brute=hemisphere_quadrature_albedo(m, 0.7; nμ = 400, nφ = 800)

        # a perfect mirror reflects everything: with F0 = 1 and ρd = 0 the
        # albedo must approach 1 as the surface becomes smooth. The small
        # deficit is the Smith shadowing term, which is energy-losing by
        # construction for a single-scattering microfacet model.
        for α in (0.1, 0.05, 0.02, 0.01)
            mm = CookTorranceBRDF(0.0, 1.0, α)
            a0 = black_sky_albedo(mm, 0.0)
            a6 = black_sky_albedo(mm, deg2rad(60))
            @test a0 <= 1.0 && a6 <= 1.0
            α <= 0.02 && (@test a0 > 0.999)
        end
        @info "Mirror limit (F0=1, ρd=0)" a_alpha0p1=black_sky_albedo(CookTorranceBRDF(0.0,1.0,0.1),0.0) a_alpha0p01=black_sky_albedo(CookTorranceBRDF(0.0,1.0,0.01),0.0)

        # Fresnel behaviour: a glazed surface is dark at normal incidence
        # and mirror-bright at grazing, which is the whole reason a façade
        # has to be modelled as specular
        g = cook_torrance_preset(:glass)
        αs = [black_sky_albedo(g, deg2rad(d)) for d in (0, 30, 60, 75, 85)]
        @info "Glass black-sky albedo vs zenith (0,30,60,75,85°)" αs
        @test issorted(αs)
        @test αs[1] < 0.10
        @test αs[end] > 4 * αs[1]
        # roughening the same material spreads the lobe but keeps the
        # energy: the white-sky albedo is only weakly affected
        smooth = CookTorranceBRDF(0.02, 0.043, 0.02)
        rough  = CookTorranceBRDF(0.02, 0.043, 0.30)
        @test isapprox(white_sky_albedo(smooth), white_sky_albedo(rough); rtol = 0.5)
        @test white_sky_albedo(smooth) > 0.0
        # the diffuse floor is exactly additive
        base = CookTorranceBRDF(0.0, 0.3, 0.2)
        plus = CookTorranceBRDF(0.2, 0.3, 0.2)
        @test isapprox(black_sky_albedo(plus, 0.6) - black_sky_albedo(base, 0.6),
                       0.2; atol = 1e-10)
        # construction guards
        @test_throws ErrorException CookTorranceBRDF(0.1, 0.1, -0.1)
        @test_throws ErrorException CookTorranceBRDF(1.5, 0.1, 0.1)
        @test_throws ErrorException RPVBRDF(0.1, 0.5, 1.2)
        @test_throws ErrorException RPVBRDF(0.1, -0.5, 0.1)
        @test_throws ErrorException rpv_preset(:unobtainium)
        @test_throws ErrorException cook_torrance_preset(:unobtainium)
        @test_throws ErrorException rossli_preset(:unobtainium)
    end

    # ---------------------------------------------------------------------
    # Blue-sky albedo: the operational mixture.
    # ---------------------------------------------------------------------
    @testset "Blue-sky albedo" begin
        m = rpv_preset(:grass)
        θ = deg2rad(40)
        bs = black_sky_albedo(m, θ)
        ws = white_sky_albedo(m)
        @test isapprox(blue_sky_albedo(m, θ, 0.0), bs; rtol = 1e-12)
        @test isapprox(blue_sky_albedo(m, θ, 1.0), ws; rtol = 1e-12)
        for f in (0.1, 0.3, 0.5, 0.9)
            b = blue_sky_albedo(m, θ, f)
            @test isapprox(b, (1 - f) * bs + f * ws; rtol = 1e-12)
            @test min(bs, ws) <= b <= max(bs, ws)
        end
        @info "Blue-sky albedo, RPV grass at 40°" black_sky=bs white_sky=ws f0p2=blue_sky_albedo(m, θ, 0.2) f0p8=blue_sky_albedo(m, θ, 0.8)
        # out-of-range diffuse fractions are clamped, not extrapolated
        @test blue_sky_albedo(m, θ, -1.0) == blue_sky_albedo(m, θ, 0.0)
        @test blue_sky_albedo(m, θ, 2.0)  == blue_sky_albedo(m, θ, 1.0)
    end

    # =====================================================================
    # Fresnel water with Cox-Munk wave slopes
    # =====================================================================
    @testset "Fresnel water surface" begin
        # flat-surface Fresnel against the closed-form limits
        n = 1.34
        @test isapprox(fresnel_reflectance(0.0, n), ((n - 1) / (n + 1))^2; atol = 1e-12)
        @test isapprox(fresnel_reflectance(0.0, n), 0.02110; atol = 1e-4)
        @test fresnel_reflectance(π / 2, n) ≈ 1.0 atol = 1e-9
        Rs = [fresnel_reflectance(deg2rad(d), n) for d in 0:10:89]
        @test issorted(Rs)
        @test all(0 .<= Rs .<= 1)
        @info "Flat-water Fresnel reflectance (0,20,40,60,70,80,85°)" [round(fresnel_reflectance(deg2rad(d), n); digits = 4) for d in (0, 20, 40, 60, 70, 80, 85)]
        @test isapprox(fresnel_reflectance(deg2rad(80), n), 0.350; atol = 0.01)
        # at the Brewster angle the parallel component vanishes, so the
        # reflectance is exactly half the perpendicular component
        θB = atan(n)
        c = cos(θB); st = sin(θB) / n; ct = sqrt(1 - st^2)
        rs = ((c - n * ct) / (c + n * ct))^2
        @test isapprox(fresnel_reflectance(θB, n), rs / 2; atol = 1e-10)

        # Cox-Munk slope variance
        @test cox_munk_slope_variance(0.0) ≈ 0.003
        @test cox_munk_slope_variance(10.0) ≈ 0.003 + 0.0512
        @test cox_munk_slope_variance(5.0) > cox_munk_slope_variance(0.0)
        @test whitecap_fraction(0.0) == 0.0
        @test whitecap_fraction(5.0) < 0.01
        @test whitecap_fraction(20.0) > 0.02
        @test whitecap_fraction(5.0) < whitecap_fraction(15.0)

        w5 = FresnelWater(sw; wind_speed = 5.0)
        # calm water reduces to the flat Fresnel curve at normal incidence
        @test isapprox(water_albedo_direct(FresnelWater(sw; wind_speed = 0.0), 0.0),
                       fresnel_reflectance(0.0, n); atol = 2e-3)
        αz = [surface_albedo(w5, VIS, cos(deg2rad(d))) for d in (0, 20, 40, 60, 70, 80)]
        @info "Water albedo vs solar zenith (U10 = 5 m/s, visible band)" αz
        @test issorted(αz)
        @test 0.02 <= αz[1] <= 0.06              # ≈0.03-0.04 overhead
        @test αz[end] > 0.20                     # >0.2 at 80°
        @test αz[end] > 5 * αz[1]                # several times larger
        bbz = [broadband_albedo(w5, cos(deg2rad(d))) for d in (0, 60, 80)]
        @info "Water broadband albedo (0°, 60°, 80°)" bbz
        @test isapprox(bbz[1], 0.03; atol = 0.012)
        @test bbz[3] > 0.2

        # wind reduces the grazing-angle albedo: tilted facets present a
        # smaller local incidence angle to the beam, moving them down the
        # steeply rising Fresnel curve. This is the counter-intuitive
        # result that distinguishes a Cox-Munk surface from a flat one.
        αU = [water_albedo_direct(FresnelWater(sw; wind_speed = U), deg2rad(80))
              for U in (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)]
        @info "Specular water albedo at 80° zenith vs wind speed (0-20 m/s)" αU
        @test issorted(αU; rev = true)
        @test αU[end] < 0.7 * αU[1]
        # ... while at high sun it barely matters (and the whitecaps then
        # push the total slightly the other way)
        αU0 = [water_albedo_direct(FresnelWater(sw; wind_speed = U), 0.0)
               for U in (0.0, 5.0, 20.0)]
        @test maximum(αU0) - minimum(αU0) < 0.005

        # diffuse (white-sky) water albedo is much larger than the
        # overhead-sun value — an isotropic sky delivers most of its
        # energy at large zenith angles
        wd = water_albedo_diffuse(w5)
        @info "Water white-sky specular albedo" wd overhead=water_albedo_direct(w5, 0.0)
        @test 0.04 <= wd <= 0.09
        @test wd > 2 * water_albedo_direct(w5, 0.0)
        @test surface_albedo(w5, VIS) > surface_albedo(w5, VIS, 1.0)
        # water-leaving reflectance is a visible-band phenomenon
        @test surface_albedo(w5, VIS, 1.0) > surface_albedo(w5, NIR, 1.0)
        @test all(surface_emissivity(w5, b) ≈ 0.96 for b in 1:nbands(lw))
        @test_throws ErrorException FresnelWater(sw; n_water = 0.9)
    end

    # =====================================================================
    # Snow: grain size, ageing, impurities
    # =====================================================================
    @testset "Snow albedo" begin
        fresh = SnowSurface(sw)
        @test fresh.effective_radius ≈ 50.0
        @info "Fresh snow spectral albedo" vis=fresh.diffuse[VIS] nir=fresh.diffuse[NIR] swir=fresh.diffuse[SWIR] broadband=broadband_albedo(fresh)
        @test fresh.diffuse[VIS] > 0.90
        @test broadband_albedo(fresh) > 0.80

        # --- grain size ---------------------------------------------------
        radii = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)
        vis  = Float64[]; nir = Float64[]; bb = Float64[]
        for r in radii
            s = SnowSurface(sw; grain_radius = r)
            push!(vis, s.diffuse[VIS]); push!(nir, s.diffuse[NIR])
            push!(bb, broadband_albedo(s))
        end
        @info "Snow albedo vs grain radius (50-2000 μm)" radii visible=vis nir broadband=bb
        @test issorted(vis; rev = true)
        @test issorted(nir; rev = true)
        @test issorted(bb; rev = true)
        @test all(0 .<= vis .<= 1)
        # grain growth costs far more in the near-infrared than in the
        # visible, because ice is a weak absorber at 0.5 μm and a strong
        # one at 1 μm — this asymmetry is the whole physical content
        @test (vis[1] - vis[end]) < 0.25
        @test (nir[1] - nir[end]) > 0.35
        # coarse old snow is still bright in the visible
        @test vis[end] > 0.80

        # --- ageing -------------------------------------------------------
        ages = (0.0, 1.0, 3.0, 7.0, 15.0, 30.0)
        aged = [SnowSurface(sw; grain_radius = 50.0, age_days = t) for t in ages]
        av = [s.diffuse[VIS] for s in aged]
        ab = [broadband_albedo(s) for s in aged]
        ar = [s.effective_radius for s in aged]
        @info "Snow albedo vs age (0-30 days)" ages r_eff=ar visible=av broadband=ab
        @test issorted(av; rev = true)
        @test issorted(ab; rev = true)
        @test issorted(ar)
        @test ab[1] - ab[end] > 0.10      # a month of metamorphism costs >0.1
        # parabolic grain growth
        @test isapprox(snow_grain_radius(50.0, 0.0), 50.0; atol = 1e-12)
        @test isapprox(snow_grain_radius(50.0, 1.0)^2 - 50.0^2,
                       snow_grain_radius(50.0, 2.0)^2 - snow_grain_radius(50.0, 1.0)^2;
                       rtol = 1e-12)
        @test_throws ErrorException snow_grain_radius(50.0, -1.0)
        @test_throws ErrorException snow_grain_radius(0.0, 1.0)

        # --- soot ---------------------------------------------------------
        soots = (0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0)
        sv = Float64[]; sn_ = Float64[]
        for c in soots
            s = SnowSurface(sw; soot_ppmw = c)
            push!(sv, s.diffuse[VIS]); push!(sn_, s.diffuse[NIR])
        end
        @info "Snow albedo vs soot loading (ppmw)" soots visible=sv nir=sn_
        @test issorted(sv; rev = true)
        @test issorted(sn_; rev = true)
        # calibration against Wiscombe & Warren (1980) Fig. 8: soot is a
        # visible-band poison and does almost nothing in the NIR
        @test isapprox(sv[3], 0.94; atol = 0.02)     # 0.1 ppmw
        @test isapprox(sv[5], 0.83; atol = 0.03)     # 1   ppmw
        @test isapprox(sv[7], 0.57; atol = 0.05)     # 10  ppmw
        @test (sv[1] - sv[5]) > 4 * (sn_[1] - sn_[5])
        @test_throws ErrorException SnowSurface(sw; soot_ppmw = -1.0)

        # grain size and soot compound: dirty coarse snow is much darker
        # than either effect alone would give
        dirty_coarse = SnowSurface(sw; grain_radius = 1000.0, soot_ppmw = 1.0)
        @test dirty_coarse.diffuse[VIS] < SnowSurface(sw; grain_radius = 1000.0).diffuse[VIS]
        @test dirty_coarse.diffuse[VIS] < SnowSurface(sw; soot_ppmw = 1.0).diffuse[VIS]
        @info "Dirty coarse snow" visible=dirty_coarse.diffuse[VIS] broadband=broadband_albedo(dirty_coarse)

        # --- solar zenith dependence --------------------------------------
        μs = (1.0, 0.8, 0.5, 0.3, 0.1, 0.0)
        az = [snow_albedo(fresh, VIS, μ) for μ in μs]
        @info "Snow albedo vs μ₀ (visible band)" μs az
        @test issorted(az)
        @test all(0 .< az .<= 1)             # bounded, unlike 1/μ₀ forms
        @test az[1] < fresh.diffuse[VIS] < az[end]
        # the escape-function exponent integrates to 1, so the direct
        # albedo averages back to the diffuse albedo
        acc = 0.0; N = 400
        for i in 1:N
            μ = (i - 0.5) / N
            acc += 2 * μ * snow_albedo(fresh, VIS, μ) / N
        end
        @info "Snow hemispheric consistency" integrated=acc diffuse=fresh.diffuse[VIS]
        @test isapprox(acc, fresh.diffuse[VIS]; rtol = 3e-3)
        @test length(snow_albedo_spectrum(fresh)) == nbands(sw)
        @test snow_albedo_spectrum(fresh; μ₀ = 0.5) != snow_albedo_spectrum(fresh)
        @test all(surface_emissivity(fresh, b) ≈ 0.99 for b in 1:nbands(lw))
        @test_throws ErrorException snow_albedo(fresh, 99, 0.5)
    end

    # =====================================================================
    # Tiled (mosaic) surfaces
    # =====================================================================
    @testset "Tiled surface" begin
        cell = tiled_surface(sw, :asphalt => 0.35, :concrete => 0.25,
                                 :grass => 0.25, :tree_deciduous => 0.15)
        @test length(tile_components(cell)) == 4
        @test sum(tile_fractions(cell)) ≈ 1.0
        for b in (VIS, NIR, SWIR, UV)
            comp = [surface_albedo(c, b) for c in tile_components(cell)]
            α = surface_albedo(cell, b)
            @test minimum(comp) <= α <= maximum(comp)
            # exactly the area-weighted mean
            @test isapprox(α, sum(tile_fractions(cell) .* comp); atol = 1e-14)
        end
        for μ in (1.0, 0.5, 0.2)
            comp = [surface_albedo(c, VIS, μ) for c in tile_components(cell)]
            α = surface_albedo(cell, VIS, μ)
            @test minimum(comp) <= α <= maximum(comp)
            @test isapprox(α, sum(tile_fractions(cell) .* comp); atol = 1e-14)
        end
        εc = [surface_emissivity(c, 7) for c in tile_components(cell)]
        @test isapprox(surface_emissivity(cell, 7),
                       sum(tile_fractions(cell) .* εc); atol = 1e-14)
        @test minimum(εc) <= surface_emissivity(cell, 7) <= maximum(εc)
        @info "Urban tile (35% asphalt, 25% concrete, 25% grass, 15% trees)" broadband_white_sky=broadband_albedo(cell) broadband_direct_mu0p8=broadband_albedo(cell, 0.8) emissivity=surface_emissivity(cell, 7)
        @test 0.10 <= broadband_albedo(cell) <= 0.30

        # a single-component mosaic reproduces its component exactly
        solo = tiled_surface(sw, :grass => 1.0)
        g = material_albedo(:grass, sw)
        @test surface_albedo(solo, VIS) ≈ surface_albedo(g, VIS)
        @test surface_albedo(solo, VIS, 0.4) ≈ surface_albedo(g, VIS, 0.4)

        # a metal roof drags the cell emissivity down far more than its
        # area share of any other material would — the practical reason
        # metal-roofed districts show a strong nocturnal heat-island signal
        plain = tiled_surface(sw, :concrete => 0.8, :grass => 0.2)
        metal = tiled_surface(sw, :concrete => 0.6, :metal_roof => 0.2, :grass => 0.2)
        @info "Cell emissivity with and without a 20% metal roof" without=surface_emissivity(plain, 7) with=surface_emissivity(metal, 7)
        @test surface_emissivity(plain, 7) - surface_emissivity(metal, 7) > 0.10

        # mosaics nest, and mixed component types are allowed
        mixed = TiledSurface([material_albedo(:asphalt, sw),
                              FresnelWater(sw; wind_speed = 4.0),
                              SnowSurface(sw; grain_radius = 200.0),
                              cell], [0.25, 0.25, 0.25, 0.25])
        @test 0.0 <= surface_albedo(mixed, VIS, 0.7) <= 1.0
        @test surface_albedo(mixed, VIS) > surface_albedo(cell, VIS)  # snow
        @test length(diffuse_albedo_vector(mixed)) == nbands(sw)

        # fractions that do not close are an error, not a silent renormalise
        @test_throws ErrorException tiled_surface(sw, :grass => 0.5, :asphalt => 0.4)
        @test_throws ErrorException TiledSurface([material_albedo(:grass, sw)],
                                                 [0.5, 0.5])
        @test_throws ErrorException TiledSurface([material_albedo(:grass, sw),
                                                  material_albedo(:asphalt, sw)],
                                                 [1.2, -0.2])
    end

    # =====================================================================
    # Contract integration: these objects must drop straight into the
    # two-stream solver.
    # =====================================================================
    @testset "RadiationCore contract and TwoStream integration" begin
        surfaces = Any[material_albedo(:asphalt, sw),
                       material_albedo(:grass, sw),
                       SnowSurface(sw),
                       FresnelWater(sw; wind_speed = 6.0),
                       BRDFSurface(rpv_preset(:grass), sw; material = :grass,
                                   emissivity = :grass),
                       tiled_surface(sw, :concrete => 0.5, :grass => 0.5)]
        for s in surfaces
            @test s isa AbstractSurfaceOptics
            αd = direct_albedo_vector(s, 0.8)
            αf = diffuse_albedo_vector(s)
            ε  = emissivity_vector(s)
            @test length(αd) == nbands(sw)
            @test length(αf) == nbands(sw)
            @test length(ε)  == nbands(lw)
            @test all(0 .<= αd .<= 1)
            @test all(0 .<= αf .<= 1)
            @test all(0 .<= ε  .<= 1)
            @test 0 <= broadband_albedo(s, 0.8) <= 1
        end

        # BRDFSurface keeps the BRDF's broadband albedo while taking the
        # material's spectral shape
        bs = BRDFSurface(rpv_preset(:grass), sw; material = :grass)
        @test isapprox(broadband_albedo(bs), white_sky_albedo(rpv_preset(:grass));
                       rtol = 1e-6)
        @test surface_albedo(bs, NIR) > 5 * surface_albedo(bs, VIS)
        flat = BRDFSurface(LambertianBRDF(0.25), sw)
        @test all(diffuse_albedo_vector(flat) .≈ 0.25)
        @info "BRDFSurface (RPV grass × grass spectrum)" broadband=broadband_albedo(bs) vis=surface_albedo(bs, VIS) nir=surface_albedo(bs, NIR)

        # --- run the actual two-stream solver ---------------------------
        nz = 20
        col = us_standard_column(; nz = nz)
        go = correlated_k_optics(sw)
        opt = allocate_optics(Float64, (nz,), sw)
        compute_optics!(opt, go, col)
        solver = two_stream_solver(sw)
        μ₀ = 0.8
        up_toa = Dict{Symbol, Float64}()
        for (nm, s) in ((:asphalt, material_albedo(:asphalt, sw)),
                        (:grass,   material_albedo(:grass, sw)),
                        (:snow,    SnowSurface(sw)),
                        (:water,   FresnelWater(sw; wind_speed = 6.0)))
            f = allocate_fluxes(Float64, (), nz + 1)
            solve_sw!(f, solver, opt; μ₀ = μ₀,
                      sfc_albedo_direct  = direct_albedo_vector(s, μ₀),
                      sfc_albedo_diffuse = diffuse_albedo_vector(s))
            up_toa[nm] = f.flux_up[end]
            @test all(isfinite, f.flux_up)
            @test all(f.flux_up .>= 0)
            @test f.flux_dn[end] > f.flux_dn[1]      # attenuated downwards
        end
        @info "TOA upward shortwave flux by surface type (μ₀ = 0.8)" up_toa
        @test up_toa[:water] < up_toa[:asphalt] < up_toa[:grass] < up_toa[:snow]
        # the snow-to-water contrast is worth hundreds of W/m² at TOA —
        # the reason surface type cannot be a tuning constant
        @test up_toa[:snow] - up_toa[:water] > 200.0

        # longwave: emissivity enters the upwelling flux
        lwg = longwave_bands()
        golw = correlated_k_optics(lwg)
        optlw = allocate_optics(Float64, (nz,), lwg)
        compute_optics!(optlw, golw, col)
        slw = two_stream_solver(lwg)
        upl = Dict{Symbol, Float64}()
        for (nm, s) in ((:asphalt, material_albedo(:asphalt, sw)),
                        (:metal,   material_albedo(:metal_roof, sw)))
            f = allocate_fluxes(Float64, (), nz + 1)
            solve_lw!(f, slw, optlw, col;
                      sfc_emissivity = emissivity_vector(s))
            upl[nm] = f.flux_up[1]
            @test all(isfinite, f.flux_up)
        end
        @info "Surface upwelling longwave flux" upl
        @test upl[:metal] < upl[:asphalt]
    end
end
