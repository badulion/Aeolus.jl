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

@testset "Spectral albedo and BRDF (Phase 36)" begin

    sw = shortwave_bands()
    lw = longwave_bands()
    VIS = band_containing(sw, 0.55)
    NIR = band_containing(sw, 0.90)
    SWIR = band_containing(sw, 1.50)

    # Every downstream number is a band integral: the wavelength mapping must be right.
    @testset "Band bookkeeping and spectral library" begin
        @test (VIS, NIR, SWIR) == (10, 8, 6)
        @test all(1:nbands(sw)) do b
            λlo, λhi = band_wavelength_limits(sw, b); νlo, νhi = band_limits(sw, b)
            isapprox(λlo, 1e4 / νhi; rtol = 1e-12) && isapprox(λhi, 1e4 / νlo; rtol = 1e-12) &&
                λlo <= band_center_wavelength(sw, b) <= λhi
        end
        @test spectral_reflectance(:grass, 0.85) ≈ (0.48 + 0.50) / 2      # linear interpolation
        @test spectral_reflectance(:grass, 100.0) == spectral_reflectance(:grass, 12.0)  # clamped
        @test all(0 <= spectral_reflectance(m, λ) <= 1 for m in material_names(), λ in 0.2:0.1:3.0)

        bb = Dict(m => broadband_albedo(material_albedo(m, sw)) for m in material_names())
        # Oke (1987) Table 1.1 ranges
        @test 0.05 <= bb[:asphalt] <= 0.15 && 0.15 <= bb[:grass] <= 0.26 &&
              0.02 <= bb[:water] <= 0.10 && 0.75 <= bb[:snow_fresh] <= 0.95
        @test bb[:soil_wet] < bb[:soil_dry] && bb[:snow_old] < bb[:snow_fresh]
        # the vegetation red edge: NIR/VIS ≥ 5 for leaves, < 2 for built surfaces
        @test all(m -> (a = material_albedo(m, sw); a.diffuse[NIR] / a.diffuse[VIS] >= 5),
                  (:grass, :tree_deciduous, :tree_coniferous))
        @test all(m -> (a = material_albedo(m, sw); a.diffuse[NIR] / a.diffuse[VIS] < 2),
                  (:asphalt, :concrete, :metal_roof))
        # emissivity: the metal-roof outlier and the quartz reststrahlen band at 9 μm
        @test material_emissivity(:metal_roof) ≈ 0.25 && material_emissivity(:asphalt) ≈ 0.95
        εb = emissivity_vector(material_albedo(:sand, sw))
        @test length(εb) == nbands(lw) && εb[band_containing(lw, 9.0)] == minimum(εb) < 0.85
        @test_throws ErrorException material_albedo(:unobtainium, sw)
    end

    # Direct-beam albedo rises as the sun drops, and the white-sky albedo is
    # the cosine-weighted mean of the black-sky one (normalised shape function).
    @testset "Direct-beam zenith dependence" begin
        hm(a, b) = sum(2 * ((i - 0.5) / 400) * surface_albedo(a, b, (i - 0.5) / 400) / 400 for i in 1:400)
        for m in (:grass, :asphalt)
            a = material_albedo(m, sw)
            @test issorted([surface_albedo(a, VIS, μ) for μ in (1.0, 0.5, 0.2)])
            @test isapprox(hm(a, VIS), surface_albedo(a, VIS); rtol = 1e-3)
        end
        # pure snow in the visible: Briegleb shape is clamped at 1, costing < 5e-3
        sn = material_albedo(:snow_fresh, sw)
        @test 0 < surface_albedo(sn, VIS) - hm(sn, VIS) < 5e-3
        @test surface_albedo(sn, VIS, 0.05) == 1.0
    end

    # ---------------------------- BRDF models ----------------------------
    models = Any[LambertianBRDF(0.3)]
    for p in (:grass, :forest, :snow, :urban)
        push!(models, rpv_preset(p), rossli_preset(p))
    end
    push!(models, cook_torrance_preset(:glass), cook_torrance_preset(:metal_roof))

    # Helmholtz reciprocity (the Li-sparse-reciprocal kernel) — a violating
    # BRDF pumps energy around a multiple-reflection solver.
    @testset "Reciprocity and energy conservation" begin
        Random.seed!(20240936)
        worst = 0.0
        for m in models, _ in 1:300
            θi, θv, φi, φv = 1.45rand(), 1.45rand(), 2π * rand(), 2π * rand()
            f1 = brdf(m, θi, φi, θv, φv); f2 = brdf(m, θv, φv, θi, φi)
            worst = max(worst, abs(f1 - f2) / max(abs(f1), 1e-12))
        end
        @test worst < 1e-10
        # the shipped parameter sets reflect no more than they receive up to 85°
        @test all(0 <= black_sky_albedo(m, deg2rad(d)) <= 1 for m in models, d in 0.0:17.0:85.0)
        @test all(0 <= white_sky_albedo(m) <= 1 for m in models)
        # beyond ~87° the published kernels genuinely diverge (documented limitation)
        @test black_sky_albedo(rpv_preset(:forest), deg2rad(89.5)) > 1
    end

    # Quadrature calibration (Lambertian ⇒ exactly a) and the MODIS MCD43
    # white-sky kernel integrals, reproduced from the kernel formulae alone.
    @testset "Lambertian limit and MODIS kernel integrals" begin
        L = LambertianBRDF(0.3)
        @test maximum(abs(hemisphere_quadrature_albedo(L, θ; nμ = 40, nφ = 72) - 0.3)
                      for θ in (0.0, 0.7, 1.4)) < 1e-12
        @test isapprox(white_sky_albedo(L), 0.3; atol = 1e-12)
        kvol = RossLiBRDF(0.0, 1.0, 0.0; clamp_negative = false)
        kgeo = RossLiBRDF(0.0, 0.0, 1.0; clamp_negative = false)
        @test isapprox(white_sky_albedo(kvol; nμi = 64), 0.189184; atol = 1e-4)
        @test isapprox(white_sky_albedo(kgeo; nμi = 64), -1.377622; atol = 1e-3)
        @test isapprox(ross_thick_kernel(0.0, 0.0, 0.0), 0.0; atol = 1e-12)
        @test isapprox(li_sparse_kernel(0.0, 0.0, 0.0), 0.0; atol = 1e-12)
    end

    # The hot spot: backscatter beats every azimuth at fixed view zenith; the
    # RPV hot-spot factor has amplitude 2 - ρ0 at G = 0.
    @testset "Hot spot" begin
        θi = deg2rad(35)
        @test all((rpv_preset(:soil), rossli_preset(:grass))) do m
            hs = brdf(m, θi, 0.0, θi, 0.0)
            azi = [brdf(m, θi, 0.0, θi, deg2rad(d)) for d in (2, 5, 10, 20)]
            all(hs .> azi) && issorted(azi; rev = true)
        end
        @test isapprox(brdf(RPVBRDF(0.05, 1.0, 0.0), θi, 0.0, θi, 0.0) / 0.05, 1.95; atol = 1e-12)
    end

    # Cook-Torrance: the half-vector quadrature against brute force, the
    # mirror limit and Fresnel brightening at grazing incidence.
    @testset "Cook-Torrance microfacet" begin
        m = CookTorranceBRDF(0.05, 0.3, 0.3)
        @test isapprox(black_sky_albedo(m, 0.7),
                       hemisphere_quadrature_albedo(m, 0.7; nμ = 400, nφ = 800); rtol = 3e-3)
        @test 0.999 < black_sky_albedo(CookTorranceBRDF(0.0, 1.0, 0.01), 0.0) <= 1.0
        αs = [black_sky_albedo(cook_torrance_preset(:glass), deg2rad(d)) for d in (0, 60, 85)]
        @test issorted(αs) && αs[1] < 0.10 && αs[end] > 4αs[1]
        @test_throws ErrorException CookTorranceBRDF(0.1, 0.1, -0.1)
    end

    # Fresnel water + Cox-Munk: closed-form normal incidence and Brewster
    # angle, and wind lowering the grazing albedo.
    @testset "Fresnel water surface" begin
        n = 1.34
        @test isapprox(fresnel_reflectance(0.0, n), ((n - 1) / (n + 1))^2; atol = 1e-12)
        θB = atan(n); c = cos(θB); ct = sqrt(1 - (sin(θB) / n)^2)
        @test isapprox(fresnel_reflectance(θB, n), ((c - n * ct) / (c + n * ct))^2 / 2; atol = 1e-10)
        @test cox_munk_slope_variance(10.0) ≈ 0.003 + 0.0512
        w5 = FresnelWater(sw; wind_speed = 5.0)
        αz = [surface_albedo(w5, VIS, cosd(d)) for d in (0, 40, 80)]
        @test issorted(αz) && 0.02 <= αz[1] <= 0.06 && αz[end] > 0.20
        @test water_albedo_direct(FresnelWater(sw; wind_speed = 15.0), deg2rad(80)) <
              0.7 * water_albedo_direct(FresnelWater(sw; wind_speed = 0.0), deg2rad(80))
    end

    # Snow: grain growth darkens the NIR far more than the visible; soot is a
    # visible-band poison (Wiscombe & Warren 1980 Fig. 8); the escape-function
    # direct albedo integrates back to the diffuse albedo.
    @testset "Snow albedo" begin
        fine, coarse = SnowSurface(sw; grain_radius = 50.0), SnowSurface(sw; grain_radius = 2000.0)
        @test fine.diffuse[VIS] - coarse.diffuse[VIS] < 0.10
        @test fine.diffuse[NIR] - coarse.diffuse[NIR] > 0.30
        @test isapprox(snow_grain_radius(50.0, 1.0)^2 - 50.0^2,
                       snow_grain_radius(50.0, 2.0)^2 - snow_grain_radius(50.0, 1.0)^2; rtol = 1e-12)
        @test isapprox(SnowSurface(sw; soot_ppmw = 1.0).diffuse[VIS], 0.83; atol = 0.03)
        @test isapprox(SnowSurface(sw; soot_ppmw = 10.0).diffuse[VIS], 0.57; atol = 0.05)
        fresh = SnowSurface(sw)
        acc = sum(2 * ((i - 0.5) / 400) * snow_albedo(fresh, VIS, (i - 0.5) / 400) / 400 for i in 1:400)
        @test isapprox(acc, fresh.diffuse[VIS]; rtol = 3e-3)
        @test 0 < snow_albedo(fresh, VIS, 0.0) <= 1
    end

    # A mosaic is exactly the area-weighted mean; fractions must close.
    @testset "Tiled surface" begin
        cell = tiled_surface(sw, :asphalt => 0.35, :concrete => 0.25, :grass => 0.25, :tree_deciduous => 0.15)
        fr, cs = tile_fractions(cell), tile_components(cell)
        @test maximum(abs(surface_albedo(cell, b, μ) - sum(fr .* [surface_albedo(c, b, μ) for c in cs]))
                      for b in (VIS, NIR), μ in (1.0, 0.3)) < 1e-14
        @test isapprox(surface_emissivity(cell, 7), sum(fr .* [surface_emissivity(c, 7) for c in cs]); atol = 1e-14)
        @test_throws ErrorException tiled_surface(sw, :grass => 0.5, :asphalt => 0.4)
    end

    # Contract: every surface type drops straight into the two-stream solver,
    # and the surface type moves the TOA upward flux by hundreds of W/m².
    @testset "RadiationCore contract and TwoStream integration" begin
        surfaces = Any[material_albedo(:asphalt, sw), SnowSurface(sw), FresnelWater(sw; wind_speed = 6.0),
                       BRDFSurface(rpv_preset(:grass), sw; material = :grass, emissivity = :grass),
                       tiled_surface(sw, :concrete => 0.5, :grass => 0.5)]
        @test all(surfaces) do s
            αd, αf, ε = direct_albedo_vector(s, 0.8), diffuse_albedo_vector(s), emissivity_vector(s)
            s isa AbstractSurfaceOptics && length(αd) == length(αf) == nbands(sw) &&
                length(ε) == nbands(lw) && all(0 .<= αd .<= 1) && all(0 .<= αf .<= 1) && all(0 .<= ε .<= 1)
        end
        @test isapprox(broadband_albedo(surfaces[4]), white_sky_albedo(rpv_preset(:grass)); rtol = 1e-6)

        nz = 20
        col = us_standard_column(; nz = nz)
        opt = allocate_optics(Float64, (nz,), sw)
        compute_optics!(opt, correlated_k_optics(sw), col)
        up = map((surfaces[3], surfaces[1], surfaces[2])) do s
            f = allocate_fluxes(Float64, (), nz + 1)
            solve_sw!(f, two_stream_solver(sw), opt; μ₀ = 0.8,
                      sfc_albedo_direct = direct_albedo_vector(s, 0.8),
                      sfc_albedo_diffuse = diffuse_albedo_vector(s))
            f.flux_up[end]
        end
        @test issorted(up) && up[3] - up[1] > 200       # water < asphalt < snow
    end
end
