# Era V, Phase 46 (#15) — SoilColumn: CLM-style soil column with
# Richards-equation water transport.
#
# Reference values are from the cited publications; the solver is checked
# against exact solutions of Richards' equation (Gardner soil), exact
# similarity laws (horizontal absorption), closed-form steady states and
# exact water-mass conservation.

using Test, Printf
using Aeolus
using Aeolus.SoilColumn
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS, SurfaceForcing, SurfaceFluxes,
                              moisture_availability, grav

const SC = Aeolus.SoilColumn

@testset "Era V — SoilColumn (#15)" begin

@testset "Layer structure" begin
    # CLM5.0 Technical Note, Table 2.2.3 (node depth, thickness, interface), m.
    g = clm5_soil_grid()
    @test length(g) == 20
    for (i, z, dz, zh) in ((1, 0.010, 0.020, 0.020), (4, 0.160, 0.080, 0.200),
                           (10, 1.360, 0.320, 1.520), (13, 2.500, 0.440, 2.720),
                           (20, 8.030, 1.140, 8.600))
        @test g.z[i] ≈ z atol = 5e-4
        @test g.dz[i] ≈ dz atol = 5e-4
        @test g.zh[i+1] ≈ zh atol = 5e-4
    end
    @test all(diff(g.dz) .> 0)                       # thickening with depth
    @test sum(g.dz) ≈ g.zh[end]
    # CLM3 exponential grid (Oleson et al. 2004, Table 6.1): layer 1 and 10.
    e = clm_exponential_soil_grid()
    @test e.z[1] ≈ 0.0071 atol = 1e-4
    @test e.dz[1] ≈ 0.0175 atol = 1e-4
    @test e.z[10] ≈ 2.8647 atol = 1e-4
    @test e.dz[10] ≈ 1.1370 atol = 1e-4
    @test e.zh[end] ≈ 3.4331 atol = 1e-4
    @test all(diff(e.dz) .> 0)
    @test_throws ArgumentError SoilGrid([0.1, 0.2])  # surface must be at 0
end

@testset "Hydraulic functions" begin
    models = (clapp_hornberger_texture(:loam), carsel_parrish_texture(:loam),
              GardnerExponential(θ_r = 0.05, θ_s = 0.40, α = 2.0, K_s = 1e-5))
    # potentials inside each model's unsaturated range (CH loam air entry is
    # -0.478 m; the Gardner θ - θ_r underflows below ψ ≈ -15 m)
    ψsets = ([-1e3, -30.0, -3.0, -1.0, -0.6], [-1e3, -30.0, -3.0, -1.0, -0.3],
             [-5.0, -3.0, -1.0, -0.3, -0.05])
    for (h, ψs) in zip(models, ψsets)
        # retention and its inverse
        for ψ in ψs
            @test soil_matric_potential(h, soil_water_content(h, ψ)) ≈ ψ rtol = 1e-10
        end
        # C = dθ/dψ, checked by centred differences
        for ψ in ψs
            δ = 1e-6 * abs(ψ)
            dθ = (soil_water_content(h, ψ + δ) - soil_water_content(h, ψ - δ)) / 2δ
            @test soil_water_capacity(h, ψ) ≈ dθ rtol = 1e-5 atol = 1e-12
        end
        # monotone: wetter soil holds more water and conducts better
        θ = soil_water_content.(Ref(h), ψs)
        K = soil_hydraulic_conductivity.(Ref(h), ψs)
        @test issorted(θ) && issorted(K)
        # saturation
        @test soil_water_content(h, 0.0) == soil_porosity(h)
        @test soil_hydraulic_conductivity(h, 0.0) == saturated_conductivity(h)
        @test soil_water_capacity(h, 0.1) == 0
        @test soil_water_content(h, -1e6) >= residual_water_content(h)
    end
    # Clapp-Hornberger: ψ(θ_s) is the air-entry value, K = K_s (θ/θ_s)^(2b+3)
    ch = clapp_hornberger_texture(:loam)
    @test soil_matric_potential(ch, ch.θ_s) == ch.ψ_s == -0.478
    θ = 0.25
    @test soil_hydraulic_conductivity(ch, soil_matric_potential(ch, θ)) ≈
          ch.K_s * (θ / ch.θ_s)^(2ch.b + 3) rtol = 1e-12
    # Gardner diffusivity is constant, D = K_s / (α (θ_s - θ_r))
    gd = models[3]
    @test soil_water_diffusivity(gd, -0.2) ≈ 1e-5 / (2.0 * 0.35) rtol = 1e-12
    @test soil_water_diffusivity(gd, -2.0) ≈ 1e-5 / (2.0 * 0.35) rtol = 1e-12
end

@testset "Texture tables and pedotransfer" begin
    # Unit conversions of the published tables (CH78 via Tarboton 2003,
    # K_s in cm/h; Carsel & Parrish 1988, K_s in cm/h, α in 1/cm).
    @test clapp_hornberger_texture(:sand).K_s ≈ 63.36 / 100 / 3600
    @test clapp_hornberger_texture(:sand).ψ_s ≈ -0.121
    @test carsel_parrish_texture(:sand).K_s ≈ 712.8 / 100 / 86400 rtol = 1e-12   # 712.8 cm/d
    @test carsel_parrish_texture(:loam).α ≈ 3.6
    # Physics of texture: coarse soils drain fast and hold little water.
    for (tab, ctor) in ((CLAPP_HORNBERGER_1978, clapp_hornberger_texture),
                        (CARSEL_PARRISH_1988, carsel_parrish_texture))
        s, l, c = ctor(:sand), ctor(:loam), ctor(:clay)
        @test s.K_s > l.K_s > c.K_s
        ψ_fc = -3.37
        @test soil_water_content(s, ψ_fc) < soil_water_content(l, ψ_fc) < soil_water_content(c, ψ_fc)
    end
    @test_throws ArgumentError clapp_hornberger_texture(:peat)
    # Cosby et al. (1984) regressions at USDA-class centroid textures land
    # within one CH78 standard deviation (CH78 Table 2 SDs) of the class
    # means: (sand %, clay %) = sand (92, 3), loam (40, 20), clay (20, 60).
    for (tex, sand, clay, sd_b, sd_θ, sd_ψ) in ((:sand, 92, 3, 1.78, 0.056, 14.3),
                                               (:loam, 40, 20, 1.87, 0.078, 51.2),
                                               (:clay, 20, 60, 3.70, 0.050, 39.7))
        cz = cosby_clapp_hornberger(sand, clay)
        ch = clapp_hornberger_texture(tex)
        @test abs(cz.b - ch.b) < sd_b
        @test abs(cz.θ_s - ch.θ_s) < sd_θ
        @test abs(cz.ψ_s - ch.ψ_s) * 100 < sd_ψ
    end
    @test cosby_clapp_hornberger(90, 5).K_s > cosby_clapp_hornberger(20, 50).K_s
end

@testset "Viscosity and temperature-dependent conductivity" begin
    # CRC Handbook of Chemistry and Physics, viscosity of liquid water at
    # 0.1 MPa (mPa s).
    for (tC, μ) in ((0, 1.7914), (10, 1.3060), (20, 1.0016), (25, 0.8900),
                    (40, 0.6527), (60, 0.4665), (80, 0.3544))
        @test water_dynamic_viscosity(273.15 + tC) * 1e3 ≈ μ rtol = 3e-3
    end
    @test viscosity_conductivity_factor(293.15) == 1
    # cold soil: K(0 °C)/K(20 °C) ≈ 0.559; warm soil conducts more
    @test viscosity_conductivity_factor(273.15) ≈ 1.0016 / 1.7914 rtol = 3e-3
    @test viscosity_conductivity_factor(303.15) > 1
    @test ice_impedance_factor(NoIceImpedance(), 0.1, 0.4) == 1
end

@testset "Parlange sorptivity" begin
    # For a constant-D (Gardner) soil the integral is exact: S² = 1.5 D Δθ²,
    # and must sit between the rigorous bounds D Δθ² and 2 D Δθ².
    gd = GardnerExponential(θ_r = 0.05, θ_s = 0.40, α = 2.0, K_s = 1e-5)
    ψi = -5.0
    Δθ = soil_porosity(gd) - soil_water_content(gd, ψi)
    D = soil_water_diffusivity(gd, -1.0)
    S = parlange_sorptivity(gd, ψi)
    # K dψ = D dθ with D constant ⇒ S² = D ∫(θ_s + θ - 2θ_i) dθ = 1.5 D Δθ²
    @test S ≈ sqrt(1.5 * D) * Δθ rtol = 1e-4
    @test Δθ * sqrt(D) < S < Δθ * sqrt(2D)
end

end # SoilColumn testset
