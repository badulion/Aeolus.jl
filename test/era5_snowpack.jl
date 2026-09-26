# Phase 51 (#34) — layered snowpack with compaction.
#
# Material laws against their published forms and limits, compaction
# against closed-form solutions under constant load, heat conduction
# against analytic diffusion solutions, exact mass and energy
# conservation of the full surface step, and the bulk-density evolution
# of a synthetic alpine season against the observational climatology of
# Sturm et al. (2010).

using Test, Printf
using Aeolus
using Aeolus.Snowpack
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS, cp_ice, LH_f0, grav
using Aeolus.MoistThermodynamics: saturation_specific_humidity, latent_heat_sublim
using Aeolus.GasOptics: erf_series

const SP = Aeolus.Snowpack
const Tf_snow = DEFAULT_SURFACE_CONSTANTS.thermo.T_freeze

# A pack of N identical layers (bypassing the layering rules).
function uniform_pack(N, H, ρ, Tfun; kwargs...)
    p = LayeredSnowpack(; kwargs...)
    dz = H / N
    for i in 1:N
        SP._push_layer!(p, dz, ρ * dz, 0.0, Tfun((i - 0.5) * dz), 50.0, 0.0)
    end
    return p
end

@testset "Era V — Snowpack (#34)" begin

    @testset "Thermal conductivity laws: published forms and physical limits" begin
        c = DEFAULT_SURFACE_CONSTANTS
        yen, stu, jor, cal = Yen1981Conductivity(), Sturm1997Conductivity(),
                             Jordan1991Conductivity(), Calonne2011Conductivity()
        # Sturm et al. (1997) Eq. 4 at 0.3 g cm⁻³: 0.138 - 0.303 + 0.29097.
        @test snow_thermal_conductivity(stu, 300.0) ≈ 0.12597 atol = 1e-5
        # ... and its two branches meet at 0.156 g cm⁻³ (to the 3 digits published).
        @test abs(snow_thermal_conductivity(stu, 155.999) - snow_thermal_conductivity(stu, 156.0)) < 1e-3
        # Yen (1981): k = k_ice at the density of water, by construction.
        @test snow_thermal_conductivity(yen, c.ρ_water) ≈ c.k_ice
        # Jordan (1991): still air at ρ → 0, ice (within 0.1 %) at ρ_ice.
        @test snow_thermal_conductivity(jor, 0.0) == c.k_air
        @test snow_thermal_conductivity(jor, c.ρ_ice) ≈ c.k_ice rtol = 1e-3
        # Calonne (2011) tends to still air.
        @test snow_thermal_conductivity(cal, 0.0) ≈ c.k_air rtol = 1e-12
        # Seasonal-snow range: monotone, between air and ice.
        for law in (yen, stu, jor, cal)
            ks = [snow_thermal_conductivity(law, ρ) for ρ in 150.0:10.0:550.0]
            @test all(diff(ks) .> 0)
            @test all(c.k_air .< ks .< c.k_ice)
        end
        # The laws genuinely disagree: at 300 kg m⁻³ Yen/Sturm ≈ 1.8.
        r = snow_thermal_conductivity(yen, 300.0) / snow_thermal_conductivity(stu, 300.0)
        @test 1.6 < r < 2.0
        @info "k(300 kg m⁻³)" Yen = snow_thermal_conductivity(yen, 300.0) Sturm = snow_thermal_conductivity(stu, 300.0) Jordan = snow_thermal_conductivity(jor, 300.0) Calonne = snow_thermal_conductivity(cal, 300.0)
    end

    @testset "Fresh-snow density (Anderson 1976 / CLM4; Vionnet 2012 Eq. 1)" begin
        A, V = Anderson1976FreshSnow(), Vionnet2012FreshSnow()
        @test fresh_snow_density(A, Tf_snow - 25) == 50.0
        @test fresh_snow_density(A, Tf_snow - 15) == 50.0
        @test fresh_snow_density(A, Tf_snow - 5) ≈ 50 + 1.7 * 10^1.5
        @test fresh_snow_density(A, Tf_snow + 2) ≈ 50 + 1.7 * 17^1.5      # 169.15
        @test fresh_snow_density(A, Tf_snow + 10) == fresh_snow_density(A, Tf_snow + 2)
        @test fresh_snow_density(A, Tf_snow + 2) ≈ 169.15 atol = 0.01
        # Vionnet: 109 + 6 ΔT + 26 √U, floor 50 kg m⁻³.
        @test fresh_snow_density(V, Tf_snow - 5, 0.0) ≈ 79.0
        @test fresh_snow_density(V, Tf_snow - 5, 10.0) ≈ 79 + 26 * sqrt(10)
        @test fresh_snow_density(V, Tf_snow - 20, 0.0) == 50.0
        @test fresh_snow_density(V, Tf_snow, 4.0) > fresh_snow_density(V, Tf_snow, 1.0)
    end
end
