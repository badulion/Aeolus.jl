# Phase 51 (#34) — layered snowpack with compaction.
#
# Material laws against their published forms and limits, compaction
# against closed-form solutions under constant load, heat conduction
# against analytic diffusion solutions, exact mass and energy
# conservation of the full surface step. The bulk-density evolution of a
# synthetic alpine season against the observational climatology of Sturm
# et al. (2010) is a validation run: validation/snowpack.jl.

using Test
using Aeolus
using Aeolus.Snowpack
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS, cp_ice, LH_f0, grav
using Aeolus.MoistThermodynamics: saturation_specific_humidity, latent_heat_sublim

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

@testset "Snowpack (#34)" begin

    @testset "Conductivity and fresh-snow density: published forms" begin
        c = DEFAULT_SURFACE_CONSTANTS
        # Sturm et al. (1997) Eq. 4 at 0.3 g cm⁻³: 0.138 − 0.303 + 0.29097
        @test snow_thermal_conductivity(Sturm1997Conductivity(), 300.0) ≈ 0.12597 atol = 1e-5
        # Jordan (1991): still air at ρ → 0, ice (0.1 %) at ρ_ice
        @test snow_thermal_conductivity(Jordan1991Conductivity(), 0.0) == c.k_air
        @test snow_thermal_conductivity(Jordan1991Conductivity(), c.ρ_ice) ≈ c.k_ice rtol = 1e-3
        # Anderson (1976) / CLM4 and Vionnet (2012) Eq. 1
        @test fresh_snow_density(Anderson1976FreshSnow(), Tf_snow + 2) ≈ 50 + 1.7 * 17^1.5
        @test fresh_snow_density(Vionnet2012FreshSnow(), Tf_snow - 5, 10.0) ≈ 79 + 26 * sqrt(10)
    end

    @testset "Compaction: closed-form solutions under constant load" begin
        c = DEFAULT_SURFACE_CONSTANTS
        day = 86400.0
        one_layer(ρ0, T, comp) = LayeredSnowpack(depth = 1.0, density = ρ0, T = T,
                                                 compaction = comp,
                                                 layering = SnowLayering(max_layers = 1))
        # (a) destructive metamorphism alone: ρ = ρ0 exp(c3 e^{−c4 ΔT} t)
        p = one_layer(60.0, Tf_snow - 5.0, Anderson1976Compaction(η0 = Inf))
        for _ in 1:48
            snow_compaction!(p, 3600.0)
        end
        @test snow_bulk_density(p) ≈ 60.0 * exp(2.777e-6 * exp(-0.2) * 2day) rtol = 1e-12
        @test snow_water_equivalent(p) ≈ 60.0 rtol = 1e-14
        # (b) Vionnet (2012) viscous settling: e^{bρ} = e^{bρ0} + b K t
        V = Vionnet2012Compaction()
        ρ0, ΔT = 100.0, 10.0
        p = one_layer(ρ0, Tf_snow - ΔT, V)
        K = grav(c) * ρ0 / 2 * V.c_η / (V.η0 * exp(V.a_η * ΔT))
        for _ in 1:120
            snow_compaction!(p, 3600.0)
        end
        @test snow_bulk_density(p) ≈ log(exp(V.b_η * ρ0) + V.b_η * K * 5day) / V.b_η rtol = 1e-4
        # (c) Anderson overburden: t(ρ) = η0 e^{c5ΔT}/P ∫ e^{c6 x}/x dx
        A = Anderson1976Compaction(destructive = nothing)
        ρ0, ΔT = 150.0, 4.0
        p = one_layer(ρ0, Tf_snow - ΔT, A)
        for _ in 1:240
            snow_compaction!(p, 3600.0)
        end
        ρ1 = snow_bulk_density(p)
        nq = 2_000
        h = (ρ1 - ρ0) / nq
        g(x) = exp(A.c6 * x) / x
        integral = h / 3 * (g(ρ0) + g(ρ1) + 4sum(g(ρ0 + (2j - 1) * h) for j in 1:nq÷2) +
                            2sum(g(ρ0 + 2j * h) for j in 1:nq÷2-1))
        @test A.η0 * exp(A.c5 * ΔT) / (ρ0 / 2) * integral ≈ 10day rtol = 1e-4
        # (d) never beyond the density of ice
        p = LayeredSnowpack(depth = 1.0, density = 800.0, T = Tf_snow,
                            layering = SnowLayering(max_layers = 1))
        p.m_ice[1] = 900.0
        for _ in 1:10
            snow_compaction!(p, 1e9)
        end
        @test snow_bulk_density(p) <= c.ρ_ice * (1 + 1e-12)
    end

    @testset "Heat conduction: energy, Fourier mode, series resistance" begin
        c = DEFAULT_SURFACE_CONSTANTS
        law = Jordan1991Conductivity()
        # (a) exact energy bookkeeping, irregular layers
        p = LayeredSnowpack(depth = 0.83, density = 230.0, T = 260.0, bottom = SnowBottomFlux(-3.0), θ = 0.5)
        p.T .= range(255.0, 271.0; length = snow_nlayers(p))
        C = [SP._heat_capacity(p, i) for i in 1:snow_nlayers(p)]
        T0 = copy(p.T)
        Ft, Fb = snow_heat_conduction!(p, 1800.0; top_flux = 40.0, dtop_flux_dT = -12.0)
        @test sum(C .* (p.T .- T0)) ≈ (Ft - Fb) * 1800 rtol = 1e-10
        # (b) insulated slab, cosine mode decays as e^{−κπ²t/H²}
        N, H, ρ = 50, 0.5, 300.0
        p = uniform_pack(N, H, ρ, z -> 263.15 + cos(π * z / H); θ = 0.5)
        κ = snow_thermal_conductivity(law, ρ) / (ρ * cp_ice(c))
        τ = H^2 / (κ * π^2)
        z = snow_layer_depths(p)
        amp(p) = sum((p.T .- 263.15) .* cos.(π .* z ./ H)) / sum(cos.(π .* z ./ H) .^ 2)
        for _ in 1:1000
            snow_heat_conduction!(p, τ / 1000)
        end
        @test abs(amp(p) / exp(-1) - 1) < 1e-3
        @test sum(p.T) / N ≈ 263.15 rtol = 1e-13
        # (c) two-density pack at steady state: flux = ΔT / Σ Δz_i/k_i
        p = LayeredSnowpack(bottom = SnowGroundTemperature(273.15))
        for i in 1:20
            ρi = i <= 10 ? 150.0 : 400.0
            SP._push_layer!(p, 0.05, ρi * 0.05, 0.0, 263.0, 50.0, 0.0)
        end
        local Ft, Fb
        for _ in 1:50
            Ft, Fb = snow_heat_conduction!(p, 1e7; T_top = 253.15)
        end
        R = 10 * 0.05 / snow_thermal_conductivity(law, 150.0) + 10 * 0.05 / snow_thermal_conductivity(law, 400.0)
        @test -Fb ≈ 20 / R rtol = 1e-9
    end

    @testset "Layering: exact conservation" begin
        p = LayeredSnowpack(depth = 1.7, density = 180.0, T = 265.0)
        p.T .= range(255.0, 272.0; length = snow_nlayers(p))
        add_snowfall!(p, 60.0, 258.0, 90.0)
        M, D, Hh = snow_water_equivalent(p), snow_depth(p), snowpack_enthalpy(p)
        relayer_snowpack!(p)
        @test snow_water_equivalent(p) ≈ M rtol = 1e-14
        @test snow_depth(p) ≈ D rtol = 1e-14
        @test snowpack_enthalpy(p) ≈ Hh rtol = 1e-13
        @test snow_nlayers(p) <= p.layering.max_layers
    end

    @testset "Mass and energy conservation through a 6-day composite run" begin
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        p = LayeredSnowpack(depth = 0.6, density = 220.0, T = 266.0,
                            bottom = SnowGroundTemperature(273.0))
        M0, H0 = snow_water_equivalent(p), snowpack_enthalpy(p)
        dt = 3600.0
        dep_seen = false; capped = true; seb = 0.0
        for d in 1:6, s in 1:24
            sun = max(0.0, sin(π * (s - 6) / 12))
            if d <= 1        # snowstorm
                f = SurfaceForcing(sw_down = 50sun, lw_down = 290.0, T_air = 267.0, q_air = 0.0025,
                                   U = 6.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 4e-4)
            elseif d <= 2    # clear, cold, dry: sublimation
                f = SurfaceForcing(sw_down = 450sun, lw_down = 190.0, T_air = 263.0, q_air = 0.0008,
                                   U = 3.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
            elseif d <= 4    # warm spell with rain: melt and runoff
                f = SurfaceForcing(sw_down = 600sun, lw_down = 320.0, T_air = 279.0, q_air = 0.005,
                                   U = 5.0, p = 90000.0, z_ref = 10.0, rain = d == 4 ? 2e-4 : 0.0, snow = 0.0)
            else             # cold and humid: hoar deposition
                f = SurfaceForcing(sw_down = 0.0, lw_down = 0.9σ * 268^4, T_air = 268.0, q_air = 0.0032,
                                   U = 2.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
            end
            fl = step_surface!(p, f, dt)
            fl.E < 0 && (dep_seen = true)
            capped &= all(p.T .<= Tf_snow)
            seb = max(seb, abs(fl.Rn - fl.H - fl.LE - fl.G) / max(abs(fl.Rn), 1.0))
        end
        b = p.budget
        ΔM = snow_water_equivalent(p) - M0
        ΔH = snowpack_enthalpy(p) - H0
        @test capped
        @test seb < 1e-12
        @test abs(ΔM - (b.snowfall + b.rainfall - b.sublimation - b.runoff)) < 1e-10 * (M0 + b.snowfall)
        @test abs(ΔH - (b.surface_heat - b.bottom_heat + b.advected_enthalpy - b.runoff_enthalpy)) < 1e-10 * abs(H0)
        @test b.melt > 1 && b.runoff > 1 && dep_seen
    end

    @testset "A pack that sublimates away hands back only the vapour it had" begin
        p = LayeredSnowpack(depth = 0.002, density = 100.0, T = 265.0)
        M0, H0 = snow_water_equivalent(p), snowpack_enthalpy(p)
        f = SurfaceForcing(sw_down = 0.0, lw_down = 250.0, T_air = 268.0, q_air = 1e-4,
                           U = 15.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
        E_total = 0.0
        while snow_nlayers(p) > 0
            fl = step_surface!(p, f, 3600.0)
            E_total += fl.E * 3600
        end
        b = p.budget
        @test E_total ≈ b.sublimation rtol = 1e-12
        @test b.sublimation ≈ M0 rtol = 1e-12
        @test abs(-H0 - (b.surface_heat - b.bottom_heat + b.advected_enthalpy)) < 1e-9 * abs(H0)
    end

    @testset "Surface physics: melt pins the skin at T_f" begin
        p = LayeredSnowpack(depth = 0.5, density = 300.0, T = 272.0)
        fday = SurfaceForcing(sw_down = 500.0, lw_down = 320.0, T_air = 280.0, q_air = 0.006,
                              U = 4.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
        for _ in 1:24
            step_surface!(p, fday, 1800.0)
        end
        @test surface_temperature(p) == Tf_snow
        @test p.budget.melt > 10
        @test p.budget.runoff ≈ p.budget.melt rtol = 1e-12
    end
end
