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

    @testset "Compaction: closed-form solutions under constant load" begin
        c = DEFAULT_SURFACE_CONSTANTS
        day = 86400.0
        one_layer(ρ0, T, comp) = LayeredSnowpack(depth = 1.0, density = ρ0, T = T,
                                                 compaction = comp,
                                                 layering = SnowLayering(max_layers = 1))
        # (a) Destructive metamorphism alone (Anderson with η0 → ∞), ρ_ice ≤ 100:
        #     d ln ρ/dt = c3 e^{-c4 ΔT}  ⇒  ρ = ρ0 exp(c3 e^{-c4 ΔT} t).
        worst = 0.0
        for ΔT in (0.0, 5.0, 15.0)
            p = one_layer(60.0, Tf_snow - ΔT, Anderson1976Compaction(η0 = Inf))
            for _ in 1:48
                snow_compaction!(p, 3600.0)
            end
            ρ_exact = 60.0 * exp(2.777e-6 * exp(-0.04ΔT) * 2day)
            worst = max(worst, abs(snow_bulk_density(p) / ρ_exact - 1))
            @test snow_water_equivalent(p) ≈ 60.0 rtol = 1e-14         # mass untouched
        end
        @test worst < 1e-12       # constant rate: the midpoint rule is exact
        # Destructive settling at the melting point halves new snow in ln2/c3 = 2.89 d.
        p = one_layer(40.0, Tf_snow, Anderson1976Compaction(η0 = Inf))
        snow_compaction!(p, log(2) / 2.777e-6)
        @test snow_depth(p) ≈ 0.5 rtol = 1e-4
        @info "Destructive metamorphism: max rel. error vs exp. solution" worst

        # (b) Vionnet (2012) viscous settling: e^{bρ} = e^{bρ0} + b K t,
        #     K = σ c_η / (η0 e^{a ΔT}), σ = g × (half the layer's own mass).
        V = Vionnet2012Compaction()
        worstV = 0.0
        for (ρ0, ΔT) in ((100.0, 0.0), (100.0, 10.0), (250.0, 3.0))
            p = one_layer(ρ0, Tf_snow - ΔT, V)
            σ = grav(c) * ρ0 * 1.0 / 2
            K = σ * V.c_η / (V.η0 * exp(V.a_η * ΔT))
            for _ in 1:120
                snow_compaction!(p, 3600.0)
            end
            ρ_exact = log(exp(V.b_η * ρ0) + V.b_η * K * 5day) / V.b_η
            worstV = max(worstV, abs(snow_bulk_density(p) / ρ_exact - 1))
            @info @sprintf("Vionnet, ρ0 = %.0f, ΔT = %.0f K: ρ(5 d) = %.2f (exact %.2f)",
                           ρ0, ΔT, snow_bulk_density(p), ρ_exact)
        end
        @test worstV < 1e-4

        # (c) Anderson overburden alone: t(ρ) = η0 e^{c5ΔT}/P ∫_{ρ0}^{ρ} e^{c6 x}/x dx.
        A = Anderson1976Compaction(destructive = nothing)
        ρ0, ΔT = 150.0, 4.0
        p = one_layer(ρ0, Tf_snow - ΔT, A)
        P = ρ0 / 2
        for _ in 1:240
            snow_compaction!(p, 3600.0)
        end
        ρ1 = snow_bulk_density(p)
        nq = 20_000                                # Simpson quadrature
        h = (ρ1 - ρ0) / nq
        g(x) = exp(A.c6 * x) / x
        integral = h / 3 * (g(ρ0) + g(ρ1) + 4sum(g(ρ0 + (2j - 1) * h) for j in 1:nq÷2) +
                            2sum(g(ρ0 + 2j * h) for j in 1:nq÷2-1))
        t_implied = A.η0 * exp(A.c5 * ΔT) / P * integral
        @test t_implied ≈ 10day rtol = 1e-4
        @info "Anderson overburden, 10 d" ρ1 t_implied_days = t_implied / day

        # (d) Physics of the laws: colder and denser snow is stiffer, water softens it.
        base = (ρ = 200.0, ρ_ice = 200.0, ρ_liq = 0.0, T = Tf_snow - 2, load = 50.0, f2 = 1.0)
        for law in (Anderson1976Compaction(), Vionnet2012Compaction())
            r0 = snow_compaction_rate(law, base)
            @test r0 < 0
            @test abs(snow_compaction_rate(law, merge(base, (T = Tf_snow - 12,)))) < abs(r0)
            @test abs(snow_compaction_rate(law, merge(base, (ρ = 350.0, ρ_ice = 350.0)))) < abs(r0)
            @test abs(snow_compaction_rate(law, merge(base, (load = 100.0,)))) > abs(r0)
            @test abs(snow_compaction_rate(law, merge(base, (ρ = 230.0, ρ_liq = 30.0)))) > abs(r0)
        end
        # Vionnet overburden: exactly e^{-a_η ΔT} per kelvin of cooling, ∝ load.
        r2  = snow_compaction_rate(V, base)
        r12 = snow_compaction_rate(V, merge(base, (T = Tf_snow - 12,)))
        @test r12 / r2 ≈ exp(-V.a_η * 10) rtol = 1e-12
        # f1: 3 % liquid by volume → viscosity ÷ (1 + 60·0.03) = ÷ 2.8.
        wet = merge(base, (ρ_liq = 30.0,))
        @test snow_viscosity(V, base) / snow_viscosity(V, wet) ≈ 2.8 rtol = 1e-12
        # Grain factor f2 (the #35 hook) stiffens linearly.
        @test snow_compaction_rate(V, merge(base, (f2 = 4.0,))) ≈ r2 / 4

        # (e) Never beyond the density of ice, even under an ice-sheet load.
        p = LayeredSnowpack(depth = 1.0, density = 800.0, T = Tf_snow,
                            layering = SnowLayering(max_layers = 1))
        p.m_ice[1] = 900.0                             # ρ = 900 kg m⁻³
        for _ in 1:100
            snow_compaction!(p, 1e9)
        end
        @test snow_bulk_density(p) <= c.ρ_ice * (1 + 1e-12)
        @test snow_water_equivalent(p) == 900.0
    end

    @testset "Heat conduction: energy, Fourier mode, erf solution, series resistance" begin
        c = DEFAULT_SURFACE_CONSTANTS
        law = Jordan1991Conductivity()
        # (a) Exact energy bookkeeping, irregular layers, both boundary types.
        for bottom in (SnowBottomFlux(-3.0), SnowGroundTemperature(272.0)), θ in (0.5, 1.0)
            p = LayeredSnowpack(depth = 0.83, density = 230.0, T = 260.0, bottom = bottom, θ = θ)
            p.T .= range(255.0, 271.0; length = snow_nlayers(p))
            C = [SP._heat_capacity(p, i) for i in 1:snow_nlayers(p)]
            T0 = copy(p.T)
            Ft, Fb = snow_heat_conduction!(p, 1800.0; top_flux = 40.0, dtop_flux_dT = -12.0)
            @test sum(C .* (p.T .- T0)) ≈ (Ft - Fb) * 1800 rtol = 1e-10
        end
        # (b) Insulated slab, cosine mode: T = T0 + A cos(πz/H) e^{-κπ²t/H²}.
        N, H, ρ = 50, 0.5, 300.0
        p = uniform_pack(N, H, ρ, z -> 263.15 + cos(π * z / H); θ = 0.5)
        κ = snow_thermal_conductivity(law, ρ) / (ρ * cp_ice(c))
        τ = H^2 / (κ * π^2)
        z = snow_layer_depths(p)
        amp(p) = sum((p.T .- 263.15) .* cos.(π .* z ./ H)) / sum(cos.(π .* z ./ H) .^ 2)
        nstep = 1000
        dt = τ / nstep
        for _ in 1:nstep
            snow_heat_conduction!(p, dt)
        end
        decay_err = abs(amp(p) / exp(-1) - 1)
        @test decay_err < 1e-3
        @test sum(p.T) / N ≈ 263.15 rtol = 1e-13            # insulated: mean conserved
        @info @sprintf("Fourier mode: τ = %.0f s, amplitude after τ = %.6f (exact e⁻¹ = %.6f), rel. err %.2e",
                       τ, amp(p), exp(-1), decay_err)
        # (c) Surface temperature step on a deep pack: erf solution.
        N, H = 200, 2.0
        p = uniform_pack(N, H, ρ, z -> 263.15; θ = 1.0)
        tend, dt = 2 * 86400.0, 60.0
        for _ in 1:round(Int, tend / dt)
            snow_heat_conduction!(p, dt; T_top = 253.15)
        end
        z = snow_layer_depths(p)
        exact = 253.15 .+ 10 .* erf_series.(z ./ (2sqrt(κ * tend)))
        erf_err = maximum(abs.(p.T .- exact))
        @test erf_err < 0.005                           # K, on a 10 K step (measured 0.0009)
        @info @sprintf("erf solution after 2 d: max |T - T_exact| = %.4f K (diffusion length %.3f m)",
                       erf_err, 2sqrt(κ * tend))
        # (d) Two-density pack at steady state: flux = ΔT / Σ Δz_i/k_i.
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
        @test -Fb ≈ 20 / R rtol = 1e-9          # ground → snow → cold surface
        @test Ft ≈ Fb rtol = 1e-9
        @info "Series-resistance steady flux" F_numeric = -Fb F_exact = 20 / R
    end

    @testset "Layering: bounds and exact conservation" begin
        p = LayeredSnowpack(depth = 1.7, density = 180.0, T = 265.0)
        l = p.layering
        @test snow_nlayers(p) <= l.max_layers
        @test p.dz[1] <= 2l.dz_top && p.dz[1] >= l.dz_top / 2
        @test issorted(p.dz[1:end-1])                      # fine at the top
        # Heavy snowfall, then relayer: mass, depth, enthalpy conserved exactly.
        p.T .= range(255.0, 272.0; length = snow_nlayers(p))
        add_snowfall!(p, 60.0, 258.0, 90.0)
        M, D, Hh = snow_water_equivalent(p), snow_depth(p), snowpack_enthalpy(p)
        relayer_snowpack!(p)
        @test snow_water_equivalent(p) ≈ M rtol = 1e-14
        @test snow_depth(p) ≈ D rtol = 1e-14
        @test snowpack_enthalpy(p) ≈ Hh rtol = 1e-13
        @test snow_nlayers(p) <= l.max_layers
        # Too many layers are merged back to the limit.
        q = LayeredSnowpack(depth = 1.0, density = 200.0, layering = SnowLayering(max_layers = 3))
        @test snow_nlayers(q) == 3
        @test snow_water_equivalent(q) ≈ 200.0
        # A vanishing pack books its last mass as runoff.
        r = LayeredSnowpack(depth = 1e-9, density = 100.0)
        relayer_snowpack!(r)
        @test snow_nlayers(r) == 0
        @test r.budget.runoff ≈ 1e-7
        @test_throws ArgumentError SnowLayering(growth = 2.5)
    end

    @testset "AbstractSurface implementation" begin
        p = LayeredSnowpack(depth = 0.5, density = 250.0, T = 265.0)
        @test p isa AbstractSurface
        @test surface_temperature(p) == 265.0
        @test surface_albedo(p) == 0.8
        @test surface_emissivity(p) == Aeolus.SurfaceOptics.material_emissivity(:snow_fresh)
        @test roughness_lengths(p) == (z0m = 1e-3, z0h = 1e-4, z0q = 1e-4)
        @test moisture_availability(p) == 1.0
        f = SurfaceForcing(T_air = 268.0, q_air = 0.001, p = 90000.0)
        @test surface_specific_humidity(p, f) ≈ saturation_specific_humidity(265.0, 90000.0; over = :ice)
        @test surface_latent_heat(p) == latent_heat_sublim(265.0)
        # The returned fluxes close the surface energy balance exactly, and
        # G is precisely the heat that entered the pack.
        f = SurfaceForcing(sw_down = 300.0, lw_down = 250.0, T_air = 266.0, q_air = 0.0015,
                           U = 4.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
        s0 = p.budget.surface_heat
        fl = step_surface!(p, f, 1800.0)
        @test fl.Rn ≈ fl.H + fl.LE + fl.G rtol = 1e-12
        @test p.budget.surface_heat - s0 ≈ fl.G * 1800 rtol = 1e-12
        @test fl.E > 0                                  # dry air: sublimation
        @test fl.LE ≈ fl.E * latent_heat_sublim(surface_temperature(p)) rtol = 0.01
        # Bare ground: fluxes of the ground temperature, snowfall starts a pack.
        b = LayeredSnowpack(bottom = SnowGroundTemperature(272.0))
        @test surface_temperature(b) == 272.0
        step_surface!(b, SurfaceForcing(T_air = 268.0, q_air = 0.002, snow = 1e-3), 3600.0)
        @test snow_water_equivalent(b) ≈ 3.6 rtol = 1e-12
        # Deposited at the fresh-snow density, then settled for the rest of
        # the step (fresh snow at -5 °C settles ~1 % per hour).
        D_fresh = 3.6 / fresh_snow_density(Anderson1976FreshSnow(), 268.0)
        @test 0.98D_fresh < snow_depth(b) < D_fresh
    end

    @testset "Mass and energy conservation through a 12-day composite run" begin
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        p = LayeredSnowpack(depth = 0.6, density = 220.0, T = 266.0,
                            bottom = SnowGroundTemperature(273.0))
        M0, H0 = snow_water_equivalent(p), snowpack_enthalpy(p)
        dt = 1800.0
        dep_seen = false
        for d in 1:12, s in 1:48
            hr = s / 2
            sun = max(0.0, sin(π * (hr - 6) / 12))
            if d <= 3        # snowstorm
                f = SurfaceForcing(sw_down = 50sun, lw_down = 290.0, T_air = 267.0, q_air = 0.0025,
                                   U = 6.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 4e-4)
            elseif d <= 6    # clear, cold, dry: radiative cooling and sublimation
                f = SurfaceForcing(sw_down = 450sun, lw_down = 190.0, T_air = 263.0, q_air = 0.0008,
                                   U = 3.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
            elseif d <= 9    # warm spell with rain: melt and runoff
                f = SurfaceForcing(sw_down = 600sun, lw_down = 320.0, T_air = 279.0, q_air = 0.005,
                                   U = 5.0, p = 90000.0, z_ref = 10.0, rain = d == 8 ? 2e-4 : 0.0, snow = 0.0)
            else             # cold and humid: hoar deposition
                f = SurfaceForcing(sw_down = 0.0, lw_down = 0.9σ * 268^4, T_air = 268.0, q_air = 0.0032,
                                   U = 2.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
            end
            fl = step_surface!(p, f, dt)
            fl.E < 0 && (dep_seen = true)
            @test all(p.T .<= Tf_snow)
        end
        b = p.budget
        ΔM = snow_water_equivalent(p) - M0
        ΔH = snowpack_enthalpy(p) - H0
        mass_resid = ΔM - (b.snowfall + b.rainfall - b.sublimation - b.runoff)
        heat_resid = ΔH - (b.surface_heat - b.bottom_heat + b.advected_enthalpy - b.runoff_enthalpy)
        @test abs(mass_resid) < 1e-10 * (M0 + b.snowfall)
        @test abs(heat_resid) < 1e-10 * abs(H0)
        @test b.melt > 1 && b.runoff > 1 && b.sublimation != 0 && dep_seen
        @info @sprintf("12-day run: SWE %.2f → %.2f kg m⁻², snowfall %.1f, rain %.1f, sublimation %.3f, melt %.1f, runoff %.1f; mass residual %.2e kg m⁻², enthalpy residual %.2e J m⁻² (of %.3e)",
                       M0, snow_water_equivalent(p), b.snowfall, b.rainfall, b.sublimation,
                       b.melt, b.runoff, mass_resid, heat_resid, H0)
    end

    @testset "A pack that sublimates away hands back only the vapour it had" begin
        # 0.2 kg m⁻² of snow under very dry, windy air: the hourly
        # sublimation demand exceeds the pack. The atmosphere must receive
        # exactly the mass the pack lost, R_n = H + LE + G must still close,
        # and the pack's own budget must still close.
        p = LayeredSnowpack(depth = 0.002, density = 100.0, T = 265.0)
        M0, H0 = snow_water_equivalent(p), snowpack_enthalpy(p)
        f = SurfaceForcing(sw_down = 0.0, lw_down = 250.0, T_air = 268.0, q_air = 1e-4,
                           U = 15.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
        E_total = 0.0
        local fl
        while snow_nlayers(p) > 0
            fl = step_surface!(p, f, 3600.0)
            E_total += fl.E * 3600
            @test fl.Rn ≈ fl.H + fl.LE + fl.G rtol = 1e-12
        end
        b = p.budget
        @test E_total ≈ b.sublimation rtol = 1e-12
        @test b.sublimation ≈ M0 rtol = 1e-12            # all of it went to vapour
        @test fl.LE ≈ fl.E * latent_heat_sublim(265.0) rtol = 0.01
        @test abs(-H0 - (b.surface_heat - b.bottom_heat + b.advected_enthalpy)) < 1e-9 * abs(H0)
    end

    @testset "Surface physics: radiative cooling, melt, insulation" begin
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        # Clear night: the snow surface decouples and cools below the air.
        p = LayeredSnowpack(depth = 0.5, density = 200.0, T = 263.0)
        fnight = SurfaceForcing(sw_down = 0.0, lw_down = 200.0, T_air = 263.15, q_air = 0.0012,
                                U = 5.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
        local fl
        for _ in 1:24
            fl = step_surface!(p, fnight, 1800.0)
        end
        @test surface_temperature(p) < 263.15 - 3
        @test fl.Rn < 0
        @test fl.H < 0                                  # heat flows down from the air
        @test abs(fl.H) > abs(fl.G)                     # ... and, with wind, dominates
        @info @sprintf("Clear night after 12 h: T_s = %.2f K (air 263.15), Rn = %.1f, H = %.1f, LE = %.1f, G = %.1f W m⁻²",
                       surface_temperature(p), fl.Rn, fl.H, fl.LE, fl.G)
        # Warm sunny day: the skin is pinned at the melting point and the pack melts.
        p = LayeredSnowpack(depth = 0.5, density = 300.0, T = 272.0)
        fday = SurfaceForcing(sw_down = 500.0, lw_down = 320.0, T_air = 280.0, q_air = 0.006,
                              U = 4.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
        D0 = snow_depth(p)
        for _ in 1:24
            step_surface!(p, fday, 1800.0)
            @test surface_temperature(p) <= Tf_snow
        end
        @test surface_temperature(p) == Tf_snow
        @test p.budget.melt > 10
        @test p.budget.runoff ≈ p.budget.melt rtol = 1e-12
        @test snow_depth(p) < D0
        @info @sprintf("Melt day: %.1f kg m⁻² melted in 12 h, depth %.3f → %.3f m",
                       p.budget.melt, D0, snow_depth(p))
        # Insulation: ground at 0 °C under 10 cm vs 1 m of snow, air -20 °C.
        fcold = SurfaceForcing(sw_down = 0.0, lw_down = 0.85σ * 253.15^4, T_air = 253.15,
                               q_air = 0.0006, U = 3.0, p = 90000.0, z_ref = 10.0, rain = 0.0, snow = 0.0)
        loss = map((0.1, 1.0)) do D
            q = LayeredSnowpack(depth = D, density = 250.0, T = 263.15,
                                bottom = SnowGroundTemperature(273.15),
                                compaction = Anderson1976Compaction(destructive = nothing, η0 = Inf))
            for _ in 1:240                                  # 10 days
                step_surface!(q, fcold, 3600.0)
            end
            -snow_heat_conduction!(q, 1.0)[2]               # ground → snow flux now
        end
        @test loss[1] > 5 * loss[2] > 0
        @info @sprintf("Ground heat loss through 0.1 m / 1.0 m of snow: %.1f / %.1f W m⁻²", loss...)
    end

    @testset "Density evolution vs Sturm et al. (2010) alpine climatology" begin
        # Sturm et al. (2010), J. Hydrometeorol. 11, 1380, Eq. 6 and Table 4:
        # ρ(h, DOY) = (ρmax - ρ0)[1 - exp(-k1 h - k2 DOY)] + ρ0, h in cm,
        # ρ in g cm⁻³, DOY from -92 (1 Oct) through 181 (30 Jun), 1 Jan = 1.
        # Alpine class: ρmax = 0.5975, ρ0 = 0.2237, k1 = 0.0012, k2 = 0.0038.
        # A regression on 25 688 observed depth-density-SWE measurements
        # (US, Canada, Switzerland) — the observed seasonal densification.
        # DOY has no 0: 31 Dec = -1, 1 Jan = 1 (the sampled days below
        # straddle this by at most one day).
        sturm(h_cm, doy) = 1000 * ((0.5975 - 0.2237) * (1 - exp(-0.0012h_cm - 0.0038doy)) + 0.2237)
        # Synthetic forcing (NOT observations): 500 mm SWE falling evenly
        # 1 Nov - 31 Mar, air -5 ± 3 °C diurnal, clear-sky-ish longwave,
        # 400 W m⁻² peak sun, ground at 0 °C.
        σ = DEFAULT_SURFACE_CONSTANTS.σ_SB
        results = Dict{String,Vector{Float64}}()
        for (name, comp) in (("Anderson1976", Anderson1976Compaction()),
                             ("Vionnet2012", Vionnet2012Compaction()))
            p = LayeredSnowpack(compaction = comp, bottom = SnowGroundTemperature(273.15))
            rate = 500.0 / (151 * 86400)
            errs = Float64[]
            ρs_model = Float64[]
            for d in 0:150, h in 0:23
                Ta = 268.15 + 3sin(π * (h - 9) / 12)
                sw = 6 <= h <= 18 ? 400sin(π * (h - 6) / 12) : 0.0
                f = SurfaceForcing(sw_down = sw, lw_down = 0.80σ * 268.15^4, T_air = Ta,
                                   q_air = 0.0025, U = 3.0, p = 85000.0, z_ref = 10.0,
                                   rain = 0.0, snow = rate)
                step_surface!(p, f, 3600.0)
                if h == 23 && (d + 1) % 30 == 0
                    doy = -61 + d + 1
                    ρs = sturm(100snow_depth(p), doy)
                    push!(errs, snow_bulk_density(p) / ρs - 1)
                    push!(ρs_model, snow_bulk_density(p))
                    @info @sprintf("%-12s DOY %4d: depth %.2f m, ρ̄ = %.0f kg m⁻³, Sturm alpine %.0f (%+.0f %%)",
                                   name, doy, snow_depth(p), snow_bulk_density(p), ρs, 100errs[end])
                end
            end
            @test p.budget.melt == 0                           # a dry-snow season
            @test snow_water_equivalent(p) ≈ 500.0 - p.budget.sublimation rtol = 1e-10
            results[name] = errs
            # Bulk density rises through the season, as observed, and stays
            # within 25 % of the climatology (Sturm et al. quote a local
            # natural variability of 7-23 % of the mean bulk density).
            @test issorted(ρs_model)
            @test all(abs.(errs) .< 0.25)
        end
    end
end
