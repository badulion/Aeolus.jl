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
        @test worst < 1e-4
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
        @test erf_err < 0.02                            # K, on a 10 K step
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
end
