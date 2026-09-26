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

@testset "Modified van Genuchten (air entry)" begin
    classic = carsel_parrish_texture(:loam; ψ_e = 0.0)
    mod = carsel_parrish_texture(:loam)                   # ψ_e = -2 cm
    @test air_entry_potential(mod) == -0.02
    # ψ_e = 0 is exactly the classical model
    v0 = VanGenuchtenMualem(θ_r = 0.078, θ_s = 0.43, α = 3.6, n = 1.56, K_s = 1.04 / 360000)
    @test soil_hydraulic_conductivity(classic, -0.7) ≈ soil_hydraulic_conductivity(v0, -0.7) rtol = 1e-12
    # continuity at the air-entry potential, and the removal of the cusp:
    # the classical K has lost 40 % of K_s within 2 cm of saturation
    @test soil_hydraulic_conductivity(mod, -0.02 - 1e-9) ≈ mod.K_s rtol = 1e-4
    @test soil_water_content(mod, -0.02 - 1e-9) ≈ mod.θ_s rtol = 1e-6
    @test soil_hydraulic_conductivity(classic, -0.02) / classic.K_s < 0.65
    # the retention curve itself moves by < 1 % of the pore space
    for ψ in (-0.1, -1.0, -10.0)
        @test abs(soil_water_content(mod, ψ) - soil_water_content(classic, ψ)) < 0.01 * mod.θ_s
    end
end

# Exact solution of Richards' equation for the Gardner soil (Srivastava &
# Yeh 1991; Tracy 2006, 1-D form): with K = K_s e^{αψ} and θ linear in K,
# c ∂K/∂t = ∂²K/∂ζ² + α ∂K/∂ζ, c = α(θ_s - θ_r)/K_s, ζ = L - z the height
# above the base. Initially K = K_r everywhere, the base held at K_r and
# the top raised to K_t at t = 0:
#   K = K_r + A (1 - e^{-αζ})/(1 - e^{-αL})
#       + e^{-αζ/2} Σ_n b_n sin(k_n ζ) e^{-λ_n t/c},
#   A = K_t - K_r, k_n = nπ/L, λ_n = k_n² + α²/4,
#   b_n = 2 A k_n (-1)^n e^{αL/2} / (L λ_n)   (derived in docs/src/theory/soil_water.md).
function gardner_exact_θ(h::GardnerExponential, L, ψ_init, ψ_top, z, t; nterms = 4000)
    α, Ks = h.α, h.K_s
    c = α * (h.θ_s - h.θ_r) / Ks
    Kr = Ks * exp(α * ψ_init)
    A = Ks * exp(α * ψ_top) - Kr
    ζ = L - z
    K = Kr + A * (1 - exp(-α * ζ)) / (1 - exp(-α * L))
    s = 0.0
    for n in 1:nterms
        k = n * π / L
        λ = k^2 + α^2 / 4
        s += 2A * k * (-1)^n * exp(α * L / 2) / (L * λ) * sin(k * ζ) * exp(-λ * t / c)
    end
    K += exp(-α * ζ / 2) * s
    return h.θ_r + (h.θ_s - h.θ_r) * K / Ks
end

@testset "Exact Gardner infiltration benchmark" begin
    h = GardnerExponential(θ_r = 0.05, θ_s = 0.40, α = 2.0, K_s = 1e-5)
    L, ψi = 1.0, -5.0
    # the series reproduces its own boundary and initial data
    @test gardner_exact_θ(h, L, ψi, 0.0, 0.0, 600.0) ≈ h.θ_s rtol = 1e-10
    @test gardner_exact_θ(h, L, ψi, 0.0, 0.8, 1.0) ≈ soil_water_content(h, ψi) atol = 1e-6
    run(N, mean, dt, tend) = begin
        local col = SoilWaterColumn(uniform_soil_grid(L, N), h; ψ = ψi, top = HeadTop(0.0),
                              bottom = HeadBottom(ψi), temperature_dependent = false,
                              interface_mean = mean)
        S0 = soil_water_storage(col)
        while col.budget.time < tend - 1e-6
            step_soil_water!(col, dt)
        end
        # water balance to round-off on every run
        @test soil_water_storage(col) - S0 ≈
              col.budget.infiltration - col.budget.drainage atol = 1e-14
        maximum(abs.(col.θ_liq .- [gardner_exact_θ(h, L, ψi, 0.0, z, tend) for z in col.grid.z]))
    end
    # Space: after 12 h (smooth profile, time error negligible at dt = 30 s)
    eL50, eL100 = run(50, :logarithmic, 30.0, 12 * 3600.0), run(100, :logarithmic, 30.0, 12 * 3600.0)
    eA50, eA100 = run(50, :arithmetic, 30.0, 12 * 3600.0), run(100, :arithmetic, 30.0, 12 * 3600.0)
    # Time: after 3 h, sharp front, fine grid, backward Euler
    eT = [run(200, :logarithmic, dt, 3 * 3600.0) for dt in (10.0, 5.0, 2.5)]
    @printf("  Gardner exact, 12 h max |Δθ|: log-mean N=50 %.2e, N=100 %.2e; arithmetic N=50 %.2e, N=100 %.2e\n",
            eL50, eL100, eA50, eA100)
    @printf("  Gardner exact, 3 h, N=200 max |Δθ|: dt=10 s %.2e, 5 s %.2e, 2.5 s %.2e\n", eT...)
    @test eL100 < 1e-4                    # < 0.03 % of θ_s - θ_r
    @test eL50 / eL100 > 3.0              # second order in space
    @test 1.8 < eA50 / eA100 < 2.3        # arithmetic mean: first order
    @test eA100 > 10eL100
    @test eT[1] / eT[2] > 1.7 && eT[2] / eT[3] > 1.7     # first order in time
    @test eT[3] < 5e-5
end

@testset "Sorptivity: horizontal absorption" begin
    # Gravity off, surface held saturated: Boltzmann similarity makes the
    # cumulative absorption exactly I = S √t.
    # (a) Gardner soil = linear diffusion: S = 2 Δθ √(D/π) exactly.
    g = GardnerExponential(θ_r = 0.05, θ_s = 0.40, α = 2.0, K_s = 1e-6)
    ψi = -3.0
    D = soil_water_diffusivity(g, -1.0)
    Δθ = g.θ_s - soil_water_content(g, ψi)
    S_exact = 2Δθ * sqrt(D / π)
    col = SoilWaterColumn(uniform_soil_grid(1.0, 200), g; ψ = ψi, top = HeadTop(0.0),
                          bottom = ZeroFluxBottom(), gravity = false, temperature_dependent = false)
    ratios = Float64[]
    for t in (2 * 3600.0, 6 * 3600.0)
        while col.budget.time < t - 1e-6
            step_soil_water!(col, 30.0)
        end
        push!(ratios, col.budget.infiltration / sqrt(t) / S_exact)
    end
    @printf("  Gardner absorption: S_num/S_exact = %.5f (2 h), %.5f (6 h)\n", ratios...)
    @test all(r -> abs(r - 1) < 2e-3, ratios)
    # (b) van Genuchten loam: against Parlange (1975), and self-similar.
    v = carsel_parrish_texture(:loam)
    ψi = -10.0
    Sp = parlange_sorptivity(v, ψi)
    col = SoilWaterColumn(uniform_soil_grid(0.6, 480), v; ψ = ψi, top = HeadTop(0.0),
                          bottom = ZeroFluxBottom(), gravity = false, temperature_dependent = false)
    S = Float64[]
    for t in (3600.0, 6 * 3600.0)
        while col.budget.time < t - 1e-6
            step_soil_water!(col, 60.0)
        end
        push!(S, col.budget.infiltration / sqrt(t))
    end
    @printf("  vG loam absorption: S_num/S_Parlange = %.4f (1 h), %.4f (6 h)\n", (S ./ Sp)...)
    @test abs(S[2] / S[1] - 1) < 0.01            # I ∝ √t
    @test abs(S[2] / Sp - 1) < 0.02              # Parlange is good to a few %
    θi = soil_water_content(v, ψi)
    # rigorous bounds: 2∫(θ-θ_i)K dψ ≤ S² ≤ 2(θ_s-θ_i)∫K dψ
    ψs = -exp.(range(log(10.0), log(1e-6), length = 20001))
    Kint(f) = sum((ψs[k+1] - ψs[k]) * (f(ψs[k]) + f(ψs[k+1])) / 2 for k in 1:length(ψs)-1)
    lo = sqrt(2 * Kint(ψ -> (soil_water_content(v, ψ) - θi) * soil_hydraulic_conductivity(v, ψ)))
    hi = sqrt(2 * (v.θ_s - θi) * Kint(ψ -> soil_hydraulic_conductivity(v, ψ)))
    @test lo < S[2] < hi
end

@testset "Philip infiltration: vertical vs horizontal" begin
    # Philip (1957): I(t) = S√t + A t + …, with the same S as horizontal
    # absorption and K_s/3 ≤ A ≤ 2K_s/3; at long times the ponded
    # infiltration rate falls to K_s.
    v = carsel_parrish_texture(:loam)
    ψi = -10.0
    mk(grav) = SoilWaterColumn(uniform_soil_grid(grav ? 3.0 : 0.6, grav ? 600 : 240), v;
                               ψ = ψi, top = HeadTop(0.0),
                               bottom = grav ? FreeDrainage() : ZeroFluxBottom(),
                               gravity = grav, temperature_dependent = false)
    cv, chz = mk(true), mk(false)
    A = Float64[]
    ratio = Float64[]
    for t in (600.0, 3600.0, 3 * 3600.0)
        for c in (cv, chz)
            while c.budget.time < t - 1e-6
                step_soil_water!(c, 60.0)
            end
        end
        push!(ratio, cv.budget.infiltration / chz.budget.infiltration)
        push!(A, (cv.budget.infiltration - chz.budget.infiltration) / t / v.K_s)
    end
    @printf("  Philip: I_v/I_h = %.3f, %.3f, %.3f; A/K_s = %.3f, %.3f, %.3f (10 min, 1 h, 3 h)\n",
            ratio..., A...)
    @test issorted(ratio) && ratio[1] < 1.10                 # gravity matters later
    @test all(a -> 1 / 3 <= a <= 2 / 3, A)
    col = SoilWaterColumn(uniform_soil_grid(5.0, 250), v; ψ = ψi, top = HeadTop(0.0),
                          temperature_dependent = false)
    rates = [step_soil_water!(col, 86400.0).infiltration / 86400 / v.K_s for _ in 1:5]
    @printf("  ponded infiltration rate / K_s, days 1-5: %s\n", string(round.(rates, digits = 4)))
    @test issorted(rates, rev = true)
    @test 1 < rates[end] < 1.01
end

@testset "Steady states: hydrostatic equilibrium and unit-gradient drainage" begin
    # Layered CLM5 column on an impermeable base, no surface flux: the only
    # steady state is hydrostatic, total head ψ - z uniform.
    g = clm5_soil_grid()
    hyd = [i <= 8 ? clapp_hornberger_texture(:loam) : clapp_hornberger_texture(:sandy_loam) for i in 1:20]
    col = SoilWaterColumn(g, hyd; θ = 0.35, bottom = ZeroFluxBottom())
    S0 = soil_water_storage(col)
    for _ in 1:73
        step_soil_water!(col, 5 * 86400.0)
    end
    H = col.ψ .- col.grid.z
    @printf("  hydrostatic (loam over sandy loam, 8.6 m, 1 yr): head spread %.1e m, ΔS = %.1e m\n",
            maximum(H) - minimum(H), soil_water_storage(col) - S0)
    @test maximum(H) - minimum(H) < 1e-8
    @test soil_water_storage(col) ≈ S0 atol = 1e-12
    # Free drainage under a steady input q < K_s: unit gradient, K(θ) = q,
    # so θ = θ_s (q / (K_s f_T))^{1/(2b+3)} — and colder water, being more
    # viscous, needs a wetter soil to carry the same flux.
    ch = clapp_hornberger_texture(:loam)
    q = 0.1 * ch.K_s
    θss = Float64[]
    for Ts in (293.15, 278.15)
        c = SoilWaterColumn(uniform_soil_grid(2.0, 40), ch; θ = 0.2, T_soil = Ts)
        for _ in 1:200
            step_soil_water!(c, 86400.0; water_input = q)
        end
        r = step_soil_water!(c, 86400.0; water_input = q)
        θex = ch.θ_s * (q / (ch.K_s * viscosity_conductivity_factor(Ts)))^(1 / (2ch.b + 3))
        @test maximum(abs.(c.θ_liq .- θex)) < 1e-6
        @test r.drainage ≈ q * 86400 rtol = 1e-6
        push!(θss, c.θ_liq[20])
    end
    @printf("  unit-gradient θ: %.5f at 20 °C, %.5f at 5 °C\n", θss...)
    @test θss[2] > θss[1]
end

@testset "Atmospheric boundary: ponding, runoff and soil-limited evaporation" begin
    v = carsel_parrish_texture(:loam)
    t_pond = Float64[]
    Iatm = 0.0
    for r in (0.5, 2.0, 5.0, 20.0)
        c = SoilWaterColumn(uniform_soil_grid(2.0, 100), v; ψ = -3.0)
        P = r * v.K_s
        tp = Inf
        worst = 0.0
        for _ in 1:720
            st = step_soil_water!(c, 60.0; water_input = P)
            # every step: rain = infiltration + runoff, exactly
            worst = max(worst, abs(st.infiltration + st.runoff - P * 60))
            st.runoff > 0 && isinf(tp) && (tp = c.budget.time)
        end
        @test worst < 1e-15
        push!(t_pond, tp)
        Iatm = c.budget.infiltration
    end
    @printf("  time to ponding (s) for rain = 0.5, 2, 5, 20 K_s: %s\n", string(t_pond))
    @test isinf(t_pond[1])                 # rain below K_s never ponds
    @test issorted(t_pond[2:end], rev = true) && isfinite(t_pond[2])
    # after early ponding the soil takes what a ponded surface would
    c = SoilWaterColumn(uniform_soil_grid(2.0, 100), v; ψ = -3.0, top = HeadTop(0.0))
    for _ in 1:720
        step_soil_water!(c, 60.0)
    end
    @printf("  12 h infiltration: rain 20 K_s %.5f m, ponded surface %.5f m\n", Iatm, c.budget.infiltration)
    @test Iatm ≈ c.budget.infiltration rtol = 5e-3
    # Evaporation from a column on bedrock: stage 1 at the potential rate,
    # then stage 2 limited by the soil, falling steadily.
    c = SoilWaterColumn(uniform_soil_grid(1.0, 100), v; ψ = -1.0, bottom = ZeroFluxBottom())
    Ep = 5e-3 / 86400
    S0 = soil_water_storage(c)
    E = [step_soil_water!(c, 86400.0; evaporation = Ep).evaporation / (Ep * 86400) for _ in 1:20]
    @printf("  E/E_pot on days 1, 2, 5, 10, 20: %.3f %.3f %.3f %.3f %.3f\n", E[[1, 2, 5, 10, 20]]...)
    @test E[1] ≈ 1 rtol = 1e-6
    @test issorted(E, rev = true) && E[end] < 0.1
    @test soil_water_storage(c) - S0 ≈ -c.budget.evaporation atol = 1e-14
end

@testset "Exact water conservation under mixed forcing" begin
    # CLM5 column, per-layer Cosby soils, rain bursts, evaporation, dew,
    # root extraction, free drainage: every step must close its budget.
    g = clm5_soil_grid()
    hyd = [cosby_clapp_hornberger(70 - 3i, 10 + i) for i in 1:20]
    col = SoilWaterColumn(g, hyd; θ = [0.25 + 0.005i for i in 1:20],
                          T_soil = [285.0 - 0.3i for i in 1:20])
    root = [exp(-g.z[i] / 0.3) for i in 1:20]
    root ./= sum(root)
    S0 = soil_water_storage(col)
    worst = 0.0
    for k in 1:24 * 30
        hour = mod(k, 24)
        P = (mod(k, 97) < 3) ? 2e-5 : 0.0              # 72 mm/h bursts
        E = 6 < hour < 18 ? 8e-8 : -5e-9               # day evaporation, night dew
        sink = (6 < hour < 18 ? 5e-8 : 0.0) .* root
        S = soil_water_storage(col)
        st = step_soil_water!(col, 3600.0; water_input = P, evaporation = E, sink = sink)
        worst = max(worst, abs(st.storage_change - (st.infiltration - st.drainage - st.extraction)),
                    abs(soil_water_storage(col) - S - st.storage_change),
                    abs(P * 3600 - st.evaporation - (st.infiltration + st.runoff)))
    end
    b = col.budget
    @printf("  30-day mixed forcing: worst per-step imbalance %.1e m; cumulative residual %.1e m of %.3f m infiltrated\n",
            worst, soil_water_storage(col) - S0 - (b.infiltration - b.drainage - b.extraction), b.infiltration)
    @test worst < 1e-13
    @test soil_water_storage(col) - S0 ≈ b.infiltration - b.drainage - b.extraction atol = 1e-12
    @test b.runoff > 0 && b.drainage > 0 && b.extraction > 0
end

@testset "Celia et al. (1990) problem: mass balance ratio" begin
    # Celia et al. (1990), van Genuchten test soil: α = 0.0335 cm⁻¹,
    # θ_s = 0.368, θ_r = 0.102, n = 2, K_s = 0.00922 cm/s; 100 cm column
    # at ψ = -1000 cm, top raised to -75 cm, bottom held at -1000 cm, 1 day.
    # Their point: the ψ-form loses mass at large Δt, the mixed form
    # conserves it. Here the mass-balance ratio at Δt = 1 h, Δz = 2.5 cm:
    v = VanGenuchtenMualem(θ_r = 0.102, θ_s = 0.368, α = 3.35, n = 2.0, K_s = 9.22e-5)
    run(N, dt) = begin
        local c = SoilWaterColumn(uniform_soil_grid(1.0, N), v; ψ = -10.0, top = HeadTop(-0.75),
                            bottom = HeadBottom(-10.0), temperature_dependent = false)
        S0 = soil_water_storage(c)
        for _ in 1:round(Int, 86400 / dt)
            step_soil_water!(c, dt)
        end
        c, (soil_water_storage(c) - S0) / (c.budget.infiltration - c.budget.drainage)
    end
    c, MB = run(40, 3600.0)
    cref, _ = run(400, 60.0)
    # wetting-front depth: where θ crosses the midpoint of its range
    θmid = (soil_water_content(v, -0.75) + soil_water_content(v, -10.0)) / 2
    front(c) = c.grid.z[findfirst(<(θmid), c.θ_liq)]
    @printf("  Celia problem: mass-balance ratio %.15f; front at %.1f cm (Δz = 2.5 cm) vs %.1f cm (Δz = 0.25 cm)\n",
            MB, 100front(c), 100front(cref))
    @test MB ≈ 1 atol = 1e-12
    @test abs(front(c) - front(cref)) <= 0.03
end

@testset "Surface-interface coupling and extension hooks" begin
    v = carsel_parrish_texture(:loam)
    # SurfaceForcing / SurfaceFluxes in kg m⁻² s⁻¹ → m s⁻¹ via the contract ρ_w
    c1 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -2.0)
    c2 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -2.0)
    f = SurfaceForcing(rain = 1e-3)
    fl = SurfaceFluxes(300.0, 50.0, 100.0, 150.0, 0.1, 4e-5)
    s1 = step_soil_water!(c1, f, fl, 3600.0)
    s2 = step_soil_water!(c2, 3600.0; water_input = 1e-6, evaporation = 4e-8)
    @test s1.infiltration == s2.infiltration && c1.ψ == c2.ψ
    # β (Lee & Pielke 1992): 1 at field capacity and wetter, 1/4 at half of it
    θfc = soil_field_capacity(v)
    @test θfc ≈ soil_water_content(v, -33e3 / (1000 * grav(DEFAULT_SURFACE_CONSTANTS)))
    set_water_content!(c1, θfc + 0.01)
    @test moisture_availability(c1) == 1
    set_water_content!(c1, θfc / 2)
    @test moisture_availability(c1) ≈ 0.25
    set_water_content!(c1, 0.08)                     # θ_r < 0.08 < θ_fc/2
    @test θfc / 2 > 0.08
    @test 0 < moisture_availability(c1) < 0.25
    # Ice-impedance hook (#16): a subtype scales K layer by layer.
    struct HalfIfIce <: AbstractIceImpedance end
    SC.ice_impedance_factor(::HalfIfIce, θi, θs) = θi > 0 ? 0.5 : 1.0
    c3 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -2.0, ice_impedance = HalfIfIce())
    K0 = soil_layer_conductivity(c3, 5)
    c3.θ_ice[5] = 0.05
    @test soil_layer_conductivity(c3, 5) ≈ K0 / 2
    # ... and ice counts as water in the storage (ρ_ice/ρ_w)
    @test soil_water_storage(c3) ≈ sum(c3.grid.dz .* c3.θ_liq) + 0.917 * 0.05 * c3.grid.dz[5]
    # Temperature: colder layers conduct less, by the viscosity ratio
    c4 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -2.0, T_soil = 273.15)
    @test soil_layer_conductivity(c4, 5) / soil_layer_conductivity(c2, 5) ≈
          viscosity_conductivity_factor(273.15) * soil_hydraulic_conductivity(v, c4.ψ[5]) /
          soil_hydraulic_conductivity(v, c2.ψ[5])
    # Stateful bottom boundary (#17): a linear reservoir fed by drainage.
    mutable struct Reservoir <: AbstractSoilBottomBoundary
        storage::Float64
    end
    SC.bottom_flux_linearization(::Reservoir, col, ψN, KN) = (col.gravity * KN, 0.0)
    SC.finalize_bottom!(r::Reservoir, col, q, dt) = (r.storage += q * dt; nothing)
    c5 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -0.3, bottom = Reservoir(0.0))
    for _ in 1:48
        step_soil_water!(c5, 3600.0)
    end
    @test c5.bottom.storage ≈ c5.budget.drainage rtol = 1e-12
    @test c5.bottom.storage > 0
end

end # SoilColumn testset
