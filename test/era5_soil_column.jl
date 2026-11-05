# Era V, Phase 46 (#15) — SoilColumn: CLM-style soil column with
# Richards-equation water transport.
#
# The solver is checked against exact solutions of Richards' equation
# (Gardner soil), exact similarity laws (horizontal absorption),
# closed-form steady states and exact water-mass conservation.

using Test
using Aeolus
using Aeolus.SoilColumn
using Aeolus.SurfaceExchange: DEFAULT_SURFACE_CONSTANTS, SurfaceForcing, SurfaceFluxes,
                              moisture_availability, grav

const SC = Aeolus.SoilColumn

@testset "Era V — SoilColumn (#15)" begin

@testset "Layer structure and hydraulic functions" begin
    # CLM5.0 Technical Note, Table 2.2.3: layer 10 node, thickness, interface
    g = clm5_soil_grid()
    @test length(g) == 20
    @test maximum(abs.((g.z[10], g.dz[10], g.zh[11]) .- (1.360, 0.320, 1.520))) < 5e-4
    # retention inverse and C = dθ/dψ by centred differences, three models
    models = (clapp_hornberger_texture(:loam), carsel_parrish_texture(:loam),
              GardnerExponential(θ_r = 0.05, θ_s = 0.40, α = 2.0, K_s = 1e-5))
    ψsets = ([-1e3, -3.0, -0.6], [-1e3, -3.0, -0.3], [-5.0, -1.0, -0.05])
    worst_inv = 0.0; worst_C = 0.0
    for (h, ψs) in zip(models, ψsets), ψ in ψs
        worst_inv = max(worst_inv, abs(soil_matric_potential(h, soil_water_content(h, ψ)) / ψ - 1))
        δ = 1e-6 * abs(ψ)
        dθ = (soil_water_content(h, ψ + δ) - soil_water_content(h, ψ - δ)) / 2δ
        worst_C = max(worst_C, abs(soil_water_capacity(h, ψ) - dθ) / max(abs(dθ), 1e-7))
    end
    @test worst_inv < 1e-10
    @test worst_C < 1e-5
    # Clapp-Hornberger K = K_s (θ/θ_s)^(2b+3); Gardner D = K_s/(α(θ_s − θ_r))
    ch = models[1]
    @test soil_hydraulic_conductivity(ch, soil_matric_potential(ch, 0.25)) ≈
          ch.K_s * (0.25 / ch.θ_s)^(2ch.b + 3) rtol = 1e-12
    @test soil_water_diffusivity(models[3], -2.0) ≈ 1e-5 / (2.0 * 0.35) rtol = 1e-12
    # viscosity of water (CRC Handbook): K(0 °C)/K(20 °C) = 1.0016/1.7914
    @test viscosity_conductivity_factor(273.15) ≈ 1.0016 / 1.7914 rtol = 3e-3
    # Parlange sorptivity, constant D: S² = 1.5 D Δθ² exactly
    gd = models[3]
    Δθ = soil_porosity(gd) - soil_water_content(gd, -5.0)
    @test parlange_sorptivity(gd, -5.0) ≈ sqrt(1.5 * soil_water_diffusivity(gd, -1.0)) * Δθ rtol = 1e-4
end

# Exact solution of Richards' equation for the Gardner soil (Srivastava &
# Yeh 1991; Tracy 2006, 1-D form); derived in docs/src/theory/soil_water.md.
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
    balance = 0.0
    run(N, dt, tend) = begin
        local col = SoilWaterColumn(uniform_soil_grid(L, N), h; ψ = ψi, top = HeadTop(0.0),
                              bottom = HeadBottom(ψi), temperature_dependent = false,
                              interface_mean = :logarithmic)
        S0 = soil_water_storage(col)
        while col.budget.time < tend - 1e-6
            step_soil_water!(col, dt)
        end
        balance = max(balance, abs(soil_water_storage(col) - S0 -
                                   (col.budget.infiltration - col.budget.drainage)))
        maximum(abs.(col.θ_liq .- [gardner_exact_θ(h, L, ψi, 0.0, z, tend) for z in col.grid.z]))
    end
    # after 12 h (smooth profile; time error negligible at dt = 30 s)
    e50, e100 = run(50, 30.0, 12 * 3600.0), run(100, 30.0, 12 * 3600.0)
    @test balance < 1e-14
    @test e100 < 1e-4                    # < 0.03 % of θ_s − θ_r
    @test e50 / e100 > 3.0               # second order in space (log mean)
end

@testset "Sorptivity: horizontal absorption I = S √t" begin
    # Gravity off, surface saturated, Gardner soil = linear diffusion:
    # S = 2 Δθ √(D/π) exactly.
    g = GardnerExponential(θ_r = 0.05, θ_s = 0.40, α = 2.0, K_s = 1e-6)
    ψi = -3.0
    D = soil_water_diffusivity(g, -1.0)
    S_exact = 2(g.θ_s - soil_water_content(g, ψi)) * sqrt(D / π)
    col = SoilWaterColumn(uniform_soil_grid(1.0, 200), g; ψ = ψi, top = HeadTop(0.0),
                          bottom = ZeroFluxBottom(), gravity = false, temperature_dependent = false)
    while col.budget.time < 2 * 3600.0 - 1e-6
        step_soil_water!(col, 30.0)
    end
    @test abs(col.budget.infiltration / sqrt(2 * 3600.0) / S_exact - 1) < 2e-3
end

@testset "Steady states: hydrostatic and unit-gradient drainage" begin
    # layered column on an impermeable base: total head ψ − z uniform
    g = clm5_soil_grid()
    hyd = [i <= 8 ? clapp_hornberger_texture(:loam) : clapp_hornberger_texture(:sandy_loam) for i in 1:20]
    col = SoilWaterColumn(g, hyd; θ = 0.35, bottom = ZeroFluxBottom())
    S0 = soil_water_storage(col)
    for _ in 1:73
        step_soil_water!(col, 5 * 86400.0)
    end
    H = col.ψ .- col.grid.z
    @test maximum(H) - minimum(H) < 1e-8
    @test soil_water_storage(col) ≈ S0 atol = 1e-12
    # free drainage under q < K_s: K(θ) = q, θ = θ_s (q/(K_s f_T))^{1/(2b+3)}
    ch = clapp_hornberger_texture(:loam)
    q = 0.1 * ch.K_s
    Ts = 278.15
    c = SoilWaterColumn(uniform_soil_grid(2.0, 40), ch; θ = 0.2, T_soil = Ts)
    for _ in 1:200
        step_soil_water!(c, 86400.0; water_input = q)
    end
    θex = ch.θ_s * (q / (ch.K_s * viscosity_conductivity_factor(Ts)))^(1 / (2ch.b + 3))
    @test maximum(abs.(c.θ_liq .- θex)) < 1e-6
end

@testset "Atmospheric boundary: ponding, runoff, soil-limited evaporation" begin
    v = carsel_parrish_texture(:loam)
    c = SoilWaterColumn(uniform_soil_grid(2.0, 100), v; ψ = -3.0)
    P = 5.0 * v.K_s
    worst = 0.0
    for _ in 1:240
        st = step_soil_water!(c, 60.0; water_input = P)
        worst = max(worst, abs(st.infiltration + st.runoff - P * 60))   # rain = infiltration + runoff
    end
    @test worst < 1e-15
    @test c.budget.runoff > 0
    # evaporation on bedrock: stage 1 at the potential rate, then falling
    c = SoilWaterColumn(uniform_soil_grid(1.0, 100), v; ψ = -1.0, bottom = ZeroFluxBottom())
    Ep = 5e-3 / 86400
    S0 = soil_water_storage(c)
    E = [step_soil_water!(c, 86400.0; evaporation = Ep).evaporation / (Ep * 86400) for _ in 1:20]
    @test E[1] ≈ 1 rtol = 1e-6
    @test issorted(E, rev = true) && E[end] < 0.1
    @test soil_water_storage(c) - S0 ≈ -c.budget.evaporation atol = 1e-14
end

@testset "Exact water conservation under mixed forcing" begin
    # per-layer Cosby soils, rain bursts, evaporation, dew, root extraction,
    # free drainage: every step closes its budget
    g = clm5_soil_grid()
    hyd = [cosby_clapp_hornberger(70 - 3i, 10 + i) for i in 1:20]
    col = SoilWaterColumn(g, hyd; θ = [0.25 + 0.005i for i in 1:20],
                          T_soil = [285.0 - 0.3i for i in 1:20])
    root = [exp(-g.z[i] / 0.3) for i in 1:20]
    root ./= sum(root)
    S0 = soil_water_storage(col)
    worst = 0.0
    for k in 1:24 * 5
        hour = mod(k, 24)
        P = (mod(k, 97) < 3) ? 2e-5 : 0.0              # 72 mm/h bursts
        E = 6 < hour < 18 ? 8e-8 : -5e-9
        sink = (6 < hour < 18 ? 5e-8 : 0.0) .* root
        S = soil_water_storage(col)
        st = step_soil_water!(col, 3600.0; water_input = P, evaporation = E, sink = sink)
        worst = max(worst, abs(st.storage_change - (st.infiltration - st.drainage - st.extraction)),
                    abs(soil_water_storage(col) - S - st.storage_change),
                    abs(P * 3600 - st.evaporation - (st.infiltration + st.runoff)))
    end
    b = col.budget
    @test worst < 1e-13
    @test soil_water_storage(col) - S0 ≈ b.infiltration - b.drainage - b.extraction atol = 1e-12
    @test b.runoff > 0 && b.extraction > 0
end

@testset "Celia et al. (1990) problem: mass balance ratio" begin
    # van Genuchten test soil, 100 cm column at ψ = −1000 cm, top raised to
    # −75 cm, 1 day at Δt = 1 h, Δz = 2.5 cm: the mixed form conserves mass.
    v = VanGenuchtenMualem(θ_r = 0.102, θ_s = 0.368, α = 3.35, n = 2.0, K_s = 9.22e-5)
    c = SoilWaterColumn(uniform_soil_grid(1.0, 40), v; ψ = -10.0, top = HeadTop(-0.75),
                        bottom = HeadBottom(-10.0), temperature_dependent = false)
    S0 = soil_water_storage(c)
    for _ in 1:24
        step_soil_water!(c, 3600.0)
    end
    @test (soil_water_storage(c) - S0) / (c.budget.infiltration - c.budget.drainage) ≈ 1 atol = 1e-12
end

@testset "Surface-interface coupling and extension hooks" begin
    v = carsel_parrish_texture(:loam)
    # SurfaceForcing / SurfaceFluxes in kg m⁻² s⁻¹ → m s⁻¹ via ρ_w
    c1 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -2.0)
    c2 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -2.0)
    s1 = step_soil_water!(c1, SurfaceForcing(rain = 1e-3), SurfaceFluxes(300.0, 50.0, 100.0, 150.0, 0.1, 4e-5), 3600.0)
    s2 = step_soil_water!(c2, 3600.0; water_input = 1e-6, evaporation = 4e-8)
    @test s1.infiltration == s2.infiltration && c1.ψ == c2.ψ
    # β (Lee & Pielke 1992): 1/4 at half of field capacity
    set_water_content!(c1, soil_field_capacity(v) / 2)
    @test moisture_availability(c1) ≈ 0.25
    # ice-impedance hook (#16)
    struct HalfIfIce <: AbstractIceImpedance end
    SC.ice_impedance_factor(::HalfIfIce, θi, θs) = θi > 0 ? 0.5 : 1.0
    c3 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -2.0, ice_impedance = HalfIfIce())
    K0 = soil_layer_conductivity(c3, 5)
    c3.θ_ice[5] = 0.05
    @test soil_layer_conductivity(c3, 5) ≈ K0 / 2
    # stateful bottom boundary (#17): a linear reservoir fed by drainage
    mutable struct Reservoir <: AbstractSoilBottomBoundary
        storage::Float64
    end
    SC.bottom_flux_linearization(::Reservoir, col, ψN, KN) = (col.gravity * KN, 0.0)
    SC.finalize_bottom!(r::Reservoir, col, q, dt) = (r.storage += q * dt; nothing)
    c5 = SoilWaterColumn(uniform_soil_grid(1.0, 20), v; ψ = -0.3, bottom = Reservoir(0.0))
    for _ in 1:24
        step_soil_water!(c5, 3600.0)
    end
    @test c5.bottom.storage ≈ c5.budget.drainage rtol = 1e-12
    @test c5.bottom.storage > 0
end

# #327 M6: a Picard iteration that cannot converge down to dt_min is a
# numerical failure (a step-rejecting driver retries it), not a bug
@testset "Picard non-convergence throws NumericalFailure (#327 M6)" begin
    v = carsel_parrish_texture(:loam)
    col = SoilWaterColumn(uniform_soil_grid(0.5, 50), v; ψ = -100.0, top = HeadTop(0.0),
                          temperature_dependent = false, maxiter = 1, dt_min = 30.0)
    err = try
        step_soil_water!(col, 60.0); nothing
    catch e
        e
    end
    @test err isa Aeolus.NumericalFailure
end

end # SoilColumn testset
