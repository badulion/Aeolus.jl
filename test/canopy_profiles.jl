# Era V, Phase 48 (#25) — within-canopy wind and temperature profiles:
# the tools the validation needs, against analytic results: measured
# (tabulated) LAD profiles, the steady K-theory scalar column, the
# displacement height and the Harman–Finnigan scalar diffusivity. The
# comparison of the 1-D canopy column of PlantCanopy with published
# profiles (Brunet et al. 1994 wind tunnel, CHATS orchard, Hyytiälä
# Scots pine) is a validation run: validation/canopy_profiles.jl.
# The data and their provenance are in test/data/canopy_profiles.jl.

using Random

include(joinpath(@__DIR__, "data", "canopy_profiles.jl"))

@testset "Era V — PlantCanopy: within-canopy profiles (#25)" begin

@testset "TabulatedLAD: measured profiles" begin
    # a triangle 0 → 2 → 0 on [1, 5] m: LAI = 4 exactly; layer integrals
    # conserve it on faces that miss the table nodes
    p = TabulatedLAD([1.0, 3.0, 5.0], [0.0, 2.0, 0.0])
    @test leaf_area_index(p) == 4.0
    @test cumulative_leaf_area(p, 3.0) ≈ 2.0 rtol = 1e-13
    zf = [0.0, 0.7, 2.9, 3.1, 6.0]
    @test sum(layer_leaf_area_density(p, zf) .* diff(zf)) ≈ 4.0 rtol = 1e-12
    # the digitised Hyytiälä profile integrates to the one-sided LAI 3.5
    # quoted by Launiainen et al. (2007, Table 2)
    hy = canopy_profile_observations().hyytiala
    @test leaf_area_index(TabulatedLAD(hy.lad_z .* hy.h, hy.lad)) ≈ 3.5 atol = 0.02
end

@testset "scalar column: analytic solutions and exact budget" begin
    # (a) no source, constant K: linear profile, uniform flux K Δc / H, any grid
    H = 10.0; K0 = 0.3
    zf = [0.0, 0.1, 0.5, 2.0, 2.2, 7.0, 10.0]
    N = length(zf) - 1
    s = solve_canopy_scalar_column(zf, fill(K0, N + 1), zeros(N); top_value = 290.0,
                                   bottom_value = 295.0)
    @test maximum(abs.(s.c .- (295.0 .- 5.0 .* s.z ./ H))) < 1e-11
    @test maximum(abs.(s.flux ./ (K0 * 5.0 / H) .- 1)) < 1e-13
    # (b) uniform source below H, no ground flux: c = S₀ (H² − z²)/(2K),
    # second order (two levels); flux S₀ z exact
    S0 = 2e-3; errs = Float64[]; ferr = 0.0
    for N in (20, 40)
        zf = collect(range(0, H; length = N + 1))
        s = solve_canopy_scalar_column(zf, fill(K0, N + 1), fill(S0, N); top_value = 0.0,
                                       bottom_flux = 0.0)
        ferr = max(ferr, maximum(abs.(s.flux .- S0 .* zf)))
        push!(errs, maximum(abs.(s.c .- S0 .* (H^2 .- s.z .^ 2) ./ (2K0))))
    end
    @test ferr < 1e-14
    @test 3.9 < errs[1] / errs[2] < 4.1
    # (c) exact discrete budget, irregular layers, sources of both signs
    rng = Random.MersenneTwister(25)
    zf = cumsum([0.0; 0.05 .+ rand(rng, 30)])
    N = 30; src = randn(rng, N) .* 1e-3; Kr = 0.01 .+ rand(rng, N + 1)
    s = solve_canopy_scalar_column(zf, Kr, src; top_value = 300.0, bottom_flux = 0.07)
    @test s.flux[end] ≈ 0.07 + sum(src .* diff(zf)) rtol = 1e-13
end

@testset "displacement height and scalar diffusivity: analytic limits" begin
    # deep uniform canopy: drag ∝ e^{2γ(z−h)}, centroid d = h − 1/(2γ)
    # (Thom 1971; Jackson 1981)
    h, LAI, C_d, β, U_h = 20.0, 10.0, 0.25, 0.3, 3.0
    a = LAI / h
    l = harman_finnigan_mixing_length(β, C_d, a)
    γ = canopy_attenuation_coefficient(C_d, a, l)
    zf = collect(range(0, h; length = 401))
    lad = layer_leaf_area_density(UniformLAD(LAI, h), zf)
    sol = solve_canopy_wind_column(zf, lad; C_d = C_d, l_mix = l, u_top = U_h, ν = 1e-10)
    @test canopy_displacement_height(sol, lad; C_d = C_d) ≈ h - 1 / (2γ) rtol = 1e-3
    # Harman & Finnigan (2008) K_c: l u*/Sc at h, e-folding depth l/β
    Kc(z) = harman_finnigan_scalar_diffusivity(z; h = h, u_star = β * U_h, l_mix = l, β = β, Sc = 0.5)
    @test Kc(h) ≈ l * β * U_h / 0.5
    @test Kc(h - l / β) ≈ Kc(h) / ℯ rtol = 1e-12
end

end
