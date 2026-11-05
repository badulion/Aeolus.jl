# Era V, Phase 48 (#25) — within-canopy wind and temperature profiles:
# the tools the validation needs, against analytic results: measured
# (tabulated) LAD profiles, the steady K-theory scalar column, the
# displacement height and the Harman–Finnigan scalar diffusivity. The
# comparison of the 1-D canopy column of PlantCanopy with published
# profiles (Brunet et al. 1994 wind tunnel, CHATS orchard, Hyytiälä
# Scots pine) is a validation run: validation/era5_canopy_profiles.jl.
# The data and their provenance are in test/data/canopy_profiles.jl.

using Printf, Random

include(joinpath(@__DIR__, "data", "canopy_profiles.jl"))

@testset "Era V — PlantCanopy: within-canopy profiles (#25)" begin

@testset "TabulatedLAD: measured profiles" begin
    # A triangle 0 → 2 → 0 on [1, 5] m: LAI = 4 exactly, apex at 3 m.
    p = TabulatedLAD([1.0, 3.0, 5.0], [0.0, 2.0, 0.0])
    @test leaf_area_index(p) == 4.0
    @test canopy_height(p) == 5.0
    @test leaf_area_density(p, 2.0) ≈ 1.0
    @test leaf_area_density(p, 0.5) == 0 && leaf_area_density(p, 5.5) == 0
    @test cumulative_leaf_area(p, 3.0) ≈ 2.0 rtol = 1e-13
    @test cumulative_leaf_area(p, 0.0) ≈ 4.0 rtol = 1e-13
    # Layer integrals conserve the LAI on any grid, including faces that
    # do not coincide with the table nodes.
    for zf in (collect(range(0, 6; length = 8)), [0.0, 0.7, 2.9, 3.1, 6.0], collect(0:0.013:6.5))
        lad = layer_leaf_area_density(p, zf)
        @test sum(lad .* diff(zf)) ≈ 4.0 rtol = 1e-12
    end
    # Rescaling to a quoted LAI keeps the shape.
    q = TabulatedLAD([1.0, 3.0, 5.0], [0.0, 2.0, 0.0]; LAI = 3.0)
    @test leaf_area_index(q) == 3.0
    @test leaf_area_density(q, 2.0) ≈ 0.75
    @test sum(layer_leaf_area_density(q, collect(0:0.1:5)) .* 0.1) ≈ 3.0 rtol = 1e-12
    @test_throws ArgumentError TabulatedLAD([0.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    @test_throws ArgumentError TabulatedLAD([0.0, 1.0], [1.0, -1.0])
    @test_throws ArgumentError TabulatedLAD([0.0, 1.0], [0.0, 0.0]; LAI = 1.0)

    # The digitised Hyytiälä needle-area profile integrates to the one-sided
    # LAI of 3.5 quoted by Launiainen et al. (2007, Table 2).
    hy = canopy_profile_observations().hyytiala
    ph = TabulatedLAD(hy.lad_z .* hy.h, hy.lad)
    @test leaf_area_index(ph) ≈ 3.5 atol = 0.02
    @printf("    Hyytiälä digitised LAD: one-sided LAI %.3f (published 3.5)\n", leaf_area_index(ph))
end

@testset "scalar column: analytic solutions and exact budget" begin
    # (a) No source, fixed surface and top values, constant K: the linear
    # profile and the uniform flux K Δc / H, exact on any grid.
    H = 10.0; K0 = 0.3
    for zf in (collect(range(0, H; length = 11)), [0.0, 0.1, 0.5, 2.0, 2.2, 7.0, 10.0])
        N = length(zf) - 1
        s = solve_canopy_scalar_column(zf, fill(K0, N + 1), zeros(N); top_value = 290.0,
                                       bottom_value = 295.0)
        @test all(isapprox.(s.c, 295.0 .- 5.0 .* s.z ./ H; rtol = 1e-14))
        @test all(isapprox.(s.flux, K0 * 5.0 / H; rtol = 1e-13))
    end
    # Variable K: the flux is (c_g - c_top) over the summed face resistances.
    zf = collect(range(0, H; length = 21)); N = 20
    Kv = 0.05 .+ 0.02 .* zf .^ 1.5
    s = solve_canopy_scalar_column(zf, Kv, zeros(N); top_value = 0.0, bottom_value = 1.0)
    zc = (zf[1:N] .+ zf[2:N+1]) ./ 2
    δ = [zc[1] - zf[1]; diff(zc); zf[end] - zc[N]]
    @test all(isapprox.(s.flux, 1.0 / sum(δ ./ Kv); rtol = 1e-13))

    # (b) Uniform source S₀ below H, no flux from the ground, constant K:
    # c = c_top + S₀ (H² - z²) / (2K). The flux S₀ z is exact at every face;
    # c converges at second order (only the half cell below the top is
    # not exact).
    S0 = 2e-3; errs = Float64[]
    for N in (20, 40, 80)
        zf = collect(range(0, H; length = N + 1))
        s = solve_canopy_scalar_column(zf, fill(K0, N + 1), fill(S0, N); top_value = 0.0,
                                       bottom_flux = 0.0)
        @test all(isapprox.(s.flux, S0 .* zf; rtol = 1e-12, atol = 1e-15))
        push!(errs, maximum(abs.(s.c .- S0 .* (H^2 .- s.z .^ 2) ./ (2K0))))
    end
    @test errs[end] < 1e-4 * S0 * H^2 / (2K0)
    @test 3.9 < errs[1] / errs[2] < 4.1 && 3.9 < errs[2] / errs[3] < 4.1

    # (c) Exact discrete budget with irregular layers, variable K and
    # sources of both signs: flux out of the top = ground flux + Σ S Δz.
    rng = Random.MersenneTwister(25)
    zf = cumsum([0.0; 0.05 .+ rand(rng, 30)])
    N = 30; src = randn(rng, N) .* 1e-3; Kr = 0.01 .+ rand(rng, N + 1)
    s = solve_canopy_scalar_column(zf, Kr, src; top_value = 300.0, bottom_flux = 0.07)
    @test s.flux[1] == 0.07
    @test s.flux[end] ≈ 0.07 + sum(src .* diff(zf)) rtol = 1e-13
    # A surface value and the flux it implies describe the same solution.
    Kr[1] = 0.2
    s1 = solve_canopy_scalar_column(zf, Kr, src; top_value = 300.0, bottom_value = 301.0)
    s2 = solve_canopy_scalar_column(zf, Kr, src; top_value = 300.0, bottom_flux = s1.flux[1])
    @test maximum(abs.(s1.c .- s2.c)) < 1e-11
    @test s1.c[1] + s1.flux[1] * (s1.z[1] - zf[1]) / Kr[1] ≈ 301.0 rtol = 1e-14
    # Heating the air only makes it warmer than the top value.
    s = solve_canopy_scalar_column(zf, Kr, abs.(src); top_value = 300.0, bottom_flux = 0.0)
    @test all(>(300.0), s.c) && all(>(0), diff(s.flux))

    @test_throws ArgumentError solve_canopy_scalar_column(zf, Kr, src; top_value = 0.0)
    @test_throws ArgumentError solve_canopy_scalar_column(zf, Kr, src; top_value = 0.0,
                                                          bottom_flux = 0.0, bottom_value = 1.0)
    @test_throws DimensionMismatch solve_canopy_scalar_column(zf, Kr[1:end-1], src;
                                                              top_value = 0.0, bottom_flux = 0.0)
end

@testset "displacement height and scalar diffusivity: analytic limits" begin
    # Deep uniform canopy: U = U_h e^{γ(z-h)} so the drag ∝ e^{2γ(z-h)} and
    # its centroid is d = h - 1/(2γ) (Thom 1971; Jackson 1981).
    h, LAI, C_d, β, U_h = 20.0, 10.0, 0.25, 0.3, 3.0
    a = LAI / h
    l = harman_finnigan_mixing_length(β, C_d, a)
    γ = canopy_attenuation_coefficient(C_d, a, l)
    zf = collect(range(0, h; length = 401))
    lad = layer_leaf_area_density(UniformLAD(LAI, h), zf)
    sol = solve_canopy_wind_column(zf, lad; C_d = C_d, l_mix = l, u_top = U_h, ν = 1e-10)
    d = canopy_displacement_height(sol, lad; C_d = C_d)
    @test d ≈ h - 1 / (2γ) rtol = 1e-3
    @printf("    deep uniform canopy: d = %.4f m, analytic h - 1/(2γ) = %.4f m\n", d, h - 1 / (2γ))
    # The eddy viscosity returned with the solution is the one it used:
    # K ∂U/∂z at the top face is the friction velocity squared.
    N = length(sol.u)
    @test sol.K[N+1] * (U_h - sol.u[N]) / (zf[N+1] - sol.z[N]) ≈ sol.u_star^2 rtol = 1e-12
    @test sol.K[1] == 0                                  # free slip
    @test sol.z_faces == zf
    # A ground that takes all the stress puts d at the ground.
    solg = solve_canopy_wind_column(zf, zeros(N); C_d = C_d, l_mix = 1.0, u_top = U_h,
                                    bottom = :no_slip)
    @test canopy_displacement_height(solg, zeros(N); C_d = C_d) == 0

    # Mixing length: κz at the wall, l_c in the canopy, continuous at h,
    # growing at κ above.
    κ = Aeolus.SurfaceExchange.von_karman()
    lm(z) = canopy_mixing_length(z; h = h, l_c = l)
    @test lm(0.0) == 0 && lm(0.1) ≈ κ * 0.1 && lm(h / 2) == l
    @test lm(h) == l && lm(h + 1e-12) ≈ l && lm(h + 3.0) ≈ l + 3κ

    # Harman & Finnigan (2008) K_c: l u*/Sc at h, decays like the wind
    # (e-folding depth l/β) and is continuous into the constant-stress layer.
    Kc(z) = harman_finnigan_scalar_diffusivity(z; h = h, u_star = β * U_h, l_mix = l, β = β, Sc = 0.5)
    @test Kc(h) ≈ l * β * U_h / 0.5
    @test Kc(h - l / β) ≈ Kc(h) / ℯ rtol = 1e-12
    @test Kc(h + 1e-9) ≈ Kc(h) rtol = 1e-8
    @test Kc(h + 2.0) ≈ (l + Aeolus.SurfaceExchange.von_karman() * 2.0) * β * U_h / 0.5
    # Deep in the canopy the ratio to the local mixing-length viscosity of
    # the exponential wind is the Schmidt number: l²|∂U/∂z| = l²γU = l u* e^{γ(z-h)}.
    @test l^2 * γ * canopy_exponential_wind(12.0; h = h, u_h = U_h, C_d = C_d, a = a, β = β) / 0.5 ≈
          Kc(12.0) rtol = 1e-12
end

end
