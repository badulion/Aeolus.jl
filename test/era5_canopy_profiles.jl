# Era V, Phase 48 (#25) — within-canopy wind and temperature profiles.
#
# Part 1 checks the tools the validation needs against analytic results:
# measured (tabulated) LAD profiles, the steady K-theory scalar column,
# the displacement height and the Harman–Finnigan scalar diffusivity.
# Part 2 compares the 1-D canopy column of PlantCanopy with published
# profiles from a wind tunnel (Brunet et al. 1994), an orchard (CHATS) and
# a Scots pine forest (Hyytiälä); the data and their provenance are in
# test/data/canopy_profiles.jl.

using Printf, Random
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, R_d, saturation_vapor_pressure

include(joinpath(@__DIR__, "data", "canopy_profiles.jl"))

# Linear interpolation of a profile (local to this file; prefixed so the
# name does not collide with helpers of other test files in this module).
function _cp_interp(zs, vs, z)
    i = clamp(searchsortedlast(zs, z), 1, length(zs) - 1)
    t = (z - zs[i]) / (zs[i+1] - zs[i])
    return (1 - t) * vs[i] + t * vs[i+1]
end

# The 1-D canopy model used for every dataset: layer LAD from the profile,
# the Harman & Finnigan (2007) canopy mixing length l_c = 2β³/(C_d LAI/h)
# with β = 0.3, wall-limited near the ground and growing at κ above h
# (`canopy_mixing_length`), a no-slip ground, and a wind prescribed at the
# column top. Nothing is fitted to the data: C_d and the LAD are the
# published ones (or, where none is published, the module default).
function _cp_wind(p::AbstractLeafAreaDensity, h; C_d, ztop, N = 400, β = 0.3, u_top = 5.0)
    zf = collect(range(0, ztop; length = N + 1))
    lad = layer_leaf_area_density(p, zf)
    l_c = harman_finnigan_mixing_length(β, C_d, leaf_area_index(p) / h)
    sol = solve_canopy_wind_column(zf, lad; C_d = C_d, u_top = u_top, bottom = :no_slip,
                                   l_mix = z -> canopy_mixing_length(z; h = h, l_c = l_c))
    # Kinematic stress K ∂U/∂z at the faces (ground stress at face 1).
    Nz = length(sol.u)
    τ = [f == 1 ? sol.tau_ground :
         f == Nz + 1 ? sol.u_star^2 :
         sol.K[f] * (sol.u[f] - sol.u[f-1]) / (sol.z[f] - sol.z[f-1]) for f in 1:Nz+1]
    U_h = _cp_interp(sol.z, sol.u, h)
    τ_h = _cp_interp(zf, τ, h)
    d = canopy_displacement_height(sol, lad; C_d = C_d)
    return (; sol, lad, l_c, U_h, τ, τ_h, d, zf)
end

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


@testset "wind tunnel: Brunet et al. (1994) waving wheat" begin
    wt = canopy_profile_observations().wind_tunnel
    h = wt.h
    m = _cp_wind(UniformLAD(wt.ah, h), h; C_d = wt.C_d, ztop = 2h)
    @test m.sol.residual < 1e-12
    @test m.sol.u_star^2 ≈ m.sol.drag_integral + m.sol.tau_ground rtol = 1e-9
    U = [_cp_interp(m.sol.z, m.sol.u, z * h) / m.U_h for z in wt.z]
    τ = [_cp_interp(m.zf, m.τ, z * h) / m.τ_h for z in wt.z]
    inside = wt.z .<= 1.0
    err_in = maximum(abs.(U[inside] .- wt.U_over_Uh[inside]))
    err_above = maximum(abs.(U[.!inside] ./ wt.U_over_Uh[.!inside] .- 1))
    err_τ = maximum(abs.(τ[inside] .- wt.stress[inside]))
    println("    Brunet et al. (1994) wind tunnel, C_d = $(wt.C_d), a·h = $(wt.ah):")
    for (i, z) in enumerate(wt.z)
        @printf("      z/h = %.3f  U/U_h obs %.3f model %.3f   τ/τ_h obs %6.3f model %6.3f\n",
                z, wt.U_over_Uh[i], U[i], wt.stress[i], τ[i])
    end
    @printf("      max |ΔU/U_h| in canopy %.3f, above canopy max rel. err %.3f, max |Δτ/τ_h| %.3f, u*/U_h %.3f, d/h %.3f\n",
            err_in, err_above, err_τ, sqrt(m.τ_h) / m.U_h, m.d / h)
    # Mean wind through the canopy depth to within 0.05 U_h (measured 0.021)
    # and above it to within 8 % (measured 4.7 %).
    @test err_in < 0.05
    @test err_above < 0.08
    # Stress: the momentum absorbed by the upper canopy is under-predicted
    # (first-order closure), max error 0.18 τ_h at z/h = 0.83.
    @test err_τ < 0.25
    # The stress profile falls monotonically from its canopy-top value to
    # ≈ 0 at the floor, as observed (-0.02 at z/h = 0.17).
    @test all(diff(τ[inside]) .> 0) && τ[1] < 0.05
    # The comparison discriminates between drag coefficients: with the
    # C_d = 1.2 quoted for the same canopy by Patton (1997) the in-canopy
    # error is several times larger.
    m2 = _cp_wind(UniformLAD(wt.ah, h), h; C_d = wt.C_d_alt, ztop = 2h)
    U2 = [_cp_interp(m2.sol.z, m2.sol.u, z * h) / m2.U_h for z in wt.z]
    err2 = maximum(abs.(U2[inside] .- wt.U_over_Uh[inside]))
    @printf("      with C_d = %.2f: max |ΔU/U_h| in canopy %.3f\n", wt.C_d_alt, err2)
    @test err2 > 3err_in
end

@testset "orchard: CHATS with and without leaves (Dupont & Patton 2012)" begin
    obs = canopy_profile_observations()
    C_d = 0.2          # no published C_d for CHATS: the module default, not tuned
    res = Dict{Symbol,Any}()
    for (key, ds) in ((:leaves, obs.chats_leaves), (:noleaves, obs.chats_noleaves))
        h = ds.h
        p = TabulatedLAD(ds.pad_z .* h, ds.pad_ah ./ h)
        m = _cp_wind(p, h; C_d = C_d, ztop = 2.9h)
        @test m.sol.residual < 1e-12
        U = [_cp_interp(m.sol.z, m.sol.u, z * h) / m.U_h for z in ds.z]
        crown = 0.4 .<= ds.z .< 1.0
        above = ds.z .> 1.0
        err_crown = maximum(abs.(U[crown] .- ds.U_over_Uh[crown]))
        err_above = maximum(abs.(U[above] ./ ds.U_over_Uh[above] .- 1))
        # shear length U_h / (∂U/∂z)_h
        k = searchsortedlast(m.sol.z, h)
        Ls = m.U_h / ((m.sol.u[k+1] - m.sol.u[k]) / (m.sol.z[k+1] - m.sol.z[k]))
        println("    $(ds.name) (PAI $(round(leaf_area_index(p), digits = 2)), C_d = $C_d):")
        for (i, z) in enumerate(ds.z)
            @printf("      z/h = %.2f  U/U_h obs %.3f model %.3f\n", z, ds.U_over_Uh[i], U[i])
        end
        @printf("      crown max |ΔU/U_h| %.3f, above-canopy max rel. err %.3f, d = %.2f m (obs %.1f), L_s/h at h = %.2f, u*/U_h = %.3f\n",
                err_crown, err_above, m.d, ds.d, Ls / h, sqrt(m.τ_h) / m.U_h)
        # Crown (0.4 ≤ z/h < 1): within 0.12 U_h (measured 0.069 with
        # leaves, 0.093 without). Above the canopy within 15 % (measured
        # 4.0 % with leaves, 12.2 % without: over the sparse leafless crown
        # l_c = 0.36 h is large and the model mixes too much above it).
        @test err_crown < 0.12
        @test err_above < 0.15
        # Displacement height within 0.1 h of the value inferred from the
        # measured momentum-flux profiles (measured +0.56 m, +0.84 m).
        @test abs(m.d - ds.d) < 0.1h
        res[key] = (; U, d = m.d, Ls, crown)
    end
    L, B = res[:leaves], res[:noleaves]
    # Leaf-out: less wind at every crown level and a higher displacement
    # height — in the observations and in the model.
    @test all(obs.chats_leaves.U_over_Uh[L.crown] .< obs.chats_noleaves.U_over_Uh[L.crown])
    @test all(L.U[L.crown] .< B.U[L.crown])
    @test obs.chats_leaves.d > obs.chats_noleaves.d && L.d > B.d
    # Shear length at the canopy top with leaves: observed 0.32 h at the
    # inflection point, which is at h (Dupont & Patton 2012, Table 1).
    @test abs(L.Ls / 10.0 - obs.chats_leaves.Ls_over_h) < 0.1
    # Known limitation: with leaves the trunk space holds a secondary wind
    # maximum (U = 0.27 U_h at 0.15 h > 0.14 U_h at 0.3 h), carried by
    # sweeps and pressure gradients that first-order (K) closure cannot
    # represent; the model profile is monotone.
    @test obs.chats_leaves.U_over_Uh[1] > obs.chats_leaves.U_over_Uh[2]
    @test_broken L.U[1] > L.U[2]
end

@testset "forest: Hyytiälä Scots pine (Launiainen et al. 2007)" begin
    hy = canopy_profile_observations().hyytiala
    h = hy.h
    p = TabulatedLAD(hy.lad_z .* h, hy.lad)
    m = _cp_wind(p, h; C_d = hy.C_d, ztop = 1.47h)
    @test m.sol.residual < 1e-12
    Uu = [_cp_interp(m.sol.z, m.sol.u, z * h) / m.sol.u_star for z in hy.z]
    println("    Hyytiälä (LAI $(round(leaf_area_index(p), digits = 2)), C_d = $(hy.C_d)):")
    for (i, z) in enumerate(hy.z)
        @printf("      z/h = %.2f  U/u* obs NN %.3f (US %.3f)  model %.3f\n",
                z, hy.U_over_ustar.NN[i], hy.U_over_ustar.US[i], Uu[i])
    end
    @printf("      d/h = %.3f (Rannik et al. 2003: %.2f), u*/U_h = %.3f\n", m.d / h, hy.d_over_h, m.sol.u_star / m.U_h)
    # Displacement height within 0.05 h (measured 0.750 against 0.78).
    @test abs(m.d / h - hy.d_over_h) < 0.05
    # In the crown (0.78 h) and above it (1.47 h) U/u* is over-predicted
    # by 26 % and 30 % (u*/U_h = 0.26 is low: too little momentum is
    # absorbed near the crown top, as in the wind-tunnel stress profile).
    # Guard: within 35 %.
    for i in (6, 7)
        @test 0 < Uu[i] / hy.U_over_ustar.NN[i] - 1 < 0.35
    end
    # Known limitation: the trunk space (0.13-0.40 h) is ventilated by
    # sweeps from the crown, U/u* ≈ 1.2 observed with a weak secondary
    # maximum; the local K closure leaves it at about half that.
    trunk = 3:5
    @printf("      trunk space (0.13-0.40 h): model/obs = %s\n",
            join((@sprintf("%.2f", Uu[i] / hy.U_over_ustar.NN[i]) for i in trunk), ", "))
    @test all(Uu[trunk] .< hy.U_over_ustar.NN[trunk])
    @test_broken all(abs.(Uu[trunk] ./ hy.U_over_ustar.NN[trunk] .- 1) .< 0.3)
end

end
