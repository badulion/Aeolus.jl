# Phase 48 (#25) — validation: the 1-D canopy column of
# PlantCanopy (implicit drag, Harman & Finnigan mixing length, coupled leaf
# energy balance and scalar column) against published within-canopy
# profiles from a wind tunnel (Brunet et al. 1994), an orchard (CHATS,
# Dupont & Patton 2012) and a Scots pine forest (Hyytiälä, Launiainen et
# al. 2007). The data and their provenance are in
# test/data/canopy_profiles.jl (shared with the tests). The tools (LAD
# tables, scalar column, displacement height, diffusivity) are unit-tested
# against analytic results in test/canopy_profiles.jl.

using Test, Printf, Random
using Aeolus
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, R_d, saturation_vapor_pressure

isdefined(@__MODULE__, :canopy_profile_observations) ||
    include(joinpath(@__DIR__, "..", "test", "data", "canopy_profiles.jl"))

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

# Coupled in-canopy air temperature: leaf energy balance of every layer
# (sunlit/shaded leaves, two-stream shortwave, longwave exchange) gives
# the heat source; the steady scalar column with the Harman & Finnigan
# (2008) diffusivity turns it into a temperature profile; repeat until the
# air temperature the leaves see is the one their heat produces.
function _cp_temperature(p, h; C_d, ztop, N, u_star, T_ref, rh, μ, S_dir, S_dif,
                         lw_down, gs, H_ground = 0.0, Sc = 0.5, β = 0.3)
    m = _cp_wind(p, h; C_d = C_d, ztop = ztop, N = N, u_top = 1.0, β = β)
    U = m.sol.u .* (u_star / m.sol.u_star)
    zf = m.zf; lad = m.lad
    K = [harman_finnigan_scalar_diffusivity(z; h = h, u_star = u_star, l_mix = m.l_c,
                                            β = β, Sc = Sc) for z in zf]
    c = DEFAULT_CONSTANTS
    ρcp = 101325 / (R_d(c) * T_ref) * c.cp_dry
    ea = rh * saturation_vapor_pressure(T_ref)
    leaves = RossGoudriaanLeaves(0.01)             # CLM needleleaf evergreen χ_L
    sw = μ > 0 ? (canopy_shortwave(zf, lad; sun = μ, sw_direct = 0.45S_dir, sw_diffuse = 0.45S_dif,
                                   optics = LeafOptics(0.07, 0.05), albedo_direct = 0.10, leaves = leaves),
                  canopy_shortwave(zf, lad; sun = μ, sw_direct = 0.55S_dir, sw_diffuse = 0.55S_dif,
                                   optics = LeafOptics(0.35, 0.10), albedo_direct = 0.20, leaves = leaves)) :
         nothing
    Ta = fill(Float64(T_ref), N); leaf = nothing; col = nothing; iters = 0
    for it in 1:100
        iters = it
        leaf = canopy_leaf_energy_balance(zf, lad; shortwave = sw, par = sw === nothing ? nothing : 1,
                                          T_air = Ta, e_air = ea, U = max.(U, 0.05), leaves = leaves,
                                          lw_down = lw_down, T_ground = Ta[1], stomata = gs,
                                          boundary_layer = CampbellNormanBoundaryLayer(leaf_width = 0.01))
        col = solve_canopy_scalar_column(zf, K, leaf.heat_source ./ ρcp; top_value = T_ref,
                                         bottom_flux = H_ground / ρcp)
        δ = maximum(abs.(col.c .- Ta))
        Ta .= col.c
        δ < 1e-7 && break
    end
    return (; Ta, z = col.z, leaf, col, ρcp, iters, K)
end

@testset "PlantCanopy: within-canopy profiles vs observations (#25)" begin

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
    # and above it to within 8 % (measured 6.2 %).
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
        # Dupont & Patton (2012) define L_s = U(h_i)/(dU/dz)(h_i) at the
        # inflection point h_i (maximum shear); the model's h_i is taken as
        # the maximum shear in the crown (0.3 h ≤ z ≤ h), away from the
        # no-slip floor.
        g = diff(m.sol.u) ./ diff(m.sol.z)
        zg = (m.sol.z[1:end-1] .+ m.sol.z[2:end]) ./ 2
        sel = findall(z -> 0.3h <= z <= h, zg)
        i = sel[argmax(g[sel])]
        h_i = zg[i]
        Ls_i = (m.sol.u[i] + m.sol.u[i+1]) / 2 / g[i]
        println("    $(ds.name) (PAI $(round(leaf_area_index(p), digits = 2)), C_d = $C_d):")
        for (i, z) in enumerate(ds.z)
            @printf("      z/h = %.2f  U/U_h obs %.3f model %.3f\n", z, ds.U_over_Uh[i], U[i])
        end
        @printf("      crown max |ΔU/U_h| %.3f, above-canopy max rel. err %.3f, d = %.2f m (obs %.1f), L_s/h at h = %.2f, u*/U_h = %.3f\n",
                err_crown, err_above, m.d, ds.d, Ls / h, sqrt(m.τ_h) / m.U_h)
        @printf("      inflection point: h_i = %.2f m, L_s/h = %.2f (obs NN: L_s/h = %.2f)\n",
                h_i, Ls_i / h, ds.Ls_over_h)
        # Crown (0.4 ≤ z/h < 1): within 0.12 U_h (measured 0.069 with
        # leaves, 0.093 without). Above the canopy within 15 % (measured
        # 4.0 % with leaves, 12.2 % without: over the sparse leafless crown
        # l_c = 0.36 h is large and the model mixes too much above it).
        @test err_crown < 0.12
        @test err_above < 0.15
        # Displacement height within 0.1 h of the value inferred from the
        # measured momentum-flux profiles (measured +0.56 m, +0.84 m).
        @test abs(m.d - ds.d) < 0.1h
        res[key] = (; U, d = m.d, Ls, Ls_i, h_i, crown)
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
    @test abs(L.Ls_i / 10.0 - obs.chats_leaves.Ls_over_h) < 0.1
    # Leaf-out shortens the shear length, observed (0.72 h → 0.32 h) and
    # modelled. Known limitation: without leaves the observed inflection
    # point is inside the crown at 6 m (L_s = 0.72 h, Table 1), whereas the
    # model's maximum shear stays at the canopy top with L_s ≈ 1.3 h — the
    # column does not reproduce the leafless shear layer.
    @test obs.chats_noleaves.Ls_over_h > obs.chats_leaves.Ls_over_h && B.Ls_i > L.Ls_i
    @test_broken abs(B.Ls_i / 10.0 - obs.chats_noleaves.Ls_over_h) < 0.2
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


@testset "forest: Hyytiälä in-canopy air temperature" begin
    # Semi-quantitative: Launiainen et al. (2007, Fig. 3b) give median
    # Θ(z) - Θ(16.8 m) per stability class but not the radiation and wind
    # of those medians. The forcing below is an assumed clear midsummer
    # midday (μ = 0.78, 600 + 120 W m⁻², u* = 0.4 m s⁻¹) and clear night
    # (sky 90 W m⁻² below σT⁴, u* = 0.1 m s⁻¹) at 61.8° N; stomata 0.1 /
    # 0.005 mol m⁻² s⁻¹; no forest-floor sensible heat. So the test asserts
    # the sign and size of the in-canopy anomaly, not a fitted value.
    hy = canopy_profile_observations().hyytiala
    h = hy.h
    p = TabulatedLAD(hy.lad_z .* h, hy.lad)
    kw = (; C_d = hy.C_d, ztop = 16.8, N = 84)
    σ = 5.670374419e-8
    day = (; u_star = 0.4, T_ref = 293.15, rh = 0.5, μ = 0.78, S_dir = 600.0, S_dif = 120.0,
           lw_down = 330.0, gs = 0.1)
    night = (; u_star = 0.1, T_ref = 285.15, rh = 0.8, μ = 0.0, S_dir = 0.0, S_dif = 0.0,
             lw_down = σ * 285.15^4 - 90, gs = 0.005)
    res = Dict{Symbol,Any}()
    for (lab, f, cls) in ((:day, day, :US), (:night, night, :S))
        r = _cp_temperature(p, h; kw..., f...)
        @test r.leaf.converged
        @test r.iters < 100
        # The coupling conserves energy: the heat leaving the column top is
        # the leaves' sensible heat (no ground flux).
        @test r.col.flux[end] * r.ρcp ≈ r.leaf.H_total rtol = 1e-10
        ΔT = [_cp_interp(r.z, r.Ta, z) - f.T_ref for z in hy.T_z_m[1:2]]
        o = getproperty(hy.T_minus_ref, cls)[1:2]
        @printf("    %-5s (%s): H_leaves = %6.1f W m⁻², Θ - Θ(16.8 m) at 4.2 / 8.4 m: obs %+.2f / %+.2f K, model %+.2f / %+.2f K\n",
                lab, cls, r.leaf.H_total, o[1], o[2], ΔT[1], ΔT[2])
        res[lab] = (; ΔT, o, H = r.leaf.H_total)
    end
    # Day: the crown heats the air; the sub-crown air is warmer than the
    # reference level, as observed (+0.29 / +0.30 K). The model's +1.38 K
    # is 4.6× too large (assumed forcing; H_leaves = 275 W m⁻²; neutral
    # Sc = 0.5, whereas convective mixing in the US class is stronger);
    # bounded here at 2 K.
    D = res[:day]
    @test D.H > 0
    @test all(D.o .> 0) && all(0 .< D.ΔT .< 2.0)
    # With no floor flux and no sources below the crown the trunk space is
    # isothermal, as the observations nearly are (0.292 vs 0.301 K).
    @test abs(D.ΔT[1] - D.ΔT[2]) < 0.02
    # Night: radiatively cooled needles cool the canopy air below the
    # reference level, as observed (-3.16 / -2.11 K). The neutral closure
    # gives about -1.5 K: stable stratification, which suppresses mixing in
    # the S regime, is not in the model.
    Nn = res[:night]
    @test Nn.H < 0
    @test all(Nn.o .< 0) && all(-5 .< Nn.ΔT .< -0.5)
    # K-theory scaling: doubling u* doubles K, the sources barely change,
    # so the anomaly halves.
    r2 = _cp_temperature(p, h; kw..., day..., u_star = 0.8)
    ΔT2 = _cp_interp(r2.z, r2.Ta, 8.4) - day.T_ref
    @printf("    day, u* doubled: ΔT(8.4 m) %+.3f K (ratio %.2f)\n", ΔT2, ΔT2 / D.ΔT[2])
    @test 0.4 < ΔT2 / D.ΔT[2] < 0.65
end

end
