# Phase 47 (#21) — the vegetated land column: soil-water stress, big-leaf
# canopy radiation, canopy aerodynamics and the coupled column.
#
# Independent references used below (none of them is the code under test):
#   * CLM4.5 Technical Note (Oleson et al. 2013, NCAR/TN-503+STR):
#     Table 8.1 (ψo, ψc in mm), Table 8.3 (Zeng 2001 ra, rb), Table 3.1
#     (leaf optics), Table 5.1 (Rz0m, Rd), eqs. 8.26-8.30 (βt, wilting
#     factor, root fractions), 5.126-5.128 (roughness), 5.115-5.119
#     (under-canopy transfer), 7.104-7.107 (root uptake).
#   * Closed-form integrals of the Beer's-law profiles, re-derived and
#     evaluated here by independent quadrature in depth and in direction.
#   * Kirchhoff / radiative equilibrium: an isothermal enclosure has no
#     net longwave exchange.

using Test
using Aeolus
using Aeolus.LandSurface
using Aeolus.SoilColumn: SoilThermalColumn

# composite Simpson on [a, b] with n (even) intervals
function _simpson(f, a, b; n = 2000)
    h = (b - a) / n
    s = f(a) + f(b)
    for i in 1:n-1
        s += (isodd(i) ? 4 : 2) * f(a + i * h)
    end
    return s * h / 3
end

@testset "LandSurface (Phase 47, #21)" begin

@testset "CLM4.5 PFT tables" begin
    # Table 8.1: ψo/ψc in mm, stored in m.
    @test CLM45_WATER_STRESS_POTENTIALS[:net_boreal] == (-66.0, -255.0)
    @test CLM45_WATER_STRESS_POTENTIALS[:bdt_temperate] == (-35.0, -224.0)
    @test CLM45_WATER_STRESS_POTENTIALS[:bes_temperate] == (-83.0, -428.0)
    @test CLM45_WATER_STRESS_POTENTIALS[:c4_grass] == (-74.0, -275.0)
    # Table 8.3
    @test CLM45_ROOT_PARAMETERS[:bet_tropical] == (7.0, 1.0)
    @test CLM45_ROOT_PARAMETERS[:c3_grass] == (11.0, 2.0)
    @test CLM45_ROOT_PARAMETERS[:corn] == (6.0, 3.0)
    # Table 3.1 / 5.1
    @test CLM45_LEAF_OPTICS[:net_temperate].ρ_nir == 0.35
    @test CLM45_LEAF_OPTICS[:bdt_temperate].τ_nir == 0.25
    @test CLM45_LEAF_OPTICS[:c3_grass].τ_nir == 0.34
    @test CLM45_AERODYNAMIC_PARAMETERS[:bet_tropical].R_z0m == 0.075
    @test CLM45_AERODYNAMIC_PARAMETERS[:c3_grass].R_d == 0.68
    # Every PFT of the photosynthesis table has the land-column columns.
    for k in keys(Aeolus.Photosynthesis.CLM45_PFT_PHOTOSYNTHESIS)
        @test haskey(CLM45_WATER_STRESS_POTENTIALS, k)
        @test haskey(CLM45_ROOT_PARAMETERS, k)
        @test haskey(CLM45_LEAF_OPTICS, k)
        @test haskey(CLM45_AERODYNAMIC_PARAMETERS, k)
    end
    v = pft_vegetation(:net_boreal)
    @test v.stomata isa MedlynStomata && v.stomata.g1 == 2.35
    @test v.photosynthesis.Vcmax25 == 62.6
    @test pft_vegetation(:c4_grass; model = :ball_berry).stomata.g1 == 4.0
    @test pft_vegetation(:c3_grass; K_n = 0.3).K_n == 0.3
    @test_throws ArgumentError pft_vegetation(:oak)
end

@testset "Root fractions (Zeng 2001)" begin
    for g in (clm5_soil_grid(), clm5_soil_grid(8), uniform_soil_grid(2.0, 7)),
        (ra, rb) in values(CLM45_ROOT_PARAMETERS)
        r = zeng_root_fractions(g, ra, rb)
        @test sum(r) ≈ 1 atol = 1e-14
        @test all(>=(0), r)
        # cumulative fraction above every interior interface = Y(z)
        Y(z) = 1 - (exp(-ra * z) + exp(-rb * z)) / 2
        for i in 1:length(g)-1
            @test sum(r[1:i]) ≈ Y(g.zh[i+1]) atol = 1e-14
        end
    end
    # Deeper-rooted types (smaller r_b) put more roots below 1 m.
    g = clm5_soil_grid()
    below1m(p) = sum(zeng_root_fractions(g, CLM45_ROOT_PARAMETERS[p]...)[g.z .> 1.0])
    @test below1m(:bet_tropical) > below1m(:net_boreal) > below1m(:c3_grass)
end

@testset "Wilting factor and root-zone stress βt" begin
    st = CLMWaterStress(-66.0, -255.0)
    @test wilting_factor(st, 0.0) == 1
    @test wilting_factor(st, -66.0) == 1
    @test wilting_factor(st, -255.0) == 0
    @test wilting_factor(st, -1000.0) == 0
    @test wilting_factor(st, (-66.0 - 255.0) / 2) ≈ 0.5
    @test_throws ArgumentError CLMWaterStress(-300.0, -100.0)

    g = clm5_soil_grid(10)
    roots = zeng_root_fractions(g, 6.0, 2.0)
    w = SoilWaterColumn(g, clapp_hornberger_texture(:loam); ψ = -1.0)
    β, wl = soil_water_stress(st, w, roots)
    @test β ≈ 1 atol = 1e-14                     # wet soil: no stress
    # Drying the profile uniformly: βt follows the wilting factor exactly.
    for ψ in (-50.0, -100.0, -160.0, -200.0, -254.0, -300.0)
        set_matric_potential!(w, ψ)
        β, wl = soil_water_stress(st, w, roots)
        @test β ≈ wilting_factor(st, ψ) atol = 1e-14
    end
    # A dry top over a wet subsoil: βt is the root fraction of the wet layers.
    ψprof = [i <= 4 ? -400.0 : -10.0 for i in 1:length(g)]
    set_matric_potential!(w, ψprof)
    β, _ = soil_water_stress(st, w, roots)
    @test β ≈ sum(roots[5:end]) atol = 1e-14
    # A frozen layer (T < T_f − 2) contributes nothing; pore ice scales w.
    set_matric_potential!(w, -10.0)
    w.T_soil[1] = 270.0
    β, wl = soil_water_stress(st, w, roots)
    @test wl[1] == 0
    @test β ≈ 1 - roots[1] atol = 1e-14
    w.T_soil[1] = 293.15
    w.θ_ice[2] = 0.25 * soil_porosity(w.hydraulics[2])
    _, wl = soil_water_stress(st, w, roots)
    @test wl[2] ≈ 0.75 atol = 1e-14

    # Root uptake: effective root fractions r_i w_i / βt, total E_t/ρw.
    w.θ_ice .= 0
    set_matric_potential!(w, [-(20.0 + 30i) for i in 1:length(g)])
    β, wl = soil_water_stress(st, w, roots)
    sink = root_water_uptake(3e-5, roots, wl, β)
    @test sum(sink) ≈ 3e-5 / 1000 rtol = 1e-13
    @test sink ≈ 3e-8 .* roots .* wl ./ β rtol = 1e-13
    @test all(iszero, root_water_uptake(3e-5, roots, wl, 0.0))
    @test all(iszero, root_water_uptake(-1e-6, roots, wl, β))
end

@testset "Big-leaf shortwave: conservation and closed forms" begin
    worst = 0.0
    for Ib in (0.0, 300.0), Id in (0.0, 120.0), μ in (0.1, 0.5, 1.0),
        L in (0.0, 0.3, 2.0, 6.0), σ in (0.0, 0.15, 0.8), αg in (0.0, 0.2, 1.0)
        r = bigleaf_shortwave(Ib, Id, μ, L; σ = σ, α_ground = αg)
        worst = max(worst, abs(r.sunlit + r.shaded + r.ground + r.reflected - Ib - Id))
        @test r.sunlit >= -1e-12 && r.shaded >= -1e-12 && r.ground >= 0
        @test r.L_sun + r.L_sha ≈ L atol = 1e-12
    end
    @test worst < 1e-12
    @info "big-leaf shortwave: worst energy residual (W m⁻²)" worst

    # Black leaves on a black ground: canopy absorbs the unintercepted
    # complement, sunlit leaves the whole beam, L_sun = (1 − e^{−kL})/k.
    μ, L = 0.6, 3.0
    kb = 0.5 / μ
    r = bigleaf_shortwave(500.0, 0.0, μ, L; σ = 0.0, α_ground = 0.0)
    @test r.sunlit ≈ 500 * (1 - exp(-kb * L)) rtol = 1e-13
    @test r.shaded ≈ 0 atol = 1e-10
    @test r.L_sun ≈ (1 - exp(-kb * L)) / kb rtol = 1e-13
    @test r.reflected == 0

    # Sunlit/shaded split against direct quadrature of the absorption
    # profiles in depth (per unit leaf area at depth l, times the sunlit
    # fraction e^{-k_b l}).
    for σ in (0.15, 0.8), L in (1.0, 5.0), μ in (0.3, 0.9)
        Ib, Id = 400.0, 100.0
        r = bigleaf_shortwave(Ib, Id, μ, L; σ = σ, α_ground = 0.0)
        kb = 0.5 / μ; kbp = kb * sqrt(1 - σ)
        # effective diffuse coefficient from the isotropic-sky transmittance,
        # computed independently by Simpson quadrature over the sky
        τd = _simpson(m -> 2m * exp(-0.5 / max(m, 1e-12) * sqrt(1 - σ) * L), 0.0, 1.0; n = 20000)
        kdp = -log(τd) / L
        ρh = (1 - sqrt(1 - σ)) / (1 + sqrt(1 - σ))
        ρcb = 1 - exp(-2ρh * kb / (1 + kb))
        ρcd = _simpson(m -> 2m * (1 - exp(-2ρh * (0.5 / max(m, 1e-12)) / (1 + 0.5 / max(m, 1e-12)))), 0.0, 1.0; n = 20000)
        tot(l) = (1 - ρcb) * kbp * Ib * exp(-kbp * l) + (1 - ρcd) * kdp * Id * exp(-kdp * l)
        dirdir(l) = (1 - σ) * kb * Ib                   # per sunlit leaf area
        sunabs(l) = exp(-kb * l) * (dirdir(l) + (1 - ρcd) * kdp * Id * exp(-kdp * l) +
                                    (1 - ρcb) * kbp * Ib * exp(-kbp * l) - (1 - σ) * kb * Ib * exp(-kb * l))
        @test r.sunlit ≈ _simpson(sunabs, 0.0, L) rtol = 1e-6
        @test r.sunlit + r.shaded ≈ _simpson(tot, 0.0, L) rtol = 1e-6
    end

    # Deep canopy: nothing reaches the ground and the reflection is ρ_cb.
    r = bigleaf_shortwave(1.0, 0.0, 0.5, 60.0; σ = 0.15, α_ground = 0.3)
    ρh = (1 - sqrt(0.85)) / (1 + sqrt(0.85))
    @test r.reflected ≈ 1 - exp(-2ρh * 1.0 / 2.0) rtol = 1e-10
    @test r.ground < 1e-12
    # No canopy: the ground takes everything it does not reflect.
    r = bigleaf_shortwave(300.0, 100.0, 0.5, 0.0; σ = 0.15, α_ground = 0.25)
    @test r.ground ≈ 300 && r.reflected ≈ 100 && r.sunlit == r.shaded == 0
    # Night: no sunlit leaves.
    r = bigleaf_shortwave(0.0, 50.0, -0.2, 3.0; σ = 0.15, α_ground = 0.2)
    @test r.L_sun == 0 && r.sunlit == 0
end

@testset "Big-leaf longwave" begin
    σSB = Aeolus.SurfaceExchange.DEFAULT_SURFACE_CONSTANTS.σ_SB
    worst = 0.0
    for Ldn in (250.0, 400.0), Tv in (270.0, 300.0), Tg in (265.0, 310.0),
        LS in (0.0, 0.5, 3.0, 8.0), εg in (0.9, 0.96, 1.0)
        r = bigleaf_longwave(Ldn, Tv, Tg, LS; ε_ground = εg, σ_SB = σSB)
        worst = max(worst, abs(r.canopy + r.ground + r.upward - Ldn))
    end
    @test worst < 1e-10
    # Isothermal enclosure: sky, leaves and ground at one temperature
    # exchange no net radiation, whatever the emissivities (Kirchhoff).
    for T in (260.0, 300.0), LS in (0.4, 4.0), εg in (0.9, 1.0)
        r = bigleaf_longwave(σSB * T^4, T, T, LS; ε_ground = εg, σ_SB = σSB)
        @test abs(r.canopy) < 1e-10 && abs(r.ground) < 1e-10
    end
    # No canopy: the ground sees the sky directly.
    r = bigleaf_longwave(320.0, 280.0, 290.0, 0.0; ε_ground = 0.95, σ_SB = σSB)
    @test r.ground ≈ 0.95 * (320 - σSB * 290.0^4) rtol = 1e-13
    @test r.canopy == 0
end

@testset "Canopy aerodynamics" begin
    # Full canopy (L+S ≥ 2): z0 = Rz0m z_top, d = Rd z_top (CLM4.5 5.126-5.128).
    a = canopy_roughness(20.0, 5.0, 0.055, 0.67, 0.01)
    @test a.z0m ≈ 1.1 rtol = 1e-13
    @test a.d ≈ 13.4 rtol = 1e-13
    @test canopy_roughness(20.0, 2.0, 0.055, 0.67, 0.01).z0m ≈ 1.1 rtol = 1e-13
    # No canopy: the ground's roughness and no displacement.
    b = canopy_roughness(20.0, 0.0, 0.055, 0.67, 0.01)
    @test b.z0m ≈ 0.01 && b.d == 0
    # Monotone in between.
    zs = [canopy_roughness(20.0, x, 0.055, 0.67, 0.01).z0m for x in 0:0.25:2]
    @test issorted(zs)
    # Under-canopy transfer: dense limit 0.004 (neutral/unstable), bare-soil
    # limit (κ/a)(z0 u*/ν)^−0.45, stable reduction by 1 + γ min(S, 10).
    u = 0.4
    @test undercanopy_transfer_coefficient(u, 50.0, 0.01; T_ac = 290.0, T_g = 295.0, z_top = 20.0) ≈ 0.004 rtol = 1e-12
    @test undercanopy_transfer_coefficient(u, 0.0, 0.01; T_ac = 290.0, T_g = 295.0, z_top = 20.0) ≈
          0.4 / 0.13 * (0.01 * u / 1.5e-5)^(-0.45) rtol = 1e-13
    S = 9.80665 * 20 * 2 / (292 * u^2)
    @test undercanopy_transfer_coefficient(u, 50.0, 0.01; T_ac = 292.0, T_g = 290.0, z_top = 20.0) ≈
          0.004 / (1 + 0.5 * min(S, 10)) rtol = 1e-12
end

end # LandSurface
