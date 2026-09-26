# Phase 47 (#20) — Ball-Berry / Medlyn stomatal conductance, leaf boundary
# layer, transpiration and the coupled A–gs–ci solution.
#
# Independent references used below (none of them is the code under test):
#   * CLM5 Technical Note (Lawrence et al. 2018), §2.9.3 eq. 2.9.1,
#     §2.9.6 eqs. 2.9.19-2.9.27, Table 2.9.1 (Medlyn g1 per PFT), §2.5.3
#     eq. 2.5.122 (leaf boundary layer, Cv = 0.01 m s^-1/2) and
#     Table 2.5.1 (d_leaf = 0.04 m).
#   * CLM5 source, PhotosynthesisMod.F90 (release-clm5.0): the Ball-Berry
#     quadratic of `ci_func` (aquad = cs, bquad = cs(gb − b) − m An,
#     cquad = −gb (cs b + m An rh)) and the Medlyn quadratic
#     (bquad = −[2(g0 + d) + (g1 d)²/(gb D)],
#      cquad = g0² + [2 g0 + d (1 − g1²/D)] d,  d = 1.6 An/cs),
#     both re-implemented here from the published formulae, larger root.
#   * Medlyn et al. (2011), Glob. Change Biol. 17, 2134-2144, eq. 13:
#     optimal ci/ca = g1 / (g1 + √D) for g0 = 0.

using Test
using Aeolus
using Aeolus.Photosynthesis
using Aeolus.MoistThermodynamics: saturation_vapor_pressure, latent_heat_vapor,
                                  DEFAULT_CONSTANTS

_larger_root(a, b, c) = (-b + sqrt(b^2 - 4a * c)) / (2a)
_leaf_surface_e(gs, gb, ea, ei) = (gb * ea + gs * ei) / (gb + gs)

@testset "Stomatal conductance (Phase 47, #20)" begin

@testset "Ball-Berry closure" begin
    bb = BallBerryStomata()                      # m = 9, b = 0.01 mol m⁻² s⁻¹
    ei = saturation_vapor_pressure(298.15)
    worst_def = 0.0; worst_clm = 0.0
    for An in (0.5, 5.0, 20.0, 40.0), cs in (150.0, 380.0, 900.0),
        rh in (0.2, 0.6, 0.95), gb in (0.05, 0.5, 3.0, 1e3)
        ea = rh * ei
        gs = stomatal_conductance(bb, An, cs, ea, ei, gb)
        hs = _leaf_surface_e(gs, gb, ea, ei) / ei
        # Defining equation with the leaf-surface humidity.
        worst_def = max(worst_def, abs(gs - (bb.g0 + bb.g1 * An * hs / cs)) / gs)
        # CLM ci_func quadratic, larger root.
        gclm = _larger_root(cs, cs * (gb - bb.g0) - bb.g1 * An,
                            -gb * (cs * bb.g0 + bb.g1 * An * rh))
        worst_clm = max(worst_clm, abs(gs / gclm - 1))
        # Leaf-surface humidity lies between the air and the leaf interior.
        @test rh <= hs <= 1
    end
    @test worst_def < 1e-12
    @test worst_clm < 1e-10
    @info "Ball-Berry: defining-equation residual, CLM-quadratic agreement" worst_def worst_clm
    # Perfect coupling (gb = ∞) is the limit of large gb; hs → h_air.
    g∞ = stomatal_conductance(bb, 10.0, 380.0, 0.5ei, ei, Inf)
    @test g∞ ≈ bb.g0 + 9 * 10 * 0.5 / 380 rtol = 1e-14
    @test stomatal_conductance(bb, 10.0, 380.0, 0.5ei, ei, 1e9) ≈ g∞ rtol = 1e-8
    # No net uptake → cuticular/minimum conductance.
    @test stomatal_conductance(bb, -1.0, 380.0, 0.5ei, ei, 1.0) == bb.g0
    @test stomatal_conductance(bb, 0.0, 380.0, 0.5ei, ei, 1.0) == bb.g0
    # Continuity at An = 0⁺ (the quadratic's root tends to g0).
    @test stomatal_conductance(bb, 1e-9, 380.0, 0.5ei, ei, 1.0) ≈ bb.g0 rtol = 1e-7
    # Supersaturated air is capped at hs = 1 (CLM's ceair = min(eair, esat)).
    @test stomatal_conductance(bb, 10.0, 380.0, 1.3ei, ei, Inf) ≈ bb.g0 + 9 * 10 / 380
end

@testset "Medlyn closure" begin
    md = MedlynStomata(g1 = 4.12, g0 = 1e-4)
    ei = saturation_vapor_pressure(298.15)
    worst_def = 0.0; worst_clm = 0.0; ncmp = 0
    for An in (0.5, 5.0, 20.0, 40.0), cs in (150.0, 380.0, 900.0),
        rh in (0.1, 0.5, 0.9), gb in (0.05, 0.5, 3.0, 1e3)
        ea = rh * ei
        gs = stomatal_conductance(md, An, cs, ea, ei, gb)
        Ds = (ei - _leaf_surface_e(gs, gb, ea, ei)) / 1000       # kPa
        # The deficit entering the closure is max(Ds, D_min) ...
        Dc = max(Ds, md.D_min / 1000)
        worst_def = max(worst_def,
                        abs(gs - (md.g0 + 1.6 * (1 + md.g1 / sqrt(Dc)) * An / cs)) / gs)
        # ... and where it is above it the solution is CLM5's:
        # eqs. 2.9.24-2.9.26 (source signs), larger root.
        Ds > md.D_min / 1000 * (1 + 1e-9) || continue
        ncmp += 1
        d = 1.6 * An / cs; Da = (ei - ea) / 1000
        gclm = _larger_root(1.0, -(2 * (md.g0 + d) + (md.g1 * d)^2 / (gb * Da)),
                            md.g0^2 + (2md.g0 + d * (1 - md.g1^2 / Da)) * d)
        worst_clm = max(worst_clm, abs(gs / gclm - 1))
    end
    @test ncmp >= 100                   # 126 of the 432 cases are unbounded
    @test worst_def < 1e-12
    @test worst_clm < 1e-9
    @info "Medlyn: defining-equation residual, CLM5-quadratic agreement" worst_def worst_clm ncmp

    # gs − g0 ∝ A / (cs √D): with gb = ∞ the leaf-surface deficit is the
    # air deficit, and (gs − g0) cs / (1.6 A) − 1 = g1 / √D exactly.
    ea_of(D) = ei - 1000D
    Ds = (0.25, 0.5, 1.0, 2.0, 4.0)
    y = [(stomatal_conductance(md, 12.0, 400.0, ea_of(D), ei, Inf) - md.g0) * 400 / (1.6 * 12) - 1
         for D in Ds]
    slopes = [log(y[i+1] / y[i]) / log(Ds[i+1] / Ds[i]) for i in 1:length(Ds)-1]
    @test all(s -> isapprox(s, -0.5; atol = 1e-12), slopes)
    @test y[3] ≈ md.g1 rtol = 1e-12                             # D = 1 kPa
    # Linear in A, inversely proportional to cs.
    g(A, cs) = stomatal_conductance(md, A, cs, ea_of(1.5), ei, Inf) - md.g0
    @test g(20.0, 400.0) ≈ 2g(10.0, 400.0) rtol = 1e-12
    @test g(10.0, 800.0) ≈ g(10.0, 400.0) / 2 rtol = 1e-12
    # Finite gb lowers the leaf-surface deficit below the air's, so gs is
    # larger than for a perfectly coupled leaf in the same air.
    @test stomatal_conductance(md, 12.0, 400.0, ea_of(1.5), ei, 0.3) >
          stomatal_conductance(md, 12.0, 400.0, ea_of(1.5), ei, Inf)
    # Saturated air: the CLM5 bound D ≥ 50 Pa keeps gs finite.
    gsat = stomatal_conductance(md, 12.0, 400.0, ei, ei, Inf)
    @test gsat ≈ md.g0 + 1.6 * (1 + md.g1 / sqrt(0.05)) * 12 / 400 rtol = 1e-12
    # Weakly coupled leaf in nearly saturated air: CLM5's air-deficit
    # bound would let the humidification feedback run away; the
    # leaf-surface bound caps gs at the same value as in saturated air.
    @test stomatal_conductance(md, 12.0, 400.0, ei - 60.0, ei, 0.1) ≈ gsat rtol = 1e-12
    @test stomatal_conductance(md, 12.0, 400.0, ea_of(1.0), ei, 0.1) <
          stomatal_conductance(md, 12.0, 400.0, ei - 60.0, ei, 0.1)
    @test stomatal_conductance(md, -2.0, 400.0, ea_of(1.0), ei, 1.0) == md.g0
    @test stomatal_conductance(md, 1e-9, 400.0, ea_of(1.0), ei, 1.0) ≈ md.g0 rtol = 1e-6
end

@testset "leaf boundary layer and transpiration" begin
    R = DEFAULT_CONSTANTS.gas_constant
    # CLM5 eq. 2.5.122: U = 1 m/s, d = 0.04 m → rb = 1/(0.01·5) = 20 s m⁻¹.
    T, p = 293.15, 101325.0
    gb = leaf_boundary_layer_conductance(1.0, T, p)
    @test gb ≈ (1 / 20) * p / (R * T) rtol = 1e-12            # 2.079 mol m⁻² s⁻¹
    @info "leaf boundary-layer conductance, U = 1 m/s, d = 4 cm, 20 °C" gb
    @test leaf_boundary_layer_conductance(4.0, T, p) ≈ 2gb rtol = 1e-12     # √U
    @test leaf_boundary_layer_conductance(1.0, T, p; d_leaf = 0.16) ≈ gb / 2 rtol = 1e-12
    @test leaf_boundary_layer_conductance(0.0, T, p) == 0
    # Molar density of an ideal gas is p/(RT): 1 m s⁻¹ at 0 °C, 1 atm is
    # the reciprocal molar volume, 44.615 mol m⁻³.
    @test molar_conductance(1.0, 273.15, 101325.0) ≈ 1 / 0.022413969 rtol = 1e-6

    # Transpiration: series resistances, driven by the leaf-to-air
    # vapour-pressure difference. 25 °C leaf, 50 % RH air.
    Tl = 298.15; ei = saturation_vapor_pressure(Tl); ea = 0.5ei
    E = leaf_transpiration(0.2, 1.0, Tl, ea, p)
    @test E ≈ (0.2 * 1.0 / 1.2) * (ei - ea) / p rtol = 1e-14
    @test leaf_transpiration(0.2, Inf, Tl, ea, p) ≈ 0.2 * (ei - ea) / p
    @test leaf_transpiration(0.2, 1.0, Tl, ei, p) == 0           # saturated air
    @test leaf_transpiration(0.2, 1.0, Tl, 1.1ei, p) < 0         # condensation
    # Latent heat, W m⁻²: λ(25 °C) M_w E ≈ 2.44 MJ/kg × 0.018 × E.
    LE = latent_heat_vapor(Tl) * DEFAULT_CONSTANTS.molmass_water * E
    @info "E, LE for gs = 0.2, gb = 1 mol m⁻² s⁻¹, 25 °C, RH 50 %" E LE
    @test 2.43e6 < latent_heat_vapor(Tl) < 2.45e6
    @test 100 < LE < 150         # = 0.1667 × 1584 Pa / 101325 Pa × 44 kJ/mol
end

@testset "coupled A–gs–ci: flux consistency and convergence" begin
    # Every combination of pathway, closure, coupling, light, temperature,
    # CO₂ and humidity must converge to a state satisfying CLM5 eq.
    # 2.9.19 (three equal CO₂ fluxes) and eq. 2.9.20 (equal H₂O fluxes).
    c3 = pft_photosynthesis_parameters(:bdt_temperate)
    c4 = pft_photosynthesis_parameters(:c4_grass)
    models = (BallBerryStomata(), MedlynStomata(),
              BallBerryStomata(g1 = 4.0, g0 = 0.04), MedlynStomata(g1 = 1.62, g0 = 0.0))
    worst_co2 = 0.0; worst_h2o = 0.0; n = 0; nconv = 0; maxit = 0
    for p in (c3, c4), m in models, gb in (0.1, 1.0, 5.0, Inf),
        Q in (0.0, 50.0, 400.0, 2000.0), Tc in (5.0, 25.0, 40.0),
        ca in (100.0, 400.0, 1500.0), rh in (0.1, 0.6, 0.98)
        T = Tc + 273.15; ei = saturation_vapor_pressure(T); ea = rh * ei
        r = leaf_gas_exchange(p, m; ca = ca, Q = Q, T_leaf = T, e_air = ea, gb = gb)
        n += 1; nconv += r.converged; maxit = max(maxit, r.iterations)
        scale = max(abs(r.An), 1e-3)
        f_total = r.gc * (ca - r.ci)
        f_pore = r.gs / 1.6 * (r.cs - r.ci)
        f_bl = isinf(gb) ? f_total : gb / 1.4 * (ca - r.cs)
        worst_co2 = max(worst_co2, abs(f_total - r.An) / scale,
                        abs(f_pore - r.An) / scale, abs(f_bl - r.An) / scale)
        if !isinf(gb)
            es = r.hs * ei
            worst_h2o = max(worst_h2o, abs(r.gs * (ei - es) - gb * (es - ea)) /
                                       max(gb * abs(ei - ea), 1e-12))
            @test r.E ≈ r.gs * (ei - es) / 101325.0 rtol = 1e-9 atol = 1e-15
        end
        @test r.An ≈ leaf_assimilation(p, r.ci, Q, T).An rtol = 1e-12
    end
    @test nconv == n
    @test worst_co2 < 1e-9
    # es ≈ ei when gs ≫ gb, so ei − es carries a cancellation error of
    # order eps·ei/(ei − es); 1e-10 is that roundoff, not solver error.
    @test worst_h2o < 1e-10
    @info "coupled solve: $n cases" converged = nconv max_iterations = maxit worst_co2 worst_h2o
end

@testset "Medlyn optimal ci/ca = g1/(g1 + √D)" begin
    # Medlyn et al. (2011) eq. 13, for g0 = 0 and a perfectly coupled leaf:
    # ci/ca is independent of A, light, temperature and ca. g0 = 0 is
    # raised to the 1 µmol m⁻² s⁻¹ floor, which shifts ci/ca by ~1e-5.
    p = pft_photosynthesis_parameters(:bdt_temperate)
    worst = 0.0
    for g1 in (2.35, 4.45), D in (0.5, 1.0, 2.0, 3.0), Q in (200.0, 1500.0),
        Tc in (15.0, 30.0), ca in (300.0, 700.0)
        T = Tc + 273.15; ei = saturation_vapor_pressure(T)
        m = MedlynStomata(g1 = g1, g0 = 0.0)
        r = leaf_gas_exchange(p, m; ca = ca, Q = Q, T_leaf = T, e_air = ei - 1000D)
        χ = g1 / (g1 + sqrt(D))
        worst = max(worst, abs(r.ci / ca - χ))
        @test r.An ≈ net_assimilation(p, χ * ca, Q, T) rtol = 1e-3
    end
    @test worst < 1e-4
    @info "Medlyn ci/ca vs g1/(g1+√D)" max_abs_err = worst
end

@testset "ci/ca at typical conditions" begin
    # 25 °C, 400 W m⁻² PAR (85 % absorbed), U = 1 m s⁻¹, D = 1.5 kPa.
    T = 298.15; ei = saturation_vapor_pressure(T); ea = ei - 1500.0
    gb = leaf_boundary_layer_conductance(1.0, T, 101325.0)
    Q = 0.85 * par_to_photon_flux(400.0)
    ratios = Dict{Tuple{Symbol,Symbol},Float64}()
    for pft in keys(CLM5_MEDLYN_G1), kind in (:ball_berry, :medlyn)
        p = pft_photosynthesis_parameters(pft)
        m = pft_stomatal_model(pft; model = kind)
        r = leaf_gas_exchange(p, m; ca = 400.0, Q = Q, T_leaf = T, e_air = ea, gb = gb)
        ratios[(pft, kind)] = r.ci / 400
    end
    c3 = [v for ((k, _), v) in ratios if CLM45_PFT_PHOTOSYNTHESIS[k].pathway === :C3]
    c4 = [v for ((k, _), v) in ratios if CLM45_PFT_PHOTOSYNTHESIS[k].pathway === :C4]
    # C3 leaves: ≈ 0.7 (0.65-0.8 across CLM's C3 PFTs and both closures).
    @test all(x -> 0.6 < x < 0.85, c3)
    @test abs(sum(c3) / length(c3) - 0.7) < 0.1
    # C4 leaves draw ci down much further (CO₂-concentrating mechanism).
    @test all(x -> x < 0.65, c4)
    @test maximum(c4) < minimum(c3)
    @info "ci/ca at D = 1.5 kPa, 25 °C" C3_mean = sum(c3) / length(c3) C3_range = extrema(c3) C4_range = extrema(c4) bdt_BB = ratios[(:bdt_temperate, :ball_berry)] bdt_Medlyn = ratios[(:bdt_temperate, :medlyn)] net_Medlyn = ratios[(:net_temperate, :medlyn)]
end

@testset "stomatal closure at high CO₂" begin
    T = 298.15; ei = saturation_vapor_pressure(T); ea = ei - 1500.0
    gb = leaf_boundary_layer_conductance(1.0, T, 101325.0)
    Q = 0.85 * par_to_photon_flux(400.0)
    for pft in (:bdt_temperate, :c4_grass), kind in (:ball_berry, :medlyn)
        p = pft_photosynthesis_parameters(pft); m = pft_stomatal_model(pft; model = kind)
        r = [leaf_gas_exchange(p, m; ca = ca, Q = Q, T_leaf = T, e_air = ea, gb = gb)
             for ca in (400.0, 567.0, 800.0)]
        # Stomata close, assimilation does not fall, transpiration falls,
        # and water-use efficiency A/E rises.
        @test r[1].gs > r[2].gs > r[3].gs
        @test r[1].An <= r[2].An <= r[3].An
        @test r[1].E > r[2].E > r[3].E
        @test r[1].An / r[1].E < r[2].An / r[2].E < r[3].An / r[3].E
        # ci tracks ca. For C3 leaves ci/ca is nearly conserved while ca
        # doubles. C4 assimilation is CO₂-saturated, so gs − g0 ∝ 1/cs and
        # the fixed intercept g0 (0.04 mol m⁻² s⁻¹ for Ball-Berry) becomes
        # a growing share of gs: ci/ca rises, by 0.076 (Ball-Berry).
        Δχ = r[3].ci / 800 - r[1].ci / 400
        if CLM45_PFT_PHOTOSYNTHESIS[pft].pathway === :C3
            @test abs(Δχ) < 0.01
        else
            @test 0 < Δχ < 0.1
        end
        @info "gs response to CO₂, $pft $kind" gs_567_vs_400 = r[2].gs / r[1].gs - 1 gs_800_vs_400 = r[3].gs / r[1].gs - 1 Δχ An_800_vs_400 = r[3].An / r[1].An - 1
    end
    # Perfectly coupled Medlyn leaf with g0 → 0: gs ∝ An / ca exactly.
    p = pft_photosynthesis_parameters(:bdt_temperate); m = MedlynStomata(g0 = 0.0)
    r4 = leaf_gas_exchange(p, m; ca = 400.0, Q = 800.0, T_leaf = T, e_air = ea)
    r8 = leaf_gas_exchange(p, m; ca = 800.0, Q = 800.0, T_leaf = T, e_air = ea)
    @test r8.gs / r4.gs ≈ (r8.An / 800) / (r4.An / 400) rtol = 1e-4
end

@testset "light, humidity and darkness" begin
    p = pft_photosynthesis_parameters(:bdt_temperate)
    T = 298.15; ei = saturation_vapor_pressure(T); gb = 1.0
    for m in (BallBerryStomata(), MedlynStomata())
        # Stomata open with light ...
        gsQ = [leaf_gas_exchange(p, m; ca = 400.0, Q = Q, T_leaf = T, e_air = 0.5ei, gb = gb).gs
               for Q in 0.0:100.0:2000.0]
        @test all(diff(gsQ) .> 0)
        # ... and close as the air dries.
        gsD = [leaf_gas_exchange(p, m; ca = 400.0, Q = 1000.0, T_leaf = T,
                                 e_air = ei - D, gb = gb).gs for D in 250.0:250.0:3000.0]
        @test all(diff(gsD) .< 0)
        # Darkness: An = −Rd, stomata at the intercept, and respired CO₂
        # raises ci above ca by exactly Rd (1.4/gb + 1.6/g0).
        r = leaf_gas_exchange(p, m; ca = 400.0, Q = 0.0, T_leaf = T, e_air = 0.5ei, gb = gb)
        @test r.An ≈ -r.Rd
        @test r.gs == m.g0
        @test r.ci ≈ 400 + r.Rd * (1.4 / gb + 1.6 / m.g0) rtol = 1e-10
        @test r.cs > 400
    end
end

@testset "soil-water stress hook β" begin
    p = pft_photosynthesis_parameters(:bdt_temperate)
    T = 298.15; ei = saturation_vapor_pressure(T); ea = ei - 1500.0
    Q = 1200.0; gb = 1.0
    for m in (BallBerryStomata(), MedlynStomata())
        base = leaf_gas_exchange(p, m; ca = 400.0, Q = Q, T_leaf = T, e_air = ea, gb = gb)
        @test leaf_gas_exchange(p, m; ca = 400.0, Q = Q, T_leaf = T, e_air = ea, gb = gb,
                                β = 1.0).gs == base.gs
        rs = [leaf_gas_exchange(p, m; ca = 400.0, Q = Q, T_leaf = T, e_air = ea, gb = gb, β = β)
              for β in 1.0:-0.1:0.0]
        @test all(diff([r.gs for r in rs]) .< 0)
        @test all(diff([r.E for r in rs]) .< 0)
        @test all(diff([r.A for r in rs]) .< 0)
        # β = 0: no Rubisco and no day respiration → no exchange of CO₂,
        # stomata at CLM's 1 µmol m⁻² s⁻¹ floor.
        @test rs[end].An == 0
        @test rs[end].gs == 1e-6
        # Slope stress (:g1) closes stomata without touching Vcmax.
        rg = leaf_gas_exchange(p, m; ca = 400.0, Q = Q, T_leaf = T, e_air = ea, gb = gb,
                               β = 0.5, stress_targets = (:g1,))
        @test rg.gs < base.gs
        @test rg.ci < base.ci
    end
    @test apply_water_stress(p, 0.3).Vcmax25 ≈ 0.3 * p.Vcmax25
    @test apply_water_stress(p, 0.3).Jmax25 == p.Jmax25
    @test apply_water_stress(p, 0.3; targets = (:g1,)).Vcmax25 == p.Vcmax25
    @test apply_water_stress(BallBerryStomata(), 0.5).g0 ≈ 0.005
    @test_throws DomainError apply_water_stress(p, 1.5)
    @test_throws ArgumentError apply_water_stress(p, 0.5; targets = (:jmax,))
end

@testset "per-PFT stomatal parameters" begin
    # CLM5 Table 2.9.1 spot checks and structure.
    @test CLM5_MEDLYN_G1[:net_temperate] == 2.35
    @test CLM5_MEDLYN_G1[:c3_grass] == 5.25
    @test CLM5_MEDLYN_G1[:c4_grass] == 1.62
    @test length(CLM5_MEDLYN_G1) == 17
    @test all(k -> haskey(CLM45_PFT_PHOTOSYNTHESIS, k), keys(CLM5_MEDLYN_G1))
    # C4 plants have lower g1 than C3 (lower ci/ca at optimum).
    for (k, g1) in CLM5_MEDLYN_G1
        CLM45_PFT_PHOTOSYNTHESIS[k].pathway === :C4 && @test g1 < 2
    end
    @test pft_stomatal_model(:c4_grass; model = :ball_berry) == BallBerryStomata(g1 = 4.0, g0 = 0.04)
    @test pft_stomatal_model(:c3_grass; model = :ball_berry) == BallBerryStomata(g1 = 9.0, g0 = 0.01)
    @test pft_stomatal_model(:c3_grass).g0 == 1e-4
    @test_throws ArgumentError pft_stomatal_model(:crop)                 # no Table 2.9.1 row
    @test pft_stomatal_model(:crop; model = :ball_berry).g1 == 9.0
    @test_throws ArgumentError pft_stomatal_model(:oak)
end

end # testset
