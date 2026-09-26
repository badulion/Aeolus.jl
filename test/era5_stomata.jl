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
    worst_def = 0.0; worst_clm = 0.0
    for An in (0.5, 5.0, 20.0, 40.0), cs in (150.0, 380.0, 900.0),
        rh in (0.1, 0.5, 0.9), gb in (0.05, 0.5, 3.0, 1e3)
        ea = rh * ei
        gs = stomatal_conductance(md, An, cs, ea, ei, gb)
        Ds = (ei - _leaf_surface_e(gs, gb, ea, ei)) / 1000       # kPa
        worst_def = max(worst_def,
                        abs(gs - (md.g0 + 1.6 * (1 + md.g1 / sqrt(Ds)) * An / cs)) / gs)
        # CLM5 eqs. 2.9.24-2.9.26 (source signs), larger root.
        d = 1.6 * An / cs; Da = (ei - ea) / 1000
        gclm = _larger_root(1.0, -(2 * (md.g0 + d) + (md.g1 * d)^2 / (gb * Da)),
                            md.g0^2 + (2md.g0 + d * (1 - md.g1^2 / Da)) * d)
        worst_clm = max(worst_clm, abs(gs / gclm - 1))
    end
    @test worst_def < 1e-12
    @test worst_clm < 1e-9
    @info "Medlyn: defining-equation residual, CLM5-quadratic agreement" worst_def worst_clm

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

end # testset
