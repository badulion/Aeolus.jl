# Phase 47 (#20) — Ball-Berry / Medlyn stomatal conductance, leaf boundary
# layer, transpiration and the coupled A–gs–ci solution.
#
# Independent references used below (none of them is the code under test):
#   * CLM5 Technical Note (Lawrence et al. 2018), §2.9.6 eqs. 2.9.19-2.9.27,
#     Table 2.9.1 (Medlyn g1 per PFT), §2.5.3 eq. 2.5.122 (leaf boundary
#     layer, Cv = 0.01 m s^-1/2, d_leaf = 0.04 m).
#   * CLM5 source, PhotosynthesisMod.F90: the Ball-Berry quadratic of
#     `ci_func` and the Medlyn quadratic, re-implemented here, larger root.
#   * Medlyn et al. (2011), Glob. Change Biol. 17, eq. 13:
#     optimal ci/ca = g1 / (g1 + √D) for g0 = 0.

using Test
using Aeolus
using Aeolus.Photosynthesis
using Aeolus.MoistThermodynamics: saturation_vapor_pressure, latent_heat_vapor,
                                  DEFAULT_CONSTANTS

_larger_root(a, b, c) = (-b + sqrt(b^2 - 4a * c)) / (2a)
_leaf_surface_e(gs, gb, ea, ei) = (gb * ea + gs * ei) / (gb + gs)

@testset "Stomatal conductance (Phase 47, #20)" begin

@testset "Ball-Berry closure vs CLM quadratic" begin
    bb = BallBerryStomata()
    ei = saturation_vapor_pressure(298.15)
    worst_def = 0.0; worst_clm = 0.0; hs_ok = true
    for An in (0.5, 20.0), cs in (150.0, 900.0), rh in (0.2, 0.95), gb in (0.05, 3.0)
        ea = rh * ei
        gs = stomatal_conductance(bb, An, cs, ea, ei, gb)
        hs = _leaf_surface_e(gs, gb, ea, ei) / ei
        worst_def = max(worst_def, abs(gs - (bb.g0 + bb.g1 * An * hs / cs)) / gs)
        gclm = _larger_root(cs, cs * (gb - bb.g0) - bb.g1 * An,
                            -gb * (cs * bb.g0 + bb.g1 * An * rh))
        worst_clm = max(worst_clm, abs(gs / gclm - 1))
        hs_ok &= rh <= hs <= 1
    end
    @test hs_ok
    @test worst_def < 1e-12
    @test worst_clm < 1e-10
    @test stomatal_conductance(bb, 10.0, 380.0, 0.5ei, ei, Inf) ≈ bb.g0 + 9 * 10 * 0.5 / 380 rtol = 1e-14
    @test stomatal_conductance(bb, -1.0, 380.0, 0.5ei, ei, 1.0) == bb.g0
end

@testset "Medlyn closure vs CLM5 quadratic" begin
    md = MedlynStomata(g1 = 4.12, g0 = 1e-4)
    ei = saturation_vapor_pressure(298.15)
    worst_def = 0.0; worst_clm = 0.0; ncmp = 0
    for An in (0.5, 20.0), cs in (150.0, 900.0), rh in (0.1, 0.5), gb in (0.05, 3.0)
        ea = rh * ei
        gs = stomatal_conductance(md, An, cs, ea, ei, gb)
        Ds = (ei - _leaf_surface_e(gs, gb, ea, ei)) / 1000
        Dc = max(Ds, md.D_min / 1000)
        worst_def = max(worst_def,
                        abs(gs - (md.g0 + 1.6 * (1 + md.g1 / sqrt(Dc)) * An / cs)) / gs)
        Ds > md.D_min / 1000 * (1 + 1e-9) || continue
        ncmp += 1
        d = 1.6 * An / cs; Da = (ei - ea) / 1000
        gclm = _larger_root(1.0, -(2 * (md.g0 + d) + (md.g1 * d)^2 / (gb * Da)),
                            md.g0^2 + (2md.g0 + d * (1 - md.g1^2 / Da)) * d)
        worst_clm = max(worst_clm, abs(gs / gclm - 1))
    end
    @test ncmp >= 8
    @test worst_def < 1e-12
    @test worst_clm < 1e-9
    # gs − g0 ∝ 1/√D: with gb = ∞, (gs − g0) cs / (1.6 A) − 1 = g1/√D.
    y(D) = (stomatal_conductance(md, 12.0, 400.0, ei - 1000D, ei, Inf) - md.g0) * 400 / (1.6 * 12) - 1
    @test y(1.0) ≈ md.g1 rtol = 1e-12
    @test y(4.0) ≈ md.g1 / 2 rtol = 1e-12
    # Saturated air: the CLM5 bound D ≥ 50 Pa keeps gs finite.
    @test stomatal_conductance(md, 12.0, 400.0, ei, ei, Inf) ≈
          md.g0 + 1.6 * (1 + md.g1 / sqrt(0.05)) * 12 / 400 rtol = 1e-12
end

@testset "leaf boundary layer and transpiration" begin
    R = DEFAULT_CONSTANTS.gas_constant
    T, p = 293.15, 101325.0
    gb = leaf_boundary_layer_conductance(1.0, T, p)
    @test gb ≈ (1 / 20) * p / (R * T) rtol = 1e-12        # CLM5 eq. 2.5.122
    @test leaf_boundary_layer_conductance(4.0, T, p) ≈ 2gb rtol = 1e-12     # √U
    Tl = 298.15; ei = saturation_vapor_pressure(Tl); ea = 0.5ei
    @test leaf_transpiration(0.2, 1.0, Tl, ea, p) ≈ (0.2 * 1.0 / 1.2) * (ei - ea) / p rtol = 1e-14
    @test leaf_transpiration(0.2, 1.0, Tl, ei, p) == 0           # saturated air
end

@testset "coupled A–gs–ci: flux consistency" begin
    # CLM5 eq. 2.9.19 (three equal CO₂ fluxes) and eq. 2.9.20 (equal H₂O
    # fluxes) at the converged state, for both pathways and closures.
    worst_co2 = 0.0; worst_h2o = 0.0; worst_A = 0.0; allconv = true
    for pft in (:bdt_temperate, :c4_grass), m in (BallBerryStomata(), MedlynStomata()),
        gb in (0.1, Inf), (Q, Tc, ca, rh) in ((50.0, 5.0, 100.0, 0.98), (2000.0, 40.0, 1500.0, 0.1),
                                              (400.0, 25.0, 400.0, 0.6))
        p = pft_photosynthesis_parameters(pft)
        T = Tc + 273.15; ei = saturation_vapor_pressure(T); ea = rh * ei
        r = leaf_gas_exchange(p, m; ca = ca, Q = Q, T_leaf = T, e_air = ea, gb = gb)
        allconv &= r.converged
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
        end
        worst_A = max(worst_A, abs(r.An - leaf_assimilation(p, r.ci, Q, T).An) / scale)
    end
    @test allconv
    @test worst_co2 < 1e-9
    @test worst_h2o < 1e-10   # cancellation in ei − es when gs ≫ gb
    @test worst_A < 1e-12
end

@testset "Medlyn optimal ci/ca = g1/(g1 + √D)" begin
    # g0 = 0 is raised to the 1 µmol m⁻² s⁻¹ floor (shift ~1e-5).
    p = pft_photosynthesis_parameters(:bdt_temperate)
    worst = 0.0
    for (g1, D, Q, Tc, ca) in ((2.35, 0.5, 200.0, 15.0, 300.0), (4.45, 3.0, 1500.0, 30.0, 700.0))
        T = Tc + 273.15; ei = saturation_vapor_pressure(T)
        r = leaf_gas_exchange(p, MedlynStomata(g1 = g1, g0 = 0.0); ca = ca, Q = Q,
                              T_leaf = T, e_air = ei - 1000D)
        worst = max(worst, abs(r.ci / ca - g1 / (g1 + sqrt(D))))
    end
    @test worst < 1e-4
end

@testset "high CO₂ closes stomata; darkness; water stress" begin
    T = 298.15; ei = saturation_vapor_pressure(T); ea = ei - 1500.0
    gb = leaf_boundary_layer_conductance(1.0, T, 101325.0)
    p = pft_photosynthesis_parameters(:bdt_temperate); m = MedlynStomata()
    r = [leaf_gas_exchange(p, m; ca = ca, Q = 800.0, T_leaf = T, e_air = ea, gb = gb)
         for ca in (400.0, 800.0)]
    @test r[1].gs > r[2].gs && r[1].E > r[2].E && r[1].An <= r[2].An
    @test 0.6 < r[1].ci / 400 < 0.85                 # C3 ci/ca ≈ 0.7
    # Darkness: An = −Rd, stomata at g0, ci = ca + Rd (1.4/gb + 1.6/g0).
    rd = leaf_gas_exchange(p, m; ca = 400.0, Q = 0.0, T_leaf = T, e_air = 0.5ei, gb = 1.0)
    @test rd.An ≈ -rd.Rd
    @test rd.ci ≈ 400 + rd.Rd * (1.4 / 1.0 + 1.6 / m.g0) rtol = 1e-10
    # β = 0: no exchange, stomata at CLM's 1 µmol m⁻² s⁻¹ floor.
    r0 = leaf_gas_exchange(p, m; ca = 400.0, Q = 1200.0, T_leaf = T, e_air = ea, gb = 1.0, β = 0.0)
    @test r0.An == 0 && r0.gs == 1e-6
    @test apply_water_stress(p, 0.3).Vcmax25 ≈ 0.3 * p.Vcmax25
    @test CLM5_MEDLYN_G1[:c4_grass] == 1.62 && length(CLM5_MEDLYN_G1) == 17
end

end # testset
