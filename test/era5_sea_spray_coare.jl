# Phase 52 — sea spray, the full COARE algorithm and its validation (issue #40).
#
# What is compared with what (see also test/data/README_coare.md):
#
#   * Sea-spray fluxes (Andreas et al. 2015, fast algorithm): against the
#     statements and the Figs. 9-11 sensitivity curves of Andreas, Mahrt &
#     Vickers (2015 AMS 21st BLT extended abstract, the conference version
#     of QJRMS 141, 642-654). The curve values were DIGITISED from the
#     published figure images (pixel positions of the plotted lines, axes
#     calibrated on the frame); they are the authors' *model* output, not
#     observations. Uncertainty of the digitising ≈ ±3 W m⁻².
#   * Full COARE algorithm (bulk fluxes + cool skin + Fairall warm layer):
#     against the NOAA-PSL COARE 3.6 reference code run on its own ship
#     test data (test/data/coare36_*.txt).
#   * Diurnal SST warming: against the OBSERVED peak afternoon warming of
#     42 days from R/V Moana Wave in TOGA COARE, Fairall et al. (1996a),
#     Table 2 (test/data/fairall1996a_table2.txt).

using Test, Printf
using Aeolus
using Aeolus.SurfaceExchange
using Aeolus.AirSeaFlux
using Aeolus.OceanMixedLayer

const _MT = Aeolus.MoistThermodynamics
const _SPRAY_DATA = joinpath(@__DIR__, "data")

function _read_cols(path)
    lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(path))
    hdr = Symbol.(split(lines[1]))
    rows = [parse.(Float64, split(l)) for l in lines[2:end]]
    return Dict(h => [r[i] for r in rows] for (i, h) in enumerate(hdr))
end

_qair_rh(T, p, RH) = (th = _MT.DEFAULT_CONSTANTS; ε = _MT.molmass_ratio(th);
                      e = RH * _MT.saturation_vapor_pressure(T, th); ε * e / (p - (1 - ε) * e))

@testset "Era V — sea spray and COARE validation (#40)" begin

    sp = AndreasSpray()

    @testset "Spray wind functions and the Andreas (2012) drag relation" begin
        # The published low-wind constants are the power laws evaluated at
        # the breakpoints (continuity of eqs. 5.5, 5.6).
        VLbp = sp.VL[2] * sp.VL[4]^sp.VL[3]
        VSbp = sp.VS[2] * sp.VS[4]^sp.VS[3]
        @test VLbp ≈ sp.VL[1] rtol = 3e-3
        @test VSbp ≈ sp.VS[1] rtol = 3e-3
        V1, V2 = spray_wind_functions(sp, 0.5), spray_wind_functions(sp, 1.0)
        @test V2.V_L / V1.V_L ≈ 2^2.39 && V2.V_S / V1.V_S ≈ 2^2.54
        @test spray_wind_functions(sp, 0.05) == (V_L = 1.76e-9, V_S = 3.92e-8)
        # Eq. 2.3: the hyperbola tends to u* = 0.0583 U - 0.243; eq. 2.4:
        # C_DN10 → 3.40e-3 (1 - 4.17/U)².
        for U in (30.0, 50.0, 70.0)
            @test andreas_friction_velocity(U) ≈ 0.0583U - 0.243 atol = 2e-3
            @test (andreas_friction_velocity(U) / U)^2 ≈ 3.40e-3 * (1 - 4.17 / U)^2 rtol = 1e-2
        end
        @test andreas_friction_velocity(0.0) > 0          # finite stress in calm
        @info "Andreas drag" u_star_10 = andreas_friction_velocity(10.0) C_DN10_at_25 = (andreas_friction_velocity(25.0) / 25)^2
    end

    Ta, Tsea, p = 291.15, 293.15, 100000.0      # Andreas et al. Figs. 9, 11
    kw = (T_air = Ta, p = p, S = 34.0, T_sea = Tsea)

    @testset "Droplet microphysics (κ-Köhler, diffusional growth)" begin
        rd = spray_dry_radius(50e-6, 34.0; T = Tsea)
        # Initial saturation ratio of a 34-psu droplet: Andreas et al. (2015,
        # §6.2) have droplets grow above RH ≈ 98.1 %; seawater's water
        # activity at 34 psu is ≈ 0.981. Here: the sea-salt κ = 1.10 of
        # Superdroplet.
        aw = 1 + _MT.kohler_supersaturation(50e-6, rd, SEASALT_KAPPA.κ, Ta)
        @test 0.978 < aw < 0.986
        # Equilibrium temperature: below the air temperature for RH < a_w,
        # above it for RH > a_w, equal at a_w.
        Teq(RH) = spray_equilibrium_temperature(100e-6; RH = RH, kw...)
        @test Teq(0.75) < Teq(0.90) < Ta < Teq(0.99)
        @test abs(Teq(aw) - Ta) < 5e-3
        # Less salt → more evaporation → cooler (fresh water: wet bulb).
        @test spray_equilibrium_temperature(100e-6; RH = 0.9, T_air = Ta, p = p, S = 1.0) < Teq(0.9)
        # Equilibrium radius: r0 at RH = a_w, shrinking with drier air.
        req(RH) = spray_radius_evolution(50e-6; RH = RH, kw...).r_eq
        @test req(aw) ≈ 50e-6 rtol = 1e-4
        @test req(0.75) < req(0.9) < 50e-6 < req(0.99)
        # Near equilibrium the e-folding time is the linear relaxation time
        # 1/λ, λ = -∂(dr/dt)/∂r at r_eq — computed here by finite
        # differences of the growth equation, independently of the RK4 march.
        th = _MT.DEFAULT_CONSTANTS
        G = 1 / (Aeolus.Superdroplet.fk_thermal(Ta, Aeolus.Superdroplet.DEFAULT_MICRO, th) +
                 Aeolus.Superdroplet.fd_vapor(Ta, p, Aeolus.Superdroplet.DEFAULT_MICRO, th))
        RHn = aw - 1e-4
        rn = spray_radius_evolution(50e-6; RH = RHn, kw...)
        f(r) = (RHn - 1 - _MT.kohler_supersaturation(r, rd, SEASALT_KAPPA.κ, Ta, th)) * G / r
        h = 1e-4 * rn.r_eq
        λ = -(f(rn.r_eq + h) - f(rn.r_eq - h)) / 2h
        @test rn.τ_r ≈ 1 / λ rtol = 2e-2
        # Warmer air → faster evaporation → shorter τ_r (Andreas 1990).
        @test spray_radius_evolution(50e-6; RH = 0.9, T_air = 303.15, p = p, S = 34.0).τ_r <
              spray_radius_evolution(50e-6; RH = 0.9, T_air = 283.15, p = p, S = 34.0).τ_r
        # Residence time τ_f = H_s / (2 u_f): a 50-µm droplet falls at ≈ 0.25 m/s.
        @test spray_residence_time(50e-6, 2.0) ≈ 1 / Aeolus.Superdroplet.terminal_velocity(50e-6)
        @info "Spray microphysics (T_air 18 °C, 34 psu)" a_w = aw Teq_minus_Ta_90pct = Teq(0.9) - Ta r_eq50_90pct_um = 1e6req(0.9) τ_r50_90pct_s = spray_radius_evolution(50e-6; RH = 0.9, kw...).τ_r
    end

    @testset "Rain heat flux" begin
        base = (T_air = 300.15, q_air = 0.018, T_s = 302.15, q_s = 0.0255, p = 101000.0)
        @test rain_heat_flux(0.0; base...) == 0
        r1 = rain_heat_flux(1e-3; base...)
        @test r1 > 0                                      # cold, dry-bulb rain cools the sea
        @test rain_heat_flux(2e-3; base...) ≈ 2r1         # linear in the rain rate
        # Fairall et al. (1996b, §2.5): 10 mm/day with a 2 K sea–air
        # difference gives about 2.5 W m⁻² (warm-pool conditions, RH ≈ 80 %).
        Tf = 301.15
        qs = salinity_humidity_factor(34.0) * _MT.saturation_specific_humidity(Tf + 2, 101000.0)
        rF = rain_heat_flux(10 / 86400; T_air = Tf, q_air = _qair_rh(Tf, 101000.0, 0.8),
                            T_s = Tf + 2, q_s = qs, p = 101000.0)
        @test 1.5 < rF < 3.5
        @info "Rain heat flux" W_m2_10mm_per_day = rF
    end

end
