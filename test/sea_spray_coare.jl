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
#   * Diurnal SST warming against the OBSERVED peak afternoon warming of
#     42 days from R/V Moana Wave in TOGA COARE (Fairall et al. 1996a,
#     Table 2) is a validation run: validation/sea_spray_coare.jl.

using Test
using Aeolus
using Aeolus.SurfaceExchange
using Aeolus.AirSeaFlux
using Aeolus.OceanMixedLayer

const _MT = Aeolus.MoistThermodynamics
const _SPRAY_DATA = joinpath(@__DIR__, "data")

isdefined(@__MODULE__, :read_coare_subset) || include(joinpath(@__DIR__, "support", "coare36_subset.jl"))

_qair_rh(T, p, RH) = (th = _MT.DEFAULT_CONSTANTS; ε = _MT.molmass_ratio(th);
                      e = RH * _MT.saturation_vapor_pressure(T, th); ε * e / (p - (1 - ε) * e))

@testset "Era V — sea spray and COARE validation (#40)" begin

    sp = AndreasSpray()

    @testset "Spray wind functions and the Andreas (2012) drag relation" begin
        # published low-wind constants = power laws at the breakpoints (eqs. 5.5, 5.6)
        @test sp.VL[2] * sp.VL[4]^sp.VL[3] ≈ sp.VL[1] rtol = 3e-3
        @test sp.VS[2] * sp.VS[4]^sp.VS[3] ≈ sp.VS[1] rtol = 3e-3
        # eq. 2.3: the hyperbola tends to u* = 0.0583 U - 0.243
        @test maximum(U -> abs(andreas_friction_velocity(U) - (0.0583U - 0.243)), (30.0, 50.0, 70.0)) < 2e-3
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
        # Equilibrium radius: r0 at RH = a_w, shrinking with drier air.
        req(RH) = spray_radius_evolution(50e-6; RH = RH, kw...).r_eq
        @test req(aw) ≈ 50e-6 rtol = 1e-4
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
        # Residence time τ_f = H_s / (2 u_f): a 50-µm droplet falls at ≈ 0.25 m/s.
        @test spray_residence_time(50e-6, 2.0) ≈ 1 / Aeolus.Superdroplet.terminal_velocity(50e-6)
    end

    @testset "Spray fluxes vs Andreas et al. (2015), Figs. 9-11 and text" begin
        us25 = andreas_friction_velocity(25.0)
        flux(; RH = 0.9, U = 25.0, Ts = Tsea, Tair = Ta, Hs = 10.0) =
            sea_spray_fluxes(sp; u_star = andreas_friction_velocity(U), T_sea = Ts,
                             T_air = Tair, RH = RH, p = p, S = 34.0, H_s = Hs)
        # (a) Sensible spray flux — independent of the wave height, because
        # 100-µm droplets reach T_eq before falling back. Digitised values
        # (W m⁻²) of H_s,sp:
        fig9 = [(15.0, 15.3), (25.0, 97.0), (30.0, 167.5), (35.0, 261.0)]          # vs U10
        fig10 = [(1.0, 81.7), (5.0, 84.6), (10.0, 88.8), (15.0, 92.9), (20.0, 97.2),
                 (25.0, 101.2), (29.0, 104.3)]                                     # vs Θs, T10 = Θs - 2
        fig11 = [(0.75, 157.7), (0.80, 136.6), (0.85, 117.7), (0.90, 97.0), (0.925, 87.2),
                 (0.95, 77.3), (0.97, 70.1)]                                        # vs RH
        pairs = vcat([(flux(U = U).H_S, v) for (U, v) in fig9],
                     [(flux(Ts = 273.15 + T, Tair = 271.15 + T).H_S, v) for (T, v) in fig10],
                     [(flux(RH = RH).H_S, v) for (RH, v) in fig11])
        @test flux(Hs = 5.0).H_S == flux(Hs = 20.0).H_S
        # Tolerance: 12 % plus 6 W m⁻² for the digitising (on Fig. 9's
        # 1300 W m⁻² axis one pixel is 2.2 W m⁻², and near 15 m/s the dotted
        # curve runs on top of the interfacial line).
        @test all(abs(x - v) < 0.12v + 6 for (x, v) in pairs)
        errS = [x / v - 1 for (x, v) in pairs if v > 50]
        # (b) Latent spray flux — depends on the significant wave height via
        # the residence time, and the figures do not state H_s (Andreas et
        # al. used the depth-dependent Andreas & Wang 2007 fit, not
        # implemented here). One parameter, H_s at 25 m/s, is therefore
        # calibrated on the Fig. 11 value at RH = 90 %; the other six RH
        # points are then predictions.
        fig11L = [(0.75, 426.8), (0.80, 360.7), (0.85, 284.4), (0.925, 132.1),
                  (0.95, 71.4), (0.97, 21.1)]
        lo, hi = 1.0, 30.0
        for _ in 1:60
            mid = (lo + hi) / 2
            flux(Hs = mid).H_L < 186.9 ? (lo = mid) : (hi = mid)
        end
        Hs_cal = (lo + hi) / 2
        errL = [flux(RH = RH, Hs = Hs_cal).H_L - v for (RH, v) in fig11L]
        @test maximum(abs, errL) < 45            # W m⁻²
        @test 4.0 < Hs_cal < 12.0                # vs 15.5 m for a fully developed sea
        # (c) The sign change of H_L,sp: "above about 98.1 %, the droplets
        # grow rather than evaporate" (34 psu). Located by bisection.
        lo, hi = 0.95, 0.995
        for _ in 1:50
            mid = (lo + hi) / 2
            flux(RH = mid).H_L > 0 ? (lo = mid) : (hi = mid)
        end
        RH0 = (lo + hi) / 2
        @test abs(RH0 - 0.981) < 3e-3
        # Energy bookkeeping of the spray: E = H_L / L_v.
        s = flux()
        @test s.E ≈ s.H_L / _MT.latent_heat_vapor(Ta)
    end

    @testset "Rain heat flux" begin
        base = (T_air = 300.15, q_air = 0.018, T_s = 302.15, q_s = 0.0255, p = 101000.0)
        r1 = rain_heat_flux(1e-3; base...)
        @test r1 > 0 && rain_heat_flux(2e-3; base...) ≈ 2r1   # cools the sea; linear in the rate
        # Fairall et al. (1996b, §2.5): 10 mm/day with a 2 K sea–air
        # difference gives about 2.5 W m⁻² (warm-pool conditions, RH ≈ 80 %).
        Tf = 301.15
        qs = salinity_humidity_factor(34.0) * _MT.saturation_specific_humidity(Tf + 2, 101000.0)
        rF = rain_heat_flux(10 / 86400; T_air = Tf, q_air = _qair_rh(Tf, 101000.0, 0.8),
                            T_s = Tf + 2, q_s = qs, p = 101000.0)
        @test 1.5 < rF < 3.5
    end

    # ---- COARE 3.6 reference: full algorithm ----------------------------
    o = read_coare_subset(joinpath(_SPRAY_DATA, "coare36_warm_cool_subset.txt"))
    m = read_coare_subset(joinpath(_SPRAY_DATA, "coare36_met_subset.txt"))
    n = length(o[:jd])

    @testset "Full COARE (bulk + cool skin + warm layer) vs COARE 3.6 reference" begin
        recs = map(1:n) do i
            T = m[:ta][i] + 273.15
            ptq = (m[:P][i] - 0.125 * 17) * 100        # reference: q at the sensor's pressure
            (jd = m[:jd][i], lon = o[:lon][i], U = m[:u][i], T_air = T,
             q_air = _qair_rh(T, ptq, m[:rh][i] / 100), p = m[:P][i] * 100,
             T_sea = o[:tsg][i] + 273.15, sw_net = o[:sw_net][i], lw_down = o[:lw_dn][i],
             rain = m[:rain][i] / 3600, S = o[:Ss][i])
        end
        # Configuration of the reference: heights 18 / 17 m, ε = 0.97,
        # seawater c_w = 4000 (the package default is the fresh-water
        # cp_water; see the PR notes), thermosalinograph at 5.334 m. The
        # net shortwave is the reference's (its Payne albedo is not part
        # of what is tested here).
        sols = coare_warm_layer_series(coare35_bulk(), recs; sensor_depth = 5.334, z_u = 18.0,
                                       z_t = 17.0, emissivity = 0.97, c_w = 4000.0)
        us = [s.bulk.surface.u_star for s in sols]
        g(f) = [getfield(s, f) for s in sols]
        e_us = maximum(abs.(us ./ o[:usr] .- 1))
        e_τ = maximum(abs.(g(:τ) ./ o[:tau] .- 1))
        e_H = maximum(abs.(g(:H) .- o[:hsb]))
        e_LE = maximum(abs.(g(:LE) .- o[:hlb]))
        rms_LE = sqrt(sum(abs2, g(:LE) .- o[:hlb]) / n)
        e_cool = maximum(abs.(g(:ΔT_cool) .- o[:dT_skin]))
        e_warm = maximum(abs.(g(:ΔT_warm) .- o[:dT_warm_to_skin]))
        @test e_us < 5e-3 && e_τ < 1e-2
        @test e_H < 1.5                          # W m⁻² (H ≈ 3-40 W m⁻²)
        @test e_LE < 3.0 && rms_LE < 2.0         # W m⁻² (LE ≈ 90-250 W m⁻²)
        @test e_cool < 0.03                      # K (ΔT_cool 0.18-0.37 K)
        @test e_warm < 0.01                      # K (warming to 5.3 m up to 0.33 K)
        # Rain heat flux: q_sat in the wet-bulb factor (here) vs the
        # reference's actual q: ours is smaller by the α ratio.
        wet = o[:hrain] .> 0.5
        ratio = g(:H_rain)[wet] ./ o[:hrain][wet]
        @test all(0.7 .< ratio .< 1.0)
    end

    @testset "COARE assembly: spray, cool skin, bookkeeping" begin
        qa = _qair_rh(299.15, 101000.0, 0.8)
        args = (U = 22.0, z_u = 10.0, T_air = 299.15, q_air = qa, p = 101000.0, T_sea = 301.15,
                S = 35.0, sw_net = 0.0, lw_down = 400.0)
        a = coare_fluxes(coare35_bulk(); args...)
        b = coare_fluxes(coare35_bulk(); args..., spray = sp)
        # No feedback of the spray on the interfacial solve (Andreas et al.
        # 2015, §6.1): identical interfacial fluxes and cool skin.
        @test b.H == a.H && b.LE == a.LE && b.ΔT_cool == a.ΔT_cool
        @test b.E_total ≈ b.E + b.LE_spray / _MT.latent_heat_vapor(299.15)
        # Night-time cool skin: 0.1-0.5 K, skin cooler than the water below.
        @test 0.05 < a.ΔT_cool < 0.5 && a.T_skin ≈ 301.15 - a.ΔT_cool
        # One seawater heat capacity throughout: rain heat and spray sensible
        # heat are both ∝ c_w, so the c_w passed to coare_fluxes must reach
        # them (it once reached only the cool skin).
        r40 = coare_fluxes(coare35_bulk(); args..., rain = 1e-3, spray = sp, c_w = 4000.0)
        r41 = coare_fluxes(coare35_bulk(); args..., rain = 1e-3, spray = sp, c_w = 4181.0)
        @test max(abs(r40.H_rain / r41.H_rain - 4000 / 4181), abs(r40.H_spray / r41.H_spray - 4000 / 4181)) < 1e-3 * 4000 / 4181
        # Supersaturated air (fog over the sea, or q_air a hair above q_sat)
        # has no κ-Köhler equilibrium; it is treated as saturation instead
        # of aborting the flux calculation: droplets then do not evaporate.
        qsat = _MT.saturation_specific_humidity(299.15, 101000.0)
        fog = coare_fluxes(coare35_bulk(); args..., q_air = 1.002qsat, spray = sp)
        @test fog.LE_spray <= 0 && fog.H_spray > 0
    end

    @testset "Coupling: WaterColumn with COARE fluxes, spray and rain" begin
        f = SurfaceForcing(sw_down = 0.0, lw_down = 400.0, T_air = 299.15,
                           q_air = _qair_rh(299.15, 101000.0, 0.8), U = 22.0, p = 101000.0,
                           z_ref = 10.0, rain = 2e-3, snow = 0.0)
        # Without cool skin the hook must give exactly the fluxes of
        # coare_fluxes at the column's skin temperature.
        col = WaterColumn(dz = fill(1.0, 30), T = 301.15, S = 35.0, cool_skin = nothing)
        hook = coare_flux_hook(coare35_bulk(); spray = sp)
        fl = hook(col, f)
        ref = coare_fluxes(coare35_bulk(); U = f.U, z_u = 10.0, T_air = f.T_air, q_air = f.q_air,
                           p = f.p, T_sea = 301.15, S = 35.0, lw_down = f.lw_down,
                           emissivity = surface_emissivity(col), rain = f.rain,
                           cool_skin = nothing, spray = sp)
        # (The hook's q_s uses the column's salinity factor, identical here.)
        @test fl.H ≈ ref.H_total rtol = 1e-9
        @test fl.LE ≈ ref.LE_total rtol = 1e-9
        @test fl.G ≈ fl.Rn - fl.H - fl.LE - ref.H_rain rtol = 1e-9
        # Heat conservation over a stormy day, with cool skin, spray and rain.
        col2 = WaterColumn(dz = fill(1.0, 30), T = 301.15, S = 35.0, bottom = :transmitting)
        H0 = column_heat_content(col2)
        budget = 0.0
        for k in 1:144
            sw = max(0.0, 800 * sinpi((k - 36) / 72))
            fk = SurfaceForcing(sw_down = sw, lw_down = 400.0, T_air = 299.15, q_air = f.q_air,
                                U = 22.0, p = 101000.0, z_ref = 10.0, rain = k < 20 ? 2e-3 : 0.0,
                                snow = 0.0)
            swnet = (1 - surface_albedo(col2)) * sw
            flk = step_surface!(col2, fk, 600.0; flux = hook)
            D = sum(col2.dz)
            budget += (flk.G - swnet * solar_transmission(col2.solar, D)) * 600
        end
        @test abs(column_heat_content(col2) - H0 - budget) < 1e-12 * column_heat_content(col2)
    end
end
