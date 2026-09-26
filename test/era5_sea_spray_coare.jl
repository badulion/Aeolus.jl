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
        # (d) Wind-speed crossovers (Fig. 9 text): spray sensible passes the
        # interfacial sensible flux at U10 ≈ 19 m/s, spray latent the
        # interfacial latent at ≈ 26 m/s. Interfacial fluxes here: COARE 3.5
        # (Andreas et al. use their own interfacial algorithm); H_s scaled as
        # U² from the calibration above.
        qa = _qair_rh(Ta, p, 0.9)
        cross = Dict{Symbol,Float64}()
        prev = nothing
        for U in 10.0:0.25:35.0
            b = coare_fluxes(coare35_bulk(); U = U, z_u = 10.0, T_air = Ta, q_air = qa, p = p,
                             T_sea = Tsea, S = 34.0, lw_down = 0.0, cool_skin = nothing)
            s = flux(U = U, Hs = Hs_cal * (U / 25)^2)
            d = (S = s.H_S - b.H, L = s.H_L - b.LE)
            if prev !== nothing
                for k in (:S, :L)
                    (getfield(prev, k) < 0 <= getfield(d, k)) && !haskey(cross, k) && (cross[k] = U)
                end
            end
            prev = d
        end
        @test abs(cross[:S] - 19) <= 2.5
        @test abs(cross[:L] - 26) <= 2.5
        # (e) "Both spray fluxes become significant [≥ 10 % of the
        # interfacial ones] in the U_N10 region 10–12 m/s".
        b10 = coare_fluxes(coare35_bulk(); U = 8.0, z_u = 10.0, T_air = Ta, q_air = qa, p = p,
                           T_sea = Tsea, S = 34.0, lw_down = 0.0, cool_skin = nothing)
        s8 = flux(U = 8.0, Hs = Hs_cal * (8 / 25)^2)
        @test s8.H_S < 0.1b10.H && s8.H_L < 0.1b10.LE
        b15 = coare_fluxes(coare35_bulk(); U = 15.0, z_u = 10.0, T_air = Ta, q_air = qa, p = p,
                           T_sea = Tsea, S = 34.0, lw_down = 0.0, cool_skin = nothing)
        s15 = flux(U = 15.0, Hs = Hs_cal * (15 / 25)^2)
        @test s15.H_S > 0.1b15.H && s15.H_L > 0.1b15.LE
        # Energy bookkeeping of the spray: E = H_L / L_v.
        s = flux()
        @test s.E ≈ s.H_L / _MT.latent_heat_vapor(Ta)
        @info "Spray vs Andreas et al. (2015)" H_S_max_rel_err = maximum(abs, errS) H_L_max_abs_err_Wm2 = maximum(abs, errL) H_s_calibrated_m = Hs_cal H_s_fully_developed_m = fully_developed_sea_state(25.0).H_s H_L_90pct_fully_developed = flux(Hs = fully_developed_sea_state(25.0).H_s).H_L sign_change_RH = RH0 crossover_sensible = cross[:S] crossover_latent = cross[:L]
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

    # ---- COARE 3.6 reference: full algorithm ----------------------------
    o = _read_cols(joinpath(_SPRAY_DATA, "coare36_warm_cool_subset.txt"))
    m = _read_cols(joinpath(_SPRAY_DATA, "coare36_met_subset.txt"))
    n = length(o[:jd])

    @testset "Full COARE (bulk + cool skin + warm layer) vs COARE 3.6 reference" begin
        @test length(m[:jd]) == n && maximum(abs.(o[:jd] .- m[:jd])) < 1e-5
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
        @test e_us < 5e-3
        @test e_τ < 1e-2
        @test e_H < 1.5                          # W m⁻² (H ≈ 3-40 W m⁻²)
        @test e_LE < 3.0 && rms_LE < 2.0         # W m⁻² (LE ≈ 90-250 W m⁻²)
        @test e_cool < 0.03                      # K (ΔT_cool 0.18-0.37 K)
        @test e_warm < 0.01                      # K (warming to 5.3 m up to 0.33 K)
        @test maximum(o[:dT_warm_to_skin]) > 0.3 # a real warm layer is exercised
        @test all(s -> s.iterations < 20, sols)
        # Rain heat flux: q_sat in the wet-bulb factor (here) vs the
        # reference's actual q: ours is smaller by the α ratio.
        wet = o[:hrain] .> 0.5
        ratio = g(:H_rain)[wet] ./ o[:hrain][wet]
        @test all(0.7 .< ratio .< 1.0)
        @info "COARE 3.6 reference, n = $n" max_rel_err_ustar = e_us max_rel_err_tau = e_τ max_abs_err_H = e_H max_abs_err_LE = e_LE rms_err_LE = rms_LE bias_LE = sum(g(:LE) .- o[:hlb]) / n max_abs_err_dT_cool = e_cool max_abs_err_dT_warm_to_sensor = e_warm hrain_ratio = extrema(ratio)
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
        @test b.H_spray > 0 && b.LE_spray > 0
        @test b.H_total == b.H + b.H_spray && b.LE_total == b.LE + b.LE_spray
        @test b.E_total ≈ b.E + b.LE_spray / _MT.latent_heat_vapor(299.15)
        # Bookkeeping (definition, not validation):
        @test b.Q_ocean ≈ b.lw_net - b.H_total - b.LE_total - b.H_rain
        # Night-time cool skin: 0.1-0.5 K, skin cooler than the water below.
        @test 0.05 < a.ΔT_cool < 0.5 && a.T_skin ≈ 301.15 - a.ΔT_cool
        # The fixed point is converged: another cool-skin pass changes nothing.
        c = coare_fluxes(coare35_bulk(); args..., tol = 1e-10)
        @test abs(c.ΔT_cool - a.ΔT_cool) < 1e-5
        # Switching the cool skin off gives the bulk-SST fluxes (larger).
        d = coare_fluxes(coare35_bulk(); args..., cool_skin = nothing)
        @test d.ΔT_cool == 0 && d.LE > a.LE && d.H > a.H
        @test_throws ArgumentError spray_dry_radius(50e-6, 0.0)
    end

end
