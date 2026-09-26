# Era V, Phase 52 — OceanMixedLayer (issue #38).
#
# Water column with a Kraus–Turner mixed layer, shortwave penetration,
# cool skin and diurnal warm layer. Validation targets:
#   * EOS-80 check values (Fofonoff & Millard 1983),
#   * the NOAA-PSL COARE 3.6 reference implementation, run on its own
#     test data set (ship observations), for the cool skin and the
#     Fairall warm layer (test/data/coare36_warm_cool_subset.txt),
#   * analytic limits: Kato–Phillips wind deepening, convective
#     encroachment, the Zeng–Beljaars steady state, exact energy
#     conservation of the column.

using Test, Printf
using Aeolus
using Aeolus.OceanMixedLayer
using Aeolus.SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, SurfaceForcing,
                              SurfaceFluxes, AbstractSurface, surface_temperature,
                              step_surface!, turbulent_fluxes, grav, cp_water
using Aeolus.MoistThermodynamics: ThermodynamicConstants

const OML = Aeolus.OceanMixedLayer

@testset "Era V — OceanMixedLayer (#38)" begin

    @testset "EOS-80: published check values, 4 °C maximum, derivatives" begin
        eos = UNESCOWaterEOS()
        # Fofonoff & Millard (1983), UNESCO Tech. Pap. Mar. Sci. 44, p = 0.
        checks = ((5.0, 0.0, 999.96675), (5.0, 35.0, 1027.67547), (25.0, 35.0, 1023.34306))
        worst = 0.0
        for (t, S, ρref) in checks
            worst = max(worst, abs(water_density(eos, t + 273.15, S) - ρref))
        end
        @test worst < 2e-5
        # Fresh water: density maximum at 3.98 °C (Tanaka et al. 2001: 3.98 °C).
        Ts = range(273.15, 283.15; length = 100_001)
        Tmax = Ts[argmax([water_density(eos, T, 0.0) for T in Ts])] - 273.15
        @test abs(Tmax - 3.98) < 0.01
        # α changes sign at the maximum; seawater α at 25 °C ≈ 2.97e-4 K⁻¹.
        @test water_thermal_expansion(eos, 273.15 + 2.0, 0.0) < 0
        @test water_thermal_expansion(eos, 273.15 + 6.0, 0.0) > 0
        @test abs(water_thermal_expansion(eos, 273.15 + Tmax, 0.0)) < 2e-7
        # Analytic derivatives against 4th-order finite differences.
        derr = 0.0
        for T in (275.0, 285.0, 300.0), S in (0.0, 10.0, 35.0)
            h = 1e-3
            ρ = water_density(eos, T, S)
            dT = (-water_density(eos, T + 2h, S) + 8water_density(eos, T + h, S) -
                  8water_density(eos, T - h, S) + water_density(eos, T - 2h, S)) / (12h)
            derr = max(derr, abs(water_thermal_expansion(eos, T, S) + dT / ρ))
            if S > 0
                dS = (-water_density(eos, T, S + 2h) + 8water_density(eos, T, S + h) -
                      8water_density(eos, T, S - h) + water_density(eos, T, S - 2h)) / (12h)
                derr = max(derr, abs(water_haline_contraction(eos, T, S) - dS / ρ))
            end
        end
        @test derr < 1e-10
        α25 = water_thermal_expansion(eos, 298.15, 35.0)
        β25 = water_haline_contraction(eos, 298.15, 35.0)
        @test 2.9e-4 < α25 < 3.05e-4
        @test 7.2e-4 < β25 < 7.6e-4
        lin = LinearWaterEOS(α = 2e-4)
        @test water_thermal_expansion(lin, lin.T0, lin.S0) == 2e-4
        @info "EOS-80" max_check_error = worst T_max_density_C = Tmax α_25C_35psu = α25 β_25C_35psu = β25
    end

    @testset "Shortwave penetration: Jerlov types, Soloviev bands" begin
        for ty in (:I, :IA, :IB, :II, :III)
            p = jerlov_water(ty)
            @test solar_transmission(p, 0.0) ≈ 1
            @test solar_transmission(p, 1e4) < 1e-12
            # absorbed in slabs + what is left = 1, exactly telescoping
            zs = [0.0, 0.3, 1.0, 2.5, 7.0, 20.0]
            tot = sum(solar_absorbed_between(p, zs[i], zs[i+1]) for i in 1:5) +
                  solar_transmission(p, zs[end])
            @test tot ≈ 1 atol = 1e-14
            # mean transmission = (1/D)∫ I dz, by quadrature
            D = 3.0
            n = 20_000
            quad = sum(solar_transmission(p, (i - 0.5) * D / n) for i in 1:n) / n
            @test solar_mean_transmission(p, D) ≈ quad rtol = 1e-7
        end
        # Clearer water lets more light through at 10 m: I > IA > IB > II > III.
        tr10 = [solar_transmission(jerlov_water(t), 10.0) for t in (:I, :IA, :IB, :II, :III)]
        @test issorted(tr10; rev = true)
        # Paulson & Simpson Type I at 10 m: 0.58 e^{-10/0.35} + 0.42 e^{-10/23}
        @test tr10[1] ≈ 0.42 * exp(-10 / 23) + 0.58 * exp(-10 / 0.35)
        c = soloviev_solar(:coare); zb = soloviev_solar(:zeng_beljaars)
        # COARE fxp for a 19 m layer (its maximum depth), from the code's formula.
        fxp19 = 1 - (0.28 * 0.014 * (1 - exp(-19 / 0.014)) + 0.27 * 0.357 * (1 - exp(-19 / 0.357)) +
                     0.45 * 12.82 * (1 - exp(-19 / 12.82))) / 19
        @test 1 - solar_mean_transmission(c, 19.0) ≈ fxp19 rtol = 1e-14
        # About half the net shortwave is absorbed in the top 3 m.
        @test 0.5 < solar_absorbed_between(zb, 0, 3) < 0.65
        @test_throws ArgumentError jerlov_water(:V)
        @test_throws ArgumentError SolarBands((0.5, 0.4), (1.0, 2.0))
        @info "Shortwave penetration" transmission_10m = round.(tr10; digits = 4) absorbed_top3m_ZB05 = solar_absorbed_between(zb, 0, 3)
    end
    # ---------------------------------------------------------------------
    # COARE 3.6 reference data (see the header of the data file)
    # ---------------------------------------------------------------------
    function read_coare_subset(path)
        lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(path))
        hdr = Symbol.(split(lines[1]))
        rows = [parse.(Float64, split(l)) for l in lines[2:end]]
        return Dict(h => [r[i] for r in rows] for (i, h) in enumerate(hdr))
    end
    coare = read_coare_subset(joinpath(@__DIR__, "data", "coare36_warm_cool_subset.txt"))
    ncoare = length(coare[:jd])
    # COARE's latitude-dependent gravity (coare36vn_zrf_et.grv).
    coare_grav(lat) = (x = sind(lat);
                       9.7803267715 * (1 + 0.0052790414x^2 + 2.32718e-5x^4 + 1.262e-7x^6 + 7e-10x^8))

    @testset "Cool skin: reproduces COARE 3.6 on its ship test data" begin
        cs = CoolSkin()
        errT = zeros(ncoare); errδ = zeros(ncoare)
        for i in 1:ncoare
            ts = coare[:tsg][i] + coare[:dT_warm_to_skin][i]      # °C, as COARE passes it
            S = coare[:Ss][i]
            # COARE 3.6 thermal expansion (salinity blend of two fits) and
            # constants ρ_w = 1022, c_w = 4000, k_w = 0.6, βS = 0.00075 S.
            Al35 = 2.1e-5 * (ts + 3.2)^0.79
            Al0 = (2.2 * (ts - 1)^0.82 - 5) * 1e-5
            α = Al0 + (Al35 - Al0) * S / 35
            u_w = coare[:usr][i] * sqrt(coare[:rhoa][i] / 1022)
            Q_out = -coare[:lw_net][i] + coare[:hsb][i] + coare[:hlb][i]
            r = cool_skin(cs, Q_out, coare[:sw_net][i], coare[:hlb][i], u_w;
                          α = α, βS = 0.00075 * S, ρ_w = 1022.0, c_w = 4000.0, k_w = 0.6,
                          g = coare_grav(coare[:lat][i]), L_v = coare[:Le][i])
            errT[i] = r.ΔT - coare[:dT_skin][i]
            errδ[i] = r.δ / coare[:dz_skin][i] - 1
        end
        # COARE iterates the skin jointly with the fluxes (the last iterate
        # uses the previous iteration's longwave) and uses 0.333 for 1/3;
        # the residual is those, not physics.
        @test maximum(abs, errT) < 2e-4
        @test maximum(abs, errδ) < 1e-3
        @test 0.17 < minimum(coare[:dT_skin]) && maximum(coare[:dT_skin]) < 0.4
        # Physics: a stronger heat loss cools the skin more; more wind thins it.
        base = (α = 3e-4, βS = 0.026, ρ_w = 1022.0, c_w = 4000.0, k_w = 0.6, g = 9.8, L_v = 2.44e6)
        r1 = cool_skin(cs, 150.0, 0.0, 100.0, 0.01; base...)
        r2 = cool_skin(cs, 250.0, 0.0, 100.0, 0.01; base...)
        r3 = cool_skin(cs, 150.0, 0.0, 100.0, 0.02; base...)
        @test r2.ΔT > r1.ΔT > 0
        @test r3.δ < r1.δ && r3.ΔT < r1.ΔT
        # Midday sun: the skin absorbs part of the shortwave and warms.
        @test cool_skin(cs, 150.0, 900.0, 100.0, 0.01; base...).ΔT < r1.ΔT
        # Non-convective skin (heating, B ≤ 0): δ = min(δ_max, 6ν/u*).
        rh = cool_skin(cs, -100.0, 0.0, 0.0, 0.002; base...)
        @test rh.δ ≈ min(0.01, 6e-6 / 0.002)
        @test rh.ΔT < 0
        @info "Cool skin vs COARE 3.6 (n = $ncoare)" max_abs_dT_err_K = maximum(abs, errT) rms_dT_err_K = sqrt(sum(abs2, errT) / ncoare) max_rel_delta_err = maximum(abs, errδ) dT_skin_range = extrema(coare[:dT_skin])
    end

    @testset "Log-linear stability: delegates the unstable branch" begin
        ll = BusingerDyerLogLinear()
        bd = Aeolus.SurfaceExchange.BusingerDyerBeljaarsHoltslag()
        for ζ in (-3.0, -0.5, -0.01)
            @test phi_h(ll, ζ) == phi_h(bd, ζ)
            @test psi_m(ll, ζ) == psi_m(bd, ζ)
        end
        @test phi_h(ll, 0.4) == 3.0 && psi_h(ll, 0.4) == -2.0
        # φ–ψ consistency on the stable side: dψ/dζ = (1 - φ)/ζ
        @test (psi_h(ll, 0.41) - psi_h(ll, 0.39)) / 0.02 ≈ (1 - phi_h(ll, 0.4)) / 0.4
    end

    @testset "Fairall warm layer: reproduces COARE 3.6, analytic growth" begin
        # The COARE 3.6 time loop (coare36vnWarm_et): local time from
        # longitude, daily reset when local time decreases, fluxes from
        # the previous record, longwave from this record's T_sea and the
        # previous cool skin.
        wl = FairallWarmLayer()
        dTw = zeros(ncoare); dTs = zeros(ncoare)
        jtime = 0.0; started = false
        for i in 1:ncoare
            jd, lon = coare[:jd][i], coare[:lon][i]
            chk = (lon + 7.5) / 15 + (jd - trunc(jd)) * 24
            chk > 24 && (chk -= 24)
            newtime = (chk - 24 * trunc(chk / 24)) * 3600
            if i > 1 && (newtime <= 21600 || started)
                started = true
                if newtime < jtime
                    reset_warm_layer!(wl)
                else
                    tsg = coare[:tsg][i]
                    lw_up = 0.97 * (5.67e-8 * (tsg - coare[:dT_skin][i-1] + 273.16)^4 - coare[:lw_dn][i])
                    q_out = lw_up + coare[:hsb][i-1] + coare[:hlb][i-1] + coare[:hrain][i-1]
                    step_warm_layer!(wl, coare[:sw_net][i], q_out, coare[:tau][i-1], newtime - jtime;
                                     α = 2.1e-5 * (tsg + 3.2)^0.79, ρ_w = 1022.0, c_w = 4000.0,
                                     g = coare_grav(coare[:lat][i]))
                end
                dTs[i] = warm_layer_warming(wl, coare[:ztsg][i])
            end
            dTw[i] = warm_layer_warming(wl)
            jtime = newtime
        end
        errw = maximum(abs, dTw .- coare[:dT_warm])
        errs = maximum(abs, dTs .- coare[:dT_warm_to_skin])
        @test errw < 5e-6            # data rounded to 7 significant digits
        @test errs < 5e-6
        @test maximum(coare[:dT_warm]) > 0.4   # a real warm layer is exercised
        @info "Fairall warm layer vs COARE 3.6 (n = $ncoare)" max_abs_err_dT_warm = errw max_abs_err_dT_to_5m = errs max_dT_warm = maximum(coare[:dT_warm])

        # Analytic growth: no shortwave, constant heating q and stress τ, so
        # Q = q t, I = τ t and ΔT = ctd2 q^{3/2} t^{1/2} / τ, D = ctd1 τ √t/√q.
        w = FairallWarmLayer(q_onset = 0.0)
        q, τ, α, ρ, c, g = 100.0, 0.01, 3e-4, 1025.0, 4000.0, 9.81
        for _ in 1:60
            step_warm_layer!(w, 0.0, -q, τ, 60.0; α = α, ρ_w = ρ, c_w = c, g = g)
        end
        t = 3600.0
        ctd1 = sqrt(2 * 0.65 * c / (α * g * ρ)); ctd2 = sqrt(2α * g / (0.65ρ)) / c^1.5
        @test warm_layer_warming(w) ≈ ctd2 * q^1.5 * sqrt(t) / τ rtol = 1e-12
        @test warm_layer_depth(w) ≈ ctd1 * τ * sqrt(t / q) rtol = 1e-12
        # The linear profile holds exactly the accumulated heat.
        @test ρ * c * warm_layer_depth(w) * warm_layer_warming(w) / 2 ≈ q * t rtol = 1e-12
        # Critical Richardson number of the layer: g α ΔT D / Δu² = Ri_c with
        # Δu = 2 I / (ρ D) (linear current profile).
        Δu = 2 * τ * t / (ρ * warm_layer_depth(w))
        @test g * α * warm_layer_warming(w) * warm_layer_depth(w) / Δu^2 ≈ 0.65 rtol = 1e-12
        reset_warm_layer!(w)
        @test warm_layer_warming(w) == 0 && warm_layer_depth(w) == w.D_max
    end

    @testset "Zeng–Beljaars warm layer: steady state, neutral limit" begin
        kw = (α = 3e-4, ρ_w = 1025.0, c_w = 4000.0, g = 9.81, κ = 0.4)
        sw, qo, τ = 600.0, 200.0, 0.02
        u = sqrt(τ / kw.ρ_w)
        # Neutral limit (φ_t ≡ 1 via β = 0): ΔT_ss = Q_d / (ρ c ν κ u*).
        zn = ZengBeljaarsWarmLayer(stability = BusingerDyerLogLinear(β = 0.0))
        for _ in 1:2000
            step_warm_layer!(zn, sw, qo, τ, 120.0; kw...)
        end
        Qd = -qo + sw * (1 - solar_transmission(zn.solar, zn.d))
        ΔT_neutral = Qd / (kw.ρ_w * kw.c_w * zn.ν * kw.κ * u)
        @test warm_layer_warming(zn) ≈ ΔT_neutral rtol = 1e-10
        # Stable φ_t: the steady state solves ΔT·damping(ΔT) = source; find it
        # by bisection and compare with the time-marched solution.
        z = ZengBeljaarsWarmLayer()
        # The lagged-φ implicit step converges at half the damping rate
        # (damping ∝ ΔT^{-1/2} here): ~8 h e-folding, so march 330 h.
        for _ in 1:20_000
            step_warm_layer!(z, sw, qo, τ, 60.0; kw...)
        end
        f(x) = (AB = OML.zeng_beljaars_tendency(z, x, sw, qo, u; kw...); x * AB[2] - AB[1])
        lo, hi = 1e-9, 20.0
        for _ in 1:200
            mid = (lo + hi) / 2
            f(mid) > 0 ? (hi = mid) : (lo = mid)
        end
        @test warm_layer_warming(z) ≈ lo rtol = 1e-8
        @test warm_layer_warming(z) > ΔT_neutral        # stability suppresses mixing
        # More wind, less warming; net cooling erases the layer.
        z2 = ZengBeljaarsWarmLayer()
        for _ in 1:5000
            step_warm_layer!(z2, sw, qo, 4τ, 60.0; kw...)
        end
        @test warm_layer_warming(z2) < warm_layer_warming(z)
        ΔT_4x = warm_layer_warming(z2)
        for _ in 1:2000
            step_warm_layer!(z2, 0.0, 150.0, 4τ, 60.0; kw...)
        end
        @test warm_layer_warming(z2) == 0
        # Profile: warming between the surface and depth z is ΔT (z/d)^ν.
        @test warm_layer_warming(z, 3.0) == warm_layer_warming(z)
        @test warm_layer_warming(z, 0.3) ≈ warm_layer_warming(z) * 0.1^0.3
        @info "Zeng–Beljaars steady state" ΔT_neutral ΔT_stable = warm_layer_warming(z) ΔT_4x_stress = ΔT_4x
    end

    @testset "Warm layers against the ship's near-surface thermometers" begin
        # Observed T(0.05 m) - T(5.3 m) from the sea snake and the TSG, and
        # the same difference from each scheme, driven by the COARE fluxes.
        # Reported, not tuned: the observed signal is small (≤ 0.14 K)
        # because the trade winds kept mixing the layer.
        ld = floor.(Int, coare[:jd] .+ ((coare[:lon] .+ 7.5) ./ 15) ./ 24)
        obs = coare[:tsnk] .- coare[:tsg]
        night = coare[:sw_net] .== 0
        offset = sort(obs[night])[cld(count(night), 2)]    # sensor offset: night median
        function drive(wl)
            out = zeros(ncoare); jtime = 0.0; started = false
            for i in 1:ncoare
                jd, lon = coare[:jd][i], coare[:lon][i]
                chk = (lon + 7.5) / 15 + (jd - trunc(jd)) * 24
                chk > 24 && (chk -= 24)
                newtime = (chk - 24 * trunc(chk / 24)) * 3600
                if i > 1 && (newtime <= 21600 || started)
                    started = true
                    if newtime < jtime
                        wl isa FairallWarmLayer && reset_warm_layer!(wl)
                    else
                        tsg = coare[:tsg][i]
                        lw_up = 0.97 * (5.67e-8 * (tsg - coare[:dT_skin][i-1] + 273.16)^4 - coare[:lw_dn][i])
                        q_out = lw_up + coare[:hsb][i-1] + coare[:hlb][i-1] + coare[:hrain][i-1]
                        step_warm_layer!(wl, coare[:sw_net][i], q_out, coare[:tau][i-1], newtime - jtime;
                                         α = 2.1e-5 * (tsg + 3.2)^0.79, ρ_w = 1022.0, c_w = 4000.0,
                                         g = coare_grav(coare[:lat][i]))
                    end
                end
                out[i] = warm_layer_warming(wl, coare[:ztsg][i]) - warm_layer_warming(wl, 0.05)
                jtime = newtime
            end
            out
        end
        F = drive(FairallWarmLayer()); Z = drive(ZengBeljaarsWarmLayer())
        days = 19:25
        mo = [maximum(obs[ld .== k]) - offset for k in days]
        mF = [maximum(F[ld .== k]) for k in days]
        mZ = [maximum(Z[ld .== k]) for k in days]
        τday = [sum(coare[:tau][ld .== k]) / count(ld .== k) for k in days]
        # The calmest day (25) has the largest observed warming, and both
        # schemes put their largest warming on it too.
        @test argmin(τday) == argmax(mo) == argmax(mF) == argmax(mZ)
        day = .!night
        rmsF = sqrt(sum(abs2, F[day] .- (obs[day] .- offset)) / count(day))
        rmsZ = sqrt(sum(abs2, Z[day] .- (obs[day] .- offset)) / count(day))
        @info "Warm layer vs ship observations, T(0.05 m) - T(5.3 m), days 19-25" sensor_offset_K = offset daily_max_obs = round.(mo; digits = 3) daily_max_Fairall = round.(mF; digits = 3) daily_max_ZengBeljaars = round.(mZ; digits = 3) daytime_rms_Fairall = rmsF daytime_rms_ZB = rmsZ
    end
end
