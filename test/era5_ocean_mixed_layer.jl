# Era V, Phase 52 — OceanMixedLayer (issue #38).
#
# Water column with a Kraus–Turner mixed layer, shortwave penetration,
# cool skin and diurnal warm layer. Validation targets:
#   * EOS-80 check values (Fofonoff & Millard 1983),
#   * the NOAA-PSL COARE 3.6 reference implementation, run on its own
#     test data set (ship observations), for the cool skin and the
#     Fairall warm layer (test/data/coare36_warm_cool_subset.txt)
#     (the comparison of both warm layers with the ship's near-surface
#     thermometers is a validation run: validation/era5_ocean_mixed_layer.jl),
#   * analytic limits: Kato–Phillips wind deepening, convective
#     encroachment, the Zeng–Beljaars steady state, exact energy
#     conservation of the column.

using Test
using Aeolus
using Aeolus.OceanMixedLayer
using Aeolus.SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, SurfaceForcing,
                              SurfaceFluxes, AbstractSurface, surface_temperature,
                              step_surface!, turbulent_fluxes, grav, cp_water
using Aeolus.MoistThermodynamics: ThermodynamicConstants

const OML = Aeolus.OceanMixedLayer

isdefined(@__MODULE__, :read_coare_subset) || include(joinpath(@__DIR__, "support", "coare36_subset.jl"))

@testset "Era V — OceanMixedLayer (#38)" begin

    @testset "EOS-80: published check values, 4 °C maximum, derivatives" begin
        eos = UNESCOWaterEOS()
        # Fofonoff & Millard (1983), UNESCO Tech. Pap. Mar. Sci. 44, p = 0
        checks = ((5.0, 0.0, 999.96675), (5.0, 35.0, 1027.67547), (25.0, 35.0, 1023.34306))
        @test maximum(((t, S, ρref),) -> abs(water_density(eos, t + 273.15, S) - ρref), checks) < 2e-5
        # fresh water: α changes sign at the 3.98 °C density maximum
        @test abs(water_thermal_expansion(eos, 273.15 + 3.98, 0.0)) < 2e-7
        # analytic derivatives against 4th-order finite differences
        derr = 0.0
        for T in (275.0, 300.0), S in (10.0, 35.0)
            h = 1e-3
            ρ = water_density(eos, T, S)
            fd(f) = (-f(2h) + 8f(h) - 8f(-h) + f(-2h)) / (12h)
            derr = max(derr, abs(water_thermal_expansion(eos, T, S) + fd(d -> water_density(eos, T + d, S)) / ρ),
                       abs(water_haline_contraction(eos, T, S) - fd(d -> water_density(eos, T, S + d)) / ρ))
        end
        @test derr < 1e-10
    end

    @testset "Shortwave penetration: Jerlov types, Soloviev bands" begin
        p = jerlov_water(:IB)
        # absorbed in slabs + what is left = 1, exactly telescoping
        zs = [0.0, 0.3, 1.0, 2.5, 7.0, 20.0]
        @test sum(solar_absorbed_between(p, zs[i], zs[i+1]) for i in 1:5) +
              solar_transmission(p, zs[end]) ≈ 1 atol = 1e-14
        # mean transmission = (1/D)∫ I dz, by quadrature
        n = 20_000
        @test solar_mean_transmission(p, 3.0) ≈ sum(solar_transmission(p, (i - 0.5) * 3.0 / n) for i in 1:n) / n rtol = 1e-7
        # Paulson & Simpson Type I at 10 m: 0.58 e^{-10/0.35} + 0.42 e^{-10/23}
        @test solar_transmission(jerlov_water(:I), 10.0) ≈ 0.42 * exp(-10 / 23) + 0.58 * exp(-10 / 0.35)
        # COARE fxp for a 19 m layer, from the reference code's formula
        fxp19 = 1 - (0.28 * 0.014 * (1 - exp(-19 / 0.014)) + 0.27 * 0.357 * (1 - exp(-19 / 0.357)) +
                     0.45 * 12.82 * (1 - exp(-19 / 12.82))) / 19
        @test 1 - solar_mean_transmission(soloviev_solar(:coare), 19.0) ≈ fxp19 rtol = 1e-14
    end
    # ---------------------------------------------------------------------
    # COARE 3.6 reference data (see the header of the data file)
    # ---------------------------------------------------------------------
    coare = coare36_warm_cool_subset()
    ncoare = length(coare[:jd])

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
    end

    @testset "Log-linear stability: delegates the unstable branch" begin
        ll = BusingerDyerLogLinear()
        bd = Aeolus.SurfaceExchange.BusingerDyerBeljaarsHoltslag()
        @test all(ζ -> phi_h(ll, ζ) == phi_h(bd, ζ) && psi_m(ll, ζ) == psi_m(bd, ζ), (-3.0, -0.5, -0.01))
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
    end

    # ---------------------------------------------------------------------
    # The water column
    # ---------------------------------------------------------------------
    g0 = grav(DEFAULT_SURFACE_CONSTANTS)

    @testset "Kato–Phillips: wind deepening into linear stratification" begin
        # Mixing a linearly stratified column (N²) to depth h costs
        # ρ₀ N² h³/12; with work ρ₀ m u*³ t that gives h = (12 m u*³ t/N²)^{1/3}.
        eos = LinearWaterEOS(ρ0 = 1000.0, α = 2e-4)
        N2, u, m = 1e-4, 0.01, 0.5
        Γ = N2 / (g0 * eos.α)                     # K per metre of depth
        for dz in (0.2,)
            col = WaterColumn(dz = fill(dz, round(Int, 40 / dz)), T = z -> 290.0 - Γ * z,
                              eos = eos, mixing = KrausTurnerMixing(m = m, n = 0.0),
                              ρ_ref = 1000.0)
            PE0 = column_potential_energy(col); H0 = column_heat_content(col)
            dt, nstep = 60.0, 1440
            used = 0.0
            for _ in 1:nstep
                used += mix_column!(col, u, dt).used
            end
            t = dt * nstep
            h_exact = (12m * u^3 * t / N2)^(1 / 3)
            relerr = mixed_layer_depth(col) / h_exact - 1
            @test abs(relerr) < 0.02
            # every joule of wind work went into potential energy
            @test used ≈ 1000.0 * m * u^3 * t rtol = 1e-9
            @test column_potential_energy(col) - PE0 ≈ used rtol = 1e-6
            @test column_heat_content(col) ≈ H0 rtol = 1e-14
        end
    end

    @testset "Free convection: encroachment and penetrative entrainment" begin
        # Surface buoyancy loss B₀ into N²: h² = 2(1 + 2n) B₀ t / N²
        # (n = 0: encroachment; n = 0.2: Deardorff's entrainment ratio).
        eos = LinearWaterEOS(ρ0 = 1000.0, α = 2e-4)
        N2, Q = 1e-4, 200.0
        Γ = N2 / (g0 * eos.α)
        for n in (0.2,)
            col = WaterColumn(dz = fill(0.1, 400), T = z -> 290.0 - Γ * z, eos = eos,
                              mixing = KrausTurnerMixing(m = 0.0, n = n), ρ_ref = 1000.0)
            B0 = g0 * eos.α * Q / (col.ρ_ref * col.c_p)
            H0 = column_heat_content(col)
            dt, nstep = 600.0, 288
            for _ in 1:nstep
                absorb_heat!(col, -Q, 0.0, dt)
                mix_column!(col, 0.0, dt)
            end
            t = dt * nstep
            h_exact = sqrt(2 * (1 + 2n) * B0 * t / N2)
            relerr = mixed_layer_depth(col) / h_exact - 1
            @test abs(relerr) < 0.02
            # exact to the round-off of the stored heat (5e10 J m⁻² over
            # 400 layers and ~10⁵ mixing operations)
            @test abs(column_heat_content(col) - H0 + Q * t) < 1e-12 * H0
        end
    end

    @testset "Lake overturn at the 4 °C density maximum" begin
        # Autumn cooling of a 20 m fresh-water lake at 10 °C, no wind: the
        # whole lake overturns down to the temperature of maximum density,
        # after which cooling only chills a light surface layer (inverse
        # stratification) — the physics that lets lakes freeze from the top.
        col = WaterColumn(dz = fill(1.0, 20), T = 283.15, S = 0.0,
                          mixing = KrausTurnerMixing(m = 0.0, n = 0.0))
        H0 = column_heat_content(col)
        dt, Q, t = 3600.0, 300.0, 0.0
        while col.T[1] > 273.15 + 1.0
            absorb_heat!(col, -Q, 0.0, dt)
            mix_column!(col, 0.0, dt)
            t += dt
        end
        Tmd = 273.15 + 3.9817                    # EOS-80 maximum (first testset)
        @test abs(col.T[end] - Tmd) < 0.02
        @test col.T[1] < col.T[end]              # inverse stratification
        ρs = [water_density(col.eos, T, 0.0) for T in col.T]
        @test issorted(ρs)                       # statically stable
        @test abs(column_heat_content(col) - H0 + Q * t) < 1e-13 * H0
    end

    @testset "WaterColumn as an AbstractSurface" begin
        col = WaterColumn(T = 293.15, S = 35.0, cool_skin = nothing)
        @test col isa AbstractSurface
        @test salinity_humidity_factor(34.0) ≈ 0.98
        f = SurfaceForcing(sw_down = 0.0, lw_down = 350.0, T_air = 290.15, q_air = 0.008, U = 6.0)
        fsea = turbulent_fluxes(col, f)
        lake = WaterColumn(T = 293.15, S = 0.0, cool_skin = nothing)
        flake = turbulent_fluxes(lake, f)
        @test fsea.H > 0 && fsea.LE > 0          # warm water, cooler drier air
        @test flake.LE > fsea.LE                 # salt lowers the vapour pressure
        @test fsea.G ≈ fsea.Rn - fsea.H - fsea.LE
        # emissivity from SurfaceOptics — one source of truth
        @test Aeolus.SurfaceExchange.surface_emissivity(col) ==
              Aeolus.SurfaceOptics.material_emissivity(:water)
        # A cool skin makes the skin cooler than the top layer at night.
        cs = WaterColumn(T = 293.15, S = 35.0)
        step_surface!(cs, f, 600.0)
        @test 0.05 < cs.ΔT_cool < 0.6
        @test surface_temperature(cs) < cs.T[1]
    end

    # A clear-sky diurnal forcing for the coupled tests.
    function diurnal_forcing(t; U = 5.0, T_air = 298.15)
        hour = mod(t, 86400.0) / 3600
        sw = max(0.0, 950.0 * sin(π * (hour - 6) / 12))
        SurfaceForcing(sw_down = sw, lw_down = 400.0, T_air = T_air, q_air = 0.016, U = U,
                       p = 101325.0, z_ref = 10.0)
    end

    @testset "Coupled column: exact energy conservation" begin
        # Every configuration: the heat content changes by ∫(G - SW_bottom - F_bottom) dt.
        configs = (
            (name = "sea + Fairall, leaky bottom", kw = (dz = fill(1.0, 10), T = 298.15, S = 35.0,
                                                   warm_layer = FairallWarmLayer(),
                                                   bottom = :transmitting, bottom_flux = 5.0,
                                                   solar = jerlov_water(:I))),
        )
        for cfg in configs
            col = WaterColumn(; cfg.kw...)
            H0 = column_heat_content(col)
            D = sum(col.dz)
            budget = 0.0; scale = 0.0
            dt = 900.0
            for i in 0:95
                f = diurnal_forcing(i * dt)
                fl = step_surface!(col, f, dt)
                sw_net = (1 - col.albedo) * f.sw_down
                out = col.bottom === :transmitting ? sw_net * solar_transmission(col.solar, D) : 0.0
                budget += (fl.G - out - col.bottom_flux) * dt
                scale += abs(fl.G) * dt
            end
            # Round-off: the column stores ~4e10 J m⁻², so Float64 resolves
            # ~1e-5 J m⁻² per step; compare against that, and quote the
            # error relative to the heat that actually crossed the surface.
            abserr = abs(column_heat_content(col) - H0 - budget)
            @test abserr < 1e-13 * H0
            @test abserr / scale < 1e-10
        end
    end

    @testset "column_heat_content is one contract generic" begin
        # UrbanSurface (#26) and this module both export column_heat_content;
        # as two separate functions the name was ambiguous in `Aeolus`.
        chc = Aeolus.SurfaceExchange.column_heat_content
        @test Aeolus.column_heat_content === chc
        @test Aeolus.UrbanSurface.column_heat_content === chc
        @test Aeolus.OceanMixedLayer.column_heat_content === chc
    end
end
