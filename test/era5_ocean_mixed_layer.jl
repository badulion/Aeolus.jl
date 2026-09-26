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
end
