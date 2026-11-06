# Surface shared core — SurfaceExchange (issue #92).
#
# Material constants, Monin–Obukhov similarity and the surface interface
# that every surface model builds on.

using Test
using Aeolus
using Aeolus.SurfaceExchange
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, PhasePartition, air_density,
                                  heat_capacity_moist, saturation_specific_humidity,
                                  latent_heat_vapor

const SE = Aeolus.SurfaceExchange

@testset "SurfaceExchange contract (#92)" begin

    @testset "Material constants delegate to the shared cores" begin
        c = DEFAULT_SURFACE_CONSTANTS
        # one source of truth: the very same numbers
        @test LH_f0(c) === DEFAULT_CONSTANTS.LH_s0 - DEFAULT_CONSTANTS.LH_v0
        @test grav(c) === DEFAULT_CONSTANTS.grav && cp_ice(c) === DEFAULT_CONSTANTS.cp_ice
        @test c.σ_SB === Aeolus.RadiationCore.STEFAN_BOLTZMANN
        c2 = SurfaceConstants(thermo = Aeolus.MoistThermodynamics.ThermodynamicConstants(grav = 9.81))
        @test grav(c2) == 9.81
    end

    @testset "MOST stability functions: φ–ψ consistency and published values" begin
        sf = BusingerDyerBeljaarsHoltslag()
        # dψ/dζ = (1 − φ)/ζ by a 4th-order central difference, both regimes
        worst = 0.0
        for ζ in (-5.0, -0.3, 0.05, 1.0, 10.0), (ψ, φ) in ((psi_m, phi_m), (psi_h, phi_h))
            h = 1e-3 * max(abs(ζ), 0.1)
            dψ = (-ψ(sf, ζ + 2h) + 8ψ(sf, ζ + h) - 8ψ(sf, ζ - h) + ψ(sf, ζ - 2h)) / (12h)
            worst = max(worst, abs(dψ - (1 - φ(sf, ζ)) / ζ))
        end
        @test worst < 1e-8
        # exact at ζ = −1/2 (x = √3): Paulson (1970)
        @test psi_h(sf, -0.5) ≈ 2log(2) rtol = 1e-14
        @test psi_m(sf, -0.5) ≈ 2log((1 + √3) / 2) + log(2) - 2π / 3 + π / 2 rtol = 1e-14
        # Beljaars & Holtslag (1991), Eqs. 28 and 32, at ζ = 1
        @test psi_m(sf, 1.0) ≈ -4.2823 atol = 1e-4
        @test psi_h(sf, 1.0) ≈ -4.4339 atol = 1e-4
        # log-linear family with separate β (GABLS1: β_m = 4.8, β_h = 7.8)
        g1 = BusingerDyerLogLinear(β_m = 4.8, β_h = 7.8)
        @test phi_m(g1, 0.1) ≈ 1.48 && phi_h(g1, 0.1) ≈ 1.78
        @test psi_m(g1, 0.5) ≈ -2.4 && psi_h(g1, 0.5) ≈ -3.9
    end

    @testset "Surface-layer solver: neutral limit and self-consistency" begin
        κ = von_karman()
        s = solve_surface_layer(; U = 5.0, z_m = 10.0, θ_a = 290.0, θ_s = 290.0, z0m = 0.01, z0h = 0.001)
        @test s.C_D ≈ (κ / log(10 / 0.01))^2 rtol = 4eps()
        @test s.C_H ≈ κ^2 / (log(10 / 0.01) * log(10 / 0.001)) rtol = 4eps()
        sf = BusingerDyerBeljaarsHoltslag()
        ε = SE._virtual_coefficient(DEFAULT_SURFACE_CONSTANTS)
        worst_prof, worst_L, allconv = 0.0, 0.0, true
        for (θs, qs, U) in ((305.0, 0.015, 1.0), (284.0, 0.005, 3.0))
            θa, qa, zm, zh, z0m, z0h, z0q = 290.0, 0.008, 10.0, 2.0, 0.1, 0.01, 0.005
            s = solve_surface_layer(sf; U = U, z_m = zm, z_h = zh, θ_a = θa, θ_s = θs,
                                    q_a = qa, q_s = qs, z0m = z0m, z0h = z0h, z0q = z0q)
            allconv &= s.converged
            L = s.L
            # flux-profile relations at the returned L reproduce the input differences
            Φm = log(zm / z0m) - psi_m(sf, zm / L) + psi_m(sf, z0m / L)
            Φh = log(zh / z0h) - psi_h(sf, zh / L) + psi_h(sf, z0h / L)
            Φq = log(zh / z0q) - psi_h(sf, zh / L) + psi_h(sf, z0q / L)
            for (got, want) in ((s.u_star / κ * Φm, U), (s.θ_star / κ * Φh, θa - θs),
                                (s.q_star / κ * Φq, qa - qs))
                worst_prof = max(worst_prof, abs(got - want) / abs(want))
            end
            # ...and L is the Obukhov length of those fluxes
            θv_star = s.θ_star * (1 + ε * qa) + ε * θa * s.q_star
            worst_L = max(worst_L, abs(obukhov_length(s.u_star, θv_star, θa * (1 + ε * qa)) / L - 1))
        end
        @test allconv
        @test worst_prof < 1e-12
        @test worst_L < 1e-10
        # beyond ζ_max the solver clamps and says so rather than returning NaN
        cl = solve_surface_layer(sf; U = 0.5, z_m = 10.0, θ_a = 290.0, θ_s = 275.0,
                                 z0m = 0.1, z0h = 0.01)
        @test !cl.converged && cl.ζ == 1000 && isfinite(cl.C_H)
    end

    # A minimal external implementer: a wet, grey, fixed-temperature surface.
    struct _WetPatch <: AbstractSurface
        T::Float64
        β::Float64
    end
    SE.surface_temperature(s::_WetPatch) = s.T
    SE.roughness_lengths(::_WetPatch) = (z0m = 1e-3, z0h = 1e-4, z0q = 1e-4)
    SE.surface_albedo(::_WetPatch) = 0.06
    SE.surface_emissivity(::_WetPatch) = 0.97
    SE.moisture_availability(s::_WetPatch) = s.β

    @testset "Surface interface: default turbulent_fluxes are the bulk formulae" begin
        @test Aeolus.Radiation.surface_temperature === SE.surface_temperature
        f = SurfaceForcing(sw_down = 500.0, lw_down = 330.0, T_air = 293.0,
                           q_air = 0.009, U = 6.0, p = 1.0e5, z_ref = 10.0)
        θa = 293.0 + Aeolus.MoistThermodynamics.dry_adiabatic_lapse_rate(
                 PhasePartition(0.009), DEFAULT_CONSTANTS) * 10
        ρ = air_density(293.0, 1.0e5, PhasePartition(0.009))
        cp = heat_capacity_moist(PhasePartition(0.009))
        fl = turbulent_fluxes(SlabLand(; T = 303.0), f)
        sol = solve_surface_layer(; U = 6.0, z_m = 10.0, θ_a = θa, θ_s = 303.0,
                                  q_a = 0.009, q_s = 0.009, z0m = 0.1, z0h = 0.01)
        @test fl.E == 0                                   # dry slab: q_s = q_a
        @test fl.H ≈ ρ * cp * sol.C_H * sol.U * (303.0 - θa) rtol = 1e-12
        @test fl.Rn ≈ 0.8 * 500 + 330 - SE.STEFAN_BOLTZMANN * 303.0^4 rtol = 1e-14
        # wet surface: E = β ρ C_E U (q_sat(T_s) − q_a), LE = L_v(T_s) E
        w = _WetPatch(298.0, 0.3)
        fw = turbulent_fluxes(w, f)
        qsat = saturation_specific_humidity(298.0, 1.0e5)
        sw = solve_surface_layer(; U = 6.0, z_m = 10.0, θ_a = θa, θ_s = 298.0,
                                 q_a = 0.009, q_s = surface_specific_humidity(w, f), z0m = 1e-3, z0h = 1e-4)
        @test fw.E ≈ 0.3 * ρ * sw.C_E * sw.U * (qsat - 0.009) rtol = 1e-12
        @test fw.LE ≈ latent_heat_vapor(298.0) * fw.E rtol = 1e-14
        @test fw.Rn ≈ fw.H + fw.LE + fw.G rtol = 1e-12
    end

    @testset "SlabLand.step_surface!: C_s ΔT = (G + Q_extra) Δt (#129)" begin
        f = SurfaceForcing(sw_down = 500.0, lw_down = 330.0, T_air = 293.0, q_air = 0.006, U = 3.0)
        land = SlabLand(; T = 300.0, β = 1.0)
        T0 = land.T
        fl = step_surface!(land, f, 30.0; Q_extra = -80.0)
        @test fl.G ≈ fl.Rn - fl.H - fl.LE - 80.0 rtol = 1e-12
        @test (land.T - T0) * land.C_s / 30.0 ≈ fl.G rtol = 1e-12
    end

    @testset "Exports: one binding per name, no ambiguities" begin
        @test all(n -> Base.isexported(Aeolus, n) && getfield(Aeolus, n) === getfield(SE, n),
                  (:SurfaceConstants, :DEFAULT_SURFACE_CONSTANTS, :BusingerDyerBeljaarsHoltslag,
                   :BusingerDyerLogLinear, :psi_m, :psi_h, :phi_m, :phi_h, :solve_surface_layer,
                   :obukhov_length, :bulk_richardson, :AbstractSurface, :SurfaceForcing, :SurfaceFluxes,
                   :surface_temperature, :roughness_lengths, :surface_albedo,
                   :surface_emissivity, :turbulent_fluxes, :step_surface!, :net_radiation))
        @test isempty(Test.detect_ambiguities(SE))
        # the fog column's mixing length uses the contract's κ, not a copy
        @test Aeolus.RadiationFog._KARMAN === DEFAULT_SURFACE_CONSTANTS.κ
    end
end
