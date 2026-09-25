# Era V contract — SurfaceExchange (issue #92).
#
# Material constants, Monin–Obukhov similarity and the surface interface
# that every Era V surface model builds on.

using Test, Printf
using Aeolus
using Aeolus.SurfaceExchange
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, PhasePartition, air_density,
                                  heat_capacity_moist, saturation_specific_humidity,
                                  latent_heat_vapor

const SE = Aeolus.SurfaceExchange

@testset "Era V — SurfaceExchange contract (#92)" begin

    @testset "Material constants delegate to the era contracts" begin
        c = DEFAULT_SURFACE_CONSTANTS
        # Delegated, not restated: these must be the very same numbers.
        @test LH_v0(c) === DEFAULT_CONSTANTS.LH_v0
        @test LH_s0(c) === DEFAULT_CONSTANTS.LH_s0
        @test LH_f0(c) === DEFAULT_CONSTANTS.LH_s0 - DEFAULT_CONSTANTS.LH_v0
        @test grav(c) === DEFAULT_CONSTANTS.grav
        @test cp_water(c) === DEFAULT_CONSTANTS.cp_liquid
        @test cp_ice(c) === DEFAULT_CONSTANTS.cp_ice
        @test c.σ_SB === Aeolus.RadiationCore.STEFAN_BOLTZMANN
        @test c.σ_SB == 5.670374419e-8
        @test von_karman(c) == 0.40
        # Physical sanity: L_f ≈ 334 kJ/kg; ice floats; ice conducts ~4x water.
        @test LH_f0(c) ≈ 3.336e5 rtol = 1e-3
        @test c.ρ_ice / c.ρ_water ≈ 0.917
        @test 3.5 < c.k_ice / c.k_water < 4.5
        @test c.k_air < c.k_water < c.k_ice
        # A different thermodynamic reference propagates through the accessors.
        c2 = SurfaceConstants(thermo = Aeolus.MoistThermodynamics.ThermodynamicConstants(grav = 9.81))
        @test grav(c2) == 9.81
        @info "SurfaceConstants" L_f = LH_f0(c) k_ice = c.k_ice σ = c.σ_SB κ = c.κ
    end
    @testset "MOST stability functions: normalisation, φ–ψ consistency" begin
        sf = BusingerDyerBeljaarsHoltslag()
        @test BusingerDyerBeljaarsHoltslag <: AbstractStabilityFunctions
        # Neutral normalisation, both branches.
        for f in (psi_m, psi_h)
            @test f(sf, 0.0) == 0
            @test abs(f(sf, -1e-12)) < 1e-11 && abs(f(sf, 1e-12)) < 1e-11
        end
        for f in (phi_m, phi_h)
            @test f(sf, 0.0) == 1
            # Continuity across ζ = 0 (the two branches are different formulas).
            @test f(sf, -1e-10) ≈ f(sf, 1e-10) atol = 1e-8
        end
        # ψ(ζ) = ∫₀^ζ (1 - φ)/x dx  ⇔  dψ/dζ = (1 - φ)/ζ, checked by a
        # 4th-order central difference in both regimes.
        worst = 0.0
        for ζ in (-5.0, -2.0, -1.0, -0.3, -0.05, 0.05, 0.3, 1.0, 2.0, 5.0, 10.0),
            (ψ, φ) in ((psi_m, phi_m), (psi_h, phi_h))
            h = 1e-3 * max(abs(ζ), 0.1)
            dψ = (-ψ(sf, ζ + 2h) + 8ψ(sf, ζ + h) - 8ψ(sf, ζ - h) + ψ(sf, ζ - 2h)) / (12h)
            err = abs(dψ - (1 - φ(sf, ζ)) / ζ)
            worst = max(worst, err)
            @test err < 1e-8
        end
        @info "MOST φ–ψ consistency: max |dψ/dζ − (1−φ)/ζ|" worst
        # Physically: shear and gradients are reduced by convection and
        # enhanced by stability, monotonically.
        ζs = range(-5, 10; length = 301)
        for φ in (phi_m, phi_h)
            v = [φ(sf, ζ) for ζ in ζs]
            @test all(diff(v) .> 0)
        end
        for ψ in (psi_m, psi_h)
            v = [ψ(sf, ζ) for ζ in ζs]
            @test all(diff(v) .< 0)
        end
    end

    @testset "MOST stability functions: published values" begin
        sf = BusingerDyerBeljaarsHoltslag()
        # Exact values at ζ = -1/2, where 1 - 16ζ = 9 so x = √3, y = 3:
        #   ψ_h = 2 ln 2,  ψ_m = 2 ln((1+√3)/2) + ln 2 - 2π/3 + π/2 (Paulson 1970).
        @test psi_h(sf, -0.5) ≈ 2log(2) rtol = 1e-14
        @test psi_m(sf, -0.5) ≈ 2log((1 + √3) / 2) + log(2) - 2π / 3 + π / 2 rtol = 1e-14
        @test phi_m(sf, -0.5) ≈ 1 / √3 rtol = 1e-14
        @test phi_h(sf, -0.5) ≈ 1 / 3 rtol = 1e-14
        # Paulson (1970) with Dyer (1974) γ = 16 at ζ = -1:
        # ψ_m = 1.1162, ψ_h = 2 ln((1+√17)/2) = 1.8812.
        @test psi_m(sf, -1.0) ≈ 1.1162 atol = 1e-4
        @test psi_h(sf, -1.0) ≈ 1.8812 atol = 1e-4
        # Beljaars & Holtslag (1991), Eqs. 28 and 32, at ζ = 1:
        # ψ_m = -[1 + (2/3)(1 - 5/0.35)e^{-0.35} + (2/3)(5/0.35)] = -4.2823,
        # ψ_h = -[(5/3)^{3/2} + (2/3)(1 - 5/0.35)e^{-0.35} + (2/3)(5/0.35) - 1] = -4.4339.
        @test psi_m(sf, 1.0) ≈ -4.2823 atol = 1e-4
        @test psi_h(sf, 1.0) ≈ -4.4339 atol = 1e-4
        # Very stable: finite, and φ_m grows only linearly (a = 1), not
        # exponentially — the property that keeps nocturnal mixing alive.
        for f in (phi_m, phi_h, psi_m, psi_h)
            @test isfinite(f(sf, 10.0)) && isfinite(f(sf, 100.0))
        end
        @test phi_m(sf, 100.0) ≈ 101 rtol = 1e-3
        @info "MOST published values" ψm_m1 = psi_m(sf, -1.0) ψh_m1 = psi_h(sf, -1.0) ψm_p1 = psi_m(sf, 1.0) ψh_p1 = psi_h(sf, 1.0) φm_p10 = phi_m(sf, 10.0) φh_p10 = phi_h(sf, 10.0)
    end
    @testset "Surface-layer solver: neutral limit is exact" begin
        κ = von_karman()
        worst = 0.0
        for z in (2.0, 10.0, 40.0), z0 in (1e-4, 0.01, 0.5), U in (0.5, 5.0, 20.0)
            s = solve_surface_layer(; U = U, z_m = z, θ_a = 290.0, θ_s = 290.0,
                                    z0m = z0, z0h = z0 / 10)
            ref = (κ / log(z / z0))^2
            worst = max(worst, abs(s.C_D - ref) / ref)
            @test s.ζ == 0 && isinf(s.L) && s.converged
            @test s.C_D ≈ ref rtol = 4eps()
            @test s.C_H ≈ κ^2 / (log(z / z0) * log(z / (z0 / 10))) rtol = 4eps()
            @test s.u_star ≈ κ * U / log(z / z0) rtol = 4eps()
        end
        @info "neutral C_D vs (κ/ln(z/z0))²" max_rel_err = worst
    end

    @testset "Surface-layer solver: self-consistency of the returned L" begin
        sf = BusingerDyerBeljaarsHoltslag()
        κ, g = von_karman(), grav()
        ε = SE._virtual_coefficient(DEFAULT_SURFACE_CONSTANTS)
        worst_prof, worst_L, worst_Ri = 0.0, 0.0, 0.0
        for (θs, qs, U) in ((300.0, 0.012, 5.0), (305.0, 0.015, 1.0), (292.0, 0.010, 8.0),
                            (287.0, 0.006, 4.0), (284.0, 0.005, 3.0), (289.5, 0.0085, 10.0))
            θa, qa, zm, zh, z0m, z0h, z0q = 290.0, 0.008, 10.0, 2.0, 0.1, 0.01, 0.005
            s = solve_surface_layer(sf; U = U, z_m = zm, z_h = zh, θ_a = θa, θ_s = θs,
                                    q_a = qa, q_s = qs, z0m = z0m, z0h = z0h, z0q = z0q)
            @test s.converged
            L = s.L
            # The flux-profile relations evaluated at the returned L reproduce
            # the input differences.
            Φm = log(zm / z0m) - psi_m(sf, zm / L) + psi_m(sf, z0m / L)
            Φh = log(zh / z0h) - psi_h(sf, zh / L) + psi_h(sf, z0h / L)
            Φq = log(zh / z0q) - psi_h(sf, zh / L) + psi_h(sf, z0q / L)
            for (got, want) in ((s.u_star / κ * Φm, U), (s.θ_star / κ * Φh, θa - θs),
                                (s.q_star / κ * Φq, qa - qs))
                worst_prof = max(worst_prof, abs(got - want) / abs(want))
                @test got ≈ want rtol = 1e-12
            end
            # ...and L is the Obukhov length of the fluxes it produced.
            θv_star = s.θ_star * (1 + ε * qa) + ε * θa * s.q_star
            @test θv_star ≈ s.θv_star rtol = 1e-14
            Lc = obukhov_length(s.u_star, θv_star, θa * (1 + ε * qa))
            worst_L = max(worst_L, abs(Lc - L) / abs(L))
            @test Lc ≈ L rtol = 1e-10
            # Signs: fluxes positive upward, L < 0 when the surface is warmer.
            @test sign(s.θ_star) == sign(θa - θs)
            @test (θs * (1 + ε * qs) > θa * (1 + ε * qa)) == (L < 0)
        end
        # Dry, z_m = z_h: the bulk Richardson number is ζ Φ_h / Φ_m².
        for θs in (280.0, 286.0, 289.0, 291.0, 295.0, 305.0)
            s = solve_surface_layer(sf; U = 4.0, z_m = 10.0, θ_a = 290.0, θ_s = θs,
                                    z0m = 0.05, z0h = 0.005)
            Ri = bulk_richardson(10.0, 4.0, 290.0, θs)
            # Φm = κ/√C_D and Φh = κ²/(C_H Φm), so ζ Φh/Φm² = ζ C_D^{3/2}/(κ C_H).
            rel = abs(Ri - s.ζ * s.C_D^1.5 / (κ * s.C_H)) / abs(Ri)
            worst_Ri = max(worst_Ri, rel)
            @test rel < 1e-10
        end
        @info "solver self-consistency" profile_rel_err = worst_prof obukhov_rel_err = worst_L richardson_rel_err = worst_Ri
    end

    @testset "Surface-layer solver: stability dependence and limits" begin
        sf = BusingerDyerBeljaarsHoltslag()
        kw = (U = 3.0, z_m = 10.0, θ_a = 290.0, z0m = 0.1, z0h = 0.01)
        stable  = solve_surface_layer(sf; kw..., θ_s = 287.0)
        neutral = solve_surface_layer(sf; kw..., θ_s = 290.0)
        unstable = solve_surface_layer(sf; kw..., θ_s = 296.0)
        @test stable.ζ > 0 && unstable.ζ < 0
        @test stable.C_H < neutral.C_H < unstable.C_H
        @test stable.C_D < neutral.C_D < unstable.C_D
        # C_H decreases monotonically as the surface cools relative to the air.
        CH = [solve_surface_layer(sf; kw..., θ_s = θs).C_H for θs in 300.0:-0.5:284.0]
        @test all(diff(CH) .< 0)
        # Strongly stable (ζ ≈ 10) is resolved, finite and still mixing.
        vs = solve_surface_layer(sf; U = 2.0, z_m = 10.0, θ_a = 290.0, θ_s = 280.0,
                                 z0m = 0.1, z0h = 0.01)
        @test vs.converged && 5 < vs.ζ < 20
        @test all(isfinite, (vs.u_star, vs.θ_star, vs.L, vs.C_D, vs.C_H))
        @test vs.C_H > 0
        # Beyond ζ_max the solver clamps and says so rather than returning NaN.
        cl = solve_surface_layer(sf; U = 0.5, z_m = 10.0, θ_a = 290.0, θ_s = 275.0,
                                 z0m = 0.1, z0h = 0.01)
        @test !cl.converged && cl.ζ == 1000 && isfinite(cl.C_H)
        # Free convection: calm wind still gives a finite, convective u*.
        fc = solve_surface_layer(sf; U = 0.0, z_m = 10.0, θ_a = 290.0, θ_s = 300.0,
                                 z0m = 0.1, z0h = 0.01)
        @test fc.converged && fc.U == 0.1 && fc.ζ < -10 && isfinite(fc.u_star)
        g = solve_surface_layer(sf; U = 0.0, z_m = 10.0, θ_a = 290.0, θ_s = 300.0,
                                z0m = 0.1, z0h = 0.01, gustiness = 1.0)
        @test g.U == 1.0 && g.u_star > fc.u_star
        # Vapour buoyancy alone destabilises: equal θ, moister surface.
        mv = solve_surface_layer(sf; U = 3.0, z_m = 10.0, θ_a = 290.0, θ_s = 290.0,
                                 q_a = 0.005, q_s = 0.015, z0m = 0.1, z0h = 0.01)
        @test mv.ζ < 0 && mv.θ_star == 0 && mv.q_star < 0
        @info "C_H stable / neutral / unstable" stable = stable.C_H neutral = neutral.C_H unstable = unstable.C_H ζ_stable = stable.ζ ζ_unstable = unstable.ζ ζ_very_stable = vs.ζ C_H_very_stable = vs.C_H ζ_free_convection = fc.ζ iterations = (stable.iterations, unstable.iterations, vs.iterations)
    end
    @testset "Surface interface: one surface_temperature, SlabLand implements it" begin
        # One generic, owned by SurfaceExchange, extended by Radiation.SlabLand.
        @test Aeolus.surface_temperature === SE.surface_temperature
        @test Aeolus.Radiation.surface_temperature === SE.surface_temperature
        @test SE.surface_albedo === Aeolus.RadiationCore.surface_albedo
        @test SE.surface_emissivity === Aeolus.RadiationCore.surface_emissivity
        @test SlabLand <: AbstractSurface
        land = SlabLand(; T = 300.0, albedo = 0.25, z0m = 0.2)
        @test surface_temperature(land) == 300.0
        @test roughness_lengths(land) == (z0m = 0.2, z0h = 0.02, z0q = 0.02)
        @test surface_albedo(land) == 0.25 && surface_emissivity(land) == 1.0
        @test moisture_availability(land) == 0
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

    @testset "Surface interface: default turbulent_fluxes" begin
        thermo = DEFAULT_CONSTANTS
        f = SurfaceForcing(sw_down = 500.0, lw_down = 330.0, T_air = 293.0,
                           q_air = 0.009, U = 6.0, p = 1.0e5, z_ref = 10.0)
        land = SlabLand(; T = 303.0)
        fl = turbulent_fluxes(land, f)
        # Dry slab: no moisture flux at all (q_s = q_a, not q_s = 0).
        @test fl.E == 0 && fl.LE == 0
        @test fl.H > 0 && fl.τ > 0                 # warm ground heats the air
        @test fl.Rn ≈ fl.H + fl.LE + fl.G rtol = 1e-14
        @test fl.Rn ≈ 0.8 * 500 + 330 - SE.STEFAN_BOLTZMANN * 303.0^4 rtol = 1e-14
        # The fluxes are the bulk formulae with the solver's coefficients.
        θa = 293.0 + Aeolus.MoistThermodynamics.dry_adiabatic_lapse_rate(
                 PhasePartition(0.009), thermo) * 10
        sol = solve_surface_layer(; U = 6.0, z_m = 10.0, θ_a = θa, θ_s = 303.0,
                                  q_a = 0.009, q_s = 0.009, z0m = 0.1, z0h = 0.01)
        ρ = air_density(293.0, 1.0e5, PhasePartition(0.009))
        cp = heat_capacity_moist(PhasePartition(0.009))
        @test fl.H ≈ ρ * cp * sol.C_H * sol.U * (303.0 - θa) rtol = 1e-12
        @test fl.τ ≈ ρ * sol.C_D * sol.U^2 rtol = 1e-12

        # Wet surface: E = β ρ C_E U (q_sat(T_s) - q_a), LE = L_v(T_s) E.
        for β in (1.0, 0.3)
            w = _WetPatch(298.0, β)
            fw = turbulent_fluxes(w, f)
            qsat = saturation_specific_humidity(298.0, 1.0e5)
            qs = surface_specific_humidity(w, f)
            @test qs ≈ 0.009 + β * (qsat - 0.009) rtol = 1e-14
            sw = solve_surface_layer(; U = 6.0, z_m = 10.0, θ_a = θa, θ_s = 298.0,
                                     q_a = 0.009, q_s = qs, z0m = 1e-3, z0h = 1e-4)
            @test fw.E ≈ β * ρ * sw.C_E * sw.U * (qsat - 0.009) rtol = 1e-12
            @test fw.LE ≈ latent_heat_vapor(298.0) * fw.E rtol = 1e-14
            @test fw.Rn ≈ 0.94 * 500 + 0.97 * 330 - 0.97 * SE.STEFAN_BOLTZMANN * 298.0^4 rtol = 1e-14
            @test fw.Rn ≈ fw.H + fw.LE + fw.G rtol = 1e-12
        end
        wet = turbulent_fluxes(_WetPatch(298.0, 1.0), f)
        # Dew: a surface colder than the air's dew point condenses (E < 0).
        dew = turbulent_fluxes(_WetPatch(280.0, 1.0), f)
        @test dew.E < 0 && dew.H < 0
        @info "turbulent_fluxes (W m⁻²)" slab_H = fl.H slab_Rn = fl.Rn wet_H = wet.H wet_LE = wet.LE wet_Bowen = wet.H / wet.LE dew_E = dew.E
    end

    @testset "SlabLand.step_surface!: energy closure and equilibrium" begin
        f = SurfaceForcing(sw_down = 0.0, lw_down = 300.0, T_air = 285.0,
                           q_air = 0.005, U = 4.0)
        land = SlabLand(; T = 295.0, C_s = 1e5)
        T0 = land.T
        fl = step_surface!(land, f, 60.0)
        @test land.T - T0 ≈ fl.G * 60.0 / 1e5 rtol = 1e-12     # C_s ΔT = G Δt exactly
        @test land.T < T0                                     # clear night: cools
        # Long integration: the slab reaches Rn = H (G → 0) at a skin
        # temperature between radiative equilibrium and the air.
        for _ in 1:20_000
            fl = step_surface!(land, f, 60.0)
        end
        T_rad = (300.0 / SE.STEFAN_BOLTZMANN)^(1 / 4)
        @test abs(fl.G) < 1e-6
        @test T_rad < land.T < 285.0
        @test fl.H < 0                                        # stable, air heats ground
        @info "SlabLand night equilibrium" T_s = land.T T_rad = T_rad H = fl.H G = fl.G
    end
end
