# Era V contract — SurfaceExchange (issue #92).
#
# Material constants, Monin–Obukhov similarity and the surface interface
# that every Era V surface model builds on.

using Test, Printf
using Aeolus
using Aeolus.SurfaceExchange
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS

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
end
