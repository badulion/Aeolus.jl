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
end
