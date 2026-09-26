# Era V, Phase 48 (#25) — within-canopy wind and temperature profiles.
#
# Part 1 checks the tools the validation needs against analytic results:
# measured (tabulated) LAD profiles, the steady K-theory scalar column,
# the displacement height and the Harman–Finnigan scalar diffusivity.
# Part 2 compares the 1-D canopy column of PlantCanopy with published
# profiles from a wind tunnel (Brunet et al. 1994), an orchard (CHATS) and
# a Scots pine forest (Hyytiälä); the data and their provenance are in
# test/data/canopy_profiles.jl.

using Printf, Random
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, R_d, saturation_vapor_pressure

include(joinpath(@__DIR__, "data", "canopy_profiles.jl"))

# Linear interpolation of a profile (local to this file; prefixed so the
# name does not collide with helpers of other test files in this module).
function _cp_interp(zs, vs, z)
    i = clamp(searchsortedlast(zs, z), 1, length(zs) - 1)
    t = (z - zs[i]) / (zs[i+1] - zs[i])
    return (1 - t) * vs[i] + t * vs[i+1]
end

@testset "Era V — PlantCanopy: within-canopy profiles (#25)" begin

@testset "TabulatedLAD: measured profiles" begin
    # A triangle 0 → 2 → 0 on [1, 5] m: LAI = 4 exactly, apex at 3 m.
    p = TabulatedLAD([1.0, 3.0, 5.0], [0.0, 2.0, 0.0])
    @test leaf_area_index(p) == 4.0
    @test canopy_height(p) == 5.0
    @test leaf_area_density(p, 2.0) ≈ 1.0
    @test leaf_area_density(p, 0.5) == 0 && leaf_area_density(p, 5.5) == 0
    @test cumulative_leaf_area(p, 3.0) ≈ 2.0 rtol = 1e-13
    @test cumulative_leaf_area(p, 0.0) ≈ 4.0 rtol = 1e-13
    # Layer integrals conserve the LAI on any grid, including faces that
    # do not coincide with the table nodes.
    for zf in (collect(range(0, 6; length = 8)), [0.0, 0.7, 2.9, 3.1, 6.0], collect(0:0.013:6.5))
        lad = layer_leaf_area_density(p, zf)
        @test sum(lad .* diff(zf)) ≈ 4.0 rtol = 1e-12
    end
    # Rescaling to a quoted LAI keeps the shape.
    q = TabulatedLAD([1.0, 3.0, 5.0], [0.0, 2.0, 0.0]; LAI = 3.0)
    @test leaf_area_index(q) == 3.0
    @test leaf_area_density(q, 2.0) ≈ 0.75
    @test sum(layer_leaf_area_density(q, collect(0:0.1:5)) .* 0.1) ≈ 3.0 rtol = 1e-12
    @test_throws ArgumentError TabulatedLAD([0.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    @test_throws ArgumentError TabulatedLAD([0.0, 1.0], [1.0, -1.0])
    @test_throws ArgumentError TabulatedLAD([0.0, 1.0], [0.0, 0.0]; LAI = 1.0)

    # The digitised Hyytiälä needle-area profile integrates to the one-sided
    # LAI of 3.5 quoted by Launiainen et al. (2007, Table 2).
    hy = canopy_profile_observations().hyytiala
    ph = TabulatedLAD(hy.lad_z .* hy.h, hy.lad)
    @test leaf_area_index(ph) ≈ 3.5 atol = 0.02
    @printf("    Hyytiälä digitised LAD: one-sided LAI %.3f (published 3.5)\n", leaf_area_index(ph))
end

end
