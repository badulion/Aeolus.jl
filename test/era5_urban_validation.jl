# Phase 49 — BUBBLE / CAPITOUL validation of the urban canopy (#30).
#
# The acceptance test of the whole urban canopy: street canyons built from the
# campaign papers, every facet an UrbanSurface conduction column coupled to the
# others through the radiosity system, the buildings BuildingEnergy zones solved
# with their facets (step_buildings!), driven offline by satellite / reanalysis
# forcing for the campaign dates (NASA POWER: CERES SYN1deg radiation, MERRA-2
# meteorology), and compared with published observations:
#
#   * CAPITOUL, Toulouse (2004-05): means and standard deviations of the road,
#     wall and roof surface temperatures over three 32-day periods — Moscicki &
#     Voogt (2007), Table 4.1 — with the skill of TEB forced by the tower
#     (same table) as the yardstick.
#   * BUBBLE, Basel-Sperrstrasse (June 2002): the clear-day surface energy
#     balance — Hamdi & Masson (2008), Table 4 (TEB skill in the same table),
#     and Hamdi & Schayes (2005), Table 2 (day 08-16 UTC / night 22-03 UTC).
#
# Tolerances are published model-observation errors of the reference model
# (TEB forced by the campaign tower), not numbers chosen after measuring. What
# fails that yardstick is recorded with @test_broken and explained, not
# loosened. Geometry, materials and forcing are labelled with their sources in
# test/urban_validation_driver.jl; what was not validated is listed in
# docs/src/theory/urban_validation.md.

using Test, Printf
using Aeolus
using Aeolus.UrbanSurface
using Aeolus.Radiosity
using Statistics: mean, std

@testset "BUBBLE / CAPITOUL validation (#30)" begin

# ===========================================================================
@testset "TEB canyon closures" begin
    # Hand-evaluated Masson (2000) / Lemonsu et al. (2004) eq. B1:
    # H = W = 20, z0 = 2, z_ref = 10: 2/π e^{-1/4} ln(3.3333) / ln(8.3333)
    ref = 2 / π * exp(-0.25) * log(20 / 3 / 2) / log((10 + 20 / 3) / 2)
    @test canyon_wind_speed(1.0, 10.0, 20.0, 20.0; z0_town = 2.0) ≈ ref rtol = 1e-14
    @test ref ≈ 0.28154 atol = 1e-5
    @test canyon_wind_speed(3.0, 10.0, 20.0, 20.0) ≈ 3 * canyon_wind_speed(1.0, 10.0, 20.0, 20.0)
    # forcing at the canyon top (z_ref → 0): only the orientation average and the
    # exponential decay remain
    @test canyon_wind_speed(1.0, 1e-9, 20.0, 10.0) ≈ 2 / π * exp(-0.5) rtol = 1e-8
    # deeper canyons are more sheltered; a higher forcing level carries less wind down
    @test canyon_wind_speed(2.0, 20.0, 30.0, 10.0) < canyon_wind_speed(2.0, 20.0, 20.0, 10.0)
    @test canyon_wind_speed(2.0, 40.0, 20.0, 10.0) < canyon_wind_speed(2.0, 20.0, 20.0, 10.0)
    @test_throws ArgumentError canyon_wind_speed(2.0, 20.0, 10.0, 20.0)   # H/W < 1
    @test_throws ArgumentError canyon_wind_speed(2.0, 0.0, 20.0, 10.0)
    @test_throws ArgumentError canyon_wind_speed(2.0, 20.0, 20.0, 10.0; z0_town = 7.0)
    # Rowley et al. (1930): 11.8 + 4.2 U
    @test rowley_convection_coefficient(0.0) == 11.8
    @test rowley_convection_coefficient(2.5) ≈ 22.3
    @test_throws ArgumentError rowley_convection_coefficient(-1.0)
end

end
