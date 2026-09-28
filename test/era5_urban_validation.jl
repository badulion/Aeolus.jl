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

include(joinpath(@__DIR__, "urban_validation_driver.jl"))

# Hour (UTC) of the maximum of a diurnal composite of `x` over the samples `sel`.
function composite(x, hours, sel)
    return [mean(x[sel .& (hours .== h)]) for h in 0:23]
end
peak_hour(c) = argmax(c) - 1

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

# ===========================================================================
@testset "canyon scenes: central street sky-view factor vs the infinite canyon" begin
    # The cases are finite canyons (~5 H long) evaluated on their two central
    # rows. Their street must see the sky like an infinitely long canyon.
    for (c, mats) in ((CAPITOUL_CANYONS[1], capitoul_materials()),
                      (CAPITOUL_CANYONS[2], capitoul_materials()),
                      (CAPITOUL_CANYONS[3], capitoul_materials()),
                      (BUBBLE_CANYONS[1], bubble_materials()))
        sc, vf, _ = canyon_scene(c; materials = mats)
        rows = central_rows(c)
        g = [i for (i, f) in enumerate(sc.facets) if f.kind === :ground && f.col[2] in rows]
        ψ = sum(vf.F_sky[g] .* vf.area[g]) / sum(vf.area[g])
        ψ∞ = canyon_sky_view_factor(c.H / c.W)
        @printf("  %-17s H/W %.2f, %d facets: central street ψ %.4f vs infinite canyon %.4f (%+.1f %%)\n",
                c.name, c.H / c.W, nfacets(sc), ψ, ψ∞, 100 * (ψ / ψ∞ - 1))
        @test ψ / ψ∞ - 1 < 0.02      # the open ends only ever add sky
        @test ψ >= ψ∞
    end
end

# ===========================================================================
@testset "CAPITOUL: road, wall and roof temperatures over three seasons" begin
    w = power_forcing("nasa_power_toulouse_capitoul_hourly.csv")
    obs = capitoul_statistics()
    layers = (wall = capitoul_wall_layers(), roof = capitoul_roof_layers(),
              ground = capitoul_road_layers())
    # 14 days of spin-up, then the 32-day periods of Moscicki & Voogt (2007),
    # Table 2.2: 15 Jul-15 Aug 2004, 15 Oct-15 Nov 2004, 15 Jan-15 Feb 2005.
    periods = ((2004, 7), (2004, 10), (2005, 1))
    spin = 14 * 24
    runs = Dict{Tuple{String, Int}, Any}()
    for (y, m) in periods
        k0 = power_index(w, y, m, 1)
        nh = 46 * 24
        # initial and deep-ground temperature: the window's mean air temperature (assumption)
        Tm = mean(w.T2M[k0:k0+nh-1]) + 273.15
        for c in CAPITOUL_CANYONS
            case = canyon_case(c; layers, materials = capitoul_materials(), T0 = Tm, T_deep = Tm)
            o = run_canyon!(case, w, k0, nh; site = CAPITOUL_SITE, dt = 3600.0)
            runs[(c.name, m)] = map(x -> x[spin+1:end], o)
        end
    end
    series(name, key) = reduce(vcat, [runs[(name, m)][key] for (_, m) in periods])
    names = [c.name for c in CAPITOUL_CANYONS]
    toC(x) = x .- 273.15
    road  = toC(mean(hcat([series(n, :road) for n in names]...); dims = 2)[:])
    walls = toC(mean(hcat([series(n, k) for n in names for k in (:wall_pos, :wall_neg)]...);
                     dims = 2)[:])
    roof  = toC(series("Alsace", :roof))      # the observed roof is on the Alsace canyon
    air   = toC(series("Alsace", :T_air))
    n = length(road)
    @test n == 3 * 32 * 24

    M = (roads = mean(road), walls = mean(walls), roof = mean(roof))
    S = (roads = std(road), walls = std(walls), roof = std(roof))
    @printf("  CAPITOUL, 3 × 32 days (%d hourly samples; forcing air %.2f ± %.2f °C):\n", n,
            mean(air), std(air))
    @printf("  %-6s %16s %16s %16s %8s %8s\n", "", "mean: model", "obs", "TEB", "bias", "TEB RMSE")
    for s in (:roads, :walls, :roof)
        @printf("  %-6s %9.2f ± %5.2f %9.2f ± %5.2f %9.2f ± %5.2f %+8.2f %8.2f\n", s,
                M[s], S[s], obs[:O][s], obs[:sO][s], obs[:M][s], obs[:sM][s],
                M[s] - obs[:O][s], obs[:RMSE][s])
    end

    # Mean temperature within the per-sample error of TEB forced by the tower
    # (its RMSE). Measured biases are quoted in the PR and the theory page.
    @test abs(M.roads - obs[:O].roads) <= obs[:RMSE].roads
    @test abs(M.walls - obs[:O].walls) <= obs[:RMSE].walls
    # The roof fails that yardstick: +3.9 K against TEB's RMSE of 3.50 K (TEB's
    # own bias is +1.40 K). The flat 6 cm tile + 3 cm wood roof of the TEB input
    # (Pigeon et al. 2008, Table 2) sits directly over the heated / free-floating
    # zone and does not undercool at night, while the observed pitched roof over
    # an attic drops below 15 °C on summer nights (Moscicki & Voogt 2007, §4.1.3).
    # Recorded, not loosened.
    @test_broken abs(M.roof - obs[:O].roof) <= obs[:RMSE].roof
    @test M.roof > obs[:O].roof                  # the direction of the known roof miss

    # Amplitude: the standard deviation (diurnal + seasonal) within the largest
    # ratio TEB itself shows against the same observations (roads, 12.09 / 9.38).
    r_teb = maximum(obs[:sM][s] / obs[:sO][s] for s in (:roads, :walls, :roof))
    for s in (:roads, :walls, :roof)
        @test 1 / r_teb <= S[s] / obs[:sO][s] <= r_teb
    end
    # Observed ordering of the variability: roof > roads > walls.
    @test S.roof > S.roads > S.walls

    # Orientation. Alsace runs north-south: its east- and west-facing walls
    # have the same mean and mirror-image days — the east wall peaks in the
    # morning, the west wall in the afternoon (solar noon ≈ 11:55 UTC).
    al = runs[("Alsace", 7)]
    summer = trues(length(al.road))
    cE = composite(al.wall_pos, al.hour, summer)       # +x wall faces 0 + 90 = east
    cW = composite(al.wall_neg, al.hour, summer)
    @printf("  Alsace (N-S) summer composite: east wall peaks %02d-%02d UTC (%.1f °C), west wall %02d-%02d UTC (%.1f °C)\n",
            peak_hour(cE), peak_hour(cE) + 1, maximum(cE) - 273.15,
            peak_hour(cW), peak_hour(cW) + 1, maximum(cW) - 273.15)
    @test peak_hour(cE) + 1 <= 12 <= peak_hour(cW)
    ΔEW = mean(series("Alsace", :wall_pos)) - mean(series("Alsace", :wall_neg))
    # Pomme (axis 120°): +x wall faces 210° (SSW), -x faces 30° (NNE);
    # Remusat (axis 60°): +x faces 150° (SSE), -x faces 330° (NNW).
    Δpomme = mean(series("Pomme", :wall_pos)) - mean(series("Pomme", :wall_neg))
    Δremus = mean(series("Remusat", :wall_pos)) - mean(series("Remusat", :wall_neg))
    @printf("  mean wall difference: Alsace E - W %+.2f K, Pomme SSW - NNE %+.2f K, Remusat SSE - NNW %+.2f K\n",
            ΔEW, Δpomme, Δremus)
    @test abs(ΔEW) < 0.3
    @test Δpomme > 0.3 && Δremus > 0.3          # equator-facing walls are warmer
end

end
