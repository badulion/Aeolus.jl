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
# validation/support/urban_validation_driver.jl; what was not validated is listed in
# docs/src/theory/urban_validation.md.

using Test, Printf
using Aeolus
using Aeolus.UrbanSurface
using Aeolus.Radiosity
using Statistics: mean, std

include(joinpath(@__DIR__, "support", "urban_validation_driver.jl"))

# Hour (UTC) of the maximum of a diurnal composite of `x` over the samples `sel`.
function composite(x, hours, sel)
    return [mean(x[sel .& (hours .== h)]) for h in 0:23]
end
peak_hour(c) = argmax(c) - 1

@testset "BUBBLE / CAPITOUL validation (#30)" begin

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
    # (Pigeon et al. 2009, Table 2) sits directly over the heated / free-floating
    # zone and does not undercool at night, while the observed (pitched, not flat)
    # roof drops below 15 °C on summer nights (Moscicki & Voogt 2007, §4.1.3).
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

# ===========================================================================
@testset "BUBBLE: surface energy balance of clear June days" begin
    w = power_forcing("nasa_power_basel_bubble_hourly.csv")
    ob = bubble_observations()
    days_hm = (17, 18, 19, 22, 23, 29, 30)     # Hamdi & Masson (2008)
    days_hs = (17, 18, 23, 26, 30)             # Hamdi & Schayes (2005)
    k0 = power_index(w, 2002, 6, 1)
    nh = 30 * 24
    ev = [k for k in k0:k0+nh-1 if w.day[k] in days_hm]

    # The forcing itself: observed S↓ + L↓ = Q* + S↑ + L↑ at the tower
    # (HM08 Table 4, all hours of the seven days) against the satellite values.
    obs_in = ob[("HM08", "T", "Qstar")].obs + ob[("HM08", "T", "Sup")].obs + ob[("HM08", "T", "Lup")].obs
    pow_in = mean(w.SW[ev] .+ w.LW[ev])
    @printf("  forcing check, 7 clear days: S↓ + L↓ = %.1f W m⁻² (CERES) vs %.0f (tower, Q* + S↑ + L↑)\n",
            pow_in, obs_in)
    @test abs(pow_in / obs_in - 1) < 0.02
    # The grid-box air temperature lacks the urban heat island (and may carry a
    # reanalysis bias; the two are not separable): bias-correct it to the tower
    # mean at 18 m over the same days (HM08 Table 3) — one constant.
    ΔT = ob[("HM08t3", "T", "Tair18m")].obs - mean(w.T2M[ev])
    @printf("  T2M (MERRA-2) is %.2f K below the tower's 18 m mean; offset applied\n", ΔT)

    layers = (wall = bubble_wall_layers(), roof = bubble_roof_layers(), ground = bubble_road_layers())
    Tm = mean(w.T2M[k0:k0+nh-1]) + 273.15 + ΔT
    # The source area holds both street directions (67° and 157°, HS05 Table 1):
    # equal weights (assumption).
    outs = map(BUBBLE_CANYONS) do c
        case = canyon_case(c; layers, materials = bubble_materials(), T0 = Tm, T_deep = Tm)
        run_canyon!(case, w, k0, nh; site = BUBBLE_SITE, dt = 3600.0, T_offset = ΔT)
    end
    avg(key) = (getfield(outs[1], key) .+ getfield(outs[2], key)) ./ 2
    day, hour = outs[1].day, outs[1].hour
    inwin(p, h) = p === :T || (p === :D && 8 <= h <= 15) || (p === :N && (h >= 22 || h <= 2))
    sel(days, p) = [d in days && inwin(p, h) for (d, h) in zip(day, hour)]
    m(key, s) = mean(avg(key)[s])

    # --- Hamdi & Masson (2008), Table 4: all hours of the seven days --------
    sT = sel(days_hm, :T)
    turb_obs = ob[("HM08", "T", "Qstar")].obs - ob[("HM08", "T", "dQS")].obs   # QH + QE
    @printf("  HM08 T   %-26s %8s %6s %8s %8s\n", "", "model", "obs", "TEB-SBL", "SBL rmse")
    rows = (("Sup", :Sup, ob[("HM08", "T", "Sup")].obs), ("Lup", :Lup, ob[("HM08", "T", "Lup")].obs),
            ("Qstar", :Qstar, ob[("HM08", "T", "Qstar")].obs),
            ("QH", :QH, turb_obs), ("dQS", :dQS, ob[("HM08", "T", "dQS")].obs))
    for (q, key, o) in rows
        r = ob[("HM08", "T", q)]
        label = q == "QH" ? "QH (obs: QH + QE)" : q
        @printf("  HM08 T   %-26s %8.1f %6.0f %8.0f %8.0f\n", label, m(key, sT), o, r.SBL, r.rmseSBL)
        # within the RMSE of TEB forced by the tower at the same site and days
        if key === :Sup
            # S↑ misses by 0.3 W m⁻²: 29.7 against 34 ± 4. The model's canyon
            # array reflects 9.2 % of S↓; the source area's 16 % vegetation is
            # not modelled — at the package's park-class albedo (0.205) it would
            # add ≈ 0.16 × (0.205 - 0.092) × 324 ≈ 6 W m⁻² (an estimate).
            # Recorded, not loosened.
            @test_broken abs(m(key, sT) - o) <= r.rmseSBL
            @test m(key, sT) < o
        else
            @test abs(m(key, sT) - o) <= r.rmseSBL
        end
    end

    # --- Hamdi & Schayes (2005), Table 2: day 08-16 UTC, night 22-03 UTC ----
    for p in (:D, :N)
        s = sel(days_hs, p); P = string(p)
        o(q) = ob[("HS05", P, q)].obs
        # Tolerance: HM08's TEB-SBL D / N RMSE, borrowed as an error scale. HM08 does
        # not define its D / N windows and uses other days, so this is a proxy, not
        # the error of TEB in the HS05 windows.
        rm(q) = ob[("HM08", P, q)].rmseSBL
        Q = m(:Qstar, s)
        @printf("  HS05 %s: S↑ %.1f (obs %.0f)  L↑ %.1f (%.0f)  Q* %.1f (%.0f)  QH/Q* %.2f (QH+QE: %.2f)  ΔQS/Q* %.2f (%.2f)\n",
                P, m(:Sup, s), o("Sup"), m(:Lup, s), o("Lup"), Q, o("Qstar"),
                m(:QH, s) / Q, o("QH/Qstar") + o("QE/Qstar"), m(:dQS, s) / Q, o("dQS/Qstar"))
        @test abs(Q - o("Qstar")) <= rm("Qstar")
        @test abs(m(:Sup, s) - o("Sup")) <= rm("Sup")
        if p === :N
            @test abs(m(:Lup, s) - o("Lup")) <= rm("Lup")
            # at night storage release exceeds the radiative loss and keeps QH > 0
            @test m(:QH, s) > 0
            @test m(:dQS, s) / Q > 1
        else
            # The daytime canopy is radiometrically too warm: L↑ +39 W m⁻² against
            # TEB's 15. The 16 % vegetation of the source area is not modelled;
            # at the air temperature (28.7 °C) it would emit ≈ 469 W m⁻² instead
            # of the urban 560, i.e. ≈ 15 W m⁻² less L↑ (an estimate). The
            # satellite S↓ in this window is 3.4 % above the tower's 741 W m⁻².
            # Recorded, not loosened.
            @test_broken abs(m(:Lup, s) - o("Lup")) <= rm("Lup")
            @test m(:Lup, s) > o("Lup")
        end
    end

    # --- storage hysteresis (HM08 §4e: ΔQS rises faster than QH in the
    # morning and peaks earlier; it turns negative 1-2 h before Q*) ----------
    cQ = composite(avg(:Qstar), hour, sT)
    cH = composite(avg(:QH), hour, sT)
    cS = composite(avg(:dQS), hour, sT)
    after(c) = findfirst(h -> h > 12 && c[h+1] < 0, 0:23) - 1
    @printf("  composite peaks (UTC): ΔQS %02d, Q* %02d, QH %02d; afternoon sign change ΔQS %02d, Q* %02d\n",
            peak_hour(cS), peak_hour(cQ), peak_hour(cH), after(cS), after(cQ))
    @test peak_hour(cS) < peak_hour(cQ) <= peak_hour(cH)
    @test 1 <= after(cQ) - after(cS) <= 2
end

end
