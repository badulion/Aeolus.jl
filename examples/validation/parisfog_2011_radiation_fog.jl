# =========================================================================
# ParisFog, 14–15 November 2011 — radiation fog with droplet feedback
#
# Phase 44 (#9). Runs the 1-D RadiationFog column for the SIRTA radiation
# fog documented by
#
#   Mazoyer, M. et al. (2017). Large eddy simulation of radiation fog:
#   impact of dynamics on the fog life cycle. ACP, 17, 13017-13035.
#
# and compares its life cycle and radiative budget with those
# observations and with the SIRTA fog radiation climatology of
#
#   Wærsted, E. G. et al. (2017). Radiation in fog: quantification of the
#   impact on fog liquid water based on ground-based remote sensing.
#   ACP, 17, 10811-10835.
#
# Three runs from 23:20 UTC: the reference, the same with the droplets
# made transparent to radiation (no droplet-radiative feedback), and the
# same with the sun switched off — the sweep
# examples/validation/configs/parisfog_2011.toml (the `column` run type,
# case "parisfog_2011"). This script reads the runs' output/results.toml
# (the fog diagnostics every fog.dt_out) and writes SVG time series to
# examples/validation/output/.
#
#   julia --project -m Aeolus run examples/validation/configs/parisfog_2011.toml
#   julia --project examples/validation/parisfog_2011_radiation_fog.jl [runs-root]
# =========================================================================

using Printf
include(joinpath(@__DIR__, "read_runs.jl"))
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

labels = Dict("reference" => "reference", "no_feedback" => "no feedback", "no_sun" => "no sun")
cols = [(labels[l], r.res) for (l, r) in member_runs(runs_root(), validation_config("parisfog_2011"))]
series(c, k) = Float64.(c[k])

sunrise = 24 + 6 + 59 / 60
println("ParisFog 2011 — observed: fog at ground 02:30 UTC, max LWP ≈ 70 g/m² at 07:30,")
println("top ≤ 300 m, ground clear ≈ 10:00 UTC (sunrise 06:59 UTC).")
for (name, c) in cols
    hh(x) = isnan(x) ? "   —   " : @sprintf("%02d:%02d", floor(Int, mod(x, 24)), round(Int, 60 * (x - floor(x))) % 60)
    @printf("%-12s onset %s  ground clear %s  max LWP %6.1f g/m² at %s  max top %5.0f m\n",
            name, hh(c["onset_utc_hour"]), hh(c["dissipation_utc_hour"]), c["max_lwp"], hh(c["t_max_lwp_utc_hour"]),
            c["max_top"])
end

# Radiative budget of the opaque night-time fog (LWP > 30 g/m²)
H = cols[1][2]
op = findall(i -> H["lwp"][i] > 30 && H["utc_hour"][i] < sunrise, eachindex(H["lwp"]))
if !isempty(op)
    mean(x) = sum(x) / length(x)
    @printf("opaque fog (%d samples): sfc net LW %.1f W/m², net LW at fog top %.1f W/m², fog LW loss %.1f W/m²\n",
            length(op), mean(series(H, "lw_net_sfc")[op]), mean(series(H, "lw_net_top")[op]),
            mean(series(H, "lw_fog")[op]))
    @printf("  fog-top LW cooling %.0f K/day, C_LW %.1f g m⁻² h⁻¹ (Wærsted et al.: 40–70)\n",
            mean(series(H, "lw_top_cooling")[op]), mean(series(H, "C_LW")[op]))
end

outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
t = series(H, "utc_hour") .- 24          # hours of 15 Nov (negative before midnight)
save_multiline(joinpath(outdir, "parisfog_lwp.svg"), t,
               [series(c, "lwp") for (_, c) in cols], [n for (n, _) in cols];
               xlabel = "hours UTC, 15 Nov 2011", ylabel = "fog liquid water path (g/m²)",
               title = "ParisFog 2011: droplet-radiative feedback")
save_multiline(joinpath(outdir, "parisfog_fog_top.svg"), t,
               [series(c, "fog_top") for (_, c) in cols], [n for (n, _) in cols];
               xlabel = "hours UTC, 15 Nov 2011", ylabel = "fog top height (m)",
               title = "ParisFog 2011: fog top")
save_multiline(joinpath(outdir, "parisfog_surface_lw.svg"), t,
               [series(c, "lw_net_sfc") for (_, c) in cols], [n for (n, _) in cols];
               xlabel = "hours UTC, 15 Nov 2011", ylabel = "net longwave at the ground (W/m²)",
               title = "ParisFog 2011: surface net longwave")
println("wrote ", outdir, "/parisfog_{lwp,fog_top,surface_lw}.svg")
