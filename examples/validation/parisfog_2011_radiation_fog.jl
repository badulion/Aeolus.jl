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
# same with the sun switched off. Writes SVG time series to output/.
#
#   julia --project=. examples/validation/parisfog_2011_radiation_fog.jl
# =========================================================================

using Aeolus
using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const HOURS = 14

function run_case(set)
    col = parisfog_2011_column(; settings = set)
    run!(col, HOURS * 3600)
    return col
end

runs = [("reference",   FogColumnSettings()),
        ("no feedback", FogColumnSettings(droplet_feedback = false)),
        ("no sun",      FogColumnSettings(shortwave = false))]
cols = [(name, run_case(set)) for (name, set) in runs]

sunrise = 24 + 6 + 59 / 60
println("ParisFog 2011 — observed: fog at ground 02:30 UTC, max LWP ≈ 70 g/m² at 07:30,")
println("top ≤ 300 m, ground clear ≈ 10:00 UTC (sunrise 06:59 UTC).")
for (name, col) in cols
    lc = fog_life_cycle(col.history)
    hh(x) = isnan(x) ? "   —   " : @sprintf("%02d:%02d", floor(Int, mod(x, 24)), round(Int, 60 * (x - floor(x))) % 60)
    @printf("%-12s onset %s  ground clear %s  max LWP %6.1f g/m² at %s  max top %5.0f m\n",
            name, hh(lc.onset), hh(lc.dissipation), lc.max_lwp, hh(lc.t_max_lwp), lc.max_top)
end

# Radiative budget of the opaque night-time fog (LWP > 30 g/m²)
H = cols[1][2].history
op = filter(h -> h.lwp > 30 && h.utc_hour < sunrise, H)
if !isempty(op)
    mean(x) = sum(x) / length(x)
    @printf("opaque fog (%d samples): sfc net LW %.1f W/m², net LW at fog top %.1f W/m², fog LW loss %.1f W/m²\n",
            length(op), mean([h.lw_net_sfc for h in op]), mean([h.lw_net_top for h in op]),
            mean([h.lw_fog for h in op]))
    @printf("  fog-top LW cooling %.0f K/day, C_LW %.1f g m⁻² h⁻¹ (Wærsted et al.: 40–70)\n",
            mean([h.lw_top_cooling for h in op]), mean([h.C_LW for h in op]))
end

outdir = joinpath(pwd(), "output"); mkpath(outdir)
t = [h.utc_hour - 24 for h in H]          # hours of 15 Nov (negative before midnight)
save_multiline(joinpath(outdir, "parisfog_lwp.svg"), t,
               [[h.lwp for h in c.history] for (_, c) in cols],
               [n for (n, _) in cols];
               xlabel = "hours UTC, 15 Nov 2011", ylabel = "fog liquid water path (g/m²)",
               title = "ParisFog 2011: droplet-radiative feedback")
save_multiline(joinpath(outdir, "parisfog_fog_top.svg"), t,
               [[h.fog_top for h in c.history] for (_, c) in cols],
               [n for (n, _) in cols];
               xlabel = "hours UTC, 15 Nov 2011", ylabel = "fog top height (m)",
               title = "ParisFog 2011: fog top")
save_multiline(joinpath(outdir, "parisfog_surface_lw.svg"), t,
               [[h.lw_net_sfc for h in c.history] for (_, c) in cols],
               [n for (n, _) in cols];
               xlabel = "hours UTC, 15 Nov 2011", ylabel = "net longwave at the ground (W/m²)",
               title = "ParisFog 2011: surface net longwave")
println("wrote ", outdir, "/parisfog_{lwp,fog_top,surface_lw}.svg")
