# =========================================================================
# GABLS2 / CASES-99 — two diurnal cycles over Kansas (22-24 Oct 1999;
# Svensson et al. 2011; Poulos et al. 2002; LES: Kumar et al. 2010) and the
# CASES-99 "hockey stick" of Sun et al. (2012), P58-d (#194)
#
# The runs are the sweep examples/sbl/configs/cases99_closures.toml
# (cases99.toml: gabls2.toml with the towers; the `sbl` run type), one run
# folder per closure: from 16 LT 22 October for 48 h, U_g = (3, -9) m/s,
# the prescribed surface temperature, z0m = 0.03 m, z0h = 0.003 m; dry
# (see the CSV header); the case's large-scale subsidence on θ from 40 h
# (16 LT 23 Oct, LargeScaleForcing.Subsidence, #248; `--set
# 'forcing.subsidence={}'` for the runs without it). The residual-layer
# inversion height (sbl_inversion_height) is recorded with the series
# (output/series.csv, every 600 s). Virtual towers in every column record
# 5-min means and variances below 60 m during the two nights (18-06 LT, as
# Sun et al. 2012); V_TKE (Transition's turbulence_velocity_scale,
# resolved + SGS energy) is binned against the wind speed per level
# (output/towers.csv).
#
# This script only reads the run folders: it compares with the SCM ranges
# and the screen-level observations, fits the hockey-stick threshold V_s
# per level to the tower bins of at least 20 samples and compares it with
# Sun et al. (2012) Fig. 3a, and makes the figures.
#
#   julia --project -m Aeolus run examples/sbl/configs/cases99_closures.toml     # 64 x 64 x 128 (37.5 m / 12.5 m)
#   julia --project examples/sbl/gabls2.jl runs/cases99_*
#
# writes examples/sbl/output/gabls2_<tag>.svg, cases99_hockey_<tag>.svg,
# gabls2_<tag>_series.csv, cases99_hockey_<tag>.csv
# =========================================================================

include(joinpath(@__DIR__, "runs.jl"))
include(joinpath(@__DIR__, "svgplot.jl"))
using Aeolus.StableBoundaryLayer: sbl_hockey_threshold      # a fit to the tower bins (no model)
using Printf

const RUNS = sbl_runs(ARGS, "julia --project examples/sbl/gabls2.jl <run folder>…")
const NX, NZ = cfgval(RUNS[1], "grid.size")[[1, 3]]
const LX, LZ = cfgval(RUNS[1], "grid.extent")[[1, 3]]
const T0 = cfgval(RUNS[1], "initial.t_start")
const T_END = cfgval(RUNS[1], "stepper.end_time")
const SUBSIDENCE = !isempty(cfgval(RUNS[1], "forcing.subsidence"))
const TAG = "nx$(NX)_nz$(NZ)_L$(round(Int, LX))x$(round(Int, LZ))" * (SUBSIDENCE ? "" : "_nosubs")
const OUTDIR = mkpath(SBL_OUT)
const P_SFC = cfgval(RUNS[1], "surface.temperature")["p_sfc"]
const EXPONENT = cfgval(RUNS[1], "surface.temperature")["exponent"]

"The bins of tower level `z` with at least `min_count` samples (V, V_TKE, σ, n)."
function tower_bins(rows, z; min_count = 20)
    b = [r for r in rows if r.z == z && r.n >= min_count]
    return (V = [r.V for r in b], V_TKE = [r.V_TKE for r in b], σ = [r.sigma for r in b], n = [Int(r.n) for r in b])
end

results = map(RUNS) do r
    stab = cfgval(r, "sgs.stability")
    # the series rows in the script's units: 2 m and surface temperatures in °C
    toC(θ) = θ * (P_SFC / 1e5)^EXPONENT - 273.15
    series = [(t = s.t, u_star = s.u_star, H = s.H, h = s.h, S_max = s.S_max, z_jet = s.z_jet, z_inv = s.z_inv,
               ls_heat = s.ls_heat, T2 = toC(s.θ_screen), U10 = s.U_screen, Ts = toC(s.T_s)) for s in r.series]
    zs = unique([t.z for t in r.towers])
    @printf("%-8s %s: %d steps, %.0f s, %d tower samples\n", stab, TAG, r.stats.nsteps, r.stats.wall_time,
            isempty(r.towers) ? 0 : round(Int, sum(t.n for t in r.towers)))
    (stab = stab, series = series, towers = (z = zs, rows = r.towers))
end

# ---- GABLS2 against the SCM ranges and the screen-level values ---------------
ref = read_table(joinpath(SBL_DATA, "gabls2_reference.csv"))
at(r, h) = r.series[argmin([abs(s.t / 3600 - h) for s in r.series])]
println("\nGABLS2 at the end of the first night (30 h = 06 LT 23 Oct) against the SCM range of Holtslag et al. (2007):")
for r in results
    s = at(r, 30)
    rng(k) = (table_value(ref, k, "low"), table_value(ref, k, "high"))
    f(x, (a, b)) = @sprintf("%.3g %s [%g, %g]", x, a <= x <= b ? "inside" : "OUTSIDE", a, b)
    println("  $(r.stab): H = ", f(s.H, rng("scm_heat_flux")), " W/m², u* = ", f(s.u_star, rng("scm_u_star")),
            " m/s, h = ", f(s.h, rng("scm_h")), " m")
end
println("Screen level against the observations and Kumar et al. (2010) LES (GABLS overview Figs. 3-4, eyeball digitised):")
for row in ref.rows
    q = row[1]; (startswith(q, "obs_") || startswith(q, "les_")) || continue
    h = row[2]
    vals = join((@sprintf("%s %.2f", r.stab, q[5:end] == "T2m" ? at(r, h).T2 : at(r, h).U10) for r in results), ", ")
    @printf("  %-8s at %2.0f h: reference [%.1f, %.1f] %s; model %s\n", q, h, row[3], row[4], row[5], vals)
end

println("\nResidual-layer inversion (largest ∂⟨θ⟩/∂z above 300 m) and the subsidence heating ∫∫(∂θ/∂t)_s dz dt",
        SUBSIDENCE ? "" : " (SUBSIDENCE=0: none)", ":")
for r in results, h in (40, 42, 48, 54)
    r.series[end].t / 3600 >= h - 0.1 || continue
    s = at(r, h)
    @printf("  %-8s at %2.0f h: z_inv = %.1f m, h = %.1f m, subsidence heat %.2f K m, jet %.2f m/s at %.0f m\n",
            r.stab, h, s.z_inv, s.h, s.ls_heat, s.S_max, s.z_jet)
end

open(joinpath(OUTDIR, "gabls2_$(TAG)_series.csv"), "w") do io
    println(io, "# GABLS2 time series (every 600 s), Aeolus.jl sbl run type, $TAG (P58-d, #194). Model output. t in s from 00 LT 22 Oct 1999.")
    println(io, "closure,t_s,u_star,H_Wm2,h_m,S_max,z_jet,T_2m_C,U_10m,T_s_C,z_inv_m,subsidence_heat_Km")
    for r in results, s in r.series
        @printf(io, "%s,%.0f,%.4f,%.2f,%.1f,%.3f,%.1f,%.3f,%.3f,%.3f,%.1f,%.4f\n", r.stab, s.t, s.u_star, s.H, s.h,
                s.S_max, s.z_jet, s.T2, s.U10, s.Ts, s.z_inv, s.ls_heat)
    end
end

fig = svg_figure(1180, 760, "GABLS2 / CASES-99, $(NX)×$(NX)×$(NZ) (Δx = $(LX/NX) m, Δz = $(LZ/NZ) m), 16 LT 22 Oct - 16 LT 24 Oct 1999")
leg = [(string(r.stab), i, "") for (i, r) in enumerate(results)]
th(r, f) = ([s.t / 3600 for s in r.series], [f(s) for s in r.series])
xl = (T0 / 3600, T_END / 3600); xt = 16:8:64
pts(q) = [((row[3] + row[4]) / 2) for row in ref.rows if row[1] == q], [row[2] for row in ref.rows if row[1] == q]
obsT = pts("obs_T2m"); lesT = pts("les_T2m"); obsU = pts("obs_U10"); lesU = pts("les_U10")
svg_panel(fig, 80, 50, 300, 280; xlim = xl, ylim = (-5.0, 20.0), xticks = xt, yticks = -5:5:20,
          xlabel = "local time (h from 00 LT 22 Oct)", ylabel = "2 m temperature (°C)", tag = "(a)",
          curves = vcat([(th(r, s -> s.T2)..., i) for (i, r) in enumerate(results)],
                        [(th(results[1], s -> s.Ts)..., 8, "5,3")]),
          points = [(obsT[2], obsT[1], 8, :circle), (lesT[2], lesT[1], 4, :square)], legend = leg)
svg_panel(fig, 460, 50, 300, 280; xlim = xl, ylim = (0.0, 10.0), xticks = xt, yticks = 0:2:10,
          xlabel = "local time (h)", ylabel = "10 m wind (m/s)", tag = "(b)",
          curves = [(th(r, s -> s.U10)..., i) for (i, r) in enumerate(results)],
          points = [(obsU[2], obsU[1], 8, :circle), (lesU[2], lesU[1], 4, :square)])
svg_panel(fig, 840, 50, 300, 280; xlim = xl, ylim = (0.0, 0.6), xticks = xt, yticks = 0:0.1:0.6,
          xlabel = "local time (h)", ylabel = "u* (m/s)", tag = "(c)",
          curves = [(th(r, s -> s.u_star)..., i) for (i, r) in enumerate(results)],
          points = [([30.0, 30.0], [table_value(ref, "scm_u_star", "low"), table_value(ref, "scm_u_star", "high")], 8, :square)])
svg_panel(fig, 80, 420, 300, 260; xlim = xl, ylim = (-100.0, 250.0), xticks = xt, yticks = -100:50:250,
          xlabel = "local time (h)", ylabel = "H (W/m²)", tag = "(d)",
          curves = [(th(r, s -> s.H)..., i) for (i, r) in enumerate(results)],
          points = [([30.0, 30.0], [table_value(ref, "scm_heat_flux", "low"), table_value(ref, "scm_heat_flux", "high")], 8, :square)])
svg_panel(fig, 460, 420, 300, 260; xlim = xl, ylim = (0.0, 1600.0), xticks = xt, yticks = 0:400:1600,
          xlabel = "local time (h)", ylabel = "h (m, 5 % stress)", tag = "(e)",
          curves = [(th(r, s -> s.h)..., i) for (i, r) in enumerate(results)],
          points = [([30.0, 30.0], [table_value(ref, "scm_h", "low"), table_value(ref, "scm_h", "high")], 8, :square)])
svg_panel(fig, 840, 420, 300, 260; xlim = xl, ylim = (0.0, 16.0), xticks = xt, yticks = 0:4:16,
          xlabel = "local time (h)", ylabel = "jet maximum (m/s)", tag = "(f)",
          curves = [(th(r, s -> s.S_max)..., i) for (i, r) in enumerate(results)])
svg_text(fig, 80, 728, "(a) dashed: prescribed surface temperature; (a, b) circles: CASES-99 observations, squares: Kumar et al. (2010) LES (GABLS overview Figs. 3-4, eyeball digitised, ±0.7 K, ±0.4 m/s).", size = 11)
svg_text(fig, 80, 744, "(c-e) squares: ends of the SCM range at 06 LT 23 Oct (30 h), Holtslag et al. (2007).", size = 11)
println("wrote ", svg_write(fig, joinpath(OUTDIR, "gabls2_$(TAG).svg")))

# ---- the CASES-99 hockey stick --------------------------------------------------
sun = let t = read_table(joinpath(SBL_DATA, "cases99_sun2012_threshold.csv"))
    (z = [r[1] for r in t.rows], V_s = [r[2] for r in t.rows], header = t.header)
end
logfit = let A = hcat(ones(length(sun.z)), log.(sun.z))   # V_s = a + b ln z (Sun et al. 2012 Fig. 3a, dashed)
    A \ sun.V_s
end
Vs_sun(z) = logfit[1] + logfit[2] * log(z)
println("\nCASES-99 hockey stick (nights 18-06 LT, 5-min tower segments): threshold V_s per level")
println("  Sun et al. (2012) log fit to Fig. 3a: V_s = $(round(logfit[1]; digits = 2)) + $(round(logfit[2]; digits = 2)) ln z")
hock = []
open(joinpath(OUTDIR, "cases99_hockey_$(TAG).csv"), "w") do io
    println(io, "# CASES-99 hockey stick from the GABLS2 LES, $TAG: bin means of V_TKE against V per tower level. Model output.")
    println(io, "closure,z_m,V_bin_mean,V_TKE_mean,V_TKE_std,n")
    for r in results, (l, z) in enumerate(r.towers.z)
        b = tower_bins(r.towers.rows, z)
        for i in eachindex(b.V)
            @printf(io, "%s,%.2f,%.3f,%.4f,%.4f,%d\n", r.stab, z, b.V[i], b.V_TKE[i], b.σ[i], b.n[i])
        end
        if length(b.V) >= 4
            h = sbl_hockey_threshold(b.V, b.V_TKE; weights = Float64.(b.n))
            push!(hock, (stab = r.stab, z = z, h = h, bins = b))
            shape = (h.slope2 > h.slope1 && h.slope2 > 0 && h.slope1 >= 0) ? "hockey stick" : "no hockey-stick shape"
            @printf("  %-8s z = %5.1f m: V_s = %.2f m/s (Sun log fit %.2f m/s), slopes %.3f / %.3f (Sun regime 2: ~0.25): %s; V range %.1f-%.1f m/s, %d samples\n",
                    r.stab, z, h.V_s, Vs_sun(z), h.slope1, h.slope2, shape, minimum(b.V), maximum(b.V), sum(b.n))
        end
    end
end

fig = svg_figure(1180, 470, "CASES-99 hockey stick from the GABLS2 LES ($(NX)×$(NX)×$(NZ)), nights 18-06 LT, 5-min tower segments")
for (pi, r) in enumerate(results)
    hs = [h for h in hock if h.stab == r.stab]
    curves = []
    for (i, h) in enumerate(hs)
        push!(curves, (h.bins.V, h.bins.V_TKE, i))
        V = collect(range(minimum(h.bins.V), maximum(h.bins.V); length = 50))
        push!(curves, (V, [h.h.intercept + h.h.slope1 * min(v, h.h.V_s) + h.h.slope2 * max(v - h.h.V_s, 0) for v in V], i, "4,3"))
    end
    svg_panel(fig, 80 + (pi - 1) * 380, 50, 300, 320; xlim = (0.0, 14.0), ylim = (0.0, 2.0), xticks = 0:2:14,
              yticks = 0:0.5:2, xlabel = "wind speed V (m/s)", ylabel = "V_TKE (m/s)", tag = "($('a' + pi - 1)) $(r.stab)",
              curves = curves, legend = [(@sprintf("z = %.1f m", h.z), i, "") for (i, h) in enumerate(hs)])
end
zz = collect(0.5:0.5:60)
svg_panel(fig, 840, 50, 300, 320; xlim = (0.0, 12.0), ylim = (0.0, 60.0), xticks = 0:2:12, yticks = 0:10:60,
          xlabel = "threshold V_s (m/s)", ylabel = "z (m)", tag = "(c)",
          curves = vcat([(Vs_sun.(zz), zz, 8, "5,3")],
                        [([h.h.V_s for h in hock if h.stab == r.stab], [h.z for h in hock if h.stab == r.stab], i) for (i, r) in enumerate(results)]),
          points = vcat([(sun.V_s, sun.z, 8, :circle)],
                        [([h.h.V_s for h in hock if h.stab == r.stab], [h.z for h in hock if h.stab == r.stab], i, :square) for (i, r) in enumerate(results)]),
          legend = vcat([("Sun et al. 2012", 8, "5,3")], [(string(r.stab), i, "") for (i, r) in enumerate(results)]),
          legend_x = 40)
svg_text(fig, 80, 440, "(a, b) solid: bin means of V_TKE = √e (resolved + SGS) per tower level; dashed: the fitted two-segment line. (c) circles: Sun et al. (2012) Fig. 3a (digitised), dashed: its log fit; squares: fitted V_s.", size = 11)
println("wrote ", svg_write(fig, joinpath(OUTDIR, "cases99_hockey_$(TAG).svg")))
