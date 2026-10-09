# =========================================================================
# GABLS3 — the Cabauw low-level-jet night of 1-2 July 2006 (Basu 2008 case
# description; Bosveld et al. 2014a,b; LES results: Basu, Holtslag &
# Bosveld 2012), P58-d (#194)
#
# The runs are the sweep examples/sbl/configs/gabls3_closures.toml (the
# `sbl` run type; one run folder per closure): time-height geostrophic
# wind and advective tendencies from LargeScaleForcing, the prescribed
# 0.25 m θ as a SurfaceTemperatureSeries, MOST surface fluxes
# (SurfaceExchange) and stress (WallModels.MOSTWallLaw), Smagorinsky
# C_s = 0.1 with the P55-e stability correction (`brown94`) or without it
# (`none`), 00-04 UTC (the published LES envelope is for 03-04 UTC), the
# profiles averaged from 03 UTC. Dry (see the theory page).
#
# This script only reads the run folders: it compares the jet with the
# digitised ensemble (examples/sbl/data/gabls3_reference.csv,
# gabls3_envelope.csv) and makes the figure.
#
#   julia --project -m Aeolus run examples/sbl/configs/gabls3_closures.toml    # 6.25 m on 400 m
#   julia --project examples/sbl/gabls3.jl runs/gabls3_*
#
# writes examples/sbl/output/gabls3_<tag>.svg, _profiles.csv, _series.csv
# =========================================================================

include(joinpath(@__DIR__, "runs.jl"))
include(joinpath(@__DIR__, "svgplot.jl"))
using Printf
using Statistics: mean

const RUNS = sbl_runs(ARGS, "julia --project examples/sbl/gabls3.jl <run folder>…")
const NX, NZ = cfgval(RUNS[1], "grid.size")[[1, 3]]
const LX, LZ = cfgval(RUNS[1], "grid.extent")[[1, 3]]
const T_END = cfgval(RUNS[1], "stepper.end_time")
const TAG = "nx$(NX)_nz$(NZ)_L$(round(Int, LX))x$(round(Int, LZ))"
const OUTDIR = mkpath(SBL_OUT)
const T_AVG = (cfgval(RUNS[1], "averaging.from"), T_END)

results = map(RUNS) do r
    st = r.stats; stab = cfgval(r, "sgs.stability")
    series = [(t = s.t, u_star = s.u_star, wθ = s.wθ, H = s.H, h = s.h, S_max = s.S_max, z_jet = s.z_jet,
               θ2 = s.θ_screen, U10 = s.U_screen, Ts = s.T_s) for s in r.series]      # every 300 s
    @printf("%-8s %s: 03-%02d UTC h = %5.1f m  u* = %.3f m/s  H = %6.1f W/m²  L = %5.1f m  h/L = %4.1f  LLJ %.2f m/s at %3.0f m  (%d steps, %.0f s)\n",
            stab, TAG, round(Int, T_END / 3600), st.h, st.u_star, st.H0, st.L, st.h / st.L, st.S_max, st.z_jet,
            st.nsteps, st.wall_time)
    (stab = stab, st = st, series = series)
end

# ---- jet against the digitised LES ensemble (03-04 UTC) ---------------------
ref = sbl_read_csv(joinpath(SBL_DATA, "gabls3_reference.csv"))
env = sbl_read_csv(joinpath(SBL_DATA, "gabls3_envelope.csv"))
col(name) = [r[findfirst(==(name), ref.columns)] for r in ref.rows]
zr = col("z_m")
jetlo, jethi = sbl_csv_value(env, "jet_height", "low"), sbl_csv_value(env, "jet_height", "high")
spdlo, spdhi = sbl_csv_value(env, "jet_speed", "low"), sbl_csv_value(env, "jet_speed", "high")
function jet_refined(z, S)
    k = clamp(argmax(S), 2, length(S) - 1)
    a, b, d = S[k-1], S[k], S[k+1]
    δ = (a - d) / (2 * (a - 2b + d))
    return z[k] + δ * (z[2] - z[1]), b - (a - d) * δ / 4
end
println("\n03-04 UTC jet against the LES ensemble (Basu et al. 2012 Fig. 1, digitised): height ∈ [$jetlo, $jethi] m, speed ∈ [$spdlo, $spdhi] m/s")
for r in results
    zj, Sj = jet_refined(r.st.z, r.st.S)
    @printf("  %-8s jet %.2f m/s at %.1f m: height %s (margin %+.1f m), speed %s\n", r.stab, Sj, zj,
            jetlo <= zj <= jethi ? "inside" : "OUTSIDE", min(zj - jetlo, jethi - zj),
            spdlo <= Sj <= spdhi ? "inside" : "OUTSIDE")
    s34 = [s for s in r.series if T_AVG[1] <= s.t <= T_AVG[2]]
    @printf("           03-04 UTC means: θ(2 m) = %.2f K (θ_0.25 = %.2f K), U(10 m) = %.2f m/s\n",
            mean(s.θ2 for s in s34), mean(s.Ts for s in s34), mean(s.U10 for s in s34))
end

# ---- CSV output ---------------------------------------------------------------
open(joinpath(OUTDIR, "gabls3_$(TAG)_profiles.csv"), "w") do io
    println(io, "# GABLS3 03-04 UTC mean profiles, Aeolus.jl sbl run type, $TAG (P58-d, #194). Model output, not reference data.")
    println(io, "closure,z_m,U,V,speed,theta")
    for r in results, k in eachindex(r.st.z)
        @printf(io, "%s,%.3f,%.4f,%.4f,%.4f,%.4f\n", r.stab, r.st.z[k], r.st.U[k], r.st.V[k], r.st.S[k], r.st.θ[k])
    end
end
open(joinpath(OUTDIR, "gabls3_$(TAG)_series.csv"), "w") do io
    println(io, "# GABLS3 time series (every 300 s), Aeolus.jl sbl run type, $TAG. Model output.")
    println(io, "closure,t_s,u_star,wtheta,H_Wm2,h_m,S_max,z_jet,theta_2m,U_10m,theta_0p25")
    for r in results, s in r.series
        @printf(io, "%s,%.0f,%.4f,%.5f,%.2f,%.1f,%.3f,%.1f,%.3f,%.3f,%.3f\n", r.stab, s.t, s.u_star, s.wθ,
                s.H, s.h, s.S_max, s.z_jet, s.θ2, s.U10, s.Ts)
    end
end

# ---- the figure ---------------------------------------------------------------
fig = svg_figure(1180, 760, "GABLS3, $(NX)×$(NX)×$(NZ) (Δx = $(LX/NX) m, Δz = $(round(LZ/NZ; digits = 2)) m): Smagorinsky C_s = 0.1 ± stability correction")
leg = [(string(r.stab), i, "") for (i, r) in enumerate(results)]
ylim = (0.0, 300.0); yt = 0:50:300
svg_panel(fig, 80, 50, 300, 280; xlim = (-9.0, 0.0), ylim = ylim, xticks = -9:3:0, yticks = yt,
          xlabel = "U (m/s), 03-04 UTC", ylabel = "z (m)", tag = "(a)",
          bands = [(col("U_p10"), col("U_p90"), zr)], curves = vcat([(col("U_p50"), zr, 8, "5,3")],
          [(r.st.U, r.st.z, i) for (i, r) in enumerate(results)]), legend = leg)
svg_panel(fig, 460, 50, 300, 280; xlim = (0.0, 8.0), ylim = ylim, xticks = 0:2:8, yticks = yt,
          xlabel = "V (m/s), 03-04 UTC", tag = "(b)",
          bands = [(col("V_p10"), col("V_p90"), zr)], curves = vcat([(col("V_p50"), zr, 8, "5,3")],
          [(r.st.V, r.st.z, i) for (i, r) in enumerate(results)]))
svg_panel(fig, 840, 50, 300, 280; xlim = (286.0, 300.0), ylim = ylim, xticks = 286:4:298, yticks = yt,
          xlabel = "θ (K), 03-04 UTC", tag = "(c)", curves = [(r.st.θ, r.st.z, i) for (i, r) in enumerate(results)])
th(r, f) = ([s.t / 3600 for s in r.series], [f(s) for s in r.series])
xl = (0.0, T_END / 3600); xt = 0:1:round(Int, T_END / 3600)
svg_panel(fig, 80, 420, 300, 260; xlim = xl, ylim = (0.0, 0.4), xticks = xt, yticks = 0:0.1:0.4,
          xlabel = "time (h UTC)", ylabel = "u* (m/s)", tag = "(d)",
          curves = [(th(r, s -> s.u_star)..., i) for (i, r) in enumerate(results)])
svg_panel(fig, 460, 420, 300, 260; xlim = xl, ylim = (-60.0, 10.0), xticks = xt, yticks = -60:10:10,
          xlabel = "time (h UTC)", ylabel = "H (W/m²)", tag = "(e)",
          curves = [(th(r, s -> s.H)..., i) for (i, r) in enumerate(results)])
svg_panel(fig, 840, 420, 300, 260; xlim = xl, ylim = (0.0, 300.0), xticks = xt, yticks = 0:50:300,
          xlabel = "time (h UTC)", ylabel = "jet height (m)", tag = "(f)",
          hbands = [(jetlo, jethi)], curves = [(th(r, s -> s.z_jet)..., i) for (i, r) in enumerate(results)])
svg_text(fig, 80, 735, "(a, b) grey: 10-90th percentile, dashed: median of the 6.25 m LES ensemble at 03-04 UTC (Basu, Holtslag & Bosveld 2012, Fig. 1, digitised). (f) grey: jet-height envelope 100-130 m.", size = 11)
println("wrote ", svg_write(fig, joinpath(OUTDIR, "gabls3_$(TAG).svg")))
