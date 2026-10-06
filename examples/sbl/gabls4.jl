# =========================================================================
# GABLS4 stage 3, experiment 3 — the very stable boundary layer over the
# Antarctic Plateau (Dome C, 11 Dec 2009; Couvreux et al. 2020, BLM 176,
# 369-400), P58-d (#194)
#
# `gabls4_case` (examples/sbl/cases.jl): from 10 UTC, the experiment-2
# ensemble-mean profiles, U_g = (1.25, 4.5) m/s, f = 2Ω sin(-75.1°), the
# official case file's surface temperature (θ at 651 hPa), z0m = 1e-3 m,
# z0h = 1e-4 m, β_m = 4.8, β_h = 7.8. Statistics of 16:30-17:30 UTC are
# compared with Table 4, experiment 3: the envelope of the six LES that ran
# at 1 m isotropic. The seventh, CSIRO-LES, ran at Δx = 3.9 m, Δz = 1.17 m
# and is printed separately; so is the experiment-2 envelope of the four
# Table 2 E2 runs at 5 m / 2 m (see examples/sbl/data/gabls4_reference.csv).
#
# The six 1 m runs used 256 × 256 × 80 m to 512 × 512 × 150 m (up to 39 M
# cells); the default here is 2 m on 128 × 128 × 96 m (64 × 64 × 48), the
# vertical resolution of experiment 2: Δz is coarser than in every
# experiment-3 reference run, CSIRO's included.
#
#   julia --project=. examples/sbl/gabls4.jl                       # 2 m
#   NX=64 NZ=64 LX=64 L=64 julia --project=. examples/sbl/gabls4.jl  # 1 m on a 64 m cube
#   CLOSURES=brown94 ... ; REUSE=1 ...
#
# writes examples/sbl/output/gabls4_<tag>.svg, _series.csv, _profiles.csv
# =========================================================================

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "svgplot.jl"))
using Printf
using Serialization

const NX = parse(Int, get(ENV, "NX", "64"))
const NZ = parse(Int, get(ENV, "NZ", "48"))
const LZ = parse(Float64, get(ENV, "L", "96"))
const LX = parse(Float64, get(ENV, "LX", "128"))
const T_END = parse(Float64, get(ENV, "T_END", string(17.5 * 3600)))
const T_AVG = 16.5 * 3600
const CLOSURES = Symbol.(split(get(ENV, "CLOSURES", "brown94,none"), ","))
const TAG = "nx$(NX)_nz$(NZ)_L$(round(Int, LX))x$(round(Int, LZ))"
const OUTDIR = joinpath(@__DIR__, "output"); mkpath(OUTDIR)

function run_closure(stab, ck = nothing)
    c = gabls4_case(; N = (NX, NX, NZ), extent = (LX, LX, LZ), sgs_stability = stab)
    series = NamedTuple[]
    nextt = Ref(c.sim.t)
    cb = function (c)
        c.sim.t >= nextt[] || return
        nextt[] += 300.0
        p = sbl_profiles(c); k = argmax(p.S)
        scr = sbl_screen(c; z_θ = 2.0, z_U = 10.0)
        push!(series, (t = c.sim.t, u_star = sqrt(hypot(c.τx, c.τy)), H = c.ρcp * mean(c.wθ),
                       h = sbl_bl_height(p.zf, p.τ), S_max = p.S[k], z_jet = p.z[k],
                       θ2 = scr.θ_screen, U10 = scr.U_screen,
                       θs = surface_temperature(c.Ts, c.sim.t)))
    end
    st = sbl_run!(c, T_END; average_from = T_AVG, sample_every = 10, callback = cb,
                  checkpoint = ck, checkpoint_components = (series = series, nextt = nextt))
    θst = mean(-s.H / c.ρcp / s.u_star for s in series if s.t >= T_AVG)
    @printf("%-8s %s: 16:30-17:30 UTC h = %5.1f m  L = %5.1f m  u* = %.3f m/s  H = %6.1f W/m²  θ* = %.3f K  LLJ %.2f m/s at %4.1f m  (%d steps, %.0f s)\n",
            stab, TAG, st.h, st.L, st.u_star, st.H0, θst, st.S_max, st.z_jet, c.nsteps, c.wall_time)
    return (stab = stab, st = st, θst = θst, series = series, nsteps = c.nsteps, wall = c.wall_time)
end

function cached(stab)
    path = joinpath(OUTDIR, "gabls4_$(TAG)_$(stab).jls")
    if get(ENV, "REUSE", "0") == "1" && isfile(path)
        println("$stab: cached $path")
        return deserialize(path)
    end
    # periodic checkpoints, automatic resume (#245); made before the case
    ck = sbl_checkpoints("gabls4_$(TAG)_$(stab)")
    r = run_closure(stab, ck)
    serialize(path, r)
    finish_checkpoints!(ck)
    return r
end
results = [cached(s) for s in CLOSURES]

ref = gabls4_reference()
row(kind) = only(r for r in ref.rows if r[1] == kind)
lo, hi = row("envelope_low"), row("envelope_high")
lo2, hi2 = row("envelope_exp2_low"), row("envelope_exp2_high")
csiro = only(r for r in ref.rows if r[1] == "exp3" && r[2] == "CSIRO-LES")
inside(x, a, b) = a <= x <= b ? "inside" : "OUTSIDE"
function compare(title, lo, hi)
    println(title)
    @printf("  envelope: h ∈ [%.1f, %.1f] m, L ∈ [%.1f, %.1f] m, u* ∈ [%.2f, %.2f] m/s, H ∈ [%.1f, %.1f] W/m², θ* ∈ [%.2f, %.2f] K\n",
            lo[3], hi[3], lo[4], hi[4], lo[5], hi[5], lo[6], hi[6], lo[7], hi[7])
    for r in results
        @printf("  %-8s h %s, L %s, u* %s, H %s, θ* %s\n", r.stab, inside(r.st.h, lo[3], hi[3]), inside(r.st.L, lo[4], hi[4]),
                inside(r.st.u_star, lo[5], hi[5]), inside(r.st.H0, lo[6], hi[6]), inside(r.θst, lo[7], hi[7]))
    end
end
compare("\n17 UTC against Couvreux et al. (2020) Table 4, experiment 3, the six LES at Δx = Δz = 1 m:", lo, hi)
@printf("  CSIRO-LES (exp. 3, Δx = %.1f m, Δz = %.2f m): h = %.1f m, L = %.1f m, u* = %.2f m/s, H = %.1f W/m², θ* = %.2f K\n",
        csiro[9], csiro[10], csiro[3], csiro[4], csiro[5], csiro[6], csiro[7])
compare("17 UTC against Table 4, experiment 2, the four E2 runs at Δx = 5 m, Δz = 2 m (MesoNH, PALM, UConn, MONC):", lo2, hi2)

open(joinpath(OUTDIR, "gabls4_$(TAG)_series.csv"), "w") do io
    println(io, "# GABLS4 experiment 3 time series (every 300 s), Aeolus.jl SBL driver, $TAG (P58-d, #194). Model output.")
    println(io, "closure,t_s,u_star,H_Wm2,h_m,S_max,z_jet,theta_2m,U_10m,theta_s")
    for r in results, s in r.series
        @printf(io, "%s,%.0f,%.4f,%.2f,%.2f,%.3f,%.2f,%.3f,%.3f,%.3f\n", r.stab, s.t, s.u_star, s.H, s.h,
                s.S_max, s.z_jet, s.θ2, s.U10, s.θs)
    end
end
open(joinpath(OUTDIR, "gabls4_$(TAG)_profiles.csv"), "w") do io
    println(io, "# GABLS4 experiment 3, 16:30-17:30 UTC mean profiles, Aeolus.jl SBL driver, $TAG. Model output.")
    println(io, "closure,z_m,U,V,speed,theta")
    for r in results, k in eachindex(r.st.z)
        @printf(io, "%s,%.3f,%.4f,%.4f,%.4f,%.4f\n", r.stab, r.st.z[k], r.st.U[k], r.st.V[k], r.st.S[k], r.st.θ[k])
    end
end

fig = svg_figure(1180, 760, "GABLS4 exp. 3, $(NX)×$(NX)×$(NZ) (Δx = $(LX/NX) m, Δz = $(LZ/NZ) m): Smagorinsky C_s = 0.1 ± stability correction")
leg = [(string(r.stab), i, "") for (i, r) in enumerate(results)]
zmax = min(LZ, 80.0)
svg_panel(fig, 80, 50, 300, 280; xlim = (0.0, 8.0), ylim = (0.0, zmax), xticks = 0:2:8, yticks = 0:20:80,
          xlabel = "wind speed (m/s), 16:30-17:30 UTC", ylabel = "z (m)", tag = "(a)",
          hbands = [(lo[3], hi[3])], curves = [(r.st.S, r.st.z, i) for (i, r) in enumerate(results)], legend = leg)
svg_panel(fig, 460, 50, 300, 280; xlim = (264.0, 280.0), ylim = (0.0, zmax), xticks = 264:4:280, yticks = 0:20:80,
          xlabel = "θ (K), 16:30-17:30 UTC", tag = "(b)", hbands = [(lo[3], hi[3])],
          curves = [(r.st.θ, r.st.z, i) for (i, r) in enumerate(results)])
th(r, f) = ([s.t / 3600 for s in r.series], [f(s) for s in r.series])
xl = (10.0, T_END / 3600); xt = 10:1:floor(Int, T_END / 3600)
svg_panel(fig, 840, 50, 300, 280; xlim = xl, ylim = (0.0, 100.0), xticks = xt, yticks = 0:20:100,
          xlabel = "time (h UTC)", ylabel = "h (m)", tag = "(c)", hbands = [(lo[3], hi[3])],
          curves = [(th(r, s -> s.h)..., i) for (i, r) in enumerate(results)])
svg_panel(fig, 80, 420, 300, 260; xlim = xl, ylim = (0.0, 0.3), xticks = xt, yticks = 0:0.1:0.3,
          xlabel = "time (h UTC)", ylabel = "u* (m/s)", tag = "(d)", hbands = [(lo[5], hi[5])],
          curves = [(th(r, s -> s.u_star)..., i) for (i, r) in enumerate(results)])
svg_panel(fig, 460, 420, 300, 260; xlim = xl, ylim = (-30.0, 20.0), xticks = xt, yticks = -30:10:20,
          xlabel = "time (h UTC)", ylabel = "H (W/m²)", tag = "(e)", hbands = [(lo[6], hi[6])],
          curves = [(th(r, s -> s.H)..., i) for (i, r) in enumerate(results)])
svg_panel(fig, 840, 420, 300, 260; xlim = xl, ylim = (260.0, 285.0), xticks = xt, yticks = 260:5:285,
          xlabel = "time (h UTC)", ylabel = "θ(2 m) (K)", tag = "(f)",
          curves = vcat([(th(r, s -> s.θ2)..., i) for (i, r) in enumerate(results)],
                        [(th(results[1], s -> s.θs)..., 8, "5,3")]),
          legend = [("θ_s (prescribed)", 8, "5,3")])
svg_text(fig, 80, 735, "grey: range of the six 1 m LES of Couvreux et al. (2020) Table 4, experiment 3, 17 UTC (CSIRO-LES, 3.9 m / 1.17 m, excluded); in (a)-(c) the range of h.", size = 11)
println("wrote ", svg_write(fig, joinpath(OUTDIR, "gabls4_$(TAG).svg")))
