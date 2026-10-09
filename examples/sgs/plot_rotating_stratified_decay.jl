# The SGS closures against filtered DNS of rotating and stratified decay
# (configs/rotating_stratified_decay.toml): the tables and the figure.
# Reads output/results.toml only.
#
#   julia --project -m Aeolus run examples/sgs/configs/rotating_stratified_decay.toml
#   julia --project=. examples/sgs/plot_rotating_stratified_decay.jl runs/<date>_rotating_stratified_decay_<id>
#
# Prints the tables quoted in docs/src/theory/sgs_tke.md and writes
# output/rotating_stratified_decay_<N>.svg (resolved KE, SGS energy and the
# Lumley-triangle trajectories of the mean invariants, one row per case).

using Printf
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
res = read_results(folder)
N = recorded_config(folder)["dns"]["n"]
vmean(x) = sum(x) / length(x)
b = res["base"]
base = (T = b["T"],)
@printf("DNS %d³: spin-up %.0f s, Re_λ = %.1f, u′ = %.3f, L = %.3f, T = %.3f, f = %.3f, N = %.3f\n",
        N, b["spinup_wall_seconds"], b["re_lambda"], b["u_prime"], b["L"], b["T"], b["f"], b["N_bv"])
CASES = Tuple(Symbol.(recorded_config(folder)["dns"]["cases"]))
MODELS = Tuple(Symbol.(recorded_config(folder)["les"]["models"]))
LABEL = Dict(:smagorinsky => "Smagorinsky", :dynamic => "dynamic Smag.", :tke => "TKE",
             :rsm => "Reynolds stress", :dns => "filtered DNS")
COLOR = Dict(:dns => "#000000", :smagorinsky => "#8c8c8c", :dynamic => "#4a7c3a",
             :tke => "#1b6ca8", :rsm => "#c8553d")
dns = Dict{Symbol,Any}(); les = Dict{Tuple{Symbol,Symbol},Any}()
for c in CASES
    r = res["cases"][String(c)]
    snaps = [(K = r["dns_K"][i], e_sgs = r["dns_e"][i], II_mean = r["dns_II"][i], III_mean = r["dns_III"][i])
             for i in eachindex(r["dns_K"])]
    dns[c] = (tsave = r["t_over_T"] .* base.T, snaps = snaps)
    @printf("\n%s: DNS continuation %.0f s, K(t₀+2T)/K(t₀) = %.3f (full resolution)\n", c, r["dns_wall_seconds"],
            r["dns_K_ratio"])
    @printf("  %-16s %9s %9s %9s %9s %8s\n", "", "KE err", "e(2T)", "II(2T)", "τ33/τ11", "LES s")
    @printf("  %-16s %9s %9.4f %9.4f %9.3f\n", "filtered DNS", "—", snaps[end].e_sgs, snaps[end].II_mean,
            r["dns_r33_mean"])
    for m in MODELS
        l = r["models"][String(m)]
        les[(c, m)] = (K = l["K"], e = l["e"], II = l["II"], III = l["III"])
        @printf("  %-16s %+8.1f%% %9.4f %9.4f %9.3f %8.1f\n", LABEL[m], l["ke_error_percent"], l["e_end"],
                l["II_end"], l["r33_mean"], l["wall_seconds"])
    end
    if occursin("stratified", String(c)) && all(m -> m in MODELS, (:tke, :rsm))
        @printf("  mean SGS heat flux over (t₀, t₀+2T]: DNS %.3f, TKE %.3f, RSM %.3f\n", r["dns_hw_mean"],
                r["models"]["tke"]["hw_mean"], r["models"]["rsm"]["hw_mean"])
    end
end

# -------------------------------------------------------------------------
# Figure: one row per case — KE(t), e(t), Lumley triangle
# -------------------------------------------------------------------------
# 1, 2 or 5 × 10ⁿ ticks from 0 covering `top`
function nice_ticks(top)
    raw = top / 4
    p = 10.0^floor(log10(raw))
    step = p * (raw / p <= 1 ? 1 : raw / p <= 2 ? 2 : raw / p <= 5 ? 5 : 10)
    top_r = ceil(top / step) * step
    return collect(0:step:top_r + step / 2), top_r
end

function write_figure(path)
    pw, ph = 300, 230                    # panel size
    ml, mt, gap = 70, 40, 60
    W = ml + 3pw + 2gap + 30; H = mt + length(CASES) * (ph + gap) + 40
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif" font-size="12">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    poly(pts, col; w = 1.6, dash = "") =
        println(io, """<polyline fill="none" stroke="$col" stroke-width="$w" $(dash == "" ? "" : "stroke-dasharray=\"$dash\"") points="$(join([@sprintf("%.1f,%.1f", p...) for p in pts], " "))"/>""")
    text(x, y, s; a = "middle", extra = "") = println(io, """<text x="$x" y="$y" text-anchor="$a" $extra>$s</text>""")
    function frame(x0, y0, xl, yl, xt, yt, X, Y, title)
        println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="black"/>""")
        for v in xt
            text(X(v), y0 + ph + 15, @sprintf("%g", v))
        end
        for v in yt
            text(x0 - 5, Y(v) + 4, @sprintf("%g", v); a = "end")
            println(io, """<line x1="$x0" y1="$(Y(v))" x2="$(x0 + pw)" y2="$(Y(v))" stroke="#ececec"/>""")
        end
        text(x0 + pw / 2, y0 + ph + 32, xl)
        text(x0 - 48, y0 + ph / 2, yl; extra = "transform=\"rotate(-90 $(x0 - 48) $(y0 + ph / 2))\"")
        text(x0 + pw / 2, y0 - 8, title; extra = "font-weight=\"bold\"")
    end
    for (row, c) in enumerate(CASES)
        d = dns[c]; y0 = mt + (row - 1) * (ph + gap)
        τt = d.tsave ./ base.T
        # (a) resolved KE
        x0 = ml
        Kt, Kmax = nice_ticks(1.05 * d.snaps[1].K)
        X(t) = x0 + pw * t / 2; Y(K) = y0 + ph * (1 - K / Kmax)
        frame(x0, y0, "t / T", "resolved KE", 0:0.5:2, Kt, X, Y, "$c: resolved KE")
        poly([(X(τt[i]), Y(d.snaps[i].K)) for i in eachindex(τt)], COLOR[:dns]; w = 2.4)
        for m in MODELS
            poly([(X(τt[i]), Y(les[(c, m)].K[i])) for i in eachindex(τt)], COLOR[m])
        end
        # (b) SGS energy
        x0 = ml + pw + gap
        et, emax = nice_ticks(1.05 * max(maximum(les[(c, :tke)].e), maximum(les[(c, :rsm)].e), d.snaps[1].e_sgs))
        Xb(t) = x0 + pw * t / 2; Yb(e) = y0 + ph * (1 - e / emax)
        frame(x0, y0, "t / T", "SGS energy e", 0:0.5:2, et, Xb, Yb, "$c: SGS energy")
        poly([(Xb(τt[i]), Yb(d.snaps[i].e_sgs)) for i in eachindex(τt)], COLOR[:dns]; w = 2.4)
        for m in (:tke, :rsm)
            poly([(Xb(τt[i]), Yb(les[(c, m)].e[i])) for i in eachindex(τt)], COLOR[m])
        end
        # (c) Lumley triangle, η = (II/6)^½, ξ = (III/6)^⅓ of the mean invariants
        x0 = ml + 2(pw + gap)
        ηmax = 0.36
        Xc(ξ) = x0 + pw * (ξ + 0.2) / 0.56; Yc(η) = y0 + ph * (1 - η / ηmax)
        frame(x0, y0, "ξ", "η", -0.1:0.1:0.3, 0:0.1:0.3, Xc, Yc, "$c: Lumley triangle")
        ξs = range(-1 / 6, 1 / 3; length = 80)
        poly([(Xc(ξ), Yc(sqrt(1 / 27 + 2ξ^3))) for ξ in ξs], "#999999"; w = 1)
        poly([(Xc(0), Yc(0)), (Xc(-1 / 6), Yc(1 / 6))], "#999999"; w = 1)
        poly([(Xc(0), Yc(0)), (Xc(1 / 3), Yc(1 / 3))], "#999999"; w = 1)
        lt(II, III) = (Xc(cbrt(III / 6)), Yc(sqrt(max(II, 0) / 6)))
        for m in (:tke, :rsm)
            l = les[(c, m)]
            tr = [lt(l.II[i], l.III[i]) for i in eachindex(l.II)]
            poly(tr, COLOR[m])
            println(io, """<circle cx="$(tr[1][1])" cy="$(tr[1][2])" r="4.5" fill="none" stroke="$(COLOR[m])"/>""")
        end
        traj = [lt(s.II_mean, s.III_mean) for s in d.snaps]      # on top
        poly(traj, COLOR[:dns]; w = 2.4)
        println(io, """<circle cx="$(traj[1][1])" cy="$(traj[1][2])" r="2.5" fill="$(COLOR[:dns])"/>""")
    end
    # legend
    ly = H - 22; lx = ml
    for k in (:dns, MODELS...)
        println(io, """<line x1="$lx" y1="$(ly - 4)" x2="$(lx + 26)" y2="$(ly - 4)" stroke="$(COLOR[k])" stroke-width="$(k === :dns ? 2.4 : 1.6)"/>""")
        text(lx + 32, ly, LABEL[k]; a = "start")
        lx += 150
    end
    text(lx + 10, ly, "(Lumley: circles and dot at t₀)"; a = "start")
    println(io, "</svg>")
    mkpath(dirname(path))
    write(path, String(take!(io)))
    return path
end

if all(m -> m in MODELS, (:tke, :rsm))
    path = write_figure(plot_path(folder, "rotating_stratified_decay_$N.svg"))
    println("\nwrote ", path)
end
