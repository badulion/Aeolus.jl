# =========================================================================
# Decaying turbulence under rotation and stratification: the SGS closures
# against filtered DNS — Era VI, P56-e (#192)
#
# The CI test (test/era6_sgs_validation.jl) at 96³: the P56-c DNS of
# decaying isotropic turbulence (Re_λ ≈ 64 at t₀ = 2) continued for two
# turnovers T = L/u′ as is, with rotation (Ro = 0.2), with a background
# stratification (Fr = 0.3), and with both; each LES (Δ = 4h, 24³) starts
# from the filtered DNS with e and τᵢⱼ from the exact subfilter stress
# and is compared with the filtered DNS. Closures: Smagorinsky
# (C_s = 0.17), dynamic Smagorinsky, the Deardorff TKE closure (P56-b)
# and the Reynolds-stress closure (P56-d). Method and fixture:
# test/fixtures/sgs_validation.jl.
#
#   julia --project=. examples/sgs/rotating_stratified_decay.jl [N]
#
# writes examples/sgs/output/rotating_stratified_decay_<N>.svg (resolved
# KE, SGS energy and the Lumley-triangle trajectories of the mean
# invariants, one row per case) and prints the tables quoted in
# docs/src/theory/sgs_tke.md.
# =========================================================================

using Aeolus
using Printf
include(joinpath(@__DIR__, "..", "..", "test", "fixtures", "sgs_validation.jl"))

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 96
CASES = (:isotropic, :rotating, :stratified, :rotating_stratified)
MODELS = (:smagorinsky, :dynamic, :tke, :rsm)
LABEL = Dict(:smagorinsky => "Smagorinsky", :dynamic => "dynamic Smag.", :tke => "TKE",
             :rsm => "Reynolds stress", :dns => "filtered DNS")
COLOR = Dict(:dns => "#000000", :smagorinsky => "#8c8c8c", :dynamic => "#4a7c3a",
             :tke => "#1b6ca8", :rsm => "#c8553d")

t_base = @elapsed base = validation_base(; N = N)
@printf("DNS %d³: spin-up %.0f s, Re_λ = %.1f, u′ = %.3f, L = %.3f, T = %.3f, f = %.3f, N = %.3f\n",
        N, t_base, base.Re_λ, base.up, base.L, base.T, base.f, base.Nbv)

dns = Dict{Symbol,Any}(); les = Dict{Tuple{Symbol,Symbol},Any}()
for c in CASES
    dns[c] = run_dns_case(base, c; nsave = 16)
    @printf("\n%s: DNS continuation %.0f s, K(t₀+2T)/K(t₀) = %.3f (full resolution)\n",
            c, dns[c].wall, dns[c].K[end] / dns[c].K[1])
    for m in MODELS
        les[(c, m)] = run_les(base, dns[c], m)
    end
    d = dns[c]
    @printf("  %-16s %9s %9s %9s %9s %8s\n", "", "KE err", "e(2T)", "II(2T)", "τ33/τ11", "LES s")
    @printf("  %-16s %9s %9.4f %9.4f %9.3f\n", "filtered DNS", "—", d.snaps[end].e_sgs,
            d.snaps[end].II_mean, vmean([s.r33 for s in d.snaps[2:end]]))
    for m in MODELS
        l = les[(c, m)]
        @printf("  %-16s %+8.1f%% %9.4f %9.4f %9.3f %8.1f\n", LABEL[m], 100 * (l.K[end] / d.snaps[end].K - 1),
                l.e[end], m in (:tke, :rsm) ? l.II[end] : NaN, m in (:tke, :rsm) ? vmean(l.r33[2:end]) : NaN, l.wall)
    end
    if d.Γ > 0
        h_d = vmean([s.hw_mean for s in d.snaps[2:end]])
        @printf("  mean SGS heat flux over (t₀, t₀+2T]: DNS %.3f, TKE %.3f, RSM %.3f\n", h_d,
                vmean(les[(c, :tke)].hw[2:end]), vmean(les[(c, :rsm)].hw[2:end]))
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

path = write_figure(joinpath(@__DIR__, "output", "rotating_stratified_decay_$N.svg"))
println("\nwrote ", path)
