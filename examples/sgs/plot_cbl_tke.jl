# The convective boundary layer of configs/cbl_tke.toml, the figure and
# the comparison with Lenschow, Wyngaard & Pennell (1980):
# σ_w²/w*² = 1.8 (z/z_i)^{2/3} (1 − 0.8 z/z_i)². Reads output/results.toml only.
#
#   julia --project -m Aeolus run examples/sgs/configs/cbl_tke.toml
#   julia --project=. examples/sgs/plot_cbl_tke.jl runs/<date>_cbl_tke_<id>
#
# Prints the σ_w² maximum against Lenschow et al. and the RMS difference
# over 0.1 < z/z_i < 0.9, and writes output/cbl_tke_<N>.svg (three panels
# against z/z_i: σ_w²/w*², the heat flux / Q with its SGS part, and the SGS
# and resolved TKE / w*²).

using Printf
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
res = read_results(folder)
N = run_config_table(folder)["grid"]["n"]
P = res["profiles"]
r = (N = N, zi = res["zi"], w_star = res["w_star"], zc = P["zc"], zf = P["zf"], w2 = P["w2"], flux = P["flux"],
     sgs_flux = P["sgs_flux"], e = P["e"], kres = P["kres"])

lenschow(x) = 1.8 * x^(2 / 3) * (1 - 0.8x)^2     # x = z/z_i ≤ 1.25
xs = range(0.01, 1.0, length = 400)
ipk = argmax(lenschow.(xs))
@printf("Lenschow et al. (1980): σ_w² max %.3f w*² at z/z_i = %.3f\n", lenschow(xs[ipk]), xs[ipk])
@printf("LES %d³:                 σ_w² max %.3f w*² at z/z_i = %.3f   (w* = %.2f m/s, z_i = %.0f m)\n",
        N, res["w2max_norm"], res["z_w2max_norm"], r.w_star, r.zi)
sel = [k for k in eachindex(r.zc) if 0.1 < r.zc[k] / r.zi < 0.9]
rms = sqrt(sum(((r.w2[k] / r.w_star^2) - lenschow(r.zc[k] / r.zi))^2 for k in sel) / length(sel))
@printf("RMS difference of σ_w²/w*² over 0.1 < z/z_i < 0.9: %.3f\n", rms)
@printf("min heat-flux ratio %.3f, max SGS share of TKE (0.2–0.8 z_i) %.3f, e peak at %.0f m (%.2f z_i); %d steps\n",
        res["min_flux_ratio"], res["max_sgs_fraction"], res["z_epeak"], res["z_epeak"] / r.zi, res["steps"])

# --- figure: three profile panels against z/z_i --------------------------
function write_figure(path, r)
    W, H = 980, 520
    panels = [("σ_w² / w*²", 0.0, 0.8), ("heat flux / Q", -0.4, 1.2), ("TKE / w*²", 0.0, 0.4)]
    pw = 260; l0 = 70; gap = 60; t, b = 40, 70
    ymax = 1.3
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif" font-size="13">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$(W / 2)" y="22" text-anchor="middle" font-size="15">CBL $(r.N)³, Deardorff–Moeng SGS TKE: z_i = $(round(Int, r.zi)) m, w* = $(round(r.w_star; digits = 2)) m/s</text>""")
    Y(x) = t + (H - t - b) * (1 - x / ymax)
    pts(xs, ys) = join(("$(round(a; digits = 2)),$(round(c; digits = 2))" for (a, c) in zip(xs, ys)), " ")
    for (p, (lab, xmin, xmax)) in enumerate(panels)
        x0 = l0 + (p - 1) * (pw + gap)
        X(v) = x0 + pw * (v - xmin) / (xmax - xmin)
        for yv in 0:0.25:1.25
            println(io, """<line x1="$x0" y1="$(Y(yv))" x2="$(x0 + pw)" y2="$(Y(yv))" stroke="#e4e4e4"/>""")
            p == 1 && println(io, """<text x="$(x0 - 8)" y="$(Y(yv) + 4)" text-anchor="end">$(yv)</text>""")
        end
        nt = 4
        for i in 0:nt
            v = xmin + i * (xmax - xmin) / nt
            println(io, """<line x1="$(X(v))" y1="$t" x2="$(X(v))" y2="$(H - b)" stroke="#e4e4e4"/>""")
            println(io, """<text x="$(X(v))" y="$(H - b + 18)" text-anchor="middle">$(round(v; digits = 2))</text>""")
        end
        xmin < 0 < xmax && println(io, """<line x1="$(X(0))" y1="$t" x2="$(X(0))" y2="$(H - b)" stroke="#999"/>""")
        println(io, """<rect x="$x0" y="$t" width="$pw" height="$(H - t - b)" fill="none" stroke="black"/>""")
        println(io, """<text x="$(x0 + pw / 2)" y="$(H - b + 40)" text-anchor="middle">$lab</text>""")
        zc = r.zc ./ r.zi; zf = r.zf ./ r.zi
        sel(z) = z .<= ymax
        if p == 1
            xs = range(0.0, 1.25, length = 200)
            println(io, """<polyline fill="none" stroke="#888" stroke-width="2" stroke-dasharray="6,4" points="$(pts(X.(lenschow.(xs)), Y.(xs)))"/>""")
            k = sel(zc)
            println(io, """<polyline fill="none" stroke="#1b6ca8" stroke-width="2.5" points="$(pts(X.(r.w2[k] ./ r.w_star^2), Y.(zc[k])))"/>""")
            println(io, """<text x="$(x0 + pw - 8)" y="$(t + 20)" text-anchor="end" fill="#1b6ca8">LES</text>""")
            println(io, """<text x="$(x0 + pw - 8)" y="$(t + 38)" text-anchor="end" fill="#666">Lenschow et al. 1980</text>""")
        elseif p == 2
            k = sel(zf)
            println(io, """<polyline fill="none" stroke="#c8553d" stroke-width="2.5" points="$(pts(X.(r.flux[k]), Y.(zf[k])))"/>""")
            println(io, """<polyline fill="none" stroke="#c8553d" stroke-width="1.5" stroke-dasharray="4,3" points="$(pts(X.(r.sgs_flux[k]), Y.(zf[k])))"/>""")
            println(io, """<text x="$(x0 + pw - 8)" y="$(t + 20)" text-anchor="end" fill="#c8553d">total</text>""")
            println(io, """<text x="$(x0 + pw - 8)" y="$(t + 38)" text-anchor="end" fill="#c8553d">SGS (dashed)</text>""")
        else
            k = sel(zc)
            println(io, """<polyline fill="none" stroke="#4a7c3a" stroke-width="2.5" points="$(pts(X.(r.e[k] ./ r.w_star^2), Y.(zc[k])))"/>""")
            println(io, """<polyline fill="none" stroke="#333" stroke-width="2" stroke-dasharray="6,4" points="$(pts(X.(r.kres[k] ./ r.w_star^2), Y.(zc[k])))"/>""")
            println(io, """<text x="$(x0 + pw - 8)" y="$(t + 20)" text-anchor="end" fill="#4a7c3a">SGS e</text>""")
            println(io, """<text x="$(x0 + pw - 8)" y="$(t + 38)" text-anchor="end" fill="#333">resolved (dashed)</text>""")
        end
    end
    println(io, """<text transform="translate(22,$((H - b + t) / 2)) rotate(-90)" text-anchor="middle">z / z_i</text>""")
    println(io, "</svg>")
    mkpath(dirname(path))
    write(path, String(take!(io)))
    return path
end

out = write_figure(plot_path(folder, "cbl_tke_$(N).svg"), r)
println("wrote ", out)
