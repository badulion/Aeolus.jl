# =========================================================================
# Convective boundary layer, LES against DNS (#503): figures and an
# animation. Output analysis only: it reads the CSV files analysis.jl wrote
# (examples/cbl_dns_les/output/) and, for the animation, the run folders'
# output/output.nc; it runs no simulation code.
#
#   julia --project examples/cbl_dns_les/analysis.jl runs/*_cbl_*
#   julia --project examples/cbl_dns_les/plots.jl runs/*_cbl_*
#
# writes into examples/cbl_dns_les/output/:
#
#   cbl_profiles.svg   the total heat flux / Q, σ_w²/w*² and the resolved
#                      TKE / w*² against z/z_i, averaged over the second
#                      hour (t ≥ 60 min, analysis.jl's window), every run a curve
#   cbl_growth.svg     z_i(t) and the entrainment ratio A(t)
#   cbl_spectra.svg    E_w(k) at 0.5 z_i at that record (log-log)
#   cbl_w_xz.apng      w in the x–z plane through the domain's middle, one
#                      panel per run, every record (blue down, red up,
#                      ±1.5 w*); cbl_w_xz.png: its last frame
# =========================================================================

using Printf
using Aeolus.NCIO: NCDataset
include(joinpath(@__DIR__, "..", "sbl", "svgplot.jl"))
include(joinpath(@__DIR__, "..", "viz", "apng.jl"))

const OUT = joinpath(@__DIR__, "output")

"A CSV file of analysis.jl as a vector of Dicts (numbers parsed, the run name kept)."
function read_csv(name)
    lines = readlines(joinpath(OUT, name))
    hdr = split(lines[1], ',')
    return [Dict(h => (h == "run" ? String(v) : parse(Float64, v)) for (h, v) in zip(hdr, split(l, ',')))
            for l in lines[2:end] if !isempty(l)]
end

"The runs of a table, the DNS first."
function runs_of(rows)
    names = unique(r["run"] for r in rows)
    return sort(names; by = n -> n == "cbl_dns" ? "" : n)
end

"Nice tick positions over (lo, hi), about n of them."
function ticks(lo, hi; n = 5)
    step = 10.0^floor(log10((hi - lo) / n))
    for m in (1, 2, 5, 10)
        (hi - lo) / (m * step) <= n && (step *= m; break)
    end
    return collect(ceil(lo / step) * step:step:hi)
end

"Linear interpolation of (xs, ys) at x (clamped)."
function _interp(xs, ys, x)
    x <= xs[1] && return ys[1]; x >= xs[end] && return ys[end]
    k = searchsortedlast(xs, x)
    return ys[k] + (ys[k+1] - ys[k]) * (x - xs[k]) / (xs[k+1] - xs[k])
end

function profiles_figure(prof; window = 3600.0)
    names = runs_of(prof)
    ζ = collect(range(0.0, 1.5; length = 76))
    "the run's window mean of column `key` on ζ"
    function wmean(n, key)
        ts = unique(r["t"] for r in prof if r["run"] == n && r["t"] >= window - 30)
        acc = zeros(length(ζ))
        for t in ts
            rs = filter(r -> r["run"] == n && r["t"] == t, prof)
            acc .+= [_interp([r["z_over_zi"] for r in rs], [r[key] for r in rs], z) for z in ζ]
        end
        return acc ./ length(ts)
    end
    fig = svg_figure(1200, 460, @sprintf("CBL: DNS and LES, mean over t ≥ %.0f min (mixed-layer scaling)", window / 60))
    leg = [(n, i, "") for (i, n) in enumerate(names)]
    for (p, (key, xlab, xlim)) in enumerate((("heat_flux_over_Q", "total heat flux / Q", (-0.4, 1.1)),
                                             ("w2_norm", "σ_w² / w*²", (0.0, 0.6)),
                                             ("tke_norm", "TKE / w*² (resolved + SGS)", (0.0, 0.8))))
        curves = [(wmean(n, key), ζ, i) for (i, n) in enumerate(names)]
        svg_panel(fig, 70 + (p - 1) * 380, 50, 320, 340; xlim = xlim, ylim = (0.0, 1.5), xticks = ticks(xlim...),
                  yticks = 0:0.25:1.5, xlabel = xlab, ylabel = p == 1 ? "z / z_i" : "", curves = curves,
                  legend = p == 2 ? leg : [], legend_x = 150)
    end
    svg_write(fig, joinpath(OUT, "cbl_profiles.svg"))
end

function growth_figure(series)
    names = runs_of(series)
    fig = svg_figure(820, 420, "CBL: boundary-layer height and entrainment ratio")
    sel(n) = filter(r -> r["run"] == n, series)
    tmax = maximum(r["t"] for r in series) / 60
    zmax = maximum(r["z_i"] for r in series)
    c(key; f = identity) = [([r["t"] / 60 for r in sel(n)], [f(r[key]) for r in sel(n)], i) for (i, n) in enumerate(names)]
    svg_panel(fig, 70, 50, 320, 300; xlim = (0.0, tmax), ylim = (0.0, 1.1 * zmax), xticks = ticks(0, tmax),
              yticks = ticks(0, 1.1 * zmax), xlabel = "t (min)", ylabel = "z_i (m)", curves = c("z_i"),
              legend = [(n, i, "") for (i, n) in enumerate(names)], legend_x = 10)
    svg_panel(fig, 470, 50, 320, 300; xlim = (0.0, tmax), ylim = (0.0, 0.4), xticks = ticks(0, tmax),
              yticks = 0:0.1:0.4, xlabel = "t (min)", ylabel = "A = −min flux / Q", curves = c("A"))
    svg_write(fig, joinpath(OUT, "cbl_growth.svg"))
end

function spectra_figure(spec, t)
    names = runs_of(spec)
    sel(n) = filter(r -> r["run"] == n && r["t"] == t && r["E_w"] > 0, spec)
    curves = Any[([log10(r["k"]) for r in sel(n)], [log10(r["E_w"]) for r in sel(n)], i) for (i, n) in enumerate(names)]
    xs = vcat(first.(curves)...); ys = vcat(getindex.(curves, 2)...)
    xlim = (floor(minimum(xs)), ceil(maximum(xs))); ylim = (floor(maximum(ys)) - 6, ceil(maximum(ys)))
    fig = svg_figure(520, 440, @sprintf("CBL: spectrum of w at 0.5 z_i, t = %.0f min", t / 60))
    # the −5/3 slope through the first run's spectral peak, for the eye
    ip = argmax(curves[1][2]); k0, E0 = curves[1][1][ip], curves[1][2][ip] + 0.3
    push!(curves, ([k0, k0 + 2], [E0, E0 - 2 * 5 / 3], 8, "4,4"))
    svg_panel(fig, 70, 50, 400, 320; xlim = xlim, ylim = ylim, xticks = xlim[1]:1:xlim[2], yticks = ylim[1]:1:ylim[2],
              xlabel = "log10 k (rad/m)", ylabel = "log10 E_w (m³/s²)", curves = curves,
              legend = vcat([(n, i, "") for (i, n) in enumerate(names)], [("k^(-5/3)", 8, "4,4")]), legend_x = 10)
    svg_write(fig, joinpath(OUT, "cbl_spectra.svg"))
end

"""
w in the x–z plane through the middle of y, every record, resampled to a
uniform z; with the pixel upscale `up` (256 px wide panels) every panel is
`height` px tall.
"""
function w_slices(folder; height = 160)
    NCDataset(joinpath(folder, "output", "output.nc")) do ds
        x = Float64.(ds["x"][:]); z = Float64.(ds["z"][:]); t = Float64.(ds["time"][:])
        up = max(1, 256 ÷ length(x)); nz = height ÷ up
        jm = cld(length(ds["y"][:]), 2)
        zs = range(0.0, z[end]; length = nz)
        frames = map(eachindex(t)) do r
            w = Float64.(ds["w"][:, jm, :, r])           # (x, z)
            # rows: z from the top down; columns: x
            [w[i, clamp(searchsortedlast(z, zs[nz + 1 - k]), 1, length(z))] for k in 1:nz, i in eachindex(x)]
        end
        (t = t, frames = frames, up = up)
    end
end

function animation(folders, series)
    isempty(folders) && return
    runs = [(name = basename(rstrip(f, '/')), slices = w_slices(f)) for f in folders]
    # the DNS first, its w* of each record sets the colour range
    ord = sortperm([occursin("cbl_dns", r.name) ? 0 : 1 for r in runs])
    runs = runs[ord]
    wstar(t) = (s = filter(r -> r["run"] == "cbl_dns" && r["t"] == t, series);
                isempty(s) ? maximum(r["w_star"] for r in series) : only(s)["w_star"])
    times = intersect([r.slices.t for r in runs]...)
    frames = Array{UInt8,3}[]
    for t in times
        ws = 1.5 * (t > 0 ? wstar(t) : 1.0)
        panels = map(runs) do r
            k = findfirst(==(t), r.slices.t)
            Z = r.slices.frames[k]
            img = scalar_to_rgb(Z, -ws, ws; upscale = r.slices.up, cmap = :rdbu)
            lbl = replace(split(r.name, "_cbl_")[end], r"_[0-9a-f]{8}$" => "")
            add_label!(img, @sprintf("%s t %.0f min", lbl, t / 60); color = (0, 0, 0))
            img
        end
        h = minimum(size(p, 1) for p in panels); w = minimum(size(p, 2) for p in panels)
        push!(frames, tile_frames(reshape([p[1:h, 1:w, :] for p in panels], 1, :)))
    end
    write_apng(joinpath(OUT, "cbl_w_xz.apng"), frames; delay_ms = 600)
    write_png(joinpath(OUT, "cbl_w_xz.png"), frames[end])
end

function make_plots(folders)
    series = read_csv("cbl_series.csv"); prof = read_csv("cbl_profiles.csv"); spec = read_csv("cbl_spectra.csv")
    # the last record every run has
    t = minimum(maximum(r["t"] for r in series if r["run"] == n) for n in runs_of(series))
    profiles_figure(prof)
    growth_figure(series)
    spectra_figure(spec, t)
    animation(folders, series)
    println("wrote ", OUT, "/cbl_{profiles,growth,spectra}.svg", isempty(folders) ? "" : " and cbl_w_xz.{apng,png}")
end

make_plots(ARGS)
