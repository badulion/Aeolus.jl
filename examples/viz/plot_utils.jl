# Minimal, pure-Julia plotting helpers.
#
# We avoid Makie / Plots / GR here because the NixOS build machine has a
# glibc / artifact mismatch that breaks several Julia C-dependent
# packages (SpecialFunctions → StatsFuns → Makie failed to precompile;
# NCDatasets did too then, but loads since #310). Rather than fight the system,
# we write our own tiny plotting utilities:
#
#   * `save_heatmap(path, Z; …)` — writes an ASCII PPM image (`.ppm`).
#     PPM opens in most image viewers directly and converts to PNG with
#     a single `convert` / `ImageMagick` call.
#   * `save_lineplot(path, xs, ys; …)` — writes a self-contained SVG.
#   * `save_multiline(path, xs, ys_list, labels; …)` — several lines
#     on one SVG.
#
# All output is text-based; no binary encoding gymnastics.

# ------------------------------------------------------------------
# Colormap: perceptual viridis approximation (5 anchor points, LERP).
# ------------------------------------------------------------------

const _VIRIDIS = (
    (68,   1,  84),
    (59,  82, 139),
    (33, 145, 140),
    (94, 201,  98),
    (253, 231, 37),
)

function _colormap(t::Real)
    t = clamp(t, 0, 1)
    n  = length(_VIRIDIS) - 1
    s  = t * n
    i  = clamp(Int(floor(s)) + 1, 1, n)
    frac = s - (i - 1)
    c0 = _VIRIDIS[i]
    c1 = _VIRIDIS[i + 1]
    r = round(Int, c0[1] * (1 - frac) + c1[1] * frac)
    g = round(Int, c0[2] * (1 - frac) + c1[2] * frac)
    b = round(Int, c0[3] * (1 - frac) + c1[3] * frac)
    return (r, g, b)
end

# ------------------------------------------------------------------
# Heatmap → PPM (ASCII P3)
# ------------------------------------------------------------------

"""
    save_heatmap(path, Z; vmin=nothing, vmax=nothing, upscale=8)

Save a 2-D array `Z` as a viridis-colored PPM image at `path`.
Missing bounds default to `extrema(Z)`. `upscale` replicates each cell
`upscale × upscale` times to give visible pixels for small grids.
Filename may end in `.ppm`.
"""
function save_heatmap(path::AbstractString, Z::AbstractMatrix;
                      vmin = nothing, vmax = nothing, upscale::Int = 8)
    ny, nx = size(Z)
    # NaN-safe bounds
    finite_vals = filter(isfinite, Z)
    isempty(finite_vals) && (finite_vals = [0.0])
    v0 = vmin === nothing ? minimum(finite_vals) : vmin
    v1 = vmax === nothing ? maximum(finite_vals) : vmax
    span = v1 == v0 ? one(eltype(Z)) : (v1 - v0)
    W = nx * upscale; H = ny * upscale
    open(path, "w") do io
        println(io, "P3")
        println(io, "$W $H")
        println(io, "255")
        for j_out in 1:H
            j = min(cld(j_out, upscale), ny)
            for i_out in 1:W
                i = min(cld(i_out, upscale), nx)
                z = Z[j, i]
                t = isfinite(z) ? (z - v0) / span : 0.5
                r, g, b = _colormap(t)
                print(io, "$r $g $b ")
            end
            println(io)
        end
    end
    return path
end

# ------------------------------------------------------------------
# Line plot → SVG
# ------------------------------------------------------------------

"""
    save_lineplot(path, xs, ys; xlabel="", ylabel="", title="",
                  width=640, height=400)

Save a single-line plot as a self-contained SVG.
"""
save_lineplot(path, xs, ys; kwargs...) =
    save_multiline(path, xs, [ys], [""]; kwargs...)

"""
    save_multiline(path, xs, ys_list, labels; xlabel, ylabel, title, width, height,
                   logx, logy, legend = :topright)

Save several polylines to one SVG. `ys_list` is a vector of same-length
`Vector`s; `labels` is a vector of legend strings.
"""
function save_multiline(path::AbstractString, xs::AbstractVector,
                        ys_list::AbstractVector, labels::AbstractVector;
                        xlabel::AbstractString = "",
                        ylabel::AbstractString = "",
                        title::AbstractString  = "",
                        width::Int = 640, height::Int = 400,
                        logx::Bool = false, logy::Bool = false,
                        legend::Symbol = :topright)
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 50
    plot_w = width  - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    xs_t = logx ? log10.(xs) : collect(xs)
    ys_ts = [logy ? log10.(y) : collect(y) for y in ys_list]

    xmin, xmax = extrema(xs_t)
    ymin = minimum(minimum, ys_ts)
    ymax = maximum(maximum, ys_ts)
    xmin == xmax && (xmax = xmin + 1)
    ymin == ymax && (ymax = ymin + 1)

    sx = plot_w / (xmax - xmin)
    sy = plot_h / (ymax - ymin)
    x2p(x) = pad_l + (x - xmin) * sx
    y2p(y) = pad_t + plot_h - (y - ymin) * sy

    colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B",
              "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"]

    open(path, "w") do io
        println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" font-family="sans-serif">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        # axes
        println(io, """<line x1="$pad_l" y1="$(pad_t)" x2="$pad_l" y2="$(pad_t + plot_h)" stroke="#333" stroke-width="1"/>""")
        println(io, """<line x1="$pad_l" y1="$(pad_t + plot_h)" x2="$(pad_l + plot_w)" y2="$(pad_t + plot_h)" stroke="#333" stroke-width="1"/>""")
        # tick labels — 5 on each axis
        for k in 0:4
            xv = xmin + k * (xmax - xmin) / 4
            xp = x2p(xv)
            xlab = logx ? "10^$(round(xv; digits=1))" : string(round(xv; digits=2))
            println(io, """<text x="$xp" y="$(pad_t + plot_h + 20)" font-size="10" text-anchor="middle" fill="#333">$xlab</text>""")
            println(io, """<line x1="$xp" y1="$(pad_t + plot_h)" x2="$xp" y2="$(pad_t + plot_h + 4)" stroke="#333" stroke-width="1"/>""")
        end
        for k in 0:4
            yv = ymin + k * (ymax - ymin) / 4
            yp = y2p(yv)
            ylab = logy ? "10^$(round(yv; digits=1))" : string(round(yv; digits=3))
            println(io, """<text x="$(pad_l - 6)" y="$(yp + 4)" font-size="10" text-anchor="end" fill="#333">$ylab</text>""")
            println(io, """<line x1="$(pad_l - 4)" y1="$yp" x2="$pad_l" y2="$yp" stroke="#333" stroke-width="1"/>""")
        end
        # labels
        isempty(xlabel) || println(io, """<text x="$(pad_l + plot_w/2)" y="$(height - 12)" font-size="12" text-anchor="middle" fill="#333">$xlabel</text>""")
        if !isempty(ylabel)
            println(io, """<text x="15" y="$(pad_t + plot_h/2)" font-size="12" text-anchor="middle" fill="#333" transform="rotate(-90 15,$(pad_t + plot_h/2))">$ylabel</text>""")
        end
        isempty(title) || println(io, """<text x="$(width/2)" y="24" font-size="16" text-anchor="middle" fill="#111" font-weight="bold">$title</text>""")
        # polylines
        for (li, ys) in enumerate(ys_ts)
            col = colors[mod1(li, length(colors))]
            pts = join(("$(x2p(xs_t[i])),$(y2p(ys[i]))" for i in eachindex(xs_t)), " ")
            println(io, """<polyline fill="none" stroke="$col" stroke-width="1.6" points="$pts"/>""")
        end
        # legend
        legend_x = pad_l + plot_w - 100
        nlab = count(!isempty, labels)
        legend_y = legend === :bottomright ? pad_t + plot_h - 14 * nlab - 6 : pad_t + 10
        for (li, lab) in enumerate(labels)
            isempty(lab) && continue
            col = colors[mod1(li, length(colors))]
            ly  = legend_y + 14 * (li - 1)
            println(io, """<line x1="$legend_x" y1="$ly" x2="$(legend_x + 20)" y2="$ly" stroke="$col" stroke-width="2"/>""")
            println(io, """<text x="$(legend_x + 26)" y="$(ly + 4)" font-size="11" fill="#333">$lab</text>""")
        end
        println(io, "</svg>")
    end
    return path
end
