# =========================================================================
# Minimal SVG line-plot helper for the examples/sbl/ figures (P58-d, #194).
# No dependency: each figure is a grid of panels with polylines, optional
# shaded bands (reference envelopes) and markers.
# =========================================================================

const SVG_COLORS = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d",
                    "#4b5563", "#65a30d")

mutable struct SVGFigure
    io::IOBuffer
    W::Int; H::Int
end

function svg_figure(W, H, title)
    io = IOBuffer()
    println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="24" font-size="16" text-anchor="middle" font-weight="bold">$(svg_escape(title))</text>""")
    return SVGFigure(io, W, H)
end

svg_escape(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

_fmt(v) = abs(v - round(v)) < 1e-9 ? string(round(Int, v)) : string(round(v; sigdigits = 3))

"""
    svg_panel(fig, x0, y0, pw, ph; xlim, ylim, xticks, yticks, xlabel, ylabel, tag,
              curves = [], bands = [], hbands = [], points = [], legend = [])

One panel at pixel origin `(x0, y0)` of size `pw × ph`. `curves` are
`(xs, ys, color_index[, dash])`, `bands` vertical-profile envelopes
`(lo_xs, hi_xs, ys)` shaded in grey, `hbands` horizontal bands
`(ylo, yhi)`, `points` markers `(xs, ys, color_index, shape)` with shape
`:circle` or `:square`. Values outside the limits are clipped to the frame.
"""
function svg_panel(fig::SVGFigure, x0, y0, pw, ph; xlim, ylim, xticks, yticks, xlabel = "", ylabel = "",
                   tag = "", curves = [], bands = [], hbands = [], vbands = [], points = [], legend = [])
    io = fig.io
    X(x) = x0 + (clamp(x, xlim...) - xlim[1]) / (xlim[2] - xlim[1]) * pw
    Y(y) = y0 + ph - (clamp(y, ylim...) - ylim[1]) / (ylim[2] - ylim[1]) * ph
    for (lo, hi, ys) in bands
        ok = [i for i in eachindex(ys) if isfinite(lo[i]) && isfinite(hi[i])]
        pts = vcat(["$(X(lo[i])),$(Y(ys[i]))" for i in ok], ["$(X(hi[i])),$(Y(ys[i]))" for i in reverse(ok)])
        println(io, """<polygon points="$(join(pts, " "))" fill="#d1d5db" stroke="none"/>""")
    end
    for (ylo, yhi) in hbands
        println(io, """<rect x="$x0" y="$(Y(yhi))" width="$pw" height="$(Y(ylo) - Y(yhi))" fill="#e5e7eb"/>""")
    end
    for (xlo, xhi) in vbands
        println(io, """<rect x="$(X(xlo))" y="$y0" width="$(X(xhi) - X(xlo))" height="$ph" fill="#e5e7eb"/>""")
    end
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    for xv in xticks
        println(io, """<line x1="$(X(xv))" y1="$(y0+ph)" x2="$(X(xv))" y2="$(y0+ph+5)" stroke="#333"/>""")
        println(io, """<text x="$(X(xv))" y="$(y0+ph+18)" font-size="12" text-anchor="middle">$(_fmt(xv))</text>""")
    end
    for yv in yticks
        println(io, """<line x1="$(x0-5)" y1="$(Y(yv))" x2="$x0" y2="$(Y(yv))" stroke="#333"/>""")
        println(io, """<text x="$(x0-8)" y="$(Y(yv)+4)" font-size="12" text-anchor="end">$(_fmt(yv))</text>""")
    end
    for cv in curves
        xs, ys, ci = cv[1], cv[2], cv[3]
        dash = length(cv) >= 4 ? cv[4] : ""
        ok = [i for i in eachindex(xs) if isfinite(xs[i]) && isfinite(ys[i])]
        pts = join(("$(X(xs[i])),$(Y(ys[i]))" for i in ok), " ")
        da = isempty(dash) ? "" : """ stroke-dasharray="$dash\""""
        println(io, """<polyline fill="none" stroke="$(SVG_COLORS[ci])" stroke-width="2"$da points="$pts"/>""")
    end
    for (xs, ys, ci, shape) in points
        for (x, y) in zip(xs, ys)
            (isfinite(x) && isfinite(y)) || continue
            if shape === :square
                println(io, """<rect x="$(X(x)-4)" y="$(Y(y)-4)" width="8" height="8" fill="none" stroke="$(SVG_COLORS[ci])" stroke-width="1.5"/>""")
            else
                println(io, """<circle cx="$(X(x))" cy="$(Y(y))" r="4" fill="none" stroke="$(SVG_COLORS[ci])" stroke-width="1.5"/>""")
            end
        end
    end
    println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+38)" font-size="13" text-anchor="middle">$(svg_escape(xlabel))</text>""")
    println(io, """<text x="$(x0-44)" y="$(y0+ph/2)" font-size="13" text-anchor="middle" transform="rotate(-90 $(x0-44),$(y0+ph/2))">$(svg_escape(ylabel))</text>""")
    isempty(tag) || println(io, """<text x="$(x0+6)" y="$(y0+17)" font-size="13" font-weight="bold">$(svg_escape(tag))</text>""")
    for (i, (label, ci, dash)) in enumerate(legend)
        ly = y0 + 14 + 16 * i
        da = isempty(dash) ? "" : """ stroke-dasharray="$dash\""""
        println(io, """<line x1="$(x0+pw-130)" y1="$(ly-4)" x2="$(x0+pw-108)" y2="$(ly-4)" stroke="$(SVG_COLORS[ci])" stroke-width="2"$da/>""")
        println(io, """<text x="$(x0+pw-104)" y="$ly" font-size="11">$(svg_escape(label))</text>""")
    end
    return nothing
end

svg_text(fig::SVGFigure, x, y, s; size = 12, anchor = "start", color = "#333") =
    println(fig.io, """<text x="$x" y="$y" font-size="$size" text-anchor="$anchor" fill="$color">$(svg_escape(s))</text>""")

function svg_write(fig::SVGFigure, path)
    println(fig.io, "</svg>")
    write(path, String(take!(fig.io)))
    return path
end
