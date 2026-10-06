# Map of the Berlin parent's surface (N-parent-d, #249): the class
# fractions of the 6 km box and the unresolved-building roughness.
#
#   julia --project=. examples/urban/parent/plot_parent_surface.jl [out.svg]
#
# writes docs/src/assets/berlin/parent_surface.svg (or `out.svg`): six
# 192 × 192 panels (31.25 m cells) — roof, park and water fraction, the
# morphometry window's λp, and Macdonald et al. (1998) z0 and d — with the
# child's 1 km footprint outlined. The rasters are PNGs (the in-package
# encoder of examples/viz/apng.jl) embedded in the SVG; all text is SVG
# text, so no bitmap font is involved.

using Aeolus, Printf
isdefined(Main, :BerlinCase) || include(joinpath(@__DIR__, "..", "BerlinCase.jl"))
include(joinpath(@__DIR__, "..", "..", "viz", "apng.jl"))
const BC = Main.BerlinCase

out = length(ARGS) >= 1 ? ARGS[1] :
      joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "berlin", "parent_surface.svg")
mkpath(dirname(out))

cfg = BC.CaseConfig(; verbose = false, output_path = nothing)
case = BC.build_case(cfg; io = devnull)
pl = BC.read_parent_landuse()
ps = BC.build_parent_surface(case, pl; columns = CartesianIndex{2}[])
f = ps.fractions
Is, Js = ps.footprint

# RFC 4648 base64 (Base64 is not a dependency of the project)
const _B64 = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
function b64(data::Vector{UInt8})
    out = IOBuffer()
    for i in 1:3:length(data)
        n = min(3, length(data) - i + 1)
        v = UInt32(data[i]) << 16 | (n > 1 ? UInt32(data[i + 1]) << 8 : 0x0) | (n > 2 ? UInt32(data[i + 2]) : 0x0)
        for k in 0:3
            print(out, k <= n ? _B64[(v >> (18 - 6k)) & 0x3f + 1] : '=')
        end
    end
    return String(take!(out))
end

# one panel: values on the parent grid (I east, J north) → north-up PNG
function panel_png(Z, vmin, vmax; up = 1)
    img = permutedims(Z[:, end:-1:1])                  # rows = north → south
    rgb = scalar_to_rgb(img, vmin, vmax; upscale = up)
    path = tempname() * ".png"
    write_png(path, rgb)
    return b64(read(path))
end

log10c(x, lo) = log10(max(x, lo))
panels = [
    ("(a) roof fraction (per 31.25 m cell)", f[:, :, BC.SURFACE_ROOF], 0.0, 1.0, "0", "1"),
    ("(b) park / green fraction", f[:, :, BC.SURFACE_PARK], 0.0, 1.0, "0", "1"),
    ("(c) water fraction", f[:, :, BC.SURFACE_WATER], 0.0, 1.0, "0", "1"),
    ("(d) λp, 250 m window", ps.λp, 0.0, 0.6, "0", "0.6"),
    ("(e) z0, Macdonald (1998), log10 m", log10c.(ps.z0, 1e-2), -2.0, 1.0, "≤ 0.01 m", "10 m"),
    ("(f) d, Macdonald (1998), m", ps.d, 0.0, 30.0, "0", "≥ 30 m"),
]

W, H = 3 * 420 + 40, 2 * 470 + 90
io = IOBuffer()
println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif" font-size="13">""")
println(io, """<rect width="$W" height="$H" fill="white"/>""")
println(io, """<text x="20" y="26" font-size="16">Berlin parent surface: 6 × 6 km around Alexanderplatz, 192 × 192 cells of 31.25 m (OSM, $(match(r"osm_base = ([^,\s]+)", join(pl.header))[1]))</text>""")
stops = join(("<stop offset=\"$(t)\" stop-color=\"rgb$(Int.(_viridis_apng(t)))\"/>" for t in 0:0.125:1), "")
println(io, """<defs><linearGradient id="vir">$stops</linearGradient></defs>""")
scale = 384 / pl.nx                                     # px per parent cell
for (p, (title, Z, lo, hi, llo, lhi)) in enumerate(panels)
    col, row = (p - 1) % 3, (p - 1) ÷ 3
    x0, y0 = 20 + col * 420, 50 + row * 470
    println(io, """<text x="$x0" y="$(y0 + 14)">$title</text>""")
    println(io, """<image x="$x0" y="$(y0 + 22)" width="384" height="384" style="image-rendering:pixelated" href="data:image/png;base64,$(panel_png(Z, lo, hi))"/>""")
    # the child's footprint: cells Is × Js (J north, so the top edge is Js[end])
    fx = x0 + (first(Is) - 1) * scale
    fy = y0 + 22 + (pl.ny - last(Js)) * scale
    println(io, """<rect x="$fx" y="$fy" width="$(length(Is) * scale)" height="$(length(Js) * scale)" fill="none" stroke="red" stroke-width="1.5"/>""")
    # colour bar
    cy = y0 + 22 + 392
    println(io, """<rect x="$x0" y="$cy" width="384" height="10" fill="url(#vir)"/>""")
    println(io, """<text x="$x0" y="$(cy + 24)">$llo</text>""")
    println(io, """<text x="$(x0 + 384)" y="$(cy + 24)" text-anchor="end">$lhi</text>""")
end
println(io, """<text x="20" y="$(H - 14)" font-size="12">Red: the child's 1 km footprint, where the parent fractions are the child's own class map coarse-grained 2 × 2. North up; x east. Data © OpenStreetMap contributors (ODbL).</text>""")
println(io, "</svg>")
write(out, take!(io))
@printf("wrote %s (%.0f kB); z0 %.3f–%.2f m, d 0–%.1f m, box-mean roof fraction %.3f\n",
        out, filesize(out) / 1024, minimum(ps.z0[ps.z0 .> 0]), maximum(ps.z0), maximum(ps.d),
        sum(f[:, :, BC.SURFACE_ROOF]) / (pl.nx * pl.ny))
