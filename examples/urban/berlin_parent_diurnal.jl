# The Berlin nesting parent alone (N-parent-e, #259): a 24 h diurnal run,
# 06 to 06 LT, of the 6 × 6 × 3 km Aeolus parent around Alexanderplatz on
# 29/30 June 2019 (the Lindenberg sounding of #247), with every parent
# forcing on: the tiled urban surface (#249), the child's column radiation,
# the geostrophic pressure gradient, θ/qv nudging, subsidence, θ damping
# and the fixed-gradient top (#248, #254), the TKE closure on the
# stretched grid. See docs/src/design/berlin_parent.md.
#
#   julia --project=. -t 3 examples/urban/berlin_parent_diurnal.jl [grid] [hours] [D]
#
# grid  `default`  (2, 2, 4): 192 × 192 × 53, Δx = 31.25 m, Δz = 25 m → 100 m
#       `fallback` (4, 4, 8): 96 × 96 × 60, Δx = 62.5 m, Δz = 50 m uniform
#       `coarse`   (16, 16, 8): 24 × 24 × 38 (the test configuration)
# hours simulated hours (default 24)
# D     large-scale divergence of the subsidence, s⁻¹ (default 3e-6; 0 for
#       the sensitivity run)
#
# Writes examples/urban/output/berlin_parent_<grid>[_D0].csv (one row per
# output: z_i, surface fluxes, the ledger terms), the θ profiles as
# berlin_parent_<grid>_theta.csv, and the figures
# docs/src/assets/berlin/parent_theta_zi.svg (θ(z, t) with z_i(t)) and
# parent_ledger.svg (the air-heat ledger terms) for the default D.

using Aeolus, Printf
isdefined(Main, :BerlinCase) || include(joinpath(@__DIR__, "BerlinCase.jl"))
include(joinpath(@__DIR__, "..", "viz", "apng.jl"))
include(joinpath(@__DIR__, "..", "sbl", "svgplot.jl"))
const BC = Main.BerlinCase

grid = length(ARGS) >= 1 ? ARGS[1] : "default"
hours = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 24.0
D = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 3e-6
refinement, vertical = grid == "default" ? ((2, 2, 4), :stretched) :
                       grid == "fallback" ? ((4, 4, 8), :uniform) :
                       grid == "coarse" ? ((16, 16, 8), :stretched) :
                       error("grid must be default, fallback or coarse")
tag = grid * (D == 0 ? "_D0" : "")

child = BC.CaseConfig(; verbose = false, output_path = nothing)
pc = BC.ParentConfig(child; refinement = refinement, vertical = vertical, hours = hours,
                     divergence = D, output_period = 1800.0)
tb = @elapsed p = BC.build_parent(pc)
@printf("build %.1f s\n", tb)
t = @elapsed BC.run_parent!(p)
m = p.mesh
@printf("%s: %d × %d × %d, %d steps in %.0f s wall = %.1f s per simulated hour; dt %.2f–%.2f s (mean %.2f), %d retries\n",
        tag, m.Nx, m.Ny, m.Nz, p.nsteps, t, t / hours, minimum(p.dt_hist), maximum(p.dt_hist),
        sum(p.dt_hist) / p.nsteps, p.retries)
println("timing: ", join((@sprintf("%s %.0f s", k, v) for (k, v) in p.timing), ", "))
print(BC.heat_report(p.heat; layers = (:column, (0.0, 100.0), (0.0, 1000.0))))
print(BC.heat_report(p.heat; window = (20, 5), layers = (:column, (0.0, 100.0))))

# --- tables -------------------------------------------------------------------
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
keys_ = [(:surface_flux, :tiles), (:column_tendency, :column_radiation), (:dynamics, :subsidence),
         (:dynamics, :nudging), (:dynamics, :damping), (:dynamics, :top_flux),
         (:dynamics, :transport), (:dynamics, :redistribution)]
open(joinpath(outdir, "berlin_parent_$tag.csv"), "w") do io
    println(io, "# Berlin parent ($tag, D = $D s^-1): one row per output; ledger terms cumulative J/m2 (column)")
    println(io, "t_s,hour_LT,zi_min_flux_m,zi_parcel_m,H_cum_J_m2,LE_cum_J_m2,sw_down,lw_down,storage_J_m2,",
            join(("$(a)_$(b)" for (a, b) in keys_), ","), ",closure_J_m2")
    for r in p.diag
        @printf(io, "%.0f,%.3f,%.1f,%.1f,%.6e,%.6e,%.2f,%.2f,%.6e", r.t, r.hour, r.zi, r.zi_parcel,
                r.sfc[:H], r.sfc[:LE], r.sw_down, r.lw_down, r.heat.storage)
        for k in keys_
            @printf(io, ",%.6e", get(r.heat.terms, k, 0.0))
        end
        @printf(io, ",%.3e\n", r.heat.closure)
    end
end
zc = BC.level_centres(m)
open(joinpath(outdir, "berlin_parent_$(tag)_theta.csv"), "w") do io
    println(io, "hour_LT,", join((@sprintf("z%.1f", z) for z in zc), ","))
    for r in p.diag
        println(io, @sprintf("%.3f", r.hour), ",", join((@sprintf("%.4f", x) for x in r.θ), ","))
    end
end

# --- figures (default subsidence only) ----------------------------------------
D == 0 && exit()
assets = joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "berlin")
mkpath(assets)
th = [r.t / 3600 for r in p.diag]
lt(h) = mod(6 + h, 24)

# θ(z, t) as a time-height raster with z_i(t) on top
zmax = 2500.0
zg = collect(range(12.5, zmax; length = 200))
interp(r, z) = (k = clamp(searchsortedlast(zc, z), 1, length(zc) - 1);
                r.θ[k] + (r.θ[k+1] - r.θ[k]) * (z - zc[k]) / (zc[k+1] - zc[k]))
Z = [interp(r, z) for z in reverse(zg), r in p.diag]          # rows: top → bottom
θlo, θhi = floor(minimum(Z)), ceil(maximum(Z))
rgb = scalar_to_rgb(Z, θlo, θhi; upscale = 1)
png = tempname() * ".png"
write_png(png, rgb)
const _B64 = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
function b64(data::Vector{UInt8})
    out = IOBuffer()
    for i in 1:3:length(data)
        n = min(3, length(data) - i + 1)
        v = UInt32(data[i]) << 16 | (n > 1 ? UInt32(data[i+1]) << 8 : 0x0) | (n > 2 ? UInt32(data[i+2]) : 0x0)
        for k in 0:3
            print(out, k <= n ? _B64[(v >> (18 - 6k)) & 0x3f + 1] : '=')
        end
    end
    return String(take!(out))
end
W, H = 900, 520
x0, y0, pw, ph = 80, 50, 700, 380
X(h) = x0 + h / hours * pw
Y(z) = y0 + ph - z / zmax * ph
io = IOBuffer()
println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif" font-size="13">""")
println(io, """<rect width="$W" height="$H" fill="white"/>""")
println(io, """<text x="$(W/2)" y="28" font-size="16" text-anchor="middle">Berlin parent, 29/30 June 2019: plane-mean potential temperature and boundary-layer height</text>""")
println(io, """<image x="$x0" y="$y0" width="$pw" height="$ph" preserveAspectRatio="none" style="image-rendering:pixelated" href="data:image/png;base64,$(b64(read(png)))"/>""")
println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
pts(f) = join(("$(X(h)),$(Y(min(f(r), zmax)))" for (h, r) in zip(th, p.diag)), " ")
println(io, """<polyline fill="none" stroke="white" stroke-width="2.5" points="$(pts(r -> r.zi_parcel))"/>""")
println(io, """<polyline fill="none" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,3" points="$(pts(r -> r.zi))"/>""")
for h in 0:3:hours
    println(io, """<line x1="$(X(h))" y1="$(y0+ph)" x2="$(X(h))" y2="$(y0+ph+5)" stroke="#333"/>""")
    println(io, """<text x="$(X(h))" y="$(y0+ph+20)" text-anchor="middle">$(@sprintf("%02d", round(Int, lt(h))))</text>""")
end
for z in 0:500:zmax
    println(io, """<line x1="$(x0-5)" y1="$(Y(z))" x2="$x0" y2="$(Y(z))" stroke="#333"/>""")
    println(io, """<text x="$(x0-8)" y="$(Y(z)+4)" text-anchor="end">$(round(Int, z))</text>""")
end
println(io, """<text x="$(x0+pw/2)" y="$(y0+ph+40)" text-anchor="middle">local time (CEST)</text>""")
println(io, """<text x="22" y="$(y0+ph/2)" text-anchor="middle" transform="rotate(-90 22,$(y0+ph/2))">height (m)</text>""")
stops = join(("<stop offset=\"$(s)\" stop-color=\"rgb$(Int.(_viridis_apng(s)))\"/>" for s in 0:0.125:1), "")
println(io, """<defs><linearGradient id="vir" x1="0" y1="1" x2="0" y2="0">$stops</linearGradient></defs>""")
println(io, """<rect x="$(x0+pw+30)" y="$y0" width="18" height="$ph" fill="url(#vir)" stroke="#333"/>""")
println(io, """<text x="$(x0+pw+54)" y="$(y0+10)">$(round(Int, θhi)) K</text>""")
println(io, """<text x="$(x0+pw+54)" y="$(y0+ph)">$(round(Int, θlo)) K</text>""")
println(io, """<text x="$(x0+pw+30)" y="$(y0+ph/2)" transform="rotate(-90 $(x0+pw+24),$(y0+ph/2))" text-anchor="middle">θ</text>""")
println(io, """<line x1="$x0" y1="$(H-28)" x2="$(x0+24)" y2="$(H-28)" stroke="white" stroke-width="2.5"/><rect x="$(x0-2)" y="$(H-36)" width="28" height="16" fill="none" stroke="#999"/>""")
println(io, """<text x="$(x0+32)" y="$(H-24)">z_i, parcel method (as for the sounding)</text>""")
println(io, """<line x1="$(x0+340)" y1="$(H-28)" x2="$(x0+364)" y2="$(H-28)" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,3"/>""")
println(io, """<text x="$(x0+372)" y="$(H-24)">z_i, minimum of the total heat flux</text>""")
println(io, "</svg>")
write(joinpath(assets, "parent_theta_zi.svg"), take!(io))

# the air-heat ledger: cumulative terms (column) in MJ/m²
fig = svg_figure(900, 480, "Berlin parent, 29/30 June 2019: heat of the air (0-3 km column), cumulative")
cols = (1, 2, 3, 4, 5, 7, 8)
names = ["storage", "surface (tiles)", "column radiation", "subsidence", "nudging", "θ damping", "top flux"]
series = vcat([[r.heat.storage for r in p.diag]], [[get(r.heat.terms, k, 0.0) for r in p.diag] for k in keys_[1:6]])
lo = minimum(minimum.(series)) / 1e6; hi = maximum(maximum.(series)) / 1e6
yl = (floor(lo), ceil(hi))
svg_panel(fig, 90, 50, 620, 360; xlim = (0, hours), ylim = yl, xticks = 0:3:hours,
          yticks = range(yl[1], yl[2]; length = 6), xlabel = "hours from 06 LT",
          ylabel = "MJ m⁻²", legend_x = 640,
          curves = [(th, s ./ 1e6, cols[i]) for (i, s) in enumerate(series)],
          legend = [(names[i], cols[i], "") for i in eachindex(names)])
svg_write(fig, joinpath(assets, "parent_ledger.svg"))
println("figures: ", joinpath(assets, "parent_theta_zi.svg"), ", ", joinpath(assets, "parent_ledger.svg"))
