# Figures of the Berlin parent run (N-parent-e, #259), drawn from the
# output of a run of the parent alone (NEST-p3 #483: a domain of the
# model's core; parent_output.jl reads it):
#
#     <run folder>/output/parent.nc   (θ, qv and heat-flux profiles, the energy budget)
#
#   docs/src/assets/berlin/parent_theta_zi.svg  θ(z, t) with z_i(t)
#   docs/src/assets/berlin/parent_ledger.svg    the air-heat ledger terms
#
# Run the parent first, then draw (reads the tables only, runs nothing):
#
#     julia --project -m Aeolus run examples/urban/berlin_parent.toml
#     julia --project examples/urban/berlin_parent_figures.jl <run folder> ["label"]

using Aeolus, Printf
isdefined(Main, :write_png) || include(joinpath(@__DIR__, "..", "viz", "apng.jl"))
isdefined(Main, :svg_figure) || include(joinpath(@__DIR__, "..", "sbl", "svgplot.jl"))
isdefined(Main, :read_parent_output) || include(joinpath(@__DIR__, "parent_output.jl"))

const _B64 = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
function _b64(data::Vector{UInt8})
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

"""
    parent_figures(run; assets, label)

Draw `parent_theta_zi.svg` and `parent_ledger.svg` into `assets` from
the parent's output in the run folder `run` ([`read_parent_output`](@ref)).
`label` (e.g. "96 × 96 × 60, D = 0") goes into the titles.
"""
function parent_figures(run; assets = joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "berlin"),
                        label = basename(rstrip(run, '/')))
    P = read_parent_output(run)
    th = P.t ./ 3600
    hours = th[end]
    zi, zip_ = P.zi_min_flux, P.zi_parcel
    zc = P.z
    θrows = [P.θ[:, n] for n in eachindex(P.t)]
    mkpath(assets)
    lt(h) = mod(6 + h, 24)

    # θ(z, t) as a time-height raster with z_i(t) on top; one raster column
    # per output, so the cells span the output interval
    zmax = 2500.0
    zg = collect(range(12.5, zmax; length = 200))
    interp(θ, z) = (k = clamp(searchsortedlast(zc, z), 1, length(zc) - 1);
                    θ[k] + (θ[k+1] - θ[k]) * (z - zc[k]) / (zc[k+1] - zc[k]))
    Z = [interp(θ, z) for z in reverse(zg), θ in θrows]       # rows: top → bottom
    θlo, θhi = floor(minimum(Z)), ceil(maximum(Z))
    png = tempname() * ".png"
    write_png(png, scalar_to_rgb(Z, θlo, θhi; upscale = 1))
    W, H = 900, 520
    x0, y0, pw, ph = 80, 50, 700, 380
    dh = hours / (length(th) - 1)
    X(h) = x0 + (h + dh / 2) / (hours + dh) * pw                 # cell centres
    Y(z) = y0 + ph - z / zmax * ph
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif" font-size="13">""")
    println(io, """<rect width="$W" height="$H" fill="white"/>""")
    println(io, """<text x="$(W/2)" y="28" font-size="15" text-anchor="middle">Berlin parent, 29/30 June 2019 ($label): plane-mean θ and boundary-layer height</text>""")
    println(io, """<image x="$x0" y="$y0" width="$pw" height="$ph" preserveAspectRatio="none" style="image-rendering:pixelated" href="data:image/png;base64,$(_b64(read(png)))"/>""")
    println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
    pts(v) = join(("$(X(h)),$(Y(min(z, zmax)))" for (h, z) in zip(th, v)), " ")
    println(io, """<polyline fill="none" stroke="black" stroke-width="4" points="$(pts(zip_))"/>""")
    println(io, """<polyline fill="none" stroke="white" stroke-width="2" points="$(pts(zip_))"/>""")
    println(io, """<polyline fill="none" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="5,3" points="$(pts(zi))"/>""")
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
    println(io, """<text x="$(x0+pw+62)" y="$(y0+ph/2)" transform="rotate(-90 $(x0+pw+62),$(y0+ph/2))" text-anchor="middle">θ</text>""")
    println(io, """<rect x="$(x0-2)" y="$(H-36)" width="28" height="16" fill="#3b528b"/>""")
    println(io, """<line x1="$x0" y1="$(H-28)" x2="$(x0+24)" y2="$(H-28)" stroke="black" stroke-width="4"/><line x1="$x0" y1="$(H-28)" x2="$(x0+24)" y2="$(H-28)" stroke="white" stroke-width="2"/>""")
    println(io, """<text x="$(x0+32)" y="$(H-24)">z_i, parcel method (as for the sounding)</text>""")
    println(io, """<line x1="$(x0+340)" y1="$(H-28)" x2="$(x0+364)" y2="$(H-28)" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="5,3"/>""")
    println(io, """<text x="$(x0+372)" y="$(H-24)">z_i, minimum of the total heat flux</text>""")
    println(io, "</svg>")
    write(joinpath(assets, "parent_theta_zi.svg"), take!(io))

    # the energy budget of the air: cumulative terms (column) in MJ/m², the
    # large-scale forcing one term (geostrophic work, nudging, subsidence, damping)
    fig = svg_figure(900, 480, "Berlin parent ($label): heat of the air, 0-3 km column, cumulative")
    names = ["storage", "surface (tiles)", "column radiation", "large-scale forcing", "transport", "redistribution"]
    keys_ = ["storage", "surface_flux__tiled_surface", "surface_flux__mean_column_radiation",
             "body_force__large_scale_forcing", "dynamics__transport", "dynamics__redistribution"]
    cols = (1, 2, 3, 5, 8, 6)
    th = P.ts ./ 3600
    series = [get(P.energy, k, zeros(length(P.ts))) ./ 1e6 for k in keys_]
    keep = [i for i in eachindex(series) if i == 1 || maximum(abs, series[i]) > 0]   # D = 0: no subsidence curve
    lo = minimum(minimum.(series)); hi = maximum(maximum.(series))
    step = hi - lo > 12 ? 4.0 : hi - lo > 6 ? 2.0 : 1.0
    yl = (step * floor(lo / step), step * ceil(hi / step))
    svg_panel(fig, 90, 50, 620, 360; xlim = (0, hours), ylim = yl, xticks = 0:3:hours,
              yticks = yl[1]:step:yl[2], xlabel = "hours from 06 LT",
              ylabel = "MJ m⁻²", legend_x = 640,
              curves = [(th, series[i], cols[i]) for i in keep],
              legend = [(names[i], cols[i], "") for i in keep])
    svg_write(fig, joinpath(assets, "parent_ledger.svg"))
    println("figures: ", joinpath(assets, "parent_theta_zi.svg"), ", ", joinpath(assets, "parent_ledger.svg"))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("usage: berlin_parent_figures.jl <run folder> [label]")
    parent_figures(ARGS[1]; label = length(ARGS) >= 2 ? ARGS[2] : basename(rstrip(ARGS[1], '/')))
end
