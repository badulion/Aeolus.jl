# =========================================================================
# Within-canopy wind and temperature profiles — Phase 48 (#25)
#
# The 1-D PlantCanopy column for three published canopies
# (examples/validation/configs/canopy_profiles.toml, the `plant_canopy` run
# type) against the observations: this script reads the runs'
# output/results.toml and writes a figure comparing model and observations:
#
#   * wind tunnel, waving-wheat model canopy (Brunet, Finnigan & Raupach
#     1994, BLM 70, 95-132), as re-plotted by Huang, Cassiani & Albertson
#     (2008, AMS extended abstract 2.2, Fig. 2);
#   * CHATS walnut orchard with and without leaves (Patton et al. 2011,
#     BAMS 92, 593-611; Dupont & Patton 2012, AFM 157, 11-29), near-neutral;
#   * Scots pine forest, Hyytiälä (Launiainen et al. 2007, Tellus 59B,
#     919-936): U/u* (near-neutral) and the day / night in-canopy air
#     temperature from a coupled leaf-energy-balance + scalar column.
#
# The data and how they were read from the papers are in
# test/data/canopy_profiles.jl; test/era5_canopy_profiles.jl asserts the
# same comparisons. Nothing is fitted: C_d and the leaf-area profiles are
# the published ones (C_d = 0.2, the PlantCanopy default, for CHATS, for
# which none is published), the mixing length is Harman & Finnigan's with
# β = 0.3.
#
#   julia --project -m Aeolus run examples/validation/configs/canopy_profiles.toml
#   julia --project examples/validation/canopy_profiles.jl [runs-root]
#
# writes examples/validation/output/canopy_profiles.svg
# =========================================================================

using Printf
include(joinpath(@__DIR__, "read_runs.jl"))
# the observations (numbers digitised from the papers, with their provenance)
include(joinpath(@__DIR__, "..", "..", "test", "data", "canopy_profiles.jl"))

const OBS = canopy_profile_observations()

interp(zs, vs, z) = (i = clamp(searchsortedlast(zs, z), 1, length(zs) - 1);
                     t = (z - zs[i]) / (zs[i+1] - zs[i]); (1 - t) * vs[i] + t * vs[i+1])

R = Dict(l => r.res for (l, r) in member_runs(runs_root(), validation_config("canopy_profiles")))
f64(r, k) = Float64.(r[k])

# ---- panels -----------------------------------------------------------------
panels = []

wt = OBS.wind_tunnel
m, m2 = R["wind_tunnel"], R["wind_tunnel_cd_1.2"]
push!(panels, (title = "Wind tunnel (Brunet 1994)", xlabel = "U / U_h", ymax = 2.0,
               lines = [(f64(m, "z") ./ wt.h, f64(m, "u") ./ m["U_h"], "#1f5fbf", "model C_d = $(wt.C_d)"),
                        (f64(m2, "z") ./ wt.h, f64(m2, "u") ./ m2["U_h"], "#999999", "model C_d = $(wt.C_d_alt)")],
               points = [(wt.z, wt.U_over_Uh, "#111111", "obs.")]))

for (ds, label) in ((OBS.chats_leaves, "chats_leaves"), (OBS.chats_noleaves, "chats_noleaves"))
    mc = R[label]
    @printf("%-36s d = %.2f m (obs %.1f m)\n", ds.name, mc["d"], ds.d)
    push!(panels, (title = replace(ds.name, "CHATS walnut orchard" => "CHATS"), xlabel = "U / U_h",
                   ymax = 2.9,
                   lines = [(f64(mc, "z") ./ ds.h, f64(mc, "u") ./ mc["U_h"], "#1f5fbf", "model")],
                   points = [(ds.z, ds.U_over_Uh, "#111111", "obs. (NN)")]))
end

hy = OBS.hyytiala
mh = R["hyytiala"]
@printf("%-36s d/h = %.3f (obs %.2f)\n", hy.name, mh["d"] / hy.h, hy.d_over_h)
push!(panels, (title = "Hyytiälä pine", xlabel = "U / u*", ymax = 1.5,
               lines = [(f64(mh, "z") ./ hy.h, f64(mh, "u") ./ mh["u_star"], "#1f5fbf", "model")],
               points = [(hy.z, hy.U_over_ustar.NN, "#111111", "obs. (NN)")]))

day, night = R["hyytiala_day"], R["hyytiala_night"]
zT = hy.T_z_m[1:3] ./ hy.h
push!(panels, (title = "Hyytiälä air temperature", xlabel = "Θ − Θ(16.8 m)  [K]", ymax = 1.2,
               lines = [(f64(day, "z") ./ hy.h, f64(day, "dT"), "#d0542a", "model day"),
                        (f64(night, "z") ./ hy.h, f64(night, "dT"), "#1f5fbf", "model night")],
               points = [(zT, hy.T_minus_ref.US[1:3], "#d0542a", "obs. unstable"),
                         (zT, hy.T_minus_ref.S[1:3], "#1f5fbf", "obs. stable")]))
@printf("Hyytiälä ΔT at 8.4 m: day model %+.2f K (obs %+.2f), night model %+.2f K (obs %+.2f)\n",
        interp(f64(day, "z"), f64(day, "dT"), 8.4), hy.T_minus_ref.US[2],
        interp(f64(night, "z"), f64(night, "dT"), 8.4), hy.T_minus_ref.S[2])

# ---- figure (self-contained SVG) --------------------------------------------
function nice_ticks(lo, hi)
    span = hi - lo
    step = 10.0^floor(log10(span / 4))
    for m in (1, 2, 2.5, 5, 10)
        span / (m * step) <= 5 && (step *= m; break)
    end
    return collect(ceil(lo / step) * step:step:hi + 1e-12)
end
fmt(x) = abs(x - round(x)) < 1e-9 ? string(round(Int, x)) : string(round(x; digits = 2))

function write_figure(path, panels)
    pw, ph, gap, left, top, bottom = 230, 330, 40, 55, 50, 120
    W = left + length(panels) * (pw + gap); H = top + ph + bottom
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="DejaVu Sans, sans-serif">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")
    for (k, P) in enumerate(panels)
        x0 = left + (k - 1) * (pw + gap); y0 = top
        xs = vcat([l[2] for l in P.lines]..., [p[2] for p in P.points]...)
        xlo, xhi = extrema(xs); pad = 0.06 * (xhi - xlo)
        xlo -= pad; xhi += pad
        X(v) = x0 + (v - xlo) / (xhi - xlo) * pw
        Y(z) = y0 + ph - z / P.ymax * ph
        println(io, """<rect x="$x0" y="$y0" width="$pw" height="$ph" fill="none" stroke="#333"/>""")
        println(io, """<line x1="$x0" y1="$(Y(1.0))" x2="$(x0 + pw)" y2="$(Y(1.0))" stroke="#bbb" stroke-dasharray="4 3"/>""")
        for t in nice_ticks(xlo, xhi)
            println(io, """<line x1="$(X(t))" y1="$(y0 + ph)" x2="$(X(t))" y2="$(y0 + ph + 4)" stroke="#333"/>""")
            println(io, """<text x="$(X(t))" y="$(y0 + ph + 17)" font-size="11" text-anchor="middle">$(fmt(t))</text>""")
        end
        for t in nice_ticks(0.0, P.ymax)
            println(io, """<line x1="$(x0 - 4)" y1="$(Y(t))" x2="$x0" y2="$(Y(t))" stroke="#333"/>""")
            println(io, """<text x="$(x0 - 7)" y="$(Y(t) + 4)" font-size="11" text-anchor="end">$(fmt(t))</text>""")
        end
        println(io, """<text x="$(x0 + pw / 2)" y="$(y0 + ph + 35)" font-size="13" text-anchor="middle">$(P.xlabel)</text>""")
        println(io, """<text x="$(x0 + pw / 2)" y="$(y0 - 12)" font-size="13" font-weight="bold" text-anchor="middle">$(P.title)</text>""")
        k == 1 && println(io, """<text x="16" y="$(y0 + ph / 2)" font-size="13" text-anchor="middle" transform="rotate(-90 16,$(y0 + ph / 2))">z / h</text>""")
        for (z, v, col, _) in P.lines
            sel = z .<= P.ymax
            pts = join(("$(round(X(v[i]), digits = 2)),$(round(Y(z[i]), digits = 2))" for i in findall(sel)), " ")
            println(io, """<polyline fill="none" stroke="$col" stroke-width="2" points="$pts"/>""")
        end
        for (z, v, col, _) in P.points, i in eachindex(z)
            println(io, """<circle cx="$(X(v[i]))" cy="$(Y(z[i]))" r="4" fill="white" stroke="$col" stroke-width="1.6"/>""")
        end
        ly = y0 + ph + 52
        for (z, v, col, lab) in P.lines
            println(io, """<line x1="$x0" y1="$ly" x2="$(x0 + 22)" y2="$ly" stroke="$col" stroke-width="2"/><text x="$(x0 + 28)" y="$(ly + 4)" font-size="11">$lab</text>""")
            ly += 15
        end
        for (z, v, col, lab) in P.points
            println(io, """<circle cx="$(x0 + 11)" cy="$ly" r="4" fill="white" stroke="$col" stroke-width="1.6"/><text x="$(x0 + 28)" y="$(ly + 4)" font-size="11">$lab</text>""")
            ly += 15
        end
    end
    println(io, "</svg>")
    mkpath(dirname(path))
    write(path, String(take!(io)))
    return path
end

out = write_figure(joinpath(@__DIR__, "output", "canopy_profiles.svg"), panels)
println("wrote ", out)
