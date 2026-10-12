# The surface-mounted cube, the figure: the mean centre-line floor velocity
# of the runs of surface_mounted_cube_closures.toml (one run folder
# per closure) with the reattachment and front-separation points of
# Martinuzzi & Tropea (1993). Reads output/floor_u.csv, output/results.toml
# and run.toml only.
#
#   julia --project -m Aeolus run examples/surface_mounted_cube/surface_mounted_cube_closures.toml
#   julia --project=. examples/surface_mounted_cube/plot_surface_mounted_cube.jl runs/<date>_surface_mounted_cube_*
#
# Prints x_R and x_F per closure and writes surface_mounted_cube_<N>.svg into
# the output/ of the first run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

const XR_MT93 = 1.612
const XF_MT93 = 1.04
isempty(ARGS) && error("usage: julia --project=. plot_surface_mounted_cube.jl <run folder>...")
folders = [run_folder_arg([a]) for a in ARGS]
col = Dict("smagorinsky" => "#8c8c8c", "tke" => "#1b6ca8", "rsm" => "#c8553d")
lab = Dict("smagorinsky" => "Smagorinsky", "tke" => "TKE", "rsm" => "Reynolds stress")

runs = map(folders) do f
    res = read_results(f)
    rows = [parse.(Float64, split(s, ",")) for s in readlines(joinpath(f, "output", "floor_u.csv"))[2:end]]
    @printf("%-16s x_R/H = %.3f (MT93 %.3f), x_F/H = %.3f (MT93 %.2f); %d steps, %.0f s (%.2f s/step); mean U_b %.4f; force %.4f\n",
            lab[res["closure"]], res["x_R"], XR_MT93, res["x_F"], XF_MT93, res["steps"], res["wall_seconds"],
            res["wall_seconds_per_step"], res["mean_U_b"], res["force"])
    (res = res, x = first.(rows), u = last.(rows))
end
g = recorded_config(folders[1])["grid"]
LX = g["extent"][1]; NH = g["cells_per_h"]
XC0, XC1 = runs[1].res["x_front"], runs[1].res["x_rear"]

W, Hh = 760, 420
l, r, t, b = 70, 190, 30, 55
X(x) = l + (W - l - r) * x / LX
Y(u) = t + (Hh - t - b) * (1 - (u + 0.4) / 1.0)
io = IOBuffer()
println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$Hh" font-family="DejaVu Sans, sans-serif" font-size="12">""")
println(io, """<rect width="$W" height="$Hh" fill="white"/>""")
println(io, """<rect x="$(X(XC0))" y="$t" width="$(X(XC1) - X(XC0))" height="$(Hh - t - b)" fill="#dddddd"/>""")
for u in -0.4:0.2:0.6
    println(io, """<line x1="$l" y1="$(Y(u))" x2="$(W - r)" y2="$(Y(u))" stroke="$(u == 0 ? "#888888" : "#ececec")"/>""")
    println(io, """<text x="$(l - 6)" y="$(Y(u) + 4)" text-anchor="end">$(round(u; digits = 1))</text>""")
end
for xv in 0:1:round(Int, LX)
    println(io, """<text x="$(X(xv))" y="$(Hh - b + 16)" text-anchor="middle">$xv</text>""")
end
println(io, """<rect x="$l" y="$t" width="$(W - l - r)" height="$(Hh - t - b)" fill="none" stroke="black"/>""")
println(io, """<text x="$((W - r + l) / 2)" y="$(Hh - 15)" text-anchor="middle">x / H  (cube: $(XC0)–$(XC1))</text>""")
println(io, """<text transform="translate(22,$((Hh - b + t) / 2)) rotate(-90)" text-anchor="middle">mean u / U_b at z = Δz/2, centre line</text>""")
for (xv, s) in ((XC1 + XR_MT93, "x_R MT93"), (XC0 - XF_MT93, "x_F MT93"))
    println(io, """<line x1="$(X(xv))" y1="$t" x2="$(X(xv))" y2="$(Hh - b)" stroke="black" stroke-dasharray="5,4"/>""")
    println(io, """<text x="$(X(xv) + 3)" y="$(t + 14)" font-size="11">$s</text>""")
end
ly = t + 10
for rr in runs
    m = rr.res["closure"]
    pts = join([@sprintf("%.1f,%.1f", X(x), Y(clamp(u, -0.4, 0.6))) for (x, u) in zip(rr.x, rr.u)], " ")
    println(io, """<polyline fill="none" stroke="$(col[m])" stroke-width="1.8" points="$pts"/>""")
    println(io, """<line x1="$(W - r + 12)" y1="$ly" x2="$(W - r + 36)" y2="$ly" stroke="$(col[m])" stroke-width="1.8"/>""")
    println(io, """<text x="$(W - r + 42)" y="$(ly + 4)">$(lab[m])</text>""")
    println(io, """<text x="$(W - r + 42)" y="$(ly + 19)" font-size="11">x_R/H = $(round(rr.res["x_R"]; digits = 2))</text>""")
    global ly += 44
end
println(io, """<text x="$(W - r + 12)" y="$(ly + 4)" font-size="11">MT93: x_R/H = $(XR_MT93)</text>""")
println(io, "</svg>")
path = plot_path(folders[1], "surface_mounted_cube_$(NH).svg")
write(path, String(take!(io)))
println("wrote ", path)
