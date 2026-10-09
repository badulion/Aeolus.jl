# Surface diurnal-cycle diagnostics for the Alexanderplatz run — NO PHYSICS.
#
# Reads the NetCDF of a berlin_summer.toml run (`julia --project -m Aeolus run
# examples/urban/configs/berlin_summer.toml`) and prints, per
# surface class, the class-mean surface temperature cycle: maximum and
# night minimum (with their local hour), their difference, and the nocturnal
# cooling from sunset to the night minimum (the minimum after sunset —
# the first records are the 285 K cold start, not a night). If the file carries the soil fields
# (`T_soil1`, `G_soil`) their class means are reported too. Pass a second
# file to compare two runs (e.g. before/after a physics change); the plot
# then shows both.
#
#   julia --project=. examples/urban/alexanderplatz_surface_diagnostics.jl \
#         [run.nc] [reference.nc]
#
# Writes `output/surface_diurnal.svg`.

using Aeolus: NCDataset
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "output")
const CLASSES = ((1, "street"), (2, "roof"), (3, "park"), (4, "water"), (5, "wall"))

"Class-mean series of a (x, y, time) variable; NaN cells (no soil) are skipped."
function class_series(A, sfc, c)
    m = sfc .== c
    return [_finite_mean(view(A, :, :, t)[m]) for t in axes(A, 3)]
end
_finite_mean(v) = (w = filter(isfinite, v); isempty(w) ? NaN : sum(w) / length(w))

function summarize(path)
    f = NCDataset(path)
    hour = Float64.(Array(f["hour"]))
    elev = Float64.(Array(f["solar_elevation"]))
    sfc  = round.(Int, Array(f["surface_class"]))
    Ts   = Array(f["Ts"])
    has_soil = haskey(f, "T_soil1")
    # local clock hour of each record, continuous across midnight
    clock = copy(hour)
    for i in 2:length(clock)
        while clock[i] < clock[i-1]; clock[i] += 24; end
    end
    iset = findlast(i -> elev[i] > 0 && i < length(elev) && elev[i+1] <= 0, eachindex(elev))
    @printf("\n%s  (%d records, %.1f–%.1f h, sunset record at %.1f h)\n",
            path, length(hour), clock[1], clock[end], iset === nothing ? NaN : clock[iset])
    @printf("  %-7s %6s %6s %6s %6s %7s %8s %8s", "class", "Tmax", "@h", "Tnight", "@h",
            "range", "ΔT_night", "K/h")
    has_soil && @printf(" %8s %8s %8s", "Tsoil1 rng", "G max", "G min")
    println()
    series = Dict{String, Vector{Float64}}()
    rows = Dict{String, NTuple{4, Float64}}()
    for (c, name) in CLASSES
        count(==(c), sfc) == 0 && continue
        s = class_series(Ts, sfc, c)
        series[name] = s
        imax = argmax(s)
        # night minimum: after sunset (the first records are the cold start)
        i0 = iset === nothing ? 1 : iset
        imin = i0 - 1 + argmin(s[i0:end])
        # nocturnal cooling: from sunset to the minimum after it
        if iset !== nothing
            jmin = iset - 1 + argmin(s[iset:end])
            dT = s[iset] - s[jmin]
            rate = dT / max(clock[jmin] - clock[iset], eps())
        else
            dT = rate = NaN
        end
        @printf("  %-7s %6.2f %6.1f %6.2f %6.1f %7.2f %8.2f %8.2f", name, s[imax], clock[imax] % 24,
                s[imin], clock[imin] % 24, s[imax] - s[imin], dT, rate)
        if has_soil && c != 4
            t1 = class_series(Array(f["T_soil1"]), sfc, c)
            G = class_series(Array(f["G_soil"]), sfc, c)
            @printf(" %8.2f %8.1f %8.1f", maximum(t1) - minimum(t1), maximum(G), minimum(G))
        end
        println()
        rows[name] = (s[imax], s[imin], s[imax] - s[imin], dT)
    end
    close(f)
    return (; clock, series, rows)
end

paths = isempty(ARGS) ? [joinpath(OUT, "alexanderplatz.nc")] : ARGS
runs = [summarize(p) for p in paths]

if length(runs) == 2
    println("\nDifference (first − second):")
    @printf("  %-7s %7s %7s %7s %9s\n", "class", "ΔTmax", "ΔTnight", "Δrange", "ΔΔT_night")
    for (_, name) in CLASSES
        haskey(runs[1].rows, name) && haskey(runs[2].rows, name) || continue
        a, b = runs[1].rows[name], runs[2].rows[name]
        @printf("  %-7s %7.2f %7.2f %7.2f %9.2f\n", name, (a .- b)...)
    end
end

mkpath(OUT)
names = [n for (c, n) in CLASSES if c != 4 && haskey(runs[1].series, n)]
ys = [runs[1].series[n] for n in names]
labels = copy(names)
if length(runs) == 2
    append!(ys, [runs[2].series[n] for n in names])
    append!(labels, [n * " (ref)" for n in names])
end
svg = joinpath(OUT, "surface_diurnal.svg")
save_multiline(svg, runs[1].clock, ys, labels;
               xlabel = "local hour", ylabel = "class-mean surface temperature (K)",
               title = "Alexanderplatz surface temperature", width = 800, height = 480)
println("\nWrote ", svg)
