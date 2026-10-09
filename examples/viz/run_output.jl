# Readers of a finished run folder (EX-d, #441), for the plotting scripts
# beside the example configs. They read files only (run.toml,
# diagnostics.csv, diagnostics.nc, output/*); no simulation code runs.
#
#   include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))
#   d = read_diagnostics_csv(folder)          # Dict column => Vector{Float64}
#   t, ke = diagnostic_series(d, "ke_mean")    # the rows that hold a value
#   z, P = read_profile(folder, "theta")       # levels, profile per costly row (Nz × nrec)
#   F = read_fields(last_fields_file(folder))  # Dict name => Array{Float32,3}

using TOML
import Aeolus
using Aeolus: NCDataset

"The run folder given on the command line (`ARGS[1]`), checked."
function run_folder_arg(args = ARGS)
    isempty(args) && error("usage: julia --project=. <script>.jl <run folder>")
    f = abspath(args[1])
    isfile(joinpath(f, "run.toml")) || error("$f is not a run folder (no run.toml)")
    return f
end

"The resolved configuration of the run (the `[config]` table of `run.toml`)."
run_config_table(folder) = TOML.parsefile(joinpath(folder, "run.toml"))["config"]

"A path in the run's `output/` directory (made if missing) for a figure."
plot_path(folder, name) = joinpath(mkpath(joinpath(folder, "output")), name)

"`output/results.toml` of the run, parsed."
read_results(folder) = TOML.parsefile(joinpath(folder, "output", "results.toml"))

"The columns of `diagnostics.csv` (header names without units) as Float64 vectors (non-numbers: NaN)."
function read_diagnostics_csv(folder)
    lines = readlines(joinpath(folder, "diagnostics.csv"))
    names = [strip(first(split(h, " ["))) for h in split(lines[1], ',')]
    cols = Dict(n => Float64[] for n in names)
    for l in lines[2:end]
        for (n, v) in zip(names, split(l, ','))
            x = tryparse(Float64, v)
            push!(cols[n], x === nothing ? NaN : x)
        end
    end
    return cols
end

"""
    diagnostic_series(d, name) -> (t, v)

The rows of the column `name` of [`read_diagnostics_csv`](@ref) that hold
a value: the costly diagnostics are written on their cadence only, the
rows between leave them empty.
"""
function diagnostic_series(d, name)
    keep = findall(isfinite, d[name])
    return d["t"][keep], d[name][keep]
end

"The profile `name` of `diagnostics.nc`: its levels and the values (levels × costly rows), and the row times."
function read_profile(folder, name)
    ds = NCDataset(joinpath(folder, "diagnostics.nc"))
    try
        v = ds[name]
        ax = first(Aeolus.NCIO.NCDatasets.dimnames(v))
        return Array(ds[ax][:]), Array(v[:, :]), Array(ds["time_costly"][:])
    finally
        close(ds)
    end
end

"The newest `fields_<step>.nc` the run wrote."
function last_fields_file(folder)
    dir = joinpath(folder, "output")
    fs = sort(filter(f -> startswith(f, "fields_") && endswith(f, ".nc"), readdir(dir)))
    isempty(fs) && error("$dir holds no fields_*.nc (set [output] fields)")
    return joinpath(dir, fs[end])
end

"Every variable of a fields file, by name, and its time attribute."
function read_fields(path)
    ds = NCDataset(path)
    try
        out = Dict{String,Array{Float32,3}}()
        for k in keys(ds)
            k in ("x", "y", "z") && continue
            out[k] = Array(ds[k][:, :, :])
        end
        return out, ds.attrib["t"]
    finally
        close(ds)
    end
end
