# Readers of a finished run folder (EX-d, #441), for the plotting scripts
# beside the example configs. They read files only (run.toml,
# diagnostics.csv, diagnostics.nc, output/output.nc, ...); no simulation code runs.
#
#   include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))
#   d = read_diagnostics_csv(folder)          # Dict column => Vector{Float64}
#   t, ke = diagnostic_series(d, "ke_mean")    # the rows that hold a value
#   dom = recorded_domain(folder)              # the model domain's config (grid, dynamics, ...)
#   z, P = read_profile(folder, "theta")       # levels, profile per costly row (Nz × nrec)
#   F, t = read_last_record(output_file(folder))  # last record of output/output.nc: Dict name => Array, time

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
recorded_config(folder) = TOML.parsefile(joinpath(folder, "run.toml"))["config"]

"""
    recorded_domain(folder, id = "main")

The resolved configuration of the model's domain `id` (a
`[[config.domain]]` table of `run.toml`: `grid`, `dynamics`, `tracer`,
...); the run-wide sections (`time`, `output`, ...) stay in
[`recorded_config`](@ref).
"""
recorded_domain(folder, id = "main") = only(d for d in recorded_config(folder)["domain"] if d["id"] == id)

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

"""
    output_file(folder) -> path

The NetCDF output of the run (ONE-c): `[output] netcdf` (default
`output.nc`) in `[output] path`, or in the run folder's `output/` when
the path is `"none"`.
"""
function output_file(folder)
    o = get(recorded_config(folder), "output", Dict{String,Any}())
    p = get(o, "path", "none")
    dir = p isa AbstractString && p != "none" ? p : joinpath(folder, "output")
    f = joinpath(dir, get(o, "netcdf", "output.nc"))
    isfile(f) || error("$folder wrote no $f (set [output] variables)")
    return f
end

"""
    read_last_record(path) -> (F, t)

The last record of the output file `path` (the last entry of its
unlimited `time` dimension): every record variable by name, without the
time axis (`Nx × Ny × Nz` for a 3-D field, `Nx × Ny` for a surface map,
`Nz` for a profile), and the simulation time `t` (s) of that record.
The scalars on the `time_series` dimension are not records.
"""
function read_last_record(path)
    ds = NCDataset(path)
    try
        n = Int(ds.dim["time"])
        n > 0 || error("$path holds no records (set [output] variables and cadence)")
        out = Dict{String,Array}()
        for k in keys(ds)
            dims = Aeolus.NCIO.NCDatasets.dimnames(ds[k])
            (length(dims) > 1 && last(dims) == "time") || continue
            out[k] = Array(ds[k][ntuple(_ -> Colon(), length(dims) - 1)..., n])
        end
        return out, Float64(ds["time"][n])
    finally
        close(ds)
    end
end
