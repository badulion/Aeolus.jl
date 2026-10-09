# =========================================================================
# Reading the run folders of the `sbl` run type for the analysis scripts
# of examples/sbl/ (EX-b, #439). Nothing is integrated here: a run is made
# with `julia --project -m Aeolus run examples/sbl/configs/<config>.toml`,
# and these functions read what it wrote:
#
#   run.toml                  the resolved config ([config])
#   output/statistics.toml    the averaged profiles and scalars
#   output/output.nc          the time series ([averaging] series_interval)
#   output/towers.csv         the tower bins ([towers])
#
# and the reference data of examples/sbl/data/.
# =========================================================================

using TOML
# The CSV readers are the run type's own (one copy, in src/sbl/case.jl).
using Aeolus.StableBoundaryLayer: sbl_read_csv, sbl_csv_value, SBL_SERIES
using Aeolus.NCIO: NCDataset

const SBL_DATA = joinpath(@__DIR__, "data")
const SBL_OUT = joinpath(@__DIR__, "output")

"The rows of a CSV as NamedTuples keyed by its columns (empty if the file does not exist)."
function csv_rows(path)
    isfile(path) || return NamedTuple[]
    t = sbl_read_csv(path)
    names = Tuple(Symbol.(t.columns))
    return [NamedTuple{names}(Tuple(r)) for r in t.rows]
end

"""
    series_rows(path) -> Vector{NamedTuple}

The time series of an `sbl` run's `output.nc` as one NamedTuple per
sample, keyed `t` (s) and the variables of `SBL_SERIES` (`u_star`,
`w_theta`, `H`, `h`, `S_max`, `z_jet`, `theta_screen`, `U_screen`, `T_s`,
`z_inv`, `ls_heat`); empty if there is no file.
"""
function series_rows(path)
    isfile(path) || return NamedTuple[]
    return NCDataset(path) do ds
        names = (:t, (Symbol(s[1]) for s in SBL_SERIES)...)
        cols = [ds["time_series"].var[:], (ds[s[1]].var[:] for s in SBL_SERIES)...]
        [NamedTuple{names}(Tuple(c[i] for c in cols)) for i in eachindex(cols[1])]
    end
end

"""
    sbl_run(folder) -> (; path, name, config, stats, series, towers)

What one `sbl` run folder holds: its resolved `config` (a Dict, as in
run.toml), `stats` (statistics.toml as a NamedTuple; vectors for the
profiles), the `series` and `towers` rows (NamedTuples).
"""
function sbl_run(folder::AbstractString)
    rt = TOML.parsefile(joinpath(folder, "run.toml"))
    cfg = rt["config"]
    p = get(get(cfg, "output", Dict()), "path", "none")
    out = p isa AbstractString && p != "none" ? p : joinpath(folder, "output")
    sp = joinpath(out, "statistics.toml")
    isfile(sp) || error("$folder: no output/statistics.toml (has the run finished?)")
    st = TOML.parsefile(sp)
    stats = NamedTuple{Tuple(Symbol.(keys(st)))}(Tuple(values(st)))
    return (path = folder, name = get(get(cfg, "run", Dict()), "name", basename(folder)), config = cfg,
            stats = stats, series = series_rows(joinpath(out, "output.nc")),
            towers = csv_rows(joinpath(out, "towers.csv")))
end

"""
    sbl_runs(args) -> Vector

The run folders named on the command line (`args`), read with
[`sbl_run`](@ref); an error with the usage line when there are none.
"""
function sbl_runs(args, usage)
    isempty(args) && error("usage: $usage")
    return [sbl_run(a) for a in args]
end

"The config value `section.key` of a run."
function cfgval(r, key)
    s, k = split(key, '.')
    return r.config[s][k]
end
