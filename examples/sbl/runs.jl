# =========================================================================
# Reading the run folders of the `sbl` run type for the analysis scripts
# of examples/sbl/ (EX-b, #439). Nothing is integrated here: a run is made
# with `julia --project -m Aeolus run examples/sbl/configs/<config>.toml`,
# and these functions read what it wrote:
#
#   run.toml                  the resolved config ([config])
#   output/statistics.toml    the averaged profiles and scalars
#   output/series.csv         the time series ([averaging] series_interval)
#   output/towers.csv         the tower bins ([towers])
#
# and the reference data of examples/sbl/data/.
# =========================================================================

using TOML

const SBL_DATA = joinpath(@__DIR__, "data")
const SBL_OUT = joinpath(@__DIR__, "output")

"""
    read_table(path) -> (; header, columns, rows)

A CSV of examples/sbl/data/ or of a run's output: the leading `#` lines
(provenance) in `header`, the first other line the column names, the rest
the rows (numbers parsed where possible).
"""
function read_table(path)
    header = String[]; columns = String[]; rows = Vector{Any}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        if startswith(line, "#")
            push!(header, line)
        elseif isempty(columns)
            columns = String.(strip.(split(line, ",")))
        else
            push!(rows, Any[something(tryparse(Float64, strip(x)), String(strip(x))) for x in split(line, ",")])
        end
    end
    return (header = header, columns = columns, rows = rows)
end

"The value of column `col` in the row whose first column is `key`."
function table_value(tab, key, col)
    j = findfirst(==(col), tab.columns)
    for r in tab.rows
        r[1] == key && return r[j]
    end
    throw(KeyError(key))
end

"The rows of a CSV as NamedTuples keyed by its columns (empty if the file does not exist)."
function csv_rows(path)
    isfile(path) || return NamedTuple[]
    t = read_table(path)
    names = Tuple(Symbol.(t.columns))
    return [NamedTuple{names}(Tuple(r)) for r in t.rows]
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
            stats = stats, series = csv_rows(joinpath(out, "series.csv")),
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
