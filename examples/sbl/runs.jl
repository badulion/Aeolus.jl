# =========================================================================
# Reading the run folders of the SBL configurations for the analysis
# scripts of examples/sbl/ (ONE-g, #456). Nothing is integrated here: a run
# is made with `julia --project -m Aeolus run examples/sbl/configs/<config>.toml`
# (the model with its [large_scale_forcing], [surface_layer], [sponge] and
# [towers] components), and these functions read what it wrote:
#
#   run.toml                  the resolved config ([config]) and the sessions
#   output/output.nc          the profiles (records, means over [output]
#                             cadence) and the time series (series_cadence)
#   output/results.toml       the checks against the reference data
#   diagnostics.csv           the step count
#
# and the reference data of examples/sbl/data/.
# =========================================================================

using TOML
using Aeolus.NCIO: NCDataset

const SBL_DATA = joinpath(@__DIR__, "data")
const SBL_OUT = joinpath(@__DIR__, "output")

"""
    sbl_read_csv(path) -> (; header, columns, rows)

A case or reference-data CSV: the leading `#` lines are its provenance
header, the first other line the column names, the rest the rows (numbers
parsed where possible).
"""
function sbl_read_csv(path)
    header = String[]; columns = String[]; rows = Vector{Any}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        if startswith(line, "#")
            push!(header, line)
        elseif isempty(columns)
            columns = strip.(split(line, ","))
        else
            push!(rows, [something(tryparse(Float64, strip(x)), strip(x)) for x in split(line, ",")])
        end
    end
    return (header = header, columns = columns, rows = rows)
end

"The value of column `col` in the row whose first column is `key`."
function sbl_csv_value(tab, key, col)
    j = findfirst(==(col), tab.columns)
    for r in tab.rows
        r[1] == key && return r[j]
    end
    throw(KeyError(key))
end

# the output variables of the time series, under the names the scripts use
const SERIES_NAMES = (u_star = "u_star", wθ = "surface_wtheta", H = "surface_heat_flux", h = "bl_height",
                      S_max = "jet_speed", z_jet = "jet_height", θ_screen = "theta_2m", U_screen = "wind_10m",
                      T_s = "surface_temperature", z_inv = "inversion_height", L = "obukhov_length")
# the profiles of the records
const PROFILE_NAMES = (U = "u_profile", V = "v_profile", θ = "theta_profile", S = "wind_speed_profile",
                       τ = "stress_profile", wθ = "heat_flux_profile", V_s = "tower_threshold")

"""
    series_rows(ds) -> Vector{NamedTuple}

The time series of a run's `output.nc` as one NamedTuple per sample:
`t` (s, model time) and the fields of `SERIES_NAMES` the run wrote (NaN
for one it did not).
"""
function series_rows(ds)
    t = ds["time_series"].var[:]
    col(n) = haskey(ds, n) ? Float64.(ds[n].var[:]) : fill(NaN, length(t))
    cols = map(col, values(SERIES_NAMES))
    return [NamedTuple{(:t, keys(SERIES_NAMES)...)}((t[i], (c[i] for c in cols)...)) for i in eachindex(t)]
end

"The profiles of the records with time in (t0, t1], averaged: a NamedTuple of `z` and `PROFILE_NAMES`."
function mean_profiles(ds, t0, t1)
    t = ds["time"].var[:]
    sel = findall(x -> t0 < x <= t1 * (1 + 1e-12), t)
    z = Float64.(ds["z"].var[:])
    prof(n) = haskey(ds, n) && !isempty(sel) ? vec(sum(Float64.(ds[n].var[:, sel]); dims = 2)) ./ length(sel) :
                                                fill(NaN, length(z))
    return merge((z = z,), NamedTuple{keys(PROFILE_NAMES)}(map(prof, values(PROFILE_NAMES))))
end

"""
    sbl_run(folder; window = nothing, last = nothing) -> (; path, name, config, stats, series, results)

What one run folder holds: its resolved `config` (a Dict, as in
run.toml), the `series` (NamedTuples, [`series_rows`](@ref)), the checks of
`results.toml` and `stats` over the model-time `window = (t0, t1)`
(or the last `last` seconds of the run; default: the last record's
interval, the output cadence before the end):
the mean profiles of the records in the window (`z`, `U`, `V`, `S`, `θ`,
the total stress `τ` and heat flux `wθ` at the cell centres, `zf = z`,
the tower thresholds `V_s`), the series means over it (`u_star`, `wθ0`,
`H0`, `h`, `L`), the jet of the mean wind-speed profile (`S_max`, `z_jet`),
`nsteps` and the run's `wall_time` (s).
"""
function sbl_run(folder::AbstractString; window = nothing, last = nothing)
    rt = TOML.parsefile(joinpath(folder, "run.toml"))
    cfg = rt["config"]
    if last !== nothing                              # the last `last` seconds of the run
        t_end = cfg["time"]["start"] + cfg["time"]["duration"]
        window = (t_end - last, t_end)
    end
    p = get(get(cfg, "output", Dict()), "path", "none")
    out = p isa AbstractString && p != "none" ? p : joinpath(folder, "output")
    nc = joinpath(out, get(get(cfg, "output", Dict()), "netcdf", "output.nc"))
    isfile(nc) || error("$folder: no $(basename(nc)) (has the run finished?)")
    series, P, w = NCDataset(nc) do ds
        tr = ds["time"].var[:]
        isempty(tr) && error("$folder: no record in $(basename(nc))")
        t1 = tr[end]
        t0 = window === nothing ? (length(tr) > 1 ? tr[end-1] : -Inf) : window[1]
        t1 = window === nothing ? t1 : window[2]
        series_rows(ds), mean_profiles(ds, t0, t1), (t0, t1)
    end
    in_w = [s for s in series if w[1] < s.t <= w[2] * (1 + 1e-12)]
    m(f) = isempty(in_w) ? NaN : sum(f, in_w) / length(in_w)
    k = argmax(P.S)
    dpath = joinpath(folder, "diagnostics.csv")
    nsteps = isfile(dpath) ? parse(Int, split(last(readlines(dpath)), ',')[1]) : 0
    wall = sum(s -> Float64(get(s, "wall_seconds", 0.0)), get(rt, "session", Dict[]); init = 0.0)
    stats = merge(P, (zf = P.z, u_star = m(s -> s.u_star), wθ0 = m(s -> s.wθ), H0 = m(s -> s.H), h = m(s -> s.h),
                      L = m(s -> s.L), S_max = P.S[k], z_jet = P.z[k], nsteps = nsteps, wall_time = wall))
    rp = joinpath(out, "results.toml")
    return (path = folder, name = get(get(cfg, "run", Dict()), "name", basename(folder)), config = cfg,
            stats = stats, series = series, results = isfile(rp) ? TOML.parsefile(rp) : Dict{String,Any}())
end

"""
    sbl_runs(args, usage; window = nothing, last = nothing) -> Vector

The run folders named on the command line (`args`), read with
[`sbl_run`](@ref); an error with the usage line when there are none.
"""
function sbl_runs(args, usage; kw...)
    isempty(args) && error("usage: $usage")
    return [sbl_run(a; kw...) for a in args]
end

"""
    cfgval(r, "section.key")

The config value `section.key` of a run: a run-level section (`time`,
`output`, …) or a per-domain one of its (first) domain (`grid`, `sgs`,
`surface_layer`, …; run.toml writes those under `[[config.domain]]`).
"""
function cfgval(r, key)
    s, k = split(key, '.')
    haskey(r.config, s) && return r.config[s][k]
    return only(r.config["domain"])[s][k]
end

"The model time of a run's end (s): `[time] start + duration`."
run_end(r) = cfgval(r, "time.start") + cfgval(r, "time.duration")
