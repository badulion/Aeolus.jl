# The round-trip harness (CFG-b, #357), used by every config migration of
# #311: a run from a config, then a re-run from that run's own run.toml,
# must give the same run, within ROUNDOFF_RTOL (test/support/agreement.jl;
# no bit-identity since #495).
#
#   include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))
#   r = roundtrip_check("examples/taylor_green_vortex/taylor_green_vortex.toml"; nsteps = 20,
#                       overrides = ["grid.size=[16,16,16]"])
#   @test r.fields_agree && r.diagnostics_agree && r.config_equal
#
# Both runs use [fftw] planner = "estimate" (planned fast) in the same
# process. The configs are compared exactly: they are parsed and written,
# not computed.

using Aeolus
using Aeolus.Config
using Aeolus.IO: read_run_toml, diagnostics_path
using TOML
include(joinpath(@__DIR__, "support", "agreement.jl"))

if !isdefined(@__MODULE__, :roundtrip_check)

"""
    diagnostics_without_wall(path) -> Vector{String}

The lines of a `diagnostics.csv` with the columns that measure the
machine rather than the run removed: the wall clock (`wall …`, and the
step cores' wall time per stage `wall_*`, also prefixed, `parent.wall_step`)
and the process (`proc.*`: memory, GC time; #336). None for a run without
a run monitor (a harness driver of CFG-e writes no diagnostics.csv).
"""
_machine_column(h) = startswith(h, "wall") || occursin(".wall", h) || startswith(h, "proc.")

function diagnostics_without_wall(path::AbstractString)
    isfile(path) || return String[]
    lines = readlines(path)
    isempty(lines) && return lines
    keep = findall(h -> !_machine_column(h), split(lines[1], ','))
    return [join(split(l, ',')[keep], ',') for l in lines]
end

"A run's `output/results.toml` as a table without its wall-clock keys; an empty table if there is none."
function _results_table(rf)
    p = joinpath(rf.path, "output", "results.toml")
    isfile(p) || return Dict{String,Any}()
    strip_wall(d::AbstractDict) = Dict(k => strip_wall(v) for (k, v) in d if !occursin("wall", k))
    strip_wall(v) = v
    return strip_wall(TOML.parsefile(p))
end

"The `[time]` overrides of the model for a run of `n` steps."
simulation_nsteps(n) = ["time.duration=\"none\"", "time.nsteps=$n"]

"""
    roundtrip_check(config_path; nsteps = nothing, overrides = String[],
                    nsteps_overrides = simulation_nsteps, root = mktempdir())
        -> NamedTuple

Run `config_path` through [`run_config`](@ref Aeolus.Config.run_config)
into the run root `root`, then load that run's `run.toml` and run it
again (a new run, a new folder). `overrides` (plus
`nsteps_overrides(nsteps)` when `nsteps` is given, and `fftw.planner =
"estimate"`) apply to the first run only; `member` picks one member of a
sweep file (index or label): the second run takes
everything from the record. Returns

  * `fields_agree`: every array of `state_fields` agrees within
    ROUNDOFF_RTOL (`fields_differing` names the ones that do not);
  * `diagnostics_agree`: the two `diagnostics.csv` agree outside the
    machine columns (`csv_agree`);
  * `results_agree`: the two `output/results.toml` (the recorded checks)
    agree outside the wall-clock keys (`values_agree`);
  * `config_equal`: the two `[config]` tables of `run.toml` are identical
    (and so are the `config_hash`es);
  * `runs`: the two [`RunResult`](@ref Aeolus.Config.RunResult)s.
"""
function roundtrip_check(config_path::AbstractString; nsteps = nothing, overrides = String[],
                         nsteps_overrides = simulation_nsteps, root = mktempdir(), member = nothing)
    ovr = String[overrides...]
    nsteps === nothing || append!(ovr, nsteps_overrides(nsteps))
    push!(ovr, "fftw.planner=\"estimate\"")
    kw = (; root = root, console = false, repo = mktempdir())    # outside any git tree
    # a sweep file (CFG-e): one member, given by index or label
    cfg1 = member === nothing ? load_run_config(config_path; overrides = ovr) :
           only(load_sweep(config_path; overrides = ovr, member = member))[2]
    r1 = run_config(cfg1; kw...)
    cfg2 = load_run_config(r1.folder.path)
    r2 = run_config(cfg2; kw...)
    # invokelatest: an example driver may have been loaded during this call
    f1 = Base.invokelatest(state_fields, r1.obj); f2 = Base.invokelatest(state_fields, r2.obj)
    differing = [k for k in keys(f1) if !(haskey(f2, k) && agree(f1[k], f2[k]))]
    keys(f1) == keys(f2) || append!(differing, setdiff(keys(f2), keys(f1)))
    rec1 = read_run_toml(r1.folder.path); rec2 = read_run_toml(r2.folder.path)
    return (fields_agree = isempty(differing), fields_differing = differing,
            diagnostics_agree = csv_agree(diagnostics_without_wall(diagnostics_path(r1.folder)),
                                          diagnostics_without_wall(diagnostics_path(r2.folder))),
            results_agree = values_agree(_results_table(r1.folder), _results_table(r2.folder)),
            # isequal: a nan option (the Berlin canopy_top) equals itself
            config_equal = isequal(rec1["config"], rec2["config"]) && rec1["config_hash"] == rec2["config_hash"],
            runs = (r1, r2))
end

"""
    resume_check(config_path; nsteps = 20, stop = 10 + 5, every_steps = 10, overrides = String[],
                 nsteps_overrides = simulation_nsteps, outputs = String[]) -> NamedTuple

The `--resume` round trip (CFG-i, #364): run `config_path` for `nsteps`
steps with a checkpoint every `every_steps`, uninterrupted; run it again,
interrupted (an `InterruptException` from the step callback) after step
`stop`; then continue that run folder as `aeolus run <folder> --resume`
does (`run_config(load_run_config(folder); resume = folder)`). Returns

  * `fields_agree`: every array of `state_fields` of the resumed run
    agrees within ROUNDOFF_RTOL with the uninterrupted run's
    (`fields_differing` names the ones that do not);
  * `outputs_agree`: the files `outputs` (names in the run folders'
    `output/`) agree (`values_agree`), a `.toml` file outside its
    `wall_time` keys (the wall clock of a session, which no two runs share);
  * `statuses`: the session statuses of the resumed folder's `run.toml`;
  * `runs`: (uninterrupted, resumed) [`RunResult`](@ref Aeolus.Config.RunResult)s.
"""
function resume_check(config_path::AbstractString; nsteps::Integer = 20, stop::Integer = 15,
                      every_steps::Integer = 10, overrides = String[], nsteps_overrides = simulation_nsteps,
                      outputs = String[])
    ovr = vcat(String[overrides...], nsteps_overrides(nsteps), "checkpoint.every_steps=$every_steps",
               "fftw.planner=\"estimate\"")
    cfg = load_run_config(config_path; overrides = ovr)
    kw = (; console = false, repo = mktempdir())
    full = run_config(cfg; root = mktempdir(), kw...)
    root = mktempdir()
    halt = (_, n) -> n == stop && throw(InterruptException())
    interrupted = try
        run_config(cfg; root = root, callback = halt, kw...); false
    catch e
        e isa InterruptException || rethrow(); true
    end
    folder = only(readdir(root; join = true))
    res = run_config(load_run_config(folder); root = root, resume = folder, kw...)
    f1 = Base.invokelatest(state_fields, full.obj); f2 = Base.invokelatest(state_fields, res.obj)
    differing = [k for k in keys(f1) if !(haskey(f2, k) && agree(f1[k], f2[k]))]
    out(r, n) = _output_content(joinpath(r.folder.path, "output", n))
    return (fields_agree = interrupted && isempty(differing), fields_differing = differing,
            outputs_agree = all(n -> out(full, n) !== nothing && values_agree(out(full, n), out(res, n)), outputs),
            statuses = [s["status"] for s in read_run_toml(folder)["session"]],
            runs = (full, res))
end

"""
A run output for comparison: the bytes; for a `.toml` the parsed table
without its `wall_time` keys; for a NetCDF file (`.nc`, the one writer's
`output.nc`, ONE-c) every variable's values and attributes (an HDF5 file's
bytes carry its own write history). Bytes of any other file are compared
exactly (`values_agree` of two byte vectors is `==`).
"""
function _output_content(p)
    isfile(p) || return nothing
    if endswith(p, ".nc")
        return Aeolus.NCIO.NCDataset(p) do ds
            Dict(k => (Array(ds[k].var), Dict(ds[k].attrib)) for k in keys(ds))
        end
    end
    endswith(p, ".toml") || return read(p)
    strip_wall(d::AbstractDict) = Dict(k => strip_wall(v) for (k, v) in d if k != "wall_time")
    strip_wall(v) = v
    return strip_wall(TOML.parsefile(p))
end

end # if !isdefined
