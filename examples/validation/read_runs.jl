# Reading the output of the validation runs (EX-c, #440): the helpers the
# analysis scripts of examples/validation/ share. They read run folders
# only (run.toml and output/), and never build or run anything.
#
# A run folder is <root>/<date>_<run name>_<run id>/ (Aeolus.IO.start_run);
# a sweep member's run name is the config's name plus "_<label>". The
# scripts take the root of the runs as their one argument (default
# "runs", the root `aeolus run` writes to).

using Aeolus.IO: latest_run, read_run_toml
using Aeolus.Config: sweep_members
import TOML

"The root of the runs: the script's first argument, else `runs/`."
runs_root(args = ARGS) = isempty(args) ? "runs" : args[1]

"""
    run_folder(root, name) -> String

The newest run folder of the run `name` under `root`; an error that says
which config to run when there is none.
"""
function run_folder(root::AbstractString, name::AbstractString)
    f = latest_run(root, name)
    f === nothing && error("no run \"$name\" under $(abspath(root)): run its config first " *
                           "(julia --project -m Aeolus run examples/validation/configs/<case>.toml)")
    sessions = get(read_run_toml(f), "session", Any[])
    st = isempty(sessions) ? "" : get(sessions[end], "status", "")
    st == "completed" || @warn "the newest run \"$name\" is $(repr(st)), not completed" folder = f
    return f
end

"The resolved configuration a run recorded (`[config]` of its run.toml)."
run_config_table(folder::AbstractString) = read_run_toml(folder)["config"]

"The numbers a run wrote to `output/<name>` (default `results.toml`)."
run_results(folder::AbstractString, name::AbstractString = "results.toml") =
    TOML.parsefile(joinpath(folder, "output", name))

"`results.toml` and the config of the newest run `name` under `root`, as a NamedTuple `(res, cfg, folder)`."
function load_run(root::AbstractString, name::AbstractString; file::AbstractString = "results.toml")
    f = run_folder(root, name)
    return (res = run_results(f, file), cfg = run_config_table(f), folder = f)
end

"The shipped config `examples/validation/configs/<name>.toml`."
validation_config(name::AbstractString) = joinpath(@__DIR__, "configs", name * ".toml")

"""
    member_runs(root, config) -> Vector{Pair{String,NamedTuple}}

`label => load_run(root, "<run name>_<label>")` for every member of the
sweep of the config file `config`, in the sweep's order (one entry,
label `""`, for a config without a sweep). Members that have not been
run are left out with a warning.
"""
function member_runs(root::AbstractString, config::AbstractString; file::AbstractString = "results.toml")
    name = TOML.parsefile(config)["run"]["name"]
    ms = sweep_members(config)
    labels = ms === nothing ? [""] : [m.label for m in ms]
    out = Pair{String,Any}[]
    for l in labels
        run = isempty(l) ? name : name * "_" * l
        if latest_run(root, run) === nothing
            @warn "no run \"$run\" under $(abspath(root)) (run its member: aeolus run $(relpath(config)) --member $l)"
            continue
        end
        push!(out, l => load_run(root, run; file))
    end
    isempty(out) && error("no member of $(relpath(config)) has been run under $(abspath(root))")
    return out
end
