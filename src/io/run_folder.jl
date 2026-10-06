# Run folder and provenance (LOG-b, #334; design in epic #293).
#
# One folder per run, `runs/<date>_<case>_<runID>/`:
#
#   run.log            the run log (debug and up; RunLog.open_run_log)
#   run.toml           provenance: run ID, config, seeds, one [[session]]
#                      per start / restart (git, versions, host, threads,
#                      command line, FFTW policy, start and end times)
#   diagnostics.csv    the diagnostics time series (#335; .nc after #310)
#   output/            output files of the run
#   checkpoints/       the run's CheckpointSchedule (#245) and FFTW wisdom
#   uncommitted-<n>.diff, untracked-<n>.txt
#                      the code that was not committed when session n
#                      started with allow_uncommitted = true
#
# A restart reuses the folder, keeps the run ID and appends: to run.log,
# and a new [[session]] to run.toml. A run that writes a run folder refuses
# to start on a dirty tree unless allow_uncommitted = true. Nothing else in
# the library touches git: tests and in-memory runs are unaffected.

using ..RunLog: RunLog, open_run_log, current_domain
using Random: RandomDevice
using Logging: Logging, AbstractLogger, @logmsg, LogLevel, Warn, Error
import LinearAlgebra

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

"""
    fnv1a64(data) -> UInt64

The 64-bit FNV-1a hash of `data` (bytes or a string). Used for the
Manifest hash in the provenance: a stable, dependency-free fingerprint
(not a cryptographic hash). `fnv1a64("") == 0xcbf29ce484222325`.
"""
function fnv1a64(data::AbstractVector{UInt8})
    h = 0xcbf29ce484222325
    @inbounds for b in data
        h = (h ⊻ b) * 0x00000100000001b3
    end
    return h
end
fnv1a64(s::AbstractString) = fnv1a64(codeunits(String(s)))

"Local wall-clock time `t` as `yyyy-mm-ddTHH:MM:SS±hhmm` (ISO 8601)."
_iso_time(t::Real = time()) = Libc.strftime("%Y-%m-%dT%H:%M:%S%z", floor(t))

"`case` made safe for a folder name: anything but letters, digits, `-`, `_` becomes `-`."
_safe_case(case) = replace(string(case), r"[^A-Za-z0-9_\-]" => "-")

"A fresh run ID: 8 hex digits from the OS entropy source (the global RNG is not touched)."
new_run_id() = string(rand(RandomDevice(), UInt32); base = 16, pad = 8)

"Run `cmd`; return `(ok, stdout)` without throwing, stderr discarded."
function _run_capture(cmd::Cmd)
    try
        out = IOBuffer()
        p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = devnull))
        return (success(p), String(take!(out)))
    catch
        return (false, "")
    end
end

"""
    toml_value(x; depth = 0)

`x` as a value `TOML.print` can write: numbers, strings and booleans as
they are; symbols as strings; dicts, named tuples and structs as tables
(a struct gets a `"_type"` key); tuples and short vectors as arrays;
`nothing` as `"nothing"`; long arrays and anything else as a short
summary string. Used for the `[config]` and `[seeds]` tables of
`run.toml`, so any configuration object (a `Dict`, a `NamedTuple`, a
`CaseConfig`) is recorded in full.
"""
function toml_value(x; depth::Int = 0)
    depth > 8 && return _summary(x)
    d = depth + 1
    x isa Bool && return x
    x isa Unsigned && return x <= typemax(Int64) ? Int64(x) : string(x)
    x isa Integer && return typemin(Int64) <= x <= typemax(Int64) ? Int64(x) : string(x)
    x isa AbstractFloat && return Float64(x)
    x isa Rational && return Float64(x)
    x isa AbstractString && return String(x)
    x isa Symbol && return String(x)
    x isa AbstractChar && return string(x)
    (x === nothing || x === missing) && return string(x)
    x isa AbstractDict &&
        return Dict{String,Any}(string(k) => toml_value(v; depth = d) for (k, v) in x)
    x isa NamedTuple &&
        return Dict{String,Any}(string(k) => toml_value(v; depth = d) for (k, v) in pairs(x))
    x isa Pair && return Dict{String,Any}(string(x.first) => toml_value(x.second; depth = d))
    if x isa Tuple || (x isa AbstractVector && length(x) <= 64)
        return Any[toml_value(v; depth = d) for v in x]
    end
    (x isa AbstractArray || x isa Function || x isa Module || x isa Type) && return _summary(x)
    T = typeof(x)
    if isstructtype(T) && fieldcount(T) > 0
        t = Dict{String,Any}("_type" => string(nameof(T)))
        for f in fieldnames(T)
            isdefined(x, f) && (t[string(f)] = toml_value(getfield(x, f); depth = d))
        end
        return t
    end
    return _summary(x)
end

function _summary(x)
    x isa AbstractArray && return string(summary(x))
    s = repr(x; context = (:compact => true, :limit => true))
    return length(s) > 200 ? first(s, 197) * "..." : s
end

# ---------------------------------------------------------------------------
# configuration hooks (extended by Aeolus.Config for a RunConfig, CFG-a #356)
# ---------------------------------------------------------------------------

"""
    config_record(config) -> Dict{String,Any}

The top-level entries `run.toml` records for `config`: for any object,
`config` as [`toml_value`](@ref) gives it. A `RunConfig`
(`Aeolus.Config`) adds its canonical table, schema version, hash,
sources, input-file hashes and replayability.
"""
config_record(config) = Dict{String,Any}("config" => toml_value(config))

"""
    run_options(config) -> NamedTuple

The [`start_run`](@ref) settings a configuration carries (`root`,
`allow_uncommitted`, `console_level`, `file_level`, `flush_interval`):
none for a plain object; a `RunConfig` takes them from `[run]` and
`[logging]`. Explicit keywords of `start_run` win.
"""
run_options(config) = NamedTuple()

"""
    check_restart_config(config, record; allow_changed_inputs = false)

Called by [`start_run`](@ref) before a restart writes anything: nothing
for a plain object; for a `RunConfig`, a `ConfigError` names every input
file whose content changed since the run started.
"""
check_restart_config(config, record; allow_changed_inputs::Bool = false) = nothing

# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

"The root of the Aeolus source tree (the default repository a run's code is checked in)."
aeolus_source_dir() = dirname(dirname(@__DIR__))

"""
    git_state(repo = aeolus_source_dir(); exclude = String[]) -> NamedTuple

The state of the git checkout containing `repo`: `available` (git is
installed and `repo` is in a work tree), `toplevel`, `commit` (full hash,
`""` if unavailable), `branch`, `dirty` and `changes`, the lines of
`git status --porcelain` (staged, unstaged and untracked files; ignored
files do not count). Paths under any of `exclude` (absolute) are left out,
so a run folder inside the work tree does not make it dirty.
"""
function git_state(repo::AbstractString = aeolus_source_dir(); exclude = String[])
    none = (available = false, toplevel = "", commit = "", branch = "", dirty = false,
            changes = String[])
    isdir(repo) || return none
    ok, top = _run_capture(`git -C $repo rev-parse --show-toplevel`)
    ok || return none
    top = strip(top)
    _, commit = _run_capture(`git -C $repo rev-parse HEAD`)
    _, branch = _run_capture(`git -C $repo rev-parse --abbrev-ref HEAD`)
    ok, status = _run_capture(`git -C $repo status --porcelain --untracked-files=all`)
    changes = String[l for l in split(status, '\n') if !isempty(strip(l))]
    excl = [_relpath_in(e, top) for e in exclude]
    filter!(l -> !any(r -> r !== nothing && _porcelain_under(l, r), excl), changes)
    return (available = true, toplevel = String(top), commit = String(strip(commit)),
            branch = String(strip(branch)), dirty = !isempty(changes), changes = changes)
end

"`path` relative to `top` (with `/` separators), or `nothing` if outside it."
function _relpath_in(path, top)
    p = realpath_or(abspath(path)); t = realpath_or(abspath(top))
    (p == t || startswith(p, t * "/")) || return nothing
    return p == t ? "" : p[length(t)+2:end]
end
realpath_or(p) = ispath(p) ? realpath(p) : p

"Whether the porcelain line `l` names a path under the relative directory `r`."
function _porcelain_under(l::AbstractString, r::AbstractString)
    isempty(r) && return true
    path = strip(l[4:end], '"')
    occursin(" -> ", path) && (path = split(path, " -> ")[end])
    return path == r || startswith(path, r * "/")
end

"The versions of every package in the active project's Manifest (direct and indirect)."
function _package_versions(manifest::AbstractString)
    v = Dict{String,Any}()
    isfile(manifest) || return v
    m = try TOML.parsefile(manifest) catch; return v end
    deps = get(m, "deps", m)                       # format 2.0 nests under "deps"
    deps isa AbstractDict || return v
    for (name, entries) in deps
        entries isa AbstractVector || continue
        for e in entries
            e isa AbstractDict && haskey(e, "version") && (v[name] = e["version"])
        end
    end
    return v
end

"""
    provenance(; repo = aeolus_source_dir(), exclude = String[]) -> Dict{String,Any}

What produced a run, as one TOML-ready table (the `[[session]]` entry of
`run.toml` without its times): `git` (commit, branch, dirty, the changed
files), `julia` (version and commit), `aeolus` (version and source
directory), `manifest` (path and FNV-1a hash, [`fnv1a64`](@ref)),
`packages` (every package version in the Manifest), `host` (name, OS,
CPU, Julia, BLAS and CPU threads), `command` (Julia command, program,
arguments, working directory), `fftw` (the FFTW policy: planner, threads,
wisdom loaded and file, bitwise reproducible; decided on #245) and `env`
(every `AEOLUS_*` environment variable).
"""
function provenance(; repo::AbstractString = aeolus_source_dir(), exclude = String[])
    g = git_state(repo; exclude = exclude)
    proj = Base.active_project()
    manifest = proj === nothing ? "" : something(Base.project_file_manifest_path(proj), "")
    mhash = isfile(manifest) ? string(fnv1a64(read(manifest)); base = 16, pad = 16) : ""
    project = TOML.parsefile(joinpath(aeolus_source_dir(), "Project.toml"))
    p = fftw_policy()
    return Dict{String,Any}(
        "git" => Dict{String,Any}("available" => g.available, "repo" => g.toplevel,
                                  "commit" => g.commit, "branch" => g.branch,
                                  "dirty" => g.dirty, "changes" => g.changes),
        "julia" => Dict{String,Any}("version" => string(VERSION),
                                    "commit" => Base.GIT_VERSION_INFO.commit_short),
        "aeolus" => Dict{String,Any}("version" => get(project, "version", ""),
                                     "source" => aeolus_source_dir()),
        "manifest" => Dict{String,Any}("path" => manifest, "fnv1a64" => mhash),
        "packages" => _package_versions(manifest),
        "host" => Dict{String,Any}("hostname" => gethostname(), "os" => string(Sys.KERNEL),
                                   "arch" => string(Sys.ARCH),
                                   "cpu" => isempty(Sys.cpu_info()) ? "" : strip(Sys.cpu_info()[1].model),
                                   "cpu_threads" => Sys.CPU_THREADS,
                                   "julia_threads" => Threads.nthreads(),
                                   "blas_threads" => LinearAlgebra.BLAS.get_num_threads()),
        "command" => Dict{String,Any}("julia" => join(Base.julia_cmd().exec, ' '),
                                      "program" => PROGRAM_FILE, "args" => copy(ARGS),
                                      "cwd" => pwd()),
        "fftw" => Dict{String,Any}("planner" => string(p.planner), "threads" => p.threads,
                                   "wisdom_loaded" => p.wisdom_loaded,
                                   "wisdom_file" => p.wisdom_file,
                                   "bitwise_reproducible" => p.bitwise_reproducible),
        "env" => Dict{String,Any}(k => v for (k, v) in ENV if startswith(k, "AEOLUS_")),
    )
end

# ---------------------------------------------------------------------------
# the run folder
# ---------------------------------------------------------------------------

"""
    UncommittedCodeError

Thrown by [`start_run`](@ref) when the code has uncommitted changes and
`allow_uncommitted = false`. Its message lists the changed files.
"""
struct UncommittedCodeError <: Exception
    repo::String
    changes::Vector{String}
end
function Base.showerror(io::Base.IO, e::UncommittedCodeError)
    n = length(e.changes)
    print(io, "UncommittedCodeError: the code in $(e.repo) has $n uncommitted change",
          n == 1 ? "" : "s", ", so this run could not be reproduced from its commit:")
    for l in first(e.changes, 20)
        print(io, "\n    ", l)
    end
    n > 20 && print(io, "\n    … and $(n - 20) more")
    print(io, "\nCommit the changes, or pass `allow_uncommitted = true` (or set " *
              "AEOLUS_ALLOW_UNCOMMITTED=1) to run anyway; the diff and the untracked " *
              "files are then saved in the run folder.")
end

"""
    RunFolder

A run's folder and provenance record, made by [`start_run`](@ref):
`path`, `run_id`, `case`, `session` (1 for the first start, n for the
(n-1)-th restart), `logger` (the run log, to install with
`Logging.with_logger`; see [`with_run`](@ref)), `record` (the contents of
`run.toml`) and `finished`. Paths inside it: [`run_log_path`](@ref),
[`run_toml_path`](@ref), [`diagnostics_path`](@ref), [`output_dir`](@ref),
[`checkpoint_dir`](@ref).
"""
mutable struct RunFolder
    path::String
    run_id::String
    case::String
    session::Int
    logger::AbstractLogger
    record::Dict{String,Any}
    started::Float64
    finished::Bool
end

Base.show(io::Base.IO, rf::RunFolder) =
    print(io, "RunFolder(\"", rf.path, "\", run ", rf.run_id, ", session ", rf.session,
          rf.finished ? ", finished)" : ")")

"`path/run.log`, the run log (appended to by every session)."
run_log_path(rf::RunFolder) = joinpath(rf.path, "run.log")
"`path/run.toml`, the provenance record."
run_toml_path(rf::RunFolder) = joinpath(rf.path, "run.toml")
"`path/diagnostics.csv`, the diagnostics time series (written by #335; `diagnostics.nc` after #310)."
diagnostics_path(rf::RunFolder) = joinpath(rf.path, "diagnostics.csv")
"`path/output`, the folder for the run's output files (created)."
output_dir(rf::RunFolder) = mkpath(joinpath(rf.path, "output"))
"`path/checkpoints`, the folder of the run's checkpoints and FFTW wisdom (created)."
checkpoint_dir(rf::RunFolder) = mkpath(joinpath(rf.path, "checkpoints"))

"""
    read_run_toml(folder) -> Dict{String,Any}

The provenance record `folder/run.toml` of a run folder (or the path of
the `run.toml` itself).
"""
read_run_toml(folder::AbstractString) =
    TOML.parsefile(isdir(folder) ? joinpath(folder, "run.toml") : folder)

"""
    latest_run(root, case) -> Union{String, Nothing}

The most recently created run folder of `case` under `root` (by the
`created_unix` time of its `run.toml`, compared as a number; ties are
broken by the `run.toml` modification time, then the folder name), or
`nothing`.
"""
function latest_run(root::AbstractString, case)
    isdir(root) || return nothing
    prefix_ok(n) = occursin(Regex("^\\d{4}-\\d{2}-\\d{2}_" * _safe_case(case) * "_[0-9a-f]+\$"), n)
    best = nothing; best_key = (-Inf, -Inf, "")
    for n in readdir(root)
        p = joinpath(root, n)
        (isdir(p) && prefix_ok(n) && isfile(joinpath(p, "run.toml"))) || continue
        c = get(read_run_toml(p), "created_unix", 0.0)
        t = c isa Real ? Float64(c) : something(tryparse(Float64, string(c)), 0.0)
        # Numeric comparison (the decimal forms of Float64s differ in length,
        # so a string comparison orders them by length); ties by mtime, name.
        key = (t, mtime(joinpath(p, "run.toml")), n)
        if best === nothing || key > best_key
            best, best_key = p, key
        end
    end
    return best
end

function _write_record(rf::RunFolder)
    tmp = run_toml_path(rf) * ".tmp"
    open(tmp, "w") do io
        println(io, "# Aeolus run record (provenance; #334). One [[session]] per start or restart.")
        TOML.print(io, rf.record; sorted = true)
    end
    mv(tmp, run_toml_path(rf); force = true)        # atomic: a crash never leaves half a file
    return nothing
end

_current_session(rf::RunFolder) = rf.record["session"][end]

"""
    start_run(case; root = "runs", config = Dict(), seeds = Dict(), resume = nothing,
              allow_uncommitted = ENV["AEOLUS_ALLOW_UNCOMMITTED"] == "1",
              repo = aeolus_source_dir(), console = stderr,
              console_level = AEOLUS_LOG or Info, file_level = Debug,
              flush_interval = 5.0, allow_changed_inputs = false,
              run_id = nothing) -> RunFolder

With `config` a `RunConfig` (`Aeolus.Config`, CFG-a #356), `root`,
`allow_uncommitted`, the log levels and `flush_interval` default to its
`[run]` and `[logging]` sections (keywords given explicitly win), and a
restart refuses to start when an input file recorded under `[inputs]`
changed, unless `allow_changed_inputs = true`.

Start a run that writes a run folder (epic #293, #334):

1. **Code check.** If the git work tree containing `repo` (the Aeolus
   source tree by default) has uncommitted changes, staged, unstaged or
   untracked, throw [`UncommittedCodeError`](@ref), unless
   `allow_uncommitted = true`: then `git diff HEAD` is saved as
   `uncommitted-<session>.diff` and the untracked files are listed in
   `untracked-<session>.txt` in the folder, and a warning is logged.
   Files under `root` are ignored, so the run folders do not count. If
   `repo` is not in a git work tree (an installed release), the run starts
   and the provenance records `git.available = false`.
2. **Folder.** A new run creates `root/<yyyy-mm-dd>_<case>_<runID>/`
   with a fresh 8-hex-digit run ID (from the OS entropy source; the global
   RNG is untouched). `resume = folder` (or `resume = :latest`, the newest
   folder of `case` under `root`, see [`latest_run`](@ref)) reuses an
   existing folder: same run ID, the log is appended and `run.toml` gets
   a new session.
3. **Provenance.** `run.toml` holds `run_id`, `case`, `created`, the full
   `config` and `seeds` (any object; see [`toml_value`](@ref); a
   `RunConfig` is stored in its canonical resolved form with
   `config_schema_version`, `config_hash`, `config_sources`, `[inputs]`
   and `replayable`, see [`config_record`](@ref)) and one
   `[[session]]` table per start or restart: `kind` (`"start"` or
   `"restart"`), `started`, `ended`, `status`, `wall_seconds` and
   everything [`provenance`](@ref) returns (git commit, Julia and package
   versions with the Manifest hash, host, threads, command line, FFTW
   planner, threads and wisdom, `AEOLUS_*` variables). A restart whose
   config, seeds or commit differ from the first session logs a warning
   and records the new values in its session.
4. **Run log.** `logger` is [`open_run_log`](@ref)`(run.log; console,
   console_level, file_level)`: info to the console, debug to `run.log`.
   The header is logged through it: one `Info` record (`subsystem =
   :run`, `event = :start` or `:restart`) with the run ID, folder, commit,
   versions, host, threads and FFTW policy, and one `Debug` record (only
   `run.log` keeps it) with the whole session record and the config.

Call [`finish_run!`](@ref) at the end, or use [`with_run`](@ref), which
does both and installs the logger.
"""
function start_run(case; root::Union{Nothing,AbstractString} = nothing,
                   config = Dict{String,Any}(), seeds = Dict{String,Any}(), resume = nothing,
                   allow_uncommitted::Union{Nothing,Bool} = nothing,
                   repo::AbstractString = aeolus_source_dir(),
                   console::Union{Bool,Nothing,Base.IO} = stderr,
                   console_level = nothing, file_level = nothing,
                   flush_interval::Union{Nothing,Real} = nothing,
                   allow_changed_inputs::Bool = false,
                   run_id::Union{Nothing,AbstractString} = nothing)
    t0 = time()
    opts = run_options(config)
    root = abspath(something(root, get(opts, :root, "runs")))
    allow_uncommitted = something(allow_uncommitted, get(opts, :allow_uncommitted,
                                  get(ENV, "AEOLUS_ALLOW_UNCOMMITTED", "0") == "1"))
    console_level = something(console_level, get(opts, :console_level, Some(nothing)))
    file_level = something(file_level, get(opts, :file_level, Logging.Debug))
    flush_interval = something(flush_interval, get(opts, :flush_interval, 5.0))
    crec = config_record(config)
    # 1. code check, before anything is written
    prov = provenance(; repo = repo, exclude = [root])
    g = prov["git"]
    if g["dirty"] && !allow_uncommitted
        throw(UncommittedCodeError(g["repo"], g["changes"]))
    end
    # 2. folder
    folder = resume === :latest ? latest_run(root, case) :
             resume === nothing ? nothing : abspath(String(resume))
    resume === :latest && folder === nothing &&
        throw(ArgumentError("start_run: resume = :latest but $root has no run of case \"$case\""))
    restart = folder !== nothing
    if restart
        isfile(joinpath(folder, "run.toml")) ||
            throw(ArgumentError("start_run: $folder is not a run folder (no run.toml)"))
        record = read_run_toml(folder)
        rid = record["run_id"]
        run_id === nothing || run_id == rid ||
            throw(ArgumentError("start_run: run_id = $run_id, but $folder is run $rid"))
        check_restart_config(config, record; allow_changed_inputs = allow_changed_inputs)
    else
        rid = run_id === nothing ? new_run_id() : String(run_id)
        folder = joinpath(root, string(Libc.strftime("%Y-%m-%d", t0), '_', _safe_case(case),
                                       '_', rid))
        isdir(folder) && throw(ArgumentError("start_run: $folder exists; pass resume = it to restart"))
        record = Dict{String,Any}("run_id" => rid, "case" => string(case),
                                  "folder" => folder, "created" => _iso_time(t0),
                                  "created_unix" => t0, "seeds" => toml_value(seeds),
                                  "session" => Any[])
        merge!(record, crec)
    end
    mkpath(folder)
    sessions = get!(record, "session", Any[])
    n = length(sessions) + 1
    # a previous session that never finished (crash, kill) is marked as such
    aborted = Int[]
    for s in sessions
        if get(s, "status", "") == "running"
            s["status"] = "aborted"; push!(aborted, s["number"])
        end
    end
    session = merge(prov, Dict{String,Any}("number" => n, "kind" => restart ? "restart" : "start",
                                           "started" => _iso_time(t0), "status" => "running",
                                           "allow_uncommitted" => allow_uncommitted))
    changed = String[]
    if restart
        sd = toml_value(seeds)
        same = haskey(crec, "config_hash") ? get(record, "config_hash", nothing) == crec["config_hash"] :
               isequal(crec["config"], get(record, "config", nothing))
        if !same
            session["config"] = crec["config"]
            haskey(crec, "config_hash") && (session["config_hash"] = crec["config_hash"])
            push!(changed, "config")
        end
        sd == record["seeds"] || (session["seeds"] = sd; push!(changed, "seeds"))
        first_commit = sessions[1]["git"]["commit"]
        first_commit == g["commit"] || push!(changed, "git commit")
    end
    if g["dirty"]
        _, diff = _run_capture(`git -C $(g["repo"]) diff HEAD`)
        untracked = [String(strip(l[4:end], '"')) for l in g["changes"] if startswith(l, "??")]
        write(joinpath(folder, "uncommitted-$n.diff"), diff)
        write(joinpath(folder, "untracked-$n.txt"), join(untracked, '\n'), isempty(untracked) ? "" : "\n")
        session["git"]["diff_file"] = "uncommitted-$n.diff"
        session["git"]["untracked_file"] = "untracked-$n.txt"
    end
    push!(sessions, session)
    # 3. log
    lvl = console_level === nothing ? RunLog._env_level(Logging.Info) : console_level
    logger = open_run_log(joinpath(folder, "run.log"); console = console,
                          console_level = lvl, file_level = file_level, append = true,
                          flush_interval = flush_interval)
    rf = RunFolder(folder, rid, string(record["case"]), n, logger, record, t0, false)
    _write_record(rf)
    Logging.with_logger(logger) do
        _log_header(rf, session, aborted, changed)
    end
    return rf
end

function _log_header(rf::RunFolder, s, aborted, changed)
    dom = current_domain()
    g = s["git"]; f = s["fftw"]; h = s["host"]
    commit = g["available"] ? first(g["commit"], 12) * (g["dirty"] ? "-dirty" : "") : "unavailable"
    @info(s["kind"] == "start" ? "run started" : "run restarted (session $(s["number"]))",
          subsystem = :run, domain = dom, event = Symbol(s["kind"]), run_id = rf.run_id,
          case = rf.case, folder = rf.path, session = s["number"], commit, branch = g["branch"],
          julia = s["julia"]["version"], aeolus = s["aeolus"]["version"],
          manifest = s["manifest"]["fnv1a64"], host = h["hostname"],
          threads = h["julia_threads"], fftw_planner = f["planner"],
          fftw_threads = f["threads"], wisdom_loaded = f["wisdom_loaded"],
          bitwise_reproducible = f["bitwise_reproducible"])
    if g["dirty"]
        @warn("running with uncommitted code (allow_uncommitted = true); the diff is saved",
              subsystem = :run, domain = dom, event = :uncommitted, changes = length(g["changes"]),
              diff = get(g, "diff_file", ""), untracked = get(g, "untracked_file", ""))
    end
    g["available"] ||
        @warn("the code is not in a git work tree: the provenance records versions only",
              subsystem = :run, domain = dom, event = :no_git, repo = s["aeolus"]["source"])
    isempty(aborted) ||
        @warn("previous session(s) of this run did not finish (no end time recorded)",
              subsystem = :run, domain = dom, event = :aborted, sessions = Tuple(aborted))
    isempty(changed) ||
        @warn("this restart differs from the first session in: $(join(changed, ", "))",
              subsystem = :run, domain = dom, event = :restart_changed, changed = Tuple(changed))
    buf = IOBuffer()
    TOML.print(buf, Dict{String,Any}("config" => rf.record["config"], "seeds" => rf.record["seeds"],
                                     "session" => s); sorted = true)
    @debug("provenance", subsystem = :run, domain = dom, event = :provenance,
           record = String(take!(buf)))
    return nothing
end

"""
    refresh_fftw_provenance!(rf::RunFolder) -> RunFolder

Record the current FFTW policy in the session of `rf` and rewrite
`run.toml`. [`run_checkpoints`](@ref) and [`finish_run!`](@ref) call it,
because the wisdom of a restart is loaded after [`start_run`](@ref).
"""
function refresh_fftw_provenance!(rf::RunFolder)
    p = fftw_policy()
    _current_session(rf)["fftw"] = Dict{String,Any}(
        "planner" => string(p.planner), "threads" => p.threads,
        "wisdom_loaded" => p.wisdom_loaded, "wisdom_file" => p.wisdom_file,
        "bitwise_reproducible" => p.bitwise_reproducible)
    _write_record(rf)
    return rf
end

"""
    run_checkpoints(rf::RunFolder; every_steps = 0, every_seconds = 1800, keep = 2,
                    prefix = "checkpoint", kwargs...) -> CheckpointSchedule

The [`CheckpointSchedule`](@ref) of a run, in its folder's
`checkpoints/` (imports the folder's FFTW wisdom, so a restart plans
the same transforms), with the FFTW provenance refreshed. After a
restart, `resume!(ck, sim)` continues from the folder's newest checkpoint.
"""
function run_checkpoints(rf::RunFolder; every_steps::Integer = 0, every_seconds::Real = 1800,
                         keep::Integer = 2, prefix::AbstractString = "checkpoint", kwargs...)
    ck = CheckpointSchedule(checkpoint_dir(rf); every_steps = every_steps,
                            every_seconds = every_seconds, keep = keep, prefix = prefix,
                            kwargs...)
    refresh_fftw_provenance!(rf)
    return ck
end

"""
    finish_run!(rf::RunFolder; status = :completed, error = nothing) -> RunFolder

End the session of `rf`: record `ended`, `status` (`:completed`,
`:failed`, `:interrupted`, …), `wall_seconds` and, if given, the error
message in `run.toml`; log the end (`subsystem = :run`, `event =
:finish`; at `Info` when completed, else `Error`); close the run log. A
second call does nothing.
"""
function finish_run!(rf::RunFolder; status::Symbol = :completed, error = nothing)
    rf.finished && return rf
    t = time()
    s = _current_session(rf)
    p = fftw_policy()
    s["fftw"] = Dict{String,Any}("planner" => string(p.planner), "threads" => p.threads,
                                 "wisdom_loaded" => p.wisdom_loaded, "wisdom_file" => p.wisdom_file,
                                 "bitwise_reproducible" => p.bitwise_reproducible)
    s["ended"] = _iso_time(t)
    s["status"] = string(status)
    s["wall_seconds"] = round(t - rf.started; digits = 3)
    error === nothing || (s["error"] = first(sprint(showerror, error), 2000))
    _write_record(rf)
    Logging.with_logger(rf.logger) do
        @logmsg((status === :completed ? Logging.Info : Logging.Error), "run finished: $status",
                subsystem = :run, domain = current_domain(), event = :finish,
                run_id = rf.run_id, session = rf.session, status,
                wall_seconds = s["wall_seconds"])
    end
    close(rf.logger)
    rf.finished = true
    return rf
end

"""
    with_run(f, case; kwargs...) -> f(rf)

[`start_run`](@ref)`(case; kwargs...)`, then call `f(rf)` with the run
log installed as the current logger, then [`finish_run!`](@ref). If `f`
throws, the error and its backtrace are logged, the session is recorded
as `failed` (or `interrupted`) and the error is rethrown.

```julia
with_run("gabls1"; config = cfg, seeds = (; noise = 42)) do rf
    ck = run_checkpoints(rf; every_steps = 500)
    resume!(ck, sim)
    for n in ck.last_step+1:nsteps
        step_adaptive!(sim, cfl)
        maybe_checkpoint!(ck, sim; step = n)
    end
end
```
"""
function with_run(f, case; kwargs...)
    rf = start_run(case; kwargs...)
    local result
    try
        result = Logging.with_logger(rf.logger) do
            f(rf)
        end
    catch err
        interrupted = err isa InterruptException
        bt = catch_backtrace()
        Logging.with_logger(rf.logger) do
            interrupted ||
                @error("run stopped by an error", subsystem = :run, domain = current_domain(),
                       exception = (err, bt))
        end
        finish_run!(rf; status = interrupted ? :interrupted : :failed, error = err)
        rethrow()
    end
    finish_run!(rf)
    return result
end
