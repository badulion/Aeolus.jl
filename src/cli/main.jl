# The aeolus command line (CFG-b, #357): one entry point for every run.
#
#   julia --project -m Aeolus run config.toml        (Julia ≥ 1.12)
#   bin/aeolus run config.toml                       (any Julia)
#
# Argument parsing is in-package (no ArgParse/Comonicon dependency).
# Exit codes: 0 success, 1 the run failed, 2 a usage or config error.

const _USAGE = """
usage: aeolus <command> [arguments]

  run <config.toml | run.toml | run-folder> [--set key=value]... [--resume]
      [--root DIR] [--allow-uncommitted] [--member k]
        Start a new run from a config file or a run's run.toml; with --resume and
        a run folder, continue that run (same run ID) from its newest checkpoint.
        A file with a [sweep] runs every member (one run folder each), or only
        member k (an index or a label).
  check <config> [--set key=value]...
        Validate the config (every member of a sweep) and print every problem.
  resolve <config> [--set key=value]... [--member k]
        Print the canonical resolved TOML (of every member of a sweep).
  options [driver] [section]
        Print the option reference of a driver (all sections, or one).
  preprocess <kind> [arguments]
        Run a registered preprocessor (no kind: list them).
  help
        Print this message.

Overrides are "dotted.key=<TOML value>": --set stepper.nsteps=20,
--set 'run.name="tgv"'. Exit codes: 0 success, 1 the run failed,
2 a usage or configuration error (nothing is written).
"""

"A command-line usage error (exit code 2)."
struct UsageError <: Exception
    msg::String
end
Base.showerror(io::Base.IO, e::UsageError) = print(io, "usage error: ", e.msg)

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

"""
Split `args` into positionals and options. `valued` are the options that
take a value (`--root DIR` or `--root=DIR`; `--set` may repeat), `flags`
the ones that do not.
"""
function _parse_args(args, valued::Tuple, flags::Tuple)
    pos = String[]; sets = String[]; opts = Dict{String,String}(); fl = Set{String}()
    i = 1
    while i <= length(args)
        a = String(args[i])
        if startswith(a, "--")
            k, v = occursin('=', a) ? split(a[3:end], '='; limit = 2) : (a[3:end], nothing)
            k = String(k)
            if k in flags
                v === nothing || throw(UsageError("--$k takes no value"))
                push!(fl, k)
            elseif k in valued
                if v === nothing
                    i += 1
                    i <= length(args) || throw(UsageError("--$k needs a value"))
                    v = args[i]
                end
                k == "set" ? push!(sets, String(v)) :
                    (haskey(opts, k) ? throw(UsageError("--$k given twice")) : (opts[k] = String(v)))
            else
                known = join(("--" * x for x in (valued..., flags...)), ", ")
                throw(UsageError("unknown option --$k" * (isempty(known) ? "" : " (known: $known)")))
            end
        else
            push!(pos, a)
        end
        i += 1
    end
    return pos, sets, opts, fl
end

_one_positional(pos, what) = length(pos) == 1 ? pos[1] :
    throw(UsageError(isempty(pos) ? "missing $what" : "expected one $what, got $(length(pos)): $(join(pos, " "))"))

# ---------------------------------------------------------------------------
# preprocessors
# ---------------------------------------------------------------------------

const _PREPROCESSORS = Dict{String,Tuple{Any,String}}()

"""
    register_preprocessor!(kind, f, usage) -> kind

Register a preprocessor for `aeolus preprocess <kind> args…`:
`f(args::Vector{String})` does the work and returns an exit code (an
`Integer`; `nothing` counts as 0); `usage` is the one-line synopsis that
`aeolus preprocess` lists. A [`ConfigError`](@ref) or an `ArgumentError`
it throws exits with code 2, any other error with 1.
"""
function register_preprocessor!(kind::AbstractString, f, usage::AbstractString)
    _PREPROCESSORS[String(kind)] = (f, String(usage))
    return String(kind)
end

"""
`aeolus preprocess static <config.toml> [--out file.nc] [--parent-out file.nc] [--set key=value]...`:
load the run config and write its static driver(s) with the driver's
[`preprocess_static`](@ref) (CFG-f, #361); prints the files written.
"""
function _preprocess_static(args)
    pos, sets, opts, _ = _parse_args(args, ("set", "out", "parent-out"), ())
    path = _one_positional(pos, "config file")
    cfg = load_run_config(path; overrides = sets)
    d = run_driver(cfg.run.driver)
    files = Base.invokelatest(preprocess_static, d, cfg; out = get(opts, "out", nothing),
                              parent_out = get(opts, "parent-out", nothing))
    for f in files
        println("wrote ", f)
    end
    return 0
end
"""
`aeolus preprocess dynamic (--from-sounding | --from-parent) <config.toml> [--child <child config>]
[--out file.nc] [--set key=value]...`: load the run config and write its dynamic driver with the
driver's [`preprocess_dynamic`](@ref) (CFG-g, #362): from the sounding or case tables the config
names (`--from-sounding`), or by running the config's nest parent with the child's boundary slabs
recorded at every parent step (`--from-parent`; `--child` names the child config when it is not
the same file); prints the files written.
"""
function _preprocess_dynamic(args)
    pos, sets, opts, fl = _parse_args(args, ("set", "out", "child"), ("from-sounding", "from-parent"))
    length(fl) == 1 || throw(UsageError("give one of --from-sounding, --from-parent"))
    from = "from-sounding" in fl ? :sounding : :parent
    path = _one_positional(pos, "config file")
    cfg = load_run_config(path; overrides = sets)
    d = run_driver(cfg.run.driver)
    child = haskey(opts, "child") ? load_run_config(opts["child"]; overrides = sets) : nothing
    files = Base.invokelatest(preprocess_dynamic, d, cfg; from = from, out = get(opts, "out", nothing),
                              child = child)
    for f in files
        println("wrote ", f)
    end
    return 0
end
register_preprocessor!("dynamic", _preprocess_dynamic,
                       "dynamic (--from-sounding | --from-parent) <config.toml> [--child <config.toml>] " *
                       "[--out file.nc] [--set key=value]...  write the run's dynamic input driver")

register_preprocessor!("static", _preprocess_static,
                       "static <config.toml> [--out file.nc] [--parent-out file.nc] [--set key=value]...  " *
                       "write the run's static input driver(s)")

# ---------------------------------------------------------------------------
# the option reference
# ---------------------------------------------------------------------------

function _type_text(T)
    T isa OpenTable && return "table"
    T isa OneOf && return join(map(_type_text, T.alts), " | ")
    T === Path && return "path"
    T <: Optional && return _type_text(_elem(T)) * " | \"none\""
    T <: Tuple && return "[$(fieldcount(T)) × $(_type_text(fieldtype(T, 1)))]"
    T <: Vector && return "[$(_type_text(eltype(T))), …]"
    T === Symbol && return "string"
    T === Float64 && return "float"
    T === Int && return "integer"
    T === Bool && return "boolean"
    return lowercase(string(nameof(T)))
end

function _default_text(o::Option)
    o.required && return "(required)"
    d = o.default isa Function ? o.default() : o.default
    ok, v, _ = _coerce(o, d, true, pwd(), o.key)
    s = ok ? _emit_value(_toml_out(v)) : _show_raw(d)
    return o.default isa Function ? s * " (from the environment if set)" : s
end

"""
    option_reference(driver = "common"; section = nothing) -> String

The option reference of a driver's schema, generated from the schema: one
block per section with its doc, then one entry per option with its type,
default, units, allowed values or range, and doc. `section` restricts it
to one section. `aeolus options [driver] [section]` prints it.
"""
function option_reference(driver::AbstractString = "common"; section = nothing)
    sch = driver_schema(driver)
    secs = section === nothing ? sch.sections : filter(s -> s.name == section, sch.sections)
    isempty(secs) && throw(ConfigError(String(section), "driver \"$driver\" has no section [$section]" *
                                       _suggest(String(section), [s.name for s in sch.sections])))
    io = IOBuffer()
    println(io, "# options of driver \"", sch.name, "\"")
    for s in secs
        println(io, "\n[", s.name, "]", isempty(s.doc) ? "" : "  " * s.doc)
        for o in s.options
            print(io, "  ", o.key, " : ", _type_text(o.type), " = ", _default_text(o))
            isempty(o.units) || print(io, "  [", o.units, "]")
            c = _constraint_text(o)
            isempty(c) || print(io, " ", strip(c))
            println(io)
            isempty(o.doc) || println(io, "      ", o.doc)
        end
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# the commands
# ---------------------------------------------------------------------------

function _cmd_check(args, io, err)
    pos, sets, _, _ = _parse_args(args, ("set",), ())
    path = _one_positional(pos, "config file")
    for (m, cfg) in load_sweep(path; overrides = sets)
        println(io, "ok: ", path, isempty(m.label) ? "" : " member $(m.index) ($(m.label))",
                " (driver \"", config_schema(cfg).name, "\", config_hash ", config_hash_string(cfg), ")")
    end
    return 0
end

function _cmd_resolve(args, io, err)
    pos, sets, opts, _ = _parse_args(args, ("set", "member"), ())
    runs = load_sweep(_one_positional(pos, "config file"); overrides = sets, member = _member_arg(opts))
    for (i, (m, cfg)) in enumerate(runs)
        length(runs) > 1 && println(io, i == 1 ? "" : "\n", "# member ", m.index, " (", m.label, ")")
        print(io, config_text(cfg))
    end
    return 0
end

function _cmd_options(args, io, err)
    pos, _, _, _ = _parse_args(args, (), ())
    length(pos) <= 2 || throw(UsageError("options takes at most a driver and a section"))
    driver = isempty(pos) ? "common" : pos[1]
    print(io, option_reference(driver; section = length(pos) == 2 ? pos[2] : nothing))
    if isempty(pos)
        println(io, "\nregistered drivers: ", join(registered_drivers(), ", "),
                isempty(DRIVER_FILES) ? "" : "; example drivers: " * join(sort!(collect(keys(DRIVER_FILES))), ", "))
    end
    return 0
end

function _cmd_preprocess(args, io, err)
    if isempty(args)
        println(io, "preprocessors:")
        isempty(_PREPROCESSORS) && println(io, "  (none registered)")
        for k in sort!(collect(keys(_PREPROCESSORS)))
            println(io, "  ", k, "  ", _PREPROCESSORS[k][2])
        end
        return 0
    end
    kind = String(args[1])
    haskey(_PREPROCESSORS, kind) ||
        throw(UsageError("unknown preprocessor \"$kind\"" * _suggest(kind, keys(_PREPROCESSORS)) *
                         "; registered: " * (isempty(_PREPROCESSORS) ? "none" :
                                             join(sort!(collect(keys(_PREPROCESSORS))), ", "))))
    r = _PREPROCESSORS[kind][1](String[a for a in args[2:end]])
    return r === nothing ? 0 : Int(r)
end

function _cmd_run(args, io, err)
    pos, sets, opts, fl = _parse_args(args, ("set", "root", "member"), ("resume", "allow-uncommitted"))
    path = _one_positional(pos, "config file, run.toml or run folder")
    resume = "resume" in fl
    root = get(opts, "root", nothing)
    allow = "allow-uncommitted" in fl ? true : nothing
    folder = nothing
    if resume
        isempty(sets) || throw(UsageError("--resume continues a run as it was configured; " *
                                          "--set is not allowed (start a new run instead)"))
        folder = isdir(path) ? abspath(path) :
                 basename(path) == "run.toml" ? dirname(abspath(path)) :
                 throw(UsageError("--resume needs a run folder (or its run.toml), got $path"))
    end
    (resume && haskey(opts, "member")) && throw(UsageError("--member selects a sweep member to start; " *
                                                         "--resume continues a run folder"))
    runs = resume ? [SweepMember(1, "", String[]) => load_run_config(folder; overrides = sets)] :
                    load_sweep(path; overrides = sets, member = _member_arg(opts))
    for (_, cfg) in runs
        run_driver(cfg.run.driver)                 # an unknown driver is a config error
    end
    for (m, cfg) in runs
        local res
        try
            res = run_config(cfg; root = root, resume = folder, allow_uncommitted = allow, console = err)
        catch e
            e isa IO.UncommittedCodeError && rethrow()   # raised before anything is written: 2
            println(err, "run failed", isempty(m.label) ? "" : " (member $(m.index), $(m.label))", ": ",
                    first(sprint(showerror, e), 2000))
            return 1
        end
        rf = res.folder
        println(io, "run ", rf.run_id, " (session ", rf.session, ") completed: ", rf.path)
    end
    return 0
end

const _COMMANDS = Dict("run" => _cmd_run, "check" => _cmd_check, "resolve" => _cmd_resolve,
                       "options" => _cmd_options, "preprocess" => _cmd_preprocess)

"""
    main(args::Vector{String}; io = stdout, err = stderr) -> Cint

The `aeolus` command line: `run`, `check`, `resolve`, `options`,
`preprocess` and `help` (see the Run configuration page). Called as
`julia --project -m Aeolus <command> …` (Julia ≥ 1.12, through
`Aeolus.main`) or `bin/aeolus <command> …`. Returns the exit code: 0 on
success, 1 when the run failed (its session is recorded as `failed`),
2 for a usage or configuration error, which is raised before anything is
written. Messages go to `io`, errors and the run's console log to `err`.
"""
function main(args::AbstractVector{<:AbstractString}; io::Base.IO = stdout, err::Base.IO = stderr)
    if isempty(args) || args[1] in ("help", "--help", "-h")
        print(isempty(args) ? err : io, _USAGE)
        return Cint(isempty(args) ? 2 : 0)
    end
    cmd = String(args[1])
    f = get(_COMMANDS, cmd, nothing)
    if f === nothing
        println(err, "usage error: unknown command \"", cmd, "\"", _suggest(cmd, keys(_COMMANDS)))
        print(err, _USAGE)
        return Cint(2)
    end
    try
        return Cint(f(args[2:end], io, err))
    catch e
        if e isa UsageError
            println(err, sprint(showerror, e), "\n(aeolus help: the commands and their options)")
            return Cint(2)
        elseif e isa ConfigError || e isa IO.UncommittedCodeError
            println(err, sprint(showerror, e))
            return Cint(2)
        elseif e isa ArgumentError && cmd == "preprocess"
            println(err, sprint(showerror, e))
            return Cint(2)
        end
        println(err, "error: ", first(sprint(showerror, e, catch_backtrace()), 4000))
        return Cint(1)
    end
end
