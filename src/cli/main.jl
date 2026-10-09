# The aeolus command line (CFG-b, #357): one entry point for every run.
#
#   julia --project -m Aeolus run config.toml        (Julia ≥ 1.12)
#   bin/aeolus run config.toml                       (any Julia)
#
# Argument parsing is in-package (no ArgParse/Comonicon dependency).
# Exit codes: 0 success, 1 the run failed, 2 a usage or config error, 3 a required
# check failed (ONE-d, #453).

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
  options [legacy-case] [section] [--markdown]
        Print the option reference of the model (all sections and the cross-field
        checks, or one section), or of a legacy case; --markdown: as the tables of
        the docs.
  help
        Print this message.

Overrides are "dotted.key=<TOML value>": --set time.nsteps=20,
--set 'run.name="tgv"' (a per-domain section's key: in every domain). Exit codes: 0 success, 1 the run failed,
2 a usage or configuration error (nothing is written), 3 the run completed but a required check of [checks]
failed (output/results.toml says which).
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
# the option reference
# ---------------------------------------------------------------------------

function _type_text(T)
    T isa AbstractString && return String(T)              # a reference_option
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

const _FROM_ENV = " (from the environment if set)"

function _default_text(o::Option)
    o.type isa AbstractString && return string(o.default)  # a reference_option
    o.required && return "(required)"
    d = o.default isa Function ? o.default() : o.default
    ok, v, _ = _coerce(o, d, true, pwd(), o.key)
    s = ok ? _emit_value(_toml_out(v)) : _show_raw(d)
    return o.default isa Function ? s * _FROM_ENV : s
end

"""
    option_reference(name = "model"; section = nothing, format = :text) -> String
    option_reference(schema::ConfigSchema; section = nothing, format = :text) -> String

The option reference of the model's schema, or of a legacy case's (`name`),
generated from the schema (the
one source of truth: nothing is written by hand). For each section its
doc, then for each option its key, type, default, units, allowed values
or range, and doc; after a section, the documented sub-tables of its
[`OpenTable`](@ref) options ([`reference_sections`](@ref)); after the
sections, the schema's cross-field checks in words ([`check!`](@ref)). `section` restricts it to one section (and
leaves the checks out). A per-domain section ([`ConfigSchema`](@ref)) is
marked so.

`format = :text` is what `aeolus options [legacy-case] [section]` prints;
`format = :markdown` is one table per section (columns key, type,
default, units, allowed / range, doc; one row per option) and a bullet
list of the checks: `aeolus options --markdown` prints it and the docs
build writes it to `docs/src/reference/options/<name>.md`.
"""
function option_reference(driver::AbstractString = MODEL; section = nothing, format::Symbol = :text)
    return option_reference(driver_schema(driver); section = section, format = format)
end

function option_reference(sch::ConfigSchema; section = nothing, format::Symbol = :text)
    format in (:text, :markdown) ||
        throw(ArgumentError("option_reference: format is :text or :markdown, got :$format"))
    secs = section === nothing ? sch.sections : filter(s -> s.name == section, sch.sections)
    isempty(secs) && throw(ConfigError(String(section), _schema_title(sch) * " has no section [$section]" *
                                       _suggest(String(section), [s.name for s in sch.sections])))
    checks = section === nothing ? _check_lines(sch) : String[]
    blocks = filter(b -> section === nothing || split(b.first, '.')[1] == section, reference_sections(sch))
    return format === :text ? _reference_text(sch, blocks, checks) : _reference_markdown(sch, blocks, checks)
end

"The documented sub-tables of an [`OpenTable`](@ref) option (none for any other)."
_subtables(o::Option) = (o.type isa OpenTable && o.type.reference !== nothing) ?
                        Section[Base.invokelatest(o.type.reference)...] : Section[]

"""
    reference_sections(schema) -> Vector{Pair{String,Section}}

Every table the option reference of `schema` documents, `"dotted.name" =>
section`: the schema's sections and, after each, the documented
sub-tables of its [`OpenTable`](@ref) options (`"physics.options.aq_chem"`).
"""
function reference_sections(sch::ConfigSchema)
    out = Pair{String,Section}[]
    for s in sch.sections
        push!(out, s.name => s)
        for o in s.options, sub in _subtables(o)
            push!(out, "$(s.name).$(o.key).$(sub.name)" => sub)
        end
    end
    return out
end

"""
    undocumented_keys(cfg::RunConfig) -> Vector{String}

The dotted keys a resolved config holds that the option reference of its
schema does not document: empty for a config that passed validation,
except inside an [`OpenTable`](@ref) with documented sub-tables (the
Berlin `[physics.options]`), whose keys are checked against those
sub-tables. A key whose value is a table and that is itself documented
(a forcing given as `{kind = …}`) counts as documented. The round-trip
coverage suite requires it to be empty for every shipped config.
"""
function undocumented_keys(cfg::RunConfig)
    sch = config_schema(cfg)
    out = String[]
    for s in sch.sections, o in s.options
        subs = _subtables(o)
        isempty(subs) && continue
        names = Dict(x.name => x for x in subs)
        v = getproperty(cfg, Symbol(s.name))[o.key]
        v isa AbstractDict && _undocumented!(out, "$(s.name).$(o.key)", "", v, names)
    end
    return sort!(out)
end

function _undocumented!(out, base, rel, tbl, names)
    sec = get(names, rel, nothing)
    for k in sort!(collect(String, string.(keys(tbl))))
        v = tbl[k]
        r = isempty(rel) ? k : rel * "." * k
        if v isa AbstractDict && haskey(names, r)
            _undocumented!(out, base, r, v, names)         # a documented sub-table
        elseif sec !== nothing && _option(sec, k) !== nothing
            continue                                       # a documented option (any value)
        else
            push!(out, base * "." * r)
        end
    end
    return out
end

"The cross-field checks of `sch` in words, one line each (a check without a description: \"(undocumented)\")."
function _check_lines(sch::ConfigSchema)
    out = String[]
    for (doc, _) in sch.checks
        ls = filter(!isempty, strip.(split(doc, '\n')))
        isempty(ls) ? push!(out, "(undocumented)") : append!(out, ls)
    end
    return out
end

"`the model` or `the legacy case \"x\"`."
_schema_title(sch) = sch.name == MODEL ? "the model" : "the legacy case \"$(sch.name)\""

const _PER_DOMAIN = "Per domain: the run-level value applies to every domain; a [[domain]] block's table overrides it."
const _DOMAIN_KEYS = "The domain's own keys: at the top level of a single-domain config, or in each [[domain]] block."

"The doc of a reference block, with the per-domain note of a per-domain section."
function _block_doc(sch, name, s)
    is_domain_section(sch, name) || return s.doc
    note = name == "domain" ? _DOMAIN_KEYS : _PER_DOMAIN
    return isempty(s.doc) ? note : s.doc * " " * note
end

function _reference_text(sch, blocks, checks)
    io = IOBuffer()
    println(io, "# options of ", _schema_title(sch))
    for (name, s) in blocks
        d = _block_doc(sch, name, s)
        println(io, "\n[", name, "]", isempty(d) ? "" : "  " * d)
        for o in s.options
            print(io, "  ", o.key, " : ", _type_text(o.type), " = ", _default_text(o))
            isempty(o.units) || print(io, "  [", o.units, "]")
            c = _constraint_text(o)
            isempty(c) || print(io, " ", strip(c))
            println(io)
            isempty(o.doc) || println(io, "      ", o.doc)
        end
    end
    if !isempty(checks)
        println(io, "\ncross-field checks:")
        foreach(l -> println(io, "  - ", l), checks)
    end
    return String(take!(io))
end

"A Markdown table cell: `|` escaped, line breaks as spaces."
_md_cell(s) = replace(String(s), '|' => "\\|", '\n' => ' ')

"""
Prose for a Markdown cell or paragraph: `_md_cell`, and `_` and `*` escaped
outside code spans, so `AEOLUS_LOG` or `set_fftw_policy!` in a doc string is
not read as emphasis.
"""
function _md_prose(s)
    parts = split(_md_cell(s), '`')
    for i in 1:2:length(parts)                    # the odd pieces are outside code spans
        parts[i] = replace(parts[i], '_' => "\\_", '*' => "\\*")
    end
    return join(parts, '`')
end

"The allowed values or the range of `o`, for the Markdown table."
function _md_constraint(o::Option)
    parts = String[]
    if o.range !== nothing
        lo, hi = o.range
        b(x) = x == typemax(Int) || x == typemax(Int32) ? "∞" : string(x)
        push!(parts, (o.open_lower ? "(" : "[") * b(lo) * ", " * b(hi) * "]")
    end
    o.allowed === nothing || push!(parts, join(("`" * _show_raw(a) * "`" for a in o.allowed), ", "))
    return join(parts, "; ")
end

function _reference_markdown(sch, blocks, checks)
    io = IOBuffer()
    m = sch.name == MODEL
    println(io, m ? "Options of the model" : "Options of the legacy case `$(sch.name)`", ", generated from its schema ",
            "(`Aeolus.Config.option_reference(\"", sch.name, "\"; format = :markdown)`, ",
            "`aeolus options ", m ? "" : sch.name * " ", "--markdown`). TOML has no null: `\"none\"` stands for nothing.")
    for (name, s) in blocks
        println(io, "\n", count('.', name) == 0 ? "## " : "### ", "`[", name, "]`\n")
        d = _block_doc(sch, name, s)
        isempty(d) || println(io, _md_prose(d), "\n")
        println(io, "| key | type | default | units | allowed / range | doc |")
        println(io, "| :-- | :-- | :-- | :-- | :-- | :-- |")
        for o in s.options
            d = _default_text(o)
            env = endswith(d, _FROM_ENV)
            env && (d = d[1:end-length(_FROM_ENV)])
            println(io, "| `", o.key, "` | ", _md_cell(_type_text(o.type)), " | `", _md_cell(d), "`",
                    env ? " (from the environment if set)" : "", " | ",
                    isempty(o.units) ? "" : "`" * _md_cell(o.units) * "`", " | ",
                    _md_cell(_md_constraint(o)), " | ", _md_prose(o.doc), " |")
        end
    end
    if !isempty(checks)
        println(io, "\n## Cross-field checks\n")
        println(io, "Checked after every option has passed its own checks; every failure is reported ",
                "with the others in one `ConfigError`.\n")
        foreach(l -> println(io, "- ", _md_prose(l)), checks)
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
                " (", _schema_title(config_schema(cfg)), ", config_hash ", config_hash_string(cfg), ")")
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
    pos, _, _, fl = _parse_args(args, (), ("markdown",))
    length(pos) <= 2 || throw(UsageError("options takes at most a legacy case and a section"))
    # [legacy-case] [section]: a first word that names a legacy case selects it
    legacy = !isempty(pos) && pos[1] in legacy_cases()
    length(pos) == 2 && !legacy &&
        throw(UsageError("\"$(pos[1])\" is not a legacy case" * _suggest(pos[1], legacy_cases()) *
                         "; legacy cases: " * join(legacy_cases(), ", ")))
    name = legacy ? pos[1] : MODEL
    sec = legacy ? (length(pos) == 2 ? pos[2] : nothing) : (isempty(pos) ? nothing : pos[1])
    print(io, option_reference(name; section = sec, format = "markdown" in fl ? :markdown : :text))
    isempty(pos) && println(io, "\nlegacy cases (aeolus options <case>): ", join(legacy_cases(), ", "))
    return 0
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
        run_driver(cfg)                            # an unknown legacy case is a config error
    end
    for (m, cfg) in runs
        local res
        try
            res = run_config(cfg; root = root, resume = folder, allow_uncommitted = allow, console = err)
        catch e
            e isa IO.UncommittedCodeError && rethrow()   # raised before anything is written: 2
            println(err, "run failed", isempty(m.label) ? "" : " (member $(m.index), $(m.label))", ": ",
                    first(sprint(showerror, e), 2000))
            return e isa ChecksFailed ? 3 : 1
        end
        rf = res.folder
        println(io, "run ", rf.run_id, " (session ", rf.session, ") completed: ", rf.path)
    end
    return 0
end

const _COMMANDS = Dict("run" => _cmd_run, "check" => _cmd_check, "resolve" => _cmd_resolve,
                       "options" => _cmd_options)

"""
    main(args::Vector{String}; io = stdout, err = stderr) -> Cint

The `aeolus` command line: `run`, `check`, `resolve`, `options` and
`help` (see the Run configuration page). Input drivers are made by
AeolusPrep.jl (`julia --project -m AeolusPrep …`; ONE-h, #457). Called as
`julia --project -m Aeolus <command> …` (Julia ≥ 1.12, through
`Aeolus.main`) or `bin/aeolus <command> …`. Returns the exit code: 0 on
success, 1 when the run failed (its session is recorded as `failed`),
2 for a usage or configuration error, which is raised before anything is
written, and 3 when the run completed but a required check of `[checks]`
failed (`Aeolus.Checks.ChecksFailed`; the session is recorded as `failed`,
`output/results.toml` says which). Messages go to `io`, errors and the run's console log to `err`.
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
        end
        println(err, "error: ", first(sprint(showerror, e, catch_backtrace()), 4000))
        return Cint(1)
    end
end
