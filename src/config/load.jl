# Loading a run configuration (CFG-a, #356): the `extends` chain, the
# key-by-key merge, `dotted.key=<TOML value>` overrides, and the strict
# validation that reports every problem in one ConfigError.

"The schema version this Aeolus writes and reads."
const CONFIG_SCHEMA_VERSION = 1

"The deepest `extends` chain accepted (number of `extends` links)."
const MAX_EXTENDS_DEPTH = 8

"Origin tag of a value set by an override (`--set` / `overrides`)."
const OVERRIDE_ORIGIN = "--set"
"Origin tag of a value set from Julia ([`with`](@ref))."
const WITH_ORIGIN = "with()"
"Origin tag of a default value."
const DEFAULT_ORIGIN = "default"

function _parse_toml_file(path::AbstractString)
    r = TOML.tryparsefile(path)
    r isa TOML.ParserError && throw(ConfigError("", "not valid TOML: " * sprint(showerror, r); file = path))
    return r
end

"""
    load_run_config(path; overrides = String[], schema = nothing,
                    allow_changed_inputs = false) -> RunConfig
    load_run_config(io::Base.IO; …)

Load, merge, validate and resolve a run configuration.

`path` is a `.toml` file, a run folder or its `run.toml`; for the last two
the canonical `[config]` table of the record is used, the input files
recorded under `[inputs]` must be unchanged (else a [`ConfigError`](@ref)
names each changed file, unless `allow_changed_inputs = true`), and a
record marked `replayable = false` is refused.

A `.toml` file may name a base file with `extends = "base.toml"`, looked
up relative to the extending file first, then in the driver's config
directory ([`register_driver!`](@ref)); chains are followed up to
$(MAX_EXTENDS_DEPTH) links deep, and a cycle is an error. Tables merge key by key;
arrays and scalars replace. `overrides` are `"dotted.key=<TOML value>"`
strings (`"clock.nsteps=12"`, `"grid.size=[16,16,16]"`, `"run.name=\\"x\\""`),
applied last: precedence is base < child < override.

The schema is `schema`, or that of the driver named by `[run] driver`
(default `"common"`). Unknown keys, wrong types, values out of range or
outside `allowed`, tuples of the wrong length and failed cross-field
checks are all collected and thrown as one `ConfigError`.
"""
function load_run_config(path::AbstractString; overrides = String[], schema = nothing,
                         allow_changed_inputs::Bool = false)
    p = abspath(path)
    if isdir(p)
        rt = joinpath(p, "run.toml")
        isfile(rt) || throw(ConfigError("", "not a run folder (no run.toml)"; file = p))
        return _load_record(rt, overrides, schema, allow_changed_inputs)
    end
    isfile(p) || throw(ConfigError("", "configuration file not found"; file = p))
    d = _parse_toml_file(p)
    if haskey(d, "run_id") && get(d, "config", nothing) isa AbstractDict
        return _load_record(p, overrides, schema, allow_changed_inputs; parsed = d)
    end
    return _load_chain(p, d, overrides, schema)
end

function load_run_config(io::Base.IO; overrides = String[], schema = nothing,
                         allow_changed_inputs::Bool = false)
    text = read(io, String)
    r = TOML.tryparse(text)
    r isa TOML.ParserError && throw(ConfigError("", "not valid TOML: " * sprint(showerror, r); file = "<io>"))
    file = joinpath(pwd(), "<io>")
    if haskey(r, "run_id") && get(r, "config", nothing) isa AbstractDict
        return _load_record(file, overrides, schema, allow_changed_inputs; parsed = r)
    end
    return _load_chain(file, r, overrides, schema)
end

# ---------------------------------------------------------------------------
# the extends chain
# ---------------------------------------------------------------------------

function _load_chain(leaf::String, leafdict::AbstractDict, overrides, schema; sweep::Symbol = :error)
    ovr, oprobs = _parse_overrides(overrides)
    files = String[leaf]; dicts = Dict{String,Any}[deepcopy(leafdict)]
    seen = Set{String}([_real(leaf)])
    while true
        cur = files[end]; d = dicts[end]
        haskey(d, "extends") || break
        ext = pop!(d, "extends")
        ext isa AbstractString ||
            throw(ConfigError("extends", "expected a file name (a string), got $(_show_raw(ext))"; file = cur))
        length(files) > MAX_EXTENDS_DEPTH &&
            throw(ConfigError("extends", "the extends chain is deeper than $(MAX_EXTENDS_DEPTH): " *
                              join(files, " -> "); file = cur))
        hint = _driver_hint(ovr, dicts, schema)
        next = _find_extends(String(ext), cur, hint)
        r = _real(next)
        r in seen && throw(ConfigError("extends", "extends cycle: " * join(vcat(files, next), " -> ");
                                       file = cur))
        push!(seen, r)
        push!(files, next); push!(dicts, _parse_toml_file(next))
    end
    raw = Dict{String,Any}(); origins = Dict{String,String}()
    for i in length(files):-1:1                     # root first, the leaf last
        _merge!(raw, origins, dicts[i], files[i], "")
    end
    # a [sweep] (CFG-e) is expanded by the runner, never part of a resolved config
    sw = pop!(raw, "sweep", nothing)
    sweep === :table && return sw
    if sw !== nothing && sweep === :error
        n = sw isa AbstractDict ? length(sweep_members(sw, leaf)) : 0
        throw(_sweep_error(leaf, n))
    end
    _apply_overrides!(raw, origins, ovr, oprobs)
    sources = Dict{String,Any}("files" => files, "overrides" => String[overrides...])
    return _resolve(raw, origins, sources, schema, oprobs, leaf)
end

_real(p) = ispath(p) ? realpath(p) : abspath(p)

"The driver whose config directory `extends` searches: the overrides', else the child-most file's."
function _driver_hint(ovr, dicts, schema)
    schema === nothing || return schema.name
    for (k, v) in ovr
        k == "run.driver" && v isa AbstractString && return String(v)
    end
    for d in dicts
        r = get(d, "run", nothing)
        r isa AbstractDict && get(r, "driver", nothing) isa AbstractString && return String(r["driver"])
    end
    return nothing
end

function _find_extends(ext::String, cur::String, hint)
    cand = normpath(joinpath(dirname(cur), ext))
    isfile(cand) && return cand
    looked = String[cand]
    dirs = String[]
    hint === nothing || _ensure_driver(hint)
    if hint !== nothing && haskey(_DRIVERS, hint)
        d = _DRIVERS[hint].config_dir
        d === nothing || push!(dirs, d)
    elseif hint === nothing
        for e in values(_DRIVERS)
            e.config_dir === nothing || push!(dirs, e.config_dir)
        end
    end
    hits = String[]
    for d in sort!(unique!(dirs))
        c = normpath(joinpath(d, ext)); push!(looked, c)
        isfile(c) && push!(hits, c)
    end
    length(hits) == 1 && return hits[1]
    length(hits) > 1 &&
        throw(ConfigError("extends", "\"$ext\" is ambiguous: found in " * join(hits, ", ") *
                          "; set [run] driver to pick one"; file = cur))
    throw(ConfigError("extends", "extends file \"$ext\" not found (looked at " * join(looked, ", ") * ")";
                      file = cur))
end

"Merge `src` into `dst`: tables key by key, arrays and scalars replace; record each key's file."
function _merge!(dst::AbstractDict, origins, src::AbstractDict, file, prefix)
    for (k, v) in src
        key = isempty(prefix) ? k : prefix * "." * k
        if v isa AbstractDict && get(dst, k, nothing) isa AbstractDict
            origins[key] = file
            _merge!(dst[k], origins, v, file, key)
        else
            _forget!(origins, key)
            if v isa AbstractDict
                dst[k] = Dict{String,Any}()
                origins[key] = file
                _merge!(dst[k], origins, v, file, key)
            else
                dst[k] = v
                origins[key] = file
            end
        end
    end
    return dst
end

_forget!(origins, key) = filter!(p -> !(p.first == key || startswith(p.first, key * ".")), origins)

# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------

"Parse `\"dotted.key=<TOML value>\"` strings into `key => value` pairs, collecting problems."
function _parse_overrides(overrides)
    out = Pair{String,Any}[]; probs = ConfigProblem[]
    for s in overrides
        i = findfirst('=', s)
        if i === nothing
            push!(probs, ConfigProblem(String(s), OVERRIDE_ORIGIN,
                                       "an override is \"dotted.key=<TOML value>\", got \"$s\""))
            continue
        end
        key = strip(s[1:prevind(s, i)]); val = strip(s[nextind(s, i):end])
        parts = split(key, '.')
        # a part is a TOML bare key, or letters and digits of any script
        # (the Berlin per-coupling options keep their Julia names: "τ", "C_σ")
        if isempty(key) || any(p -> !occursin(r"^[\p{L}\p{N}_-]+$", p), parts)
            push!(probs, ConfigProblem(String(key), OVERRIDE_ORIGIN, "not a dotted key: \"$key\""))
            continue
        end
        r = TOML.tryparse("v = " * val)
        if r isa TOML.ParserError || !haskey(r, "v")
            push!(probs, ConfigProblem(String(key), OVERRIDE_ORIGIN,
                "the value \"$val\" is not a TOML value (strings are quoted: $key=\"$val\")"))
            continue
        end
        push!(out, String(key) => r["v"])
    end
    return out, probs
end

function _apply_overrides!(raw, origins, ovr, probs)
    for (key, v) in ovr
        parts = split(key, '.')
        t = raw; ok = true
        for p in parts[1:end-1]
            nxt = get!(t, p) do
                Dict{String,Any}()
            end
            if !(nxt isa AbstractDict)
                push!(probs, ConfigProblem(key, OVERRIDE_ORIGIN, "\"$p\" is not a table"))
                ok = false; break
            end
            t = nxt
        end
        ok || continue
        _forget!(origins, key)
        if v isa AbstractDict
            t[parts[end]] = Dict{String,Any}()
            _merge!(t[parts[end]], origins, v, OVERRIDE_ORIGIN, key)
        else
            t[parts[end]] = v
        end
        origins[key] = OVERRIDE_ORIGIN
    end
    return raw
end

# ---------------------------------------------------------------------------
# validation and resolution
# ---------------------------------------------------------------------------

_origin(origins, key, fallback) = get(origins, key, fallback)
_base_dir(origin) = (origin == OVERRIDE_ORIGIN || origin == WITH_ORIGIN || origin == DEFAULT_ORIGIN) ?
                    pwd() : dirname(origin)

function _pick_schema(raw, schema, leaf)
    schema === nothing || return schema
    r = get(raw, "run", nothing)
    name = r isa AbstractDict ? get(r, "driver", "common") : "common"
    name isa AbstractString ||
        throw(ConfigError("run.driver", "expected a string, got $(_show_raw(name))"; file = leaf))
    try
        return driver_schema(String(name))
    catch e
        e isa ConfigError || rethrow()
        throw(ConfigError([ConfigProblem(p.key, leaf, p.message) for p in e.problems]))
    end
end

function _check_version(raw, origins, leaf)
    v = pop!(raw, "schema_version", nothing)
    v === nothing && return
    file = _origin(origins, "schema_version", leaf)
    (v isa Integer && !(v isa Bool)) ||
        throw(ConfigError("schema_version", "expected an integer, got $(_show_raw(v))"; file = file))
    v > CONFIG_SCHEMA_VERSION &&
        throw(ConfigError("schema_version", "schema_version = $v is newer than this Aeolus supports " *
                          "(schema_version = $(CONFIG_SCHEMA_VERSION)); update Aeolus"; file = file))
    v < 1 && throw(ConfigError("schema_version", "schema_version must be ≥ 1, got $v"; file = file))
    return
end

"""
Validate the merged raw table against the schema, resolve defaults and
paths, run the cross-field checks; throw one ConfigError with every problem.
"""
function _resolve(raw, origins, sources, schema, probs::Vector{ConfigProblem}, leaf)
    _check_version(raw, origins, leaf)
    sch = _pick_schema(raw, schema, leaf)
    names = [s.name for s in sch.sections]
    for k in sort!(collect(keys(raw)))
        k in names && continue
        push!(probs, ConfigProblem(k, _origin(origins, k, leaf),
              "unknown " * (raw[k] isa AbstractDict ? "table [$k]" : "key") *
              " for driver \"$(sch.name)\"" * _suggest(k, names)))
    end
    values = Dict{String,Dict{String,Any}}()
    vorig = Dict{String,String}()
    for s in sch.sections
        tbl = get(raw, s.name, Dict{String,Any}())
        if !(tbl isa AbstractDict)
            push!(probs, ConfigProblem(s.name, _origin(origins, s.name, leaf),
                                       "expected a table [$(s.name)], got $(_show_raw(tbl))"))
            tbl = Dict{String,Any}()
        end
        keys_ = [o.key for o in s.options]
        for k in sort!(collect(keys(tbl)))
            k in keys_ && continue
            push!(probs, ConfigProblem("$(s.name).$k", _origin(origins, "$(s.name).$k", leaf),
                  "unknown key in [$(s.name)]" * _suggest(k, keys_)))
        end
        vals = Dict{String,Any}()
        for o in s.options
            key = "$(s.name).$(o.key)"
            if haskey(tbl, o.key)
                org = _origin(origins, key, leaf)
                ok, v, why = _coerce(o, tbl[o.key], false, _base_dir(org), key)
                ok ? (vals[o.key] = v) : push!(probs, ConfigProblem(key, org, why))
                vorig[key] = org
            elseif o.required
                push!(probs, ConfigProblem(key, leaf, "required, expected $(_describe(o.type))$(_constraint_text(o))"))
            else
                ok, v, why = _default(o, key)
                ok ? (vals[o.key] = v) : push!(probs, ConfigProblem(key, DEFAULT_ORIGIN, why))
                vorig[key] = DEFAULT_ORIGIN
            end
        end
        values[s.name] = vals
    end
    isempty(probs) || throw(ConfigError(_sorted(probs)))
    return _finish(sch, values, vorig, sources)
end

"The default of `o`, evaluated (a function is called) and checked."
function _default(o::Option, key)
    # invokelatest: a driver loaded from DRIVER_FILES during this load is newer than the caller's world
    d = o.default isa Function ? Base.invokelatest(o.default) : o.default
    ok, v, why = _coerce(o, d, true, pwd(), key)
    ok || return (false, nothing, "the default $(_show_raw(d)) is invalid: $why")
    return (true, v, "")
end

_sorted(probs) = sort(probs; by = p -> (p.key, p.file, p.message))

"Build the RunConfig and run the cross-field checks."
function _finish(sch::ConfigSchema, values, vorig, sources)
    cfg = RunConfig(sch, Dict{String,ConfigSection}(n => ConfigSection(n, v) for (n, v) in values),
                    vorig, sources, _nonreplayable(values))
    probs = ConfigProblem[]
    for (_, f) in sch.checks
        for e in Base.invokelatest(f, cfg)     # an example driver may have been loaded just now
            for p in (e isa ConfigError ? e.problems : [ConfigProblem("", "", string(e))])
                push!(probs, isempty(p.file) ? ConfigProblem(p.key, get(vorig, p.key, ""), p.message) : p)
            end
        end
    end
    isempty(probs) || throw(ConfigError(_sorted(probs)))
    return cfg
end

# ---------------------------------------------------------------------------
# from a run record
# ---------------------------------------------------------------------------

function _load_record(rt::String, overrides, schema, allow_changed_inputs::Bool; parsed = nothing)
    rec = parsed === nothing ? _parse_toml_file(rt) : parsed
    cfgt = get(rec, "config", nothing)
    cfgt isa AbstractDict ||
        throw(ConfigError("config", "the run record has no [config] table"; file = rt))
    if get(rec, "replayable", true) == false
        throw(ConfigError("config", "this run is not replayable: it was configured from Julia with " *
                          "values TOML cannot hold (" * join(get(rec, "nonreplayable_keys", String[]), ", ") *
                          ")"; file = rt))
    end
    haskey(rec, "config_schema_version") || haskey(cfgt, "schema_version") ||
        throw(ConfigError("config", "the run record's [config] was not written from a RunConfig " *
                          "(no config_schema_version)"; file = rt))
    allow_changed_inputs || _check_inputs(get(rec, "inputs", Dict{String,Any}()), rt)
    ovr, oprobs = _parse_overrides(overrides)
    raw = deepcopy(Dict{String,Any}(cfgt)); origins = Dict{String,String}()
    _merge!(Dict{String,Any}(), origins, raw, rt, "")
    _apply_overrides!(raw, origins, ovr, oprobs)
    sources = Dict{String,Any}("files" => String[rt], "overrides" => String[overrides...])
    return _resolve(raw, origins, sources, schema, oprobs, rt)
end
