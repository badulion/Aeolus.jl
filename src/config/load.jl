# Loading a run configuration (CFG-a, #356; schema 2, ONE-b #451): one
# self-contained file, `dotted.key=<TOML value>` overrides, the domains
# (`[[domain]]` blocks over the run-level sections), and the strict
# validation that reports every problem in one ConfigError.

"""
The schema version this Aeolus writes and reads: 2 (ONE-b, #451: one model,
no `[run] driver`, no `extends`, `[[domain]]`). A file of version 1 is
refused with what changed.
"""
const CONFIG_SCHEMA_VERSION = 2

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

A config file is self-contained: there is no `extends` (a file that has
one is refused). `overrides` are `"dotted.key=<TOML value>"` strings
(`"time.nsteps=12"`, `"grid.size=[16,16,16]"`, `"run.name=\\"x\\""`),
applied last; an override of a per-domain section applies to every domain,
over the `[[domain]]` blocks.

The schema is `schema`, or that of the legacy case named by `[run]
legacy`, or else the model's ([`driver_schema`](@ref)). The model's
per-domain sections are resolved for every domain: the run-level values,
replaced key by key by the domain's `[domain.<section>]` tables (a config
without `[[domain]]` has one domain, `"main"`). Unknown keys, wrong types,
values out of range or outside `allowed`, tuples of the wrong length and
failed cross-field checks are all collected and thrown as one
`ConfigError`.
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
    return _load_file(p, d, overrides, schema)
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
    return _load_file(file, r, overrides, schema)
end

# ---------------------------------------------------------------------------
# one file
# ---------------------------------------------------------------------------

function _load_file(leaf::String, leafdict::AbstractDict, overrides, schema; sweep::Symbol = :error)
    ovr, oprobs = _parse_overrides(overrides)
    d = _drop_test!(deepcopy(leafdict), leaf)
    haskey(d, "extends") &&
        throw(ConfigError("extends", "extends was removed in schema_version 2 (ONE-b, #451): every config " *
                          "is self-contained; write the options of $(_show_raw(d["extends"])) into this file"; file = leaf))
    raw = Dict{String,Any}(); origins = Dict{String,String}()
    _merge!(raw, origins, d, leaf, "")
    # a [sweep] (CFG-e) is expanded by the runner, never part of a resolved config
    sw = pop!(raw, "sweep", nothing)
    sweep === :table && return sw
    if sw !== nothing && sweep === :error
        n = sw isa AbstractDict ? length(sweep_members(sw, leaf)) : 0
        throw(_sweep_error(leaf, n))
    end
    _apply_overrides!(raw, origins, ovr, oprobs)
    sources = Dict{String,Any}("files" => String[leaf], "overrides" => String[overrides...])
    return _resolve(raw, origins, sources, schema, oprobs, leaf)
end

# ---------------------------------------------------------------------------
# the [test] table (CFG-i, #364)
# ---------------------------------------------------------------------------

"""
    config_test_table(path) -> Union{Nothing, NamedTuple}

The `[test]` table of the config file `path`, or `nothing` if it has none:

```toml
[test]
overrides = ["grid.size=[16,16,16]", "stepper.nsteps=4"]   # the smoke size
member = "lasd_32x32x32"                                   # a sweep: the member replayed (default: the first)
```

`overrides` are `dotted.key=<TOML value>` strings that make the file a
smoke run; the round-trip coverage suite (`test/config_roundtrip.jl`)
runs every shipped config with them and replays it from its `run.toml`.
The loader ignores `[test]` (it checks only its shape): it is not part
of the resolved config, of `run.toml` or of the config hash.
"""
function config_test_table(path::AbstractString)
    d = _parse_toml_file(abspath(path))
    t = get(d, "test", nothing)
    t === nothing && return nothing
    _check_test_table(t, abspath(path))
    m = get(t, "member", nothing)
    return (overrides = String[get(t, "overrides", String[])...], member = m isa Integer ? Int(m) : m)
end

function _check_test_table(t, file)
    t isa AbstractDict || throw(ConfigError("test", "expected a table [test], got $(_show_raw(t))"; file = file))
    probs = ConfigProblem[]
    for k in sort!(collect(keys(t)))
        k in ("overrides", "member") || push!(probs, ConfigProblem("test.$k", file,
            "unknown key in [test] (keys: overrides, member)" * _suggest(k, ["overrides", "member"])))
    end
    o = get(t, "overrides", String[])
    (o isa AbstractVector && all(x -> x isa AbstractString, o)) ||
        push!(probs, ConfigProblem("test.overrides", file, "expected an array of \"dotted.key=<TOML value>\" strings, got $(_show_raw(o))"))
    m = get(t, "member", nothing)
    (m === nothing || m isa AbstractString || (m isa Integer && !(m isa Bool) && m >= 1)) ||
        push!(probs, ConfigProblem("test.member", file, "expected a member label (a string) or index (≥ 1), got $(_show_raw(m))"))
    isempty(probs) || throw(ConfigError(_sorted(probs)))
    return nothing
end

"Remove the `[test]` table of a config file (after checking its shape)."
function _drop_test!(d::AbstractDict, file)
    t = pop!(d, "test", nothing)
    t === nothing || _check_test_table(t, file)
    return d
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

"The prefix of the origin entries that keep the file's value of an overridden string key."
const _WAS = "\0was:"

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
        # the value it replaces, for a slot section's selector (ONE-m, #476):
        # an override that changes the provider resets its options
        old = get(t, parts[end], nothing)
        old isa AbstractString && !haskey(origins, _WAS * key) && (origins[_WAS * key] = String(old))
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
    name = r isa AbstractDict ? get(r, "legacy", MODEL) : MODEL
    name isa AbstractString ||
        throw(ConfigError("run.legacy", "expected a string, got $(_show_raw(name))"; file = leaf))
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
    v < CONFIG_SCHEMA_VERSION &&
        throw(ConfigError("schema_version", "schema_version = $v is no longer read: schema_version 2 " *
                          "(ONE-b, #451) has one model and no run types: `[run] driver` and `extends` are " *
                          "gone, a case not yet ported to components is `[run] legacy = \"<case>\"`, and " *
                          "the model's sections are those of the Configuration page; convert the file"; file = file))
    return
end

"A message for a key schema 2 removed, or `nothing`."
function _removed_key(k::AbstractString)
    k == "run.driver" && return "`[run] driver` was removed in schema_version 2 (ONE-b, #451): a config " *
                                "runs the one model and needs no key; a case not yet ported to components " *
                                "is `[run] legacy = \"<case>\"`"
    return nothing
end

# the problems of one section's table `tbl` against its spec `s`: typed values
# into `vals`, origins into `vorig`; `prefix` is the key prefix of messages
# ("" or "domain[i].") and `korig(k)` the origin of key "section.k"
function _check_section!(probs, vals, vorig, s::Section, tbl, korig, prefix, leaf;
                         tables = nothing, rank = k -> (korig(k) in (OVERRIDE_ORIGIN, WITH_ORIGIN) ? 2 : 0),
                         lower = k -> nothing)
    s.selector === nothing ||
        return _check_slot_section!(probs, vals, vorig, s, tbl, korig, prefix, leaf, tables, rank, lower)
    keys_ = [o.key for o in s.options]
    for k in sort!(collect(keys(tbl)))
        k in keys_ && continue
        full = "$(s.name).$k"
        why = _removed_key(full)
        push!(probs, ConfigProblem(prefix * full, korig(full),
              why === nothing ? "unknown key in [$(s.name)]" * _suggest(k, keys_) : why))
    end
    return _check_options!(probs, vals, vorig, s.name, s.options, tbl, korig, prefix, leaf)
end

# A slot section (ONE-m, #476): the selector names a provider of the
# domain's kind; the other keys are options of that provider (or shared
# options of the section). A selector set above the section's other keys
# (`rank`: 0 the file's run-level section, 1 a [[domain]] block, 2 an
# override) that names another provider than the level below it (`lower`:
# that level's raw value, `nothing` for the default) replaces the
# provider: the options set below it are dropped.
function _check_slot_section!(probs, vals, vorig, s::Section, tbl, korig, prefix, leaf, tables, rank, lower)
    sel = s.selector
    kind = domain_kind(s, tables === nothing ? Dict{String,Any}() : tables)
    provs = providers(s, kind)
    if isempty(provs)
        on = unique(reduce(vcat, (p.kinds for p in sel.providers)))
        for k in sort!(collect(keys(tbl)))
            full = "$(s.name).$k"
            push!(probs, ConfigProblem(prefix * full, korig(full),
                  "[$(s.name)] has no provider on $(kind_text(s, kind)); it configures " *
                  join((kind_text(s, x) for x in on), " or ")))
        end
        return vals
    end
    skey = "$(s.name).$(sel.key)"
    names = [p.name for p in provs]
    pname = sel.defaults[kind]
    if haskey(tbl, sel.key)
        raw = tbl[sel.key]; org = korig(skey)
        vorig[prefix * skey] = org
        name = raw isa Union{AbstractString,Symbol} ? String(raw) : nothing
        if name === nothing || !(name in names)
            other = name === nothing ? nothing :
                    findfirst(p -> p.name == name, sel.providers)
            push!(probs, ConfigProblem(prefix * skey, org,
                  "expected one of " * join(repr.(names), ", ") * " on $(kind_text(s, kind)), got $(_show_raw(raw))" *
                  (other === nothing ? (name === nothing ? "" : _suggest(name, names)) :
                   " (a provider on " * join((kind_text(s, x) for x in sel.providers[other].kinds), " and ") * " only)")))
            return vals
        end
        pname = name
    else
        vorig[prefix * skey] = DEFAULT_ORIGIN
    end
    vals[sel.key] = Symbol(pname)
    p = find_provider(s, kind, pname)
    opts = vcat(s.options, p.options)
    keys_ = [o.key for o in opts]
    srank = rank(skey)
    low = srank == 0 ? pname : lower(skey)
    replaced = srank > 0 && String(something(low, sel.defaults[kind])) != pname
    tbl = replaced ? Dict{String,Any}(k => v for (k, v) in tbl if k == sel.key || rank("$(s.name).$k") >= srank) : tbl
    for k in sort!(collect(keys(tbl)))
        (k == sel.key || k in keys_) && continue
        full = "$(s.name).$k"
        owners = unique([q.name for q in sel.providers if any(o -> o.key == k, q.options)])
        push!(probs, ConfigProblem(prefix * full, korig(full),
              isempty(owners) ?
                  "unknown key in [$(s.name)] for $(sel.key) = \"$pname\"" * _suggest(k, keys_) *
                  (isempty(keys_) ? " (it has no options)" : " (its options: " * join(keys_, ", ") * ")") :
                  "an option of $(sel.key) = " * join(repr.(owners), " or ") * ", not of $(sel.key) = \"$pname\"" *
                  (isempty(keys_) ? " (it has no options)" : " (its options: " * join(keys_, ", ") * ")")))
    end
    return _check_options!(probs, vals, vorig, s.name, opts, tbl, korig, prefix, leaf)
end

function _check_options!(probs, vals, vorig, sname, options, tbl, korig, prefix, leaf)
    s = (; name = sname)
    for o in options
        key = "$(s.name).$(o.key)"
        if haskey(tbl, o.key)
            org = korig(key)
            ok, v, why = _coerce(o, tbl[o.key], false, _base_dir(org), key)
            ok ? (vals[o.key] = v) : push!(probs, ConfigProblem(prefix * key, org, why))
            vorig[prefix * key] = org
        elseif o.required
            push!(probs, ConfigProblem(prefix * key, leaf, "required, expected $(_describe(o.type))$(_constraint_text(o))"))
        else
            ok, v, why = _default(o, key)
            ok ? (vals[o.key] = v) : push!(probs, ConfigProblem(prefix * key, DEFAULT_ORIGIN, why))
            vorig[prefix * key] = DEFAULT_ORIGIN
        end
    end
    return vals
end

"""
Validate the raw table against the schema, resolve defaults, paths and the
domains, run the cross-field checks; throw one ConfigError with every
problem.
"""
function _resolve(raw, origins, sources, schema, probs::Vector{ConfigProblem}, leaf)
    _check_version(raw, origins, leaf)
    sch = _pick_schema(raw, schema, leaf)
    dsec = sch.domain_sections
    blocks, base = isempty(dsec) ? (nothing, nothing) : _domain_input!(raw, origins, sch, probs, leaf)
    names = [s.name for s in sch.sections]          # "domain": the [[domain]] blocks
    for k in sort!(collect(keys(raw)))
        k in names && continue
        push!(probs, ConfigProblem(k, _origin(origins, k, leaf),
              "unknown " * (raw[k] isa AbstractDict ? "table [$k]" : "key") *
              (sch.name == MODEL ? "" : " for the legacy case \"$(sch.name)\"") * _suggest(k, names)))
    end
    values = Dict{String,Dict{String,Any}}()
    vorig = Dict{String,String}()
    korig(key) = _origin(origins, key, leaf)
    for s in sch.sections
        is_domain_section(sch, s.name) && continue
        tbl = get(raw, s.name, Dict{String,Any}())
        if !(tbl isa AbstractDict)
            push!(probs, ConfigProblem(s.name, _origin(origins, s.name, leaf),
                                       "expected a table [$(s.name)], got $(_show_raw(tbl))"))
            tbl = Dict{String,Any}()
        end
        # a run-level slot section ([timestep]) tells the domain kind from the
        # run-level tables and, with one domain, its block's
        kt = s.selector === nothing ? raw : (length(blocks === nothing ? () : blocks) == 1 ? merge(raw, blocks[1]) : raw)
        values[s.name] = _check_section!(probs, Dict{String,Any}(), vorig, s, tbl, korig, "", leaf; tables = kt,
                                         lower = k -> get(origins, _WAS * k, nothing))
    end
    domains = Vector{Dict{String,Dict{String,Any}}}()
    if !isempty(dsec)
        for (i, blk) in enumerate(blocks)
            push!(domains, _resolve_domain!(probs, vorig, sch, raw, base, origins, blk, i, length(blocks), leaf))
        end
        ids = [get(d["domain"], "id", "") for d in domains]
        for id in unique(ids)
            count(==(id), ids) > 1 &&
                push!(probs, ConfigProblem("domain", _origin(origins, "domain", leaf),
                                           "the domain id \"$id\" is given $(count(==(id), ids)) times; ids are unique"))
        end
    end
    isempty(probs) || throw(ConfigError(_sorted(unique(probs))))
    return _finish(sch, values, domains, vorig, sources)
end

# Pop the domain input of `raw`: the `[[domain]]` blocks (one empty block, the
# domain "main", when there are none) and the run-level `domain` section (the
# top-level keys static_driver, dynamic_driver).
function _domain_input!(raw, origins, sch, probs, leaf)
    dspec = _section(sch, "domain")
    base = Dict{String,Any}()
    if dspec !== nothing
        for o in dspec.options
            o.key == "id" && continue
            haskey(raw, o.key) || continue
            v = pop!(raw, o.key)
            base[o.key] = v
            origins["domain.$(o.key)"] = _origin(origins, o.key, leaf)
        end
    end
    blk = pop!(raw, "domain", nothing)
    if blk === nothing
        return [Dict{String,Any}()], base
    end
    if !(blk isa AbstractVector && !isempty(blk) && all(x -> x isa AbstractDict, blk))
        push!(probs, ConfigProblem("domain", _origin(origins, "domain", leaf),
              "expected [[domain]] blocks (an array of tables), got $(_show_raw(blk))"))
        return [Dict{String,Any}()], base
    end
    return [Dict{String,Any}(b) for b in blk], base
end

# One domain: the run-level values of the per-domain sections, replaced key
# by key by the block's (an override, "--set", is not replaced), validated.
function _resolve_domain!(probs, vorig, sch, raw, base, origins, blk, i, n, leaf)
    prefix = "domain[$i]."
    borg = _origin(origins, "domain", leaf)
    dorig = Dict{String,String}()
    merged = Dict{String,Any}()
    for name in sch.domain_sections
        t = name == "domain" ? base : get(raw, name, Dict{String,Any}())
        if !(t isa AbstractDict)
            i == 1 && push!(probs, ConfigProblem(name, _origin(origins, name, leaf),
                                                 "expected a table [$name], got $(_show_raw(t))"))
            t = Dict{String,Any}()
        end
        merged[name] = Dict{String,Any}(t)
        for k in keys(t)
            dorig["$name.$k"] = _origin(origins, "$name.$k", leaf)
        end
    end
    dkeys = (d = _section(sch, "domain"); d === nothing ? String[] : [o.key for o in d.options])
    secs = [s for s in sch.domain_sections if s != "domain"]
    fromblock = Set{String}()
    for k in sort!(collect(keys(blk)))
        v = blk[k]
        if k in secs
            if !(v isa AbstractDict)
                push!(probs, ConfigProblem(prefix * k, borg, "expected a table [domain.$k], got $(_show_raw(v))"))
                continue
            end
            for (kk, vv) in v
                get(dorig, "$k.$kk", "") == OVERRIDE_ORIGIN && continue       # --set wins over the block
                merged[k][kk] = vv; dorig["$k.$kk"] = borg; push!(fromblock, "$k.$kk")
            end
        elseif k in dkeys
            get(dorig, "domain.$k", "") == OVERRIDE_ORIGIN && continue
            merged["domain"][k] = v; dorig["domain.$k"] = borg; push!(fromblock, "domain.$k")
        else
            push!(probs, ConfigProblem(prefix * k, borg,
                  "unknown " * (v isa AbstractDict ? "table [domain.$k]" : "key") * " in a [[domain]] block" *
                  _suggest(k, vcat(dkeys, secs))))
        end
    end
    vals = Dict{String,Dict{String,Any}}()
    korig(key) = get(dorig, key, leaf)
    for name in sch.domain_sections
        s = _section(sch, name)
        p = ConfigProblem[]
        v = Dict{String,Any}()
        # the level below a selector: the file's for an override (the value it
        # replaced), the run level's for a [[domain]] block
        runlevel = get(raw, name, Dict{String,Any}())
        lowerval(k) = korig(k) == OVERRIDE_ORIGIN ? get(origins, _WAS * k, nothing) :
                      (runlevel isa AbstractDict ? get(runlevel, split(k, '.'; limit = 2)[2], nothing) : nothing)
        _check_section!(p, v, vorig, s, merged[name], korig, "domain[$i].", leaf; tables = merged,
                        rank = k -> korig(k) == OVERRIDE_ORIGIN ? 2 : k in fromblock ? 1 : 0, lower = lowerval)
        # a problem of a run-level value is reported under its own key (once:
        # the caller drops duplicates), one of the block's under domain[i]
        for q in p
            k = q.key[length("domain[$i].")+1:end]
            push!(probs, k in fromblock ? q : ConfigProblem(k, q.file, q.message))
        end
        vals[name] = v
    end
    return vals
end
"The default of `o`, evaluated (a function is called) and checked."
function _default(o::Option, key)
    d = o.default isa Function ? o.default() : o.default
    ok, v, why = _coerce(o, d, true, pwd(), key)
    ok || return (false, nothing, "the default $(_show_raw(d)) is invalid: $why")
    return (true, v, "")
end

_sorted(probs) = sort(probs; by = p -> (p.key, p.file, p.message))

"""
Build the RunConfig and run the cross-field checks, on each domain's view of
it ([`domain_view`](@ref)) when there are several domains; a problem of a
per-domain section is then reported under `domain[i].`. A
[`ModelCheck`](@ref) runs once, on the whole config.
"""
function _finish(sch::ConfigSchema, values, domains, vorig, sources)
    secs(v) = Dict{String,ConfigSection}(n => ConfigSection(n, x) for (n, x) in v)
    cfg = RunConfig(sch, secs(values), DomainConfig[DomainConfig(secs(d)) for d in domains],
                    vorig, sources, _nonreplayable(values, domains))
    probs = ConfigProblem[]
    n = length(domains)
    for i in (n <= 1 ? (0:0) : (1:n))
        v = i == 0 ? cfg : domain_view(cfg, i)
        for (_, f) in sch.checks
            f isa ModelCheck && continue
            for e in Base.invokelatest(f, v)     # an example case may have been loaded just now
                for p in (e isa ConfigError ? e.problems : [ConfigProblem("", "", string(e))])
                    sec = first(split(p.key, '.'))
                    key = i > 0 && is_domain_section(sch, sec) ? "domain[$i]." * p.key : p.key
                    file = isempty(p.file) ? get(vorig, key, get(vorig, "domain[$(max(i, 1))]." * p.key, "")) : p.file
                    push!(probs, ConfigProblem(key, file, p.message))
                end
            end
        end
    end
    # the checks of the whole model (the [[domain]] graph), once
    for (_, f) in sch.checks
        f isa ModelCheck || continue
        for e in Base.invokelatest(f, cfg)
            for p in (e isa ConfigError ? e.problems : [ConfigProblem("", "", string(e))])
                push!(probs, ConfigProblem(p.key, isempty(p.file) ? get(vorig, p.key, "") : p.file, p.message))
            end
        end
    end
    isempty(probs) || throw(ConfigError(_sorted(unique(probs))))
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
