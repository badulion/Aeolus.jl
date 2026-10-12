# The resolved configuration (CFG-a, #356): RunConfig, its canonical TOML
# form and hash, the Julia escape hatch `with`, and input-file hashes.

# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

"""
    ConfigSection

One resolved section of a [`RunConfig`](@ref): `sec.key` is the typed
value of option `key` (`Symbol`s, tuples, absolute paths, `nothing`, …).
"""
struct ConfigSection
    name::String
    values::Dict{String,Any}
end

function Base.getproperty(s::ConfigSection, k::Symbol)
    v = getfield(s, :values)
    key = String(k)
    haskey(v, key) || throw(ArgumentError("[$(getfield(s, :name))] has no option \"$key\"" *
                                          _suggest(key, keys(v))))
    return v[key]
end
Base.propertynames(s::ConfigSection) = Tuple(Symbol.(sort!(collect(keys(getfield(s, :values))))))
Base.getindex(s::ConfigSection, k::AbstractString) = getproperty(s, Symbol(k))
Base.haskey(s::ConfigSection, k::AbstractString) = haskey(getfield(s, :values), k)
Base.show(io::Base.IO, s::ConfigSection) =
    print(io, "ConfigSection [", getfield(s, :name), "] (", length(getfield(s, :values)), " options)")

"""
    DomainConfig

The resolved per-domain sections of one domain of a [`RunConfig`](@ref)
(ONE-b, #451): `d.grid`, `d.sgs`, … and `d.domain` (its `id`,
`static_driver`, `dynamic_driver`); [`config_domains`](@ref) lists them.
"""
struct DomainConfig
    sections::Dict{String,ConfigSection}
end

function Base.getproperty(d::DomainConfig, k::Symbol)
    secs = getfield(d, :sections)
    key = String(k)
    haskey(secs, key) || throw(ArgumentError("the domain has no section [$key]" * _suggest(key, keys(secs))))
    return secs[key]
end
Base.propertynames(d::DomainConfig) = Tuple(Symbol.(sort!(collect(keys(getfield(d, :sections))))))
Base.show(io::Base.IO, d::DomainConfig) =
    print(io, "DomainConfig \"", d.domain.id, "\" (", length(getfield(d, :sections)), " sections)")

"""
    RunConfig

A fully resolved, validated, typed run configuration, made by
[`load_run_config`](@ref) or [`with`](@ref). Field access goes through
the sections: `cfg.time.dt`, `cfg.run.seed`; `cfg["grid.size"]` takes a
dotted key. The model's per-domain sections are resolved for every domain
([`config_domains`](@ref)); with one domain, `cfg.grid` is that domain's
section. It is not modified in place: [`with`](@ref) returns a new
config. Two configs are `==` when their canonical texts
([`write_config`](@ref)) are identical.

Internals (use the accessors): the schema ([`config_schema`](@ref)), the
file each value came from ([`config_origin`](@ref)), the sources
(the file and the overrides, [`config_sources`](@ref)) and the keys
holding values TOML cannot hold ([`nonreplayable_keys`](@ref)).
"""
struct RunConfig
    schema::ConfigSchema
    sections::Dict{String,ConfigSection}
    domains::Vector{DomainConfig}
    origins::Dict{String,String}
    sources::Dict{String,Any}
    nonreplayable::Vector{String}
end

function Base.getproperty(cfg::RunConfig, k::Symbol)
    secs = getfield(cfg, :sections)
    key = String(k)
    haskey(secs, key) && return secs[key]
    sch = getfield(cfg, :schema)
    if is_domain_section(sch, key)
        ds = getfield(cfg, :domains)
        length(ds) == 1 && return getfield(ds[1], :sections)[key]
        throw(ArgumentError("[$key] is per domain and the configuration has $(length(ds)) domains: " *
                            "use config_domains(cfg)[i].$key"))
    end
    throw(ArgumentError("the configuration has no section [$key]" *
                        _suggest(key, vcat(collect(keys(secs)), sch.domain_sections))))
end
Base.propertynames(cfg::RunConfig) =
    Tuple(Symbol.(sort!(vcat(collect(keys(getfield(cfg, :sections))), getfield(cfg, :schema).domain_sections))))

function Base.getindex(cfg::RunConfig, key::AbstractString)
    i = findfirst('.', key)
    i === nothing && return getproperty(cfg, Symbol(key))
    return getproperty(getproperty(cfg, Symbol(key[1:prevind(key, i)])), Symbol(key[nextind(key, i):end]))
end

"""
    config_domains(cfg) -> Vector{DomainConfig}

The resolved domains of the model's config `cfg`, in the order of its
`[[domain]]` blocks (one, `"main"`, without them); empty for a legacy case.
"""
config_domains(cfg::RunConfig) = copy(getfield(cfg, :domains))

"""
    domain_view(cfg, i) -> RunConfig

`cfg` with only its `i`-th domain: a single-domain config whose per-domain
sections (`cfg.grid`, …) are those of domain `i` (what a domain is built
from, and what the cross-field checks see).
"""
function domain_view(cfg::RunConfig, i::Integer)
    ds = getfield(cfg, :domains)
    1 <= i <= length(ds) || throw(BoundsError(ds, i))
    org = Dict{String,String}(k => v for (k, v) in getfield(cfg, :origins) if !startswith(k, "domain["))
    for (k, v) in getfield(cfg, :origins)
        startswith(k, "domain[$i].") && (org["domain[1]." * k[length("domain[$i].")+1:end]] = v)
    end
    return RunConfig(getfield(cfg, :schema), getfield(cfg, :sections), DomainConfig[ds[i]], org,
                     getfield(cfg, :sources), getfield(cfg, :nonreplayable))
end

"The [`ConfigSchema`](@ref) a config was validated against."
config_schema(cfg::RunConfig) = getfield(cfg, :schema)
"""
    config_origin(cfg, key) -> String

The file the value of the dotted `key` came from: a path, `"default"`,
`"--set"` (an override) or `"with()"`.
"""
function config_origin(cfg::RunConfig, key::AbstractString)
    o = getfield(cfg, :origins)
    haskey(o, key) && return o[key]
    return get(o, "domain[1]." * key, "")
end
"""
    config_sources(cfg) -> Dict{String,Any}

How a config was made: `files` (the config file, or the `run.toml` it
was read from), `overrides` and, after [`with`](@ref),
`with` (the keys it set).
"""
config_sources(cfg::RunConfig) = getfield(cfg, :sources)
"""
    nonreplayable_keys(cfg) -> Vector{String}

The dotted keys holding a value TOML cannot hold (a closure set with
[`with`](@ref)); a run with any is recorded `replayable = false`.
"""
nonreplayable_keys(cfg::RunConfig) = copy(getfield(cfg, :nonreplayable))

Base.:(==)(a::RunConfig, b::RunConfig) = config_text(a) == config_text(b)
Base.isequal(a::RunConfig, b::RunConfig) = a == b
Base.hash(cfg::RunConfig, h::UInt) = hash(config_text(cfg), hash(:RunConfig, h))

function Base.show(io::Base.IO, cfg::RunConfig)
    n = config_schema(cfg).name
    print(io, "RunConfig(", n == MODEL ? "the model" : "legacy \"$n\"", ", ",
          length(getfield(cfg, :sections)), " sections, hash ", config_hash_string(cfg), ")")
end
function Base.show(io::Base.IO, ::MIME"text/plain", cfg::RunConfig)
    show(io, cfg); println(io); print(io, config_text(cfg))
end

function _nonreplayable(values, domains = ())
    out = String[]
    for (s, vals) in values, (k, v) in vals
        _toml_native(v) || push!(out, "$s.$k")
    end
    for (i, d) in enumerate(domains), (s, vals) in d, (k, v) in vals
        _toml_native(v) || push!(out, "domain[$i].$s.$k")
    end
    return sort!(out)
end

# ---------------------------------------------------------------------------
# with: the Julia escape hatch
# ---------------------------------------------------------------------------

"""
    with(cfg::RunConfig, "section.key" => value, …) -> RunConfig

A new, validated config with the given options replaced (an option of a
per-domain section in every domain; Julia values:
`Symbol`s, tuples, `nothing` for an `Optional`, …; a relative `Path` is
resolved against the working directory). Unknown keys and invalid values
throw a [`ConfigError`](@ref) listing every problem; the cross-field
checks run again. An option declared `allow_julia = true` also takes a
value TOML cannot hold (a closure); the config is then not replayable
([`nonreplayable_keys`](@ref)), and `run.toml` records it so.
"""
function with(cfg::RunConfig, kvs::Pair...)
    sch = config_schema(cfg)
    values = Dict{String,Dict{String,Any}}(n => copy(getfield(s, :values))
                                          for (n, s) in getfield(cfg, :sections))
    domains = [Dict{String,Dict{String,Any}}(n => copy(getfield(s, :values)) for (n, s) in getfield(d, :sections))
               for d in getfield(cfg, :domains)]
    vorig = copy(getfield(cfg, :origins))
    probs = ConfigProblem[]
    set = String[]
    # the selectors of slot sections first (ONE-m, #476): a new provider
    # takes the options it shares with the old one, the rest at its defaults
    rest = Pair[]
    for (k, v) in kvs
        key = String(k)
        i = findfirst('.', key)
        s = i === nothing ? nothing : _section(sch, key[1:prevind(key, i)])
        if s === nothing || s.selector === nothing || key[nextind(key, i):end] != s.selector.key
            push!(rest, k => v)
            continue
        end
        targets = is_domain_section(sch, s.name) ? [(d, "domain[$j].") for (j, d) in enumerate(domains)] :
                                                   [(values, "")]
        for (tabs, prefix) in targets
            why = _with_provider!(tabs, s, v, _kind_tables(tabs, domains))
            why === nothing ? (vorig[prefix * key] = WITH_ORIGIN) : push!(probs, ConfigProblem(key, WITH_ORIGIN, why))
        end
        push!(set, key)
    end
    for (k, v) in rest
        key = String(k)
        i = findfirst('.', key)
        sname = i === nothing ? key : key[1:prevind(key, i)]
        okey = i === nothing ? "" : key[nextind(key, i):end]
        s = _section(sch, sname)
        if s === nothing
            push!(probs, ConfigProblem(key, WITH_ORIGIN, "unknown section [$sname]" *
                                       _suggest(sname, [x.name for x in sch.sections])))
            continue
        end
        if s.selector !== nothing                       # a slot section: the chosen provider's option
            targets = is_domain_section(sch, sname) ? [(d, "domain[$j].") for (j, d) in enumerate(domains)] :
                                                      [(values, "")]
            for (tabs, prefix) in targets
                o = _slot_option(s, tabs, okey, _kind_tables(tabs, domains))
                if o === nothing
                    push!(probs, ConfigProblem(key, WITH_ORIGIN, "not an option of [$sname] $(s.selector.key) = " *
                                               "\"$(get(tabs[sname], s.selector.key, "?"))\""))
                    continue
                end
                ok, val, why = _coerce(o, v, true, pwd(), key)
                ok ? (tabs[sname][okey] = val; vorig[prefix * key] = WITH_ORIGIN) :
                     push!(probs, ConfigProblem(key, WITH_ORIGIN, why))
            end
            push!(set, key)
            continue
        end
        o = _option(s, okey)
        if o === nothing
            push!(probs, ConfigProblem(key, WITH_ORIGIN, "unknown key in [$sname]" *
                                       _suggest(okey, [x.key for x in s.options])))
            continue
        end
        ok, val, why = _coerce(o, v, true, pwd(), key)
        if ok && is_domain_section(sch, sname)
            for (i, d) in enumerate(domains)
                d[sname][okey] = val; vorig["domain[$i].$key"] = WITH_ORIGIN
            end
            push!(set, key)
        elseif ok
            values[sname][okey] = val; vorig[key] = WITH_ORIGIN; push!(set, key)
        else
            push!(probs, ConfigProblem(key, WITH_ORIGIN, why))
        end
    end
    isempty(probs) || throw(ConfigError(_sorted(probs)))
    src = deepcopy(config_sources(cfg))
    src["with"] = unique!(vcat(String[get(src, "with", String[])...], set))
    return _finish(sch, values, domains, vorig, src)
end

# The tables that tell the kind of a domain (`with`): one domain's, or for
# a run-level slot section the run-level ones with those of a single domain
_kind_tables(tabs, domains) = any(d -> d === tabs, domains) || length(domains) != 1 ? tabs : merge(domains[1], tabs)

# The option `okey` of the slot section `s` under the provider the tables
# `tabs` (one domain's, or the run-level ones) choose; `nothing` if it has none.
function _slot_option(s::Section, tabs, okey, kt = tabs)
    o = _option(s, okey)
    o === nothing || return o
    v = get(tabs[s.name], s.selector.key, nothing)
    v === nothing && return nothing
    p = find_provider(s, domain_kind(s, kt), String(v))
    p === nothing && return nothing
    i = findfirst(x -> x.key == okey, p.options)
    return i === nothing ? nothing : p.options[i]
end

# Switch the slot section `s` of the tables `tabs` to the provider `v`
# (`with`): the options the new provider shares with the old one keep
# their values, the others take its defaults. A problem as a message.
function _with_provider!(tabs, s::Section, v, kt = tabs)
    sel = s.selector
    kind = domain_kind(s, kt)
    name = v isa Union{AbstractString,Symbol} ? String(v) : nothing
    p = name === nothing ? nothing : find_provider(s, kind, name)
    p === nothing && return "expected one of " * join(repr.([q.name for q in providers(s, kind)]), ", ") *
                            " on $(kind_text(s, kind)), got $(_show_raw(v))"
    old = tabs[s.name]
    new = Dict{String,Any}(sel.key => Symbol(name))
    for o in vcat(s.options, p.options)
        key = "$(s.name).$(o.key)"
        ok, val, _ = haskey(old, o.key) ? _coerce(o, old[o.key], true, pwd(), key) : (false, nothing, "")
        ok || ((ok, val, why) = _default(o, key))
        ok || return why
        new[o.key] = val
    end
    tabs[s.name] = new
    return nothing
end

# ---------------------------------------------------------------------------
# the canonical resolved form
# ---------------------------------------------------------------------------

_toml_out(v::Symbol) = String(v)
_toml_out(::Nothing) = "none"
_toml_out(v::Tuple) = Any[_toml_out(x) for x in v]
_toml_out(v::AbstractVector) = Any[_toml_out(x) for x in v]
_toml_out(v::AbstractDict) = Dict{String,Any}(string(k) => _toml_out(x) for (k, x) in v)
_toml_out(v::NamedTuple) = Dict{String,Any}(string(k) => _toml_out(x) for (k, x) in pairs(v))
_toml_out(v::Union{Bool,Int64,Float64,String,Date,Time,DateTime}) = v
_toml_out(v::Integer) = Int64(v)
_toml_out(v::AbstractFloat) = Float64(v)
_toml_out(v::AbstractString) = String(v)
function _toml_out(v)                        # a closure or other Julia object
    s = repr(v; context = :compact => true)
    return "<julia: " * (length(s) > 200 ? first(s, 197) * "..." : s) * ">"
end

"""
    to_toml(cfg::RunConfig) -> Dict{String,Any}

The canonical resolved form as a TOML-ready table: `schema_version = 2`
and every section with **every** option, defaults included; symbols as
strings, `nothing` as `"none"`, tuples as arrays, paths absolute. The
model's per-domain sections are written for **every domain**, under
`[[domain]]` (its `id`, `static_driver`, `dynamic_driver`, then one
`[domain.<section>]` table per section): the resolved config lists every
component's options per domain. A value TOML cannot hold is written as
`"<julia: …>"`.
"""
function to_toml(cfg::RunConfig)
    d = Dict{String,Any}("schema_version" => CONFIG_SCHEMA_VERSION)
    sec(s) = Dict{String,Any}(k => _toml_out(v) for (k, v) in getfield(s, :values))
    for (n, s) in getfield(cfg, :sections)
        d[n] = sec(s)
    end
    ds = getfield(cfg, :domains)
    if !isempty(ds)
        d["domain"] = Any[]
        for dc in ds
            t = Dict{String,Any}()
            for (n, s) in getfield(dc, :sections)
                n == "domain" ? merge!(t, sec(s)) : (t[n] = sec(s))
            end
            push!(d["domain"], t)
        end
    end
    return d
end

"""
    config_text(cfg::RunConfig) -> String

The canonical TOML text of `cfg` ([`write_config`](@ref)).
"""
function config_text(cfg::RunConfig)
    io = IOBuffer()
    _emit_table(io, to_toml(cfg), String[])
    return String(take!(io))
end

"""
    write_config(io::Base.IO, cfg) / write_config(path, cfg) / write_config(cfg) -> String

Write the canonical resolved form of `cfg` ([`to_toml`](@ref)): keys
sorted, top-level scalars first, then one `[table]` per section (and
`[section.sub]` for nested tables), floats in Julia's shortest
round-trip representation (`inf`, `-inf`, `nan` for the specials).
Loading the text back gives the same config, and writing that again
gives the same bytes. `write_config(cfg)` returns the text.
"""
write_config(io::Base.IO, cfg::RunConfig) = (write(io, config_text(cfg)); nothing)
write_config(path::AbstractString, cfg::RunConfig) = (write(path, config_text(cfg)); path)
write_config(cfg::RunConfig) = config_text(cfg)

"""
    config_hash(cfg::RunConfig) -> UInt64

The [`fnv1a64`](@ref Aeolus.IO.fnv1a64) hash of the canonical text
([`config_text`](@ref)). `run.toml` records it as 16 hex digits.
"""
config_hash(cfg::RunConfig) = fnv1a64(config_text(cfg))
config_hash_string(cfg::RunConfig) = string(config_hash(cfg); base = 16, pad = 16)

const _BARE_KEY = r"^[A-Za-z0-9_-]+$"
_emit_key(k::AbstractString) = occursin(_BARE_KEY, k) ? k : _emit_string(k)

# the [[domain]] blocks of a resolved config: an array of tables at the top
_aot(d, k, path) = isempty(path) && k == "domain" && d[k] isa AbstractVector

function _emit_table(io, d::AbstractDict, path::Vector{String})
    ks = sort!(collect(String, keys(d)))
    aots = [k for k in ks if _aot(d, k, path)]
    scal = [k for k in ks if !(d[k] isa AbstractDict) && !(k in aots)]
    subs = [k for k in ks if d[k] isa AbstractDict]
    for k in scal
        print(io, _emit_key(k), " = ", _emit_value(d[k]), "\n")
    end
    for k in subs
        p = vcat(path, k)
        !isempty(scal) || !isempty(path) || k != subs[1] ? print(io, "\n") : nothing
        print(io, "[", join(_emit_key.(p), "."), "]\n")
        _emit_table(io, d[k], p)
    end
    for k in aots, e in d[k]
        p = vcat(path, k)
        print(io, "\n[[", join(_emit_key.(p), "."), "]]\n")
        _emit_table(io, e, p)
    end
end

function _emit_value(v)
    v isa Bool && return v ? "true" : "false"
    v isa Integer && return string(v)
    v isa AbstractFloat && return _emit_float(Float64(v))
    v isa AbstractString && return _emit_string(v)
    v isa Date && return string(v)
    v isa Time && return string(v)
    v isa DateTime && return string(v)
    v isa AbstractVector && return "[" * join(map(_emit_value, v), ", ") * "]"
    v isa AbstractDict && return "{" * join([_emit_key(k) * " = " * _emit_value(v[k])
                                             for k in sort!(collect(String, keys(v)))], ", ") * "}"
    throw(ArgumentError("cannot write $(typeof(v)) as TOML"))
end

function _emit_float(x::Float64)
    isnan(x) && return signbit(x) ? "-nan" : "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    return repr(x)                     # shortest round-trip: 0.1, 1.0e-4, 5.0e-324, -0.0
end

function _emit_string(s::AbstractString)
    io = IOBuffer()
    print(io, '"')
    for c in s
        if c == '"'; print(io, "\\\"")
        elseif c == '\\'; print(io, "\\\\")
        elseif c == '\n'; print(io, "\\n")
        elseif c == '\t'; print(io, "\\t")
        elseif c == '\r'; print(io, "\\r")
        elseif c < ' ' || c == '\x7f'; print(io, "\\u", string(UInt32(c); base = 16, pad = 4))
        else print(io, c)
        end
    end
    print(io, '"')
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# input files
# ---------------------------------------------------------------------------

"""
    content_hash(path) -> String

The `fnv1a64` content hash of the file `path` as 16 hex digits; for a
directory, the hash of its sorted `relative path => file hash` list;
`"missing"` if nothing is there.
"""
function content_hash(path::AbstractString)
    isfile(path) && return string(fnv1a64(read(path)); base = 16, pad = 16)
    if isdir(path)
        lines = String[]
        for (root, _, files) in walkdir(path), f in files
            p = joinpath(root, f)
            push!(lines, relpath(p, path) * "\0" * content_hash(p))
        end
        return string(fnv1a64(join(sort!(lines), "\n")); base = 16, pad = 16)
    end
    return "missing"
end

_is_path_type(T) = T === Path || (T isa Type && ((T <: Optional && _is_path_type(_elem(T))) ||
                                                 (T <: Tuple && fieldtype(T, 1) === Path) ||
                                                 (T <: Vector && _is_path_type(eltype(T)))))

"""
    config_inputs(cfg) -> Dict{String,Any}

The `[inputs]` table: for every `Path` option declared `input = true`
and set, `key => {path, fnv1a64}` (vector and tuple elements as
`key[i]`; an option of a per-domain section of the domain `id` as
`domain.<id>.<section>.<key>`, a key of the domain itself as
`domain.<id>.<key>`).
"""
function config_inputs(cfg::RunConfig)
    out = Dict{String,Any}()
    sch = config_schema(cfg)
    for s in sch.sections, o in unique(x -> x.key, filter(x -> x.input && _is_path_type(x.type), section_options(s)))
        if is_domain_section(sch, s.name)
            for dc in getfield(cfg, :domains)
                vals = getfield(getfield(dc, :sections)[s.name], :values)
                haskey(vals, o.key) || continue          # an option of a provider not chosen
                v = vals[o.key]
                key = "domain.$(dc.domain.id)." * (s.name == "domain" ? o.key : "$(s.name).$(o.key)")
                _input!(out, key, v)
            end
        else
            vals = getfield(getfield(cfg, :sections)[s.name], :values)
            haskey(vals, o.key) && _input!(out, "$(s.name).$(o.key)", vals[o.key])
        end
    end
    return out
end

function _input!(out, key, v)
    if v isa AbstractString
        out[key] = Dict{String,Any}("path" => v, "fnv1a64" => content_hash(v))
    elseif v isa Union{Tuple,AbstractVector}
        for (i, p) in enumerate(v)
            p isa AbstractString &&
                (out["$key[$i]"] = Dict{String,Any}("path" => p, "fnv1a64" => content_hash(p)))
        end
    end
    return out
end

"Throw a ConfigError naming every recorded input whose content hash changed."
function _check_inputs(inputs::AbstractDict, rt::AbstractString)
    probs = ConfigProblem[]
    for k in sort!(collect(keys(inputs)))
        e = inputs[k]
        (e isa AbstractDict && haskey(e, "path") && haskey(e, "fnv1a64")) || continue
        now = content_hash(e["path"])
        now == e["fnv1a64"] && continue
        push!(probs, ConfigProblem(k, rt,
            "the input file $(e["path"]) changed since the run started (fnv1a64 $(e["fnv1a64"]), " *
            "now $now); restore it, or pass allow_changed_inputs = true to run anyway"))
    end
    isempty(probs) || throw(ConfigError(probs))
    return nothing
end
