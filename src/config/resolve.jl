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
    RunConfig

A fully resolved, validated, typed run configuration, made by
[`load_run_config`](@ref) or [`with`](@ref). Field access goes through
the sections: `cfg.grid.size`, `cfg.run.seed`; `cfg["grid.size"]` takes a
dotted key. It is not modified in place: [`with`](@ref) returns a new
config. Two configs are `==` when their canonical texts
([`write_config`](@ref)) are identical.

Internals (use the accessors): the schema ([`config_schema`](@ref)), the
file each value came from ([`config_origin`](@ref)), the sources
(extends chain and overrides, [`config_sources`](@ref)) and the keys
holding values TOML cannot hold ([`nonreplayable_keys`](@ref)).
"""
struct RunConfig
    schema::ConfigSchema
    sections::Dict{String,ConfigSection}
    origins::Dict{String,String}
    sources::Dict{String,Any}
    nonreplayable::Vector{String}
end

function Base.getproperty(cfg::RunConfig, k::Symbol)
    secs = getfield(cfg, :sections)
    key = String(k)
    haskey(secs, key) || throw(ArgumentError("the configuration has no section [$key]" *
                                             _suggest(key, keys(secs))))
    return secs[key]
end
Base.propertynames(cfg::RunConfig) = Tuple(Symbol.(sort!(collect(keys(getfield(cfg, :sections))))))

function Base.getindex(cfg::RunConfig, key::AbstractString)
    i = findfirst('.', key)
    i === nothing && return getproperty(cfg, Symbol(key))
    return getproperty(getproperty(cfg, Symbol(key[1:prevind(key, i)])), Symbol(key[nextind(key, i):end]))
end

"The [`ConfigSchema`](@ref) a config was validated against."
config_schema(cfg::RunConfig) = getfield(cfg, :schema)
"""
    config_origin(cfg, key) -> String

The file the value of the dotted `key` came from: a path, `"default"`,
`"--set"` (an override) or `"with()"`.
"""
config_origin(cfg::RunConfig, key::AbstractString) = get(getfield(cfg, :origins), key, "")
"""
    config_sources(cfg) -> Dict{String,Any}

How a config was made: `files` (the extends chain, the leaf first, or
the `run.toml` it was read from), `overrides` and, after [`with`](@ref),
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
    print(io, "RunConfig(driver \"", config_schema(cfg).name, "\", ",
          length(getfield(cfg, :sections)), " sections, hash ", config_hash_string(cfg), ")")
end
function Base.show(io::Base.IO, ::MIME"text/plain", cfg::RunConfig)
    show(io, cfg); println(io); print(io, config_text(cfg))
end

function _nonreplayable(values)
    out = String[]
    for (s, vals) in values, (k, v) in vals
        _toml_native(v) || push!(out, "$s.$k")
    end
    return sort!(out)
end

# ---------------------------------------------------------------------------
# with: the Julia escape hatch
# ---------------------------------------------------------------------------

"""
    with(cfg::RunConfig, "section.key" => value, …) -> RunConfig

A new, validated config with the given options replaced (Julia values:
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
    vorig = copy(getfield(cfg, :origins))
    probs = ConfigProblem[]
    set = String[]
    for (k, v) in kvs
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
        o = _option(s, okey)
        if o === nothing
            push!(probs, ConfigProblem(key, WITH_ORIGIN, "unknown key in [$sname]" *
                                       _suggest(okey, [x.key for x in s.options])))
            continue
        end
        ok, val, why = _coerce(o, v, true, pwd(), key)
        if ok
            values[sname][okey] = val; vorig[key] = WITH_ORIGIN; push!(set, key)
        else
            push!(probs, ConfigProblem(key, WITH_ORIGIN, why))
        end
    end
    isempty(probs) || throw(ConfigError(_sorted(probs)))
    src = deepcopy(config_sources(cfg))
    src["with"] = unique!(vcat(String[get(src, "with", String[])...], set))
    return _finish(sch, values, vorig, src)
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

The canonical resolved form as a TOML-ready table: `schema_version = 1`
and every section with **every** option, defaults included; symbols as
strings, `nothing` as `"none"`, tuples as arrays, paths absolute, no
`extends`. A value TOML cannot hold is written as `"<julia: …>"`.
"""
function to_toml(cfg::RunConfig)
    d = Dict{String,Any}("schema_version" => CONFIG_SCHEMA_VERSION)
    for (n, s) in getfield(cfg, :sections)
        d[n] = Dict{String,Any}(k => _toml_out(v) for (k, v) in getfield(s, :values))
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

function _emit_table(io, d::AbstractDict, path::Vector{String})
    ks = sort!(collect(String, keys(d)))
    scal = [k for k in ks if !(d[k] isa AbstractDict)]
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
`key[i]`).
"""
function config_inputs(cfg::RunConfig)
    out = Dict{String,Any}()
    for s in config_schema(cfg).sections, o in s.options
        (o.input && _is_path_type(o.type)) || continue
        v = getfield(getfield(cfg, :sections)[s.name], :values)[o.key]
        key = "$(s.name).$(o.key)"
        if v isa AbstractString
            out[key] = Dict{String,Any}("path" => v, "fnv1a64" => content_hash(v))
        elseif v isa Union{Tuple,AbstractVector}
            for (i, p) in enumerate(v)
                p isa AbstractString &&
                    (out["$key[$i]"] = Dict{String,Any}("path" => p, "fnv1a64" => content_hash(p)))
            end
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
