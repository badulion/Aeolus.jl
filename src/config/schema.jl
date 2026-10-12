# The declarative configuration schema (CFG-a, #356).
#
# A schema is a list of sections ([run], [grid], …), each a list of typed
# options with a default, a doc string, units and constraints. Values are
# checked in two modes: from TOML (the loader) and from Julia (`with`).
# Every check returns problems instead of throwing, so the loader can
# report all of them at once in one ConfigError.

# ---------------------------------------------------------------------------
# option type markers
# ---------------------------------------------------------------------------

"""
    Path

Option type marker for a filesystem path. Written as a TOML string; a
relative path is resolved against the directory of the file that set it
(the working directory for an override or [`with`](@ref)), and the
resolved config holds it absolute. An option declared with `input = true`
(the default) is an input file: its content hash is recorded under
`[inputs]` of `run.toml`.
"""
struct Path end

"""
    Optional{T}

Option type marker for a value of type `T` or `nothing`. TOML has no null:
`nothing` is written as the string `"none"`. A missing key means "take
the default", never `nothing`.
"""
struct Optional{T} end

"""
    OpenTable(validator = tbl -> String[]; reference = nothing)

Option type for a table whose keys a driver decides (the per-coupling
options of the Berlin case). `validator(tbl::Dict{String,Any})` returns
the problems it finds: strings (attached to the option's key) or
[`ConfigError`](@ref)s whose keys are relative to the table.

`reference`, when given, is a zero-argument function returning the
documented sub-tables, a `Vector{Section}` whose names are dotted paths
below the option (`"aq_chem"`, `"nesting.parent"`); the option reference
([`option_reference`](@ref)) lists them after the option's section, and
[`undocumented_keys`](@ref) checks a config's keys against them. Their
options may be [`reference_option`](@ref)s (a type and a default given
as text).
"""
struct OpenTable
    validator::Any
    reference::Any
end
OpenTable(validator; reference = nothing) = OpenTable(validator, reference)
OpenTable() = OpenTable(_ -> String[], nothing)

"""
    OneOf(T1, T2, …)

Option type for a value of any of the option types `T1, T2, …` (scalars,
`Optional`, tuples, vectors, [`OpenTable`](@ref)): the first that accepts
the value wins. A forcing quantity that is a number or a table is
`OneOf(Float64, OpenTable(validator))`; a viscosity that is a number or
the keyword `"scaled"` is `OneOf(Float64, Symbol)` with `allowed =
[:scaled]` (`allowed` applies to the non-numeric alternatives only;
`range` to the numeric ones). An option of this type needs a `default`.
"""
struct OneOf
    alts::Tuple
    OneOf(alts...) = length(alts) >= 2 ? new(Tuple(alts)) :
                     throw(ArgumentError("OneOf needs at least two alternatives"))
end

const _SCALARS = (Float64, Int, Bool, String, Symbol, Path, Date, Time)

"Whether `T` is a supported option type (see [`option!`](@ref))."
function _valid_spec(T)
    T isa OpenTable && return true
    T isa OneOf && return all(a -> !(a isa OneOf) && _valid_spec(a), T.alts)
    T isa Type || return false
    T in _SCALARS && return true
    T <: Optional && return T !== Optional && _valid_spec(T.parameters[1])
    if T <: Tuple && isconcretetype(T)
        ps = fieldtypes(T)
        return !isempty(ps) && all(==(ps[1]), ps) && ps[1] in _SCALARS
    end
    T <: Vector && isconcretetype(T) && return _valid_spec(eltype(T)) && !(eltype(T) isa OpenTable)
    return false
end

_elem(T::Type{<:Optional}) = T.parameters[1]

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

"One problem of a configuration: the dotted `key`, the `file` it came from, the `message`."
struct ConfigProblem
    key::String
    file::String
    message::String
end

"""
    ConfigError(problems)
    ConfigError(key, message; file = "")

The error every configuration check throws. It holds **every** problem
found, each with the dotted key, the file the value came from and a
message naming the expected type, range or allowed values and the value
given. Cross-field checks ([`check!`](@ref)) return `ConfigError`s; their
problems are merged into the one error the loader throws.
"""
struct ConfigError <: Exception
    problems::Vector{ConfigProblem}
end
ConfigError(key::AbstractString, message::AbstractString; file::AbstractString = "") =
    ConfigError([ConfigProblem(String(key), String(file), String(message))])

function Base.showerror(io::Base.IO, e::ConfigError)
    n = length(e.problems)
    print(io, "ConfigError: ", n, n == 1 ? " problem" : " problems", " in the run configuration")
    for p in e.problems
        print(io, "\n  - ", isempty(p.key) ? "(top level)" : p.key)
        isempty(p.file) || print(io, " (", p.file, ")")
        print(io, ": ", p.message)
    end
end

# ---------------------------------------------------------------------------
# options, sections, schemas
# ---------------------------------------------------------------------------

"Marker for an option declared without a default."
struct NoDefault end

"""
    Option

One typed option of a [`Section`](@ref): `key`, `type` (see
[`option!`](@ref)), `default` (a value, a zero-argument function giving
it at load time, or `NoDefault()`), `doc`, `units`, `allowed`, `range`,
`open_lower`, `required`, `input` (a `Path` is an input file) and
`allow_julia` ([`with`](@ref) may set a value TOML cannot hold).
"""
struct Option
    key::String
    type::Any
    default::Any
    doc::String
    units::String
    allowed::Union{Nothing,Vector{Any}}
    range::Union{Nothing,Tuple{Real,Real}}
    open_lower::Bool
    required::Bool
    input::Bool
    allow_julia::Bool
end

"""
    reference_option(key, type_text, default_text; doc = "", units = "") -> Option

An option of a documented sub-table of an [`OpenTable`](@ref) whose type
and default are given as text (the per-coupling options of the Berlin
case, validated by the driver itself): it appears in the option
reference and nowhere else.
"""
reference_option(key::AbstractString, type_text::AbstractString, default_text::AbstractString;
                 doc::AbstractString = "", units::AbstractString = "") =
    Option(String(key), String(type_text), String(default_text), String(doc), String(units),
           nothing, nothing, false, false, false, false)

"""
    Provider

One provider of a slot section ([`slot_section`](@ref)): its `name` (the
value of the section's selector key that chooses it), `doc`, the domain
`kinds` it is a provider on, and its own `options`, built with
[`provider!`](@ref).
"""
struct Provider
    name::String
    doc::String
    kinds::Vector{Symbol}
    options::Vector{Option}
end

"""
    Selector

The selector of a slot section ([`slot_section`](@ref)): the selector
`key` (`"model"`, `"solver"`, `"scheme"`), the `providers`, the default
provider per domain kind (`defaults`), `kind(tables)`, which tells the
kind of a domain from its section tables (raw or resolved: `tables[name]`
is the table of section `name`), and `kind_text`, how a message names a
kind.
"""
struct Selector
    key::String
    providers::Vector{Provider}
    defaults::Dict{Symbol,String}
    kind::Any
    kind_text::Dict{Symbol,String}
end

"""
    Section

A named table of [`Option`](@ref)s, built with [`section`](@ref) and
[`option!`](@ref); or a **slot section** with a selector key
([`slot_section`](@ref), `selector` set): one exclusive choice among
providers, each with its own options.
"""
struct Section
    name::String
    doc::String
    options::Vector{Option}
    selector::Union{Nothing,Selector}
end
Section(name::AbstractString, doc::AbstractString, options::Vector{Option}) =
    Section(String(name), String(doc), options, nothing)

const _KEY_RE = r"^[a-z][a-z0-9_]*$"

"""
    section(f, name; doc = "") -> Section

A configuration section `[name]`, filled by `f(s)` with [`option!`](@ref):

```julia
grid = section("grid"; doc = "The mesh") do s
    option!(s, "size", NTuple{3,Int}; default = (64, 64, 32), range = (1, 4096),
            doc = "cells per direction")
    option!(s, "extent", NTuple{3,Float64}; default = (3200.0, 3200.0, 1600.0),
            units = "m", range = (0, Inf), open_lower = true, doc = "domain size")
end
```
"""
function section(f, name::AbstractString; doc::AbstractString = "")
    occursin(_KEY_RE, name) ||
        throw(ArgumentError("section name \"$name\" is not ASCII snake_case"))
    s = Section(String(name), String(doc), Option[])
    f(s)
    return s
end

"""
    slot_section(f, name; key, kind = tables -> :model, kind_text = Dict(), doc = "") -> Section

An **exclusive slot** as one configuration section `[name]` (ONE-m, #476):
the selector `key` names the provider, and the section holds that
provider's options and no other's.

```julia
slot_section("pressure"; key = "solver", doc = "The pressure solver.") do s
    provider!(s, "cosine_fft"; default = [:model])
    provider!(s, "multigrid") do p
        option!(p, "tol", Float64; default = 1e-10, doc = "relative residual")
    end
end
```

```toml
[pressure]
solver = "multigrid"
tol = 1e-8
```

`f(s)` declares the providers with [`provider!`](@ref) (and options every
provider shares with [`option!`](@ref) on `s`). A provider is available
on the domain `kinds` it lists; `kind(tables)` tells the kind of a domain
from its section tables (the model's: `:urban` for an urban site, else
`:model`), and every kind with a provider has exactly one default
provider (checked when the schema is made: the parts of one slot may be
declared apart and merged, [`merge_sections`](@ref)). On a kind with no provider the section takes no key.

Validation ([`load_run_config`](@ref)): the selector must name a provider
of the domain's kind; every other key must be an option of the chosen
provider (a key of another provider is an error that names it). The
resolved config holds the selector and the chosen provider's options,
defaults included, and nothing else. A selector set at a higher level
than an option — an override (`--set`, [`with`](@ref)) over the file, or
a `[domain.<name>]` block over the run-level section — replaces the
provider: the lower level's options that the new provider does not have
are dropped.
"""
function slot_section(f, name::AbstractString; key::AbstractString, kind = _ -> :model,
                      kind_text = Dict{Symbol,String}(), doc::AbstractString = "")
    occursin(_KEY_RE, name) ||
        throw(ArgumentError("section name \"$name\" is not ASCII snake_case"))
    occursin(_KEY_RE, key) ||
        throw(ArgumentError("[$name]: the selector key \"$key\" is not ASCII snake_case"))
    sel = Selector(String(key), Provider[], Dict{Symbol,String}(), kind,
                   Dict{Symbol,String}(k => String(v) for (k, v) in kind_text))
    s = Section(String(name), String(doc), Option[], sel)
    f(s)
    isempty(sel.providers) && throw(ArgumentError("[$name]: a slot section needs a provider"))
    any(o -> o.key == sel.key, s.options) &&
        throw(ArgumentError("[$name]: the option \"$(sel.key)\" is the selector key"))
    return s
end

"""
    check_slot_section(s::Section) -> s

What a slot section of a schema must satisfy (checked by
[`ConfigSchema`](@ref), after the parts of one slot are merged,
[`merge_sections`](@ref)): one default provider on every domain kind it
has providers on.
"""
function check_slot_section(s::Section)
    sel = s.selector
    sel === nothing && return s
    for k in unique(reduce(vcat, (p.kinds for p in sel.providers)))
        haskey(sel.defaults, k) || throw(ArgumentError("[$(s.name)]: no default provider on the domain kind :$k"))
    end
    return s
end

"""
    provider!(s::Section, name; kinds = [:model], default = Symbol[], doc = "") -> Provider
    provider!(f, s::Section, name; …)

Declare the provider `name` of the slot section `s` ([`slot_section`](@ref)),
available on the domain `kinds`, the default provider of the kinds in
`default`; `f(p)` declares its options with [`option!`](@ref) on `p`.
"""
provider!(s::Section, name::AbstractString; kw...) = provider!(_ -> nothing, s, name; kw...)
function provider!(f, s::Section, name::AbstractString; kinds = [:model], default = Symbol[],
                   doc::AbstractString = "")
    sel = s.selector
    sel === nothing && throw(ArgumentError("[$(s.name)] is not a slot section"))
    occursin(_KEY_RE, name) ||
        throw(ArgumentError("[$(s.name)]: the provider name \"$name\" is not ASCII snake_case"))
    any(p -> p.name == name && !isempty(intersect(p.kinds, kinds)), sel.providers) &&
        throw(ArgumentError("[$(s.name)]: the provider \"$name\" is declared twice"))
    p = Section("$(s.name)", String(doc), Option[])
    f(p)
    for o in p.options
        (o.key == sel.key || any(x -> x.key == o.key, s.options)) &&
            throw(ArgumentError("[$(s.name)] $name: the option \"$(o.key)\" is the selector or a shared option"))
    end
    for k in default
        k in kinds || throw(ArgumentError("[$(s.name)] $name: default on :$k, which is not one of its kinds"))
        haskey(sel.defaults, k) && throw(ArgumentError("[$(s.name)]: two default providers on :$k"))
        sel.defaults[k] = String(name)
    end
    pr = Provider(String(name), String(doc), Symbol[kinds...], p.options)
    push!(sel.providers, pr)
    return pr
end

"""
    merge_sections(sections; kind = nothing, kind_text = nothing) -> Vector{Section}

`sections` with the slot sections of the same name merged into one (their
providers, defaults and shared options pooled; the selector keys must
agree), in the order of first appearance. Two parts of a schema can so
declare providers of one slot: the model's SGS component its closures and
the urban site its own. `kind` and `kind_text`, when given, become every
slot section's (see [`slot_section`](@ref)). Two plain sections of one name,
two providers of one name on one kind or two defaults on one kind are an
`ArgumentError`.
"""
function merge_sections(secs; kind = nothing, kind_text = nothing)
    out = Section[]
    for s in secs
        i = findfirst(x -> x.name == s.name, out)
        if i === nothing
            push!(out, s)
            continue
        end
        a = out[i]
        (a.selector === nothing || s.selector === nothing) &&
            throw(ArgumentError("section [$(s.name)] is declared twice"))
        a.selector.key == s.selector.key ||
            throw(ArgumentError("[$(s.name)]: two selector keys, \"$(a.selector.key)\" and \"$(s.selector.key)\""))
        for p in s.selector.providers, q in a.selector.providers
            p.name == q.name && !isempty(intersect(p.kinds, q.kinds)) &&
                throw(ArgumentError("[$(s.name)]: the provider \"$(p.name)\" is declared twice"))
        end
        for k in intersect(keys(a.selector.defaults), keys(s.selector.defaults))
            throw(ArgumentError("[$(s.name)]: two default providers on :$k"))
        end
        sel = Selector(a.selector.key, vcat(a.selector.providers, s.selector.providers),
                       merge(a.selector.defaults, s.selector.defaults), a.selector.kind,
                       merge(a.selector.kind_text, s.selector.kind_text))
        out[i] = Section(a.name, isempty(a.doc) ? s.doc : a.doc, vcat(a.options, s.options), sel)
    end
    kind === nothing && kind_text === nothing && return out
    return Section[s.selector === nothing ? s :
                   Section(s.name, s.doc, s.options,
                           Selector(s.selector.key, s.selector.providers, s.selector.defaults,
                                    something(kind, s.selector.kind),
                                    kind_text === nothing ? s.selector.kind_text :
                                                            Dict{Symbol,String}(k => String(v) for (k, v) in kind_text)))
                   for s in out]
end

"The providers of the slot section `s` on the domain kind `kind`."
providers(s::Section, kind::Symbol) = Provider[p for p in s.selector.providers if kind in p.kinds]

"The provider `name` of the slot section `s` on the domain kind `kind`, or `nothing`."
function find_provider(s::Section, kind::Symbol, name::AbstractString)
    i = findfirst(p -> p.name == name && kind in p.kinds, s.selector.providers)
    return i === nothing ? nothing : s.selector.providers[i]
end

"""
    section_options(s::Section) -> Vector{Option}

Every option `s` can hold: its own and, for a slot section, those of every
provider (a key two providers share appears once per provider).
"""
section_options(s::Section) =
    s.selector === nothing ? s.options : vcat(s.options, (p.options for p in s.selector.providers)...)

"The kind of a domain whose section tables are `tables`, for the slot section `s` (`:model` on error)."
function domain_kind(s::Section, tables)
    k = try
        s.selector.kind(tables)
    catch
        :model
    end
    return k isa Symbol ? k : :model
end

"How a message names the domain kind `k` of slot section `s`."
kind_text(s::Section, k::Symbol) = get(s.selector.kind_text, k, "a domain of kind :$k")

"""
    option!(s::Section, key, T; default, doc = "", units = "", allowed = nothing,
            range = nothing, open_lower = false, required = false, input = true,
            allow_julia = false) -> Option

Declare the option `key` of type `T` in section `s`. Types:

  * `Float64` (a TOML integer is accepted and converted), `Int`, `Bool`, `String`;
  * `Symbol`, written as a string and checked against `allowed`;
  * [`Path`](@ref), resolved against the directory of the file that set it;
  * `NTuple{N,T}` of a scalar type, a TOML array of exactly `N` elements;
  * `Vector{T}`;
  * `Date` and `Time` (TOML local date and local time);
  * [`Optional{T}`](@ref Optional), where `"none"` stands for `nothing`;
  * [`OpenTable(validator)`](@ref OpenTable), a table whose keys a driver decides.

`default` is the value a missing key takes (a zero-argument function is
called at load time, so an environment variable can fill it; the resolved
value is always written). An `Optional` option defaults to `nothing`, an
`OpenTable` to an empty table; any other option needs a `default` or
`required = true`. `range = (lo, hi)` bounds numbers (closed, or open at
`lo` with `open_lower = true`; `Inf` for no upper bound), elementwise for
tuples and vectors; `allowed` lists the permitted values (elementwise too).
`units` and `doc` go into the generated option reference.
"""
function option!(s::Section, key::AbstractString, T; default = NoDefault(),
                 doc::AbstractString = "", units::AbstractString = "", allowed = nothing,
                 range = nothing, open_lower::Bool = false, required::Bool = false,
                 input::Bool = true, allow_julia::Bool = false)
    occursin(_KEY_RE, key) ||
        throw(ArgumentError("option key \"$key\" of [$(s.name)] is not ASCII snake_case"))
    any(o -> o.key == key, s.options) &&
        throw(ArgumentError("option \"$key\" is declared twice in [$(s.name)]"))
    _valid_spec(T) || throw(ArgumentError("[$(s.name)] $key: unsupported option type $T"))
    if default isa NoDefault && !required
        T isa Type && T <: Optional ? (default = nothing) :
        T isa OpenTable ? (default = Dict{String,Any}()) :
        throw(ArgumentError("[$(s.name)] $key needs a default or required = true"))
    end
    required && !(default isa NoDefault) &&
        throw(ArgumentError("[$(s.name)] $key: a required option has no default"))
    rng = range === nothing ? nothing : (range[1], range[2])
    al = allowed === nothing ? nothing : Any[allowed...]
    o = Option(String(key), T, default, String(doc), String(units), al, rng, open_lower,
               required, input, allow_julia)
    if !(default isa NoDefault || default isa Function)
        ok, _, why = _coerce(o, default, true, pwd(), String(key))
        ok || throw(ArgumentError("[$(s.name)] $key: the default $(repr(default)) is invalid: $why"))
    end
    push!(s.options, o)
    return o
end

"""
    ConfigSchema(name; sections = Section[], checks = Function[], defaults = (),
                 domain_sections = String[])

The schema of the model (`name = "model"`) or of a legacy case `name`: the
common sections every run has (`[run]`, `[fftw]`, `[logging]`,
`[checkpoint]`, `[output]`; see the Configuration page) followed by
`sections`. A section named like a common one adds its options to that
section (the model's `[output] variables`); an option already in the common
section, or a section declared twice, is an `ArgumentError`. A legacy
case's `[run] legacy` defaults to, and must equal, `name`. `checks` are
cross-field checks ([`check!`](@ref)): functions, or `"what it checks, in
words" => f` pairs, whose text goes into the option reference
([`option_reference`](@ref)). `defaults` (`"section.key" => value` pairs)
give a common option this schema's own default (the Berlin case's
`[output] cadence` of 1800 s); its type and constraints stay the common
ones.

`domain_sections` names the **per-domain** sections (ONE-b, #451): the
model's grid, core and component sections, plus `"domain"`, the domain's
own keys (`id`, `static_driver`, `dynamic_driver`). Run-level values of
these sections apply to every domain; a `[[domain]]` block, with
`[domain.<section>]` tables, replaces keys for its domain. A schema without
domain sections (a legacy case) has no `[[domain]]`.
"""
struct ConfigSchema
    name::String
    sections::Vector{Section}
    checks::Vector{Pair{String,Any}}
    domain_sections::Vector{String}
    function ConfigSchema(name::String, sections::Vector{Section}, checks, domain_sections = String[])
        names = [s.name for s in sections]
        foreach(check_slot_section, sections)
        for d in domain_sections
            d in names || throw(ArgumentError("schema \"$name\": the domain section [$d] is not a section"))
        end
        return new(name, sections, Pair{String,Any}[_check_entry(c) for c in checks], String[domain_sections...])
    end
end

"A cross-field check as `description => f` (a bare function has the description `\"\"`)."
_check_entry(c::Pair) = String(first(c)) => last(c)
_check_entry(f) = "" => f

"Whether `s` (a section name) is per-domain in `sch`."
is_domain_section(sch::ConfigSchema, s::AbstractString) = s in sch.domain_sections

function ConfigSchema(name::AbstractString; sections = Section[], checks = Any[], defaults = (),
                      domain_sections = String[])
    all_s = _common_sections(String(name))
    ncommon = length(all_s)
    defaults isa Pair && (defaults = (defaults,))
    for (key, d) in defaults                # a driver's own default of a common option
        sname, okey = _split_key(String(key))
        i = findfirst(c -> c.name == sname, all_s)
        j = i === nothing ? nothing : findfirst(o -> o.key == okey, all_s[i].options)
        j === nothing && throw(ArgumentError("schema \"$name\": \"$key\" in defaults is not a common option"))
        o = all_s[i].options[j]
        o2 = Option(o.key, o.type, d, o.doc, o.units, o.allowed, o.range, o.open_lower, o.required,
                    o.input, o.allow_julia)
        if !(d isa Function)
            ok, _, why = _coerce(o2, d, true, pwd(), String(key))
            ok || throw(ArgumentError("schema \"$name\": the default $(repr(d)) of $key is invalid: $why"))
        end
        opts = copy(all_s[i].options); opts[j] = o2
        all_s[i] = Section(all_s[i].name, all_s[i].doc, opts)
    end
    for s in sections
        i = findfirst(c -> c.name == s.name, all_s)
        if i === nothing
            push!(all_s, s)
        elseif i <= ncommon                 # a driver's options added to a common section
            c = all_s[i]
            for o in s.options
                any(x -> x.key == o.key, c.options) &&
                    throw(ArgumentError("schema \"$name\": option \"$(o.key)\" of [$(s.name)] " *
                                        "is already a common option"))
            end
            all_s[i] = Section(c.name, c.doc, vcat(c.options, s.options))
        else
            throw(ArgumentError("schema \"$name\": section [$(s.name)] is declared twice"))
        end
    end
    return ConfigSchema(String(name), all_s, Any[checks...], String[domain_sections...])
end

"""
    check!(schema, f; doc = "") -> schema

Add a cross-field check: `f(cfg::RunConfig)` returns a vector of
[`ConfigError`](@ref)s (empty when the config is consistent). Run after
every option has passed its own checks; their problems go into the one
error the loader throws. `doc` says what it checks, in words: the option
reference lists it under "Cross-field checks".
"""
check!(schema::ConfigSchema, f; doc::AbstractString = "") =
    (push!(schema.checks, String(doc) => f); schema)

_split_key(key::String) = (i = findfirst('.', key);
                           i === nothing ? (key, "") : (key[1:prevind(key, i)], key[nextind(key, i):end]))

_section(schema::ConfigSchema, name) =
    (i = findfirst(s -> s.name == name, schema.sections); i === nothing ? nothing : schema.sections[i])
_option(s::Section, key) =
    (i = findfirst(o -> o.key == key, s.options); i === nothing ? nothing : s.options[i])

# ---------------------------------------------------------------------------
# the schema registry: the model and the legacy cases
# ---------------------------------------------------------------------------

"""The name the model's schema is registered under (ONE-b, #451)."""
const MODEL = "model"

const _DRIVERS = Dict{String,ConfigSchema}()

"""
    register_driver!(name; sections = Section[], checks = Function[], defaults = (),
                     domain_sections = String[]) -> ConfigSchema

Register a configuration schema ([`ConfigSchema`](@ref)): the model's
(`name = "model"`, [`MODEL`](@ref)), which a config without `[run] legacy`
selects, or that of a **legacy case** `name`, which `[run] legacy =
"name"` selects (an assembly not yet ported to components: the urban,
SBL, validation and example run types, removed by the later steps of the
single-model epic). Registering a name again replaces it. `"common"` (the
common sections only) is always registered.
"""
function register_driver!(name::AbstractString; sections = Section[], checks = Any[], defaults = (),
                          domain_sections = String[])
    schema = ConfigSchema(name; sections = sections, checks = checks, defaults = defaults,
                          domain_sections = domain_sections)
    _DRIVERS[String(name)] = schema
    return schema
end

"""
    driver_schema(name = "model") -> ConfigSchema

The schema registered as `name` ([`register_driver!`](@ref)): the model's
or a legacy case's.
"""
function driver_schema(name::AbstractString = MODEL)
    haskey(_DRIVERS, name) || throw(ConfigError("run.legacy",
        "unknown legacy case \"$name\"" * _suggest(name, legacy_cases()) *
        "; legacy cases: " * join(repr.(legacy_cases()), ", ")))
    return _DRIVERS[name]
end

"""
    registered_drivers() -> Vector{String}

The names of the registered schemas (the model's, `"common"` and the legacy
cases'), sorted.
"""
registered_drivers() = sort!(collect(keys(_DRIVERS)))

"""
    legacy_cases() -> Vector{String}

The legacy cases `[run] legacy` may name (the registered schemas but the
model's and `"common"`), sorted.
"""
legacy_cases() = sort!(filter(n -> !(n in (MODEL, "common")), collect(keys(_DRIVERS))))

# ---------------------------------------------------------------------------
# value checks
# ---------------------------------------------------------------------------

"A short description of what an option of type `T` takes, for messages."
_describe(::Type{Float64}) = "a float"
_describe(::Type{Int}) = "an integer"
_describe(::Type{Bool}) = "a boolean"
_describe(::Type{String}) = "a string"
_describe(::Type{Symbol}) = "a string"
_describe(::Type{Path}) = "a path (a string)"
_describe(::Type{Date}) = "a local date (yyyy-mm-dd)"
_describe(::Type{Time}) = "a local time (hh:mm:ss)"
_describe(::OpenTable) = "a table"
_describe(T::OneOf) = join(map(_describe, T.alts), " or ")
_describe(T::Type{<:Optional}) = _describe(_elem(T)) * " or \"none\""
function _describe(T::Type)
    if T <: Tuple
        return "an array of $(fieldcount(T)) $(_plural(fieldtype(T, 1)))"
    elseif T <: Vector
        return "an array of $(_plural(eltype(T)))"
    end
    return string(T)
end
_plural(T) = T === Float64 ? "floats" : T === Int ? "integers" : T === Bool ? "booleans" :
             T === Date ? "local dates" : T === Time ? "local times" :
             T === Path ? "paths" : T === String || T === Symbol ? "strings" :
             "values ($(_describe(T)))"

"The range / allowed-value clause of a message."
function _constraint_text(o::Option)
    s = ""
    if o.range !== nothing
        lo, hi = o.range
        s *= " in " * (o.open_lower ? "(" : "[") * string(lo) * ", " * string(hi) * "]"
    end
    o.allowed === nothing || (s *= " (allowed: " * join(map(_show_raw, o.allowed), ", ") * ")")
    return s
end

"`x` as the user wrote it, for messages: strings quoted, arrays bracketed."
function _show_raw(x)
    s = x isa AbstractString ? repr(String(x)) :
        x isa Symbol ? repr(String(x)) :
        x isa AbstractDict ? "a table {" * join(sort!(string.(collect(keys(x)))), ", ") * "}" :
        (x isa AbstractVector || x isa Tuple) ? "[" * join(map(_show_raw, x), ", ") * "]" :
        x === nothing ? "nothing" :
        x isa Function ? "a Julia function" : repr(x)
    return length(s) > 120 ? first(s, 117) * "..." : s
end

"""
    _coerce(o, x, julia, base, key) -> (ok, value, message)

Check the raw value `x` of option `o` and convert it to the typed value.
`julia = false` for a TOML value, `true` for a Julia value ([`with`](@ref));
`base` is the directory a relative `Path` resolves against.
"""
function _coerce(o::Option, x, julia::Bool, base::AbstractString, key::AbstractString)
    ok, v, why = _coerce_type(o.type, o, x, julia, base, key)
    if !ok && o.allow_julia && julia && !_toml_native(x)
        return (true, x, "")                     # a closure etc.: not replayable
    end
    ok || return (false, nothing,
                  isempty(why) ? "expected $(_describe(o.type))$(_constraint_text(o)), got $(_show_raw(x))" : why)
    return (true, v, "")
end

_fail() = (false, nothing, "")

function _coerce_type(T, o, x, julia, base, key)
    if T isa OneOf
        why = ""
        for a in T.alts
            ok, v, w = _coerce_type(a, o, x, julia, base, key)
            ok && return (true, v, "")
            # a table that one alternative takes but finds invalid: its reasons
            (a isa OpenTable && x isa AbstractDict && !isempty(w)) && (why = w)
        end
        return (false, nothing, why)
    elseif T isa OpenTable
        tbl = x isa AbstractDict ? _plain_table(x) : (julia && x isa NamedTuple) ? _plain_table(x) : nothing
        tbl === nothing && return _fail()
        probs = Base.invokelatest(T.validator, tbl)     # a driver loaded at run time
        isempty(probs) && return (true, tbl, "")
        msgs = String[p isa ConfigError ? join(("$(q.key): $(q.message)" for q in p.problems), "; ") :
                      string(p) for p in probs]
        return (false, nothing, join(msgs, "; "))
    elseif T <: Optional
        (x === nothing && julia) && return (true, nothing, "")
        x == "none" && return (true, nothing, "")
        return _coerce_type(_elem(T), o, x, julia, base, key)
    elseif T <: Tuple
        xs = x isa AbstractVector || (julia && x isa Tuple) ? collect(x) : nothing
        xs === nothing && return _fail()
        n = fieldcount(T)
        length(xs) == n || return (false, nothing,
            "expected $(_describe(T))$(_constraint_text(o)), got $(_show_raw(x)) ($(length(xs)) element$(length(xs) == 1 ? "" : "s"))")
        vals = Any[]
        for e in xs
            ok, v, _ = _coerce_type(fieldtype(T, 1), o, e, julia, base, key)
            ok || return _fail()
            push!(vals, v)
        end
        return (true, Tuple(vals), "")
    elseif T <: Vector
        xs = x isa AbstractVector || (julia && x isa Tuple) ? collect(x) : nothing
        xs === nothing && return _fail()
        out = eltype(T) <: Optional ? Vector{Union{Nothing,_valtype(_elem(eltype(T)))}}() :
              Vector{_valtype(eltype(T))}()
        for e in xs
            ok, v, _ = _coerce_type(eltype(T), o, e, julia, base, key)
            ok || return _fail()
            push!(out, v)
        end
        return (true, out, "")
    end
    ok, v = _scalar(T, x, julia, base)
    ok || return _fail()
    _within(o, v) || return _fail()
    return (true, v, "")
end

_valtype(::Type{Path}) = String
_valtype(T::Type{<:Tuple}) = Tuple{ntuple(_ -> _valtype(fieldtype(T, 1)), fieldcount(T))...}
_valtype(T::Type{<:Vector}) = Vector{_valtype(eltype(T))}
_valtype(T::Type) = T

function _scalar(T, x, julia, base)
    if T === Float64
        x isa Bool && return (false, nothing)
        x isa AbstractFloat && return (true, Float64(x))
        x isa Integer && return (true, Float64(x))
        julia && x isa Real && return (true, Float64(x))
    elseif T === Int
        x isa Bool && return (false, nothing)
        x isa Integer && typemin(Int) <= x <= typemax(Int) && return (true, Int(x))
    elseif T === Bool
        x isa Bool && return (true, x)
    elseif T === String
        x isa AbstractString && return (true, String(x))
    elseif T === Symbol
        x isa AbstractString && return (true, Symbol(x))
        julia && x isa Symbol && return (true, x)
    elseif T === Path
        x isa AbstractString && !isempty(x) && return (true, normpath(abspath(base, String(x))))
    elseif T === Date
        x isa Date && return (true, x)
    elseif T === Time
        x isa Time && return (true, x)
    end
    return (false, nothing)
end

function _within(o::Option, v)
    if o.allowed !== nothing && !(o.type isa OneOf && v isa Real)
        any(a -> isequal(a, v) || (v isa Symbol && a isa AbstractString && String(v) == a) ||
                 (a isa Symbol && v isa AbstractString && String(a) == v), o.allowed) || return false
    end
    if o.range !== nothing && v isa Real && !(v isa Bool)
        lo, hi = o.range
        isnan(v) && return false
        (o.open_lower ? v > lo : v >= lo) || return false
        v <= hi || return false
    end
    return true
end

"A table as a `Dict{String,Any}` with nested tables converted too."
function _plain_table(x)
    d = Dict{String,Any}()
    for (k, v) in pairs(x)
        d[string(k)] = (v isa AbstractDict || v isa NamedTuple) ? _plain_table(v) : v
    end
    return d
end

"Whether `x` is a value TOML can hold (recursively)."
_toml_native(x) = x isa Union{Bool,Integer,AbstractFloat,AbstractString,Symbol,Date,Time,DateTime,Nothing} ||
                  ((x isa AbstractVector || x isa Tuple) && all(_toml_native, x)) ||
                  (x isa AbstractDict && all(p -> _toml_native(p.second), x)) ||
                  (x isa NamedTuple && all(_toml_native, values(x)))

# ---------------------------------------------------------------------------
# "did you mean"
# ---------------------------------------------------------------------------

"""
    damerau_levenshtein(a, b) -> Int

The optimal-string-alignment Damerau–Levenshtein distance between `a`
and `b`: insertions, deletions, substitutions and transpositions of
adjacent characters each cost 1.
"""
function damerau_levenshtein(a::AbstractString, b::AbstractString)
    s = collect(a); t = collect(b)
    m, n = length(s), length(t)
    d = zeros(Int, m + 1, n + 1)
    for i in 0:m; d[i+1, 1] = i; end
    for j in 0:n; d[1, j+1] = j; end
    for i in 1:m, j in 1:n
        c = s[i] == t[j] ? 0 : 1
        d[i+1, j+1] = min(d[i, j+1] + 1, d[i+1, j] + 1, d[i, j] + c)
        if i > 1 && j > 1 && s[i] == t[j-1] && s[i-1] == t[j]
            d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + 1)
        end
    end
    return d[m+1, n+1]
end

"`\"; did you mean \\\"x\\\"?\"` for the closest candidate within distance 2, else `\"\"`."
function _suggest(key::AbstractString, candidates)
    best = ""; bd = 3
    for c in sort!(collect(String, candidates))
        dd = damerau_levenshtein(key, c)
        dd < bd && (best = c; bd = dd)
    end
    return isempty(best) ? "" : "; did you mean \"$best\"?"
end
