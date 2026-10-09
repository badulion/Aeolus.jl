# The `berlin` run driver (CFG-c, #358): the configuration schema of the
# Berlin case, its conversion to the CaseConfig that build_case consumes,
# and the run-driver contract of CFG-b (#357), so that
#
#   julia --project -m Aeolus run examples/urban/configs/berlin_summer.toml [--set clock.nsteps=…]
#
# runs the flagship. Included last in BerlinCase.jl: the schema is built
# from the coupling registry (every slot, provider, addon and
# default_options) and from the parent's configuration.
#
# One source of truth for the defaults: the schema holds them.
# `CaseConfig(; kw...)` and `ParentConfig(; kw...)` start from them, and
# the per-coupling options are validated against the couplings'
# `default_options` (the key set is closed, the type of a value is the
# type of its default), so no second schema has to be kept in step with
# the coupling files.

using Aeolus.Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, Option, OpenTable,
                     Optional, OneOf, Path, section, option!, config_origin, config_schema, config_sources
using Aeolus.IO: RunFolder, CheckpointSchedule, with_monitor, check_blowup!, output_dir
using Dates: Date, year, month, day

"Directory of the shipped Berlin configs (`extends = \"berlin_summer.toml\"` resolves here)."
const CONFIG_DIR = joinpath(URBAN_DIR, "configs")

"""
    BerlinDriver()

The `berlin` run driver ([`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver)):
`[run] driver = "berlin"`. `build` makes the [`CaseConfig`](@ref) of the
config and [`build_case`](@ref)s it; `run!` runs it with
[`run_case!`](@ref) under the run monitor (the diagnostics CSV, the
blowup guard), writing the NetCDF output into the run folder's `output/`
unless `[output] path` says otherwise. `aeolus options berlin` prints
the schema. Shipped configs: `examples/urban/configs/` ([`SCENARIO_FILES`](@ref)).
"""
struct BerlinDriver <: AbstractRunDriver end

Config.driver_name(::BerlinDriver) = "berlin"
Config.config_dir(::BerlinDriver) = CONFIG_DIR
Config.driver_defaults(::BerlinDriver) = ("output.cadence" => 1800.0, "run.name" => "alexanderplatz")

# ---------------------------------------------------------------------------
# CaseConfig fields ↔ schema options
# ---------------------------------------------------------------------------

"""
    CASE_FIELDS

Every [`CaseConfig`](@ref) field and the dotted keys of the `berlin`
schema it is made of (`physics` is every key of `[physics]` but
`options`). The options of the schema that make a `CaseConfig` are
exactly the union of these keys, each used once (`test/config_berlin.jl`).
"""
const CASE_FIELDS = (
    name = ("run.name",), scenario = ("run.scenario",),
    size = ("grid.size",), extent = ("grid.extent",), halo = ("grid.halo",),
    static = ("geometry.static",), parent_static = ("geometry.parent_static",),
    lat0 = ("geometry.lat0",), lon0 = ("geometry.lon0",), offset = ("geometry.offset",),
    elevation = ("geometry.elevation",), wall_subdivisions = ("geometry.wall_subdivisions",),
    osm_path = ("preprocess.osm_path",), min_height = ("preprocess.min_height",),
    park_lai = ("preprocess.park_lai",), landuse = ("preprocess.landuse",),
    date = ("clock.date",), utc_offset = ("clock.utc_offset",), start_h = ("clock.start_h",),
    dt = ("clock.dt",), nsteps = ("clock.nsteps",),
    cadence = ("cadence.*", "output.cadence"),
    ν = ("dynamics.nu",), θ₀ = ("dynamics.theta0",), g = ("dynamics.g",),
    f_coriolis = ("dynamics.f_coriolis",), C_s = ("dynamics.c_s",), Pr_t = ("dynamics.pr_t",),
    sgs_stability = ("dynamics.sgs_stability",), sponge_α_max = ("dynamics.sponge_alpha_max",),
    sponge_frac = ("dynamics.sponge_frac",), θ_top = ("dynamics.theta_top",),
    lateral = ("boundaries.lateral",), outflow = ("boundaries.outflow",),
    boundary_frame = ("boundaries.boundary_frame",), θ_relaxation = ("boundaries.theta_relaxation",),
    analysis_buffer = ("boundaries.analysis_buffer",),
    sounding = ("initial.sounding",), dynamic = ("initial.dynamic",), θ_surface = ("initial.theta_surface",),
    θ_inversion_z = ("initial.theta_inversion_z",), θ_lapse = ("initial.theta_lapse",),
    u0 = ("initial.u0",), qv0 = ("initial.qv0",), qv_scale = ("initial.qv_scale",),
    Ts0 = ("initial.ts0",), Ts_water_offset = ("initial.ts_water_offset",),
    T_soil0 = ("initial.t_soil0",), p_surface = ("initial.p_surface",),
    physics = ("physics.*",), options = ("physics.options",),
    canopy_top = ("ledger.canopy_top",),
    output_path = ("output.path", "output.netcdf"), child_boundaries = ("output.child_boundaries",),
    backend = ("run.backend", "run.serial"))

"""
    PARENT_FIELDS

Every [`ParentConfig`](@ref) field but `child` and its key in
`[physics.options.nesting.parent]` (the ASCII name of the field).
"""
const PARENT_FIELDS = (
    center = "center", extent = "extent", refinement = "refinement", vertical = "vertical",
    z_uniform = "z_uniform", ratio = "ratio", dz_max = "dz_max", halo = "halo",
    surface = "surface", Q_surface = "q_surface", E_surface = "e_surface",
    surface_period = "surface_period", U_min = "u_min",
    radiation = "radiation", radiation_period = "radiation_period",
    geostrophic = "geostrophic", z_g = "z_g", nudging_τ = "nudging_tau",
    nudging_factor = "nudging_factor", nudging_depth = "nudging_depth", divergence = "divergence",
    damping_rate = "damping_rate", damping_frac = "damping_frac", θ_top = "theta_top",
    θ_init = "theta_init", qv_init = "qv_init", u_init = "u_init",
    perturbation = "perturbation", perturbation_depth = "perturbation_depth", seed = "seed",
    hours = "hours", output_period = "output_period",
    sgs = "sgs", stepper = "stepper", C_advective = "c_advective", moisture = "moisture")

# ---------------------------------------------------------------------------
# the cadences the couplings declare
# ---------------------------------------------------------------------------

"""
    declared_cadences(C) -> Vector{Symbol}

The named cadences (`cfg.cadence` keys) coupling `C` uses: the `Symbol`
periods of its `stages` (required when `C` is enabled) and its
[`cadence_keys`](@ref) (read if present). `:output` is `[output] cadence`.
"""
declared_cadences(C) = unique!(vcat(Symbol[st[3].period for st in stages(C) if st[3].period isa Symbol],
                                    Symbol[cadence_keys(C)...]))
_required_cadences(C) = Symbol[st[3].period for st in stages(C) if st[3].period isa Symbol]

"The defaults (s) of the named cadences; a key without one is unset unless a file sets it."
const CADENCE_DEFAULTS = Dict(:radiation => 600.0, :soil => 30.0)

# ---------------------------------------------------------------------------
# per-coupling options: the type of a value is the type of its default
# ---------------------------------------------------------------------------

_isnum(x) = x isa Real && !(x isa Bool)

"What a value shaped like the default `d` is, for messages."
_shape(d::Bool) = "a boolean"
_shape(d::UInt8) = "an integer in [0, 255]"
_shape(d::Integer) = "an integer"
_shape(d::AbstractFloat) = "a number"
_shape(d::Symbol) = "a string"
_shape(d::AbstractString) = "a string"
_shape(::Nothing) = "\"none\" or a number"
_shape(d::Tuple) = "an array of $(length(d)) values ($(join(unique(_shape.(d)), ", ")))"
_shape(d::AbstractVector) = isempty(d) ? "an array" : "an array of $(_shape(d[1]))s"
_shape(d::NamedTuple) = "a table with the keys $(join(keys(d), ", "))"
_shape(d::AbstractDict) = "a table of surface class (an integer key) => $(_shape(first(values(d))))"
_shape(d) = "a value like $(repr(d))"

_raw(x) = x isa AbstractString ? repr(x) : x isa AbstractDict ? "a table" : repr(x)

"""
    option_value(d, x, path) -> (ok, value, message)

The Julia value of the TOML value `x` for an option whose default is
`d`: the same type (a TOML integer is taken for a float, a string for a
`Symbol`, `"none"` or a number for a `nothing` default, an array for a
tuple or a vector, a table for a `NamedTuple` (its missing keys keep the
default's) or for a table by surface class). A value TOML cannot hold (a
function, an `AbstractBoundaryData`, set from Julia with `with`) is taken
as it is.
"""
function option_value(d, x, path::AbstractString)
    Config._toml_native(x) || return (true, x, "")
    bad = (false, nothing, "$path: expected $(_shape(d)), got $(_raw(x))")
    if d isa Bool
        return x isa Bool ? (true, x, "") : bad
    elseif d isa UInt8
        return x isa Integer && !(x isa Bool) && 0 <= x <= 255 ? (true, UInt8(x), "") : bad
    elseif d isa Integer
        return x isa Integer && !(x isa Bool) ? (true, Int(x), "") : bad
    elseif d isa AbstractFloat
        return _isnum(x) ? (true, Float64(x), "") : bad
    elseif d isa Symbol
        return x isa Union{AbstractString,Symbol} ? (true, Symbol(x), "") : bad
    elseif d isa AbstractString
        return x isa AbstractString ? (true, String(x), "") : bad
    elseif d === nothing
        x == "none" && return (true, nothing, "")
        return _isnum(x) ? (true, Float64(x), "") : bad
    elseif d isa Tuple
        (x isa Union{AbstractVector,Tuple} && length(x) == length(d)) || return bad
        vals = Any[]
        for (i, e) in enumerate(x)
            ok, v, why = option_value(d[i], e, "$path[$i]")
            ok || return (false, nothing, why)
            push!(vals, v)
        end
        return (true, Tuple(vals), "")
    elseif d isa AbstractVector
        x isa Union{AbstractVector,Tuple} || return bad
        isempty(d) && return (true, collect(x), "")
        vals = eltype(d)[]
        for (i, e) in enumerate(x)
            ok, v, why = option_value(d[1], e, "$path[$i]")
            ok || return (false, nothing, why)
            push!(vals, v)
        end
        return (true, vals, "")
    elseif d isa NamedTuple
        x isa Union{AbstractDict,NamedTuple} || return bad
        t = x isa NamedTuple ? Dict(string(k) => v for (k, v) in pairs(x)) : x
        out = Dict{Symbol,Any}(pairs(d))
        for k in sort!(collect(String, keys(t)))
            haskey(d, Symbol(k)) ||
                return (false, nothing, "$path.$k: unknown key (keys: $(join(keys(d), ", ")))")
            ok, v, why = option_value(d[Symbol(k)], t[k], "$path.$k")
            ok || return (false, nothing, why)
            out[Symbol(k)] = v
        end
        return (true, NamedTuple{keys(d)}(Tuple(out[k] for k in keys(d))), "")
    elseif d isa AbstractDict
        x isa AbstractDict || return bad
        K, V = keytype(d), valtype(d)
        out = Dict{K,V}(d)
        for (k, v) in x
            kk = k isa Integer ? k : tryparse(Int, string(k))
            (kk !== nothing && (!(K <: Unsigned) || 0 <= kk <= typemax(K))) ||
                return (false, nothing, "$path: the key $(repr(string(k))) is not a surface class (an integer)")
            ok, vv, why = option_value(first(values(d)), v, "$path.$k")
            ok || return (false, nothing, why)
            out[K(kk)] = vv
        end
        return (true, out, "")
    end
    return isequal(d, x) ? (true, x, "") : bad
end

"""
    toml_form(x)

The TOML form of a Julia option value, the inverse of
[`option_value`](@ref): a `Symbol` as a string, `nothing` as `"none"`,
tuples and vectors as arrays, a `NamedTuple` as a table, a table by
surface class with string keys, [`ConstantForcing`](@ref),
[`SeriesForcing`](@ref) and [`ProfileTable`](@ref) as their tables. A
value TOML cannot hold (a function) is an `ArgumentError`.
"""
toml_form(x::Bool) = x
toml_form(x::Integer) = Int(x)
toml_form(x::AbstractFloat) = Float64(x)
toml_form(x::Union{AbstractString,Symbol}) = String(x)
toml_form(::Nothing) = "none"
toml_form(x::Date) = x
toml_form(x::Union{Tuple,AbstractVector}) = Any[toml_form(e) for e in x]
toml_form(x::NamedTuple) = Dict{String,Any}(string(k) => toml_form(v) for (k, v) in pairs(x))
toml_form(x::AbstractDict) = Dict{String,Any}(string(k) => toml_form(v) for (k, v) in x)
toml_form(f::ConstantForcing) = Dict{String,Any}("kind" => "constant", "value" => f.value)
toml_form(f::SeriesForcing) = Dict{String,Any}("kind" => "series", "t" => copy(f.t), "value" => copy(f.value))
toml_form(p::ProfileTable) = Dict{String,Any}("kind" => "table", "z" => copy(p.z), "values" => copy(p.values))
toml_form(x) = throw(ArgumentError("toml_form: $(typeof(x)) has no TOML form (set it from Julia with `with`)"))

"""
    toml_literal(x) -> String

`x` written as a TOML value ([`toml_form`](@ref)), the right-hand side of
an override `--set key=<value>`.
"""
toml_literal(x) = Config._emit_value(toml_form(x))

"""
    OPTION_FORMS

The per-coupling options whose TOML form is not the type of their
default: `coupling => option => f(x, path) -> (ok, value, message)`.

  * `nesting.parent`: the parent's own schema ([`parent_section`](@ref));
  * `nesting.z_anterp`: `"canopy"` or a height in m;
  * `nesting.parent_driver`: `"aeolus"`, `"profile"` or `"offline"` (an
    `AbstractBoundaryData` only from Julia);
  * `surface_exchange.z0m`: a table by surface class or `"dynamic"`;
  * `sgs_mlsgs.fn`: `"none"` (a model function only from Julia).
"""
const OPTION_FORMS = Dict{Symbol,Dict{Symbol,Function}}(
    :nesting => Dict{Symbol,Function}(
        :parent => (x, path) -> parent_value(x, path),
        :z_anterp => (x, path) -> x == "canopy" ? (true, :canopy, "") : _isnum(x) ? (true, Float64(x), "") :
                                 (false, nothing, "$path: expected \"canopy\" or a height in m, got $(_raw(x))"),
        :parent_driver => (x, path) -> x in ("aeolus", "profile", "offline") ? (true, Symbol(x), "") :
                                 (false, nothing, "$path: expected \"aeolus\", \"profile\" or \"offline\", got $(_raw(x))")),
    :surface_exchange => Dict{Symbol,Function}(
        :z0m => (x, path) -> x == "dynamic" ? (true, :dynamic, "") :
                             option_value(default_options(SurfaceExchangeMOST).z0m, x, path)),
    :sgs_mlsgs => Dict{Symbol,Function}(
        :fn => (x, path) -> x == "none" ? (true, nothing, "") :
                            (false, nothing, "$path: expected \"none\" (a model function can only be set from Julia), got $(_raw(x))")))

"The Julia value of option `key` of coupling `C` from the TOML value `x`."
function coupling_option_value(C, key::Symbol, x, path)
    # a value TOML cannot hold (set with `with`) is taken as it is; a table
    # holding one is still checked key by key (the parent's q_surface)
    (x isa AbstractDict || Config._toml_native(x)) || return (true, x, "")
    f = get(get(OPTION_FORMS, coupling_name(C), Dict{Symbol,Function}()), key, nothing)
    f === nothing || return f(x, path)
    return option_value(default_options(C)[key], x, path)
end

"""
    coupling_options_value(C, tbl, path) -> (problems, NamedTuple)

The options of coupling `C` set in the table `tbl` (`[physics.options.<name>]`)
whose value differs from the default, and the problems found (unknown
keys, listing `default_options(C)`'s keys, and values of the wrong type).
"""
function coupling_options_value(C, tbl::AbstractDict, path)
    probs = String[]; out = Pair{Symbol,Any}[]
    d = default_options(C)
    for k in sort!(collect(String, keys(tbl)))
        key = Symbol(k)
        if !haskey(d, key)
            push!(probs, "$path.$k: unknown option of coupling $(coupling_name(C))" *
                         Config._suggest(k, string.(keys(d))) *
                         (isempty(d) ? "; it has no options" : "; its options: " * join(keys(d), ", ")))
            continue
        end
        ok, v, why = coupling_option_value(C, key, tbl[k], "$path.$k")
        ok || (push!(probs, why); continue)
        isequal(v, d[key]) || push!(out, key => v)
    end
    return probs, (; out...)
end

"""
    options_value(tbl) -> (problems, Dict{Symbol,Any})

`[physics.options]`: a table per coupling name, each validated by
[`coupling_options_value`](@ref); the result holds, per coupling, the
options that differ from its `default_options` (what `CaseConfig.options`
holds).
"""
function options_value(tbl::AbstractDict)
    probs = String[]; out = Dict{Symbol,Any}()
    names = Dict(String(coupling_name(C)) => C for C in REGISTRY)
    for k in sort!(collect(String, keys(tbl)))
        C = get(names, k, nothing)
        if C === nothing
            push!(probs, "$k: no coupling named \"$k\"" * Config._suggest(k, keys(names)) *
                         "; couplings: " * join(sort!(collect(keys(names))), ", "))
            continue
        end
        v = tbl[k]
        if !(v isa AbstractDict)
            push!(probs, "$k: expected a table of the options of coupling $k, got $(_raw(v))")
            continue
        end
        p, nt = coupling_options_value(C, v, k)
        append!(probs, p)
        isempty(nt) || (out[Symbol(k)] = nt)
    end
    return probs, out
end

# ---------------------------------------------------------------------------
# the parent's schema
# ---------------------------------------------------------------------------

const _FORCING_DOC = "a function of the simulation time t (s): {kind = \"constant\", value} or " *
                     "{kind = \"series\", t = […], value = […]} (linear in t, constant beyond the ends)"
const _PROFILE_DOC = "{kind = \"constant\", value} or {kind = \"table\", z = […], values = […]} " *
                     "(linear in z, constant beyond the ends)"

"Check a strictly increasing, non-empty array of numbers and a matching value array."
function _series_problem(xs, vs, xname, vname)
    (xs isa AbstractVector && !isempty(xs) && all(_isnum, xs)) || return "$xname: expected a non-empty array of numbers"
    (vs isa AbstractVector && all(_isnum, vs)) || return "$vname: expected an array of numbers"
    length(xs) == length(vs) || return "$xname and $vname have different lengths ($(length(xs)) and $(length(vs)))"
    issorted(xs; lt = <=) || return "$xname: expected strictly increasing values"
    return ""
end

"A [`ConstantForcing`](@ref) or [`SeriesForcing`](@ref) from its TOML table."
function surface_forcing_value(x, path)
    Config._toml_native(x) || return (true, x, "")
    x isa AbstractDict || return (false, nothing, "$path: expected $_FORCING_DOC, got $(_raw(x))")
    kind = get(x, "kind", nothing)
    want = kind == "constant" ? ["kind", "value"] : kind == "series" ? ["kind", "t", "value"] : nothing
    want === nothing && return (false, nothing, "$path.kind: expected \"constant\" or \"series\", got $(_raw(kind))")
    extra = setdiff(collect(keys(x)), want)
    isempty(extra) || return (false, nothing, "$path: unknown key $(repr(first(sort!(extra)))) for kind = $(repr(kind)) (keys: $(join(want, ", ")))")
    for k in want
        haskey(x, k) || return (false, nothing, "$path.$k: required for kind = $(repr(kind))")
    end
    if kind == "constant"
        _isnum(x["value"]) || return (false, nothing, "$path.value: expected a number, got $(_raw(x["value"]))")
        return (true, ConstantForcing(Float64(x["value"])), "")
    end
    why = _series_problem(x["t"], x["value"], "$path.t", "$path.value")
    isempty(why) || return (false, nothing, why)
    return (true, SeriesForcing(Float64.(x["t"]), Float64.(x["value"])), "")
end

"An initial profile from its TOML form: `special` (`\"sounding\"` / `\"geostrophic\"`) or a table."
function profile_value(x, path, special::String, special_value)
    Config._toml_native(x) || return (true, x, "")
    x == special && return (true, special_value, "")
    x isa AbstractDict || return (false, nothing, "$path: expected \"$special\" or $_PROFILE_DOC, got $(_raw(x))")
    kind = get(x, "kind", nothing)
    want = kind == "constant" ? ["kind", "value"] : kind == "table" ? ["kind", "z", "values"] : nothing
    want === nothing && return (false, nothing, "$path.kind: expected \"constant\" or \"table\", got $(_raw(kind))")
    extra = setdiff(collect(keys(x)), want)
    isempty(extra) || return (false, nothing, "$path: unknown key $(repr(first(sort!(extra)))) for kind = $(repr(kind)) (keys: $(join(want, ", ")))")
    for k in want
        haskey(x, k) || return (false, nothing, "$path.$k: required for kind = $(repr(kind))")
    end
    if kind == "constant"
        _isnum(x["value"]) || return (false, nothing, "$path.value: expected a number, got $(_raw(x["value"]))")
        v = Float64(x["value"])
        return (true, special == "geostrophic" ? v : ProfileTable([0.0], [v]), "")
    end
    why = _series_problem(x["z"], x["values"], "$path.z", "$path.values")
    isempty(why) || return (false, nothing, why)
    return (true, ProfileTable(Float64.(x["z"]), Float64.(x["values"])), "")
end

"The parent options whose TOML form is not a schema type: `key => f(x, path)`."
const PARENT_FORMS = Dict{String,Function}(
    "q_surface" => surface_forcing_value, "e_surface" => surface_forcing_value,
    "theta_init" => (x, path) -> profile_value(x, path, "sounding", nothing),
    "qv_init" => (x, path) -> profile_value(x, path, "sounding", nothing),
    "u_init" => (x, path) -> profile_value(x, path, "geostrophic", :geostrophic))

"""
    parent_section() -> Section

The schema of [`ParentConfig`](@ref), `[physics.options.nesting.parent]`
(the defaults of `ParentConfig(; kw...)`). The keys are the ASCII names of
the fields ([`PARENT_FIELDS`](@ref)); `q_surface` / `e_surface`,
`theta_init` / `qv_init` and `u_init` take the tables of
[`surface_forcing_value`](@ref) and [`profile_value`](@ref).
"""
function parent_section()
    return section("parent"; doc = "The nest parent (ParentConfig, #259): physics.options.nesting.parent.") do s
        option!(s, "center", Optional{NTuple{2,Float64}}; units = "m",
                doc = "horizontal centre of the parent box relative to the OSM anchor; \"none\": on the child")
        option!(s, "extent", NTuple{3,Float64}; default = (6000.0, 6000.0, 3000.0), range = (0, Inf),
                open_lower = true, units = "m", doc = "the parent box")
        option!(s, "refinement", NTuple{3,Int}; default = (2, 2, 4), range = (1, 64),
                doc = "a parent cell is rx × ry × rz child cells")
        option!(s, "vertical", Symbol; default = :stretched, allowed = [:stretched, :uniform],
                doc = "stretched: Δz uniform up to z_uniform, then × ratio per level up to dz_max; uniform: Δz everywhere")
        option!(s, "z_uniform", Float64; default = 500.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "top of the uniform layers (stretched)")
        option!(s, "ratio", Float64; units = "1", default = 1.08, range = (1, Inf), doc = "growth of Δz per level (stretched)")
        option!(s, "dz_max", Float64; default = 100.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "largest Δz (stretched)")
        option!(s, "halo", NTuple{3,Int}; default = (2, 2, 2), range = (1, 8), doc = "halo cells per side")
        option!(s, "surface", Symbol; default = :tiles, allowed = [:tiles, :prescribed],
                doc = "tiles: the tiled urban mosaic (#249); prescribed: the kinematic fluxes q_surface, e_surface")
        option!(s, "q_surface", OpenTable(t -> _form_problems(surface_forcing_value, t, "q_surface"));
                default = Dict{String,Any}("kind" => "constant", "value" => 0.0), units = "K m s^-1",
                doc = "prescribed kinematic heat flux, " * _FORCING_DOC)
        option!(s, "e_surface", OpenTable(t -> _form_problems(surface_forcing_value, t, "e_surface"));
                default = Dict{String,Any}("kind" => "constant", "value" => 0.0), units = "kg kg^-1 m s^-1",
                doc = "prescribed kinematic moisture flux, " * _FORCING_DOC)
        option!(s, "surface_period", Float64; default = 60.0, range = (0, Inf), open_lower = true,
                units = "s", doc = "step of the surface tiles")
        option!(s, "u_min", Float64; default = 0.5, range = (0, Inf), units = "m s^-1",
                doc = "wind floor of the surface layer")
        option!(s, "radiation", Bool; default = true, doc = "column radiation on the parent's mean column")
        option!(s, "radiation_period", Float64; default = 600.0, range = (0, Inf), open_lower = true,
                units = "s", doc = "period of the column radiation")
        option!(s, "geostrophic", Bool; default = true,
                doc = "the geostrophic pressure gradient of the sounding wind above z_g")
        option!(s, "z_g", Float64; default = NaN, units = "m",
                doc = "height of the geostrophic wind; nan: 1.2 × the sounding's largest z_i")
        option!(s, "nudging_tau", Float64; default = 6 * 3600.0, range = (0, Inf), open_lower = true,
                units = "s", doc = "θ/qv nudging time scale; inf: off")
        option!(s, "nudging_factor", Float64; units = "1", default = 1.2, range = (0, Inf),
                doc = "the nudging acts above nudging_factor × the running maximum of z_i")
        option!(s, "nudging_depth", Float64; default = 200.0, range = (0, Inf), units = "m",
                doc = "depth of the nudging weight's ramp")
        option!(s, "divergence", Float64; default = 0.0, units = "s^-1",
                doc = "large-scale divergence of the subsidence w_s = -D z (3e-6 diverges by day, #312)")
        option!(s, "damping_rate", Float64; default = 0.01, range = (0, Inf), units = "s^-1",
                doc = "θ reference damping and velocity sponge rate")
        option!(s, "damping_frac", Float64; units = "1", default = 0.25, range = (0, 1),
                doc = "the damping acts over the top damping_frac of the domain")
        option!(s, "theta_top", Symbol; default = :initial_gradient, allowed = [:initial_gradient, :noflux],
                doc = "the θ top: the fixed initial gradient (#254) or no flux")
        option!(s, "theta_init", Symbol; default = :sounding,
                doc = "initial θ(z): \"sounding\" (the child's) or " * _PROFILE_DOC)
        option!(s, "qv_init", Symbol; default = :sounding,
                doc = "initial qv(z): \"sounding\" (the child's) or " * _PROFILE_DOC)
        option!(s, "u_init", Symbol; default = :geostrophic,
                doc = "initial u(z): \"geostrophic\" or " * _PROFILE_DOC)
        option!(s, "perturbation", Float64; default = 0.2, range = (0, Inf), units = "K",
                doc = "uniform random θ amplitude in the lowest perturbation_depth")
        option!(s, "perturbation_depth", Float64; default = 100.0, range = (0, Inf), units = "m",
                doc = "depth of the initial θ perturbation")
        option!(s, "seed", Int; default = 259, range = (0, typemax(Int)), doc = "seed of the perturbation")
        option!(s, "hours", Float64; default = 24.0, range = (0, Inf), units = "h",
                doc = "simulated length of a stand-alone parent run")
        option!(s, "output_period", Float64; default = 600.0, range = (0, Inf), open_lower = true,
                units = "s", doc = "cadence of the parent's diagnostics")
        option!(s, "sgs", Symbol; default = :tke, allowed = [:tke, :smagorinsky],
                doc = "the parent's closure: tke (Deardorff) or smagorinsky (the core's)")
        option!(s, "stepper", Symbol; default = :rk3, allowed = [:rk3, :euler],
                doc = "rk3 (SSP-RK3 under step_adaptive!) or euler")
        option!(s, "c_advective", Float64; units = "1", default = 0.9, range = (0, Inf), open_lower = true,
                doc = "advective Courant number of the parent's step pick")
        option!(s, "moisture", Bool; default = true, doc = "carry qv")
    end
end

_form_problems(f, t, key) = (r = f(t, key); r[1] ? String[] : String[r[3]])

const _PARENT_SECTION = Ref{Any}(nothing)
"The parent schema section, built once."
function parent_schema_section()
    _PARENT_SECTION[] === nothing && (_PARENT_SECTION[] = parent_section())
    return _PARENT_SECTION[]
end

"The Julia value of parent option `o` from `x` (TOML or Julia)."
function _parent_option_value(o::Option, x, path)
    f = get(PARENT_FORMS, o.key, nothing)
    f === nothing || return f(x, path)
    ok, v, why = Config._coerce(o, x, true, pwd(), path)
    return ok ? (true, v, "") : (false, nothing, "$path: $why")
end

"""
    parent_defaults() -> Dict{Symbol,Any}

The default of every [`ParentConfig`](@ref) field but `child`, from the
parent schema ([`parent_section`](@ref)), as Julia values.
"""
function parent_defaults()
    s = parent_schema_section()
    d = Dict{Symbol,Any}()
    for (f, key) in pairs(PARENT_FIELDS)
        o = Config._option(s, key)
        ok, v, why = _parent_option_value(o, o.key in keys(PARENT_FORMS) && o.default isa Symbol ?
                                                String(o.default) : o.default, key)
        ok || error("parent schema: the default of $key is invalid: $why")
        d[f] = v
    end
    return d
end

"""
    parent_value(x, path) -> (ok, NamedTuple, message)

`[physics.options.nesting.parent]`: every key checked against the parent
schema ([`parent_section`](@ref)); the result holds the `ParentConfig`
keywords whose value differs from the default.
"""
function parent_value(x, path)
    (x isa Union{AbstractDict,NamedTuple} || Config._toml_native(x)) || return (true, x, "")
    x isa Union{AbstractDict,NamedTuple} || return (false, nothing, "$path: expected a table of the parent's options, got $(_raw(x))")
    t = x isa NamedTuple ? Dict(string(k) => v for (k, v) in pairs(x)) : x
    s = parent_schema_section()
    field = Dict(v => k for (k, v) in pairs(PARENT_FIELDS))
    defaults = parent_defaults()
    probs = String[]; out = Pair{Symbol,Any}[]
    for k in sort!(collect(String, keys(t)))
        o = Config._option(s, k)
        if o === nothing
            push!(probs, "$path.$k: unknown option of the parent" * Config._suggest(k, [p.key for p in s.options]) *
                         "; its options: " * join((p.key for p in s.options), ", "))
            continue
        end
        ok, v, why = _parent_option_value(o, t[k], "$path.$k")
        ok || (push!(probs, why); continue)
        isequal(v, defaults[field[k]]) || push!(out, field[k] => v)
    end
    isempty(probs) || return (false, nothing, join(probs, "; "))
    return (true, (; out...), "")
end

# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------

_options_problems(tbl) = options_value(tbl)[1]

# the TOML form of a coupling option for the option reference (the OPTION_FORMS keys by hand)
function _option_form_text(name::Symbol, k::Symbol, v)
    (name, k) == (:nesting, :parent) && return "a table ([physics.options.nesting.parent])"
    (name, k) == (:nesting, :z_anterp) && return "\"canopy\" or a number (m)"
    (name, k) == (:nesting, :parent_driver) && return "\"aeolus\", \"profile\" or \"offline\""
    (name, k) == (:surface_exchange, :z0m) && return "\"dynamic\" or " * _shape(v)
    (name, k) == (:sgs_mlsgs, :fn) && return "\"none\" (a model function only from Julia)"
    return _shape(v)
end

"The default of a coupling option as TOML text, or what it is when TOML cannot hold it."
function _option_default_text(v)
    v isa ParentConfig && return "(the parent's defaults)"
    try
        return toml_literal(v)
    catch err
        err isa ArgumentError || rethrow()
        return "(a $(nameof(typeof(v))), set from Julia)"
    end
end

"""
    berlin_options_reference() -> Vector{Section}

The documented sub-tables of `[physics.options]` for the option
reference: one per coupling with options, `[physics.options.<name>]`,
listing its `default_options` (key, TOML form, default; generated, so it
cannot drift), and `[physics.options.nesting.parent]`, the parent's
schema ([`parent_section`](@ref)).
"""
function berlin_options_reference()
    out = Section[]
    for C in sort(REGISTRY; by = C -> String(coupling_name(C)))
        d = default_options(C)
        isempty(d) && continue
        name = coupling_name(C)
        opts = [Config.reference_option(String(k),
                    _option_form_text(name, k, v), _option_default_text(v))
                for (k, v) in pairs(d)]
        push!(out, Section(String(name), "The options of the coupling `$name` (`default_options(::Type{" *
                           "$(nameof(C))})`): only these keys, each a value of its default's form; " *
                           "their meaning is documented on the coupling (`?$(nameof(C))`).", opts))
    end
    p = parent_schema_section()
    push!(out, Section("nesting.parent", p.doc, p.options))
    return out
end

function Config.driver_sections(::BerlinDriver)
    run = section("run") do s
        option!(s, "scenario", Symbol; default = :summer,
                doc = "the scenario label (summer, winter, …): the output's comment; winter selects the subarctic-winter humidity of the column radiation")
        option!(s, "backend", Symbol; default = :cpu, allowed = [:cpu], doc = "the compute backend (Aeolus.Backends)")
        option!(s, "serial", Bool; default = false,
                doc = "CPUBackend(serial = true): the single-task reference")
    end
    grid = section("grid"; doc = "The child mesh (RegularGrid).") do s
        option!(s, "size", NTuple{3,Int}; default = (64, 64, 64), range = (1, 65536), doc = "cells in x, y, z")
        option!(s, "extent", NTuple{3,Float64}; default = (1000.0, 1000.0, 400.0), range = (0, Inf),
                open_lower = true, units = "m",
                doc = "domain size; 400 m deep so that the Fernsehturm fits and the top sits in the parent's uniform layers (#246)")
        option!(s, "halo", NTuple{3,Int}; default = (3, 3, 3), range = (1, 8),
                doc = "halo cells per side (3: WENO-Z5 moisture transport)")
    end
    geo = section("geometry"; doc = "The ground (static drivers) and the site.") do s
        option!(s, "static", Optional{Path};
                doc = "the static driver of the child (CFG-f #361; `aeolus preprocess static` writes it); " *
                      "\"none\": rasterise [preprocess] osm_path in memory")
        option!(s, "parent_static", Optional{Path};
                doc = "the static driver (land-use tiles) of the nest parent; \"none\": from [preprocess] landuse in memory")
        option!(s, "lat0", Float64; default = 52.5220, range = (-90, 90), units = "degree",
                doc = "latitude of the OSM projection's anchor (and of the solar position)")
        option!(s, "lon0", Float64; default = 13.4100, range = (-180, 180), units = "degree",
                doc = "longitude of the anchor")
        option!(s, "offset", NTuple{2,Float64}; default = (0.0, 0.0), units = "m",
                doc = "the window centre relative to the anchor")
        option!(s, "elevation", Float64; default = 34.0, units = "m", doc = "height above sea level")
        option!(s, "wall_subdivisions", Int; default = 1, range = (1, 64),
                doc = "wall facets per cell face of the radiosity scene")
    end
    pre = section("preprocess"; doc = "The sources of the static drivers (`aeolus preprocess static`, or in memory with static = \"none\").") do s
        option!(s, "osm_path", Path; default = joinpath(DATA_DIR, "alexanderplatz_real.osm"),
                doc = "the OpenStreetMap file of the buildings, parks and water")
        option!(s, "min_height", Float64; default = 8.0, range = (0, Inf), units = "m",
                doc = "buildings lower than this are dropped")
        option!(s, "park_lai", Float64; units = "m2 m-2", default = 2.5, range = (0, Inf), doc = "leaf-area index of park cells")
        option!(s, "landuse", Path; default = joinpath(DATA_DIR, "berlin_landuse_6km.csv"),
                doc = "the nest parent's land-use table (data/fetch_berlin_landuse.jl)")
    end
    clock = section("clock"; doc = "Date, start and steps.") do s
        option!(s, "date", Date; default = Date(2019, 6, 29),
                doc = "the local date (29 June 2019: a clear day of the heat wave, #247)")
        option!(s, "utc_offset", Float64; default = 2.0, range = (-14, 14), units = "h",
                doc = "local clock = UTC + utc_offset (2: CEST)")
        option!(s, "start_h", Float64; default = 6.0, range = (0, 24), units = "h", doc = "local hour at t = 0")
        option!(s, "dt", Float64; default = 2.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "the step (with the adaptive stepper: nsteps × dt is the simulated time)")
        option!(s, "nsteps", Int; default = 43_200, range = (0, typemax(Int)), doc = "steps to run (43200: 24 h)")
    end
    keys_ = sort!(unique!(filter!(!=(:output), reduce(vcat, (declared_cadences(C) for C in REGISTRY); init = Symbol[]))))
    users(k) = join(sort!([String(coupling_name(C)) for C in REGISTRY if k in declared_cadences(C)]), ", ")
    cad = section("cadence"; doc = "Periods (s) of the couplings' stages; [output] cadence is the output's.") do s
        for k in keys_
            if haskey(CADENCE_DEFAULTS, k)
                option!(s, String(k), Float64; default = CADENCE_DEFAULTS[k], range = (0, Inf), units = "s",
                        doc = "the period named :$k (used by $(users(k)))")
            else
                option!(s, String(k), Optional{Float64}; range = (0, Inf), units = "s",
                        doc = "the period named :$k (used by $(users(k))); \"none\": not set")
            end
        end
    end
    dyn = section("dynamics"; doc = "Parameters of the dynamical core.") do s
        option!(s, "nu", Float64; default = 2.0, range = (0, Inf), units = "m^2 s^-1", doc = "kinematic viscosity ν")
        option!(s, "theta0", Float64; default = 293.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "reference potential temperature θ₀ of the buoyancy")
        option!(s, "g", Float64; default = 9.81, range = (0, Inf), open_lower = true, units = "m s^-2",
                doc = "gravitational acceleration")
        option!(s, "f_coriolis", Float64; default = 1e-4, units = "s^-1", doc = "Coriolis parameter")
        option!(s, "c_s", Float64; units = "1", default = 0.1, range = (0, Inf), doc = "Smagorinsky coefficient C_s of the core")
        option!(s, "pr_t", Float64; units = "1", default = 1 / 3, range = (0, Inf), open_lower = true,
                doc = "turbulent Prandtl number")
        option!(s, "sgs_stability", Symbol; default = :brown94,
                allowed = [:brown94, :lilly, :long_tail, :short_tail, :most, :none],
                doc = "the stability correction every eddy-viscosity sgs provider wraps its closure in (P55-e, #187); none: uncorrected")
        option!(s, "sponge_alpha_max", Float64; default = 0.05, range = (0, Inf), units = "s^-1",
                doc = "velocity sponge rate at the top")
        option!(s, "sponge_frac", Float64; units = "1", default = 0.75, range = (0, 1),
                doc = "the sponge starts at sponge_frac × Lz")
        option!(s, "theta_top", OneOf(Symbol, Float64); default = :initial_gradient,
                allowed = [:initial_gradient, :noflux], units = "K m^-1",
                doc = "θ condition of a closed top (#254): initial_gradient, noflux or a gradient in K/m")
    end
    bnd = section("boundaries"; doc = "Lateral boundaries without the nest (N-open-d, #262).") do s
        option!(s, "lateral", Symbol; default = :periodic, allowed = [:periodic, :open],
                doc = "the four sides when physics.nesting = false: periodic or open (fed by the geostrophic wind and the initial θ)")
        option!(s, "outflow", Symbol; default = :dirichlet, allowed = [:dirichlet, :convective, :orlanski],
                doc = "outflow condition of the open or nest faces (N-open-c, #260)")
        option!(s, "boundary_frame", Int; default = 3, range = (0, typemax(Int)),
                doc = "cells: buildings closer than this to an open face are removed")
        option!(s, "theta_relaxation", Bool; default = true,
                doc = "relax θ over the sponge layer toward the boundary data's profile")
        option!(s, "analysis_buffer", Float64; default = ANALYSIS_BUFFER_DEFAULT, range = (0, Inf), units = "m",
                doc = "columns closer than this to a lateral open face are left out of the diagnostics " *
                      "compared with observations (the inflow fetch, #268)")
    end
    ini = section("initial"; doc = "The initial state: the sounding, or the idealised profile.") do s
        option!(s, "sounding", Optional{Path}; default = joinpath(DATA_DIR, "lindenberg_10393_20190629.csv"),
                doc = "radiosonde CSV of the initial θ(z), qv(z) and surface pressure; \"none\": the idealised profile below")
        option!(s, "dynamic", Optional{Path};
                doc = "the dynamic driver (CFG-g #362, `aeolus preprocess dynamic`): a sounding driver (--from-sounding), " *
                      "used instead of sounding; or, with physics.options.nesting.parent_driver = \"offline\", the " *
                      "parent-made driver of the nest faces (--from-parent); \"none\": no driver")
        option!(s, "theta_surface", Float64; default = 293.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "idealised: θ at the ground")
        option!(s, "theta_inversion_z", Float64; default = 100.0, range = (0, Inf), units = "m",
                doc = "idealised: base of the capping inversion")
        option!(s, "theta_lapse", Float64; default = 3e-3, units = "K m^-1", doc = "idealised: dθ/dz above it")
        option!(s, "u0", Float64; default = 1.0, units = "m s^-1", doc = "initial u (without the nest)")
        option!(s, "qv0", Float64; default = 0.010, range = (0, 1), units = "kg kg^-1",
                doc = "idealised: qv at the ground")
        option!(s, "qv_scale", Float64; default = 800.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "idealised: qv(z) = qv0 exp(-z / qv_scale)")
        option!(s, "ts0", Float64; default = 285.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "initial surface temperature")
        option!(s, "ts_water_offset", Float64; default = 5.0, units = "K",
                doc = "water surfaces start at ts0 + ts_water_offset")
        option!(s, "t_soil0", Float64; default = 288.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "initial soil temperature")
        option!(s, "p_surface", Optional{Float64}; range = (0, Inf), open_lower = true, units = "Pa",
                doc = "pressure at the ground; \"none\": from the sounding (101325 Pa for the idealised profile)")
    end
    led = section("ledger"; doc = "The energy and water ledgers.") do s
        option!(s, "canopy_top", Float64; default = NaN, units = "m",
                doc = "top of the :canopy layer of the air-heat budget; nan: the tallest roof")
    end
    out = section("output") do s
        option!(s, "netcdf", Optional{String}; default = "alexanderplatz.nc",
                doc = "file name of the NetCDF-4 output in [output] path (the run folder's output/ under aeolus run, else examples/urban/output/); \"none\": no NetCDF")
        option!(s, "child_boundaries", Optional{Path}; input = false,
                doc = "with the online nest: write the child's boundary slabs at every parent step, the 3-D initial wind " *
                      "and the parent's mean profiles to this dynamic driver (CFG-g #362, `aeolus preprocess dynamic " *
                      "--from-parent`; the file of an offline child); \"none\": not written")
    end
    sp = slot_providers()
    phys = section("physics"; doc = "Which couplings run: a provider per exclusive slot, on/off per addon.") do s
        for sl in sort!(collect(keys(sp)))
            ps = sort!(copy(sp[sl]))
            option!(s, String(sl), Symbol; default = default_provider(sl), allowed = ps,
                    doc = "the $sl provider: " * join(ps, ", "))
        end
        for C in REGISTRY
            slot(C) === :addon || continue
            option!(s, String(coupling_name(C)), Bool; default = is_default(C),
                    doc = "the additive coupling $(coupling_name(C)) on or off")
        end
        option!(s, "options", OpenTable(_options_problems; reference = berlin_options_reference);
                doc = "[physics.options.<coupling>]: that coupling's options over its default_options " *
                      "(the keys are closed, a value has its default's type); " *
                      "[physics.options.nesting.parent] is the parent's schema")
    end
    return Section[run, grid, geo, pre, clock, cad, dyn, bnd, ini, led, out, phys]
end

"""
    berlin_physics(cfg::RunConfig) -> Dict{Symbol,Any}

The `[physics]` selection of `cfg` that differs from the registry
defaults (what `CaseConfig.physics` holds).
"""
function berlin_physics(cfg::RunConfig)
    p = cfg.physics
    out = Dict{Symbol,Any}()
    for (sl, _) in slot_providers()
        v = p[String(sl)]
        v === default_provider(sl) || (out[sl] = v)
    end
    for C in REGISTRY
        slot(C) === :addon || continue
        v = p[String(coupling_name(C))]
        v == is_default(C) || (out[coupling_name(C)] = v)
    end
    return out
end

function _berlin_checks(cfg::RunConfig)
    e = ConfigError[]
    # the declared incompatibilities, with their reasons
    sel = try
        selected_couplings((physics = berlin_physics(cfg), options = Dict{Symbol,Any}()))
    catch err
        err isa ErrorException || rethrow()
        push!(e, ConfigError("physics", err.msg))
        return e
    end
    # the cadences: the ones the enabled couplings need are set, and a
    # cadence a file sets is used by one of them
    need = Set{Symbol}(); may = Set{Symbol}()
    for C in sel
        union!(need, _required_cadences(C)); union!(may, declared_cadences(C))
    end
    for o in config_schema(cfg).sections[findfirst(s -> s.name == "cadence", config_schema(cfg).sections)].options
        k = Symbol(o.key); v = cfg.cadence[o.key]
        if v === nothing
            k in need && push!(e, ConfigError("cadence.$(o.key)", "the enabled couplings need the cadence :$k " *
                "(used by $(join(sort!([String(coupling_name(C)) for C in sel if k in _required_cadences(C)]), ", ")))"))
        elseif !(k in may) && config_origin(cfg, "cadence.$(o.key)") != "default"
            push!(e, ConfigError("cadence.$(o.key)", "no enabled coupling uses the cadence :$k " *
                "(it is used by $(join(sort!([String(coupling_name(C)) for C in REGISTRY if k in declared_cadences(C)]), ", ")))"))
        end
    end
    tt = cfg.dynamics.theta_top
    (tt isa Real || tt in (:initial_gradient, :noflux)) ||
        push!(e, ConfigError("dynamics.theta_top", "expected \"initial_gradient\", \"noflux\" or a gradient in K/m"))
    (cfg.checkpoint.every_steps == 0 && cfg.checkpoint.every_seconds == Inf) ||
        push!(e, ConfigError("checkpoint.every_steps",
            "the Berlin case has no checkpoint/restart yet (the coupling state is not checkpointed); keep [checkpoint] off"))
    return e
end

const _BERLIN_CHECKS_DOC = """
the selected couplings are compatible (the declared incompatibilities of `[physics]`, with their reasons)
every cadence an enabled coupling needs is set, and a cadence a file sets is used by an enabled coupling
`dynamics.theta_top` is `"initial_gradient"`, `"noflux"` or a gradient in K/m
`[checkpoint]` stays off: the Berlin case has no checkpoint/restart yet (#269)"""

Config.driver_checks(::BerlinDriver) = Any[_BERLIN_CHECKS_DOC => _berlin_checks]

const _BERLIN_SCHEMA = Ref{Any}(nothing)

"""
    berlin_schema() -> ConfigSchema

The schema of the `berlin` driver of this module (registered when
BerlinCase.jl is loaded).
"""
berlin_schema() = _BERLIN_SCHEMA[]

"The all-defaults `RunConfig` of the `berlin` driver (environment variables fill their defaults)."
default_run_config() = load_run_config(IOBuffer("[run]\ndriver = \"berlin\"\n"); schema = berlin_schema())

# ---------------------------------------------------------------------------
# RunConfig → CaseConfig
# ---------------------------------------------------------------------------

function CaseConfig(cfg::RunConfig; run_output::Union{Nothing,AbstractString} = nothing)
    config_schema(cfg).name == "berlin" ||
        throw(ArgumentError("CaseConfig: the config is for the driver \"$(config_schema(cfg).name)\", not \"berlin\""))
    r, g, geo, c, dyn, b, ini, out = cfg.run, cfg.grid, cfg.geometry, cfg.clock, cfg.dynamics,
                                      cfg.boundaries, cfg.initial, cfg.output
    pre = cfg.preprocess
    cad = Dict{Symbol,Float64}(Symbol(k) => cfg.cadence[k] for k in propertynames(cfg.cadence) .|> String
                               if cfg.cadence[k] !== nothing)
    cad[:output] = out.cadence
    probs, options = options_value(cfg.physics.options)
    isempty(probs) || throw(ConfigError("physics.options", join(probs, "; ")))
    dir = out.path !== nothing ? out.path : run_output !== nothing ? run_output : joinpath(URBAN_DIR, "output")
    d = c.date
    return CaseConfig(Symbol(r.name), r.scenario, g.size, g.extent, g.halo,
                      geo.static, geo.parent_static, geo.lat0, geo.lon0, geo.offset, geo.elevation,
                      geo.wall_subdivisions, pre.osm_path, pre.min_height, pre.park_lai, pre.landuse,
                      (year(d), month(d), day(d)), c.utc_offset, c.start_h, c.dt, c.nsteps, cad,
                      dyn.nu, dyn.theta0, dyn.g, dyn.f_coriolis, dyn.c_s, dyn.pr_t, dyn.sgs_stability,
                      dyn.sponge_alpha_max, dyn.sponge_frac, dyn.theta_top,
                      b.lateral, b.outflow, b.boundary_frame, b.theta_relaxation, b.analysis_buffer,
                      ini.sounding, ini.dynamic, ini.theta_surface, ini.theta_inversion_z, ini.theta_lapse, ini.u0,
                      ini.qv0, ini.qv_scale, ini.ts0, ini.ts_water_offset, ini.t_soil0, ini.p_surface,
                      berlin_physics(cfg), options,
                      cfg.ledger.canopy_top,
                      out.netcdf === nothing ? nothing : joinpath(dir, out.netcdf), out.child_boundaries,
                      CPUBackend(; serial = r.serial))
end

# ---------------------------------------------------------------------------
# the run-driver contract
# ---------------------------------------------------------------------------

"""
    BerlinRun

What the `berlin` driver builds: the `cfg` (`RunConfig`) and the built
[`Case`](@ref) (`case`, `nothing` until `run!` builds it into the run
folder).
"""
mutable struct BerlinRun
    cfg::RunConfig
    case::Any
end

# The case is built in run!: its NetCDF output goes into the run folder,
# which build does not know yet. A missing static driver is reported here,
# before a run folder exists, with the command that makes it.
function Config.build(::BerlinDriver, cfg::RunConfig)
    cc = CaseConfig(cfg)
    src = first(config_sources(cfg)["files"])
    for (f, key) in ((cc.static, "geometry.static"), (cc.parent_static, "geometry.parent_static"))
        (f === nothing || isfile(f)) && continue
        throw(ConfigError(key, "the static driver $f does not exist; generate it with\n" *
                               "  julia --project -m Aeolus preprocess static $src\n" *
                               "or set it to \"none\" to preprocess the source in memory"))
    end
    if cc.dynamic !== nothing && !isfile(cc.dynamic)
        how = nest_offline(cc) ? "--from-parent <the online config, with [output] child_boundaries = this file>" :
                                 "--from-sounding $src"
        throw(ConfigError("initial.dynamic", "the dynamic driver $(cc.dynamic) does not exist; generate it with\n" *
                                             "  julia --project -m Aeolus preprocess dynamic $how\n" *
                                             "or set it to \"none\" to read [initial] sounding in memory"))
    end
    return BerlinRun(cfg, nothing)
end

"""
    preprocess_static(::BerlinDriver, cfg; out = nothing, parent_out = nothing) -> Vector{String}

`aeolus preprocess static berlin_<s>.toml`: write the child's static
driver (`[geometry] static`, or `out`) from `[preprocess] osm_path`
([`berlin_static_inputs`](@ref)) and, for a nested case with a tiled
parent surface, the parent's (`[geometry] parent_static`, or
`parent_out`) from `[preprocess] landuse`
([`parent_landuse_source`](@ref)), each with its source hashed.
"""
function Config.preprocess_static(::BerlinDriver, cfg::RunConfig; out = nothing, parent_out = nothing)
    cc = CaseConfig(cfg)
    files = String[]
    child = out !== nothing ? abspath(out) : cc.static
    child === nothing && throw(ArgumentError("[geometry] static is \"none\": pass --out file.nc"))
    mesh = RegularGrid(; size = cc.size, extent = cc.extent, halo = cc.halo)
    write_static_driver(child, berlin_static_inputs(cc, mesh); source = cc.osm_path)
    push!(files, child)
    if case_nested(cc)
        pc = nest_parent_config(cc)
        ppath = parent_out !== nothing ? abspath(parent_out) : cc.parent_static
        if pc.surface === :tiles && ppath !== nothing
            pmesh = parent_mesh(pc)
            write_static_tiles(ppath, parent_static_tiles(parent_landuse_source(pc, pmesh), pc, pmesh);
                               source = cc.landuse)
            push!(files, ppath)
        end
    end
    return files
end

"""
    preprocess_dynamic(::BerlinDriver, cfg; from, out = nothing, child = nothing) -> Vector{String}

`aeolus preprocess dynamic --from-sounding berlin_<s>.toml`: write the
sounding driver (`[initial] dynamic`, or `out`) of the case from
`[initial] sounding` ([`case_dynamic_inputs`](@ref): the height-time
tables, times since the case start, and the ground pressure), the CSV
hashed; for an open, non-nested case (`boundaries.lateral = "open"`,
`physics.nesting = false`) also the steady boundary planes of its
[`case_boundary_data`](@ref) (one `time_bc` record). `--from-parent`: run the online nested case of `cfg` for its
`[clock] nsteps` with the child's boundary slabs recorded at every parent
step (`[output] child_boundaries`, or `out`): the parent-made driver an
offline child (`parent_driver = "offline"`, `[initial] dynamic` = this
file) replays. The Berlin parent is a section of the child's config
(`[physics.options.nesting.parent]`), so there is no separate `child`.
"""
function Config.preprocess_dynamic(::BerlinDriver, cfg::RunConfig; from::Symbol, out = nothing, child = nothing)
    cc = CaseConfig(cfg)
    mesh = RegularGrid(; size = cc.size, extent = cc.extent, halo = cc.halo)
    if from === :sounding
        path = out !== nothing ? abspath(out) : cc.dynamic
        path === nothing && throw(ArgumentError("[initial] dynamic is \"none\": pass --out file.nc"))
        cc.sounding === nothing && throw(ArgumentError("[initial] sounding is \"none\": there is no sounding to preprocess"))
        ox, oy = anchor_xy(cc, mesh)
        d = case_dynamic_inputs(cc)
        kw = (; source = cc.sounding, preprocessor = "aeolus preprocess dynamic --from-sounding",
              origin_lat = cc.lat0, origin_lon = cc.lon0, origin_x = ox, origin_y = oy)
        if !case_nested(cc) && cc.lateral === :open
            # the open case: also its steady planes, the ProfileBoundaryData of
            # case_boundary_data on the faces (from the CSV in memory: the file
            # being written is not read)
            cc0 = deepcopy(cc); cc0.dynamic = nothing
            st = only(open_boundary_states(case_boundary_conditions(cc0, mesh).bcs))
            Aeolus.BoundaryConditions.boundary_values!(st.planes, st.data, 0.0)
            record_boundary!(NestBoundaryRecorder(path, st.planes, mesh; inputs = d, kw...), 0.0, NaN,
                             plane_snapshot(st.planes, mesh))
        else
            write_dynamic_driver(path, d; dx = mesh.dx, dy = mesh.dy, kw...)
        end
        return [path]
    end
    from === :parent || throw(ArgumentError("preprocess_dynamic: from = :$from; use :sounding or :parent"))
    child === nothing || throw(ArgumentError("the Berlin nest's parent is the [physics.options.nesting.parent] " *
                                             "section of the child's config: give the child config alone, without --child"))
    case_nested(cc) || throw(ArgumentError("--from-parent: the case is not nested (physics.nesting = false)"))
    nest_offline(cc) && throw(ArgumentError("--from-parent: the case replays its parent (parent_driver = \"offline\"); " *
                                            "give the online config"))
    path = out !== nothing ? abspath(out) : cc.child_boundaries
    path === nothing && throw(ArgumentError("[output] child_boundaries is \"none\": pass --out file.nc"))
    cc.child_boundaries = path
    cc.output_path = nothing
    run_case!(build_case(cc))
    return [path]
end

function Aeolus.Simulations.run!(::BerlinDriver, r::BerlinRun, rf::RunFolder, ck::CheckpointSchedule;
                                 callback = nothing)
    rf.session == 1 || error("the Berlin case cannot resume a run (no checkpoint/restart yet)")
    cc = CaseConfig(r.cfg; run_output = output_dir(rf))
    case = build_case(cc)
    r.case = case
    t_end = case.sim.t + cc.nsteps * cc.dt
    # the case's diagnostics: the core's, the frame's, the ledgers, the stage
    # timing and every coupling's (core/diagnostics.jl, #338); explicit rows,
    # one per case step (a case step may be several core steps; the step
    # record sums them, #336)
    with_monitor(case.sim; run = rf, t_end = t_end, cadence = r.cfg.logging.diagnostics_cadence,
                 diagnostics = case_diagnostics(case), profiles = case_profiles(case),
                 attach = false) do m
        run_case!(case; monitor = m, callback = callback === nothing ? nothing : (c, n) -> callback(r, n))
        check_blowup!(m.guard, case.sim; step = case.clock.n, force = true)
    end
    case.clock.blowup && error("the Berlin case blew up at step $(case.clock.n)")
    return r
end

"""
    case_state(case) -> NamedTuple

The prognostic state of a built Berlin case, interior cells as
`Array{Float64}`: the child's `u, v, w, p, θ`, every tracer, the surface
temperature `Ts`, and, with the Aeolus nest parent, the parent's
`parent_u, …, parent_θ` (and `parent_qv`). What the bitwise round trip
compares ([`state_fields`](@ref Aeolus.Config.state_fields)).
"""
function case_state(case)
    s = case.sim
    out = Pair{Symbol,Array{Float64}}[]
    for f in (:u, :v, :w, :p, :θ)
        push!(out, f => Array{Float64}(interior_view(getproperty(s, f))))
    end
    for t in case.tracers.list
        push!(out, t.name => Array{Float64}(interior_view(t.field)))
    end
    push!(out, :Ts => Array{Float64}(case.bus.Ts))
    c = find_coupling(case, :nesting)
    if c !== nothing && c.parent !== nothing
        ps = c.parent.sim
        for f in (:u, :v, :w, :p, :θ)
            push!(out, Symbol("parent_", f) => Array{Float64}(interior_view(getproperty(ps, f))))
        end
        c.parent.pc.moisture && push!(out, :parent_qv => Array{Float64}(interior_view(ps.qv)))
    end
    return (; out...)
end

Config.state_fields(r::BerlinRun) = case_state(r.case)

register_driver!(BerlinDriver())
_BERLIN_SCHEMA[] = Config.driver_schema("berlin")
