# The urban site on the model (ONE-e, #454; formerly the `urban` run
# driver, CFG-c #358, EX-a #438): its sections of the model's configuration
# schema (`[urban]` and one per urban component), the site's options
# build_case consumes (`site_options`), and the build of an urban site
# under `aeolus run` (`build_model`):
#
#   julia --project -m Aeolus run examples/urban/berlin_summer.toml [--set time.nsteps=…]
#
# One source of truth for the defaults: the schema holds them.
# `site_options(default_run_config())` and `ParentConfig(; kw...)` start from them, and
# the per-component options are typed and checked against the components'
# `default_options` (the key set is closed, the form of a value is the
# form of its default), so no second schema has to be kept in step with
# the component files.

using ..Aeolus.Config: Config, RunConfig, ConfigError, Section, Option, OpenTable,
                     Optional, OneOf, Path, section, option!, config_origin, config_schema, config_sources
using ..Aeolus.IO: RunFolder, CheckpointSchedule, with_monitor, check_blowup!, output_dir,
                 resume!, maybe_checkpoint!
using Dates: Date, year, month, day

# ---------------------------------------------------------------------------
# the site's options ↔ the model's options
# ---------------------------------------------------------------------------

"""
    SITE_FIELDS

Every field of [`site_options`](@ref) and the dotted keys of the
model's schema it is made of (`components` is the `enabled` key of every
urban component's section and their other keys). The options that make
the site's options are exactly the union of these keys, each used once
(`test/config_berlin.jl`).
"""
const SITE_FIELDS = (
    name = ("run.name",), scenario = ("urban.scenario",),
    size = ("grid.size",), extent = ("grid.extent",), halo = ("grid.halo",),
    static = ("domain.static_driver",), parent_static = ("urban.parent_static",),
    lat0 = ("urban.lat0",), lon0 = ("urban.lon0",), offset = ("urban.offset",),
    elevation = ("urban.elevation",), wall_subdivisions = ("urban.wall_subdivisions",),
    landuse = ("urban.landuse",),
    date = ("urban.date",), utc_offset = ("urban.utc_offset",), start_h = ("urban.start_h",),
    dt = ("time.dt",), nsteps = ("time.nsteps",),
    cadence = ("urban.radiation_period", "urban.soil_period", "urban.surface_period", "output.cadence"),
    ν = ("dynamics.nu",), θ₀ = ("dynamics.theta0",), g = ("dynamics.g",),
    f_coriolis = ("urban.f_coriolis",), C_s = ("urban.c_s",), Pr_t = ("urban.pr_t",),
    sgs_stability = ("urban.sgs_stability",), sponge_α_max = ("urban.sponge_alpha_max",),
    sponge_frac = ("urban.sponge_frac",), θ_top = ("urban.theta_top",),
    lateral = ("urban.lateral",), outflow = ("urban.outflow",),
    boundary_frame = ("urban.boundary_frame",), θ_relaxation = ("urban.theta_relaxation",),
    analysis_buffer = ("urban.analysis_buffer",),
    sounding = ("urban.sounding",), dynamic = ("domain.dynamic_driver",), θ_surface = ("urban.theta_surface",),
    θ_inversion_z = ("urban.theta_inversion_z",), θ_lapse = ("urban.theta_lapse",),
    u0 = ("urban.u0",), qv0 = ("urban.qv0",), qv_scale = ("urban.qv_scale",),
    Ts0 = ("urban.ts0",), Ts_water_offset = ("urban.ts_water_offset",),
    T_soil0 = ("urban.t_soil0",), p_surface = ("urban.p_surface",),
    components = ("*.enabled", "*.*"),
    canopy_top = ("urban.canopy_top",),
    child_boundaries = ("urban.child_boundaries",),
    backend = ("urban.serial",))

"""
    PARENT_FIELDS

Every [`ParentConfig`](@ref) field but `child` and its key in
`[nesting] parent` (the ASCII name of the field).
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
    f = get(get(OPTION_FORMS, component_name(C), Dict{Symbol,Function}()), key, nothing)
    f === nothing || return f(x, path)
    return option_value(default_options(C)[key], x, path)
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

The schema of [`ParentConfig`](@ref), `[nesting] parent`
(the defaults of `ParentConfig(; kw...)`). The keys are the ASCII names of
the fields ([`PARENT_FIELDS`](@ref)); `q_surface` / `e_surface`,
`theta_init` / `qv_init` and `u_init` take the tables of
[`surface_forcing_value`](@ref) and [`profile_value`](@ref).
"""
function parent_section()
    return section("parent"; doc = "The nest parent (ParentConfig, #259): [nesting] parent.") do s
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

`[nesting] parent`: every key checked against the parent
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
# the urban site's sections of the model's schema (ONE-e, #454)
# ---------------------------------------------------------------------------

# The Config option type of a coupling option with the default `d` (its
# TOML form: a scalar, an array of the default's length, an array of
# arrays, or a table checked by `option_value`).
_scalar_spec(x) = x isa Bool ? Bool : x isa Integer ? Int : x isa AbstractFloat ? Float64 :
                  x isa Symbol ? Symbol : x isa AbstractString ? String : nothing
function _option_spec(C, key::Symbol, d)
    name = component_name(C)
    (name, key) == (:nesting, :parent) &&
        return OpenTable(t -> _form_problems(parent_value, t, "parent");
                         reference = () -> (p = parent_schema_section(); Section[Section("", p.doc, p.options)]))
    (name, key) == (:nesting, :z_anterp) && return OneOf(Float64, Symbol)
    (name, key) == (:nesting, :parent_driver) && return Symbol
    (name, key) == (:surface_exchange, :z0m) && return OneOf(Symbol, OpenTable(t -> _form_problems(
                        (x, p) -> option_value(default_options(SurfaceExchangeMOST).z0m, x, p), t, "z0m")))
    (name, key) == (:sgs_mlsgs, :fn) && return Optional{Float64}
    d === nothing && return Optional{Float64}
    T = _scalar_spec(d)
    T === nothing || return T
    if d isa Tuple && !isempty(d)
        Ts = unique(map(_scalar_spec, d))
        length(Ts) == 1 && Ts[1] !== nothing && return NTuple{length(d),Ts[1]}
    end
    if d isa AbstractVector && !isempty(d) && d[1] isa Tuple
        return Vector{Vector{Float64}}
    end
    (d isa Union{NamedTuple,AbstractDict}) && return OpenTable(t -> _form_problems((x, p) -> option_value(d, x, p), t, String(key)))
    error("the option $key of the coupling $name has a default the schema cannot type: $(repr(d))")
end

# The schema default of a coupling option: its default's TOML form, typed as
# the option (`_option_spec`)
function _option_default(C, key::Symbol, d)
    name = component_name(C)
    (name, key) == (:nesting, :parent) && return Dict{String,Any}()
    d === nothing && return nothing
    d isa Symbol && return d
    d isa Tuple && return Tuple(x isa Symbol ? x : x isa Integer && !(x isa Bool) ? Int(x) : x for x in d)
    d isa AbstractVector && !isempty(d) && d[1] isa Tuple && return [Float64.(collect(x)) for x in d]
    d isa Union{NamedTuple,AbstractDict} && return toml_form(d)
    return d isa Integer && !(d isa Bool) ? Int(d) : d
end

"""
    option_key(k::Symbol) -> String

The TOML key of the urban component option `k` (a field of its
`default_options`): ASCII snake_case, the Greek letters spelled out (`ν`
→ `nu`, `θ` → `theta`, `τ` → `tau`, `σ` → `sigma`) and lower case
(`U_g` → `u_g`, `OH_max` → `oh_max`, `C_σ` → `c_sigma`).
"""
function option_key(k::Symbol)
    s = String(k)
    for (g, a) in ("ν" => "nu", "θ" => "theta", "τ" => "tau", "σ" => "sigma")
        s = replace(s, g => a)
    end
    return lowercase(s)
end

"""
    OPTION_UNITS

The units of the urban components' numeric options, `(component, key) =>
units` ("1" for a dimensionless one), as their docstrings give them; the
model's option reference shows them.
"""
const OPTION_UNITS = Dict{Tuple{Symbol,Symbol},String}(
    (:geostrophic, :U_g) => "m s^-1", (:geostrophic, :V_g) => "m s^-1", (:geostrophic, :τ) => "s",
    (:radiosity_sun, :aod380) => "1", (:radiosity_sun, :aod500) => "1", (:radiosity_sun, :precipitable_water) => "cm",
    (:radiosity_sun, :ozone) => "atm-cm", (:radiosity_sun, :albedo) => "1",
    (:column_radiation, :r_eff_liq) => "µm", (:column_radiation, :p_surf) => "Pa",
    (:cloud_shadow, :r_eff_liq) => "µm", (:cloud_shadow, :r_eff_ice) => "µm",
    (:sgs_amd, :C) => "1", (:sgs_vreman, :C_s) => "1", (:sgs_sigma, :C_σ) => "1",
    (:sgs_smagorinsky, :diapycnal) => "1", (:sgs_smagorinsky, :slope_max) => "1",
    (:sgs_dynamic, :diapycnal) => "1", (:sgs_dynamic, :slope_max) => "1", (:sgs_dynamic, :filter_ratio) => "1",
    (:sgs_dynamic, :cs_max) => "1", (:sgs_mlsgs, :diapycnal) => "1", (:sgs_mlsgs, :slope_max) => "1",
    (:sgs_mlsgs, :fn) => "1",
    (:sgs_tke, :e_min) => "m^2 s^-2", (:sgs_reynolds_stress, :e_min) => "m^2 s^-2",
    (:veg_drag, :cd) => "1", (:pollutant, :stacks) => "m", (:pollutant, :tau) => "s",
    (:inflow_turbulence, :u_min) => "m s^-1", (:inflow_turbulence, :statistics_interval) => "s",
    (:inflow_turbulence, :length_interval) => "s", (:inflow_turbulence, :length_scale) => "m",
    (:traffic_heat, :Q_urban) => "W m^-2", (:micro_kessler, :qv_max) => "kg kg^-1",
    (:aq_chem, :nox_street) => "kg m^-2 s^-1", (:aq_chem, :no2_fraction) => "1",
    (:aq_chem, :isoprene_park) => "kg m^-2 s^-1", (:aq_chem, :OH_max) => "cm^-3", (:aq_chem, :ν) => "1",
    (:aq_chem, :z0m) => "m",
    (:surface_exchange, :U_min) => "m s^-1", (:surface_exchange, :z0h_ratio) => "1",
    (:surface_exchange, :period) => "s", (:surface_exchange, :input_T_f) => "s",
    (:surface_exchange, :dynamic_C_d) => "1", (:surface_exchange, :z0m) => "m",
    (:sw_modifiers, :k_leaf) => "1", (:nesting, :z_anterp) => "m", (:nesting, :τ) => "s",
    (:timestep_adaptive, :C_advective) => "1", (:timestep_adaptive, :C_diffusive) => "1",
    (:timestep_adaptive, :C_buoyancy) => "1", (:timestep_adaptive, :C_coriolis) => "1",
    (:timestep_adaptive, :dt_min) => "s", (:timestep_adaptive, :dt_max) => "s",
    (:timestep_adaptive, :growth_max) => "1", (:timestep_adaptive, :divergence_tol) => "s^-1",
    (:timestep_adaptive, :max_velocity) => "m s^-1", (:timestep_imex, :tol) => "1",
    (:pressure_multigrid, :tol) => "1", (:pressure_mg_cg, :tol) => "1", (:pressure_density_weighted, :tol) => "1",
    (:turbulence_diag, :gamma_threshold) => "1", (:turbulence_diag, :relam_νt_ratio) => "1")

"""
    coupling_section(C) -> Section

The model's section of the urban coupling `C`, `[<component_name>]`:
`enabled` (default `false`: every component is off until its section
enables it) and one option per key of `default_options(C)`, typed by the
form of its default (the per-key forms of [`OPTION_FORMS`](@ref) for the
nest's `parent`, `z_anterp` and `parent_driver`, the surface exchange's
`z0m` and the ML closure's `fn`).
"""
function coupling_section(C)
    name = String(component_name(C))
    s = slot(C)
    doc = s === nothing ? "The urban component `$name` (`$(nameof(C))`)." :
                         "The urban component `$name` (`$(nameof(C))`): `$(provider(C))` of the urban slot `$s` (a site enables one component per urban slot)."
    return section(name; doc = doc) do sec
        option!(sec, "enabled", Bool; default = false, doc = "run the component")
        keys_ = [option_key(k) for k in keys(default_options(C))]
        allunique(keys_) || error("the options of $(nameof(C)) collide as TOML keys: $keys_")
        for (k, d) in pairs(default_options(C))
            T = _option_spec(C, k, d)
            kw = T === Symbol && (component_name(C), k) == (:nesting, :parent_driver) ?
                 (; allowed = [:aeolus, :profile, :offline]) :
                 T isa OneOf && (component_name(C), k) == (:nesting, :z_anterp) ? (; allowed = [:canopy]) :
                 T isa OneOf ? (; allowed = [:dynamic]) : (;)
            u = get(OPTION_UNITS, (component_name(C), k), "")
            option!(sec, option_key(k), T; default = _option_default(C, k, d), kw..., units = u,
                    doc = "see `?$(nameof(C))`" * ((component_name(C), k) == (:nesting, :parent) ?
                          ": the nest parent's options, [$name.parent] (`aeolus options` lists them)" : ""))
        end
    end
end

"""
    urban_site_section() -> Section

`[urban]`: the urban site of a domain (ONE-e, #454). `enabled = true`
makes the domain an urban site: its ground from the domain's
`static_driver`, its initial state from the sounding (the domain's
`dynamic_driver` when that is a sounding driver) or the idealised profile,
its dynamical core and its time stepping assembled by `Aeolus.Urban`
from these options and the enabled urban components
([`coupling_section`](@ref)). The keys are those of the former `urban`
run type's `[geometry]`, `[clock]`, `[cadence]`, `[dynamics]`,
`[boundaries]`, `[initial]`, `[budgets]` and `[output]` sections that the
model's own sections do not hold.
"""
function urban_site_section()
    return section("urban"; doc = "The urban site of the domain (Aeolus.Urban, ONE-e #454).") do s
        option!(s, "enabled", Bool; default = false, doc = "the domain is an urban site")
        option!(s, "scenario", Symbol; default = :summer,
                doc = "the scenario label (summer, winter, …): the output's comment; winter selects the subarctic-winter humidity of the column radiation")
        option!(s, "serial", Bool; default = false, doc = "CPUBackend(serial = true): the single-task reference")
        # the site
        option!(s, "parent_static", Optional{Path};
                doc = "the static driver (land-use tiles) of the nest parent; \"none\": from landuse in memory")
        option!(s, "landuse", Optional{Path};
                doc = "the nest parent's land-use table (`aeolusprep fetch-landuse` writes it); \"none\": no table " *
                      "(a tiled parent surface then needs parent_static)")
        option!(s, "lat0", Float64; default = 52.5220, range = (-90, 90), units = "degree",
                doc = "latitude of the OSM projection's anchor (and of the solar position)")
        option!(s, "lon0", Float64; default = 13.4100, range = (-180, 180), units = "degree",
                doc = "longitude of the anchor")
        option!(s, "offset", NTuple{2,Float64}; default = (0.0, 0.0), units = "m",
                doc = "the window centre relative to the anchor")
        option!(s, "elevation", Float64; default = 34.0, units = "m", doc = "height above sea level")
        option!(s, "wall_subdivisions", Int; default = 1, range = (1, 64),
                doc = "wall facets per cell face of the radiosity scene")
        # the clock
        option!(s, "date", Date; default = Date(2019, 6, 29),
                doc = "the local date (29 June 2019: a clear day of the heat wave, #247)")
        option!(s, "utc_offset", Float64; default = 2.0, range = (-14, 14), units = "h",
                doc = "local clock = UTC + utc_offset (2: CEST)")
        option!(s, "start_h", Float64; default = 6.0, range = (0, 24), units = "h", doc = "local hour at t = 0")
        # the components' periods
        option!(s, "radiation_period", Float64; default = 600.0, range = (0, Inf), units = "s",
                doc = "the period of the radiation components (column_radiation, radiosity_sun, cloud_shadow, buildings_citymean)")
        option!(s, "soil_period", Float64; default = 30.0, range = (0, Inf), units = "s",
                doc = "the period of the soil (the surfaces' step unless surface_period is set) and of the deposition (aq_chem)")
        option!(s, "surface_period", Optional{Float64}; range = (0, Inf), units = "s",
                doc = "the surfaces' step (surface_exchange); \"none\": soil_period")
        # the core
        option!(s, "f_coriolis", Float64; default = 1e-4, units = "s^-1", doc = "Coriolis parameter of the site's core")
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
        # the lateral boundaries
        option!(s, "lateral", Symbol; default = :periodic, allowed = [:periodic, :open],
                doc = "the four sides without the nest: periodic or open (fed by the geostrophic wind and the initial θ)")
        option!(s, "outflow", Symbol; default = :dirichlet, allowed = [:dirichlet, :convective, :orlanski],
                doc = "outflow condition of the open or nest faces (N-open-c, #260)")
        option!(s, "boundary_frame", Int; default = 3, range = (0, typemax(Int)),
                doc = "cells: buildings closer than this to an open face are removed")
        option!(s, "theta_relaxation", Bool; default = true,
                doc = "relax θ over the sponge layer toward the boundary data's profile")
        option!(s, "analysis_buffer", Float64; default = ANALYSIS_BUFFER_DEFAULT, range = (0, Inf), units = "m",
                doc = "columns closer than this to a lateral open face are left out of the diagnostics " *
                      "compared with observations (the inflow fetch, #268)")
        # the initial state
        option!(s, "sounding", Optional{Path};
                doc = "radiosonde CSV of the initial θ(z), qv(z) and surface pressure; \"none\": the idealised profile below " *
                      "(the domain's dynamic_driver, a sounding driver, replaces it)")
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
        # the budgets and the output
        option!(s, "canopy_top", Float64; default = NaN, units = "m",
                doc = "top of the :canopy layer of the air-heat budget; nan: the tallest roof")
        option!(s, "child_boundaries", Optional{Path}; input = false,
                doc = "with the online nest: write the child's boundary slabs at every parent step, the 3-D initial wind " *
                      "and the parent's mean profiles to this dynamic driver (CFG-g #362; the file of an offline child); " *
                      "\"none\": not written")
    end
end

"""
    urban_section_names() -> Vector{String}

The names of the urban site's per-domain sections of the model:
`"urban"` and one per urban component (`component_name`, registry order).
"""
urban_section_names() = vcat(["urban"], [String(component_name(C)) for C in URBAN_COMPONENTS])

"The urban site's sections of the model's schema: [`urban_site_section`](@ref) and every [`coupling_section`](@ref)."
urban_sections() = Section[urban_site_section(), (coupling_section(C) for C in URBAN_COMPONENTS)...]

"""
    urban_enabled(cfg) -> Bool

Whether the (resolved) model config `cfg` makes a domain an urban site
(`[urban] enabled`; any of its domains, or the config itself for a legacy
case, which has none).
"""
function urban_enabled(cfg)
    ds = Config.config_domains(cfg)
    return isempty(ds) ? cfg.urban.enabled : any(d -> d.urban.enabled, ds)
end

# A resolved option value in its TOML form (a symbol as a string, nothing as
# "none", tuples as arrays, tables with string keys), a Julia object that
# TOML cannot hold (set with `with`) kept as it is: what `option_value`
# reads.
_raw_form(x::Symbol) = String(x)
_raw_form(::Nothing) = "none"
_raw_form(x::Union{Tuple,AbstractVector}) = Any[_raw_form(e) for e in x]
_raw_form(x::AbstractDict) = Dict{String,Any}(string(k) => _raw_form(v) for (k, v) in x)
_raw_form(x) = x

# the Julia value of every option of component C's resolved section: its
# default_options with the section's values on top
function _section_options(C, sec)
    probs = String[]; out = Pair{Symbol,Any}[]
    d = default_options(C)
    name = String(component_name(C))
    for k in keys(d)
        raw = _raw_form(sec[option_key(k)])
        ok, v, why = coupling_option_value(C, k, raw, "$name.$k")
        ok || (push!(probs, why); continue)
        isequal(v, d[k]) || push!(out, k => v)
    end
    return probs, merge(d, (; out...))
end

# The urban components a resolved config enables (in the order of
# URBAN_COMPONENTS), and its problems: an urban slot without exactly one
# enabled component, two enabled components one of which declares the
# other incompatible.
function _urban_selection(cfg)
    probs = ConfigError[]
    on = Type[C for C in URBAN_COMPONENTS if cfg[String(component_name(C))].enabled]
    for sl in URBAN_SLOTS
        these = [C for C in on if slot(C) === sl]
        length(these) == 1 && continue
        names = [String(component_name(C)) for C in URBAN_COMPONENTS if slot(C) === sl]
        push!(probs, ConfigError(isempty(these) ? names[1] : String(component_name(these[2])),
            "the urban slot :$sl needs exactly one enabled component, of [" * join(names, "], [") * "]" *
            (isempty(these) ? "; none is enabled" : "; enabled: " * join(String.(component_name.(these)), ", "))))
    end
    for C in on, D in on
        D in incompatible(C) || continue
        why = incompatible_reason(C)
        push!(probs, ConfigError(String(component_name(D)),
            "[$(component_name(C))] is incompatible with [$(component_name(D))]" * (isempty(why) ? "" : ": " * why)))
    end
    return probs, on
end

"""
    urban_components(cfg) -> Dict{Symbol,Any}

The urban components a resolved model config enables, `component_name =>`
their options (the `components` of [`site_options`](@ref)); a problem of
the selection or of an option's form is a `ConfigError`.
"""
function urban_components(cfg)
    probs, on = _urban_selection(cfg)
    isempty(probs) || throw(ConfigError(reduce(vcat, (p.problems for p in probs))))
    po, out = String[], Dict{Symbol,Any}()
    for C in URBAN_COMPONENTS
        p, nt = _section_options(C, cfg[String(component_name(C))])
        append!(po, p)
        C in on && (out[component_name(C)] = nt)
    end
    isempty(po) || throw(ConfigError("urban", join(po, "; ")))
    return out
end

"""
    urban_checks(cfg) -> Vector{ConfigError}

The cross-field checks of an urban site (`[urban] enabled`): exactly one
enabled provider per urban slot; the selection's declared
incompatibilities, with their reasons; every component's options of the
right form; a fixed `[time] dt` and `[time] nsteps` (the time stepping is
the urban `timestep_*` provider's: with `timestep_adaptive` `nsteps × dt`
is the simulated time); a uniform grid, the Boussinesq `[dynamics]` core
(the site picks its core with the `core_*` providers) and no model
component; `[checkpoint]` only with the Boussinesq core.
"""
function urban_checks(cfg)
    e, on = _urban_selection(cfg)
    for C in URBAN_COMPONENTS
        po, _ = _section_options(C, cfg[String(component_name(C))])
        for x in po
            push!(e, ConfigError(first(split(x, ':')), x))
        end
    end
    t = cfg.time
    t.dt isa Real || push!(e, ConfigError("time.dt", "an urban site takes a fixed dt (its time stepping is the enabled timestep_* component's)"))
    (t.nsteps === nothing || t.duration !== nothing) &&
        push!(e, ConfigError("time.nsteps", "an urban site runs nsteps steps of dt (duration = \"none\")"))
    cfg.grid.stretching === :uniform || push!(e, ConfigError("grid.stretching", "an urban site needs a uniform grid"))
    cfg.dynamics.core === :boussinesq ||
        push!(e, ConfigError("dynamics.core", "an urban site picks its core with [core_boussinesq], [core_anelastic] or [core_pseudo_incompressible]"))
    cfg.dynamics.moisture && push!(e, ConfigError("dynamics.moisture", "an urban site carries its water as tracers (qv, ql, qr)"))
    tt = cfg.urban.theta_top
    (tt isa Real || tt in (:initial_gradient, :noflux)) ||
        push!(e, ConfigError("urban.theta_top", "expected \"initial_gradient\", \"noflux\" or a gradient in K/m"))
    ckon = !(cfg.checkpoint.every_steps == 0 && cfg.checkpoint.every_seconds == Inf)
    ic = findfirst(C -> slot(C) === :core, on)
    core = ic === nothing ? :boussinesq : provider(on[ic])
    (ckon && core !== :boussinesq) &&
        push!(e, ConfigError("checkpoint.every_steps",
            "checkpoints of an urban site need the Boussinesq core ([core_boussinesq]); " *
            "the :$core core has none yet: keep [checkpoint] off"))
    return e
end

const URBAN_CHECKS_DOC = """
an urban site (`[urban] enabled`) enables exactly one provider per urban slot, a compatible selection (the declared incompatibilities, with their reasons), and options of the right form
an urban site takes a fixed `[time] dt` with `nsteps` (duration = \"none\"), a uniform grid, `[dynamics] core = \"boussinesq\"` (its core is a `core_*` provider) and no moisture (its water is tracers)
`[urban] theta_top` is `\"initial_gradient\"`, `\"noflux\"` or a gradient in K/m; `[checkpoint]` needs the Boussinesq core"""

"""
    site_options(cfg::RunConfig) -> NamedTuple

The urban site of the model's config `cfg` (`[urban] enabled`; its one
domain): the grid of `[grid]`, the ground of the domain's
`static_driver`, the dynamic driver of its `dynamic_driver`, `ν`, `θ₀`,
`g` of `[dynamics]`, `dt` and `nsteps` of `[time]`, the output cadence
of `[output]`, every other field of `[urban]`, and the urban components
the sections enable with their options (`components`,
[`urban_components`](@ref)). Its fields are documented at
[`site_options`](@ref) (site/config.jl).
"""
function site_options(cfg::RunConfig)
    config_schema(cfg).name in (Config.MODEL, "urban_parent") ||
        throw(ArgumentError("site_options: the config is for \"$(config_schema(cfg).name)\", not the model"))
    u = cfg.urban
    components = urban_components(cfg)
    g, t, dyn, out, dom = cfg.grid, cfg.time, cfg.dynamics, cfg.output, cfg.domain
    t.dt isa Real || throw(ConfigError("time.dt", "an urban site takes a fixed dt"))
    cad = Dict{Symbol,Float64}(:radiation => u.radiation_period, :soil => u.soil_period, :output => out.cadence)
    u.surface_period === nothing || (cad[:surface] = u.surface_period)
    d = u.date
    f64(x) = Float64(x)
    return (name = Symbol(cfg.run.name), scenario = Symbol(u.scenario),
            size = NTuple{3,Int}(g.size), extent = NTuple{3,Float64}(g.extent), halo = NTuple{3,Int}(g.halo),
            static = _path(dom.static_driver), parent_static = _path(u.parent_static),
            lat0 = f64(u.lat0), lon0 = f64(u.lon0), offset = NTuple{2,Float64}(u.offset),
            elevation = f64(u.elevation), wall_subdivisions = Int(u.wall_subdivisions), landuse = _path(u.landuse),
            date = (year(d), month(d), day(d)), utc_offset = f64(u.utc_offset), start_h = f64(u.start_h),
            dt = f64(t.dt), nsteps = Int(something(t.nsteps, 0)), cadence = cad,
            ν = f64(dyn.nu), θ₀ = f64(dyn.theta0), g = f64(dyn.g), f_coriolis = f64(u.f_coriolis),
            C_s = f64(u.c_s), Pr_t = f64(u.pr_t), sgs_stability = Symbol(u.sgs_stability),
            sponge_α_max = f64(u.sponge_alpha_max), sponge_frac = f64(u.sponge_frac),
            θ_top = u.theta_top isa Real ? f64(u.theta_top) : Symbol(u.theta_top),
            lateral = Symbol(u.lateral), outflow = Symbol(u.outflow), boundary_frame = Int(u.boundary_frame),
            θ_relaxation = Bool(u.theta_relaxation), analysis_buffer = f64(u.analysis_buffer),
            sounding = _path(u.sounding), dynamic = _path(dom.dynamic_driver),
            θ_surface = f64(u.theta_surface), θ_inversion_z = f64(u.theta_inversion_z), θ_lapse = f64(u.theta_lapse),
            u0 = f64(u.u0), qv0 = f64(u.qv0), qv_scale = f64(u.qv_scale), Ts0 = f64(u.ts0),
            Ts_water_offset = f64(u.ts_water_offset), T_soil0 = f64(u.t_soil0),
            p_surface = u.p_surface === nothing ? nothing : f64(u.p_surface),
            components = components, canopy_top = f64(u.canopy_top), child_boundaries = _path(u.child_boundaries),
            backend = CPUBackend(; serial = u.serial))
end
_path(::Nothing) = nothing
_path(p) = String(p)

"""
    default_run_config() -> RunConfig

The model's config of an urban site with every option at its default,
the components of [`URBAN_DEFAULTS`](@ref) enabled, and the former `urban` run type's grid (64³ on 1 km × 1 km ×
400 m, halo 3), `ν = 2`, `θ₀ = 293 K`, 43 200 steps of 2 s and output
every 1800 s: the generic site (`site_options(default_run_config())`,
the child of `ParentConfig(; kw...)`).
"""
function default_run_config()
    io = IOBuffer()
    println(io, "[run]\nname = \"urban\"\n[time]\ndt = 2.0\nduration = \"none\"\nnsteps = 43200\n[output]\ncadence = 1800.0\n",
            "[grid]\nsize = [64, 64, 64]\nextent = [1000.0, 1000.0, 400.0]\nhalo = [3, 3, 3]\n",
            "[dynamics]\nnu = 2.0\ntheta0 = 293.0\n[urban]\nenabled = true")
    for C in URBAN_DEFAULTS
        println(io, "[", component_name(C), "]\nenabled = true")
    end
    seekstart(io)
    return load_run_config(io)
end

"""
    urban_input_problems(cfg) -> NamedTuple

Refuse, before a run folder exists, an urban site whose static or
dynamic driver file is missing, naming the command that makes it;
otherwise its [`site_options`](@ref).
"""
function urban_input_problems(cfg::RunConfig)
    cc = site_options(cfg)
    src = first(config_sources(cfg)["files"])
    for (f, key) in ((cc.static, "domain.static_driver"), (cc.parent_static, "urban.parent_static"))
        (f === nothing || isfile(f)) && continue
        throw(ConfigError(key, key == "domain.static_driver" ?
                               "the static driver $f does not exist; make it with AeolusPrep.jl:\n" *
                               "  julia --project -m AeolusPrep static $src --source <file.osm | file.geojson>" :
                               "the static driver $f does not exist; make it with AeolusPrep.jl:\n" *
                               "  julia --project -m AeolusPrep static $src --source <file.osm | file.geojson>\n" *
                               "or set it to \"none\" to read [urban] landuse in memory"))
    end
    if cc.dynamic !== nothing && !isfile(cc.dynamic)
        how = nest_offline(cc) ?
              "run the online config with [urban] child_boundaries = this file:\n  julia --project -m Aeolus run <the online config>" :
              "make it with AeolusPrep.jl:\n  julia --project -m AeolusPrep dynamic --from-sounding $src"
        throw(ConfigError("domain.dynamic_driver", "the dynamic driver $(cc.dynamic) does not exist; $how\n" *
                                                   "or set it to \"none\" to read [urban] sounding in memory"))
    end
    return cc
end

"""
    build_model(cfg::RunConfig) -> Model

The urban site of the model's config `cfg` (`[urban] enabled`) built as
a `Model` of one domain, what `aeolus run` builds and runs with the
model's run loop (ONE-k, #474): the site ([`build_case`](@ref)) is the
domain's first component and its `:timestepper` component takes the
core step. A missing driver is refused first
([`urban_input_problems`](@ref)), and `[output] variables` must name
variables of the domain's catalogue (the core's and the site's,
[`Components.outputs`](@ref)).
"""
function build_model(cfg::RunConfig)
    case = build_case(urban_input_problems(cfg))
    have = Set(v.name for v in Components.domain_catalogue(case.domain))
    bad = [n for n in cfg.output.variables if !(n in have)]
    isempty(bad) ||
        throw(ConfigError("output.variables", "$(join(bad, ", ")): not in the urban site's output catalogue " *
                                              "(it has: $(join(sort!(collect(have)), ", ")))"))
    return case.model
end

"Height (m) below which [`run_results`](@ref) reports the fluid's mean θ."
const RESULTS_LAYER = 100.0

"""
    BUDGET_TOLERANCES

What [`run_results`](@ref) calls a closed ledger: the surface-energy
closure of every class below `surface` W m⁻² (mean over the run;
measured ≲ 3e-10 on the smoke runs), the water residual of the air below
`water` kg m⁻² (measured ≲ 3e-16), the air-heat closure below `air_heat`
W m⁻² of ground (mean over the run), with nothing booked as
`(:core, :unattributed)` or `(:output, :between_steps)` (#253).
"""
const BUDGET_TOLERANCES = (surface = 1e-6, water = 1e-12, air_heat = 1e-10)

"""
    mean_θ_below(case, zmax) -> Float64

The mean θ (K) of the fluid cells of the child whose centres are below
`zmax` (m); `NaN` when there are none.
"""
function mean_θ_below(case, zmax)
    m, θ = case.mesh, interior_view(case.sim.θ)
    mask = case.geom.sm.mask
    s = 0.0; n = 0
    for k in 1:m.Nz
        zc = (k - 0.5) * m.dz
        zc < zmax || break
        for j in 1:m.Ny, i in 1:m.Nx
            mask[m.Hx+i, m.Hy+j, m.Hz+k] && continue
            s += θ[i, j, k]; n += 1
        end
    end
    return n == 0 ? NaN : s / n
end

"""
    run_results(case; θ_start = NaN, wall = NaN, wall_first_step = NaN, simulated = NaN, steps = 0)
        -> Dict{String,Any}

The checks and summary numbers of a finished run, what the `urban`
driver writes to the run folder's `output/results.toml` (EX-a, #438;
they were the by-hand checks of the Berlin example):

* `finite`: every interior value of u, v, w, θ and every tracer, and the
  surface temperature, is finite;
* `[budgets]`: the surface-energy budget per class (`[budgets.surface.<class>]`:
  Rn, H, LE, G, dS and the closure, W m⁻², mean over the run), its
  largest |closure|, the water residual of the air (kg m⁻²), the
  air-heat closure (W m⁻² of ground), whether anything changed θ
  unattributed, and `closed` against [`BUDGET_TOLERANCES`](@ref);
* `[air_heat]`: every term of the air-heat ledger of the whole column,
  `"<stage>.<name>"` in W m⁻² of ground (mean over the run; the nest's
  or open faces' boundary terms among them), and the storage;
* `[open_boundary_heat]` (a nested or open case): the heat carried in
  through the four sides and the top (W m⁻² of ground, mean over the run);
* `theta_mean_below_100m_start` / `_end` (K, [`mean_θ_below`](@ref));
* `[timing]`: steps, simulated seconds, wall seconds of the run, of its
  first step (compilation), wall seconds per simulated hour without the
  first step, and, nested, the wall seconds of the parent's steps.

Reading the state only: computing it changes nothing.
"""
function run_results(case; θ_start = NaN, wall = NaN, wall_first_step = NaN, simulated = NaN, steps = 0)
    l, sim, bus = case.budgets, case.sim, case.bus
    tol = BUDGET_TOLERANCES
    finite = all(f -> all(isfinite, interior_view(f)), (sim.u, sim.v, sim.w, sim.θ)) &&
             all(t -> all(isfinite, interior_view(t.field)), case.tracers.list) && all(isfinite, bus.Ts)
    rows = surface_budget(l)
    surf = Dict{String,Any}(r.name => Dict{String,Any}("Rn" => r.Rn, "H" => r.H, "LE" => r.LE, "G" => r.G,
                                                       "dS" => r.dS, "closure" => r.closure, "columns" => r.ncol)
                            for r in rows)
    smax = isempty(rows) ? 0.0 : maximum(abs(r.closure) for r in rows)
    w = water_budget(l)
    hb = heat_budget(l)
    heat = hb.elapsed > 0 ? hb.closure / hb.elapsed : 0.0
    unattributed = haskey(hb.terms, (:core, :unattributed)) || haskey(hb.terms, (:output, :between_steps))
    closed = smax < tol.surface && abs(w.residual) < tol.water && abs(heat) <= tol.air_heat && !unattributed
    budgets = Dict{String,Any}("surface" => surf, "surface_max_abs_closure" => smax,
                               "water_residual" => w.residual, "air_heat_closure" => heat,
                               "unattributed" => unattributed, "closed" => closed,
                               "tolerances" => Dict{String,Any}(string(k) => v for (k, v) in pairs(tol)))
    air = Dict{String,Any}("$(k[1]).$(k[2])" => (hb.elapsed > 0 ? v / hb.elapsed : 0.0) for (k, v) in hb.terms)
    air["storage"] = hb.elapsed > 0 ? hb.storage / hb.elapsed : 0.0
    sim_h = simulated / 3600
    rest = steps > 1 && sim_h > 0 ? (wall - wall_first_step) / (sim_h * (steps - 1) / steps) : NaN
    timing = Dict{String,Any}("steps" => steps, "simulated_s" => simulated, "wall_s" => wall,
                              "wall_first_step_s" => wall_first_step, "wall_s_per_simulated_h" => rest)
    nest = find_coupling(case, :nesting)
    nest === nothing || nest.parent === nothing || (timing["wall_parent_s"] = nest.wall_parent)
    out = Dict{String,Any}("finite" => finite, "budgets" => budgets, "air_heat" => air,
                           "theta_mean_below_100m_start" => θ_start,
                           "theta_mean_below_100m_end" => mean_θ_below(case, RESULTS_LAYER),
                           "timing" => timing)
    ob = case.open
    if ob !== nothing && hb.elapsed > 0
        out["open_boundary_heat"] = Dict{String,Any}("lateral" => ob.lateral / hb.elapsed, "top" => ob.top / hb.elapsed)
    end
    return out
end

"""
    case_state(case) -> NamedTuple

The prognostic state of a built urban case, interior cells as
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

