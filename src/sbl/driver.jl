# =========================================================================
# The `sbl` run type (CFG-d, #359; in src since EX-b, #439): the
# stable-boundary-layer cases configured from TOML (`[run] driver = "sbl"`).
#
#   julia --project -m Aeolus run examples/sbl/configs/gabls1.toml --set grid.size=[32,32,32]
#
# One driver, four kinds (`[case] kind`):
#
#   homogeneous               the periodic SBL LES of `sbl_case`
#                             (case.jl): GABLS1-4, CASES-99
#   stratified_channel        `stratified_channel_run` (García-Villalba &
#                             del Álamo 2011)
#   relaminarisation_channel  `relam_spinup` + `relam_run` (P59-e)
#   collapse_channel          `collapse_run` on the flux-cooled half
#                             channel (P58-e)
#
# The functions of case.jl are the implementation; this file only turns a
# RunConfig into their arguments (no physics here) and writes the run's
# output (statistics.toml, series.csv, towers.csv). Case constants live in
# examples/sbl/configs/*.toml; the case tables are the CSVs of
# examples/sbl/data/ (`[case] data`, hashed under [inputs] of run.toml),
# read by tables.jl. Named kinds (perturbations, surface-temperature
# formulas, profile shapes) are added in code, never in TOML.
# =========================================================================

"""
    SBLDriver()

The `sbl` run driver: `[run] driver = "sbl"`. `aeolus options sbl` prints
its sections; shipped configs are in `examples/sbl/configs/`.
"""
struct SBLDriver <: AbstractRunDriver end

"The shipped `sbl` configs: `examples/sbl/configs/` of the package."
const SBL_CONFIG_DIR = normpath(joinpath(@__DIR__, "..", "..", "examples", "sbl", "configs"))

Config.driver_name(::SBLDriver) = "sbl"
Config.config_dir(::SBLDriver) = SBL_CONFIG_DIR

"The `[case] kind`s of the `sbl` run type."
const SBL_KINDS = [:homogeneous, :stratified_channel, :relaminarisation_channel, :collapse_channel]

# ---------------------------------------------------------------------------
# table validators: forcing quantities, profiles, perturbations, surface
# ---------------------------------------------------------------------------

_sbl_isnum(x) = x isa Real && !(x isa Bool)
_sbl_isnumvec(x) = x isa AbstractVector && !isempty(x) && all(_sbl_isnum, x)
_sbl_isint(x) = x isa Integer && !(x isa Bool)

"Problems of a `{table, column}` reference to a column of the case table (`what` names it)."
function _sbl_colref_problems(r, what)
    r isa AbstractDict || return ["$what: expected a table {table, column}"]
    p = String[]
    get(r, "table", nothing) isa AbstractString || push!(p, "$what.table: expected the name of a case table")
    c = get(r, "column", nothing)
    (_sbl_isint(c) && c >= 1) || push!(p, "$what.column: expected a column number ≥ 1")
    for k in keys(r)
        k in ("table", "column", "row") || push!(p, "$what.$k: unknown key (keys: table, column, row)")
    end
    haskey(r, "row") && !(_sbl_isint(r["row"]) && r["row"] >= 1) && push!(p, "$what.row: expected a row number ≥ 1")
    return p
end

function _sbl_unknown_keys(t, want)
    return ["$k: unknown key (keys: $(join(want, ", ")))" for k in sort!(collect(keys(t))) if !(k in want)]
end

"""
A forcing quantity table (`kind = "table"`): a `ForcingTable` on the
heights `z`. The times are `t = […]` (inline) or `time = {table, column}`
(a column of the case table, `[case] data`); without either it is a
height profile. `levels` has one entry per height, each `{value = x}`
(constant in time), `{values = […]}` (one value per inline time) or
`{column = j}` (column `j` of the time's case table).
"""
function _sbl_quantity_problems(q)
    p = String[]
    get(q, "kind", nothing) == "table" || return ["kind: expected \"table\" (or give a number)"]
    append!(p, _sbl_unknown_keys(q, ["kind", "z", "t", "time", "levels"]))
    _sbl_isnumvec(get(q, "z", nothing)) || (push!(p, "z: expected a non-empty array of heights"); return p)
    issorted(q["z"]; lt = <=) || push!(p, "z: expected strictly increasing heights")
    haskey(q, "t") && haskey(q, "time") && push!(p, "give t or time, not both")
    nt = 0
    if haskey(q, "t")
        _sbl_isnumvec(q["t"]) ? (nt = length(q["t"])) : push!(p, "t: expected a non-empty array of times")
    elseif haskey(q, "time")
        append!(p, _sbl_colref_problems(q["time"], "time"))
    end
    lv = get(q, "levels", nothing)
    (lv isa AbstractVector && length(lv) == length(q["z"])) ||
        (push!(p, "levels: expected one entry per height ($(length(q["z"])))"); return p)
    for (i, l) in enumerate(lv)
        w = "levels[$i]"
        (l isa AbstractDict && length(l) == 1) ||
            (push!(p, "$w: expected one of {value = x}, {values = […]}, {column = j}"); continue)
        k = only(keys(l))
        if k == "value"
            _sbl_isnum(l[k]) || push!(p, "$w.value: expected a number")
        elseif k == "values"
            haskey(q, "t") || push!(p, "$w.values: needs inline times t")
            (_sbl_isnumvec(l[k]) && length(l[k]) == nt) || push!(p, "$w.values: expected $nt numbers (one per time)")
        elseif k == "column"
            haskey(q, "time") || push!(p, "$w.column: needs time = {table, column}")
            (_sbl_isint(l[k]) && l[k] >= 1) || push!(p, "$w.column: expected a column number ≥ 1")
        else
            push!(p, "$w.$k: unknown key (one of value, values, column)")
        end
    end
    (!haskey(q, "t") && !haskey(q, "time") && any(l -> l isa AbstractDict && !haskey(l, "value"), lv)) &&
        push!(p, "levels: a height profile (no times) takes {value = x} levels only")
    return p
end

const _SBL_TEMPERATURE_KEYS = Dict("constant" => ["value"],
                               "cooling_rate" => ["initial", "rate_per_hour"],
                               "series" => ["table", "time_column", "column", "p_sfc", "exponent"],
                               "gabls2" => ["p_sfc", "exponent", "sample_every", "until"])

function _sbl_temperature_problems(t)
    kind = get(t, "kind", nothing)
    haskey(_SBL_TEMPERATURE_KEYS, kind) ||
        return ["kind: expected one of " * join(repr.(sort!(collect(keys(_SBL_TEMPERATURE_KEYS)))), ", ") *
                ", got $(repr(kind))"]
    want = _SBL_TEMPERATURE_KEYS[kind]
    p = _sbl_unknown_keys(t, vcat("kind", want))
    opt = kind == "series" ? ("p_sfc", "exponent") : ()
    for k in want
        k in opt && !haskey(t, k) && continue
        haskey(t, k) || (push!(p, "$k: required for kind = \"$kind\""); continue)
        if k in ("time_column", "column")
            (_sbl_isint(t[k]) && t[k] >= 1) || push!(p, "$k: expected a column number ≥ 1")
        elseif k == "table"
            t[k] isa AbstractString || push!(p, "table: expected the name of a case table")
        else
            _sbl_isnum(t[k]) && isfinite(t[k]) || push!(p, "$k: expected a finite number")
        end
    end
    kind == "series" && haskey(t, "p_sfc") != haskey(t, "exponent") &&
        push!(p, "p_sfc and exponent go together (values are T, converted to θ = T (1e5/p_sfc)^exponent)")
    return p
end

const _SBL_STABILITY_KEYS = Dict("businger_dyer_log_linear" => ["beta", "beta_m", "beta_h", "gamma_m", "gamma_h"],
                             "businger_dyer_beljaars_holtslag" => ["gamma_m", "gamma_h", "a", "b", "c", "d"])

function _sbl_stability_problems(t)
    fam = get(t, "family", nothing)
    haskey(_SBL_STABILITY_KEYS, fam) ||
        return ["family: expected one of " * join(repr.(sort!(collect(keys(_SBL_STABILITY_KEYS)))), ", ") *
                ", got $(repr(fam))"]
    p = _sbl_unknown_keys(t, vcat("family", _SBL_STABILITY_KEYS[fam]))
    for k in _SBL_STABILITY_KEYS[fam]
        haskey(t, k) && !(_sbl_isnum(t[k]) && isfinite(t[k])) && push!(p, "$k: expected a finite number")
    end
    return p
end

const _SBL_PROFILE_KEYS = Dict("constant" => ["value"],
                           "mixed_layer_lapse" => ["value", "z_i", "lapse"],
                           "table" => ["table", "column", "z_column", "below"])

function _sbl_profile_problems(t)
    isempty(t) && return String[]
    kind = get(t, "kind", nothing)
    haskey(_SBL_PROFILE_KEYS, kind) ||
        return ["kind: expected one of " * join(repr.(sort!(collect(keys(_SBL_PROFILE_KEYS)))), ", ") *
                ", got $(repr(kind))"]
    p = _sbl_unknown_keys(t, vcat("kind", _SBL_PROFILE_KEYS[kind]))
    if kind == "table"
        get(t, "table", nothing) isa AbstractString || push!(p, "table: expected the name of a case table")
        for k in ("column", "z_column")
            (k == "z_column" && !haskey(t, k)) && continue
            (_sbl_isint(get(t, k, nothing)) && t[k] >= 1) || push!(p, "$k: expected a column number ≥ 1")
        end
        b = get(t, "below", nothing)
        if b !== nothing
            bk = b isa AbstractDict ? get(b, "kind", nothing) : nothing
            if bk == "log"
                append!(p, _sbl_unknown_keys(b, ["kind", "z0"]))
                _sbl_isnum(get(b, "z0", nothing)) && b["z0"] > 0 || push!(p, "below.z0: expected a roughness length > 0")
            elseif bk == "linear"
                append!(p, _sbl_unknown_keys(b, ["kind", "z", "value"]))
                _sbl_isnum(get(b, "z", nothing)) || push!(p, "below.z: expected the height of the surface value")
                v = get(b, "value", nothing)
                _sbl_isnum(v) || append!(p, _sbl_colref_problems(v, "below.value"))
            else
                push!(p, "below: expected {kind = \"log\", z0} or {kind = \"linear\", z, value}" *
                         " (default: the lowest tabulated value held)")
            end
        end
    else
        for k in _SBL_PROFILE_KEYS[kind]
            _sbl_isnum(get(t, k, nothing)) && isfinite(t[k]) || push!(p, "$k: expected a finite number")
        end
    end
    return p
end

const _SBL_PERTURBATION_KEYS = Dict("none" => String[],
                                "theta_uniform" => ["amplitude", "depth"],
                                "uniform" => ["amplitude", "depth"],
                                "gaussian" => ["variance", "depth", "taper"])

function _sbl_perturbation_problems(t)
    kind = get(t, "kind", nothing)
    haskey(_SBL_PERTURBATION_KEYS, kind) ||
        return ["kind: expected one of " * join(repr.(sort!(collect(keys(_SBL_PERTURBATION_KEYS)))), ", ") *
                ", got $(repr(kind))"]
    p = _sbl_unknown_keys(t, vcat("kind", _SBL_PERTURBATION_KEYS[kind]))
    for k in _SBL_PERTURBATION_KEYS[kind]
        haskey(t, k) || (push!(p, "$k: required for kind = \"$kind\""); continue)
        v = t[k]
        if k == "taper"
            (v isa AbstractVector && length(v) == 3 && all(x -> x in ("linear", "none"), v)) ||
                push!(p, "taper: expected 3 of \"linear\", \"none\" (u, v, θ)")
        elseif k == "variance" || (k == "amplitude" && kind == "uniform")
            (_sbl_isnumvec(v) && length(v) == 3 && all(>=(0), v)) || push!(p, "$k: expected 3 numbers ≥ 0 (u, v, θ)")
        else
            _sbl_isnum(v) && v >= 0 || push!(p, "$k: expected a number ≥ 0")
        end
    end
    return p
end

const _SBL_LSF_NAME = Dict("u" => :u, "v" => :v, "theta" => :θ, "qv" => :qv)

function _sbl_fields_quantities_problems(t, want; extra = String[])
    p = _sbl_unknown_keys(t, vcat(want, extra))
    for k in want
        haskey(t, k) || continue
        q = t[k]
        _sbl_isnum(q) && continue
        q isa AbstractDict ? append!(p, ["$k.$x" for x in _sbl_quantity_problems(q)]) :
                             push!(p, "$k: expected a number or a table")
    end
    return p
end

function _sbl_advective_problems(t)
    isempty(t) && return String[]
    p = _sbl_fields_quantities_problems(t, ["u", "v", "theta"])
    any(k -> haskey(t, k), ("u", "v", "theta")) || push!(p, "give at least one of u, v, theta")
    return p
end

function _sbl_subsidence_problems(t)
    isempty(t) && return String[]
    p = _sbl_fields_quantities_problems(t, ["w"]; extra = ["fields", "form"])
    haskey(t, "w") || push!(p, "w: required (the subsidence velocity, m s⁻¹)")
    f = get(t, "fields", ["theta", "qv"])
    (f isa AbstractVector && !isempty(f) && all(x -> haskey(_SBL_LSF_NAME, x), f)) ||
        push!(p, "fields: expected a non-empty array of \"u\", \"v\", \"theta\", \"qv\"")
    get(t, "form", "mean_gradient") in ("mean_gradient", "local") ||
        push!(p, "form: expected \"mean_gradient\" or \"local\"")
    return p
end

function _sbl_nudging_problems(t)
    isempty(t) && return String[]
    p = _sbl_fields_quantities_problems(t, ["u", "v", "theta", "qv"]; extra = ["tau"])
    (_sbl_isnum(get(t, "tau", nothing)) && t["tau"] > 0) || push!(p, "tau: expected the relaxation time > 0 (s)")
    any(k -> haskey(t, k), ("u", "v", "theta", "qv")) || push!(p, "give at least one target (u, v, theta, qv)")
    return p
end

_sbl_forcing_quantity() = OneOf(Float64, OpenTable(_sbl_quantity_problems))

# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------

function Config.driver_sections(::SBLDriver)
    case = section("case"; doc = "Which SBL case and its case table.") do s
        option!(s, "kind", Symbol; default = :homogeneous, allowed = SBL_KINDS,
                doc = "homogeneous (sbl_case: GABLS1-4, CASES-99), stratified_channel (stratified_channel_run), relaminarisation_channel (relam_spinup + relam_run), collapse_channel (collapse_run)")
        option!(s, "data", Optional{Path};
                doc = "the case table (a CSV of examples/sbl/data/ with a table name in its first column) that table references read; \"none\": no table")
        option!(s, "dynamic", Optional{Path};
                doc = "homogeneous: a dynamic driver (CFG-g #362; AeolusPrep.jl, `aeolusprep dynamic --from-sounding`) whose " *
                      "tables replace the config's: geostrophic wind, advective tendencies, subsidence, nudging and " *
                      "the surface θ series, each when the driver holds it; \"none\": the config's own")
        option!(s, "richardson", Float64; units = "1", default = 0.0, range = (0, Inf),
                doc = "stratified_channel: the friction Richardson number Ri_τ (0: θ passive, g = 0)")
        option!(s, "re_tau", Float64; units = "1", default = 180.0, range = (0, Inf), open_lower = true,
                doc = "relaminarisation_channel: the friction Reynolds number Re_τ")
        option!(s, "h_over_l", Float64; units = "1", default = 0.0, range = (0, Inf),
                doc = "relaminarisation_channel: the stability h/L of the cooling (0: neutral)")
        option!(s, "heat_flux_fraction", Float64; units = "1", default = 0.0, range = (0, Inf),
                doc = "collapse_channel: the surface cooling in units of the maximum sustainable heat flux at Δz/2 (0: neutral)")
        option!(s, "z0", Float64; units = "H", default = 1e-4, range = (0, Inf), open_lower = true,
                doc = "collapse_channel: the roughness length z0/H")
        option!(s, "theta_ref", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "collapse_channel: the reference temperature of the flux wall law and the MSHF")
        option!(s, "spinup_time", Float64; units = "δ/u_τ, h/u_τ or H/u*", default = 10.0, range = (0, Inf),
                doc = "channels: the neutral spin-up (δ/u_τ, h/u_τ, H/u*) the run starts from; stratified_channel: 0 starts from noise")
    end
    grid = section("grid"; doc = "The mesh.") do s
        option!(s, "size", NTuple{3,Int}; default = (32, 32, 32), range = (1, 65536),
                doc = "cells in x, y, z")
        option!(s, "extent", Optional{NTuple{3,Float64}}; range = (0, Inf), open_lower = true,
                units = "m",
                doc = "homogeneous: the domain size; \"none\": the GABLS1 400 m cube. The channels' domains are their builders' (stratified_channel 2πδ × πδ × 2δ, relaminarisation_channel 2πh × πh × h, collapse_channel 2πH × 2πH × H)")
    end
    dyn = section("dynamics"; doc = "The Boussinesq core (homogeneous).") do s
        option!(s, "theta0", Float64; default = 263.5, range = (0, Inf), open_lower = true, units = "K",
                doc = "reference potential temperature of the buoyancy")
    end
    forcing = section("forcing"; doc = "Large-scale forcing (homogeneous): LargeScaleForcing objects.") do s
        option!(s, "u_g", _sbl_forcing_quantity(); default = 8.0, units = "m s^-1",
                doc = "geostrophic wind U_g: a number or a table (a ForcingTable over z and/or t)")
        option!(s, "v_g", _sbl_forcing_quantity(); default = 0.0, units = "m s^-1",
                doc = "geostrophic wind V_g: a number or a table")
        option!(s, "f", Optional{Float64}; units = "s^-1", doc = "Coriolis parameter; give f or latitude")
        option!(s, "latitude", Optional{Float64}; range = (-90, 90), units = "degree",
                doc = "latitude: f = 2Ω sin(latitude)")
        option!(s, "subsidence", OpenTable(_sbl_subsidence_problems);
                doc = "{w = <quantity>, fields = [\"theta\", …], form = \"mean_gradient\" | \"local\"}: Subsidence; {}: none")
        option!(s, "advective_tendencies", OpenTable(_sbl_advective_problems);
                doc = "{u, v, theta = <quantity>} (per second): AdvectiveTendencies; {}: none")
        option!(s, "nudging", OpenTable(_sbl_nudging_problems);
                doc = "{tau = s, u, v, theta, qv = <target quantity>}: ProfileNudging; {}: none")
    end
    surface = section("surface"; doc = "The surface layer (homogeneous): SurfaceExchange MOST.") do s
        option!(s, "z0m", Float64; default = 0.1, range = (0, Inf), open_lower = true, units = "m",
                doc = "momentum roughness length")
        option!(s, "z0h", Optional{Float64}; range = (0, Inf), open_lower = true, units = "m",
                doc = "scalar roughness length (the height of a prescribed θ); \"none\": z0m")
        option!(s, "temperature", OpenTable(_sbl_temperature_problems);
                default = Dict{String,Any}("kind" => "cooling_rate", "initial" => 265.0, "rate_per_hour" => 0.25),
                doc = "the surface θ: {kind = \"constant\", value}, {kind = \"cooling_rate\", initial, rate_per_hour}, " *
                      "{kind = \"series\", table, time_column, column[, p_sfc, exponent]} (a case-table series; with p_sfc the values are T, θ = T (1e5/p_sfc)^exponent), " *
                      "{kind = \"gabls2\", p_sfc, exponent, sample_every, until} (the GABLS2 formula sampled from t_start)")
        option!(s, "stability", OpenTable(_sbl_stability_problems);
                default = Dict{String,Any}("family" => "businger_dyer_log_linear", "beta_m" => 4.8, "beta_h" => 7.8),
                doc = "the stability functions: {family = \"businger_dyer_log_linear\", beta | beta_m, beta_h, gamma_m, gamma_h} or {family = \"businger_dyer_beljaars_holtslag\", gamma_m, gamma_h, a, b, c, d}")
        option!(s, "rho_cp", Float64; default = 1306.5, range = (0, Inf), open_lower = true,
                units = "J m^-3 K^-1", doc = "ρ c_p of the W m⁻² surface heat flux (diagnostic only)")
    end
    initial = section("initial"; doc = "The initial state (homogeneous).") do s
        option!(s, "theta", OpenTable(_sbl_profile_problems);
                default = Dict{String,Any}("kind" => "mixed_layer_lapse", "value" => 265.0, "z_i" => 100.0, "lapse" => 0.01),
                doc = "θ(z): {kind = \"constant\", value}, {kind = \"mixed_layer_lapse\", value, z_i, lapse} (value up to and at z_i), " *
                      "{kind = \"table\", table, column[, z_column = 1, below]} (interpolated, the ends held; below the lowest height " *
                      "{kind = \"log\", z0} or {kind = \"linear\", z, value}: from the surface value at z)")
        option!(s, "u", OpenTable(_sbl_profile_problems); doc = "u(z), as theta; {}: U_g(z, t_start)")
        option!(s, "v", OpenTable(_sbl_profile_problems); doc = "v(z), as theta; {}: V_g(z, t_start)")
        option!(s, "perturbation", OpenTable(_sbl_perturbation_problems);
                default = Dict{String,Any}("kind" => "theta_uniform", "amplitude" => 0.1, "depth" => 50.0),
                doc = "{kind = \"theta_uniform\", amplitude, depth}: θ ± amplitude below depth (GABLS1); " *
                      "{kind = \"uniform\", amplitude = [u, v, θ], depth}: u, v, θ uniform ± amplitude below depth, velocity projected; " *
                      "{kind = \"gaussian\", variance = [u, v, θ], depth, taper = [\"linear\" | \"none\", …]}: normal with the variances at and below depth, " *
                      "tapered by (1 - z/depth) where \"linear\"; {kind = \"none\"}")
        option!(s, "t_start", Float64; default = 0.0, units = "s",
                doc = "the initial model time (the forcing tables are read at the model time)")
    end
    sgs = section("sgs"; doc = "The SGS closure.") do s
        option!(s, "closure", Symbol; default = :smagorinsky,
                allowed = [:smagorinsky, :vreman, :sigma, :amd, :dynamic_smagorinsky],
                doc = "homogeneous: smagorinsky, vreman, sigma or amd; stratified_channel: dynamic_smagorinsky; relaminarisation_channel: sigma; collapse_channel: smagorinsky")
        option!(s, "c_s", Optional{Float64}; units = "1", range = (0, Inf), open_lower = true,
                doc = "the closure's coefficient; \"none\": smagorinsky 0.1, vreman 0.17, sigma 1.35, amd 0.3 (dynamic_smagorinsky has none)")
        option!(s, "stability", Symbol; default = :brown94,
                allowed = [:brown94, :lilly, :long_tail, :short_tail, :none],
                doc = "the Richardson-number correction (StabilityCorrected): brown94 (Brown et al. 1994), lilly, long_tail, short_tail, none; relaminarisation_channel and collapse_channel: brown94; stratified_channel: brown94 or none")
        option!(s, "stress", Symbol; default = :symmetric, allowed = [:symmetric, :laplacian],
                doc = "the momentum SGS divergence: symmetric (the stress form) or laplacian; collapse_channel: laplacian")
    end
    sponge = section("sponge"; doc = "The top Rayleigh sponge (homogeneous).") do s
        option!(s, "z", Float64; default = 300.0, range = (0, Inf), units = "m", doc = "where it starts")
        option!(s, "alpha", Float64; default = 0.01, range = (0, Inf), units = "s^-1", doc = "its top rate")
        option!(s, "target", Symbol; default = :geostrophic, allowed = [:geostrophic, :mean],
                doc = "geostrophic: relax toward U_g(z, t); mean: damp only the departures from the plane mean")
    end
    stepper = section("stepper"; doc = "Time stepping (SSP-RK3).") do s
        option!(s, "cfl", Optional{Float64}; units = "1", range = (0, Inf), open_lower = true,
                doc = "advective Courant number of the adaptive step; \"none\": the kind's (homogeneous 1.2, channels 0.5, the only value they take)")
        option!(s, "end_time", Optional{Float64}; units = "s (channels: δ/u_τ)",
                doc = "the model time to run to (homogeneous: from t_start; channels: the run after the spin-up)")
        option!(s, "nsteps", Optional{Int}; range = (1, typemax(Int)),
                doc = "homogeneous: steps to run instead of end_time")
    end
    avg = section("averaging"; doc = "Statistics.") do s
        option!(s, "from", Float64; default = Inf, units = "s (channels: δ/u_τ)",
                doc = "model time from which the profiles are averaged (homogeneous, stratified_channel, collapse_channel)")
        option!(s, "sample_every", Int; default = 10, range = (1, typemax(Int)),
                doc = "homogeneous: sample every n-th step")
        option!(s, "interval", Float64; units = "h/u_τ or H/u*", default = 0.5, range = (0, Inf), open_lower = true,
                doc = "relaminarisation_channel, collapse_channel: model time between samples")
        option!(s, "series_interval", Float64; default = 0.0, range = (0, Inf), units = "s",
                doc = "homogeneous: model time between the rows of output/series.csv; 0: no series")
    end
    towers = section("towers"; doc = "Virtual towers (homogeneous; CASES-99 hockey stick, Sun et al. 2012).") do s
        option!(s, "below", Float64; default = 0.0, range = (0, Inf), units = "m",
                doc = "towers at every cell centre below this height; 0: no towers")
        option!(s, "window", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "averaging segment")
        option!(s, "hours", NTuple{2,Float64}; units = "h", default = (18.0, 6.0), range = (0, 24),
                doc = "local hours (from, to) in which the towers record; (0, 24): always")
    end
    return Section[case, grid, dyn, forcing, surface, initial, sgs, sponge, stepper, avg, towers]
end

# ---------------------------------------------------------------------------
# cross-field checks
# ---------------------------------------------------------------------------

# the kinds that use each option; an option set away from its default for a
# kind that does not use it is an error (a silently ignored key is a bug)
const _SBL_HOMOG = (:homogeneous,)
const _SBL_KIND_USE = Dict(
    "case.data" => _SBL_HOMOG, "case.dynamic" => _SBL_HOMOG, "case.richardson" => (:stratified_channel,),
    "case.re_tau" => (:relaminarisation_channel,), "case.h_over_l" => (:relaminarisation_channel,),
    "case.heat_flux_fraction" => (:collapse_channel,), "case.z0" => (:collapse_channel,),
    "case.theta_ref" => (:collapse_channel,),
    "case.spinup_time" => (:stratified_channel, :relaminarisation_channel, :collapse_channel),
    "grid.extent" => _SBL_HOMOG,
    "stepper.nsteps" => _SBL_HOMOG, "stepper.cfl" => _SBL_HOMOG,
    "averaging.from" => (:homogeneous, :stratified_channel, :collapse_channel),
    "averaging.sample_every" => _SBL_HOMOG, "averaging.series_interval" => _SBL_HOMOG,
    "averaging.interval" => (:relaminarisation_channel, :collapse_channel),
    "sgs.c_s" => (:homogeneous, :collapse_channel))
_sbl_kind_use(key) = get(_SBL_KIND_USE, key,
                     split(key, '.')[1] in ("dynamics", "forcing", "surface", "initial", "sponge", "towers") ?
                     _SBL_HOMOG : Tuple(SBL_KINDS))

function _sbl_default(cfg, sec, key)
    o = only(o for s in config_schema(cfg).sections if s.name == sec for o in s.options if o.key == key)
    d = o.default isa Function ? o.default() : o.default
    return d
end

_sbl_same(a, b) = isequal(a, b) || (a isa AbstractDict && b isa AbstractDict && Dict(a) == Dict(b)) ||
              (a isa Tuple && b isa AbstractVector && collect(a) == b) || (a isa Symbol && b isa AbstractString && String(a) == b)

"""
The case-table cells a value refers to, `[(table, column, row, key), …]`:
a `{table, column[, row]}` reference, a profile or temperature table
(`column`, `z_column`, `time_column`), and the `levels` columns of a
quantity table (columns of its `time` table); nested tables searched too.
"""
function _sbl_table_refs(x, where, out = Tuple{String,Int,Int,String}[])
    x isa AbstractDict || return out
    if get(x, "table", nothing) isa AbstractString
        for k in ("column", "z_column", "time_column")
            haskey(x, k) && _sbl_isint(x[k]) && push!(out, (x["table"], x[k], get(x, "row", 1), where))
        end
    end
    tm = get(x, "time", nothing)
    if tm isa AbstractDict && get(tm, "table", nothing) isa AbstractString
        for (i, l) in enumerate(get(x, "levels", Any[]))
            l isa AbstractDict && _sbl_isint(get(l, "column", nothing)) &&
                push!(out, (tm["table"], l["column"], 1, "$where.levels[$i]"))
        end
    end
    for (k, v) in x
        v isa AbstractDict && _sbl_table_refs(v, "$where.$k", out)
    end
    return out
end

function _sbl_checks(cfg)
    e = ConfigError[]
    kind = cfg.case.kind
    for s in config_schema(cfg).sections, o in s.options
        s.name in ("run", "fftw", "logging", "checkpoint", "output") && continue
        key = "$(s.name).$(o.key)"
        kind in _sbl_kind_use(key) && continue
        _sbl_same(cfg[key], _sbl_default(cfg, s.name, o.key)) && continue
        push!(e, ConfigError(key, "not used by case.kind = \"$kind\" (used by " *
                                  join(map(string, _sbl_kind_use(key)), ", ") * "); remove it"))
    end
    st = cfg.stepper; sg = cfg.sgs
    ck_on = cfg.checkpoint.every_steps > 0 || isfinite(cfg.checkpoint.every_seconds)
    if kind === :homogeneous
        (st.end_time === nothing) == (st.nsteps === nothing) &&
            push!(e, ConfigError("stepper.end_time", "give exactly one of stepper.end_time and stepper.nsteps"))
        (cfg.forcing.f === nothing) == (cfg.forcing.latitude === nothing) &&
            push!(e, ConfigError("forcing.f", "give exactly one of forcing.f and forcing.latitude"))
        sg.closure in (:smagorinsky, :vreman, :sigma, :amd) ||
            push!(e, ConfigError("sgs.closure", "homogeneous takes smagorinsky, vreman, sigma or amd, not $(sg.closure)"))
    else
        st.end_time === nothing && push!(e, ConfigError("stepper.end_time", "a channel runs to stepper.end_time"))
        want = kind === :stratified_channel ? :dynamic_smagorinsky :
               kind === :relaminarisation_channel ? :sigma : :smagorinsky
        sg.closure === want ||
            push!(e, ConfigError("sgs.closure", "$kind runs $want (the builder's closure), not $(sg.closure)"))
        (kind === :stratified_channel ? sg.stability in (:brown94, :none) : sg.stability === :brown94) ||
            push!(e, ConfigError("sgs.stability", "$kind takes " *
                                 (kind === :stratified_channel ? "brown94 or none" : "brown94") * ", not $(sg.stability)"))
        kind === :collapse_channel && sg.stress !== :laplacian &&
            push!(e, ConfigError("sgs.stress", "collapse_channel (half_channel_case) uses the laplacian form"))
        kind in (:stratified_channel, :collapse_channel) && ck_on &&
            push!(e, ConfigError("checkpoint.every_steps", "checkpoints need kind = homogeneous or relaminarisation_channel"))
    end
    # the case table and every reference into it
    d = cfg.case.data
    refs = Tuple{String,Int,Int,String}[]
    for (k, v) in (("forcing.u_g", cfg.forcing.u_g), ("forcing.v_g", cfg.forcing.v_g),
                   ("forcing.subsidence", cfg.forcing.subsidence),
                   ("forcing.advective_tendencies", cfg.forcing.advective_tendencies),
                   ("forcing.nudging", cfg.forcing.nudging), ("surface.temperature", cfg.surface.temperature),
                   ("initial.theta", cfg.initial.theta), ("initial.u", cfg.initial.u), ("initial.v", cfg.initial.v))
        _sbl_table_refs(v, k, refs)
    end
    dd = cfg.case.dynamic
    dd === nothing || isfile(dd) ||
        push!(e, ConfigError("case.dynamic", "the dynamic driver $dd does not exist; make it with AeolusPrep.jl:\n" *
                             "  julia --project -m AeolusPrep dynamic --from-sounding <this config> --out $dd"))
    if d !== nothing && !isfile(d)
        push!(e, ConfigError("case.data", "the case table $d does not exist"))
    elseif !isempty(refs) && d === nothing
        push!(e, ConfigError(refs[1][4], "refers to the case table \"$(refs[1][1])\" but case.data is \"none\""))
    elseif d !== nothing
        T = sbl_case_tables(d)
        for (name, col, row, where) in refs
            if !haskey(T, name)
                push!(e, ConfigError(where, "no table \"$name\" in $d (tables: $(join(sort!(collect(keys(T))), ", ")))"))
            elseif col > size(T[name], 2) || row > size(T[name], 1)
                push!(e, ConfigError(where, "table \"$name\" of $d has $(size(T[name], 1)) rows and $(size(T[name], 2)) columns"))
            end
        end
    end
    return e
end

const _SBL_CHECKS_DOC = """
a key of a section that `case.kind` does not use must keep its default (it would be ignored)
homogeneous: exactly one of `stepper.end_time` and `stepper.nsteps`, exactly one of `forcing.f` and `forcing.latitude`; `sgs.closure` is `smagorinsky`, `vreman`, `sigma` or `amd`
channels: run to `stepper.end_time` with the builder's closure (`stratified_channel`: `dynamic_smagorinsky` with `brown94` or no stability correction; `relaminarisation_channel`: `sigma`; `collapse_channel`: `smagorinsky`, laplacian stress) and `sgs.stability = "brown94"`
checkpoints only for `homogeneous` and `relaminarisation_channel`
`case.dynamic`, when given, exists (the message prints the `aeolusprep dynamic` command that makes it)
`case.data` exists, and every table and column a forcing, surface or initial option refers to is in it"""

Config.driver_checks(::SBLDriver) = Any[_SBL_CHECKS_DOC => _sbl_checks]

# ---------------------------------------------------------------------------
# config -> Julia objects
# ---------------------------------------------------------------------------

_sbl_tables(cfg) = cfg.case.data === nothing ? nothing : sbl_case_tables(cfg.case.data)

"The `LargeScaleForcing` quantity of a forcing value: a number, or a `ForcingTable`."
function sbl_quantity(q, T)
    _sbl_isnum(q) && return Float64(q)
    z = Float64.(q["z"])
    t = haskey(q, "t") ? Float64.(q["t"]) :
        haskey(q, "time") ? T[q["time"]["table"]][:, q["time"]["column"]] : nothing
    t === nothing && return ForcingTable(z, Float64[Float64(l["value"]) for l in q["levels"]])
    level(l) = haskey(l, "value") ? fill(Float64(l["value"]), length(t)) :
               haskey(l, "values") ? Float64.(l["values"]) : T[q["time"]["table"]][:, l["column"]]
    return ForcingTable(z, t, permutedims(reduce(hcat, [level(l) for l in q["levels"]])))
end

_sbl_cell(v, T) = _sbl_isnum(v) ? Float64(v) : T[v["table"]][get(v, "row", 1), v["column"]]

"A profile `z -> value` of `[initial]` (`nothing` for `{}`)."
function sbl_profile(p, T)
    isempty(p) && return nothing
    kind = p["kind"]
    if kind == "constant"
        v = Float64(p["value"]); return z -> v
    elseif kind == "mixed_layer_lapse"
        v, zi, Γ = Float64(p["value"]), Float64(p["z_i"]), Float64(p["lapse"])
        return z -> z <= zi ? v : v + Γ * (z - zi)              # sbl_case's GABLS1 profile
    end
    tb = T[p["table"]]
    zs = tb[:, get(p, "z_column", 1)]; ys = tb[:, p["column"]]
    b = get(p, "below", nothing)
    b === nothing && return z -> _sbl_interp(zs, ys, z)
    z1, y1 = zs[1], ys[1]
    if b["kind"] == "log"                    # log from zero at z0 to the lowest tabulated value
        z0 = Float64(b["z0"])
        return z -> z >= z1 ? _sbl_interp(zs, ys, z) : y1 * (log(max(z, z0) / z0) / log(z1 / z0))
    end
    zsfc = Float64(b["z"]); ysfc = _sbl_cell(b["value"], T)       # linear from the surface value at zsfc
    return z -> z >= z1 ? _sbl_interp(zs, ys, z) : ysfc + (y1 - ysfc) * (max(z, zsfc) - zsfc) / (z1 - zsfc)
end

"The `sbl_case` keywords of `[initial] perturbation`."
function sbl_perturbation(p)
    kind = p["kind"]
    kind == "none" && return (; perturbation = 0.0, z_pert = 0.0)
    kind == "theta_uniform" && return (; perturbation = Float64(p["amplitude"]), z_pert = Float64(p["depth"]))
    depth = Float64(p["depth"])
    if kind == "uniform"
        au, av, aθ = Float64.(p["amplitude"])
        return (; perturb = (rng, x, y, z) -> z < depth ?
                    (au * (2rand(rng) - 1), av * (2rand(rng) - 1), aθ * (2rand(rng) - 1)) : (0.0, 0.0, 0.0))
    end
    vu, vv, vθ = Float64.(p["variance"])
    lin = p["taper"] .== "linear"
    tap(k, z) = lin[k] ? 1 - z / depth : 1.0
    return (; perturb = (rng, x, y, z) -> z <= depth ?
                (sqrt(vu) * tap(1, z) * randn(rng), sqrt(vv) * tap(2, z) * randn(rng),
                 sqrt(vθ) * tap(3, z) * randn(rng)) : (0.0, 0.0, 0.0))
end

"The `sbl_case` keywords of `[surface] temperature` (`t_start` for the GABLS2 sampling)."
function sbl_surface_temperature(t, T, t_start)
    kind = t["kind"]
    kind == "constant" && return (; T_s0 = Float64(t["value"]), cooling = 0.0)
    kind == "cooling_rate" && return (; T_s0 = Float64(t["initial"]), cooling = Float64(t["rate_per_hour"]) / 3600)
    if kind == "series"
        tb = T[t["table"]]
        v = tb[:, t["column"]]
        haskey(t, "p_sfc") && (v = v .* (1e5 / Float64(t["p_sfc"]))^Float64(t["exponent"]))
        return (; T_s = SurfaceTemperatureSeries(tb[:, t["time_column"]], v))
    end
    ps, κ = Float64(t["p_sfc"]), Float64(t["exponent"])       # gabls2: the case formula, sampled
    ts = collect(t_start:Float64(t["sample_every"]):Float64(t["until"]))
    θs = [(gabls2_surface_temperature_C(τ / 3600) + 273.15) * (1e5 / ps)^κ for τ in ts]
    return (; T_s = SurfaceTemperatureSeries(ts, θs))
end

"The `SurfaceExchange` stability functions of `[surface] stability`."
function sbl_stability(t)
    g(k, d) = Float64(get(t, k, d))
    if t["family"] == "businger_dyer_log_linear"
        β = g("beta", 5.0)
        return BusingerDyerLogLinear(; γ_m = g("gamma_m", 16.0), γ_h = g("gamma_h", 16.0), β = β,
                                     β_m = g("beta_m", β), β_h = g("beta_h", β))
    end
    return BusingerDyerBeljaarsHoltslag(; γ_m = g("gamma_m", 16.0), γ_h = g("gamma_h", 16.0), a = g("a", 1.0),
                                        b = g("b", 2 / 3), c = g("c", 5.0), d = g("d", 0.35))
end

"The base eddy-viscosity closure of `[sgs]` (homogeneous)."
function sbl_base_closure(s)
    c = s.c_s
    s.closure === :smagorinsky && return StaticSmagorinsky(something(c, 0.1))
    s.closure === :vreman && return VremanClosure(something(c, 0.17))
    s.closure === :sigma && return SigmaModel(something(c, 1.35))
    return AMDClosure(; C = something(c, 0.3))
end

"The forcings after the geostrophic pressure gradient: advective tendencies, subsidence, nudging."
function sbl_extra_forcing(f, T)
    out = Any[]
    a = f.advective_tendencies
    if !isempty(a)
        q(k) = haskey(a, k) ? sbl_quantity(a[k], T) : nothing
        push!(out, AdvectiveTendencies(; u = q("u"), v = q("v"), θ = q("theta")))
    end
    s = f.subsidence
    if !isempty(s)
        fs = Tuple(_SBL_LSF_NAME[x] for x in get(s, "fields", ["theta", "qv"]))
        push!(out, Subsidence(sbl_quantity(s["w"], T); fields = fs, form = Symbol(get(s, "form", "mean_gradient"))))
    end
    n = f.nudging
    if !isempty(n)
        ks = [k for k in ("u", "v", "theta", "qv") if haskey(n, k)]
        targets = NamedTuple{Tuple(_SBL_LSF_NAME[k] for k in ks)}(Tuple(sbl_quantity(n[k], T) for k in ks))
        push!(out, ProfileNudging(targets, Float64(n["tau"])))
    end
    return Tuple(out)
end

"""
    sbl_forcing(cfg::RunConfig) -> (; U_g, V_g, extra, surface)

The large-scale forcing of a homogeneous `sbl` config as `LargeScaleForcing`
objects: the geostrophic wind `U_g`, `V_g`, the `extra` forcings
([`sbl_extra_forcing`](@ref)) and the `sbl_case` keywords of the surface
temperature ([`sbl_surface_temperature`](@ref)), from the config's own
quantities and case table, or, for those a dynamic driver `case.dynamic`
holds (CFG-g #362), from the driver (its times shifted from "seconds
since init_time" to the model time, `initial.t_start` at the start): the
geostrophic wind, the extra forcings (all of them, when it holds any) and
the surface θ series.
"""
function sbl_forcing(cfg::RunConfig)
    T = _sbl_tables(cfg)
    f = cfg.forcing; t0 = cfg.initial.t_start
    Ug = sbl_quantity(f.u_g, T); Vg = sbl_quantity(f.v_g, T)
    extra = sbl_extra_forcing(f, T)
    sfc = sbl_surface_temperature(cfg.surface.temperature, T, t0)
    cfg.case.dynamic === nothing && return (; U_g = Ug, V_g = Vg, extra, surface = sfc)
    d = read_dynamic_driver(cfg.case.dynamic; planes = false)
    F = dynamic_forcing(d; t0 = t0)
    F.U_g === nothing || (Ug = F.U_g)
    F.V_g === nothing || (Vg = F.V_g)
    isempty(F.extra) || (extra = F.extra)
    s = d.surface
    s === nothing || s.temperature === nothing ||
        (sfc = (; T_s = SurfaceTemperatureSeries(s.t .+ t0, s.temperature)))
    return (; U_g = Ug, V_g = Vg, extra, surface = sfc)
end

"""
    sbl_dynamic_inputs(cfg::RunConfig) -> DynamicInputs

The contents of the case-table driver of a homogeneous SBL run (GABLS2-4,
CASES-99): the config's table quantities ([`sbl_forcing`](@ref) without a
driver: the geostrophic wind, advective tendencies, subsidence, nudging,
the surface θ series), times since the run start `initial.t_start`.
Numbers and formulas stay in the config; `init_time` is empty (the
configs carry no calendar date). What a run reading `case.dynamic` gets
back, table for table; AeolusPrep.jl writes it (`aeolusprep dynamic
--from-sounding`, ONE-h #457). An `ArgumentError` for a channel case or a
config with no tables.
"""
function sbl_dynamic_inputs(cfg::RunConfig)
    cfg.case.kind === :homogeneous ||
        throw(ArgumentError("case.kind = $(cfg.case.kind) has no forcing tables (only homogeneous cases do)"))
    F = sbl_forcing(with_case_dynamic(cfg, nothing))
    sfc = haskey(F.surface, :T_s) ? (t = F.surface.T_s.t, temperature = F.surface.T_s.T_s, pressure = nothing) : nothing
    d = forcing_dynamic_inputs(; U_g = F.U_g, V_g = F.V_g, extra = F.extra, surface = sfc,
                               t_start = cfg.initial.t_start)
    isempty(d.tables) && d.surface === nothing &&
        throw(ArgumentError("the config has no forcing tables (only numbers and formulas)"))
    return d
end

"""
    write_sbl_driver(cfg::RunConfig, path; preprocessor) -> path

The case-table driver of the homogeneous SBL run `cfg` at `path`: the
tables of [`sbl_dynamic_inputs`](@ref), the case table `case.data` hashed
as the source, `forcing.latitude` as `origin_lat` (NaN when unset) and
`preprocessor` recorded. The one writer of a case-table driver:
AeolusPrep.jl's `aeolusprep dynamic --from-sounding` and the tests call it.
"""
function write_sbl_driver(cfg::RunConfig, path::AbstractString; preprocessor::AbstractString)
    d = sbl_dynamic_inputs(cfg)
    lat = cfg.forcing.latitude === nothing ? NaN : cfg.forcing.latitude
    mkpath(dirname(abspath(path)))
    return write_dynamic_driver(path, d; source = something(cfg.case.data, ""), origin_lat = lat,
                                preprocessor = preprocessor)
end

# `cfg` with case.dynamic replaced (sbl_dynamic_inputs reads the config's own tables)
with_case_dynamic(cfg::RunConfig, path) = path === cfg.case.dynamic ? cfg : with(cfg, "case.dynamic" => path)

"""
    sbl_case(cfg::RunConfig) -> SBLCase

The homogeneous SBL case of a resolved `sbl` config (`[case] kind =
"homogeneous"`): every section turned into the keywords of
[`sbl_case`](@ref).
"""
function sbl_case(cfg::RunConfig)
    T = _sbl_tables(cfg)
    f = cfg.forcing; sf = cfg.surface; ini = cfg.initial; sp = cfg.sponge
    fc = f.f === nothing ? sbl_coriolis(f.latitude) : f.f
    F = sbl_forcing(cfg)
    Ug = F.U_g; Vg = F.V_g
    θi = sbl_profile(ini.theta, T)
    kw = (; N = cfg.grid.size, extent = something(cfg.grid.extent, (400.0, 400.0, 400.0)),
          U_g = Ug, V_g = Vg, f = fc, z0m = sf.z0m, z0h = something(sf.z0h, sf.z0m), θ₀ = cfg.dynamics.theta0,
          base = sbl_base_closure(cfg.sgs), sgs_stability = cfg.sgs.stability,
          sgs_stress = cfg.sgs.stress, stability = sbl_stability(sf.stability),
          sponge_z = sp.z, sponge_α = sp.alpha, sponge_target = sp.target, seed = cfg.run.seed,
          extra_forcing = F.extra, θ_init = θi, u_init = sbl_profile(ini.u, T),
          v_init = sbl_profile(ini.v, T), t_start = ini.t_start, ρcp = sf.rho_cp)
    return sbl_case(; kw..., sbl_perturbation(ini.perturbation)..., F.surface...)
end

"""
    sbl_channel_case(cfg::RunConfig) -> WallBoundedCase

The channel of a channel kind (`stratified_channel`, `relaminarisation_channel`,
`collapse_channel`), before its spin-up state is copied in.
"""
function sbl_channel_case(cfg::RunConfig)
    k = cfg.case.kind; N = cfg.grid.size; seed = cfg.run.seed
    if k === :stratified_channel
        return stratified_channel_case(cfg.case.richardson, cfg.sgs.stability; N = N, seed = seed,
                                       sgs_stress = cfg.sgs.stress)
    elseif k === :relaminarisation_channel
        return relam_case(cfg.case.re_tau, cfg.case.h_over_l; N = N, seed = seed, sgs_stress = cfg.sgs.stress)
    end
    sc = collapse_heat_flux_scales(N; z0 = cfg.case.z0, θ_ref = cfg.case.theta_ref)
    fr = cfg.case.heat_flux_fraction
    return collapse_case(N; heat_flux = fr == 0 ? nothing : -fr * sc.Q_wall, θ_ref = cfg.case.theta_ref,
                         z0 = cfg.case.z0, c_s = something(cfg.sgs.c_s, 0.1))
end

"""
    sbl_channel_spinup(cfg::RunConfig; checkpoint = nothing) -> snapshot

The neutral spin-up a channel kind starts from (`case.spinup_time`):
`stratified_channel_spinup`, `relam_spinup` or the neutral collapse half
channel (`collapse_case`, run with `run_channel!`). `nothing` for a
stratified channel with `spinup_time = 0` (it then starts from noise).
`checkpoint` (a `CheckpointSchedule`) resumes an interrupted
relaminarisation spin-up (`relam_spinup`'s).
"""
function sbl_channel_spinup(cfg::RunConfig; checkpoint = nothing)
    k = cfg.case.kind; N = cfg.grid.size; T = cfg.case.spinup_time; seed = cfg.run.seed
    if k === :stratified_channel
        return T == 0 ? nothing : stratified_channel_spinup(; N = N, T_spin = T, seed = seed)
    elseif k === :relaminarisation_channel
        return relam_spinup(cfg.case.re_tau; N = N, T_spin = T, seed = seed, checkpoint = checkpoint)
    end
    spin = collapse_case(N; z0 = cfg.case.z0, c_s = something(cfg.sgs.c_s, 0.1))
    run_channel!(spin, T)
    return spin
end

"""
    sbl_channel_run(cfg, case, spinup; callback = nothing, checkpoint = nothing) -> NamedTuple

Run the channel `case` of a channel-kind config from `spinup`
(`sbl_channel_spinup`): `stratified_channel_run`, `relam_run` or
`collapse_run` with the config's times. `callback(case)` after every step.
"""
function sbl_channel_run(cfg::RunConfig, case, spinup; callback = nothing, checkpoint = nothing)
    k = cfg.case.kind; T_end = cfg.stepper.end_time
    if k === :stratified_channel
        return stratified_channel_run(cfg.case.richardson, cfg.sgs.stability; N = cfg.grid.size,
                                      T_end = T_end, T_avg = cfg.averaging.from, seed = cfg.run.seed,
                                      spinup = spinup, case = case, callback = callback)
    elseif k === :relaminarisation_channel
        return relam_run(cfg.case.re_tau, cfg.case.h_over_l, spinup; T_end = T_end,
                         sample = cfg.averaging.interval, checkpoint = checkpoint, case = case,
                         callback = callback)
    end
    sc = collapse_heat_flux_scales(cfg.grid.size; z0 = cfg.case.z0, θ_ref = cfg.case.theta_ref)
    r = collapse_run(case, spinup; T_end = T_end, average_from = cfg.averaging.from,
                     sample = cfg.averaging.interval, U0 = sc.U0, callback = callback)
    return merge(r, (; Q_wall = sc.Q_wall, Q_bulk = sc.Q_bulk, U0 = sc.U0, case = case))
end

"""
    SBLRun

What the `sbl` driver builds: `case` (an `SBLCase` for the homogeneous
kind, a `WallBoundedCase` for the channels), the resolved `cfg`, and,
after the run, `stats` (`sbl_run!`'s statistics or the channel run's
result), `series` (the rows of `output/series.csv`) and `towers`.
"""
mutable struct SBLRun
    case::Any
    cfg::RunConfig
    stats::Any
    series::Vector{NamedTuple}
    towers::Any
end

Config.build(::SBLDriver, cfg::RunConfig) =
    SBLRun(cfg.case.kind === :homogeneous ? sbl_case(cfg) : sbl_channel_case(cfg), cfg, nothing,
           NamedTuple[], nothing)

"""
    sbl_build(path_or_cfg; overrides = String[]) -> SBLRun

Build (not run) the case of an `sbl` config: a path, a shipped config
name (`"gabls1"`, from `examples/sbl/configs/`) or a `RunConfig`.
"""
sbl_build(cfg::RunConfig) = Config.build(SBLDriver(), cfg)
sbl_build(path::AbstractString; overrides = String[]) = sbl_build(sbl_config(path; overrides = overrides))

"""
    sbl_config(path_or_name; overrides = String[]) -> RunConfig

Load an `sbl` config: a path, or the name of a shipped config of
`examples/sbl/configs/` (`"gabls3"`).
"""
function sbl_config(path::AbstractString; overrides = String[])
    p = isfile(path) ? path : joinpath(SBL_CONFIG_DIR, endswith(path, ".toml") ? path : path * ".toml")
    return load_run_config(p; overrides = overrides)
end

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

function _sbl_series_row(c::SBLCase)
    p = sbl_profiles(c); k = argmax(p.S)
    scr = sbl_screen(c)
    zinv = p.z[end] > 300.0 + 2 * (p.z[2] - p.z[1]) ? sbl_inversion_height(p.z, p.θ) : NaN
    return (t = c.sim.t, u_star = sqrt(hypot(c.τx, c.τy)), wθ = mean(c.wθ), H = c.ρcp * mean(c.wθ),
            h = sbl_bl_height(p.zf, p.τ), S_max = p.S[k], z_jet = p.z[k], θ_screen = scr.θ_screen,
            U_screen = scr.U_screen, T_s = surface_temperature(c.Ts, c.sim.t), z_inv = zinv,
            ls_heat = c.ls_heat_applied)
end

_sbl_in_hours(t, (h0, h1)) = (h = mod(t / 3600, 24); h0 <= h1 ? h0 <= h < h1 : (h >= h0 || h < h1))

function _sbl_write_csv(path, rows::Vector)
    isempty(rows) && return nothing
    open(path, "w") do io
        println(io, join(string.(keys(rows[1])), ","))
        for r in rows
            println(io, join((repr(Float64(x)) for x in values(r)), ","))
        end
    end
    return path
end

# every bin with a sample; V_s is the hockey-stick fit to the bins of at
# least 20 samples (sbl_hockey_threshold), NaN with fewer than 4 of them
function _sbl_towers_rows(T::SBLTowers)
    rows = NamedTuple[]
    for l in eachindex(T.ks)
        b = sbl_towers_bins(T, l; min_count = 1)
        f = sbl_towers_bins(T, l)
        fit = length(f.V) >= 4 ? sbl_hockey_threshold(f.V, f.V_TKE; weights = Float64.(f.n)).V_s : NaN
        for i in eachindex(b.V)
            push!(rows, (z = T.z[l], V = b.V[i], V_TKE = b.V_TKE[i], sigma = b.σ[i], n = b.n[i], V_s = fit))
        end
    end
    return rows
end

_sbl_stat_value(x::Real) = Float64(x)
_sbl_stat_value(x::AbstractVector{<:Real}) = Float64.(x)
_sbl_stat_value(x::Symbol) = String(x)
_sbl_stat_value(x::Tuple) = collect(x)
_sbl_stat_value(x) = nothing

"Write the scalars and profiles of a result NamedTuple to `output/statistics.toml`."
function _sbl_write_stats(path, st)
    d = Dict{String,Any}()
    for (k, v) in pairs(st)
        v isa NamedTuple && (for (k2, v2) in pairs(v)
            x = _sbl_stat_value(v2); x === nothing || (d["$(k)_$(k2)"] = x)
        end; continue)
        x = _sbl_stat_value(v)
        x === nothing || (d[string(k)] = x)
    end
    open(io -> TOML.print(io, d; sorted = true), path, "w")
    return path
end

function Simulations.run!(::SBLDriver, r::SBLRun, rf::RunFolder, ck::CheckpointSchedule; callback = nothing)
    cfg = r.cfg
    ck_on = ck.every_steps > 0 || isfinite(ck.every_seconds)
    outdir = cfg.output.path === nothing ? output_dir(rf) : mkpath(cfg.output.path)
    if cfg.case.kind === :homogeneous
        c = r.case; st = cfg.stepper; av = cfg.averaging; tw = cfg.towers
        t_end = something(st.end_time, Inf)
        dz = c.sim.mesh.dz
        towers = tw.below > 0 ?
            sbl_towers(c, [(k - 0.5) * dz for k in 1:c.sim.mesh.Nz if (k - 0.5) * dz < tw.below];
                       window = tw.window) : nothing
        series = r.series; nextt = Ref(c.sim.t + av.series_interval); lastt = Ref(c.sim.t)
        comps = towers === nothing ? (series = series, nextt = nextt, lastt = lastt) :
                                     (series = series, nextt = nextt, lastt = lastt, towers = towers)
        # explicit rows (the run resumes inside sbl_run!, so the driver numbers
        # the steps); the record supplies the limiter, CFL and solver numbers
        with_monitor(c.sim; run = rf, components = (; sgs = c.closure), t_end = st.end_time,
                     cadence = cfg.logging.diagnostics_cadence, sgs = c.closure, attach = false) do m
            cb = function (c)
                dt = c.sim.t - lastt[]; lastt[] = c.sim.t
                monitor!(m, c.sim; step = c.nsteps, dt = dt)
                if towers !== nothing
                    towers.active = _sbl_in_hours(c.sim.t, tw.hours)
                    sbl_towers_sample!(towers, c, dt)
                end
                if av.series_interval > 0 && c.sim.t >= nextt[] - 1e-9
                    push!(series, _sbl_series_row(c)); nextt[] += av.series_interval
                end
                callback === nothing || callback(r, c.nsteps)
            end
            r.stats = sbl_run!(c, t_end; cfl = something(st.cfl, 1.2), average_from = av.from,
                               sample_every = av.sample_every, nsteps = something(st.nsteps, typemax(Int)),
                               callback = cb, checkpoint = ck, checkpoint_components = comps,
                               final_checkpoint = ck_on)
        end
        r.towers = towers
        _sbl_write_csv(joinpath(outdir, "series.csv"), series)
        towers === nothing || _sbl_write_csv(joinpath(outdir, "towers.csv"), _sbl_towers_rows(towers))
    else
        c = r.case
        spin = sbl_channel_spinup(cfg)
        with_monitor(c.sim; run = rf, components = (; sgs = c.closure), t_end = cfg.stepper.end_time,
                     cadence = cfg.logging.diagnostics_cadence, sgs = c.closure, attach = false) do m
            lastt = Ref(c.sim.t)
            cb = function (c)
                monitor!(m, c.sim; step = c.nsteps, dt = c.sim.t - lastt[]); lastt[] = c.sim.t
                callback === nothing || callback(r, c.nsteps)
            end
            r.stats = sbl_channel_run(cfg, c, spin; callback = cb,
                                      checkpoint = ck_on ? ck : nothing)
        end
        # the neutral state the cooling starts from (statistics.toml: spinup_*)
        cfg.case.kind === :relaminarisation_channel &&
            (r.stats = merge(r.stats, (; spinup = (tke = spin.tke, τ_wall = spin.τ_wall, Ub = spin.Ub))))
    end
    r.stats === nothing || _sbl_write_stats(joinpath(outdir, "statistics.toml"), r.stats)
    @info "sbl run finished" subsystem = :run kind = cfg.case.kind nsteps = r.case.nsteps t = r.case.sim.t
    return r
end

"""
    state_fields(r::SBLRun) -> NamedTuple

The prognostic arrays (with halos) of the run's simulation: `u, v, w, p, θ`.
"""
function Config.state_fields(r::SBLRun)
    s = r.case.sim
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data, θ = s.θ.data)
end

register_driver!(SBLDriver())
