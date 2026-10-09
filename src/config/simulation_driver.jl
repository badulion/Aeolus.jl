"""
    Aeolus.SimulationDriver

The model's configuration and run (CFG-b, #357; ONE-a #450; the schema,
ONE-b #451): a config without `[run] legacy` builds a
[`Model`](@ref Aeolus.Components.Model) of the Boussinesq or the
anelastic core on a box with its components, and runs it. See
[`SimulationDriverType`](@ref) and the Run configuration page; shipped
configs are in `examples/configs/`, `examples/kernels/configs/` and
`examples/coupled/configs/`.
"""
module SimulationDriver

# The model's assembly from a config (CFG-b #357, ONE-a #450, ONE-b #451):
# the schema (run-level [time], [output], [checks], [preprocess]; per
# domain [grid], [dynamics], [boundaries], [initial] and one section per
# component, generated from the component schemas), the build of the Model
# and its run loop.
#
# Its own module because it needs the dynamical cores (Simulations,
# Anelastic, TimeIntegration, StretchedGrids) and the components, which are
# included after Aeolus.Config; included at the end of Aeolus.jl.

using Random: Xoshiro, randn
using ..Grids: RegularGrid, CellCenter
using ..Fields: Field, set!, interior_view
using ..BoundaryConditions: FieldBoundaryConditions, open_boundary_conditions, open_boundary_states,
                            set_boundary_data!
using ..InputDrivers: InputDrivers, read_dynamic_driver, read_era5, write_era5_driver, check_planes,
                      check_coverage, ERA5_OUTPUTS
using ..Simulations: Simulations, BoussinesqSimulation, step!
using ..TimeIntegration: CFLConstraints
using ..StretchedGrids: StretchedGrid
using ..Anelastic: AnelasticSimulation, MassWeightedSimulation, isentropic_reference
using ..Components: Components, Domain, Model, CentredAdvection, SGS, Coriolis, Sponge, SurfaceThetaFlux,
                    build_components, checkpoint_components, sgs_closure, step_adaptive!, component_problems
using ..IO: IO, RunFolder, CheckpointSchedule, StepperState, record_step!, previous_dt, resume!,
            maybe_checkpoint!, with_monitor, monitored_step!, check_blowup!, output_dir
using ..NCIO: nc_create, nc_defvar!
using ..Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, OpenTable, OneOf, Optional,
                Path, section, option!, register_driver!, config_domains, MODEL

export SimulationDriverType, SimulationRun, simulation, profile_value

"""
    SimulationDriverType()

The model ([`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver),
registered as `"model"`): what a config without `[run] legacy` runs
(schema version 2, ONE-b #451). Its run-level sections are `[time]`,
`[output]` (`variables`), `[checks]` and `[preprocess]`; its per-domain
sections are `[grid]`, `[dynamics]`, `[boundaries]`, `[initial]`, one
section per component of [`COMPONENTS`](@ref), generated from the
component schemas (`[advection]`, `[sgs]`, `[coriolis]`, `[sponge]`,
`[surface_theta_flux]`), and the sections of the EX-d couplings (#441),
`[forcing]`, `[surface]`, `[microphysics]`, `[tracer]`, `[particles]` and
`[obstacles]` ([`SimCouplings`](@ref)). A domain's own keys are `id`,
`static_driver` and `dynamic_driver`. `aeolus options` prints them all.

A bare config (no section at all) is a minimal dry LES: 32³ cells on
3.2 km × 3.2 km × 1.6 km, periodic in x and y between free-slip walls,
the Boussinesq core with buoyancy on θ₀ = 300 K, Smagorinsky SGS, adaptive
SSP-RK3 steps (CFL 0.5, at most 10 s) for one hour. Every other component
is off until its section enables it.

The domain's `dynamic_driver` (CFG-g, #362) sets the initial u, v, θ, qv
from its profiles, and open lateral faces (`boundaries.x` or `y = "open"`)
take its boundary planes, blended linearly in time. `aeolus preprocess
dynamic --from-era5 <files…> --config <config.toml>` writes one from ERA5
(CFG-h, #363, [`preprocess_dynamic`](@ref
Aeolus.Config.preprocess_dynamic)) with the site and outputs of
`[preprocess.era5]`. A config with several `[[domain]]` blocks is
validated domain by domain, but the model runs one domain until nesting
by domains (ONE-f, #455).
"""
struct SimulationDriverType <: AbstractRunDriver end

Config.driver_name(::SimulationDriverType) = MODEL

"""
    COMPONENTS

The component types of the model whose sections the configuration schema
generates from their [`schema`](@ref Aeolus.Components.schema), in the
order of the option reference: the `:advection` and `:sgs` slots,
Coriolis, the sponge and the surface θ flux. (The EX-d couplings are one
component, [`SimulationCouplings`](@ref), with six sections, until their
port to components, ONE-e #454.)
"""
const COMPONENTS = (CentredAdvection, SGS, Coriolis, Sponge, SurfaceThetaFlux)

_isnum(x) = x isa Real && !(x isa Bool)

# the couplings around the core's step (EX-d, #441)
include("simulation_couplings.jl")

# ---------------------------------------------------------------------------
# initial profiles
# ---------------------------------------------------------------------------

const _PROFILE_KEYS = Dict("constant" => ["value"],
                           "mixed_layer_lapse" => ["value", "z_i", "lapse"],
                           "table" => ["z", "values"],
                           "exponential" => ["value", "scale_height"],
                           "taylor_green" => ["amplitude", "k"],
                           "box" => ["value", "x", "y", "z"])

# the profile kinds of each variable: the vertical ones for all, the
# Taylor–Green field for u and v, a box for the cloud water (EX-d, #441)
_profile_kinds(var) = sort!(vcat(["constant", "exponential", "mixed_layer_lapse", "table"],
                                var in ("u", "v") ? ["taylor_green"] : String[], var == "ql" ? ["box"] : String[]))

"The validator of the profile table of variable `var` (an empty table: the default profile)."
function _profile_validator(var::String)
    return function (t)
        isempty(t) && return String[]
        kind = get(t, "kind", nothing)
        kinds = _profile_kinds(var)
        kind in kinds || return ["kind: expected one of " * join(repr.(kinds), ", ") *
                                 ", got $(repr(kind))"]
        want = _PROFILE_KEYS[kind]
        probs = String[]
        for k in sort!(collect(keys(t)))
            k == "kind" || k in want || push!(probs, "$k: unknown key for kind = \"$kind\" (keys: $(join(want, ", ")))")
        end
        for k in want
            haskey(t, k) || (push!(probs, "$k: required for kind = \"$kind\""); continue)
            v = t[k]
            if kind == "table"
                (v isa AbstractVector && !isempty(v) && all(_isnum, v)) ||
                    push!(probs, "$k: expected a non-empty array of numbers")
            elseif kind == "box" && k != "value"
                (v isa AbstractVector && length(v) == 2 && all(_isnum, v) && v[1] < v[2]) ||
                    push!(probs, "$k: expected [lower, upper] with lower < upper (m)")
            else
                _isnum(v) && isfinite(v) || push!(probs, "$k: expected a finite number, got $(repr(v))")
            end
        end
        if kind == "table" && isempty(probs)
            length(t["z"]) == length(t["values"]) ||
                push!(probs, "z and values have different lengths ($(length(t["z"])) and $(length(t["values"])))")
            issorted(t["z"]; lt = <=) || push!(probs, "z: expected strictly increasing heights")
        end
        kind == "exponential" && isempty(probs) && !(t["scale_height"] > 0) &&
            push!(probs, "scale_height: expected a positive height (m)")
        return probs
    end
end

"The validator of `[initial] velocity_perturbation` (an empty table: none)."
function _velocity_noise_validator(t)
    isempty(t) && return String[]
    probs = String[]
    for k in sort!(collect(keys(t)))
        k in ("kind", "amplitude") || push!(probs, "$k: unknown key (keys: kind, amplitude)")
    end
    get(t, "kind", nothing) in ("uniform", "normal") ||
        push!(probs, "kind: expected \"uniform\" or \"normal\", got $(repr(get(t, "kind", nothing)))")
    a = get(t, "amplitude", nothing)
    (_isnum(a) && isfinite(a) && a >= 0) || push!(probs, "amplitude: required, a finite number ≥ 0 (m s⁻¹)")
    return probs
end

"""
    profile_value(p::AbstractDict, z) -> Float64

The value at height `z` of a vertical profile of `[initial]`:

  * `{kind = "constant", value}`;
  * `{kind = "mixed_layer_lapse", value, z_i, lapse}`: `value` below `z_i`,
    `value + lapse (z − z_i)` above (a capping inversion; with `z_i` at the
    ground, a linear profile);
  * `{kind = "table", z = […], values = […]}`: linear interpolation,
    constant beyond the ends;
  * `{kind = "exponential", value, scale_height}`: `value exp(−z/scale_height)`.
"""
function profile_value(p::AbstractDict, z::Real)
    kind = p["kind"]
    if kind == "constant"
        return Float64(p["value"])
    elseif kind == "mixed_layer_lapse"
        zi = Float64(p["z_i"])
        return z < zi ? Float64(p["value"]) : Float64(p["value"]) + Float64(p["lapse"]) * (z - zi)
    elseif kind == "table"
        zs = Float64.(p["z"]); vs = Float64.(p["values"])
        z <= zs[1] && return vs[1]
        z >= zs[end] && return vs[end]
        i = searchsortedlast(zs, z)
        w = (z - zs[i]) / (zs[i+1] - zs[i])
        return (1 - w) * vs[i] + w * vs[i+1]
    elseif kind == "exponential"
        return Float64(p["value"]) * exp(-z / Float64(p["scale_height"]))
    end
    throw(ArgumentError("profile_value: kind $(repr(kind)) is not a vertical profile"))
end

# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------

const _OUTPUT_FIELDS = ["u", "v", "w", "p", "theta", "qv", "ql", "qr", "tracer"]

"The per-domain sections of the model: the domain's own keys, its grid, core, boundaries, initial state and components."
const DOMAIN_SECTIONS = vcat(["domain", "grid", "dynamics", "boundaries", "initial"],
                             [Components.section(C) for C in COMPONENTS],
                             ["forcing", "surface", "microphysics", "tracer", "particles", "obstacles"])

Config.domain_sections(::SimulationDriverType) = DOMAIN_SECTIONS

function Config.driver_sections(::SimulationDriverType)
    dom = section("domain"; doc = "A domain: its id and its input drivers (CFG-f/g).") do s
        option!(s, "id", String; default = "main", doc = "the domain's name (unique; a config without [[domain]] has one, \"main\")")
        option!(s, "static_driver", Optional{Path};
                doc = "the static driver (aeolus preprocess static): its buildings are the obstacles of [obstacles]; hashed under [inputs]")
        option!(s, "dynamic_driver", Optional{Path};
                doc = "the dynamic driver (CFG-g, #362; aeolus preprocess dynamic --from-era5): its initial profiles set u, v, theta (and qv) — not together with the [initial] tables — and open faces take its boundary planes; hashed under [inputs]")
    end
    grid = section("grid"; doc = "The box and its mesh.") do s
        option!(s, "size", NTuple{3,Int}; default = (32, 32, 32), range = (1, 65536),
                doc = "cells in x, y, z")
        option!(s, "extent", NTuple{3,Float64}; default = (3200.0, 3200.0, 1600.0), range = (0, Inf),
                open_lower = true, units = "m", doc = "domain size Lx, Ly, Lz")
        option!(s, "halo", NTuple{3,Int}; default = (2, 2, 2), range = (1, 8),
                doc = "halo cells per side")
        option!(s, "stretching", Symbol; default = :uniform,
                allowed = [:uniform, :geometric, :tanh, :tanh_both],
                doc = "vertical spacing (StretchedGrid): uniform (a RegularGrid), geometric (dz[k+1] = r dz[k]), tanh (clustered at the bottom), tanh_both (at both walls)")
        option!(s, "stretching_parameter", Float64; units = "1", default = 1.0, range = (0, Inf), open_lower = true,
                doc = "r of geometric, β of tanh / tanh_both; unused for uniform")
    end
    dyn = section("dynamics"; doc = "The dynamical core and its parameters.") do s
        option!(s, "core", Symbol; default = :boussinesq, allowed = [:boussinesq, :anelastic],
                doc = "boussinesq (BoussinesqSimulation) or anelastic (AnelasticSimulation on an isentropic base state)")
        option!(s, "nu", Float64; default = 0.0, range = (0, Inf), units = "m^2 s^-1",
                doc = "kinematic viscosity; θ and qv diffuse with the same coefficient (Prandtl number 1)")
        option!(s, "theta0", Float64; default = 300.0, range = (0, Inf), units = "K",
                doc = "reference potential temperature of the buoyancy; 0 turns buoyancy off (boussinesq); anelastic: the base state's θ (> 0)")
        option!(s, "g", Float64; default = 9.81, range = (0, Inf), units = "m s^-2",
                doc = "gravitational acceleration")
        option!(s, "moisture", Bool; default = false,
                doc = "carry the water-vapour mixing ratio qv (boussinesq only)")
    end
    bnd = section("boundaries"; doc = "Boundary conditions per axis.") do s
        for ax in ("x", "y", "z")
            option!(s, ax, Symbol; default = ax == "z" ? :free_slip : :periodic,
                    allowed = ax == "z" ? [:periodic, :free_slip, :noslip] : [:periodic, :free_slip, :noslip, :open],
                    doc = "the $ax faces: periodic, free_slip (impermeable, stress free) or noslip (z only); θ and qv have no flux through a wall, except the prescribed θ fluxes ([surface_theta_flux])" *
                          (ax == "z" ? "" : "; open: the boundary planes of the domain's dynamic_driver, blended linearly in time (boussinesq, uniform grid, z walls)"))
        end
    end
    ini = section("initial"; doc = "The initial state.") do s
        for (v, d) in (("u", "0"), ("v", "0"), ("theta", "theta0 (anelastic: the base state)"), ("qv", "0"))
            option!(s, v, OpenTable(_profile_validator(v)),
                    doc = "initial $v: {kind = \"constant\", value}, {kind = \"mixed_layer_lapse\", value, z_i, lapse}, " *
                          "{kind = \"table\", z, values}" *
                          (v in ("u", "v") ? ", or {kind = \"taylor_green\", amplitude, k} (u = A sin kx cos ky cos kz, v = -A cos kx sin ky cos kz)" : "") *
                          "; {} (the default): $d")
        end
        option!(s, "ql", OpenTable(_profile_validator("ql")),
                doc = "initial cloud water (microphysics.scheme = \"kessler\"): a vertical profile as for theta, or " *
                      "{kind = \"box\", value, x = [x₀, x₁], y = [y₀, y₁], z = [z₀, z₁]} (value strictly inside the box, 0 outside); {}: 0")
        option!(s, "velocity_perturbation", OpenTable(_velocity_noise_validator),
                doc = "noise added to u, v and w (in that order, after the θ noise, from the same Xoshiro([run] seed)): " *
                      "{kind = \"uniform\", amplitude} (uniform in [-a/2, a/2], one draw per cell) or " *
                      "{kind = \"normal\", amplitude} (a × standard normal, one randn array per field); {}: none")
        option!(s, "perturbation_amplitude", Float64; default = 0.0, range = (0, Inf), units = "K",
                doc = "white noise on θ, uniform in [-a/2, a/2], drawn from Xoshiro([run] seed)")
        option!(s, "perturbation_depth", Float64; default = Inf, range = (0, Inf), units = "m",
                doc = "the noise is added below this height")
    end
    tim = section("time"; doc = "Time stepping: a fixed dt or adaptive steps, for a duration or a number of steps.") do s
        option!(s, "scheme", Symbol; default = :rk3, allowed = [:euler, :rk3],
                doc = "rk3 (step_rk3!, SSP-RK3) or euler (step!, the forward-Euler projection); anelastic: euler")
        option!(s, "dt", OneOf(Float64, Symbol); default = :adaptive, allowed = [:adaptive], range = (0, Inf),
                open_lower = true, units = "s",
                doc = "a fixed step (with duration it is shortened to duration / ceil(duration / dt)), or \"adaptive\" (step_adaptive!, from cfl and dt_max)")
        option!(s, "cfl", Float64; default = 0.5, units = "1", range = (0, Inf), open_lower = true,
                doc = "adaptive steps: the advective Courant number")
        option!(s, "dt_max", Float64; default = 10.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "adaptive steps: the largest step (the step of a fluid at rest)")
        option!(s, "duration", Optional{Float64}; default = 3600.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "simulated time to run; \"none\" with nsteps")
        option!(s, "nsteps", Optional{Int}; range = (0, typemax(Int)), doc = "steps to run (with duration = \"none\")")
    end
    out = section("output"; doc = "") do s
        option!(s, "variables", Vector{String}; default = String[], allowed = _OUTPUT_FIELDS,
                doc = "fields written to output/fields_<step>.nc every cadence (cell-index arrays, Float32); [] : none")
    end
    # one key per published check, `name = {required, tolerance}`: no
    # component publishes a check yet (ONE-d, #453), so the section is empty
    chk = section(_ -> nothing, "checks";
                  doc = "The checks the enabled components publish, `name = {required, tolerance}` (ONE-d, #453); none is published yet.")
    pre = section("preprocess"; doc = "Inputs of the preprocessors (aeolus preprocess dynamic, static).") do s
        option!(s, "obstacles", Optional{Path};
                doc = "a GeoJSON FeatureCollection of obstacle footprints in mesh metres with a `height` each: aeolus preprocess static writes the domain's static_driver from it (EX-d, #441)")
        option!(s, "era5", OpenTable(_era5_validator),
                doc = "ERA5 (--from-era5, CFG-h #363): {lat, lon (degrees, the domain centre; required), box = 0.5 (degrees, the half-width of the centred differences of the geostrophic wind and the advective tendencies), levels = \"pressure\" | \"model\", outputs = [\"profiles\", \"planes\", \"large_scale\", \"geostrophic\", \"subsidence\", \"advection\", \"surface\"] (the default: all)}")
    end
    comps = Section[Components.schema(C) for C in COMPONENTS]
    return Section[tim, out, chk, pre, dom, grid, dyn, bnd, ini, comps..., _coupling_sections()...]
end

const _ERA5_KEYS = ("lat", "lon", "box", "levels", "outputs")

"The validator of `[preprocess.era5]` (an empty table: no ERA5 preprocessing configured)."
function _era5_validator(t)
    isempty(t) && return String[]
    probs = String[]
    for k in sort!(collect(keys(t)))
        k in _ERA5_KEYS || push!(probs, "$k: unknown key (keys: $(join(_ERA5_KEYS, ", ")))")
    end
    for (k, lo, hi) in (("lat", -90, 90), ("lon", -360, 360))
        haskey(t, k) || (push!(probs, "$k: required (degrees, the domain centre)"); continue)
        (_isnum(t[k]) && lo <= t[k] <= hi) || push!(probs, "$k: expected a number in [$lo, $hi], got $(repr(t[k]))")
    end
    haskey(t, "box") && !(_isnum(t["box"]) && t["box"] > 0) &&
        push!(probs, "box: expected a positive number of degrees, got $(repr(t["box"]))")
    haskey(t, "levels") && !(t["levels"] in ("pressure", "model")) &&
        push!(probs, "levels: expected \"pressure\" or \"model\", got $(repr(t["levels"]))")
    if haskey(t, "outputs")
        o = t["outputs"]
        ok = String.(ERA5_OUTPUTS)
        (o isa AbstractVector && all(x -> x in ok, o)) ||
            push!(probs, "outputs: expected an array of " * join(repr.(ok), ", ") * ", got $(repr(o))")
    end
    return probs
end

"The schema default of `section.key` of the model."
_schema_default(cfg, sec, key) =
    only(o for o in only(x for x in Config.config_schema(cfg).sections if x.name == sec).options if o.key == key).default

"Whether `cfg` steps adaptively (`time.dt = \"adaptive\"`)."
_adaptive(cfg) = cfg.time.dt === :adaptive

"The component types a domain's config enables (in the order of [`COMPONENTS`](@ref))."
enabled_components(cfg) = Any[C for C in COMPONENTS if Components.enabled(C, cfg[Components.section(C)])]

function _cross_checks(cfg)
    e = ConfigError[]
    t = cfg.time; d = cfg.dynamics; b = cfg.boundaries; s = cfg.sgs; g = cfg.grid; dom = cfg.domain
    adaptive = _adaptive(cfg)
    (t.duration === nothing) == (t.nsteps === nothing) &&
        push!(e, ConfigError("time.duration", "give exactly one of time.duration and time.nsteps (nsteps with duration = \"none\")"))
    if !adaptive
        for k in ("cfl", "dt_max")
            isequal(t[k], _schema_default(cfg, "time", k)) ||
                push!(e, ConfigError("time.$k", "belongs to adaptive steps (time.dt = \"adaptive\"), not a fixed dt: it would be ignored"))
        end
    end
    for ax in ("x", "y")
        b[ax] === :noslip &&
            push!(e, ConfigError("boundaries.$ax", "noslip is a z-wall condition; use free_slip or periodic"))
    end
    if b.x === :open || b.y === :open
        ax = b.x === :open ? "x" : "y"
        dom.dynamic_driver === nothing &&
            push!(e, ConfigError("boundaries.$ax", "open faces take the boundary planes of a dynamic driver: set the domain's dynamic_driver (aeolus preprocess dynamic --from-era5)"))
        (b.x in (:open, :periodic) && b.y in (:open, :periodic)) ||
            push!(e, ConfigError("boundaries.$ax", "with an open axis the other lateral axis is open or periodic"))
        b.z === :periodic && push!(e, ConfigError("boundaries.z", "an open box needs z walls"))
        d.core === :boussinesq || push!(e, ConfigError("dynamics.core", "open faces need the boussinesq core"))
        g.stretching === :uniform || push!(e, ConfigError("grid.stretching", "open faces need a uniform grid"))
        cfg.sponge.alpha_max == 0 ||
            push!(e, ConfigError("sponge.alpha_max", "the sponge of an open box has no target here; use 0"))
    end
    if dom.dynamic_driver !== nothing
        for (k, p) in (("u", cfg.initial.u), ("v", cfg.initial.v), ("theta", cfg.initial.theta), ("qv", cfg.initial.qv))
            isempty(p) || push!(e, ConfigError("initial.$k", "the domain's dynamic_driver sets the initial $k: give one of the two"))
        end
        d.core === :boussinesq || push!(e, ConfigError("dynamic_driver", "a dynamic driver initialises the boussinesq core"))
    end
    if d.core === :anelastic
        b.z === :periodic && push!(e, ConfigError("boundaries.z", "the anelastic core needs z walls"))
        d.theta0 > 0 || push!(e, ConfigError("dynamics.theta0", "the anelastic core needs theta0 > 0 (its base state's θ)"))
        d.moisture && push!(e, ConfigError("dynamics.moisture", "the anelastic core has no qv"))
        t.scheme === :euler || push!(e, ConfigError("time.scheme", "the anelastic core steps with euler"))
        adaptive && push!(e, ConfigError("time.dt", "the anelastic core takes a fixed dt"))
        g.stretching === :uniform || push!(e, ConfigError("grid.stretching", "the anelastic core needs a uniform grid"))
        (cfg.checkpoint.every_steps == 0 && cfg.checkpoint.every_seconds == Inf) ||
            push!(e, ConfigError("checkpoint.every_steps", "checkpoints need core = boussinesq"))
    end
    if g.stretching !== :uniform
        s.closure in (:none, :smagorinsky) ||
            push!(e, ConfigError("sgs.closure", "on a stretched grid the closure is none or smagorinsky, not $(s.closure)"))
        s.stability === :none ||
            push!(e, ConfigError("sgs.stability", "the stability correction needs a uniform grid"))
    end
    s.stability === :none || s.closure in (:smagorinsky, :dynamic_smagorinsky, :vreman, :sigma) ||
        push!(e, ConfigError("sgs.stability", "the stability correction wraps smagorinsky, dynamic_smagorinsky, vreman or sigma, not $(s.closure)"))
    sp = cfg.sponge
    (sp.alpha_max != 0 && sp.z_start !== nothing && !(sp.z_start < g.extent[3])) &&
        push!(e, ConfigError("sponge.z_start", "the sponge starts below the top (grid.extent[3] = $(g.extent[3]) m), got $(sp.z_start) m"))
    (!d.moisture && !isempty(cfg.initial.qv)) &&
        push!(e, ConfigError("initial.qv", "a qv profile needs dynamics.moisture = true"))
    ("qv" in cfg.output.variables && !d.moisture) &&
        push!(e, ConfigError("output.variables", "qv is written only with dynamics.moisture = true"))
    for (k, p) in (("u", cfg.initial.u), ("v", cfg.initial.v))
        (get(p, "kind", "") == "taylor_green" && (b.x !== :periodic || b.y !== :periodic)) &&
            push!(e, ConfigError("initial.$k", "the Taylor-Green field needs periodic x and y"))
    end
    # slots, requirements, incompatibilities of the enabled components (ONE-b)
    for p in component_problems(enabled_components(cfg))
        push!(e, ConfigError(p.key, p.message))
    end
    append!(e, _coupling_checks(cfg))
    return e
end

const _CROSS_CHECKS_DOC = """
exactly one of `time.duration` and `time.nsteps`; with a fixed `time.dt`, `time.cfl` and `time.dt_max` keep their defaults (they belong to adaptive steps)
`noslip` is a z-wall condition: not for `boundaries.x`, `boundaries.y`
an open face needs a domain `dynamic_driver`, the other lateral axis open or periodic, z walls, the boussinesq core, a uniform grid and no sponge
the domain's `dynamic_driver` and the `initial` tables of u, v, theta, qv exclude each other; it initialises the boussinesq core
the anelastic core needs z walls, `dynamics.theta0` > 0, no moisture, `time.scheme = "euler"`, a fixed `time.dt`, a uniform grid and no checkpoints
on a stretched grid `sgs.closure` is `none` or `smagorinsky` and `sgs.stability` is `none`
`sgs.stability` wraps `smagorinsky`, `dynamic_smagorinsky`, `vreman` or `sigma` only
`sponge.z_start` lies below the top
an `initial.qv` profile and the `qv` output variable need `dynamics.moisture = true`
a Taylor-Green initial `u` or `v` needs periodic x and y
the enabled components hold at most one component per slot and an `:advection` provider, and meet each other's `requires` and `incompatible` (`Components.component_problems`)"""

Config.driver_checks(::SimulationDriverType) = Any[_CROSS_CHECKS_DOC * "\n" * _COUPLING_CHECKS_DOC => _cross_checks]

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

"""
    SimulationRun

What the model builds from a config: `model`, a [`Model`](@ref
Aeolus.Components.Model) of one [`Domain`](@ref Aeolus.Components.Domain)
(ONE-a, #450) — the core (a `BoussinesqSimulation` or an
`AnelasticSimulation`, [`simulation`](@ref)), its time scheme, its
components (the `:advection` and `:sgs` slots, Coriolis, the sponge, the
surface θ flux) and its drivers (`boundary`, the
[`FileBoundaryFeed`](@ref Aeolus.InputDrivers.FileBoundaryFeed) of the
open faces); the EX-d couplings ([`SimCouplings`](@ref): `[forcing]`,
`[surface]`, `[microphysics]`, `[tracer]`, `[particles]`, `[obstacles]`,
#441) are its [`SimulationCouplings`](@ref) component when any is on;
`stepper` (a [`StepperState`](@ref)) and the resolved `cfg`.
"""
struct SimulationRun{M<:Model}
    model::M
    stepper::StepperState
    cfg::RunConfig
end

"The core of a `simulation` run: the `BoussinesqSimulation` or `AnelasticSimulation` of its one domain."
simulation(r::SimulationRun) = only(Components.domains(r.model)).state
_domain(r::SimulationRun) = only(Components.domains(r.model))

function _mesh(g)
    if g.stretching === :uniform
        return RegularGrid(; size = g.size, extent = g.extent, halo = g.halo)
    end
    return StretchedGrid(; size = g.size, extent = g.extent, halo = g.halo,
                         stretching = (g.stretching, g.stretching_parameter))
end

_bc(s::Symbol) = s === :free_slip ? :wall : s
_scalar_bc(s::Symbol) = s === :periodic ? :periodic : :noflux

# The open box of `cfg` (an open lateral axis): the faces' planes, data set later.
function _open_bcs(cfg, mesh)
    b = cfg.boundaries
    lat(a) = a === :open ? :open : :periodic
    wall(a) = a === :noslip ? :noslip : :wall
    fields = cfg.dynamics.moisture ? (:u, :v, :w, :θ, :qv) : (:u, :v, :w, :θ)
    return open_boundary_conditions(mesh, nothing; x = lat(b.x), y = lat(b.y), bottom = wall(b.z), top = wall(b.z),
                                    fields = fields, halo = 1)
end

_is_open(cfg) = cfg.boundaries.x === :open || cfg.boundaries.y === :open

function Config.build(::SimulationDriverType, cfg::RunConfig)
    n = length(config_domains(cfg))
    n == 1 || throw(ConfigError("domain", "the model runs one domain; $n [[domain]] blocks are validated but " *
                                          "run only with nesting by domains (ONE-f, #455)"))
    mesh = _mesh(cfg.grid)
    d = cfg.dynamics; b = cfg.boundaries; s = cfg.sgs; ddrv = cfg.domain.dynamic_driver
    dyn = ddrv === nothing ? nothing : read_dynamic_driver(ddrv; mesh = mesh, planes = _is_open(cfg))
    feed = nothing
    if _is_open(cfg)
        dyn.boundary === nothing &&
            throw(ConfigError("dynamic_driver", "$ddrv has no boundary planes, which the open faces need (write it with the run's open boundaries)"))
        bcs = θ_bcs = _open_bcs(cfg, mesh)
        feed = dyn.boundary
        check_planes(feed, only(open_boundary_states(bcs)).planes)
        set_boundary_data!(bcs, feed.data)
    else
        bcs = FieldBoundaryConditions(_bc(b.x), _bc(b.y), _bc(b.z))
        θ_bcs = FieldBoundaryConditions(_scalar_bc(b.x), _scalar_bc(b.y), _scalar_bc(b.z))
    end
    # the core carries the dynamics only: SGS, advection, Coriolis, the
    # surface θ flux and the sponge are its domain's components (ONE-a)
    kw = (; ν = d.nu, g_accel = d.g, Pr_t = s.pr_t, bcs = bcs, θ_bcs = θ_bcs)
    sim = if d.core === :anelastic
        AnelasticSimulation(mesh; ref = isentropic_reference(mesh; θ_const = d.theta0, g_accel = d.g), kw...)
    else
        BoussinesqSimulation(mesh; θ₀ = d.theta0, moisture = d.moisture, kw...)
    end
    couplings = _couplings(cfg, mesh)
    _initialise!(sim, cfg)
    dyn === nothing || _initialise_dynamic!(sim, dyn, cfg)
    couplings.water === nothing || isempty(cfg.initial.ql) || _set_profile!(couplings.water.ql, cfg.initial.ql, :ql)
    t = cfg.time
    cfl = _adaptive(cfg) ? CFLConstraints(; C_advective = t.cfl, dt_max = t.dt_max) : nothing
    d0 = Domain(sim; id = cfg.domain.id, scheme = t.scheme, cfl = cfl, validate = false,
                drivers = feed === nothing ? (;) : (; boundary = feed))
    comps = build_components(Tuple(C => cfg[Components.section(C)] for C in COMPONENTS), d0)
    _any_coupling(cfg) && (comps = (comps..., SimulationCouplings(couplings)))
    scheme = _adaptive(cfg) ? :adaptive : t.scheme
    return SimulationRun(Model(Domain(d0; components = comps)), StepperState(scheme), cfg)
end

# The initial profiles of a dynamic driver (linear in z, constant beyond
# the ends: `profile_value` of a table).
function _initialise_dynamic!(sim, dyn, cfg)
    P = dyn.profiles
    P === nothing && throw(ConfigError("dynamic_driver", "$(cfg.domain.dynamic_driver) has no initial profiles (init_theta, …)"))
    for (n, f) in ((:u, sim.u), (:v, sim.v), (:theta, sim.θ), (:qv, sim.qv))
        n === :qv && !cfg.dynamics.moisture && continue
        a = getproperty(P, n)
        a === nothing && continue
        _set_profile!(f, Dict{String,Any}("kind" => "table", "z" => P.z, "values" => a), n)
    end
    return sim
end

function _set_profile!(f::Field, p::AbstractDict, comp::Symbol)
    if get(p, "kind", "") == "taylor_green"
        A = Float64(p["amplitude"]); k = Float64(p["k"])
        comp === :u ? set!(f, (x, y, z) -> A * (sin(k * x) * cos(k * y) * cos(k * z))) :
                      set!(f, (x, y, z) -> A * (-cos(k * x) * sin(k * y) * cos(k * z)))
    elseif get(p, "kind", "") == "box"
        v = Float64(p["value"]); bx = Float64.(p["x"]); by = Float64.(p["y"]); bz = Float64.(p["z"])
        set!(f, (x, y, z) -> (bx[1] < x < bx[2] && by[1] < y < by[2] && bz[1] < z < bz[2]) ? v : 0.0)
    else
        set!(f, (x, y, z) -> profile_value(p, z))
    end
    return f
end

function _initialise!(sim, cfg)
    ini = cfg.initial
    isempty(ini.u) || _set_profile!(sim.u, ini.u, :u)
    isempty(ini.v) || _set_profile!(sim.v, ini.v, :v)
    if !isempty(ini.theta)
        _set_profile!(sim.θ, ini.theta, :theta)
    elseif sim isa BoussinesqSimulation && cfg.dynamics.theta0 != 0
        set!(sim.θ, (x, y, z) -> cfg.dynamics.theta0)
    end                                  # anelastic: the constructor set θ = θ̄(z)
    isempty(ini.qv) || _set_profile!(sim.qv, ini.qv, :qv)
    a = ini.perturbation_amplitude
    rng = Xoshiro(cfg.run.seed)
    if a > 0
        depth = ini.perturbation_depth
        noise = Field(CellCenter(), sim.mesh)
        set!(noise, (x, y, z) -> z < depth ? a * (rand(rng) - 0.5) : 0.0)
        interior_view(sim.θ) .+= interior_view(noise)
    end
    vp = ini.velocity_perturbation
    if !isempty(vp)
        A = Float64(vp["amplitude"]); uniform = vp["kind"] == "uniform"
        for f in (sim.u, sim.v, sim.w)          # one field at a time, in this order
            if uniform                          # one draw per cell, in set!'s order
                noise = Field(location(f), sim.mesh)
                set!(noise, (x, y, z) -> A * (rand(rng) - 0.5))
                interior_view(f) .+= interior_view(noise)
            else                                # one array of normals per field (randn's bulk stream)
                interior_view(f) .+= A .* randn(rng, size(interior_view(f))...)
            end
        end
    end
    return sim
end

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

"The fixed dt and step count of a fixed-step run."
function _fixed_steps(t)
    if t.duration !== nothing
        n = ceil(Int, t.duration / t.dt)
        return t.duration / n, n
    end
    return t.dt, t.nsteps
end

# One step of the model through the stage pipeline (Components): a fixed
# dt, or the adaptive pick landing on `t_end`.
_take_step!(r::SimulationRun, dt) = (step!(r.model, dt); (simulation(r), Float64(dt), 0))
_adaptive_step!(r::SimulationRun, t_end) =
    step_adaptive!(r.model; dt_prev = previous_dt(r.stepper), t_end = t_end)

# output: the selected fields every `cadence` simulated seconds
const _FIELD_OF = Dict("u" => :u, "v" => :v, "w" => :w, "p" => :p, "theta" => :θ, "qv" => :qv)

function _write_output(r::SimulationRun, dir, step)
    sim = simulation(r); m = sim.mesh
    path = joinpath(dir, string("fields_", lpad(step, 9, '0'), ".nc"))
    nc = nc_create(path; attributes = Dict("t" => Float64(sim.t), "step" => Int32(step),
                                           "note" => "cell-index arrays: face fields at their own staggered positions"))
    try
        nc.dim["x"] = m.Nx; nc.dim["y"] = m.Ny; nc.dim["z"] = m.Nz
        c = sim_couplings(r)
        for f in r.cfg.output.variables
            v = nc_defvar!(nc, f, Float32, ("x", "y", "z"))
            src = f == "ql" ? c.water.ql : f == "qr" ? c.water.qr : f == "tracer" ? c.tracer.c :
                  getproperty(sim, _FIELD_OF[f])
            v[:, :, :] = Float32.(Array(interior_view(src)))
        end
    finally
        close(nc)
    end
    return path
end

function Simulations.run!(::SimulationDriverType, r::SimulationRun, rf::RunFolder,
                          ck::CheckpointSchedule; callback = nothing)
    sim = simulation(r); cfg = r.cfg; st = cfg.time
    dom = _domain(r); comps = checkpoint_components(dom); closure = sgs_closure(dom)
    resume!(ck, sim; components = comps, stepper = r.stepper)
    n = ck.last_step
    adaptive = _adaptive(cfg)
    dt_fixed, nfix = adaptive ? (NaN, typemax(Int)) : _fixed_steps(st)
    nmax = adaptive ? something(st.nsteps, typemax(Int)) : nfix
    t_end = adaptive ? st.duration : sim.t + (nmax - n) * dt_fixed      # for the progress line
    (t_end === nothing || t_end > sim.t) || (t_end = nothing)           # resumed at the end
    feed = get(dom.drivers, :boundary, nothing)
    (feed === nothing || t_end === nothing) || check_coverage(feed, sim.t, t_end)
    outdir = isempty(cfg.output.variables) ? nothing :
             cfg.output.path === nothing ? output_dir(rf) : mkpath(cfg.output.path)
    cadence = cfg.output.cadence
    next_out = outdir === nothing ? Inf : (n == 0 ? 0.0 : (floor(sim.t / cadence) + 1) * cadence)
    done() = n >= nmax || (adaptive && st.duration !== nothing && sim.t >= st.duration * (1 - 1e-12))
    cpl = sim_couplings(r)
    n == 0 && _record_initial!(cpl, sim)
    # attached (#336): the step cores report every step; monitored_step!
    # turns a step that throws into a diagnosed blowup
    mon = merge(closure === nothing ? (;) : (; sgs = closure), _coupling_components(cpl))
    with_monitor(sim; run = rf, components = mon, t_end = t_end,
                 cadence = cfg.logging.diagnostics_cadence, checkpoint_components = comps,
                 stepper = r.stepper, step0 = n) do m
        if outdir !== nothing && sim.t >= next_out * (1 - 1e-9)
            _write_output(r, outdir, n); next_out += cadence
        end
        while !done()
            n += 1
            _, dt, retries = monitored_step!(m, sim; step = n) do
                adaptive ? _adaptive_step!(r, st.duration) : _take_step!(r, dt_fixed)
            end
            record_step!(r.stepper, dt; retries = retries)
            if outdir !== nothing && sim.t >= next_out * (1 - 1e-9)
                _write_output(r, outdir, n)
                next_out = (floor(sim.t / cadence * (1 + 1e-9)) + 1) * cadence
            end
            maybe_checkpoint!(ck, sim; step = n, components = comps, stepper = r.stepper)
            callback === nothing || callback(r, n)
        end
        # the guard checks on its cadence; a run never ends `completed` with
        # non-finite fields between two checks
        check_blowup!(m.guard, sim; step = n, force = true)
    end
    _write_tracks(cpl, cpl.particles === nothing ? nothing :
                               cfg.output.path === nothing ? output_dir(rf) : mkpath(cfg.output.path))
    return r
end

"""
    state_fields(r::SimulationRun) -> NamedTuple

The prognostic arrays with their halos: `u, v, w, p, θ` (and, for the
Boussinesq core, `qv` and the eddy coefficients `νt, κt` of the last
step, as a checkpoint stores them), and those of the couplings: the slab
temperature `T_s`, `ql`, `qr`, `tracer` and the particle positions
(`particles`, 3 × n).
"""
function Config.state_fields(r::SimulationRun)
    s = simulation(r)
    s isa BoussinesqSimulation && return merge(map(f -> f.data, IO.checkpoint_fields(s)), _coupling_fields(sim_couplings(r)))
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data, θ = s.θ.data)
end

"""
    preprocess_dynamic(::SimulationDriverType, cfg; from = :era5, files, box = nothing, out = nothing)
        -> Vector{String}

`aeolus preprocess dynamic --from-era5 <files…> --config <config.toml>
[--box deg]` (CFG-h, #363): the dynamic driver of the run `cfg` from the
ERA5 `files` ([`read_era5`](@ref Aeolus.InputDrivers.read_era5),
[`write_era5_driver`](@ref Aeolus.InputDrivers.write_era5_driver)) at the
domain's `dynamic_driver` (or `out`): the domain centre at `[preprocess.era5]`
`lat`, `lon`; its `levels`, `outputs` and `box` (`box` here replaces the
config's; default 0.5°); on the run's level centres, with one boundary
record per ERA5 time on the open faces of the run (`boundaries.x`/`y =
"open"`).
"""
function Config.preprocess_dynamic(::SimulationDriverType, cfg::RunConfig; from::Symbol, out = nothing,
                                   child = nothing, files = String[], box = nothing)
    from === :era5 || throw(ArgumentError("the model's dynamic preprocessor reads ERA5 (--from-era5), not :$from"))
    child === nothing || throw(ArgumentError("--from-era5 takes no child config"))
    isempty(files) && throw(ArgumentError("--from-era5: no ERA5 files"))
    t = cfg.preprocess.era5
    isempty(t) && throw(ConfigError("preprocess.era5", "the ERA5 preprocessor needs [preprocess.era5] with lat and lon (the domain centre)"))
    length(config_domains(cfg)) == 1 || throw(ArgumentError("--from-era5 writes the dynamic driver of a single-domain config"))
    path = out !== nothing ? abspath(out) : cfg.domain.dynamic_driver
    path === nothing && throw(ArgumentError("the domain's dynamic_driver is \"none\": pass --out file.nc"))
    mesh = _mesh(cfg.grid)
    mesh isa RegularGrid || throw(ConfigError("grid.stretching", "the ERA5 preprocessor needs a uniform grid"))
    planes = _is_open(cfg) ? only(open_boundary_states(_open_bcs(cfg, mesh))).planes : nothing
    e = read_era5(files; levels = Symbol(get(t, "levels", "pressure")))
    outs = haskey(t, "outputs") ? Tuple(Symbol.(t["outputs"])) : ERA5_OUTPUTS
    write_era5_driver(path, e; lat = Float64(t["lat"]), lon = Float64(t["lon"]), mesh, planes,
                      box = box === nothing ? Float64(get(t, "box", 0.5)) : Float64(box), outputs = outs)
    return [path]
end

register_driver!(SimulationDriverType())

end # module
