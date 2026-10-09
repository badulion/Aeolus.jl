"""
    Aeolus.SimulationDriver

The `simulation` run driver (CFG-b, #357): a bare run of the Boussinesq
or the anelastic core on a box, configured from TOML (`[run] model =
"simulation"`). See [`SimulationDriverType`](@ref) and the Run
configuration page; shipped configs are in `examples/configs/`.
"""
module SimulationDriver

# The `simulation` driver (CFG-b, #357): a bare run of one dynamical core
# on a box, configured entirely from TOML. It covers what the scripts
# examples/taylor_green_vortex.jl, dry_convective_bl.jl and
# buoyant_convection.jl used to build in code; their configs are
# examples/configs/*.toml.
#
# Its own module because it needs the dynamical cores (Simulations,
# Anelastic, TimeIntegration, StretchedGrids), which are included after
# Aeolus.Config; included at the end of Aeolus.jl.

using Random: Xoshiro, randn
using ..Grids: RegularGrid, CellCenter
using ..Fields: Field, set!, interior_view
using ..BoundaryConditions: FieldBoundaryConditions, open_boundary_conditions, open_boundary_states,
                            set_boundary_data!
using ..InputDrivers: InputDrivers, read_dynamic_driver, read_era5, write_era5_driver, check_planes,
                      check_coverage, advance_feed!, FileBoundaryFeed, ERA5_OUTPUTS
using ..Simulations: Simulations, BoussinesqSimulation, step!, step_rk3!
using ..TimeIntegration: CFLConstraints, adaptive_dt, step_adaptive!
using ..StretchedGrids: StretchedGrid
using ..Anelastic: AnelasticSimulation, MassWeightedSimulation, isentropic_reference
using ..Operators: StaticSmagorinsky, DynamicSmagorinsky, VremanClosure, SigmaModel, AMDClosure,
                   StabilityCorrected
using ..TurbulenceStability: BrownEtAl1994, LillyCutoff, ShortTail, LongTail
using ..IO: IO, RunFolder, CheckpointSchedule, StepperState, record_step!, previous_dt, resume!,
            maybe_checkpoint!, with_monitor, monitored_step!, check_blowup!, output_dir
using ..NCIO: nc_create, nc_defvar!
using ..Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, OpenTable, Optional,
                Path, section, option!, register_driver!

export SimulationDriverType, SimulationRun, profile_value

"""
    SimulationDriverType()

The `simulation` driver ([`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver)):
`[run] model = "simulation"`. Its sections are `[grid]`, `[dynamics]`,
`[boundaries]`, `[sgs]`, `[initial]`, `[stepper]`, `fields` in
`[output]` and `[preprocess]`, and the couplings around the step
(EX-d, #441): `[forcing]`, `[surface]`, `[microphysics]`, `[tracer]`,
`[particles]` and `[obstacles]` ([`SimCouplings`](@ref)); `aeolus
options simulation` prints them. Shipped configs: `examples/configs/`,
`examples/kernels/configs/`, `examples/coupled/configs/`.

`[initial] dynamic` names a dynamic driver (CFG-g, #362): its initial
profiles set u, v, θ, qv, and open lateral faces (`boundaries.x` or `y =
"open"`) take its boundary planes, blended linearly in time. `aeolus
preprocess dynamic --from-era5 <files…> --config <config.toml>` writes
one from ERA5 (CFG-h, #363, [`preprocess_dynamic`](@ref
Aeolus.Config.preprocess_dynamic)) with the site and outputs of
`[preprocess.era5]`.
"""
struct SimulationDriverType <: AbstractRunDriver end

const CONFIG_DIR = normpath(joinpath(@__DIR__, "..", "..", "examples", "configs"))

Config.driver_name(::SimulationDriverType) = "simulation"
Config.config_dir(::SimulationDriverType) = CONFIG_DIR

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

function Config.driver_sections(::SimulationDriverType)
    grid = section("grid"; doc = "The box and its mesh.") do s
        option!(s, "size", NTuple{3,Int}; default = (32, 32, 32), range = (1, 65536),
                doc = "cells in x, y, z")
        option!(s, "extent", NTuple{3,Float64}; default = (1.0, 1.0, 1.0), range = (0, Inf),
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
        option!(s, "theta0", Float64; default = 0.0, range = (0, Inf), units = "K",
                doc = "reference potential temperature of the buoyancy; 0 turns buoyancy off (boussinesq); anelastic: the base state's θ (> 0)")
        option!(s, "g", Float64; default = 9.81, range = (0, Inf), units = "m s^-2",
                doc = "gravitational acceleration")
        option!(s, "f_coriolis", Float64; default = 0.0, units = "s^-1", doc = "Coriolis parameter")
        option!(s, "moisture", Bool; default = false,
                doc = "carry the water-vapour mixing ratio qv (boussinesq only)")
        option!(s, "sponge_alpha_max", Float64; default = 0.0, range = (0, Inf), units = "s^-1",
                doc = "Rayleigh-damping rate at the top; 0: no sponge")
        option!(s, "sponge_z_start", Optional{Float64}; units = "m",
                doc = "height where the sponge starts; \"none\": the top (no sponge)")
    end
    bnd = section("boundaries"; doc = "Boundary conditions per axis.") do s
        for ax in ("x", "y", "z")
            option!(s, ax, Symbol; default = :periodic,
                    allowed = ax == "z" ? [:periodic, :free_slip, :noslip] : [:periodic, :free_slip, :noslip, :open],
                    doc = "the $ax faces: periodic, free_slip (impermeable, stress free) or noslip (z only); θ and qv have no flux through a wall, except the prescribed θ fluxes" *
                          (ax == "z" ? "" : "; open: the boundary planes of initial.dynamic, blended linearly in time (boussinesq, uniform grid, z walls)"))
        end
        option!(s, "theta_flux_bottom", Float64; default = 0.0, units = "K m s^-1",
                doc = "prescribed kinematic θ flux w'θ' through the bottom (z walls)")
        option!(s, "theta_flux_top", Float64; default = 0.0, units = "K m s^-1",
                doc = "prescribed kinematic θ flux through the top (z walls)")
    end
    sgs = section("sgs"; doc = "The subgrid-scale closure.") do s
        option!(s, "closure", Symbol; default = :none,
                allowed = [:none, :smagorinsky, :dynamic_smagorinsky, :vreman, :sigma, :amd],
                doc = "none, smagorinsky (Smagorinsky-Lilly), dynamic_smagorinsky (Germano-Lilly, plane averaged), vreman, sigma (Nicoud et al. 2011), amd (anisotropic minimum dissipation)")
        option!(s, "c_s", Optional{Float64}; units = "1", range = (0, Inf), open_lower = true,
                doc = "the closure's coefficient; \"none\": its default (smagorinsky 0.17, vreman 0.17, sigma 1.35, amd 0.3); not used by dynamic_smagorinsky")
        option!(s, "pr_t", Float64; units = "1", default = 1 / 3, range = (0, Inf), open_lower = true,
                doc = "turbulent Prandtl number")
        option!(s, "stability", Symbol; default = :none,
                allowed = [:none, :brown_et_al_1994, :lilly_cutoff, :short_tail, :long_tail],
                doc = "Richardson-number stability correction of the eddy viscosity (StabilityCorrected); none: off")
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
        option!(s, "dynamic", Optional{Path};
                doc = "a dynamic driver (CFG-g, #362; aeolus preprocess dynamic --from-era5): its initial profiles set u, v, theta (and qv) — not together with the tables above — and open faces take its boundary planes; hashed under [inputs]")
        option!(s, "perturbation_amplitude", Float64; default = 0.0, range = (0, Inf), units = "K",
                doc = "white noise on θ, uniform in [-a/2, a/2], drawn from Xoshiro([run] seed)")
        option!(s, "perturbation_depth", Float64; default = Inf, range = (0, Inf), units = "m",
                doc = "the noise is added below this height")
    end
    stp = section("stepper"; doc = "Time stepping. Give dt or cfl, and end_time or nsteps.") do s
        option!(s, "scheme", Symbol; default = :euler, allowed = [:euler, :rk3],
                doc = "euler (step!, the forward-Euler projection) or rk3 (step_rk3!, SSP-RK3); anelastic: euler")
        option!(s, "dt", Optional{Float64}; range = (0, Inf), open_lower = true, units = "s",
                doc = "fixed step; with end_time it is shortened to end_time / ceil(end_time / dt)")
        option!(s, "cfl", Optional{Float64}; units = "1", range = (0, Inf), open_lower = true,
                doc = "adaptive step (step_adaptive!) with this advective Courant number")
        option!(s, "dt_max", Float64; default = Inf, range = (0, Inf), open_lower = true, units = "s",
                doc = "the largest adaptive step (with cfl; the step of a fluid at rest)")
        option!(s, "end_time", Optional{Float64}; range = (0, Inf), open_lower = true, units = "s",
                doc = "simulated time to run")
        option!(s, "nsteps", Optional{Int}; range = (0, typemax(Int)), doc = "steps to run")
    end
    out = section("output"; doc = "") do s
        option!(s, "fields", Vector{String}; default = String[], allowed = _OUTPUT_FIELDS,
                doc = "fields written to output/fields_<step>.nc every cadence (cell-index arrays, Float32); [] : none")
    end
    pre = section("preprocess"; doc = "Inputs of the preprocessors (aeolus preprocess dynamic, static).") do s
        option!(s, "obstacles", Optional{Path};
                doc = "a GeoJSON FeatureCollection of obstacle footprints in mesh metres with a `height` each: aeolus preprocess static writes obstacles.static from it (EX-d, #441)")
        option!(s, "era5", OpenTable(_era5_validator),
                doc = "ERA5 (--from-era5, CFG-h #363): {lat, lon (degrees, the domain centre; required), box = 0.5 (degrees, the half-width of the centred differences of the geostrophic wind and the advective tendencies), levels = \"pressure\" | \"model\", outputs = [\"profiles\", \"planes\", \"large_scale\", \"geostrophic\", \"subsidence\", \"advection\", \"surface\"] (the default: all)}")
    end
    return Section[grid, dyn, bnd, sgs, ini, stp, out, pre, _coupling_sections()...]
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

function _cross_checks(cfg)
    e = ConfigError[]
    st = cfg.stepper; d = cfg.dynamics; b = cfg.boundaries; s = cfg.sgs; g = cfg.grid
    (st.dt === nothing) == (st.cfl === nothing) &&
        push!(e, ConfigError("stepper.dt", "give exactly one of stepper.dt and stepper.cfl"))
    (st.end_time === nothing) == (st.nsteps === nothing) &&
        push!(e, ConfigError("stepper.end_time", "give exactly one of stepper.end_time and stepper.nsteps"))
    (st.cfl !== nothing && !isfinite(st.dt_max)) &&
        push!(e, ConfigError("stepper.dt_max", "an adaptive run (stepper.cfl) needs a finite dt_max: the CFL step of a fluid at rest is infinite"))
    for ax in ("x", "y")
        b[ax] === :noslip &&
            push!(e, ConfigError("boundaries.$ax", "noslip is a z-wall condition; use free_slip or periodic"))
    end
    if b.x === :open || b.y === :open
        ax = b.x === :open ? "x" : "y"
        cfg.initial.dynamic === nothing &&
            push!(e, ConfigError("boundaries.$ax", "open faces take the boundary planes of a dynamic driver: set initial.dynamic (aeolus preprocess dynamic --from-era5)"))
        (b.x in (:open, :periodic) && b.y in (:open, :periodic)) ||
            push!(e, ConfigError("boundaries.$ax", "with an open axis the other lateral axis is open or periodic"))
        b.z === :periodic && push!(e, ConfigError("boundaries.z", "an open box needs z walls"))
        d.core === :boussinesq || push!(e, ConfigError("dynamics.core", "open faces need the boussinesq core"))
        g.stretching === :uniform || push!(e, ConfigError("grid.stretching", "open faces need a uniform grid"))
        d.sponge_alpha_max == 0 ||
            push!(e, ConfigError("dynamics.sponge_alpha_max", "the sponge of an open box has no target here; use 0"))
    end
    if cfg.initial.dynamic !== nothing
        for (k, p) in (("u", cfg.initial.u), ("v", cfg.initial.v), ("theta", cfg.initial.theta), ("qv", cfg.initial.qv))
            isempty(p) || push!(e, ConfigError("initial.$k", "initial.dynamic sets the initial $k: give one of the two"))
        end
        d.core === :boussinesq || push!(e, ConfigError("initial.dynamic", "a dynamic driver initialises the boussinesq core"))
    end
    if d.core === :anelastic
        b.z === :periodic && push!(e, ConfigError("boundaries.z", "the anelastic core needs z walls"))
        d.theta0 > 0 || push!(e, ConfigError("dynamics.theta0", "the anelastic core needs theta0 > 0 (its base state's θ)"))
        d.moisture && push!(e, ConfigError("dynamics.moisture", "the anelastic core has no qv"))
        st.scheme === :euler || push!(e, ConfigError("stepper.scheme", "the anelastic core steps with euler"))
        st.cfl === nothing || push!(e, ConfigError("stepper.cfl", "the anelastic core takes a fixed dt"))
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
    (!d.moisture && !isempty(cfg.initial.qv)) &&
        push!(e, ConfigError("initial.qv", "a qv profile needs dynamics.moisture = true"))
    ("qv" in cfg.output.fields && !d.moisture) &&
        push!(e, ConfigError("output.fields", "qv is written only with dynamics.moisture = true"))
    for (k, p) in (("u", cfg.initial.u), ("v", cfg.initial.v))
        (get(p, "kind", "") == "taylor_green" && (b.x !== :periodic || b.y !== :periodic)) &&
            push!(e, ConfigError("initial.$k", "the Taylor-Green field needs periodic x and y"))
    end
    append!(e, _coupling_checks(cfg))
    return e
end

const _CROSS_CHECKS_DOC = """
exactly one of `stepper.dt` and `stepper.cfl`, and exactly one of `stepper.end_time` and `stepper.nsteps`
an adaptive step (`stepper.cfl`) needs a finite `stepper.dt_max`
`noslip` is a z-wall condition: not for `boundaries.x`, `boundaries.y`
the anelastic core needs z walls, `dynamics.theta0` > 0, no moisture, `stepper.scheme = "euler"`, a fixed `stepper.dt`, a uniform grid and no checkpoints
on a stretched grid `sgs.closure` is `none` or `smagorinsky` and `sgs.stability` is `none`
`sgs.stability` wraps `smagorinsky`, `dynamic_smagorinsky`, `vreman` or `sigma` only
an `initial.qv` profile and the `qv` output field need `dynamics.moisture = true`
a Taylor-Green initial `u` or `v` needs periodic x and y"""

Config.driver_checks(::SimulationDriverType) = Any[_CROSS_CHECKS_DOC * "\n" * _COUPLING_CHECKS_DOC => _cross_checks]

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

"""
    SimulationRun

What the `simulation` driver builds: `sim` (a `BoussinesqSimulation` or
an `AnelasticSimulation`), `closure` (the SGS closure passed to the
steps; `nothing` for none or the built-in Smagorinsky), `stepper` (a
[`StepperState`](@ref)), the resolved `cfg`, and `feed`, the
[`FileBoundaryFeed`](@ref Aeolus.InputDrivers.FileBoundaryFeed) of the
open faces (`nothing`: none), and `couplings`, the [`SimCouplings`](@ref)
of `[forcing]`, `[surface]`, `[microphysics]`, `[tracer]`, `[particles]`
and `[obstacles]` (EX-d, #441).
"""
struct SimulationRun
    sim::Any
    closure::Any
    stepper::StepperState
    cfg::RunConfig
    feed::Union{Nothing,FileBoundaryFeed}
    couplings::Any
end

function _mesh(g)
    if g.stretching === :uniform
        return RegularGrid(; size = g.size, extent = g.extent, halo = g.halo)
    end
    return StretchedGrid(; size = g.size, extent = g.extent, halo = g.halo,
                         stretching = (g.stretching, g.stretching_parameter))
end

_bc(s::Symbol) = s === :free_slip ? :wall : s
_scalar_bc(s::Symbol) = s === :periodic ? :periodic : :noflux

function _closure(s, mesh)
    s.closure === :none && return nothing
    cs = s.c_s; pr = s.pr_t
    base = s.closure === :smagorinsky ? StaticSmagorinsky(something(cs, 0.17); Pr_t = pr) :
           s.closure === :dynamic_smagorinsky ? DynamicSmagorinsky(mesh; Pr_t = pr) :
           s.closure === :vreman ? VremanClosure(something(cs, 0.17); Pr_t = pr) :
           s.closure === :sigma ? SigmaModel(something(cs, 1.35); Pr_t = pr) :
           AMDClosure(; C = something(cs, 0.3), Pr_t = pr)
    s.stability === :none && return base
    scheme = s.stability === :brown_et_al_1994 ? BrownEtAl1994() :
             s.stability === :lilly_cutoff ? LillyCutoff() :
             s.stability === :short_tail ? ShortTail() : LongTail()
    return StabilityCorrected(base, mesh; scheme = scheme)
end

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
    mesh = _mesh(cfg.grid)
    d = cfg.dynamics; b = cfg.boundaries; s = cfg.sgs
    dyn = cfg.initial.dynamic === nothing ? nothing :
          read_dynamic_driver(cfg.initial.dynamic; mesh = mesh, planes = _is_open(cfg))
    feed = nothing
    if _is_open(cfg)
        dyn.boundary === nothing &&
            throw(ConfigError("initial.dynamic", "$(cfg.initial.dynamic) has no boundary planes, which the open faces need (write it with the run's open boundaries)"))
        bcs = θ_bcs = _open_bcs(cfg, mesh)
        feed = dyn.boundary
        check_planes(feed, only(open_boundary_states(bcs)).planes)
        set_boundary_data!(bcs, feed.data)
    else
        bcs = FieldBoundaryConditions(_bc(b.x), _bc(b.y), _bc(b.z))
        θ_bcs = FieldBoundaryConditions(_scalar_bc(b.x), _scalar_bc(b.y), _scalar_bc(b.z))
    end
    # the plain Smagorinsky closure is the simulation's built-in one (C_s ≠ 0),
    # bit for bit StaticSmagorinsky(C_s) (Simulations.step!)
    builtin = s.closure === :smagorinsky && s.stability === :none
    C_s = builtin ? something(s.c_s, 0.17) : 0.0
    closure = builtin ? nothing : _closure(s, mesh)
    z_top = mesh.z0 + mesh.Lz
    kw = (; ν = d.nu, g_accel = d.g, f_coriolis = d.f_coriolis, C_s = C_s, Pr_t = s.pr_t,
          θ_flux_bottom = b.theta_flux_bottom, θ_flux_top = b.theta_flux_top,
          sponge_α_max = d.sponge_alpha_max, sponge_z_start = something(d.sponge_z_start, z_top),
          bcs = bcs, θ_bcs = θ_bcs)
    sim = if d.core === :anelastic
        AnelasticSimulation(mesh; ref = isentropic_reference(mesh; θ_const = d.theta0, g_accel = d.g), kw...)
    else
        BoussinesqSimulation(mesh; θ₀ = d.theta0, moisture = d.moisture, kw...)
    end
    couplings = _couplings(cfg, mesh)
    # the slab sets the floor's flux every step: the core must apply it even
    # when it is 0 at construction (the flag is frozen there)
    couplings.slab === nothing || (sim.has_surface_flux = true)
    _initialise!(sim, cfg)
    dyn === nothing || _initialise_dynamic!(sim, dyn, cfg)
    couplings.water === nothing || isempty(cfg.initial.ql) || _set_profile!(couplings.water.ql, cfg.initial.ql, :ql)
    scheme = cfg.stepper.cfl === nothing ? cfg.stepper.scheme : :adaptive
    return SimulationRun(sim, closure, StepperState(scheme), cfg, feed, couplings)
end

# The initial profiles of a dynamic driver (linear in z, constant beyond
# the ends: `profile_value` of a table).
function _initialise_dynamic!(sim, dyn, cfg)
    P = dyn.profiles
    P === nothing && throw(ConfigError("initial.dynamic", "$(cfg.initial.dynamic) has no initial profiles (init_theta, …)"))
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
function _fixed_steps(st)
    if st.end_time !== nothing
        n = ceil(Int, st.end_time / st.dt)
        return st.end_time / n, n
    end
    return st.dt, st.nsteps
end

# The open faces' window: the boundary records around the step's start
# (the blend extrapolates linearly over the rest of a step that crosses a
# record).
_advance!(r::SimulationRun) = (r.feed === nothing || advance_feed!(r.feed, r.sim.t); r)

function _take_step!(r::SimulationRun, dt, step = 0)
    _advance!(r)
    sim = r.sim; sgs = r.closure; scheme = r.cfg.stepper.scheme; c = r.couplings
    _before_step!(c, sim, dt)
    bf = _body_force(c)
    if bf === nothing
        scheme === :rk3 ? step_rk3!(sim, dt; sgs = sgs) : step!(sim, dt; sgs = sgs)
    else
        scheme === :rk3 ? step_rk3!(sim, dt; sgs = sgs, body_force! = bf) : step!(sim, dt; sgs = sgs, body_force! = bf)
    end
    _after_step!(c, sim, dt, step)
    return (sim, Float64(dt), 0)
end

function _adaptive_step!(r::SimulationRun, c::CFLConstraints, t_end)
    _advance!(r)
    sim = r.sim; sgs = r.closure; scheme = r.cfg.stepper.scheme
    dt = adaptive_dt(sim, c; dt_prev = previous_dt(r.stepper), sgs = sgs)
    t_end === nothing || (dt = min(dt, t_end - sim.t))
    stepper = scheme === :rk3 ? ((s, h; kw...) -> step_rk3!(s, h; kw..., sgs = sgs)) : nothing
    return step_adaptive!(sim, c; dt = dt, sgs = sgs, stepper = stepper)
end

# output: the selected fields every `cadence` simulated seconds
const _FIELD_OF = Dict("u" => :u, "v" => :v, "w" => :w, "p" => :p, "theta" => :θ, "qv" => :qv)

function _write_output(r::SimulationRun, dir, step)
    sim = r.sim; m = sim.mesh
    path = joinpath(dir, string("fields_", lpad(step, 9, '0'), ".nc"))
    nc = nc_create(path; attributes = Dict("t" => Float64(sim.t), "step" => Int32(step),
                                           "note" => "cell-index arrays: face fields at their own staggered positions"))
    try
        nc.dim["x"] = m.Nx; nc.dim["y"] = m.Ny; nc.dim["z"] = m.Nz
        c = r.couplings
        for f in r.cfg.output.fields
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
    sim = r.sim; cfg = r.cfg; st = cfg.stepper
    resume!(ck, sim; sgs = r.closure, stepper = r.stepper)
    n = ck.last_step
    adaptive = st.cfl !== nothing
    dt_fixed, nfix = adaptive ? (NaN, typemax(Int)) : _fixed_steps(st)
    nmax = adaptive ? something(st.nsteps, typemax(Int)) : nfix
    t_end = adaptive ? st.end_time : sim.t + (nmax - n) * dt_fixed      # for the progress line
    (t_end === nothing || t_end > sim.t) || (t_end = nothing)           # resumed at the end
    (r.feed === nothing || t_end === nothing) || check_coverage(r.feed, sim.t, t_end)
    c = adaptive ? CFLConstraints(; C_advective = st.cfl, dt_max = st.dt_max) : nothing
    outdir = isempty(cfg.output.fields) ? nothing :
             cfg.output.path === nothing ? output_dir(rf) : mkpath(cfg.output.path)
    cadence = cfg.output.cadence
    next_out = outdir === nothing ? Inf : (n == 0 ? 0.0 : (floor(sim.t / cadence) + 1) * cadence)
    done() = n >= nmax || (adaptive && st.end_time !== nothing && sim.t >= st.end_time * (1 - 1e-12))
    comps = merge(r.closure === nothing ? (;) : (; sgs = r.closure), _coupling_components(r.couplings))
    n == 0 && _record_initial!(r.couplings, sim)
    # attached (#336): the step cores report every step; monitored_step!
    # turns a step that throws into a diagnosed blowup
    with_monitor(sim; run = rf, components = comps, t_end = t_end,
                 cadence = cfg.logging.diagnostics_cadence, sgs = r.closure,
                 stepper = r.stepper, step0 = n) do m
        if outdir !== nothing && sim.t >= next_out * (1 - 1e-9)
            _write_output(r, outdir, n); next_out += cadence
        end
        while !done()
            n += 1
            _, dt, retries = monitored_step!(m, sim; step = n) do
                adaptive ? _adaptive_step!(r, c, st.end_time) : _take_step!(r, dt_fixed, n)
            end
            record_step!(r.stepper, dt; retries = retries)
            if outdir !== nothing && sim.t >= next_out * (1 - 1e-9)
                _write_output(r, outdir, n)
                next_out = (floor(sim.t / cadence * (1 + 1e-9)) + 1) * cadence
            end
            maybe_checkpoint!(ck, sim; step = n, sgs = r.closure, stepper = r.stepper)
            callback === nothing || callback(r, n)
        end
        # the guard checks on its cadence; a run never ends `completed` with
        # non-finite fields between two checks
        check_blowup!(m.guard, sim; step = n, force = true)
    end
    _write_tracks(r.couplings, r.couplings.particles === nothing ? nothing :
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
    s = r.sim
    s isa BoussinesqSimulation && return merge(map(f -> f.data, IO.checkpoint_fields(s)), _coupling_fields(r.couplings))
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data, θ = s.θ.data)
end

"""
    preprocess_dynamic(::SimulationDriverType, cfg; from = :era5, files, box = nothing, out = nothing)
        -> Vector{String}

`aeolus preprocess dynamic --from-era5 <files…> --config <config.toml>
[--box deg]` (CFG-h, #363): the dynamic driver of the run `cfg` from the
ERA5 `files` ([`read_era5`](@ref Aeolus.InputDrivers.read_era5),
[`write_era5_driver`](@ref Aeolus.InputDrivers.write_era5_driver)) at
`[initial] dynamic` (or `out`): the domain centre at `[preprocess.era5]`
`lat`, `lon`; its `levels`, `outputs` and `box` (`box` here replaces the
config's; default 0.5°); on the run's level centres, with one boundary
record per ERA5 time on the open faces of the run (`boundaries.x`/`y =
"open"`).
"""
function Config.preprocess_dynamic(::SimulationDriverType, cfg::RunConfig; from::Symbol, out = nothing,
                                   child = nothing, files = String[], box = nothing)
    from === :era5 || throw(ArgumentError("the simulation driver's dynamic preprocessor reads ERA5 (--from-era5), not :$from"))
    child === nothing || throw(ArgumentError("--from-era5 takes no child config"))
    isempty(files) && throw(ArgumentError("--from-era5: no ERA5 files"))
    t = cfg.preprocess.era5
    isempty(t) && throw(ConfigError("preprocess.era5", "the ERA5 preprocessor needs [preprocess.era5] with lat and lon (the domain centre)"))
    path = out !== nothing ? abspath(out) : cfg.initial.dynamic
    path === nothing && throw(ArgumentError("[initial] dynamic is \"none\": pass --out file.nc"))
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
