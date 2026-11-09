"""
    Aeolus.SimulationDriver

The model's configuration and run (CFG-b, #357; ONE-a #450; the schema,
ONE-b #451): a config without `[run] legacy` builds a
[`Model`](@ref Aeolus.Components.Model) of the Boussinesq or the
anelastic core on a box with its components, and runs it. See
[`SimulationDriverType`](@ref) and the Run configuration page; shipped
configs are the TOML files of the example cases, `examples/<case>/<case>.toml`.
"""
module SimulationDriver

# The model's assembly from a config (CFG-b #357, ONE-a #450, ONE-b #451):
# the schema (run-level [time], [output], [checks]; per
# domain [grid], [dynamics], [boundaries], [initial] and one section per
# component, generated from the component schemas), the build of the Model
# and its run loop.
#
# Its own module because it needs the dynamical cores (Simulations,
# Anelastic, TimeIntegration, StretchedGrids) and the components, which are
# included after Aeolus.Config; included at the end of Aeolus.jl.

using Random: Xoshiro, randn
using ..Grids: RegularGrid, CellCenter
using ..Operators: PassotPouquetSpectrum, isotropic_turbulence_field!, spectrum_wavenumbers
using ..Fields: Field, set!, interior_view
using ..BoundaryConditions: FieldBoundaryConditions, FluxBC, open_boundary_conditions, open_boundary_states,
                            set_boundary_data!
using ..InputDrivers: InputDrivers, read_dynamic_driver, check_planes, check_coverage, read_static_tiles,
                      dynamic_forcing
using ..LargeScaleForcing: forcing_value
using ..StretchedGrids: surface_refined_faces
using ..Simulations: Simulations, BoussinesqSimulation, step!
using ..Backends: Backends
using ..TimeIntegration: CFLConstraints, _project!
using ..StretchedGrids: StretchedGrid
using ..Anelastic: AnelasticSimulation, MassWeightedSimulation, isentropic_reference
using ..Components: Components, Domain, Model, CentredAdvection, SGS, Coriolis, Sponge, SurfaceThetaFlux,
                    LargeScale, SurfaceLayer, Towers, ParentStatistics, KinematicFlow, Fringe,
                    build_components, checkpoint_components, sgs_closure, step_adaptive!, component_problems
using ..IO: IO, RunFolder, CheckpointSchedule, StepperState, record_step!, previous_dt, resume!,
            maybe_checkpoint!, with_monitor, monitored_step!, check_blowup!, output_dir
using ..RunLog: current_domain
using ..Checks: Checks, CheckRunner, enabled_checks, sample_checks!, evaluate_checks, stopped_checks,
                results_table, required_failures, advisory_failures, ChecksFailed, read_reference
using ..Output: OutputVariable, OutputWriter, select_variables, cell_centres, accumulate_output!, write_record!, write_series!,
                close_output!, output_means, restore_output_means!
using ..Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, OpenTable, OneOf, Optional,
                Path, section, option!, register_driver!, config_domains, domain_view, MODEL,
                slot_section, provider!
using ..Nesting: nested_grid, nest_bcs
# the urban site (ONE-e, #454): its sections, its build and its run
using ..Urban: Urban

export SimulationDriverType, SimulationRun, simulation, profile_value

"""
    SimulationDriverType()

The model ([`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver),
registered as `"model"`): what a config without `[run] legacy` runs
(schema version 2, ONE-b #451). Its run-level sections are `[time]`,
`[output]` (`variables`) and `[checks]`; its per-domain
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
take its boundary planes, blended linearly in time. AeolusPrep.jl makes
one from ERA5 (CFG-h #363, ONE-h #457: `aeolusprep dynamic --from-era5
<files…> --config <config.toml> --lat … --lon …`). A config with several
`[[domain]]` blocks is a nest (ONE-f, #455): the first domain is the
root, every later one names its `parent`, its `origin` and `refinement`
in it, and its `feedback` (`two_way` by default, or `one_way`); see
[`Components.nested_model`](@ref Aeolus.Components.nested_model) and the
Run configuration page.
"""
struct SimulationDriverType <: AbstractRunDriver end

Config.driver_name(::SimulationDriverType) = MODEL

"""
    COMPONENTS

The component types of the model whose sections the configuration schema
generates from their [`schema`](@ref Aeolus.Components.schema), in the
order of the option reference: the `:advection` and `:sgs` slots,
Coriolis, the sponge, the surface θ flux, the large-scale forcing, the
Monin–Obukhov surface layer (the `:surface` slot), the virtual towers
(ONE-g, #456), and the deep boundary-layer domain's tiled urban surface,
prescribed surface fluxes and column radiation of the plane-mean column
(NEST-p3, #483: the urban nest parent's physics). (The EX-d couplings are one
component, [`SimulationCouplings`](@ref), with six sections, until their
port to components, ONE-e #454.)
"""
const COMPONENTS = (CentredAdvection, SGS, Coriolis, Sponge, SurfaceThetaFlux, LargeScale, SurfaceLayer, Towers,
                    ParentStatistics, KinematicFlow, Fringe,
                    Urban.TiledUrbanSurface, Urban.PrescribedSurfaceFlux, Urban.MeanColumnRadiation)

_isnum(x) = x isa Real && !(x isa Bool)

# the couplings around the core's step (EX-d, #441)
include("simulation_couplings.jl")
# several [[domain]] blocks: nesting by domains (ONE-f, #455)
include("simulation_nesting.jl")

# ---------------------------------------------------------------------------
# initial profiles
# ---------------------------------------------------------------------------

const _PROFILE_KEYS = Dict("constant" => ["value"],
                           "mixed_layer_lapse" => ["value", "z_i", "lapse"],
                           "table" => ["z", "values"],
                           "exponential" => ["value", "scale_height"],
                           "taylor_green" => ["amplitude", "k"],
                           "box" => ["value", "x", "y", "z"],
                           "gaussian" => ["value", "amplitude", "center", "sigma"],
                           "tanh" => ["value", "amplitude", "center", "thickness"],
                           "seed_mode" => ["amplitude", "wavenumber", "center", "thickness"],
                           "geostrophic" => String[])

# the profile kinds of each variable: the vertical ones for all, the
# Taylor–Green field for u and v, a box for the cloud water (EX-d, #441)
_profile_kinds(var) = var == "w" ? ["seed_mode"] :
    sort!(vcat(["constant", "exponential", "mixed_layer_lapse", "table"],
               var in ("u", "v") ? ["taylor_green", "geostrophic"] : String[], var == "ql" ? ["box"] : String[],
               var == "theta" ? ["gaussian"] : String[], var in ("u", "v", "theta") ? ["tanh"] : String[]))

"The validator of the profile table of variable `var` (an empty table: the default profile)."
function _profile_validator(var::String)
    return function (t)
        isempty(t) && return String[]
        kind = get(t, "kind", nothing)
        kinds = _profile_kinds(var)
        kind in kinds || return ["kind: expected one of " * join(repr.(kinds), ", ") *
                                 ", got $(repr(kind))"]
        want = _PROFILE_KEYS[kind]
        opt = kind == "table" ? ["below"] : String[]
        probs = String[]
        for k in sort!(collect(keys(t)))
            k == "kind" || k in want || k in opt ||
                push!(probs, "$k: unknown key for kind = \"$kind\" (keys: $(join(vcat(want, opt), ", ")))")
        end
        haskey(t, "below") && append!(probs, _below_problems(t["below"]))
        for k in want
            haskey(t, k) || (push!(probs, "$k: required for kind = \"$kind\""); continue)
            v = t[k]
            if kind == "table"
                (v isa AbstractVector && !isempty(v) && all(_isnum, v)) ||
                    push!(probs, "$k: expected a non-empty array of numbers")
            elseif kind == "box" && k != "value"
                (v isa AbstractVector && length(v) == 2 && all(_isnum, v) && v[1] < v[2]) ||
                    push!(probs, "$k: expected [lower, upper] with lower < upper (m)")
            elseif kind == "gaussian" && k == "center"
                (v isa AbstractVector && length(v) == 2 && all(_isnum, v)) ||
                    push!(probs, "center: expected [x₀, y₀] (m)")
            else
                _isnum(v) && isfinite(v) || push!(probs, "$k: expected a finite number, got $(repr(v))")
            end
        end
        if kind == "table" && isempty(probs)
            length(t["z"]) == length(t["values"]) ||
                push!(probs, "z and values have different lengths ($(length(t["z"])) and $(length(t["values"])))")
            issorted(t["z"]; lt = <=) || push!(probs, "z: expected strictly increasing heights")
        end
        kind == "gaussian" && isempty(probs) && !(t["sigma"] > 0) && push!(probs, "sigma: expected a positive width (m)")
        kind in ("tanh", "seed_mode") && isempty(probs) && !(t["thickness"] > 0) &&
            push!(probs, "thickness: expected a positive thickness (m)")
        kind == "exponential" && isempty(probs) && !(t["scale_height"] > 0) &&
            push!(probs, "scale_height: expected a positive height (m)")
        return probs
    end
end

# the extension of a table profile below its lowest height
function _below_problems(b)
    b isa AbstractDict || return ["below: expected {kind = \"log\", z0} or {kind = \"linear\", z, value}"]
    kind = get(b, "kind", nothing)
    want = kind == "log" ? ["z0"] : kind == "linear" ? ["z", "value"] :
           return ["below.kind: expected \"log\" or \"linear\", got $(repr(kind))"]
    probs = String[]
    for k in sort!(collect(keys(b)))
        k == "kind" || k in want || push!(probs, "below.$k: unknown key for kind = \"$kind\" (keys: $(join(want, ", ")))")
    end
    for k in want
        v = get(b, k, nothing)
        (_isnum(v) && isfinite(v)) || push!(probs, "below.$k: required, a finite number")
    end
    kind == "log" && isempty(probs) && !(b["z0"] > 0) && push!(probs, "below.z0: expected a positive roughness length (m)")
    return probs
end

const _NOISE_KEYS = ("kind", "amplitude", "depth", "components", "taper", "project")

"The validator of `[initial] velocity_perturbation` (an empty table: none)."
function _velocity_noise_validator(t)
    isempty(t) && return String[]
    probs = String[]
    for k in sort!(collect(keys(t)))
        k in _NOISE_KEYS || push!(probs, "$k: unknown key (keys: $(join(_NOISE_KEYS, ", ")))")
    end
    get(t, "kind", nothing) in ("uniform", "normal") ||
        push!(probs, "kind: expected \"uniform\" or \"normal\", got $(repr(get(t, "kind", nothing)))")
    a = get(t, "amplitude", nothing)
    (_isnum(a) && isfinite(a) && a >= 0) || push!(probs, "amplitude: required, a finite number ≥ 0 (m s⁻¹)")
    d = get(t, "depth", Inf)
    (_isnum(d) && d > 0) || push!(probs, "depth: expected a height > 0 (m)")
    c = get(t, "components", ["u", "v", "w"])
    (c isa AbstractVector && !isempty(c) && allunique(c) && all(x -> x in ("u", "v", "w"), c)) ||
        push!(probs, "components: expected a non-empty subset of [\"u\", \"v\", \"w\"]")
    get(t, "taper", "none") in ("none", "linear") ||
        push!(probs, "taper: expected \"none\" or \"linear\" (× (1 − z/depth))")
    get(t, "taper", "none") == "linear" && !isfinite(d) && push!(probs, "taper: \"linear\" needs a finite depth")
    get(t, "project", false) isa Bool || push!(probs, "project: expected true or false")
    return probs
end

const _FIELD_KEYS = ("kind", "spectrum", "u_rms", "k_peak", "seed")

"The validator of `[initial] field`, a named analytic initial field (an empty table: none)."
function _field_validator(t)
    isempty(t) && return String[]
    probs = String[]
    get(t, "kind", nothing) == "isotropic_turbulence" ||
        return ["kind: expected \"isotropic_turbulence\", got $(repr(get(t, "kind", nothing)))"]
    for k in sort!(collect(keys(t)))
        k in _FIELD_KEYS || push!(probs, "$k: unknown key (keys: $(join(_FIELD_KEYS, ", ")))")
    end
    get(t, "spectrum", "passot_pouquet") == "passot_pouquet" ||
        push!(probs, "spectrum: expected \"passot_pouquet\" (E ∝ k⁴ exp(-2 (k/k_peak)²))")
    for k in ("u_rms", "k_peak")
        v = get(t, k, nothing)
        (_isnum(v) && isfinite(v) && v > 0) || push!(probs, "$k: required, a number > 0")
    end
    s = get(t, "seed", 0)
    (s isa Integer && !(s isa Bool) && s >= 0) || push!(probs, "seed: expected an integer ≥ 0")
    return probs
end

"""
    initial_field!(sim, t::AbstractDict, seed) -> sim

Set the velocity of `sim` to the named analytic initial field `t` of
`[initial] field`: `{kind = "isotropic_turbulence", spectrum =
"passot_pouquet", u_rms, k_peak, seed}` is
`Operators.isotropic_turbulence_field!` with the
`Operators.PassotPouquetSpectrum(u_rms, k_peak)` (k_peak in rad m⁻¹) from
the field's `seed` (default: `seed`, the run's): defined in spectral
space, so a coarser grid on the same box holds exactly the large scales
of a finer one (an LES and its DNS start alike). The field is
divergence-free in the continuous sense; it is then projected once onto
the grid's discrete divergence-free fields (the core's Chorin
projection), which changes a mode by O((kΔ)²) of its divergent part: the
SSP-RK3 step carries a third of the step-start state into the next one,
so a start that is not discretely solenoidal would keep its divergence.
"""
function initial_field!(sim, t::AbstractDict, seed::Integer)
    E = PassotPouquetSpectrum(Float64(t["u_rms"]), Float64(t["k_peak"]))
    isotropic_turbulence_field!(sim.u, sim.v, sim.w, E; seed = get(t, "seed", seed))
    _project!(sim, 1.0)
    return sim
end

"""
    profile_value(p::AbstractDict, z) -> Float64

The value at height `z` of a vertical profile of `[initial]`:

  * `{kind = "constant", value}`;
  * `{kind = "mixed_layer_lapse", value, z_i, lapse}`: `value` below `z_i`,
    `value + lapse (z − z_i)` above (a capping inversion; with `z_i` at the
    ground, a linear profile);
  * `{kind = "table", z = […], values = […]}`: linear interpolation,
    constant beyond the ends; below the lowest height, with `below =
    {kind = "log", z0}`, the log law from zero at `z0` to the lowest value
    (`v₁ ln(z/z0)/ln(z₁/z0)`), or with `below = {kind = "linear", z, value}`
    linear from `value` at `z` to the lowest value;
  * `{kind = "exponential", value, scale_height}`: `value exp(−z/scale_height)`;
  * `{kind = "tanh", value, amplitude, center, thickness}`: `value +
    amplitude tanh((z − center)/thickness)`, a shear layer.
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
        if z < zs[1] && haskey(p, "below")
            b = p["below"]
            if b["kind"] == "log"                 # from zero at z0 to the lowest tabulated value
                z0 = Float64(b["z0"])
                return vs[1] * (log(max(z, z0) / z0) / log(zs[1] / z0))
            end
            zb = Float64(b["z"]); yb = Float64(b["value"])   # linear from (z, value) to the lowest one
            return yb + (vs[1] - yb) * (max(z, zb) - zb) / (zs[1] - zb)
        end
        z <= zs[1] && return vs[1]
        z >= zs[end] && return vs[end]
        i = searchsortedlast(zs, z)
        w = (z - zs[i]) / (zs[i+1] - zs[i])
        return (1 - w) * vs[i] + w * vs[i+1]
    elseif kind == "exponential"
        return Float64(p["value"]) * exp(-z / Float64(p["scale_height"]))
    elseif kind == "tanh"
        return Float64(p["value"]) + Float64(p["amplitude"]) * tanh((z - Float64(p["center"])) / Float64(p["thickness"]))
    end
    throw(ArgumentError("profile_value: kind $(repr(kind)) is not a vertical profile"))
end

# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------

"The per-domain sections of the model: the domain's own keys, its grid, core, boundaries, initial state and components."
const DOMAIN_SECTIONS = unique(vcat(["domain", "grid", "core", "timestep", "dynamics", "boundaries", "initial"],
                                    [Components.section(C) for C in COMPONENTS],
                                    ["forcing", "surface", "microphysics", "tracer", "particles", "obstacles"],
                                    Urban.urban_section_names()))

Config.domain_sections(::SimulationDriverType) = DOMAIN_SECTIONS

"The names of every check the model can publish: the core's and those of the component types of [`COMPONENTS`](@ref)."
check_names_all() = vcat(collect(String, Components.CORE_CHECKS), String[n for C in COMPONENTS for n in Components.check_names(C)],
                         Urban.urban_check_names())

const _CHECK_DOCS = merge(
    Dict(n => "the closure of the $(join(v.qs, " and ")) budget, (|closure| + |unattributed|) / elapsed " *
              "($(v.units); default tolerance $(v.tolerance)); turns the budget on (as $(v.output) does, same requirements)"
         for (n, v) in Components.CLOSURE_CHECKS),
    Dict("finite" => "the sanity check: the number of non-finite interior values of the prognostic fields at the " *
                     "end (default tolerance 0); a run that blows up records it failed"),
    Dict(n => "a check of [$(Components.section(C))]" for C in COMPONENTS for n in Components.check_names(C)),
    Dict("urban_budgets" => "an urban site's budgets closed against Urban.BUDGET_TOLERANCES: the largest closure relative to its tolerance (default tolerance 1)"),
    Dict("$(Urban.component_name(C))_checks" => "the smoke checks of the urban component [$(Urban.component_name(C))] that fail (default tolerance 0)"
         for C in Urban.URBAN_COMPONENTS))

function _reference_validator(t)
    probs = String[]
    for q in sort!(collect(keys(t)))
        for p in Checks.reference_entry_problems(t[q])
            push!(probs, "$q.$p")
        end
    end
    return probs
end

function Config.driver_sections(::SimulationDriverType)
    dom = section("domain"; doc = "A domain: its id and its input drivers (CFG-f/g).") do s
        option!(s, "id", String; default = "main", doc = "the domain's name (unique; a config without [[domain]] has one, \"main\")")
        option!(s, "static_driver", Optional{Path};
                doc = "the static driver (AeolusPrep.jl: aeolusprep static --source <obstacles.geojson>): its buildings are the obstacles of [obstacles]; hashed under [inputs]")
        option!(s, "dynamic_driver", Optional{Path};
                doc = "the dynamic driver (CFG-g, #362; AeolusPrep.jl: aeolusprep dynamic --from-era5): its initial profiles set u, v, theta (and qv) — not together with the [initial] tables — and open faces take its boundary planes; hashed under [inputs]")
        option!(s, "parent", Optional{String};
                doc = "the id of the domain this one is nested in (ONE-f, #455): an earlier [[domain]] block; \"none\": the root, the first domain")
        option!(s, "origin", Optional{NTuple{2,Float64}}; units = "m",
                doc = "a nested domain: its lower-left corner (x, y) in its parent's coordinates; it stands on the parent's ground, grid.extent its size")
        option!(s, "refinement", Optional{NTuple{3,Int}}; range = (1, 64),
                doc = "a nested domain: its parent's spacing over its own, per axis (grid.size = grid.extent / parent spacing × refinement)")
        option!(s, "feedback", Symbol; default = :two_way, allowed = [:two_way, :one_way],
                doc = "a nested domain: two_way (the self-nesting default: anterpolation onto the parent, flux correction, refluxing) or one_way (the parent never sees it)")
    end
    grid = section("grid"; doc = "The box and its mesh.") do s
        option!(s, "size", NTuple{3,Int}; default = (32, 32, 32), range = (1, 65536),
                doc = "cells in x, y, z")
        option!(s, "extent", NTuple{3,Float64}; default = (3200.0, 3200.0, 1600.0), range = (0, Inf),
                open_lower = true, units = "m", doc = "domain size Lx, Ly, Lz")
        option!(s, "halo", NTuple{3,Int}; default = (2, 2, 2), range = (1, 8),
                doc = "halo cells per side")
        option!(s, "stretching", Symbol; default = :uniform,
                allowed = [:uniform, :geometric, :tanh, :tanh_both, :surface_refined],
                doc = "vertical spacing (StretchedGrid): uniform (a RegularGrid), geometric (dz[k+1] = r dz[k]), tanh (clustered at the bottom), tanh_both (at both walls), " *
                      "surface_refined (dz_surface up to z_uniform, then × stretching_parameter per level up to dz_max: StretchedGrids.surface_refined_faces; size[3] is its level count)")
        option!(s, "stretching_parameter", Float64; units = "1", default = 1.0, range = (0, Inf), open_lower = true,
                doc = "r of geometric and surface_refined, β of tanh / tanh_both; unused for uniform")
        option!(s, "dz_surface", Float64; default = NaN, units = "m",
                doc = "surface_refined: the uniform spacing of the lowest levels")
        option!(s, "z_uniform", Float64; default = NaN, units = "m",
                doc = "surface_refined: the top of the uniform levels")
        option!(s, "dz_max", Float64; default = Inf, range = (0, Inf), open_lower = true, units = "m",
                doc = "surface_refined: the largest spacing")
        option!(s, "origin", NTuple{2,Float64}; default = (0.0, 0.0), units = "m",
                doc = "the root domain's lower-left corner (x, y); a nested domain's is its parent's plus its domain.origin")
    end
    core = slot_section("core"; key = "model", doc = "The dynamical core.") do s
        provider!(s, "boussinesq"; default = [:model], doc = "BoussinesqSimulation.")
        provider!(s, "anelastic"; doc = "AnelasticSimulation on an isentropic base state (dynamics.theta0).")
    end
    dyn = section("dynamics"; doc = "The parameters of the dynamical core.") do s
        option!(s, "nu", Float64; default = 0.0, range = (0, Inf), units = "m^2 s^-1",
                doc = "kinematic viscosity; θ and qv diffuse with the same coefficient (Prandtl number 1)")
        option!(s, "theta0", Float64; default = 300.0, range = (0, Inf), units = "K",
                doc = "reference potential temperature of the buoyancy; 0 turns buoyancy off (boussinesq); anelastic: the base state's θ (> 0)")
        option!(s, "g", Float64; default = 9.81, range = (0, Inf), units = "m s^-2",
                doc = "gravitational acceleration")
        option!(s, "moisture", Bool; default = false,
                doc = "carry the water-vapour mixing ratio qv (boussinesq only)")
        option!(s, "p_surface", Optional{Float64}; range = (0, Inf), open_lower = true, units = "Pa",
                doc = "the ground pressure of the hydrostatic reference state of the boussinesq core (from the initial " *
                      "theta profile): the components that convert fluxes with one ([surface] model = \"tiles\" or " *
                      "\"prescribed\", [radiation]) and the absolute temperature outputs (temperature, " *
                      "temperature_2m, surface_temperature); \"none\": the dynamic driver's, else 101325 Pa")
    end
    bnd = section("boundaries"; doc = "Boundary conditions per axis.") do s
        for ax in ("x", "y", "z")
            option!(s, ax, Symbol; default = ax == "z" ? :free_slip : :periodic,
                    allowed = ax == "z" ? [:periodic, :free_slip, :noslip] : [:periodic, :free_slip, :noslip, :open],
                    doc = "the $ax faces: periodic, free_slip (impermeable, stress free) or noslip (z only); θ and qv have no flux through a wall, except the prescribed θ fluxes ([surface_theta_flux])" *
                          (ax == "z" ? "" : "; open: the boundary planes of the domain's dynamic_driver, blended linearly in time (boussinesq, uniform grid, z walls)"))
        end
        option!(s, "theta_top", Symbol; default = :noflux, allowed = [:noflux, :initial_gradient],
                doc = "θ at a top wall: noflux, or initial_gradient: the fixed gradient of the initial θ profile at the top face " *
                      "(a flux condition, the deep domain's top of #254)")
    end
    ini = section("initial"; doc = "The initial state.") do s
        for (v, d) in (("u", "0"), ("v", "0"), ("theta", "theta0 (anelastic: the base state)"), ("qv", "0"))
            option!(s, v, OpenTable(_profile_validator(v)),
                    doc = "initial $v: {kind = \"constant\", value}, {kind = \"mixed_layer_lapse\", value, z_i, lapse}, " *
                          "{kind = \"table\", z, values}" *
                          (v == "theta" ? ", {kind = \"gaussian\", value, amplitude, center = [x₀, y₀], sigma} (value + amplitude exp(−((x − x₀)² + (y − y₀)²)/(2σ²)), a column)" : "") *
                          (v == "qv" ? "" : ", {kind = \"tanh\", value, amplitude, center, thickness} (value + amplitude tanh((z − center)/thickness), a shear layer)") *
                          (v in ("u", "v") ? ", {kind = \"taylor_green\", amplitude, k} (u = A sin kx cos ky cos kz, v = -A cos kx sin ky cos kz), " *
                                             "or {kind = \"geostrophic\"} (the [large_scale_forcing] geostrophic wind at the start)" : "") *
                          (v in ("theta", "qv") ? "; with a sounding driver as the domain's dynamic_driver, {} is its profile at the start" : "") *
                          "; {} (the default): $d")
        end
        option!(s, "w", OpenTable(_profile_validator("w")),
                doc = "initial w (at the z faces): {kind = \"seed_mode\", amplitude, wavenumber, center, thickness}, " *
                      "w = amplitude sin(wavenumber x) exp(−((z − center)/thickness)²), a normal-mode seed of a shear " *
                      "layer (not projected); {} (the default): 0")
        option!(s, "ql", OpenTable(_profile_validator("ql")),
                doc = "initial cloud water (microphysics.scheme = \"kessler\"): a vertical profile as for theta, or " *
                      "{kind = \"box\", value, x = [x₀, x₁], y = [y₀, y₁], z = [z₀, z₁]} (value strictly inside the box, 0 outside); {}: 0")
        option!(s, "velocity_perturbation", OpenTable(_velocity_noise_validator),
                doc = "noise added to u, v and w (in that order, after the θ noise, from the same Xoshiro([run] seed)): " *
                      "{kind = \"uniform\", amplitude} (uniform in [-a/2, a/2], one draw per cell) or " *
                      "{kind = \"normal\", amplitude} (a × standard normal, one randn array per field); optional: " *
                      "depth (m; only below it, uniform: one draw per cell below it), components (a subset of " *
                      "[\"u\", \"v\", \"w\"], default all), taper (\"linear\": × (1 − z/depth); \"none\") and project " *
                      "(true: the velocity is projected once onto a divergence-free field afterwards, boussinesq); {}: none")
        option!(s, "field", OpenTable(_field_validator),
                doc = "a named analytic initial velocity field: {kind = \"isotropic_turbulence\", spectrum = " *
                      "\"passot_pouquet\", u_rms (m s⁻¹), k_peak (rad m⁻¹), seed} — decaying isotropic turbulence " *
                      "defined in spectral space (Operators.isotropic_turbulence_field!), so a coarser grid on the same " *
                      "box starts from exactly the large scales of a finer one; seed: the field's (default [run] seed); " *
                      "a triply periodic regular grid; {}: none")
        option!(s, "perturbation_kind", Symbol; default = :uniform, allowed = [:uniform, :normal],
                doc = "the θ noise: uniform in [-a/2, a/2] or a × standard normal, one draw per cell below the depth")
        option!(s, "perturbation_amplitude", Float64; default = 0.0, range = (0, Inf), units = "K",
                doc = "white noise on θ (perturbation_kind), drawn from Xoshiro([run] seed)")
        option!(s, "perturbation_depth", Float64; default = Inf, range = (0, Inf), units = "m",
                doc = "the noise is added below this height")
    end
    stepping = slot_section("timestep"; key = "scheme", doc = "The time-stepping scheme.") do s
        provider!(s, "rk3"; default = [:model], doc = "step_rk3!, SSP-RK3, on either core.")
        provider!(s, "euler"; doc = "step!, the forward-Euler projection, on either core.")
        option!(s, "loop_fusion", Symbol; default = :if_possible, allowed = [:if_possible, :always, :never],
                doc = "compute all tendency terms and the predictor in one cache-blocked pass over the grid " *
                      "(loop fusion, #533; PALM's loop_optimization = 'cache'): if_possible (fused when every " *
                      "term has a fused form, otherwise one loop per term, and results.toml [loop_fusion] names " *
                      "the terms without one), always (fused, or the run stops at its first step naming them), " *
                      "never (one loop per term, the reference). Results agree within round-off. The step " *
                      "core's fusion is Boussinesq only: an anelastic core runs one loop per term and refuses " *
                      "always. The setting also chooses the update of a prognostic TKE closure on either core " *
                      "(#554) and of a Reynolds-stress closure with transport (#560): one cache-blocked pass unless " *
                      "never; results.toml tke_update_blocked / reynolds_stress_update_blocked says which ran")
    end
    tim = section("time"; doc = "The clock: a fixed dt or adaptive steps, for a duration or a number of steps.") do s
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
        option!(s, "start", Float64; default = 0.0, units = "s",
                doc = "the model time at the start: the run ends at start + duration; the dynamic driver's times " *
                      "(s since its init_time) are read at the model time − start, a [checks.reference] window is in model time")
    end
    spn = section("spinup"; doc = "The surface spin-up before the run (#519; PALM's spinup_time): the slow " *
                                  "surfaces (soil, ground, roofs, building envelopes) advance alone over `time` " *
                                  "seconds before the start, at negative model times, against a prescribed " *
                                  "near-surface air and the sun, without the 3-D atmosphere, which starts as " *
                                  "initialised. A fresh run only (a restart has spun up already). An urban site's surfaces.") do s
        option!(s, "time", Float64; default = 0.0, range = (0, Inf), units = "s",
                doc = "the spin-up's length; 0: none. PALM recommends ≥ 24 h (Maronga et al. 2020); the slowest " *
                      "modes of the soil (≈ 30 d) are not removed by any spin-up of hours (#521: initial soil profiles)")
        option!(s, "dt", Optional{Float64}; range = (0, Inf), open_lower = true, units = "s",
                doc = "the surfaces' step during the spin-up; \"none\": their step in the run (an urban site: " *
                      "surface_period or soil_period). It divides the radiation period")
        option!(s, "forcing", Symbol; default = :solar, allowed = [:solar, :driver],
                doc = "the air the surfaces see: solar (PALM: θ = pt_mean + pt_amplitude cos(solar zenith), " *
                      "lagged 1 h) or driver (the dynamic driver's sfc_temperature series, which must cover " *
                      "the spin-up); the wind and humidity stay at the initial state")
        option!(s, "pt_mean", Optional{Float64}; range = (0, Inf), open_lower = true, units = "K",
                doc = "solar forcing: the mean potential temperature of the air; \"none\": the initial " *
                      "mean of the air cells next to the surfaces")
        option!(s, "pt_amplitude", Float64; default = 0.0, range = (0, Inf), units = "K",
                doc = "solar forcing: the amplitude multiplying cos(solar zenith) (negative at night)")
        option!(s, "soil_moisture", Bool; default = false,
                doc = "the soil water evolves during the spin-up (PALM's calc_soil_moisture_during_spinup); " *
                      "false: frozen at its initial state")
        option!(s, "buildings", Bool; default = true,
                doc = "the building energy model (zone, envelope, HVAC) runs during the spin-up " *
                      "(PALM's indoor_during_spinup, default off there)")
    end
    out = section("output"; doc = "") do s
        option!(s, "variables", Vector{String}; default = collect(String, Components.DEFAULT_OUTPUTS),
                doc = "the variables written, from the output catalogues of the core and the enabled components " *
                      "(ONE-c: 3-D fields, profiles and surface maps every cadence, time series every series_cadence; " *
                      "the core's: " * join(Components.CORE_OUTPUTS, ", ") * "; the components': ql, qr, tracer, " *
                      "[surface] model = \"most\"'s " * join(Components.output_names(SurfaceLayer, nothing), ", ") *
                      ", [towers]' tower_threshold); [] : none")
        option!(s, "averaging", Symbol; default = :instant, allowed = [:instant, :mean],
                doc = "instant: the values at each record; mean: the time mean over the interval since the previous record")
        option!(s, "series_cadence", Float64; default = 60.0, range = (0, Inf), units = "s",
                doc = "simulated time between the samples of the time series (the scalar variables); 0: every step")
        option!(s, "netcdf", String; default = "output.nc",
                doc = "file name of the NetCDF-4 output in [output] path (the run folder's output/ by default)")
    end
    # one key per published check, `name = {required, tolerance}`, and the
    # reference comparisons (ONE-d, #453)
    chk = section("checks"; doc = "The checks the core and the enabled components publish, each enabled as " *
                                  "`name = {required, tolerance}`, and the reference comparisons against " *
                                  "`reference_file` (ONE-d, #453). The run writes their results to " *
                                  "results.toml; a failed required check ends `aeolus run` with exit code 3.") do s
        for n in check_names_all()
            option!(s, n, OpenTable(Checks.check_entry_problems); doc = _CHECK_DOCS[n])
        end
        option!(s, "reference_file", Optional{Path};
                doc = "the reference data of [checks.reference] (Checks.read_reference): a CSV of " *
                      "quantity,low,high (or quantity,value) rows, or of z and one column per profile; or a NetCDF file; hashed under [inputs]")
        option!(s, "reference", OpenTable(_reference_validator);
                doc = "the reference comparisons, `<quantity> = {required, tolerance, variable, window}`: the output " *
                      "variable `variable` (default: the quantity's name; a time-series scalar or a profile) against the " *
                      "quantity of reference_file — at the end of the run, or its time mean over the steps ending in " *
                      "`window = [t0, t1]` (s). A scalar fails outside the reference band [low, high] widened by " *
                      "`tolerance` (default 0); a profile when the RMS difference at the reference heights exceeds it. " *
                      "Results as check reference_<quantity>")
    end
    comps = Section[Components.schema(C) for C in COMPONENTS]
    # one section per exclusive slot (ONE-m, #476): the model's providers and
    # the urban site's of a slot pooled, each available on its domain kind
    return Config.merge_sections(Section[tim, spn, out, chk, dom, grid, core, stepping, dyn, bnd, ini, comps...,
                                         _coupling_sections()..., Urban.urban_sections()...];
                                 kind = domain_kind, kind_text = KIND_TEXT)
end

"""
    domain_kind(tables) -> Symbol

The kind of a domain from its section tables (raw or resolved), which
decides the providers of its slot sections (ONE-m, #476): `:urban` for
an urban site (`[urban] enabled = true`), else `:model`.
"""
function domain_kind(tables)
    u = get(tables, "urban", nothing)
    u === nothing && return :model
    on = u isa Config.ConfigSection ? (haskey(u, "enabled") ? u["enabled"] : false) :
         u isa AbstractDict ? get(u, "enabled", false) : false
    return on === true ? :urban : :model
end

"How the messages name the domain kinds of [`domain_kind`](@ref)."
const KIND_TEXT = Dict(:model => "a domain of the model's core", :urban => "an urban site ([urban] enabled = true)")

"The dynamical core `[core] model` of a (one-domain) config selects: `:boussinesq` or `:anelastic` (on an urban site also `:pseudo_incompressible`)."
_core(cfg) = cfg.core.model

"The schema default of `section.key` of the model."
_schema_default(cfg, sec, key) =
    only(o for o in only(x for x in Config.config_schema(cfg).sections if x.name == sec).options if o.key == key).default

"Whether `cfg` steps adaptively (`time.dt = \"adaptive\"`)."
_adaptive(cfg) = cfg.time.dt === :adaptive

"The component types a domain's config enables (in the order of [`COMPONENTS`](@ref))."
enabled_components(cfg) = Any[C for C in COMPONENTS if Components.enabled(C, cfg[Components.section(C)])]

const _OUTPUT_CHECKS_DOC = "every `output.variables` entry is in the catalogue of the core or of an enabled component (`Components.output_problems`): the `qv` ones and `budget_water` need moisture, `ql`, `qr` Kessler, `tracer` and `budget_tracer` the tracer, `budget_energy` on the boussinesq core `dynamics.theta0` > 0; in a nest, of at least one domain (each domain writes those it has)"

const _TIME_CHECKS_DOC = "with a fixed `time.dt`, `time.cfl` and `time.dt_max` keep their defaults (they belong to adaptive steps), except in a nest with an urban site, whose root steps adaptively with them while `time.dt` is the site's step"

# the [output] selection of a model domain's view against the catalogues of
# its core and its enabled components (ONE-c): per variable, its problem
_output_problems(v) =
    Dict(p => split(p, ':')[1] for p in Components.output_problems(v.output.variables,
                                        [C => v[Components.section(C)] for C in enabled_components(v)];
                                        moisture = v.dynamics.moisture, theta0 = v.dynamics.theta0, core = _core(v),
                                        published = _coupling_outputs(v)))

# [output] variables (a Config.ModelCheck): each against the catalogues of
# the core and the enabled components of a model domain (ONE-c). In a nest
# every domain writes the variables of the selection it has (one file per
# domain), and a variable is a problem when no domain has it: when every
# model domain reports it and no urban site is among the domains (whose
# catalogue the build checks, `Urban.build_model`, NEST-p3 #483).
function _output_checks(cfg)
    e = ConfigError[]
    n = length(config_domains(cfg))
    views = n <= 1 ? [cfg] : [domain_view(cfg, i) for i in 1:n]
    Urban.urban_enabled(views[1]) && n <= 1 && return e
    any(Urban.urban_enabled, views) && return e
    per = [Set(values(_output_problems(v))) => _output_problems(v) for v in views]
    for (msg, var) in last(per[1])
        all(x -> var in first(x), per) && push!(e, ConfigError("output.variables", msg))
    end
    return e
end

# [time] (a Config.ModelCheck: the root of an urban site's nest steps
# adaptively with time.cfl and dt_max under the site's fixed time.dt)
function _time_checks(cfg)
    e = ConfigError[]
    t = cfg.time
    if !_adaptive(cfg) && !_urban_nest(cfg)
        for k in ("cfl", "dt_max")
            isequal(t[k], _schema_default(cfg, "time", k)) ||
                push!(e, ConfigError("time.$k", "belongs to adaptive steps (time.dt = \"adaptive\"), not a fixed dt: it would be ignored"))
        end
    end
    return e
end

function _cross_checks(cfg)
    Urban.urban_enabled(cfg) && return _urban_site_checks(cfg)
    e = ConfigError[]
    t = cfg.time; d = cfg.dynamics; b = cfg.boundaries; s = cfg.sgs; g = cfg.grid; dom = cfg.domain
    adaptive = _adaptive(cfg)
    (t.duration === nothing) == (t.nsteps === nothing) &&
        push!(e, ConfigError("time.duration", "give exactly one of time.duration and time.nsteps (nsteps with duration = \"none\")"))
    for ax in ("x", "y")
        b[ax] === :noslip &&
            push!(e, ConfigError("boundaries.$ax", "noslip is a z-wall condition; use free_slip or periodic"))
    end
    if b.x === :open || b.y === :open
        ax = b.x === :open ? "x" : "y"
        dom.dynamic_driver === nothing &&
            push!(e, ConfigError("boundaries.$ax", "open faces take the boundary planes of a dynamic driver: set the domain's dynamic_driver (AeolusPrep.jl: aeolusprep dynamic --from-era5)"))
        (b.x in (:open, :periodic) && b.y in (:open, :periodic)) ||
            push!(e, ConfigError("boundaries.$ax", "with an open axis the other lateral axis is open or periodic"))
        b.z === :periodic && push!(e, ConfigError("boundaries.z", "an open box needs z walls"))
        _core(cfg) === :boussinesq || push!(e, ConfigError("core.model", "open faces need the boussinesq core"))
        g.stretching === :uniform || push!(e, ConfigError("grid.stretching", "open faces need a uniform grid"))
        cfg.sponge.alpha_max == 0 ||
            push!(e, ConfigError("sponge.alpha_max", "the sponge of an open box has no target here; use 0"))
    end
    if !isempty(cfg.initial.field)
        all(==(:periodic), (b.x, b.y, b.z)) ||
            push!(e, ConfigError("initial.field", "isotropic turbulence needs a triply periodic box (boundaries x, y, z periodic)"))
        g.stretching === :uniform || push!(e, ConfigError("initial.field", "isotropic turbulence needs a uniform grid"))
        _core(cfg) === :boussinesq || push!(e, ConfigError("initial.field", "isotropic turbulence initialises the boussinesq core"))
        for k in ("u", "v")
            isempty(cfg.initial[k]) || push!(e, ConfigError("initial.$k", "initial.field sets the velocity: give one of the two"))
        end
    end
    for n in vcat(cfg.output.variables, [get(v, "variable", q) for (q, v) in cfg.checks.reference])
        n in ("energy_spectrum", "dissipation", "dissipation_sgs") || continue
        g.stretching === :uniform || push!(e, ConfigError("output.variables", "$n needs a uniform grid"))
        n == "energy_spectrum" && !all(==(:periodic), (b.x, b.y, b.z)) &&
            push!(e, ConfigError("output.variables", "energy_spectrum needs a triply periodic box"))
    end
    get(cfg.initial.velocity_perturbation, "project", false) && _core(cfg) !== :boussinesq &&
        push!(e, ConfigError("initial.velocity_perturbation", "project = true projects with the boussinesq core's projection"))
    (t.start != 0 && (b.x === :open || b.y === :open)) &&
        push!(e, ConfigError("time.start", "open faces read their boundary planes from the run start: time.start = 0"))
    (cfg.coriolis.f != 0 && cfg.coriolis.latitude !== nothing) &&
        push!(e, ConfigError("coriolis.latitude", "give coriolis.f or coriolis.latitude, not both"))
    if _core(cfg) === :anelastic
        b.z === :periodic && push!(e, ConfigError("boundaries.z", "the anelastic core needs z walls"))
        d.theta0 > 0 || push!(e, ConfigError("dynamics.theta0", "the anelastic core needs theta0 > 0 (its base state's θ)"))
        d.moisture && push!(e, ConfigError("dynamics.moisture", "the anelastic core has no qv"))
        g.stretching === :uniform || push!(e, ConfigError("grid.stretching", "the anelastic core needs a uniform grid"))
    end
    if g.stretching !== :uniform
        s.model in (:none, :smagorinsky, :tke) ||
            push!(e, ConfigError("sgs.model", "on a stretched grid the closure is none, smagorinsky or tke, not $(s.model)"))
        Components.option(s, :stability, :none) === :none ||
            push!(e, ConfigError("sgs.stability", "the stability correction needs a uniform grid"))
    end
    sp = cfg.sponge
    (sp.alpha_max != 0 && sp.z_start !== nothing && !(sp.z_start < g.extent[3])) &&
        push!(e, ConfigError("sponge.z_start", "the sponge starts below the top (grid.extent[3] = $(g.extent[3]) m), got $(sp.z_start) m"))
    (!d.moisture && !isempty(cfg.initial.qv)) &&
        push!(e, ConfigError("initial.qv", "a qv profile needs dynamics.moisture = true"))
    for (k, p) in (("u", cfg.initial.u), ("v", cfg.initial.v))
        (get(p, "kind", "") == "taylor_green" && (b.x !== :periodic || b.y !== :periodic)) &&
            push!(e, ConfigError("initial.$k", "the Taylor-Green field needs periodic x and y"))
    end
    append!(e, _boundary_layer_checks(cfg))
    append!(e, _one_l_checks(cfg))
    append!(e, _checks_problems(cfg))
    append!(e, _nest_domain_checks(cfg))
    # slots, requirements, incompatibilities of the enabled components (ONE-b)
    for p in component_problems(enabled_components(cfg))
        push!(e, ConfigError(p.key, p.message))
    end
    append!(e, _coupling_checks(cfg))
    return e
end

# An urban site (ONE-e, #454): its own checks, and none of the model's own
# components or couplings enabled (the site's core carries its Coriolis
# force, sponge and closure until those are the model's components)
function _urban_site_checks(cfg)
    e = Urban.urban_checks(cfg)
    for C in (Coriolis, Sponge, SurfaceThetaFlux)
        Components.enabled(C, cfg[Components.section(C)]) || continue
        push!(e, ConfigError(Components.section(C), "an urban site carries its own $(Components.section(C)) ([urban]); the model's component is not part of it"))
    end
    _any_coupling(cfg) && push!(e, ConfigError("urban", "an urban site takes no [forcing], [tracer], [particles] or [obstacles]"))
    # the checks an urban site publishes: the sanity check, the closure of
    # the heat of the air (its domain's budget), its budgets' and those of
    # its enabled components
    own = Set(vcat(["finite", "energy_closure", "urban_budgets"],
                   ["$(Urban.component_name(C))_checks" for C in Urban.URBAN_COMPONENTS if Urban.is_selected(cfg, C)]))
    for n in _published_enabled(cfg)
        n in own || push!(e, ConfigError("checks.$n", "an urban site publishes no check $n (its checks: $(join(sort!(collect(own)), ", ")))"))
    end
    isempty(cfg.checks.reference) || push!(e, ConfigError("checks.reference", "an urban site takes no reference comparisons yet"))
    append!(e, _spinup_checks(cfg))
    return e
end

# the surface spin-up of an urban site (#519): its step divides the
# radiation period, which divides its length (each radiation interval of
# the spin-up is whole), and the driver forcing needs a driver
_divides(a, b) = (q = b / a; q >= 1 - 1e-9 && abs(q - round(q)) < 1e-9 * max(1, q))
function _spinup_checks(cfg)
    e = ConfigError[]
    sp, u = cfg.spinup, cfg.urban
    sp.time > 0 || return e
    P = u.radiation_period
    dt = something(sp.dt, something(u.surface_period, u.soil_period))
    _divides(dt, P) ||
        push!(e, ConfigError("spinup.dt", "the spin-up's surface step ($dt s) divides the radiation period " *
                                          "(urban.radiation_period = $P s)"))
    _divides(P, sp.time) ||
        push!(e, ConfigError("spinup.time", "the spin-up is a whole number of radiation periods " *
                                            "(urban.radiation_period = $P s), got $(sp.time) s"))
    (sp.forcing === :driver && cfg.domain.dynamic_driver === nothing) &&
        push!(e, ConfigError("spinup.forcing", "driver forcing reads the dynamic driver's sfc_temperature series: " *
                                               "set the domain's dynamic_driver, or forcing = \"solar\""))
    return e
end

# the boundary-layer components (ONE-g): what they need of the domain
function _boundary_layer_checks(cfg)
    e = ConfigError[]
    sl = cfg.surface; ls = cfg.large_scale_forcing; g = cfg.grid; b = cfg.boundaries
    nodrv = cfg.domain.dynamic_driver === nothing
    if Components.enabled(SurfaceLayer, sl)
        g.stretching === :uniform || push!(e, ConfigError("surface.model", "the surface layer needs a uniform grid"))
        b.z === :periodic && push!(e, ConfigError("surface.model", "the surface layer needs z walls"))
        cfg.surface_theta_flux.bottom == 0 ||
            push!(e, ConfigError("surface_theta_flux.bottom", "the surface layer sets the floor's θ flux: leave surface_theta_flux.bottom at 0"))
        get(sl.temperature, "kind", "") == "driver" && nodrv &&
            push!(e, ConfigError("surface.temperature", "kind = \"driver\" reads the domain's dynamic_driver: set it"))
    end
    Components.lsf_reads_driver(ls) && nodrv &&
        push!(e, ConfigError("large_scale_forcing", "reads the domain's dynamic_driver (\"driver\" values, " *
                                                     "advective_tendencies, subsidence, nudging): set it"))
    (ls.u_g != 0 || ls.v_g != 0) && !Components.enabled(Coriolis, cfg.coriolis) &&
        push!(e, ConfigError("large_scale_forcing.u_g", "the geostrophic pressure gradient balances the Coriolis term: set coriolis.f or coriolis.latitude"))
    sp = cfg.sponge
    sp.target === :geostrophic && Components.enabled(Sponge, sp) && !Components.enabled(LargeScale, ls) &&
        push!(e, ConfigError("sponge.target", "\"geostrophic\" relaxes toward the geostrophic wind of [large_scale_forcing], which is off"))
    cfg.towers.below > 0 && g.stretching !== :uniform &&
        push!(e, ConfigError("towers.below", "the towers need a uniform grid"))
    return e
end

# the components of the validation cases (ONE-l, #475): what they need of the domain
function _one_l_checks(cfg)
    e = ConfigError[]
    fr = cfg.fringe; b = cfg.boundaries; g = cfg.grid; d = cfg.dynamics
    if fr.profile !== :none
        b.x === :periodic || push!(e, ConfigError("fringe.profile", "the fringe recycles the outflow: boundaries.x = \"periodic\""))
        b.z === :noslip || push!(e, ConfigError("fringe.profile", "the plate is a no-slip floor: boundaries.z = \"noslip\""))
        g.stretching === :uniform || push!(e, ConfigError("fringe.profile", "the fringe needs a uniform grid"))
        fr.length < g.extent[1] || push!(e, ConfigError("fringe.length", "the fringe is longer than the domain"))
        fr.rise + fr.fall <= fr.length || push!(e, ConfigError("fringe.rise", "rise + fall exceed the fringe length"))
        d.nu == fr.velocity / fr.inflow_reynolds ||
            push!(e, ConfigError("dynamics.nu", "the fringe's layer has ν = velocity / inflow_reynolds = " *
                                                "$(fr.velocity / fr.inflow_reynolds) m² s⁻¹ (δ*₀ = 1 m), got $(d.nu)"))
    end
    kf = cfg.kinematic_flow
    if kf.kind !== :none
        any(==(:open), (b.x, b.y)) && push!(e, ConfigError("kinematic_flow.kind", "the prescribed flow runs in a closed or periodic box"))
        kf.kind === :solid_body_rotation && kf.omega == 0 &&
            push!(e, ConfigError("kinematic_flow.omega", "a solid-body rotation needs omega ≠ 0"))
    end
    st = cfg.inflow_statistics
    st.enabled && cfg.sgs.model === :none &&
        push!(e, ConfigError("inflow_statistics.enabled", "the statistics' SGS part reads the domain's SGS closure: set [sgs] model"))
    st.enabled && g.stretching !== :uniform &&
        push!(e, ConfigError("inflow_statistics.enabled", "the inflow statistics need a uniform grid"))
    return e
end

"The values of the resolved `[checks]` table."
_checks_table(cfg) = getfield(cfg.checks, :values)

"The published checks (not the reference comparisons) `[checks]` enables."
_published_enabled(cfg) = sort!([k for (k, v) in _checks_table(cfg) if k != "reference" && v isa AbstractDict && !isempty(v)])

# the [checks] against the published checks and the reference data (ONE-d)
function _checks_problems(cfg)
    e = ConfigError[]
    d = cfg.dynamics; c = cfg.checks
    tops = [C => cfg[Components.section(C)] for C in enabled_components(cfg)]
    for (n, p) in Components.check_problems(_published_enabled(cfg), tops;
                                            moisture = d.moisture, theta0 = d.theta0, core = _core(cfg),
                                            published = _coupling_outputs(cfg))
        push!(e, ConfigError("checks.$n", p))
    end
    ref = c.reference
    if isempty(ref)
        c.reference_file === nothing ||
            push!(e, ConfigError("checks.reference_file", "no [checks.reference] comparison reads it; add one or remove the file"))
        return e
    end
    if c.reference_file === nothing
        push!(e, ConfigError("checks.reference", "the reference comparisons need checks.reference_file"))
        return e
    end
    data = try
        read_reference(c.reference_file)
    catch err
        err isa ArgumentError || rethrow()
        push!(e, ConfigError("checks.reference_file", sprint(showerror, err)))
        return e
    end
    for q in sort!(collect(keys(ref)))
        haskey(data, q) || push!(e, ConfigError("checks.reference.$q", "$(c.reference_file) has no quantity $q " *
                                                                       "(it has: $(join(sort!(collect(keys(data))), ", ")))"))
        var = get(ref[q], "variable", q)
        for p in Components.output_problems([var], tops;
                                            moisture = d.moisture, theta0 = d.theta0, core = _core(cfg),
                                            published = _coupling_outputs(cfg))
            push!(e, ConfigError("checks.reference.$q", "variable " * p))
        end
    end
    return e
end

const _CROSS_CHECKS_DOC = """
exactly one of `time.duration` and `time.nsteps`
`noslip` is a z-wall condition: not for `boundaries.x`, `boundaries.y`
an open face needs a domain `dynamic_driver`, the other lateral axis open or periodic, z walls, the boussinesq core, a uniform grid and no sponge
`initial.field` (isotropic turbulence) needs a triply periodic uniform grid and sets u, v (no `initial.u`, `initial.v`); `energy_spectrum` needs a triply periodic uniform grid, `dissipation` a uniform one; `initial.velocity_perturbation.project` needs the boussinesq core; `time.start` is 0 with open faces; `coriolis.f` and `coriolis.latitude` exclude each other
the surface layer (`surface.model = "most"`) needs a uniform grid, z walls and `surface_theta_flux.bottom = 0`; its `temperature = {kind = "driver"}` and the `[large_scale_forcing]` driver tables need the domain's `dynamic_driver`; a geostrophic wind needs `[coriolis]`; `sponge.target = "geostrophic"` needs `[large_scale_forcing]`; the towers need a uniform grid
the anelastic core needs z walls, `dynamics.theta0` > 0, no moisture and a uniform grid
`[fringe]` needs periodic x, a no-slip floor (`boundaries.z = "noslip"`), a uniform grid, a fringe shorter than the domain with `rise + fall ≤ length`, and `dynamics.nu = velocity / inflow_reynolds`; `[kinematic_flow]` runs in a box without open faces (a rotation with `omega ≠ 0`); `[inflow_statistics]` needs an `[sgs]` closure and a uniform grid
on a stretched grid `sgs.model` is `none`, `smagorinsky` or `tke` and `sgs.stability` is `none`
`sponge.z_start` lies below the top
an `initial.qv` profile needs `dynamics.moisture = true`
a Taylor-Green initial `u` or `v` needs periodic x and y
the enabled components hold at most one component per slot and an `:advection` provider, and meet each other's `requires` and `incompatible` (`Components.component_problems`)
every `[checks]` entry is published by the core or an enabled component (`Components.check_problems`): a closure check has the requirements of its `budget_*` output
`[checks.reference]` comparisons need `checks.reference_file`, which must hold each quantity, and their variables are in the output catalogue; a `reference_file` without comparisons is an error"""

Config.driver_checks(::SimulationDriverType) =
    Any[_CROSS_CHECKS_DOC * "\n" * _COUPLING_CHECKS_DOC * "\n" * Urban.URBAN_CHECKS_DOC => _cross_checks,
        _NEST_CHECKS_DOC => Config.ModelCheck(_nest_checks),
        _TIME_CHECKS_DOC => Config.ModelCheck(_time_checks),
        _OUTPUT_CHECKS_DOC => Config.ModelCheck(_output_checks)]

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

"""
    SimulationRun

What the model builds from a config: `model`, a [`Model`](@ref
Aeolus.Components.Model) of one [`Domain`](@ref Aeolus.Components.Domain)
(ONE-a, #450), or of several with their nest coupling (ONE-f, #455).
Each domain has its core (a `BoussinesqSimulation` or an
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

"The core of a `simulation` run: the `BoussinesqSimulation` or `AnelasticSimulation` of its (root) domain."
simulation(r::SimulationRun) = first(Components.domains(r.model)).state
_domain(r::SimulationRun) = first(Components.domains(r.model))

function _mesh(g)
    o = (g.origin..., 0.0)
    if g.stretching === :uniform
        return RegularGrid(; size = g.size, extent = g.extent, halo = g.halo, origin = o)
    elseif g.stretching === :surface_refined
        return StretchedGrid(; size = g.size, extent = g.extent, halo = g.halo, origin = o,
                             stretching = (:custom, _surface_refined_faces(g)))
    end
    return StretchedGrid(; size = g.size, extent = g.extent, halo = g.halo, origin = o,
                         stretching = (g.stretching, g.stretching_parameter))
end

"The interior face heights of a `[grid] stretching = \"surface_refined\"` grid (`StretchedGrids.surface_refined_faces`)."
_surface_refined_faces(g) = surface_refined_faces(; dz_surface = g.dz_surface, z_uniform = g.z_uniform, z_top = g.extent[3],
                                                  ratio = g.stretching_parameter, dz_max = g.dz_max)

"Whether `dyn` (a `DynamicInputs` or `nothing`) is a sounding driver (`aeolusprep dynamic --from-sounding`)."
_sounding(dyn) = dyn !== nothing && get(dyn.attributes, "dynamic_source", "") == "sounding"

"""
    initial_theta_profile(cfg, dyn) -> z -> θ

The initial potential-temperature profile of a domain (before the noise):
its `[initial] theta` profile, the sounding driver's at the start
(`ls_profile_theta`), or `dynamics.theta0`. What the reference state of the
domain's components, the fixed-gradient θ top and the damping toward the
initial profile read.
"""
function initial_theta_profile(cfg, dyn)
    p = cfg.initial.theta
    !isempty(p) && get(p, "kind", "") in ("constant", "mixed_layer_lapse", "table", "exponential", "tanh") &&
        return z -> profile_value(p, z)
    if isempty(p) && _sounding(dyn)
        tab = dyn.tables[:profile_theta]
        return z -> forcing_value(tab, z, 0.0)
    end
    θ0 = Float64(cfg.dynamics.theta0)
    return z -> θ0
end

# The geostrophic wind of the domain's [large_scale_forcing] at the start, functions of z
function _geostrophic_at_start(cfg, dyn)
    ls = cfg.large_scale_forcing; t0 = cfg.time.start
    sg = (ls.u_g === :sounding || ls.v_g === :sounding) ?
         Components.sounding_geostrophic(dyn, ls.z_g; t0 = t0) : nothing
    F = (ls.u_g === :driver || ls.v_g === :driver) ? dynamic_forcing(dyn; t0 = t0) : nothing
    function q(k, i)
        v = ls[k]
        v isa Real && return z -> Float64(v)
        v === :sounding && return z -> sg[i](z, t0)
        tab = i == 1 ? F.U_g : F.V_g
        return z -> forcing_value(tab, z, t0)
    end
    return q("u_g", 1), q("v_g", 2)
end

_bc(s::Symbol) = s === :free_slip ? :wall : s

# the top of a closed box: the [boundaries] z face, or the one an enabled
# component sets (Components.core_boundaries: the fringe's free stream, ONE-l)
function _top_bc(cfg, mesh)
    for C in enabled_components(cfg)
        o = Components.core_boundaries(C, cfg[Components.section(C)], mesh, cfg.dynamics.nu)
        o === nothing || return o.top
    end
    return _bc(cfg.boundaries.z)
end
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
    # FFTW on Julia's threads and BLAS pinned to one (KA-h, maintainer decision
    # on #291) for every model, before any plan is made; an explicit count
    # ([fftw] threads, AEOLUS_FFTW_THREADS, set_fftw_policy!) wins. Until #524
    # only the urban site called it, so every other domain's FFTs ran on one thread.
    Backends.init_threads!(Backends.default_backend())
    if Urban.urban_enabled(cfg) && !_nested(cfg)
        # the urban site (ONE-e #454, ONE-k #474): one domain, the site its
        # first component, stepped by its :timestepper component
        return SimulationRun(Urban.build_model(cfg), StepperState(:euler), cfg)
    end
    # several domains: the nest (an urban site among them is a child, NEST-p3 #483)
    model = _nested(cfg) ? _build_nested(cfg) : Model(_build_domain(cfg))
    adaptive = _adaptive(cfg) || _urban_nest(cfg)
    return SimulationRun(model, StepperState(adaptive ? :adaptive : config_domains(cfg)[1].timestep.scheme), cfg)
end

# One domain of the model from its config (a single-domain config, or one
# domain's view of a nested one): its core on `mesh` with the faces of its
# [boundaries] (or `bcs`, a nested domain's nest faces), its initial
# state, components, couplings and budgets.
function _build_domain(cfg; mesh = _mesh(cfg.grid), bcs = nothing, cfl = :config)
    d = cfg.dynamics; b = cfg.boundaries; s = cfg.sgs; ddrv = cfg.domain.dynamic_driver
    dyn = ddrv === nothing ? nothing : read_dynamic_driver(ddrv; mesh = mesh, planes = _is_open(cfg))
    feed = nothing
    if bcs !== nothing
        θ_bcs = bcs
    elseif _is_open(cfg)
        dyn.boundary === nothing &&
            throw(ConfigError("dynamic_driver", "$ddrv has no boundary planes, which the open faces need (write it with the run's open boundaries)"))
        bcs = θ_bcs = _open_bcs(cfg, mesh)
        feed = dyn.boundary
        check_planes(feed, only(open_boundary_states(bcs)).planes)
        set_boundary_data!(bcs, feed.data)
    else
        bcs = FieldBoundaryConditions(_bc(b.x), _bc(b.y), _bc(b.z); top = _top_bc(cfg, mesh))
        θ_bcs = b.theta_top === :initial_gradient ?
                FieldBoundaryConditions(_scalar_bc(b.x), _scalar_bc(b.y); bottom = :noflux,
                                        top = FluxBC(_initial_top_gradient(initial_theta_profile(cfg, dyn), mesh))) :
                FieldBoundaryConditions(_scalar_bc(b.x), _scalar_bc(b.y), _scalar_bc(b.z))
    end
    # the core carries the dynamics only: SGS, advection, Coriolis, the
    # surface θ flux and the sponge are its domain's components (ONE-a)
    # (the core's Pr_t: the closure's; 1/3 for the closures without one, none and tke)
    kw = (; ν = d.nu, g_accel = d.g, Pr_t = Components.option(s, :pr_t, 1 / 3), bcs = bcs, θ_bcs = θ_bcs)
    sim = if _core(cfg) === :anelastic
        AnelasticSimulation(mesh; ref = isentropic_reference(mesh; θ_const = d.theta0, g_accel = d.g), kw...)
    else
        BoussinesqSimulation(mesh; θ₀ = d.theta0, moisture = d.moisture, kw...)
    end
    Simulations.set_loop_fusion!(sim, cfg.timestep.loop_fusion)   # #533
    couplings = _couplings(cfg, mesh)
    _initialise!(sim, cfg, dyn)
    dyn === nothing || _initialise_dynamic!(sim, dyn, cfg)
    sim.t = cfg.time.start
    couplings.water === nothing || isempty(cfg.initial.ql) || _set_profile!(couplings.water.ql, cfg.initial.ql, :ql)
    t = cfg.time
    cfl === :config && (cfl = _adaptive(cfg) ? CFLConstraints(; C_advective = t.cfl, dt_max = t.dt_max) : nothing)
    drivers = merge(feed === nothing ? (;) : (; boundary = feed), dyn === nothing ? (;) : (; dynamic = dyn),
                    _domain_inputs(cfg, mesh, sim, dyn))
    d0 = Domain(sim; id = cfg.domain.id, scheme = cfg.timestep.scheme, cfl = cfl, validate = false, drivers = drivers)
    comps = build_components(Tuple(C => cfg[Components.section(C)] for C in COMPONENTS), d0)
    _any_coupling(cfg) && (comps = (comps..., SimulationCouplings(couplings)))
    # the budgets the [output] selection measures (ONE-c): off unless selected,
    # and the closure checks [checks] enables (ONE-d)
    qs = unique!(vcat(Symbol[q for n in cfg.output.variables for q in get(Components.BUDGET_OUTPUTS, n, ())],
                      Components.check_budgets(_published_enabled(cfg))))
    return Components.with_budgets(Domain(d0; components = comps), qs)
end

# The fixed gradient (K m⁻¹) of the initial profile θ(z) at the top face of
# `mesh` (#254): across the top face, over one top-level thickness.
function _initial_top_gradient(θf, mesh)
    m = mesh
    dz, H = hasproperty(m, :zf) ? (m.zf[m.Hz+m.Nz+1] - m.zf[m.Hz+m.Nz], m.zf[m.Hz+m.Nz+1] - m.zf[m.Hz+1]) : (m.dz, m.Lz)
    return (θf(H + dz / 2) - θf(H - dz / 2)) / dz
end

# The domain's inputs the components read besides its dynamic driver
# (NEST-p3): its initial θ profile (`theta_profile`), the hydrostatic
# reference state of a Boussinesq domain (`reference`: the one its
# components convert fluxes with and θ becomes T with, PALM's `hyp` from
# the initial θ and the surface pressure; since #536 every Boussinesq
# domain whose initial θ is a temperature, `Components.thermodynamic_reference`;
# not the buoyancy-free ones that start at θ = 0, `theta0 = 0`), and the
# tile form of its static driver (`static_tiles`, for [surface] model =
# "tiles", with the driver's initial soil, `soil`, #521).
function _domain_inputs(cfg, mesh, sim, dyn)
    θf = initial_theta_profile(cfg, dyn)
    out = (; theta_profile = θf)
    enabled = enabled_components(cfg)
    needs_ref = any(C -> C in enabled, (Urban.TiledUrbanSurface, Urban.PrescribedSurfaceFlux, Urban.MeanColumnRadiation))
    # a θ that is not a temperature (θ ≤ 0 somewhere: the buoyancy-free runs)
    # has no hydrostatic state; a component that needs one fails in its build
    zc = last(cell_centres(mesh))
    physical = cfg.dynamics.g > 0 && all(z -> θf(z) > 0, zc)
    if sim isa BoussinesqSimulation && !physical && !needs_ref
        bad = [n for n in cfg.output.variables if n in _ABSOLUTE_T_OUTPUTS]
        isempty(bad) ||
            throw(ConfigError("output.variables", "$(join(bad, ", ")): an absolute temperature needs the hydrostatic " *
                                                  "reference of the initial theta, which is not positive here " *
                                                  "(dynamics.theta0 or [initial] theta > 0 K, dynamics.g > 0)"))
    end
    if sim isa BoussinesqSimulation && (physical || needs_ref)
        ps = cfg.dynamics.p_surface
        ps === nothing && (ps = dyn === nothing || dyn.p_surface === nothing ? 101325.0 : dyn.p_surface)
        out = merge(out, (; reference = Urban.domain_reference(mesh, θf, ps; g = cfg.dynamics.g)))
    end
    if Urban.TiledUrbanSurface in enabled
        path = cfg.domain.static_driver
        path === nothing &&
            throw(ConfigError("surface.model", "\"tiles\" reads the tile form of the domain's static_driver, which is \"none\" " *
                                               "(AeolusPrep.jl: aeolusprep static)"))
        out = merge(out, (; static_tiles = _read_tiles(path, mesh)))
        # the initial soil of the tiles (#521): the dynamic driver's, if it holds one
        dyn === nothing || dyn.soil === nothing ||
            (out = merge(out, (; soil = (data = dyn.soil, path = cfg.domain.dynamic_driver))))
    end
    return out
end

# the outputs that turn θ into an absolute temperature with the domain's reference (#536)
const _ABSOLUTE_T_OUTPUTS = ("temperature", "temperature_profile", "temperature_mean", "surface_temperature",
                             "temperature_2m")

# the tile form of a static driver on `mesh`, at the anchor it records
function _read_tiles(path, mesh)
    a = InputDrivers.NCDatasets.NCDataset(path) do ds
        Dict{String,Any}(k => v for (k, v) in ds.attrib)
    end
    return read_static_tiles(path; mesh = mesh, origin_lat = a["origin_lat"], origin_lon = a["origin_lon"],
                             origin_x = a["origin_x"], origin_y = a["origin_y"])
end

# The initial profiles of a dynamic driver (linear in z, constant beyond
# the ends: `profile_value` of a table).
# A driver without profiles (one that holds only forcing tables or a
# surface series, read by the components) initialises nothing; the
# [initial] tables and the driver's profiles exclude each other per
# variable, and only the boussinesq core takes the driver's profiles.
function _initialise_dynamic!(sim, dyn, cfg)
    P = dyn.profiles
    P === nothing && return sim
    path = cfg.domain.dynamic_driver
    sim isa BoussinesqSimulation ||
        throw(ConfigError("dynamic_driver", "$path has initial profiles, which initialise the boussinesq core only"))
    for (n, f) in ((:u, sim.u), (:v, sim.v), (:theta, sim.θ), (:qv, sim.qv))
        n === :qv && !cfg.dynamics.moisture && continue
        a = getproperty(P, n)
        a === nothing && continue
        isempty(cfg.initial[String(n)]) ||
            throw(ConfigError("initial.$n", "$path has an initial $n profile: give one of the two"))
        _set_profile!(f, Dict{String,Any}("kind" => "table", "z" => P.z, "values" => a), n)
    end
    return sim
end

function _set_profile!(f::Field, p::AbstractDict, comp::Symbol)
    if get(p, "kind", "") == "taylor_green"
        A = Float64(p["amplitude"]); k = Float64(p["k"])
        comp === :u ? set!(f, (x, y, z) -> A * (sin(k * x) * cos(k * y) * cos(k * z))) :
                      set!(f, (x, y, z) -> A * (-cos(k * x) * sin(k * y) * cos(k * z)))
    elseif get(p, "kind", "") == "seed_mode"          # amplitude sin(k x) exp(−((z − center)/thickness)²)
        a = Float64(p["amplitude"]); k = Float64(p["wavenumber"]); c = Float64(p["center"]); h = Float64(p["thickness"])
        set!(f, (x, y, z) -> a * sin(k * x) * exp(-((z - c) / h)^2))
    elseif get(p, "kind", "") == "gaussian"           # a column: value + amplitude exp(−r²/(2σ²))
        v = Float64(p["value"]); a = Float64(p["amplitude"]); x0, y0 = Float64.(p["center"]); σ = Float64(p["sigma"])
        set!(f, (x, y, z) -> v + a * exp(-((x - x0)^2 + (y - y0)^2) / (2σ^2)))
    elseif get(p, "kind", "") == "box"
        v = Float64(p["value"]); bx = Float64.(p["x"]); by = Float64.(p["y"]); bz = Float64.(p["z"])
        set!(f, (x, y, z) -> (bx[1] < x < bx[2] && by[1] < y < by[2] && bz[1] < z < bz[2]) ? v : 0.0)
    else
        set!(f, (x, y, z) -> profile_value(p, z))
    end
    return f
end

function _initialise!(sim, cfg, dyn = nothing)
    ini = cfg.initial
    geo = any(p -> get(p, "kind", "") == "geostrophic", (ini.u, ini.v)) ? _geostrophic_at_start(cfg, dyn) : nothing
    for (k, f, i) in (("u", sim.u, 1), ("v", sim.v, 2))
        p = ini[k]
        isempty(p) && continue
        get(p, "kind", "") == "geostrophic" ? (g = geo[i]; set!(f, (x, y, z) -> g(z))) : _set_profile!(f, p, Symbol(k))
    end
    isempty(ini.w) || _set_profile!(sim.w, ini.w, :w)
    if !isempty(ini.theta)
        _set_profile!(sim.θ, ini.theta, :theta)
    elseif _sounding(dyn)                # a sounding driver: its profiles at the start (before the noise)
        tab = dyn.tables[:profile_theta]
        set!(sim.θ, (x, y, z) -> forcing_value(tab, z, 0.0))
    elseif sim isa BoussinesqSimulation && cfg.dynamics.theta0 != 0
        set!(sim.θ, (x, y, z) -> cfg.dynamics.theta0)
    end                                  # anelastic: the constructor set θ = θ̄(z)
    if !isempty(ini.qv)
        _set_profile!(sim.qv, ini.qv, :qv)
    elseif _sounding(dyn) && cfg.dynamics.moisture
        tab = dyn.tables[:profile_qv]
        set!(sim.qv, (x, y, z) -> forcing_value(tab, z, 0.0))
    end
    isempty(ini.field) || initial_field!(sim, ini.field, cfg.run.seed)
    a = ini.perturbation_amplitude
    rng = Xoshiro(cfg.run.seed)
    if a > 0
        depth = ini.perturbation_depth
        noise = Field(CellCenter(), sim.mesh)
        if ini.perturbation_kind === :normal
            set!(noise, (x, y, z) -> z < depth ? a * randn(rng) : 0.0)
        else
            set!(noise, (x, y, z) -> z < depth ? a * (rand(rng) - 0.5) : 0.0)
        end
        interior_view(sim.θ) .+= interior_view(noise)
    end
    vp = ini.velocity_perturbation
    if !isempty(vp)
        A = Float64(vp["amplitude"]); uniform = vp["kind"] == "uniform"
        depth = Float64(get(vp, "depth", Inf)); comps = get(vp, "components", ["u", "v", "w"])
        linear = get(vp, "taper", "none") == "linear"
        shaped = isfinite(depth) || linear
        tap(z) = linear ? 1 - z / depth : 1.0
        for (name, f) in (("u", sim.u), ("v", sim.v), ("w", sim.w))   # one field at a time, in this order
            name in comps || continue
            if uniform                          # one draw per cell (below depth), in set!'s order
                noise = Field(location(f), sim.mesh)
                set!(noise, (x, y, z) -> z < depth ? A * (rand(rng) - 0.5) * tap(z) : 0.0)
                interior_view(f) .+= interior_view(noise)
            elseif !shaped                      # one array of normals per field (randn's bulk stream)
                interior_view(f) .+= A .* randn(rng, size(interior_view(f))...)
            else                                # the same draws, zero above depth, tapered
                r = randn(rng, size(interior_view(f))...)
                shape = Field(location(f), sim.mesh)
                set!(shape, (x, y, z) -> z < depth ? tap(z) : 0.0)
                interior_view(f) .+= A .* r .* interior_view(shape)
            end
        end
        # random velocities are not solenoidal: the flux-form scalar advection
        # would turn their divergence into a source (θ ∇·u), so they are
        # projected once (the core's Chorin projection)
        get(vp, "project", false) && _project!(sim, 1.0)
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

# output: the one writer (Aeolus.Output, ONE-c) of the variables
# `[output]` selects from the domain's catalogue; in a nested model one
# file per domain, the root's `netcdf` and `<stem>_<id>.nc` for the others
function _output_writer(r::SimulationRun, rf, resume_t, dom = _domain(r))
    cfg = r.cfg; o = cfg.output
    cat = Components.domain_catalogue(dom)
    # a nest: each domain writes the selected variables it has
    names = length(Components.domains(r.model)) == 1 ? o.variables :
            (have = Set(v.name for v in cat); [n for n in o.variables if n in have])
    vars = select_variables(cat, names)
    dir = o.path === nothing ? output_dir(rf) : mkpath(o.path)
    m = dom.state.mesh
    x, y, z = cell_centres(m)
    x .+= m.x0; y .+= m.y0                         # a nested domain's cells in the root's frame
    k = any(v -> v.dims === :k, vars) ? spectrum_wavenumbers(m) : nothing
    stem, ext = splitext(o.netcdf)
    file = dom === _domain(r) ? o.netcdf : string(stem, "_", dom.id, ext)
    # the components' static maps (the urban site's geometry), in a file that has variables
    statics = isempty(vars) ? OutputVariable[] :
              OutputVariable[v for c in dom.components for v in Components.static_outputs(c)]
    return OutputWriter(joinpath(dir, file); x = x, y = y, z = z, k = k,
                        records = filter(v -> v.dims !== :scalar, vars), series = filter(v -> v.dims === :scalar, vars),
                        statics = statics, ctx = dom, averaging = o.averaging, resume_time = resume_t,
                        attributes = Dict{String,Any}("title" => "Aeolus.jl — $(cfg.run.name)",
                                                      "source" => "the Aeolus model, domain $(dom.id)",
                                                      "components" => join(string.(map(Components.component_name, dom.components)), " ")))
end

# the open averaging intervals of a `:mean` writer as a checkpoint component:
# the checkpoint is restored before the writer is reopened (the writer needs
# the restored time), so the state waits in `pending` until `attach!`
# (one writer per domain of a nest, ONE-f; a single domain's state is its
# writer's alone, as before)
mutable struct OutputMeans
    ws::Union{Nothing,Vector{OutputWriter}}
    pending::Any
end
IO.checkpoint_state(o::OutputMeans) = length(o.ws) == 1 ? output_means(o.ws[1]) : map(output_means, o.ws)
IO.restore_checkpoint_state!(o::OutputMeans, s) = (o.pending = s; o)
function _attach!(o::OutputMeans, ws::AbstractVector)
    o.ws = collect(OutputWriter, ws)
    if o.pending !== nothing
        states = length(ws) == 1 ? (o.pending,) : o.pending
        length(states) == length(ws) ||
            throw(ArgumentError("output means: $(length(states)) writers in the checkpoint, $(length(ws)) domains"))
        foreach(restore_output_means!, ws, states)
    end
    o.pending = nothing
    return ws
end

# the next output time after `t` of a stream with cadence `c` (0: every step)
_next_out(t, c) = c == 0 ? t : (floor(t / c * (1 + 1e-9)) + 1) * c

function Simulations.run!(::SimulationDriverType, r::SimulationRun, rf::RunFolder,
                          ck::CheckpointSchedule; callback = nothing)
    sim = simulation(r); cfg = r.cfg; st = cfg.time
    dom = _domain(r); comps = checkpoint_components(dom); closure = sgs_closure(dom)
    doms = Components.domains(r.model)
    nest = r.model.nest                            # several domains: the nest coupling (ONE-f)
    nest === nothing || (comps = merge(comps, (; nest = nest)))
    checks = _check_runner(r)                      # nothing without [checks] (ONE-d)
    checks === nothing || (comps = merge(comps, (; checks = checks)))
    means = cfg.output.averaging === :mean ? OutputMeans(nothing, nothing) : nothing
    means === nothing || (comps = merge(comps, (; output_means = means)))
    resume!(ck, sim; components = comps, stepper = r.stepper)
    n = ck.last_step
    # the surface spin-up (#519), before the first step of a fresh run
    n == 0 && _spinup!(cfg, doms)
    # a domain whose core step a component takes (the urban site, ONE-k
    # #474): it picks every step and lands on the end itself
    taker = Components.core_taker(dom)
    # a nest with an urban site (NEST-p3 #483): the root steps adaptively,
    # the site's nsteps × dt is the run's length
    urbannest = nest !== nothing && _urban_nest(cfg)
    adaptive = _adaptive(cfg) || urbannest
    dt_fixed, nfix = _adaptive(cfg) ? (NaN, typemax(Int)) : _fixed_steps(st)
    nmax = urbannest ? typemax(Int) : adaptive ? something(st.nsteps, typemax(Int)) : nfix
    t_stop = st.duration === nothing ? nothing : st.start + st.duration   # the model time of the end
    (taker === nothing && !urbannest) || (t_stop = st.start + nfix * dt_fixed)
    t_end = adaptive || taker !== nothing ? t_stop : sim.t + (nmax - n) * dt_fixed      # for the progress line
    (t_end === nothing || t_end > sim.t) || (t_end = nothing)           # resumed at the end
    feed = get(dom.drivers, :boundary, nothing)
    (feed === nothing || t_end === nothing) || check_coverage(feed, sim.t, t_end)
    ws = [_output_writer(r, rf, n == 0 ? nothing : sim.t, d) for d in doms]
    means === nothing || _attach!(means, ws)
    o = cfg.output; mean = o.averaging === :mean
    # instant: a record at the start, then every cadence; mean: the first
    # record is the mean of the first cadence
    next_rec = n == 0 && !mean ? 0.0 : _next_out(sim.t, o.cadence)
    next_ser = n == 0 && !mean ? 0.0 : _next_out(sim.t, o.series_cadence)
    function write_due!()
        if sim.t >= next_rec * (1 - 1e-9)
            foreach((w, d) -> write_record!(w, d, sim.t), ws, doms); next_rec = _next_out(sim.t, o.cadence)
        end
        if o.series_cadence == 0 || sim.t >= next_ser * (1 - 1e-9)
            foreach((w, d) -> write_series!(w, d, sim.t), ws, doms); next_ser = _next_out(sim.t, o.series_cadence)
        end
    end
    m_ref = Ref{Any}(nothing)                      # the run monitor, for taken_step!
    done() = taker !== nothing ? !Components.run_continues(taker, dom) :
             n >= nmax || (adaptive && t_stop !== nothing && sim.t >= t_stop * (1 - 1e-12))
    # one step of a domain whose step a component picks: its pick, what
    # limited it and the retries it took go into the diagnostics row
    function taken_step!()
        dt = Components.pick_dt(r.model)
        lim, r0 = Components.step_report(taker, dom)
        res = monitored_step!(m_ref[], sim; step = n, limiter = lim) do
            step!(r.model, dt; checked = false)
            (sim, Float64(dt), Components.step_report(taker, dom)[2] - r0)
        end
        return res
    end
    cpl = sim_couplings(r)
    n == 0 && _record_initial!(cpl, sim)
    # attached (#336): the step cores report every step; monitored_step!
    # turns a step that throws into a diagnosed blowup. A nest is not
    # attached: the root's step core ends before its children subcycle, so
    # the guard checks (every domain) and keeps its good state after the
    # whole model step, never a root ahead of its children. Nor is a domain
    # whose step a component takes: one of its steps may be several core
    # steps (retried pieces), and it reports once per step.
    mon = merge(closure === nothing ? (;) : (; sgs = closure), _coupling_components(cpl))
    # (a nest: the diagnostics of a domain whose step a component takes, the urban site)
    idom = findfirst(d -> Components.core_taker(d) !== nothing, doms)
    rd = idom === nothing ? nothing : Components.run_diagnostics(Components.core_taker(doms[idom]), doms[idom])
    monkw = rd === nothing ? (; components = mon) : (; diagnostics = rd[1], profiles = rd[2])
    try
    with_monitor(sim; run = rf, monkw..., t_end = t_end, attach = nest === nothing && taker === nothing,
                 cadence = cfg.logging.diagnostics_cadence, checkpoint_components = comps,
                 stepper = r.stepper, step0 = n) do m
        m_ref[] = m
        foreach(d -> foreach(c -> Components.run_begin!(c, d, t_end === nothing ? NaN : Float64(t_end)), d.components), doms)
        try
            mean || n > 0 || write_due!()
            while !done()
                n += 1
                _, dt, retries = taker !== nothing ? taken_step!() :
                                 monitored_step!(m, sim; step = n) do
                                     adaptive ? _adaptive_step!(r, t_stop) : _take_step!(r, dt_fixed)
                                 end
                record_step!(r.stepper, dt; retries = retries)
                checks === nothing || sample_checks!(checks, sim.t, dt)     # stage 13: :checks, then :output
                foreach((w, d) -> accumulate_output!(w, d, dt), ws, doms)
                write_due!()
                maybe_checkpoint!(ck, sim; step = n, components = comps, stepper = r.stepper)
                callback === nothing || callback(r, n)
            end
        finally
            foreach(d -> foreach(c -> Components.run_end!(c, d), d.components), doms)
            foreach(close_output!, ws)
        end
        # the guard checks on its cadence; a run never ends `completed` with
        # non-finite fields between two checks
        check_blowup!(m.guard, sim; step = n, force = true)
    end
    catch err
        # a run that stops (a blow-up) records its failed sanity check
        (checks === nothing || err isa InterruptException) ||
            _write_checks(rf, cfg, stopped_checks(checks, err; step = n, t = sim.t))
        rethrow()
    end
    _write_tracks(cpl, cpl.particles === nothing ? nothing :
                               cfg.output.path === nothing ? output_dir(rf) : mkpath(cfg.output.path))
    # results.toml: the components' numbers of the run and the checks [checks] enables
    extra = Dict{String,Any}()
    foreach(d -> foreach(c -> merge!(extra, Components.run_results(c, d)), d.components), doms)
    # each domain's loop-fusion setting and whether its substages ran fused,
    # with the terms without a fused form (#533); so every run writes
    # results.toml, also one without checks or component results
    extra["loop_fusion"] = Dict{String,Any}(d.id => Components.sweep_report(d) for d in doms)
    if checks !== nothing
        _finish_checks(rf, cfg, evaluate_checks(checks); extra = extra)
    elseif !isempty(extra)
        Config.write_results(rf, extra; dir = cfg.output.path)
    end
    return r
end

# The surface spin-up of a fresh run (#519): every domain's components
# that have slow surfaces advance them over [spinup] time before the
# start ([`Components.spinup!`](@ref)). A spin-up that no component takes
# is a configuration error, not a silent no-op.
function _spinup!(cfg, doms)
    t = cfg.spinup.time
    t > 0 || return false
    sp = (time = Float64(t), dt = cfg.spinup.dt, forcing = cfg.spinup.forcing, pt_mean = cfg.spinup.pt_mean,
          pt_amplitude = Float64(cfg.spinup.pt_amplitude), soil_moisture = cfg.spinup.soil_moisture,
          buildings = cfg.spinup.buildings)
    done = false
    for d in doms, c in d.components
        done |= Components.spinup!(c, d, sp)
    end
    done || throw(ConfigError("spinup.time", "no domain of the run has surfaces to spin up (an urban site's)"))
    return true
end

# the enabled checks of a run: the domain's published checks and the
# reference comparisons that [checks] enables, or nothing (ONE-d); in a
# nested model each published check is the worst over the domains, the
# reference comparisons are the root's
function _check_runner(r::SimulationRun)
    cfg = r.cfg; c = cfg.checks; dom = _domain(r)
    settings = enabled_checks(_checks_table(cfg))
    isempty(settings) && return nothing
    doms = Components.domains(r.model)
    cs = length(doms) == 1 ? Components.domain_checks(dom) : _nest_checks_merged(doms)
    isempty(c.reference) ||
        append!(cs, Components.reference_checks(dom, read_reference(c.reference_file), c.reference))
    return CheckRunner(cs, settings)
end

_write_checks(rf, cfg, results; extra = Dict{String,Any}()) =
    Config.write_results(rf, merge(extra, results_table(results)); dir = cfg.output.path)

# results.toml; an advisory failure warns, a required one ends the run (exit code 3)
function _finish_checks(rf, cfg, results; extra = Dict{String,Any}())
    path = _write_checks(rf, cfg, results; extra = extra)
    for x in results
        @info("check $(x.name) $(x.passed ? "passed" : "FAILED")", subsystem = :checks, domain = current_domain(),
              event = :check, required = x.required, value = x.value, tolerance = x.tolerance, units = x.units)
    end
    adv = advisory_failures(results)
    isempty(adv) || @warn("advisory checks failed: $(join((x.name for x in adv), ", ")) ($path)",
                          subsystem = :checks, domain = current_domain(), event = :checks_failed)
    req = required_failures(results)
    isempty(req) || throw(ChecksFailed(req))
    return results
end

"""
    state_fields(r::SimulationRun) -> NamedTuple

The prognostic arrays with their halos, as a checkpoint stores them
(`IO.checkpoint_fields`): `u, v, w, p, θ`, the eddy coefficients `νt, κt`
of the last step and, on the Boussinesq core, `qv`; and those of the
couplings: the slab temperature `T_s`, `ql`, `qr`, `tracer` and the
particle positions (`particles`, 3 × n); and what the components carry
(`Components.state_arrays`: the urban site's tracers and surface
temperature). Those of the root domain; in a nested model also every
other domain's, named `<id>_<name>` (`inner_u`, …).
"""
function Config.state_fields(r::SimulationRun)
    root = merge(map(f -> f.data, IO.checkpoint_fields(simulation(r))), _coupling_fields(sim_couplings(r)),
                 (Components.state_arrays(c, _domain(r)) for c in _domain(r).components)...)
    doms = Components.domains(r.model)
    length(doms) == 1 && return root
    more = Pair{Symbol,Any}[Symbol(d.id, "_", k) => f.data for d in doms[2:end] for (k, f) in pairs(IO.checkpoint_fields(d.state))]
    for d in doms[2:end], c in d.components, (k, a) in pairs(Components.state_arrays(c, d))
        push!(more, Symbol(d.id, "_", k) => a)
    end
    return merge(root, NamedTuple(more))
end

register_driver!(SimulationDriverType())

end # module
