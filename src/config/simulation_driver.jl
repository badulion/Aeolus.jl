"""
    Aeolus.SimulationDriver

The `simulation` run driver (CFG-b, #357): a bare run of the Boussinesq
or the anelastic core on a box, configured from TOML (`[run] driver =
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

using Random: Xoshiro
using ..Grids: RegularGrid, CellCenter
using ..Fields: Field, set!, interior_view
using ..BoundaryConditions: FieldBoundaryConditions
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
`[run] driver = "simulation"`. Its sections are `[grid]`, `[dynamics]`,
`[boundaries]`, `[sgs]`, `[initial]`, `[stepper]` and `fields` in
`[output]`; `aeolus options simulation` prints them. Shipped configs:
`examples/configs/`.
"""
struct SimulationDriverType <: AbstractRunDriver end

const CONFIG_DIR = normpath(joinpath(@__DIR__, "..", "..", "examples", "configs"))

Config.driver_name(::SimulationDriverType) = "simulation"
Config.config_dir(::SimulationDriverType) = CONFIG_DIR

# ---------------------------------------------------------------------------
# initial profiles
# ---------------------------------------------------------------------------

const _PROFILE_KEYS = Dict("constant" => ["value"],
                           "mixed_layer_lapse" => ["value", "z_i", "lapse"],
                           "table" => ["z", "values"],
                           "taylor_green" => ["amplitude", "k"])

_isnum(x) = x isa Real && !(x isa Bool)

"The validator of the profile table of variable `var` (an empty table: the default profile)."
function _profile_validator(var::String)
    return function (t)
        isempty(t) && return String[]
        kind = get(t, "kind", nothing)
        kinds = var in ("u", "v") ? sort!(collect(keys(_PROFILE_KEYS))) :
                filter!(!=("taylor_green"), sort!(collect(keys(_PROFILE_KEYS))))
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
            else
                _isnum(v) && isfinite(v) || push!(probs, "$k: expected a finite number, got $(repr(v))")
            end
        end
        if kind == "table" && isempty(probs)
            length(t["z"]) == length(t["values"]) ||
                push!(probs, "z and values have different lengths ($(length(t["z"])) and $(length(t["values"])))")
            issorted(t["z"]; lt = <=) || push!(probs, "z: expected strictly increasing heights")
        end
        return probs
    end
end

"""
    profile_value(p::AbstractDict, z) -> Float64

The value at height `z` of a vertical profile of `[initial]`:

  * `{kind = "constant", value}`;
  * `{kind = "mixed_layer_lapse", value, z_i, lapse}`: `value` below `z_i`,
    `value + lapse (z − z_i)` above (a capping inversion; with `z_i` at the
    ground, a linear profile);
  * `{kind = "table", z = […], values = […]}`: linear interpolation,
    constant beyond the ends.
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
    end
    throw(ArgumentError("profile_value: kind $(repr(kind)) is not a vertical profile"))
end

# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------

const _OUTPUT_FIELDS = ["u", "v", "w", "p", "theta", "qv"]

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
            option!(s, ax, Symbol; default = :periodic, allowed = [:periodic, :free_slip, :noslip],
                    doc = "the $ax faces: periodic, free_slip (impermeable, stress free) or noslip (z only); θ and qv have no flux through a wall, except the prescribed θ fluxes")
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
    return Section[grid, dyn, bnd, sgs, ini, stp, out]
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

Config.driver_checks(::SimulationDriverType) = Any[_CROSS_CHECKS_DOC => _cross_checks]

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

"""
    SimulationRun

What the `simulation` driver builds: `sim` (a `BoussinesqSimulation` or
an `AnelasticSimulation`), `closure` (the SGS closure passed to the
steps; `nothing` for none or the built-in Smagorinsky), `stepper` (a
[`StepperState`](@ref)), and the resolved `cfg`.
"""
struct SimulationRun
    sim::Any
    closure::Any
    stepper::StepperState
    cfg::RunConfig
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

function Config.build(::SimulationDriverType, cfg::RunConfig)
    mesh = _mesh(cfg.grid)
    d = cfg.dynamics; b = cfg.boundaries; s = cfg.sgs
    bcs = FieldBoundaryConditions(_bc(b.x), _bc(b.y), _bc(b.z))
    θ_bcs = FieldBoundaryConditions(_scalar_bc(b.x), _scalar_bc(b.y), _scalar_bc(b.z))
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
    _initialise!(sim, cfg)
    scheme = cfg.stepper.cfl === nothing ? cfg.stepper.scheme : :adaptive
    return SimulationRun(sim, closure, StepperState(scheme), cfg)
end

function _set_profile!(f::Field, p::AbstractDict, comp::Symbol)
    if get(p, "kind", "") == "taylor_green"
        A = Float64(p["amplitude"]); k = Float64(p["k"])
        comp === :u ? set!(f, (x, y, z) -> A * (sin(k * x) * cos(k * y) * cos(k * z))) :
                      set!(f, (x, y, z) -> A * (-cos(k * x) * sin(k * y) * cos(k * z)))
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
    if a > 0
        rng = Xoshiro(cfg.run.seed); depth = ini.perturbation_depth
        noise = Field(CellCenter(), sim.mesh)
        set!(noise, (x, y, z) -> z < depth ? a * (rand(rng) - 0.5) : 0.0)
        interior_view(sim.θ) .+= interior_view(noise)
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

function _take_step!(r::SimulationRun, dt)
    sim = r.sim; sgs = r.closure; scheme = r.cfg.stepper.scheme
    scheme === :rk3 ? step_rk3!(sim, dt; sgs = sgs) : step!(sim, dt; sgs = sgs)
    return (sim, Float64(dt), 0)
end

function _adaptive_step!(r::SimulationRun, c::CFLConstraints, t_end)
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
        for f in r.cfg.output.fields
            v = nc_defvar!(nc, f, Float32, ("x", "y", "z"))
            v[:, :, :] = Float32.(Array(interior_view(getproperty(sim, _FIELD_OF[f]))))
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
    c = adaptive ? CFLConstraints(; C_advective = st.cfl, dt_max = st.dt_max) : nothing
    outdir = isempty(cfg.output.fields) ? nothing :
             cfg.output.path === nothing ? output_dir(rf) : mkpath(cfg.output.path)
    cadence = cfg.output.cadence
    next_out = outdir === nothing ? Inf : (n == 0 ? 0.0 : (floor(sim.t / cadence) + 1) * cadence)
    done() = n >= nmax || (adaptive && st.end_time !== nothing && sim.t >= st.end_time * (1 - 1e-12))
    comps = r.closure === nothing ? (;) : (; sgs = r.closure)
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
                adaptive ? _adaptive_step!(r, c, st.end_time) : _take_step!(r, dt_fixed)
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
    return r
end

"""
    state_fields(r::SimulationRun) -> NamedTuple

The prognostic arrays with their halos: `u, v, w, p, θ` (and, for the
Boussinesq core, `qv` and the eddy coefficients `νt, κt` of the last
step, as a checkpoint stores them).
"""
function Config.state_fields(r::SimulationRun)
    s = r.sim
    s isa BoussinesqSimulation && return map(f -> f.data, IO.checkpoint_fields(s))
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data, θ = s.θ.data)
end

register_driver!(SimulationDriverType())

end # module
