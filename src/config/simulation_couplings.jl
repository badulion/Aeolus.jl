# The couplings of the `simulation` driver (EX-d, #441): what the example
# scripts of examples/kernels/ and examples/coupled/ did around the
# dynamical core's step, as configuration. Included into module
# SimulationDriver (src/config/simulation_driver.jl).
#
#   [forcing]       a uniform radiative heating rate of θ, after each step
#   [surface]       a slab land surface that exchanges a sensible flux with the
#                   air (model = "slab"), or a prescribed map of the surface
#                   heat flux (model = "flux_map")
#   [microphysics]  Kessler warm rain on cloud and rain water (kessler)
#   [tracer]        a passive tracer with box sources and first-order decay
#   [particles]     Lagrangian particles, their trajectories written to CSV
#   [obstacles]     immersed obstacles from a static driver (IBM penalty)
#
# Each runs after the step of the core, in the order of the list above,
# except the surface flux, which is set (slab) or applied (flux_map) before
# it. None of them is checkpointed: a run with a coupling on has
# [checkpoint] off (a cross check).

using ..IBM: SolidMask, add_ibm_penalty!
using ..Radiation: SlabLand, step_land!, surface_temperature, apply_radiative_cooling!
using ..Microphysics: apply_kessler_step!
using ..Chemistry: apply_linear_chemistry!
using ..Particles: ParticleSet, add_particles!, advance_particles!, positions
using ..Operators: advect_centered!
using ..Fields: location
using ..BoundaryConditions: fill_halos!
using ..Grids: xnodes, ynodes, znodes
using ..InputDrivers: read_static_driver, write_static_driver, solid_and_classes, load_city_geojson
import ..RunLog: diagnostics, profile_diagnostics
using ..RunLog: Diagnostic, ProfileDiagnostic
using ..Backends: plane_mean!

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

_range_ok(v) = v isa AbstractVector && length(v) == 2 && all(_isnum, v) && v[1] < v[2]

"The validator of a table of boxes `{x = [a, b], y = [c, d], z = [e, f], <value>}` (`value_key`)."
function _boxes_validator(value_key::String; axes = ("x", "y", "z"))
    return function (t)
        probs = String[]
        for name in sort!(collect(keys(t)))
            b = t[name]
            b isa AbstractDict || (push!(probs, "$name: expected a table {$(join(axes, ", ")), $value_key}"); continue)
            for k in sort!(collect(keys(b)))
                k in axes || k == value_key || push!(probs, "$name.$k: unknown key (keys: $(join(axes, ", ")), $value_key)")
            end
            for a in axes
                haskey(b, a) || (push!(probs, "$name.$a: required ([lower, upper], m)"); continue)
                _range_ok(b[a]) || push!(probs, "$name.$a: expected [lower, upper] with lower < upper (m)")
            end
            haskey(b, value_key) && _isnum(b[value_key]) && isfinite(b[value_key]) ||
                push!(probs, "$name.$value_key: required, a finite number")
        end
        return probs
    end
end

function _coupling_sections()
    frc = section("forcing"; doc = "Prescribed forcing of the air.") do s
        option!(s, "radiative_cooling", Float64; default = 0.0, units = "K s^-1",
                doc = "uniform heating rate of θ (negative: cooling; −2.315e-5 is −2 K/day), added as rate × dt after each step (not inside obstacles); 0: off")
    end
    sfc = section("surface"; doc = "The lower boundary's heat source.") do s
        option!(s, "model", Symbol; default = :none, allowed = [:none, :slab, :flux_map],
                doc = "none (boundaries.theta_flux_bottom only); slab (a SlabLand whose sensible flux is the θ flux of the floor, " *
                      "H = exchange_velocity (T_s − θ_ref)); flux_map (the θ flux of flux_patches over flux_background, " *
                      "added to the first level as Q dt/Δz before each step)")
        option!(s, "temperature", Float64; default = 297.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "slab: initial surface temperature T_s")
        option!(s, "heat_capacity", Float64; default = 1e5, range = (0, Inf), open_lower = true, units = "J m^-2 K^-1",
                doc = "slab: areal heat capacity C_s")
        option!(s, "albedo", Float64; units = "1", default = 0.25, range = (0, 1), doc = "slab: shortwave albedo")
        option!(s, "exchange_velocity", Float64; default = 0.01, range = (0, Inf), units = "m s^-1",
                doc = "slab: the bulk transfer velocity c_H U of the kinematic flux Q = c (T_s − θ_ref)")
        option!(s, "theta_ref", Optional{Float64}; units = "K",
                doc = "slab: the air θ of the bulk flux; \"none\": dynamics.theta0")
        option!(s, "rho_cp", Float64; default = 1005.0, range = (0, Inf), open_lower = true, units = "J m^-3 K^-1",
                doc = "slab: ρ c_p of the air, H = ρ c_p Q (W m⁻²) cools the slab")
        option!(s, "sw_peak", Float64; default = 800.0, range = (0, Inf), units = "W m^-2",
                doc = "slab: downward shortwave SW = sw_peak sin(π t / sw_duration), ≥ 0")
        option!(s, "sw_duration", Float64; default = 10800.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "slab: the length of the shortwave half-wave (the daylight)")
        option!(s, "lw_down", Float64; default = 340.0, range = (0, Inf), units = "W m^-2",
                doc = "slab: downward longwave")
        option!(s, "flux_background", Float64; default = 0.0, units = "K m s^-1",
                doc = "flux_map: the kinematic θ flux outside the patches")
        option!(s, "flux_patches", OpenTable(_boxes_validator("value"; axes = ("x", "y")));
                doc = "flux_map: named patches {x = [x₀, x₁], y = [y₀, y₁], value} (K m s⁻¹) over the cells whose centres are strictly inside; a later name (sorted) wins where they overlap")
    end
    mp = section("microphysics"; doc = "Cloud and rain water (with dynamics.moisture).") do s
        option!(s, "scheme", Symbol; default = :none, allowed = [:none, :kessler],
                doc = "none, or kessler: apply_kessler_step! on (qv, ql, qr) after each step, the Marshall–Palmer fall-speed spectrum; " *
                      "ql and qr are not advected (stated simplification of the scripts this replaces)")
    end
    tr = section("tracer"; doc = "A passive tracer (advect_centered!, forward Euler after each step, the θ boundary conditions).") do s
        option!(s, "enabled", Bool; default = false, doc = "carry the tracer `tracer` (initially 0)")
        option!(s, "decay_time", Float64; default = Inf, range = (0, Inf), open_lower = true, units = "s",
                doc = "first-order decay time τ (apply_linear_chemistry!); Inf: inert")
        option!(s, "sources", OpenTable(_boxes_validator("rate"));
                doc = "named sources {x = [x₀, x₁], y = [y₀, y₁], z = [z₀, z₁], rate} adding `rate` (tracer units s⁻¹) to every cell whose centre is strictly inside")
    end
    pt = section("particles"; doc = "Lagrangian particles (advance_particles!, after each step).") do s
        option!(s, "count", Int; default = 0, range = (0, typemax(Int)), doc = "particles; 0: none")
        option!(s, "seed", Int; default = 0, range = (0, typemax(Int)), doc = "seed of the initial positions (Xoshiro)")
        for ax in ("x", "y", "z")
            option!(s, ax, Optional{NTuple{2,Float64}}; units = "m",
                    doc = "initial $ax uniform in [lower, upper] (equal bounds: that value, no draw); \"none\": the domain")
        end
        option!(s, "record_every", Int; default = 1, range = (1, typemax(Int)),
                doc = "positions written to output/particles.csv every this many steps")
    end
    ob = section("obstacles"; doc = "Immersed obstacles.") do s
        option!(s, "static", Optional{Path};
                doc = "a static driver (aeolus preprocess static, from preprocess.obstacles): its buildings are solid in every cell whose centre is at or below their height; hashed under [inputs]")
        option!(s, "anchor", NTuple{2,Float64}; default = (0.0, 0.0), units = "degrees",
                doc = "latitude, longitude of the domain's south-west corner (the static driver's anchor; checked, not used)")
        option!(s, "penalty_time", Float64; units = "1", default = 3.0, range = (0, Inf), open_lower = true,
                doc = "the IBM penalty relaxes the velocity in solid cells over τ = penalty_time × dt (add_ibm_penalty!)")
    end
    return Section[frc, sfc, mp, tr, pt, ob]
end

# keys of [surface] that belong to one model
const _SLAB_KEYS = ("temperature", "heat_capacity", "albedo", "exchange_velocity", "theta_ref", "rho_cp",
                    "sw_peak", "sw_duration", "lw_down")
const _FLUX_MAP_KEYS = ("flux_background", "flux_patches")

"Whether any coupling of the simulation driver is on."
_any_coupling(cfg) = cfg.forcing.radiative_cooling != 0 || cfg.surface.model !== :none ||
                     cfg.microphysics.scheme !== :none || cfg.tracer.enabled || cfg.particles.count > 0 ||
                     cfg.obstacles.static !== nothing

function _coupling_checks(cfg)
    e = ConfigError[]
    s = cfg.surface; d = cfg.dynamics; b = cfg.boundaries
    if _any_coupling(cfg)
        (cfg.checkpoint.every_steps == 0 && cfg.checkpoint.every_seconds == Inf) ||
            push!(e, ConfigError("checkpoint.every_steps", "the couplings ([forcing], [surface], [microphysics], [tracer], [particles], [obstacles]) are not checkpointed: leave [checkpoint] off"))
        d.core === :boussinesq || push!(e, ConfigError("dynamics.core", "the couplings run with the boussinesq core"))
        cfg.stepper.cfl === nothing || push!(e, ConfigError("stepper.cfl", "the couplings take a fixed stepper.dt"))
        cfg.grid.stretching === :uniform || push!(e, ConfigError("grid.stretching", "the couplings need a uniform grid"))
        (b.x === :open || b.y === :open) && push!(e, ConfigError("boundaries.x", "the couplings run in a closed box (no open faces)"))
    end
    defaults = Dict(o.key => o.default for o in only(x for x in getfield(cfg, :schema).sections if x.name == "surface").options)
    for (model, keys_) in ((:slab, _SLAB_KEYS), (:flux_map, _FLUX_MAP_KEYS))
        s.model === model && continue
        for k in keys_
            isequal(s[k], defaults[k]) ||
                push!(e, ConfigError("surface.$k", "belongs to surface.model = \"$model\", not \"$(s.model)\": it would be ignored"))
        end
    end
    if s.model !== :none
        b.z === :periodic && push!(e, ConfigError("surface.model", "a surface model needs z walls"))
        (b.theta_flux_bottom == 0) ||
            push!(e, ConfigError("boundaries.theta_flux_bottom", "the surface model sets the floor's θ flux: leave theta_flux_bottom at 0"))
    end
    s.model === :slab && s.theta_ref === nothing && d.theta0 == 0 &&
        push!(e, ConfigError("surface.theta_ref", "the slab's bulk flux needs a reference θ: set theta_ref or dynamics.theta0"))
    if cfg.microphysics.scheme !== :none
        d.moisture || push!(e, ConfigError("microphysics.scheme", "Kessler needs dynamics.moisture = true (qv)"))
        b.z === :periodic && push!(e, ConfigError("microphysics.scheme", "rain falls out through a floor: Kessler needs z walls"))
    end
    !isempty(cfg.initial.ql) && cfg.microphysics.scheme === :none &&
        push!(e, ConfigError("initial.ql", "cloud water needs microphysics.scheme = \"kessler\""))
    for (f, need) in (("ql", cfg.microphysics.scheme !== :none), ("qr", cfg.microphysics.scheme !== :none),
                      ("tracer", cfg.tracer.enabled))
        f in cfg.output.fields && !need && push!(e, ConfigError("output.fields", "$f is written only when it is carried"))
    end
    (!cfg.tracer.enabled && (!isempty(cfg.tracer.sources) || cfg.tracer.decay_time != Inf)) &&
        push!(e, ConfigError("tracer.enabled", "[tracer] sources or decay_time are set but the tracer is off"))
    if cfg.particles.count > 0
        all(==(:periodic), (b.x, b.y, b.z)) ||
            push!(e, ConfigError("particles.count", "particles wrap periodically: they need a triply periodic box"))
    end
    for ax in ("x", "y", "z")
        r = cfg.particles[ax]
        r === nothing || r[1] <= r[2] || push!(e, ConfigError("particles.$ax", "expected [lower, upper] with lower ≤ upper"))
    end
    cfg.obstacles.static !== nothing && cfg.dynamics.core !== :boussinesq &&
        push!(e, ConfigError("obstacles.static", "obstacles need the boussinesq core"))
    return e
end

const _COUPLING_CHECKS_DOC = """
a run with a coupling on ([forcing], [surface], [microphysics], [tracer], [particles], [obstacles]) has `[checkpoint]` off, the boussinesq core, a fixed `stepper.dt`, a uniform grid and no open faces
a key of `[surface]` that belongs to a model `surface.model` does not select keeps its default; a surface model needs z walls and `boundaries.theta_flux_bottom = 0`; the slab needs `surface.theta_ref` or `dynamics.theta0`
Kessler microphysics needs `dynamics.moisture` and z walls; `initial.ql` needs it; the `ql`, `qr`, `tracer` output fields need what they write
`[tracer]` sources or a decay time need `tracer.enabled`; particles need a triply periodic box"""

# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

"The slab land surface of `[surface] model = \"slab\"` and its flux of the current step."
mutable struct SlabSurface
    land::SlabLand{Float64}
    c::Float64              # exchange velocity
    θref::Float64
    ρcp::Float64
    sw_peak::Float64
    sw_duration::Float64
    lw::Float64
    Q::Float64              # kinematic flux of the current step
    sw::Float64             # shortwave of the current step
end

"Cloud and rain water of `[microphysics] scheme = \"kessler\"`."
struct KesslerWater
    ql::Field
    qr::Field
end

"The passive tracer of `[tracer]`: the field, its tendency, the decay time and the source cells."
struct PassiveTracer
    c::Field
    tc::Field
    τ::Float64
    sources::Vector{Tuple{Int,Int,Int,Float64}}    # storage indices and rate
end

"The particles of `[particles]` and their recorded trajectories (`rows`: step, t, id, x, y, z)."
struct ParticleTracks
    ps::ParticleSet
    every::Int
    rows::Vector{NTuple{6,Float64}}
end

"""
    SimCouplings

The couplings of a `simulation` run (each `nothing` when off):
`cooling` (K s⁻¹), `slab` ([`SlabSurface`](@ref)), `flux_map` (the
kinematic θ flux per surface column), `water` ([`KesslerWater`](@ref)),
`tracer` ([`PassiveTracer`](@ref)), `particles` ([`ParticleTracks`](@ref)),
`solid` (the `SolidMask` of the obstacles) and `penalty_time`.
"""
struct SimCouplings
    cooling::Float64
    slab::Union{Nothing,SlabSurface}
    flux_map::Union{Nothing,Matrix{Float64}}
    water::Union{Nothing,KesslerWater}
    tracer::Union{Nothing,PassiveTracer}
    particles::Union{Nothing,ParticleTracks}
    solid::Union{Nothing,SolidMask}
    penalty_time::Float64
end

"Storage indices of the cells whose centres are strictly inside the box `b` (`axes` of x, y, z)."
function _box_cells(mesh, b; axes = ("x", "y", "z"))
    xs = xnodes(mesh, CellCenter()); ys = ynodes(mesh, CellCenter()); zs = znodes(mesh, CellCenter())
    inside(r, v) = Float64(r[1]) < v < Float64(r[2])
    ir = [i for i in eachindex(xs) if inside(b["x"], xs[i])]
    jr = [j for j in eachindex(ys) if inside(b["y"], ys[j])]
    kr = "z" in axes ? [k for k in eachindex(zs) if inside(b["z"], zs[k])] : Int[]
    return ir, jr, kr
end

function _flux_map(cfg, mesh)
    s = cfg.surface
    Q = fill(s.flux_background, mesh.Nx, mesh.Ny)
    for name in sort!(collect(keys(s.flux_patches)))
        p = s.flux_patches[name]
        ir, jr, _ = _box_cells(mesh, p; axes = ("x", "y"))
        Q[ir, jr] .= Float64(p["value"])
    end
    return Q
end

function _tracer(cfg, mesh)
    t = cfg.tracer
    c = Field(CellCenter(), mesh); tc = Field(CellCenter(), mesh)
    src = Tuple{Int,Int,Int,Float64}[]
    for name in sort!(collect(keys(t.sources)))
        b = t.sources[name]
        ir, jr, kr = _box_cells(mesh, b)
        (isempty(ir) || isempty(jr) || isempty(kr)) &&
            throw(ConfigError("tracer.sources.$name", "the box holds no cell centre of the grid"))
        for i in ir, j in jr, k in kr
            push!(src, (mesh.Hx + i, mesh.Hy + j, mesh.Hz + k, Float64(b["rate"])))
        end
    end
    return PassiveTracer(c, tc, t.decay_time, src)
end

function _particles(cfg, mesh)
    p = cfg.particles
    rng = Xoshiro(p.seed)
    span(r, lo, L) = r === nothing ? (lo, lo + L) : r
    rs = (span(p.x, mesh.x0, mesh.Lx), span(p.y, mesh.y0, mesh.Ly), span(p.z, mesh.z0, mesh.Lz))
    draw(r) = r[1] == r[2] ? r[1] : r[1] + (r[2] - r[1]) * rand(rng)
    pos = [(draw(rs[1]), draw(rs[2]), draw(rs[3])) for _ in 1:p.count]
    ps = ParticleSet(mesh)
    add_particles!(ps, pos)
    return ParticleTracks(ps, p.record_every, NTuple{6,Float64}[])
end

"The `SolidMask` of the static driver `path` on `mesh`: buildings solid up to their height."
function _static_solid(cfg, mesh)
    lat, lon = cfg.obstacles.anchor
    st = read_static_driver(cfg.obstacles.static; mesh = mesh, origin_lat = lat, origin_lon = lon,
                            origin_x = mesh.x0, origin_y = mesh.y0)
    sm, _ = solid_and_classes(st, mesh; tag_walls = false)
    return sm
end

function _couplings(cfg, mesh)
    s = cfg.surface
    slab = s.model === :slab ?
        SlabSurface(SlabLand(; T = s.temperature, C_s = s.heat_capacity, albedo = s.albedo), s.exchange_velocity,
                    something(s.theta_ref, cfg.dynamics.theta0), s.rho_cp, s.sw_peak, s.sw_duration, s.lw_down, 0.0, 0.0) :
        nothing
    water = cfg.microphysics.scheme === :kessler ? KesslerWater(Field(CellCenter(), mesh), Field(CellCenter(), mesh)) : nothing
    return SimCouplings(cfg.forcing.radiative_cooling, slab, s.model === :flux_map ? _flux_map(cfg, mesh) : nothing,
                        water, cfg.tracer.enabled ? _tracer(cfg, mesh) : nothing,
                        cfg.particles.count > 0 ? _particles(cfg, mesh) : nothing,
                        cfg.obstacles.static === nothing ? nothing : _static_solid(cfg, mesh),
                        cfg.obstacles.penalty_time)
end

# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------

"What runs before the core's step: the slab's flux of the step, or the flux map's increment of the first level."
function _before_step!(c::SimCouplings, sim, dt)
    if c.slab !== nothing
        s = c.slab
        s.sw = max(0.0, s.sw_peak * sin(π * sim.t / s.sw_duration))
        s.Q = (surface_temperature(s.land) - s.θref) * s.c
        sim.θ_flux_bottom = s.Q
    end
    if c.flux_map !== nothing
        m = sim.mesh; Q = c.flux_map; θ = sim.θ.data
        @inbounds for j in 1:m.Ny, i in 1:m.Nx
            θ[m.Hx+i, m.Hy+j, m.Hz+1] += Q[i, j] * dt / m.dz
        end
    end
    return nothing
end

"The body force of the obstacles (`nothing`: none)."
_body_force(c::SimCouplings) = c.solid === nothing ? nothing :
    (let sm = c.solid, k = c.penalty_time
         (s, Δt) -> add_ibm_penalty!(s, sm; τ = k * Δt)
     end)

"What runs after the core's step, in the order of the file header."
function _after_step!(c::SimCouplings, sim, dt, step)
    m = sim.mesh
    if c.cooling != 0
        rate = fill(c.cooling, m.Nz) .* dt
        c.solid === nothing ? apply_radiative_cooling!(sim.θ, rate) :
                              apply_radiative_cooling!(sim.θ, rate; solid = c.solid.mask)
    end
    if c.slab !== nothing
        s = c.slab
        step_land!(s.land, s.sw, s.lw, s.ρcp * s.Q, dt)
    end
    c.water === nothing || apply_kessler_step!(sim.qv, c.water.ql, c.water.qr, dt; Vt = :spectrum)
    if c.tracer !== nothing
        t = c.tracer
        fill_halos!(t.c, sim.θ_bcs)
        fill!(t.tc.data, 0)
        advect_centered!(t.tc, sim.u, sim.v, sim.w, t.c)
        apply_linear_chemistry!(t.tc, t.c, t.τ)
        d = t.tc.data
        @inbounds for (i, j, k, r) in t.sources
            d[i, j, k] += r
        end
        interior_view(t.c) .+= dt .* interior_view(t.tc)
    end
    if c.particles !== nothing
        p = c.particles
        fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
        advance_particles!(p.ps, sim, dt)
        if step % p.every == 0
            for (n, x) in enumerate(positions(p.ps))
                push!(p.rows, (Float64(step), Float64(sim.t), Float64(n), x...))
            end
        end
    end
    return nothing
end

"The initial positions as rows of step 0."
function _record_initial!(c::SimCouplings, sim)
    p = c.particles
    p === nothing && return nothing
    for (n, x) in enumerate(positions(p.ps))
        push!(p.rows, (0.0, Float64(sim.t), Float64(n), x...))
    end
    return nothing
end

"Write `output/particles.csv` (step, t, id, x, y, z) of a run with particles."
function _write_tracks(c::SimCouplings, dir)
    p = c.particles
    (p === nothing || dir === nothing) && return nothing
    path = joinpath(dir, "particles.csv")
    open(path, "w") do io
        println(io, "step,t,id,x,y,z")
        for r in p.rows
            println(io, Int(r[1]), ",", r[2], ",", Int(r[3]), ",", r[4], ",", r[5], ",", r[6])
        end
    end
    return path
end

# ---------------------------------------------------------------------------
# diagnostics (diagnostics.csv / diagnostics.nc, prefixed by the component key)
# ---------------------------------------------------------------------------

diagnostics(s::SlabSurface, sim) = Diagnostic[
    Diagnostic("T_s", "K", () -> surface_temperature(s.land); description = "slab surface temperature"),
    Diagnostic("H", "W m-2", () -> s.ρcp * s.Q; description = "sensible heat flux of the last step, ρ c_p Q"),
    Diagnostic("sw_in", "W m-2", () -> s.sw; description = "downward shortwave of the last step")]

diagnostics(w::KesslerWater, sim) = Diagnostic[
    Diagnostic("ql_sum", "kg kg-1", () -> w.ql; reduction = :sum, description = "sum of ql over the cells"),
    Diagnostic("qr_sum", "kg kg-1", () -> w.qr; reduction = :sum, description = "sum of qr over the cells"),
    Diagnostic("ql_max", "kg kg-1", () -> w.ql; reduction = :max),
    Diagnostic("qr_max", "kg kg-1", () -> w.qr; reduction = :max)]

profile_diagnostics(w::KesslerWater, sim) = ProfileDiagnostic[
    ProfileDiagnostic("ql", "kg kg-1", "z", collect(Float64, znodes(sim.mesh, CellCenter())),
                      () -> plane_mean!(zeros(Float64, sim.mesh.Nz), w.ql); description = "horizontal mean cloud water"),
    ProfileDiagnostic("qr", "kg kg-1", "z", collect(Float64, znodes(sim.mesh, CellCenter())),
                      () -> plane_mean!(zeros(Float64, sim.mesh.Nz), w.qr); description = "horizontal mean rain water")]

diagnostics(t::PassiveTracer, sim) = Diagnostic[
    Diagnostic("max", "1", () -> t.c; reduction = :max, description = "tracer maximum"),
    Diagnostic("sum", "1", () -> t.c; reduction = :sum, description = "tracer sum over the cells")]

"The diagnostics components of the couplings that have any."
function _coupling_components(c::SimCouplings)
    comps = Pair{Symbol,Any}[]
    c.slab === nothing || push!(comps, :surface => c.slab)
    c.water === nothing || push!(comps, :micro => c.water)
    c.tracer === nothing || push!(comps, :tracer => c.tracer)
    return (; comps...)
end

"The coupling arrays for `state_fields` (the round trip compares them bit for bit)."
function _coupling_fields(c::SimCouplings)
    out = Pair{Symbol,Any}[]
    c.slab === nothing || push!(out, :T_s => [c.slab.land.T])
    if c.water !== nothing
        push!(out, :ql => c.water.ql.data); push!(out, :qr => c.water.qr.data)
    end
    c.tracer === nothing || push!(out, :tracer => c.tracer.c.data)
    c.particles === nothing ||
        push!(out, :particles => [x[d] for d in 1:3, x in positions(c.particles.ps)])
    return (; out...)
end

# ---------------------------------------------------------------------------
# preprocess static: the obstacles' static driver from a GeoJSON
# ---------------------------------------------------------------------------

"""
    preprocess_static(::SimulationDriverType, cfg; out = nothing, parent_out = nothing) -> Vector{String}

`aeolus preprocess static <config.toml>` (EX-d, #441): rasterise the
GeoJSON footprints of `[preprocess] obstacles` (`load_city_geojson`,
mesh metres, a `height` per feature) on the run's grid and write the
static driver `[obstacles] static` (or `out`), anchored at
`[obstacles] anchor` on the domain's south-west corner, with the GeoJSON
as its hashed source.
"""
function Config.preprocess_static(::SimulationDriverType, cfg::RunConfig; out = nothing, parent_out = nothing)
    parent_out === nothing || throw(ArgumentError("the simulation driver has no nest parent (--parent-out)"))
    src = cfg.preprocess.obstacles
    src === nothing && throw(ConfigError("preprocess.obstacles", "the static preprocessor needs a GeoJSON of the obstacles"))
    path = out !== nothing ? abspath(out) : cfg.obstacles.static
    path === nothing && throw(ArgumentError("[obstacles] static is \"none\": pass --out file.nc"))
    mesh = _mesh(cfg.grid)
    mesh isa RegularGrid || throw(ConfigError("grid.stretching", "the obstacles need a uniform grid"))
    lat, lon = cfg.obstacles.anchor
    write_static_driver(path, load_city_geojson(src), mesh; origin_lat = lat, origin_lon = lon,
                        origin_x = mesh.x0, origin_y = mesh.y0, source = src)
    return [path]
end
