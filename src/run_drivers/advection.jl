"""
    Aeolus.RunDrivers.AdvectionDriver

The `advection` run driver (EX-d, #441): a passive scalar carried by a
prescribed, steady (kinematic) flow on a periodic box, integrated with the
SSP-RK3 of `Aeolus.TimeSteppers` and the advection kernels of
`Aeolus.Operators`. No dynamics: the velocity never changes. It is the
end-to-end check of the mesh, the fields, the halo fill, an advection
kernel and the time stepper (formerly `examples/scalar_advection_2d.jl`;
config `examples/configs/scalar_advection_2d.toml`).

`results.toml`: the time series `t`, `mass` (∫ϕ dV) and `max` (max ϕ),
every `nsteps ÷ records` steps,
and at the end `t_end` and the position of the maximum (`peak_x`, `peak_y`).
"""
module AdvectionDriver

using ...Grids: RegularGrid, CellCenter, xnodes, ynodes
using ...Fields: Field, set!, interior_view
using ...BoundaryConditions: FieldBoundaryConditions, fill_halos!
using ...Operators: advect_upwind!, advect_centered!
using ...TimeSteppers: RK3
import ...TimeSteppers
using ...IO: RunFolder, CheckpointSchedule
import ...Simulations
using ...Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, section, option!,
                 register_driver!, write_results, no_checkpoints

export AdvectionDriverType, AdvectionRun

"""
    AdvectionDriverType()

The `advection` driver: `[run] model = "advection"`. Sections `[grid]`,
`[flow]` (the prescribed velocity), `[scalar]` (the initial scalar),
`[advection]` (the kernel), `[stepper]` and `[diagnostics]`; `aeolus
options advection` prints them.
"""
struct AdvectionDriverType <: AbstractRunDriver end

Config.driver_name(::AdvectionDriverType) = "advection"
Config.config_dir(::AdvectionDriverType) = normpath(joinpath(@__DIR__, "..", "..", "examples", "configs"))

function Config.driver_sections(::AdvectionDriverType)
    grid = section("grid"; doc = "The periodic box and its mesh.") do s
        option!(s, "size", NTuple{3,Int}; default = (128, 128, 1), range = (1, 65536), doc = "cells in x, y, z")
        option!(s, "extent", NTuple{3,Float64}; default = (1.0, 1.0, 1.0), range = (0, Inf), open_lower = true,
                units = "m", doc = "domain size Lx, Ly, Lz")
        option!(s, "origin", NTuple{3,Float64}; default = (0.0, 0.0, 0.0), units = "m",
                doc = "the south-west-bottom corner")
        option!(s, "halo", NTuple{3,Int}; default = (2, 2, 2), range = (1, 8), doc = "halo cells per side")
    end
    flow = section("flow"; doc = "The prescribed velocity (cell-centred, steady).") do s
        option!(s, "kind", Symbol; default = :solid_body_rotation, allowed = [:solid_body_rotation, :uniform],
                doc = "solid_body_rotation (u = −ω (y − y_c), v = ω (x − x_c), w = 0) or uniform (velocity)")
        option!(s, "omega", Float64; default = 2π, units = "s^-1", doc = "solid_body_rotation: the angular velocity ω")
        option!(s, "center", NTuple{2,Float64}; default = (0.0, 0.0), units = "m",
                doc = "solid_body_rotation: the axis (x_c, y_c)")
        option!(s, "velocity", NTuple{3,Float64}; default = (0.0, 0.0, 0.0), units = "m s^-1",
                doc = "uniform: (u, v, w)")
    end
    sc = section("scalar"; doc = "The initial scalar ϕ.") do s
        option!(s, "kind", Symbol; default = :gaussian, allowed = [:gaussian],
                doc = "gaussian: exp(−((x − x₀)² + (y − y₀)²)/(2σ²)), a column (independent of z)")
        option!(s, "center", NTuple{2,Float64}; default = (0.25, 0.0), units = "m", doc = "(x₀, y₀)")
        option!(s, "sigma", Float64; default = 0.08, range = (0, Inf), open_lower = true, units = "m",
                doc = "the width σ")
    end
    adv = section("advection"; doc = "The advection kernel.") do s
        option!(s, "scheme", Symbol; default = :upwind, allowed = [:upwind, :centered],
                doc = "upwind (advect_upwind!, first order) or centered (advect_centered!, second order)")
    end
    stp = section("stepper"; doc = "SSP-RK3 at a fixed step.") do s
        option!(s, "cfl", Float64; default = 0.4, units = "1", range = (0, Inf), open_lower = true,
                doc = "Courant number of the largest velocity component: dt = cfl Δx / max|u_i| (Δx: the x spacing)")
        option!(s, "end_time", Float64; default = 1.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "simulated time; dt is shortened to end_time / ceil(end_time / dt)")
    end
    dg = section("diagnostics"; doc = "The time series of results.toml.") do s
        option!(s, "records", Int; default = 8, range = (1, typemax(Int)),
                doc = "mass and maximum every max(1, nsteps ÷ records) steps (and at t = 0)")
    end
    return Section[grid, flow, sc, adv, stp, dg]
end

Config.driver_checks(::AdvectionDriverType) = Any[
    "`[checkpoint]` stays off (the run is one call)" => no_checkpoints]

"""
    AdvectionRun

What the `advection` driver builds: the `mesh`, the prescribed velocity
`u, v, w` (cell-centred), the scalar `state = (ϕ = …,)`, the `RK3`
stepper, the halo conditions `bcs`, the step `dt` and count `nsteps`,
the time `t` and the resolved `cfg`.
"""
mutable struct AdvectionRun
    mesh::Any
    u::Field
    v::Field
    w::Field
    state::NamedTuple
    stepper::RK3
    bcs::FieldBoundaryConditions
    dt::Float64
    nsteps::Int
    t::Float64
    cfg::RunConfig
end

# the largest velocity component over the box (for the step)
function _umax(cfg, mesh)
    f = cfg.flow
    f.kind === :uniform && return maximum(abs, f.velocity)
    xc, yc = f.center
    d = max(abs(mesh.x0 - xc), abs(mesh.x0 + mesh.Lx - xc), abs(mesh.y0 - yc), abs(mesh.y0 + mesh.Ly - yc))
    return abs(f.omega) * d
end

function Config.build(::AdvectionDriverType, cfg::RunConfig)
    g = cfg.grid
    mesh = RegularGrid(Float64; size = g.size, extent = g.extent, halo = g.halo, origin = g.origin)
    u = Field(CellCenter(), mesh); v = Field(CellCenter(), mesh); w = Field(CellCenter(), mesh)
    f = cfg.flow
    if f.kind === :solid_body_rotation
        ω = f.omega; xc, yc = f.center
        set!(u, (x, y, z) -> -ω * (y - yc))
        set!(v, (x, y, z) -> ω * (x - xc))
    else
        U, V, W = f.velocity
        set!(u, (x, y, z) -> U); set!(v, (x, y, z) -> V); set!(w, (x, y, z) -> W)
    end
    ϕ = Field(CellCenter(), mesh)
    x0, y0 = cfg.scalar.center; σ = cfg.scalar.sigma
    set!(ϕ, (x, y, z) -> exp(-((x - x0)^2 + (y - y0)^2) / (2σ^2)))
    bcs = FieldBoundaryConditions(:periodic)
    fill_halos!(u, bcs); fill_halos!(v, bcs); fill_halos!(w, bcs); fill_halos!(ϕ, bcs)
    state = (ϕ = ϕ,)
    T = cfg.stepper.end_time
    dt = cfg.stepper.cfl * mesh.dx / _umax(cfg, mesh)
    n = ceil(Int, T / dt)
    return AdvectionRun(mesh, u, v, w, state, RK3(state), bcs, T / n, n, 0.0, cfg)
end

function _tendency(r::AdvectionRun)
    kernel = r.cfg.advection.scheme === :upwind ? advect_upwind! : advect_centered!
    return function (k, state, t)
        fill!(k.ϕ.data, 0)
        fill_halos!(state.ϕ, r.bcs)
        kernel(k.ϕ, r.u, r.v, r.w, state.ϕ)
        return k
    end
end

_mass(r) = sum(interior_view(r.state.ϕ)) * r.mesh.dx * r.mesh.dy * r.mesh.dz
_max(r) = maximum(interior_view(r.state.ϕ))

function Simulations.run!(::AdvectionDriverType, r::AdvectionRun, rf::RunFolder, ck::CheckpointSchedule;
                          callback = nothing)
    tend! = _tendency(r)
    every = max(1, r.nsteps ÷ r.cfg.diagnostics.records)
    ts = [r.t]; mass = [_mass(r)]; mx = [_max(r)]
    @info "advection" nsteps = r.nsteps dt = r.dt
    for n in 1:r.nsteps
        TimeSteppers.step!(r.stepper, r.state, tend!, r.dt, r.t)
        r.t += r.dt
        if n % every == 0
            push!(ts, r.t); push!(mass, _mass(r)); push!(mx, _max(r))
        end
        callback === nothing || callback(r, n)
    end
    ϕf = interior_view(r.state.ϕ)
    xs = collect(xnodes(r.mesh, CellCenter())); ys = collect(ynodes(r.mesh, CellCenter()))
    I = argmax(ϕf[:, :, 1])
    write_results(rf, Dict("t" => ts, "mass" => mass, "max" => mx, "dt" => r.dt, "nsteps" => r.nsteps,
                           "t_end" => r.t, "peak_x" => xs[I[1]], "peak_y" => ys[I[2]]))
    return r
end

"""
    state_fields(r::AdvectionRun) -> (ϕ = …,)

The scalar with its halos.
"""
Config.state_fields(r::AdvectionRun) = (ϕ = r.state.ϕ.data,)

register_driver!(AdvectionDriverType())

end # module
