"""
    Aeolus.RunDrivers.ObstacleChannelDriver

The `obstacle_channel` run driver (EX-d, #441): a channel of height `H`
over a floor with immersed obstacles, periodic in x and y, held at a
bulk velocity `U_b` by a PI-controlled streamwise body force, with the
time-mean streamwise velocity at the first fluid level along the centre
line and the separation and reattachment points behind and ahead of the
obstacle. Formerly `examples/sgs/surface_mounted_cube.jl` (Era VI, P56-e,
#192): the cube of Martinuzzi & Tropea (1993) at Re_H = 40 000.

Everything is in units of `H` and `U_b`. The domain extends
`floor_cells` cells below z = 0, which are solid: the floor is an
immersed wall like the obstacles, which come from a static driver (its
buildings: solid in every cell whose centre is at or below their height;
`aeolus preprocess static` writes it from a GeoJSON). The lid is a
free-slip wall. The immersed boundary is direct forcing
(`IBM.implicit_ibm_correction!`, τ = 0) after every RK3 predictor
(`step_rk3!`'s `post_predictor!`): a no-slip wall with the SGS stress as
the only wall model. The closures: Smagorinsky (`StaticSmagorinsky`), the
TKE closure and the Reynolds-stress closure with the wall cap κ ×
`IBM.wall_distance` and Moeng's factor at the first level above a
surface, zero inside the solid. The step is `cfl Δx / max|u_i|`.

`results.toml`: `x_R` (the reattachment behind the obstacle, from its
rear face, where the mean floor u turns from negative to positive), `x_F`
(the separation ahead of it, from its front face), the steps, the mean
bulk velocity over the second half of the steps and the final force;
`output/floor_u.csv`: x/H and the mean centre-line floor u.
Config: `examples/sgs/configs/surface_mounted_cube.toml` (a sweep over
the three closures).
"""
module ObstacleChannelDriver

using ...Grids: RegularGrid, CellCenter, xnodes
using ...Fields: Field, interior_view
using ...BoundaryConditions: FieldBoundaryConditions, fill_halos!
using ...Simulations: BoussinesqSimulation, step_rk3!
import ...Simulations
using ...Operators: StaticSmagorinsky
using ...IBM: SolidMask, implicit_ibm_correction!, wall_distance, first_level_cells
using ...PrognosticSGS: TKEClosure, ReynoldsStressClosure, initialize_tke!, initialize_reynolds_stress!
using ...InputDrivers: read_static_driver, write_static_driver, solid_and_classes, load_city_geojson
using ...IO: RunFolder, CheckpointSchedule, output_dir
using ...Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, Path, Optional, section, option!,
                 register_driver!, write_results, no_checkpoints
using Random: Xoshiro, randn

export ObstacleChannelDriverType, ObstacleChannelRun, reattachment

"""
    ObstacleChannelDriverType()

The `obstacle_channel` driver: `[run] legacy = "obstacle_channel"`.
Sections `[grid]`, `[flow]`, `[sgs]`, `[obstacles]`, `[stepper]` and
`[preprocess]`; `aeolus options obstacle_channel` prints them.
"""
struct ObstacleChannelDriverType <: AbstractRunDriver end

Config.driver_name(::ObstacleChannelDriverType) = "obstacle_channel"

function Config.driver_sections(::ObstacleChannelDriverType)
    grid = section("grid"; doc = "The channel, in units of its height H.") do s
        option!(s, "cells_per_h", Int; default = 16, range = (1, 1024), doc = "cells per H (in every direction)")
        option!(s, "extent", NTuple{3,Float64}; default = (10.0, 6.0, 2.0), range = (0, Inf), open_lower = true,
                units = "1", doc = "L_x, L_y and the height of the lid above the floor (H)")
        option!(s, "halo", Int; default = 3, range = (1, 8), doc = "halo cells")
    end
    flow = section("flow"; doc = "The flow (U_b = 1).") do s
        option!(s, "re_h", Float64; default = 40000.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "Reynolds number U_b H/ν")
        option!(s, "kp", Float64; default = 2.0, range = (0, Inf), units = "1",
                doc = "proportional gain of the bulk-velocity controller, F = kp e + ki ∫e dt, e = 1 − U_b")
        option!(s, "ki", Float64; default = 0.5, range = (0, Inf), units = "1", doc = "integral gain")
        option!(s, "noise", Float64; default = 0.1, range = (0, Inf), units = "1",
                doc = "initial velocity: U_b + noise × N(0, 1) in u, noise × N(0, 1) in v and w (one randn array per component, Xoshiro([run] seed))")
    end
    sgs = section("sgs"; doc = "The SGS closure.") do s
        option!(s, "closure", Symbol; default = :smagorinsky, allowed = [:smagorinsky, :tke, :rsm],
                doc = "smagorinsky (StaticSmagorinsky(c_s)), tke (TKEClosure) or rsm (ReynoldsStressClosure); the prognostic ones with the solid mask, the wall distance cap and the first-level factor, seeded with e_min")
        option!(s, "c_s", Float64; units = "1", default = 0.1, range = (0, Inf), open_lower = true,
                doc = "smagorinsky: the coefficient (0.1, the wall-bounded value)")
        option!(s, "e_min", Float64; default = 1e-4, range = (0, Inf), units = "1",
                doc = "tke, rsm: the initial SGS energy floor (U_b²)")
    end
    ob = section("obstacles"; doc = "The floor and the obstacles.") do s
        option!(s, "static", Path; required = true,
                doc = "the static driver of the obstacles (buildings), on the run's horizontal grid; hashed under [inputs]")
        option!(s, "anchor", NTuple{2,Float64}; default = (0.0, 0.0), units = "degrees",
                doc = "latitude, longitude of the domain's south-west corner (the static driver's anchor; checked, not used)")
        option!(s, "floor_cells", Int; default = 1, range = (1, 64),
                doc = "solid cells below z = 0: the floor as an immersed wall")
    end
    stp = section("stepper"; doc = "SSP-RK3 at the CFL step; the statistics.") do s
        option!(s, "cfl", Float64; units = "1", default = 0.4, range = (0, Inf), open_lower = true,
                doc = "dt = cfl Δx / max|u_i|")
        option!(s, "t_spin", Float64; default = 30.0, range = (0, Inf), units = "1", doc = "spin-up (H/U_b)")
        option!(s, "t_avg", Float64; default = 30.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "averaging time of the floor velocity after the spin-up (H/U_b)")
    end
    pre = section("preprocess"; doc = "Inputs of the static preprocessor.") do s
        option!(s, "obstacles", Optional{Path};
                doc = "a GeoJSON FeatureCollection of the obstacle footprints (H) with a `height` each: aeolus preprocess static writes obstacles.static from it")
    end
    return Section[grid, flow, sgs, ob, stp, pre]
end

function _checks(cfg)
    e = no_checkpoints(cfg)
    (cfg.sgs.closure !== :smagorinsky && cfg.sgs.c_s != 0.1) &&
        push!(e, ConfigError("sgs.c_s", "c_s belongs to closure = \"smagorinsky\""))
    (cfg.sgs.closure === :smagorinsky && cfg.sgs.e_min != 1e-4) &&
        push!(e, ConfigError("sgs.e_min", "e_min belongs to the prognostic closures"))
    return e
end
Config.driver_checks(::ObstacleChannelDriverType) = Any[
    """`[checkpoint]` stays off (the run is one call)
    `sgs.c_s` only with smagorinsky, `sgs.e_min` only with tke and rsm""" => _checks]

"The mesh of the channel: the floor cells below z = 0."
function _mesh(cfg)
    g = cfg.grid; NH = g.cells_per_h; Lx, Ly, Lz = g.extent
    nf = cfg.obstacles.floor_cells; h = 1 / NH
    return RegularGrid(; size = (round(Int, Lx * NH), round(Int, Ly * NH), round(Int, Lz * NH) + nf),
                       extent = (Lx, Ly, Lz + nf * h), origin = (0.0, 0.0, -nf * h), halo = (g.halo, g.halo, g.halo))
end

"""
    ObstacleChannelRun

What the `obstacle_channel` driver builds: the simulation `sim`, the
`SolidMask` `sm`, the `closure`, the resolved `cfg`, and after the run
`x` and `u_floor` (the mean centre-line floor velocity), `x_R`, `x_F`,
`steps`, `U` (the bulk velocity before each step) and `force`.
"""
mutable struct ObstacleChannelRun
    sim::Any
    sm::SolidMask
    closure::Any
    cfg::RunConfig
    x::Vector{Float64}
    u_floor::Vector{Float64}
    x_R::Float64
    x_F::Float64
    steps::Int
    U::Vector{Float64}
    force::Float64
end

function Config.build(::ObstacleChannelDriverType, cfg::RunConfig)
    mesh = _mesh(cfg)
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(mesh; ν = 1 / cfg.flow.re_h, bcs = bcs)
    lat, lon = cfg.obstacles.anchor
    st = read_static_driver(cfg.obstacles.static; mesh = mesh, origin_lat = lat, origin_lon = lon,
                            origin_x = mesh.x0, origin_y = mesh.y0)
    sm, _ = solid_and_classes(st, mesh; tag_walls = false)
    for k in 1:cfg.obstacles.floor_cells
        sm.mask[mesh.Hx+1:mesh.Hx+mesh.Nx, mesh.Hy+1:mesh.Hy+mesh.Ny, mesh.Hz+k] .= true
    end
    mf = Field(CellCenter(), mesh)
    mf.data .= sm.mask
    fill_halos!(mf, FieldBoundaryConditions(:periodic, :periodic, :noflux))
    mask = mf.data .> 0.5
    d = wall_distance(sm; bottom = false)
    fl = first_level_cells(sm; bottom = false)
    s = cfg.sgs
    closure = s.closure === :smagorinsky ? StaticSmagorinsky(s.c_s) :
              s.closure === :tke ? TKEClosure(mesh; mask = mask, wall_distance = d, first_level = fl) :
              ReynoldsStressClosure(mesh; mask = mask, wall_distance = d, first_level = fl)
    rng = Xoshiro(cfg.run.seed)
    a = cfg.flow.noise
    interior_view(sim.u) .= 1 .+ a .* randn(rng, size(interior_view(sim.u)))
    interior_view(sim.v) .= a .* randn(rng, size(interior_view(sim.v)))
    interior_view(sim.w) .= a .* randn(rng, size(interior_view(sim.w)))
    implicit_ibm_correction!(sim, sm)
    return ObstacleChannelRun(sim, sm, closure, cfg, Float64[], Float64[], NaN, NaN, 0, Float64[], 0.0)
end

# bulk velocity over the fluid
function _bulk_u(sim, sm)
    m = sim.mesh
    s = 0.0; n = 0
    u = sim.u.data; mk = sm.mask
    @inbounds for k in m.Hz+1:m.Hz+m.Nz, j in m.Hy+1:m.Hy+m.Ny, i in m.Hx+1:m.Hx+m.Nx
        mk[i, j, k] && continue
        s += u[i, j, k]; n += 1
    end
    return s / n
end

"""
    reattachment(x, u, x_front, x_rear) -> (x_R, x_F)

Behind the obstacle: the first sign change of the mean floor `u` from
negative to positive downstream of the rear face `x_rear` (linear
interpolation), measured from the rear face. Ahead of it: the first point
upstream of the front face `x_front` where `u` turns negative, measured
from the front face. `NaN` where there is none.
"""
function reattachment(x, u, x_front, x_rear)
    xR = NaN
    for i in findfirst(>(x_rear), x):length(x)-1
        if u[i] < 0 && u[i+1] >= 0
            xR = x[i] + (x[i+1] - x[i]) * (-u[i]) / (u[i+1] - u[i]) - x_rear
            break
        end
    end
    xF = NaN
    i0 = findlast(<(x_front), x)
    if u[i0] < 0
        for i in i0:-1:2
            if u[i-1] >= 0 && u[i] < 0
                xF = x_front - (x[i-1] + (x[i] - x[i-1]) * u[i-1] / (u[i-1] - u[i]))
                break
            end
        end
    end
    return xR, xF
end

function Simulations.run!(::ObstacleChannelDriverType, r::ObstacleChannelRun, rf::RunFolder, ::CheckpointSchedule;
                          callback = nothing)
    sim, sm, closure, cfg = r.sim, r.sm, r.closure, r.cfg
    m = sim.mesh
    T_SPIN, T_AVG = cfg.stepper.t_spin, cfg.stepper.t_avg
    kp, ki, cfl = cfg.flow.kp, cfg.flow.ki, cfg.stepper.cfl
    cfg.sgs.closure === :tke && initialize_tke!(closure, sim; e_min = cfg.sgs.e_min)
    cfg.sgs.closure === :rsm && initialize_reynolds_stress!(closure, sim; e_min = cfg.sgs.e_min)
    Fi = Ref(0.0); Fp = Ref(0.0)
    force = (s, Δt) -> (interior_view(s.tu) .+= Fi[] + Fp[]; nothing)
    ibm = (s, Δt) -> (implicit_ibm_correction!(s, sm); nothing)
    kf = cfg.obstacles.floor_cells + 1                 # the first fluid level (z = Δz/2)
    jc = (m.Ny ÷ 2, m.Ny ÷ 2 + 1)                      # the two cells about the centre line
    ufl = zeros(m.Nx); tavg = 0.0
    U = Float64[]; nsteps = 0
    T_END = T_SPIN + T_AVG
    wall = @elapsed while sim.t < T_END - 1e-9
        ub = _bulk_u(sim, sm)
        umax = maximum(f -> maximum(abs, interior_view(f)), (sim.u, sim.v, sim.w))
        dt = min(cfl * m.dx / umax, T_END - sim.t)
        # PI controller on U_b = 1: F = K_p e + K_i ∫e dt
        Fp[] = kp * (1 - ub)
        Fi[] += ki * dt * (1 - ub)
        step_rk3!(sim, dt; sgs = closure, body_force! = force, post_predictor! = ibm)
        nsteps += 1
        push!(U, ub)
        all(isfinite, interior_view(sim.u)) || error("$(cfg.sgs.closure): non-finite velocity at t = $(sim.t)")
        if sim.t > T_SPIN
            u = interior_view(sim.u)
            # u at the cell centre in x: the average of the two faces
            for i in 1:m.Nx
                ip = i == m.Nx ? 1 : i + 1
                ufl[i] += dt * sum((u[i, j, kf] + u[ip, j, kf]) / 2 for j in jc) / 2
            end
            tavg += dt
        end
        nsteps % 200 == 0 && @info "obstacle channel" t = round(sim.t; digits = 2) dt = round(dt; digits = 5) U_b = ub F = Fi[] + Fp[]
        callback === nothing || callback(r, nsteps)
    end
    ufl ./= tavg
    x = collect(Float64, xnodes(m, CellCenter()))
    # the obstacle's streamwise faces: the solid cells of the first fluid level on the centre line
    jm = m.Hy + jc[1]
    solid = [i for i in 1:m.Nx if sm.mask[m.Hx+i, jm, m.Hz+kf]]
    isempty(solid) && error("no obstacle on the centre line at the first fluid level")
    x_front = m.x0 + (first(solid) - 1) * m.dx
    x_rear = m.x0 + last(solid) * m.dx
    r.x_R, r.x_F = reattachment(x, ufl, x_front, x_rear)
    r.x, r.u_floor, r.steps, r.U, r.force = x, ufl, nsteps, U, Fi[] + Fp[]
    Ub = sum(U[end ÷ 2:end]) / length(U[end ÷ 2:end])
    @info "obstacle channel" closure = cfg.sgs.closure x_R = r.x_R x_F = r.x_F steps = nsteps wall_s = round(wall; digits = 1) mean_U_b = Ub
    write_results(rf, Dict{String,Any}("x_R" => r.x_R, "x_F" => r.x_F, "x_front" => x_front, "x_rear" => x_rear,
                                       "steps" => nsteps, "wall_seconds" => wall, "wall_seconds_per_step" => wall / nsteps,
                                       "mean_U_b" => Ub, "force" => r.force, "closure" => String(cfg.sgs.closure)))
    open(joinpath(output_dir(rf), "floor_u.csv"), "w") do io
        println(io, "x_over_H,u_mean_floor")
        for i in eachindex(x)
            println(io, x[i], ",", ufl[i])
        end
    end
    return r
end

"""
    state_fields(r::ObstacleChannelRun) -> NamedTuple

The velocity and the mean centre-line floor velocity.
"""
Config.state_fields(r::ObstacleChannelRun) = (u = r.sim.u.data, v = r.sim.v.data, w = r.sim.w.data,
                                             u_floor = r.u_floor)

"""
    preprocess_static(::ObstacleChannelDriverType, cfg; out = nothing, parent_out = nothing) -> Vector{String}

`aeolus preprocess static <config.toml>`: rasterise the GeoJSON of
`[preprocess] obstacles` on the run's horizontal grid and write the static
driver `[obstacles] static` (or `out`), the GeoJSON hashed as its source.
"""
function Config.preprocess_static(::ObstacleChannelDriverType, cfg::RunConfig; out = nothing, parent_out = nothing)
    parent_out === nothing || throw(ArgumentError("the obstacle_channel driver has no nest parent (--parent-out)"))
    src = cfg.preprocess.obstacles
    src === nothing && throw(ConfigError("preprocess.obstacles", "the static preprocessor needs a GeoJSON of the obstacles"))
    path = out !== nothing ? abspath(out) : cfg.obstacles.static
    mesh = _mesh(cfg)
    lat, lon = cfg.obstacles.anchor
    write_static_driver(path, load_city_geojson(src), mesh; origin_lat = lat, origin_lon = lon,
                        origin_x = mesh.x0, origin_y = mesh.y0, source = src)
    return [path]
end

register_driver!(ObstacleChannelDriverType())

end # module
