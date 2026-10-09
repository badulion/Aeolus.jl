"""
    Aeolus.RunDrivers.FlatPlateDriver

The `flat_plate` run driver (EX-d, #441): bypass transition of a
spatially developing zero-pressure-gradient flat-plate boundary layer
under free-stream turbulence (Era VI, P59-d, #185; formerly
`examples/transition/flat_plate_t3a.jl`, the ERCOFTAC T3A-like case at
Tu = 3.3 %). In units of the inflow displacement thickness ``δ^*_0`` and
``U_∞ = 1``: an x-periodic box whose last `fringe_length` relaxes the
outflow to the Blasius inflow (Nordström et al. 1999) and carries the
free-stream turbulence in (Brandt, Schlatter & Henningson 2004)
(`Transition.flat_plate_setup`, `fringe_forcing`); a no-slip plate, a
free-stream top, RK3 at a fixed step and no SGS model. The disturbance is
`modes` random Fourier modes of a von Kármán spectrum
(`FreeStreamDisturbance`), kept out of the inflow layer below
`damping_height` × δ₉₉ in the fringe.

The statistics over the last `average_steps` steps: ``c_f(x)``, the mean
and rms streamwise velocity per (x, z), and from them (all
`Aeolus.Transition`) the onset (`transition_onset`: the minimum of
``c_f`` upstream of the persistent rise), ``Re_θ`` at onset with θ
integrated to the δ₉₉.₅ edge against Mayle (1991) and Abu-Ghannam & Shaw
(1980), the downstream ``c_f`` against ``0.0576\\,Re_x^{-1/5}``, the shape
factor H(x) and the free-stream Tu(x). `results.toml` holds the numbers
and the plate profiles. Config: `examples/transition/configs/flat_plate_t3a.toml`.
"""
module FlatPlateDriver

using ...Grids: xnodes, FaceX
using ...Fields: interior_view
using ...Simulations: step_rk3!, max_divergence
import ...Simulations
using ...Operators: smooth_step
using ...Transition: flat_plate_setup, FreeStreamDisturbance, blasius_solution, disturbance_fringe_update,
                     fringe_forcing, set_blasius!, skin_friction, boundary_layer_edge, momentum_thickness,
                     shape_factor, transition_onset, blasius_skin_friction, turbulent_skin_friction,
                     onset_reynolds_theta_mayle, onset_reynolds_theta_abu_ghannam_shaw
using ...IO: RunFolder, CheckpointSchedule
using ...Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, section, option!,
                 register_driver!, write_results, no_checkpoints

export FlatPlateDriverType, FlatPlateRun

"""
    FlatPlateDriverType()

The `flat_plate` driver: `[run] driver = "flat_plate"`. Sections
`[plate]`, `[grid]`, `[disturbance]`, `[stepper]` and `[analysis]`;
`aeolus options flat_plate` prints them.
"""
struct FlatPlateDriverType <: AbstractRunDriver end

Config.driver_name(::FlatPlateDriverType) = "flat_plate"
Config.config_dir(::FlatPlateDriverType) = normpath(joinpath(@__DIR__, "..", "..", "examples", "transition", "configs"))

function Config.driver_sections(::FlatPlateDriverType)
    pl = section("plate"; doc = "The plate and its box, in units of the inflow displacement thickness δ*₀ (U∞ = 1).") do s
        option!(s, "re_delta0", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "Re_δ*₀ = U∞ δ*₀/ν at the inflow")
        option!(s, "length", Float64; default = 600.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "L_x, the fringe included")
        option!(s, "width", Float64; default = 36.0, range = (0, Inf), open_lower = true, units = "1", doc = "L_y")
        option!(s, "height", Float64; default = 24.0, range = (0, Inf), open_lower = true, units = "1", doc = "L_z")
        option!(s, "fringe_length", Float64; default = 80.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "the last part of the box: the fringe")
        option!(s, "plate_margin", Float64; default = 20.0, range = (0, Inf), units = "1",
                doc = "the statistics stop this far ahead of the fringe (clear of its rise)")
    end
    grid = section("grid"; doc = "The mesh.") do s
        option!(s, "size", NTuple{3,Int}; default = (576, 64, 96), range = (1, 65536), doc = "cells in x, y, z")
    end
    dist = section("disturbance"; doc = "The free-stream turbulence (FreeStreamDisturbance).") do s
        option!(s, "tu", Float64; units = "1", default = 0.033, range = (0, Inf), doc = "turbulence intensity Tu")
        option!(s, "length_scale", Float64; default = 10.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "integral scale L11 (δ*₀) of the von Kármán spectrum")
        option!(s, "modes", Int; default = 200, range = (1, typemax(Int)), doc = "random Fourier modes")
        option!(s, "seed", Int; default = 1, range = (0, typemax(Int)), doc = "seed of the modes")
        option!(s, "kappa_cells", Float64; units = "1", default = 3.0, range = (0, Inf), open_lower = true,
                doc = "the largest wavenumber is 2π/(kappa_cells × max(Δx, Δy, Δz))")
        option!(s, "damping_height", Float64; units = "1", default = 2.0, range = (0, Inf), open_lower = true,
                doc = "the modes are faded in above the inflow layer: smooth_step(z / (damping_height × δ₉₉))")
    end
    stp = section("stepper"; doc = "SSP-RK3 at a fixed step.") do s
        option!(s, "dt", Float64; default = 1.0, range = (0, Inf), open_lower = true, units = "1", doc = "the step (δ*₀/U∞)")
        option!(s, "nsteps", Int; default = 1500, range = (1, typemax(Int)), doc = "steps")
        option!(s, "average_steps", Int; default = 600, range = (1, typemax(Int)),
                doc = "the statistics are averaged over the last this many steps")
    end
    an = section("analysis"; doc = "The diagnostics of the averaged layer.") do s
        option!(s, "edge_fraction", Float64; units = "1", default = 0.995, range = (0, 1), open_lower = true,
                doc = "θ and H are integrated to the edge where the mean u reaches this fraction of its maximum")
        option!(s, "downstream_re_x", Float64; units = "1", default = 1.2e5, range = (0, Inf),
                doc = "the turbulent c_f is compared with 0.0576 Re_x^(−1/5) from this Re_x to the end of the plate")
    end
    return Section[pl, grid, dist, stp, an]
end

function _checks(cfg)
    e = no_checkpoints(cfg)
    cfg.stepper.average_steps <= cfg.stepper.nsteps ||
        push!(e, ConfigError("stepper.average_steps", "more averaging steps than steps"))
    cfg.plate.fringe_length + cfg.plate.plate_margin < cfg.plate.length ||
        push!(e, ConfigError("plate.fringe_length", "the fringe and the margin leave no plate"))
    return e
end
Config.driver_checks(::FlatPlateDriverType) = Any[
    """`[checkpoint]` stays off (the run is one call)
    `stepper.average_steps` ≤ `stepper.nsteps`; the fringe and the margin are shorter than the box""" => _checks]

"""
    FlatPlateRun

What the `flat_plate` driver builds: the case `c` of
`flat_plate_setup` (its `sim`), the fringe forcing `force!`, the resolved
`cfg`, and after the run the averages `cf`, `ū`, `urms`.
"""
mutable struct FlatPlateRun
    c::Any
    force!::Any
    cfg::RunConfig
    cf::Vector{Float64}
    ū::Matrix{Float64}
    urms::Matrix{Float64}
end

function Config.build(::FlatPlateDriverType, cfg::RunConfig)
    p = cfg.plate; d = cfg.disturbance
    Nx, Ny, Nz = cfg.grid.size
    c = flat_plate_setup(; Re_δ0 = p.re_delta0, Lx = p.length, Ly = p.width, H = p.height, Nx = Nx, Ny = Ny, Nz = Nz,
                         fringe_length = p.fringe_length)
    sim, g = c.sim, c.sim.mesh
    fst = FreeStreamDisturbance(d.tu, d.length_scale, d.modes; seed = d.seed, period_y = g.Ly,
                                κ_max = 2π / (d.kappa_cells * max(g.dx, g.dy, g.dz)))
    δ99 = blasius_solution().eta99 * sqrt(c.ν * c.x_in)
    target = deepcopy(c.target)
    h = d.damping_height
    update! = disturbance_fringe_update(fst, c.fringe, c.target; damping = z -> smooth_step(z / (h * δ99)))
    force! = fringe_forcing(c.fringe, target; update! = update!)
    set_blasius!(sim; x_le = c.x_le, x_min = c.x_in)
    return FlatPlateRun(c, force!, cfg, Float64[], zeros(0, 0), zeros(0, 0))
end

function Simulations.run!(::FlatPlateDriverType, r::FlatPlateRun, rf::RunFolder, ::CheckpointSchedule;
                          callback = nothing)
    c = r.c; sim = c.sim; g = sim.mesh; cfg = r.cfg
    nsteps, navg, dt = cfg.stepper.nsteps, cfg.stepper.average_steps, cfg.stepper.dt
    cf = zeros(g.Nx); ū = zeros(g.Nx, g.Nz); u2 = zeros(g.Nx, g.Nz); nacc = 0
    t0 = time()
    for n in 1:nsteps
        step_rk3!(sim, dt; body_force! = r.force!)
        if n > nsteps - navg
            nacc += 1
            cf .+= skin_friction(sim; U_e = 1.0)
            ui = interior_view(sim.u)
            ū .+= dropdims(sum(ui; dims = 2); dims = 2) ./ g.Ny
            u2 .+= dropdims(sum(ui .^ 2; dims = 2); dims = 2) ./ g.Ny
        end
        n % 250 == 0 && @info "flat plate" step = n t = sim.t max_u = maximum(abs, interior_view(sim.u)) max_div = max_divergence(sim)
        callback === nothing || callback(r, n)
    end
    wall = time() - t0
    cf ./= nacc; ū ./= nacc; u2 ./= nacc
    urms = sqrt.(max.(u2 .- ū .^ 2, 0.0))
    r.cf, r.ū, r.urms = cf, ū, urms

    # the analysis of the averaged layer
    x = collect(xnodes(g, FaceX()))
    Re_x = (x .- c.x_le) ./ c.ν
    Lp = g.Lx - cfg.plate.fringe_length                    # plate length (fringe excluded)
    plate = findall(<(Lp - cfg.plate.plate_margin), x)       # stay clear of the fringe rise
    # boundary-layer integrals to the edge (the free stream carries a
    # 0.5 % mean deficit from the disturbance forcing)
    ke = boundary_layer_edge(ū; fraction = cfg.analysis.edge_fraction)
    Ue = vec(maximum(ū; dims = 2))                           # free-stream velocity per x
    θ = momentum_thickness(ū, g.dz; U_e = Ue, kmax = ke)
    H = shape_factor(ū, g.dz; U_e = Ue, kmax = ke)
    # free-stream Tu(x): streamwise rms over the levels from the edge to 2 δ*₀
    # below the top (the six levels below that if the layer reaches it)
    function fs_levels(i)
        rr = (ke[i] + 1):(g.Nz - 4)
        return isempty(rr) ? ((g.Nz - 9):(g.Nz - 4)) : rr
    end
    Tu_x = [sum(urms[i, k] for k in fs_levels(i)) / length(fs_levels(i)) for i in 1:g.Nx]
    xo, io = transition_onset(x[plate], cf[plate], blasius_skin_friction.(Re_x[plate]))
    Tu = cfg.disturbance.tu
    # (no onset on the plate, io = 0: a layer still laminar at its end)
    Reθ_on = io == 0 ? NaN : θ[io] / c.ν
    Re_on = io == 0 ? NaN : Re_x[io]
    Reθ_blasius = 2 * blasius_solution().fpp0 * sqrt(Re_on)
    Mayle = onset_reynolds_theta_mayle(100Tu)
    AGS = onset_reynolds_theta_abu_ghannam_shaw(100Tu)
    down = findall(i -> Re_x[i] >= cfg.analysis.downstream_re_x && i in plate, eachindex(x))
    ratio_t = cf[down] ./ turbulent_skin_friction.(Re_x[down])
    @info "flat plate" onset_x = xo Re_theta = Reθ_on Mayle AGS cf_ratio_mean = isempty(ratio_t) ? NaN : sum(ratio_t) / length(ratio_t)
    i5 = argmin(abs.(Re_x .- 1e5))
    out = Dict{String,Any}(
        "steps" => nsteps, "wall_seconds" => wall, "wall_seconds_per_step" => wall / nsteps,
        "tu_inflow" => Tu_x[1], "tu_re_x_1e5" => Tu_x[i5], "tu_plate_end" => Tu_x[plate[end]],
        "onset_x" => xo, "onset_re_x" => Re_on, "onset_re_theta" => Reθ_on, "onset_re_theta_blasius" => Reθ_blasius,
        "mayle" => Mayle, "abu_ghannam_shaw" => AGS,
        "plate" => Dict{String,Any}("x" => x[plate], "re_x" => Re_x[plate], "cf" => cf[plate],
                                    "cf_laminar" => blasius_skin_friction.(Re_x[plate]),
                                    "cf_turbulent" => turbulent_skin_friction.(Re_x[plate]),
                                    "shape_factor" => H[plate], "tu" => Tu_x[plate], "theta" => θ[plate]))
    if !isempty(down)
        out["downstream"] = Dict{String,Any}(
            "re_x" => [Re_x[down[1]], Re_x[down[end]]], "cf_ratio_min" => minimum(ratio_t),
            "cf_ratio_max" => maximum(ratio_t), "cf_ratio_mean" => sum(ratio_t) / length(ratio_t),
            "shape_factor_min" => minimum(H[down]), "shape_factor_max" => maximum(H[down]))
    end
    write_results(rf, out)
    return r
end

"""
    state_fields(r::FlatPlateRun) -> NamedTuple

The velocity with its halos, and the averaged c_f, mean and rms u.
"""
Config.state_fields(r::FlatPlateRun) = (u = r.c.sim.u.data, v = r.c.sim.v.data, w = r.c.sim.w.data, cf = r.cf,
                                        u_mean = r.ū, u_rms = r.urms)

register_driver!(FlatPlateDriverType())

end # module
