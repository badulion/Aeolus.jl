"""
    Aeolus.RunDrivers.ShearLayerDriver

The `shear_layer` run driver (EX-d, #441): the Kelvin–Helmholtz billow of
a weakly stratified tanh shear layer, ``u = U \\tanh(z/δ)``,
``θ = θ_0 + Δθ \\tanh(z/δ)`` with minimum gradient Richardson number
``J = g Δθ δ/(θ_0 U^2)`` at the centre, in 2-D (`Ny = 1`) between
free-slip walls, seeded at the wavenumber α with
``w' = a U \\sin(αx) e^{-(z/δ)^2}``; the base state is free to diffuse,
and the billow runs into its nonlinear stage (Era VI, P59-c, #180;
formerly `examples/transition/kelvin_helmholtz.jl`). Diagnostics, all
from `Aeolus.Transition`, every ``≈ 1\\,δ/U``: the amplitude ``A(t)`` =
rms w and its linear growth rate (`growth_rate`), the turbulent fraction
γ of the layer ``|z| <`` `layer` δ (`turbulent_fraction`), and the flux
Richardson number ``Ri_f = −B/P`` of the layer (`flux_richardson`),
instantaneous and cumulative from `rif_from`.

`results.toml`: the series, the growth-rate fit, the saturation and the
cumulative ``Ri_f``; `output/vorticity.nc`: the spanwise vorticity
``ω_y(x, z)`` at the `snapshots` times. Config:
`examples/transition/configs/kelvin_helmholtz.toml`.
"""
module ShearLayerDriver

using ...Grids: RegularGrid
using ...Fields: set!, interior_view
using ...BoundaryConditions: FieldBoundaryConditions, fill_halos!
using ...Simulations: BoussinesqSimulation, step_rk3!
import ...Simulations
using ...Operators: _vorticity_at
using ...Transition: growth_rate, turbulent_fraction, flux_richardson
using ...NCIO: nc_create, nc_defvar!
using ...IO: RunFolder, CheckpointSchedule, output_dir
using ...Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, section, option!,
                 register_driver!, write_results, no_checkpoints

export ShearLayerDriverType, ShearLayerRun

"""
    ShearLayerDriverType()

The `shear_layer` driver: `[run] driver = "shear_layer"`. Sections
`[layer]`, `[grid]`, `[stepper]` and `[diagnostics]`; `aeolus options
shear_layer` prints them.
"""
struct ShearLayerDriverType <: AbstractRunDriver end

Config.driver_name(::ShearLayerDriverType) = "shear_layer"
Config.config_dir(::ShearLayerDriverType) = normpath(joinpath(@__DIR__, "..", "..", "examples", "transition", "configs"))

function Config.driver_sections(::ShearLayerDriverType)
    lay = section("layer"; doc = "The shear layer and its seed.") do s
        option!(s, "delta", Float64; default = 1.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "half-thickness δ of the tanh profiles")
        option!(s, "velocity", Float64; default = 1.0, range = (0, Inf), open_lower = true, units = "m s^-1",
                doc = "velocity scale U (u = ±U far from the layer)")
        option!(s, "re", Float64; default = 1000.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "Reynolds number U δ/ν")
        option!(s, "richardson", Float64; default = 0.05, range = (0, Inf), units = "1",
                doc = "J, the gradient Richardson number at the centre (Δθ = J U² θ₀/(g δ))")
        option!(s, "theta0", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "reference θ₀")
        option!(s, "g", Float64; default = 9.81, range = (0, Inf), open_lower = true, units = "m s^-2",
                doc = "gravitational acceleration")
        option!(s, "alpha", Float64; default = 0.4446, range = (0, Inf), open_lower = true, units = "1",
                doc = "seed wavenumber αδ (0.4446: the most unstable of the inviscid tanh layer); L_x = 2π δ/α")
        option!(s, "seed_amplitude", Float64; default = 1e-4, range = (0, Inf), units = "1",
                doc = "a in w' = a U sin(α x/δ) exp(−(z/δ)²)")
    end
    grid = section("grid"; doc = "A 2-D (x, z) box, square cells, L_x = 2π δ/α, centred on the layer.") do s
        option!(s, "nx", Int; default = 128, range = (4, 65536), doc = "cells in x")
        option!(s, "nz", Int; default = 256, range = (4, 65536), doc = "cells in z (L_z = nz Δx)")
        option!(s, "halo", Int; default = 2, range = (1, 8), doc = "halo cells")
    end
    stp = section("stepper"; doc = "SSP-RK3 at a fixed step.") do s
        option!(s, "cfl", Float64; units = "1", default = 0.6, range = (0, Inf), open_lower = true,
                doc = "dt = cfl Δx/U, shortened to end_time / ceil(end_time / dt)")
        option!(s, "end_time", Float64; default = 160.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "simulated time (δ/U at δ = U = 1)")
    end
    dg = section("diagnostics"; doc = "The diagnostics of the layer.") do s
        option!(s, "every", Float64; default = 1.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "a record every max(1, round(every/dt)) steps (and at the last)")
        option!(s, "layer", Float64; default = 4.0, range = (0, Inf), open_lower = true, units = "1",
                doc = "the layer is |z| < layer × δ")
        option!(s, "rif_from", Float64; default = 10.0, range = (0, Inf), units = "s",
                doc = "the cumulative Ri_f = −∫B dt / ∫P dt is accumulated from this time (after the seed's adjustment)")
        option!(s, "snapshots", Vector{Float64}; default = [60.0, 90.0, 140.0], range = (0, Inf), units = "s",
                doc = "ω_y is written to output/vorticity.nc at the first record at or after each")
    end
    return Section[lay, grid, stp, dg]
end

Config.driver_checks(::ShearLayerDriverType) = Any[
    "`[checkpoint]` stays off (the run is one call)" => no_checkpoints]

"""
    ShearLayerRun

What the `shear_layer` driver builds: the simulation `sim`, the cell
heights `zc`, the layer's levels `layer`, the step `dt` and count
`nsteps`, the resolved `cfg`, and the records filled by the run.
"""
mutable struct ShearLayerRun
    sim::Any
    zc::Vector{Float64}
    layer::Vector{Int}
    dt::Float64
    nsteps::Int
    cfg::RunConfig
    records::Any
end

function Config.build(::ShearLayerDriverType, cfg::RunConfig)
    l = cfg.layer; g = cfg.grid
    δ, U, Re, J, αδ = l.delta, l.velocity, l.re, l.richardson, l.alpha
    Nx, Nz = g.nx, g.nz
    Lx = 2π * δ / αδ; dx = Lx / Nx; Lz = Nz * dx
    mesh = RegularGrid(; size = (Nx, 1, Nz), extent = (Lx, dx, Lz),
                       origin = (0.0, 0.0, -Lz / 2), halo = (g.halo, g.halo, g.halo))
    θ₀, gg = l.theta0, l.g
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(mesh; ν = U * δ / Re, θ₀ = θ₀, g_accel = gg, bcs = bcs)
    Δθ = J * U^2 * θ₀ / (gg * δ)
    a = l.seed_amplitude
    set!(sim.u, (x, y, z) -> U * tanh(z / δ))
    set!(sim.θ, (x, y, z) -> θ₀ + Δθ * tanh(z / δ))
    set!(sim.w, (x, y, z) -> a * U * sin(αδ * x / δ) * exp(-(z / δ)^2))
    zc = [mesh.z0 + (k - 0.5) * mesh.dz for k in 1:Nz]
    layer = findall(z -> abs(z) < cfg.diagnostics.layer * δ, zc)
    tend = cfg.stepper.end_time
    dt = cfg.stepper.cfl * dx / U; nsteps = ceil(Int, tend / dt); dt = tend / nsteps
    return ShearLayerRun(sim, zc, layer, dt, nsteps, cfg, nothing)
end

"The spanwise vorticity ω_y at the cell centres, as (z, x)."
function omega_y(sim)
    m = sim.mesh
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    ω = zeros(m.Nz, m.Nx)
    for k in 1:m.Nz, i in 1:m.Nx
        ω[k, i] = _vorticity_at(sim.u.data, sim.v.data, sim.w.data, m.Hx + i, m.Hy + 1,
                                m.Hz + k, 1 / m.dx, 1 / m.dy, 1 / m.dz)[2]
    end
    return ω
end

function Simulations.run!(::ShearLayerDriverType, r::ShearLayerRun, rf::RunFolder, ::CheckpointSchedule;
                          callback = nothing)
    sim, dt, nsteps, layer = r.sim, r.dt, r.nsteps, r.layer
    dg = r.cfg.diagnostics
    every = max(1, round(Int, dg.every / dt))
    ts, As, γm, Rif, Bs, Ps = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    snaps_t = Tuple(dg.snapshots); snaps = Dict{Float64,Matrix{Float64}}(); snap_at = Dict{Float64,Float64}()
    t0 = time()
    for n in 1:nsteps
        step_rk3!(sim, dt)
        if n % every == 0 || n == nsteps
            wi = interior_view(sim.w)
            push!(ts, sim.t); push!(As, sqrt(sum(abs2, wi) / length(wi)))
            γ = turbulent_fraction(sim)
            push!(γm, sum(γ[layer]) / length(layer))
            fr = flux_richardson(sim)
            B = sum(fr.B[layer]); P = sum(fr.P[layer])
            push!(Bs, B); push!(Ps, P); push!(Rif, P == 0 ? NaN : -B / P)
            for ts_ in snaps_t
                if !haskey(snaps, ts_) && sim.t >= ts_
                    snaps[ts_] = omega_y(sim); snap_at[ts_] = sim.t
                end
            end
        end
        callback === nothing || callback(r, n)
    end
    wall = time() - t0
    # the fit needs a long enough exponential stage (a short run has none: NaN)
    fit = try
        growth_rate(ts, As)
    catch err
        (err isa ErrorException && startswith(err.msg, "growth_rate: no window")) || rethrow()
        @warn "shear layer: no growth-rate window" exception = err
        (rate = NaN, r2 = NaN, window = (NaN, NaN))
    end
    ipk = argmax(As)
    i10 = findfirst(>=(dg.rif_from), ts)
    Rif_run = -cumsum(Bs[i10:end]) ./ cumsum(Ps[i10:end])
    r.records = (; ts, As, γm, Rif, Bs, Ps, Rif_run, fit, snaps)
    @info "shear layer" steps = nsteps wall_s = round(wall; digits = 1) growth_rate = fit.rate r2 = fit.r2 A_max = As[ipk] t_max = ts[ipk]
    write_results(rf, Dict{String,Any}(
        "steps" => nsteps, "dt" => dt, "wall_seconds" => wall,
        "growth_rate" => fit.rate, "growth_rate_r2" => fit.r2, "growth_window" => collect(fit.window),
        "A_max" => As[ipk], "t_saturation" => ts[ipk],
        "rif_cumulative_saturation" => Rif_run[ipk-i10+1], "rif_cumulative_end" => Rif_run[end],
        "series" => Dict{String,Any}("t" => ts, "A" => As, "gamma" => γm, "rif" => Rif, "B" => Bs, "P" => Ps),
        "rif_cumulative" => Dict{String,Any}("t" => ts[i10:end], "value" => Rif_run)))
    # the vorticity snapshots
    m = sim.mesh
    ks = sort!(collect(keys(snaps)))
    nc = nc_create(joinpath(output_dir(rf), "vorticity.nc");
                   attributes = Dict("note" => "spanwise vorticity ω_y at the cell centres, (x, z, snapshot)"))
    try
        nc.dim["x"] = m.Nx; nc.dim["z"] = m.Nz; nc.dim["snapshot"] = length(ks)
        nc_defvar!(nc, "x", Float64, ("x",))[:] = [m.x0 + (i - 0.5) * m.dx for i in 1:m.Nx]
        nc_defvar!(nc, "z", Float64, ("z",))[:] = r.zc
        nc_defvar!(nc, "t", Float64, ("snapshot",))[:] = [snap_at[k] for k in ks]
        v = nc_defvar!(nc, "omega_y", Float64, ("x", "z", "snapshot"))
        for (s, k) in enumerate(ks)
            v[:, :, s] = permutedims(snaps[k])
        end
    finally
        close(nc)
    end
    return r
end

"""
    state_fields(r::ShearLayerRun) -> NamedTuple

u, w and θ with their halos.
"""
Config.state_fields(r::ShearLayerRun) = (u = r.sim.u.data, w = r.sim.w.data, θ = r.sim.θ.data)

register_driver!(ShearLayerDriverType())

end # module
