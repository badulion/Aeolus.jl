# The `patchy_surface` run driver (CFG-e, #360): the convective boundary
# layer over strips of hot and cool ground of Phase 53 (#43), built with
# `patchy_les` and run with `patchy_run!` (examples/validation/patchy_surface/
# driver.jl): every surface column computes its own Monin–Obukhov fluxes
# through the SurfaceExchange contract, resolved, tiled or aggregated.
#
# A driver of its own (the issue names it): the surface is a matrix of
# AbstractSurfaces with per-column fluxes and conservation book-keeping,
# and the run samples tile-model comparisons and phase averages on the
# fly. Loaded on first use (Aeolus.Config.DRIVER_FILES). Config:
# examples/validation/configs/patchy_surface.toml.

using Aeolus
using Aeolus.Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section,
                     section, option!, register_driver!, write_results, no_checkpoints
using Aeolus.IO: RunFolder, CheckpointSchedule
import Aeolus.Simulations

include(joinpath(@__DIR__, "patchy_surface", "driver.jl"))

struct PatchySurfaceDriver <: AbstractRunDriver end

Config.driver_name(::PatchySurfaceDriver) = "patchy_surface"
Config.config_dir(::PatchySurfaceDriver) = joinpath(@__DIR__, "configs")

function Config.driver_sections(::PatchySurfaceDriver)
    g = section("grid"; doc = "The box: periodic in x and y, a lid with a sponge in the top quarter.") do s
        option!(s, "size", NTuple{3,Int}; default = (64, 32, 32), range = (1, 4096), doc = "cells in x, y, z")
        option!(s, "extent", NTuple{3,Float64}; default = (3200.0, 1600.0, 1600.0), range = (0, Inf),
                open_lower = true, units = "m", doc = "Lx, Ly, Lz")
    end
    sf = section("surface"; doc = "The strips (normal to x) and how the LES sees them.") do s
        option!(s, "wavelength", Float64; default = 800.0, range = (0, Inf), open_lower = true, units = "m",
                doc = "strip wavelength λ (one hot and one cool strip)")
        option!(s, "hot_fraction", Float64; default = 0.5, range = (0, 1), doc = "hot share of a wavelength")
        option!(s, "t_hot", Float64; default = 310.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "skin temperature of the hot strips")
        option!(s, "t_cool", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "skin temperature of the cool strips")
        option!(s, "mode", Symbol; default = :resolved, allowed = [:resolved, :tiled, :parameter],
                doc = "resolved: each column its own strip; tiled: a SurfaceMosaic of both in every column; " *
                      "parameter: the parameter aggregate (mean skin temperature) in every column")
        option!(s, "z0m", Float64; default = 0.1, range = (0, Inf), open_lower = true, units = "m",
                doc = "momentum roughness length (z0h = z0m / 10)")
        option!(s, "p_sfc", Float64; default = 1e5, range = (0, Inf), open_lower = true, units = "Pa",
                doc = "surface pressure")
    end
    ini = section("initial"; doc = "Mixed layer, inversion, wind (white noise of 0.2 K below 100 m, MersenneTwister([run] seed)).") do s
        option!(s, "theta0", Float64; default = 300.0, range = (0, Inf), open_lower = true, units = "K",
                doc = "mixed-layer potential temperature (and the Boussinesq reference)")
        option!(s, "z_inv", Float64; default = 600.0, range = (0, Inf), units = "m", doc = "inversion height")
        option!(s, "lapse", Float64; default = 0.006, range = (0, Inf), units = "K m^-1", doc = "lapse rate above z_inv")
        option!(s, "u_g", Float64; default = 0.0, units = "m s^-1",
                doc = "initial (and, with f_coriolis > 0, geostrophic) wind along x")
    end
    dyn = section("dynamics"; doc = "Rotation and the closure.") do s
        option!(s, "f_coriolis", Float64; default = 0.0, units = "s^-1", doc = "Coriolis parameter")
        option!(s, "c_s", Float64; default = 0.17, range = (0, Inf), open_lower = true,
                doc = "Smagorinsky coefficient")
    end
    st = section("stepper"; doc = "SSP-RK3 with a fixed step; statistics sampled on the fly.") do s
        option!(s, "end_time", Float64; default = 7200.0, range = (0, Inf), open_lower = true, units = "s",
                doc = "simulated time")
        option!(s, "dt", Float64; default = 3.0, range = (0, Inf), open_lower = true, units = "s", doc = "the step")
        option!(s, "average_from", Float64; default = 3600.0, range = (0, Inf), units = "s",
                doc = "statistics from this time on")
        option!(s, "sample_every", Int; default = 5, range = (1, typemax(Int)), doc = "steps between samples")
        option!(s, "z_mid", Float64; default = 400.0, range = (0, Inf), units = "m",
                doc = "height of the phase-averaged vertical velocity")
    end
    return Section[g, sf, ini, dyn, st]
end

function _checks(cfg)
    e = no_checkpoints(cfg)
    st = cfg.stepper
    st.average_from <= st.end_time ||
        push!(e, ConfigError("stepper.average_from", "no samples: average_from is after end_time"))
    return e
end
Config.driver_checks(::PatchySurfaceDriver) = Any[_checks]

"What the `patchy_surface` driver builds: `les` (a `PatchyLES`), `result` (of `patchy_run!`) and `cfg`."
mutable struct PatchyRun
    les::Any
    result::Any
    cfg::RunConfig
end

function Config.build(::PatchySurfaceDriver, cfg::RunConfig)
    g = cfg.grid; s = cfg.surface; i = cfg.initial; d = cfg.dynamics
    les = patchy_les(; Nx = g.size[1], Ny = g.size[2], Nz = g.size[3],
                     Lx = g.extent[1], Ly = g.extent[2], Lz = g.extent[3],
                     λ = s.wavelength, f_hot = s.hot_fraction, T_hot = s.t_hot, T_cool = s.t_cool,
                     mode = s.mode, θ0 = i.theta0, z_inv = i.z_inv, Γ = i.lapse, Ug = i.u_g,
                     f_cor = d.f_coriolis, z0m = s.z0m, C_s = d.c_s, seed = cfg.run.seed, p_sfc = s.p_sfc)
    return PatchyRun(les, nothing, cfg)
end

function Simulations.run!(::PatchySurfaceDriver, r::PatchyRun, rf::RunFolder, ::CheckpointSchedule;
                          callback = nothing)
    st = r.cfg.stepper
    wall = @elapsed r.result = patchy_run!(r.les; t_end = st.end_time, dt = st.dt, t_avg = st.average_from,
                                           every = st.sample_every, z_mid = st.z_mid)
    x = r.result
    @info("patchy surface statistics", subsystem = :run, les_flux = x.les, zi = x.zi, wstar = x.wstar,
          heat_error = x.heat_error, nsamples = x.nsamples)
    keys_ = (:les, :tile_vec, :tile_speed, :tile_gust, :tile_patch, :param, :blended, :ustar2, :wstar, :zi,
             :U_vec, :U_speed, :heat_error, :nsamples, :z_mid)
    write_results(rf, merge(Dict(String(k) => getfield(x, k) for k in keys_),
                            Dict("u_p" => x.u_p, "wtheta_p" => x.wθ_p, "w_p" => x.w_p, "x_p" => collect(x.x_p),
                                 "theta_profile" => x.θ_profile, "z" => collect(x.z), "wall_time" => wall)))
    callback === nothing || callback(r, round(Int, st.end_time / st.dt))
    return r
end

function Config.state_fields(r::PatchyRun)
    s = r.les.sim
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data, θ = s.θ.data)
end

register_driver!(PatchySurfaceDriver())
