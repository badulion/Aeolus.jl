# The `cbc_decay` run driver (CFG-e, #360): the decaying grid turbulence of
# Comte-Bellot & Corrsin (1971), Phase 54 (#168), built and run by the
# harness the CI test shares (test/cbc_benchmark_driver.jl,
# `cbc_benchmark`): a periodic box of side 11 M started from the measured
# spectrum at tU0/M = 42 (evolve-and-rescale spin-up), averaged over seeds,
# compared with the spectra at 98 and 171.
#
# A driver of its own: the harness runs several realisations with spin-up
# cycles and spectral rescaling, a structure the simulation driver has not.
# Loaded on first use (Aeolus.Config.DRIVER_FILES). Config:
# examples/validation/configs/cbc_decay.toml.

using Aeolus
using Aeolus.Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, Optional,
                     section, option!, register_driver!, write_results, no_checkpoints
using Aeolus.IO: RunFolder, CheckpointSchedule
import Aeolus.Simulations

include(joinpath(@__DIR__, "..", "..", "test", "cbc_benchmark_driver.jl"))

struct CBCDriver <: AbstractRunDriver end

Config.driver_name(::CBCDriver) = "cbc_decay"
Config.config_dir(::CBCDriver) = joinpath(@__DIR__, "configs")

const _CBC_CLOSURES = [:none, :smagorinsky, :dynamic, :lagrangian, :lasd, :tke, :amd, :vreman, :sigma]

function Config.driver_sections(::CBCDriver)
    g = section("grid"; doc = "The periodic box of side 11 M (M = 5.08 cm).") do s
        option!(s, "n", Int; default = 32, range = (8, 1024), doc = "cells per side (N³)")
    end
    sgs = section("sgs"; doc = "The subgrid closure.") do s
        option!(s, "closure", Symbol; default = :dynamic, allowed = _CBC_CLOSURES,
                doc = "none (molecular ν only), smagorinsky (C_s = 0.17), dynamic (plane-averaged Germano-Lilly), " *
                      "lagrangian / lasd (Lagrangian dynamic, scale invariant / dependent), tke (Deardorff-Moeng), " *
                      "amd (C = 0.3), vreman (0.17), sigma (1.35)")
    end
    b = section("benchmark"; doc = "The realisations and the spin-up (cbc_benchmark).") do s
        option!(s, "seeds", Vector{Int}; default = [1, 2, 3], range = (0, typemax(Int)),
                doc = "one realisation per seed of the random initial field; the spectra are averaged")
        option!(s, "spinup", Int; default = 2, range = (0, 100),
                doc = "evolve-and-rescale cycles before tU0/M = 42 (Rogallo 1981)")
        option!(s, "t_spin", Optional{Float64}; range = (0, Inf), open_lower = true, units = "s",
                doc = "length of one spin-up cycle; \"none\": the 42 → 98 interval, (98 − 42) M/U0")
        option!(s, "cfl", Float64; default = 0.5, range = (0, Inf), open_lower = true,
                doc = "advective Courant number of the step")
    end
    return Section[g, sgs, b]
end

function _checks(cfg)
    e = no_checkpoints(cfg)
    isempty(cfg.benchmark.seeds) && push!(e, ConfigError("benchmark.seeds", "give at least one seed"))
    return e
end
Config.driver_checks(::CBCDriver) = Any[_checks]

"What the `cbc_decay` driver builds and runs: `result` (the NamedTuple of `cbc_benchmark`) and `cfg`."
mutable struct CBCRun
    result::Any
    cfg::RunConfig
end

Config.build(::CBCDriver, cfg::RunConfig) = CBCRun(nothing, cfg)

function Simulations.run!(::CBCDriver, r::CBCRun, rf::RunFolder, ::CheckpointSchedule; callback = nothing)
    b = r.cfg.benchmark
    kw = b.t_spin === nothing ? (;) : (; t_spin = b.t_spin)
    r.result = cbc_benchmark(r.cfg.sgs.closure; N = r.cfg.grid.n, cfl = b.cfl, seeds = b.seeds,
                             spinup = b.spinup, kw...)
    s = r.result
    @info(cbc_summary(String(r.cfg.sgs.closure), s), subsystem = :run)
    write_results(rf, Dict("K_ratio" => collect(s.K_ratio), "K_ratio_ref" => collect(s.K_ratio_ref),
                           "rms_log_err" => collect(s.rms_log_err), "E_kc_ratio" => collect(s.E_kc_ratio),
                           "k" => s.k, "E_42" => s.E_les[1], "E_98" => s.E_les[2], "E_171" => s.E_les[3],
                           "nsteps" => s.nsteps, "wall_time" => s.wall))
    callback === nothing || callback(r, s.nsteps)
    return r
end

"The averaged LES spectra at the three stations (the run's result)."
Config.state_fields(r::CBCRun) = (E_42 = r.result.E_les[1], E_98 = r.result.E_les[2], E_171 = r.result.E_les[3])

register_driver!(CBCDriver())
