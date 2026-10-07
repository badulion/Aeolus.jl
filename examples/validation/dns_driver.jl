# The `dns_decay` run driver (CFG-e, #360): the decaying isotropic DNS of
# Phase 56 (P56-c, #178), built and run by the fixture the CI test shares
# (test/fixtures/dns_decay.jl, `dns_decay`): a triply periodic box of side
# 2π started from a von Kármán-Pao spectrum scaled to K = 3/2, SSP-RK3 at
# a fixed advective CFL, with the energy budget and the derivative
# skewness recorded every step.
#
# A driver of its own: the fixture's step is cfl·dx/max|u| with the
# budget and skewness bookkeeping of every step; the a-priori analysis
# (examples/validation/apriori_sgs.jl) reads the final fields. Loaded on
# first use (Aeolus.Config.DRIVER_FILES). Config:
# examples/validation/configs/apriori_sgs.toml.

using Aeolus
using Aeolus.Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, Optional,
                     section, option!, register_driver!, write_results
using Aeolus.IO: RunFolder, CheckpointSchedule
import Aeolus.Simulations

include(joinpath(@__DIR__, "..", "..", "test", "fixtures", "dns_decay.jl"))

struct DNSDecayDriver <: AbstractRunDriver end

Config.driver_name(::DNSDecayDriver) = "dns_decay"
Config.config_dir(::DNSDecayDriver) = joinpath(@__DIR__, "configs")

function Config.driver_sections(::DNSDecayDriver)
    g = section("grid"; doc = "The periodic box of side 2π.") do s
        option!(s, "n", Int; default = 48, range = (8, 4096), doc = "cells per side (N³)")
    end
    d = section("dns"; doc = "The decaying DNS (dns_decay of test/fixtures/dns_decay.jl).") do s
        option!(s, "nu", Optional{Float64}; range = (0, Inf), open_lower = true, units = "m^2 s^-1",
                doc = "kinematic viscosity; \"none\": 0.018 (48/N)^(4/3), which holds k_max η ≈ 1.5")
        option!(s, "end_time", Float64; default = DNS_T_END, range = (0, Inf), open_lower = true,
                doc = "simulated time (in units of L/u', u' = 1)")
        option!(s, "past_peak", Float64; default = 0.0, range = (0, Inf),
                doc = "run on this long past the dissipation peak (0: stop at end_time)")
        option!(s, "cfl", Float64; default = 0.5, range = (0, Inf), open_lower = true,
                doc = "advective Courant number of the step")
    end
    return Section[g, d]
end

"What the `dns_decay` driver builds and runs: `result` (the NamedTuple of `dns_decay`) and `cfg`."
mutable struct DNSDecayRun
    result::Any
    cfg::RunConfig
end

Config.build(::DNSDecayDriver, cfg::RunConfig) = DNSDecayRun(nothing, cfg)

function Simulations.run!(::DNSDecayDriver, r::DNSDecayRun, rf::RunFolder, ck::CheckpointSchedule;
                          callback = nothing)
    c = r.cfg; d = c.dns
    kw = d.nu === nothing ? (;) : (; ν = d.nu)
    r.result = dns_decay(; N = c.grid.n, seed = c.run.seed, t_end = d.end_time, cfl = d.cfl,
                         past_peak = d.past_peak, checkpoint = ck, kw...)
    x = r.result
    @info("decaying DNS", subsystem = :run, nsteps = x.nsteps, Re_lambda = x.Re_λ, kmax_eta = x.kmax_eta,
          skewness = x.S[end], budget_error = maximum(abs, x.budget_err; init = 0.0))
    write_results(rf, Dict("nu" => x.ν, "nsteps" => x.nsteps, "Re_lambda" => x.Re_λ, "kmax_eta" => x.kmax_eta,
                           "kmax_eta_end" => x.kmax_eta_end, "u_rms" => x.u_rms, "eta" => x.η,
                           "t" => x.t, "K" => x.K, "epsilon" => x.ε, "skewness" => x.S,
                           "budget_error" => x.budget_err, "wall_time" => x.wall))
    callback === nothing || callback(r, x.nsteps)
    return r
end

function Config.state_fields(r::DNSDecayRun)
    s = r.result.sim
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data)
end

register_driver!(DNSDecayDriver())
