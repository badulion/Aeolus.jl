# =========================================================================
# The `dns` run driver (CFG-d, #359): the decaying-turbulence DNS of the
# Era VI filtered-DNS fixture (test/fixtures/dns_decay.jl, P56-c #178) and
# its a-priori record (test/fixtures/dns_apriori.jl), configured from TOML
# (`[run] driver = "dns"`).
#
#   julia -m Aeolus run examples/validation/configs/dns256.toml
#   julia -m Aeolus run examples/validation/configs/dns256.toml --set grid.n=64
#
# The fixture's functions stay the implementation: the run is dns_start,
# then dns_step! while dns_running, then dns_summary; the record is
# dns_apriori_record. The resolution refusal of the offline generator
# (k_max η ≥ 1.5 over the run, with the other required DNS checks) is a
# check after the run: a failed run ends `failed` with the measured value
# in the error, and its record goes to <out>.failed only.
#
# Loaded by Aeolus.Config on first use of the driver name (DRIVER_FILES),
# or included directly (tests): either way it calls register_driver!(DNSDriver()).
# =========================================================================

include(joinpath(@__DIR__, "..", "..", "test", "fixtures", "dns_apriori.jl"))

using Aeolus.Config
using Aeolus.Config: Config, AbstractRunDriver, RunConfig, ConfigError, Section, OpenTable, OneOf,
                     Optional, Path, section, option!, register_driver!
using Aeolus.IO: IO, RunFolder, CheckpointSchedule, with_monitor, monitor!, output_dir
using Aeolus.Simulations: Simulations

"""
    DNSDriver()

The `dns` run driver: `[run] driver = "dns"`. Shipped config:
`examples/validation/configs/dns256.toml`.
"""
struct DNSDriver <: AbstractRunDriver end

const DNS_CONFIG_DIR = joinpath(@__DIR__, "configs")

Config.driver_name(::DNSDriver) = "dns"
Config.config_dir(::DNSDriver) = DNS_CONFIG_DIR

function _dns_spectrum_problems(t)
    get(t, "kind", nothing) == "k4_gaussian" ||
        return ["kind: expected \"k4_gaussian\" (E(k) ∝ k⁴ exp(-2 (k/k_p)²)), got $(repr(get(t, "kind", nothing)))"]
    p = ["$k: unknown key (keys: kind, k_p)" for k in keys(t) if !(k in ("kind", "k_p"))]
    kp = get(t, "k_p", nothing)
    (kp isa Real && !(kp isa Bool) && kp > 0) || push!(p, "k_p: expected the peak wavenumber > 0")
    return p
end

function Config.driver_sections(::DNSDriver)
    grid = section("grid"; doc = "The triply periodic 2π box.") do s
        option!(s, "n", Int; default = 256, range = (4, 4096), doc = "cells per direction (N³)")
    end
    fluid = section("fluid"; doc = "The fluid.") do s
        option!(s, "nu", OneOf(Float64, Symbol); default = :scaled, allowed = [:scaled], range = (0, Inf),
                open_lower = true, units = "m^2 s^-1",
                doc = "kinematic viscosity; \"scaled\": nu_margin × 0.018 (48/N)^{4/3}, the fixture's scaling that holds k_max η fixed")
        option!(s, "nu_margin", Float64; units = "1", default = 1.035, range = (0, Inf), open_lower = true,
                doc = "the factor of \"scaled\" over the fixture's ν (1.035: the offline generator's 3.5 % margin, k_max η × 1.026; 1: the CI fixture)")
    end
    initial = section("initial"; doc = "The initial field (isotropic_random_field!, u' = 1).") do s
        option!(s, "spectrum", OpenTable(_dns_spectrum_problems);
                default = Dict{String,Any}("kind" => "k4_gaussian", "k_p" => 1.5),
                doc = "{kind = \"k4_gaussian\", k_p}: E(k) ∝ k⁴ exp(-2 (k/k_p)²); the seed is [run] seed")
    end
    stepper = section("stepper"; doc = "Projected SSP-RK3, no SGS model.") do s
        option!(s, "cfl", Float64; units = "1", default = 0.5, range = (0, Inf), open_lower = true,
                doc = "advective Courant number")
        option!(s, "end_time", Float64; units = "L/u'", default = 2.0, range = (0, Inf), open_lower = true,
                doc = "the run goes at least to this time")
        option!(s, "past_peak", Float64; units = "L/u'", default = 0.25, range = (0, Inf),
                doc = "then on until this far past the maximum of ε (0: stop at end_time)")
        option!(s, "max_time", Float64; units = "L/u'", default = 6.0, range = (0, Inf), open_lower = true,
                doc = "the cap of the run past the peak")
        option!(s, "nsteps", Optional{Int}; range = (1, typemax(Int)),
                doc = "stop after this many steps (\"none\": no cap)")
    end
    apriori = section("apriori"; doc = "The filtered-DNS a-priori record (dns_apriori_record).") do s
        option!(s, "widths", Vector{Int}; default = [4, 8, 16, 32], range = (1, 4096),
                doc = "Gaussian filter widths Δ in grid spacings (each also the LES-grid coarsening); [] : no record")
        option!(s, "out", Optional{Path}; input = false,
                doc = "the record's TOML file; \"none\": dns_apriori.toml in the run folder's output/")
        option!(s, "require_checks", Bool; default = true,
                doc = "refuse a run whose required DNS checks fail (k_max η ≥ 1.5 over the run, skewness band, energy budget, monotone decay, past the peak): the record goes to <out>.failed and the run fails")
    end
    return Section[grid, fluid, initial, stepper, apriori]
end

function _dns_checks(cfg)
    e = ConfigError[]
    cfg.stepper.max_time >= cfg.stepper.end_time ||
        push!(e, ConfigError("stepper.max_time", "must be ≥ stepper.end_time"))
    n = cfg.grid.n
    for w in cfg.apriori.widths
        n % w == 0 || push!(e, ConfigError("apriori.widths", "the width $w does not divide grid.n = $n"))
    end
    return e
end

Config.driver_checks(::DNSDriver) = Any[
    """`stepper.max_time` ≥ `stepper.end_time`
    every `apriori.widths` entry divides `grid.n`""" => _dns_checks]

"""
    dns_kwargs(cfg::RunConfig) -> NamedTuple

The [`dns_decay`](@ref) keywords of a `dns` config: `N`, `seed`, `ν`
(`"scaled"`: `nu_margin` × 0.018 (48/N)^{4/3}), `t_end`, `past_peak`,
`t_max`, `cfl`.
"""
function dns_kwargs(cfg::RunConfig)
    N = cfg.grid.n
    ν = cfg.fluid.nu === :scaled ? cfg.fluid.nu_margin * DNS_NU48 * (48 / N)^(4 / 3) : cfg.fluid.nu
    st = cfg.stepper
    return (; N, seed = cfg.run.seed, ν, t_end = st.end_time, past_peak = st.past_peak,
              t_max = st.max_time, cfl = st.cfl)
end

"""
    dns_config(path_or_name = "dns256"; overrides = String[]) -> RunConfig

Load a `dns` config: a path, or a shipped config of `examples/validation/configs/`.
"""
function dns_config(path::AbstractString = "dns256"; overrides = String[])
    p = isfile(path) ? path : joinpath(DNS_CONFIG_DIR, endswith(path, ".toml") ? path : path * ".toml")
    return load_run_config(p; overrides = overrides)
end

"""
    DNSRun

What the `dns` driver builds: the DNS state `st` ([`dns_start`](@ref)),
the resolved `cfg`, and after the run the `summary` ([`dns_summary`](@ref))
and the a-priori `record` (or `nothing`).
"""
mutable struct DNSRun
    st::Any
    cfg::RunConfig
    summary::Any
    record::Any
end

function Config.build(::DNSDriver, cfg::RunConfig)
    k = dns_kwargs(cfg)
    kp = Float64(cfg.initial.spectrum["k_p"])
    st = dns_start(; N = k.N, seed = k.seed, ν = k.ν, spectrum = q -> q^4 * exp(-2 * (q / kp)^2))
    return DNSRun(st, cfg, nothing, nothing)
end

function Simulations.run!(::DNSDriver, r::DNSRun, rf::RunFolder, ck::CheckpointSchedule; callback = nothing)
    cfg = r.cfg; k = dns_kwargs(cfg); st = r.st; sim = st.sim
    nmax = something(cfg.stepper.nsteps, typemax(Int))
    wall = @elapsed begin
        resume!(ck, sim; components = st.hist)
        with_monitor(sim; run = rf, t_end = k.t_end, cadence = cfg.logging.diagnostics_cadence) do m
            while dns_running(st; t_end = k.t_end, past_peak = k.past_peak, t_max = k.t_max) &&
                  length(st.hist.t) - 1 < nmax
                dt = dns_step!(st; cfl = k.cfl, t_end = k.t_end, t_max = k.t_max)
                n = length(st.hist.t) - 1
                monitor!(m, sim; step = n, dt = dt)
                maybe_checkpoint!(ck, sim; step = n, components = st.hist)
                callback === nothing || callback(r, n)
            end
        end
    end
    # keep the velocity and pressure only: the solver's work arrays are ~40
    # fields (6 GB at 256³), freed before the statistics
    slim = (; mesh = sim.mesh, u = sim.u, v = sim.v, w = sim.w, p = sim.p, t = sim.t)
    r.st = merge(st, (; sim = slim)); st = nothing; sim = nothing
    GC.gc()
    d = dns_summary(r.st, wall)
    r.summary = d
    @info(@sprintf("DNS %d³: ν = %.4g, %d steps to t = %.2f, %.0f s; Re_λ %.1f; k_max η ≥ %.2f; ε peak at t = %.2f; S = %.3f; max |budget err| %.1e",
                   k.N, d.ν, d.nsteps, d.t[end], d.wall, d.Re_λ, d.kmax_eta, d.t[argmax(d.ε)], d.S[end],
                   isempty(d.budget_err) ? 0.0 : maximum(abs, d.budget_err)),
          subsystem = :run, N = k.N, Re_λ = d.Re_λ, nsteps = d.nsteps, kmax_eta = d.kmax_eta)
    out = something(cfg.apriori.out, joinpath(output_dir(rf), "dns_apriori.toml"))
    if isempty(cfg.apriori.widths)
        (cfg.apriori.require_checks && d.kmax_eta < 1.5) &&
            error("DNS checks failed at $(k.N)³: kmax_eta_ge_1.5; measured k_max η = " *
                  "$(round(d.kmax_eta; sigdigits = 4)) (minimum over the run; the 2nd-order solver needs ≥ 1.5)")
        return r
    end
    r.record = dns_apriori_finish(d; out, ratios = Tuple(cfg.apriori.widths), seed = k.seed,
                                  require_checks = cfg.apriori.require_checks)
    return r
end

"""
    state_fields(r::DNSRun) -> NamedTuple

The DNS velocity and pressure (with halos): `u, v, w, p`.
"""
function Config.state_fields(r::DNSRun)
    s = r.st.sim
    return (u = s.u.data, v = s.v.data, w = s.w.data, p = s.p.data)
end

register_driver!(DNSDriver())
