# The `dns` run driver (CFG-d, #359): the decaying-turbulence DNS of the
# filtered-DNS fixture and its a-priori record configured from TOML
# (examples/validation/dns_driver.jl, examples/validation/configs/dns256.toml).
#
# Acceptance tests of the issue covered here:
#   * the driver runs the fixture's own steps: its fields are bitwise those
#     of `dns_decay` with the same arguments (`nu = "scaled"` is the
#     generator's 1.035 · 0.018 (48/N)^{4/3});
#   * round trip from run.toml, bit-identical under ESTIMATE FFTW at 32³
#     (20 steps), a-priori record included;
#   * a run at grid.n = 32 with nu = 1e-4 fails the k_max η check, with the
#     measured value in the error, and its record goes to <out>.failed only.
# The full 32³ pipeline (DNS past the peak → statistics → TOML → read back)
# is test/era6_apriori_sgs.jl's, now through this driver.

using Test
using Aeolus
using Aeolus.Config
using Aeolus.IO: read_run_toml
import TOML

include(joinpath(@__DIR__, "..", "examples", "validation", "dns_driver.jl"))
include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const DNSCFG = normpath(joinpath(@__DIR__, "..", "examples", "validation", "configs", "dns256.toml"))
dns_run(cfg; kw...) = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir(), kw...)

@testset "dns driver: the shipped config" begin
    cfg = load_run_config(DNSCFG)
    @test cfg.run.driver == "dns" && cfg.grid.n == 256 && cfg.fluid.nu === :scaled
    @test DNSCFG in shipped_configs()
    # "scaled" is the offline generator's ν: 1.035 · 0.018 (48/256)^{4/3} = 2.00e-3
    k = dns_kwargs(cfg)
    @test k.ν == 1.035 * 0.018 * (48 / 256)^(4 / 3) && round(k.ν; sigdigits = 3) == 2.00e-3
    @test dns_kwargs(with(cfg, "fluid.nu_margin" => 1.0, "grid.n" => 96)).ν == DNS_NU48 * (48 / 96)^(4 / 3)
    @test dns_kwargs(with(cfg, "fluid.nu" => 0.01)).ν == 0.01
    # the canonical form reloads; the a-priori output path is not an input
    @test load_run_config(write_config(joinpath(mktempdir(), "c.toml"), cfg)) == cfg
    @test isempty(config_inputs(cfg))
    # cross-field: every filter width divides N; nu is a number or "scaled"
    e = try load_run_config(DNSCFG; overrides = ["grid.n=48", "apriori.widths=[4,32]"]); nothing
        catch err; err; end
    @test e isa ConfigError && occursin("the width 32 does not divide grid.n = 48", sprint(showerror, e))
    e = try load_run_config(DNSCFG; overrides = ["fluid.nu=\"auto\""]); nothing catch err; err; end
    @test e isa ConfigError && occursin("fluid.nu", sprint(showerror, e)) && occursin("\"scaled\"", sprint(showerror, e))
end

@testset "dns driver: the fixture's steps, bit for bit" begin
    cfg = load_run_config(DNSCFG; overrides = ["grid.n=32", "stepper.end_time=0.3", "stepper.past_peak=0.0",
                                               "apriori.widths=[]", "fftw.planner=\"estimate\""])
    res = dns_run(cfg)
    d = dns_decay(; N = 32, seed = 1, ν = 1.035 * DNS_NU48 * (48 / 32)^(4 / 3), t_end = 0.3, past_peak = 0)
    s = res.obj.st.sim
    @test s.t == d.sim.t == 0.3
    @test all(bitwise_equal(getproperty(s, f).data, getproperty(d.sim, f).data) for f in (:u, :v, :w, :p))
    @test res.obj.summary.K == d.K && res.obj.summary.ε == d.ε && res.obj.summary.nsteps == d.nsteps
    # decaying turbulence: K falls every step, at the rate ε (budget to 2 %)
    @test all(diff(d.K) .< 0) && maximum(abs, d.budget_err) < 0.02
    @test read_run_toml(res.folder.path)["session"][end]["status"] == "completed"
end

@testset "dns driver: round trip from run.toml is bit-identical (32³, 20 steps)" begin
    r = roundtrip_check(DNSCFG; nsteps = 20, overrides = ["grid.n=32", "apriori.widths=[4,8]",
                                                          "apriori.require_checks=false"],
                        nsteps_overrides = n -> ["stepper.nsteps=$n"])
    @test r.fields_equal && isempty(r.fields_differing)
    @test r.diagnostics_equal && r.config_equal
    r1, r2 = r.runs
    @test r1.obj.summary.nsteps == r2.obj.summary.nsteps == 20
    # the a-priori record of both runs: the same statistics
    rec1, rec2 = r1.obj.record, r2.obj.record
    @test isequal(rec1["filter"], rec2["filter"]) && isequal(rec1["spectrum"], rec2["spectrum"])   # NaN-aware
    @test isfile(joinpath(r1.folder.path, "output", "dns_apriori.toml"))
    # 20 steps do not reach t = 2 or the dissipation peak: recorded, not refused
    @test rec1["checks"]["past_peak"] == false
end

@testset "dns driver: the k_max η refusal names the measured value" begin
    out = joinpath(mktempdir(), "dns32.toml")
    cfg = load_run_config(DNSCFG; overrides = ["grid.n=32", "fluid.nu=1e-4", "stepper.nsteps=3",
                                               "apriori.widths=[4]", "apriori.out=\"$out\"",
                                               "fftw.planner=\"estimate\""])
    root = mktempdir()
    err = try run_config(cfg; root = root, console = false, repo = mktempdir()); nothing catch e; e; end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("kmax_eta_ge_1.5", msg) && occursin("measured k_max η = ", msg)
    # the measured value: k_max η = (π/Δx) (ν³/ε)^{1/4} at its minimum, far below 1.5 at ν = 1e-4
    m = match(r"measured k_max η = ([0-9.e+-]+)", msg)
    @test m !== nothing && parse(Float64, m[1]) < 0.2
    @test !isfile(out) && isfile(out * ".failed")
    @test TOML.parsefile(out * ".failed")["checks"]["kmax_eta_ge_1.5"] == false
    runtoml = only(joinpath(r, f) for (r, _, fs) in walkdir(root) for f in fs if f == "run.toml")
    @test TOML.parsefile(runtoml)["session"][end]["status"] == "failed"
    # with no record requested the same refusal holds
    err = try dns_run(with(cfg, "apriori.widths" => Int[])); nothing catch e; e; end
    @test err isa ErrorException && occursin("measured k_max η", sprint(showerror, err))
end
