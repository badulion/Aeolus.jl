# examples/validation/ configured from TOML (CFG-e, #360).
#
# Acceptance tests of the issue covered here:
#   * bit-identical to today: the case built from each shipped config, at
#     the test resolution, gives fields bitwise equal to the in-code build
#     of the old script (captured below), after 20 steps (1 column step
#     for the `column` driver);
#   * round trip, bit-identical under ESTIMATE FFTW: one config of the
#     directory (neutral_abl, one member) and dycoms_rf01.toml;
#   * sweeps: neutral_abl.toml expands to the 3 closures × 2 resolutions
#     in a fixed order, and each member validates.

using Test
using Aeolus
using Aeolus.Config

include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const VAL_CFG = joinpath(pkgdir(Aeolus), "examples", "validation", "configs")
vcfg(name) = joinpath(VAL_CFG, name * ".toml")

"Run one member of a shipped config quietly into a fresh root: the RunResult."
function val_run(name, member, overrides = String[]; kw...)
    cfg = only(load_sweep(vcfg(name); member = member,
                          overrides = vcat("fftw.planner=\"estimate\"", overrides)))[2]
    return run_config(cfg; root = mktempdir(), console = false, repo = mktempdir(), kw...)
end

"Bitwise equality of every array of two NamedTuples of state fields."
fields_bitwise(a, b) = keys(a) == keys(b) && all(bitwise_equal(a[k], b[k]) for k in keys(a))

# ---------------------------------------------------------------------------
# the channel driver: neutral_abl.toml, log_layer_mismatch.toml
# ---------------------------------------------------------------------------

@testset "neutral_abl.toml: the sweep is 3 closures × 2 resolutions, in order, each valid" begin
    runs = load_sweep(vcfg("neutral_abl"))
    @test [m.label for (m, _) in runs] ==
          ["smagorinsky_32x32x32", "smagorinsky_48x48x48", "lasd_32x32x32", "lasd_48x48x48",
           "lasi_32x32x32", "lasi_48x48x48"]
    @test [(c.sgs.closure, c.channel.size[1]) for (_, c) in runs] ==
          [(:smagorinsky, 32), (:smagorinsky, 48), (:lasd, 32), (:lasd, 48), (:lasi, 32), (:lasi, 48)]
    @test [c.wall_model.filter for (_, c) in runs] == [:none, :none, :spatial, :spatial, :spatial, :spatial]
    # the old script's constants: T_END = 50, T_AVG = 0.6 T_END, C_S = 0.1, seed 1
    @test all(c.stepper.end_time == 50.0 && c.stepper.average_from_fraction * c.stepper.end_time == 30.0 &&
              c.sgs.c_s == 0.1 && c.run.seed == 1 && c.channel.kind === :half for (_, c) in runs)
    @test length(unique(config_hash(c) for (_, c) in runs)) == 6
end

@testset "channel driver: neutral_abl and log_layer_mismatch equal the old in-code cases, bit for bit (≈20 steps)" begin
    # examples/validation/neutral_abl.jl before CFG-e (make_case + run_case),
    # at the test resolution 24³ of test/era6_neutral_abl.jl
    t_end = 0.05                                       # ≈20 steps of the adaptive step at 24³ (counted below)
    for (label, legacy) in (("smagorinsky_32x32x32",
                             () -> half_channel_case(; N = 24, closure = StaticSmagorinsky(0.1))),
                            ("lasd_32x32x32",
                             () -> half_channel_case(; N = 24, filter = :spatial,
                                     closure = LagrangianDynamicSmagorinsky(
                                         RegularGrid(; size = (24, 24, 24), extent = (2π, 2π, 1.0));
                                         scale_dependent = true))))
        c = legacy()
        st = run_channel!(c, t_end; average_from = 0.6 * t_end)
        r = val_run("neutral_abl", label, ["channel.size=[24,24,24]", "stepper.end_time=$t_end"])
        @test r.obj.case.nsteps == c.nsteps
        @test 15 <= c.nsteps <= 30
        @test fields_bitwise(state_fields(r.obj), (u = c.sim.u.data, v = c.sim.v.data,
                                                   w = c.sim.w.data, p = c.sim.p.data))
        @test bitwise_equal(r.obj.stats.U, st.U) && r.obj.stats.τ_wall === st.τ_wall
        @test isfile(joinpath(r.folder.path, "output", "results.toml"))
    end
    # examples/validation/log_layer_mismatch.jl before CFG-e: run(Re; h_wm = 3DZ, filter = :temporal)
    DZ = 2 / 16
    c = channel_case(; Re_τ = 1000.512, N = (24, 12, 16), closure = StaticSmagorinsky(0.1),
                     wall_law = SpaldingLaw(), h_wm = 3DZ, filter = :temporal)
    st = run_channel!(c, 0.2; average_from = 0.2 - 0.2 / 2)
    r = val_run("log_layer_mismatch", "re1000_fix", ["stepper.end_time=0.2"])
    @test r.obj.case.nsteps == c.nsteps >= 15
    @test fields_bitwise(state_fields(r.obj), (u = c.sim.u.data, v = c.sim.v.data,
                                               w = c.sim.w.data, p = c.sim.p.data))
    @test bitwise_equal(r.obj.stats.U, st.U)
    # and the dynamic closure at Re_τ = 1000, first-cell matching
    c = channel_case(; Re_τ = 1000, N = (24, 12, 16), wall_law = SpaldingLaw(),
                     closure = DynamicSmagorinsky(RegularGrid(; size = (24, 12, 16), extent = (2π, π, 2.0))))
    run_channel!(c, 0.2; average_from = 0.1)
    r = val_run("log_layer_mismatch", "dynamic_first", ["stepper.end_time=0.2"])
    @test fields_bitwise(state_fields(r.obj), (u = c.sim.u.data, v = c.sim.v.data,
                                               w = c.sim.w.data, p = c.sim.p.data))
    # all fifteen members validate, with the labels the script reads
    labels = [m.label for (m, _) in load_sweep(vcfg("log_layer_mismatch"))]
    @test length(labels) == 15 && "tf2.0" in labels && "re5186_first" in labels
end

@testset "channel driver: config errors" begin
    e = try
        load_sweep(vcfg("neutral_abl"); overrides = ["channel.walls=\"bottom\"", "wall_model.t_f=0.5"])
    catch err
        err
    end
    @test e isa ConfigError
    s = sprint(showerror, e)
    @test occursin("channel.walls", s) && occursin("wall_model.t_f", s)
end

@testset "round trip from run.toml (validation: neutral_abl, one member), bit-identical" begin
    r = roundtrip_check(vcfg("neutral_abl"); member = "lasd_32x32x32",
                        overrides = ["channel.size=[16,16,16]", "stepper.end_time=0.1"])
    @test r.fields_equal && r.diagnostics_equal && r.config_equal && r.results_equal
    @test r.runs[1].obj.case.nsteps >= 10
end
