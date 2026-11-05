# The example scripts configured from TOML (CFG-e, #360).
#
# Acceptance tests of the issue covered in this file:
#   * sweeps: a [sweep] table expands into override sets in a fixed order,
#     one run folder per member, each member validated.
# The per-directory bit-identity and round-trip tests are in
# config_examples_*.jl.

using Test
using Aeolus
using Aeolus.Config
using Aeolus.IO: read_run_toml
using Aeolus.Fields: interior_view
import NCDatasets

const EX_DIR = joinpath(pkgdir(Aeolus), "examples")
const TGV_CFG = joinpath(EX_DIR, "taylor_green_vortex", "taylor_green_vortex.toml")

ex_file(text) = (p = joinpath(mktempdir(), "sweep.toml"); write(p, text); p)

"The TGV config's text without its [test] table, then `extra` (no extends: ONE-b, #451)."
tgv_with(extra) = first(split(read(TGV_CFG, String), "\n[test]")) * "\n" * extra

function ex_error(f)
    try
        f()
    catch e
        e isa ConfigError && return e
        rethrow()
    end
    error("expected a ConfigError")
end

@testset "sweeps: expansion order, labels, run names, precedence" begin
    p = ex_file(tgv_with("""
        [sweep]
        axis = [ { key = "sgs.model", values = ["none", "vreman", "amd"] },
                 { key = "grid.size", values = [[8, 8, 8], [16, 16, 16]] } ]
        """))
    ms = sweep_members(p)
    @test length(ms) == 6
    # first axis outermost, the last fastest: the order of `for a in A for b in B`
    @test [m.label for m in ms] == ["none_8x8x8", "none_16x16x16", "vreman_8x8x8",
                                    "vreman_16x16x16", "amd_8x8x8", "amd_16x16x16"]
    @test ms[3].overrides == ["sgs.model=\"vreman\"", "grid.size=[8, 8, 8]"]
    runs = load_sweep(p)
    @test [c.sgs.model for (_, c) in runs] == [:none, :none, :vreman, :vreman, :amd, :amd]
    @test [c.grid.size for (_, c) in runs] == repeat([(8, 8, 8), (16, 16, 16)], 3)
    @test [c.run.name for (_, c) in runs] == ["taylor_green_vortex_" * m.label for m in ms]
    # every member carries no [sweep]: its canonical text has no such table
    @test all(!occursin("[sweep]", config_text(c)) for (_, c) in runs)
    # user overrides come last: base < member < --set
    runs = load_sweep(p; overrides = ["grid.size=[4,4,4]", "run.name=\"x\""])
    @test all(c.grid.size == (4, 4, 4) && c.run.name == "x" for (_, c) in runs)
    # one member by index or label
    @test only(load_sweep(p; member = 4))[2].sgs.model === :vreman
    @test only(load_sweep(p; member = "amd_8x8x8"))[1].index == 5
    e = ex_error(() -> load_sweep(p; member = 7))
    @test occursin("no member 7", sprint(showerror, e)) && occursin("amd_16x16x16", sprint(showerror, e))
    # the plain loader refuses a sweep and says how to run it
    e = ex_error(() -> load_run_config(p))
    @test occursin("sweep of 6 members", sprint(showerror, e))
end

@testset "sweeps: explicit members, validation of each member" begin
    p = ex_file(tgv_with("""
        [[sweep.member]]
        label = "viscous"
        dynamics = { nu = 0.02 }
        [[sweep.member]]
        sgs.model = "vreman"
        sgs.c_s = 0.2
        [sweep]
        axis = [ { key = "timestep.scheme", values = ["euler", "rk3"] } ]
        """))
    runs = load_sweep(p)
    @test [m.label for (m, _) in runs] == ["viscous_euler", "viscous_rk3", "2_euler", "2_rk3"]
    @test [c.dynamics.nu for (_, c) in runs] == [0.02, 0.02, 0.01, 0.01]
    @test runs[3][2].sgs.c_s == 0.2 && runs[3][2].timestep.scheme === :euler
    # a member that is invalid is reported with its key
    bad = ex_file(tgv_with("""
        [[sweep.member]]
        sgs = { model = "smagorinksy" }
        """))
    e = ex_error(() -> load_sweep(bad))
    @test occursin("sgs.model", sprint(showerror, e)) && occursin("smagorinsky", sprint(showerror, e))
    # malformed sweep tables
    e = ex_error(() -> sweep_members(ex_file("[sweep]\naxis = [ { key = \"nu\", values = [1] } ]\nmembers = 1\n")))
    s = sprint(showerror, e)
    @test occursin("section.option", s) && occursin("sweep.members", s) && occursin("did you mean \"member\"", s)
    @test sweep_members(TGV_CFG) === nothing
end

@testset "sweeps: aeolus run runs every member into its own folder; a member replays alone" begin
    p = ex_file(tgv_with("""
        [sweep]
        axis = [ { key = "dynamics.nu", values = [0.01, 0.02] } ]
        """))
    root = mktempdir()
    sets = ["grid.size=[8,8,8]", "time.duration=\"none\"", "time.nsteps=3", "fftw.planner=\"estimate\""]
    args = vcat(collect(Iterators.flatten(("--set", s) for s in sets)), "--root", root)
    # every member, one run each, as aeolus run takes them
    res = [run_config(cfg; root = root, console = false, repo = mktempdir()) for (_, cfg) in load_sweep(p; overrides = sets)]
    @test length(res) == 2
    @test res[1].folder.path != res[2].folder.path
    @test [Aeolus.SimulationDriver.simulation(r.obj).ν for r in res] == [0.01, 0.02]
    @test all(r.obj.stepper.step == 3 for r in res)
    rec = read_run_toml(res[2].folder.path)
    @test !haskey(rec["config"], "sweep")
    @test "dynamics.nu=0.02" in rec["config_sources"]["overrides"]
    # the member's run.toml loads as a single run with the member's values
    cfg = load_run_config(res[2].folder.path)
    @test cfg.dynamics.nu == 0.02 && cfg.run.name == "taylor_green_vortex_0.02"
    # --member through the command line; unexpected positionals are refused
    out = IOBuffer()
    @test Aeolus.main(["run", p, args..., "--member", "2", "--allow-uncommitted"]; io = out, err = IOBuffer()) == 0
    @test count("completed", String(take!(out))) == 1
    @test Aeolus.main(["run", p, "32"]; io = IOBuffer(), err = IOBuffer()) != 0
    out = IOBuffer()
    @test Aeolus.main(["check", p]; io = out) == 0
    @test count("ok:", String(take!(out))) == 2
end
