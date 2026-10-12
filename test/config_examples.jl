# The example scripts configured from TOML (CFG-e, #360).
#
# Acceptance tests of the issue covered in this file:
#   * sweeps: a [sweep] table expands into override sets in a fixed order,
#     one run folder per member, each member validated;
#   * the Kessler box of the kernel demos (cloud_spinup.toml) on the model.
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
        axis = [ { key = "sgs.closure", values = ["none", "vreman", "amd"] },
                 { key = "grid.size", values = [[8, 8, 8], [16, 16, 16]] } ]
        """))
    ms = sweep_members(p)
    @test length(ms) == 6
    # first axis outermost, the last fastest: the order of `for a in A for b in B`
    @test [m.label for m in ms] == ["none_8x8x8", "none_16x16x16", "vreman_8x8x8",
                                    "vreman_16x16x16", "amd_8x8x8", "amd_16x16x16"]
    @test ms[3].overrides == ["sgs.closure=\"vreman\"", "grid.size=[8, 8, 8]"]
    runs = load_sweep(p)
    @test [c.sgs.closure for (_, c) in runs] == [:none, :none, :vreman, :vreman, :amd, :amd]
    @test [c.grid.size for (_, c) in runs] == repeat([(8, 8, 8), (16, 16, 16)], 3)
    @test [c.run.name for (_, c) in runs] == ["taylor_green_vortex_" * m.label for m in ms]
    # every member carries no [sweep]: its canonical text has no such table
    @test all(!occursin("[sweep]", config_text(c)) for (_, c) in runs)
    # user overrides come last: base < member < --set
    runs = load_sweep(p; overrides = ["grid.size=[4,4,4]", "run.name=\"x\""])
    @test all(c.grid.size == (4, 4, 4) && c.run.name == "x" for (_, c) in runs)
    # one member by index or label
    @test only(load_sweep(p; member = 4))[2].sgs.closure === :vreman
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
        sgs.closure = "vreman"
        sgs.c_s = 0.2
        [sweep]
        axis = [ { key = "time.scheme", values = ["euler", "rk3"] } ]
        """))
    runs = load_sweep(p)
    @test [m.label for (m, _) in runs] == ["viscous_euler", "viscous_rk3", "2_euler", "2_rk3"]
    @test [c.dynamics.nu for (_, c) in runs] == [0.02, 0.02, 0.01, 0.01]
    @test runs[3][2].sgs.c_s == 0.2 && runs[3][2].time.scheme === :euler
    # a member that is invalid is reported with its key
    bad = ex_file(tgv_with("""
        [[sweep.member]]
        sgs = { closure = "smagorinksy" }
        """))
    e = ex_error(() -> load_sweep(bad))
    @test occursin("sgs.closure", sprint(showerror, e)) && occursin("smagorinsky", sprint(showerror, e))
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

# ---------------------------------------------------------------------------
# examples/cloud_spinup/: kernel demo 04, the Kessler box on the model (ONE-i #458)
# ---------------------------------------------------------------------------

include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

@testset "kernels: cloud_spinup.toml is the Kessler box at rest on the model" begin
    # the same box with the same eight Kessler steps applied directly
    # (Microphysics.apply_kessler_step!, the Marshall-Palmer fall speeds)
    H = 2000.0
    mesh = RegularGrid(; size = (4, 4, 32), extent = (H, H, H), halo = (1, 1, 1))
    qv = Field(CellCenter(), mesh); Aeolus.Fields.set!(qv, (x, y, z) -> 0.01)
    ql = Field(CellCenter(), mesh); Aeolus.Fields.set!(ql, (x, y, z) -> 0.6H < z < 0.9H ? 3e-3 : 0.0)
    qr = Field(CellCenter(), mesh); Aeolus.Fields.set!(qr, (x, y, z) -> 0.0)
    q0 = sum(interior_view(qv)) + sum(interior_view(ql))
    for _ in 1:8
        apply_kessler_step!(qv, ql, qr, 60.0; Vt = :spectrum)
    end
    cfg = load_run_config(joinpath(EX_DIR, "cloud_spinup", "cloud_spinup.toml"))
    root = mktempdir()
    r = run_config(cfg; root = root, console = false, repo = mktempdir())
    sim = Aeolus.SimulationDriver.simulation(r.obj)
    water = Aeolus.SimulationDriver.sim_couplings(r.obj).water
    @test sim.t ≈ 480.0
    # horizontally uniform fields: the layer's buoyancy is balanced by the
    # pressure, the air stays at rest and the run is the microphysics alone
    @test maximum(abs, interior_view(sim.u)) == 0 && maximum(abs, interior_view(sim.v)) == 0
    @test maximum(abs, interior_view(sim.w)) < 1e-12
    @test isapprox(interior_view(sim.qv), interior_view(qv); rtol = 1e-12)
    @test isapprox(interior_view(water.ql), interior_view(ql); rtol = 1e-12, atol = 1e-15)
    @test isapprox(interior_view(water.qr), interior_view(qr); rtol = 1e-12, atol = 1e-15)
    # the physics, from the run's profiles (diagnostics.nc): autoconversion
    # turns cloud into rain (the cloud maximum falls monotonically) and the
    # rain reaches below the cloud base
    ds = NCDatasets.NCDataset(joinpath(r.folder.path, "diagnostics.nc"))
    z = Array(ds["z"][:]); QL = Array(ds["micro.ql"][:, :]); QR = Array(ds["micro.qr"][:, :])
    t = Array(ds["time_costly"][:]); close(ds)
    # a row per step, written after it (the run monitor records no row of
    # the initial state): t = 60, 120, …, 480 s
    @test t ≈ 60:60:480
    qlmax = vec(maximum(QL; dims = 1))
    @test issorted(qlmax; rev = true) && qlmax[end] < qlmax[1]
    # the core step writes the row (the run monitor is attached to it), before
    # the step's Kessler update: the row at 60 s still holds the initial
    # cloud and no rain (the direct steps above: 0.0192 summed qr after one);
    # at the end the rain is there
    @test maximum(QR[:, 1]) == 0 && maximum(QR[:, end]) > 1e-5
    @test any(QR[k, end] > 0 for k in eachindex(z) if z[k] < 0.6H)
    # vapour and cloud only lose water to the rain (no source of water)
    @test sum(interior_view(sim.qv)) + sum(interior_view(water.ql)) <= q0
end

@testset "round trip from run.toml (kernels: cloud_spinup.toml), bit-identical" begin
    r = roundtrip_check(joinpath(EX_DIR, "cloud_spinup", "cloud_spinup.toml"))
    @test r.fields_equal && r.config_equal && r.results_equal
end

# (The coverage test of the scripts that integrate the equations is
# test/examples_enforcement.jl since ONE-i #458: no file under examples/
# builds or steps a model, with no exemptions.)
