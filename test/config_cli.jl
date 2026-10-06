# The aeolus command line and the driver registry (CFG-b, #357):
# src/cli/main.jl, src/config/drivers.jl, bin/aeolus, Aeolus.main.
#
# Acceptance tests of the issue covered here (in-process, through
# Aeolus.main([...]), except the one subprocess smoke test):
#   * `check` on a bad file returns 2 and prints the dotted key;
#   * `resolve` prints text that load_run_config reads back to an equal config;
#   * `run` returns 0 and leaves a run folder with run.log, run.toml and
#     diagnostics.csv;
#   * `run --resume` keeps the run ID and appends a session;
#   * subprocess: `julia --project -m Aeolus check …` (Julia ≥ 1.12;
#     `bin/aeolus check …` before) exits with 0;
#   * exit codes: 1 for a failed run (its session recorded as failed), 2 for
#     usage and config errors, with nothing written.

using Test
using Aeolus
using Aeolus.Config
using Aeolus.IO: read_run_toml

const CLI_CFGS = joinpath(pkgdir(Aeolus), "examples", "configs")
const TGV = joinpath(CLI_CFGS, "taylor_green_vortex.toml")
const SMALL = ["--set", "grid.size=[8,8,8]", "--set", "stepper.end_time=\"none\"",
               "--set", "stepper.nsteps=4", "--set", "fftw.planner=\"estimate\""]

"Run `Aeolus.main(args)` in-process: (exit code, stdout text, stderr text)."
function cli(args...)
    out = IOBuffer(); err = IOBuffer()
    rc = Aeolus.main(String[args...]; io = out, err = err)
    return rc, String(take!(out)), String(take!(err))
end

cli_file(text) = (p = joinpath(mktempdir(), "c.toml"); write(p, text); p)

@testset "aeolus check / resolve" begin
    rc, out, _ = cli("check", TGV)
    @test rc == 0 && occursin("ok", out) && occursin("simulation", out)
    bad = cli_file("""
        [run]
        driver = "simulation"
        [grid]
        size = [16, 16]
        [stepper]
        dt = 0.1
        end_time = 1.0
        shceme = "euler"
        """)
    rc, out, err = cli("check", bad)
    @test rc == 2
    @test occursin("grid.size", err) && occursin("3 integers", err)
    @test occursin("stepper.shceme", err) && occursin("did you mean \"scheme\"", err)
    @test occursin(bad, err)
    @test isempty(out)
    # a bad override is a config error too
    rc, _, err = cli("check", TGV, "--set", "stepper.dt=-1.0")
    @test rc == 2 && occursin("stepper.dt", err)
    # resolve: the canonical text, which loads back to the same config
    rc, out, _ = cli("resolve", TGV, "--set", "grid.size=[16,16,16]")
    @test rc == 0
    cfg = load_run_config(IOBuffer(out))
    @test cfg == load_run_config(TGV; overrides = ["grid.size=[16,16,16]"])
    @test cfg.grid.size == (16, 16, 16)
    @test startswith(out, "schema_version = 1")
end

@testset "aeolus run, run --resume, run from run.toml" begin
    root = mktempdir()
    rc, out, err = cli("run", TGV, SMALL..., "--set", "checkpoint.every_steps=2",
                       "--root", root, "--allow-uncommitted")
    @test rc == 0
    folder = only(readdir(root; join = true))
    @test occursin(folder, out) && occursin("completed", out)
    for f in ("run.log", "run.toml", "diagnostics.csv")
        @test isfile(joinpath(folder, f))
    end
    @test countlines(joinpath(folder, "diagnostics.csv")) == 5          # header + 4 steps
    rec = read_run_toml(folder)
    @test rec["config"]["stepper"]["nsteps"] == 4
    @test rec["config"]["run"]["root"] == "runs"         # --root is not part of the config
    @test occursin("run started", read(joinpath(folder, "run.log"), String))
    # --resume continues that run: the same ID, a second session
    rc, out, _ = cli("run", folder, "--resume", "--allow-uncommitted")
    @test rc == 0
    rec2 = read_run_toml(folder)
    @test rec2["run_id"] == rec["run_id"]
    @test [s["kind"] for s in rec2["session"]] == ["start", "restart"]
    @test all(s -> s["status"] == "completed", rec2["session"])
    @test only(readdir(root)) == basename(folder)
    # a run.toml given to run starts a new run (a new ID and folder) with that config
    rc, _, _ = cli("run", joinpath(folder, "run.toml"), "--root", root, "--allow-uncommitted")
    @test rc == 0
    folders = readdir(root; join = true)
    @test length(folders) == 2
    new = only(filter(!=(folder), folders))
    @test read_run_toml(new)["run_id"] != rec["run_id"]
    @test read_run_toml(new)["config_hash"] == rec["config_hash"]
end

@testset "aeolus exit codes: 1 for a failed run, 2 for usage and config errors" begin
    root = mktempdir()
    # a step far beyond the stability limit: the fields are NaN by step 11 and the
    # blowup guard stops the run (after 4 steps they are still finite, |u| ~ 5e10)
    rc, _, err = cli("run", TGV, SMALL..., "--set", "stepper.dt=50.0", "--set", "stepper.nsteps=20", "--root", root,
                     "--allow-uncommitted")
    @test rc == 1
    @test occursin("run failed", err) && occursin("blew up", err)
    folder = only(readdir(root; join = true))
    @test read_run_toml(folder)["session"][1]["status"] == "failed"
    # usage and config errors write nothing
    root2 = mktempdir()
    for args in (["bogus"], ["run"], ["run", TGV, "--frobnicate", "--root", root2],
                 ["run", TGV, "--resume", "--root", root2],                  # not a run folder
                 ["run", folder, "--resume", "--set", "stepper.nsteps=8"],   # --set with --resume
                 ["run", TGV, "--set", "grid.size=[0,8,8]", "--root", root2],
                 ["run", TGV, "--set", "run.driver=\"common\"", "--root", root2],
                 ["check"], ["check", "/nonexistent/c.toml"], ["options", "nodriver"],
                 ["options", "simulation", "nosection"], ["preprocess", "nokind"])
        rc, _, err = cli(args...)
        @test rc == 2
        @test !isempty(err)
    end
    @test isempty(readdir(root2))
    @test length(read_run_toml(folder)["session"]) == 1
    rc, out, _ = cli("help")
    @test rc == 0 && occursin("usage: aeolus", out)
    @test cli()[1] == 2
end

@testset "aeolus options: the reference generated from the schema" begin
    rc, out, _ = cli("options", "simulation", "grid")
    @test rc == 0
    @test occursin("[grid]", out) && !occursin("[stepper]", out)
    @test occursin("size : [3 × integer] = [32, 32, 32]", out)
    @test occursin("extent : [3 × float] = [1.0, 1.0, 1.0]  [m]", out)
    rc, out, _ = cli("options", "simulation")
    sch = Config.driver_schema("simulation")
    for s in sch.sections, o in s.options          # every option of every section is listed
        @test occursin("\n  $(o.key) : ", out)
    end
    @test occursin("fields : [string, …] = []", out)        # [output] extended by the driver
    rc, out, _ = cli("options")
    @test rc == 0 && occursin("registered drivers: ", out) && occursin("simulation", out)
end

@testset "aeolus preprocess: registered preprocessors" begin
    seen = String[]
    register_preprocessor!("cfgtest_echo", args -> (append!(seen, args); 0), "cfgtest_echo <words…>")
    register_preprocessor!("cfgtest_fail", args -> throw(ArgumentError("bad input file")), "cfgtest_fail")
    try
        rc, out, _ = cli("preprocess")
        @test rc == 0 && occursin("cfgtest_echo <words…>", out)
        @test cli("preprocess", "cfgtest_echo", "a", "b")[1] == 0
        @test seen == ["a", "b"]
        rc, _, err = cli("preprocess", "cfgtest_fail")
        @test rc == 2 && occursin("bad input file", err)
        rc, _, err = cli("preprocess", "cfgtest_ech")
        @test rc == 2 && occursin("did you mean \"cfgtest_echo\"", err)
    finally
        delete!(Config._PREPROCESSORS, "cfgtest_echo"); delete!(Config._PREPROCESSORS, "cfgtest_fail")
    end
end

@testset "example drivers are loaded on first use (DRIVER_FILES)" begin
    dir = mktempdir()
    drv = joinpath(dir, "toy_driver.jl")
    write(drv, """
        using Aeolus, Aeolus.Config
        struct ToyDriver <: Config.AbstractRunDriver end
        mutable struct ToyRun; x::Vector{Float64}; n::Int; end
        Config.driver_name(::ToyDriver) = "cfgtest_toy"
        Config.driver_sections(::ToyDriver) = [section("toy") do s
            option!(s, "steps", Int; default = 3, range = (0, 100))
        end]
        Config.build(::ToyDriver, cfg) = ToyRun([1.0], cfg.toy.steps)
        function Aeolus.run!(::ToyDriver, r::ToyRun, rf, ck; callback = nothing)
            for _ in 1:r.n; r.x[1] *= 2; end
            return r
        end
        Config.state_fields(r::ToyRun) = (x = r.x,)
        register_driver!(ToyDriver())
        """)
    Config.DRIVER_FILES["cfgtest_toy"] = drv            # absolute: joinpath keeps it
    try
        @test !("cfgtest_toy" in registered_drivers())
        p = cli_file("[run]\ndriver = \"cfgtest_toy\"\n[toy]\nsteps = 5\n")
        cfg = load_run_config(p)                        # the schema lookup loads the file
        @test "cfgtest_toy" in registered_drivers()
        @test cfg.toy.steps == 5
        res = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir())
        @test Base.invokelatest(state_fields, res.obj).x == [32.0]   # defined after this testset compiled
        # a listed file that registers nothing is a config error
        Config.DRIVER_FILES["cfgtest_none"] = cli_file("x = 1")
        e = try load_run_config(cli_file("[run]\ndriver = \"cfgtest_none\"\n")); catch err; err; end
        @test e isa ConfigError && occursin("did not register", sprint(showerror, e))
        # "common" has a schema but no driver
        @test_throws ConfigError run_driver("common")
    finally
        delete!(Config.DRIVER_FILES, "cfgtest_toy"); delete!(Config.DRIVER_FILES, "cfgtest_none")
    end
end

@testset "aeolus as a subprocess" begin
    # julia --project -m Aeolus needs Julia ≥ 1.12 (the -m flag); bin/aeolus otherwise
    root = pkgdir(Aeolus)
    cmd = VERSION >= v"1.12" ?
        `$(Base.julia_cmd()) --startup-file=no --project=$root -m Aeolus check $TGV` :
        `$(Base.julia_cmd()) --startup-file=no --project=$root $(joinpath(root, "bin", "aeolus")) check $TGV`
    p = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = devnull))
    @test p.exitcode == 0
    bad = cli_file("[run]\ndriver = \"simulation\"\n[grid]\nsize = 3\n")
    p = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --startup-file=no --project=$root $(joinpath(root, "bin", "aeolus")) check $bad`);
                     stdout = devnull, stderr = devnull))
    @test p.exitcode == 2
end
