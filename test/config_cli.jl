# The aeolus command line and the schema registry (CFG-b, #357; ONE-b, #451):
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
const SMALL = ["--set", "grid.size=[8,8,8]", "--set", "time.duration=\"none\"",
               "--set", "time.nsteps=4", "--set", "fftw.planner=\"estimate\""]

"Run `Aeolus.main(args)` in-process: (exit code, stdout text, stderr text)."
function cli(args...)
    out = IOBuffer(); err = IOBuffer()
    rc = Aeolus.main(String[args...]; io = out, err = err)
    return rc, String(take!(out)), String(take!(err))
end

cli_file(text) = (p = joinpath(mktempdir(), "c.toml"); write(p, text); p)

@testset "aeolus check / resolve" begin
    rc, out, _ = cli("check", TGV)
    @test rc == 0 && occursin("ok", out) && occursin("the model", out)
    bad = cli_file("""
        schema_version = 2
        [grid]
        size = [16, 16]
        [time]
        dt = 0.1
        duration = 1.0
        shceme = "euler"
        """)
    rc, out, err = cli("check", bad)
    @test rc == 2
    @test occursin("grid.size", err) && occursin("3 integers", err)
    @test occursin("time.shceme", err) && occursin("did you mean \"scheme\"", err)
    @test occursin(bad, err)
    @test isempty(out)
    # a bad override is a config error too
    rc, _, err = cli("check", TGV, "--set", "time.dt=-1.0")
    @test rc == 2 && occursin("time.dt", err)
    # resolve: the canonical text, which loads back to the same config
    rc, out, _ = cli("resolve", TGV, "--set", "grid.size=[16,16,16]")
    @test rc == 0
    cfg = load_run_config(IOBuffer(out))
    @test cfg == load_run_config(TGV; overrides = ["grid.size=[16,16,16]"])
    @test cfg.grid.size == (16, 16, 16)
    @test startswith(out, "schema_version = 2")
    @test occursin("\n[[domain]]\n", out) && occursin("\n[domain.grid]\n", out)
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
    @test rec["config"]["time"]["nsteps"] == 4
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
    rc, _, err = cli("run", TGV, SMALL..., "--set", "time.dt=50.0", "--set", "time.nsteps=20", "--root", root,
                     "--allow-uncommitted")
    @test rc == 1
    @test occursin("run failed", err) && occursin("blew up", err)
    folder = only(readdir(root; join = true))
    @test read_run_toml(folder)["session"][1]["status"] == "failed"
    # usage and config errors write nothing
    root2 = mktempdir()
    for args in (["bogus"], ["run"], ["run", TGV, "--frobnicate", "--root", root2],
                 ["run", TGV, "--resume", "--root", root2],                  # not a run folder
                 ["run", folder, "--resume", "--set", "time.nsteps=8"],      # --set with --resume
                 ["run", TGV, "--set", "grid.size=[0,8,8]", "--root", root2],
                 ["run", TGV, "--set", "run.driver=\"simulation\"", "--root", root2],   # removed (ONE-b)
                 ["run", TGV, "--set", "run.legacy=\"nocase\"", "--root", root2],
                 ["check"], ["check", "/nonexistent/c.toml"], ["options", "nosection"],
                 ["options", "nocase", "grid"], ["options", "cbl", "nosection"],
                 ["preprocess", "static", TGV])                               # AeolusPrep.jl's since ONE-h (#457)
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
    rc, out, _ = cli("options", "grid")                  # a section of the model
    @test rc == 0
    @test occursin("[grid]", out) && !occursin("[time]", out)
    @test occursin("size : [3 × integer] = [32, 32, 32]", out)
    @test occursin("extent : [3 × float] = [3200.0, 3200.0, 1600.0]  [m]", out)
    @test occursin("Per domain", out)                    # a per-domain section says so
    rc, out, _ = cli("options")
    @test rc == 0
    sch = Config.driver_schema()
    @test sch.name == "model"
    for s in sch.sections, o in s.options          # every option of every section is listed
        @test occursin("\n  $(o.key) : ", out)
    end
    # [output] extended by the model; its default set (ONE-c): u, v, w, theta and the domain means
    @test occursin("variables : [string, …] = [\"u\", \"v\", \"w\", \"theta\", \"u_mean\"", out)
    @test occursin("legacy cases (aeolus options <case>): ", out) && occursin("cbl", out) && occursin("urban", out)
    # the cross-field checks, in words (CFG-i, #364): after the sections, one line each
    @test occursin("\ncross-field checks:\n  - exactly one of `time.duration` and `time.nsteps`", out)
    rc, out1, _ = cli("options", "grid")
    @test !occursin("cross-field checks", out1)          # one section: no checks
    # a legacy case, and one of its sections
    rc, out, _ = cli("options", "cbl", "cbl")
    @test rc == 0 && occursin("[cbl]", out) && occursin("the legacy case \"cbl\"", out)
    # --markdown: the docs' tables, the same text option_reference(...; format = :markdown) gives
    rc, md, _ = cli("options", "--markdown")
    @test rc == 0 && startswith(md, option_reference(; format = :markdown))
    md = option_reference(; format = :markdown)
    @test occursin("\n## `[grid]`\n", md) && occursin("\n## Cross-field checks\n", md)
    @test occursin("| `extent` | [3 × float] | `[3200.0, 3200.0, 1600.0]` | `m` | (0, Inf] |", md)
    # `|` inside a cell is escaped, so a row keeps its six cells
    @test occursin("| `dt` | float \\| string | `\"adaptive\"` | `s` |", md)
    @test_throws ArgumentError option_reference(; format = :html)
end

@testset "a legacy case is a registered schema; nothing is loaded from examples/" begin
    # no file under examples/ registers a case or builds a model (ONE-i #458):
    # the legacy cases are exactly the schemas registered by the package
    @test legacy_cases() == sort!(filter(n -> !(n in (Config.MODEL, "common")), registered_drivers()))
    @test !isdefined(Config, :DRIVER_FILES)
    for gone in ("column", "dns_decay")                 # their files left examples/ (ONE-i #458)
        e = try load_run_config(cli_file("[run]\nlegacy = \"$gone\"\n")); nothing; catch err; err; end
        @test e isa ConfigError && occursin("unknown legacy case", sprint(showerror, e))
    end
    # "common" has a schema but nothing runs it
    @test_throws ConfigError run_driver("common")
end

@testset "aeolus as a subprocess" begin
    # julia --project -m Aeolus needs Julia ≥ 1.12 (the -m flag); bin/aeolus otherwise
    root = pkgdir(Aeolus)
    cmd = VERSION >= v"1.12" ?
        `$(Base.julia_cmd()) --startup-file=no --project=$root -m Aeolus check $TGV` :
        `$(Base.julia_cmd()) --startup-file=no --project=$root $(joinpath(root, "bin", "aeolus")) check $TGV`
    p = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = devnull))
    @test p.exitcode == 0
end
