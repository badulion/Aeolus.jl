# Config loader: extends, overrides, run.toml, inputs, common sections
# (CFG-a, #356): src/config/{load,run_record,common}.jl.
#
# Acceptance tests of the issue covered here:
#   * merge semantics: a 3-level extends chain plus 2 overrides gives the
#     resolved table written out literally below; base < child < override;
#   * extends cycles and missing extends files are ConfigErrors;
#   * run.toml: a run started with a RunConfig stores [config] such that
#     load_run_config(folder) == cfg with an equal config_hash, also after
#     a restart (test/logging_run_folder.jl covers the toml_value callers,
#     unchanged);
#   * inputs: changing one byte of a referenced input file makes
#     load_run_config(folder) throw, naming the file;
#   * common sections: checkpoint_schedule(cfg) and apply_fftw_config!(cfg)
#     give the same results from a RunConfig as from a Dict.

using Test
using TOML
using Aeolus
using Aeolus.Config
using Aeolus.Operators: fftw_policy, set_fftw_policy!

include(joinpath(@__DIR__, "support", "config_test_schema.jl"))

function cfgl_error(f)
    try
        f()
    catch e
        e isa ConfigError && return e
        rethrow()
    end
    error("expected a ConfigError")
end

"Start a run of `cfg` in a fresh root, outside any git work tree, silently."
cfgl_start(cfg; kw...) = start_run(cfg; root = get(kw, :root, mktempdir()), repo = mktempdir(),
                                   console = false, kw...)

@testset "Config: 3-level extends chain + 2 overrides" begin
    dir = mktempdir()
    mkpath(joinpath(dir, "base"))
    write(joinpath(dir, "base", "base.toml"), """
        [run]
        driver = "cfgtest"
        name = "base"
        [grid]
        size = [64, 64, 32]
        stretch = 1.1
        levels = [0.0, 5.0, 10.0]
        [clock]
        nsteps = 100
        dt = 1.0
        [physics]
        label = "base"
        tags = ["a", "b"]
        terrain = "dem.tif"           # relative to base/
        [physics.couplings.wall]
        z0 = 0.1
        kind = "log"
        """)
    write(joinpath(dir, "mid.toml"), """
        extends = "base/base.toml"
        [grid]
        stretch = 1.2
        levels = [1.0]                # arrays replace, they do not merge
        [clock]
        dt = 0.5
        [physics]
        tags = ["c"]
        [physics.couplings.wall]
        z0 = 0.5                      # tables merge key by key
        [physics.couplings.canopy]
        lai = 3.0
        """)
    child = joinpath(dir, "child.toml")
    write(child, """
        extends = "mid.toml"
        [run]
        name = "child"
        [clock]
        nsteps = 200
        scheme = "ab2"
        """)
    cfg = load_run_config(child; overrides = ["clock.nsteps=300", "grid.size=[16,16,8]"])
    t = to_toml(cfg)
    @test t["grid"] == Dict{String,Any}("size" => [16, 16, 8], "extent" => [100.0, 100.0, 50.0],
                                        "stretch" => 1.2, "levels" => [1.0])
    @test t["clock"] == Dict{String,Any}("nsteps" => 300, "dt" => 0.5, "scheme" => "ab2",
                                         "adaptive" => true, "dt_max" => "none",
                                         "start_date" => Date(2024, 7, 1), "start_time" => Time(6))
    @test t["physics"] == Dict{String,Any}(
        "label" => "base", "tags" => ["c"], "terrain" => joinpath(dir, "base", "dem.tif"),
        "inputs" => Any[], "profile" => "linear",
        "couplings" => Dict{String,Any}("wall" => Dict{String,Any}("z0" => 0.5, "kind" => "log"),
                                        "canopy" => Dict{String,Any}("lai" => 3.0)))
    @test t["run"]["name"] == "child" && t["run"]["driver"] == "cfgtest"
    @test !haskey(t, "extends") && t["schema_version"] == 1
    # where each value came from
    @test config_origin(cfg, "clock.nsteps") == "--set"
    @test config_origin(cfg, "clock.scheme") == child
    @test config_origin(cfg, "clock.dt") == joinpath(dir, "mid.toml")
    @test config_origin(cfg, "physics.label") == joinpath(dir, "base", "base.toml")
    @test config_sources(cfg)["files"] == [child, joinpath(dir, "mid.toml"), joinpath(dir, "base", "base.toml")]
    @test config_sources(cfg)["overrides"] == ["clock.nsteps=300", "grid.size=[16,16,8]"]
    # precedence base < child < override, one key at a time
    @test load_run_config(child).clock.nsteps == 200
    @test load_run_config(joinpath(dir, "mid.toml")).clock.nsteps == 100
    # an override with a table value, and one with a string
    c2 = load_run_config(child; overrides = ["physics.couplings.wall={z0=2.0}", "run.name=\"x y\""])
    @test c2.physics.couplings["wall"] == Dict("z0" => 2.0)          # an override table replaces
    @test c2.run.name == "x y"
    # override problems are collected with the file "--set"
    e = cfgl_error(() -> load_run_config(child; overrides = ["clock.nsteps", "run.name=x y",
                                                              "clock.nstep=3", "grid.size=[1,2]"]))
    @test length(e.problems) == 4
    m = sprint(showerror, e)
    @test occursin("(--set)", m) && occursin("strings are quoted", m) && occursin("did you mean \"nsteps\"", m)
    # the extends chain also resolves through a run folder-less IOBuffer (base dir = pwd)
    @test load_run_config(IOBuffer("[run]\ndriver = \"cfgtest\"\n[clock]\nnsteps = 5\n")).clock.nsteps == 5
end

@testset "Config: extends lookup, cycles, missing files, depth" begin
    # not next to the extending file: the driver's config directory
    write(joinpath(CFGT_DIR, "cfgt_base.toml"), "[run]\ndriver = \"cfgtest\"\n[clock]\nnsteps = 42\n")
    dir = mktempdir()
    p = joinpath(dir, "case.toml")
    write(p, "extends = \"cfgt_base.toml\"\n[run]\ndriver = \"cfgtest\"\n")
    cfg = load_run_config(p)
    @test cfg.clock.nsteps == 42
    @test config_sources(cfg)["files"][2] == joinpath(CFGT_DIR, "cfgt_base.toml")
    # without a driver in the file: every registered config directory is searched
    write(p, "extends = \"cfgt_base.toml\"\n")
    @test load_run_config(p).clock.nsteps == 42
    # a file next to the extending one wins over the driver directory
    write(joinpath(dir, "cfgt_base.toml"), "[run]\ndriver = \"cfgtest\"\n[clock]\nnsteps = 7\n")
    @test load_run_config(p).clock.nsteps == 7
    # the shipped-config fixed point covers files in driver directories
    @test joinpath(CFGT_DIR, "cfgt_base.toml") in shipped_configs()
    # a cycle
    write(joinpath(dir, "a.toml"), "extends = \"b.toml\"\n")
    write(joinpath(dir, "b.toml"), "extends = \"a.toml\"\n")
    m = sprint(showerror, cfgl_error(() -> load_run_config(joinpath(dir, "a.toml"))))
    @test occursin("extends cycle", m) && occursin("a.toml -> ", m) && occursin("b.toml", m)
    # self-extension is a cycle too
    write(joinpath(dir, "self.toml"), "extends = \"self.toml\"\n")
    @test_throws ConfigError load_run_config(joinpath(dir, "self.toml"))
    # a missing file, named with where it was looked for
    write(joinpath(dir, "m.toml"), "extends = \"nowhere.toml\"\n")
    m = sprint(showerror, cfgl_error(() -> load_run_config(joinpath(dir, "m.toml"))))
    @test occursin("extends", m) && occursin("nowhere.toml", m) && occursin("not found", m) &&
          occursin(joinpath(dir, "m.toml"), m)
    # depth: 8 extends links are accepted, 9 are not
    for n in 0:9
        write(joinpath(dir, "d$n.toml"), n == 9 ? "[clock]\nnsteps = 9\n[run]\ndriver = \"cfgtest\"\n" :
                                          "extends = \"d$(n + 1).toml\"\n")
    end
    @test load_run_config(joinpath(dir, "d1.toml")).clock.nsteps == 9            # 8 links
    m = sprint(showerror, cfgl_error(() -> load_run_config(joinpath(dir, "d0.toml"))))   # 9 links
    @test occursin("deeper than 8", m)
    # extends must be a string
    write(joinpath(dir, "bad.toml"), "extends = 3\n")
    @test_throws ConfigError load_run_config(joinpath(dir, "bad.toml"))
    # a missing configuration file
    @test_throws ConfigError load_run_config(joinpath(dir, "absent.toml"))
end

@testset "Config: run.toml stores the canonical config" begin
    dir = mktempdir()
    input = joinpath(dir, "terrain.dat"); write(input, "elevation 34.0\n")
    p = joinpath(dir, "case.toml")
    write(p, """
        [run]
        driver = "cfgtest"
        name = "cfgrun"
        seed = 7
        [clock]
        dt = 0.1
        start_time = 07:32:00.5
        [physics]
        terrain = "terrain.dat"
        [fftw]
        wisdom = "wisdom.dat"         # input = false: not hashed
        """)
    cfg = load_run_config(p; overrides = ["clock.nsteps=12"])
    root = mktempdir()
    rf = cfgl_start(cfg; root = root)
    finish_run!(rf)
    @test startswith(basename(rf.path), Libc.strftime("%Y-%m-%d", time()) * "_cfgrun_")
    rec = read_run_toml(rf.path)
    @test rec["config"] == to_toml(cfg)
    @test rec["config_schema_version"] == 1
    @test rec["config_hash"] == string(config_hash(cfg); base = 16, pad = 16)
    @test rec["config_sources"]["files"] == [p] && rec["config_sources"]["overrides"] == ["clock.nsteps=12"]
    @test rec["replayable"] === true && !haskey(rec, "nonreplayable_keys")
    @test rec["inputs"] == Dict("physics.terrain" => Dict("path" => input,
                                                          "fnv1a64" => content_hash(input)))
    @test content_hash(input) == string(Aeolus.IO.fnv1a64(read(input)); base = 16, pad = 16)
    # the folder, its run.toml and the record's text all load back to cfg
    for src in (rf.path, joinpath(rf.path, "run.toml"))
        c = load_run_config(src)
        @test c == cfg && config_hash(c) == config_hash(cfg)
        @test write_config(c) == write_config(cfg)
    end
    # overrides apply on top of a run folder too
    @test load_run_config(rf.path; overrides = ["clock.nsteps=13"]).clock.nsteps == 13
    # a restart from the folder's config: same config, no "changed" entry
    cfgr = load_run_config(rf.path)
    rf2 = cfgl_start(cfgr; root = root, resume = rf.path)
    finish_run!(rf2)
    rec2 = read_run_toml(rf.path)
    @test length(rec2["session"]) == 2 && !haskey(rec2["session"][2], "config")
    @test rec2["config_hash"] == rec["config_hash"]
    @test load_run_config(rf.path) == cfg
    # a restart with a different config records it in its session
    rf3 = cfgl_start(with(cfg, "clock.nsteps" => 24); root = root, resume = rf.path)
    finish_run!(rf3)
    s3 = read_run_toml(rf.path)["session"][3]
    @test s3["config"]["clock"]["nsteps"] == 24 && s3["config_hash"] != rec["config_hash"]
    @test load_run_config(rf.path) == cfg              # the run's config is the first session's
    # [run] and [logging] set start_run's defaults: root, allow_uncommitted
    root2 = mktempdir()
    rf4 = start_run(with(cfg, "run.root" => root2); repo = mktempdir(), console = false)
    finish_run!(rf4)
    @test dirname(rf4.path) == root2
    @test Aeolus.IO.run_options(cfg).flush_interval == 5.0
    # a config not made from a RunConfig has no config_schema_version: refused
    rfd = start_run("plain"; root = root, repo = mktempdir(), console = false,
                    config = Dict("clock" => Dict("nsteps" => 3)))
    finish_run!(rfd)
    @test_throws ConfigError load_run_config(rfd.path)
end

@testset "Config: changed input files are refused" begin
    dir = mktempdir()
    input = joinpath(dir, "terrain.dat"); write(input, "elevation 34.0\n")
    extra = joinpath(dir, "extra.nc"); write(extra, "x")
    p = joinpath(dir, "case.toml")
    write(p, "[run]\ndriver = \"cfgtest\"\n[physics]\nterrain = \"terrain.dat\"\ninputs = [\"extra.nc\"]\n")
    cfg = load_run_config(p)
    root = mktempdir()
    rf = cfgl_start(cfg; root = root); finish_run!(rf)
    @test Set(keys(read_run_toml(rf.path)["inputs"])) == Set(["physics.terrain", "physics.inputs[1]"])
    # one byte changed
    bytes = read(input); bytes[1] = UInt8('E'); write(input, bytes)
    e = cfgl_error(() -> load_run_config(rf.path))
    m = sprint(showerror, e)
    @test length(e.problems) == 1 && e.problems[1].key == "physics.terrain"
    @test occursin(input, m) && occursin("changed since the run started", m) &&
          occursin("allow_changed_inputs", m)
    @test load_run_config(rf.path; allow_changed_inputs = true) == cfg
    # a restart refuses too, before writing anything
    nsess = length(read_run_toml(rf.path)["session"])
    @test_throws ConfigError cfgl_start(cfg; root = root, resume = rf.path)
    @test length(read_run_toml(rf.path)["session"]) == nsess
    rf2 = cfgl_start(cfg; root = root, resume = rf.path, allow_changed_inputs = true)
    finish_run!(rf2)
    @test length(read_run_toml(rf.path)["session"]) == nsess + 1
    # a deleted input is a change as well
    rm(extra)
    e = cfgl_error(() -> load_run_config(rf.path))
    @test Set(q.key for q in e.problems) == Set(["physics.terrain", "physics.inputs[1]"])
    @test occursin("now missing", sprint(showerror, e))
end

@testset "Config: a closure makes the run not replayable" begin
    cfg = with(load_run_config(IOBuffer("[run]\ndriver = \"cfgtest\"\nname = \"jl\"\n")),
               "physics.profile" => z -> 300.0 + 0.003z)
    rf = cfgl_start(cfg); finish_run!(rf)
    rec = read_run_toml(rf.path)
    @test rec["replayable"] === false && rec["nonreplayable_keys"] == ["physics.profile"]
    @test startswith(rec["config"]["physics"]["profile"], "<julia: ")
    m = sprint(showerror, cfgl_error(() -> load_run_config(rf.path)))
    @test occursin("not replayable", m) && occursin("physics.profile", m)
end

@testset "Config: common sections drive checkpoints and FFTW like the Dict" begin
    p0 = fftw_policy()
    try
        d = Dict{String,Any}("fftw" => Dict{String,Any}("planner" => "estimate", "threads" => 1),
                             "checkpoint" => Dict{String,Any}("every_steps" => 5, "every_seconds" => 100.0,
                                                              "keep" => 3, "prefix" => "ck"))
        cfg = load_run_config(IOBuffer("""
            [fftw]
            planner = "estimate"
            threads = 1
            [checkpoint]
            every_steps = 5
            every_seconds = 100.0
            keep = 3
            prefix = "ck"
            """))
        fields(ck) = (ck.dir, ck.prefix, ck.every_steps, ck.every_seconds, ck.keep, ck.wisdom)
        dir = mktempdir()
        @test fields(checkpoint_schedule(cfg, dir)) == fields(checkpoint_schedule(d, dir))
        @test fields(checkpoint_schedule(cfg, dir)) == (dir, "ck", 5, 100.0, 3, true)
        set_fftw_policy!(; planner = :measure)
        pd = apply_fftw_config!(d)
        set_fftw_policy!(; planner = :measure)
        pc = apply_fftw_config!(cfg)
        @test pc == pd && pc.planner === :estimate && pc.threads == 1
        # defaults: an empty Dict and an empty RunConfig give the same schedule
        cd(mktempdir()) do
            empty = load_run_config(IOBuffer("[fftw]\nplanner = \"estimate\"\n"))
            @test fields(checkpoint_schedule(empty)) == fields(checkpoint_schedule(Dict{String,Any}()))
            @test fields(checkpoint_schedule(empty))[2:5] == ("checkpoint", 0, Inf, 2)
        end
        # the [fftw] planner of a config is applied; the Dict without one changes nothing
        set_fftw_policy!(; planner = :patient)
        @test apply_fftw_config!(Dict{String,Any}()).planner === :patient
        @test apply_fftw_config!(cfg).planner === :estimate
        # a wisdom file is imported when it exists
        wf = joinpath(mktempdir(), "wisdom")
        Aeolus.Operators.save_fftw_wisdom(wf)
        @test apply_fftw_config!(with(cfg, "fftw.wisdom" => wf)).wisdom_file == wf
    finally
        set_fftw_policy!(; planner = p0.planner, threads = p0.threads)
    end
end
