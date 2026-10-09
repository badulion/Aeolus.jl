# Config loader: one self-contained file, overrides, run.toml, inputs,
# common sections (CFG-a, #356; schema 2, ONE-b #451):
# src/config/{load,run_record,common}.jl.
#
# Acceptance tests covered here:
#   * a file plus 2 overrides gives the resolved table written out literally
#     below; file < override; an override table replaces;
#   * `extends` and schema_version 1 are refused with what changed (ONE-b:
#     every config is self-contained);
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

@testset "Config: one file + 2 overrides" begin
    dir = mktempdir()
    mkpath(joinpath(dir, "base"))
    child = joinpath(dir, "child.toml")
    write(child, """
        schema_version = 2
        [run]
        legacy = "cfgtest"
        name = "child"
        [grid]
        size = [64, 64, 32]
        stretch = 1.2
        levels = [1.0]
        [clock]
        nsteps = 200
        dt = 0.5
        scheme = "ab2"
        [physics]
        label = "base"
        tags = ["c"]
        terrain = "base/dem.tif"      # relative to this file
        [physics.couplings.wall]
        z0 = 0.5
        kind = "log"
        [physics.couplings.canopy]
        lai = 3.0
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
    @test t["run"]["name"] == "child" && t["run"]["legacy"] == "cfgtest"
    @test t["schema_version"] == 2 && !haskey(t, "domain")      # a legacy case has no domains
    # where each value came from
    @test config_origin(cfg, "clock.nsteps") == "--set"
    @test config_origin(cfg, "clock.scheme") == child
    @test config_origin(cfg, "clock.adaptive") == "default"
    @test config_sources(cfg)["files"] == [child]
    @test config_sources(cfg)["overrides"] == ["clock.nsteps=300", "grid.size=[16,16,8]"]
    @test load_run_config(child).clock.nsteps == 200              # file < override
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
    # an IOBuffer (base dir = pwd)
    @test load_run_config(IOBuffer("[run]\nlegacy = \"cfgtest\"\n[clock]\nnsteps = 5\n")).clock.nsteps == 5
end

@testset "Config: extends, schema_version 1 and [run] driver are refused (ONE-b)" begin
    dir = mktempdir()
    write(joinpath(dir, "base.toml"), "[run]\nlegacy = \"cfgtest\"\n[clock]\nnsteps = 42\n")
    p = joinpath(dir, "case.toml")
    write(p, "extends = \"base.toml\"\n[run]\nlegacy = \"cfgtest\"\n")
    e = cfgl_error(() -> load_run_config(p))
    @test only(e.problems).key == "extends" && only(e.problems).file == p
    @test occursin("self-contained", sprint(showerror, e)) && occursin("base.toml", sprint(showerror, e))
    # a schema-1 file: refused with what changed, whatever else it holds
    write(p, "schema_version = 1\n[run]\ndriver = \"cfgtest\"\n")
    m = sprint(showerror, cfgl_error(() -> load_run_config(p)))
    @test occursin("schema_version = 1 is no longer read", m) && occursin("[run] legacy", m)
    # [run] driver: an unknown key with the reason, also without a version
    for t in ("[run]\ndriver = \"cfgtest\"\n", "schema_version = 2\n[run]\ndriver = \"simulation\"\n")
        e = cfgl_error(() -> load_run_config(IOBuffer(t)))
        @test only(e.problems).key == "run.driver"
        @test occursin("removed in schema_version 2", only(e.problems).message)
    end
    # a newer version, a non-integer version
    @test occursin("newer", sprint(showerror, cfgl_error(() -> load_run_config(IOBuffer("schema_version = 3\n")))))
    @test_throws ConfigError load_run_config(IOBuffer("schema_version = \"2\"\n"))
    # an unknown legacy case, with a suggestion
    m = sprint(showerror, cfgl_error(() -> load_run_config(IOBuffer("[run]\nlegacy = \"cfgtets\"\n"))))
    @test occursin("unknown legacy case \"cfgtets\"", m) && occursin("did you mean \"cfgtest\"", m)
    # a missing configuration file
    @test_throws ConfigError load_run_config(joinpath(dir, "absent.toml"))
end

@testset "Config: run.toml stores the canonical config" begin
    dir = mktempdir()
    input = joinpath(dir, "terrain.dat"); write(input, "elevation 34.0\n")
    p = joinpath(dir, "case.toml")
    write(p, """
        [run]
        legacy = "cfgtest"
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
    @test rec["config_schema_version"] == 2
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
    write(p, "[run]\nlegacy = \"cfgtest\"\n[physics]\nterrain = \"terrain.dat\"\ninputs = [\"extra.nc\"]\n")
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
    cfg = with(load_run_config(IOBuffer("[run]\nlegacy = \"cfgtest\"\nname = \"jl\"\n")),
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
            empty = load_run_config(IOBuffer("[fftw]\nplanner = \"estimate\"\n"))     # the model
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
