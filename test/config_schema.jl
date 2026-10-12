# Config schema, strict validation and the canonical form (CFG-a, #356):
# src/Config.jl, src/config/{schema,resolve}.jl.
#
# Acceptance tests of the issue covered here:
#   * canonical fixed point: write_config(load(write_config(cfg))) is
#     byte-identical, for every shipped config and 200 seeded random
#     configs per test schema;
#   * float exactness: 0.1, 1/3, 1e-4, nextfloat(1.0), 5e-324, floatmax(),
#     ±Inf, NaN and -0.0 survive TOML text → RunConfig → TOML text with
#     identical bits;
#   * errors: unknown key, wrong type, out of range, outside `allowed`,
#     tuple of the wrong length each throw a ConfigError whose message has
#     the dotted key, the file and the range / allowed values; a file with
#     5 independent errors reports all 5.

using Test
using TOML
using Dates: Date, Time
using Random
using Aeolus
using Aeolus.Config
using Aeolus.Config: ConfigProblem

include(joinpath(@__DIR__, "support", "config_test_schema.jl"))

"Write `text` to a fresh file `name` in a temporary directory; return its path."
function cfg_file(text; name = "case.toml", dir = mktempdir())
    p = joinpath(dir, name)
    write(p, text)
    return p
end

"The ConfigError `f()` throws (fails the test if it does not throw one)."
function cfg_error(f)
    try
        f()
    catch e
        e isa ConfigError && return e
        rethrow()
    end
    error("expected a ConfigError")
end

@testset "Config: schema construction" begin
    s = section("demo") do s
        option!(s, "a", Int; default = 1, range = (0, 10))
    end
    @test s.name == "demo" && length(s.options) == 1
    @test_throws ArgumentError section("Bad Name") do s end
    @test_throws ArgumentError section("demo") do s
        option!(s, "CamelCase", Int; default = 1)
    end
    @test_throws ArgumentError section("demo") do s          # no default, not required
        option!(s, "a", Int)
    end
    @test_throws ArgumentError section("demo") do s          # default outside range
        option!(s, "a", Int; default = 11, range = (0, 10))
    end
    @test_throws ArgumentError section("demo") do s          # unsupported type
        option!(s, "a", Dict{String,Int}; default = Dict{String,Int}())
    end
    @test_throws ArgumentError section("demo") do s          # declared twice
        option!(s, "a", Int; default = 1); option!(s, "a", Int; default = 2)
    end
    @test_throws ArgumentError section("demo") do s          # required with a default
        option!(s, "a", Int; default = 1, required = true)
    end
    # Optional defaults to nothing, OpenTable to an empty table
    s2 = section("demo") do s
        option!(s, "o", Optional{Int}); option!(s, "t", OpenTable())
    end
    @test s2.options[1].default === nothing && s2.options[2].default == Dict{String,Any}()
    # a driver section named like a common one extends it (CFG-b, #357),
    # but may not redeclare a common option; a driver section twice is an error
    sx = ConfigSchema("x"; sections = [section("output") do s
        option!(s, "fields", Vector{String}; default = String[])
    end])
    @test [o.key for o in only(filter(s -> s.name == "output", sx.sections)).options] ==
          ["path", "cadence", "fields"]
    @test count(s -> s.name == "output", sx.sections) == 1
    @test_throws ArgumentError ConfigSchema("x"; sections = [section("run") do s
        option!(s, "seed", Int; default = 1)
    end])
    @test_throws ArgumentError ConfigSchema("x"; sections = [section("a") do s end,
                                                             section("a") do s end])
    @test "cfgtest" in registered_drivers() && "common" in registered_drivers()
end

@testset "Config: every option type from TOML" begin
    dir = mktempdir()
    p = cfg_file("""
        [run]
        legacy = "cfgtest"
        [grid]
        size = [32, 16, 8]
        extent = [1000, 500.5, 250.0]      # an integer is taken as a float
        levels = [0.0, 10.0, 25.0]
        [clock]
        start_date = 2023-01-15
        start_time = 07:32:00.5
        scheme = "ab2"
        dt_max = "none"
        [physics]
        terrain = "../data/dem.tif"
        inputs = ["a.nc", "/abs/b.nc"]
        tags = ["a", "c"]
        [physics.couplings.wall]
        z0 = 0.1
        """; dir = joinpath(dir, "cases") |> mkpath)
    cfg = load_run_config(p)
    @test cfg.grid.size === (32, 16, 8)
    @test cfg.grid.extent === (1000.0, 500.5, 250.0)
    @test cfg.grid.levels == [0.0, 10.0, 25.0] && cfg.grid.levels isa Vector{Float64}
    @test cfg.clock.start_date == Date(2023, 1, 15)
    @test cfg.clock.start_time == Time(7, 32, 0, 500)
    @test cfg.clock.scheme === :ab2
    @test cfg.clock.dt_max === nothing
    @test cfg.clock.nsteps == 10                      # default
    # a path is resolved against the directory of the file that set it
    @test cfg.physics.terrain == normpath(joinpath(dir, "data", "dem.tif"))
    @test cfg.physics.inputs == [joinpath(dir, "cases", "a.nc"), "/abs/b.nc"]
    @test cfg.physics.tags == [:a, :c]
    @test cfg.physics.couplings == Dict("wall" => Dict("z0" => 0.1))
    @test cfg["grid.size"] === (32, 16, 8)
    @test config_origin(cfg, "grid.size") == p
    @test config_origin(cfg, "clock.nsteps") == "default"
    # canonical text: every option, sorted keys, "none", absolute paths
    t = write_config(cfg)
    @test startswith(t, "schema_version = 2\n\n[checkpoint]\n")
    @test occursin("\n[clock]\nadaptive = true\ndt = 0.5\ndt_max = \"none\"\nnsteps = 10\n", t)
    @test occursin("extent = [1000.0, 500.5, 250.0]\n", t)
    @test occursin("start_time = 07:32:00.5\n", t)
    @test occursin("terrain = \"$(normpath(joinpath(dir, "data", "dem.tif")))\"", t)
    @test occursin("\n[physics.couplings.wall]\nz0 = 0.1\n", t)
    @test config_hash(cfg) == Aeolus.IO.fnv1a64(t)
end

@testset "Config: OneOf, a value of either type (CFG-d)" begin
    pos(t) = haskey(t, "z") && t["z"] isa AbstractVector ? String[] : ["z: expected an array of heights"]
    register_driver!("cfgtest_oneof"; sections = [section("forcing") do s
        option!(s, "u_g", OneOf(Float64, OpenTable(pos)); default = 8.0, doc = "a number or a table")
        option!(s, "nu", OneOf(Float64, Symbol); default = :scaled, allowed = [:scaled], range = (0, Inf),
                open_lower = true, doc = "a number or \"scaled\"")
    end])
    load(text) = load_run_config(cfg_file("[run]\nlegacy = \"cfgtest_oneof\"\n[forcing]\n" * text))
    @test load("").forcing.u_g === 8.0 && load("").forcing.nu === :scaled
    @test load("u_g = 3").forcing.u_g === 3.0                     # an integer is a float
    cfg = load("u_g = {z = [0.0, 10.0]}\nnu = 1e-4")
    @test cfg.forcing.u_g == Dict("z" => [0.0, 10.0]) && cfg.forcing.nu === 1e-4
    @test load_run_config(cfg_file(write_config(cfg))) == cfg     # canonical round trip
    e = cfg_error(() -> load("u_g = {zz = 1}"))                   # the table's own reason
    @test occursin("forcing.u_g", sprint(showerror, e)) && occursin("z: expected an array of heights", sprint(showerror, e))
    e = cfg_error(() -> load("nu = \"fast\""))                     # allowed applies to the keyword
    @test occursin("a float or a string", sprint(showerror, e)) && occursin("\"scaled\"", sprint(showerror, e))
    @test occursin("(0, Inf]", sprint(showerror, cfg_error(() -> load("nu = -1.0"))))   # range to the number
    @test_throws ArgumentError section("x") do s                    # OneOf needs a default
        option!(s, "a", OneOf(Float64, Symbol))
    end
    @test_throws ArgumentError OneOf(Float64)
    @test occursin("float | table", option_reference("cfgtest_oneof"))
end

"Every shipped config: examples/**/configs/*.toml and a case directory's examples/<case>/*.toml (ONE-e, #454)."
function cfg_shipped()
    ex = normpath(joinpath(@__DIR__, "..", "examples"))
    return sort!([joinpath(r, f) for (r, _, fs) in walkdir(ex) for f in fs
                  if endswith(f, ".toml") && (basename(r) == "configs" || dirname(r) == rstrip(ex, '/'))])
end

@testset "Config: canonical fixed point (shipped + 200 random configs per schema)" begin
    # shipped configs: every examples/**/configs/*.toml and examples/urban/*.toml, every member of a sweep
    shipped = cfg_shipped()
    # 44 since ONE-g (#456): 15 column, channel and a-priori configs dropped, 6 DNS-vs-LES added
    @test length(shipped) >= 40
    for f in shipped
        runs = cd(() -> load_sweep(f), dirname(f))
        for (_, cfg) in runs
            t1 = write_config(cfg)
            t2 = write_config(load_run_config(IOBuffer(t1)))
            @test t1 == t2
        end
    end
    nfail = 0; nnan = 0
    for name in ("cfgtest", "common")
        schema = Aeolus.Config.driver_schema(name)
        rng = Xoshiro(356 + length(name))
        dir = mktempdir()
        for i in 1:200
            raw = cfgt_random_raw(rng, schema)
            p = joinpath(dir, "r$i.toml")
            open(io -> TOML.print(io, raw), p, "w")
            cfg = load_run_config(p)
            t1 = write_config(cfg)
            q = write_config(joinpath(dir, "w$i.toml"), cfg)
            cfg2 = load_run_config(q)
            t2 = write_config(cfg2)
            ok = t1 == t2 && cfg2 == cfg && config_hash(cfg2) == config_hash(cfg)
            ok || (nfail += 1)
            occursin("nan", t1) && (nnan += 1)
        end
    end
    @test nfail == 0
    @test nnan > 0          # the generator does reach the NaN path
end

@testset "Config: float exactness through TOML text" begin
    vals = [(0.1, "0.1"), (1 / 3, "0.3333333333333333"), (1e-4, "1e-4"),
            (nextfloat(1.0), "1.0000000000000002"), (5e-324, "5e-324"),
            (floatmax(), "1.7976931348623157e308"), (Inf, "inf"), (-Inf, "-inf"),
            (NaN, "nan"), (-0.0, "-0.0")]
    for (x, lit) in vals
        p = cfg_file("[run]\nlegacy = \"cfgtest\"\n[clock]\ndt = $lit\n")
        cfg = load_run_config(p)
        @test reinterpret(UInt64, cfg.clock.dt) == reinterpret(UInt64, x)
        cfg2 = load_run_config(IOBuffer(write_config(cfg)))
        @test reinterpret(UInt64, cfg2.clock.dt) == reinterpret(UInt64, x)
        # and from Julia through with()
        cfg3 = load_run_config(IOBuffer(write_config(with(cfg, "clock.dt" => x))))
        @test reinterpret(UInt64, cfg3.clock.dt) == reinterpret(UInt64, x)
        @test write_config(cfg3) == write_config(cfg)
    end
end

@testset "Config: strict errors name key, file and constraint" begin
    base = "[run]\nlegacy = \"cfgtest\"\n"
    # unknown key, with a suggestion (Damerau–Levenshtein ≤ 2)
    p = cfg_file(base * "[grid]\nsise = [8, 8, 8]\n")
    e = cfg_error(() -> load_run_config(p))
    m = sprint(showerror, e)
    @test length(e.problems) == 1
    @test occursin("grid.sise", m) && occursin(p, m) && occursin("did you mean \"size\"", m)
    # unknown table
    e = cfg_error(() -> load_run_config(cfg_file(base * "[clok]\nnsteps = 3\n")))
    @test occursin("did you mean \"clock\"", sprint(showerror, e))
    # wrong type
    p = cfg_file(base * "[clock]\nnsteps = \"ten\"\n")
    m = sprint(showerror, cfg_error(() -> load_run_config(p)))
    @test occursin("clock.nsteps", m) && occursin(p, m) && occursin("an integer", m) &&
          occursin("\"ten\"", m) && occursin("[0, $(typemax(Int))]", m)
    # a float where an integer is expected is not truncated
    @test_throws ConfigError load_run_config(cfg_file(base * "[clock]\nnsteps = 3.0\n"))
    # out of range (closed and open bounds)
    p = cfg_file(base * "[grid]\nstretch = 3.0\n")
    m = sprint(showerror, cfg_error(() -> load_run_config(p)))
    @test occursin("grid.stretch", m) && occursin(p, m) && occursin("[1.0, 2.0]", m) && occursin("3.0", m)
    m = sprint(showerror, cfg_error(() -> load_run_config(cfg_file(base * "[grid]\nextent = [1.0, 0.0, 1.0]\n"))))
    @test occursin("(0, Inf]", m)
    # outside allowed
    p = cfg_file(base * "[clock]\nscheme = \"rk4\"\n")
    m = sprint(showerror, cfg_error(() -> load_run_config(p)))
    @test occursin("clock.scheme", m) && occursin(p, m) &&
          occursin("allowed: \"rk3\", \"ab2\", \"imex\"", m) && occursin("\"rk4\"", m)
    # elementwise allowed in a vector
    @test_throws ConfigError load_run_config(cfg_file(base * "[physics]\ntags = [\"a\", \"z\"]\n"))
    # tuple of the wrong length
    p = cfg_file(base * "[grid]\nsize = [16, 16]\n")
    m = sprint(showerror, cfg_error(() -> load_run_config(p)))
    @test occursin("grid.size", m) && occursin(p, m) && occursin("an array of 3 integers", m) &&
          occursin("2 elements", m)
    # an open table's validator
    m = sprint(showerror, cfg_error(() -> load_run_config(cfg_file(base * "[physics.couplings]\nwall = 3\n"))))
    @test occursin("physics.couplings", m) && occursin("wall: expected a table", m)
    # cross-field check
    p = cfg_file(base * "[clock]\nscheme = \"imex\"\n")
    m = sprint(showerror, cfg_error(() -> load_run_config(p)))
    @test occursin("clock.adaptive", m) && occursin("needs adaptive = false", m)
    # a newer schema_version is refused, naming the supported one
    m = sprint(showerror, cfg_error(() -> load_run_config(cfg_file("schema_version = 3\n" * base))))
    @test occursin("schema_version = 3", m) && occursin("schema_version = 2", m)
    # an unknown legacy case, with a suggestion
    m = sprint(showerror, cfg_error(() -> load_run_config(cfg_file("[run]\nlegacy = \"cfgtst\"\n"))))
    @test occursin("unknown legacy case \"cfgtst\"", m) && occursin("did you mean \"cfgtest\"", m)
    # five independent errors are reported together
    p = cfg_file(base * """
        [grid]
        sise = [8, 8, 8]
        stretch = 3.0
        size = [16, 16]
        [clock]
        nsteps = "ten"
        scheme = "rk4"
        """)
    e = cfg_error(() -> load_run_config(p))
    @test length(e.problems) == 5
    @test Set(q.key for q in e.problems) ==
          Set(["grid.sise", "grid.stretch", "grid.size", "clock.nsteps", "clock.scheme"])
    @test all(q -> q.file == p, e.problems)
    m = sprint(showerror, e)
    @test startswith(m, "ConfigError: 5 problems")
    # invalid TOML
    m = sprint(showerror, cfg_error(() -> load_run_config(cfg_file("[grid\nsize = 1\n"))))
    @test occursin("not valid TOML", m)
end

@testset "Config: Damerau–Levenshtein" begin
    @test damerau_levenshtein("size", "size") == 0
    @test damerau_levenshtein("sise", "size") == 1          # substitution
    @test damerau_levenshtein("szie", "size") == 1          # transposition
    @test damerau_levenshtein("nstep", "nsteps") == 1       # insertion
    @test damerau_levenshtein("kitten", "sitting") == 3
    @test damerau_levenshtein("", "abc") == 3
    @test damerau_levenshtein("ca", "abc") == 3             # OSA distance (not 2)
    @test Aeolus.Config._suggest("planer", ["planner", "threads"]) == "; did you mean \"planner\"?"
    @test Aeolus.Config._suggest("xyzzy", ["planner", "threads"]) == ""
end

@testset "Config: with, the Julia escape hatch" begin
    cfg = load_run_config(cfg_file("[run]\nlegacy = \"cfgtest\"\n"))
    c2 = with(cfg, "clock.nsteps" => 12, "grid.size" => [8, 8, 4], "clock.scheme" => :ab2,
              "clock.dt_max" => 2, "physics.terrain" => "dem.tif")
    @test c2.clock.nsteps == 12 && cfg.clock.nsteps == 10          # cfg is unchanged
    @test c2.grid.size === (8, 8, 4)
    @test c2.clock.dt_max === 2.0
    @test c2.physics.terrain == joinpath(pwd(), "dem.tif")
    @test config_origin(c2, "clock.nsteps") == "with()"
    @test config_sources(c2)["with"] == ["clock.nsteps", "grid.size", "clock.scheme",
                                         "clock.dt_max", "physics.terrain"]
    @test with(c2, "clock.dt_max" => nothing).clock.dt_max === nothing
    @test c2 != cfg && with(c2, "clock.nsteps" => 10, "grid.size" => (16, 16, 16),
                            "clock.scheme" => :rk3, "clock.dt_max" => nothing,
                            "physics.terrain" => nothing) == cfg
    # every problem at once, cross-field checks included
    e = cfg_error(() -> with(cfg, "clock.nsteps" => -1, "grid.sise" => 1, "clok.dt" => 1.0))
    @test length(e.problems) == 3
    @test occursin("did you mean \"size\"", sprint(showerror, e))
    @test_throws ConfigError with(cfg, "clock.scheme" => :imex)            # adaptive = true
    @test with(cfg, "clock.scheme" => :imex, "clock.adaptive" => false).clock.scheme === :imex
    @test_throws ConfigError with(cfg, "clock.nsteps" => true)             # Bool is not Int
    # a closure: only where allow_julia = true, and the config is not replayable
    @test_throws ConfigError with(cfg, "clock.dt" => z -> z)
    c3 = with(cfg, "physics.profile" => z -> 2z)
    @test c3.physics.profile(2.0) == 4.0
    @test nonreplayable_keys(c3) == ["physics.profile"] && isempty(nonreplayable_keys(cfg))
    @test occursin("profile = \"<julia: ", write_config(c3))
    # a property that does not exist suggests the right one
    @test_throws ArgumentError cfg.grid.sise
    @test_throws ArgumentError cfg.gird
end

@testset "Config: environment variables only fill defaults" begin
    withenv("AEOLUS_FFTW_PLANNER" => "patient", "AEOLUS_FFTW_THREADS" => "3",
            "AEOLUS_LOG" => "debug", "AEOLUS_ALLOW_UNCOMMITTED" => "1") do
        cfg = load_run_config(cfg_file("[run]\nname = \"e\"\n"))
        @test cfg.fftw.planner === :patient && cfg.fftw.threads == 3
        @test cfg.logging.console_level === :debug && cfg.run.allow_uncommitted
        t = write_config(cfg)          # the resolved value is written
        @test occursin("planner = \"patient\"", t) && occursin("threads = 3", t)
        # a value in the file wins over the environment
        cfg2 = load_run_config(cfg_file("[fftw]\nplanner = \"estimate\"\n"))
        @test cfg2.fftw.planner === :estimate
    end
    withenv("AEOLUS_FFTW_PLANNER" => nothing, "AEOLUS_FFTW_THREADS" => nothing,
            "AEOLUS_LOG" => nothing, "AEOLUS_ALLOW_UNCOMMITTED" => nothing) do
        cfg = load_run_config(cfg_file("[run]\nname = \"e\"\n"))
        @test cfg.fftw.planner === :measure && cfg.fftw.threads === nothing
        @test cfg.logging.console_level === :info && !cfg.run.allow_uncommitted
    end
    withenv("AEOLUS_FFTW_THREADS" => "zero") do       # a bad variable is a config error
        @test_throws ConfigError load_run_config(cfg_file("[run]\nname = \"e\"\n"))
    end
end

@testset "Config: a name or a number (OneOf) and a driver's defaults of common options (CFG-c, #358)" begin
    # OneOf(Symbol, Float64): a name from `allowed` or a number in `range` (CFG-c's Either{A,B},
    # merged into CFG-d's OneOf so that one option type means "either")
    register_driver!("eithertest"; sections = [section("top") do s
        option!(s, "theta", OneOf(Symbol, Float64); default = :initial_gradient,
                allowed = [:initial_gradient, :noflux], range = (-1, 1))
    end], defaults = ("output.cadence" => 1800.0, "run.name" => "alexanderplatz"))
    ld(t) = load_run_config(cfg_file("[run]\nlegacy = \"eithertest\"\n" * t))
    @test ld("").top.theta === :initial_gradient
    @test ld("[top]\ntheta = \"noflux\"\n").top.theta === :noflux
    @test ld("[top]\ntheta = 0.003\n").top.theta === 0.003
    @test ld("[top]\ntheta = 0\n").top.theta === 0.0               # an integer is a float
    e = cfg_error(() -> ld("[top]\ntheta = \"lid\"\n"))
    @test occursin("a string or a float", e.problems[1].message) && occursin("noflux", e.problems[1].message)
    e = cfg_error(() -> ld("[top]\ntheta = 2.0\n"))                 # outside the range
    @test only(e.problems).key == "top.theta"
    c = ld("[top]\ntheta = -0.5\n")                                 # canonical text round-trips
    @test load_run_config(IOBuffer(write_config(c))).top.theta === -0.5
    # the driver's own defaults replace the common ones; type and range stay
    cfg = ld("")
    @test cfg.output.cadence == 1800.0 && cfg.run.name == "alexanderplatz"
    @test load_run_config(cfg_file("[run]\nname = \"e\"\n")).output.cadence == 3600.0   # the model's: unchanged
    @test_throws ArgumentError ConfigSchema("x"; defaults = ("output.cadence" => -1.0))
    @test_throws ArgumentError ConfigSchema("x"; defaults = ("output.nope" => 1.0))
end
