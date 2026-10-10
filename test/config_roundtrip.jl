# The round-trip coverage suite (CFG-i, #364): every shipped config stays
# valid, canonical, documented and replayable bit for bit.
#
# Every `examples/**/configs/*.toml` (and so `examples/configs/*.toml`) and
# every config of a case directory, `examples/<case>/*.toml` (the Berlin
# case, examples/urban/berlin_*.toml, ONE-e #454), is found by globbing the
# tree, not from a list, and each one must carry its own smoke size, a
# `[test]` table:
#
#   [test]
#   overrides = ["grid.size=[16,16,16]", "time.nsteps=10"]
#   member = "lasd_32x32x32"          # a sweep: the member replayed (default: the first)
#
# For each file:
#   1. it validates (every member of a sweep), with and without its overrides;
#   2. `resolve` is a fixed point: the canonical text of the resolved config
#      is byte-identical after a reload;
#   3. every key it holds appears in its schema's generated option reference
#      (the Markdown the docs build writes), the Berlin case's component
#      sections and the nest parent's table included;
#   4. it replays: run at its smoke size, then re-run from the first run's
#      run.toml; the state fields, diagnostics.csv, results.toml and
#      [config] are bit-identical under ESTIMATE FFTW (roundtrip_check).
# A configs/*.toml without a [test] table fails the suite with a message
# naming the file. Resume: for the model and the sbl and urban legacy cases, a run
# interrupted after its middle checkpoint and resumed the way
# `aeolus run <folder> --resume` does is bitwise the uninterrupted run
# (the Berlin case's since #269).

using Test
using Aeolus
using Aeolus.Config
import Aeolus.Config as C
import NCDatasets

include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const RT_EXAMPLES = normpath(joinpath(@__DIR__, "..", "examples"))

"""
Every shipped config: the `.toml` files directly in a directory named
`configs` anywhere under `root` (`examples/configs/`, `examples/sbl/configs/`, …),
and those directly in a case directory `root/<case>/` (`examples/urban/`,
ONE-e #454).
"""
function shipped_config_files(root::AbstractString = RT_EXAMPLES)
    out = String[]
    for (dir, _, files) in walkdir(root)
        (basename(dir) == "configs" || dirname(dir) == rstrip(root, '/')) || continue
        append!(out, joinpath(dir, f) for f in files if endswith(f, ".toml"))
    end
    return sort!(out)
end

"""
The smoke plan of every file: `file => (overrides, member)`; an error naming
every file without a `[test]` table (a new config must say how it is tested).
"""
function smoke_plan(files)
    missing = [f for f in files if config_test_table(f) === nothing]
    isempty(missing) || error("these shipped configs have no [test] table with their smoke-size " *
                              "overrides (see test/config_roundtrip.jl):\n  " * join(missing, "\n  "))
    return [f => config_test_table(f) for f in files]
end

"Run `f()` in the directory of `file`: a relative path in its [test] overrides is relative to the file."
in_dir(f, file) = cd(f, dirname(file))

# the pages the docs build writes (docs/make.jl calls the same generate_reference)
include(joinpath(@__DIR__, "..", "docs", "generate_reference.jl"))
const RT_REFERENCE = mktempdir()
const RT_PAGES = generate_reference(RT_REFERENCE)

"The `[section]` keys of the option reference page the docs build writes for the schema `driver`, per table."
function reference_rows(driver)
    rows = Dict{String,Set{String}}(); cur = nothing
    page = joinpath(RT_REFERENCE, "options", driver * ".md")
    isfile(page) || error("the docs build writes no option reference for the schema \"$driver\"")
    for l in eachline(page)
        m = match(r"^#+ `\[(.+)\]`$", l)
        if m !== nothing
            cur = String(m[1]); rows[cur] = Set{String}()
        elseif cur !== nothing && startswith(l, "| `")
            push!(rows[cur], String(match(r"^\| `([^`]+)` \|", l)[1]))
        end
    end
    return rows
end

const RT_FILES = shipped_config_files()

@testset "round-trip coverage: the docs build writes a reference page per schema and the input pages" begin
    @test Set(basename.(RT_PAGES)) ==
          Set(vcat([d * ".md" for d in reference_drivers()], "input_static.md", "input_dynamic.md"))
    @test issubset(["model", "urban_parent", "sbl", "dns", "column"], reference_drivers())
    @test !("urban" in reference_drivers())             # the urban site is the model's (ONE-e, #454)
    # every option page is the schema's generated Markdown, verbatim after its header
    for d in reference_drivers()
        @test endswith(read(joinpath(RT_REFERENCE, "options", d * ".md"), String),
                       option_reference(d; format = :markdown))
    end
    for (f, s) in (("input_static.md", Aeolus.InputDrivers.STATIC_SCHEMA),
                   ("input_static.md", Aeolus.InputDrivers.STATIC_TILES_SCHEMA),
                   ("input_dynamic.md", Aeolus.InputDrivers.DYNAMIC_SCHEMA))
        txt = read(joinpath(RT_REFERENCE, f), String)
        @test all(v -> occursin("| `$(v.name)` |", txt), s.variables)
        @test occursin("netcdf ", txt)                       # the ncdump -h example
    end
end

@testset "round-trip coverage: every shipped config is found and has a [test] table" begin
    @test length(RT_FILES) >= 25
    @test all(f -> startswith(f, RT_EXAMPLES), RT_FILES)
    @test joinpath(RT_EXAMPLES, "configs", "taylor_green_vortex.toml") in RT_FILES
    plan = smoke_plan(RT_FILES)
    @test length(plan) == length(RT_FILES)
    # a config without [test] fails, naming the file
    dir = mkpath(joinpath(mktempdir(), "examples", "new", "configs"))
    write(joinpath(dir, "fresh.toml"), "[run]\nname = \"fresh\"\n")
    write(joinpath(dir, "tested.toml"), "[run]\nname = \"fresh\"\n\n[test]\noverrides = []\n")
    files = shipped_config_files(dirname(dirname(dir)))
    @test length(files) == 2
    e = try
        smoke_plan(files); nothing
    catch err
        err
    end
    @test e isa ErrorException && occursin(joinpath(dir, "fresh.toml"), e.msg) && !occursin("tested.toml", e.msg)
    # [test] is checked for shape and stays out of the resolved config
    write(joinpath(dir, "bad.toml"), "[run]\nname = \"fresh\"\n\n[test]\noverride = []\nmember = 0\n")
    e = try
        load_run_config(joinpath(dir, "bad.toml")); nothing
    catch err
        err
    end
    @test e isa ConfigError && Set(p.key for p in e.problems) == Set(["test.override", "test.member"])
    cfg = load_run_config(joinpath(dir, "tested.toml"))
    @test !occursin("[test]", config_text(cfg))
    @test config_hash(cfg) == config_hash(load_run_config(IOBuffer("[run]\nname = \"fresh\"\n")))
end

const RT_PLAN = smoke_plan(RT_FILES)

@testset "round-trip coverage: $(relpath(f, RT_EXAMPLES))" for (f, t) in RT_PLAN
    # 1. it validates: every member as written, and with its smoke overrides
    runs = load_sweep(f)
    @test !isempty(runs)
    smoke = in_dir(() -> load_sweep(f; overrides = t.overrides), f)
    @test length(smoke) == length(runs)
    sweep = !isempty(first(runs).first.label)
    member = sweep ? something(t.member, 1) : nothing
    sweep || @test t.member === nothing                   # a member only for a sweep
    rows = reference_rows(config_schema(first(runs).second).name)
    for (_, cfg) in runs
        # 2. resolve is a fixed point: the canonical text, byte for byte
        t1 = config_text(cfg)
        @test config_text(load_run_config(IOBuffer(t1))) == t1
        # 3. every key it holds is in the option reference the docs build
        sch = config_schema(cfg)
        for s in sch.sections, k in keys(getfield(getproperty(cfg, Symbol(s.name)), :values))
            @test k in rows[s.name]
        end
        @test isempty(undocumented_keys(cfg))
    end
    # 4. it replays from its run.toml, bit for bit
    r = in_dir(() -> roundtrip_check(f; overrides = t.overrides, member = member), f)
    @test r.config_equal
    @test r.fields_equal
    isempty(r.fields_differing) || @info "round trip differs" f r.fields_differing
    @test r.diagnostics_equal
    @test r.results_equal
    # the replay is of a run that ran: some state, finite
    fs = Base.invokelatest(state_fields, r.runs[1].obj)
    @test !isempty(keys(fs))
    @test all(k -> all(isfinite, fs[k]), keys(fs))
end

@testset "resume: interrupted after the middle checkpoint, resumed, bitwise the uninterrupted run" begin
    # the model: buoyant_convection at 16³, 20 steps (dry_convective_bl and the
    # adaptive TGV resume in config_simulation_driver.jl)
    r = resume_check(joinpath(RT_EXAMPLES, "configs", "buoyant_convection.toml");
                     overrides = ["grid.size=[16,16,16]", "output.series_cadence=0.0", "output.cadence=0.2"],
                     outputs = ["output.nc"])
    @test r.fields_equal && isempty(r.fields_differing)
    # the one writer's file (ONE-c): the resumed run continues it after the checkpoint,
    # over what the interrupted session wrote after it, to the same records and series
    @test r.outputs_equal
    @test r.statuses == ["interrupted", "completed"]
    @test r.runs[2].obj.stepper.step == 20 && Aeolus.SimulationDriver.simulation(r.runs[2].obj).t === Aeolus.SimulationDriver.simulation(r.runs[1].obj).t
    # sbl legacy case: cases99 (GABLS2 with towers and series) at 16 × 16 × 32, 20 steps; the
    # towers and the series are restored from the checkpoint too
    r = resume_check(joinpath(RT_EXAMPLES, "sbl", "configs", "cases99.toml");
                     overrides = ["grid.size=[16,16,32]", "towers.hours=[0.0,24.0]", "towers.window=30.0",
                                  "averaging.series_interval=60.0"], nsteps_overrides = sbl_nsteps,
                     outputs = ["towers.csv", "output.nc", "statistics.toml"])
    @test r.fields_equal && isempty(r.fields_differing)
    @test r.outputs_equal
    @test r.statuses == ["interrupted", "completed"]
    @test r.runs[2].obj.case.nsteps == 20 && r.runs[2].obj.case.sim.t === r.runs[1].obj.case.sim.t
    # the urban site (#269, ONE-e #454): berlin_smoke.toml, nested, every component on, 24 s (15 adaptive
    # steps), a checkpoint every 6 steps, interrupted after step 9 and resumed from step 6;
    # the state of the child, every tracer, Ts and the nest parent bitwise, and the NetCDF
    # output's record written after the checkpoint is rewritten to the same values
    r = resume_check(joinpath(RT_EXAMPLES, "urban", "berlin_smoke.toml");
                     nsteps = 12, stop = 9, every_steps = 6, nsteps_overrides = n -> ["time.nsteps=$n"],
                     overrides = ["urban.netcdf=\"smoke.nc\"", "output.path=\"none\""])
    @test r.fields_equal && isempty(r.fields_differing)
    isempty(r.fields_differing) || @info "Berlin resume: fields differing" r.fields_differing
    @test r.statuses == ["interrupted", "completed"]
    c1, c2 = r.runs[1].obj.case, r.runs[2].obj.case
    @test c2.clock.n == c1.clock.n && c2.sim.t === c1.sim.t === 24.0 && c2.clock.nrecord == c1.clock.nrecord == 2
    @test any(k -> startswith(String(k), "parent_"), keys(Base.invokelatest(state_fields, r.runs[2].obj)))
    nc(case) = NCDatasets.NCDataset(case.cfg.output_path) do ds
        Dict(k => Array(ds[k].var) for k in keys(ds))
    end
    @test c1.cfg.output_path != c2.cfg.output_path && isequal(nc(c1), nc(c2))
end
