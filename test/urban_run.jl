# The urban site of the model (EX-a #438, ONE-e #454): Aeolus.UrbanCase,
# a model config with `[urban] enabled`, and the `urban_parent` run driver.
# Small synthetic setups:
#
#   * a synthetic city (test/data/urban/small_city.toml: two buildings, a
#     park and a pond from GeoJSON, 16³, periodic, local noon) runs through
#     `aeolus run`: the run folder's results.toml holds the run type's own
#     checks (the ledgers closed, a finite state, the sun's net gain), and
#     the ground is the GeoJSON's its static driver holds (heights,
#     classes, the shed below the minimum height dropped);
#   * the nest parent alone (`urban_parent`, berlin_parent.toml at its
#     smoke size): its tables, results and tiled-surface file.
# The input preprocessors (osm-sample, the IGRA2 parser, the parent's
# land-use table) are AeolusPrep.jl's since ONE-h (#457), with their tests.
#   * the examples policy: no script under examples/urban builds or runs the
#     model (only TOML configs, input data and analysis of output files).

using Test
using TOML
using Aeolus
using Aeolus.Config
using Aeolus.Fields: interior_view
using Aeolus.IO: read_run_toml

const UC = Aeolus.UrbanCase
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))
include(joinpath(@__DIR__, "support", "berlin_fixture.jl"))     # digest, berlin_state

const SMALL_CITY = joinpath(@__DIR__, "data", "urban", "small_city.toml")

"Run `args` through the aeolus command line quietly; the exit code."
quiet_main(args) = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        Aeolus.main(args)
    end
end

@testset "urban run type: the synthetic city through aeolus run" begin
    root = mktempdir()
    rc = quiet_main(["run", SMALL_CITY, "--root", root, "--allow-uncommitted", "--set", "fftw.planner=\"estimate\""])
    @test rc == 0
    run = only(readdir(root; join = true))
    rec = read_run_toml(run)
    @test rec["session"][end]["status"] == "completed" && !haskey(rec["config"]["run"], "legacy")
    res = TOML.parsefile(joinpath(run, "output", "results.toml"))
    led = res["budgets"]
    @test res["finite"] === true
    @test led["closed"] === true
    @test led["surface_max_abs_closure"] < UC.BUDGET_TOLERANCES.surface
    @test abs(led["water_residual"]) < UC.BUDGET_TOLERANCES.water
    @test abs(led["air_heat_closure"]) <= UC.BUDGET_TOLERANCES.air_heat
    # local noon in June: the ground gains net radiation, the roofs and the
    # street are in the ledger, the pond and the park too
    dom = led["surface"]["domain"]
    @test dom["Rn"] > 100
    @test all(haskey(led["surface"], c) for c in ("street", "roof", "park", "water"))
    # 6 × 2 s of simulated time (the adaptive stepper picks its own steps)
    @test res["timing"]["simulated_s"] ≈ 12.0 && res["timing"]["steps"] >= 1
    # the idealised profile: 293 K at the ground below the 60 m inversion
    @test 292.9 < res["theta_mean_below_100m_start"] < 293.5
    @test isfile(joinpath(run, "diagnostics.csv")) || isfile(joinpath(run, "output", "diagnostics.csv"))
end

@testset "urban site: its checks through [checks] (ONE-d) in results.toml" begin
    root = mktempdir()
    chk = ["--set", "checks.urban_budgets={required = true}", "--set", "checks.surface_exchange_checks={required = true}",
           "--set", "checks.energy_closure={required = true, tolerance = 1e-9}", "--set", "checks.finite={required = true}"]
    rc = quiet_main(vcat(["run", SMALL_CITY, "--root", root, "--allow-uncommitted", "--set", "fftw.planner=\"estimate\""], chk))
    @test rc == 0
    res = TOML.parsefile(joinpath(only(readdir(root; join = true)), "output", "results.toml"))
    @test res["passed"] === true && isempty(res["failed_required"])
    c = res["checks"]
    @test Set(keys(c)) == Set(["urban_budgets", "surface_exchange_checks", "energy_closure", "finite"])
    @test c["urban_budgets"]["passed"] && c["urban_budgets"]["value"] <= 1.0
    @test c["surface_exchange_checks"]["value"] == 0 && c["surface_exchange_checks"]["checks"] > 0
    @test c["energy_closure"]["value"] < 1e-9 && c["finite"]["value"] == 0
    # the same figures as the run's own verdict
    @test c["urban_budgets"]["air_heat_closure"] == res["budgets"]["air_heat_closure"]
    # a required check that fails ends the run with exit code 3, its results written
    root = mktempdir()
    rc = quiet_main(["run", SMALL_CITY, "--root", root, "--allow-uncommitted", "--set", "fftw.planner=\"estimate\"",
                     "--set", "checks.urban_budgets={required = true, tolerance = 0.0}"])
    @test rc == 3
    res = TOML.parsefile(joinpath(only(readdir(root; join = true)), "output", "results.toml"))
    @test res["failed_required"] == ["urban_budgets"]
    # a check the site does not publish is a config error
    rc = quiet_main(["run", SMALL_CITY, "--root", mktempdir(), "--allow-uncommitted",
                     "--set", "checks.nesting_checks={required = true}"])
    @test rc == 2
end

@testset "urban run type: the GeoJSON ground from its static driver" begin
    Aeolus.Operators.set_fftw_policy!(; planner = :estimate)
    cfg = load_run_config(SMALL_CITY)
    cc = UC.CaseConfig(cfg)
    case = UC.build_case(cc)
    m, g = case.mesh, case.geom
    # the ground is the GeoJSON's: the 40 m and 20 m blocks are solid to
    # their heights (6.25 m levels), the 5 m shed (below min_height = 8 m) is not
    hcol(x, y) = count(k -> g.sm.mask[m.Hx + floor(Int, x / m.dx) + 1, m.Hy + floor(Int, y / m.dy) + 1, m.Hz + k], 1:m.Nz) * m.dz
    @test hcol(75.0, 75.0) ≈ 40.0 atol = m.dz
    @test hcol(140.0, 70.0) ≈ 20.0 atol = m.dz
    @test hcol(137.5, 137.5) == 0.0
    @test g.sfc[floor(Int, 60.0 / m.dx) + 1, floor(Int, 150.0 / m.dy) + 1] == UC.SURFACE_PARK
    @test g.sfc[floor(Int, 175.0 / m.dx) + 1, floor(Int, 150.0 / m.dy) + 1] == UC.SURFACE_WATER
    @test length(g.city.buildings) == 2
    @test endswith(cc.static, joinpath("input", "small_city_static.nc"))
    UC.run_case!(case)
    st = UC.case_state(case)
    @test all(a -> all(isfinite, a), values(st))
    # no static driver at all is an error, not a flat city
    @test_throws ErrorException UC.build_case(UC.CaseConfig(SMALL_CITY; static = nothing))
    # run_results reads the state only: the digests do not move
    d0 = Dict(k => digest(v) for (k, v) in pairs(UC.case_state(case)))
    r = UC.run_results(case)
    @test Dict(k => digest(v) for (k, v) in pairs(UC.case_state(case))) == d0
    @test r["budgets"]["closed"] === true
end

@testset "urban_parent: the nest parent alone, its tables and results" begin
    toml = berlin_toml("parent")
    tt = Aeolus.Config.config_test_table(toml)
    cfg = load_run_config(toml; overrides = tt.overrides)
    @test cfg.run.legacy == "urban_parent"
    r = run_config(cfg; root = mktempdir(), console = devnull, allow_uncommitted = true)
    out = joinpath(r.folder.path, "output")
    res = TOML.parsefile(joinpath(out, "results.toml"))
    @test res["grid"] == [24, 24, 38] && res["finite"] === true && !haskey(res, "failure")
    @test res["simulated_h"] ≈ 0.05 atol = 1e-9
    @test res["closure_rel"] < 1e-10
    @test res["dt_min"] > 0 && res["dt_max"] <= 10
    # one record at the start and every output_period (60 s) to 180 s
    lines = filter(l -> !startswith(l, "#"), readlines(joinpath(out, "parent.csv")))
    @test length(lines) == 1 + 4
    @test startswith(lines[1], "t_s,hour_LT,zi_min_flux_m")
    @test parse(Float64, split(lines[end], ',')[1]) ≈ 180.0
    th = filter(l -> !startswith(l, "#"), readlines(joinpath(out, "parent_theta.csv")))
    @test length(th) == 5 && length(split(th[1], ',')) == 1 + 38
    # the tiled surface: fractions of every cell sum to one, the child's
    # 1 km footprint covers 4 × 4 parent cells of 250 m
    ds = NCDataset(joinpath(out, "parent_surface.nc"))
    f = Array(ds["fraction"]); fp = Array(ds["footprint"]); z0 = Array(ds["z0"])
    close(ds)
    @test maximum(abs.(sum(f; dims = 3) .- 1)) < 1e-12
    @test count(>(0.5), fp) == 16
    @test all(z0 .> 0)
    @test Base.invokelatest(state_fields, r.obj).θ isa Array{Float64,3}
end

# ---------------------------------------------------------------------------
# examples/urban holds configuration, input data and analysis only
# ---------------------------------------------------------------------------

@testset "examples/urban: configs, input data and analysis of output files only" begin
    ex = joinpath(pkgdir(Aeolus), "examples", "urban")
    jl = String[]
    for (dir, _, fs) in walkdir(ex), f in fs
        endswith(f, ".jl") && push!(jl, joinpath(dir, f))
    end
    # the model is never built or run from a script
    forbidden = r"UrbanCase|BerlinCase|build_case|run_case!|step_case!|build_parent|run_parent!|step_parent!|CaseConfig|\brun_config\b|Aeolus\.main|load_run_config\([^)]*\)\s*\|>\s*run"
    offenders = String[]
    for f in jl, (i, l) in enumerate(readlines(f))
        startswith(lstrip(l), "#") && continue
        occursin(forbidden, l) && push!(offenders, "$(relpath(f, ex)):$i: $(strip(l))")
    end
    @test isempty(offenders)
    isempty(offenders) || @info "examples/urban scripts that build or run the model" offenders
    # every config validates; every script is analysis of output
    for f in readdir(ex; join = true)
        endswith(f, ".toml") || continue
        cfg = load_run_config(f)
        @test cfg.urban.enabled && (!hasproperty(cfg.run, :legacy) || cfg.run.legacy == "urban_parent")
    end
    @test sort(basename.(jl)) == ["alexanderplatz_animation.jl", "alexanderplatz_surface_diagnostics.jl",
                                  "berlin_parent_figures.jl", "berlin_parent_lindenberg_check.jl",
                                  "plot_parent_surface.jl"]
end
