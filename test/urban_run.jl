# The urban site of the model (EX-a #438, ONE-e #454): Aeolus.Urban,
# a model config with `[urban] enabled`, and its nest parent alone (a
# domain of the model's core, NEST-p3 #483).
# Small synthetic setups:
#
#   * a synthetic city (test/data/urban/small_city.toml: two buildings, a
#     park and a pond from GeoJSON, 16³, periodic, local noon) runs through
#     `aeolus run`: the run folder's results.toml holds the run type's own
#     checks (the ledgers closed, a finite state, the sun's net gain), and
#     the ground is the GeoJSON's its static driver holds (heights,
#     classes, the shed below the minimum height dropped);
#   * step!(model, dt) rejects a dt above the adaptive stable step (#392),
#     on the synthetic city (the nest parent's domain: test/config_parent_domain.jl).
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

const UC = Aeolus.Urban
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))

const SMALL_CITY = joinpath(@__DIR__, "data", "urban", "small_city.toml")

"Run `args` through the aeolus command line quietly; the exit code."
quiet_main(args) = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        Aeolus.main(args)
    end
end

@testset "urban run type: the synthetic city through aeolus run, its checks through [checks] (ONE-d)" begin
    root = mktempdir()
    chk = ["--set", "checks.urban_budgets={required = true}", "--set", "checks.surface_exchange_checks={required = true}",
           "--set", "checks.energy_closure={required = true, tolerance = 1e-9}", "--set", "checks.finite={required = true}"]
    rc = quiet_main(vcat(["run", SMALL_CITY, "--root", root, "--allow-uncommitted", "--set", "fftw.planner=\"estimate\""], chk))
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
    # the site's checks through [checks]
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

@testset "urban run type: [budgets] enabled = false, the same configuration without the budgets (#539)" begin
    # the configuration of the run above: its budget checks are dropped, the site builds no budgets
    root = mktempdir()
    chk = ["--set", "checks.urban_budgets={required = true}", "--set", "checks.energy_closure={required = true, tolerance = 1e-9}",
           "--set", "checks.finite={required = true}", "--set", "budgets.enabled=false"]
    rc = quiet_main(vcat(["run", SMALL_CITY, "--root", root, "--allow-uncommitted", "--set", "fftw.planner=\"estimate\""], chk))
    @test rc == 0
    run = only(readdir(root; join = true))
    res = TOML.parsefile(joinpath(run, "output", "results.toml"))
    @test res["budgets"] == Dict("enabled" => false)
    @test !haskey(res, "air_heat") && res["finite"] === true
    @test haskey(res["checks"], "finite") && !haskey(res["checks"], "urban_budgets") && !haskey(res["checks"], "energy_closure")
    # the site's domain measures nothing
    cfg = load_run_config(SMALL_CITY; overrides = ["budgets.enabled=false", "fftw.planner=\"estimate\""])
    d = only(UC.build_model(cfg).domains)
    @test d.budgets === nothing
    @test !UC.budgets_enabled(UC.urban_case(d))
end

@testset "urban run type: the GeoJSON ground from its static driver" begin
    Aeolus.Operators.set_fftw_policy!(; planner = :estimate)
    cfg = load_run_config(SMALL_CITY)
    cc = UC.site_options(cfg)
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
    # step!(model, dt) rejects a dt above the adaptive stable step (#392)
    for _ in 1:2
        step!(case.model)
    end
    @test !case.clock.blowup
    ts = UC.timestepper(case)
    @test UC.variable_dt(typeof(ts))
    lim = UC.stable_dt(ts, case)
    lims = copy(ts.stats.limiters)
    @test UC.stable_dt(ts, case) == lim && ts.stats.limiters == lims        # pure: nothing counted
    @test ts.constraints.dt_min <= lim <= ts.constraints.dt_max
    dt_ok = UC.step_dt(case)
    @test 0 < dt_ok <= lim                                                  # the stepper's own pick
    n0, t0, u0 = case.clock.n, case.sim.t, copy(interior_view(case.sim.u))
    err = try
        step!(case.model, 1.5lim); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("stable step", err.msg)
    @test case.clock.n == n0 && case.sim.t == t0 && interior_view(case.sim.u) == u0   # rejected before any stage
    run_site!(case)
    st = UC.case_state(case)
    @test all(a -> all(isfinite, a), values(st))
    # no static driver at all is an error, not a flat city
    @test_throws ErrorException UC.build_case(site_config(SMALL_CITY; static = nothing))
    # run_results reads the state only: nothing it touches is written (an
    # unmodified array is its copy, exactly: no arithmetic between them)
    s0 = Dict(k => copy(v) for (k, v) in pairs(UC.case_state(case)))
    r = UC.run_results(case)
    @test all(k -> isequal(UC.case_state(case)[k], s0[k]), keys(s0))
    @test r["budgets"]["closed"] === true
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
    forbidden = r"Aeolus\.Urban\b|\bUrban\.|UrbanCase|BerlinCase|build_case|build_model|run_case!|step_case!|run_to!|build_parent|run_parent!|step_parent!|CaseConfig|site_options|\brun_config\b|Aeolus\.main|load_run_config\([^)]*\)\s*\|>\s*run"
    offenders = String[]
    for f in jl, (i, l) in enumerate(readlines(f))
        startswith(lstrip(l), "#") && continue
        occursin(forbidden, l) && push!(offenders, "$(relpath(f, ex)):$i: $(strip(l))")
    end
    @test isempty(offenders)
    # every config validates: an urban site (alone, or nested in its parent domain), or the parent alone (its tiles)
    ok(cfg) = !hasproperty(cfg.run, :legacy) &&
              (UC.urban_enabled(cfg) || Aeolus.Config.config_domains(cfg)[1].surface.model === :tiles)
    @test all(f -> ok(load_run_config(f)), filter(endswith(".toml"), readdir(ex; join = true)))
    @test sort(basename.(jl)) == ["alexanderplatz_animation.jl", "alexanderplatz_surface_diagnostics.jl",
                                  "berlin_parent_figures.jl", "berlin_parent_lindenberg_check.jl", "parent_output.jl",
                                  "plot_parent_surface.jl"]
end
