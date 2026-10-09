# The in-model checks (ONE-d, #453): src/Checks.jl, src/components/checks.jl,
# the [checks] section of the model (src/config/simulation_driver.jl) and the
# exit code of `aeolus run` (src/cli/main.jl).
#
#   * the check entries and the reference data (CSV quantity tables and
#     profiles, NetCDF) and their errors;
#   * the reference comparison: the distance outside a band, the RMS of a
#     profile at the reference heights, the step-weighted window mean;
#   * the runner: settings, tolerances, required against advisory, a check
#     that throws, a stopped run, the results table, the checkpoint state;
#   * the domain: `finite` counts the non-finite values; a closure check is
#     round-off on a run and measures exactly the heat put into the air
#     between two steps; a component's own check; the config validation;
#   * the run: results.toml, exit code 0 / 3 (a failed required check) / 1
#     (a blow-up, which records the failed sanity check), the state bit for
#     bit that of a run without checks, and a resumed run's results equal to
#     an uninterrupted run's.

using Test
using TOML
using Aeolus
using Aeolus.Config
using Aeolus.Checks
using Aeolus.Components
const CKC = Aeolus.Components
using Aeolus.Fields: interior_view
using Aeolus.IO: read_run_toml, checkpoint_state, restore_checkpoint_state!
using Aeolus.NCIO: NCDataset
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, R_d
using Aeolus.SimulationDriver: SimulationDriverType

include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

const CK_CFGS = joinpath(pkgdir(Aeolus), "examples", "configs")
const CK_DCBL = joinpath(CK_CFGS, "dry_convective_bl.toml")
# the smoke size, with the checks the shipped file enables switched off ({} is off)
const CK_SMALL = ["grid.size=[8,8,8]", "fftw.planner=\"estimate\"", "time.duration=\"none\"", "time.nsteps=6",
                  "checks.finite={}", "checks.energy_closure={}", "checks.momentum_closure={}"]

ck_file(name, text) = (p = joinpath(mktempdir(), name); write(p, text); p)
ck_cfg(sets...) = load_run_config(CK_DCBL; overrides = vcat(CK_SMALL, String[sets...]))
ck_error(f) = try
    f(); nothing
catch e
    e isa ConfigError || rethrow(); e
end
"Run `Aeolus.main(args)` in-process: (exit code, stderr text)."
function ck_cli(args...)
    out = IOBuffer(); err = IOBuffer()
    rc = Aeolus.main(String[args...]; io = out, err = err)
    return rc, String(take!(err))
end
ck_sets(sets) = collect(Iterators.flatten(("--set", s) for s in sets))
ck_results(folder) = TOML.parsefile(joinpath(folder, "output", "results.toml"))

# ---------------------------------------------------------------------------
# entries and reference data
# ---------------------------------------------------------------------------

@testset "check entries: {required, tolerance}, reference entries" begin
    @test isempty(check_entry_problems(Dict{String,Any}()))                       # off
    @test isempty(check_entry_problems(Dict{String,Any}("required" => true, "tolerance" => 1e-8)))
    @test any(occursin("required:", p) for p in check_entry_problems(Dict{String,Any}("tolerance" => 1.0)))
    @test any(occursin("tolerance:", p) for p in check_entry_problems(Dict{String,Any}("required" => true, "tolerance" => -1.0)))
    @test any(occursin("required: expected", p) for p in check_entry_problems(Dict{String,Any}("required" => "yes")))
    @test any(occursin("windows: unknown key", p) for p in check_entry_problems(Dict{String,Any}("required" => true, "windows" => 1)))
    @test isempty(reference_entry_problems(Dict{String,Any}("required" => false, "variable" => "u_mean", "window" => [0, 10.0])))
    @test any(occursin("window:", p) for p in reference_entry_problems(Dict{String,Any}("required" => false, "window" => [5.0, 1.0])))
    @test any(occursin("required:", p) for p in reference_entry_problems(Dict{String,Any}()))       # always on
end

@testset "reference data: CSV quantity tables and profiles, NetCDF" begin
    band = ck_file("band.csv", """
        # a comment, then the header (unit and source are ignored)
        quantity,low,high,unit,source
        u_star,0.245,0.283,m s-1,Beare et al. 2006, via Basu & Porte-Agel 2006
        h,185,185,m,LASD 64^3
        """)
    r = read_reference(band)
    @test r["u_star"] == ReferenceValue(0.245, 0.283) && r["h"] == ReferenceValue(185.0, 185.0)
    single = read_reference(ck_file("v.csv", "quantity,value\nke,0.5\n"))
    @test single["ke"] == ReferenceValue(0.5, 0.5)
    prof = read_reference(ck_file("p.csv", "z,theta,u\n10,300.0,1.0\n20,300.5,2.0\n40,301.0,3.0\n"))
    @test prof["theta"].z == [10.0, 20.0, 40.0] && prof["u"].values == [1.0, 2.0, 3.0]
    # errors name the file and what is wrong
    @test_throws ArgumentError read_reference("/nonexistent/ref.csv")
    @test_throws "low 2.0 > high 1.0" read_reference(ck_file("b.csv", "quantity,low,high\nx,2.0,1.0\n"))
    @test_throws "expected a number" read_reference(ck_file("c.csv", "quantity,value\nx,abc\n"))
    @test_throws "z must increase" read_reference(ck_file("d.csv", "z,t\n2,1\n1,1\n"))
    @test_throws "the header needs" read_reference(ck_file("e.csv", "name,value\nx,1\n"))
    # NetCDF: a one-value variable, a band in attributes, a profile along z
    nc = joinpath(mktempdir(), "ref.nc")
    NCDataset(nc, "c") do ds
        ds.dim["z"] = 3; ds.dim["one"] = 1
        Aeolus.NCIO.NCDatasets.defVar(ds, "z", [5.0, 15.0, 25.0], ("z",))
        Aeolus.NCIO.NCDatasets.defVar(ds, "theta_profile", [300.0, 300.2, 300.4], ("z",))
        Aeolus.NCIO.NCDatasets.defVar(ds, "ke", [0.7], ("one",))
        Aeolus.NCIO.NCDatasets.defVar(ds, "u_star", [0.0], ("one",); attrib = Dict("low" => 0.2, "high" => 0.3))
    end
    rn = read_reference(nc)
    @test rn["ke"] == ReferenceValue(0.7, 0.7) && rn["u_star"] == ReferenceValue(0.2, 0.3)
    @test rn["theta_profile"].z == [5.0, 15.0, 25.0] && rn["theta_profile"].values == [300.0, 300.2, 300.4]
    @test !haskey(rn, "z")
end

# ---------------------------------------------------------------------------
# the reference comparison
# ---------------------------------------------------------------------------

@testset "reference comparison: band distance, window mean, profile RMS" begin
    x = Ref(0.0)
    # at the end of the run: the distance outside [low, high]
    c = reference_check("q", ReferenceValue(1.0, 2.0), () -> x[]; variable = "q_mean", units = "K")
    @test c.name == "reference_q" && c.units == "K" && c.tolerance == 0.0 && c.sample === nothing
    for (v, want) in ((1.5, 0.0), (2.5, 0.5), (0.25, 0.75), (2.0, 0.0))
        x[] = v
        val, info = c.evaluate(c.acc)
        @test val == want && info["model"] == v && info["low"] == 1.0 && info["high"] == 2.0
    end
    x[] = NaN
    @test isnan(first(c.evaluate(c.acc)))
    # a window: the mean weighted by the step length, over the steps ending in it
    c = reference_check("q", ReferenceValue(3.0, 3.5), () -> x[]; variable = "q_mean", window = (10.0, 20.0))
    for (t, dt, v) in ((5.0, 5.0, 100.0), (10.0, 5.0, 1.0), (12.0, 2.0, 7.0), (20.0, 8.0, 4.0), (25.0, 5.0, 100.0))
        x[] = v
        c.sample(c.acc, t, dt)
    end
    val, info = c.evaluate(c.acc)
    mean = (1.0 * 5 + 7.0 * 2 + 4.0 * 8) / 15               # = 3.4: inside the band
    @test info["model"] ≈ mean rtol = 1e-15
    @test val == 0.0 && info["steps_in_window"] == 3
    c2 = reference_check("q", ReferenceValue(0.0, 1.0), () -> 2.0; variable = "q_mean", window = (100.0, 200.0))
    c2.sample(c2.acc, 5.0, 5.0)
    val, info = c2.evaluate(c2.acc)
    @test isnan(val) && occursin("no step ended in the window", info["error"])
    # a profile: the RMS of model − reference at the reference heights inside the column
    zm = [0.5, 1.5, 2.5]
    ref = ReferenceProfile([-1.0, 1.0, 2.0, 9.0], [0.0, 1.5 + 0.3, 2.5 - 0.4, 0.0])
    c = reference_check("t", ref, () -> [1.0, 2.0, 3.0]; variable = "theta_profile", z = zm)
    val, info = c.evaluate(c.acc)
    @test val ≈ sqrt((0.3^2 + 0.4^2) / 2) rtol = 1e-14            # heights -1 and 9 are outside the column
    @test info["levels"] == 2
    @test info["max_abs_difference"] ≈ 0.4 rtol = 1e-14
    @test_throws ArgumentError reference_check("t", ref, () -> [1.0]; variable = "theta_profile")
    # gaps (ONE-g #456): a NaN reference value, and a height where the model
    # profile is undefined (NaN: above the virtual towers), are skipped; a
    # profile undefined everywhere fails
    ref2 = ReferenceProfile([0.6, 1.0, 2.0, 2.4], [NaN, 1.5 + 0.3, 2.5 - 0.4, 0.0])
    c = reference_check("t", ref2, () -> [1.0, 2.0, NaN]; variable = "theta_profile", z = zm)
    val, info = c.evaluate(c.acc)
    @test val ≈ 0.3 rtol = 1e-14 && info["levels"] == 1 && info["undefined_levels"] == 2
    c = reference_check("t", ref2, () -> [NaN, NaN, NaN]; variable = "theta_profile", z = zm)
    val, info = c.evaluate(c.acc)
    @test isnan(val) && occursin("undefined", info["error"])
end

# ---------------------------------------------------------------------------
# the runner and the results
# ---------------------------------------------------------------------------

ck_const(name, v; tol = 1.0) = Check(name; doc = "constant $v", units = "u", tolerance = tol,
                                     evaluate = _ -> (v, Dict{String,Any}("v" => v)))

@testset "runner: settings, required and advisory, results.toml, the error" begin
    cs = [ck_const("a", 0.5), ck_const("b", 2.0), ck_const("c", 0.0),
          Check("boom"; doc = "throws", tolerance = 1.0, evaluate = _ -> error("no data"))]
    @test_throws "not published: zz" CheckRunner(cs, Dict("zz" => CheckSetting(true, nothing)))
    @test_throws "two published checks" CheckRunner([cs[1], cs[1]], Dict("a" => CheckSetting(true, nothing)))
    # from a [checks] table: empty entries are off, references are named reference_<quantity>
    tbl = Dict{String,Any}("a" => Dict{String,Any}("required" => true), "b" => Dict{String,Any}("required" => false),
                           "c" => Dict{String,Any}(), "boom" => Dict{String,Any}("required" => false, "tolerance" => 5.0),
                           "reference_file" => nothing,
                           "reference" => Dict{String,Any}("ke" => Dict{String,Any}("required" => true)))
    st = enabled_checks(tbl)
    @test sort!(collect(keys(st))) == ["a", "b", "boom", "reference_ke"]
    @test st["boom"].tolerance == 5.0 && st["a"].tolerance === nothing && st["reference_ke"].required
    delete!(st, "reference_ke")
    st["c"] = CheckSetting(true, 0.0)
    res = evaluate_checks(CheckRunner(cs, st))
    by = Dict(x.name => x for x in res)
    @test by["a"].passed && by["a"].required && by["a"].tolerance == 1.0      # 0.5 ≤ the default 1
    @test !by["b"].passed && !by["b"].required                                # 2 > 1: advisory
    @test by["c"].passed                                                      # 0 ≤ 0
    @test !by["boom"].passed && isnan(by["boom"].value) && occursin("no data", by["boom"].info["error"])
    @test isempty(required_failures(res)) && Set(x.name for x in advisory_failures(res)) == Set(["b", "boom"])
    t = results_table(res)
    @test t["passed"] && t["failed_required"] == String[] && sort(t["failed_advisory"]) == ["b", "boom"]
    @test t["checks"]["b"]["value"] == 2.0 && t["checks"]["b"]["v"] == 2.0 && t["checks"]["b"]["units"] == "u"
    # a tolerance from [checks] overrides the default: b required at 1.5 fails the run
    res = evaluate_checks(CheckRunner(cs, Dict("b" => CheckSetting(true, 1.5))))
    @test !results_table(res)["passed"] && only(required_failures(res)).name == "b"
    msg = sprint(showerror, ChecksFailed(required_failures(res)))
    @test occursin("1 required check failed", msg) && occursin("b: 2 > tolerance 1.5 u", msg)
    # a stopped run: finite fails (added when not enabled), the others are not evaluated
    r = CheckRunner(cs, Dict("a" => CheckSetting(false, nothing)))
    res = stopped_checks(r, ErrorException("blew up"); step = 7, t = 3.5)
    @test [x.name for x in res] == ["finite", "a"]
    @test res[1].required && res[1].evaluated && !res[1].passed && res[1].info["stopped_at_step"] == 7
    @test !res[2].evaluated && !res[2].passed && occursin("blew up", res[2].info["error"])
    # the accumulators are checkpointed and restored (a resumed run continues them)
    x = Ref(1.0)
    rc = reference_check("q", ReferenceValue(0.0, 10.0), () -> x[]; variable = "q", window = (0.0, 100.0))
    r1 = CheckRunner([rc], Dict("reference_q" => CheckSetting(true, nothing)))
    sample_checks!(r1, 1.0, 1.0); sample_checks!(r1, 2.0, 1.0)
    s = checkpoint_state(r1)
    rc2 = reference_check("q", ReferenceValue(0.0, 10.0), () -> x[]; variable = "q", window = (0.0, 100.0))
    r2 = restore_checkpoint_state!(CheckRunner([rc2], Dict("reference_q" => CheckSetting(true, nothing))), s)
    @test rc2.acc.n == 2 && rc2.acc.weight == 2.0 && rc2.acc.sum == 2.0
    sample_checks!(r1, 3.0, 1.0); sample_checks!(r2, 3.0, 1.0)
    @test only(evaluate_checks(r1)).info == only(evaluate_checks(r2)).info
end

# ---------------------------------------------------------------------------
# the domain's checks
# ---------------------------------------------------------------------------

ck_build(sets...) = Config.build(SimulationDriverType(), ck_cfg(sets...))

@testset "domain checks: finite counts the non-finite values" begin
    d = only(ck_build().model.domains)
    @test [c.name for c in CKC.domain_checks(d)] == ["finite"]            # budgets off: no closure check
    c = only(CKC.domain_checks(d))
    @test first(c.evaluate(c.acc)) == 0.0
    th = interior_view(d.state.θ)
    th[1, 2, 3] = NaN; th[2, 2, 3] = Inf; interior_view(d.state.u)[1, 1, 1] = NaN
    val, info = c.evaluate(c.acc)
    @test val == 3.0 && Set(info["fields"]) == Set(["u", "θ"])
end

@testset "domain checks: a closure check is round-off, and measures heat added outside the stages" begin
    r = ck_build("checks.energy_closure={required=true}", "checks.momentum_closure={required=true}")
    d = only(r.model.domains)
    @test Set(d.budgets.quantities) == Set([:energy, :momentum_x, :momentum_y])   # the checks turned them on
    @test [c.name for c in CKC.domain_checks(d)] == ["finite", "energy_closure", "momentum_closure"]
    for _ in 1:4
        step!(r.model, 5.0)
    end
    ce = CKC.domain_checks(d)[2]
    val, info = ce.evaluate(ce.acc)
    # the surface flux puts ρ₀ c_p F t = 1.18 × 1005 × 0.24 × 20 J m⁻² into the air: the closure is round-off of that
    ρ0 = DEFAULT_CONSTANTS.p_ref / (R_d(DEFAULT_CONSTANTS) * 300.0); cp = DEFAULT_CONSTANTS.cp_dry
    @test info["elapsed"] == 20.0 && info["unattributed"] == 0.0
    @test val <= 1e-12 * ρ0 * cp * 300.0 * 1600.0 / 20.0                   # round-off of the air's heat content
    @test val <= 1e-6                                                      # the default tolerance
    # heat put into one cell between two steps, outside every stage entry: booked as
    # (:output, :between_steps) and measured by the check as its rate over the run
    m = d.state.mesh
    interior_view(d.state.θ)[3, 4, 2] += 0.5
    for _ in 1:2
        step!(r.model, 5.0)
    end
    added = ρ0 * cp * 0.5 * m.dz / (m.Nx * m.Ny)                           # J m⁻² of ground
    val, info = ce.evaluate(ce.acc)
    @test info["elapsed"] == 30.0
    @test val ≈ added / 30.0 rtol = 1e-9
    @test info["unattributed"] ≈ added / 30.0 rtol = 1e-9
    @test abs(info["closure"]) < 1e-9 * added
    cm = CKC.domain_checks(d)[3]
    val, info = cm.evaluate(cm.acc)
    @test haskey(info, "closure_x") && haskey(info, "closure_y") && val <= 1e-9
end

# a component that publishes a check of its own
struct CKProbe <: AbstractComponent end
CKC.component_name(::Type{CKProbe}) = :probe
CKC.section(::Type{CKProbe}) = "probe"
CKC.check_names(::Type{CKProbe}) = ("probe_max_w",)
CKC.checks(::CKProbe) = [Check("probe_max_w"; doc = "the largest |w| (m/s)", units = "m s-1", tolerance = 10.0,
                               evaluate = _ -> (3.0, Dict{String,Any}()))]

@testset "domain checks: a component's own checks, and their validation" begin
    r = ck_build()
    d0 = only(r.model.domains)
    d = Domain(d0; components = (d0.components..., CKProbe()))
    @test [c.name for c in CKC.domain_checks(d)] == ["finite", "probe_max_w"]
    res = evaluate_checks(CheckRunner(CKC.domain_checks(d), Dict("probe_max_w" => CheckSetting(true, 2.0))))
    @test !only(res).passed && only(res).value == 3.0
    kw = (moisture = false, theta0 = 300.0, core = :boussinesq)
    @test isempty(CKC.check_problems(["probe_max_w", "finite"], [CKProbe => (;)]; kw...))
    p = only(CKC.check_problems(["probe_max_w"], Pair[]; kw...))
    @test p.first == "probe_max_w" && occursin("no enabled component publishes", p.second)
    p = only(CKC.check_problems(["water_closure"], Pair[]; kw...))
    @test p.first == "water_closure" && occursin("water_closure needs dynamics.moisture", p.second)
    @test only(CKC.check_problems(["energy_closure"], Pair[]; kw..., theta0 = 0.0)).first == "energy_closure"
    @test occursin("tracer", only(CKC.check_problems(["tracer_closure"], Pair[]; kw...)).second)
end

@testset "[checks] validation" begin
    key(e) = [p.key for p in e.problems]
    @test key(ck_error(() -> ck_cfg("checks.nonsense={required=true}"))) == ["checks.nonsense"]
    @test key(ck_error(() -> ck_cfg("checks.water_closure={required=true}"))) == ["checks.water_closure"]
    @test key(ck_error(() -> ck_cfg("checks.finite={tolerance=1.0}"))) == ["checks.finite"]
    ref = ck_file("ref.csv", "quantity,low,high\ntheta_mean,299,301\nsome_quantity,0,1\n")
    @test key(ck_error(() -> ck_cfg("checks.reference={theta_mean={required=true}}"))) == ["checks.reference"]
    @test key(ck_error(() -> ck_cfg("checks.reference_file=\"$ref\""))) == ["checks.reference_file"]
    e = ck_error(() -> ck_cfg("checks.reference_file=\"$ref\"", "checks.reference={missing_q={required=true}}"))
    @test any(p -> p.key == "checks.reference.missing_q" && occursin("has no quantity missing_q", p.message), e.problems)
    e = ck_error(() -> ck_cfg("checks.reference_file=\"$ref\"", "checks.reference={some_quantity={required=true}}"))
    @test any(p -> p.key == "checks.reference.some_quantity" && occursin("no enabled component publishes", p.message), e.problems)
    e = ck_error(() -> ck_cfg("checks.reference_file=\"/nonexistent/r.csv\"", "checks.reference={theta_mean={required=true}}"))
    @test any(p -> p.key == "checks.reference_file" && occursin("does not exist", p.message), e.problems)
    # valid: the reference file is an input, hashed under [inputs]
    cfg = ck_cfg("checks.reference_file=\"$ref\"", "checks.reference={theta_mean={required=true, window=[0.0, 30.0]}}",
                 "checks.finite={required=true}")
    @test haskey(config_inputs(cfg), "checks.reference_file")
    # a variable compared with a reference of the wrong shape: a ConfigError at the build
    prof = ck_file("prof.csv", "z,theta_mean\n100,300\n200,300\n")
    cfg = ck_cfg("checks.reference_file=\"$prof\"", "checks.reference={theta_mean={required=true}}")
    e = ck_error(() -> CKC.reference_checks(only(Config.build(SimulationDriverType(), cfg).model.domains),
                                            read_reference(prof), cfg.checks.reference))
    @test only(e.problems).key == "checks.reference.theta_mean" && occursin("cannot be compared with a profile", only(e.problems).message)
end

# ---------------------------------------------------------------------------
# the run: results.toml and the exit code
# ---------------------------------------------------------------------------

const CK_REF = ck_file("dcbl_ref.csv", """
    # theta_mean of the 8³ smoke run of dry_convective_bl.toml: the mixed layer at
    # 300 K to 800 m, then 3 K/km to 1600 m: 300 + 0.003 × 800² / (2 × 1600) = 300.6 K,
    # plus the surface heating F t / H ≤ 0.24 × 30 / 1600 = 0.0045 K
    quantity,low,high
    theta_mean,300.59,300.62
    w_mean,-1e-12,1e-12
    """)
const CK_PROF = ck_file("dcbl_prof.csv", "z,theta_profile,u_profile\n100,300.0,5.0\n700,300.0,5.0\n1200,301.2,5.0\n")

@testset "aeolus run: results.toml, exit code 0 and 3, the state unchanged" begin
    root = mktempdir()
    sets = ["checks.finite={required=true}", "checks.energy_closure={required=true}",
            "checks.momentum_closure={required=true}",
            "checks.reference_file=\"$CK_REF\"",
            "checks.reference={theta_mean={required=true, window=[10.0, 30.0]}, w_mean={required=true}}"]
    rc, err = ck_cli("run", CK_DCBL, ck_sets(vcat(CK_SMALL, sets))..., "--root", root, "--allow-uncommitted")
    @test rc == 0
    folder = only(readdir(root; join = true))
    res = ck_results(folder)
    @test res["passed"] && isempty(res["failed_required"]) && isempty(res["failed_advisory"])
    ch = res["checks"]
    @test Set(keys(ch)) == Set(["finite", "energy_closure", "momentum_closure", "reference_theta_mean", "reference_w_mean"])
    @test all(c -> c["passed"] && c["evaluated"] && c["required"], values(ch))
    @test ch["energy_closure"]["units"] == "W m-2" && ch["energy_closure"]["elapsed"] == 30.0
    @test ch["energy_closure"]["value"] < 1e-9
    # the surface flux's heat is the storage of the run: ρ₀ c_p F (W m⁻², the sponge acts on the wind only)
    ρ0 = DEFAULT_CONSTANTS.p_ref / (R_d(DEFAULT_CONSTANTS) * 300.0)
    @test ch["energy_closure"]["storage"] ≈ ρ0 * DEFAULT_CONSTANTS.cp_dry * 0.24 rtol = 1e-9
    @test ch["reference_theta_mean"]["steps_in_window"] == 5 && ch["reference_theta_mean"]["window"] == [10.0, 30.0]
    @test 300.59 < ch["reference_theta_mean"]["model"] < 300.62
    # the state is bit for bit that of the run without [checks] (the budgets the closures turn on measure only)
    r0 = run_config(ck_cfg(); root = mktempdir(), allow_uncommitted = true, console = false)
    r1 = run_config(ck_cfg(sets...); root = mktempdir(), allow_uncommitted = true, console = false)
    f0 = Config.state_fields(r0.obj); f1 = Config.state_fields(r1.obj)
    @test all(k -> bitwise_equal(f0[k], f1[k]), keys(f0))
    # without [checks] nothing is written
    @test !isfile(joinpath(r0.folder.path, "output", "results.toml"))
    # a failed required check: exit code 3, results.toml says which, the session is failed.
    # The θ profile fails: the surface flux has warmed the lowest level (z = 100 m) by
    # about F t / Δz = 0.24 × 30 / 200 = 0.036 K over the 300 K of the reference; the
    # advisory u profile fails too (the run has no mean wind, the reference 5 m/s)
    root = mktempdir()
    fail = ["checks.energy_closure={required=true, tolerance=0.0}", "checks.finite={required=false}",
            "checks.reference_file=\"$CK_PROF\"",
            "checks.reference={theta_profile={required=true, tolerance=1e-3}, u_profile={required=false, tolerance=1.0}}"]
    rc, err = ck_cli("run", CK_DCBL, ck_sets(vcat(CK_SMALL, fail))..., "--root", root, "--allow-uncommitted")
    folder = only(readdir(root; join = true))
    res = ck_results(folder)
    tp = res["checks"]["reference_theta_profile"]
    @test rc == 3 && occursin("reference_theta_profile", err) && occursin("required check failed", err)
    @test !res["passed"] && res["failed_required"] == ["reference_theta_profile"]
    @test tp["levels"] == 3 && 0.01 < tp["max_abs_difference"] < 0.1 && tp["value"] > 1e-3
    @test res["failed_advisory"] == ["reference_u_profile"]
    @test res["checks"]["reference_u_profile"]["value"] ≈ 5.0 atol = 0.1
    @test res["checks"]["energy_closure"]["passed"]              # 0 ≤ 0: the closure is exact here
    @test read_run_toml(folder)["session"][1]["status"] == "failed"
    @test isfile(joinpath(folder, "output", "output.nc"))
    # an advisory failure alone: exit code 0
    root = mktempdir()
    rc, err = ck_cli("run", CK_DCBL, ck_sets(vcat(CK_SMALL, ["checks.reference_file=\"$CK_PROF\"",
                                                             "checks.reference={u_profile={required=false}}"]))...,
                     "--root", root, "--allow-uncommitted")
    @test rc == 0 && ck_results(only(readdir(root; join = true)))["failed_advisory"] == ["reference_u_profile"]
end

@testset "aeolus run: a blow-up records the failed sanity check (exit code 1)" begin
    root = mktempdir()
    tgv = joinpath(CK_CFGS, "taylor_green_vortex.toml")
    # a step far beyond the stability limit (the blowup test of config_cli.jl): NaN by step 11
    sets = ["grid.size=[8,8,8]", "time.duration=\"none\"", "time.nsteps=20", "fftw.planner=\"estimate\"",
            "time.dt=50.0", "checks.reference_file=\"$(ck_file("tgv.csv", "quantity,value\nu_mean,0.0\n"))\"",
            "checks.reference={u_mean={required=false}}"]
    rc, err = ck_cli("run", tgv, ck_sets(sets)..., "--root", root, "--allow-uncommitted")
    @test rc == 1 && occursin("blew up", err)
    res = ck_results(only(readdir(root; join = true)))
    @test !res["passed"] && res["failed_required"] == ["finite"] && res["failed_advisory"] == ["reference_u_mean"]
    f = res["checks"]["finite"]
    @test f["evaluated"] && !f["passed"] && occursin("blew up", f["error"]) && f["stopped_at_step"] > 4
    @test !res["checks"]["reference_u_mean"]["evaluated"]
end

@testset "aeolus run --resume: the checks continue (window means checkpointed)" begin
    sets = ["grid.size=[8,8,8]", "checks.finite={required=true}", "checks.energy_closure={required=true}",
            "checks.reference_file=\"$CK_REF\"", "checks.reference={theta_mean={required=true, window=[10.0, 60.0]}}"]
    r = resume_check(CK_DCBL; nsteps = 12, stop = 7, every_steps = 5, overrides = sets, outputs = ["results.toml"])
    @test r.fields_equal && r.outputs_equal
    @test r.statuses == ["interrupted", "completed"]
    @test ck_results(r.runs[2].folder.path)["checks"]["reference_theta_mean"]["steps_in_window"] == 11
end
