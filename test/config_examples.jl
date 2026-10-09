# The example scripts configured from TOML (CFG-e, #360).
#
# Acceptance tests of the issue covered in this file:
#   * sweeps: a [sweep] table expands into override sets in a fixed order,
#     one run folder per member, each member validated (the neutral_abl
#     3 closures × 2 resolutions is in config_examples_validation.jl);
#   * coverage: every examples/**/*.jl that calls a step!/run!/sbl_run!/
#     run_case! entry point has a configs/*.toml file or a reasoned entry
#     in EXEMPT below.
# The per-directory bit-identity and round-trip tests are in
# config_examples_*.jl.

using Test
using Aeolus
using Aeolus.Config
using Aeolus.IO: read_run_toml

const EX_DIR = joinpath(pkgdir(Aeolus), "examples")
const TGV_CFG = joinpath(EX_DIR, "configs", "taylor_green_vortex.toml")

ex_file(text) = (p = joinpath(mktempdir(), "sweep.toml"); write(p, text); p)

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
    p = ex_file("""
        extends = $(repr(TGV_CFG))
        [sweep]
        axis = [ { key = "sgs.closure", values = ["none", "vreman", "amd"] },
                 { key = "grid.size", values = [[8, 8, 8], [16, 16, 16]] } ]
        """)
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
    p = ex_file("""
        extends = $(repr(TGV_CFG))
        [[sweep.member]]
        label = "viscous"
        dynamics = { nu = 0.02 }
        [[sweep.member]]
        sgs.closure = "vreman"
        sgs.c_s = 0.2
        [sweep]
        axis = [ { key = "stepper.scheme", values = ["euler", "rk3"] } ]
        """)
    runs = load_sweep(p)
    @test [m.label for (m, _) in runs] == ["viscous_euler", "viscous_rk3", "2_euler", "2_rk3"]
    @test [c.dynamics.nu for (_, c) in runs] == [0.02, 0.02, 0.01, 0.01]
    @test runs[3][2].sgs.c_s == 0.2 && runs[3][2].stepper.scheme === :euler
    # a member that is invalid is reported with its key
    bad = ex_file("""
        extends = $(repr(TGV_CFG))
        [[sweep.member]]
        sgs = { closure = "smagorinksy" }
        """)
    e = ex_error(() -> load_sweep(bad))
    @test occursin("sgs.closure", sprint(showerror, e)) && occursin("smagorinsky", sprint(showerror, e))
    # malformed sweep tables
    e = ex_error(() -> sweep_members(ex_file("[sweep]\naxis = [ { key = \"nu\", values = [1] } ]\nmembers = 1\n")))
    s = sprint(showerror, e)
    @test occursin("section.option", s) && occursin("sweep.members", s) && occursin("did you mean \"member\"", s)
    @test sweep_members(TGV_CFG) === nothing
end

@testset "sweeps: aeolus run runs every member into its own folder; a member replays alone" begin
    p = ex_file("""
        extends = $(repr(TGV_CFG))
        [sweep]
        axis = [ { key = "dynamics.nu", values = [0.01, 0.02] } ]
        """)
    root = mktempdir()
    args = ["--set", "grid.size=[8,8,8]", "--set", "stepper.end_time=\"none\"", "--set", "stepper.nsteps=3",
            "--set", "fftw.planner=\"estimate\"", "--root", root]
    res = run_configs(p, args; console = false, repo = mktempdir())
    @test length(res) == 2
    @test res[1].folder.path != res[2].folder.path
    @test [r.obj.sim.ν for r in res] == [0.01, 0.02]
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
    @test_throws Aeolus.Config.UsageError run_configs(p, ["32"])
    out = IOBuffer()
    @test Aeolus.main(["check", p]; io = out) == 0
    @test count("ok:", String(take!(out))) == 2
end

# ---------------------------------------------------------------------------
# examples/kernels/: 04_cloud_spinup (the column driver's kessler_box)
# ---------------------------------------------------------------------------

include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))

@testset "kernels: cloud_spinup.toml equals the old in-code Kessler box, bit for bit" begin
    # examples/kernels/04_cloud_spinup.jl before CFG-e, verbatim but for the plots
    H = 2000.0
    mesh = RegularGrid(; size = (4, 4, 32), extent = (H, H, H), halo = (1, 1, 1))
    qv = Field(CellCenter(), mesh); Aeolus.Fields.set!(qv, (x, y, z) -> 0.01)
    ql = Field(CellCenter(), mesh); Aeolus.Fields.set!(ql, (x, y, z) -> 0.6H < z < 0.9H ? 3e-3 : 0.0)
    qr = Field(CellCenter(), mesh); Aeolus.Fields.set!(qr, (x, y, z) -> 0.0)
    q0 = sum(qv.data) + sum(ql.data)
    for dt in (60.0, 60.0, 120.0, 120.0, 120.0)
        apply_kessler_step!(qv, ql, qr, dt; Vt = :spectrum)
    end
    cfg = load_run_config(joinpath(EX_DIR, "kernels", "configs", "cloud_spinup.toml"))
    r = run_config(cfg; root = mktempdir(), console = false, repo = mktempdir())
    f = Base.invokelatest(state_fields, r.obj)
    @test f.qv == qv.data && f.ql == ql.data && f.qr == qr.data
    @test all(bitwise_equal(a, b) for (a, b) in ((f.qv, qv.data), (f.ql, ql.data), (f.qr, qr.data)))
    # the physics: autoconversion turns cloud into rain (the cloud-layer
    # maximum falls monotonically) and the rain reaches below the cloud base
    snaps = r.obj.snapshots
    @test [s.t for s in snaps] == [0.0, 60.0, 120.0, 240.0, 360.0, 480.0]
    qlmax = [maximum(s.ql) for s in snaps]
    @test issorted(qlmax; rev = true) && qlmax[end] < qlmax[1]
    @test maximum(snaps[1].qr) == 0 && maximum(snaps[end].qr) > 1e-5
    zc = collect(Aeolus.Grids.znodes(mesh, Aeolus.Grids.CellCenter()))
    @test any(snaps[end].qr[k] > 0 for k in eachindex(zc) if zc[k] < 0.6H)
    # vapour and cloud only lose water to the rain (no source of water)
    @test sum(f.qv) + sum(f.ql) <= q0
end

@testset "round trip from run.toml (kernels: cloud_spinup.toml), bit-identical" begin
    r = roundtrip_check(joinpath(EX_DIR, "kernels", "configs", "cloud_spinup.toml"))
    @test r.fields_equal && r.config_equal && r.results_equal
end

# ---------------------------------------------------------------------------
# coverage: every example that integrates the equations runs from a config
# ---------------------------------------------------------------------------

# A script integrates the equations if it calls a stepping entry point.
const ENTRY_POINT = r"(^|[^A-Za-z0-9_.])(step[A-Za-z0-9_]*!|run!|sbl_run!|run_case!)\("
# A script is configured if it runs a shipped configs/*.toml through
# run_configs or the aeolus command line.
const CONFIG_RUN = r"(run_configs|Aeolus\.main)\("
const CONFIG_REF = r"\"configs\",\s*\"([A-Za-z0-9_.-]+\.toml)\""

const URBAN_LIB = "library of the berlin driver (examples/urban/BerlinCase.jl; configs in examples/urban/configs/, CFG-c #358)"

# Files that call an entry point (or look like runs) but are not runs of
# their own, each with its reason. Paths relative to examples/.
const EXEMPT = Dict(
    "validation/canopy_profiles.jl" =>
        "not a run: steady 1-D canopy wind / scalar boundary-value solves against published profiles",
    "validation/dynamic_roughness_cubes.jl" =>
        "not a run: a-priori dynamic roughness of a cube height map under a prescribed log-law wind",
    "validation/dns256_generate.jl" =>
        "fixture generator: writes test/fixtures/dns256_apriori.toml offline (generate_dns_apriori, its own checkpoints)",
    "poisson_manufactured.jl" => "not a run: manufactured-solution check of the Poisson solver",
    "poisson_neumann_manufactured.jl" => "not a run: manufactured-solution check of the Neumann Poisson solver",
    "urban/berlin_checks.jl" =>
        "not a run: by-hand end-to-end checks of the Berlin example (the smoke matrix over configs/berlin_smoke.toml, the parent validations; #320)",
    "urban/core/bus.jl" => URBAN_LIB,
    "urban/core/config_schema.jl" => URBAN_LIB,
    "urban/core/stages.jl" => URBAN_LIB,
    "urban/couplings/buildings_citymean.jl" => URBAN_LIB,
    "urban/couplings/nesting.jl" => URBAN_LIB,
    "urban/couplings/surface_exchange.jl" => URBAN_LIB,
    "urban/couplings/timestepper.jl" => URBAN_LIB,
    "urban/parent/case.jl" => URBAN_LIB,
    "urban/parent/surface.jl" => URBAN_LIB,
)

# Directories migrated by their own issues.
const OTHER_ISSUE_DIRS = Dict("sbl/" => "#359 (CFG-d, SBL)")

# The rest of #360's scope, migrated in later parts of the issue (it may be
# split per directory). This list may only shrink: a file migrated to a
# config must leave it (checked below).
const PENDING_360 = Set([
    "kernels/01_tgv_energy_decay.jl", "kernels/02_decaying_hit.jl", "kernels/03_channel_flow.jl",
    "kernels/05_pollution_plume.jl", "kernels/06_particle_trail.jl",
    "coupled/01_dcbl_radiation_lsm.jl", "coupled/02_moist_convection.jl", "coupled/03_heated_plume_rain.jl",
    "coupled/04_flow_past_cube.jl",
    "sgs/cbl_tke.jl", "sgs/rotating_stratified_decay.jl", "sgs/surface_mounted_cube.jl",
    "transition/flat_plate_t3a.jl", "transition/kelvin_helmholtz.jl",
    "nesting/cbl_inflow_statistics.jl", "nesting/inflow_fetch.jl",
    "scalar_advection_2d.jl",
])

function example_scripts()
    files = String[]
    for (dir, _, fs) in walkdir(EX_DIR), f in fs
        endswith(f, ".jl") && push!(files, relpath(joinpath(dir, f), EX_DIR))
    end
    return sort!(replace.(files, '\\' => '/'))
end

"The configs a script runs, as absolute paths (empty if it runs none)."
function script_configs(rel)
    src = read(joinpath(EX_DIR, rel), String)
    occursin(CONFIG_RUN, src) || return String[]
    return [joinpath(dirname(joinpath(EX_DIR, rel)), "configs", m[1]) for m in eachmatch(CONFIG_REF, src)]
end

@testset "coverage: every example that integrates the equations has a config or a reasoned exemption" begin
    drivers = Set(relpath(joinpath(pkgdir(Aeolus), p), EX_DIR) for p in values(Aeolus.Config.DRIVER_FILES))
    missing_ = String[]
    nconfigured = 0
    for rel in example_scripts()
        rel in drivers && continue                        # a run driver (registers itself)
        any(startswith(rel, d) for d in keys(OTHER_ISSUE_DIRS)) && continue
        cfgs = script_configs(rel)
        if !isempty(cfgs)
            nconfigured += 1
            # each config the script names exists and validates (every member of a sweep)
            for c in cfgs
                @test isfile(c)
                @test !isempty(load_sweep(c))
            end
            @test !(rel in PENDING_360)                    # migrated: leaves the pending list
            @test !haskey(EXEMPT, rel)
            # and the script parses no ARGS / ENV of its own (they go to run_configs)
            src = read(joinpath(EX_DIR, rel), String)
            @test !occursin(r"ARGS\[|parse\([^)]*ARGS|ENV\[|get\(ENV", src)
        elseif occursin(ENTRY_POINT, read(joinpath(EX_DIR, rel), String))
            haskey(EXEMPT, rel) || rel in PENDING_360 || push!(missing_, rel)
        end
    end
    @test isempty(missing_)
    isempty(missing_) || @info "examples that integrate the equations with neither a config nor an exemption" missing_
    # no stale entries: every exemption and pending file exists
    @test all(isfile(joinpath(EX_DIR, p)) for p in keys(EXEMPT))
    @test all(isfile(joinpath(EX_DIR, p)) for p in PENDING_360)
    @test all(!isempty(r) for r in values(EXEMPT))
    # examples/validation/ is complete: every script there is configured, a driver or exempt
    val = [r for r in example_scripts() if startswith(r, "validation/") && !(r in drivers)]
    @test all(!isempty(script_configs(r)) || haskey(EXEMPT, r) for r in val)
    @test nconfigured == 13        # the 3 bare runners of CFG-b, 8 validation scripts, kernels/04, urban/alexanderplatz_diurnal (CFG-c)
end
