# The urban run type (EX-a, #438): Aeolus.UrbanCase, the `urban` and
# `urban_parent` run drivers, moved from examples/urban/ into src/. Small
# synthetic setups:
#
#   * a synthetic city (test/data/urban/small_city.toml: two buildings, a
#     park and a pond from GeoJSON, 16³, periodic, local noon) runs through
#     `aeolus run`: the run folder's results.toml holds the run type's own
#     checks (the ledgers closed, a finite state, the sun's net gain), the
#     ground is the GeoJSON's (heights, classes, the shed below min_height
#     dropped), and the static driver `aeolus preprocess static` writes from
#     it gives the in-memory run bit for bit;
#   * the nest parent alone (`urban_parent`, berlin_parent.toml at its
#     smoke size): its tables, results and tiled-surface file;
#   * the input preprocessors that replaced the example's data scripts:
#     osm-sample reproduces the committed synthetic OSM file byte for byte,
#     the IGRA2 parser turns synthetic records into a sounding CSV that
#     read_sounding reads back, and the parent land-use table of an OSM
#     extract is self-consistent;
# The examples policy (no example script builds or runs a model) is
# test/examples_config_only.jl for every example directory.

using Test
using TOML
using Printf
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
    @test rec["session"][end]["status"] == "completed" && rec["config"]["run"]["model"] == "urban"
    res = TOML.parsefile(joinpath(run, "output", "results.toml"))
    led = res["ledgers"]
    @test res["finite"] === true
    @test led["closed"] === true
    @test led["surface_max_abs_closure"] < UC.LEDGER_TOLERANCES.surface
    @test abs(led["water_residual"]) < UC.LEDGER_TOLERANCES.water
    @test abs(led["air_heat_closure"]) <= UC.LEDGER_TOLERANCES.air_heat
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

@testset "urban run type: the GeoJSON ground, in memory and from its static driver, bit for bit" begin
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
    UC.run_case!(case)
    st = UC.case_state(case)
    @test all(a -> all(isfinite, a), values(st))
    # aeolus preprocess static writes the driver; the run from it is the same, bit for bit
    drv = joinpath(mktempdir(), "small_city_static.nc")
    @test quiet_main(["preprocess", "static", SMALL_CITY, "--out", drv]) == 0
    @test isfile(drv)
    case2 = UC.build_case(UC.CaseConfig(SMALL_CITY; static = drv))
    UC.run_case!(case2)
    st2 = UC.case_state(case2)
    @test keys(st) == keys(st2)
    @test all(digest(st[k]) == digest(st2[k]) for k in keys(st))
    # no ground source at all is an error, not a flat city
    @test_throws ErrorException UC.build_case(UC.CaseConfig(SMALL_CITY; osm_path = nothing))
    # run_results reads the state only: the digests do not move
    d0 = Dict(k => digest(v) for (k, v) in pairs(UC.case_state(case2)))
    r = UC.run_results(case2)
    @test Dict(k => digest(v) for (k, v) in pairs(UC.case_state(case2))) == d0
    @test r["ledgers"]["closed"] === true
end

@testset "urban_parent: the nest parent alone, its tables and results" begin
    toml = berlin_toml("parent")
    tt = Aeolus.Config.config_test_table(toml)
    cfg = load_run_config(toml; overrides = tt.overrides)
    @test cfg.run.model == "urban_parent"
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
# the preprocessors
# ---------------------------------------------------------------------------

@testset "aeolus preprocess osm-sample reproduces the committed synthetic OSM file" begin
    out = joinpath(mktempdir(), "sample.osm")
    @test quiet_main(["preprocess", "osm-sample", "--lat0", "52.5220", "--lon0", "13.4100", "--out", out]) == 0
    @test read(out) == read(joinpath(berlin_data_dir(), "sample_berlin_mitte.osm"))
    # a usage error exits 2 and writes nothing
    @test quiet_main(["preprocess", "osm-sample", "--lat0", "north", "--lon0", "13.41", "--out", out * "2"]) == 2
    @test !isfile(out * "2")
    @test quiet_main(["preprocess", "fetch-osm", "--out", out * "3"]) == 2
    @test quiet_main(["preprocess", "fetch-sounding", "--station", "X"]) == 2
end

"One fixed-width IGRA2 data record (igra2-data-format.txt)."
function igra_level(lvl, p, gph, temp, dpdp, wdir, wspd)
    c = fill(' ', 51)
    put!(s, a, b) = (t = lpad(s, b - a + 1); c[a:b] .= collect(t))
    put!(lvl, 1, 2); put!(string(p), 10, 15); put!("B", 16, 16); put!(string(gph), 17, 21)
    put!(string(temp), 23, 27); put!(string(dpdp), 35, 39); put!(string(wdir), 41, 45); put!(string(wspd), 47, 51)
    return String(c)
end
igra_header(y, m, d, h, rel) = @sprintf("#%-11s %4d %02d %02d %02d %04d", "TEST0000001", y, m, d, h, rel)

@testset "IGRA2 records to a sounding CSV, read back by read_sounding" begin
    dir = mktempdir()
    txt = joinpath(dir, "igra.txt")
    open(txt, "w") do io
        for (h, rel, dT) in ((0, 2245, 0), (12, 1045, 50))       # 22:45 of the day before; 10:45
            println(io, igra_header(2019, 6, (h == 0 ? 29 : 29), h, rel))
            println(io, igra_level("21", 101500, 112, 200 + dT, 50, 270, 30))     # surface
            println(io, igra_level("20", 102000, -9999, 210, 50, 270, 30))        # below the surface: dropped
            println(io, igra_level("10", 100000, 240, 190 + dT, 60, 280, 50))
            println(io, igra_level("10", 85000, 1500, 100 + dT, 80, 290, 80))
            println(io, igra_level("10", 70000, 3100, -20 + dT, 100, 300, 120))
            println(io, igra_level("20", 92500, -9999, -8888, -9999, -9999, -9999))  # QC-removed, no wind: dropped
        end
        println(io, igra_header(2019, 6, 30, 0, 9999))               # not asked for
        println(io, igra_level("21", 101400, 112, 180, 40, 250, 20))
    end
    out = joinpath(dir, "s.csv")
    n, dropped = UC.igra_sounding_csv(txt, out; launches = [(2019, 6, 29, 0), (2019, 6, 29, 12)],
                                      epoch = (2019, 6, 29), station = "TEST0000001", elevation = 112.0)
    @test n == 8
    @test dropped["below the surface (p > p_sfc)"] == 2 && dropped["neither temperature nor wind"] == 2
    rows = [split(l, ',') for l in readlines(out) if !startswith(l, "#") && !startswith(l, "launch")]
    @test length(rows) == 8
    # the 00 UTC launch was released at 22:45 the day before: t = -4500 s
    @test rows[1][3] == "-4500" && rows[5][3] == string(10 * 3600 + 45 * 60)
    @test parse(Float64, rows[1][9]) ≈ 20.0 + 273.15 && parse(Float64, rows[1][11]) ≈ 15.0 + 273.15
    @test parse(Float64, rows[5][9]) ≈ 25.0 + 273.15 && parse(Float64, rows[5][13]) ≈ 3.0
    s = read_sounding(out)
    @test length(s.launches) == 2 && s.elevation == 112.0 && s.epoch == (2019, 6, 29)
    @test s.launches[1].t == -4500.0 && s.launches[2].T[1] ≈ 298.15
    # a launch missing from the record is an error
    @test_throws ErrorException UC.igra_sounding_csv(txt, out; launches = [(2019, 7, 1, 0)], epoch = (2019, 6, 29),
                                                     station = "X", elevation = 0.0)
end

@testset "the parent land-use table of an OSM extract is self-consistent" begin
    # the smoke case's parent (1 km box, refinement 4 on 15.625 m cells: 16 × 16
    # cells of 62.5 m) over the synthetic Berlin-Mitte OSM sample
    cc = UC.CaseConfig(berlin_toml("smoke"))
    csv = joinpath(mktempdir(), "landuse.csv")
    UC.write_parent_landuse(csv, joinpath(berlin_data_dir(), "sample_berlin_mitte.osm"), cc; fetched = "test")
    pl = UC.read_parent_landuse(csv)          # its own checks: every cell once, counts sum to r², λp = n2/r²
    @test (pl.nx, pl.ny, pl.r, pl.dx) == (16, 16, 4, 62.5)
    @test all(sum(pl.counts; dims = 3) .== 16)
    roof = sum(pl.counts[:, :, 2]) / (16 * 16 * 16)
    @test 0.1 < roof < 0.6                    # perimeter blocks: a fifth to a half of the ground is roof
    @test maximum(pl.hmax) <= 90.0 + 1e-9 && maximum(pl.hmax) >= 10.0   # the sample's heights (10–90 m)
    @test all(pl.λf .>= 0)
    @test any(>(0), pl.counts[:, :, 3]) && any(>(0), pl.counts[:, :, 4])   # the park and the water
end
