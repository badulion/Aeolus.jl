# The dynamic driver (CFG-g, #362): Aeolus.InputDrivers — the initial
# state, the boundary planes over time (FileBoundaryData, FileBoundaryFeed),
# the large-scale and surface forcing; sounding, case-table and
# Aeolus-parent drivers. The preprocessors that write the sounding drivers
# are AeolusPrep.jl's since ONE-h (#457); they call Aeolus's writer
# (`Urban.write_sounding_driver`), which the tests here call too
# (`write_sounding_driver` below), and check what a run reads. The
# case-table drivers of the SBL configs are committed
# (examples/sbl/input/, ONE-g #456).
#
# Acceptance tests of the issue covered here:
#   * the time blend is exact for data linear in t (u = 1 + 0.01 t at
#     time_bc = 0, 60, 120 s, read back at t = 17, 60, 95 s), and for data
#     quadratic in t its error is the analytic bound Δt²/8 · max|u''|;
#   * memory stays bounded: over 300 records at most two are resident
#     (Base.summarysize of the feed);
#   * errors: a face missing for an open side, a time_bc that does not
#     cover the run (both intervals printed), a grid mismatch; a case
#     starting at another time reads the sounding driver as the CSV;
#   * the committed case-table drivers of the SBL configs hold the case
#     tables' forcing (GABLS3: the geostrophic wind and the advective
#     tendencies of gabls3_case.csv; GABLS2: the subsidence and the case's
#     surface-temperature formula), and the model reading them runs, its
#     driver hashed under [inputs], in a round trip;
#   * the initial state from the sounding driver (θ(z), qv(z), p_surface,
#     the reference state, the parent's geostrophic wind and z_i) is the
#     CSV path's on master within ROUNDOFF_RTOL (its fingerprints,
#     test/data/input_dynamic/fingerprints.toml, #495; the scalars in
#     master_digests.toml), and the smoke case
#     from a sounding driver matches the CSV path over 12 steps, in a
#     round trip with the driver hashed under [inputs];
#   * offline nest = online: the smoke child alone, 12 steps from the
#     driver its online parent wrote (`[output] child_boundaries`), is
#     the online one-way child within ROUNDOFF_RTOL, in a round trip with the driver
#     hashed under [inputs];
#   * the committed summer sounding driver is valid and holds the master
#     ground pressure (AeolusPrep.jl reproduces it, its test/recipes.jl).

using Test
using TOML
using Printf
using Aeolus
using Aeolus.InputDrivers
using Aeolus.InputDrivers: VOLATILE_ATTRIBUTES
using Aeolus.Config
using Aeolus.Grids: RegularGrid
using Aeolus.BoundaryConditions: open_boundary_conditions, open_boundary_states, boundary_values!,
                                 ProfileBoundaryData
import Aeolus.BoundaryConditions: snapshot_values
using Aeolus.LargeScaleForcing: ForcingTable, AdvectiveTendencies, SurfaceTemperatureSeries
using Aeolus.NCIO: NCDataset
using Aeolus.IO: read_run_toml
import NCDatasets

const BCD = Aeolus.Urban
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))
include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))
isdefined(Main, :gabls3_data) || include(joinpath(@__DIR__, "support", "sbl_data.jl"))

const ID_DATA = joinpath(@__DIR__, "data", "input_dynamic")
const ID_MASTER = TOML.parsefile(joinpath(ID_DATA, "master_digests.toml"))
const ID_SMOKE = joinpath(berlin_configs_dir(), "berlin_smoke.toml")
# the smoke site alone (no parent domain): the offline and the open cases
const ID_SMOKE_SITE = joinpath(berlin_configs_dir(), "berlin_smoke_site.toml")
const ID_CSV = normpath(joinpath(berlin_data_dir(), "lindenberg_10393_20190629.csv"))

"The sounding driver of the urban run `cfg` at `out` (`Urban.write_sounding_driver`, what `aeolusprep dynamic --from-sounding` calls)."
write_sounding_driver(cfg, out) = BCD.write_sounding_driver(BCD.site_options(site_view(cfg)), out; preprocessor = "test/input_dynamic.jl")


# A synthetic boundary snapshot: every field has the value `val` everywhere.
struct ConstSnapshot
    val::Float64
    dims::NTuple{3,Int}
end
snapshot_values(s::ConstSnapshot, ::Symbol) = fill(s.val, s.dims)

const SYN_MESH = RegularGrid(; size = (8, 6, 5), extent = (80.0, 60.0, 50.0), halo = (2, 2, 2))
syn_dims(m) = (m.Nx + 2m.Hx, m.Ny + 2m.Hy, m.Nz + 2m.Hz)
syn_planes(m; kw...) = open_boundary_states(open_boundary_conditions(m, ProfileBoundaryData(u = 1.0, v = 0.0, θ = 300.0);
                                                                     kw...))[1].planes

"A driver of the planes of `m`'s open box with every field `f(t)` at the record times `times`."
function syn_driver(path, f, times; m = SYN_MESH, planes = syn_planes(m))
    rec = NestBoundaryRecorder(path, planes, m; init_time = "2019-06-29T10:00:00Z")
    for (n, t) in enumerate(times)
        record_boundary!(rec, t, n == 1 ? NaN : t - times[n-1], ConstSnapshot(f(t), syn_dims(m)))
    end
    return rec
end

"The largest |plane value − v| over the non-mirror planes, in units of eps(v)."
plane_ulps(planes, v) = maximum(maximum(abs, p.data .- v) for p in planes if p.mode !== :mirror) / eps(v)

errmsg(f) = try
    f(); ""
catch e
    sprint(showerror, e)
end

@testset "InputDrivers dynamic: the time blend of FileBoundaryData" begin
    dir = mktempdir()
    # linear in t: exact (to the last bit of the blend's two products)
    lin(t) = 1 + 0.01t
    syn_driver(joinpath(dir, "lin.nc"), lin, [0.0, 60.0, 120.0])
    d = read_dynamic_driver(joinpath(dir, "lin.nc"); mesh = SYN_MESH)
    @test d.boundary isa FileBoundaryFeed && d.boundary.data isa FileBoundaryData
    @test get(d.attributes, "dynamic_source", "") == "parent" && d.init_time == "2019-06-29T10:00:00Z"
    planes = syn_planes(SYN_MESH)
    check_planes(d.boundary, planes)
    ulps = Float64[]
    for t in (17.0, 60.0, 95.0)
        advance_feed!(d.boundary, t)
        boundary_values!(planes, d.boundary.data, t)
        push!(ulps, plane_ulps(planes, lin(t)))
    end
    @test all(<=(1), ulps)
    @test d.boundary.data.t0 == 60.0 && d.boundary.data.Δt == 60.0
    # quadratic in t: the error of the linear blend is at most Δt²/8 · max|u''|,
    # reached at the window midpoints
    a = 1e-4
    quad(t) = 1 + a * t^2
    syn_driver(joinpath(dir, "quad.nc"), quad, [0.0, 60.0, 120.0])
    q = read_dynamic_driver(joinpath(dir, "quad.nc"); mesh = SYN_MESH)
    errs = Float64[]
    for t in 0.0:1.0:120.0
        advance_feed!(q.boundary, t)
        boundary_values!(planes, q.boundary.data, t)
        push!(errs, maximum(maximum(abs, p.data .- quad(t)) for p in planes))
    end
    bound = 60.0^2 / 8 * 2a
    emax, tmax = findmax(errs)
    @test emax <= bound * (1 + 1e-12)
    @test emax >= bound * (1 - 1e-9)                 # the bound is attained (the midpoints)
    @test (tmax - 1) in (30.0, 90.0)
    # a one-record file is steady: Δt = 0, covering every time
    syn_driver(joinpath(dir, "steady.nc"), _ -> 2.5, [0.0])
    s = read_dynamic_driver(joinpath(dir, "steady.nc"); mesh = SYN_MESH)
    @test feed_coverage(s.boundary) == (-Inf, Inf) && s.boundary.data.Δt == 0
    boundary_values!(planes, s.boundary.data, 1e6)
    @test plane_ulps(planes, 2.5) == 0
end

# A synthetic boundary snapshot whose values differ at every storage point,
# field and time (all exactly representable), so a slab read at a wrong
# offset or from the wrong face shows.
struct PosSnapshot
    t::Float64
    dims::NTuple{3,Int}
end
pos_value(name, i, j, k, t) = i + 100.0j + 10000.0k + 0.25findfirst(==(name), (:u, :v, :w, :θ, :qv)) + 1e6 + 1e3t
snapshot_values(s::PosSnapshot, name::Symbol) =
    [pos_value(name, i, j, k, s.t) for i in 1:s.dims[1], j in 1:s.dims[2], k in 1:s.dims[3]]

@testset "InputDrivers dynamic: the time blend reads each plane's slab directly (#529)" begin
    path = joinpath(mktempdir(), "pos.nc")
    planes = syn_planes(SYN_MESH)
    rec = NestBoundaryRecorder(path, planes, SYN_MESH; init_time = "2019-06-29T10:00:00Z")
    for (n, t) in enumerate((0.0, 60.0))
        record_boundary!(rec, t, n == 1 ? NaN : 60.0, PosSnapshot(t, syn_dims(SYN_MESH)))
    end
    d = read_dynamic_driver(path; mesh = SYN_MESH)
    t = 25.0
    advance_feed!(d.boundary, t)
    data = d.boundary.data
    open_planes = [p for p in planes if p.mode !== :mirror]
    @test length(open_planes) >= 8
    # every plane's box lies in one slab: the blend reads that dense array,
    # and at its offset it holds exactly the values getindex finds point by point
    for p in open_planes
        r = (p.irange, p.jrange, p.krange)
        for snap in (data.a, data.b)
            A = snapshot_values(snap, p.name)
            src, o = Aeolus.BoundaryConditions.blend_source(A, r)
            @test src isa Array{Float64,3}
            @test all(src[i - o[1], j - o[2], k - o[3]] == A[i, j, k] for i in r[1], j in r[2], k in r[3])
        end
    end
    # the blend: the same two products on the same copied values as the
    # point-by-point read, so the planes agree exactly
    τ = window_fraction(data, t)
    boundary_values!(planes, data, t)
    for p in open_planes
        A = snapshot_values(data.a, p.name); B = snapshot_values(data.b, p.name)
        want = [(1 - τ) * A[i, j, k] + τ * B[i, j, k] for i in p.irange, j in p.jrange, k in p.krange]
        @test p.data == want
        # and it is the time blend of the recorded field (0.5 ulp per product)
        @test maximum(abs, p.data .- [(1 - τ) * pos_value(p.name, i, j, k, 0.0) + τ * pos_value(p.name, i, j, k, 60.0)
                                      for i in p.irange, j in p.jrange, k in p.krange]) <= 2eps(2e6)
    end
    # a box no single slab holds (the whole array) keeps the point-by-point view
    A = snapshot_values(data.a, :u)
    src, o = Aeolus.BoundaryConditions.blend_source(A, map(n -> 1:n, size(A)))
    @test src === A && o == (0, 0, 0)
end

@testset "InputDrivers dynamic: at most two records resident over 300" begin
    dir = mktempdir()
    n = 300
    times = collect(0.0:2.0:2.0 * (n - 1))
    rec = syn_driver(joinpath(dir, "long.nc"), t -> 300 + 1e-3t, times)
    @test rec.records == n
    d = read_dynamic_driver(joinpath(dir, "long.nc"); mesh = SYN_MESH)
    f = d.boundary
    record_bytes = sum(8 * prod(length.(l.ranges)) for l in f.layout)
    fixed = Base.summarysize(f.time) + Base.summarysize(f.step) + Base.summarysize(f.layout)
    sizes = Int[]
    for t in times[1:end-1] .+ 1.0
        advance_feed!(f, t)
        push!(sizes, Base.summarysize(f))
    end
    @test f.reads == n                                # every record read once, in order
    @test f.k == n - 1
    resident = (maximum(sizes) - fixed) / record_bytes
    @test resident < 2.5                              # two records and their containers
    @test maximum(sizes) == minimum(sizes)            # nothing accumulates
    # the values are those of the window's records
    boundary_values!(syn_planes(SYN_MESH), f.data, times[end] - 1.0)
    @test f.data.a !== f.data.b && f.data.t0 == times[end-1]
end

@testset "InputDrivers dynamic: errors" begin
    dir = mktempdir()
    # a face missing for an open side: the file has open x faces, the run open y faces too
    p = joinpath(dir, "xonly.nc")
    syn_driver(p, _ -> 1.0, [0.0, 60.0]; planes = syn_planes(SYN_MESH; y = :periodic))
    d = read_dynamic_driver(p; mesh = SYN_MESH)
    msg = errmsg(() -> check_planes(d.boundary, syn_planes(SYN_MESH)))
    @test occursin("no boundary data for", msg) && occursin("south face", msg)
    # the run interval not covered: both intervals in the message
    msg = errmsg(() -> check_coverage(d.boundary, 0.0, 200.0))
    @test occursin("[0.0, 60.0]", msg) && occursin("[0.0, 200.0]", msg)
    check_coverage(d.boundary, 0.0, 60.0)             # covered: no error
    # a grid mismatch: another resolution, and another halo (slabs elsewhere)
    m2 = RegularGrid(; size = (8, 6, 5), extent = (96.0, 60.0, 50.0), halo = (2, 2, 2))
    msg = errmsg(() -> read_dynamic_driver(p; mesh = m2))
    @test occursin("not on the run's grid", msg) && occursin("file:", msg) && occursin("run:", msg)
    m3 = RegularGrid(; size = (8, 6, 5), extent = (80.0, 60.0, 50.0), halo = (3, 3, 3))
    d3 = read_dynamic_driver(p; mesh = m3)
    msg = errmsg(() -> check_planes(d3.boundary, syn_planes(m3; y = :periodic)))
    @test occursin("covers storage indices", msg)
    @test occursin("read it with the run's mesh", errmsg(() -> read_dynamic_driver(p)))
    # a static driver is not a dynamic one
    st = joinpath(berlin_configs_dir(), "input", "berlin_smoke_static.nc")
    @test occursin("aeolus_driver = \"static\", expected \"dynamic\"", errmsg(() -> read_dynamic_driver(st)))
    # a case starting at another time than the sounding driver (08 instead
    # of 06 CEST): the same series with the axis moved and the ground
    # pressure at its own start, the CSV read in memory (within ROUNDOFF_RTOL)
    a = BCD.case_sounding(site_config(berlin_toml("summer"); start_h = 8.0))
    b = BCD.case_sounding(site_config(berlin_toml("summer"); start_h = 8.0, dynamic = nothing))
    @test a.inputs.init_time == "2019-06-29T04:00:00Z" && b.inputs.init_time == "2019-06-29T06:00:00Z"
    @test a.t0 == b.t0 == 6 * 3600.0 && agree(a.p_surface, b.p_surface) && a.times == b.times
    @test all(k -> (x = getproperty(a.profiles, k); y = getproperty(b.profiles, k);
                    agree(x.z, y.z) && agree(x.t, y.t) && agree(x.values, y.values)), (:θ, :qv, :u, :v, :p))
    @test a.p_surface != BCD.case_sounding(site_config(berlin_toml("summer"))).p_surface
    # the parent-made driver offered as the initial state of a case without the offline nest
    msg = errmsg(() -> BCD.case_sounding(site_config(berlin_toml("summer"); dynamic = p)))
    @test occursin("not a sounding driver", msg)
end

@testset "InputDrivers dynamic: the schema-only groups round-trip (init_* profiles, surface series)" begin
    # No preprocessor fills these groups yet: the writer and the reader
    # must still agree on them, value for value, and the file must pass
    # its schema.
    dir = mktempdir()
    p = joinpath(dir, "profiles.nc")
    z = [0.0, 10.0, 50.0, 200.0, 1000.0]
    prof = (z = z, theta = 290.0 .+ 0.003 .* z, qv = 8e-3 .* exp.(-z ./ 2000), u = 1.0 .+ z ./ 100, v = nothing)
    sfc = (t = [0.0, 1800.0, 3600.0], temperature = [289.1, 290.3, 291.7], pressure = [101325.0, 101300.0, 101280.0])
    write_dynamic_driver(p, DynamicInputs(; init_time = "2019-06-29T04:00:00Z", profiles = prof, p_surface = 100870.5,
                                          surface = sfc))
    @test isempty(check_driver(p, DYNAMIC_SCHEMA))
    d = read_dynamic_driver(p)
    @test d.init_time == "2019-06-29T04:00:00Z" && d.p_surface === 100870.5
    @test d.profiles.z == z && d.profiles.theta == prof.theta && d.profiles.qv == prof.qv && d.profiles.u == prof.u
    @test d.profiles.v === nothing
    @test d.surface.t == sfc.t && d.surface.temperature == sfc.temperature && d.surface.pressure == sfc.pressure
    @test isempty(d.fields) && isempty(d.tables) && d.boundary === nothing
    # a surface series with one variable: the other reads back as nothing
    p2 = joinpath(dir, "sfc_p.nc")
    write_dynamic_driver(p2, DynamicInputs(; surface = (t = sfc.t, temperature = nothing, pressure = sfc.pressure)))
    d2 = read_dynamic_driver(p2)
    @test d2.profiles === nothing && d2.surface.temperature === nothing && d2.surface.pressure == sfc.pressure
    # out of the schema's range: init_theta in °C is caught by the reader
    p3 = joinpath(dir, "celsius.nc")
    write_dynamic_driver(p3, DynamicInputs(; profiles = (z = z, theta = prof.theta .- 273.15, qv = nothing,
                                                         u = nothing, v = nothing)))
    @test !isempty(check_driver(p3, DYNAMIC_SCHEMA))
    @test_throws Exception read_dynamic_driver(p3)
end

@testset "InputDrivers dynamic: the committed SBL case-table drivers" begin
    drv(n) = joinpath(SBL_CONFIGS, "input", n)
    keys_of = Dict("gabls2_dynamic.nc" => [:w_subs], "gabls3_dynamic.nc" => [:theta_adv, :u_adv, :u_g, :v_adv, :v_g],
                   "gabls4_exp3_dynamic.nc" => Symbol[])
    table_ok((n, ks)) = (d = read_dynamic_driver(drv(n); planes = false);
        isempty(check_driver(drv(n), DYNAMIC_SCHEMA)) && sort!(collect(keys(d.tables))) == ks &&
        d.attributes["dynamic_source"] == "case_table" && d.init_time == "" && d.profiles === nothing &&
        d.surface !== nothing && length(d.surface.t) > 1)
    @test all(table_ok, keys_of)
    # GABLS3: the tables are the case table's columns (times since 00 UTC 2 July = the run start)
    g = gabls3_data()
    F = dynamic_forcing(read_dynamic_driver(drv("gabls3_dynamic.nc"); planes = false))
    @test all(r -> (t = g.geostrophic[r, 1];
                    forcing_value(F.U_g, 0.0, t) == g.geostrophic[r, 2] && forcing_value(F.V_g, 0.0, t) == g.geostrophic[r, 3] &&
                    forcing_value(F.U_g, 2000.0, t) == -2.0 && forcing_value(F.V_g, 2000.0, t) == 2.0),
              1:size(g.geostrophic, 1))
    A = only(F.extra)
    @test A isa AdvectiveTendencies && A.q === nothing
    @test all(r -> (t = g.advection[r, 1];
                    forcing_value(A.u, 300.0, t) == g.advection[r, 2] && forcing_value(A.θ, 300.0, t) == g.advection[r, 4] &&
                    forcing_value(A.u, 0.0, t) == 0.0), 1:size(g.advection, 1))
    # GABLS2: the subsidence from 40 h local (24 h after the 16 LT start) and
    # the case's surface-temperature formula, as θ with p_s = 972 hPa
    d2 = read_dynamic_driver(drv("gabls2_dynamic.nc"); planes = false)
    S = only(dynamic_forcing(d2; t0 = 57600.0).extra)
    @test S isa Subsidence && S.fields == (:θ,) && S.form === :mean_gradient
    @test forcing_value(S.w_s, 500.0, 143999.0) == 0.0 && forcing_value(S.w_s, 500.0, 150000.0) ≈ -0.0025
    @test d2.surface.temperature ≈ [(gabls2_surface_temperature_C((t + 57600) / 3600) + 273.15) * (1e5 / 97200)^0.286
                                    for t in d2.surface.t] rtol = 1e-14
    # the model reading the GABLS3 driver: a round trip, the driver hashed under [inputs]
    # the model's config hashes the GABLS3 driver under [inputs] (the run itself: validation/gabls3.jl)
    inp = config_inputs(load_run_config(joinpath(SBL_CONFIGS, "gabls3.toml")))
    @test inp["domain.main.dynamic_driver"]["fnv1a64"] == content_hash(drv("gabls3_dynamic.nc"))
end

@testset "InputDrivers dynamic: the committed summer sounding driver" begin
    cc = site_config(berlin_toml("summer"))
    @test filesize(cc.dynamic) <= 2 * 1024^2
    @test isempty(check_driver(cc.dynamic, DYNAMIC_SCHEMA))
    NCDataset(cc.dynamic) do ds
        @test agree(ds["init_p_surface"].var[], ID_MASTER["summer_p_surface"])
        # made by AeolusPrep (ONE-h, #457) from the CSV in Aeolus, hashed
        @test ds.attrib["preprocessor"] == "aeolusprep dynamic --from-sounding"
        @test ds.attrib["source"] == "../data/lindenberg_10393_20190629.csv"
    end
    # a driver written now from the CSV holds the committed tables, value for value
    out = joinpath(mktempdir(), "summer.nc")
    write_sounding_driver(load_run_config(berlin_toml("summer")), out)
    a, b = read_dynamic_driver(out; planes = false), read_dynamic_driver(cc.dynamic; planes = false)
    @test keys(a.tables) == keys(b.tables) && all(k -> isequal(a.tables[k].values, b.tables[k].values), keys(a.tables))
    @test a.init_time == b.init_time && a.p_surface == b.p_surface
end

@testset "InputDrivers dynamic: a missing driver is a config error naming the command that makes it" begin
    dir = mktempdir()
    msg = errmsg(() -> Config.build(Aeolus.SimulationDriver.SimulationDriverType(),
                                    load_run_config(ID_SMOKE; overrides = ["domain.site.dynamic_driver=\"$(joinpath(dir, "none.nc"))\""])))
    @test occursin("AeolusPrep dynamic --from-sounding $ID_SMOKE", msg)
    # the offline nest: the online run writes it
    msg = errmsg(() -> Config.build(Aeolus.SimulationDriver.SimulationDriverType(),
                                    load_run_config(ID_SMOKE_SITE; overrides = ["dynamic_driver=\"$(joinpath(dir, "none.nc"))\"",
                                                                                "nesting.enabled=true",
                                                                                "nesting.parent_driver=\"offline\""])))
    @test occursin("child_boundaries", msg)
end

# ---------------------------------------------------------------------------
# the open simulation box over Lindenberg from its committed ERA5 driver
# (CFG-h, #363). The ERA5 preprocessor and its analytic tests are in
# AeolusPrep.jl since ONE-h (#457); what stays here is what the model does
# with the file it reads.
# ---------------------------------------------------------------------------

const E5_CFG = joinpath(pkgdir(Aeolus), "examples", "lindenberg_era5", "lindenberg_era5.toml")
const E5_DRIVER = joinpath(pkgdir(Aeolus), "examples", "lindenberg_era5", "input", "lindenberg_era5_dynamic.nc")

@testset "meters_to_latlon inverts latlon_to_meters" begin
    lat0, lon0 = 52.2085, 14.118
    @test all(((la, lo),) -> (xy = latlon_to_meters(la, lo, lat0, lon0); (la2, lo2) = meters_to_latlon(xy..., lat0, lon0);
                              abs(la2 - la) < 1e-12 && abs(lo2 - lo) < 1e-12),
              ((52.6, 13.4), (51.3, 15.9), (52.21, 14.12)))
    # 1 km north is 1000/R radians of latitude
    @test meters_to_latlon(0.0, 1000.0, lat0, lon0)[1] - lat0 ≈ rad2deg(1000.0 / 6_371_000.0) rtol = 1e-9
end

@testset "ERA5 driver: the committed driver of the open simulation box" begin
    @test filesize(E5_DRIVER) <= 2 * 1024^2                 # committed: 2 MB or smaller
    @test isempty(check_driver(E5_DRIVER, DYNAMIC_SCHEMA))
    d = read_dynamic_driver(E5_DRIVER; planes = false)
    @test d.attributes["dynamic_source"] == "era5" && d.attributes["origin_lat"] == 52.2085
    @test occursin("era5_lindenberg_20190629_sfc.nc", d.attributes["era5_files"])
    @test d.init_time == "2019-06-29T00:00:00Z" && size(d.tables[:profile_theta].values) == (16, 13)
    # the config names the committed driver, hashed under [inputs] (the run: validation/)
    cfg = load_run_config(E5_CFG)
    @test cfg.domain.dynamic_driver == E5_DRIVER                                # the domain's (ONE-b)
    @test config_inputs(cfg)["domain.main.dynamic_driver"]["fnv1a64"] == content_hash(E5_DRIVER)
    # its boundary planes are on the run's grid: the record blend at t = 12 s
    # is (1 − τ) A + τ B of the two hourly records
    m = Aeolus.SimulationDriver._mesh(cfg.grid)
    f = read_dynamic_driver(E5_DRIVER; mesh = m).boundary
    @test f isa FileBoundaryFeed && f.data.t0 == 0.0 && f.data.Δt == 3600.0
end

@testset "ERA5 driver: the model's config checks" begin
    err(ov) = try
        Config.build(Aeolus.SimulationDriver.SimulationDriverType(), load_run_config(E5_CFG; overrides = ov)); nothing
    catch e
        e isa ConfigError || rethrow(); e
    end
    e1 = err(["dynamic_driver=\"none\""])
    @test e1 !== nothing && occursin("set the domain's dynamic_driver", sprint(showerror, e1))
    e2 = err(["initial.theta={kind=\"constant\", value=300.0}"])
    @test e2 !== nothing && occursin("give one of the two", sprint(showerror, e2))
    # the preprocessor's table left the model's config with the preprocessor
    e3 = try
        load_run_config(E5_CFG; overrides = ["preprocess.era5={lat=52.2, lon=14.1}"]); nothing
    catch e
        e isa ConfigError || rethrow(); e
    end
    @test e3 !== nothing && occursin("preprocess", sprint(showerror, e3))
end
