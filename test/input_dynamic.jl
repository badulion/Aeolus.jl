# The dynamic driver (CFG-g, #362): Aeolus.InputDrivers — the initial
# state, the boundary planes over time (FileBoundaryData, FileBoundaryFeed),
# the large-scale and surface forcing; sounding, case-table and
# Aeolus-parent drivers. The preprocessors that write the sounding drivers
# are AeolusPrep.jl's since ONE-h (#457); they call Aeolus's writer
# (`UrbanCase.write_sounding_driver`), which the tests here call too
# (`write_sounding_driver` below), and check what a run reads. The
# case-table drivers of the SBL configs are committed
# (examples/sbl/input/, ONE-g #456).
#
# Acceptance tests of the issue covered here:
#   * the time blend is exact for data linear in t (u = 1 + 0.01 t at
#     time_bc = 0, 60, 120 s, read back at t = 17, 60, 95 s), and for data
#     quadratic in t its error is the analytic bound Δt²/8 · max|u''|;
#   * memory stays bounded: over 1000 records at most two are resident
#     (Base.summarysize of the feed);
#   * errors: a face missing for an open side, a time_bc that does not
#     cover the run (both intervals printed), a grid mismatch; a case
#     starting at another time reads the sounding driver bitwise as the CSV;
#   * the committed case-table drivers of the SBL configs hold the case
#     tables' forcing (GABLS3: the geostrophic wind and the advective
#     tendencies of gabls3_case.csv; GABLS2: the subsidence and the case's
#     surface-temperature formula), and the model reading them runs, its
#     driver hashed under [inputs], in a bitwise round trip;
#   * the initial state from the sounding driver (θ(z), qv(z), p_surface,
#     the reference state, the parent's geostrophic wind and z_i) is
#     bitwise the CSV path on master (digests captured on master,
#     test/data/input_dynamic/master_digests.toml), and the smoke case
#     from a sounding driver matches the CSV path over 12 steps, in a
#     round trip with the driver hashed under [inputs];
#   * offline nest = online: the smoke child alone, 12 steps from the
#     driver its online parent wrote (`[output] child_boundaries`), is
#     bitwise the online one-way child, in a round trip with the driver
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

const BCD = Aeolus.UrbanCase
include(joinpath(@__DIR__, "support", "berlin_configs.jl"))
include(joinpath(@__DIR__, "support", "berlin_fixture.jl"))
include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))
isdefined(Main, :gabls3_data) || include(joinpath(@__DIR__, "support", "sbl_data.jl"))

const ID_DATA = joinpath(@__DIR__, "data", "input_dynamic")
const ID_MASTER = TOML.parsefile(joinpath(ID_DATA, "master_digests.toml"))
const ID_SMOKE = joinpath(berlin_configs_dir(), "berlin_smoke.toml")
const ID_CSV = normpath(joinpath(berlin_data_dir(), "lindenberg_10393_20190629.csv"))

"The sounding driver of the urban run `cfg` at `out` (`UrbanCase.write_sounding_driver`, what `aeolusprep dynamic --from-sounding` calls)."
write_sounding_driver(cfg, out) = BCD.write_sounding_driver(BCD.CaseConfig(cfg), out; preprocessor = "test/input_dynamic.jl")


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
    @info "linear boundary data u = 1 + 0.01 t: blend error at t = 17, 60, 95 s: $(ulps) ulp"
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
    @info @sprintf("quadratic boundary data u = 1 + %.0e t²: max blend error %.6e at t = %.0f s, bound Δt²/8·max|u''| = %.6e",
                   a, emax, tmax - 1, bound)
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

@testset "InputDrivers dynamic: at most two records resident over 1000" begin
    dir = mktempdir()
    n = 1000
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
    @info @sprintf("1000 boundary records of %d B each: feed %d–%d B, of which %d B the time axes and layout: %.3f records resident",
                   record_bytes, minimum(sizes), maximum(sizes), fixed, resident)
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
    # pressure at its own start, bitwise the CSV read in memory
    a = BCD.case_sounding(BCD.CaseConfig(berlin_toml("summer"); start_h = 8.0))
    b = BCD.case_sounding(BCD.CaseConfig(berlin_toml("summer"); start_h = 8.0, dynamic = nothing))
    @test a.inputs.init_time == "2019-06-29T04:00:00Z" && b.inputs.init_time == "2019-06-29T06:00:00Z"
    @test a.t0 == b.t0 == 6 * 3600.0 && isequal(a.p_surface, b.p_surface) && a.times == b.times
    @test all(k -> (x = getproperty(a.profiles, k); y = getproperty(b.profiles, k);
                    x.z == y.z && x.t == y.t && isequal(x.values, y.values)), (:θ, :qv, :u, :v, :p))
    @test a.p_surface != BCD.case_sounding(BCD.CaseConfig(berlin_toml("summer"))).p_surface
    # the parent-made driver offered as the initial state of a case without the offline nest
    msg = errmsg(() -> BCD.case_sounding(BCD.CaseConfig(berlin_toml("summer"); dynamic = p)))
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
    for (n, ks) in keys_of
        @test isempty(check_driver(drv(n), DYNAMIC_SCHEMA))
        d = read_dynamic_driver(drv(n); planes = false)
        @test sort!(collect(keys(d.tables))) == ks
        @test d.attributes["dynamic_source"] == "case_table" && d.init_time == "" && d.profiles === nothing
        @test d.surface !== nothing && length(d.surface.t) > 1
    end
    # GABLS3: the tables are the case table's columns (times since 00 UTC 2 July = the run start)
    g = gabls3_data()
    F = dynamic_forcing(read_dynamic_driver(drv("gabls3_dynamic.nc"); planes = false))
    for r in 1:size(g.geostrophic, 1)
        t = g.geostrophic[r, 1]
        @test forcing_value(F.U_g, 0.0, t) == g.geostrophic[r, 2] && forcing_value(F.V_g, 0.0, t) == g.geostrophic[r, 3]
        @test forcing_value(F.U_g, 2000.0, t) == -2.0 && forcing_value(F.V_g, 2000.0, t) == 2.0
    end
    A = only(F.extra)
    @test A isa AdvectiveTendencies && A.q === nothing
    for r in 1:size(g.advection, 1)
        t = g.advection[r, 1]
        @test forcing_value(A.u, 300.0, t) == g.advection[r, 2] && forcing_value(A.θ, 300.0, t) == g.advection[r, 4]
        @test forcing_value(A.u, 0.0, t) == 0.0
    end
    # GABLS2: the subsidence from 40 h local (24 h after the 16 LT start) and
    # the case's surface-temperature formula, as θ with p_s = 972 hPa
    d2 = read_dynamic_driver(drv("gabls2_dynamic.nc"); planes = false)
    S = only(dynamic_forcing(d2; t0 = 57600.0).extra)
    @test S isa Subsidence && S.fields == (:θ,) && S.form === :mean_gradient
    @test forcing_value(S.w_s, 500.0, 143999.0) == 0.0 && forcing_value(S.w_s, 500.0, 150000.0) ≈ -0.0025
    @test d2.surface.temperature ≈ [(gabls2_surface_temperature_C((t + 57600) / 3600) + 273.15) * (1e5 / 97200)^0.286
                                    for t in d2.surface.t] rtol = 1e-14
    # the model reading the GABLS3 driver: a bitwise round trip, the driver hashed under [inputs]
    r = roundtrip_check(joinpath(SBL_CONFIGS, "gabls3.toml"); nsteps = 20,
                        overrides = ["grid.size=[16,16,48]", "grid.extent=[400.0,400.0,400.0]", "sponge.z_start=300.0",
                                     "checks.reference={}", "checks.reference_file=\"none\""])
    @test r.fields_equal && r.config_equal
    inp = read_run_toml(r.runs[1].folder.path)["inputs"]
    @test inp["domain.main.dynamic_driver"]["fnv1a64"] == content_hash(drv("gabls3_dynamic.nc"))
end

@testset "InputDrivers dynamic: the Berlin initial state from the sounding driver = the CSV path on master" begin
    M = ID_MASTER
    function initial_digests(cfg)
        mesh = RegularGrid(; size = cfg.size, extent = cfg.extent, halo = cfg.halo)
        zc = [mesh.z0 + (k - 0.5) * mesh.dz for k in 1:mesh.Nz]
        zf = collect(range(0.0, 4000.0; length = 4001))
        θ0, q0 = BCD.initial_θ(cfg), BCD.initial_qv(cfg)
        ref = BCD.build_reference(cfg, mesh)
        Ug, Vg = BCD.parent_geostrophic(BCD.ParentConfig(cfg))
        tz = [(z, t) for z in zf[1:40:end], t in (0.0, 17.3, 3600.0, 40000.5)]
        return Dict{String,Any}("summer_theta_zc" => digest([θ0(0.0, 0.0, z) for z in zc]),
            "summer_qv_zc" => digest([q0(0.0, 0.0, z) for z in zc]),
            "summer_theta_fine" => digest([θ0(0.0, 0.0, z) for z in zf]),
            "summer_qv_fine" => digest([q0(0.0, 0.0, z) for z in zf]),
            "summer_p_surface" => BCD.case_surface_pressure(cfg),
            "summer_ref_p" => digest(ref.p̄), "summer_ref_rho" => digest(ref.ρ̄),
            "summer_ref_T" => digest(ref.T̄), "summer_ref_pi" => digest(ref.π̄),
            "summer_zi" => BCD.initial_inversion_height(cfg), "summer_zi_max" => BCD.sounding_zi_max(cfg),
            "summer_ug" => digest([Ug(z, t) for (z, t) in tz]), "summer_vg" => digest([Vg(z, t) for (z, t) in tz]))
    end
    file = BCD.CaseConfig(berlin_toml("summer"))
    @test file.dynamic == joinpath(berlin_configs_dir(), "input", "berlin_20190629_dynamic.nc")
    @test BCD.case_sounding(file).inputs.attributes["dynamic_source"] == "sounding"
    for (label, cfg) in (("sounding driver", file), ("CSV in memory", BCD.CaseConfig(berlin_toml("summer"); dynamic = nothing)))
        got = initial_digests(cfg)
        bad = sort!([k for (k, v) in got if !isequal(M[k], v)])
        isempty(bad) || @info "initial state from the $label differs from master" bad
        @test isempty(bad)
    end
    @info @sprintf("summer initial state from the sounding driver: p_surface %.6f Pa, z_i %.0f m, max sounding z_i %.1f m (bitwise master)",
                   M["summer_p_surface"], M["summer_zi"], M["summer_zi_max"])
end

@testset "InputDrivers dynamic: the committed summer sounding driver" begin
    cc = BCD.CaseConfig(berlin_toml("summer"))
    @test filesize(cc.dynamic) <= 2 * 1024^2
    @test isempty(check_driver(cc.dynamic, DYNAMIC_SCHEMA))
    NCDataset(cc.dynamic) do ds
        @test ds["init_p_surface"].var[] == ID_MASTER["summer_p_surface"]
        # made by AeolusPrep (ONE-h, #457) from the CSV in Aeolus, hashed
        @test ds.attrib["preprocessor"] == "aeolusprep dynamic --from-sounding"
        @test ds.attrib["source"] == "../../data/lindenberg_10393_20190629.csv"
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
    msg = errmsg(() -> Config.build(BCD.UrbanDriver(),
                                    load_run_config(ID_SMOKE; overrides = ["initial.dynamic=\"$(joinpath(dir, "none.nc"))\""])))
    @test occursin("AeolusPrep dynamic --from-sounding $ID_SMOKE", msg)
    # the offline nest: the online run writes it
    msg = errmsg(() -> Config.build(BCD.UrbanDriver(),
                                    load_run_config(ID_SMOKE; overrides = ["initial.dynamic=\"$(joinpath(dir, "none.nc"))\"",
                                                                           "physics.options.nesting.parent_driver=\"offline\""])))
    @test occursin("child_boundaries", msg)
end

@testset "InputDrivers dynamic: the smoke case from a sounding driver = the CSV path (12 steps, round trip)" begin
    dir = mktempdir()
    out = joinpath(dir, "smoke_sounding.nc")
    write_sounding_driver(load_run_config(ID_SMOKE; overrides = ["initial.sounding=\"$ID_CSV\""]), out)
    r = roundtrip_check(ID_SMOKE; nsteps = 12, nsteps_overrides = n -> ["clock.nsteps=$n"],
                        overrides = ["initial.dynamic=\"$out\""])
    @test r.fields_equal && r.diagnostics_equal && r.config_equal
    rec = read_run_toml(r.runs[1].folder.path)
    @test rec["inputs"]["initial.dynamic"]["fnv1a64"] == content_hash(out)
    case = r.runs[1].obj.case
    @test BCD.case_sounding(case.cfg) !== nothing && case.cfg.sounding === nothing
    # the CSV path, run the same way in this process. The 12-step state is
    # compared here rather than against digests captured on master: the CSV
    # path itself reproduced those digests in some processes and missed them
    # in the last bits in others (3 of 12 fresh processes on the same code;
    # cause not identified, see the PR), while the initial state above
    # matched every time.
    csv = load_run_config(ID_SMOKE; overrides = ["initial.sounding=\"$ID_CSV\"", "clock.nsteps=12",
                                                 "fftw.planner=\"estimate\""])
    ref = run_config(csv; root = mktempdir(), console = false, repo = mktempdir()).obj.case
    want, got = berlin_state(ref), berlin_state(case)
    @test sort(collect(keys(got))) == sort(collect(keys(want))) == sort(Symbol.(keys(ID_MASTER["smoke_sounding_state"])))
    bad = sort!([k for k in keys(want) if !isequal(got[k], want[k])])
    isempty(bad) || @info "smoke from the sounding driver: fields differing from the CSV path" bad
    @test isempty(bad) && case.clock.n == ref.clock.n && case.sim.t == ref.sim.t
    @info @sprintf("smoke from the sounding driver: %d steps to t = %.1f s, bitwise the CSV path in %d fields",
                   case.clock.n, case.sim.t, length(want))
end

@testset "InputDrivers dynamic: offline nest = online one-way nest (smoke, 12 steps, round trip)" begin
    dir = mktempdir()
    drv = joinpath(dir, "smoke_child_boundaries.nc")
    # the planner of the round trip below (ESTIMATE): the online runs built
    # here and the offline child plan their FFTs alike
    p0 = Aeolus.Operators.fftw_policy()
    Aeolus.Operators.set_fftw_policy!(; planner = :estimate)
    try
        # the online run, recording the child's boundaries at every parent step.
        # The synthetic inflow turbulence (#268) is off: its perturbations are
        # drawn from the running parent's statistics and are not part of the
        # driver, so an offline child cannot replay them (it would add its own)
        t0 = time()
        nogen = Dict{Symbol,Any}(:inflow_turbulence => false)
        online = BCD.build_case(BCD.CaseConfig(berlin_toml("smoke"); output_path = nothing, child_boundaries = drv,
                                               physics = nogen))
        BCD.run_case!(online)
        t_online = time() - t0
        nc = BCD.find_coupling(online, :nesting)
        st_on = berlin_state(online)
        # recording does not change the run: the same case without the recorder
        plain = BCD.build_case(BCD.CaseConfig(berlin_toml("smoke"); output_path = nothing, physics = nogen))
        BCD.run_case!(plain)
        st_plain = berlin_state(plain)
        @test keys(st_plain) == keys(st_on) && all(k -> isequal(st_plain[k], st_on[k]), keys(st_on))
        d = read_dynamic_driver(drv; mesh = online.mesh)
        f = d.boundary
        # a record per parent step from t = 0; the last window may reach past the run's end
        @test length(f.time) == nc.windows + 1 && f.time[1] == 0.0 && f.time[end-1] < online.sim.t <= f.time[end]
        @test isnan(f.step[1]) && all(isfinite, f.step[2:end])
        @test Set(keys(d.fields)) == Set([:u, :v, :w])
        @test haskey(d.tables, :profile_u) && size(d.tables[:profile_u].values) == (online.mesh.Nz, length(f.time))
        @info @sprintf("online smoke with child_boundaries: %d parent windows, %d records, driver %.1f MB, %.1f s",
                       nc.windows, length(f.time), filesize(drv) / 1024^2, t_online)
        # the child alone from the file, through aeolus run and its replay
        r = roundtrip_check(ID_SMOKE; nsteps = 12, nsteps_overrides = n -> ["clock.nsteps=$n"],
                            overrides = ["initial.dynamic=\"$drv\"", "physics.options.nesting.parent_driver=\"offline\""])
        @test r.fields_equal && r.diagnostics_equal && r.config_equal
        rec = read_run_toml(r.runs[1].folder.path)
        @test rec["inputs"]["initial.dynamic"]["fnv1a64"] == content_hash(drv)
        off = r.runs[1].obj.case
        oc = BCD.find_coupling(off, :nesting)
        @test BCD.offline(oc) && oc.parent === nothing && oc.windows == nc.windows
        st_off = berlin_state(off)
        @test !haskey(st_off, :parent_θ)
        ks = [k for k in keys(st_off)]
        @test Set(ks) == Set(k for k in keys(st_on) if !startswith(String(k), "parent_"))
        bad = [k for k in ks if !isequal(st_off[k], st_on[k])]
        isempty(bad) || @info "offline child differs from the online child" bad
        @test isempty(bad)
        @test off.sim.t == online.sim.t && off.clock.n == online.clock.n
        @info @sprintf("offline smoke child: %d steps, bitwise the online one-way child in %d fields; %d records read",
                       off.clock.n, length(ks), oc.feed.reads)
    finally
        Aeolus.Operators.set_fftw_policy!(; planner = p0.planner, threads = p0.threads)
    end
end

@testset "InputDrivers dynamic: the open case from the sounding driver's steady planes (smoke, 12 steps)" begin
    dir = mktempdir()
    out = joinpath(dir, "smoke_open.nc")
    ov = ["initial.sounding=\"$ID_CSV\"", "physics.nesting=false", "boundaries.lateral=\"open\""]
    write_sounding_driver(load_run_config(ID_SMOKE; overrides = ov), out)
    m = RegularGrid(; size = (16, 16, 16), extent = (250.0, 250.0, 100.0), halo = (3, 3, 3))
    d = read_dynamic_driver(out; mesh = m)
    @test d.boundary !== nothing && length(d.boundary.time) == 1 && haskey(d.tables, :profile_theta)
    @test d.attributes["dynamic_source"] == "sounding"
    run(dyn) = (c = BCD.build_case(BCD.CaseConfig(berlin_toml("smoke"); output_path = nothing, physics = Dict{Symbol,Any}(:nesting => false),
                                                  lateral = :open, sounding = ID_CSV, dynamic = dyn));
                BCD.run_case!(c); c)
    a, b = run(nothing), run(out)
    @test all(st -> st.data isa ProfileBoundaryData, open_boundary_states(a.sim))
    @test all(st -> st.data isa FileBoundaryData, open_boundary_states(b.sim))
    sa, sb = berlin_state(a), berlin_state(b)
    bad = [k for k in keys(sa) if !isequal(sa[k], sb[k])]
    isempty(bad) || @info "open case from the driver's planes differs from the profile data" bad
    @test isempty(bad) && b.clock.n == a.clock.n
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
    for (la, lo) in ((52.6, 13.4), (51.3, 15.9), (52.21, 14.12))
        x, y = latlon_to_meters(la, lo, lat0, lon0)
        la2, lo2 = meters_to_latlon(x, y, lat0, lon0)
        @test abs(la2 - la) < 1e-12 && abs(lo2 - lo) < 1e-12
    end
    # 1 km north is 1000/R radians of latitude
    @test meters_to_latlon(0.0, 1000.0, lat0, lon0)[1] - lat0 ≈ rad2deg(1000.0 / 6_371_000.0) rtol = 1e-9
end

@testset "ERA5 driver: the open simulation box (16³, 20 steps, round trip)" begin
    @test filesize(E5_DRIVER) <= 2 * 1024^2                 # committed: 2 MB or smaller
    @test isempty(check_driver(E5_DRIVER, DYNAMIC_SCHEMA))
    d = read_dynamic_driver(E5_DRIVER; planes = false)
    @test d.attributes["dynamic_source"] == "era5" && d.attributes["origin_lat"] == 52.2085
    @test occursin("era5_lindenberg_20190629_sfc.nc", d.attributes["era5_files"])
    @test d.init_time == "2019-06-29T00:00:00Z" && size(d.tables[:profile_theta].values) == (16, 13)
    # the config names the committed driver
    @test load_run_config(E5_CFG).domain.dynamic_driver == E5_DRIVER          # the domain's (ONE-b)
    # the round trip, the driver hashed under [inputs]
    r = roundtrip_check(E5_CFG; nsteps = 20)
    @test r.fields_equal && r.diagnostics_equal && r.config_equal
    rec = read_run_toml(r.runs[1].folder.path)
    @test rec["inputs"]["domain.main.dynamic_driver"]["fnv1a64"] == content_hash(E5_DRIVER)
    run1 = r.runs[1].obj
    sim1 = Aeolus.SimulationDriver.simulation(run1)
    feed1 = only(run1.model.domains).drivers.boundary          # the domain's dynamic-driver feed (ONE-a)
    @test feed1 isa FileBoundaryFeed && sim1.t == 40.0 && feed1.data.t0 == 0.0
    # the initial state is the ERA5 profile at the domain centre
    b0 = Config.build(Aeolus.SimulationDriver.SimulationDriverType(), load_run_config(E5_CFG))
    θ0 = Aeolus.Fields.interior_view(Aeolus.SimulationDriver.simulation(b0).θ)
    @test θ0[3, 5, :] == d.profiles.theta && θ0[1, 1, :] == d.profiles.theta
    # after 20 steps the open faces hold the ERA5 planes blended at t = 40 s
    st = only(open_boundary_states(sim1.bcs))
    pu = only(p for p in st.planes if p.name === :θ && p.side === :west)
    f = read_dynamic_driver(E5_DRIVER; mesh = sim1.mesh).boundary
    A = snapshot_values(f.data.a, :θ); B = snapshot_values(f.data.b, :θ)
    τ = 40.0 / 3600.0
    want = [(1 - τ) * A[i, j, k] + τ * B[i, j, k] for i in pu.irange, j in pu.jrange, k in pu.krange]
    @test maximum(abs.(pu.data .- want)) < 1e-12
    @info @sprintf("open ERA5 box over Lindenberg: 20 steps to t = %.0f s, bitwise in the round trip; θ at the west face %.2f-%.2f K, u in the box %.2f-%.2f m/s",
                   sim1.t, extrema(pu.data)..., extrema(Aeolus.Fields.interior_view(sim1.u))...)
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
