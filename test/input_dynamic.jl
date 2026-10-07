# The dynamic driver (CFG-g, #362): Aeolus.InputDrivers — the initial
# state, the boundary planes over time (FileBoundaryData, FileBoundaryFeed),
# the large-scale and surface forcing; the sounding / case-table and the
# Aeolus-parent preprocessors.
#
# Acceptance tests of the issue covered here:
#   * the time blend is exact for data linear in t (u = 1 + 0.01 t at
#     time_bc = 0, 60, 120 s, read back at t = 17, 60, 95 s), and for data
#     quadratic in t its error is the analytic bound Δt²/8 · max|u''|;
#   * memory stays bounded: over 1000 records at most two are resident
#     (Base.summarysize of the feed);
#   * errors: a face missing for an open side, a time_bc that does not
#     cover the run (both intervals printed), a grid mismatch, a sounding
#     driver of another start;
#   * the forcing tables of GABLS3 read from its driver are the ForcingTables
#     built from gabls3_case.csv, bit for bit, and GABLS3 run 50 steps from
#     the file is bitwise the CSV path (and the pre-#359 digests);
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
#   * the committed summer sounding driver is reproducible.

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

isdefined(Main, :BerlinCase) ||
    include(joinpath(@__DIR__, "..", "examples", "urban", "BerlinCase.jl"))
const BCD = Main.BerlinCase
include(joinpath(@__DIR__, "support", "berlin_fixture.jl"))
include(joinpath(@__DIR__, "config_roundtrip_harness.jl"))
isdefined(Main, :SBLDriver) || include(joinpath(@__DIR__, "..", "examples", "sbl", "config_schema.jl"))

const ID_DATA = joinpath(@__DIR__, "data", "input_dynamic")
const ID_MASTER = TOML.parsefile(joinpath(ID_DATA, "master_digests.toml"))
const ID_SMOKE = joinpath(BCD.CONFIG_DIR, "berlin_smoke.toml")
const ID_CSV = normpath(joinpath(BCD.DATA_DIR, "lindenberg_10393_20190629.csv"))

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
    st = joinpath(BCD.CONFIG_DIR, "input", "berlin_smoke_static.nc")
    @test occursin("aeolus_driver = \"static\", expected \"dynamic\"", errmsg(() -> read_dynamic_driver(st)))
    # a sounding driver of another start
    msg = errmsg(() -> BCD.case_sounding(BCD.CaseConfig(:summer; start_h = 8.0)))
    @test occursin("starts at init_time 2019-06-29T04:00:00Z", msg) && occursin("2019-06-29T06:00:00Z", msg)
    # the parent-made driver offered as the initial state of a case without the offline nest
    msg = errmsg(() -> BCD.case_sounding(BCD.CaseConfig(:summer; dynamic = p)))
    @test occursin("not a sounding driver", msg)
end

@testset "InputDrivers dynamic: GABLS3 forcing from its driver" begin
    dir = mktempdir()
    est = ["fftw.planner=\"estimate\""]
    ov = ["grid.size=[16,16,48]", "grid.extent=[400.0,400.0,400.0]", "sponge.z=300.0"]
    cfg = load_run_config(joinpath(@__DIR__, "..", "examples", "sbl", "configs", "gabls3.toml"); overrides = vcat(ov, est))
    out = joinpath(dir, "gabls3_dynamic.nc")
    @test Config.preprocess_dynamic(SBLDriver(), cfg; from = :sounding, out = out) == [out]
    @test isempty(check_driver(out, DYNAMIC_SCHEMA))
    d = read_dynamic_driver(out; planes = false)
    @test sort!(collect(keys(d.tables))) == [:theta_adv, :u_adv, :u_g, :v_adv, :v_g]
    @test d.attributes["dynamic_source"] == "case_table" && d.init_time == ""
    a = sbl_forcing(cfg)                                            # the CSV path
    b = sbl_forcing(with(cfg, "case.dynamic" => out))               # the driver
    same(x::ForcingTable, y::ForcingTable) = x.z == y.z && x.t == y.t && isequal(x.values, y.values)
    @test same(a.U_g, b.U_g) && same(a.V_g, b.V_g)
    A, B = only(a.extra), only(b.extra)
    @test A isa AdvectiveTendencies && B isa AdvectiveTendencies
    @test same(A.θ, B.θ) && same(A.u, B.u) && same(A.v, B.v) && A.q === B.q === nothing
    @test a.surface.T_s.t == b.surface.T_s.t && a.surface.T_s.T_s == b.surface.T_s.T_s
    # 50 steps from the file, bitwise the CSV path and the pre-#359 builders
    ref = Dict("u" => "65331d3fb5e04caa", "v" => "e95dee427de2378d", "w" => "5e1de8f10c1e8fab",
               "p" => "19c322cceb048273", "θ" => "59734570be6010d9")
    steps = ["stepper.end_time=\"none\"", "stepper.nsteps=50"]
    dig(sim) = Dict(k => digest(Aeolus.Fields.interior_view(getproperty(sim, s)))
                    for (k, s) in (("u", :u), ("v", :v), ("w", :w), ("p", :p), ("θ", :θ)))
    runs = map((String[], ["case.dynamic=\"$out\""])) do extra
        c = load_run_config(joinpath(@__DIR__, "..", "examples", "sbl", "configs", "gabls3.toml");
                            overrides = vcat(ov, est, steps, extra))
        run_config(c; root = mktempdir(), console = false, repo = mktempdir())
    end
    @test dig(runs[1].obj.case.sim) == ref
    @test dig(runs[2].obj.case.sim) == dig(runs[1].obj.case.sim)
    @test runs[2].obj.case.nsteps == 50
    inp = read_run_toml(runs[2].folder.path)["inputs"]
    @test inp["case.dynamic"]["path"] == out && inp["case.dynamic"]["fnv1a64"] == content_hash(out)
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
    file = BCD.CaseConfig(:summer)
    @test file.dynamic == joinpath(BCD.CONFIG_DIR, "input", "berlin_20190629_dynamic.nc")
    @test BCD.case_sounding(file).inputs.attributes["dynamic_source"] == "sounding"
    for (label, cfg) in (("sounding driver", file), ("CSV in memory", BCD.CaseConfig(:summer; dynamic = nothing)))
        got = initial_digests(cfg)
        bad = sort!([k for (k, v) in got if !isequal(M[k], v)])
        isempty(bad) || @info "initial state from the $label differs from master" bad
        @test isempty(bad)
    end
    @info @sprintf("summer initial state from the sounding driver: p_surface %.6f Pa, z_i %.0f m, max sounding z_i %.1f m (bitwise master)",
                   M["summer_p_surface"], M["summer_zi"], M["summer_zi_max"])
end

@testset "InputDrivers dynamic: the committed summer sounding driver is reproducible" begin
    dyn_content(path) = NCDataset(path) do ds
        (Dict(k => Array(ds[k].var[ntuple(_ -> Colon(), ndims(ds[k]))...]) for k in keys(ds) if ndims(ds[k]) > 0),
         Dict(k => v for (k, v) in ds.attrib if !(k in VOLATILE_ATTRIBUTES)))
    end
    cfg = load_run_config(joinpath(BCD.CONFIG_DIR, "berlin_summer.toml"))
    cc = BCD.CaseConfig(cfg)
    out = joinpath(BCD.CONFIG_DIR, "input", "_repro_dynamic.nc")
    try
        @test Config.preprocess_dynamic(BCD.BerlinDriver(), cfg; from = :sounding, out = out) == [out]
        @test filesize(cc.dynamic) <= 2 * 1024^2
        @test isempty(check_driver(cc.dynamic, DYNAMIC_SCHEMA))
        vn, an = dyn_content(out); vo, ao = dyn_content(cc.dynamic)
        @test keys(vn) == keys(vo) && all(k -> isequal(vn[k], vo[k]), keys(vn))
        @test an == ao
        NCDataset(cc.dynamic) do ds
            @test ds["init_p_surface"].var[] == ID_MASTER["summer_p_surface"]
        end
    finally
        rm(out; force = true)
    end
end

@testset "InputDrivers dynamic: aeolus preprocess dynamic" begin
    dir = mktempdir()
    out = joinpath(dir, "smoke_sounding.nc")
    rc = redirect_stdout(devnull) do
        Aeolus.main(["preprocess", "dynamic", "--from-sounding", ID_SMOKE, "--set", "initial.sounding=\"$ID_CSV\"",
                     "--out", out])
    end
    @test rc == 0 && isfile(out)
    d = read_dynamic_driver(out; planes = false)
    @test d.init_time == "2019-06-29T10:00:00Z" && d.attributes["source"] == ID_CSV
    lf = joinpath(dir, "list.txt")
    open(lf, "w") do io
        redirect_stdout(() -> Aeolus.main(["preprocess"]), io)
    end
    @test occursin("dynamic (--from-sounding | --from-parent)", read(lf, String))
    rc = redirect_stderr(devnull) do
        Aeolus.main(["preprocess", "dynamic", ID_SMOKE])
    end
    @test rc == 2
    # a missing driver is a config error with the command that makes it
    msg = errmsg(() -> Config.build(BCD.BerlinDriver(),
                                    load_run_config(ID_SMOKE; overrides = ["initial.dynamic=\"$(joinpath(dir, "none.nc"))\""])))
    @test occursin("preprocess dynamic --from-sounding", msg)
end

@testset "InputDrivers dynamic: the smoke case from a sounding driver = the CSV path (12 steps, round trip)" begin
    dir = mktempdir()
    out = joinpath(dir, "smoke_sounding.nc")
    Config.preprocess_dynamic(BCD.BerlinDriver(), load_run_config(ID_SMOKE; overrides = ["initial.sounding=\"$ID_CSV\""]);
                              from = :sounding, out = out)
    r = roundtrip_check(ID_SMOKE; nsteps = 12, nsteps_overrides = n -> ["clock.nsteps=$n"],
                        overrides = ["initial.dynamic=\"$out\""])
    @test r.fields_equal && r.diagnostics_equal && r.config_equal
    rec = read_run_toml(r.runs[1].folder.path)
    @test rec["inputs"]["initial.dynamic"]["fnv1a64"] == content_hash(out)
    case = r.runs[1].obj.case
    @test BCD.case_sounding(case.cfg) !== nothing && case.cfg.sounding === nothing
    # the CSV path, run the same way in this process. The 12-step state is
    # compared here rather than against digests captured on master: those
    # depend on the process environment (the runner's single-threaded BLAS
    # and test logger give other last bits than a plain `julia` on the same
    # code, measured on the CSV path itself), the initial state above does not.
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
    # the online run, recording the child's boundaries at every parent step
    t0 = time()
    online = BCD.build_case(BCD.CaseConfig(:smoke; output_path = nothing, child_boundaries = drv))
    BCD.run_case!(online)
    t_online = time() - t0
    nc = BCD.find_coupling(online, :nesting)
    st_on = berlin_state(online)
    # recording does not change the run: the CFG-c fixture's smoke digests
    fx = TOML.parsefile(joinpath(@__DIR__, "data", "config_berlin", "fixture.toml"))["smoke_digests"]
    @test all(k -> digest(st_on[Symbol(k)]) == fx[k], collect(keys(fx)))
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
end

@testset "InputDrivers dynamic: the open case from the sounding driver's steady planes (smoke, 12 steps)" begin
    dir = mktempdir()
    out = joinpath(dir, "smoke_open.nc")
    ov = ["initial.sounding=\"$ID_CSV\"", "physics.nesting=false", "boundaries.lateral=\"open\""]
    Config.preprocess_dynamic(BCD.BerlinDriver(), load_run_config(ID_SMOKE; overrides = ov); from = :sounding, out = out)
    m = RegularGrid(; size = (16, 16, 16), extent = (250.0, 250.0, 100.0), halo = (3, 3, 3))
    d = read_dynamic_driver(out; mesh = m)
    @test d.boundary !== nothing && length(d.boundary.time) == 1 && haskey(d.tables, :profile_theta)
    @test d.attributes["dynamic_source"] == "sounding"
    run(dyn) = (c = BCD.build_case(BCD.CaseConfig(:smoke; output_path = nothing, physics = Dict{Symbol,Any}(:nesting => false),
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
