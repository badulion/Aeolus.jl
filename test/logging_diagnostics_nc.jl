# diagnostics.nc (LOG-e, #337; design in epic #293): the profile registry
# (src/logging/diagnostics.jl, src/logging/component_diagnostics.jl), the
# NetCDF-4 writer and its provenance attributes (src/io/diagnostics_nc.jl)
# and the run monitor writing both files (src/io/run_monitor.jl).
#
# What is asserted:
#   * every shipped profile equals the one computed here from the raw
#     arrays with own loops (means, the w variance on the faces, the
#     resolved heat flux, eddy viscosity, SGS energy), on the right levels;
#   * diagnostics.nc holds every row of the CSV on `time`
#     and the costly scalars and profiles on `time_costly`, at the cadence;
#   * each scalar is evaluated once per row for both files;
#   * the global attributes are the run folder's provenance; a restart
#     appends to the same file with the same run ID, another layout goes
#     to diagnostics-2.nc;
#   * a blowup leaves every row before it in the file.

using Test
using Logging
using Random
using Dates: DateTime
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Grids: znodes, CellCenter, FaceZ
using Aeolus.Nesting
using Aeolus.TimeIntegration: step_adaptive!, CFLConstraints
import NCDatasets

function dn_sim(; N = (10, 8, 6), seed = 7, moisture = false)
    m = RegularGrid(; size = N, extent = (500.0, 400.0, 300.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; θ₀ = 300.0, C_s = 0.17, moisture = moisture,
                               bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(seed)
    set!(sim.u, (x, y, z) -> 2.0 + 0.004 * z + 0.4 * randn(rng))
    set!(sim.v, (x, y, z) -> -0.5 + 0.4 * randn(rng))
    set!(sim.θ, (x, y, z) -> 300.0 + 0.003 * z + 0.05 * randn(rng))
    moisture && set!(sim.qv, (x, y, z) -> 0.008 - 1e-5 * z + 1e-4 * randn(rng))
    Aeolus.TimeIntegration._project!(sim, 1.0)
    return sim
end

dn_raw(f) = (m = f.mesh; f.data[m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz])
dn_mean(A) = [sum(A[:, :, k]) / (size(A, 1) * size(A, 2)) for k in 1:size(A, 3)]
dn_var(A) = (μ = dn_mean(A); [sum(abs2, A[:, :, k] .- μ[k]) / (size(A, 1) * size(A, 2))
                              for k in 1:size(A, 3)])

const DN_GIT = `git -c user.name=aeolus-test -c user.email=test@example.org -c commit.gpgsign=false`
function dn_repo()
    d = mktempdir()
    run(`git -C $d init -q`)
    write(joinpath(d, "model.jl"), "x = 1\n")
    run(`$DN_GIT -C $d add model.jl`)
    run(`$DN_GIT -C $d commit -q -m initial`)
    return d
end

# a component whose diagnostics count their evaluations
mutable struct DNCounter
    n_step::Int
    n_costly::Int
end
Aeolus.RunLog.diagnostics(c::DNCounter, sim) = Diagnostic[
    Diagnostic("calls", "1", () -> (c.n_step += 1; Float64(c.n_step)); cadence = :step),
    Diagnostic("costly_calls", "1", () -> (c.n_costly += 1; Float64(c.n_costly)))]

dn_kwargs(r) = Dict{Symbol,Any}(r.kwargs)

@testset "diagnostics.nc (LOG-e, #337)" begin

@testset "ProfileDiagnostic: declaration, evaluation, failure" begin
    p = ProfileDiagnostic("theta", "K", "z", [5.0, 15.0], () -> [300.0, 301.0])
    @test p.levels == [5.0, 15.0] && evaluate_profile(p) == [300.0, 301.0]
    @test_throws ArgumentError ProfileDiagnostic("bad name", "K", "z", [1.0], () -> [0.0])
    @test_throws ArgumentError ProfileDiagnostic("x", "K", "z/f", [1.0], () -> [0.0])
    @test_throws ArgumentError ProfileDiagnostic("x", "K", "z", Float64[], () -> Float64[])
    @test profile_diagnostics(:anything, nothing) == ProfileDiagnostic[]
    # a source that throws or has the wrong length: NaN and one warning, no stop
    bad = ProfileDiagnostic("bad", "1", "z", [1.0, 2.0], () -> error("boom"))
    short = ProfileDiagnostic("short", "1", "z", [1.0, 2.0], () -> [1.0])
    lg = Test.TestLogger(; min_level = Logging.Warn)
    r1, r2 = with_logger(() -> (evaluate_profile(bad), evaluate_profile(short)), lg)
    @test all(isnan, r1) && length(r1) == 2 && all(isnan, r2) && length(r2) == 2
    @test length(lg.logs) == 2 && all(r -> dn_kwargs(r)[:subsystem] === :diagnostics, lg.logs)
    # prefixed renames the axis too
    q = only(prefixed("child", [p]))
    @test q.name == "child.theta" && q.axis == "child.z" && q.levels == p.levels
end

@testset "the shipped profiles equal independently computed ones" begin
    sim = dn_sim(; moisture = true)
    closure = StaticSmagorinsky(0.17)
    cfl = CFLConstraints(dt_max = 1.0)
    for _ in 1:3
        step_adaptive!(sim, cfl; dt = 1.0, sgs = closure)
    end
    m = sim.mesh
    ps = collect_profiles(sim, (; sgs = closure))
    @test [p.name for p in ps] == ["u", "v", "theta", "w_var", "qv", "w_theta", "sgs.nu_v"]
    @test [p.axis for p in ps] == ["z", "z", "z", "zf", "z", "z", "sgs.z"]
    zc = [(k - 0.5) * m.dz for k in 1:m.Nz]
    zf = [(k - 1) * m.dz for k in 1:m.Nz]
    @test ps[1].levels ≈ zc && ps[4].levels ≈ zf && ps[7].levels == ps[1].levels
    v = Dict(p.name => evaluate_profile(p) for p in ps)
    u, vv, w, θ, qv = dn_raw(sim.u), dn_raw(sim.v), dn_raw(sim.w), dn_raw(sim.θ), dn_raw(sim.qv)
    @test v["u"] ≈ dn_mean(u) rtol = 1e-14
    @test v["v"] ≈ dn_mean(vv) rtol = 1e-14
    @test v["theta"] ≈ dn_mean(θ) rtol = 1e-14
    @test v["qv"] ≈ dn_mean(qv) rtol = 1e-14
    @test v["w_var"] ≈ dn_var(w) rtol = 1e-12
    @test v["w_var"][1] == 0.0                     # the impermeable bottom face
    @test all(>(0), v["w_var"][2:end])              # the flow is three-dimensional
    # ⟨w'θ'⟩ at the centres, with w averaged from the faces above and below
    W = sim.w.data
    wc = [(W[m.Hx+i, m.Hy+j, m.Hz+k] + W[m.Hx+i, m.Hy+j, m.Hz+k+1]) / 2
          for i in 1:m.Nx, j in 1:m.Ny, k in 1:m.Nz]
    μθ = dn_mean(θ)
    flux = [sum(wc[:, :, k] .* (θ[:, :, k] .- μθ[k])) / (m.Nx * m.Ny) for k in 1:m.Nz]
    @test v["w_theta"] ≈ flux rtol = 1e-10 atol = 1e-16
    # Smagorinsky: νv is the eddy viscosity of the last evaluation
    @test v["sgs.nu_v"] ≈ dn_mean(dn_raw(sim.νt)) rtol = 1e-14
    @test all(>(0), v["sgs.nu_v"])

    # TKE closure: the eddy viscosity plus the SGS energy
    tke = TKEClosure(sim.mesh)
    interior_view(tke.e) .= reshape(collect(1.0:length(interior_view(tke.e))),
                                    size(interior_view(tke.e))) ./ 100
    tp = profile_diagnostics(tke, sim)
    @test [p.name for p in tp] == ["nu_v", "e"]
    @test evaluate_profile(tp[2]) ≈ dn_mean(dn_raw(tke.e)) rtol = 1e-14
    # stretched grid: the levels are the grid's own heights
    sg = StretchedGrid(; size = (6, 6, 8), extent = (300.0, 300.0, 400.0), halo = (3, 3, 3),
                       stretching = (:custom, cumsum([0.0; fill(20.0, 4); fill(80.0, 4)])))
    ssim = BoussinesqSimulation(sg; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    sp = profile_diagnostics(ssim)
    @test sp[3].levels ≈ [10, 30, 50, 70, 120, 200, 280, 360]
    @test sp[4].levels ≈ [0, 20, 40, 60, 80, 160, 240, 320]
end

@testset "nested run: both domains, prefixed axes" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0),
                     halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    pA = AnelasticSimulation(pm; ref = isothermal_reference(pm; T_surf = 300.0),
                             bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    cA = AnelasticSimulation(cm; ref = isothermal_reference(cm; T_surf = 300.0),
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    fill!(pA.u.data, 4.0)
    run = NestedRun(pA, cA, nest; feedback = :one_way, cfl_parent = nothing, cfl_child = nothing)
    step_nested!(run; dt = 5.0)
    ps = collect_profiles(run)
    @test [p.name for p in ps] == ["parent.u", "parent.v", "parent.theta", "parent.w_var",
                                   "child.u", "child.v", "child.theta", "child.w_var"]
    @test ps[1].axis == "parent.z" && ps[5].axis == "child.z"
    @test length(ps[1].levels) == 12 && length(ps[5].levels) == cm.Nz
    @test ps[5].levels ≈ [(k - 0.5) * cm.dz for k in 1:cm.Nz]
    @test evaluate_profile(ps[1]) ≈ dn_mean(dn_raw(pA.u)) rtol = 1e-14
    @test evaluate_profile(ps[7]) ≈ dn_mean(dn_raw(cA.θ)) rtol = 1e-14
    # in diagnostics.nc the two domains keep their own axes
    path = joinpath(mktempdir(), "nest.nc")
    w = DiagnosticsNCWriter(path, Diagnostic[], ps)
    record_diagnostics!(w, run.parent.t; step = 1)
    close(w)
    NCDataset(path) do ds
        @test ds.dim["parent.z"] == 12 && ds.dim["child.z"] == cm.Nz
        @test NCDatasets.dimnames(ds["child.theta"]) == ("child.z", "time_costly")
        @test ds["child.z"][:] == ps[5].levels
    end
end

@testset "DiagnosticsNCWriter: two record dimensions, cadence, buffering, layout" begin
    dir = mktempdir()
    k = Ref(0)
    diags = [Diagnostic("count", "1", () -> Float64(k[]); cadence = :step),
             Diagnostic("sq", "m2", () -> Float64(k[])^2),
             Diagnostic("peak", "m s-1", () -> [1.0, -k[]]; reduction = :maxabs)]
    lev = [10.0, 30.0, 50.0]
    profs = [ProfileDiagnostic("prof", "K", "z", lev, () -> lev .* k[]),
             ProfileDiagnostic("other", "1", "zz", lev, () -> fill(-1.0 * k[], 3)),
             ProfileDiagnostic("faces", "1", "z", [0.0, 20.0, 40.0], () -> [1.0, 2.0, 3.0])]
    path = joinpath(dir, "d.nc")
    w = DiagnosticsNCWriter(path, diags, profs; cadence = 2.5, flush_every = 4, flush_interval = 1e9,
                            attributes = Dict("run_id" => "abc", "session" => 1),
                            time_origin = DateTime(2019, 6, 29, 4))
    lims = ["advective", "", "diffusive"]
    costly = Bool[]
    for n in 1:10
        k[] = n
        push!(costly, record_diagnostics!(w, Float64(n); step = n, dt = 1.0, limiter = lims[mod1(n, 3)],
                                          retries = n == 4 ? 2 : 0))
        # rows are buffered: on disk after 4 rows, or with a costly row
        n == 2 && @test w.ds.dim["time"] == 1         # row 1 was costly
        n == 3 && @test w.ds.dim["time"] == 1
    end
    @test costly == [n in (1, 4, 7, 10) for n in 1:10]
    close(w)
    @test read(path, 4) == UInt8[0x89, 'H', 'D', 'F']
    NCDataset(path) do ds
        @test ds.attrib["Conventions"] == "CF-1.8" && ds.attrib["run_id"] == "abc"
        @test ds.attrib["title"] == "Aeolus run diagnostics"
        @test occursin("session 1: created", ds.attrib["history"])
        @test ds.dim["time"] == 10 && ds.dim["time_costly"] == 4
        # time: every row; CF time from the origin
        @test ds["time"].var[:] == 1.0:10.0
        @test ds["time"][1] == DateTime(2019, 6, 29, 4, 0, 1)
        @test ds["step"][:] == 1:10 && eltype(ds["step"]) == Int64
        @test ds["dt"][:] == fill(1.0, 10)
        @test ds["dt_limiter"][:] == [lims[mod1(n, 3)] for n in 1:10]
        @test ds["retries"][:] == [n == 4 ? 2 : 0 for n in 1:10]
        @test ds["count"][:] == 1.0:10.0
        @test issorted(ds["wall"][:])
        # time_costly: the costly scalars and profiles at 1, 4, 7, 10
        @test ds["time_costly"].var[:] == [1.0, 4.0, 7.0, 10.0]
        @test ds["step_costly"][:] == [1, 4, 7, 10]
        @test ds["sq"][:] == [1.0, 16.0, 49.0, 100.0]
        @test ds["peak"][:] == [1.0, 4.0, 7.0, 10.0]
        @test ds["peak"].attrib["cell_methods"] == "area: maximum_absolute_value"
        @test ds["peak"].attrib["units"] == "m s-1"
        @test ds["prof"][:, :] == lev * [1.0 4.0 7.0 10.0]
        # equal levels share an axis whatever its name; other levels get a suffix
        @test NCDatasets.dimnames(ds["other"]) == ("z", "time_costly")
        @test NCDatasets.dimnames(ds["faces"]) == ("z_2", "time_costly")
        @test ds["z"][:] == lev && ds["z_2"][:] == [0.0, 20.0, 40.0]
        @test ds["z"].attrib["axis"] == "Z" && ds["z"].attrib["units"] == "m"
        @test NCDatasets.deflate(ds["prof"])[1]                # compressed
    end
    # flush_interval = 0: every row is on disk at once
    w0 = DiagnosticsNCWriter(joinpath(dir, "f0.nc"), diags; flush_interval = 0.0)
    for n in 1:3
        record_diagnostics!(w0, Float64(n); step = n)
        @test w0.ds.dim["time"] == n
    end
    close(w0)
    # names are one namespace: scalars, profiles, axes, fixed variables
    @test_throws ArgumentError DiagnosticsNCWriter(joinpath(dir, "x.nc"),
        [Diagnostic("theta", "K", () -> 0.0)], [ProfileDiagnostic("theta", "K", "z", lev, () -> lev)])
    @test_throws ArgumentError DiagnosticsNCWriter(joinpath(dir, "y.nc"),
        [Diagnostic("time_costly", "s", () -> 0.0)])
    @test_throws ArgumentError DiagnosticsNCWriter(joinpath(dir, "z.nc"),
        [Diagnostic("z", "m", () -> 0.0)], profs)

    # append: same layout continues the file; another layout goes to -2
    w2 = DiagnosticsNCWriter(path, diags, profs; cadence = 2.5,
                             attributes = Dict("run_id" => "abc", "session" => 2))
    k[] = 11
    record_diagnostics!(w2, 11.0; step = 11)
    close(w2)
    lg = Test.TestLogger(; min_level = Logging.Warn)
    w3 = with_logger(lg) do
        DiagnosticsNCWriter(path, diags[1:2], profs; attributes = Dict("session" => 3))
    end
    @test w3.path == joinpath(dir, "d-2.nc")
    @test occursin("another layout", only(lg.logs).message)
    record_diagnostics!(w3, 12.0; step = 12)
    close(w3)
    NCDataset(path) do ds
        @test ds["step"][:] == 1:11 && ds["step_costly"][:] == [1, 4, 7, 10, 11]
        @test ds["prof"][:, end] == lev * 11
        @test ds.attrib["session"] == 2
        @test occursin("session 2: appended", ds.attrib["history"])
    end
    NCDataset(joinpath(dir, "d-2.nc")) do ds
        @test ds["step"][:] == [12] && !haskey(ds, "peak")
    end
end

@testset "run monitor: CSV and NetCDF agree, each scalar evaluated once per row" begin
    sim = dn_sim()
    closure = StaticSmagorinsky(0.17)
    cfl = CFLConstraints(dt_max = 1.0)
    dir = mktempdir()
    ctr = DNCounter(0, 0)
    prof_at = Dict{Int,Vector{Float64}}()
    # the driver reports (attach = false): one row per monitor! call
    with_monitor(sim; csv = joinpath(dir, "d.csv"), nc = joinpath(dir, "d.nc"),
                 components = (; sgs = closure, c = ctr), cadence = 3.0, attach = false) do mon
        for n in 1:12
            step_adaptive!(sim, cfl; dt = 1.0, sgs = closure)
            monitor!(mon, sim; step = n, dt = 1.0, limiter = :advective)
            prof_at[n] = dn_mean(dn_raw(sim.θ))
        end
    end
    # one evaluation per row (12) and per costly row (t = 1, 4, 7, 10), not two
    @test ctr.n_step == 12 && ctr.n_costly == 4
    lines = readlines(joinpath(dir, "d.csv"))
    header = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end]]
    col(name) = findfirst(h -> startswith(h, name * " ["), header)
    NCDataset(joinpath(dir, "d.nc")) do ds
        @test ds.dim["time"] == 12 && ds.dim["time_costly"] == 4
        @test ds["c.calls"][:] == [parse(Float64, r[col("c.calls")]) for r in rows] == 1.0:12.0
        costly_rows = [r for r in rows if r[col("ke_mean")] != ""]
        @test [parse(Int, r[1]) for r in costly_rows] == ds["step_costly"][:] == [1, 4, 7, 10]
        for name in ("ke_mean", "max_div", "theta_max", "sgs.nu_h_max", "c.costly_calls")
            @test ds[name][:] == [parse(Float64, r[col(name)]) for r in costly_rows]   #
        end
        @test ds["c.costly_calls"][:] == 1.0:4.0
        for (j, n) in enumerate((1, 4, 7, 10))
            @test ds["theta"][:, j] ≈ prof_at[n] rtol = 1e-14
        end
        @test ds["time"][:] == [parse(Float64, r[2]) for r in rows]
    end
end

@testset "run folder: provenance attributes, restart appends, blowup keeps the rows" begin
    if Sys.which("git") === nothing
        @test_skip "git is not installed"
    else
        repo = dn_repo(); root = mktempdir()
        commit = readchomp(`git -C $repo rev-parse HEAD`)
        cfl = CFLConstraints(dt_max = 1.0)
        sim = dn_sim()
        rfpath = Ref("")
        with_run("nc case"; root, repo, console = false, seeds = (; noise = 42),
                 config = Dict("mesh" => Dict("N" => [10, 8, 6]))) do rf
            rfpath[] = rf.path
            # while the session runs, session_finished is empty
            a = nc_provenance_attributes(rf)
            @test a["session_finished"] == "" && a["session_started"] == rf.record["session"][1]["started"]
            with_monitor(sim; run = rf, cadence = 2.0, attach = false) do mon
                for n in 1:5
                    step_adaptive!(sim, cfl; dt = 1.0)
                    monitor!(mon, sim; step = n, dt = 1.0)
                end
            end
        end
        nc = joinpath(rfpath[], "diagnostics.nc")
        @test isfile(nc) && isfile(joinpath(rfpath[], "diagnostics.csv"))
        rec = read_run_toml(rfpath[])
        s1 = rec["session"][1]
        NCDataset(nc) do ds
            a = ds.attrib
            @test a["run_id"] == rec["run_id"] && a["case"] == "nc case"
            @test a["session"] == 1 && a["run_folder"] == rfpath[]
            @test a["git_commit"] == commit == s1["git"]["commit"]
            @test a["git_dirty"] == 0
            @test a["julia_version"] == string(VERSION)
            @test a["manifest_fnv1a64"] == s1["manifest"]["fnv1a64"]
            @test a["hostname"] == gethostname() && a["julia_threads"] == Threads.nthreads()
            @test a["fftw_planner"] == s1["fftw"]["planner"]
            @test a["bitwise_reproducible"] == Int8(s1["fftw"]["bitwise_reproducible"])
            @test occursin("N = [10, 8, 6]", a["config"])
            @test occursin("noise = 42", a["seeds"]) && occursin("[seeds]", a["seeds"])
            @test a["created"] == rec["created"]
            @test a["session_started"] == s1["started"]
            # set when the writer closes, before the session ends (same ISO format and zone
            # as run.toml, so the strings order as the times)
            @test !isempty(a["session_finished"])
            @test s1["started"] <= a["session_finished"] <= s1["ended"]
            @test ds["step"][:] == 1:5 && ds["step_costly"][:] == [1, 3, 5]
        end
        # restart: same folder, same run ID, records appended, session 2
        with_run("nc case"; root, repo, console = false, resume = rfpath[],
                 seeds = (; noise = 7), config = Dict("mesh" => Dict("N" => [10, 8, 6]))) do rf
            with_monitor(sim; run = rf, cadence = 2.0, attach = false) do mon
                for n in 6:8
                    step_adaptive!(sim, cfl; dt = 1.0)
                    monitor!(mon, sim; step = n, dt = 1.0)
                end
            end
        end
        NCDataset(nc) do ds
            @test ds.attrib["run_id"] == rec["run_id"] && ds.attrib["session"] == 2
            @test ds["step"][:] == 1:8 && ds["time"][:] == 1.0:8.0
            @test ds["step_costly"][:] == [1, 3, 5, 6, 8]     # the cadence restarts with the session
            @test length(split(ds.attrib["history"], '\n')) == 2
            # refreshed to session 2: its start, its end, its seeds override
            s2 = read_run_toml(rfpath[])["session"][2]
            @test ds.attrib["session_started"] == s2["started"]
            @test s2["started"] <= ds.attrib["session_finished"] <= s2["ended"]
            @test occursin("noise = 7", ds.attrib["seeds"])
        end
        @test !isfile(joinpath(rfpath[], "diagnostics-2.nc"))

        # a blowup: the rows before it are in the file (buffered rows flushed)
        root2 = mktempdir()
        bsim = dn_sim()
        f2 = Ref("")
        err = try
            with_run("nc blowup"; root = root2, repo, console = false) do rf
                f2[] = rf.path
                with_monitor(bsim; run = rf, cadence = 100.0, check_every = 1) do mon
                    for n in 1:10
                        monitored_step!(mon, bsim; step = n) do
                            step_adaptive!(bsim, cfl; dt = 1.0)
                        end
                        n == 6 && (bsim.θ.data[6, 6, 6] = NaN)
                    end
                end
            end
            nothing
        catch e
            e
        end
        @test err isa BlowupError && err.step == 7
        NCDataset(joinpath(f2[], "diagnostics.nc")) do ds
            @test ds["step"][:] == 1:6
        end
    end
end

@testset "the writer never prints" begin
    sim = dn_sim()
    path = joinpath(mktempdir(), "q.nc")
    out = mktemp() do p, io
        redirect_stdout(io) do
            redirect_stderr(io) do
                with_logger(NullLogger()) do
                    w = DiagnosticsNCWriter(path, collect_diagnostics(sim), collect_profiles(sim))
                    record_diagnostics!(w, 0.0; step = 0)
                    close(w)
                end
            end
        end
        close(io); read(p, String)
    end
    @test isempty(out)
end

end
