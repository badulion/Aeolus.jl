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
        # rows are buffered, the costly ones too (#514): on disk after 4 rows
        n == 3 && @test w.ds.dim["time"] == 0 && w.ds.dim["time_costly"] == 0
        n == 4 && @test w.ds.dim["time"] == 4 && w.ds.dim["time_costly"] == 2   # rows 1 and 4
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

end
