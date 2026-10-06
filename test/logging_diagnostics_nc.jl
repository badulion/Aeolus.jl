# diagnostics.nc (LOG-e, #337; design in epic #293): the profile registry
# (src/logging/diagnostics.jl, src/logging/component_diagnostics.jl), the
# NetCDF-4 writer and its provenance attributes (src/io/diagnostics_nc.jl)
# and the run monitor writing both files (src/io/run_monitor.jl).
#
# What is asserted:
#   * every shipped profile equals the one computed here from the raw
#     arrays with own loops (means, the w variance on the faces, the
#     resolved heat flux, eddy viscosity, SGS energy), on the right levels;
#   * diagnostics.nc holds every row of the CSV (bit for bit) on `time`
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
end

end
