# The SGS diffusion of the urban tracers as one cache-blocked pass (#559):
# `Operators.blocked_eddy_diffusion!` on the blocked driver of #533 / #554,
# chosen by `sgs_diffuse!` of `Urban.TracerSGSDiffusion` unless
# loop_fusion = "never" (the separate loops, the reference).
#
#   * the pass against the separate loops: the tracer, the per-level sums
#     of |q| and of the tendency, the water ledger (`water_split_levels`)
#     and the conservation residual, on the Boussinesq and the anelastic
#     core (the constraint weights), with the solid mask, κh = κv and
#     κh ≠ κv, periodic and mirror (open-case) halos, at 1 to 5 slabs in
#     forward and reverse order;
#   * conservation: the step keeps Σ q (Σ W q) of the fluid cells to
#     round-off and leaves solid cells untouched;
#   * the setting, the report, :always, no allocation, and the slabs
#     running concurrently in a -t 4 subprocess.

using Test
using Aeolus
using Aeolus.Backends: CPUBackend, with_backend
include(joinpath(@__DIR__, "support", "tracer_diffusion_cases.jl"))

const _TD_DT = 0.8

@testset "tracer diffusion blocked pass: agrees with the separate loops (#559)" begin
    worst = (q = 0.0, Sq = 0.0, St = 0.0, W = 0.0)
    nrun = 0
    for core in (:boussinesq, :anelastic), aniso in (false, true),
        bcs in (FieldBoundaryConditions(:periodic, :periodic, :noflux), FieldBoundaryConditions(:noflux))
        g = RegularGrid(; size = (14, 11, 10), extent = (140.0, 110.0, 100.0), halo = (3, 3, 3))
        case, d, q, κv = tdc_case(g; core = core, aniso = aniso, bcs = bcs)
        q0 = copy(q.data)
        Aeolus.Simulations.set_loop_fusion!(case.sim, :never)
        ref = tdc_run!(case, d, q, q0, _TD_DT; κv = κv)
        @test maximum(abs.(ref.q .- q0)) > 1e-4 * maximum(abs, q0)       # the step does something
        Aeolus.Simulations.set_loop_fusion!(case.sim, :if_possible)
        for ns in 1:5, rev in (false, true)
            dn = tdc_diffusion(g, d.mask, d.weights, bcs; ns = ns)
            b = with_backend(CPUBackend(threshold = 0)) do
                if rev        # the driver's hook: the slabs one after another, last first
                    copyto!(q.data, q0)
                    fill_halos!(q, bcs)
                    Aeolus.Operators.blocked_eddy_diffusion!(dn.blocked, dn.Sq, dn.St, q, case.sim.κt,
                                                             κv === nothing ? case.sim.κt : κv, _TD_DT;
                                                             mask = dn.mask, weights = dn.weights, ns = ns,
                                                             reverse_slabs = true)
                    (q = copy(q.data), Sq = copy(dn.Sq), St = copy(dn.St))
                else
                    tdc_run!(case, dn, q, q0, _TD_DT; κv = κv)
                end
            end
            if rev
                @test maximum(abs.(b.q .- ref.q)) <= 4eps() * maximum(abs, ref.q)
                @test maximum(abs.(b.St .- ref.St)) <= 1e-13 * maximum(abs, ref.St)
            else
                ok, e = tdc_agree(b, ref, case, _TD_DT)
                @test ok
                worst = map(max, worst, e)
            end
            nrun += 1
        end
    end
    @test nrun == 80
    @info "tracer diffusion blocked pass against the separate loops: worst relative errors" worst
end

@testset "tracer diffusion blocked pass: conservation, solid cells, the ledger (#559)" begin
    # the diffusion is a flux form with no flux through solid faces and
    # mirror / periodic halos: Σ q (Boussinesq) or Σ W q (anelastic) of the
    # fluid cells is kept to round-off, the solid cells keep their value,
    # and the booked conserved part `dW` is ~0 while the residual is < 1e-12
    for core in (:boussinesq, :anelastic)
        g = RegularGrid(; size = (12, 10, 9), extent = (120.0, 100.0, 90.0), halo = (3, 3, 3))
        case, d, q, κv = tdc_case(g; core = core)
        q0 = copy(q.data)
        r = tdc_run!(case, d, q, q0, _TD_DT)
        H = (g.Hx, g.Hy, g.Hz)
        I = (H[1]+1:H[1]+g.Nx, H[2]+1:H[2]+g.Ny, H[3]+1:H[3]+g.Nz)
        W = d.weights === nothing ? ones(g.Nz) : d.weights[1][I[3]]
        tot(a) = sum(W[k] * a[i, j, I[3][k]] for i in I[1], j in I[2], k in 1:g.Nz)
        @test abs(tot(r.q) - tot(q0)) <= 1e-13 * tot(abs.(q0))
        solid = d.mask[I...]
        @test any(solid) && r.q[I...][solid] == q0[I...][solid]           # untouched: copied, not computed
        @test r.resid < 1e-12
        wsc = _TD_DT * sum(case.budgets.ρ .* abs.(r.St)) * g.dz / (g.Nx * g.Ny)
        @test abs(r.dW) <= 1e-12 * max(wsc, 1e-300) || abs(r.dW) < 1e-18
        @test Aeolus.Simulations.sweep_report(case.sim)["tracer_diffusion_blocked"] == "all"
    end
end

@testset "tracer diffusion blocked pass: setting, report, allocation (#559)" begin
    g = RegularGrid(; size = (10, 8, 8), extent = (100.0, 80.0, 80.0), halo = (3, 3, 3))
    case, d, q, _ = tdc_case(g)
    q0 = copy(q.data)
    tdc_run!(case, d, q, q0, _TD_DT)
    Aeolus.Simulations.set_loop_fusion!(case.sim, :never)
    tdc_run!(case, d, q, q0, _TD_DT)
    r = Aeolus.Simulations.sweep_report(case.sim)
    @test r["tracer_diffusion_blocked"] == "some" && !haskey(r, "tracer_diffusion_not_blocked")
    # without its scratch the pass cannot run: :if_possible falls back and
    # says why, :always refuses
    dn = TDU.TracerSGSDiffusion(d.tend, d.mask, d.weights, d.bcs, Ref(0.0), Ref(0.0), Ref(0.0),
                                zeros(g.Nz), zeros(g.Nz), nothing)
    Aeolus.Simulations.set_loop_fusion!(case.sim, :if_possible)
    tdc_run!(case, dn, q, q0, _TD_DT)
    @test occursin("scratch", Aeolus.Simulations.sweep_report(case.sim)["tracer_diffusion_not_blocked"])
    Aeolus.Simulations.set_loop_fusion!(case.sim, :always)
    @test_throws ArgumentError tdc_run!(case, dn, q, q0, _TD_DT)
    # a stretched grid has no masked diffusion: the reason names it
    gs = StretchedGrid(; size = (6, 6, 6), extent = (60.0, 60.0, 60.0))
    @test occursin("RegularGrid", something(TDU._tracer_blocked_blocker(Field(CellCenter(), gs)), ""))
    # the pass reuses its scratch, at the scratch's slab count, and
    # allocates nothing on a serial backend (review of #559: a scratch made
    # for 3 slabs where the backend's default is 1 was replaced by a new
    # one, 31 kB, on every call, and the agreement test above ran 1 slab
    # whatever `ns` it asked for)
    Aeolus.Simulations.set_loop_fusion!(case.sim, :if_possible)
    with_backend(CPUBackend(serial = true)) do
        for ns in (1, 3)
            ds = tdc_diffusion(g, d.mask, d.weights, d.bcs; ns = ns)
            κ = case.sim.κt
            TDU.sgs_diffuse!(ds, case, q, _TD_DT; water = true, κ = κ, κv = κ)
            a = @allocated TDU.sgs_diffuse!(ds, case, q, _TD_DT; water = true, κ = κ, κv = κ)
            @test a == 0
            @test Aeolus.Operators.nslabs(ds.blocked) == ns
        end
    end
end

# The test gate runs each file at one thread, where the slabs above run
# one after another. The concurrent case runs in a -t 4 subprocess (skipped
# when this process already has 4 threads): both cores, κh ≠ κv, both halo
# kinds, at 2 to 5 slabs, against the separate loops (`tdc_agree`).
const _TD_THREADED = """
using Aeolus
using Aeolus.Backends: CPUBackend, with_backend
include($(repr(joinpath(@__DIR__, "support", "tracer_diffusion_cases.jl"))))
runs = 0; ok = true
for core in (:boussinesq, :anelastic), bcs in (FieldBoundaryConditions(:periodic, :periodic, :noflux),
                                               FieldBoundaryConditions(:noflux))
    g = RegularGrid(; size = (16, 12, 20), extent = (160.0, 120.0, 200.0), halo = (3, 3, 3))
    case, d, q, κv = tdc_case(g; core = core, aniso = true, bcs = bcs)
    q0 = copy(q.data)
    Aeolus.Simulations.set_loop_fusion!(case.sim, :never)
    ref = tdc_run!(case, d, q, q0, 0.8; κv = κv)
    Aeolus.Simulations.set_loop_fusion!(case.sim, :if_possible)
    for ns in 2:5
        dn = tdc_diffusion(g, d.mask, d.weights, bcs; ns = ns)
        b = with_backend(CPUBackend(threshold = 0)) do
            tdc_run!(case, dn, q, q0, 0.8; κv = κv)
        end
        global ok &= first(tdc_agree(b, ref, case, 0.8))
        global runs += 1
    end
end
print("threads ", Threads.nthreads(), " runs ", runs, " ok ", ok)
"""

@testset "tracer diffusion blocked pass: slabs run concurrently, -t 4 (#559)" begin
    if Threads.nthreads() >= 4
        @test true
    else
        project = dirname(Base.active_project())
        cmd = `$(Base.julia_cmd()) --startup-file=no -t 4 --project=$project -e $_TD_THREADED`
        out = IOBuffer(); err = IOBuffer()
        p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
        msg = String(take!(out))
        success(p) || @info "tracer_diffusion_blocked.jl: the -t 4 subprocess failed" String(take!(err))
        @test success(p)
        m = match(r"threads (\d+) runs (\d+) ok (true|false)", msg)
        @test m !== nothing
        m === nothing || @test parse(Int, m[1]) == 4 && parse(Int, m[2]) == 16 && m[3] == "true"
    end
end
