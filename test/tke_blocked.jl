# The update of the SGS TKE as one cache-blocked pass (#554, part 2):
# `advance_sgs!(::TKEClosure)` on the blocked driver of #533
# (src/prognostic_sgs/tke_blocked.jl) against the separate loops it
# replaces (`_advance_separate!`, the reference, what loop_fusion =
# "never" runs).
#
# Every value of the pass is computed by the point body of the separate
# loop, so e is expected to agree to round-off; the budget sums are the
# same row sums combined in the same order. Tolerances: e within 4 eps
# of max |e| (a different vectorisation may reorder a row's sum; measured
# 0 in every case), the budget sums within a relative 1e-12 (the same
# row sums; measured ≤ 2.3e-16), the cancelling transport sum on the
# scale of the shear production.

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Operators: advance_sgs!
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.IBM: SolidMask, mark_solid!
using Aeolus.WallModels: immersed_walls, RoughLogLaw
using Random: Xoshiro, randn

const TKB = Aeolus.PrognosticSGS

include(joinpath(@__DIR__, "support", "tke_blocked_cases.jl"))

@testset "TKE update as one cache-blocked pass (#554)" begin
    n = (14, 10, 40)        # 40 levels: 5 slabs of the WENO-Z5 pass (4 edge planes per end)
    rg = RegularGrid(; size = n, extent = (140.0, 100.0, 200.0), halo = (3, 3, 3))
    rg2 = RegularGrid(; size = n, extent = (140.0, 100.0, 200.0), halo = (2, 2, 2))   # centred
    sg = StretchedGrid(; size = n, extent = (140.0, 100.0, 400.0), halo = (2, 2, 2), stretching = (:tanh, 1.5))
    Δt = 0.4
    nlim = Ref(0)            # cases in which the limiter acted (ratio < 1 somewhere)
    for (gname, g) in (("regular WENO-Z5", rg), ("regular centred", rg2), ("stretched", sg)),
        transport in (true, false), diss in (:semi_implicit, :exact), solid in (false, true)
        solid && g isa StretchedGrid && continue      # the mask is RegularGrid-only
        sim, c = tkb_case(g; transport = transport, dissipation = diss, mask = solid,
                          walls = solid && g === rg, surface = true)
        e0 = copy(c.e.data)
        ref_e, ref_s = tkb_run!(TKB._advance_separate!, c, sim, e0, Δt)
        @test minimum(ref_e) >= 0
        transport && (nlim[] += any(<(1), interior_view(c.ws.ratio)))
        scale = maximum(abs, ref_e)
        lz = TKB.tke_blocked_lags(c)[2]
        for (bname, b) in (("serial", CPUBackend(serial = true)), ("threaded", CPUBackend(threshold = 0))),
            ns in 1:5, rev in (false, true)
            (ns > 1 && 2lz * ns > g.Nz) && continue
            (rev && ns == 1) && continue
            e, s = with_backend(b) do
                tkb_run!((c, sim, Δt) -> TKB._advance_blocked!(c, sim, Δt; ns = ns, reverse_slabs = rev),
                         c, sim, e0, Δt)
            end
            ok_e = maximum(abs.(e .- ref_e)) <= 4eps() * scale
            ok_s = tkb_stats_agree(s, ref_s)
            ok_e && ok_s || @info "TKE blocked pass disagrees" gname transport diss solid bname ns rev maximum(abs.(e .- ref_e)) s ref_s
            @test ok_e
            @test ok_s
        end
    end
    @test nlim[] > 0          # the limiter was exercised
end

@testset "TKE blocked pass: N² and D row by row, every N² variant (#556)" begin
    # Without immersed walls the pass computes N² (by `Operators._n2_body!`)
    # and the dual strain (by `_dual_row!`) row by row instead of reading
    # c.N2 / c.D (set to NaN before each run by `tkb_run!`). The four
    # variants of N² (none, dry above, moist θ_v, cloudy θ_ρ with the
    # saturated form where qc > 0), on the three grids, with and without
    # transport and the mask; e within 4 eps of max e and the budget sums
    # as above (the same bodies: expected the same bits).
    n = (14, 10, 24)
    rg = RegularGrid(; size = n, extent = (140.0, 100.0, 120.0), halo = (3, 3, 3))
    rg2 = RegularGrid(; size = n, extent = (140.0, 100.0, 120.0), halo = (2, 2, 2))
    sg = StretchedGrid(; size = n, extent = (140.0, 100.0, 240.0), halo = (2, 2, 2), stretching = (:tanh, 1.5))
    nsat = Ref(0)            # cloudy cases with saturated cells
    for (gname, g) in (("regular WENO-Z5", rg), ("regular centred", rg2), ("stretched", sg)),
        buoyancy in (:none, :moist, :cloudy), transport in (true, false), solid in (false, true)
        solid && g isa StretchedGrid && continue
        sim, c = tkb_case(g; transport = transport, mask = solid, buoyancy = buoyancy, surface = true)
        e0 = copy(c.e.data)
        ref_e, ref_s = tkb_run!(TKB._advance_separate!, c, sim, e0, 0.4)
        buoyancy === :none && @test all(iszero, interior_view(c.N2))
        buoyancy === :cloudy && (nsat[] += any(>(0), interior_view(c.cloud.qc)))
        @test maximum(abs, interior_view(c.D)) > 0
        lz = TKB.tke_blocked_lags(c)[2]
        for (bname, b) in (("serial", CPUBackend(serial = true)), ("threaded", CPUBackend(threshold = 0))),
            ns in 1:3, rev in (false, true)
            (ns > 1 && 2lz * ns > g.Nz) && continue
            (rev && ns == 1) && continue
            e, s = with_backend(b) do
                tkb_run!((c, sim, Δt) -> TKB._advance_blocked!(c, sim, Δt; ns = ns, reverse_slabs = rev),
                         c, sim, e0, 0.4)
            end
            ok_e = maximum(abs.(e .- ref_e)) <= 4eps() * maximum(abs, ref_e)
            ok_s = tkb_stats_agree(s, ref_s)
            ok_e && ok_s || @info "TKE blocked pass disagrees (#556)" gname buoyancy transport solid bname ns rev maximum(abs.(e .- ref_e)) s ref_s
            @test ok_e
            @test ok_s
        end
    end
    @test nsat[] > 0
end

@testset "TKE blocked pass: no-flux lateral faces, a short column (#554)" begin
    for (bcs, n) in ((FieldBoundaryConditions(:noflux, :noflux, :wall), (9, 7, 12)),
                     (FieldBoundaryConditions(:periodic, :periodic, :wall), (6, 1, 3)))
        g = RegularGrid(; size = n, extent = (90.0, 70.0, 60.0), halo = (3, 3, 3))
        sim, c = tkb_case(g; bcs = bcs)
        e0 = copy(c.e.data)
        ref_e, ref_s = tkb_run!(TKB._advance_separate!, c, sim, e0, 0.4)
        for ns in (1, 2)
            (ns > 1 && 8ns > g.Nz) && continue
            e, s = tkb_run!((c, sim, Δt) -> TKB._advance_blocked!(c, sim, Δt; ns = ns), c, sim, e0, 0.4)
            @test maximum(abs.(e .- ref_e)) <= 4eps() * maximum(abs, ref_e)
            @test tkb_stats_agree(s, ref_s)
        end
    end
end

@testset "TKE blocked pass: open lateral faces and top (#554)" begin
    # an open box (boundary data, a wall below): ν and the ratio have no
    # boundary data, so their open-face halos are the mirror of the first
    # interior layer, which the pass's boundary shell reproduces
    for halo in (2, 3)
        g = RegularGrid(; size = (12, 10, 16), extent = (120.0, 100.0, 160.0), halo = (halo, halo, halo))
        bcs = open_boundary_conditions(g, ProfileBoundaryData(u = 3.0, v = -1.0, w = 0.0, θ = 300.0); halo = halo)
        sim, c = tkb_case(g; bcs = bcs)
        Aeolus.BoundaryConditions.update_open_boundaries!(sim, 0.0, 0.4)
        @test TKB._tke_blocked_blocker(c, sim) === nothing
        e0 = copy(c.e.data)
        ref_e, ref_s = tkb_run!(TKB._advance_separate!, c, sim, e0, 0.4)
        for ns in (1, 2), rev in (false, true)
            (ns > 1 && 2 * TKB.tke_blocked_lags(c)[2] * ns > g.Nz) && continue
            e, s = tkb_run!((c, sim, Δt) -> TKB._advance_blocked!(c, sim, Δt; ns = ns, reverse_slabs = rev),
                            c, sim, e0, 0.4)
            @test maximum(abs.(e .- ref_e)) <= 4eps() * maximum(abs, ref_e)
            @test tkb_stats_agree(s, ref_s)
        end
    end
end

@testset "TKE blocked pass: advance_sgs!, loop_fusion, report, allocation (#554)" begin
    g = RegularGrid(; size = (16, 12, 16), extent = (160.0, 120.0, 160.0), halo = (3, 3, 3))
    sim, c = tkb_case(g)
    e0 = copy(c.e.data)
    # the default (if_possible) runs the blocked pass, "never" the separate loops
    e_b, s_b = tkb_run!(advance_sgs!, c, sim, e0, 0.4)
    Aeolus.Simulations.set_loop_fusion!(sim, :never)
    e_s, s_s = tkb_run!(advance_sgs!, c, sim, e0, 0.4)
    @test maximum(abs.(e_b .- e_s)) <= 4eps() * maximum(abs, e_s)
    @test tkb_stats_agree(s_b, s_s)
    r = Aeolus.Simulations.sweep_report(sim)
    @test r["tke_update_blocked"] == "some" && !haskey(r, "tke_not_blocked")
    # allocation: the pass reuses its scratch, and its entry is type-stable
    # (the cached scratch in the closure's `Any` slot is split to a concrete
    # type, the dissipation form and the wall production are types), so on a
    # serial backend it allocates nothing of its own: no more than the
    # separate loops (0 bytes; with walls both allocate the 368 B of the
    # wall production); with and without transport, both dissipation forms
    Aeolus.Simulations.set_loop_fusion!(sim, :if_possible)
    with_backend(CPUBackend(serial = true)) do
        for (tr, diss, solid) in ((true, :semi_implicit, false), (false, :exact, false), (true, :exact, true))
            sa, ca = tkb_case(g; transport = tr, dissipation = diss, mask = solid, walls = solid)
            advance_sgs!(ca, sa, 0.4)
            TKB._advance_separate!(ca, sa, 0.4)
            ab = @allocated advance_sgs!(ca, sa, 0.4)
            as = @allocated TKB._advance_separate!(ca, sa, 0.4)
            @test ab <= as
            solid || @test ab == 0
        end
        # every variant of N² computed inside the pass (#556: the variant is
        # chosen in `_with_n2_args`, each branch with concrete arguments),
        # on the regular and the stretched grid: 0 bytes as well
        sg = StretchedGrid(; size = (16, 12, 16), extent = (160.0, 120.0, 240.0), halo = (2, 2, 2),
                           stretching = (:tanh, 1.5))
        for gg in (g, sg), buoyancy in (:none, :moist, :cloudy), tr in (true, false)
            sa, ca = tkb_case(gg; transport = tr, buoyancy = buoyancy)
            advance_sgs!(ca, sa, 0.4)
            ab = @allocated advance_sgs!(ca, sa, 0.4)
            ab == 0 || @info "allocation of the blocked pass (#556)" typeof(gg) buoyancy tr ab
            @test ab == 0
        end
    end
    sim2, c2 = tkb_case(g)
    advance_sgs!(c2, sim2, 0.4)
    @test Aeolus.Simulations.sweep_report(sim2)["tke_update_blocked"] == "all"
end

@testset "TKE blocked pass: the step's νt and κt (sim) agree with the separate loops (#554)" begin
    # Output and diagnostics read the step's coefficients from sim.νt /
    # sim.κt (sgs_coefficients!, from the committed e), not the closure's
    # scratch ν / κt, which the blocked pass does not keep (a validation
    # read c.κt and lost most of its SGS heat flux; review of #554). Two
    # identical runs of three steps, blocked and separate: e, νt and κt
    # agree within 1e-12 of their maxima (e is the same bits per update;
    # the flow then evolves through the same arithmetic, so only round-off
    # of a different summation order could separate them).
    g = RegularGrid(; size = (12, 10, 16), extent = (120.0, 100.0, 160.0), halo = (3, 3, 3))
    runs = map((:if_possible, :never)) do setting
        sim, c = tkb_case(g; mask = true)
        Aeolus.Simulations.set_loop_fusion!(sim, setting)
        for _ in 1:3
            step!(sim, 0.05; sgs = c)
        end
        (e = copy(interior_view(c.e)), νt = copy(interior_view(sim.νt)), κt = copy(interior_view(sim.κt)),
         rep = Aeolus.Simulations.sweep_report(sim)["tke_update_blocked"])
    end
    b, s = runs
    @test b.rep == "all" && s.rep == "none"
    for f in (:e, :νt, :κt)
        x, y = getfield(b, f), getfield(s, f)
        @test maximum(y) > 0
        @test maximum(abs.(x .- y)) <= 1e-12 * maximum(abs, y)
    end
end

@testset "TKE blocked pass: loop_fusion on an anelastic core (#554)" begin
    # the setting chooses the TKE update on either core (documented in
    # set_loop_fusion! and [timestep] loop_fusion): blocked by default,
    # the separate loops under "never"; e agrees within 4 eps of max e
    g = RegularGrid(; size = (10, 8, 12), extent = (100.0, 80.0, 120.0), halo = (3, 3, 3))
    θf(z) = 300 + 3e-3 * z
    out = map((:if_possible, :never)) do setting
        sim = Aeolus.Anelastic.AnelasticSimulation(g; ref = Aeolus.Anelastic.reference_from_profile(g, θf, 101325.0),
                                                   θ_flux_bottom = 0.05)
        set!(sim.θ, (x, y, z) -> θf(z))
        rng = Xoshiro(7)
        for f in (sim.u, sim.v, sim.w)
            interior_view(f) .= randn(rng, size(interior_view(f)))
        end
        c = TKEClosure(g; transport = true)
        initialize_tke!(c, sim; e_min = 1e-3)
        Aeolus.Simulations.set_loop_fusion!(sim, setting)
        advance_sgs!(c, sim, 0.4)
        (e = copy(interior_view(c.e)), rep = Aeolus.Simulations.sweep_report(sim)["tke_update_blocked"])
    end
    @test out[1].rep == "all" && out[2].rep == "none"
    @test maximum(abs.(out[1].e .- out[2].e)) <= 4eps() * maximum(out[2].e)
end

# The slabs of the pass run concurrently only with more than one thread;
# the test gate runs each file at one thread, where the "threaded" cases
# above run the slabs one after another. So the concurrent case runs here
# in a -t 4 subprocess (skipped when this process already has 4 threads,
# where the main set ran them concurrently): WENO-Z5 and centred, with the
# solid mask, dry, moist and cloudy N² (#556), at 2 to 5 slabs, against
# the separate loops (e within 4 eps of max e, the budget sums as above).
const _TKB_THREADED = """
using Aeolus
using Aeolus.Backends: CPUBackend, with_backend
const TKB = Aeolus.PrognosticSGS
include($(repr(joinpath(@__DIR__, "support", "tke_blocked_cases.jl"))))
n = (14, 10, 40); worst = 0.0; stats_ok = true; runs = 0
for h in (2, 3), (solid, buoyancy) in ((false, :dry), (true, :dry), (false, :cloudy), (true, :moist))
    g = RegularGrid(; size = n, extent = (140.0, 100.0, 200.0), halo = (h, h, h))
    sim, c = tkb_case(g; mask = solid, surface = true, buoyancy = buoyancy)
    e0 = copy(c.e.data)
    ref_e, ref_s = tkb_run!(TKB._advance_separate!, c, sim, e0, 0.4)
    for ns in 2:5
        2 * TKB.tke_blocked_lags(c)[2] * ns > g.Nz && continue
        e, s = with_backend(CPUBackend(threshold = 0)) do
            tkb_run!((c, sim, Δt) -> TKB._advance_blocked!(c, sim, Δt; ns = ns), c, sim, e0, 0.4)
        end
        global worst = max(worst, maximum(abs.(e .- ref_e)) / maximum(abs, ref_e))
        global stats_ok &= tkb_stats_agree(s, ref_s)
        global runs += 1
    end
end
print("threads ", Threads.nthreads(), " runs ", runs, " worst ", worst, " stats ", stats_ok)
"""

@testset "TKE blocked pass: slabs run concurrently, -t 4 (#554)" begin
    if Threads.nthreads() >= 4
        @test true            # the main set above ran the slabs concurrently
    else
        project = dirname(Base.active_project())
        cmd = `$(Base.julia_cmd()) --startup-file=no -t 4 --project=$project -e $_TKB_THREADED`
        out = IOBuffer(); err = IOBuffer()
        p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
        msg = String(take!(out))
        success(p) || @info "tke_blocked.jl: the -t 4 subprocess failed" String(take!(err))
        @test success(p)
        m = match(r"threads (\d+) runs (\d+) worst ([0-9.eE+-]+) stats (true|false)", msg)
        @test m !== nothing
        if m !== nothing
            @test parse(Int, m[1]) == 4 && parse(Int, m[2]) >= 20
            @test parse(Float64, m[3]) <= 4eps()
            @test m[4] == "true"
        end
    end
end
