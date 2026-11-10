# The update of the Reynolds-stress closure (#560): its source update
# allocates nothing per cell, and the update with transport as one
# cache-blocked pass on the blocked driver of #533/#554 (src/
# prognostic_sgs/reynolds_stress_blocked.jl) against the separate loops it
# replaces (`_rsm_advance_separate!`, the reference, what loop_fusion =
# "never" runs).
#
# Every value of the pass is computed by the point body of the separate
# loop, so the stresses are expected to agree to round-off; the budget sums
# are the same row sums combined in the same order. Tolerances: the six
# stresses within 4 eps of the largest stress (a different vectorisation
# may reorder a row's sum; measured 0 in every case), the budget sums
# within a relative 1e-12 of the largest budget term (measured 0), the
# integer counts (clips, split cells, unrealizable inputs) exactly.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Operators: advance_sgs!
using Aeolus.Backends: CPUBackend, with_backend

const RSB = Aeolus.PrognosticSGS

include(joinpath(@__DIR__, "support", "rsc_blocked_cases.jl"))

@testset "Reynolds-stress source update: no allocation per cell (#560)" begin
    # The 6 × 6 system of every cell is stack-allocated (MMatrix); it
    # stays on the stack only if the elimination is inlined into the cell's
    # body. Before #560 it was not, and every cell allocated 368 B on the
    # heap. The update allocates its per-plane sums only: the same bytes on
    # a grid four times larger in x-y.
    al = map(((8, 8, 6), (16, 16, 6))) do n
        g = RegularGrid(; size = n, extent = 10.0 .* n, halo = (3, 3, 3))
        sim, c = rsc_case(g; transport = false)
        with_backend(CPUBackend(serial = true)) do
            RSB._rsm_update!(c, sim, 0.4); RSB._rsm_update!(c, sim, 0.4)
            @allocated RSB._rsm_update!(c, sim, 0.4)
        end
    end
    @test al[1] == al[2]
    @test al[2] < 2048
end

@testset "Reynolds-stress update as one cache-blocked pass (#560)" begin
    n = (14, 10, 40)        # 40 levels: 5 slabs of the WENO-Z5 pass (4 edge planes per end)
    Δt = 0.4
    nclip = Ref(0)          # cases in which the realizability projection acted
    for (sname, h, adv) in (("WENO-Z5", 3, :weno_z5), ("centred", 2, :centered)),
        solid in (false, true), rot in (false, true)
        g = RegularGrid(; size = n, extent = (140.0, 100.0, 200.0), halo = (h, h, h))
        sim, c = rsc_case(g; advection = adv, mask = solid, rotation = rot)
        τ0 = map(f -> copy(f.data), c.τ)
        ref, ref_s = rsc_run!(RSB._rsm_advance_separate!, c, sim, τ0, Δt)
        nclip[] += ref_s.step_clips > 0
        lz = RSB.rsm_blocked_lags(c)[2]
        for (bname, b) in (("serial", CPUBackend(serial = true)), ("threaded", CPUBackend(threshold = 0))),
            ns in 1:5, rev in (false, true)
            (ns > 1 && 2lz * ns > g.Nz) && continue
            (rev && ns == 1) && continue
            τb, s = with_backend(b) do
                rsc_run!((c, sim, Δt) -> RSB._rsm_advance_blocked!(c, sim, Δt; ns = ns, reverse_slabs = rev),
                         c, sim, τ0, Δt)
            end
            ok_τ = rsc_diff(τb, ref) <= 4eps()
            ok_s = rsc_stats_agree(s, ref_s)
            ok_τ && ok_s || @info "Reynolds-stress blocked pass disagrees" sname solid rot bname ns rev rsc_diff(τb, ref) s ref_s
            @test ok_τ
            @test ok_s
        end
    end
    @test nclip[] == 8         # the projection acted in every case
end

@testset "Reynolds-stress blocked pass: the lag of the off-diagonals (#560)" begin
    # The plain advection of the off-diagonals reads the old stresses hr
    # rows back, so the ring must hold hr rows (3 with WENO-Z5), more than
    # the TKE pass's max(1, hr − 2) = 1. The lags, and a pass run with the
    # TKE pass's lag disagreeing with the reference (so the test above
    # would catch a ring that is too short).
    g = RegularGrid(; size = (14, 10, 16), extent = (140.0, 100.0, 160.0), halo = (3, 3, 3))
    sim, c = rsc_case(g)
    @test RSB.rsm_blocked_lags(c) == (3, 4)
    g2 = RegularGrid(; size = (14, 10, 16), extent = (140.0, 100.0, 160.0), halo = (2, 2, 2))
    @test RSB.rsm_blocked_lags(rsc_case(g2; advection = :centered)[2]) == (1, 2)
    τ0 = map(f -> copy(f.data), c.τ)
    ref, _ = rsc_run!(RSB._rsm_advance_separate!, c, sim, τ0, 0.4)
    short(c, sim, Δt) = RSB._rsm_advance_blocked!(c, sim, Δt; lags = (1, 4))
    τs, _ = rsc_run!(short, c, sim, τ0, 0.4)
    @test rsc_diff(τs, ref) > 1e-6
end

@testset "Reynolds-stress blocked pass: no-flux lateral faces, a short column, TENO5, WENO-Z7, moist (#560)" begin
    # WENO-Z7 (halo 4): lag 4 and 5 edge planes per slab end; moist: the
    # update reads q_v (N² and the θ_v gradient) through the pass's arguments
    per = FieldBoundaryConditions(:periodic, :periodic, :wall)
    for (bcs, n, adv, h, moist) in ((FieldBoundaryConditions(:noflux, :noflux, :wall), (9, 7, 12), :weno_z5, 3, false),
                                    (per, (6, 1, 3), :weno_z5, 3, false),
                                    (per, (9, 7, 16), :teno5, 3, false),
                                    (per, (9, 7, 20), :weno_z7, 4, false),
                                    (per, (9, 7, 16), :weno_z5, 3, true))
        g = RegularGrid(; size = n, extent = 10.0 .* n, halo = (h, h, h))
        sim, c = rsc_case(g; bcs = bcs, advection = adv, mask = n[3] >= 12, moist = moist)
        moist && @test RSB._qv(c, sim) !== nothing
        @test RSB.rsm_blocked_lags(c) == (h, h + 1)
        τ0 = map(f -> copy(f.data), c.τ)
        ref, ref_s = rsc_run!(RSB._rsm_advance_separate!, c, sim, τ0, 0.4)
        for ns in (1, 2)
            (ns > 1 && 2 * (h + 1) * ns > g.Nz) && continue
            τb, s = rsc_run!((c, sim, Δt) -> RSB._rsm_advance_blocked!(c, sim, Δt; ns = ns), c, sim, τ0, 0.4)
            @test rsc_diff(τb, ref) <= 4eps()
            @test rsc_stats_agree(s, ref_s)
        end
    end
end

@testset "Reynolds-stress blocked pass: open lateral faces and top (#560)" begin
    # ν and the ratios have no boundary data: their open-face halos are the
    # mirror of the first interior layer, which the boundary shell reproduces
    for halo in (2, 3)
        g = RegularGrid(; size = (12, 10, 16), extent = (120.0, 100.0, 160.0), halo = (halo, halo, halo))
        bcs = open_boundary_conditions(g, ProfileBoundaryData(u = 3.0, v = -1.0, w = 0.0, θ = 300.0); halo = halo)
        sim, c = rsc_case(g; bcs = bcs, advection = halo == 3 ? :weno_z5 : :centered)
        Aeolus.BoundaryConditions.update_open_boundaries!(sim, 0.0, 0.4)
        @test RSB._rsm_blocked_blocker(c, sim) === nothing
        τ0 = map(f -> copy(f.data), c.τ)
        ref, ref_s = rsc_run!(RSB._rsm_advance_separate!, c, sim, τ0, 0.4)
        for ns in (1, 2), rev in (false, true)
            (ns > 1 && 2 * RSB.rsm_blocked_lags(c)[2] * ns > g.Nz) && continue
            τb, s = rsc_run!((c, sim, Δt) -> RSB._rsm_advance_blocked!(c, sim, Δt; ns = ns, reverse_slabs = rev),
                             c, sim, τ0, 0.4)
            @test rsc_diff(τb, ref) <= 4eps()
            @test rsc_stats_agree(s, ref_s)
        end
    end
end

@testset "Reynolds-stress blocked pass: advance_sgs!, loop_fusion, report, allocation (#560)" begin
    g = RegularGrid(; size = (16, 12, 16), extent = (160.0, 120.0, 160.0), halo = (3, 3, 3))
    sim, c = rsc_case(g; surface = true, mask = true)
    τ0 = map(f -> copy(f.data), c.τ)
    # "always" runs the blocked pass, "never" the separate loops; the
    # surface value follows either
    Aeolus.Simulations.set_loop_fusion!(sim, :always)
    τb, sb = rsc_run!(advance_sgs!, c, sim, τ0, 0.4)
    Aeolus.Simulations.set_loop_fusion!(sim, :never)
    τs, ss = rsc_run!(advance_sgs!, c, sim, τ0, 0.4)
    @test rsc_diff(τb, τs) <= 4eps()
    @test rsc_stats_agree(sb, ss)
    @test ss.surface_energy != 0
    r = Aeolus.Simulations.sweep_report(sim)
    @test r["reynolds_stress_update_blocked"] == "some" && !haskey(r, "reynolds_stress_not_blocked")
    # "if_possible" (the default) takes the pass only where it was measured
    # faster: centred advection on several threads (#560). WENO-Z5, or one
    # thread: the separate loops, with the reason in the report (the
    # threaded centred case is checked in the -t 4 subprocess below)
    simi, ci = rsc_case(g)
    @test Aeolus.Simulations.loop_fusion(simi) === :if_possible
    advance_sgs!(ci, simi, 0.4)
    ri = Aeolus.Simulations.sweep_report(simi)
    @test ri["reynolds_stress_update_blocked"] == "none" && occursin("#560", ri["reynolds_stress_not_blocked"])
    g2 = RegularGrid(; size = (16, 12, 16), extent = (160.0, 120.0, 160.0), halo = (2, 2, 2))
    simc, cc = rsc_case(g2; advection = :centered)
    with_backend(CPUBackend(serial = true)) do
        @test RSB._rsm_blocked_slower(cc, simc) !== nothing
    end
    # :weno5 has no row form: the separate loops under if_possible (with the
    # reason in the report), an error under always
    sim5, c5 = rsc_case(g; advection = :weno5)
    @test RSB._rsm_blocked_blocker(c5, sim5) !== nothing
    advance_sgs!(c5, sim5, 0.4)
    r5 = Aeolus.Simulations.sweep_report(sim5)
    @test r5["reynolds_stress_update_blocked"] == "none" && occursin("weno5", r5["reynolds_stress_not_blocked"])
    Aeolus.Simulations.set_loop_fusion!(sim5, :always)
    @test_throws ArgumentError advance_sgs!(c5, sim5, 0.4)
    # without transport the update is one pointwise pass in every setting:
    # nothing to report
    simn, cn = rsc_case(g; transport = false)
    Aeolus.Simulations.set_loop_fusion!(simn, :always)
    advance_sgs!(cn, simn, 0.4)
    @test !haskey(Aeolus.Simulations.sweep_report(simn), "reynolds_stress_update_blocked")
    # allocation: the pass reuses its scratch and its entry is type-stable,
    # so on a serial backend it allocates no more than the separate loops
    # (both: the bytes of the halo fills and the advection scheme's call)
    Aeolus.Simulations.set_loop_fusion!(sim, :if_possible)
    with_backend(CPUBackend(serial = true)) do
        for (adv, solid) in ((:weno_z5, false), (:weno_z5, true))
            sa, ca = rsc_case(g; advection = adv, mask = solid)
            RSB._rsm_advance_blocked!(ca, sa, 0.4); RSB._rsm_advance_separate!(ca, sa, 0.4)
            ab = @allocated RSB._rsm_advance_blocked!(ca, sa, 0.4)
            as = @allocated RSB._rsm_advance_separate!(ca, sa, 0.4)
            @test ab <= as
        end
    end
end

# The slabs of the pass run concurrently only with more than one thread;
# the test gate runs each file at one thread. So the concurrent case runs
# here in a -t 4 subprocess: WENO-Z5 and centred,
# with the solid mask, at 2 to 5 slabs, against the separate loops (the
# stresses within 4 eps of the largest, the statistics as above). Also
# here, because "if_possible" takes the pass only on a threaded launch with
# centred advection (and "always" refuses the step core's unfused terms):
# the selection, and three `step!`s with "if_possible" against "never",
# whose stresses and νt, κt agree within 1e-12 of their maxima (each update
# agrees to round-off; the flow then evolves through the same arithmetic,
# so only round-off of a different summation order can separate them).
const _RSB_THREADED = """
using Aeolus
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Fields: interior_view
const RSB = Aeolus.PrognosticSGS
include($(repr(joinpath(@__DIR__, "support", "rsc_blocked_cases.jl"))))
n = (14, 10, 40); worst = 0.0; stats_ok = true; runs = 0
for (h, adv) in ((2, :centered), (3, :weno_z5)), solid in (false, true)
    g = RegularGrid(; size = n, extent = (140.0, 100.0, 200.0), halo = (h, h, h))
    sim, c = rsc_case(g; advection = adv, mask = solid, rotation = true)
    τ0 = map(f -> copy(f.data), c.τ)
    ref, ref_s = rsc_run!(RSB._rsm_advance_separate!, c, sim, τ0, 0.4)
    for ns in 2:5
        2 * RSB.rsm_blocked_lags(c)[2] * ns > g.Nz && continue
        τb, s = with_backend(CPUBackend(threshold = 0)) do
            rsc_run!((c, sim, Δt) -> RSB._rsm_advance_blocked!(c, sim, Δt; ns = ns), c, sim, τ0, 0.4)
        end
        global worst = max(worst, rsc_diff(τb, ref))
        global stats_ok &= rsc_stats_agree(s, ref_s)
        global runs += 1
    end
end
# the selection under if_possible: threaded centred blocked, WENO-Z5 not
sel = with_backend(CPUBackend(threshold = 0)) do
    g = RegularGrid(; size = n, extent = (140.0, 100.0, 200.0), halo = (3, 3, 3))
    RSB._rsm_blocked_slower(rsc_case(g; advection = :centered)[2], nothing) === nothing &&
        RSB._rsm_blocked_slower(rsc_case(g)[2], nothing) !== nothing
end
steps = with_backend(CPUBackend(threshold = 0)) do
    g = RegularGrid(; size = (12, 10, 16), extent = (120.0, 100.0, 160.0), halo = (2, 2, 2))
    map((:if_possible, :never)) do setting
        sim, c = rsc_case(g; advection = :centered, mask = true)
        Aeolus.Simulations.set_loop_fusion!(sim, setting)
        for _ in 1:3
            step!(sim, 0.05; sgs = c)
        end
        (τ = map(f -> copy(interior_view(f)), c.τ), νt = copy(interior_view(sim.νt)),
         κt = copy(interior_view(sim.κt)), rep = Aeolus.Simulations.sweep_report(sim)["reynolds_stress_update_blocked"])
    end
end
sb, ss = steps
rel(x, y) = maximum(abs.(x .- y)) / maximum(abs, y)
steps_ok = sb.rep == "all" && ss.rep == "none" && rsc_diff(sb.τ, ss.τ) <= 1e-12 &&
           maximum(ss.νt) > 0 && maximum(ss.κt) > 0 && rel(sb.νt, ss.νt) <= 1e-12 && rel(sb.κt, ss.κt) <= 1e-12
print("threads ", Threads.nthreads(), " runs ", runs, " worst ", worst, " stats ", stats_ok, " select ", sel,
      " steps ", steps_ok, " (", sb.rep, "/", ss.rep, " dτ ", rsc_diff(sb.τ, ss.τ), " dνt ", rel(sb.νt, ss.νt), ")")
"""

@testset "Reynolds-stress blocked pass: slabs run concurrently, -t 4 (#560)" begin
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --startup-file=no -t 4 --project=$project -e $_RSB_THREADED`
    out = IOBuffer(); err = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    msg = String(take!(out))
    success(p) || @info "reynolds_stress_blocked.jl: the -t 4 subprocess failed" String(take!(err))
    @test success(p)
    m = match(r"threads (\d+) runs (\d+) worst ([0-9.eE+-]+) stats (true|false) select (true|false) steps (true|false)", msg)
    @test m !== nothing
    if m !== nothing
        @test parse(Int, m[1]) == 4 && parse(Int, m[2]) >= 10
        @test parse(Float64, m[3]) <= 4eps()
        @test m[4] == "true"
        @test m[5] == "true"
        m[6] == "true" || @info "reynolds_stress_blocked.jl: the subprocess's steps" msg
        @test m[6] == "true"
    end
end
