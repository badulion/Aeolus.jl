# SC-c (#403, scaling epic #406): the nest parent stepped concurrently
# with the child's substeps. In a one-way run the parent step of window
# n + 1 needs nothing but the parent at the end of window n, so it runs on
# its own task (`prefetch_parent_step!`, `step_nested!(run, n)`) while the
# child subcycles through window n. The boundary data move only at the
# existing synchronisation points, so the result must be the sequential
# one bit for bit. These tests check, on synthetic boxes:
#
#   1. step_nested!(run, n) == n × step_nested!(run), bit for bit: child,
#      parent, clocks, statistics, step records — fixed and adaptive parent
#      steps, SSP-RK3 and the projection step, a moist pair with a tracer,
#      a sponged child, a z-stretched parent; and the parent steps were
#      really taken by the prefetching task (stats.prefetched = n − 1);
#   2. the externally driven child (the Berlin pattern: begin_parent_step!,
#      prefetch_parent_step!, own child steps, end_parent_step!) == the
#      sequential step_nested!;
#   3. the data dependency is exact: while a step is prefetched the
#      parent is one window ahead and the child reads only the previous
#      window's data; a begin_parent_step! asking for another dt than was
#      prefetched throws instead of using it;
#   4. two-way runs and sync = :shared_min_dt cannot be prefetched (the
#      next parent step needs the child): prefetch_parent_step! refuses
#      them, step_nested!(run, n) steps them sequentially, bit-identical;
#   5. a failing prefetched parent step surfaces at the next window.

using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Fields: interior_view
using Aeolus.StretchedGrids: StretchedGrid

_cc_fields(s) = (:u, :v, :w, :θ, :p)

"Two identical copies of a one-way-able nested pair with a random projected parent."
function _cc_pairs(; seed = 403, moisture = false, tracer = false, stretched = false, open_top = true)
    rng = Xoshiro(seed)
    pairs = map(1:2) do _
        pm = stretched ?
            StretchedGrid(Float64; size = (16, 16, 8), extent = (800.0, 800.0, 500.0), halo = (3, 3, 3),
                          stretching = (:custom, [0.0, 50.0, 100.0, 150.0, 200.0, 260.0, 330.0, 410.0, 500.0])) :
            RegularGrid(Float64; size = (16, 16, 8), extent = (800.0, 800.0, 400.0), halo = (3, 3, 3))
        nest = nested_grid(pm, ((200.0, 500.0), (250.0, 550.0), (0.0, 200.0)), (2, 2, 2); halo = (3, 3, 3))
        cm = nest_child_mesh(nest)
        p = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall), ν = 0.5,
                                 moisture)
        c = BoussinesqSimulation(cm; bcs = nest_bcs(cm; bottom = :wall, top = open_top ? :nest : :wall,
                                                    moisture, scalars = tracer ? (:co2,) : ()),
                                 ν = 0.5, moisture)
        (p, c, nest)
    end
    p1 = pairs[1][1]
    u0 = 2.0 .+ randn(rng, size(interior_view(p1.u)))
    v0 = 0.5 .* randn(rng, size(interior_view(p1.v)))
    θ0 = 300 .+ 0.2 .* randn(rng, size(interior_view(p1.θ)))
    q0 = 0.01 .+ 0.001 .* rand(rng, size(interior_view(p1.θ)))
    for (p, c, nest) in pairs
        interior_view(p.u) .= u0; interior_view(p.v) .= v0; interior_view(p.θ) .= θ0
        moisture && (interior_view(p.qv) .= q0)
        step!(p, 1e-10); p.t = 0.0
        st = nest_interp_state(nest)
        interpolate_parent_state!(st, p, nest; order = 3, region = :all)
        for (cf, sf) in ((c.u, st.u), (c.v, st.v), (c.w, st.w), (c.θ, st.θ))
            cf.data .= sf.data
        end
        moisture && fill!(c.qv.data, 0.01)
    end
    return pairs
end

function _cc_runs(; tracer = false, kw...)
    pairs = _cc_pairs(; tracer, NamedTuple(k => v for (k, v) in kw if k in (:seed, :moisture, :stretched, :open_top))...)
    rkw = merge((; feedback = :one_way), NamedTuple(k => v for (k, v) in kw if !(k in (:seed, :moisture, :stretched, :open_top))))
    map(pairs) do (p, c, nest)
        sc = tracer ? (:co2 => (Field(CellCenter(), c.mesh), _cc_tracer(p)),) : ()
        NestedRun(p, c, nest; scalars = sc, rkw...)
    end
end

function _cc_tracer(p)
    f = Field(CellCenter(), p.mesh)
    interior_view(f) .= reshape(collect(1.0:length(interior_view(f))), size(interior_view(f))) ./ length(interior_view(f))
    return f
end

# The stored points of a state member (a halo-only state keeps six slabs).
_cc_stored(a::Array) = a
_cc_stored(a) = vcat((vec(getfield(a, f)) for f in (:xl, :xh, :yl, :yh, :zl, :zh))...)

"""
Every prognostic of both domains, every bound scalar and the clocks
identical; with `stats`, also the run's statistics and step records (an
externally driven child is not counted by the run).
"""
function _cc_same(A, B; stats = true)
    same = all(getproperty(A.child, f).data == getproperty(B.child, f).data for f in _cc_fields(A.child)) &&
           all(getproperty(A.parent, f).data == getproperty(B.parent, f).data for f in _cc_fields(A.parent))
    if A.parent.has_moisture
        same &= A.parent.qv.data == B.parent.qv.data && A.child.qv.data == B.child.qv.data
    end
    for (fa, fb) in zip(A.s_old.child, B.s_old.child)
        same &= fa.data == fb.data
    end
    same &= A.parent.t == B.parent.t && A.child.t == B.child.t && A.t == B.t
    stats || return same
    sa, sb = A.stats, B.stats
    same &= (sa.parent_steps, sa.child_substeps, sa.parent_retries, sa.child_retries, sa.n_min, sa.n_max,
             sa.dt_parent_min, sa.dt_parent_max, sa.dt_child_min, sa.dt_child_max, sa.max_child_cfl) ==
            (sb.parent_steps, sb.child_substeps, sb.parent_retries, sb.child_retries, sb.n_min, sb.n_max,
             sb.dt_parent_min, sb.dt_parent_max, sb.dt_child_min, sb.dt_child_max, sb.max_child_cfl)
    ra, rb = step_record(A), step_record(B)
    same &= (ra.steps, ra.dt_sum, ra.dt, ra.limiter, ra.retries_sum) == (rb.steps, rb.dt_sum, rb.dt, rb.limiter, rb.retries_sum)
    pa, pb = step_record(A.parent), step_record(B.parent)
    same &= (pa.steps, pa.dt_sum, pa.limiter) == (pb.steps, pb.dt_sum, pb.limiter)
    return same
end

@testset "SC-c: the nest parent stepped concurrently with the child (#403)" begin
@info "era2_nest_concurrent.jl: $(Threads.nthreads()) Julia threads"

# -------------------------------------------------------------------------
@testset "step_nested!(run, n) == n × step_nested!(run), bit for bit" begin
    rk3 = domain_advance(step_rk3!)
    cases = (
        ("fixed dt, RK3, 3 substeps", (; advance_parent = rk3, advance_child = rk3, cfl_parent = nothing,
                                         cfl_child = nothing, substeps = 3), 0.6),
        ("adaptive parent and child, projection step", (; cfl_parent = CFLConstraints(C_advective = 0.4),
                                                          cfl_child = CFLConstraints(C_advective = 0.3)), nothing),
        ("moist pair + tracer, adaptive child", (; moisture = true, tracer = true, advance_parent = rk3,
                                                   advance_child = rk3, cfl_parent = nothing,
                                                   cfl_child = CFLConstraints(C_advective = 0.5)), 0.8),
        ("sponged child (storage = :full states)", (; sponge = NestSponge(; width = 2), cfl_parent = nothing,
                                                      cfl_child = nothing), 0.5),
        ("z-stretched parent, reuse_boundary = false", (; stretched = true, reuse_boundary = false,
                                                          cfl_parent = nothing, cfl_child = nothing), 0.5),
    )
    for (label, kw, dt) in cases
        A, B = _cc_runs(; kw...)
        n = 5
        step_nested!(A, n; dt, concurrent = true)
        for _ in 1:n
            step_nested!(B; dt)
        end
        same = _cc_same(A, B)
        @printf("  %-46s %d windows (%d child substeps): bit-identical = %s, prefetched %d\n",
                label, n, A.stats.child_substeps, same, A.stats.prefetched)
        @test same
        @test A.stats.prefetched == n - 1          # every window but the first
        @test B.stats.prefetched == 0
        @test A.ahead === nothing                  # nothing pending at return
        # the child moved and saw the parent: not a trivial identity
        @test A.child.t == A.parent.t > 0
        @test maximum(abs, interior_view(A.child.u)) > 0.1
    end
end

# -------------------------------------------------------------------------
@testset "externally driven child with prefetch_parent_step! == step_nested!" begin
    A, B = _cc_runs(; cfl_parent = nothing, cfl_child = nothing, substeps = 3)
    Δt_p = 0.6
    for _ in 1:4
        step_nested!(A; dt = Δt_p)
    end
    c = B.child
    nwin = 4
    for k in 1:nwin
        w = begin_parent_step!(B; dt = Δt_p)
        # the next window's parent step, on its own task, while the child steps
        k < nwin && prefetch_parent_step!(B, w; dt = Δt_p)
        while w.t_child_end - c.t > 1e-9 * Δt_p
            h = subcycle_dt(w.t_child_end - c.t, Δt_p / 3)
            hook = begin_child_step!(B)
            step!(c, h; body_force! = hook)
            end_child_step!(B)
        end
        end_parent_step!(B, w)
    end
    same = _cc_same(A, B; stats = false)
    @printf("  external driver, 4 windows × 3 child steps: bit-identical = %s, prefetched %d\n",
            same, B.stats.prefetched)
    @test same
    @test B.stats.prefetched == nwin - 1
end

# -------------------------------------------------------------------------
@testset "the exchange is exact: the parent runs one window ahead, the child reads the old data" begin
    A, = _cc_runs(; cfl_parent = nothing, cfl_child = nothing, substeps = 2)
    w = begin_parent_step!(A; dt = 0.5)
    @test A.parent.t == w.t_parent == 0.5
    b_end = copy(_cc_stored(A.s_new.u.data))         # B(t_p + Δt_p) of this window
    prefetch_parent_step!(A, w; dt = 0.5)
    @test_throws ArgumentError prefetch_parent_step!(A, w; dt = 0.5)   # one at a time
    wait_parent_step!(A)
    @test A.parent.t == 1.0                          # the parent took the next window's step
    @test _cc_stored(A.s_new.u.data) == b_end        # and the child's window data did not change
    @test A.data.b === A.s_new && A.data.a === A.s_old
    while w.t_child_end - A.child.t > 1e-9
        step!(A.child, 0.25)
    end
    end_parent_step!(A, w)
    @test A.t == 0.5                                 # the run's clock is the window's, not the parent's
    @test step_record(A).limiter === w.limiter
    # the next window must ask for the dt that was prefetched
    @test_throws ErrorException begin_parent_step!(A; dt = 0.4)
    @test A.ahead === nothing                        # a refused step is not taken twice ...
    @test A.parent.t == 1.0                          # ... and the parent is not stepped again
end

@testset "a window opened from the prefetched step: its data are the parent's at both ends" begin
    A, B = _cc_runs(; cfl_parent = nothing, cfl_child = nothing, substeps = 2)
    w = begin_parent_step!(A; dt = 0.5)
    prefetch_parent_step!(A, w; dt = 0.5)
    while w.t_child_end - A.child.t > 1e-9
        step!(A.child, 0.25)
    end
    end_parent_step!(A, w)
    w2 = begin_parent_step!(A; dt = 0.5)
    @test w2.t_parent == A.parent.t == 1.0 && w2.Δt == 0.5
    # B(t_p + Δt_p) is the prolongation of the parent now (one-way: unchanged since)
    ref = nest_interp_state(A.nest; storage = :shell)
    interpolate_parent_state!(ref, A.parent, A.nest; order = A.order, region = A.region)
    @test _cc_stored(A.s_new.u.data) == _cc_stored(ref.u.data) &&
          _cc_stored(A.s_new.θ.data) == _cc_stored(ref.θ.data)
    @test A.stats.prefetched == 1
    @test A.stats.wall_parent > 0 && A.stats.wall_parent_wait >= 0
end

# -------------------------------------------------------------------------
@testset "two-way and :shared_min_dt runs are not prefetched (the next step needs the child)" begin
    A, B = _cc_runs(; feedback = :two_way, cfl_parent = nothing, cfl_child = nothing, substeps = 2)
    w = begin_parent_step!(A; dt = 0.5)
    @test_throws ArgumentError prefetch_parent_step!(A, w; dt = 0.5)
    A, B = _cc_runs(; feedback = :two_way, cfl_parent = nothing, cfl_child = nothing, substeps = 2)
    step_nested!(A, 4; dt = 0.5, concurrent = true)
    for _ in 1:4
        step_nested!(B; dt = 0.5)
    end
    same = _cc_same(A, B) && A.nest.ledger == B.nest.ledger
    @printf("  two-way, concurrent = true: stepped sequentially, bit-identical = %s, prefetched %d\n",
            same, A.stats.prefetched)
    @test same
    @test A.stats.prefetched == 0
    C, = _cc_runs(; sync = :shared_min_dt, cfl_parent = CFLConstraints(), cfl_child = CFLConstraints())
    w = begin_parent_step!(C)
    @test_throws ArgumentError prefetch_parent_step!(C, w)
    D, = _cc_runs(; cfl_parent = nothing, cfl_child = nothing)
    w = begin_parent_step!(D; dt = 0.5)
    @test_throws ArgumentError prefetch_parent_step!(D, w)      # no cfl_parent and no dt
end

# -------------------------------------------------------------------------
@testset "a failing prefetched parent step surfaces at the next window" begin
    pairs = _cc_pairs()
    p, c, nest = pairs[1]
    calls = Ref(0)
    bad = domain_advance((s, h; kw...) -> (calls[] += 1; calls[] >= 2 && throw(ArgumentError("parent core failed"));
                                           step!(s, h; kw...)))
    run = NestedRun(p, c, nest; feedback = :one_way, advance_parent = bad, cfl_parent = nothing, cfl_child = nothing,
                    validate = false)
    w = begin_parent_step!(run; dt = 0.5)
    prefetch_parent_step!(run, w; dt = 0.5)
    while w.t_child_end - c.t > 1e-9
        step!(c, 0.25)
    end
    end_parent_step!(run, w)
    @test_throws ArgumentError begin_parent_step!(run; dt = 0.5)
    @test run.ahead === nothing
end

end
