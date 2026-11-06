using Test
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view

# =========================================================================
# N-nest-c (#257) — NestedRun: the time coupling of parent and child.
# Per-domain advance closures, subcycling under adaptive steppers, linear
# temporal interpolation of the boundary data at the child's stage times,
# retries that roll back only the failing domain, one-/two-way feedback.
# =========================================================================

# --- coordinates of a cell centre / x-face (storage index, halos included)
_nr_xc(m, i) = m.x0 + (i - m.Hx - 0.5) * m.dx
_nr_yc(m, j) = m.y0 + (j - m.Hy - 0.5) * m.dy
_nr_zc(m, k) = m.z0 + (k - m.Hz - 0.5) * m.dz
_nr_xf(m, i) = m.x0 + (i - m.Hx - 1) * m.dx

"Set every storage cell of the four prognostics from `f(name, x, y, z)`."
function _nr_set_fields!(sim, f)
    m = sim.mesh
    for k in axes(sim.θ.data, 3), j in axes(sim.θ.data, 2), i in axes(sim.θ.data, 1)
        x = _nr_xc(m, i); y = _nr_yc(m, j); z = _nr_zc(m, k)
        sim.θ.data[i, j, k] = f(:θ, x, y, z)
    end
    for k in axes(sim.u.data, 3), j in axes(sim.u.data, 2), i in axes(sim.u.data, 1)
        sim.u.data[i, j, k] = f(:u, _nr_xf(m, i), _nr_yc(m, j), _nr_zc(m, k))
    end
    fill!(sim.v.data, 0.0); fill!(sim.w.data, 0.0)
    return sim
end

"Copy the parent prolongated over the whole child (a smooth initial child)."
function _nr_init_child_from_parent!(csim, psim, nest)
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, psim, nest; order=3, region=:all)
    for (cf, sf) in ((csim.u, st.u), (csim.v, st.v), (csim.w, st.w), (csim.θ, st.θ))
        cf.data .= sf.data
    end
    return csim
end

"A user core that sets the parent analytically: fields `f(name, x, y, z, t)`."
_nr_analytic_core(f) = (sim, dt; kw...) -> begin
    sim.t += dt
    tt = sim.t
    _nr_set_fields!(sim, (n, x, y, z) -> f(n, x, y, z, tt))
    sim
end


"""
The time whose boundary data the child core's current tendency evaluation
reads: the run's substep window `[run.sub_t0, run.sub_t0 + run.sub_h]` and
the stage counter `ctr` (reset when a new substep starts).
"""
function _nr_stage_time!(ctr, run)
    if ctr[1] != run.sub_t0
        ctr[1] = run.sub_t0; ctr[2] = 0
    end
    ctr[2] += 1
    return run.sub_t0 + stage_times(run.advance_child)[Int(ctr[2])] * run.sub_h
end

# =========================================================================
@testset "NestedRun time coupling (N-nest-c, #257)" begin

@testset "stage times of the cores" begin
    @test stage_times(domain_advance(step_rk3!)) == (0.0, 1.0, 0.5)
    @test stage_times(domain_advance(step_imex!, ARK2())) == (0.0, ARK2().γ)
end

# Parent 16³ on the unit cube, child (0.25, 0.75)³ at r = 2. The parent is
# advanced by a user core that sets it analytically, linear in x, y, z, t:
# the child's boundary data at every substep and RK3 stage are exact. (16³,
# not 8³: on 8³ the prolongation stencils of the child halos reach the
# parent's periodic halos, where the linear field is not linear.)
@testset "exact temporal interpolation at every substep and stage" begin
    pm = RegularGrid(Float64; size=(16, 16, 16), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    # u divergence-free (no x dependence): the open faces' mass-flux
    # correction leaves it unchanged
    f(n, x, y, z, t) = n === :θ ? 300.0 + 2x - 3y + z + 7.0 * t :
                                  1.0 + 0.25y - 0.125z - 4.0 * t
    psim = BoussinesqSimulation(pm)
    _nr_set_fields!(psim, (n, x, y, z) -> f(n, x, y, z, 0.0))
    csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm))
    _nr_init_child_from_parent!(csim, psim, nest)
    RUN = Ref{Any}(nothing)
    err = Ref(0.0); nrec = Ref(0); ctr = [NaN, 0.0]
    ii, jj, kk = interior(cm)
    function record!(sim, Δt)            # runs where the core reads the stage halos
        ts = _nr_stage_time!(ctr, RUN[])
        for k in axes(sim.θ.data, 3), j in axes(sim.θ.data, 2), i in axes(sim.θ.data, 1)
            (i in ii) && (j in jj) && (k in kk) && continue
            ex = f(:θ, _nr_xc(cm, i), _nr_yc(cm, j), _nr_zc(cm, k), ts)
            exu = f(:u, _nr_xf(cm, i), _nr_yc(cm, j), _nr_zc(cm, k), ts)
            err[] = max(err[], abs(sim.θ.data[i, j, k] - ex) / abs(ex),
                        abs(sim.u.data[i, j, k] - exu) / abs(exu))
        end
        nrec[] += 1
        return sim
    end
    run = NestedRun(psim, csim, nest; feedback = :one_way,
                    advance_parent = domain_advance(_nr_analytic_core(f)),
                    advance_child = domain_advance(step_rk3!; body_force! = record!),
                    cfl_parent = nothing, cfl_child = nothing, substeps = 3)
    RUN[] = run
    for _ in 1:2
        step_nested!(run; dt = 1e-3)
    end
    @test nrec[] == 2 * 3 * 3            # 2 parent steps × 3 substeps × 3 stages
    @test err[] < 1e-14
    s = boundary_time(run, run.t0 + 0.3 * run.Δt)
    tq = run.t0 + 0.3 * run.Δt
    @test maximum(abs(s.θ.data[i, j, k] - f(:θ, _nr_xc(cm, i), _nr_yc(cm, j), _nr_zc(cm, k), tq))
                  for k in axes(s.θ.data, 3), j in axes(s.θ.data, 2), i in axes(s.θ.data, 1)
                  if !((i in ii) && (j in jj) && (k in kk))) < 1e-11
    @test run.child.t == run.parent.t
end

@testset "linear temporal interpolation: error within Δt² A ω²/8 (sin ωt)" begin
    ω = 2π; A = 1.5; dtp = 1 / 10
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    f(n, x, y, z, t) = n === :θ ? 300.0 + A * sin(ω * t) : 0.0
    psim = BoussinesqSimulation(pm); _nr_set_fields!(psim, (n, x, y, z) -> f(n, x, y, z, 0.0))
    csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm))
    _nr_init_child_from_parent!(csim, psim, nest)
    RUN = Ref{Any}(nothing); err = Ref(0.0); ctr = [NaN, 0.0]
    rec!(sim, Δt) = (ts = _nr_stage_time!(ctr, RUN[]);
                     err[] = max(err[], abs(sim.θ.data[1, 1, 1] - (300.0 + A * sin(ω * ts)))); sim)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = domain_advance(_nr_analytic_core(f)),
                    advance_child = domain_advance(step_rk3!; body_force! = rec!),
                    cfl_parent = nothing, cfl_child = nothing, substeps = 2)
    RUN[] = run
    for _ in 1:10
        step_nested!(run; dt = dtp)
    end
    bound = dtp^2 * A * ω^2 / 8           # leading error of linear interpolation
    @test 0 < err[] <= bound * (1 + 1e-9)
end

# Uniform flow along x: the child's advective limit is the parent's over r.
@testset "substep count: child limit Δt_p/r gives n = r" begin
    pm = RegularGrid(Float64; size=(8, 4, 4), extent=(1.0, 0.5, 0.5), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.5), (0.125, 0.375), (0.125, 0.375)), 3; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm); fill!(psim.u.data, 1.3); fill!(psim.θ.data, 300.0)
    csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm))
    fill!(csim.u.data, 1.3); fill!(csim.θ.data, 300.0)
    run = NestedRun(psim, csim, nest; feedback = :one_way, cfl_parent = CFLConstraints(C_advective = 0.5))
    step_nested!(run)
    st = run.stats
    @test st.n_min == 3 && st.n_max == 3
    @test st.dt_child_max ≈ st.dt_parent_max / 3 rtol = 1e-12
    @test st.max_child_cfl <= 0.5 * (1 + 1e-9)
end

"""
Parent and child with a Taylor-Green flow plus a mean wind and a θ blob;
the child's x and y faces are open nest faces (N-nest-d).
"""
function _nr_tg_case(; Np = 16, r = 2, U = 0.5, ν = 5e-3, kwargs...)
    L = 1.0
    pm = RegularGrid(Float64; size=(Np, Np, 4), extent=(L, L, L / 4), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.0, L / 4)), (r, r, 1); halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; ν = ν)
    for k in axes(psim.u.data, 3), j in axes(psim.u.data, 2), i in axes(psim.u.data, 1)
        x = _nr_xf(pm, i); xc = _nr_xc(pm, i); y = _nr_yc(pm, j)
        yf = pm.y0 + (j - pm.Hy - 1) * pm.dy
        psim.u.data[i, j, k] = U + sin(2π * x) * cos(2π * y)
        psim.v.data[i, j, k] = -cos(2π * xc) * sin(2π * yf)
        psim.θ.data[i, j, k] = 300.0 + exp(-((xc - 0.4)^2 + (y - 0.5)^2) / 0.02)
    end
    step!(psim, 1e-6); psim.t = 0.0           # project once
    csim = BoussinesqSimulation(cm; ν = ν, bcs = nest_bcs(cm, :nest, :nest, :periodic))
    _nr_init_child_from_parent!(csim, psim, nest)
    return NestedRun(psim, csim, nest; merge((; feedback = :one_way), NamedTuple(kwargs))...)
end

@testset "one-way: the parent is a standalone run" begin
    run = _nr_tg_case(; feedback = :one_way, cfl_parent = CFLConstraints())
    alone = deepcopy(run.parent)
    c = CFLConstraints(); dtp = nothing
    for _ in 1:3
        step_nested!(run)
        dt = adaptive_dt(alone, c; dt_prev = dtp)
        step!(alone, dt); dtp = dt
    end
    @test agree(interior_view(run.parent.u), interior_view(alone.u)) &&
          agree(interior_view(run.parent.θ), interior_view(alone.θ))
end

@testset "two-way adaptive run: clock, child CFL, interface mass" begin
    run = _nr_tg_case(; feedback = :two_way, cfl_parent = CFLConstraints(C_advective = 0.5))
    kep0 = kinetic_energy(run.parent)
    mass = 0.0
    for n in 1:5
        step_nested!(run)
        mass = max(mass, nest_conservation_error(run.parent, run.child, run.nest, :mass))
    end
    st = run.stats
    @test run.child.t == run.parent.t
    @test st.max_child_cfl <= 0.5 * (1 + 1e-9) && st.n_min >= 2
    @test mass < 1e-10
    @test kinetic_energy(run.parent) <= kep0 * (1 + 1e-12)     # viscous: no energy gain
end

# Retry isolation: the child carries uniform flow decaying under an exactly
# integrated drag (independent of the dt sequence); a child NaN rolls back
# and redoes only the child.
@testset "retry isolation: a child NaN rolls back only the child" begin
    function retry_case(inject::Bool)
        pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
        nest = nested_grid(pm, ((0.0, 1.0), (0.0, 1.0), (0.25, 0.75)), 2; halo=(3, 3, 3))
        cm = nest_child_mesh(nest)
        psim = BoussinesqSimulation(pm)
        for k in axes(psim.u.data, 3), j in axes(psim.u.data, 2), i in axes(psim.u.data, 1)
            psim.u.data[i, j, k] = 1.0 + 0.3 * sin(2π * _nr_xf(pm, i)) * cos(2π * _nr_yc(pm, j))
        end
        fill!(psim.θ.data, 300.0)
        csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm, :periodic, :periodic, :nest))
        fill!(csim.u.data, 1.0); fill!(csim.θ.data, 300.0)
        ncalls = Ref(0); armed = Ref(inject)
        counting(sim, dt; kw...) = (ncalls[] += 1; step!(sim, dt; kw...))
        pp(s, Δt) = begin
            interior_view(s.u) .*= exp(-0.5 * Δt)
            armed[] && s.t > 0.03 && (s.u.data[6, 6, 6] = NaN; armed[] = false)
            s
        end
        run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = domain_advance(counting),
                        advance_child = domain_advance(; post_predictor! = pp),
                        cfl_parent = CFLConstraints(dt_max = 0.02))
        for _ in 1:4
            step_nested!(run)
        end
        return run, ncalls[]
    end
    ref, ncp_ref = retry_case(false)
    inj, ncp = retry_case(true)
    @test inj.stats.child_retries == 1 && inj.stats.parent_retries == 0
    @test ncp == ncp_ref == 4                          # the parent is never advanced twice
    @test agree(interior_view(inj.parent.u), interior_view(ref.parent.u))
    @test maximum(abs, interior_view(inj.child.u) .- interior_view(ref.child.u)) < 1e-12
    @test inj.child.t == inj.parent.t
end

# #327 M6: only a numerical failure is a rejected step; a bug from a finite
# state propagates at once instead of being retried at half the step.
@testset "exceptions: bugs are rethrown, numerical failures retried (#327 M6)" begin
    function exc_case(child_throw)
        pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
        nest = nested_grid(pm, ((0.0, 1.0), (0.0, 1.0), (0.25, 0.75)), 2; halo=(3, 3, 3))
        cm = nest_child_mesh(nest)
        psim = BoussinesqSimulation(pm); fill!(psim.u.data, 1.0); fill!(psim.θ.data, 300.0)
        csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm, :periodic, :periodic, :nest))
        fill!(csim.u.data, 1.0); fill!(csim.θ.data, 300.0)
        armed = Ref(true)
        ccore(sim, dt; kw...) = (step!(sim, dt; kw...); armed[] && (armed[] = false; child_throw(sim)); sim)
        return NestedRun(psim, csim, nest; feedback = :one_way, advance_child = domain_advance(ccore),
                         cfl_parent = CFLConstraints(dt_max = 0.02))
    end
    run = exc_case(s -> zeros(2)[3])
    @test_throws BoundsError step_nested!(run)
    @test run.stats.child_retries == 0
    run = exc_case(s -> throw(NumericalFailure("solver diverged")))
    step_nested!(run)
    @test run.stats.child_retries == 1 && run.child.t == run.parent.t
    s = BoussinesqSimulation(RegularGrid(Float64; size=(4, 4, 4), extent=(1.0, 1.0, 1.0), halo=(1, 1, 1)))
    @test !is_numerical_failure(BoundsError(zeros(1), 2), s)
    s.θ.data[2, 2, 2] = Inf
    @test is_numerical_failure(BoundsError(zeros(1), 2), s)   # from a non-finite state: numerical
end

end # testset
