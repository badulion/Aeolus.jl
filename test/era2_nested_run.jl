using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view
using Aeolus.PressureSolvers: FFTPoissonSolver

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

_nr_mean_u(sim) = sum(interior_view(sim.u)) / length(interior_view(sim.u))

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
@testset "Era II — NestedRun time coupling (N-nest-c, #257)" begin

# -------------------------------------------------------------------------
@testset "adaptive_dt and StateSnapshot extend to the sound-proof cores" begin
    m = RegularGrid(Float64; size=(16, 8, 8), extent=(1.0, 0.5, 0.5), halo=(3, 3, 3))
    sim = AnelasticSimulation(m; buoyancy=false)
    U = 2.0
    fill!(sim.u.data, U)
    c = CFLConstraints(C_advective=0.5)
    dt = adaptive_dt(sim, c)
    @printf("  anelastic adaptive_dt, uniform U=2, dx=1/16: dt = %.15g (C dx/U = %.15g)\n",
            dt, 0.5 * m.dx / U)
    @test dt ≈ 0.5 * m.dx / U rtol = 1e-14
    # molecular diffusive limit binds once ν is large
    simν = AnelasticSimulation(m; buoyancy=false, ν=1.0)
    Δ²h = 1 / (1 / m.dx^2 + 1 / m.dy^2 + 1 / m.dz^2)
    @test adaptive_dt(simν, c) ≈ 0.25 * Δ²h / 1.0 rtol = 1e-14
    # growth limiter and clamps, as for the Boussinesq core
    @test adaptive_dt(sim, c; dt_prev = 1e-3) ≈ 1.1e-3 rtol = 1e-14
    @test adaptive_dt(sim, CFLConstraints(dt_max = 1e-4)) == 1e-4
    sim.u.data[5, 5, 5] = NaN
    @test adaptive_dt(sim, c) == c.dt_min
    # snapshot round trip is bit-identical, clock included
    fill!(sim.u.data, U); sim.θ.data .+= randn(size(sim.θ.data)); sim.t = 3.5
    snap = state_snapshot(sim)
    @test snap === state_snapshot(sim)                   # cached per simulation
    save_state!(snap, sim)
    θ0 = copy(sim.θ.data); u0 = copy(sim.u.data)
    step!(sim, 1e-3)
    @test sim.θ.data != θ0
    restore_state!(sim, snap)
    @test sim.θ.data == θ0 && sim.u.data == u0 && sim.t == 3.5
end

# -------------------------------------------------------------------------
@testset "domain_advance: hooks, dispatch and stage times" begin
    @test stage_times(domain_advance()) == (0.0,)
    @test stage_times(domain_advance(step_rk3!)) == (0.0, 1.0, 0.5)
    @test stage_times(domain_advance(step_imex!, ARK2())) == (0.0, ARK2().γ)
    @test stage_times(domain_advance(step_imex!, Rosenbrock2())) == (0.0, Rosenbrock2().a21)
    # the closure is exactly the core with its keywords: bit-identical to a
    # direct call, and the run's hook is chained after the domain's own
    m = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    mk() = (s = BoussinesqSimulation(m; ν=1e-3);
            Random.seed!(3); s.θ.data .= randn(size(s.θ.data)); fill!(s.u.data, 1.0); s)
    drag!(sim, Δt) = (sim.tu.data .-= 0.3 .* sim.u.data; sim)
    a = mk(); b = mk()
    order = Symbol[]
    adv = domain_advance(step_rk3!; body_force! = drag!,
                         post_predictor! = (s, h) -> push!(order, :user))
    adv(a, 1e-2; post_predictor! = (s, h) -> push!(order, :run))
    step_rk3!(b, 1e-2; body_force! = drag!, post_predictor! = (s, h) -> nothing)
    @test a.u.data == b.u.data && a.θ.data == b.θ.data && a.t == b.t
    @test order == [:user, :run, :user, :run, :user, :run]
end

# -------------------------------------------------------------------------
# Parent 16³ on the unit cube, child (0.25, 0.75)³ at r = 2. The parent is
# advanced by a user core that sets it analytically, linear in x, y, z, t.
function _nr_linear_case(child_adv; substeps=3, dt=1e-3, nsteps=4)
    pm = RegularGrid(Float64; size=(16, 16, 16), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    # u is divergence-free (no x dependence): the child's open faces make
    # the boundary flux compatible with the projection (N-nest-d), so data
    # with ∂u/∂x ≠ 0 would be (correctly) changed by the mass-flux correction
    f(n, x, y, z, t) = n === :θ ? 300.0 + 2x - 3y + z + 7.0 * t :
                                  1.0 + 0.25y - 0.125z - 4.0 * t
    psim = BoussinesqSimulation(pm)
    _nr_set_fields!(psim, (n, x, y, z) -> f(n, x, y, z, 0.0))
    csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm))
    _nr_init_child_from_parent!(csim, psim, nest)
    # stage halos, recorded where the core reads them (body_force! runs after
    # the tendencies, before anything writes the halos again)
    RUN = Ref{Any}(nothing)
    err = Ref(0.0); nrec = Ref(0); ctr = [NaN, 0.0]
    ii, jj, kk = interior(cm)
    function record!(sim, Δt)
        run = RUN[]
        ts = _nr_stage_time!(ctr, run)
        for k in axes(sim.θ.data, 3), j in axes(sim.θ.data, 2), i in axes(sim.θ.data, 1)
            (i in ii) && (j in jj) && (k in kk) && continue
            ex = f(:θ, _nr_xc(cm, i), _nr_yc(cm, j), _nr_zc(cm, k), ts)
            err[] = max(err[], abs(sim.θ.data[i, j, k] - ex) / abs(ex))
            exu = f(:u, _nr_xf(cm, i), _nr_yc(cm, j), _nr_zc(cm, k), ts)
            err[] = max(err[], abs(sim.u.data[i, j, k] - exu) / abs(exu))
        end
        nrec[] += 1
        return sim
    end
    adv = domain_advance(child_adv.core, child_adv.args...;
                         child_adv.kwargs..., body_force! = record!)
    run = NestedRun(psim, csim, nest; advance_parent = domain_advance(_nr_analytic_core(f)),
                    advance_child = adv, cfl_parent = nothing, cfl_child = nothing,
                    substeps)
    RUN[] = run
    for _ in 1:nsteps
        step_nested!(run; dt)
    end
    # boundary_time over the last window, all stored fields (halo cells)
    eb = 0.0
    for τ in (0.0, 0.3, 0.5, 1.0)
        t = run.t0 + τ * run.Δt
        s = boundary_time(run, t)
        for k in axes(s.θ.data, 3), j in axes(s.θ.data, 2), i in axes(s.θ.data, 1)
            (i in ii) && (j in jj) && (k in kk) && continue
            ex = f(:θ, _nr_xc(cm, i), _nr_yc(cm, j), _nr_zc(cm, k), t)
            eb = max(eb, abs(s.θ.data[i, j, k] - ex) / abs(ex))
        end
    end
    return (; err = err[], nrec = nrec[], eb, run)
end

@testset "exact temporal interpolation at every substep and stage" begin
    for (name, adv, nst) in (("step!", domain_advance(), 1),
                             ("step_rk3!", domain_advance(step_rk3!), 3),
                             ("step_imex! ARK2", domain_advance(step_imex!, ARK2()), 2),
                             ("step_imex! Rosenbrock2", domain_advance(step_imex!, Rosenbrock2()), 2))
        r = _nr_linear_case(adv)
        @printf("  %-24s %2d stage evaluations: max rel. halo error %.2e, boundary_time %.2e\n",
                name, r.nrec, r.err, r.eb)
        @test r.nrec == 4 * 3 * nst           # 4 parent steps × 3 substeps × stages
        @test r.err < 1e-14
        @test r.eb < 1e-14
        @test r.run.child.t == r.run.parent.t
    end
end

# -------------------------------------------------------------------------
@testset "temporal interpolation is second order (sin ωt)" begin
    ω = 2π; A = 1.5
    function boundary_error(dtp)
        pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
        nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo=(3, 3, 3))
        cm = nest_child_mesh(nest)
        f(n, x, y, z, t) = n === :θ ? 300.0 + A * sin(ω * t) : 0.0
        psim = BoussinesqSimulation(pm); _nr_set_fields!(psim, (n, x, y, z) -> f(n, x, y, z, 0.0))
        csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm))
        _nr_init_child_from_parent!(csim, psim, nest)
        RUN = Ref{Any}(nothing); err = Ref(0.0); ctr = [NaN, 0.0]
        function rec!(sim, Δt)
            run = RUN[]
            ts = _nr_stage_time!(ctr, run)
            err[] = max(err[], abs(sim.θ.data[1, 1, 1] - (300.0 + A * sin(ω * ts))))
            sim
        end
        run = NestedRun(psim, csim, nest; advance_parent = domain_advance(_nr_analytic_core(f)),
                        advance_child = domain_advance(step_rk3!; body_force! = rec!),
                        cfl_parent = nothing, cfl_child = nothing, substeps = 2)
        RUN[] = run
        for _ in 1:round(Int, 1 / dtp)
            step_nested!(run; dt = dtp)
        end
        return err[]
    end
    e1 = boundary_error(1 / 20); e2 = boundary_error(1 / 40); e3 = boundary_error(1 / 80)
    @printf("  boundary-data error: Δt_p = 1/20 → %.3e, 1/40 → %.3e (×%.3f), 1/80 → %.3e (×%.3f)\n",
            e1, e2, e1 / e2, e3, e2 / e3)
    # leading error of linear interpolation: Δt²/8 · max|f''| = Δt² A ω² / 8
    @printf("  bound Δt_p² A ω²/8 at 1/20: %.3e\n", (1 / 20)^2 * A * ω^2 / 8)
    @test e1 / e2 >= 3.7
    @test e2 / e3 >= 3.7
    @test e1 <= (1 / 20)^2 * A * ω^2 / 8 * (1 + 1e-9)
end

# -------------------------------------------------------------------------
# Substep count: uniform flow along x, so the child's advective limit is
# exactly the parent's divided by r.
@testset "substep count: child limit Δt_p/r gives n = r" begin
    for r in (2, 3, 4, 5)
        pm = RegularGrid(Float64; size=(8, 4, 4), extent=(1.0, 0.5, 0.5), halo=(3, 3, 3))
        nest = nested_grid(pm, ((0.25, 0.5), (0.125, 0.375), (0.125, 0.375)), r; halo=(3, 3, 3))
        cm = nest_child_mesh(nest)
        psim = BoussinesqSimulation(pm); fill!(psim.u.data, 1.3); fill!(psim.θ.data, 300.0)
        csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm))
        fill!(csim.u.data, 1.3); fill!(csim.θ.data, 300.0)
        run = NestedRun(psim, csim, nest; cfl_parent = CFLConstraints(C_advective = 0.5))
        for _ in 1:3
            step_nested!(run)
        end
        st = run.stats
        @printf("  r = %d: Δt_p = %.6g, n = %d..%d, Δt_c ∈ [%.6g, %.6g], max child CFL %.15f\n",
                r, st.dt_parent_max, st.n_min, st.n_max, st.dt_child_min, st.dt_child_max,
                st.max_child_cfl)
        @test st.n_min == r && st.n_max == r
        @test st.dt_child_max ≈ st.dt_parent_max / r rtol = 1e-12
        @test st.max_child_cfl <= 0.5 * (1 + 1e-9)
    end
end

# -------------------------------------------------------------------------
# Uniform flow along x, w = 0, no viscosity: the child spans the parent
# horizontally (periodic) and is nested only in z, where w = 0 means the
# halos cannot act on u. A drag hook then has an exact answer.
function _nr_drag_case(; drag_on::Symbol, child_core::Symbol = :boussinesq, c = 1.0,
                    T = 1.0, U0 = 1.0)
    pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.0, 1.0), (0.0, 1.0), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm); fill!(psim.u.data, U0); fill!(psim.θ.data, 300.0)
    csim = child_core === :boussinesq ?
        BoussinesqSimulation(cm; bcs=nest_bcs(cm, :periodic, :periodic, :nest)) :
        AnelasticSimulation(cm; buoyancy=false, bcs=nest_bcs(cm, :periodic, :periodic, :nest))
    fill!(csim.u.data, U0)
    child_core === :boussinesq && fill!(csim.θ.data, 300.0)
    # RK3 with the drag as a tendency (Boussinesq), or the drag integrated
    # exactly, u ← u e^{-cΔt}, as a post-predictor correction (sound-proof
    # core: its forward-Euler step would carry an O(cΔt) error)
    drag_rk3 = domain_advance(step_rk3!; body_force! = (s, Δt) -> (s.tu.data .-= c .* s.u.data; s))
    drag_pp  = domain_advance(; post_predictor! = (s, Δt) -> (interior_view(s.u) .*= exp(-c * Δt); s))
    plain_rk3 = domain_advance(step_rk3!)
    ap = drag_on === :parent ? drag_rk3 : plain_rk3
    ac = drag_on === :child ? (child_core === :boussinesq ? drag_rk3 : drag_pp) :
         (child_core === :boussinesq ? plain_rk3 : domain_advance())
    cfl = CFLConstraints(dt_max = 0.01)
    run = NestedRun(psim, csim, nest; advance_parent = ap, advance_child = ac,
                    cfl_parent = cfl)
    while run.t < T - 1e-12
        step_nested!(run; dt = min(0.01, T - run.t))
    end
    return run
end

@testset "hooks reach both domains (linear drag), other domain unchanged" begin
    for (drag_on, core) in ((:child, :boussinesq), (:parent, :boussinesq),
                            (:child, :anelastic))
        run = _nr_drag_case(; drag_on, child_core = core)
        p, c = run.parent, run.child
        dragged, other = drag_on === :child ? (c, p) : (p, c)
        e = abs(_nr_mean_u(dragged) / exp(-run.t) - 1.0)
        o = abs(_nr_mean_u(other) - 1.0)
        @printf("  drag on %-6s (%s child): t = %.3f, |ū/e^{-ct} − 1| = %.2e, other domain |ū − 1| = %.2e\n",
                drag_on, core, run.t, e, o)
        @test run.t ≈ 1.0 rtol = 1e-14
        @test e < 1e-6
        @test o < 1e-14
    end
end

# -------------------------------------------------------------------------
"""
Parent and child with a Taylor-Green flow plus a mean wind and a θ blob.
The child's x and y faces are open nest faces (N-nest-d): the parent's
normal velocity on the boundary faces, Dirichlet tangential velocity and
θ, and an all-Neumann projection that sees that flux. Before N-nest-d the
child kept a periodic FFT projection and needed the normal velocity
imposed on its interface planes *and* a 3-cell sponge to stay stable; it
now runs without a sponge (the N-nest-c default).
"""
function _nr_tg_case(; Np = 16, r = 2, U = 0.5, seed = 1, ν = 5e-3,
                  sponge = nothing, kwargs...)
    L = 1.0
    pm = RegularGrid(Float64; size=(Np, Np, 8), extent=(L, L, L / 2), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.0, L / 2)), (r, r, 1); halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; ν = ν)
    Random.seed!(seed)
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
    return NestedRun(psim, csim, nest; sponge, kwargs...)
end

"KE of the parent over the cells the child covers (u, v, w averaged to centres)."
function _nr_covered_ke(p, nest)
    m = p.mesh; cx, cy, cz = nest_covered_range(nest)
    s = 0.0
    for k in cz, j in cy, i in cx
        I = m.Hx + i; J = m.Hy + j; K = m.Hz + k
        uc = 0.5 * (p.u.data[I, J, K] + p.u.data[I+1, J, K])
        vc = 0.5 * (p.v.data[I, J, K] + p.v.data[I, J+1, K])
        wc = 0.5 * (p.w.data[I, J, K] + p.w.data[I, J, K+1])
        s += uc^2 + vc^2 + wc^2
    end
    return 0.5 * s * m.dx * m.dy * m.dz
end

@testset "one-way: the parent is bit-identical to a standalone run" begin
    run = _nr_tg_case(; feedback = :one_way, cfl_parent = CFLConstraints())
    alone = deepcopy(run.parent)
    c = CFLConstraints(); dtp = nothing
    for _ in 1:20
        step_nested!(run)
        dt = adaptive_dt(alone, c; dt_prev = dtp)
        step!(alone, dt); dtp = dt
    end
    same = interior_view(run.parent.u) == interior_view(alone.u) &&
           interior_view(run.parent.v) == interior_view(alone.v) &&
           interior_view(run.parent.w) == interior_view(alone.w) &&
           interior_view(run.parent.θ) == interior_view(alone.θ)
    @printf("  20 one-way steps: parent bit-identical = %s, t = %.6g vs %.6g, child substeps %d\n",
            same, run.parent.t, alone.t, run.stats.child_substeps)
    @test same
    @test run.parent.t == alone.t
end

# -------------------------------------------------------------------------
@testset "clock alignment and child CFL over 1000 adaptive parent steps" begin
    for fb in (:one_way, :two_way)
        run = _nr_tg_case(; feedback = fb, cfl_parent = CFLConstraints(C_advective = 0.5))
        kep0 = kinetic_energy(run.parent)
        maxdrift = 0.0; kepmax = kep0; mass = 0.0
        rmin = Inf; rmax = 0.0
        for n in 1:1000
            step_nested!(run)
            maxdrift = max(maxdrift, abs(run.child.t - run.parent.t) / run.parent.t)
            kepmax = max(kepmax, kinetic_energy(run.parent))
            r = kinetic_energy(run.child) / _nr_covered_ke(run.parent, run.nest)
            rmin = min(rmin, r); rmax = max(rmax, r)
            fb === :two_way && (mass = max(mass, nest_conservation_error(run.parent, run.child,
                                                                        run.nest, :mass)))
        end
        st = run.stats
        @printf("  %-7s 1000 steps to t = %.3f: %d child substeps (n %d..%d), Δt_p ∈ [%.4g, %.4g], Δt_c ∈ [%.4g, %.4g]\n",
                fb, run.t, st.child_substeps, st.n_min, st.n_max, st.dt_parent_min,
                st.dt_parent_max, st.dt_child_min, st.dt_child_max)
        @printf("          clock error before snap %.2e, after %.2e; max child CFL %.6f (C = 0.5)\n",
                st.max_clock_error, maxdrift, st.max_child_cfl)
        @printf("          KE_child / KE_parent(covered) ∈ [%.4f, %.4f]; max KE_parent / KE_parent(0) %.4f%s\n",
                rmin, rmax, kepmax / kep0,
                fb === :two_way ? @sprintf("; interface mass imbalance ≤ %.2e", mass) : "")
        @printf("          wall: parent %.2f s, child %.2f s, boundary %.2f s, feedback %.2f s\n",
                st.wall_parent, st.wall_child, st.wall_boundary, st.wall_feedback)
        @test st.parent_steps == 1000
        @test st.max_clock_error <= 1e-12
        @test maxdrift == 0.0
        @test st.max_child_cfl <= 0.5 * (1 + 1e-9)
        @test st.n_min >= 2                         # the child never outruns its CFL
        @test st.child_retries == 0 && st.parent_retries == 0
        # stable: the child carries the parent's energy (±15 %) and the
        # viscous parent never gains energy
        @test 0.85 <= rmin && rmax <= 1.15
        @test kepmax <= kep0 * (1 + 1e-12)
        fb === :two_way && @test mass < 1e-10
    end
end

# -------------------------------------------------------------------------
@testset "sync = :shared_min_dt: one dt for both domains (PALM)" begin
    run = _nr_tg_case(; cfl_parent = CFLConstraints(), sync = :shared_min_dt)
    for _ in 1:10
        step_nested!(run)
    end
    st = run.stats
    @printf("  shared min dt: n = %d..%d, Δt_p ∈ [%.5g, %.5g], Δt_c ∈ [%.5g, %.5g]\n",
            st.n_min, st.n_max, st.dt_parent_min, st.dt_parent_max, st.dt_child_min,
            st.dt_child_max)
    @test st.n_min == 1 && st.n_max == 1
    @test st.dt_parent_min ≈ st.dt_child_min rtol = 1e-12
    @test st.dt_parent_max ≈ st.dt_child_max rtol = 1e-12
    @test st.max_child_cfl <= 0.5 * (1 + 1e-9)
    # the shared step is the child's (finer) limit, below the parent's own
    sub = _nr_tg_case(; cfl_parent = CFLConstraints())
    step_nested!(sub)
    @test st.dt_parent_max < sub.stats.dt_parent_max
end

# -------------------------------------------------------------------------
# Retry isolation. The child carries uniform flow decaying under an exactly
# integrated drag, so its state is independent of the dt sequence; the
# parent is a Taylor-Green flow.
@testset "retry isolation: a child NaN rolls back only the child" begin
    function retry_case(inject::Bool; parent_nan::Bool = false)
        pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
        nest = nested_grid(pm, ((0.0, 1.0), (0.0, 1.0), (0.25, 0.75)), 2; halo=(3, 3, 3))
        cm = nest_child_mesh(nest)
        psim = BoussinesqSimulation(pm)
        Random.seed!(7)
        for k in axes(psim.u.data, 3), j in axes(psim.u.data, 2), i in axes(psim.u.data, 1)
            psim.u.data[i, j, k] = 1.0 + 0.3 * sin(2π * _nr_xf(pm, i)) * cos(2π * _nr_yc(pm, j))
            psim.v.data[i, j, k] = -0.3 * cos(2π * _nr_xc(pm, i)) *
                                   sin(2π * (pm.y0 + (j - pm.Hy - 1) * pm.dy))
        end
        psim.θ.data .= 300.0 .+ randn(size(psim.θ.data))
        csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm, :periodic, :periodic, :nest))
        fill!(csim.u.data, 1.0); fill!(csim.θ.data, 300.0)
        ncalls = Ref(0); ncore_child = Ref(0); armed = Ref(inject); parmed = Ref(parent_nan)
        counting(sim, dt; kw...) = (ncalls[] += 1; step!(sim, dt; kw...);
                                    parmed[] && ncalls[] == 3 && (sim.u.data[5, 5, 5] = NaN;
                                                                  parmed[] = false); sim)
        cdrag(sim, dt; kw...) = (ncore_child[] += 1; step!(sim, dt; kw...))
        pp(s, Δt) = begin
            interior_view(s.u) .*= exp(-0.5 * Δt)
            if armed[] && s.t > 0.05
                s.u.data[6, 6, 6] = NaN; armed[] = false
            end
            s
        end
        run = NestedRun(psim, csim, nest; advance_parent = domain_advance(counting),
                        advance_child = domain_advance(cdrag; post_predictor! = pp),
                        cfl_parent = CFLConstraints(dt_max = 0.02))
        for _ in 1:10
            step_nested!(run)
        end
        return run, ncalls[], ncore_child[]
    end
    ref, ncp_ref, ncc_ref = retry_case(false)
    inj, ncp, ncc = retry_case(true)
    dc = maximum(abs, interior_view(inj.child.u) .- interior_view(ref.child.u))
    @printf("  child NaN: retries %d (parent %d); parent core calls %d (ref %d), child core calls %d (ref %d)\n",
            inj.stats.child_retries, inj.stats.parent_retries, ncp, ncp_ref, ncc, ncc_ref)
    @printf("  max |u_child − u_child,ref| = %.2e, parent bit-identical = %s\n", dc,
            interior_view(inj.parent.u) == interior_view(ref.parent.u))
    @test inj.stats.child_retries == 1 && inj.stats.parent_retries == 0
    @test ncp == 10 && ncp_ref == 10                   # the parent is never advanced twice
    # the rejected piece is redone at half size; the growth limiter
    # (growth_max = 1.1) then ramps the child dt back up over a few pieces
    @test ncc >= ncc_ref + 2
    @test interior_view(inj.parent.u) == interior_view(ref.parent.u)
    @test interior_view(inj.parent.θ) == interior_view(ref.parent.θ)
    @test dc < 1e-12
    @test inj.child.t == inj.parent.t
    # a parent rejection is retried before any child substep
    pr, ncp2, ncc2 = retry_case(false; parent_nan = true)
    @printf("  parent NaN: parent retries %d, child retries %d, child core calls %d = substeps %d\n",
            pr.stats.parent_retries, pr.stats.child_retries, ncc2, pr.stats.child_substeps)
    @test pr.stats.parent_retries == 1 && pr.stats.child_retries == 0
    @test ncp2 == 11 && ncc2 == pr.stats.child_substeps
    @test all(isfinite, interior_view(pr.parent.u))
    @test pr.child.t == pr.parent.t
end

# -------------------------------------------------------------------------
# #327 M6: only a numerical failure is a rejected step. A bug (MethodError,
# BoundsError, ArgumentError from a finite state) propagates at once instead
# of being retried at half the step.
@testset "exceptions: bugs are rethrown, numerical failures retried (#327 M6)" begin
    function exc_case(child_throw, parent_throw = nothing)
        pm = RegularGrid(Float64; size=(8, 8, 8), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
        nest = nested_grid(pm, ((0.0, 1.0), (0.0, 1.0), (0.25, 0.75)), 2; halo=(3, 3, 3))
        cm = nest_child_mesh(nest)
        psim = BoussinesqSimulation(pm)
        fill!(psim.u.data, 1.0); fill!(psim.θ.data, 300.0)
        csim = BoussinesqSimulation(cm; bcs=nest_bcs(cm, :periodic, :periodic, :nest))
        fill!(csim.u.data, 1.0); fill!(csim.θ.data, 300.0)
        armed_c = Ref(child_throw !== nothing); armed_p = Ref(parent_throw !== nothing)
        ccalls = Ref(0); pcalls = Ref(0)
        ccore(sim, dt; kw...) = (ccalls[] += 1; step!(sim, dt; kw...);
                                 armed_c[] && (armed_c[] = false; child_throw(sim)); sim)
        pcore(sim, dt; kw...) = (pcalls[] += 1; step!(sim, dt; kw...);
                                 armed_p[] && (armed_p[] = false; parent_throw(sim)); sim)
        run = NestedRun(psim, csim, nest; advance_parent = domain_advance(pcore),
                        advance_child = domain_advance(ccore),
                        cfl_parent = CFLConstraints(dt_max = 0.02))
        return run, ccalls, pcalls
    end
    # bugs from a finite state: rethrown, no retry, nothing rolled back silently
    for (name, thr, E) in (("BoundsError", s -> zeros(2)[3], BoundsError),
                           ("MethodError", s -> sqrt("x"), MethodError),
                           ("ArgumentError", s -> throw(ArgumentError("bad option")), ArgumentError),
                           ("ErrorException", s -> error("a bug"), ErrorException))
        run, cc, pc = exc_case(thr)
        @test_throws E step_nested!(run)
        @test run.stats.child_retries == 0 && cc[] == 1
        run, cc, pc = exc_case(nothing, thr)
        @test_throws E step_nested!(run)
        @test run.stats.parent_retries == 0 && pc[] == 1 && cc[] == 0
        @printf("  %s in a child / parent core: rethrown after 1 call, 0 retries\n", name)
    end
    # numerical failures: rolled back and retried at half the step
    for (name, thr) in (("NumericalFailure", s -> throw(NumericalFailure("solver diverged"))),
                        ("DomainError", s -> sqrt(-1.0)),
                        ("InexactError", s -> ceil(Int, Inf)),
                        ("error from a non-finite state", s -> (s.u.data[6, 6, 6] = NaN;
                                                                error("NaN downstream"))))
        run, cc, pc = exc_case(thr)
        step_nested!(run)
        @test run.stats.child_retries == 1
        @test all(isfinite, interior_view(run.child.u)) && run.child.t == run.parent.t
        run, cc, pc = exc_case(nothing, thr)
        step_nested!(run)
        @test run.stats.parent_retries == 1 && run.stats.child_retries == 0
        @test all(isfinite, interior_view(run.parent.u))
        @printf("  %s in a child / parent core: rolled back, 1 retry each\n", name)
    end
    # the predicate itself
    g = RegularGrid(Float64; size=(4, 4, 4), extent=(1.0, 1.0, 1.0), halo=(1, 1, 1))
    s = BoussinesqSimulation(g)
    @test !is_numerical_failure(BoundsError(zeros(1), 2), s)
    @test !is_numerical_failure(InterruptException(), s)
    @test is_numerical_failure(NumericalFailure("x"), s)
    s.θ.data[2, 2, 2] = Inf
    @test is_numerical_failure(BoundsError(zeros(1), 2), s)
end

# -------------------------------------------------------------------------
# #329 item 4: without a sponge a NestedRun reads the prolongated parent only
# at the child's open-boundary points, so its three field sets keep only
# their halo slabs (and the boundary-normal face plane). Bit-identical
# results to child-sized sets, a fraction of the memory.
@testset "halo-only field sets of a NestedRun (#329)" begin
    pm = RegularGrid(Float64; size=(32, 32, 32), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    function mk(fb; state = nothing)
        rng = Xoshiro(329)
        p = BoussinesqSimulation(pm; ν = 1e-3)
        for f in (p.u, p.v, p.w)
            interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f)))
        end
        interior_view(p.θ) .= 300 .+ randn(rng, size(interior_view(p.θ)))
        step!(p, 1e-10); p.t = 0.0
        c = BoussinesqSimulation(cm; ν = 1e-3, bcs = nest_bcs(cm))
        _nr_init_child_from_parent!(c, p, nest)
        st = state === nothing ? nothing :
             ntuple(_ -> nest_interp_state(nest, p, c; storage = state), 3)
        return NestedRun(p, c, nest; cfl_parent = nothing, cfl_child = nothing,
                         feedback = fb, state = st)
    end
    for fb in (:one_way, :two_way)
        a = mk(fb); b = mk(fb; state = :full)
        @test nest_storage(a.s_old) === nest_storage(a.s_new) === nest_storage(a.s_bc) === :shell
        @test nest_storage(b.s_bc) === :full
        for _ in 1:4
            step_nested!(a; dt = 2e-3); step_nested!(b; dt = 2e-3)
        end
        for f in (:u, :v, :w, :θ)
            @test getproperty(a.child, f).data == getproperty(b.child, f).data
            @test getproperty(a.parent, f).data == getproperty(b.parent, f).data
        end
        # boundary_time over every stored (halo) point is the same blend
        sa = boundary_time(a, a.t0 + 0.4a.Δt); sb = boundary_time(b, b.t0 + 0.4b.Δt)
        ii, jj, kk = interior(cm)
        same = true
        for k in axes(sb.θ.data, 3), j in axes(sb.θ.data, 2), i in axes(sb.θ.data, 1)
            (i in ii) && (j in jj) && (k in kk) && continue
            same &= sa.θ[i, j, k] == sb.θ[i, j, k] && sa.u[i, j, k] == sb.u[i, j, k]
        end
        @test same
        if fb === :one_way
            bytes(r) = sum(st -> sum(Aeolus.Nesting._data_bytes(getproperty(st, f).data)
                                     for f in (:u, :v, :w, :θ)), (r.s_old, r.s_new, r.s_bc))
            n = (cm.Nx + 2cm.Hx) * (cm.Ny + 2cm.Hy) * (cm.Nz + 2cm.Hz)
            nin = cm.Nx * cm.Ny * cm.Nz
            plane = cm.Ny * cm.Nz                           # one face plane per velocity
            @test bytes(b) == 3 * 4 * 8n
            @test bytes(a) == 3 * 8 * (4 * (n - nin) + 3plane)
            @printf("  child %d³, halo %d: the run's three field sets hold %.0f KiB (child-sized: %.0f KiB, %.0f %%)\n",
                    cm.Nx, cm.Hx, bytes(a) / 1024, bytes(b) / 1024, 100 * bytes(a) / bytes(b))
        end
    end
    # a shell has no interior: reading it, prolongating region = :all into
    # it or relaxing toward it is an error
    a = mk(:one_way)
    @test_throws ArgumentError a.s_bc.θ.data[cm.Hx + 2, cm.Hy + 2, cm.Hz + 2]
    @test_throws ArgumentError interpolate_parent_state!(a.s_bc, a.parent, nest; region = :all)
    @test_throws ArgumentError apply_nest_sponge!(a.child, a.s_bc, NestSponge())
    @test_throws ArgumentError NestedRun(a.parent, a.child, nest; sponge = NestSponge(),
                                         state = (a.s_old, a.s_new, a.s_bc),
                                         cfl_parent = nothing, cfl_child = nothing)
end

# -------------------------------------------------------------------------
# #329 items 5 and 6: the nesting layer of a step allocates (almost)
# nothing beyond what the two cores allocate on their own, and its state is
# concretely typed (the KernelAbstractions port, #291, needs both). Before
# #329 a nested step of this set-up allocated 2.9 MiB (one-way) and 6.6 MiB
# (two-way) on top of the cores: fresh rollback copies of the open-boundary
# buffers, two child- and parent-sized fields per composite momentum
# integral, the anterpolation boxes, the mismatch matrices.
@testset "a nested step allocates little beyond the cores; typed state (#329)" begin
    pm = RegularGrid(Float64; size=(32, 32, 32), extent=(1.0, 1.0, 1.0), halo=(3, 3, 3))
    nest = nested_grid(pm, ((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)), 2; halo=(3, 3, 3))
    cm = nest_child_mesh(nest)
    function mk(fb)
        rng = Xoshiro(5)
        p = BoussinesqSimulation(pm)
        interior_view(p.θ) .= 300 .+ 0.1 .* rand(rng, size(interior_view(p.θ)))
        fill!(p.u.data, 0.5)
        c = BoussinesqSimulation(cm; bcs = nest_bcs(cm))
        fill!(c.θ.data, 300.0)
        return NestedRun(p, c, nest; cfl_parent = nothing, cfl_child = nothing, feedback = fb)
    end
    loose(T) = T === Any ||
        (T <: AbstractArray && eltype(T) in (Any, Field)) ||
        (T <: AbstractDict && (keytype(T) === Any || valtype(T) === Any))
    for fb in (:one_way, :two_way)
        run = mk(fb); ref = mk(fb)
        for _ in 1:3; step_nested!(run; dt = 1e-3); end
        a = @allocated step_nested!(run; dt = 1e-3)
        # the two cores alone, on an identical pair, with the same hooks
        p, c = ref.parent, ref.child
        step!(p, 1e-3); step!(c, 5e-4)
        ap = @allocated step!(p, 1e-3)
        ac = @allocated step!(c, 5e-4)
        over = a - ap - run.stats.n_last * ac
        @printf("  %-8s nested step allocates %.1f KiB; the cores alone %.1f KiB; nesting %.1f KiB\n",
                fb, a / 1024, (ap + run.stats.n_last * ac) / 1024, over / 1024)
        @test over < 96 * 1024
        @test (@allocated composite_integral(run.parent, run.child, nest, :momentum_x)) == 0
        @test (@allocated composite_integral(run.parent, run.child, nest, :mass)) == 0
        @test @inferred(composite_integral(run.parent, run.child, nest, :momentum_x)) isa Float64
        # no Any, Vector{Any}, Vector{Field} or Dict{Any,Any} left in the state
        for T in (typeof(run), typeof(run.s_bc), typeof(run.nest), typeof(run.rb_child))
            bad = [n for (n, F) in zip(fieldnames(T), fieldtypes(T)) if loose(F)]
            @test isempty(bad)
        end
        run.regs === nothing || @test isempty([n for (n, F) in zip(fieldnames(typeof(run.regs)),
                                                   fieldtypes(typeof(run.regs))) if loose(F)])
    end
    @test @inferred(Aeolus.Nesting._plan(nest, :avg, 1, 3)) isa Aeolus.Nesting._Plan1D
    @test @inferred(nest_measure(nest)) isa NestMeasure
end

end # testset
