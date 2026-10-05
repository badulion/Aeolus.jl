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
_xc(m, i) = m.x0 + (i - m.Hx - 0.5) * m.dx
_yc(m, j) = m.y0 + (j - m.Hy - 0.5) * m.dy
_zc(m, k) = m.z0 + (k - m.Hz - 0.5) * m.dz
_xf(m, i) = m.x0 + (i - m.Hx - 1) * m.dx

"Set every storage cell of the four prognostics from `f(name, x, y, z)`."
function _set_fields!(sim, f)
    m = sim.mesh
    for k in axes(sim.θ.data, 3), j in axes(sim.θ.data, 2), i in axes(sim.θ.data, 1)
        x = _xc(m, i); y = _yc(m, j); z = _zc(m, k)
        sim.θ.data[i, j, k] = f(:θ, x, y, z)
    end
    for k in axes(sim.u.data, 3), j in axes(sim.u.data, 2), i in axes(sim.u.data, 1)
        sim.u.data[i, j, k] = f(:u, _xf(m, i), _yc(m, j), _zc(m, k))
    end
    fill!(sim.v.data, 0.0); fill!(sim.w.data, 0.0)
    return sim
end

"Copy the parent prolongated over the whole child (a smooth initial child)."
function _init_child_from_parent!(csim, psim, nest)
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, psim, nest; order=3, region=:all)
    for (cf, sf) in ((csim.u, st.u), (csim.v, st.v), (csim.w, st.w), (csim.θ, st.θ))
        cf.data .= sf.data
    end
    return csim
end

"A user core that sets the parent analytically: fields `f(name, x, y, z, t)`."
analytic_core(f) = (sim, dt; kw...) -> begin
    sim.t += dt
    tt = sim.t
    _set_fields!(sim, (n, x, y, z) -> f(n, x, y, z, tt))
    sim
end

_mean_u(sim) = sum(interior_view(sim.u)) / length(interior_view(sim.u))

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

end # testset
