# N-berlin-nest (#265, epic #246): a nest whose child picks its own steps.
# The urban site advances its child through a whole stage loop, so the
# model's nest engine (Components.ModelNest, NEST-p3 #483) does not hand it
# `step_nested!`'s subcycling: it opens a parent-step window (the parent's
# step, the boundary data at both ends), lets the child take steps that land
# on the window end (`subcycle_dt`, `nest_window_end`), books a two-way
# child's steps between `_begin!` and `_commit!`, and closes the window
# (`_end_parent_step!`). These tests run that pattern (`_nx_window!`, the
# engine's internals; the public window API of NestedRun is gone since
# NEST-p3) on synthetic boxes (no Berlin geometry, rule of #321):
#
#   1. the split is exact: the window driven from outside agrees with
#      `step_nested!(run)` (two-way, ledger included);
#   2. mass: net boundary volume flux after the correction and child
#      max|∇·u| at round-off, every child step;
#   3. scalars added after construction (`add_open_scalars!`), including the
#      `:inflow` halo of a tracer the parent does not carry;
#   4. boundary_scalar_flux books the positivity-limited fluxes (#265).
#
# References: Hellsten et al. (2021), GMD 14, 3185, §3.1–3.3 (PALM nesting:
# Dirichlet child boundary, child substeps, mass-flux correction).

using Test
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: FieldBoundaryConditions, OpenBoundary, open_boundary_states,
                                 add_open_scalars!, update_open_boundaries!
using Aeolus.Operators: divergence
using Aeolus.AdvectionSchemes: boundary_scalar_flux
using Aeolus.BoundaryConditions: ProfileBoundaryData, open_boundary_conditions

"Child initialised from the parent prolongated over the whole child."
function _nx_init_child!(csim, psim, nest)
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, psim, nest; order = 3, region = :all)
    for (cf, sf) in ((csim.u, st.u), (csim.v, st.v), (csim.w, st.w), (csim.θ, st.θ))
        cf.data .= sf.data
    end
    return csim
end

"""
Drive the child of `run` through one parent window of length `Δt_p`, with
child steps of at most `h_lim` placed by `subcycle_dt`, and `each(h)`
called after every child step. Returns the number of child steps.
"""
function _nx_window!(run, adv, Δt_p, h_lim; each = h -> nothing)
    NX = Aeolus.Nesting
    w = NX._begin_parent_step!(run; dt = Δt_p)
    @assert nest_window_end(run) == w.t_child_end
    c = run.child
    n = 0
    while w.t_child_end - c.t > 1e-9 * Δt_p
        h = subcycle_dt(w.t_child_end - c.t, h_lim)
        # the child-step bracket: a refluxing two-way run books the step
        run.sub_t0 = Float64(c.t)
        NX._begin!(run, :child)
        hook = run.regs === nothing || !any(m -> NX._by_core(m, :child), run.regs.models) ? nothing :
               NX._hook(run, :child)
        adv(c, h; body_force! = hook)
        NX._commit!(run, :child)
        each(h)
        n += 1
    end
    NX._end_parent_step!(run, w, true)
    return n
end

_nx_open_area(csim) = (o = nest_open_sides(csim); m = csim.mesh;
    (o[1] + o[2]) * m.Ly * m.Lz + (o[3] + o[4]) * m.Lx * m.Lz + (o[5] + o[6]) * m.Lx * m.Ly)

@testset "Era II — externally driven nest child (N-berlin-nest, #265)" begin

# -------------------------------------------------------------------------
@testset "a window driven from outside + subcycle_dt == step_nested!(run)" begin
    let feedback = :two_way
        rng = Xoshiro(265)
        runs = map(1:2) do _
            pm = RegularGrid(Float64; size = (16, 16, 8), extent = (800.0, 800.0, 400.0), halo = (3, 3, 3))
            nest = nested_grid(pm, ((200.0, 500.0), (250.0, 550.0), (0.0, 200.0)), (2, 2, 2); halo = (3, 3, 3))
            cm = nest_child_mesh(nest)
            psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall), ν = 0.5)
            csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm; bottom = :wall, top = :nest), ν = 0.5)
            (psim, csim, nest)
        end
        # the same random, projected parent state in both copies
        u0 = randn(rng, size(interior_view(runs[1][1].u)))
        θ0 = 300 .+ 0.1 .* randn(rng, size(interior_view(runs[1][1].θ)))
        for (psim, csim, nest) in runs
            interior_view(psim.u) .= u0; interior_view(psim.θ) .= θ0
            step!(psim, 1e-10); psim.t = 0.0
            _nx_init_child!(csim, psim, nest)
        end
        adv = domain_advance(step_rk3!)
        mk(r) = NestedRun(r[1], r[2], r[3]; advance_parent = adv, advance_child = adv,
                          cfl_parent = nothing, cfl_child = nothing, substeps = 3, feedback)
        A = mk(runs[1]); B = mk(runs[2])
        ns = map(1:2) do _
            step_nested!(A; dt = 0.6)
            _nx_window!(B, adv, 0.6, 0.6 / 3)
        end
        @test ns == [3, 3]
        same = all(agree(getproperty(A.child, f).data, getproperty(B.child, f).data) for f in (:u, :v, :w, :θ)) &&
               all(agree(getproperty(A.parent, f).data, getproperty(B.parent, f).data) for f in (:u, :v, :w, :θ))
        @test same
        @test values_agree(A.child.t, B.child.t) && values_agree(A.t, B.t)
        @test A.stats.parent_steps == B.stats.parent_steps == 2
        @test values_agree(A.nest.ledger, B.nest.ledger)
    end
    # the substep rule: n = ⌈R/limit⌉, landing exactly on the window end
    @test subcycle_dt(1.0, 0.3) == 0.25
    @test subcycle_dt(0.9, 0.3) ≈ 0.3            # an integer ratio up to round-off: 3 steps, not 4
    @test subcycle_dt(0.2, 0.3) == 0.2
end

# -------------------------------------------------------------------------
@testset "mass: boundary volume flux and child ∇·u at round-off, every step" begin
    rng = Xoshiro(2651)
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    for f in (psim.u, psim.v, psim.w)
        interior_view(f) .= randn(rng, size(interior_view(f)))
    end
    step!(psim, 1e-10); psim.t = 0.0
    nest = nested_grid(pm, ((0.25, 0.625), (0.25, 0.625), (0.0, 0.5)), (2, 2, 4); halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm; bottom = :wall, top = :nest))
    _nx_init_child!(csim, psim, nest)
    adv = domain_advance(step_rk3!)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = adv, advance_child = adv,
                    cfl_parent = nothing, cfl_child = nothing)
    step!(csim, 1e-12); csim.t = psim.t
    U = maximum(abs, interior_view(psim.u)); A = _nx_open_area(csim)
    st = only(open_boundary_states(csim)); st.record = true
    worst_res = 0.0; worst_div = 0.0; nsteps = 0
    each = h -> begin
        for (_, _, res) in st.log
            worst_res = max(worst_res, abs(res))
        end
        empty!(st.log)
        divergence(csim.div_buf, csim.u, csim.v, csim.w)
        worst_div = max(worst_div, maximum(abs, interior_view(csim.div_buf)))
        nsteps += 1
    end
    for _ in 1:2
        _nx_window!(run, adv, 0.01, 0.0041; each)
    end
    @test nsteps >= 6
    @test worst_res <= 1e-12 * U * A
    @test worst_div <= 1e-10
end

# -------------------------------------------------------------------------
@testset "add_open_scalars!: a parent tracer, and the :inflow halo of one it lacks" begin
    U = 1.5
    pm = RegularGrid(Float64; size = (16, 8, 8), extent = (640.0, 320.0, 320.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((160.0, 400.0), (0.0, 320.0), (0.0, 320.0)), (2, 1, 1); halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; θ₀ = 300.0)
    bcs = nest_bcs(cm, :nest, :periodic, :periodic)
    csim = BoussinesqSimulation(cm; bcs = bcs, θ₀ = 300.0)
    for s in (psim, csim)
        fill!(s.u.data, U); fill!(s.θ.data, 300.0)
    end
    # tracers: `a` exists in both domains, `b` only in the child
    ap = Field(CellCenter(), pm); ac = Field(CellCenter(), cm); b0 = Field(CellCenter(), pm)
    bc_ = Field(CellCenter(), cm)
    fill!(ap.data, 2.0); fill!(ac.data, 1.0); fill!(bc_.data, 5.0)
    add_open_scalars!(bcs, :a)
    add_open_scalars!(bcs, :b; mode = :inflow)
    @test_throws ArgumentError add_open_scalars!(bcs, :a)
    st = ntuple(_ -> nest_interp_state(nest; scalars = (:a => (ac, ap), :b => (bc_, b0))), 3)
    adv = domain_advance(step!)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = adv, advance_child = adv,
                    cfl_parent = nothing, cfl_child = nothing, state = st)
    fill_halos!(ac, csim.θ_bcs); fill_halos!(bc_, csim.θ_bcs)
    Hx, Nx = cm.Hx, cm.Nx
    # west face (inflow, u > 0): `a` takes the parent's 2, `b` the parent's 0
    # east face (outflow): `a` takes the parent's value (Dirichlet), `b` the
    # mirror of the interior (its own 5); the values are copied, not computed
    @test all(==(2.0), ac.data[1:Hx, :, :]) && all(==(0.0), bc_.data[1:Hx, :, :])
    @test all(==(2.0), ac.data[Hx+Nx+1:end, :, :]) && all(==(5.0), bc_.data[Hx+Nx+1:end, :, :])
    # reversing the wind swaps the roles of the faces for `b`
    for s in (psim, csim)
        fill!(s.u.data, -U)
    end
    update_open_boundaries!(csim, csim.t, 0.0)            # buffers from the (unchanged) data …
    interpolate_parent_state!(run.s_old, psim, nest; order = 3, region = :halo_interface)
    apply_nest_boundary!(csim, run.s_old)                  # … refreshed from the reversed parent
    fill_halos!(bc_, csim.θ_bcs)
    @test all(==(5.0), bc_.data[1:Hx, :, :]) && all(==(0.0), bc_.data[Hx+Nx+1:end, :, :])
end

# -------------------------------------------------------------------------
# The water ledger of the Berlin case books what crosses the open faces with
# boundary_scalar_flux. Its tracers go through the positivity-limited
# transport (advect_positive!), which scales a boundary flux where the cell
# next to an outflow face would go negative; the booking must take those
# limited fluxes (face_fluxes), not re-reconstruct them. Found when the nest's
# zero-ql inflow met seeded fog in the smoke window: the ledger's `transport`
# term showed 2.4e-8 kg/m² of spurious water (#265).
@testset "boundary_scalar_flux books the positivity-limited face fluxes" begin
    m = RegularGrid(Float64; size = (16, 8, 8), extent = (160.0, 80.0, 80.0), halo = (3, 3, 3))
    data = ProfileBoundaryData(u = 4.0, v = 0.0, w = 0.0, θ = 300.0)
    bcs = open_boundary_conditions(m, data; halo = 3)
    sim = BoussinesqSimulation(m; bcs = bcs, θ₀ = 300.0)
    fill!(sim.u.data, 4.0); fill!(sim.θ.data, 300.0)
    Aeolus.BoundaryConditions.prepare_open_boundaries!(sim)
    q = Field(CellCenter(), m)
    qi = interior_view(q)
    # tracer only in the last cell before the outflow (east) face, its
    # mirror halo next to it: the WENO face value there is 1.2–1.8 times the
    # cell's content, so at CFL 0.9 the flux would empty the cell below zero
    # and the limiter scales it down
    for k in 1:m.Nz, j in 1:m.Ny
        qi[m.Nx, j, k] = 1e-3
    end
    fill_halos!(q, FieldBoundaryConditions(:noflux))           # zero-gradient halos
    ws = Aeolus.AdvectionSchemes.FluxLimiterWorkspace(m)
    tend = Field(CellCenter(), m)
    dt = 0.9 * m.dx / 4.0
    Aeolus.AdvectionSchemes.advect_positive!(tend, sim.u, sim.v, sim.w, q, dt;
                                             scheme = :weno_z5, workspace = ws,
                                             bcs = FieldBoundaryConditions(:noflux))
    V = m.dx * m.dy * m.dz
    change = dt * sum(interior_view(tend)) * V
    lim = boundary_scalar_flux(sim, q; scheme = :weno_z5, bcs = FieldBoundaryConditions(:noflux),
                               fill = false, face_fluxes = (ws.Fx, ws.Fy, ws.Fz))
    rec = boundary_scalar_flux(sim, q; scheme = :weno_z5, bcs = FieldBoundaryConditions(:noflux),
                               fill = false)
    out = dt * lim.total
    @test abs(change + out) <= 1e-12 * abs(change)
    @test abs(change + dt * rec.total) > 1e-6 * abs(change)    # the reconstruction is not what left
    @test minimum(qi .+ dt .* interior_view(tend)) >= 0
end


end
