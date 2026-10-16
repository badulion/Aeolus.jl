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
#   1. the split is exact: the window driven from outside equals
#      `step_nested!(run)` bit for bit (one- and two-way);
#   2. a parent at rest with a hydrostatic θ(z) leaves a calm child at rest
#      with the parent's profile (adaptive child steps that do not divide
#      the parent step);
#   3. Galilean invariance: a parent at (U, V) leaves the child at (U, V);
#   4. mass: net boundary volume flux after the correction and child
#      max|∇·u| at round-off, every child step;
#   5. warm-bubble export: the time-integrated boundary heat flux of an
#      open child equals the heat that left it; a periodic child keeps it;
#   6. hooks in both domains: the parent's geostrophic tendency is
#      (−f V_g, +f U_g) everywhere, the child's IBM keeps u = 0 in solids;
#   7. scalars added after construction (`add_open_scalars!`), including the
#      `:inflow` halo of a tracer the parent does not carry.
#
# References: Hellsten et al. (2021), GMD 14, 3185, §3.1–3.3 (PALM nesting:
# Dirichlet child boundary, child substeps, mass-flux correction).

using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Fields: interior_view
using Aeolus.BoundaryConditions: FieldBoundaryConditions, OpenBoundary, open_boundary_states,
                                 add_open_scalars!, update_open_boundaries!
using Aeolus.Operators: divergence
using Aeolus.IBM: SolidMask, mark_solid!, apply_solid_mask!
using Aeolus.LargeScaleForcing: GeostrophicPressureGradient, apply_forcing!
using Aeolus.AdvectionSchemes: boundary_scalar_flux
using Aeolus.BoundaryConditions: ProfileBoundaryData, open_boundary_conditions

_nx_zc(m, k) = m.z0 + (k - m.Hz - 0.5) * m.dz

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

# The Berlin pattern of a two-way nest (#267): the caller drives each core
# itself, so the run's hooks cannot book the core-transported θ (and the
# moist parent's qv); the core's body_force! books them explicitly
# (book_nest_flux!: centred advection, the core's diffusion), and qv, a
# tracer of the dry child, is booked by the separate transport that moves
# it (forward-Euler WENO-Z5 after the dynamics, the :scalar_advection slot).
struct _NXBookedCore{F}
    q::F            # the tracer the slot moves after the dynamics (nothing: none)
    book::Bool
end
function (c::_NXBookedCore)(sim, dt; kw...)
    bf = (s, h) -> begin
        c.book || return nothing
        book_nest_flux!(s, :θ, s.θ, h; scheme = :centered, diffusivity = :theta)
        s.has_moisture && book_nest_flux!(s, :qv, s.qv, h; scheme = :centered, diffusivity = :tracer)
        return nothing
    end
    step!(sim, dt; kw..., body_force! = bf)
    c.q === nothing && return sim
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    fill_halos!(c.q, sim.θ_bcs)
    c.book && book_nest_flux!(sim, :qv, c.q, dt; scheme = :weno_z5)
    t = Field(CellCenter(), sim.mesh)
    Aeolus.AdvectionSchemes.advection_scheme(:weno_z5)(t, sim.u, sim.v, sim.w, c.q)
    interior_view(c.q) .+= dt .* interior_view(t)
    return sim
end

function _nx_transport_run(; book = true, nwin = 20)
    L = 1600.0
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (L, L, L), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; C_s = 0.17, ν = 2.0, moisture = true,
                             bcs = FieldBoundaryConditions(:periodic))
    nest = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; C_s = 0.17, ν = 2.0,
                             bcs = nest_bcs(cm, :nest, :nest, :nest; scalars = (:qv,)))
    rng = Xoshiro(267)
    fill!(p.u.data, 3.0); fill!(p.v.data, 1.0)
    interior_view(p.u) .+= 0.5 .* randn(rng, size(interior_view(p.u)))
    interior_view(p.v) .+= 0.5 .* randn(rng, size(interior_view(p.v)))
    Aeolus.Simulations._project_velocity!(p)
    for k in axes(p.θ.data, 3), j in axes(p.θ.data, 2), i in axes(p.θ.data, 1)
        b = exp(-((i - 8)^2 + (j - 8)^2 + (k - 8)^2) / 6)
        p.θ.data[i, j, k] = 300.0 + 2b
        p.qv.data[i, j, k] = 0.01 * (1 + 0.5b)
    end
    qc = Field(CellCenter(), cm)
    for (cf, pf) in ((c.θ, p.θ), (c.u, p.u), (c.v, p.v), (c.w, p.w), (qc, p.qv))
        interpolate_parent_to_child!(cf, pf, nest)
    end
    # a moist parent around a dry child that carries qv as a tracer: the
    # field set is built from the pairs (nest_interp_state(nest, p, c)
    # requires qv on both cores or neither)
    state = ntuple(_ -> nest_interp_state(nest; scalars = (:qv => (qc, p.qv),)), 3)
    fb = TwoWayFeedback(theta_flux = NestScalarFlux(applied_by = :transport),
                        tracer_fluxes = (:qv => NestScalarFlux(applied_by = :transport),))
    # the cores are closures with no declared stage weights: accepted, since
    # every scalar is booked by its transport
    adv_p = domain_advance(_NXBookedCore(nothing, book))
    adv_c = domain_advance(_NXBookedCore(qc, book))
    run = NestedRun(p, c, nest; state, feedback = fb, advance_parent = adv_p,
                    advance_child = adv_c, cfl_parent = nothing, cfl_child = nothing)
    I(f, g) = composite_integral(f, g, nest)
    θ0, q0 = I(p.θ, c.θ), I(p.qv, qc)
    dθ = dq = 0.0
    for _ in 1:nwin
        θa, qa = I(p.θ, c.θ), I(p.qv, qc)
        _nx_window!(run, adv_c, 4.0, 2.0)
        dθ = max(dθ, abs(I(p.θ, c.θ) - θa) / θ0)
        dq = max(dq, abs(I(p.qv, qc) - qa) / q0)
    end
    return dθ, dq, run
end

@testset "Era II — externally driven nest child (N-berlin-nest, #265)" begin

# -------------------------------------------------------------------------
@testset "a window driven from outside + subcycle_dt == step_nested!(run), bit for bit" begin
    for feedback in (:one_way, :two_way)
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
        for _ in 1:4
            step_nested!(A; dt = 0.6)
            n = _nx_window!(B, adv, 0.6, 0.6 / 3)
            @test n == 3
        end
        same = all(getproperty(A.child, f).data == getproperty(B.child, f).data for f in (:u, :v, :w, :θ)) &&
               all(getproperty(A.parent, f).data == getproperty(B.parent, f).data for f in (:u, :v, :w, :θ))
        @printf("  %-8s 4 windows × 3 substeps: child and parent bit-identical = %s\n", feedback, same)
        @test same
        @test A.child.t == B.child.t && A.t == B.t
        @test A.stats.parent_steps == B.stats.parent_steps == 4
        feedback === :two_way && @test A.nest.ledger == B.nest.ledger
    end
    # the substep rule: n = ⌈R/limit⌉, landing exactly on the window end
    @test subcycle_dt(1.0, 0.3) == 0.25
    @test subcycle_dt(0.9, 0.3) ≈ 0.3            # an integer ratio up to round-off: 3 steps, not 4
    @test subcycle_dt(0.2, 0.3) == 0.2
end

# -------------------------------------------------------------------------
# A parent at rest with a stable, hydrostatic θ(z) = 300 + 0.004 z; the
# child (no buildings, no fluxes) sits on the ground with open sides and
# top. Adaptive child steps limited to 0.37 Δt_p (they do not divide the
# window): 12 child steps over several windows.
#
# "The parent profile" on the child grid is the parent state prolongated
# (conservative, third order): exactly linear above the parent's lowest
# layer, but inside that layer the stencil reads the parent's mirror ghost
# below the wall, so the prolongated profile deviates from the line by up
# to 0.035 K (4 K/km, Δz_p = 40 m). The child therefore starts from the
# prolongated parent; with ν = 0 (no diffusion of that curvature, in either
# domain) the state must then not change at all.
@testset "uniform state preserved: parent at rest, hydrostatic θ(z)" begin
    θp(z) = 300 + 0.004z
    pm = RegularGrid(Float64; size = (16, 16, 12), extent = (640.0, 640.0, 480.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((160.0, 400.0), (240.0, 480.0), (0.0, 240.0)), (2, 2, 4); halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                                θ₀ = 300.0, ν = 0.0)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm; bottom = :wall, top = :nest), θ₀ = 300.0, ν = 0.0)
    for k in axes(psim.θ.data, 3)
        psim.θ.data[:, :, k] .= θp(_nx_zc(pm, k))
    end
    _nx_init_child!(csim, psim, nest)
    θ_init = copy(interior_view(csim.θ))
    dev = maximum(abs(θ_init[i, j, k] - θp(_nx_zc(cm, k + cm.Hz))) for i in 1:cm.Nx, j in 1:cm.Ny, k in 1:cm.Nz)
    adv = domain_advance(step!)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = adv, advance_child = adv,
                    cfl_parent = nothing, cfl_child = nothing)
    Δt_p = 2.0
    n = 0
    while n < 12
        n += _nx_window!(run, adv, Δt_p, 0.37Δt_p)
    end
    umax = maximum(maximum(abs, interior_view(f)) for f in (csim.u, csim.v, csim.w))
    θerr = maximum(abs, interior_view(csim.θ) .- θ_init)
    @printf("  at rest: %d child steps, max|u| = %.2e m/s, max|θ − θ_parent| = %.2e K (prolongated profile vs line: %.3f K)\n",
            n, umax, θerr, dev)
    @test n >= 12
    @test umax <= 1e-12
    @test θerr <= 1e-11            # round-off of 300 K values (eps·300 = 5.7e-14 per operation)
end

# -------------------------------------------------------------------------
@testset "Galilean: a parent at (2, 1) m/s leaves the child at (2, 1)" begin
    U, V = 2.0, 1.0
    pm = RegularGrid(Float64; size = (16, 16, 8), extent = (640.0, 640.0, 320.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((160.0, 400.0), (200.0, 440.0), (80.0, 240.0)), (2, 2, 2); halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; θ₀ = 300.0, ν = 1.0)            # triply periodic
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm), θ₀ = 300.0, ν = 1.0)   # six open faces
    for s in (psim, csim)
        fill!(s.u.data, U); fill!(s.v.data, V); fill!(s.θ.data, 300.0)
    end
    adv = domain_advance(step_rk3!)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = adv, advance_child = adv,
                    cfl_parent = nothing, cfl_child = nothing)
    for _ in 1:6
        _nx_window!(run, adv, 4.0, 1.7)
    end
    eu = maximum(abs, interior_view(csim.u) .- U)
    ev = maximum(abs, interior_view(csim.v) .- V)
    ew = maximum(abs, interior_view(csim.w))
    @printf("  Galilean: max|u − U| = %.2e, max|v − V| = %.2e, max|w| = %.2e m/s\n", eu, ev, ew)
    @test max(eu, ev, ew) <= 1e-10
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
    for _ in 1:4
        _nx_window!(run, adv, 0.01, 0.0041; each)
    end
    @printf("  mass: %d child steps, max net boundary flux / (U A) = %.2e, max|∇·u| = %.2e s⁻¹\n",
            nsteps, worst_res / (U * A), worst_div)
    @test nsteps >= 12
    @test worst_res <= 1e-12 * U * A
    @test worst_div <= 1e-10
end

# -------------------------------------------------------------------------
# A +1 K Gaussian bubble in a calm child under a calm, neutral parent. The
# child takes forward-Euler projection steps (`step!`, the Berlin
# stepper's core; centred θ advection, ν diffusion), and the heat flux
# through its open faces is taken in the body-force hook of every step —
# `boundary_scalar_flux` of the state the step's tendencies were built
# from, advective (mass-flux correction included) plus diffusive. Its
# time integral must equal the heat that left. The same bubble in a
# periodic box with a wall top keeps its heat.
function _nx_bubble!(sim, θ0, x0, y0, z0, σ)
    m = sim.mesh
    for k in axes(sim.θ.data, 3), j in axes(sim.θ.data, 2), i in axes(sim.θ.data, 1)
        x = m.x0 + (i - m.Hx - 0.5) * m.dx; y = m.y0 + (j - m.Hy - 0.5) * m.dy
        z = _nx_zc(m, k)
        sim.θ.data[i, j, k] = θ0 + exp(-((x - x0)^2 + (y - y0)^2 + (z - z0)^2) / (2σ^2))
    end
    return sim
end

_nx_heat(sim, θ0) = sum(interior_view(sim.θ) .- θ0) * sim.mesh.dx * sim.mesh.dy * sim.mesh.dz

@testset "warm-bubble export: boundary heat flux = heat that left; periodic keeps it" begin
    θ0, ν = 300.0, 0.5
    pm = RegularGrid(Float64; size = (16, 16, 12), extent = (640.0, 640.0, 480.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((160.0, 400.0), (240.0, 480.0), (0.0, 240.0)), (2, 2, 4); halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                                θ₀ = θ0, ν = ν)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm; bottom = :wall, top = :nest), θ₀ = θ0, ν = ν)
    fill!(psim.θ.data, θ0)
    # the bubble sits close to the top face (centre 30 m below it, σ = 15 m)
    _nx_bubble!(csim, θ0, 280.0, 360.0, 210.0, 15.0)
    H0 = _nx_heat(csim, θ0)
    F = Ref(0.0)
    hook = (s, Δt) -> (F[] = boundary_scalar_flux(s, s.θ; scheme = :centered, fill = false,
                                                     diffusivity = (ν = ν, κh = nothing, κv = nothing)).total; s)
    advc = domain_advance(step!; body_force! = hook)
    adv = domain_advance(step!)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = adv, advance_child = advc,
                    cfl_parent = nothing, cfl_child = nothing)
    exported = 0.0
    for _ in 1:40
        _nx_window!(run, advc, 2.0, 0.7; each = h -> (exported += F[] * h))
    end
    H1 = _nx_heat(csim, θ0)
    err = abs((H0 - H1) - exported)
    @printf("  open child: heat at start %.4e K m³, left %.4e (%.1f %%), booked %.4e, |difference|/H0 = %.2e\n",
            H0, H0 - H1, 100 * (H0 - H1) / H0, exported, err / H0)
    @test (H0 - H1) / H0 > 1e-3              # heat really left through the faces
    @test err <= 1e-10 * H0
    # the periodic run keeps it
    pcm = RegularGrid(Float64; size = (cm.Nx, cm.Ny, cm.Nz), extent = (cm.Lx, cm.Ly, cm.Lz),
                      halo = (3, 3, 3), origin = (cm.x0, cm.y0, cm.z0))
    per = BoussinesqSimulation(pcm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall), θ₀ = θ0, ν = ν)
    _nx_bubble!(per, θ0, 280.0, 360.0, 210.0, 15.0)
    P0 = _nx_heat(per, θ0)
    for _ in 1:40, _ in 1:3
        step!(per, 2.0 / 3)
    end
    perr = abs(_nx_heat(per, θ0) - P0) / P0
    @printf("  periodic child: |Δ heat|/H0 = %.2e\n", perr)
    @test perr <= 1e-11
end

# -------------------------------------------------------------------------
@testset "hooks in both domains: parent geostrophic tendency, child IBM" begin
    f, Ug, Vg = 1e-4, 3.0, -1.5
    pm = RegularGrid(Float64; size = (16, 16, 8), extent = (640.0, 640.0, 320.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((160.0, 400.0), (200.0, 440.0), (0.0, 160.0)), (2, 2, 2); halo = (3, 3, 3))
    cm = nest_child_mesh(nest)
    psim = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall),
                                θ₀ = 300.0, f_coriolis = f, ν = 1.0)
    csim = BoussinesqSimulation(cm; bcs = nest_bcs(cm; bottom = :wall, top = :nest),
                                θ₀ = 300.0, f_coriolis = f, ν = 1.0)
    for s in (psim, csim)
        fill!(s.u.data, Ug); fill!(s.v.data, Vg); fill!(s.θ.data, 300.0)
    end
    G = GeostrophicPressureGradient(Ug, Vg)
    worst = Ref(0.0)
    phook = (s, Δt) -> begin
        tu0 = copy(s.tu.data); tv0 = copy(s.tv.data)
        apply_forcing!(s, G, s.t, Δt)
        worst[] = max(worst[], maximum(abs, (s.tu.data .- tu0)[s.mesh.Hx+1:s.mesh.Hx+s.mesh.Nx,
                                                                s.mesh.Hy+1:s.mesh.Hy+s.mesh.Ny,
                                                                s.mesh.Hz+1:s.mesh.Hz+s.mesh.Nz] .+ f * Vg),
                      maximum(abs, (s.tv.data .- tv0)[s.mesh.Hx+1:s.mesh.Hx+s.mesh.Nx,
                                                      s.mesh.Hy+1:s.mesh.Hy+s.mesh.Ny,
                                                      s.mesh.Hz+1:s.mesh.Hz+s.mesh.Nz] .- f * Ug))
        s
    end
    # the Berlin IBM coupling's correction (direct forcing, τ = 0) in the
    # child's post-predictor hook: it zeroes the predicted velocity inside
    # the solid; the projection then adds only its pressure-gradient
    # correction there
    sm = SolidMask(cm)
    mark_solid!(sm, (x, y, z) -> 240.0 < x < 300.0 && 280.0 < y < 340.0 && z < 60.0)
    zeroed = Ref(true)
    ibm = (s, Δt) -> begin
        Aeolus.IBM.implicit_ibm_correction!(s, sm; τ = 0.0, dt = Δt)
        zeroed[] &= all(iszero(s.u.data[i, j, k]) for (i, j, k) in Tuple.(findall(sm.mask)))
        s
    end
    advp = domain_advance(step!; body_force! = phook)
    advc = domain_advance(step!; body_force! = (s, Δt) -> (apply_forcing!(s, G, s.t, Δt); s),
                          post_predictor! = ibm)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = advp, advance_child = advc,
                    cfl_parent = nothing, cfl_child = nothing, solid = sm)
    for _ in 1:5
        _nx_window!(run, advc, 4.0, 1.5)
    end
    mk = view(sm.mask, cm.Hx+1:cm.Hx+cm.Nx, cm.Hy+1:cm.Hy+cm.Ny, cm.Hz+1:cm.Hz+cm.Nz)
    Uh = sqrt.(interior_view(csim.u) .^ 2 .+ interior_view(csim.v) .^ 2)
    u_in, u_out = maximum(Uh[mk]), maximum(Uh[.!mk])
    @printf("  parent geostrophic tendency error %.2e m/s²; child |u_h| in solids %.3f, outside %.2f m/s\n",
            worst[], u_in, u_out)
    @test worst[] <= 1e-18
    @test count(mk) > 0
    @test zeroed[]                               # the hook ran in the child, every step
    @test u_in < 0.1 * u_out                     # the solid holds (the Berlin smoke check asks < 0.3)
    # the geostrophic balance keeps the parent's uniform wind
    @test maximum(abs, interior_view(psim.u) .- Ug) <= 1e-12
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
    @test_throws ArgumentError add_open_scalars!(bcs, :u)
    @test_throws ArgumentError Aeolus.BoundaryConditions.OpenPlane(:u, FaceX(), :west, cm; mode = :inflow)
    st = ntuple(_ -> nest_interp_state(nest; scalars = (:a => (ac, ap), :b => (bc_, b0))), 3)
    adv = domain_advance(step!)
    run = NestedRun(psim, csim, nest; feedback = :one_way, advance_parent = adv, advance_child = adv,
                    cfl_parent = nothing, cfl_child = nothing, state = st)
    fill_halos!(ac, csim.θ_bcs); fill_halos!(bc_, csim.θ_bcs)
    Hx, Nx = cm.Hx, cm.Nx
    # west face (inflow, u > 0): `a` takes the parent's 2, `b` the parent's 0
    @test all(==(2.0), ac.data[1:Hx, :, :])
    @test all(==(0.0), bc_.data[1:Hx, :, :])
    # east face (outflow): `a` takes the parent's value (Dirichlet), `b` the
    # mirror of the interior (its own 5)
    @test all(==(2.0), ac.data[Hx+Nx+1:end, :, :])
    @test all(==(5.0), bc_.data[Hx+Nx+1:end, :, :])
    # reversing the wind swaps the roles of the faces for `b`
    for s in (psim, csim)
        fill!(s.u.data, -U)
    end
    update_open_boundaries!(csim, csim.t, 0.0)            # buffers from the (unchanged) data …
    interpolate_parent_state!(run.s_old, psim, nest; order = 3, region = :halo_interface)
    apply_nest_boundary!(csim, run.s_old)                  # … refreshed from the reversed parent
    fill_halos!(bc_, csim.θ_bcs)
    @test all(==(5.0), bc_.data[1:Hx, :, :])
    @test all(==(0.0), bc_.data[Hx+Nx+1:end, :, :])
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
    @printf("  limited outflow: Δ∫q dV = %.6e, booked (limited fluxes) %.6e, re-reconstructed %.6e\n",
            change, -out, -dt * rec.total)
    @test abs(change + out) <= 1e-12 * abs(change)
    @test abs(change + dt * rec.total) > 1e-6 * abs(change)    # the reconstruction is not what left
    @test minimum(qi .+ dt .* interior_view(tend)) >= 0
end


# -------------------------------------------------------------------------
# The Berlin two-way pattern (#267): an externally driven child whose core θ
# and tracer qv, and the moist parent's θ and qv, are all booked by their
# transports (book_nest_flux!, inside the child-step bracket). Composite ∫θ
# and ∫qv must be conserved to round-off per parent window; without the
# bookings they drift. A moist parent may declare qv a transport-booked
# tracer when the child carries it as a tracer.
@testset "two-way, every scalar booked by its transport (the Berlin pattern, #267)" begin
    dθ, dq, run = _nx_transport_run()
    dθn, dqn, _ = _nx_transport_run(; book = false)
    @printf("  externally driven two-way, 20 windows (Δt_p = 4 s, 2 child steps): composite drift per window ∫θ %.1e, ∫qv %.1e; not booked ∫θ %.1e, ∫qv %.1e\n",
            dθ, dq, dθn, dqn)
    @test run.regs !== nothing && run.regs.names == [:θ, :qv]
    @test run.regs.pweights == () && run.regs.cweights == ()     # no core hook needed
    @test dθ <= 1e-12 && dq <= 1e-12
    @test dθn > 1e3 * max(dθ, 1e-16) && dqn > 1e3 * max(dq, 1e-16)
    @test run.nest.ledger[:mass] <= 1e-12
    @test all(isfinite, run.child.u.data) && all(isfinite, run.parent.θ.data)
    # outside a child-step bracket nothing is booked
    @test !book_nest_flux!(run.child, :θ, run.child.θ, 1.0; scheme = :centered)
    Aeolus.Nesting._begin!(run, :child)
    @test_throws ArgumentError book_nest_flux!(run.child, :θ, run.child.θ, 1.0; diffusivity = :nope)
    Aeolus.Nesting._commit!(run, :child)
end

end
