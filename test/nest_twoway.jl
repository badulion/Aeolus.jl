using Test
using Printf
using Random
using Aeolus
using Aeolus.Nesting
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view

# =========================================================================
# N-nest-e (#263) — two-way feedback: anterpolation (Clark & Farley 1984)
# with buffer zone, canopy restriction, inflow exclusion and relaxation,
# Berger–Colella scalar refluxing from flux registers, re-projection of the
# parent. The long runs (CBL, neutral, interface noise, outgoing vortex)
# are in validation/nest_twoway_longrun.jl.
#
#   1. composite conservation of ∫θ, ∫qv, ∫tracer in a closed periodic
#      parent (step!, Smagorinsky, molecular diffusion), with the negative
#      control without refluxing; mixed-core refusal, one flux kernel
#      (#327 M1, M2), a tracer booked by its transport (#327 M3);
#   2. relaxation: (1 − Δt/τ)ⁿ decay, τ = Δt replaces;
#   3. masks: buffer, canopy restriction and inflow band are the same
#      to feedback off, the rest is changed;
#   4. idempotence of the feedback;
#   5. the heat budget of a two-way nest with a child-only heat source Q;
#   6. the solid-aware field-set restriction (a bug from #256 fixed here).
# =========================================================================

const _TW_L = 1600.0
const _TW_UPWIND = advection_scheme(:upwind)

# Closed, triply periodic parent (16³, Δ = 100 m) and a child with six
# open faces at r = 2 covering the middle 8³ parent cells. No buoyancy
# (θ₀ = 0), so θ is a passive scalar of the flow: a smooth blob in a
# (3, 1, 0) m/s stream with projected noise.
function _tw_box(; moist = false, C_s = 0.17, ν = 2.0, tracer = false, seed = 1,
                 feedback = TwoWayFeedback(), core = step!, U = (3.0, 1.0),
                 noise = 0.5, kw...)
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (_TW_L, _TW_L, _TW_L),
                     halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; C_s, ν, moisture = moist,
                             bcs = FieldBoundaryConditions(:periodic))
    nest = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; C_s, ν, moisture = moist,
                             bcs = nest_bcs(cm, :nest, :nest, :nest; moisture = moist,
                                            scalars = tracer ? (:co2,) : ()))
    rng = Xoshiro(seed)
    blob(i, j, k, i0) = exp(-((i - i0)^2 + (j - 8)^2 + (k - 8)^2) / 6)
    for k in axes(p.θ.data, 3), j in axes(p.θ.data, 2), i in axes(p.θ.data, 1)
        p.θ.data[i, j, k] = 1.0 + blob(i, j, k, 8)
        p.qv.data[i, j, k] = moist ? 0.01 * (1 + blob(i, j, k, 10)) : 0.0
    end
    fill!(p.u.data, U[1]); fill!(p.v.data, U[2])
    interior_view(p.u) .+= noise .* randn(rng, size(interior_view(p.u)))
    interior_view(p.v) .+= noise .* randn(rng, size(interior_view(p.v)))
    Aeolus.Simulations._project_velocity!(p)
    for (cf, pf) in ((c.θ, p.θ), (c.u, p.u), (c.v, p.v), (c.w, p.w), (c.qv, p.qv))
        interpolate_parent_to_child!(cf, pf, nest)
    end
    scal = ()
    if tracer
        trp = Field(CellCenter(), pm); trc = Field(CellCenter(), cm)
        for k in axes(trp.data, 3), j in axes(trp.data, 2), i in axes(trp.data, 1)
            trp.data[i, j, k] = 400.0 + 50 * blob(i, j, k, 6)
        end
        interpolate_parent_to_child!(trc, trp, nest)
        scal = (:co2 => (trc, trp),)
    end
    # the tracer's owner: first-order upwind advection (forward Euler with
    # centred advection and no diffusion is unstable: the tracer's 2Δ noise
    # grows without bound), inside body_force! (after the run's register
    # hook, so the registers see the stage-start tracer); with step! that is
    # one stage per step. The registers book the same scheme:
    # NestScalarFlux(scheme = :upwind).
    own(f) = (sim, Δt) -> begin
        fill_halos!(f, sim.θ_bcs)
        t = Field(CellCenter(), sim.mesh)
        _TW_UPWIND(t, sim.u, sim.v, sim.w, f)
        interior_view(f) .+= Δt .* interior_view(t)
    end
    ap = tracer ? domain_advance(core; body_force! = own(last(scal[1])[2])) :
                  domain_advance(core)
    ac = tracer ? domain_advance(core; body_force! = own(last(scal[1])[1])) :
                  domain_advance(core)
    run = NestedRun(p, c, nest; feedback, scalars = scal, advance_parent = ap,
                    advance_child = ac, cfl_parent = nothing, cfl_child = nothing, kw...)
    return p, c, nest, run, scal
end

# Composite integrals of θ (and qv, co2) and their ∫|φ| normalisers.
function _tw_integrals(p, c, nest, scal)
    out = Dict{Symbol,Float64}(:θ => composite_integral(p.θ, c.θ, nest))
    p.has_moisture && (out[:qv] = composite_integral(p.qv, c.qv, nest))
    for pr in scal
        out[first(pr)] = composite_integral(last(pr)[2], last(pr)[1], nest)
    end
    return out
end

# #327 M3: a tracer moved by a separate transport, after the dynamics, with
# its own scheme (forward-Euler WENO-Z5, the Berlin :scalar_advection
# pattern), booked by the transport itself through book_nest_flux!. The user
# core takes a step! (one body_force! stage) and then transports, so it
# declares the weights of step!.
const _TW_WENO = advection_scheme(:weno_z5)
struct _TWSlotCore{F}
    f::F                               # the tracer the slot transports
    book::Bool                         # call book_nest_flux! (the transport's own booking)
    weighted::Bool                     # W-weighted centred transport (sound-proof core)
end
function (c::_TWSlotCore)(sim, dt; kw...)
    step!(sim, dt; kw...)
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    fill_halos!(c.f, sim.θ_bcs)
    t = Field(CellCenter(), sim.mesh)
    if c.weighted
        c.book && book_nest_flux!(sim, :qv, c.f, dt; scheme = :centered)
        Wc, Wf = Aeolus.Anelastic.constraint_weights(sim)
        Aeolus.Anelastic.weighted_scalar_advect!(t, sim.u, sim.v, sim.w, c.f, Wc, Wf, sim.mesh)
    else
        c.book && book_nest_flux!(sim, :co2, c.f, dt; scheme = :weno_z5)
        _TW_WENO(t, sim.u, sim.v, sim.w, c.f)
    end
    interior_view(c.f) .+= dt .* interior_view(t)
    return sim
end
Aeolus.Nesting.nest_stage_weights(::_TWSlotCore, args...) = (1.0,)

function _tw_slot_run(model; book = true, nsteps = 100)
    pm = RegularGrid(Float64; size = (16, 16, 16), extent = (_TW_L, _TW_L, _TW_L),
                     halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; C_s = 0.17, ν = 2.0, bcs = FieldBoundaryConditions(:periodic))
    nest = nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; C_s = 0.17, ν = 2.0,
                             bcs = nest_bcs(cm, :nest, :nest, :nest; scalars = (:co2,)))
    rng = Xoshiro(4)
    fill!(p.θ.data, 1.0); fill!(p.u.data, 3.0); fill!(p.v.data, 1.0)
    interior_view(p.u) .+= 0.5 .* randn(rng, size(interior_view(p.u)))
    interior_view(p.v) .+= 0.5 .* randn(rng, size(interior_view(p.v)))
    Aeolus.Simulations._project_velocity!(p)
    trp = Field(CellCenter(), pm); trc = Field(CellCenter(), cm)
    for k in axes(trp.data, 3), j in axes(trp.data, 2), i in axes(trp.data, 1)
        trp.data[i, j, k] = 400.0 + 50 * exp(-((i - 8)^2 + (j - 8)^2 + (k - 8)^2) / 6)
    end
    for (cf, pf) in ((c.θ, p.θ), (c.u, p.u), (c.v, p.v), (c.w, p.w), (trc, trp))
        interpolate_parent_to_child!(cf, pf, nest)
    end
    run = NestedRun(p, c, nest; scalars = (:co2 => (trc, trp),),
                    feedback = TwoWayFeedback(tracer_fluxes = (:co2 => model,)),
                    advance_parent = domain_advance(_TWSlotCore(trp, book, false)),
                    advance_child = domain_advance(_TWSlotCore(trc, book, false)),
                    cfl_parent = nothing, cfl_child = nothing)
    I0 = composite_integral(trp, trc, nest)
    worst = 0.0
    for _ in 1:nsteps
        Ia = composite_integral(trp, trc, nest)
        step_nested!(run; dt = 4.0)            # forward-Euler WENO: CFL ≲ 0.2
        worst = max(worst, abs(composite_integral(trp, trc, nest) - Ia) / I0)
    end
    return worst, abs(composite_integral(trp, trc, nest) - I0) / I0, run
end

@testset "two-way nest feedback (N-nest-e, #263)" begin

# -------------------------------------------------------------------------
# 1. Composite conservation. A closed periodic parent with no sources: the
#    composite integral of every refluxed scalar may change only by
#    round-off. Smagorinsky SGS and molecular diffusion are on, so the
#    registers must book the advective, molecular and SGS face fluxes the
#    cores used. Without refluxing the integrals drift (negative control).
@testset "composite conservation of ∫θ, ∫qv, ∫tracer: refluxing on and off" begin
    function drift(reflux, nsteps)
        dp, dc, dnest, drun, dscal = _tw_box(; moist = true, tracer = true,
            feedback = TwoWayFeedback(; reflux,
                                      tracer_fluxes = (:co2 => NestScalarFlux(scheme = :upwind),)))
        J0 = _tw_integrals(dp, dc, dnest, dscal)
        dper = Dict(k => 0.0 for k in keys(J0))
        Jprev = copy(J0)
        for n in 1:nsteps
            step_nested!(drun; dt = 4.0)
            J = _tw_integrals(dp, dc, dnest, dscal)
            for k in keys(J)
                dper[k] = max(dper[k], abs(J[k] - Jprev[k]) / abs(J0[k]))
            end
            Jprev = J
        end
        dtot = Dict(k => abs(Jprev[k] - J0[k]) / abs(J0[k]) for k in keys(J0))
        return dper, dtot, drun
    end
    per, tot, run = drift(true, 10)
    per0, _, _ = drift(false, 5)
    @test maximum(per[k] for k in (:θ, :qv, :co2)) <= 1e-12
    @test maximum(tot[k] for k in (:θ, :qv, :co2)) <= 1e-11
    @test all(per0[k] > 1e4 * per[k] for k in (:θ, :qv, :co2))      # negative control
    @test run.stats.parent_steps == 10
    @test max_divergence(run.parent) < 1e-10
    # SSP-RK3 cores: the registers weight the three Euler stages by 1/6, 1/6, 2/3
    @test Aeolus.Nesting._stage_weights(domain_advance(step_rk3!)) == (1 / 6, 1 / 6, 2 / 3)
    @test nest_stage_weights(step_imex!, ARK2()) === nothing
end

# #327 M1: a mixed Boussinesq / sound-proof pair has no consistent measure
# for two-way transfers. Two-way is refused; the same pair nests one-way.
@testset "mixed-core pairs: two-way refused, one-way allowed (#327 M1)" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0),
                     halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    rc = isothermal_reference(cm; T_surf = 300.0)
    pB = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    cA = AnelasticSimulation(cm; ref = rc, bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    @test_throws ArgumentError NestedRun(pB, cA, nest; feedback = :two_way, cfl_parent = nothing)
    @test NestedRun(pB, cA, nest; feedback = :one_way, cfl_parent = nothing).feedback === :one_way
end

# #327 M2: one per-face advective + diffusive kernel for the flux registers
# and boundary_scalar_flux; they must agree face by face.
@testset "one flux kernel for the flux registers and boundary_scalar_flux (#327 M2)" begin
    p, c, nest, run, _ = _tw_box(; ν = 2.0)
    for _ in 1:3
        step_nested!(run; dt = 10.0)
    end
    cm = c.mesh
    cl = Aeolus.Operators._active_sgs(c, nothing, nothing)
    f = Aeolus.Operators.sgs_fields(cl, c)
    K = Aeolus.Operators.sgs_scalar_face_diffusivity(cl, f, c)        # (νt, νt, 1/Prₜ)
    @test maximum(K[1].data) > 0
    B = boundary_scalar_flux(c, c.θ; diffusivity = (ν = c.ν, κh = K[1], κv = K[2], scale = K[3]))
    rec = Aeolus.AdvectionSchemes.Centered2()
    A = (cm.dy * cm.dz, cm.dy * cm.dz, cm.dx * cm.dz, cm.dx * cm.dz, cm.dx * cm.dy, cm.dx * cm.dy)
    worst_reg = 0.0
    for (s, side) in enumerate((:west, :east, :south, :north, :bottom, :top))
        _, csz = Aeolus.Nesting._plane_sizes(nest, s)
        M = zeros(csz)
        Aeolus.Nesting._book_plane!(M, c, c.θ, rec, K, Float64(c.ν), nothing, nest, s, false, 1.0)
        reg = (iseven(s) ? 1 : -1) * sum(M) * A[s]                   # outward, m³/s · K
        worst_reg = max(worst_reg, abs(reg - getproperty(B, side)) / abs(getproperty(B, side)))
    end
    @test worst_reg < 1e-12
end

# #327 M3: the registers book the flux the transport actually applied.
@testset "separately transported tracer: WENO-Z5 slot (#327 M3)" begin
    # booked by the transport (applied_by = :transport) at its own state
    per, tot, run = _tw_slot_run(NestScalarFlux(applied_by = :transport); nsteps = 10)
    # the pre-#327 way: the hook books WENO-Z5 at the core's stage state
    per_c, _, _ = _tw_slot_run(NestScalarFlux(scheme = :weno_z5); nsteps = 5)
    @test per <= 1e-12 && tot <= 1e-11
    @test per_c > 1e3 * per                     # the stage-state booking is wrong
    trc = run.s_bc.child[1]
    @test !book_nest_flux!(run.child, :co2, trc, 1.0; scheme = :weno_z5)   # no step under way
    @test_throws ArgumentError TwoWayFeedback(tracer_fluxes = (:θ => NestScalarFlux(),))
end

# -------------------------------------------------------------------------
# 2. Relaxation. Parent and child at rest, no diffusion: the covered parent
#    cells, initially 0, relax toward the child's uniform 1,
#    d_n = (1 − Δt/τ)ⁿ d₀. τ = Δt (and τ = nothing) replace in one step.
@testset "relaxation: (1 − Δt/τ)ⁿ decay, τ = Δt replaces exactly" begin
    function rest_run(τ)
        pm = RegularGrid(Float64; size = (8, 8, 8), extent = (800.0, 800.0, 800.0),
                         halo = (3, 3, 3))
        p = BoussinesqSimulation(pm)
        nest = nested_grid(pm, ((200.0, 600.0), (200.0, 600.0), (200.0, 600.0)), 2)
        cm = nest_child_mesh(nest)
        c = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest, :nest))
        fill!(p.θ.data, 0.0); fill!(c.θ.data, 1.0)
        run = NestedRun(p, c, nest; feedback = TwoWayFeedback(; τ, buffer = 0),
                        cfl_parent = nothing, cfl_child = nothing)
        return p, c, nest, run
    end
    dt = 2.0
    p, c, nest, run = rest_run(10dt)
    cx, cy, cz = nest_covered_range(nest)
    dev() = maximum(abs, interior_view(p.θ)[cx, cy, cz] .- 1)
    d = Float64[dev()]
    for n in 1:20
        step_nested!(run; dt)
        push!(d, dev())
    end
    @test maximum(abs, d .- [(1 - dt / (10dt))^n for n in 0:20]) < 1e-12
    mask = trues(size(interior_view(p.θ))); mask[cx, cy, cz] .= false
    @test all(iszero, interior_view(p.θ)[mask])      # outside the nest nothing happens
    worst = 0.0
    for τ in (dt, nothing)
        p, c, nest, run = rest_run(τ)
        step_nested!(run; dt)
        worst = max(worst, maximum(abs, interior_view(p.θ)[cx, cy, cz] .- 1))
    end
    @test worst < 1e-15
end

# -------------------------------------------------------------------------
# 3. Masks. From the same state, one parent step one-way and one two-way:
#    the covered parent cells in the buffer (2 cells), below z_anterp (the
#    lowest 2 covered layers) and in the inflow band (4 cells at the west
#    face) agree with feedback off, the others are changed.
@testset "masks: buffer, canopy restriction, inflow band the same as feedback off" begin
    fb = TwoWayFeedback(; buffer = 2, z_anterp = 600.0, exclude_width = 4,
                        exclude_sides = (:west,), reproject = false)
    p1, c1, nest, run1, _ = _tw_box(; moist = true, feedback = :one_way, ν = 0.0)
    p2, c2, _, run2, _ = _tw_box(; moist = true, feedback = fb, ν = 0.0)
    step_nested!(run1; dt = 4.0); step_nested!(run2; dt = 4.0)
    act = feedback_mask(run2)
    nx, ny, nz = nest.nx, nest.ny, nest.nz
    expect = trues(nx, ny, nz)
    expect[[1:4; nx-1:nx], :, :] .= false             # west band (4), east buffer (2)
    expect[:, [1, 2, ny - 1, ny], :] .= false          # south / north buffer
    expect[:, :, [1, 2, nz - 1, nz]] .= false          # bottom buffer and z < 600 m, top buffer
    @test act == expect
    cx, cy, cz = nest_covered_range(nest)
    a = interior_view(p1.θ)[cx, cy, cz]; b = interior_view(p2.θ)[cx, cy, cz]
    @test values_agree(a[.!act], b[.!act])
    @test all(a[act] .!= b[act])
end

# -------------------------------------------------------------------------
# 4. Idempotence: the feedback applied a second time changes nothing but
#    round-off (the registers were consumed, the anterpolated parent is
#    the child mean already, the flux-corrected interface agrees).
@testset "idempotence: feedback twice equals once" begin
    p, c, nest, run, _ = _tw_box(; moist = true, feedback = TwoWayFeedback(; buffer = 0))
    for n in 1:3
        step_nested!(run; dt = 4.0)
    end
    snap = [copy(f.data) for f in (p.u, p.v, p.w, p.θ, p.qv)]
    two_way_feedback!(run)
    dmax = maximum(maximum(abs, f.data .- s) / max(maximum(abs, s), 1)
                   for (f, s) in zip((p.u, p.v, p.w, p.θ, p.qv), snap))
    @test dmax <= 1e-13
end

# -------------------------------------------------------------------------
# 5. A heat source Q = 50 MW in a block of child cells (a source the parent
#    does not see), two-way, SSP-RK3: the heat reaches the parent through
#    the feedback and the refluxed interface fluxes, and the parent's θ
#    excess outside the nest is exactly Q·t/(ρc_p) minus what is still in
#    the child.
@testset "two-way: a child-only heat source, heat budget" begin
    ρ, cp = 1.2, 1005.0
    Q = 50.0e6                                         # W
    nsteps, dt = 40, 5.0
    pm = RegularGrid(Float64; size = (32, 8, 8), extent = (3200.0, 800.0, 800.0), halo = (3, 3, 3))
    p = BoussinesqSimulation(pm; C_s = 0.17, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    fill!(p.θ.data, 300.0); fill!(p.u.data, 5.0)
    nest = nested_grid(pm, ((800.0, 1600.0), (0.0, 800.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    c = BoussinesqSimulation(cm; C_s = 0.17, bcs = nest_bcs(cm, :nest, :periodic; bottom = :wall, top = :nest))
    fill!(c.θ.data, 300.0); fill!(c.u.data, 5.0)
    src = (cm.Hx .+ (5:8), cm.Hy .+ (1:cm.Ny), cm.Hz .+ (2:4))   # a 200 m × 800 m × 150 m block
    Vs = length(src[1]) * length(src[2]) * length(src[3]) * cm.dx * cm.dy * cm.dz
    q = Q / (ρ * cp * Vs)                              # K/s in the block
    heat!(sim, Δt) = (view(sim.tθ.data, src...) .+= q; nothing)
    run = NestedRun(p, c, nest; feedback = :two_way, cfl_parent = nothing, cfl_child = nothing,
                    advance_parent = domain_advance(step_rk3!),
                    advance_child = domain_advance(step_rk3!; body_force! = heat!))
    for n in 1:nsteps
        step_nested!(run; dt)
    end
    cx, cy, cz = nest_covered_range(nest)
    outside = 0.0
    for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
        (i in cx && j in cy && k in cz) && continue
        outside += (p.θ.data[pm.Hx+i, pm.Hy+j, pm.Hz+k] - 300.0) * pm.dx * pm.dy * pm.dz
    end
    inchild = sum(interior_view(c.θ) .- 300.0) * cm.dx * cm.dy * cm.dz
    injected = Q * nsteps * dt / (ρ * cp)
    @test abs(outside - (injected - inchild)) / injected < 1e-10
    @test outside > 0                                  # the plume left the nest into the parent
end

# -------------------------------------------------------------------------
# 6. The solid-aware restriction of the whole field set. Before N-nest-e,
#    restrict_state_to_parent! handed its fluid mask back as `solid`, which
#    was inverted again: the field set was averaged over the solid
#    children. It must equal restrict_to_parent! with the same mask.
@testset "solid-aware field-set restriction equals the single-field operator" begin
    pm = RegularGrid(Float64; size = (8, 8, 8), extent = (8.0, 8.0, 8.0), halo = (3, 3, 3))
    nest = nested_grid(pm, ((2.0, 6.0), (2.0, 6.0), (0.0, 4.0)), 2)
    cm = nest_child_mesh(nest)
    p = BoussinesqSimulation(pm; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    c = BoussinesqSimulation(cm; bcs = nest_bcs(cm, :nest, :nest, :wall))
    rng = Xoshiro(5)
    c.θ.data .= randn(rng, size(c.θ.data))
    solid = falses(size(c.θ.data)); solid[cm.Hx .+ (1:3), cm.Hy .+ (1:5), cm.Hz .+ (1:3)] .= true
    ref = Field(CellCenter(), pm)
    restrict_to_parent!(ref, c.θ, nest; solid)
    restrict_state_to_parent!(p, c, nest; solid)
    cx, cy, cz = nest_covered_range(nest)
    @test agree(interior_view(p.θ)[cx, cy, cz], interior_view(ref)[cx, cy, cz])
end

end # testset
