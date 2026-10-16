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
# are in era2_nest_twoway_longrun.jl.
#
#   1. composite conservation of ∫θ, ∫qv, ∫tracer in a closed periodic
#      parent (step!, Smagorinsky, molecular diffusion), with the negative
#      control without refluxing; SSP-RK3 cores; an anelastic pair (∫ρ̄θ);
#   2. relaxation: (1 − Δt/τ)ⁿ decay, e-folding = τ, τ = Δt replaces;
#   3. masks: buffer, canopy restriction and inflow band are the same
#      to feedback off, the rest is changed;
#   4. idempotence of the feedback;
#   5. one-way against two-way with a child-only heat source Q;
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

@testset "Era II — two-way nest feedback (N-nest-e, #263)" begin

# -------------------------------------------------------------------------
# 1. Composite conservation. A closed periodic parent with no sources: the
#    composite integral of every refluxed scalar may change only by
#    round-off. Smagorinsky SGS and molecular diffusion are on, so the
#    registers must book the advective, molecular and SGS face fluxes the
#    cores used. 2000 parent steps (4000 child substeps).
@testset "composite conservation of ∫θ, ∫qv, ∫tracer: refluxing on and off" begin
    # (the names inside are distinct from the testset's: a closure assigning
    # a name the enclosing scope also assigns would share that variable)
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
    per, tot, run = drift(true, 2000)
    per0, tot0, _ = drift(false, 200)
    for k in (:θ, :qv, :co2)
        @printf("  %-3s: with refluxing max drift per step %.2e, after 2000 steps %.2e | without: per step %.2e, after 200 steps %.2e (ratio per step %.1e)\n",
                k, per[k], tot[k], per0[k], tot0[k], per0[k] / max(per[k], 1e-300))
        @test per[k] <= 1e-12
        @test tot[k] <= 1e-10
        @test per0[k] > 1e4 * per[k]               # negative control
    end
    @test run.stats.parent_steps == 2000
    @test max_divergence(run.parent) < 1e-10
end

# SSP-RK3 cores: the registers weight the three Euler stages by 1/6, 1/6,
# 2/3, the weights of the SSP-RK3 combination.
@testset "composite conservation with step_rk3! (stage weights 1/6, 1/6, 2/3)" begin
    p, c, nest, run, scal = _tw_box(; moist = true, core = step_rk3!)
    I0 = _tw_integrals(p, c, nest, scal)
    for n in 1:100
        step_nested!(run; dt = 8.0)
    end
    I = _tw_integrals(p, c, nest, scal)
    for k in (:θ, :qv)
        d = abs(I[k] - I0[k]) / abs(I0[k])
        @printf("  step_rk3!: %s composite drift after 100 steps %.2e\n", k, d)
        @test d <= 1e-11
    end
    # wrong weights would be visible: Euler weights (1, 1, 1) triple the booked flux
    @test Aeolus.Nesting._stage_weights(domain_advance(step_rk3!)) == (1 / 6, 1 / 6, 2 / 3)
end

# A sound-proof (anelastic) pair: the registers book the ρ̄-weighted mass
# fluxes, refluxing conserves the composite ∫ρ̄θ dV (inviscid; the
# anelastic core's diffusion is not in the W-weighted flux form, see the
# theory page).
@testset "anelastic pair: composite ∫ρ̄θ conserved by mass-flux refluxing" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0),
                     halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    rp = isothermal_reference(pm; T_surf = 300.0); rc = isothermal_reference(cm; T_surf = 300.0)
    pA = AnelasticSimulation(pm; ref = rp, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    cA = AnelasticSimulation(cm; ref = rc,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest))
    rng = Xoshiro(3)
    for k in axes(pA.θ.data, 3), j in axes(pA.θ.data, 2), i in axes(pA.θ.data, 1)
        pA.θ.data[i, j, k] = rp.θ̄c_pad[k] + 0.5 * exp(-((i - 7)^2 + (j - 7)^2 + (k - 5)^2) / 5)
    end
    fill!(pA.u.data, 4.0)
    interior_view(pA.v) .+= 0.3 .* randn(rng, size(interior_view(pA.v)))
    Aeolus.Anelastic._project_mass_weighted!(pA)
    for (cf, pf) in ((cA.θ, pA.θ), (cA.u, pA.u), (cA.v, pA.v), (cA.w, pA.w))
        interpolate_parent_to_child!(cf, pf, nest; measure = nest_measure(nest, pA, cA))
    end
    run = NestedRun(pA, cA, nest; feedback = :two_way, cfl_parent = nothing, cfl_child = nothing)
    meas = run.s_bc.measure
    I0 = composite_integral(pA.θ, cA.θ, nest; measure = meas)
    # (20 steps: the inviscid forward-Euler anelastic core is itself
    # unstable beyond ~140 s here, one-way as well as two-way)
    for n in 1:20
        step_nested!(run; dt = 5.0)
    end
    d = abs(composite_integral(pA.θ, cA.θ, nest; measure = meas) - I0) / I0
    @printf("  anelastic two-way, 20 steps: composite ∫ρ̄θ drift %.2e\n", d)
    @test d <= 1e-12
    @test run.regs.pweights == (1.0,)
end

# #327 M1: a mixed Boussinesq / sound-proof pair has no consistent measure
# for two-way transfers (volume vs reference mass in the restriction, volume
# vs mass flux in the registers). Every two-way entry point refuses it; the
# same pair nests one-way.
@testset "mixed-core pairs: two-way refused, one-way allowed (#327 M1)" begin
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0),
                     halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    rp = isothermal_reference(pm; T_surf = 300.0); rc = isothermal_reference(cm; T_surf = 300.0)
    pbc = FieldBoundaryConditions(:periodic, :periodic, :wall)
    cbc() = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest)
    pB = BoussinesqSimulation(pm; bcs = pbc)
    pA = AnelasticSimulation(pm; ref = rp, bcs = pbc)
    cA = AnelasticSimulation(cm; ref = rc, bcs = cbc())
    cB = BoussinesqSimulation(cm; bcs = cbc())
    cP = PseudoIncompressibleSimulation(cm; ref = rc, bcs = cbc())
    for (p, c) in ((pB, cA), (pA, cB))
        for fb in (:two_way, TwoWayFeedback(), TwoWayFeedback(reflux = false))
            @test_throws ArgumentError NestedRun(p, c, nest; feedback = fb, cfl_parent = nothing)
        end
        @test_throws ArgumentError step_nested!(p, c, nest, 1.0)
        st = nest_interp_state(nest, p, c)
        @test_throws ArgumentError anterpolate!(p, c, nest, st;
                                               mask = trues(nest.nx, nest.ny, nest.nz))
        @test_throws ArgumentError restrict_state_to_parent!(p, c, nest, st)
        run = NestedRun(p, c, nest; feedback = :one_way, cfl_parent = nothing)  # one-way: fine
        @test run.feedback === :one_way
    end
    # two sound-proof systems (W = ρ̄ against W = ρ̄θ̄): refused by the measure
    @test_throws ErrorException NestedRun(pA, cP, nest; feedback = :two_way, cfl_parent = nothing)
    # a sound-proof pair with a volume-measure state
    vol = ntuple(_ -> nest_interp_state(nest), 3)
    @test_throws ArgumentError NestedRun(pA, cA, nest; feedback = :two_way, cfl_parent = nothing,
                                         state = vol)
    # the guarded pair itself is accepted
    @test NestedRun(pA, cA, nest; feedback = :two_way, cfl_parent = nothing).feedback === :two_way
end

# #327 M2: one per-face advective + diffusive kernel. The flux registers of
# the nest and boundary_scalar_flux (the open-face budgets, the Berlin heat
# ledger) used two copies of the same formula; both now call
# AdvectionSchemes.scalar_face_flux and must agree face by face with each
# other and with the formula they replaced.
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
    # the replaced formula, written out: u_n ϕ_f − (ν + s·½(K_L + K_R)) ∂ϕ/∂n
    worst_old = 0.0
    θd = c.θ.data; ud = c.u.data
    ie = cm.Hx + cm.Nx + 1
    for k in cm.Hz .+ (1:cm.Nz), j in cm.Hy .+ (1:cm.Ny)
        old = ud[ie, j, k] * 0.5 * (θd[ie, j, k] + θd[ie - 1, j, k]) -
              (c.ν + K[3] * 0.5 * (K[1].data[ie - 1, j, k] + K[1].data[ie, j, k])) *
              (θd[ie, j, k] - θd[ie - 1, j, k]) / cm.dx
        Fa, Fd, _ = scalar_face_flux(rec, θd, ud, 1, ie, j, k,
                                     face_diffusivity(c.ν, K[1], K[3], 1, ie, j, k),
                                     face_inv_spacing(cm, 1, k))
        worst_old = max(worst_old, abs(Fa + Fd - old) / max(abs(old), 1e-30))
    end
    @printf("  one kernel: registers vs boundary_scalar_flux, max relative difference over the six faces %.1e; kernel vs the replaced formula %.1e\n",
            worst_reg, worst_old)
    @test worst_reg < 1e-12
    @test worst_old < 1e-12
end

# #327 M3: the registers book the flux the transport actually applied.
@testset "separately transported tracers: WENO-Z5 slot, qv as a tracer (#327 M3)" begin
    # booked by the transport (applied_by = :transport) at its own state
    per, tot, run = _tw_slot_run(NestScalarFlux(applied_by = :transport))
    # the pre-#327 way: the hook books WENO-Z5 at the core's stage state
    # (step-start velocity), which is not the state the slot used
    per_c, tot_c, _ = _tw_slot_run(NestScalarFlux(scheme = :weno_z5))
    # not booked at all (declared for transport, but the slot never books)
    per_n, tot_n, _ = _tw_slot_run(NestScalarFlux(applied_by = :transport); book = false)
    @printf("  WENO-Z5 tracer slot, 100 two-way steps, composite ∫co2 drift per step / total (Δt = 4 s): booked by the transport %.1e / %.1e; booked by the core hook (pre-#327) %.1e / %.1e; not booked %.1e / %.1e\n",
            per, tot, per_c, tot_c, per_n, tot_n)
    @test per <= 1e-12 && tot <= 1e-11
    @test per_c > 1e3 * per                     # the stage-state booking is wrong
    @test per_n > 1e3 * per
    @test all(isfinite, run.child.u.data)
    # book_nest_flux! is a no-op where nothing is to be booked
    trc = run.s_bc.child[1]
    @test !book_nest_flux!(run.child, :co2, trc, 1.0; scheme = :weno_z5)   # no step under way
    @test !book_nest_flux!(run.child, :nothere, trc, 1.0; scheme = :weno_z5)
    g = RegularGrid(Float64; size = (4, 4, 4), extent = (1.0, 1.0, 1.0), halo = (3, 3, 3))
    @test !book_nest_flux!(BoussinesqSimulation(g), :co2, Field(CellCenter(), g), 1.0)

    # qv as a tracer of a moist sound-proof nest: declared, transported by a
    # W-weighted slot, composite ∫ρ̄ qv conserved
    pm = RegularGrid(Float64; size = (12, 12, 12), extent = (1200.0, 1200.0, 1200.0),
                     halo = (3, 3, 3))
    nest = nested_grid(pm, ((300.0, 900.0), (300.0, 900.0), (0.0, 600.0)), 2)
    cm = nest_child_mesh(nest)
    rp = isothermal_reference(pm; T_surf = 300.0); rc = isothermal_reference(cm; T_surf = 300.0)
    pA = AnelasticSimulation(pm; ref = rp, bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
    cA = AnelasticSimulation(cm; ref = rc,
                             bcs = nest_bcs(cm, :nest, :nest; bottom = :wall, top = :nest,
                                            scalars = (:qv,)))
    rng = Xoshiro(3)
    for k in axes(pA.θ.data, 3)
        pA.θ.data[:, :, k] .= rp.θ̄c_pad[k]
    end
    fill!(pA.u.data, 4.0)
    interior_view(pA.v) .+= 0.3 .* randn(rng, size(interior_view(pA.v)))
    Aeolus.Anelastic._project_mass_weighted!(pA)
    qp = Field(CellCenter(), pm); qc = Field(CellCenter(), cm)
    for k in axes(qp.data, 3), j in axes(qp.data, 2), i in axes(qp.data, 1)
        qp.data[i, j, k] = 0.01 * (1 + 0.5 * exp(-((i - 6)^2 + (j - 7)^2 + (k - 5)^2) / 5))
    end
    meas = nest_measure(nest, pA, cA)
    for (cf, pf) in ((cA.θ, pA.θ), (cA.u, pA.u), (cA.v, pA.v), (cA.w, pA.w), (qc, qp))
        interpolate_parent_to_child!(cf, pf, nest; measure = meas)
    end
    qrun(book) = NestedRun(pA, cA, nest; scalars = (:qv => (qc, qp),),
                           feedback = TwoWayFeedback(tracer_fluxes =
                               (:qv => NestScalarFlux(applied_by = :transport),)),
                           advance_parent = domain_advance(_TWSlotCore(qp, book, true)),
                           advance_child = domain_advance(_TWSlotCore(qc, book, true)),
                           cfl_parent = nothing, cfl_child = nothing)
    run = qrun(true)
    @test :qv in run.regs.names
    Q0 = composite_integral(qp, qc, nest; measure = meas)
    for _ in 1:20
        step_nested!(run; dt = 5.0)
    end
    dq = abs(composite_integral(qp, qc, nest; measure = meas) - Q0) / Q0
    @printf("  anelastic pair, qv as a W-weighted tracer slot, 20 steps: composite ∫ρ̄qv drift %.2e\n", dq)
    @test dq <= 1e-12

    # refusals: θ is not a tracer; qv of a moist Boussinesq pair is the core's
    @test_throws ArgumentError TwoWayFeedback(tracer_fluxes = (:θ => NestScalarFlux(),))
    pb, cb, nb, _, _ = _tw_box(; moist = true)
    @test_throws ArgumentError NestedRun(pb, cb, nb; cfl_parent = nothing,
        feedback = TwoWayFeedback(tracer_fluxes = (:qv => NestScalarFlux(),)))
    # cores without known stage weights: refused clearly with refluxing on
    pb, cb, nb, _, _ = _tw_box()
    for (ap, ac) in ((domain_advance(step_imex!, ARK2()), domain_advance()),
                     (domain_advance(), domain_advance((s, h; kw...) -> step!(s, h; kw...))))
        err = try
            NestedRun(pb, cb, nb; feedback = :two_way, cfl_parent = nothing,
                      advance_parent = ap, advance_child = ac)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("nest_stage_weights", err.msg)
    end
    @test NestedRun(pb, cb, nb; feedback = TwoWayFeedback(reflux = false), cfl_parent = nothing,
                    advance_parent = domain_advance(step_imex!, ARK2())).regs === nothing
    @test nest_stage_weights(step_rk3!) == (1 / 6, 1 / 6, 2 / 3)
    @test nest_stage_weights(step_imex!, ARK2()) === nothing
end

# -------------------------------------------------------------------------
# 2. Relaxation. Parent and child at rest, no diffusion: the child stays a
#    uniform θ = 1 and the covered parent cells, initially 0, relax toward
#    it, d_n = (1 − Δt/τ)ⁿ d₀. The e-folding time of the fitted decay is
#    −Δt / ln(1 − Δt/τ) = τ (1 − Δt/(2τ) + …): within 1 % of τ for
#    Δt/τ = 0.01. τ = Δt replaces in one step.
@testset "relaxation: (1 − Δt/τ)ⁿ decay, e-folding τ, τ = Δt replaces exactly" begin
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
    p, c, nest, run = rest_run(100dt)
    cx, cy, cz = nest_covered_range(nest)
    dev() = maximum(abs, interior_view(p.θ)[cx, cy, cz] .- 1)
    d = Float64[dev()]
    for n in 1:300
        step_nested!(run; dt)
        push!(d, dev())
    end
    pred = [(1 - dt / (100dt))^n for n in 0:300]
    @test maximum(abs, d .- pred) < 1e-12
    # e-folding time from a least-squares fit of ln d against t
    t = dt .* (0:300); y = log.(d)
    slope = sum((t .- sum(t) / length(t)) .* (y .- sum(y) / length(y))) /
            sum(abs2, t .- sum(t) / length(t))
    Te = -1 / slope
    @printf("  relaxation τ = %.0f s: fitted e-folding %.3f s (%.3f %% from τ); max |d − (1−Δt/τ)ⁿ| %.1e\n",
            100dt, Te, 100 * abs(Te / (100dt) - 1), maximum(abs, d .- pred))
    @test abs(Te / (100dt) - 1) < 0.01
    # outside the nest nothing happens (at rest, no fluxes)
    mask = trues(size(interior_view(p.θ))); mask[cx, cy, cz] .= false
    @test all(iszero, interior_view(p.θ)[mask])
    # τ = Δt_p (and τ = nothing, PALM's direct replacement) replace exactly
    for τ in (dt, nothing)
        p, c, nest, run = rest_run(τ)
        step_nested!(run; dt)
        @test maximum(abs, interior_view(p.θ)[cx, cy, cz] .- 1) < 1e-15
    end
end

# -------------------------------------------------------------------------
# 3. Masks. From the same state, one parent step one-way and one two-way:
#    the covered parent cells in the buffer (2 cells), below z_anterp (the
#    lowest 2 covered layers) and in the inflow band (4 cells at the west
# face) are the same as feedback off, the others are changed.
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
    @test count(act) == 2 * 4 * 4
    cx, cy, cz = nest_covered_range(nest)
    H = nest.parent_mesh.Hx
    for (name, f1, f2) in (("θ", p1.θ, p2.θ), ("qv", p1.qv, p2.qv))
        a = interior_view(f1)[cx, cy, cz]; b = interior_view(f2)[cx, cy, cz]
        @test all(a[.!act] .=== b[.!act])
        @test all(a[act] .!= b[act])
    end
    # velocity faces: a face is anterpolated only between two active cells
    for (f1, f2, ax) in ((p1.u, p2.u, 1), (p1.v, p2.v, 2), (p1.w, p2.w, 3))
        changed = 0; same_masked = true
        for K in 1:nz, J in 1:ny, I in 1:nx
            for q in (ax == 1 ? (I:I) : ax == 2 ? (J:J) : (K:K))
                i = H + nest.ix + I - 1 + (ax == 1 ? 1 : 0)
                j = H + nest.iy + J - 1 + (ax == 2 ? 1 : 0)
                k = H + nest.iz + K - 1 + (ax == 3 ? 1 : 0)
                lo = (I, J, K); hi = Base.setindex(lo, lo[ax] + 1, ax)
                on = hi[ax] <= (nx, ny, nz)[ax] && act[lo...] && act[hi...]
                if on
                    changed += f1.data[i, j, k] != f2.data[i, j, k]
                else
                    same_masked &= f1.data[i, j, k] === f2.data[i, j, k] ||
                                   hi[ax] > (nx, ny, nz)[ax]       # interface planes: flux correction
                end
            end
        end
        @test same_masked
        @test changed > 0
    end
    # the automatic inflow detection: the open faces whose child-plane mean
    # normal velocity points inward
    m = c2.mesh; ii, jj, kk = interior(m)
    um(a, idx...) = sum(view(a, idx...)) / length(view(a, idx...))
    flows = (um(c2.u.data, first(ii), jj, kk) > 0, um(c2.u.data, last(ii) + 1, jj, kk) < 0,
             um(c2.v.data, ii, first(jj), kk) > 0, um(c2.v.data, ii, last(jj) + 1, kk) < 0,
             um(c2.w.data, ii, jj, first(kk)) > 0, um(c2.w.data, ii, jj, last(kk) + 1) < 0)
    auto = feedback_mask(nest, TwoWayFeedback(; buffer = 0, exclude_width = 3), c2)
    manual = feedback_mask(nest, TwoWayFeedback(; buffer = 0, exclude_width = 3,
                                                exclude_sides = flows), c2)
    @test auto == manual
    @test flows[1] && !flows[2]                       # the mean wind blows from the west
end

# -------------------------------------------------------------------------
# 4. Idempotence: the feedback applied a second time changes nothing but
#    round-off (the registers were consumed, the anterpolated parent is
#    the child mean already, the flux-corrected interface agrees).
@testset "idempotence: feedback twice equals once" begin
    for (fb, tol) in ((TwoWayFeedback(; reproject = false), 0.0),
                      (TwoWayFeedback(; buffer = 0), 1e-13))
        p, c, nest, run, _ = _tw_box(; moist = true, feedback = fb)
        for n in 1:3
            step_nested!(run; dt = 4.0)
        end
        snap = [copy(f.data) for f in (p.u, p.v, p.w, p.θ, p.qv)]
        two_way_feedback!(run)
        dmax = maximum(maximum(abs, f.data .- s) / max(maximum(abs, s), 1)
                       for (f, s) in zip((p.u, p.v, p.w, p.θ, p.qv), snap))
        @printf("  feedback applied again (buffer %d, reproject %s): max relative change %.1e\n",
                fb.buffer, fb.reproject, dmax)
        @test dmax <= max(tol, 2e-16)
    end
end

# -------------------------------------------------------------------------
# 5. One-way against two-way. A uniform 5 m/s parent stream at θ = 300 K and
#    a heat source Q = 50 MW in a block of child cells (a source the parent
#    does not see; ΔT of the plume ≈ Q/(ρ c_p U A) ≈ 0.07 K). SSP-RK3 in both
#    domains (forward Euler with centred θ advection lets 2Δ noise at the
# child's outflow face grow to ~0.5 K by 750 s, one-way as well). One-way: the parent is the same as a standalone
#    run. Two-way: the heat reaches the parent through the feedback and the
#    refluxed interface fluxes, and the parent's θ excess outside the nest
#    is exactly Q·t/(ρc_p) minus what is still in the child.
@testset "one-way vs two-way: a child-only heat source" begin
    ρ, cp = 1.2, 1005.0
    Q = 50.0e6                                         # W
    function heat_run(feedback; nsteps = 150, dt = 5.0)
        pm = RegularGrid(Float64; size = (32, 8, 8), extent = (3200.0, 800.0, 800.0),
                         halo = (3, 3, 3))
        p = BoussinesqSimulation(pm; C_s = 0.17,
                                 bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
        fill!(p.θ.data, 300.0); fill!(p.u.data, 5.0)
        feedback === :standalone && (for n in 1:nsteps; step_rk3!(p, dt); end; return p, nothing, nothing, 0.0)
        nest = nested_grid(pm, ((800.0, 1600.0), (0.0, 800.0), (0.0, 600.0)), 2)
        cm = nest_child_mesh(nest)
        c = BoussinesqSimulation(cm; C_s = 0.17,
                                 bcs = nest_bcs(cm, :nest, :periodic; bottom = :wall, top = :nest))
        fill!(c.θ.data, 300.0); fill!(c.u.data, 5.0)
        src = (cm.Hx .+ (5:8), cm.Hy .+ (1:cm.Ny), cm.Hz .+ (2:4))   # a 200 m × 800 m × 150 m block
        Vs = length(src[1]) * length(src[2]) * length(src[3]) * cm.dx * cm.dy * cm.dz
        q = Q / (ρ * cp * Vs)                          # K/s in the block
        heat!(sim, Δt) = (view(sim.tθ.data, src...) .+= q; nothing)
        run = NestedRun(p, c, nest; feedback, cfl_parent = nothing, cfl_child = nothing,
                        advance_parent = domain_advance(step_rk3!),
                        advance_child = domain_advance(step_rk3!; body_force! = heat!))
        for n in 1:nsteps
            step_nested!(run; dt)
        end
        return p, c, nest, nsteps * dt
    end
    ps, _, _, _ = heat_run(:standalone)
    p1, c1, nest, t = heat_run(:one_way)
    @test agree(p1.θ.data, ps.θ.data) && agree(p1.u.data, ps.u.data) && agree(p1.w.data, ps.w.data)
    p2, c2, nest, t = heat_run(:two_way)
    pm = nest.parent_mesh; cm = nest.child_mesh
    cx, cy, cz = nest_covered_range(nest)
    outside = 0.0
    for k in 1:pm.Nz, j in 1:pm.Ny, i in 1:pm.Nx
        (i in cx && j in cy && k in cz) && continue
        outside += (p2.θ.data[pm.Hx+i, pm.Hy+j, pm.Hz+k] - 300.0) * pm.dx * pm.dy * pm.dz
    end
    inchild = sum(interior_view(c2.θ) .- 300.0) * cm.dx * cm.dy * cm.dz
    injected = Q * t / (ρ * cp)
    err = abs(outside - (injected - inchild)) / injected
    @printf("  heat source %.1f MW for %.0f s: injected %.4e K m³, child keeps %.4e, parent plume %.4e (%.1f %%); budget error %.1e\n",
            Q / 1e6, t, injected, inchild, outside, 100 * outside / injected, err)
    @test err < 1e-10
    @test outside > 0.3 * injected                    # the plume left the nest into the parent
    @test maximum(interior_view(p2.θ)) > 300.0 + 1e-3  # and the parent sees it
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
