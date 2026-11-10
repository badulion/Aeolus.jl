# The fused sweep in the other step cores and its per-point (GPU) form
# (#533 plan step 8):
#
#   1. `Backends.launch_points!` (one KernelAbstractions work item per
#      point) and the per-point form of the sweep,
#      `FiniteVolume.pointwise_predictor!`, on KernelAbstractions' CPU
#      backend as the stand-in for a GPU: the same composed term tuple as
#      the ring-buffer sweep (`blocked_predictor!`, 1-5 slabs), regular and
#      stretched grids, dry and moist, with and without the RK3 stage
#      combination; and `step!` / `step_rk3!` with the per-point form;
#   2. the explicit part of `step_imex!` (ARK2, Rosenbrock2) in one fused
#      tendency launch (`term_tendencies!`, rows on the CPU or per point),
#      against its separate passes (loop_fusion = "never"), dry and moist,
#      with a pipeline component's sweep term; the terms without a fused
#      form are reported;
#   3. the sound-proof cores (`step_anelastic!`,
#      `step_pseudo_incompressible!`, their `step_rk3!`): the W-weighted
#      base term and the buoyancy in the sweep against their whole-field
#      passes at 1-5 slabs, and the steps fused against loop_fusion =
#      "never", with and without viscosity and buoyancy, with a component's
#      sweep term, on the serial and threaded backends; their terms
#      without a fused form are reported and refused under "always";
#   4. type stability and allocation;
#   5. guards: every sweep term in Aeolus has the point kernel the
#      per-point form calls, and every built-in term of the sound-proof
#      step and of the IMEX explicit part is in the sweep or documented.
#
# Tolerance: the fused forms evaluate the expressions of the separate
# passes in the same order, so the agreement is round-off (`agree`,
# ROUNDOFF_RTOL = 1e-10 of the field's scale; measured: bit-identical,
# also for the per-point form).

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Grids: interior
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Components: AbstractComponent, Stage, Domain, CentredAdvection
using Aeolus.TimeIntegration: step_imex!, ARK2, Rosenbrock2
using KernelAbstractions: KernelAbstractions
using Random: Xoshiro, randn
using InteractiveUtils: subtypes

const FVc = Aeolus.Operators.FiniteVolume
const SIMc = Aeolus.Simulations
const CPc = Aeolus.Components
const ANc = Aeolus.Anelastic
const KAc = KernelAbstractions

const _FC_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)),
                      ("default", CPUBackend()))

# ---------------------------------------------------------------------------
# test terms (pointwise; radius 2 in y and z) and a component with a term
# ---------------------------------------------------------------------------

struct FCDamp{T} <: FVc.SweepTerm
    λ::T
    γ::T
end
FVc.footprint(::FCDamp) = (0, 0, 0)
@inline function FVc.add_tendency!(t::FCDamp, tb, i, j, k, st)
    @inbounds wk = st.w[i, j, k]
    tb[i, FVc.SWEEP_W] += -t.λ * wk
    tb[i, FVc.SWEEP_Θ] += t.γ * wk
    return nothing
end
function fc_damp_ref!(s, t::FCDamp)
    ii, jj, kk = interior(s.mesh)
    for k in kk, j in jj, i in ii
        s.tw.data[i, j, k] += -t.λ * s.w.data[i, j, k]
        s.tθ.data[i, j, k] += t.γ * s.w.data[i, j, k]
    end
end
struct FCHyper{T} <: FVc.SweepTerm
    c::T
end
FVc.footprint(::FCHyper) = (0, 2, 2)
@inline _fc_d4(u, i, j, k, dj, dk) = @inbounds u[i, j-2dj, k-2dk] - 4u[i, j-dj, k-dk] + 6u[i, j, k] -
                                              4u[i, j+dj, k+dk] + u[i, j+2dj, k+2dk]
@inline FVc.add_tendency!(t::FCHyper, tb, i, j, k, st) =
    (tb[i, FVc.SWEEP_U] += -t.c * (_fc_d4(st.u, i, j, k, 1, 0) + _fc_d4(st.u, i, j, k, 0, 1)); nothing)

struct FCDampComponent <: AbstractComponent
    λ::Float64
end
CPc.component_name(::Type{FCDampComponent}) = :fcdampcomponent
CPc.section(::Type{FCDampComponent}) = "FCDampComponent"
CPc.stages(::FCDampComponent) = (Stage(:body_force),)
CPc.run_stage!(c::FCDampComponent, d, ::Val{:body_force}, dt) = (fc_damp_ref!(d.state, FCDamp(c.λ, 0.0)); nothing)
FVc.sweep_term(c::FCDampComponent, d) = FCDamp(c.λ, 0.0)

function _fc_seed!(s; U = 0.3, seed = 533)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= U .* randn(rng, size(interior_view(f)))
    end
    if s isa ANc.MassWeightedSimulation
        interior_view(s.θ) .+= 0.2 .* randn(rng, size(interior_view(s.θ)))   # on top of θ̄(z)
    else
        interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
        s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    end
    return s
end
_fc_fields(s) = s isa ANc.MassWeightedSimulation ? (:u, :v, :w, :p, :θ) : (:u, :v, :w, :p, :θ, :qv)
_fc_agree(a, b) = all(f -> agree(interior_view(getfield(a, f)), interior_view(getfield(b, f))), _fc_fields(a)) &&
                  a.t ≈ b.t
# the state moved (a comparison of two unchanged states proves nothing)
_fc_moved(a, b) = !agree(interior_view(a.u), interior_view(b.u)) && !agree(interior_view(a.θ), interior_view(b.θ))

@testset "fused sweep: other step cores and the per-point form (#533 step 8)" begin

g = RegularGrid(; size = (20, 12, 12), extent = (200.0, 120.0, 120.0), halo = (3, 3, 3))
sg = StretchedGrid(; size = (12, 10, 16), extent = (240.0, 200.0, 600.0), stretching = (:tanh, 1.5))
grids = (("regular", g, FieldBoundaryConditions(:periodic)),
         ("stretched, walls", sg, FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)))
noop = (s, Δt) -> nothing

@testset "launch_points!: one work item per point, any range" begin
    a = zeros(7, 6, 5)
    body!(i, j, k, out, c) = (@inbounds out[i, j, k] = c * (i + 10j + 100k); nothing)
    Aeolus.Backends.launch_points!(KAc.CPU(), body!, (2:6, 3:5, 2:4), a, 2.0)
    ref = zeros(7, 6, 5)
    for k in 2:4, j in 3:5, i in 2:6
        ref[i, j, k] = 2.0 * (i + 10j + 100k)
    end
    @test a == ref                  # exact: integers times 2, copied
    Aeolus.Backends.launch_points!(KAc.CPU(), body!, (2:1, 3:5, 2:4), a, 0.0)   # empty: nothing
    @test a == ref
    @test Aeolus.Backends.ka_backend(CPUBackend()) isa KAc.CPU
    @test_throws ErrorException Aeolus.Backends.ka_backend(Aeolus.Backends.GPUBackend())
end

@testset "per-point form = ring-buffer sweep, any slab count" begin
    for (gname, mesh, bcs) in grids
        nk = length(interior(mesh)[3])
        for moist in (false, true), rk3 in (false, true), (bn, b) in _FC_BACKENDS[1:2]
            with_backend(b) do
                mk() = (s = _fc_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs));
                        SIMc.fill_state_halos!(s); s)
                Δt = 0.3
                terms(s) = (FVc.BaseTendencies(s.ν), FCDamp(0.05, 2e-3), FCHyper(0.01))
                c = SIMc._Lin2r(0.75, 0.25)
                ps = rk3 ? SIMc._AxpyStage(SIMc._Axpy(Δt), c) : SIMc._Axpy(Δt)
                s0 = mk()
                snap = (copy(s0.θ.data), copy(s0.qv.data))
                sc(s) = ((s.θ, ps, rk3 ? snap[1] : nothing), moist ? (s.qv, ps, rk3 ? snap[2] : nothing) : nothing)
                # the per-point form, on KA's CPU backend
                p = mk()
                tf = FVc.TendencyFields(p.tu.data, p.tv.data, p.tw.data, p.tθ.data, moist ? p.tqv.data : nothing)
                FVc.pointwise_predictor!(KAc.CPU(), tf, terms(p), p.u, p.v, p.w, SIMc._Axpy(Δt), sc(p)..., mesh)
                @test _fc_moved(p, mk())
                for ns in 1:min(5, nk ÷ 4)
                    a = mk()
                    bufs = FVc.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, nk; lag = 2, lz = 2)
                    FVc.blocked_predictor!(bufs, terms(a), a.u, a.v, a.w, SIMc._Axpy(Δt), sc(a)..., mesh)
                    ok = all(f -> agree(interior_view(getfield(a, f)), interior_view(getfield(p, f))),
                             (:u, :v, :w, :θ, :qv))
                    ok || println("  [#533 s8] per-point mismatch: $gname moist=$moist rk3=$rk3 $bn ns=$ns")
                    @test ok
                end
            end
        end
    end
    # the tendencies it wrote are those of the fused tendency launch (rows) as well
    s = _fc_seed!(BoussinesqSimulation(g; ν = 1e-2, moisture = true)); SIMc.fill_state_halos!(s)
    t1 = map(f -> fill!(similar(f.data), NaN), (s.tu, s.tv, s.tw, s.tθ, s.tqv))
    t2 = map(f -> fill!(similar(f.data), NaN), (s.tu, s.tv, s.tw, s.tθ, s.tqv))
    terms = (FVc.BaseTendencies(s.ν), FCDamp(0.05, 2e-3), FCHyper(0.01))
    FVc.term_tendencies!(FVc.TendencyFields(t1...), terms, s.u, s.v, s.w, s.θ, s.qv, g)
    FVc.term_tendencies!(FVc.TendencyFields(t2...), terms, s.u, s.v, s.w, s.θ, s.qv, g; ka = KAc.CPU())
    I = interior(g)
    @test all(n -> agree(view(t1[n], I...), view(t2[n], I...)) && !any(isnan, view(t1[n], I...)), 1:5)
    @test isnan(t1[1][1, 1, 1]) && isnan(t2[1][1, 1, 1])    # halos not written
    # a scalar without its tendency field is refused
    @test_throws ArgumentError FVc.pointwise_predictor!(KAc.CPU(), FVc.TendencyFields(t1[1], t1[2], t1[3], nothing, nothing),
                                                        terms, s.u, s.v, s.w, SIMc._Axpy(0.1), (s.θ, SIMc._Axpy(0.1), nothing),
                                                        nothing, g)
end

@testset "step! and step_rk3! in the per-point form = the ring-buffer sweep" begin
    for (gname, mesh, bcs) in grids, moist in (false, true), drv in (step!, step_rk3!)
        mk() = _fc_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs))
        a = mk(); p = mk(); p.workspaces.fused.per_point = true
        fill!(a.tu.data, NaN); fill!(p.tu.data, NaN)
        for _ in 1:2
            drv(a, 0.5); drv(p, 0.5)
        end
        @test _fc_agree(a, p) && _fc_moved(a, mk())
        @test all(isnan, a.tu.data) && !any(isnan, interior_view(p.tu))   # ring: unused; per point: written
        @test SIMc.sweep_report(p)["substages_fused"] == "all"
        @test p.workspaces.sweep === nothing                              # no ring buffer
    end
end

@testset "step_imex!: the explicit part in one fused launch" begin
    for scheme in (ARK2(), Rosenbrock2()), moist in (false, true), (bn, b) in _FC_BACKENDS, per_point in (false, true)
        (per_point && bn != "default") && continue
        with_backend(b) do
            mk() = _fc_seed!(BoussinesqSimulation(g; ν = 1e-2, moisture = moist))
            a = mk(); r = SIMc.set_loop_fusion!(mk(), :never)
            a.workspaces.fused.per_point = per_point
            SIMc.set_loop_fusion!(a, :always)
            for _ in 1:2
                step_imex!(a, 0.5, scheme); step_imex!(r, 0.5, scheme)
            end
            ok = _fc_agree(a, r) && _fc_moved(a, mk())
            ok || println("  [#533 s8] IMEX mismatch: $(nameof(typeof(scheme))) moist=$moist $bn per_point=$per_point")
            @test ok
            @test a.workspaces.fused.fused == 4 && a.workspaces.fused.unfused == 0      # 2 stages × 2 steps
            @test SIMc.sweep_report(a)["substages_fused"] == "all" && SIMc.sweep_report(r)["substages_fused"] == "none"
        end
    end
    # a pipeline component's sweep term in the explicit part (the hook passed to step_imex!)
    mk() = (s = _fc_seed!(BoussinesqSimulation(g; ν = 1e-2));
            Domain(Domain(s; validate = false); components = (CentredAdvection(), FCDampComponent(0.05))))
    a = mk(); r = mk()
    SIMc.set_loop_fusion!(r.state, :never)
    for _ in 1:2
        step_imex!(a.state, 0.5, ARK2(); body_force! = CPc.core_hooks(a).body_force!)
        step_imex!(r.state, 0.5, ARK2(); body_force! = CPc.core_hooks(r).body_force!)
    end
    z = mk()
    for _ in 1:2
        step_imex!(z.state, 0.5, ARK2())
    end
    @test _fc_agree(a.state, r.state) && !agree(interior_view(a.state.w), interior_view(z.state.w))
    @test SIMc.sweep_report(a.state)["substages_fused"] == "all"
    # the built-in pointwise terms (#533 step 2) in the explicit part: fused, and
    # equal to one loop per term. (Before steps 2 and 8 were combined, buoyancy
    # and Coriolis were this test's examples of terms without a fused form;
    # that expectation was made outdated by the integration, not weakened.)
    for kw in ((θ₀ = 300.0,), (f_coriolis = 1e-4,), (θ₀ = 300.0, moisture = true), (θ_flux_bottom = 0.05,),
               (sponge_α_max = 0.01, sponge_z_start = 0.5 * g.Lz))
        mk2(setting) = SIMc.set_loop_fusion!(_fc_seed!(BoussinesqSimulation(g; ν = 1e-2, kw...)), setting)
        a2 = mk2(:always); r2 = mk2(:never)
        for _ in 1:2
            step_imex!(a2, 0.5, ARK2()); step_imex!(r2, 0.5, ARK2())
        end
        ok = _fc_agree(a2, r2)
        ok || println("  [#533 s2×s8] IMEX explicit part with $kw: fused ≠ one loop per term")
        @test ok && SIMc.sweep_report(a2)["substages_fused"] == "all"
    end
    # a term without a fused form: reported; refused under "always"
    c = SIMc.set_loop_fusion!(_fc_seed!(BoussinesqSimulation(g; ν = 1e-2)), :always)
    err = try step_imex!(c, 0.5, ARK2(); body_force! = noop); nothing catch e; e end
    @test err isa ArgumentError && occursin("builtin.body_force", err.msg)
    s = _fc_seed!(BoussinesqSimulation(g; ν = 1e-2, C_s = 0.17))
    step_imex!(s, 0.5, ARK2(); sgs_stress = :symmetric)          # the explicit transposed stress
    @test any(startswith("builtin.imex_sgs:"), SIMc.sweep_report(s)["not_fusable"])
    s = _fc_seed!(BoussinesqSimulation(g; ν = 1e-2, C_s = 0.17))
    step_imex!(s, 0.5, ARK2())                                   # the implicit Laplacian SGS: explicit part fused
    @test SIMc.sweep_report(s)["substages_fused"] == "all"
    s = _fc_seed!(BoussinesqSimulation(g; ν = 1e-2))
    step_imex!(s, 0.5, ARK2(); body_force! = noop)
    @test SIMc.sweep_report(s)["not_fusable"] == ["builtin.body_force: " * SIMc.sweep_reason("builtin.body_force")]
end

# the sound-proof cores
ag = RegularGrid(; size = (16, 12, 16), extent = (1600.0, 1200.0, 1600.0), halo = (3, 3, 3))
mkan(; per_point = false, S = ANc.AnelasticSimulation, kw...) =
    (s = _fc_seed!(S(ag; kw...); U = 0.5); s.workspaces.fused.per_point = per_point; s)

@testset "sound-proof base term and buoyancy in the sweep = their passes, any slab count" begin
    for (S, ν, buoy) in ((ANc.AnelasticSimulation, 1.0, true), (ANc.AnelasticSimulation, 0.0, true),
                         (ANc.PseudoIncompressibleSimulation, 1.0, false)), (bn, b) in _FC_BACKENDS[1:2]
        with_backend(b) do
            mk() = (s = mkan(; S = S, ν = ν, buoyancy = buoy);
                    Aeolus.BoundaryConditions.fill_halos!((s.u, s.v, s.w, s.θ), (s.bcs, s.bcs, s.bcs, s.θ_bcs)); s)
            Δt = 2.0
            # the whole-field reference: the weighted advection, ν∇², the buoyancy, the predictor
            r = mk()
            Wc, Wf = ANc.constraint_weights(r)
            foreach(f -> fill!(f.data, 0), (r.tu, r.tv, r.tw, r.tθ))
            ANc.weighted_momentum_advect!(r.tu, r.tv, r.tw, r.u, r.v, r.w, Wc, Wf, ag)
            ANc.weighted_scalar_advect!(r.tθ, r.u, r.v, r.w, r.θ, Wc, Wf, ag)
            if r.has_viscosity
                for (t, f, l) in ((r.tu, r.u, r.lap_u), (r.tv, r.v, r.lap_v), (r.tw, r.w, r.lap_w), (r.tθ, r.θ, r.lap_θ))
                    Aeolus.Operators.laplacian(l, f); SIMc.axpy!(t, r.ν, l)
                end
            end
            buoy && ANc.add_anelastic_buoyancy!(r.tw, r.θ, r.ref, ag)
            for (f, t) in ((r.u, r.tu), (r.v, r.tv), (r.w, r.tw), (r.θ, r.tθ))
                SIMc.axpy!(f, Δt, t)
            end
            @test _fc_moved(r, mk())
            terms(s) = (base = ANc.WeightedBaseTendencies(s.has_viscosity ? s.ν : nothing, Wc, Wf,
                                                          1 / ag.dx, 1 / ag.dy, 1 / ag.dz);
                        buoy ? (base, ANc.AnelasticBuoyancy(s.ref.θ̄f_pad, s.ref.g)) : (base,))
            ax = SIMc._Axpy(Δt)
            nk = length(interior(ag)[3])
            for ns in 1:5
                a = mk()
                bufs = FVc.sweep_buffers(nothing, a.u.data, 4, ns, nk)
                FVc.blocked_predictor!(bufs, terms(a), a.u, a.v, a.w, ax, (a.θ, ax, nothing), nothing, ag)
                ok = all(f -> agree(interior_view(getfield(a, f)), interior_view(getfield(r, f))), (:u, :v, :w, :θ))
                ok || println("  [#533 s8] sound-proof sweep mismatch: $(nameof(S)) ν=$ν $bn ns=$ns")
                @test ok
            end
            p = mk()
            tf = FVc.TendencyFields(p.tu.data, p.tv.data, p.tw.data, p.tθ.data, nothing)
            FVc.pointwise_predictor!(KAc.CPU(), tf, terms(p), p.u, p.v, p.w, ax, (p.θ, ax, nothing), nothing, ag)
            @test all(f -> agree(interior_view(getfield(p, f)), interior_view(getfield(r, f))), (:u, :v, :w, :θ))
        end
    end
end

@testset "step_anelastic!, step_pseudo_incompressible!, step_rk3!: fused = separate loops" begin
    for (S, drv) in ((ANc.AnelasticSimulation, ANc.step_anelastic!), (ANc.AnelasticSimulation, step_rk3!),
                     (ANc.PseudoIncompressibleSimulation, ANc.step_pseudo_incompressible!)),
        (ν, buoy) in ((1.0, true), (0.0, true), (1.0, false)), (bn, b) in _FC_BACKENDS, per_point in (false, true)
        (per_point && bn != "default") && continue
        with_backend(b) do
            a = SIMc.set_loop_fusion!(mkan(; S = S, ν = ν, buoyancy = buoy, per_point = per_point), :always)
            r = SIMc.set_loop_fusion!(mkan(; S = S, ν = ν, buoyancy = buoy), :never)
            fill!(a.tu.data, NaN)
            for _ in 1:3
                drv(a, 2.0); drv(r, 2.0)
            end
            ok = _fc_agree(a, r) && _fc_moved(a, mkan(; S = S))
            ok || println("  [#533 s8] sound-proof step mismatch: $(nameof(S)) $drv ν=$ν buoyancy=$buoy $bn per_point=$per_point")
            @test ok
            @test per_point ? !any(isnan, interior_view(a.tu)) : all(isnan, a.tu.data)
            nsub = drv === step_rk3! ? 9 : 3
            @test a.workspaces.fused.fused == nsub && SIMc.sweep_report(a)["substages_fused"] == "all"
            @test r.workspaces.fused.unfused == nsub && SIMc.sweep_report(r)["substages_fused"] == "none"
            # the constraint ∇·(W u) = 0 holds after the fused Euler step, to the projection's round-off
            # (not after step_rk3!: its combination with the seeded, unprojected uⁿ is not constrained)
            if drv !== step_rk3!
                div = S === ANc.AnelasticSimulation ? ANc.max_anelastic_divergence(a) :
                                                      ANc.max_pseudo_incompressible_divergence(a)
                @test div < 1e-12          # measured ~1e-17 s⁻¹ (velocities 0.5 m/s over 100 m cells)
            end
        end
    end
    # a component's sweep term through the pipeline (an anelastic Domain), Euler and RK3
    for scheme in (:euler, :rk3)
        mk() = (s = mkan(; ν = 1.0);
                Domain(Domain(s; scheme = scheme, validate = false); components = (CentredAdvection(), FCDampComponent(0.01))))
        a = mk(); r = mk(); SIMc.set_loop_fusion!(r.state, :never)
        for _ in 1:3
            step!(a, 2.0); step!(r, 2.0)
        end
        @test _fc_agree(a.state, r.state)
        @test CPc.sweep_report(a)["substages_fused"] == "all"
    end
end

# a damping of w that also heats θ by γ w: a component whose tendency the energy budget books
struct FCHeatComponent <: AbstractComponent end
CPc.component_name(::Type{FCHeatComponent}) = :fcheatcomponent
CPc.section(::Type{FCHeatComponent}) = "FCHeatComponent"
CPc.stages(::FCHeatComponent) = (Stage(:body_force),)
CPc.run_stage!(c::FCHeatComponent, d, ::Val{:body_force}, dt) = (fc_damp_ref!(d.state, FCDamp(0.01, 1e-3)); nothing)
FVc.sweep_term(c::FCHeatComponent, d) = FCDamp(0.01, 1e-3)

@testset "budgets with the other step cores and the per-point form (integration of #533 steps 7 and 8)" begin
    # The fused sound-proof core books the in-core component through the budget term's
    # finish_sweep! as the hook does (bookings within 1e-10 of the largest term: the same
    # expressions, summed per row instead of per field); the per-point form has no per-level sums,
    # so with budgets on it keeps one loop per term and names builtin.budgets_per_point.
    for scheme in (:euler, :rk3)
        mk(; pp = false) = (s = mkan(; ν = 1.0, per_point = pp);
                            CPc.with_budgets(Domain(Domain(s; scheme = scheme, validate = false);
                                                    components = (CentredAdvection(), FCHeatComponent())),
                                             (:energy, :momentum_x)))
        a = mk(); r = mk(); SIMc.set_loop_fusion!(r.state, :never)
        for _ in 1:3
            step!(CPc.Model(a), 2.0); step!(CPc.Model(r), 2.0)
        end
        @test _fc_agree(a.state, r.state)
        @test CPc.sweep_report(a)["substages_fused"] == "all"
        for (q, sb) in CPc.budgets(a).measured
            A = Aeolus.BudgetSystem.stage_budget(sb); R = Aeolus.BudgetSystem.stage_budget(CPc.budgets(r).measured[q])
            @test Set(keys(A.terms)) == Set(keys(R.terms))
            big = maximum(abs, values(R.terms))
            @test big > 0 && all(k -> abs(A.terms[k] - R.terms[k]) <= 1e-10 * big, keys(R.terms))
            q === :energy && @test haskey(A.terms, (:body_force, :fcheatcomponent))
        end
        p = mk(; pp = true)
        step!(CPc.Model(p), 2.0)
        rep = CPc.sweep_report(p)
        @test rep["substages_fused"] == "none" && any(startswith("builtin.budgets_per_point:"), rep["not_fusable"])
        pd = mk(; pp = true); SIMc.set_loop_fusion!(pd.state, :always)
        err = try step!(CPc.Model(pd), 2.0); nothing catch e; e end
        @test err isa ArgumentError && occursin("builtin.budgets_per_point", err.msg)
    end
end

@testset "sound-proof cores: terms without a fused form are reported" begin
    for (kw, hook, key) in (((f_coriolis = 1e-4,), NamedTuple(), "builtin.anelastic_coriolis"),
                            ((C_s = 0.17,), NamedTuple(), "builtin.anelastic_sgs"),
                            ((θ_flux_bottom = 0.1,), NamedTuple(), "builtin.anelastic_surface_flux"),
                            ((sponge_α_max = 0.01, sponge_z_start = 1000.0), NamedTuple(), "builtin.anelastic_sponge"),
                            (NamedTuple(), (post_predictor! = noop,), "builtin.post_predictor"),
                            (NamedTuple(), (post_predictor! = noop,), "builtin.anelastic_post_predictor"),
                            (NamedTuple(), (body_force! = noop,), "builtin.body_force"))
        a = mkan(; ν = 1.0, kw...)
        fill!(a.tu.data, NaN)
        ANc.step_anelastic!(a, 2.0; hook...)
        @test !any(isnan, interior_view(a.tu))
        rep = SIMc.sweep_report(a)
        @test rep["substages_fused"] == "none" && any(startswith(key * ":"), rep["not_fusable"])
        @test !occursin("no documented reason", join(rep["not_fusable"]))
        c = SIMc.set_loop_fusion!(mkan(; ν = 1.0, kw...), :always)
        err = try ANc.step_anelastic!(c, 2.0; hook...); nothing catch e; e end
        @test err isa ArgumentError && occursin(key, err.msg)
        @test c.t == 0.0                                    # exact: the clock did not advance
    end
    # nothing stepped yet: no substage, nothing to report
    @test SIMc.sweep_report(mkan())["substages_fused"] == "none" && isempty(SIMc.sweep_report(mkan())["not_fusable"])
end

@testset "type stability and allocation" begin
    a = mkan(; ν = 1.0)
    Wc, Wf = ANc.constraint_weights(a)
    base = ANc.WeightedBaseTendencies(a.ν, Wc, Wf, 1 / ag.dx, 1 / ag.dy, 1 / ag.dz)
    ex = (ANc.AnelasticBuoyancy(a.ref.θ̄f_pad, a.ref.g),)
    @test (@inferred Union{Nothing,typeof((base, ex...))} ANc._mw_sweep_terms(a, nothing, nothing, nothing, base, ex)) ==
          (base, ex...)
    @test (@inferred FVc.sweep_lags((base, ex...))) == (1, 1)
    for drv in (ANc.step_anelastic!, step_rk3!)
        drv(a, 1.0); drv(a, 1.0)
        n = @allocated drv(a, 1.0)
        r = SIMc.set_loop_fusion!(mkan(; ν = 1.0), :never)
        drv(r, 1.0); drv(r, 1.0)
        nr = @allocated drv(r, 1.0)
        println("  [#533 s8] $drv allocates $n bytes fused, $nr bytes with one loop per term")
        @test n <= nr + 64      # the fused substages add no allocation (the step's own: the solver, the record)
    end
    s = _fc_seed!(BoussinesqSimulation(g; ν = 1e-2))
    step_imex!(s, 0.1, ARK2()); step_imex!(s, 0.1, ARK2())
    SIMc.fill_state_halos!(s)
    terms = (FVc.BaseTendencies(nothing),)
    tf = FVc.TendencyFields(s.tu.data, s.tv.data, s.tw.data, s.tθ.data, nothing)
    f() = FVc.term_tendencies!(tf, terms, s.u, s.v, s.w, s.θ, nothing, g)
    f()
    n = @allocated f()
    println("  [#533 s8] term_tendencies! (rows) allocates $n bytes")
    @test n <= 64           # the launch's task closure on the threaded backend, none per point
end

# Review of step 8: the step drivers, fused and unfused under "if_possible",
# allocate no more than with one loop per term ("never"). Two bugs this
# caught: the unfused report's blocker closure was boxed on every unfused
# substage (+16 B per step!), and the fused IMEX launch boxed its Union-typed
# arguments (+176 B per stage). Measured after the fix: equal bytes.
@testset "no allocation added by the fused or the unfused report, any step core" begin
    drivers = (("step!", (s; kw...) -> step!(s, 0.1; kw...)), ("step_rk3!", (s; kw...) -> step_rk3!(s, 0.1; kw...)),
               ("step_imex! ARK2", (s; kw...) -> step_imex!(s, 0.1, ARK2(); kw...)),
               ("step_imex! Rosenbrock2", (s; kw...) -> step_imex!(s, 0.1, Rosenbrock2(); kw...)))
    # (Coriolis was the unfused example until steps 2 and 8 were combined: it
    # now has a fused form in every Boussinesq driver, so it is a fusable
    # case here, and a body_force! function is the unfused one)
    for (lab, kw, skw, fused) in (("fusable", (;), (;), "all"),
                                  ("fusable (Coriolis, buoyancy)", (; f_coriolis = 1e-4, θ₀ = 300.0), (;), "all"),
                                  ("unfused (a body_force! function)", (;), (; body_force! = noop), "none")),
        moist in (false, true), (dn, drv) in drivers
        mk(setting) = SIMc.set_loop_fusion!(_fc_seed!(BoussinesqSimulation(g; ν = 1e-2, moisture = moist, kw...)), setting)
        a = mk(:if_possible); r = mk(:never)
        bytes(s) = (drv(s; skw...); drv(s; skw...); @allocated drv(s; skw...))
        n, nr = bytes(a), bytes(r)
        n <= nr || println("  [#533 s8] $dn $lab moist=$moist allocates $n bytes, $nr with one loop per term")
        @test n <= nr
        @test SIMc.sweep_report(a)["substages_fused"] == fused
    end
    # the sound-proof cores, unfused (Coriolis has no fused form there)
    for drv in (ANc.step_anelastic!, step_rk3!)
        a = mkan(; ν = 1.0, f_coriolis = 1e-4)
        r = SIMc.set_loop_fusion!(mkan(; ν = 1.0, f_coriolis = 1e-4), :never)
        bytes(s) = (drv(s, 1.0); drv(s, 1.0); @allocated drv(s, 1.0))
        n, nr = bytes(a), bytes(r)
        @test n <= nr
        @test SIMc.sweep_report(a)["substages_fused"] == "none"
    end
end

@testset "guards: point kernels, built-in terms of the other cores" begin
    # every sweep term defined in Aeolus has the point kernel the per-point
    # (GPU) form calls; every base term both of its initialising kernels
    leaves(T) = (s = subtypes(T); isempty(s) ? [T] : reduce(vcat, map(leaves, s)))
    terms = filter(T -> Base.moduleroot(parentmodule(T)) === Aeolus, leaves(FVc.SweepTerm))
    @test FVc.BaseTendencies in terms && ANc.WeightedBaseTendencies in terms && ANc.AnelasticBuoyancy in terms
    for T in terms
        if T <: FVc.BaseTerm
            sig = Tuple{T,Any,Int,Int,Int,Int,Int,UnitRange{Int},Any}
            ok = hasmethod(FVc.init_point!, Tuple{T,Any,Int,Int,Int,Any}) &&
                 Base.unwrap_unionall(which(FVc.init_row!, sig).sig).parameters[2] !== FVc.SweepTerm
        else
            ok = hasmethod(FVc.add_tendency!, Tuple{T,Any,Int,Int,Int,Any})
        end
        ok || println("  [#533 s8] the sweep term $T has no point kernel (the per-point form needs add_tendency!)")
        @test ok
    end
    # the built-in terms of the sound-proof step: every term
    # `_mass_weighted_step!` adds is in the sweep or has a documented reason
    src = read(joinpath(pkgdir(Aeolus), "src", "Anelastic.jl"), String)
    a = findfirst("function _mass_weighted_step!", src); b = findnext("\nend\n", src, last(a))
    body = src[first(a):last(b)]
    builtin = Dict(
        "has_viscosity" => :base, "weighted_momentum_advect!" => :base, "weighted_scalar_advect!" => :base,
        "laplacian" => :base, "zero_storage!" => :base, "axpy!" => :predictor,
        "has_buoyancy" => :sweep, "add_anelastic_buoyancy!" => :sweep,
        "compute_sgs!" => "builtin.anelastic_sgs", "add_sgs_momentum!" => "builtin.anelastic_sgs",
        "add_sgs_scalar!" => "builtin.anelastic_sgs",
        "has_coriolis" => "builtin.anelastic_coriolis", "add_coriolis!" => "builtin.anelastic_coriolis",
        "has_surface_flux" => "builtin.anelastic_surface_flux", "add_surface_flux!" => "builtin.anelastic_surface_flux",
        "has_sponge" => "builtin.anelastic_sponge", "add_velocity_sponge!" => "builtin.anelastic_sponge",
        "body_force!" => "builtin.body_force", "post_predictor!" => "builtin.post_predictor",
        "update_open_boundaries!" => "builtin.anelastic_open_boundaries")
    rx = r"\bsim\.(has_\w+)|\b(add_\w+!|compute_sgs!|weighted_\w+!|laplacian|axpy!|zero_storage!)\(|\b(body_force!|post_predictor!)\("
    for m in eachmatch(rx, body)
        name = something(m.captures...)
        haskey(builtin, name) || println("  [#533 s8] a built-in term of _mass_weighted_step! not classified: $name")
        @test haskey(builtin, name)
        v = get(builtin, name, nothing)
        v isa String && @test haskey(SIMc.SWEEP_FALLBACKS, v)
    end
    # the explicit part of step_imex!: its terms are those of the Boussinesq core
    src = read(joinpath(pkgdir(Aeolus), "src", "TimeIntegration.jl"), String)
    a = findfirst("function _explicit_tendencies!", src); b = findnext("\nend\n", src, last(a))
    body = src[first(a):last(b)]
    builtin = Dict(
        "has_moisture" => :base, "advect_centered!" => :base, "momentum_advection" => "builtin.momentum_advection",
        "add_sgs_stress_divergence!" => "builtin.imex_sgs", "add_tensor_diffusion_scalar!" => "builtin.imex_sgs",
        # the pointwise terms of #533 step 2, in the sweep (`_builtin_terms`)
        "has_buoyancy" => :sweep, "add_buoyancy!" => :sweep, "has_coriolis" => :sweep, "add_coriolis!" => :sweep,
        "has_surface_flux" => :sweep, "add_surface_flux!" => :sweep, "has_sponge" => :sweep,
        "add_velocity_sponge!" => :sweep,
        "body_force!" => "builtin.body_force")
    rx = r"\bsim\.(has_\w+)|\b(add_\w+!|advect_centered!)\(|\b(body_force!|momentum_advection)\("
    for m in eachmatch(rx, body)
        name = something(m.captures...)
        haskey(builtin, name) || println("  [#533 s8] a term of the IMEX explicit part not classified: $name")
        @test haskey(builtin, name)
    end
    # (the other keys are the Boussinesq core's: `_sweep_terms` decides for both)
    @test haskey(SIMc.SWEEP_FALLBACKS, "builtin.imex_sgs")
end

end
