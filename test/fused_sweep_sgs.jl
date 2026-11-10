# The SGS closures inside the fused sweep (#533 plan step 4): the
# coefficients computed before the sweep (`compute_sgs!` in the term's
# `prepare!`), the momentum, θ and qv tendencies inside it
# (`Operators.SGSTerm`).
#
#   1. the sweep over (base, SGS term) = the separate loops (the SC-d base
#      tendencies, compute_sgs!, add_sgs_momentum!, add_sgs_scalar!,
#      _add_sgs_tracer!) followed by the predictor, for every closure of the
#      core, both sgs_stress forms, 1-5 slabs (forward and, for the front
#      edge, reverse slab order), the serial and the threaded backends,
#      Euler and the RK3 combination, dry and moist, periodic and walls,
#      regular and (for the closures that run there) stretched grids;
#   2. step! and step_rk3! with loop_fusion = "always" agree with "never"
#      (the reference) for every closure, and report every substage fused;
#   3. a closure the sweep cannot run is reported by its type, and the
#      stress form on a stretched grid keeps the reference's error;
#   4. footprint, type stability and allocation;
#   5. the guard: every closure of Aeolus that adds its own momentum,
#      scalar or tracer tendency adds the matching sweep form;
#   6. at run time, a closure from any module that adds its own tendency
#      without the matching sweep form (also inside SolidMaskedSGS) stays
#      out of the sweep, reported by its type, and the Reynolds stress has
#      no sweep form on a stretched grid (`sgs_sweep_form_matches`).
#
# Tolerance: the sweep evaluates the separate loops' point bodies, in the
# same order, so the agreement is round-off (`agree`, ROUNDOFF_RTOL of
# the field's scale; measured: bit-identical).

using Test
using Aeolus
using Aeolus.Fields: Field, interior_view
using Aeolus.Grids: interior, CellCenter
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Random: Xoshiro, randn

const FVg = Aeolus.Operators.FiniteVolume
const OPg = Aeolus.Operators
const SIMg = Aeolus.Simulations

function _sg_seed!(s; seed = 5334)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f)))
    end
    nz = size(interior_view(s.θ), 3)
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ))) .+
                          reshape(0.02 .* (1:nz), 1, 1, nz)          # a stratified θ: a tilted tensor
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    return s
end
_sg_fields(s) = (s.u, s.v, s.w, s.θ, s.qv)
_sg_agree(a, b) = all(i -> agree(interior_view(_sg_fields(a)[i]), interior_view(_sg_fields(b)[i])), 1:5)

# the separate loops of step 4 of `_euler_projection!` after the base
# tendencies, then the predictor
function _sg_reference!(r, c, stress, Δt, rk3, snap)
    g = r.mesh
    SIMg.fill_state_halos!(r)
    moist = r.has_moisture
    FVg._fused_tendencies!(r.tu, r.tv, r.tw, r.u, r.v, r.w, r.ν, (r.tθ, r.θ),
                           moist ? (r.tqv, r.qv) : nothing, g)
    f = OPg.compute_sgs!(c, r)
    OPg.add_sgs_momentum!(r.tu, r.tv, r.tw, c, f, r; stress = stress)
    OPg.add_sgs_scalar!(r.tθ, r.θ, c, f, r)
    moist && SIMg._add_sgs_tracer!(r.tqv, r.qv, c, f, r, r.lap_θ)
    for (x, t) in ((r.u, r.tu), (r.v, r.tv), (r.w, r.tw))
        SIMg.axpy!(x, Δt, t)
    end
    cc = SIMg._Lin2r(0.75, 0.25)
    for (x, t, s0) in (moist ? ((r.θ, r.tθ, snap[1]), (r.qv, r.tqv, snap[2])) : ((r.θ, r.tθ, snap[1]),))
        rk3 ? SIMg._advance_scalar!(x, Δt, t, (c = cc, ws = nothing), s0) : SIMg.axpy!(x, Δt, t)
    end
    return r
end

# the sweep over (base, SGS term), its slabs in launch order or (`reverse`)
# one after another from the last
function _sg_sweep!(a, c, stress, Δt, rk3, snap, ns; reverse = false)
    g = a.mesh
    SIMg.fill_state_halos!(a)
    moist = a.has_moisture
    ii, jj, kk = interior(g)
    term = OPg.sgs_sweep_term(c, a, stress, a.lap_θ)
    terms = (FVg.BaseTendencies(a.ν), term)
    lag, lz = FVg.sweep_lags(terms)
    bufs = FVg.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, length(kk); lag = lag, lz = lz)
    foreach(t -> FVg.prepare!(t, a, Δt), terms)
    cc = SIMg._Lin2r(0.75, 0.25)
    ps = rk3 ? SIMg._AxpyStage(SIMg._Axpy(Δt), cc) : SIMg._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    pu = SIMg._Axpy(Δt)
    if !reverse
        FVg.blocked_predictor!(bufs, terms, a.u, a.v, a.w, pu, s1, s2, g)
    else
        st = (u = a.u.data, v = a.v.data, w = a.w.data, θ = FVg._sweep_field(s1), qv = FVg._sweep_field(s2),
              m = FVg.grid_metrics(g))
        p1, p2 = FVg._sweep_pred(s1), FVg._sweep_pred(s2)
        for s in ns:-1:1
            FVg._b_sweep_slab!(s, terms, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
        end
        for n in 1:2lz*ns, j in jj, i in ii
            FVg._b_sweep_edge!(i, j, n, st, pu, p1, p2, bufs.edge, kk, ns, lz)
        end
    end
    return a
end

# a closure with its own (explicit) stress and no sweep form: the sweep
# must not run it, and reports it by its type
struct SGTestStress <: OPg.AbstractSGSClosure end
OPg.sgs_coefficients!(f::OPg.SGSFields, ::SGTestStress, sim) = (fill!(f.νh.data, 0.01); fill!(f.κh.data, 0.01); f)
OPg.sgs_explicit_stress(::SGTestStress) = true
OPg.add_sgs_momentum!(tu, tv, tw, ::SGTestStress, f::OPg.SGSFields, sim; stress::Symbol = :laplacian) =
    (OPg.add_eddy_viscosity_u!(tu, sim.u, f.νh); nothing)     # (only u: its own, odd stress)

# closures from outside Aeolus that override a tendency the sweep would
# otherwise take from the contract defaults (the review of #533 step 4:
# like validation/fixtures/sgs_validation.jl's BackgroundGradient, which
# adds a background θ flux in its own add_sgs_scalar!)
abstract type SGTestBase <: OPg.AbstractSGSClosure end
function OPg.sgs_coefficients!(f::OPg.SGSFields, ::SGTestBase, sim)
    nz = size(f.νh.data, 3)
    f.νh.data .= 0.01 .+ 0.001 .* reshape(1:nz, 1, 1, nz)     # varies in z: a background flux would show
    f.κh.data .= f.νh.data
    return f
end
struct SGTestScalar <: SGTestBase end            # θ: its own flux, no sweep form
function OPg.add_sgs_scalar!(tϕ, ϕ, c::SGTestScalar, f::OPg.SGSFields, sim)
    OPg.add_eddy_diffusion_scalar!(tϕ, ϕ, f.κh)
    interior_view(tϕ) .+= 1e-3                   # a background flux divergence the default does not add
    return nothing
end
struct SGTestMom <: SGTestBase end               # momentum: its own (eddy-viscosity, doubled) form
OPg.add_sgs_momentum!(tu, tv, tw, ::SGTestMom, f::OPg.SGSFields, sim; stress::Symbol = :laplacian) =
    foreach(_ -> OPg.add_sgs_momentum!(tu, tv, tw, StaticSmagorinsky(0.17), f, sim; stress), 1:2)
struct SGTestTracer <: SGTestBase end            # qv: its own diffusivity, no trait
OPg.sgs_tracer_diffusivity!(κ, ::SGTestTracer, f::OPg.SGSFields, sim, ϕ) = (κ.data .= 2 .* f.κh.data; κ)
struct SGTestFace <: SGTestBase end              # θ: its own flux and the matching face form
OPg.add_sgs_scalar!(tϕ, ϕ, ::SGTestFace, f::OPg.SGSFields, sim) = OPg.add_eddy_diffusion_scalar!(tϕ, ϕ, f.κh, 2.0)
OPg.sgs_scalar_face_diffusivity(::SGTestFace, f::OPg.SGSFields, sim) = (f.κh, f.κh, 0.5)

@testset "fused sweep: the SGS closures (#533 step 4)" begin

g = RegularGrid(; size = (16, 12, 12), extent = (160.0, 120.0, 120.0), halo = (3, 3, 3))
sg = StretchedGrid(; size = (12, 10, 12), extent = (240.0, 200.0, 600.0), stretching = (:tanh, 1.5))
periodic = FieldBoundaryConditions(:periodic)
walls = FieldBoundaryConditions(:periodic, :periodic, :wall)
swalls = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)
cs = Field(CellCenter(), g)
mask = falses(size(cs.data))
mask[g.Hx+4:g.Hx+7, g.Hy+3:g.Hy+6, g.Hz+1:g.Hz+5] .= true       # a block on the floor
smag(νt, u, v, w) = OPg.compute_smagorinsky_νt!(νt, u, v, w, 0.12)

# (name, closure maker (sim -> closure), grids, momentum forms): every
# closure the Boussinesq core runs (src/operators, PrognosticSGS, ML, the
# urban site's mask)
closures = (
    ("Smagorinsky",       s -> StaticSmagorinsky(0.17), (:regular, :stretched), (:laplacian, :symmetric)),
    ("dynamic (plane)",   s -> DynamicSmagorinsky(g), (:regular,), (:laplacian, :symmetric)),
    ("Lagrangian dynamic", s -> LagrangianDynamicSmagorinsky(g), (:regular,), (:laplacian,)),
    ("Vreman",            s -> VremanClosure(), (:regular,), (:laplacian,)),
    ("sigma",             s -> SigmaModel(), (:regular,), (:laplacian,)),
    ("function",          s -> OPg.FunctionSGS(smag), (:regular,), (:laplacian,)),
    ("ML",                s -> Aeolus.ML.MLSGS((νt, u, v, w, m) -> smag(νt, u, v, w)), (:regular,), (:laplacian,)),
    ("AMD",               s -> AMDClosure(), (:regular,), (:laplacian, :symmetric)),
    ("TKE",               s -> TKEClosure(s.mesh; e0 = 1e-3), (:regular, :stretched), (:laplacian, :symmetric)),
    ("TKE, solid mask",   s -> TKEClosure(g; e0 = 1e-3, mask = mask), (:regular,), (:laplacian,)),
    ("stability-corrected", s -> StabilityCorrected(StaticSmagorinsky(0.17), g), (:regular,), (:laplacian, :symmetric)),
    ("isentropic tensor", s -> IsentropicDiffusion(StabilityCorrected(StaticSmagorinsky(0.17), g), g;
                                                   diapycnal = 0.1), (:regular,), (:laplacian, :symmetric)),
    # (three committed steps: an isotropic initial stress has no divergence)
    ("Reynolds stress",   s -> (c = ReynoldsStressClosure(g); initialize_reynolds_stress!(c, s; e_min = 1e-3);
                                SIMg.fill_state_halos!(s); foreach(_ -> OPg.advance_sgs!(c, s, 0.5), 1:3); c),
                          (:regular,), (:laplacian,)),
    ("solid-masked AMD",  s -> Aeolus.Urban.SolidMaskedSGS(AMDClosure(), mask), (:regular,), (:laplacian, :symmetric)),
    ("solid-masked Smagorinsky", s -> Aeolus.Urban.SolidMaskedSGS(StaticSmagorinsky(0.17), mask), (:regular,),
                          (:laplacian,)),
)
meshes = Dict(:regular => ((g, periodic), (g, walls)), :stretched => ((sg, swalls),))

@testset "the sweep = the separate loops + predictor: $name" for (name, mkc, gk, forms) in closures
    for gkey in gk, (mesh, bcs) in meshes[gkey], stress in forms, moist in (false, true)
        (gkey === :stretched && stress === :symmetric) && continue   # no stretched stress form (tested below)
        mk() = _sg_seed!(BoussinesqSimulation(mesh; ν = 1e-3, moisture = moist, bcs = bcs))
        nk = length(interior(mesh)[3])
        Δt = 0.3
        for rk3 in (false, true), (bn, b) in (("serial", CPUBackend(serial = true)), ("threaded", CPUBackend(threshold = 0)))
            with_backend(b) do
                r = mk(); c = mkc(r)
                snap = (copy(r.θ.data), copy(r.qv.data))
                _sg_reference!(r, c, stress, Δt, rk3, snap)
                # the closure acted: without it the state differs
                if bn == "serial" && !rk3
                    r0 = mk(); SIMg.fill_state_halos!(r0)
                    FVg._fused_tendencies!(r0.tu, r0.tv, r0.tw, r0.u, r0.v, r0.w, r0.ν, (r0.tθ, r0.θ),
                                           moist ? (r0.tqv, r0.qv) : nothing, mesh)
                    SIMg.axpy!(r0.u, Δt, r0.tu); SIMg.axpy!(r0.θ, Δt, r0.tθ)
                    @test !agree(interior_view(r.u), interior_view(r0.u)) &&
                          !agree(interior_view(r.θ) .- 300, interior_view(r0.θ) .- 300)   # (θ's anomaly)
                end
                for ns in 1:min(5, nk ÷ 2), rev in (false, true)
                    (rev && (ns == 1 || bn != "serial")) && continue
                    a = mk(); ca = mkc(a)
                    foreach(t -> fill!(t.data, NaN), (a.tu, a.tv, a.tw, a.tθ, a.tqv))   # neither read nor written
                    _sg_sweep!(a, ca, stress, Δt, rk3, snap, ns; reverse = rev)
                    ok = _sg_agree(a, r)
                    ok || println("  [#533 s4] mismatch: $name $gkey $(bcs.bottom) $stress moist=$moist rk3=$rk3 $bn ns=$ns rev=$rev")
                    @test ok
                    @test all(t -> all(isnan, t.data), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                end
            end
        end
    end
end

@testset "step! and step_rk3!: loop_fusion always = never: $name" for (name, mkc, gk, forms) in closures
    mesh, bcs = first(meshes[first(gk)])
    # per_point: the per-point (GPU) form of #533 step 8, the SGS term's point
    # kernel (integration of steps 4 and 8)
    for stress in forms, moist in (false, true), (drv, per_point) in ((step!, false), (step_rk3!, false), (step_rk3!, true))
        a = _sg_seed!(BoussinesqSimulation(mesh; ν = 1e-3, moisture = moist, bcs = bcs))
        r = deepcopy(a)
        a.workspaces.fused.per_point = per_point
        ca = mkc(a); cr = mkc(r)
        SIMg.set_loop_fusion!(a, :always); SIMg.set_loop_fusion!(r, :never)
        fill!(a.tu.data, NaN)
        for _ in 1:3
            drv(a, 0.3; sgs = ca, sgs_stress = stress); drv(r, 0.3; sgs = cr, sgs_stress = stress)
        end
        ok = _sg_agree(a, r)
        ok || println("  [#533 s4] step mismatch: $name $stress moist=$moist $drv per_point=$per_point")
        @test ok
        @test all(isnan, a.tu.data) != per_point     # ring: the tendency fields unused; per point: written
        @test SIMg.sweep_report(a)["substages_fused"] == "all" &&
              a.workspaces.fused.fused == (drv === step! ? 3 : 9)
        @test a.workspaces.rec.wall_sgs > 0          # the coefficients are timed as the SGS step
    end
end

@testset "the built-in Smagorinsky (C_s) and the dry CBL's closure take the sweep" begin
    a = _sg_seed!(BoussinesqSimulation(g; ν = 1e-3, C_s = 0.17)); r = deepcopy(a)
    SIMg.set_loop_fusion!(r, :never)
    fill!(a.tu.data, NaN)
    for _ in 1:2
        step_rk3!(a, 0.3); step_rk3!(r, 0.3)
    end
    @test _sg_agree(a, r) && all(isnan, a.tu.data)
    @test SIMg.sweep_report(a)["substages_fused"] == "all"
end

@testset "a closure without a sweep form is reported; the stress form on a stretched grid errs as before" begin
    a = _sg_seed!(BoussinesqSimulation(g; ν = 1e-3)); r = deepcopy(a)
    SIMg.set_loop_fusion!(r, :never)
    @test OPg.sgs_sweep_term(SGTestStress(), a, :laplacian, a.lap_θ) === nothing
    for _ in 1:2
        step!(a, 0.3; sgs = SGTestStress()); step!(r, 0.3; sgs = SGTestStress())
    end
    @test _sg_agree(a, r)
    rep = SIMg.sweep_report(a)
    @test rep["substages_fused"] == "none" &&
          rep["not_fusable"] == ["Main.SGTestStress: no sweep_term and no documented reason in Simulations.SWEEP_FALLBACKS"]
    b = SIMg.set_loop_fusion!(_sg_seed!(BoussinesqSimulation(g; ν = 1e-3)), :always)
    err = try step!(b, 0.3; sgs = SGTestStress()); nothing catch e; e end
    @test err isa ArgumentError && occursin("Main.SGTestStress", err.msg)
    # the stress form has no stretched stencil: the sweep leaves it to the
    # separate loops, whose operator refuses it by name, as before #533
    s = _sg_seed!(BoussinesqSimulation(sg; ν = 1e-3, bcs = swalls))
    @test OPg.sgs_sweep_term(StaticSmagorinsky(0.17), s, :symmetric, s.lap_θ) === nothing
    @test_throws ArgumentError step!(s, 0.3; sgs = StaticSmagorinsky(0.17), sgs_stress = :symmetric)
    @test_throws ArgumentError OPg.sgs_sweep_term(StaticSmagorinsky(0.17), s, :lagrangian, s.lap_θ)
end

@testset "a closure's own tendency without its sweep form stays out of the sweep" begin
    # the sweep would take the default kernel for it: other physics than
    # one loop per term. Each runs one loop per term, reported by its type,
    # and agrees with loop_fusion = "never"; one with the matching face form fuses.
    for (c, moist, fuses) in ((SGTestScalar(), false, false), (SGTestMom(), false, false),
                              (SGTestTracer(), true, false), (SGTestTracer(), false, false),
                              (Aeolus.Urban.SolidMaskedSGS(SGTestScalar(), mask), false, false),
                              (SGTestFace(), true, true))
        a = _sg_seed!(BoussinesqSimulation(g; ν = 1e-3, moisture = moist, bcs = walls)); r = deepcopy(a)
        SIMg.set_loop_fusion!(r, :never)
        @test (OPg.sgs_sweep_term(c, a, :laplacian, a.lap_θ) === nothing) == !fuses
        for _ in 1:2
            step_rk3!(a, 0.3; sgs = c); step_rk3!(r, 0.3; sgs = c)
        end
        ok = _sg_agree(a, r)
        ok || println("  [#533 s4] own-tendency mismatch: $(typeof(c)) moist=$moist")
        @test ok
        rep = SIMg.sweep_report(a)
        if fuses
            @test rep["substages_fused"] == "all"
        else
            @test rep["substages_fused"] == "none" && length(rep["not_fusable"]) == 1 &&
                  occursin(string(nameof(typeof(c))), only(rep["not_fusable"]))
        end
    end
    # the check follows methods added later (the cache is per world)
    a = _sg_seed!(BoussinesqSimulation(g; ν = 1e-3))
    @test OPg.sgs_sweep_term(SGTestMom(), a, :laplacian, a.lap_θ) === nothing
    @eval OPg.sgs_momentum_sweep(::SGTestMom, f::OPg.SGSFields, sim, stress::Symbol) =
        OPg._eddy_momentum_part(f, sim.mesh, stress)
    @test Base.invokelatest(OPg.sgs_sweep_term, SGTestMom(), a, :laplacian, a.lap_θ) isa OPg.SGSTerm
    # the Reynolds stress has a RegularGrid stencil only, as its separate loop
    s = _sg_seed!(BoussinesqSimulation(sg; ν = 1e-3, bcs = swalls))
    rsm = ReynoldsStressClosure(g)
    @test OPg.sgs_momentum_sweep(rsm, OPg.sgs_fields(rsm, s), s, :laplacian) === nothing
end

@testset "footprint, inputs, type stability, allocation" begin
    s = _sg_seed!(BoussinesqSimulation(g; ν = 1e-3, moisture = true))
    for c in (StaticSmagorinsky(0.17), AMDClosure(), IsentropicDiffusion(StaticSmagorinsky(0.17), g))
        t = OPg.sgs_sweep_term(c, s, :laplacian, s.lap_θ)
        @test FVg.footprint(t) == (1, 1, 1)
        @test FVg.sweep_lags((FVg.BaseTendencies(s.ν), t)) == (1, 1)      # the ring and edges of the base term
        @test all(x -> x isa Array{Float64,3}, FVg.sweep_inputs(t))
    end
    rsm = ReynoldsStressClosure(g)
    @test FVg.part_footprint(OPg.sgs_momentum_sweep(rsm, OPg.sgs_fields(rsm, s), s, :laplacian)) == (0, 0, 0)
    # qv: AMD its own diffusivity (the scratch), Smagorinsky θ's form
    ta = OPg.sgs_sweep_term(AMDClosure(), s, :laplacian, s.lap_θ)
    @test ta.own && ta.qv.Kh === s.lap_θ.data
    ts = OPg.sgs_sweep_term(StaticSmagorinsky(0.17), s, :laplacian, s.lap_θ)
    @test !ts.own && ts.qv === ts.θ && ts.θ.Kh === s.νt.data && ts.θ.s == 1 / s.Pr_t
    # the decision: a concrete term tuple or nothing; both stress forms one type
    ch = SIMg._sgs_choice(StaticSmagorinsky(0.17), :laplacian)
    T = Union{Nothing, typeof((FVg.BaseTendencies(s.ν), ts))}
    @test (@inferred T SIMg._sweep_terms(s, ch, nothing, nothing, Aeolus.Operators.momentum_advect!, s.ν)) isa Tuple
    @test typeof(OPg.sgs_sweep_term(StaticSmagorinsky(0.17), s, :symmetric, s.lap_θ)) === typeof(ts)
    # a step with the closure in the sweep allocates (almost) nothing
    for c in (StaticSmagorinsky(0.17), AMDClosure())
        for _ in 1:2
            step_rk3!(s, 0.1; sgs = c)
        end
        n = @allocated step_rk3!(s, 0.1; sgs = c)
        println("  [#533 s4] step_rk3! with $(nameof(typeof(c))) in the fused sweep allocates $n bytes")
        @test SIMg.sweep_report(s)["substages_fused"] == "all"
        @test n <= 64       # as with one loop per term (measured: 32 and 48 bytes, the keyword call)
    end
    # the sweep alone: nothing per point
    t = OPg.sgs_sweep_term(StaticSmagorinsky(0.17), s, :laplacian, s.lap_θ)
    terms = (FVg.BaseTendencies(s.ν), t)
    SIMg.fill_state_halos!(s); FVg.prepare!(t, s, 0.1)
    bufs = FVg.sweep_buffers(nothing, s.u.data, 5, 2, 12)
    sw() = FVg.blocked_predictor!(bufs, terms, s.u, s.v, s.w, SIMg._Axpy(1e-3), (s.θ, SIMg._Axpy(1e-3), nothing),
                                  (s.qv, SIMg._Axpy(1e-3), nothing), g)
    sw()
    n = @allocated sw()
    println("  [#533 s4] blocked_predictor! with the SGS term allocates $n bytes")
    @test n <= 64
end

@testset "guard: a closure's own tendency has its sweep form" begin
    # every closure type of Aeolus with its own method of a tendency
    # generic has its own method of the generic the sweep reads for it
    # (the type of the closure argument, method by method)
    closure_types(f, n) = Set{Any}(Base.unwrap_unionall(m.sig).parameters[n] for m in methods(f)
                                   if Base.moduleroot(m.module) === Aeolus &&
                                      length(Base.unwrap_unionall(m.sig).parameters) >= n)
    generic = OPg.AbstractSGSClosure
    mom = setdiff(closure_types(OPg.add_sgs_momentum!, 5), Any[generic])
    sca = setdiff(closure_types(OPg.add_sgs_scalar!, 4), Any[generic])
    trc = setdiff(closure_types(OPg.sgs_tracer_diffusivity!, 3), Any[generic])
    println("  [#533 s4] own momentum: ", join(sort(string.(mom)), ", "), "; own scalar: ",
            join(sort(string.(sca)), ", "), "; own tracer diffusivity: ", join(sort(string.(trc)), ", "))
    @test length(mom) >= 3 && length(sca) >= 4 && length(trc) >= 2
    msw = closure_types(OPg.sgs_momentum_sweep, 2)
    fd = closure_types(OPg.sgs_scalar_face_diffusivity, 2) ∪ closure_types(OPg.sgs_scalar_tensor, 2)
    own = closure_types(OPg.sgs_tracer_own_diffusivity, 2)
    for T in mom
        T in msw || println("  [#533 s4] $T: an own add_sgs_momentum! without sgs_momentum_sweep")
        @test T in msw
    end
    for T in sca
        T in fd || println("  [#533 s4] $T: an own add_sgs_scalar! without sgs_scalar_face_diffusivity/_tensor")
        @test T in fd
    end
    for T in trc
        T in own || println("  [#533 s4] $T: an own sgs_tracer_diffusivity! without sgs_tracer_own_diffusivity")
        @test T in own
    end
end

end
