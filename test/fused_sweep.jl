# The fused sweep of the step core: the local-term interface and its
# driver (SC-e #405, #533 plan step 1).
#
#   1. `FiniteVolume.blocked_predictor!` over the base term alone is the
#      fused tendency launch followed by the predictor passes, for 1-5
#      slabs, on the serial and threaded backends, regular and stretched
#      grids, dry and moist, with and without the RK3 stage combination,
#      and neither reads nor writes the tendency fields (SC-e's test);
#   2. composed terms: a pointwise term, a term whose stencil reaches two
#      rows and two planes (ring of 3 rows, 2 edge planes per slab end)
#      and a term with a reduction in `prepare!` give, inside the sweep,
#      the whole-field passes of the same terms followed by the predictor;
#   3. step! and step_rk3! on the fused path agree with the unfused path
#      (the reference) and report which path they took; a component with a
#      sweep term keeps a domain on the fused path, one without moves it to
#      the unfused path and is reported with its reason, also in a run's
#      results.toml;
#   4. type stability and allocation of the driver;
#   5. the completeness guard: every component attached to :tendencies,
#      :body_force or :post_predictor and every built-in term of the core
#      has a sweep term or a documented reason (Simulations.SWEEP_FALLBACKS),
#      and no reason is stale.
#
# Tolerances: the sweep evaluates the same floating-point expressions as
# the passes, in the same order, so the agreement is round-off
# (`agree`, ROUNDOFF_RTOL of the field's scale; measured: bit-identical).

using Test
using TOML
using Aeolus
using Aeolus.Fields: Field, set!, interior_view
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Components: AbstractComponent, Stage, Domain, CentredAdvection
using Random: Xoshiro, randn

const FVs = Aeolus.Operators.FiniteVolume
const SIMs = Aeolus.Simulations
const CPs = Aeolus.Components

const _FS_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)),
                      ("default", CPUBackend()))

function _fs_seed!(s; U = 0.3, seed = 533)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= U .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    return s
end
_fs_state(s) = (s.u, s.v, s.w, s.p, s.θ, s.qv)
_fs_agree(a, b) = all(i -> agree(interior_view(_fs_state(a)[i]), interior_view(_fs_state(b)[i])),
                      eachindex(_fs_state(a))) && a.t ≈ b.t

# ---------------------------------------------------------------------------
# test terms: their point kernels and their whole-field references
# ---------------------------------------------------------------------------

# pointwise: Newtonian damping of w and a θ source proportional to w,
# −λ w and γ w at the w point (a linear relaxation; footprint 0)
struct FSDamp{T} <: FVs.SweepTerm
    λ::T
    γ::T
end
FVs.footprint(::FSDamp) = (0, 0, 0)
@inline function FVs.add_tendency!(t::FSDamp, tb, i, j, k, st)
    @inbounds wk = st.w[i, j, k]
    tb[i, FVs.SWEEP_W] += -t.λ * wk
    tb[i, FVs.SWEEP_Θ] += t.γ * wk
    return nothing
end
function fs_damp_ref!(s, t::FSDamp)
    ii, jj, kk = interior(s.mesh)
    for k in kk, j in jj, i in ii
        s.tw.data[i, j, k] += -t.λ * s.w.data[i, j, k]
        s.tθ.data[i, j, k] += t.γ * s.w.data[i, j, k]
    end
end

# radius 2 in y and z: a fourth-difference hyperdiffusion of u along y and
# z, −c (δ⁴_y u + δ⁴_z u) (index form; the sweep must hold 3 rows and keep
# 2 edge planes per slab end for it)
struct FSHyper{T} <: FVs.SweepTerm
    c::T
end
FVs.footprint(::FSHyper) = (0, 2, 2)
@inline _fs_d4(u, i, j, k, dj, dk) = @inbounds u[i, j-2dj, k-2dk] - 4u[i, j-dj, k-dk] + 6u[i, j, k] -
                                              4u[i, j+dj, k+dk] + u[i, j+2dj, k+2dk]
@inline FVs.add_tendency!(t::FSHyper, tb, i, j, k, st) =
    (tb[i, FVs.SWEEP_U] += -t.c * (_fs_d4(st.u, i, j, k, 1, 0) + _fs_d4(st.u, i, j, k, 0, 1)); nothing)
function fs_hyper_ref!(s, t::FSHyper)
    ii, jj, kk = interior(s.mesh)
    u = s.u.data
    for k in kk, j in jj, i in ii
        s.tu.data[i, j, k] += -t.c * (_fs_d4(u, i, j, k, 1, 0) + _fs_d4(u, i, j, k, 0, 1))
    end
end

# a reduction in prepare!: relax θ toward its stage-start horizontal mean,
# −(θ − ⟨θ⟩(k))/τ, the profile computed before the sweep (an input array)
struct FSRelax{T,V<:AbstractVector{T}} <: FVs.SweepTerm
    τ::T
    mean::V
end
FVs.footprint(::FSRelax) = (0, 0, 0)
FVs.sweep_inputs(t::FSRelax) = (t.mean,)
function FVs.prepare!(t::FSRelax, sim, Δt)
    ii, jj, kk = interior(sim.mesh)
    for k in kk
        acc = 0.0
        for j in jj, i in ii
            acc += sim.θ.data[i, j, k]
        end
        t.mean[k] = acc / (length(ii) * length(jj))
    end
    return nothing
end
@inline FVs.add_tendency!(t::FSRelax, tb, i, j, k, st) =
    (@inbounds tb[i, FVs.SWEEP_Θ] += -(st.θ[i, j, k] - t.mean[k]) / t.τ; nothing)
function fs_relax_ref!(s, t::FSRelax)
    FVs.prepare!(t, s, 0.0)
    ii, jj, kk = interior(s.mesh)
    for k in kk, j in jj, i in ii
        s.tθ.data[i, j, k] += -(s.θ.data[i, j, k] - t.mean[k]) / t.τ
    end
end

_fs_ref!(s, t::FSDamp) = fs_damp_ref!(s, t)
_fs_ref!(s, t::FSHyper) = fs_hyper_ref!(s, t)
_fs_ref!(s, t::FSRelax) = fs_relax_ref!(s, t)

# the whole-field reference of a sweep: the SC-d base tendencies, each
# term's pass, then the predictors (the scalars with the RK3 combination)
function _fs_reference!(r, mesh, Δt, extra, rk3, snap)
    moist = r.has_moisture
    FVs._fused_tendencies!(r.tu, r.tv, r.tw, r.u, r.v, r.w, r.ν, (r.tθ, r.θ),
                           moist ? (r.tqv, r.qv) : nothing, mesh)
    foreach(t -> _fs_ref!(r, t), extra)
    for (f, t) in ((r.u, r.tu), (r.v, r.tv), (r.w, r.tw))
        SIMs.axpy!(f, Δt, t)
    end
    c = SIMs._Lin2r(0.75, 0.25)
    for (f, t, s0) in (moist ? ((r.θ, r.tθ, snap[1]), (r.qv, r.tqv, snap[2])) : ((r.θ, r.tθ, snap[1]),))
        rk3 ? SIMs._advance_scalar!(f, Δt, t, (c = c, ws = nothing), s0) : SIMs.axpy!(f, Δt, t)
    end
    return r
end

function _fs_sweep!(a, mesh, Δt, terms, rk3, snap, ns)
    moist = a.has_moisture
    nk = length(interior(mesh)[3])
    lag, lz = FVs.sweep_lags(terms)
    bufs = FVs.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, nk; lag = lag, lz = lz)
    foreach(t -> FVs.prepare!(t, a, Δt), terms)
    c = SIMs._Lin2r(0.75, 0.25)
    ps = rk3 ? SIMs._AxpyStage(SIMs._Axpy(Δt), c) : SIMs._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    FVs.blocked_predictor!(bufs, terms, a.u, a.v, a.w, SIMs._Axpy(Δt), s1, s2, mesh)
    return bufs
end

# The same sweep with the slabs run one after another in reverse order
# (ns, ns−1, …, 1), then the edge planes: the body of blocked_predictor!
# with its slab launch replaced. In forward order (what one thread runs)
# slab s reads slab s+1's front planes before slab s+1 has touched them, so
# only the back edge is exercised; in reverse order slab s+1 has finished
# when slab s reads its front planes, so a front edge kept shallower than a
# term's rz reads an updated plane and fails the comparison. Together the
# two orders bound what concurrent slabs can see, deterministically, at
# any thread count (#533 review).
function _fs_sweep_reversed!(a, mesh, Δt, terms, rk3, snap, ns)
    moist = a.has_moisture
    ii, jj, kk = interior(mesh)
    lag, lz = FVs.sweep_lags(terms)
    bufs = FVs.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, length(kk); lag = lag, lz = lz)
    foreach(t -> FVs.prepare!(t, a, Δt), terms)
    c = SIMs._Lin2r(0.75, 0.25)
    ps = rk3 ? SIMs._AxpyStage(SIMs._Axpy(Δt), c) : SIMs._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    pu = SIMs._Axpy(Δt)
    st = (u = a.u.data, v = a.v.data, w = a.w.data, θ = FVs._sweep_field(s1), qv = FVs._sweep_field(s2),
          m = FVs.grid_metrics(mesh))
    p1, p2 = FVs._sweep_pred(s1), FVs._sweep_pred(s2)
    for s in ns:-1:1
        FVs._b_sweep_slab!(s, terms, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
    end
    for n in 1:2lz*ns, j in jj, i in ii
        FVs._b_sweep_edge!(i, j, n, st, pu, p1, p2, bufs.edge, kk, ns, lz)
    end
    return bufs
end

# ---------------------------------------------------------------------------
# components: one with a sweep term, one without
# ---------------------------------------------------------------------------

# Newtonian damping of w as a :body_force component; its run_stage! is the
# whole-field reference, its sweep term the same in the sweep
struct FSDampComponent <: AbstractComponent
    λ::Float64
end
# the same physics without a sweep term
struct FSDampWhole <: AbstractComponent
    λ::Float64
end
for C in (FSDampComponent, FSDampWhole)
    @eval CPs.component_name(::Type{$C}) = Symbol(lowercase(string(nameof($C))))
    @eval CPs.section(::Type{$C}) = string(nameof($C))
    @eval CPs.stages(::$C) = (Stage(:body_force),)
    @eval function CPs.run_stage!(c::$C, d, ::Val{:body_force}, dt)
        fs_damp_ref!(d.state, FSDamp(c.λ, 0.0))
        return nothing
    end
end
FVs.sweep_term(c::FSDampComponent, d) = FSDamp(c.λ, 0.0)

# ---------------------------------------------------------------------------
# the completeness guard
# ---------------------------------------------------------------------------

# Every method of the stage functions (`Components.run_stage!`, the urban
# site's `Urban.run_stage!`) that runs a component inside the core (:tendencies,
# :body_force, :post_predictor), defined in Aeolus: `(type, stage)`; a
# method generic over the stage `(type, :any)`.
function _fs_in_core_methods()
    out = Tuple{Type,Symbol}[]
    for f in (CPs.run_stage!, Aeolus.Urban.run_stage!), m in methods(f)
        Base.moduleroot(m.module) === Aeolus || continue
        sig = Base.unwrap_unionall(m.sig)
        length(sig.parameters) == 5 || continue
        C = sig.parameters[2]; V = sig.parameters[4]
        (V isa DataType && V <: Val) || continue
        S = V.parameters[1]
        if S isa Symbol
            S in CPs.IN_CORE && push!(out, (C, S))
        else
            push!(out, (C, :any))
        end
    end
    return out
end

# the wrappers and defaults that run a stage generically: they delegate to
# the component they wrap, whose own methods are listed
const _FS_WRAPPERS = Dict{Type,String}(
    Aeolus.Urban.UrbanEntry => "runs its coupling's Urban.run_stage!; sweep_term and sweep_type delegate to it",
    Aeolus.Urban.UrbanComponent => "the default of an urban coupling for a stage it does not run (a no-op)",
)

_fs_own_sweep_term(C) = which(FVs.sweep_term, Tuple{C, Any}) !== which(FVs.sweep_term, Tuple{Any, Any})

@testset "fused sweep: local-term interface and driver (#533)" begin

g = RegularGrid(; size = (20, 12, 12), extent = (200.0, 120.0, 120.0), halo = (3, 3, 3))
sg = StretchedGrid(; size = (12, 10, 16), extent = (240.0, 200.0, 600.0), stretching = (:tanh, 1.5))
grids = (("regular", g, FieldBoundaryConditions(:periodic)),
         ("stretched, walls", sg, FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)))

@testset "base term: = fused tendencies + predictor, for any slab count" begin
    for (gname, mesh, bcs) in grids
        nk = length(interior(mesh)[3])
        for moist in (false, true), rk3 in (false, true), (bn, b) in _FS_BACKENDS
            with_backend(b) do
                mk() = (s = _fs_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs));
                        SIMs.fill_state_halos!(s); s)
                Δt = 0.3
                r = mk()
                snap = (copy(r.θ.data), copy(r.qv.data))
                _fs_reference!(r, mesh, Δt, (), rk3, snap)
                for ns in 1:min(5, nk ÷ 2)
                    a = mk()
                    foreach(t -> fill!(t.data, NaN), (a.tu, a.tv, a.tw, a.tθ, a.tqv))   # neither read nor written
                    _fs_sweep!(a, mesh, Δt, (FVs.BaseTendencies(a.ν),), rk3, snap, ns)
                    ok = all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                             ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                    ok || println("  [#533] base mismatch: $gname moist=$moist rk3=$rk3 $bn ns=$ns")
                    @test ok
                    @test all(t -> all(isnan, t.data), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                    if ns > 1 && bn == "serial"         # the slabs in reverse order (front edge)
                        a = mk()
                        _fs_sweep_reversed!(a, mesh, Δt, (FVs.BaseTendencies(a.ν),), rk3, snap, ns)
                        @test all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                                  ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                    end
                end
            end
        end
    end
    # the buffers: reused when they fit; a slab count, lag or edge depth they do not fit is refused
    s = BoussinesqSimulation(g; ν = 1e-2)
    b2 = FVs.sweep_buffers(nothing, s.u.data, 4, 2, 12)
    @test FVs.sweep_buffers(b2, s.u.data, 4, 2, 12) === b2
    @test FVs.sweep_buffers(b2, s.u.data, 4, 3, 12) !== b2
    @test FVs.sweep_buffers(b2, s.u.data, 4, 2, 12; lag = 2) !== b2
    base = (FVs.BaseTendencies(s.ν),)
    s1 = (s.θ, SIMs._Axpy(0.1), nothing)
    @test_throws DimensionMismatch FVs.blocked_predictor!(FVs.sweep_buffers(nothing, s.u.data, 4, 7, 12),
                                                          base, s.u, s.v, s.w, SIMs._Axpy(0.1), s1, nothing, g)
    @test_throws DimensionMismatch FVs.blocked_predictor!(b2, (base..., FSHyper(0.01)), s.u, s.v, s.w,
                                                          SIMs._Axpy(0.1), s1, nothing, g)
    # only the base term initialises a row
    @test_throws MethodError FVs.blocked_predictor!(b2, (FSDamp(0.1, 0.0),), s.u, s.v, s.w,
                                                    SIMs._Axpy(0.1), s1, nothing, g)
    @test FVs.sweep_slabs(CPUBackend(serial = true), 10^9, 256) == 1
    @test FVs.sweep_slabs(CPUBackend(threshold = 0), 10^9, 256) == min(Threads.nthreads(), 128)
    @test FVs.sweep_slabs(CPUBackend(threshold = 0), 10^9, 256; lz = 2) == min(Threads.nthreads(), 64)
    @test FVs.sweep_slabs(CPUBackend(threshold = 0), 10^9, 3) == 1
end

@testset "composed terms: the sweep = the terms' whole-field passes + predictor" begin
    @test FVs.sweep_lags((FVs.BaseTendencies(1.0),)) == (1, 1)
    @test FVs.sweep_lags((FVs.BaseTendencies(1.0), FSDamp(1.0, 1.0))) == (1, 1)
    @test FVs.sweep_lags((FVs.BaseTendencies(1.0), FSHyper(1.0))) == (2, 2)
    for (gname, mesh, bcs) in grids
        nk = length(interior(mesh)[3])
        for moist in (false, true), rk3 in (false, true), (bn, b) in _FS_BACKENDS[[1, 2]]
            with_backend(b) do
                mk() = (s = _fs_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs));
                        SIMs.fill_state_halos!(s); s)
                Δt = 0.3
                extra = (FSDamp(0.05, 2e-3), FSHyper(0.01), FSRelax(50.0, zeros(size(mk().θ.data, 3))))
                r = mk()
                snap = (copy(r.θ.data), copy(r.qv.data))
                _fs_reference!(r, mesh, Δt, extra, rk3, snap)
                # the terms change the result (the comparison is not of the base alone)
                r0 = _fs_reference!(mk(), mesh, Δt, (), rk3, snap)
                @test !agree(interior_view(r.u), interior_view(r0.u)) &&
                      !agree(interior_view(r.w), interior_view(r0.w)) &&
                      !agree(interior_view(r.θ), interior_view(r0.θ))
                for ns in 1:min(5, nk ÷ 4)
                    a = mk()
                    foreach(t -> fill!(t.data, NaN), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                    bufs = _fs_sweep!(a, mesh, Δt, (FVs.BaseTendencies(a.ν), extra...), rk3, snap, ns)
                    @test bufs.lag == 2 && bufs.lz == 2 && size(bufs.ring, 3) == 3
                    ok = all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                             ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                    ok || println("  [#533] composed mismatch: $gname moist=$moist rk3=$rk3 $bn ns=$ns")
                    @test ok
                    @test all(t -> all(isnan, t.data), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                    # the slabs in reverse order: the front edge planes (rz = 2) are kept
                    if ns > 1 && bn == "serial"
                        a = mk()
                        _fs_sweep_reversed!(a, mesh, Δt, (FVs.BaseTendencies(a.ν), extra...), rk3, snap, ns)
                        ok = all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                                 ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                        ok || println("  [#533] reversed-slab mismatch: $gname moist=$moist rk3=$rk3 ns=$ns")
                        @test ok
                    end
                end
            end
        end
    end
end

cases = (
    ("periodic, viscous (DNS)",   () -> BoussinesqSimulation(g; ν = 1e-2)),
    ("periodic, inviscid",        () -> BoussinesqSimulation(g)),
    ("periodic, moist",           () -> BoussinesqSimulation(g; ν = 1e-2, moisture = true)),
    ("walls (cosine solver)",     () -> BoussinesqSimulation(g; ν = 1e-2,
                                            bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))),
    ("stretched, walls (column)", () -> BoussinesqSimulation(sg; ν = 1e-2,
                                            bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall))),
)
noop = (s, Δt) -> nothing
@testset "$name: step! and step_rk3! fused = unfused" for (name, mk) in cases
    for (bn, b) in _FS_BACKENDS, drv in (step!, step_rk3!)
        with_backend(b) do
            a = _fs_seed!(mk()); r = _fs_seed!(mk())
            fill!(a.tu.data, NaN)
            for _ in 1:3
                drv(a, 0.5); drv(r, 0.5; body_force! = noop)   # a no-op hook: the unfused reference
            end
            @test _fs_agree(a, r)
            @test all(isnan, a.tu.data)          # the fused path: tendencies unused
            ra = SIMs.sweep_report(a); rr = SIMs.sweep_report(r)
            nsub = drv === step! ? 3 : 9
            @test ra["path"] == "fused" && a.workspaces.fused.fused == nsub && isempty(ra["fallback"])
            @test rr["path"] == "unfused" && r.workspaces.fused.unfused == nsub &&
                  rr["fallback"] == ["builtin.body_force: " * SIMs.sweep_reason("builtin.body_force")]
        end
    end
end

@testset "a built-in term or a hook keeps the unfused path, and is reported" begin
    for (kw, hook, key) in (((θ₀ = 300.0,), NamedTuple(), "builtin.buoyancy"),
                            ((f_coriolis = 1e-4,), NamedTuple(), "builtin.coriolis"),
                            ((C_s = 0.17,), NamedTuple(), "builtin.sgs"),
                            ((θ_flux_bottom = 0.1,), NamedTuple(), "builtin.surface_flux"),
                            ((sponge_α_max = 0.01, sponge_z_start = 80.0), NamedTuple(), "builtin.sponge"),
                            (NamedTuple(), (post_predictor! = noop,), "builtin.post_predictor"))
        a = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2, kw...))
        fill!(a.tu.data, NaN)
        step!(a, 0.5; hook...); step!(a, 0.5; hook...)
        @test !any(isnan, interior_view(a.tu))       # the tendencies were written
        rep = SIMs.sweep_report(a)
        @test rep["path"] == "unfused" && a.workspaces.fused.unfused == 2 && a.workspaces.fused.fused == 0
        @test any(startswith(key * ":"), rep["fallback"])
        @test !occursin("no documented reason", join(rep["fallback"]))
    end
    # several terms: every one is reported
    a = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-4))
    step!(a, 0.5)
    @test Set(first.(split.(SIMs.sweep_report(a)["fallback"], ":"))) == Set(["builtin.buoyancy", "builtin.coriolis"])
    # a core without a fused sweep (the anelastic core) reports its reason
    an = Aeolus.Anelastic.AnelasticSimulation(RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0)))
    @test SIMs.sweep_report(an)["path"] == "unfused" &&
          SIMs.sweep_report(an)["fallback"] == ["builtin.core: " * SIMs.sweep_reason("builtin.core")]
end

@testset "components: a sweep term keeps the fused path, a missing one is reported" begin
    for drv in (:euler, :rk3)
        mk(c) = (s = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2));
                 d0 = Domain(s; scheme = drv, validate = false);
                 Domain(d0; components = (CentredAdvection(), c)))
        a = mk(FSDampComponent(0.05)); r = mk(FSDampWhole(0.05))
        h = CPs.core_hooks(a)
        @test SIMs.hook_terms(h.body_force!, a.state) == (FSDamp(0.05, 0.0),)
        @test (@inferred SIMs.hook_terms(h.body_force!, a.state)) isa Tuple{FSDamp{Float64}}
        @test SIMs.hook_terms(CPs.core_hooks(r).body_force!, r.state) === nothing
        fill!(a.state.tw.data, NaN)
        for _ in 1:3
            step!(a, 0.5); step!(r, 0.5)
        end
        @test _fs_agree(a.state, r.state)
        @test all(isnan, a.state.tw.data)
        # without the damping the state differs: the component acted on both paths
        z = mk(FSDampWhole(0.0)); foreach(_ -> step!(z, 0.5), 1:3)
        @test !agree(interior_view(z.state.w), interior_view(a.state.w))
        ra = CPs.sweep_report(a); rr = CPs.sweep_report(r)
        @test ra["path"] == "fused" && isempty(ra["fallback"])
        @test rr["path"] == "unfused" && only(rr["fallback"]) == "Main.FSDampWhole: " *
              "no sweep_term and no documented reason in Simulations.SWEEP_FALLBACKS"
    end
    # with the budgets on the hook counts the substages: unfused, reported
    # (the energy budget needs the reference θ₀, so buoyancy is on too)
    s = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0))
    d = Domain(Domain(s; validate = false); components = (CentredAdvection(), FSDampComponent(0.05)))
    db = CPs.with_budgets(d, (:energy,))
    step!(db, 0.5)
    @test any(startswith("builtin.budgets:"), CPs.sweep_report(db)["fallback"])
end

@testset "results.toml: the run records its path ([fused_sweep])" begin
    ex = joinpath(pkgdir(Aeolus), "examples")
    run(cfg, sets) = (root = mktempdir();
                      rc = Aeolus.main(String["run", cfg, collect(Iterators.flatten(("--set", x) for x in sets))...,
                                              "--root", root, "--allow-uncommitted"]; io = IOBuffer(), err = IOBuffer());
                      (rc, TOML.parsefile(joinpath(only(readdir(root; join = true)), "output", "results.toml"))))
    small = ["grid.size=[8,8,8]", "fftw.planner=\"estimate\"", "time.duration=\"none\"", "time.nsteps=3"]
    # a DNS (the Taylor–Green vortex): every substage fused
    rc, res = run(joinpath(ex, "taylor_green_vortex", "taylor_green_vortex.toml"), small)
    @test rc == 0
    fs = only(values(res["fused_sweep"]))
    @test fs["path"] == "fused" && isempty(fs["fallback"])
    # the dry convective boundary layer: SGS, buoyancy, a θ flux and a sponge
    rc, res = run(joinpath(ex, "dry_convective_bl", "dry_convective_bl.toml"),
                  vcat(small, ["checks.finite={}", "checks.energy_closure={}", "checks.momentum_closure={}"]))
    @test rc == 0
    fs = only(values(res["fused_sweep"]))
    who = Set(first.(split.(fs["fallback"], ":")))
    @test fs["path"] == "unfused"
    @test issubset(["builtin.sgs", "builtin.buoyancy", "Components.SurfaceThetaFlux", "Components.Sponge"], who)
    @test !occursin("no documented reason", join(fs["fallback"]))
end

@testset "type stability and allocation" begin
    s = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2))
    # the plan is a concrete term tuple or nothing (a two-type union the caller splits)
    @test (@inferred Union{Nothing,Tuple{FVs.BaseTendencies{Float64}}} SIMs._sweep_terms(
               s, nothing, nothing, nothing, Aeolus.Operators.momentum_advect!, s.ν)) == (FVs.BaseTendencies(s.ν),)
    sb = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0))
    @test SIMs._sweep_terms(sb, nothing, nothing, nothing, Aeolus.Operators.momentum_advect!, sb.ν) === nothing
    step_rk3!(s, 0.1); step_rk3!(s, 0.1)
    bufs = s.workspaces.sweep
    @test bufs isa FVs.SweepBuffers
    step_rk3!(s, 0.1)
    @test s.workspaces.sweep === bufs
    n = @allocated step_rk3!(s, 0.1)
    println("  [#533] step_rk3! on the fused path allocates $n bytes")
    @test n <= 64          # (measured 16 bytes, as before #533)
    # a composed sweep allocates nothing
    SIMs.fill_state_halos!(s)
    terms = (FVs.BaseTendencies(s.ν), FSDamp(0.05, 2e-3), FSHyper(0.01))
    nk = length(interior(g)[3])
    b2 = FVs.sweep_buffers(nothing, s.u.data, 4, 2, nk; lag = 2, lz = 2)
    s1 = (s.θ, SIMs._Axpy(1e-3), nothing)
    sweep() = FVs.blocked_predictor!(b2, terms, s.u, s.v, s.w, SIMs._Axpy(1e-3), s1, nothing, g)
    sweep()
    n = @allocated sweep()
    println("  [#533] a composed blocked_predictor! allocates $n bytes")
    @test n <= 64          # the launch's task closure on the threaded backend, none per point
    @test (@inferred FVs.sweep_lags(terms)) == (2, 2)
end

@testset "completeness guard: every in-core term has a sweep term or a documented reason" begin
    found = _fs_in_core_methods()
    @test !isempty(found)
    types = unique(first.(found))
    println("  [#533] components inside the core: ",
            join(sort([SIMs.sweep_key(C) * " (" * join(sort([string(s) for (T, s) in found if T === C]), ", ") * ")"
                       for C in types]), "; "))
    for (C, S) in found
        S === :any || continue
        @test haskey(_FS_WRAPPERS, C)            # a new generic in-core method: list it or make it specific
    end
    listed = Set{String}()
    for C in types
        haskey(_FS_WRAPPERS, C) && continue
        key = SIMs.sweep_key(C)
        has = _fs_own_sweep_term(C)
        documented = haskey(SIMs.SWEEP_FALLBACKS, key)
        has || documented || println("  [#533] $key runs inside the core with neither a sweep_term nor a reason " *
                                     "in Simulations.SWEEP_FALLBACKS")
        @test has || documented
        documented && push!(listed, key)
    end
    # the urban couplings that declare an in-core stage are among them (no
    # in-core work through a path the method scan misses)
    for C in Aeolus.Urban.URBAN_COMPONENTS
        isconcretetype(C) || continue
        any(st -> st[1] in CPs.IN_CORE, Aeolus.Urban.stages(C)) || continue
        @test C in types || any(T -> C <: T && !haskey(_FS_WRAPPERS, T), types)
    end
    # built-in terms of the Boussinesq core: every term `_euler_projection!`
    # adds (its `add_*!` calls and its physics switches) is the base term
    # or has a documented reason
    src = read(joinpath(pkgdir(Aeolus), "src", "Simulations.jl"), String)
    a = findfirst("function _euler_projection!", src); b = findnext("\nend\n", src, last(a))
    body = src[first(a):last(b)]
    builtin = Dict(
        # the sweep's base term (FiniteVolume.BaseTendencies): advection, diffusion, moisture
        "has_viscosity" => :base, "has_moisture" => :base, "advect_centered!" => :base,
        "laplacian" => :base, "_base_tendencies!" => :base, "zero_storage!" => :base, "axpy!" => :predictor,
        "_advance_scalar!" => :predictor,
        "add_sgs_momentum!" => "builtin.sgs", "add_sgs_scalar!" => "builtin.sgs", "_add_sgs_tracer!" => "builtin.sgs",
        "compute_sgs!" => "builtin.sgs", "momentum_advection" => "builtin.momentum_advection",
        "has_buoyancy" => "builtin.buoyancy", "add_buoyancy!" => "builtin.buoyancy",
        "has_coriolis" => "builtin.coriolis", "add_coriolis!" => "builtin.coriolis",
        "has_surface_flux" => "builtin.surface_flux", "add_surface_flux!" => "builtin.surface_flux",
        "has_sponge" => "builtin.sponge", "add_velocity_sponge!" => "builtin.sponge",
        "body_force!" => "builtin.body_force", "post_predictor!" => "builtin.post_predictor",
        "update_open_boundaries!" => "builtin.open_boundaries")
    seen = Set{String}()
    for m in eachmatch(r"\bsim\.(has_\w+)|\b(add_\w+!|_add_\w+!|compute_sgs!|advect_centered!|laplacian|axpy!|_advance_scalar!|_base_tendencies!|zero_storage!)\(|\b(body_force!|post_predictor!|momentum_advection)\(", body)
        name = something(m.captures...)
        push!(seen, name)
        haskey(builtin, name) || println("  [#533] a built-in term of _euler_projection! not classified: $name")
        @test haskey(builtin, name)
    end
    for name in seen
        v = get(builtin, name, nothing)
        v isa String && @test haskey(SIMs.SWEEP_FALLBACKS, v)
    end
    # the base term covers what it claims: a moist, viscous sweep was checked above
    # no stale entries: every reason is reported by the code (built-ins) or
    # belongs to an in-core component without a sweep term
    fsrc = read(joinpath(pkgdir(Aeolus), "src", "simulations", "fused_sweep.jl"), String) *
           read(joinpath(pkgdir(Aeolus), "src", "components", "domain.jl"), String)
    for key in keys(SIMs.SWEEP_FALLBACKS)
        if startswith(key, "builtin.")
            occursin("\"$key\"", replace(fsrc, r"const SWEEP_FALLBACKS = Dict\{String,String\}\((?s:.*?)\n\)\n" => "")) ||
                println("  [#533] stale reason: $key is never reported")
            @test occursin("\"$key\"", replace(fsrc, r"const SWEEP_FALLBACKS = Dict\{String,String\}\((?s:.*?)\n\)\n" => ""))
        else
            key in listed || println("  [#533] stale reason: $key is not an in-core component without a sweep term")
            @test key in listed
        end
    end
end

end
