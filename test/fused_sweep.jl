# The fused sweep of the step core: the local-term interface and its
# driver (SC-e #405, #533 plan step 1).
#
#   1. `FiniteVolume.blocked_predictor!` over the base term alone is the
#      fused tendency launch followed by the predictor passes, for 1-5
#      slabs times 1-3 row ranges in j (#558), on the serial and threaded
#      backends, regular and stretched grids, dry and moist, with and
#      without the RK3 stage combination, and neither reads nor writes the
#      tendency fields (SC-e's test); the blocks also run concurrently in a
#      -t 4 subprocess;
#   2. composed terms: a pointwise term, a term whose stencil reaches two
#      rows and two planes (ring of 3 rows, 2 edge planes per slab end)
#      and a term with a reduction in `prepare!` give, inside the sweep,
#      the whole-field passes of the same terms followed by the predictor;
#   3. step! and step_rk3! on the fused sweep agree with the separate loops
#      (the reference) and report whether they ran it; a component with a
#      sweep term keeps a domain on the fused sweep, one without moves it to
#      the separate loops and is reported with its reason, also in a run's
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

function _fs_sweep!(a, mesh, Δt, terms, rk3, snap, ns, nt = 1)
    moist = a.has_moisture
    nk = length(interior(mesh)[3])
    lag, lz = FVs.sweep_lags(terms)
    bufs = FVs.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, nk; lag = lag, lz = lz, nt = nt, ly = lag)
    foreach(t -> FVs.prepare!(t, a, Δt), terms)
    c = SIMs._Lin2r(0.75, 0.25)
    ps = rk3 ? SIMs._AxpyStage(SIMs._Axpy(Δt), c) : SIMs._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    FVs.blocked_predictor!(bufs, terms, a.u, a.v, a.w, SIMs._Axpy(Δt), s1, s2, mesh)
    return bufs
end

# The same sweep with the blocks run one after another in reverse order
# (nb, nb−1, …, 1), then the edge planes and rows: the body of
# blocked_predictor! with its block launch replaced. In forward order (what
# one thread runs) slab s reads slab s+1's front planes before slab s+1 has
# touched them, so only the back edge is exercised; in reverse order slab
# s+1 has finished when slab s reads its front planes, so a front edge kept
# shallower than a term's rz reads an updated plane and fails the
# comparison; likewise for the row ranges in j. Together the two orders
# bound what concurrent blocks can see, deterministically, at any thread
# count (#533 review).
function _fs_sweep_reversed!(a, mesh, Δt, terms, rk3, snap, ns, nt = 1)
    moist = a.has_moisture
    nk = length(interior(mesh)[3])
    lag, lz = FVs.sweep_lags(terms)
    bufs = FVs.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, nk; lag = lag, lz = lz, nt = nt, ly = lag)
    foreach(t -> FVs.prepare!(t, a, Δt), terms)
    c = SIMs._Lin2r(0.75, 0.25)
    ps = rk3 ? SIMs._AxpyStage(SIMs._Axpy(Δt), c) : SIMs._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    FVs.blocked_predictor!(bufs, terms, a.u, a.v, a.w, SIMs._Axpy(Δt), s1, s2, mesh; reverse_slabs = true)
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

@testset "base term: = fused tendencies + predictor, for any slab and row-range count" begin
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
                for ns in 1:min(5, nk ÷ 2), nt in 1:3
                    a = mk()
                    foreach(t -> fill!(t.data, NaN), (a.tu, a.tv, a.tw, a.tθ, a.tqv))   # neither read nor written
                    _fs_sweep!(a, mesh, Δt, (FVs.BaseTendencies(a.ν),), rk3, snap, ns, nt)
                    ok = all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                             ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                    ok || println("  [#533] base mismatch: $gname moist=$moist rk3=$rk3 $bn ns=$ns nt=$nt")
                    @test ok
                    @test all(t -> all(isnan, t.data), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                    if ns * nt > 1 && bn == "serial"    # the blocks in reverse order (front edge)
                        a = mk()
                        _fs_sweep_reversed!(a, mesh, Δt, (FVs.BaseTendencies(a.ν),), rk3, snap, ns, nt)
                        @test all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                                  ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                    end
                end
            end
        end
    end
    # the buffers: reused when they fit; a block count, lag or edge depth they do not fit is refused
    s = BoussinesqSimulation(g; ν = 1e-2)
    b2 = FVs.sweep_buffers(nothing, s.u.data, 4, 2, 12)
    @test FVs.sweep_buffers(b2, s.u.data, 4, 2, 12) === b2
    @test FVs.sweep_buffers(b2, s.u.data, 4, 3, 12) !== b2
    @test FVs.sweep_buffers(b2, s.u.data, 4, 2, 12; lag = 2) !== b2
    @test FVs.sweep_buffers(b2, s.u.data, 4, 2, 12; nt = 2) !== b2
    b22 = FVs.sweep_buffers(nothing, s.u.data, 4, 2, 12; nt = 2, ly = 1)
    @test FVs.sweep_buffers(b22, s.u.data, 4, 2, 12; nt = 2, ly = 1) === b22
    @test size(b22.ring, 3) == 2 + 2 && size(b22.ring, 5) == 4      # ring rows + edge rows, ns·nt blocks
    base = (FVs.BaseTendencies(s.ν),)
    s1 = (s.θ, SIMs._Axpy(0.1), nothing)
    @test_throws DimensionMismatch FVs.blocked_predictor!(FVs.sweep_buffers(nothing, s.u.data, 4, 7, 12),
                                                          base, s.u, s.v, s.w, SIMs._Axpy(0.1), s1, nothing, g)
    @test_throws DimensionMismatch FVs.blocked_predictor!(b2, (base..., FSHyper(0.01)), s.u, s.v, s.w,
                                                          SIMs._Axpy(0.1), s1, nothing, g)
    # only the base term initialises a row
    @test_throws MethodError FVs.blocked_predictor!(b2, (FSDamp(0.1, 0.0),), s.u, s.v, s.w,
                                                    SIMs._Axpy(0.1), s1, nothing, g)
    # more row ranges than rows is refused
    @test_throws DimensionMismatch FVs.blocked_predictor!(FVs.sweep_buffers(nothing, s.u.data, 4, 1, 12; nt = 13),
                                                          base, s.u, s.v, s.w, SIMs._Axpy(0.1), s1, nothing, g)
end

@testset "sweep_blocks: the split into slabs and row ranges (#558)" begin
    @test FVs.sweep_blocks(CPUBackend(serial = true), 10^9, 256, 256) == (1, 1)
    # the split of T threads (no thread count needed: the rule itself)
    sb = FVs._split_blocks
    for T in (1, 2, 4, 6, 12, 24), (nk, nj) in ((60, 200), (12, 12), (3, 3), (256, 256)), (lz, ly) in ((1, 1), (2, 2), (4, 4))
        ns, nt = sb(T, nk, nj, lz, ly)
        @test ns * nt <= T                                   # at most one block per thread
        @test ns == 1 || 2lz * ns <= nk                      # at least 2lz planes per slab
        @test nt == 1 || 2ly * nt <= nj                      # at least 2ly rows per range
    end
    # the cap of #554 is lifted: 60 levels, WENO-Z5 (lz = ly = 4), 200 rows
    @test prod(sb(12, 60, 200, 4, 4)) == 12 && prod(sb(24, 60, 200, 4, 4)) == 24
    @test prod(sb(24, 60, 200, 2, 2)) == 24 && prod(sb(24, 60, 200, 1, 1)) == 24
    @test sb(1, 60, 200, 4, 4) == (1, 1) && sb(24, 3, 3, 1, 1) == (1, 1)
    # slabs no taller than 20 planes where the threads allow (whole columns measured slower)
    @test all(T -> cld(60, sb(T, 60, 200, 1, 1)[1]) <= 20, (3, 6, 12, 24))
    @test sb(2, 60, 200, 1, 1) == (2, 1)                     # fewer threads than the 3 slabs
    @test sb(24, 60, 3, 1, 1) == (24, 1)                     # one row range: k alone, as before
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
                for ns in 1:min(5, nk ÷ 4), nt in 1:3
                    a = mk()
                    foreach(t -> fill!(t.data, NaN), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                    bufs = _fs_sweep!(a, mesh, Δt, (FVs.BaseTendencies(a.ν), extra...), rk3, snap, ns, nt)
                    @test bufs.lag == 2 && bufs.lz == 2 && size(bufs.ring, 3) == 3 + (nt > 1 ? 4 : 0)
                    ok = all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                             ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                    ok || println("  [#533] composed mismatch: $gname moist=$moist rk3=$rk3 $bn ns=$ns nt=$nt")
                    @test ok
                    @test all(t -> all(isnan, t.data), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                    # the blocks in reverse order: the front edge planes and rows (rz = ry = 2) are kept
                    if ns * nt > 1 && bn == "serial"
                        a = mk()
                        _fs_sweep_reversed!(a, mesh, Δt, (FVs.BaseTendencies(a.ν), extra...), rk3, snap, ns, nt)
                        ok = all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                                 ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                        ok || println("  [#533] reversed-block mismatch: $gname moist=$moist rk3=$rk3 ns=$ns nt=$nt")
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
            @test all(isnan, a.tu.data)          # the fused sweep: tendencies unused
            ra = SIMs.sweep_report(a); rr = SIMs.sweep_report(r)
            nsub = drv === step! ? 3 : 9
            @test ra["substages_fused"] == "all" && a.workspaces.fused.fused == nsub && isempty(ra["not_fusable"])
            @test rr["substages_fused"] == "none" && r.workspaces.fused.unfused == nsub &&
                  rr["not_fusable"] == ["builtin.body_force: " * SIMs.sweep_reason("builtin.body_force")]
        end
    end
end

@testset "a built-in term or a hook keeps one loop per term, and is reported" begin
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
        @test rep["substages_fused"] == "none" && a.workspaces.fused.unfused == 2 && a.workspaces.fused.fused == 0
        @test any(startswith(key * ":"), rep["not_fusable"])
        @test !occursin("no documented reason", join(rep["not_fusable"]))
    end
    # several terms: every one is reported
    a = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-4))
    step!(a, 0.5)
    @test Set(first.(split.(SIMs.sweep_report(a)["not_fusable"], ":"))) == Set(["builtin.buoyancy", "builtin.coriolis"])
    # a core without a fused sweep (the anelastic core) reports its reason
    an = Aeolus.Anelastic.AnelasticSimulation(RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0)))
    @test SIMs.sweep_report(an)["substages_fused"] == "none" &&
          SIMs.sweep_report(an)["not_fusable"] == ["builtin.core: " * SIMs.sweep_reason("builtin.core")]
end

@testset "loop_fusion: if_possible, always, never" begin
    # never: one loop per term although the DNS has a fused form; the
    # results agree with the fused step within round-off (ROUNDOFF_RTOL)
    a = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2)); r = deepcopy(a)
    @test SIMs.loop_fusion(a) === :if_possible                  # the default
    SIMs.set_loop_fusion!(r, "never")
    fill!(a.tu.data, NaN); fill!(r.tu.data, NaN)
    step_rk3!(a, 0.5); step_rk3!(r, 0.5)
    @test all(isnan, a.tu.data) && !any(isnan, interior_view(r.tu))   # fused vs one loop per term
    @test _fs_agree(a, r)
    rr = SIMs.sweep_report(r)
    @test rr["setting"] == "never" && rr["substages_fused"] == "none" && isempty(rr["not_fusable"])
    @test SIMs.sweep_report(a)["setting"] == "if_possible" && SIMs.sweep_report(a)["substages_fused"] == "all"
    # always: fine where every term has a fused form ...
    b = SIMs.set_loop_fusion!(_fs_seed!(BoussinesqSimulation(g; ν = 1e-2)), :always)
    step!(b, 0.5)
    @test SIMs.sweep_report(b)["substages_fused"] == "all"
    # ... and an error naming every term without one, before anything is stepped
    c = SIMs.set_loop_fusion!(_fs_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-4)), :always)
    t0 = c.t
    err = try step!(c, 0.5); nothing catch e; e end
    @test err isa ArgumentError && occursin("builtin.buoyancy", err.msg) && occursin("builtin.coriolis", err.msg)
    @test c.t == t0                                             # exact: the clock did not advance
    # invalid settings, and always on a core without a fused form
    @test_throws ArgumentError SIMs.set_loop_fusion!(a, :sometimes)
    an = Aeolus.Anelastic.AnelasticSimulation(RegularGrid(; size = (8, 8, 8), extent = (1.0, 1.0, 1.0)))
    @test_throws ArgumentError SIMs.set_loop_fusion!(an, :always)
    @test SIMs.loop_fusion(SIMs.set_loop_fusion!(an, :never)) === :never
end

@testset "components: a sweep term keeps the fused sweep, a missing one is reported" begin
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
        # without the damping the state differs: the component acted in both variants
        z = mk(FSDampWhole(0.0)); foreach(_ -> step!(z, 0.5), 1:3)
        @test !agree(interior_view(z.state.w), interior_view(a.state.w))
        ra = CPs.sweep_report(a); rr = CPs.sweep_report(r)
        @test ra["substages_fused"] == "all" && isempty(ra["not_fusable"])
        @test rr["substages_fused"] == "none" && only(rr["not_fusable"]) == "Main.FSDampWhole: " *
              "no sweep_term and no documented reason in Simulations.SWEEP_FALLBACKS"
    end
    # with the budgets on the hook counts the substages: unfused, reported
    # (the energy budget needs the reference θ₀, so buoyancy is on too)
    s = _fs_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0))
    d = Domain(Domain(s; validate = false); components = (CentredAdvection(), FSDampComponent(0.05)))
    db = CPs.with_budgets(d, (:energy,))
    step!(db, 0.5)
    @test any(startswith("builtin.budgets:"), CPs.sweep_report(db)["not_fusable"])
end

@testset "results.toml [loop_fusion] and the [timestep] loop_fusion key" begin
    ex = joinpath(pkgdir(Aeolus), "examples")
    run(cfg, sets) = (root = mktempdir();
                      rc = Aeolus.main(String["run", cfg, collect(Iterators.flatten(("--set", x) for x in sets))...,
                                              "--root", root, "--allow-uncommitted"]; io = IOBuffer(), err = IOBuffer());
                      (rc, TOML.parsefile(joinpath(only(readdir(root; join = true)), "output", "results.toml"))))
    small = ["grid.size=[8,8,8]", "fftw.planner=\"estimate\"", "time.duration=\"none\"", "time.nsteps=3"]
    # a DNS (the Taylor–Green vortex): every substage fused; a run without
    # [checks] or component results writes results.toml too (since #533)
    rc, res = run(joinpath(ex, "taylor_green_vortex", "taylor_green_vortex.toml"), small)
    @test rc == 0
    fs = only(values(res["loop_fusion"]))
    @test fs["substages_fused"] == "all" && isempty(fs["not_fusable"])
    # the dry convective boundary layer: SGS, buoyancy, a θ flux and a sponge
    rc, res = run(joinpath(ex, "dry_convective_bl", "dry_convective_bl.toml"),
                  vcat(small, ["checks.finite={}", "checks.energy_closure={}", "checks.momentum_closure={}"]))
    @test rc == 0
    fs = only(values(res["loop_fusion"]))
    who = Set(first.(split.(fs["not_fusable"], ":")))
    @test fs["substages_fused"] == "none" && fs["setting"] == "if_possible"
    @test issubset(["builtin.sgs", "builtin.buoyancy", "Components.SurfaceThetaFlux", "Components.Sponge"], who)
    @test !occursin("no documented reason", join(fs["not_fusable"]))
    # the key: never on the DNS; always on the CBL stops the run (rc ≠ 0)
    rc, res = run(joinpath(ex, "taylor_green_vortex", "taylor_green_vortex.toml"),
                  vcat(small, ["timestep.loop_fusion=\"never\""]))
    fs = only(values(res["loop_fusion"]))
    @test rc == 0 && fs["setting"] == "never" && fs["substages_fused"] == "none"
    root = mktempdir()
    rc = Aeolus.main(String["run", joinpath(ex, "dry_convective_bl", "dry_convective_bl.toml"),
                            collect(Iterators.flatten(("--set", x) for x in vcat(small, ["timestep.loop_fusion=\"always\""])))...,
                            "--root", root, "--allow-uncommitted"]; io = IOBuffer(), err = IOBuffer())
    @test rc != 0
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
    println("  [#533] step_rk3! with the fused sweep allocates $n bytes")
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
    # in-core work through a route the method scan misses)
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

# The blocks of the sweep run concurrently only with more than one thread;
# the test gate runs each file at one thread, where the threaded backends
# above run them one after another. So the concurrent case runs here in a
# -t 4 subprocess (#558): the base term, dry and moist, at 1 to 4 slabs
# times 1 to 4 row ranges, against the one-block sweep (which the sets
# above compare with the separate passes): every field within 4 eps of its
# maximum (the same expressions per point; measured 0).
const _FS_THREADED = """
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Backends: CPUBackend, with_backend
using Random: Xoshiro, randn
const FVs = Aeolus.Operators.FiniteVolume
const SIMs = Aeolus.Simulations
g = RegularGrid(; size = (20, 17, 16), extent = (200.0, 170.0, 160.0), halo = (3, 3, 3))
function mk(moist)
    s = BoussinesqSimulation(g; ν = 1e-2, moisture = moist)
    rng = Xoshiro(558)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
    moist && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    SIMs.fill_state_halos!(s)
    return s
end
function sweep!(s, ns, nt)
    terms = (FVs.BaseTendencies(s.ν),)
    bufs = FVs.sweep_buffers(nothing, s.u.data, s.has_moisture ? 5 : 4, ns, 16; nt = nt)
    s2 = s.has_moisture ? (s.qv, SIMs._Axpy(0.3), nothing) : nothing
    FVs.blocked_predictor!(bufs, terms, s.u, s.v, s.w, SIMs._Axpy(0.3), (s.θ, SIMs._Axpy(0.3), nothing), s2, g)
    return map(f -> copy(interior_view(f)), s.has_moisture ? (s.u, s.v, s.w, s.θ, s.qv) : (s.u, s.v, s.w, s.θ))
end
worst = 0.0; runs = 0
for moist in (false, true)
    ref = with_backend(() -> sweep!(mk(moist), 1, 1), CPUBackend(serial = true))
    for ns in 1:4, nt in 1:4
        ns * nt > 1 || continue
        out = with_backend(() -> sweep!(mk(moist), ns, nt), CPUBackend(threshold = 0))
        for (a, r) in zip(out, ref)
            global worst = max(worst, maximum(abs.(a .- r)) / maximum(abs, r))
        end
        global runs += 1
    end
end
print("threads ", Threads.nthreads(), " runs ", runs, " worst ", worst)
"""

@testset "fused sweep: blocks run concurrently, -t 4 (#558)" begin
    if Threads.nthreads() >= 4
        @test true            # the sets above ran the blocks concurrently
    else
        project = dirname(Base.active_project())
        cmd = `$(Base.julia_cmd()) --startup-file=no -t 4 --project=$project -e $_FS_THREADED`
        out = IOBuffer(); err = IOBuffer()
        p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
        msg = String(take!(out))
        success(p) || @info "fused_sweep.jl: the -t 4 subprocess failed" String(take!(err))
        @test success(p)
        m = match(r"threads (\d+) runs (\d+) worst ([0-9.eE+-]+)", msg)
        @test m !== nothing
        if m !== nothing
            @test parse(Int, m[1]) == 4 && parse(Int, m[2]) == 30
            @test parse(Float64, m[3]) <= 4eps()
        end
    end
end
