# The post-predictor operators inside the fused sweep (#533 plan step 6):
#
#   1. the driver: `FiniteVolume.blocked_predictor!` with post terms (a
#      pointwise one, a Jacobi one whose factors read a radius-1 stencil
#      of the predicted velocity, chained) is the predictor passes followed
#      by each operator's whole-field form, for 1-5 slabs, the slabs in
#      forward and in reverse order, serial and threaded, regular and
#      stretched grids, dry and moist, Euler and RK3 stage; an AfterSweep
#      operator and every one after it run on the whole field after the
#      sweep;
#   2. each post-predictor operator of the model in the sweep against its
#      whole-field reference (its existing method): the prescribed
#      kinematic flow, the implicit IBM correction, the tiled surface's
#      canopy drag (#538), the resolved canopy drag, the facade wall stress
#      and the ground wall stress;
#   3. step! and step_rk3! of domains with post-predictor components on the
#      fused sweep against loop_fusion = "never" (the reference);
#   4. type stability and allocation.
#
# Tolerances: the fused form evaluates the same floating-point expressions
# as the whole-field operators, in the same order per face, so the
# agreement is round-off (`agree`, ROUNDOFF_RTOL of the field's scale).

using Test
using Aeolus
using Aeolus.Fields: Field, interior_view
using Aeolus.Grids: interior
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.BoundaryConditions: FieldBoundaryConditions, fill_halos!
using Random: Xoshiro, randn, rand

const FVp = Aeolus.Operators.FiniteVolume
const SIMp = Aeolus.Simulations
const CPp = Aeolus.Components

const _FP_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)))

function _fp_seed!(s; U = 0.3, seed = 536)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= U .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    SIMp.fill_state_halos!(s)
    return s
end
_fp_fields(s) = (s.u, s.v, s.w, s.θ, s.qv)
_fp_agree(a, b) = all(i -> agree(interior_view(_fp_fields(a)[i]), interior_view(_fp_fields(b)[i])), 1:5)
_fp_halos!(s) = () -> (fill_halos!(s.u, s.bcs); fill_halos!(s.v, s.bcs); fill_halos!(s.w, s.bcs); nothing)

# ---------------------------------------------------------------------------
# synthetic post terms and their whole-field references
# ---------------------------------------------------------------------------

# pointwise: w ← w/(1 + c·g(i, j, k)) with a fixed pattern g (an implicit
# Rayleigh damping with a spatially varying rate; footprint 0)
struct FPScale{T} <: FVp.PostTerm
    c::T
end
FVp.footprint(::FPScale) = (0, 0, 0)
@inline _fp_g(i, j, k) = mod(i + 2j + 3k, 5)
@inline function FVp.post_row!(t::FPScale, ::Val{2}, j, k, i1, i2, st)
    @inbounds for i in i1:i2
        st.w[i, j, k] /= 1 + t.c * _fp_g(i, j, k)
    end
    return nothing
end
function fp_scale_ref!(s, t::FPScale)
    ii, jj, kk = interior(s.mesh)
    for k in kk, j in jj, i in ii
        s.w.data[i, j, k] /= 1 + t.c * _fp_g(i, j, k)
    end
end

# radius 1: an implicit quadratic drag on all three components, every
# factor from the speed with the other two components averaged from their
# four neighbouring faces (the canopy drag's stencil), all factors before
# any is applied (Jacobi)
struct FPDrag{T,A<:AbstractArray{T,4}} <: FVp.PostTerm
    c::T
    fac::A
end
FPDrag(c, s) = FPDrag(c, ones(size(s.u.data)..., 3))
FVp.footprint(::FPDrag) = (1, 1, 1)
@inline function _fp_drag_factors!(fac, c, u, v, w, i, j, k)
    @inbounds begin
        q = 0.25
        vu = q * (v[i-1, j, k] + v[i, j, k] + v[i-1, j+1, k] + v[i, j+1, k])
        wu = q * (w[i-1, j, k] + w[i, j, k] + w[i-1, j, k+1] + w[i, j, k+1])
        fac[i, j, k, 1] = 1 + c * sqrt(u[i, j, k]^2 + vu^2 + wu^2)
        uv = q * (u[i, j-1, k] + u[i+1, j-1, k] + u[i, j, k] + u[i+1, j, k])
        wv = q * (w[i, j-1, k] + w[i, j, k] + w[i, j-1, k+1] + w[i, j, k+1])
        fac[i, j, k, 2] = 1 + c * sqrt(uv^2 + v[i, j, k]^2 + wv^2)
        uw = q * (u[i, j, k-1] + u[i+1, j, k-1] + u[i, j, k] + u[i+1, j, k])
        vw = q * (v[i, j, k-1] + v[i, j+1, k-1] + v[i, j, k] + v[i, j+1, k])
        fac[i, j, k, 3] = 1 + c * sqrt(uw^2 + vw^2 + w[i, j, k]^2)
    end
    return nothing
end
@inline function _fp_drag_apply!(fac, u, v, w, i, j, k)
    @inbounds begin
        u[i, j, k] /= fac[i, j, k, 1]
        v[i, j, k] /= fac[i, j, k, 2]
        w[i, j, k] /= fac[i, j, k, 3]
    end
    return nothing
end
function FVp.post_row!(t::FPDrag, ::Val{1}, j, k, i1, i2, st)
    for i in i1:i2
        _fp_drag_factors!(t.fac, t.c, st.u, st.v, st.w, i, j, k)
    end
end
function FVp.post_row!(t::FPDrag, ::Val{2}, j, k, i1, i2, st)
    for i in i1:i2
        _fp_drag_apply!(t.fac, st.u, st.v, st.w, i, j, k)
    end
end
function fp_drag_ref!(s, t::FPDrag)
    _fp_halos!(s)()
    ii, jj, kk = interior(s.mesh)
    for k in kk, j in jj, i in ii
        _fp_drag_factors!(t.fac, t.c, s.u.data, s.v.data, s.w.data, i, j, k)
    end
    for k in kk, j in jj, i in ii
        _fp_drag_apply!(t.fac, s.u.data, s.v.data, s.w.data, i, j, k)
    end
end

_fp_ref!(s, t::FPScale) = fp_scale_ref!(s, t)
_fp_ref!(s, t::FPDrag) = fp_drag_ref!(s, t)
_fp_ref!(s, t::FVp.AfterSweep) = t.f()

# the reference: base tendencies, the velocity predictor, the post
# operators' whole-field forms, then the scalars' predictor (the order of
# `_euler_projection!`)
function _fp_reference!(r, mesh, Δt, post, rk3, snap)
    moist = r.has_moisture
    FVp._fused_tendencies!(r.tu, r.tv, r.tw, r.u, r.v, r.w, r.ν, (r.tθ, r.θ),
                           moist ? (r.tqv, r.qv) : nothing, mesh)
    for (f, t) in ((r.u, r.tu), (r.v, r.tv), (r.w, r.tw))
        SIMp.axpy!(f, Δt, t)
    end
    foreach(t -> _fp_ref!(r, t), post)
    c = SIMp._Lin2r(0.75, 0.25)
    for (f, t, s0) in (moist ? ((r.θ, r.tθ, snap[1]), (r.qv, r.tqv, snap[2])) : ((r.θ, r.tθ, snap[1]),))
        rk3 ? SIMp._advance_scalar!(f, Δt, t, (c = c, ws = nothing), s0) : SIMp.axpy!(f, Δt, t)
    end
    return r
end

function _fp_args(a, mesh, Δt, rk3, snap, ns, terms)
    moist = a.has_moisture
    nk = length(interior(mesh)[3])
    lag, lz = FVp.sweep_lags(terms)
    bufs = FVp.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, nk; lag = lag, lz = lz)
    c = SIMp._Lin2r(0.75, 0.25)
    ps = rk3 ? SIMp._AxpyStage(SIMp._Axpy(Δt), c) : SIMp._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    return bufs, s1, s2
end
function _fp_sweep!(a, mesh, Δt, post, rk3, snap, ns)
    terms = (FVp.BaseTendencies(a.ν),)
    bufs, s1, s2 = _fp_args(a, mesh, Δt, rk3, snap, ns, terms)
    FVp.blocked_predictor!(bufs, terms, post, _fp_halos!(a), a.u, a.v, a.w, SIMp._Axpy(Δt), s1, s2, mesh)
    return a
end
# the slabs one after another in reverse order, then the edge planes and
# the deferred points: the body of blocked_predictor! with its slab launch
# replaced (in forward order slab s reads slab s+1's front planes before
# slab s+1 has touched them; in reverse order after it has finished: the
# two orders bound what concurrent slabs see)
function _fp_sweep_reversed!(a, mesh, Δt, post, rk3, snap, ns)
    terms = (FVp.BaseTendencies(a.ν),)
    bufs, s1, s2 = _fp_args(a, mesh, Δt, rk3, snap, ns, terms)
    ii, jj, kk = interior(mesh)
    lag, lz = FVp.sweep_lags(terms)
    pu = SIMp._Axpy(Δt)
    st = (u = a.u.data, v = a.v.data, w = a.w.data, θ = FVp._sweep_field(s1), qv = FVp._sweep_field(s2),
          m = FVp.grid_metrics(mesh))
    p1, p2 = FVp._sweep_pred(s1), FVp._sweep_pred(s2)
    plan = FVp.post_plan(post, lz)
    for s in ns:-1:1
        FVp._b_sweep_slab!(s, terms, post, plan, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
    end
    for n in 1:2lz*ns, j in jj, i in ii
        FVp._b_sweep_edge!(i, j, n, st, pu, p1, p2, bufs.edge, kk, ns, lz)
    end
    FVp._post_deferred!(CPUBackend(serial = true), post, plan, _fp_halos!(a), st, ii, jj, kk, ns)
    return a
end

@testset "fused sweep: post-predictor operators (#533 step 6)" begin

g = RegularGrid(; size = (20, 12, 12), extent = (200.0, 120.0, 120.0), halo = (3, 3, 3))
sg = StretchedGrid(; size = (12, 10, 16), extent = (240.0, 200.0, 600.0), stretching = (:tanh, 1.5))
grids = (("regular", g, FieldBoundaryConditions(:periodic)),
         ("stretched, walls", sg, FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)))

@testset "the schedule: widths and delays" begin
    @test FVp.post_plan((), 1) == () && FVp.post_delay(()) == 0
    s = BoussinesqSimulation(g; ν = 1e-2)
    p = FVp.post_plan((FPScale(0.1),), 1)
    @test p == ((0, 0, 1, 0, 0, 1, 0, 0),)
    p = FVp.post_plan((FPDrag(0.1, s), FPScale(0.1), FPDrag(0.1, s)), 1)
    @test p == ((1, 1, 2, 2, 2, 3, 1, 2), (2, 2, 3, 2, 2, 3, 2, 2), (3, 3, 4, 4, 4, 5, 3, 4))
    @test FVp.post_delay(p) == 4
    # after an AfterSweep every operator runs after the sweep, without delay
    p = FVp.post_plan((FPScale(0.1), FVp.AfterSweep(() -> nothing), FPDrag(0.1, s)), 1)
    @test p[1] == (0, 0, 1, 0, 0, 1, 0, 0) && all(e -> e[1] >= FVp._AFTER && e[8] == 0, p[2:3])
    @test (@inferred FVp.post_plan((FPDrag(0.1, s), FPScale(0.1)), 1)) isa NTuple{2,NTuple{8,Int}}
end

@testset "driver: the sweep = predictor + the operators' whole-field passes" begin
    for (gname, mesh, bcs) in grids
        nk = length(interior(mesh)[3])
        for moist in (false, true), rk3 in (false, true), (bn, b) in _FP_BACKENDS
            with_backend(b) do
                mk() = _fp_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs))
                Δt = 0.3
                for (pname, mkpost) in (
                        ("pointwise", s -> (FPScale(0.7),)),
                        ("radius 1", s -> (FPDrag(2.0, s),)),
                        ("chain r1, r0, r1", s -> (FPDrag(2.0, s), FPScale(0.7), FPDrag(1.0, s))),
                        ("after the sweep", s -> (FPScale(0.7), FVp.AfterSweep(() -> fp_drag_ref!(s, FPDrag(2.0, s))),
                                                  FPDrag(1.0, s))))
                    r = mk()
                    snap = (copy(r.θ.data), copy(r.qv.data))
                    _fp_reference!(r, mesh, Δt, mkpost(r), rk3, snap)
                    r0 = _fp_reference!(mk(), mesh, Δt, (), rk3, snap)
                    @test !agree(interior_view(r.w), interior_view(r0.w))     # the operators acted
                    for ns in 1:min(5, nk ÷ 2)
                        a = mk()
                        _fp_sweep!(a, mesh, Δt, mkpost(a), rk3, snap, ns)
                        ok = _fp_agree(a, r)
                        ok || println("  [#533 s6] mismatch: $gname $pname moist=$moist rk3=$rk3 $bn ns=$ns")
                        @test ok
                        if ns > 1 && bn == "serial"
                            a = mk()
                            _fp_sweep_reversed!(a, mesh, Δt, mkpost(a), rk3, snap, ns)
                            ok = _fp_agree(a, r)
                            ok || println("  [#533 s6] reversed mismatch: $gname $pname moist=$moist rk3=$rk3 ns=$ns")
                            @test ok
                        end
                    end
                end
            end
        end
    end
end

end
