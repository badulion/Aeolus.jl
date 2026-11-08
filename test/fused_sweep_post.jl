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
using Aeolus.IBM: SolidMask, mark_solid!, implicit_ibm_correction!, ibm_correction_term
using Aeolus.PlantCanopy: ResolvedCanopyDrag, apply_canopy_drag!, canopy_drag_term, canopy_drag_force
using Aeolus.WallModels: RoughLogLaw, WallModel, immersed_walls, apply_immersed_wall_stress!, immersed_wall_term,
                         apply_wall_stress!, wall_stress_term
using Aeolus.Radiation: SlabLand
using Aeolus.SurfaceTiles: SurfaceMosaic
using Aeolus.Anelastic: isentropic_reference
using Aeolus.Config: load_run_config
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
_fp_ref!(s, f::Function) = f()          # an operator's whole-field reference

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
    foreach(t -> FVp.prepare!(t, a, Δt), post)
    FVp.blocked_predictor!(bufs, terms, post, _fp_halos!(a), a.u, a.v, a.w, SIMp._Axpy(Δt), s1, s2, mesh)
    foreach(t -> FVp.finish!(t, a, Δt), post)
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
    foreach(t -> FVp.prepare!(t, a, Δt), post)
    for s in ns:-1:1
        FVp._b_sweep_slab!(s, terms, post, plan, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
    end
    for n in 1:2lz*ns, j in jj, i in ii
        FVp._b_sweep_edge!(i, j, n, st, pu, p1, p2, bufs.edge, kk, ns, lz)
    end
    FVp._post_deferred!(CPUBackend(serial = true), post, plan, _fp_halos!(a), st, ii, jj, kk, ns)
    foreach(t -> FVp.finish!(t, a, Δt), post)
    return a
end


# ---------------------------------------------------------------------------
# the model's post-predictor operators: a synthetic setup for each
# ---------------------------------------------------------------------------

# two buildings, one across the periodic x boundary; the first air level
# of every column (the ground wall's level mask, roofs included)
function _fp_buildings(mesh)
    sm = SolidMask(mesh)
    mark_solid!(sm, (x, y, z) -> (40 < x < 80 && 20 < y < 70 && z < 60) ||
                                 ((x > 170 || x < 20) && 90 < y < 130 && z < 35))
    m = mesh
    lev = [something(findfirst(k -> !sm.mask[m.Hx+i, m.Hy+j, m.Hz+k], 1:m.Nz), m.Nz + 1) for i in 1:m.Nx, j in 1:m.Ny]
    return sm, lev
end
# an elevated, patchy crown (levels 9-14, so the sweep reaches it inside
# its slabs as well as on their edges)
function _fp_lad(mesh)
    rng = Xoshiro(22)
    lad = zeros(mesh.Nx, mesh.Ny, mesh.Nz)
    for k in 9:14, j in 1:mesh.Ny, i in 1:mesh.Nx
        (i + j) % 3 == 0 || (lad[i, j, k] = 0.5 + rand(rng))
    end
    return lad
end
# a tiled surface (one street tile per column, forced at level 3), its
# drag replaced by a random one over 14 levels
function _fp_tiles(sim)
    m = sim.mesh
    cols = Matrix{Union{Nothing,Aeolus.Urban.ParentColumn}}(nothing, m.Nx, m.Ny)
    for I in CartesianIndices((m.Nx, m.Ny))
        st = SlabLand(; T = 308.0, C_s = 1e6, albedo = 0.12, emissivity = 0.95, z0m = 0.5, β = 0.0)
        cols[I] = Aeolus.Urban.ParentColumn(SurfaceMosaic([st], [1.0]), UInt8[1], [0.0], [3], 3)
    end
    z = zeros(m.Nx, m.Ny)
    pl = Aeolus.Urban.ParentLanduse(m.Nx, m.Ny, m.dx, 0.0, 0.0, 1, zeros(Int, m.Nx, m.Ny, 5), z, z, z, String[])
    ps = Aeolus.Urban.ParentSurface(pl, zeros(m.Nx, m.Ny, 5), z, z, z, z, z,
                                    [m.z0 + (k - 0.5) * m.dz for k in 1:m.Nz], nothing, cols)
    d0 = CPp.Domain(sim; validate = false)
    ts = CPp.build(Aeolus.Urban.TiledUrbanSurface,
                   (; surface = ps, period = 60.0, u_min = 0.5, reference = isentropic_reference(m; θ_const = 300.0)), d0)
    ts.drag = 0.05 .* rand(Xoshiro(538), m.Nx, m.Ny, 14)
    ts.drag_den = ones(m.Nx, m.Ny, 14, 2)
    return ts, CPp.Domain(sim; components = (CPp.CentredAdvection(), ts), validate = false)
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

@testset "row pointers: the entries of a row, and of an x-range of it by bisection" begin
    # rows 1..5 of a list sorted by row, then x (repeated x and empty rows included)
    rows = [1, 1, 1, 3, 3, 3, 3, 3, 4, 5, 5]
    xs = [2, 5, 5, 1, 1, 4, 7, 9, 3, 6, 6]
    ptr = FVp.row_index(e -> rows[e], length(rows), 5)
    @test [collect(FVp.row_entries(ptr, r)) for r in 1:5] == [[1, 2, 3], Int[], [4, 5, 6, 7, 8], [9], [10, 11]]
    @test FVp.row_index(e -> reverse(rows)[e], length(rows), 5) === nothing
    @test FVp.rows_sorted(ptr, e -> xs[e]) && !FVp.rows_sorted(ptr, e -> -xs[e])
    for r in 1:5, I1 in 0:10, I2 in I1-1:10
        @test FVp.row_entries(ptr, r, e -> xs[e], I1, I2) == filter(e -> I1 <= xs[e] <= I2, FVp.row_entries(ptr, r))
    end
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


# ---------------------------------------------------------------------------
# 2. each operator of the model inside the sweep = its whole-field method
# ---------------------------------------------------------------------------

gU = RegularGrid(; size = (20, 16, 24), extent = (200.0, 160.0, 120.0), halo = (3, 3, 3))
ugrids = (("periodic", FieldBoundaryConditions(:periodic)),
          ("walls", FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)))

# Each setup: (name, mk(sim, Δt) -> (post terms, reference closure, diagnostics closure)).
# The diagnostics closure returns what the operator records besides the
# velocity (forces, wall statistics, u*), compared with `agree` as well.
function _fp_setups(sm, lev, lad)
    wind = 3.0 .+ rand(Xoshiro(7), gU.Nx, gU.Ny)
    height = fill(gU.dz, gU.Nx, gU.Ny)
    return (
        ("kinematic flow", (s, Δt) -> begin
            pu, pv, pw = (0.2 .* randn(Xoshiro(n), size(s.u.data)) for n in 1:3)
            ((CPp.KinematicFlowTerm(pu, pv, pw),),
             () -> (copyto!(s.u.data, pu); copyto!(s.v.data, pv); copyto!(s.w.data, pw)), () -> ())
        end),
        ("IBM, τ = 0", (s, Δt) -> ((ibm_correction_term(sm; dt = Δt),),
                                   () -> implicit_ibm_correction!(s, sm; dt = Δt), () -> ())),
        ("IBM, τ = 2 s (each face scaled once per solid cell)",
         (s, Δt) -> ((ibm_correction_term(sm; τ = 2.0, dt = Δt),),
                     () -> implicit_ibm_correction!(s, sm; τ = 2.0, dt = Δt), () -> ())),
        ("tiled surface's canopy drag (#538)", (s, Δt) -> begin
            ts, d = _fp_tiles(s)
            ((FVp.post_sweep_term(ts, d, Δt),), () -> CPp.run_stage!(ts, d, Val(:post_predictor), Δt), () -> ())
        end),
        ("resolved canopy drag", (s, Δt) -> begin
            dr = ResolvedCanopyDrag(gU, lad; C_d = 0.2)
            F = Ref{Any}(nothing)
            ((canopy_drag_term(dr, s, Δt),), () -> (F[] = apply_canopy_drag!(s, dr, Δt)),
             () -> (F[] === nothing ? Tuple(canopy_drag_force(dr, s.mesh, Δt)) : Tuple(F[])))
        end),
        ("facade wall stress", (s, Δt) -> begin
            iw = immersed_walls(s, sm, RoughLogLaw(0.05))
            ((immersed_wall_term(iw, Δt),), () -> apply_immersed_wall_stress!(s, iw, Δt),
             () -> (iw.stats[:F]..., iw.stats[:τA], iw.stats[:area], iw.stats[:τ_min]))
        end),
        ("ground wall stress, first level", (s, Δt) -> begin
            wm = WallModel(RoughLogLaw(0.1))
            t = wall_stress_term(s, wm, Δt)
            F = Ref{Any}(nothing)
            ((t,), () -> (F[] = apply_wall_stress!(s, wm, Δt)),
             () -> (F[] === nothing ? t.F[] : Tuple(F[])))
        end),
        ("ground wall stress, roofs, matching wind, u* out", (s, Δt) -> begin
            wm = WallModel(RoughLogLaw(0.1))
            us, wo = zeros(gU.Nx, gU.Ny), zeros(gU.Nx, gU.Ny)
            kw = (mask = lev, wind = wind, height = height, ustar = us, wind_out = wo)
            t = wall_stress_term(s, wm, Δt; kw...)
            F = Ref{Any}(nothing)
            ((t,), () -> (F[] = apply_wall_stress!(s, wm, Δt; kw...)),
             () -> (F[] === nothing ? t.F[] : Tuple(F[]), copy(us), copy(wo)))
        end),
        ("urban chain: IBM, facades, ground, canopy", (s, Δt) -> begin
            iw = immersed_walls(s, sm, RoughLogLaw(0.05))
            wm = WallModel(RoughLogLaw(0.1))
            dr = ResolvedCanopyDrag(gU, lad; C_d = 0.2)
            ((ibm_correction_term(sm; dt = Δt), immersed_wall_term(iw, Δt), wall_stress_term(s, wm, Δt; mask = lev),
              canopy_drag_term(dr, s, Δt)),
             () -> (implicit_ibm_correction!(s, sm; dt = Δt); apply_immersed_wall_stress!(s, iw, Δt);
                    apply_wall_stress!(s, wm, Δt; mask = lev); apply_canopy_drag!(s, dr, Δt)),
             () -> (iw.stats[:τA], iw.stats[:F]...))
        end),
    )
end

_fp_diag_agree(a, b) = all(i -> agree(collect(Float64, Iterators.flatten(a[i])), collect(Float64, Iterators.flatten(b[i]))),
                           eachindex(a))

@testset "$oname: inside the sweep = its whole-field method" for (oname, _) in _fp_setups(_fp_buildings(gU)..., _fp_lad(gU))
    sm, lev = _fp_buildings(gU)
    setup = Dict(_fp_setups(sm, lev, _fp_lad(gU)))[oname]
    for (bname, bcs) in ugrids, rk3 in (false, true), (bn, b) in _FP_BACKENDS
        with_backend(b) do
            mk() = _fp_seed!(BoussinesqSimulation(gU; ν = 1e-2, bcs = bcs); U = 2.0)
            Δt = 0.5
            r = mk()
            snap = (copy(r.θ.data), copy(r.qv.data))
            post, ref, diag = setup(r, Δt)
            _fp_reference!(r, gU, Δt, (ref,), rk3, snap)
            dr = diag()
            r0 = _fp_reference!(mk(), gU, Δt, (), rk3, snap)
            @test !agree(interior_view(r.u), interior_view(r0.u)) || !agree(interior_view(r.w), interior_view(r0.w))
            for ns in 1:5
                for rev in (false, true)
                    (rev && (ns == 1 || bn != "serial")) && continue
                    a = mk()
                    post, ref, diag = setup(a, Δt)
                    (rev ? _fp_sweep_reversed! : _fp_sweep!)(a, gU, Δt, post, rk3, snap, ns)
                    ok = _fp_agree(a, r) && _fp_diag_agree(diag(), dr)
                    ok || println("  [#533 s6] $oname: mismatch $bname rk3=$rk3 $bn ns=$ns reversed=$rev")
                    @test ok
                end
            end
        end
    end
end


# ---------------------------------------------------------------------------
# 3. components: the hooks' post terms, and steps on the fused sweep
# ---------------------------------------------------------------------------

@testset "components: a domain with post-predictor components steps fused = one loop per term" begin
    g2 = RegularGrid(; size = (20, 16, 12), extent = (200.0, 160.0, 60.0), halo = (3, 3, 3))
    for scheme in (:euler, :rk3), what in (:kinematic, :tiles)
        mk() = begin
            s = _fp_seed!(BoussinesqSimulation(g2; ν = 1e-2); U = 2.0)
            d0 = CPp.Domain(s; scheme = scheme, validate = false)
            c = what === :kinematic ?
                CPp.build(CPp.KinematicFlow, (; kind = :solid_body_rotation, omega = 0.01, center = (100.0, 80.0),
                                              velocity = (0.0, 0.0, 0.0)), d0) : first(_fp_tiles(s))
            CPp.Domain(d0; components = (CPp.CentredAdvection(), c))
        end
        a = mk(); r = mk()
        SIMp.set_loop_fusion!(a.state, :always); SIMp.set_loop_fusion!(r.state, :never)
        h = CPp.core_hooks(a)
        post = SIMp.post_hook_terms(h.post_predictor!, a.state, 0.5)
        # the pointwise prescribed flow on the sweep's rows; the tiles' drag (footprint 1)
        # runs its whole-field stage after the sweep (the measured choice of _in_rows_or_after)
        @test post isa Tuple{what === :kinematic ? CPp.KinematicFlowTerm : FVp.AfterSweep}
        what === :tiles && @test FVp.post_sweep_term(last(a.components), a, 0.5) isa Aeolus.Urban.TiledDragTerm
        fill!(a.state.tu.data, NaN)
        for _ in 1:3
            step!(a, 0.5); step!(r, 0.5)
        end
        ok = _fp_agree(a.state, r.state)
        ok || println("  [#533 s6] domain mismatch: $what $scheme")
        @test ok
        @test all(isnan, a.state.tu.data)            # fused: the tendency fields unused
        ra = CPp.sweep_report(a)
        @test ra["substages_fused"] == "all" && isempty(ra["not_fusable"])
        @test a.state.workspaces.fused.fused == (scheme === :rk3 ? 9 : 3)
    end
    # a plain post_predictor! function still keeps one loop per term
    s = _fp_seed!(BoussinesqSimulation(g2; ν = 1e-2))
    step!(s, 0.5; post_predictor! = (sim, Δt) -> nothing)
    @test SIMp.sweep_report(s)["not_fusable"] == ["builtin.post_predictor: " * SIMp.sweep_reason("builtin.post_predictor")]
end

@testset "SurfaceLayer: the wall stress in the sweep, with its substage stress sums" begin
    g2 = RegularGrid(; size = (16, 12, 12), extent = (160.0, 120.0, 120.0), halo = (3, 3, 3))
    mk() = begin
        s = _fp_seed!(BoussinesqSimulation(g2; ν = 1e-2, θ₀ = 300.0,
                                           bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)); U = 2.0)
        interior_view(s.u) .+= 4.0
        d0 = CPp.Domain(s; validate = false)
        sl = CPp.build(CPp.SurfaceLayer, (; model = :most, z0m = 0.1, z0h = nothing,
                                        temperature = Dict{String,Any}("kind" => "constant", "value" => 302.0),
                                        stability = Dict{String,Any}("family" => "businger_dyer_beljaars_holtslag"),
                                        rho_cp = 1206.0), d0)
        d = CPp.Domain(d0; components = (CPp.CentredAdvection(), sl))
        CPp.run_stage!(sl, d, Val(:surface_flux), 1.0)
        (s, sl, d)
    end
    (r, slr, dr), (a, sla, da) = mk(), mk()
    post = SIMp.post_hook_terms(CPp.core_hooks(da).post_predictor!, a, 0.5)
    @test post isa Tuple{FVp.AfterSweep}                      # the domain runs it after the sweep (footprint 1) ...
    @test FVp.post_sweep_term(sla, da, 0.5) isa CPp.SurfaceLayerPostTerm   # ... its row form, checked here
    snap = (copy(r.θ.data), copy(r.qv.data))
    for n in 1:2
        SIMp.fill_state_halos!(r); SIMp.fill_state_halos!(a)      # step 1 of a substage (the operators refill some halos)
        _fp_reference!(r, g2, 0.5, (() -> CPp.run_stage!(slr, dr, Val(:post_predictor), 0.5),), false, snap)
        _fp_sweep!(a, g2, 0.5, (FVp.post_sweep_term(sla, da, 0.5),), false, snap, 2)
        @test _fp_agree(a, r)
        @test agree(slr.τ, sla.τ) && slr.nsub[] == sla.nsub[] == n && slr.τ[1] != 0
    end
    # the domain is not fused for the surface layer's heat flux (:tendencies), only
    rep = (step!(da, 0.5); CPp.sweep_report(da))
    @test !any(startswith("builtin.post_predictor"), rep["not_fusable"])
    @test any(startswith("Components.SurfaceLayer:"), rep["not_fusable"])
end

@testset "post hook: no row form built for a component that runs after the sweep; budgets reported once" begin
    # review of #533 step 6: the domain built the SurfaceLayer's row form (wall_stress_term: three
    # Nx × Ny arrays and the face lists) every substage, then ran its run_stage! instead
    g2 = RegularGrid(; size = (64, 48, 12), extent = (640.0, 480.0, 120.0), halo = (3, 3, 3))
    s = _fp_seed!(BoussinesqSimulation(g2; ν = 1e-2, θ₀ = 300.0,
                                       bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)); U = 2.0)
    d0 = CPp.Domain(s; validate = false)
    sl = CPp.build(CPp.SurfaceLayer, (; model = :most, z0m = 0.1, z0h = nothing,
                                    temperature = Dict{String,Any}("kind" => "constant", "value" => 302.0),
                                    stability = Dict{String,Any}("family" => "businger_dyer_beljaars_holtslag"),
                                    rho_cp = 1206.0), d0)
    d = CPp.Domain(d0; components = (CPp.CentredAdvection(), sl))
    CPp.run_stage!(sl, d, Val(:surface_flux), 1.0)
    h = CPp.core_hooks(d).post_predictor!
    @test CPp.post_after_sweep(sl)
    SIMp.post_hook_terms(h, s, 0.5); SIMp.post_hook_blockers(h, s)
    plane = 8 * g2.Nx * g2.Ny                  # bytes of one Float64 Nx × Ny array
    @test (@allocated FVp.post_sweep_term(sl, d, 0.5)) > plane      # the row form costs planes ...
    @test (@allocated SIMp.post_hook_terms(h, s, 0.5)) < plane      # ... the hook builds none
    @test (@allocated SIMp.post_hook_blockers(h, s)) < plane
    @test SIMp.post_hook_terms(h, s, 0.5) isa Tuple{FVp.AfterSweep}
    # with budgets on, both hooks name them: the report lists them once
    s2 = _fp_seed!(BoussinesqSimulation(g2; ν = 1e-2, θ₀ = 300.0))
    d2 = CPp.Domain(s2; validate = false)
    kf = CPp.build(CPp.KinematicFlow, (; kind = :uniform, omega = 0.0, center = (0.0, 0.0), velocity = (1.0, 0.5, 0.0)), d2)
    db = CPp.with_budgets(CPp.Domain(d2; components = (CPp.CentredAdvection(), kf)), (:energy,))
    step!(db, 0.5)
    nf = CPp.sweep_report(db)["not_fusable"]
    @test count(startswith("builtin.budgets:"), nf) == 1
    @test allunique(nf)
end

@testset "the urban site's couplings: post terms, the sweep = the site's post-predictor hook" begin
    cfg = load_run_config(joinpath(@__DIR__, "data", "urban", "small_city.toml");
                          overrides = ["fftw.planner=\"estimate\""])
    m = Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model
    d = only(m.domains)
    case = Aeolus.Urban.urban_case(d)
    step!(m, 1.0)                                  # the ground's matching-input filter is primed after a step
    names = [CPp.component_name(c) for c in d.components if CPp.attaches(c, :post_predictor)]
    @test issubset([:ibm, :wall_model_facades, :surface_exchange], names)
    Δt = 0.5
    post = CPp._post_stage_terms(d.components, d, Δt)
    @test post isa Tuple && length(post) == length(names)
    # every coupling has its row form; the domain keeps the pointwise IBM correction on the
    # rows and runs the ones with a stencil after the sweep (_in_rows_or_after)
    rows = Tuple(FVp.post_sweep_term(c, d, Δt) for c in d.components if CPp.attaches(c, :post_predictor))
    @test !any(t -> t isa FVp.AfterSweep, rows)
    @test any(t -> t isa Aeolus.IBM.IBMCorrectionTerm, rows) &&
          any(t -> t isa Aeolus.WallModels.ImmersedWallTerm, rows) &&
          any(t -> t isa Aeolus.Urban.GroundStressTerm, rows)
    @test count(t -> t isa FVp.AfterSweep, post) == count(t -> FVp.footprint(t) != (0, 0, 0), rows)
    # with the budgets on (always, at the site) the hook is not fused: plan step 7
    @test SIMp.post_hook_terms(case.post_predictor, case.sim, Δt) === nothing
    s = case.sim
    SIMp.fill_state_halos!(s)
    save = deepcopy((s.u.data, s.v.data, s.w.data, s.θ.data, s.qv.data))
    restore!() = foreach((x, y) -> copyto!(x, y), (s.u.data, s.v.data, s.w.data, s.θ.data, s.qv.data), save)
    snap = (copy(s.θ.data), copy(s.qv.data))
    _fp_reference!(s, s.mesh, Δt, (() -> case.post_predictor(s, Δt),), false, snap)
    ref = deepcopy((s.u, s.v, s.w, s.θ, s.qv)); τA = case.couplings[findfirst(c -> c isa Aeolus.Urban.WallModelFacades,
                                                                                    case.couplings)].iw.stats[:τA]
    ustar = copy(case.bus.u_star)
    r0 = (restore!(); _fp_reference!(s, s.mesh, Δt, (), false, snap); interior_view(s.u) |> copy)
    @test !agree(r0, interior_view(ref[1]))                     # the couplings acted
    for ns in (1, 3), terms in (rows, post)
        restore!()
        _fp_sweep!(s, s.mesh, Δt, terms, false, snap, ns)
        @test all(i -> agree(interior_view((s.u, s.v, s.w, s.θ, s.qv)[i]), interior_view(ref[i])), 1:5)
        @test agree(case.couplings[findfirst(c -> c isa Aeolus.Urban.WallModelFacades, case.couplings)].iw.stats[:τA], τA)
        @test agree(case.bus.u_star, ustar)
    end
end


# ---------------------------------------------------------------------------
# 4. type stability and allocation
# ---------------------------------------------------------------------------

_fp_call(bufs, terms, post, h, s, s1, g) =
    FVp.blocked_predictor!(bufs, terms, post, h, s.u, s.v, s.w, SIMp._Axpy(1e-3), s1, nothing, g)
function _fp_allocs(bufs, terms, post, h, s, s1, g)
    _fp_call(bufs, terms, post, h, s, s1, g)
    n = typemax(Int)
    for _ in 1:3
        n = min(n, @allocated _fp_call(bufs, terms, post, h, s, s1, g))
    end
    return n
end

@testset "type stability and allocation" begin
    sm, lev = _fp_buildings(gU)
    for (bn, b) in _FP_BACKENDS
        with_backend(b) do
            s = _fp_seed!(BoussinesqSimulation(gU; ν = 1e-2); U = 2.0)
            iw = immersed_walls(s, sm, RoughLogLaw(0.05))
            dr = ResolvedCanopyDrag(gU, _fp_lad(gU); C_d = 0.2)
            post = (ibm_correction_term(sm; dt = 0.1), immersed_wall_term(iw, 0.1),
                    wall_stress_term(s, WallModel(RoughLogLaw(0.1)), 0.1; mask = lev), canopy_drag_term(dr, s, 0.1),
                    FPDrag(1.0, s), FPScale(0.5))
            terms = (FVp.BaseTendencies(s.ν),)
            bufs = FVp.sweep_buffers(nothing, s.u.data, 4, 3, gU.Nz)
            s1 = (s.θ, SIMp._Axpy(1e-3), nothing)
            h = SIMp._VelocityHalos(s)
            n = _fp_allocs(bufs, terms, post, h, s, s1, gU)
            println("  [#533 s6] blocked_predictor! with six post terms allocates $n bytes ($bn)")
            # serial: nothing; threaded: the task closures of the launches (the slab launch, the
            # edge launch, two per post term after the sweep) and the halo fills, none per point
            @test bn == "serial" ? n == 0 : n <= 64_000
            @test (@inferred FVp.post_plan(post, 1)) isa NTuple{6,NTuple{8,Int}}
        end
    end
    g2 = RegularGrid(; size = (20, 16, 12), extent = (200.0, 160.0, 60.0), halo = (3, 3, 3))
    s = _fp_seed!(BoussinesqSimulation(g2; ν = 1e-2))
    d0 = CPp.Domain(s; validate = false)
    kf = CPp.build(CPp.KinematicFlow, (; kind = :uniform, omega = 0.0, center = (0.0, 0.0), velocity = (1.0, 0.5, 0.0)), d0)
    d = CPp.Domain(d0; components = (CPp.CentredAdvection(), kf))
    h = CPp.core_hooks(d).post_predictor!
    @test (@inferred SIMp.post_hook_terms(h, s, 0.5)) isa Tuple{CPp.KinematicFlowTerm{Array{Float64,3}}}
    @test (@inferred SIMp._post_terms(s, nothing, 0.5)) === ()
end

end
