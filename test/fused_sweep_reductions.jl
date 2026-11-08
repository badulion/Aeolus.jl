# The fused sweep, plan step 3 of #533: terms that need a reduction of the
# stage-start state. The large-scale forcing component (`LargeScale`:
# geostrophic pressure gradient, advective tendencies, nudging, subsidence
# in both forms, reference damping) and the urban site (`UrbanSite`: the
# captures of the fixed-gradient top's and the open faces' heat, the open
# faces' θ relaxation) compute their reductions (plane means, plane sums)
# in `prepare!` and add per-level column coefficients inside the sweep.
#
#   1. the column coefficients (`LargeScaleForcing.column_forcing!` +
#      `add_column_rows!`) give `apply_forcing!`'s tendency for every
#      forcing, and the same records (`tendency_integrals`,
#      `damping_tendency`);
#   2. the `LargeScale` term inside `blocked_predictor!` = the base
#      tendencies, its whole-field `run_stage!` and the predictor, for 1-5
#      slabs (forward and reversed), serial and threaded, regular and
#      stretched, dry and moist, Euler and the RK3 combination;
#   3. step! and step_rk3! of a domain with `LargeScale`: fused
#      (`loop_fusion = "always"`) = one loop per term (`"never"`), both
#      clocks; the term is type-stable and the fused step does not allocate
#      per point;
#   4. the urban site: its term's `prepare!` records what its `:tendencies`
#      stage records; the open faces' relaxation in the column form is the
#      `apply_forcing!` tendency.
#
# Tolerances: the column form sums the forcings' per-level contributions
# before adding them and writes the upwind difference as cp (φ[k+1] − φ[k])
# instead of −w (φ[ku] − φ[kl]) / Δz, so it differs from the passes by
# round-off of the tendencies (≈ 1e-16 relative); the states are compared
# with `agree` (ROUNDOFF_RTOL = 1e-10 of the field's scale), the records
# (sums over the plane instead of point by point) with rtol 1e-10 of their
# scale.

using Test
using Aeolus
using Aeolus.Fields: interior_view
using Aeolus.Grids: interior
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Components: Domain, CentredAdvection, LargeScale
using Aeolus.LargeScaleForcing: ForcingTable, GeostrophicPressureGradient, AdvectiveTendencies, ProfileNudging,
                                NudgingWeight, Subsidence, ReferenceDamping, apply_forcing!, tendency_integrals,
                                damping_tendency, ColumnForcing, column_forcing!, column_forcing_supported,
                                reset_column_forcing!, finish_column_forcing!, add_column_rows!
using Aeolus.Operators: sponge_profile
using Aeolus.Config: load_run_config
using Random: Xoshiro, randn

const FVr = Aeolus.Operators.FiniteVolume
const SIMr = Aeolus.Simulations
const CPr = Aeolus.Components

const _FR_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)))

function _fr_seed!(s; U = 0.3, seed = 5333)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= U .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.u) .+= 2.0                      # a mean wind to nudge and subside
    # a stable θ profile with noise: plane means vary with height
    m = s.mesh
    θ = interior_view(s.θ)
    for k in axes(θ, 3)
        θ[:, :, k] .= 300 .+ 0.01 .* k .* (m.Nz > 0 ? 1 : 0) .+ 0.1 .* randn(rng, size(θ, 1), size(θ, 2))
    end
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    return s
end
# a 3-D θ target (a Field), which has no column form
_fr_field3(s) = (f = Aeolus.Fields.Field(Aeolus.Grids.CellCenter(), s.mesh); fill!(f.data, 300.0); f)
_fr_state(s) = (s.u, s.v, s.w, s.p, s.θ, s.qv)
_fr_agree(a, b) = all(i -> agree(interior_view(_fr_state(a)[i]), interior_view(_fr_state(b)[i])),
                      eachindex(_fr_state(a))) && a.t ≈ b.t

# Every forcing of LargeScaleForcing, on every field it takes: time-
# dependent tables and functions, a weight, both subsidence forms, damping
# toward a table, the slab mean, a number and a level vector
function _fr_forcings(mesh, moist; H = mesh isa StretchedGrid ? 600.0 : 120.0)
    zt = [0.0, H / 2, H]
    tab(a, b) = ForcingTable(zt, [0.0, 100.0], [a b; 1.3a 0.7b; 0.5a 1.1b])
    geo = GeostrophicPressureGradient(tab(5.0, 6.0), (z, t) -> -1.0 + 0.01z + 1e-3t)
    adv = AdvectiveTendencies(; θ = tab(1e-4, -2e-4), u = (z, t) -> 1e-4 * sin(z / H), v = 2e-5,
                              q = moist ? tab(1e-8, 3e-8) : nothing)
    nud = ProfileNudging(moist ? (u = 4.0, v = tab(1.0, 2.0), θ = (z, t) -> 300 + 0.012z, qv = 0.0079) :
                                 (u = 4.0, v = tab(1.0, 2.0), θ = (z, t) -> 300 + 0.012z), 600.0;
                         weight = NudgingWeight(H / 4; depth = H / 4))
    sub_mean = Subsidence(tab(-0.01, -0.02))                    # θ (and qv when moist)
    sub_local = Subsidence((z, t) -> 0.02 * cos(2π * z / H); fields = (:θ, :u), form = :local)   # both signs
    sp = sponge_profile(0.01, 0.6H, H)
    damp = (ReferenceDamping(:θ, sp, tab(300.5, 301.0)), ReferenceDamping(:u, sp, 3.0),
            ReferenceDamping(:w, sp, 0.0), ReferenceDamping(:v, sp, fill(0.5, mesh.Nz)),
            ReferenceDamping(:θ, sponge_profile(0.02, 0.8H, H), :mean))   # two on θ: they combine
    moist && (damp = (damp..., ReferenceDamping(:qv, sp, :mean)))
    return geo, (adv, nud, sub_mean, sub_local, damp...)
end

g = RegularGrid(; size = (20, 12, 12), extent = (200.0, 120.0, 120.0), halo = (3, 3, 3))
sg = StretchedGrid(; size = (12, 10, 16), extent = (240.0, 200.0, 600.0), stretching = (:tanh, 1.5))
grids = (("regular", g, FieldBoundaryConditions(:periodic)),
         ("stretched, walls", sg, FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)))

# the tendency the column coefficients add, on every interior point
# (`add_column_rows!` row by row into a one-row buffer)
function _fr_column_tendency(col, s)
    ii, jj, kk = interior(s.mesh)
    st = (u = s.u.data, v = s.v.data, w = s.w.data, θ = s.θ.data, qv = s.has_moisture ? s.qv.data : nothing)
    out = [zeros(size(s.u.data)) for _ in 1:5]
    buf = zeros(size(s.u.data, 1), 1, 1, 5, 1)
    for k in kk, j in jj
        fill!(buf, 0.0)
        add_column_rows!(col, buf, 1, 1, 1, j, k, ii, st)
        for f in 1:5, i in ii
            out[f][i, j, k] = buf[i, 1, 1, f, 1]
        end
    end
    return out
end

# the same through the point kernel (`add_column_point!`, the per-point form's) into
# tendency fields, for the comparison of the two forms (integration of #533 steps 3 and 8)
function _fr_column_point_tendency(col, s)
    ii, jj, kk = interior(s.mesh)
    st = (u = s.u.data, v = s.v.data, w = s.w.data, θ = s.θ.data, qv = s.has_moisture ? s.qv.data : nothing)
    out = [zeros(size(s.u.data)) for _ in 1:5]
    tf = FVr.TendencyFields(out...)
    for k in kk, j in jj, i in ii
        Aeolus.LargeScaleForcing.add_column_point!(col, FVr.SweepRow(tf, j, k, 1), i, j, k, st)
    end
    return out
end

_fr_rec(F) = F isa ReferenceDamping ? (copy(damping_tendency(F).levels), damping_tendency(F).integral) :
             F isa Union{ProfileNudging,Subsidence} ? tendency_integrals(F) : nothing
# The records are sums over the plane: their round-off scale is that of
# the summed terms, not of the result. A damping toward the slab mean has
# records that are zero up to round-off of λ N |φ| (N the columns of a
# level), measured 1e-13 point by point and exactly 0 from the plane sums;
# so the absolute tolerance is 1e-12 of λ_max N |φ|_max (λ ≤ 0.03 s⁻¹ here,
# |φ| the field's largest value), integrated over the volume for the
# integrals; the other records agree within rtol 1e-10.
_fr_scale(s, F) = 0.03 * s.mesh.Nx * s.mesh.Ny * maximum(abs, interior_view(getproperty(s, F.field)))
_fr_rec_agree(s, F, a::Nothing, b::Nothing) = true
_fr_rec_agree(s, F, a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) &&
    all(k -> isapprox(a[k], b[k]; rtol = 1e-10, atol = 1e-10 * maximum(abs, values(a); init = 0.0) + 1e-300), keys(a))
function _fr_rec_agree(s, F, a::Tuple, b::Tuple)
    sc = 1e-12 * _fr_scale(s, F)
    return agree(a[1], b[1]; atol = sc) && isapprox(a[2], b[2]; rtol = 1e-10, atol = sc * s.mesh.Lx * s.mesh.Ly * s.mesh.Lz /
                                                                                      (s.mesh.Nx * s.mesh.Ny))
end

@testset "fused sweep, plan step 3: reductions of the stage-start state (#533)" begin

@testset "column form = apply_forcing!: every forcing, its records ($gname, moist = $moist)" for
        (gname, mesh, bcs) in grids, moist in (false, true)
    s = _fr_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs))
    SIMr.fill_state_halos!(s)
    geo, extra = _fr_forcings(mesh, moist)
    t, Δt, f = 30.0, 2.0, 1.2e-4
    col = ColumnForcing()
    for F in (geo, extra...)
        @test column_forcing_supported(F)
        # the reference: apply_forcing! into zeroed tendency fields
        foreach(x -> fill!(x.data, 0.0), (s.tu, s.tv, s.tw, s.tθ, s.tqv))
        F isa GeostrophicPressureGradient ? apply_forcing!(s, F, t, Δt; f = f) : apply_forcing!(s, F, t, Δt)
        ref = (copy(s.tu.data), copy(s.tv.data), copy(s.tw.data), copy(s.tθ.data), copy(s.tqv.data))
        rrec = _fr_rec(F)
        reset_column_forcing!(col, s)
        foreach(x -> fill!(x.data, NaN), (s.tu, s.tv, s.tw, s.tθ, s.tqv))
        F isa GeostrophicPressureGradient ? column_forcing!(col, s, F, t, Δt; f = f) : column_forcing!(col, s, F, t, Δt)
        finish_column_forcing!(col)
        @test all(x -> all(isnan, x.data), (s.tu, s.tv, s.tw, s.tθ, s.tqv))     # nothing written
        got = _fr_column_tendency(col, s)
        ii, jj, kk = interior(mesh)
        ok = all(fi -> agree(view(got[fi], ii, jj, kk), view(ref[fi], ii, jj, kk);
                             atol = 1e-13 * maximum(abs, view(ref[fi], ii, jj, kk); init = 0.0)), 1:5)
        ok || println("  [#533 s3] column mismatch: $(nameof(typeof(F))) $gname moist=$moist")
        @test ok
        @test any(fi -> any(!iszero, view(ref[fi], ii, jj, kk)), 1:5)     # the forcing acted
        @test _fr_rec_agree(s, F, rrec, _fr_rec(F))
    end
    # all at once: the contributions sum per level
    foreach(x -> fill!(x.data, 0.0), (s.tu, s.tv, s.tw, s.tθ, s.tqv))
    apply_forcing!(s, geo, t, Δt; f = f); apply_forcing!(s, extra, t, Δt)
    reset_column_forcing!(col, s)
    column_forcing!(col, s, geo, t, Δt; f = f); column_forcing!(col, s, extra, t, Δt)
    finish_column_forcing!(col)
    got = _fr_column_tendency(col, s)
    ii, jj, kk = interior(mesh)
    for (fi, x) in enumerate((s.tu, s.tv, s.tw, s.tθ, s.tqv))
        @test agree(view(got[fi], ii, jj, kk), view(x.data, ii, jj, kk);
                    atol = 1e-13 * maximum(abs, view(x.data, ii, jj, kk); init = 0.0))
    end
    # what has no column form: a damping toward a 3-D target; a dry qv target is refused as by apply_forcing!
    @test !column_forcing_supported(ReferenceDamping(:θ, sponge_profile(0.01, 60.0, 120.0), _fr_field3(s)))
    @test !column_forcing_supported((geo, ReferenceDamping(:θ, sponge_profile(0.01, 60.0, 120.0), _fr_field3(s).data)))
    if !moist
        @test_throws ArgumentError column_forcing!(col, s, ProfileNudging((qv = 0.0,), 10.0), t, Δt)
    end
end

# the whole-field reference of a sweep with the LargeScale term: the base
# tendencies, the component's run_stage!, then the predictors
function _fr_reference!(d, Δt, rk3, snap)
    r = d.state
    moist = r.has_moisture
    FVr._fused_tendencies!(r.tu, r.tv, r.tw, r.u, r.v, r.w, r.ν, (r.tθ, r.θ),
                           moist ? (r.tqv, r.qv) : nothing, r.mesh)
    CPr.run_stage!(d.components[2], d, Val(:body_force), Δt)
    for (f, t) in ((r.u, r.tu), (r.v, r.tv), (r.w, r.tw))
        SIMr.axpy!(f, Δt, t)
    end
    c = SIMr._Lin2r(0.75, 0.25)
    for (f, t, s0) in (moist ? ((r.θ, r.tθ, snap[1]), (r.qv, r.tqv, snap[2])) : ((r.θ, r.tθ, snap[1]),))
        rk3 ? SIMr._advance_scalar!(f, Δt, t, (c = c, ws = nothing), s0) : SIMr.axpy!(f, Δt, t)
    end
    return r
end

# the sweep over (base, the domain's hook terms), the slabs forward
# (blocked_predictor!) or one after another in reverse order (the front edge)
function _fr_sweep!(d, Δt, rk3, snap, ns; reversed = false)
    a = d.state
    moist = a.has_moisture
    terms = (FVr.BaseTendencies(a.ν), SIMr.hook_terms(CPr.core_hooks(d).body_force!, a)...)
    ii, jj, kk = interior(a.mesh)
    lag, lz = FVr.sweep_lags(terms)
    bufs = FVr.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, length(kk); lag = lag, lz = lz)
    foreach(t -> FVr.prepare!(t, a, Δt), terms)
    c = SIMr._Lin2r(0.75, 0.25)
    ps = rk3 ? SIMr._AxpyStage(SIMr._Axpy(Δt), c) : SIMr._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    if !reversed
        FVr.blocked_predictor!(bufs, terms, a.u, a.v, a.w, SIMr._Axpy(Δt), s1, s2, a.mesh)
    else
        pu = SIMr._Axpy(Δt)
        st = (u = a.u.data, v = a.v.data, w = a.w.data, θ = FVr._sweep_field(s1), qv = FVr._sweep_field(s2),
              m = FVr.grid_metrics(a.mesh))
        p1, p2 = FVr._sweep_pred(s1), FVr._sweep_pred(s2)
        for s in ns:-1:1
            FVr._b_sweep_slab!(s, terms, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
        end
        for n in 1:2lz*ns, j in jj, i in ii
            FVr._b_sweep_edge!(i, j, n, st, pu, p1, p2, bufs.edge, kk, ns, lz)
        end
    end
    return terms
end

_fr_domain(s, geo, extra; clock = :step) =
    (d0 = Domain(s; validate = false);
     Domain(d0; components = (CentredAdvection(), LargeScale(geo, extra; clock = clock, t0 = 40.0))))

@testset "LargeScale term in the sweep = its run_stage! + predictor ($gname)" for (gname, mesh, bcs) in grids
    nk = length(interior(mesh)[3])
    for moist in (false, true), rk3 in (false, true), (bn, b) in _FR_BACKENDS
        with_backend(b) do
            geo, extra = _fr_forcings(mesh, moist)
            mk() = (s = _fr_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs));
                    SIMr.fill_state_halos!(s); _fr_domain(s, geo, extra))
            Δt = 0.3
            r = mk()
            snap = (copy(r.state.θ.data), copy(r.state.qv.data))
            _fr_reference!(r, Δt, rk3, snap)
            rec = map(_fr_rec, extra)
            # the forcing changes the result (the comparison is not of the base alone)
            r0 = mk()
            _fr_reference!(Domain(r0; components = (CentredAdvection(), LargeScale(nothing, ()))), Δt, rk3, snap)
            @test !agree(interior_view(r.state.u), interior_view(r0.state.u)) &&
                  !agree(interior_view(r.state.θ), interior_view(r0.state.θ))
            for ns in 1:min(5, nk ÷ 2), reversed in (false, true)
                reversed && (ns == 1 || bn != "serial") && continue
                a = mk()
                foreach(t -> fill!(t.data, NaN), (a.state.tu, a.state.tv, a.state.tw, a.state.tθ, a.state.tqv))
                terms = _fr_sweep!(a, Δt, rk3, snap, ns; reversed = reversed)
                @test length(terms) == 2 && terms[2] isa CPr.LargeScaleTerm
                ok = all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                         ((a.state.u, r.state.u), (a.state.v, r.state.v), (a.state.w, r.state.w),
                          (a.state.θ, r.state.θ), (a.state.qv, r.state.qv)))
                ok || println("  [#533 s3] LargeScale sweep mismatch: $gname moist=$moist rk3=$rk3 $bn ns=$ns reversed=$reversed")
                @test ok
                @test all(t -> all(isnan, t.data), (a.state.tu, a.state.tv, a.state.tw, a.state.tθ, a.state.tqv))
                @test all(i -> _fr_rec_agree(a.state, extra[i], rec[i], _fr_rec(a.components[2].extra[i])), eachindex(rec))
            end
        end
    end
end

@testset "step! and step_rk3! with LargeScale: fused = one loop per term (clock = $clk)" for clk in (:step, :core)
    for (gname, mesh, bcs) in grids, moist in (false, true), (drv, scheme) in ((step!, :euler), (step_rk3!, :rk3))
        geo, extra = _fr_forcings(mesh, moist)
        mk(setting) = (s = _fr_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs));
                       SIMr.set_loop_fusion!(s, setting);
                       d0 = Domain(s; scheme = scheme, validate = false);
                       Domain(d0; components = (CentredAdvection(), LargeScale(geo, extra; clock = clk))))
        a = mk(:always); r = mk(:never)
        p = mk(:always); p.state.workspaces.fused.per_point = true   # the per-point form of step 8
        fill!(a.state.tu.data, NaN)
        for _ in 1:3
            step!(a, 0.5); step!(r, 0.5); step!(p, 0.5)
        end
        ok = _fr_agree(a.state, r.state)
        ok || println("  [#533 s3] step mismatch: $gname moist=$moist $scheme clock=$clk")
        @test ok
        okp = _fr_agree(p.state, r.state)
        okp || println("  [#533 s3+s8] per-point mismatch: $gname moist=$moist $scheme clock=$clk")
        @test okp && CPr.sweep_report(p)["substages_fused"] == "all"
        @test all(isnan, a.state.tu.data)                  # fused: the tendency fields unused
        nsub = scheme === :euler ? 3 : 9
        ra = CPr.sweep_report(a)
        @test ra["substages_fused"] == "all" && a.state.workspaces.fused.fused == nsub
        @test CPr.sweep_report(r)["substages_fused"] == "none"
        @test all(i -> _fr_rec_agree(r.state, extra[i], _fr_rec(r.components[2].extra[i]), _fr_rec(a.components[2].extra[i])),
                  eachindex(extra))
    end
end

# the fused RK3 step of a domain with every forcing on `mesh`, after warm-up
function _fr_alloc_domain(mesh)
    s = _fr_seed!(BoussinesqSimulation(mesh; ν = 1e-2))
    geo, extra = _fr_forcings(mesh, false)
    d = Domain(Domain(s; scheme = :rk3, validate = false);
               components = (CentredAdvection(), LargeScale(geo, extra)))
    for _ in 1:3
        step!(d, 0.2)
    end
    return d
end

@testset "LargeScale term: type-stable, no allocation per level, row or point; unsupported forcing reported" begin
    d = _fr_alloc_domain(g)
    h = CPr.core_hooks(d)
    T = typeof(FVr.sweep_term(d.components[2], d))
    @test T <: CPr.LargeScaleTerm
    @test (@inferred SIMr.hook_terms(h.body_force!, d.state)) isa Tuple{T}
    n = @allocated step!(d, 0.2)
    # the same case with twice the levels and more columns: an allocation per
    # level (the first version of prepare! boxed a Float64 per level), per row
    # or per point would make the two differ; the forcings' records are
    # NamedTuples behind Refs (as apply_forcing!'s), a fixed number of small
    # allocations per substage, so the counts agree (64 B: allocator rounding)
    g2 = RegularGrid(; size = (24, 16, 24), extent = (200.0, 120.0, 120.0), halo = (3, 3, 3))
    d2 = _fr_alloc_domain(g2)
    n2 = @allocated step!(d2, 0.2)
    println("  [#533 s3] step! (RK3) with LargeScale, fused, allocates $n bytes at 20×12×12, $n2 bytes at 24×16×24")
    @test abs(n2 - n) <= 64
    @test n < 20_000
    @test CPr.sweep_report(d)["substages_fused"] == "all"
    @test CPr.sweep_report(d2)["substages_fused"] == "all"
    # a damping toward a 3-D target has no column form: one loop per term, reported
    s3 = _fr_seed!(BoussinesqSimulation(g; ν = 1e-2))
    d3 = Domain(Domain(s3; validate = false);
                components = (CentredAdvection(),
                              LargeScale(nothing, (ReferenceDamping(:θ, sponge_profile(0.01, 60.0, 120.0), _fr_field3(s3)),))))
    @test FVr.sweep_term(d3.components[2], d3) === nothing
    step!(d3, 0.2)
    @test CPr.sweep_report(d3)["substages_fused"] == "none"
    @test any(startswith("Components.LargeScale:"), CPr.sweep_report(d3)["not_fusable"])
end


@testset "urban site term: the captures of its :tendencies stage, the open faces' θ relaxation ($lat)" for
        lat in ("periodic", "open")
    Aeolus.Operators.set_fftw_policy!(; planner = :estimate)
    UCr = Aeolus.Urban
    ov = lat == "open" ? ["urban.lateral=\"open\""] : String[]
    cfg = load_run_config(joinpath(@__DIR__, "data", "urban", "small_city.toml"); overrides = ov)
    case = redirect_stderr(devnull) do
        UCr.build_case(UCr.site_options(cfg))
    end
    d = case.domain; sim = case.sim; m = sim.mesh
    redirect_stderr(devnull) do
        step!(case.model)                    # a developed state and SGS diffusivities
    end
    SIMr.fill_state_halos!(sim)
    dt = 0.5
    t = FVr.sweep_term(case, d)
    @test t isa UCr.UrbanSiteTerm && FVr.footprint(t) == (0, 0, 0)
    @test (t.col === nothing) == (lat == "periodic")
    # the reference: the stage, its θ relaxation into the zeroed tendency
    fill!(sim.tθ.data, 0.0)
    CPr.run_stage!(case, d, Val(:tendencies), dt)
    tθ_ref = copy(sim.tθ.data)
    snap(ob) = ob === nothing ? nothing : map(copy, (ob.lev, ob.clev, ob.rate, ob.crate, ob.relax, [ob.rlat, ob.rtop]))
    top_ref = case.top === nothing ? nothing : case.top.rate
    ob_ref = snap(case.open)
    # the term's prepare!: the same records, nothing written to the tendency
    fill!(sim.tθ.data, NaN)
    case.top === nothing || (case.top.rate = NaN)
    FVr.prepare!(t, sim, dt)
    @test all(isnan, sim.tθ.data)
    if lat == "periodic"
        @test case.top !== nothing && isapprox(case.top.rate, top_ref; rtol = 1e-12)
        @test all(iszero, tθ_ref)            # no tendency: the stage only records
    else
        ob = case.open
        got = snap(ob)
        # every capture but the relaxation is the same computation (round-off: rtol 1e-12);
        # the relaxation's levels come from the plane sums (scale λ_max N |θ| of each level)
        @test all(i -> agree(got[i], ob_ref[i]; rtol = 1e-12), (1, 2, 3, 4, 6))
        V = m.dx * m.dy * m.dz
        sc = 1e-12 * maximum(ob.heat) * sim.sponge_α_max * m.Nx * m.Ny * maximum(interior_view(sim.θ)) * V / ob.A
        @test any(!iszero, ob_ref[5]) && agree(got[5], ob_ref[5]; atol = sc)
        # the relaxation the sweep adds = what the stage added to tθ
        tend = _fr_column_tendency(t.col, sim)[4]
        ii, jj, kk = interior(m)
        @test any(!iszero, view(tθ_ref, ii, jj, kk))
        @test agree(view(tend, ii, jj, kk), view(tθ_ref, ii, jj, kk);
                    atol = 1e-13 * maximum(abs, view(tθ_ref, ii, jj, kk)))
        @test all(f -> all(iszero, _fr_column_tendency(t.col, sim)[f]), (1, 2, 3, 5))
        # the point kernel of the per-point form adds the same (same expression: round-off)
        tp = _fr_column_point_tendency(t.col, sim)
        @test all(f -> agree(view(tp[f], ii, jj, kk), view(_fr_column_tendency(t.col, sim)[f], ii, jj, kk);
                             atol = 1e-13 * maximum(abs, view(tθ_ref, ii, jj, kk))), 1:5)
    end
end

end
