# The pointwise terms of the fused sweep (#533 plan step 2): the core's
# buoyancy (dry and moist), Coriolis, surface flux and sponge, and the
# components Coriolis, SurfaceThetaFlux, Sponge, Fringe, the urban site's
# GeostrophicCoupling and MoistBuoyancy and the slab's floor flux of
# SimulationCouplings.
#
#   1. each term inside `FiniteVolume.blocked_predictor!` gives the base
#      tendencies + its whole-field reference (the `add_*!` operator or the
#      component's `run_stage!`, which stay) + the predictor, for 1-5 slabs
#      (also in reverse slab order), serial and threaded, regular and
#      stretched grids, dry and moist, with and without the RK3 stage
#      combination;
#   2. step! and step_rk3! with every built-in pointwise term on run fully
#      fused (loop_fusion = "always") and agree with one loop per term
#      ("never"); likewise a domain with the pointwise components;
#   3. the recorded diagnostics of the geostrophic coupling agree with the
#      ones its run_stage! records;
#   4. type stability and allocation of the built-in plan and the sweep.
#
# Tolerances: the terms evaluate the expressions of their whole-field
# kernels in the same order, so the agreement is round-off (`agree`, the
# ROUNDOFF_RTOL of the field's scale). The geostrophic coupling's mean
# tendencies are the level means of what is added in the sweep and a
# difference of two tendency-field means in run_stage!: they agree to the
# round-off of those means (rtol 1e-9 of the forcing).

using Test
using Aeolus
using Aeolus.Fields: Field, interior_view
using Aeolus.Grids: interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Components: Domain, CentredAdvection
using Random: Xoshiro, randn

const _FP_FV = Aeolus.Operators.FiniteVolume
const _FP_SIM = Aeolus.Simulations
const _FP_CP = Aeolus.Components
const _FP_OPS = Aeolus.Operators
const _FP_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)))

function _fp_seed!(s; U = 0.3, seed = 5332)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= U .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    return s
end
_fp_rand!(f, a, b; seed = 1) = (interior_view(f) .= a .+ b .* randn(Xoshiro(seed), size(interior_view(f))); f)
_fp_fields(s) = (s.u, s.v, s.w, s.θ, s.qv)
_fp_agree(a, b) = all(i -> agree(interior_view(_fp_fields(a)[i]), interior_view(_fp_fields(b)[i])), 1:5)

# the reference: base tendencies, the terms' whole-field passes, predictor
function _fp_reference!(r, Δt, ref!, rk3, snap)
    moist = r.has_moisture
    _FP_FV._fused_tendencies!(r.tu, r.tv, r.tw, r.u, r.v, r.w, r.ν, (r.tθ, r.θ),
                              moist ? (r.tqv, r.qv) : nothing, r.mesh)
    ref!(r, Δt)
    for (f, t) in ((r.u, r.tu), (r.v, r.tv), (r.w, r.tw))
        _FP_SIM.axpy!(f, Δt, t)
    end
    c = _FP_SIM._Lin2r(0.75, 0.25)
    for (f, t, s0) in (moist ? ((r.θ, r.tθ, snap[1]), (r.qv, r.tqv, snap[2])) : ((r.θ, r.tθ, snap[1]),))
        rk3 ? _FP_SIM._advance_scalar!(f, Δt, t, (c = c, ws = nothing), s0) : _FP_SIM.axpy!(f, Δt, t)
    end
    return r
end

# the sweep over (base, extra...), slabs launched or run in reverse order
function _fp_sweep!(a, Δt, extra, rk3, snap, ns; reversed = false)
    moist = a.has_moisture
    mesh = a.mesh
    ii, jj, kk = interior(mesh)
    terms = (_FP_FV.BaseTendencies(a.ν), extra...)
    lag, lz = _FP_FV.sweep_lags(terms)
    bufs = _FP_FV.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, length(kk); lag = lag, lz = lz)
    foreach(t -> _FP_FV.prepare!(t, a, Δt), terms)
    c = _FP_SIM._Lin2r(0.75, 0.25)
    ps = rk3 ? _FP_SIM._AxpyStage(_FP_SIM._Axpy(Δt), c) : _FP_SIM._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    pu = _FP_SIM._Axpy(Δt)
    if !reversed
        _FP_FV.blocked_predictor!(bufs, terms, a.u, a.v, a.w, pu, s1, s2, mesh)
    else
        st = (u = a.u.data, v = a.v.data, w = a.w.data, θ = _FP_FV._sweep_field(s1), qv = _FP_FV._sweep_field(s2),
              m = _FP_FV.grid_metrics(mesh))
        p1, p2 = _FP_FV._sweep_pred(s1), _FP_FV._sweep_pred(s2)
        for s in ns:-1:1
            _FP_FV._b_sweep_slab!(s, terms, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
        end
        for n in 1:2lz*ns, j in jj, i in ii
            _FP_FV._b_sweep_edge!(i, j, n, st, pu, p1, p2, bufs.edge, kk, ns, lz)
        end
    end
    return a
end

# the same terms in the per-point (GPU) form (#533 step 8) on KernelAbstractions'
# CPU backend: their point kernels (`add_tendency!`), written to the
# tendency fields, then the predictor (integration of steps 2 and 8)
function _fp_point!(a, Δt, extra, rk3, snap)
    moist = a.has_moisture
    terms = (_FP_FV.BaseTendencies(a.ν), extra...)
    foreach(t -> _FP_FV.prepare!(t, a, Δt), terms)
    c = _FP_SIM._Lin2r(0.75, 0.25)
    ps = rk3 ? _FP_SIM._AxpyStage(_FP_SIM._Axpy(Δt), c) : _FP_SIM._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    tf = _FP_FV.TendencyFields(a.tu.data, a.tv.data, a.tw.data, a.tθ.data, moist ? a.tqv.data : nothing)
    _FP_FV.pointwise_predictor!(Aeolus.Backends.ka_backend(CPUBackend()), tf, terms, a.u, a.v, a.w,
                                _FP_SIM._Axpy(Δt), s1, s2, a.mesh)
    return a
end

# Every grid × moisture × RK3 × backend × slab count: the sweep over
# `terms(a)` (built on the simulation `a`) agrees with the passes `ref!`;
# and the terms change the result (the comparison is not of the base alone).
function _fp_check(name, grids, terms, ref!; moists = (false, true), kw = NamedTuple(), setup! = (s -> s))
    nfail = 0
    for (gname, mesh, bcs) in grids
        nk = length(interior(mesh)[3])
        for moist in moists, rk3 in (false, true), (bn, b) in _FP_BACKENDS
            with_backend(b) do
                mk() = (s = setup!(_fp_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs, kw...)));
                        _FP_SIM.fill_state_halos!(s); s)
                Δt = 0.3
                r = mk()
                snap = (copy(r.θ.data), copy(r.qv.data))
                _fp_reference!(r, Δt, ref!, rk3, snap)
                r0 = _fp_reference!(mk(), Δt, (s, Δt) -> nothing, rk3, snap)
                @test !_fp_agree(r, r0)
                for ns in 1:min(5, nk ÷ 2), reversed in (false, true)
                    (reversed && (ns == 1 || bn != "serial")) && continue
                    a = mk()
                    foreach(t -> fill!(t.data, NaN), (a.tu, a.tv, a.tw, a.tθ, a.tqv))   # neither read nor written
                    _fp_sweep!(a, Δt, terms(a), rk3, snap, ns; reversed = reversed)
                    ok = _fp_agree(a, r) && all(t -> all(isnan, t.data), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                    ok || (nfail += 1; println("  [#533 step 2] $name mismatch: $gname moist=$moist rk3=$rk3 $bn ns=$ns reversed=$reversed"))
                    @test ok
                end
                # the per-point form of the same terms (their point kernels)
                p = mk()
                _fp_point!(p, Δt, terms(p), rk3, snap)
                ok = _fp_agree(p, r)
                ok || (nfail += 1; println("  [#533 s2×s8] $name per-point mismatch: $gname moist=$moist rk3=$rk3 $bn"))
                @test ok
            end
        end
    end
    return nfail
end

@testset "fused sweep: pointwise terms (#533 plan step 2)" begin

g = RegularGrid(; size = (20, 12, 12), extent = (200.0, 120.0, 120.0), halo = (3, 3, 3))
sg = StretchedGrid(; size = (12, 10, 16), extent = (240.0, 200.0, 600.0), stretching = (:tanh, 1.5))
wallbcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)
grids = (("regular", g, FieldBoundaryConditions(:periodic)), ("stretched, walls", sg, wallbcs))

@testset "the built-in terms: buoyancy, Coriolis, surface flux, sponge" begin
    @test _FP_FV.footprint(_FP_FV.BuoyancyTerm(300.0, 9.81 / 300)) == (0, 0, 1)
    @test _FP_FV.footprint(_FP_FV.CoriolisTerm(1e-4)) == (1, 1, 0)
    @test _FP_FV.sweep_lags((_FP_FV.BaseTendencies(1.0), _FP_FV.CoriolisTerm(1e-4),
                             _FP_FV.BuoyancyTerm(300.0, 0.03))) == (1, 1)
    θ₀, gr, f, Fb, Ft = 300.0, 9.81, 1e-3, 0.2, 0.05
    # the sponge toward a 3-D target for u, a level vector for v and a number for w
    function targets(s)
        tu = _fp_rand!(Field(FaceX(), s.mesh), 0.2, 0.05; seed = 7)
        tv = collect(range(-0.1, 0.1; length = s.mesh.Nz))
        return (u = tu, v = tv, w = 0.01)
    end
    tg = Dict{Any,Any}()
    tgt(s) = get!(() -> targets(s), tg, s.mesh)
    zs(s) = s.mesh.z0 + 0.4 * s.mesh.Lz
    terms(s) = (_FP_FV.BuoyancyTerm(θ₀, gr / θ₀), _FP_FV.CoriolisTerm(f),
                _FP_FV.SurfaceFluxTerm(Fb, Ft, s.mesh),
                _FP_FV.sponge_term(0.02, zs(s), tgt(s), s.mesh))
    function ref!(s, Δt)
        s.has_moisture ? _FP_OPS.add_buoyancy!(s.tw, s.θ, s.qv, θ₀, gr) : _FP_OPS.add_buoyancy!(s.tw, s.θ, θ₀, gr)
        _FP_OPS.add_coriolis!(s.tu, s.tv, s.u, s.v, f)
        _FP_OPS.add_surface_flux!(s.tθ, Fb, Ft)
        _FP_SIM.add_velocity_sponge!(s, 0.02, zs(s), tgt(s))
    end
    @test _fp_check("built-in", grids, terms, ref!) == 0
    # each one alone (the moist buoyancy, and the switch of OptionalTerm)
    @test _fp_check("buoyancy", grids, s -> (_FP_FV.BuoyancyTerm(θ₀, gr / θ₀),),
                    (s, Δt) -> s.has_moisture ? _FP_OPS.add_buoyancy!(s.tw, s.θ, s.qv, θ₀, gr) :
                                                _FP_OPS.add_buoyancy!(s.tw, s.θ, θ₀, gr)) == 0
    @test _fp_check("optional off", grids[1:1],
                    s -> (_FP_FV.OptionalTerm(false, _FP_FV.CoriolisTerm(f)), _FP_FV.OptionalTerm(true, _FP_FV.CoriolisTerm(2f))),
                    (s, Δt) -> _FP_OPS.add_coriolis!(s.tu, s.tv, s.u, s.v, 2f); moists = (false,)) == 0
    # the sponge's damping heights: the level heights of the whole-field kernel
    for (_, mesh, _) in grids, loc in (FaceX(), FaceY(), FaceZ(), CellCenter())
        @test _FP_FV.damping_levels(mesh, loc) == _FP_FV.damping_coordinates(mesh, loc)    # exact: the same values, copied
    end
end

@testset "the fringe: damping along x toward 3-D targets" begin
    fr(s) = Aeolus.Transition.FringeRegion(x_start = 0.5 * s.mesh.Lx, x_end = s.mesh.Lx,
                                          rise = 0.2 * s.mesh.Lx, fall = 0.1 * s.mesh.Lx, λ_max = 0.4)
    tg = Dict{Any,Any}()
    tgt(s) = get!(tg, s.mesh) do
        (u = _fp_rand!(Field(FaceX(), s.mesh), 1.0, 0.1; seed = 11), v = _fp_rand!(Field(FaceY(), s.mesh), 0.0, 0.1; seed = 12),
         w = _fp_rand!(Field(FaceZ(), s.mesh), 0.0, 0.1; seed = 13))
    end
    @test _fp_check("fringe", grids,
                    s -> (_FP_FV.DampingTerm(fr(s).profile, 1, (Aeolus.Grids.xnodes(s.mesh, FaceX()), Aeolus.Grids.xnodes(s.mesh, FaceY()),
                                                                Aeolus.Grids.xnodes(s.mesh, FaceZ())), tgt(s), s.mesh),),
                    (s, Δt) -> Aeolus.Transition.add_fringe!(s.tu, s.tv, s.tw, s.u, s.v, s.w, fr(s), tgt(s));
                    moists = (false,)) == 0
    @test_throws ArgumentError _FP_FV.DampingTerm(fr((mesh = g,)).profile, 2, (1:2, 1:2, 1:2), (0, 0, 0), g)
end

# the urban site's geostrophic coupling, built directly (its build needs a case)
function _fp_geo(sim, U, V; sponge_ref = false)
    F = Aeolus.LargeScaleForcing.GeostrophicPressureGradient(U, V)
    m = sim.mesh
    return Aeolus.Urban.GeostrophicCoupling(F, :pressure_gradient, 0.0, 0.0, sponge_ref,
                                            sponge_ref ? Field(FaceX(), m) : nothing, sponge_ref ? Field(FaceY(), m) : nothing,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zeros(m.Nz), zeros(m.Nz))
end
_fp_geo_diag(c) = (c.U_g, c.V_g, c.dtu, c.dtv, c.dtu_sp, c.dtv_sp, c.ū, c.v̄)

@testset "urban: GeostrophicCoupling (pressure gradient, sponge retargeted)" begin
    kw = (f_coriolis = 1e-3, sponge_α_max = 0.02, sponge_z_start = 60.0)
    Ug(z, t) = 5.0 + 0.01z + 0.1t
    Vg(z, t) = -1.0 + 0.002z
    for sponge_ref in (false, true)
        cs = Dict{Any,Any}()
        geo(s) = get!(() -> _fp_geo(s, Ug, Vg; sponge_ref = sponge_ref), cs, objectid(s))
        @test _fp_check("geostrophic sponge_ref=$sponge_ref", grids[1:1],
                        s -> (Aeolus.Operators.FiniteVolume.sweep_term(geo(s), (state = s,)),),
                        (s, Δt) -> Aeolus.Urban.run_stage!(geo(s), (sim = s,), Val(:body_force), Δt); kw = kw) == 0
    end
    # constant winds on the stretched grid
    cs = Dict{Any,Any}()
    geo2(s) = get!(() -> _fp_geo(s, 4.0, 1.0; sponge_ref = true), cs, objectid(s))
    kws = (f_coriolis = 1e-3, sponge_α_max = 0.02, sponge_z_start = 300.0)
    @test _fp_check("geostrophic stretched", grids[2:2], s -> (_FP_FV.sweep_term(geo2(s), (state = s,)),),
                    (s, Δt) -> Aeolus.Urban.run_stage!(geo2(s), (sim = s,), Val(:body_force), Δt); kw = kws,
                    moists = (false,)) == 0
    # the recorded diagnostics: the same as run_stage!'s, within the round-off of the means
    s = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, kw...)); _FP_SIM.fill_state_halos!(s)
    s.t = 100.0
    cr = _fp_geo(s, Ug, Vg; sponge_ref = true); ca = _fp_geo(s, Ug, Vg; sponge_ref = true)
    fill!(s.tu.data, 0.3); fill!(s.tv.data, -0.2)
    Aeolus.Urban.run_stage!(cr, (sim = s,), Val(:body_force), 0.5)
    _FP_FV.prepare!(_FP_FV.sweep_term(ca, (state = s,)), s, 0.5)
    dr, da = _fp_geo_diag(cr), _fp_geo_diag(ca)
    scale = 1e-3 * 10                                  # f × |U_g|: the forcing's size
    @test dr[1] == da[1] && dr[2] == da[2]             # exact: the same level means, the same code
    @test all(i -> isapprox(da[i], dr[i]; rtol = 1e-9, atol = 1e-9 * scale), 3:6)
    @test da[7] == dr[7] && da[8] == dr[8]             # exact: the same mean of the same field
    @test da[5] > 0                                    # the retargeting acted (U_g > 0)
    # the nudging mode has no sweep term (plan step 3), and is reported with its reason
    cn = _fp_geo(s, 1.0, 0.0)
    cn = Aeolus.Urban.GeostrophicCoupling(Aeolus.LargeScaleForcing.GeostrophicNudging(1.0, 0.0, 1800.0), :nudging,
                                          1.0, 0.0, false, nothing, nothing, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          zeros(g.Nz), zeros(g.Nz))
    @test _FP_FV.sweep_term(cn, (state = s,)) === nothing
    @test occursin("nudging", _FP_SIM.sweep_reason(typeof(cn)))
end

@testset "urban: MoistBuoyancy (virtual temperature and condensate loading)" begin
    for (gname, mesh, bcs) in grids
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        qvf = _fp_rand!(Field(CellCenter(), mesh), 0.008, 1e-3; seed = 21)
        qlf = _fp_rand!(Field(CellCenter(), mesh), 2e-4, 5e-5; seed = 22)
        fluid = [!(i <= 3 && j <= 3 && k <= 2) for i in 1:Nx, j in 1:Ny, k in 1:Nz]   # a block of solid cells
        V = typeof(interior_view(qvf))
        mkc() = Aeolus.Urban.MoistBuoyancy(zeros(Nx, Ny, Nz), fluid, 300.0, 9.81, V[interior_view(qlf)])
        cs = Dict{Any,Any}()
        mb(s) = get!(mkc, cs, objectid(s))
        @test _fp_check("moist buoyancy, $gname", ((gname, mesh, bcs),),
                        s -> (Aeolus.Urban.moist_buoyancy_sweep(mb(s), s.mesh, interior_view(qvf)),),
                        function (s, Δt)
                            c = mb(s)
                            moist_buoyancy_perturbation!(c.Δ, interior_view(s.θ), interior_view(qvf), c.qc, c.fluid)
                            add_moist_buoyancy!(s.tw, c.Δ, c.θ₀, c.g)
                        end; moists = (false,)) == 0
    end
end

@testset "SimulationCouplings: the slab's floor flux" begin
    SD = Aeolus.SimulationDriver
    land = Aeolus.Radiation.SlabLand(; T = 300.0)
    slab = SD.SlabSurface(land, 0.01, 295.0, 1200.0, 0.0, 1.0, 0.0, 0.07, 0.0)
    c = SD.SimulationCouplings(SD.SimCouplings(0.0, slab, nothing, nothing, nothing, nothing, nothing, 0.0))
    @test _fp_check("slab floor flux", grids[2:2], s -> (_FP_FV.sweep_term(c, (state = s,)),),
                    (s, Δt) -> _FP_CP.run_stage!(c, (state = s,), Val(:tendencies), Δt); moists = (false,)) == 0
    # with obstacles the IBM penalty is the sweep term of :body_force (#533
    # step 5; the old expectation that it keeps the couplings out of the sweep
    # was made outdated by the integration of steps 2 and 5)
    mask = Aeolus.IBM.SolidMask(g)
    co = SD.SimulationCouplings(SD.SimCouplings(0.0, slab, nothing, nothing, nothing, nothing, mask, 0.0))
    dd = (state = BoussinesqSimulation(g),)
    @test first(_FP_CP.stage_sweep_term(co, dd, Val(:tendencies))) isa _FP_FV.SurfaceFluxTerm
    @test first(_FP_CP.stage_sweep_term(co, dd, Val(:body_force))) isa Aeolus.IBM.IBMPenaltyTerm
end

@testset "step! and step_rk3!: every built-in pointwise term fused = one loop per term" begin
    cases = (
        ("regular, dry", () -> BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-3,
                                                      θ_flux_bottom = 0.1, θ_flux_top = 0.02,
                                                      sponge_α_max = 0.01, sponge_z_start = 70.0)),
        ("regular, moist, open-top style level-vector target",
         () -> BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-3, moisture = true,
                                    sponge_α_max = 0.01, sponge_z_start = 70.0,
                                    sponge_target = (u = collect(range(0.0, 1.0; length = g.Nz)), v = 0.5))),
        ("stretched, walls", () -> BoussinesqSimulation(sg; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-3,
                                                          θ_flux_bottom = 0.1, sponge_α_max = 0.01,
                                                          sponge_z_start = 400.0, bcs = wallbcs)),
    )
    for (name, mk) in cases, (bn, b) in (_FP_BACKENDS..., ("default", CPUBackend())), drv in (step!, step_rk3!)
        with_backend(b) do
            a = _fp_seed!(mk()); r = _fp_seed!(mk())
            _FP_SIM.set_loop_fusion!(a, :always); _FP_SIM.set_loop_fusion!(r, :never)
            fill!(a.tu.data, NaN)
            for _ in 1:3
                drv(a, 0.5); drv(r, 0.5)
            end
            ok = _fp_agree(a, r) && agree(interior_view(a.p), interior_view(r.p))
            ok || println("  [#533 step 2] step mismatch: $name $bn $drv")
            @test ok
            @test all(isnan, a.tu.data)                    # fused: the tendency fields unused
            @test _FP_SIM.sweep_report(a)["substages_fused"] == "all"
        end
    end
    # the per-point (GPU) form of #533 step 8 with the same built-in terms:
    # their point kernels, inside OptionalTerm (integration of steps 2 and 8)
    for (name, mk) in cases, drv in (step!, step_rk3!)
        a = _fp_seed!(mk()); r = _fp_seed!(mk())
        a.workspaces.fused.per_point = true
        _FP_SIM.set_loop_fusion!(a, :always); _FP_SIM.set_loop_fusion!(r, :never)
        for _ in 1:3
            drv(a, 0.5); drv(r, 0.5)
        end
        ok = _fp_agree(a, r) && agree(interior_view(a.p), interior_view(r.p))
        ok || println("  [#533 s2×s8] per-point step mismatch: $name $drv")
        @test ok && _FP_SIM.sweep_report(a)["substages_fused"] == "all"
    end
    # a sponge target replaced on the simulation is seen by the next substage
    a = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, sponge_α_max = 0.05, sponge_z_start = 60.0))
    r = deepcopy(a); _FP_SIM.set_loop_fusion!(r, :never)
    step!(a, 0.5); step!(r, 0.5)
    a.sponge_target = r.sponge_target = (u = 2.0, v = -1.0, w = 0)
    step!(a, 0.5); step!(r, 0.5)
    @test _fp_agree(a, r)
    @test a.workspaces.sponge.targets.c[1] == 2.0        # exact: the target, copied
end

@testset "components: Coriolis, SurfaceThetaFlux, Sponge, Fringe in the fused sweep" begin
    fr = Aeolus.Transition.FringeRegion(x_start = 100.0, x_end = 200.0, rise = 40.0, fall = 20.0, λ_max = 0.3)
    function comps(s, which)
        m = s.mesh
        if which === :rest
            return (_FP_CP.Coriolis(1e-3), _FP_CP.SurfaceThetaFlux(0.1, 0.02),
                    _FP_CP.Sponge(0.02, 70.0, _FP_SIM._sponge_target(nothing, m, s.bcs, true)))
        elseif which === :mean
            ut, vt = zeros(m.Nz), zeros(m.Nz)
            return (_FP_CP.Coriolis(1e-3), _FP_CP.Sponge(0.02, 70.0, (u = ut, v = vt, w = 0), :mean, ut, vt))
        else
            tg = (u = _fp_rand!(Field(FaceX(), m), 1.0, 0.1; seed = 31), v = Field(FaceY(), m), w = Field(FaceZ(), m))
            # a target that moves with time: refreshed before every substage
            upd!(t, time) = (interior_view(t.v) .= 0.05 * sin(time); nothing)
            return (_FP_CP.Fringe(fr, tg, upd!, 1.0),)
        end
    end
    # (per_point: the per-point form of #533 step 8, the components' point
    # kernels; integration of steps 2 and 8)
    for which in (:rest, :mean, :fringe), drv in (:euler, :rk3), per_point in (false, true)
        mk() = (s = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0));
                d0 = Domain(s; scheme = drv, validate = false);
                Domain(d0; components = (CentredAdvection(), comps(s, which)...)))
        a = mk(); r = mk()
        a.state.workspaces.fused.per_point = per_point
        _FP_SIM.set_loop_fusion!(a.state, :always); _FP_SIM.set_loop_fusion!(r.state, :never)
        h = _FP_CP.core_hooks(a)
        @test _FP_SIM.hook_terms(h.body_force!, a.state) !== nothing
        fill!(a.state.tw.data, NaN)
        for _ in 1:3
            step!(a, 0.5); step!(r, 0.5)
        end
        ok = _fp_agree(a.state, r.state)
        ok || println("  [#533 step 2] component mismatch: $which $drv per_point=$per_point")
        @test ok
        @test all(isnan, a.state.tw.data) != per_point     # ring: unused; per point: written
        rep = _FP_CP.sweep_report(a)
        @test rep["substages_fused"] == "all" && isempty(rep["not_fusable"])
        # the components acted: without them the state differs
        z = (s = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0));
             Domain(Domain(s; scheme = drv, validate = false); components = (CentredAdvection(),)))
        foreach(_ -> step!(z, 0.5), 1:3)
        @test !agree(interior_view(z.state.u), interior_view(a.state.u))
    end
end

@testset "type stability and allocation" begin
    s = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-3, θ_flux_bottom = 0.1,
                                       sponge_α_max = 0.01, sponge_z_start = 70.0, moisture = true))
    plan = _FP_SIM._sweep_terms(s, nothing, nothing, nothing, Aeolus.Operators.momentum_advect!, s.ν)
    @test isconcretetype(typeof(plan)) && all(t -> t.on, Base.tail(plan))
    @test (@inferred Union{Nothing,typeof(plan)} _FP_SIM._sweep_terms(
               s, nothing, nothing, nothing, Aeolus.Operators.momentum_advect!, s.ν)) isa typeof(plan)
    # a simulation without the terms has the same plan type: one compiled sweep
    s0 = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, moisture = true))
    @test typeof(_FP_SIM._sweep_terms(s0, nothing, nothing, nothing, Aeolus.Operators.momentum_advect!, s0.ν)) ===
          typeof(plan)
    for _ in 1:3
        step_rk3!(s, 0.1)
    end
    n = @allocated step_rk3!(s, 0.1)
    println("  [#533 step 2] step_rk3! with every built-in pointwise term fused allocates $n bytes")
    @test n <= 64                     # the threaded launch's task closure, none per term or point
    # the components' terms: a concrete tuple
    d = Domain(Domain(s; validate = false); components = (CentredAdvection(), _FP_CP.Coriolis(1e-3),
                                                          _FP_CP.SurfaceThetaFlux(0.1, 0.0),
                                                          _FP_CP.Sponge(0.02, 70.0, (u = 0.0, v = 0.0, w = 0.0))))
    ht = @inferred _FP_SIM.hook_terms(_FP_CP.core_hooks(d).body_force!, d.state)
    @test ht isa Tuple && isconcretetype(typeof(ht))
    @test count(t -> t.on, ht) == 3                  # each attached at one stage (an OptionalTerm per stage)
    # a component without a sweep term next to one with: still a concrete tuple
    # (LargeScale attaches at run time; inference once widened this to a Vararg tuple)
    lsf = (u_g = 8.0, v_g = -2.0, z_g = NaN, advective_tendencies = false, subsidence = false, nudging = false,
           sounding_nudging_tau = Inf, nudging_factor = 1.2, nudging_depth = 200.0, divergence = 0.0,
           damping_rate = 0.0, damping_frac = 0.25, damping_target = :initial)
    s2 = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0))
    d0 = Domain(s2; scheme = :rk3, validate = false)
    d2 = Domain(d0; components = (CentredAdvection(),
                                  _FP_CP.build_components((_FP_CP.Coriolis => (; f = 1e-3), _FP_CP.LargeScale => lsf), d0)...))
    # (with #533 step 3 merged, LargeScale has a sweep term too: the old
    # expectation that it blocks was made outdated by the integration of
    # steps 2 and 3; the inferred plan must still be one concrete tuple)
    ht2 = _FP_SIM.hook_terms(_FP_CP.core_hooks(d2).body_force!, s2)
    @test ht2 isa Tuple && isconcretetype(typeof(ht2)) && length(ht2) == 4 && count(t -> t.on, ht2) == 2
    rt = only(Base.return_types(_FP_SIM.hook_terms, typeof.((_FP_CP.core_hooks(d2).body_force!, s2))))
    rt2 = rt isa Union ? (rt.a === Nothing ? rt.b : rt.a) : rt
    @test rt2 === typeof(ht2)                         # Nothing or this one tuple type, never a Vararg
    step!(d2, 1.0)                                   # (the widened plan crashed Julia 1.13's code generation)

end

# --- the review of step 2 -----------------------------------------------------
@testset "review of step 2: target checks, 3-D sponge targets, concrete plans" begin
    # a level-vector target of the wrong length is refused as by the
    # whole-field kernel (the sweep reads it without bounds checks)
    for n in (g.Nz + 4, g.Nz - 3), mode in (:never, :if_possible, :always)
        s = BoussinesqSimulation(g; ν = 1e-2, sponge_α_max = 0.05, sponge_z_start = 60.0,
                                 sponge_target = (u = ones(n), v = 0.0))
        _FP_SIM.set_loop_fusion!(s, mode)
        @test_throws DimensionMismatch step!(s, 0.1)
    end
    @test_throws DimensionMismatch _FP_FV.sponge_term(0.05, 60.0, (u = ones(g.Nz + 1), v = 0.0, w = 0.0), g)
    fr = Aeolus.Transition.FringeRegion(x_start = 100.0, x_end = 200.0, rise = 40.0, fall = 20.0, λ_max = 0.3)
    s = BoussinesqSimulation(g; ν = 1e-2)
    @test_throws DimensionMismatch _FP_FV.sweep_term(_FP_CP.Fringe(fr, (u = ones(3), v = 0.0, w = 0.0), nothing, 1.0),
                                                     (state = s,))

    # the core's sponge toward 3-D targets (a Field, a halo-padded array):
    # the DampingTargets kind 2, fused = one loop per term
    for (gname, mesh, bcs) in grids, drv in (step!, step_rk3!), (bn, b) in _FP_BACKENDS
        with_backend(b) do
            zs = mesh.z0 + 0.5 * mesh.Lz
            mk() = (tu = _fp_rand!(Field(FaceX(), mesh), 0.5, 0.2; seed = 41);
                    tv = _fp_rand!(Field(FaceY(), mesh), -0.3, 0.2; seed = 42);
                    _fp_seed!(BoussinesqSimulation(mesh; ν = 1e-2, θ₀ = 300.0, bcs = bcs, sponge_α_max = 0.05,
                                                   sponge_z_start = zs, sponge_target = (u = tu, v = tv.data, w = 0.0))))
            a = mk(); r = mk(); z = mk()
            _FP_SIM.set_loop_fusion!(a, :always); _FP_SIM.set_loop_fusion!(r, :never)
            z.has_sponge = false
            for _ in 1:3
                drv(a, 0.5); drv(r, 0.5); drv(z, 0.5)
            end
            @test a.workspaces.sponge.targets.kind == (2, 2, 0)
            ok = _fp_agree(a, r)
            ok || println("  [#533 step 2] 3-D sponge target mismatch: $gname $drv $bn")
            @test ok
            @test !agree(interior_view(a.u), interior_view(z.u))      # the sponge acted
            @test _FP_SIM.sweep_report(a)["substages_fused"] == "all"
        end
    end

    # SimulationCouplings: the sweep term's type follows from the component's
    # type (obstacles are its parameter), so the plan stays one concrete tuple
    SD = Aeolus.SimulationDriver
    land = Aeolus.Radiation.SlabLand(; T = 300.0)
    slab = SD.SlabSurface(land, 0.01, 295.0, 1200.0, 0.0, 1.0, 0.0, 0.07, 0.0)
    c = SD.SimulationCouplings(SD.SimCouplings(0.0, slab, nothing, nothing, nothing, nothing, nothing, 0.0))
    s3 = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0))
    d3 = Domain(Domain(s3; validate = false); components = (CentredAdvection(), _FP_CP.Coriolis(1e-3), c,
                                                            _FP_CP.Sponge(0.02, 70.0, (u = 0.0, v = 0.0, w = 0.0))))
    @test Base.return_types(_FP_FV.sweep_term, typeof.((c, d3))) == [_FP_FV.SurfaceFluxTerm{Float64}]
    h3 = _FP_CP.core_hooks(d3).body_force!
    rt = Base.return_types(_FP_SIM.hook_terms, typeof.((h3, s3)))
    @test length(rt) == 1 && isconcretetype(rt[1])
    co = SD.SimulationCouplings(SD.SimCouplings(0.0, slab, nothing, nothing, nothing, nothing, Aeolus.IBM.SolidMask(g), 0.0))
    # (with obstacles, each stage has its own term: #533 steps 2 and 5 integrated)
    @test Base.return_types(_FP_FV.sweep_term, typeof.((co, d3))) == [_FP_FV.SurfaceFluxTerm{Float64}]
    rtb = only(Base.return_types(_FP_CP.stage_sweep_term, typeof.((co, d3, Val(:body_force)))))
    @test isconcretetype(rtb) && rtb <: Tuple{Aeolus.IBM.IBMPenaltyTerm,Bool}
    # no slab: the component does not attach to :tendencies, its term is off
    cn = SD.SimulationCouplings(SD.SimCouplings(0.0, nothing, zeros(g.Nx, g.Ny), nothing, nothing, nothing, nothing, 0.0))
    dn = Domain(Domain(_fp_seed!(BoussinesqSimulation(g; ν = 1e-2)); validate = false); components = (CentredAdvection(), cn))
    @test _FP_CP.core_hooks(dn).body_force! === nothing          # attached to no force stage
    htn = _FP_SIM.hook_terms(_FP_CP._ForceHook(dn), dn.state)
    # (one entry: without obstacles its :body_force stage has no term, the
    # per-stage terms of the integration of #533 steps 2 and 5)
    @test htn isa Tuple && length(htn) == 1 && !any(t -> t.on, htn)

    # a Domain's hooks are specialised on: the plan is built without a
    # dynamic call (an unspecialised Function argument once left it to one,
    # 1-2 kB per substage)
    for comps in ((CentredAdvection(),), (CentredAdvection(), _FP_CP.Coriolis(1e-3)))
        sd = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0))
        dd = Domain(Domain(sd; validate = false); components = comps)
        _FP_SIM.set_loop_fusion!(sd, :always)
        foreach(_ -> step!(dd, 0.5), 1:3)
        n = @allocated step!(dd, 0.5)
        println("  [#533 step 2] step! of a domain with $(length(comps) - 1) pointwise component(s), fused: $n bytes")
        @test n <= 64
    end
    s4 = _fp_seed!(BoussinesqSimulation(g; ν = 1e-2, θ₀ = 300.0, f_coriolis = 1e-3, sponge_α_max = 0.01,
                                        sponge_z_start = 70.0))
    foreach(_ -> step!(s4, 0.5), 1:3)
    @test (@allocated step!(s4, 0.5)) <= 64      # forward Euler too, not only RK3
end

end
