# The fused sweep near walls, at open faces and in the urban site (#533
# plan step 5): the terms that add to the tendencies next to a boundary
# run inside the cache-blocked pass and agree with their whole-field
# references (their `run_stage!` / operator, which stays).
#
#   1. open lateral faces: the boundary update (the data, the radiating
#      outflow from the stage-start layers, the mass correction) runs after
#      the sweep and before the projection, as after the separate predictor
#      passes, so an open box takes the fused step; step! and step_rk3!
#      agree with one loop per term, Dirichlet, Orlanski and convective
#      outflow, regular and stretched grids, dry and moist, serial and
#      threaded;
#   2. the terms of SurfaceLayer's heat flux (Components.SurfaceHeatFluxTerm),
#      of the removal of the stencil's flux through immersed walls
#      (WallModels.StencilWallFluxTerm, the facade wall model's :body_force)
#      and of the IBM penalty (IBM.IBMPenaltyTerm, the obstacles of
#      SimulationCouplings): FiniteVolume.blocked_predictor! over them = the
#      base tendencies, each term's whole-field pass, then the predictor,
#      for 1-5 slabs (also run in reverse slab order, the front edge),
#      serial and threaded, dry and moist, with and without the RK3 stage
#      combination;
#   3. step! and step_rk3! of a domain with those components (their
#      tendency parts) fused = one loop per term (loop_fusion "always" /
#      "never");
#   4. a two-way nest: the flux-register hooks (Nesting.RegisterBookingTerm)
#      book in prepare!, the fused nest = the unfused one, and the
#      composite ∫θ is conserved to round-off;
#   5. the urban facade wall model's adapter on the synthetic city, and
#      the type stability and allocation of the terms.
#
# Tolerances: every term evaluates the floating-point expressions of its
# whole-field reference, in the same order, so the agreement is round-off
# (`agree`, ROUNDOFF_RTOL = 1e-10 of the field's scale).

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Grids: interior
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation, step_rk3!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Components: AbstractComponent, Stage, Domain, CentredAdvection
using Aeolus.Config: load_run_config
using Random: Xoshiro, randn

const NWs = Aeolus.Simulations
const NWFV = Aeolus.Operators.FiniteVolume
const NWC = Aeolus.Components
const NWW = Aeolus.WallModels
const NWI = Aeolus.IBM
const NWSD = Aeolus.SimulationDriver

const _NW_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)))

_nw_state(s) = (s.u, s.v, s.w, s.p, s.θ, s.qv)
_nw_agree(a, b) = all(i -> agree(interior_view(_nw_state(a)[i]), interior_view(_nw_state(b)[i])),
                      eachindex(_nw_state(a))) && a.t ≈ b.t


function _nw_seed!(s; U = 0.3, seed = 533)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= U .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    return s
end

# the whole-field reference of one substage: the SC-d base tendencies, the
# terms' passes (`extra!(sim)`), then the predictors (the scalars with the
# RK3 combination 3/4, 1/4 and the step-start `snap`)
function _nw_reference!(r, Δt, extra!, rk3, snap)
    moist = r.has_moisture
    NWFV._fused_tendencies!(r.tu, r.tv, r.tw, r.u, r.v, r.w, r.ν, (r.tθ, r.θ),
                            moist ? (r.tqv, r.qv) : nothing, r.mesh)
    extra!(r)
    for (f, t) in ((r.u, r.tu), (r.v, r.tv), (r.w, r.tw))
        NWs.axpy!(f, Δt, t)
    end
    c = NWs._Lin2r(0.75, 0.25)
    for (f, t, s0) in (moist ? ((r.θ, r.tθ, snap[1]), (r.qv, r.tqv, snap[2])) : ((r.θ, r.tθ, snap[1]),))
        rk3 ? NWs._advance_scalar!(f, Δt, t, (c = c, ws = nothing), s0) : NWs.axpy!(f, Δt, t)
    end
    return r
end

# the fused sweep over `terms` with `ns` slabs; `reversed`: the slabs one
# after another from the last to the first, then the edge planes (the
# front edges read by a neighbour that has finished, test/fused_sweep.jl)
function _nw_sweep!(a, Δt, terms, rk3, snap, ns; reversed = false)
    moist = a.has_moisture
    ii, jj, kk = interior(a.mesh)
    lag, lz = NWFV.sweep_lags(terms)
    bufs = NWFV.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, length(kk); lag = lag, lz = lz)
    foreach(t -> NWFV.prepare!(t, a, Δt), terms)
    c = NWs._Lin2r(0.75, 0.25)
    ps = rk3 ? NWs._AxpyStage(NWs._Axpy(Δt), c) : NWs._Axpy(Δt)
    s1 = (a.θ, ps, rk3 ? snap[1] : nothing)
    s2 = moist ? (a.qv, ps, rk3 ? snap[2] : nothing) : nothing
    pu = NWs._Axpy(Δt)
    if !reversed
        NWFV.blocked_predictor!(bufs, terms, a.u, a.v, a.w, pu, s1, s2, a.mesh)
        return bufs
    end
    st = (u = a.u.data, v = a.v.data, w = a.w.data, θ = NWFV._sweep_field(s1), qv = NWFV._sweep_field(s2),
          m = NWFV.grid_metrics(a.mesh))
    p1, p2 = NWFV._sweep_pred(s1), NWFV._sweep_pred(s2)
    for s in ns:-1:1
        NWFV._b_sweep_slab!(s, terms, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
    end
    for n in 1:2lz*ns, j in jj, i in ii
        NWFV._b_sweep_edge!(i, j, n, st, pu, p1, p2, bufs.edge, kk, ns, lz)
    end
    return bufs
end

# a component that runs only the tendency stages of the component it wraps
# (its post-predictor part is plan step 6): the stages before the core and
# the in-core ones up to the predictor, and its sweep term
struct NWTendencyPart{C} <: AbstractComponent
    c::C
    stages::Tuple
end
NWC.component_name(::Type{<:NWTendencyPart}) = :nw_tendency_part
NWC.section(::Type{<:NWTendencyPart}) = "nw_tendency_part"
NWC.stages(w::NWTendencyPart) = w.stages
NWC.run_stage!(w::NWTendencyPart, d, v::Val, dt) = NWC.run_stage!(w.c, d, v, dt)
NWFV.sweep_term(w::NWTendencyPart, d) = NWFV.sweep_term(w.c, d)

# the removal of the stencil's flux through immersed walls as a :body_force
# component (what the urban facade wall model's stage does, with `sgs`)
struct NWWallFlux{W} <: AbstractComponent
    iw::W
    rows::Any
end
NWWallFlux(iw) = NWWallFlux(iw, NWW.wall_rows(iw))
NWC.component_name(::Type{<:NWWallFlux}) = :nw_wall_flux
NWC.section(::Type{<:NWWallFlux}) = "nw_wall_flux"
NWC.stages(::NWWallFlux) = (Stage(:body_force),)
NWC.run_stage!(c::NWWallFlux, d, ::Val{:body_force}, dt) = (NWW.remove_stencil_wall_flux!(d.state, c.iw); nothing)
NWFV.sweep_term(c::NWWallFlux, d) = NWW.stencil_wall_flux_term(d.state, c.iw; rows = c.rows)

# two blocks of solid cells: one on the floor, one touching the y-boundary
# (periodic), so walls cross the slab edges and the halo
function _nw_solid(mesh)
    sm = NWI.SolidMask(mesh)
    NWI.mark_solid!(sm, (x, y, z) -> (40 < x < 90 && 30 < y < 70 && z < 60) || (120 < x < 170 && y > 95 && z < 110))
    return sm
end

_nw_sl_opts() = (model = :most, z0m = 0.1, z0h = nothing, rho_cp = 1206.0,
                 temperature = Dict{String,Any}("kind" => "constant", "value" => 299.0),
                 stability = Dict{String,Any}("family" => "businger_dyer_log_linear", "beta_m" => 4.8, "beta_h" => 7.8))

@testset "fused sweep near walls and open faces (#533 step 5)" begin

@testset "open lateral faces: the fused step = one loop per term" begin
    Lx, Ly, Lz = 800.0, 600.0, 400.0
    # time-dependent data with a net imbalance (the mass correction acts)
    # and a θ, qv that differ from the interior (the radiation acts)
    data(moist) = FunctionBoundaryData(; u = (x, y, z, t) -> 2.0 + 0.3 * x / Lx + 0.5 * sin(2π * t / 60),
                                       v = (x, y, z, t) -> 0.5 + 0.2 * cos(2π * t / 45), w = 0.0,
                                       θ = (x, y, z, t) -> 300.0 + 0.01 * z + 0.2 * sin(2π * t / 30),
                                       (moist ? (qv = (x, y, z, t) -> 0.008 + 1e-3 * sin(2π * t / 40),) : ())...)
    reg = RegularGrid(size = (16, 12, 8), extent = (Lx, Ly, Lz))
    str = StretchedGrid(; size = (16, 12, 10), extent = (Lx, Ly, Lz), stretching = (:tanh, 1.5))
    cases = (("regular, Dirichlet", reg, :dirichlet, :open, false),
             ("regular, Orlanski, moist", reg, :orlanski, :open, true),
             ("regular, convective", reg, :convective, :open, false),
             # (a radiating top is refused on a z-stretched mesh)
             ("stretched, Orlanski sides, wall top", str, :orlanski, :wall, true),
             ("stretched, Dirichlet", str, :dirichlet, :open, false))
    for (name, m, outflow, top, moist) in cases, (bn, b) in _NW_BACKENDS, drv in (step!, step_rk3!)
        with_backend(b) do
            fields = moist ? (:u, :v, :w, :θ, :qv) : (:u, :v, :w, :θ)
            mk() = (s = BoussinesqSimulation(m; ν = 1.0, moisture = moist,
                                             bcs = open_boundary_conditions(m, data(moist); outflow = outflow,
                                                                            top = top, fields = fields));
                    set!(s.u, 2.0); set!(s.v, 0.5); set!(s.θ, 300.0);
                    rng = Xoshiro(533);
                    interior_view(s.θ) .+= 0.1 .* randn(rng, size(interior_view(s.θ)));
                    interior_view(s.u) .+= 0.1 .* randn(rng, size(interior_view(s.u)));
                    moist && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))));
                    s)
            a = NWs.set_loop_fusion!(mk(), :always)     # errors if any term has no fused form
            r = NWs.set_loop_fusion!(mk(), :never)
            fill!(a.tu.data, NaN)
            for _ in 1:4
                drv(a, 0.5); drv(r, 0.5)
            end
            ok = _nw_agree(a, r)
            ok || println("  [#533 s5] open faces mismatch: $name $bn $drv")
            @test ok
            @test all(isnan, a.tu.data)                 # the tendencies stayed in the sweep
            @test NWs.sweep_report(a)["substages_fused"] == "all"
        end
    end
end


g = RegularGrid(; size = (20, 12, 15), extent = (200.0, 120.0, 150.0), halo = (2, 2, 2))
wall_bcs = FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)

@testset "near-wall terms: the sweep = the terms' whole-field passes + predictor" begin
    nk = length(interior(g)[3])
    pt = 3.0                                     # the obstacles' penalty_time
    for moist in (false, true), rk3 in (false, true), (bn, b) in _NW_BACKENDS
        with_backend(b) do
            Δt = 0.3
            mk() = (s = _nw_seed!(BoussinesqSimulation(g; ν = 1e-2, moisture = moist, bcs = wall_bcs));
                    NWs.fill_state_halos!(s); s)
            proto = mk()
            # the surface layer: a per-column heat flux of a step's solve
            d0 = Domain(proto; validate = false)
            sl = only(NWC.build_components((NWC.SurfaceLayer => _nw_sl_opts(),), d0))
            sl.wθ .= 0.05 .* randn(Xoshiro(7), size(sl.wθ))
            sm = _nw_solid(g)
            iw = NWW.immersed_walls(proto, sm, NWW.RoughLogLaw(0.1))
            @test sum(NWW.immersed_wall_count(iw)) > 0
            # the step's eddy viscosity (random, positive): the removal with sgs
            νt = copy(proto.νt.data); νt .= 0.5 .+ rand(Xoshiro(9), size(νt))
            # each term alone, then all of them composed (in pipeline order)
            sets = (("surface heat flux", (s, d) -> (NWFV.sweep_term(sl, d),),
                     (s, d) -> NWC.run_stage!(sl, d, Val(:tendencies), Δt)),
                    ("stencil wall flux, sgs", (s, d) -> (NWW.stencil_wall_flux_term(s, iw; νt = νt, sgs = true),),
                     (s, d) -> NWW.remove_stencil_wall_flux!(s, iw; νt = νt, sgs = true)),
                    ("IBM penalty", (s, d) -> (NWI.ibm_penalty_term(sm, pt),),
                     (s, d) -> NWI.add_ibm_penalty!(s, sm; τ = pt * Δt)),
                    ("all three", (s, d) -> (NWFV.sweep_term(sl, d), NWW.stencil_wall_flux_term(s, iw),
                                             NWI.ibm_penalty_term(sm, pt)),
                     (s, d) -> (NWC.run_stage!(sl, d, Val(:tendencies), Δt); NWW.remove_stencil_wall_flux!(s, iw);
                                NWI.add_ibm_penalty!(s, sm; τ = pt * Δt))))
            for (name, terms, ref!) in sets
                r = mk(); dr = Domain(r; validate = false)
                snap = (copy(r.θ.data), copy(r.qv.data))
                _nw_reference!(r, Δt, s -> ref!(s, dr), rk3, snap)
                F_ref = iw.stats[:F_stencil]
                # the term changes the result (the comparison is not of the base alone)
                r0 = _nw_reference!(mk(), Δt, s -> nothing, rk3, snap)
                @test !(agree(interior_view(r.u), interior_view(r0.u)) && agree(interior_view(r.θ), interior_view(r0.θ)))
                for ns in 1:min(5, nk ÷ 2), reversed in (false, true)
                    reversed && (ns == 1 || bn != "serial") && continue
                    a = mk(); da = Domain(a; validate = false)
                    foreach(t -> fill!(t.data, NaN), (a.tu, a.tv, a.tw, a.tθ, a.tqv))   # neither read nor written
                    _nw_sweep!(a, Δt, (NWFV.BaseTendencies(a.ν), terms(a, da)...), rk3, snap, ns; reversed)
                    ok = all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                             ((a.u, r.u), (a.v, r.v), (a.w, r.w), (a.θ, r.θ), (a.qv, r.qv)))
                    ok || println("  [#533 s5] $name mismatch: moist=$moist rk3=$rk3 $bn ns=$ns reversed=$reversed")
                    @test ok
                    @test all(t -> all(isnan, t.data), (a.tu, a.tv, a.tw, a.tθ, a.tqv))
                    # the wall force of the diagnostics (folded in prepare!) is the reference's
                    occursin("wall", name) && @test agree(collect(iw.stats[:F_stencil]), collect(F_ref))
                end
            end
        end
    end
end

@testset "step! and step_rk3! with the near-wall components: fused = one loop per term" begin
    sm = _nw_solid(g)
    for scheme in (:euler, :rk3), (bn, b) in _NW_BACKENDS
        with_backend(b) do
            function mk(fusion)
                s = _nw_seed!(BoussinesqSimulation(g; ν = 1e-2, bcs = wall_bcs))
                NWs.set_loop_fusion!(s, fusion)
                d0 = Domain(s; scheme = scheme, validate = false)
                sl = only(NWC.build_components((NWC.SurfaceLayer => _nw_sl_opts(),), d0))
                iw = NWW.immersed_walls(s, sm, NWW.RoughLogLaw(0.1))
                cpl = NWSD.SimulationCouplings(NWSD.SimCouplings(0.0, nothing, nothing, nothing, nothing, nothing, sm, 3.0))
                comps = (CentredAdvection(), NWTendencyPart(sl, (Stage(:surface_flux), Stage(:tendencies))),
                         NWWallFlux(iw), cpl)
                return Domain(d0; components = comps)
            end
            a = mk(:always); r = mk(:never)
            fill!(a.state.tu.data, NaN)
            for _ in 1:3
                step!(a, 0.4); step!(r, 0.4)
            end
            ok = _nw_agree(a.state, r.state)
            ok || println("  [#533 s5] near-wall domain mismatch: $scheme $bn")
            @test ok
            @test all(isnan, a.state.tu.data)
            ra = NWC.sweep_report(a)
            @test ra["substages_fused"] == "all" && isempty(ra["not_fusable"])
            # the terms acted: the same domain without them differs
            s0 = _nw_seed!(BoussinesqSimulation(g; ν = 1e-2, bcs = wall_bcs))
            z = Domain(Domain(s0; scheme = scheme, validate = false); components = (CentredAdvection(),))
            foreach(_ -> step!(z, 0.4), 1:3)
            @test !agree(interior_view(z.state.u), interior_view(a.state.u))
            @test !agree(interior_view(z.state.θ), interior_view(a.state.θ))
        end
    end
    # a slab surface with the obstacles: the floor flux has no sweep term yet,
    # so the couplings keep one loop per term, with the documented reason
    cpl = NWSD.SimulationCouplings(NWSD.SimCouplings(0.0, nothing, nothing, nothing, nothing, nothing, sm, 3.0))
    @test NWFV.sweep_term(cpl, nothing) isa NWI.IBMPenaltyTerm
    @test !occursin("no documented reason", NWs.sweep_reason(NWSD.SimulationCouplings))
end

@testset "two-way nest: the flux registers book in prepare!, ∫θ conserved" begin
    L = 1600.0
    function nest_run(fusion)
        pm = RegularGrid(Float64; size = (16, 16, 16), extent = (L, L, L), halo = (3, 3, 3))
        p = BoussinesqSimulation(pm; ν = 2.0, bcs = FieldBoundaryConditions(:periodic))
        nest = Aeolus.Nesting.nested_grid(pm, ((400.0, 1200.0), (400.0, 1200.0), (400.0, 1200.0)), 2)
        cm = Aeolus.Nesting.nest_child_mesh(nest)
        c = BoussinesqSimulation(cm; ν = 2.0, bcs = Aeolus.Nesting.nest_bcs(cm, :nest, :nest, :nest))
        rng = Xoshiro(3)
        for k in axes(p.θ.data, 3), j in axes(p.θ.data, 2), i in axes(p.θ.data, 1)
            p.θ.data[i, j, k] = 1.0 + exp(-((i - 8)^2 + (j - 8)^2 + (k - 8)^2) / 6)
        end
        fill!(p.u.data, 3.0); fill!(p.v.data, 1.0)
        interior_view(p.u) .+= 0.5 .* randn(rng, size(interior_view(p.u)))
        interior_view(p.v) .+= 0.5 .* randn(rng, size(interior_view(p.v)))
        NWs._project_velocity!(p)
        for (cf, pf) in ((c.θ, p.θ), (c.u, p.u), (c.v, p.v), (c.w, p.w))
            Aeolus.Nesting.interpolate_parent_to_child!(cf, pf, nest)
        end
        NWs.set_loop_fusion!(p, fusion); NWs.set_loop_fusion!(c, fusion)
        run = Aeolus.Nesting.NestedRun(p, c, nest; feedback = Aeolus.Nesting.TwoWayFeedback(),
                                       cfl_parent = nothing, cfl_child = nothing)
        I0 = Aeolus.Nesting.composite_integral(p.θ, c.θ, nest)
        for _ in 1:4
            Aeolus.Nesting.step_nested!(run; dt = 4.0)
        end
        return p, c, abs(Aeolus.Nesting.composite_integral(p.θ, c.θ, nest) - I0) / I0
    end
    pa, ca, da = nest_run(:always)            # errors if a hook has no fused form
    pr, cr, dr = nest_run(:never)
    @test NWs.sweep_report(pa)["substages_fused"] == "all" && NWs.sweep_report(ca)["substages_fused"] == "all"
    @test _nw_agree(pa, pr) && _nw_agree(ca, cr)
    # refluxing closes the composite budget in the fused nest as in the
    # unfused one (round-off of 4 steps of 16³ + 16³ sums: 1e-12)
    println("  [#533 s5] two-way nest ∫θ drift: fused $da, one loop per term $dr")
    @test da < 1e-12 && dr < 1e-12
end

@testset "the urban facade wall model's sweep term (synthetic city)" begin
    Aeolus.Operators.set_fftw_policy!(; planner = :estimate)
    city = joinpath(@__DIR__, "data", "urban", "small_city.toml")
    case = Aeolus.Urban.build_case(Aeolus.Urban.site_options(load_run_config(city; overrides = ["fftw.planner=\"estimate\""])))
    fac = Aeolus.Urban.find_coupling(case, :wall_model_facades)
    sim = case.sim
    _nw_seed!(sim; U = 1.0)
    NWs.fill_state_halos!(sim)
    t = NWFV.sweep_term(fac, case.domain)
    @test t isa NWW.StencilWallFluxTerm && t.sgs == Aeolus.Urban._facade_sgs(case)
    # the reference: the stage on zeroed tendencies
    foreach(f -> fill!(f.data, 0.0), (sim.tu, sim.tv, sim.tw))
    Aeolus.Urban.run_stage!(fac, case, Val(:body_force), 1.0)
    ref = (copy(sim.tu.data), copy(sim.tv.data), copy(sim.tw.data))
    # the term's rows on a zeroed buffer, row by row
    NWFV.prepare!(t, sim, 1.0)
    ii, jj, kk = interior(sim.mesh)
    buf = zeros(size(sim.u.data, 1), 1, 1, 3, 1)
    got = (zero(ref[1]), zero(ref[2]), zero(ref[3]))
    for k in kk, j in jj
        fill!(buf, 0.0)
        NWFV.add_row!(t, buf, 1, 1, 1, j, k, ii, nothing)
        for c in 1:3, i in ii
            got[c][i, j, k] = buf[i, 1, 1, c, 1]
        end
    end
    @test count(!iszero, got[1]) + count(!iszero, got[2]) > 0
    @test all(c -> agree(got[c][ii, jj, kk], ref[c][ii, jj, kk]), 1:3)
end

@testset "type stability and allocation of the near-wall terms" begin
    s = _nw_seed!(BoussinesqSimulation(g; ν = 1e-2, bcs = wall_bcs))
    NWs.fill_state_halos!(s)
    d0 = Domain(s; validate = false)
    sl = only(NWC.build_components((NWC.SurfaceLayer => _nw_sl_opts(),), d0))
    sm = _nw_solid(g)
    iw = NWW.immersed_walls(s, sm, NWW.RoughLogLaw(0.1))
    rows = NWW.wall_rows(iw)
    slot = Ref(NaN)
    @inferred NWFV.sweep_term(sl, d0)
    @inferred NWW.stencil_wall_flux_term(s, iw; rows = rows)
    @inferred NWI.ibm_penalty_term(sm, 3.0, slot)
    terms = (NWFV.BaseTendencies(s.ν), NWFV.sweep_term(sl, d0), NWW.stencil_wall_flux_term(s, iw; rows = rows),
             NWI.ibm_penalty_term(sm, 3.0, slot))
    nk = length(interior(g)[3])
    b2 = NWFV.sweep_buffers(nothing, s.u.data, 4, 2, nk)
    s1 = (s.θ, NWs._Axpy(1e-3), nothing)
    sweep() = (foreach(t -> NWFV.prepare!(t, s, 1e-3), terms);
               NWFV.blocked_predictor!(b2, terms, s.u, s.v, s.w, NWs._Axpy(1e-3), s1, nothing, g))
    sweep()
    n = @allocated sweep()
    println("  [#533 s5] a near-wall blocked_predictor! (with prepare!) allocates $n bytes")
    @test n <= 1024        # the launches' task closures on the threaded backend; none per point or per wall
    # building the terms each substage allocates nothing per wall
    build() = (NWFV.sweep_term(sl, d0), NWW.stencil_wall_flux_term(s, iw; rows = rows), NWI.ibm_penalty_term(sm, 3.0, slot))
    build()
    @test (@allocated build()) <= 256
end

end
