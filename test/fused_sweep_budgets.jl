# The budgets inside the fused sweep (#533 plan step 7):
# src/operators/fused_sweep_budgets.jl (`LevelSums`, `BookedTerm`),
# components/budgets.jl (`SweepBudget`), BudgetSystem.book_context!.
#
#   1. the per-level sums a `BookedTerm` accumulates in the sweep are
#      Σ_ij (air cells only, where masked) of its term's contribution,
#      computed from the whole-field reference of the term, for 1-5 slabs
#      (also run in reverse order), serial and threaded, regular and
#      stretched grids, dry and moist; booking does not change the state;
#   2. a domain with budgets on and in-core components with sweep terms
#      runs fused (`loop_fusion = "always"`), and its budgets agree term by
#      term with those of the same domain on one loop per term (the
#      reference: contexts on the whole tendency fields), for step! and
#      step_rk3!, regular and stretched grids, dry and moist, with a solid
#      mask; the closure stays round-off; the attributed heating is the
#      analytic one; the fused run makes 2 fewer passes per in-core
#      component, quantity and substage; an entry the budgets do not
#      attribute is not booked;
#   3. a retried adaptive step books its in-core terms once (the discard);
#   4. type stability of the term tuple, no allocation in the sweep.
#
# Tolerances. The sweep books Σ_rows (Σ_i t_after − Σ_i t_before) of the
# row buffer, the reference the difference of the two whole-level sums
# of the tendency field: both lose round-off of eps × Σ|t| (the whole
# tendency, base terms included) to the cancellation. TB_RTOL bounds it
# relative to the largest term of the comparison (measured below; the
# largest seen is printed).

using Test
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.Grids: interior, CellCenter
using Aeolus.Fields: Field
using Aeolus.Backends: CPUBackend, with_backend
using Aeolus.Simulations: BoussinesqSimulation
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.Components: AbstractComponent, Stage, Domain, CentredAdvection, Model, with_budgets
using Aeolus.BudgetSystem: stage_budget
using Aeolus.TimeIntegration: CFLConstraints
using Aeolus.MoistThermodynamics: DEFAULT_CONSTANTS, R_d
using Random: Xoshiro, randn, rand

const TBF = Aeolus.Operators.FiniteVolume
const TBS = Aeolus.Simulations
const TBC = Aeolus.Components

"The agreement of booked terms: round-off of the cancellation in both sums (see the header)."
const TB_RTOL = 1e-10

const TB_BACKENDS = (("serial", CPUBackend(serial = true)), ("threshold 0", CPUBackend(threshold = 0)))

function tb_seed!(s; seed = 537)
    rng = Xoshiro(seed)
    for f in (s.u, s.v, s.w)
        interior_view(f) .= 0.3 .* randn(rng, size(interior_view(f)))
    end
    interior_view(s.θ) .= 300 .+ 0.1 .* randn(rng, size(interior_view(s.θ)))
    s.has_moisture && (interior_view(s.qv) .= 0.008 .+ 1e-4 .* randn(rng, size(interior_view(s.qv))))
    return s
end

# ---------------------------------------------------------------------------
# test terms and their whole-field references
# ---------------------------------------------------------------------------

# pointwise, on every booked field and w: −λ(u − U₀), a·w, −λw, a heating
# profile Q(k), and a moisture source s(θ − 300)
struct TBForce{T,V<:AbstractVector{T}} <: TBF.SweepTerm
    λ::T
    U0::T
    a::T
    sq::T
    Q::V            # storage-indexed heating profile (K s⁻¹)
end
TBF.footprint(::TBForce) = (0, 0, 0)
TBF.sweep_inputs(t::TBForce) = (t.Q,)
@inline function tb_force(t::TBForce, i, j, k, u, v, w, θ)
    @inbounds return (-t.λ * (u[i, j, k] - t.U0), t.a * w[i, j, k], -t.λ * w[i, j, k], t.Q[k],
                      t.sq * (θ[i, j, k] - 300))
end
@inline function TBF.add_tendency!(t::TBForce, tb, i, j, k, st)
    cu, cv, cw, cθ, cq = tb_force(t, i, j, k, st.u, st.v, st.w, st.θ)
    tb[i, TBF.SWEEP_U] += cu
    tb[i, TBF.SWEEP_V] += cv
    tb[i, TBF.SWEEP_W] += cw
    tb[i, TBF.SWEEP_Θ] += cθ
    st.qv === nothing || (tb[i, TBF.SWEEP_QV] += cq)
    return nothing
end

# radius 2 in y and z (the 3-row ring and 2 edge planes): a fourth
# difference of u along y and z, and of θ along z
struct TBHyper{T} <: TBF.SweepTerm
    c::T
end
TBF.footprint(::TBHyper) = (0, 2, 2)
@inline tb_d4(u, i, j, k, dj, dk) = @inbounds u[i, j-2dj, k-2dk] - 4u[i, j-dj, k-dk] + 6u[i, j, k] -
                                             4u[i, j+dj, k+dk] + u[i, j+2dj, k+2dk]
@inline tb_hyper(t::TBHyper, i, j, k, u, θ) =
    (-t.c * (tb_d4(u, i, j, k, 1, 0) + tb_d4(u, i, j, k, 0, 1)), -t.c * tb_d4(θ, i, j, k, 0, 1))
@inline function TBF.add_tendency!(t::TBHyper, tb, i, j, k, st)
    cu, cθ = tb_hyper(t, i, j, k, st.u, st.θ)
    tb[i, TBF.SWEEP_U] += cu
    tb[i, TBF.SWEEP_Θ] += cθ
    return nothing
end

# each term's contribution at a point, as (u, v, w, θ, qv)
tb_contrib(t::TBForce, s, i, j, k) =
    tb_force(t, i, j, k, s.u.data, s.v.data, s.w.data, s.θ.data)
tb_contrib(t::TBHyper, s, i, j, k) =
    ((cu, cθ) = tb_hyper(t, i, j, k, s.u.data, s.θ.data); (cu, 0.0, 0.0, cθ, 0.0))

# the whole-field reference of a term: added into the tendency fields
function tb_ref!(s, t)
    ii, jj, kk = interior(s.mesh)
    for k in kk, j in jj, i in ii
        c = tb_contrib(t, s, i, j, k)
        s.tu.data[i, j, k] += c[1]; s.tv.data[i, j, k] += c[2]; s.tw.data[i, j, k] += c[3]
        s.tθ.data[i, j, k] += c[4]
        s.has_moisture && (s.tqv.data[i, j, k] += c[5])
    end
    return s
end

# the reference level sums of a term's contribution to booked field f
# (1: u, 2: v, 3: θ, 4: qv), over the air cells of `fluid` (or all)
function tb_level_sums(s, t, f, fluid)
    ii, jj, kk = interior(s.mesh)
    H = (s.mesh.Hx, s.mesh.Hy, s.mesh.Hz)
    out = zeros(length(kk))
    for k in kk, j in jj, i in ii
        fluid === nothing || fluid[i-H[1], j-H[2], k-H[3]] || continue
        out[k-H[3]] += tb_contrib(t, s, i, j, k)[(1, 2, 4, 5)[f]]
    end
    return out
end

# ---------------------------------------------------------------------------
# components with these terms (their run_stage! is the reference)
# ---------------------------------------------------------------------------

struct TBForceComp{V,M} <: AbstractComponent
    term::TBForce{Float64,V}
    solid::M                     # the budgets' solid mask (halo-padded) or nothing
end
struct TBHyperComp <: AbstractComponent
    term::TBHyper{Float64}
end
# a damping the budgets do not attribute (it books into the dynamics' share)
struct TBQuietComp <: AbstractComponent
    term::TBForce{Float64,Vector{Float64}}
end
for (C, S) in ((TBForceComp, :tendencies), (TBHyperComp, :body_force), (TBQuietComp, :body_force))
    @eval TBC.component_name(::Type{<:$C}) = Symbol(lowercase(string(nameof($C))))
    @eval TBC.section(::Type{<:$C}) = string(nameof($C))
    @eval TBC.stages(::$C) = (Stage($(QuoteNode(S))),)
    @eval TBC.run_stage!(c::$C, d, ::Val{$(QuoteNode(S))}, dt) = (tb_ref!(d.state, c.term); nothing)
    @eval TBF.sweep_term(c::$C, d) = c.term
end
TBC.solid_mask(c::TBForceComp) = c.solid
TBC.attributed(::TBQuietComp, ::Val) = false

# A heater whose sweep term's prepare! throws on its first call (the
# adaptive step retries with dt/2): 1e-3 K s⁻¹ in the lowest level.
mutable struct TBHeat <: TBF.SweepTerm
    thrown::Bool
    k1::Int
end
TBF.footprint(::TBHeat) = (0, 0, 0)
TBF.prepare!(t::TBHeat, sim, Δt) = (t.thrown || (t.thrown = true; error("TBHeat: the first attempt fails")); nothing)
@inline TBF.add_tendency!(t::TBHeat, tb, i, j, k, st) = (k == t.k1 && (tb[i, TBF.SWEEP_Θ] += 1e-3); nothing)
struct TBHeaterComp <: AbstractComponent
    t::TBHeat
end
TBC.component_name(::Type{TBHeaterComp}) = :heater
TBC.section(::Type{TBHeaterComp}) = "heater"
TBC.stages(::TBHeaterComp) = (Stage(:tendencies),)
TBF.sweep_term(c::TBHeaterComp, d) = c.t

# A damping whose sweep term appears only later (`on[]` false: no fused
# form, the substage runs one loop per term)
struct TBLateComp <: AbstractComponent
    term::TBForce{Float64,Vector{Float64}}
    on::Base.RefValue{Bool}
end
TBC.component_name(::Type{TBLateComp}) = :tblatecomp
TBC.section(::Type{TBLateComp}) = "tblatecomp"
TBC.stages(::TBLateComp) = (Stage(:tendencies),)
TBC.run_stage!(c::TBLateComp, d, ::Val{:tendencies}, dt) = (tb_ref!(d.state, c.term); nothing)
TBF.sweep_term(c::TBLateComp, d) = c.on[] ? c.term : nothing

# ---------------------------------------------------------------------------

@testset "budgets inside the fused sweep (#533 plan step 7)" begin

g = RegularGrid(; size = (14, 10, 12), extent = (140.0, 100.0, 120.0), halo = (3, 3, 3))
sg = StretchedGrid(; size = (12, 10, 14), extent = (240.0, 200.0, 600.0), stretching = (:tanh, 1.5), halo = (3, 3, 3))
grids = (("regular", g, FieldBoundaryConditions(:periodic)),
         ("stretched, walls", sg, FieldBoundaryConditions(:periodic, :periodic; bottom = :noslip, top = :wall)))
tb_profile(mesh) = [1e-4 * (1 + 0.5sin(0.7k)) for k in 1:size(Field(CellCenter(), mesh).data, 3)]
function tb_solid(mesh; seed = 1)
    sol = falses(size(Field(CellCenter(), mesh).data))
    rng = Xoshiro(seed)
    for _ in 1:12
        sol[mesh.Hx+rand(rng, 1:mesh.Nx), mesh.Hy+rand(rng, 1:mesh.Ny), mesh.Hz+rand(rng, 1:mesh.Nz)] = true
    end
    return sol
end

worst = Ref(0.0)

@testset "level sums: Σ_ij of each term's contribution, any slab count" begin
    for (gname, mesh, bcs) in grids
        nk = mesh.Nz
        H = (mesh.Hx, mesh.Hy, mesh.Hz)
        sol = tb_solid(mesh)
        fluid = .!sol[H[1]+1:H[1]+mesh.Nx, H[2]+1:H[2]+mesh.Ny, H[3]+1:H[3]+nk]
        for moist in (false, true), (bn, b) in TB_BACKENDS
            with_backend(b) do
                mk() = (s = tb_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs));
                        TBS.fill_state_halos!(s); s)
                t1 = TBForce(0.02, 0.1, 0.3, 1e-5, tb_profile(mesh)); t2 = TBHyper(0.01)
                r = mk()
                # the reference sums, on the stage-start state; θ and qv masked, u and v not
                on = (true, true, true, moist); masked = (false, false, true, true)
                ref = [tb_level_sums(r, t, f, masked[f] ? fluid : nothing) for t in (t1, t2), f in 1:4]
                # the unbooked sweep: the state the booked one must reproduce
                Δt = 0.3
                lag, lz = 2, 2
                function sweep!(a, terms, ns; reversed = false)
                    bufs = TBF.sweep_buffers(nothing, a.u.data, moist ? 5 : 4, ns, nk; lag = lag, lz = lz)
                    p = TBS._Axpy(Δt)
                    s1 = (a.θ, p, nothing); s2 = moist ? (a.qv, p, nothing) : nothing
                    if !reversed
                        TBF.blocked_predictor!(bufs, terms, a.u, a.v, a.w, p, s1, s2, mesh)
                    else               # the slabs one after another, last first (the front edge)
                        ii, jj, kk = interior(mesh)
                        st = (u = a.u.data, v = a.v.data, w = a.w.data, θ = TBF._sweep_field(s1),
                              qv = TBF._sweep_field(s2), m = TBF.grid_metrics(mesh))
                        p1, p2 = TBF._sweep_pred(s1), TBF._sweep_pred(s2)
                        for sl in ns:-1:1
                            TBF._b_sweep_slab!(sl, terms, st, p, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
                        end
                        for n in 1:2lz*ns, j in jj, i in ii
                            TBF._b_sweep_edge!(i, j, n, st, p, p1, p2, bufs.edge, kk, ns, lz)
                        end
                    end
                end
                plain = mk(); sweep!(plain, (TBF.BaseTendencies(plain.ν), t1, t2), 1)
                for ns in 1:min(5, nk ÷ 4), reversed in (false, true)
                    reversed && (ns == 1 || bn != "serial") && continue
                    a = mk()
                    sums = TBF.LevelSums(2, nk, on, masked, fluid, H...)
                    terms = (TBF.BaseTendencies(a.ν), TBF.BookedTerm(t1, sums, 1), TBF.BookedTerm(t2, sums, 2))
                    @test TBF.sweep_lags(terms) == (2, 2)
                    sweep!(a, terms, ns; reversed)
                    # booking adds nothing: the state is the unbooked sweep's
                    @test all(f -> agree(interior_view(f[1]), interior_view(f[2])),
                              ((a.u, plain.u), (a.v, plain.v), (a.w, plain.w), (a.θ, plain.θ), (a.qv, plain.qv)))
                    for (n, t) in enumerate((t1, t2)), f in 1:4
                        got = sums.acc[f, n, :]
                        if !on[f]
                            @test all(iszero, got)
                            continue
                        end
                        want = ref[n, f]
                        scale = maximum(abs, want)
                        scale == 0 && (@test all(iszero, got); continue)
                        err = maximum(abs, got .- want) / scale
                        worst[] = max(worst[], err)
                        err <= TB_RTOL || println("  [#533 s7] level sums: $gname moist=$moist $bn ns=$ns ",
                                                  "rev=$reversed term $n field $f: rel err $err")
                        @test err <= TB_RTOL
                    end
                end
            end
        end
    end
    println("  [#533 s7] level sums: largest relative difference to the reference $(worst[])")
end

ρ0 = DEFAULT_CONSTANTS.p_ref / (R_d(DEFAULT_CONSTANTS) * 300.0)
cp = DEFAULT_CONSTANTS.cp_dry
# a domain with the budgets on: the core needs θ₀ for the budgets' reference
# density; its buoyancy (plan step 2, not fused on this branch) is switched
# off so that every term of a substage has a fused form
function tb_domain(mesh, bcs, moist, scheme, solid; quiet = true, qs = nothing, cfl = nothing, extra = ())
    s = tb_seed!(BoussinesqSimulation(mesh; ν = 1e-2, moisture = moist, bcs = bcs, θ₀ = 300.0))
    s.has_buoyancy = false
    comps = (TBForceComp(TBForce(0.02, 0.1, 0.3, 1e-5, tb_profile(mesh)), solid), TBHyperComp(TBHyper(0.01)))
    quiet && (comps = (comps..., TBQuietComp(TBForce(0.05, 0.0, 0.0, 0.0, zeros(size(s.θ.data, 3))))))
    d0 = Domain(s; scheme = scheme, cfl = cfl, validate = false)
    d = Domain(d0; components = (CentredAdvection(), comps..., extra...))
    q = qs === nothing ? (moist ? (:energy, :water, :momentum_x, :momentum_y) : (:energy, :momentum_x, :momentum_y)) : qs
    return with_budgets(d, q)
end

worst[] = 0.0
@testset "domain: fused budgets = the per-term loops' ($gname, $scheme, moist $moist)" for
        (gname, mesh, bcs) in grids, scheme in (:euler, :rk3), moist in (false, true)
    sol = tb_solid(mesh)
    a = tb_domain(mesh, bcs, moist, scheme, sol); r = tb_domain(mesh, bcs, moist, scheme, sol)
    TBS.set_loop_fusion!(a.state, :always); TBS.set_loop_fusion!(r.state, :never)
    nsteps, dt = 3, 0.4
    n0 = Dict(q => sb.nmeasure for (q, sb) in TBC.budgets(a).measured)
    nr0 = Dict(q => sb.nmeasure for (q, sb) in TBC.budgets(r).measured)
    for _ in 1:nsteps
        step!(Model(a), dt); step!(Model(r), dt)
    end
    @test TBC.sweep_report(a)["substages_fused"] == "all"
    @test TBC.sweep_report(r)["substages_fused"] == "none"
    @test all(f -> agree(interior_view(getfield(a.state, f)), interior_view(getfield(r.state, f))),
              (:u, :v, :w, :θ, :p, :qv))
    nsub = scheme === :rk3 ? 3 : 1
    for (q, sb) in TBC.budgets(a).measured
        A = stage_budget(sb); R = stage_budget(TBC.budgets(r).measured[q])
        @test Set(keys(A.terms)) == Set(keys(R.terms))
        big = maximum(abs, values(R.terms))
        for (key, v) in R.terms
            err = abs(A.terms[key] - v) / big
            worst[] = max(worst[], err)
            err <= TB_RTOL || println("  [#533 s7] $gname $scheme moist=$moist $q $key: fused $(A.terms[key]), ",
                                      "reference $v (rel. to the largest term $err)")
            @test err <= TB_RTOL
        end
        # the in-core components are booked; the quiet one is not
        @test haskey(A.terms, (:tendencies, :tbforcecomp))
        q in (:energy, :momentum_x) && @test haskey(A.terms, (:body_force, :tbhypercomp))
        @test !any(k -> k[2] === :tbquietcomp, keys(A.terms))
        # the closure stays round-off, as on the per-term loops
        @test abs(A.closure) <= 1e-12 * big
        @test abs(R.closure) <= 1e-12 * big
        # the fused run measures no in-core entry: 2 passes fewer per
        # attributed component (2) and substage
        @test (sb.nmeasure - n0[q]) ==
              (TBC.budgets(r).measured[q].nmeasure - nr0[q]) - 2 * 2 * nsub * nsteps
    end
    # the heating profile, analytically: ρ₀ c_p Σ_k Q_k Δz_k (air cells / Nx Ny) t
    ii, jj, kk = interior(mesh)
    Q = tb_profile(mesh)
    Δz = Aeolus.BudgetSystem.level_thickness(mesh)
    nair = [count(!, sol[ii, jj, k]) for k in kk]
    want = ρ0 * cp * sum(Q[k] * Δz[k-mesh.Hz] * nair[k-mesh.Hz] / (mesh.Nx * mesh.Ny) for k in kk) * a.state.t
    @test stage_budget(TBC.budgets(a).measured[:energy]).terms[(:tendencies, :tbforcecomp)] ≈ want rtol = 1e-12
end
println("  [#533 s7] domain budgets: largest difference to the reference, relative to the largest term $(worst[])")

@testset "a retried adaptive step books its fused in-core terms once" begin
    mesh, bcs = grids[1][2], grids[1][3]
    d = tb_domain(mesh, bcs, false, :rk3, nothing; quiet = false, qs = (:energy,),
                  cfl = CFLConstraints(; dt_max = 4.0), extra = (TBHeaterComp(TBHeat(false, mesh.Hz + 1)),))
    TBS.set_loop_fusion!(d.state, :always)
    # a divergence-free start (the adaptive step checks the divergence)
    set!(d.state.u, (x, y, z) -> 0.3 * sin(2π * y / 100)); set!(d.state.v, (x, y, z) -> 0.2 * sin(2π * x / 140))
    set!(d.state.w, (x, y, z) -> 0.0)
    res = Aeolus.TimeIntegration.step_adaptive!(Model(d))
    @test res[3] == 1                                   # one retry
    Aeolus.TimeIntegration.step_adaptive!(Model(d))
    @test TBC.sweep_report(d)["substages_fused"] == "all"
    r = stage_budget(TBC.budgets(d).measured[:energy])
    want = ρ0 * cp * 1e-3 * mesh.dz * d.state.t
    @test r.terms[(:tendencies, :heater)] ≈ want rtol = 1e-12
    @test abs(r.closure) <= 1e-12 * maximum(abs, values(r.terms))
end

@testset "a sweep term that appears after the budget term was built is booked" begin
    # review finding: the budget term is built (and cached) at the first
    # substage that asks for the sweep's terms, also when that substage
    # runs unfused because a component has no sweep term yet; the
    # component's term, once it appears, must still be booked
    mesh, bcs = grids[1][2], grids[1][3]
    mk(on) = tb_domain(mesh, bcs, false, :rk3, nothing; quiet = false, qs = (:energy, :momentum_x),
                       extra = (TBLateComp(TBForce(0.03, 0.2, 0.0, 0.0, tb_profile(mesh)), Ref(on)),))
    a = mk(false); r = mk(false)
    TBS.set_loop_fusion!(a.state, :if_possible); TBS.set_loop_fusion!(r.state, :never)
    late(d) = d.components[findfirst(c -> c isa TBLateComp, d.components)]
    step!(Model(a), 0.4); step!(Model(r), 0.4)
    @test TBC.sweep_report(a)["substages_fused"] == "none"
    @test a.budgets.sweep isa TBC.SweepBudget        # built while unfused
    late(a).on[] = true; late(r).on[] = true
    for _ in 1:2
        step!(Model(a), 0.4); step!(Model(r), 0.4)
    end
    @test TBC.sweep_report(a)["substages_fused"] == "some"   # the first step's 3 unfused, the rest fused
    @test a.state.workspaces.fused.fused == 6
    for q in (:energy, :momentum_x)
        A = stage_budget(TBC.budgets(a).measured[q]); R = stage_budget(TBC.budgets(r).measured[q])
        big = maximum(abs, values(R.terms))
        @test R.terms[(:tendencies, :tblatecomp)] != 0     # it books something
        @test Set(keys(A.terms)) == Set(keys(R.terms))
        for (key, v) in R.terms
            @test abs(A.terms[key] - v) <= TB_RTOL * big
        end
        @test abs(A.closure) <= 1e-12 * big
    end
end

@testset "type stability and allocation" begin
    mesh, bcs = grids[1][2], grids[1][3]
    d = tb_domain(mesh, bcs, true, :euler, tb_solid(mesh))
    h = TBC.core_hooks(d)
    ts = TBS.hook_terms(h.body_force!, d.state)
    @test (@inferred TBS.hook_terms(h.body_force!, d.state)) === ts       # cached: the same objects
    @test ts[1] isa TBC.SweepBudget
    @test ts[2] isa TBF.BookedTerm && ts[3] isa TBF.BookedTerm && ts[4] isa TBForce   # the quiet one unbooked
    @test (@allocated TBS.hook_terms(h.body_force!, d.state)) == 0
    step!(Model(d), 0.2)
    s = d.state
    TBS.fill_state_halos!(s)
    terms = (TBF.BaseTendencies(s.ν), ts...)
    lag, lz = TBF.sweep_lags(terms)
    bufs = TBF.sweep_buffers(nothing, s.u.data, 5, 2, mesh.Nz; lag = lag, lz = lz)
    p = TBS._Axpy(1e-3)
    sweep() = TBF.blocked_predictor!(bufs, terms, s.u, s.v, s.w, p, (s.θ, p, nothing), (s.qv, p, nothing), mesh)
    sweep()
    n = @allocated sweep()
    println("  [#533 s7] a blocked_predictor! with booked terms allocates $n bytes")
    @test n <= 64          # the launch's task closure on the threaded backend, none per point
end

end
