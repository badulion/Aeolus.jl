# Nesting: the time coupling of a parent and a child (N-nest-c, #257).
#
#   * DomainAdvance — a per-domain `advance!(sim, dt)` closure carrying the
#     domain's hooks and keywords, calling any dynamical core by dispatch;
#   * NestedRun     — parent + child + NestedGrid + boundary buffers +
#     schedule (subcycling under adaptive steppers, or a shared minimum
#     dt) + feedback (one-way / two-way) + statistics;
#   * step_nested!  — one parent step of a NestedRun; the positional
#     method is the historical fixed-dt two-way entry point, now a thin
#     wrapper over a NestedRun.

# =========================================================================
# Per-domain advance closures
# =========================================================================

"""
    DomainAdvance

A per-domain `advance!(sim, dt)` closure built by [`domain_advance`](@ref):
a dynamical core plus the hooks and keywords that domain's steps must
carry (`body_force!`, `post_predictor!`, `momentum_advection`, `sgs`,
`sgs_stress`, …). Calling it, `adv(sim, dt)`, runs
`core(sim, dt, args...; kwargs...)`; a [`NestedRun`](@ref) additionally
passes its own boundary-data hook, which is chained *after* the domain's
own `post_predictor!`.
"""
struct DomainAdvance{F, A<:Tuple, K<:NamedTuple}
    core::F
    args::A
    kwargs::K
end

"""
    domain_advance(core = step!, args...; kwargs...) -> DomainAdvance

Bundle a dynamical core with the hooks and keywords of one domain. The
core is called through dispatch, so no step logic is copied:

| `core`, `args`                  | step taken                                             |
| :------------------------------ | :----------------------------------------------------- |
| `step!` (default)               | Boussinesq projection step, or the anelastic / pseudo-incompressible step of a `MassWeightedSimulation` |
| `step_rk3!`                     | SSP-RK3 Boussinesq step                                |
| `step_imex!, ARK2()`            | IMEX ARS(2,2,2) step (or `Rosenbrock2()`)              |
| any `f(sim, dt, args...; kw...)`| a user core; it must accept `post_predictor!` if it is the child of a run |

```julia
child_adv = domain_advance(step_rk3!; body_force! = ibm_force!,
                           post_predictor! = wall_stress!, sgs = AMDClosure())
```

The keywords are forwarded verbatim. `sgs` (or `eddy_viscosity!`) and
`sgs_stress` are also what the run's [`adaptive_dt`](@ref) pick for this
domain uses, and a prognostic SGS closure's memory is rolled back with
the state when a step is rejected.
"""
domain_advance(core = step!, args...; kwargs...) =
    DomainAdvance(core, args, NamedTuple(kwargs))

function (a::DomainAdvance)(sim, dt; post_predictor! = nothing)
    if post_predictor! === nothing
        a.core(sim, dt, a.args...; a.kwargs...)
    else
        user = get(a.kwargs, :post_predictor!, nothing)
        pp = user === nothing ? post_predictor! :
             (s, h) -> (user(s, h); post_predictor!(s, h))
        a.core(sim, dt, a.args...;
               merge(a.kwargs, NamedTuple{(:post_predictor!,)}((pp,)))...)
    end
    return sim
end

_imex_scheme(a::DomainAdvance{typeof(step_imex!)}) = a.args[1]
_imex_scheme(::DomainAdvance) = nothing

_domain_sgs(a::DomainAdvance) = get(a.kwargs, :sgs, get(a.kwargs, :eddy_viscosity!, nothing))

"""
    stage_times(adv::DomainAdvance) -> NTuple

The times, as fractions `cᵢ` of the step `Δt`, at which `adv`'s core
evaluates its tendencies, i.e. the times at which it reads the boundary
data in the halos: `(0,)` for the projection step (`step!`, Boussinesq or
sound-proof) and for a user core, `(0, 1, 1/2)` for SSP-RK3
(Shu & Osher 1988), `(0, γ)` for ARK2 and `(0, a₂₁)` for Rosenbrock2.
"""
stage_times(a::DomainAdvance) = _stage_times(a.core, a.args)
_stage_times(core, args) = (0.0,)
_stage_times(::typeof(step_rk3!), args) = (0.0, 1.0, 0.5)
_stage_times(::typeof(step_imex!), args) = _imex_stage_times(args[1])
_imex_stage_times(s::ARK2) = (0.0, Float64(s.γ))
_imex_stage_times(s::Rosenbrock2) = (0.0, Float64(s.a21))

# The time whose boundary data the k-th `post_predictor!` call of one step
# writes into the halos (`t0` = step start, `dt` = step, `h` = the stage
# increment the core passes to the hook). The halos must hold B(t0 + cᵢ dt)
# when stage i computes its tendencies.
#
# Projection step, IMEX stages, sound-proof step: the halo is not combined
# by the core, and the stage being formed ends at t0 + h → write B(t0 + h).
#
# SSP-RK3: the core combines the *whole* arrays, halos included:
# u⁽²⁾ = ¾uⁿ + ¼ũ, uⁿ⁺¹ = ⅓uⁿ + ⅔ũ. Writing B(t0+dt), B(t0+2dt),
# B(t0+3dt/2) after the three predictors makes the halos
# B(t0+dt), ¾B(t0)+¼B(t0+2dt) = B(t0+dt/2), ⅓B(t0)+⅔B(t0+3dt/2) = B(t0+dt):
# exactly the stage times (0, 1, ½) and the step end, because B is linear
# in time inside a parent step (the temporal interpolation).
_halo_write_time(a::DomainAdvance, k::Int, t0, dt, h) = t0 + h
_halo_write_time(::DomainAdvance{typeof(step_rk3!)}, k::Int, t0, dt, h) =
    t0 + (k == 1 ? dt : k == 2 ? 2dt : 1.5dt)

# =========================================================================
# NestedRun
# =========================================================================

"""
    NestRunStats

Bookkeeping of a [`NestedRun`](@ref) (all times in seconds):

| field | meaning |
| :-- | :-- |
| `parent_steps`, `child_substeps` | accepted steps of each domain |
| `parent_retries`, `child_retries` | rejected (rolled-back) steps |
| `n_last`, `n_min`, `n_max` | child substeps per parent step |
| `dt_parent_min/max`, `dt_child_min/max` | accepted step sizes |
| `max_child_cfl` | largest advective CFL `dt_c · max_cell Σ|uᵢ|/Δxᵢ` of an accepted child substep, at its start |
| `max_clock_error` | largest `|t_child − t_parent| / t_parent` before the child clock is snapped to the parent's |
| `wall_parent`, `wall_child`, `wall_boundary`, `wall_feedback` | wall time in the parent step, the child substeps, the prolongation of boundary data, the two-way feedback |
"""
mutable struct NestRunStats
    parent_steps::Int
    child_substeps::Int
    parent_retries::Int
    child_retries::Int
    n_last::Int
    n_min::Int
    n_max::Int
    dt_parent_min::Float64
    dt_parent_max::Float64
    dt_child_min::Float64
    dt_child_max::Float64
    max_child_cfl::Float64
    max_clock_error::Float64
    wall_parent::Float64
    wall_child::Float64
    wall_boundary::Float64
    wall_feedback::Float64
end

NestRunStats() = NestRunStats(0, 0, 0, 0, 0, typemax(Int), 0, Inf, 0.0, Inf, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# The post_predictor! hook through which a run refreshes the child halos at
# the times the child's core asks for (see `_halo_write_time`).
mutable struct _BoundaryHook{R} <: Function
    run::R
    k::Int
    t0::Float64
    dt::Float64
end

function (h::_BoundaryHook)(sim, Δt_stage)
    h.k += 1
    _write_boundary!(h.run, _halo_write_time(h.run.advance_child, h.k, h.t0, h.dt, Δt_stage))
    return sim
end

"""
    NestedRun(parent, child, nest; kwargs...)

A parent and a child simulation coupled through `nest` (a
[`NestedGrid`](@ref)) and advanced together, one parent step per
[`step_nested!`](@ref)`(run)`. Either domain may be a
`BoussinesqSimulation` or a sound-proof `MassWeightedSimulation`.

Keywords:

  * `advance_parent`, `advance_child` — per-domain [`DomainAdvance`](@ref)
    closures (default `domain_advance(step!)`), each carrying its own
    hooks and keywords;
  * `cfl_parent`, `cfl_child` — `CFLConstraints` for each domain's
    [`adaptive_dt`](@ref) (`cfl_child` defaults to `cfl_parent`).
    `cfl_parent = nothing` means a fixed step: pass `dt` to
    `step_nested!`. `cfl_child = nothing` means a fixed number of child
    substeps, `substeps` (default `max(rx, ry, rz)`);
  * `sync` — `:subcycle` (default): the parent takes its own adaptive
    step and the child subcycles to land exactly on its end;
    `:shared_min_dt`: both domains take one step of the smaller of the two
    limits (PALM's scheme, Hellsten et al. 2021, §3.1);
  * `feedback` — `:one_way` (default; the parent never sees the child) or
    `:two_way` (conservative restriction of the child field set onto the
    parent, then the interface flux correction, as in
    Berger & Colella 1989);
  * `order` (3), `sponge` (`nothing` or a [`NestSponge`](@ref)),
    `flux_correction` (true), `scalars`, `state` — as for the operators of
    [`NestInterpState`](@ref);
  * `impose_interface` (**true**) — also write the boundary data into the
    child's boundary-normal interior face planes (Dirichlet normal
    velocity) along its nest directions, at every stage. With the child's periodic FFT projection
    (the open-boundary projection is N-open-a/b), a Taylor-Green child
    without it, and without a sponge, blew up within about ten steps;
    with it, it ran 1000 adaptive parent steps (test/era2_nested_run.jl);
  * `max_retries` (3), `divergence_tol` (`Inf`, parent only, through
    `max_divergence`), `max_velocity` (`Inf`), `validate` (true) — the
    step-rejection test, as in
    [`step_adaptive!`](@ref Aeolus.TimeIntegration.step_adaptive!);
  * `reuse_boundary` (true) — in one-way runs, take the boundary data at
    the start of a parent step from the end of the previous one instead of
    prolongating again (valid as long as nothing but `advance_parent`
    changes the parent);
  * `record` (true) — write the two-way conservation ledger
    ([`nest_conservation_error`](@ref)).

The clocks of the two simulations must agree when the run is built
(`check_clocks = false` lets the child keep its own clock origin; it then
advances by the parent's `Δt_p` per step).
`run.t` is the common time, `run.stats` a [`NestRunStats`](@ref).
"""
mutable struct NestedRun{P<:_NestSim, C<:_NestSim, N<:NestedGrid, S<:NestInterpState,
                         AP<:DomainAdvance, AC<:DomainAdvance, CP, CC, SP}
    parent::P
    child::C
    nest::N
    s_old::S
    s_new::S
    s_bc::S
    advance_parent::AP
    advance_child::AC
    cfl_parent::CP
    cfl_child::CC
    feedback::Symbol
    sync::Symbol
    order::Int
    region::Symbol
    sponge::SP
    impose_interface::Bool
    flux_correction::Bool
    substeps::Int
    max_retries::Int
    divergence_tol::Float64
    max_velocity::Float64
    validate::Bool
    reuse_boundary::Bool
    record::Bool
    synced::Bool                   # the child runs on the parent's clock
    # clock and the current parent-step window [t0, t0 + Δt]
    t::Float64
    t0::Float64
    Δt::Float64
    new_valid_at::Float64          # parent time s_new was prolongated at (NaN: stale)
    dt_prev_parent::Union{Nothing,Float64}
    dt_prev_child::Union{Nothing,Float64}
    hook::Any
    stats::NestRunStats
end

function NestedRun(parent::_NestSim, child::_NestSim, nest::NestedGrid;
                   advance_parent::DomainAdvance = domain_advance(),
                   advance_child::DomainAdvance = domain_advance(),
                   cfl_parent = CFLConstraints(),
                   cfl_child = cfl_parent,
                   sync::Symbol = :subcycle,
                   feedback::Symbol = :one_way,
                   order::Int = 3,
                   sponge::Union{Nothing,NestSponge} = nothing,
                   impose_interface::Bool = true,
                   flux_correction::Bool = true,
                   scalars = (),
                   state = nothing,
                   substeps::Union{Nothing,Integer} = nothing,
                   max_retries::Integer = 3,
                   divergence_tol::Real = Inf,
                   max_velocity::Real = Inf,
                   validate::Bool = true,
                   reuse_boundary::Bool = true,
                   record::Bool = true,
                   check_clocks::Bool = true)
    feedback in (:one_way, :two_way) ||
        error("NestedRun: feedback must be :one_way or :two_way, got :$feedback")
    sync in (:subcycle, :shared_min_dt) ||
        error("NestedRun: sync must be :subcycle or :shared_min_dt, got :$sync")
    sync === :shared_min_dt && (cfl_parent === nothing || cfl_child === nothing) &&
        error("NestedRun: sync = :shared_min_dt needs cfl_parent and cfl_child")
    size(parent.θ.data) == totalsize(nest.parent_mesh) ||
        error("NestedRun: the parent is not on the nest's parent mesh")
    size(child.θ.data) == totalsize(nest.child_mesh) ||
        error("NestedRun: the child is not on the nest's child mesh")
    t = Float64(parent.t)
    !check_clocks || abs(Float64(child.t) - t) <= 1e-12 * max(1.0, abs(t)) ||
        error("NestedRun: parent clock $(parent.t) and child clock $(child.t) differ")
    ns = substeps === nothing ? max(nest.rx, nest.ry, nest.rz) : Int(substeps)
    ns >= 1 || error("NestedRun: substeps must be >= 1")
    s_old, s_new, s_bc = state === nothing ?
        ntuple(_ -> nest_interp_state(nest, parent, child; scalars), 3) : state
    region = sponge !== nothing ? :all : impose_interface ? :halo_interface : :halo
    run = NestedRun(parent, child, nest, s_old, s_new, s_bc, advance_parent, advance_child,
                    cfl_parent, cfl_child, feedback, sync, order, region, sponge,
                    impose_interface, flux_correction, ns, Int(max_retries),
                    Float64(divergence_tol), Float64(max_velocity), validate,
                    reuse_boundary, record, check_clocks, t, t, 0.0, NaN, nothing, nothing,
                    nothing,
                    NestRunStats())
    run.hook = _BoundaryHook(run, 0, 0.0, 0.0)
    return run
end

function Base.show(io::IO, r::NestedRun)
    s = r.stats
    print(io, "NestedRun(", r.feedback, ", ", r.sync, ", t=", r.t, ", ",
          s.parent_steps, " parent steps, ", s.child_substeps, " child substeps)")
end

"""
    boundary_time(run, t) -> NestInterpState

The child's boundary data at time `t` (child clock; the parent's, in a
synchronised run) of the current parent step
`[t_p, t_p + Δt_p]`: the linear blend
`(1 − τ) B(t_p) + τ B(t_p + Δt_p)`, `τ = (t − t_p)/Δt_p`, of the parent
states prolongated at the two ends of the step, over the whole field set
(`u, v, w, θ`, `qv`, extra scalars). Linear interpolation in time is
second-order accurate, `O(Δt_p²)`. The result is the run's scratch state;
read it before the next call.
"""
function boundary_time(run::NestedRun, t::Real)
    run.Δt > 0 || error("boundary_time: the run has not taken a parent step yet")
    tol = 1e-9 * run.Δt
    run.t0 - tol <= t <= run.t0 + run.Δt + tol ||
        error("boundary_time: t = $t is outside the current parent step " *
              "[$(run.t0), $(run.t0 + run.Δt)]")
    _blend_state!(run.s_bc, run.s_old, run.s_new, (t - run.t0) / run.Δt)
    return run.s_bc
end

# Halos of the child (and its interface planes when imposed) ← B(t).
function _write_boundary!(run::NestedRun, t::Real)
    τ = (t - run.t0) / run.Δt
    c = run.child; a = run.s_old; b = run.s_new
    ii = run.impose_interface
    dx, dy, dz = _nest_dirs(c)
    _blend_halo!(c.u, a.u, b.u, τ; interface=ii && dx)
    _blend_halo!(c.v, a.v, b.v, τ; interface=ii && dy)
    _blend_halo!(c.w, a.w, b.w, τ; interface=ii && dz)
    _blend_halo!(c.θ, a.θ, b.θ, τ)
    for i in eachindex(a.names)
        _blend_halo!(a.child[i], a.bufs[i], b.bufs[i], τ)
    end
    return run
end

# --- step validation and rollback ----------------------------------------

_bound_fields(run::NestedRun, which::Symbol) =
    which === :child ? run.s_old.child : run.s_old.parent

mutable struct _Rollback{SN}
    snap::SN
    scalars::Vector{Any}
    sgs::Any
end

function _save!(run::NestedRun, sim, adv::DomainAdvance, which::Symbol)
    snap = state_snapshot(sim)
    save_state!(snap, sim)
    fs = _bound_fields(run, which)
    sc = Any[copy(f.data) for f in fs]
    closure = sgs_closure(_domain_sgs(adv))
    return _Rollback(snap, sc, closure === nothing ? nothing : (closure, sgs_state(closure)))
end

function _restore!(run::NestedRun, sim, rb::_Rollback, which::Symbol)
    restore_state!(sim, rb.snap)
    for (f, d) in zip(_bound_fields(run, which), rb.scalars)
        copyto!(f.data, d)
    end
    rb.sgs === nothing || restore_sgs_state!(rb.sgs[1], rb.sgs[2])
    return sim
end

_max_speed(sim) = max(maximum(abs, interior_view(sim.u)), maximum(abs, interior_view(sim.v)),
                      maximum(abs, interior_view(sim.w)))

function _accept(run::NestedRun, sim, which::Symbol)
    _all_finite(sim) || return false
    for f in _bound_fields(run, which)
        all(isfinite, interior_view(f)) || return false
    end
    _max_speed(sim) <= run.max_velocity || return false
    if which === :parent && isfinite(run.divergence_tol)
        d = max_divergence(sim)
        (isfinite(d) && d <= run.divergence_tol) || return false
    end
    return true
end

function _try_advance!(adv::DomainAdvance, sim, h, hook)
    try
        adv(sim, h; post_predictor! = hook)
        return true
    catch err
        err isa InterruptException && rethrow()
        return false
    end
end

# --- the step size of a domain --------------------------------------------

_dt_limit(sim, adv::DomainAdvance, c::CFLConstraints, dt_prev) =
    Float64(adaptive_dt(sim, c; dt_prev = dt_prev, scheme = _imex_scheme(adv),
                        sgs = _domain_sgs(adv),
                        sgs_stress = get(adv.kwargs, :sgs_stress, :laplacian)))

# Substeps to cover `R` with steps ≤ `lim`. A ratio that is an integer up
# to round-off (1e-9 relative) is not rounded up: a child whose limit is
# exactly Δt_p/r takes r substeps, not r + 1.
_nsub(R, lim) = max(1, ceil(Int, (R / lim) * (1 - 1e-9)))

_advective_rate(sim) = (fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs);
                        fill_halos!(sim.w, sim.bcs);
                        _max_advective_rate(sim.u.data, sim.v.data, sim.w.data, sim.mesh))

# =========================================================================
# One parent step
# =========================================================================

"""
    step_nested!(run::NestedRun; dt = nothing) -> run

Advance `run` by one parent step:

 1. the parent step `Δt_p` is `adaptive_dt(parent)` (or `dt`; under
    `sync = :shared_min_dt` the smaller of the two domains' limits);
 2. the parent field set is prolongated onto the child at `t_p` (in a
    one-way run, reused from the end of the previous step);
 3. the parent takes its step. A rejected parent step (non-finite state,
    exception, `divergence_tol`, `max_velocity`) is rolled back and
    retried with `Δt_p/2` — **before** any child substep;
 4. the parent field set is prolongated at `t_p + Δt_p`;
 5. the child subcycles to land exactly on `t_p + Δt_p`:
    `n = ⌈Δt_p / Δt_c^max⌉` substeps, re-evaluated before every substep as
    `n_rem = max(n_rem, ⌈R/Δt_c^max⌉)` with `Δt_c = R / n_rem` (`R` the
    remaining time; the last substep takes exactly `R`). The child halos
    hold the linearly interpolated boundary data at every stage time of
    its core ([`stage_times`](@ref), [`boundary_time`](@ref)). A rejected
    child substep rolls back **only the child** (its `StateSnapshot`, its
    bound scalars and its SGS memory); the rejected piece is redone at
    half its size and the remaining time is re-divided;
 6. two-way: conservative restriction and interface flux correction.

The child clock is set to the parent's at the end (the discrepancy it
had is recorded in `run.stats.max_clock_error`).
"""
function step_nested!(run::NestedRun; dt = nothing)
    p = run.parent; c = run.child; nest = run.nest; st = run.stats
    t_p = Float64(p.t)

    # --- 1. parent step size ---------------------------------------------
    Δt = if dt !== nothing
        Float64(dt)
    elseif run.cfl_parent === nothing
        error("step_nested!: the run has no cfl_parent; pass dt")
    else
        lim = _dt_limit(p, run.advance_parent, run.cfl_parent, run.dt_prev_parent)
        run.sync === :shared_min_dt ?
            min(lim, _dt_limit(c, run.advance_child, run.cfl_child, run.dt_prev_child)) : lim
    end
    Δt > 0 || error("step_nested!: parent step must be > 0, got $Δt")

    two_way = run.feedback === :two_way
    if two_way && run.record
        I0 = _composite_scalars(p, c, nest, run.s_bc)
        m0 = composite_integral(p, c, nest, :momentum_x)
    end

    # --- 2. boundary data at t_p -------------------------------------------
    tb = time_ns()
    if !two_way && run.reuse_boundary && run.new_valid_at == t_p
        run.s_old, run.s_new = run.s_new, run.s_old
    else
        interpolate_parent_state!(run.s_old, p, nest; order=run.order, region=run.region)
    end
    run.new_valid_at = NaN
    st.wall_boundary += (time_ns() - tb) / 1e9

    # --- 3. parent step, with retries before any child substep -------------
    tw = time_ns()
    Δt = _advance_parent!(run, Δt)
    st.wall_parent += (time_ns() - tw) / 1e9
    t_end = Float64(p.t)

    # --- 4. boundary data at t_p + Δt_p ------------------------------------
    tb = time_ns()
    interpolate_parent_state!(run.s_new, p, nest; order=run.order, region=run.region)
    run.new_valid_at = t_end
    st.wall_boundary += (time_ns() - tb) / 1e9
    # the window in the child's clock (the parent's, in a synchronised run,
    # where the child lands on the parent's clock bit for bit)
    run.t0 = Float64(c.t)
    run.Δt = t_end - t_p

    # --- 5. child substeps --------------------------------------------------
    tw = time_ns()
    _subcycle_child!(run, run.synced ? t_end : run.t0 + run.Δt)
    st.wall_child += (time_ns() - tw) / 1e9

    # --- 6. feedback ---------------------------------------------------------
    if two_way
        tw = time_ns()
        _two_way_feedback!(run)
        if run.record
            I1 = _composite_scalars(p, c, nest, run.s_bc)
            for (k, v0) in I0
                nest.ledger[k] = abs(I1[k] - v0) / max(abs(v0), 1e-300)
            end
            m1 = composite_integral(p, c, nest, :momentum_x)
            nest.ledger[:momentum_x] = abs(m1 - m0) / max(abs(m0), 1e-300)
        end
        run.new_valid_at = NaN                 # the parent changed after s_new
        st.wall_feedback += (time_ns() - tw) / 1e9
    end

    st.parent_steps += 1
    st.dt_parent_min = min(st.dt_parent_min, Δt)
    st.dt_parent_max = max(st.dt_parent_max, Δt)
    run.dt_prev_parent = Δt
    run.t = t_end
    return run
end

function _advance_parent!(run::NestedRun, Δt::Float64)
    p = run.parent; adv = run.advance_parent
    run.validate || (adv(p, Δt); return Δt)
    rb = _save!(run, p, adv, :parent)
    dt_min = run.cfl_parent === nothing ? 0.0 : Float64(run.cfl_parent.dt_min)
    tries = 0
    while true
        _try_advance!(adv, p, Δt, nothing) && _accept(run, p, :parent) && return Δt
        _restore!(run, p, rb, :parent)
        tries += 1
        run.stats.parent_retries += 1
        Δt_new = max(Δt / 2, dt_min)
        (tries > run.max_retries || Δt_new == Δt) &&
            error("step_nested!: the parent step failed $tries times (last dt = $Δt); " *
                  "the parent has been restored to t = $(p.t), the child was not advanced.")
        Δt = Δt_new
    end
end

function _subcycle_child!(run::NestedRun, t_end::Float64)
    c = run.child; adv = run.advance_child; st = run.stats
    adaptive = run.sync === :subcycle && run.cfl_child !== nothing
    R = t_end - c.t
    n_rem = run.sync === :shared_min_dt ? 1 :
            adaptive ? _nsub(R, _dt_limit(c, adv, run.cfl_child, run.dt_prev_child)) :
            run.substeps
    first_piece = true
    n_taken = 0
    tries = 0
    h_retry = 0.0                       # > 0: redo a rejected piece at this size
    hook = run.hook
    while n_rem > 0
        R = t_end - c.t
        if adaptive && !first_piece
            n_rem = max(n_rem, _nsub(R, _dt_limit(c, adv, run.cfl_child, run.dt_prev_child)))
        end
        first_piece = false
        h = h_retry > 0 ? h_retry : n_rem == 1 ? R : R / n_rem
        cfl = h * _advective_rate(c)

        rb = run.validate ? _save!(run, c, adv, :child) : nothing
        t0 = Float64(c.t)
        _write_boundary!(run, t0)
        hook.k = 0; hook.t0 = t0; hook.dt = h
        ok = if run.validate
            _try_advance!(adv, c, h, hook) && _accept(run, c, :child)
        else
            adv(c, h; post_predictor! = hook); true
        end
        if !ok
            _restore!(run, c, rb, :child)
            tries += 1
            st.child_retries += 1
            tries > run.max_retries &&
                error("step_nested!: a child substep failed $tries times (last dt = $h) " *
                      "in the parent step ending at t = $t_end; the parent has been " *
                      "advanced, the child restored to t = $(c.t).")
            h_retry = h / 2             # the rejected piece becomes two halves
            n_rem += 1
            continue
        end
        h_retry = 0.0
        tries = 0
        if run.sponge !== nothing
            _blend_state!(run.s_bc, run.s_old, run.s_new, (Float64(c.t) - run.t0) / run.Δt)
            apply_nest_sponge!(c, run.s_bc, run.sponge; dt=h)
        end
        n_taken += 1
        n_rem -= 1
        st.child_substeps += 1
        st.dt_child_min = min(st.dt_child_min, h)
        st.dt_child_max = max(st.dt_child_max, h)
        st.max_child_cfl = max(st.max_child_cfl, cfl)
        run.dt_prev_child = h
    end
    st.max_clock_error = max(st.max_clock_error,
                             abs(Float64(c.t) - t_end) / max(abs(t_end), floatmin(Float64)))
    c.t = t_end
    st.n_last = n_taken
    st.n_min = min(st.n_min, n_taken)
    st.n_max = max(st.n_max, n_taken)
    return run
end

function _two_way_feedback!(run::NestedRun)
    p = run.parent; c = run.child; nest = run.nest
    meas = run.s_bc.measure
    restrict_state_to_parent!(p, c, nest, run.s_bc)
    e_before = _interface_imbalance(p, c, nest, meas)
    if run.flux_correction
        for (pu, cu) in ((p.u, c.u), (p.v, c.v), (p.w, c.w))
            apply_flux_correction!(pu, compute_nest_flux_mismatch(pu, cu, nest; measure=meas),
                                   nest; measure=meas)
        end
    end
    if run.record
        nest.ledger[:mass] = _interface_imbalance(p, c, nest, meas)
        nest.ledger[:mass_before_correction] = e_before
    end
    return run
end

# =========================================================================
# The historical fixed-dt entry point
# =========================================================================

"""
    step_nested!(parent_sim, child_sim, nest, dt; substeps=nothing, order=3,
                 flux_correction=true, sponge=nothing, impose_interface=false,
                 scalars=(), state=nothing, record=true)

One **two-way** parent step of fixed size `dt` plus `substeps` equal
child steps (default `max(rx,ry,rz)`, which keeps the child CFL equal to
the parent's), both with `step!` and no hooks. A thin wrapper over
[`step_nested!`](@ref)`(::NestedRun)`: it builds a
[`NestedRun`](@ref) with `feedback = :two_way`, fixed steps and no step
validation, and takes one step. Use a `NestedRun` directly for adaptive
steps, hooks, other dynamical cores or one-way nesting.

The transferred field set is `u, v, w, θ`, `qv` when both simulations
are moist, and the extra cell-centred `scalars`
(`:co2 => (child_field, parent_field)`, …); see [`NestInterpState`](@ref).
The child halos hold the parent state linearly interpolated in time to
the child's stage times; after the child substeps the child field set is
conservatively restricted onto the parent
([`restrict_state_to_parent!`](@ref)) and the interface flux mismatch is
corrected. Conservation diagnostics for `:mass`, `:momentum_x` and every
scalar of the set are recorded in `nest.ledger`; read them with
[`nest_conservation_error`](@ref). Returns `(parent_sim, child_sim)`.

`state` may be a pre-allocated `NTuple{3,NestInterpState}` (built with
`nest_interp_state(nest, parent_sim, child_sim; scalars)`) to avoid
re-allocating the boundary buffers every step; it then defines the field
set and `scalars` is ignored.
"""
function step_nested!(psim::_NestSim, csim::_NestSim, nest::NestedGrid, dt::Real;
                      substeps::Union{Nothing,Integer}=nothing,
                      order::Int=3,
                      flux_correction::Bool=true,
                      sponge::Union{Nothing,NestSponge}=nothing,
                      impose_interface::Bool=false,
                      scalars=(),
                      state=nothing,
                      record::Bool=true)
    run = NestedRun(psim, csim, nest; cfl_parent=nothing, cfl_child=nothing,
                    substeps, feedback=:two_way, order, flux_correction, sponge,
                    impose_interface, scalars, state, record, validate=false,
                    check_clocks=false)
    step_nested!(run; dt)
    return (psim, csim)
end
