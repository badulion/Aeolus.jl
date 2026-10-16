# Nesting: the time coupling of a parent and a child (N-nest-c, #257).
# (DomainAdvance, the per-domain advance closure, is in domain_advance.jl.)
#
#   * NestedRun     — parent + child + NestedGrid + boundary buffers +
#     schedule (subcycling under adaptive steppers, or a shared minimum
#     dt) + feedback (one-way / two-way) + statistics. The child's nest
#     faces are OpenBoundary faces fed by a NestBoundaryData (N-nest-d,
#     #261): every core refreshes them itself at its stage times;
#   * step_nested!  — one parent step of a NestedRun; the positional
#     method is the historical fixed-dt two-way entry point, now a thin
#     wrapper over a NestedRun.

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
  * `feedback` — `:two_way` (the default: the default
    [`TwoWayFeedback`](@ref), as for every Aeolus parent driving an Aeolus
    child, ONE-f #455), a `TwoWayFeedback(…)`, or `:one_way` (the parent
    never sees the child; what a mixed core pair, or a child without a
    conservation box, needs). Two-way is anterpolation of the child field set onto the
    covered parent cells outside a buffer zone, an optional relaxation,
    canopy restriction and inflow exclusion, the interface flux
    correction on the open faces, Berger–Colella scalar refluxing from
    flux registers and the re-projection of the parent (N-nest-e, #263);
  * `limit_scalars` (`:positive`, for `qv` and the tracers), `limit_theta`
    (`:none`), `limit_momentum` (`:none`) — the limiters of the
    prolongation (#329), see [`interpolate_parent_state!`](@ref);
  * `substeps` — the fixed number of child substeps when `cfl_child = nothing`
    (default: the largest refinement ratio, rounded up for a rational one);
  * `order` (3), `flux_correction` (true), `scalars`, `state` — as for
    the operators of [`NestInterpState`](@ref). The run allocates its three
    field sets (the parent prolongated at both ends of the parent step and
    their blend) with `storage = :shell`, halo slabs only, unless it has a
    sponge, whose target is the child interior (#329);
  * `sponge` (`nothing`, the default, as in Kadasch et al. 2021 §2.2–2.3
    and Hellsten et al. 2021, which use none) or a [`NestSponge`](@ref):
    relaxation of every member of the field set toward the time-blended
    parent (at the end of the child substep), on the child's open faces
    only ([`nest_open_sides`](@ref)). The velocities are relaxed on the
    predicted velocity of every stage, in the core's `post_predictor!`
    hook, so the pressure solve that follows keeps the child
    divergence-free; the scalars after each child substep (#329). A user
    core must call `post_predictor!` for the velocity part to act;
  * `solid` (`nothing`) — the child's IBM [`SolidMask`](@ref) (or a
    `Bool` array). The open faces inside solids are marked with
    [`mark_open_boundary_solids!`](@ref
    Aeolus.BoundaryConditions.mark_open_boundary_solids!): zero normal and
    tangential velocity there, left out of the mass-flux correction; and
    the two-way restriction is solid-aware ([`restrict_state_to_parent!`](@ref));
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

A same-core pair (two Boussinesq sims, or two `MassWeightedSimulation`s
of one sound-proof system sharing one `ReferenceState`, whose transfers
then use the reference-mass measure [`nest_measure`](@ref)) needs nothing
more. A **mixed pair** — a Boussinesq domain and an anelastic one, either
way round, one-way or two-way (NEST-p2, #482) — needs
`boussinesq_density`, the Boussinesq side's constant density
(`p_ref / (R_d θ₀)`): the transfers then measure each side with its own
density and convert the velocities so that the mass flux through the
interface, and with it every scalar flux, is the same on both sides
([`NestCoreConversion`](@ref), [`nest_core_conversion`](@ref)).

**Boundary conditions.** The child must be built with
[`nest_bcs`](@ref)`(child_mesh, …; moisture, scalars)` naming the same
field set as the run: its nest faces are
[`OpenBoundary`](@ref Aeolus.BoundaryConditions.OpenBoundary) faces, and
the run attaches a [`NestBoundaryData`](@ref) to them. Every core then
refreshes the child's boundary at its own stage times
(`update_open_boundaries!`), with the parent field set linearly blended in
time over the parent step and the single global mass-flux correction of
N-open-b, so the child's pressure solve (all-Neumann `CosineFFTPoissonSolver`)
sees the parent's boundary flux. A child without open faces (periodic or
walled everywhere) runs, but nothing reaches it from the parent.

The clocks of the two simulations must agree when the run is built
(`check_clocks = false` lets the child keep its own clock origin; it then
advances by the parent's `Δt_p` per step).
`run.t` is the common time, `run.stats` a [`NestRunStats`](@ref).
"""
mutable struct NestedRun{P<:_NestSim, C<:_NestSim, N<:NestedGrid, S<:NestInterpState,
                         AP<:DomainAdvance, AC<:DomainAdvance, CP, CC, SP, D<:NestBoundaryData,
                         SO, FM, RP, RC, R<:Union{Nothing,NestFluxRegisters}, W}
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
    limit_scalars::Symbol          # the prolongation limiters (#329)
    limit_theta::Symbol
    limit_momentum::Symbol
    sponge::SP
    data::D                        # the child's boundary data (NestBoundaryData)
    flux_correction::Bool
    substeps::Int
    max_retries::Int
    divergence_tol::Float64
    max_velocity::Float64
    validate::Bool
    reuse_boundary::Bool
    record::Bool
    solid::SO                      # child solid mask as given (or nothing)
    fluid::FM                      # its fluid mask for the restriction, computed once
    synced::Bool                   # the child runs on the parent's clock
    # clock and the current parent-step window [t0, t0 + Δt]
    t::Float64
    t0::Float64
    Δt::Float64
    new_valid_at::Float64          # parent time s_new was prolongated at (NaN: stale)
    dt_prev_parent::Union{Nothing,Float64}
    dt_prev_child::Union{Nothing,Float64}
    sub_t0::Float64                # start and length of the current child substep
    sub_h::Float64
    sponge_τ::Float64              # window fraction s_bc holds the sponge target at (NaN: stale)
    stats::NestRunStats
    twoway::Union{Nothing,TwoWayFeedback}
    regs::R
    rb_parent::RP                  # preallocated rollback state of each domain
    rb_child::RC
    work::W                        # two-way scratch (nothing in a one-way run)
    step_record::StepRecord        # the run's per-step record and diagnostics hook (#336)
    retry_mark::Int                # parent + child retries at the start of the window
    conv::Union{Nothing,NestCoreConversion} # a mixed core pair's density conversion (NEST-p2, #482)
end

# Scratch of the two-way feedback, allocated once (#329, item 6): the
# feedback mask, the anterpolation boxes, the reference values of the
# refluxed scalars and the composite integrals of the ledger.
struct _TwoWayWork{AS}
    mask::BitArray{3}
    anterp::AS
    refs::Vector{Float64}
    I0::Vector{Float64}
    I1::Vector{Float64}
    keys::Vector{Symbol}
end

_two_way_work(nest, s) = _TwoWayWork(BitArray(undef, nest.nx, nest.ny, nest.nz),
                                     _anterp_scratch(nest), Float64[], Float64[], Float64[],
                                     _scalar_ledger_keys(s))

function NestedRun(parent::_NestSim, child::_NestSim, nest::NestedGrid;
                   advance_parent::DomainAdvance = domain_advance(),
                   advance_child::DomainAdvance = domain_advance(),
                   cfl_parent = CFLConstraints(),
                   cfl_child = cfl_parent,
                   sync::Symbol = :subcycle,
                   feedback::Union{Symbol,TwoWayFeedback} = :two_way,
                   order::Int = 3,
                   limit_scalars::Symbol = :positive,
                   limit_theta::Symbol = :none,
                   limit_momentum::Symbol = :none,
                   sponge::Union{Nothing,NestSponge} = nothing,
                   solid = nothing,
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
                   check_clocks::Bool = true,
                   boussinesq_density::Real = NaN)
    twoway = feedback isa TwoWayFeedback ? feedback :
             feedback === :two_way ? TwoWayFeedback() :
             feedback === :one_way ? nothing :
             error("NestedRun: feedback must be :one_way, :two_way or a TwoWayFeedback, " *
                   "got :$feedback")
    feedback = twoway === nothing ? :one_way : :two_way
    sync in (:subcycle, :shared_min_dt) ||
        error("NestedRun: sync must be :subcycle or :shared_min_dt, got :$sync")
    conv = nest_core_conversion(nest, parent, child; rho0 = boussinesq_density)
    feedback === :two_way && conv === nothing && _check_two_way_pair(parent, child, "NestedRun")
    sync === :shared_min_dt && (cfl_parent === nothing || cfl_child === nothing) &&
        error("NestedRun: sync = :shared_min_dt needs cfl_parent and cfl_child")
    size(parent.θ.data) == totalsize(nest.parent_mesh) ||
        error("NestedRun: the parent is not on the nest's parent mesh")
    size(child.θ.data) == totalsize(nest.child_mesh) ||
        error("NestedRun: the child is not on the nest's child mesh")
    t = Float64(parent.t)
    !check_clocks || abs(Float64(child.t) - t) <= 1e-12 * max(1.0, abs(t)) ||
        error("NestedRun: parent clock $(parent.t) and child clock $(child.t) differ")
    ns = substeps === nothing ? max(1, ceil(Int, _max_ratio(nest) * (1 - 1e-9))) : Int(substeps)
    feedback === :two_way && _check_two_way_box(nest, "NestedRun")
    ns >= 1 || error("NestedRun: substeps must be >= 1")
    # without a sponge only the child's open-boundary points are read: the
    # three field sets keep their halo slabs only (#329, item 4)
    storage = sponge === nothing ? :shell : :full
    s_old, s_new, s_bc = state !== nothing ? state :
        conv === nothing ? ntuple(_ -> nest_interp_state(nest, parent, child; scalars, storage), 3) :
        ntuple(_ -> nest_interp_state(nest; scalars = (_mixed_moisture(parent, child)..., scalars...),
                                      measure = _conversion_measure(nest, conv), storage), 3)
    sponge === nothing || all(st -> nest_storage(st) === :full, (s_old, s_new, s_bc)) ||
        throw(ArgumentError("NestedRun: a sponge needs storage = :full states (it relaxes " *
                            "the child interior toward them)"))
    feedback === :two_way && _check_two_way_pair(parent, child, "NestedRun", s_bc)
    # the boundary buffers hold the halos *and* the boundary-normal interior
    # face plane of the west/south/bottom faces; the sponge needs the whole field
    region = sponge !== nothing ? :all : :halo_interface
    sp = sponge === nothing ? nothing :
         NestSponge{Float64}(sponge.width, sponge.α_max, sponge.profile,
                             sponge.sides .& nest_open_sides(child))
    # the child's boundary: the parent prolongated now, attached to its open faces
    _prolongate_conv!(s_old, parent, nest, conv; order, region, limit_scalars, limit_theta,
                      limit_momentum)
    _mark_nest_solids!(child, solid)
    data = nest_boundary_data(s_old, s_new; t0 = child.t, Δt = 0.0)
    _attach_nest_data!(child, data)
    fill_nest_halos!(child, s_old)
    regs = twoway !== nothing && twoway.reflux ?
           NestFluxRegisters(nest, parent, child, s_bc, twoway, advance_parent, advance_child, conv) :
           nothing
    run = NestedRun(parent, child, nest, s_old, s_new, s_bc, advance_parent, advance_child,
                    cfl_parent, cfl_child, feedback, sync, order, region,
                    limit_scalars, limit_theta, limit_momentum, sp, data,
                    flux_correction, ns, Int(max_retries),
                    Float64(divergence_tol), Float64(max_velocity), validate,
                    reuse_boundary, record, solid, _fluid_mask(nest, solid), check_clocks,
                    t, t, 0.0, NaN, nothing, nothing,
                    Float64(child.t), 0.0, NaN, NestRunStats(), twoway, regs,
                    _Rollback(parent, advance_parent, s_bc.parent),
                    _Rollback(child, advance_child, s_bc.child),
                    twoway === nothing ? nothing : _two_way_work(nest, s_bc), StepRecord(), 0, conv)
    if run.regs !== nothing                  # separate transports book through the sims
        _register_booker!(parent, run.regs, :parent, nest)
        _register_booker!(child, run.regs, :child, nest)
    end
    return run
end

"""
    step_record(run::NestedRun) -> StepRecord

The run's own [`StepRecord`](@ref Aeolus.Simulations.StepRecord) (#336),
completed at the end of every parent step
(a nest engine's window too): `dt` the parent step, `limiter` the parent's, `retries` the
rejected parent steps and child substeps of the window, `cfl_adv` the
largest advective CFL number of a child substep (at its start), the wall
time of the whole window; its hook ([`set_step_hook!`](@ref
Aeolus.Simulations.set_step_hook!)) is then called as `hook(run, record)`.
Each domain's own record is completed once per **accepted** step of that
domain: the parent step with its retries, each child substep with the
retries of its piece (a rejected attempt is not a step), and the domain's
hook is called then.
"""
step_record(run::NestedRun) = run.step_record

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
read it before the next call. Calling it from a child hook (`body_force!`,
`post_predictor!`) is safe in a sponged run: the sponge re-blends its own
target (the same scratch state) after any such call.
"""
function boundary_time(run::NestedRun, t::Real)
    run.Δt > 0 || error("boundary_time: the run has not taken a parent step yet")
    tol = 1e-9 * run.Δt
    run.t0 - tol <= t <= run.t0 + run.Δt + tol ||
        error("boundary_time: t = $t is outside the current parent step " *
              "[$(run.t0), $(run.t0 + run.Δt)]")
    _blend_state!(run.s_bc, run.s_old, run.s_new, (t - run.t0) / run.Δt)
    run.sponge_τ = NaN       # s_bc no longer holds the sponge target (#329)
    return run.s_bc
end

# The parent field set prolongated into one of the run's states (a mixed
# core pair: the parent's velocity in the child's density, NestCoreConversion).
_prolongate!(run::NestedRun, s::NestInterpState) =
    _prolongate_conv!(s, run.parent, run.nest, run.conv; order = run.order, region = run.region,
                      limit_scalars = run.limit_scalars,
                      limit_theta = run.limit_theta,
                      limit_momentum = run.limit_momentum)

_prolongate_conv!(s, parent, nest, ::Nothing; kw...) = interpolate_parent_state!(s, parent, nest; kw...)
function _prolongate_conv!(s, parent, nest, cv::NestCoreConversion; kw...)
    _scale_parent!(parent, cv)
    try
        interpolate_parent_state!(s, parent, nest; kw...)
    finally
        _restore_parent!(parent, cv)
    end
    return s
end

# qv of a mixed pair: only when both carry it (the sound-proof core has none)
function _mixed_moisture(p, c)
    _moist(p) == _moist(c) ||
        throw(ArgumentError("NestedRun: the parent carries qv = $(_moist(p)), the child $(_moist(c)); " *
                            "a nest carries qv on both domains or on neither"))
    return _moist(p) ? (:qv => (c.qv, p.qv),) : ()
end

# Point the child's boundary data at the current parent-step window and
# refresh its open boundaries at the child's time (B(t₀) of the new window:
# in a two-way run the parent changed after the previous window's B(t₀+Δt)).
function _set_window!(run::NestedRun)
    d = run.data
    d.a = run.s_old; d.b = run.s_new
    d.t0 = run.t0; d.Δt = run.Δt
    run.sponge_τ = NaN
    fill_nest_halos!(run.child, run.s_old)
    return run
end

# --- the sponge (#329, item 3) ---------------------------------------------

# The sponge target, the parent blended to the end of the current child
# substep, in run.s_bc (blended once per substep).
function _sponge_target!(run::NestedRun)
    τ = (run.sub_t0 + run.sub_h - run.t0) / run.Δt
    τ == run.sponge_τ && return run.s_bc
    _blend_state!(run.s_bc, run.s_old, run.s_new, τ)
    run.sponge_τ = τ
    return run.s_bc
end

# The velocity part of the sponge, chained after the child's own
# post_predictor!: it relaxes the predicted velocity of every stage, before
# the pressure solve, so the projection keeps ∇·u = 0.
struct _SpongeHook{R} <: Function
    run::R
end

function (h::_SpongeHook)(sim, Δt)
    run = h.run
    apply_nest_sponge!(sim, _sponge_target!(run), run.sponge; dt = Δt, scalars = false)
    return nothing
end

# --- step validation and rollback ----------------------------------------

_bound_fields(run::NestedRun, which::Symbol) =
    which === :child ? run.s_old.child : run.s_old.parent

# The state a rejected step is rolled back to, allocated once per domain
# and refilled in place before every attempt (#329: the open-boundary
# buffers were copied into fresh arrays each attempt, 853 KiB per child
# substep of a 32³ child).
mutable struct _Rollback{SN, SC<:AbstractArray, SG, OP}
    snap::SN
    scalars::Vector{SC}             # the bound scalars' data
    sgs::SG                         # (closure, its state), or nothing
    open::OP                        # the open-boundary buffers (part of the state), or nothing
end

function _Rollback(sim, adv::DomainAdvance, fs::AbstractVector)
    closure = sgs_closure(_domain_sgs(adv))
    sg = closure === nothing ? nothing : (closure, sgs_state(closure))
    sc = [copy(f.data) for f in fs]
    return _Rollback(state_snapshot(sim), isempty(sc) ? Array{Float64,3}[] : sc, sg,
                     open_boundary_snapshot(sim))
end

_save!(run::NestedRun, sim, adv::DomainAdvance, which::Symbol) =
    _save_rb!(which === :child ? run.rb_child : run.rb_parent, sim, _bound_fields(run, which))

function _save_rb!(rb::_Rollback, sim, fields)
    save_state!(rb.snap, sim)
    _copy_data!(rb.scalars, fields)
    if rb.sgs !== nothing
        closure = rb.sgs[1]
        rb.sgs = (closure, sgs_state(closure))
    end
    rb.open = _snapshot_open!(rb.open, sim)
    return rb
end

function _copy_data!(dst::AbstractVector, fs::AbstractVector)
    for i in eachindex(dst, fs)
        copyto!(dst[i], fs[i].data)
    end
    return dst
end

function _restore_data!(fs::AbstractVector, src::AbstractVector)
    for i in eachindex(fs, src)
        copyto!(fs[i].data, src[i])
    end
    return fs
end

# The open-boundary buffers copied into the preallocated snapshot (a fresh
# one if planes were added since).
_snapshot_open!(::Nothing, sim) = open_boundary_snapshot(sim)
function _snapshot_open!(snap::AbstractVector, sim)
    n = 0
    for st in open_boundary_states(sim), p in st.planes
        n += 1
        (n <= length(snap) && size(snap[n]) == size(p.data)) ||
            return open_boundary_snapshot(sim)
        copyto!(snap[n], p.data)
    end
    n == length(snap) || return open_boundary_snapshot(sim)
    return snap
end

function _restore_rb!(sim, rb::_Rollback, fields)
    restore_state!(sim, rb.snap)
    _restore_data!(fields, rb.scalars)
    rb.sgs === nothing || restore_sgs_state!(rb.sgs[1], rb.sgs[2])
    restore_open_boundaries!(sim, rb.open)
    return sim
end

# exact and NaN-propagating: the bits of `maximum(abs, interior_view(·))`,
# thread-invariant (KA-b, #296)
_max_speed(sim) = max(_max_abs(sim.u), _max_abs(sim.v),
                      _max_abs(sim.w))

function _accept_fields(run::NestedRun, sim, which::Symbol, fields)
    _all_finite(sim) || return false
    for f in fields
        interior_all_finite(f) || return false
    end
    _max_speed(sim) <= run.max_velocity || return false
    if which === :parent && isfinite(run.divergence_tol)
        d = max_divergence(sim)
        (isfinite(d) && d <= run.divergence_tol) || return false
    end
    return true
end

# One attempted step. A numerical failure (`is_numerical_failure`: a
# `NumericalFailure`, a `DomainError`/`InexactError`/`OverflowError`, or any
# exception thrown from a non-finite state) rejects the step, which is then
# rolled back and retried; anything else — a `MethodError`, a `BoundsError`,
# a wrong argument — is a bug or a set-up error and is rethrown at once
# (#327, M6: it used to become a "rejected step" retried at half the size).
function _try_advance!(adv::DomainAdvance, sim, h, bf = nothing, pp = nothing)
    try
        adv(sim, h; body_force! = bf, post_predictor! = pp)
        return true
    catch err
        is_numerical_failure(err, sim) || rethrow()
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

"""
    subcycle_dt(remaining, limit) -> dt

The next child substep of a parent-step window with `remaining` seconds
left, for a child whose own step limit is `limit`: `remaining / n`,
`n = ⌈remaining / limit⌉` (a ratio that is an integer up to 1e-9 relative
is not rounded up), so the substeps land exactly on the window end. The
rule [`step_nested!`](@ref)'s subcycling re-evaluates before every
substep; a child that picks its own steps in a nest engine's window
(the urban site, [`nest_window_end`](@ref)) uses it to pick them.
"""
subcycle_dt(R::Real, lim::Real) = R / _nsub(R, lim)

"""
    nest_window_end(run::NestedRun) -> t

The child time at which the window that a nest engine opened on `run`
ends (the parent's time at the end of its step, on the parent's clock):
where a child that picks its own steps lands ([`subcycle_dt`](@ref) of
the time left). Read between the window's opening and its close.
"""
nest_window_end(run::NestedRun) = run.synced ? run.new_valid_at : run.t0 + run.Δt

_advective_rate(sim) = (fill_halos!((sim.u, sim.v, sim.w), sim.bcs);
                        _max_advective_rate(sim.u.data, sim.v.data, sim.w.data, sim.mesh))

# =========================================================================
# One parent step
# =========================================================================

const _NO_LEDGER = Float64[]          # the window's I0 of a run with no two-way scratch

"""
    step_nested!(run::NestedRun; dt = nothing) -> run

Advance `run` by one parent step:

 1. the parent step `Δt_p` is `adaptive_dt(parent)` (or `dt`; under
    `sync = :shared_min_dt` the smaller of the two domains' limits);
 2. the parent field set is prolongated onto the child at `t_p` (in a
    one-way run, reused from the end of the previous step);
 3. the parent takes its step. A rejected parent step (non-finite state,
    exception, `divergence_tol`, `max_velocity`) is rolled back and
    retried with `Δt_p/2` — **before** any child substep. Only a
    numerical failure ([`is_numerical_failure`](@ref
    Aeolus.TimeIntegration.is_numerical_failure)) is rejected; any other
    exception (a `MethodError`, a `BoundsError`, …) propagates at once;
 4. the parent field set is prolongated at `t_p + Δt_p`;
 5. the child subcycles to land exactly on `t_p + Δt_p`:
    `n = ⌈Δt_p / Δt_c^max⌉` substeps, re-evaluated before every substep as
    `n_rem = max(n_rem, ⌈R/Δt_c^max⌉)` with `Δt_c = R / n_rem` (`R` the
    remaining time; the last substep takes exactly `R`). The child's open
    faces hold the linearly interpolated, mass-corrected boundary data at
    every stage time of its core ([`stage_times`](@ref),
    [`boundary_time`](@ref), [`NestBoundaryData`](@ref)). A rejected
    child substep rolls back **only the child** (its `StateSnapshot`, its
    bound scalars and its SGS memory); the rejected piece is redone at
    half its size and the remaining time is re-divided;
 6. two-way: the feedback of the run's [`TwoWayFeedback`](@ref)
    ([`two_way_feedback!`](@ref)): anterpolation, interface flux
    correction, scalar refluxing from the flux registers booked at every
    stage of the parent step and the child substeps, re-projection.

The child clock is set to the parent's at the end (the discrepancy it
had is recorded in `run.stats.max_clock_error`).
"""
function step_nested!(run::NestedRun; dt = nothing)
    w = _begin_parent_step!(run; dt)
    # --- 5. child substeps --------------------------------------------------
    tw = time_ns()
    _subcycle_child!(run, w.t_child_end)
    run.stats.wall_child += (time_ns() - tw) / 1e9
    _end_parent_step!(run, w, true)
    return run
end

function _begin_parent_step!(run::NestedRun; dt = nothing)
    p = run.parent; c = run.child; nest = run.nest; st = run.stats
    rr = run.step_record                      # the window's record (#336)
    rr.t_wall0 = time_ns(); rr.t_sim0 = run.t
    run.retry_mark = st.parent_retries + st.child_retries
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

    m0 = _prepare_window!(run, t_p)

    # --- 3. parent step, with retries before any child substep -------------
    tw = time_ns()
    Δt = _advance_parent!(run, Δt)
    st.wall_parent += (time_ns() - tw) / 1e9
    return _open_window!(run, t_p, Float64(p.t), Δt, m0)
end

# Steps 1b–2 of a parent step at parent time t_p, before the parent moves:
# the two-way ledger's starting composite integrals (returned: the x
# momentum; the scalars in the run's scratch) and the boundary data at t_p.
# A nest hierarchy (#328) calls it for every link of a domain before that
# domain's step.
function _prepare_window!(run::NestedRun, t_p::Float64)
    p = run.parent; c = run.child; nest = run.nest; st = run.stats
    two_way = run.feedback === :two_way
    m0 = 0.0
    if two_way && run.record
        _composite_scalars!(run.work.I0, p, c, nest, run.s_bc)
        m0 = composite_integral(p, c, nest, :momentum_x)
    end

    # --- 2. boundary data at t_p -------------------------------------------
    tb = time_ns()
    if !two_way && run.reuse_boundary && run.new_valid_at == t_p
        run.s_old, run.s_new = run.s_new, run.s_old
    else
        _prolongate!(run, run.s_old)
    end
    run.new_valid_at = NaN
    st.wall_boundary += (time_ns() - tb) / 1e9
    return m0
end

# Step 4 of a parent step that took the parent from t_p to t_end (a step of
# Δt): the boundary data at t_end, the child's window, the two-way
# re-projection of the child. Returns the window of end_parent_step!.
function _open_window!(run::NestedRun, t_p::Float64, t_end::Float64, Δt::Float64, m0;
                       prolongated::Bool = false,
                       limiter::Symbol = step_record(run.parent).limiter)
    c = run.child; st = run.stats
    two_way = run.feedback === :two_way

    # --- 4. boundary data at t_p + Δt_p ------------------------------------
    # (already in s_new when the parent step was prefetched)
    if !prolongated
        tb = time_ns()
        _prolongate!(run, run.s_new)
        st.wall_boundary += (time_ns() - tb) / 1e9
    end
    run.new_valid_at = t_end
    # the window in the child's clock (the parent's, in a synchronised run,
    # where the child lands on the parent's clock bit for bit)
    run.t0 = Float64(c.t)
    run.Δt = t_end - t_p
    _set_window!(run)
    # Two-way: the feedback changed the parent after the previous window's
    # end data, so the child's boundary-normal faces just jumped to the new
    # B(t₀). Project the child onto them, or its first stage starts
    # divergent near the open faces (and SSP-RK3, which combines that start
    # state into the step's result, keeps a third of it: ∇·u grew to 5e-9
    # in 200 steps of the vortex harness).
    two_way && _reproject!(c)
    # the parent's clock and the limiter of its step travel with the window:
    # end_parent_step! must not read the parent, which a prefetched step of
    # the next window may be advancing by then (SC-c, #403)
    return (t_child_end = run.synced ? t_end : run.t0 + run.Δt, Δt = Δt,
            I0 = run.work === nothing ? _NO_LEDGER : run.work.I0, m0 = m0,
            t_parent = t_end, limiter = limiter)
end

# `reproject = false` (a nest hierarchy, #328): the two-way feedback leaves
# the parent's re-projection to the caller, which projects once after the
# feedback of every child of the domain.
function _end_parent_step!(run::NestedRun, w::NamedTuple, reproject::Bool)
    p = run.parent; c = run.child; nest = run.nest; st = run.stats
    t_end = w.t_child_end
    abs(Float64(c.t) - t_end) <= 1e-9 * max(run.Δt, floatmin(Float64)) ||
        error("end_parent_step!: the child is at t = $(c.t), not at the window end $t_end")
    st.max_clock_error = max(st.max_clock_error,
                             abs(Float64(c.t) - t_end) / max(abs(t_end), floatmin(Float64)))
    c.t = t_end

    # --- 6. feedback ---------------------------------------------------------
    if run.feedback === :two_way
        tw = time_ns()
        two_way_feedback!(run; reproject)
        if run.record
            wk = run.work
            _composite_scalars!(wk.I1, p, c, nest, run.s_bc)
            for n in eachindex(wk.keys)
                nest.ledger[wk.keys[n]] = abs(wk.I1[n] - w.I0[n]) / max(abs(w.I0[n]), 1e-300)
            end
            m1 = composite_integral(p, c, nest, :momentum_x)
            nest.ledger[:momentum_x] = abs(m1 - w.m0) / max(abs(w.m0), 1e-300)
        end
        run.new_valid_at = NaN                 # the parent changed after s_new
        st.wall_feedback += (time_ns() - tw) / 1e9
    end

    st.parent_steps += 1
    st.dt_parent_min = min(st.dt_parent_min, w.Δt)
    st.dt_parent_max = max(st.dt_parent_max, w.Δt)
    run.dt_prev_parent = w.Δt
    # the parent's clock and limiter as the window recorded them (a prefetched
    # step of the next window may be advancing the parent now, SC-c #403)
    run.t = haskey(w, :t_parent) ? w.t_parent : Float64(p.t)
    # the run's step record and hook (#336): the window is the run's step,
    # limited as the parent step was
    rr = run.step_record
    rr.pick_t = rr.t_sim0; rr.pick_dt = Float64(w.Δt); rr.pick_rate = NaN; rr.pick_diff = NaN
    rr.pick_limiter = haskey(w, :limiter) ? w.limiter : step_record(p).limiter
    _step_end!(rr, run, w.Δt, true; retries = st.parent_retries + st.child_retries - run.retry_mark)
    return run
end

# The flux-register hook of a domain (nothing without refluxing), and the
# begin / commit of one attempted step.
_hook(run::NestedRun, which::Symbol) =
    run.regs === nothing ? nothing : _RegisterHook(run.regs, run.nest, which)
_begin!(run::NestedRun, which) = run.regs === nothing || _begin_attempt!(run.regs, which)
_commit!(run::NestedRun, which) = run.regs === nothing || _commit_attempt!(run.regs, which)

# The parent step is one step of the parent's own record (#336): its
# retries are part of it, a rejected attempt is not a step. `sub` (a nest
# hierarchy's domain, #328, or nothing): the root of a hierarchy is stepped
# through its first link, `sub` adding the register hooks and bookings of
# its other links and its own rollback (all bound scalars of all its links).
#
# `ctr` counts the rejected attempts (`ctr.parent_retries`): the run's
# statistics, or the counters of a prefetched step, merged into them when
# the run takes it (the run's statistics are the main task's, SC-c #403).
function _advance_parent!(run::NestedRun, Δt::Float64, sub = nothing, ctr = run.stats)
    p = run.parent; rec = step_record(p)
    d0 = rec.depth; r0 = ctr.parent_retries
    outer = _step_begin!(rec, p)
    Δt = try
        _advance_parent_tries!(run, Δt, sub, ctr)
    catch
        rec.depth = d0; rethrow()
    end
    _step_end!(rec, p, Δt, outer; retries = ctr.parent_retries - r0)
    return Δt
end

function _advance_parent_tries!(run::NestedRun, Δt::Float64, sub, ctr)
    p = run.parent; adv = run.advance_parent
    hook = sub === nothing ? _hook(run, :parent) : sub.hook
    if !run.validate
        _begin!(run, :parent); _begin_sub!(sub)
        adv(p, Δt; body_force! = hook)
        _commit!(run, :parent); _commit_sub!(sub)
        return Δt
    end
    fields = sub === nothing ? _bound_fields(run, :parent) : sub.fields
    rb = sub === nothing ? _save!(run, p, adv, :parent) : _save_rb!(sub.rb, p, fields)
    dt_min = run.cfl_parent === nothing ? 0.0 : Float64(run.cfl_parent.dt_min)
    tries = 0
    while true
        _begin!(run, :parent); _begin_sub!(sub)
        if _try_advance!(adv, p, Δt, hook) && _accept_fields(run, p, :parent, fields)
            _commit!(run, :parent); _commit_sub!(sub)
            return Δt
        end
        _restore_rb!(p, rb, fields)
        tries += 1
        ctr.parent_retries += 1
        Δt_new = max(Δt / 2, dt_min)
        (tries > run.max_retries || Δt_new == Δt) &&
            error("step_nested!: the parent step failed $tries times (last dt = $Δt); " *
                  "the parent has been restored to t = $(p.t), the child was not advanced.")
        Δt = Δt_new
    end
end

# Each accepted child substep is one step of the child's own record (#336),
# with the retries of its piece. `sub` (#328): the child is itself the
# coarse domain of further links of a nest hierarchy. Each substep then also
# prepares their boundary data (before the attempt), books their registers,
# rolls back all its bound scalars, and after it is accepted advances its
# own children to its end (Berger & Oliger 1984: the recursion of
# _sub_after_step!).
function _subcycle_child!(run::NestedRun, t_end::Float64, sub = nothing)
    rec = step_record(run.child)
    d0 = rec.depth
    try
        return _subcycle_child_pieces!(run, t_end, rec, sub)
    catch
        rec.depth = d0; rethrow()
    end
end

function _subcycle_child_pieces!(run::NestedRun, t_end::Float64, rec::StepRecord, sub)
    c = run.child; adv = run.advance_child; st = run.stats
    hook = sub === nothing ? _hook(run, :child) : sub.hook
    fields = sub === nothing ? _bound_fields(run, :child) : sub.fields
    pp = run.sponge === nothing ? nothing : _SpongeHook(run)
    adaptive = run.sync === :subcycle && run.cfl_child !== nothing
    R = t_end - c.t
    n_rem = run.sync === :shared_min_dt ? 1 :
            adaptive ? _nsub(R, _dt_limit(c, adv, run.cfl_child, run.dt_prev_child)) :
            run.substeps
    first_piece = true
    n_taken = 0
    tries = 0
    outer = false
    h_retry = 0.0                       # > 0: redo a rejected piece at this size
    while n_rem > 0
        R = t_end - c.t
        if adaptive && !first_piece
            n_rem = max(n_rem, _nsub(R, _dt_limit(c, adv, run.cfl_child, run.dt_prev_child)))
        end
        first_piece = false
        h = h_retry > 0 ? h_retry : n_rem == 1 ? R : R / n_rem
        cfl = h * _advective_rate(c)

        sub === nothing || _sub_prepare!(sub, Float64(c.t))
        rb = !run.validate ? nothing : sub === nothing ? _save!(run, c, adv, :child) :
                                                         _save_rb!(sub.rb, c, fields)
        tries == 0 && (outer = _step_begin!(rec, c))   # the piece's first attempt opens its step
        run.sub_t0 = Float64(c.t); run.sub_h = h
        _begin!(run, :child); _begin_sub!(sub)
        ok = if run.validate
            _try_advance!(adv, c, h, hook, pp) && _accept_fields(run, c, :child, fields)
        else
            adv(c, h; body_force! = hook, post_predictor! = pp); true
        end
        ok && (_commit!(run, :child); _commit_sub!(sub))
        if !ok
            _restore_rb!(c, rb, fields)
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
        piece_retries = tries
        tries = 0
        if run.sponge !== nothing               # the scalars; the velocity was relaxed
            apply_nest_sponge!(c, _sponge_target!(run), run.sponge; dt = h, velocity = false)
        end
        sub === nothing || _sub_after_step!(sub, h)
        n_taken += 1
        n_rem -= 1
        st.child_substeps += 1
        st.dt_child_min = min(st.dt_child_min, h)
        st.dt_child_max = max(st.dt_child_max, h)
        st.max_child_cfl = max(st.max_child_cfl, cfl)
        run.step_record.cfl_adv = _nmax(run.step_record.cfl_adv, cfl)
        _step_end!(rec, c, h, outer; retries = piece_retries)
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

"""
    two_way_feedback!(run::NestedRun) -> run

Apply the child → parent feedback of a two-way run now (what
[`step_nested!`](@ref)`(run)` does once per parent step, after the child
substeps), as configured by the run's [`TwoWayFeedback`](@ref):

 1. [`anterpolate!`](@ref) the field set onto the cells of
    [`feedback_mask`](@ref)`(run)`, relaxed with `α = min(1, Δt_p/τ)`
    (`Δt_p` the last parent step) and solid-aware with the run's `solid`;
 2. correct the parent's velocity on the interface planes of the child's
    open faces to the child's flux (`flux_correction`);
 3. reflux `θ`, `qv` and the declared tracers from the flux registers
    booked since the last feedback, and clear them. What is refluxed is
    the flux of `ϕ − ϕ_ref`, `ϕ_ref` the parent's mean of `ϕ` over the
    covered cells (one constant per scalar): the two grids are each
    volume-conserving, but their time-integrated volume fluxes through a
    face patch differ (the child's boundary velocity is interpolated in
    time, the parent's is its stage velocity), and the flux of `ϕ` itself
    would book `ϕ ×` that volume mismatch — for `θ ≈ 300 K` a spurious
    heating of the cells outside the nest by up to kelvins per step. The
    volume mismatches of the patches sum to zero over the closed interface
    (both velocity fields are divergence-free), so the total correction,
    and with it the exact conservation of the composite `∫ϕ dV`, is the
    same for every `ϕ_ref`;
 4. re-project the parent.

The interface mass-flux imbalance is written to the ledger before the
flux correction (`:mass_before_correction`), after it (`:mass`, round-off)
and after the re-projection (`:mass_after_projection`: the projection
moves the parent's interface faces too, by the divergence that the
anterpolation next to the buffer zone and the flux correction left in the
parent; the next step's child boundary data take the projected parent).
"""
function two_way_feedback!(run::NestedRun; reproject::Bool = true)
    fb = run.twoway
    fb === nothing && error("two_way_feedback!: the run is one-way")
    p = run.parent; c = run.child; nest = run.nest
    meas = run.s_bc.measure
    α = fb.τ === nothing ? 1.0 : min(1.0, run.Δt / fb.τ)
    # the reference value of each refluxed scalar (volume-consistent refluxing)
    wk = run.work
    refs = wk.refs
    if run.regs !== nothing
        resize!(refs, length(run.regs.pfield))
        for n in eachindex(run.regs.pfield)
            refs[n] = _covered_mean(run.regs.pfield[n], nest)
        end
    end
    _feedback_mask!(wk.mask, nest, fb, c)
    run.sponge_τ = NaN                         # the sponge target is re-blended after this
    cv = run.conv
    cv === nothing || _scale_parent!(p, cv)    # a mixed pair: the parent's velocity in the child's density
    anterpolate!(p, c, nest, run.s_bc; mask = wk.mask, α, solid = run.fluid,
                 scratch = wk.anterp)
    sides = nest_open_sides(c)
    # the child's normal faces without the synthetic part of their boundary
    # data (inflow turbulence, #418): the parent's interface faces get, and
    # the imbalance measures, the flux the boundary data carry
    syn = _strip_synthetic_normal!(c, sides)
    e_before = _interface_imbalance(p, c, nest, meas)
    if run.flux_correction
        _correct_interface_flux!(p.u, c.u, nest, 1, meas, sides)
        _correct_interface_flux!(p.v, c.v, nest, 2, meas, sides)
        _correct_interface_flux!(p.w, c.w, nest, 3, meas, sides)
    end
    if run.regs !== nothing
        r = run.regs
        per = nest_periodic_axes(p)
        for n in eachindex(r.names)
            apply_flux_correction!(r.pfield[n], _register_mismatch(r, n, nest, refs[n]),
                                   nest; measure = meas, periodic = per)
        end
        _clear_registers!(r)
    end
    e_after = _interface_imbalance(p, c, nest, meas)
    cv === nothing || _unscale_parent!(p, cv)  # the mass fluxes back in the parent's own density
    proj = fb.reproject && reproject
    proj && _reproject!(p)
    if run.record
        nest.ledger[:mass] = e_after
        nest.ledger[:mass_before_correction] = e_before
        nest.ledger[:mass_after_projection] =
            !proj ? e_after : cv === nothing ? _interface_imbalance(p, c, nest, meas) :
            (_scale_parent!(p, cv); e = _interface_imbalance(p, c, nest, meas); _restore_parent!(p, cv); e)
    end
    _restore_synthetic_normal!(syn)
    return run
end

# Subtract from the child's normal velocity on every open face the
# synthetic part its boundary data declare (`synthetic_normal_velocity`,
# #418) and return the saved face planes (`nothing` when no face has one,
# so a run without synthetic inflow is untouched). The interface flux
# correction would otherwise copy the synthetic eddies' flux into the
# parent's interface faces; the next window's boundary data would carry
# it back to the child, where the generator adds its next perturbation on
# top: a sum over parent steps (measured on the 2 h CBL nest: child KE
# ×9.7, parent ×2.1, overlap θ +0.24 K; docs/src/theory/transition.md).
function _strip_synthetic_normal!(c, sides::NTuple{6,Bool})
    saved = nothing
    m = c.mesh
    H = (m.Hx, m.Hy, m.Hz); N = (m.Nx, m.Ny, m.Nz)
    for s in 1:6
        sides[s] || continue
        side = _SIDE_NAMES[s]
        P = synthetic_normal_velocity(getproperty(c.bcs, side).state.data, side, c.t)
        P === nothing && continue
        ax = _side_axis(s)
        t1, t2 = _tang(ax)
        size(P) == (N[t1], N[t2]) || throw(DimensionMismatch(
            "synthetic_normal_velocity: the :$side plane is $(size(P)), the face $((N[t1], N[t2]))"))
        q = H[ax] + (_side_high(s) ? N[ax] + 1 : 1)
        v = view((c.u, c.v, c.w)[ax].data, ntuple(d -> d == ax ? (q:q) : (H[d]+1:H[d]+N[d]), 3)...)
        saved === nothing && (saved = Tuple{typeof(v),Array{eltype(v),3}}[])
        push!(saved, (v, copy(v)))
        v .-= reshape(P, size(v))
    end
    return saved
end

# Put the saved planes back, bit for bit.
_restore_synthetic_normal!(::Nothing) = nothing
function _restore_synthetic_normal!(saved)
    for (v, a) in saved
        copyto!(v, a)
    end
    return nothing
end

# The parent's mean of a cell-centred field over the covered cells.
function _covered_mean(f::Field, nest::NestedGrid)
    m = nest.parent_mesh
    cx, cy, cz = nest_covered_range(nest)
    # canonical plane sums over the covered storage range, then ascending k
    # (thread-invariant, KA-b #296)
    r = (m.Hx .+ cx, m.Hy .+ cy, m.Hz .+ cz)
    p = plane_partials!(zeros(length(cz)), identity, map(UnitRange{Int}, r), f.data)
    return ordered_sum(p) / (length(cx) * length(cy) * length(cz))
end

feedback_mask(run::NestedRun) = run.twoway === nothing ?
    error("feedback_mask: the run is one-way") : feedback_mask(run.nest, run.twoway, run.child)

# =========================================================================
# The historical fixed-dt entry point
# =========================================================================

"""
    step_nested!(parent_sim, child_sim, nest, dt; substeps=nothing, order=3,
                 flux_correction=true, sponge=nothing, solid=nothing,
                 scalars=(), state=nothing, record=true,
                 feedback=TwoWayFeedback())

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
The child's open faces ([`nest_bcs`](@ref)) hold the parent state
linearly interpolated in time to the child's stage times; after the child substeps the child field set is
fed back onto the parent with `feedback` (a [`TwoWayFeedback`](@ref):
anterpolation outside the buffer zone, interface flux correction,
scalar refluxing, re-projection). Conservation diagnostics for `:mass`, `:momentum_x` and every
scalar of the set are recorded in `nest.ledger`; read them with
[`nest_conservation_error`](@ref). Returns `(parent_sim, child_sim)`.

`state` may be a pre-allocated `NTuple{3,NestInterpState}` (built with
`nest_interp_state(nest, parent_sim, child_sim; scalars)`) to avoid
re-allocating the boundary buffers every step; it then defines the field
set and `scalars` is ignored.

The parent may be z-stretched (a `BoussinesqSimulation` on a
`StretchedGrid`). Both sims may instead be
[`MassWeightedSimulation`](@ref)s sharing one `ReferenceState` profile:
every transfer then uses the reference-mass measure
([`nest_measure`](@ref)) and the ledger records `∫ W φ dV`. `solid`
(the child's IBM mask) makes the restriction solid-aware.
"""
function step_nested!(psim::_NestSim, csim::_NestSim, nest::NestedGrid, dt::Real;
                      substeps::Union{Nothing,Integer}=nothing,
                      order::Int=3,
                      flux_correction::Bool=true,
                      sponge::Union{Nothing,NestSponge}=nothing,
                      solid=nothing,
                      scalars=(),
                      state=nothing,
                      record::Bool=true,
                      feedback::TwoWayFeedback=TwoWayFeedback())
    run = NestedRun(psim, csim, nest; cfl_parent=nothing, cfl_child=nothing,
                    substeps, feedback, order, flux_correction, sponge,
                    solid, scalars, state, record, validate=false,
                    check_clocks=false)
    step_nested!(run; dt)
    return (psim, csim)
end
