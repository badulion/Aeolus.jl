# The one nest engine (NEST-p2, #482): one window of a ModelNest, the
# root's step with every descendant subcycling to land on its end.
#
#   * the pipeline of a nested domain is split around the nest: stages
#     1–11 (the drivers, the stages before the core, the core with stages
#     6–8 and the nest's register hooks, :surface_update, :column_tendency),
#     then the children's windows and the two-way feedback (stage 12,
#     :nest_feedback), then stage 13 (:budgets, :checks, :output);
#   * a child takes n equal substeps per step of its parent: the largest
#     refinement for fixed steps, the smallest whole number that keeps it
#     within its own stability limit for adaptive steps (chosen at the start
#     of every window of the child);
#   * a failed step (non-finite state, a numerical failure) retries the
#     window: the whole nest is restored from the state saved at the window
#     start (Aeolus.IO's structural walk: every core, component, budget,
#     register and statistic) and redone, with half the root step when the
#     root failed, with twice the substeps of the failing child otherwise.
#
# The operators, the boundary data, the registers and the feedback are
# those of Aeolus.Nesting (the links of the NestHierarchy); only the order
# in which they run is the engine's. With fixed steps it is the order of
# Nesting.step_nested!(::NestHierarchy), so the prognostic state is that
# path's bit for bit (test/model_nesting.jl).

using ..Nesting: _sub_prepare!, _open_window!, _end_parent_step!, _begin!, _commit!, _nsub,
                 _advective_rate, _SpongeHook, _reproject!, apply_nest_sponge!, _sponge_target!
using ..Simulations: step_record, _step_begin!, _step_end!, _step_abort!, _nmax
using ..TimeIntegration: is_numerical_failure, _all_finite
using ..Backends: interior_all_finite
using ..RunLog: current_domain, with_domain

"The model domain `i` of a nest (the domain its hierarchy's advance steps)."
_ndom(mn::ModelNest, i::Int) = mn.hierarchy.advance[i].core.d

"""
    NestStepFailure(domain)

Thrown inside a window when a step of the nest's domain number `domain`
failed (a non-finite state, or a numerical failure of its step): the
engine ([`nest_window!`](@ref)) restores the window start and redoes it.
"""
struct NestStepFailure <: Exception
    domain::Int
end
Base.showerror(io::IO, e::NestStepFailure) = print(io, "NestStepFailure: a step of nest domain $(e.domain) failed")

# --- one domain's pipeline, split around the nest -----------------------------

const _STAGES_11 = (:surface_update, :column_tendency)

# stages 1–11 of a step of `d` over `dt`, with the nest's hooks
function _nest_core!(d::Domain, sim, dt, bf, pp)
    sync_drivers!(d)
    t0 = clock(d)
    _budgets_begin!(d.budgets, d)
    _run_stages!(d, PRE_CORE, dt, t0, t0 + dt)
    _dynamics_open!(d.budgets, d)
    _core_step!(sim, d, dt, bf, pp)
    _dynamics_close!(d.budgets, d)
    _run_stages!(d, _STAGES_11, dt, t0, clock(d))
    return nothing
end

# stages 12–13 (the budgets close after :budgets, before :checks and :output)
function _nest_finish!(d::Domain, dt, t0, t1)
    _run_stages!(d, (:nest_feedback,), dt, t0, t1)
    if d.budgets === nothing
        _run_stages!(d, (:budgets,), dt, t0, t1)
    else
        _run_stages!(d, (:budgets,), dt, t0, t1)
        _budgets_end!(d.budgets, d, dt)
    end
    _run_stages!(d, _POST_BUDGETS, dt, t0, t1)
    return nothing
end

# One attempt at stages 1–11; false when it failed (a numerical failure, or
# a non-finite state of the core or of a field the nest binds).
function _nest_attempt!(mn::ModelNest, d::Domain, sim, dt, bf, pp, fields)
    if !mn.validate
        _nest_core!(d, sim, dt, bf, pp)
        return true
    end
    try
        _nest_core!(d, sim, dt, bf, pp)
    catch err
        is_numerical_failure(err, sim) || rethrow()
        return false
    end
    return _nest_state_ok(sim, fields)
end

function _nest_state_ok(sim, fields)
    _all_finite(sim) || return false
    for f in fields
        interior_all_finite(f) || return false
    end
    return true
end

# the bookings of a step of domain i: its uplink as the child, its links as the parent
function _nest_begin!(h, i::Int)
    hd = h.domains[i]
    hd.up == 0 || _begin!(h.links[hd.up], :child)
    for l in hd.links
        _begin!(h.links[l], :parent)
    end
    return nothing
end
function _nest_commit!(h, i::Int)
    hd = h.domains[i]
    hd.up == 0 || _commit!(h.links[hd.up], :child)
    for l in hd.links
        _commit!(h.links[l], :parent)
    end
    return nothing
end

"""
    nest_domain_limit(mn::ModelNest, i) -> dt

The stability limit of the nest's domain `i` now: `adaptive_dt` of its
core with its `cfl`, its SGS closure, the components' limits
([`component_limits`](@ref)) and the growth limit from its last step.
"""
function nest_domain_limit(mn::ModelNest, i::Int)
    d = _ndom(mn, i)
    return Float64(adaptive_dt(d.state, d.cfl; dt_prev = mn.dt_prev[i], sgs = sgs_closure(d),
                               sgs_stress = sgs_stress(d), limits = component_limits(d)))
end

# --- the recursion ----------------------------------------------------------------

# One step of domain i over Δt with its subtree: stages 1–11, the windows of
# its children (each subcycling to land on the step's end), the feedback,
# stages 12–13.
function _nest_domain_step!(mn::ModelNest, i::Int, Δt::Float64)
    h = mn.hierarchy
    hd = h.domains[i]
    _sub_prepare!(hd, Float64(_ndom(mn, i).state.t))     # every child's boundary data at t0
    t0, t1 = _nest_own_step!(mn, i, Δt)
    isempty(hd.links) || _nest_children!(mn, i, t0, Δt)
    _nest_finish!(_ndom(mn, i), Δt, t0, t1)
    return nothing
end

# Stages 1–11 of a step of domain i over Δt (its children's boundary data at
# the step start prepared by the caller); returns its start and end time.
function _nest_own_step!(mn::ModelNest, i::Int, Δt::Float64)
    h = mn.hierarchy
    hd = h.domains[i]
    d = _ndom(mn, i)
    sim = d.state
    t0 = Float64(sim.t)
    up = hd.up == 0 ? nothing : h.links[hd.up]
    pp = up === nothing || up.sponge === nothing ? nothing : _SpongeHook(up)
    if up !== nothing
        up.sub_t0 = t0; up.sub_h = Δt
    end
    rec = step_record(sim)
    outer = _step_begin!(rec, sim)
    ok = try
        _nest_begin!(h, i)
        _nest_attempt!(mn, d, sim, Δt, hd.hook, pp, hd.fields)
    catch
        _step_abort!(rec); rethrow()
    end
    ok || (_step_abort!(rec); throw(NestStepFailure(i)))
    _nest_commit!(h, i)
    _step_end!(rec, sim, Δt, outer)
    t1 = Float64(sim.t)
    # the scalars of a nest sponge (its velocity part ran in the core)
    up === nothing || up.sponge === nothing ||
        apply_nest_sponge!(sim, _sponge_target!(up), up.sponge; dt = Δt, velocity = false)
    mn.dt_prev[i] = Δt
    return t0, t1
end

# Run `f(k)` for k in ks on tasks of their own (each logging as `dom`), wait
# for all, and rethrow the first failure in the order of ks (a NestStepFailure
# unwrapped, so the window retry sees it).
function _nest_parallel(f, ks, dom::Symbol)
    tasks = [Threads.@spawn with_domain(() -> f(k), dom) for k in ks]
    err = nothing
    for t in tasks
        try
            wait(t)
        catch e
            err === nothing && (err = e isa TaskFailedException ? t.exception : e)
        end
    end
    err === nothing || throw(err)
    return nothing
end

# The windows of domain i's children over its step [t0, t0 + Δt], and their
# feedback (stage 12).
function _nest_children!(mn::ModelNest, i::Int, t0::Float64, Δt::Float64)
    h = mn.hierarchy
    hd = h.domains[i]
    sim = _ndom(mn, i).state
    t_end = Float64(sim.t)
    ws = _nest_open_windows!(mn, i, t0, t_end, Δt)
    _nest_windows!(mn, i, ws)
    _nest_feedback!(mn, i, ws)
    return nothing
end

# Open the windows of domain i's children over its step [t0, t1] (the
# boundary data at t1), and fold the parent's statistics into the children's
# inflow turbulence (the parent is at rest here, #268).
function _nest_open_windows!(mn::ModelNest, i::Int, t0::Float64, t1::Float64, Δt::Float64)
    h = mn.hierarchy
    hd = h.domains[i]
    ws = [_open_window!(h.links[l], t0, t1, Δt, hd.m0[n]) for (n, l) in enumerate(hd.links)]
    for l in hd.links
        it = mn.inflow[l]
        it === nothing || fold_inflow_statistics!(it; interval = _nest_inflow_interval(mn, l))
    end
    return ws
end
_nest_inflow_interval(mn::ModelNest, l::Int) =
    mn.links[findfirst(k -> mn.hierarchy.domains[k.child].up == l, mn.links)].inflow.statistics_interval

# The children's windows of domain i (opened, `ws`): the siblings in
# parallel when the nest is concurrent (they read only their own boundary
# data, write their own domains and links, and feed back afterwards in link
# order, so the result is that of stepping them one after the other, bit
# for bit), else one after the other.
function _nest_windows!(mn::ModelNest, i::Int, ws)
    h = mn.hierarchy
    links = h.domains[i].links
    function one(n)
        L = h.links[links[n]]
        tw = time_ns()
        _nest_child_window!(mn, links[n], ws[n].t_child_end)
        L.stats.wall_child += (time_ns() - tw) / 1e9
        return nothing
    end
    if mn.concurrent && length(links) > 1
        _nest_parallel(one, eachindex(links), current_domain())
    else
        foreach(one, eachindex(links))
    end
    return nothing
end

# stage 12: the two-way feedback of domain i's children, the parent
# re-projected once after all of it
function _nest_feedback!(mn::ModelNest, i::Int, ws)
    h = mn.hierarchy
    hd = h.domains[i]
    b = _ndom(mn, i).budgets
    twoway = any(l -> h.links[l].feedback === :two_way, hd.links)
    # the parent's budgets book the feedback as its nest exchange
    ctx = b !== nothing && twoway
    ctx && foreach(q -> budget_open!(b.budgets.measured[q], NEST_EXCHANGE..., b.state[q]), b.quantities)
    reproject = false
    for (n, l) in enumerate(hd.links)
        L = h.links[l]
        _end_parent_step!(L, ws[n], false)
        L.twoway !== nothing && L.twoway.reproject && (reproject = true)
    end
    if twoway
        reproject && _reproject!(_ndom(mn, i).state)
        for l in hd.links                       # the parent changed after every link's end data
            h.links[l].new_valid_at = NaN
        end
    end
    ctx && foreach(q -> budget_close!(b.budgets.measured[q], b.state[q]), Iterators.reverse(b.quantities))
    return nothing
end

"""
    nest_substeps(mn::ModelNest, l, R) -> n

The substeps the child of the nest's link `l` takes over a window of `R`
seconds: the link's fixed `substeps` (the largest refinement) for fixed
steps; adaptive, the smallest whole `n` with `R/n` within the child's
stability limit ([`nest_domain_limit`](@ref)). Never below the floor a
retried window puts on the link.
"""
function nest_substeps(mn::ModelNest, l::Int, R::Float64)
    L = mn.hierarchy.links[l]
    n = mn.adaptive ? _nsub(R, nest_domain_limit(mn, mn.hierarchy.fine[l])) : L.substeps
    return max(n, mn.nfloor[l])
end

# The child of link l subcycles from its time to t_end: n equal substeps
# (the last one lands on t_end exactly).
function _nest_child_window!(mn::ModelNest, l::Int, t_end::Float64)
    h = mn.hierarchy
    L = h.links[l]
    c = L.child
    ci = h.fine[l]
    st = L.stats
    n = nest_substeps(mn, l, t_end - Float64(c.t))
    mn.nused[l] = n
    if mn.concurrent && n > 1 && nest_prefetches(mn, ci)
        _nest_subcycle_ahead!(mn, l, n, t_end)
    else
        n_rem = n
        while n_rem > 0
            R = t_end - c.t
            hs = n_rem == 1 ? R : R / n_rem
            cfl = hs * _advective_rate(c)
            _nest_domain_step!(mn, ci, hs)
            n_rem -= 1
            _nest_substep_stats!(L, hs, cfl)
        end
    end
    st.max_clock_error = max(st.max_clock_error,
                             abs(Float64(c.t) - t_end) / max(abs(t_end), floatmin(Float64)))
    c.t = t_end
    st.n_last = n
    st.n_min = min(st.n_min, n)
    st.n_max = max(st.n_max, n)
    return nothing
end

function _nest_substep_stats!(L, hs, cfl)
    st = L.stats
    st.child_substeps += 1
    st.dt_child_min = min(st.dt_child_min, hs)
    st.dt_child_max = max(st.dt_child_max, hs)
    st.max_child_cfl = max(st.max_child_cfl, cfl)
    L.step_record.cfl_adv = _nmax(L.step_record.cfl_adv, cfl)
    L.dt_prev_child = hs
    return nothing
end

"""
    nest_prefetches(mn::ModelNest, i) -> Bool

Whether the nest's domain `i` steps ahead of its children (SC-c, #403;
the design's decision of 2026-10-12): a domain below the root whose
children are all one-way (and take their boundary data at a window's
start from the end of the one before, `reuse_boundary`) needs nothing
from them, so while they subcycle through its substep `s` it takes its
substep `s + 1` on a task of its own ([`nest_window!`](@ref)). The root
is not prefetched: its next step is the model's next window, and the run
reads every domain at the end of each window (output, checks, the
monitor, checkpoints).
"""
nest_prefetches(mn::ModelNest, i::Int) =
    (h = mn.hierarchy; hd = h.domains[i];
     i > 1 && !isempty(hd.links) &&
     all(l -> h.links[l].feedback === :one_way && h.links[l].reuse_boundary && h.links[l].sponge === nothing, hd.links))

# The n substeps of domain ci (the child of link l) to t_end, each overlapped
# with its children's window of the substep before: substep s + 1 of ci runs
# on a task while its one-way children subcycle through substep s. The
# operations and the data they read are the sequential ones (the children
# read only the boundary data prolongated before the task starts; the
# start of their next window is the end of this one, reuse_boundary), so
# the result is bit for bit that of the sequential steps.
function _nest_subcycle_ahead!(mn::ModelNest, l::Int, n::Int, t_end::Float64)
    h = mn.hierarchy
    L = h.links[l]
    ci = h.fine[l]
    c = L.child
    hd = h.domains[ci]
    d = _ndom(mn, ci)
    dom = current_domain()
    step_size(n_rem) = (R = t_end - c.t; n_rem == 1 ? R : R / n_rem)
    hs = step_size(n)
    cfl = hs * _advective_rate(c)
    _sub_prepare!(hd, Float64(c.t))
    t0, t1 = _nest_own_step!(mn, ci, hs)
    n_rem = n
    while true
        ws = _nest_open_windows!(mn, ci, t0, t1, hs)
        _nest_finish!(d, hs, t0, t1)                  # one-way children: no feedback into ci
        _nest_substep_stats!(L, hs, cfl)
        n_rem -= 1
        if n_rem == 0
            _nest_windows!(mn, ci, ws)
            _nest_feedback!(mn, ci, ws)
            break
        end
        hn = step_size(n_rem)
        cfln = hn * _advective_rate(c)
        next = Ref((0.0, 0.0))
        task = Threads.@spawn with_domain(() -> (next[] = _nest_own_step!(mn, ci, hn); nothing), dom)
        err = nothing
        try
            _nest_windows!(mn, ci, ws)
            _nest_feedback!(mn, ci, ws)
        catch e
            err = e
        end
        try
            wait(task)
        catch e
            err === nothing && (err = e isa TaskFailedException ? task.exception : e)
        end
        err === nothing || throw(err)
        # the children's boundary data at the start of the next substep: the
        # end of this one (one-way, reuse_boundary: nothing reads ci now)
        _sub_prepare!(hd, next[][1])
        t0, t1 = next[]
        hs = hn; cfl = cfln
    end
    return nothing
end

# --- the window, with its retries --------------------------------------------------

"""
    nest_window!(mn::ModelNest, Δt; dt_min = 0) -> (Δt_taken, retries)

One window of the nest: the root's step of `Δt` with every descendant
subcycling to land on its end (see [`ModelNest`](@ref)). With
`mn.validate`, the state at the window start is saved first, and a failed
step (a non-finite state of a core or a nest-bound field, or a
[numerical failure](@ref Aeolus.TimeIntegration.is_numerical_failure))
retries the window from it (the design's decision of 2026-10-12):

  * the root failed — the window is redone with half the root step (not
    below `dt_min`), every child choosing its substeps anew;
  * a child failed — the window is redone with the same root step and
    twice that child's substeps (the floor holds for the rest of the
    window). The parent and the other domains repeat their steps bit for
    bit: the restore puts back everything (cores, components, budgets,
    registers, statistics), so a child failure costs the parent a repeated
    step but the result is that of the child alone redoing its window.

After `mn.max_retries` retries the window is restored and an error raised.
"""
function nest_window!(mn::ModelNest, Δt::Float64; dt_min::Float64 = 0.0)
    h = mn.hierarchy
    fill!(mn.nfloor, 0)
    mn.last_retries = 0
    if !mn.validate
        _nest_domain_step!(mn, 1, Δt)
        _nest_window_done!(mn)
        return Δt, 0
    end
    mn.snapshot = mn.snapshot === nothing ? structural_state(h) : _resnapshot!(mn.snapshot, h)
    dtp = copy(mn.dt_prev)
    tries = 0
    while true
        try
            _nest_domain_step!(mn, 1, Δt)
            _nest_state_ok(_ndom(mn, 1).state, h.domains[1].fields) || throw(NestStepFailure(1))
            break
        catch err
            err isa NestStepFailure || rethrow()
            tries += 1
            restore_structural_state!(h, mn.snapshot)
            copyto!(mn.dt_prev, dtp)
            if err.domain == 1
                Δt_new = max(Δt / 2, dt_min)
                (tries > mn.max_retries || Δt_new == Δt) &&
                    error("nest_window!: the root step failed $tries times (last dt = $Δt); the nest has " *
                          "been restored to the window start t = $(h.domains[1].sim.t)")
                Δt = Δt_new
                fill!(mn.nfloor, 0)
                h.links[h.domains[1].links[1]].stats.parent_retries += 1
            else
                l = h.domains[err.domain].up
                tries > mn.max_retries &&
                    error("nest_window!: a step of domain $(h.domains[err.domain].id) failed $tries times " *
                          "(last with $(mn.nused[l]) substeps per step of its parent); the nest has been " *
                          "restored to the window start t = $(h.domains[1].sim.t)")
                mn.nfloor[l] = 2 * mn.nused[l]
                h.links[l].stats.child_retries += 1
            end
        end
    end
    mn.last_retries = tries
    mn.retries += tries
    _nest_window_done!(mn)
    return Δt, tries
end

function _nest_window_done!(mn::ModelNest)
    h = mn.hierarchy
    h.dt_prev_root = mn.dt_prev[1]
    h.root_steps += 1
    h.t = Float64(h.domains[1].sim.t)
    return nothing
end

# The window-start state into the arrays of the previous one (the graph does
# not change between windows; a fresh record when it did).
_resnapshot!(s, h) = refresh_structural_state!(s, h)

"""
    step!(m::Model{<:Tuple,<:ModelNest}, dt) -> m

One window of a nest with fixed steps: the root steps `dt`, every child
`n = max(r)` substeps per step of its parent ([`nest_window!`](@ref)).
"""
function step!(m::Model{<:Tuple,<:ModelNest}, dt::Real; checked::Bool = true)
    m.nest.adaptive && throw(ArgumentError("step!: the nest steps adaptively (step_adaptive!(model))"))
    nest_window!(m.nest, Float64(dt))
    return m
end

"""
    step_adaptive!(m::Model{<:Tuple,<:ModelNest}; dt_prev = nothing, t_end = nothing) -> (state, dt, retries)

One adaptive window of a nest: the root's step is its own stability
limit ([`nest_domain_limit`](@ref), from `dt_prev` or its last step),
shortened to land on `t_end`; each child takes the smallest whole number
of equal substeps that keeps it stable, chosen at every window; a failed
step retries the window ([`nest_window!`](@ref)). Returns the root's core,
the root step taken and the retries.
"""
function step_adaptive!(m::Model{<:Tuple,<:ModelNest}; dt_prev = nothing, t_end = nothing)
    mn = m.nest
    mn.adaptive || throw(ArgumentError("step_adaptive!: the nest takes fixed steps (step!(model, dt))"))
    dt_prev === nothing || (mn.dt_prev[1] = Float64(dt_prev))
    root = m.domains[1]
    foreach(sync_drivers!, m.domains)
    dt = nest_domain_limit(mn, 1)
    t_end === nothing || (dt = min(dt, Float64(t_end) - Float64(clock(root))))
    dt > 0 || throw(ArgumentError("step_adaptive!: the nest is at its end t = $(clock(root))"))
    Δt, tries = nest_window!(mn, dt; dt_min = Float64(root.cfl.dt_min))
    return (root.state, Δt, tries)
end
