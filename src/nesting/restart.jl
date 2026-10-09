# Nesting: checkpoint / restart of a nested run (N-berlin-restart, #269).
#
# A NestedRun carries, besides its two simulations, the state of the nest:
# the parent field set prolongated at both ends of the current window
# (s_old, s_new) and the window itself, the clocks and the step sizes the
# adaptive controllers continue from, the flux registers of a refluxing
# two-way run, the statistics, the child's boundary data (and an inflow
# generator wrapped around it, with its RNG and planes), and a parent step
# that may be running ahead on another task (SC-c, #403). `nest_state`
# records all of it with the structural walk of `Aeolus.IO`
# (`structural_state`, io/structural_state.jl): every array and field
# reachable from the run, so nothing that the next step reads is left out.
# The walk needs one adaptation, below: a parent step running ahead is
# waited for, recorded as plain values and detached, and rebuilt (with an
# already finished task) on restore.

using ..IO: IO, StructuralState, structural_state, restore_structural_state!

"Format tag of a [`nest_state`](@ref)."
const NEST_STATE_FORMAT = "Aeolus.nest_state"

"""
    NEST_STATE_VERSION

Version of the [`nest_state`](@ref) record. Version 1 (#269): the
structural state of the whole run (both domains, both parent time
levels, window, clocks, step-size memory, flux registers, statistics,
boundary data and inflow generator), the open window, and a parent step
running ahead.
"""
const NEST_STATE_VERSION = 1

# the prefetched parent step detached during a save (run => the _ParentAhead)
const _DETACHED = IdDict{Any,Any}()
const _DETACHED_LOCK = ReentrantLock()

function IO.structural_save_begin(run::NestedRun)
    a = run.ahead
    a === nothing && return (ahead = nothing, spare = run.s_spare !== nothing)
    wait_parent_step!(run)                      # rethrows the step's error
    rec = (t_p = a.t_p, dt = a.dt, Δt = a.Δt, t_end = a.t_end, limiter = a.limiter,
           parent_retries = a.parent_retries, wall_parent = a.wall_parent,
           wall_boundary = a.wall_boundary)
    # recorded as an idle run whose spare state holds the prefetched B(t_p + Δt_p)
    lock(() -> (_DETACHED[run] = (a, run.s_spare)), _DETACHED_LOCK)
    run.ahead = nothing
    run.s_spare = a.s
    return (ahead = rec, spare = true)
end

function IO.structural_save_end(run::NestedRun, extra)
    extra.ahead === nothing && return nothing
    a, spare = lock(() -> pop!(_DETACHED, run), _DETACHED_LOCK)
    run.s_spare = spare
    run.ahead = a
    return nothing
end

function IO.structural_restore_begin(run::NestedRun, extra)
    # wait for (and drop) a step a fresh run may have started ahead
    if run.ahead !== nothing
        wait_parent_step!(run)
        run.s_spare = run.ahead.s
        run.ahead = nothing
    end
    extra.spare && run.s_spare === nothing && (run.s_spare = _spare_state(run.s_new))
    return nothing
end

function IO.structural_restore_end(run::NestedRun, extra)
    extra.ahead === nothing && return nothing
    r = extra.ahead
    task = Threads.@spawn nothing
    wait(task)
    run.ahead = _ParentAhead(task, r.t_p, r.dt, run.s_spare, r.Δt, r.t_end, r.limiter,
                             r.parent_retries, r.wall_parent, r.wall_boundary)
    run.s_spare = nothing
    return nothing
end

"""
    nest_state(run::NestedRun; window = nothing) -> NamedTuple
    restore_nest_state!(run::NestedRun, state) -> window

The whole state of a nested run, for checkpoint / restart (#269): a
self-contained record, serialisable with `Serialization`, from which a run
**built the same way** (same meshes, cores, closures, boundary data,
feedback, generator) continues bit for bit. It holds

* both domains in full: the prognostic fields with their halos, the
  eddy coefficients, the SGS closures' memory (the closures of
  `advance_parent` / `advance_child`), the clocks, the open-boundary
  buffers and their mass-flux correction state;
* the nest: the parent field set prolongated at both ends of the current
  window (`s_old`, `s_new`) and the window `[t₀, t₀ + Δt_p]`, the step
  sizes the adaptive controllers continue from, the flux registers of a
  refluxing two-way run, the statistics ([`NestRunStats`](@ref)) and the
  feedback mode;
* the child's boundary data, including an inflow generator wrapped
  around it (`Transition.InflowTurbulence`: its planes, RNG and statistics);
* a parent step running ahead on another task
  ([`prefetch_parent_step!`](@ref)): it is waited for and recorded, and a
  restored run takes it as the sequential run would.

`window` is the open window of a caller that drives the child itself
([`begin_parent_step!`](@ref)), to checkpoint between two child steps of
a parent step; `restore_nest_state!` returns it (re-attached to the run's
two-way ledger scratch) for the [`end_parent_step!`](@ref) that closes it,
or `nothing`.

The record is [`structural_state`](@ref Aeolus.IO.structural_state) of the
run: everything reachable from it is recorded, which includes the two
simulations — a checkpoint that also stores them through
[`checkpoint!`](@ref Aeolus.IO.checkpoint!) stores them twice. As a
[`checkpoint_state`](@ref Aeolus.IO.checkpoint_state) component
(`checkpoint!(sim, path; components = (nest = run,))`) the window is not
passed: checkpoint at a window boundary there, or hold the window in a
component of your own.

`restore_nest_state!` throws an `ArgumentError` for a record that is not a
nest state (a snapshot without one: a nesting-on case cannot be restarted
from it), of a newer version, or of a run with another feedback mode,
synchronisation or meshes.
"""
function nest_state(run::NestedRun; window::Union{Nothing,NamedTuple} = nothing)
    return (format = NEST_STATE_FORMAT, version = NEST_STATE_VERSION,
            feedback = run.feedback, sync = run.sync,
            parent_size = size(run.parent.θ.data), child_size = size(run.child.θ.data),
            window = window === nothing ? nothing : _plain_window(window),
            run = structural_state(run))
end

# the window without its alias of the run's ledger scratch
_plain_window(w::NamedTuple) = merge(w, (I0 = copy(w.I0),))

@doc (@doc nest_state)
function restore_nest_state!(run::NestedRun, s)
    (s isa NamedTuple && get(s, :format, nothing) == NEST_STATE_FORMAT) ||
        throw(ArgumentError("restore_nest_state!: the snapshot holds no nest state " *
                            "(got $(s === nothing ? "nothing" : typeof(s))). A nested run cannot be " *
                            "restarted from it: the parent, its two time levels and the window would " *
                            "cold-start. Restart from a checkpoint written with the nest state (#269)."))
    s.version <= NEST_STATE_VERSION ||
        throw(ArgumentError("restore_nest_state!: nest state version $(s.version); this build " *
                            "reads up to $NEST_STATE_VERSION"))
    s.feedback === run.feedback ||
        throw(ArgumentError("restore_nest_state!: the checkpointed run is $(s.feedback), this run is " *
                            "$(run.feedback): a restart cannot change the feedback mode"))
    s.sync === run.sync ||
        throw(ArgumentError("restore_nest_state!: the checkpointed run has sync = :$(s.sync), this run :$(run.sync)"))
    (s.parent_size, s.child_size) == (size(run.parent.θ.data), size(run.child.θ.data)) ||
        throw(ArgumentError("restore_nest_state!: meshes differ: checkpoint parent $(s.parent_size), " *
                            "child $(s.child_size); this run $(size(run.parent.θ.data)), $(size(run.child.θ.data))"))
    restore_structural_state!(run, s.run)
    w = s.window
    w === nothing && return nothing
    run.work === nothing && return w
    copyto!(resize!(run.work.I0, length(w.I0)), w.I0)
    return merge(w, (I0 = run.work.I0,))
end

IO.checkpoint_state(run::NestedRun) = nest_state(run)
IO.restore_checkpoint_state!(run::NestedRun, s) = (restore_nest_state!(run, s); run)
