# Nesting: checkpoint / restart of a nested run (N-berlin-restart, #269).
#
# A NestedRun carries, besides its two simulations, the state of the nest:
# the parent field set prolongated at both ends of the last window
# (s_old, s_new), the clocks and the step sizes the adaptive controllers
# continue from, the flux registers of a refluxing two-way run, the
# statistics, and the child's boundary data (and an inflow generator
# wrapped around it, with its RNG and planes). `nest_state` records all of
# it with the structural walk of `Aeolus.IO` (`structural_state`,
# io/structural_state.jl): every array and field reachable from the run,
# so nothing that the next step reads is left out.

using ..IO: IO, StructuralState, structural_state, restore_structural_state!

"Format tag of a [`nest_state`](@ref)."
const NEST_STATE_FORMAT = "Aeolus.nest_state"

"""
    NEST_STATE_VERSION

Version of the [`nest_state`](@ref) record. Version 1 (#269): the
structural state of the whole run (both domains, both parent time
levels, clocks, step-size memory, flux registers, statistics,
boundary data and inflow generator).
"""
const NEST_STATE_VERSION = 1

"""
    nest_state(run::NestedRun) -> NamedTuple
    restore_nest_state!(run::NestedRun, state) -> run

The whole state of a nested run, for checkpoint / restart (#269): a
self-contained record, serialisable with `Serialization`, from which a run
**built the same way** (same meshes, cores, closures, boundary data,
feedback, generator) continues bit for bit. It holds

* both domains in full: the prognostic fields with their halos, the
  eddy coefficients, the SGS closures' memory (the closures of
  `advance_parent` / `advance_child`), the clocks, the open-boundary
  buffers and their mass-flux correction state;
* the nest: the parent field set prolongated at both ends of the last
  window (`s_old`, `s_new`), the step sizes the adaptive controllers
  continue from, the flux registers of a refluxing two-way run, the
  statistics ([`NestRunStats`](@ref)) and the feedback mode;
* the child's boundary data, including an inflow generator wrapped
  around it (`Transition.InflowTurbulence`: its planes, RNG and statistics).

The record is [`structural_state`](@ref Aeolus.IO.structural_state) of the
run: everything reachable from it is recorded, which includes the two
simulations — a checkpoint that also stores them through
[`checkpoint!`](@ref Aeolus.IO.checkpoint!) stores them twice. As a
[`checkpoint_state`](@ref Aeolus.IO.checkpoint_state) component
(`checkpoint!(sim, path; components = (nest = run,))`) it is taken
between two parent steps. (A model's nest, `Components.ModelNest`, is
checkpointed through its whole hierarchy.)

`restore_nest_state!` throws an `ArgumentError` for a record that is not a
nest state (a snapshot without one: a nesting-on case cannot be restarted
from it), of a newer version, or of a run with another feedback mode,
synchronisation or meshes.
"""
function nest_state(run::NestedRun)
    return (format = NEST_STATE_FORMAT, version = NEST_STATE_VERSION,
            feedback = run.feedback, sync = run.sync,
            parent_size = size(run.parent.θ.data), child_size = size(run.child.θ.data),
            run = structural_state(run))
end

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
    return run
end

IO.checkpoint_state(run::NestedRun) = nest_state(run)
IO.restore_checkpoint_state!(run::NestedRun, s) = (restore_nest_state!(run, s); run)
