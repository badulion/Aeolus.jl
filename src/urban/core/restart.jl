# Checkpoint / restart of the Berlin case (N-berlin-restart, #269).
#
# The case is checkpointed whole: `checkpoint_state(case)` is the
# structural state of the built case (Aeolus.IO.structural_state): the
# child core and its open-boundary buffers, every coupling's state (soil
# and building columns, the surface bus, tracers, ledgers, radiation and
# chemistry caches, statistics), the clock, and with the online nest the
# parent case and the nested run (`Nesting.nest_state`'s content: both
# parent time levels, the open window, the parent step running ahead, the
# inflow generator). A resumed run builds the case from its config as a
# fresh run does and restores this record into it, so a coupling added
# later is covered without a checkpoint method of its own.
#
# Not state: the NetCDF output (a resumed run reopens the run's file and
# continues at the checkpoint's record), compiled plans and the run
# monitor.

# the open NetCDF file of the case: the resumed run reopens it
Aeolus.IO.checkpoint_opaque(::CaseOutput) = true

Aeolus.IO.checkpoint_state(case::Case) = structural_state(case)
Aeolus.IO.restore_checkpoint_state!(case::Case, s) = restore_structural_state!(case, s)

"""
    case_checkpoint_components(case) -> NamedTuple

The [`checkpoint!`](@ref Aeolus.IO.checkpoint!) components of a Berlin
case: `(case = case,)`, the whole built case as its structural state
(`structural_state`, #269). With the online nest that includes the parent
case and the nested run (`nest_state`).
"""
case_checkpoint_components(case) = (case = case,)

"""
    check_case_checkpoint(case, payload, path)

Refuse a checkpoint that cannot restart `case`: one without the case's
own state (written by `checkpoint!` of the core alone, or by a build
before #269) — a nested case would cold-start its parent, the two parent
time levels and every coupling; and any checkpoint of a run that records
the child's boundary driver (`[output] child_boundaries`, one file per
session).
"""
function check_case_checkpoint(case, d::AbstractDict, path)
    case.cfg.child_boundaries === nothing ||
        throw(ArgumentError("a run that records the child's boundary driver ([output] child_boundaries) " *
                            "cannot be resumed: the recorder writes the file in one session; start a new run"))
    haskey(d[:components], :case) && return nothing
    nested = find_coupling(case, :nesting) !== nothing
    throw(ArgumentError("the checkpoint $path holds no Berlin case state" *
                        (nested ? " and no nest state" : "") * " (only the core's fields): " *
                        "the case cannot be resumed from it — " *
                        (nested ? "the nest parent, its two time levels and the open window, and " : "") *
                        "every coupling would cold-start. It was not written by the berlin driver " *
                        "(#269); start a new run."))
end
