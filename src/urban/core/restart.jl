# Checkpoint / restart of the urban site (N-berlin-restart, #269; ONE-k
# #474: through the run loop's checkpoint components).
#
# The site is checkpointed whole: `checkpoint_state(case)` is the
# structural state of the built site (Aeolus.IO.structural_state): the
# child core and its open-boundary buffers, every coupling's state (soil
# and building columns, the surface bus, tracers, ledgers, radiation and
# chemistry caches, statistics), the clock, and with the online nest the
# parent case and the nested run (`Nesting.nest_state`'s content: both
# parent time levels, the open window, the parent step running ahead, the
# inflow generator). A resumed run builds the site from its config as a
# fresh run does and restores this record into it (the run loop's
# `resume!` with `Components.checkpoint_components` of its domain, where
# the site is `urban`), so a coupling added later is covered without a
# checkpoint method of its own.
#
# Not state: the output (the run's writer reopens its file), compiled plans
# and the run monitor.

Aeolus.IO.checkpoint_state(case::UrbanSite) = structural_state(case)

"""
    restore_checkpoint_state!(case::UrbanSite, s)

Restore the site's structural state `s`. A run that records the child's
boundary driver (`[urban] child_boundaries`, written in one session) is
refused: start a new run.
"""
function Aeolus.IO.restore_checkpoint_state!(case::UrbanSite, s)
    case.cfg.child_boundaries === nothing ||
        throw(ArgumentError("a run that records the child's boundary driver ([urban] child_boundaries) " *
                            "cannot be resumed: the recorder writes the file in one session; start a new run"))
    return restore_structural_state!(case, s)
end
