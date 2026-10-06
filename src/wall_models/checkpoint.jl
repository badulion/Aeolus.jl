# =========================================================================
# Checkpoint state of the wall models (#245): methods of the IO component
# generic `checkpoint_state` / `restore_checkpoint_state!` for every
# WallModels object that carries memory between steps, so a restarted
# wall-modelled run continues bit for bit.
# =========================================================================

import ..IO: checkpoint_state, restore_checkpoint_state!
using ..IO: fieldwise_checkpoint_state, restore_fieldwise!

# MatchingInput: the temporal filter's state (`work` is scratch).
checkpoint_state(mi::MatchingInput) = (uf = copy(mi.uf), vf = copy(mi.vf), primed = mi.primed)
function restore_checkpoint_state!(mi::MatchingInput, s)
    mi.uf = copy(s.uf); mi.vf = copy(s.vf); mi.primed = s.primed
    return mi
end

# WallModel: immutable; a per-column law matrix may be rewritten in place
# (the SBL driver's MOST laws, the dynamic roughness's laws).
checkpoint_state(wm::WallModel) = wm.law isa AbstractMatrix ? copy(wm.law) : nothing
function restore_checkpoint_state!(wm::WallModel, s)
    s === nothing || copyto!(wm.law, s)
    return wm
end

# TBLEWallModel: the per-wall columns (profiles, stresses), the walls'
# default inputs and the dynamic κ — all in `walls`, created on first use.
checkpoint_state(wm::TBLEWallModel) = deepcopy(wm.walls)
function restore_checkpoint_state!(wm::TBLEWallModel, s)
    empty!(wm.walls)
    merge!(wm.walls, deepcopy(s))
    return wm
end

# DynamicRoughness: α, its convergence flag, the z0 field and the counters.
checkpoint_state(dr::DynamicRoughness) =
    fieldwise_checkpoint_state(dr, (:α, :converged, :z0, :nsolves, :nfail))
restore_checkpoint_state!(dr::DynamicRoughness, s) = restore_fieldwise!(dr, s)

# WallBoundedCase: everything but the simulation (stored by checkpoint!
# itself) — the closure memory, the wall model, the inputs, the roughness,
# and the scalar fields (forcing, stresses, fluxes, counters).
const _CASE_SCALARS = (:forcing, :u_star, :height, :τ_wall_bottom, :τ_wall_top, :nsteps,
                       :_stage, :τ_form, :q_wall_bottom, :q_wall_top)

checkpoint_state(c::WallBoundedCase) =
    (closure = checkpoint_state(c.closure), wall = checkpoint_state(c.wall),
     inputs = checkpoint_state(c.inputs), roughness = checkpoint_state(c.roughness),
     scalars = fieldwise_checkpoint_state(c, _CASE_SCALARS))
function restore_checkpoint_state!(c::WallBoundedCase, s)
    restore_checkpoint_state!(c.closure, s.closure)
    restore_checkpoint_state!(c.wall, s.wall)
    restore_checkpoint_state!(c.inputs, s.inputs)
    restore_checkpoint_state!(c.roughness, s.roughness)
    restore_fieldwise!(c, s.scalars)
    return c
end
