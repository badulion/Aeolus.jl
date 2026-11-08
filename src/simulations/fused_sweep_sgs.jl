# The SGS closure inside the fused sweep of the Boussinesq step core (#533
# plan step 4). Included into `module Simulations`, after fused_sweep.jl.
#
# `_euler_projection!` hands the sweep's decision the closure together
# with its `sgs_stress` (an `SGSChoice`, or `nothing` without a closure),
# so the decision functions keep their signatures. The closure's term
# (`Operators.sgs_sweep_term`) computes the coefficients in its
# `prepare!`, as the separate loops do in step 4 of `_euler_projection!`,
# and adds the momentum, θ and qv tendencies inside the sweep.

"""
    SGSChoice(closure, stress)

The SGS closure of a substage and the form of its momentum divergence
(the `sgs_stress` keyword), as the fused sweep's decision receives them.
"""
struct SGSChoice{C}
    closure::C
    stress::Symbol
end
_sgs_choice(::Nothing, stress::Symbol) = nothing
_sgs_choice(c, stress::Symbol) = SGSChoice(c, stress)

# The sweep terms of the closure: () without one, (term,) when it has one,
# `nothing` when it cannot run inside the sweep (its momentum or scalar
# tendency has no part the sweep evaluates).
_sgs_terms(sim, ::Nothing) = ()
function _sgs_terms(sim, c::SGSChoice)
    t = Operators.sgs_sweep_term(c.closure, sim, c.stress, sim.lap_θ)
    return t === nothing ? nothing : (t,)
end

# what keeps the closure out of the sweep: its type, with its documented
# reason (or the default "no sweep_term …")
_sgs_blockers!(out, sim, ::Nothing) = out
_sgs_blockers!(out, sim, c::SGSChoice) =
    _sgs_terms(sim, c) === nothing ? push!(out, _blocker(typeof(c.closure))) : out

# the fused sweep times the closure's coefficients as the separate loops
# time the whole SGS step (its tendencies run inside the sweep)
function Operators.FiniteVolume.prepare!(t::Operators.SGSTerm, sim::BoussinesqSimulation, Δt)
    t0 = time_ns()
    invoke(prepare!, Tuple{Operators.SGSTerm, Any, Any}, t, sim, Δt)
    sim.workspaces.rec.wall_sgs += _wall(t0)
    return nothing
end
