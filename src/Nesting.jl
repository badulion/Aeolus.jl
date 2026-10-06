module Nesting

# =========================================================================
# Phase 30 — parent ↔ child grid nesting. The single home of every
# parent-child operator (N-nest-a, #250: moved here from AdaptiveMesh.jl,
# which keeps only the Phase 31 block-structured AMR and delegates to this
# module for its prolongation / restriction / refluxing).
# =========================================================================
#
# Everything here is built around three primitives:
#
#   1. A *conservative, high-order* prolongation (parent -> child).
#      Implemented via the primitive-function (cumulative-integral)
#      construction: the parent cell averages define the primitive `F`
#      exactly at parent faces; a degree-`ns` polynomial interpolates `F`
#      there; the child cell average is the polynomial's difference
#      quotient across the child cell. Because `F` is matched *exactly* at
#      the parent faces, the r sub-cell averages inside a parent cell sum
#      back to the parent average to machine precision — conservation is
#      structural, not approximate. `ns = order` parent cells in the
#      stencil gives a formal order-`order` reconstruction.
#
#   2. A *conservative* restriction (child -> parent): volume-weighted
#      average for cell-centred fields, area-weighted average for
#      face-located (flux) fields.
#
#   3. A *flux register / refluxing* correction at the coarse-fine
#      interface (Berger-Colella). Without it two-way nesting leaks: the
#      coarse cells outside the nest were advanced with the coarse flux
#      while the fine grid used the fine fluxes, so the composite
#      integral drifts.
#
# Staggering is respected throughout: a `FaceX` field is a *point value*
# in x (interpolated with a nodal Lagrange stencil, so a child face that
# coincides with a parent face reproduces the parent value bit-for-bit)
# and a *face-area average* in y and z (interpolated conservatively, so
# the summed fine fluxes through a parent face equal the parent flux).
# That combination is what makes the prolongation flux-preserving.
#
# N-nest-b (#256) generalises the three primitives to a z-stretched parent
# (Lagrange nodes on the actual parent face heights), to the reference-mass
# measure of the sound-proof systems (W-weighted restriction, prolongation
# conserving ∫Wφ dV, W = ρ̄ or ρ̄θ̄ from the sims' ReferenceState), and to
# solid-aware restriction (child cells / faces inside IBM buildings are
# left out of the parent average, Hellsten et al. 2021, GMD 14, 3185, §3.5).
#
# Files: nesting/grid.jl (nest_bcs, NestedGrid), nesting/measure.jl
# (NestMeasure, fluid mask), nesting/operators.jl
# (weights, prolongation, restriction, refluxing), nesting/interp_state.jl
# (NestSponge, NestInterpState), nesting/boundary_data.jl (NestBoundaryData:
# the parent as the boundary data of the child's OpenBoundary faces, N-nest-d
# #261), nesting/step.jl (composite diagnostics,
# restriction / refluxing of the field set), nesting/feedback.jl (two-way
# feedback: TwoWayFeedback, flux registers, anterpolation with buffer,
# canopy restriction, relaxation and inflow exclusion, re-projection;
# N-nest-e #263), nesting/nested_run.jl (NestedRun:
# per-domain advance closures, subcycling under adaptive steppers,
# temporal interpolation of the boundary data, step_nested!; N-nest-c #257).

using Printf: @sprintf
using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ,
               Location, interior, totalsize
using ..Fields
using ..Fields: Field, location, interior_view, set!
using ..BoundaryConditions
using ..BoundaryConditions: AbstractBC, FieldBoundaryConditions, fill_halos!,
                            AbstractBoundaryData, OpenPlane, OpenBoundary,
                            OpenBoundaryState, open_boundary_conditions,
                            set_boundary_data!, open_boundary_states,
                            has_open_boundaries, update_open_boundaries!,
                            open_boundary_snapshot, restore_open_boundaries!,
                            Periodic, NoFlux, ImpermeableWall, NoSlipWall,
                            mark_open_boundary_solids!
using ..Simulations
using ..Simulations: BoussinesqSimulation, step_rk3!, _dzf, _project_velocity!
using ..StretchedGrids: StretchedGrid, zfaces, cell_thickness
using ..Anelastic: MassWeightedSimulation, ReferenceState, constraint_weights,
                   _project_mass_weighted!
using ..IBM: SolidMask
using ..Operators: sgs_closure, sgs_state, restore_sgs_state!, sgs_fields,
                   sgs_scalar_face_diffusivity, sgs_tracer_diffusivity!, _active_sgs
using ..AdvectionSchemes: advective_face_flux, _flux_reconstruction, scalar_face_flux,
                           face_diffusivity, face_inv_spacing
using ..TimeIntegration: CFLConstraints, adaptive_dt, StateSnapshot, save_state!, state_snapshot,
                         restore_state!, IMEXScheme, ARK2, Rosenbrock2, step_imex!,
                         _all_finite, _max_advective_rate, is_numerical_failure
using ..Operators: divergence
using ..Operators.FiniteVolume: DampingProfile, SinSquaredRamp, damping_rate
using ..TimeSteppers: step!

import ..BoundaryConditions: boundary_values!

export NestedGrid, nested_grid, refinement_ratio, nest_offset,
       nest_covered_range, nest_child_mesh, nest_parent_mesh,
       nest_bcs, NestBoundaryData, nest_boundary_data, nest_open_sides,
       interpolate_to_nest_boundary!, interpolate_parent_to_child!,
       restrict_to_parent!,
       NestFluxMismatch, compute_nest_flux_mismatch, apply_flux_correction!,
       mismatch_l1, mismatch_area,
       NestSponge, apply_nest_sponge!,
       NestInterpState, nest_interp_state, interpolate_parent_state!,
       nest_scalar_names, apply_nest_boundary!,
       restrict_state_to_parent!, reflux!, reflux_scalars!,
       step_nested!, nest_conservation_error, composite_integral,
       NestMeasure, nest_measure, nest_reference_mismatch, nest_fluid_mask,
       DomainAdvance, domain_advance, stage_times,
       NestedRun, NestRunStats, boundary_time, begin_parent_step!, end_parent_step!, subcycle_dt,
       begin_child_step!, end_child_step!,
       TwoWayFeedback, NestScalarFlux, NestFluxRegisters, two_way_feedback!,
       anterpolate!, feedback_mask, book_nest_flux!, nest_stage_weights

"Simulations a nest can couple: the Boussinesq core (uniform or z-stretched) and the sound-proof `MassWeightedSimulation`s (anelastic, pseudo-incompressible)."
const _NestSim = Union{BoussinesqSimulation, MassWeightedSimulation}

_moist(sim::BoussinesqSimulation) = sim.has_moisture
_moist(::MassWeightedSimulation)  = false

include("nesting/grid.jl")
include("nesting/measure.jl")
include("nesting/operators.jl")
include("nesting/interp_state.jl")
include("nesting/boundary_data.jl")
include("nesting/step.jl")
include("nesting/domain_advance.jl")
include("nesting/feedback.jl")
include("nesting/nested_run.jl")

end # module
