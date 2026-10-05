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
# Files: nesting/grid.jl (NestBC, NestedGrid), nesting/operators.jl
# (weights, prolongation, restriction, refluxing), nesting/interp_state.jl
# (NestSponge, NestInterpState), nesting/step.jl (composite diagnostics,
# step_nested!).

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, CellCenter, FaceX, FaceY, FaceZ,
               Location, interior, totalsize
using ..Fields
using ..Fields: Field, location, interior_view, set!
using ..BoundaryConditions
using ..BoundaryConditions: AbstractBC, FieldBoundaryConditions, fill_halos!,
                            Periodic, NoFlux, ImpermeableWall, NoSlipWall
using ..Simulations
using ..Simulations: BoussinesqSimulation
using ..Operators: divergence
using ..Operators.FiniteVolume: DampingProfile, SinSquaredRamp, damping_rate
using ..TimeSteppers: step!

import ..BoundaryConditions: _fill_x!, _fill_y!, _fill_z!

export NestedGrid, nested_grid, refinement_ratio, nest_offset,
       nest_covered_range, nest_child_mesh, nest_parent_mesh,
       NestBC, nest_bcs,
       interpolate_to_nest_boundary!, interpolate_parent_to_child!,
       restrict_to_parent!,
       NestFluxMismatch, compute_nest_flux_mismatch, apply_flux_correction!,
       mismatch_l1, mismatch_area,
       NestSponge, apply_nest_sponge!,
       NestInterpState, nest_interp_state, interpolate_parent_state!,
       nest_scalar_names, apply_nest_boundary!,
       restrict_state_to_parent!, reflux!, reflux_scalars!,
       step_nested!, nest_conservation_error, composite_integral

include("nesting/grid.jl")
include("nesting/operators.jl")
include("nesting/interp_state.jl")
include("nesting/step.jl")

end # module
