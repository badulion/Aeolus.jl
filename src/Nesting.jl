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
       step_nested!, nest_conservation_error, composite_integral

include("nesting/grid.jl")
include("nesting/operators.jl")
include("nesting/interp_state.jl")
include("nesting/step.jl")

# -------------------------------------------------------------------------
# Phase 16 scaffold (refine = 1 direct copy), kept until it is deleted in
# the next commit.
# -------------------------------------------------------------------------

export NestSpec, apply_one_way_forcing!, apply_two_way_feedback!

"""
    NestSpec(parent, child; refine=1, offset=(1,1,1))

Records that `child` sits inside `parent` starting at parent-cell index
`offset`, with an integer refinement `refine` in each dimension.

`refine = 1` means the child grid is a strict subset of the parent
grid — no interpolation is needed, just copy. Higher refinements will
use linear interpolation once implemented.
"""
struct NestSpec{TP<:BoussinesqSimulation, TC<:BoussinesqSimulation}
    parent::TP
    child::TC
    refine::Int
    offset::NTuple{3,Int}
end

NestSpec(parent, child; refine::Int=1, offset::NTuple{3,Int}=(1,1,1)) =
    NestSpec(parent, child, refine, offset)

"""
    apply_one_way_forcing!(nest)

Copy parent field values at the *boundary* of the child domain into the
child's halo cells. Currently implemented for `refine = 1` only (direct
copy).
"""
function apply_one_way_forcing!(nest::NestSpec)
    nest.refine == 1 || error("apply_one_way_forcing!: only refine=1 implemented so far")
    _copy_boundary!(nest.child.u, nest.parent.u, nest)
    _copy_boundary!(nest.child.v, nest.parent.v, nest)
    _copy_boundary!(nest.child.w, nest.parent.w, nest)
    _copy_boundary!(nest.child.θ, nest.parent.θ, nest)
    return nest
end

"""
    apply_two_way_feedback!(nest)

Average child-interior values back into the parent cells the child
overlaps. Currently implemented for `refine = 1` only (direct copy).
"""
function apply_two_way_feedback!(nest::NestSpec)
    nest.refine == 1 || error("apply_two_way_feedback!: only refine=1 implemented so far")
    _copy_interior_back!(nest.parent.u, nest.child.u, nest)
    _copy_interior_back!(nest.parent.v, nest.child.v, nest)
    _copy_interior_back!(nest.parent.w, nest.child.w, nest)
    _copy_interior_back!(nest.parent.θ, nest.child.θ, nest)
    return nest
end

function _copy_boundary!(child::Field, parent::Field, nest)
    # Copy every boundary face of the child from the parent
    cn = child.mesh
    o1, o2, o3 = nest.offset
    Hxc, Hyc, Hzc = cn.Hx, cn.Hy, cn.Hz
    Nxc, Nyc, Nzc = cn.Nx, cn.Ny, cn.Nz
    Hxp, Hyp, Hzp = parent.mesh.Hx, parent.mesh.Hy, parent.mesh.Hz
    cd = child.data; pd = parent.data
    @inbounds for k in 1:Nzc
        for j in 1:Nyc
            # west face i=1 halo
            cd[Hxc, Hyc + j, Hzc + k] = pd[Hxp + o1 - 1, Hyp + o2 + j - 1, Hzp + o3 + k - 1]
            cd[Hxc + Nxc + 1, Hyc + j, Hzc + k] = pd[Hxp + o1 + Nxc, Hyp + o2 + j - 1, Hzp + o3 + k - 1]
        end
    end
    return child
end

function _copy_interior_back!(parent::Field, child::Field, nest)
    cn = child.mesh
    o1, o2, o3 = nest.offset
    Hxc, Hyc, Hzc = cn.Hx, cn.Hy, cn.Hz
    Nxc, Nyc, Nzc = cn.Nx, cn.Ny, cn.Nz
    Hxp, Hyp, Hzp = parent.mesh.Hx, parent.mesh.Hy, parent.mesh.Hz
    cd = child.data; pd = parent.data
    @inbounds for k in 1:Nzc, j in 1:Nyc, i in 1:Nxc
        pd[Hxp + o1 + i - 1, Hyp + o2 + j - 1, Hzp + o3 + k - 1] =
            cd[Hxc + i, Hyc + j, Hzc + k]
    end
    return parent
end

end # module
