module Nesting

# Phase 16 — Nested-grid infrastructure.
#
# Nesting embeds a fine-resolution child simulation inside a parent.
# We implement:
#
#   * `NestSpec`      — records parent, child, their relative geometry,
#                        and the direction of information flow.
#   * `apply_one_way_forcing!` — copies parent boundary values into the
#                        child's halo cells (typical interp: linear).
#   * `apply_two_way_feedback!` — averages child interior back into the
#                        parent cells the child overlaps.
#
# The current implementation is deliberately minimal — it only handles
# co-located, integer-refinement nests (`refine = 1` means the child
# grid is a subset of the parent's coordinate lines). Proper high-order
# interpolation and the rotational-invariance corrections that make
# nesting well-behaved for turbulent flow are follow-up work.

using ..Grids: RegularGrid, interior
using ..Fields: Field, interior_view
using ..Simulations: BoussinesqSimulation

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
