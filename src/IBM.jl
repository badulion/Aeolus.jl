module IBM

# Phase 13 — Immersed Boundary Method (direct forcing).
#
# We represent solid obstacles inside the fluid domain by a Boolean mask
# defined at cell centers. The direct-forcing approach sets velocity in
# solid cells to zero every time step; the pressure-projection solve
# then makes the surrounding flow divergence-free naturally, giving a
# no-penetration wall around each solid cell.
#
# Extensions (later phases): ghost-cell IBM for higher-order wall
# treatment; moving obstacles; sub-cell reconstruction of the interface.

using ..Grids: RegularGrid, interior
using ..Fields: Field, interior_view
using ..Backends: launch_weighted!, backend     # KA-k (#307): the direct forcing is a launch over the faces
using ..Simulations: BoussinesqSimulation
using ..BoundaryConditions: FieldBoundaryConditions, neighbor_index

export SolidMask, mark_solid!, apply_solid_mask!, add_ibm_penalty!,
       implicit_ibm_correction!, wall_distance, first_level_cells

"""
    SolidMask(mesh; init = zeros)

A Boolean mask on `mesh` (cell centers) marking solid cells. `true` =
solid, `false` = fluid. Storage matches a `Field(CellCenter, mesh)`
(interior + halos) so it can be inspected with `interior_view` on the
underlying `mask` array.
"""
struct SolidMask{M<:RegularGrid, A<:AbstractArray{Bool,3}}
    mesh::M
    mask::A
end

function SolidMask(mesh::RegularGrid; init = nothing)
    N = (mesh.Nx + 2mesh.Hx, mesh.Ny + 2mesh.Hy, mesh.Nz + 2mesh.Hz)
    m = falses(N...)
    init === nothing || (m .= init)
    return SolidMask(mesh, m)
end

"""
    mark_solid!(sm, predicate)

Set `sm.mask[i,j,k] = true` for every interior cell whose center
satisfies `predicate(x, y, z)`.
"""
function mark_solid!(sm::SolidMask, predicate::Function)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    xs = m.x0 .+ ((1:m.Nx) .- 0.5) .* m.dx
    ys = m.y0 .+ ((1:m.Ny) .- 0.5) .* m.dy
    zs = m.z0 .+ ((1:m.Nz) .- 0.5) .* m.dz
    @inbounds for (ki, kv) in enumerate(kk),
                   (ji, jv) in enumerate(jj),
                   (ii_, iv) in enumerate(ii)
        sm.mask[iv, jv, kv] = predicate(xs[ii_], ys[ji], zs[ki])
    end
    return sm
end

"""
    apply_solid_mask!(sim, sm)

Zero the face-centered velocity components on any face that touches a
solid cell (either side of the face is masked). Applied after the
pressure projection every time step.
"""
function apply_solid_mask!(sim::BoussinesqSimulation, sm::SolidMask)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    mk = sm.mask
    @inbounds for k in kk, j in jj, i in ii
        if mk[i, j, k]
            # zero all six faces of the solid cell
            ud[i,   j,   k] = 0
            ud[i+1, j,   k] = 0
            vd[i,   j,   k] = 0
            vd[i,   j+1, k] = 0
            wd[i,   j,   k] = 0
            wd[i,   j,   k+1] = 0
        end
    end
    return sim
end

"""
    add_ibm_penalty!(sim, sm; τ = 1e-3)

Angot-style volume penalization: add `-(mask / τ) · u` to each momentum
tendency (`sim.tu`, `sim.tv`, `sim.tw`). With small `τ` the tendency
strongly drives `u → 0` in solid cells while leaving fluid cells alone;
the pressure projection then resolves the flow around the obstacle in
one consistent solve.

Call inside a custom step just after computing tendencies and *before*
the predictor step, so the penalty is included in the flow the
projection sees.
"""
function add_ibm_penalty!(sim, sm::SolidMask; τ::Real = 1e-3)
    m  = sm.mesh
    invτ = 1 / τ
    _ibm_faces!(_penalty_op, sm, (sim.tu.data, sim.tv.data, sim.tw.data),
                (sim.u.data, sim.v.data, sim.w.data), invτ)
    return sim
end

# The face loops of the direct forcing (KA-k, #307). A face "in solid" is
# one with a solid cell on either side; the former loop over the cells
# visited it once per solid cell (lower cell first) and applied `op!` each
# time. Here every face belongs to one point of a launch, which applies
# `op!` once per solid neighbour, lower first: the same operations in the
# same order, so the same bits, and no two work items write the same face.
# `op!(out, src, I, J, K, x)`; `out` and `src` are the (u, v, w) data arrays
# written and read.
@inline _penalty_op(out, src, I, J, K, invτ) = (@inbounds out[I, J, K] -= invτ * src[I, J, K]; nothing)
@inline _scale_op(out, src, I, J, K, factor) = (@inbounds out[I, J, K] *= factor; nothing)

# One launch over the rows (j, k) of cells, plus the rows above the last
# ones: row (j, k) owns the u faces i1:i2+1 of the row and the v and w faces
# of its cells' lower sides (at j and at k). A row whose cells and lower
# neighbour rows have no solid cell — most rows — is skipped after three
# word-wise scans of the (bit) mask. Otherwise the row applies, per face,
# the operation of its lower solid neighbour first (the cells of rows j-1
# and k-1 for the v and w faces; cell i-1, met one iteration earlier, for
# the u face i), then that of its own solid cells: one mask read per cell,
# the order of the former loop over the cells. Walking the row face by face
# instead (up to six mask reads per face) measured 1.14x the former loop in
# the flagship (244 vs 219 µs per call); this cell-wise walk 0.76x (167 µs).
@inline function _ibm_row_body!(j, k, op!::F, ou, ov, ow, su, sv, sw, mk, x,
                                i1, i2, j1, j2, k1, k2) where {F}
    @inbounds begin
        a0 = j <= j2 && k <= k2 && _row_any(mk, i1, i2, j, k)          # cells (·, j, k)
        ay = j - 1 >= j1 && k <= k2 && _row_any(mk, i1, i2, j - 1, k)  # cells (·, j-1, k)
        az = k - 1 >= k1 && j <= j2 && _row_any(mk, i1, i2, j, k - 1)  # cells (·, j, k-1)
        if ay                                       # v faces (·, j, k) of solid cells (·, j-1, k)
            for i in i1:i2
                mk[i, j-1, k] && op!(ov, sv, i, j, k, x)
            end
        end
        if az                                       # w faces (·, j, k) of solid cells (·, j, k-1)
            for i in i1:i2
                mk[i, j, k-1] && op!(ow, sw, i, j, k, x)
            end
        end
        if a0                                       # the row's own solid cells
            for i in i1:i2
                if mk[i, j, k]
                    op!(ou, su, i, j, k, x); op!(ou, su, i + 1, j, k, x)
                    op!(ov, sv, i, j, k, x); op!(ow, sw, i, j, k, x)
                end
            end
        end
    end
    return nothing
end

# whether any of mk[i1:i2, j, k] is true; a BitArray is scanned 64 cells per word
_row_any(mk::AbstractArray{Bool,3}, i1, i2, j, k) = (@inbounds for i in i1:i2; mk[i, j, k] && return true; end; false)
@inline function _row_any(mk::BitArray{3}, i1, i2, j, k)
    n1, n2 = size(mk, 1), size(mk, 2)
    l1 = i1 + (j - 1) * n1 + (k - 1) * n1 * n2            # linear indices of the row
    l2 = l1 + (i2 - i1)
    c = mk.chunks
    w1, w2 = ((l1 - 1) >> 6) + 1, ((l2 - 1) >> 6) + 1
    lo = ~UInt64(0) << ((l1 - 1) & 63)                     # bits from l1 on, in word w1
    hi = ~UInt64(0) >> (63 - ((l2 - 1) & 63))              # bits up to l2, in word w2
    @inbounds begin
        w1 == w2 && return (c[w1] & lo & hi) != 0
        (c[w1] & lo) != 0 && return true
        for w in w1+1:w2-1
            c[w] != 0 && return true
        end
        return (c[w2] & hi) != 0
    end
end

function _ibm_faces!(op!::F, sm::SolidMask, out::NTuple{3}, src::NTuple{3}, x) where {F}
    ii, jj, kk = interior(sm.mesh)
    i1, i2, j1, j2, k1, k2 = first(ii), last(ii), first(jj), last(jj), first(kk), last(kk)
    # a row covers its i2 - i1 + 2 faces of each component
    launch_weighted!(backend(src[1]), i2 - i1 + 2, _ibm_row_body!, (j1:j2+1, k1:k2+1), op!,
                     out..., src..., sm.mask, x, i1, i2, j1, j2, k1, k2)
    return nothing
end

"""
    implicit_ibm_correction!(sim, sm; τ = 0.0, dt = 0.0)

Volume-penalization applied *implicitly* to the intermediate velocity
after the predictor step. For each face touching a solid cell:

    u* ← u* / (1 + Δt/τ)

Unlike [`add_ibm_penalty!`](@ref) — which is *explicit* and requires
`Δt / τ < 1` for stability — this form is unconditionally stable for
any positive `τ`. `τ = 0` treats the penalty as infinite and just zeros
the face velocity (`u* ← 0`). Best combined with `step!(sim, dt;
post_predictor! = (s, Δt) -> implicit_ibm_correction!(s, sm; dt = Δt))`.

A face between two solid cells is scaled once per solid cell (the factor
squared; it lies inside the solid). The faces are written by one launch
(KA-k, #307), each by one work item: the same bits at every thread count
and as the former loop over the cells.
"""
function implicit_ibm_correction!(sim, sm::SolidMask; τ::Real = 0.0, dt::Real = 0.0)
    factor = τ > 0 ? 1 / (1 + dt / τ) : 0.0
    q = (sim.u.data, sim.v.data, sim.w.data)
    _ibm_faces!(_scale_op, sm, q, q, factor)
    return sim
end

# =========================================================================
# Wall distance (Era VI, P56-b #182) — the one source of truth for the
# distance from a cell centre to the nearest solid surface.
# =========================================================================

"""
    wall_distance(sm::SolidMask; bottom = true, top = false,
                  periodic = (true, true, false), bcs = nothing,
                  maxsweeps = 50) -> Array{T,3}

Distance (m) from the centre of every interior cell to the nearest solid
surface, an interior-sized `Nx × Ny × Nz` array (0 in solid cells). The
surfaces are the faces of the solid cells of `sm` (each solid cell is the
box it occupies) and, with `bottom = true` (`top = true`), the domain floor
(lid).

A **vector distance transform** (Danielsson 1980), the chamfer-type
raster-sweep algorithm that propagates the nearest surface *point* rather
than a scalar distance, with no dependencies:

1. *Seed.* A fluid cell next to a surface stores the displacement to the
   exact nearest point of it: the centre clamped into the box of each solid
   neighbour among its 26 (`Δz/2` above a floor or roof, `Δx/2` beside a
   wall, `√(Δx² + Δz²)/2` off a roof edge), or the foot point on the floor
   (lid) for level 1 (`Nz`).
2. *Propagate.* Forward and backward raster sweeps over the
   26-neighbourhood offer each cell its neighbours' nearest points,
   `r(c) ← r(n) + (x_n − x_c)`, and keep the closest, until nothing changes
   (`periodic` wraps the lateral neighbours across the seam; a velocity
   boundary-condition set `bcs`, when given, decides instead, through
   `BoundaryConditions.neighbor_index`: an open or wall face has no
   neighbour beyond it, N-open-d #262).

The distance is exact for a plane surface aligned with the grid
(`d = Δz/2 + (k−1)Δz` above a floor) and next to every surface; elsewhere
the raster propagation can miss the true nearest point only in rare
configurations (Danielsson 1980). Around a cube on a 2:2:1 grid it equals
the Euclidean distance to the box to round-off (`test/era6_tke.jl`); a
plain chamfer transform with the 26 step lengths overestimated it there by
up to 24 %.

The TKE closure (`PrognosticSGS.TKEClosure`) caps its mixing length at
`κ · wall_distance`; every Era VI closure that needs a wall distance takes
it from here.

References: Danielsson, P.-E. (1980), *Euclidean distance mapping*, Comput.
Graph. Image Process. 14, 227; Borgefors, G. (1986), *Distance
transformations in digital images*, Comput. Vision Graph. Image Process.
34, 344.
"""
function wall_distance(sm::SolidMask; bottom::Bool = true, top::Bool = false,
                       periodic::NTuple{3,Bool} = (true, true, false),
                       bcs::Union{Nothing,FieldBoundaryConditions} = nothing,
                       maxsweeps::Integer = 50)
    m = sm.mesh
    T = typeof(m.dx)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    dx, dy, dz = m.dx, m.dy, m.dz
    solid = Array{Bool}(undef, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        solid[i, j, k] = sm.mask[Hx+i, Hy+j, Hz+k]
    end
    offs = [(a, b, c) for c in -1:1 for b in -1:1 for a in -1:1 if (a, b, c) != (0, 0, 0)]
    # neighbour index with wrap-around (or `0` outside a non-periodic edge):
    # the one BC-aware helper, on `bcs` or on the axes `periodic` names
    nb = bcs === nothing ?
        FieldBoundaryConditions(ntuple(a -> periodic[a] ? :periodic : :noflux, 3)...) : bcs
    @inline wrap(i, a) = neighbor_index(m, nb, i, a; outside = :none)
    # displacement from the cell centre to its nearest surface point
    inf = T(Inf)
    rx = fill(inf, Nx, Ny, Nz); ry = fill(inf, Nx, Ny, Nz); rz = fill(inf, Nx, Ny, Nz)
    n2(x, y, z) = x * x + y * y + z * z
    # 1. seed: the exact nearest point of every adjacent surface
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if solid[i, j, k]
            rx[i, j, k] = ry[i, j, k] = rz[i, j, k] = zero(T)
            continue
        end
        bx, by, bz = inf, inf, inf
        if bottom && k == 1
            bx, by, bz = zero(T), zero(T), -dz / 2
        end
        if top && k == Nz && n2(0, 0, dz / 2) < n2(bx, by, bz)
            bx, by, bz = zero(T), zero(T), dz / 2
        end
        for o in offs
            ii = wrap(i + o[1], 1); jj = wrap(j + o[2], 2); kk = wrap(k + o[3], 3)
            (ii == 0 || jj == 0 || kk == 0) && continue
            solid[ii, jj, kk] || continue
            # the centre clamped into the neighbour's box: offset o·(Δ/2) per axis
            cx, cy, cz = o[1] * dx / 2, o[2] * dy / 2, o[3] * dz / 2
            n2(cx, cy, cz) < n2(bx, by, bz) && ((bx, by, bz) = (cx, cy, cz))
        end
        rx[i, j, k], ry[i, j, k], rz[i, j, k] = bx, by, bz
    end
    # 2. propagate the nearest points: forward (neighbours earlier in raster
    #    order), then backward, until no displacement changes
    fwd = [o for o in offs if (o[3], o[2], o[1]) < (0, 0, 0)]
    bwd = [o for o in offs if (o[3], o[2], o[1]) > (0, 0, 0)]
    function sweep!(order, os)
        changed = false
        @inbounds for k in order[3], j in order[2], i in order[1]
            solid[i, j, k] && continue
            bx, by, bz = rx[i, j, k], ry[i, j, k], rz[i, j, k]
            best = n2(bx, by, bz)
            for o in os
                ii = wrap(i + o[1], 1); jj = wrap(j + o[2], 2); kk = wrap(k + o[3], 3)
                (ii == 0 || jj == 0 || kk == 0) && continue
                solid[ii, jj, kk] && continue            # seeds already hold solid boxes
                isfinite(rx[ii, jj, kk]) || continue
                cx = rx[ii, jj, kk] + o[1] * dx
                cy = ry[ii, jj, kk] + o[2] * dy
                cz = rz[ii, jj, kk] + o[3] * dz
                c = n2(cx, cy, cz)
                if c < best
                    best = c; bx, by, bz = cx, cy, cz; changed = true
                end
            end
            rx[i, j, k], ry[i, j, k], rz[i, j, k] = bx, by, bz
        end
        return changed
    end
    for _ in 1:maxsweeps
        a = sweep!((1:Nx, 1:Ny, 1:Nz), fwd)
        b = sweep!((Nx:-1:1, Ny:-1:1, Nz:-1:1), bwd)
        (a || b) || break
    end
    return sqrt.(rx .^ 2 .+ ry .^ 2 .+ rz .^ 2)
end

"""
    first_level_cells(sm::SolidMask; bottom = true) -> Array{Bool,3}

The fluid cells at the first grid level above a surface: those with a solid
cell directly below, and (with `bottom = true`) every fluid cell of level 1
(above the domain floor). Interior-sized. The TKE closure multiplies its
dissipation coefficient there by Moeng's (1984) factor 3.9.
"""
function first_level_cells(sm::SolidMask; bottom::Bool = true)
    m = sm.mesh
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    f = falses(Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        sm.mask[Hx+i, Hy+j, Hz+k] && continue
        f[i, j, k] = k == 1 ? bottom : sm.mask[Hx+i, Hy+j, Hz+k-1]
    end
    return Array{Bool}(f)
end

end # module
