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
    ii, jj, kk = interior(m)
    mk = sm.mask
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    tud = sim.tu.data; tvd = sim.tv.data; twd = sim.tw.data
    invτ = 1 / τ
    @inbounds for k in kk, j in jj, i in ii
        # A face is "in solid" if either bordering cell is solid.
        if mk[i, j, k]
            tud[i,   j, k] -= invτ * ud[i,   j, k]
            tud[i+1, j, k] -= invτ * ud[i+1, j, k]
            tvd[i, j,   k] -= invτ * vd[i, j,   k]
            tvd[i, j+1, k] -= invτ * vd[i, j+1, k]
            twd[i, j, k]   -= invτ * wd[i, j, k]
            twd[i, j, k+1] -= invτ * wd[i, j, k+1]
        end
    end
    return sim
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
"""
function implicit_ibm_correction!(sim, sm::SolidMask; τ::Real = 0.0, dt::Real = 0.0)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    mk = sm.mask
    factor = τ > 0 ? 1 / (1 + dt / τ) : 0.0
    @inbounds for k in kk, j in jj, i in ii
        if mk[i, j, k]
            ud[i,   j, k] *= factor
            ud[i+1, j, k] *= factor
            vd[i, j,   k] *= factor
            vd[i, j+1, k] *= factor
            wd[i, j, k]   *= factor
            wd[i, j, k+1] *= factor
        end
    end
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
