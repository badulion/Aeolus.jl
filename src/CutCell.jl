module CutCell

# =========================================================================
# Phases 21-22 — cut-cell immersed boundary method with ghost-cell wall
#                reconstruction.
# =========================================================================
#
# This module supersedes the volume-penalization / direct-forcing IBM in
# `IBM.jl`. Instead of smearing the wall over a cell with a penalty term,
# the solid surface is represented *geometrically*:
#
#   * every cell carries a wetted volume fraction  V_f = V_cell * vol_frac
#   * every face carries an open-area fraction     A_f = A_face  * aperture
#   * every cut cell carries the wall normal and the wetted wall area
#
# All finite-volume operators are then re-derived on that geometry. The
# discrete divergence of a cell is the aperture-weighted face-flux balance
# divided by the wetted volume, and the discrete gradient is the ordinary
# two-point difference multiplied by the aperture. With those definitions
# the pair (D, G) is an *exact negative adjoint pair* with respect to the
# wetted-volume inner product,
#
#       <p, D u>_V  =  - <G p, u>_A ,
#
# which is what makes the fractional-step projection discretely exact on a
# cut geometry (and makes L = D G symmetric negative-definite, so the
# existing CG pressure solver still converges). See `cutcell_cell_inner`
# and `cutcell_face_inner` for the two inner products, and the Era II test
# file for the numerical verification of the identity.
#
# Geometry is built by sub-cell sampling of a level set (`nsub^3` samples
# per cell, `nsub^2` per face). This is robust for arbitrary implicit
# geometry (terrain, buildings, bluff bodies) and converges like
# O(1/nsub) in the fractions. A marching-cubes / exact-polyhedron variant
# would give machine-exact fractions for piecewise-planar surfaces and is
# the natural future refinement; the rest of the module is agnostic to how
# the fractions were produced, so it can be dropped in without touching
# the operators.
#
# Phase 22 adds ghost-cell reconstruction: for every sample point that
# falls inside the solid but is touched by a fluid stencil we precompute
# the foot point on the *true* surface, an image point out in the fluid,
# and the interpolation weights that evaluate a field there. Imposing the
# wall condition on the line (ghost, wall, image) is second-order accurate
# and — unlike a staircase boundary — puts the boundary where the surface
# actually is.
#
# Conventions
# -----------
#   * Level set `φ(x,y,z)`: `φ < 0` is solid, `φ > 0` is fluid. `φ` need
#     not be a signed distance; normals and distances are normalized by
#     |∇φ| so any smooth level set works.
#   * Storage arrays have the same shape as a `Field` on the mesh (interior
#     + halos), so geometry and fields share indices.
#   * A face array entry `ax[i,j,k]` is the *west* face of cell `(i,j,k)`,
#     exactly matching the Arakawa-C convention used for `u @ FaceX`.
#   * A face value stores the velocity of the *open* part of the face, so
#     the mass flux through it is `aperture * A_face * u`.

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, interior, totalsize,
               CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields
using ..Fields: Field, interior_view
import ..Fields: location
using ..IBM: SolidMask

export CutCellGeometry, cut_cell_geometry, cut_cell_geometry_from_heightmap,
       solid_mask, fluid_volume, wetted_area, cut_cell_count,
       cutcell_divergence!, cutcell_gradient_x!, cutcell_gradient_y!,
       cutcell_gradient_z!, cutcell_laplacian!,
       cutcell_cell_inner, cutcell_face_inner,
       apply_cutcell_velocity_bc!,
       GhostCellStencil, build_ghost_stencils,
       reconstruct_wall_velocity!, reconstruct_wall_scalar!,
       wall_shear_stress

# =========================================================================
# Coordinate helpers (storage index -> physical coordinate)
# =========================================================================
# Cell (i,j,k) of the storage array spans
#   x in [x0 + (i-Hx-1)dx, x0 + (i-Hx)dx]   etc.
# so its center is at `_xc` and its west face at `_xw`. These agree with
# `Grids.xnodes(mesh, CellCenter())` / `xnodes(mesh, FaceX())` on the
# interior indices.

@inline _xc(g::RegularGrid, i::Int) = g.x0 + (i - g.Hx - oftype(g.dx, 0.5)) * g.dx
@inline _yc(g::RegularGrid, j::Int) = g.y0 + (j - g.Hy - oftype(g.dy, 0.5)) * g.dy
@inline _zc(g::RegularGrid, k::Int) = g.z0 + (k - g.Hz - oftype(g.dz, 0.5)) * g.dz

@inline _xw(g::RegularGrid, i::Int) = g.x0 + (i - g.Hx - 1) * g.dx
@inline _yw(g::RegularGrid, j::Int) = g.y0 + (j - g.Hy - 1) * g.dy
@inline _zw(g::RegularGrid, k::Int) = g.z0 + (k - g.Hz - 1) * g.dz

# Sample-point offset of each staggered location, in units of the cell size
# relative to the cell center.
@inline _loc_shift(::CellCenter) = (0.0,  0.0,  0.0)
@inline _loc_shift(::FaceX)      = (-0.5, 0.0,  0.0)
@inline _loc_shift(::FaceY)      = (0.0, -0.5,  0.0)
@inline _loc_shift(::FaceZ)      = (0.0,  0.0, -0.5)

# Index direction along which a location is staggered (0 = collocated).
@inline _stagger_dir(::CellCenter) = 0
@inline _stagger_dir(::FaceX)      = 1
@inline _stagger_dir(::FaceY)      = 2
@inline _stagger_dir(::FaceZ)      = 3

@inline function _sample_coord(g::RegularGrid{T}, loc::Location, i::Int, j::Int, k::Int) where {T}
    ox, oy, oz = _loc_shift(loc)
    return (_xc(g, i) + T(ox) * g.dx,
            _yc(g, j) + T(oy) * g.dy,
            _zc(g, k) + T(oz) * g.dz)
end

"""
    _normal_from(φ, mesh, i, j, k) -> (nx, ny, nz, |∇φ|)

Unit normal of the level-set field `φ` at index `(i,j,k)`, obtained from
central differences (one-sided at array edges) and normalized. The normal
points **into the fluid** (direction of increasing `φ`). Returns the
gradient magnitude as well, which converts a level-set value into a
wall-normal distance: `d ≈ φ / |∇φ|`.

Degenerate gradients (flat `φ`) fall back to `(0,0,1)` with magnitude 1.
"""
@inline function _normal_from(φ::AbstractArray{T,3}, g::RegularGrid, i::Int, j::Int, k::Int) where {T}
    n1, n2, n3 = size(φ)
    ip = min(i + 1, n1); im = max(i - 1, 1)
    jp = min(j + 1, n2); jm = max(j - 1, 1)
    kp = min(k + 1, n3); km = max(k - 1, 1)
    gx = ip == im ? zero(T) : (φ[ip, j, k] - φ[im, j, k]) / ((ip - im) * g.dx)
    gy = jp == jm ? zero(T) : (φ[i, jp, k] - φ[i, jm, k]) / ((jp - jm) * g.dy)
    gz = kp == km ? zero(T) : (φ[i, j, kp] - φ[i, j, km]) / ((kp - km) * g.dz)
    mag = sqrt(gx * gx + gy * gy + gz * gz)
    if !(mag > 0) || !isfinite(mag)
        return (zero(T), zero(T), one(T), one(T))
    end
    return (gx / mag, gy / mag, gz / mag, mag)
end

# =========================================================================
# Phase 21 — geometry
# =========================================================================

"""
    CutCellGeometry{T,M}

Discrete description of a solid body immersed in a [`RegularGrid`](@ref).
All arrays have the shape of a `Field` on the mesh (interior + halos) so
that geometry and field indices coincide.

| Field          | Meaning                                                        |
| :------------- | :------------------------------------------------------------- |
| `vol_frac`     | wetted volume fraction of each cell, in `[0,1]`                |
| `vol_eff`      | `vol_frac` with the small-cell floor applied (see `vol_min`)    |
| `ax, ay, az`   | open-area fraction of the west / south / bottom face, `[0,1]`   |
| `nx, ny, nz`   | unit surface normal (into the fluid) — zero in full/empty cells |
| `area`         | wetted surface area inside the cell, m² (zero if not cut)       |
| `phi`          | level-set value sampled at the cell centers                    |
| `is_cut`       | `0 < vol_frac < 1`                                             |
| `is_solid`     | `vol_frac == 0`                                                |
| `is_small`     | `0 < vol_frac < vol_min` (needs merging/stabilization)          |
| `merge_target` | linear index of the merge partner of a small cell (`0` if none) |
| `vol_min`      | small-cell threshold                                           |
| `nsub`         | sub-samples per direction used to build the fractions          |

`area` and the *discrete* normal are mutually consistent with the
apertures: `area * n = ((axₑ-ax_w)ΔyΔz, (ay_n-ay_s)ΔxΔz, (az_t-az_b)ΔxΔy)`,
i.e. the discrete Gauss theorem holds cell by cell, which is why the
cut-cell divergence of a uniform flow is exactly zero. The stored `n`
comes from `∇φ` (smoother, and defined even for degenerate aperture
patterns); `area` is the norm of the aperture-balance vector.

Build one with [`cut_cell_geometry`](@ref) or
[`cut_cell_geometry_from_heightmap`](@ref).
"""
struct CutCellGeometry{T<:AbstractFloat, M<:RegularGrid{T}}
    mesh::M
    vol_frac::Array{T,3}
    vol_eff::Array{T,3}
    ax::Array{T,3}
    ay::Array{T,3}
    az::Array{T,3}
    nx::Array{T,3}
    ny::Array{T,3}
    nz::Array{T,3}
    area::Array{T,3}
    phi::Array{T,3}
    is_cut::Array{Bool,3}
    is_solid::Array{Bool,3}
    is_small::Array{Bool,3}
    merge_target::Array{Int,3}
    vol_min::T
    nsub::Int
end

Base.eltype(::CutCellGeometry{T}) where {T} = T

function Base.show(io::IO, geom::CutCellGeometry{T}) where {T}
    ii, jj, kk = interior(geom.mesh)
    ncut = count(view(geom.is_cut, ii, jj, kk))
    nsol = count(view(geom.is_solid, ii, jj, kk))
    print(io, "CutCellGeometry{$T}(mesh=$(geom.mesh), cut=$ncut, solid=$nsol, nsub=$(geom.nsub))")
end

"""
    cut_cell_geometry(mesh, levelset::Function; nsub=4, vol_min=1e-3)

Build a [`CutCellGeometry`](@ref) on `mesh` from a level set
`levelset(x,y,z)` with the convention **`φ < 0` = solid**, `φ > 0` =
fluid. A signed distance function is ideal but any smooth level set works
(distances are normalized by `|∇φ|`).

Volume and area fractions are computed by *sub-cell sampling*: `nsub^3`
equally spaced sub-samples per cell and `nsub^2` per face, counting the
fraction that lies in the fluid. The fractions are therefore quantized in
steps of `nsub^{-3}` (`nsub^{-2}` for apertures) and converge like
`O(1/nsub)` for a smooth surface. `nsub = 4` (the default, 64 samples per
cell) is a good accuracy/cost compromise for LES; increase it for
sharply curved bodies. An exact marching-cubes polyhedron intersection is
a future refinement — nothing downstream depends on how the fractions were
obtained.

Robustness passes applied after sampling (in this order, iterated to a
fixed point):

1. **Aperture/volume consistency** — a face is closed (`aperture = 0`)
   whenever either cell it separates is dry. Independent sampling of
   volumes and areas can otherwise produce a face that leaks mass into a
   cell with no volume.
2. **Isolated-pocket removal** — a wetted cell with no open face cannot
   exchange mass with anything and would leave the pressure matrix
   singular, so it is reclassified as solid.
3. **Small-cell bookkeeping** — cells with `0 < vol_frac < vol_min` are
   flagged in `is_small` and given a merge partner (`merge_target`, the
   most open neighbour); `vol_eff` floors their volume at `vol_min`. See
   [`cutcell_divergence!`](@ref) for how they are treated.

The level set is also sampled at cell centers into `geom.phi`, including
halo cells (evaluated at the extrapolated coordinates) — that array is
what the Phase-22 ghost-cell machinery uses.
"""
function cut_cell_geometry(mesh::RegularGrid{T}, levelset::Function;
                           nsub::Int = 4, vol_min::Real = 1e-3) where {T}
    nsub >= 1 || throw(ArgumentError("nsub must be >= 1, got $nsub"))
    0 <= vol_min < 1 || throw(ArgumentError("vol_min must lie in [0,1), got $vol_min"))
    dims = totalsize(mesh)
    n1, n2, n3 = dims

    φ   = Array{T,3}(undef, dims)
    vol = zeros(T, dims)
    ax  = zeros(T, dims)
    ay  = zeros(T, dims)
    az  = zeros(T, dims)

    invn = one(T) / T(nsub)
    hx = mesh.dx * invn
    hy = mesh.dy * invn
    hz = mesh.dz * invn
    volnorm  = one(T) / T(nsub^3)
    areanorm = one(T) / T(nsub^2)

    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        φ[i, j, k] = T(levelset(_xc(mesh, i), _yc(mesh, j), _zc(mesh, k)))
    end

    # --- wetted volume fractions: nsub^3 sub-samples per cell -------------
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        xb = _xw(mesh, i); yb = _yw(mesh, j); zb = _zw(mesh, k)
        cnt = 0
        for c in 1:nsub
            zs = zb + (c - oftype(hz, 0.5)) * hz
            for b in 1:nsub
                ys = yb + (b - oftype(hy, 0.5)) * hy
                for a in 1:nsub
                    xs = xb + (a - oftype(hx, 0.5)) * hx
                    cnt += levelset(xs, ys, zs) > 0
                end
            end
        end
        vol[i, j, k] = cnt * volnorm
    end

    # --- open-area fractions: nsub^2 sub-samples per face -----------------
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        xf = _xw(mesh, i); yb = _yw(mesh, j); zb = _zw(mesh, k)
        cnt = 0
        for c in 1:nsub
            zs = zb + (c - oftype(hz, 0.5)) * hz
            for b in 1:nsub
                ys = yb + (b - oftype(hy, 0.5)) * hy
                cnt += levelset(xf, ys, zs) > 0
            end
        end
        ax[i, j, k] = cnt * areanorm
    end
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        xb = _xw(mesh, i); yf = _yw(mesh, j); zb = _zw(mesh, k)
        cnt = 0
        for c in 1:nsub
            zs = zb + (c - oftype(hz, 0.5)) * hz
            for a in 1:nsub
                xs = xb + (a - oftype(hx, 0.5)) * hx
                cnt += levelset(xs, yf, zs) > 0
            end
        end
        ay[i, j, k] = cnt * areanorm
    end
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        xb = _xw(mesh, i); yb = _yw(mesh, j); zf = _zw(mesh, k)
        cnt = 0
        for b in 1:nsub
            ys = yb + (b - oftype(hy, 0.5)) * hy
            for a in 1:nsub
                xs = xb + (a - oftype(hx, 0.5)) * hx
                cnt += levelset(xs, ys, zf) > 0
            end
        end
        az[i, j, k] = cnt * areanorm
    end

    return _finalize_geometry(mesh, φ, vol, ax, ay, az, T(vol_min), nsub)
end

"""
    cut_cell_geometry(mesh, sm::SolidMask; vol_min=1e-3)

Staircase geometry from an existing [`SolidMask`](@ref) (Phase-13 IBM
interop): `vol_frac` is 0/1 and apertures are 0/1, so the cut-cell
operators reduce exactly to the Cartesian ones with blocked faces. Useful
for A/B-testing the cut-cell machinery against the old penalization IBM on
the same obstacle.
"""
function cut_cell_geometry(mesh::RegularGrid{T}, sm::SolidMask; vol_min::Real = 1e-3) where {T}
    size(sm.mask) == totalsize(mesh) ||
        throw(ArgumentError("SolidMask shape $(size(sm.mask)) does not match mesh $(totalsize(mesh))"))
    dims = totalsize(mesh)
    n1, n2, n3 = dims
    vol = zeros(T, dims)
    φ   = Array{T,3}(undef, dims)
    h   = min(mesh.dx, mesh.dy, mesh.dz)
    @inbounds for I in eachindex(sm.mask)
        vol[I] = sm.mask[I] ? zero(T) : one(T)
        φ[I]   = sm.mask[I] ? -T(0.5) * h : T(0.5) * h
    end
    ax = zeros(T, dims); ay = zeros(T, dims); az = zeros(T, dims)
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        ax[i, j, k] = (i > 1  && vol[i-1, j, k] > 0 && vol[i, j, k] > 0) ? one(T) : zero(T)
        ay[i, j, k] = (j > 1  && vol[i, j-1, k] > 0 && vol[i, j, k] > 0) ? one(T) : zero(T)
        az[i, j, k] = (k > 1  && vol[i, j, k-1] > 0 && vol[i, j, k] > 0) ? one(T) : zero(T)
    end
    return _finalize_geometry(mesh, φ, vol, ax, ay, az, T(vol_min), 1)
end

"""
    cut_cell_geometry_from_heightmap(mesh, height::AbstractMatrix; nsub=4, vol_min=1e-3)

Convenience constructor for terrain and flat-roofed buildings: builds the
level set `φ(x,y,z) = z - h(x,y)` where `h` is taken piecewise-constant
over each grid column (`height[i,j]` is the solid top of column `(i,j)`).

`height` may be given on the interior (`Nx × Ny`) or on the full storage
array (`Nx+2Hx × Ny+2Hy`); interior-sized input is extended to the halos
by edge replication.

Piecewise-constant columns are the right model for a flat roof: the
vertical cut is resolved exactly (`vol_frac` measures how much of the cell
is above the roof) while the horizontal outline stays on grid lines. For
smooth orography give `cut_cell_geometry` a bilinearly interpolated
level-set closure instead, so the cut-cell normals pick up the slope.
"""
function cut_cell_geometry_from_heightmap(mesh::RegularGrid{T}, height::AbstractMatrix;
                                          nsub::Int = 4, vol_min::Real = 1e-3) where {T}
    dims = totalsize(mesh)
    hfull = Array{T,2}(undef, dims[1], dims[2])
    if size(height) == (dims[1], dims[2])
        hfull .= height
    elseif size(height) == (mesh.Nx, mesh.Ny)
        @inbounds for j in 1:dims[2], i in 1:dims[1]
            ii = clamp(i - mesh.Hx, 1, mesh.Nx)
            jj = clamp(j - mesh.Hy, 1, mesh.Ny)
            hfull[i, j] = height[ii, jj]
        end
    else
        throw(ArgumentError("height must be $(mesh.Nx)×$(mesh.Ny) or " *
                            "$(dims[1])×$(dims[2]), got $(size(height))"))
    end
    # Column lookup: index of the column containing x (clamped to the array).
    idx(x, x0, dx, Hx, n) = clamp(floor(Int, (x - x0) / dx) + Hx + 1, 1, n)
    φ = function (x, y, z)
        i = idx(x, mesh.x0, mesh.dx, mesh.Hx, dims[1])
        j = idx(y, mesh.y0, mesh.dy, mesh.Hy, dims[2])
        return z - hfull[i, j]
    end
    return cut_cell_geometry(mesh, φ; nsub = nsub, vol_min = vol_min)
end

function _finalize_geometry(mesh::RegularGrid{T}, φ, vol, ax, ay, az,
                            vol_min::T, nsub::Int) where {T}
    dims = size(vol)
    n1, n2, n3 = dims

    is_solid = fill(false, dims)
    @inbounds for I in eachindex(vol)
        is_solid[I] = !(vol[I] > 0)
    end

    # The outermost low-side faces separate the array from nothing; close
    # them so no kernel can read outside the array through an open face.
    @inbounds ax[1, :, :] .= zero(T)
    @inbounds ay[:, 1, :] .= zero(T)
    @inbounds az[:, :, 1] .= zero(T)

    # --- consistency sweeps (see `cut_cell_geometry` docstring) ----------
    for _sweep in 1:16
        changed = false
        @inbounds for k in 1:n3, j in 1:n2, i in 2:n1
            if is_solid[i-1, j, k] || is_solid[i, j, k]
                ax[i, j, k] = zero(T)
            end
        end
        @inbounds for k in 1:n3, j in 2:n2, i in 1:n1
            if is_solid[i, j-1, k] || is_solid[i, j, k]
                ay[i, j, k] = zero(T)
            end
        end
        @inbounds for k in 2:n3, j in 1:n2, i in 1:n1
            if is_solid[i, j, k-1] || is_solid[i, j, k]
                az[i, j, k] = zero(T)
            end
        end
        @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
            is_solid[i, j, k] && continue
            open = ax[i, j, k] + ay[i, j, k] + az[i, j, k]
            i < n1 && (open += ax[i+1, j, k])
            j < n2 && (open += ay[i, j+1, k])
            k < n3 && (open += az[i, j, k+1])
            if !(open > 0)
                is_solid[i, j, k] = true
                vol[i, j, k] = zero(T)
                changed = true
            end
        end
        changed || break
    end

    # --- classification, small-cell floor --------------------------------
    is_cut   = fill(false, dims)
    is_small = fill(false, dims)
    vol_eff  = zeros(T, dims)
    @inbounds for I in eachindex(vol)
        v = vol[I]
        if is_solid[I]
            vol_eff[I] = zero(T)
        else
            is_cut[I] = v < one(T)
            if v < vol_min
                is_small[I] = true
                vol_eff[I]  = vol_min
            else
                vol_eff[I] = v
            end
        end
    end

    # --- merge partner for each small cell: most open neighbour ----------
    li = LinearIndices(dims)
    merge_target = zeros(Int, dims)
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        is_small[i, j, k] || continue
        best = 0; bestv = zero(T)
        # (aperture connecting to the neighbour, neighbour index)
        for (ap, i2, j2, k2) in ((i > 1  ? ax[i, j, k]     : zero(T), i-1, j, k),
                                 (i < n1 ? ax[i+1, j, k]   : zero(T), i+1, j, k),
                                 (j > 1  ? ay[i, j, k]     : zero(T), i, j-1, k),
                                 (j < n2 ? ay[i, j+1, k]   : zero(T), i, j+1, k),
                                 (k > 1  ? az[i, j, k]     : zero(T), i, j, k-1),
                                 (k < n3 ? az[i, j, k+1]   : zero(T), i, j, k+1))
            ap > 0 || continue
            (1 <= i2 <= n1 && 1 <= j2 <= n2 && 1 <= k2 <= n3) || continue
            v2 = vol[i2, j2, k2]
            if v2 > bestv
                bestv = v2
                best  = li[i2, j2, k2]
            end
        end
        merge_target[i, j, k] = best
    end

    # --- wetted area (aperture balance) and normal (level-set gradient) ---
    nx = zeros(T, dims); ny = zeros(T, dims); nz = zeros(T, dims)
    area = zeros(T, dims)
    Axy = mesh.dx * mesh.dy
    Axz = mesh.dx * mesh.dz
    Ayz = mesh.dy * mesh.dz
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        is_cut[i, j, k] || continue
        sx = ((i < n1 ? ax[i+1, j, k] : zero(T)) - ax[i, j, k]) * Ayz
        sy = ((j < n2 ? ay[i, j+1, k] : zero(T)) - ay[i, j, k]) * Axz
        sz = ((k < n3 ? az[i, j, k+1] : zero(T)) - az[i, j, k]) * Axy
        area[i, j, k] = sqrt(sx * sx + sy * sy + sz * sz)
        gx, gy, gz, _ = _normal_from(φ, mesh, i, j, k)
        nx[i, j, k] = gx; ny[i, j, k] = gy; nz[i, j, k] = gz
    end

    return CutCellGeometry(mesh, vol, vol_eff, ax, ay, az, nx, ny, nz,
                           area, φ, is_cut, is_solid, is_small, merge_target,
                           vol_min, nsub)
end

"""
    solid_mask(geom) -> SolidMask

A Phase-13 [`SolidMask`](@ref) holding `geom.is_solid`, so cut-cell
geometry can drive the legacy penalization helpers (or any code written
against the mask interface).
"""
solid_mask(geom::CutCellGeometry) = SolidMask(geom.mesh, copy(geom.is_solid))

"""
    fluid_volume(geom)

Total wetted volume of the interior domain, `Σ V_cell · vol_frac` (m³).
"""
function fluid_volume(geom::CutCellGeometry{T}) where {T}
    m = geom.mesh
    V = m.dx * m.dy * m.dz
    ii, jj, kk = interior(m)
    s = zero(T)
    @inbounds for k in kk, j in jj, i in ii
        s += geom.vol_frac[i, j, k]
    end
    return s * V
end

"""
    wetted_area(geom)

Total immersed-surface area inside the interior domain (m²).
"""
function wetted_area(geom::CutCellGeometry{T}) where {T}
    ii, jj, kk = interior(geom.mesh)
    s = zero(T)
    @inbounds for k in kk, j in jj, i in ii
        s += geom.area[i, j, k]
    end
    return s
end

"""
    cut_cell_count(geom) -> (cut, solid, small)

Number of interior cells that are cut, fully solid, and below the
small-cell threshold.
"""
function cut_cell_count(geom::CutCellGeometry)
    ii, jj, kk = interior(geom.mesh)
    return (count(view(geom.is_cut, ii, jj, kk)),
            count(view(geom.is_solid, ii, jj, kk)),
            count(view(geom.is_small, ii, jj, kk)))
end

# =========================================================================
# Phase 21 — conservative cut-cell operators
# =========================================================================

"""
    cutcell_divergence!(div, u, v, w, geom, mesh = geom.mesh; redistribute = true)

Conservative cut-cell divergence

    (∇·u)_ijk = 1/(V vf) [ (aₑuₑ - a_w u_w)ΔyΔz + (a_n v_n - a_s v_s)ΔxΔz
                                                 + (a_t w_t - a_b w_b)ΔxΔy ]

i.e. the aperture-weighted face-flux balance divided by the wetted volume
`V vf`. The wall itself contributes no flux term — no-penetration is built
into the operator rather than imposed on it — so a flow tangential to the
surface is exactly divergence-free.

Solid cells (`vol_frac == 0`) are set to zero.

**Small cells.** Dividing by a vanishing `vol_frac` would destroy the time
step and the conditioning of `D·G`, so two mechanisms act together:

  * `vol_eff = max(vol_frac, vol_min)` floors the volume (this is what the
    operator actually divides by). It keeps `(D, G)` an exact adjoint pair
    with respect to the `vol_eff`-weighted inner product, hence keeps the
    projection exact — see [`cutcell_cell_inner`](@ref).
  * `redistribute = true` (default) additionally *merges* each flagged
    small cell with its most open neighbour (`geom.merge_target`): the two
    cells' fluxes are summed and the cluster's mean divergence is written
    back to both. The move conserves `Σ V·vol_eff·(∇·u)` exactly, so
    global mass conservation is untouched, and it removes the O(1/vf)
    amplification from the small cell. Note that flux redistribution is
    only *adjoint*-consistent on the subspace where the pressure is
    constant across a merged cluster (true cluster merging would make the
    cluster a single unknown — a future refinement that needs the pressure
    solver to carry the cluster map). Pass `redistribute = false` for the
    strict adjoint operator; with sampled geometry small cells are rare,
    since `vol_frac` is quantized in steps of `nsub^{-3}`.
"""
function cutcell_divergence!(div::Field{CellCenter}, u::Field{FaceX},
                             v::Field{FaceY}, w::Field{FaceZ},
                             geom::CutCellGeometry, mesh::RegularGrid = geom.mesh;
                             redistribute::Bool = true)
    ii, jj, kk = interior(mesh)
    idx = 1 / mesh.dx; idy = 1 / mesh.dy; idz = 1 / mesh.dz
    ax = geom.ax; ay = geom.ay; az = geom.az; ve = geom.vol_eff
    ud = u.data; vd = v.data; wd = w.data; dd = div.data
    @inbounds for k in kk, j in jj, i in ii
        vc = ve[i, j, k]
        if !(vc > 0)
            dd[i, j, k] = 0
            continue
        end
        fx = (ax[i+1, j, k] * ud[i+1, j, k] - ax[i, j, k] * ud[i, j, k]) * idx
        fy = (ay[i, j+1, k] * vd[i, j+1, k] - ay[i, j, k] * vd[i, j, k]) * idy
        fz = (az[i, j, k+1] * wd[i, j, k+1] - az[i, j, k] * wd[i, j, k]) * idz
        dd[i, j, k] = (fx + fy + fz) / vc
    end
    redistribute && _redistribute_small_cells!(div, geom, mesh)
    return div
end

# Cluster a small cell with its merge target: both get the cluster-mean
# divergence, which conserves Σ V·vol_eff·div. Applied sequentially, so a
# target shared by several small cells accumulates them one at a time.
function _redistribute_small_cells!(div::Field{CellCenter}, geom::CutCellGeometry{T},
                                    mesh::RegularGrid) where {T}
    any(geom.is_small) || return div
    ii, jj, kk = interior(mesh)
    ve = geom.vol_eff; dd = div.data
    @inbounds for k in kk, j in jj, i in ii
        geom.is_small[i, j, k] || continue
        t = geom.merge_target[i, j, k]
        t == 0 && continue
        vs = ve[i, j, k]; vt = ve[t]
        vtot = vs + vt
        vtot > 0 || continue
        dcl = (vs * dd[i, j, k] + vt * dd[t]) / vtot
        dd[i, j, k] = dcl
        dd[t]       = dcl
    end
    return div
end

"""
    cutcell_gradient_x!(out, p, geom, mesh = geom.mesh)
    cutcell_gradient_y!(out, p, geom, mesh = geom.mesh)
    cutcell_gradient_z!(out, p, geom, mesh = geom.mesh)

Aperture-scaled pressure gradient at the faces,

    (G p)_face = a_face · (p⁺ - p⁻) / Δ ,

which is exactly zero on a fully blocked face (`a = 0`) — the face then
carries neither flux nor pressure force. Together with
[`cutcell_divergence!`](@ref) this forms a negative adjoint pair; see
[`cutcell_cell_inner`](@ref).

Following the convention of the Cartesian kernels in `FiniteVolume.jl`,
face index `i` holds the west face of cell `i`, and only interior indices
are written (the opposite domain-boundary face lives in the halo and is
supplied by the boundary conditions / geometry).
"""
function cutcell_gradient_x!(out::Field{FaceX}, p::Field{CellCenter},
                             geom::CutCellGeometry, mesh::RegularGrid = geom.mesh)
    ii, jj, kk = interior(mesh)
    idx = 1 / mesh.dx
    ax = geom.ax; pd = p.data; od = out.data
    @inbounds for k in kk, j in jj, i in ii
        a = ax[i, j, k]
        od[i, j, k] = a > 0 ? a * (pd[i, j, k] - pd[i-1, j, k]) * idx : zero(eltype(od))
    end
    return out
end

function cutcell_gradient_y!(out::Field{FaceY}, p::Field{CellCenter},
                             geom::CutCellGeometry, mesh::RegularGrid = geom.mesh)
    ii, jj, kk = interior(mesh)
    idy = 1 / mesh.dy
    ay = geom.ay; pd = p.data; od = out.data
    @inbounds for k in kk, j in jj, i in ii
        a = ay[i, j, k]
        od[i, j, k] = a > 0 ? a * (pd[i, j, k] - pd[i, j-1, k]) * idy : zero(eltype(od))
    end
    return out
end

function cutcell_gradient_z!(out::Field{FaceZ}, p::Field{CellCenter},
                             geom::CutCellGeometry, mesh::RegularGrid = geom.mesh)
    ii, jj, kk = interior(mesh)
    idz = 1 / mesh.dz
    az = geom.az; pd = p.data; od = out.data
    @inbounds for k in kk, j in jj, i in ii
        a = az[i, j, k]
        od[i, j, k] = a > 0 ? a * (pd[i, j, k] - pd[i, j, k-1]) * idz : zero(eltype(od))
    end
    return out
end

"""
    cutcell_laplacian!(out, f, geom, mesh = geom.mesh; aperture_power = 2)

The cut-cell Poisson operator — the operator the pressure solve must
invert:

    (L f)_ijk = 1/(V vf) Σ_faces a_face^q A_face (f_nbr - f_ijk) / Δ .

Blocked faces drop out of the sum, which is exactly a homogeneous Neumann
(zero normal derivative) condition *on the immersed surface* — the correct
boundary condition for the projection pressure. Solid cells give zero.
`V·vol_eff·L` is symmetric for either exponent, so `L` is self-adjoint and
negative semi-definite in the `vol_eff`-weighted inner product (see
[`cutcell_cell_inner`](@ref)) and stays solvable by CG / multigrid.

The exponent `q = aperture_power` selects which operator you get:

  * `q = 2` (default) is **exactly** the composition `D·G` of
    [`cutcell_divergence!`](@ref) with [`cutcell_gradient_x!`](@ref) &co.
    One aperture comes from the flux weighting in `D`, the other from the
    aperture scaling of `G`. This is the operator to use in the
    fractional-step projection, because the projection is discretely exact
    only when the Poisson operator is the composition of the *same* `D`
    and `G` that are applied to the velocity: solving `D G φ = D u*` and
    correcting `u = u* - G φ` then gives `D u = 0` to round-off. In
    operator terms this is the aperture-weighted projection
    `u = u* - A ∇φ`, an oblique projection in the `A`-weighted metric.
  * `q = 1` is the classical flux-form cut-cell Laplacian
    `∇·(a ∇f)`, i.e. the second-order approximation of `∇²f` on the wetted
    volume. Use it for diffusion of a scalar (heat, species) across a cut
    geometry, where the physical flux through a partially open face is
    `a·A·κ·∂f/∂n` and must carry a *single* aperture.
"""
function cutcell_laplacian!(out::Field{CellCenter}, f::Field{CellCenter},
                            geom::CutCellGeometry, mesh::RegularGrid = geom.mesh;
                            aperture_power::Int = 2)
    aperture_power in (1, 2) ||
        throw(ArgumentError("aperture_power must be 1 or 2, got $aperture_power"))
    ii, jj, kk = interior(mesh)
    idx2 = 1 / (mesh.dx * mesh.dx)
    idy2 = 1 / (mesh.dy * mesh.dy)
    idz2 = 1 / (mesh.dz * mesh.dz)
    ax = geom.ax; ay = geom.ay; az = geom.az; ve = geom.vol_eff
    fd = f.data; od = out.data
    sq = aperture_power == 2
    @inbounds for k in kk, j in jj, i in ii
        vc = ve[i, j, k]
        if !(vc > 0)
            od[i, j, k] = 0
            continue
        end
        c = fd[i, j, k]
        axp = ax[i+1, j, k]; axm = ax[i, j, k]
        ayp = ay[i, j+1, k]; aym = ay[i, j, k]
        azp = az[i, j, k+1]; azm = az[i, j, k]
        if sq
            axp *= axp; axm *= axm
            ayp *= ayp; aym *= aym
            azp *= azp; azm *= azm
        end
        s  = (axp * (fd[i+1, j, k] - c) + axm * (fd[i-1, j, k] - c)) * idx2
        s += (ayp * (fd[i, j+1, k] - c) + aym * (fd[i, j-1, k] - c)) * idy2
        s += (azp * (fd[i, j, k+1] - c) + azm * (fd[i, j, k-1] - c)) * idz2
        od[i, j, k] = s / vc
    end
    return out
end

"""
    cutcell_cell_inner(a, b, geom, mesh = geom.mesh)

Wetted-volume inner product of two cell-centered fields,
`Σ_interior V·vol_eff·a·b`. This is the inner product in which the
cut-cell operators are adjoint:

    cutcell_cell_inner(p, D·u)  ==  -cutcell_face_inner(G·p, u)

(exactly, to round-off) whenever the velocity vanishes on the
domain-boundary faces or the geometry closes them. That identity is what
makes the fractional-step projection discretely divergence-free on a cut
geometry, and it is verified numerically in the Era II test set.
"""
function cutcell_cell_inner(a::Field{CellCenter}, b::Field{CellCenter},
                            geom::CutCellGeometry{T}, mesh::RegularGrid = geom.mesh) where {T}
    ii, jj, kk = interior(mesh)
    V = mesh.dx * mesh.dy * mesh.dz
    s = zero(T)
    ad = a.data; bd = b.data; ve = geom.vol_eff
    @inbounds for k in kk, j in jj, i in ii
        s += ve[i, j, k] * ad[i, j, k] * bd[i, j, k]
    end
    return s * V
end

"""
    cutcell_face_inner(au, av, aw, bu, bv, bw, geom, mesh = geom.mesh)

Face inner product of two staggered vector fields, `Σ_faces V·a·b`, summed
over the *interior* faces (those separating two interior cells) of all
three directions. Paired with [`cutcell_cell_inner`](@ref) it expresses the
discrete adjointness of the cut-cell divergence and gradient.
"""
function cutcell_face_inner(au::Field{FaceX}, av::Field{FaceY}, aw::Field{FaceZ},
                            bu::Field{FaceX}, bv::Field{FaceY}, bw::Field{FaceZ},
                            geom::CutCellGeometry{T}, mesh::RegularGrid = geom.mesh) where {T}
    ii, jj, kk = interior(mesh)
    V = mesh.dx * mesh.dy * mesh.dz
    s = zero(T)
    @inbounds for k in kk, j in jj, i in (first(ii)+1):last(ii)
        s += au.data[i, j, k] * bu.data[i, j, k]
    end
    @inbounds for k in kk, j in (first(jj)+1):last(jj), i in ii
        s += av.data[i, j, k] * bv.data[i, j, k]
    end
    @inbounds for k in (first(kk)+1):last(kk), j in jj, i in ii
        s += aw.data[i, j, k] * bw.data[i, j, k]
    end
    return s * V
end

"""
    apply_cutcell_velocity_bc!(u, v, w, geom, mesh = geom.mesh)
    apply_cutcell_velocity_bc!(sim, geom)

Enforce no-penetration on the immersed surface exactly.

  * Every face with `aperture == 0` gets **exactly** zero normal velocity.
    That covers all faces of a solid cell and every face the surface
    closes completely, so no flux can cross a dry face.
  * Partially open faces keep their value: by convention a face value is
    the velocity of the *open* part of the face, so the flux it carries is
    `a · A · u` and is already consistent with what
    [`cutcell_divergence!`](@ref) sums. As a safety net the aperture of any
    face whose neighbouring cell is dry is re-zeroed here as well (it
    should already be zero after the geometry's consistency sweep), which
    guarantees the flux into a solid cell is identically zero even if the
    geometry was edited by hand.

Nothing needs to be done *on* the surface: because the wall is not a face
of the control volume, the cut-cell divergence contains no wall-flux term,
so the no-penetration condition `u·n = 0` is satisfied by construction.
The tangential (no-slip / wall-model) condition is the job of the Phase-22
ghost-cell reconstruction — see [`reconstruct_wall_velocity!`](@ref).

Typical use with the fractional-step driver:

```julia
step!(sim, dt; post_predictor! = (s, Δt) -> apply_cutcell_velocity_bc!(s, geom))
```
"""
function apply_cutcell_velocity_bc!(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                    geom::CutCellGeometry, mesh::RegularGrid = geom.mesh)
    ud = u.data; vd = v.data; wd = w.data
    ax = geom.ax; ay = geom.ay; az = geom.az
    @inbounds for I in eachindex(ax)
        ax[I] > 0 || (ud[I] = 0)
        ay[I] > 0 || (vd[I] = 0)
        az[I] > 0 || (wd[I] = 0)
    end
    # flux-consistency safety net (dry neighbour => no flux)
    n1, n2, n3 = size(ax)
    sol = geom.is_solid
    @inbounds for k in 1:n3, j in 1:n2, i in 2:n1
        if sol[i-1, j, k] || sol[i, j, k]
            ax[i, j, k] = 0; ud[i, j, k] = 0
        end
    end
    @inbounds for k in 1:n3, j in 2:n2, i in 1:n1
        if sol[i, j-1, k] || sol[i, j, k]
            ay[i, j, k] = 0; vd[i, j, k] = 0
        end
    end
    @inbounds for k in 2:n3, j in 1:n2, i in 1:n1
        if sol[i, j, k-1] || sol[i, j, k]
            az[i, j, k] = 0; wd[i, j, k] = 0
        end
    end
    return u, v, w
end

function apply_cutcell_velocity_bc!(sim, geom::CutCellGeometry)
    apply_cutcell_velocity_bc!(sim.u, sim.v, sim.w, geom, geom.mesh)
    return sim
end

# =========================================================================
# Phase 22 — ghost-cell wall reconstruction
# =========================================================================

"""
    GhostCellStencil{L,T}

Precomputed wall reconstruction data for one sample point of a field at
[`Location`](@ref) `L`. Built by [`build_ghost_stencils`](@ref).

| Field          | Meaning                                                            |
| :------------- | :----------------------------------------------------------------- |
| `i, j, k`      | storage index of the sample point being reconstructed              |
| `dn`           | signed wall-normal distance of the sample point (`< 0` ⇒ in solid)  |
| `probe`        | distance from the wall foot point to the image point (`> 0`)        |
| `n`            | unit wall normal, pointing into the fluid                          |
| `xw`           | foot point on the *true* surface                                   |
| `ximg`         | image point out in the fluid                                       |
| `donors`       | linear indices of the (≤ 8) interpolation donors, `0` = unused      |
| `weights`      | interpolation weights, normalized to sum to 1                       |
| `exact_linear` | all eight tri-linear donors are fluid ⇒ interpolation exact for linear fields |
| `ghost`        | the sample point lies in the solid, so its value must be reconstructed |
| `valid`        | at least one usable donor was found                                 |

The reconstruction line is `(xw + dn·n) → xw → (xw + probe·n)`; the
one-dimensional profile along `n` is what carries the boundary condition,
so the condition is imposed **on the surface**, not on a Cartesian face.
"""
struct GhostCellStencil{L<:Location, T<:AbstractFloat}
    i::Int
    j::Int
    k::Int
    dn::T
    probe::T
    n::NTuple{3,T}
    xw::NTuple{3,T}
    ximg::NTuple{3,T}
    donors::NTuple{8,Int}
    weights::NTuple{8,T}
    exact_linear::Bool
    ghost::Bool
    valid::Bool
end

"""
    location(stencil) -> Location

The staggered [`Location`](@ref) whose sample grid `stencil` was built for.
"""
location(::GhostCellStencil{L}) where {L} = L()

# --- sample-grid level set and fluid mask for a given location -----------
# The level set at a face sample point is the average of the two adjacent
# cell values (second-order for a smooth surface). A sample is "fluid" if
# it sits on the fluid side *and* its own control volume / face is open.

function _sample_fields(geom::CutCellGeometry{T}, ::CellCenter) where {T}
    sφ = geom.phi
    fl = fill(false, size(sφ))
    @inbounds for I in eachindex(sφ)
        fl[I] = sφ[I] > 0 && geom.vol_frac[I] > 0
    end
    return sφ, fl
end

function _sample_fields(geom::CutCellGeometry{T}, ::FaceX) where {T}
    φ = geom.phi; dims = size(φ)
    sφ = similar(φ); fl = fill(false, dims)
    @inbounds for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        s = i == 1 ? φ[i, j, k] : (φ[i-1, j, k] + φ[i, j, k]) / 2
        sφ[i, j, k] = s
        fl[i, j, k] = s > 0 && geom.ax[i, j, k] > 0
    end
    return sφ, fl
end

function _sample_fields(geom::CutCellGeometry{T}, ::FaceY) where {T}
    φ = geom.phi; dims = size(φ)
    sφ = similar(φ); fl = fill(false, dims)
    @inbounds for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        s = j == 1 ? φ[i, j, k] : (φ[i, j-1, k] + φ[i, j, k]) / 2
        sφ[i, j, k] = s
        fl[i, j, k] = s > 0 && geom.ay[i, j, k] > 0
    end
    return sφ, fl
end

function _sample_fields(geom::CutCellGeometry{T}, ::FaceZ) where {T}
    φ = geom.phi; dims = size(φ)
    sφ = similar(φ); fl = fill(false, dims)
    @inbounds for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        s = k == 1 ? φ[i, j, k] : (φ[i, j, k-1] + φ[i, j, k]) / 2
        sφ[i, j, k] = s
        fl[i, j, k] = s > 0 && geom.az[i, j, k] > 0
    end
    return sφ, fl
end

# Does the sample point (i,j,k) belong to a cut cell?
@inline _host_cut(geom::CutCellGeometry, ::CellCenter, i, j, k) = geom.is_cut[i, j, k]
@inline _host_cut(geom::CutCellGeometry, ::FaceX, i, j, k) =
    geom.is_cut[i, j, k] || (i > 1 && geom.is_cut[i-1, j, k])
@inline _host_cut(geom::CutCellGeometry, ::FaceY, i, j, k) =
    geom.is_cut[i, j, k] || (j > 1 && geom.is_cut[i, j-1, k])
@inline _host_cut(geom::CutCellGeometry, ::FaceZ, i, j, k) =
    geom.is_cut[i, j, k] || (k > 1 && geom.is_cut[i, j, k-1])

const _NBR6 = ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))

"""
    build_ghost_stencils(geom, mesh = geom.mesh, loc = CellCenter();
                         probe_factor = 1.0, search_radius = 2)

Precompute a [`GhostCellStencil`](@ref) for every sample point of a field
at `loc` that the immersed surface touches, namely

  * points inside the solid (or behind a closed face) that have at least
    one fluid neighbour — the classic *ghost points*: they appear in fluid
    stencils, so they must carry the wall condition; and
  * points hosted by a cut cell, even when they lie on the fluid side —
    these are not overwritten by the reconstruction routines but give
    [`wall_shear_stress`](@ref) a probe on every piece of wetted surface.

Points buried deep in the solid get no stencil: no fluid stencil reads
them, and extrapolating that far from the wall would be meaningless.

For each point the routine computes

  * the unit normal from `∇φ` at the sample point, pointing into the fluid;
  * the signed wall distance `dn = φ/|∇φ|`, and the foot point
    `xw = x - dn·n` on the true surface;
  * the image point `ximg = xw + probe·n` with
    `probe = max(|dn|, probe_factor·hₙ)`, `hₙ = |(nₓΔx, n_yΔy, n_zΔz)|`.
    With `probe = |dn|` this is the textbook mirror point,
    `ximg = x + 2|dn|·n`, and the reconstruction reduces to
    `f_ghost = 2·f_wall - f_image`. The floor at one grid spacing keeps the
    image far enough from the surface that its interpolation donors are all
    fluid, which is what preserves the second-order (linear-exact)
    behaviour for cut cells whose center is only just inside the solid;
  * tri-linear interpolation weights at `ximg` on the `loc` sample grid.
    Donors that are not fluid are dropped and the remaining weights
    renormalized (this loses linear exactness, flagged by
    `exact_linear = false`); if that leaves nothing, the routine falls back
    to inverse-distance weighting over the nearest fluid samples inside a
    `(2·search_radius+1)³` box.
"""
function build_ghost_stencils(geom::CutCellGeometry{T}, mesh::RegularGrid{T} = geom.mesh,
                              loc::Location = CellCenter();
                              probe_factor::Real = 1.0,
                              search_radius::Int = 2) where {T}
    sφ, fl = _sample_fields(geom, loc)
    dims = size(sφ)
    ii, jj, kk = interior(mesh)
    rng = Any[ii, jj, kk]
    d = _stagger_dir(loc)
    if d > 0
        r = rng[d]
        rng[d] = first(r):min(last(r) + 1, dims[d])
    end
    pf = T(probe_factor)
    out = GhostCellStencil{typeof(loc), T}[]
    @inbounds for k in rng[3], j in rng[2], i in rng[1]
        isfluid = fl[i, j, k]
        if isfluid
            _host_cut(geom, loc, i, j, k) || continue
        else
            touches = false
            for (a, b, c) in _NBR6
                i2 = i + a; j2 = j + b; k2 = k + c
                (1 <= i2 <= dims[1] && 1 <= j2 <= dims[2] && 1 <= k2 <= dims[3]) || continue
                if fl[i2, j2, k2]
                    touches = true
                    break
                end
            end
            touches || _host_cut(geom, loc, i, j, k) || continue
        end

        nx, ny, nz, mag = _normal_from(sφ, mesh, i, j, k)
        dn = sφ[i, j, k] / mag
        X  = _sample_coord(mesh, loc, i, j, k)
        hn = sqrt((nx * mesh.dx)^2 + (ny * mesh.dy)^2 + (nz * mesh.dz)^2)
        probe = max(abs(dn), pf * hn)
        probe > 0 || (probe = hn)
        xw   = (X[1] - dn * nx, X[2] - dn * ny, X[3] - dn * nz)
        ximg = (xw[1] + probe * nx, xw[2] + probe * ny, xw[3] + probe * nz)

        donors, weights, exact, valid =
            _interp_weights(fl, mesh, loc, ximg, search_radius)

        push!(out, GhostCellStencil{typeof(loc), T}(i, j, k, dn, probe,
                                                   (nx, ny, nz), xw, ximg,
                                                   donors, weights, exact,
                                                   !isfluid, valid))
    end
    return out
end

# Tri-linear weights at `X` on the sample grid of `loc`, restricted to
# fluid donors, with an inverse-distance fallback.
function _interp_weights(fl::Array{Bool,3}, mesh::RegularGrid{T}, loc::Location,
                         X::NTuple{3,T}, search_radius::Int) where {T}
    dims = size(fl)
    ox, oy, oz = _loc_shift(loc)
    ξ = (X[1] - mesh.x0) / mesh.dx + mesh.Hx + T(0.5) - T(ox)
    η = (X[2] - mesh.y0) / mesh.dy + mesh.Hy + T(0.5) - T(oy)
    ζ = (X[3] - mesh.z0) / mesh.dz + mesh.Hz + T(0.5) - T(oz)
    i0 = clamp(floor(Int, ξ), 1, max(dims[1] - 1, 1))
    j0 = clamp(floor(Int, η), 1, max(dims[2] - 1, 1))
    k0 = clamp(floor(Int, ζ), 1, max(dims[3] - 1, 1))
    tx = clamp(ξ - i0, zero(T), one(T))
    ty = clamp(η - j0, zero(T), one(T))
    tz = clamp(ζ - k0, zero(T), one(T))

    li = LinearIndices(dims)
    dn_ = zeros(Int, 8)
    wt_ = zeros(T, 8)
    wsum = zero(T)
    exact = true
    m = 0
    @inbounds for c in 0:1, b in 0:1, a in 0:1
        m += 1
        w = (a == 0 ? one(T) - tx : tx) *
            (b == 0 ? one(T) - ty : ty) *
            (c == 0 ? one(T) - tz : tz)
        i = min(i0 + a, dims[1]); j = min(j0 + b, dims[2]); k = min(k0 + c, dims[3])
        if fl[i, j, k]
            dn_[m] = li[i, j, k]
            wt_[m] = w
            wsum  += w
        elseif w > 0
            exact = false
        end
    end

    if wsum > T(1e-12)
        @inbounds for m in 1:8
            wt_[m] /= wsum
        end
        return (Tuple(dn_), Tuple(wt_), exact, true)
    end

    # --- inverse-distance fallback over the nearest fluid samples --------
    fill!(dn_, 0); fill!(wt_, zero(T))
    best_d = fill(T(Inf), 8)
    best_i = zeros(Int, 8)
    nfound = 0
    @inbounds for c in -search_radius:search_radius,
                  b in -search_radius:search_radius,
                  a in -search_radius:search_radius
        i = i0 + a; j = j0 + b; k = k0 + c
        (1 <= i <= dims[1] && 1 <= j <= dims[2] && 1 <= k <= dims[3]) || continue
        fl[i, j, k] || continue
        Y = _sample_coord(mesh, loc, i, j, k)
        d2 = (Y[1] - X[1])^2 + (Y[2] - X[2])^2 + (Y[3] - X[3])^2
        # insertion into the 8 nearest
        pos = 0
        for m in 1:8
            if d2 < best_d[m]
                pos = m
                break
            end
        end
        pos == 0 && continue
        for m in 8:-1:(pos+1)
            best_d[m] = best_d[m-1]; best_i[m] = best_i[m-1]
        end
        best_d[pos] = d2; best_i[pos] = li[i, j, k]
        nfound = min(nfound + 1, 8)
    end
    if nfound == 0
        return (Tuple(dn_), Tuple(wt_), false, false)
    end
    tiny = T(1e-30)
    wsum = zero(T)
    @inbounds for m in 1:8
        best_i[m] == 0 && continue
        w = one(T) / (best_d[m] + tiny)
        dn_[m] = best_i[m]; wt_[m] = w; wsum += w
    end
    @inbounds for m in 1:8
        wt_[m] /= wsum
    end
    return (Tuple(dn_), Tuple(wt_), false, true)
end

# Value of `data` at the image point of `s`.
@inline function _image_value(data::AbstractArray{T,3}, s::GhostCellStencil{L,T}) where {L,T}
    acc = zero(T)
    @inbounds for m in 1:8
        d = s.donors[m]
        d == 0 && continue
        acc += s.weights[m] * data[d]
    end
    return acc
end

@inline _wall_at(v::Real, s::GhostCellStencil) = v
@inline _wall_at(f, s::GhostCellStencil) = f(s.xw[1], s.xw[2], s.xw[3])

"""
    reconstruct_wall_velocity!(field, stencils, geom; wall_value = 0.0)

Second-order ghost-cell reconstruction of a Dirichlet condition on the
**true surface**. For every ghost sample point (one inside the solid) the
value is set from the linear profile along the wall normal through the
wall value and the image value,

    f_ghost = f_wall + (dn / probe) · (f_image - f_wall) ,

which for the mirror image (`probe = |dn|`, `dn < 0`) is exactly the
familiar `f_ghost = 2·f_wall - f_image`. The scheme is exact whenever the
field varies linearly along the normal, so the imposed wall value is
second-order accurate.

`wall_value` may be a number or a function `(x,y,z) -> value` evaluated at
the foot point on the surface — the latter is how a moving wall or a
prescribed surface velocity distribution is imposed. For no-slip pass
`0.0` (the default).

Apply it to each velocity component with stencils built at that
component's location:

```julia
su = build_ghost_stencils(geom, mesh, FaceX())
reconstruct_wall_velocity!(sim.u, su, geom)
```

Sample points that lie on the fluid side of the surface are left
untouched, and so are points for which no usable donor exists (isolated
pockets) — those keep whatever value they had.
"""
function reconstruct_wall_velocity!(field::Field{L}, stencils::AbstractVector{<:GhostCellStencil{L}},
                                    geom::CutCellGeometry; wall_value = 0.0) where {L<:Location}
    return _reconstruct_dirichlet!(field, stencils, geom, wall_value)
end

function _reconstruct_dirichlet!(field::Field{L,T}, stencils::AbstractVector{<:GhostCellStencil{L}},
                                 geom::CutCellGeometry, wall_value) where {L<:Location, T}
    _check_shapes(field, geom)
    fd = field.data
    @inbounds for s in stencils
        (s.ghost && s.valid) || continue
        wv   = T(_wall_at(wall_value, s))
        fimg = _image_value(fd, s)
        fd[s.i, s.j, s.k] = wv + (s.dn / s.probe) * (fimg - wv)
    end
    return field
end

"""
    reconstruct_wall_scalar!(field, stencils, geom; wall_value = nothing,
                             wall_flux = nothing, κ = 1)

Ghost-cell reconstruction for θ and tracers. Exactly one of the two
boundary conditions must be given:

  * `wall_value` (number or `(x,y,z) -> value`) — Dirichlet, identical to
    [`reconstruct_wall_velocity!`](@ref).
  * `wall_flux` (number or function) — Neumann. The imposed condition is
    `κ ∂φ/∂n |_wall = wall_flux` with `n` pointing **into the fluid**, so a
    positive `wall_flux` means the scalar increases away from the surface
    (e.g. a surface heat flux warming the air above). The ghost value
    follows the same normal line,

        φ_ghost = φ_image + (dn - probe) · wall_flux / κ ,

    which for the mirror image is `φ_ghost = φ_image - 2|dn|·flux/κ`;
    written with the ghost-to-wall distance it is the familiar
    `φ_ghost = φ_image + flux·Δn/κ`.

`κ` is the diffusivity that makes `wall_flux` a flux; pass `κ = 1` (the
default) to prescribe the normal derivative directly.
"""
function reconstruct_wall_scalar!(field::Field{L,T}, stencils::AbstractVector{<:GhostCellStencil{L}},
                                  geom::CutCellGeometry;
                                  wall_value = nothing, wall_flux = nothing,
                                  κ::Real = 1) where {L<:Location, T}
    if wall_value !== nothing && wall_flux !== nothing
        throw(ArgumentError("give either wall_value (Dirichlet) or wall_flux (Neumann), not both"))
    elseif wall_value !== nothing
        return _reconstruct_dirichlet!(field, stencils, geom, wall_value)
    elseif wall_flux !== nothing
        κ != 0 || throw(ArgumentError("κ must be nonzero for a flux boundary condition"))
        _check_shapes(field, geom)
        fd = field.data
        invκ = one(T) / T(κ)
        @inbounds for s in stencils
            (s.ghost && s.valid) || continue
            g    = T(_wall_at(wall_flux, s)) * invκ
            fimg = _image_value(fd, s)
            fd[s.i, s.j, s.k] = fimg + (s.dn - s.probe) * g
        end
        return field
    else
        throw(ArgumentError("one of wall_value or wall_flux must be given"))
    end
end

function _check_shapes(field::Field, geom::CutCellGeometry)
    size(field.data) == size(geom.vol_frac) ||
        throw(ArgumentError("field shape $(size(field.data)) does not match geometry " *
                            "$(size(geom.vol_frac))"))
    return nothing
end

"""
    wall_shear_stress(u, v, w, stencils, geom, ν) -> Vector

Wall shear stress `τ_w = ν |∂u_t/∂n|` (m²/s²) at every stencil point,
evaluated **on the true normal** rather than on a Cartesian face.

For each stencil the cell-centered velocity is interpolated to the image
point, its wall-normal component removed, and the remaining tangential
speed divided by the probe distance:

    τ_w = ν · |u_t(ximg)| / probe .

The returned vector is indexed like `stencils`, so it pairs with
`geom.area[s.i,s.j,s.k]` (the wetted area of that cut cell) to integrate a
drag force, and with `stencils[m].n` to give the stress its direction.

`stencils` must be built at `CellCenter()`. The cell-centered velocity is
obtained by aperture-weighted averaging of the two bounding faces, which
is the consistent reconstruction on a cut cell (a blocked face contributes
neither value nor weight).

This is the quantity a wall model consumes: replace the linear
`ν·u_t/probe` law with a log-law or Moeng-type relation for `u_τ` at the
same probe point and feed the result back through
[`reconstruct_wall_velocity!`](@ref) as an effective wall value.
"""
function wall_shear_stress(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                           stencils::AbstractVector{<:GhostCellStencil{CellCenter}},
                           geom::CutCellGeometry{T}, ν::Real) where {T}
    mesh = geom.mesh
    uc, vc, wc = _center_velocity(u, v, w, geom, mesh)
    νT = T(ν)
    τ = Vector{T}(undef, length(stencils))
    @inbounds for (m, s) in enumerate(stencils)
        if !s.valid
            τ[m] = zero(T)
            continue
        end
        ui = _image_value(uc, s)
        vi = _image_value(vc, s)
        wi = _image_value(wc, s)
        un = ui * s.n[1] + vi * s.n[2] + wi * s.n[3]
        t2 = ui * ui + vi * vi + wi * wi - un * un
        ut = sqrt(max(t2, zero(T)))
        τ[m] = νT * ut / s.probe
    end
    return τ
end

# Aperture-weighted face-to-center velocity reconstruction.
function _center_velocity(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                          geom::CutCellGeometry{T}, mesh::RegularGrid) where {T}
    dims = totalsize(mesh)
    n1, n2, n3 = dims
    uc = zeros(T, dims); vc = zeros(T, dims); wc = zeros(T, dims)
    ax = geom.ax; ay = geom.ay; az = geom.az
    ud = u.data; vd = v.data; wd = w.data
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        a1 = ax[i, j, k]; a2 = i < n1 ? ax[i+1, j, k] : zero(T)
        s  = a1 + a2
        uc[i, j, k] = s > 0 ? (a1 * ud[i, j, k] + a2 * (i < n1 ? ud[i+1, j, k] : zero(T))) / s : zero(T)
        b1 = ay[i, j, k]; b2 = j < n2 ? ay[i, j+1, k] : zero(T)
        s  = b1 + b2
        vc[i, j, k] = s > 0 ? (b1 * vd[i, j, k] + b2 * (j < n2 ? vd[i, j+1, k] : zero(T))) / s : zero(T)
        c1 = az[i, j, k]; c2 = k < n3 ? az[i, j, k+1] : zero(T)
        s  = c1 + c2
        wc[i, j, k] = s > 0 ? (c1 * wd[i, j, k] + c2 * (k < n3 ? wd[i, j, k+1] : zero(T))) / s : zero(T)
    end
    return uc, vc, wc
end

end # module
