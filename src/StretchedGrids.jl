module StretchedGrids

# =========================================================================
# Phase 24 — Stretched and terrain-following vertical coordinates.
# =========================================================================
#
# Era I shipped a single mesh type, `RegularGrid`: uniform in all three
# directions. That is fine for an idealised periodic box but useless for
# an atmospheric LES, where the interesting physics lives in the first
# few tens of metres above the surface and the domain has to reach a
# kilometre or more. This module adds the two vertical coordinates that
# every production LES/NWP code has:
#
#   1. `StretchedGrid`   — Cartesian in x/y, *arbitrary* non-uniform in z.
#   2. `TerrainFollowingGrid` — a Gal-Chen/Somerville σ-coordinate that
#      maps a flat computational slab onto orography `h(x,y)`.
#
# Both are `AbstractMesh` subtypes, so `Field`, `fill_halos!` and the
# `Operators` façade work on them unchanged. The numerics are retrofitted
# by *adding methods* to the private kernels in `Operators.FiniteVolume`
# (see the `import` block below) — `Operators.∂z(out, f)` forwards to
# `_∂z!(out, f, mesh(f))`, so a field carrying a `StretchedGrid` picks up
# the metric-correct stencil purely by multiple dispatch. No call site in
# the rest of Aeolus changes.
#
# Vertical index convention (identical to `RegularGrid`):
#
#     storage index k ∈ 1 : Nz + 2Hz,  interior k ∈ Hz+1 : Hz+Nz
#     zf[k]  = *bottom* face of cell k          (length Nz + 2Hz + 1)
#     zc[k]  = centre of cell k  = ½(zf[k] + zf[k+1])
#     dzc[k] = zf[k+1] - zf[k]                  (cell thickness, at centres)
#     dzf[k] = zc[k]   - zc[k-1]                (centre spacing, at faces)
#
# `dzc` is the metric for quantities differenced *across* a cell
# (face→centre derivatives, divergence); `dzf` is the metric for
# quantities differenced *between* centres (centre→face gradients). Using
# the wrong one is the classic first-order-accuracy bug in stretched-grid
# codes, so the two are stored separately and named after the location
# they live at.
#
# Halo levels are filled by linear extrapolation of the face sequence,
#     zf[k] = 2 zf[k+1] - zf[k+2]   (below)
#     zf[k] = 2 zf[k-1] - zf[k-2]   (above)
# i.e. the ghost cells continue with the thickness of the nearest
# interior cell. This keeps `dzc`/`dzf` strictly positive and makes the
# boundary stencils well defined without special-casing them.

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, Location, CellCenter, FaceX, FaceY, FaceZ
using ..Fields
using ..Fields: Field, mesh

import ..Grids: totalsize, interior, xnodes, ynodes, znodes

# The FV kernels are a submodule of Operators (`Operators.FiniteVolume`).
# We import the *functions* so that the methods defined below are added to
# the same generic functions the Operators façade already forwards to.
import ..Operators.FiniteVolume: _∂x!, _∂y!, _∂z!, _laplacian!, _divergence!,
                                 _interp_to_face!, _interp_to_center!,
                                 _add_surface_flux!, _add_sponge!, _add_buoyancy!
using ..Operators.FiniteVolume: sponge_profile, in_damping, damping_rate

export StretchedGrid, TerrainFollowingGrid,
       terrain_following_grid,
       grid_stretching_ratio,
       zfaces, zcenters, cell_thickness, center_spacing,
       metric_jacobian, terrain_height, physical_heights,
       set_physical!,
       hydrostatic_balance_error

# =========================================================================
# StretchedGrid
# =========================================================================

"""
    StretchedGrid{T} <: AbstractMesh{T}

Finite-volume mesh that is uniform Cartesian in `x` and `y` and carries an
**arbitrary non-uniform** vertical coordinate. Storage layout, halo
convention and interior index ranges are byte-compatible with
[`RegularGrid`](@ref) — only the vertical metrics differ.

# Vertical metric arrays (all include halo levels)

| field | length | lives at | meaning |
|:------|:-------|:---------|:--------|
| `zf`  | `Nz+2Hz+1` | faces   | bottom face of cell `k` |
| `zc`  | `Nz+2Hz`   | centres | `½(zf[k]+zf[k+1])` |
| `dzc` | `Nz+2Hz`   | centres | `zf[k+1]-zf[k]` — cell thickness |
| `dzf` | `Nz+2Hz+1` | faces   | `zc[k]-zc[k-1]` — centre spacing |

`dzc` is the control-volume height (use it for face→centre differences and
for the divergence); `dzf` is the distance between the two centres a face
separates (use it for centre→face gradients). See
[`grid_stretching_ratio`](@ref) for the quality metric.

# Construction

    StretchedGrid(T=Float64; size, extent, halo=(2,2,2), origin=(0,0,0),
                  stretching=:uniform)

`stretching` accepts

  * `:uniform`             — reproduces `RegularGrid` bit-for-bit.
  * `(:geometric, r)`      — `dz[k+1] = r·dz[k]`, rescaled to fill `Lz`.
  * `(:tanh, β)`           — tanh clustering at the **bottom** wall.
  * `(:tanh_both, β)`      — tanh clustering at both walls (channel LES).
  * `(:custom, zf_vector)` — user-supplied interior faces, length `Nz+1`.

See also [`TerrainFollowingGrid`](@ref), [`hydrostatic_balance_error`](@ref).
"""
struct StretchedGrid{T<:AbstractFloat} <: AbstractMesh{T}
    Nx::Int; Ny::Int; Nz::Int
    Hx::Int; Hy::Int; Hz::Int
    Lx::T;   Ly::T;   Lz::T
    x0::T;   y0::T;   z0::T
    dx::T;   dy::T
    zf::Vector{T}
    zc::Vector{T}
    dzc::Vector{T}
    dzf::Vector{T}
    stretching::Symbol
end

# --- vertical face generators (interior faces only, length Nz+1) ----------

"""
    _faces_uniform(T, Nz, z0, Lz)

Uniform face distribution. Computed from the *exact* affine formula
`z0 + (k-1)·dz` rather than by accumulation so that a `StretchedGrid` with
`stretching=:uniform` reproduces `RegularGrid` to the last bit.
"""
function _faces_uniform(::Type{T}, Nz::Int, z0::T, Lz::T) where {T}
    dz = Lz / Nz
    return T[z0 + (k - 1) * dz for k in 1:Nz+1]
end

"""
    _faces_geometric(T, Nz, z0, Lz, r)

Geometric (constant-ratio) stretching: cell `k+1` is `r` times thicker
than cell `k`. The first thickness follows from `Σ dz = Lz`:

```math
dz_1 = L_z \\frac{r - 1}{r^{N_z} - 1}, \\qquad dz_k = dz_1 r^{k-1}.
```

`r = 1` degenerates to the uniform distribution. `r < 1` coarsens upward.
"""
function _faces_geometric(::Type{T}, Nz::Int, z0::T, Lz::T, r::Real) where {T}
    rr = T(r)
    rr > 0 || throw(ArgumentError("geometric stretching ratio must be > 0, got $rr"))
    isapprox(rr, one(T); atol = 1e-12) && return _faces_uniform(T, Nz, z0, Lz)
    dz1 = Lz * (rr - 1) / (rr^Nz - 1)
    zf = Vector{T}(undef, Nz + 1)
    zf[1] = z0
    dz = dz1
    for k in 1:Nz
        zf[k+1] = zf[k] + dz
        dz *= rr
    end
    zf[Nz+1] = z0 + Lz          # kill accumulated round-off
    return zf
end

"""
    _faces_tanh(T, Nz, z0, Lz, β)

Hyperbolic-tangent clustering at the **bottom** wall — the standard
channel-LES stretching with one resolved wall:

```math
\\xi_k = \\frac{k-1}{N_z}, \\qquad
z_k = z_0 + L_z\\left(1 - \\frac{\\tanh\\!\\big(\\beta(1-\\xi_k)\\big)}{\\tanh\\beta}\\right).
```

Note the reflection `ξ → 1-ξ`. The unreflected form `L·tanh(βξ)/tanh(β)`
is the same one-parameter family but clusters at the **top** of the
domain (its Jacobian `dz/dξ ∝ sech²(βξ)` is *largest* at `ξ=0`), which is
the opposite of what a surface-layer LES wants. `β → 0` recovers the
uniform grid; `β ≈ 1.5–3` is the useful range for LES (larger `β` pushes
[`grid_stretching_ratio`](@ref) past the accuracy limit).
"""
function _faces_tanh(::Type{T}, Nz::Int, z0::T, Lz::T, β::Real) where {T}
    b = T(β)
    b > 0 || throw(ArgumentError("tanh stretching β must be > 0, got $b"))
    tb = tanh(b)
    zf = Vector{T}(undef, Nz + 1)
    for k in 1:Nz+1
        ξ = T(k - 1) / Nz
        zf[k] = z0 + Lz * (1 - tanh(b * (1 - ξ)) / tb)
    end
    zf[1]      = z0
    zf[Nz+1]   = z0 + Lz
    return zf
end

"""
    _faces_tanh_both(T, Nz, z0, Lz, β)

Symmetric tanh clustering at **both** walls (closed channel):

```math
z_k = z_0 + \\frac{L_z}{2}\\left(1 + \\frac{\\tanh\\!\\big(\\beta(2\\xi_k-1)\\big)}{\\tanh\\beta}\\right).
```
"""
function _faces_tanh_both(::Type{T}, Nz::Int, z0::T, Lz::T, β::Real) where {T}
    b = T(β)
    b > 0 || throw(ArgumentError("tanh stretching β must be > 0, got $b"))
    tb = tanh(b)
    zf = Vector{T}(undef, Nz + 1)
    for k in 1:Nz+1
        ξ = T(k - 1) / Nz
        zf[k] = z0 + Lz / 2 * (1 + tanh(b * (2ξ - 1)) / tb)
    end
    zf[1]    = z0
    zf[Nz+1] = z0 + Lz
    return zf
end

function _interior_faces(::Type{T}, Nz, z0, Lz, stretching) where {T}
    spec = stretching isa Symbol ? (stretching,) : Tuple(stretching)
    kind = spec[1]
    if kind === :uniform
        return _faces_uniform(T, Nz, z0, Lz), :uniform
    elseif kind === :geometric
        length(spec) == 2 || throw(ArgumentError("(:geometric, ratio) expects one parameter"))
        # a unit ratio *is* the uniform grid — report it as such so that the
        # exact uniform metrics below are used
        isapprox(T(spec[2]), one(T); atol = 1e-12) &&
            return _faces_uniform(T, Nz, z0, Lz), :uniform
        return _faces_geometric(T, Nz, z0, Lz, spec[2]), :geometric
    elseif kind === :tanh
        length(spec) == 2 || throw(ArgumentError("(:tanh, β) expects one parameter"))
        return _faces_tanh(T, Nz, z0, Lz, spec[2]), :tanh
    elseif kind === :tanh_both
        length(spec) == 2 || throw(ArgumentError("(:tanh_both, β) expects one parameter"))
        return _faces_tanh_both(T, Nz, z0, Lz, spec[2]), :tanh_both
    elseif kind === :custom
        length(spec) == 2 || throw(ArgumentError("(:custom, zf) expects a face vector"))
        zf = collect(T, spec[2])
        length(zf) == Nz + 1 ||
            throw(ArgumentError("custom face vector must have length Nz+1 = $(Nz+1), got $(length(zf))"))
        all(diff(zf) .> 0) ||
            throw(ArgumentError("custom face vector must be strictly increasing"))
        tol = 1e-8 * max(one(T), abs(Lz))
        isapprox(zf[1], z0; atol = tol) ||
            throw(ArgumentError("custom zf[1] = $(zf[1]) must equal origin z0 = $z0"))
        isapprox(zf[end], z0 + Lz; atol = tol) ||
            throw(ArgumentError("custom zf[end] = $(zf[end]) must equal z0+Lz = $(z0+Lz); " *
                                "set `extent[3]` to match your face vector"))
        zf[1] = z0; zf[end] = z0 + Lz
        return zf, :custom
    else
        throw(ArgumentError("unknown stretching $kind; expected :uniform, " *
                            "(:geometric, r), (:tanh, β), (:tanh_both, β) or (:custom, zf)"))
    end
end

"""
    _extend_faces(zf_interior, Nz, Hz)

Pad the interior face vector with `Hz` ghost levels on each side by linear
extrapolation of the face sequence (`zf[k] = 2zf[k+1] - zf[k+2]` below,
mirror image above). Ghost cells therefore inherit the thickness of the
adjacent interior cell, which keeps every `dzc`/`dzf` strictly positive
and makes the near-boundary stencils well defined.
"""
function _extend_faces(zfi::Vector{T}, Nz::Int, Hz::Int) where {T}
    Ntz = Nz + 2Hz
    zf = Vector{T}(undef, Ntz + 1)
    @inbounds for k in 1:Nz+1
        zf[Hz + k] = zfi[k]
    end
    # below the bottom boundary
    @inbounds for k in Hz:-1:1
        zf[k] = 2zf[k+1] - zf[k+2]
    end
    # above the top boundary
    @inbounds for k in (Hz + Nz + 2):(Ntz + 1)
        zf[k] = 2zf[k-1] - zf[k-2]
    end
    return zf
end

function StretchedGrid(::Type{T} = Float64;
                       size::NTuple{3,Int},
                       extent::NTuple{3,Real},
                       halo::NTuple{3,Int} = (2, 2, 2),
                       origin::NTuple{3,Real} = (0, 0, 0),
                       stretching = :uniform) where {T<:AbstractFloat}
    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo
    (Nx > 0 && Ny > 0 && Nz > 0) || throw(ArgumentError("size must be positive, got $size"))
    (Hx ≥ 1 && Hy ≥ 1 && Hz ≥ 1) ||
        throw(ArgumentError("halo must be ≥ 1 in every direction, got $halo"))
    Lx, Ly, Lz = T.(extent)
    x0, y0, z0 = T.(origin)
    Lz > 0 || throw(ArgumentError("vertical extent must be positive, got $Lz"))

    zfi, kind = _interior_faces(T, Nz, z0, Lz, stretching)
    Ntz = Nz + 2Hz

    if kind === :uniform
        # Closed-form metrics, deliberately spelled with the *same* floating-
        # point expressions `RegularGrid` uses, so that a `:uniform`
        # StretchedGrid reproduces it bit-for-bit rather than to round-off.
        # Differencing accumulated face positions would leave a ~1e-13 drift
        # whenever Lz/Nz is not exactly representable, which would blunt the
        # regression check that uniform stretching changes nothing.
        dz  = Lz / Nz
        zf  = T[z0 + (k - Hz - 1)   * dz for k in 1:Ntz+1]
        zc  = T[z0 + (k - Hz - 0.5) * dz for k in 1:Ntz]
        dzc = fill(dz, Ntz)
        dzf = fill(dz, Ntz + 1)
    else
        zf  = _extend_faces(zfi, Nz, Hz)
        dzc = T[zf[k+1] - zf[k] for k in 1:Ntz]
        zc  = T[(zf[k] + zf[k+1]) / 2 for k in 1:Ntz]
        # dzf[k] = zc[k] - zc[k-1]; the two outermost entries have no pair of
        # centres to straddle, so they inherit the adjacent cell thickness
        # (consistent with the constant-thickness ghost extrapolation).
        dzf = Vector{T}(undef, Ntz + 1)
        dzf[1] = dzc[1]
        @inbounds for k in 2:Ntz
            dzf[k] = zc[k] - zc[k-1]
        end
        dzf[Ntz+1] = dzc[Ntz]
    end

    return StretchedGrid{T}(Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz,
                            x0, y0, z0, Lx / Nx, Ly / Ny,
                            zf, zc, dzc, dzf, kind)
end

"""
    StretchedGrid(g::RegularGrid)

Lift a uniform [`RegularGrid`](@ref) into the stretched representation.
The resulting mesh is numerically identical (`stretching = :uniform`);
this exists so that terrain-following grids and other stretched-grid
machinery can accept either mesh type.
"""
function StretchedGrid(g::RegularGrid{T}) where {T}
    return StretchedGrid(T; size = (g.Nx, g.Ny, g.Nz),
                            extent = (g.Lx, g.Ly, g.Lz),
                            halo = (g.Hx, g.Hy, g.Hz),
                            origin = (g.x0, g.y0, g.z0),
                            stretching = :uniform)
end

StretchedGrid(g::StretchedGrid) = g

# --- AbstractMesh interface ----------------------------------------------

totalsize(g::StretchedGrid) = (g.Nx + 2g.Hx, g.Ny + 2g.Hy, g.Nz + 2g.Hz)

interior(g::StretchedGrid) = (g.Hx+1 : g.Hx+g.Nx,
                              g.Hy+1 : g.Hy+g.Ny,
                              g.Hz+1 : g.Hz+g.Nz)

xnodes(g::StretchedGrid, ::CellCenter) = @. g.x0 + ((1:g.Nx) - 0.5) * g.dx
ynodes(g::StretchedGrid, ::CellCenter) = @. g.y0 + ((1:g.Ny) - 0.5) * g.dy
znodes(g::StretchedGrid, ::CellCenter) = g.zc[g.Hz+1 : g.Hz+g.Nz]

xnodes(g::StretchedGrid, ::FaceX) = @. g.x0 + (0:g.Nx-1) * g.dx
ynodes(g::StretchedGrid, ::FaceY) = @. g.y0 + (0:g.Ny-1) * g.dy
znodes(g::StretchedGrid, ::FaceZ) = g.zf[g.Hz+1 : g.Hz+g.Nz]

xnodes(g::StretchedGrid, ::Union{FaceY,FaceZ}) = xnodes(g, CellCenter())
ynodes(g::StretchedGrid, ::Union{FaceX,FaceZ}) = ynodes(g, CellCenter())
znodes(g::StretchedGrid, ::Union{FaceX,FaceY}) = znodes(g, CellCenter())

# --- accessors -----------------------------------------------------------

"""
    zfaces(g)        # bottom face of every cell, halos included (length Nz+2Hz+1)
    zcenters(g)      # cell centres, halos included (length Nz+2Hz)
    cell_thickness(g)  # dzc — control-volume height at cell centres
    center_spacing(g)  # dzf — centre-to-centre distance at faces

Raw vertical metric vectors of a [`StretchedGrid`](@ref). Indices are
*storage* indices: the interior spans `Hz+1 : Hz+Nz`. For a
[`TerrainFollowingGrid`](@ref) these return the **computational**
(σ-space) metrics; multiply by [`metric_jacobian`](@ref) for physical
lengths.
"""
zfaces(g::StretchedGrid)         = g.zf
@doc (@doc zfaces) zcenters(g::StretchedGrid)       = g.zc
@doc (@doc zfaces) cell_thickness(g::StretchedGrid) = g.dzc
@doc (@doc zfaces) center_spacing(g::StretchedGrid) = g.dzf

"""
    grid_stretching_ratio(g) -> Real

Largest ratio between the thicknesses of two *adjacent interior* cells,

```math
r_{\\max} = \\max_k \\max\\!\\left(\\frac{\\Delta z_{k+1}}{\\Delta z_k},
                                    \\frac{\\Delta z_k}{\\Delta z_{k+1}}\\right) \\ge 1 .
```

A grid-quality metric, not a diagnostic of the solver. The truncation
error of the non-uniform second-order stencils contains a term
proportional to `(r-1)`, so accuracy degrades gracefully but noticeably
once `r ≳ 1.2`; most operational LES setups keep `r < 1.1`. Returns
exactly `1` for a uniform grid.
"""
function grid_stretching_ratio(g::StretchedGrid)
    kk = g.Hz+1 : g.Hz+g.Nz
    r = one(eltype(g))
    @inbounds for k in first(kk):last(kk)-1
        a, b = g.dzc[k], g.dzc[k+1]
        r = max(r, b / a, a / b)
    end
    return r
end

function Base.show(io::IO, g::StretchedGrid{T}) where {T}
    print(io, "StretchedGrid{$T}(size=($(g.Nx),$(g.Ny),$(g.Nz)), ")
    print(io, "extent=($(g.Lx),$(g.Ly),$(g.Lz)), halo=($(g.Hx),$(g.Hy),$(g.Hz)), ")
    print(io, "stretching=:$(g.stretching), ")
    kk = g.Hz+1 : g.Hz+g.Nz
    print(io, "dz=[$(g.dzc[first(kk)]) … $(g.dzc[last(kk)])], ")
    print(io, "r_max=$(round(grid_stretching_ratio(g); digits=4)))")
end

# =========================================================================
# Full-storage physical coordinates + halo-aware initialisation
# =========================================================================

"""
    _xcoords(g, loc), _ycoords(g, loc), _zcoords(g, loc)

Physical coordinates over the **whole storage array** (halos included) for
a field at `loc`. `Grids.xnodes` & friends only cover the interior, which
is what user diagnostics want; the halo-aware versions are what
[`set_physical!`](@ref) needs to fill ghost cells analytically.
"""
_xcoords(g::StretchedGrid, ::FaceX) = [g.x0 + (i - g.Hx - 1) * g.dx for i in 1:g.Nx+2g.Hx]
_xcoords(g::StretchedGrid, ::Location) = [g.x0 + (i - g.Hx - 0.5) * g.dx for i in 1:g.Nx+2g.Hx]
_ycoords(g::StretchedGrid, ::FaceY) = [g.y0 + (j - g.Hy - 1) * g.dy for j in 1:g.Ny+2g.Hy]
_ycoords(g::StretchedGrid, ::Location) = [g.y0 + (j - g.Hy - 0.5) * g.dy for j in 1:g.Ny+2g.Hy]
_zcoords(g::StretchedGrid, ::FaceZ)   = g.zf[1:g.Nz+2g.Hz]
_zcoords(g::StretchedGrid, ::Location) = g.zc

"""
    set_physical!(f::Field, fun) -> f

Fill **every** entry of `f` — interior *and* halos — with
`fun(x, y, z)` evaluated at the true physical position of each sample.

Differs from `Fields.set!` in two ways that matter for stretched and
terrain-following meshes:

  * halos are filled analytically instead of being left to a
    `fill_halos!` extrapolation, which is what makes exactness tests on
    polynomial profiles meaningful right up to the boundary;
  * on a [`TerrainFollowingGrid`](@ref) the vertical coordinate is
    genuinely three-dimensional (`z = z(x,y,σ)`), so a separable
    `set!(f, (x,y,z) -> …)` over `znodes` would sample the wrong heights.
"""
set_physical!(f::Field, fun) = _set_physical!(f, fun, mesh(f))

# =========================================================================
# Finite-volume stencils on a StretchedGrid
# =========================================================================
#
# x and y are uniform, so those kernels are literal copies of the
# `RegularGrid` ones (they cannot be inherited: dispatch is on the
# concrete mesh type). The z kernels carry the metrics.

# --- horizontal first derivatives (uniform, unchanged) -------------------

function _∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g::StretchedGrid)
    inv2dx = 1 / (2g.dx)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i+1, j, k] - f.data[i-1, j, k]) * inv2dx
    end
    return out
end

function _∂y!(out::Field{CellCenter}, f::Field{CellCenter}, g::StretchedGrid)
    inv2dy = 1 / (2g.dy)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i, j+1, k] - f.data[i, j-1, k]) * inv2dy
    end
    return out
end

function _∂x!(out::Field{FaceX}, f::Field{CellCenter}, g::StretchedGrid)
    idx = 1 / g.dx
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i, j, k] - f.data[i-1, j, k]) * idx
    end
    return out
end

function _∂y!(out::Field{FaceY}, f::Field{CellCenter}, g::StretchedGrid)
    idy = 1 / g.dy
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i, j, k] - f.data[i, j-1, k]) * idy
    end
    return out
end

function _∂x!(out::Field{CellCenter}, u::Field{FaceX}, g::StretchedGrid)
    idx = 1 / g.dx
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (u.data[i+1, j, k] - u.data[i, j, k]) * idx
    end
    return out
end

function _∂y!(out::Field{CellCenter}, v::Field{FaceY}, g::StretchedGrid)
    idy = 1 / g.dy
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (v.data[i, j+1, k] - v.data[i, j, k]) * idy
    end
    return out
end

# --- vertical first derivatives (metric-correct) -------------------------

"""
    _∂z!(out::Field{FaceZ}, f::Field{CellCenter}, g::StretchedGrid)

Centre → face gradient. The face `k` separates the centres `k-1` and `k`,
whose distance is `dzf[k]`:

```math
\\left.\\frac{\\partial f}{\\partial z}\\right|_{z_f[k]}
    = \\frac{f_k - f_{k-1}}{\\Delta z_f[k]} .
```

Exact for any linear profile on *any* face distribution — a two-point
divided difference is an exact derivative of the interpolating line — and
second-order accurate in general because a face sits exactly midway
between the two centres it separates *only* when the grid is uniform; on
a stretched grid the formula is a first-order-accurate approximation to
the derivative at the face but a second-order-accurate one at the
midpoint of the two centres, which is where the flux is actually needed
in the conservative Laplacian below. Using this pair
(`dzf` here, `dzc` in the divergence) is what makes the composed
Laplacian second order.
"""
function _∂z!(out::Field{FaceZ}, f::Field{CellCenter}, g::StretchedGrid)
    dzf = g.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        idz = 1 / dzf[k]
        for j in jj, i in ii
            out.data[i, j, k] = (f.data[i, j, k] - f.data[i, j, k-1]) * idz
        end
    end
    return out
end

"""
    _∂z!(out::Field{CellCenter}, w::Field{FaceZ}, g::StretchedGrid)

Face → centre divergence contribution. `w[k]` and `w[k+1]` are the bottom
and top faces of control volume `k`, whose height is `dzc[k]`:

```math
\\left.\\frac{\\partial w}{\\partial z}\\right|_{z_c[k]}
    = \\frac{w_{k+1} - w_k}{\\Delta z_c[k]} .
```

This is the *exact* mean divergence over the cell (Gauss' theorem), which
is why the discrete projection stays conservative on a stretched grid.
"""
function _∂z!(out::Field{CellCenter}, w::Field{FaceZ}, g::StretchedGrid)
    dzc = g.dzc
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        idz = 1 / dzc[k]
        for j in jj, i in ii
            out.data[i, j, k] = (w.data[i, j, k+1] - w.data[i, j, k]) * idz
        end
    end
    return out
end

# Three-point unequal-spacing centred derivative weights.
#   h₋ = zc[k]   - zc[k-1] = dzf[k]
#   h₊ = zc[k+1] - zc[k]   = dzf[k+1]
# Differentiating the quadratic Lagrange interpolant through the three
# centres and evaluating at the middle node gives
#
#   f'(z_k) ≈ -h₊/(h₋(h₋+h₊)) f_{k-1}
#             + (h₊-h₋)/(h₋h₊) f_k
#             + h₋/(h₊(h₋+h₊)) f_{k+1}
#
# which collapses to (f_{k+1}-f_{k-1})/2h for h₋=h₊=h. The naive
# (f_{k+1}-f_{k-1})/(h₋+h₊) is *not* equivalent: it loses the f_k term
# and is only first-order accurate when h₊≠h₋.
@inline function _dz_weights(hm::T, hp::T) where {T}
    s = hm + hp
    return (-hp / (hm * s), (hp - hm) / (hm * hp), hm / (hp * s))
end

"""
    _∂z!(out::Field{CellCenter}, f::Field{CellCenter}, g::StretchedGrid)

Collocated centre → centre vertical derivative using the **unequal-spacing
three-point** stencil. With `h₋ = dzf[k]` and `h₊ = dzf[k+1]`,

```math
\\left.\\frac{\\partial f}{\\partial z}\\right|_{z_c[k]}
 = \\frac{-h_+}{h_-(h_-+h_+)} f_{k-1}
 + \\frac{h_+ - h_-}{h_- h_+} f_k
 + \\frac{h_-}{h_+(h_-+h_+)} f_{k+1}
 + O(h_- h_+) ,
```

the derivative at the middle node of the quadratic through the three cell
centres. It is therefore **exact for every polynomial up to degree 2** on
an arbitrary face distribution, and reduces to `(f_{k+1}-f_{k-1})/2Δz` on
a uniform grid. The `f_k` term is essential: dropping it (the tempting
`(f_{k+1}-f_{k-1})/(h_-+h_+)`) demotes the stencil to first order.
"""
function _∂z!(out::Field{CellCenter}, f::Field{CellCenter}, g::StretchedGrid)
    dzf = g.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        wm, wc, wp = _dz_weights(dzf[k], dzf[k+1])
        for j in jj, i in ii
            out.data[i, j, k] = wm * f.data[i, j, k-1] +
                                wc * f.data[i, j, k]   +
                                wp * f.data[i, j, k+1]
        end
    end
    return out
end

# --- Laplacian ----------------------------------------------------------

"""
    _laplacian!(out::Field{L}, f::Field{L}, g::StretchedGrid) where {L}

Seven-point Laplacian in the conservative non-uniform form. Horizontally
the usual uniform stencil; vertically the flux difference

```math
\\left.\\nabla^2_z f\\right|_k =
 \\frac{1}{\\Delta z_c[k]}
 \\left[\\frac{f_{k+1}-f_k}{\\Delta z_f[k+1]}
      - \\frac{f_k-f_{k-1}}{\\Delta z_f[k]}\\right].
```

This is `div(grad)` built from the two stencils above: the gradient lives
on faces (metric `dzf`), its divergence over the control volume uses the
cell height (metric `dzc`). The operator is consequently
**self-adjoint and negative-definite** on a non-uniform grid, which the
naive `(f_{k+1}-2f_k+f_{k-1})/Δz_c^2` is not — and self-adjointness is
what makes the CG pressure solve converge. Second order accurate for
smooth stretching (the leading error term scales with the local
thickness ratio minus one).
"""
function _laplacian!(out::Field{L}, f::Field{L}, g::StretchedGrid) where {L<:Location}
    idx2 = 1 / g.dx^2
    idy2 = 1 / g.dy^2
    dzc, dzf = g.dzc, g.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        iv   = 1 / dzc[k]
        idzm = 1 / dzf[k]
        idzp = 1 / dzf[k+1]
        for j in jj, i in ii
            c = f.data[i, j, k]
            zterm = ((f.data[i,j,k+1] - c) * idzp - (c - f.data[i,j,k-1]) * idzm) * iv
            out.data[i, j, k] = (f.data[i+1,j,k] - 2c + f.data[i-1,j,k]) * idx2 +
                                (f.data[i,j+1,k] - 2c + f.data[i,j-1,k]) * idy2 +
                                zterm
        end
    end
    return out
end

# --- divergence ---------------------------------------------------------

"""
    _divergence!(out, u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, g::StretchedGrid)

Arakawa-C divergence with the vertical metric:

```math
(\\nabla\\!\\cdot\\!\\mathbf{u})_{ijk} =
 \\frac{u_{i+1}-u_i}{\\Delta x} + \\frac{v_{j+1}-v_j}{\\Delta y}
 + \\frac{w_{k+1}-w_k}{\\Delta z_c[k]} .
```

Exactly the net volume flux through the faces of control volume `(i,j,k)`
divided by its volume, so a discretely solenoidal field has *zero*
divergence to round-off on any face distribution.
"""
function _divergence!(out::Field{CellCenter},
                      u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                      g::StretchedGrid)
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc = g.dzc
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        idz = 1 / dzc[k]
        for j in jj, i in ii
            out.data[i,j,k] = (u.data[i+1,j,k] - u.data[i,j,k]) * idx +
                              (v.data[i,j+1,k] - v.data[i,j,k]) * idy +
                              (w.data[i,j,k+1] - w.data[i,j,k]) * idz
        end
    end
    return out
end

"""
    _divergence!(out, u, v, w, g::StretchedGrid)   # all at CellCenter

Collocated diagnostic divergence. The vertical term uses the unequal-
spacing three-point weights (see the collocated `_∂z!`), not a naive
`(w_{k+1}-w_{k-1})/(h_-+h_+)`.
"""
function _divergence!(out::Field{CellCenter},
                      u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                      g::StretchedGrid)
    inv2dx = 1 / (2g.dx); inv2dy = 1 / (2g.dy)
    dzf = g.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        wm, wc, wp = _dz_weights(dzf[k], dzf[k+1])
        for j in jj, i in ii
            out.data[i,j,k] = (u.data[i+1,j,k] - u.data[i-1,j,k]) * inv2dx +
                              (v.data[i,j+1,k] - v.data[i,j-1,k]) * inv2dy +
                              wm * w.data[i,j,k-1] + wc * w.data[i,j,k] + wp * w.data[i,j,k+1]
        end
    end
    return out
end

# --- staggered interpolation --------------------------------------------

# horizontal: uniform, plain averages
function _interp_to_face!(out::Field{FaceX}, f::Field{CellCenter}, g::StretchedGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (f.data[i-1,j,k] + f.data[i,j,k])
    end
    return out
end
function _interp_to_face!(out::Field{FaceY}, f::Field{CellCenter}, g::StretchedGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (f.data[i,j-1,k] + f.data[i,j,k])
    end
    return out
end

"""
    _interp_to_face!(out::Field{FaceZ}, f::Field{CellCenter}, g::StretchedGrid)

Volume-weighted centre → z-face interpolation:

```math
f_{k}^{\\text{face}} =
 \\frac{\\Delta z_c[k-1]\\, f_k + \\Delta z_c[k]\\, f_{k-1}}
       {\\Delta z_c[k-1] + \\Delta z_c[k]} .
```

Note the *cross* pairing — the weight on `f_k` is the thickness of the
**other** cell. That is the geometric linear interpolation: face `k` sits
at distance `Δz_c[k-1]/2` above centre `k-1` and `Δz_c[k]/2` below centre
`k`, so the nearer centre gets the larger weight. The result is exact for
linear profiles; the naive `½(f_{k-1}+f_k)` carries an
`O((Δz_c[k]-Δz_c[k-1]))` first-order error that pollutes buoyancy and
advective fluxes right where the grid is stretched hardest.
"""
function _interp_to_face!(out::Field{FaceZ}, f::Field{CellCenter}, g::StretchedGrid)
    dzc = g.dzc
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        wlo = dzc[k]   / (dzc[k-1] + dzc[k])   # weight on f[k-1]
        whi = dzc[k-1] / (dzc[k-1] + dzc[k])   # weight on f[k]
        for j in jj, i in ii
            out.data[i,j,k] = wlo * f.data[i,j,k-1] + whi * f.data[i,j,k]
        end
    end
    return out
end

function _interp_to_center!(out::Field{CellCenter}, u::Field{FaceX}, g::StretchedGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (u.data[i,j,k] + u.data[i+1,j,k])
    end
    return out
end
function _interp_to_center!(out::Field{CellCenter}, v::Field{FaceY}, g::StretchedGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (v.data[i,j,k] + v.data[i,j+1,k])
    end
    return out
end

"""
    _interp_to_center!(out::Field{CellCenter}, w::Field{FaceZ}, g::StretchedGrid)

Volume-weighted z-face → centre interpolation:

```math
w_k^{\\text{centre}} = \\frac{(z_c[k]-z_f[k])\\,w_{k+1} + (z_f[k+1]-z_c[k])\\,w_k}
                             {\\Delta z_c[k]} .
```

Because `z_c[k]` is *defined* as the midpoint of its two faces, both
weights equal `½` on every grid — a stretched grid needs no correction in
this direction. The distance form is kept explicit so the symmetry with
the centre→face kernel is visible and so the code stays correct if a
future mesh ever places centres off-midpoint (e.g. mass-weighted levels).
"""
function _interp_to_center!(out::Field{CellCenter}, w::Field{FaceZ}, g::StretchedGrid)
    zf, zc, dzc = g.zf, g.zc, g.dzc
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        whi = (zc[k] - zf[k])   / dzc[k]   # weight on w[k+1]
        wlo = (zf[k+1] - zc[k]) / dzc[k]   # weight on w[k]
        for j in jj, i in ii
            out.data[i,j,k] = wlo * w.data[i,j,k] + whi * w.data[i,j,k+1]
        end
    end
    return out
end

# --- a few physics kernels that carry the vertical metric ---------------

"""
    _add_surface_flux!(tϕ, F_bot, F_top, g::StretchedGrid)

Inject prescribed wall fluxes into the first and last interior cells. The
tendency is the flux divided by the **local** cell thickness, so a
stretched grid with a thin first cell feels a correspondingly larger
heating rate — the whole point of refining near the surface.
"""
function _add_surface_flux!(tϕ::Field{CellCenter}, F_bot::Real, F_top::Real,
                            g::StretchedGrid)
    ii, jj, kk = interior(g)
    k_bot, k_top = first(kk), last(kk)
    inv_bot = 1 / g.dzc[k_bot]
    inv_top = 1 / g.dzc[k_top]
    @inbounds for j in jj, i in ii
        tϕ.data[i, j, k_bot] += F_bot * inv_bot
        tϕ.data[i, j, k_top] -= F_top * inv_top
    end
    return tϕ
end

_zloc_coords(g::StretchedGrid, ::FaceZ) = g.zf
_zloc_coords(g::StretchedGrid, ::Location) = g.zc

"""
    _add_sponge!(tf, f, α_max, z_start, z_top, f_ref, g::StretchedGrid)

Rayleigh damping toward `f_ref` for `z ≥ z_start`, with
`α(z) = α_max sin²(½π (z-z_start)/(z_top-z_start))`. Identical to the
`RegularGrid` kernel except that the sponge profile is sampled at the
*actual* stretched heights, so the damping layer covers the physical
region the user asked for rather than a fixed number of levels.
"""
function _add_sponge!(tf::Field{L}, f::Field{L},
                      α_max::Real, z_start::Real, z_top::Real,
                      f_ref::Real, g::StretchedGrid) where {L<:Location}
    zs = _zloc_coords(g, L())
    ii, jj, kk = interior(g)
    prof = sponge_profile(α_max, z_start, z_top)   # the shared damping profile
    @inbounds for k in kk
        z = zs[k]
        in_damping(prof, z) || continue
        α = damping_rate(prof, z)
        for j in jj, i in ii
            tf.data[i, j, k] -= α * (f.data[i, j, k] - f_ref)
        end
    end
    return tf
end

"""
    _add_buoyancy!(tw, θ, θ₀, g_accel, g::StretchedGrid)

Boussinesq buoyancy `g(θ-θ₀)/θ₀` added to the `FaceZ` tendency, with `θ`
brought to the face by the **volume-weighted** interpolation. Using the
naive half-sum here would put a spurious first-order buoyancy source
wherever the grid stretches, which shows up as a persistent artificial
circulation in a supposedly quiescent stratified column.
"""
function _add_buoyancy!(tw::Field{FaceZ}, θ::Field{CellCenter},
                        θ₀::Real, g_accel::Real, g::StretchedGrid)
    coef = g_accel / θ₀
    dzc = g.dzc
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        wlo = dzc[k]   / (dzc[k-1] + dzc[k])
        whi = dzc[k-1] / (dzc[k-1] + dzc[k])
        for j in jj, i in ii
            θface = wlo * θ.data[i,j,k-1] + whi * θ.data[i,j,k]
            tw.data[i,j,k] += coef * (θface - θ₀)
        end
    end
    return tw
end

# =========================================================================
# TerrainFollowingGrid — Gal-Chen / Somerville σ-coordinate
# =========================================================================
#
# The computational domain is the flat slab (x, y, σ) with σ ∈ [0, H],
# H = Lz. It is mapped onto physical space by
#
#       z(x, y, σ) = h(x,y) + σ (H - h(x,y)) / H                    (GC1)
#
# so that σ = 0 follows the orography (z = h) and σ = H is the flat model
# lid (z = H). Inverting,
#
#       σ(x, y, z) = H (z - h) / (H - h).                           (GC2)
#
# The metric terms follow by differentiating (GC1)/(GC2):
#
#       J  ≡ ∂z/∂σ  = (H - h)/H            (vertical Jacobian, σ-free)
#       ∂σ/∂z       = 1/J
#       ∂σ/∂x|_z    = -(H - σ)/(H - h) · ∂h/∂x                      (GC3)
#       ∂σ/∂y|_z    = -(H - σ)/(H - h) · ∂h/∂y
#
# (GC3) is the crucial one. A horizontal derivative taken along a
# coordinate surface is *not* a horizontal derivative at constant height:
#
#       ∂f/∂x|_z = ∂f/∂x|_σ + (∂σ/∂x) ∂f/∂σ .                       (GC4)
#
# Dropping the second term is the classic terrain-following blunder: in a
# hydrostatic atmosphere over a hill it produces a spurious horizontal
# pressure gradient of order (∂h/∂x)·ρg, i.e. hundreds of times the real
# signal. Note that (GC3) → 0 at σ = H, so the terrain influence decays
# linearly with height and vanishes at the lid — that is the defining
# property of the Gal-Chen coordinate (and its weakness: terrain features
# are felt throughout the whole column, which is what SLEVE later fixes).
#
# Divergence is discretised in the *conservative* contravariant form
#
#       ∇·u = (1/J) [ ∂ₓ(J u) + ∂_y(J v) + ∂_σ(J w̃) ],              (GC5)
#       J w̃ = w + J ( σ_x u + σ_y v ),
#
# which telescopes exactly over the grid and therefore keeps the pressure
# projection discretely mass-conserving over orography. Likewise the
# Laplacian is assembled as div(grad) with the full contravariant metric
#
#       G^{xx}=G^{yy}=1, G^{xσ}=σ_x, G^{yσ}=σ_y,
#       G^{σσ}=σ_x² + σ_y² + 1/J²,                                  (GC6)
#
# so it stays self-adjoint (a requirement for the CG pressure solver).
#
# For h ≡ 0 every metric collapses to J = 1, σ_x = σ_y = 0, G^{σσ} = 1 and
# all kernels reduce *identically* (to round-off) to their StretchedGrid
# counterparts.

"""
    TerrainFollowingGrid{T} <: AbstractMesh{T}

Gal-Chen/Somerville terrain-following (σ-) mesh built on top of a
[`StretchedGrid`](@ref) that supplies the computational vertical
coordinate. See [`terrain_following_grid`](@ref) to construct one.

    z(x, y, σ) = h(x,y) + σ (H - h(x,y)) / H,   σ ∈ [0, H], H = Lz

Stored metrics (all on the full storage array, halos included):

  * `h`, `hx`, `hy` — terrain height and its horizontal slopes.
  * `J` — vertical Jacobian `∂z/∂σ = (H-h)/H`, at cell centres;
    `Jx`, `Jy`, `Jzf` are the same quantity at x-, y- and σ-faces.
  * `dσdx`, `dσdy` — `∂σ/∂x|_z`, `∂σ/∂y|_z` at cell centres, plus the
    `_xf` / `_yf` / `_zf` variants co-located with the faces where the
    cross-derivative terms of each stencil are needed.
  * `zc3`, `zf3` — physical heights of cell centres and σ-faces.

`znodes` returns the **computational** σ levels (it must return a vector);
physical heights are three-dimensional and live in
[`physical_heights`](@ref). Initialise fields with
[`set_physical!`](@ref), not `Fields.set!`, so that samples land at their
true height.

Accessors: [`metric_jacobian`](@ref), [`terrain_height`](@ref),
[`physical_heights`](@ref).
"""
struct TerrainFollowingGrid{T<:AbstractFloat} <: AbstractMesh{T}
    base::StretchedGrid{T}
    Nx::Int; Ny::Int; Nz::Int
    Hx::Int; Hy::Int; Hz::Int
    Lx::T;   Ly::T;   Lz::T
    x0::T;   y0::T;   z0::T
    dx::T;   dy::T
    h::Matrix{T}
    hx::Matrix{T}
    hy::Matrix{T}
    J::Array{T,3}
    Jx::Array{T,3}
    Jy::Array{T,3}
    Jzf::Array{T,3}
    dσdx::Array{T,3}
    dσdy::Array{T,3}
    dσdx_xf::Array{T,3}
    dσdy_yf::Array{T,3}
    dσdx_zf::Array{T,3}
    dσdy_zf::Array{T,3}
    zc3::Array{T,3}
    zf3::Array{T,3}
end

# --- terrain field ingestion ---------------------------------------------

function _terrain_matrix(::Type{T}, g::StretchedGrid, h) where {T}
    Ntx, Nty = g.Nx + 2g.Hx, g.Ny + 2g.Hy
    H = Matrix{T}(undef, Ntx, Nty)
    if h isa Function
        xs = _xcoords(g, CellCenter()); ys = _ycoords(g, CellCenter())
        @inbounds for j in 1:Nty, i in 1:Ntx
            H[i, j] = T(h(xs[i], ys[j]))
        end
    elseif h isa Real
        fill!(H, T(h))
    elseif h isa AbstractMatrix
        if Base.size(h) == (Ntx, Nty)
            H .= T.(h)
        elseif Base.size(h) == (g.Nx, g.Ny)
            # interior given; halos filled by periodic wrap, matching the
            # default periodic horizontal boundary conditions.
            @inbounds for j in 1:Nty, i in 1:Ntx
                ii = mod(i - g.Hx - 1, g.Nx) + 1
                jj = mod(j - g.Hy - 1, g.Ny) + 1
                H[i, j] = T(h[ii, jj])
            end
        else
            throw(ArgumentError("terrain matrix must be ($(g.Nx),$(g.Ny)) or " *
                                "($Ntx,$Nty), got $(Base.size(h))"))
        end
    else
        throw(ArgumentError("terrain must be a Function h(x,y), a Real, or a Matrix"))
    end
    return H
end

# centred slope on the full array; one-sided at the outermost ghost ring
function _slopes(hmat::Matrix{T}, dx::T, dy::T) where {T}
    Ntx, Nty = Base.size(hmat)
    hx = similar(hmat); hy = similar(hmat)
    @inbounds for j in 1:Nty, i in 1:Ntx
        if i == 1
            hx[i,j] = (hmat[2,j] - hmat[1,j]) / dx
        elseif i == Ntx
            hx[i,j] = (hmat[Ntx,j] - hmat[Ntx-1,j]) / dx
        else
            hx[i,j] = (hmat[i+1,j] - hmat[i-1,j]) / (2dx)
        end
        if j == 1
            hy[i,j] = (hmat[i,2] - hmat[i,1]) / dy
        elseif j == Nty
            hy[i,j] = (hmat[i,Nty] - hmat[i,Nty-1]) / dy
        else
            hy[i,j] = (hmat[i,j+1] - hmat[i,j-1]) / (2dy)
        end
    end
    return hx, hy
end

"""
    terrain_following_grid(mesh, h) -> TerrainFollowingGrid

Wrap `mesh` (a [`RegularGrid`](@ref) or [`StretchedGrid`](@ref), which
becomes the computational σ-coordinate) around the orography `h` and
precompute every metric of the Gal-Chen transformation

```math
z(x,y,\\sigma) = h(x,y) + \\sigma\\,\\frac{H - h(x,y)}{H},
\\qquad H = L_z .
```

`h` may be

  * a function `h(x, y)` — sampled at cell centres including halos, which
    gives smooth ghost-cell metrics even for non-periodic terrain;
  * a `Real` — flat terrain at that elevation;
  * a `Matrix` of size `(Nx, Ny)` (halos filled by periodic wrap) or of
    the full storage size `(Nx+2Hx, Ny+2Hy)` (used verbatim).

Terrain slopes are evaluated with centred differences on the same mesh the
stencils use, so the discrete metric identities the conservative kernels
rely on hold exactly rather than approximately.

Throws if the terrain reaches the model lid (`max h ≥ H`), which would
make the Jacobian vanish.
"""
function terrain_following_grid(mesh::AbstractMesh{T}, h) where {T}
    base = StretchedGrid(mesh)
    Ntx, Nty, Ntz = totalsize(base)
    H = base.Lz
    hmat = _terrain_matrix(T, base, h)

    hmin, hmax = extrema(hmat)
    hmax < H || throw(ArgumentError("terrain height max = $hmax reaches or exceeds " *
                                    "the model lid H = $H; the Jacobian would vanish"))
    hmin > base.z0 - H ||
        throw(ArgumentError("terrain height min = $hmin is unphysically far below z0"))

    hx, hy = _slopes(hmat, base.dx, base.dy)

    σc = [base.zc[k] - base.z0 for k in 1:Ntz]
    σf = [base.zf[k] - base.z0 for k in 1:Ntz]

    J   = Array{T,3}(undef, Ntx, Nty, Ntz)
    Jx  = similar(J); Jy = similar(J); Jzf = similar(J)
    dσdx = similar(J); dσdy = similar(J)
    dσdx_xf = similar(J); dσdy_yf = similar(J)
    dσdx_zf = similar(J); dσdy_zf = similar(J)
    zc3 = similar(J); zf3 = similar(J)

    @inbounds for j in 1:Nty, i in 1:Ntx
        hij = hmat[i, j]
        Jij = (H - hij) / H
        # terrain and slope at the x-/y-faces of this cell (west / south)
        im = max(i - 1, 1); jm = max(j - 1, 1)
        h_xf  = (hmat[i,j] + hmat[im,j]) / 2
        h_yf  = (hmat[i,j] + hmat[i,jm]) / 2
        hx_xf = (hmat[i,j] - hmat[im,j]) / base.dx
        hy_yf = (hmat[i,j] - hmat[i,jm]) / base.dy
        Jx_ij = (H - h_xf) / H
        Jy_ij = (H - h_yf) / H
        for k in 1:Ntz
            J[i,j,k]   = Jij
            Jx[i,j,k]  = Jx_ij
            Jy[i,j,k]  = Jy_ij
            Jzf[i,j,k] = Jij                    # J is σ-independent (Gal-Chen)
            dσdx[i,j,k] = -(H - σc[k]) * hx[i,j] / (H - hij)
            dσdy[i,j,k] = -(H - σc[k]) * hy[i,j] / (H - hij)
            dσdx_xf[i,j,k] = -(H - σc[k]) * hx_xf / (H - h_xf)
            dσdy_yf[i,j,k] = -(H - σc[k]) * hy_yf / (H - h_yf)
            dσdx_zf[i,j,k] = -(H - σf[k]) * hx[i,j] / (H - hij)
            dσdy_zf[i,j,k] = -(H - σf[k]) * hy[i,j] / (H - hij)
            zc3[i,j,k] = hij + σc[k] * Jij
            zf3[i,j,k] = hij + σf[k] * Jij
        end
    end

    return TerrainFollowingGrid{T}(base,
        base.Nx, base.Ny, base.Nz, base.Hx, base.Hy, base.Hz,
        base.Lx, base.Ly, base.Lz, base.x0, base.y0, base.z0,
        base.dx, base.dy,
        hmat, hx, hy, J, Jx, Jy, Jzf,
        dσdx, dσdy, dσdx_xf, dσdy_yf, dσdx_zf, dσdy_zf, zc3, zf3)
end

# --- AbstractMesh interface ----------------------------------------------

totalsize(g::TerrainFollowingGrid) = totalsize(g.base)
interior(g::TerrainFollowingGrid)  = interior(g.base)

xnodes(g::TerrainFollowingGrid, loc::Location) = xnodes(g.base, loc)
ynodes(g::TerrainFollowingGrid, loc::Location) = ynodes(g.base, loc)
"""
    znodes(g::TerrainFollowingGrid, loc)

**Computational** σ levels, not physical heights — the `AbstractMesh`
interface requires a vector and physical height varies with `(x,y)`. Use
[`physical_heights`](@ref) for the real geometry.
"""
znodes(g::TerrainFollowingGrid, loc::Location) = znodes(g.base, loc)

zfaces(g::TerrainFollowingGrid)         = zfaces(g.base)
zcenters(g::TerrainFollowingGrid)       = zcenters(g.base)
cell_thickness(g::TerrainFollowingGrid) = cell_thickness(g.base)
center_spacing(g::TerrainFollowingGrid) = center_spacing(g.base)
grid_stretching_ratio(g::TerrainFollowingGrid) = grid_stretching_ratio(g.base)

_xcoords(g::TerrainFollowingGrid, loc::Location) = _xcoords(g.base, loc)
_ycoords(g::TerrainFollowingGrid, loc::Location) = _ycoords(g.base, loc)

"""
    metric_jacobian(g::TerrainFollowingGrid) -> Array{T,3}

The vertical Jacobian `J = ∂z/∂σ = (H - h)/H` at cell centres, over the
full storage array. Physical cell thickness at `(i,j,k)` is
`J[i,j,k] * cell_thickness(g)[k]`; `J ≡ 1` for flat terrain. `J` is the
volume element of the transformation, so the conservative kernels weight
every flux by it.
"""
metric_jacobian(g::TerrainFollowingGrid) = g.J

"""
    terrain_height(g::TerrainFollowingGrid) -> Matrix{T}

Orography `h(x,y)` on the full horizontal storage array (halos included).
"""
terrain_height(g::TerrainFollowingGrid) = g.h

"""
    physical_heights(g, loc = CellCenter())

True geometric height of each sample.

  * [`StretchedGrid`](@ref): a vector over the full vertical storage
    (`zc` for centre-like locations, `zf` for `FaceZ`).
  * [`TerrainFollowingGrid`](@ref): an `Array{T,3}` — height depends on
    `(x, y)` through the terrain.
"""
physical_heights(g::StretchedGrid, loc::Location = CellCenter()) = _zcoords(g, loc)
physical_heights(g::TerrainFollowingGrid, ::FaceZ) = g.zf3
physical_heights(g::TerrainFollowingGrid, ::Location = CellCenter()) = g.zc3

function Base.show(io::IO, g::TerrainFollowingGrid{T}) where {T}
    hmin, hmax = extrema(g.h)
    print(io, "TerrainFollowingGrid{$T}(size=($(g.Nx),$(g.Ny),$(g.Nz)), ")
    print(io, "H=$(g.Lz), h∈[$(round(hmin;digits=3)), $(round(hmax;digits=3))], ")
    print(io, "σ-stretching=:$(g.base.stretching), ")
    print(io, "J∈[$(round(minimum(g.J);digits=5)), $(round(maximum(g.J);digits=5))])")
end

# --- halo-aware initialisation ------------------------------------------

function _set_physical!(f::Field, fun, g::Union{StretchedGrid,RegularGrid})
    loc = Fields.location(f)
    sg  = g isa StretchedGrid ? g : StretchedGrid(g)
    xs  = _xcoords(sg, loc); ys = _ycoords(sg, loc); zs = _zcoords(sg, loc)
    @inbounds for k in eachindex(zs), j in eachindex(ys), i in eachindex(xs)
        f.data[i, j, k] = fun(xs[i], ys[j], zs[k])
    end
    return f
end

function _set_physical!(f::Field, fun, g::TerrainFollowingGrid)
    loc = Fields.location(f)
    xs  = _xcoords(g, loc); ys = _ycoords(g, loc)
    zs  = physical_heights(g, loc)
    @inbounds for k in axes(zs, 3), j in eachindex(ys), i in eachindex(xs)
        f.data[i, j, k] = fun(xs[i], ys[j], zs[i, j, k])
    end
    return f
end

# =========================================================================
# Terrain-following stencils
# =========================================================================

# σ-derivative of a cell-centred quantity at (i,j,k), unequal spacing.
@inline function _dσ_c(d::AbstractArray, i, j, k, dzf)
    wm, wc, wp = _dz_weights(dzf[k], dzf[k+1])
    return wm * d[i,j,k-1] + wc * d[i,j,k] + wp * d[i,j,k+1]
end

# σ-derivative of the x-face field u interpolated to the cell centre in x
@inline function _dσ_cx(d::AbstractArray, i, j, k, dzf)
    wm, wc, wp = _dz_weights(dzf[k], dzf[k+1])
    return wm * (d[i,j,k-1] + d[i+1,j,k-1]) / 2 +
           wc * (d[i,j,k]   + d[i+1,j,k])   / 2 +
           wp * (d[i,j,k+1] + d[i+1,j,k+1]) / 2
end

@inline function _dσ_cy(d::AbstractArray, i, j, k, dzf)
    wm, wc, wp = _dz_weights(dzf[k], dzf[k+1])
    return wm * (d[i,j,k-1] + d[i,j+1,k-1]) / 2 +
           wc * (d[i,j,k]   + d[i,j+1,k])   / 2 +
           wp * (d[i,j,k+1] + d[i,j+1,k+1]) / 2
end

# --- vertical derivatives: pure ∂/∂z = (1/J) ∂/∂σ ------------------------

"""
    _∂z!(out::Field{FaceZ}, f::Field{CellCenter}, g::TerrainFollowingGrid)

`∂f/∂z = (1/J) ∂f/∂σ` at the σ-face. No cross terms appear: the two cell
centres straddling a σ-face share the same `(x,y)`, so their difference is
a genuine vertical difference, and `J·Δσ_f` is the physical distance
between them.
"""
function _∂z!(out::Field{FaceZ}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        idσ = 1 / dzf[k]
        for j in jj, i in ii
            out.data[i,j,k] = (f.data[i,j,k] - f.data[i,j,k-1]) * idσ / g.Jzf[i,j,k]
        end
    end
    return out
end

"""
    _∂z!(out::Field{CellCenter}, w::Field{FaceZ}, g::TerrainFollowingGrid)

`∂w/∂z = (1/J)(w_{k+1}-w_k)/Δσ_c[k]`. This is the plain vertical
derivative of `w`; the *divergence* additionally needs the contravariant
flux `J w̃` (see `_divergence!`).
"""
function _∂z!(out::Field{CellCenter}, w::Field{FaceZ}, g::TerrainFollowingGrid)
    dzc = g.base.dzc
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        idσ = 1 / dzc[k]
        for j in jj, i in ii
            out.data[i,j,k] = (w.data[i,j,k+1] - w.data[i,j,k]) * idσ / g.J[i,j,k]
        end
    end
    return out
end

"""
    _∂z!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)

Collocated `∂f/∂z = (1/J) ∂f/∂σ` with the unequal-spacing three-point
σ-derivative.
"""
function _∂z!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = _dσ_c(f.data, i, j, k, dzf) / g.J[i,j,k]
    end
    return out
end

# --- horizontal derivatives: the metric cross-terms ---------------------

"""
    _∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)

Horizontal derivative **at constant physical height**:

```math
\\left.\\frac{\\partial f}{\\partial x}\\right|_z =
\\left.\\frac{\\partial f}{\\partial x}\\right|_\\sigma
+ \\frac{\\partial\\sigma}{\\partial x}\\frac{\\partial f}{\\partial\\sigma},
\\qquad
\\frac{\\partial\\sigma}{\\partial x} = -\\frac{H-\\sigma}{H-h}\\frac{\\partial h}{\\partial x} .
```

The first term is the ordinary centred difference along the coordinate
surface; the second is the terrain correction. Omitting it is the standard
terrain-following error and shows up immediately as a spurious
pressure-gradient force over sloping ground.
"""
function _∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    inv2dx = 1 / (2g.dx)
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = (f.data[i+1,j,k] - f.data[i-1,j,k]) * inv2dx +
                          g.dσdx[i,j,k] * _dσ_c(f.data, i, j, k, dzf)
    end
    return out
end

@doc (@doc _∂x!)
function _∂y!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    inv2dy = 1 / (2g.dy)
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = (f.data[i,j+1,k] - f.data[i,j-1,k]) * inv2dy +
                          g.dσdy[i,j,k] * _dσ_c(f.data, i, j, k, dzf)
    end
    return out
end

"""
    _∂x!(out::Field{FaceX}, f::Field{CellCenter}, g::TerrainFollowingGrid)

Centre → x-face gradient with the terrain cross-term, using `∂σ/∂x`
evaluated **at the x-face** and `∂f/∂σ` averaged from the two straddling
columns. This is the kernel the pressure-gradient force goes through, so
the metric must be co-located with the face or the hydrostatic
cancellation over a slope is lost.
"""
function _∂x!(out::Field{FaceX}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    idx = 1 / g.dx
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        dfdσ = (_dσ_c(f.data, i-1, j, k, dzf) + _dσ_c(f.data, i, j, k, dzf)) / 2
        out.data[i,j,k] = (f.data[i,j,k] - f.data[i-1,j,k]) * idx +
                          g.dσdx_xf[i,j,k] * dfdσ
    end
    return out
end

@doc (@doc _∂x!)
function _∂y!(out::Field{FaceY}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    idy = 1 / g.dy
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        dfdσ = (_dσ_c(f.data, i, j-1, k, dzf) + _dσ_c(f.data, i, j, k, dzf)) / 2
        out.data[i,j,k] = (f.data[i,j,k] - f.data[i,j-1,k]) * idy +
                          g.dσdy_yf[i,j,k] * dfdσ
    end
    return out
end

"""
    _∂x!(out::Field{CellCenter}, u::Field{FaceX}, g::TerrainFollowingGrid)

Face → centre horizontal derivative of `u` at constant height. The
σ-derivative of `u` is taken after averaging `u` to the cell centre in
`x`, keeping the stencil symmetric about the cell.
"""
function _∂x!(out::Field{CellCenter}, u::Field{FaceX}, g::TerrainFollowingGrid)
    idx = 1 / g.dx
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = (u.data[i+1,j,k] - u.data[i,j,k]) * idx +
                          g.dσdx[i,j,k] * _dσ_cx(u.data, i, j, k, dzf)
    end
    return out
end

@doc (@doc _∂x!)
function _∂y!(out::Field{CellCenter}, v::Field{FaceY}, g::TerrainFollowingGrid)
    idy = 1 / g.dy
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = (v.data[i,j+1,k] - v.data[i,j,k]) * idy +
                          g.dσdy[i,j,k] * _dσ_cy(v.data, i, j, k, dzf)
    end
    return out
end

# --- interpolation: pure geometry in σ, delegated to the base grid ------

_interp_to_face!(out::Field{FaceX}, f::Field{CellCenter}, g::TerrainFollowingGrid) =
    _interp_to_face!(out, f, g.base)
_interp_to_face!(out::Field{FaceY}, f::Field{CellCenter}, g::TerrainFollowingGrid) =
    _interp_to_face!(out, f, g.base)
"""
    _interp_to_face!(out::Field{FaceZ}, f, g::TerrainFollowingGrid)

Volume-weighted σ-interpolation, delegated to the underlying
[`StretchedGrid`](@ref). The Jacobian cancels: physical thicknesses along a
column are `J·Δσ` with the *same* `J`, so the weights are the σ-space ones.
"""
_interp_to_face!(out::Field{FaceZ}, f::Field{CellCenter}, g::TerrainFollowingGrid) =
    _interp_to_face!(out, f, g.base)

_interp_to_center!(out::Field{CellCenter}, u::Field{FaceX}, g::TerrainFollowingGrid) =
    _interp_to_center!(out, u, g.base)
_interp_to_center!(out::Field{CellCenter}, v::Field{FaceY}, g::TerrainFollowingGrid) =
    _interp_to_center!(out, v, g.base)
_interp_to_center!(out::Field{CellCenter}, w::Field{FaceZ}, g::TerrainFollowingGrid) =
    _interp_to_center!(out, w, g.base)

# --- divergence: conservative contravariant form -------------------------

"""
    _divergence!(out, u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                 g::TerrainFollowingGrid)

Conservative Gal-Chen divergence (GC5):

```math
\\nabla\\!\\cdot\\!\\mathbf u = \\frac{1}{J}\\left[
 \\frac{(Ju)_{i+1}-(Ju)_i}{\\Delta x} +
 \\frac{(Jv)_{j+1}-(Jv)_j}{\\Delta y} +
 \\frac{\\Omega_{k+1}-\\Omega_k}{\\Delta\\sigma_c[k]}\\right],
```

with the contravariant vertical mass flux

```math
\\Omega  =  J\\tilde w  =  w + J\\left(\\frac{\\partial\\sigma}{\\partial x}\\bar u
 + \\frac{\\partial\\sigma}{\\partial y}\\bar v\\right)\\Big|_{\\sigma\\text{-face}} ,
```

where `ū`, `v̄` are the horizontal velocities averaged to the cell centre
in their own direction and then volume-weighted to the σ-face. `Ω` is the
flux *through* the sloping coordinate surface: over a hill, horizontal
flow crossing the tilted surface contributes to the vertical flux
divergence, which is exactly the term the naive Cartesian formula misses.
Because the discrete form telescopes, the pressure projection remains
mass-conserving cell by cell. For `h ≡ 0`, `Ω ≡ w` and the kernel is
bit-identical to the [`StretchedGrid`](@ref) one.
"""
function _divergence!(out::Field{CellCenter},
                      u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                      g::TerrainFollowingGrid{T}) where {T}
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc = g.base.dzc
    ii, jj, kk = interior(g)
    k1, k2 = first(kk), last(kk)
    Ω = Vector{T}(undef, k2 - k1 + 2)
    @inbounds for j in jj, i in ii
        for k in k1:k2+1
            wlo = dzc[k]   / (dzc[k-1] + dzc[k])
            whi = dzc[k-1] / (dzc[k-1] + dzc[k])
            ucm = (u.data[i,j,k-1] + u.data[i+1,j,k-1]) / 2
            uck = (u.data[i,j,k]   + u.data[i+1,j,k])   / 2
            vcm = (v.data[i,j,k-1] + v.data[i,j+1,k-1]) / 2
            vck = (v.data[i,j,k]   + v.data[i,j+1,k])   / 2
            ū = wlo * ucm + whi * uck
            v̄ = wlo * vcm + whi * vck
            Ω[k - k1 + 1] = w.data[i,j,k] +
                            g.Jzf[i,j,k] * (g.dσdx_zf[i,j,k] * ū + g.dσdy_zf[i,j,k] * v̄)
        end
        for k in k1:k2
            hx_flux = (g.Jx[i+1,j,k] * u.data[i+1,j,k] - g.Jx[i,j,k] * u.data[i,j,k]) * idx
            hy_flux = (g.Jy[i,j+1,k] * v.data[i,j+1,k] - g.Jy[i,j,k] * v.data[i,j,k]) * idy
            vz_flux = (Ω[k - k1 + 2] - Ω[k - k1 + 1]) / dzc[k]
            out.data[i,j,k] = (hx_flux + hy_flux + vz_flux) / g.J[i,j,k]
        end
    end
    return out
end

"""
    _divergence!(out, u, v, w, g::TerrainFollowingGrid)   # all at CellCenter

Collocated diagnostic divergence: the sum of the three metric-corrected
collocated derivatives (non-conservative; for diagnostics only).
"""
function _divergence!(out::Field{CellCenter},
                      u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                      g::TerrainFollowingGrid)
    inv2dx = 1 / (2g.dx); inv2dy = 1 / (2g.dy)
    dzf = g.base.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        dudx = (u.data[i+1,j,k] - u.data[i-1,j,k]) * inv2dx +
               g.dσdx[i,j,k] * _dσ_c(u.data, i, j, k, dzf)
        dvdy = (v.data[i,j+1,k] - v.data[i,j-1,k]) * inv2dy +
               g.dσdy[i,j,k] * _dσ_c(v.data, i, j, k, dzf)
        dwdz = _dσ_c(w.data, i, j, k, dzf) / g.J[i,j,k]
        out.data[i,j,k] = dudx + dvdy + dwdz
    end
    return out
end

# --- Laplacian: div(grad) with the full contravariant metric -------------

"""
    _laplacian!(out::Field{L}, f::Field{L}, g::TerrainFollowingGrid) where {L}

Laplace–Beltrami operator of the Gal-Chen transformation, assembled in
conservative flux form so that it is self-adjoint (needed by the CG
pressure solve):

```math
\\nabla^2 f = \\frac{1}{J}\\Big[
 \\partial_x\\!\\big(J F^x\\big) + \\partial_y\\!\\big(J F^y\\big)
 + \\partial_\\sigma\\!\\big(J F^\\sigma\\big)\\Big],
```
```math
F^x = \\partial_x f + \\sigma_x \\partial_\\sigma f, \\quad
F^y = \\partial_y f + \\sigma_y \\partial_\\sigma f, \\quad
F^\\sigma = \\sigma_x \\partial_x f + \\sigma_y \\partial_y f
          + \\big(\\sigma_x^2+\\sigma_y^2+J^{-2}\\big)\\partial_\\sigma f ,
```

i.e. the contravariant metric (GC6). Fluxes are built on the faces they
cross (`F^x` on x-faces with `σ_x` co-located there, `F^σ` on σ-faces),
then differenced back to the centre — the same discrete `div(grad)`
pairing of `dzf`/`dzc` as on a [`StretchedGrid`](@ref). For `h ≡ 0` the
cross terms vanish, `J = 1`, `G^{σσ} = 1`, and the result is identical to
the stretched-grid Laplacian.

!!! note "Face-located inputs"
    For `L ∈ {FaceX, FaceY, FaceZ}` the cell-centre metrics are reused
    rather than re-interpolated to the staggered position; that is an
    `O(Δ·∂²h)` approximation of the metric (exact for flat or
    constant-slope terrain) and is intended for diffusion-type terms, not
    for the pressure Poisson operator, which acts on cell-centred fields.

!!! note "Allocation"
    Three flux scratch arrays of `totalsize(g)` are allocated per call.
    Acceptable for diagnostics and once-per-substep diffusion; a
    scratch-carrying in-place variant is deferred.
"""
function _laplacian!(out::Field{L}, f::Field{L}, g::TerrainFollowingGrid{T}) where {L<:Location, T}
    dx, dy = g.dx, g.dy
    dzc, dzf = g.base.dzc, g.base.dzf
    ii, jj, kk = interior(g)
    i1, i2 = first(ii), last(ii)
    j1, j2 = first(jj), last(jj)
    k1, k2 = first(kk), last(kk)
    d = f.data
    Fx = zeros(T, Base.size(d)); Fy = zeros(T, Base.size(d)); Fs = zeros(T, Base.size(d))

    @inbounds for k in k1:k2, j in j1:j2, i in i1:i2+1
        dfdσ = (_dσ_c(d, i-1, j, k, dzf) + _dσ_c(d, i, j, k, dzf)) / 2
        Fx[i,j,k] = g.Jx[i,j,k] * ((d[i,j,k] - d[i-1,j,k]) / dx +
                                   g.dσdx_xf[i,j,k] * dfdσ)
    end
    @inbounds for k in k1:k2, j in j1:j2+1, i in i1:i2
        dfdσ = (_dσ_c(d, i, j-1, k, dzf) + _dσ_c(d, i, j, k, dzf)) / 2
        Fy[i,j,k] = g.Jy[i,j,k] * ((d[i,j,k] - d[i,j-1,k]) / dy +
                                   g.dσdy_yf[i,j,k] * dfdσ)
    end
    @inbounds for k in k1:k2+1, j in j1:j2, i in i1:i2
        wlo = dzc[k]   / (dzc[k-1] + dzc[k])
        whi = dzc[k-1] / (dzc[k-1] + dzc[k])
        dfdx = wlo * (d[i+1,j,k-1] - d[i-1,j,k-1]) / (2dx) +
               whi * (d[i+1,j,k]   - d[i-1,j,k])   / (2dx)
        dfdy = wlo * (d[i,j+1,k-1] - d[i,j-1,k-1]) / (2dy) +
               whi * (d[i,j+1,k]   - d[i,j-1,k])   / (2dy)
        dfdσ = (d[i,j,k] - d[i,j,k-1]) / dzf[k]
        sx, sy = g.dσdx_zf[i,j,k], g.dσdy_zf[i,j,k]
        Jf = g.Jzf[i,j,k]
        Gσσ = sx * sx + sy * sy + 1 / (Jf * Jf)
        Fs[i,j,k] = Jf * (sx * dfdx + sy * dfdy + Gσσ * dfdσ)
    end
    @inbounds for k in k1:k2, j in j1:j2, i in i1:i2
        out.data[i,j,k] = ((Fx[i+1,j,k] - Fx[i,j,k]) / dx +
                           (Fy[i,j+1,k] - Fy[i,j,k]) / dy +
                           (Fs[i,j,k+1] - Fs[i,j,k]) / dzc[k]) / g.J[i,j,k]
    end
    return out
end

# --- physics kernels carrying the terrain metric ------------------------

"""
    _add_surface_flux!(tϕ, F_bot, F_top, g::TerrainFollowingGrid)

Wall fluxes divided by the **physical** thickness `J·Δσ_c` of the first
and last interior cell, so a cell sitting on high ground (small `J`, thin
cell) receives the correspondingly larger tendency.
"""
function _add_surface_flux!(tϕ::Field{CellCenter}, F_bot::Real, F_top::Real,
                            g::TerrainFollowingGrid)
    ii, jj, kk = interior(g)
    k_bot, k_top = first(kk), last(kk)
    dzc = g.base.dzc
    @inbounds for j in jj, i in ii
        tϕ.data[i,j,k_bot] += F_bot / (g.J[i,j,k_bot] * dzc[k_bot])
        tϕ.data[i,j,k_top] -= F_top / (g.J[i,j,k_top] * dzc[k_top])
    end
    return tϕ
end

"""
    _add_sponge!(tf, f, α_max, z_start, z_top, f_ref, g::TerrainFollowingGrid)

Rayleigh damping keyed on the **physical** height of each sample, so the
sponge is a horizontal slab in real space even though the coordinate
surfaces below it are tilted by the orography.
"""
function _add_sponge!(tf::Field{L}, f::Field{L},
                      α_max::Real, z_start::Real, z_top::Real,
                      f_ref::Real, g::TerrainFollowingGrid) where {L<:Location}
    zs = physical_heights(g, L())
    ii, jj, kk = interior(g)
    prof = sponge_profile(α_max, z_start, z_top)   # the shared damping profile
    @inbounds for k in kk, j in jj, i in ii
        z = zs[i,j,k]
        in_damping(prof, z) || continue
        α = damping_rate(prof, z)
        tf.data[i,j,k] -= α * (f.data[i,j,k] - f_ref)
    end
    return tf
end

"""
    _add_buoyancy!(tw, θ, θ₀, g_accel, g::TerrainFollowingGrid)

Boussinesq buoyancy at σ-faces with volume-weighted σ-interpolation of θ.
Buoyancy acts along true vertical; in a σ-coordinate the `w`-equation is
the one for the physical vertical velocity, so no metric factor enters
here — the metric appears in the pressure gradient instead.
"""
function _add_buoyancy!(tw::Field{FaceZ}, θ::Field{CellCenter},
                        θ₀::Real, g_accel::Real, g::TerrainFollowingGrid)
    return _add_buoyancy!(tw, θ, θ₀, g_accel, g.base)
end

# =========================================================================
# Hydrostatic balance diagnostic
# =========================================================================

# Physical distance between the centres straddling σ-face k, per column.
@inline _face_spacing(g::StretchedGrid, i, j, k) = g.dzf[k]
@inline _face_spacing(g::RegularGrid, i, j, k)   = g.dz
@inline _face_spacing(g::TerrainFollowingGrid, i, j, k) = g.Jzf[i,j,k] * g.base.dzf[k]

# Volume weights (lower, upper) for interpolating a centred quantity to face k.
@inline function _face_weights(g::StretchedGrid, k)
    dzc = g.dzc
    s = dzc[k-1] + dzc[k]
    return (dzc[k] / s, dzc[k-1] / s)
end
@inline _face_weights(g::RegularGrid, k) = (0.5, 0.5)
@inline _face_weights(g::TerrainFollowingGrid, k) = _face_weights(g.base, k)

"""
    hydrostatic_balance_error(θ, p, grid = mesh(p);
                              g_accel = 9.80665, R_d = 287.05, c_p = 1004.6,
                              p_ref = 100000.0, relative = false)

Maximum residual of the hydrostatic relation

```math
\\mathcal{R}  =  \\frac{\\partial p}{\\partial z} + \\rho g ,
\\qquad
\\rho = \\frac{p}{R_d T}, \\quad T = \\theta\\,\\Pi, \\quad
\\Pi = \\left(\\frac{p}{p_{\\text{ref}}}\\right)^{R_d/c_p} ,
```

evaluated at every interior z-face of a [`StretchedGrid`](@ref),
[`TerrainFollowingGrid`](@ref) or `RegularGrid`, from cell-centred
potential temperature `θ` and pressure `p`.

This is *the* validation quantity for a new vertical coordinate. An
atmosphere at rest satisfies `R = 0` exactly in the continuum, so whatever
the discretisation leaves behind is pure numerical error — and on a
stretched or terrain-following grid it is precisely the error produced by
using the wrong metric (`dzc` instead of `dzf`), by naive `½` averaging of
density onto faces, or by dropping a terrain cross-term. A first-order
metric bug shows up here as a residual thousands of times larger than the
second-order one.

Discretisation, chosen to mirror the model's own pressure-gradient term:

  * `∂p/∂z` at face `k` is `(p_k - p_{k-1})/Δz`, with `Δz` the **physical**
    centre spacing (`dzf[k]`, or `J·Δσ_f[k]` over terrain — the two
    centres share a column, so no cross-term arises);
  * `ρ` is interpolated to the face with the volume weights of
    `_interp_to_face!`.

Keyword `relative = true` divides each residual by the local `|ρ g|`,
giving a dimensionless number that is directly comparable across
resolutions and domains. Returns a scalar.
"""
function hydrostatic_balance_error(θ::Field, p::Field, grid::AbstractMesh = mesh(p);
                                   g_accel::Real = 9.80665,
                                   R_d::Real = 287.05,
                                   c_p::Real = 1004.6,
                                   p_ref::Real = 100000.0,
                                   relative::Bool = false)
    T = eltype(p)
    κ = T(R_d / c_p)
    ii, jj, kk = interior(grid)
    ρ = similar(p.data)
    @inbounds for k in first(kk)-1:last(kk)+1, j in jj, i in ii
        pv = p.data[i,j,k]
        pv > 0 || throw(ArgumentError("hydrostatic_balance_error needs p > 0; " *
                                      "got p[$i,$j,$k] = $pv"))
        Π = (pv / T(p_ref))^κ
        Tabs = θ.data[i,j,k] * Π
        ρ[i,j,k] = pv / (T(R_d) * Tabs)
    end
    worst = zero(T)
    @inbounds for k in first(kk)+1:last(kk), j in jj, i in ii
        wlo, whi = _face_weights(grid, k)
        ρf = wlo * ρ[i,j,k-1] + whi * ρ[i,j,k]
        dpdz = (p.data[i,j,k] - p.data[i,j,k-1]) / _face_spacing(grid, i, j, k)
        r = dpdz + ρf * T(g_accel)
        relative && (r /= ρf * T(g_accel))
        worst = max(worst, abs(r))
    end
    return worst
end

end # module
