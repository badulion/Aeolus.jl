module StretchedGrids

# =========================================================================
# Phase 24 — Stretched and terrain-following vertical coordinates.
# =========================================================================
#
# The LES core first shipped a single mesh type, `RegularGrid`: uniform in all three
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
# `Operators` façade work on them unchanged. A `StretchedGrid` runs the
# one set of kernels of `Operators.FiniteVolume`, which read its per-level
# vertical metrics (`grid_metrics`, KA-d #299); a `TerrainFollowingGrid`
# adds its own methods to those private kernels (see the `import` block
# below) — `Operators.∂z(out, f)` forwards to `_∂z!(out, f, mesh(f))`, so a
# field carrying either mesh picks up the metric-correct stencil purely by
# multiple dispatch. No call site in the rest of Aeolus changes.
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
                                 _add_surface_flux!, _add_sponge!, _add_buoyancy!,
                                 damping_coordinates, grid_metrics
using ..Operators.FiniteVolume: sponge_profile, in_damping, damping_rate, _add_damping!,
                                LevelZ, GridMetrics
using ..Backends: Backends, launch!

export StretchedGrid, TerrainFollowingGrid,
       terrain_following_grid,
       grid_stretching_ratio,
       zfaces, zcenters, cell_thickness, center_spacing,
       metric_jacobian, terrain_height, physical_heights,
       set_physical!,
       hydrostatic_balance_error,
       surface_refined_faces, StretchedPoissonSolver

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
struct StretchedGrid{T<:AbstractFloat, Z<:LevelZ{T}} <: AbstractMesh{T}
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
    zm::Z                # the kernels' per-level metrics (`level_metrics`, KA-d #299)
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

"""
    surface_refined_faces(; dz_surface, z_uniform, z_top, ratio = 1.08,
                          dz_max = Inf, z0 = 0) -> Vector{Float64}

Interior face heights (length `Nz + 1`, from `z0` to `z_top`) of the
standard ABL vertical grid: uniform `dz_surface` from `z0` up to
`z_uniform`, then geometric growth by `ratio` per level, capped at
`dz_max`, up to `z_top`. Pass the result as `stretching = (:custom, zf)`
with `size[3] = length(zf) - 1` and `extent[3] = z_top - z0`.

`z_uniform - z0` must be a whole number of `dz_surface` cells. The stretched
part takes the smallest number of cells that keeps every cell `≤ dz_max`
and is then scaled by one common factor `f ≤ 1` so that the last face
lands exactly on `z_top` (`f ≥ 1 - dz_max/(z_top - z_uniform)`; the ratio
between stretched cells stays `ratio`, the first stretched cell is
`f·ratio·dz_surface`).

This is the layout of the Berlin nesting parent (#246): `dz_surface = 25`,
uniform to `z_uniform ≥ 500` m so that the ~400 m child sits in the
uniform region, stretched to `z_top = 3000` m. With `ratio = 1.08`,
`dz_max = 100` it has 53 levels (20 uniform + 33 stretched). Keep
`ratio ≲ 1.1` (see [`grid_stretching_ratio`](@ref)).
"""
function surface_refined_faces(; dz_surface::Real, z_uniform::Real, z_top::Real,
                               ratio::Real = 1.08, dz_max::Real = Inf, z0::Real = 0)
    dz0 = Float64(dz_surface); zu = Float64(z_uniform); zt = Float64(z_top)
    zb = Float64(z0); r = Float64(ratio); dzm = Float64(dz_max)
    dz0 > 0 || throw(ArgumentError("dz_surface must be > 0, got $dz0"))
    r ≥ 1 || throw(ArgumentError("ratio must be ≥ 1, got $r"))
    dzm ≥ dz0 || throw(ArgumentError("dz_max ($dzm) must be ≥ dz_surface ($dz0)"))
    zb ≤ zu < zt || throw(ArgumentError("need z0 ≤ z_uniform < z_top, got $zb, $zu, $zt"))
    nu = round(Int, (zu - zb) / dz0)
    isapprox(zb + nu * dz0, zu; atol = 1e-9 * max(1.0, abs(zu))) ||
        throw(ArgumentError("z_uniform - z0 = $(zu - zb) is not a whole number of " *
                            "dz_surface = $dz0 cells"))
    zf = [zb + (k - 1) * dz0 for k in 1:nu+1]
    zf[end] = zu
    # natural geometric growth until the column reaches z_top
    dzs = Float64[]
    dz = dz0; acc = 0.0
    while acc < zt - zu - 1e-9 * zt
        dz = min(dz * r, dzm)
        push!(dzs, dz); acc += dz
    end
    f = (zt - zu) / acc                 # ≤ 1: the overshoot of the last cell is shared
    for d in dzs
        push!(zf, zf[end] + f * d)
    end
    zf[end] = zt
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

    dx, dy = Lx / Nx, Ly / Ny
    return StretchedGrid(Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz,
                         x0, y0, z0, dx, dy,
                         zf, zc, dzc, dzf, kind, level_metrics(dx, dy, zf, zc, dzc, dzf))
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
# Finite-volume stencils on a StretchedGrid (KA-d, #299)
# =========================================================================
#
# The StretchedGrid has no kernels of its own: every finite-volume operator
# (derivatives, Laplacian, divergence, interpolation, surface flux, sponge,
# buoyancy, momentum and scalar advection, Smagorinsky, eddy viscosity and
# diffusion) is the one kernel of `Operators.FiniteVolume`, which reads the
# grid only through its metrics. A StretchedGrid's metrics are the
# per-level vectors of `level_metrics` (a `FiniteVolume.LevelZ`), built
# once with the grid and returned by `grid_metrics`. Before KA-d the
# Phase-24 stencils here and the #151 / #276 Boussinesq kernels of
# src/stretched/boussinesq.jl were copies of the RegularGrid kernels with
# the vertical metric swapped in; the folded kernels give the same bits
# (test/ka_fv.jl (removed)).

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
    level_metrics(dx, dy, zf, zc, dzc, dzf) -> FiniteVolume.LevelZ

The per-level vertical metrics of a stretched column (storage indices,
halos included) in the form the finite-volume kernels read them: the
reciprocal thicknesses, the interpolation weights, the three-point
derivative weights and the Smagorinsky filter widths. Each entry is the
expression the pre-port StretchedGrid kernels evaluated per level, so the
kernels give the same bits; an entry whose stencil leaves the storage
(the outermost halo levels) is zero.
"""
function level_metrics(dx::T, dy::T, zf::Vector{T}, zc::Vector{T}, dzc::Vector{T},
                       dzf::Vector{T}) where {T}
    n = length(dzc)
    z = zeros(T, n)
    idzc = T[1 / dzc[k] for k in 1:n]
    idzf = T[1 / dzf[k] for k in 1:n+1]
    idz_off = T[2 / (dzf[k] + dzf[k+1]) for k in 1:n]
    wlo = copy(z); whi = copy(z); alo = copy(z); ahi = copy(z)
    d3m = copy(z); d3c = copy(z); d3p = copy(z)
    for k in 2:n
        wlo[k] = dzc[k]   / (dzc[k-1] + dzc[k])
        whi[k] = dzc[k-1] / (dzc[k-1] + dzc[k])
        s = dzc[k-1] + dzc[k]
        alo[k] = dzc[k-1] / s
        ahi[k] = dzc[k]   / s
        d3m[k], d3c[k], d3p[k] = _dz_weights(dzf[k], dzf[k+1])
    end
    clo = T[(zf[k+1] - zc[k]) / dzc[k] for k in 1:n]
    chi = T[(zc[k] - zf[k])   / dzc[k] for k in 1:n]
    Δ = [(dx * dy * dzc[k]) ^ (1/3) for k in 1:n]
    return LevelZ{T, Vector{T}, typeof(Δ)}(idzc, idzf, idz_off, wlo, whi, alo, ahi, clo, chi,
                                           d3m, d3c, d3p, Δ)
end

grid_metrics(g::StretchedGrid) = GridMetrics(g.dx, g.dy, g.zm)


_zloc_coords(g::StretchedGrid, ::FaceZ) = g.zf
_zloc_coords(g::StretchedGrid, ::Location) = g.zc

# The heights of the interior levels the damping kernel samples a vertical
# DampingProfile at, on a StretchedGrid (the generic is in FiniteVolume).
damping_coordinates(g::StretchedGrid, loc::Location) =
    _zloc_coords(g, loc)[g.Hz+1:g.Hz+g.Nz]

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
struct TerrainFollowingGrid{T<:AbstractFloat, B<:StretchedGrid{T}} <: AbstractMesh{T}
    base::B
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

    return TerrainFollowingGrid{T, typeof(base)}(base,
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

# One launch over the whole storage (halos included); `fun` is called once
# per sample, on any thread (KA-d, #299): it must not mutate shared state.
@inline _b_set_physical!(i, j, k, o, fun::F, xs, ys, zs::AbstractVector) where {F} =
    (@inbounds o[i, j, k] = fun(xs[i], ys[j], zs[k]); nothing)
@inline _b_set_physical!(i, j, k, o, fun::F, xs, ys, zs::AbstractArray{<:Any,3}) where {F} =
    (@inbounds o[i, j, k] = fun(xs[i], ys[j], zs[i, j, k]); nothing)

function _set_physical!(f::Field, fun::F, g::Union{StretchedGrid,RegularGrid}) where {F}
    loc = Fields.location(f)
    sg  = g isa StretchedGrid ? g : StretchedGrid(g)
    xs  = _xcoords(sg, loc); ys = _ycoords(sg, loc); zs = _zcoords(sg, loc)
    r = (1:length(xs), 1:length(ys), 1:length(zs))
    checkbounds(f.data, r...)
    launch!(Backends.backend(f.data), _b_set_physical!, r, f.data, fun, xs, ys, zs)
    return f
end

function _set_physical!(f::Field, fun::F, g::TerrainFollowingGrid) where {F}
    loc = Fields.location(f)
    xs  = _xcoords(g, loc); ys = _ycoords(g, loc)
    zs  = physical_heights(g, loc)
    r = (1:length(xs), 1:length(ys), 1:size(zs, 3))
    checkbounds(f.data, r...)
    launch!(Backends.backend(f.data), _b_set_physical!, r, f.data, fun, xs, ys, zs)
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
@inline _b_tf_∂z_cf!(i, j, k, o, f, idzf, Jzf) =
    (@inbounds o[i,j,k] = (f[i,j,k] - f[i,j,k-1]) * idzf[k] / Jzf[i,j,k]; nothing)

function _∂z!(out::Field{FaceZ}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂z_cf!, interior(g), out.data, f.data,
            g.base.zm.idzf, g.Jzf)
    return out
end

"""
    _∂z!(out::Field{CellCenter}, w::Field{FaceZ}, g::TerrainFollowingGrid)

`∂w/∂z = (1/J)(w_{k+1}-w_k)/Δσ_c[k]`. This is the plain vertical
derivative of `w`; the *divergence* additionally needs the contravariant
flux `J w̃` (see `_divergence!`).
"""
@inline _b_tf_∂z_fc!(i, j, k, o, w, idzc, J) =
    (@inbounds o[i,j,k] = (w[i,j,k+1] - w[i,j,k]) * idzc[k] / J[i,j,k]; nothing)

function _∂z!(out::Field{CellCenter}, w::Field{FaceZ}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂z_fc!, interior(g), out.data, w.data,
            g.base.zm.idzc, g.J)
    return out
end

"""
    _∂z!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)

Collocated `∂f/∂z = (1/J) ∂f/∂σ` with the unequal-spacing three-point
σ-derivative.
"""
@inline _b_tf_∂z_cc!(i, j, k, o, f, dzf, J) =
    (@inbounds o[i,j,k] = _dσ_c(f, i, j, k, dzf) / J[i,j,k]; nothing)

function _∂z!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂z_cc!, interior(g), out.data, f.data,
            g.base.dzf, g.J)
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
@inline _b_tf_∂x_cc!(i, j, k, o, f, inv2dx, s, dzf) =
    (@inbounds o[i,j,k] = (f[i+1,j,k] - f[i-1,j,k]) * inv2dx + s[i,j,k] * _dσ_c(f, i, j, k, dzf); nothing)
@inline _b_tf_∂y_cc!(i, j, k, o, f, inv2dy, s, dzf) =
    (@inbounds o[i,j,k] = (f[i,j+1,k] - f[i,j-1,k]) * inv2dy + s[i,j,k] * _dσ_c(f, i, j, k, dzf); nothing)

function _∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂x_cc!, interior(g), out.data, f.data,
            1 / (2g.dx), g.dσdx, g.base.dzf)
    return out
end

@doc (@doc _∂x!)
function _∂y!(out::Field{CellCenter}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂y_cc!, interior(g), out.data, f.data,
            1 / (2g.dy), g.dσdy, g.base.dzf)
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
@inline function _b_tf_∂x_cf!(i, j, k, o, f, idx, s, dzf)
    @inbounds begin
        dfdσ = (_dσ_c(f, i-1, j, k, dzf) + _dσ_c(f, i, j, k, dzf)) / 2
        o[i,j,k] = (f[i,j,k] - f[i-1,j,k]) * idx + s[i,j,k] * dfdσ
    end
    return nothing
end
@inline function _b_tf_∂y_cf!(i, j, k, o, f, idy, s, dzf)
    @inbounds begin
        dfdσ = (_dσ_c(f, i, j-1, k, dzf) + _dσ_c(f, i, j, k, dzf)) / 2
        o[i,j,k] = (f[i,j,k] - f[i,j-1,k]) * idy + s[i,j,k] * dfdσ
    end
    return nothing
end

function _∂x!(out::Field{FaceX}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂x_cf!, interior(g), out.data, f.data,
            1 / g.dx, g.dσdx_xf, g.base.dzf)
    return out
end

@doc (@doc _∂x!)
function _∂y!(out::Field{FaceY}, f::Field{CellCenter}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂y_cf!, interior(g), out.data, f.data,
            1 / g.dy, g.dσdy_yf, g.base.dzf)
    return out
end

"""
    _∂x!(out::Field{CellCenter}, u::Field{FaceX}, g::TerrainFollowingGrid)

Face → centre horizontal derivative of `u` at constant height. The
σ-derivative of `u` is taken after averaging `u` to the cell centre in
`x`, keeping the stencil symmetric about the cell.
"""
@inline _b_tf_∂x_fc!(i, j, k, o, u, idx, s, dzf) =
    (@inbounds o[i,j,k] = (u[i+1,j,k] - u[i,j,k]) * idx + s[i,j,k] * _dσ_cx(u, i, j, k, dzf); nothing)
@inline _b_tf_∂y_fc!(i, j, k, o, v, idy, s, dzf) =
    (@inbounds o[i,j,k] = (v[i,j+1,k] - v[i,j,k]) * idy + s[i,j,k] * _dσ_cy(v, i, j, k, dzf); nothing)

function _∂x!(out::Field{CellCenter}, u::Field{FaceX}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂x_fc!, interior(g), out.data, u.data,
            1 / g.dx, g.dσdx, g.base.dzf)
    return out
end

@doc (@doc _∂x!)
function _∂y!(out::Field{CellCenter}, v::Field{FaceY}, g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_∂y_fc!, interior(g), out.data, v.data,
            1 / g.dy, g.dσdy, g.base.dzf)
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
# The contravariant flux Ω through σ-face k of column (i, j). Each work
# item evaluates the Ω of its two faces (a face is shared by two cells, so
# it is computed twice, with the same arithmetic as the old column buffer).
@inline function _tf_Ω(u, v, w, Jzf, sx, sy, dzc, i, j, k)
    @inbounds begin
        wlo = dzc[k]   / (dzc[k-1] + dzc[k])
        whi = dzc[k-1] / (dzc[k-1] + dzc[k])
        ucm = (u[i,j,k-1] + u[i+1,j,k-1]) / 2
        uck = (u[i,j,k]   + u[i+1,j,k])   / 2
        vcm = (v[i,j,k-1] + v[i,j+1,k-1]) / 2
        vck = (v[i,j,k]   + v[i,j+1,k])   / 2
        ū = wlo * ucm + whi * uck
        v̄ = wlo * vcm + whi * vck
        return w[i,j,k] + Jzf[i,j,k] * (sx[i,j,k] * ū + sy[i,j,k] * v̄)
    end
end

@inline function _b_tf_divergence!(i, j, k, o, u, v, w, g, idx, idy)
    @inbounds begin
        T = eltype(o)
        dzc = g.base.dzc
        Ωb = convert(T, _tf_Ω(u, v, w, g.Jzf, g.dσdx_zf, g.dσdy_zf, dzc, i, j, k))
        Ωt = convert(T, _tf_Ω(u, v, w, g.Jzf, g.dσdx_zf, g.dσdy_zf, dzc, i, j, k + 1))
        hx_flux = (g.Jx[i+1,j,k] * u[i+1,j,k] - g.Jx[i,j,k] * u[i,j,k]) * idx
        hy_flux = (g.Jy[i,j+1,k] * v[i,j+1,k] - g.Jy[i,j,k] * v[i,j,k]) * idy
        vz_flux = (Ωt - Ωb) / dzc[k]
        o[i,j,k] = (hx_flux + hy_flux + vz_flux) / g.J[i,j,k]
    end
    return nothing
end

function _divergence!(out::Field{CellCenter},
                      u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                      g::TerrainFollowingGrid{T}) where {T}
    launch!(Backends.backend(out.data), _b_tf_divergence!, interior(g), out.data,
            u.data, v.data, w.data, g, 1 / g.dx, 1 / g.dy)
    return out
end

"""
    _divergence!(out, u, v, w, g::TerrainFollowingGrid)   # all at CellCenter

Collocated diagnostic divergence: the sum of the three metric-corrected
collocated derivatives (non-conservative; for diagnostics only).
"""
@inline function _b_tf_divergence_cc!(i, j, k, o, u, v, w, g, inv2dx, inv2dy)
    @inbounds begin
        dzf = g.base.dzf
        dudx = (u[i+1,j,k] - u[i-1,j,k]) * inv2dx + g.dσdx[i,j,k] * _dσ_c(u, i, j, k, dzf)
        dvdy = (v[i,j+1,k] - v[i,j-1,k]) * inv2dy + g.dσdy[i,j,k] * _dσ_c(v, i, j, k, dzf)
        dwdz = _dσ_c(w, i, j, k, dzf) / g.J[i,j,k]
        o[i,j,k] = dudx + dvdy + dwdz
    end
    return nothing
end

function _divergence!(out::Field{CellCenter},
                      u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                      g::TerrainFollowingGrid)
    launch!(Backends.backend(out.data), _b_tf_divergence_cc!, interior(g), out.data,
            u.data, v.data, w.data, g, 1 / (2g.dx), 1 / (2g.dy))
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
    For `L ∈ {FaceX, FaceY, FaceZ}` the cell-centre terrain metrics (`J`,
    `σ_x`, `σ_y`) are reused rather than re-interpolated to the staggered
    position; that is an `O(Δ·∂²h)` approximation of the metric (exact for
    flat or constant-slope terrain) and is intended for diffusion-type
    terms, not for the pressure Poisson operator, which acts on
    cell-centred fields. The *vertical* σ-metrics do follow the location:
    a z-face field has control volumes of height `Δσ_f[k]` and neighbours
    `Δσ_c` apart (before #151 the centre metrics were used for it too, a
    first-order error on a stretched base grid even over flat terrain).

!!! note "Allocation"
    Three flux scratch arrays of `totalsize(g)` are allocated per call.
    Acceptable for diagnostics and once-per-substep diffusion; a
    scratch-carrying in-place variant is deferred.
"""
@inline function _b_tf_lap_fx!(i, j, k, Fx, d, g, dzf, dx)
    @inbounds begin
        dfdσ = (_dσ_c(d, i-1, j, k, dzf) + _dσ_c(d, i, j, k, dzf)) / 2
        Fx[i,j,k] = g.Jx[i,j,k] * ((d[i,j,k] - d[i-1,j,k]) / dx + g.dσdx_xf[i,j,k] * dfdσ)
    end
    return nothing
end
@inline function _b_tf_lap_fy!(i, j, k, Fy, d, g, dzf, dy)
    @inbounds begin
        dfdσ = (_dσ_c(d, i, j-1, k, dzf) + _dσ_c(d, i, j, k, dzf)) / 2
        Fy[i,j,k] = g.Jy[i,j,k] * ((d[i,j,k] - d[i,j-1,k]) / dy + g.dσdy_yf[i,j,k] * dfdσ)
    end
    return nothing
end
@inline function _b_tf_lap_fs!(i, j, k, Fs, d, g, dzc, dzf, dx, dy)
    @inbounds begin
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
    return nothing
end
@inline function _b_tf_lap_div!(i, j, k, o, Fx, Fy, Fs, J, dzc, dx, dy)
    @inbounds o[i,j,k] = ((Fx[i+1,j,k] - Fx[i,j,k]) / dx +
                          (Fy[i,j+1,k] - Fy[i,j,k]) / dy +
                          (Fs[i,j,k+1] - Fs[i,j,k]) / dzc[k]) / J[i,j,k]
    return nothing
end

function _laplacian!(out::Field{L}, f::Field{L}, g::TerrainFollowingGrid{T}) where {L<:Location, T}
    dx, dy = g.dx, g.dy
    dzc, dzf = _σ_metrics(g.base, L())
    ii, jj, kk = interior(g)
    i1, i2 = first(ii), last(ii)
    j1, j2 = first(jj), last(jj)
    k1, k2 = first(kk), last(kk)
    d = f.data
    Fx = zeros(T, Base.size(d)); Fy = zeros(T, Base.size(d)); Fs = zeros(T, Base.size(d))
    b = Backends.backend(out.data)
    # the three face fluxes, then their divergence (KA-d: four launches)
    launch!(b, _b_tf_lap_fx!, (i1:i2+1, j1:j2, k1:k2), Fx, d, g, dzf, dx)
    launch!(b, _b_tf_lap_fy!, (i1:i2, j1:j2+1, k1:k2), Fy, d, g, dzf, dy)
    launch!(b, _b_tf_lap_fs!, (i1:i2, j1:j2, k1:k2+1), Fs, d, g, dzc, dzf, dx, dy)
    launch!(b, _b_tf_lap_div!, (i1:i2, j1:j2, k1:k2), out.data, Fx, Fy, Fs, g.J, dzc, dx, dy)
    return out
end

# The vertical σ-metrics seen by a field at `L`: (control-volume height,
# spacing to the neighbour below), indexed like `dzc`, `dzf`. Centre-located
# in z (CellCenter, FaceX, FaceY): the grid's own. On z-faces: the volume of
# face k is Δσ_f[k] and its neighbour below sits Δσ_c[k-1] away.
_σ_metrics(b::StretchedGrid, ::Union{CellCenter,FaceX,FaceY}) = (b.dzc, b.dzf)
function _σ_metrics(b::StretchedGrid{T}, ::FaceZ) where {T}
    n = length(b.dzc)
    vol = T[b.dzf[k] for k in 1:n]
    sp  = T[k == 1 ? b.dzc[1] : b.dzc[k-1] for k in 1:n+1]
    return vol, sp
end

# --- physics kernels carrying the terrain metric ------------------------

"""
    _add_surface_flux!(tϕ, F_bot, F_top, g::TerrainFollowingGrid)

Wall fluxes divided by the **physical** thickness `J·Δσ_c` of the first
and last interior cell, so a cell sitting on high ground (small `J`, thin
cell) receives the correspondingly larger tendency.
"""
@inline function _b_tf_surface_flux!(i, j, t, J, F_bot, F_top, dzb, dzt, kb, kt)
    @inbounds t[i,j,kb] += F_bot / (J[i,j,kb] * dzb)
    @inbounds t[i,j,kt] -= F_top / (J[i,j,kt] * dzt)
    return nothing
end

function _add_surface_flux!(tϕ::Field{CellCenter}, F_bot::Real, F_top::Real,
                            g::TerrainFollowingGrid)
    ii, jj, kk = interior(g)
    k_bot, k_top = first(kk), last(kk)
    dzc = g.base.dzc
    launch!(Backends.backend(tϕ.data), _b_tf_surface_flux!, (ii, jj), tϕ.data, g.J,
            F_bot, F_top, dzc[k_bot], dzc[k_top], k_bot, k_top)
    return tϕ
end

"""
    _add_sponge!(tf, f, α_max, z_start, z_top, f_ref, g::TerrainFollowingGrid)

Rayleigh damping keyed on the **physical** height of each sample, so the
sponge is a horizontal slab in real space even though the coordinate
surfaces below it are tilted by the orography. The one damping kernel
`_add_damping!`, sampled at the 3-D physical heights; `f_ref` is a
number, a `Field` or a halo-padded array (a level vector is a target per
σ-level).
"""
function _add_sponge!(tf::Field{L}, f::Field{L},
                      α_max::Real, z_start::Real, z_top::Real,
                      f_ref, g::TerrainFollowingGrid) where {L<:Location}
    return _add_damping!(tf, f, sponge_profile(α_max, z_start, z_top),
                         physical_heights(g, L()), 3, f_ref, g)
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

# #151 — the Boussinesq core and its pressure solve on a StretchedGrid
include("stretched/boussinesq.jl")
include("stretched/poisson.jl")

end # module
