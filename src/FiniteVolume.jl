module FiniteVolume

# Finite-volume kernels. Each kernel dispatches on the mesh in the trailing
# argument AND on the location parameters of its input/output `Field`s.
# This is how staggered vs. collocated stencils are chosen: e.g.
# `_∂x!(out::Field{FaceX}, f::Field{CellCenter}, g)` is the
# pressure-gradient stencil and is a different method from the collocated
# `_∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g)`.
#
# Halos of input fields must be filled before calling these kernels.
#
# Kernel signatures follow the Arakawa C-grid convention (Phase 1a):
#   * u lives on FaceX (west face of each cell)
#   * v lives on FaceY (south face)
#   * w lives on FaceZ (bottom face)
#   * p, θ, tracers on CellCenter.
#
# KA-d (#299): every kernel is a point *body* run by `Backends.launch!`
# (docs/src/design/ka_port.md). The bodies take the grid's *metrics*
# (`GridMetrics`, below), never the grid itself, so one body serves the
# `RegularGrid` (uniform metrics) and the `StretchedGrid` (per-level
# vertical metrics, `StretchedGrids` adds its `grid_metrics` method). The
# StretchedGrid copies of #151 / #276 (src/stretched/boussinesq.jl and the
# Phase-24 stencils of src/StretchedGrids.jl) are folded into these
# kernels: one source of truth per operator.

using ..Grids
using ..Grids: RegularGrid, AbstractMesh, interior, CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields: Field
using ..Operators: _strain_at
using ..Backends: Backends, launch!, launch_weighted!, plane_reduce!, ordered_sum

export AbstractRamp, SinSquaredRamp, NordstromRamp, smooth_step,
       DampingProfile, sponge_profile, in_damping, damping_rate, damping_coordinates

# =========================================================================
# Grid metrics (KA-d, #299) — the contract between a mesh and the kernels
# =========================================================================
#
# x and y are uniform on every mesh these kernels serve; z is uniform on a
# RegularGrid (`UniformZ`) and per level on a StretchedGrid (`LevelZ`,
# vectors indexed by the *storage* level k). A body asks for a metric by
# an @inline accessor and never sees which grid it runs on:
#
#   _idx(m, i), _idy(m, j)    1/Δx, 1/Δy
#   _idz(m, k)                1/Δz_c[k]: across cell k (face → centre;
#                             the control volume of a centre-located field)
#   _idzf(m, k)               1/Δz_f[k]: between centres k-1 and k
#                             (centre → face; the w control volume)
#   _idz_off(m, k)            2/(Δz_f[k] + Δz_f[k+1]): across the two
#                             neighbouring centres (∂u/∂z at a centre)
#   _zface(m, k, lo, hi)      centre → z-face k (geometric weights)
#   _zflux(m, k, lo, hi)      the volume-flux mean of a w control volume's
#                             side face (u, v at k-1 and k)
#   _zcentre(m, k, lo, hi)    z-face → centre k
#   _lapz(m, loc, k, …)       the vertical term of the conservative Laplacian
#   _addz3(m, k, acc, …)      acc + the collocated three-point ∂/∂z
#   _filter_width(m, k)       (Δx Δy Δz_c[k])^{1/3}
#
# Each accessor evaluates the *same floating-point expression* the
# pre-port kernel of that grid evaluated (on a uniform grid `0.5*(a + b)`,
# on a stretched grid the precomputed weight vectors), so every kernel is
# bit for bit the old one on both grids (test/ka_fv.jl (removed)).

"""
    UniformZ

Uniform vertical metrics (a `RegularGrid`): `dz`, `1/dz`, `1/dz²`,
`1/(2dz)` and the Smagorinsky filter width `(Δx Δy Δz)^{1/3}`.
"""
struct UniformZ{T, D}
    dz::T
    idz::T
    idz2::T
    inv2dz::T
    Δ::D
end

"""
    LevelZ

Per-level vertical metrics (a `StretchedGrid`), each a vector over the
storage levels `k` (halos included; an entry whose stencil leaves the
storage is 0):

| field | value at storage level `k` |
| :---- | :------------------------- |
| `idzc` | `1/Δz_c[k]` |
| `idzf` | `1/Δz_f[k]` (length `Nz + 2Hz + 1`) |
| `idz_off` | `2/(Δz_f[k] + Δz_f[k+1])` |
| `wlo`, `whi` | `Δz_c[k]/(Δz_c[k-1]+Δz_c[k])`, `Δz_c[k-1]/(Δz_c[k-1]+Δz_c[k])`: centre → face `k` |
| `alo`, `ahi` | `Δz_c[k-1]/s`, `Δz_c[k]/s`, `s = Δz_c[k-1]+Δz_c[k]`: the side-face flux of a w volume |
| `clo`, `chi` | `(z_f[k+1]-z_c[k])/Δz_c[k]`, `(z_c[k]-z_f[k])/Δz_c[k]`: face → centre `k` |
| `d3m`, `d3c`, `d3p` | the unequal-spacing three-point ∂/∂z weights at centre `k` |
| `Δ` | `(Δx Δy Δz_c[k])^{1/3}` |

Built once with the grid (`StretchedGrids.level_metrics`).
"""
struct LevelZ{T, V<:AbstractVector{T}, D<:AbstractVector}
    idzc::V
    idzf::V
    idz_off::V
    wlo::V
    whi::V
    alo::V
    ahi::V
    clo::V
    chi::V
    d3m::V
    d3c::V
    d3p::V
    Δ::D
end

"""
    GridMetrics(dx, dy, z)

The metrics a finite-volume body reads: the uniform horizontal spacings
(`1/Δx`, `1/Δy`, `1/Δx²`, `1/Δy²`, `1/(2Δx)`, `1/(2Δy)`) and the vertical
metrics `z` ([`UniformZ`](@ref) or [`LevelZ`](@ref)). Obtained from a mesh
by [`grid_metrics`](@ref).
"""
struct GridMetrics{T, Z}
    idx::T
    idy::T
    idx2::T
    idy2::T
    inv2dx::T
    inv2dy::T
    z::Z
end

GridMetrics(dx::T, dy::T, z) where {T} =
    GridMetrics(1 / dx, 1 / dy, 1 / dx^2, 1 / dy^2, 1 / (2dx), 1 / (2dy), z)

"""
    grid_metrics(g) -> GridMetrics

The kernel metrics of mesh `g`: uniform on a `RegularGrid`; per level on
a `StretchedGrid` (method in `StretchedGrids`, cached in the grid). A mesh
without a method (a `TerrainFollowingGrid` has its own kernels) raises a
`MethodError`.
"""
function grid_metrics end

grid_metrics(g::RegularGrid) =
    GridMetrics(g.dx, g.dy, UniformZ(g.dz, 1 / g.dz, 1 / g.dz^2, 1 / (2g.dz),
                                     (g.dx * g.dy * g.dz) ^ (1/3)))

@inline _idx(m::GridMetrics, i) = m.idx
@inline _idy(m::GridMetrics, j) = m.idy
@inline _idz(m::GridMetrics, k) = _idz(m.z, k)
@inline _idzf(m::GridMetrics, k) = _idzf(m.z, k)
@inline _idz_off(m::GridMetrics, k) = _idz_off(m.z, k)
@inline _zface(m::GridMetrics, k, lo, hi) = _zface(m.z, k, lo, hi)
@inline _zflux(m::GridMetrics, k, lo, hi) = _zflux(m.z, k, lo, hi)
@inline _zcentre(m::GridMetrics, k, lo, hi) = _zcentre(m.z, k, lo, hi)
@inline _lapz(m::GridMetrics, loc, k, fm, c, fp) = _lapz(m.z, loc, k, fm, c, fp)
@inline _addz3(m::GridMetrics, k, acc, fm, fc, fp) = _addz3(m.z, k, acc, fm, fc, fp)
@inline _filter_width(m::GridMetrics, k) = _filter_width(m.z, k)

@inline _idz(z::UniformZ, k) = z.idz
@inline _idzf(z::UniformZ, k) = z.idz
@inline _idz_off(z::UniformZ, k) = z.idz
@inline _zface(z::UniformZ, k, lo, hi) = 0.5 * (lo + hi)
@inline _zflux(z::UniformZ, k, lo, hi) = 0.5 * (lo + hi)
@inline _zcentre(z::UniformZ, k, lo, hi) = 0.5 * (lo + hi)
@inline _lapz(z::UniformZ, loc, k, fm, c, fp) = (fp - 2c + fm) * z.idz2
@inline _addz3(z::UniformZ, k, acc, fm, fc, fp) = acc + (fp - fm) * z.inv2dz
@inline _filter_width(z::UniformZ, k) = z.Δ

@inline _idz(z::LevelZ, k) = @inbounds z.idzc[k]
@inline _idzf(z::LevelZ, k) = @inbounds z.idzf[k]
@inline _idz_off(z::LevelZ, k) = @inbounds z.idz_off[k]
@inline _zface(z::LevelZ, k, lo, hi) = @inbounds z.wlo[k] * lo + z.whi[k] * hi
@inline _zflux(z::LevelZ, k, lo, hi) = @inbounds z.alo[k] * lo + z.ahi[k] * hi
@inline _zcentre(z::LevelZ, k, lo, hi) = @inbounds z.clo[k] * lo + z.chi[k] * hi
# a z-face field: control volume Δz_f[k], neighbours Δz_c[k-1] below and Δz_c[k] above
@inline _lapz(z::LevelZ, ::FaceZ, k, fm, c, fp) =
    @inbounds ((fp - c) * z.idzc[k] - (c - fm) * z.idzc[k-1]) * z.idzf[k]
# centre-located in z (CellCenter, FaceX, FaceY): volume Δz_c[k], neighbours Δz_f apart
@inline _lapz(z::LevelZ, ::Location, k, fm, c, fp) =
    @inbounds ((fp - c) * z.idzf[k+1] - (c - fm) * z.idzf[k]) * z.idzc[k]
@inline _addz3(z::LevelZ, k, acc, fm, fc, fp) =
    @inbounds acc + z.d3m[k] * fm + z.d3c[k] * fc + z.d3p[k] * fp
@inline _filter_width(z::LevelZ, k) = @inbounds z.Δ[k]

# The launch of a body over `ranges` on the backend of `out` (the array the
# body writes). Vararg{Any,M}: see the specialisation trap in ka_port.md.
@inline _fv!(body!::F, ranges, out::AbstractArray, args::Vararg{Any,M}) where {F, M} =
    launch!(Backends.backend(out), body!, ranges, out, args...)

# =========================================================================
# First derivatives
# =========================================================================

# --- collocated cell-center → cell-center (2nd-order centered) -----------
# Kept for scalar diagnostics; not used in the Boussinesq core.

@inline _b_∂x_cc!(i, j, k, o, f, m) =
    (@inbounds o[i, j, k] = (f[i+1, j, k] - f[i-1, j, k]) * m.inv2dx; nothing)
@inline _b_∂y_cc!(i, j, k, o, f, m) =
    (@inbounds o[i, j, k] = (f[i, j+1, k] - f[i, j-1, k]) * m.inv2dy; nothing)
# uniform: (f[k+1] - f[k-1])/(2Δz); stretched: the three-point weights
@inline _b_∂z_cc!(i, j, k, o, f, m) =
    (@inbounds o[i, j, k] = _dz3(m.z, k, f[i, j, k-1], f[i, j, k], f[i, j, k+1]); nothing)
@inline _dz3(z::UniformZ, k, fm, fc, fp) = (fp - fm) * z.inv2dz
@inline _dz3(z::LevelZ, k, fm, fc, fp) = @inbounds z.d3m[k] * fm + z.d3c[k] * fc + z.d3p[k] * fp

_∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_∂x_cc!, interior(g), out.data, f.data, grid_metrics(g)); out)
_∂y!(out::Field{CellCenter}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_∂y_cc!, interior(g), out.data, f.data, grid_metrics(g)); out)
_∂z!(out::Field{CellCenter}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_∂z_cc!, interior(g), out.data, f.data, grid_metrics(g)); out)

# --- cell-center → face (pressure gradient etc., 2-point stencil) --------
# west face of cell (i,j,k) sits between (i-1) and (i); grad = (c[i] - c[i-1])/dx.
# z: the face k separates the centres k-1 and k, Δz_f[k] apart.

@inline _b_∂x_cf!(i, j, k, o, f, m) =
    (@inbounds o[i, j, k] = (f[i, j, k] - f[i-1, j, k]) * _idx(m, i); nothing)
@inline _b_∂y_cf!(i, j, k, o, f, m) =
    (@inbounds o[i, j, k] = (f[i, j, k] - f[i, j-1, k]) * _idy(m, j); nothing)
@inline _b_∂z_cf!(i, j, k, o, f, m) =
    (@inbounds o[i, j, k] = (f[i, j, k] - f[i, j, k-1]) * _idzf(m, k); nothing)

_∂x!(out::Field{FaceX}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_∂x_cf!, interior(g), out.data, f.data, grid_metrics(g)); out)
_∂y!(out::Field{FaceY}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_∂y_cf!, interior(g), out.data, f.data, grid_metrics(g)); out)

"""
    _∂z!(out::Field{FaceZ}, f::Field{CellCenter}, g)

Centre → face gradient `(f_k − f_{k−1})/Δz_f[k]`: face `k` separates the
centres `k−1` and `k`, a distance `Δz_f[k]` apart (`Δz` on a uniform grid).
Exact for any linear profile on any face distribution; with the face →
centre divergence (metric `Δz_c`) it composes the second-order
conservative Laplacian.
"""
_∂z!(out::Field{FaceZ}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_∂z_cf!, interior(g), out.data, f.data, grid_metrics(g)); out)

# --- face → cell-center (used by divergence and diagnostics) --------------
# The east face of cell (i,j,k) is index i+1 for u; hence
# ∂u/∂x at cell center = (u[i+1] - u[i]) / dx. In z the control volume of
# cell k has height Δz_c[k] (Gauss: the exact mean over the cell).

@inline _b_∂x_fc!(i, j, k, o, u, m) =
    (@inbounds o[i, j, k] = (u[i+1, j, k] - u[i, j, k]) * _idx(m, i); nothing)
@inline _b_∂y_fc!(i, j, k, o, v, m) =
    (@inbounds o[i, j, k] = (v[i, j+1, k] - v[i, j, k]) * _idy(m, j); nothing)
@inline _b_∂z_fc!(i, j, k, o, w, m) =
    (@inbounds o[i, j, k] = (w[i, j, k+1] - w[i, j, k]) * _idz(m, k); nothing)

_∂x!(out::Field{CellCenter}, u::Field{FaceX}, g::AbstractMesh) =
    (_fv!(_b_∂x_fc!, interior(g), out.data, u.data, grid_metrics(g)); out)
_∂y!(out::Field{CellCenter}, v::Field{FaceY}, g::AbstractMesh) =
    (_fv!(_b_∂y_fc!, interior(g), out.data, v.data, grid_metrics(g)); out)
_∂z!(out::Field{CellCenter}, w::Field{FaceZ}, g::AbstractMesh) =
    (_fv!(_b_∂z_fc!, interior(g), out.data, w.data, grid_metrics(g)); out)

# =========================================================================
# Laplacian — the 7-point stencil, conservative in z.
# The output must live at the same location as the input.
# =========================================================================

# ∇²f at one point (the body below, and the viscous term of the fused
# tendency body of the step cores, SC-d #404)
@inline function _lap_at(i, j, k, f, m, loc)
    @inbounds begin
        c = f[i, j, k]
        r = (f[i+1,j,k] - 2c + f[i-1,j,k]) * m.idx2 +
            (f[i,j+1,k] - 2c + f[i,j-1,k]) * m.idy2 +
            _lapz(m, loc, k, f[i,j,k-1], c, f[i,j,k+1])
    end
    return r
end
@inline _b_laplacian!(i, j, k, o, f, m, loc) =
    (@inbounds o[i, j, k] = _lap_at(i, j, k, f, m, loc); nothing)

"""
    _laplacian!(out::Field{L}, f::Field{L}, g) where {L}

Seven-point Laplacian. Horizontally the uniform stencil; vertically
`(f_{k+1} − 2f_k + f_{k−1})/Δz²` on a uniform grid and, on a stretched
grid, the conservative flux difference

```math
\\left.\\nabla^2_z f\\right|_k = \\frac{1}{V_k}
 \\left[\\frac{f_{k+1}-f_k}{h^+_k} - \\frac{f_k-f_{k-1}}{h^-_k}\\right],
```

with `V = Δz_c[k]`, `h⁺ = Δz_f[k+1]`, `h⁻ = Δz_f[k]` for a field centred
in z (`CellCenter`, `FaceX`, `FaceY`) and `V = Δz_f[k]`, `h⁺ = Δz_c[k]`,
`h⁻ = Δz_c[k−1]` for a z-face field (`w`). That is `div(grad)` with the
gradient on faces and the divergence over the control volume, so the
operator is self-adjoint and negative definite on any face distribution
(which the naive `(f_{k+1} − 2f_k + f_{k−1})/Δz_c²` is not); the z-face
form keeps `w` second order where the grid stretches (#151).
"""
_laplacian!(out::Field{L}, f::Field{L}, g::AbstractMesh) where {L<:Location} =
    (_fv!(_b_laplacian!, interior(g), out.data, f.data, grid_metrics(g), L()); out)

# =========================================================================
# Divergence
# =========================================================================

# ∇·u at cell (i, j, k): also read by the fused step guard of
# TimeIntegration (KA-j, #306), so the guard's |∇·u| is this stencil's
@inline _div_at(i, j, k, u, v, w, m) =
    @inbounds (u[i+1,j,k] - u[i,j,k]) * _idx(m, i) +
              (v[i,j+1,k] - v[i,j,k]) * _idy(m, j) +
              (w[i,j,k+1] - w[i,j,k]) * _idz(m, k)
@inline function _b_divergence!(i, j, k, o, u, v, w, m)
    @inbounds o[i,j,k] = _div_at(i, j, k, u, v, w, m)
    return nothing
end

"""
    _divergence!(out, u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, g)

Arakawa-C divergence
`(u_{i+1}−u_i)/Δx + (v_{j+1}−v_j)/Δy + (w_{k+1}−w_k)/Δz_c[k]`: the net
volume flux through the faces of cell `(i, j, k)` over its volume, so a
discretely solenoidal field has zero divergence to round-off on any face
distribution.
"""
function _divergence!(out::Field{CellCenter},
                      u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                      g::AbstractMesh)
    _fv!(_b_divergence!, interior(g), out.data, u.data, v.data, w.data, grid_metrics(g))
    return out
end

# Collocated fallback (all at cell centers) — used for diagnostics only.
# The vertical term is the three-point ∂/∂z of the collocated `_∂z!`.
@inline function _b_divergence_cc!(i, j, k, o, u, v, w, m)
    @inbounds o[i,j,k] = _addz3(m, k, (u[i+1,j,k] - u[i-1,j,k]) * m.inv2dx +
                                      (v[i,j+1,k] - v[i,j-1,k]) * m.inv2dy,
                                w[i,j,k-1], w[i,j,k], w[i,j,k+1])
    return nothing
end

function _divergence!(out::Field{CellCenter},
                      u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                      g::AbstractMesh)
    _fv!(_b_divergence_cc!, interior(g), out.data, u.data, v.data, w.data, grid_metrics(g))
    return out
end

# =========================================================================
# Interpolation between staggered locations (linear averaging)
# =========================================================================
# Used to bring quantities to a common location for e.g. momentum advection.
# Horizontally plain averages; in z the geometric (volume-weighted) weights
# of the grid (½ on a uniform grid).

@inline _b_c2fx!(i, j, k, o, f, m) = (@inbounds o[i,j,k] = 0.5 * (f[i-1,j,k] + f[i,j,k]); nothing)
@inline _b_c2fy!(i, j, k, o, f, m) = (@inbounds o[i,j,k] = 0.5 * (f[i,j-1,k] + f[i,j,k]); nothing)
@inline _b_c2fz!(i, j, k, o, f, m) = (@inbounds o[i,j,k] = _zface(m, k, f[i,j,k-1], f[i,j,k]); nothing)
@inline _b_fx2c!(i, j, k, o, u, m) = (@inbounds o[i,j,k] = 0.5 * (u[i,j,k] + u[i+1,j,k]); nothing)
@inline _b_fy2c!(i, j, k, o, v, m) = (@inbounds o[i,j,k] = 0.5 * (v[i,j,k] + v[i,j+1,k]); nothing)
@inline _b_fz2c!(i, j, k, o, w, m) = (@inbounds o[i,j,k] = _zcentre(m, k, w[i,j,k], w[i,j,k+1]); nothing)

# center → face
_interp_to_face!(out::Field{FaceX}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_c2fx!, interior(g), out.data, f.data, grid_metrics(g)); out)
_interp_to_face!(out::Field{FaceY}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_c2fy!, interior(g), out.data, f.data, grid_metrics(g)); out)

"""
    _interp_to_face!(out::Field{FaceZ}, f::Field{CellCenter}, g)

Centre → z-face interpolation with the geometric weights
`(Δz_c[k] f_{k−1} + Δz_c[k−1] f_k)/(Δz_c[k−1] + Δz_c[k])` — the nearer
centre gets the larger weight, exact for a linear profile (½ each on a
uniform grid). The naive `½(f_{k−1} + f_k)` would carry a first-order
error wherever the grid stretches.
"""
_interp_to_face!(out::Field{FaceZ}, f::Field{CellCenter}, g::AbstractMesh) =
    (_fv!(_b_c2fz!, interior(g), out.data, f.data, grid_metrics(g)); out)

# face → cell-center
_interp_to_center!(out::Field{CellCenter}, u::Field{FaceX}, g::AbstractMesh) =
    (_fv!(_b_fx2c!, interior(g), out.data, u.data, grid_metrics(g)); out)
_interp_to_center!(out::Field{CellCenter}, v::Field{FaceY}, g::AbstractMesh) =
    (_fv!(_b_fy2c!, interior(g), out.data, v.data, grid_metrics(g)); out)

"""
    _interp_to_center!(out::Field{CellCenter}, w::Field{FaceZ}, g)

z-face → centre interpolation
`((z_f[k+1] − z_c[k]) w_k + (z_c[k] − z_f[k]) w_{k+1})/Δz_c[k]`. A centre
is the midpoint of its faces, so both weights are ½ on every grid; the
distance form keeps the symmetry with the centre → face kernel explicit.
"""
_interp_to_center!(out::Field{CellCenter}, w::Field{FaceZ}, g::AbstractMesh) =
    (_fv!(_b_fz2c!, interior(g), out.data, w.data, grid_metrics(g)); out)

# =========================================================================
# Surface flux for cell-centered scalars — Phase 7.
# =========================================================================
# Adds the effect of prescribed diffusive fluxes at the bottom and top
# walls into the scalar tendency. Sign convention: `F_bot > 0` means
# flux INTO the domain from below (surface warmer than air for θ).
# `F_top > 0` means flux OUT of the domain through the top.
#
# This bypasses the halo-fill route and injects the flux directly into
# the tendency at the boundary cells — the natural place given that the
# eddy-diffusion kernel produces zero wall-face flux under the mirror
# halo used for `NoFlux` / `ImpermeableWall`. The tendency is the flux
# over the *local* cell thickness Δz_c (a thin first cell of a stretched
# grid feels a correspondingly larger heating rate). A 2-D launch.

@inline function _b_surface_flux!(i, j, t, fb, ft, kb, kt)
    @inbounds t[i, j, kb] += fb
    @inbounds t[i, j, kt] -= ft
    return nothing
end

"""
    _add_surface_flux!(tϕ, F_bot, F_top, g)

Inject prescribed wall fluxes into the first and last interior cells:
`tϕ += F_bot/Δz_c` at the bottom, `tϕ −= F_top/Δz_c` at the top, each
over the local cell thickness.
"""
function _add_surface_flux!(tϕ::Field{CellCenter}, F_bot::Real, F_top::Real,
                            g::AbstractMesh)
    m = grid_metrics(g)
    ii, jj, kk = interior(g)
    k_bot = first(kk)
    k_top = last(kk)
    _fv!(_b_surface_flux!, (ii, jj), tϕ.data,
         F_bot * _idz(m, k_bot), F_top * _idz(m, k_top), k_bot, k_top)
    return tϕ
end

# =========================================================================
# Damping profiles — the one implementation of relaxation rates.
# =========================================================================
# Every relaxation term `-λ(ξ)·(φ - φ_target)` in Aeolus (the Rayleigh
# sponge of Phase 7 on every mesh type, the fringe region of the
# spatially developing boundary layer, P59-d #185) takes its rate λ(ξ) from
# a `DampingProfile`. The profile is a 1-D function of one coordinate ξ
# (z for a sponge, x for a fringe) built from a ramp shape:
#
#   * `SinSquaredRamp` — λ = λ_max sin²(½π (ξ - ξ_start)/w) for ξ ≥ ξ_start
#     (Klemp & Lilly 1978 style top sponge; no fall, the ramp ends at the
#     domain top);
#   * `NordstromRamp`  — λ = λ_max [S((ξ - ξ_s)/Δ_rise) - S((ξ - ξ_e)/Δ_fall + 1)]
#     with the smooth step S of Nordström, Nordin & Henningson (1999),
#     eq. (2.3): C^∞, zero outside [ξ_s, ξ_e], λ_max on the plateau.
#
# The sin² rate is evaluated with exactly the arithmetic of the original
# Phase-7 kernels, `s = sin(½π·(z - z_start)·inv_width); α = α_max·s·s`, so
# `add_sponge!` results are unchanged bit for bit.

"""
    AbstractRamp

Shape of a [`DampingProfile`](@ref): [`SinSquaredRamp`](@ref) (Rayleigh
sponge) or [`NordstromRamp`](@ref) (fringe region).
"""
abstract type AbstractRamp end

"""
    SinSquaredRamp()

Rise-only ramp `λ(ξ) = λ_max sin²(½π (ξ − ξ_start)/w)` for `ξ ≥ ξ_start`
(zero below). The shape of the Phase-7 Rayleigh sponge
[`add_sponge!`](@ref Aeolus.Operators.add_sponge!), whose width `w` is the
distance from `z_start` to the domain top. Reference: Klemp, J. B. &
Lilly, D. K. (1978), *J. Atmos. Sci.*, 35, 78-107.
"""
struct SinSquaredRamp <: AbstractRamp end

"""
    NordstromRamp()

Rise-plateau-fall ramp of Nordström, Nordin & Henningson (1999), eq. (2.3),

    λ(ξ) = λ_max [ S((ξ − ξ_start)/Δ_rise) − S((ξ − ξ_end)/Δ_fall + 1) ],

with the smooth step `S(x) = 0` (x ≤ 0), `1/(1 + exp(1/(x − 1) + 1/x))`
(0 < x < 1), `1` (x ≥ 1) — see [`smooth_step`](@ref). Infinitely
differentiable; zero outside `[ξ_start, ξ_end]`.

Reference: Nordström, J., Nordin, N. & Henningson, D. (1999). The fringe
region technique and the Fourier method used in the direct numerical
simulation of spatially evolving viscous flows. *SIAM J. Sci. Comput.*,
20(4), 1365-1393.
"""
struct NordstromRamp <: AbstractRamp end

"""
    smooth_step(x) -> Real

The C^∞ step of Nordström et al. (1999), eq. (2.3):
`0` for `x ≤ 0`, `1/(1 + exp(1/(x − 1) + 1/x))` for `0 < x < 1`, `1` for
`x ≥ 1`. `smooth_step(1/2) = 1/2` and `S(x) + S(1 − x) = 1`.
"""
function smooth_step(x::Real)
    x <= 0 && return zero(float(x))
    x >= 1 && return one(float(x))
    return 1 / (1 + exp(1 / (x - 1) + 1 / x))
end

"""
    DampingProfile(ramp, rate, start, rise; stop = Inf, fall = rise)

A one-dimensional relaxation-rate profile `λ(ξ)` (s⁻¹) for the
relaxation term `−λ(ξ)(φ − φ_target)`. `rate` is the plateau / maximum
value, `start` the coordinate where the rise begins, `rise` its width;
`stop` (where the fall *ends*) and `fall` are used by
[`NordstromRamp`](@ref) only. Evaluate with [`damping_rate`](@ref).

This is the single damping-profile implementation shared by the Rayleigh
sponge ([`add_sponge!`](@ref Aeolus.Operators.add_sponge!), every mesh
type) and the fringe region (`Transition.FringeRegion`, P59-d).
The fields keep the argument types (no promotion) so that the sponge
arithmetic is that of the original kernels.
"""
struct DampingProfile{R<:AbstractRamp, A, S, W, E, F}
    ramp::R
    rate::A
    start::S
    inv_rise::W
    stop::E
    inv_fall::F
end

DampingProfile(ramp::AbstractRamp, rate, start, rise; stop = Inf, fall = rise) =
    DampingProfile(ramp, rate, start, 1 / rise, stop, 1 / fall)

"""
    sponge_profile(α_max, z_start, z_top) -> DampingProfile

The Rayleigh-sponge profile of [`add_sponge!`](@ref Aeolus.Operators.add_sponge!):
a [`SinSquaredRamp`](@ref) from `z_start` to `z_top`.
"""
sponge_profile(α_max, z_start, z_top) =
    DampingProfile(SinSquaredRamp(), α_max, z_start, 1 / (z_top - z_start), Inf, 0)

"""
    in_damping(p::DampingProfile, ξ) -> Bool

`true` where the profile can be non-zero (the kernels skip other cells, so
a cell outside the region is never touched).
"""
in_damping(p::DampingProfile{SinSquaredRamp}, ξ) = ξ >= p.start
in_damping(p::DampingProfile{NordstromRamp}, ξ) = p.start < ξ < p.stop

"""
    damping_rate(p::DampingProfile, ξ) -> Real

The relaxation rate `λ(ξ)` of profile `p` at coordinate `ξ` (zero outside
the region).
"""
function damping_rate(p::DampingProfile{SinSquaredRamp}, ξ)
    in_damping(p, ξ) || return zero(p.rate * p.inv_rise)
    half_π = 0.5 * π
    s = sin(half_π * (ξ - p.start) * p.inv_rise)
    return p.rate * s * s
end

function damping_rate(p::DampingProfile{NordstromRamp}, ξ)
    in_damping(p, ξ) || return zero(float(p.rate))
    return p.rate * (smooth_step((ξ - p.start) * p.inv_rise) -
                     smooth_step((ξ - p.stop) * p.inv_fall + 1))
end

# Relaxation target: a constant, a level vector (one value per interior
# level, indexed by `kl = k - k0`, the level of the field's own location),
# or an array / Field indexed like the field. The 5-argument form is what
# the kernel calls; `kl` is ignored by every target but the level vector.
@inline _damping_target(t::Real, i, j, k) = t
@inline _damping_target(t::Field, i, j, k) = @inbounds t.data[i, j, k]
@inline _damping_target(t::AbstractArray{<:Real,3}, i, j, k) = @inbounds t[i, j, k]
@inline _damping_target(t, i, j, k, kl) = _damping_target(t, i, j, k)
@inline _damping_target(t::AbstractVector{<:Real}, i, j, k, kl) = @inbounds t[kl]

# A level-vector target must have one value per interior level.
_check_damping_target(t, g) = nothing
function _check_damping_target(t::AbstractVector, g)
    length(t) == g.Nz ||
        throw(DimensionMismatch("damping target: a level vector needs Nz = $(g.Nz) values, got $(length(t))"))
    return nothing
end

# The coordinate the profile is sampled at: a 1-D vector along `axis`
# (interior-indexed), or a halo-padded 3-D array of per-cell coordinates.
@inline _damping_coord(c::AbstractVector, axis, i, j, k, i0, j0, k0) =
    @inbounds axis == 3 ? c[k - k0] : axis == 1 ? c[i - i0] : c[j - j0]
@inline _damping_coord(c::AbstractArray{<:Real,3}, axis, i, j, k, i0, j0, k0) = @inbounds c[i, j, k]


"""
    _add_damping!(tf, f, profile, coords, axis, target, g; levels = nothing)

Add `−λ(ξ)(f − target)` to `tf` on the interior cells where
`in_damping(profile, ξ)`, with `ξ = coords[n]` the coordinate of the
`n`-th interior index along `axis` (`1` = x, `3` = z) — or, when `coords`
is a halo-padded 3-D array, `ξ = coords[i, j, k]` of every cell (the
physical heights of a `TerrainFollowingGrid`; `axis` is then unused).
The one kernel behind the sponge (`axis = 3` on a `RegularGrid` and a
`StretchedGrid`, the 3-D heights on a `TerrainFollowingGrid`), the
fringe (`axis = 1`) and `LargeScaleForcing.ReferenceDamping`.

`target` is a number, a level vector (length `Nz`, one value per interior
level), a `Field` or a halo-padded 3-D array. With `levels` (a vector of
length `Nz`) the tendency added to each interior level is also summed
into `levels[kl]` (units of `tf` per cell, summed over the level's
columns): the integral a budget books.

KA-d (#299): without `levels` one 3-D launch (pointwise). With `levels`
the launch is over the levels `k` only: one work item owns plane `k` and
runs its `(j, i)` loop serially, so `levels[kl]` is accumulated in the
pre-port cell order by one task (a 3-D launch would race on it). The
plane is a canonical per-level reduction as it stands (ka_port.md).
"""
function _add_damping!(tf::Field{L}, f::Field{L}, p::DampingProfile,
                       coords::AbstractArray, axis::Int, target,
                       g::AbstractMesh; levels = nothing) where {L}
    _check_damping_target(target, g)
    ii, jj, kk = interior(g)
    i0, j0, k0 = first(ii) - 1, first(jj) - 1, first(kk) - 1
    if levels === nothing
        _fv!(_b_damping!, (ii, jj, kk), tf.data, f.data, p, coords, axis, target, i0, j0, k0)
    else
        length(levels) >= length(kk) ||
            throw(DimensionMismatch("damping levels: need $(length(kk)) entries, got $(length(levels))"))
        launch_weighted!(Backends.backend(tf.data), length(ii) * length(jj), _b_damping_plane!,
                         (kk,), tf.data, f.data, p, coords, axis, target, ii, jj, i0, j0, k0, levels)
    end
    return tf
end

@inline function _b_damping!(i, j, k, t, fd, p, coords, axis, target, i0, j0, k0)
    @inbounds begin
        ξ = _damping_coord(coords, axis, i, j, k, i0, j0, k0)
        in_damping(p, ξ) || return nothing
        λ = damping_rate(p, ξ)
        t[i, j, k] -= λ * (fd[i, j, k] - _damping_target(target, i, j, k, k - k0))
    end
    return nothing
end

# One plane per work item: the pre-port (j, i) order of the `levels` sum.
@inline function _b_damping_plane!(k, t, fd, p, coords, axis, target, ii, jj, i0, j0, k0, levels)
    @inbounds for j in jj, i in ii
        ξ = _damping_coord(coords, axis, i, j, k, i0, j0, k0)
        in_damping(p, ξ) || continue
        λ = damping_rate(p, ξ)
        δ = λ * (fd[i, j, k] - _damping_target(target, i, j, k, k - k0))
        t[i, j, k] -= δ
        levels[k - k0] -= δ
    end
    return nothing
end

# =========================================================================
# Rayleigh sponge damping — Phase 7.
# =========================================================================
# Adds  -α(z) · (φ - φ_ref)  to the tendency inside a top-of-domain
# region [z_start, z_top]. `α(z)` grows smoothly with height:
#     α(z) = α_max · sin²( π/2 · (z - z_start) / (z_top - z_start) ).
# Below `z_start` the damping is zero. Used to absorb gravity waves and
# turbulent structures that would otherwise reflect off the rigid lid.
# The rate comes from the shared `DampingProfile` above, the arithmetic
# from the one kernel `_add_damping!`. `f_ref` is a number, a level vector
# (length Nz), a Field or a halo-padded array (#254). One method for the
# RegularGrid and the StretchedGrid: the profile is sampled at
# `damping_coordinates(g, loc)`, the actual heights of the levels, so the
# layer covers the physical region asked for (a TerrainFollowingGrid
# samples its 3-D physical heights, StretchedGrids).

function _add_sponge!(tf::Field{L}, f::Field{L},
                      α_max::Real, z_start::Real, z_top::Real,
                      f_ref, g::AbstractMesh) where {L}
    zs = damping_coordinates(g, L())
    return _add_damping!(tf, f, sponge_profile(α_max, z_start, z_top), zs, 3, f_ref, g)
end

"""
    damping_coordinates(g, loc) -> Vector

The heights (m) of the `Nz` interior levels of a field at location `loc`
on `g` (a `RegularGrid` here; `StretchedGrids` adds the `StretchedGrid`
method, the actual stretched heights): the coordinate vector the damping
kernel `_add_damping!` samples a vertical [`DampingProfile`](@ref) at.
"""
damping_coordinates(g::RegularGrid, loc::Location) = _znodes_for(g, loc)

# helper — dispatch to `znodes` for a location, since `L` here is a type
_znodes_for(g::RegularGrid, ::CellCenter) = collect(Grids.znodes(g, CellCenter()))
_znodes_for(g::RegularGrid, ::FaceX)      = collect(Grids.znodes(g, FaceX()))
_znodes_for(g::RegularGrid, ::FaceY)      = collect(Grids.znodes(g, FaceY()))
_znodes_for(g::RegularGrid, ::FaceZ)      = collect(Grids.znodes(g, FaceZ()))


# =========================================================================
# Momentum advection — staggered Arakawa-C, flux form, 2nd-order centered.
# =========================================================================
# Adds  -∂(u_j u_i)/∂x_j  into (tu, tv, tw) at their respective face
# locations. Cross-terms use linear interpolation of the transporting
# velocity and the transported momentum to the flux face. Correct for
# incompressible flow (∇·u = 0).
#
# Vertically stretched grids (#151): the u and v control volumes have
# height Δz_c[k], the w control volume Δz_f[k]. The transported velocity
# is the arithmetic face mean; the transporting velocity is the *volume
# flux* through the face. A w control volume's side face spans half of the
# u-cell k-1 below and half of the u-cell k above, so its transporting u is
# the thickness-weighted mean (Δz_c[k-1] u[k-1] + Δz_c[k] u[k]) /
# (Δz_c[k-1] + Δz_c[k]) (`_zflux`; ½ each on a uniform grid). With these
# fluxes the volume-flux divergence of every u, v and w control volume is a
# thickness-weighted mean of the divergence of the pressure cells it
# overlaps, so a discretely solenoidal field transports momentum without
# creating or destroying kinetic energy: the operator is skew-symmetric in
# the volume-weighted inner product (Morinishi, Lund, Vasilyev & Moin 1998,
# J. Comput. Phys. 143, 90; Vasilyev 2000, J. Comput. Phys. 157, 746, for
# the non-uniform extension).

# The flux divergence ∂(uⱼuᵢ)/∂xⱼ of each component at one point: the
# advection bodies subtract it from the tendency, the fused tendency body
# bodies of the step cores (`_b_tendency_momentum!`, SC-d #404) start from it.
@inline function _mom_u_at(i, j, k, u, v, w, m)
    @inbounds begin
        uce = 0.5*(u[i,   j, k] + u[i+1, j, k])
        ucw = 0.5*(u[i-1, j, k] + u[i,   j, k])
        d_uu_dx = (uce*uce - ucw*ucw) * _idx(m, i)

        vun = 0.5*(v[i-1, j+1, k] + v[i, j+1, k])
        uvn = 0.5*(u[i,   j,   k] + u[i, j+1, k])
        vus = 0.5*(v[i-1, j,   k] + v[i, j,   k])
        uvs = 0.5*(u[i,   j-1, k] + u[i, j,   k])
        d_vu_dy = (vun*uvn - vus*uvs) * _idy(m, j)

        wut = 0.5*(w[i-1, j, k+1] + w[i, j, k+1])
        uwt = 0.5*(u[i,   j, k  ] + u[i, j, k+1])
        wub = 0.5*(w[i-1, j, k  ] + w[i, j, k  ])
        uwb = 0.5*(u[i,   j, k-1] + u[i, j, k  ])
        d_wu_dz = (wut*uwt - wub*uwb) * _idz(m, k)

        r = d_uu_dx + d_vu_dy + d_wu_dz
    end
    return r
end

@inline function _mom_v_at(i, j, k, u, v, w, m)
    @inbounds begin
        uve = 0.5*(u[i+1, j-1, k] + u[i+1, j, k])
        vue = 0.5*(v[i,   j,   k] + v[i+1, j, k])
        uvw = 0.5*(u[i,   j-1, k] + u[i,   j, k])
        vuw = 0.5*(v[i-1, j,   k] + v[i,   j, k])
        d_uv_dx = (uve*vue - uvw*vuw) * _idx(m, i)

        vcn = 0.5*(v[i, j,   k] + v[i, j+1, k])
        vcs = 0.5*(v[i, j-1, k] + v[i, j,   k])
        d_vv_dy = (vcn*vcn - vcs*vcs) * _idy(m, j)

        wvt = 0.5*(w[i, j-1, k+1] + w[i, j, k+1])
        vwt = 0.5*(v[i, j,   k  ] + v[i, j, k+1])
        wvb = 0.5*(w[i, j-1, k  ] + w[i, j, k  ])
        vwb = 0.5*(v[i, j,   k-1] + v[i, j, k  ])
        d_wv_dz = (wvt*vwt - wvb*vwb) * _idz(m, k)

        r = d_uv_dx + d_vv_dy + d_wv_dz
    end
    return r
end

@inline function _mom_w_at(i, j, k, u, v, w, m)
    @inbounds begin
        uwe = _zflux(m, k, u[i+1, j, k-1], u[i+1, j, k])
        wue = 0.5*(w[i,   j, k  ] + w[i+1, j, k])
        uww = _zflux(m, k, u[i,   j, k-1], u[i,   j, k])
        wuw = 0.5*(w[i-1, j, k  ] + w[i,   j, k])
        d_uw_dx = (uwe*wue - uww*wuw) * _idx(m, i)

        vwn = _zflux(m, k, v[i, j+1, k-1], v[i, j+1, k])
        wvn = 0.5*(w[i, j,   k  ] + w[i, j+1, k])
        vws = _zflux(m, k, v[i, j,   k-1], v[i, j,   k])
        wvs = 0.5*(w[i, j-1, k  ] + w[i, j,   k])
        d_vw_dy = (vwn*wvn - vws*wvs) * _idy(m, j)

        wct = 0.5*(w[i, j, k  ] + w[i, j, k+1])
        wcb = 0.5*(w[i, j, k-1] + w[i, j, k  ])
        d_ww_dz = (wct*wct - wcb*wcb) * _idzf(m, k)

        r = d_uw_dx + d_vw_dy + d_ww_dz
    end
    return r
end

@inline _b_momentum_u!(i, j, k, tu, u, v, w, m) =
    (@inbounds tu[i, j, k] -= _mom_u_at(i, j, k, u, v, w, m); nothing)
@inline _b_momentum_v!(i, j, k, tv, u, v, w, m) =
    (@inbounds tv[i, j, k] -= _mom_v_at(i, j, k, u, v, w, m); nothing)
@inline _b_momentum_w!(i, j, k, tw, u, v, w, m) =
    (@inbounds tw[i, j, k] -= _mom_w_at(i, j, k, u, v, w, m); nothing)

"""
    _momentum_advect!(tu, tv, tw, u, v, w, g)

Second-order flux-form momentum advection `-∂(uⱼuᵢ)/∂xⱼ` on the
Arakawa-C grid, three launches (u, v, w). On a vertically stretched grid
the u and v control volumes have height `Δz_c[k]`, the w control volume
`Δz_f[k]`, and the transporting velocity on each face is the volume flux
through it (thickness-weighted for the horizontal faces of a w volume),
which keeps the operator energy-conserving for a discretely solenoidal
field (Morinishi et al. 1998; Vasilyev 2000). On a uniform grid it is the
plain centred stencil.
"""
function _momentum_advect!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                           u::Field{FaceX},  v::Field{FaceY},  w::Field{FaceZ},
                           g::AbstractMesh)
    m = grid_metrics(g)
    I = interior(g)
    _fv!(_b_momentum_u!, I, tu.data, u.data, v.data, w.data, m)
    _fv!(_b_momentum_v!, I, tv.data, u.data, v.data, w.data, m)
    _fv!(_b_momentum_w!, I, tw.data, u.data, v.data, w.data, m)
    return (tu, tv, tw)
end

# =========================================================================
# Buoyancy — Boussinesq g'β(θ - θ₀)/θ₀ added to w-momentum at z-faces.
# =========================================================================
# θ is at cell centers; interpolated in z to the w-face with the geometric
# weights of the grid (`_zface`; ½ each on a uniform grid), so a
# horizontally uniform column is in exact discrete hydrostatic balance on
# a stretched grid too. For the standard atmospheric convention we take
# α_θ = 1/θ₀ so the buoyancy tendency reads   b = g (θ - θ₀) / θ₀.

@inline function _b_buoyancy!(i, j, k, tw, θ, θ₀, coef, m)
    @inbounds begin
        θface = _zface(m, k, θ[i,j,k-1], θ[i,j,k])
        tw[i,j,k] += coef * (θface - θ₀)
    end
    return nothing
end

"""
    _add_buoyancy!(tw, θ, θ₀, g_accel, g)

Boussinesq buoyancy `g(θ − θ₀)/θ₀` added to the `FaceZ` tendency, with `θ`
brought to the face by the geometric (volume-weighted) interpolation of
the grid. The naive half-sum would put a spurious first-order buoyancy
source wherever the grid stretches.
"""
function _add_buoyancy!(tw::Field{FaceZ}, θ::Field{CellCenter},
                        θ₀::Real, g_accel::Real, g::AbstractMesh)
    _fv!(_b_buoyancy!, interior(g), tw.data, θ.data, θ₀, g_accel / θ₀, grid_metrics(g))
    return tw
end

# --- moist buoyancy (Phase 11) ------------------------------------------
# Virtual potential temperature θ_v = θ (1 + 0.61 qv - qL), where qL is
# a bulk liquid-water mass fraction. Uses the standard 0.61 approximation
# for dry-air / water-vapor gas-constant ratio (Rv/Rd - 1 = 0.608).

@inline function _b_buoyancy_moist!(i, j, k, tw, θ, qv, θ₀, coef, m)
    @inbounds begin
        θface  = _zface(m, k, θ[i,j,k-1],  θ[i,j,k])
        qvface = _zface(m, k, qv[i,j,k-1], qv[i,j,k])
        θv     = θface * (1 + 0.61 * qvface)
        tw[i,j,k] += coef * (θv - θ₀)
    end
    return nothing
end

"""
    _add_buoyancy_moist!(tw, θ, qv, θ₀, g_accel, g)

Moist Boussinesq buoyancy `g(θᵥ − θ₀)/θ₀`, `θᵥ = θ(1 + 0.61 qv)`, on the
z-faces, with `θ` and `qv` brought to the face by the same geometric
interpolation as the dry [`_add_buoyancy!`](@ref).
"""
function _add_buoyancy_moist!(tw::Field{FaceZ},
                              θ::Field{CellCenter}, qv::Field{CellCenter},
                              θ₀::Real, g_accel::Real, g::AbstractMesh)
    _fv!(_b_buoyancy_moist!, interior(g), tw.data, θ.data, qv.data, θ₀, g_accel / θ₀,
         grid_metrics(g))
    return tw
end

# =========================================================================
# Coriolis — f-plane, adds f×u to horizontal momenta.
# =========================================================================
# u lives at FaceX, v at FaceY. For u-momentum we need v at the u-face;
# for v-momentum we need u at the v-face. Both require a 4-point
# horizontal average of the neighbouring face values.

# Purely horizontal (level by level), so one kernel serves every mesh with
# the RegularGrid horizontal layout (RegularGrid, StretchedGrid, #151).
# The body writes tu and tv at its own point: one launch.
@inline function _b_coriolis!(i, j, k, tu, tv, u, v, f)
    @inbounds begin
        v_at_u = 0.25 * (v[i-1,j,k]   + v[i,j,k]   +
                         v[i-1,j+1,k] + v[i,j+1,k])
        tu[i,j,k] += f * v_at_u

        u_at_v = 0.25 * (u[i,j-1,k]   + u[i+1,j-1,k] +
                         u[i,j,  k]   + u[i+1,j,  k])
        tv[i,j,k] -= f * u_at_v
    end
    return nothing
end

function _add_coriolis!(tu::Field{FaceX}, tv::Field{FaceY},
                        u::Field{FaceX},  v::Field{FaceY},
                        f::Real, g::AbstractMesh)
    _fv!(_b_coriolis!, interior(g), tu.data, tv.data, u.data, v.data, f)
    return (tu, tv)
end

# =========================================================================
# Smagorinsky-Lilly SGS
# =========================================================================
# Computes eddy viscosity ν_t at cell centers from the full strain-rate
# magnitude |S| = √(2 S_ij S_ij), where S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i).
# Diagonal strain components live naturally at cell centers on a C-grid;
# off-diagonal components (S12, S13, S23) live at cell edges and are
# interpolated to the cell center with a 4-point average.
#
#     ν_t = (C_s Δ)² |S|          Δ = (dx dy dz)^{1/3}
#
# On a stretched grid Δ is the local width (Δx Δy Δz_c[k])^{1/3} of
# Deardorff (1970, J. Fluid Mech. 41, 453), and the strain uses the local
# vertical metrics: ∂w/∂z across the cell (1/Δz_c[k]), ∂u/∂z and ∂v/∂z
# across the two neighbouring centres (2/(Δz_f[k] + Δz_f[k+1])).

@inline function _b_smagorinsky!(i, j, k, νt, ud, vd, wd, C_s, m)
    @inbounds begin
        coef = (C_s * _filter_width(m, k)) ^ 2
        # the one strain stencil (Operators._strain_at / _velocity_gradient_at)
        ms = _strain_at(ud, vd, wd, i, j, k, _idx(m, i), _idy(m, j), _idz(m, k), _idz_off(m, k))[7]
        νt[i,j,k] = coef * ms
    end
    return nothing
end

"""
    _compute_smagorinsky_νt!(νt, u, v, w, C_s, g)

Smagorinsky–Lilly `νt = (C_s Δ_k)² |S|` at the cell centres with
`Δ_k = (Δx Δy Δz_c[k])^{1/3}` (Deardorff 1970; uniform on a `RegularGrid`)
and the strain from the one strain stencil (`Operators._strain_at`) with
the grid's vertical metrics.
"""
function _compute_smagorinsky_νt!(νt::Field{CellCenter},
                                  u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                  C_s::Real, g::AbstractMesh)
    _fv!(_b_smagorinsky!, interior(g), νt.data, u.data, v.data, w.data, C_s, grid_metrics(g))
    return νt
end

# --- eddy-viscosity flux divergence --------------------------------------
# ∂/∂x_j (ν_t ∂u_i/∂x_j) added into the tendency at the velocity's faces.
# Diffusive flux in the velocity's own direction lives at cell centers, the
# others at edges — with ν_t interpolated to the flux location (4-cell
# edge means). In z the gradient between two centres uses Δz_f, between
# two z-faces Δz_c, and the flux difference is divided by the height of
# the control volume (Δz_c for u, v; Δz_f for w).

@inline function _b_eddy_u!(i, j, k, tu, u, n, m, wc, wf)
    @inbounds begin
        idx = _idx(m, i); idy = _idy(m, j)
        fxE = n[i  ,j,k] * (u[i+1,j,k] - u[i  ,j,k]) * idx
        fxW = n[i-1,j,k] * (u[i  ,j,k] - u[i-1,j,k]) * idx

        νt_eN = 0.25*(n[i-1,j,k]   + n[i,j,k]   +
                      n[i-1,j+1,k] + n[i,j+1,k])
        νt_eS = 0.25*(n[i-1,j-1,k] + n[i,j-1,k] +
                      n[i-1,j,k]   + n[i,j,k])
        fyN = νt_eN * (u[i,j+1,k] - u[i,j  ,k]) * idy
        fyS = νt_eS * (u[i,j,  k] - u[i,j-1,k]) * idy

        νt_eT = 0.25*(n[i-1,j,k]   + n[i,j,k]   +
                      n[i-1,j,k+1] + n[i,j,k+1])
        νt_eB = 0.25*(n[i-1,j,k-1] + n[i,j,k-1] +
                      n[i-1,j,k]   + n[i,j,k])
        fzT = νt_eT * (u[i,j,k+1] - u[i,j,k  ]) * _idzf(m, k+1)
        fzB = νt_eB * (u[i,j,k  ] - u[i,j,k-1]) * _idzf(m, k)

        tu[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + _vertical_divergence(fzT, fzB, _idz(m, k), wc, wf, k)
    end
    return nothing
end

@inline function _b_eddy_v!(i, j, k, tv, v, n, m, wc, wf)
    @inbounds begin
        idx = _idx(m, i); idy = _idy(m, j)
        fyN = n[i,j  ,k] * (v[i,j+1,k] - v[i,j  ,k]) * idy
        fyS = n[i,j-1,k] * (v[i,j  ,k] - v[i,j-1,k]) * idy

        νt_eE = 0.25*(n[i,j-1,k]   + n[i+1,j-1,k] +
                      n[i,j,k]     + n[i+1,j,k])
        νt_eW = 0.25*(n[i-1,j-1,k] + n[i,j-1,k]   +
                      n[i-1,j,k]   + n[i,j,k])
        fxE = νt_eE * (v[i+1,j,k] - v[i,  j,k]) * idx
        fxW = νt_eW * (v[i,  j,k] - v[i-1,j,k]) * idx

        νt_eT = 0.25*(n[i,j-1,k]   + n[i,j,k]   +
                      n[i,j-1,k+1] + n[i,j,k+1])
        νt_eB = 0.25*(n[i,j-1,k-1] + n[i,j,k-1] +
                      n[i,j-1,k]   + n[i,j,k])
        fzT = νt_eT * (v[i,j,k+1] - v[i,j,k  ]) * _idzf(m, k+1)
        fzB = νt_eB * (v[i,j,k  ] - v[i,j,k-1]) * _idzf(m, k)

        tv[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + _vertical_divergence(fzT, fzB, _idz(m, k), wc, wf, k)
    end
    return nothing
end

@inline function _b_eddy_w!(i, j, k, tw, w, n, m, wc, wf)
    @inbounds begin
        idx = _idx(m, i); idy = _idy(m, j)
        fzT = n[i,j,k  ] * (w[i,j,k+1] - w[i,j,k  ]) * _idz(m, k)
        fzB = n[i,j,k-1] * (w[i,j,k  ] - w[i,j,k-1]) * _idz(m, k-1)

        νt_eE = 0.25*(n[i,j,k-1]   + n[i+1,j,k-1] +
                      n[i,j,k]     + n[i+1,j,k])
        νt_eW = 0.25*(n[i-1,j,k-1] + n[i,j,k-1]   +
                      n[i-1,j,k]   + n[i,j,k])
        fxE = νt_eE * (w[i+1,j,k] - w[i,  j,k]) * idx
        fxW = νt_eW * (w[i,  j,k] - w[i-1,j,k]) * idx

        νt_eN = 0.25*(n[i,j,k-1]   + n[i,j+1,k-1] +
                      n[i,j,k]     + n[i,j+1,k])
        νt_eS = 0.25*(n[i,j-1,k-1] + n[i,j,k-1]   +
                      n[i,j-1,k]   + n[i,j,k])
        fyN = νt_eN * (w[i,j+1,k] - w[i,j  ,k]) * idy
        fyS = νt_eS * (w[i,j,  k] - w[i,j-1,k]) * idy

        tw[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + _vertical_divergence_w(fzT, fzB, _idzf(m, k), wc, wf, k)
    end
    return nothing
end

"""
    _add_eddy_viscosity_u!(tu, u, νt, g)
    _add_eddy_viscosity_v!(tv, v, νt, g)
    _add_eddy_viscosity_w!(tw, w, νt, g)

`∂/∂xⱼ(νₜ ∂uᵢ/∂xⱼ)` added to the tendency of one velocity component, νₜ
averaged to the flux location (4-cell edge means). In z the gradient
between two centres uses `Δz_f`, between two z-faces `Δz_c`, and the flux
difference is divided by the control-volume height (`Δz_c` for u, v;
`Δz_f` for w): conservative on a stretched grid, the uniform stencil on a
`RegularGrid`.

`wc, wf` (default `nothing`, the unweighted operator bit for bit) are the
per-level weights of a sound-proof core (`Anelastic.constraint_weights`:
`W = ρ̄` or `ρ̄θ̄` at the cell centres and the bottom z faces, ONE-j #460):
the vertical part becomes `(1/W) ∂z(W νₜ ∂z uᵢ)` — for u, v the fluxes on
the z faces weighted by `wf`, divided by `wc[k]`; for w the fluxes at the
centres weighted by `wc`, divided by `wf[k]` — so `Σ W uᵢ ΔV` is conserved
(the flux form of the weighted momentum advection). The horizontal parts
are unchanged: `W` depends on z only.
"""
function _add_eddy_viscosity_u!(tu::Field{FaceX}, u::Field{FaceX},
                                 νt::Field{CellCenter}, g::AbstractMesh, wc = nothing, wf = nothing)
    _fv!(_b_eddy_u!, interior(g), tu.data, u.data, νt.data, grid_metrics(g), wc, wf)
    return tu
end

function _add_eddy_viscosity_v!(tv::Field{FaceY}, v::Field{FaceY},
                                 νt::Field{CellCenter}, g::AbstractMesh, wc = nothing, wf = nothing)
    _fv!(_b_eddy_v!, interior(g), tv.data, v.data, νt.data, grid_metrics(g), wc, wf)
    return tv
end

function _add_eddy_viscosity_w!(tw::Field{FaceZ}, w::Field{FaceZ},
                                 νt::Field{CellCenter}, g::AbstractMesh, wc = nothing, wf = nothing)
    _fv!(_b_eddy_w!, interior(g), tw.data, w.data, νt.data, grid_metrics(g), wc, wf)
    return tw
end
# --- stress-form SGS momentum divergence (P55-b, #175) -----------
# −∂τᵢⱼ/∂xⱼ with τᵢⱼ = −2νᵢⱼSᵢⱼ on the C-grid. Placement (the locations
# where the Laplacian kernels above form their fluxes):
#
#   τ11, τ22, τ33 at cell centres            τ11[i,j,k] at (xᵢ, yⱼ, zₖ)
#   τ12 on xy-edges                          τ12[i,j,k] at (x_{i-½}, y_{j-½}, zₖ)
#   τ13 on xz-edges                          τ13[i,j,k] at (x_{i-½}, yⱼ, z_{k-½})
#   τ23 on yz-edges                          τ23[i,j,k] at (xᵢ, y_{j-½}, z_{k-½})
#
# i.e. the edge with index (i, j) is the south-west corner of cell (i, j),
# where u-face i and v-face j meet. The strain at each location is the
# compact face difference: Sᵢᵢ is the diagonal of `_velocity_gradient_at`
# (face to face through the cell), and S12 = ½(∂u/∂y + ∂v/∂x) on the edge
# from the two velocities adjacent to it. ν is the cell-centred value for
# the diagonals and the 4-cell average onto the edge for the off-diagonals,
# exactly as in `_add_eddy_viscosity_*!`. Anisotropy (the CM1 convention,
# Bryan 2002): ν11 = ν22 = ν12 = νh, ν33 = ν13 = ν23 = νv.
#
# The flux of row i is split as τᵢⱼ = −νᵢⱼ(a ∂uᵢ/∂xⱼ + b ∂uⱼ/∂xᵢ):
#   (a, b) = (1, 1) the full stress −2νᵢⱼSᵢⱼ;
#   (a, b) = (1, 0) the Laplacian ("diagonal") part ∂ⱼ(νᵢⱼ∂ⱼuᵢ), which is the
#                   operator the IMEX steppers treat implicitly;
#   (a, b) = (0, 1) the transposed part ∂ⱼ(νᵢⱼ∂ᵢuⱼ), the explicit remainder.

# 4-cell averages of a cell-centred ν onto the edge with index (i, j, k)
Base.@propagate_inbounds _νe12(n, i, j, k) = 0.25 * (n[i-1,j-1,k] + n[i,j-1,k] + n[i-1,j,k] + n[i,j,k])
Base.@propagate_inbounds _νe13(n, i, j, k) = 0.25 * (n[i-1,j,k-1] + n[i,j,k-1] + n[i-1,j,k] + n[i,j,k])
Base.@propagate_inbounds _νe23(n, i, j, k) = 0.25 * (n[i,j-1,k-1] + n[i,j,k-1] + n[i,j-1,k] + n[i,j,k])

# diagonal stresses at the centre of cell (i, j, k): −(a+b) ν ∂uᵢ/∂xᵢ
Base.@propagate_inbounds _τ11(u, nh, i, j, k, idx, a, b) = -(nh[i,j,k] * ((a + b) * ((u[i+1,j,k] - u[i,j,k]) * idx)))
Base.@propagate_inbounds _τ22(v, nh, i, j, k, idy, a, b) = -(nh[i,j,k] * ((a + b) * ((v[i,j+1,k] - v[i,j,k]) * idy)))
Base.@propagate_inbounds _τ33(w, nv, i, j, k, idz, a, b) = -(nv[i,j,k] * ((a + b) * ((w[i,j,k+1] - w[i,j,k]) * idz)))

# off-diagonal stresses of row `own` on the edge (i, j, k):
# −ν_e (a ∂u_own/∂x_other + b ∂u_other/∂x_own)
Base.@propagate_inbounds _τ12_u(u, v, nh, i, j, k, idx, idy, a, b) =
    -(_νe12(nh, i, j, k) * (a * ((u[i,j,k] - u[i,j-1,k]) * idy) + b * ((v[i,j,k] - v[i-1,j,k]) * idx)))
Base.@propagate_inbounds _τ12_v(u, v, nh, i, j, k, idx, idy, a, b) =
    -(_νe12(nh, i, j, k) * (a * ((v[i,j,k] - v[i-1,j,k]) * idx) + b * ((u[i,j,k] - u[i,j-1,k]) * idy)))
Base.@propagate_inbounds _τ13_u(u, w, nv, i, j, k, idx, idz, a, b) =
    -(_νe13(nv, i, j, k) * (a * ((u[i,j,k] - u[i,j,k-1]) * idz) + b * ((w[i,j,k] - w[i-1,j,k]) * idx)))
Base.@propagate_inbounds _τ13_w(u, w, nv, i, j, k, idx, idz, a, b) =
    -(_νe13(nv, i, j, k) * (a * ((w[i,j,k] - w[i-1,j,k]) * idx) + b * ((u[i,j,k] - u[i,j,k-1]) * idz)))
Base.@propagate_inbounds _τ23_v(v, w, nv, i, j, k, idy, idz, a, b) =
    -(_νe23(nv, i, j, k) * (a * ((v[i,j,k] - v[i,j,k-1]) * idz) + b * ((w[i,j,k] - w[i,j-1,k]) * idy)))
Base.@propagate_inbounds _τ23_w(v, w, nv, i, j, k, idy, idz, a, b) =
    -(_νe23(nv, i, j, k) * (a * ((w[i,j,k] - w[i,j-1,k]) * idy) + b * ((v[i,j,k] - v[i,j,k-1]) * idz)))

# The stress form has no vertically stretched stencil yet (#151 covers the
# Laplacian form only): refuse it by name instead of a MethodError.
_add_sgs_stress_divergence!(tu, tv, tw, u, v, w, nh, nv, a, b, g::AbstractMesh, wc = nothing, wf = nothing) =
    throw(ArgumentError("sgs_stress = :symmetric (the SGS stress form) is implemented " *
                        "on a RegularGrid only, not on a $(nameof(typeof(g))); use " *
                        "sgs_stress = :laplacian"))


# −∂τᵢⱼ/∂xⱼ into (tu, tv, tw), with the stresses formed on the fly from
# (u, v, w) and the cell-centred (νh, νv) arrays (halos filled), part (a, b).
# One launch: the body writes the three tendencies at its own point.
@inline function _b_sgs_stress_div!(i, j, k, tud, tvd, twd, ud, vd, wd, nh, nv, idx, idy, idz, A, B, wc, wf)
    @inbounds begin
        tud[i,j,k] += -((_τ11(ud, nh, i, j, k, idx, A, B) - _τ11(ud, nh, i-1, j, k, idx, A, B)) * idx +
                        (_τ12_u(ud, vd, nh, i, j+1, k, idx, idy, A, B) -
                         _τ12_u(ud, vd, nh, i, j, k, idx, idy, A, B)) * idy +
                        _vertical_divergence(_τ13_u(ud, wd, nv, i, j, k+1, idx, idz, A, B),
                                             _τ13_u(ud, wd, nv, i, j, k, idx, idz, A, B), idz, wc, wf, k))
        tvd[i,j,k] += -((_τ12_v(ud, vd, nh, i+1, j, k, idx, idy, A, B) -
                         _τ12_v(ud, vd, nh, i, j, k, idx, idy, A, B)) * idx +
                        (_τ22(vd, nh, i, j, k, idy, A, B) - _τ22(vd, nh, i, j-1, k, idy, A, B)) * idy +
                        _vertical_divergence(_τ23_v(vd, wd, nv, i, j, k+1, idy, idz, A, B),
                                             _τ23_v(vd, wd, nv, i, j, k, idy, idz, A, B), idz, wc, wf, k))
        twd[i,j,k] += -((_τ13_w(ud, wd, nv, i+1, j, k, idx, idz, A, B) -
                         _τ13_w(ud, wd, nv, i, j, k, idx, idz, A, B)) * idx +
                        (_τ23_w(vd, wd, nv, i, j+1, k, idy, idz, A, B) -
                         _τ23_w(vd, wd, nv, i, j, k, idy, idz, A, B)) * idy +
                        _vertical_divergence_w(_τ33(wd, nv, i, j, k, idz, A, B),
                                               _τ33(wd, nv, i, j, k-1, idz, A, B), idz, wc, wf, k))
    end
    return nothing
end

function _add_sgs_stress_divergence!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                     u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                     nh::AbstractArray{<:Real,3}, nv::AbstractArray{<:Real,3},
                                     a::Real, b::Real, g::RegularGrid, wc = nothing, wf = nothing)
    T = eltype(tu.data)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    _fv!(_b_sgs_stress_div!, interior(g), tu.data, tv.data, tw.data, u.data, v.data, w.data,
         nh, nv, idx, idy, idz, T(a), T(b), wc, wf)
    return nothing
end

# −∂τᵢⱼ/∂xⱼ from a stored stress (six cell-centre-shaped arrays, placement
# above). Reads τ11 on i ∈ first-1:last, τ12 on i, j ∈ first:last+1, etc.
@inline function _b_stress_div!(i, j, k, tud, tvd, twd, τ11, τ22, τ33, τ12, τ13, τ23, idx, idy, idz)
    @inbounds begin
        tud[i,j,k] += -((τ11[i,j,k] - τ11[i-1,j,k]) * idx +
                        (τ12[i,j+1,k] - τ12[i,j,k]) * idy +
                        (τ13[i,j,k+1] - τ13[i,j,k]) * idz)
        tvd[i,j,k] += -((τ12[i+1,j,k] - τ12[i,j,k]) * idx +
                        (τ22[i,j,k] - τ22[i,j-1,k]) * idy +
                        (τ23[i,j,k+1] - τ23[i,j,k]) * idz)
        twd[i,j,k] += -((τ13[i+1,j,k] - τ13[i,j,k]) * idx +
                        (τ23[i,j+1,k] - τ23[i,j,k]) * idy +
                        (τ33[i,j,k] - τ33[i,j,k-1]) * idz)
    end
    return nothing
end

function _add_stress_divergence!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                 τ11, τ22, τ33, τ12, τ13, τ23, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    _fv!(_b_stress_div!, interior(g), tu.data, tv.data, tw.data,
         τ11, τ22, τ33, τ12, τ13, τ23, idx, idy, idz)
    return nothing
end

# τ = −2νᵢⱼSᵢⱼ into the six arrays, on exactly the index ranges
# `_add_stress_divergence!` reads (the interior plus the shell below for
# the diagonals, plus the shell above for the edges): one launch per
# component range.
@inline _b_τ11!(i, j, k, τ, ud, nh, idx, o) = (@inbounds τ[i,j,k] = _τ11(ud, nh, i, j, k, idx, o, o); nothing)
@inline _b_τ22!(i, j, k, τ, vd, nh, idy, o) = (@inbounds τ[i,j,k] = _τ22(vd, nh, i, j, k, idy, o, o); nothing)
@inline _b_τ33!(i, j, k, τ, wd, nv, idz, o) = (@inbounds τ[i,j,k] = _τ33(wd, nv, i, j, k, idz, o, o); nothing)
@inline _b_τ12!(i, j, k, τ, ud, vd, nh, idx, idy, o) =
    (@inbounds τ[i,j,k] = _τ12_u(ud, vd, nh, i, j, k, idx, idy, o, o); nothing)
@inline _b_τ13!(i, j, k, τ, ud, wd, nv, idx, idz, o) =
    (@inbounds τ[i,j,k] = _τ13_u(ud, wd, nv, i, j, k, idx, idz, o, o); nothing)
@inline _b_τ23!(i, j, k, τ, vd, wd, nv, idy, idz, o) =
    (@inbounds τ[i,j,k] = _τ23_v(vd, wd, nv, i, j, k, idy, idz, o, o); nothing)

function _sgs_stress_tensor!(τ11, τ22, τ33, τ12, τ13, τ23,
                             u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                             nh::AbstractArray{<:Real,3}, nv::AbstractArray{<:Real,3},
                             g::RegularGrid)
    T = eltype(τ11)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    o = one(T)
    ii, jj, kk = interior(g)
    ud, vd, wd = u.data, v.data, w.data
    xm = first(ii)-1:last(ii); ym = first(jj)-1:last(jj); zm = first(kk)-1:last(kk)
    xp = first(ii):last(ii)+1; yp = first(jj):last(jj)+1; zp = first(kk):last(kk)+1
    _fv!(_b_τ11!, (xm, jj, kk), τ11, ud, nh, idx, o)
    _fv!(_b_τ22!, (ii, ym, kk), τ22, vd, nh, idy, o)
    _fv!(_b_τ33!, (ii, jj, zm), τ33, wd, nv, idz, o)
    _fv!(_b_τ12!, (xp, yp, kk), τ12, ud, vd, nh, idx, idy, o)
    _fv!(_b_τ13!, (xp, jj, zp), τ13, ud, wd, nv, idx, idz, o)
    _fv!(_b_τ23!, (ii, yp, zp), τ23, vd, wd, nv, idy, idz, o)
    return nothing
end
# Σ 2νᵢⱼSᵢⱼSᵢⱼ ΔV over the interior cells (diagonals) and the edges with
# interior indices (off-diagonals, each counted for both ij and ji):
# the exact discrete dissipation of `_add_sgs_stress_divergence!` (a = b = 1)
# on a periodic domain.
function _sgs_stress_dissipation(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                 nh::AbstractArray{<:Real,3}, nv::AbstractArray{<:Real,3},
                                 g::RegularGrid)
    T = eltype(u.data)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    I3 = interior(g)
    # canonical plane sums, then ascending k (KA-b, #296)
    parts = plane_reduce!(zeros(T, length(I3[3])), Backends.backend(u.data), _sgs_dissipation_at,
                          I3, u.data, v.data, w.data, nh, nv, idx, idy, idz)
    return ordered_sum(parts) * (g.dx * g.dy * g.dz)
end

@inline function _sgs_dissipation_at(i, j, k, ud, vd, wd, nh, nv, idx, idy, idz)
    @inbounds begin
        S11 = (ud[i+1,j,k] - ud[i,j,k]) * idx
        S22 = (vd[i,j+1,k] - vd[i,j,k]) * idy
        S33 = (wd[i,j,k+1] - wd[i,j,k]) * idz
        S12 = 0.5 * ((ud[i,j,k] - ud[i,j-1,k]) * idy + (vd[i,j,k] - vd[i-1,j,k]) * idx)
        S13 = 0.5 * ((ud[i,j,k] - ud[i,j,k-1]) * idz + (wd[i,j,k] - wd[i-1,j,k]) * idx)
        S23 = 0.5 * ((vd[i,j,k] - vd[i,j,k-1]) * idz + (wd[i,j,k] - wd[i,j-1,k]) * idy)
        return 2 * (nh[i,j,k] * (S11^2 + S22^2) + nv[i,j,k] * S33^2) +
               4 * (_νe12(nh, i, j, k) * S12^2 + _νe13(nv, i, j, k) * S13^2 +
                    _νe23(nv, i, j, k) * S23^2)
    end
end

# --- eddy diffusion for a cell-centered scalar --------------------------
# s · [∂/∂x(Kh ∂ϕ/∂x) + ∂/∂y(Kh ∂ϕ/∂y) + ∂/∂z(Kv ∂ϕ/∂z)], each coefficient
# applied conservatively at the faces (interpolated linearly from cell
# centres). The single kernel behind every public form:
# `add_eddy_diffusion_scalar!(tϕ, ϕ, κt)` (Kh = Kv = κt, s = 1, exact),
# `add_eddy_diffusion_scalar!(tϕ, ϕ, νt, Pr_t)` (Kh = Kv = νt, s = 1/Pr_t)
# and the diagonal `add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv)` (P55-c,
# #181). With Kh === Kv the arithmetic is the pre-P55-c kernel's, bit for bit.
# On a stretched grid the z-face gradient spans the two centres (Δz_f) and
# the flux difference the cell (Δz_c), so Σ tϕ ΔV telescopes to the wall
# fluxes and the operator is symmetric negative semi-definite in the
# volume-weighted inner product.

_add_eddy_diffusion_scalar!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                            K::Field{CellCenter}, s::Real, g::AbstractMesh) =
    _add_eddy_diffusion_scalar!(tϕ, ϕ, K, K, s, g)

@inline function _b_eddy_scalar!(i, j, k, t, d, H, V, s, m)
    @inbounds begin
        idx = _idx(m, i); idy = _idy(m, j)
        νE = 0.5 * (H[i,j,k]   + H[i+1,j,k])
        νW = 0.5 * (H[i-1,j,k] + H[i,j,k])
        νN = 0.5 * (H[i,j,k]   + H[i,j+1,k])
        νS = 0.5 * (H[i,j-1,k] + H[i,j,k])
        νT = 0.5 * (V[i,j,k]   + V[i,j,k+1])
        νB = 0.5 * (V[i,j,k-1] + V[i,j,k])

        fxE = νE * (d[i+1,j,k] - d[i,  j,k]) * idx
        fxW = νW * (d[i,  j,k] - d[i-1,j,k]) * idx
        fyN = νN * (d[i,j+1,k] - d[i,j,  k]) * idy
        fyS = νS * (d[i,j,  k] - d[i,j-1,k]) * idy
        fzT = νT * (d[i,j,k+1] - d[i,j,k  ]) * _idzf(m, k+1)
        fzB = νB * (d[i,j,k  ] - d[i,j,k-1]) * _idzf(m, k)

        t[i,j,k] += s * ((fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*_idz(m, k))
    end
    return nothing
end

"""
    _add_eddy_diffusion_scalar!(tϕ, ϕ, Kh, Kv, s, g)

`s · [∂x(Kh ∂xϕ) + ∂y(Kh ∂yϕ) + ∂z(Kv ∂zϕ)]` with face-averaged
coefficients. On a stretched grid the z-face gradient spans the two
centres (`Δz_f`), the flux difference the cell (`Δz_c`), so `Σ tϕ ΔV`
telescopes to the wall fluxes and the operator is symmetric negative
semi-definite in the volume-weighted inner product.
"""
function _add_eddy_diffusion_scalar!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                      Kh::Field{CellCenter}, Kv::Field{CellCenter},
                                      s::Real, g::AbstractMesh)
    _fv!(_b_eddy_scalar!, interior(g), tϕ.data, ϕ.data, Kh.data, Kv.data, s, grid_metrics(g))
    return tϕ
end

# --- tensor eddy diffusion for a cell-centred scalar (P55-c, #181) ------
# ∂ᵢ(Kᵢⱼ ∂ⱼϕ) with the six components of a symmetric K at cell centres, in
# the triad (functional) form of Griffies et al. (1998, JPO 28, 805): each
# cell is split into eight octants, octant (a, b, c) pairing one x-face,
# one y-face and one z-face gradient of the cell, g = (gx, gy, gz), and
#
#     F[ϕ] = −½ Σ_cells (ΔV/8) Σ_octants gᵀ K_cell g,   ΔV ∂ₜϕₙ = ∂F/∂ϕₙ.
#
# Differentiating gives the flux form below, with the flux on a face the
# mean of (K g)ₙ over the eight octants that touch it. Because K is per
# cell, the four octant terms of a cell collapse to
#
#   Fx(i+½) = ½(K11ᵢ + K11ᵢ₊₁) gxᵢ₊½
#           + ¼ Σ_{c ∈ {i, i+1}} [K12_c (gy_c,S + gy_c,N) + K13_c (gz_c,B + gz_c,T)]
#
# and likewise Fy, Fz. The diagonal part is exactly the 2-point face average
# of `_add_eddy_diffusion_scalar!` (K11 on x, K22 on y, K33 on z faces).
# Consequences of the functional form: Σ ΔV ∂ₜϕ = 0 (F is shift-invariant),
# Σ ΔV ϕ ∂ₜϕ = 2F ≤ 0 when every K_cell is positive semi-definite, and
# the operator is symmetric.
#
# `bounded[d]`: the faces at the two ends of direction d are walls. Their
# gradient is not used in the cross terms of the interior faces and the wall
# flux has no cross part; its diagonal part K_nn g_n (from the halo) is
# kept, exactly as in the diagonal kernel (zero for a no-flux halo, for
# which the functional properties above hold with walls).
# `dodiag`, `dooff` select the diagonal and the off-diagonal part (IMEX).
# Fx, Fy, Fz: face-flux scratch, shaped like a CellCenter field's data;
# `Fx[i, j, k]` is the flux through the face between cells i and i+1.
#
# KA-d: one launch per face direction (each over its own extended range,
# e.g. i ∈ Hx:i2 for the x faces), then the divergence launch, which reads
# the finished fluxes. The face gradients of a cell are the named functions
# below (zero on a wall of a bounded direction).

# the face gradients of cell (i, j, k): (lo, hi) = the first and last
# interior index of the direction, b = bounded
@inline _tg_W(d, i, j, k, idx, b, lo) = @inbounds((b && i == lo) ? zero(eltype(d)) : (d[i, j, k] - d[i-1, j, k]) * idx)
@inline _tg_E(d, i, j, k, idx, b, hi) = @inbounds((b && i == hi) ? zero(eltype(d)) : (d[i+1, j, k] - d[i, j, k]) * idx)
@inline _tg_S(d, i, j, k, idy, b, lo) = @inbounds((b && j == lo) ? zero(eltype(d)) : (d[i, j, k] - d[i, j-1, k]) * idy)
@inline _tg_N(d, i, j, k, idy, b, hi) = @inbounds((b && j == hi) ? zero(eltype(d)) : (d[i, j+1, k] - d[i, j, k]) * idy)
@inline _tg_B(d, i, j, k, idz, b, lo) = @inbounds((b && k == lo) ? zero(eltype(d)) : (d[i, j, k] - d[i, j, k-1]) * idz)
@inline _tg_T(d, i, j, k, idz, b, hi) = @inbounds((b && k == hi) ? zero(eltype(d)) : (d[i, j, k+1] - d[i, j, k]) * idz)

# The bodies take the wall flags and the interior bounds as separate scalar
# arguments: bundled into one struct they made the x-face flux 2.3x the old
# loop (the compiler kept reloading the fields; measured at KA-d).
@inline function _b_tensor_fx!(i, j, k, Fx, d, K11, K12, K13, idx, idy, idz, bx, by, bz,
                               j1, j2, k1, k2, Hx, i2, dodiag, dooff)
    @inbounds begin
        T = eltype(d)
        f = zero(T)
        dodiag && (f += 0.5 * (K11[i, j, k] + K11[i+1, j, k]) * (d[i+1, j, k] - d[i, j, k]) * idx)
        if dooff && !(bx && (i == Hx || i == i2))
            f += 0.25 * (K12[i,   j, k] * (_tg_S(d, i,   j, k, idy, by, j1) + _tg_N(d, i,   j, k, idy, by, j2)) +
                         K12[i+1, j, k] * (_tg_S(d, i+1, j, k, idy, by, j1) + _tg_N(d, i+1, j, k, idy, by, j2)) +
                         K13[i,   j, k] * (_tg_B(d, i,   j, k, idz, bz, k1) + _tg_T(d, i,   j, k, idz, bz, k2)) +
                         K13[i+1, j, k] * (_tg_B(d, i+1, j, k, idz, bz, k1) + _tg_T(d, i+1, j, k, idz, bz, k2)))
        end
        Fx[i, j, k] = f
    end
    return nothing
end

@inline function _b_tensor_fy!(i, j, k, Fy, d, K22, K12, K23, idx, idy, idz, bx, by, bz,
                               i1, i2, k1, k2, Hy, j2, dodiag, dooff)
    @inbounds begin
        T = eltype(d)
        f = zero(T)
        dodiag && (f += 0.5 * (K22[i, j, k] + K22[i, j+1, k]) * (d[i, j+1, k] - d[i, j, k]) * idy)
        if dooff && !(by && (j == Hy || j == j2))
            f += 0.25 * (K12[i, j,   k] * (_tg_W(d, i, j,   k, idx, bx, i1) + _tg_E(d, i, j,   k, idx, bx, i2)) +
                         K12[i, j+1, k] * (_tg_W(d, i, j+1, k, idx, bx, i1) + _tg_E(d, i, j+1, k, idx, bx, i2)) +
                         K23[i, j,   k] * (_tg_B(d, i, j,   k, idz, bz, k1) + _tg_T(d, i, j,   k, idz, bz, k2)) +
                         K23[i, j+1, k] * (_tg_B(d, i, j+1, k, idz, bz, k1) + _tg_T(d, i, j+1, k, idz, bz, k2)))
        end
        Fy[i, j, k] = f
    end
    return nothing
end

@inline function _b_tensor_fz!(i, j, k, Fz, d, K33, K13, K23, idx, idy, idz, bx, by, bz,
                               i1, i2, j1, j2, Hz, k2, dodiag, dooff)
    @inbounds begin
        T = eltype(d)
        f = zero(T)
        dodiag && (f += 0.5 * (K33[i, j, k] + K33[i, j, k+1]) * (d[i, j, k+1] - d[i, j, k]) * idz)
        if dooff && !(bz && (k == Hz || k == k2))
            f += 0.25 * (K13[i, j, k  ] * (_tg_W(d, i, j, k,   idx, bx, i1) + _tg_E(d, i, j, k,   idx, bx, i2)) +
                         K13[i, j, k+1] * (_tg_W(d, i, j, k+1, idx, bx, i1) + _tg_E(d, i, j, k+1, idx, bx, i2)) +
                         K23[i, j, k  ] * (_tg_S(d, i, j, k,   idy, by, j1) + _tg_N(d, i, j, k,   idy, by, j2)) +
                         K23[i, j, k+1] * (_tg_S(d, i, j, k+1, idy, by, j1) + _tg_N(d, i, j, k+1, idy, by, j2)))
        end
        Fz[i, j, k] = f
    end
    return nothing
end

@inline function _b_flux_divergence!(i, j, k, t, Fx, Fy, Fz, idx, idy, idz)
    @inbounds t[i, j, k] += (Fx[i, j, k] - Fx[i-1, j, k]) * idx +
                            (Fy[i, j, k] - Fy[i, j-1, k]) * idy +
                            (Fz[i, j, k] - Fz[i, j, k-1]) * idz
    return nothing
end

function _add_tensor_diffusion_scalar!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                       K11, K22, K33, K12, K13, K23, Fx, Fy, Fz,
                                       bounded::NTuple{3,Bool}, dodiag::Bool, dooff::Bool,
                                       g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    d = ϕ.data
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    i1, i2 = Hx + 1, Hx + g.Nx
    j1, j2 = Hy + 1, Hy + g.Ny
    k1, k2 = Hz + 1, Hz + g.Nz
    bx, by, bz = bounded
    # x faces i+½ (i ∈ Hx:i2), y faces j+½, z faces k+½
    _fv!(_b_tensor_fx!, (Hx:i2, j1:j2, k1:k2), Fx, d, K11, K12, K13, idx, idy, idz, bx, by, bz,
         j1, j2, k1, k2, Hx, i2, dodiag, dooff)
    _fv!(_b_tensor_fy!, (i1:i2, Hy:j2, k1:k2), Fy, d, K22, K12, K23, idx, idy, idz, bx, by, bz,
         i1, i2, k1, k2, Hy, j2, dodiag, dooff)
    _fv!(_b_tensor_fz!, (i1:i2, j1:j2, Hz:k2), Fz, d, K33, K13, K23, idx, idy, idz, bx, by, bz,
         i1, i2, j1, j2, Hz, k2, dodiag, dooff)
    _fv!(_b_flux_divergence!, (i1:i2, j1:j2, k1:k2), tϕ.data, Fx, Fy, Fz, idx, idy, idz)
    return tϕ
end

# Masked and/or mass-weighted form of the same flux divergence (P56-a, #177):
#
#   * `mask` (Bool, storage-sized, halos consistent with ϕ's BCs; or nothing):
#     a face with a solid cell on either side carries no flux, and a solid
#     cell's tendency is left untouched. The flux on every open face is the
#     unmasked one, so Σ_fluid tϕ dV telescopes to zero.
#   * `wc`, `wf` (per storage level; or nothing): the sound-proof form
#     (1/W_k) ∇·(W κ ∇ϕ) with the cell weight `wc[k]` and the z-face weight
#     `wf[k]` (bottom face of cell k, the FaceZ storage index). The cell
#     weight cancels in the horizontal fluxes, so only the vertical flux is
#     weighted; Σ W_k tϕ_k telescopes to zero.
#
# Kept separate from the kernel above so the unmasked, unweighted path stays
# bit for bit what it was.
#
# `Kh` on the x and y faces, `Kv` on the z faces (the diagonal operator of
# P55-c with a mask; a stability-corrected closure's κh ≠ κv, P55-e). With
# Kh === Kv (the one-coefficient method) the arithmetic is unchanged.
_add_eddy_diffusion_scalar_masked!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                   K::Field{CellCenter}, g::RegularGrid, mask, wc, wf) =
    _add_eddy_diffusion_scalar_masked!(tϕ, ϕ, K, K, g, mask, wc, wf)

@inline function _b_eddy_scalar_masked!(i, j, k, t, ϕd, Kd, Vd, mask, wc, wf, idx, idy, idz)
    @inbounds begin
        T = eltype(ϕd)
        # branchless: each face flux times its openness (1 or 0), the cell's
        # whole tendency times its own (a solid cell gets + 0)
        oC = _open(T, mask, i, j, k)
        oE = oC * _open(T, mask, i+1, j, k); oW = oC * _open(T, mask, i-1, j, k)
        oN = oC * _open(T, mask, i, j+1, k); oS = oC * _open(T, mask, i, j-1, k)
        oT = oC * _open(T, mask, i, j, k+1); oB = oC * _open(T, mask, i, j, k-1)

        fxE = oE * (0.5 * (Kd[i,j,k]   + Kd[i+1,j,k]) * (ϕd[i+1,j,k] - ϕd[i,  j,k]) * idx)
        fxW = oW * (0.5 * (Kd[i-1,j,k] + Kd[i,j,k])   * (ϕd[i,  j,k] - ϕd[i-1,j,k]) * idx)
        fyN = oN * (0.5 * (Kd[i,j,k]   + Kd[i,j+1,k]) * (ϕd[i,j+1,k] - ϕd[i,j,  k]) * idy)
        fyS = oS * (0.5 * (Kd[i,j-1,k] + Kd[i,j,k])   * (ϕd[i,j,  k] - ϕd[i,j-1,k]) * idy)
        fzT = oT * (0.5 * (Vd[i,j,k]   + Vd[i,j,k+1]) * (ϕd[i,j,k+1] - ϕd[i,j,k  ]) * idz)
        fzB = oB * (0.5 * (Vd[i,j,k-1] + Vd[i,j,k])   * (ϕd[i,j,k  ] - ϕd[i,j,k-1]) * idz)

        vert = _vertical_divergence(fzT, fzB, idz, wc, wf, k)
        t[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + vert
    end
    return nothing
end

function _add_eddy_diffusion_scalar_masked!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                             Kh::Field{CellCenter}, Kv::Field{CellCenter},
                                             g::RegularGrid, mask, wc, wf)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    _fv!(_b_eddy_scalar_masked!, interior(g), tϕ.data, ϕ.data, Kh.data, Kv.data,
         mask, wc, wf, idx, idy, idz)
    return tϕ
end

# --- the rotated isentropic tensor (P55-c, #181) --------------------------
# K = κh I − f (κh − κv) n nᵀ, n = ∇θ/|∇θ| (centred differences), with
# f = f_slope · f_grad: f_slope = min(1, (S_max/S)²), S = |∇ₕθ|/∂zθ
# (Gerdes et al. 1991), 0 where ∂zθ ≤ 0 (N² ≤ 0); f_grad = min(1, |∇θ|/g_min).
# κh, κv: cell-centred arrays or numbers (`_kval`). The body writes the six
# components at its own point: one launch.
@inline _kval(a::AbstractArray, i, j, k) = @inbounds a[i, j, k]
@inline _kval(a::Real, i, j, k) = a

@inline function _b_isentropic!(i, j, k, K11, K22, K33, K12, K13, K23, d, κh, κv,
                                slope_max, grad_min, i2dx, i2dy, i2dz)
    @inbounds begin
        gx = (d[i+1, j, k] - d[i-1, j, k]) * i2dx
        gy = (d[i, j+1, k] - d[i, j-1, k]) * i2dy
        gz = (d[i, j, k+1] - d[i, j, k-1]) * i2dz
        kh = _kval(κh, i, j, k); kv = _kval(κv, i, j, k)
        G2h = gx * gx + gy * gy
        G = sqrt(G2h + gz * gz)
        f = zero(G)
        if gz > 0 && G > 0
            # S² = |∇ₕθ|²/∂zθ²; taper (S_max/S)² above S_max
            S2 = G2h / (gz * gz)
            f = S2 <= slope_max^2 ? one(G) : slope_max^2 / S2
            G < grad_min && (f *= G / grad_min)
        end
        a = f * (kh - kv) / (G > 0 ? G * G : one(G))     # f (κh − κv) / |∇θ|²
        K11[i, j, k] = kh - a * gx * gx
        K22[i, j, k] = kh - a * gy * gy
        K33[i, j, k] = kh - a * gz * gz
        K12[i, j, k] = -a * gx * gy
        K13[i, j, k] = -a * gx * gz
        K23[i, j, k] = -a * gy * gz
    end
    return nothing
end

function _isentropic_tensor!(K11, K22, K33, K12, K13, K23, θ::Field{CellCenter},
                             κh, κv, slope_max::Real, grad_min::Real, g::RegularGrid)
    i2dx = 1 / (2 * g.dx); i2dy = 1 / (2 * g.dy); i2dz = 1 / (2 * g.dz)
    _fv!(_b_isentropic!, interior(g), K11, K22, K33, K12, K13, K23, θ.data, κh, κv,
         slope_max, grad_min, i2dx, i2dy, i2dz)
    return nothing
end

@inline _open(::Type{T}, ::Nothing, i, j, k) where {T} = one(T)
@inline _open(::Type{T}, mask, i, j, k) where {T} = T(!mask[i,j,k])
@inline _vertical_divergence(fzT, fzB, idz, ::Nothing, ::Nothing, k) = (fzT - fzB) * idz
@inline _vertical_divergence(fzT, fzB, idz, wc, wf, k) = (wf[k+1] * fzT - wf[k] * fzB) * idz / wc[k]
# the same for a w-located tendency: its vertical fluxes sit at the cell
# centres k (top) and k-1 (bottom), its control volume around face k
@inline _vertical_divergence_w(fzT, fzB, idz, ::Nothing, ::Nothing, k) = (fzT - fzB) * idz
@inline _vertical_divergence_w(fzT, fzB, idz, wc, wf, k) = (wc[k] * fzT - wc[k-1] * fzB) * idz / wf[k]
# =========================================================================
# WENO-5 scalar advection (Phase 10) — staggered.
# =========================================================================
# Weighted Essentially Non-Oscillatory 5th-order upwind reconstruction of
# the scalar at each face, then flux differencing. Requires halo ≥ 3 in
# every direction (5-cell stencil biased upstream).
#
# Reference: Jiang & Shu (1996), JCP 126.

# NOTE on the parameter name: the centre sample is called `fc`, not `f0`.
# In Julia `2f0` is the *Float32 literal* 2.0f0, not `2 * f0` — the
# `<digits>f<digits>` form is Float32 scientific notation. An earlier
# version of this function used `f0` and wrote `2f0` / `3f0` in the
# smoothness indicators, which silently substituted the constants 2.0f0
# and 3.0f0 for `2*f0` and `3*f0`. That broke translation invariance,
# degraded the scheme from 5th to ~3rd order, and made it *produce*
# scalar variance instead of dissipating it. Renaming the argument makes
# the mistake unrepresentable.
@inline function _weno5_left(fm2, fm1, fc, fp1, fp2)
    q0 = (1/3)*fm2 - (7/6)*fm1 + (11/6)*fc
    q1 = (-1/6)*fm1 + (5/6)*fc + (1/3)*fp1
    q2 = (1/3)*fc + (5/6)*fp1 - (1/6)*fp2
    b0 = (13/12)*(fm2 - 2fm1 + fc)^2 + (1/4)*(fm2 - 4fm1 + 3fc)^2
    b1 = (13/12)*(fm1 - 2fc  + fp1)^2 + (1/4)*(fm1 - fp1)^2
    b2 = (13/12)*(fc  - 2fp1 + fp2)^2 + (1/4)*(3fc - 4fp1 + fp2)^2
    ε = 1e-6
    a0 = (1/10) / (b0 + ε)^2
    a1 = (6/10) / (b1 + ε)^2
    a2 = (3/10) / (b2 + ε)^2
    s = a0 + a1 + a2
    return (a0*q0 + a1*q1 + a2*q2) / s
end

# Right-biased reconstruction = left-biased on the mirrored stencil.
@inline _weno5_right(fm1, f0, fp1, fp2, fp3) = _weno5_left(fp3, fp2, fp1, f0, fm1)

@inline function _b_weno5!(i, j, k, tend, ud, vd, wd, ϕd, idx, idy, idz)
    @inbounds begin
        # --- x direction ---
        uw = ud[i,   j, k]
        ue = ud[i+1, j, k]
        ϕx_w = uw >= 0 ?
            _weno5_left( ϕd[i-3,j,k], ϕd[i-2,j,k], ϕd[i-1,j,k], ϕd[i,j,k],   ϕd[i+1,j,k]) :
            _weno5_right(ϕd[i-2,j,k], ϕd[i-1,j,k], ϕd[i,j,k],   ϕd[i+1,j,k], ϕd[i+2,j,k])
        ϕx_e = ue >= 0 ?
            _weno5_left( ϕd[i-2,j,k], ϕd[i-1,j,k], ϕd[i,j,k],   ϕd[i+1,j,k], ϕd[i+2,j,k]) :
            _weno5_right(ϕd[i-1,j,k], ϕd[i,j,k],   ϕd[i+1,j,k], ϕd[i+2,j,k], ϕd[i+3,j,k])
        Fx = (ue*ϕx_e - uw*ϕx_w) * idx

        # --- y direction ---
        vs = vd[i, j,   k]
        vn = vd[i, j+1, k]
        ϕy_s = vs >= 0 ?
            _weno5_left( ϕd[i,j-3,k], ϕd[i,j-2,k], ϕd[i,j-1,k], ϕd[i,j,k],   ϕd[i,j+1,k]) :
            _weno5_right(ϕd[i,j-2,k], ϕd[i,j-1,k], ϕd[i,j,k],   ϕd[i,j+1,k], ϕd[i,j+2,k])
        ϕy_n = vn >= 0 ?
            _weno5_left( ϕd[i,j-2,k], ϕd[i,j-1,k], ϕd[i,j,k],   ϕd[i,j+1,k], ϕd[i,j+2,k]) :
            _weno5_right(ϕd[i,j-1,k], ϕd[i,j,k],   ϕd[i,j+1,k], ϕd[i,j+2,k], ϕd[i,j+3,k])
        Fy = (vn*ϕy_n - vs*ϕy_s) * idy

        # --- z direction ---
        wb = wd[i, j, k]
        wt = wd[i, j, k+1]
        ϕz_b = wb >= 0 ?
            _weno5_left( ϕd[i,j,k-3], ϕd[i,j,k-2], ϕd[i,j,k-1], ϕd[i,j,k],   ϕd[i,j,k+1]) :
            _weno5_right(ϕd[i,j,k-2], ϕd[i,j,k-1], ϕd[i,j,k],   ϕd[i,j,k+1], ϕd[i,j,k+2])
        ϕz_t = wt >= 0 ?
            _weno5_left( ϕd[i,j,k-2], ϕd[i,j,k-1], ϕd[i,j,k],   ϕd[i,j,k+1], ϕd[i,j,k+2]) :
            _weno5_right(ϕd[i,j,k-1], ϕd[i,j,k],   ϕd[i,j,k+1], ϕd[i,j,k+2], ϕd[i,j,k+3])
        Fz = (wt*ϕz_t - wb*ϕz_b) * idz

        tend[i, j, k] -= Fx + Fy + Fz
    end
    return nothing
end

function _advect_weno5!(tend::Field{CellCenter},
                        u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                        ϕ::Field{CellCenter}, g::RegularGrid)
    (g.Hx >= 3 && g.Hy >= 3 && g.Hz >= 3) ||
        error("WENO-5 requires halo ≥ 3 in every direction (got $(g.Hx),$(g.Hy),$(g.Hz))")
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    _fv!(_b_weno5!, interior(g), tend.data, u.data, v.data, w.data, ϕ.data, idx, idy, idz)
    return tend
end

# =========================================================================
# Scalar advection (Phase 0 — collocated).
# =========================================================================

@inline function _b_advect_centered_cc!(i, j, k, tend, u, v, w, ϕ, inv2dx, inv2dy, inv2dz)
    @inbounds begin
        dϕdx = (ϕ[i+1,j,k] - ϕ[i-1,j,k]) * inv2dx
        dϕdy = (ϕ[i,j+1,k] - ϕ[i,j-1,k]) * inv2dy
        dϕdz = (ϕ[i,j,k+1] - ϕ[i,j,k-1]) * inv2dz
        tend[i,j,k] -= u[i,j,k]*dϕdx + v[i,j,k]*dϕdy + w[i,j,k]*dϕdz
    end
    return nothing
end

function _advect_centered!(tend::Field{CellCenter},
                          u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                          ϕ::Field{CellCenter}, g::RegularGrid)
    inv2dx = 1 / (2g.dx); inv2dy = 1 / (2g.dy); inv2dz = 1 / (2g.dz)
    _fv!(_b_advect_centered_cc!, interior(g), tend.data, u.data, v.data, w.data, ϕ.data,
         inv2dx, inv2dy, inv2dz)
    return tend
end

# --- Staggered scalar advection (flux form) -----------------------------
# Adds -∂(u_j ϕ)/∂x_j to `tend` when the transporting velocity lives on
# faces and ϕ lives at cell centers. Used for the potential-temperature
# equation in the Boussinesq core. The vertical flux difference is divided
# by the cell thickness Δz_c[k], so Σ ϕ ΔV changes only by the boundary
# fluxes on a stretched grid too; the face value is the arithmetic mean of
# the two neighbours, which keeps Σ ϕ² ΔV invariant under a solenoidal flow.

# the flux divergence ∂(uⱼϕ)/∂xⱼ at one point
@inline function _adv_centered_at(i, j, k, u, v, w, d, m)
    @inbounds begin
        uϕe = u[i+1,j,k] * 0.5 * (d[i,  j,k] + d[i+1,j,k])
        uϕw = u[i,  j,k] * 0.5 * (d[i-1,j,k] + d[i,  j,k])
        vϕn = v[i,j+1,k] * 0.5 * (d[i,j,  k] + d[i,j+1,k])
        vϕs = v[i,j,  k] * 0.5 * (d[i,j-1,k] + d[i,j,  k])
        wϕt = w[i,j,k+1] * 0.5 * (d[i,j,k  ] + d[i,j,k+1])
        wϕb = w[i,j,k  ] * 0.5 * (d[i,j,k-1] + d[i,j,k  ])
        r = (uϕe - uϕw)*_idx(m, i) + (vϕn - vϕs)*_idy(m, j) + (wϕt - wϕb)*_idz(m, k)
    end
    return r
end
@inline _b_advect_centered!(i, j, k, tend, u, v, w, d, m) =
    (@inbounds tend[i,j,k] -= _adv_centered_at(i, j, k, u, v, w, d, m); nothing)

"""
    _advect_centered!(tend, u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, ϕ, g)

Flux-form `-∂(uⱼϕ)/∂xⱼ` for a cell-centred scalar with arithmetic face
means; the vertical flux difference is divided by the cell thickness
`Δz_c[k]`, so `Σ ϕ ΔV` changes only by the boundary fluxes and
`Σ ϕ² ΔV` is invariant under a solenoidal flow on any face distribution.
"""
function _advect_centered!(tend::Field{CellCenter},
                          u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                          ϕ::Field{CellCenter}, g::AbstractMesh)
    _fv!(_b_advect_centered!, interior(g), tend.data, u.data, v.data, w.data, ϕ.data, grid_metrics(g))
    return tend
end

@inline function _b_advect_upwind!(i, j, k, tend, u, v, w, ϕ, idx, idy, idz)
    @inbounds begin
        ui = u[i,j,k]; vi = v[i,j,k]; wi = w[i,j,k]
        dϕdx = ui >= 0 ? (ϕ[i,j,k] - ϕ[i-1,j,k]) * idx :
                         (ϕ[i+1,j,k] - ϕ[i,j,k]) * idx
        dϕdy = vi >= 0 ? (ϕ[i,j,k] - ϕ[i,j-1,k]) * idy :
                         (ϕ[i,j+1,k] - ϕ[i,j,k]) * idy
        dϕdz = wi >= 0 ? (ϕ[i,j,k] - ϕ[i,j,k-1]) * idz :
                         (ϕ[i,j,k+1] - ϕ[i,j,k]) * idz
        tend[i,j,k] -= ui*dϕdx + vi*dϕdy + wi*dϕdz
    end
    return nothing
end

function _advect_upwind!(tend::Field{CellCenter},
                        u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                        ϕ::Field{CellCenter}, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    _fv!(_b_advect_upwind!, interior(g), tend.data, u.data, v.data, w.data, ϕ.data, idx, idy, idz)
    return tend
end

# =========================================================================
# Fused step-core kernels (SC-d, #404)
# =========================================================================
# The Boussinesq step cores (`Simulations._euler_projection!`) used to build
# each substage from single-purpose passes: zero the tendencies, add the
# momentum advection (three launches, each reading u, v and w), add the
# scalar advection, write ∇²f into a scratch field and add ν∇²f with an
# axpy; then ∂φ/∂x into a scratch field and an axpy for the projection.
# These bodies do the same arithmetic, in the same order, in one pass per
# group, so each field is streamed once (docs/src/design/ka_port.md,
# "SC-d"). Every value is bit for bit the one of the separate passes
# (test/ka_fusion.jl (removed)).

# t + ν∇²f, the molecular viscosity / diffusion (`axpy!(t, ν, ∇²f)`); none
# when the core has no viscosity
@inline _visc(::Nothing, t, i, j, k, f, m, loc) = t
@inline _visc(ν, t, i, j, k, f, m, loc) = t + ν * _lap_at(i, j, k, f, m, loc)

# The tendencies of a substage before the SGS, buoyancy and forcing terms,
# each written once: t = 0 − ∂(uⱼuᵢ)/∂xⱼ (+ ν∇²uᵢ) for the three
# components in one launch (u, v, w read once), t = 0 − ∂(uⱼϕ)/∂xⱼ
# (+ ν∇²ϕ) for θ and for qv in one launch each: `zero_storage!`, the
# advection's read-modify-write and the Laplacian + axpy of the viscosity
# in one pass. Measured (ms, 1 / 12 threads; ka_port.md, SC-d): all five tendencies in one launch, 82 / 48 on a viscous
# 256³ box but 33 / 6.5 on the moist 192 × 192 × 53 parent (the loop
# stops vectorising with five stores); one launch per tendency 109 / 66
# and 15 / 6.7; this split 86 / 53 and 12 / 6.1.
# The tendency values at one point (also those of the blocked sweep below,
# SC-e #405).
@inline _tend_u(i, j, k, u, v, w, m, ν) =
    _visc(ν, zero(eltype(u)) - _mom_u_at(i, j, k, u, v, w, m), i, j, k, u, m, FaceX())
@inline _tend_v(i, j, k, u, v, w, m, ν) =
    _visc(ν, zero(eltype(v)) - _mom_v_at(i, j, k, u, v, w, m), i, j, k, v, m, FaceY())
@inline _tend_w(i, j, k, u, v, w, m, ν) =
    _visc(ν, zero(eltype(w)) - _mom_w_at(i, j, k, u, v, w, m), i, j, k, w, m, FaceZ())
@inline _tend_ϕ(i, j, k, u, v, w, ϕ, m, ν) =
    _visc(ν, zero(eltype(ϕ)) - _adv_centered_at(i, j, k, u, v, w, ϕ, m), i, j, k, ϕ, m, CellCenter())

@inline function _b_tendency_momentum!(i, j, k, tu, tv, tw, u, v, w, m, ν)
    @inbounds begin
        tu[i, j, k] = _tend_u(i, j, k, u, v, w, m, ν)
        tv[i, j, k] = _tend_v(i, j, k, u, v, w, m, ν)
        tw[i, j, k] = _tend_w(i, j, k, u, v, w, m, ν)
    end
    return nothing
end
@inline function _b_tendency_scalar!(i, j, k, t, u, v, w, ϕ, m, ν)
    @inbounds t[i, j, k] = _tend_ϕ(i, j, k, u, v, w, ϕ, m, ν)
    return nothing
end

_scalar_tendency!(::Nothing, I, u, v, w, m, ν) = nothing
_scalar_tendency!((t, ϕ)::Tuple{Field{CellCenter}, Field{CellCenter}}, I, u, v, w, m, ν) =
    _fv!(_b_tendency_scalar!, I, t.data, u.data, v.data, w.data, ϕ.data, m, ν)

# _fused_tendencies!(tu, tv, tw, u, v, w, ν, s1, s2, g)
#
# Overwrite the interior of `tu, tv, tw` with `-∂(uⱼuᵢ)/∂xⱼ + ν∇²uᵢ` and,
# for each scalar pair `s = (tϕ, ϕ)` (or `nothing`), `tϕ` with
# `-∂(uⱼϕ)/∂xⱼ + ν∇²ϕ`; `ν = nothing` drops the diffusion. Write-only
# launches (momentum, each scalar), the same bits as zeroing the tendencies,
# `_momentum_advect!`, `_advect_centered!` and adding `ν` times
# `_laplacian!` (SC-d, #404). Halos of the tendencies are not written;
# halos of `u, v, w` and the scalars must be filled.
function _fused_tendencies!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                            u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                            ν, s1, s2, g::AbstractMesh)
    m = grid_metrics(g)
    I = interior(g)
    _fv!(_b_tendency_momentum!, I, tu.data, tv.data, tw.data, u.data, v.data, w.data, m, ν)
    _scalar_tendency!(s1, I, u, v, w, m, ν)
    _scalar_tendency!(s2, I, u, v, w, m, ν)
    return nothing
end

# --- the projection's correction ------------------------------------------

@inline _stage(::Nothing, x, s) = x
@inline _stage(c, x, s) = c(x, s)
@inline _point(::Nothing, i, j, k) = nothing
@inline _point(a::AbstractArray, i, j, k) = @inbounds a[i, j, k]
@inline _set_pressure!(::Nothing, pf, φc, i, j, k) = nothing
@inline _set_pressure!(p, pf, φc, i, j, k) = (@inbounds p[i, j, k] = pf(φc); nothing)

# u ← u − ∂φ/∂x (likewise v, w): `∂x(dφdx, φ)` and `axpy!(u, -1, dφdx)`
# in one pass (u + (−1)·d is exactly u − d). With `p`, the pressure
# p = pf(φ) (`pressure_from_potential!`) at the same point; with a stage
# combination `c` and the step-start state `u0, v0, w0`, the combined
# c(u − ∂φ/∂x, u0) (the RK3 combination after the projection).
@inline function _b_correct!(i, j, k, u, v, w, φ, m, p, pf, c, u0, v0, w0)
    @inbounds begin
        φc = φ[i, j, k]
        u[i, j, k] = _stage(c, u[i, j, k] - (φc - φ[i-1, j, k]) * _idx(m, i), _point(u0, i, j, k))
        v[i, j, k] = _stage(c, v[i, j, k] - (φc - φ[i, j-1, k]) * _idy(m, j), _point(v0, i, j, k))
        w[i, j, k] = _stage(c, w[i, j, k] - (φc - φ[i, j, k-1]) * _idzf(m, k), _point(w0, i, j, k))
    end
    _set_pressure!(p, pf, φc, i, j, k)
    return nothing
end

# _correct_velocity!(u, v, w, φ, g; p = nothing, pf = nothing,
# stage = nothing, u0 = nothing, v0 = nothing, w0 = nothing)
#
# The pressure correction `u ← u − ∇φ` over the interior, in one launch
# (the centre → face gradients of `_∂x!` and their axpys, the same
# bits). With `p` (an array) also `p = pf(φ)`, and with `stage` (a
# callable `(x, snap) -> …`) the corrected value is combined with the
# step-start arrays `u0, v0, w0` (SC-d, #404). Halos of `φ` must be filled.
function _correct_velocity!(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                            φ::Field{CellCenter}, g::AbstractMesh;
                            p = nothing, pf = nothing, stage = nothing,
                            u0 = nothing, v0 = nothing, w0 = nothing)
    _fv!(_b_correct!, interior(g), u.data, v.data, w.data, φ.data, grid_metrics(g),
         p, pf, stage, u0, v0, w0)
    return nothing
end

# The centre → face gradients of φ on the west (x), south (y) and bottom
# (z) faces only (the interior slots H+1), where an open boundary's check
# (`assert_open_face_correction`) reads them: the same values `_∂x!` &c.
# write there.
function _face_gradient_slabs!(dφdx::Field{FaceX}, dφdy::Field{FaceY}, dφdz::Field{FaceZ},
                               φ::Field{CellCenter}, g::AbstractMesh,
                               west::Bool, south::Bool, bottom::Bool)
    ii, jj, kk = interior(g)
    m = grid_metrics(g)
    west   && _fv!(_b_∂x_cf!, (first(ii):first(ii), jj, kk), dφdx.data, φ.data, m)
    south  && _fv!(_b_∂y_cf!, (ii, first(jj):first(jj), kk), dφdy.data, φ.data, m)
    bottom && _fv!(_b_∂z_cf!, (ii, jj, first(kk):first(kk)), dφdz.data, φ.data, m)
    return nothing
end

# ∇·u written straight into an interior-sized array (a spectral solver's
# source buffer): buf[i, j, k] is the divergence at storage point
# (i + oi, j + oj, k + ok), the bits of `_divergence!` then the copy-in.
@inline _b_divergence_into!(i, j, k, buf, u, v, w, m, oi, oj, ok) =
    (@inbounds buf[i, j, k] = _div_at(i + oi, j + oj, k + ok, u, v, w, m); nothing)

# _divergence_into!(buf, u, v, w, g)
#
# The Arakawa-C divergence of `_divergence!` over the interior,
# written into the interior-sized array `buf` (`buf[i, j, k]` ↔ storage
# `(i + Hx, j + Hy, k + Hz)`): a spectral solver's source buffer
# (`PressureSolvers.source_buffer`, SC-d #404). Halos of `u, v, w` must be
# filled.
function _divergence_into!(buf::AbstractArray{T,3}, u::Field{FaceX}, v::Field{FaceY},
                           w::Field{FaceZ}, g::AbstractMesh) where {T}
    ii, jj, kk = interior(g)
    size(buf) == (length(ii), length(jj), length(kk)) ||
        throw(DimensionMismatch("_divergence_into!: buffer $(size(buf)) ≠ interior $((length(ii), length(jj), length(kk)))"))
    checkbounds(u.data, first(ii) - 1:last(ii) + 1, jj, kk)
    _fv!(_b_divergence_into!, (1:length(ii), 1:length(jj), 1:length(kk)), buf,
         u.data, v.data, w.data, grid_metrics(g), first(ii) - 1, first(jj) - 1, first(kk) - 1)
    return buf
end

# The fused sweep of the step core: the local-term interface and its
# driver (SC-e #405, #533).
include("operators/fused_sweep.jl")
# the budgets inside the sweep: per-level sums of what booked terms add (#533 plan step 7)
include("operators/fused_sweep_budgets.jl")
# its pointwise terms: buoyancy, Coriolis, boundary fluxes, damping (#533 plan step 2)
include("operators/sweep_pointwise.jl")

end # module
