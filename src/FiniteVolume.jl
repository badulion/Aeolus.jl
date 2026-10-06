module FiniteVolume

# Finite-volume kernels for `RegularGrid`. Each kernel dispatches on
# a `::RegularGrid` in the trailing mesh argument AND on the location
# parameters of its input/output `Field`s. This is how staggered vs.
# collocated stencils are chosen: e.g. `_∂x!(out::Field{FaceX}, f::Field{CellCenter}, g)`
# is the pressure-gradient stencil and is a different method from the
# collocated `_∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g)`.
#
# Halos of input fields must be filled before calling these kernels.
#
# Kernel signatures follow the Arakawa C-grid convention (Phase 1a):
#   * u lives on FaceX (west face of each cell)
#   * v lives on FaceY (south face)
#   * w lives on FaceZ (bottom face)
#   * p, θ, tracers on CellCenter.

using ..Grids
using ..Grids: RegularGrid, interior, CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields: Field
using ..Operators: _strain_at

export AbstractRamp, SinSquaredRamp, NordstromRamp, smooth_step,
       DampingProfile, sponge_profile, in_damping, damping_rate, damping_coordinates

# =========================================================================
# First derivatives
# =========================================================================

# --- collocated cell-center → cell-center (2nd-order centered) -----------
# Kept for scalar diagnostics; not used in the Boussinesq core.

function _∂x!(out::Field{CellCenter}, f::Field{CellCenter}, g::RegularGrid)
    inv2dx = 1 / (2g.dx)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i+1, j, k] - f.data[i-1, j, k]) * inv2dx
    end
    return out
end

function _∂y!(out::Field{CellCenter}, f::Field{CellCenter}, g::RegularGrid)
    inv2dy = 1 / (2g.dy)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i, j+1, k] - f.data[i, j-1, k]) * inv2dy
    end
    return out
end

function _∂z!(out::Field{CellCenter}, f::Field{CellCenter}, g::RegularGrid)
    inv2dz = 1 / (2g.dz)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i, j, k+1] - f.data[i, j, k-1]) * inv2dz
    end
    return out
end

# --- cell-center → face (pressure gradient etc., 2-point stencil) --------
# west face of cell (i,j,k) sits between (i-1) and (i); grad = (c[i] - c[i-1])/dx.

function _∂x!(out::Field{FaceX}, f::Field{CellCenter}, g::RegularGrid)
    idx = 1 / g.dx
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i, j, k] - f.data[i-1, j, k]) * idx
    end
    return out
end

function _∂y!(out::Field{FaceY}, f::Field{CellCenter}, g::RegularGrid)
    idy = 1 / g.dy
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i, j, k] - f.data[i, j-1, k]) * idy
    end
    return out
end

function _∂z!(out::Field{FaceZ}, f::Field{CellCenter}, g::RegularGrid)
    idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (f.data[i, j, k] - f.data[i, j, k-1]) * idz
    end
    return out
end

# --- face → cell-center (used by divergence and diagnostics) --------------
# The east face of cell (i,j,k) is index i+1 for u; hence
# ∂u/∂x at cell center = (u[i+1] - u[i]) / dx.

function _∂x!(out::Field{CellCenter}, u::Field{FaceX}, g::RegularGrid)
    idx = 1 / g.dx
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (u.data[i+1, j, k] - u.data[i, j, k]) * idx
    end
    return out
end

function _∂y!(out::Field{CellCenter}, v::Field{FaceY}, g::RegularGrid)
    idy = 1 / g.dy
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (v.data[i, j+1, k] - v.data[i, j, k]) * idy
    end
    return out
end

function _∂z!(out::Field{CellCenter}, w::Field{FaceZ}, g::RegularGrid)
    idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i, j, k] = (w.data[i, j, k+1] - w.data[i, j, k]) * idz
    end
    return out
end

# =========================================================================
# Laplacian — same 7-point stencil at every location on a uniform grid.
# The output must live at the same location as the input.
# =========================================================================

function _laplacian!(out::Field{L}, f::Field{L}, g::RegularGrid) where {L<:Location}
    idx2 = 1 / g.dx^2
    idy2 = 1 / g.dy^2
    idz2 = 1 / g.dz^2
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        c = f.data[i, j, k]
        out.data[i, j, k] = (f.data[i+1,j,k] - 2c + f.data[i-1,j,k]) * idx2 +
                            (f.data[i,j+1,k] - 2c + f.data[i,j-1,k]) * idy2 +
                            (f.data[i,j,k+1] - 2c + f.data[i,j,k-1]) * idz2
    end
    return out
end

# =========================================================================
# Divergence
# =========================================================================

# Staggered: u on FaceX, v on FaceY, w on FaceZ → cell-center.
function _divergence!(out::Field{CellCenter},
                     u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                     g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = (u.data[i+1,j,k] - u.data[i,j,k]) * idx +
                          (v.data[i,j+1,k] - v.data[i,j,k]) * idy +
                          (w.data[i,j,k+1] - w.data[i,j,k]) * idz
    end
    return out
end

# Collocated fallback (all at cell centers) — used for diagnostics only.
function _divergence!(out::Field{CellCenter},
                     u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                     g::RegularGrid)
    inv2dx = 1 / (2g.dx); inv2dy = 1 / (2g.dy); inv2dz = 1 / (2g.dz)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = (u.data[i+1,j,k] - u.data[i-1,j,k]) * inv2dx +
                          (v.data[i,j+1,k] - v.data[i,j-1,k]) * inv2dy +
                          (w.data[i,j,k+1] - w.data[i,j,k-1]) * inv2dz
    end
    return out
end

# =========================================================================
# Interpolation between staggered locations (linear averaging)
# =========================================================================
# Used to bring quantities to a common location for e.g. momentum advection.

# center → x-face
function _interp_to_face!(out::Field{FaceX}, f::Field{CellCenter}, g::RegularGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (f.data[i-1,j,k] + f.data[i,j,k])
    end
    return out
end
function _interp_to_face!(out::Field{FaceY}, f::Field{CellCenter}, g::RegularGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (f.data[i,j-1,k] + f.data[i,j,k])
    end
    return out
end
function _interp_to_face!(out::Field{FaceZ}, f::Field{CellCenter}, g::RegularGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (f.data[i,j,k-1] + f.data[i,j,k])
    end
    return out
end

# face → cell-center
function _interp_to_center!(out::Field{CellCenter}, u::Field{FaceX}, g::RegularGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (u.data[i,j,k] + u.data[i+1,j,k])
    end
    return out
end
function _interp_to_center!(out::Field{CellCenter}, v::Field{FaceY}, g::RegularGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (v.data[i,j,k] + v.data[i,j+1,k])
    end
    return out
end
function _interp_to_center!(out::Field{CellCenter}, w::Field{FaceZ}, g::RegularGrid)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        out.data[i,j,k] = 0.5 * (w.data[i,j,k] + w.data[i,j,k+1])
    end
    return out
end

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
# halo used for `NoFlux` / `ImpermeableWall`.

function _add_surface_flux!(tϕ::Field{CellCenter},
                            F_bot::Real, F_top::Real,
                            g::RegularGrid)
    inv_dz = 1 / g.dz
    ii, jj, kk = interior(g)
    k_bot = first(kk)
    k_top = last(kk)
    @inbounds for j in jj, i in ii
        tϕ.data[i, j, k_bot] += F_bot * inv_dz
        tϕ.data[i, j, k_top] -= F_top * inv_dz
    end
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
"""
function _add_damping!(tf::Field{L}, f::Field{L}, p::DampingProfile,
                       coords::AbstractArray, axis::Int, target,
                       g::AbstractMesh; levels = nothing) where {L}
    _check_damping_target(target, g)
    ii, jj, kk = interior(g)
    i0, j0, k0 = first(ii) - 1, first(jj) - 1, first(kk) - 1
    @inbounds for k in kk, j in jj, i in ii
        ξ = _damping_coord(coords, axis, i, j, k, i0, j0, k0)
        in_damping(p, ξ) || continue
        λ = damping_rate(p, ξ)
        δ = λ * (f.data[i, j, k] - _damping_target(target, i, j, k, k - k0))
        tf.data[i, j, k] -= δ
        levels === nothing || (levels[k - k0] -= δ)
    end
    return tf
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
# (length Nz), a Field or a halo-padded array (#254).

function _add_sponge!(tf::Field{L}, f::Field{L},
                      α_max::Real, z_start::Real, z_top::Real,
                      f_ref, g::RegularGrid) where {L}
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
# incompressible flow (∇·u = 0). Not strictly energy-conserving on the
# discrete level — a skew-symmetric / Morinishi form comes later.

function _momentum_advect!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                           u::Field{FaceX},  v::Field{FaceY},  w::Field{FaceZ},
                           g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)

    # --- u-momentum (at west face of cell i,j,k) ---
    @inbounds for k in kk, j in jj, i in ii
        uce = 0.5*(u.data[i,   j, k] + u.data[i+1, j, k])
        ucw = 0.5*(u.data[i-1, j, k] + u.data[i,   j, k])
        d_uu_dx = (uce*uce - ucw*ucw) * idx

        vun = 0.5*(v.data[i-1, j+1, k] + v.data[i, j+1, k])
        uvn = 0.5*(u.data[i,   j,   k] + u.data[i, j+1, k])
        vus = 0.5*(v.data[i-1, j,   k] + v.data[i, j,   k])
        uvs = 0.5*(u.data[i,   j-1, k] + u.data[i, j,   k])
        d_vu_dy = (vun*uvn - vus*uvs) * idy

        wut = 0.5*(w.data[i-1, j, k+1] + w.data[i, j, k+1])
        uwt = 0.5*(u.data[i,   j, k  ] + u.data[i, j, k+1])
        wub = 0.5*(w.data[i-1, j, k  ] + w.data[i, j, k  ])
        uwb = 0.5*(u.data[i,   j, k-1] + u.data[i, j, k  ])
        d_wu_dz = (wut*uwt - wub*uwb) * idz

        tu.data[i, j, k] -= d_uu_dx + d_vu_dy + d_wu_dz
    end

    # --- v-momentum (at south face of cell i,j,k) ---
    @inbounds for k in kk, j in jj, i in ii
        uve = 0.5*(u.data[i+1, j-1, k] + u.data[i+1, j, k])
        vue = 0.5*(v.data[i,   j,   k] + v.data[i+1, j, k])
        uvw = 0.5*(u.data[i,   j-1, k] + u.data[i,   j, k])
        vuw = 0.5*(v.data[i-1, j,   k] + v.data[i,   j, k])
        d_uv_dx = (uve*vue - uvw*vuw) * idx

        vcn = 0.5*(v.data[i, j,   k] + v.data[i, j+1, k])
        vcs = 0.5*(v.data[i, j-1, k] + v.data[i, j,   k])
        d_vv_dy = (vcn*vcn - vcs*vcs) * idy

        wvt = 0.5*(w.data[i, j-1, k+1] + w.data[i, j, k+1])
        vwt = 0.5*(v.data[i, j,   k  ] + v.data[i, j, k+1])
        wvb = 0.5*(w.data[i, j-1, k  ] + w.data[i, j, k  ])
        vwb = 0.5*(v.data[i, j,   k-1] + v.data[i, j, k  ])
        d_wv_dz = (wvt*vwt - wvb*vwb) * idz

        tv.data[i, j, k] -= d_uv_dx + d_vv_dy + d_wv_dz
    end

    # --- w-momentum (at bottom face of cell i,j,k) ---
    @inbounds for k in kk, j in jj, i in ii
        uwe = 0.5*(u.data[i+1, j, k-1] + u.data[i+1, j, k])
        wue = 0.5*(w.data[i,   j, k  ] + w.data[i+1, j, k])
        uww = 0.5*(u.data[i,   j, k-1] + u.data[i,   j, k])
        wuw = 0.5*(w.data[i-1, j, k  ] + w.data[i,   j, k])
        d_uw_dx = (uwe*wue - uww*wuw) * idx

        vwn = 0.5*(v.data[i, j+1, k-1] + v.data[i, j+1, k])
        wvn = 0.5*(w.data[i, j,   k  ] + w.data[i, j+1, k])
        vws = 0.5*(v.data[i, j,   k-1] + v.data[i, j,   k])
        wvs = 0.5*(w.data[i, j-1, k  ] + w.data[i, j,   k])
        d_vw_dy = (vwn*wvn - vws*wvs) * idy

        wct = 0.5*(w.data[i, j, k  ] + w.data[i, j, k+1])
        wcb = 0.5*(w.data[i, j, k-1] + w.data[i, j, k  ])
        d_ww_dz = (wct*wct - wcb*wcb) * idz

        tw.data[i, j, k] -= d_uw_dx + d_vw_dy + d_ww_dz
    end

    return (tu, tv, tw)
end

# =========================================================================
# Buoyancy — Boussinesq g'β(θ - θ₀)/θ₀ added to w-momentum at z-faces.
# =========================================================================
# θ is at cell centers; interpolate in z to w-face and add.
# For the standard atmospheric convention we take α_θ = 1/θ₀ so the
# buoyancy tendency reads   b = g (θ - θ₀) / θ₀.

function _add_buoyancy!(tw::Field{FaceZ}, θ::Field{CellCenter},
                        θ₀::Real, g_accel::Real, g::RegularGrid)
    coef = g_accel / θ₀
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        θface = 0.5 * (θ.data[i,j,k-1] + θ.data[i,j,k])
        tw.data[i,j,k] += coef * (θface - θ₀)
    end
    return tw
end

# --- moist buoyancy (Phase 11) ------------------------------------------
# Virtual potential temperature θ_v = θ (1 + 0.61 qv - qL), where qL is
# a bulk liquid-water mass fraction. Uses the standard 0.61 approximation
# for dry-air / water-vapor gas-constant ratio (Rv/Rd - 1 = 0.608).

function _add_buoyancy_moist!(tw::Field{FaceZ},
                              θ::Field{CellCenter}, qv::Field{CellCenter},
                              θ₀::Real, g_accel::Real, g::RegularGrid)
    coef = g_accel / θ₀
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        θface  = 0.5 * (θ.data[i,j,k-1]  + θ.data[i,j,k])
        qvface = 0.5 * (qv.data[i,j,k-1] + qv.data[i,j,k])
        θv     = θface * (1 + 0.61 * qvface)
        tw.data[i,j,k] += coef * (θv - θ₀)
    end
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
function _add_coriolis!(tu::Field{FaceX}, tv::Field{FaceY},
                        u::Field{FaceX},  v::Field{FaceY},
                        f::Real, g::AbstractMesh)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        v_at_u = 0.25 * (v.data[i-1,j,k]   + v.data[i,j,k]   +
                         v.data[i-1,j+1,k] + v.data[i,j+1,k])
        tu.data[i,j,k] += f * v_at_u

        u_at_v = 0.25 * (u.data[i,j-1,k]   + u.data[i+1,j-1,k] +
                         u.data[i,j,  k]   + u.data[i+1,j,  k])
        tv.data[i,j,k] -= f * u_at_v
    end
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

function _compute_smagorinsky_νt!(νt::Field{CellCenter},
                                  u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                  C_s::Real, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    Δ   = (g.dx * g.dy * g.dz) ^ (1/3)
    coef = (C_s * Δ) ^ 2
    ii, jj, kk = interior(g)
    ud, vd, wd = u.data, v.data, w.data
    @inbounds for k in kk, j in jj, i in ii
        # the one strain stencil (Operators._strain_at / _velocity_gradient_at)
        ms = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)[7]
        νt.data[i,j,k] = coef * ms
    end
    return νt
end

# --- eddy-viscosity flux divergence for u-momentum ----------------------
# ∂/∂x_j (ν_t ∂u/∂x_j) added into tu at u-faces. Diffusive flux in x
# lives at cell centers, in y at x-y edges, in z at x-z edges — with
# ν_t interpolated to the flux location.

function _add_eddy_viscosity_u!(tu::Field{FaceX}, u::Field{FaceX},
                                 νt::Field{CellCenter}, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        fxE = νt.data[i  ,j,k] * (u.data[i+1,j,k] - u.data[i  ,j,k]) * idx
        fxW = νt.data[i-1,j,k] * (u.data[i  ,j,k] - u.data[i-1,j,k]) * idx

        νt_eN = 0.25*(νt.data[i-1,j,k]   + νt.data[i,j,k]   +
                      νt.data[i-1,j+1,k] + νt.data[i,j+1,k])
        νt_eS = 0.25*(νt.data[i-1,j-1,k] + νt.data[i,j-1,k] +
                      νt.data[i-1,j,k]   + νt.data[i,j,k])
        fyN = νt_eN * (u.data[i,j+1,k] - u.data[i,j  ,k]) * idy
        fyS = νt_eS * (u.data[i,j,  k] - u.data[i,j-1,k]) * idy

        νt_eT = 0.25*(νt.data[i-1,j,k]   + νt.data[i,j,k]   +
                      νt.data[i-1,j,k+1] + νt.data[i,j,k+1])
        νt_eB = 0.25*(νt.data[i-1,j,k-1] + νt.data[i,j,k-1] +
                      νt.data[i-1,j,k]   + νt.data[i,j,k])
        fzT = νt_eT * (u.data[i,j,k+1] - u.data[i,j,k  ]) * idz
        fzB = νt_eB * (u.data[i,j,k  ] - u.data[i,j,k-1]) * idz

        tu.data[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*idz
    end
    return tu
end

function _add_eddy_viscosity_v!(tv::Field{FaceY}, v::Field{FaceY},
                                 νt::Field{CellCenter}, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        fyN = νt.data[i,j  ,k] * (v.data[i,j+1,k] - v.data[i,j  ,k]) * idy
        fyS = νt.data[i,j-1,k] * (v.data[i,j  ,k] - v.data[i,j-1,k]) * idy

        νt_eE = 0.25*(νt.data[i,j-1,k]   + νt.data[i+1,j-1,k] +
                      νt.data[i,j,k]     + νt.data[i+1,j,k])
        νt_eW = 0.25*(νt.data[i-1,j-1,k] + νt.data[i,j-1,k]   +
                      νt.data[i-1,j,k]   + νt.data[i,j,k])
        fxE = νt_eE * (v.data[i+1,j,k] - v.data[i,  j,k]) * idx
        fxW = νt_eW * (v.data[i,  j,k] - v.data[i-1,j,k]) * idx

        νt_eT = 0.25*(νt.data[i,j-1,k]   + νt.data[i,j,k]   +
                      νt.data[i,j-1,k+1] + νt.data[i,j,k+1])
        νt_eB = 0.25*(νt.data[i,j-1,k-1] + νt.data[i,j,k-1] +
                      νt.data[i,j-1,k]   + νt.data[i,j,k])
        fzT = νt_eT * (v.data[i,j,k+1] - v.data[i,j,k  ]) * idz
        fzB = νt_eB * (v.data[i,j,k  ] - v.data[i,j,k-1]) * idz

        tv.data[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*idz
    end
    return tv
end

function _add_eddy_viscosity_w!(tw::Field{FaceZ}, w::Field{FaceZ},
                                 νt::Field{CellCenter}, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        fzT = νt.data[i,j,k  ] * (w.data[i,j,k+1] - w.data[i,j,k  ]) * idz
        fzB = νt.data[i,j,k-1] * (w.data[i,j,k  ] - w.data[i,j,k-1]) * idz

        νt_eE = 0.25*(νt.data[i,j,k-1]   + νt.data[i+1,j,k-1] +
                      νt.data[i,j,k]     + νt.data[i+1,j,k])
        νt_eW = 0.25*(νt.data[i-1,j,k-1] + νt.data[i,j,k-1]   +
                      νt.data[i-1,j,k]   + νt.data[i,j,k])
        fxE = νt_eE * (w.data[i+1,j,k] - w.data[i,  j,k]) * idx
        fxW = νt_eW * (w.data[i,  j,k] - w.data[i-1,j,k]) * idx

        νt_eN = 0.25*(νt.data[i,j,k-1]   + νt.data[i,j+1,k-1] +
                      νt.data[i,j,k]     + νt.data[i,j+1,k])
        νt_eS = 0.25*(νt.data[i,j-1,k-1] + νt.data[i,j,k-1]   +
                      νt.data[i,j-1,k]   + νt.data[i,j,k])
        fyN = νt_eN * (w.data[i,j+1,k] - w.data[i,j  ,k]) * idy
        fyS = νt_eS * (w.data[i,j,  k] - w.data[i,j-1,k]) * idy

        tw.data[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*idz
    end
    return tw
end

# --- stress-form SGS momentum divergence (Era VI, P55-b, #175) -----------
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
_add_sgs_stress_divergence!(tu, tv, tw, u, v, w, nh, nv, a, b, g::AbstractMesh) =
    throw(ArgumentError("sgs_stress = :symmetric (the SGS stress form) is implemented " *
                        "on a RegularGrid only, not on a $(nameof(typeof(g))); use " *
                        "sgs_stress = :laplacian"))

# −∂τᵢⱼ/∂xⱼ into (tu, tv, tw), with the stresses formed on the fly from
# (u, v, w) and the cell-centred (νh, νv) arrays (halos filled), part (a, b).
function _add_sgs_stress_divergence!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                     u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                     nh::AbstractArray{<:Real,3}, nv::AbstractArray{<:Real,3},
                                     a::Real, b::Real, g::RegularGrid)
    T = eltype(tu.data)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    A = T(a); B = T(b)
    ii, jj, kk = interior(g)
    ud, vd, wd = u.data, v.data, w.data
    tud, tvd, twd = tu.data, tv.data, tw.data
    @inbounds for k in kk, j in jj, i in ii
        tud[i,j,k] += -((_τ11(ud, nh, i, j, k, idx, A, B) - _τ11(ud, nh, i-1, j, k, idx, A, B)) * idx +
                        (_τ12_u(ud, vd, nh, i, j+1, k, idx, idy, A, B) -
                         _τ12_u(ud, vd, nh, i, j, k, idx, idy, A, B)) * idy +
                        (_τ13_u(ud, wd, nv, i, j, k+1, idx, idz, A, B) -
                         _τ13_u(ud, wd, nv, i, j, k, idx, idz, A, B)) * idz)
        tvd[i,j,k] += -((_τ12_v(ud, vd, nh, i+1, j, k, idx, idy, A, B) -
                         _τ12_v(ud, vd, nh, i, j, k, idx, idy, A, B)) * idx +
                        (_τ22(vd, nh, i, j, k, idy, A, B) - _τ22(vd, nh, i, j-1, k, idy, A, B)) * idy +
                        (_τ23_v(vd, wd, nv, i, j, k+1, idy, idz, A, B) -
                         _τ23_v(vd, wd, nv, i, j, k, idy, idz, A, B)) * idz)
        twd[i,j,k] += -((_τ13_w(ud, wd, nv, i+1, j, k, idx, idz, A, B) -
                         _τ13_w(ud, wd, nv, i, j, k, idx, idz, A, B)) * idx +
                        (_τ23_w(vd, wd, nv, i, j+1, k, idy, idz, A, B) -
                         _τ23_w(vd, wd, nv, i, j, k, idy, idz, A, B)) * idy +
                        (_τ33(wd, nv, i, j, k, idz, A, B) - _τ33(wd, nv, i, j, k-1, idz, A, B)) * idz)
    end
    return nothing
end

# −∂τᵢⱼ/∂xⱼ from a stored stress (six cell-centre-shaped arrays, placement
# above). Reads τ11 on i ∈ first-1:last, τ12 on i, j ∈ first:last+1, etc.
function _add_stress_divergence!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                 τ11, τ22, τ33, τ12, τ13, τ23, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    tud, tvd, twd = tu.data, tv.data, tw.data
    @inbounds for k in kk, j in jj, i in ii
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

# τ = −2νᵢⱼSᵢⱼ into the six arrays, on exactly the index ranges
# `_add_stress_divergence!` reads (the interior plus the shell below for
# the diagonals, plus the shell above for the edges).
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
    @inbounds begin
        for k in kk, j in jj, i in xm; τ11[i,j,k] = _τ11(ud, nh, i, j, k, idx, o, o); end
        for k in kk, j in ym, i in ii; τ22[i,j,k] = _τ22(vd, nh, i, j, k, idy, o, o); end
        for k in zm, j in jj, i in ii; τ33[i,j,k] = _τ33(wd, nv, i, j, k, idz, o, o); end
        for k in kk, j in yp, i in xp; τ12[i,j,k] = _τ12_u(ud, vd, nh, i, j, k, idx, idy, o, o); end
        for k in zp, j in jj, i in xp; τ13[i,j,k] = _τ13_u(ud, wd, nv, i, j, k, idx, idz, o, o); end
        for k in zp, j in yp, i in ii; τ23[i,j,k] = _τ23_v(vd, wd, nv, i, j, k, idy, idz, o, o); end
    end
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
    ii, jj, kk = interior(g)
    ud, vd, wd = u.data, v.data, w.data
    s = zero(T)
    @inbounds for k in kk, j in jj, i in ii
        S11 = (ud[i+1,j,k] - ud[i,j,k]) * idx
        S22 = (vd[i,j+1,k] - vd[i,j,k]) * idy
        S33 = (wd[i,j,k+1] - wd[i,j,k]) * idz
        S12 = 0.5 * ((ud[i,j,k] - ud[i,j-1,k]) * idy + (vd[i,j,k] - vd[i-1,j,k]) * idx)
        S13 = 0.5 * ((ud[i,j,k] - ud[i,j,k-1]) * idz + (wd[i,j,k] - wd[i-1,j,k]) * idx)
        S23 = 0.5 * ((vd[i,j,k] - vd[i,j,k-1]) * idz + (wd[i,j,k] - wd[i,j-1,k]) * idy)
        s += 2 * (nh[i,j,k] * (S11^2 + S22^2) + nv[i,j,k] * S33^2) +
             4 * (_νe12(nh, i, j, k) * S12^2 + _νe13(nv, i, j, k) * S13^2 +
                  _νe23(nv, i, j, k) * S23^2)
    end
    return s * (g.dx * g.dy * g.dz)
end

# --- eddy diffusion for a cell-centered scalar --------------------------
# s · [∂/∂x(Kh ∂ϕ/∂x) + ∂/∂y(Kh ∂ϕ/∂y) + ∂/∂z(Kv ∂ϕ/∂z)], each coefficient
# applied conservatively at the faces (interpolated linearly from cell
# centres). The single kernel behind every public form:
# `add_eddy_diffusion_scalar!(tϕ, ϕ, κt)` (Kh = Kv = κt, s = 1, exact),
# `add_eddy_diffusion_scalar!(tϕ, ϕ, νt, Pr_t)` (Kh = Kv = νt, s = 1/Pr_t)
# and the diagonal `add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv)` (P55-c,
# #181). With Kh === Kv the arithmetic is the pre-P55-c kernel's, bit for bit.

_add_eddy_diffusion_scalar!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                            K::Field{CellCenter}, s::Real, g::RegularGrid) =
    _add_eddy_diffusion_scalar!(tϕ, ϕ, K, K, s, g)

function _add_eddy_diffusion_scalar!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                      Kh::Field{CellCenter}, Kv::Field{CellCenter},
                                      s::Real, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    H = Kh.data; V = Kv.data
    @inbounds for k in kk, j in jj, i in ii
        νE = 0.5 * (H[i,j,k]   + H[i+1,j,k])
        νW = 0.5 * (H[i-1,j,k] + H[i,j,k])
        νN = 0.5 * (H[i,j,k]   + H[i,j+1,k])
        νS = 0.5 * (H[i,j-1,k] + H[i,j,k])
        νT = 0.5 * (V[i,j,k]   + V[i,j,k+1])
        νB = 0.5 * (V[i,j,k-1] + V[i,j,k])

        fxE = νE * (ϕ.data[i+1,j,k] - ϕ.data[i,  j,k]) * idx
        fxW = νW * (ϕ.data[i,  j,k] - ϕ.data[i-1,j,k]) * idx
        fyN = νN * (ϕ.data[i,j+1,k] - ϕ.data[i,j,  k]) * idy
        fyS = νS * (ϕ.data[i,j,  k] - ϕ.data[i,j-1,k]) * idy
        fzT = νT * (ϕ.data[i,j,k+1] - ϕ.data[i,j,k  ]) * idz
        fzB = νB * (ϕ.data[i,j,k  ] - ϕ.data[i,j,k-1]) * idz

        tϕ.data[i,j,k] += s * ((fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*idz)
    end
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
    T = eltype(d)
    # the face gradients of cell (i, j, k), zero on a wall (bounded direction)
    @inline gW(i, j, k) = @inbounds((bx && i == i1) ? zero(T) : (d[i, j, k] - d[i-1, j, k]) * idx)
    @inline gE(i, j, k) = @inbounds((bx && i == i2) ? zero(T) : (d[i+1, j, k] - d[i, j, k]) * idx)
    @inline gS(i, j, k) = @inbounds((by && j == j1) ? zero(T) : (d[i, j, k] - d[i, j-1, k]) * idy)
    @inline gN(i, j, k) = @inbounds((by && j == j2) ? zero(T) : (d[i, j+1, k] - d[i, j, k]) * idy)
    @inline gB(i, j, k) = @inbounds((bz && k == k1) ? zero(T) : (d[i, j, k] - d[i, j, k-1]) * idz)
    @inline gT(i, j, k) = @inbounds((bz && k == k2) ? zero(T) : (d[i, j, k+1] - d[i, j, k]) * idz)
    @inbounds begin
        # x faces i+½, i ∈ Hx:i2
        for k in k1:k2, j in j1:j2, i in Hx:i2
            f = zero(T)
            dodiag && (f += 0.5 * (K11[i, j, k] + K11[i+1, j, k]) * (d[i+1, j, k] - d[i, j, k]) * idx)
            if dooff && !(bx && (i == Hx || i == i2))
                f += 0.25 * (K12[i,   j, k] * (gS(i,   j, k) + gN(i,   j, k)) +
                             K12[i+1, j, k] * (gS(i+1, j, k) + gN(i+1, j, k)) +
                             K13[i,   j, k] * (gB(i,   j, k) + gT(i,   j, k)) +
                             K13[i+1, j, k] * (gB(i+1, j, k) + gT(i+1, j, k)))
            end
            Fx[i, j, k] = f
        end
        # y faces j+½
        for k in k1:k2, j in Hy:j2, i in i1:i2
            f = zero(T)
            dodiag && (f += 0.5 * (K22[i, j, k] + K22[i, j+1, k]) * (d[i, j+1, k] - d[i, j, k]) * idy)
            if dooff && !(by && (j == Hy || j == j2))
                f += 0.25 * (K12[i, j,   k] * (gW(i, j,   k) + gE(i, j,   k)) +
                             K12[i, j+1, k] * (gW(i, j+1, k) + gE(i, j+1, k)) +
                             K23[i, j,   k] * (gB(i, j,   k) + gT(i, j,   k)) +
                             K23[i, j+1, k] * (gB(i, j+1, k) + gT(i, j+1, k)))
            end
            Fy[i, j, k] = f
        end
        # z faces k+½
        for k in Hz:k2, j in j1:j2, i in i1:i2
            f = zero(T)
            dodiag && (f += 0.5 * (K33[i, j, k] + K33[i, j, k+1]) * (d[i, j, k+1] - d[i, j, k]) * idz)
            if dooff && !(bz && (k == Hz || k == k2))
                f += 0.25 * (K13[i, j, k  ] * (gW(i, j, k  ) + gE(i, j, k  )) +
                             K13[i, j, k+1] * (gW(i, j, k+1) + gE(i, j, k+1)) +
                             K23[i, j, k  ] * (gS(i, j, k  ) + gN(i, j, k  )) +
                             K23[i, j, k+1] * (gS(i, j, k+1) + gN(i, j, k+1)))
            end
            Fz[i, j, k] = f
        end
        t = tϕ.data
        for k in k1:k2, j in j1:j2, i in i1:i2
            t[i, j, k] += (Fx[i, j, k] - Fx[i-1, j, k]) * idx +
                          (Fy[i, j, k] - Fy[i, j-1, k]) * idy +
                          (Fz[i, j, k] - Fz[i, j, k-1]) * idz
        end
    end
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

function _add_eddy_diffusion_scalar_masked!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                             Kh::Field{CellCenter}, Kv::Field{CellCenter},
                                             g::RegularGrid, mask, wc, wf)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    Kd = Kh.data; Vd = Kv.data; ϕd = ϕ.data
    T = eltype(ϕd)
    @inbounds for k in kk, j in jj, i in ii
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
        tϕ.data[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + vert
    end
    return tϕ
end

# --- the rotated isentropic tensor (P55-c, #181) --------------------------
# K = κh I − f (κh − κv) n nᵀ, n = ∇θ/|∇θ| (centred differences), with
# f = f_slope · f_grad: f_slope = min(1, (S_max/S)²), S = |∇ₕθ|/∂zθ
# (Gerdes et al. 1991), 0 where ∂zθ ≤ 0 (N² ≤ 0); f_grad = min(1, |∇θ|/g_min).
# κh, κv: cell-centred arrays or numbers (`_kval`).
@inline _kval(a::AbstractArray, i, j, k) = @inbounds a[i, j, k]
@inline _kval(a::Real, i, j, k) = a

function _isentropic_tensor!(K11, K22, K33, K12, K13, K23, θ::Field{CellCenter},
                             κh, κv, slope_max::Real, grad_min::Real, g::RegularGrid)
    d = θ.data
    i2dx = 1 / (2 * g.dx); i2dy = 1 / (2 * g.dy); i2dz = 1 / (2 * g.dz)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
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

@inline _open(::Type{T}, ::Nothing, i, j, k) where {T} = one(T)
@inline _open(::Type{T}, mask, i, j, k) where {T} = T(!mask[i,j,k])
@inline _vertical_divergence(fzT, fzB, idz, ::Nothing, ::Nothing, k) = (fzT - fzB) * idz
@inline _vertical_divergence(fzT, fzB, idz, wc, wf, k) = (wf[k+1] * fzT - wf[k] * fzB) * idz / wc[k]

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

function _advect_weno5!(tend::Field{CellCenter},
                        u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                        ϕ::Field{CellCenter}, g::RegularGrid)
    (g.Hx >= 3 && g.Hy >= 3 && g.Hz >= 3) ||
        error("WENO-5 requires halo ≥ 3 in every direction (got $(g.Hx),$(g.Hy),$(g.Hz))")
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    ϕd = ϕ.data; ud = u.data; vd = v.data; wd = w.data
    @inbounds for k in kk, j in jj, i in ii
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

        tend.data[i, j, k] -= Fx + Fy + Fz
    end
    return tend
end

# =========================================================================
# Scalar advection (Phase 0 — collocated).
# =========================================================================

function _advect_centered!(tend::Field{CellCenter},
                          u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                          ϕ::Field{CellCenter}, g::RegularGrid)
    inv2dx = 1 / (2g.dx); inv2dy = 1 / (2g.dy); inv2dz = 1 / (2g.dz)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        dϕdx = (ϕ.data[i+1,j,k] - ϕ.data[i-1,j,k]) * inv2dx
        dϕdy = (ϕ.data[i,j+1,k] - ϕ.data[i,j-1,k]) * inv2dy
        dϕdz = (ϕ.data[i,j,k+1] - ϕ.data[i,j,k-1]) * inv2dz
        tend.data[i,j,k] -= u.data[i,j,k]*dϕdx + v.data[i,j,k]*dϕdy + w.data[i,j,k]*dϕdz
    end
    return tend
end

# --- Staggered scalar advection (flux form) -----------------------------
# Adds -∂(u_j ϕ)/∂x_j to `tend` when the transporting velocity lives on
# faces and ϕ lives at cell centers. Used for the potential-temperature
# equation in the Boussinesq core.

function _advect_centered!(tend::Field{CellCenter},
                          u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                          ϕ::Field{CellCenter}, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        uϕe = u.data[i+1,j,k] * 0.5 * (ϕ.data[i,  j,k] + ϕ.data[i+1,j,k])
        uϕw = u.data[i,  j,k] * 0.5 * (ϕ.data[i-1,j,k] + ϕ.data[i,  j,k])
        vϕn = v.data[i,j+1,k] * 0.5 * (ϕ.data[i,j,  k] + ϕ.data[i,j+1,k])
        vϕs = v.data[i,j,  k] * 0.5 * (ϕ.data[i,j-1,k] + ϕ.data[i,j,  k])
        wϕt = w.data[i,j,k+1] * 0.5 * (ϕ.data[i,j,k  ] + ϕ.data[i,j,k+1])
        wϕb = w.data[i,j,k  ] * 0.5 * (ϕ.data[i,j,k-1] + ϕ.data[i,j,k  ])
        tend.data[i,j,k] -= (uϕe - uϕw)*idx + (vϕn - vϕs)*idy + (wϕt - wϕb)*idz
    end
    return tend
end

function _advect_upwind!(tend::Field{CellCenter},
                        u::Field{CellCenter}, v::Field{CellCenter}, w::Field{CellCenter},
                        ϕ::Field{CellCenter}, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        ui = u.data[i,j,k]; vi = v.data[i,j,k]; wi = w.data[i,j,k]
        dϕdx = ui >= 0 ? (ϕ.data[i,j,k] - ϕ.data[i-1,j,k]) * idx :
                         (ϕ.data[i+1,j,k] - ϕ.data[i,j,k]) * idx
        dϕdy = vi >= 0 ? (ϕ.data[i,j,k] - ϕ.data[i,j-1,k]) * idy :
                         (ϕ.data[i,j+1,k] - ϕ.data[i,j,k]) * idy
        dϕdz = wi >= 0 ? (ϕ.data[i,j,k] - ϕ.data[i,j,k-1]) * idz :
                         (ϕ.data[i,j,k+1] - ϕ.data[i,j,k]) * idz
        tend.data[i,j,k] -= ui*dϕdx + vi*dϕdy + wi*dϕdz
    end
    return tend
end

end # module
