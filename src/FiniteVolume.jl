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
# Rayleigh sponge damping — Phase 7.
# =========================================================================
# Adds  -α(z) · (φ - φ_ref)  to the tendency inside a top-of-domain
# region [z_start, z_top]. `α(z)` grows smoothly with height:
#     α(z) = α_max · sin²( π/2 · (z - z_start) / (z_top - z_start) ).
# Below `z_start` the damping is zero. Used to absorb gravity waves and
# turbulent structures that would otherwise reflect off the rigid lid.

function _add_sponge!(tf::Field{L}, f::Field{L},
                      α_max::Real, z_start::Real, z_top::Real,
                      f_ref::Real, g::RegularGrid) where {L}
    L() isa Location || nothing
    zs = _znodes_for(g, L())
    ii, jj, kk = interior(g)
    inv_width = 1 / (z_top - z_start)
    half_π = 0.5 * π
    @inbounds for (kk_, kv) in enumerate(kk), j in jj, i in ii
        z = zs[kk_]
        if z >= z_start
            s = sin(half_π * (z - z_start) * inv_width)
            α = α_max * s * s
            tf.data[i, j, kv] -= α * (f.data[i, j, kv] - f_ref)
        end
    end
    return tf
end

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

function _add_coriolis!(tu::Field{FaceX}, tv::Field{FaceY},
                        u::Field{FaceX},  v::Field{FaceY},
                        f::Real, g::RegularGrid)
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
@inline _νe12(n, i, j, k) = 0.25 * (n[i-1,j-1,k] + n[i,j-1,k] + n[i-1,j,k] + n[i,j,k])
@inline _νe13(n, i, j, k) = 0.25 * (n[i-1,j,k-1] + n[i,j,k-1] + n[i-1,j,k] + n[i,j,k])
@inline _νe23(n, i, j, k) = 0.25 * (n[i,j-1,k-1] + n[i,j,k-1] + n[i,j-1,k] + n[i,j,k])

# diagonal stresses at the centre of cell (i, j, k): −(a+b) ν ∂uᵢ/∂xᵢ
@inline _τ11(u, nh, i, j, k, idx, a, b) = -(nh[i,j,k] * ((a + b) * ((u[i+1,j,k] - u[i,j,k]) * idx)))
@inline _τ22(v, nh, i, j, k, idy, a, b) = -(nh[i,j,k] * ((a + b) * ((v[i,j+1,k] - v[i,j,k]) * idy)))
@inline _τ33(w, nv, i, j, k, idz, a, b) = -(nv[i,j,k] * ((a + b) * ((w[i,j,k+1] - w[i,j,k]) * idz)))

# off-diagonal stresses of row `own` on the edge (i, j, k):
# −ν_e (a ∂u_own/∂x_other + b ∂u_other/∂x_own)
@inline _τ12_u(u, v, nh, i, j, k, idx, idy, a, b) =
    -(_νe12(nh, i, j, k) * (a * ((u[i,j,k] - u[i,j-1,k]) * idy) + b * ((v[i,j,k] - v[i-1,j,k]) * idx)))
@inline _τ12_v(u, v, nh, i, j, k, idx, idy, a, b) =
    -(_νe12(nh, i, j, k) * (a * ((v[i,j,k] - v[i-1,j,k]) * idx) + b * ((u[i,j,k] - u[i,j-1,k]) * idy)))
@inline _τ13_u(u, w, nv, i, j, k, idx, idz, a, b) =
    -(_νe13(nv, i, j, k) * (a * ((u[i,j,k] - u[i,j,k-1]) * idz) + b * ((w[i,j,k] - w[i-1,j,k]) * idx)))
@inline _τ13_w(u, w, nv, i, j, k, idx, idz, a, b) =
    -(_νe13(nv, i, j, k) * (a * ((w[i,j,k] - w[i-1,j,k]) * idx) + b * ((u[i,j,k] - u[i,j,k-1]) * idz)))
@inline _τ23_v(v, w, nv, i, j, k, idy, idz, a, b) =
    -(_νe23(nv, i, j, k) * (a * ((v[i,j,k] - v[i,j,k-1]) * idz) + b * ((w[i,j,k] - w[i,j-1,k]) * idy)))
@inline _τ23_w(v, w, nv, i, j, k, idy, idz, a, b) =
    -(_νe23(nv, i, j, k) * (a * ((w[i,j,k] - w[i,j-1,k]) * idy) + b * ((v[i,j,k] - v[i,j,k-1]) * idz)))

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
# s · ∂/∂x_j (K ∂ϕ/∂x_j), K applied conservatively at faces (interpolated
# linearly from cell centres). The single kernel behind both public forms:
# `add_eddy_diffusion_scalar!(tϕ, ϕ, κt)` (K = κt, s = 1, exact) and
# `add_eddy_diffusion_scalar!(tϕ, ϕ, νt, Pr_t)` (K = νt, s = 1/Pr_t).

function _add_eddy_diffusion_scalar!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                      K::Field{CellCenter}, s::Real,
                                      g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        νE = 0.5 * (K.data[i,j,k]   + K.data[i+1,j,k])
        νW = 0.5 * (K.data[i-1,j,k] + K.data[i,j,k])
        νN = 0.5 * (K.data[i,j,k]   + K.data[i,j+1,k])
        νS = 0.5 * (K.data[i,j-1,k] + K.data[i,j,k])
        νT = 0.5 * (K.data[i,j,k]   + K.data[i,j,k+1])
        νB = 0.5 * (K.data[i,j,k-1] + K.data[i,j,k])

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
