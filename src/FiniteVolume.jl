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
function _add_eddy_diffusion_scalar_masked!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                             K::Field{CellCenter}, g::RegularGrid,
                                             mask, wc, wf)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    ii, jj, kk = interior(g)
    Kd = K.data; ϕd = ϕ.data
    hasmask = mask !== nothing
    weighted = wc !== nothing
    @inbounds for k in kk, j in jj, i in ii
        hasmask && mask[i,j,k] && continue
        oE = !hasmask || !mask[i+1,j,k]
        oW = !hasmask || !mask[i-1,j,k]
        oN = !hasmask || !mask[i,j+1,k]
        oS = !hasmask || !mask[i,j-1,k]
        oT = !hasmask || !mask[i,j,k+1]
        oB = !hasmask || !mask[i,j,k-1]

        fxE = oE ? 0.5 * (Kd[i,j,k]   + Kd[i+1,j,k]) * (ϕd[i+1,j,k] - ϕd[i,  j,k]) * idx : zero(eltype(ϕd))
        fxW = oW ? 0.5 * (Kd[i-1,j,k] + Kd[i,j,k])   * (ϕd[i,  j,k] - ϕd[i-1,j,k]) * idx : zero(eltype(ϕd))
        fyN = oN ? 0.5 * (Kd[i,j,k]   + Kd[i,j+1,k]) * (ϕd[i,j+1,k] - ϕd[i,j,  k]) * idy : zero(eltype(ϕd))
        fyS = oS ? 0.5 * (Kd[i,j-1,k] + Kd[i,j,k])   * (ϕd[i,j,  k] - ϕd[i,j-1,k]) * idy : zero(eltype(ϕd))
        fzT = oT ? 0.5 * (Kd[i,j,k]   + Kd[i,j,k+1]) * (ϕd[i,j,k+1] - ϕd[i,j,k  ]) * idz : zero(eltype(ϕd))
        fzB = oB ? 0.5 * (Kd[i,j,k-1] + Kd[i,j,k])   * (ϕd[i,j,k  ] - ϕd[i,j,k-1]) * idz : zero(eltype(ϕd))

        vert = weighted ? (wf[k+1] * fzT - wf[k] * fzB) * idz / wc[k] : (fzT - fzB) * idz
        tϕ.data[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + vert
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
