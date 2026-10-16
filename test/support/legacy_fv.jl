# KA-d (#299): the finite-volume kernels as they were before the port
# (KA-b branch at a9d1740 + master c617bc1): the serial loops of
# src/FiniteVolume.jl, the StretchedGrid / TerrainFollowingGrid stencils of
# src/StretchedGrids.jl and the StretchedGrid copies of
# src/stretched/boussinesq.jl, transcribed verbatim. The reference of
# test/ka_fv.jl and of the KA-d timings (old loops vs
# launches). Nothing here is used by the package.
# ---------------------------------------------------------------------------
module LegacyFV
using Aeolus
using Aeolus.Grids
using Aeolus.Grids: RegularGrid, AbstractMesh, interior, CellCenter, FaceX, FaceY, FaceZ, Location
using Aeolus.Fields
using Aeolus.Fields: Field, mesh
using Aeolus.Operators: _strain_at
using Aeolus.Operators.FiniteVolume: DampingProfile, in_damping, damping_rate, sponge_profile,
                                     _damping_coord, _damping_target,
                                     _check_damping_target, _νe12, _νe13, _νe23,
                                     _τ11, _τ22, _τ33, _τ12_u, _τ12_v, _τ13_u, _τ13_w, _τ23_v, _τ23_w,
                                     _weno5_left, _weno5_right, _kval
using Aeolus.StretchedGrids: StretchedGrid, TerrainFollowingGrid, _xcoords, _ycoords, _zcoords,
                             physical_heights

# ===================== src/FiniteVolume.jl ==================================
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
# --- eddy diffusion for a cell-centered scalar --------------------------
# s · [∂/∂x(Kh ∂ϕ/∂x) + ∂/∂y(Kh ∂ϕ/∂y) + ∂/∂z(Kv ∂ϕ/∂z)], each coefficient
# applied conservatively at the faces (interpolated linearly from cell
# centres). The single kernel behind every public form:
# `add_eddy_diffusion_scalar!(tϕ, ϕ, κt)` (Kh = Kv = κt, s = 1, exact),
# `add_eddy_diffusion_scalar!(tϕ, ϕ, νt, Pr_t)` (Kh = Kv = νt, s = 1/Pr_t)
# and the diagonal `add_eddy_diffusion_scalar!(tϕ, ϕ, κh, κv)` (P55-c,
# #181). With Kh === Kv the arithmetic is the pre-P55-c kernel's.

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
# what it was.
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

# ===================== src/StretchedGrids.jl (StretchedGrid) ================
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
`α(z) = α_max sin²(½π (z-z_start)/(z_top-z_start))`. The one damping
kernel `_add_damping!` of the `RegularGrid` sponge, with the profile
sampled at the *actual* stretched heights, so the damping layer covers
the physical region the user asked for rather than a fixed number of
levels. `f_ref` is a number, a level vector (length `Nz`), a `Field` or a
halo-padded array, as on the `RegularGrid` (#254).
"""
function _add_sponge!(tf::Field{L}, f::Field{L},
                      α_max::Real, z_start::Real, z_top::Real,
                      f_ref, g::StretchedGrid) where {L<:Location}
    return _add_damping!(tf, f, sponge_profile(α_max, z_start, z_top),
                         damping_coordinates(g, L()), 3, f_ref, g)
end

# The heights of the interior levels the damping kernel samples a vertical
# DampingProfile at, on a StretchedGrid (the generic is in FiniteVolume).
damping_coordinates(g::StretchedGrid, loc::Location) =
    _zloc_coords(g, loc)[g.Hz+1:g.Hz+g.Nz]

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

# ===================== src/StretchedGrids.jl (_set_physical!, TerrainFollowingGrid) ==
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
the same as the [`StretchedGrid`](@ref) one.
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
function _laplacian!(out::Field{L}, f::Field{L}, g::TerrainFollowingGrid{T}) where {L<:Location, T}
    dx, dy = g.dx, g.dy
    dzc, dzf = _σ_metrics(g.base, L())
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

# ===================== src/stretched/boussinesq.jl =========================
# --- Laplacian of a field on z-faces ---------------------------------------

"""
    _laplacian!(out::Field{FaceZ}, f::Field{FaceZ}, g::StretchedGrid)

Conservative Laplacian of a field living on z-faces (the `w` velocity).
Its control volume is `[z_c[k-1], z_c[k]]`, of height `Δz_f[k]`, and its
two vertical neighbours sit `Δz_c[k-1]` below and `Δz_c[k]` above:

```math
\\left.\\nabla^2_z w\\right|_k =
 \\frac{1}{\\Delta z_f[k]}
 \\left[\\frac{w_{k+1}-w_k}{\\Delta z_c[k]} - \\frac{w_k-w_{k-1}}{\\Delta z_c[k-1]}\\right].
```

The generic method (all other locations) uses the cell-centre metrics,
which are right for `u`, `v`, `θ` (centred in z) but put a first-order
error on `w` wherever the grid stretches (caught by the #151
manufactured-solution test).
"""
function _laplacian!(out::Field{FaceZ}, f::Field{FaceZ}, g::StretchedGrid)
    idx2 = 1 / g.dx^2
    idy2 = 1 / g.dy^2
    dzc, dzf = g.dzc, g.dzf
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        iv   = 1 / dzf[k]
        idzm = 1 / dzc[k-1]
        idzp = 1 / dzc[k]
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

# --- momentum advection -----------------------------------------------------

"""
    _momentum_advect!(tu, tv, tw, u, v, w, g::StretchedGrid)

Second-order flux-form momentum advection `-∂(uⱼuᵢ)/∂xⱼ` on a vertically
stretched Arakawa-C grid. The u and v control volumes have height
`Δz_c[k]`, the w control volume `Δz_f[k]`; the transporting velocity on
each face is the volume flux through it (thickness-weighted for the
horizontal faces of a w volume) and the transported velocity the
arithmetic face mean. This keeps the operator energy-conserving for a
discretely solenoidal field (see the file header); on a uniform grid it is
the RegularGrid kernel.
"""
function _momentum_advect!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                           u::Field{FaceX},  v::Field{FaceY},  w::Field{FaceZ},
                           g::StretchedGrid)
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc, dzf = g.dzc, g.dzf
    ii, jj, kk = interior(g)
    ud, vd, wd = u.data, v.data, w.data

    # --- u-momentum (west face of cell i, height dzc[k]) ---
    @inbounds for k in kk
        idz = 1 / dzc[k]
        for j in jj, i in ii
            uce = 0.5*(ud[i,   j, k] + ud[i+1, j, k])
            ucw = 0.5*(ud[i-1, j, k] + ud[i,   j, k])
            d_uu_dx = (uce*uce - ucw*ucw) * idx

            vun = 0.5*(vd[i-1, j+1, k] + vd[i, j+1, k])
            uvn = 0.5*(ud[i,   j,   k] + ud[i, j+1, k])
            vus = 0.5*(vd[i-1, j,   k] + vd[i, j,   k])
            uvs = 0.5*(ud[i,   j-1, k] + ud[i, j,   k])
            d_vu_dy = (vun*uvn - vus*uvs) * idy

            wut = 0.5*(wd[i-1, j, k+1] + wd[i, j, k+1])
            uwt = 0.5*(ud[i,   j, k  ] + ud[i, j, k+1])
            wub = 0.5*(wd[i-1, j, k  ] + wd[i, j, k  ])
            uwb = 0.5*(ud[i,   j, k-1] + ud[i, j, k  ])
            d_wu_dz = (wut*uwt - wub*uwb) * idz

            tu.data[i, j, k] -= d_uu_dx + d_vu_dy + d_wu_dz
        end
    end

    # --- v-momentum (south face of cell j, height dzc[k]) ---
    @inbounds for k in kk
        idz = 1 / dzc[k]
        for j in jj, i in ii
            uve = 0.5*(ud[i+1, j-1, k] + ud[i+1, j, k])
            vue = 0.5*(vd[i,   j,   k] + vd[i+1, j, k])
            uvw = 0.5*(ud[i,   j-1, k] + ud[i,   j, k])
            vuw = 0.5*(vd[i-1, j,   k] + vd[i,   j, k])
            d_uv_dx = (uve*vue - uvw*vuw) * idx

            vcn = 0.5*(vd[i, j,   k] + vd[i, j+1, k])
            vcs = 0.5*(vd[i, j-1, k] + vd[i, j,   k])
            d_vv_dy = (vcn*vcn - vcs*vcs) * idy

            wvt = 0.5*(wd[i, j-1, k+1] + wd[i, j, k+1])
            vwt = 0.5*(vd[i, j,   k  ] + vd[i, j, k+1])
            wvb = 0.5*(wd[i, j-1, k  ] + wd[i, j, k  ])
            vwb = 0.5*(vd[i, j,   k-1] + vd[i, j, k  ])
            d_wv_dz = (wvt*vwt - wvb*vwb) * idz

            tv.data[i, j, k] -= d_uv_dx + d_vv_dy + d_wv_dz
        end
    end

    # --- w-momentum (bottom face of cell k, height dzf[k]) ---
    @inbounds for k in kk
        idz = 1 / dzf[k]
        s   = dzc[k-1] + dzc[k]
        alo = dzc[k-1] / s            # volume-flux weights of the side faces
        ahi = dzc[k]   / s
        for j in jj, i in ii
            uwe = alo*ud[i+1, j, k-1] + ahi*ud[i+1, j, k]
            wue = 0.5*(wd[i,   j, k  ] + wd[i+1, j, k])
            uww = alo*ud[i,   j, k-1] + ahi*ud[i,   j, k]
            wuw = 0.5*(wd[i-1, j, k  ] + wd[i,   j, k])
            d_uw_dx = (uwe*wue - uww*wuw) * idx

            vwn = alo*vd[i, j+1, k-1] + ahi*vd[i, j+1, k]
            wvn = 0.5*(wd[i, j,   k  ] + wd[i, j+1, k])
            vws = alo*vd[i, j,   k-1] + ahi*vd[i, j,   k]
            wvs = 0.5*(wd[i, j-1, k  ] + wd[i, j,   k])
            d_vw_dy = (vwn*wvn - vws*wvs) * idy

            wct = 0.5*(wd[i, j, k  ] + wd[i, j, k+1])
            wcb = 0.5*(wd[i, j, k-1] + wd[i, j, k  ])
            d_ww_dz = (wct*wct - wcb*wcb) * idz

            tw.data[i, j, k] -= d_uw_dx + d_vw_dy + d_ww_dz
        end
    end
    return (tu, tv, tw)
end

# --- scalar advection (staggered, flux form) --------------------------------

"""
    _advect_centered!(tend, u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, ϕ, g::StretchedGrid)

Flux-form `-∂(uⱼϕ)/∂xⱼ` for a cell-centred scalar on a stretched grid: the
vertical flux difference is divided by the cell thickness `Δz_c[k]`, so
`Σ ϕ ΔV` changes only by the boundary fluxes. The face value is the
arithmetic mean of the two neighbours (the RegularGrid stencil), which
keeps `Σ ϕ² ΔV` invariant under a solenoidal flow.
"""
function _advect_centered!(tend::Field{CellCenter},
                           u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                           ϕ::Field{CellCenter}, g::StretchedGrid)
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc = g.dzc
    ii, jj, kk = interior(g)
    d = ϕ.data
    @inbounds for k in kk
        idz = 1 / dzc[k]
        for j in jj, i in ii
            uϕe = u.data[i+1,j,k] * 0.5 * (d[i,  j,k] + d[i+1,j,k])
            uϕw = u.data[i,  j,k] * 0.5 * (d[i-1,j,k] + d[i,  j,k])
            vϕn = v.data[i,j+1,k] * 0.5 * (d[i,j,  k] + d[i,j+1,k])
            vϕs = v.data[i,j,  k] * 0.5 * (d[i,j-1,k] + d[i,j,  k])
            wϕt = w.data[i,j,k+1] * 0.5 * (d[i,j,k  ] + d[i,j,k+1])
            wϕb = w.data[i,j,k  ] * 0.5 * (d[i,j,k-1] + d[i,j,k  ])
            tend.data[i,j,k] -= (uϕe - uϕw)*idx + (vϕn - vϕs)*idy + (wϕt - wϕb)*idz
        end
    end
    return tend
end

# --- moist buoyancy ---------------------------------------------------------

"""
    _add_buoyancy_moist!(tw, θ, qv, θ₀, g_accel, g::StretchedGrid)

Moist Boussinesq buoyancy `g(θᵥ - θ₀)/θ₀`, `θᵥ = θ(1 + 0.61 qv)`, on the
z-faces, with `θ` and `qv` brought to the face by the same volume-weighted
(geometric linear) interpolation as the dry [`_add_buoyancy!`](@ref)
method, so a horizontally uniform moist column is in exact discrete
hydrostatic balance.
"""
function _add_buoyancy_moist!(tw::Field{FaceZ},
                              θ::Field{CellCenter}, qv::Field{CellCenter},
                              θ₀::Real, g_accel::Real, g::StretchedGrid)
    coef = g_accel / θ₀
    dzc = g.dzc
    ii, jj, kk = interior(g)
    @inbounds for k in kk
        wlo = dzc[k]   / (dzc[k-1] + dzc[k])
        whi = dzc[k-1] / (dzc[k-1] + dzc[k])
        for j in jj, i in ii
            θface  = wlo * θ.data[i,j,k-1]  + whi * θ.data[i,j,k]
            qvface = wlo * qv.data[i,j,k-1] + whi * qv.data[i,j,k]
            θv     = θface * (1 + 0.61 * qvface)
            tw.data[i,j,k] += coef * (θv - θ₀)
        end
    end
    return tw
end

# --- Smagorinsky ------------------------------------------------------------

"""
    _compute_smagorinsky_νt!(νt, u, v, w, C_s, g::StretchedGrid)

Smagorinsky–Lilly `νt = (C_s Δ_k)² |S|` with the local filter width of
Deardorff (1970, J. Fluid Mech. 41, 453), `Δ_k = (Δx Δy Δz_c[k])^{1/3}`, and
the strain from the one strain stencil (`Operators._strain_at`) with the
local vertical metrics: `∂w/∂z` across the cell (`1/Δz_c[k]`), `∂u/∂z` and
`∂v/∂z` across the two neighbouring centres (`2/(Δz_f[k] + Δz_f[k+1])`).
"""
function _compute_smagorinsky_νt!(νt::Field{CellCenter},
                                  u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                  C_s::Real, g::StretchedGrid)
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc, dzf = g.dzc, g.dzf
    ii, jj, kk = interior(g)
    ud, vd, wd = u.data, v.data, w.data
    @inbounds for k in kk
        idz     = 1 / dzc[k]
        idz_off = 2 / (dzf[k] + dzf[k+1])
        Δ    = (g.dx * g.dy * dzc[k]) ^ (1/3)
        coef = (C_s * Δ) ^ 2
        for j in jj, i in ii
            νt.data[i,j,k] = coef * _strain_at(ud, vd, wd, i, j, k, idx, idy, idz, idz_off)[7]
        end
    end
    return νt
end

# --- eddy-viscosity momentum fluxes ----------------------------------------
# ∂/∂xⱼ(νₜ ∂uᵢ/∂xⱼ) with νₜ averaged to the flux location exactly as on the
# RegularGrid (4-cell edge means); the vertical gradient between two
# centres uses dzf, between two z-faces dzc, and the flux difference is
# divided by the height of the control volume (dzc for u, v; dzf for w).

function _add_eddy_viscosity_u!(tu::Field{FaceX}, u::Field{FaceX},
                                 νt::Field{CellCenter}, g::StretchedGrid)
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc, dzf = g.dzc, g.dzf
    ii, jj, kk = interior(g)
    n, d = νt.data, u.data
    @inbounds for k in kk
        iv = 1 / dzc[k]; idzt = 1 / dzf[k+1]; idzb = 1 / dzf[k]
        for j in jj, i in ii
            fxE = n[i  ,j,k] * (d[i+1,j,k] - d[i  ,j,k]) * idx
            fxW = n[i-1,j,k] * (d[i  ,j,k] - d[i-1,j,k]) * idx

            νt_eN = 0.25*(n[i-1,j,k]   + n[i,j,k]   + n[i-1,j+1,k] + n[i,j+1,k])
            νt_eS = 0.25*(n[i-1,j-1,k] + n[i,j-1,k] + n[i-1,j,k]   + n[i,j,k])
            fyN = νt_eN * (d[i,j+1,k] - d[i,j  ,k]) * idy
            fyS = νt_eS * (d[i,j,  k] - d[i,j-1,k]) * idy

            νt_eT = 0.25*(n[i-1,j,k]   + n[i,j,k]   + n[i-1,j,k+1] + n[i,j,k+1])
            νt_eB = 0.25*(n[i-1,j,k-1] + n[i,j,k-1] + n[i-1,j,k]   + n[i,j,k])
            fzT = νt_eT * (d[i,j,k+1] - d[i,j,k  ]) * idzt
            fzB = νt_eB * (d[i,j,k  ] - d[i,j,k-1]) * idzb

            tu.data[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*iv
        end
    end
    return tu
end

function _add_eddy_viscosity_v!(tv::Field{FaceY}, v::Field{FaceY},
                                 νt::Field{CellCenter}, g::StretchedGrid)
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc, dzf = g.dzc, g.dzf
    ii, jj, kk = interior(g)
    n, d = νt.data, v.data
    @inbounds for k in kk
        iv = 1 / dzc[k]; idzt = 1 / dzf[k+1]; idzb = 1 / dzf[k]
        for j in jj, i in ii
            fyN = n[i,j  ,k] * (d[i,j+1,k] - d[i,j  ,k]) * idy
            fyS = n[i,j-1,k] * (d[i,j  ,k] - d[i,j-1,k]) * idy

            νt_eE = 0.25*(n[i,j-1,k]   + n[i+1,j-1,k] + n[i,j,k]   + n[i+1,j,k])
            νt_eW = 0.25*(n[i-1,j-1,k] + n[i,j-1,k]   + n[i-1,j,k] + n[i,j,k])
            fxE = νt_eE * (d[i+1,j,k] - d[i,  j,k]) * idx
            fxW = νt_eW * (d[i,  j,k] - d[i-1,j,k]) * idx

            νt_eT = 0.25*(n[i,j-1,k]   + n[i,j,k]   + n[i,j-1,k+1] + n[i,j,k+1])
            νt_eB = 0.25*(n[i,j-1,k-1] + n[i,j,k-1] + n[i,j-1,k]   + n[i,j,k])
            fzT = νt_eT * (d[i,j,k+1] - d[i,j,k  ]) * idzt
            fzB = νt_eB * (d[i,j,k  ] - d[i,j,k-1]) * idzb

            tv.data[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*iv
        end
    end
    return tv
end

function _add_eddy_viscosity_w!(tw::Field{FaceZ}, w::Field{FaceZ},
                                 νt::Field{CellCenter}, g::StretchedGrid)
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc, dzf = g.dzc, g.dzf
    ii, jj, kk = interior(g)
    n, d = νt.data, w.data
    @inbounds for k in kk
        iv = 1 / dzf[k]; idzt = 1 / dzc[k]; idzb = 1 / dzc[k-1]
        for j in jj, i in ii
            fzT = n[i,j,k  ] * (d[i,j,k+1] - d[i,j,k  ]) * idzt
            fzB = n[i,j,k-1] * (d[i,j,k  ] - d[i,j,k-1]) * idzb

            νt_eE = 0.25*(n[i,j,k-1]   + n[i+1,j,k-1] + n[i,j,k]   + n[i+1,j,k])
            νt_eW = 0.25*(n[i-1,j,k-1] + n[i,j,k-1]   + n[i-1,j,k] + n[i,j,k])
            fxE = νt_eE * (d[i+1,j,k] - d[i,  j,k]) * idx
            fxW = νt_eW * (d[i,  j,k] - d[i-1,j,k]) * idx

            νt_eN = 0.25*(n[i,j,k-1]   + n[i,j+1,k-1] + n[i,j,k]   + n[i,j+1,k])
            νt_eS = 0.25*(n[i,j-1,k-1] + n[i,j,k-1]   + n[i,j-1,k] + n[i,j,k])
            fyN = νt_eN * (d[i,j+1,k] - d[i,j  ,k]) * idy
            fyS = νt_eS * (d[i,j,  k] - d[i,j-1,k]) * idy

            tw.data[i,j,k] += (fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*iv
        end
    end
    return tw
end

# --- eddy diffusion of a cell-centred scalar --------------------------------

_add_eddy_diffusion_scalar!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                            K::Field{CellCenter}, s::Real, g::StretchedGrid) =
    _add_eddy_diffusion_scalar!(tϕ, ϕ, K, K, s, g)

"""
    _add_eddy_diffusion_scalar!(tϕ, ϕ, Kh, Kv, s, g::StretchedGrid)

`s · [∂x(Kh ∂xϕ) + ∂y(Kh ∂yϕ) + ∂z(Kv ∂zϕ)]` on a stretched grid: the
z-face gradient spans the two centres (`Δz_f`), the flux difference the
cell (`Δz_c`), so `Σ tϕ ΔV` telescopes to the wall fluxes and the operator
is symmetric negative semi-definite in the volume-weighted inner product.
"""
function _add_eddy_diffusion_scalar!(tϕ::Field{CellCenter}, ϕ::Field{CellCenter},
                                      Kh::Field{CellCenter}, Kv::Field{CellCenter},
                                      s::Real, g::StretchedGrid)
    idx = 1 / g.dx; idy = 1 / g.dy
    dzc, dzf = g.dzc, g.dzf
    ii, jj, kk = interior(g)
    H = Kh.data; V = Kv.data; d = ϕ.data
    @inbounds for k in kk
        iv = 1 / dzc[k]; idzt = 1 / dzf[k+1]; idzb = 1 / dzf[k]
        for j in jj, i in ii
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
            fzT = νT * (d[i,j,k+1] - d[i,j,k  ]) * idzt
            fzB = νB * (d[i,j,k  ] - d[i,j,k-1]) * idzb

            tϕ.data[i,j,k] += s * ((fxE - fxW)*idx + (fyN - fyS)*idy + (fzT - fzB)*iv)
        end
    end
    return tϕ
end

end # module LegacyFV
