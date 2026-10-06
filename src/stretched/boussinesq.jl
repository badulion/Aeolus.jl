# =========================================================================
# #151 (N-grid) — the Boussinesq core on a StretchedGrid.
# =========================================================================
#
# Included into `module StretchedGrids`. Phase 24 gave the StretchedGrid the
# metric-correct derivative, divergence, Laplacian, interpolation, sponge,
# surface-flux and buoyancy kernels; this file adds the rest of what
# `BoussinesqSimulation` calls in a step, so that `BoussinesqSimulation(g)`
# runs on a vertically stretched mesh:
#
#   * flux-form momentum advection          `_momentum_advect!`
#   * flux-form scalar advection            `_advect_centered!` (staggered)
#   * the Laplacian of a z-face field       `_laplacian!(::Field{FaceZ}, …)`
#   * moist buoyancy                        `_add_buoyancy_moist!`
#   * the Smagorinsky eddy viscosity        `_compute_smagorinsky_νt!`
#   * eddy-viscosity momentum fluxes        `_add_eddy_viscosity_{u,v,w}!`
#   * eddy diffusion of a scalar            `_add_eddy_diffusion_scalar!`
#   * the vertical metrics of the core      `Simulations._dzc/_dzf/_dz_min`
#
# Every vertical metric is read from the grid's own vectors (`dzc` =
# `cell_thickness(g)`, `dzf` = `center_spacing(g)`); x and y are uniform and
# use the RegularGrid stencils verbatim. On a `:uniform` StretchedGrid
# (`dzc ≡ dzf ≡ Δz`) each kernel is algebraically the RegularGrid kernel, so
# the two meshes give the same run (tested to 1e-12 over 20 steps).
#
# Control volumes of the staggered grid in z (storage index k):
#
#   u, v, θ, p  at zc[k], control volume [zf[k],  zf[k+1]], height dzc[k]
#   w           at zf[k], control volume [zc[k-1], zc[k]],  height dzf[k]
#
# Momentum advection is the divergence form with (i) the transported
# velocity averaged arithmetically to the control-volume face and (ii) the
# transporting velocity equal to the *volume flux* through that face. For a
# w control volume the side face spans half of the u-cell k-1 below and half
# of the u-cell k above, so its transporting u is the thickness-weighted
# mean (dzc[k-1] u[k-1] + dzc[k] u[k]) / (dzc[k-1] + dzc[k]). With these
# fluxes the volume-flux divergence of every u, v and w control volume is a
# thickness-weighted mean of the divergence of the pressure cells it
# overlaps, so a discretely solenoidal field transports momentum without
# creating or destroying kinetic energy: the advection operator is
# skew-symmetric in the volume-weighted inner product (Morinishi, Lund,
# Vasilyev & Moin 1998, J. Comput. Phys. 143, 90; Vasilyev 2000, J. Comput.
# Phys. 157, 746, for the non-uniform extension). The arithmetic face mean
# is second-order accurate on a smoothly stretched grid (it is evaluated a
# distance (dzc[k+1] - dzc[k])/4 = O(Δz²) from the face).

import ..Operators.FiniteVolume: _momentum_advect!, _advect_centered!,
                                 _add_buoyancy_moist!, _compute_smagorinsky_νt!,
                                 _add_eddy_viscosity_u!, _add_eddy_viscosity_v!,
                                 _add_eddy_viscosity_w!, _add_eddy_diffusion_scalar!
using ..Operators: _strain_at
import ..Simulations: _dzc, _dzf, _dz_min, _check_boussinesq_mesh

# --- the vertical metrics of the core -------------------------------------

_dzc(g::StretchedGrid, k) = @inbounds g.dzc[k]
_dzf(g::StretchedGrid, k) = @inbounds g.dzf[k]
_dz_min(g::StretchedGrid) = minimum(@view g.dzc[g.Hz+1:g.Hz+g.Nz])
_check_boussinesq_mesh(::StretchedGrid) = nothing

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

# --- N² of the stratification (the TKE closure's buoyancy term, #259) ------

import ..Operators: _buoyancy_frequency_squared!
using ..Operators: _n2_cell, _humidity_data, _θref

"""
    _buoyancy_frequency_squared!(N2, θ, qv, θ₀, g, m::StretchedGrid)

``N^2_k = (g/θ_0)(θ_{v,k+1/2} - θ_{v,k-1/2})/Δz_c[k]``: the face values of
the uniform form (the mean of the two adjacent centres) differenced over
the cell thickness. Then ``N^2_k = (g/θ_0)(θ_{k+1} - θ_{k-1})/(2Δz_c[k])``
is exactly the per-cell share of the stretched κ_t operator's
potential-energy change (`_add_eddy_diffusion_scalar!` above: face
gradients over `dzf`, divergence over `dzc`), so ``Σ(−κ_t N^2)V`` is the
buoyancy flux of the SGS heat flux, as on the uniform grid. For a linear
θ it is ``Γ(Δz_c[k-1]/2 + Δz_c[k] + Δz_c[k+1]/2)/(2Δz_c[k])``: Γ on the
uniform levels, Γ(1 + 1.5e-3) on a geometric part of ratio 1.08, and
first order where the ratio jumps (+1.7 % in the last 25 m cell under the
first stretched one of the nesting parent, #259). With condensate
(`Operators._Cloudy`, #239) the per-cell stencil is the uniform one's
(condensate loading, the saturated N² in cells with ``q_c > 0``), over
the same ``Δz_c[k]``.
"""
function _buoyancy_frequency_squared!(N2::Field{CellCenter}, θ::Field{CellCenter}, qv,
                                      θ₀, g, m::StretchedGrid)
    qd = _humidity_data(qv)
    θd, nd = θ.data, N2.data
    ii, jj, kk = interior(m)
    @inbounds for k in kk
        dz = m.dzc[k]
        for j in jj, i in ii
            nd[i, j, k] = _n2_cell(θd, qd, i, j, k, dz, g, _θref(θ₀, k))
        end
    end
    return N2
end
