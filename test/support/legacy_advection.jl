# KA-e (#302): the scalar-transport and momentum-advection kernels of
# src/AdvectionSchemes.jl as they were before the port (KA-d branch
# 69a5dea + master 3a23110 merged), transcribed verbatim: the serial
# loops of the flux-form scalar sweep, the face-flux assembly, the
# positivity limiter, the WENO-Z / skew-symmetric momentum advection, the
# variance-conserving scalar advection and boundary_scalar_flux. The
# point functions (reconstructions, `_face_x/_y/_z`, `_bsf_face_flux`, …)
# are unchanged by the port and are taken from the package. The reference
# of test/ka_advection.jl and of the KA-e timings (old
# loops vs launches). Nothing here is used by the package.
# ---------------------------------------------------------------------------
module LegacyAdvection
using Aeolus
using Aeolus.Grids: RegularGrid, interior, CellCenter, FaceX, FaceY, FaceZ
using Aeolus.Fields: Field, mesh
using Aeolus.BoundaryConditions: FieldBoundaryConditions, fill_halos!, AbstractOpenBoundary, Periodic
using Aeolus.StretchedGrids: StretchedGrid
using Aeolus.AdvectionSchemes: AbstractReconstruction, FluxLimiterWorkspace, required_halo,
                               _face_x, _face_y, _face_z, _check_halo, _atlo, _data_or_nothing,
                               _inv_dz, _check_stretched_recon, _reconstruction,
                               _flux_reconstruction, _check_open_halo, _bsf_face_flux, _wflux,
                               _diffusive_parts, _correction_of, _corrected
const _PosMesh = Union{RegularGrid,StretchedGrid}

function _advect_reconstructed!(tend::Field{CellCenter},
                                u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                ϕ::Field{CellCenter},
                                r::AbstractReconstruction, g::RegularGrid)
    _check_halo(r, g)
    idx = 1/g.dx; idy = 1/g.dy; idz = 1/g.dz
    ii, jj, kk = interior(g)
    ϕd = ϕ.data; ud = u.data; vd = v.data; wd = w.data; td = tend.data
    @inbounds for k in kk, j in jj, i in ii
        uw = ud[i,   j, k]; ue = ud[i+1, j, k]
        Fx = (ue*_face_x(r, ϕd, i, j, k, ue) - uw*_face_x(r, ϕd, i-1, j, k, uw))*idx

        vs = vd[i, j,   k]; vn = vd[i, j+1, k]
        Fy = (vn*_face_y(r, ϕd, i, j, k, vn) - vs*_face_y(r, ϕd, i, j-1, k, vs))*idy

        wb = wd[i, j, k];   wt = wd[i, j, k+1]
        Fz = (wt*_face_z(r, ϕd, i, j, k, wt) - wb*_face_z(r, ϕd, i, j, k-1, wb))*idz

        td[i, j, k] -= Fx + Fy + Fz
    end
    return tend
end

function _reconstructed_fluxes!(Fx::Field{FaceX}, Fy::Field{FaceY}, Fz::Field{FaceZ},
                                u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                ϕ::Field{CellCenter}, r::AbstractReconstruction,
                                g::_PosMesh)
    _check_stretched_recon(r, g)
    _check_halo(r, g)
    ii, jj, kk = interior(g)
    ϕd = ϕ.data
    @inbounds for k in kk, j in jj, i in first(ii):last(ii)+1
        vel = u.data[i, j, k]
        Fx.data[i, j, k] = vel * _face_x(r, ϕd, i-1, j, k, vel)
    end
    @inbounds for k in kk, j in first(jj):last(jj)+1, i in ii
        vel = v.data[i, j, k]
        Fy.data[i, j, k] = vel * _face_y(r, ϕd, i, j-1, k, vel)
    end
    @inbounds for k in first(kk):last(kk)+1, j in jj, i in ii
        vel = w.data[i, j, k]
        Fz.data[i, j, k] = vel * _face_z(r, ϕd, i, j, k-1, vel)
    end
    return (Fx, Fy, Fz)
end

function _upwind_fluxes!(Fx::Field{FaceX}, Fy::Field{FaceY}, Fz::Field{FaceZ},
                         u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                         ϕ::Field{CellCenter}, g::_PosMesh)
    ii, jj, kk = interior(g)
    ϕd = ϕ.data
    @inbounds for k in kk, j in jj, i in first(ii):last(ii)+1
        vel = u.data[i, j, k]
        Fx.data[i, j, k] = vel * (vel >= 0 ? ϕd[i-1, j, k] : ϕd[i, j, k])
    end
    @inbounds for k in kk, j in first(jj):last(jj)+1, i in ii
        vel = v.data[i, j, k]
        Fy.data[i, j, k] = vel * (vel >= 0 ? ϕd[i, j-1, k] : ϕd[i, j, k])
    end
    @inbounds for k in first(kk):last(kk)+1, j in jj, i in ii
        vel = w.data[i, j, k]
        Fz.data[i, j, k] = vel * (vel >= 0 ? ϕd[i, j, k-1] : ϕd[i, j, k])
    end
    return (Fx, Fy, Fz)
end

function _add_flux_divergence!(tend::Field{CellCenter},
                               Fx::Field{FaceX}, Fy::Field{FaceY}, Fz::Field{FaceZ},
                               g::_PosMesh)
    idx = 1/g.dx; idy = 1/g.dy
    ii, jj, kk = interior(g)
    fx = Fx.data; fy = Fy.data; fz = Fz.data; td = tend.data
    @inbounds for k in kk, j in jj, i in ii
        idz = _inv_dz(g, k)
        td[i,j,k] -= (fx[i+1,j,k] - fx[i,j,k])*idx +
                     (fy[i,j+1,k] - fy[i,j,k])*idy +
                     (fz[i,j,k+1] - fz[i,j,k])*idz
    end
    return tend
end

# one-layer face-normal periodic wrap of a cell-centred field
function _wrap_one_layer!(f::Field{CellCenter}, g::_PosMesh)
    ii, jj, kk = interior(g)
    d = f.data
    i1, i2 = first(ii), last(ii); j1, j2 = first(jj), last(jj); k1, k2 = first(kk), last(kk)
    @inbounds for k in kk, j in jj
        d[i1-1, j, k] = d[i2, j, k]; d[i2+1, j, k] = d[i1, j, k]
    end
    @inbounds for k in kk, i in ii
        d[i, j1-1, k] = d[i, j2, k]; d[i, j2+1, k] = d[i, j1, k]
    end
    @inbounds for j in jj, i in ii
        d[i, j, k1-1] = d[i, j, k2]; d[i, j, k2+1] = d[i, j, k1]
    end
    return f
end

function apply_positivity_limiter!(flux_x::Field{FaceX,T}, flux_y::Field{FaceY,T},
                                   flux_z::Field{FaceZ,T}, ϕ::Field{CellCenter,T},
                                   dt::Real, m::_PosMesh;
                                   flux_lo = nothing,
                                   ratio::Union{Nothing,Field{CellCenter}} = nothing,
                                   bcs::Union{Nothing,FieldBoundaryConditions} = nothing) where {T}
    Δt = T(dt)
    R  = ratio === nothing ? Field(CellCenter(), m) : ratio
    Lx = flux_lo === nothing ? nothing : _data_or_nothing(flux_lo[1])
    Ly = flux_lo === nothing ? nothing : _data_or_nothing(flux_lo[2])
    Lz = flux_lo === nothing ? nothing : _data_or_nothing(flux_lo[3])
    return _positivity_limit!(flux_x, flux_y, flux_z, ϕ, Δt, m, R, Lx, Ly, Lz, bcs)
end

function _positivity_limit!(flux_x, flux_y, flux_z, ϕ, Δt::T, m::_PosMesh,
                            R, Lx, Ly, Lz, bcs) where {T}
    idx = 1/m.dx; idy = 1/m.dy
    ii, jj, kk = interior(m)
    fx = flux_x.data; fy = flux_y.data; fz = flux_z.data
    ϕd = ϕ.data; Rd = R.data
    z = zero(T)

    # --- pass 1: antidiffusive outflow budget per cell --------------------
    @inbounds for k in kk, j in jj, i in ii
        idz = T(_inv_dz(m, k))
        lxe = _atlo(Lx, i+1, j, k); lxw = _atlo(Lx, i,   j, k)
        lyn = _atlo(Ly, i, j+1, k); lys = _atlo(Ly, i,   j, k)
        lzt = _atlo(Lz, i, j, k+1); lzb = _atlo(Lz, i, j, k  )

        axe = fx[i+1,j,k] - lxe; axw = fx[i,j,k] - lxw
        ayn = fy[i,j+1,k] - lyn; ays = fy[i,j,k] - lys
        azt = fz[i,j,k+1] - lzt; azb = fz[i,j,k] - lzb

        Pm = Δt * ((max(axe, z) + max(-axw, z))*idx +
                   (max(ayn, z) + max(-ays, z))*idy +
                   (max(azt, z) + max(-azb, z))*idz)

        div_lo = (lxe - lxw)*idx + (lyn - lys)*idy + (lzt - lzb)*idz
        Q = max(ϕd[i,j,k] - Δt*div_lo, z)

        Rd[i,j,k] = Pm > z ? min(one(T), Q/Pm) : one(T)
    end

    # --- limiter halo (needed by the closing faces) -----------------------
    if bcs === nothing
        _wrap_one_layer!(R, m)
    else
        fill_halos!(R, bcs)
    end

    # --- pass 2: scale each antidiffusive flux by the donor's factor ------
    @inbounds for k in kk, j in jj, i in first(ii):last(ii)+1
        lo = _atlo(Lx, i, j, k)
        a  = fx[i,j,k] - lo
        c  = a >= z ? Rd[i-1,j,k] : Rd[i,j,k]
        fx[i,j,k] = lo + c*a
    end
    @inbounds for k in kk, j in first(jj):last(jj)+1, i in ii
        lo = _atlo(Ly, i, j, k)
        a  = fy[i,j,k] - lo
        c  = a >= z ? Rd[i,j-1,k] : Rd[i,j,k]
        fy[i,j,k] = lo + c*a
    end
    @inbounds for k in first(kk):last(kk)+1, j in jj, i in ii
        lo = _atlo(Lz, i, j, k)
        a  = fz[i,j,k] - lo
        c  = a >= z ? Rd[i,j,k-1] : Rd[i,j,k]
        fz[i,j,k] = lo + c*a
    end
    return (flux_x, flux_y, flux_z)
end

function advect_positive!(tend::Field{CellCenter,T}, u::Field{FaceX,T},
                          v::Field{FaceY,T}, w::Field{FaceZ,T},
                          ϕ::Field{CellCenter,T}, dt::Real;
                          scheme::Union{Symbol,AbstractReconstruction} = :weno_z5,
                          C_T::Real = 1e-5, p::Integer = 2,
                          workspace::Union{Nothing,FluxLimiterWorkspace} = nothing,
                          bcs::Union{Nothing,FieldBoundaryConditions} = nothing) where {T}
    m  = mesh(ϕ)
    r  = _reconstruction(scheme; C_T = C_T, p = p)
    ws = workspace === nothing ? FluxLimiterWorkspace(m) : workspace

    _reconstructed_fluxes!(ws.Fx, ws.Fy, ws.Fz, u, v, w, ϕ, r, m)
    _upwind_fluxes!(ws.Lx, ws.Ly, ws.Lz, u, v, w, ϕ, m)
    apply_positivity_limiter!(ws.Fx, ws.Fy, ws.Fz, ϕ, dt, m;
                              flux_lo = (ws.Lx, ws.Ly, ws.Lz), ratio = ws.ratio, bcs = bcs)
    _add_flux_divergence!(tend, ws.Fx, ws.Fy, ws.Fz, m)
    return tend
end

function _momentum_advect_recon!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                 u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                 r::AbstractReconstruction, g::RegularGrid)
    _check_halo(r, g)
    idx = 1/g.dx; idy = 1/g.dy; idz = 1/g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data

    # --- u-momentum (west face of cell i,j,k) ---------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i,   j, k] + ud[i+1, j, k])/2      # at cell centre i
        Uw = (ud[i-1, j, k] + ud[i,   j, k])/2      # at cell centre i-1
        d_uu_dx = (Ue*_face_x(r, ud, i, j, k, Ue) - Uw*_face_x(r, ud, i-1, j, k, Uw))*idx

        Vn = (vd[i-1, j+1, k] + vd[i, j+1, k])/2    # at x-y edge (i,j+1)
        Vs = (vd[i-1, j,   k] + vd[i, j,   k])/2    # at x-y edge (i,j)
        d_vu_dy = (Vn*_face_y(r, ud, i, j, k, Vn) - Vs*_face_y(r, ud, i, j-1, k, Vs))*idy

        Wt = (wd[i-1, j, k+1] + wd[i, j, k+1])/2    # at x-z edge (i,k+1)
        Wb = (wd[i-1, j, k  ] + wd[i, j, k  ])/2    # at x-z edge (i,k)
        d_wu_dz = (Wt*_face_z(r, ud, i, j, k, Wt) - Wb*_face_z(r, ud, i, j, k-1, Wb))*idz

        tu.data[i, j, k] -= d_uu_dx + d_vu_dy + d_wu_dz
    end

    # --- v-momentum (south face of cell i,j,k) --------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i+1, j-1, k] + ud[i+1, j, k])/2
        Uw = (ud[i,   j-1, k] + ud[i,   j, k])/2
        d_uv_dx = (Ue*_face_x(r, vd, i, j, k, Ue) - Uw*_face_x(r, vd, i-1, j, k, Uw))*idx

        Vn = (vd[i, j,   k] + vd[i, j+1, k])/2
        Vs = (vd[i, j-1, k] + vd[i, j,   k])/2
        d_vv_dy = (Vn*_face_y(r, vd, i, j, k, Vn) - Vs*_face_y(r, vd, i, j-1, k, Vs))*idy

        Wt = (wd[i, j-1, k+1] + wd[i, j, k+1])/2
        Wb = (wd[i, j-1, k  ] + wd[i, j, k  ])/2
        d_wv_dz = (Wt*_face_z(r, vd, i, j, k, Wt) - Wb*_face_z(r, vd, i, j, k-1, Wb))*idz

        tv.data[i, j, k] -= d_uv_dx + d_vv_dy + d_wv_dz
    end

    # --- w-momentum (bottom face of cell i,j,k) -------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i+1, j, k-1] + ud[i+1, j, k])/2
        Uw = (ud[i,   j, k-1] + ud[i,   j, k])/2
        d_uw_dx = (Ue*_face_x(r, wd, i, j, k, Ue) - Uw*_face_x(r, wd, i-1, j, k, Uw))*idx

        Vn = (vd[i, j+1, k-1] + vd[i, j+1, k])/2
        Vs = (vd[i, j,   k-1] + vd[i, j,   k])/2
        d_vw_dy = (Vn*_face_y(r, wd, i, j, k, Vn) - Vs*_face_y(r, wd, i, j-1, k, Vs))*idy

        Wt = (wd[i, j, k  ] + wd[i, j, k+1])/2
        Wb = (wd[i, j, k-1] + wd[i, j, k  ])/2
        d_ww_dz = (Wt*_face_z(r, wd, i, j, k, Wt) - Wb*_face_z(r, wd, i, j, k-1, Wb))*idz

        tw.data[i, j, k] -= d_uw_dx + d_vw_dy + d_ww_dz
    end

    return (tu, tv, tw)
end

function _momentum_advect_skew!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                g::RegularGrid)
    h2x = 1/(2g.dx); h2y = 1/(2g.dy); h2z = 1/(2g.dz)
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data

    # --- u-momentum ------------------------------------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i,   j, k] + ud[i+1, j, k])/2
        Uw = (ud[i-1, j, k] + ud[i,   j, k])/2
        Vn = (vd[i-1, j+1, k] + vd[i, j+1, k])/2
        Vs = (vd[i-1, j,   k] + vd[i, j,   k])/2
        Wt = (wd[i-1, j, k+1] + wd[i, j, k+1])/2
        Wb = (wd[i-1, j, k  ] + wd[i, j, k  ])/2

        tu.data[i,j,k] -= (Ue*ud[i+1,j,k] - Uw*ud[i-1,j,k])*h2x +
                          (Vn*ud[i,j+1,k] - Vs*ud[i,j-1,k])*h2y +
                          (Wt*ud[i,j,k+1] - Wb*ud[i,j,k-1])*h2z
    end

    # --- v-momentum ------------------------------------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i+1, j-1, k] + ud[i+1, j, k])/2
        Uw = (ud[i,   j-1, k] + ud[i,   j, k])/2
        Vn = (vd[i, j,   k] + vd[i, j+1, k])/2
        Vs = (vd[i, j-1, k] + vd[i, j,   k])/2
        Wt = (wd[i, j-1, k+1] + wd[i, j, k+1])/2
        Wb = (wd[i, j-1, k  ] + wd[i, j, k  ])/2

        tv.data[i,j,k] -= (Ue*vd[i+1,j,k] - Uw*vd[i-1,j,k])*h2x +
                          (Vn*vd[i,j+1,k] - Vs*vd[i,j-1,k])*h2y +
                          (Wt*vd[i,j,k+1] - Wb*vd[i,j,k-1])*h2z
    end

    # --- w-momentum ------------------------------------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i+1, j, k-1] + ud[i+1, j, k])/2
        Uw = (ud[i,   j, k-1] + ud[i,   j, k])/2
        Vn = (vd[i, j+1, k-1] + vd[i, j+1, k])/2
        Vs = (vd[i, j,   k-1] + vd[i, j,   k])/2
        Wt = (wd[i, j, k  ] + wd[i, j, k+1])/2
        Wb = (wd[i, j, k-1] + wd[i, j, k  ])/2

        tw.data[i,j,k] -= (Ue*wd[i+1,j,k] - Uw*wd[i-1,j,k])*h2x +
                          (Vn*wd[i,j+1,k] - Vs*wd[i,j-1,k])*h2y +
                          (Wt*wd[i,j,k+1] - Wb*wd[i,j,k-1])*h2z
    end

    return (tu, tv, tw)
end

function _advect_variance_conserving!(tend::Field{CellCenter},
                                     u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                     ϕ::Field{CellCenter}, g::RegularGrid)
    h2x = 1/(2g.dx); h2y = 1/(2g.dy); h2z = 1/(2g.dz)
    ii, jj, kk = interior(g)
    ϕd = ϕ.data; ud = u.data; vd = v.data; wd = w.data
    @inbounds for k in kk, j in jj, i in ii
        tend.data[i,j,k] -= (ud[i+1,j,k]*ϕd[i+1,j,k] - ud[i,j,k]*ϕd[i-1,j,k])*h2x +
                            (vd[i,j+1,k]*ϕd[i,j+1,k] - vd[i,j,k]*ϕd[i,j-1,k])*h2y +
                            (wd[i,j,k+1]*ϕd[i,j,k+1] - wd[i,j,k]*ϕd[i,j,k-1])*h2z
    end
    return tend
end

function boundary_scalar_flux(sim, ϕ::Field{CellCenter};
                              scheme = :centered, bcs::FieldBoundaryConditions = sim.θ_bcs,
                              weights = nothing, fill::Bool = true,
                              diffusivity = nothing, levels = nothing,
                              correction_levels = nothing, face_fluxes = nothing)
    r = _flux_reconstruction(scheme)
    g = mesh(ϕ)
    vb = sim.bcs
    H = required_halo(r)
    # one call per face, not a loop over a heterogeneous tuple (which boxes
    # its elements: an allocation on every call, #327)
    _check_open_halo(:west, vb.west, g.Hx, H, r); _check_open_halo(:east, vb.east, g.Hx, H, r)
    _check_open_halo(:south, vb.south, g.Hy, H, r); _check_open_halo(:north, vb.north, g.Hy, H, r)
    _check_open_halo(:bottom, vb.bottom, g.Hz, H, r); _check_open_halo(:top, vb.top, g.Hz, H, r)
    _check_halo(r, g)
    levels === nothing || length(levels) == g.Nz ||
        throw(DimensionMismatch("boundary_scalar_flux: levels needs Nz = $(g.Nz) entries"))
    correction_levels === nothing || length(correction_levels) == g.Nz ||
        throw(DimensionMismatch("boundary_scalar_flux: correction_levels needs Nz = $(g.Nz) entries"))
    if fill
        fill_halos!(sim.u, vb); fill_halos!(sim.v, vb); fill_halos!(sim.w, vb)
        fill_halos!(ϕ, bcs)
    end
    Wc, Wf = weights === nothing ? (nothing, nothing) : weights
    T = eltype(ϕ.data)
    ϕd = ϕ.data; ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    Fxd, Fyd, Fzd = face_fluxes === nothing ? (nothing, nothing, nothing) :
                    (face_fluxes[1].data, face_fluxes[2].data, face_fluxes[3].data)
    ii, jj, kk = interior(g)
    k0 = first(kk) - 1
    Ax = g.dy * g.dz; Ay = g.dx * g.dz; Az = g.dx * g.dy
    # the diffusive part: ν + s·½(κ_in + κ_out) on the open scalar faces
    dν, Kh, Kv, ks, Dc, Df = _diffusive_parts(diffusivity, Wc, Wf)
    # the mass-flux correction of each open velocity face: (δ, corrected?(a, b, u_n))
    cW, cE = _correction_of(vb.west, g), _correction_of(vb.east, g)
    cS, cN = _correction_of(vb.south, g), _correction_of(vb.north, g)
    cT = _correction_of(vb.top, g)
    dW_ = diffusivity !== nothing && bcs.west isa AbstractOpenBoundary
    dE_ = diffusivity !== nothing && bcs.east isa AbstractOpenBoundary
    dS_ = diffusivity !== nothing && bcs.south isa AbstractOpenBoundary
    dN_ = diffusivity !== nothing && bcs.north isa AbstractOpenBoundary
    dB_ = diffusivity !== nothing && bcs.bottom isa AbstractOpenBoundary
    dT_ = diffusivity !== nothing && bcs.top isa AbstractOpenBoundary
    fW = fE = fS = fN = fB = fT = zero(T)
    @inline addl!(v, k, x) = (v === nothing || (@inbounds v[k - k0] += x); nothing)
    # every face flux is _bsf_face_flux (scalar_face_flux, diffusive part only
    # where the scalar face is open): a top-level function with its arrays as
    # arguments, not a captured closure, which did not infer and allocated
    # on every face (#327)
    if !(vb.west isa Periodic)
        iw = first(ii); ie = last(ii) + 1
        @inbounds for k in kk, (b, j) in enumerate(jj)
            w = _wflux(Wc, k) * Ax; wd_ = _wflux(Dc, k) * Ax
            Fa_w, Fd_w, φw = _bsf_face_flux(r, ϕd, ud, 1, iw, j, k, dν, Kh, ks, g, dW_, Fxd)
            Fa_e, Fd_e, φe = _bsf_face_flux(r, ϕd, ud, 1, ie, j, k, dν, Kh, ks, g, dE_, Fxd)
            aw = -(w * Fa_w + wd_ * Fd_w); ae = w * Fa_e + wd_ * Fd_e
            fW += aw; fE += ae
            addl!(levels, k, aw + ae)
            if correction_levels !== nothing
                _corrected(cW, b, k - k0, -ud[iw, j, k]) && addl!(correction_levels, k, -cW[1] * w * φw)
                _corrected(cE, b, k - k0, ud[ie, j, k]) && addl!(correction_levels, k, -cE[1] * w * φe)
            end
        end
    end
    if !(vb.south isa Periodic)
        js = first(jj); jn = last(jj) + 1
        @inbounds for k in kk, (a, i) in enumerate(ii)
            w = _wflux(Wc, k) * Ay; wd_ = _wflux(Dc, k) * Ay
            Fa_s, Fd_s, φs = _bsf_face_flux(r, ϕd, vd, 2, i, js, k, dν, Kh, ks, g, dS_, Fyd)
            Fa_n, Fd_n, φn = _bsf_face_flux(r, ϕd, vd, 2, i, jn, k, dν, Kh, ks, g, dN_, Fyd)
            as = -(w * Fa_s + wd_ * Fd_s); an = w * Fa_n + wd_ * Fd_n
            fS += as; fN += an
            addl!(levels, k, as + an)
            if correction_levels !== nothing
                _corrected(cS, a, k - k0, -vd[i, js, k]) && addl!(correction_levels, k, -cS[1] * w * φs)
                _corrected(cN, a, k - k0, vd[i, jn, k]) && addl!(correction_levels, k, -cN[1] * w * φn)
            end
        end
    end
    if !(vb.bottom isa Periodic)
        kb = first(kk); kt = last(kk) + 1
        wb_ = _wflux(Wf, kb) * Az; wt_ = _wflux(Wf, kt) * Az
        db_ = _wflux(Df, kb) * Az; dt_ = _wflux(Df, kt) * Az
        @inbounds for (b, j) in enumerate(jj), (a, i) in enumerate(ii)
            Fa_b, Fd_b, _ = _bsf_face_flux(r, ϕd, wd, 3, i, j, kb, dν, Kv, ks, g, dB_, Fzd)
            Fa_t, Fd_t, φt = _bsf_face_flux(r, ϕd, wd, 3, i, j, kt, dν, Kv, ks, g, dT_, Fzd)
            ab = -(wb_ * Fa_b + db_ * Fd_b)
            at = wt_ * Fa_t + dt_ * Fd_t
            fB += ab; fT += at
            addl!(levels, kb, ab); addl!(levels, kt - 1, at)
            if correction_levels !== nothing
                _corrected(cT, a, b, wd[i, j, kt]) && addl!(correction_levels, kt - 1, -cT[1] * wt_ * φt)
            end
        end
    end
    return (west = fW, east = fE, south = fS, north = fN, bottom = fB, top = fT,
            total = fW + fE + fS + fN + fB + fT)
end

end # module
