# KA-l (#304): the SGS-closure loops as they were before the port (the
# issue-299 branch merged with master at c1f6693): the dynamic-Smagorinsky
# passes of src/Operators.jl, the box filter of operators/filters.jl, the
# Lagrangian closure's loops and the plane-by-plane second test filter of
# operators/lagrangian_sgs.jl, the AMD / Vreman / σ kernels, the
# stability correction, the N² / Ri / S² stencils, the TKE closure's dual
# strain (the scatter and halo fold), coefficients and update, and the
# Reynolds-stress closure's coefficients, edge stresses and update (with
# its shared 6 × 6 scratch matrices), transcribed verbatim with the arrays
# as arguments. The point functions they call (`_strain_at`,
# `_germano_terms`, `_amd_point`, `sigma_operator`, `tke_length_scale`, …)
# are unchanged by the port and are taken from the package. The reference
# of test/ka_sgs.jl and of the
# KA-l timings (old loops vs launches). Nothing here is used by the package.
# ---------------------------------------------------------------------------
module LegacySGS
using Aeolus
using Aeolus.Grids: RegularGrid, interior
using Aeolus.Fields: Field, mesh
using Aeolus.BoundaryConditions: Periodic
using Aeolus.Operators: _strain_at, _germano_terms, _germano_LM_MM, _box_stencil_sum,
                        _upstream_n, _relax_pairs!, _relax_memories, _βg,
                        _velocity_gradient_at, _scalar_gradient_at, _amd_point, _amd_scalar_point,
                        _vreman_point, sigma_operator, _n2_cell, _θref, _humidity_data
using Aeolus.TurbulenceStability: stability_factor_m, stability_factor_h, gradient_richardson
using Aeolus.PrognosticSGS: tke_length_scale, tke_dissipation_coefficient, tke_eddy_coefficients,
                            tke_equilibrium, _solid, _lwall, _first, _tke_volume0, _surface!,
                            _τat, _tke, _rsm_local, _dual_gradient, _θv_gradient, _Edges, _oxy, _oxz,
                            _oyz, rotate_stress, is_realizable, coriolis_rotation, reynolds_stress_rhs,
                            _qv, _β, _rsm_Ω, _UNIT6, _images
using Aeolus.Operators: bounded_directions

# ===================== src/Operators.jl =====================================

# _masked_velocities!: the velocity with 0 on every solid face, halos included
function masked_velocities!(ub, vb, wb, fx, fy, fz, ud, vd, wd)
    @inbounds for I in eachindex(ud)
        ub[I] = fx[I] ? zero(eltype(ub)) : ud[I]
        vb[I] = fy[I] ? zero(eltype(vb)) : vd[I]
        wb[I] = fz[I] ? zero(eltype(wb)) : wd[I]
    end
    return ub, vb, wb
end

# Halos of a storage-sized cell mask: periodic wrap, or the nearest interior layer.
function fill_mask_halos!(m, g, periodic::NTuple{3,Bool})
    N = (g.Nx, g.Ny, g.Nz); H = (g.Hx, g.Hy, g.Hz)
    @inbounds for d in 1:3
        n, h = N[d], H[d]
        for I in CartesianIndices(m)
            l = I[d]
            (l <= h || l > h + n) || continue
            s = periodic[d] ? h + mod(l - h - 1, n) + 1 : clamp(l, h + 1, h + n)
            J = Base.setindex(Tuple(I), s, d)
            m[I] = m[J...]
        end
    end
    return m
end

function germano_pass1!(ws, ud, vd, wd, g)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    Uc, Vc, Wc, magS = ws.Uc, ws.Vc, ws.Wc, ws.magS
    S11, S22, S33, S12, S13, S23 = ws.S11, ws.S22, ws.S33, ws.S12, ws.S13, ws.S23
    @inbounds for lk in 1:Nz+2, lj in 1:Ny+2, li in 1:Nx+2
        i = Hx - 1 + li; j = Hy - 1 + lj; k = Hz - 1 + lk
        Uc[li,lj,lk] = 0.5 * (ud[i,j,k] + ud[i+1,j,k])
        Vc[li,lj,lk] = 0.5 * (vd[i,j,k] + vd[i,j+1,k])
        Wc[li,lj,lk] = 0.5 * (wd[i,j,k] + wd[i,j,k+1])
        s11, s22, s33, s12, s13, s23, ms = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)
        S11[li,lj,lk] = s11; S22[li,lj,lk] = s22; S33[li,lj,lk] = s33
        S12[li,lj,lk] = s12; S13[li,lj,lk] = s13; S23[li,lj,lk] = s23
        magS[li,lj,lk] = ms
    end
    return nothing
end

function strain_magnitude!(ws, ud, vd, wd, g)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        ws.magS[i+1,j+1,k+1] = _strain_at(ud, vd, wd, Hx + i, Hy + j, Hz + k, idx, idy, idz)[7]
    end
    return nothing
end

function germano_pass2!(ws, g, α2, ::Nothing)
    T = eltype(ws.LM)
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    inv27 = one(T) / 27
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        t = _box_stencil_sum((a, b, c) -> _germano_terms(ws, a, b, c), i + 1, j + 1, k + 1, 1, 1, 1)
        ws.LM[i,j,k], ws.MM[i,j,k] = _germano_LM_MM(t, inv27, Δ2, α2)
    end
    return nothing
end

function germano_pass2!(ws, g, α2, cell)
    T = eltype(ws.LM)
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    oH = (g.Hx - 1, g.Hy - 1, g.Hz - 1)                 # shell → data index
    z21 = ntuple(_ -> zero(T), Val(21))
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        if cell[g.Hx + i, g.Hy + j, g.Hz + k]
            ws.LM[i,j,k] = zero(T); ws.MM[i,j,k] = zero(T)
            continue
        end
        s = _box_stencil_sum(i + 1, j + 1, k + 1, 1, 1, 1) do a, b, c
            t = _germano_terms(ws, a, b, c)
            @inbounds cell[oH[1] + a, oH[2] + b, oH[3] + c] ? z21 : t
        end
        (n,) = _box_stencil_sum(i + 1, j + 1, k + 1, 1, 1, 1) do a, b, c
            @inbounds (cell[oH[1] + a, oH[2] + b, oH[3] + c] ? 0 : 1,)
        end
        ws.LM[i,j,k], ws.MM[i,j,k] = _germano_LM_MM(s, one(T) / n, Δ2, α2)
    end
    return nothing
end

# `mask`: nothing, or (cell, fx, fy, fz, ub, vb, wb) — the old _SGSMask
function germano_identity!(ws, ud, vd, wd, g, α2, mask = nothing)
    if mask === nothing
        ub, vb, wb = ud, vd, wd
    else
        ub, vb, wb = masked_velocities!(mask[5], mask[6], mask[7], mask[2], mask[3], mask[4], ud, vd, wd)
    end
    germano_pass1!(ws, ub, vb, wb, g)
    germano_pass2!(ws, g, α2, mask === nothing ? nothing : mask[1])
    return nothing
end

function dynamic_νt!(nd, ws, g, cell)
    T = eltype(nd)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    cs2, magS = ws.cs2, ws.magS
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        solid = cell !== nothing && cell[Hx+i, Hy+j, Hz+k]
        nd[Hx+i, Hy+j, Hz+k] = solid ? zero(T) : cs2[i,j,k] * Δ2 * magS[i+1, j+1, k+1]
    end
    return nothing
end

function dynamic_smagorinsky!(nd, ud, vd, wd, ws, g, plane::Bool, α2, cs2max, mask = nothing)
    T = eltype(nd)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    germano_identity!(ws, ud, vd, wd, g, α2, mask)
    cell = mask === nothing ? nothing : mask[1]
    cs2 = ws.cs2
    if plane
        @inbounds for k in 1:Nz
            lm = zero(T); mm = zero(T)
            for j in 1:Ny, i in 1:Nx
                cell !== nothing && cell[g.Hx+i, g.Hy+j, g.Hz+k] && continue
                lm += ws.LM[i,j,k]; mm += ws.MM[i,j,k]
            end
            c = mm > 0 ? clamp(lm / mm, zero(T), T(cs2max)) : zero(T)
            for j in 1:Ny, i in 1:Nx
                cs2[i,j,k] = cell !== nothing && cell[g.Hx+i, g.Hy+j, g.Hz+k] ? zero(T) : c
            end
        end
    else
        @inbounds for I in eachindex(cs2)
            mm = ws.MM[I]
            cs2[I] = mm > 0 ? clamp(ws.LM[I] / mm, zero(T), T(cs2max)) : zero(T)
        end
    end
    dynamic_νt!(nd, ws, g, cell)
    return nothing
end

# ===================== src/operators/filters.jl ==============================

function box_filter!(od, fd::AbstractArray{T}, g, rx, ry, rz, ::Nothing) where {T}
    inv_n = one(T) / ((2rx + 1) * (2ry + 1) * (2rz + 1))
    @inbounds for k in g.Hz+1:g.Hz+g.Nz, j in g.Hy+1:g.Hy+g.Ny, i in g.Hx+1:g.Hx+g.Nx
        (s,) = _box_stencil_sum((a, b, q) -> (@inbounds(fd[a,b,q]),), i, j, k, rx, ry, rz)
        od[i,j,k] = s * inv_n
    end
    return nothing
end

function box_filter!(od, fd::AbstractArray{T}, g, rx, ry, rz, mask) where {T}
    @inbounds for k in g.Hz+1:g.Hz+g.Nz, j in g.Hy+1:g.Hy+g.Ny, i in g.Hx+1:g.Hx+g.Nx
        if mask[i,j,k]
            od[i,j,k] = zero(T)
            continue
        end
        s, c = _box_stencil_sum((a, b, q) -> @inbounds(mask[a,b,q] ? (zero(T), 0) : (fd[a,b,q], 1)),
                                i, j, k, rx, ry, rz)
        od[i,j,k] = c > 0 ? s / c : zero(T)
    end
    return nothing
end

# ===================== src/operators/lagrangian_sgs.jl =======================

# the coefficient of the Lagrangian closure from its memories (sgs_viscosity!)
function lagrangian_cs2!(cs2, ILM, IMM, cs2max, cell, g)
    T = eltype(cs2)
    @inbounds for I in eachindex(cs2)
        mm = IMM[I]
        cs2[I] = mm > 0 ? clamp(ILM[I] / mm, zero(T), cs2max) : zero(T)
    end
    if cell !== nothing
        @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            cell[g.Hx+i, g.Hy+j, g.Hz+k] && (cs2[i,j,k] = zero(T))
        end
    end
    return nothing
end

function scale_dependent_cs2!(cs2, beta, c, cs2max)
    T = eltype(cs2)
    lnα = log(c.filter_ratio)
    inv_lr = one(T) / log2(c.filter_ratio2 / c.filter_ratio)     # φ per log₂ of the ratio
    φmin, φmax = log2(c.beta_min), log2(c.beta_max)
    φfix = c.beta_fixed === nothing ? zero(T) : log2(c.beta_fixed)
    @inbounds for I in eachindex(cs2)
        mm = c.I_MM[I]
        ca = mm > 0 ? c.I_LM[I] / mm : zero(T)              # C²(αΔ)
        if !(ca > 0)
            cs2[I] = zero(T); beta[I] = one(T)
            continue
        end
        φ = if c.beta_fixed !== nothing
            φfix
        else
            nn = c.I_NN[I]
            cb = nn > 0 ? c.I_QN[I] / nn : zero(T)          # C²(α₂Δ)
            cb > 0 ? clamp(log2(cb / ca) * inv_lr, φmin, φmax) : φmin
        end
        beta[I] = clamp(exp2(φ), c.beta_min, c.beta_max)    # β = 2^φ
        cs2[I] = clamp(ca * exp(-φ * lnα), zero(T), cs2max) # C²(αΔ) α^(-φ)
    end
    return nothing
end

function cell_velocity!(U, V, W, ud, vd, wd, g)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        U[i,j,k] = 0.5 * (ud[I,J,K] + ud[I+1,J,K])
        V[i,j,k] = 0.5 * (vd[I,J,K] + vd[I,J+1,K])
        W[i,j,k] = 0.5 * (wd[I,J,K] + wd[I,J,K+1])
    end
    return nothing
end

function lagrangian_relax!(pairs::Tuple, U, V, W, dt, g, θ, memory_time, cell, per, dtε)
    T = eltype(first(first(pairs)))
    Δ = (g.dx * g.dy * g.dz)^(1/3)
    cx = dt / g.dx; cy = dt / g.dy; cz = dt / g.dz
    εfix = memory_time === nothing ? zero(T) : (dtε / memory_time) / (1 + dtε / memory_time)
    mems = _relax_memories(pairs)
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        if cell !== nothing && cell[g.Hx+i, g.Hy+j, g.Hz+k]
            foreach(p -> (p[1][i,j,k] = zero(T); p[2][i,j,k] = zero(T)), pairs)
            continue
        end
        up = _upstream_n(mems, i - U[i,j,k] * cx, j - V[i,j,k] * cy, k - W[i,j,k] * cz,
                         i, j, k, g, per, cell)
        _relax_pairs!(pairs, up, 1, i, j, k, θ, Δ, εfix, memory_time, dtε)
    end
    return nothing
end

# the plane-by-plane second test filter and its buffers
struct SecondTestFilter{T}
    Ap::Array{T,3}
    Bp::Array{T,3}
    C::Array{T,4}
    Ep::Array{T,3}
    QN::Array{T,3}
    NN::Array{T,3}
end

function SecondTestFilter(g::RegularGrid{T}) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    return SecondTestFilter{T}(zeros(T, 22, Nx + 2, Ny + 2), zeros(T, 22, Nx, Ny + 2),
                               zeros(T, 22, Nx, Ny, Nz + 2), zeros(T, 22, Nx, Ny),
                               zeros(T, Nx, Ny, Nz), zeros(T, Nx, Ny, Nz))
end

function box5_stencil(N, per)
    return [ntuple(m -> begin
                       s = i + m - 3
                       per ? mod1(s, N) + 1 : (1 <= s <= N ? s + 1 : 0)
                   end, 5) for i in 1:N]
end

function box5_plane_pass!(dst::AbstractArray{T,3}, src::AbstractArray{T,3}, st, dim) where {T}
    D = size(dst, 1)
    @inbounds if dim == 2
        for c2 in axes(dst, 3), i in axes(dst, 2)
            first = true
            for s in st[i]
                s == 0 && continue
                if first
                    @simd for c in 1:D; dst[c, i, c2] = src[c, s, c2]; end
                    first = false
                else
                    @simd for c in 1:D; dst[c, i, c2] += src[c, s, c2]; end
                end
            end
        end
    else
        for j in axes(dst, 3)
            first = true
            for s in st[j]
                s == 0 && continue
                for c1 in axes(dst, 2)
                    if first
                        @simd for c in 1:D; dst[c, c1, j] = src[c, c1, s]; end
                    else
                        @simd for c in 1:D; dst[c, c1, j] += src[c, c1, s]; end
                    end
                end
                first = false
            end
        end
    end
    return dst
end

function second_test_filter!(sf::SecondTestFilter{T}, ws, g, α2sq, cell, per) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    oH = (g.Hx - 1, g.Hy - 1, g.Hz - 1)             # shell → data index
    stx = box5_stencil(Nx, per[1]); sty = box5_stencil(Ny, per[2])
    stz = box5_stencil(Nz, per[3])
    Ap, Bp, C, Ep = sf.Ap, sf.Bp, sf.C, sf.Ep
    @inbounds for c in 1:Nz+2
        for b in 1:Ny+2, a in 1:Nx+2
            fluid = cell === nothing || !cell[oH[1] + a, oH[2] + b, oH[3] + c]
            t = _germano_terms(ws, a, b, c)
            for q in 1:21
                Ap[q, a, b] = fluid ? t[q] : zero(T)
            end
            Ap[22, a, b] = fluid ? one(T) : zero(T)
        end
        box5_plane_pass!(Bp, Ap, stx, 2)
        box5_plane_pass!(view(C, :, :, :, c), Bp, sty, 3)
    end
    Δ2 = (g.dx * g.dy * g.dz)^(2/3)
    D = 22
    @inbounds for k in 1:Nz
        first = true
        for s in stz[k]
            s == 0 && continue
            for j in 1:Ny, i in 1:Nx
                if first
                    @simd for q in 1:D; Ep[q, i, j] = C[q, i, j, s]; end
                else
                    @simd for q in 1:D; Ep[q, i, j] += C[q, i, j, s]; end
                end
            end
            first = false
        end
        for j in 1:Ny, i in 1:Nx
            n = Ep[22, i, j]
            if (cell !== nothing && cell[g.Hx + i, g.Hy + j, g.Hz + k]) || !(n > 0)
                sf.QN[i,j,k] = zero(T); sf.NN[i,j,k] = zero(T)
                continue
            end
            t = ntuple(q -> Ep[q, i, j], Val(21))
            sf.QN[i,j,k], sf.NN[i,j,k] = _germano_LM_MM(t, one(T) / n, Δ2, α2sq)
        end
    end
    return nothing
end

# ===================== src/operators/sgs_amd_vreman.jl / sigma_closure.jl ===

function amd_kernel!(nd, kd, ud, vd, wd, θd, g, C, Pr, buoy)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    d2 = (g.dx^2, g.dy^2, g.dz^2)
    @inbounds for kk in 1:g.Nz
        βg = float(_βg(buoy, kk))
        k = Hz + kk
        for jj in 1:g.Ny, ii in 1:g.Nx
            i = Hx + ii; j = Hy + jj
            G = _velocity_gradient_at(ud, vd, wd, i, j, k, idx, idy, idz)
            gθ = _scalar_gradient_at(θd, i, j, k, idx, idy, idz)
            νn, νd_, κn, κd_ = _amd_point(G, gθ, d2, βg)
            ν = νd_ > 0 ? C * max(zero(νn), νn) / νd_ : zero(νn)
            nd[i,j,k] = ν
            kd[i,j,k] = κd_ > 0 ? C * max(zero(κn), κn) / κd_ : ν / Pr
        end
    end
    return nothing
end

function amd_scalar_kernel!(kd, nd, ud, vd, wd, ϕd, g, C, Pr)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    d2 = (g.dx^2, g.dy^2, g.dz^2)
    @inbounds for kk in 1:g.Nz, jj in 1:g.Ny, ii in 1:g.Nx
        i = Hx + ii; j = Hy + jj; k = Hz + kk
        G = _velocity_gradient_at(ud, vd, wd, i, j, k, idx, idy, idz)
        gϕ = _scalar_gradient_at(ϕd, i, j, k, idx, idy, idz)
        κn, κd_ = _amd_scalar_point(G, gϕ, d2)
        kd[i,j,k] = κd_ > 0 ? C * max(zero(κn), κn) / κd_ : nd[i,j,k] / Pr
    end
    return nothing
end

function vreman_kernel!(nd, ud, vd, wd, g, c)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    d2 = (g.dx^2, g.dy^2, g.dz^2)
    @inbounds for kk in 1:g.Nz, jj in 1:g.Ny, ii in 1:g.Nx
        i = Hx + ii; j = Hy + jj; k = Hz + kk
        G = _velocity_gradient_at(ud, vd, wd, i, j, k, idx, idy, idz)
        Bβ, aa = _vreman_point(G, d2)
        nd[i,j,k] = aa > 0 ? c * sqrt(Bβ / aa) : zero(Bβ)
    end
    return nothing
end

function sigma_νt!(nd, ud, vd, wd, C_σ, g)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    Δ = (g.dx * g.dy * g.dz)^(1 / 3)
    coef = (C_σ * Δ)^2
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        G = _velocity_gradient_at(ud, vd, wd, i, j, k, idx, idy, idz)
        nd[i, j, k] = coef * sigma_operator(G)
    end
    return nothing
end

# ===================== sgs_stability.jl / stratification.jl / sgs_closures.jl

function stability_correct!(νh, νv, κh, κv, av, Ri, scheme, ℓh, ℓv, Pr, dynamic::Bool, g)
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        nb = νh[i, j, k]
        r  = Ri[i, j, k]
        fm = stability_factor_m(scheme, r)
        fh = stability_factor_h(scheme, r)
        gm, gh = dynamic ? (one(fm), fm > 0 ? fh / fm : zero(fh)) : (fm, fh)
        h = nb * ℓh
        νh[i, j, k] = h
        νv[i, j, k] = nb * ℓv * gm
        a = nb * ℓv * gh
        av[i, j, k] = a
        κh[i, j, k] = h / Pr
        κv[i, j, k] = a / Pr
    end
    return nothing
end

function n2!(nd, θd, qv, θ₀, g, m)
    qd = _humidity_data(qv)
    dz = m.dz
    ii, jj, kk = interior(m)
    @inbounds for k in kk, j in jj, i in ii
        nd[i, j, k] = _n2_cell(θd, qd, i, j, k, dz, g, _θref(θ₀, k))
    end
    return nd
end

function ri!(rd, nd, ud, vd, wd, S2_min, m)
    idx = 1 / m.dx; idy = 1 / m.dy; idz = 1 / m.dz
    ii, jj, kk = interior(m)
    @inbounds for k in kk, j in jj, i in ii
        magS = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)[7]
        rd[i, j, k] = gradient_richardson(nd[i, j, k], magS^2; S2_min = S2_min)
    end
    return rd
end

function s2!(sd, ud, vd, wd, m)
    idx = 1 / m.dx; idy = 1 / m.dy; idz = 1 / m.dz
    ii, jj, kk = interior(m)
    @inbounds for k in kk, j in jj, i in ii
        sd[i, j, k] = _strain_at(ud, vd, wd, i, j, k, idx, idy, idz)[7]^2
    end
    return sd
end

function tke_from_viscosity!(ed, νd, c_k, g)
    Δ = (g.dx * g.dy * g.dz)^(1 / 3)
    s = 1 / (c_k * Δ)
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = g.Hx + i, g.Hy + j, g.Hz + k
        ed[I, J, K] = (max(νd[I, J, K], 0) * s)^2
    end
    return ed
end

# ===================== src/PrognosticSGS.jl ==================================

function dual_strain!(d, ud, vd, wd, bcs, g::RegularGrid)
    T = eltype(d)
    idx2 = 1 / g.dx^2; idy2 = 1 / g.dy^2; idz2 = 1 / g.dz^2
    fill!(d, zero(T))
    ii, jj, kk = interior(g)
    q4(x) = T(0.25) * x
    @inbounds for k in kk, j in jj, i in ii
        up = ud[i, j, k]
        a = -up * (ud[i+1, j, k] - ud[i, j, k]) * idx2
        d[i, j, k] += a
        a = up * (ud[i, j, k] - ud[i-1, j, k]) * idx2
        d[i-1, j, k] += a
        a = q4(-up * (ud[i, j+1, k] - ud[i, j, k]) * idy2)
        d[i-1, j, k] += a; d[i, j, k] += a; d[i-1, j+1, k] += a; d[i, j+1, k] += a
        a = q4(up * (ud[i, j, k] - ud[i, j-1, k]) * idy2)
        d[i-1, j-1, k] += a; d[i, j-1, k] += a; d[i-1, j, k] += a; d[i, j, k] += a
        a = q4(-up * (ud[i, j, k+1] - ud[i, j, k]) * idz2)
        d[i-1, j, k] += a; d[i, j, k] += a; d[i-1, j, k+1] += a; d[i, j, k+1] += a
        a = q4(up * (ud[i, j, k] - ud[i, j, k-1]) * idz2)
        d[i-1, j, k-1] += a; d[i, j, k-1] += a; d[i-1, j, k] += a; d[i, j, k] += a

        vp = vd[i, j, k]
        a = -vp * (vd[i, j+1, k] - vd[i, j, k]) * idy2
        d[i, j, k] += a
        a = vp * (vd[i, j, k] - vd[i, j-1, k]) * idy2
        d[i, j-1, k] += a
        a = q4(-vp * (vd[i+1, j, k] - vd[i, j, k]) * idx2)
        d[i, j-1, k] += a; d[i+1, j-1, k] += a; d[i, j, k] += a; d[i+1, j, k] += a
        a = q4(vp * (vd[i, j, k] - vd[i-1, j, k]) * idx2)
        d[i-1, j-1, k] += a; d[i, j-1, k] += a; d[i-1, j, k] += a; d[i, j, k] += a
        a = q4(-vp * (vd[i, j, k+1] - vd[i, j, k]) * idz2)
        d[i, j-1, k] += a; d[i, j, k] += a; d[i, j-1, k+1] += a; d[i, j, k+1] += a
        a = q4(vp * (vd[i, j, k] - vd[i, j, k-1]) * idz2)
        d[i, j-1, k-1] += a; d[i, j, k-1] += a; d[i, j-1, k] += a; d[i, j, k] += a

        wp = wd[i, j, k]
        a = -wp * (wd[i, j, k+1] - wd[i, j, k]) * idz2
        d[i, j, k] += a
        a = wp * (wd[i, j, k] - wd[i, j, k-1]) * idz2
        d[i, j, k-1] += a
        a = q4(-wp * (wd[i+1, j, k] - wd[i, j, k]) * idx2)
        d[i, j, k-1] += a; d[i+1, j, k-1] += a; d[i, j, k] += a; d[i+1, j, k] += a
        a = q4(wp * (wd[i, j, k] - wd[i-1, j, k]) * idx2)
        d[i-1, j, k-1] += a; d[i, j, k-1] += a; d[i-1, j, k] += a; d[i, j, k] += a
        a = q4(-wp * (wd[i, j+1, k] - wd[i, j, k]) * idy2)
        d[i, j, k-1] += a; d[i, j+1, k-1] += a; d[i, j, k] += a; d[i, j+1, k] += a
        a = q4(wp * (wd[i, j, k] - wd[i, j-1, k]) * idy2)
        d[i, j-1, k-1] += a; d[i, j, k-1] += a; d[i, j-1, k] += a; d[i, j, k] += a
    end
    fold_halos!(d, bcs, g)
    return d
end

function dual_strain!(d, ud, vd, wd, bcs, g)      # StretchedGrid
    T = eltype(d)
    idx2 = 1 / g.dx^2; idy2 = 1 / g.dy^2
    dzc, dzf = g.dzc, g.dzf
    fill!(d, zero(T))
    ii, jj, kk = interior(g)
    q4(x) = T(0.25) * x
    @inbounds for k in kk
        hc = dzc[k]
        hf = dzf[k]
        izt = 1 / dzf[k+1]; izb = 1 / dzf[k]
        iwt = 1 / dzc[k];   iwb = 1 / dzc[k-1]
        for j in jj, i in ii
            up = ud[i, j, k]
            a = -up * (ud[i+1, j, k] - ud[i, j, k]) * idx2 * hc
            d[i, j, k] += a
            a = up * (ud[i, j, k] - ud[i-1, j, k]) * idx2 * hc
            d[i-1, j, k] += a
            a = q4(-up * (ud[i, j+1, k] - ud[i, j, k]) * idy2 * hc)
            d[i-1, j, k] += a; d[i, j, k] += a; d[i-1, j+1, k] += a; d[i, j+1, k] += a
            a = q4(up * (ud[i, j, k] - ud[i, j-1, k]) * idy2 * hc)
            d[i-1, j-1, k] += a; d[i, j-1, k] += a; d[i-1, j, k] += a; d[i, j, k] += a
            a = q4(-up * (ud[i, j, k+1] - ud[i, j, k]) * izt)
            d[i-1, j, k] += a; d[i, j, k] += a; d[i-1, j, k+1] += a; d[i, j, k+1] += a
            a = q4(up * (ud[i, j, k] - ud[i, j, k-1]) * izb)
            d[i-1, j, k-1] += a; d[i, j, k-1] += a; d[i-1, j, k] += a; d[i, j, k] += a

            vp = vd[i, j, k]
            a = -vp * (vd[i, j+1, k] - vd[i, j, k]) * idy2 * hc
            d[i, j, k] += a
            a = vp * (vd[i, j, k] - vd[i, j-1, k]) * idy2 * hc
            d[i, j-1, k] += a
            a = q4(-vp * (vd[i+1, j, k] - vd[i, j, k]) * idx2 * hc)
            d[i, j-1, k] += a; d[i+1, j-1, k] += a; d[i, j, k] += a; d[i+1, j, k] += a
            a = q4(vp * (vd[i, j, k] - vd[i-1, j, k]) * idx2 * hc)
            d[i-1, j-1, k] += a; d[i, j-1, k] += a; d[i-1, j, k] += a; d[i, j, k] += a
            a = q4(-vp * (vd[i, j, k+1] - vd[i, j, k]) * izt)
            d[i, j-1, k] += a; d[i, j, k] += a; d[i, j-1, k+1] += a; d[i, j, k+1] += a
            a = q4(vp * (vd[i, j, k] - vd[i, j, k-1]) * izb)
            d[i, j-1, k-1] += a; d[i, j, k-1] += a; d[i, j-1, k] += a; d[i, j, k] += a

            wp = wd[i, j, k]
            a = -wp * (wd[i, j, k+1] - wd[i, j, k]) * iwt
            d[i, j, k] += a
            a = wp * (wd[i, j, k] - wd[i, j, k-1]) * iwb
            d[i, j, k-1] += a
            a = q4(-wp * (wd[i+1, j, k] - wd[i, j, k]) * idx2 * hf)
            d[i, j, k-1] += a; d[i+1, j, k-1] += a; d[i, j, k] += a; d[i+1, j, k] += a
            a = q4(wp * (wd[i, j, k] - wd[i-1, j, k]) * idx2 * hf)
            d[i-1, j, k-1] += a; d[i, j, k-1] += a; d[i-1, j, k] += a; d[i, j, k] += a
            a = q4(-wp * (wd[i, j+1, k] - wd[i, j, k]) * idy2 * hf)
            d[i, j, k-1] += a; d[i, j+1, k-1] += a; d[i, j, k] += a; d[i, j+1, k] += a
            a = q4(wp * (wd[i, j, k] - wd[i, j-1, k]) * idy2 * hf)
            d[i, j-1, k-1] += a; d[i, j, k-1] += a; d[i, j-1, k] += a; d[i, j, k] += a
        end
    end
    fold_halos!(d, bcs, g)
    @inbounds for k in kk
        ih = 1 / dzc[k]
        for j in jj, i in ii
            d[i, j, k] *= ih
        end
    end
    return d
end

function fold_halos!(d, bcs, g)
    Hx, Hy, Hz, Nx, Ny, Nz = g.Hx, g.Hy, g.Hz, g.Nx, g.Ny, g.Nz
    sx, sy, sz = size(d)
    lo, hi = _images(bcs.west, Hx, Nx)
    @inbounds for k in 1:sz, j in 1:sy
        d[lo, j, k] += d[Hx, j, k];        d[Hx, j, k] = 0
        d[hi, j, k] += d[Hx+Nx+1, j, k];   d[Hx+Nx+1, j, k] = 0
    end
    lo, hi = _images(bcs.south, Hy, Ny)
    @inbounds for k in 1:sz, i in 1:sx
        d[i, lo, k] += d[i, Hy, k];        d[i, Hy, k] = 0
        d[i, hi, k] += d[i, Hy+Ny+1, k];   d[i, Hy+Ny+1, k] = 0
    end
    lo, hi = _images(bcs.bottom, Hz, Nz)
    @inbounds for j in 1:sy, i in 1:sx
        d[i, j, lo] += d[i, j, Hz];        d[i, j, Hz] = 0
        d[i, j, hi] += d[i, j, Hz+Nz+1];   d[i, j, Hz+Nz+1] = 0
    end
    return d
end

function tke_coefficients!(nd, kd, c, n2)
    T = eltype(nd)
    g = mesh(c.e)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    ed = c.e.data
    ck, cl = c.c_k, c.c_l
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        Δ = c.Δk[k]
        if _solid(c.mask, I, J, K)
            nd[I, J, K] = zero(T); kd[I, J, K] = zero(T)
            continue
        end
        e = ed[I, J, K]
        l, _ = tke_length_scale(e, n2[I, J, K], Δ; l_wall = _lwall(c.l_wall, i, j, k, T), c_l = cl)
        νv, κv = tke_eddy_coefficients(e, l, Δ; c_k = ck)
        nd[I, J, K] = νv
        kd[I, J, K] = κv
    end
    return nothing
end

# the TKE update of c.e and c.stats (the closure's scratch ν, N2, D, tend, Pw
# must be filled as advance_sgs! leaves them)
function tke_update!(c, Δt, pbs)
    T = eltype(c.e.data)
    g = mesh(c.e)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    V0 = T(_tke_volume0(g))
    ed, nd, n2, dd, td, pw = c.e.data, c.ν.data, c.N2.data, c.D.data, c.tend.data, c.Pw.data
    ck, cl = c.c_k, c.c_l
    sPs = zero(T); sPb = zero(T); sε = zero(T); sB = zero(T); sT = zero(T); sPw = zero(T)
    nclip = zero(T); eclip = zero(T)
    exact = c.dissipation === :exact
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        if _solid(c.mask, I, J, K)
            ed[I, J, K] = zero(T)
            continue
        end
        Δ = c.Δk[k]; vr = c.vrel[k]; V = V0 * vr
        e = ed[I, J, K]
        N2 = n2[I, J, K]
        l, s = tke_length_scale(e, N2, Δ; l_wall = _lwall(c.l_wall, i, j, k, T), c_l = cl)
        fac = _first(c.first_level, i, j, k) ? c.surface_factor : one(T)
        cε = tke_dissipation_coefficient(l, Δ; c_ε0 = c.c_ε0, c_ε1 = c.c_ε1, factor = fac)
        Ps = nd[I, J, K] * dd[I, J, K] + pw[I, J, K]
        κv = (1 + 2l / Δ) * nd[I, J, K]
        Psf = (k == 1 ? pbs[1] : zero(T)) + (k == g.Nz ? pbs[2] : zero(T))
        Pb = -κv * N2 + Psf
        src = Ps + (N2 < 0 ? -κv * N2 : zero(T)) + max(Psf, zero(T))
        rb = (N2 > 0 && s > 0 ? (1 + 2l / Δ) * ck * N2 / s : zero(T)) +
             (Psf < 0 ? -Psf / max(e, floatmin(T)) : zero(T))
        tr = td[I, J, K]
        estar = e + Δt * (tr + src)
        if estar < 0
            nclip += 1; eclip -= estar * V
            estar = zero(T)
        end
        rε = cε * s
        if exact && l > 0 && estar > 0
            e1 = inv(inv(sqrt(estar)) + cε * Δt / (2l))^2
            enew = e1 / (1 + Δt * rb)
            dε = (estar - e1) / Δt
            dB = (e1 - enew) / Δt
        else
            R = rε + rb
            enew = estar / (1 + Δt * R)
            rem = (estar - enew) / Δt
            dε = R > 0 ? rem * (rε / R) : zero(T)
            dB = R > 0 ? rem * (rb / R) : zero(T)
        end
        ed[I, J, K] = enew
        sPs += Ps * vr; sPb += Pb * vr; sT += tr * vr; sPw += pw[I, J, K] * vr
        sε += dε * vr; sB += dB * vr
    end
    st = c.stats
    st[1] += nclip; st[2] += eclip
    st[4] = sPs * V0; st[5] = sPb * V0; st[6] = sε * V0; st[7] = sB * V0; st[8] = sT * V0
    st[9] += 1
    st[10] = sPw * V0
    c.surface === nothing || _surface!(c, Δt)
    return c
end

function init_tke!(c, e_min)
    T = eltype(c.e.data)
    g = mesh(c.e)
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        if _solid(c.mask, I, J, K)
            c.e.data[I, J, K] = zero(T)
            continue
        end
        fac = _first(c.first_level, i, j, k) ? c.surface_factor : one(T)
        eq = tke_equilibrium(max(c.D.data[I, J, K], zero(T)), c.N2.data[I, J, K], c.Δk[k];
                             c_k = c.c_k, c_ε0 = c.c_ε0, c_ε1 = c.c_ε1, c_l = c.c_l,
                             l_wall = _lwall(c.l_wall, i, j, k, T), factor = fac)
        c.e.data[I, J, K] = max(T(eq), T(e_min))
    end
    return c
end

# ===================== src/prognostic_sgs/reynolds_stress.jl =================

function rsm_coefficients!(nd, kd, Kd, c)
    T = eltype(nd)
    g = mesh(c.τ[1])
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    τd = map(x -> x.data, c.τ)
    n2 = c.N2.data
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K_ = Hx + i, Hy + j, Hz + k
        τc = _τat(τd, I, J, K_)
        e = _tke(τc)
        if _solid(c.mask, I, J, K_) || !(e > 0)
            nd[I, J, K_] = zero(T); kd[I, J, K_] = zero(T)
            for m in 1:6
                Kd[m][I, J, K_] = zero(T)
            end
            continue
        end
        _, _, ν, cθ = _rsm_local(c, e, n2[I, J, K_], i, j, k)
        nd[I, J, K_] = ν
        for m in 1:6
            Kd[m][I, J, K_] = cθ * τc[m]
        end
        kd[I, J, K_] = cθ * (2e) / 3
    end
    return nothing
end

# S: a tuple of six arrays (τ11, τ22, τ33, τ12, τ13, τ23)
function stress_edges!(S, τ, E, g::RegularGrid{T}) where {T}
    t11, t22, t33, t12, t13, t23 = τ
    i1, i2, j1, j2, k1, k2 = E.i1, E.i2, E.j1, E.j2, E.k1, E.k2
    q = T(0.25); c23 = T(2) / 3
    @inline dev(a, i, j, k) = @inbounds a[i, j, k] - c23 * (t11[i, j, k] + t22[i, j, k] + t33[i, j, k]) / 2
    @inbounds begin
        for k in k1:k2, j in j1:j2, i in i1-1:i2
            S[1][i, j, k] = dev(t11, i, j, k)
        end
        for k in k1:k2, j in j1-1:j2, i in i1:i2
            S[2][i, j, k] = dev(t22, i, j, k)
        end
        for k in k1-1:k2, j in j1:j2, i in i1:i2
            S[3][i, j, k] = dev(t33, i, j, k)
        end
        for k in k1:k2, j in j1:j2+1, i in i1:i2+1
            S[4][i, j, k] = _oxy(E, T, i, j, k) *
                (q * (t12[i-1, j-1, k] + t12[i, j-1, k] + t12[i-1, j, k] + t12[i, j, k]))
        end
        for k in k1:k2+1, j in j1:j2, i in i1:i2+1
            S[5][i, j, k] = _oxz(E, T, i, j, k) *
                (q * (t13[i-1, j, k-1] + t13[i, j, k-1] + t13[i-1, j, k] + t13[i, j, k]))
        end
        for k in k1:k2+1, j in j1:j2+1, i in i1:i2
            S[6][i, j, k] = _oyz(E, T, i, j, k) *
                (q * (t23[i, j-1, k-1] + t23[i, j, k-1] + t23[i, j-1, k] + t23[i, j, k]))
        end
    end
    return S
end

function rsm_viscosity!(nd, c)
    T = eltype(nd)
    g = mesh(c.τ[1])
    τd = map(x -> x.data, c.τ)
    n2 = c.N2.data
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = g.Hx + i, g.Hy + j, g.Hz + k
        e = _tke(_τat(τd, I, J, K))
        nd[I, J, K] = (_solid(c.mask, I, J, K) || !(e > 0)) ? zero(T) :
                      _rsm_local(c, e, n2[I, J, K], i, j, k)[3]
    end
    return nd
end

function solve!(A::Matrix{T}, b::Vector{T}) where {T}
    n = length(b)
    @inbounds for p in 1:n
        piv = p; amax = abs(A[p, p])
        for r in p+1:n
            abs(A[r, p]) > amax && (piv = r; amax = abs(A[r, p]))
        end
        if piv != p
            for q in 1:n
                A[p, q], A[piv, q] = A[piv, q], A[p, q]
            end
            b[p], b[piv] = b[piv], b[p]
        end
        d = A[p, p]
        for r in p+1:n
            f = A[r, p] / d
            f == 0 && continue
            for q in p+1:n
                A[r, q] -= f * A[p, q]
            end
            b[r] -= f * b[p]
        end
    end
    @inbounds for p in n:-1:1
        s = b[p]
        for q in p+1:n
            s -= A[p, q] * b[q]
        end
        b[p] = s / A[p, p]
    end
    return b
end

function rsm_substeps!(τ, G, dθ, r, cθ, β, dt, C1, C2, C3, L::Matrix{T}, A::Matrix{T},
                       b::Vector{T}) where {T}
    z3 = (zero(T), zero(T), zero(T))
    @inbounds for m in 1:6
        col = reynolds_stress_rhs(_UNIT6[m], G, dθ, r, cθ, β; C1 = C1, C2 = C2, C3 = C3, Ω = z3)
        for q in 1:6
            L[q, m] = col[q]
        end
    end
    μ = typemin(T)
    @inbounds for q in 1:6
        s = L[q, q]
        for m in 1:6
            m == q || (s += abs(L[q, m]))
        end
        μ = max(μ, s)
    end
    n = μ > 0 ? max(1, ceil(Int, min(2 * dt * μ, T(10_000)))) : 1
    h = T(dt) / n
    x = τ
    acc = ntuple(_ -> zero(T), Val(6))
    for _ in 1:n
        @inbounds for m in 1:6, q in 1:6
            A[q, m] = (q == m) - h * L[q, m]
        end
        @inbounds for m in 1:6
            b[m] = x[m]
        end
        solve!(A, b)
        x = @inbounds (b[1], b[2], b[3], b[4], b[5], b[6])
        acc = acc .+ x
    end
    return x, acc ./ n, n
end

# the Jacobi projection with heap matrices (the pre-port _jacobi3)
function jacobi3(τ)
    T = float(eltype(τ))
    a = [T(τ[1]) T(τ[4]) T(τ[5]); T(τ[4]) T(τ[2]) T(τ[6]); T(τ[5]) T(τ[6]) T(τ[3])]
    v = [one(T) zero(T) zero(T); zero(T) one(T) zero(T); zero(T) zero(T) one(T)]
    scale = abs(a[1, 1]) + abs(a[2, 2]) + abs(a[3, 3]) + abs(a[1, 2]) + abs(a[1, 3]) + abs(a[2, 3])
    for _ in 1:50
        off = abs(a[1, 2]) + abs(a[1, 3]) + abs(a[2, 3])
        off <= eps(T) * scale && break
        for (p, q) in ((1, 2), (1, 3), (2, 3))
            apq = a[p, q]
            apq == 0 && continue
            θ = (a[q, q] - a[p, p]) / (2apq)
            t = sign(θ) / (abs(θ) + sqrt(θ^2 + 1))
            θ == 0 && (t = one(T))
            c = 1 / sqrt(t^2 + 1); s = t * c
            for k in 1:3
                akp = a[k, p]; akq = a[k, q]
                a[k, p] = c * akp - s * akq
                a[k, q] = s * akp + c * akq
            end
            for k in 1:3
                apk = a[p, k]; aqk = a[q, k]
                a[p, k] = c * apk - s * aqk
                a[q, k] = s * apk + c * aqk
            end
            for k in 1:3
                vkp = v[k, p]; vkq = v[k, q]
                v[k, p] = c * vkp - s * vkq
                v[k, q] = s * vkp + c * vkq
            end
        end
    end
    return (a[1, 1], a[2, 2], a[3, 3]),
           ((v[1, 1], v[1, 2], v[1, 3]), (v[2, 1], v[2, 2], v[2, 3]), (v[3, 1], v[3, 2], v[3, 3]))
end

@inline _pack(f) = (f(1, 1), f(2, 2), f(3, 3), f(1, 2), f(1, 3), f(2, 3))
function project_realizable(τ)
    is_realizable(τ) && return τ, false
    λ, V = jacobi3(τ)
    λp = (max(λ[1], zero(λ[1])), max(λ[2], zero(λ[2])), max(λ[3], zero(λ[3])))
    τp = _pack((i, j) -> V[i][1] * λp[1] * V[j][1] + V[i][2] * λp[2] * V[j][2] +
                         V[i][3] * λp[3] * V[j][3])
    return τp, true
end

# the stress update of c.τ and c.stats (the stresses' halos, c.N2 and c.tend
# as advance_sgs! leaves them)
function rsm_update!(c, sim, Δt)
    T = eltype(c.N2.data)
    L6 = zeros(T, 6, 6); A6 = zeros(T, 6, 6); b6 = zeros(T, 6)
    g = mesh(c.τ[1])
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    V = g.dx * g.dy * g.dz
    idx, idy, idz = 1 / g.dx, 1 / g.dy, 1 / g.dz
    τd = map(x -> x.data, c.τ)
    td = map(x -> x.data, c.tend)
    ud, vd, wd = sim.u.data, sim.v.data, sim.w.data
    θd = sim.θ.data
    q = _qv(c, sim)
    qd = q === nothing ? nothing : q.data
    n2 = c.N2.data
    β = T(_β(c, sim))
    Ω = _rsm_Ω(c, sim)
    rot = Ω != (zero(T), zero(T), zero(T))
    R = coriolis_rotation(Ω, Δt)
    E = _Edges(c.mask, bounded_directions(sim.bcs), g)
    C1, C2, C3 = c.C1, c.C2, c.C3
    sP = zero(T); sB = zero(T); sε = zero(T); sT = zero(T)
    nclip = zero(T); eclip = zero(T); nsplit = 0; nin = 0
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = Hx + i, Hy + j, Hz + k
        if _solid(c.mask, I, J, K)
            for m in 1:6
                τd[m][I, J, K] = zero(T)
            end
            continue
        end
        τn = _τat(τd, I, J, K)
        e = _tke(τn)
        τs = τn
        if c.transport
            tr = _τat(td, I, J, K)
            sT += _tke(tr)
            τs = ntuple(m -> τn[m] + Δt * tr[m], Val(6))
        end
        rot && (τs = rotate_stress(τs, R))
        is_realizable(τs) || (nin += 1)
        Gc = _dual_gradient(ud, vd, wd, I, J, K, idx, idy, idz, E)
        dθ = _θv_gradient(θd, qd, I, J, K, idx, idy, idz)
        if e > 0
            _, r, _, cθ = _rsm_local(c, e, n2[I, J, K], i, j, k)
            τ1, τb, nsub = rsm_substeps!(τs, Gc, dθ, r, cθ, β, Δt, C1, C2, C3, L6, A6, b6)
            nsub > 1 && (nsplit += 1)
        else
            r = zero(T); cθ = zero(T)
            τ1 = τs; τb = τs
        end
        τp, changed = project_realizable(τ1)
        if changed
            nclip += 1
            eclip += (_tke(τp) - _tke(τ1)) * V
        end
        for m in 1:6
            τd[m][I, J, K] = τp[m]
        end
        if e > 0
            sP -= τb[1] * Gc[1] + τb[2] * Gc[5] + τb[3] * Gc[9] +
                  τb[4] * (Gc[2] + Gc[4]) + τb[5] * (Gc[3] + Gc[7]) + τb[6] * (Gc[6] + Gc[8])
            sB += -β * cθ * (τb[5] * dθ[1] + τb[6] * dθ[2] + τb[3] * dθ[3])
            sε += r * _tke(τb)
        end
    end
    st = c.stats
    st[1] += nclip; st[2] += eclip
    st[3] = sP * V; st[4] = sB * V; st[5] = sε * V; st[6] = sT * V
    st[7] += 1
    st[8] = nclip
    st[9] = eclip
    st[10] = nsplit
    st[12] = nin
    return c
end

end # module
