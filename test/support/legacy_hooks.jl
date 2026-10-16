# KA-k (#307): the Berlin dynamics hooks as they were before the port
# (master f838d5f + KA-d #377), transcribed verbatim with their scratch
# passed in: the immersed-wall operators of src/wall_models/immersed.jl
# (the scatter `A[c][I,J,K] += a[n]` into full work arrays), the canopy
# drag of src/PlantCanopy.jl (full-grid loops), the direct forcing of
# src/IBM.jl (a loop over the solid cells), the moist-buoyancy loops of
# examples/urban/couplings/moist_buoyancy.jl and the per-level loops of
# src/LargeScaleForcing.jl and its reference_damping.jl. The reference of test/ka_hooks.jl and of the
# KA-k timings (old loops vs launches). Nothing here is used by the package.
# ---------------------------------------------------------------------------
module LegacyHooks
using Aeolus
using Aeolus.BoundaryConditions: fill_halos!
using Aeolus.WallModels: wall_drag, _tangential_speed
using Aeolus.Grids: interior
using Aeolus.PlantCanopy: canopy_drag_factor
using Aeolus.LargeScaleForcing: forcing_mean, _zc, _dzc, _grad_levels

# ===================== src/wall_models/immersed.jl ==========================

# `faces`, `ν`, `Δ` as in an ImmersedWalls; `A` the three work arrays (the
# size of u, v, w data, zero on entry and on exit); `stats` a Dict.
function immersed_wall_drag!(a::AbstractVector, faces, Δ, ν, u, v, w)
    T = eltype(a)
    @inbounds for (n, f) in enumerate(faces)
        U = _tangential_speed(f, u, v, w)
        a[n] = T(wall_drag(f.law, U, Δ[f.d] / 2; ν = ν)) / Δ[f.d]
    end
    return a
end

function apply_immersed_wall_stress!(sim, faces, Δ, ν, A, stats, Δt::Real)
    T = eltype(A[1])
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    u, v, w = sim.u.data, sim.v.data, sim.w.data
    q = (u, v, w)
    dV = Δ[1] * Δ[2] * Δ[3]
    # 1. the drag coefficient of every (face, wall) pair, summed per face
    a = immersed_wall_drag!(Vector{T}(undef, length(faces)), faces, Δ, ν, u, v, w)
    @inbounds for (n, f) in enumerate(faces)
        A[f.c][f.I, f.J, f.K] += a[n]
    end
    # 2. the implicit update, once per face (the first of its entries)
    F = zeros(T, 3)
    dt = T(Δt)
    @inbounds for f in faces
        Af = A[f.c][f.I, f.J, f.K]
        Af == 0 && continue
        q0 = q[f.c][f.I, f.J, f.K]
        q1 = q0 / (1 + dt * Af)
        q[f.c][f.I, f.J, f.K] = q1
        F[f.c] += q1 - q0
        A[f.c][f.I, f.J, f.K] = 0
    end
    # 3. diagnostics: τ_w ΔA at the end-of-step speed along each wall
    τA = zero(T); area = zero(T); τmin = T(Inf)
    @inbounds for (n, f) in enumerate(faces)
        dA = dV / Δ[f.d]
        τ = a[n] * Δ[f.d] * abs(q[f.c][f.I, f.J, f.K])
        τA += τ * dA; area += dA
        τmin = min(τmin, a[n])
    end
    Fo = (F[1] * dV / dt, F[2] * dV / dt, F[3] * dV / dt)
    stats[:F] = Fo
    stats[:τA] = τA
    stats[:area] = area
    stats[:τ_min] = isempty(faces) ? zero(T) : τmin
    return Fo
end

function remove_stencil_wall_flux!(sim, faces, Δ, stats; ν::Real, νt, sgs::Bool)
    T = eltype(sim.u.data)
    q = (sim.u.data, sim.v.data, sim.w.data)
    t = (sim.tu.data, sim.tv.data, sim.tw.data)
    nt = νt isa AbstractArray ? νt : νt.data
    dV = Δ[1] * Δ[2] * Δ[3]
    νT = T(ν)
    F = zeros(T, 3)
    @inbounds for f in faces
        c, d, s = Int(f.c), Int(f.d), Int(f.s)
        I, J, K = Int(f.I), Int(f.J), Int(f.K)
        Ni, Nj, Nk = I + s * (d == 1), J + s * (d == 2), K
        ci, cj, ck = (c == 1), (c == 2), (c == 3)
        νe = νT
        if sgs
            νe += (nt[I, J, K] + nt[I-ci, J-cj, K-ck] +
                   nt[Ni, Nj, Nk] + nt[Ni-ci, Nj-cj, Nk-ck]) / 4
        end
        flux = νe * (q[c][Ni, Nj, Nk] - q[c][I, J, K]) / Δ[d]^2
        t[c][I, J, K] -= flux
        F[c] += flux
    end
    Fo = (F[1] * dV, F[2] * dV, F[3] * dV)
    stats[:F_stencil] = Fo
    return Fo
end

# ===================== src/PlantCanopy.jl ===================================
# (the scratch f_u, f_v, f_w passed in: ones(T, Nx, Ny, Nz) each)

function apply_canopy_drag!(sim, drag, f_u, f_v, f_w, dt::Real)
    T = eltype(drag.a_u)
    mesh = sim.mesh
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    Δt = T(dt); C = drag.C_d
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    u = sim.u.data; v = sim.v.data; w = sim.w.data
    q = T(1) / 4
    # 1. All factors from the pre-drag velocity.
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        I = Hx + i; J = Hy + j; K = Hz + k
        au = drag.a_u[i, j, k]
        if au > 0
            uu = u[I, J, K]
            vv = q * (v[I-1, J, K] + v[I, J, K] + v[I-1, J+1, K] + v[I, J+1, K])
            ww = q * (w[I-1, J, K] + w[I, J, K] + w[I-1, J, K+1] + w[I, J, K+1])
            f_u[i, j, k] = canopy_drag_factor(C, au, sqrt(uu^2 + vv^2 + ww^2), Δt)
        end
        av = drag.a_v[i, j, k]
        if av > 0
            vv = v[I, J, K]
            uu = q * (u[I, J-1, K] + u[I+1, J-1, K] + u[I, J, K] + u[I+1, J, K])
            ww = q * (w[I, J-1, K] + w[I, J, K] + w[I, J-1, K+1] + w[I, J, K+1])
            f_v[i, j, k] = canopy_drag_factor(C, av, sqrt(uu^2 + vv^2 + ww^2), Δt)
        end
        aw = drag.a_w[i, j, k]
        if aw > 0
            ww = w[I, J, K]
            uu = q * (u[I, J, K-1] + u[I+1, J, K-1] + u[I, J, K] + u[I+1, J, K])
            vv = q * (v[I, J, K-1] + v[I, J+1, K-1] + v[I, J, K] + v[I, J+1, K])
            f_w[i, j, k] = canopy_drag_factor(C, aw, sqrt(uu^2 + vv^2 + ww^2), Δt)
        end
    end
    # 2. Apply, and integrate the momentum removed.
    Fx = zero(T); Fy = zero(T); Fz = zero(T)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        I = Hx + i; J = Hy + j; K = Hz + k
        if drag.a_u[i, j, k] > 0
            u0 = u[I, J, K]; u[I, J, K] = u0 * f_u[i, j, k]; Fx += u[I, J, K] - u0
        end
        if drag.a_v[i, j, k] > 0
            v0 = v[I, J, K]; v[I, J, K] = v0 * f_v[i, j, k]; Fy += v[I, J, K] - v0
        end
        if drag.a_w[i, j, k] > 0
            w0 = w[I, J, K]; w[I, J, K] = w0 * f_w[i, j, k]; Fz += w[I, J, K] - w0
        end
    end
    dV = mesh.dx * mesh.dy * mesh.dz
    return (Fx = Fx * dV / Δt, Fy = Fy * dV / Δt, Fz = Fz * dV / Δt)
end


# ===================== src/IBM.jl ===========================================

function add_ibm_penalty!(sim, sm; τ::Real = 1e-3)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    mk = sm.mask
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    tud = sim.tu.data; tvd = sim.tv.data; twd = sim.tw.data
    invτ = 1 / τ
    @inbounds for k in kk, j in jj, i in ii
        # A face is "in solid" if either bordering cell is solid.
        if mk[i, j, k]
            tud[i,   j, k] -= invτ * ud[i,   j, k]
            tud[i+1, j, k] -= invτ * ud[i+1, j, k]
            tvd[i, j,   k] -= invτ * vd[i, j,   k]
            tvd[i, j+1, k] -= invτ * vd[i, j+1, k]
            twd[i, j, k]   -= invτ * wd[i, j, k]
            twd[i, j, k+1] -= invτ * wd[i, j, k+1]
        end
    end
    return sim
end


function implicit_ibm_correction!(sim, sm; τ::Real = 0.0, dt::Real = 0.0)
    m  = sm.mesh
    ii, jj, kk = interior(m)
    ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    mk = sm.mask
    factor = τ > 0 ? 1 / (1 + dt / τ) : 0.0
    @inbounds for k in kk, j in jj, i in ii
        if mk[i, j, k]
            ud[i,   j, k] *= factor
            ud[i+1, j, k] *= factor
            vd[i, j,   k] *= factor
            vd[i, j+1, k] *= factor
            wd[i, j, k]   *= factor
            wd[i, j, k+1] *= factor
        end
    end
    return sim
end


# ===================== examples/urban/couplings/moist_buoyancy.jl ===========
# (the deviation loop of moist_buoyancy_perturbation!, given the level sums
# S and counts n of its plane_reduce!, and add_moist_buoyancy!)

function moist_deviation!(Δ, S, n, fluid)
    Nx, Ny, Nz = size(Δ)
    @inbounds for k in 1:Nz
        m = n[k] > 0 ? S[k] / n[k] : 0.0
        for j in 1:Ny, i in 1:Nx
            fluid[i, j, k] && (Δ[i, j, k] -= m)
        end
    end
    return Δ
end


function add_moist_buoyancy!(tw, Δ::AbstractArray{<:Real,3}, θ₀::Real, g::Real)
    m = tw.mesh
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nx, Ny, Nz = size(Δ)
    coef = g / θ₀
    d = tw.data
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        d[Hx+i, Hy+j, Hz+k] += coef * 0.5 * (Δ[i, j, k-1] + Δ[i, j, k])
    end
    return tw
end


# ===================== src/LargeScaleForcing.jl =============================

function apply_gpg!(sim, F, t, Δt)
    m = sim.mesh
    tu, tv = sim.tu.data, sim.tv.data
    T = eltype(tu)
    f = sim.f_coriolis
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    t1 = t + Δt
    @inbounds for k in 1:Nz
        z = _zc(m, k)
        fu = T(-(f * forcing_mean(F.V_g, z, t, t1)))
        fv = T(f * forcing_mean(F.U_g, z, t, t1))
        for j in 1:Ny, i in 1:Nx
            tu[Hx+i, Hy+j, Hz+k] += fu
            tv[Hx+i, Hy+j, Hz+k] += fv
        end
    end
    return sim
end


function _nudge_levels!(d, td, m, ref, invτ, wfn, t, t1)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    dA = m.dx * m.dy * (Nx * Ny)
    I = 0.0
    @inbounds for k in 1:Nz
        z = _zc(m, k)
        φref = forcing_mean(ref, z, t, t1)
        φ̄ = 0.0
        for j in 1:Ny, i in 1:Nx
            φ̄ += d[Hx+i, Hy+j, Hz+k]
        end
        φ̄ /= (Nx * Ny)
        c = invτ * (φ̄ - φref)
        wfn === nothing || (c = wfn(z) * c)
        for j in 1:Ny, i in 1:Nx
            td[Hx+i, Hy+j, Hz+k] -= c
        end
        I -= c * dA * _dzc(m, k)
    end
    return I
end


function _subside_mean!(d, td, m, w_s, t, t1, buf)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nz >= 2 || return 0.0
    resize!(buf, Nz)
    @inbounds for k in 1:Nz
        s = 0.0
        for j in 1:Ny, i in 1:Nx
            s += d[Hx+i, Hy+j, Hz+k]
        end
        buf[k] = s / (Nx * Ny)
    end
    dA = m.dx * m.dy * (Nx * Ny)
    I = 0.0
    @inbounds for k in 1:Nz
        kl, ku = _grad_levels(k, Nz)
        G = (buf[ku] - buf[kl]) / (_zc(m, ku) - _zc(m, kl))
        z = _zc(m, k)
        s = -forcing_mean(w_s, z, t, t1) * G
        for j in 1:Ny, i in 1:Nx
            td[Hx+i, Hy+j, Hz+k] += s
        end
        I += s * dA * _dzc(m, k)
    end
    return I
end


function _subside_local!(d, td, m, w_s, t, t1)
    Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
    Hx, Hy, Hz = m.Hx, m.Hy, m.Hz
    Nz >= 2 || return 0.0
    dAc = m.dx * m.dy
    I = 0.0
    @inbounds for k in 1:Nz
        z = _zc(m, k)
        w = forcing_mean(w_s, z, t, t1)
        # upwind: from above for descent (w < 0), from below for ascent;
        # one-sided toward the interior where the upwind level is missing
        kl, ku = w < 0 ? (k == Nz ? (Nz - 1, Nz) : (k, k + 1)) :
                         (k == 1 ? (1, 2) : (k - 1, k))
        idz = 1 / (_zc(m, ku) - _zc(m, kl))
        Ik = 0.0
        for j in 1:Ny, i in 1:Nx
            s = -w * (d[Hx+i, Hy+j, Hz+ku] - d[Hx+i, Hy+j, Hz+kl]) * idz
            td[Hx+i, Hy+j, Hz+k] += s
            Ik += s
        end
        I += Ik * dAc * _dzc(m, k)
    end
    return I
end


function _add_level_tendency!(d, m, x, t, Δt)
    T = eltype(d)
    t1 = t + Δt
    @inbounds for k in 1:m.Nz
        s = T(forcing_mean(x, _zc(m, k), t, t1))
        for j in 1:m.Ny, i in 1:m.Nx
            d[m.Hx+i, m.Hy+j, m.Hz+k] += s
        end
    end
    return d
end


# ===================== src/large_scale_forcing/reference_damping.jl =========

function resolve_plane_means!(buf, f)
    m = f.mesh
    d = f.data
    n = m.Nx * m.Ny
    @inbounds for k in 1:m.Nz
        s = 0.0
        for j in 1:m.Ny, i in 1:m.Nx
            s += d[m.Hx+i, m.Hy+j, m.Hz+k]
        end
        buf[k] = s / n
    end
    return buf
end

end # module
