# KA-k (#307): the Berlin dynamics hooks as they were before the port
# (master f838d5f + KA-d #377): the immersed-wall operators of
# src/wall_models/immersed.jl (the scatter `A[c][I,J,K] += a[n]` into full
# work arrays), transcribed verbatim with the work arrays and the pairs
# passed in. The reference of test/ka_hooks.jl (bit for bit) and of the
# KA-k timings (old loops vs launches). Nothing here is used by the package.
# ---------------------------------------------------------------------------
module LegacyHooks
using Aeolus
using Aeolus.BoundaryConditions: fill_halos!
using Aeolus.WallModels: wall_drag, _tangential_speed

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

end # module
