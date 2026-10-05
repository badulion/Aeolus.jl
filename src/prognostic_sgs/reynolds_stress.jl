# =========================================================================
# Era VI, Phase 56 (P56-d, #188) — the differential SGS Reynolds-stress
# closure. Included into `module PrognosticSGS`, after the TKE closure,
# whose length scale, dissipation rate and eddy coefficients it calls.
# =========================================================================
#
# Six prognostic cell-centred stresses τᵢⱼ = ⟨uᵢ'uⱼ'⟩ (m² s⁻², kinematic),
#
#   Dτᵢⱼ/Dt = Pᵢⱼ + Gᵢⱼ + Cᵢⱼ + Φᵢⱼ − (2/3) ε δᵢⱼ + ∇·(2ν_t ∇τᵢⱼ),
#
# with e = ½ τₖₖ, ε = c_ε e^{3/2}/l and l the Deardorff length scale of
# the TKE closure (`tke_length_scale`, `tke_dissipation_coefficient`), so
# that ½ tr of the equation is the TKE equation of P56-b with the exact
# production −τᵢₖ∂ₖuᵢ in place of ν_t|S|².
#
# Component order everywhere: (11, 22, 33, 12, 13, 23).

using ..Operators: StressTensor, add_stress_divergence!, DiffusivityTensor,
                   add_tensor_diffusion_scalar!, bounded_directions, _θv_face_dir,
                   _scalar_bcs
import ..Operators: add_sgs_momentum!, add_sgs_scalar!, sgs_scalar_tensor,
                    sgs_explicit_stress, _fill_sgs_halos!
using ..AdvectionSchemes: advection_scheme

# -------------------------------------------------------------------------
# Small symmetric-tensor helpers
# -------------------------------------------------------------------------

@inline _sym(τ) = ((τ[1], τ[4], τ[5]), (τ[4], τ[2], τ[6]), (τ[5], τ[6], τ[3]))
@inline _rows(G) = ((G[1], G[2], G[3]), (G[4], G[5], G[6]), (G[7], G[8], G[9]))
@inline _pack(f) = (f(1, 1), f(2, 2), f(3, 3), f(1, 2), f(1, 3), f(2, 3))
@inline _δ(i, j) = i == j
@inline _tke(τ) = (τ[1] + τ[2] + τ[3]) / 2

"""
    reynolds_stress_terms(τ, G, dθ, r, cθ, β; C1 = 1.8, C2 = 0.6, C3 = 0.5,
                          Ω = (0, 0, 0)) -> NamedTuple

The source terms of the Reynolds-stress equation at one point, each a
6-tuple in the order (11, 22, 33, 12, 13, 23):

* `production` ``P_{ij} = −τ_{ik}∂_k u_j − τ_{jk}∂_k u_i`` (exact);
* `buoyancy` ``G_{ij} = β(δ_{i3}h_j + δ_{j3}h_i)`` with the Daly & Harlow
  (1970) heat flux ``h_i = −c_θ τ_{ij}∂_jθ_v`` (``c_θ = C_θ e/ε``);
* `coriolis` ``C_{ij} = −2Ω_k(ε_{ikm}τ_{mj} + ε_{jkm}τ_{mi})`` (trace-free);
* `pressure_strain`, the LRR-IP model of Launder, Reece & Rodi (1975) with
  the buoyancy term of Gibson & Launder (1978):
  ``Φ_{ij} = −C_1 (ε/e)(τ_{ij} − \\tfrac23 eδ_{ij}) − C_2(P_{ij} − \\tfrac13 P_{kk}δ_{ij})
  − C_3(G_{ij} − \\tfrac13 G_{kk}δ_{ij})``;
* `dissipation` ``−\\tfrac23 ε δ_{ij}`` (local isotropy),

and `rhs`, their sum. `G` is the velocity gradient ``G_{ik} = ∂u_i/∂x_k``
as a 9-tuple, row by row; `dθ` is ``∇θ_v``; `r = ε/e` (s⁻¹, so
``ε = r e``); `cθ` the Daly–Harlow factor (s; ``h = −c_θ τ ∇θ_v``); `β =
g/θ₀`. For frozen `r` and `cθ` every term is **linear** in `τ` — the
3-D closure builds its implicit step from that (see
[`ReynoldsStressClosure`](@ref)).

Half the trace of `rhs` is ``−τ_{ik}∂_k u_i + β h_3 − ε``, the TKE
equation's production, buoyancy and dissipation (the pressure-strain and
Coriolis terms are trace-free).

References: Launder, B. E., Reece, G. J. & Rodi, W. (1975), J. Fluid Mech.
68, 537; Gibson, M. M. & Launder, B. E. (1978), J. Fluid Mech. 86, 491;
Daly, B. J. & Harlow, F. H. (1970), Phys. Fluids 13, 2634; Deardorff, J.
W. (1973), J. Fluids Eng. 95, 429.
"""
@inline function reynolds_stress_terms(τ, G, dθ, r, cθ, β; C1 = 1.8, C2 = 0.6, C3 = 0.5,
                                       Ω = (0.0, 0.0, 0.0))
    T = _sym(τ)
    Gm = _rows(G)
    e = _tke(τ)
    # A_ij = Σ_k τ_ik G_jk; P_ij = −(A_ij + A_ji)
    A(i, j) = T[i][1] * Gm[j][1] + T[i][2] * Gm[j][2] + T[i][3] * Gm[j][3]
    P = _pack((i, j) -> -(A(i, j) + A(j, i)))
    # Daly–Harlow heat flux h = −cθ τ ∇θ_v
    h = ntuple(i -> -cθ * (T[i][1] * dθ[1] + T[i][2] * dθ[2] + T[i][3] * dθ[3]), Val(3))
    B = _pack((i, j) -> β * ((i == 3) * h[j] + (j == 3) * h[i]))
    # Coriolis: W = [Ω]×, C = −2(Wτ + (Wτ)ᵀ)
    W = ((zero(β), -Ω[3], Ω[2]), (Ω[3], zero(β), -Ω[1]), (-Ω[2], Ω[1], zero(β)))
    M(i, j) = W[i][1] * T[1][j] + W[i][2] * T[2][j] + W[i][3] * T[3][j]
    C = _pack((i, j) -> -2 * (M(i, j) + M(j, i)))
    Pk = (P[1] + P[2] + P[3]) / 2
    Bk = (B[1] + B[2] + B[3]) / 2
    Tv = _pack((i, j) -> T[i][j])
    Φ = ntuple(m -> (d = m <= 3;
                     -C1 * r * (Tv[m] - d * (2e / 3)) - C2 * (P[m] - d * (2Pk / 3)) -
                     C3 * (B[m] - d * (2Bk / 3))), Val(6))
    D = ntuple(m -> m <= 3 ? -(2 / 3) * r * e : zero(e), Val(6))
    rhs = ntuple(m -> P[m] + B[m] + C[m] + Φ[m] + D[m], Val(6))
    return (production = P, buoyancy = B, coriolis = C, pressure_strain = Φ,
            dissipation = D, rhs = rhs, heat_flux = h)
end

"""
    reynolds_stress_rhs(τ, G, dθ, r, cθ, β; C1, C2, C3, Ω) -> NTuple{6}

The local right-hand side ``P + G + C + Φ − \\tfrac23 εδ`` of the
Reynolds-stress equation (the `rhs` of [`reynolds_stress_terms`](@ref)).
"""
@inline reynolds_stress_rhs(τ, G, dθ, r, cθ, β; kw...) =
    reynolds_stress_terms(τ, G, dθ, r, cθ, β; kw...).rhs

"""
    daly_harlow_coefficient(e, l, Δ, r; C_θ = :tke, c_k = 0.1) -> c_θ

The factor ``c_θ`` (s) of the Daly & Harlow (1970) scalar flux
``h_i = −c_θ τ_{ij}∂_jθ`` (diffusivity tensor ``K_{ij} = c_θ τ_{ij}``).

* `C_θ` a number: ``c_θ = C_θ e/ε = C_θ/r`` (Daly & Harlow's form);
* `C_θ = :tke` (default): ``c_θ = \\tfrac32 κ_t/e`` with the TKE closure's
  ``κ_t = (1 + 2l/Δ) c_k l\\sqrt e`` ([`tke_eddy_coefficients`](@ref)), so an
  isotropic ``τ = \\tfrac23 e I`` diffuses scalars with exactly the P56-b
  ``κ_t``. It is the Daly–Harlow form with
  ``C_θ = \\tfrac32 (1 + 2l/Δ) c_k c_ε`` (0.315 at ``l = Δ``, ``c_ε = 0.70``;
  0.099 in the ``l → 0`` stable limit), within the 0.1–0.3 range of the
  published constants — a calibration, not a published value.

Zero for ``e ≤ 0``.
"""
@inline function daly_harlow_coefficient(e, l, Δ, r; C_θ = :tke, c_k = 0.1)
    e > 0 || return zero(float(e))
    if C_θ === :tke
        _, κ = tke_eddy_coefficients(e, l, Δ; c_k = c_k)
        return 3κ / (2e)
    end
    return r > 0 ? C_θ / r : zero(float(e))
end

"""
    coriolis_rotation(Ω, dt) -> R

The rotation ``R = \\exp(−2\\,dt\\,[Ω]_×)`` (a 3×3 tuple of rows, Rodrigues'
formula) that solves the Coriolis part of the stress equation exactly:
``dτ/dt = C(τ)`` gives ``τ(t + dt) = R τ(t) R^T`` — a rotation about Ω by
the angle ``−2|Ω|dt``, which keeps the trace, the determinant and
realizability.
"""
function coriolis_rotation(Ω, dt)
    T = float(promote_type(eltype(Ω), typeof(dt)))
    a = sqrt(Ω[1]^2 + Ω[2]^2 + Ω[3]^2)
    I3 = ((one(T), zero(T), zero(T)), (zero(T), one(T), zero(T)), (zero(T), zero(T), one(T)))
    a == 0 && return I3
    n = (Ω[1] / a, Ω[2] / a, Ω[3] / a)
    K = ((zero(T), -n[3], n[2]), (n[3], zero(T), -n[1]), (-n[2], n[1], zero(T)))
    θ = -2 * a * dt
    s, c = sincos(θ)
    K2(i, j) = K[i][1] * K[1][j] + K[i][2] * K[2][j] + K[i][3] * K[3][j]
    return ntuple(i -> ntuple(j -> I3[i][j] + s * K[i][j] + (1 - c) * K2(i, j), Val(3)), Val(3))
end

"""
    rotate_stress(τ, R) -> τ′

``τ′ = R τ R^T`` for a symmetric `τ` (6-tuple) and a 3×3 tuple `R`.
"""
@inline function rotate_stress(τ, R)
    T = _sym(τ)
    X(i, j) = T[i][1] * R[j][1] + T[i][2] * R[j][2] + T[i][3] * R[j][3]     # τ Rᵀ
    return _pack((i, j) -> R[i][1] * X(1, j) + R[i][2] * X(2, j) + R[i][3] * X(3, j))
end

"""
    is_realizable(τ; rtol = 1e-12) -> Bool

Schumann's (1977) realizability conditions for a symmetric stress `τ`:
``τ_{αα} ≥ 0``, ``τ_{αβ}^2 ≤ τ_{αα}τ_{ββ}`` and ``\\det τ ≥ 0`` — all the
principal minors non-negative, i.e. `τ` positive semi-definite — with a
round-off slack `rtol` relative to ``e^n`` (n the order of the minor).
"""
@inline function is_realizable(τ; rtol = 1e-12)
    e = _tke(τ)
    e >= 0 || return false
    s = rtol * e
    a, b, c, d, f, g = τ
    (a >= -s && b >= -s && c >= -s) || return false
    s2 = s * e
    (a * b - d^2 >= -s2 && a * c - f^2 >= -s2 && b * c - g^2 >= -s2) || return false
    det = a * (b * c - g^2) - d * (d * c - g * f) + f * (d * g - b * f)
    return det >= -s2 * e
end

"""
    project_realizable(τ) -> (τp, changed::Bool)

The nearest realizable (positive semi-definite) stress to `τ` in the
Frobenius norm: the eigenvalues of `τ` clipped at zero (Jacobi
diagonalisation). `τ` itself, unchanged, when [`is_realizable`](@ref).
The trace can only grow: ``½ \\operatorname{tr}(τ_p − τ)`` is the energy the
correction adds.
"""
function project_realizable(τ)
    is_realizable(τ) && return τ, false
    λ, V = _jacobi3(τ)
    λp = (max(λ[1], zero(λ[1])), max(λ[2], zero(λ[2])), max(λ[3], zero(λ[3])))
    τp = _pack((i, j) -> V[i][1] * λp[1] * V[j][1] + V[i][2] * λp[2] * V[j][2] +
                         V[i][3] * λp[3] * V[j][3])
    return τp, true
end

# Cyclic Jacobi for a symmetric 3×3 (6-tuple): eigenvalues λ and the
# eigenvector matrix V (columns), τ = V diag(λ) Vᵀ.
function _jacobi3(τ)
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

"""
    stress_anisotropy(τ) -> (b, II, III)

The anisotropy tensor ``b_{ij} = τ_{ij}/(2e) − δ_{ij}/3`` (6-tuple) and its
invariants ``II = b_{ij}b_{ji}`` and ``III = b_{ij}b_{jk}b_{ki}`` (Lumley
1978; 0 for isotropic turbulence, ``II = 2/3`` for one-component
turbulence).
"""
function stress_anisotropy(τ)
    e = _tke(τ)
    b = ntuple(m -> τ[m] / (2e) - (m <= 3) / 3, Val(6))
    B = _sym(b)
    II = sum(B[i][j] * B[j][i] for i in 1:3, j in 1:3)
    III = sum(B[i][j] * B[j][k] * B[k][i] for i in 1:3, j in 1:3, k in 1:3)
    return b, II, III
end

"""
    reynolds_stress_shear_equilibrium(; C1 = 1.8, C2 = 0.6) -> (a11, a22, a13)

The LRR-IP equilibrium of a homogeneous shear ``∂u_1/∂x_3`` at production
= dissipation (the equilibrium of the SGS closure, whose ``ε/e`` grows
with ``e``): ``τ_{11}/e = a11``, ``τ_{22}/e = τ_{33}/e = a22`` and
``−τ_{13}/e = a13``, from ``P_{ij} + Φ_{ij} = \\tfrac23 εδ_{ij}`` with
``P = ε``:

``a22 = \\tfrac23 − \\tfrac23(1 − C_2)/C_1``,
``a11 = \\tfrac23 + (2(1 − C_2) + \\tfrac23 C_2 − \\tfrac23)/C_1``,
``a13 = \\sqrt{(1 − C_2) a22/C_1}`` — 0.962963, 0.518519, 0.339450 for
the LRR-IP constants. The surface value of [`ReynoldsStressClosure`](@ref)
uses it.
"""
function reynolds_stress_shear_equilibrium(; C1 = 1.8, C2 = 0.6)
    a22 = 2 / 3 - (2 / 3) * (1 - C2) / C1
    a11 = 2 / 3 + (2 * (1 - C2) + (2 / 3) * C2 - 2 / 3) / C1
    return a11, a22, sqrt((1 - C2) * a22 / C1)
end

# -------------------------------------------------------------------------
# The grid: edge stresses and their dual velocity gradient
# -------------------------------------------------------------------------

# Which stress edges are open: not on a wall of a bounded direction and
# with no solid cell among the four around them.
struct _Edges{M}
    mask::M
    bx::Bool; by::Bool; bz::Bool
    i1::Int; i2::Int; j1::Int; j2::Int; k1::Int; k2::Int
end
function _Edges(mask, bounded::NTuple{3,Bool}, g::RegularGrid)
    return _Edges(mask, bounded..., g.Hx + 1, g.Hx + g.Nx, g.Hy + 1, g.Hy + g.Ny,
                  g.Hz + 1, g.Hz + g.Nz)
end
@inline _any4(::Nothing, a...) = false
@inline _any4(m, a, b, c, d) = @inbounds m[a...] | m[b...] | m[c...] | m[d...]
# xy-edge (i-½, j-½, k), xz-edge (i-½, j, k-½), yz-edge (i, j-½, k-½)
@inline _oxy(E::_Edges, ::Type{T}, i, j, k) where {T} =
    T(!((E.bx && (i == E.i1 || i == E.i2 + 1)) || (E.by && (j == E.j1 || j == E.j2 + 1)) ||
        _any4(E.mask, (i-1, j-1, k), (i, j-1, k), (i-1, j, k), (i, j, k))))
@inline _oxz(E::_Edges, ::Type{T}, i, j, k) where {T} =
    T(!((E.bx && (i == E.i1 || i == E.i2 + 1)) || (E.bz && (k == E.k1 || k == E.k2 + 1)) ||
        _any4(E.mask, (i-1, j, k-1), (i, j, k-1), (i-1, j, k), (i, j, k))))
@inline _oyz(E::_Edges, ::Type{T}, i, j, k) where {T} =
    T(!((E.by && (j == E.j1 || j == E.j2 + 1)) || (E.bz && (k == E.k1 || k == E.k2 + 1)) ||
        _any4(E.mask, (i, j-1, k-1), (i, j, k-1), (i, j-1, k), (i, j, k))))

"""
    stress_edges!(S::StressTensor, τ::NTuple{6,<:Field}; bounded = (false, false, false),
                  mask = nothing) -> S

Place a cell-centred symmetric stress on the C-grid of
`Operators.add_stress_divergence!` (P55-b): the **deviatoric** diagonals
``τ_{ii} − \\tfrac23 e`` at the cell centres (the isotropic part is a
pressure), the off-diagonals averaged from the four cells around each edge.
An edge on a wall of a `bounded` direction, or touching a solid cell of
`mask` (storage-sized, halos consistent with the boundary conditions),
carries no stress: the wall stress is the surface exchange's. The halos of
`τ` must be filled. With this placement the momentum operator and
[`dual_velocity_gradient`](@ref) are exact duals,
``−Σ_{faces} \\mathbf u·∇·τ^{dev} V = Σ_{cells} τ^{dev}_{ik} G_{ik} V``.
"""
function stress_edges!(S::StressTensor, τ::NTuple{6,<:Field}; bounded::NTuple{3,Bool} = (false, false, false),
                       mask = nothing)
    g = mesh(τ[1])
    _stress_edges!(S, map(f -> f.data, τ), _Edges(mask, bounded, g), g)
    return S
end

function _stress_edges!(S::StressTensor, τ, E::_Edges, g::RegularGrid{T}) where {T}
    t11, t22, t33, t12, t13, t23 = τ
    i1, i2, j1, j2, k1, k2 = E.i1, E.i2, E.j1, E.j2, E.k1, E.k2
    q = T(0.25); c23 = T(2) / 3
    @inline dev(a, i, j, k) = @inbounds a[i, j, k] - c23 * (t11[i, j, k] + t22[i, j, k] + t33[i, j, k]) / 2
    @inbounds begin
        for k in k1:k2, j in j1:j2, i in i1-1:i2
            S.τ11[i, j, k] = dev(t11, i, j, k)
        end
        for k in k1:k2, j in j1-1:j2, i in i1:i2
            S.τ22[i, j, k] = dev(t22, i, j, k)
        end
        for k in k1-1:k2, j in j1:j2, i in i1:i2
            S.τ33[i, j, k] = dev(t33, i, j, k)
        end
        for k in k1:k2, j in j1:j2+1, i in i1:i2+1
            S.τ12[i, j, k] = _oxy(E, T, i, j, k) *
                (q * (t12[i-1, j-1, k] + t12[i, j-1, k] + t12[i-1, j, k] + t12[i, j, k]))
        end
        for k in k1:k2+1, j in j1:j2, i in i1:i2+1
            S.τ13[i, j, k] = _oxz(E, T, i, j, k) *
                (q * (t13[i-1, j, k-1] + t13[i, j, k-1] + t13[i-1, j, k] + t13[i, j, k]))
        end
        for k in k1:k2+1, j in j1:j2+1, i in i1:i2
            S.τ23[i, j, k] = _oyz(E, T, i, j, k) *
                (q * (t23[i, j-1, k-1] + t23[i, j, k-1] + t23[i, j-1, k] + t23[i, j, k]))
        end
    end
    return S
end

"""
    dual_velocity_gradient(u, v, w, i, j, k; bounded = (false, false, false), mask = nothing)
        -> NTuple{9}

The velocity gradient ``G_{ik} = ∂u_i/∂x_k`` at the centre of cell
`(i, j, k)` (storage indices), row by row, that is the exact dual of the
stress placement of [`stress_edges!`](@ref): the diagonal is the
face-to-face difference through the cell, each off-diagonal the mean of
its edge differences over the cell's four edges in that plane (a closed
edge counts zero). The production ``P_{ij}`` of
[`ReynoldsStressClosure`](@ref) is built from it, so the resolved kinetic
energy the stress divergence removes is exactly the SGS energy produced.
For a linear velocity field it is the exact gradient away from closed
edges.
"""
function dual_velocity_gradient(u::Field, v::Field, w::Field, i, j, k;
                                bounded::NTuple{3,Bool} = (false, false, false), mask = nothing)
    g = mesh(u)
    return _dual_gradient(u.data, v.data, w.data, i, j, k, 1 / g.dx, 1 / g.dy, 1 / g.dz,
                          _Edges(mask, bounded, g))
end

@inline function _dual_gradient(ud, vd, wd, i, j, k, idx, idy, idz, E::_Edges)
    T = eltype(ud)
    q = T(0.25)
    @inbounds begin
        G11 = (ud[i+1, j, k] - ud[i, j, k]) * idx
        G22 = (vd[i, j+1, k] - vd[i, j, k]) * idy
        G33 = (wd[i, j, k+1] - wd[i, j, k]) * idz
        G12 = zero(T); G21 = zero(T)
        for (a, b) in ((i, j), (i+1, j), (i, j+1), (i+1, j+1))
            o = _oxy(E, T, a, b, k)
            G12 += o * (ud[a, b, k] - ud[a, b-1, k]) * idy
            G21 += o * (vd[a, b, k] - vd[a-1, b, k]) * idx
        end
        G13 = zero(T); G31 = zero(T)
        for (a, c) in ((i, k), (i+1, k), (i, k+1), (i+1, k+1))
            o = _oxz(E, T, a, j, c)
            G13 += o * (ud[a, j, c] - ud[a, j, c-1]) * idz
            G31 += o * (wd[a, j, c] - wd[a-1, j, c]) * idx
        end
        G23 = zero(T); G32 = zero(T)
        for (b, c) in ((j, k), (j+1, k), (j, k+1), (j+1, k+1))
            o = _oyz(E, T, i, b, c)
            G23 += o * (vd[i, b, c] - vd[i, b, c-1]) * idz
            G32 += o * (wd[i, b, c] - wd[i, b-1, c]) * idy
        end
    end
    return (G11, q * G12, q * G13, q * G21, G22, q * G23, q * G31, q * G32, G33)
end

# centred ∇θ_v at the cell centre: the face values of the moist buoyancy
# variable (Operators._θv_face_dir, the one definition N² also uses)
@inline function _θv_gradient(θd, qd, i, j, k, idx, idy, idz)
    @inbounds return ((_θv_face_dir(θd, qd, i+1, j, k, 1, 0, 0) - _θv_face_dir(θd, qd, i, j, k, 1, 0, 0)) * idx,
                      (_θv_face_dir(θd, qd, i, j+1, k, 0, 1, 0) - _θv_face_dir(θd, qd, i, j, k, 0, 1, 0)) * idy,
                      (_θv_face_dir(θd, qd, i, j, k+1, 0, 0, 1) - _θv_face_dir(θd, qd, i, j, k, 0, 0, 1)) * idz)
end

# -------------------------------------------------------------------------
# The closure
# -------------------------------------------------------------------------

"""
    ReynoldsStressClosure(mesh::RegularGrid; τ = nothing, e0 = 0, C1 = 1.8, C2 = 0.6,
                          C3 = 0.5, C_θ = :tke, c_k = 0.1, c_ε0 = 0.19, c_ε1 = 0.51,
                          c_l = 0.76, κ = 0.4, surface_factor = 3.9, Ω = :auto,
                          wall_distance = nothing, first_level = nothing, mask = nothing,
                          surface = nothing, θ₀ = nothing, g_accel = nothing, qv = :auto,
                          transport = true, advection = :auto)

A differential subgrid-scale Reynolds-stress closure (Deardorff 1973):
six prognostic cell-centred stresses ``τ_{ij}`` (m² s⁻²) with

```math
\\frac{Dτ_{ij}}{Dt} = P_{ij} + G_{ij} + C_{ij} + Φ_{ij} − \\tfrac23 ε δ_{ij}
  + ∇·(2ν_t ∇τ_{ij})
```

(terms in [`reynolds_stress_terms`](@ref); LRR-IP pressure–strain with
`C1`, `C2`, `C3`). It is an `Operators.AbstractSGSClosure`:

* **shared with the TKE closure**, by function call: ``l`` from
  [`tke_length_scale`](@ref) (with ``N^2`` of θ_v, the wall cap
  ``κ d_{wall}``), ``ε = c_ε e^{3/2}/l`` with
  [`tke_dissipation_coefficient`](@ref) (Moeng's 3.9 in `first_level`),
  ``ν_t`` from [`tke_eddy_coefficients`](@ref);
* **momentum**: `add_sgs_momentum!` adds ``−∂_jτ^{dev}_{ij}`` through
  `Operators.add_stress_divergence!` (P55-b) with the edge placement of
  [`stress_edges!`](@ref) (the `stress` keyword of the cores is ignored);
* **scalars**: `add_sgs_scalar!` applies the Daly–Harlow tensor
  ``K_{ij} = c_θ τ_{ij}`` ([`daly_harlow_coefficient`](@ref)) through
  `Operators.add_tensor_diffusion_scalar!` (P55-c) — θ and q_v; `sim.νt`
  holds ``ν_t`` and `sim.κt` the isotropic part ``\\operatorname{tr}K/3``;
* `advance_sgs!` integrates the stresses over each accepted step: the
  transport (with `transport`: the flux-limited advection of the
  diagonals, the plain `advection` scheme for the off-diagonals, and
  ``∇·(2ν_t∇τ_{ij})``) explicitly; the Coriolis term exactly
  ([`coriolis_rotation`](@ref)); and the remaining sources by one
  backward-Euler step of the 6×6 linear system they form with ``ε/e``
  and ``c_θ`` lagged at ``τ^n`` — so a steady state of the discrete
  update is exactly a root of the stress equations. Then every cell is
  projected onto the realizable set ([`project_realizable`](@ref), Schumann
  1977), and every correction is booked ([`reynolds_stress_statistics`](@ref));
* `sgs_state` / `restore_sgs_state!` roll a rejected adaptive step back
  exactly; `step_imex!` refuses the closure (`sgs_explicit_stress`).

`Ω` is the rotation vector (rad s⁻¹): `:auto` takes ``(0, 0, f/2)`` from a
simulation with Coriolis (the f-plane the momentum uses), else zero.
`surface` — `nothing`, or `(u_star = M, k = K)` as for [`TKEClosure`](@ref):
after each step the first air cell of every column takes the
constant-stress surface-layer equilibrium — ``τ_{13}, τ_{23} = −u_*^2
(\\cos α, \\sin α)`` against the cell's wind direction ``α``, the normal
stresses ``(a_{11}, a_{22}, a_{22})e_s`` in the wind frame with
``e_s = u_*^2/a_{13}`` ([`reynolds_stress_shear_equilibrium`](@ref)), the
LRR-IP counterpart of the TKE closure's ``e_s`` (booked:
`surface_energy`).
`τ` passes in six fields to share (the Berlin case passes its tracers);
else they are allocated isotropic, ``τ = \\tfrac23 e_0 I``. Seed them with
[`initialize_reynolds_stress!`](@ref). `transport = false` when a driver
transports the stresses itself. `advection` (`:auto` = WENO-Z5 if the halo
is ≥ 3, else centred) is the scheme of the transport: the diagonals use it
flux-limited (`advect_positive!`), the off-diagonals plain. Prefer a halo of
3: with centred differences the off-diagonals overshoot and the
realizability projection fires (measured in a 16³ decaying box, amplitude
0.3 m/s, 60 s: 9140 corrections adding 2.8 % of the SGS energy, against
none with WENO-Z5). The other keywords are those of [`TKEClosure`](@ref).

References: Deardorff, J. W. (1973), *The use of subgrid transport
equations in a three-dimensional model of atmospheric turbulence*, J.
Fluids Eng. 95, 429–438; Launder, B. E., Reece, G. J. & Rodi, W. (1975),
*Progress in the development of a Reynolds-stress turbulence closure*, J.
Fluid Mech. 68, 537–566; Schumann, U. (1977), *Realizability of
Reynolds-stress turbulence models*, Phys. Fluids 20, 721–725.
"""
struct ReynoldsStressClosure{T, F<:Field, LW, FL, MK, SF, Q, CT, WS, AS} <: AbstractSGSClosure
    τ::NTuple{6,F}
    C1::T; C2::T; C3::T
    C_θ::CT                 # :tke or a number
    c_k::T; c_ε0::T; c_ε1::T; c_l::T; κ::T; surface_factor::T
    Δ::T
    θ₀::T                   # NaN: from the simulation
    g::T                    # NaN: from the simulation
    Ω::NTuple{3,T}
    Ω_auto::Bool
    qv::Q
    l_wall::LW
    first_level::FL
    mask::MK
    surface::SF             # nothing or (u_star, k)
    transport::Bool
    advection::Symbol
    # scratch
    N2::F; ν::F; K2::F
    K::DiffusivityTensor
    S::StressTensor
    tend::NTuple{6,F}
    ws::WS
    adv::AS
    L6::Matrix{T}
    A6::Matrix{T}
    b6::Vector{T}
    snap::NTuple{6,Array{T,3}}
    stats::Vector{T}
    snapstats::Vector{T}
end

const _NRSTATS = 12

function ReynoldsStressClosure(g::RegularGrid{T}; τ = nothing, e0::Real = 0,
                               C1::Real = 1.8, C2::Real = 0.6, C3::Real = 0.5, C_θ = :tke,
                               c_k::Real = 0.1, c_ε0::Real = 0.19, c_ε1::Real = 0.51,
                               c_l::Real = 0.76, κ::Real = 0.4, surface_factor::Real = 3.9,
                               Ω = :auto, wall_distance = nothing, first_level = nothing,
                               mask = nothing, surface = nothing, θ₀ = nothing, g_accel = nothing,
                               qv = :auto, transport::Bool = true, advection::Symbol = :auto) where {T}
    (C_θ === :tke || C_θ isa Real) ||
        throw(ArgumentError("C_θ must be :tke or a number, got $(repr(C_θ))"))
    mkf() = Field(CellCenter(), g)
    τf = if τ === nothing
        fs = ntuple(_ -> mkf(), 6)
        for m in 1:3
            fs[m].data .= T(2 * e0 / 3)
        end
        fs
    else
        length(τ) == 6 || throw(ArgumentError("τ must be six fields (11, 22, 33, 12, 13, 23)"))
        Tuple(τ)
    end
    N = (g.Nx, g.Ny, g.Nz)
    lw = wall_distance === nothing ? nothing : (size(wall_distance) == N ? T(κ) .* T.(wall_distance) :
         throw(DimensionMismatch("wall_distance must be interior-sized $N, got $(size(wall_distance))")))
    fl = first_level === nothing ? nothing : (size(first_level) == N ? Array{Bool}(first_level) :
         throw(DimensionMismatch("first_level must be interior-sized $N")))
    mk = mask === nothing ? nothing : (size(mask) == size(τf[1].data) ? Array{Bool}(mask) :
         throw(DimensionMismatch("mask must be storage-sized $(size(τf[1].data))")))
    Ωt, Ωa = Ω === :auto ? ((zero(T), zero(T), zero(T)), true) :
             (length(Ω) == 3 ? ((T(Ω[1]), T(Ω[2]), T(Ω[3])), false) :
              throw(ArgumentError("Ω must be :auto or a 3-vector")))
    advs = advection === :auto ? (min(g.Hx, g.Hy, g.Hz) >= 3 ? :weno_z5 : :centered) : advection
    ws = transport ? FluxLimiterWorkspace(g) : nothing
    adv = transport ? advection_scheme(advs) : nothing
    Δ = T((g.dx * g.dy * g.dz)^(1 / 3))
    return ReynoldsStressClosure(τf, T(C1), T(C2), T(C3), C_θ isa Real ? T(C_θ) : C_θ,
                                 T(c_k), T(c_ε0), T(c_ε1), T(c_l), T(κ), T(surface_factor), Δ,
                                 θ₀ === nothing ? T(NaN) : T(θ₀),
                                 g_accel === nothing ? T(NaN) : T(g_accel),
                                 Ωt, Ωa, qv, lw, fl, mk, surface, transport, advs,
                                 mkf(), mkf(), mkf(), DiffusivityTensor(g), StressTensor(g),
                                 ntuple(_ -> mkf(), 6), ws, adv,
                                 zeros(T, 6, 6), zeros(T, 6, 6), zeros(T, 6),
                                 ntuple(_ -> similar(τf[1].data), 6),
                                 zeros(T, _NRSTATS), zeros(T, _NRSTATS))
end

Base.show(io::IO, c::ReynoldsStressClosure) =
    print(io, "ReynoldsStressClosure(C1 = $(c.C1), C2 = $(c.C2), C3 = $(c.C3), C_θ = $(repr(c.C_θ)), ",
          "Δ = $(round(c.Δ; sigdigits = 4)) m, transport = $(c.transport))")

sgs_explicit_stress(::ReynoldsStressClosure) = true

_rsm_Ω(c::ReynoldsStressClosure{T}, sim) where {T} =
    c.Ω_auto ? (hasproperty(sim, :has_coriolis) && sim.has_coriolis ?
                (zero(T), zero(T), T(sim.f_coriolis / 2)) : (zero(T), zero(T), zero(T))) : c.Ω
_β(c, sim) = (θ₀ = _θref(c, sim); g = _gref(c, sim); θ₀ > 0 && g > 0 ? g / θ₀ : zero(θ₀))

@inline _τat(τd, I, J, K) = @inbounds (τd[1][I, J, K], τd[2][I, J, K], τd[3][I, J, K],
                                       τd[4][I, J, K], τd[5][I, J, K], τd[6][I, J, K])

# the local coefficients of a cell: (l, r = ε/e, ν_t, c_θ), from e and N²
@inline function _rsm_local(c::ReynoldsStressClosure{T}, e, N2, i, j, k) where {T}
    l, s = tke_length_scale(e, N2, c.Δ; l_wall = _lwall(c.l_wall, i, j, k, T), c_l = c.c_l)
    fac = _first(c.first_level, i, j, k) ? c.surface_factor : one(T)
    cε = tke_dissipation_coefficient(l, c.Δ; c_ε0 = c.c_ε0, c_ε1 = c.c_ε1, factor = fac)
    r = cε * s
    ν, _ = tke_eddy_coefficients(e, l, c.Δ; c_k = c.c_k)
    cθ = daly_harlow_coefficient(e, l, c.Δ, r; C_θ = c.C_θ, c_k = c.c_k)
    return l, r, ν, cθ
end

_fill_stress_halos!(c::ReynoldsStressClosure, bcs) = (foreach(f -> fill_halos!(f, bcs), c.τ); c)

"""
    sgs_coefficients!(f::SGSFields, c::ReynoldsStressClosure, sim)

From the committed stresses and the current ``N^2``: ``ν_t`` into `f.νh`
([`tke_eddy_coefficients`](@ref)), the Daly–Harlow tensor
``K = c_θ τ`` into the closure's `DiffusivityTensor` and its isotropic
part ``\\operatorname{tr}K/3`` into `f.κh`, and the deviatoric edge
stresses ([`stress_edges!`](@ref)) the momentum operator reads. Pure in
the closure's memory (it fills halos and scratch only).
"""
function sgs_coefficients!(f::SGSFields, c::ReynoldsStressClosure{T}, sim) where {T}
    _stratification!(c, sim)
    _fill_stress_halos!(c, _scalar_bcs(sim))
    g = mesh(c.τ[1])
    Hx, Hy, Hz = g.Hx, g.Hy, g.Hz
    τd = map(x -> x.data, c.τ)
    K = c.K
    Kd = (K.K11.data, K.K22.data, K.K33.data, K.K12.data, K.K13.data, K.K23.data)
    nd, kd, n2 = f.νh.data, f.κh.data, c.N2.data
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
    _stress_edges!(c.S, τd, _Edges(c.mask, bounded_directions(sim.bcs), g), g)
    return f
end

function _fill_sgs_halos!(f::SGSFields, c::ReynoldsStressClosure, sim)
    invoke(_fill_sgs_halos!, Tuple{SGSFields, AbstractSGSClosure, Any}, f, c, sim)
    fill_halos!(c.K, sim.bcs)
    return f
end

"""
    add_sgs_momentum!(tu, tv, tw, c::ReynoldsStressClosure, f, sim; stress)

``−∂_jτ^{dev}_{ij}`` with `Operators.add_stress_divergence!` (P55-b) and the
edge stresses [`sgs_coefficients!`](@ref) placed. `stress` is ignored: the
closure's momentum flux is its own stress.
"""
add_sgs_momentum!(tu, tv, tw, c::ReynoldsStressClosure, f::SGSFields, sim; stress::Symbol = :laplacian) =
    (add_stress_divergence!(tu, tv, tw, c.S); nothing)

"""
    add_sgs_scalar!(tϕ, ϕ, c::ReynoldsStressClosure, f, sim)

The Daly–Harlow flux divergence ``∂_i(c_θ τ_{ij}∂_jϕ)`` with
`Operators.add_tensor_diffusion_scalar!` (P55-c; walls from the scalar
boundary conditions).
"""
add_sgs_scalar!(tϕ, ϕ, c::ReynoldsStressClosure, f::SGSFields, sim) =
    add_tensor_diffusion_scalar!(tϕ, ϕ, c.K; bounded = bounded_directions(_scalar_bcs(sim)))

sgs_scalar_tensor(c::ReynoldsStressClosure, f::SGSFields, sim) = c.K

# ---- the commit ----------------------------------------------------------

"""
    advance_sgs!(c::ReynoldsStressClosure, sim, dt)

Integrate the stresses over the accepted step `dt` (once per step, after
the projection):

1. with `transport`: the explicit tendency ``T_{ij}`` of the advection
   (flux-limited for the diagonals) and of ``∇·(2ν_t∇τ_{ij})``, ``ν_t`` of
   ``τ^n``;
2. per cell, ``τ^* = R(τ^n + Δt\\,T)R^T`` with the exact Coriolis rotation
   [`coriolis_rotation`](@ref);
3. ``(I − Δt\\,L)τ^{n+1} = τ^*`` with ``L`` the (linear) source operator of
   [`reynolds_stress_terms`](@ref) at the cell's velocity gradient
   ([`dual_velocity_gradient`](@ref)), ``∇θ_v``, and ``ε/e``, ``c_θ`` of
   ``τ^n``;
4. ``τ^{n+1}`` projected onto the realizable set, the correction booked;
   ``τ = 0`` in solids.
"""
function advance_sgs!(c::ReynoldsStressClosure{T}, sim, dt) where {T}
    Δt = T(dt)
    bcs = sim.bcs
    sbcs = _scalar_bcs(sim)
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    _stratification!(c, sim)
    _fill_stress_halos!(c, sbcs)
    if c.transport
        _rsm_transport!(c, sim, Δt, sbcs)
    end
    _rsm_update!(c, sim, Δt)
    c.surface === nothing || _rsm_surface!(c, sim)
    return c
end

function _rsm_surface!(c::ReynoldsStressClosure{T}, sim) where {T}
    g = mesh(c.τ[1])
    us, kk = c.surface.u_star, c.surface.k
    a11, a22, a13 = reynolds_stress_shear_equilibrium(; C1 = c.C1, C2 = c.C2)
    V = g.dx * g.dy * g.dz
    τd = map(x -> x.data, c.τ)
    ud, vd = sim.u.data, sim.v.data
    de = zero(T)
    @inbounds for j in 1:g.Ny, i in 1:g.Nx
        k = kk[i, j]
        1 <= k <= g.Nz || continue
        I, J, K = g.Hx + i, g.Hy + j, g.Hz + k
        _solid(c.mask, I, J, K) && continue
        u = us[i, j]
        isfinite(u) || continue
        uc = (ud[I, J, K] + ud[I+1, J, K]) / 2
        vc = (vd[I, J, K] + vd[I, J+1, K]) / 2
        sp = sqrt(uc^2 + vc^2)
        sp > 0 || continue
        ca, sa = uc / sp, vc / sp
        u2 = T(u)^2
        es = u2 / a13
        tss, tnn = a11 * es, a22 * es
        new = (tss * ca^2 + tnn * sa^2, tss * sa^2 + tnn * ca^2, a22 * es,
               (tss - tnn) * ca * sa, -u2 * ca, -u2 * sa)
        de += (es - _tke(_τat(τd, I, J, K))) * V
        for m in 1:6
            τd[m][I, J, K] = new[m]
        end
    end
    c.stats[11] += de
    return c
end

function _rsm_viscosity!(c::ReynoldsStressClosure{T}) where {T}
    g = mesh(c.τ[1])
    τd = map(x -> x.data, c.τ)
    nd, n2 = c.ν.data, c.N2.data
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = g.Hx + i, g.Hy + j, g.Hz + k
        e = _tke(_τat(τd, I, J, K))
        nd[I, J, K] = (_solid(c.mask, I, J, K) || !(e > 0)) ? zero(T) :
                      _rsm_local(c, e, n2[I, J, K], i, j, k)[3]
    end
    return c.ν
end

function _rsm_transport!(c::ReynoldsStressClosure, sim, Δt, sbcs)
    _rsm_viscosity!(c)
    fill_halos!(c.ν, sim.bcs)
    c.K2.data .= 2 .* c.ν.data
    for m in 1:6
        t = c.tend[m]
        fill!(t.data, 0)
        if m <= 3
            advect_positive!(t, sim.u, sim.v, sim.w, c.τ[m], Δt;
                             scheme = c.advection, workspace = c.ws, bcs = sbcs)
        else
            c.adv(t, sim.u, sim.v, sim.w, c.τ[m])
        end
        add_eddy_diffusion_scalar!(t, c.τ[m], c.K2; mask = c.mask)
    end
    return c
end

# Gaussian elimination with partial pivoting, A (n×n) x = b in place (b ← x)
function _solve!(A::Matrix{T}, b::Vector{T}) where {T}
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

const _UNIT6 = ntuple(m -> ntuple(n -> Float64(n == m), 6), 6)

"""
    reynolds_stress_step(τ, G, dθ, r, cθ, β, dt; C1 = 1.8, C2 = 0.6, C3 = 0.5) -> τ′

The local source update [`ReynoldsStressClosure`](@ref) applies in every
cell: backward Euler, ``(I − h\\,L)τ_{k+1} = τ_k``, with ``L`` the linear
map of [`reynolds_stress_terms`](@ref) at frozen `r`, `cθ` (no Coriolis).
A fixed point ``τ′ = τ`` is a root of the stress equations.

Backward Euler inverts a *growing* mode (production faster than the
sinks) once ``h λ ≥ 1`` — the update changes sign and a realizable stress
becomes wildly unrealizable. So the step is split into
``n = \\lceil 2\\,dt\\,μ_∞(L) \\rceil`` equal sub-steps when the logarithmic norm
``μ_∞(L) = \\max_i (L_{ii} + Σ_{j≠i}|L_{ij}|)`` — an upper bound on the
growth rate of every mode — is positive, which keeps ``1 − hλ ≥ ½``. Same
operator in every sub-step, so the fixed point is unchanged.
"""
function reynolds_stress_step(τ, G, dθ, r, cθ, β, dt; C1 = 1.8, C2 = 0.6, C3 = 0.5)
    T = float(promote_type(eltype(τ), typeof(dt)))
    τ1, _, _ = _rsm_substeps!(τ, G, dθ, r, cθ, β, dt, C1, C2, C3,
                              zeros(T, 6, 6), zeros(T, 6, 6), zeros(T, 6))
    return τ1
end

# the sub-stepped backward Euler: returns (τ′, τ̄, n) with τ̄ the mean of the
# sub-step results, so Σₖ h ℓ(τₖ) = dt ℓ(τ̄) for the (linear) ½-trace ℓ —
# the exact budget of the update
function _rsm_substeps!(τ, G, dθ, r, cθ, β, dt, C1, C2, C3, L::Matrix{T}, A::Matrix{T},
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
        _solve!(A, b)
        x = @inbounds (b[1], b[2], b[3], b[4], b[5], b[6])
        acc = acc .+ x
    end
    return x, acc ./ n, n
end

function _rsm_update!(c::ReynoldsStressClosure{T}, sim, Δt::T) where {T}
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
            τ1, τb, nsub = _rsm_substeps!(τs, Gc, dθ, r, cθ, β, Δt, C1, C2, C3, c.L6, c.A6, c.b6)
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
        # the budget of the step, before the correction: ½ tr of the
        # (sub-stepped) implicit update is exactly Δt (P + βh₃ − r e) at the
        # mean τb of the sub-step results (τb = τ^{n+1} without sub-steps)
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

"""
    sgs_state(c::ReynoldsStressClosure) / restore_sgs_state!(c, s)

Snapshot of the six stresses and the booked statistics into buffers the
closure owns, and its restoration (a rejected adaptive step leaves them
bit for bit as they were).
"""
function sgs_state(c::ReynoldsStressClosure)
    for m in 1:6
        copyto!(c.snap[m], c.τ[m].data)
    end
    copyto!(c.snapstats, c.stats)
    return c.snap
end
function restore_sgs_state!(c::ReynoldsStressClosure, s)
    for m in 1:6
        copyto!(c.τ[m].data, s[m])
    end
    copyto!(c.stats, c.snapstats)
    return c
end

"""
    reynolds_stress_statistics(c::ReynoldsStressClosure) -> NamedTuple

What [`advance_sgs!`](@ref) booked: `clips` (cell updates projected onto
the realizable set, cumulative) and `clipped_energy` (the SGS energy
``Σ ½ \\operatorname{tr}(τ_p − τ) V`` those projections added, m⁵ s⁻²,
cumulative); `step_clips`, `step_clipped_energy` (the last step's); and
the volume integrals of the last step (m⁵ s⁻³, at ``τ^{n+1}``): `shear`
(``Σ −τ_{ik}G_{ik} V``), `buoyancy` (``Σ βh_3 V``), `dissipation`
(``Σ ε V``, the rate the implicit step applied), `transport`; `steps`;
`surface_energy` (the SGS energy setting the surface value added, m⁵ s⁻²,
cumulative); `step_split_cells` (cells of the last step whose source update
was sub-stepped, see [`reynolds_stress_step`](@ref));
`step_unrealizable_input` (cells of the last step whose stresses were
already unrealizable before the source update — after the transport,
whether the closure's own or a driver's — the share of `step_clips` the
transport causes).
"""
reynolds_stress_statistics(c::ReynoldsStressClosure) =
    (clips = Int(c.stats[1]), clipped_energy = c.stats[2], shear = c.stats[3],
     buoyancy = c.stats[4], dissipation = c.stats[5], transport = c.stats[6],
     steps = Int(c.stats[7]), step_clips = Int(c.stats[8]), step_clipped_energy = c.stats[9],
     surface_energy = c.stats[11], step_split_cells = Int(c.stats[10]),
     step_unrealizable_input = Int(c.stats[12]))

"""
    reynolds_stress_budget(c::ReynoldsStressClosure, sim) -> NamedTuple

Instantaneous volume integrals at the current state (m⁵ s⁻³; pure):
`shear` ``Σ −τ_{ik}G_{ik} V`` with the dual gradient (so it is minus the
resolved-KE tendency of the stress divergence, up to the isotropic part
times the discrete divergence), `buoyancy` ``Σ βh_3 V``, `dissipation`
``Σ ε V``, and `tke` ``Σ e V`` (m⁵ s⁻²). Fills halos.
"""
function reynolds_stress_budget(c::ReynoldsStressClosure{T}, sim) where {T}
    g = mesh(c.τ[1])
    bcs = sim.bcs
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    _stratification!(c, sim)
    V = g.dx * g.dy * g.dz
    idx, idy, idz = 1 / g.dx, 1 / g.dy, 1 / g.dz
    τd = map(x -> x.data, c.τ)
    q = _qv(c, sim)
    qd = q === nothing ? nothing : q.data
    β = T(_β(c, sim))
    E = _Edges(c.mask, bounded_directions(bcs), g)
    sP = zero(T); sB = zero(T); sε = zero(T); se = zero(T)
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = g.Hx + i, g.Hy + j, g.Hz + k
        _solid(c.mask, I, J, K) && continue
        τc = _τat(τd, I, J, K)
        e = _tke(τc)
        se += e
        e > 0 || continue
        _, r, _, cθ = _rsm_local(c, e, c.N2.data[I, J, K], i, j, k)
        Gc = _dual_gradient(sim.u.data, sim.v.data, sim.w.data, I, J, K, idx, idy, idz, E)
        dθ = _θv_gradient(sim.θ.data, qd, I, J, K, idx, idy, idz)
        sP -= τc[1] * Gc[1] + τc[2] * Gc[5] + τc[3] * Gc[9] +
              τc[4] * (Gc[2] + Gc[4]) + τc[5] * (Gc[3] + Gc[7]) + τc[6] * (Gc[6] + Gc[8])
        sB += -β * cθ * (τc[5] * dθ[1] + τc[6] * dθ[2] + τc[3] * dθ[3])
        sε += r * e
    end
    return (shear = sP * V, buoyancy = sB * V, dissipation = sε * V, tke = se * V)
end

"""
    initialize_reynolds_stress!(c::ReynoldsStressClosure, sim; e_min = 0) -> c

Seed the stresses isotropic, ``τ = \\tfrac23 e I``, with ``e`` the local
equilibrium of the TKE closure, [`tke_equilibrium`](@ref) of the cell's
[`dual_strain!`](@ref) shear, ``N^2``, wall cap and ``c_ε`` factor, at least
`e_min`. (``τ = 0`` is a fixed point of the stress equations.)
"""
function initialize_reynolds_stress!(c::ReynoldsStressClosure{T}, sim; e_min::Real = 0) where {T}
    g = mesh(c.τ[1])
    bcs = sim.bcs
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    _stratification!(c, sim)
    D = c.ν                                  # scratch
    dual_strain!(D, sim.u, sim.v, sim.w, bcs)
    @inbounds for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        I, J, K = g.Hx + i, g.Hy + j, g.Hz + k
        for m in 4:6
            c.τ[m].data[I, J, K] = zero(T)
        end
        if _solid(c.mask, I, J, K)
            for m in 1:3
                c.τ[m].data[I, J, K] = zero(T)
            end
            continue
        end
        fac = _first(c.first_level, i, j, k) ? c.surface_factor : one(T)
        eq = tke_equilibrium(max(D.data[I, J, K], zero(T)), c.N2.data[I, J, K], c.Δ;
                             c_k = c.c_k, c_ε0 = c.c_ε0, c_ε1 = c.c_ε1, c_l = c.c_l,
                             l_wall = _lwall(c.l_wall, i, j, k, T), factor = fac)
        e = max(T(eq), T(e_min))
        for m in 1:3
            c.τ[m].data[I, J, K] = 2e / 3
        end
    end
    return c
end
