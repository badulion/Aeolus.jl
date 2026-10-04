# =========================================================================
# Era VI, Phase 59 (P59-b, #184) — the σ-model SGS closure.
# =========================================================================
#
# Included into `module Operators`, after the closure contract
# (`sgs_closures.jl`). The eddy viscosity is built from the singular values
# σ₁ ≥ σ₂ ≥ σ₃ ≥ 0 of the resolved velocity-gradient tensor g_ij, read from
# the one gradient stencil `_velocity_gradient_at` (P54-a):
#
#     ν_t = (C_σ Δ)² D_σ,     D_σ = σ₃ (σ₁ − σ₂)(σ₂ − σ₃) / σ₁²
#
# (Nicoud, Baya Toda, Cabrit, Bose & Lee 2011). D_σ vanishes whenever g has
# rank ≤ 2 (σ₃ = 0: any two-dimensional flow, pure shear, solid-body
# rotation), whenever two singular values coincide (isotropic expansion,
# axisymmetric strain), and it falls off as z³ at a no-slip wall.

"""
    _sym3_eigvals(G11, G22, G33, G12, G13, G23) -> (λ₁, λ₂, λ₃)

Eigenvalues `λ₁ ≥ λ₂ ≥ λ₃` of the real symmetric 3 × 3 matrix with the given
entries, in closed form: no LAPACK call.

1. The trigonometric solution of the characteristic cubic (Smith 1961; the
   form for the σ-model in Hasan, Basri & Kopp 2010 and Nicoud et al. 2011,
   appendix). It is written in the deviator `G' = G − (tr G/3) I`, so that
   `α₁ = tr(G'²)/6` and `α₂ = det(G')/2` are formed without cancellation and
   `G ∝ I` gives exactly `α₁ = 0` (three equal eigenvalues).
2. One deflation step where the smaller gap is below 1 % of the spread
   `λ₁ − λ₃`. Near a double eigenvalue the cubic determines the *gap*
   between the pair only to `√ε` (the arccos argument is near ±1), far too
   coarse for `D_σ`, a product of such gaps; above 1 % the gap error is
   `≲ ε/(gap/spread)² ≤ 1e-12` relative. The simple eigenvalue (the one
   further from the other two) is well conditioned, so its eigenvector is
   taken as the largest cross product of two rows of `G − λI`, and the pair
   is recomputed from the 2 × 2 compression of `G` on the orthogonal
   complement, `μ± = (p + r)/2 ± ½√((p − r)² + 4q²)`, whose gap is accurate
   to `ε‖G‖`. The simple eigenvalue is replaced by its Rayleigh quotient.

References: Smith, O. K. (1961), *Eigenvalues of a symmetric 3 × 3 matrix*,
Commun. ACM 4, 168; Hasan, K. M., Basri, A. H. & Kopp, J. (2010) as cited by
Nicoud et al. (2011); Kopp, J. (2008), *Efficient numerical diagonalization
of Hermitian 3 × 3 matrices*, Int. J. Mod. Phys. C 19, 523.
"""
@inline function _sym3_eigvals(G11, G22, G33, G12, G13, G23)
    T = promote_type(typeof(G11), typeof(G12))
    m = (G11 + G22 + G33) / 3
    a = G11 - m; b = G22 - m; c = G33 - m
    off2 = G12^2 + G13^2 + G23^2
    α1 = (a^2 + b^2 + c^2 + 2 * off2) / 6
    α1 > 0 || return (m, m, m)
    detd = a * (b * c - G23^2) - G12 * (G12 * c - G23 * G13) + G13 * (G12 * G23 - b * G13)
    α2 = detd / 2
    r = clamp(α2 / (α1 * sqrt(α1)), -one(T), one(T))
    sφ, cφ = sincos(acos(r) / 3)
    s = 2 * sqrt(α1)
    λ1 = m + s * cφ
    λ3 = m - s * (cφ + sqrt(T(3)) * sφ) / 2        # cos(φ + 2π/3)
    λ2 = 3m - λ1 - λ3

    # Well-separated eigenvalues are accurate as they are: the error of a
    # gap from the arccos is ~ε/(gap/spread)², i.e. ≤ 1e-12 relative of the
    # gap when both gaps exceed 1 % of the spread. Otherwise deflate the
    # simple eigenvalue and recompute the close pair.
    min(λ1 - λ2, λ2 - λ3) >= T(0.01) * (λ1 - λ3) && return (λ1, λ2, λ3)
    top = (λ1 - λ2) >= (λ2 - λ3)     # the simple eigenvalue is λ₁ (else λ₃)
    iso = top ? λ1 : λ3
    # rows of G − iso·I
    r1 = (G11 - iso, G12, G13)
    r2 = (G12, G22 - iso, G23)
    r3 = (G13, G23, G33 - iso)
    c12 = _cross3(r1, r2); c13 = _cross3(r1, r3); c23 = _cross3(r2, r3)
    n12 = _dot3(c12, c12); n13 = _dot3(c13, c13); n23 = _dot3(c23, c23)
    v, nv = c12, n12
    n13 > nv && ((v, nv) = (c13, n13))
    n23 > nv && ((v, nv) = (c23, n23))
    nv > 0 || return (λ1, λ2, λ3)
    v = v ./ sqrt(nv)
    # orthonormal basis (e1, e2) of the complement of v
    ax = abs(v[1]); ay = abs(v[2]); az = abs(v[3])
    t = ax <= ay && ax <= az ? (one(T), zero(T), zero(T)) :
        ay <= az            ? (zero(T), one(T), zero(T)) : (zero(T), zero(T), one(T))
    e1 = _cross3(v, t); e1 = e1 ./ sqrt(_dot3(e1, e1))
    e2 = _cross3(v, e1)
    Gv  = _symmul3(G11, G22, G33, G12, G13, G23, v)
    Ge1 = _symmul3(G11, G22, G33, G12, G13, G23, e1)
    Ge2 = _symmul3(G11, G22, G33, G12, G13, G23, e2)
    λiso = _dot3(v, Gv)
    p = _dot3(e1, Ge1); q = _dot3(e1, Ge2); rr = _dot3(e2, Ge2)
    h = sqrt((p - rr)^2 + 4q^2) / 2
    μp = (p + rr) / 2 + h
    μm = (p + rr) / 2 - h
    if top
        return (max(λiso, μp), μp, μm)
    else
        return (μp, μm, min(λiso, μm))
    end
end

@inline _cross3(a, b) = (a[2] * b[3] - a[3] * b[2],
                         a[3] * b[1] - a[1] * b[3],
                         a[1] * b[2] - a[2] * b[1])
@inline _dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
@inline _symmul3(G11, G22, G33, G12, G13, G23, x) =
    (G11 * x[1] + G12 * x[2] + G13 * x[3],
     G12 * x[1] + G22 * x[2] + G23 * x[3],
     G13 * x[1] + G23 * x[2] + G33 * x[3])

"""
    sigma_singular_values(g) -> (σ₁, σ₂, σ₃)

Singular values `σ₁ ≥ σ₂ ≥ σ₃ ≥ 0` of the 3 × 3 velocity-gradient tensor
`g` (a tuple of rows, `g[a][b] = ∂u_a/∂x_b`, as returned by
`Operators._velocity_gradient_at`): the square roots of the eigenvalues of
`gᵀg` ([`_sym3_eigvals`](@ref), closed form). The smallest is taken from the
determinant, `σ₃ = |det g| / (σ₁σ₂)`, which is exactly zero for any `g` of
rank ≤ 2 (a two-dimensional flow, pure shear, solid-body rotation) instead of
the `√ε`-sized value the square root of a round-off eigenvalue would give.
Round-off can leave `σ₃` slightly above `σ₂` when the two coincide; the
σ-model clips the resulting negative operator (see [`sigma_operator`](@ref)).

Reference: Nicoud, F., Baya Toda, H., Cabrit, O., Bose, S. & Lee, J. (2011),
*Using singular values to build a subgrid-scale model for large eddy
simulations*, Phys. Fluids 23, 085106.
"""
@inline function sigma_singular_values(g)
    λ1, λ2, _ = _sigma_eigs(g)
    σ1 = sqrt(max(λ1, zero(λ1)))
    σ2 = sqrt(max(λ2, zero(λ2)))
    σ3 = _sigma_smallest(g, σ1, σ2)
    return σ1, σ2, σ3
end

# eigenvalues of gᵀg
@inline function _sigma_eigs(g)
    @inbounds begin
        g11, g12, g13 = g[1]; g21, g22, g23 = g[2]; g31, g32, g33 = g[3]
    end
    G11 = g11^2 + g21^2 + g31^2
    G22 = g12^2 + g22^2 + g32^2
    G33 = g13^2 + g23^2 + g33^2
    G12 = g11 * g12 + g21 * g22 + g31 * g32
    G13 = g11 * g13 + g21 * g23 + g31 * g33
    G23 = g12 * g13 + g22 * g23 + g32 * g33
    return _sym3_eigvals(G11, G22, G33, G12, G13, G23)
end

@inline function _sigma_smallest(g, σ1, σ2)
    @inbounds begin
        g11, g12, g13 = g[1]; g21, g22, g23 = g[2]; g31, g32, g33 = g[3]
    end
    detg = g11 * (g22 * g33 - g23 * g32) - g12 * (g21 * g33 - g23 * g31) +
           g13 * (g21 * g32 - g22 * g31)
    p = σ1 * σ2
    return p > 0 ? abs(detg) / p : zero(p)
end

"""
    _sigma_operator_raw(g) -> D

The σ-model differential operator `σ₃(σ₁ − σ₂)(σ₂ − σ₃)/σ₁²` (s⁻¹) *before*
the clip at zero. `σ₁ − σ₂` is formed as `(λ₁ − λ₂)/(σ₁ + σ₂)` from the
accurate eigenvalue gap. Where `σ₂ = σ₃` to round-off the factor `σ₂ − σ₃`
can come out a few ulp negative; [`sigma_operator`](@ref) clips that.
"""
@inline function _sigma_operator_raw(g)
    λ1, λ2, _ = _sigma_eigs(g)
    σ1 = sqrt(max(λ1, zero(λ1)))
    σ1 > 0 || return zero(σ1)
    σ2 = sqrt(max(λ2, zero(λ2)))
    σ3 = _sigma_smallest(g, σ1, σ2)
    d12 = max(λ1 - λ2, zero(λ1)) / (σ1 + σ2)
    return σ3 * d12 * (σ2 - σ3) / σ1^2
end

"""
    sigma_operator(g) -> D_σ ≥ 0

The σ-model operator ``D_σ = σ_3(σ_1 − σ_2)(σ_2 − σ_3)/σ_1^2`` (s⁻¹) of the
velocity-gradient tensor `g` (tuple of rows, `g[a][b] = ∂u_a/∂x_b`), clipped
at zero against round-off where singular values coincide. Mathematically
`0 ≤ D_σ ≤ σ₃ ≤ |g|`, with `D_σ = 0` exactly when `σ₃ = 0` (rank ≤ 2: any
2-D flow, pure shear, solid-body rotation) or when two singular values are
equal (isotropic expansion or contraction, axisymmetric strain); near a
no-slip wall `D_σ = O(z³)` (Nicoud et al. 2011, Sec. II).
"""
@inline sigma_operator(g) = max(_sigma_operator_raw(g), zero(g[1][1]))

"""
    compute_sigma_νt!(νt, u, v, w, C_σ = 1.35) -> νt

The σ-model eddy viscosity at cell centres,

```math
ν_t = (C_σ Δ)^2 \\, \\frac{σ_3 (σ_1 − σ_2)(σ_2 − σ_3)}{σ_1^2}, \\qquad Δ = (Δx Δy Δz)^{1/3},
```

with ``σ_1 ≥ σ_2 ≥ σ_3`` the singular values of the resolved gradient
``g_{ij} = ∂u_i/∂x_j`` from `Operators._velocity_gradient_at` (the one
gradient stencil of the SGS closures), computed in closed form
([`sigma_singular_values`](@ref)), and clipped at ``ν_t ≥ 0``. Halos of
`u, v, w` must be filled. ``C_σ = 1.35`` is the value of Nicoud et al.
(2011), calibrated in decaying isotropic turbulence.

Reference: Nicoud, F., Baya Toda, H., Cabrit, O., Bose, S. & Lee, J. (2011),
*Using singular values to build a subgrid-scale model for large eddy
simulations*, Phys. Fluids 23, 085106.
"""
compute_sigma_νt!(νt, u, v, w, C_σ::Real = 1.35) =
    _compute_sigma_νt!(νt, u, v, w, C_σ, mesh(u))

function _compute_sigma_νt!(νt::Field{CellCenter}, u::Field{FaceX}, v::Field{FaceY},
                            w::Field{FaceZ}, C_σ::Real, g::RegularGrid)
    idx = 1 / g.dx; idy = 1 / g.dy; idz = 1 / g.dz
    Δ = (g.dx * g.dy * g.dz)^(1 / 3)
    coef = (C_σ * Δ)^2
    ii, jj, kk = interior(g)
    ud, vd, wd, nd = u.data, v.data, w.data, νt.data
    @inbounds for k in kk, j in jj, i in ii
        G = _velocity_gradient_at(ud, vd, wd, i, j, k, idx, idy, idz)
        nd[i, j, k] = coef * sigma_operator(G)
    end
    return νt
end

"""
    SigmaModel(C_σ = 1.35; Pr_t = nothing)

The σ-model of Nicoud, Baya Toda, Cabrit, Bose & Lee (2011) as an SGS
closure of the Era VI contract (an [`AbstractEddyViscosityClosure`](@ref):
`ν_t` from [`compute_sigma_νt!`](@ref), `κ_t = ν_t/Prₜ`; `Pr_t = nothing`
takes the simulation's). Stateless; pass it as `sgs = SigmaModel()` to any
core.

Unlike Smagorinsky, its eddy viscosity vanishes in laminar flow that the
Smagorinsky ``|S|`` sees as strained: pure shear (so a laminar Couette or
Poiseuille flow keeps its molecular wall stress), solid-body rotation, any
two-dimensional flow (so a 2-D Kelvin–Helmholtz layer grows at its
molecular rate), isotropic expansion and axisymmetric strain; and it falls
off as ``z^3`` at a no-slip wall without a damping function. Theory and the
measured numbers: *Smagorinsky-Lilly SGS*, section "σ-model".

Reference: Nicoud, F., Baya Toda, H., Cabrit, O., Bose, S. & Lee, J. (2011),
*Using singular values to build a subgrid-scale model for large eddy
simulations*, Phys. Fluids 23, 085106.
"""
struct SigmaModel{T<:Real, P} <: AbstractEddyViscosityClosure
    C_σ::T
    Pr_t::P
end
SigmaModel(C_σ::Real = 1.35; Pr_t = nothing) = SigmaModel(C_σ, Pr_t)

sgs_viscosity!(νt, c::SigmaModel, sim) = compute_sigma_νt!(νt, sim.u, sim.v, sim.w, c.C_σ)
turbulent_prandtl(c::SigmaModel, sim) = _prandtl(c.Pr_t, sim)
laminar_consistent(::SigmaModel) = true      # zero in shear, rotation, 2-D flow (P59-e)
