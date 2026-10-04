# =========================================================================
# Era VI, Phase 55 (P55-d, #176) — anisotropic minimum-dissipation (AMD)
# and Vreman SGS closures.
# =========================================================================
#
# Included into `module Operators` after sgs_closures.jl. Two closures of the
# P54-a contract that are built for anisotropic grids (they weight each
# direction with its own spacing δ_k instead of a single Δ = (ΔxΔyΔz)^{1/3})
# and that vanish in laminar flow (uniform flow, pure shear; AMD also for
# solid-body rotation), where the Smagorinsky ν_t = (C_s Δ)²|S| does not.
#
# Both read the one velocity-gradient stencil, `_velocity_gradient_at`
# (Operators.jl), so there is no second gradient discretisation. The θ
# gradient is the centred cell-centre difference over two cells (exact for
# a linear field, like the off-diagonal velocity gradients).

# θ gradient at the centre of cell (i, j, k): centred differences.
@inline function _scalar_gradient_at(θ, i, j, k, idx, idy, idz)
    @inbounds begin
        gx = (θ[i+1,j,k] - θ[i-1,j,k]) * (0.5 * idx)
        gy = (θ[i,j+1,k] - θ[i,j-1,k]) * (0.5 * idy)
        gz = (θ[i,j,k+1] - θ[i,j,k-1]) * (0.5 * idz)
    end
    return (gx, gy, gz)
end

@inline _gsq(g) = g[1][1]^2 + g[1][2]^2 + g[1][3]^2 +
                  g[2][1]^2 + g[2][2]^2 + g[2][3]^2 +
                  g[3][1]^2 + g[3][2]^2 + g[3][3]^2

# Σ_k δ_k² a_k b_k (the scaled-gradient product (δ_k∂_k a)(δ_k∂_k b)).
@inline _sdot(a, b, d2) = d2[1] * a[1] * b[1] + d2[2] * a[2] * b[2] + d2[3] * a[3] * b[3]

"""
    _amd_point(g, gθ, d2, βg) -> (ν_num, ν_den, κ_num, κ_den)

The AMD numerators and denominators at one point from the velocity gradient
`g[a][b] = ∂u_a/∂x_b`, the θ gradient `gθ`, the squared spacings `d2 =
(δx², δy², δz²)` and the buoyancy parameter `βg = g/θ₀`:

    ν_num = −(δ_k∂_k u_i)(δ_k∂_k u_j)S_ij + βg (δ_k∂_k w)(δ_k∂_k θ),  ν_den = ∂_l u_m ∂_l u_m
    κ_num = −(δ_k∂_k u_i)(δ_k∂_k θ)∂_i θ,                           κ_den = ∂_l θ ∂_l θ
"""
@inline function _amd_point(g, gθ, d2, βg)
    S12 = 0.5 * (g[1][2] + g[2][1])
    S13 = 0.5 * (g[1][3] + g[3][1])
    S23 = 0.5 * (g[2][3] + g[3][2])
    # (δ_k∂_k u_i)(δ_k∂_k u_j) S_ij, symmetric in (i, j)
    P = _sdot(g[1], g[1], d2) * g[1][1] + _sdot(g[2], g[2], d2) * g[2][2] +
        _sdot(g[3], g[3], d2) * g[3][3] +
        2 * (_sdot(g[1], g[2], d2) * S12 + _sdot(g[1], g[3], d2) * S13 +
             _sdot(g[2], g[3], d2) * S23)
    B = βg * _sdot(g[3], gθ, d2)
    κn, κd = _amd_scalar_point(g, gθ, d2)
    return (-P + B, _gsq(g), κn, κd)
end

# The AMD scalar numerator −(δ_k∂_k u_i)(δ_k∂_k ϕ)∂_i ϕ and denominator ∂_lϕ ∂_lϕ
# for any scalar ϕ (Abkar, Bae & Moin 2016): homogeneous of degree 2 in ϕ,
# so κ_t is unchanged by ϕ → aϕ + b.
@inline function _amd_scalar_point(g, gϕ, d2)
    Q = _sdot(g[1], gϕ, d2) * gϕ[1] + _sdot(g[2], gϕ, d2) * gϕ[2] + _sdot(g[3], gϕ, d2) * gϕ[3]
    return (-Q, gϕ[1]^2 + gϕ[2]^2 + gϕ[3]^2)
end

@inline _cross2(a, b) = (a[2] * b[3] - a[3] * b[2])^2 + (a[3] * b[1] - a[1] * b[3])^2 +
                        (a[1] * b[2] - a[2] * b[1])^2

"""
    _vreman_point(g, d2) -> (B_β, α_ij α_ij)

Vreman's (2004) invariants at one point: with `α_ij = ∂u_j/∂x_i` and
`β_ij = Σ_m Δ_m² α_mi α_mj`, `B_β = β₁₁β₂₂ − β₁₂² + β₁₁β₃₃ − β₁₃² + β₂₂β₃₃ − β₂₃²`,
the sum of the principal 2 × 2 minors of `β`. Evaluated in the equivalent
Cauchy–Binet form `B_β = Σ_{m<n} Δ_m²Δ_n² |a_m × a_n|²`, with `a_m = ∂_m u`
the derivative of the velocity vector in direction `m` (row `m` of `α`):
non-negative by construction, and exactly zero (not round-off) whenever the
velocity varies in one direction only, e.g. any shear flow `u(z), v(z)`.
"""
@inline function _vreman_point(g, d2)
    a1 = (g[1][1], g[2][1], g[3][1])
    a2 = (g[1][2], g[2][2], g[3][2])
    a3 = (g[1][3], g[2][3], g[3][3])
    Bβ = d2[1] * d2[2] * _cross2(a1, a2) + d2[1] * d2[3] * _cross2(a1, a3) +
         d2[2] * d2[3] * _cross2(a2, a3)
    return (Bβ, _gsq(g))
end

function _check_regular(u, name)
    m = mesh(u)
    m isa RegularGrid || throw(ArgumentError("$name: RegularGrid only, got $(typeof(m))"))
    return m
end

"""
    compute_amd_coefficients!(νt, κt, u, v, w, θ; C = 0.3, Pr_t = 1/3,
                              buoyancy = 0.0) -> (νt, κt)

Eddy viscosity and diffusivity of the **anisotropic minimum-dissipation**
(AMD) model of Rozema, Bae, Moin & Verstappen (2015), with the scalar
diffusivity of Abkar, Bae & Moin (2016) and the buoyancy term of Abkar &
Moin (2017), on the interior of the cell-centred fields `νt`, `κt`:

```math
ν_t = C\\,\\frac{\\max\\!\\left(0,\\; -(δ_k ∂_k u_i)(δ_k ∂_k u_j) S_{ij}
      + \\frac{g}{θ_0}(δ_k ∂_k w)(δ_k ∂_k θ)\\right)}{∂_l u_m\\, ∂_l u_m},
\\qquad
κ_t = C\\,\\frac{\\max\\!\\left(0,\\; -(δ_k ∂_k u_i)(δ_k ∂_k θ)\\, ∂_i θ\\right)}{∂_l θ\\, ∂_l θ},
```

with `δ_k = (Δx, Δy, Δz)` (summation over the repeated `k` inside each
product) and `S_ij` the resolved strain rate. Where `∂_l θ ∂_l θ = 0` (no
resolved θ gradient) `κ_t = ν_t/Pr_t`; where `∂_l u_m ∂_l u_m = 0`,
`ν_t = 0`.

* `C` — the modified Poincaré constant; 0.3 for a second-order central
  discretisation (Rozema et al. 2015, their decaying-turbulence tests;
  1/12 for a spectral method, 0.212 for fourth order), the value of this
  code's stencil.
* `buoyancy` — `g/θ₀` [m s⁻² K⁻¹], a number or a function `k -> g/θ̄(z_k)`
  of the interior level; `0` switches the buoyancy term off. The term is
  the gradient-model estimate of the subfilter buoyancy flux: in stable
  stratification it lowers `ν_t` and switches it off where it cancels the
  shear production, which is AMD's own stratification response (no
  separate Richardson-number correction is applied).

The velocity gradient is [`_velocity_gradient_at`](@ref), the stencil of
every Era VI closure; the θ gradient is the centred difference over two
cells. Halos of `u, v, w, θ` must be filled.

References: Rozema, Bae, Moin & Verstappen (2015), *Phys. Fluids* 27,
085107; Abkar, Bae & Moin (2016), *Phys. Rev. Fluids* 1, 041701(R); Abkar &
Moin (2017), *Boundary-Layer Meteorol.* 165, 405 (eq. 10–11 as restated by
Gadde, Stieren & Stevens 2021, *Boundary-Layer Meteorol.* 178, 363).
"""
function compute_amd_coefficients!(νt::Field, κt::Field, u::Field, v::Field, w::Field, θ::Field;
                                   C::Real = 0.3, Pr_t::Real = 1/3, buoyancy = 0.0)
    g = _check_regular(u, "compute_amd_coefficients!")
    _amd_kernel!(νt.data, κt.data, u.data, v.data, w.data, θ.data, g,
                 float(C), float(Pr_t), buoyancy)
    return νt, κt
end

_βg(b::Real, k) = b
_βg(b, k) = b(k)

function _amd_kernel!(nd, kd, ud, vd, wd, θd, g, C, Pr, buoy)
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

"""
    compute_amd_scalar_diffusivity!(κt, νt, u, v, w, ϕ; C = 0.3, Pr_t = 1/3) -> κt

The AMD eddy diffusivity of a scalar `ϕ` (Abkar, Bae & Moin 2016) on the
interior of `κt`: the `κ_t` of [`compute_amd_coefficients!`](@ref) with `ϕ`
in place of θ,

```math
κ_t = C\\,\\frac{\\max\\!\\left(0,\\; -(δ_k ∂_k u_i)(δ_k ∂_k ϕ)\\, ∂_i ϕ\\right)}{∂_l ϕ\\, ∂_l ϕ},
```

and `κ_t = ν_t/Pr_t` (from the interior of `νt`) where `ϕ` has no resolved
gradient. Each scalar gets its own diffusivity: the model bounds the
subfilter variance of that scalar, so `qv` is not diffused with θ's `κ_t`.
`κ_t` is unchanged under `ϕ → aϕ + b` (`a ≠ 0`). Halos of `u, v, w, ϕ`
must be filled.
"""
function compute_amd_scalar_diffusivity!(κt::Field, νt::Field, u::Field, v::Field, w::Field,
                                         ϕ::Field; C::Real = 0.3, Pr_t::Real = 1/3)
    g = _check_regular(u, "compute_amd_scalar_diffusivity!")
    _amd_scalar_kernel!(κt.data, νt.data, u.data, v.data, w.data, ϕ.data, g,
                        float(C), float(Pr_t))
    return κt
end

function _amd_scalar_kernel!(kd, nd, ud, vd, wd, ϕd, g, C, Pr)
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

"""
    compute_vreman_νt!(νt, u, v, w, C_s = 0.17) -> νt

The eddy viscosity of Vreman (2004) on the interior of `νt`:

```math
ν_t = c\\,\\sqrt{\\frac{B_β}{α_{ij} α_{ij}}}, \\qquad
α_{ij} = ∂_i u_j,\\quad β_{ij} = Δ_m^2 α_{mi} α_{mj},\\quad
B_β = β_{11}β_{22} - β_{12}^2 + β_{11}β_{33} - β_{13}^2 + β_{22}β_{33} - β_{23}^2,
```

`c = 2.5 C_s²` (Vreman's eq. 9 relation to the Smagorinsky constant;
`C_s = 0.17` gives `c = 0.0723`), `Δ_m = (Δx, Δy, Δz)`; `ν_t = 0` where
`α_ij α_ij = 0`. `B_β` vanishes for every flow whose velocity varies in
one direction only, e.g. the shear flow `u(z), v(z)` (the laminar
switch-off); it is evaluated in a Cauchy–Binet form that makes this zero
exact. The
gradient is [`_velocity_gradient_at`](@ref). Halos of `u, v, w` must be
filled.

Reference: Vreman, A. W. (2004), *An eddy-viscosity subgrid-scale model
for turbulent shear flow: algebraic theory and applications*, Phys.
Fluids 16, 3670.
"""
function compute_vreman_νt!(νt::Field, u::Field, v::Field, w::Field, C_s::Real = 0.17)
    g = _check_regular(u, "compute_vreman_νt!")
    _vreman_kernel!(νt.data, u.data, v.data, w.data, g, 2.5 * float(C_s)^2)
    return νt
end

function _vreman_kernel!(nd, ud, vd, wd, g, c)
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

# -------------------------------------------------------------------------
# The closures
# -------------------------------------------------------------------------

"""
    AMDClosure(; C = 0.3, Pr_t = nothing, buoyancy = true)

The anisotropic minimum-dissipation closure
([`compute_amd_coefficients!`](@ref)) as an [`AbstractSGSClosure`](@ref):
`ν_t` and its **own** `κ_t` (not `ν_t/Pr_t`; `Pr_t` is only the fallback
where the resolved θ gradient vanishes, `nothing` → the simulation's
`Pr_t`). With `buoyancy = true` the buoyancy parameter is the core's:
`g/θ₀` on the Boussinesq core (`0` when it was built with `θ₀ = 0`),
`g/θ̄(z)` of the reference state on the anelastic / pseudo-incompressible
cores (`0` with `buoyancy = false`). The θ gradient is that of the
simulation's θ (dry: moisture does not enter the buoyancy term), and
the stored `κ_t` is θ's. Another scalar gets its **own** AMD diffusivity
from its own gradient ([`sgs_tracer_diffusivity!`](@ref) →
[`compute_amd_scalar_diffusivity!`](@ref)); that is how `qv` is diffused
by `step!`, `step_rk3!` and [`step_imex!`](@ref Aeolus.TimeIntegration.step_imex!). Stateless. Do not stack a Richardson-number stability correction on it:
the buoyancy term is its stratification response.

References: Rozema et al. (2015), Abkar, Bae & Moin (2016), Abkar & Moin
(2017); see [`compute_amd_coefficients!`](@ref).
"""
struct AMDClosure{T<:Real, P} <: AbstractSGSClosure
    C::T
    Pr_t::P
    buoyancy::Bool
end
AMDClosure(; C::Real = 0.3, Pr_t = nothing, buoyancy::Bool = true) = AMDClosure(C, Pr_t, buoyancy)

"""
    sgs_buoyancy_parameter(sim) -> βg

`g/θ₀` of a simulation's buoyancy (a number, or a function `k -> g/θ̄(z_k)`
of the interior level on the sound-proof cores); `0` when buoyancy is off.
"""
function sgs_buoyancy_parameter(sim)
    if hasproperty(sim, :ref)                       # anelastic / pseudo-incompressible
        sim.has_buoyancy || return 0.0
        θ̄ = sim.ref.θ̄; gg = sim.g_accel
        return k -> gg / θ̄[k]
    elseif hasproperty(sim, :θ₀)                    # Boussinesq
        return sim.θ₀ == 0 ? 0.0 : sim.g_accel / sim.θ₀
    end
    return 0.0
end

function sgs_coefficients!(f::SGSFields, c::AMDClosure, sim)
    βg = c.buoyancy ? sgs_buoyancy_parameter(sim) : 0.0
    compute_amd_coefficients!(f.νh, f.κh, sim.u, sim.v, sim.w, sim.θ;
                              C = c.C, Pr_t = _prandtl(c.Pr_t, sim), buoyancy = βg)
    return f
end

function sgs_tracer_diffusivity!(κ, c::AMDClosure, f::SGSFields, sim, ϕ)
    compute_amd_scalar_diffusivity!(κ, f.νh, sim.u, sim.v, sim.w, ϕ;
                                    C = c.C, Pr_t = _prandtl(c.Pr_t, sim))
    fill_halos!(κ, sim.bcs)
    return κ
end

"""
    VremanClosure(C_s = 0.17; Pr_t = nothing)

The Vreman (2004) closure ([`compute_vreman_νt!`](@ref), `c = 2.5 C_s²`)
as an [`AbstractEddyViscosityClosure`](@ref): `κ_t = ν_t/Pr_t` (`nothing` →
the simulation's `Pr_t`). Stateless.
"""
struct VremanClosure{T<:Real, P} <: AbstractEddyViscosityClosure
    C_s::T
    Pr_t::P
end
VremanClosure(C_s::Real = 0.17; Pr_t = nothing) = VremanClosure(C_s, Pr_t)

sgs_viscosity!(νt, c::VremanClosure, sim) = compute_vreman_νt!(νt, sim.u, sim.v, sim.w, c.C_s)
turbulent_prandtl(c::VremanClosure, sim) = _prandtl(c.Pr_t, sim)
