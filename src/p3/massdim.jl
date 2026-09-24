# P3 part-file: four-regime mass-dimension relation, projected area,
# predicted rime density (issue #2).
#
# The central idea of P3 (Morrison & Milbrandt 2015, hereafter MM15, §2b
# and Appendix): a single ice category whose mass-dimension relation m(D)
# is a piecewise power law whose breakpoints D_th, D_gr, D_cr are
# *diagnosed* from two prognostic ratios,
#
#     rime fraction   F_rim = q_rim / q_i
#     rime density    ρ_rim = q_rim / B_rim
#
# (q_i total ice mass, q_rim rime mass, B_rim rime volume, all per kg of
# air). Because the breakpoints move continuously with F_rim and ρ_rim,
# the bulk properties do too: there is no category jump from "snow" to
# "graupel".

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

"""
    P3IceParameters{T}(; kwargs...)

Fixed parameters of the P3 ice particle model (SI units).

| field | default | meaning |
| :-- | :-- | :-- |
| `ρ_ice`     | 917.0 | bulk density of solid ice, kg m⁻³ |
| `α_va`      | 0.0121 | prefactor of `m = α_va D^β_va`, kg m⁻β |
| `β_va`      | 1.9 | exponent of the aggregate mass-dimension relation |
| `γ_va`      | 0.2285·100^1.88/100² ≈ 0.1315 | prefactor of `A = γ_va D^σ_va`, m^(2-σ) |
| `σ_va`      | 1.88 | exponent of the aggregate area-dimension relation |
| `ρ_rim_min` | 50.0 | lower bound on rime density, kg m⁻³ |
| `ρ_rim_max` | 900.0 | upper bound on rime density, kg m⁻³ |
| `ρ_rim_default` | 400.0 | rime density used when there is no rime volume |
| `δ0`, `C0`  | 5.83, 0.6 | boundary-layer drag constants (Mitchell & Heymsfield 2005) |
| `a0`, `b0`  | 0.0, 0.0 | turbulence correction `a0 X^b0` (Mitchell & Heymsfield 2005 give 1.7e-3, 0.8 for aggregates); off, as in the P3 reference code |
| `T_ref`, `p_ref` | 253.15 K, 6.0e4 Pa | default air state for fall speeds |

`α_va, β_va` are the Brown & Francis (1995) relation for aggregates of
unrimed ice, with the Hogan et al. (2012) correction of the maximum
dimension that MM15 adopt; `γ_va, σ_va` the Mitchell (1996) aggregate
area-dimension relation (0.2285 D^1.88 in cgs, converted to SI). The
rime-density bounds are those of MM15 §2b.
"""
Base.@kwdef struct P3IceParameters{T<:AbstractFloat}
    ρ_ice::T = 917.0
    α_va::T = 0.0121
    β_va::T = 1.9
    γ_va::T = 0.2285 * 100.0^1.88 / 100.0^2
    σ_va::T = 1.88
    ρ_rim_min::T = 50.0
    ρ_rim_max::T = 900.0
    ρ_rim_default::T = 400.0
    δ0::T = 5.83
    C0::T = 0.6
    a0::T = 0.0
    b0::T = 0.0
    T_ref::T = 253.15
    p_ref::T = 6.0e4
end

"Default P3 ice parameters (Float64), see [`P3IceParameters`](@ref)."
const P3_DEFAULT_ICE = P3IceParameters{Float64}()

# -------------------------------------------------------------------------
# Prognostic rime variables -> (F_rim, ρ_rim)
# -------------------------------------------------------------------------

"""
    p3_rime_state(q_i, q_rim, B_rim, p = P3_DEFAULT_ICE) -> (F_rim, ρ_rim, B_rim′)

Diagnose rime fraction and rime density from P3's prognostic variables:
total ice mass `q_i` (kg kg⁻¹), rime mass `q_rim` (kg kg⁻¹) and rime
volume `B_rim` (m³ kg⁻¹),

    F_rim = q_rim / q_i          clamped to [0, 1]
    ρ_rim = q_rim / B_rim        clamped to [ρ_rim_min, ρ_rim_max].

When the rime density has to be clamped, the rime volume is made
consistent with it, `B_rim′ = q_rim / ρ_rim`, and returned so the
caller can write it back (MM15, §2b); otherwise `B_rim′ = B_rim`.

Carrying the rime *volume* rather than a density is what lets density
emerge: when two populations mix, or new rime of density ρ_new is
accreted (`q_rim += dq`, `B_rim += dq/ρ_new`), the resulting density is
the mass-weighted harmonic mean of the parts, exactly, with no rule
needed for how densities combine.

With no ice (`q_i ≤ 0`) or no rime (`q_rim ≤ 0`) the rime fraction is
zero and `ρ_rim_default` is returned for the (then irrelevant) density.

Reference: Morrison & Milbrandt (2015, JAS 72, 287-311), §2b.
"""
function p3_rime_state(q_i::Real, q_rim::Real, B_rim::Real,
                       p::P3IceParameters = P3_DEFAULT_ICE)
    qi, qr, Br = promote(float(q_i), float(q_rim), float(B_rim))
    T = typeof(qi)
    if qi <= 0 || qr <= 0
        return (zero(T), T(p.ρ_rim_default), max(Br, zero(T)))
    end
    F = clamp(qr / qi, zero(T), one(T))
    if Br <= 0
        ρ = T(p.ρ_rim_default)
        return (F, ρ, qr / ρ)
    end
    ρraw = qr / Br
    ρ = clamp(ρraw, T(p.ρ_rim_min), T(p.ρ_rim_max))
    return (F, ρ, ρ == ρraw ? Br : qr / ρ)
end

# -------------------------------------------------------------------------
# Thresholds and graupel density
# -------------------------------------------------------------------------

"""
    P3MassDimension{T}

The diagnosed mass-dimension state of the P3 ice category for one
`(F_rim, ρ_rim)` pair, built by [`p3_mass_dimension`](@ref). Fields:

  * `F_rim`, `ρ_rim` — rime fraction and rime density it was built from;
  * `ρ_g` — density of graupel (the fully rimed regime), kg m⁻³;
  * `ρ_d` — density of the unrimed ("deposited") part of graupel;
  * `D_th`, `D_gr`, `D_cr` — regime thresholds, m (`D_gr = D_cr = Inf`
    for unrimed ice; `D_cr = Inf` when `F_rim = 1`);
  * `params` — the [`P3IceParameters`](@ref).
"""
struct P3MassDimension{T<:AbstractFloat}
    F_rim::T
    ρ_rim::T
    ρ_g::T
    ρ_d::T
    D_th::T
    D_gr::T
    D_cr::T
    params::P3IceParameters{T}
end

# h(L) = 1 - (1-F) g,  g = (r^{β-2} - 1)/((β-2)(r-1)),  r = D_cr/D_gr = e^L.
# With s = 3-β, c = β-2 = 1-s and 1-F = e^{-sL}:
#   h = [c e^L - c - e^{(c-s)L} + e^{-sL}] / (c (e^L - 1)).
# Returned is h/L, with the factor L divided out analytically. The
# numerator's O(1) and O(L) terms cancel exactly, so for small L it is
# summed as L² Σ_{n≥2} L^{n-2}/n! · (c - (c-s)ⁿ + (-s)ⁿ) — no cancellation —
# and with L² never formed, so nothing underflows even for F_rim ~ 1e-300
# (forming L² directly lost all digits below F_rim ≈ 1e-154).
_p3_exprel(x::T) where {T<:AbstractFloat} = x == 0 ? one(T) : expm1(x) / x

function _p3_h_graupel_over_L(L::T, s::T) where {T<:AbstractFloat}
    c = one(T) - s
    if L < T(0.5)
        S = zero(T)
        term = one(T)                    # L^{n-2}/n!; after n = 2 it is 1/2
        for n in 2:40
            term = n == 2 ? term / 2 : term * L / n
            add = term * (c - (c - s)^n + (-s)^n)
            S += add
            abs(add) <= eps(T) * abs(S) && break
        end
        return S / (c * _p3_exprel(L))
    else
        num = c * expm1(L) - exp(-s * L) * expm1(c * L)
        return num / (c * expm1(L) * L)
    end
end

"""
    p3_mass_dimension(F_rim, ρ_rim, p = P3_DEFAULT_ICE) -> P3MassDimension

Solve for the P3 regime thresholds and graupel density given the rime
fraction `F_rim ∈ [0,1]` and rime density `ρ_rim` (kg m⁻³). The four
regimes of the mass-dimension relation (MM15 eqs. 1-6, Appendix) are

| regime | range | `m(D)` | `A(D)` |
| :-- | :-- | :-- | :-- |
| 1 small spherical ice | `D ≤ D_th` | `(π/6) ρ_ice D³` | `(π/4) D²` |
| 2 dense nonspherical (unrimed) | `D_th < D ≤ D_gr` | `α_va D^β_va` | `γ_va D^σ_va` |
| 3 graupel | `D_gr < D ≤ D_cr` | `(π/6) ρ_g D³` | `(π/4) D²` |
| 4 partially rimed | `D > D_cr` | `α_va/(1-F_rim) D^β_va` | `F_rim (π/4)D² + (1-F_rim) γ_va D^σ_va` |

Each threshold is where the neighbouring masses are equal, so `m(D)` is
continuous:

    D_th = (π ρ_ice / (6 α_va))^{1/(β_va-3)}
    D_gr = (6 α_va / (π ρ_g))^{1/(3-β_va)}
    D_cr = (6 α_va / (π ρ_g (1-F_rim)))^{1/(3-β_va)}.

The graupel density is the rime plus the unrimed mass that filled in
between the rime, `ρ_g = F_rim ρ_rim + (1-F_rim) ρ_d`, where `ρ_d` is
the mean density of unrimed aggregates over `[D_gr, D_cr]`,

    ρ_d = 6 α_va (D_cr^{β-2} - D_gr^{β-2}) / (π (β-2)(D_cr - D_gr)).

MM15 and the P3 reference code solve this coupled system by fixed-point
iteration. It is in fact linear in `ρ_g`: with `r = D_cr/D_gr =
(1-F_rim)^{-1/(3-β)}`, `ρ_d = g ρ_g` with `g = (r^{β-2}-1)/((β-2)(r-1))`
independent of `ρ_g`, so

    ρ_g = F_rim ρ_rim / (1 - (1-F_rim) g),

evaluated here in closed form (without cancellation for small `F_rim`).
As `F_rim → 0`, `ρ_g → 2ρ_rim/3` for any `β_va`, and the graupel and
partially rimed regimes collapse onto the aggregate one; at `F_rim = 0`
exactly, `D_gr = D_cr = Inf` and `m(D)` is the unrimed two-regime
relation. At `F_rim = 1`, `ρ_g = ρ_rim` and `D_cr = Inf`.

Since `ρ_g ≤ ρ_rim ≤ ρ_ice`, the thresholds always satisfy
`D_th ≤ D_gr ≤ D_cr`. `ρ_rim` is clamped to `[ρ_rim_min, ρ_rim_max]`.

References: Morrison & Milbrandt (2015, JAS 72, 287-311), §2b and
Appendix; Brown & Francis (1995); Mitchell (1996).
"""
function p3_mass_dimension(F_rim::Real, ρ_rim::Real,
                           p::P3IceParameters{T} = P3_DEFAULT_ICE) where {T}
    (0 <= F_rim <= 1) || throw(DomainError(F_rim, "F_rim must lie in [0, 1]"))
    β = p.β_va
    (β < 3 && β != 2) ||
        throw(ArgumentError("β_va must be < 3 and ≠ 2 (got $β)"))
    F = T(F_rim)
    ρr = clamp(T(ρ_rim), p.ρ_rim_min, min(p.ρ_rim_max, p.ρ_ice))
    s = 3 - β
    D_th = (π * p.ρ_ice / (6 * p.α_va))^(1 / (β - 3))
    if F == 0
        ρg0 = 2 * ρr / 3                # F → 0⁺ limit, where ρ_d → ρ_g
        return P3MassDimension{T}(F, ρr, ρg0, ρg0, D_th, T(Inf), T(Inf), p)
    end
    if F == 1
        ρg = ρr
        D_gr = (6 * p.α_va / (π * ρg))^(1 / s)
        return P3MassDimension{T}(F, ρr, ρg, zero(T), D_th, D_gr, T(Inf), p)
    end
    mlog = -log1p(-F)                          # = s L,  L = log(D_cr / D_gr)
    L = mlog / s
    # ρ_g = F ρ_rim / h = (F/L) ρ_rim / (h/L), with F/L = s F / (-log(1-F))
    ρg = (s * (F / mlog)) * ρr / _p3_h_graupel_over_L(L, s)
    c = β - 2
    g = _p3_exprel(c * L) / _p3_exprel(L)      # ρ_d / ρ_g = (e^{cL}-1)/(c(e^L-1))
    D_gr = (6 * p.α_va / (π * ρg))^(1 / s)
    D_cr = D_gr * exp(L)
    return P3MassDimension{T}(F, ρr, ρg, g * ρg, D_th, D_gr, D_cr, p)
end

"""
    p3_mass_dimension(q_i, q_rim, B_rim, p = P3_DEFAULT_ICE) -> P3MassDimension

Convenience method: diagnose `(F_rim, ρ_rim)` from the prognostic
variables with [`p3_rime_state`](@ref) and build the mass-dimension
state from them.
"""
function p3_mass_dimension(q_i::Real, q_rim::Real, B_rim::Real,
                           p::P3IceParameters = P3_DEFAULT_ICE)
    F, ρ, _ = p3_rime_state(q_i, q_rim, B_rim, p)
    return p3_mass_dimension(F, ρ, p)
end

# -------------------------------------------------------------------------
# m(D), A(D)
# -------------------------------------------------------------------------

"""
    p3_regime(D, md::P3MassDimension) -> Int

Mass-dimension regime of a particle of maximum dimension `D` (m):
`1` small spherical ice, `2` dense nonspherical unrimed ice, `3`
graupel, `4` partially rimed ice. A particle exactly at a threshold is
assigned to the smaller-size regime. See [`p3_mass_dimension`](@ref).
"""
function p3_regime(D::Real, md::P3MassDimension)
    D <= md.D_th && return 1
    D <= md.D_gr && return 2
    D <= md.D_cr && return 3
    return 4
end

"""
    p3_particle_mass(D, md::P3MassDimension) -> kg

Mass of an ice particle of maximum dimension `D` (m) under the P3
four-regime mass-dimension relation (table in
[`p3_mass_dimension`](@ref)). Continuous in `D` by construction of the
thresholds, and continuous in `F_rim` and `ρ_rim`.

Reference: Morrison & Milbrandt (2015), §2b.
"""
function p3_particle_mass(D::Real, md::P3MassDimension{T}) where {T}
    p = md.params
    r = p3_regime(D, md)
    r == 1 && return T(π) / 6 * p.ρ_ice * D^3
    r == 2 && return p.α_va * D^p.β_va
    r == 3 && return T(π) / 6 * md.ρ_g * D^3
    return p.α_va / (1 - md.F_rim) * D^p.β_va
end

"""
    p3_particle_area(D, md::P3MassDimension) -> m²

Projected area of an ice particle of maximum dimension `D` (m): a disc
`(π/4)D²` for small spheres and graupel, the Mitchell (1996) aggregate
relation `γ_va D^σ_va` for unrimed nonspherical ice, and for partially
rimed ice the rime-fraction weighted blend
`F_rim (π/4)D² + (1-F_rim) γ_va D^σ_va` (MM15, §2b).

Unlike the mass, `A(D)` is *not* continuous at the thresholds (the
disc and aggregate relations are independent fits), exactly as in MM15.
At fixed `D` it therefore jumps when a moving threshold `D_gr(F_rim,
ρ_rim)` or `D_cr` crosses `D`. The bulk integrals over the size
distribution are nevertheless continuous in `F_rim` and `ρ_rim`, since
the jump location moves continuously.
"""
function p3_particle_area(D::Real, md::P3MassDimension{T}) where {T}
    p = md.params
    r = p3_regime(D, md)
    (r == 1 || r == 3) && return T(π) / 4 * D^2
    r == 2 && return p.γ_va * D^p.σ_va
    return md.F_rim * T(π) / 4 * D^2 + (1 - md.F_rim) * p.γ_va * D^p.σ_va
end

# The four mass-dimension segments as (coefficient, exponent, D_lo, D_hi);
# empty segments have D_lo = D_hi and a zero coefficient.
function _p3_mass_segments(md::P3MassDimension{T}) where {T}
    p = md.params
    cs = T(π) / 6 * p.ρ_ice
    cg = T(π) / 6 * md.ρ_g
    cp = md.F_rim < 1 ? p.α_va / (1 - md.F_rim) : zero(T)
    D_cr = md.D_cr
    return ((cs, T(3), zero(T), md.D_th),
            (p.α_va, p.β_va, md.D_th, md.D_gr),
            (isinf(md.D_gr) ? zero(T) : cg, T(3), md.D_gr, md.D_cr),
            (isinf(D_cr) ? zero(T) : cp, p.β_va, D_cr, T(Inf)))
end

# -------------------------------------------------------------------------
# Size-distribution integrals and the slope solve
# -------------------------------------------------------------------------

"""
    p3_mean_mass(μ, λ, md::P3MassDimension) -> kg

Mean particle mass `q_i/N_i` of a gamma size distribution
`N(D) ∝ D^μ e^{-λD}` under the four-regime P3 mass-dimension relation,

    q/N = Σ_regimes c_j ∫_{D_lo,j}^{D_hi,j} D^{b_j} D^μ e^{-λD} dD
          / ∫_0^∞ D^μ e^{-λD} dD,

each term a truncated gamma moment ([`gamma_psd_partial_moment`](@ref)),
so the result is exact up to the incomplete-gamma accuracy. Strictly
decreasing in `λ`, since `m(D)` is increasing in `D`.

Reference: Morrison & Milbrandt (2015), §2b and Appendix.
"""
function p3_mean_mass(μ::Real, λ::Real, md::P3MassDimension{T}) where {T}
    μT, λT = T(μ), T(λ)
    M0 = gamma_psd_moment(one(T), μT, λT, 0)
    s = zero(T)
    for (c, b, D1, D2) in _p3_mass_segments(md)
        c == 0 && continue
        s += c * gamma_psd_partial_moment(one(T), μT, λT, b, D1, D2)
    end
    return s / M0
end

"""
    p3_ice_mass(N0, μ, λ, md::P3MassDimension)

Ice mass content `q_i = ∫ m(D) N(D) dD` of the gamma distribution
`N(D) = N₀ D^μ e^{-λD}` (per kg of air if `N₀` is, kg kg⁻¹; per m³ if
`N₀` is per m³). See [`p3_mean_mass`](@ref).
"""
p3_ice_mass(N0::Real, μ::Real, λ::Real, md::P3MassDimension{T}) where {T} =
    gamma_psd_moment(T(N0), T(μ), T(λ), 0) * p3_mean_mass(μ, λ, md)

"""
    p3_bulk_density(μ, λ, md::P3MassDimension) -> kg m⁻³

Bulk density of the ice population: total mass divided by the total
volume of the circumscribing spheres of diameter `D`,

    ρ_bulk = q_i / ((π/6) ∫ D³ N(D) dD) = (q/N) / ((π/6) M₃/M₀).

Equals `ρ_ice` for a population of small spheres, `ρ_g` for pure
graupel, and falls towards the aggregate density `6α_va D^{β-3}/π` for
large unrimed snow.
"""
function p3_bulk_density(μ::Real, λ::Real, md::P3MassDimension{T}) where {T}
    μT, λT = T(μ), T(λ)
    V = T(π) / 6 * gamma_psd_moment(one(T), μT, λT, 3) /
        gamma_psd_moment(one(T), μT, λT, 0)
    return p3_mean_mass(μT, λT, md) / V
end

"""
    p3_slope(q_i, N_i, md::P3MassDimension; μ = 0, λ_min = 1.0, λ_max = 1e8) -> λ

Slope `λ` (m⁻¹) of the gamma distribution with shape `μ` whose mean
particle mass under the P3 mass-dimension relation equals `q_i/N_i`,
i.e. the root of `p3_mean_mass(μ, λ, md) = q_i/N_i`.

Because `m(D)` is piecewise, there is no closed form (unlike the
single power law case, [`gamma_psd_slope`](@ref)); the mean mass is
strictly decreasing in `λ`, and the root is bracketed in
`[λ_min, λ_max]` and found by Illinois-modified regula falsi in `log λ`
to ~1e-14 relative. If `q_i/N_i` lies outside the range spanned by the
bracket the nearer bound is returned; P3 then adjusts `N_i` to be
consistent with the limited `λ` (MM15, §2b), which is the caller's job.

The shape parameter is held fixed here. MM15 diagnose `μ` from `λ`
(Heymsfield 2003); a caller doing so iterates `μ`, `λ` jointly.

Reference: Morrison & Milbrandt (2015), §2b.
"""
function p3_slope(q_i::Real, N_i::Real, md::P3MassDimension{T};
                  μ::Real = 0, λ_min::Real = 1.0, λ_max::Real = 1e8) where {T}
    (q_i > 0 && N_i > 0) ||
        throw(DomainError((q_i, N_i), "q_i and N_i must be positive"))
    target = log(T(q_i) / T(N_i))
    f(y) = log(p3_mean_mass(μ, exp(y), md)) - target      # decreasing in y
    a, b = log(T(λ_min)), log(T(λ_max))
    fa, fb = f(a), f(b)
    fa <= 0 && return exp(a)          # even the widest PSD is too light
    fb >= 0 && return exp(b)          # even the narrowest is too heavy
    side = 0
    y = a
    for _ in 1:200
        y = (a * fb - b * fa) / (fb - fa)
        fy = f(y)
        if fy == 0
            break
        elseif fy > 0
            a, fa = y, fy
            side == -1 && (fb /= 2)
            side = -1
        else
            b, fb = y, fy
            side == 1 && (fa /= 2)
            side = 1
        end
        abs(b - a) <= 4eps(T) * max(abs(a), abs(b)) && break
    end
    return exp(y)
end

"""
    p3_size_distribution(q_i, N_i, md::P3MassDimension; μ = 0, kwargs...) -> (N0, μ, λ)

Gamma size distribution parameters consistent with ice mass `q_i` and
number `N_i`: `λ` from [`p3_slope`](@ref) (keyword arguments are passed
on), then `N₀ = N_i λ^{μ+1}/Γ(μ+1)` ([`gamma_psd_intercept`](@ref)).
"""
function p3_size_distribution(q_i::Real, N_i::Real, md::P3MassDimension{T};
                              μ::Real = 0, kwargs...) where {T}
    λ = p3_slope(q_i, N_i, md; μ = μ, kwargs...)
    return (gamma_psd_intercept(T(N_i), T(μ), λ), T(μ), λ)
end

# -------------------------------------------------------------------------
# Fall speed
# -------------------------------------------------------------------------

"""
    p3_air_viscosity(T) -> Pa s

Dynamic viscosity of dry air from Sutherland's law,

    η(T) = 1.458e-6 T^{3/2} / (T + 110.4),

(U.S. Standard Atmosphere 1976), 1.716e-5 Pa s at 273.15 K. Only
weakly dependent on humidity, which is neglected, as in MM15.
"""
p3_air_viscosity(T::Real) = 1.458e-6 * float(T)^1.5 / (float(T) + 110.4)

# Reference air state for fall speeds (P3IceParameters.T_ref, p_ref).
_p3_ref_air(p::P3IceParameters) =
    (air_density(p.T_ref, p.p_ref), p3_air_viscosity(p.T_ref))

"""
    p3_fall_speed(D, md::P3MassDimension; ρ_air, η_air) -> m s⁻¹

Still-air terminal fall speed of a P3 ice particle of maximum dimension
`D` (m), from its regime mass `m(D)` and projected area `A(D)` by the
boundary-layer drag theory of Mitchell & Heymsfield (2005, JAS 62,
1637-1644), the formulation used to build the P3 lookup tables
(`create_p3_lookupTable_1.f90`):

    X  = 2 m g ρ_a D² / (A η²)                            (Best number)
    Re = δ₀²/4 · [ (1 + 4√X/(δ₀²√C₀))^{1/2} - 1 ]² - a₀ X^{b₀}
    V  = η Re / (ρ_a D),

with `δ₀ = 5.83`, `C₀ = 0.6`. The turbulence correction `a₀ X^{b₀}`
is **off by default** (`a₀ = b₀ = 0`), as in the P3 reference
lookup-table code ("neglect turbulent correction for aggregates"). MH05
fitted `a₀ = 1.7e-3`, `b₀ = 0.8` to aggregates; applied to dense
particles it outgrows the first term (at `X ≈ 4e9`), so `V(D)` of
900 kg m⁻³ graupel peaks near 2 cm and reaches zero before 5 cm. Set
`a0`, `b0` in [`P3IceParameters`](@ref) to use it for snow only. Air density `ρ_air` (kg m⁻³) and dynamic
viscosity `η_air` (Pa s) default to 253.15 K, 600 hPa
(`air_density` from `MoistThermodynamics`, [`p3_air_viscosity`](@ref)).
Gravity is `MoistThermodynamics.DEFAULT_CONSTANTS.grav`.

In the Stokes limit (`X → 0`) the formula gives `Re = X/(δ₀²C₀)`,
17.7 % above the exact sphere result `X/24`, a known property of the
MH05 boundary-layer fit; 10-50 μm ice spheres fall within 15.8 % of
Stokes' law (measured in the tests). Unrimed aggregates of 0.5-10 mm
fall within 12.2 % of the Locatelli & Hobbs (1974) fit `0.8 D^0.16`
(D in mm) at 1.2 kg m⁻³. With a nonzero `a₀`, `Re` is floored at zero
where the correction would make it negative.

`m(D)` is continuous but `A(D)` jumps at the thresholds, so `V(D)`
jumps there too, as in MM15. The bulk speeds
([`p3_bulk_fall_speed`](@ref)) are continuous in `F_rim` and `ρ_rim`.
"""
function p3_fall_speed(D::Real, md::P3MassDimension{T};
                       ρ_air::Real = _p3_ref_air(md.params)[1],
                       η_air::Real = _p3_ref_air(md.params)[2]) where {T}
    D > 0 || return zero(T)
    DT = T(D)                 # work in the precision of md (D may be Float64)
    m = p3_particle_mass(DT, md)
    A = p3_particle_area(DT, md)
    return _p3_vt(DT, m, A, T(ρ_air), T(η_air), md.params)
end

function _p3_vt(D::T, m::T, A::T, ρa::T, η::T, p::P3IceParameters) where {T}
    g = T(DEFAULT_CONSTANTS.grav)
    X = 2 * m * g * ρa * D^2 / (A * η^2)
    δ0, C0 = T(p.δ0), T(p.C0)
    c1 = 4 / (δ0^2 * sqrt(C0))
    # (√(1+y) - 1) = y / (√(1+y) + 1): no cancellation at small X
    y = c1 * sqrt(X)
    Re = δ0^2 / 4 * (y / (sqrt(1 + y) + 1))^2 - T(p.a0) * X^T(p.b0)
    return η * max(Re, zero(T)) / (ρa * D)
end

# Normalised number-weighted integral E_N[f] = ∫ f(D) D^μ e^{-λD} dD / ∫ D^μ e^{-λD} dD
# by composite 8-point Gauss-Legendre in x = λD, with panel edges at the
# regime thresholds (where V(D) has a jump or a kink), geometric panels
# towards x = 0 and unit panels out to where the D³·D^μ tail is < 1e-30.
# The Gauss-Legendre rule is RadiationCore's (one source of truth).
const _P3_GL8 = _gauss_legendre(8)

function _p3_psd_expectation(f, μ::T, λ::T, md::P3MassDimension{T}) where {T}
    a = μ + 4
    xmax = a + 20 * sqrt(a) + 40
    edges = T[0]
    for k in 40:-1:1
        push!(edges, T(2.0)^(-k))
    end
    append!(edges, T(1):T(1):ceil(xmax))
    for Db in (md.D_th, md.D_gr, md.D_cr)
        xb = λ * Db
        0 < xb < xmax && push!(edges, xb)
    end
    sort!(edges)
    unique!(edges)
    xs, ws = _P3_GL8
    lg = lgamma_lanczos(μ + 1)
    acc = nothing
    for i in 1:(length(edges) - 1)
        x1, x2 = edges[i], edges[i + 1]
        h = (x2 - x1) / 2
        mid = (x1 + x2) / 2
        for j in eachindex(xs)
            x = mid + h * T(xs[j])
            wt = h * T(ws[j]) * exp(μ * log(x) - x - lg)
            v = f(x / λ) .* wt
            acc = acc === nothing ? v : acc .+ v
        end
    end
    return acc
end

"""
    p3_bulk_fall_speed(μ, λ, md::P3MassDimension; ρ_air, η_air) -> (V_N, V_M)

Number- and mass-weighted mean fall speeds (m s⁻¹) of the gamma size
distribution `N(D) ∝ D^μ e^{-λD}`,

    V_N = ∫ V(D) N(D) dD / ∫ N(D) dD
    V_M = ∫ V(D) m(D) N(D) dD / ∫ m(D) N(D) dD,

the sedimentation velocities of `N_i` and of `q_i` (and of `q_rim`,
`B_rim`, which fall with the mass; MM15 §2d). `V(D)` from
[`p3_fall_speed`](@ref) is not a power law, so the integrals are taken
by composite Gauss-Legendre quadrature split at the regime thresholds
(P3 itself tabulates the same integrals offline). The mass-weighted
integral of `m(D)` by the same quadrature reproduces the analytic
truncated-moment value [`p3_mean_mass`](@ref) to 7e-15 (tests).
"""
function p3_bulk_fall_speed(μ::Real, λ::Real, md::P3MassDimension{T};
                            ρ_air::Real = _p3_ref_air(md.params)[1],
                            η_air::Real = _p3_ref_air(md.params)[2]) where {T}
    ρa, η = T(ρ_air), T(η_air)
    p = md.params
    VN, VM, M = _p3_psd_expectation(T(μ), T(λ), md) do D
        m = p3_particle_mass(D, md)
        v = _p3_vt(D, m, p3_particle_area(D, md), ρa, η, p)
        (v, v * m, m)
    end
    return (VN, VM / M)
end
