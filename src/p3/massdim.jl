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
| `a0`, `b0`  | 1.7e-3, 0.8 | aggregate drag correction (Mitchell & Heymsfield 2005) |
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
    a0::T = 1.7e-3
    b0::T = 0.8
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
# The numerator's O(1) and O(L) terms cancel exactly, so for small L it is
# summed as Σ_{n≥2} Lⁿ/n! · (c - (c-s)ⁿ + (-s)ⁿ) — no cancellation.
function _p3_h_graupel(L::T, s::T) where {T<:AbstractFloat}
    c = one(T) - s
    if L < T(0.5)
        num = zero(T)
        term = L                         # Lⁿ/n!, starting at n = 1
        for n in 2:40
            term *= L / n
            add = term * (c - (c - s)^n + (-s)^n)
            num += add
            abs(add) <= eps(T) * abs(num) && break
        end
    else
        num = c * expm1(L) - exp(-s * L) * expm1(c * L)
    end
    return num / (c * expm1(L))
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
    L = -log1p(-F) / s                         # log(D_cr / D_gr)
    ρg = F * ρr / _p3_h_graupel(L, s)
    c = β - 2
    g = expm1(c * L) / (c * expm1(L))          # ρ_d / ρ_g
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

Unlike the mass, `A(D)` is *not* continuous at `D_th` (the sphere and
aggregate relations are independent fits), exactly as in MM15; it is
continuous in `F_rim` and `ρ_rim` at fixed `D`.
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
