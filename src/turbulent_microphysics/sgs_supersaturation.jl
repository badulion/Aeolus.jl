# Phase 45, issue #12 — sub-grid supersaturation distribution.
#
# Part-file of `TurbulentMicrophysics`; included from
# src/TurbulentMicrophysics.jl.

export AbstractSaturationPDF, GaussianSaturationPDF,
       DoubleGaussianSaturationPDF,
       saturation_pdf_density, exceedance_fraction,
       pdf_cloud_fraction, pdf_condensate

# =========================================================================
# 1. The PDF of the saturation deficit
# =========================================================================
#
# Everything here is written in terms of the Mellor (1977) variable `s`
# (kg kg⁻¹), the *extended* condensate: `s > 0` is the condensate a
# sub-volume holds after saturation adjustment, `s < 0` is (a_l times)
# its saturation deficit. Its distribution within a cell is described by
# a mean `s̄` and a standard deviation `σ` plus, for skewed families, a
# shape parameter. The two moments everything else needs are
#
#     C    = P(s > 0)                        cloud fraction
#     q̄_c  = ∫₀^∞ s P(s) ds                  mean condensate
#
# and both have closed forms for the families below.

"""
    AbstractSaturationPDF

A family of sub-grid probability densities for the saturation deficit
`s`, parameterised by the cell mean `s̄` and standard deviation `σ`.
Concrete families implement [`saturation_pdf_density`](@ref),
[`exceedance_fraction`](@ref) and [`pdf_condensate`](@ref).
"""
abstract type AbstractSaturationPDF end

"""
    GaussianSaturationPDF()

Gaussian distribution of `s` — the Sommeria & Deardorff (1977) /
Mellor (1977) model. With `Q = s̄/σ` the normalised saturation excess,

```math
C = \\tfrac12\\left[1 + \\mathrm{erf}\\!\\left(Q/\\sqrt2\\right)\\right],\\qquad
\\bar q_c = \\sigma\\left[Q\\,C + \\frac{e^{-Q^2/2}}{\\sqrt{2\\pi}}\\right].
```
"""
struct GaussianSaturationPDF <: AbstractSaturationPDF end

"""
    DoubleGaussianSaturationPDF(skewness; shape = 1.0)

Binormal (two-Gaussian mixture) distribution of `s` with prescribed
skewness `Sk`, in the "analytic double Gaussian" form of Larson, Golaz &
Cotton (2002). Shallow cumulus fields are strongly positively skewed in
`s` — a few moist updrafts in a dry environment — which a single
Gaussian cannot represent (Bechtold et al. 1995).

Both components share the standard deviation `σ_c = σ √(1 - γ)`, where
`γ ∈ [0, 1)` is the fraction of the variance carried by the separation
of the component means. With `S̃ = Sk / γ^{3/2}`,

```math
w = \\tfrac12\\left[1 - \\operatorname{sign}(\\tilde S)
      \\sqrt{\\tilde S^2/(4 + \\tilde S^2)}\\right],\\quad
\\mu_1 = \\bar s + \\sigma\\sqrt{\\gamma(1-w)/w},\\quad
\\mu_2 = \\bar s - \\sigma\\sqrt{\\gamma w/(1-w)},
```

which reproduces the mean, the variance and the third moment exactly
for any `γ`. The remaining freedom is closed here by

```math
\\gamma^{3/2} = \\frac{|Sk|}{|Sk| + c},\\qquad c = \\texttt{shape} > 0,
```

a choice made in this package (not taken from Larson et al.) so that the
component separation vanishes continuously as `Sk → 0`: the family then
reduces **exactly** to [`GaussianSaturationPDF`](@ref), and a very
skewed field (`|Sk| → ∞`) approaches two delta-like components.
"""
struct DoubleGaussianSaturationPDF{T<:AbstractFloat} <: AbstractSaturationPDF
    skewness::T
    shape::T
    function DoubleGaussianSaturationPDF{T}(skewness, shape) where {T}
        (shape > 0 && isfinite(shape)) || throw(ArgumentError(
            "DoubleGaussianSaturationPDF: shape must be finite and > 0, got $shape"))
        isfinite(skewness) || throw(ArgumentError(
            "DoubleGaussianSaturationPDF: skewness must be finite, got $skewness"))
        return new{T}(skewness, shape)
    end
end

DoubleGaussianSaturationPDF(skewness::Real = 0.0; shape::Real = 1.0) =
    DoubleGaussianSaturationPDF{Float64}(float(skewness), float(shape))

# Standard normal density and CDF, the latter from GasOptics' erfc (no
# SpecialFunctions dependency). erfc rather than erf so the lower tail
# keeps relative accuracy.
_φ(x) = exp(-x * x / 2) / sqrt(2π)
_Φ(x) = 0.5 * erfc_series(-x / sqrt(2.0))

# ∫_{-Q}^{∞} (Q + z) φ(z) dz  =  Q Φ(Q) + φ(Q): mean positive part of
# N(Q, 1), smooth through Q = 0. Exactly it is > 0, but for Q ≲ -38 the
# two terms cancel to round-off and the difference can come out as a
# negative denormal; clamp so a condensate is never negative.
_gauss_positive_part(Q) = max(Q * _Φ(Q) + _φ(Q), 0.0)

# Component weights / means / common std of the double Gaussian for a
# given (s̄, σ). Returns ((w₁, μ₁), (w₂, μ₂), σ_c).
function _components(pdf::DoubleGaussianSaturationPDF, s̄::Real, σ::Real)
    Sk = pdf.skewness
    if Sk == 0             # γ = 0: a single Gaussian
        return ((0.5, float(s̄)), (0.5, float(s̄))), float(σ)
    end
    aS = abs(Sk)
    g32 = aS / (aS + pdf.shape)                    # γ^{3/2}
    γ = g32^(2 / 3)
    # 1 - γ and the small weight are formed without cancellation, so that
    # σ_c > 0 and w ∈ (0, 1) hold for every finite Sk. The naive
    # `sqrt(1 - γ)` and `½(1 - √(S̃²/(4+S̃²)))` round to exactly 0 once
    # |Sk| ≳ 1e8 and turn C and q̄_c into NaN.
    one_minus_γ = -expm1((2 / 3) * log1p(-pdf.shape / (aS + pdf.shape)))
    S̃ = aS / g32                                  # |Sk| / γ^{3/2}
    r = hypot(2.0, S̃)
    w_small = 2 / (r * (r + S̃))                   # ½(1 - S̃/√(4+S̃²))
    w1, w2 = Sk > 0 ? (w_small, 1 - w_small) : (1 - w_small, w_small)
    μ1 = s̄ + σ * sqrt(γ * w2 / w1)
    μ2 = s̄ - σ * sqrt(γ * w1 / w2)
    σc = σ * sqrt(one_minus_γ)
    return ((w1, μ1), (w2, μ2)), σc
end

"""
    saturation_pdf_density(pdf, s, s̄, σ) -> P(s)

Probability density (per kg kg⁻¹) of the saturation deficit at `s`,
for a cell with mean `s̄` and standard deviation `σ > 0`.
"""
function saturation_pdf_density(::GaussianSaturationPDF, s::Real, s̄::Real, σ::Real)
    σ > 0 || throw(ArgumentError("saturation_pdf_density: σ must be > 0"))
    return _φ((s - s̄) / σ) / σ
end

function saturation_pdf_density(pdf::DoubleGaussianSaturationPDF, s::Real,
                                s̄::Real, σ::Real)
    σ > 0 || throw(ArgumentError("saturation_pdf_density: σ must be > 0"))
    comps, σc = _components(pdf, s̄, σ)
    return sum(w * _φ((s - μ) / σc) / σc for (w, μ) in comps)
end

"""
    exceedance_fraction(pdf, x, s̄, σ) -> P(s > x)

Fraction of the cell in which the saturation deficit exceeds `x`.
With `x = 0` this is the cloud fraction; with `x` set to the deficit
equivalent of an activation threshold it is the fraction of the cell in
which that threshold is crossed. `σ = 0` returns the all-or-nothing
step `s̄ > x`.
"""
function exceedance_fraction(::GaussianSaturationPDF, x::Real, s̄::Real, σ::Real)
    σ <= 0 && return s̄ > x ? 1.0 : 0.0
    return _Φ((s̄ - x) / σ)
end

function exceedance_fraction(pdf::DoubleGaussianSaturationPDF, x::Real,
                             s̄::Real, σ::Real)
    σ <= 0 && return s̄ > x ? 1.0 : 0.0
    comps, σc = _components(pdf, s̄, σ)
    return sum(w * _Φ((μ - x) / σc) for (w, μ) in comps)
end

"""
    pdf_cloud_fraction(pdf, s̄, σ) -> C ∈ [0, 1]

Sub-grid cloud fraction `P(s > 0)`. For the Gaussian family this is
`½(1 + erf(Q/√2))` with `Q = s̄/σ` (Sommeria & Deardorff 1977).
"""
pdf_cloud_fraction(pdf::AbstractSaturationPDF, s̄::Real, σ::Real) =
    exceedance_fraction(pdf, 0.0, s̄, σ)

"""
    pdf_condensate(pdf, s̄, σ) -> q̄_c

Cell-mean condensate `∫₀^∞ s P(s) ds` (kg kg⁻¹). For the Gaussian
family this is `σ [Q C + exp(-Q²/2)/√(2π)]` (Mellor 1977). It is always
`≥ max(s̄, 0)` — by Jensen's inequality, since `max(s, 0)` is convex —
so sub-grid variability can only *add* condensate relative to the
mean-state saturation adjustment. `σ = 0` returns `max(s̄, 0)`.
"""
function pdf_condensate(::GaussianSaturationPDF, s̄::Real, σ::Real)
    σ <= 0 && return max(float(s̄), 0.0)
    return σ * _gauss_positive_part(s̄ / σ)
end

function pdf_condensate(pdf::DoubleGaussianSaturationPDF, s̄::Real, σ::Real)
    σ <= 0 && return max(float(s̄), 0.0)
    comps, σc = _components(pdf, s̄, σ)
    return sum(w * σc * _gauss_positive_part(μ / σc) for (w, μ) in comps)
end

# =========================================================================
# 2. From (q_t, θ_li) fluctuations to the saturation deficit
# =========================================================================
#
# Mellor (1977): linearise q_sat about the liquid-water temperature T_l,
# the temperature the parcel would have with all condensate evaporated.
# Condensing δq_c raises T by Γ δq_c (Γ = ∂T/∂q_c|θ_li ≈ L/c_pm) and
# q_sat by q_s' Γ δq_c, so the condensate a sub-volume holds after
# adjustment is
#
#     s = a (q_t − q_sat(T_l)),      a = 1 / (1 + Γ ∂q_s/∂T).
#
# A fluctuation (q_t', θ_li') moves T_l by Π θ_li' + (∂T_l/∂q_t) q_t'
# (Π = T_l/θ_li, and the second term is the q_t-dependence of the moist
# Exner exponent R_m/c_pm), hence
#
#     s' = a_q q_t' − b θ_li',
#     a_q = a (1 − ∂q_s/∂T · ∂T_l/∂q_t),   b = a ∂q_s/∂T · Π,
#     σ_s² = a_q² σ_qt² − 2 a_q b ⟨q_t'θ_li'⟩ + b² σ_θ².
#
# Only the FLUCTUATIONS are linearised. The cell-mean s̄ is anchored on
# the exact, nonlinear saturation adjustment of the Phase 39 contract:
# where that adjustment condenses water, s̄ is exactly its condensate;
# where it does not, s̄ = a (q_t − q_sat(T_l)) < 0. The two branches meet
# at s̄ = 0, and the σ → 0 limit of everything below is therefore the
# all-or-nothing adjustment itself, not a linearisation of it.
#
# Saturation is taken over liquid throughout, with condensate split into
# liquid and ice by `liquid_fraction` — the same convention as
# `MoistThermodynamics.saturation_adjustment`, which this must agree with.

export MellorCoefficients, mellor_coefficients, saturation_deficit_variance,
       extended_condensate, SubgridCloud, subgrid_cloud

"""
    MellorCoefficients{T}

Linearisation of the saturation deficit about the liquid-water
temperature (Mellor 1977), returned by [`mellor_coefficients`](@ref):

  * `a`     — thermodynamic factor `1/(1 + Γ ∂q_s/∂T)`, `Γ = dTdqc`
  * `a_q`   — `∂s/∂q_t` (dimensionless)
  * `b`     — `-∂s/∂θ_li` (kg kg⁻¹ K⁻¹)
  * `T_l`   — liquid-water temperature (K)
  * `q_sl`  — saturation specific humidity at `T_l` (kg kg⁻¹)
  * `dqsdT` — `∂q_s/∂T` at `T_l` (kg kg⁻¹ K⁻¹)
  * `dTdqc` — heating per unit condensate at fixed `θ_li`, `p`, `q_t`
              (K per kg kg⁻¹); `≈ L/c_pm`
"""
struct MellorCoefficients{T<:AbstractFloat}
    a::T
    a_q::T
    b::T
    T_l::T
    q_sl::T
    dqsdT::T
    dTdqc::T
end

"""
    mellor_coefficients(θ_li, p, q_t, c = DEFAULT_CONSTANTS) -> MellorCoefficients

Coefficients of the linearised saturation deficit
`s' = a_q q_t' − b θ_li'` at the state `(θ_li, p, q_t)`.

`∂q_s/∂T` is the exact derivative of the contract's
`q_s = ε e_s/(p − (1−ε) e_s)` with Clausius-Clapeyron
`de_s/dT = e_s L_v/(R_v T²)`, so it is consistent with
`saturation_specific_humidity` by construction. The latent heat
is the liquid/ice blend `λ L_v + (1−λ) L_s` with `λ =`
`liquid_fraction``(T_l)`, matching the partition used by
`saturation_adjustment`.

Reference: Mellor (1977); Sommeria & Deardorff (1977), eq. (7)-(10).
"""
function mellor_coefficients(θ_li::Real, p::Real, q_t::Real,
                             c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    qt = max(0.0, float(q_t))
    pp = float(p)
    q0 = PhasePartition(qt)
    T_l = temperature_from_theta_li(θ_li, pp, q0, c)
    es = saturation_vapor_pressure(T_l, c)
    q_sl = saturation_specific_humidity(T_l, pp, c)
    ε = molmass_ratio(c)
    Lv = latent_heat_vapor(T_l, c)
    # Past the boiling point q_sat is pinned at 1 and does not vary.
    dqsdT = (es >= pp || q_sl >= 1) ? 0.0 :
            q_sl * pp / (pp - (1 - ε) * es) * Lv / (R_v(c) * T_l^2)
    λ = liquid_fraction(T_l, c)
    L = λ * Lv + (1 - λ) * latent_heat_sublim(T_l, c)
    cpm = heat_capacity_moist(q0, c)
    Rm = gas_constant_moist(q0, c)
    lnp = log(pp / c.p_ref)
    # Heating per unit condensate at fixed (θ_li, p, q_t), differentiated
    # from the contract's own θ_li = T/Π(q) · exp(-(L_v q_l + L_s q_i)/(c_pm T))
    # at q_c = 0: the familiar L/c_pm plus the change of the moist Exner
    # exponent κ = R_m/c_pm as vapour becomes condensate (+1.4 % at
    # 290 K / 900 hPa, +8.8 % at 280 K / 500 hPa; ∝ ln(p/p_ref)). Using
    # bare L/c_pm here leaves a ~1 % mismatch with `saturation_adjustment`
    # that does not vanish as σ → 0.
    dRm_c = -R_v(c)                                            # ∂R_m/∂q_c
    dcp_c = λ * c.cp_liquid + (1 - λ) * c.cp_ice - c.cp_vapor  # ∂c_pm/∂q_c
    dκ_c = (dRm_c * cpm - Rm * dcp_c) / cpm^2
    dTdqc = L / cpm + T_l * lnp * dκ_c
    a = 1 / (1 + dTdqc * dqsdT)
    # T_l = θ_li Π(p, q_t): ∂T_l/∂θ_li = Π, and ∂T_l/∂q_t through the
    # vapour dependence of κ.
    Π = T_l / float(θ_li)
    dRm_t = R_v(c) - c.gas_constant / c.molmass_dry            # ∂R_m/∂q_t
    dcp_t = c.cp_vapor - c.cp_dry                              # ∂c_pm/∂q_t
    dκ_t = (dRm_t * cpm - Rm * dcp_t) / cpm^2
    dTl_dqt = T_l * lnp * dκ_t
    a_q = a * (1 - dqsdT * dTl_dqt)
    b = a * dqsdT * Π
    return MellorCoefficients(a, a_q, b, T_l, q_sl, dqsdT, dTdqc)
end

"""
    saturation_deficit_variance(m::MellorCoefficients, var_qt, var_θ = 0, cov_qθ = 0) -> σ_s²

Variance of the saturation deficit from the sub-grid (co)variances of
total water `var_qt` (kg² kg⁻²), liquid-ice potential temperature
`var_θ` (K²) and their covariance `cov_qθ` (kg kg⁻¹ K):

```math
\\sigma_s^2 = a_q^2\\,\\overline{q_t'^2} - 2 a_q b\\,\\overline{q_t'\\theta_{li}'}
            + b^2\\,\\overline{\\theta_{li}'^2}.
```

Positively correlated `q_t` and `θ_li` fluctuations (moist and warm
together, typical of surface-driven thermals) *reduce* `σ_s`, because
the warm anomaly raises the saturation humidity the moist anomaly would
otherwise exceed. Throws if the inputs violate Cauchy-Schwarz.
"""
function saturation_deficit_variance(m::MellorCoefficients, var_qt::Real,
                                     var_θ::Real = 0.0, cov_qθ::Real = 0.0)
    (var_qt < 0 || var_θ < 0) && throw(ArgumentError(
        "saturation_deficit_variance: variances must be ≥ 0"))
    cov_qθ^2 > var_qt * var_θ * (1 + 1e-12) &&
        throw(ArgumentError("saturation_deficit_variance: |cov| exceeds " *
                            "√(var_qt var_θ) (Cauchy-Schwarz)"))
    v = m.a_q^2 * var_qt - 2 * m.a_q * m.b * cov_qθ + m.b^2 * var_θ
    return max(v, 0.0)
end

"""
    extended_condensate(θ_li, p, q_t, c = DEFAULT_CONSTANTS) -> (s̄, T, q::PhasePartition)

Cell-mean extended condensate `s̄` together with the all-or-nothing
`saturation_adjustment` of the mean state it is anchored on.
`s̄ > 0` is exactly that adjustment's condensate; `s̄ ≤ 0` is the
Mellor-scaled deficit `a (q_t − q_sat(T_l))`.
"""
function extended_condensate(θ_li::Real, p::Real, q_t::Real,
                             c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    T, q = saturation_adjustment(θ_li, p, q_t, c)
    qc = q.liq + q.ice
    qc > 0 && return qc, T, q
    return _clear_deficit(mellor_coefficients(θ_li, p, q_t, c), q_t), T, q
end

# s̄ on the clear side of the boundary, a (q_t − q_sat(T_l)) ≤ 0.
_clear_deficit(m::MellorCoefficients, q_t) =
    min(m.a * (max(0.0, float(q_t)) - m.q_sl), 0.0)

"""
    SubgridCloud{T}

Partial-cloudiness diagnosis of one grid cell, returned by
[`subgrid_cloud`](@ref):

  * `cloud_fraction` — `P(s > 0)`, in `[0, 1]`
  * `condensate`     — cell-mean condensate `q̄_c` (kg kg⁻¹)
  * `q`              — cell-mean `PhasePartition` (total water unchanged)
  * `T`              — cell-mean temperature consistent with `θ_li` and `q`
  * `s̄`, `σ_s`       — mean and standard deviation of the deficit
  * `Q`              — normalised saturation excess `s̄/σ_s` (±Inf if `σ_s = 0`)
  * `σ_S`            — standard deviation of the sub-grid *relative
                       supersaturation* `S = q_v/q_sat − 1` in the clear
                       part, `σ_s/(a q_sl)` (dimensionless)
  * `coefficients`   — the [`MellorCoefficients`](@ref) used
"""
struct SubgridCloud{T<:AbstractFloat}
    cloud_fraction::T
    condensate::T
    q::PhasePartition{T}
    T::T
    s̄::T
    σ_s::T
    Q::T
    σ_S::T
    coefficients::MellorCoefficients{T}
end

"""
    subgrid_cloud(θ_li, p, q_t; var_qt = 0, var_θ = 0, cov_qθ = 0,
                  σ_s = nothing, pdf = GaussianSaturationPDF(),
                  c = DEFAULT_CONSTANTS) -> SubgridCloud

Sub-grid cloud fraction and mean condensate of a cell with mean state
`(θ_li, p, q_t)` and sub-grid (co)variances of `q_t` and `θ_li`. Pass
`σ_s` directly to bypass the (co)variance mapping.

The deficit distribution has mean [`extended_condensate`](@ref) and
standard deviation [`saturation_deficit_variance`](@ref)`^{1/2}`; the
cloud fraction and condensate are [`pdf_cloud_fraction`](@ref) and
[`pdf_condensate`](@ref) of `pdf`. The condensate is split between
liquid and ice by `liquid_fraction` at the resulting mean
temperature, and that temperature is recovered from `θ_li` by
`temperature_from_theta_li`, so `θ_li` and `q_t` — the two
conserved variables — are left exactly as they were.

With `σ_s = 0` the result **is** the all-or-nothing
`saturation_adjustment` (same temperature and partition, cloud
fraction 0 or 1).

Mean condensate is capped at `q_t`: a Gaussian tail can otherwise ask
for more water than the cell contains when `σ_s ≳ q_t`, which is outside
the model's validity anyway.

References: Sommeria & Deardorff (1977); Mellor (1977); Bechtold et al.
(1995); Chaboureau & Bechtold (2002).
"""
function subgrid_cloud(θ_li::Real, p::Real, q_t::Real;
                       var_qt::Real = 0.0, var_θ::Real = 0.0, cov_qθ::Real = 0.0,
                       σ_s::Union{Nothing,Real} = nothing,
                       pdf::AbstractSaturationPDF = GaussianSaturationPDF(),
                       c::ThermodynamicConstants = DEFAULT_CONSTANTS,
                       tol::Real = 1e-10)
    qt = max(0.0, float(q_t))
    pp = float(p)
    m = mellor_coefficients(θ_li, pp, qt, c)
    σ = σ_s === nothing ? sqrt(saturation_deficit_variance(m, var_qt, var_θ, cov_qθ)) :
                          float(σ_s)
    σ < 0 && throw(ArgumentError("subgrid_cloud: σ_s must be ≥ 0"))
    # = extended_condensate(θ_li, pp, qt, c), reusing `m` on the clear side
    T_adj, q_adj = saturation_adjustment(θ_li, pp, qt, c)
    qc_adj = q_adj.liq + q_adj.ice
    s̄ = qc_adj > 0 ? qc_adj : _clear_deficit(m, qt)
    σ_S = m.q_sl > 0 ? σ / (m.a * m.q_sl) : 0.0

    if σ == 0
        # All-or-nothing: return the contract's adjustment untouched.
        C = s̄ > 0 ? 1.0 : 0.0
        Q = s̄ > 0 ? Inf : -Inf
        return SubgridCloud(C, q_adj.liq + q_adj.ice, q_adj, T_adj, s̄, σ, Q,
                            σ_S, m)
    end

    C = pdf_cloud_fraction(pdf, s̄, σ)
    qc = min(pdf_condensate(pdf, s̄, σ), qt)

    # Mean temperature: invert θ_li at the new partition. λ(T) couples the
    # split to T, so iterate; λ is smooth and this converges in a few
    # passes (one pass above freezing, where λ ≡ 1).
    T = T_adj
    q = phase_partition(T, qt, qc, c)
    for _ in 1:20
        Tn = temperature_from_theta_li(θ_li, pp, q, c; tol = tol)
        q = phase_partition(Tn, qt, qc, c)
        done = abs(Tn - T) < 10 * tol
        T = Tn
        done && break
    end
    return SubgridCloud(C, qc, q, T, s̄, σ, s̄ / σ, σ_S, m)
end

# =========================================================================
# 3. Sub-grid (co)variances from the Smagorinsky-Lilly closure
# =========================================================================
#
# The LES closes SGS fluxes with an eddy viscosity ν_t = (C_s Δ)² |S|
# (`Operators.compute_smagorinsky_νt!`) and an eddy diffusivity
# K = ν_t / Pr_t for scalars. The SGS scalar-variance budget in local
# equilibrium (Deardorff 1980; the diagnostic closure of Bechtold et al.
# 1995 and Chaboureau & Bechtold 2002) balances gradient production
# against dissipation on the SGS turnover time Δ/√e:
#
#     2 K |∇φ̄|²  =  C_χ (√e / Δ) φ'²
#
# With the Smagorinsky ν_t written in Deardorff form, ν_t = C_k Δ √e,
# ν_t cancels and
#
#     φ'² = c_var Δ² |∇φ̄|²,      c_var = 2 C_k / (C_χ Pr_t),
#     φ'ψ' = c_var Δ² ∇φ̄·∇ψ̄                 (same budget for a covariance).
#
# So in equilibrium the variance does not depend on C_s at all — only on
# the filter width and the resolved gradient. What ν_t still decides is
# whether there is SGS turbulence: where ν_t = 0 there is neither
# production nor dissipation, and the variance is taken as zero.
#
# `Δ = (dx dy dz)^{1/3}` is the Smagorinsky filter width used by
# `compute_smagorinsky_νt!`. The default c_var = 1/12 is the variance of
# a linear profile of slope |∇φ̄| across a cell of width Δ — what the
# resolved gradient alone puts inside a cell; a turbulent closure with
# explicit (C_k, C_χ, Pr_t) is available via
# `equilibrium_variance_coefficient`.

export GEOMETRIC_VARIANCE_COEFF, equilibrium_variance_coefficient,
       filter_width, sgs_scalar_covariance!, sgs_scalar_variance!,
       subgrid_cloud!

"""
    GEOMETRIC_VARIANCE_COEFF = 1/12

Default `c_var` in `φ'² = c_var Δ² |∇φ̄|²`: the variance of a linear
profile of slope `|∇φ̄|` across a cubic cell of width `Δ`,
`(1/Δ)∫(G z)² dz = G²Δ²/12` over `z ∈ [-Δ/2, Δ/2]`.
"""
const GEOMETRIC_VARIANCE_COEFF = 1 / 12

"""
    equilibrium_variance_coefficient(; C_k = 0.1, C_χ, Pr_t = 1/3) -> c_var

Coefficient of the local-equilibrium SGS scalar variance
`φ'² = c_var Δ² |∇φ̄|²` implied by a Deardorff-type closure with
eddy viscosity `ν_t = C_k Δ √e`, eddy diffusivity `ν_t/Pr_t` and scalar
variance dissipation `C_χ (√e/Δ) φ'²`:

```math
c_\\mathrm{var} = \\frac{2 C_k}{C_\\chi\\,\\mathrm{Pr}_t}.
```

`C_k = 0.1` is Deardorff's (1980) value; `Pr_t = 1/3` is the default of
the package's `Simulations` and `Anelastic` cores. `C_χ` has no default
here: published values differ between closures, and the choice should
be made — and stated — by the caller.
"""
function equilibrium_variance_coefficient(; C_k::Real = 0.1, C_χ::Real,
                                          Pr_t::Real = 1 / 3)
    (C_k > 0 && C_χ > 0 && Pr_t > 0) || throw(ArgumentError(
        "equilibrium_variance_coefficient: constants must be > 0"))
    return 2 * float(C_k) / (float(C_χ) * float(Pr_t))
end

"""
    filter_width(grid::RegularGrid) -> Δ

Smagorinsky filter width `(dx dy dz)^{1/3}`, the same `Δ` as
`compute_smagorinsky_νt!` uses.
"""
filter_width(g::RegularGrid) = (g.dx * g.dy * g.dz)^(1 / 3)

"""
    sgs_scalar_covariance!(cov, φ, ψ; c_var = GEOMETRIC_VARIANCE_COEFF, νt = nothing)

Local-equilibrium SGS covariance `φ'ψ' = c_var Δ² ∇φ̄·∇ψ̄` at cell
centres, with second-order centred gradients. Halos of `φ` and `ψ` must
be filled. With an eddy-viscosity field `νt` (from
`compute_smagorinsky_νt!`), cells with `νt ≤ 0` — no SGS turbulence —
get zero. See the section comment in the source for the derivation from
the Smagorinsky-Lilly closure.
"""
function sgs_scalar_covariance!(cov::Field, φ::Field, ψ::Field;
                                c_var::Real = GEOMETRIC_VARIANCE_COEFF,
                                νt::Union{Nothing,Field} = nothing)
    g = φ.mesh
    Δ2 = filter_width(g)^2
    h2x = 1 / (2g.dx); h2y = 1 / (2g.dy); h2z = 1 / (2g.dz)
    F, P, C = φ.data, ψ.data, cov.data
    ii, jj, kk = interior(g)
    @inbounds for k in kk, j in jj, i in ii
        if νt !== nothing && νt.data[i, j, k] <= 0
            C[i, j, k] = 0
            continue
        end
        fx = (F[i+1, j, k] - F[i-1, j, k]) * h2x
        fy = (F[i, j+1, k] - F[i, j-1, k]) * h2y
        fz = (F[i, j, k+1] - F[i, j, k-1]) * h2z
        px = (P[i+1, j, k] - P[i-1, j, k]) * h2x
        py = (P[i, j+1, k] - P[i, j-1, k]) * h2y
        pz = (P[i, j, k+1] - P[i, j, k-1]) * h2z
        C[i, j, k] = c_var * Δ2 * (fx * px + fy * py + fz * pz)
    end
    return cov
end

"""
    sgs_scalar_variance!(var, φ; c_var = GEOMETRIC_VARIANCE_COEFF, νt = nothing)

Local-equilibrium SGS variance `φ'² = c_var Δ² |∇φ̄|²`; the `ψ = φ` case
of [`sgs_scalar_covariance!`](@ref).
"""
sgs_scalar_variance!(var::Field, φ::Field; kw...) =
    sgs_scalar_covariance!(var, φ, φ; kw...)

_pressure_at(p::Real, i, j, k, kl) = float(p)
_pressure_at(p::Field, i, j, k, kl) = p.data[i, j, k]
_pressure_at(p::AbstractVector, i, j, k, kl) = float(p[kl])

"""
    subgrid_cloud!(cloud_fraction, condensate, θ_li, q_t, p;
                   νt = nothing, c_var = GEOMETRIC_VARIANCE_COEFF,
                   pdf = GaussianSaturationPDF(), c = DEFAULT_CONSTANTS,
                   T = nothing, liquid = nothing, ice = nothing)

Diagnose partial cloudiness on a grid. For every interior cell the
sub-grid (co)variances of `q_t` and `θ_li` are formed from the resolved
gradients by the Smagorinsky-consistent equilibrium closure of
[`sgs_scalar_covariance!`](@ref), mapped to `σ_s` by
[`saturation_deficit_variance`](@ref), and passed to
[`subgrid_cloud`](@ref). Halos of `θ_li` and `q_t` must be filled.

`p` is a number, a cell-centred `Field`, or a vector over the interior
vertical levels (an anelastic reference profile). Optional `T`,
`liquid`, `ice` fields receive the cell-mean temperature and phase
split. `θ_li` and `q_t` are not modified: they are the conserved
variables, and this is a diagnosis of how the water they describe is
distributed, not a tendency.

With `νt` identically zero (SGS closure off) every cell has `σ_s = 0`
and the result is the all-or-nothing `saturation_adjustment`.
"""
function subgrid_cloud!(cf::Field, qc::Field, θ_li::Field, q_t::Field, p;
                        νt::Union{Nothing,Field} = nothing,
                        c_var::Real = GEOMETRIC_VARIANCE_COEFF,
                        pdf::AbstractSaturationPDF = GaussianSaturationPDF(),
                        c::ThermodynamicConstants = DEFAULT_CONSTANTS,
                        T::Union{Nothing,Field} = nothing,
                        liquid::Union{Nothing,Field} = nothing,
                        ice::Union{Nothing,Field} = nothing)
    g = θ_li.mesh
    Δ2 = filter_width(g)^2
    h2x = 1 / (2g.dx); h2y = 1 / (2g.dy); h2z = 1 / (2g.dz)
    Θ, Qt = θ_li.data, q_t.data
    ii, jj, kk = interior(g)
    for (kl, k) in enumerate(kk), j in jj, i in ii
        θ = Θ[i, j, k]; qt = Qt[i, j, k]
        pr = _pressure_at(p, i, j, k, kl)
        if νt !== nothing && νt.data[i, j, k] <= 0
            vq = vθ = cqθ = 0.0
        else
            tx = (Θ[i+1, j, k] - Θ[i-1, j, k]) * h2x
            ty = (Θ[i, j+1, k] - Θ[i, j-1, k]) * h2y
            tz = (Θ[i, j, k+1] - Θ[i, j, k-1]) * h2z
            qx = (Qt[i+1, j, k] - Qt[i-1, j, k]) * h2x
            qy = (Qt[i, j+1, k] - Qt[i, j-1, k]) * h2y
            qz = (Qt[i, j, k+1] - Qt[i, j, k-1]) * h2z
            vq = c_var * Δ2 * (qx^2 + qy^2 + qz^2)
            vθ = c_var * Δ2 * (tx^2 + ty^2 + tz^2)
            cqθ = c_var * Δ2 * (qx * tx + qy * ty + qz * tz)
            # the gradient-model covariance obeys Cauchy-Schwarz exactly;
            # clamp the last ulp so round-off cannot trip the check
            lim = sqrt(vq * vθ)
            cqθ = clamp(cqθ, -lim, lim)
        end
        r = subgrid_cloud(θ, pr, qt; var_qt = vq, var_θ = vθ, cov_qθ = cqθ,
                          pdf = pdf, c = c)
        cf.data[i, j, k] = r.cloud_fraction
        qc.data[i, j, k] = r.condensate
        T === nothing || (T.data[i, j, k] = r.T)
        liquid === nothing || (liquid.data[i, j, k] = r.q.liq)
        ice === nothing || (ice.data[i, j, k] = r.q.ice)
    end
    return cf, qc
end
