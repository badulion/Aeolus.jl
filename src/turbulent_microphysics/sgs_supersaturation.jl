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
        shape > 0 || throw(ArgumentError(
            "DoubleGaussianSaturationPDF: shape must be > 0, got $shape"))
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
# N(Q, 1). Written this way it is ≥ 0 and smooth through Q = 0.
_gauss_positive_part(Q) = Q * _Φ(Q) + _φ(Q)

# Component weights / means / common std of the double Gaussian for a
# given (s̄, σ). Returns ((w₁, μ₁), (w₂, μ₂), σ_c).
function _components(pdf::DoubleGaussianSaturationPDF, s̄::Real, σ::Real)
    Sk = pdf.skewness
    if Sk == 0             # γ = 0: a single Gaussian
        return ((0.5, float(s̄)), (0.5, float(s̄))), float(σ)
    end
    g32 = abs(Sk) / (abs(Sk) + pdf.shape)          # γ^{3/2}
    γ = g32^(2 / 3)
    S̃ = Sk / g32
    w = 0.5 * (1 - sign(S̃) * sqrt(S̃^2 / (4 + S̃^2)))
    μ1 = s̄ + σ * sqrt(γ * (1 - w) / w)
    μ2 = s̄ - σ * sqrt(γ * w / (1 - w))
    σc = σ * sqrt(1 - γ)
    return ((w, μ1), (1 - w, μ2)), σc
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
