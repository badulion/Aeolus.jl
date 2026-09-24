# Phase 45, issue #13 — sub-grid supersaturation and aerosol activation.
#
# Part-file of `TurbulentMicrophysics`; included from
# src/TurbulentMicrophysics.jl after sgs_supersaturation.jl, whose PDF
# families it reuses.
#
# The Phase 42 activation scheme (Abdul-Razzak & Ghan 2000) returns ONE
# peak supersaturation S_max per cell, computed from the cell-mean
# updraught, and activates every particle whose critical supersaturation
# lies below it: N_act = N_CCN(S_max). Unresolved turbulence spreads the
# cell over a distribution of supersaturations; each sub-volume activates
# the particles below ITS supersaturation, so the cell activates
#
#     ⟨N_act⟩ = ∫ N_CCN(S) P(S) dS,        P = PDF of S with mean S_max.
#
# N_CCN(S) is not linear, so ⟨N_act⟩ ≠ N_CCN(S_max) (Jensen): where the
# CCN spectrum is convex — few particles activated, the steep low-S
# flank of the log-normal — fluctuations ADD droplets; where it is
# concave — most of the population already activated — they remove
# them. For small σ the change is ½ N_CCN''(S̄) σ².
#
# The CCN spectrum itself is `AerosolCloud.ccn_fraction`, the very
# function `arg_activation` evaluates at S_max, so σ → 0 is the Phase 42
# result bit for bit. Nothing about Köhler theory is re-derived here.

export pdf_expectation, sgs_ccn_fraction, sgs_ccn_spectrum,
       SGSActivationResult, sgs_activation

# Gaussian components (weight, mean) and common std of a PDF family.
_gauss_components(::GaussianSaturationPDF, s̄, σ) = (((1.0, float(s̄)),), float(σ))
_gauss_components(pdf::DoubleGaussianSaturationPDF, s̄, σ) = _components(pdf, s̄, σ)

# Composite trapezoid of f(z) φ(z) on [-zmax, zmax]. For a smooth f the
# trapezoid rule on a Gaussian-weighted integrand converges spectrally,
# so 2001 nodes on ±10 are far past round-off for any f that varies on a
# scale ≳ 0.1 in z; `sgs_ccn_fraction` chooses the integration variable
# so that this holds.
function _gauss_trapezoid(f, n::Integer, zmax::Real)
    h = 2zmax / (n - 1)
    acc = 0.0
    for j in 0:n-1
        z = -zmax + j * h
        wj = (j == 0 || j == n - 1) ? 0.5 : 1.0
        acc += wj * f(z) * _φ(z)
    end
    return acc * h
end

"""
    pdf_expectation(f, pdf, s̄, σ; n = 2001, zmax = 10) -> ⟨f(s)⟩

Expectation of `f` over the sub-grid distribution `pdf` with mean `s̄`
and standard deviation `σ`, by trapezoidal quadrature over each
Gaussian component (`n` nodes on `±zmax` component standard
deviations). `σ = 0` returns `f(s̄)` exactly. The PDF families are those
of issue #12 ([`GaussianSaturationPDF`](@ref),
[`DoubleGaussianSaturationPDF`](@ref)); the variable can be the Mellor
deficit or, as in [`sgs_activation`](@ref), the relative
supersaturation.
"""
function pdf_expectation(f, pdf::AbstractSaturationPDF, s̄::Real, σ::Real;
                         n::Integer = 2001, zmax::Real = 10.0)
    σ < 0 && throw(ArgumentError("pdf_expectation: σ must be ≥ 0"))
    σ == 0 && return float(f(float(s̄)))
    comps, σc = _gauss_components(pdf, s̄, σ)
    return sum(w * _gauss_trapezoid(z -> f(μ + σc * z), n, zmax)
               for (w, μ) in comps)
end

"""
    sgs_ccn_fraction(m::ARGMode, S̄, σ_S; pdf = GaussianSaturationPDF(),
                     T_K = 298.15, method = :auto, n = 2001) -> fraction

Activated fraction of aerosol mode `m` in a cell whose supersaturation
is distributed as `pdf` with mean `S̄` and standard deviation `σ_S`
(both fractional):

```math
\\langle f\\rangle = \\int f_\\mathrm{CCN}(S)\\,P(S)\\,dS
 = \\int φ(u)\\,P\\!\\left(S > s_c(u)\\right)du ,
```

where `f_CCN` is `AerosolCloud.ccn_fraction` (the Phase 42 CCN
spectrum), `u = ln(r_d/r_g)/ln σ_g` the standardised dry radius and
`s_c(u) = S_m e^{-3u lnσ_g/2}` its critical supersaturation. The two
forms are the same 2-D Gaussian probability integrated in opposite
order; `method = :supersaturation` uses the first, `:radius` the second
(with the closed-form [`exceedance_fraction`](@ref) of `pdf`), and
`:auto` picks whichever has the smoother integrand — `:radius` when the
width of the activation threshold in `u`, `σ_c / (1.5 lnσ_g max(S̄, σ_c))`,
exceeds one, `:supersaturation` otherwise.

`σ_S = 0` returns `ccn_fraction(m, S̄; T_K)` — the Phase 42 answer
exactly.
"""
function sgs_ccn_fraction(m::ARGMode, S̄::Real, σ_S::Real;
                          pdf::AbstractSaturationPDF = GaussianSaturationPDF(),
                          T_K::Real = 298.15, method::Symbol = :auto,
                          n::Integer = 2001)
    σ_S < 0 && throw(ArgumentError("sgs_ccn_fraction: σ_S must be ≥ 0"))
    σ_S == 0 && return ccn_fraction(m, S̄; T_K = T_K)
    m.κ > KAPPA_MIN || return 0.0
    S_m = mode_critical_supersaturation(m, T_K)
    isfinite(S_m) || return 0.0
    lnσ = log(m.σ_g)
    if method === :auto
        _, σc = _gauss_components(pdf, S̄, σ_S)
        width_u = σc / (1.5 * lnσ * max(float(S̄), σc))
        method = width_u >= 1 ? :radius : :supersaturation
    end
    if method === :supersaturation
        return pdf_expectation(S -> ccn_fraction(m, S; T_K = T_K), pdf, S̄, σ_S; n = n)
    elseif method === :radius
        return _gauss_trapezoid(u -> exceedance_fraction(pdf, S_m * exp(-1.5 * lnσ * u),
                                                         S̄, σ_S), n, 10.0)
    end
    throw(ArgumentError("sgs_ccn_fraction: method must be :auto, :radius or :supersaturation"))
end

"""
    sgs_ccn_spectrum(modes, S̄, σ_S; pdf, T_K, method, n) -> m⁻³

`Σ_i N_i` [`sgs_ccn_fraction`](@ref)`(m_i, S̄, σ_S)`: the sub-grid mean of
`AerosolCloud.ccn_spectrum`.
"""
sgs_ccn_spectrum(modes::AbstractVector{<:ARGMode}, S̄::Real, σ_S::Real; kw...) =
    sum((m.N * sgs_ccn_fraction(m, S̄, σ_S; kw...) for m in modes); init = 0.0)

"""
    SGSActivationResult{T}

Activation of a cell with a sub-grid supersaturation distribution,
returned by [`sgs_activation`](@ref):

  * `mean`        — the Phase 42 `ActivationResult` of the cell-mean state
                    (its `S_max` is the mean of the distribution)
  * `σ_S`         — standard deviation of the supersaturation (fractional)
  * `fraction`    — sub-grid mean activated fraction per mode
  * `N_act`       — activated number per mode, m⁻³
  * `N_act_total` — summed activated number, m⁻³
"""
struct SGSActivationResult{T<:AbstractFloat}
    mean::ActivationResult{T}
    σ_S::T
    fraction::Vector{T}
    N_act::Vector{T}
    N_act_total::T
end

Base.show(io::IO, r::SGSActivationResult) = print(io,
    "SGSActivationResult(S̄=", round(r.mean.S_max * 100; digits = 4), " %, σ_S=",
    round(r.σ_S * 100; digits = 4), " %, N_act=",
    round(r.N_act_total * 1e-6; digits = 2), " cm⁻³ (mean-state ",
    round(r.mean.N_act_total * 1e-6; digits = 2), " cm⁻³))")

"""
    sgs_activation(modes, w, T_K, p, σ_S; pdf = GaussianSaturationPDF(), q, constants,
                   kinetic_radius = Inf, method = :auto) -> SGSActivationResult
    sgs_activation(modes, w, coeffs::ActivationCoefficients, σ_S; pdf, method)
    sgs_activation(modes, w, T_K, p, cloud::SubgridCloud; kwargs...)

Phase 42 activation with sub-grid supersaturation fluctuations. The
cell-mean peak supersaturation is `arg_activation`'s `S_max` for the
mean updraught `w`; unresolved turbulence spreads it by `σ_S`
(fractional), with the shape of `pdf`, and each mode's activated
fraction is the PDF-mean of the CCN spectrum
([`sgs_ccn_fraction`](@ref)). Passing a [`SubgridCloud`](@ref) uses its
`σ_S` — the clear-air relative-supersaturation spread implied by the
sub-grid `q_t`, `θ_li` (co)variances of issue #12.

Physics and limits:

  * `σ_S = 0` is `arg_activation` exactly (same `fraction`, `N_act`).
  * For small `σ_S`, `N_act − N_act(S̄) ≈ ½ N_CCN''(S̄) σ_S²`: sub-grid
    fluctuations increase activation where the CCN spectrum is convex
    in `S` (low activated fraction) and decrease it where it is concave.
  * The fluctuations are superimposed on `S_max`; they do not feed back
    on the competition for vapour that sets `S_max` (ARG's `η` term), and
    with `w ≤ 0` nothing activates, as in Phase 42.

A related, widely used alternative integrates activation over a
sub-grid *updraught* PDF instead (Ghan et al. 1997; Morrison &
Gettelman 2008); this scheme fluctuates the supersaturation directly, as
diagnosed from the scalar variances.

References: Abdul-Razzak & Ghan (2000); Ghan, S. J. et al. (1997),
*J. Geophys. Res.*, 102, 21777-21794; Morrison, H. & Gettelman, A.
(2008), *J. Climate*, 21, 3642-3659.
"""
function sgs_activation(modes::AbstractVector{<:ARGMode}, w::Real,
                        co::ActivationCoefficients{T}, σ_S::Real;
                        pdf::AbstractSaturationPDF = GaussianSaturationPDF(),
                        method::Symbol = :auto) where {T}
    σ_S < 0 && throw(ArgumentError("sgs_activation: σ_S must be ≥ 0"))
    r = arg_activation(modes, w, co)
    n = length(modes)
    if σ_S == 0 || !(r.S_max > 0) || !isfinite(r.S_max)
        return SGSActivationResult{T}(r, T(σ_S), copy(r.fraction), copy(r.N_act),
                                      r.N_act_total)
    end
    frac = zeros(T, n)
    N_act = zeros(T, n)
    for i in 1:n
        m = modes[i]
        m.N > 0 || continue
        frac[i] = sgs_ccn_fraction(m, r.S_max, σ_S; pdf = pdf, T_K = co.T_K,
                                   method = method)
        N_act[i] = m.N * frac[i]
    end
    return SGSActivationResult{T}(r, T(σ_S), frac, N_act, sum(N_act; init = zero(T)))
end

function sgs_activation(modes::AbstractVector{<:ARGMode}, w::Real, T_K::Real,
                        p::Real, σ_S::Real;
                        pdf::AbstractSaturationPDF = GaussianSaturationPDF(),
                        q::PhasePartition = PhasePartition(0.0),
                        constants::ThermodynamicConstants = DEFAULT_CONSTANTS,
                        kinetic_radius::Real = Inf, method::Symbol = :auto)
    co = activation_coefficients(T_K, p; q = q, constants = constants,
                                 kinetic_radius = kinetic_radius)
    return sgs_activation(modes, w, co, σ_S; pdf = pdf, method = method)
end

sgs_activation(modes::AbstractVector{<:ARGMode}, w::Real, T_K::Real, p::Real,
               cloud::SubgridCloud; kw...) =
    sgs_activation(modes, w, T_K, p, cloud.σ_S; kw...)
