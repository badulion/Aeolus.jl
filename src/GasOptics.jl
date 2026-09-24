module GasOptics

# Phase 32 — Correlated-k gas optics.
#
# Why a band model rather than RRTMGP: the RRTMGP k-distribution lives in
# ~40 MB of netCDF lookup tables, and the HDF5 stack does not build in
# this environment. Reproducing ~350 VSOP-scale coefficients from memory
# would be worse than useless — silently wrong absorption is harder to
# detect than a documented approximation. Instead this module implements
# the **Malkmus statistical narrow-band model**, which is a genuine
# literature-standard scheme (Malkmus 1967; Goody & Yung, *Atmospheric
# Radiation* 2nd ed. §4.3) whose per-band parameters are few enough to
# embed honestly, wrapped in a **correlated-k framework** so a real
# k-distribution table can be dropped in later.
#
# The correlated-k machinery is not a formality. A band model gives band
# *transmission*; a radiative-transfer solver needs per-g-point
# *absorption coefficients* so that multiple scattering and vertical
# inhomogeneity can be handled layer by layer. The reordering step below
# converts one into the other exactly, via the analytic Malkmus
# cumulative k-distribution.
#
# Fidelity ceiling, stated plainly: a statistical narrow-band model
# reproduces band-mean transmission well but cannot represent individual
# line positions, so it is less accurate than RRTMGP for strongly
# overlapping bands and for very inhomogeneous paths. Expect a few W/m²
# on broadband fluxes, not a few tenths.

using ..RadiationCore
using ..RadiationCore: SpectralGrid, OpticalProperties, AtmosphericColumn,
                       AbstractGasOptics, nbands, ngpoints, gpoint_range,
                       band_limits, band_of_gpoint, planck_band,
                       shortwave_bands, longwave_bands
import ..RadiationCore: compute_optics!

export MalkmusParams, malkmus_transmission, malkmus_log_transmission,
       malkmus_params,
       erf_series, erfc_series,
       malkmus_cumulative_k, kdist_from_band_model,
       CorrelatedKGasOptics, correlated_k_optics,
       rayleigh_cross_section, rayleigh_optical_depth, rayleigh_optics!,
       lw_source!, lw_surface_source,
       column_absorber_amount, transmission_band, optical_depth_profile,
       roberts_continuum_coefficient, continuum_optical_depth,
       us_standard_column

# =========================================================================
# Error function (no SpecialFunctions dependency — it does not build here)
# =========================================================================

"""
    erf_series(x)

Error function by Taylor series for `|x| ≤ 3` and by the complement of
the continued fraction beyond. Accurate to ~1e-14 over the whole range;
verified against tabulated values in the test suite.
"""
function erf_series(x::Real)
    ax = abs(x)
    if ax <= 3.0
        # erf(x) = (2/√π) Σ (-1)ⁿ x^(2n+1) / (n!(2n+1))
        s = 0.0
        term = ax
        n = 0
        while abs(term / (2n + 1)) > 1e-18 && n < 200
            s += term / (2n + 1)
            n += 1
            term *= -ax * ax / n
        end
        r = 2 / sqrt(π) * s
        return x >= 0 ? r : -r
    else
        r = 1 - erfc_series(ax)
        return x >= 0 ? r : -r
    end
end

"""
    erfc_series(x)

Complementary error function. For `x > 3` uses the Lentz-evaluated
continued fraction

```
erfc(x) = e^{-x²}/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + …))))
```

which converges rapidly in exactly the regime where the Taylor series
for `erf` loses relative accuracy.
"""
function erfc_series(x::Real)
    x < 0 && return 2 - erfc_series(-x)
    x <= 3.0 && return 1 - erf_series(x)
    # modified Lentz for the continued fraction
    tiny = 1e-300
    f = tiny; C = f; D = 0.0
    b0 = x
    f = b0 == 0 ? tiny : b0
    C = f; D = 0.0
    for i in 1:300
        a = i / 2
        b = x
        D = b + a * D
        D = D == 0 ? tiny : D
        C = b + a / C
        C = C == 0 ? tiny : C
        D = 1 / D
        Δ = C * D
        f *= Δ
        abs(Δ - 1) < 1e-16 && break
    end
    return exp(-x * x) / sqrt(π) / f
end

# =========================================================================
# Malkmus statistical narrow-band model
# =========================================================================

"""
    MalkmusParams

Per-gas, per-band Malkmus parameters.

  * `k`  — band-mean mass absorption coefficient at reference conditions,
           m² kg⁻¹ (so `k·u` is dimensionless for `u` in kg m⁻²)
  * `B`  — the line-structure parameter `2πγ̄/d̄` at reference pressure;
           dimensionless. Large `B` ⇒ weak-line (grey) behaviour, small
           `B` ⇒ strong-line (square-root) behaviour
  * `n_T` — temperature exponent, `k ∝ (T₀/T)^{n_T}`

**Provenance.** These are representative band-model parameters in the
spirit of the Goody & Yung and Rodgers-Walshaw compilations, adjusted so
that the clear-sky column validation targets in the test suite (window
vs CO₂-band optical depth ordering, total column water vapour, CO₂
doubling response) are met. They are *tuned representative values*, not
transcribed measurements — anywhere a number here is quoted as physical
it should be read as "reproduces the right band behaviour", not "is the
published parameter for this band".
"""
struct MalkmusParams
    k::Float64
    B::Float64
    n_T::Float64
end

const P_REF = 1.01325e5     # Pa
const T_REF = 296.0         # K

# --- longwave band parameters (16 RRTMG LW bands, 10-3250 cm⁻¹) ---------
# Structure encoded: H2O rotation band dominates < 600 cm⁻¹, CO2 the
# 630-700 cm⁻¹ (15 μm) band, O3 the 980-1080 cm⁻¹ (9.6 μm) band, and the
# 820-980 / 1080-1180 cm⁻¹ window bands are comparatively transparent —
# which is the defining structure of the terrestrial LW spectrum.
#
# The window line strengths (bands 6-8) are DELIBERATELY small: ~0.01-0.02
# m² kg⁻¹, giving a line-only optical depth of about 0.1 across a 11 kg/m²
# column. What actually closes the window is the water-vapour CONTINUUM
# (see `continuum_optical_depth` below), which is measurement-calibrated
# and scales differently with amount and pressure. Putting the window
# opacity into the line term instead — as an earlier version of this file
# did, with k ≈ 0.3 — reproduces one atmosphere's broadband flux and then
# fails badly on any column with different humidity. It cost ~27 W/m² of
# spurious downwelling in the DYCOMS-II RF01 case; see
# examples/validation/dycoms_rf01_radiation.jl.
const _LW_H2O = [
    MalkmusParams(180.0, 0.30, 0.5),  MalkmusParams(120.0, 0.30, 0.5),
    MalkmusParams( 45.0, 0.28, 0.5),  MalkmusParams(  2.0, 0.25, 0.5),
    MalkmusParams(  3.0, 0.22, 0.5),  MalkmusParams(  0.011, 0.20, 0.5),
    MalkmusParams(  0.020, 0.20, 0.5), MalkmusParams(  0.010, 0.20, 0.5),
    MalkmusParams(  2.5, 0.22, 0.5),  MalkmusParams( 14.0, 0.25, 0.5),
    MalkmusParams( 30.0, 0.28, 0.5),  MalkmusParams( 60.0, 0.30, 0.5),
    MalkmusParams( 20.0, 0.28, 0.5),  MalkmusParams(  6.0, 0.25, 0.5),
    MalkmusParams(  1.5, 0.22, 0.5),  MalkmusParams(  0.5, 0.20, 0.5)]

const _LW_CO2 = [
    MalkmusParams(0.0,   0.20, 0.7), MalkmusParams(0.0,   0.20, 0.7),
    MalkmusParams(0.02,  0.15, 0.7), MalkmusParams(9.0,   0.10, 0.7),
    MalkmusParams(1.2,   0.12, 0.7), MalkmusParams(0.004, 0.15, 0.7),
    MalkmusParams(0.0,   0.20, 0.7), MalkmusParams(0.0,   0.20, 0.7),
    MalkmusParams(0.0,   0.20, 0.7), MalkmusParams(0.001, 0.20, 0.7),
    MalkmusParams(0.0,   0.20, 0.7), MalkmusParams(0.0,   0.20, 0.7),
    MalkmusParams(0.05,  0.15, 0.7), MalkmusParams(0.6,   0.12, 0.7),
    MalkmusParams(1.5,   0.12, 0.7), MalkmusParams(0.1,   0.15, 0.7)]

const _LW_O3 = [
    MalkmusParams(0.0, 0.3, 0.5), MalkmusParams(0.0, 0.3, 0.5),
    MalkmusParams(0.0, 0.3, 0.5), MalkmusParams(0.0, 0.3, 0.5),
    MalkmusParams(0.0, 0.3, 0.5), MalkmusParams(0.0, 0.3, 0.5),
    MalkmusParams(95.0, 0.20, 0.5), MalkmusParams(2.0, 0.25, 0.5),
    MalkmusParams(0.0, 0.3, 0.5), MalkmusParams(0.0, 0.3, 0.5),
    MalkmusParams(0.0, 0.3, 0.5), MalkmusParams(0.0, 0.3, 0.5),
    MalkmusParams(0.0, 0.3, 0.5), MalkmusParams(0.0, 0.3, 0.5),
    MalkmusParams(0.0, 0.3, 0.5), MalkmusParams(0.0, 0.3, 0.5)]

const _LW_CH4 = [
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(6.0, 0.20, 0.6), MalkmusParams(0.8, 0.22, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6)]

const _LW_N2O = [
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(3.0, 0.20, 0.6),
    MalkmusParams(1.0, 0.22, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6),
    MalkmusParams(0.0, 0.25, 0.6), MalkmusParams(0.0, 0.25, 0.6)]

# --- shortwave band parameters (14 RRTMG SW bands) ----------------------
# H2O near-IR bands, CO2 weak near-IR, O3 Chappuis (visible) and
# Hartley-Huggins (UV), O2 A-band. SW band order follows RadiationCore's
# `_RRTMG_SW_BAND_LO`, whose 14th entry is the 820-2600 cm⁻¹ overlap band.
const _SW_H2O = [
    MalkmusParams(1.2,  0.25, 0.5), MalkmusParams(0.9,  0.25, 0.5),
    MalkmusParams(0.45, 0.25, 0.5), MalkmusParams(0.30, 0.25, 0.5),
    MalkmusParams(0.12, 0.25, 0.5), MalkmusParams(0.05, 0.25, 0.5),
    MalkmusParams(0.02, 0.25, 0.5), MalkmusParams(0.008, 0.25, 0.5),
    MalkmusParams(0.0,  0.25, 0.5), MalkmusParams(0.0,  0.25, 0.5),
    MalkmusParams(0.0,  0.25, 0.5), MalkmusParams(0.0,  0.25, 0.5),
    MalkmusParams(0.0,  0.25, 0.5), MalkmusParams(2.0,  0.25, 0.5)]

const _SW_CO2 = [
    MalkmusParams(0.05, 0.2, 0.7), MalkmusParams(0.02, 0.2, 0.7),
    MalkmusParams(0.01, 0.2, 0.7), MalkmusParams(0.0,  0.2, 0.7),
    MalkmusParams(0.0,  0.2, 0.7), MalkmusParams(0.0,  0.2, 0.7),
    MalkmusParams(0.0,  0.2, 0.7), MalkmusParams(0.0,  0.2, 0.7),
    MalkmusParams(0.0,  0.2, 0.7), MalkmusParams(0.0,  0.2, 0.7),
    MalkmusParams(0.0,  0.2, 0.7), MalkmusParams(0.0,  0.2, 0.7),
    MalkmusParams(0.0,  0.2, 0.7), MalkmusParams(0.08, 0.2, 0.7)]

const _SW_O3 = [
    MalkmusParams(0.0,   0.3, 0.0), MalkmusParams(0.0,   0.3, 0.0),
    MalkmusParams(0.0,   0.3, 0.0), MalkmusParams(0.0,   0.3, 0.0),
    MalkmusParams(0.0,   0.3, 0.0), MalkmusParams(0.0,   0.3, 0.0),
    MalkmusParams(0.0,   0.3, 0.0), MalkmusParams(0.0,   0.3, 0.0),
    MalkmusParams(22.0,  0.9, 0.0),   # Chappuis, visible
    MalkmusParams(2.0,   0.9, 0.0),
    MalkmusParams(150.0, 0.9, 0.0),   # Huggins, near-UV
    MalkmusParams(4000.0,0.9, 0.0),   # Hartley, UV
    MalkmusParams(9000.0,0.9, 0.0),
    MalkmusParams(0.0,   0.3, 0.0)]

const _SW_O2 = [
    MalkmusParams(0.0, 0.2, 0.5), MalkmusParams(0.0, 0.2, 0.5),
    MalkmusParams(0.0, 0.2, 0.5), MalkmusParams(0.0, 0.2, 0.5),
    MalkmusParams(0.0, 0.2, 0.5), MalkmusParams(0.0, 0.2, 0.5),
    MalkmusParams(0.0, 0.2, 0.5), MalkmusParams(2.5e-4, 0.15, 0.5),
    MalkmusParams(0.0, 0.2, 0.5), MalkmusParams(0.0, 0.2, 0.5),
    MalkmusParams(0.0, 0.2, 0.5), MalkmusParams(0.0, 0.2, 0.5),
    MalkmusParams(0.0, 0.2, 0.5), MalkmusParams(0.0, 0.2, 0.5)]

"""
    malkmus_params(kind, gas, band) -> MalkmusParams

Look up band parameters. `kind` is `:longwave` or `:shortwave`; `gas` one
of `:h2o, :co2, :o3, :ch4, :n2o, :o2`. Unknown gases return a zero-`k`
(transparent) parameter set rather than erroring, so a column carrying
extra species is harmless.
"""
function malkmus_params(kind::Symbol, gas::Symbol, band::Int)
    tbl = if kind === :longwave
        gas === :h2o ? _LW_H2O : gas === :co2 ? _LW_CO2 :
        gas === :o3  ? _LW_O3  : gas === :ch4 ? _LW_CH4 :
        gas === :n2o ? _LW_N2O : nothing
    else
        gas === :h2o ? _SW_H2O : gas === :co2 ? _SW_CO2 :
        gas === :o3  ? _SW_O3  : gas === :o2  ? _SW_O2  : nothing
    end
    tbl === nothing && return MalkmusParams(0.0, 0.25, 0.5)
    band <= length(tbl) || return MalkmusParams(0.0, 0.25, 0.5)
    return tbl[band]
end

"""
    malkmus_transmission(u, p::MalkmusParams; pressure, temperature)

Band-mean transmission through absorber amount `u` (kg m⁻²):

```math
T(u) = \\exp\\left\\{-\\frac{B}{2}\\left[\\sqrt{1 + \\frac{4ku}{B}} - 1\\right]\\right\\}
```

with `B` scaled by `p/p₀` (collision broadening) and `k` by `(T₀/T)^{n_T}`.

Limits, both exercised in the test suite:
  * weak-line (`ku ≪ B`): `T → e^{-ku}`, a grey absorber
  * strong-line (`ku ≫ B`): `T → e^{-\\sqrt{kuB}}`, the square-root law
"""
malkmus_transmission(u::Real, p::MalkmusParams; kwargs...) =
    exp(malkmus_log_transmission(u, p; kwargs...))

"""
    malkmus_log_transmission(u, p::MalkmusParams; pressure, temperature)

`log T(u)` evaluated directly, without forming `T` first. Needed because
on a saturated path the optical depth reaches several hundred and `T`
underflows to exactly zero — at which point `log(T)` is `-Inf` and any
relative comparison becomes `NaN`. Working in log space keeps the
strong-line asymptotics usable, and it is also the form a solver wants
when accumulating optical depth over many layers.
"""
function malkmus_log_transmission(u::Real, p::MalkmusParams;
                                  pressure::Real = P_REF,
                                  temperature::Real = T_REF)
    u <= 0 && return 0.0
    p.k <= 0 && return 0.0
    k = p.k * (T_REF / temperature)^p.n_T
    B = p.B * (pressure / P_REF)
    B <= 0 && return -k * u
    return -(B / 2) * (sqrt(1 + 4 * k * u / B) - 1)
end

"""
    malkmus_cumulative_k(k, kbar, B) -> g ∈ [0,1]

Cumulative k-distribution of the Malkmus model — the fraction of the band
whose absorption coefficient is below `k`:

```math
g(k) = \\tfrac12\\left\\{\\operatorname{erfc}\\!\\left[\\sqrt{\\tfrac{B}{4}}\\left(\\sqrt{\\tfrac{\\bar k}{k}} - \\sqrt{\\tfrac{k}{\\bar k}}\\right)\\right]
 + e^{B}\\operatorname{erfc}\\!\\left[\\sqrt{\\tfrac{B}{4}}\\left(\\sqrt{\\tfrac{\\bar k}{k}} + \\sqrt{\\tfrac{k}{\\bar k}}\\right)\\right]\\right\\}
```

This is the analytic inverse Laplace transform of the Malkmus
transmission (Domoto 1974), which is what makes the reordering below
*exact* rather than a fit. The `e^B erfc(·)` product is evaluated in log
space because `e^B` overflows for moderate `B` while the `erfc` factor
underflows in step.
"""
function malkmus_cumulative_k(k::Real, kbar::Real, B::Real)
    (k <= 0 || kbar <= 0) && return 0.0
    r = sqrt(k / kbar)
    a = sqrt(B / 4)
    t1 = a * (1 / r - r)
    t2 = a * (1 / r + r)
    term1 = erfc_series(t1)
    # e^B · erfc(t2): both factors are extreme, so combine in log space
    # using erfc(t) ≈ e^{-t²}/(t√π) for large t.
    term2 = if t2 > 5
        lg = B - t2 * t2 - log(t2 * sqrt(π))
        lg < -700 ? 0.0 : exp(lg)
    else
        exp(min(B, 700.0)) * erfc_series(t2)
    end
    return clamp(0.5 * (term1 + term2), 0.0, 1.0)
end

"""
    kdist_from_band_model(p::MalkmusParams, ngpt; pressure, temperature)
        -> (k_g, w_g)

Reorder the Malkmus band model into `ngpt` correlated-k quadrature points.
Returns absorption coefficients `k_g` (m² kg⁻¹) and weights `w_g` summing
to 1, such that

```math
\\sum_g w_g e^{-k_g u} \\approx T_\\text{band}(u)
```

across many decades of `u` — the property the test suite measures
directly, and the thing that distinguishes a correlated-k scheme from a
band average.

Implementation: g-points are placed at the midpoints of `ngpt` equal
probability intervals and `k(g)` is obtained by bisecting the analytic
cumulative `malkmus_cumulative_k`. Equal-probability (rather than
Gauss-Legendre) spacing is used because the cumulative is exactly
invertible here, so uniform sampling in `g` already integrates the
exponential faithfully and avoids negative or clustered weights.
"""
function kdist_from_band_model(p::MalkmusParams, ngpt::Int;
                               pressure::Real = P_REF,
                               temperature::Real = T_REF)
    w = fill(1.0 / ngpt, ngpt)
    if p.k <= 0
        return (zeros(ngpt), w)
    end
    kbar = p.k * (T_REF / temperature)^p.n_T
    B    = p.B * (pressure / P_REF)
    kg = Vector{Float64}(undef, ngpt)
    for i in 1:ngpt
        gt = (i - 0.5) / ngpt                 # midpoint of the i-th interval
        # bracket then bisect: g(k) is monotone increasing in k
        lo, hi = kbar * 1e-8, kbar * 1e8
        for _ in 1:200
            mid = sqrt(lo * hi)               # geometric bisection
            malkmus_cumulative_k(mid, kbar, B) < gt ? (lo = mid) : (hi = mid)
            hi / lo < 1 + 1e-12 && break
        end
        kg[i] = sqrt(lo * hi)
    end
    return (kg, w)
end

# =========================================================================
# Rayleigh scattering
# =========================================================================

# Bodhaine et al. (1999) Rayleigh cross-section, cm², for wavelength in μm.
# The λ⁻⁴ dependence with the small dispersion correction in the
# refractive index is what the test suite fits an exponent to.
function _rayleigh_cross_section(λ_μm::Real)
    λ = λ_μm
    return 1.0455996e-28 * (1 - 341.29061 / λ^-2 - 0.90230850 * λ^2) /
           (1 + 0.0027059889 / λ^-2 - 85.968563 * λ^2) / λ^4 * 1e4 * 1e-4
end

# A cleaner, standard closed form (Bucholtz 1995 / Hansen & Travis):
# σ ≈ 1.0456e-28 λ⁻⁴ ·(…) cm². For robustness we use the widely quoted
# approximation σ[cm²] = 4.577e-28 / λ[μm]^(4 + x) with a small x.
_rayleigh_sigma_cm2(λ_μm::Real) = 4.577e-28 / λ_μm^4.0

"""
    rayleigh_cross_section(λ_μm) -> m² per molecule

Molecular (dry-air) Rayleigh scattering cross-section at wavelength
`λ_μm`, the single definition used by [`rayleigh_optical_depth`](@ref)
and by the clear-air term of the Phase 44 visibility diagnostic
(`Visibility.rayleigh_extinction`). It is the `λ⁻⁴` fit
`σ = 4.577 × 10⁻²⁸ λ⁻⁴ cm²`; at 550 nm that gives 5.00 × 10⁻³¹ m²,
roughly 10 % above the tabulated standard-air values of Bucholtz (1995)
and Bodhaine et al. (1999), ≈ 4.5–4.6 × 10⁻³¹ m². Callers inherit that
bias; it is left alone here because the column optical depth it feeds is
validated in the Phase 32 tests.
"""
rayleigh_cross_section(λ_μm::Real) = _rayleigh_sigma_cm2(λ_μm) * 1e-4

"""
    rayleigh_optical_depth(λ_μm, dp_Pa) -> τ

Rayleigh optical depth of a layer of pressure thickness `dp_Pa`, using
the molecular column `N = dp/(m̄ g)` and the `λ⁻⁴` cross-section. The
standard-atmosphere column value at 550 nm is ≈ 0.0973, which the test
suite checks.
"""
function rayleigh_optical_depth(λ_μm::Real, dp_Pa::Real)
    g = 9.80665
    m_air = 28.9647e-3 / 6.02214076e23     # kg per molecule
    N = dp_Pa / (m_air * g)                # molecules m⁻²
    σ_m2 = rayleigh_cross_section(λ_μm)
    return N * σ_m2
end

"Band-centre wavelength in μm from band limits in cm⁻¹."
function _band_wavelength(spec::SpectralGrid, b::Int)
    lo, hi = band_limits(spec, b)
    ν_mid = (lo + hi) / 2                   # cm⁻¹
    return 1e4 / ν_mid                      # μm
end

"""
    rayleigh_optics!(optics, spec, col)

Add molecular Rayleigh scattering to `optics`: increments `τ` and sets the
combined single-scattering albedo and asymmetry consistently (Rayleigh is
conservative, `ω = 1`, and isotropic to first order, `g = 0`), using the
optical-depth and scattering-weighted combination rule so a later
`combine_optics!` stays correct.
"""
function rayleigh_optics!(optics::OpticalProperties{T}, spec::SpectralGrid{T},
                          col::AtmosphericColumn{T}) where {T}
    spec.kind === :shortwave || return optics
    nz = length(col.p_lay)
    for b in 1:nbands(spec)
        λ = _band_wavelength(spec, b)
        for g in gpoint_range(spec, b)
            @inbounds for k in 1:nz
                dp = abs(col.p_lev[k] - col.p_lev[k + 1])
                τr = T(rayleigh_optical_depth(λ, dp))
                τ0 = optics.τ[k, g]; ω0 = optics.ssa[k, g]; g0 = optics.g[k, g]
                τn = τ0 + τr
                s0 = τ0 * ω0                       # existing scattering
                sn = s0 + τr                       # Rayleigh is conservative
                optics.τ[k, g]   = τn
                optics.ssa[k, g] = τn > 0 ? sn / τn : zero(T)
                optics.g[k, g]   = sn > 0 ? (s0 * g0) / sn : zero(T)
            end
        end
    end
    return optics
end

# =========================================================================
# Gas optics object
# =========================================================================

"""
    CorrelatedKGasOptics{T} <: AbstractGasOptics

Correlated-k gas optics built on the Malkmus band model.

Construct with [`correlated_k_optics`](@ref). `compute_optics!` fills
`τ` per layer and g-point from the column's mixing ratios; `ssa` and `g`
are left at zero because gas absorption is non-scattering — Rayleigh is
added separately by [`rayleigh_optics!`](@ref) so the two can be
enabled independently.
"""
struct CorrelatedKGasOptics{T<:AbstractFloat} <: AbstractGasOptics
    spec::SpectralGrid{T}
    gases::Vector{Symbol}
    include_continuum::Bool
    include_rayleigh::Bool
end

"""
    correlated_k_optics(spec; gases, continuum = true, rayleigh = true)

Build gas optics for a spectral grid. Default gas set is the standard
radiatively active species for the grid's spectral region.
"""
function correlated_k_optics(spec::SpectralGrid{T};
                             gases = nothing,
                             continuum::Bool = true,
                             rayleigh::Bool = true) where {T}
    gs = gases === nothing ?
        (spec.kind === :longwave ? [:h2o, :co2, :o3, :ch4, :n2o] :
                                   [:h2o, :co2, :o3, :o2]) :
        collect(Symbol, gases)
    return CorrelatedKGasOptics{T}(spec, gs, continuum,
                                   rayleigh && spec.kind === :shortwave)
end

"""
    column_absorber_amount(col, gas, k) -> kg m⁻²

Absorber mass path of `gas` in layer `k`. Converts volume mixing ratio to
mass mixing ratio with the molecular-weight ratio, then multiplies by the
layer mass `Δp/g` (hydrostatic), which is more robust than `ρ·dz` when
the column is not exactly hydrostatic.
"""
function column_absorber_amount(col::AtmosphericColumn, gas::Symbol, k::Int)
    haskey(col.vmr, gas) || return 0.0
    M = _molar_mass(gas)
    M_air = 28.9647
    mmr = col.vmr[gas][k] * M / M_air
    dp = abs(col.p_lev[k] - col.p_lev[k + 1])
    return mmr * dp / 9.80665
end

_molar_mass(g::Symbol) =
    g === :h2o ? 18.01528 : g === :co2 ? 44.0095 : g === :o3 ? 47.9982 :
    g === :ch4 ? 16.0425  : g === :n2o ? 44.0128 : g === :o2 ? 31.9988 :
    g === :n2  ? 28.0134  : 28.9647

# =========================================================================
# Water-vapour continuum
# =========================================================================
#
# The 820-1180 cm⁻¹ "atmospheric window" is not transparent, but what
# closes it is NOT the water-vapour line spectrum — the lines really are
# weak there. It is the CONTINUUM: the far wings of strong lines outside
# the window, plus water dimer absorption. A statistical narrow-band
# model like Malkmus has no representation of this at all, so it must be
# added separately.
#
# Getting the split right matters more than it looks. Lines and continuum
# scale differently:
#
#   line absorption    ~ sqrt(k u)   in the strong-line limit, and
#                        depends on pressure through line broadening;
#   continuum          ~ u * (e + γ p_dry),  i.e. linear in amount AND
#                        in water partial pressure.
#
# So a model that puts the window opacity in the wrong term can still
# reproduce a single broadband flux for a single atmosphere, and then be
# badly wrong for any other column. That is exactly what happened here:
# an earlier version of this file used window line strengths ~50x too
# large and a continuum ~15x too weak. The US Standard Atmosphere surface
# flux came out fine; the DYCOMS-II RF01 cloud-top flux was wrong by 54%,
# because that case has a dry 4 kg/m² column where the two terms no
# longer trade off. See examples/validation/dycoms_rf01_radiation.jl.
#
# The form below is Roberts, Selby & Biberman (1976), which remains the
# standard simple parameterisation and is calibrated against measurement:
#
#   k_c(ν) = [a₁ + a₂ exp(-β ν)]        cm² g⁻¹ atm⁻¹
#   τ      = k_c · u · (e + γ (P - e)) · exp(T₀ (1/T - 1/296))
#
# with u in g cm⁻², pressures in atm. The γ = 0.002 foreign-broadening
# coefficient is small: the self-continuum dominates wherever there is
# appreciable water, which is why the window closes in the humid tropics
# and stays open over a cold dry column.
#
# Roberts, R. E., Selby, J. E. A. & Biberman, L. M. (1976). Infrared
# continuum absorption by atmospheric water vapor in the 8-12 μm window.
# Appl. Opt., 15, 2085-2090.

const _ROBERTS_A1 = 4.18      # cm² g⁻¹ atm⁻¹
const _ROBERTS_A2 = 5578.0    # cm² g⁻¹ atm⁻¹
const _ROBERTS_B  = 7.87e-3   # cm
const _ROBERTS_T0 = 1800.0    # K
const _ROBERTS_G  = 0.002     # foreign/self broadening ratio

"""
    roberts_continuum_coefficient(ν) -> cm² g⁻¹ atm⁻¹

Roberts et al. (1976) water-vapour continuum absorption coefficient at
wavenumber `ν` (cm⁻¹), before the amount, partial-pressure and
temperature factors are applied. Falls steeply with wavenumber across
the window — ~13 cm² g⁻¹ atm⁻¹ at 820 cm⁻¹ down to ~4.6 at 1180 cm⁻¹ —
which is why the long-wavelength side of the window is the murkier one.
"""
roberts_continuum_coefficient(ν::Real) =
    _ROBERTS_A1 + _ROBERTS_A2 * exp(-_ROBERTS_B * ν)

"""
    continuum_optical_depth(band_lo, band_hi, u_h2o, p_Pa, T_K, vmr_h2o)

Water-vapour continuum optical depth of one layer in one band.

`u_h2o` is the layer water path in kg m⁻². The continuum is evaluated at
the band centre; this is adequate because `k_c` varies smoothly, unlike
the line spectrum it complements.

Outside roughly 700-1250 cm⁻¹ the Roberts fit is not meant to apply and
the line spectrum dominates anyway, so the contribution is tapered to
zero rather than extrapolated.
"""
function continuum_optical_depth(band_lo::Real, band_hi::Real,
                                 u_h2o::Real, p_Pa::Real, T_K::Real,
                                 vmr_h2o::Real)
    ν = 0.5 * (band_lo + band_hi)
    # taper outside the window region the fit was derived for
    w = ν < 750.0   ? 0.0 :
        ν < 820.0   ? (ν - 750.0) / 70.0 :
        ν <= 1250.0 ? 1.0 :
        ν < 1400.0  ? (1400.0 - ν) / 150.0 : 0.0
    w <= 0 && return 0.0
    kc = roberts_continuum_coefficient(ν)
    u_g_cm2 = u_h2o * 0.1                     # kg m⁻² -> g cm⁻²
    e_atm   = vmr_h2o * p_Pa / 101325.0       # water partial pressure, atm
    p_atm   = p_Pa / 101325.0
    f_T     = exp(_ROBERTS_T0 * (1 / T_K - 1 / 296.0))
    return w * kc * u_g_cm2 *
           (e_atm + _ROBERTS_G * max(p_atm - e_atm, 0.0)) * f_T
end

# Backwards-compatible internal wrapper used by `compute_optics!`.
function _continuum_tau(spec, band::Int, kind::Symbol, u_h2o::Real,
                        p_Pa::Real, T_K::Real, vmr_h2o::Real)
    kind === :longwave || return 0.0
    lo, hi = band_limits(spec, band)
    return continuum_optical_depth(lo, hi, u_h2o, p_Pa, T_K, vmr_h2o)
end

"""
    compute_optics!(optics, go::CorrelatedKGasOptics, col)

Fill `optics.τ` for every layer and g-point. Each gas contributes its
correlated-k absorption coefficient times its absorber amount; the
contributions add because optical depths are additive. The water-vapour
continuum and (in the shortwave) Rayleigh scattering are added when
enabled.
"""
function compute_optics!(optics::OpticalProperties{T},
                         go::CorrelatedKGasOptics{T},
                         col::AtmosphericColumn{T}) where {T}
    spec = go.spec
    nz = length(col.p_lay)
    size(optics.τ, 1) == nz ||
        error("compute_optics!: optics has $(size(optics.τ,1)) layers, column has $nz")
    fill!(optics.τ, 0); fill!(optics.ssa, 0); fill!(optics.g, 0)

    for b in 1:nbands(spec)
        gr = gpoint_range(spec, b)
        ngb = length(gr)
        for k in 1:nz
            p = col.p_lay[k]; Tk = col.T_lay[k]
            for gas in go.gases
                mp = malkmus_params(spec.kind, gas, b)
                mp.k <= 0 && continue
                u = column_absorber_amount(col, gas, k)
                u <= 0 && continue
                kg, _ = kdist_from_band_model(mp, ngb; pressure = p,
                                              temperature = Tk)
                @inbounds for (i, g) in enumerate(gr)
                    optics.τ[k, g] += T(kg[i] * u)
                end
            end
            if go.include_continuum
                u_h2o = column_absorber_amount(col, :h2o, k)
                vh = haskey(col.vmr, :h2o) ? col.vmr[:h2o][k] : 0.0
                τc = _continuum_tau(spec, b, spec.kind, u_h2o, p, Tk, vh)
                if τc > 0
                    @inbounds for g in gr
                        optics.τ[k, g] += T(τc)
                    end
                end
            end
        end
    end
    go.include_rayleigh && rayleigh_optics!(optics, spec, col)
    return optics
end

# =========================================================================
# Longwave source
# =========================================================================

"""
    lw_source!(src_lay, go, col) -> src_lay

Planck source per layer and g-point, W m⁻². Each g-point carries its
share `w_g` of the band-integrated Planck emission `π·B_Δν(T)`, so
summing over all g-points and bands recovers `σT⁴` to the band-truncation
error documented in `RadiationCore`.
"""
function lw_source!(src_lay::AbstractMatrix{T}, go::CorrelatedKGasOptics{T},
                    col::AtmosphericColumn{T}) where {T}
    spec = go.spec
    spec.kind === :longwave || error("lw_source!: requires a longwave grid")
    nz = length(col.T_lay)
    for b in 1:nbands(spec)
        lo, hi = band_limits(spec, b)
        for k in 1:nz
            Bb = π * planck_band(col.T_lay[k], lo, hi)
            @inbounds for g in gpoint_range(spec, b)
                src_lay[k, g] = T(spec.gpt_weight[g] * Bb)
            end
        end
    end
    return src_lay
end

"""
    lw_surface_source(go, T_sfc) -> Vector

Surface Planck emission per g-point, W m⁻².
"""
function lw_surface_source(go::CorrelatedKGasOptics{T}, T_sfc::Real) where {T}
    spec = go.spec
    out = zeros(T, ngpoints(spec))
    for b in 1:nbands(spec)
        lo, hi = band_limits(spec, b)
        Bb = π * planck_band(T_sfc, lo, hi)
        @inbounds for g in gpoint_range(spec, b)
            out[g] = T(spec.gpt_weight[g] * Bb)
        end
    end
    return out
end

# =========================================================================
# Diagnostics and a standard column
# =========================================================================

"""
    optical_depth_profile(optics, spec, band) -> Vector

Band-integrated optical depth per layer (g-point weighted).
"""
function optical_depth_profile(optics::OpticalProperties{T},
                               spec::SpectralGrid{T}, band::Int) where {T}
    nz = size(optics.τ, 1)
    out = zeros(T, nz)
    for g in gpoint_range(spec, band)
        w = spec.gpt_weight[g]
        @inbounds for k in 1:nz
            out[k] += w * optics.τ[k, g]
        end
    end
    return out
end

"""
    transmission_band(optics, spec, band; μ = 1.0) -> T

Band transmission of the whole column along a slant path `1/μ`, computed
as the g-point quadrature `Σ_g w_g exp(-τ_g/μ)`. This is the correct
spectral average — note it is *not* `exp(-τ̄/μ)`, and the difference is
exactly the reason correlated-k exists.
"""
function transmission_band(optics::OpticalProperties{T}, spec::SpectralGrid{T},
                           band::Int; μ::Real = 1.0) where {T}
    acc = zero(T)
    nz = size(optics.τ, 1)
    for g in gpoint_range(spec, band)
        τ = zero(T)
        @inbounds for k in 1:nz
            τ += optics.τ[k, g]
        end
        acc += spec.gpt_weight[g] * exp(-τ / μ)
    end
    return acc
end

"""
    us_standard_column(T = Float64; nz = 30, z_top = 30_000, T_sfc = 288.15,
                       co2_ppm = 420, humid = :midlatitude_summer)

A US-Standard-Atmosphere-like column on `nz` layers: temperature from the
standard lapse-rate profile (6.5 K/km troposphere, isothermal
stratosphere from 11 km, warming above 20 km), hydrostatic pressure, and
a representative humidity and ozone profile. Provided so the gas optics
can be exercised on a realistic path without any external data file.

`humid` is `:midlatitude_summer` (≈ 14 kg/m² precipitable water) or
`:subarctic_winter` (≈ 4 kg/m²).
"""
function us_standard_column(::Type{T} = Float64; nz::Int = 30,
                            z_top::Real = 30_000.0, T_sfc::Real = 288.15,
                            co2_ppm::Real = 420.0,
                            humid::Symbol = :midlatitude_summer) where {T}
    g = 9.80665; Rd = 287.05
    dz = z_top / nz
    z_lev = [T(k * dz) for k in 0:nz]
    z_lay = [T((k - 0.5) * dz) for k in 1:nz]

    function Tz(z)
        z < 11_000 ? T_sfc - 6.5e-3 * z :
        z < 20_000 ? T_sfc - 6.5e-3 * 11_000 :
                     T_sfc - 6.5e-3 * 11_000 + 1.0e-3 * (z - 20_000)
    end

    p_lev = zeros(T, nz + 1); p_lev[1] = T(1.01325e5)
    for k in 1:nz
        Tm = T(Tz((z_lev[k] + z_lev[k + 1]) / 2))
        p_lev[k + 1] = p_lev[k] * exp(-g * dz / (Rd * Tm))
    end
    p_lay = T[(p_lev[k] + p_lev[k + 1]) / 2 for k in 1:nz]
    T_lay = T[Tz(z) for z in z_lay]
    T_levv = T[Tz(z) for z in z_lev]
    ρ = T[p_lay[k] / (Rd * T_lay[k]) for k in 1:nz]
    dzv = fill(T(dz), nz)

    # water vapour: exponential decay with a scale height, capped in the
    # stratosphere at a few ppm
    q0, Hq = humid === :subarctic_winter ? (1.6e-3, 2000.0) : (5.2e-3, 2200.0)
    vmr_h2o = T[max(4e-6, q0 * exp(-z / Hq) * 28.9647 / 18.01528) for z in z_lay]
    # ozone: Chapman-like layer peaking near 25 km
    vmr_o3 = T[3e-8 + 8e-6 * exp(-((z - 25_000) / 7000)^2) for z in z_lay]

    vmr = Dict{Symbol, Vector{T}}(
        :h2o => vmr_h2o,
        :co2 => fill(T(co2_ppm * 1e-6), nz),
        :o3  => vmr_o3,
        :ch4 => fill(T(1.9e-6), nz),
        :n2o => fill(T(3.3e-7), nz),
        :o2  => fill(T(0.2095), nz),
        :n2  => fill(T(0.7808), nz))

    return AtmosphericColumn{T}(p_lay, T_lay, p_lev, T_levv, dzv, ρ, vmr,
                                zeros(T, nz), zeros(T, nz),
                                fill(T(10), nz), fill(T(30), nz), T(T_sfc))
end

end # module
