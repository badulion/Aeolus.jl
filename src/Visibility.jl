"""
    Visibility

Phase 44 — visibility diagnostic for transport and air-quality use.

Meteorological visibility is the distance at which a black object against
the horizon sky can just be distinguished. Koschmieder (1924) showed that,
in a homogeneous atmosphere of extinction coefficient `β_ext`, the
apparent contrast of such an object decays as `C(x) = exp(−β_ext x)`, so
the distance at which it falls to the observer's contrast threshold `ε` is

```math
V = \\frac{-\\ln ε}{β_{ext}}.
```

Two thresholds are in use and they differ by 23 %:

  * `ε = 0.02` — Koschmieder's original value, the one behind the
    `3.912/β` rule and the one the Kunkel (1984) fit assumes.
    [`KOSCHMIEDER_THRESHOLD`](@ref), the default here.
  * `ε = 0.05` — the WMO *meteorological optical range* (MOR),
    `V = 2.996/β` (WMO-No. 8, 2018, Vol. I §9.1).
    [`MOR_THRESHOLD`](@ref).

The extinction coefficient is built bottom-up at 550 nm (the peak of
photopic sensitivity) from three contributions that simply add:

```math
β_{ext} = \\underbrace{\\int π r^2 Q_{ext}(2πr/λ, m_w)\\,n(r)\\,dr}_{\\text{droplets}}
        + \\underbrace{β_{aer}(\\mathrm{RH})}_{\\text{haze}}
        + \\underbrace{n_{air}\\,σ_R(λ)}_{\\text{Rayleigh}}
```

None of the three is re-implemented here:

  * `Q_ext` is the Phase 37 Mie solution `AerosolOptics.mie_sphere` with
    the Hale & Querry water index `AerosolOptics.water_refractive_index`.
    For `x > 10⁴` (drops above ~0.9 mm at 550 nm) the Nussenzveig &
    Wiscombe (1980) large-sphere asymptote `Q_ext = 2 + 1.9923861 x^{-2/3}`
    is used instead. Single-`x` Mie values oscillate about it by up to
    3 × 10⁻³ (interference ripple), but averaged over ±10 % in `x` the
    Mie excess `Q_ext − 2` matches the asymptote to 0.03 % at `x = 10³`
    and 0.14 % at `x = 9 × 10³`; the jump at the switch is 5 × 10⁻⁵.
  * The haze term is `AerosolOptics.bulk_optics` — κ-Köhler wet growth
    and Mie over the log-normal modes.
  * Rayleigh is `GasOptics.rayleigh_cross_section`, the same `σ_R` the
    radiation code uses for `rayleigh_optical_depth`.

Droplet spectra are given in SI (radius in m, concentration in m⁻³, liquid
water content in kg m⁻³), as in `AerosolCloud`. The effective radius of a
spectrum built from `(LWC, N_d)` comes from
`AerosolCloud.droplet_effective_radius`, with the spectral-shape factor
`k = (r_v/r_eff)³` of the Hansen gamma distribution, so the droplet
geometry is the same one the Phase 42 indirect-effect diagnostics use.

# References

  * Koschmieder, H. (1924). Theorie der horizontalen Sichtweite.
    *Beitr. Phys. freien Atm.*, 12, 33–53, 171–181.
  * Hansen, J. E. (1971). Multiple scattering of polarized light in
    planetary atmospheres. Part II. *J. Atmos. Sci.*, 28, 1400–1426.
  * Nussenzveig, H. M. & Wiscombe, W. J. (1980). Efficiency factors in
    Mie scattering. *Phys. Rev. Lett.*, 45, 1490–1494.
  * Kunkel, B. A. (1984). Parameterization of droplet terminal velocity
    and extinction coefficient in fog models. *J. Climate Appl. Meteor.*,
    23, 34–41.
  * Gultepe, I., Müller, M. D. & Boybeyi, Z. (2006). A new visibility
    parameterization for warm-fog applications in numerical weather
    prediction models. *J. Appl. Meteor. Climatol.*, 45, 1469–1480.
  * WMO (2018). *Guide to Instruments and Methods of Observation*,
    WMO-No. 8, Vol. I, Ch. 9.
"""
module Visibility

using ..AerosolOptics: mie_sphere, size_parameter, water_refractive_index,
                       bulk_optics, AerosolMode
using ..GasOptics: rayleigh_cross_section
using ..AerosolCloud: droplet_effective_radius, RHO_WATER
using ..Superdroplet: SuperdropletSet

export KOSCHMIEDER_THRESHOLD, MOR_THRESHOLD, VISIBILITY_WAVELENGTH,
       koschmieder_visibility, extinction_from_visibility,
       DropletSpectrum, MonodisperseDroplets, GammaDroplets, BinnedDroplets,
       gamma_droplets, droplet_spectrum, spectrum_number, spectrum_moment,
       spectrum_water_content, water_extinction_efficiency,
       droplet_extinction, fog_extinction, aerosol_extinction,
       rayleigh_extinction, total_extinction, visibility,
       kunkel_visibility, gultepe_visibility

# =========================================================================
# 1. Koschmieder's law
# =========================================================================

"Koschmieder's contrast threshold, `ε = 0.02` (`V = 3.912/β`). The default."
const KOSCHMIEDER_THRESHOLD = 0.02

"WMO meteorological-optical-range threshold, `ε = 0.05` (`V = 2.996/β`)."
const MOR_THRESHOLD = 0.05

"Reference wavelength for visibility, μm (peak photopic sensitivity)."
const VISIBILITY_WAVELENGTH = 0.55

"""
    koschmieder_visibility(β_ext; ε = KOSCHMIEDER_THRESHOLD) -> m

Koschmieder visibility `V = −ln(ε)/β_ext` for an extinction coefficient
`β_ext` in m⁻¹. Returns `Inf` for `β_ext ≤ 0`. Pass `ε = MOR_THRESHOLD`
for the WMO meteorological optical range, which is shorter by the factor
`ln 20 / ln 50 = 0.766`.
"""
function koschmieder_visibility(β_ext::Real; ε::Real = KOSCHMIEDER_THRESHOLD)
    0 < ε < 1 || error("koschmieder_visibility: need 0 < ε < 1, got $ε")
    β_ext > 0 || return Inf
    return -log(float(ε)) / float(β_ext)
end

"""
    extinction_from_visibility(V; ε = KOSCHMIEDER_THRESHOLD) -> m⁻¹

Inverse of [`koschmieder_visibility`](@ref): `β_ext = −ln(ε)/V`. Used to
turn an observed visibility into an extinction, e.g. to compare a
measured MOR with a model.
"""
function extinction_from_visibility(V::Real; ε::Real = KOSCHMIEDER_THRESHOLD)
    0 < ε < 1 || error("extinction_from_visibility: need 0 < ε < 1, got $ε")
    V > 0 || error("extinction_from_visibility: V must be positive")
    return isinf(V) ? 0.0 : -log(float(ε)) / float(V)
end

# =========================================================================
# 2. Droplet spectra
# =========================================================================

"""
    DropletSpectrum

Abstract supertype of a liquid-droplet size distribution. Concrete types:
[`MonodisperseDroplets`](@ref), [`GammaDroplets`](@ref),
[`BinnedDroplets`](@ref). All carry radius in m and number in m⁻³.
"""
abstract type DropletSpectrum end

"""
    MonodisperseDroplets(N, r)

`N` droplets per m³, all of radius `r` (m). The analytic reference case:
`β_ext = N π r² Q_ext(2πr/λ)`.
"""
struct MonodisperseDroplets <: DropletSpectrum
    N::Float64
    r::Float64
    function MonodisperseDroplets(N::Real, r::Real)
        N >= 0 || error("MonodisperseDroplets: N must be ≥ 0")
        r > 0 || error("MonodisperseDroplets: r must be > 0")
        new(float(N), float(r))
    end
end

"""
    GammaDroplets(N, r_eff, v_eff)

Hansen (1971) gamma distribution of `N` droplets per m³ with effective
radius `r_eff` (m) and effective variance `v_eff` (0 < v_eff < 0.5):

```math
n(r) ∝ r^{(1-3b)/b}\\,e^{-r/(ab)}, \\qquad a = r_{eff},\\; b = v_{eff}.
```

Its moments are analytic, `⟨r^k⟩ = (ab)^k Γ(α+1+k)/Γ(α+1)` with
`α = (1−3b)/b`, which gives exactly `⟨r³⟩/⟨r²⟩ = a` and the shape factor
`k = (r_v/r_eff)³ = (1−b)(1−2b)` (0.72 at the stratocumulus/fog value
`b = 0.1`, inside the Martin et al. 1994 range 0.67–0.80).
"""
struct GammaDroplets <: DropletSpectrum
    N::Float64
    r_eff::Float64
    v_eff::Float64
    function GammaDroplets(N::Real, r_eff::Real, v_eff::Real)
        N >= 0 || error("GammaDroplets: N must be ≥ 0")
        r_eff > 0 || error("GammaDroplets: r_eff must be > 0")
        0 < v_eff < 0.5 || error("GammaDroplets: need 0 < v_eff < 0.5, got $v_eff")
        new(float(N), float(r_eff), float(v_eff))
    end
end

"""
    BinnedDroplets(r, n)

A discrete spectrum: `n[i]` droplets per m³ at radius `r[i]` (m). This is
the form a bin scheme, a disdrometer or a superdroplet ensemble
(see [`droplet_spectrum`](@ref)) naturally produces.
"""
struct BinnedDroplets <: DropletSpectrum
    r::Vector{Float64}
    n::Vector{Float64}
    function BinnedDroplets(r::AbstractVector{<:Real}, n::AbstractVector{<:Real})
        length(r) == length(n) || error("BinnedDroplets: length mismatch")
        all(>(0), r) || error("BinnedDroplets: radii must be > 0")
        all(>=(0), n) || error("BinnedDroplets: concentrations must be ≥ 0")
        new(Float64.(r), Float64.(n))
    end
end

_gamma_alpha(s::GammaDroplets) = (1 - 3 * s.v_eff) / s.v_eff
_gamma_scale(s::GammaDroplets) = s.r_eff * s.v_eff

"""
    spectrum_number(spec) -> m⁻³

Total droplet number concentration.
"""
spectrum_number(s::MonodisperseDroplets) = s.N
spectrum_number(s::GammaDroplets) = s.N
spectrum_number(s::BinnedDroplets) = sum(s.n; init = 0.0)

"""
    spectrum_moment(spec, k::Integer) -> m^k

Number-weighted mean `⟨r^k⟩` of the spectrum (not multiplied by `N`).
Analytic for the monodisperse and gamma forms.
"""
spectrum_moment(s::MonodisperseDroplets, k::Integer) = s.r^k
function spectrum_moment(s::GammaDroplets, k::Integer)
    k >= 0 || error("spectrum_moment: k must be ≥ 0 for GammaDroplets")
    α = _gamma_alpha(s); θ = _gamma_scale(s)
    p = 1.0
    for j in 1:k
        p *= (α + j)
    end
    return θ^k * p
end
function spectrum_moment(s::BinnedDroplets, k::Integer)
    N = spectrum_number(s)
    N > 0 || return 0.0
    return sum(s.n[i] * s.r[i]^k for i in eachindex(s.r)) / N
end

"""
    spectrum_water_content(spec) -> kg m⁻³

Liquid water content `(4π/3) ρ_w N ⟨r³⟩`, with `ρ_w =
AerosolCloud.RHO_WATER`.
"""
spectrum_water_content(s::DropletSpectrum) =
    4π / 3 * RHO_WATER * spectrum_number(s) * spectrum_moment(s, 3)

"""
    gamma_droplets(lwc, N_d; v_eff = 0.1) -> DropletSpectrum

The droplet spectrum implied by a bulk liquid water content `lwc`
(kg m⁻³) and droplet number `N_d` (m⁻³): a [`GammaDroplets`](@ref) whose
effective radius is `AerosolCloud.droplet_effective_radius(lwc, N_d; k)`
with the gamma shape factor `k = (1−v_eff)(1−2v_eff)`, so that
[`spectrum_water_content`](@ref) returns `lwc` exactly. `v_eff = 0` gives
a [`MonodisperseDroplets`](@ref) at the mean-volume radius.

`v_eff = 0.1` is a representative fog/stratus value; measured radiation
fog spectra span roughly 0.05–0.3 and are often bimodal.
"""
function gamma_droplets(lwc::Real, N_d::Real; v_eff::Real = 0.1)
    (lwc > 0 && N_d > 0) || error("gamma_droplets: lwc and N_d must be > 0")
    if v_eff == 0
        return MonodisperseDroplets(N_d, droplet_effective_radius(lwc, N_d; k = 1.0))
    end
    0 < v_eff < 0.5 || error("gamma_droplets: need 0 ≤ v_eff < 0.5")
    k = (1 - v_eff) * (1 - 2 * v_eff)
    return GammaDroplets(N_d, droplet_effective_radius(lwc, N_d; k = k), v_eff)
end

"""
    droplet_spectrum(sd::SuperdropletSet; volume, cell = nothing) -> BinnedDroplets

The Phase 41 superdroplet ensemble as a [`BinnedDroplets`](@ref)
spectrum: each superdroplet contributes `ξ/volume` droplets per m³ at its
wet radius. `volume` (m³) is the volume the (selected) superdroplets
occupy; `cell` restricts to one Eulerian cell.
"""
function droplet_spectrum(sd::SuperdropletSet; volume::Real,
                          cell::Union{Nothing,Integer} = nothing)
    volume > 0 || error("droplet_spectrum: volume must be > 0")
    idx = [i for i in eachindex(sd.ξ)
           if sd.ξ[i] > 0 && (cell === nothing || sd.cell[i] == cell)]
    return BinnedDroplets(Float64.(sd.radius[idx]), sd.ξ[idx] ./ float(volume))
end

# =========================================================================
# 3. Droplet extinction
# =========================================================================

# Above this size parameter the Mie series (≈ x + 4x^{1/3} terms) is
# replaced by the large-sphere asymptote. Measured for water at 550 nm:
# relative jump in Q_ext across the switch 5.5e-5; single-x Mie differs
# from the asymptote by up to 2.6e-3 below it because of ripple.
const _X_ASYMPTOTIC = 1.0e4

# Quadrature density in size parameter for spectrum integrals (see
# `droplet_extinction`).
const _POINTS_PER_X = 10

"""
    water_extinction_efficiency(r, λ = VISIBILITY_WAVELENGTH) -> Q_ext

Extinction efficiency of a liquid-water sphere of radius `r` (m) at
wavelength `λ` (μm). Mie (`AerosolOptics.mie_sphere` with
`water_refractive_index(λ)`) up to size parameter `x = 10⁴`; beyond,
the Nussenzveig-Wiscombe asymptote `Q_ext = 2 + 1.9923861 x^{-2/3}`,
which tends to the geometric-optics extinction-paradox limit 2.
"""
function water_extinction_efficiency(r::Real, λ::Real = VISIBILITY_WAVELENGTH)
    x = size_parameter(float(r) * 1e6, float(λ))
    x > 0 || return 0.0
    x > _X_ASYMPTOTIC && return 2 + 1.9923861 * x^(-2 / 3)
    return mie_sphere(x, water_refractive_index(λ)).Q_ext
end

"""
    droplet_extinction(spec; λ = 0.55, method = :mie, nquad = 200) -> m⁻¹

Extinction coefficient of a droplet spectrum,
`β = ∫ π r² Q_ext(r) n(r) dr`.

  * `method = :mie` — `Q_ext` from [`water_extinction_efficiency`](@ref).
  * `method = :geometric` — the large-particle limit `Q_ext = 2`, i.e.
    `β = 2πN⟨r²⟩ = 3 LWC / (2 ρ_w r_eff)`. This is the extinction implied
    by the `τ = 3 LWP/(2ρ_w r_eff)` cloud optical depth of
    `AerosolCloud.cloud_optical_depth_lwp` and `TwoStream`, and it is
    what the bulk parameterisations are built on. For fog spectra
    (r_eff 2–15 μm at 550 nm) Mie is 3–14 % larger; the excess falls with
    size as the edge term `≈ 2 x^{-2/3}`.

For [`GammaDroplets`](@ref) the integral is composite Simpson in `ln r`
over `[θ·max(α+1−6√(α+1), 10⁻³(α+1)), θ(α+3+10√(α+3))]` (`θ = ab`),
covering the `r²`-weighted integrand to beyond 6 standard deviations,
with the weights normalised numerically so the result is exactly `N`
times the discrete mean of `π r² Q_ext`. The number of intervals is at
least `nquad` and at least 10 per unit of size parameter: water is so
weakly absorbing at 550 nm that `Q_ext(x)` carries narrow resonance
spikes on top of the ripple, and a coarser rule (5 points per `π`)
aliased them into errors of up to 5 × 10⁻³. Measured against a
60 000-interval reference over `r_eff = 2–30 μm`, `v_eff = 0.05–0.3`:
worst 1.9 × 10⁻³ (at `r_eff = 2 μm`), typically 3 × 10⁻⁴. Cost is
~15 ms per call at `r_eff = 8 μm`; use `method = :geometric` (analytic)
where that matters.
"""
function droplet_extinction(s::MonodisperseDroplets;
                            λ::Real = VISIBILITY_WAVELENGTH,
                            method::Symbol = :mie, nquad::Int = 200)
    Q = _q(s.r, λ, method)
    return s.N * π * s.r^2 * Q
end

function droplet_extinction(s::BinnedDroplets;
                            λ::Real = VISIBILITY_WAVELENGTH,
                            method::Symbol = :mie, nquad::Int = 200)
    β = 0.0
    @inbounds for i in eachindex(s.r)
        s.n[i] > 0 || continue
        β += s.n[i] * π * s.r[i]^2 * _q(s.r[i], λ, method)
    end
    return β
end

function droplet_extinction(s::GammaDroplets;
                            λ::Real = VISIBILITY_WAVELENGTH,
                            method::Symbol = :mie, nquad::Int = 200)
    s.N > 0 || return 0.0
    if method === :geometric
        return 2π * s.N * spectrum_moment(s, 2)
    end
    method === :mie || error("droplet_extinction: method must be :mie or :geometric")
    α = _gamma_alpha(s); θ = _gamma_scale(s)
    lo = θ * max(α + 1 - 6 * sqrt(α + 1), 1e-3 * (α + 1))
    hi = θ * (α + 3 + 10 * sqrt(α + 3))
    xspan = size_parameter(hi * 1e6, λ) - size_parameter(lo * 1e6, λ)
    nq = max(nquad, ceil(Int, _POINTS_PER_X * xspan))
    nq += isodd(nq)
    a = log(lo); h = (log(hi) - a) / nq
    # number per ln r ∝ r^{α+1} e^{-r/θ}; log-weights avoid overflow
    lwmax = (α + 1) * log(α + 1) - (α + 1)          # value at the mode r = (α+1)θ
    W = 0.0; S = 0.0
    @inbounds for i in 0:nq
        w = (i == 0 || i == nq) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        r = exp(a + i * h)
        u = r / θ
        pdf = exp((α + 1) * log(u) - u - lwmax)
        W += w * pdf
        S += w * pdf * π * r^2 * water_extinction_efficiency(r, λ)
    end
    return s.N * S / W
end

function _q(r, λ, method::Symbol)
    method === :geometric && return 2.0
    method === :mie && return water_extinction_efficiency(r, λ)
    error("droplet_extinction: method must be :mie or :geometric, got :$method")
end

"""
    fog_extinction(lwc, N_d; v_eff = 0.1, λ = 0.55, method = :mie) -> m⁻¹

Droplet extinction for a bulk liquid water content `lwc` (kg m⁻³) and
droplet number `N_d` (m⁻³), through [`gamma_droplets`](@ref) and
[`droplet_extinction`](@ref). At fixed `N_d` and spectral shape,
`β ∝ N_d^{1/3} LWC^{2/3}` in the geometric limit.
"""
fog_extinction(lwc::Real, N_d::Real; v_eff::Real = 0.1,
               λ::Real = VISIBILITY_WAVELENGTH, method::Symbol = :mie) =
    lwc > 0 && N_d > 0 ?
        droplet_extinction(gamma_droplets(lwc, N_d; v_eff = v_eff);
                           λ = λ, method = method) : 0.0

# =========================================================================
# 4. Haze and clear air
# =========================================================================

"""
    aerosol_extinction(modes; rh = 0.0, λ = 0.55, T_K = 298.15) -> m⁻¹

Extinction of an interstitial (unactivated) aerosol population, i.e.
haze: `AerosolOptics.bulk_optics(modes, λ; rh)`, with κ-Köhler wet growth
of each mode at relative humidity `rh` (fraction, < 1). Use
`AerosolOptics.aerosol_mixture(:urban)` etc. for the OPAC mixtures.

At 550 nm and RH = 0.8 the library mixtures give V(ε = 0.02) of
177 km (continental clean), 26 km (continental polluted) and 12 km
(urban), falling to 60, 9 and 5 km at RH = 0.95.
"""
aerosol_extinction(modes::AbstractVector{<:AerosolMode}; rh::Real = 0.0,
                   λ::Real = VISIBILITY_WAVELENGTH, T_K::Real = 298.15) =
    isempty(modes) ? 0.0 : first(bulk_optics(modes, λ; rh = rh, T_K = T_K))

const _K_BOLTZMANN = 1.380649e-23   # J K⁻¹ (exact, SI 2019)

"""
    rayleigh_extinction(; p = 101325, T = 288.15, λ = 0.55) -> m⁻¹

Molecular scattering of particle-free air, `β_R = n σ_R(λ)` with the
number density `n = p/(k_B T)` and `σ_R = GasOptics.rayleigh_cross_section`.
At 1013.25 hPa, 288.15 K and 550 nm this is 1.27 × 10⁻⁵ m⁻¹, i.e. a
Rayleigh-limited visibility of 307 km at `ε = 0.02` — the upper bound on
visibility in the lower atmosphere.
"""
rayleigh_extinction(; p::Real = 101325.0, T::Real = 288.15,
                    λ::Real = VISIBILITY_WAVELENGTH) =
    float(p) / (_K_BOLTZMANN * float(T)) * rayleigh_cross_section(λ)

"""
    total_extinction(; droplets = nothing, aerosol = nothing, rh = 0.0,
                     p = 101325, T = 288.15, λ = 0.55,
                     rayleigh = true, method = :mie) -> m⁻¹

Sum of droplet, haze and molecular extinction. `droplets` is a
[`DropletSpectrum`](@ref) (or `nothing`), `aerosol` a vector of
`AerosolOptics` modes (or `nothing`). Extinction coefficients add because
the three populations scatter independently.
"""
function total_extinction(; droplets::Union{Nothing,DropletSpectrum} = nothing,
                          aerosol::Union{Nothing,AbstractVector{<:AerosolMode}} = nothing,
                          rh::Real = 0.0, p::Real = 101325.0, T::Real = 288.15,
                          λ::Real = VISIBILITY_WAVELENGTH,
                          rayleigh::Bool = true, method::Symbol = :mie)
    β = 0.0
    droplets === nothing || (β += droplet_extinction(droplets; λ = λ, method = method))
    aerosol === nothing || (β += aerosol_extinction(aerosol; rh = rh, λ = λ))
    rayleigh && (β += rayleigh_extinction(; p = p, T = T, λ = λ))
    return β
end

"""
    visibility(; ε = KOSCHMIEDER_THRESHOLD, kwargs...) -> m

Koschmieder visibility of the atmosphere described by `kwargs` (see
[`total_extinction`](@ref)): `koschmieder_visibility(total_extinction(;
kwargs...); ε)`.
"""
visibility(; ε::Real = KOSCHMIEDER_THRESHOLD, kwargs...) =
    koschmieder_visibility(total_extinction(; kwargs...); ε = ε)

# =========================================================================
# 5. Empirical fog parameterisations
# =========================================================================

"""
    kunkel_visibility(lwc) -> m

Kunkel (1984) fog visibility, `β = 144.7 LWC^{0.88}` km⁻¹ with LWC in
g m⁻³, converted through `ε = 0.02`: `V = 27 m × LWC^{−0.88}`. Here `lwc`
is in kg m⁻³ like the rest of the module. Fitted to fog drop spectra; it
carries no droplet-number information, so it cannot see the aerosol
indirect effect on visibility.
"""
function kunkel_visibility(lwc::Real)
    lwc > 0 || return Inf
    return 27.0 * (1e3 * float(lwc))^(-0.88)
end

"""
    gultepe_visibility(lwc, N_d) -> m

Gultepe et al. (2006) warm-fog visibility,
`V = 1.002 km / (LWC · N_d)^{0.6473}` with LWC in g m⁻³ and N_d in cm⁻³,
fitted to the FRAM-C observations (Nova Scotia). Here `lwc` is in
kg m⁻³ and `N_d` in m⁻³. Unlike [`kunkel_visibility`](@ref) it decreases
with droplet number at fixed water content, as Koschmieder with
`β ∝ N_d^{1/3}LWC^{2/3}` does, although with a different exponent.
"""
function gultepe_visibility(lwc::Real, N_d::Real)
    (lwc > 0 && N_d > 0) || return Inf
    return 1002.0 * (1e3 * float(lwc) * 1e-6 * float(N_d))^(-0.6473)
end

end # module Visibility
