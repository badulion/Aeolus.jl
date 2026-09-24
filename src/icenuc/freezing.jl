# Phase 43, issue #6 — immersion and deposition freezing.
#
# Units are SI throughout, matching `inp.jl`: number in m⁻³, mass in
# kg m⁻³, radius in m, surface area in m², temperature in K, pressure in
# Pa. The saturation ratio over ice `S_i = e / e_si(T)` is dimensionless
# (1 = ice saturation); compute it with
# `MoistThermodynamics.relative_humidity_ice`.
#
# Everything here is *singular* (time-independent) nucleation: a scheme
# states how many of a parcel's INP are frozen at (T, S_i), and
# `inp_nucleate_singular!` (inp.jl) turns that statement into the
# increment over what is already consumed. Δt enters only to convert
# increments into rates for the host.

# =========================================================================
# 1. Parameterisations
# =========================================================================

"""
    FreezingScheme

Supertype of the heterogeneous ice-nucleation parameterisations. Two
families, which differ in what they need from an INP mode:

  * [`SiteDensityScheme`](@ref) — ice-nucleation active surface-site
    (INAS) density `n_s(T[, S_i])` in m⁻²; a particle of radius `r` is
    frozen with probability `1 - exp(-n_s 4πr²)` (Hoose & Möhler 2012;
    Niemand et al. 2012; Ullrich et al. 2017).
  * [`AerosolCountScheme`](@ref) — INP number as a function of `T` and
    the number of particles larger than 0.5 μm diameter (DeMott et al.
    2010, 2015; Tobo et al. 2013).

Every scheme is evaluated only inside the temperature range it was
fitted over ([`scheme_temperature_range`](@ref)): warmer than the range
it predicts **no** ice (a fitted exponential extrapolated towards 0 °C
produces spurious ice where the fit is unconstrained), colder than the
range it is held at its cold-edge value. At or above
`T_freeze` every scheme returns zero, whatever its range.
"""
abstract type FreezingScheme end

"INAS-density (surface-area) freezing schemes; see [`FreezingScheme`](@ref)."
abstract type SiteDensityScheme <: FreezingScheme end

"Aerosol-number (`N_{>0.5 μm}`) freezing schemes; see [`FreezingScheme`](@ref)."
abstract type AerosolCountScheme <: FreezingScheme end

"""
    Niemand2012()

Immersion freezing on desert dust, INAS density from AIDA cloud-chamber
expansions (Niemand et al. 2012, JAS 69, 3077, their Eq. 5):

```math
n_s(T) = \\exp\\left[-0.517\\,(T - 273.15) + 8.934\\right]\\ \\mathrm{m^{-2}},
```

valid −36 to −12 °C at or above water saturation. `n_s(−20 °C) =
2.35×10⁸ m⁻²`. The default dust immersion scheme.
"""
struct Niemand2012 <: SiteDensityScheme end

"""
    Ullrich2017Immersion()

Immersion freezing on desert dust from the 11-year AIDA data set of
Ullrich et al. (2017), JAS 74, 699:

```math
n_s(T) = \\exp(150.577 - 0.517\\,T)\\ \\mathrm{m^{-2}},
```

applied over −30 to −14 °C (the range tabulated for this fit by
Marinou et al. 2019, ACP 19, 11315, Table 1). It has the same slope as
[`Niemand2012`](@ref) and sits a constant factor `e^{0.424} = 1.53`
above it.
"""
struct Ullrich2017Immersion <: SiteDensityScheme end

"""
    Ullrich2017Deposition()

Deposition nucleation on desert dust below water saturation, Ullrich et
al. (2017), JAS 74, 699, their Eq. (2.11) and Table 1 for dust:

```math
n_s(T, S_i) = \\exp\\left\\{α (S_i - 1)^{1/4}
    \\cos^2[β(T - γ)]\\,\\frac{\\mathrm{arccot}[κ(T - λ)]}{π}\\right\\} - 1,
```

with `α = 285.692`, `β = 0.017`, `γ = 256.692`, `κ = 0.080`,
`λ = 200.745`, `arccot x = π/2 - arctan x`, valid 206–240 K and
`1 < S_i ≤ S_{i,w}(T)` (water saturation). Two deliberate choices:

  * the published form is `exp{…}`, which gives `n_s = 1 m⁻²` rather
    than zero at ice saturation; `exp{…} - 1` makes the site density
    vanish continuously at `S_i = 1`. The difference is at most 1 m⁻²,
    i.e. a relative change below 10⁻⁶ wherever `n_s > 10⁶ m⁻²` and a
    frozen fraction below 10⁻¹¹ for a 1 μm particle.
  * `S_i` above water saturation is clamped to `S_{i,w}(T) =
    e_s(T)/e_{si}(T)` from `MoistThermodynamics`: above water
    saturation droplets form and the INP freeze by immersion instead.

This is the deposition scheme chosen for Phase 43. Meyers et al. (1992)
is not used: it predicts ice number from `S_i` alone, with no reference
to the INP present, and so cannot consume a prognostic INP population.
"""
struct Ullrich2017Deposition <: SiteDensityScheme end

"""
    DeMott2015(; cf = 3.0)

Immersion freezing on mineral dust in terms of `N_{>0.5 μm}`, DeMott
et al. (2015), ACP 15, 393, Eq. (2) with `α = 0`, `β = 1.25`,
`γ = 0.46`, `δ = -11.6`:

```math
n_\\mathrm{INP} = c_f\\,N_{>0.5}^{\\,β}\\,\\exp[γ(273.16 - T) + δ],
```

`n_INP` in std L⁻¹ and `N_{>0.5}` in std cm⁻³ (273.15 K, 1013.25 hPa).
The calibration factor `cf = 3` is the value DeMott et al. recommend
for immersion freezing (CFDC at 105 % RH_w under-counts by ≈ 3).
Applied over −35 to −20 °C: DeMott et al. note both their fit and D10
are weakly constrained warmer than −20 °C. Provided mainly as an
independent cross-check of [`Niemand2012`](@ref), with which DeMott et
al. report "excellent agreement" in a Saharan dust layer.
"""
Base.@kwdef struct DeMott2015 <: AerosolCountScheme
    cf::Float64 = 3.0
end

"""
    Tobo2013()

Immersion/condensation freezing by forest-site INP dominated by primary
biological particles, Tobo et al. (2013), JGR 118, 10100 (coefficients
as in DeMott et al. 2015, Eq. 2 with `cf = 1`):

```math
n_\\mathrm{INP} = N_{>0.5}^{\\,α(273.16 - T) + β}\\,\\exp[γ(273.16 - T) + δ],
```

`α = -0.074`, `β = 3.8`, `γ = 0.414`, `δ = -9.671`, same std units as
[`DeMott2015`](@ref), measured over −34 to −9 °C. Applied here to the
`N_{>0.5 μm}` of a `:biological` INP mode; the default scheme for that
species.
"""
struct Tobo2013 <: AerosolCountScheme end

"""
    scheme_temperature_range(scheme) -> (T_min, T_max) in K

Temperature range a scheme was fitted over. See [`FreezingScheme`](@ref)
for how it is used (zero warmer, held at `T_min` colder).
"""
scheme_temperature_range(::Niemand2012) = (237.15, 261.15)
scheme_temperature_range(::Ullrich2017Immersion) = (243.15, 259.15)
scheme_temperature_range(::Ullrich2017Deposition) = (206.0, 240.0)
scheme_temperature_range(::DeMott2015) = (238.15, 253.15)
scheme_temperature_range(::Tobo2013) = (239.15, 264.15)

"""
    freezing_mechanism(scheme) -> :immersion or :deposition

Which INP a scheme applies to: `:immersion` schemes act on INP immersed
in droplets, `:deposition` schemes on interstitial INP below water
saturation.
"""
freezing_mechanism(::FreezingScheme) = :immersion
freezing_mechanism(::Ullrich2017Deposition) = :deposition

# Temperature at which to evaluate a fit, or `nothing` when the scheme
# must predict no ice.
function _fit_temperature(s::FreezingScheme, T::Real, c::ThermodynamicConstants)
    Tk = float(T)
    Tk >= c.T_freeze && return nothing
    lo, hi = scheme_temperature_range(s)
    Tk > hi && return nothing
    return max(Tk, lo)
end

# Ice saturation ratio of air saturated over liquid water, e_s / e_si.
_ice_saturation_at_water_saturation(T::Real, c::ThermodynamicConstants) =
    saturation_vapor_pressure(T, c) / saturation_vapor_pressure_ice(T, c)

"""
    ice_active_site_density(scheme::SiteDensityScheme, T, S_i = Inf;
                            c = DEFAULT_CONSTANTS) -> m⁻²

INAS density `n_s` of an INAS scheme at temperature `T` (K) and, for
deposition, ice saturation ratio `S_i`. Immersion schemes ignore `S_i`
(they apply to INP inside droplets, at water saturation). Zero for
`T ≥ c.T_freeze`, for `T` warmer than the scheme's fitted range, and —
deposition only — for `S_i ≤ 1`. Non-decreasing with supercooling.
"""
function ice_active_site_density(s::Niemand2012, T::Real, S_i::Real = Inf;
                                 c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tf = _fit_temperature(s, T, c)
    Tf === nothing && return 0.0
    return exp(-0.517 * (Tf - 273.15) + 8.934)
end

function ice_active_site_density(s::Ullrich2017Immersion, T::Real, S_i::Real = Inf;
                                 c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tf = _fit_temperature(s, T, c)
    Tf === nothing && return 0.0
    return exp(150.577 - 0.517 * Tf)
end

function ice_active_site_density(s::Ullrich2017Deposition, T::Real, S_i::Real = Inf;
                                 c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    isnan(S_i) && error("ice_active_site_density: S_i is NaN")
    S_i <= 1 && return 0.0
    Tf = _fit_temperature(s, T, c)
    Tf === nothing && return 0.0
    S = min(float(S_i), _ice_saturation_at_water_saturation(T, c))
    S <= 1 && return 0.0
    α, β, γ, κ, λ = 285.692, 0.017, 256.692, 0.080, 200.745
    arccot = π / 2 - atan(κ * (Tf - λ))
    return expm1(α * (S - 1)^0.25 * cos(β * (Tf - γ))^2 * arccot / π)
end

"""
    inas_frozen_fraction(n_s, A) -> ∈ [0, 1]

Singular INAS frozen fraction of particles of surface area `A` (m²) at
site density `n_s` (m⁻²): the probability that at least one active site
lies on the surface (Poisson),

```math
f = 1 - \\exp(-n_s A) \\;\\xrightarrow{\\;n_s A \\ll 1\\;}\\; n_s A .
```

Evaluated with `expm1` so that the small-`n_s A` limit is exact to
rounding rather than lost to cancellation.

    inas_frozen_fraction(scheme, mode::ARGMode, T, S_i = Inf; c) -> ∈ [0, 1]

Number fraction of a log-normal mode frozen at `(T, S_i)`, integrating
`1 - exp(-n_s 4πr²)` over the size distribution (the quadrature of
`inp.jl`). For `n_s A/N ≪ 1` it tends to `n_s A/N` with `A` the mode
surface area ([`mode_surface_area`](@ref)).
"""
inas_frozen_fraction(n_s::Real, A::Real) = -expm1(-float(n_s) * float(A))

_inas_probability(n_s::Real) = r -> -expm1(-n_s * 4π * r^2)

function inas_frozen_fraction(s::SiteDensityScheme, m::ARGMode, T::Real,
                              S_i::Real = Inf;
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    n_s = ice_active_site_density(s, T, S_i; c)
    (n_s > 0 && m.N > 0) || return 0.0
    return _moment_fraction(_inas_probability(n_s), m.r_g, m.σ_g, 0)
end
inas_frozen_fraction(s::SiteDensityScheme, m::LogNormalMode, T::Real,
                     S_i::Real = Inf; c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    inas_frozen_fraction(s, arg_mode(m), T, S_i; c)

"""
    STANDARD_TEMPERATURE = 273.15

Reference state (K, Pa) of the "standard" concentrations (std cm⁻³,
std L⁻¹) in which [`DeMott2015`](@ref) and [`Tobo2013`](@ref) are
written.
"""
const STANDARD_TEMPERATURE = 273.15

"Standard pressure (Pa) of std concentrations; see [`STANDARD_TEMPERATURE`](@ref)."
const STANDARD_PRESSURE = 101325.0

"""
    singular_inp_concentration(scheme::AerosolCountScheme, T, N_05;
                               p = STANDARD_PRESSURE, c = DEFAULT_CONSTANTS) -> m⁻³

INP number concentration (ambient m⁻³) predicted by an aerosol-count
scheme at temperature `T` (K) and pressure `p` (Pa), given the ambient
number `N_05` (m⁻³) of particles with diameter above 0.5 μm (use
[`inp_number_above_radius`](@ref) with [`DEMOTT_CUTOFF_RADIUS`](@ref)).
The schemes are written in standard concentrations, so `N_05` is
converted to std cm⁻³ and the result back to ambient with the ideal-gas
factor `(p/p₀)(T₀/T)`. Zero for `T ≥ c.T_freeze` and warmer than the
scheme's fitted range.
"""
function singular_inp_concentration(s::AerosolCountScheme, T::Real, N_05::Real;
                                    p::Real = STANDARD_PRESSURE,
                                    c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    N_05 >= 0 || error("singular_inp_concentration: N_05 must be non-negative")
    p > 0 || error("singular_inp_concentration: p must be positive")
    Tf = _fit_temperature(s, T, c)
    (Tf === nothing || N_05 == 0) && return 0.0
    amb_per_std = (p / STANDARD_PRESSURE) * (STANDARD_TEMPERATURE / float(T))
    N_std_cm3 = N_05 / amb_per_std * 1e-6
    n_std_L = _count_scheme(s, 273.16 - Tf, N_std_cm3)
    return n_std_L * 1e3 * amb_per_std
end

_count_scheme(s::DeMott2015, ΔT, N) = s.cf * N^1.25 * exp(0.46 * ΔT - 11.6)
_count_scheme(::Tobo2013, ΔT, N) = N^(-0.074 * ΔT + 3.8) * exp(0.414 * ΔT - 9.671)

"""
    default_freezing_scheme(species::AerosolSpecies) -> (immersion = …, deposition = …)
    default_freezing_schemes(inp::INPPopulation) -> Vector of the above, one per mode

The freezing schemes used for a species unless the caller passes its
own: mineral dust — [`Niemand2012`](@ref) immersion and
[`Ullrich2017Deposition`](@ref) deposition; biological —
[`Tobo2013`](@ref) immersion and no deposition scheme (no deposition
parameterisation for biological particles is established; Kanji et al.
2017, §4); any other species — `nothing` for both (not an INP).
"""
function default_freezing_scheme(s::AerosolSpecies)
    s.name === :mineral_dust &&
        return (immersion = Niemand2012(), deposition = Ullrich2017Deposition())
    s.name === :biological &&
        return (immersion = Tobo2013(), deposition = nothing)
    return (immersion = nothing, deposition = nothing)
end

default_freezing_schemes(inp::INPPopulation) =
    [default_freezing_scheme(s) for s in inp.species]
