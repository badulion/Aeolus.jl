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
expansions (Niemand et al. 2012, JAS 69, 3077):

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
al. (2017), JAS 74, 699, with their dust coefficients:

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
parameterisation for biological particles is established; see the
review of Kanji et al. 2017); any other species — `nothing` for both (not an INP).
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

# =========================================================================
# 2. Source terms
# =========================================================================

"""
    NUCLEATED_ICE_MASS = 4π/3 × 900 kg m⁻³ × (1 μm)³ ≈ 3.77e-15 kg

Mass of a crystal freshly nucleated by deposition: a 1-μm-radius ice
sphere at 900 kg m⁻³, the nucleated-ice mass of the P3 scheme (Morrison
& Milbrandt 2015) that issue #7 couples into. Immersion-frozen crystals
instead take the mass of the droplet that froze.
"""
const NUCLEATED_ICE_MASS = 4π / 3 * 900.0 * (1e-6)^3

"""
    IceNucleationTendency

Source terms of one call to [`ice_nucleation!`](@ref), as increments
over the step `Δt` (s). All are non-negative; divide by `Δt` for rates
([`nucleation_rates`](@ref)).

| field | unit | meaning |
| :-- | :-- | :-- |
| `N_ice_imm` | m⁻³ | ice crystals formed by immersion freezing (droplets frozen by an INP) |
| `N_ice_dep` | m⁻³ | ice crystals formed by deposition nucleation (one per INP) |
| `N_ice_hom` | m⁻³ | ice crystals formed by homogeneous freezing at `T ≤ T_icenuc` |
| `M_ice`     | kg m⁻³ | ice mass formed, `= M_liq + M_vap` exactly |
| `N_inp`, `M_inp` | m⁻³, kg m⁻³ | INP moved available → consumed (dry aerosol mass) |
| `N_liq`, `M_liq` | m⁻³, kg m⁻³ | droplets and liquid water frozen (liquid sink) |
| `M_vap`     | kg m⁻³ | vapour deposited onto new crystals (vapour sink) |
| `heat`      | J m⁻³ | latent heat released, `L_f(T) M_liq + L_s(T) M_vap` |

`L_f` and `L_s` are `MoistThermodynamics.latent_heat_fusion` and
`latent_heat_sublim` at the parcel temperature (Kirchhoff-consistent
with the saturation vapour pressures). The temperature change of the
host is `heat / (ρ c_pm)` with its own moist heat capacity.
"""
struct IceNucleationTendency
    Δt::Float64
    N_ice_imm::Float64
    N_ice_dep::Float64
    N_ice_hom::Float64
    M_ice::Float64
    N_inp::Float64
    M_inp::Float64
    N_liq::Float64
    M_liq::Float64
    M_vap::Float64
    heat::Float64
end

_no_nucleation(Δt) = IceNucleationTendency(Δt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0)

"Total ice crystals formed, `N_ice_imm + N_ice_dep + N_ice_hom` (m⁻³)."
ice_number_formed(t::IceNucleationTendency) = t.N_ice_imm + t.N_ice_dep + t.N_ice_hom

"""
    nucleation_rates(t::IceNucleationTendency) -> NamedTuple

The increments of `t` as rates over its step (per second): `dNi_dt`,
`dMi_dt` (ice number and mass sources), `dNl_dt`, `dMl_dt` (liquid
sinks, positive = removed), `dMv_dt` (vapour sink), `dNinp_dt`,
`dMinp_dt` (INP consumed) and `heating` (W m⁻³).
"""
function nucleation_rates(t::IceNucleationTendency)
    r = 1 / t.Δt
    return (dNi_dt = ice_number_formed(t) * r, dMi_dt = t.M_ice * r,
            dNl_dt = t.N_liq * r, dMl_dt = t.M_liq * r, dMv_dt = t.M_vap * r,
            dNinp_dt = t.N_inp * r, dMinp_dt = t.M_inp * r, heating = t.heat * r)
end

"""
    freezing_regime(T, c = DEFAULT_CONSTANTS) -> Symbol

Which freezing physics applies at temperature `T`. The boundaries are
the **same constants** that bound the Phase 39 mixed-phase ramp
`MoistThermodynamics.liquid_fraction`, so the two cannot disagree about
where ice is possible and where it is compulsory:

| regime | `T` | `liquid_fraction(T, c)` | nucleation |
| :-- | :-- | :-- | :-- |
| `:liquid`        | `T ≥ T_freeze`            | 1       | none: no scheme returns ice, no INP is consumed |
| `:heterogeneous` | `T_icenuc < T < T_freeze` | (0, 1)  | immersion and deposition on INP only |
| `:homogeneous`   | `T ≤ T_icenuc`            | 0       | all liquid freezes; deposition on interstitial INP continues |
"""
function freezing_regime(T::Real, c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    isnan(Tk) && error("freezing_regime: T is NaN")
    Tk >= c.T_freeze && return :liquid
    Tk <= c.T_icenuc && return :homogeneous
    return :heterogeneous
end

# What one scheme says about mode i's TOTAL population: the number frozen
# by now and, for INAS schemes, the per-particle probability p(r).
_mechanism(::Nothing, inp, i, T, S_i, p, c) = (0.0, r -> 0.0)

function _mechanism(s::SiteDensityScheme, inp::INPPopulation, i, T, S_i, p, c)
    n_s = ice_active_site_density(s, T, S_i; c)
    pf = _inas_probability(n_s)
    n_s > 0 || return (0.0, pf)
    N, _ = _moments(inp, i, :total)
    r_g = inp_median_radius(inp, i; population = :total)
    return (N * _moment_fraction(pf, r_g, inp.σ_g[i], 0), pf)
end

function _mechanism(s::AerosolCountScheme, inp::INPPopulation, i, T, S_i, p, c)
    N, _ = _moments(inp, i, :total)
    N05 = inp_number_above_radius(inp, DEMOTT_CUTOFF_RADIUS; mode = i,
                                  population = :total)
    # cannot freeze more particles than the mode holds
    return (min(N, singular_inp_concentration(s, T, N05; p, c)), nothing)
end

# Singular nucleation on mode i with a fraction f of its INP immersed.
# Returns (ΔN immersion, ΔN deposition, ΔM INP) actually moved.
function _nucleate_mode!(inp::INPPopulation, i::Integer, imm, dep, f::Real,
                         T, S_i, p, c)
    N, _ = _moments(inp, i, :total)
    N > 0 || return (0.0, 0.0, 0.0)
    Nimm, pimm = _mechanism(f > 0 ? imm : nothing, inp, i, T, S_i, p, c)
    Ndep, pdep = _mechanism(f < 1 ? dep : nothing, inp, i, T, S_i, p, c)
    a_imm = f * Nimm
    a_dep = (1 - f) * Ndep
    a_imm + a_dep > 0 || return (0.0, 0.0, 0.0)
    M0 = inp.M_nuc[i]
    if pimm !== nothing && pdep !== nothing
        # both INAS: one mixed per-particle probability keeps the mass
        # selection size-resolved
        ΔN = inp_nucleate_singular!(inp, i, r -> f * pimm(r) + (1 - f) * pdep(r))
    else
        ΔN = inp_nucleate_singular!(inp, i, a_imm + a_dep)
    end
    share = a_imm / (a_imm + a_dep)
    return (ΔN * share, ΔN * (1 - share), inp.M_nuc[i] - M0)
end

"""
    ice_nucleation!(inp, T, S_i, Δt; p = STANDARD_PRESSURE,
                    N_liq = 0, M_liq = 0, immersed_fraction,
                    schemes = default_freezing_schemes(inp),
                    m_ice0 = NUCLEATED_ICE_MASS, c = DEFAULT_CONSTANTS)
        -> IceNucleationTendency

Heterogeneous (and, at the cold end, homogeneous) ice nucleation in one
parcel over a step `Δt` (s): consume INP from `inp` and return the ice
number and mass sources with the matching liquid, vapour and INP sinks
and the latent heat ([`IceNucleationTendency`](@ref)). The host owns the
hydrometeors: it adds the ice sources to its ice category and removes
the liquid and vapour; nothing here touches a hydrometeor state.

Arguments: temperature `T` (K); ice saturation ratio `S_i = e/e_si`;
pressure `p` (Pa, for the std-concentration schemes); cloud droplet
number `N_liq` (m⁻³) and liquid water content `M_liq` (kg m⁻³);
`immersed_fraction` — the fraction of available INP inside droplets,
by default 1 when liquid is present and 0 otherwise (INP larger than
0.25 μm activate as CCN readily; the rest are interstitial); `schemes`
— one `(immersion = …, deposition = …)` pair per mode (see
[`default_freezing_scheme`](@ref)); `m_ice0` — mass of a
deposition-nucleated crystal.

Physics, by [`freezing_regime`](@ref):

  * `:liquid` (`T ≥ T_freeze`) — nothing happens; `inp` is untouched.
  * `:heterogeneous` — for each mode, the immersion scheme applies to
    the immersed fraction and the deposition scheme to the rest; the
    combined singular statement is applied with
    [`inp_nucleate_singular!`](@ref), so INP are consumed only once and
    holding `(T, S_i)` fixed nucleates nothing after the first call.
    One immersion-frozen INP freezes one droplet of the mean droplet
    mass `M_liq/N_liq` (at most `N_liq` droplets freeze; if more INP
    than droplets freeze, the extra INP are inside the same ice and form
    no extra crystal). One deposition-nucleated INP makes one crystal of
    mass `m_ice0`, drawn from vapour but never more than the vapour in
    excess of ice saturation, `(S_i - 1) e_si(T)/(R_v T)`.
  * `:homogeneous` (`T ≤ T_icenuc`) — every droplet freezes
    (`N_ice_hom = N_liq`, all of `M_liq`), matching
    `liquid_fraction = 0`; the immersed INP are inside the frozen drops
    and are moved to the consumed population (counted in `N_inp` but
    forming no crystal of their own); deposition on the interstitial
    INP continues as above.

Number conservation in the heterogeneous regime:
`N_ice_imm + N_ice_dep = N_inp` whenever `N_liq` exceeds the
immersion-frozen INP. Water conservation always: `M_ice = M_liq + M_vap`.
"""
function ice_nucleation!(inp::INPPopulation, T::Real, S_i::Real, Δt::Real;
                         p::Real = STANDARD_PRESSURE,
                         N_liq::Real = 0.0, M_liq::Real = 0.0,
                         immersed_fraction::Real = (N_liq > 0 && M_liq > 0) ? 1.0 : 0.0,
                         schemes::AbstractVector = default_freezing_schemes(inp),
                         m_ice0::Real = NUCLEATED_ICE_MASS,
                         c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Δt > 0 || error("ice_nucleation!: Δt must be positive")
    isnan(S_i) && error("ice_nucleation!: S_i is NaN")
    S_i >= 0 || error("ice_nucleation!: S_i must be non-negative")
    (N_liq >= 0 && M_liq >= 0) || error("ice_nucleation!: N_liq, M_liq must be non-negative")
    (N_liq > 0) == (M_liq > 0) ||
        error("ice_nucleation!: N_liq and M_liq must both be zero or both positive")
    0 <= immersed_fraction <= 1 || error("ice_nucleation!: immersed_fraction must be in [0, 1]")
    m_ice0 > 0 || error("ice_nucleation!: m_ice0 must be positive")
    length(schemes) == inp_nmodes(inp) ||
        error("ice_nucleation!: need one scheme pair per mode ($(inp_nmodes(inp)))")

    regime = freezing_regime(T, c)
    regime === :liquid && return _no_nucleation(float(Δt))
    # no droplets, nothing to be immersed in
    f = N_liq > 0 ? float(immersed_fraction) : 0.0

    N_inp = 0.0; M_inp = 0.0; N_imm_inp = 0.0; N_dep = 0.0
    for i in 1:inp_nmodes(inp)
        s = schemes[i]
        if regime === :heterogeneous
            a, b, m = _nucleate_mode!(inp, i, s.immersion, s.deposition, f,
                                      T, S_i, p, c)
            N_imm_inp += a; N_dep += b; N_inp += a + b; M_inp += m
        else
            # deposition on the interstitial part only …
            _, b, m = _nucleate_mode!(inp, i, nothing, s.deposition, 0.0,
                                      T, S_i, p, c)
            # … weighted by the interstitial fraction
            N_dep += b; N_inp += b; M_inp += m
            # the immersed INP end up inside homogeneously frozen drops
            if f > 0 && inp.N_avail[i] > 0
                M0 = inp.M_nuc[i]
                n = inp_consume!(inp, i, f * inp.N_avail[i]; selection = :random)
                N_inp += n; M_inp += inp.M_nuc[i] - M0
            end
        end
    end

    if regime === :heterogeneous
        N_frz = min(N_imm_inp, float(N_liq))
        M_frz = N_liq > 0 ? min(N_frz * (M_liq / N_liq), float(M_liq)) : 0.0
        N_hom = 0.0
        N_ice_imm = N_frz
    else
        N_frz = float(N_liq); M_frz = float(M_liq)
        N_hom = N_frz
        N_ice_imm = 0.0
    end

    # vapour for deposition-nucleated crystals, limited to the excess over
    # ice saturation (never draws the parcel below ice saturation)
    excess = S_i > 1 ? (S_i - 1) * saturation_vapor_pressure_ice(T, c) /
                       (MoistThermodynamics.R_v(c) * float(T)) : 0.0
    M_vap = min(N_dep * m_ice0, excess)

    heat = latent_heat_fusion(T, c) * M_frz + latent_heat_sublim(T, c) * M_vap
    return IceNucleationTendency(float(Δt), N_ice_imm, N_dep, N_hom,
                                 M_frz + M_vap, N_inp, M_inp, N_frz, M_frz,
                                 M_vap, heat)
end
