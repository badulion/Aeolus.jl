"""
    Photosynthesis

Leaf-level biochemical photosynthesis for Era V (Phase 47): the
Farquhar, von Caemmerer & Berry (1980) model for C3 leaves and the
Collatz, Ribas-Carbo & Berry (1992) model for C4 leaves, in the form
used by the Community Land Model (CLM4.5 / CLM5, Bonan et al. 2011).

It answers one question — *given the CO₂ inside a leaf, the light it
absorbs and its temperature, how fast does it fix carbon?* — and is the
single owner in Aeolus of that biochemistry, of its temperature
responses, of the PAR → photon-flux conversion and of the per-PFT
photosynthetic parameter table. Stomatal conductance (#20), the
vegetated land column (#21) and the leaf energy balance of the resolved
canopy (#24) delegate here.

# Units

| Quantity | Unit |
| :-- | :-- |
| rates `Vcmax`, `Jmax`, `Tp`, `Rd`, `A` | µmol m⁻² (leaf) s⁻¹ |
| CO₂ `ci`, `Kc`, `Γ*` | µmol mol⁻¹ (mole fraction) |
| O₂ `oi`, `Ko` | mmol mol⁻¹ |
| absorbed photon flux `Q` | µmol photons m⁻² s⁻¹ |
| C4 PEP-carboxylase slope `kp` | µmol m⁻² s⁻¹ per (mol mol⁻¹) |
| leaf temperature | K |

CLM writes the same equations in partial pressures, with every
Michaelis constant and compensation point scaled by ``P_{atm}``
(`Kc25 = 404.9e-6 Patm`, …). The Farquhar rates are homogeneous of
degree zero in those quantities, so working in mole fractions is
*exactly* equivalent and removes the pressure argument.

# References

  * Farquhar, G. D., von Caemmerer, S. & Berry, J. A. (1980). A
    biochemical model of photosynthetic CO₂ assimilation in leaves of C3
    species. *Planta*, 149, 78-90.
  * von Caemmerer, S. (2000). *Biochemical Models of Leaf
    Photosynthesis*. CSIRO Publishing.
  * Collatz, G. J., Ball, J. T., Grivet, C. & Berry, J. A. (1991).
    Physiological and environmental regulation of stomatal conductance,
    photosynthesis and transpiration. *Agric. For. Meteorol.*, 54, 107-136.
  * Collatz, G. J., Ribas-Carbo, M. & Berry, J. A. (1992). Coupled
    photosynthesis-stomatal conductance model for leaves of C4 plants.
    *Aust. J. Plant Physiol.*, 19, 519-538.
  * Bernacchi, C. J., Singsaas, E. L., Pimentel, C., Portis, A. R. &
    Long, S. P. (2001). Improved temperature response functions for
    models of Rubisco-limited photosynthesis. *Plant Cell Environ.*, 24,
    253-259.
  * Kattge, J. & Knorr, W. (2007). Temperature acclimation in a
    biochemical model of photosynthesis: a reanalysis of data from 36
    species. *Plant Cell Environ.*, 30, 1176-1190.
  * Medlyn, B. E. et al. (2002). Temperature response of parameters of a
    biochemically based model of photosynthesis. II. A review of
    experimental data. *Plant Cell Environ.*, 25, 1167-1179.
  * Bonan, G. B. et al. (2011). Improving canopy processes in the
    Community Land Model version 4 (CLM4) using global flux fields
    empirically inferred from FLUXNET data. *J. Geophys. Res.*, 116,
    G02014.
  * Oleson, K. W. et al. (2013). *Technical description of version 4.5 of
    the Community Land Model (CLM)*. NCAR/TN-503+STR, §8.2-8.3, Tables
    8.1-8.2.
  * Lawrence, D. M. et al. (2018). *CLM5 Documentation*, NCAR, §9.4.
"""
module Photosynthesis

using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS

export AbstractTemperatureResponse, ArrheniusResponse, PeakedArrheniusResponse,
       Q10Response, temperature_factor, optimum_temperature,
       C3TemperatureResponses, bonan2011_c3_temperature,
       kattge_knorr_c3_temperature, kattge_knorr_jmax_ratio

# Reference temperature of every "…25" parameter.
const T_REF = 298.15

# Universal gas constant: one source of truth, MoistThermodynamics.
_gas_constant() = DEFAULT_CONSTANTS.gas_constant
_T_freeze()     = DEFAULT_CONSTANTS.T_freeze

# =========================================================================
# 1. Temperature responses
# =========================================================================

"""
    AbstractTemperatureResponse

A multiplicative temperature scaling ``f(T)`` of a rate or kinetic
constant, normalised to ``f(298.15\\,\\mathrm{K}) = 1``. Implementers
extend [`temperature_factor`](@ref).
"""
abstract type AbstractTemperatureResponse end

"""
    ArrheniusResponse(ΔHa)

Plain Arrhenius scaling normalised at 25 °C (CLM4.5 eq. 8.10):

```math
f(T) = \\exp\\left[\\frac{ΔH_a}{R\\,T_{25}}\\left(1 - \\frac{T_{25}}{T}\\right)\\right]
```

`ΔHa` in J mol⁻¹. Used for the Michaelis constants `Kc`, `Ko` and the
photorespiratory compensation point `Γ*` (Bernacchi et al. 2001).
"""
struct ArrheniusResponse <: AbstractTemperatureResponse
    ΔHa::Float64
end

"""
    PeakedArrheniusResponse(ΔHa, ΔHd, ΔS)

Arrhenius activation multiplied by high-temperature deactivation
(Johnson-Eyring-Williams / Medlyn et al. 2002 eq. 18; CLM4.5 eqs.
8.10-8.11), normalised at 25 °C:

```math
f(T) = \\exp\\left[\\frac{ΔH_a}{R T_{25}}\\left(1-\\frac{T_{25}}{T}\\right)\\right]
       \\frac{1 + \\exp\\left(\\frac{T_{25} ΔS - ΔH_d}{R T_{25}}\\right)}
             {1 + \\exp\\left(\\frac{T ΔS - ΔH_d}{R T}\\right)}
```

`ΔHa`, `ΔHd` in J mol⁻¹, `ΔS` in J mol⁻¹ K⁻¹. It has a single maximum at
[`optimum_temperature`](@ref).
"""
struct PeakedArrheniusResponse <: AbstractTemperatureResponse
    ΔHa::Float64
    ΔHd::Float64
    ΔS::Float64
end

"""
    Q10Response(; Q10 = 2.0, s_high = 0.0, T_high = Inf, s_low = 0.0, T_low = -Inf)

The Collatz et al. (1992) form used for C4 leaves (CLM4.5 eqs.
8.12-8.14):

```math
f(T) = \\frac{Q_{10}^{(T - T_{25})/10}}
            {\\left[1 + e^{s_h (T - T_h)}\\right]\\left[1 + e^{s_l (T_l - T)}\\right]}
```

The inhibition terms are *not* renormalised at 25 °C, exactly as in CLM
(for CLM's C4 `Vcmax` they reduce ``f(T_{25})`` to 0.953). A term is
omitted when its slope is zero.
"""
Base.@kwdef struct Q10Response <: AbstractTemperatureResponse
    Q10::Float64    = 2.0
    s_high::Float64 = 0.0
    T_high::Float64 = Inf
    s_low::Float64  = 0.0
    T_low::Float64  = -Inf
end

"""
    temperature_factor(resp::AbstractTemperatureResponse, T) -> Float64

Multiplicative factor on the 25 °C value at leaf temperature `T` (K).
"""
function temperature_factor end

function temperature_factor(r::ArrheniusResponse, T::Real)
    R = _gas_constant()
    return exp(r.ΔHa / (R * T_REF) * (1 - T_REF / T))
end

function temperature_factor(r::PeakedArrheniusResponse, T::Real)
    R = _gas_constant()
    fa = exp(r.ΔHa / (R * T_REF) * (1 - T_REF / T))
    fh = (1 + exp((T_REF * r.ΔS - r.ΔHd) / (R * T_REF))) /
         (1 + exp((T * r.ΔS - r.ΔHd) / (R * T)))
    return fa * fh
end

function temperature_factor(r::Q10Response, T::Real)
    f = r.Q10^((T - T_REF) / 10)
    r.s_high == 0 || (f /= 1 + exp(r.s_high * (T - r.T_high)))
    r.s_low  == 0 || (f /= 1 + exp(r.s_low * (r.T_low - T)))
    return f
end

"""
    optimum_temperature(r::PeakedArrheniusResponse) -> K

Closed-form temperature of the maximum of a peaked Arrhenius response
(Medlyn et al. 2002, eq. 19):

```math
T_{opt} = \\frac{ΔH_d}{ΔS - R \\ln\\left(\\frac{ΔH_a}{ΔH_d - ΔH_a}\\right)}
```
"""
function optimum_temperature(r::PeakedArrheniusResponse)
    R = _gas_constant()
    return r.ΔHd / (r.ΔS - R * log(r.ΔHa / (r.ΔHd - r.ΔHa)))
end

"""
    C3TemperatureResponses(; Vcmax, Jmax, Tp, Rd, Kc, Ko, Γstar)

The temperature responses of the seven C3 quantities. Construct one with
[`bonan2011_c3_temperature`](@ref) (no acclimation, CLM4.5 Table 8.2) or
[`kattge_knorr_c3_temperature`](@ref) (acclimated to the growth
temperature, CLM4.5 eq. 8.15 / CLM5 eq. 9.15).
"""
Base.@kwdef struct C3TemperatureResponses{V<:AbstractTemperatureResponse,
                                          J<:AbstractTemperatureResponse,
                                          P<:AbstractTemperatureResponse,
                                          D<:AbstractTemperatureResponse,
                                          C<:AbstractTemperatureResponse,
                                          O<:AbstractTemperatureResponse,
                                          G<:AbstractTemperatureResponse}
    Vcmax::V
    Jmax::J
    Tp::P
    Rd::D
    Kc::C    = ArrheniusResponse(79430.0)
    Ko::O    = ArrheniusResponse(36380.0)
    Γstar::G = ArrheniusResponse(37830.0)
end

# Bernacchi et al. (2001) activation energies for Rubisco kinetics and
# day respiration (CLM4.5 Table 8.2, CLM5 Table 9.2).
const _RD_RESPONSE = PeakedArrheniusResponse(46390.0, 150650.0, 490.0)

"""
    bonan2011_c3_temperature() -> C3TemperatureResponses

Non-acclimating C3 temperature responses of Bonan et al. (2011), as
tabulated in CLM4.5 Table 8.2 (J mol⁻¹, J mol⁻¹ K⁻¹):

| | ΔHa | ΔHd | ΔS |
| :-- | --: | --: | --: |
| Vcmax, Tp | 65330 | 149250 | 485 |
| Jmax | 43540 | 152040 | 495 |
| Rd | 46390 | 150650 | 490 |
| Kc | 79430 | – | – |
| Ko | 36380 | – | – |
| Γ* | 37830 | – | – |

The activation energies of `Vcmax`, `Kc`, `Ko`, `Γ*` and `Rd` are those
of Bernacchi et al. (2001); that of `Jmax` is Bernacchi et al. (2003).
"""
function bonan2011_c3_temperature()
    vc = PeakedArrheniusResponse(65330.0, 149250.0, 485.0)
    return C3TemperatureResponses(Vcmax = vc,
                                  Jmax = PeakedArrheniusResponse(43540.0, 152040.0, 495.0),
                                  Tp = vc, Rd = _RD_RESPONSE)
end

# Kattge & Knorr acclimation is fitted for growth temperatures of
# 11-35 °C; CLM clamps to that range and so do we.
_growth_celsius(T_growth::Real) = clamp(T_growth - _T_freeze(), 11.0, 35.0)

"""
    kattge_knorr_c3_temperature(T_growth) -> C3TemperatureResponses

C3 temperature responses acclimated to the growth temperature `T_growth`
(K; CLM uses the 10-day mean air temperature), after Kattge & Knorr
(2007) as implemented in CLM4.5 eq. 8.15 / CLM5 eq. 9.15:

  * `Vcmax`, `Tp`: ΔHa = 72000, ΔHd = 200000, ΔS = 668.39 − 1.07 t_g
  * `Jmax`: ΔHa = 50000, ΔHd = 200000, ΔS = 659.70 − 0.75 t_g

with ``t_g`` the growth temperature in °C, clamped to 11-35 °C. The
optimum temperature of `Vcmax` and `Jmax` therefore rises with growth
temperature. `Rd`, `Kc`, `Ko` and `Γ*` do not acclimate.
"""
function kattge_knorr_c3_temperature(T_growth::Real)
    tg = _growth_celsius(T_growth)
    vc = PeakedArrheniusResponse(72000.0, 200000.0, 668.39 - 1.07 * tg)
    return C3TemperatureResponses(Vcmax = vc,
                                  Jmax = PeakedArrheniusResponse(50000.0, 200000.0, 659.70 - 0.75 * tg),
                                  Tp = vc, Rd = _RD_RESPONSE)
end

"""
    kattge_knorr_jmax_ratio(T_growth) -> Jmax25 / Vcmax25

Acclimated ratio of the 25 °C electron-transport and carboxylation
capacities, ``2.59 - 0.035\\,t_g`` (Kattge & Knorr 2007; CLM4.5 eq.
8.16), with ``t_g`` in °C clamped to 11-35 °C.
"""
kattge_knorr_jmax_ratio(T_growth::Real) = 2.59 - 0.035 * _growth_celsius(T_growth)

end # module
