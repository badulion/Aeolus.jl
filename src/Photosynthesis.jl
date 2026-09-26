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
photosynthetic parameter table.

It also couples that biochemistry to the stomata (#20): the Ball-Berry
(1987) and Medlyn et al. (2011) conductance closures, the leaf boundary
layer, transpiration and the coupled ``A_n``–``g_s``–``c_i`` solution
[`leaf_gas_exchange`](@ref), with soil-water stress as an input factor
β. The vegetated land column (#21) and the leaf energy balance of the
resolved canopy (#24) delegate here.

# Units

| Quantity | Unit |
| :-- | :-- |
| rates `Vcmax`, `Jmax`, `Tp`, `Rd`, `A` | µmol m⁻² (leaf) s⁻¹ |
| CO₂ `ci`, `Kc`, `Γ*` | µmol mol⁻¹ (mole fraction) |
| O₂ `oi`, `Ko` | mmol mol⁻¹ |
| absorbed photon flux `Q` | µmol photons m⁻² s⁻¹ |
| C4 PEP-carboxylase slope `kp` | µmol m⁻² s⁻¹ per (mol mol⁻¹) |
| leaf temperature | K |
| conductances `gs`, `gb`, `gc` | mol m⁻² (leaf) s⁻¹ (to H₂O unless stated) |
| vapour pressures | Pa |

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
  * Ball, J. T., Woodrow, I. E. & Berry, J. A. (1987). A model predicting
    stomatal conductance and its contribution to the control of
    photosynthesis under different environmental conditions. In
    *Progress in Photosynthesis Research*, vol. IV, 221-224. Nijhoff.
  * Medlyn, B. E. et al. (2011). Reconciling the optimal and empirical
    approaches to modelling stomatal conductance. *Glob. Change Biol.*,
    17, 2134-2144.
  * De Kauwe, M. G. et al. (2015). A test of an optimal stomatal
    conductance scheme within the CABLE land surface model. *Geosci.
    Model Dev.*, 8, 431-452.
  * Sun et al. (2012), as cited by the CLM5 Technical Note §2.9.6 for the
    non-convergence of CLM4's ci iteration.
  * Oleson, K. W. et al. (2013). *Technical description of version 4.5 of
    the Community Land Model (CLM)*. NCAR/TN-503+STR, §8.2-8.3, Tables
    8.1-8.2.
  * Lawrence, D. M. et al. (2018). *CLM5 Documentation*, NCAR, §9.4
    (online Technical Note §2.9.3, §2.9.6, Table 2.9.1; §2.5.3
    eq. 2.5.122).
"""
module Photosynthesis

using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS,
                             saturation_vapor_pressure, latent_heat_vapor
# The bracketed Illinois root finder of the Era V contract (used by the
# Monin-Obukhov solver) also solves the coupled A-gs-ci system here.
using ..SurfaceExchange: _illinois

export AbstractTemperatureResponse, ArrheniusResponse, PeakedArrheniusResponse,
       Q10Response, temperature_factor, optimum_temperature,
       C3TemperatureResponses, bonan2011_c3_temperature,
       kattge_knorr_c3_temperature, kattge_knorr_jmax_ratio,
       PAR_PHOTONS_PER_JOULE, par_to_photon_flux, colimited_rate,
       electron_transport_rate,
       AbstractPhotosynthesisParameters, C3Parameters, C4Parameters, collatz1992_c4,
       leaf_capacities, photorespiratory_compensation_point,
       LeafAssimilation, leaf_assimilation, net_assimilation, co2_compensation_point,
       PFTPhotosynthesis, CLM45_PFT_PHOTOSYNTHESIS, vcmax25_from_leaf_nitrogen,
       pft_photosynthesis_parameters,
       AbstractStomatalModel, BallBerryStomata, MedlynStomata, stomatal_conductance,
       leaf_boundary_layer_conductance, molar_conductance, leaf_transpiration,
       STOMATAL_DIFFUSIVITY_RATIO, BOUNDARY_LAYER_DIFFUSIVITY_RATIO,
       LeafGasExchange, leaf_gas_exchange, apply_water_stress,
       CLM5_MEDLYN_G1, pft_stomatal_model

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
(for CLM's C4 `Vcmax` they reduce ``f(T_{25})`` to 0.871). A term is
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
`Tp` reuses the `Vcmax` row, as Table 8.2 does ("because Tp as
implemented here varies with Vcmax"). Note that the pre-acclimation
CLM4.5 source (tag `clm4_5_00`, `CanopyFluxesMod.F90`, commented-out
block) instead used Harley et al. (1992) for TPU: ΔHa = 53100,
ΔHd = 150650, ΔS = 490. Pass a custom [`C3TemperatureResponses`](@ref)
to use that variant.
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

# =========================================================================
# 2. Light and co-limitation
# =========================================================================

"""
    PAR_PHOTONS_PER_JOULE = 4.6

Photosynthetic photon flux per unit PAR energy, µmol photons J⁻¹, as in
CLM4.5 eqs. 8.4 and 8.7. Aeolus's single source for this conversion.
"""
const PAR_PHOTONS_PER_JOULE = 4.6

"""
    par_to_photon_flux(par) -> µmol photons m⁻² s⁻¹

Convert photosynthetically active radiation `par` (W m⁻²) to photon flux
with [`PAR_PHOTONS_PER_JOULE`](@ref).
"""
par_to_photon_flux(par::Real) = PAR_PHOTONS_PER_JOULE * par

"""
    colimited_rate(a, b, Θ) -> x

Smaller root of ``Θ x^2 - (a + b) x + a b = 0`` for non-negative `a`,
`b` and curvature ``0 ≤ Θ ≤ 1``: a smooth minimum of two limiting rates
(Collatz et al. 1991). ``Θ = 1`` gives ``\\min(a, b)``; ``Θ = 0`` the
rectangular hyperbola ``ab/(a+b)``; for ``Θ < 1`` the result is always
below the minimum. Evaluated in the cancellation-free form
``2ab / \\left[(a+b) + \\sqrt{(a+b)^2 - 4Θab}\\right]``. It is homogeneous
of degree one, ``x(ca, cb) = c\\,x(a, b)``.
"""
function colimited_rate(a::Real, b::Real, Θ::Real)
    s = a + b
    s > 0 || return zero(float(s))
    return 2 * a * b / (s + sqrt(max(s^2 - 4 * Θ * a * b, zero(s))))
end

"""
    electron_transport_rate(Jmax, Q, Φ_PSII = 0.85, Θ_PSII = 0.7) -> J

Whole-chain electron transport rate (µmol e⁻ m⁻² s⁻¹) at absorbed photon
flux `Q` (µmol m⁻² s⁻¹): the smaller root of
``Θ J^2 - (I + J_{max}) J + I J_{max} = 0`` with
``I = \\tfrac12 Φ_{PSII} Q`` the light used by photosystem II (one photon
per photosystem per electron) — CLM4.5 eqs. 8.6-8.7. Zero in the dark,
initial slope ``\\tfrac12 Φ_{PSII}``, saturating at `Jmax`.
"""
electron_transport_rate(Jmax::Real, Q::Real, Φ::Real = 0.85, Θ::Real = 0.7) =
    colimited_rate(0.5 * Φ * max(Q, zero(Q)), Jmax, Θ)

# =========================================================================
# 3. Leaf parameter sets
# =========================================================================

"""
    AbstractPhotosynthesisParameters

A leaf's photosynthetic parameters: [`C3Parameters`](@ref) or
[`C4Parameters`](@ref). Both are evaluated by [`leaf_assimilation`](@ref).
"""
abstract type AbstractPhotosynthesisParameters end

"""
    C3Parameters(Vcmax25; T_growth = nothing, Jmax25, Tp25, Rd25, kwargs...)

Farquhar, von Caemmerer & Berry (1980) parameters for a C3 leaf, with
the CLM4.5/CLM5 defaults (Oleson et al. 2013 §8.2):

| field | default | meaning |
| :-- | :-- | :-- |
| `Vcmax25` | – | Rubisco capacity at 25 °C, µmol m⁻² s⁻¹ |
| `Jmax25` | `1.97 Vcmax25`, or the Kattge-Knorr ratio | electron-transport capacity |
| `Tp25` | `0.167 Vcmax25` | triose-phosphate utilisation (so `Ap = 3Tp ≈ Vcmax/2`) |
| `Rd25` | `0.015 Vcmax25` | day (mitochondrial) respiration |
| `Kc25` | 404.9 µmol mol⁻¹ | Michaelis constant for CO₂ (Bernacchi 2001) |
| `Ko25` | 278.4 mmol mol⁻¹ | Michaelis constant for O₂ (Bernacchi 2001) |
| `Γstar25` | 42.75 µmol mol⁻¹ | photorespiratory compensation point at `o_ref` |
| `o_ref` | 209.0 mmol mol⁻¹ | O₂ at which `Γstar25` applies |
| `Φ_PSII`, `Θ_PSII` | 0.85, 0.7 | PSII quantum yield and curvature |
| `Θ_cj`, `Θ_ip` | 0.98, 0.95 | co-limitation curvatures |

With `T_growth` (K, e.g. the 10-day mean air temperature) the temperature
responses acclimate after Kattge & Knorr (2007)
([`kattge_knorr_c3_temperature`](@ref)) and `Jmax25` defaults to
[`kattge_knorr_jmax_ratio`](@ref)` × Vcmax25`; without it they are the
non-acclimating Bonan et al. (2011) set
([`bonan2011_c3_temperature`](@ref)). A `temperature` keyword overrides
either.
"""
struct C3Parameters{R<:C3TemperatureResponses} <: AbstractPhotosynthesisParameters
    Vcmax25::Float64
    Jmax25::Float64
    Tp25::Float64
    Rd25::Float64
    Kc25::Float64
    Ko25::Float64
    Γstar25::Float64
    o_ref::Float64
    Φ_PSII::Float64
    Θ_PSII::Float64
    Θ_cj::Float64
    Θ_ip::Float64
    temperature::R
end

function C3Parameters(Vcmax25::Real; T_growth = nothing,
                      temperature = T_growth === nothing ? bonan2011_c3_temperature() :
                                                           kattge_knorr_c3_temperature(T_growth),
                      Jmax25 = (T_growth === nothing ? 1.97 : kattge_knorr_jmax_ratio(T_growth)) * Vcmax25,
                      Tp25 = 0.167 * Vcmax25, Rd25 = 0.015 * Vcmax25,
                      Kc25 = 404.9, Ko25 = 278.4, Γstar25 = 42.75, o_ref = 209.0,
                      Φ_PSII = 0.85, Θ_PSII = 0.7, Θ_cj = 0.98, Θ_ip = 0.95)
    return C3Parameters(Float64(Vcmax25), Float64(Jmax25), Float64(Tp25), Float64(Rd25),
                        Float64(Kc25), Float64(Ko25), Float64(Γstar25), Float64(o_ref),
                        Float64(Φ_PSII), Float64(Θ_PSII), Float64(Θ_cj), Float64(Θ_ip),
                        temperature)
end

"""
    C4Parameters(Vcmax25; kp25 = 20000 Vcmax25, Rd25 = 0.025 Vcmax25, α = 0.05,
                 Θ_cj = 0.80, Θ_ip = 0.95, Vcmax_T, kp_T, Rd_T)

Collatz, Ribas-Carbo & Berry (1992) parameters for a C4 leaf with the
CLM4.5/CLM5 defaults: quantum efficiency `α` (mol CO₂ mol⁻¹ photons),
PEP-carboxylase initial slope `kp25` (µmol m⁻² s⁻¹ per mol mol⁻¹ of
intercellular CO₂) and the Q10 temperature responses of CLM4.5 eqs.
8.12-8.14 (`Q10 = 2`; `Vcmax` inhibited above 313.15 K with slope
0.3 K⁻¹ and below 288.15 K with 0.2 K⁻¹; `Rd` above 328.15 K with
1.3 K⁻¹). See [`collatz1992_c4`](@ref) for the original paper's values.
"""
struct C4Parameters{V<:AbstractTemperatureResponse,K<:AbstractTemperatureResponse,
                    D<:AbstractTemperatureResponse} <: AbstractPhotosynthesisParameters
    Vcmax25::Float64
    kp25::Float64
    Rd25::Float64
    α::Float64
    Θ_cj::Float64
    Θ_ip::Float64
    Vcmax_T::V
    kp_T::K
    Rd_T::D
end

function C4Parameters(Vcmax25::Real; kp25 = 20000 * Vcmax25, Rd25 = 0.025 * Vcmax25,
                      α = 0.05, Θ_cj = 0.80, Θ_ip = 0.95,
                      Vcmax_T = Q10Response(Q10 = 2.0, s_high = 0.3, T_high = 313.15,
                                            s_low = 0.2, T_low = 288.15),
                      kp_T = Q10Response(Q10 = 2.0),
                      Rd_T = Q10Response(Q10 = 2.0, s_high = 1.3, T_high = 328.15))
    return C4Parameters(Float64(Vcmax25), Float64(kp25), Float64(Rd25), Float64(α),
                        Float64(Θ_cj), Float64(Θ_ip), Vcmax_T, kp_T, Rd_T)
end

"""
    collatz1992_c4() -> C4Parameters

The C4 leaf of Collatz et al. (1992) at 25 °C: `Vmax = 39`,
`Rd = 0.8` µmol m⁻² s⁻¹, `α = 0.04`, `k = 0.7` mol m⁻² s⁻¹
(= 7 × 10⁵ µmol m⁻² s⁻¹ per mol mol⁻¹), curvatures `θ = 0.83`,
`β = 0.93` — the values carried as defaults of BioCro's `c4photo`,
which implements that paper. Temperature responses are the CLM ones of
[`C4Parameters`](@ref); the paper's own inhibition terms differ slightly.
"""
collatz1992_c4() = C4Parameters(39.0; kp25 = 0.7e6, Rd25 = 0.8, α = 0.04,
                                Θ_cj = 0.83, Θ_ip = 0.93)

# =========================================================================
# 4. Temperature-adjusted capacities
# =========================================================================

"""
    leaf_capacities(p::C3Parameters, T; oi = p.o_ref) -> NamedTuple
    leaf_capacities(p::C4Parameters, T) -> NamedTuple

The biochemical capacities at leaf temperature `T` (K). For C3:
`(Vcmax, Jmax, Tp, Rd, Kc, Ko, Γstar)`; `Γstar` is scaled linearly with
the O₂ mole fraction `oi` (mmol mol⁻¹), since
``Γ^* = \\tfrac12\\, O / S_{c/o}`` with the Rubisco specificity
``S_{c/o}`` independent of O₂ (Farquhar et al. 1980, eq. 38). For C4:
`(Vcmax, kp, Rd)`.
"""
function leaf_capacities(p::C3Parameters, T::Real; oi::Real = p.o_ref)
    t = p.temperature
    return (Vcmax = p.Vcmax25 * temperature_factor(t.Vcmax, T),
            Jmax  = p.Jmax25 * temperature_factor(t.Jmax, T),
            Tp    = p.Tp25 * temperature_factor(t.Tp, T),
            Rd    = p.Rd25 * temperature_factor(t.Rd, T),
            Kc    = p.Kc25 * temperature_factor(t.Kc, T),
            Ko    = p.Ko25 * temperature_factor(t.Ko, T),
            Γstar = p.Γstar25 * temperature_factor(t.Γstar, T) * oi / p.o_ref)
end

leaf_capacities(p::C4Parameters, T::Real) =
    (Vcmax = p.Vcmax25 * temperature_factor(p.Vcmax_T, T),
     kp    = p.kp25 * temperature_factor(p.kp_T, T),
     Rd    = p.Rd25 * temperature_factor(p.Rd_T, T))

"""
    photorespiratory_compensation_point(p::C3Parameters, T; oi = p.o_ref) -> Γ* (µmol mol⁻¹)

The CO₂ mole fraction at which Rubisco carboxylation exactly balances
photorespiratory CO₂ release, so that gross assimilation is zero.
"""
photorespiratory_compensation_point(p::C3Parameters, T::Real; oi::Real = p.o_ref) =
    leaf_capacities(p, T; oi = oi).Γstar

# =========================================================================
# 5. Assimilation
# =========================================================================

"""
    LeafAssimilation

Result of [`leaf_assimilation`](@ref), all in µmol CO₂ m⁻² (leaf) s⁻¹:

  * `Ac`, `Aj`, `Ap` — the Rubisco-, light (RuBP regeneration)- and
    export (C3: triose-phosphate utilisation; C4: PEP-carboxylase)-limited
    gross rates;
  * `A` — gross assimilation after co-limitation;
  * `Rd` — day respiration;
  * `An = A − Rd` — net assimilation;
  * `limitation` — `:rubisco`, `:light` or `:export`, the smallest of the
    three limiting rates.
"""
struct LeafAssimilation
    Ac::Float64
    Aj::Float64
    Ap::Float64
    A::Float64
    Rd::Float64
    An::Float64
    limitation::Symbol
end

_limitation(Ac, Aj, Ap) = Ac <= Aj ? (Ac <= Ap ? :rubisco : :export) :
                                     (Aj <= Ap ? :light : :export)

"""
    leaf_assimilation(p, ci, Q, T; oi = 209.0, colimitation = true) -> LeafAssimilation

Leaf photosynthesis at intercellular CO₂ `ci` (µmol mol⁻¹), absorbed
photon flux `Q` (µmol m⁻² s⁻¹; see [`par_to_photon_flux`](@ref)) and leaf
temperature `T` (K).

**C3** (Farquhar et al. 1980; CLM4.5 eqs. 8.3-8.8):

```math
A_c = V_{cmax}\\frac{c_i - Γ^*}{c_i + K_c(1 + o_i/K_o)},\\quad
A_j = J\\frac{c_i - Γ^*}{4 c_i + 8 Γ^*},\\quad
A_p = 3 T_p ,
```

with `J` from [`electron_transport_rate`](@ref). **C4** (Collatz et al.
1992): ``A_c = V_{cmax}``, ``A_j = α Q``, ``A_p = k_p c_i``.

With `colimitation = true` the gross rate is
``A = \\mathrm{colim}(\\mathrm{colim}(A_c, A_j; Θ_{cj}), A_p; Θ_{ip})``
([`colimited_rate`](@ref)) as in CLM; with `false`, the plain Farquhar
minimum. Net assimilation is ``A_n = A - R_d``.

Below the compensation point (``c_i < Γ^*``) the C3 gross rate is
negative — photorespiration releases more CO₂ than Rubisco fixes. It is
evaluated as ``(c_i - Γ^*)\\,\\mathrm{colim}(V_{cmax}/(c_i + K_m), J/(4c_i + 8Γ^*))``,
which is identical to CLM's form for ``c_i ≥ Γ^*`` (the co-limitation is
homogeneous of degree one) and continues it correctly below, where CLM
clips; TPU cannot limit a negative flux.
"""
function leaf_assimilation(p::C3Parameters, ci::Real, Q::Real, T::Real;
                           oi::Real = p.o_ref, colimitation::Bool = true)
    c = leaf_capacities(p, T; oi = oi)
    ci = max(float(ci), 0.0)
    Km = c.Kc * (1 + oi / c.Ko)
    J = electron_transport_rate(c.Jmax, Q, p.Φ_PSII, p.Θ_PSII)
    wc = c.Vcmax / (ci + Km)           # carboxylation per unit (ci − Γ*)
    wj = J / (4ci + 8c.Γstar)
    Ac = wc * (ci - c.Γstar)
    Aj = wj * (ci - c.Γstar)
    Ap = 3 * c.Tp
    if colimitation
        Ai = (ci - c.Γstar) * colimited_rate(wc, wj, p.Θ_cj)
        A = Ai > 0 ? colimited_rate(Ai, Ap, p.Θ_ip) : Ai
    else
        Ai = (ci - c.Γstar) * min(wc, wj)
        A = Ai > 0 ? min(Ai, Ap) : Ai
    end
    lim = ci >= c.Γstar ? _limitation(Ac, Aj, Ap) : (wc <= wj ? :rubisco : :light)
    return LeafAssimilation(Ac, Aj, Ap, A, c.Rd, A - c.Rd, lim)
end

function leaf_assimilation(p::C4Parameters, ci::Real, Q::Real, T::Real;
                           oi::Real = 209.0, colimitation::Bool = true)
    c = leaf_capacities(p, T)
    Ac = c.Vcmax
    Aj = p.α * max(float(Q), 0.0)
    Ap = c.kp * max(float(ci), 0.0) * 1e-6       # ci: µmol/mol → mol/mol
    if colimitation
        A = colimited_rate(colimited_rate(Ac, Aj, p.Θ_cj), Ap, p.Θ_ip)
    else
        A = min(Ac, Aj, Ap)
    end
    return LeafAssimilation(Ac, Aj, Ap, A, c.Rd, A - c.Rd, _limitation(Ac, Aj, Ap))
end

"""
    net_assimilation(p, ci, Q, T; kwargs...) -> An (µmol m⁻² s⁻¹)

Shorthand for `leaf_assimilation(p, ci, Q, T; kwargs...).An`.
"""
net_assimilation(p::AbstractPhotosynthesisParameters, ci, Q, T; kwargs...) =
    leaf_assimilation(p, ci, Q, T; kwargs...).An

"""
    co2_compensation_point(p, Q, T; ci_max = 2000.0, kwargs...) -> Γ (µmol mol⁻¹)

The intercellular CO₂ at which net assimilation vanishes,
``A_n(Γ) = 0``, found by bisection on ``[0, c_{i,max}]`` (``A_n`` is
monotone in ``c_i``). For C3 leaves ``Γ > Γ^*`` because day respiration
must also be repaid; for C4 leaves it is a few µmol mol⁻¹. Returns `NaN`
if the leaf cannot reach positive net uptake below `ci_max` (e.g. in
darkness).
"""
function co2_compensation_point(p::AbstractPhotosynthesisParameters, Q::Real, T::Real;
                                ci_max::Real = 2000.0, kwargs...)
    f(ci) = net_assimilation(p, ci, Q, T; kwargs...)
    lo, hi = 0.0, Float64(ci_max)
    f(hi) > 0 || return NaN
    f(lo) >= 0 && return lo
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) > 0 ? (hi = mid) : (lo = mid)
        hi - lo < 1e-10 * max(1.0, hi) && break
    end
    return 0.5 * (lo + hi)
end

# =========================================================================
# 6. Plant functional types (CLM4.5 Table 8.1)
# =========================================================================

"""
    PFTPhotosynthesis

Photosynthetic traits of one plant functional type, transcribed from
CLM4.5 Technical Note Table 8.1 (Oleson et al. 2013):

  * `name` — CLM's PFT label;
  * `pathway` — `:C3` or `:C4`;
  * `Vcmax25` — canopy-top Rubisco capacity at 25 °C, µmol m⁻² s⁻¹;
  * `CN_L` — leaf carbon : nitrogen ratio, g C g⁻¹ N;
  * `FLNR` — fraction of leaf N in Rubisco, g N g⁻¹ N;
  * `SLA0` — canopy-top specific leaf area, m² g⁻¹ C;
  * `α` — C4 quantum efficiency, mol CO₂ mol⁻¹ photons (`NaN` for C3).

The tabulated `Vcmax25` is derived from the other three columns by
[`vcmax25_from_leaf_nitrogen`](@ref) (CLM4.5 eq. 8.17). The stomatal
slope `m` of that table is carried by [`pft_stomatal_model`](@ref); the
soil-water (`ψo`, `ψc`) columns belong to the land column (#21).
"""
struct PFTPhotosynthesis
    name::String
    pathway::Symbol
    Vcmax25::Float64
    CN_L::Float64
    FLNR::Float64
    SLA0::Float64
    α::Float64
end

"""
    CLM45_PFT_PHOTOSYNTHESIS :: Dict{Symbol, PFTPhotosynthesis}

CLM4.5 Table 8.1, natural vegetation and the unmanaged/rainfed crop rows
(irrigated rows are identical). Keys:

| key | CLM PFT | pathway | Vcmax25 |
| :-- | :-- | :-- | --: |
| `:net_temperate` | NET Temperate | C3 | 62.5 |
| `:net_boreal` | NET Boreal | C3 | 62.6 |
| `:ndt_boreal` | NDT Boreal | C3 | 39.1 |
| `:bet_tropical` | BET Tropical | C3 | 55.0 |
| `:bet_temperate` | BET temperate | C3 | 61.5 |
| `:bdt_tropical` | BDT tropical | C3 | 41.0 |
| `:bdt_temperate` | BDT temperate | C3 | 57.7 |
| `:bdt_boreal` | BDT boreal | C3 | 57.7 |
| `:bes_temperate` | BES temperate | C3 | 61.7 |
| `:bds_temperate` | BDS temperate | C3 | 54.0 |
| `:bds_boreal` | BDS boreal | C3 | 54.0 |
| `:c3_arctic_grass` | C3 arctic grass | C3 | 78.2 |
| `:c3_grass` | C3 grass | C3 | 78.2 |
| `:c4_grass` | C4 grass | C4 | 51.6 |
| `:crop` | Crop R | C3 | 100.7 |
| `:corn` | Corn R | C4 | 100.7 |
| `:temperate_cereal` | Temp Cereal R | C3 | 100.7 |
| `:winter_cereal` | Winter Cereal R | C3 | 100.7 |
| `:soybean` | Soybean R | C3 | 100.7 |

(N = needleleaf, B = broadleaf, E = evergreen, D = deciduous, T = tree,
S = shrub.) Every tabulated `Vcmax25` is reproduced from the same row's
`CN_L`, `SLA0` and `FLNR` by [`vcmax25_from_leaf_nitrogen`](@ref) to
within 0.1 % (the table's rounding), which is how the transcription is
checked.
"""
const CLM45_PFT_PHOTOSYNTHESIS = Dict{Symbol,PFTPhotosynthesis}(
    :net_temperate    => PFTPhotosynthesis("NET Temperate",   :C3,  62.5, 35, 0.0509, 0.010, NaN),
    :net_boreal       => PFTPhotosynthesis("NET Boreal",      :C3,  62.6, 40, 0.0466, 0.008, NaN),
    :ndt_boreal       => PFTPhotosynthesis("NDT Boreal",      :C3,  39.1, 25, 0.0546, 0.024, NaN),
    :bet_tropical     => PFTPhotosynthesis("BET Tropical",    :C3,  55.0, 30, 0.0461, 0.012, NaN),
    :bet_temperate    => PFTPhotosynthesis("BET temperate",   :C3,  61.5, 30, 0.0515, 0.012, NaN),
    :bdt_tropical     => PFTPhotosynthesis("BDT tropical",    :C3,  41.0, 25, 0.0716, 0.030, NaN),
    :bdt_temperate    => PFTPhotosynthesis("BDT temperate",   :C3,  57.7, 25, 0.1007, 0.030, NaN),
    :bdt_boreal       => PFTPhotosynthesis("BDT boreal",      :C3,  57.7, 25, 0.1007, 0.030, NaN),
    :bes_temperate    => PFTPhotosynthesis("BES temperate",   :C3,  61.7, 30, 0.0517, 0.012, NaN),
    :bds_temperate    => PFTPhotosynthesis("BDS temperate",   :C3,  54.0, 25, 0.0943, 0.030, NaN),
    :bds_boreal       => PFTPhotosynthesis("BDS boreal",      :C3,  54.0, 25, 0.0943, 0.030, NaN),
    :c3_arctic_grass  => PFTPhotosynthesis("C3 arctic grass", :C3,  78.2, 25, 0.1365, 0.030, NaN),
    :c3_grass         => PFTPhotosynthesis("C3 grass",        :C3,  78.2, 25, 0.1365, 0.030, NaN),
    :c4_grass         => PFTPhotosynthesis("C4 grass",        :C4,  51.6, 25, 0.0900, 0.030, 0.05),
    :crop             => PFTPhotosynthesis("Crop R",          :C3, 100.7, 25, 0.1758, 0.030, NaN),
    :corn             => PFTPhotosynthesis("Corn R",          :C4, 100.7, 25, 0.2930, 0.050, 0.05),
    :temperate_cereal => PFTPhotosynthesis("Temp Cereal R",   :C3, 100.7, 25, 0.4102, 0.070, NaN),
    :winter_cereal    => PFTPhotosynthesis("Winter Cereal R", :C3, 100.7, 25, 0.4102, 0.070, NaN),
    :soybean          => PFTPhotosynthesis("Soybean R",       :C3, 100.7, 25, 0.4102, 0.070, NaN),
)

"""
    vcmax25_from_leaf_nitrogen(CN_L, SLA0, FLNR; FNR = 7.16, aR25 = 60.0) -> µmol m⁻² s⁻¹

Rubisco capacity from leaf nitrogen (Thornton & Zimmermann 2007; CLM4.5
eqs. 8.17-8.18):
``V_{cmax25} = N_a\\,F_{LNR}\\,F_{NR}\\,a_{R25}`` with area-based leaf N
``N_a = 1/(CN_L\\,SLA_0)`` (g N m⁻²), `FNR` the mass ratio of Rubisco to
its nitrogen and `aR25` the Rubisco specific activity
(µmol CO₂ g⁻¹ Rubisco s⁻¹).
"""
vcmax25_from_leaf_nitrogen(CN_L::Real, SLA0::Real, FLNR::Real; FNR::Real = 7.16,
                           aR25::Real = 60.0) = FLNR * FNR * aR25 / (CN_L * SLA0)

"""
    pft_photosynthesis_parameters(pft::Symbol; kwargs...) -> C3Parameters | C4Parameters

Leaf parameters for a CLM plant functional type (a key of
[`CLM45_PFT_PHOTOSYNTHESIS`](@ref)) with its canopy-top `Vcmax25` and,
for C4 types, its quantum efficiency. Keyword arguments are passed to
[`C3Parameters`](@ref) / [`C4Parameters`](@ref) (e.g. `T_growth`).
"""
function pft_photosynthesis_parameters(pft::Symbol; kwargs...)
    haskey(CLM45_PFT_PHOTOSYNTHESIS, pft) ||
        throw(ArgumentError("unknown PFT :$pft; see keys(CLM45_PFT_PHOTOSYNTHESIS)"))
    t = CLM45_PFT_PHOTOSYNTHESIS[pft]
    return t.pathway === :C4 ? C4Parameters(t.Vcmax25; α = t.α, kwargs...) :
                               C3Parameters(t.Vcmax25; kwargs...)
end

# =========================================================================
# 7. Stomatal conductance (Phase 47, #20)
# =========================================================================
#
# Conductances are molar, mol m⁻² (leaf) s⁻¹, and — unless stated — to
# WATER VAPOUR. CO₂ conductances follow from the diffusivity ratios
# 1.6 (stomatal pore) and 1.4 (leaf boundary layer), as in CLM4.5 eq. 8.1
# / CLM5 eq. 2.9.19. With CO₂ in µmol mol⁻¹ and conductance in
# mol m⁻² s⁻¹ a flux g Δc is in µmol m⁻² s⁻¹: no pressure appears.

"Ratio of the diffusivities of H₂O and CO₂ through a stomatal pore (CLM4.5 eq. 8.1)."
const STOMATAL_DIFFUSIVITY_RATIO = 1.6
"Ratio of the H₂O and CO₂ conductances of the leaf boundary layer, ``1.6^{2/3} ≈ 1.4`` (CLM4.5 eq. 8.1)."
const BOUNDARY_LAYER_DIFFUSIVITY_RATIO = 1.4

# Absolute floor on stomatal conductance, mol m⁻² s⁻¹: CLM's
# `bbb = max(bbbopt*btran, 1 µmol m⁻² s⁻¹)` (PhotosynthesisMod.F90).
const _GS_FLOOR = 1.0e-6

"""
    AbstractStomatalModel

A stomatal-conductance closure ``g_s = f(A_n, c_s, \\text{leaf-surface
humidity})``. Implementers extend [`stomatal_conductance`](@ref); the
coupled solver [`leaf_gas_exchange`](@ref) works with any of them.
Concrete models: [`BallBerryStomata`](@ref), [`MedlynStomata`](@ref).
"""
abstract type AbstractStomatalModel end

"""
    BallBerryStomata(; g1 = 9.0, g0 = 0.01)

Ball, Woodrow & Berry (1987) stomatal conductance to water vapour, in the
form of Collatz et al. (1991) and CLM4.5 (Oleson et al. 2013, eq. 8.1):

```math
g_s = g_0 + g_1 \\frac{A_n\\, h_s}{c_s}\\qquad (A_n > 0),\\qquad g_s = g_0 \\quad (A_n ≤ 0),
```

with ``h_s = e_s / e_{sat}(T_\\ell)`` the relative humidity *at the leaf
surface*, ``c_s`` the leaf-surface CO₂ (µmol mol⁻¹), ``A_n`` in
µmol m⁻² s⁻¹. `g1` (CLM's `m`) is dimensionless; `g0` (CLM's `b`) is in
mol m⁻² s⁻¹. Defaults are CLM's C3 values (`m = 9`, `b = 10000`
µmol m⁻² s⁻¹); C4 uses `g1 = 4`, `g0 = 0.04` — see
[`pft_stomatal_model`](@ref). As in CLM the air vapour pressure entering
``h_s`` is capped at saturation.
"""
Base.@kwdef struct BallBerryStomata <: AbstractStomatalModel
    g1::Float64 = 9.0
    g0::Float64 = 0.01
end

"""
    MedlynStomata(; g1 = 4.45, g0 = 1e-4, D_min = 50.0)

Medlyn et al. (2011) optimal stomatal conductance to water vapour, as in
CLM5 (Lawrence et al. 2018, eq. 2.9.1):

```math
g_s = g_0 + 1.6\\left(1 + \\frac{g_1}{\\sqrt{D_s}}\\right)\\frac{A_n}{c_s}
\\qquad (A_n > 0),\\qquad g_s = g_0 \\quad (A_n ≤ 0),
```

with ``D_s`` the vapour-pressure deficit at the leaf surface in **kPa**,
`g1` in kPa^½ and `g0` in mol m⁻² s⁻¹ (CLM5: 100 µmol m⁻² s⁻¹).
The leaf-surface deficit is bounded below by `D_min` (Pa), so the
closure stays finite in saturated air. CLM5 applies its 50 Pa bound to
the *air* deficit ``e_{sat}(T_\\ell) - e_a`` instead; that leaves the
leaf-surface deficit unbounded below for a weakly coupled leaf, where
opening stomata humidifies the leaf surface and opens them further —
at 5 °C, 98 % RH and ``g_b = 0.1`` mol m⁻² s⁻¹ it gives
``g_s ≈ 10^3`` mol m⁻² s⁻¹. Here the deficit entering the closure is
``\\max(D_s, D_{min})``, which caps ``g_s`` at
``g_0 + 1.6(1 + g_1/\\sqrt{D_{min}})A_n/c_s``. Whenever
``D_s ≥ D_{min}`` the two agree exactly. For ``g_0 = 0`` and a perfectly coupled leaf it gives the optimal
ratio ``c_i/c_a = g_1/(g_1 + \\sqrt{D})`` independent of ``A_n``. The
default `g1` is CLM5's broadleaf-deciduous-tree value; per-PFT values
are in [`CLM5_MEDLYN_G1`](@ref).
"""
Base.@kwdef struct MedlynStomata <: AbstractStomatalModel
    g1::Float64    = 4.45
    g0::Float64    = 1.0e-4
    D_min::Float64 = 50.0
end

"""
    stomatal_conductance(model, An, cs, e_air, e_sat_leaf, gb) -> gs (mol H₂O m⁻² s⁻¹)

Stomatal conductance at net assimilation `An` (µmol m⁻² s⁻¹) and
leaf-surface CO₂ `cs` (µmol mol⁻¹), given the vapour pressure of the air
`e_air` and the saturation vapour pressure at leaf temperature
`e_sat_leaf` (Pa) and the leaf boundary-layer conductance to water
vapour `gb` (mol m⁻² s⁻¹; `Inf` for a perfectly coupled leaf).

The leaf-surface humidity is not an input: with no storage at the
surface, the vapour flux through the boundary layer equals the one
through the stomata,
``e_s = (g_b e_a + g_s e_{sat}) / (g_b + g_s)`` (CLM5 eq. 2.9.23), so
``h_s`` and ``D_s`` depend on ``g_s`` itself. Both closures are solved
exactly for that dependence:

  * Ball-Berry: a quadratic in ``g_s`` — the one CLM4.5 solves
    (PhotosynthesisMod `ci_func`) — whose positive root is taken in a
    cancellation-free form;
  * Medlyn: since ``D_s = g_b D_a / (g_b + g_s)``, the closure is a
    quadratic in ``x = \\sqrt{g_b + g_s}`` with exactly one positive
    root. Squaring it gives CLM5's quadratic (eqs. 2.9.24-2.9.26), whose
    larger root is the same ``g_s``.
"""
function stomatal_conductance end

function stomatal_conductance(m::BallBerryStomata, An::Real, cs::Real, e_air::Real,
                              e_sat_leaf::Real, gb::Real)
    g0 = m.g0
    An > 0 || return g0
    h_air = min(e_air, e_sat_leaf) / e_sat_leaf        # CLM: ceair = min(eair, esat)
    a = m.g1 * An / cs
    isinf(gb) && return g0 + a * h_air
    # gs² + (gb − g0 − a) gs − gb (g0 + a h_air) = 0, positive root.
    B = gb - g0 - a
    C = gb * (g0 + a * h_air)
    disc = sqrt(B^2 + 4C)
    return B > 0 ? 2C / (B + disc) : (disc - B) / 2
end

function stomatal_conductance(m::MedlynStomata, An::Real, cs::Real, e_air::Real,
                              e_sat_leaf::Real, gb::Real)
    g0 = m.g0
    An > 0 || return g0
    c1 = STOMATAL_DIFFUSIVITY_RATIO * An / cs
    Dmin = m.D_min * 1e-3                                  # kPa
    gs_bound = g0 + c1 * (1 + m.g1 / sqrt(Dmin))           # closure at Ds = D_min
    Da = (e_sat_leaf - min(e_air, e_sat_leaf)) * 1e-3      # air deficit, kPa
    Da <= Dmin && return gs_bound
    isinf(gb) && return g0 + c1 * (1 + m.g1 / sqrt(Da))
    # gs = g0 + c1 + c2 x with x = √(gb + gs):  x² − c2 x − (gb + g0 + c1) = 0.
    c2 = m.g1 * c1 / sqrt(gb * Da)
    x = (c2 + sqrt(c2^2 + 4 * (gb + g0 + c1))) / 2
    gs = g0 + c1 + c2 * x
    # Leaf-surface deficit Ds = gb Da / x². If it is below D_min the
    # closure evaluated at max(Ds, D_min) has the single root gs_bound.
    return gb * Da / x^2 >= Dmin ? gs : gs_bound
end

"""
    molar_conductance(g, T, p; c = DEFAULT_CONSTANTS) -> mol m⁻² s⁻¹

Convert a conductance `g` in m s⁻¹ to molar units with the ideal-gas
molar density of air, ``g_{mol} = g\\,p/(R T)`` (`T` in K, `p` in Pa;
CLM5 §2.9.3). The gas constant is `MoistThermodynamics`'.
"""
molar_conductance(g::Real, T::Real, p::Real; c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    g * p / (c.gas_constant * T)

"""
    leaf_boundary_layer_conductance(U, T_air, p_air; d_leaf = 0.04, Cv = 0.01,
                                    c = DEFAULT_CONSTANTS) -> mol m⁻² s⁻¹

Leaf boundary-layer conductance to heat and water vapour for wind speed
`U` (m s⁻¹) past a leaf of characteristic dimension `d_leaf` (m), after
CLM5 eq. 2.5.122 (Oleson et al. 2013 eq. 5.122):

```math
g_b = C_v \\sqrt{U / d_{leaf}}\\ \\ [\\mathrm{m\\,s^{-1}}],
\\qquad C_v = 0.01\\ \\mathrm{m\\,s^{-1/2}},
```

converted to molar units by [`molar_conductance`](@ref) at the air
temperature and pressure. `d_leaf = 0.04` m is CLM5 Table 2.5.1 for
every PFT. The ``\\sqrt{U/d}`` scaling is laminar forced convection over
a flat plate; ``C_v`` absorbs the two leaf sides and outdoor turbulence.
The CO₂ conductance is this divided by
[`BOUNDARY_LAYER_DIFFUSIVITY_RATIO`](@ref).
"""
function leaf_boundary_layer_conductance(U::Real, T_air::Real, p_air::Real;
                                         d_leaf::Real = 0.04, Cv::Real = 0.01,
                                         c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    return molar_conductance(Cv * sqrt(max(U, zero(U)) / d_leaf), T_air, p_air; c = c)
end

"""
    leaf_transpiration(gs, gb, T_leaf, e_air, p_air; c = DEFAULT_CONSTANTS) -> mol H₂O m⁻² s⁻¹

Leaf transpiration through stomata and boundary layer in series,

```math
E = \\frac{e_{sat}(T_\\ell) - e_a}{p}\\,\\frac{g_s g_b}{g_s + g_b}
```

(CLM5 eq. 2.9.20; the leaf interior is saturated at leaf temperature).
`e_sat` is `MoistThermodynamics.saturation_vapor_pressure`. Multiply by
`c.molmass_water` for kg m⁻² s⁻¹. The small mass-flow ("ternary")
correction of von Caemmerer & Farquhar (1981) is neglected, as in CLM.
Negative `E` means the air is supersaturated with respect to the leaf;
dew on the leaf surface is the business of the land column (#21).
"""
function leaf_transpiration(gs::Real, gb::Real, T_leaf::Real, e_air::Real, p_air::Real;
                            c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    gw = isinf(gb) ? float(gs) : gs * gb / (gs + gb)
    return gw * (saturation_vapor_pressure(T_leaf, c) - e_air) / p_air
end

# =========================================================================
# 8. Water stress hook and the coupled A–gs–ci solution
# =========================================================================

const _STRESS_TARGETS = (:vcmax, :rd, :g0, :g1)

"""
    apply_water_stress(p::AbstractPhotosynthesisParameters, β; targets = (:vcmax, :rd, :g0))
    apply_water_stress(m::AbstractStomatalModel, β; targets = (:vcmax, :rd, :g0))

The soil-water-stress hook. `β ∈ [0, 1]` is the plant wetness factor
(CLM's `btran`); it is an *input* here — the land column (#21) computes
it from soil matric potential. `targets` says what it multiplies:

  * `:vcmax` — the Rubisco capacity `Vcmax25` (CLM4.5 eq. 8.10;
    `vcmax_z = vcmax_z * btran` in CLM5's non-hydraulic path);
  * `:rd` — day respiration `Rd25` (CLM, `lmr_z = lmr_z * btran`);
  * `:g0` — the stomatal intercept (CLM's Ball-Berry path,
    `bbb = max(bbbopt*btran, 1 µmol)`);
  * `:g1` — the stomatal slope (the alternative of e.g. Zhou et al. 2013
    and De Kauwe et al. 2015, not CLM's default).

For [`BallBerryStomata`](@ref) the default reproduces CLM4.5 and CLM5's
non-hydraulic Ball-Berry path. CLM5's Medlyn closure does *not* stress
its intercept: `ci_func_PHS` in PhotosynthesisMod.F90 (release-clm5.0)
uses the unstressed `medlynintercept`, and water stress enters through
plant hydraulics instead. Scaling a [`MedlynStomata`](@ref) intercept by
β is therefore an Aeolus choice, carried over from Ball-Berry. Its effect
is small, because the CLM5 intercept is only 10⁻⁴ mol m⁻² s⁻¹. Pass
`targets = (:vcmax, :rd)` to leave the Medlyn intercept unstressed.
A parameter set is returned
unchanged for targets that do not apply to it. The stomatal intercept is
always bounded below by 1 µmol m⁻² s⁻¹ (CLM's floor), which keeps the
coupled system well posed in the dark even for `g0 = 0`.
"""
function apply_water_stress end

function _check_stress(β, targets)
    0 <= β <= 1 || throw(DomainError(β, "water-stress factor β must lie in [0, 1]"))
    for t in targets
        t in _STRESS_TARGETS ||
            throw(ArgumentError("unknown water-stress target :$t; use a subset of $_STRESS_TARGETS"))
    end
    return nothing
end

function apply_water_stress(p::C3Parameters, β::Real; targets = (:vcmax, :rd, :g0))
    _check_stress(β, targets)
    fv = :vcmax in targets ? β : 1.0
    fr = :rd in targets ? β : 1.0
    return C3Parameters(p.Vcmax25 * fv, p.Jmax25, p.Tp25, p.Rd25 * fr, p.Kc25, p.Ko25,
                        p.Γstar25, p.o_ref, p.Φ_PSII, p.Θ_PSII, p.Θ_cj, p.Θ_ip,
                        p.temperature)
end

function apply_water_stress(p::C4Parameters, β::Real; targets = (:vcmax, :rd, :g0))
    _check_stress(β, targets)
    fv = :vcmax in targets ? β : 1.0
    fr = :rd in targets ? β : 1.0
    return C4Parameters(p.Vcmax25 * fv, p.kp25, p.Rd25 * fr, p.α, p.Θ_cj, p.Θ_ip,
                        p.Vcmax_T, p.kp_T, p.Rd_T)
end

function apply_water_stress(m::BallBerryStomata, β::Real; targets = (:vcmax, :rd, :g0))
    _check_stress(β, targets)
    g0 = max((:g0 in targets ? β : 1.0) * m.g0, _GS_FLOOR)
    return BallBerryStomata(g1 = (:g1 in targets ? β : 1.0) * m.g1, g0 = g0)
end

function apply_water_stress(m::MedlynStomata, β::Real; targets = (:vcmax, :rd, :g0))
    _check_stress(β, targets)
    g0 = max((:g0 in targets ? β : 1.0) * m.g0, _GS_FLOOR)
    return MedlynStomata(g1 = (:g1 in targets ? β : 1.0) * m.g1, g0 = g0, D_min = m.D_min)
end

"""
    LeafGasExchange

Result of [`leaf_gas_exchange`](@ref):

| field | unit | meaning |
| :-- | :-- | :-- |
| `An`, `A`, `Rd` | µmol m⁻² s⁻¹ | net and gross assimilation, day respiration |
| `ci`, `cs` | µmol mol⁻¹ | intercellular and leaf-surface CO₂ |
| `gs`, `gb` | mol m⁻² s⁻¹ | stomatal and boundary-layer conductance to H₂O |
| `gc` | mol m⁻² s⁻¹ | total leaf conductance to CO₂, ``(1.4/g_b + 1.6/g_s)^{-1}`` |
| `hs` | – | relative humidity at the leaf surface |
| `Ds` | Pa | vapour-pressure deficit at the leaf surface |
| `E` | mol m⁻² s⁻¹ | transpiration ([`leaf_transpiration`](@ref)) |
| `LE` | W m⁻² | latent heat of `E` at leaf temperature |
| `limitation` | – | `:rubisco`, `:light` or `:export` at the solution |
| `iterations`, `converged` | – | root-finder diagnostics |

At the solution ``A_n = g_c (c_a - c_i) = \\frac{g_b}{1.4}(c_a - c_s)
= \\frac{g_s}{1.6}(c_s - c_i)``.
"""
struct LeafGasExchange
    An::Float64
    A::Float64
    Rd::Float64
    ci::Float64
    cs::Float64
    gs::Float64
    gb::Float64
    gc::Float64
    hs::Float64
    Ds::Float64
    E::Float64
    LE::Float64
    limitation::Symbol
    iterations::Int
    converged::Bool
end

"""
    leaf_gas_exchange(p, model; ca, Q, T_leaf, e_air, gb = Inf, p_air = 101325.0,
                      β = 1.0, stress_targets = (:vcmax, :rd, :g0),
                      oi = nothing, colimitation = true,
                      c = DEFAULT_CONSTANTS, maxiter = 200) -> LeafGasExchange

Leaf photosynthesis, stomatal conductance and transpiration solved
together. Inputs: photosynthetic parameters `p` ([`C3Parameters`](@ref)
/ [`C4Parameters`](@ref)), a stomatal `model`
([`BallBerryStomata`](@ref) / [`MedlynStomata`](@ref)), ambient CO₂ `ca`
(µmol mol⁻¹), absorbed photon flux `Q` (µmol m⁻² s⁻¹), leaf temperature
`T_leaf` (K), vapour pressure of the surrounding air `e_air` (Pa), leaf
boundary-layer conductance to water vapour `gb` (mol m⁻² s⁻¹; see
[`leaf_boundary_layer_conductance`](@ref); `Inf` = perfectly coupled)
and air pressure `p_air` (Pa, used only for transpiration). `β` is the
soil-water-stress factor applied by [`apply_water_stress`](@ref);
`oi` (mmol mol⁻¹) overrides the C3 O₂ mole fraction.

Four equations close the system (CLM5 §2.9.6, eqs. 2.9.19-2.9.27): the
biochemistry ``A_n = f(c_i)`` ([`leaf_assimilation`](@ref)); diffusion
through the boundary layer, ``c_s = c_a - 1.4 A_n / g_b``; the stomatal
closure ``g_s(A_n, c_s, e_s)`` ([`stomatal_conductance`](@ref), exact in
the leaf-surface humidity); and diffusion through the pore,
``c_i = c_s - 1.6 A_n / g_s``. They reduce to one scalar equation
``R(c_i) = c_i - [c_a - A_n(c_i)(1.4/g_b + 1.6/g_s)] = 0``, solved on a
bracket by the Illinois method of `SurfaceExchange`, like CLM5's hybrid
secant/Brent solver (Sun et al. 2012), which replaced CLM4's
non-convergent fixed-point iteration. ``R(0) < 0`` always (no gross
uptake at ``c_i = 0``); the upper end is ``c_a``, or, when respiration
exceeds uptake there (darkness), the ``c_i > c_a`` it implies.
Typically 5-20 iterations; near the compensation point, where ``g_s``
drops from the closure value to the intercept as ``A_n`` changes sign,
``R`` is steep and up to ~100 may be needed (hence `maxiter = 200`);
`converged` reports the outcome.

The leaf temperature is an input: the leaf energy balance that sets it
belongs to the land column (#21) and the resolved canopy (#24).
"""
function leaf_gas_exchange(p::AbstractPhotosynthesisParameters, model::AbstractStomatalModel;
                           ca::Real, Q::Real, T_leaf::Real, e_air::Real, gb::Real = Inf,
                           p_air::Real = 101325.0, β::Real = 1.0,
                           stress_targets = (:vcmax, :rd, :g0), oi = nothing,
                           colimitation::Bool = true,
                           c::ThermodynamicConstants = DEFAULT_CONSTANTS, maxiter::Integer = 200)
    ca > 0 || throw(DomainError(ca, "ambient CO₂ must be positive"))
    gb > 0 || throw(DomainError(gb, "boundary-layer conductance must be positive"))
    ps = apply_water_stress(p, β; targets = stress_targets)
    sm = apply_water_stress(model, β; targets = stress_targets)
    ei = saturation_vapor_pressure(T_leaf, c)
    rbc = isinf(gb) ? 0.0 : BOUNDARY_LAYER_DIFFUSIVITY_RATIO / gb   # CO₂ resistance, m² s mol⁻¹
    kw = oi === nothing ? (; colimitation = colimitation) : (; oi = oi, colimitation = colimitation)

    function state(ci)
        la = leaf_assimilation(ps, ci, Q, T_leaf; kw...)
        An = la.An
        cs = max(ca - rbc * An, 1e-6)                      # CLM: cs = max(cs, 1e-6)
        gs = stomatal_conductance(sm, An, cs, e_air, ei, gb)
        return la, cs, gs, ca - An * (rbc + STOMATAL_DIFFUSIVITY_RATIO / gs)
    end
    R(ci) = ci - state(ci)[4]

    lo, hi = 0.0, Float64(ca)
    flo, fhi = R(lo), R(hi)
    iters, conv = 0, true
    if fhi < 0                       # net loss at ci = ca: the root lies above ca
        for _ in 1:60
            hi = state(hi)[4] + 1e-9 * hi
            fhi = R(hi)
            fhi >= 0 && break
        end
    end
    if flo >= 0
        ci = lo
    elseif fhi == 0
        ci = hi
    elseif fhi > 0
        ci, iters, conv = _illinois(R, lo, hi, flo, fhi, maxiter)
    else
        ci, conv = hi, false
    end

    la, cs, gs, _ = state(ci)
    gc = 1 / (rbc + STOMATAL_DIFFUSIVITY_RATIO / gs)
    es = isinf(gb) ? float(e_air) : (gb * e_air + gs * ei) / (gb + gs)
    E = leaf_transpiration(gs, gb, T_leaf, e_air, p_air; c = c)
    LE = latent_heat_vapor(T_leaf, c) * c.molmass_water * E
    return LeafGasExchange(la.An, la.A, la.Rd, ci, cs, gs, Float64(gb), gc, es / ei, ei - es,
                           E, LE, la.limitation, iters, conv)
end

# =========================================================================
# 9. Per-PFT stomatal parameters
# =========================================================================

"""
    CLM5_MEDLYN_G1 :: Dict{Symbol, Float64}

Medlyn slope `g1` (kPa^½) per plant functional type, transcribed from
CLM5 Technical Note Table 2.9.1 (Lawrence et al. 2018; the values of the
CABLE model, De Kauwe et al. 2015). Keys are those of
[`CLM45_PFT_PHOTOSYNTHESIS`](@ref):

| key | CLM5 row | g1 |
| :-- | :-- | --: |
| `:net_temperate`, `:net_boreal`, `:ndt_boreal` | NET Temperate/Boreal, NDT Boreal | 2.35 |
| `:bet_tropical`, `:bet_temperate` | BET Tropical/temperate | 4.12 |
| `:bdt_tropical`, `:bdt_temperate`, `:bdt_boreal` | BDT tropical/temperate/boreal | 4.45 |
| `:bes_temperate`, `:bds_temperate`, `:bds_boreal` | BES temperate, BDS temperate/boreal | 4.70 |
| `:c3_arctic_grass` | C3 arctic grass | 2.22 |
| `:c3_grass` | C3 grass | 5.25 |
| `:c4_grass` | C4 grass | 1.62 |
| `:corn` | Temperate Corn | 1.79 |
| `:temperate_cereal` | Spring Wheat | 5.79 |
| `:soybean` | Temperate Soybean | 5.79 |

The generic `:crop` and `:winter_cereal` rows of CLM4.5 have no row in
Table 2.9.1 and are deliberately absent rather than guessed.
"""
const CLM5_MEDLYN_G1 = Dict{Symbol,Float64}(
    :net_temperate => 2.35, :net_boreal => 2.35, :ndt_boreal => 2.35,
    :bet_tropical => 4.12, :bet_temperate => 4.12,
    :bdt_tropical => 4.45, :bdt_temperate => 4.45, :bdt_boreal => 4.45,
    :bes_temperate => 4.70, :bds_temperate => 4.70, :bds_boreal => 4.70,
    :c3_arctic_grass => 2.22, :c3_grass => 5.25, :c4_grass => 1.62,
    :corn => 1.79, :temperate_cereal => 5.79, :soybean => 5.79,
)

"""
    pft_stomatal_model(pft::Symbol; model = :medlyn) -> MedlynStomata | BallBerryStomata

Stomatal closure for a CLM plant functional type.

  * `model = :medlyn` — CLM5: `g1` from [`CLM5_MEDLYN_G1`](@ref),
    `g0 = 100 µmol m⁻² s⁻¹` for C3 and C4 (CLM5 §2.9.3).
  * `model = :ball_berry` — CLM4.5: slope `m = 9` (C3) or `4` (C4)
    (Table 8.1) and intercept `b = 10000` (C3) or `40000` (C4)
    µmol m⁻² s⁻¹ (`bbbopt` in CLM's PhotosynthesisMod.F90).
"""
function pft_stomatal_model(pft::Symbol; model::Symbol = :medlyn)
    haskey(CLM45_PFT_PHOTOSYNTHESIS, pft) ||
        throw(ArgumentError("unknown PFT :$pft; see keys(CLM45_PFT_PHOTOSYNTHESIS)"))
    c4 = CLM45_PFT_PHOTOSYNTHESIS[pft].pathway === :C4
    if model === :ball_berry
        return c4 ? BallBerryStomata(g1 = 4.0, g0 = 0.04) : BallBerryStomata(g1 = 9.0, g0 = 0.01)
    elseif model === :medlyn
        haskey(CLM5_MEDLYN_G1, pft) ||
            throw(ArgumentError("CLM5 Table 2.9.1 has no Medlyn g1 for :$pft"))
        return MedlynStomata(g1 = CLM5_MEDLYN_G1[pft], g0 = 1.0e-4)
    else
        throw(ArgumentError("model must be :medlyn or :ball_berry, got :$model"))
    end
end

end # module
