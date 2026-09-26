"""
    OceanMixedLayer

Lake and coastal-sea surface for Era V (Phase 52, issue #38): a layered
water column with a Kraus–Turner / Niiler–Kraus bulk mixed layer,
shortwave penetration, a Fairall et al. (1996a) cool skin and a diurnal
warm layer (Fairall et al. 1996b, the COARE form, or the prognostic
Zeng & Beljaars 2005 scheme), so the water surface temperature is
prognostic.

The pieces, bottom-up:

  1. **Equation of state** — [`AbstractWaterEOS`](@ref): the UNESCO
     (EOS-80) one-atmosphere seawater density, valid for fresh water
     (with its 4 °C density maximum) and seawater alike, and a linear
     EOS for analytic tests. Thermal expansion and haline contraction
     are its analytic derivatives.
  2. **Shortwave penetration** — [`SolarBands`](@ref): sums of
     exponentials, with the Paulson & Simpson (1977) fits to the Jerlov
     water types ([`jerlov_water`](@ref)) and the Soloviev three-band
     profile that COARE and ECMWF use for the warm layer
     ([`soloviev_solar`](@ref)).
  3. **Cool skin** — [`cool_skin`](@ref): the Saunders/Fairall
     conductive sublayer.
  4. **Warm layer** — [`FairallWarmLayer`](@ref) and
     [`ZengBeljaarsWarmLayer`](@ref), both behind
     [`step_warm_layer!`](@ref).
  5. **The column** — [`WaterColumn`](@ref), an
     `SurfaceExchange.AbstractSurface`: heating by the surface
     fluxes and the penetrating shortwave, convective adjustment, and
     wind-driven entrainment by a mixed-layer energy budget.

Air–sea *fluxes* are not computed here: `step_surface!` takes them from
`SurfaceExchange.turbulent_fluxes` or from any flux function passed in
(the hook that `AirSeaFlux`, issue #39, will use for COARE and
wave-dependent roughness). Energy bookkeeping: every joule the surface
flux delivers ends up in the column or leaves through its bottom, to
round-off; the cool skin and warm layer are *diagnostic* skin
corrections that hold no heat of their own.

# References

  * Fairall, C. W., Bradley, E. F., Godfrey, J. S., Wick, G. A., Edson,
    J. B. & Young, G. S. (1996a). Cool-skin and warm-layer effects on sea
    surface temperature. *J. Geophys. Res.*, 101(C1), 1295-1308.
  * Fairall, C. W., Bradley, E. F., Rogers, D. P., Edson, J. B. & Young,
    G. S. (1996b). Bulk parameterization of air-sea fluxes for TOGA
    COARE. *J. Geophys. Res.*, 101(C2), 3747-3764.
  * Fairall, C. W., Bradley, E. F., Hare, J. E., Grachev, A. A. & Edson,
    J. B. (2003). Bulk parameterization of air-sea fluxes: updates and
    verification for the COARE algorithm. *J. Climate*, 16, 571-591.
  * Zeng, X. & Beljaars, A. (2005). A prognostic scheme of sea surface
    skin temperature for modeling and data assimilation. *Geophys. Res.
    Lett.*, 32, L14605.
  * ECMWF (2017). IFS Documentation Cy43r1, Part IV: Physical processes,
    §8.10 (cool skin and warm layer).
  * Paulson, C. A. & Simpson, J. J. (1977). Irradiance measurements in
    the upper ocean. *J. Phys. Oceanogr.*, 7, 952-956.
  * Kraus, E. B. & Turner, J. S. (1967). A one-dimensional model of the
    seasonal thermocline II. *Tellus*, 19, 98-106.
  * Niiler, P. P. & Kraus, E. B. (1977). One-dimensional models of the
    upper ocean. In *Modelling and Prediction of the Upper Layers of the
    Ocean*, Pergamon, 143-172.
  * Kato, H. & Phillips, O. M. (1969). On the penetration of a turbulent
    layer into stratified fluid. *J. Fluid Mech.*, 37, 643-655.
  * UNESCO (1981). Background papers and supporting data on the
    International Equation of State of Seawater 1980. *UNESCO Tech. Pap.
    Mar. Sci.* 38; check values from Fofonoff & Millard (1983), *UNESCO
    Tech. Pap. Mar. Sci.* 44.
  * Kraus, E. B. & Businger, J. A. (1994). *Atmosphere-Ocean
    Interaction*, 2nd ed., Oxford University Press.
"""
module OceanMixedLayer

using ..MoistThermodynamics: saturation_specific_humidity, latent_heat_vapor
using ..SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, grav,
                         cp_water, von_karman, AbstractSurface, SurfaceForcing,
                         SurfaceFluxes, turbulent_fluxes, AbstractStabilityFunctions,
                         BusingerDyerBeljaarsHoltslag
import ..SurfaceExchange: surface_temperature, roughness_lengths, surface_albedo,
                          surface_emissivity, step_surface!, moisture_availability,
                          surface_specific_humidity, phi_m, phi_h, psi_m, psi_h
using ..SurfaceOptics: material_emissivity

export AbstractWaterEOS, UNESCOWaterEOS, LinearWaterEOS,
       water_density, water_thermal_expansion, water_haline_contraction,
       SolarBands, jerlov_water, soloviev_solar,
       solar_transmission, solar_absorbed_between, solar_mean_transmission

const _T0C = 273.15   # K at 0 °C

# =========================================================================
# 1. Equation of state
# =========================================================================

"""
    AbstractWaterEOS

An equation of state for lake or sea water at the surface pressure. A
subtype implements `water_density(eos, T, S)` (kg m⁻³) with `T` in
kelvin and salinity `S` in psu, and the derivatives
[`water_thermal_expansion`](@ref) ``α = -ρ^{-1} ∂ρ/∂T`` and
[`water_haline_contraction`](@ref) ``β = ρ^{-1} ∂ρ/∂S``.

Pressure (thermobaric) effects are ignored: the column compares
densities at one atmosphere, which for lakes shallower than ~100 m
shifts the 4 °C density maximum by less than 0.2 K.
"""
abstract type AbstractWaterEOS end

"""
    UNESCOWaterEOS()

The one-atmosphere International Equation of State of Seawater 1980
(EOS-80; Millero & Poisson 1981, UNESCO 1981):

    ρ(t, S) = ρ_w(t) + A(t) S + B(t) S^{3/2} + C S²

with ``ρ_w`` the SMOW density polynomial. Valid 0–40 °C, 0–42 psu; at
`S = 0` it is pure water, with the density maximum at 3.98 °C that
drives lake overturning. `t` is converted from kelvin as ``T - 273.15``
(the IPTS-68 / ITS-90 difference, < 0.01 K here, is ignored). Check
values (Fofonoff & Millard 1983): ρ(5 °C, 0) = 999.96675,
ρ(5 °C, 35) = 1027.67547, ρ(25 °C, 35) = 1023.34306 kg m⁻³.
"""
struct UNESCOWaterEOS <: AbstractWaterEOS end

"""
    LinearWaterEOS(; ρ0 = 1000, T0 = 283.15, S0 = 0, α = 2e-4, β = 7.6e-4)

``ρ = ρ_0 [1 - α (T - T_0) + β (S - S_0)]`` — constant expansion
coefficients, for analytic tests (a linear temperature profile is then a
constant buoyancy frequency).
"""
Base.@kwdef struct LinearWaterEOS{T<:AbstractFloat} <: AbstractWaterEOS
    ρ0::T = 1000.0
    T0::T = 283.15
    S0::T = 0.0
    α::T  = 2e-4
    β::T  = 7.6e-4
end

"""
    water_density(eos, T, S = 0) -> kg m⁻³

Density of water at temperature `T` (K) and salinity `S` (psu), surface
pressure. See [`UNESCOWaterEOS`](@ref), [`LinearWaterEOS`](@ref).
"""
function water_density end

"""
    water_thermal_expansion(eos, T, S = 0) -> K⁻¹

``α = -ρ^{-1} ∂ρ/∂T``. Negative for fresh water below 3.98 °C.
"""
function water_thermal_expansion end

"""
    water_haline_contraction(eos, T, S = 0) -> psu⁻¹

``β = ρ^{-1} ∂ρ/∂S`` (≈ 7.5 × 10⁻⁴ psu⁻¹ for seawater).
"""
function water_haline_contraction end

@inline _rho_smow(t) = 999.842594 + t * (6.793952e-2 + t * (-9.095290e-3 +
                        t * (1.001685e-4 + t * (-1.120083e-6 + t * 6.536332e-9))))
@inline _drho_smow(t) = 6.793952e-2 + t * (2 * -9.095290e-3 + t * (3 * 1.001685e-4 +
                         t * (4 * -1.120083e-6 + t * 5 * 6.536332e-9)))
@inline _eosA(t) = 8.24493e-1 + t * (-4.0899e-3 + t * (7.6438e-5 + t * (-8.2467e-7 + t * 5.3875e-9)))
@inline _deosA(t) = -4.0899e-3 + t * (2 * 7.6438e-5 + t * (3 * -8.2467e-7 + t * 4 * 5.3875e-9))
@inline _eosB(t) = -5.72466e-3 + t * (1.0227e-4 + t * -1.6546e-6)
@inline _deosB(t) = 1.0227e-4 + t * 2 * -1.6546e-6
const _EOS_C = 4.8314e-4

function water_density(::UNESCOWaterEOS, T::Real, S::Real = 0.0)
    t = T - _T0C
    s = max(S, zero(S))
    return _rho_smow(t) + _eosA(t) * s + _eosB(t) * s * sqrt(s) + _EOS_C * s^2
end
function water_thermal_expansion(eos::UNESCOWaterEOS, T::Real, S::Real = 0.0)
    t = T - _T0C
    s = max(S, zero(S))
    return -(_drho_smow(t) + _deosA(t) * s + _deosB(t) * s * sqrt(s)) / water_density(eos, T, S)
end
function water_haline_contraction(eos::UNESCOWaterEOS, T::Real, S::Real = 0.0)
    t = T - _T0C
    s = max(S, zero(S))
    return (_eosA(t) + 1.5 * _eosB(t) * sqrt(s) + 2 * _EOS_C * s) / water_density(eos, T, S)
end

water_density(e::LinearWaterEOS, T::Real, S::Real = 0.0) =
    e.ρ0 * (1 - e.α * (T - e.T0) + e.β * (S - e.S0))
water_thermal_expansion(e::LinearWaterEOS, T::Real, S::Real = 0.0) =
    e.ρ0 * e.α / water_density(e, T, S)
water_haline_contraction(e::LinearWaterEOS, T::Real, S::Real = 0.0) =
    e.ρ0 * e.β / water_density(e, T, S)

# =========================================================================
# 2. Shortwave penetration
# =========================================================================

"""
    SolarBands(fractions, scales)

Net shortwave transmitted to depth `z` (m, positive down) as a sum of
exponentials,

    I(z)/I(0) = Σᵢ aᵢ exp(-z/ζᵢ),    Σᵢ aᵢ = 1,

with fractions `aᵢ` and e-folding depths `ζᵢ` (m). Construct the
standard profiles with [`jerlov_water`](@ref) or
[`soloviev_solar`](@ref).
"""
struct SolarBands{N,T<:AbstractFloat}
    fractions::NTuple{N,T}
    scales::NTuple{N,T}
    function SolarBands(a::NTuple{N,<:Real}, ζ::NTuple{N,<:Real}) where {N}
        T = float(promote_type(eltype(a), eltype(ζ)))
        all(>(0), ζ) || throw(ArgumentError("SolarBands: e-folding depths must be > 0"))
        isapprox(sum(a), 1; atol = 1e-6) ||
            throw(ArgumentError("SolarBands: fractions must sum to 1 (got $(sum(a)))"))
        new{N,T}(T.(a), T.(ζ))
    end
end

# Paulson & Simpson (1977), Table 2: (R, ζ₁, ζ₂) for Jerlov's types.
const _JERLOV = Dict(:I   => (0.58, 0.35, 23.0),
                     :IA  => (0.62, 0.60, 20.0),
                     :IB  => (0.67, 1.00, 17.0),
                     :II  => (0.77, 1.50, 14.0),
                     :III => (0.78, 1.40, 7.9))

"""
    jerlov_water(type = :IB) -> SolarBands

Paulson & Simpson (1977) two-band fit
``I/I_0 = R e^{-z/ζ_1} + (1-R) e^{-z/ζ_2}`` for Jerlov's oceanic water
types `:I` (clearest), `:IA`, `:IB`, `:II`, `:III` (most turbid; a fair
choice for productive lakes and coastal water):

| type | R    | ζ₁ (m) | ζ₂ (m) |
|:---- |:---- |:------ |:------ |
| I    | 0.58 | 0.35   | 23     |
| IA   | 0.62 | 0.60   | 20     |
| IB   | 0.67 | 1.0    | 17     |
| II   | 0.77 | 1.5    | 14     |
| III  | 0.78 | 1.4    | 7.9    |

The first band is the red/near-infrared absorbed within a metre or so,
the second the blue-green that penetrates.
"""
function jerlov_water(type::Symbol = :IB)
    haskey(_JERLOV, type) ||
        throw(ArgumentError("jerlov_water: type must be one of $(sort(collect(keys(_JERLOV))))"))
    R, ζ1, ζ2 = _JERLOV[type]
    return SolarBands((R, 1 - R), (ζ1, ζ2))
end

"""
    soloviev_solar(variant = :coare) -> SolarBands

The three-band Soloviev profile used by the diurnal warm-layer schemes,
fractions (0.28, 0.27, 0.45):

  * `:coare` — e-folding depths (0.014, 0.357, 12.82) m, as in the
    COARE 3.x warm-layer code (Fairall et al. 1996b);
  * `:zeng_beljaars` — attenuation (71.5, 2.8, 0.06) m⁻¹, i.e. depths
    (0.0140, 0.357, 16.7) m, as in Zeng & Beljaars (2005) and the IFS
    (Cy43r1 Part IV eq. 8.157).
"""
function soloviev_solar(variant::Symbol = :coare)
    a = (0.28, 0.27, 0.45)
    variant === :coare && return SolarBands(a, (0.014, 0.357, 12.82))
    variant === :zeng_beljaars && return SolarBands(a, (1 / 71.5, 1 / 2.8, 1 / 0.06))
    throw(ArgumentError("soloviev_solar: variant must be :coare or :zeng_beljaars"))
end

"""
    solar_transmission(p::SolarBands, z) -> I(z)/I(0)

Fraction of the net surface shortwave reaching depth `z ≥ 0` (m).
"""
function solar_transmission(p::SolarBands, z::Real)
    zz = max(z, zero(z))
    s = zero(float(zz))
    for i in eachindex(p.fractions)
        s += p.fractions[i] * exp(-zz / p.scales[i])
    end
    return s
end

"""
    solar_absorbed_between(p::SolarBands, z1, z2) -> fraction

Fraction of the net surface shortwave absorbed between depths
`0 ≤ z1 ≤ z2`, ``I(z_1) - I(z_2)``.
"""
solar_absorbed_between(p::SolarBands, z1::Real, z2::Real) =
    solar_transmission(p, z1) - solar_transmission(p, z2)

"""
    solar_mean_transmission(p::SolarBands, D) -> (1/D) ∫₀ᴰ I(z)/I(0) dz

Depth-averaged transmission over the top `D` metres,
``Σ a_i ζ_i (1 - e^{-D/ζ_i}) / D``. One minus it is the solar fraction
that heats a layer with a linear warming profile — the `fxp` of the
COARE warm-layer code.
"""
function solar_mean_transmission(p::SolarBands, D::Real)
    D > 0 || return one(float(D))
    s = zero(float(D))
    for i in eachindex(p.fractions)
        ζ = p.scales[i]
        s += p.fractions[i] * ζ * (-expm1(-D / ζ))
    end
    return s / D
end

end # module
