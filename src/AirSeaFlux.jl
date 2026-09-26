"""
    AirSeaFlux

Air–sea exchange over open water (Era V, Phase 52): the sea-surface
roughness that responds to wind and waves, the scalar roughness that
follows from it, the COARE convective stability functions and
gustiness, and a COARE-style bulk algorithm that ties them together.

It *extends* the Era V contract `SurfaceExchange` rather than restating
it: the Monin–Obukhov solve is `SurfaceExchange.solve_surface_layer`,
the stability functions are a new `AbstractStabilityFunctions` subtype
([`COAREStability`](@ref)), the sea surface is an `AbstractSurface`
([`SeaSurface`](@ref)), saturation and latent heats come from
`MoistThermodynamics`, the air viscosity from `P3Microphysics`
(Sutherland's law), the water emissivity from `SurfaceOptics`.

Four parts:

  1. **Stability and gustiness** — [`COAREStability`](@ref) (Kansas
     forms blended into the free-convection ``(-ζ)^{-1/3}`` limit,
     Beljaars–Holtslag on the stable side) and
     [`ConvectiveGustiness`](@ref) (``w_g = β w_*``).
  2. **Momentum roughness** — Charnock plus the smooth-flow limit
     (Smith 1988), the wind-dependent Charnock of COARE 3.0 and 3.5,
     and the wave-dependent forms: wave age (Edson et al. 2013,
     Oost et al. 2002) and wave steepness / sea state (Taylor & Yelland
     2001, Edson et al. 2013). See [`sea_roughness_length`](@ref).
  3. **Scalar roughness** — ``z_{0t}, z_{0q}`` as functions of the
     roughness Reynolds number (COARE 3.0, COARE 3.5,
     Brutsaert–Garratt). See [`sea_scalar_roughness`](@ref).
  4. **Bulk algorithm** — [`air_sea_fluxes`](@ref) iterates roughness,
     gustiness and Monin–Obukhov stability to a joint fixed point;
     [`coare30_bulk`](@ref) and [`coare35_bulk`](@ref) are the published
     configurations.

Sign conventions are those of `SurfaceExchange`: fluxes positive upward,
``θ_* < 0`` for a surface warmer than the air, ``ζ > 0`` stable.

# References

  * Charnock, H. (1955). Wind stress on a water surface. *Q. J. R.
    Meteorol. Soc.*, 81, 639-640.
  * Smith, S. D. (1988). Coefficients for sea surface wind stress, heat
    flux, and wind profiles as a function of wind speed and temperature.
    *J. Geophys. Res.*, 93, 15467-15472.
  * Fairall, C. W., Bradley, E. F., Rogers, D. P., Edson, J. B. & Young,
    G. S. (1996). Bulk parameterization of air-sea fluxes for TOGA
    COARE. *J. Geophys. Res.*, 101, 3747-3764.
  * Grachev, A. A. & Fairall, C. W. (1997). Dependence of the
    Monin–Obukhov stability parameter on the bulk Richardson number
    over the ocean. *J. Appl. Meteorol.*, 36, 406-414.
  * Fairall, C. W., Bradley, E. F., Hare, J. E., Grachev, A. A. & Edson,
    J. B. (2003). Bulk parameterization of air–sea fluxes: updates and
    verification for the COARE algorithm. *J. Climate*, 16, 571-591.
  * Taylor, P. K. & Yelland, M. J. (2001). The dependence of sea surface
    roughness on the height and steepness of the waves. *J. Phys.
    Oceanogr.*, 31, 572-590.
  * Oost, W. A., Komen, G. J., Jacobs, C. M. J. & van Oort, C. (2002).
    New evidence for a relation between wind stress and wave age from
    measurements during ASGAMAGE. *Boundary-Layer Meteorol.*, 103,
    409-438.
  * Edson, J. B. et al. (2013). On the exchange of momentum over the
    open ocean. *J. Phys. Oceanogr.*, 43, 1589-1610 (corrigendum 2014,
    44, 2590: ``m = 0.0017``).
  * Beljaars, A. C. M. (1995). The parametrization of surface fluxes in
    large-scale models under free convection. *Q. J. R. Meteorol. Soc.*,
    121, 255-270.
"""
module AirSeaFlux

using ..MoistThermodynamics: ThermodynamicConstants, PhasePartition, R_d, R_v,
                             air_density, saturation_vapor_pressure, molmass_ratio,
                             heat_capacity_moist, latent_heat_vapor,
                             dry_adiabatic_lapse_rate
using ..SurfaceExchange
using ..SurfaceExchange: grav, von_karman, _illinois
import ..SurfaceExchange: phi_m, phi_h, psi_m, psi_h,
                          surface_temperature, roughness_lengths,
                          surface_albedo, surface_emissivity,
                          moisture_availability, surface_specific_humidity,
                          turbulent_fluxes, step_surface!
using ..P3Microphysics: p3_air_viscosity
using ..SurfaceOptics: material_emissivity

export COAREStability, ConvectiveGustiness, gustiness_velocity,
       convective_velocity_scale

# =========================================================================
# 1. COARE stability functions (Grachev–Fairall convective blend)
# =========================================================================

"""
    COAREStability(version::Symbol = :coare35)
    COAREStability(; momentum, heat, γc_m = 10.15, γc_h = 34.15)

The stability functions of the COARE bulk algorithm (Fairall et al.
1996, 2003; Grachev & Fairall 1997), a further
`SurfaceExchange.AbstractStabilityFunctions`.

*Unstable* (``ζ < 0``): the Kansas (Businger–Dyer) form ``ψ_K`` is
blended into a free-convection form ``ψ_C`` that obeys the
``φ ∝ (-ζ)^{-1/3}`` scaling of local free convection,

    ψ = (1 - f) ψ_K + f ψ_C,        f = ζ² / (1 + ζ²),
    ψ_C = (3/2) ln[(y² + y + 1)/3] - √3 atan[(2y + 1)/√3] + π/√3,
    y   = (1 - γ_c ζ)^{1/3},

with ``γ_c = 10.15`` for momentum and ``34.15`` for scalars (Fairall et
al. 2003: "gave the smoothest blend"). ``ψ_C`` is the exact integral of
``φ_C = (1 - γ_c ζ)^{-1/3}``.

*Stable* (``ζ ≥ 0``): Beljaars & Holtslag (1991).

Both the Kansas and the stable branches **are** the contract's
`BusingerDyerBeljaarsHoltslag` with COARE's coefficients (the fields
`momentum` and `heat`), so only the convective part is new here:

| version   | momentum (Kansas γ, stable a, b)      | heat (γ, a, b)        |
|:--------- |:------------------------------------- |:--------------------- |
| `:coare30`| 15; a = 1, b = 2/3 (`psiu_26`, 3.0)   | 15; a = 1, b = 2/3    |
| `:coare35`| 15; a = 0.7, b = 3/4 (`psiu_26`, 3.5) | 15; a = 1, b = 2/3    |

with ``c = 5``, ``d = 0.35``. The reference codes write the stable heat
function with the rounded constants 14.28 and 8.525, so their
``ψ_h(0) = -0.0045``; the exact ``c/d`` used here gives ``ψ_h(0) = 0``.
"""
Base.@kwdef struct COAREStability{T<:AbstractFloat} <: AbstractStabilityFunctions
    momentum::BusingerDyerBeljaarsHoltslag{T} =
        BusingerDyerBeljaarsHoltslag(γ_m = 15.0, γ_h = 15.0, a = 0.7, b = 0.75, c = 5.0, d = 0.35)
    heat::BusingerDyerBeljaarsHoltslag{T} =
        BusingerDyerBeljaarsHoltslag(γ_m = 15.0, γ_h = 15.0, a = 1.0, b = 2 / 3, c = 5.0, d = 0.35)
    γc_m::T = 10.15
    γc_h::T = 34.15
end

function COAREStability(version::Symbol)
    version === :coare35 && return COAREStability()
    version === :coare30 && return COAREStability(
        momentum = BusingerDyerBeljaarsHoltslag(γ_m = 15.0, γ_h = 15.0, a = 1.0, b = 2 / 3,
                                                c = 5.0, d = 0.35))
    throw(ArgumentError("COAREStability: version must be :coare30 or :coare35"))
end

# Free-convection form and its gradient.
function _psi_conv(ζ, γc)
    y = cbrt(1 - γc * ζ)
    return 1.5 * log((y^2 + y + 1) / 3) - sqrt(3) * atan((2y + 1) / sqrt(3)) + π / sqrt(3)
end
_phi_conv(ζ, γc) = 1 / cbrt(1 - γc * ζ)

_blend(ζ) = ζ^2 / (1 + ζ^2)
_dblend(ζ) = 2ζ / (1 + ζ^2)^2

function psi_m(sf::COAREStability, ζ::Real)
    ψK = psi_m(sf.momentum, ζ)
    ζ < 0 || return ψK
    f = _blend(ζ)
    return (1 - f) * ψK + f * _psi_conv(ζ, sf.γc_m)
end

function psi_h(sf::COAREStability, ζ::Real)
    ψK = psi_h(sf.heat, ζ)
    ζ < 0 || return ψK
    f = _blend(ζ)
    return (1 - f) * ψK + f * _psi_conv(ζ, sf.γc_h)
end

# φ = 1 - ζ dψ/dζ applied to the blend:
#   φ = (1-f) φ_K + f φ_C - ζ f′ (ψ_C - ψ_K)
function phi_m(sf::COAREStability, ζ::Real)
    φK = phi_m(sf.momentum, ζ)
    ζ < 0 || return φK
    f = _blend(ζ)
    ψK, ψC = psi_m(sf.momentum, ζ), _psi_conv(ζ, sf.γc_m)
    return (1 - f) * φK + f * _phi_conv(ζ, sf.γc_m) - ζ * _dblend(ζ) * (ψC - ψK)
end

function phi_h(sf::COAREStability, ζ::Real)
    φK = phi_h(sf.heat, ζ)
    ζ < 0 || return φK
    f = _blend(ζ)
    ψK, ψC = psi_h(sf.heat, ζ), _psi_conv(ζ, sf.γc_h)
    return (1 - f) * φK + f * _phi_conv(ζ, sf.γc_h) - ζ * _dblend(ζ) * (ψC - ψK)
end

# =========================================================================
# 2. Convective gustiness
# =========================================================================

"""
    ConvectiveGustiness(; β = 1.2, z_i = 600, w_min = 0.2, w_init = 0.5)

Sub-grid wind variability from boundary-layer-scale convective eddies
(Godfrey & Beljaars 1991; Beljaars 1995; Fairall et al. 1996):

    w_g = β w_*,   w_* = (B z_i)^{1/3}   if B > 0,
    w_g = w_min                          otherwise,

with the surface buoyancy flux ``B = (g/θ_v) \\overline{w'θ_v'}``
(m² s⁻³) and mixed-layer depth `z_i` (m). The bulk solver then uses the
effective wind ``S = \\sqrt{U² + w_g²}``, which keeps the scalar fluxes
finite as ``U → 0`` (free convection). β = 1.2 and ``w_{min} = 0.2``
m s⁻¹ are the values of the COARE 3.0/3.5 reference codes (Fairall et
al. 2003 quote β = 1.25 in the text); `w_init` is the first guess.
As in those codes, ``w_{min}`` applies only when ``B ≤ 0``: a weakly
unstable surface has ``w_g = β w_*`` even if that is below ``w_{min}``.
"""
Base.@kwdef struct ConvectiveGustiness{T<:AbstractFloat}
    β::T      = 1.2
    z_i::T    = 600.0
    w_min::T  = 0.2
    w_init::T = 0.5
end

"""
    convective_velocity_scale(B, z_i) -> w_*

Deardorff convective velocity ``w_* = (B z_i)^{1/3}`` for a surface
buoyancy flux `B > 0` (m² s⁻³); zero for ``B ≤ 0``.
"""
convective_velocity_scale(B::Real, z_i::Real) = B > 0 ? cbrt(B * z_i) : zero(float(B))

"""
    gustiness_velocity(g::ConvectiveGustiness, B) -> w_g

Gustiness velocity for surface buoyancy flux `B` (m² s⁻³, positive
upward = unstable); see [`ConvectiveGustiness`](@ref).
"""
gustiness_velocity(g::ConvectiveGustiness, B::Real) =
    B > 0 ? g.β * convective_velocity_scale(B, g.z_i) : g.w_min

end # module
