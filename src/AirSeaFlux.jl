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
       convective_velocity_scale,
       SeaState, deep_water_sea_state, fully_developed_sea_state, wave_age,
       AbstractSeaRoughness, AbstractCharnockRoughness, CharnockRoughness,
       COARE30Charnock, COARE35Charnock, WaveAgeCharnock, OostRoughness,
       TaylorYellandRoughness, SeaStateRoughness,
       smooth_flow_roughness, charnock_parameter, sea_roughness_length,
       AbstractScalarRoughness, COARE30ScalarRoughness, COARE35ScalarRoughness,
       BrutsaertGarrattRoughness, sea_scalar_roughness, roughness_reynolds,
       neutral_transfer_coefficients,
       BulkAirSea, coare30_bulk, coare35_bulk, AirSeaSolution, air_sea_fluxes,
       SeaSurface, sea_surface_solution

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

# =========================================================================
# 3. Air viscosity, sea state
# =========================================================================

"""
    air_kinematic_viscosity(T, p; q = 0, c = DEFAULT_SURFACE_CONSTANTS) -> m² s⁻¹

Kinematic viscosity of (moist) air, ``ν = μ(T)/ρ(T, p, q)``, with the
dynamic viscosity from Sutherland's law (`P3Microphysics.p3_air_viscosity`)
and the density from `MoistThermodynamics.air_density` — delegated, not
restated. At 20 °C and 1013.25 hPa, ν = 1.506e-5 m² s⁻¹; the COARE
reference polynomial gives 1.504e-5.
"""
air_kinematic_viscosity(T::Real, p::Real; q::Real = 0.0,
                        c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    p3_air_viscosity(T) / air_density(T, p, PhasePartition(q), c.thermo)

"""
    SeaState(; c_p, H_s, λ_p)

The dominant-wave description the wave-dependent roughness laws need:
phase speed of the spectral-peak waves `c_p` (m s⁻¹), significant wave
height `H_s` (m, ≈ 4 × rms elevation) and peak wavelength `λ_p` (m).
Build it from a peak period with [`deep_water_sea_state`](@ref) or for a
fully developed sea with [`fully_developed_sea_state`](@ref).
"""
Base.@kwdef struct SeaState{T<:AbstractFloat}
    c_p::T
    H_s::T
    λ_p::T
end

"""
    deep_water_sea_state(T_p, H_s; g = grav()) -> SeaState

Deep-water linear gravity waves of peak period `T_p` (s):
``λ_p = g T_p²/2π``, ``c_p = g T_p/2π`` (Fairall et al. 2003, eq. 26).
"""
function deep_water_sea_state(T_p::Real, H_s::Real; g::Real = grav())
    Tp = float(T_p)
    return SeaState(c_p = g * Tp / 2π, H_s = float(H_s), λ_p = g * Tp^2 / 2π)
end

"""
    fully_developed_sea_state(U10N; g = grav()) -> SeaState

Fully developed wind sea for a 10-m neutral wind `U10N` (m s⁻¹), from
Taylor & Yelland (2001) as used by Fairall et al. (2003, eq. 27):
``H_s = 0.0248\\,U_{10N}²``, ``T_p = 0.729\\,U_{10N}``, i.e. wave age
``c_p/U_{10N} = 1.14``.
"""
fully_developed_sea_state(U10N::Real; g::Real = grav()) =
    deep_water_sea_state(0.729 * U10N, 0.0248 * U10N^2; g = g)

"""
    wave_age(sea::SeaState, u_star) -> c_p/u_*

Wave age in friction-velocity units; a fully developed sea has
``c_p/u_* ≈ 32`` (Donelan 1990; Edson et al. 2013), young (growing)
seas smaller values.
"""
wave_age(sea::SeaState, u_star::Real) = sea.c_p / u_star

# =========================================================================
# 4. Momentum roughness of the sea surface
# =========================================================================

"""
    AbstractSeaRoughness

A law for the momentum roughness length ``z_0`` of the sea surface. A
subtype implements

    sea_roughness_length(r, u_star; ν, g, U10N, sea) -> z0 (m)

where `ν` is the air kinematic viscosity, `U10N` the 10-m neutral wind
(needed by the wind-dependent Charnock laws) and `sea` a
[`SeaState`](@ref) (needed by the wave-dependent laws). Every law here
is the sum of a rough-flow part and the smooth-flow limit
``z_0^{smooth} = 0.11 ν/u_*`` (Smith 1988), and
[`charnock_parameter`](@ref) returns the equivalent
``α = g z_0^{rough}/u_*²`` for any of them.
"""
abstract type AbstractSeaRoughness end

"""
    AbstractCharnockRoughness <: AbstractSeaRoughness

Roughness laws of Charnock form, ``z_0 = α u_*²/g + β ν/u_*``, that differ
only in how the Charnock parameter α is specified; a subtype implements
`charnock_parameter(r, u_star; g, U10N, sea)` and has a field `β`.
"""
abstract type AbstractCharnockRoughness <: AbstractSeaRoughness end

"""
    smooth_flow_roughness(u_star, ν; β = 0.11) -> β ν/u_*

Aerodynamically smooth limit of the roughness length (Smith 1988).
"""
smooth_flow_roughness(u_star::Real, ν::Real; β::Real = 0.11) = β * ν / u_star

"""
    CharnockRoughness(; α = 0.011, β = 0.11)

Charnock (1955) with the smooth-flow term of Smith (1988):
``z_0 = α u_*²/g + β ν/u_*``, the COARE 2.5 law (α = 0.011; Fairall et
al. 1996). Beljaars (1995) and the ECMWF model use α = 0.018.
"""
Base.@kwdef struct CharnockRoughness{T<:AbstractFloat} <: AbstractCharnockRoughness
    α::T = 0.011
    β::T = 0.11
end

"""
    COARE30Charnock(; α_low = 0.011, α_high = 0.018, U_low = 10, U_high = 18, β = 0.11)

The COARE 3.0 Charnock parameter (Fairall et al. 2003, Fig. 1): 0.011
up to ``U_{10N} = 10`` m s⁻¹, rising linearly to 0.018 at 18 m s⁻¹ and
constant beyond. (The reference code evaluates it with the gust-inclusive
wind at measurement height; the paper, followed here, with ``U_{10N}``.)
"""
Base.@kwdef struct COARE30Charnock{T<:AbstractFloat} <: AbstractCharnockRoughness
    α_low::T  = 0.011
    α_high::T = 0.018
    U_low::T  = 10.0
    U_high::T = 18.0
    β::T      = 0.11
end

"""
    COARE35Charnock(; m = 0.0017, b = -0.005, U_max = 19, β = 0.11)

The COARE 3.5 wind-speed-dependent Charnock parameter (Edson et al.
2013, eq. 13 with the 2014 corrigendum m = 0.0017 s m⁻¹):
``α = m U_{10N} + b`` for ``U_{10N} ≤ U_{max}``, constant
``α = m U_{max} + b ≈ 0.028`` above. Below ``U_{10N} = -b/m = 2.9`` m s⁻¹
the fit turns negative; α is floored at 0 there (pure smooth flow). The
reference code does not floor it, which is harmless along its own
iteration path (the smooth term dominates at such winds) but gives a
negative ``z_0`` for larger trial ``u_*``; the difference in ``C_D`` is
below 1 % where it applies.
"""
Base.@kwdef struct COARE35Charnock{T<:AbstractFloat} <: AbstractCharnockRoughness
    m::T     = 0.0017
    b::T     = -0.005
    U_max::T = 19.0
    β::T     = 0.11
end

"""
    WaveAgeCharnock(; A = 0.114, B = 0.622, β = 0.11)

Wave-age-dependent Charnock parameter ``α = A (u_*/c_p)^B`` — younger
waves (larger inverse wave age) are rougher. The default coefficients
are the COARE 3.5 open-ocean fit of Edson et al. (2013, eq. 15), which
gives α = 0.013 at the fully developed ``c_p/u_* = 32`` and spans
0.011–0.018 for ``43 > c_p/u_* > 20``. Needs a [`SeaState`](@ref).
"""
Base.@kwdef struct WaveAgeCharnock{T<:AbstractFloat} <: AbstractCharnockRoughness
    A::T = 0.114
    B::T = 0.622
    β::T = 0.11
end

"""
    OostRoughness(; A = 50, B = 4.5, β = 0.11)

Wave-age law of Oost et al. (2002), α = A (u_*/c_p)^{B-2}, written as a
roughness length with the smooth-flow term as in COARE 3.0 (Fairall et
al. 2003, eq. 25b):

    z_0 = (A/2π) λ_p (u_*/c_p)^B + β ν/u_*.

Needs a [`SeaState`](@ref) (`λ_p`, `c_p`). Derived from North Sea
platform data; Edson et al. (2013) find it too smooth for mature and
too rough for young open-ocean seas.
"""
Base.@kwdef struct OostRoughness{T<:AbstractFloat} <: AbstractSeaRoughness
    A::T = 50.0
    B::T = 4.5
    β::T = 0.11
end

"""
    TaylorYellandRoughness(; A = 1200, B = 4.5, β = 0.11)

Wave-steepness law of Taylor & Yelland (2001), with the smooth-flow term
(Fairall et al. 2003, eq. 25a):

    z_0 = A H_s (H_s/λ_p)^B + β ν/u_*.

The rough part depends only on the sea state (height and steepness of
the dominant waves), not on the wind. Needs a [`SeaState`](@ref).
"""
Base.@kwdef struct TaylorYellandRoughness{T<:AbstractFloat} <: AbstractSeaRoughness
    A::T = 1200.0
    B::T = 4.5
    β::T = 0.11
end

"""
    SeaStateRoughness(; D = 0.091, B = 2, β = 0.11)

Sea-state (wave-slope and wave-age) law of COARE 3.5 (Edson et al. 2013,
eqs. 18–19; Hsu 1974): the rough-flow roughness scales with the
significant wave height,

    z_0 = D H_s (u_*/c_p)^B + β ν/u_*,

D = 0.091, B = 2 (the fit gave 2.02). Edson et al. define the fit for
the rough part and add the smooth-flow term, as here; the COARE 3.5
reference code instead subtracts the smooth term inside the fit (its
total ``z_0 = D H_s (u_*/c_p)^2``). Needs a [`SeaState`](@ref).
"""
Base.@kwdef struct SeaStateRoughness{T<:AbstractFloat} <: AbstractSeaRoughness
    D::T = 0.091
    B::T = 2.0
    β::T = 0.11
end

_need(x, what, r) = x === nothing ?
    throw(ArgumentError("$(nameof(typeof(r))) needs `$what`")) : x

"""
    charnock_parameter(r, u_star; g = grav(), U10N = nothing, sea = nothing, ν = 1.5e-5) -> α

The Charnock parameter of roughness law `r`. For the Charnock-type laws
this is the specified α; for any other law it is the equivalent
``α = g (z_0 - β ν/u_*)/u_*²`` implied by its rough-flow part — the
quantity Edson et al. (2013) and Fairall et al. (2003) plot.
"""
charnock_parameter(r::CharnockRoughness, u_star::Real; kwargs...) = r.α

function charnock_parameter(r::COARE30Charnock, u_star::Real; U10N = nothing, kwargs...)
    U = _need(U10N, :U10N, r)
    w = clamp((U - r.U_low) / (r.U_high - r.U_low), 0, 1)
    return r.α_low + w * (r.α_high - r.α_low)
end

charnock_parameter(r::COARE35Charnock, u_star::Real; U10N = nothing, kwargs...) =
    max(r.m * min(_need(U10N, :U10N, r), r.U_max) + r.b, zero(r.b))

charnock_parameter(r::WaveAgeCharnock, u_star::Real; sea = nothing, kwargs...) =
    r.A * (u_star / _need(sea, :sea, r).c_p)^r.B

function charnock_parameter(r::AbstractSeaRoughness, u_star::Real;
                            g::Real = grav(), ν::Real = 1.5e-5, kwargs...)
    z0 = sea_roughness_length(r, u_star; g = g, ν = ν, kwargs...)
    return g * (z0 - smooth_flow_roughness(u_star, ν; β = r.β)) / u_star^2
end

"""
    sea_roughness_length(r::AbstractSeaRoughness, u_star; ν, g = grav(),
                         U10N = nothing, sea = nothing) -> z0 (m)

Momentum roughness length of the sea surface under law `r` for friction
velocity `u_star` (m s⁻¹) and air kinematic viscosity `ν` (m² s⁻¹).
The wind-dependent laws need `U10N`, the wave-dependent ones a
[`SeaState`](@ref) `sea`. See [`AbstractSeaRoughness`](@ref).
"""
function sea_roughness_length(r::AbstractCharnockRoughness, u_star::Real;
                              ν::Real, g::Real = grav(), U10N = nothing, sea = nothing)
    α = charnock_parameter(r, u_star; g = g, U10N = U10N, sea = sea)
    return α * u_star^2 / g + smooth_flow_roughness(u_star, ν; β = r.β)
end

function sea_roughness_length(r::OostRoughness, u_star::Real;
                              ν::Real, g::Real = grav(), U10N = nothing, sea = nothing)
    s = _need(sea, :sea, r)
    return r.A / 2π * s.λ_p * (u_star / s.c_p)^r.B + smooth_flow_roughness(u_star, ν; β = r.β)
end

function sea_roughness_length(r::TaylorYellandRoughness, u_star::Real;
                              ν::Real, g::Real = grav(), U10N = nothing, sea = nothing)
    s = _need(sea, :sea, r)
    return r.A * s.H_s * (s.H_s / s.λ_p)^r.B + smooth_flow_roughness(u_star, ν; β = r.β)
end

function sea_roughness_length(r::SeaStateRoughness, u_star::Real;
                              ν::Real, g::Real = grav(), U10N = nothing, sea = nothing)
    s = _need(sea, :sea, r)
    return r.D * s.H_s * (u_star / s.c_p)^r.B + smooth_flow_roughness(u_star, ν; β = r.β)
end

# =========================================================================
# 5. Scalar (heat and moisture) roughness
# =========================================================================

"""
    roughness_reynolds(z0, u_star, ν) -> R_r = z_0 u_*/ν

Roughness Reynolds number: ``R_r ≲ 0.11`` aerodynamically smooth,
``R_r ≳ 2`` rough (Fairall et al. 2003).
"""
roughness_reynolds(z0::Real, u_star::Real, ν::Real) = z0 * u_star / ν

"""
    AbstractScalarRoughness

A law for the heat and moisture roughness lengths of water in terms of
the momentum roughness; a subtype implements

    sea_scalar_roughness(r, z0m, u_star; ν) -> (z0h = …, z0q = …)
"""
abstract type AbstractScalarRoughness end

"""
    COARE30ScalarRoughness(; A = 5.5e-5, B = 0.6, z0_max = 1.1e-4)

COARE 3.0 fit to the COARE-plus and HEXOS humidity data (Fairall et al.
2003, eq. 28), a simplification of the Liu–Katsaros–Businger (1979)
surface-renewal relation: ``z_{0q} = \\min(z_{max}, A R_r^{-B})``, and
``z_{0t} = z_{0q}``. (The reference code uses ``z_{max} = 1.15×10^{-4}``;
the paper, followed here, ``1.1×10^{-4}``.)
"""
Base.@kwdef struct COARE30ScalarRoughness{T<:AbstractFloat} <: AbstractScalarRoughness
    A::T      = 5.5e-5
    B::T      = 0.6
    z0_max::T = 1.1e-4
end

"""
    COARE35ScalarRoughness(; A = 5.8e-5, B = 0.72, z0_max = 1.6e-4)

The COARE 3.5 refit (reference code `coare35vn`, chosen so that the
Stanton and Dalton numbers stay close to COARE 3.0 with the rougher
3.5 momentum roughness): ``z_{0q} = z_{0t} = \\min(z_{max}, A R_r^{-B})``.
"""
Base.@kwdef struct COARE35ScalarRoughness{T<:AbstractFloat} <: AbstractScalarRoughness
    A::T      = 5.8e-5
    B::T      = 0.72
    z0_max::T = 1.6e-4
end

"""
    BrutsaertGarrattRoughness(; a = 2.0, b = 2.28)

Surface-renewal scalar roughness of Brutsaert (1975) in Garratt's (1992)
form, as quoted by Fairall et al. (2003, eq. 29):
``z_{0q} = z_0 \\exp(a - b R_r^{1/4})``, applied here to heat as well
(``z_{0t} = z_{0q}``). It is the form used by Zeng et al. (1998) and
Beljaars (1995).
"""
Base.@kwdef struct BrutsaertGarrattRoughness{T<:AbstractFloat} <: AbstractScalarRoughness
    a::T = 2.0
    b::T = 2.28
end

"""
    sea_scalar_roughness(r::AbstractScalarRoughness, z0m, u_star; ν) -> (z0h, z0q)

Heat and moisture roughness lengths (m) for momentum roughness `z0m`,
friction velocity `u_star` and air kinematic viscosity `ν`.
"""
function sea_scalar_roughness(r::Union{COARE30ScalarRoughness,COARE35ScalarRoughness},
                              z0m::Real, u_star::Real; ν::Real)
    Rr = roughness_reynolds(z0m, u_star, ν)
    z = min(r.z0_max, r.A * Rr^(-r.B))
    return (z0h = z, z0q = z)
end

function sea_scalar_roughness(r::BrutsaertGarrattRoughness, z0m::Real, u_star::Real; ν::Real)
    Rr = roughness_reynolds(z0m, u_star, ν)
    z = z0m * exp(r.a - r.b * Rr^(1 / 4))
    return (z0h = z, z0q = z)
end

"""
    neutral_transfer_coefficients(roughness, scalar, U10N;
                                  z_ref = 10, ν = air_kinematic_viscosity(288.15, 101325),
                                  sea = nothing, c = DEFAULT_SURFACE_CONSTANTS)
        -> (C_D, C_H, C_E, u_star, z0m, z0h, z0q)

Neutral transfer coefficients at height `z_ref` for a neutral wind
`U10N` at that height (the usual ``C_{D10N}``, ``C_{H10N}``, ``C_{E10N}``
when `z_ref = 10`). Solves the neutral log law with a roughness that
depends on its own friction velocity,

    u_* = κ U / ln(z_ref / z_0(u_*)),

by a bracketed secant, then ``C_D = κ²/\\ln²(z/z_0)``,
``C_{H,E} = κ²/[\\ln(z/z_0) \\ln(z/z_{0t,q})]``.
"""
function neutral_transfer_coefficients(r::AbstractSeaRoughness, rs::AbstractScalarRoughness,
                                       U10N::Real; z_ref::Real = 10.0,
                                       ν::Real = air_kinematic_viscosity(288.15, 101325.0),
                                       sea = nothing,
                                       c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    U10N > 0 || throw(ArgumentError("neutral_transfer_coefficients: need U10N > 0"))
    κ, g = von_karman(c), grav(c)
    z0(u) = sea_roughness_length(r, u; ν = ν, g = g, U10N = U10N, sea = sea)
    # F(u) = u ln(z/z0(u)) - κU is negative as u → 0 (the smooth-flow z0
    # grows without bound). With a steep wave law (z0 ∝ u^4.5) it turns
    # negative again at large u, where z0 → z_ref; that upper root is an
    # unstable fixed point of u ← κU/ln(z/z0(u)). Scanning upward from
    # small u brackets the lower, physical root.
    F(u) = (z = z0(u); z >= z_ref ? -κ * U10N : u * log(z_ref / z) - κ * U10N)
    lo = 1e-6 * κ * U10N
    while F(lo) >= 0
        lo /= 10
    end
    hi, Fhi = lo, F(lo)
    while Fhi < 0
        lo = hi
        hi *= 2
        hi > 10 * U10N &&
            throw(ArgumentError("neutral_transfer_coefficients: no neutral solution (roughness too large)"))
        Fhi = F(hi)
    end
    u, _, _ = _illinois(F, lo, hi, F(lo), Fhi, 200)
    zm = z0(u)
    zs = sea_scalar_roughness(rs, zm, u; ν = ν)
    lm = log(z_ref / zm)
    return (C_D = (κ / lm)^2, C_H = κ^2 / (lm * log(z_ref / zs.z0h)),
            C_E = κ^2 / (lm * log(z_ref / zs.z0q)), u_star = u,
            z0m = zm, z0h = zs.z0h, z0q = zs.z0q)
end

# =========================================================================
# 6. Bulk algorithm
# =========================================================================

"""
    BulkAirSea(; roughness = COARE35Charnock(), scalar = COARE35ScalarRoughness(),
                 stability = COAREStability(:coare35), gustiness = ConvectiveGustiness(),
                 maxiter = 50, rtol = 1e-9)

A COARE-style bulk air–sea flux algorithm: which momentum roughness law,
scalar roughness law, stability functions and gustiness model to use.
Any `AbstractSeaRoughness`, `AbstractScalarRoughness` and
`AbstractStabilityFunctions` combine. [`coare30_bulk`](@ref) and
[`coare35_bulk`](@ref) give the published configurations; pass e.g.
`roughness = WaveAgeCharnock()` for the COARE 3.5 wave-age option.
"""
Base.@kwdef struct BulkAirSea{R<:AbstractSeaRoughness, S<:AbstractScalarRoughness,
                              F<:AbstractStabilityFunctions, G<:ConvectiveGustiness, T<:AbstractFloat}
    roughness::R = COARE35Charnock()
    scalar::S    = COARE35ScalarRoughness()
    stability::F = COAREStability(:coare35)
    gustiness::G = ConvectiveGustiness()
    maxiter::Int = 50
    rtol::T      = 1e-9
end

"""
    coare30_bulk(; kwargs...) -> BulkAirSea

COARE 3.0 (Fairall et al. 2003): wind-dependent Charnock 0.011 → 0.018,
the eq. (28) scalar roughness, COARE 3.0 stability, β = 1.2 gustiness.
Keywords override fields.
"""
coare30_bulk(; kwargs...) = BulkAirSea(; roughness = COARE30Charnock(),
                                       scalar = COARE30ScalarRoughness(),
                                       stability = COAREStability(:coare30), kwargs...)

"""
    coare35_bulk(; kwargs...) -> BulkAirSea

COARE 3.5 (Edson et al. 2013): Charnock ``0.0017 U_{10N} - 0.005``,
the 3.5 scalar-roughness refit, COARE 3.5 stability, β = 1.2
gustiness. Keywords override fields (e.g. `roughness = WaveAgeCharnock()`).
"""
coare35_bulk(; kwargs...) = BulkAirSea(; kwargs...)

"""
    AirSeaSolution{T}

Result of [`air_sea_fluxes`](@ref):

  * `surface` — the `SurfaceExchange.SurfaceLayerSolution` of the final
    Monin–Obukhov solve (``u_*``, ``θ_*``, ``q_*``, ``L``, ζ, and ``C_D``,
    ``C_H``, ``C_E`` referred to the gust-inclusive wind `surface.U`);
  * `z0m`, `z0h`, `z0q` — roughness lengths (m); `charnock` — the
    equivalent Charnock parameter; `Re_r` — roughness Reynolds number;
  * `w_g` — gustiness velocity (m s⁻¹); `B` — surface buoyancy flux
    ``(g/θ_v)\\overline{w'θ_v'}`` (m² s⁻³);
  * `U10N` — 10-m neutral wind relative to the mean wind (COARE's
    ``u_*/(κ G) \\ln(10/z_0)``, with gustiness factor ``G = S/U``);
  * `C_D10N`, `C_H10N`, `C_E10N` — 10-m neutral transfer coefficients;
  * `ν` — air kinematic viscosity used; `iterations`, `converged`.
"""
struct AirSeaSolution{T<:AbstractFloat}
    surface::SurfaceLayerSolution{T}
    z0m::T
    z0h::T
    z0q::T
    charnock::T
    Re_r::T
    w_g::T
    B::T
    U10N::T
    C_D10N::T
    C_H10N::T
    C_E10N::T
    ν::T
    iterations::Int
    converged::Bool
end

"""
    air_sea_fluxes(b::BulkAirSea = coare35_bulk();
                   U, z_m, z_h = z_m, θ_a, θ_s, q_a, q_s,
                   T_air = θ_a, p = 101325, sea = nothing, ν = nothing,
                   c = DEFAULT_SURFACE_CONSTANTS, U_min = 0.1) -> AirSeaSolution

Bulk air–sea fluxes, COARE style. Inputs as for
`SurfaceExchange.solve_surface_layer`: wind speed `U` relative to the
surface current at height `z_m`, potential temperature `θ_a` and
specific humidity `q_a` at `z_h`, and the **interface** values `θ_s`
(skin temperature: a cool-skin / warm-layer correction, if any, is
applied by the caller) and `q_s` (e.g. ``0.98\\,q_{sat}(T_s)`` over
seawater — see [`SeaSurface`](@ref)). `T_air` and `p` set the viscosity
unless `ν` is given; `sea` is the [`SeaState`](@ref) for wave laws.

Iteration: starting from COARE's first guess (``u_* = 0.035 U_{10}``,
``w_g = 0.5``), each pass

  1. updates ``z_0`` from ``u_*`` (and ``U_{10N}`` / `sea`) and
     ``z_{0t}, z_{0q}`` from ``R_r = z_0 u_*/ν``;
  2. solves Monin–Obukhov similarity exactly for that roughness and the
     effective wind ``S = \\sqrt{U² + w_g²}``
     (`SurfaceExchange.solve_surface_layer` with the stability functions
     of `b`);
  3. updates the gustiness from the buoyancy flux
     ``B = -(g/θ_{va}) u_* θ_{v*}`` and ``U_{10N}``,

until ``u_*`` and ``w_g`` change by less than `rtol` (relative). Because
step 2 is solved to convergence inside each pass, the outer loop only
has to converge the roughness–gustiness feedback, which contracts by a
factor ≈ ``2/\\ln(z/z_0)`` per pass for Charnock-type laws.
"""
function air_sea_fluxes(b::BulkAirSea = coare35_bulk();
                        U::Real, z_m::Real, z_h::Real = z_m,
                        θ_a::Real, θ_s::Real, q_a::Real = 0.0, q_s::Real = 0.0,
                        T_air::Real = θ_a, p::Real = 101325.0, sea = nothing,
                        ν = nothing, c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                        U_min::Real = 0.1)
    T = float(promote_type(typeof(U), typeof(z_m), typeof(θ_a), typeof(θ_s),
                           typeof(q_a), typeof(q_s)))
    κ, g = T(von_karman(c)), T(grav(c))
    νa = ν === nothing ? T(air_kinematic_viscosity(T_air, p; q = q_a, c = c)) : T(ν)
    ε = T(R_v(c.thermo) / R_d(c.thermo) - 1)
    θva = T(θ_a) * (1 + ε * T(q_a))

    # COARE first guess.
    w_g = T(b.gustiness.w_init)
    S0 = sqrt(T(U)^2 + w_g^2)
    U10 = S0 * log(10 / T(1e-4)) / log(z_m / T(1e-4))
    u_star, U10N = T(0.035) * U10, U10
    sol = nothing
    zm = zh = zq = zero(T)
    Bf = zero(T)
    w_used = w_g                     # the gustiness of the final solve
    iters, conv = 0, false
    for it in 1:b.maxiter
        iters = it
        zm = T(sea_roughness_length(b.roughness, u_star; ν = νa, g = g, U10N = U10N, sea = sea))
        zs = sea_scalar_roughness(b.scalar, zm, u_star; ν = νa)
        zh, zq = T(zs.z0h), T(zs.z0q)
        sol = solve_surface_layer(b.stability; U = U, z_m = z_m, z_h = z_h,
                                  θ_a = θ_a, θ_s = θ_s, q_a = q_a, q_s = q_s,
                                  z0m = zm, z0h = zh, z0q = zq, c = c,
                                  U_min = U_min, gustiness = w_g)
        w_used = w_g
        Bf = -g / θva * sol.u_star * sol.θv_star
        w_new = T(gustiness_velocity(b.gustiness, Bf))
        # 10-m neutral wind relative to the mean (not gust-inclusive) wind.
        U10N = sol.u_star / κ * log(10 / zm) * T(U) / sol.U
        du = abs(sol.u_star - u_star) / sol.u_star
        dw = abs(w_new - w_g) / max(w_new, w_g, sqrt(eps(T)))
        u_star, w_g = sol.u_star, w_new
        if du <= b.rtol && dw <= b.rtol
            conv = true
            break
        end
    end
    conv &= sol.converged
    α = T(g * (zm - smooth_flow_roughness(u_star, νa; β = b.roughness.β)) / u_star^2)
    lm = log(10 / zm)
    return AirSeaSolution{T}(sol, zm, zh, zq, α, roughness_reynolds(zm, u_star, νa),
                             w_used,
                             Bf, U10N, (κ / lm)^2, κ^2 / (lm * log(10 / zh)),
                             κ^2 / (lm * log(10 / zq)), νa, iters, conv)
end

# =========================================================================
# 7. The sea surface as an Era V AbstractSurface
# =========================================================================

"""
    SeaSurface(; T_skin = 293.15, bulk = coare35_bulk(), sea = nothing,
                 albedo = 0.055, emissivity = material_emissivity(:water),
                 salinity_factor = 0.98, U10N_init = 5.0)

Open water with wave-dependent roughness, as a
`SurfaceExchange.AbstractSurface`:

| method                          | value                                                   |
|:------------------------------- |:------------------------------------------------------- |
| `surface_temperature(s)`        | `T_skin` (K), the interface temperature                 |
| `roughness_lengths(s)`          | the roughness of the last flux computation (`z0m`, `z0h`, `z0q`) |
| `surface_albedo(s)`             | `albedo` (0.055, the COARE broadband value)             |
| `surface_emissivity(s)`         | `SurfaceOptics.material_emissivity(:water)` (0.96)      |
| `moisture_availability(s)`      | 1                                                       |
| `surface_specific_humidity(s,f)`| saturation over seawater, ``e = 0.98\\,e_{sat}(T_s)`` (Kraus & Businger 1994) |
| `turbulent_fluxes(s, f)`        | [`air_sea_fluxes`](@ref) with `s.bulk` and `s.sea`       |
| `step_surface!(s, f, dt)`       | updates the roughness state, returns the fluxes         |

`turbulent_fluxes` specialises the contract's default: the roughness is
solved *with* the fluxes (it depends on ``u_*``) rather than read from
the stored state, and gustiness is applied. The stress follows COARE,
``τ = ρ u_*² U/S``: gust-driven stress has no mean direction, so only
the fraction ``U/S`` of ``ρ u_*²`` is momentum taken from the mean wind.

`T_skin` is **prescribed**: `step_surface!` does not change it. The
prognostic ocean temperature (mixed layer, diurnal warm layer, cool
skin) belongs to `OceanMixedLayer`, which sets `s.T_skin`; the `G`
returned is the heat flux into the ocean it must absorb. `sea` (a
[`SeaState`](@ref)) is likewise set by whatever supplies the waves.
"""
mutable struct SeaSurface{B<:BulkAirSea, S, T<:AbstractFloat} <: AbstractSurface
    T_skin::T
    bulk::B
    sea::S
    albedo::T
    emissivity::T
    salinity_factor::T
    z0m::T
    z0h::T
    z0q::T
end

function SeaSurface(; T_skin::Real = 293.15, bulk::BulkAirSea = coare35_bulk(),
                    sea::Union{Nothing,SeaState} = nothing, albedo::Real = 0.055,
                    emissivity::Real = material_emissivity(:water),
                    salinity_factor::Real = 0.98, U10N_init::Real = 5.0)
    n = neutral_transfer_coefficients(bulk.roughness, bulk.scalar, U10N_init; sea = sea)
    return SeaSurface(float(T_skin), bulk, sea, float(albedo), float(emissivity),
                      float(salinity_factor), n.z0m, n.z0h, n.z0q)
end

surface_temperature(s::SeaSurface) = s.T_skin
roughness_lengths(s::SeaSurface) = (z0m = s.z0m, z0h = s.z0h, z0q = s.z0q)
surface_albedo(s::SeaSurface) = s.albedo
surface_emissivity(s::SeaSurface) = s.emissivity
moisture_availability(::SeaSurface) = 1.0

function surface_specific_humidity(s::SeaSurface, f::SurfaceForcing;
                                   c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    e = s.salinity_factor * saturation_vapor_pressure(s.T_skin, c.thermo)
    ε = molmass_ratio(c.thermo)
    return ε * e / (f.p - (1 - ε) * e)
end

"""
    sea_surface_solution(s::SeaSurface, f::SurfaceForcing; c, U_min = 0.1) -> AirSeaSolution

The full [`AirSeaSolution`](@ref) behind `turbulent_fluxes(s, f)`: the
air temperature is referred to the surface by the dry adiabat,
``θ_a = T_a + Γ_d z_{ref}``, ``θ_s = T_{skin}``, as in the contract's
default `turbulent_fluxes`.
"""
function sea_surface_solution(s::SeaSurface, f::SurfaceForcing;
                              sf::AbstractStabilityFunctions = s.bulk.stability,
                              c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                              U_min::Real = 0.1)
    pa = PhasePartition(f.q_air)
    θ_a = f.T_air + dry_adiabatic_lapse_rate(pa, c.thermo) * f.z_ref
    b = sf === s.bulk.stability ? s.bulk :
        BulkAirSea(s.bulk.roughness, s.bulk.scalar, sf, s.bulk.gustiness,
                   s.bulk.maxiter, s.bulk.rtol)
    return air_sea_fluxes(b; U = f.U, z_m = f.z_ref, θ_a = θ_a, θ_s = s.T_skin,
                          q_a = f.q_air, q_s = surface_specific_humidity(s, f; c = c),
                          T_air = f.T_air, p = f.p, sea = s.sea, c = c, U_min = U_min)
end

function _fluxes(s::SeaSurface, f::SurfaceForcing, r::AirSeaSolution, c::SurfaceConstants)
    pa = PhasePartition(f.q_air)
    ρ = air_density(f.T_air, f.p, pa, c.thermo)
    cp = heat_capacity_moist(pa, c.thermo)
    sl = r.surface
    H = -ρ * cp * sl.u_star * sl.θ_star
    E = -ρ * sl.u_star * sl.q_star
    LE = latent_heat_vapor(s.T_skin, c.thermo) * E
    Rn = net_radiation(s, f; c = c)
    τ = ρ * sl.u_star^2 * max(f.U, zero(f.U)) / sl.U
    return SurfaceFluxes(promote(Rn, H, LE, Rn - H - LE, τ, E)...)
end

"""
    turbulent_fluxes(s::SeaSurface, forcing; sf = s.bulk.stability,
                     c = DEFAULT_SURFACE_CONSTANTS, U_min = 0.1) -> SurfaceFluxes

Fluxes over open water from the bulk algorithm of `s` (see
[`SeaSurface`](@ref)); `G = Rn - H - LE` is the heat flux into the
ocean. Does not modify `s`.
"""
function turbulent_fluxes(s::SeaSurface, f::SurfaceForcing;
                          sf::AbstractStabilityFunctions = s.bulk.stability,
                          c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS, U_min::Real = 0.1)
    return _fluxes(s, f, sea_surface_solution(s, f; sf = sf, c = c, U_min = U_min), c)
end

"""
    step_surface!(s::SeaSurface, forcing, dt; c = DEFAULT_SURFACE_CONSTANTS) -> SurfaceFluxes

Computes the air–sea fluxes, stores the roughness lengths they imply (so
`roughness_lengths(s)` reports the current, wind- and wave-dependent
roughness to a tile aggregator), and returns the fluxes. The skin
temperature is prescribed and left unchanged (see [`SeaSurface`](@ref)).
"""
function step_surface!(s::SeaSurface, f::SurfaceForcing, dt::Real;
                       c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    r = sea_surface_solution(s, f; c = c)
    s.z0m, s.z0h, s.z0q = r.z0m, r.z0h, r.z0q
    return _fluxes(s, f, r, c)
end

end # module
