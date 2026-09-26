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
       solar_transmission, solar_absorbed_between, solar_mean_transmission,
       CoolSkin, cool_skin,
       BusingerDyerLogLinear,
       AbstractWarmLayer, NoWarmLayer, FairallWarmLayer, ZengBeljaarsWarmLayer,
       step_warm_layer!, reset_warm_layer!, warm_layer_warming, warm_layer_depth

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

# =========================================================================
# 3. Cool skin
# =========================================================================

"""
    CoolSkin(; ν_w = 1e-6, λ0 = 6.0, δ_max = 0.01,
               f_s = (0.065, 11.0, 6.6e-5, 8e-4), maxiter = 50)

Parameters of the Fairall et al. (1996a) cool skin: kinematic viscosity
of water `ν_w` (m² s⁻¹), Saunders' constant `λ0`, the cap `δ_max` (m) on
the sublayer thickness when the skin is not convectively unstable, and
the coefficients of the fraction of net shortwave absorbed in the skin,

    f_s(δ) = a + b δ - (c/δ) (1 - e^{-δ/d})

(Fairall et al. 1996a eq. 16 with the COARE 3.0+ value a = 0.065 of
Fairall et al. 2003, replacing the original 0.137).
"""
Base.@kwdef struct CoolSkin{T<:AbstractFloat}
    ν_w::T = 1e-6
    λ0::T = 6.0
    δ_max::T = 0.01
    f_s::NTuple{4,T} = (0.065, 11.0, 6.6e-5, 8e-4)
    maxiter::Int = 50
end

_skin_solar_fraction(cs::CoolSkin, δ) =
    cs.f_s[1] + cs.f_s[2] * δ - cs.f_s[3] / δ * (-expm1(-δ / cs.f_s[4]))

"""
    cool_skin(cs::CoolSkin, Q_out, sw_net, LE, u_star_w;
              α, βS = 0, ρ_w, c_w, k_w, g, L_v, δ0 = 1e-3)
        -> (ΔT, δ, f_s)

Cool-skin temperature depression ``ΔT = T_{-δ} - T_{skin}`` (K,
positive when the skin is cooler than the water just below it) and the
conductive sublayer thickness `δ` (m) (Saunders 1967; Fairall et al.
1996a):

    q_c = Q_out - f_s(δ) SW_net,          ΔT = q_c δ / k_w
    δ   = λ ν_w / u_{*w},  λ = λ0 [1 + (16 g c_w ρ_w ν_w³ B / (k_w² u_{*w}⁴))^{3/4}]^{-1/3}

with the water-side buoyancy flux (in heat-flux units)
``B = α q_c + βS c_w LE / L_v`` — the second term is the salt left
behind by evaporation, with `βS` = haline contraction × salinity (0 for
a lake). If ``B ≤ 0`` the skin is not convective and
``δ = \\min(δ_{max}, λ_0 ν_w / u_{*w})``. The fixed point in δ (f_s
depends on δ) is iterated to machine precision.

Inputs (W m⁻², positive as named): `Q_out` the non-solar heat loss
``LW_{up,net} + H + LE``, `sw_net` the net shortwave into the water,
`LE` the latent heat flux, `u_star_w` the water-side friction velocity
``u_* \\sqrt{ρ_a/ρ_w}``.
"""
function cool_skin(cs::CoolSkin, Q_out::Real, sw_net::Real, LE::Real, u_star_w::Real;
                   α::Real, βS::Real = 0.0, ρ_w::Real, c_w::Real, k_w::Real,
                   g::Real, L_v::Real, δ0::Real = 1e-3)
    T = float(promote_type(typeof(Q_out), typeof(sw_net), typeof(LE), typeof(u_star_w)))
    u = max(T(u_star_w), T(1e-6))
    ν = T(cs.ν_w)
    bigc = 16 * T(g) * T(c_w) * T(ρ_w) * ν^3 / T(k_w)^2
    δ = T(δ0)
    fs = _skin_solar_fraction(cs, δ)
    qc = T(Q_out) - fs * T(sw_net)
    for _ in 1:cs.maxiter
        fs = _skin_solar_fraction(cs, δ)
        qc = T(Q_out) - fs * T(sw_net)
        B = T(α) * qc + T(βS) * T(c_w) * T(LE) / T(L_v)
        δnew = if B > 0
            λ = cs.λ0 / (1 + (bigc * B / u^4)^(3 / 4))^(1 / 3)
            λ * ν / u
        else
            min(T(cs.δ_max), cs.λ0 * ν / u)
        end
        conv = abs(δnew - δ) <= 8 * eps(T) * δnew
        δ = δnew
        conv && break
    end
    fs = _skin_solar_fraction(cs, δ)
    qc = T(Q_out) - fs * T(sw_net)
    return (ΔT = qc * δ / T(k_w), δ = δ, f_s = fs)
end

# =========================================================================
# 4. Diurnal warm layer
# =========================================================================

"""
    BusingerDyerLogLinear(; γ_m = 16, γ_h = 16, β = 5)

Monin–Obukhov functions with the classical *log-linear* stable branch,
``φ_m = φ_h = 1 + βζ`` and ``ψ = -βζ`` for ``ζ ≥ 0`` (Businger et al.
1971; Dyer 1974). The unstable branch is delegated to
`SurfaceExchange.BusingerDyerBeljaarsHoltslag` with the same γ, so the
two families agree exactly for ``ζ < 0``. This is the water-side
``φ_t`` of Zeng & Beljaars (2005) (IFS Cy43r1 eq. 8.158).
"""
Base.@kwdef struct BusingerDyerLogLinear{T<:AbstractFloat} <: AbstractStabilityFunctions
    γ_m::T = 16.0
    γ_h::T = 16.0
    β::T = 5.0
end
_unstable(sf::BusingerDyerLogLinear) = BusingerDyerBeljaarsHoltslag(γ_m = sf.γ_m, γ_h = sf.γ_h)
phi_m(sf::BusingerDyerLogLinear, ζ::Real) = ζ < 0 ? phi_m(_unstable(sf), ζ) : 1 + sf.β * ζ
phi_h(sf::BusingerDyerLogLinear, ζ::Real) = ζ < 0 ? phi_h(_unstable(sf), ζ) : 1 + sf.β * ζ
psi_m(sf::BusingerDyerLogLinear, ζ::Real) = ζ < 0 ? psi_m(_unstable(sf), ζ) : -sf.β * ζ
psi_h(sf::BusingerDyerLogLinear, ζ::Real) = ζ < 0 ? psi_h(_unstable(sf), ζ) : -sf.β * ζ

"""
    AbstractWarmLayer

A diurnal warm-layer scheme: the stably stratified, solar-heated layer a
few metres deep that forms under weak wind. Implementations carry their
own state and provide

  * `step_warm_layer!(wl, sw_net, q_out, τ, dt; α, ρ_w, c_w, g, κ)` —
    advance by `dt` (s) under net shortwave `sw_net` into the water,
    non-solar heat loss `q_out` (``LW_{up,net} + H + LE``, W m⁻², positive
    cooling the water) and wind stress `τ` (N m⁻²); `α` is the thermal
    expansion of the water;
  * `warm_layer_warming(wl)` — warming across the whole layer,
    ``T(0^-) - T(-D)`` (K), and `warm_layer_warming(wl, z)` the part
    between the surface and depth `z` (m);
  * `warm_layer_depth(wl)` (m) and `reset_warm_layer!(wl)`.
"""
abstract type AbstractWarmLayer end

"""
    NoWarmLayer()

No sub-grid warm layer: the top layer of the column is the bulk
temperature right up to the skin. The right choice when the column's
top layer is thin enough (≲ 0.5 m) to resolve the warm layer itself.
"""
struct NoWarmLayer <: AbstractWarmLayer end
step_warm_layer!(::NoWarmLayer, args...; kwargs...) = 0.0
warm_layer_warming(::NoWarmLayer, z::Real = 0.0) = 0.0
warm_layer_depth(::NoWarmLayer) = 0.0
reset_warm_layer!(w::NoWarmLayer) = w

"""
    FairallWarmLayer(; Ri_c = 0.65, D_max = 19, q_onset = 50, τ_min = 0.002,
                       solar = soloviev_solar(:coare))

The diurnal warm layer of Fairall et al. (1996b), exactly as coded in
COARE 3.x (`coare36vnWarm_et`). From the moment the absorbed heat
``q = f_x SW_{net} - q_{out}`` first exceeds `q_onset` (W m⁻²) in a day,
the heat and momentum put into the layer are accumulated,

    Q_ac = ∫ q dt,    I_ac = ∫ max(τ, τ_min) dt,

and a layer with a linear temperature and velocity profile, marginally
stable at critical Richardson number `Ri_c`, has depth and warming

    D  = I_ac [2 Ri_c c_w / (α g ρ_w Q_ac)]^{1/2}  (≤ D_max),
    ΔT = 2 Q_ac / (ρ_w c_w D).

The absorbed solar fraction ``f_x = 1 - (1/D)∫_0^D I(z)/I_0 dz`` is
iterated five times with D, as in the reference code. If the
accumulated heat turns negative the layer is wiped out (``f_x = 0.75``,
``D = D_{max}``). The scheme is diagnostic in the accumulated integrals
and assumes a *daily* reset: call [`reset_warm_layer!`](@ref) at local
midnight (the [`WaterColumn`](@ref) does this from its `local_time`).
"""
Base.@kwdef mutable struct FairallWarmLayer{T<:AbstractFloat,P<:SolarBands} <: AbstractWarmLayer
    Ri_c::T = 0.65
    D_max::T = 19.0
    q_onset::T = 50.0
    τ_min::T = 0.002
    solar::P = soloviev_solar(:coare)
    # state
    Q_ac::T = 0.0        # accumulated heat, J m⁻²
    I_ac::T = 0.0        # accumulated momentum, N s m⁻²
    D::T = 19.0          # depth, m
    fx::T = 0.5          # absorbed solar fraction
    active::Bool = false # onset threshold crossed today
    ΔT::T = 0.0          # warming across the layer, K
end

function reset_warm_layer!(w::FairallWarmLayer)
    w.Q_ac = 0; w.I_ac = 0; w.D = w.D_max; w.fx = 0.5; w.active = false; w.ΔT = 0
    return w
end
warm_layer_depth(w::FairallWarmLayer) = w.D
"Warming ``T(0) - T(-z)`` of a linear profile: `ΔT min(z, D)/D`."
warm_layer_warming(w::FairallWarmLayer) = w.ΔT
warm_layer_warming(w::FairallWarmLayer, z::Real) = w.D <= z ? w.ΔT : w.ΔT * z / w.D

function step_warm_layer!(w::FairallWarmLayer, sw_net::Real, q_out::Real, τ::Real, dt::Real;
                          α::Real, ρ_w::Real, c_w::Real, g::Real, κ::Real = 0.4)
    ctd1 = sqrt(2 * w.Ri_c * c_w / (α * g * ρ_w))
    ctd2 = sqrt(2 * α * g / (w.Ri_c * ρ_w)) / c_w^1.5
    q = w.fx * sw_net - q_out
    (q >= w.q_onset || w.active) || return w.ΔT
    w.active = true
    w.I_ac += max(w.τ_min, τ) * dt
    if w.Q_ac + q * dt > 0
        qj = zero(w.Q_ac)
        for _ in 1:5
            w.fx = 1 - solar_mean_transmission(w.solar, w.D)
            qj = (w.fx * sw_net - q_out) * dt
            if w.Q_ac + qj > 0
                w.D = min(w.D_max, ctd1 * w.I_ac / sqrt(w.Q_ac + qj))
            end
        end
    else                                     # warm layer wiped out
        w.fx = 0.75
        w.D = w.D_max
        qj = (w.fx * sw_net - q_out) * dt
    end
    w.Q_ac += qj
    w.ΔT = w.Q_ac > 0 ? ctd2 * w.Q_ac^1.5 / w.I_ac : zero(w.ΔT)
    return w.ΔT
end

"""
    ZengBeljaarsWarmLayer(; d = 3, ν = 0.3, solar = soloviev_solar(:zeng_beljaars),
                            stability = BusingerDyerLogLinear(), ΔT = 0)

The prognostic warm layer of Zeng & Beljaars (2005), as in the IFS
(Cy43r1 Part IV §8.10.2). With the similarity profile
``T(z) - T(-d) = ΔT [1 - (-z/d)^ν]``, the warming ``ΔT = T_{-δ} - T_{-d}``
obeys

    ∂ΔT/∂t = (Q + R_s - R(-d)) / (d ρ_w c_w ν/(ν+1)) - (ν+1) κ u_{*w} ΔT / (d φ_t(d/L))

with ``Q = -q_{out}`` the net non-solar flux into the water, ``R_s -
R(-d)`` the shortwave absorbed above depth `d`, water-side friction
velocity ``u_{*w} = \\sqrt{τ/ρ_w}``, ``L = ρ_w c_w u_{*w}^3/(κ F_d)`` and
buoyancy flux

    F_d = g α (Q + R_s - R(-d))                            if ΔT ≤ 0
    F_d = ρ_w c_w u_{*w}² (ν g α ΔT / (5 d))^{1/2}          if ΔT > 0.

Integrated with the damping term implicit (unconditionally stable).
Being prognostic, it needs no daily reset and a residual layer can
survive a calm night. ΔT is kept ≥ 0: a layer colder than its base is
convectively unstable and is mixed away.

At steady state under constant forcing the neutral (φ_t = 1) limit is
``ΔT = (Q + R_s - R(-d)) / (ρ_w c_w ν κ u_{*w})``.
"""
Base.@kwdef mutable struct ZengBeljaarsWarmLayer{T<:AbstractFloat,P<:SolarBands,
                                                 F<:AbstractStabilityFunctions} <: AbstractWarmLayer
    d::T = 3.0
    ν::T = 0.3
    solar::P = soloviev_solar(:zeng_beljaars)
    stability::F = BusingerDyerLogLinear()
    ΔT::T = 0.0
end

reset_warm_layer!(w::ZengBeljaarsWarmLayer) = (w.ΔT = 0; w)
warm_layer_depth(w::ZengBeljaarsWarmLayer) = w.d
warm_layer_warming(w::ZengBeljaarsWarmLayer) = w.ΔT
"Warming between the surface and depth `z`: ``ΔT \\min(z/d, 1)^ν``."
warm_layer_warming(w::ZengBeljaarsWarmLayer, z::Real) = w.ΔT * min(max(z, 0) / w.d, 1)^w.ν

"""
    OceanMixedLayer.zeng_beljaars_tendency(w, ΔT, sw_net, q_out, u_star_w; α, ρ_w, c_w, g, κ)
        -> (source, damping)

The two terms of the Zeng–Beljaars equation, ``∂ΔT/∂t = source -
damping × ΔT``, with φ_t evaluated at the given ΔT.
"""
function zeng_beljaars_tendency(w::ZengBeljaarsWarmLayer, ΔT::Real, sw_net::Real, q_out::Real,
                                u_star_w::Real; α::Real, ρ_w::Real, c_w::Real, g::Real,
                                κ::Real = 0.4)
    d, ν = w.d, w.ν
    u = max(u_star_w, 1e-5)
    Qd = -q_out + sw_net * (1 - solar_transmission(w.solar, d))
    Fd = ΔT > 0 ? ρ_w * c_w * u^2 * sqrt(ν * g * α * ΔT / (5 * d)) : g * α * Qd
    ζ = d * κ * Fd / (ρ_w * c_w * u^3)          # d/L
    φ = phi_h(w.stability, ζ)
    return Qd / (d * ρ_w * c_w * ν / (ν + 1)), (ν + 1) * κ * u / (d * φ)
end

function step_warm_layer!(w::ZengBeljaarsWarmLayer, sw_net::Real, q_out::Real, τ::Real, dt::Real;
                          α::Real, ρ_w::Real, c_w::Real, g::Real, κ::Real = 0.4)
    u = sqrt(max(τ, 0) / ρ_w)
    A, B = zeng_beljaars_tendency(w, w.ΔT, sw_net, q_out, u; α = α, ρ_w = ρ_w,
                                  c_w = c_w, g = g, κ = κ)
    w.ΔT = max(zero(w.ΔT), (w.ΔT + dt * A) / (1 + dt * B))
    return w.ΔT
end

end # module
