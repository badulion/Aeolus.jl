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
using ..SurfaceExchange: BusingerDyerLogLinear, SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, grav,
                         cp_water, von_karman, AbstractSurface, SurfaceForcing,
                         SurfaceFluxes, turbulent_fluxes, AbstractStabilityFunctions,
                         BusingerDyerBeljaarsHoltslag
import ..SurfaceExchange: surface_temperature, roughness_lengths, surface_albedo,
                          surface_emissivity, step_surface!, moisture_availability,
                          surface_specific_humidity, phi_m, phi_h, psi_m, psi_h,
                          column_heat_content
using ..SurfaceOptics: material_emissivity

export AbstractWaterEOS, UNESCOWaterEOS, LinearWaterEOS,
       water_density, water_thermal_expansion, water_haline_contraction,
       SolarBands, jerlov_water, soloviev_solar,
       solar_transmission, solar_absorbed_between, solar_mean_transmission,
       CoolSkin, cool_skin,
       AbstractWarmLayer, NoWarmLayer, FairallWarmLayer, ZengBeljaarsWarmLayer,
       step_warm_layer!, reset_warm_layer!, warm_layer_warming, warm_layer_depth,
       KrausTurnerMixing, WaterColumn, layer_depths,
       absorb_heat!, convective_adjustment!, wind_mixing!, mix_column!,
       column_heat_content, column_potential_energy, mixed_layer_depth,
       salinity_humidity_factor

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

# `BusingerDyerLogLinear` (the water-side φ_t of Zeng & Beljaars 2005)
# lives in SurfaceExchange with the other Monin–Obukhov families; the
# warm layer uses it from there.

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
    step_warm_layer!(wl, sw_net, q_out, τ, dt; α, ρ_w, c_w, g, κ = 0.4) -> ΔT

Advance warm layer `wl` by `dt` seconds under net shortwave `sw_net` into
the water, non-solar heat loss `q_out` (W m⁻², positive cooling) and
wind stress `τ` (N m⁻²), with water thermal expansion `α`, density
`ρ_w`, heat capacity `c_w`, gravity `g` and von Kármán `κ`. Returns the
warming across the layer. See [`AbstractWarmLayer`](@ref).
"""
function step_warm_layer! end

"Reset a warm layer to its no-warming state (the daily reset of the Fairall scheme)."
function reset_warm_layer! end

"""
    warm_layer_warming(wl) -> K
    warm_layer_warming(wl, z) -> K

Warming across the whole warm layer, ``T(0^-) - T(-D)``, or the part of
it between the surface and depth `z` (m) according to the scheme's
profile shape.
"""
function warm_layer_warming end

"Depth of the warm layer (m): `D` for the Fairall scheme, the fixed `d` for Zeng–Beljaars."
function warm_layer_depth end

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
``D = D_{max}``). Where the water's thermal expansion `α ≤ 0` (fresh
water below 3.98 °C) heating does not stabilise and the layer is reset;
the same holds for [`ZengBeljaarsWarmLayer`](@ref). The scheme is diagnostic in the accumulated integrals
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
warm_layer_warming(w::FairallWarmLayer) = w.ΔT
# linear profile: warming between the surface and z is ΔT min(z, D)/D
warm_layer_warming(w::FairallWarmLayer, z::Real) = w.D <= z ? w.ΔT : w.ΔT * z / w.D

function step_warm_layer!(w::FairallWarmLayer, sw_net::Real, q_out::Real, τ::Real, dt::Real;
                          α::Real, ρ_w::Real, c_w::Real, g::Real, κ::Real = 0.4)
    α > 0 || (reset_warm_layer!(w); return w.ΔT)    # heating does not stabilise
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
# similarity profile: warming between the surface and z is ΔT min(z/d, 1)^ν
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
    α > 0 || (w.ΔT = 0; return w.ΔT)                # heating does not stabilise
    u = sqrt(max(τ, 0) / ρ_w)
    A, B = zeng_beljaars_tendency(w, w.ΔT, sw_net, q_out, u; α = α, ρ_w = ρ_w,
                                  c_w = c_w, g = g, κ = κ)
    w.ΔT = max(zero(w.ΔT), (w.ΔT + dt * A) / (1 + dt * B))
    return w.ΔT
end

# =========================================================================
# 5. The water column
# =========================================================================

"""
    KrausTurnerMixing(; m = 0.5, n = 0.2, decay_depth = Inf)

Energetics of the bulk mixed layer (Kraus & Turner 1967; Niiler & Kraus
1977). Over a step `dt` the turbulence available to entrain water from
below is

    E = ρ₀ m u_{*w}³ dt e^{-h/h_d} + n ΔPE_conv          (J m⁻²)

— the wind-stirring work, `m` times the rate ``ρ_0 u_{*w}^3`` at which
the water-side friction velocity supplies turbulent energy (optionally
dissipated over `decay_depth` ``h_d``), plus the fraction `n` of the
potential energy released by convective overturning. The mixed layer is
then deepened (see [`wind_mixing!`](@ref)) until the potential-energy
increase of mixing equals `E`. With `n = 0` convection is
non-penetrative (encroachment).

The coefficients are tunable; published bulk-model values span
m ≈ 0.4–1.25 and n ≈ 0.15–0.2 (Niiler & Kraus 1977 and the models it
reviews). The defaults sit inside that range and are **not** tuned
against the tests here.
"""
Base.@kwdef struct KrausTurnerMixing{T<:AbstractFloat}
    m::T = 0.5
    n::T = 0.2
    decay_depth::T = Inf
end

"""
    WaterColumn(; dz = fill(1.0, 20), T = 288.15, S = 0.0,
                  eos = UNESCOWaterEOS(), solar = jerlov_water(:IB),
                  mixing = KrausTurnerMixing(), warm_layer = NoWarmLayer(),
                  cool_skin = CoolSkin(), albedo = 0.06,
                  emissivity = SurfaceOptics.material_emissivity(:water),
                  z0m = 1e-4, z0h = 1e-5, z0q = z0h,
                  bottom = :absorbing, bottom_flux = 0.0, local_time = 0.0,
                  c = DEFAULT_SURFACE_CONSTANTS,
                  ρ_ref = c.ρ_water, c_p = cp_water(c))

A lake or coastal-sea water column and an `AbstractSurface`. Layers of
thickness `dz` (m, top first; the top one can be a thick slab) carry
temperatures `T` (K; a scalar, a vector, or a function of centre depth)
at a constant salinity `S` (psu).

**Skin temperature** (`surface_temperature`):

    T_s = T₁ + ΔT_warm(dz₁/2) - ΔT_cool

— the top layer, plus the sub-grid warm layer's warming between the
surface and the top layer's centre ([`warm_layer_warming`](@ref)), minus
the cool skin. Use a warm-layer scheme when the top layer is thicker
than the warm layer (a slab of a few metres or more); with thin top
layers (≲ 0.5 m) keep `NoWarmLayer()`, since the column then resolves
the warm layer itself. `cool_skin = nothing` switches the cool skin off.

**One `step_surface!`**:

 1. fluxes from `SurfaceExchange.turbulent_fluxes` (or the `flux`
    keyword), iterated with the cool skin;
 2. [`absorb_heat!`](@ref): the non-solar flux ``G - SW_{net}`` into the
    top layer, the net shortwave distributed by `solar`, the shortwave
    left at the bottom absorbed by the bottom layer (`bottom = :absorbing`,
    a lake bed) or lost (`:transmitting`, deep water), and `bottom_flux`
    (W m⁻², positive downward) removed from the bottom layer;
 3. [`mix_column!`](@ref): convective adjustment, then wind and
    convective entrainment with the [`KrausTurnerMixing`](@ref) budget;
 4. the warm layer and the local clock (a `FairallWarmLayer` is reset at
    local midnight; set `local_time`, seconds after local midnight, to
    the start time).

Heat is conserved exactly: the column's heat content changes by
``(G - SW_{bottom} - F_{bottom}) dt`` per step, to round-off.

Not included (left to other modules or later work): ice formation below
the freezing point (`IceThermodynamics`, #37), wave-dependent roughness
and COARE fluxes (`AirSeaFlux`, #39; pass them through `flux`), the rain
heat flux, currents, shear instability at the mixed-layer base and
background diffusion below it.
"""
mutable struct WaterColumn{T<:AbstractFloat,E<:AbstractWaterEOS,P<:SolarBands,
                           W<:AbstractWarmLayer,CS,C<:SurfaceConstants} <: AbstractSurface
    dz::Vector{T}
    z_top::Vector{T}
    T::Vector{T}
    S::T
    eos::E
    solar::P
    mixing::KrausTurnerMixing{T}
    warm_layer::W
    cool_skin::CS
    albedo::T
    emissivity::T
    z0m::T
    z0h::T
    z0q::T
    ρ_ref::T
    c_p::T
    bottom::Symbol
    bottom_flux::T
    ΔT_cool::T
    δ_skin::T
    h_mix::T
    local_time::T
    c::C
end

function WaterColumn(; dz = fill(1.0, 20), T = 288.15, S::Real = 0.0,
                     eos::AbstractWaterEOS = UNESCOWaterEOS(),
                     solar::SolarBands = jerlov_water(:IB),
                     mixing::KrausTurnerMixing = KrausTurnerMixing(),
                     warm_layer::AbstractWarmLayer = NoWarmLayer(),
                     cool_skin::Union{Nothing,CoolSkin} = CoolSkin(),
                     albedo::Real = 0.06,
                     emissivity::Real = material_emissivity(:water),
                     z0m::Real = 1e-4, z0h::Real = 1e-5, z0q::Real = z0h,
                     bottom::Symbol = :absorbing, bottom_flux::Real = 0.0,
                     local_time::Real = 0.0,
                     c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                     ρ_ref::Real = c.ρ_water, c_p::Real = cp_water(c))
    dzv = collect(Float64, dz)
    (isempty(dzv) || any(<=(0), dzv)) && throw(ArgumentError("WaterColumn: dz must be non-empty and positive"))
    bottom in (:absorbing, :transmitting) ||
        throw(ArgumentError("WaterColumn: bottom must be :absorbing or :transmitting"))
    ztop = [0.0; cumsum(dzv)[1:end-1]]
    zc = ztop .+ dzv ./ 2
    Tv = T isa Real ? fill(Float64(T), length(dzv)) :
         T isa AbstractVector ? collect(Float64, T) : Float64[T(z) for z in zc]
    length(Tv) == length(dzv) || throw(ArgumentError("WaterColumn: T and dz lengths differ"))
    mix = KrausTurnerMixing{Float64}(mixing.m, mixing.n, mixing.decay_depth)
    return WaterColumn(dzv, ztop, Tv, Float64(S), eos, solar, mix, warm_layer, cool_skin,
                       Float64(albedo), Float64(emissivity), Float64(z0m), Float64(z0h),
                       Float64(z0q), Float64(ρ_ref), Float64(c_p), bottom,
                       Float64(bottom_flux), 0.0, 0.0, dzv[1], Float64(local_time), c)
end

"Centre depths of the layers (m, positive down)."
layer_depths(col::WaterColumn) = col.z_top .+ col.dz ./ 2

"""
    column_heat_content(col) -> J m⁻²

``ρ_{ref} c_p Σ_k T_k Δz_k`` (relative to 0 K) — the quantity
`step_surface!` conserves.
"""
column_heat_content(col::WaterColumn) = col.ρ_ref * col.c_p * sum(col.T .* col.dz)

"""
    column_potential_energy(col) -> J m⁻²

``Σ_k ρ(T_k, S) g z_k Δz_k`` with `z` the (negative) height of the layer
centre: mixing a stable column raises it.
"""
function column_potential_energy(col::WaterColumn)
    g = grav(col.c)
    zc = layer_depths(col)
    return -sum(water_density(col.eos, col.T[k], col.S) * g * zc[k] * col.dz[k]
                for k in eachindex(col.T))
end

"""
    mixed_layer_depth(col) -> m

Depth reached by the last mixing step: the homogeneous top layers plus
the entrained fraction of the partially entrained layer below.
"""
mixed_layer_depth(col::WaterColumn) = col.h_mix

# ---- AbstractSurface interface ------------------------------------------

surface_temperature(col::WaterColumn) =
    col.T[1] + warm_layer_warming(col.warm_layer, col.dz[1] / 2) - col.ΔT_cool
roughness_lengths(col::WaterColumn) = (z0m = col.z0m, z0h = col.z0h, z0q = col.z0q)
surface_albedo(col::WaterColumn) = col.albedo
surface_emissivity(col::WaterColumn) = col.emissivity
moisture_availability(::WaterColumn) = 1.0

"""
    salinity_humidity_factor(S) -> q_s / q_sat

Reduction of the saturation humidity over salt water, linear in salinity
through the standard 0.98 at S = 34 psu (Sverdrup et al. 1942; IFS
Cy43r1 eq. 8.161): ``1 - 0.02 S/34``. Exactly 1 for fresh water.
"""
salinity_humidity_factor(S::Real) = 1 - 0.02 * S / 34

"""
    surface_specific_humidity(col::WaterColumn, forcing; c)

Saturated over the skin, reduced for salinity:
``q_s = f(S)\\, q_{sat}(T_s, p)`` with [`salinity_humidity_factor`](@ref).
"""
surface_specific_humidity(col::WaterColumn, f::SurfaceForcing;
                          c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    salinity_humidity_factor(col.S) * saturation_specific_humidity(surface_temperature(col), f.p, c.thermo)

# ---- column physics -------------------------------------------------------

"""
    absorb_heat!(col, Q_ns, sw_net, dt) -> SW leaving through the bottom (W m⁻²)

Heat the column for `dt` seconds: the non-solar flux `Q_ns` (W m⁻²,
positive into the water) into the top layer, the net shortwave `sw_net`
layer by layer from `col.solar`, the shortwave reaching the bottom
absorbed by the bottom layer (`:absorbing`) or returned (`:transmitting`),
and `col.bottom_flux` removed from the bottom layer.
"""
function absorb_heat!(col::WaterColumn, Q_ns::Real, sw_net::Real, dt::Real)
    ρc = col.ρ_ref * col.c_p
    N = length(col.T)
    D = col.z_top[end] + col.dz[end]
    sw_bottom = sw_net * solar_transmission(col.solar, D)
    out = col.bottom === :absorbing ? zero(sw_bottom) : sw_bottom
    for k in 1:N
        heat = sw_net * solar_absorbed_between(col.solar, col.z_top[k], col.z_top[k] + col.dz[k])
        k == 1 && (heat += Q_ns)
        k == N && (heat += sw_bottom - out - col.bottom_flux)
        col.T[k] += heat * dt / (ρc * col.dz[k])
    end
    return out
end

# ΔPE (J m⁻²) of replacing the temperatures of layers ks by Tnew(k).
function _dpe(col::WaterColumn, ks, Tnew)
    g = grav(col.c)
    s = 0.0
    for k in ks
        zc = col.z_top[k] + col.dz[k] / 2
        s -= (water_density(col.eos, Tnew(k), col.S) - water_density(col.eos, col.T[k], col.S)) *
             g * zc * col.dz[k]
    end
    return s
end

"""
    convective_adjustment!(col) -> released potential energy (J m⁻², ≥ 0)

Remove static instability by mixing (heat-conserving, thickness-weighted)
every group of layers in which denser water lies above lighter water,
using the column's equation of state — so fresh water cooled below
3.98 °C stays at the surface. Returns the potential energy the overturn
released, the convective energy source of [`KrausTurnerMixing`](@ref).
"""
function convective_adjustment!(col::WaterColumn)
    ρ(T) = water_density(col.eos, T, col.S)
    groups = Vector{Tuple{Int,Int,Float64,Float64}}()     # (first, last, T, thickness)
    for k in eachindex(col.T)
        g = (k, k, col.T[k], col.dz[k])
        while !isempty(groups) && ρ(groups[end][3]) > ρ(g[3])
            a = pop!(groups)
            h = a[4] + g[4]
            g = (a[1], g[2], (a[4] * a[3] + g[4] * g[3]) / h, h)
        end
        push!(groups, g)
    end
    released = 0.0
    for (k1, k2, Tm, _) in groups
        k1 == k2 && continue
        released -= _dpe(col, k1:k2, _ -> Tm)
        col.T[k1:k2] .= Tm
    end
    return released
end

"""
    wind_mixing!(col, E) -> energy used (J m⁻²)

Deepen the mixed layer from the surface with turbulent energy `E`
(J m⁻²): layers are absorbed one by one while the potential-energy cost
of mixing them in is affordable; the first unaffordable layer is
entrained *partially* — a fraction `f` of it joins the mixed layer, the
rest keeps its temperature, and `f` is found by bisection so the cost
equals the energy left. Heat is conserved exactly and the column's
potential energy rises by the energy used. Sets `col.h_mix`.
"""
function wind_mixing!(col::WaterColumn, E::Real)
    N = length(col.T)
    Erem = float(E)
    Tml, h, k = col.T[1], col.dz[1], 1
    col.h_mix = h
    D = col.z_top[end] + col.dz[end]
    # round-off allowance: re-mixing an already uniform layer costs ~1 ulp
    tol = 1e-12 * max(abs(E), col.ρ_ref * grav(col.c) * D^2)
    while k < N
        Tb, db = col.T[k+1], col.dz[k+1]
        Tfull = Tb == Tml ? Tml : (h * Tml + db * Tb) / (h + db)
        cost = Tb == Tml ? 0.0 : _dpe(col, 1:k+1, _ -> Tfull)
        if cost <= Erem + tol
            Erem -= cost
            col.T[1:k+1] .= Tfull
            Tml, h, k = Tfull, h + db, k + 1
            col.h_mix = h
            continue
        end
        # partial entrainment of layer k+1
        Tmix(f) = (h * Tml + f * db * Tb) / (h + f * db)
        pcost(f) = (Tm = Tmix(f);
                    _dpe(col, 1:k, _ -> Tm) + _dpe(col, k+1:k+1, _ -> f * Tm + (1 - f) * Tb))
        lo, hi = 0.0, 1.0
        if Erem > 0
            for _ in 1:60
                mid = (lo + hi) / 2
                pcost(mid) <= Erem ? (lo = mid) : (hi = mid)
            end
        end
        f = lo
        if f > 0
            Erem -= pcost(f)
            Tm = Tmix(f)
            col.T[k+1] = f * Tm + (1 - f) * Tb
            col.T[1:k] .= Tm
        end
        col.h_mix = h + f * db
        break
    end
    return E - Erem
end

"""
    mix_column!(col, u_star_w, dt) -> (released, used)

Convective adjustment followed by entrainment with the
[`KrausTurnerMixing`](@ref) energy
``E = ρ_{ref} m u_{*w}^3 dt\\, e^{-h/h_d} + n\\, ΔPE_{conv}``. Returns the
convectively released and the entrainment-used energies (J m⁻²).
"""
function mix_column!(col::WaterColumn, u_star_w::Real, dt::Real)
    released = convective_adjustment!(col)
    mx = col.mixing
    decay = isfinite(mx.decay_depth) ? exp(-col.h_mix / mx.decay_depth) : 1.0
    E = col.ρ_ref * mx.m * u_star_w^3 * dt * decay + mx.n * released
    used = wind_mixing!(col, E)
    return (released = released, used = used)
end

function _update_cool_skin!(col::WaterColumn, fl::SurfaceFluxes, sw_net)
    cs = col.cool_skin
    cs === nothing && return col
    T1, S, c = col.T[1], col.S, col.c
    Q_out = -(fl.Rn - sw_net) + fl.H + fl.LE
    u_w = sqrt(max(fl.τ, 0) / col.ρ_ref)
    r = cool_skin(cs, Q_out, sw_net, fl.LE, u_w;
                  α = water_thermal_expansion(col.eos, T1, S),
                  βS = water_haline_contraction(col.eos, T1, S) * S,
                  ρ_w = col.ρ_ref, c_w = col.c_p, k_w = c.k_water, g = grav(c),
                  L_v = latent_heat_vapor(surface_temperature(col), c.thermo),
                  δ0 = col.δ_skin > 0 ? col.δ_skin : 1e-3)
    col.ΔT_cool, col.δ_skin = r.ΔT, r.δ
    return col
end

"""
    step_surface!(col::WaterColumn, forcing, dt; flux = nothing,
                  skin_iterations = 2, kwargs...) -> SurfaceFluxes

Advance the water column by `dt` seconds (see [`WaterColumn`](@ref) for
the sequence). `flux(col, forcing) -> SurfaceFluxes` replaces the default
`turbulent_fluxes(col, forcing; c = col.c, kwargs...)` — the hook for
COARE-type air–sea fluxes; whatever it returns is exactly what the
column absorbs (``G = R_n - H - LE``). The cool skin and the fluxes are
iterated `skin_iterations` times so the returned fluxes are those of
the final skin temperature.
"""
function step_surface!(col::WaterColumn, f::SurfaceForcing, dt::Real;
                       flux = nothing, skin_iterations::Integer = 2, kwargs...)
    getflux() = flux === nothing ? turbulent_fluxes(col, f; c = col.c, kwargs...) : flux(col, f)
    sw_net = (1 - surface_albedo(col)) * f.sw_down
    fl = getflux()
    if col.cool_skin !== nothing
        for _ in 1:skin_iterations
            _update_cool_skin!(col, fl, sw_net)
            fl = getflux()
        end
    end
    Q_ns = fl.G - sw_net
    absorb_heat!(col, Q_ns, sw_net, dt)
    u_w = sqrt(max(fl.τ, 0) / col.ρ_ref)
    mix_column!(col, u_w, dt)
    # warm layer, with a daily reset for the accumulating scheme
    t_new = col.local_time + dt
    if col.warm_layer isa FairallWarmLayer && fld(t_new, 86400) > fld(col.local_time, 86400)
        reset_warm_layer!(col.warm_layer)
    end
    col.local_time = mod(t_new, 86400.0)
    if !(col.warm_layer isa NoWarmLayer)
        step_warm_layer!(col.warm_layer, sw_net, -Q_ns, fl.τ, dt;
                         α = water_thermal_expansion(col.eos, col.T[1], col.S),
                         ρ_w = col.ρ_ref, c_w = col.c_p, g = grav(col.c), κ = von_karman(col.c))
    end
    return fl
end

end # module
