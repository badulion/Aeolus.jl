"""
    IceThermodynamics

Lake- and sea-ice thermodynamics for Era V (Phase 51, issue #37): the
energy-conserving multi-layer model of Bitz & Lipscomb (1999, "BL99")
with brine pockets, an optional snow layer, surface and basal melting,
basal accretion, sublimation, snow-ice formation and an ocean (or lake)
heat flux, exposed as an [`AbstractSurface`](@ref) so the tile framework
can aggregate it.

# The physics in one paragraph

Sea ice is a mushy mixture of fresh ice and brine pockets held at the
liquidus, ``T_m = -μS`` (Assur 1958). Warming it melts the pocket walls,
so its heat capacity ``c(T,S) = c_0 + L μ S / T^2`` (Untersteiner 1961,
Ono 1967) diverges near 0 °C, and the energy needed to *finish* melting
a unit volume — the *energy of melting* (BL99 eq. 5)

    q(S,T) = ρ_i [ c_0 (T_m - T) + L (1 - T_m/T) ]

— goes to zero at ``T = T_m``. BL99's point is that a model which melts
the surface with ``q = ρ_i L`` while also letting brine pockets melt
internally counts the same energy twice. Here the prognostic variable of
every layer is its enthalpy ``E = -q`` (J m⁻³, zero for meltwater at
``T_m``), the heat equation is solved implicitly for temperature and the
enthalpies are then updated *from the interface fluxes*, so the column
conserves energy to round-off by construction; growth and melt consume
exactly ``-E`` per unit volume.

# Ownership (Era V "one source of truth")

  * ``ρ_i``, ``k_i`` (fresh-ice conductivity ``k_0``), ``c_0 = c_{p,ice}``
    and ``L = L_f`` come from `SurfaceExchange.SurfaceConstants` — never
    restated here. The BL99 benchmark overrides ``k_i`` through a
    `SurfaceConstants(k_ice = 2.034)` instance, not a local constant.
  * Saturation humidity over ice and the latent heat of sublimation come
    from `MoistThermodynamics`; the turbulent fluxes from
    `SurfaceExchange.turbulent_fluxes` (Monin–Obukhov).
  * The error function used by the Neumann solution is
    `GasOptics.erf_series`.
  * This module *owns* the sea-water freezing point
    [`seawater_freezing_point`](@ref) and the sea-ice material functions;
    other Era V modules (e.g. the ocean mixed layer) should delegate.

# References

  * Bitz, C. M. & Lipscomb, W. H. (1999). An energy-conserving
    thermodynamic model of sea ice. *J. Geophys. Res.*, 104(C7),
    15669-15677.
  * Maykut, G. A. & Untersteiner, N. (1971). Some results from a
    time-dependent thermodynamic model of sea ice. *J. Geophys. Res.*,
    76, 1550-1575.
  * Semtner, A. J. (1976). A model for the thermodynamic growth of sea
    ice in numerical investigations of climate. *J. Phys. Oceanogr.*, 6,
    379-389.
  * Untersteiner, N. (1964). Calculations of temperature regime and heat
    budget of sea ice in the Central Arctic. *J. Geophys. Res.*, 69,
    4755-4766.
  * Assur, A. (1958). Composition of sea ice and its tensile strength.
    *Arctic Sea Ice*, NAS-NRC Publ. 598, 106-138.
  * Stefan, J. (1891). Über die Theorie der Eisbildung, insbesondere über
    die Eisbildung im Polarmeere. *Ann. Phys.*, 278, 269-286.
  * Carslaw, H. S. & Jaeger, J. C. (1959). *Conduction of Heat in Solids*,
    2nd ed., §11.2 (the Neumann solution). Oxford.
"""
module IceThermodynamics

using ..MoistThermodynamics: saturation_specific_humidity, latent_heat_sublim,
                             latent_heat_vapor
using ..SurfaceExchange
using ..SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, LH_f0,
                         cp_ice, SurfaceForcing, SurfaceFluxes, turbulent_fluxes
import ..SurfaceExchange: surface_temperature, roughness_lengths, surface_albedo,
                          surface_emissivity, step_surface!, moisture_availability,
                          surface_specific_humidity, surface_latent_heat
using ..GasOptics: erf_series

export IceParameters, seawater_freezing_point, bl99_salinity_profile,
       ice_melting_temperature, ice_energy_of_melting, ice_enthalpy,
       ice_temperature_from_enthalpy, ice_heat_capacity, ice_conductivity,
       snow_enthalpy, stefan_ice_thickness, neumann_lambda, neumann_ice_thickness

# =========================================================================
# 1. Parameters
# =========================================================================

"""
    IceParameters(; kwargs...)

Parameters of the ice/snow column (SI units; temperatures in the BL99
formulas are in °C). Defaults are the BL99 standard case unless noted.

| field          | default | meaning |
|:-------------- |:------- |:------- |
| `μ`            | 0.054   | liquidus slope, ``T_m = -μS`` (K ppt⁻¹; Assur 1958) |
| `β`            | 0.1172  | brine conductivity term, ``k = k_0 + βS/T`` (W m⁻¹; Untersteiner 1964) |
| `k_min`        | 0.1     | floor on sea-ice conductivity (W m⁻¹ K⁻¹; BL99 code) |
| `T_k_cap`      | -0.1    | °C; ``T`` in ``βS/T`` is capped here to keep ``k`` finite near 0 °C |
| `ρ_snow`       | 330     | snow density (kg m⁻³; MU71) |
| `k_snow`       | 0.31    | snow conductivity (W m⁻¹ K⁻¹; MU71) |
| `κ_i`          | 1.5     | shortwave extinction in ice (m⁻¹; Untersteiner 1961) |
| `i0`           | 0.3     | fraction of net SW penetrating snow-free ice (Grenfell & Maykut 1977) |
| `i0_snow_scale`| 0.0     | m; if > 0, SW also penetrates snow with fraction ``i_0 d/(h_s + d)`` (Bettge et al. 1996 use d = 0.1 m; BL99 default); 0 = snow-free ice only (Semtner 1976; BL99 §4.4 and code) |
| `T_melt_ice`   | 0.0     | °C; melting temperature of a snow-free ice *surface* (0 for the flushed, fresh top of multiyear ice; BL99 use -0.1 for isosaline ice) |
| `α_ice`        | 0.63    | bare-ice albedo (BL99) |
| `α_snow`       | 0.80    | dry-snow albedo (BL99) |
| `α_snow_melt`  | 0.75    | melting-snow albedo (BL99) |
| `α_water`      | 0.06    | open-water albedo when the ice has melted away |
| `ε`            | 0.99    | longwave emissivity of snow/ice (BL99 used 1.0) |
| `h_snow_min`   | 0.01    | m; thinner snow is thermally inert (not in the heat equation) |
| `h_snow_bare`  | 5e-6    | m; thinner snow counts as snow-free for albedo/``i_0`` purposes |
| `h_ice_min`    | 1e-4    | m; thinner ice is melted into the water below |
| `z0m`, `z0h`   | 5e-4, 5e-5 | roughness lengths of snow/ice (m) |
| `snow_ice`     | true    | convert snow pushed below the waterline into ice |
| `constants`    | `DEFAULT_SURFACE_CONSTANTS` | ``ρ_i``, ``k_0``, ``c_0``, ``L``, σ |

Snow is fresh: its heat capacity and latent heat are those of ice
(BL99 §3.1). A prognostic snowpack (compaction, metamorphism, albedo
ageing) is `Snowpack`'s scope (issues #34-#36); here snow is the thermal
blanket of the ice.
"""
Base.@kwdef struct IceParameters{T<:AbstractFloat, C<:SurfaceConstants}
    μ::T           = 0.054
    β::T           = 0.1172
    k_min::T       = 0.1
    T_k_cap::T     = -0.1
    ρ_snow::T      = 330.0
    k_snow::T      = 0.31
    κ_i::T         = 1.5
    i0::T          = 0.3
    i0_snow_scale::T = 0.0
    T_melt_ice::T  = 0.0
    α_ice::T       = 0.63
    α_snow::T      = 0.80
    α_snow_melt::T = 0.75
    α_water::T     = 0.06
    ε::T           = 0.99
    h_snow_min::T  = 0.01
    h_snow_bare::T = 5e-6
    h_ice_min::T   = 1e-4
    z0m::T         = 5e-4
    z0h::T         = 5e-5
    snow_ice::Bool = true
    constants::C   = DEFAULT_SURFACE_CONSTANTS
end

# Short accessors to the contract constants.
_ρi(p::IceParameters) = p.constants.ρ_ice
_k0(p::IceParameters) = p.constants.k_ice
_c0(p::IceParameters) = cp_ice(p.constants)
_L(p::IceParameters)  = LH_f0(p.constants)
_T0(p::IceParameters) = p.constants.thermo.T_freeze        # 273.15 K

# =========================================================================
# 2. Material functions of sea ice (BL99 §2)
# =========================================================================

"""
    seawater_freezing_point(S; μ = 0.054, c = DEFAULT_SURFACE_CONSTANTS) -> K

Freezing temperature of water of salinity `S` (ppt), the linear liquidus
``T_f = T_0 - μS`` (Assur 1958) used by BL99 and CICE. `S = 0` gives the
fresh-water (lake) freezing point ``T_0`` = 273.15 K; ``S = 34`` gives
-1.84 °C.
"""
seawater_freezing_point(S::Real; μ::Real = 0.054,
                        c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    c.thermo.T_freeze - μ * S

"""
    bl99_salinity_profile(N; S_max = 3.2) -> Vector

Time-independent multiyear salinity profile of MU71/BL99 (eq. 12, a fit
to Schwarzacher 1959) at the midpoints of `N` equal layers,

    S(z) = (S_max/2) [1 - cos(π z^{a/(z+b)})],  a = 0.407, b = 0.573,

with ``z`` the relative depth (0 at the top, 1 at the base): fresh at the
surface, `S_max` at the base.
"""
function bl99_salinity_profile(N::Integer; S_max::Real = 3.2)
    z = ((1:N) .- 0.5) ./ N
    return S_max / 2 .* (1 .- cos.(π .* z .^ (0.40706205 ./ (z .+ 0.57265966))))
end

"""
    ice_melting_temperature(S; p = IceParameters()) -> K

Melting (liquidus) temperature of sea ice of bulk salinity `S` (ppt),
``T_m = T_0 - μS``: the temperature at which the brine pockets have
swallowed the whole volume.
"""
ice_melting_temperature(S::Real; p::IceParameters = IceParameters()) = _T0(p) - p.μ * S

# ---- °C kernels (T ≤ T_m ≤ 0) ------------------------------------------

# q(S,T) = ρ [c0 (Tm - T) + L (1 - Tm/T)], BL99 eq. (5). For fresh ice
# (Tm = 0) the second term is L at every T ≤ 0, including T = 0.
function _q(p::IceParameters, Tc::Real, S::Real)
    Tm = -p.μ * S
    lat = Tm == 0 ? one(Tc) : 1 - Tm / Tc
    return _ρi(p) * (_c0(p) * (Tm - Tc) + _L(p) * lat)
end

# Inverse of E = -q(S,T): the negative root of
# c0 T² - (c0 Tm + L + E/ρ) T + L Tm = 0, capped at Tm.
function _T_of_E(p::IceParameters, E::Real, S::Real)
    Tm = -p.μ * S
    c0 = _c0(p)
    b = -(c0 * Tm + _L(p) + E / _ρi(p))
    disc = b^2 - 4 * c0 * _L(p) * Tm
    T = (-b - sqrt(max(disc, zero(disc)))) / (2c0)
    return min(T, Tm)
end

# dE/dT = ρ c(T,S), BL99 eq. (1) with γ = L μ (eq. 2). Fresh ice has no
# brine term — written out so that fresh ice sitting at exactly 0 °C
# (partly melted lake ice in spring) does not evaluate 0/0.
_C(p::IceParameters, Tc::Real, S::Real) =
    S == 0 ? _ρi(p) * _c0(p) : _ρi(p) * (_c0(p) + _L(p) * p.μ * S / Tc^2)

# k(S,T) = k0 + β S / T, BL99 eq. (9), with the BL99-code safeguards.
_k(p::IceParameters, Tc::Real, S::Real) =
    max(_k0(p) + p.β * S / min(Tc, p.T_k_cap), p.k_min)

# Snow: fresh, E_s = ρ_s (c0 T - L).
_Es(p::IceParameters, Tc::Real) = p.ρ_snow * (_c0(p) * Tc - _L(p))
_Ts_of_E(p::IceParameters, E::Real) = min((E / p.ρ_snow + _L(p)) / _c0(p), zero(E))

# ---- public, Kelvin ---------------------------------------------------

"""
    ice_energy_of_melting(T, S; p = IceParameters()) -> J m⁻³

Energy needed to melt a unit volume of sea ice of salinity `S` (ppt) at
temperature `T` (K) into brine at its melting temperature, BL99 eq. (5),

    q(S,T) = ρ_i [ c_0 (T_m - T) + L (1 - T_m/T) ]     (T, T_m in °C).

For fresh ice it is ``ρ_i [c_0 (0 - T) + L]``; for saline ice it falls
to zero at ``T = T_m`` (the ice is then all brine). At -2 °C and
S = 3.2 ppt it is 0.92 ``ρ_i L`` (BL99 §2).
"""
ice_energy_of_melting(T::Real, S::Real; p::IceParameters = IceParameters()) =
    _q(p, min(T - _T0(p), -p.μ * S), S)

"""
    ice_enthalpy(T, S; p = IceParameters()) -> J m⁻³

Enthalpy of sea ice per unit volume relative to meltwater at its melting
point, ``E = -q(S,T)`` — the prognostic variable of every ice layer.
"""
ice_enthalpy(T::Real, S::Real; p::IceParameters = IceParameters()) =
    -ice_energy_of_melting(T, S; p = p)

"""
    ice_temperature_from_enthalpy(E, S; p = IceParameters()) -> K

Inverse of [`ice_enthalpy`](@ref): the temperature of ice of salinity `S`
with enthalpy `E` (J m⁻³), from the quadratic
``c_0 T^2 - (c_0 T_m + L + E/ρ_i) T + L T_m = 0`` (T in °C), capped at
``T_m``. Fresh ice with ``-ρ_i L < E ≤ 0`` is partly melted and sits at
0 °C.
"""
ice_temperature_from_enthalpy(E::Real, S::Real; p::IceParameters = IceParameters()) =
    _T0(p) + _T_of_E(p, E, S)

"""
    ice_heat_capacity(T, S; p = IceParameters()) -> J kg⁻¹ K⁻¹

Effective specific heat of sea ice including brine-pocket melting,
``c(T,S) = c_0 + L μ S / T^2`` (Untersteiner 1961; Ono 1967; BL99 eq. 1,
with γ = Lμ ≈ 18 kJ K kg⁻¹). Equals ``(1/ρ_i) ∂E/∂T``.
"""
ice_heat_capacity(T::Real, S::Real; p::IceParameters = IceParameters()) =
    _C(p, T - _T0(p), S) / _ρi(p)

"""
    ice_conductivity(T, S; p = IceParameters()) -> W m⁻¹ K⁻¹

Thermal conductivity of sea ice, ``k = k_0 + β S / T`` (Untersteiner
1964; BL99 eq. 9), T in °C capped at `p.T_k_cap` and the result floored
at `p.k_min` as in the BL99 code. ``k_0`` is the contract's `k_ice`.
"""
ice_conductivity(T::Real, S::Real; p::IceParameters = IceParameters()) =
    _k(p, T - _T0(p), S)

"""
    snow_enthalpy(T; p = IceParameters()) -> J m⁻³

Enthalpy of (fresh) snow per unit volume relative to meltwater at 0 °C,
``E_s = ρ_s (c_0 T - L)`` (T in °C, ≤ 0).
"""
snow_enthalpy(T::Real; p::IceParameters = IceParameters()) =
    _Es(p, min(T - _T0(p), zero(T)))

# =========================================================================
# 3. Analytic growth laws (validation references and quick estimates)
# =========================================================================

"""
    stefan_ice_thickness(t; ΔT, h0 = 0, c = DEFAULT_SURFACE_CONSTANTS,
                         k = c.k_ice, ρ = c.ρ_ice, L = LH_f0(c)) -> m

Stefan's (1891) law for ice growing on water at its freezing point under
a surface held `ΔT` kelvin below it, neglecting the heat capacity of the
ice (quasi-steady linear profile) and any ocean heat flux:

    h(t)² = h0² + 2 k ΔT t / (ρ L).

Hence ``h ∝ \\sqrt{t}`` from ``h_0 = 0``; the basis of the degree-day
ice-growth rules used on lakes.
"""
function stefan_ice_thickness(t::Real; ΔT::Real, h0::Real = 0.0,
                              c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                              k::Real = c.k_ice, ρ::Real = c.ρ_ice, L::Real = LH_f0(c))
    return sqrt(h0^2 + 2k * ΔT * t / (ρ * L))
end

"""
    neumann_lambda(St) -> λ

Root of ``λ e^{λ²} \\mathrm{erf}(λ) = St/\\sqrt{π}``, the similarity
constant of the one-phase Stefan (Neumann) problem with Stefan number
``St = c ΔT / L``. For small `St`, ``λ ≈ \\sqrt{St/2}\\,(1 - St/6)``
(the Stefan limit plus the first heat-capacity correction). Solved by
Newton's method; the error function is `GasOptics.erf_series`.
"""
function neumann_lambda(St::Real)
    St > 0 || throw(ArgumentError("neumann_lambda: St must be positive"))
    λ = sqrt(St / 2)
    for _ in 1:60
        f  = λ * exp(λ^2) * erf_series(λ) - St / sqrt(π)
        df = exp(λ^2) * ((1 + 2λ^2) * erf_series(λ) + 2λ / sqrt(π))
        δ = f / df
        λ -= δ
        abs(δ) < 1e-15 * max(λ, 1.0) && break
    end
    return λ
end

"""
    neumann_ice_thickness(t; ΔT, c = DEFAULT_SURFACE_CONSTANTS, k = c.k_ice,
                          ρ = c.ρ_ice, cp = cp_ice(c), L = LH_f0(c)) -> m

Exact thickness of fresh ice grown from water at its freezing point
with the surface held `ΔT` below it, *including* the sensible heat of
the ice (Neumann solution; Carslaw & Jaeger 1959 §11.2):
``h = 2λ\\sqrt{α t}``, ``α = k/(ρ c_p)``, λ from [`neumann_lambda`](@ref).
The temperature profile is ``T(z) = T_s + ΔT\\,\\mathrm{erf}(z/2\\sqrt{αt})/\\mathrm{erf}(λ)``.
"""
function neumann_ice_thickness(t::Real; ΔT::Real,
                               c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS,
                               k::Real = c.k_ice, ρ::Real = c.ρ_ice,
                               cp::Real = cp_ice(c), L::Real = LH_f0(c))
    λ = neumann_lambda(cp * ΔT / L)
    return 2λ * sqrt(k / (ρ * cp) * t)
end

end # module
