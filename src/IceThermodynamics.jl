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

export IceParameters, IceColumn, sea_ice_column, lake_ice_column,
       seawater_freezing_point, bl99_salinity_profile,
       ice_melting_temperature, ice_energy_of_melting, ice_enthalpy,
       ice_temperature_from_enthalpy, ice_heat_capacity, ice_conductivity,
       ice_snow_enthalpy,
       PrescribedSurfaceTemperature, PrescribedSurfaceFluxes, IceStepBudget,
       step_ice!, ice_thickness, ice_snow_depth, ice_layer_temperatures,
       ice_column_enthalpy, ice_freeboard,
       stefan_ice_thickness, neumann_lambda, neumann_ice_thickness

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
    ice_snow_enthalpy(T; p = IceParameters()) -> J m⁻³

Enthalpy of (fresh) snow per unit volume relative to meltwater at 0 °C,
``E_s = ρ_s (c_0 T - L)`` (T in °C, ≤ 0).
"""
ice_snow_enthalpy(T::Real; p::IceParameters = IceParameters()) =
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


# =========================================================================
# 4. The ice column
# =========================================================================

"""
    IceStepBudget{T}

What one [`step_ice!`](@ref) did, as fluxes (W m⁻², positive *into* the
ice/snow column unless stated) and thickness changes (m):

  * `F_surface` — non-penetrating energy entering at the top surface
    (conduction into the column plus the energy spent on surface melt);
  * `F_penetrating` — shortwave that entered below the surface
    (``i_0`` fraction), `F_transmitted` — the part of it that left
    through the base into the water;
  * `F_conductive_top` — conductive flux at the top, positive downward;
    `F_conductive_bottom` — conductive flux at the base, positive
    *upward* (heat drawn from the freezing interface);
  * `F_ocean` — ocean/lake heat flux applied at the base;
  * `F_to_water` — energy handed to the water because the ice melted
    through or the surface was open water;
  * `enthalpy_in`, `enthalpy_out` — enthalpy (J m⁻²) carried in by
    snowfall/deposition and out by sublimation or by ice/snow dumped
    into the water;
  * `dh_top`, `dh_bottom`, `dh_snow` — net changes of ice (top, base)
    and snow thickness; `dh_snow_ice` — ice formed from flooded snow;
  * `T_surface` (K), `melting` — whether the surface was at melting point;
  * `energy_residual` — ``ΔU/Δt`` minus all the sources above (W m⁻²);
    zero to round-off in an energy-conserving model;
  * `iterations` — Newton iterations of the heat equation.
"""
struct IceStepBudget{T<:AbstractFloat}
    F_surface::T
    F_penetrating::T
    F_transmitted::T
    F_conductive_top::T
    F_conductive_bottom::T
    F_ocean::T
    F_to_water::T
    enthalpy_in::T
    enthalpy_out::T
    dh_top::T
    dh_bottom::T
    dh_snow::T
    dh_snow_ice::T
    T_surface::T
    melting::Bool
    energy_residual::T
    iterations::Int
end

IceStepBudget{T}() where {T} =
    IceStepBudget{T}(ntuple(_ -> zero(T), 14)..., false, zero(T), 0)

"""
    IceColumn

Thermodynamic ice column (BL99): `N` equal ice layers of thickness
`h_ice/N` with prognostic enthalpy `E` (J m⁻³) and a fixed salinity
profile `S` (ppt, attached to relative depth), one snow layer of depth
`h_snow` with enthalpy `E_snow`, a skin temperature `T_sfc` (K), and the
water below at its freezing point `T_bottom` (K) delivering the heat
flux `F_ocean` (W m⁻², into the ice). `S_new` is the salinity of ice
accreted at the base; `ρ_water` the density of the water (for the
freeboard and snow-ice formation). `budget` holds the last
[`IceStepBudget`](@ref).

Build one with [`sea_ice_column`](@ref) or [`lake_ice_column`](@ref).
`IceColumn <: AbstractSurface`: `step_surface!(col, forcing, dt)` couples
it to the atmosphere through `SurfaceExchange` (Monin–Obukhov fluxes,
sublimation); [`step_ice!`](@ref) drives it with a prescribed surface
temperature or prescribed fluxes.
"""
mutable struct IceColumn{T<:AbstractFloat, P<:IceParameters} <: AbstractSurface
    N::Int
    h_ice::T
    h_snow::T
    T_sfc::T
    E::Vector{T}
    S::Vector{T}
    S_new::T
    E_snow::T
    T_bottom::T
    F_ocean::T
    ρ_water::T
    params::P
    budget::IceStepBudget{T}
end

function _column(; N::Integer, h_ice::Real, h_snow::Real, T_surface::Real,
                 T_bottom::Real, S::AbstractVector, S_new::Real, F_ocean::Real,
                 ρ_water::Real, params::IceParameters,
                 T_layers::Union{Nothing, AbstractVector} = nothing,
                 T_snow::Union{Nothing, Real} = nothing)
    N >= 1 || throw(ArgumentError("need N ≥ 1 ice layers"))
    length(S) == N || throw(ArgumentError("salinity profile must have N = $N entries"))
    p = params
    T0 = _T0(p)
    # Default initial state: linear in depth from the surface to the base.
    Ts = min(T_surface, T0)
    if T_layers === nothing
        zc = ((1:N) .- 0.5) ./ N
        T_layers = Ts .+ (T_bottom - Ts) .* zc
    end
    length(T_layers) == N || throw(ArgumentError("T_layers must have N entries"))
    E = [_q(p, min(T_layers[j] - T0, -p.μ * S[j]), S[j]) for j in 1:N] .* -1
    Tsn = T_snow === nothing ? Ts : T_snow
    E_s = _Es(p, min(Tsn - T0, 0.0))
    h_ice > 0 || (E .= 0)
    return IceColumn{Float64, typeof(p)}(Int(N), Float64(h_ice), Float64(h_snow),
        Float64(h_ice > 0 ? Ts : T_bottom), Float64.(E), Float64.(S), Float64(S_new),
        Float64(E_s), Float64(T_bottom), Float64(F_ocean), Float64(ρ_water), p,
        IceStepBudget{Float64}())
end

"""
    sea_ice_column(; h_ice = 2.5, h_snow = 0.0, N = 10, T_surface = 253.15,
                     S_water = 34, T_bottom = seawater_freezing_point(S_water),
                     salinity = :bl99, S_max = 3.2, F_ocean = 2.0,
                     ρ_water = 1026, params = IceParameters(),
                     T_layers = nothing, T_snow = nothing) -> IceColumn

Sea-ice column. `salinity` is `:bl99` (the MU71/BL99 multiyear profile
[`bl99_salinity_profile`](@ref) with base value `S_max`), a number
(isosaline ice) or a vector of `N` layer salinities; new basal ice gets
`S_max` (BL99). The initial temperature is linear from `T_surface` to
`T_bottom` unless `T_layers` (K, top to bottom) is given. `F_ocean`
defaults to the 2 W m⁻² of MU71. `ρ_water` = 1026 kg m⁻³ is a
representative sea-water density, used only for the freeboard.
"""
function sea_ice_column(; h_ice::Real = 2.5, h_snow::Real = 0.0, N::Integer = 10,
                        T_surface::Real = 253.15, S_water::Real = 34.0,
                        params::IceParameters = IceParameters(),
                        T_bottom::Real = seawater_freezing_point(S_water; μ = params.μ,
                                                                 c = params.constants),
                        salinity = :bl99, S_max::Real = 3.2, F_ocean::Real = 2.0,
                        ρ_water::Real = 1026.0, T_layers = nothing, T_snow = nothing)
    S = salinity === :bl99 ? bl99_salinity_profile(N; S_max = S_max) :
        salinity isa Real ? fill(float(salinity), N) : collect(float.(salinity))
    S_new = salinity isa Real ? float(salinity) : float(S_max)
    return _column(; N, h_ice, h_snow, T_surface, T_bottom, S, S_new, F_ocean,
                   ρ_water, params, T_layers, T_snow)
end

"""
    lake_ice_column(; h_ice = 0.1, h_snow = 0.0, N = 10, T_surface = 263.15,
                      F_ocean = 0.0, params = IceParameters(),
                      T_layers = nothing, T_snow = nothing) -> IceColumn

Fresh-water (lake) ice: salinity zero everywhere, the water below at
0 °C (`seawater_freezing_point(0)`), fresh-water density from the
contract. With zero salinity the BL99 formulas reduce exactly to pure
ice (``c = c_0``, ``k = k_0``, ``q = ρ_i [c_0(0 - T) + L]``). `h_ice = 0`
starts from open water, which freezes over once it loses heat.
"""
function lake_ice_column(; h_ice::Real = 0.1, h_snow::Real = 0.0, N::Integer = 10,
                         T_surface::Real = 263.15, F_ocean::Real = 0.0,
                         params::IceParameters = IceParameters(),
                         T_layers = nothing, T_snow = nothing)
    T_bottom = seawater_freezing_point(0.0; μ = params.μ, c = params.constants)
    return _column(; N, h_ice, h_snow, T_surface, T_bottom, S = zeros(N), S_new = 0.0,
                   F_ocean, ρ_water = params.constants.ρ_water, params, T_layers, T_snow)
end

"Ice thickness (m)."
ice_thickness(c::IceColumn) = c.h_ice
"Snow depth on the ice (m)."
ice_snow_depth(c::IceColumn) = c.h_snow
"Layer temperatures (K), top to bottom, diagnosed from the enthalpies."
ice_layer_temperatures(c::IceColumn) =
    [_T0(c.params) + _T_of_E(c.params, c.E[j], c.S[j]) for j in 1:c.N]

"""
    ice_column_enthalpy(col) -> J m⁻²

Total enthalpy of the ice and snow relative to meltwater at the melting
point, ``U = Σ_j E_j h/N + E_s h_s`` (negative: the energy the column
would need to melt completely). Its change over a step is what the
energy budget closes against.
"""
ice_column_enthalpy(c::IceColumn) = sum(c.E) * c.h_ice / c.N + c.E_snow * c.h_snow

"""
    ice_freeboard(col) -> m

Height of the ice surface above the waterline by Archimedes,
``h - (ρ_i h + ρ_s h_s)/ρ_w``. Negative freeboard floods the snow, which
[`step_ice!`](@ref) then converts to snow ice (when `params.snow_ice`).
"""
ice_freeboard(c::IceColumn) =
    c.h_ice - (_ρi(c.params) * c.h_ice + c.params.ρ_snow * c.h_snow) / c.ρ_water

# ---- AbstractSurface interface ----------------------------------------

surface_temperature(c::IceColumn) = c.T_sfc
roughness_lengths(c::IceColumn) =
    (z0m = c.params.z0m, z0h = c.params.z0h, z0q = c.params.z0h)
function surface_albedo(c::IceColumn)
    p = c.params
    c.h_ice <= 0 && return p.α_water
    if c.h_snow > p.h_snow_min
        # "melting" = the surface was at the melting point last step (BL99
        # code: T_s ≥ -0.01 °C). A flag, not T_s itself, so the albedo does
        # not jump inside one step's surface-temperature iteration.
        return c.budget.melting ? p.α_snow_melt : p.α_snow
    end
    return p.α_ice
end
surface_emissivity(c::IceColumn) = c.params.ε
"Ice, snow and open water are saturated at the skin: β = 1."
moisture_availability(::IceColumn) = 1.0
"Saturation humidity over ice (over water when the ice has melted away)."
surface_specific_humidity(col::IceColumn, f::SurfaceForcing;
                          c::SurfaceConstants = col.params.constants) =
    saturation_specific_humidity(col.T_sfc, f.p, c.thermo;
                                 over = col.h_ice > 0 ? :ice : :liquid)
"Latent heat of sublimation over ice/snow, of vaporisation over open water."
surface_latent_heat(col::IceColumn; c::SurfaceConstants = col.params.constants) =
    col.h_ice > 0 ? latent_heat_sublim(col.T_sfc, c.thermo) :
                    latent_heat_vapor(col.T_sfc, c.thermo)

# =========================================================================
# 5. Boundary conditions
# =========================================================================

"""
    PrescribedSurfaceTemperature(T; snowfall = 0, T_snowfall = 273.15)

Hold the top surface at `T` (K) — the Stefan-problem boundary condition.
If `T` is at or above the melting point the surface melts with whatever
energy conduction cannot remove. `snowfall` in kg m⁻² s⁻¹.
"""
Base.@kwdef struct PrescribedSurfaceTemperature{R<:AbstractFloat}
    T::R
    snowfall::R   = 0.0
    T_snowfall::R = 273.15
end
PrescribedSurfaceTemperature(T::Real; kw...) = PrescribedSurfaceTemperature(; T = float(T), kw...)

"""
    PrescribedSurfaceFluxes(; sw_down, lw_down, H, LE, snowfall = 0,
                              T_snowfall = 273.15)

Atmospheric forcing as fluxes, the MU71/Semtner/BL99 protocol:
downwelling short- and longwave (W m⁻²), and *turbulent* sensible and
latent heat fluxes `H`, `LE` (W m⁻², positive upward, the contract sign)
that do not depend on the surface temperature. The surface balance is

    F(T_s) = (1-α)(1-i_0) SW↓ + ε LW↓ - ε σ T_s⁴ - H - LE.

As in MU71 and BL99, `LE` is an energy flux only (no sublimation mass
loss). `snowfall` in kg m⁻² s⁻¹, added at `T_snowfall` (K).
"""
Base.@kwdef struct PrescribedSurfaceFluxes{T<:AbstractFloat}
    sw_down::T
    lw_down::T
    H::T          = 0.0
    LE::T         = 0.0
    snowfall::T   = 0.0
    T_snowfall::T = 273.15
end

# Uniform access to the pieces the solver needs.
_snowfall(bc::Union{PrescribedSurfaceTemperature, PrescribedSurfaceFluxes}) = bc.snowfall
_snowfall(f::SurfaceForcing) = f.snow
_T_snowfall(bc::Union{PrescribedSurfaceTemperature, PrescribedSurfaceFluxes}, p) = bc.T_snowfall
_T_snowfall(f::SurfaceForcing, p) = min(f.T_air, _T0(p))
_sw(bc::PrescribedSurfaceTemperature) = 0.0
_sw(bc::PrescribedSurfaceFluxes) = bc.sw_down
_sw(f::SurfaceForcing) = f.sw_down

# Net non-penetrating flux into the surface at skin temperature Ts (K),
# F(Ts), with I0 the penetrating shortwave already subtracted.
function _surface_flux(col::IceColumn, bc::PrescribedSurfaceFluxes, Ts, I0)
    p = col.params
    ε, σ = p.ε, p.constants.σ_SB
    α = surface_albedo(col)
    F = (1 - α) * bc.sw_down - I0 + ε * bc.lw_down - ε * σ * Ts^4 - bc.H - bc.LE
    return F, -4ε * σ * Ts^3
end

function _surface_flux(col::IceColumn, f::SurfaceForcing, Ts, I0; kwargs...)
    F(T) = begin
        Told = col.T_sfc
        col.T_sfc = T
        fl = turbulent_fluxes(col, f; c = col.params.constants, kwargs...)
        col.T_sfc = Told
        fl.Rn - fl.H - fl.LE - I0
    end
    δ = 1e-3
    F0 = F(Ts)
    return F0, (F(Ts + δ) - F(Ts - δ)) / (2δ)
end

# =========================================================================
# 6. Heat equation (backward Euler in enthalpy form, Newton)
# =========================================================================

# Thomas algorithm, in place on copies.
function _tridiag(a, b, c, r)
    n = length(b)
    cp = similar(b); dp = similar(b); x = similar(b)
    cp[1] = c[1] / b[1]; dp[1] = r[1] / b[1]
    for i in 2:n
        m = b[i] - a[i] * cp[i-1]
        cp[i] = i < n ? c[i] / m : zero(m)
        dp[i] = (r[i] - a[i] * dp[i-1]) / m
    end
    x[n] = dp[n]
    for i in n-1:-1:1
        x[i] = dp[i] - cp[i] * x[i+1]
    end
    return x
end

# Solve for the new temperatures of the thermal stack (snow layer if
# active, then the ice layers). Returns the interface fluxes and the
# surface state. All temperatures in °C here.
function _heat_equation(col::IceColumn, bc, dt, I0, A; kwargs...)
    p = col.params
    T0 = _T0(p)
    snow = col.h_snow > p.h_snow_min
    N = col.N
    M = N + (snow ? 1 : 0)
    Δz = Vector{Float64}(undef, M)
    En = Vector{Float64}(undef, M)          # old enthalpies
    Tm = Vector{Float64}(undef, M)          # melting temperature of each layer
    Sv = Vector{Float64}(undef, M)
    src = zeros(M)                           # absorbed shortwave (W m⁻²)
    isnow = snow ? 1 : 0
    if snow
        Δz[1] = col.h_snow; En[1] = col.E_snow; Tm[1] = 0.0; Sv[1] = 0.0
    end
    for j in 1:N
        Δz[isnow+j] = col.h_ice / N
        En[isnow+j] = col.E[j]
        Tm[isnow+j] = -p.μ * col.S[j]
        Sv[isnow+j] = col.S[j]
        src[isnow+j] = A[j]
    end
    Tb = col.T_bottom - T0
    T = [k <= isnow ? _Ts_of_E(p, En[k]) : _T_of_E(p, En[k], Sv[k]) for k in 1:M]
    Tmelt = col.h_snow > 0 ? 0.0 : p.T_melt_ice
    E_of(k, Tk)  = k <= isnow ? _Es(p, Tk) : -_q(p, Tk, Sv[k])
    C_of(k, Tk)  = k <= isnow ? p.ρ_snow * _c0(p) : _C(p, Tk, Sv[k])
    k_of(k, Tk)  = k <= isnow ? p.k_snow : _k(p, Tk, Sv[k])

    prescribed = bc isa PrescribedSurfaceTemperature
    Ts = prescribed ? min(bc.T - T0, Tmelt) : min(col.T_sfc - T0, Tmelt)
    Fatm(Tc) = _surface_flux(col, bc, Tc + T0, I0; kwargs...)

    Kint = zeros(M + 1)          # Kint[k] = conductance of interface above layer k; Kint[M+1] = base
    function conductances!(T)
        kk = [k_of(k, T[k]) for k in 1:M]
        Kint[1] = 2kk[1] / Δz[1]
        for k in 2:M
            Kint[k] = 1 / (Δz[k-1] / (2kk[k-1]) + Δz[k] / (2kk[k]))
        end
        Kint[M+1] = 2kk[M] / Δz[M]
        return Kint
    end

    # Surface mode: melting if the melting-point surface would still gain energy.
    melting = if prescribed
        bc.T - T0 >= Tmelt
    else
        conductances!(T)
        first(Fatm(Tmelt)) + Kint[1] * (T[1] - Tmelt) > 0
    end
    fixed = falses(M)
    iters = 0
    a = zeros(M + 1); b = zeros(M + 1); c = zeros(M + 1); r = zeros(M + 1)
    for _switch in 1:4
        fill!(fixed, false)
        for it in 1:100
            iters += 1
            conductances!(T)
            # Row 1 = surface, rows 2..M+1 = layers.
            if prescribed || melting
                a[1] = 0; b[1] = 1; c[1] = 0; r[1] = prescribed ? Ts : Tmelt
            else
                F0, D = Fatm(Ts)
                a[1] = 0; b[1] = D - Kint[1]; c[1] = Kint[1]; r[1] = D * Ts - F0
            end
            for k in 1:M
                i = k + 1
                if fixed[k]
                    a[i] = 0; b[i] = 1; c[i] = 0; r[i] = Tm[k]
                    continue
                end
                Kup, Kdn = Kint[k], Kint[k+1]
                Ck = C_of(k, T[k])
                a[i] = -Kup
                b[i] = Δz[k] * Ck / dt + Kup + Kdn
                c[i] = k < M ? -Kdn : 0.0
                r[i] = src[k] + Δz[k] / dt * (En[k] - E_of(k, T[k]) + Ck * T[k]) +
                       (k == M ? Kdn * Tb : 0.0)
            end
            x = _tridiag(a, b, c, r)
            Tsn = x[1]
            Tn = x[2:end]
            for k in 1:M
                if Tn[k] > Tm[k]
                    Tn[k] = Tm[k]; fixed[k] = true
                end
            end
            err = max(abs(Tsn - Ts), maximum(abs.(Tn .- T)))
            Ts = Tsn; T .= Tn
            err < 1e-10 && break
        end
        prescribed && break
        if !melting && Ts > Tmelt
            melting = true; Ts = Tmelt
            continue
        end
        if melting
            conductances!(T)
            Fm = first(Fatm(Tmelt)) + Kint[1] * (T[1] - Tmelt)
            if Fm < 0 && _switch < 3
                melting = false
                continue
            end
        end
        break
    end
    conductances!(T)
    # Interface fluxes, positive upward: F[k] above layer k, F[M+1] at the base.
    F = zeros(M + 1)
    F[1] = Kint[1] * (T[1] - Ts)
    for k in 2:M
        F[k] = Kint[k] * (T[k] - T[k-1])
    end
    F[M+1] = Kint[M+1] * (Tb - T[M])
    F_melt = 0.0
    F_top_in = -F[1]
    if !prescribed
        Fs, _ = Fatm(Ts)
        if melting
            F_melt = Fs + F[1]           # surplus at the melting surface
            F_top_in = Fs
        end
    elseif Ts >= Tmelt
        F_melt = 0.0                     # prescribed: surface held, no atmosphere
    end
    # Conservative enthalpy update from the fluxes actually used.
    Enew = [En[k] + dt / Δz[k] * (F[k+1] - F[k] + src[k]) for k in 1:M]
    return (; Enew, F, Ts = Ts + T0, melting, F_melt, F_top_in, snow, iters)
end

# =========================================================================
# 7. The time step: heat equation, melt, growth, snow, remap
# =========================================================================

# Conservative remap of piecewise-constant enthalpy from layers of
# thickness dz (top to bottom) onto N equal layers of the total thickness.
function _remap(dz::Vector{Float64}, E::Vector{Float64}, N::Int)
    H = sum(dz)
    H <= 0 && return zeros(N), 0.0
    Δ = H / N
    out = zeros(N)
    zt_old = 0.0
    for (d, e) in zip(dz, E)
        d <= 0 && continue
        zb_old = zt_old + d
        j1 = clamp(floor(Int, zt_old / Δ) + 1, 1, N)
        j2 = clamp(ceil(Int, zb_old / Δ), 1, N)
        for j in j1:j2
            ov = min(zb_old, j * Δ) - max(zt_old, (j - 1) * Δ)
            ov > 0 && (out[j] += e * ov)
        end
        zt_old = zb_old
    end
    return out ./ Δ, H
end

"""
    step_ice!(col, bc, dt; kwargs...) -> IceStepBudget

Advance the ice column `col` by `dt` seconds under the top boundary
condition `bc` — a [`PrescribedSurfaceTemperature`](@ref), a
[`PrescribedSurfaceFluxes`](@ref) or a `SurfaceExchange.SurfaceForcing`
(Monin–Obukhov fluxes; `kwargs` go to `turbulent_fluxes`). The order of
operations follows BL99:

 1. shortwave: ``I_0 = i_0(1-α)SW↓`` penetrates snow-free ice and is
    absorbed with Beer's law (``κ_i``); what reaches the base goes to the
    water;
 2. heat equation, backward Euler in enthalpy form: Newton on the
    tridiagonal system for the layer temperatures (and the skin
    temperature from the surface balance ``F(T_s) = `` conduction),
    brine-pocket heat capacity ``ρ c(T,S)`` in the Jacobian; layers
    driven past their melting point are pinned there; the enthalpies are
    then updated from the interface fluxes, so energy is conserved;
 3. sublimation / deposition (only with a `SurfaceForcing`);
 4. surface melt with the surplus ``F(T_{melt}) - F_c`` — snow first,
    then ice, each consuming ``-E`` per unit volume; plus any layer whose
    enthalpy was pushed above zero (fully melted brine);
 5. basal accretion ``Δh = Δt(F_w - F_{cb})/E(T_f, S_{new})`` or
    basal melt layer by layer from below;
 6. snowfall, snow-ice formation below the waterline, and a
    conservative remap to `N` equal layers.

Open water (``h = 0``) freezes when the net surface flux at ``T_f``
plus `F_ocean` is negative (the new ice has ``E(T_f, S_{new})``);
otherwise the energy goes to the water. Returns the step's
[`IceStepBudget`](@ref), also stored in `col.budget`.
"""
step_ice!(col::IceColumn, bc, dt::Real; kwargs...) = first(_step!(col, bc, dt; kwargs...))

function _step!(col::IceColumn, bc, dt::Real; kwargs...)
    p = col.params
    T0 = _T0(p)
    ρi = _ρi(p)
    dt = float(dt)
    U0 = ice_column_enthalpy(col)
    E_in = 0.0; E_out = 0.0; to_water = 0.0
    h_ice0, h_snow0 = col.h_ice, col.h_snow

    # --- open water: freeze-up or hand the energy to the water ----------
    if col.h_ice <= 0
        bc isa PrescribedSurfaceTemperature &&
            throw(ArgumentError("step_ice!: a prescribed surface temperature needs ice (h_ice > 0)"))
        col.T_sfc = col.T_bottom
        fl = bc isa SurfaceForcing ? turbulent_fluxes(col, bc; c = p.constants, kwargs...) : nothing
        # snowfall on open water melts into it (enthalpy in and straight out)
        sf = _snowfall(bc) * dt
        Esf = sf / p.ρ_snow * _Es(p, min(_T_snowfall(bc, p) - T0, 0.0))
        E_in += Esf; E_out += Esf
        Fs = first(_surface_flux(col, bc, col.T_bottom, 0.0; kwargs...))
        Fnet = Fs + col.F_ocean
        Eg = -_q(p, col.T_bottom - T0, col.S_new)
        dh = 0.0
        if Fnet < 0 && Eg < 0
            dh = Fnet * dt / Eg
            col.h_ice = dh
            col.E .= Eg
            col.h_snow = 0.0
        else
            to_water = Fnet * dt
        end
        U1 = ice_column_enthalpy(col)
        res = (U1 - U0 - E_in + E_out + to_water) / dt - (Fs + col.F_ocean)
        col.budget = IceStepBudget{Float64}(Fs, 0.0, 0.0, 0.0, 0.0, col.F_ocean,
            to_water / dt, E_in, E_out, 0.0, dh, 0.0, 0.0, col.T_sfc,
            false, res, 0)
        return col.budget, fl
    end

    # --- wipe negligible snow (into the water) ----------------------------
    if col.h_snow > 0 && (col.h_snow < p.h_snow_bare || col.E_snow > 0)
        E_out += col.E_snow * col.h_snow
        col.h_snow = 0.0
    end

    # --- shortwave penetration --------------------------------------------
    N = col.N
    Δ = col.h_ice / N
    sw = _sw(bc)
    α = surface_albedo(col)
    I0 = col.h_snow < p.h_snow_bare ? p.i0 * (1 - α) * sw :
         p.i0_snow_scale > 0 ? p.i0 * p.i0_snow_scale / (col.h_snow + p.i0_snow_scale) * (1 - α) * sw :
         0.0
    A = [I0 * (exp(-p.κ_i * (j - 1) * Δ) - exp(-p.κ_i * j * Δ)) for j in 1:N]
    I_b = I0 * exp(-p.κ_i * col.h_ice)

    # --- heat equation -----------------------------------------------------
    sol = _heat_equation(col, bc, dt, I0, A; kwargs...)
    col.T_sfc = sol.Ts
    isnow = sol.snow ? 1 : 0
    sol.snow && (col.E_snow = sol.Enew[1])
    for j in 1:N
        col.E[j] = sol.Enew[isnow+j]
    end
    F_cond_top = -sol.F[1]
    F_cb = sol.F[end]

    # --- layer bookkeeping for melt/growth --------------------------------
    dz = fill(Δ, N)
    Ev = copy(col.E)
    hs = col.h_snow
    Es = col.E_snow
    Qtop = sol.F_melt * dt
    # A melting surface that turned out to be losing energy cools the top.
    if Qtop < 0
        if sol.snow
            Es += Qtop / hs
        else
            Ev[1] += Qtop / dz[1]
        end
        Qtop = 0.0
    end
    # Layers pushed above zero enthalpy are fully melted brine: their
    # surplus joins the surface-melt energy.
    if Es > 0 && hs > 0
        Qtop += Es * hs; Es = 0.0
    end
    for j in 1:N
        if Ev[j] > 0
            Qtop += Ev[j] * dz[j]; Ev[j] = 0.0
        end
    end

    # --- sublimation / deposition (Monin–Obukhov coupling only) ------------
    dh_sub = 0.0
    fl = nothing
    if bc isa SurfaceForcing
        fl = turbulent_fluxes(col, bc; c = p.constants, kwargs...)
        m = fl.E * dt                               # kg m⁻², + = sublimation
        if m > 0
            # snow first, then ice from the top
            ds = min(hs, m / p.ρ_snow)
            E_out += Es * ds; hs -= ds; m -= ds * p.ρ_snow
            for j in 1:N
                m <= 0 && break
                d = min(dz[j], m / ρi)
                E_out += Ev[j] * d; dz[j] -= d; m -= d * ρi
                dh_sub -= d
            end
        elseif m < 0
            if hs > 0
                d = -m / p.ρ_snow
                E_in += Es * d; hs += d
            else
                d = -m / ρi
                E_in += Ev[1] * d; dz[1] += d; dh_sub += d
            end
        end
    end

    # --- surface melt: snow first, then ice layers from the top ------------
    dh_top = dh_sub
    if Qtop > 0 && hs > 0
        cost = -Es * hs
        if Qtop < cost
            hs -= Qtop / -Es; Qtop = 0.0
        else
            Qtop -= cost; hs = 0.0
        end
    end
    for j in 1:N
        Qtop <= 0 && break
        dz[j] <= 0 && continue
        cost = -Ev[j] * dz[j]
        if Qtop < cost
            d = Qtop / -Ev[j]
            dz[j] -= d; dh_top -= d; Qtop = 0.0
        else
            Qtop -= cost; dh_top -= dz[j]; dz[j] = 0.0
        end
    end
    to_water += max(Qtop, 0.0)                    # melted through from the top

    # --- basal growth / melt ------------------------------------------------
    ebot = dt * (col.F_ocean - F_cb)
    dh_bot = 0.0
    if sum(dz) > 0
        if ebot < 0
            Eg = -_q(p, col.T_bottom - T0, col.S_new)
            dh_bot = ebot / Eg
            push!(dz, dh_bot); push!(Ev, Eg)
        elseif ebot > 0
            for j in N:-1:1
                ebot <= 0 && break
                dz[j] <= 0 && continue
                cost = -Ev[j] * dz[j]
                if ebot < cost
                    d = ebot / -Ev[j]
                    dz[j] -= d; dh_bot -= d; ebot = 0.0
                else
                    ebot -= cost; dh_bot -= dz[j]; dz[j] = 0.0
                end
            end
            if ebot > 0 && hs > 0                 # finally the snow
                cost = -Es * hs
                if ebot < cost
                    hs -= ebot / -Es; ebot = 0.0
                else
                    ebot -= cost; hs = 0.0
                end
            end
            to_water += max(ebot, 0.0)
        end
    else
        to_water += ebot                           # no ice left: the flux passes on
    end

    # --- snowfall -----------------------------------------------------------
    sf = _snowfall(bc) * dt / p.ρ_snow            # m of snow
    if sf > 0
        Ef = _Es(p, min(_T_snowfall(bc, p) - T0, 0.0))
        E_in += Ef * sf
        Es = hs > 0 ? (Es * hs + Ef * sf) / (hs + sf) : Ef
        hs += sf
    end

    # --- snow-ice formation (flooded snow → ice, mass and energy kept) ------
    dh_si = 0.0
    hi = sum(dz)
    if p.snow_ice && hs > 0 && hi > 0
        draft = (ρi * hi + p.ρ_snow * hs) / col.ρ_water
        if draft > hi
            dh_si = draft - hi
            dhs = dh_si * ρi / p.ρ_snow
            if dhs > hs                              # cannot convert more than exists
                dhs = hs; dh_si = hs * p.ρ_snow / ρi
            end
            pushfirst!(dz, dh_si); pushfirst!(Ev, Es * dhs / dh_si)
            hs -= dhs
        end
    end

    # --- remap to N equal layers --------------------------------------------
    Enew, H = _remap(dz, Ev, N)
    col.h_ice = H
    col.E .= Enew
    col.h_snow = hs
    col.E_snow = hs > 0 ? Es : _Es(p, 0.0)
    # Negligible ice melts into the water (with any snow on it).
    if col.h_ice < p.h_ice_min
        E_out += ice_column_enthalpy(col)
        col.h_ice = 0.0; col.h_snow = 0.0
        col.E .= 0.0
        col.T_sfc = col.T_bottom
    end
    if col.h_snow <= 0
        col.h_snow = 0.0
    end

    U1 = ice_column_enthalpy(col)
    F_surface = sol.F_top_in
    sources = F_surface + I0 - I_b + col.F_ocean
    res = (U1 - U0 - E_in + E_out + to_water) / dt - sources
    col.budget = IceStepBudget{Float64}(F_surface, I0, I_b, F_cond_top, F_cb,
        col.F_ocean, to_water / dt, E_in, E_out, dh_top, dh_bot,
        col.h_snow - h_snow0, dh_si, col.T_sfc, sol.melting, res, sol.iters)
    return col.budget, fl
end

"""
    step_surface!(col::IceColumn, forcing::SurfaceForcing, dt; kwargs...) -> SurfaceFluxes

`AbstractSurface` entry point: one [`step_ice!`](@ref) under
Monin–Obukhov coupling, returning the fluxes at the *new* skin
temperature, with `G` the energy that actually entered the ice/snow
column (surface conduction + surface melt + penetrating shortwave), so
``R_n - H - LE - G`` is the (Newton-converged, ~1e-9 W m⁻²) residual of
the surface balance. Over open water `G` is what the water received.
"""
function step_surface!(col::IceColumn, f::SurfaceForcing, dt::Real; kwargs...)
    b, fl = _step!(col, f, dt; kwargs...)
    G = b.F_surface + b.F_penetrating
    return SurfaceFluxes(fl.Rn, fl.H, fl.LE, G, fl.τ, fl.E)
end

end # module
