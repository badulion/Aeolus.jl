"""
    MoistThermodynamics

Exact moist thermodynamics — Era IV, Phase 39.

This module is the shared contract for Era IV the way `RadiationCore` is
for Era III: every microphysics scheme, activation parameterisation and
cloud diagnostic in Phases 40-45 speaks it, so they agree on what
temperature, saturation and buoyancy mean.

WHAT "EXACT" MEANS HERE, AND WHY IT MATTERS

Three approximations are routine in LES codes and all three are dropped:

 1. **Constant latent heat.** `L_v` is temperature dependent through
    Kirchhoff's relation, `dL_v/dT = c_pv - c_l`. Over the 220-320 K
    range an atmosphere actually spans, `L_v` varies by about 10 %.
    Holding it at its 273 K value biases condensational heating in deep
    convection by a comparable amount.

 2. **Fitted saturation vapour pressure.** Tetens/Bolton/Magnus fits are
    accurate near 273 K and degrade in the cold upper troposphere, which
    is exactly where ice microphysics needs them. Here `e_s` is the
    *analytic integral* of Clausius-Clapeyron with the Kirchhoff `L(T)`,
    so it is thermodynamically consistent with the latent heat used
    everywhere else, by construction rather than by coincidence.

 3. **Constant heat capacity.** The moist heat capacity `c_pm` depends
    on the vapour and condensate loading and is carried explicitly.

CONSISTENCY IS THE POINT. A scheme that uses one `L_v` in its saturation
adjustment and a different effective `L_v` implicit in its `e_s` fit does
not conserve moist static energy, and the error accumulates over a long
integration rather than cancelling.

TWO SATURATION ADJUSTMENTS (issue #82)

  * `saturation_adjustment(θ_li, p, q_tot)` is a **diagnosis**: it
    recovers `T` and the partition that a prognostic `θ_li` denotes, and
    reproduces `θ_li` exactly.
  * `isobaric_saturation_adjustment(T, p, q_v, q_c; q_r, q_i)` is a
    **process**: condensation or evaporation during an isobaric step,
    conserving `moist_enthalpy` exactly, with rain and ice in `c_pm`.

They differ because this module's θ_li is only approximately conserved
through condensation: its Exner exponent R_m/c_pm is taken at the current
partition (Bryan & Fritsch's exact form uses total water only), so
holding it fixed misstates latent heating by -0.2 % at 1000 hPa, +1.1 %
at 900 hPa and +6.3 % at 600 hPa (285 K, S = 1.1). Microphysics (P3,
fog) uses the process form; the sub-grid cloud scheme and parcel CAPE
use the diagnosis, which must invert whatever θ_li the model carries.

REFERENCES

  * Emanuel, K. A. (1994). *Atmospheric Convection*. Oxford. Ch. 4.
  * Bolton, D. (1980). The computation of equivalent potential
    temperature. *Mon. Wea. Rev.*, 108, 1046-1053.
  * Bryan, G. H. & Fritsch, J. M. (2004). A reevaluation of ice-liquid
    water potential temperature. *Mon. Wea. Rev.*, 132, 2421-2431.
  * Tripoli, G. J. & Cotton, W. R. (1981). The use of ice-liquid water
    potential temperature as a thermodynamic variable in deep atmospheric
    models. *Mon. Wea. Rev.*, 109, 1094-1102.
  * Romps, D. M. (2017). Exact expression for the lifting condensation
    level. *J. Atmos. Sci.*, 74, 3891-3900.
"""
module MoistThermodynamics

export ThermodynamicConstants, DEFAULT_CONSTANTS,
       latent_heat_vapor, latent_heat_sublim, latent_heat_fusion,
       saturation_vapor_pressure, saturation_vapor_pressure_ice,
       saturation_specific_humidity, saturation_mixing_ratio,
       saturation_specific_humidity_ice,
       vapor_pressure, relative_humidity, relative_humidity_ice,
       specific_humidity_from_relative_humidity,
       dewpoint_temperature, frost_point_temperature, wet_bulb_temperature,
       gas_constant_moist, heat_capacity_moist, exner_function,
       potential_temperature, temperature_from_theta,
       liquid_ice_potential_temperature, temperature_from_theta_li,
       virtual_temperature, density_temperature,
       virtual_potential_temperature, density_potential_temperature,
       equivalent_potential_temperature, wet_bulb_potential_temperature,
       moist_static_energy, liquid_water_static_energy,
       air_density, specific_volume,
       saturation_adjustment, saturation_adjustment_ice,
       PhasePartition, phase_partition, liquid_fraction,
       moist_adiabatic_lapse_rate, dry_adiabatic_lapse_rate,
       lifting_condensation_level, level_of_free_convection,
       cape_cin, buoyancy, brunt_vaisala_frequency_moist,
       saturated_buoyancy_frequency_squared,
       supersaturation, supersaturation_ice,
       kelvin_coefficient, kohler_supersaturation, kohler_critical_point,
       kohler_equilibrium_radius, surface_tension_water,
       kappa_from_solute, solute_from_kappa,
       effective_radius,
       moist_enthalpy, temperature_from_moist_enthalpy,
       isobaric_saturation_adjustment

# =========================================================================
# Constants
# =========================================================================

"""
    ThermodynamicConstants{T}

Thermodynamic constants, bundled so a caller can run in `Float32` or
substitute a different reference state without touching call sites.

Reference point: the triple point of water, `T_triple` = 273.16 K,
`p_triple` = 611.657 Pa. Latent heats are given *at* the triple point and
extrapolated by Kirchhoff's relation, which is what makes them consistent
with the saturation vapour pressure below.
"""
Base.@kwdef struct ThermodynamicConstants{T<:AbstractFloat}
    gas_constant::T = 8.314462618      # J mol⁻¹ K⁻¹ (CODATA 2018)
    molmass_dry::T  = 0.02896546       # kg mol⁻¹
    molmass_water::T = 0.018015268     # kg mol⁻¹
    cp_dry::T       = 1004.6           # J kg⁻¹ K⁻¹
    cp_vapor::T     = 1859.0           # J kg⁻¹ K⁻¹
    cp_liquid::T    = 4181.0           # J kg⁻¹ K⁻¹  (at 298 K)
    cp_ice::T       = 2108.0           # J kg⁻¹ K⁻¹
    LH_v0::T        = 2.5008e6         # J kg⁻¹ at the triple point
    LH_s0::T        = 2.8344e6         # J kg⁻¹ at the triple point
    T_triple::T     = 273.16           # K
    p_triple::T     = 611.657          # Pa
    T_freeze::T     = 273.15           # K
    T_icenuc::T     = 233.15           # K, all-ice below this
    p_ref::T        = 1.0e5            # Pa, potential-temperature reference
    grav::T         = 9.80665          # m s⁻²
end

const DEFAULT_CONSTANTS = ThermodynamicConstants{Float64}()

R_d(c::ThermodynamicConstants) = c.gas_constant / c.molmass_dry     # 287.06
R_v(c::ThermodynamicConstants) = c.gas_constant / c.molmass_water   # 461.52
"Ratio of molar masses, ε ≈ 0.622."
molmass_ratio(c::ThermodynamicConstants) = c.molmass_water / c.molmass_dry
"Latent heat of fusion at the triple point."
LH_f0(c::ThermodynamicConstants) = c.LH_s0 - c.LH_v0

# =========================================================================
# Phase partition
# =========================================================================

"""
    PhasePartition{T}

Specific humidities of the three water phases: `tot` (total water),
`liq` (cloud liquid) and `ice` (cloud ice), all in kg water per kg of
*moist* air. Vapour is the remainder, `tot - liq - ice`.

Total water rather than vapour is the prognostic because it is conserved
under phase change; vapour is not.
"""
struct PhasePartition{T<:AbstractFloat}
    tot::T
    liq::T
    ice::T
end

PhasePartition(tot::Real, liq::Real = 0, ice::Real = 0) =
    PhasePartition{typeof(float(tot))}(float(tot), float(liq), float(ice))

"Vapour specific humidity of a phase partition."
vapor(q::PhasePartition) = q.tot - q.liq - q.ice
"Total condensate specific humidity."
condensate(q::PhasePartition) = q.liq + q.ice

Base.show(io::IO, q::PhasePartition) =
    print(io, "PhasePartition(tot=", q.tot, ", liq=", q.liq, ", ice=", q.ice, ")")

"""
    liquid_fraction(T, c = DEFAULT_CONSTANTS) -> λ ∈ [0, 1]

Fraction of condensate that is liquid at temperature `T`.

Liquid above freezing, ice below the homogeneous-nucleation threshold
(233.15 K), and a smooth ramp between — the mixed-phase regime, where
supercooled liquid and ice genuinely coexist. The ramp is quadratic
rather than linear so that both endpoints join with zero slope; a kink
in `λ(T)` becomes a kink in the latent heating and shows up as grid-scale
noise at the 0 °C level.
"""
function liquid_fraction(T::Real, c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    Tk >= c.T_freeze && return one(Tk)
    Tk <= c.T_icenuc && return zero(Tk)
    x = (Tk - c.T_icenuc) / (c.T_freeze - c.T_icenuc)   # 0 → 1
    return x * x * (3 - 2x)                              # smoothstep
end

"""
    phase_partition(T, q_tot, q_cond, c = DEFAULT_CONSTANTS) -> PhasePartition

Split a given amount of condensate between liquid and ice using
[`liquid_fraction`](@ref).
"""
function phase_partition(T::Real, q_tot::Real, q_cond::Real,
                         c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    λ = liquid_fraction(T, c)
    qc = max(zero(float(q_cond)), float(q_cond))
    return PhasePartition(float(q_tot), λ * qc, (1 - λ) * qc)
end

# =========================================================================
# Latent heats — Kirchhoff's relation
# =========================================================================

"""
    latent_heat_vapor(T, c = DEFAULT_CONSTANTS) -> J kg⁻¹

Latent heat of vaporisation, `L_v(T) = L_v0 + (c_pv - c_l)(T - T_triple)`.

This is Kirchhoff's relation, and it is exact given temperature-independent
heat capacities. Since `c_pv - c_l` ≈ -2322 J kg⁻¹ K⁻¹, `L_v` *falls* with
temperature: 2.50e6 at 273 K, 2.26e6 at 373 K, 2.62e6 at 223 K. Holding it
constant at the triple-point value understates condensational heating in a
cold cloud by ~5 % and overstates it in a warm one.
"""
latent_heat_vapor(T::Real, c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    c.LH_v0 + (c.cp_vapor - c.cp_liquid) * (float(T) - c.T_triple)

"Latent heat of sublimation, `L_s(T) = L_s0 + (c_pv - c_i)(T - T_triple)`."
latent_heat_sublim(T::Real, c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    c.LH_s0 + (c.cp_vapor - c.cp_ice) * (float(T) - c.T_triple)

"Latent heat of fusion, `L_f = L_s - L_v`. Rises with temperature."
latent_heat_fusion(T::Real, c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    latent_heat_sublim(T, c) - latent_heat_vapor(T, c)

# =========================================================================
# Saturation vapour pressure — exact Clausius-Clapeyron
# =========================================================================

# Integrating  d(ln e_s)/dT = L(T) / (R_v T²)  with L(T) = L0 + Δcp (T - T0)
# gives the closed form
#
#   e_s(T) = e_triple · (T/T0)^(Δcp/R_v) · exp[ (L0 - Δcp T0)/R_v · (1/T0 - 1/T) ]
#
# which is exact for the Kirchhoff L(T) above — not a fit. Every use of
# L_v elsewhere in this module is therefore thermodynamically consistent
# with every use of e_s, which is the property that makes moist static
# energy conserve to round-off instead of drifting.

function _saturation_vapor_pressure(T::Real, L0::Real, Δcp::Real,
                                    c::ThermodynamicConstants)
    Tk = float(T)
    Tk <= 0 && throw(DomainError(T, "saturation_vapor_pressure: T must be > 0 K"))
    Rv = R_v(c)
    T0 = c.T_triple
    # one fused exponential: (T/T0)^a · exp(b) = exp(a ln(T/T0) + b). The
    # power was a second exp/log pair; fusing them is a named one-time
    # change of the last bits of e_s (and q_s) everywhere (KA-m, #301:
    # max relative change 8.3e-16 over 230-320 K, 2.4e-15 over 150-400 K;
    # docs/src/design/ka_port.md).
    return c.p_triple * exp(Δcp / Rv * log(Tk / T0) +
                            (L0 - Δcp * T0) / Rv * (1 / T0 - 1 / Tk))
end

"""
    saturation_vapor_pressure(T, c = DEFAULT_CONSTANTS) -> Pa

Saturation vapour pressure over a plane surface of **liquid** water,
valid (as supercooled liquid) well below freezing.

Analytic integral of Clausius-Clapeyron with the Kirchhoff `L_v(T)`, so
it is consistent with [`latent_heat_vapor`](@ref) by construction. At
273.15 K it gives 611.2 Pa; at 373.15 K, 101.3 kPa — i.e. it reproduces
the normal boiling point without being fitted to it.
"""
saturation_vapor_pressure(T::Real, c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    _saturation_vapor_pressure(T, c.LH_v0, c.cp_vapor - c.cp_liquid, c)

"""
    saturation_vapor_pressure_ice(T, c = DEFAULT_CONSTANTS) -> Pa

Saturation vapour pressure over **ice**.

Below freezing this is *lower* than the value over liquid, and that gap
is the engine of the Wegener-Bergeron-Findeisen process: in a mixed-phase
cloud the air is supersaturated with respect to ice while subsaturated
with respect to liquid, so ice grows at the expense of evaporating
droplets. The gap peaks near 261 K at about 1.5 % of `e_s`.
"""
saturation_vapor_pressure_ice(T::Real, c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    _saturation_vapor_pressure(T, c.LH_s0, c.cp_vapor - c.cp_ice, c)

"""
    saturation_specific_humidity(T, ρ_or_p, c; over = :liquid, use_density = false)

Saturation specific humidity. By default `ρ_or_p` is interpreted as
*pressure* (Pa) and

    q_sat = ε e_s / (p - (1 - ε) e_s)

is used. With `use_density = true` it is interpreted as moist-air density
and `q_sat = e_s / (ρ R_v T)` — the form microphysics schemes want,
because it does not require knowing the pressure.
"""
function saturation_specific_humidity(T::Real, ρ_or_p::Real,
                                      c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                                      over::Symbol = :liquid,
                                      use_density::Bool = false)
    es = over === :liquid ? saturation_vapor_pressure(T, c) :
         over === :ice    ? saturation_vapor_pressure_ice(T, c) :
         throw(ArgumentError("saturation_specific_humidity: `over` must be :liquid or :ice"))
    if use_density
        return es / (float(ρ_or_p) * R_v(c) * float(T))
    end
    return _saturation_specific_humidity(T, es, ρ_or_p, c)
end

# q_sat from e_s at pressure p (the pressure form of saturation_specific_humidity)
function _saturation_specific_humidity(T::Real, es::Real, p::Real, c::ThermodynamicConstants)
    p = float(p)
    ε = molmass_ratio(c)
    # Once e_s reaches ambient pressure the water boils: there is no
    # saturated *mixture* any more, the air is all vapour, and q_sat = 1.
    # The formula below is continuous into that limit (at e_s = p it gives
    # exactly 1) but overshoots past it — at 400 K and 1 bar it returns
    # 15.2, which is not a specific humidity. Guarding only on the
    # denominator is not enough: it stays positive well past boiling.
    # The overshoot also makes θ_e non-monotone, which silently breaks any
    # bisection that brackets above the boiling point.
    es >= p && return one(float(T))
    den = p - (1 - ε) * es
    den <= 0 && return one(float(T))
    return min(ε * es / den, one(float(T)))
end

"Saturation specific humidity over ice."
saturation_specific_humidity_ice(T::Real, p::Real,
                                 c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    saturation_specific_humidity(T, p, c; over = :ice)

"""
    saturation_mixing_ratio(T, p, c) -> kg kg⁻¹

Saturation *mixing* ratio, `r_sat = ε e_s / (p - e_s)` — mass of vapour
per mass of **dry** air, as distinct from specific humidity which is per
mass of moist air. The two differ by ~2 % in the warm tropics; mixing up
which one a scheme wants is a classic source of small persistent biases.
"""
function saturation_mixing_ratio(T::Real, p::Real,
                                 c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                                 over::Symbol = :liquid)
    es = over === :liquid ? saturation_vapor_pressure(T, c) :
                            saturation_vapor_pressure_ice(T, c)
    return _saturation_mixing_ratio(es, p, c)
end

# r_sat from e_s at pressure p
function _saturation_mixing_ratio(es::Real, p::Real, c::ThermodynamicConstants)
    den = float(p) - es
    den <= 0 && return float(Inf)
    return molmass_ratio(c) * es / den
end

"""
    vapor_pressure(q_v, p, c) -> Pa

Partial pressure of water vapour given specific humidity and total
pressure.
"""
function vapor_pressure(q_v::Real, p::Real,
                        c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    ε = molmass_ratio(c)
    qv = float(q_v)
    return float(p) * qv / (ε + (1 - ε) * qv)
end

"Relative humidity over liquid, as a fraction (1.0 = saturated)."
relative_humidity(T::Real, p::Real, q_v::Real,
                  c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    vapor_pressure(q_v, p, c) / saturation_vapor_pressure(T, c)

"""
    specific_humidity_from_relative_humidity(T, p, RH, c = DEFAULT_CONSTANTS) -> kg kg⁻¹

Inverse of [`relative_humidity`](@ref): the specific humidity of air at
`T` (K) and `p` (Pa) whose vapour pressure is ``e = RH\\, e_s(T)`` (over
liquid), ``q = ε e / (p - (1 - ε) e)``. Note that this is *not*
``RH \\cdot q_{sat}``, which differs by the factor
``(p - (1-ε)e_s)/(p - (1-ε)e)``.
"""
function specific_humidity_from_relative_humidity(T::Real, p::Real, RH::Real,
                                                  c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    RH >= 0 || throw(ArgumentError("specific_humidity_from_relative_humidity: RH must be ≥ 0"))
    e = float(RH) * saturation_vapor_pressure(T, c)
    ε = molmass_ratio(c)
    return ε * e / (float(p) - (1 - ε) * e)
end

"Relative humidity over ice."
relative_humidity_ice(T::Real, p::Real, q_v::Real,
                      c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    vapor_pressure(q_v, p, c) / saturation_vapor_pressure_ice(T, c)

"Supersaturation over liquid, `S - 1`. Positive means condensation."
supersaturation(T::Real, p::Real, q_v::Real,
                c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    relative_humidity(T, p, q_v, c) - 1

"Supersaturation over ice."
supersaturation_ice(T::Real, p::Real, q_v::Real,
                    c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    relative_humidity_ice(T, p, q_v, c) - 1

"""
    dewpoint_temperature(p, q_v, c) -> K

Temperature at which the given vapour would saturate over liquid at
constant pressure. Solved by bisection on `e_s(T_d) = e(q_v, p)`, which
is monotone, rather than by an inverted fit.
"""
function dewpoint_temperature(p::Real, q_v::Real,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    e = vapor_pressure(q_v, p, c)
    e <= 0 && return float(150.0)
    return _invert_monotone(T -> saturation_vapor_pressure(T, c) - e, 150.0, 400.0)
end

"Frost point: as [`dewpoint_temperature`](@ref) but saturating over ice."
function frost_point_temperature(p::Real, q_v::Real,
                                 c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    e = vapor_pressure(q_v, p, c)
    e <= 0 && return float(150.0)
    return _invert_monotone(T -> saturation_vapor_pressure_ice(T, c) - e, 150.0, 400.0)
end

"""
    wet_bulb_temperature(T, p, q_v, c = DEFAULT_CONSTANTS) -> K

Thermodynamic (adiabatic-saturation) wet-bulb temperature ``T_w`` of air
at `T` (K), pressure `p` (Pa) and specific humidity `q_v`: the
temperature reached when liquid water at ``T_w`` evaporates into the air
at constant pressure until it saturates. With mixing ratio ``r =
q_v/(1-q_v)``, the isobaric enthalpy balance of this module's
thermodynamics (constant ``c_{pd}``, ``c_{pv}``, ``c_l`` and the
Kirchhoff ``L_v(T)``) is exactly

```math
(c_{pd} + r c_{pv}) (T - T_w) = \\bigl(r_s(T_w, p) - r\\bigr) L_v(T_w),
```

solved by bisection on ``[150 K, T]`` (the residual is monotone in
``T_w``). This is the psychrometric wet bulb of ASHRAE *Fundamentals*
(2017) ch. 1, eq. 33-35, with the same saturation curve as the rest of
the model. Saturated or supersaturated air returns `T`.

Used to drive equipment curves rated on wet-bulb temperature (cooling
coils, evaporative condensers and cooling towers).
"""
function wet_bulb_temperature(T::Real, p::Real, q_v::Real,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk, pp = float(T), float(p)
    r = float(q_v) / (1 - float(q_v))
    saturation_mixing_ratio(Tk, pp, c) <= r && return Tk
    f(Tw) = (saturation_mixing_ratio(Tw, pp, c) - r) * latent_heat_vapor(Tw, c) -
            (c.cp_dry + r * c.cp_vapor) * (Tk - Tw)
    return _invert_monotone(f, 150.0, Tk)
end

# Bisection on a monotonically increasing function. Bisection rather than
# Newton on purpose: it cannot diverge, and these are called inside
# microphysics loops where a single non-converged column would otherwise
# poison an entire timestep with NaN.
function _invert_monotone(f, lo::Real, hi::Real; tol::Real = 1e-10,
                          maxiter::Int = 200)
    a, b = float(lo), float(hi)
    fa = f(a)
    fa > 0 && return a
    f(b) < 0 && return b
    for _ in 1:maxiter
        m = 0.5 * (a + b)
        (b - a) < tol && return m
        f(m) < 0 ? (a = m) : (b = m)
    end
    return 0.5 * (a + b)
end

# =========================================================================
# Moist gas "constants"
# =========================================================================

"""
    gas_constant_moist(q, c) -> J kg⁻¹ K⁻¹

Gas constant of moist air per unit mass of *moist* air:

    R_m = R_d (1 - q_tot) + R_v q_vap

Note it is `q_tot` that is subtracted, not `q_vap`: condensate occupies
negligible volume but carries mass, so it dilutes the gas constant.
"""
gas_constant_moist(q::PhasePartition, c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    R_d(c) * (1 - q.tot) + R_v(c) * vapor(q)

gas_constant_moist(q_tot::Real, q_liq::Real = 0, q_ice::Real = 0,
                   c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    gas_constant_moist(PhasePartition(q_tot, q_liq, q_ice), c)

"""
    heat_capacity_moist(q, c) -> J kg⁻¹ K⁻¹

Isobaric heat capacity of moist air including condensate loading:

    c_pm = c_pd (1 - q_tot) + c_pv q_vap + c_l q_liq + c_i q_ice

Liquid water has four times the heat capacity of dry air, so even a
modest cloud loading is not negligible at the accuracy this module aims
for.
"""
heat_capacity_moist(q::PhasePartition, c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    c.cp_dry * (1 - q.tot) + c.cp_vapor * vapor(q) +
    c.cp_liquid * q.liq + c.cp_ice * q.ice

heat_capacity_moist(q_tot::Real, q_liq::Real = 0, q_ice::Real = 0,
                    c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    heat_capacity_moist(PhasePartition(q_tot, q_liq, q_ice), c)

"""
    exner_function(p, q, c) -> Π

Exner function `Π = (p/p_ref)^(R_m/c_pm)`. With a `PhasePartition` the
moist exponent is used; the dry exponent `R_d/c_pd` ≈ 0.2857 is the
`q = 0` limit.
"""
exner_function(p::Real, q::PhasePartition = PhasePartition(0.0),
               c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    (float(p) / c.p_ref)^(gas_constant_moist(q, c) / heat_capacity_moist(q, c))

"""
    air_density(T, p, q = PhasePartition(0.0), c = DEFAULT_CONSTANTS) -> kg m⁻³

Density of moist air (dry air + vapour + condensate) from the ideal-gas
law with the moist gas constant of [`gas_constant_moist`](@ref):

    ρ = p / (R_m T)

Condensate adds mass but negligible volume, so at fixed `T`, `p` and
vapour it *raises* `ρ` through the `(1 - q_tot)` factor in `R_m`.
`specific_volume` is `1/ρ`.
"""
air_density(T::Real, p::Real, q::PhasePartition = PhasePartition(0.0),
            c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    float(p) / (gas_constant_moist(q, c) * float(T))

specific_volume(T::Real, p::Real, q::PhasePartition = PhasePartition(0.0),
                c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    1 / air_density(T, p, q, c)

# =========================================================================
# Potential temperatures
# =========================================================================

"Potential temperature `θ = T / Π`."
potential_temperature(T::Real, p::Real, q::PhasePartition = PhasePartition(0.0),
                      c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    float(T) / exner_function(p, q, c)

"Temperature from potential temperature, `T = θ Π`."
temperature_from_theta(θ::Real, p::Real, q::PhasePartition = PhasePartition(0.0),
                       c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    float(θ) * exner_function(p, q, c)

"""
    liquid_ice_potential_temperature(T, p, q, c) -> θ_li

Liquid-ice potential temperature, in an exponential approximation of the
kind reviewed by Bryan & Fritsch (2004) (cf. Tripoli & Cotton 1981):

    θ_li = θ · exp( -(L_v q_liq + L_s q_ice) / (c_pm T) )

with `θ = T/Π` taken at the moist Exner exponent `R_m/c_pm` of the
**current** partition `q`.

This is the quantity an LES carries as its thermodynamic prognostic,
because it is **approximately conserved under reversible phase change** —
condensation raises `T` and lowers `q_vap` in nearly the combination that
leaves `θ_li` unchanged. Advecting `θ_li` and `q_tot` therefore means
phase change never has to be advected, which is what keeps a cloud edge
sharp.

The conservation is not exact, and the dominant error is the exponent.
Bryan & Fritsch's exact form evaluates it with total water only, so it is
fixed during phase change; here `R_m/c_pm` moves as vapour becomes
condensate. Measured against isobaric moist enthalpy (issue #82: 285 K,
10 % supersaturation), holding this `θ_li` fixed gives
`c_pm ΔT/(L_v(T₀) Δq)` = 0.998, 1.011 and 1.063 at 1000, 900 and 600 hPa.
With the exponent frozen at the all-vapour partition the same measure is
0.998 at every pressure (tests), the residual being the exponential
approximation, first order in the supersaturation. That is why
condensation as a *process* uses [`isobaric_saturation_adjustment`](@ref)
and not an inversion of this function.

The exponential form is used rather than the linearised
`θ - (L/c_p)(p₀/p)^κ q_l`. The linearisation errs by ~0.3 K at 3 g/kg of
liquid — small, but it is a *systematic* bias, so it accumulates across a
deep convective column instead of averaging out.
"""
function liquid_ice_potential_temperature(T::Real, p::Real, q::PhasePartition,
                                          c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    cpm = heat_capacity_moist(q, c)
    Lv = latent_heat_vapor(Tk, c)
    Ls = latent_heat_sublim(Tk, c)
    θ = potential_temperature(Tk, p, q, c)
    return θ * exp(-(Lv * q.liq + Ls * q.ice) / (cpm * Tk))
end

"""
    temperature_from_theta_li(θ_li, p, q, c) -> T

Invert [`liquid_ice_potential_temperature`](@ref) for temperature at a
**known** phase partition. Bisection over a physically generous bracket;
see [`saturation_adjustment`](@ref) for the case where the partition is
itself unknown.
"""
function temperature_from_theta_li(θ_li::Real, p::Real, q::PhasePartition,
                                   c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                                   tol::Real = 1e-10)
    f(T) = liquid_ice_potential_temperature(T, p, q, c) - float(θ_li)
    # θ_li increases monotonically with T at fixed p and partition.
    return _invert_monotone(f, 100.0, 1000.0; tol = tol)
end

"""
    virtual_temperature(T, q, c) -> K

Virtual temperature: the temperature dry air would need to have the same
density as this moist parcel at the same pressure. Vapour is lighter than
dry air, so moist air is buoyant at equal temperature.
"""
virtual_temperature(T::Real, q::PhasePartition,
                    c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    float(T) * gas_constant_moist(q, c) / R_d(c)

"""
    density_temperature(T, q, c) -> K

Density temperature — virtual temperature *including condensate loading*:

    T_ρ = T (1 + q_vap/ε - q_tot)

Condensate adds mass without adding volume, so it is a negative buoyancy
term. This is the quantity that belongs in a buoyancy calculation; using
plain virtual temperature omits the weight of the cloud water and
systematically over-predicts updraught strength. At 3 g/kg of liquid the
loading is worth about 0.9 K of density temperature, comparable to the
excess that drives shallow convection in the first place.
"""
function density_temperature(T::Real, q::PhasePartition,
                             c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    ε = molmass_ratio(c)
    return float(T) * (1 + vapor(q) / ε - q.tot)
end

"Virtual potential temperature, `θ_v = T_v / Π`."
virtual_potential_temperature(T::Real, p::Real, q::PhasePartition,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    virtual_temperature(T, q, c) / exner_function(p, q, c)

"""
    density_potential_temperature(T, p, q, c) -> θ_ρ

Density potential temperature — the correct buoyancy variable for a
cloudy parcel, since it includes condensate loading.
"""
density_potential_temperature(T::Real, p::Real, q::PhasePartition,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    density_temperature(T, q, c) / exner_function(p, q, c)

"""
    equivalent_potential_temperature(T, p, q, c) -> θ_e

Equivalent potential temperature — the temperature a parcel would reach
if all its vapour condensed and it were brought to `p_ref`. Conserved
under *both* dry and moist adiabatic processes, which makes it the
standard tracer for air-mass identity.

Uses the Bolton (1980) formulation, accurate to ~0.1 K, with the
saturation temperature of the LCL computed from his eq. 21.
"""
function equivalent_potential_temperature(T::Real, p::Real, q::PhasePartition,
                                          c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T); pp = float(p)
    qv = max(vapor(q), 1e-12)
    r = qv / (1 - q.tot)                       # mixing ratio, kg/kg
    e = vapor_pressure(qv, pp, c) / 100        # hPa, as Bolton uses
    e = max(e, 1e-10)
    # Bolton (1980) eq. 21: temperature at the LCL
    T_L = 2840 / (3.5 * log(Tk) - log(e) - 4.805) + 55
    # Bolton (1980) eq. 38
    return Tk * (1000 / (pp / 100))^(0.2854 * (1 - 0.28 * r)) *
           exp((3376 / T_L - 2.54) * r * (1 + 0.81 * r))
end

"""
    wet_bulb_potential_temperature(T, p, q, c) -> θ_w

Wet-bulb potential temperature, obtained by inverting Bolton's `θ_e`
relation for the temperature a saturated parcel would have at `p_ref`.
Monotone in `θ_e`, so bisection is safe.
"""
function wet_bulb_potential_temperature(T::Real, p::Real, q::PhasePartition,
                                        c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    θe = equivalent_potential_temperature(T, p, q, c)
    f(Tw) = begin
        qs = saturation_specific_humidity(Tw, c.p_ref, c)
        equivalent_potential_temperature(Tw, c.p_ref, PhasePartition(qs, 0.0, 0.0), c) - θe
    end
    # The bracket must stay BELOW the boiling point at p_ref (373 K): a
    # saturated parcel above it has q_sat → 1, and θ_e diverges and then
    # folds back, so the residual is no longer monotone and bisection
    # converges to nonsense. θ_w never approaches boiling in any case —
    # it is bounded above by the parcel's own temperature.
    return _invert_monotone(f, 150.0, 350.0)
end

"""
    moist_static_energy(T, z, q, c) -> J kg⁻¹

Moist static energy `h = c_pm T + g z + L_v q_vap`.

Approximately conserved under adiabatic vertical motion *including*
condensation, which makes its drift a sharp diagnostic of whether a
scheme's thermodynamics are self-consistent. A scheme using a fitted
`e_s` with a different implicit `L_v` than it uses in its adjustment will
show a slow systematic drift here. It is still only *approximately*
conserved even with exact thermodynamics: it treats `c_pm` as fixed while
vapour becomes liquid. For the exact isobaric invariant see
[`moist_enthalpy`](@ref).
"""
moist_static_energy(T::Real, z::Real, q::PhasePartition,
                    c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    heat_capacity_moist(q, c) * float(T) + c.grav * float(z) +
    latent_heat_vapor(T, c) * vapor(q)

"""
    moist_enthalpy(T, q, c = DEFAULT_CONSTANTS) -> J kg⁻¹

Specific enthalpy of moist air *including its condensate*, referenced to
liquid water, and **exactly** consistent with the Kirchhoff latent heats
of this module:

    h = (c_pd q_d + c_l q_tot) T + L_v(T) q_v - L_f(T) q_ice

It is the quantity conserved by a reversible isobaric phase change, and
differs from [`moist_static_energy`](@ref) (at z = 0) by
`(c_pv - c_l) T q_v`, which is *not* constant under condensation — about
a quarter of `L_v Δq` near 285 K. Use `moist_static_energy` as the
familiar diagnostic, and this function wherever a scheme must conserve
energy exactly through condensation.

A useful identity follows directly: condensing `Δq` of vapour to liquid
at fixed `h`, `q_tot` and pressure warms the parcel by exactly

    c_pm(q_final) · ΔT = L_v(T_initial) · Δq ,

with `c_pm` = [`heat_capacity_moist`](@ref). It is what
[`isobaric_saturation_adjustment`](@ref) conserves, and
[`temperature_from_moist_enthalpy`](@ref) inverts it.
"""
function moist_enthalpy(T::Real, q::PhasePartition,
                        c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    return (c.cp_dry * (1 - q.tot) + c.cp_liquid * q.tot) * Tk +
           latent_heat_vapor(Tk, c) * vapor(q) -
           latent_heat_fusion(Tk, c) * q.ice
end

"""
    temperature_from_moist_enthalpy(h, q, c = DEFAULT_CONSTANTS; T_ref = c.T_triple) -> T

Temperature at which the parcel with phase partition `q` has moist
enthalpy `h` ([`moist_enthalpy`](@ref)), i.e. the exact inverse of
`moist_enthalpy` at a known partition.

With Kirchhoff latent heats (constant specific heats) `moist_enthalpy` is
exactly *linear* in `T` at fixed partition, and its slope is
[`heat_capacity_moist`](@ref)`(q)`:

    ∂h/∂T = c_pd q_d + c_l q_tot + (c_pv − c_l) q_v − (c_l − c_i) q_ice = c_pm(q),

so one linear step from any reference temperature is exact:

    T = T_ref + (h − h(T_ref, q)) / c_pm(q).

The result does not depend on `T_ref` except at round-off; passing the
pre-step temperature as `T_ref` keeps the answer closest to it in floating
point.
"""
function temperature_from_moist_enthalpy(h::Real, q::PhasePartition,
                                         c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                                         T_ref::Real = c.T_triple)
    Tr = float(T_ref)
    return Tr + (float(h) - moist_enthalpy(Tr, q, c)) / heat_capacity_moist(q, c)
end

"""
    liquid_water_static_energy(T, z, q, c) -> J kg⁻¹

Liquid-water static energy `s_l = c_pm T + g z - L_v q_liq - L_s q_ice`,
the condensate-referenced counterpart of moist static energy.
"""
liquid_water_static_energy(T::Real, z::Real, q::PhasePartition,
                           c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    heat_capacity_moist(q, c) * float(T) + c.grav * float(z) -
    latent_heat_vapor(T, c) * q.liq - latent_heat_sublim(T, c) * q.ice

# =========================================================================
# Saturation adjustment
# =========================================================================
#
# TWO OPERATIONS, TWO INVARIANTS (issue #82). "Saturation adjustment" names
# two different things, and they must not be confused:
#
#  (a) DIAGNOSIS — `saturation_adjustment(θ_li, p, q_tot)`. A model that
#      carries θ_li and q_tot as prognostic variables must, at every point
#      and every step, *recover* T and the vapour/condensate split that its
#      prognostic θ_li denotes. That is an inversion of the θ_li definition
#      at saturation, so the answer must reproduce θ_li exactly — anything
#      else would change the model's prognostic state just by reading it.
#      Also used for reversible parcel ascent at constant θ_li (CAPE, LFC)
#      and by the sub-grid cloud scheme (TurbulentMicrophysics).
#
#  (b) PROCESS — `isobaric_saturation_adjustment(T, p, q_v, q_c; q_r, q_i)`.
#      Condensation or evaporation *during* a step at fixed pressure is a
#      physical process, and the first law says it conserves the moist
#      enthalpy `moist_enthalpy` (`c_pm ΔT = L_v Δq` exactly). This
#      module's θ_li is not that invariant: its Exner exponent R_m/c_pm is
#      taken at the current partition and changes as vapour becomes
#      condensate, so holding θ_li fixed gives `c_pm ΔT/(L_v(T₀) Δq_c)` =
#      0.998 at 1000 hPa, 1.011 at 900 hPa and 1.063 at 600 hPa (285 K,
#      S = 1.1; measured, issue #82). With the exponent frozen at total
#      water, as in Bryan & Fritsch's exact form, it is 0.998 at every
#      pressure; the rest is the exponential approximation. Microphysics schemes (P3, fog) that
#      take T as their state use (b).
#
# Both saturate over the same `saturation_specific_humidity`; they differ
# only in which quantity is held fixed while doing it.

"""
    saturation_adjustment(θ_li, p, q_tot, c; ...) -> (T, PhasePartition)

**Diagnosis** of the thermodynamic state from the conserved variables:
given the prognostic `θ_li` and `q_tot`, find the temperature and the
liquid/ice split that put the parcel exactly at saturation (or leave it
subsaturated with no condensate) *and* reproduce `θ_li` exactly through
[`liquid_ice_potential_temperature`](@ref).

This is an inversion of [`liquid_ice_potential_temperature`](@ref), so
θ_li-exactness is the correct invariant here. It is **not** the right
operation for condensation *as a process* during an isobaric step: this
θ_li is only approximately conserved by phase change (its Exner exponent
follows the partition), and holding it fixed misstates latent heating by
+1.1 % at 900 hPa and +6.3 % at 600 hPa (issue #82). For that,
use [`isobaric_saturation_adjustment`](@ref), which conserves
[`moist_enthalpy`](@ref).

# Why this is bisection and not a fixed-point iteration

The obvious scheme — guess `T`, compute `q_liq = q_tot - q_sat(T)`,
recompute `T` from the latent heating, repeat — **does not converge**.
It oscillates, because condensing enough water to raise `T` also raises
`q_sat` past `q_tot`, which un-condenses all of it, which drops `T` back.
Measured on a marine stratocumulus sounding it cycles 281.9 → 285.2 →
281.9 → … indefinitely, and a naive implementation that caps the iteration
count silently returns zero liquid water — i.e. the cloud never forms and
nothing reports an error.

The residual solved here,

    f(T) = θ_li(T, p, partition(T)) - θ_li_target

is continuous and monotonically increasing in `T`, and the bracket
`[T_dry_at_saturation, T_all_condensed]` is guaranteed to contain the
root because condensate is bounded by `0 ≤ q_cond ≤ q_tot`. Bisection on
it cannot fail.

Returns `(T, PhasePartition)`.
"""
function saturation_adjustment(θ_li::Real, p::Real, q_tot::Real,
                               c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                               tol::Real = 1e-10, maxiter::Int = 100)
    qt = max(zero(float(q_tot)), float(q_tot))
    pp = float(p)

    # Unsaturated guess: no condensate at all.
    q_dry = PhasePartition(qt, 0.0, 0.0)
    T_dry = temperature_from_theta_li(θ_li, pp, q_dry, c)
    q_sat_dry = saturation_specific_humidity(T_dry, pp, c)
    if qt <= q_sat_dry
        return T_dry, q_dry                     # subsaturated: done
    end

    # Saturated. f is increasing in T; bracket by the two extremes of the
    # condensate range, which physically bound the answer.
    q_wet = phase_partition(T_dry, qt, qt, c)   # everything condensed
    T_wet = temperature_from_theta_li(θ_li, pp, q_wet, c)

    function residual(T)
        qs = saturation_specific_humidity(T, pp, c)
        qc = max(0.0, qt - qs)
        qc = min(qc, qt)
        qp = phase_partition(T, qt, qc, c)
        return liquid_ice_potential_temperature(T, pp, qp, c) - float(θ_li)
    end

    a, b = min(T_dry, T_wet), max(T_dry, T_wet)
    fa = residual(a)
    fb = residual(b)
    # Degenerate bracket (can happen when q_tot is tiny): fall back to the
    # dry solution rather than returning something unbracketed.
    if fa * fb > 0
        return T_dry, q_dry
    end
    T = a
    for _ in 1:maxiter
        T = 0.5 * (a + b)
        (b - a) < tol && break
        residual(T) < 0 ? (a = T) : (b = T)
    end
    qs = saturation_specific_humidity(T, pp, c)
    qc = min(max(0.0, qt - qs), qt)
    return T, phase_partition(T, qt, qc, c)
end

"""
    saturation_adjustment_ice(θ_li, p, q_tot, c) -> (T, PhasePartition)

As [`saturation_adjustment`](@ref) but saturating with respect to **ice**
throughout, for the cirrus regime where no liquid is present. Like it,
this is a **diagnosis** from `θ_li` (exact inversion), not an
enthalpy-conserving process step.
"""
function saturation_adjustment_ice(θ_li::Real, p::Real, q_tot::Real,
                                   c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                                   tol::Real = 1e-10, maxiter::Int = 100)
    qt = max(zero(float(q_tot)), float(q_tot))
    pp = float(p)
    q_dry = PhasePartition(qt, 0.0, 0.0)
    T_dry = temperature_from_theta_li(θ_li, pp, q_dry, c)
    if qt <= saturation_specific_humidity_ice(T_dry, pp, c)
        return T_dry, q_dry
    end
    q_wet = PhasePartition(qt, 0.0, qt)
    T_wet = temperature_from_theta_li(θ_li, pp, q_wet, c)
    function residual(T)
        qs = saturation_specific_humidity_ice(T, pp, c)
        qi = min(max(0.0, qt - qs), qt)
        return liquid_ice_potential_temperature(T, pp,
                   PhasePartition(qt, 0.0, qi), c) - float(θ_li)
    end
    a, b = min(T_dry, T_wet), max(T_dry, T_wet)
    residual(a) * residual(b) > 0 && return T_dry, q_dry
    T = a
    for _ in 1:maxiter
        T = 0.5 * (a + b)
        (b - a) < tol && break
        residual(T) < 0 ? (a = T) : (b = T)
    end
    qi = min(max(0.0, qt - saturation_specific_humidity_ice(T, pp, c)), qt)
    return T, PhasePartition(qt, 0.0, qi)
end

"""
    isobaric_saturation_adjustment(T, p, q_v, q_c, c = DEFAULT_CONSTANTS;
                                   q_r = 0, q_i = 0, tol = 1e-14, maxiter = 100)
        -> (Δq, T)

**Process** saturation adjustment: condensation of vapour to cloud liquid,
or evaporation of cloud liquid, at constant pressure `p` and constant
moist enthalpy ([`moist_enthalpy`](@ref)) — the first-law invariant of an
isobaric phase change. Returns the amount condensed `Δq` (kg kg⁻¹;
negative when cloud evaporates) and the adjusted temperature `T`.

Only vapour `q_v` and cloud liquid `q_c` exchange mass. Rain `q_r` and
ice `q_i` are **non-adjusting**: they do not change, but they are part of
the parcel, so they enter its heat capacity `c_pm` (and `q_i` its
enthalpy through `L_f`). Leaving them out overstates the condensational
`ΔT` by about 1.5 % per 5 g kg⁻¹ of rain and 0.5 % per 5 g kg⁻¹ of ice.

The target is saturation over **liquid** (supercooled below 0 °C — cloud
liquid becomes ice only by explicit microphysics); if even evaporating all
of `q_c` leaves the air subsaturated, all of it evaporates. Subsaturated
air without cloud is returned unchanged. The adjusted state satisfies,
exactly up to round-off,

    h(T, q_v − Δq, q_c + Δq, q_r, q_i) = h(T₀, q_v, q_c, q_r, q_i),

equivalently `c_pm(q_final) ΔT = L_v(T₀) Δq`, and `q_v + q_c` is conserved.
Saturation is reached to `tol` in *absolute* specific humidity
(`|q_v − q_s| < 1e-14` kg kg⁻¹ by default), so relative to `q_s` it is
~1e-12 in warm air but up to ~2e-8 near 190 K where `q_s` ~ 1e-6
(measured by random sampling over 190-335 K, 200-1050 hPa).

# Algorithm

`h` is linear in `T` at a fixed partition
([`temperature_from_moist_enthalpy`](@ref)), so `T(Δ)` is explicit and
the problem is the scalar root of `g(Δ) = (q_v − Δ) − q_s(T(Δ), p)`,
which is strictly decreasing (condensing removes vapour *and* warms, which
raises `q_s`) on the bracket `Δ ∈ [−q_c, q_v]`. A safeguarded Newton
iteration (Clausius–Clapeyron slope) inside the bracket cannot fail; the
convergence test is made on the current iterate *before* the update, so a
root hit exactly is kept rather than replaced by the bracket midpoint
(a bug found in review of #85, which had returned half the condensate for
some cold inputs).

Compare [`saturation_adjustment`](@ref), which *diagnoses* `(T, q_c)` from
a prognostic `θ_li` and conserves θ_li instead (see the section comment).

References: Emanuel (1994) §4.5; issue #82.
"""
function isobaric_saturation_adjustment(T::Real, p::Real, q_v::Real, q_c::Real,
                                        c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                                        q_r::Real = 0.0, q_i::Real = 0.0,
                                        tol::Real = 1e-14, maxiter::Int = 100)
    T0 = float(T); pp = float(p)
    qv0 = max(float(q_v), 0.0); qc0 = max(float(q_c), 0.0)
    qr = max(float(q_r), 0.0);  qi = max(float(q_i), 0.0)
    qt = qv0 + qc0 + qr + qi
    # Partition after condensing Δ: rain counts as liquid, ice as ice.
    part(Δ) = PhasePartition(qt, qc0 + Δ + qr, qi)
    h0 = moist_enthalpy(T0, part(0.0), c)
    Tof(Δ) = temperature_from_moist_enthalpy(h0, part(Δ), c; T_ref = T0)
    g(Δ)   = (qv0 - Δ) - saturation_specific_humidity(Tof(Δ), pp, c)

    g0 = g(0.0)
    g0 == 0 && return (Δq = 0.0, T = T0)
    if g0 < 0                      # subsaturated: can only evaporate cloud
        qc0 == 0 && return (Δq = 0.0, T = T0)
        lo, hi = -qc0, 0.0
        g(lo) <= 0 && return (Δq = lo, T = Tof(lo))   # all cloud evaporates
    else                           # supersaturated: condense
        lo, hi = 0.0, qv0
    end
    # Safeguarded Newton on the bracket [lo, hi] with g(lo) > 0 > g(hi).
    Rv = R_v(c)
    Δ = 0.5 * (lo + hi)
    for _ in 1:maxiter
        gΔ = g(Δ)
        # Converged: keep THIS Δ (test before the update — see docstring).
        abs(gΔ) < tol && break
        gΔ > 0 ? (lo = Δ) : (hi = Δ)
        (hi - lo) < tol && break
        Tn = Tof(Δ)
        qs = qv0 - Δ - gΔ
        # dq_s/dT from Clausius-Clapeyron, dT/dΔ ≈ L_v / c_pm.
        Lv = latent_heat_vapor(Tn, c)
        dg = -1 - qs * Lv / (Rv * Tn^2) * Lv / heat_capacity_moist(part(Δ), c)
        Δn = Δ - gΔ / dg
        Δ  = (lo < Δn < hi) ? Δn : 0.5 * (lo + hi)
    end
    return (Δq = Δ, T = Tof(Δ))
end

# =========================================================================
# Lapse rates, parcel theory
# =========================================================================

"Dry adiabatic lapse rate `g/c_pm` (K m⁻¹); ≈ 9.76e-3 for dry air."
dry_adiabatic_lapse_rate(q::PhasePartition = PhasePartition(0.0),
                         c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    c.grav / heat_capacity_moist(q, c)

"""
    moist_adiabatic_lapse_rate(T, p, c) -> K m⁻¹

Saturated (pseudo)adiabatic lapse rate,

    Γ_m = g (1 + L_v r_s /(R_d T)) / (c_pd + L_v² r_s ε /(R_d T²))

About 4e-3 K/m in the warm tropics (latent heat release offsets roughly
half the adiabatic cooling) rising toward the dry value 9.8e-3 K/m in
cold air, where there is little vapour left to condense. That variation
is why the tropical troposphere is so much closer to neutral than the
polar one.
"""
function moist_adiabatic_lapse_rate(T::Real, p::Real,
                                    c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    return _moist_adiabatic_lapse_rate(Tk, saturation_mixing_ratio(Tk, p, c),
                                       latent_heat_vapor(Tk, c), c)
end

# Γ_m from r_s and L_v at T (shared with saturated_buoyancy_frequency_squared)
function _moist_adiabatic_lapse_rate(Tk::Real, rs::Real, Lv::Real, c::ThermodynamicConstants)
    Rd = R_d(c); ε = molmass_ratio(c)
    num = 1 + Lv * rs / (Rd * Tk)
    den = c.cp_dry + Lv^2 * rs * ε / (Rd * Tk^2)
    return c.grav * num / den
end

"""
    lifting_condensation_level(T, p, q_v, c) -> (T_lcl, p_lcl)

Temperature and pressure at which a parcel lifted dry-adiabatically first
saturates. Solved by bisection on the residual `q_v - q_sat` along the dry
adiabat, which is monotone in pressure, rather than by an empirical fit —
the fits are good near the surface and drift aloft.
"""
function lifting_condensation_level(T::Real, p::Real, q_v::Real,
                                    c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T); pp = float(p); qv = float(q_v)
    q = PhasePartition(qv, 0.0, 0.0)
    θ = potential_temperature(Tk, pp, q, c)
    # Along a dry adiabat q_v is fixed, so RH rises as p falls.
    f(pl) = begin
        Tl = temperature_from_theta(θ, pl, q, c)
        qv - saturation_specific_humidity(Tl, pl, c)
    end
    f(pp) >= 0 && return (Tk, pp)              # already saturated
    lo, hi = 1.0, pp
    for _ in 1:200
        m = 0.5 * (lo + hi)
        (hi - lo) < 1e-6 && break
        f(m) < 0 ? (hi = m) : (lo = m)
    end
    p_lcl = 0.5 * (lo + hi)
    return (temperature_from_theta(θ, p_lcl, q, c), p_lcl)
end

"""
    buoyancy(T, p, q, T_env, p_env, q_env, c) -> m s⁻²

Buoyancy of a parcel relative to its environment, using **density**
potential temperature so that condensate loading is included:

    b = g (θ_ρ,parcel - θ_ρ,env) / θ_ρ,env
"""
function buoyancy(T::Real, p::Real, q::PhasePartition,
                  T_env::Real, p_env::Real, q_env::PhasePartition,
                  c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    θρ   = density_potential_temperature(T, p, q, c)
    θρ_e = density_potential_temperature(T_env, p_env, q_env, c)
    return c.grav * (θρ - θρ_e) / θρ_e
end

"""
    brunt_vaisala_frequency_moist(θρ, dθρ_dz, c) -> N² in s⁻²

Squared buoyancy frequency from the density-potential-temperature
gradient. Negative means statically unstable.
"""
brunt_vaisala_frequency_moist(θρ::Real, dθρ_dz::Real,
                              c::ThermodynamicConstants = DEFAULT_CONSTANTS) =
    c.grav * float(dθρ_dz) / float(θρ)

"""
    saturated_buoyancy_frequency_squared(θ, p, q_v, q_c, dθ_dz, dq_v_dz, dq_w_dz,
                                         c = DEFAULT_CONSTANTS; θ_ref = θ, g = c.grav) -> N² in s⁻²

Squared buoyancy frequency of a **saturated** cell: the restoring force
on a saturated parcel displaced along the moist adiabat (reversible,
total water ``q_w = q_v + q_c`` conserved, condensate carried along),
measured against the environment's gradients. With the buoyancy variable
of the dynamics, the density potential temperature
``θ_ρ = θ(1 + 0.61 q_v - q_c)``, linearised about the cell,

```math
N^2 = \\frac{g}{θ_{\\rm ref}}\\Bigl[(1 + 0.61 q_v - q_c)
      \\Bigl(\\frac{∂θ}{∂z} - \\frac{Γ_d - Γ_m}{Π}\\Bigr)
      + θ\\Bigl(1.61\\Bigl(\\frac{∂q_v}{∂z} - \\frac{dq_s}{dz}\\Big|_{\\rm ad}\\Bigr)
      - \\frac{∂q_w}{∂z}\\Bigr)\\Bigr],
```

where ``(Γ_d - Γ_m)/Π`` is the parcel's θ gradient on the moist adiabat
(``T = θΠ``, ``θ ∂Π/∂z = -Γ_d`` in a hydrostatic column) and
``dq_s/dz|_{\\rm ad}`` the change of its saturation humidity along it.
In a **saturated environment** ``∂q_v/∂z - dq_s/dz|_{\\rm ad} =
(∂q_s/∂T)(∂T/∂z + Γ_m)``, and the expression is the saturated
Brunt–Väisälä frequency of Durran & Klemp (1982, eq. 36),

```math
N_m^2 = \\frac{g}{T}\\Bigl(\\frac{∂T}{∂z} + Γ_m\\Bigr)
        \\Bigl(1 + \\frac{L_v r_s}{R_d T}\\Bigr) - \\frac{g}{1 + q_w}\\frac{∂q_w}{∂z},
```

to ``O(q)`` (with ``g/T → g/θ_{\\rm ref}`` per unit θ, the Boussinesq form):
the factor ``1 + L_v r_s/(R_d T)`` is the virtual effect of the vapour
that the displaced parcel and its saturated environment exchange. At a
cloud **edge**, where the environment above or below is unsaturated,
Durran & Klemp's form would still assume a saturated environment and
drop the virtual effect of the humidity jump; there this form keeps it
(``∂q_v/∂z`` is the environment's), so a saturated parcel lifted into
drier air of the same θ is buoyant, as a parcel computation says.

`θ` (K), `p` (Pa), `q_v`, `q_c` (kg kg⁻¹) are the cell's; ``T = θΠ(p)``
with the dry Exner function. ``Γ_m`` is [`moist_adiabatic_lapse_rate`](@ref),
``Γ_d`` [`dry_adiabatic_lapse_rate`](@ref), ``q_s``
[`saturation_specific_humidity`](@ref) (``dq_s/dz|_{\\rm ad}`` from its
derivatives in T and p, hydrostatic with ``R_d T``): one
source of truth with the rest of the contract. `θ_ref` is the reference
temperature of a Boussinesq core (``θ_0``; default θ). The factor 1.61
is ``1 + 0.61``, the 0.61 of the cores' θ_v. The result vanishes on a
moist adiabat of constant total water, is ``(g/θ_{\\rm ref})∂θ/∂z`` for
``q → 0``, and is below the dry ``N^2`` of the same θ gradient in a stable
saturated layer, because condensation in a rising parcel offsets part of
its adiabatic cooling.

Reference: Durran, D. R. & Klemp, J. B. (1982). On the effects of
moisture on the Brunt–Väisälä frequency. *J. Atmos. Sci.*, 39,
2152-2158.
"""
function saturated_buoyancy_frequency_squared(θ::Real, p::Real, q_v::Real, q_c::Real,
                                              dθ_dz::Real, dq_v_dz::Real, dq_w_dz::Real,
                                              c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                                              θ_ref::Real = θ, g::Real = c.grav)
    Π  = exner_function(p, PhasePartition(0.0), c)
    T  = float(θ) * Π
    pf = float(p)
    es = saturation_vapor_pressure(T, c)
    Lv = latent_heat_vapor(T, c)
    ε  = molmass_ratio(c)
    Γd = dry_adiabatic_lapse_rate(PhasePartition(0.0), c)
    Γm = _moist_adiabatic_lapse_rate(T, _saturation_mixing_ratio(es, pf, c), Lv, c)
    # the parcel's saturation humidity along the moist adiabat:
    # dq_s/dz = ∂q_s/∂T (−Γ_m) + ∂q_s/∂p (−g p/(R_d T)), with
    # q_s = ε e_s/(p − (1−ε) e_s) and de_s/dT = e_s L_v/(R_v T²) (exact for
    # the Kirchhoff e_s of this module)
    den = pf - (1 - ε) * es
    qs  = _saturation_specific_humidity(T, es, pf, c)
    dqs_ad = -(ε * pf / den^2) * es * Lv / (R_v(c) * T^2) * Γm +
              (qs / den) * c.grav * pf / (R_d(c) * T)
    return (g / θ_ref) * ((1 + 0.61 * q_v - q_c) * (dθ_dz - (Γd - Γm) / Π) +
                          θ * (1.61 * (dq_v_dz - dqs_ad) - dq_w_dz))
end

"""
    level_of_free_convection(T_p, p_p, q_p, sounding, c) -> p_lfc or NaN

Pressure of the level of free convection for a parcel lifted from
`(T_p, p_p, q_p)` through `sounding`, a vector of `(p, T, q_v)` ordered
from the surface upward. Returns `NaN` if the parcel never becomes
positively buoyant.
"""
function level_of_free_convection(T_p::Real, p_p::Real, q_p::Real,
                                  sounding::AbstractVector,
                                  c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    θli = liquid_ice_potential_temperature(T_p, p_p, PhasePartition(q_p), c)
    for (pe, Te, qe) in sounding
        pe >= p_p && continue
        Tp, qp = saturation_adjustment(θli, pe, q_p, c)
        b = buoyancy(Tp, pe, qp, Te, pe, PhasePartition(qe), c)
        b > 0 && return float(pe)
    end
    return NaN
end

"""
    cape_cin(T_p, p_p, q_p, sounding, c) -> (CAPE, CIN) in J kg⁻¹

Convective available potential energy and convective inhibition for a
parcel lifted through `sounding` (a vector of `(p, T, q_v)` from the
surface up). The parcel is lifted reversibly — condensate is retained and
loads the parcel — which gives a lower, more realistic CAPE than the
pseudo-adiabatic assumption that precipitates everything out instantly.
"""
function cape_cin(T_p::Real, p_p::Real, q_p::Real,
                  sounding::AbstractVector,
                  c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    θli = liquid_ice_potential_temperature(T_p, p_p, PhasePartition(q_p), c)
    Rd = R_d(c)
    cape = 0.0; cin = 0.0
    prev_p = float(p_p); prev_b = 0.0
    for (pe, Te, qe) in sounding
        pe >= p_p && continue
        Tp, qp = saturation_adjustment(θli, pe, q_p, c)
        b = buoyancy(Tp, pe, qp, Te, pe, PhasePartition(qe), c)
        # dz = -(R_d T_v / g) dlnp  ⇒  b dz = -R_d T_v b/g dlnp
        Tv = virtual_temperature(Te, PhasePartition(qe), c)
        dlnp = log(prev_p / pe)
        contrib = 0.5 * (b + prev_b) * Rd * Tv / c.grav * dlnp
        contrib > 0 ? (cape += contrib) : (cin += contrib)
        prev_p = float(pe); prev_b = b
    end
    return (cape, cin)
end


"""
    effective_radius(population, args...)

Effective radius of a size distribution: the ratio of its third to its
second moment, `⟨r³⟩/⟨r²⟩`.

Declared here with no method, so the modules that own a size distribution
can each add one without importing each other — the same pattern
`RadiationCore` uses for `surface_albedo` and `material_albedo`.

It is the *same* quantity in every case, which is why it is one name: the
3/2 moment ratio is what controls how much light a population of spheres
scatters per unit mass, so it is the radius radiation actually sees.

Implemented by:
  * `AerosolOptics.effective_radius` for an aerosol size distribution.
  * `Superdroplet.effective_radius` for a superdroplet ensemble.
"""
function effective_radius end

# =========================================================================
# Köhler theory — equilibrium over a curved solution droplet
# =========================================================================
#
# Everything above concerns saturation over a PLANE surface of PURE water.
# A cloud droplet is neither: it is curved, which raises its equilibrium
# vapour pressure (Kelvin), and it contains dissolved solute, which lowers
# it (Raoult). Köhler theory is the competition between the two, and it is
# what decides whether a haze particle stays a haze particle or becomes a
# cloud droplet.
#
# It lives HERE, in the Era IV thermodynamic contract, for the same reason
# the plane-surface saturation does: several modules need it — the
# superdroplet solver (Phase 41), the activation parameterisation (Phase
# 42), and the hygroscopic-growth code in `AerosolOptics` (Phase 37) — and
# if each carries its own copy they drift. Era III already taught that
# lesson with the urban material tables.
#
# TWO PARAMETERISATIONS, ONE PHYSICS. The literature writes the solute
# term two ways:
#
#   classical:  s = A/r − B/r³            B is the "Raoult B", m³
#   kappa:      s = (r³ − r_d³) / (r³ − r_d³(1−κ)) · exp(A/r) − 1
#
# and they are the same thing to leading order with `B = κ r_d³`. The
# κ form (Petters & Kreidenweis 2007) is used as canonical because it
# stays correct as r → r_d, where the classical expansion does not, and
# because κ is what is actually measured. `kappa_from_solute` and
# `solute_from_kappa` convert, so a caller holding solute mass and a
# caller holding κ reach the same numbers rather than two similar ones.
#
# References
#   * Köhler, H. (1936). Trans. Faraday Soc., 32, 1152-1161.
#   * Petters, M. D. & Kreidenweis, S. M. (2007). ACP, 7, 1961-1971.

"""
    surface_tension_water(T) -> N m⁻¹

Surface tension of the water/air interface. Linear in temperature over
the atmospheric range (Pruppacher & Klett eq. 5-12): 0.0756 at 0 °C
falling to 0.0728 at 20 °C. The temperature dependence is small but it
enters the Kelvin term directly.
"""
surface_tension_water(T::Real) = 0.0761 - 1.55e-4 * (float(T) - 273.15)

"""
    kelvin_coefficient(T, c = DEFAULT_CONSTANTS) -> m

The Kelvin (curvature) coefficient `A = 2σ M_w / (R T ρ_w)`, in metres,
such that the curvature term in the Köhler equation is `exp(A/r)`.

About 1.1 nm at 293 K. Small — but divided by a 10 nm dry radius it is a
supersaturation of order 10 %, which is why the smallest particles never
activate.
"""
function kelvin_coefficient(T::Real, c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Tk = float(T)
    σ = surface_tension_water(Tk)
    ρ_w = 1000.0
    return 2 * σ / (R_v(c) * Tk * ρ_w)
end

"Raoult `B` (m³) from κ and dry radius: `B = κ r_d³`."
solute_from_kappa(r_dry::Real, κ::Real) = float(κ) * float(r_dry)^3
"κ from a Raoult `B` and dry radius."
kappa_from_solute(r_dry::Real, B::Real) = float(B) / float(r_dry)^3

"""
    kohler_supersaturation(r, r_dry, κ, T, c = DEFAULT_CONSTANTS) -> s

Equilibrium supersaturation `s = S − 1` over a solution droplet of wet
radius `r` grown on a dry particle of radius `r_dry` (both metres) with
hygroscopicity `κ`.

Uses the Petters & Kreidenweis (2007) form,

```math
S = \\frac{r^3 - r_d^3}{r^3 - r_d^3 (1 - \\kappa)} \\exp\\left(\\frac{A}{r}\\right)
```

which remains correct as `r → r_d` where the classical `A/r − B/r³`
expansion diverges from the truth.
"""
function kohler_supersaturation(r::Real, r_dry::Real, κ::Real, T::Real,
                                c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    rr = float(r); rd = float(r_dry); kap = float(κ)
    rr > rd || throw(DomainError(r, "kohler_supersaturation: need r > r_dry"))
    A = kelvin_coefficient(T, c)
    r3 = rr^3; rd3 = rd^3
    kap <= 0 && return expm1(A / rr)          # insoluble: Kelvin only
    return (r3 - rd3) / (r3 - rd3 * (1 - kap)) * exp(A / rr) - 1
end

"""
    kohler_critical_point(r_dry, κ, T, c) -> (r_crit, s_crit)

Critical radius and critical supersaturation — the maximum of the Köhler
curve. A droplet held above `s_crit` grows without bound (it activates);
below it, it sits at a stable haze equilibrium.

For κ not too small the classical closed form
`r_c = sqrt(3B/A)`, `s_c = sqrt(4A³/27B)` with `B = κ r_d³` is exact
enough to seed a refinement; the maximum is then located by golden-section
search on the full κ-Köhler expression, so the returned point is the
maximum of the function actually used elsewhere rather than of its
approximation. For κ → 0 there is no maximum (the curve is monotone
decreasing) and the particle never activates; `s_crit = A/r_dry` is
returned as the effective barrier.
"""
function kohler_critical_point(r_dry::Real, κ::Real, T::Real,
                               c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    rd = float(r_dry); kap = float(κ)
    rd > 0 || throw(DomainError(r_dry, "kohler_critical_point: r_dry must be > 0"))
    A = kelvin_coefficient(T, c)
    kap <= 1e-9 && return (rd, expm1(A / rd))
    B = solute_from_kappa(rd, kap)
    r0 = sqrt(3B / A)                          # classical estimate
    # Golden-section maximisation of the exact κ-Köhler curve, bracketed
    # generously around the classical estimate.
    f(r) = kohler_supersaturation(r, rd, kap, T, c)
    # The bracket must straddle the maximum AND stay above the dry radius.
    # For nearly insoluble particles the classical estimate r0 = sqrt(3B/A)
    # collapses below r_dry, which inverts `[r0/8, r0*8]` and makes the
    # search sample r < r_dry, where the Köhler expression is undefined.
    a = rd * (1 + 1e-9)
    b = max(r0 * 8, rd * 100)
    φ = (sqrt(5.0) - 1) / 2
    x1 = b - φ * (b - a); x2 = a + φ * (b - a)
    f1 = f(x1); f2 = f(x2)
    for _ in 1:200
        if f1 < f2
            a = x1; x1 = x2; f1 = f2
            x2 = a + φ * (b - a); f2 = f(x2)
        else
            b = x2; x2 = x1; f2 = f1
            x1 = b - φ * (b - a); f1 = f(x1)
        end
        (b - a) < 1e-12 * max(1.0, b) && break
    end
    rc = 0.5 * (a + b)
    return (rc, f(rc))
end

"""
    kohler_equilibrium_radius(s, r_dry, κ, T, c) -> r

Stable equilibrium wet radius of a haze particle held at subcritical
supersaturation `s`. Bisection on the *rising* branch of the Köhler curve
(below the critical radius), which is the branch that is stable: a
perturbation there is restored, whereas beyond `r_crit` it runs away.

Returns `NaN` if `s` exceeds `s_crit`, since no equilibrium exists — the
particle activates.
"""
function kohler_equilibrium_radius(s::Real, r_dry::Real, κ::Real, T::Real,
                                   c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    rd = float(r_dry); kap = float(κ); ss = float(s)
    rc, sc = kohler_critical_point(rd, kap, T, c)
    ss >= sc && return NaN
    lo = rd * (1 + 1e-12); hi = rc
    for _ in 1:300
        mid = 0.5 * (lo + hi)
        (hi - lo) < 1e-16 * max(1.0, hi) && break
        kohler_supersaturation(mid, rd, kap, T, c) < ss ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

end # module
