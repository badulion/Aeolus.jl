"""
    BuildingEnergy

Phase 50 — the building interior behind the urban facets: a prognostic
indoor air temperature coupled to the thermal mass of the envelope and
of the building's internal structure, with ventilation and infiltration.

The zone is an RC network whose capacitances are *resolved*:

  * **Air node** — the zone air, heat capacity ``C_a = ρ c_p V`` (ρ and
    ``c_p`` from `MoistThermodynamics`), advanced implicitly.
  * **Envelope** — walls, roofs and floors are `UrbanSurface`
    conduction columns, reused as they are (no copy of the solver):
    either a bare [`EnvelopeElement`](@ref) (outside air with a sol-air
    source, the ground, or an adiabatic party wall) or a
    [`FacetElement`](@ref) wrapping a full `UrbanFacet` whose outer face
    is closed by the surface energy balance of `SurfaceExchange`.
  * **Internal mass** — floors, partitions and furniture as
    [`InternalMass`](@ref) columns exposed to the room on one or both
    faces.
  * **Massless conductances** — glazing (a U-value times area; glazing
    physics is issue #27), infiltration and mechanical ventilation with
    optional heat recovery.

Interior surfaces exchange with the air node through the combined
(convective + radiative) ISO 6946 surface coefficient ``h = 1/R_{si}``,
and receive the radiative part of internal gains and the transmitted
solar gain as an imposed flux. [`step_zone!`](@ref) is implicit in the
air temperature and conserves the zone's heat content exactly.

Occupancy (issue #33): [`WeeklySchedule`](@ref)s of piecewise-constant
[`DayProfile`](@ref)s drive [`InternalGains`](@ref) (people, lighting,
equipment, after the EnergyPlus objects) and a [`ThermostatSchedule`](@ref);
the DOE medium-office reference building's schedules are
[`DOE_MEDIUM_OFFICE_SCHEDULES`](@ref).

For reference the module also carries the ISO 13790 simple hourly
5R1C model ([`ISO13790Zone`](@ref)), with a mapping
([`iso13790_zone`](@ref)) that derives its parameters from the same
construction.

# Sign conventions

All zone budget terms ([`ZoneBudget`](@ref)) are in W and positive when
they *add* heat to the zone. Column fluxes follow `UrbanSurface`:
positive into the column.

# References

  * ISO 13790:2008. *Energy performance of buildings — Calculation of
    energy use for space heating and cooling.* §7.2.2 and Annex C (the
    simple hourly 5R1C method), §9.3 (ventilation heat transfer), §12.3
    (effective mass area and internal heat capacity).
  * ISO 13786:2007. *Thermal performance of building components —
    Dynamic thermal characteristics — Calculation methods.* Annex A
    (simplified estimate of the internal areal heat capacity).
  * ISO 6946:2017, Table 7 (interior surface resistances).
  * Bueno, B., Norford, L., Pigeon, G. & Britter, R. (2011). Combining a
    detailed building energy model with a physically-based urban canopy
    model. *Boundary-Layer Meteorol.*, 140, 471-489 (TEB-BEM: the zone
    air node coupled to TEB wall and roof columns, with internal mass).
  * Incropera / Bergman et al. (2011), *Fundamentals of Heat and Mass
    Transfer*, 7th ed., §5.2 (lumped capacitance) and §5.5 (plane wall
    with convection, ``β \\tan β = Bi``), used in the tests.
"""
module BuildingEnergy

using ..SurfaceExchange: SurfaceForcing, SurfaceFluxes, step_surface!
using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS,
                             PhasePartition, air_density, heat_capacity_moist,
                             saturation_specific_humidity, relative_humidity,
                             latent_heat_vapor, wet_bulb_temperature,
                             specific_humidity_from_relative_humidity
using ..UrbanSurface: UrbanSurface, ConductionColumn, ColumnBoundary, WallLayer,
                      UrbanFacet, step_conduction!, steady_conduction!, face_temperature,
                      column_heat_content, thermal_resistance,
                      fixed_temperature_boundary, heat_flux_boundary, convective_boundary,
                      set_interior_boundary!, ISO6946_SURFACE_RESISTANCE,
                      volumetric_heat_capacity, UrbanFacetScene,
                      example_wall_layers, example_roof_layers
using ..Fields: interior_view

export AbstractZoneElement, EnvelopeElement, FacetElement, InternalMass, steady_envelope!,
       BuildingZone, ZoneForcing, ZoneBudget, step_zone!, facet_fluxes,
       zone_air_temperature, zone_heat_content, zone_heat_loss_coefficient,
       zone_mean_radiant_temperature, zone_operative_temperature,
       air_heat_capacity, ventilation_conductance, element_transmittance,
       iso13786_areal_heat_capacity,
       ISO13790Zone, iso13790_zone, step_iso13790!, iso13790_ideal_load!,
       zone_specific_humidity, zone_relative_humidity,
       BiquadraticCurve, DXCoolingCurves, DX_RATING_POINT,
       DX_CURVES_WINDOW_AC, DX_CURVES_PACKAGED_AC,
       IdealLoadsHVAC, HVACBudget, condenser_inlet_temperature,
       cooling_cop, available_cooling_capacity,
       DayProfile, WeeklySchedule, schedule_value, schedule_mean, full_load_hours,
       occupant_sensible_heat, InternalGains, InternalGainRates, internal_gains,
       ThermostatSchedule, scheduled_hvac, DOE_MEDIUM_OFFICE_SCHEDULES,
       office_internal_gains, office_thermostat,
       BuildingHeatRelease, building_heat_release, energy_consumption, total_heat_release,
       anthropogenic_sensible_heat, anthropogenic_latent_heat,
       step_buildings!, building_facets, scene_building_zone, anthropogenic_heat_maps,
       traffic_heat_flux, add_traffic_heat!,
       CityBuildings, city_buildings, step_city_buildings!, apply_building_heat!, CITY_BUILDING_L_V

const _H_WALL = 1 / ISO6946_SURFACE_RESISTANCE.inside_horizontal   # 7.69 W m⁻² K⁻¹
const _H_OUT  = 1 / ISO6946_SURFACE_RESISTANCE.outside             # 25 W m⁻² K⁻¹

# =========================================================================
# 1. Zone elements
# =========================================================================

"""
    AbstractZoneElement

A heat-storing surface of a zone: [`EnvelopeElement`](@ref),
[`FacetElement`](@ref) or [`InternalMass`](@ref). Each wraps an
`UrbanSurface.ConductionColumn` and exchanges with the zone air node
through its interior surface coefficient.
"""
abstract type AbstractZoneElement end

# Work arrays for the affine (two-solve) step, see `_affine_step!`.
struct _AffineWork{T}
    T_old::Vector{T}
    T_a::Vector{T}
    dT::Vector{T}
end
_AffineWork(n::Integer, ::Type{T}) where {T} = _AffineWork{T}(zeros(T, n), zeros(T, n), zeros(T, n))

const _EXTERIORS = (:air, :ground, :adiabatic)

"""
    EnvelopeElement(layers, area; exterior = :air, h_inside = 1/0.13,
                    h_outside = 1/0.04, absorptance = 0, T_ground = 283.15,
                    T = 293.15, θ = 0.5, name = :envelope)

An opaque envelope element of `area` m² — a `ConductionColumn` through
`layers` (**outside first**, as in `UrbanSurface`) whose inner face
exchanges with the zone air through `h_inside` (W m⁻² K⁻¹; the default
is the ISO 6946 wall value ``1/R_{si} = 1/0.13``; use `1/0.10` for a
ceiling, `1/0.17` for a floor). The outer face is set by `exterior`:

| `exterior`   | outer boundary |
|:-------------|:---------------|
| `:air`       | `ColumnBoundary(T_out, h_outside, absorptance·I)`: outdoor air through ``1/R_{se}`` plus absorbed irradiance ``I`` (the sol-air form) |
| `:ground`    | `fixed_temperature_boundary(T_ground)`: the column ends in ground at `T_ground` |
| `:adiabatic` | `heat_flux_boundary(0)`: a party wall or floor to an identical neighbour |

For an envelope whose outer face should see the full surface energy
balance (radiation, Monin–Obukhov exchange) use a [`FacetElement`](@ref).
`T_ground` can be changed between steps (`el.T_ground = …`).
"""
mutable struct EnvelopeElement{T<:AbstractFloat} <: AbstractZoneElement
    name::Symbol
    column::ConductionColumn{T}
    area::T
    h_inside::T
    exterior::Symbol
    h_outside::T
    absorptance::T
    T_ground::T
    q_inside::T      # last step: imposed radiative flux on the inner face, W m⁻²
    F_outer::T       # last step: flux into the column at the outer face, W m⁻²
    Q_air::T         # last step: heat from the inner face to the air, W m⁻²
    _F::NTuple{4,T}  # (F_outer, dF_outer, F_inner, dF_inner) of the pending step
    _w::_AffineWork{T}
end

function EnvelopeElement(layers::AbstractVector{<:WallLayer}, area::Real;
                         exterior::Symbol = :air, h_inside::Real = _H_WALL,
                         h_outside::Real = _H_OUT, absorptance::Real = 0.0,
                         T_ground::Real = 283.15, T = 293.15, θ::Real = 0.5,
                         name::Symbol = :envelope)
    area > 0 || throw(ArgumentError("EnvelopeElement: area must be positive"))
    exterior in _EXTERIORS ||
        throw(ArgumentError("EnvelopeElement: exterior must be one of $_EXTERIORS"))
    (0 < h_inside < Inf) || throw(ArgumentError("EnvelopeElement: h_inside must be positive and finite"))
    (0 < h_outside) || throw(ArgumentError("EnvelopeElement: h_outside must be positive"))
    (0 <= absorptance <= 1) || throw(ArgumentError("EnvelopeElement: absorptance must lie in [0, 1]"))
    col = ConductionColumn(layers; T = T, θ = θ)
    FT = eltype(col.T)
    z = zero(FT)
    return EnvelopeElement{FT}(name, col, FT(area), FT(h_inside), exterior, FT(h_outside),
                               FT(absorptance), FT(T_ground), z, z, z, (z, z, z, z),
                               _AffineWork(length(col), FT))
end

"""
    FacetElement(facet::UrbanFacet, area; name = :facet)

An envelope element whose outer face is a full `UrbanSurface.UrbanFacet`:
the facet's surface energy balance (net radiation, Monin–Obukhov or
fixed-coefficient turbulent exchange) closes the outside, and the zone
drives its interior boundary. The facet's interior coefficient
(`facet.interior.h`, by default the ISO 6946 value for its kind) is the
interior surface coefficient; it must be finite, so a `:ground` facet
(Dirichlet deep-soil base) cannot be a zone element.

This is the two-way coupling: a roof or wall that is part of the urban
canopy *and* bounds the zone. The facet is stepped through its own
`step_surface!`; the fluxes it returned over the last zone step are
[`facet_fluxes`](@ref)`(el)` → what the canopy air receives.
"""
mutable struct FacetElement{T<:AbstractFloat} <: AbstractZoneElement
    name::Symbol
    facet::UrbanFacet{T}
    area::T
    q_inside::T
    F_outer::T
    Q_air::T
    fluxes::SurfaceFluxes{T}
    _F::NTuple{4,T}               # (G, dG, F_inner, dF_inner)
    _fa::SurfaceFluxes{T}         # fluxes of the base solve
    _dfl::SurfaceFluxes{T}        # their derivative w.r.t. the air temperature
    _skin::NTuple{2,T}            # (T_skin of base solve, derivative)
    _w::_AffineWork{T}
end

"""
    steady_envelope!(el::EnvelopeElement, T_out, T_in) -> el

Start `el`'s wall in the steady state between the outside and the zone
air: the outer face exchanges with the outdoor air `T_out` (K) through
`h_outside` (an `exterior = :ground` element: held at its `T_ground`),
an adiabatic one stays adiabatic), the inner face with `T_in` through `h_inside`
(`UrbanSurface.steady_conduction!`). A wall started uniform at the zone
temperature must instead relax over its slowest time scale — 9 h for the
example brick wall, 26 h for the example roof (#519) — and its outer flux
is wrong by the whole indoor–outdoor difference at the start.
"""
function steady_envelope!(el::EnvelopeElement, T_out::Real, T_in::Real)
    outer = el.exterior === :air ? convective_boundary(T_out, el.h_outside) :
            el.exterior === :ground ? fixed_temperature_boundary(el.T_ground) : heat_flux_boundary(0.0)
    steady_conduction!(el.column, outer, convective_boundary(T_in, el.h_inside))
    return el
end

function FacetElement(facet::UrbanFacet{T}, area::Real; name::Symbol = facet.kind) where {T}
    area > 0 || throw(ArgumentError("FacetElement: area must be positive"))
    (0 < facet.interior.h < Inf) ||
        throw(ArgumentError("FacetElement: the facet's interior coefficient must be positive " *
                            "and finite (a :ground facet with a Dirichlet base cannot bound a zone)"))
    z = zero(T)
    zf = SurfaceFluxes{T}(z, z, z, z, z, z)
    return FacetElement{T}(name, facet, T(area), z, z, z, zf, (z, z, z, z), zf, zf, (z, z),
                           _AffineWork(length(facet.column), T))
end

"The `SurfaceFluxes` a [`FacetElement`](@ref)'s facet applied over the last zone step."
facet_fluxes(el::FacetElement) = el.fluxes

"""
    InternalMass(layers, area; sides = 2, h_inside = 1/0.13, T = 293.15,
                 θ = 0.5, name = :internal_mass)

Thermal mass inside the zone — partitions, intermediate floors,
furniture — as a `ConductionColumn` through `layers`. `area` is the
area of **one** face (m²).

  * `sides = 2`: both faces are exposed to the room (a partition wall or
    an intermediate floor between identical zones); the exposed area is
    `2 area`.
  * `sides = 1`: only the *inner* face (the last layer) is exposed; the
    first layer's face is adiabatic (a floor slab on insulation, a
    massive piece of furniture against an adiabatic backing).

Both exposed faces exchange with the air node through `h_inside` and
receive the same share of radiative gains as every interior surface.
"""
mutable struct InternalMass{T<:AbstractFloat} <: AbstractZoneElement
    name::Symbol
    column::ConductionColumn{T}
    area::T
    h_inside::T
    sides::Int
    q_inside::T
    Q_air::T
    _F::NTuple{4,T}               # (F_outer, dF_outer, F_inner, dF_inner)
    _w::_AffineWork{T}
end

function InternalMass(layers::AbstractVector{<:WallLayer}, area::Real; sides::Integer = 2,
                      h_inside::Real = _H_WALL, T = 293.15, θ::Real = 0.5,
                      name::Symbol = :internal_mass)
    area > 0 || throw(ArgumentError("InternalMass: area must be positive"))
    sides in (1, 2) || throw(ArgumentError("InternalMass: sides must be 1 or 2"))
    (0 < h_inside < Inf) || throw(ArgumentError("InternalMass: h_inside must be positive and finite"))
    col = ConductionColumn(layers; T = T, θ = θ)
    FT = eltype(col.T)
    z = zero(FT)
    return InternalMass{FT}(name, col, FT(area), FT(h_inside), Int(sides), z, z,
                            (z, z, z, z), _AffineWork(length(col), FT))
end

_column(el::Union{EnvelopeElement, InternalMass}) = el.column
_column(el::FacetElement) = el.facet.column
_h_inside(el::Union{EnvelopeElement, InternalMass}) = el.h_inside
_h_inside(el::FacetElement) = el.facet.interior.h
"Interior area of an element exposed to the zone air (m²)."
_exposed_area(el::Union{EnvelopeElement, FacetElement}) = el.area
_exposed_area(el::InternalMass) = el.sides * el.area

"""
    element_transmittance(el; R_se = 0.04) -> U (W m⁻² K⁻¹)

Steady air-to-air (or air-to-ground) transmittance of an envelope
element: ``U = 1/(1/h_{in} + Σ d/k + R_{out})`` with ``R_{out} =
1/h_{out}`` for `:air`, 0 for `:ground` and ``∞`` (U = 0) for
`:adiabatic`. A [`FacetElement`](@ref) uses the conventional ISO 6946
`R_se` (its real outer exchange is the full energy balance). Internal
mass has no transmittance (0).
"""
function element_transmittance(el::EnvelopeElement)
    el.exterior === :adiabatic && return zero(el.area)
    R_out = el.exterior === :air ? 1 / el.h_outside : zero(el.area)
    return 1 / (1 / el.h_inside + thermal_resistance(el.column.layers) + R_out)
end
element_transmittance(el::FacetElement; R_se::Real = ISO6946_SURFACE_RESISTANCE.outside) =
    1 / (1 / el.facet.interior.h + thermal_resistance(el.facet.column.layers) + R_se)
element_transmittance(el::InternalMass) = zero(el.area)

# =========================================================================
# 2. The zone
# =========================================================================

"""
    BuildingZone(; volume, floor_area, envelope = [], internal_mass = [],
                 window_area = 0, window_U = 0, infiltration_ach = 0.5,
                 ventilation_ach = 0, heat_recovery = 0,
                 radiative_fraction = 0.5, T_air = 293.15,
                 q_air = nothing, p_ref = 101325,
                 thermo = DEFAULT_CONSTANTS, air_capacity = nothing)

A single thermal zone of `volume` m³ and `floor_area` m².

  * `envelope` — [`EnvelopeElement`](@ref)s and [`FacetElement`](@ref)s;
    `internal_mass` — [`InternalMass`](@ref)es.
  * Glazing: a massless conductance ``H_w = U_w A_w`` (W K⁻¹). Solar
    transmitted through it enters through [`ZoneForcing`](@ref)`.Q_solar`.
  * Air exchange (ISO 13790 §9.3): infiltration and mechanical
    ventilation air-change rates (h⁻¹), both supplied from outdoors; a
    heat-recovery efficiency ``η`` preheats the ventilation supply to
    ``T_{out} + η (T_{air} - T_{out})``. The ventilation conductance is
    ``H_{ve} = ρ c_p V (n_{inf} + (1 - η) n_{vent}) / 3600``.
  * `radiative_fraction` of internal gains, and all transmitted solar
    gain, is spread over the interior surfaces in proportion to area;
    the rest of the internal gains is convective (to the air node).
  * `air_capacity` (J K⁻¹) overrides ``C_a = ρ c_p V``, where
    ``ρ = p_{ref}/(R_d T_{air})`` and ``c_p`` are the dry-air values of
    `thermo` (`MoistThermodynamics`) at construction. Many BEMs multiply
    ``C_a`` by a furniture factor; here furniture is internal mass.
  * `q_air` — indoor specific humidity (kg kg⁻¹); by default 50 %
    relative humidity at the initial `T_air` and `p_ref`. It is advanced
    only when the forcing supplies an outdoor humidity
    ([`ZoneForcing`](@ref)`.q_out`), by an implicit moisture balance of
    the air mass ``ρV`` with the same air-change rates (heat recovery is
    sensible only), latent internal gains, and dehumidification by an
    [`IdealLoadsHVAC`](@ref). Hygroscopic buffering by the walls and
    furniture is not modelled.

The interior long-wave exchange between surfaces is lumped into the
surface coefficients to the air node (the ISO 6946 combined ``h``), so
the node is strictly a mix of air and radiant temperature; see
[`zone_operative_temperature`](@ref).
"""
mutable struct BuildingZone{T<:AbstractFloat}
    volume::T
    floor_area::T
    envelope::Vector{AbstractZoneElement}
    internal_mass::Vector{InternalMass{T}}
    window_area::T
    window_U::T
    infiltration_ach::T
    ventilation_ach::T
    heat_recovery::T
    radiative_fraction::T
    T_air::T
    C_air::T
    ρc_air::T        # ρ c_p of the air (J m⁻³ K⁻¹), for the ventilation conductance
    q_air::T         # indoor specific humidity, kg kg⁻¹
    ρ_air::T         # air density at construction, kg m⁻³ (moisture mass balance)
    p_ref::T
    thermo::ThermodynamicConstants{T}
end

function BuildingZone(; volume::Real, floor_area::Real,
                      envelope::AbstractVector = AbstractZoneElement[],
                      internal_mass::AbstractVector = InternalMass{Float64}[],
                      window_area::Real = 0.0, window_U::Real = 0.0,
                      infiltration_ach::Real = 0.5, ventilation_ach::Real = 0.0,
                      heat_recovery::Real = 0.0, radiative_fraction::Real = 0.5,
                      T_air::Real = 293.15, q_air = nothing, p_ref::Real = 101325.0,
                      thermo::ThermodynamicConstants = DEFAULT_CONSTANTS,
                      air_capacity = nothing)
    volume > 0 || throw(ArgumentError("BuildingZone: volume must be positive"))
    floor_area > 0 || throw(ArgumentError("BuildingZone: floor_area must be positive"))
    (window_area >= 0 && window_U >= 0) ||
        throw(ArgumentError("BuildingZone: window_area and window_U must be ≥ 0"))
    (infiltration_ach >= 0 && ventilation_ach >= 0) ||
        throw(ArgumentError("BuildingZone: air-change rates must be ≥ 0"))
    (0 <= heat_recovery <= 1) || throw(ArgumentError("BuildingZone: heat_recovery must lie in [0, 1]"))
    (0 <= radiative_fraction <= 1) ||
        throw(ArgumentError("BuildingZone: radiative_fraction must lie in [0, 1]"))
    all(el -> el isa Union{EnvelopeElement, FacetElement}, envelope) ||
        throw(ArgumentError("BuildingZone: envelope takes EnvelopeElements and FacetElements"))
    all(el -> el isa InternalMass, internal_mass) ||
        throw(ArgumentError("BuildingZone: internal_mass takes InternalMass elements"))
    FT = Float64
    q0 = PhasePartition(0.0)
    ρ = air_density(T_air, p_ref, q0, thermo)
    ρc = ρ * heat_capacity_moist(q0, thermo)
    C = air_capacity === nothing ? ρc * volume : float(air_capacity)
    C > 0 || throw(ArgumentError("BuildingZone: air_capacity must be positive"))
    q = q_air === nothing ? specific_humidity_from_relative_humidity(T_air, p_ref, 0.5, thermo) :
                            float(q_air)
    (0 <= q < 1) || throw(ArgumentError("BuildingZone: q_air must lie in [0, 1)"))
    return BuildingZone{FT}(FT(volume), FT(floor_area), AbstractZoneElement[envelope...],
                            InternalMass{FT}[internal_mass...], FT(window_area), FT(window_U),
                            FT(infiltration_ach), FT(ventilation_ach), FT(heat_recovery),
                            FT(radiative_fraction), FT(T_air), FT(C), FT(ρc), FT(q),
                            FT(ρ), FT(p_ref), thermo)
end

_elements(z::BuildingZone) = Iterators.flatten((z.envelope, z.internal_mass))

"Zone air temperature (K)."
zone_air_temperature(z::BuildingZone) = z.T_air

"Indoor specific humidity (kg kg⁻¹)."
zone_specific_humidity(z::BuildingZone) = z.q_air

"Indoor relative humidity over liquid water (fraction), from `MoistThermodynamics`."
zone_relative_humidity(z::BuildingZone) = relative_humidity(z.T_air, z.p_ref, z.q_air, z.thermo)

"Heat capacity of the zone air node, ``C_a`` (J K⁻¹)."
air_heat_capacity(z::BuildingZone) = z.C_air

"""
    ventilation_conductance(z) -> H_ve (W K⁻¹)

``H_{ve} = ρ c_p V (n_{inf} + (1-η) n_{vent}) / 3600`` — the heat
transfer coefficient by air exchange (ISO 13790 §9.3, with the
heat-recovery temperature-adjustment factor ``1-η`` on the mechanical
part).
"""
ventilation_conductance(z::BuildingZone) =
    z.ρc_air * z.volume * (z.infiltration_ach + (1 - z.heat_recovery) * z.ventilation_ach) / 3600

"""
    zone_heat_loss_coefficient(z) -> H (W K⁻¹)

Steady heat-loss coefficient of the zone to the outdoor air,
``H = H_{ve} + U_w A_w + Σ_{air, facet} U_j A_j`` (ground-contact and
adiabatic elements excluded; see [`element_transmittance`](@ref)).
Under steady conditions with no gains, no ground coupling and all heat
exchanged through the outdoor air, ``T_{air} - T_{out} = Φ/H`` for a
purely convective gain ``Φ``.
"""
function zone_heat_loss_coefficient(z::BuildingZone)
    H = ventilation_conductance(z) + z.window_U * z.window_area
    for el in z.envelope
        el isa EnvelopeElement && el.exterior !== :air && continue
        H += element_transmittance(el) * el.area
    end
    return H
end

"""
    zone_heat_content(z; T_ref = 0) -> J

Sensible heat stored in the zone relative to `T_ref`: the air node plus
every envelope and internal-mass column (area times
`UrbanSurface.column_heat_content`). Over a [`step_zone!`](@ref) its
change equals `dt` times the sum of the [`ZoneBudget`](@ref) sources.
"""
function zone_heat_content(z::BuildingZone; T_ref::Real = 0)
    E = z.C_air * (z.T_air - T_ref)
    for el in _elements(z)
        E += el.area * column_heat_content(_column(el); T_ref = T_ref)
    end
    return E
end

# Interior (room-side) face temperature of an element, from the last state.
function _inner_surface_temperatures(el::EnvelopeElement, T_air)
    bc = ColumnBoundary(T_air, el.h_inside, el.q_inside)
    return (face_temperature(el.column, bc, :inner),)
end
_inner_surface_temperatures(el::FacetElement, T_air) =
    (face_temperature(el.facet.column, el.facet.interior, :inner),)
function _inner_surface_temperatures(el::InternalMass, T_air)
    bc = ColumnBoundary(T_air, el.h_inside, el.q_inside)
    Ti = face_temperature(el.column, bc, :inner)
    return el.sides == 2 ? (face_temperature(el.column, bc, :outer), Ti) : (Ti,)
end

"""
    zone_mean_radiant_temperature(z) -> K

Area-weighted mean temperature of all interior surfaces (envelope and
internal mass faces). Glazing, which carries no state, is left out.
"""
function zone_mean_radiant_temperature(z::BuildingZone)
    num = zero(z.T_air); den = zero(z.T_air)
    for el in _elements(z)
        for Ts in _inner_surface_temperatures(el, z.T_air)
            num += el.area * Ts
            den += el.area
        end
    end
    return den > 0 ? num / den : z.T_air
end

"""
    zone_operative_temperature(z) -> K

``T_{op} = (T_{air} + T_{mrt})/2``, the operative temperature at low air
speed (ISO 7726 / ISO 7730: the mean of air and mean radiant temperature
when the air speed is below 0.2 m s⁻¹).

Note that the ISO 13790 5R1C model defines its operative temperature
differently, ``θ_{op} = 0.3 θ_{air} + 0.7 θ_s`` (ISO 13790:2008 eq. C.12),
with ``θ_s`` its central star-node temperature, not a mean radiant
temperature; the two are not interchangeable.
"""
zone_operative_temperature(z::BuildingZone) = (z.T_air + zone_mean_radiant_temperature(z)) / 2

# =========================================================================
# 3. Forcing, budget and the step
# =========================================================================

"""
    ZoneForcing(; T_out, Q_internal = 0, Q_solar = 0, Q_hvac = 0,
                irradiance = 0, surface = nothing, q_out = nothing,
                Q_latent = 0, radiative_fraction = nothing)

What drives a zone over one step:

  * `T_out` — outdoor air temperature (K), seen by `:air` envelope
    elements, glazing and the air exchange;
  * `Q_internal` — internal gains (W; people, lighting, appliances),
    split by the zone's `radiative_fraction`;
  * `Q_solar` — solar radiation transmitted through glazing (W),
    absorbed by the interior surfaces;
  * `Q_hvac` — sensible heat delivered to the air node by a prescribed
    heating or cooling input (W, negative for cooling). With an
    [`IdealLoadsHVAC`](@ref) (`step_zone!(zone, forcing, dt, hvac)`) the
    ideal load is computed and added to this;
  * `irradiance` — total irradiance on the outer face of `:air`
    [`EnvelopeElement`](@ref)s (W m⁻², absorbed with the element's
    `absorptance`): a number for all, or a vector indexed like
    `zone.envelope` (entries of other element types are ignored);
  * `surface` — the `SurfaceForcing` of the [`FacetElement`](@ref)s: one
    for all, or a vector indexed like `zone.envelope`. Required when the
    zone has facet elements;
  * `q_out` — outdoor specific humidity (kg kg⁻¹). When given, the
    zone humidity is advanced; `nothing` leaves it unchanged;
  * `Q_latent` — latent internal gains (W; occupants, cooking), i.e. a
    vapour source ``Q_{latent}/L_v``. Requires `q_out`;
  * `radiative_fraction` — the radiative share of `Q_internal` for this
    step, overriding the zone's `radiative_fraction` (`nothing`: use the
    zone's). Scheduled gains whose components have different radiant
    fractions (people, lighting, equipment) change it every step; see
    [`internal_gains`](@ref) and `ZoneForcing(::InternalGainRates)`.
"""
struct ZoneForcing{T<:AbstractFloat, I, S, Q, R}
    T_out::T
    Q_internal::T
    Q_solar::T
    Q_hvac::T
    irradiance::I
    surface::S
    q_out::Q
    Q_latent::T
    radiative_fraction::R
end

function ZoneForcing(; T_out::Real, Q_internal::Real = 0.0, Q_solar::Real = 0.0,
                     Q_hvac::Real = 0.0, irradiance = 0.0, surface = nothing,
                     q_out = nothing, Q_latent::Real = 0.0, radiative_fraction = nothing)
    Q_solar >= 0 || throw(ArgumentError("ZoneForcing: Q_solar must be ≥ 0"))
    Q_latent >= 0 || throw(ArgumentError("ZoneForcing: Q_latent must be ≥ 0"))
    (Q_latent == 0 || q_out !== nothing) ||
        throw(ArgumentError("ZoneForcing: latent gains need the outdoor humidity `q_out`"))
    T = float(promote_type(typeof(T_out), typeof(Q_internal), typeof(Q_solar), typeof(Q_hvac)))
    qo = q_out === nothing ? nothing : T(q_out)
    (qo === nothing || 0 <= qo < 1) || throw(ArgumentError("ZoneForcing: q_out must lie in [0, 1)"))
    fr = radiative_fraction === nothing ? nothing : T(radiative_fraction)
    (fr === nothing || 0 <= fr <= 1) ||
        throw(ArgumentError("ZoneForcing: radiative_fraction must lie in [0, 1]"))
    return ZoneForcing{T, typeof(irradiance), typeof(surface), typeof(qo), typeof(fr)}(
        T(T_out), T(Q_internal), T(Q_solar), T(Q_hvac), irradiance, surface, qo, T(Q_latent), fr)
end

_pick(x::Real, i) = x
_pick(x::AbstractVector, i) = x[i]
_pick(x::SurfaceForcing, i) = x
_pick(::Nothing, i) = throw(ArgumentError("ZoneForcing: the zone has FacetElements; pass `surface`"))

"""
    ZoneBudget

Heat budget of a zone over the last [`step_zone!`](@ref), all in W and
positive when adding heat to the zone:

  * `T_air` — the new air temperature (K);
  * `Q_exterior` — heat entering the envelope columns through their
    outer faces (from outdoor air, sun, sky, ground);
  * `Q_windows` — ``U_w A_w (T_{out} - T_{air})``;
  * `Q_ventilation` — ``H_{ve} (T_{out} - T_{air})`` (infiltration and
    ventilation, after heat recovery);
  * `Q_internal`, `Q_solar`, `Q_hvac` — as forced;
  * `Q_surfaces` — heat from all interior surfaces to the air node (the
    air-side view: envelope *and* internal mass);
  * `Q_envelope_air` — the envelope part of `Q_surfaces` (what a BEM
    calls the envelope load);
  * `q_air` — the new indoor specific humidity (kg kg⁻¹);
  * `W_ventilation`, `W_internal`, `W_hvac` — water vapour added to the
    zone air by air exchange, latent gains and the HVAC (negative:
    dehumidification), in kg s⁻¹ (zero when humidity is not advanced).

Conservation: ``ΔE/Δt = Q_{exterior} + Q_{windows} + Q_{ventilation} +
Q_{internal} + Q_{solar} + Q_{hvac}`` with ``E`` the
[`zone_heat_content`](@ref), and ``ρV Δq/Δt = W_{ventilation} +
W_{internal} + W_{hvac}``.
"""
struct ZoneBudget{T<:AbstractFloat}
    T_air::T
    Q_exterior::T
    Q_windows::T
    Q_ventilation::T
    Q_internal::T
    Q_solar::T
    Q_hvac::T
    Q_surfaces::T
    Q_envelope_air::T
    q_air::T
    W_ventilation::T
    W_internal::T
    W_hvac::T
end

# ---- the affine step --------------------------------------------------------
#
# With the zone air temperature x held over the step as the environment of
# every interior face, each θ-scheme column step is an affine map of x: the
# new cell temperatures and both face fluxes are T_a + (x - x0) dT. Each
# element is therefore stepped twice, at x0 and x0 + 1 K, from the same
# initial state, which gives the base solution and its exact derivative.
# The air balance is then a scalar linear equation for x, after which the
# elements are finalised at that x. The result is identical (to round-off)
# to stepping every column directly with the final x as its interior
# environment — the tests check this against an independently stepped facet.

function _affine_step!(el::EnvelopeElement{T}, x0, forcing::ZoneForcing, i, dt) where {T}
    col, w = el.column, el._w
    outer = el.exterior === :air ?
        ColumnBoundary{T}(T(forcing.T_out), el.h_outside,
                          el.absorptance * T(_pick(forcing.irradiance, i))) :
        el.exterior === :ground ? fixed_temperature_boundary(el.T_ground) :
        heat_flux_boundary(0.0)
    w.T_old .= col.T
    Fo_a, Fi_a = step_conduction!(col, outer, ColumnBoundary(x0, el.h_inside, el.q_inside), dt)
    w.T_a .= col.T
    col.T .= w.T_old
    Fo_b, Fi_b = step_conduction!(col, outer, ColumnBoundary(x0 + 1, el.h_inside, el.q_inside), dt)
    w.dT .= col.T .- w.T_a
    col.T .= w.T_a
    el._F = (Fo_a, Fo_b - Fo_a, Fi_a, Fi_b - Fi_a)
    # heat to the air per unit area = q - F_inner
    return el.area * (el.q_inside - Fi_a), -el.area * (Fi_b - Fi_a)
end

function _affine_step!(el::InternalMass{T}, x0, forcing::ZoneForcing, i, dt) where {T}
    col, w = el.column, el._w
    bc(x) = ColumnBoundary(x, el.h_inside, el.q_inside)
    outer(x) = el.sides == 2 ? bc(x) : heat_flux_boundary(0.0)
    w.T_old .= col.T
    Fo_a, Fi_a = step_conduction!(col, outer(x0), bc(x0), dt)
    w.T_a .= col.T
    col.T .= w.T_old
    Fo_b, Fi_b = step_conduction!(col, outer(x0 + 1), bc(x0 + 1), dt)
    w.dT .= col.T .- w.T_a
    col.T .= w.T_a
    el._F = (Fo_a, Fo_b - Fo_a, Fi_a, Fi_b - Fi_a)
    # every exposed face gives q - F to the air; the adiabatic face gives nothing
    Qa = el.sides * el.q_inside - Fi_a - (el.sides == 2 ? Fo_a : zero(T))
    dQ = -(Fi_b - Fi_a) - (el.sides == 2 ? (Fo_b - Fo_a) : zero(T))
    return el.area * Qa, el.area * dQ
end

_saxpy(a::SurfaceFluxes{T}, s, d::SurfaceFluxes{T}) where {T} =
    SurfaceFluxes{T}(a.Rn + s * d.Rn, a.H + s * d.H, a.LE + s * d.LE,
                     a.G + s * d.G, a.τ + s * d.τ, a.E + s * d.E)
_sdiff(b::SurfaceFluxes{T}, a::SurfaceFluxes{T}) where {T} =
    SurfaceFluxes{T}(b.Rn - a.Rn, b.H - a.H, b.LE - a.LE, b.G - a.G, b.τ - a.τ, b.E - a.E)

function _affine_step!(el::FacetElement{T}, x0, forcing::ZoneForcing, i, dt;
                       kwargs...) where {T}
    f, w = el.facet, el._w
    col = f.column
    sf = _pick(forcing.surface, i)
    h = f.interior.h
    Ts0, G0, Q0 = f.T_skin, f.G, f.Q_interior
    w.T_old .= col.T
    set_interior_boundary!(f, ColumnBoundary(x0, h, el.q_inside))
    fa = step_surface!(f, sf, dt; kwargs...)
    Fi_a, Ts_a = -f.Q_interior, f.T_skin
    w.T_a .= col.T
    # restore and repeat at x0 + 1: the outer linearisation depends only on
    # the (restored) skin temperature, so the two solves share it exactly
    col.T .= w.T_old
    f.T_skin, f.G, f.Q_interior = Ts0, G0, Q0
    set_interior_boundary!(f, ColumnBoundary(x0 + 1, h, el.q_inside))
    fb = step_surface!(f, sf, dt; kwargs...)
    Fi_b, Ts_b = -f.Q_interior, f.T_skin
    w.dT .= col.T .- w.T_a
    col.T .= w.T_a
    el._fa, el._dfl = fa, _sdiff(fb, fa)
    el._skin = (Ts_a, Ts_b - Ts_a)
    el._F = (fa.G, fb.G - fa.G, Fi_a, Fi_b - Fi_a)
    return el.area * (el.q_inside - Fi_a), -el.area * (Fi_b - Fi_a)
end

function _finalize!(el::Union{EnvelopeElement, InternalMass}, s)
    Fo, dFo, Fi, dFi = el._F
    _column(el).T .+= s .* el._w.dT
    Fo_x, Fi_x = Fo + s * dFo, Fi + s * dFi
    if el isa EnvelopeElement
        el.F_outer = Fo_x
        el.Q_air = el.q_inside - Fi_x
    else
        el.Q_air = el.sides == 2 ? 2el.q_inside - Fi_x - Fo_x : el.q_inside - Fi_x
    end
    return el.area * (el isa EnvelopeElement ? Fo_x : zero(Fo_x))
end

function _finalize!(el::FacetElement{T}, s, x) where {T}
    f = el.facet
    Fo, dFo, Fi, dFi = el._F
    f.column.T .+= s .* el._w.dT
    f.T_skin = el._skin[1] + s * el._skin[2]
    Fo_x, Fi_x = Fo + s * dFo, Fi + s * dFi
    f.G = Fo_x
    f.Q_interior = -Fi_x
    set_interior_boundary!(f, ColumnBoundary(x, f.interior.h, el.q_inside))
    el.fluxes = _saxpy(el._fa, s, el._dfl)
    el.F_outer = Fo_x
    el.Q_air = el.q_inside - Fi_x
    return el.area * Fo_x
end

"""
    step_zone!(zone, forcing::ZoneForcing, dt; kwargs...) -> ZoneBudget

Advance the zone by `dt` seconds. The air node balance

```math
C_a \\frac{T^{n+1} - T^n}{Δt} = Σ_j A_j h_j (T_{s,j} - T^{n+1})
  + (U_w A_w + H_{ve})(T_{out} - T^{n+1}) + (1 - f_r) Φ_{int} + Φ_{hvac}
```

is solved **implicitly** in the air temperature, simultaneously with
every element column (θ-scheme, `UrbanSurface.step_conduction!` /
`step_surface!`) whose interior environment is ``T^{n+1}``. The zone air
time constant (minutes) is far shorter than any sensible step, so an
explicit air update would be unstable; this one is unconditionally
stable. Radiative gains ``f_r Φ_{int} + Φ_{sol}`` are imposed on the
interior faces, spread by exposed area; with no interior surface they go
to the air.

When the forcing carries an outdoor humidity `q_out`, the indoor
specific humidity is advanced after the temperature by the implicit
vapour balance of the air mass ``M = ρV``,

```math
M \\frac{q^{n+1} - q^n}{Δt} = M \\frac{n_{inf} + n_{vent}}{3600} (q_{out} - q^{n+1})
  + \\frac{Φ_{lat}}{L_v(T^{n+1})} + W_{hvac}.
```

`kwargs` are passed to the `step_surface!` of facet elements (`sf`, `c`,
`δT`, …). The step conserves [`zone_heat_content`](@ref) exactly.

    step_zone!(zone, forcing, dt, hvac::IdealLoadsHVAC; kwargs...) -> (ZoneBudget, HVACBudget)

The same step with an ideal-loads HVAC system that holds the air
temperature between its setpoints; see [`IdealLoadsHVAC`](@ref). The
ideal load is added to `forcing.Q_hvac`.
"""
function step_zone!(zone::BuildingZone, forcing::ZoneForcing, dt::Real; kwargs...)
    sys = _assemble_zone!(zone, forcing, dt; kwargs...)
    return _complete_zone_step!(zone, forcing, dt, sys, zero(sys.lhs), zero(sys.lhs))
end

# Step every element at the old air temperature and at +1 K and return the
# scalar air balance lhs (x - x0) = R + Φ, Φ being any extra heat input to
# the air node (the HVAC load). Leaves the elements in the pending state
# that `_finalize_zone!` completes.
_assemble_zone!(zone::BuildingZone, forcing::ZoneForcing, dt::Real; kwargs...) =
    _assemble_zone!(zone, forcing, dt, falses(length(zone.envelope)); kwargs...)

# `in_scene[i]`: envelope element i is a FacetElement stepped by an
# `UrbanFacetScene` together with the zone (`step_buildings!`). It receives
# its share of the radiative gains but is not stepped here, and it is not in
# the returned air balance.
function _assemble_zone!(zone::BuildingZone{T}, forcing::ZoneForcing, dt::Real,
                         in_scene::AbstractVector{Bool}; kwargs...) where {T}
    dt > 0 || throw(ArgumentError("step_zone!: dt must be positive"))
    x0 = zone.T_air
    T_out = T(forcing.T_out)
    f_rad = forcing.radiative_fraction === nothing ? zone.radiative_fraction :
                                                     T(forcing.radiative_fraction)
    Φ_rad = f_rad * T(forcing.Q_internal) + T(forcing.Q_solar)
    A_int = sum((_exposed_area(el) for el in _elements(zone)); init = zero(T))
    q = A_int > 0 ? Φ_rad / A_int : zero(T)
    Φ_air = T(forcing.Q_internal) + T(forcing.Q_solar) - (A_int > 0 ? Φ_rad : zero(T)) +
            T(forcing.Q_hvac)
    a = zero(T); b = zero(T)
    for (i, el) in enumerate(zone.envelope)
        el.q_inside = q
        in_scene[i] && continue
        qa, qb = el isa FacetElement ? _affine_step!(el, x0, forcing, i, dt; kwargs...) :
                                        _affine_step!(el, x0, forcing, i, dt)
        a += qa; b += qb
    end
    for el in zone.internal_mass
        el.q_inside = q
        qa, qb = _affine_step!(el, x0, forcing, 0, dt)
        a += qa; b += qb
    end
    H_w = zone.window_U * zone.window_area
    H_ve = ventilation_conductance(zone)
    # C/dt (x - x0) = a + b (x - x0) + (H_w + H_ve)(T_out - x) + Φ_air + Φ
    lhs = zone.C_air / T(dt) - b + H_w + H_ve
    R = a + (H_w + H_ve) * (T_out - x0) + Φ_air
    return (; x0, T_out, lhs, R, H_w, H_ve, q)
end

# Finalise the elements at the solved air temperature, advance humidity,
# and build the budget. `Φ` is the extra (HVAC) heat to the air node,
# `W` the HVAC vapour source (kg s⁻¹, negative for dehumidification).
function _complete_zone_step!(zone::BuildingZone{T}, forcing::ZoneForcing, dt::Real, sys, Φ, W;
                              s = (sys.R + Φ) / sys.lhs,
                              in_scene::AbstractVector{Bool} = falses(length(zone.envelope)),
                              scene_fluxes = nothing) where {T}
    x = sys.x0 + s
    Q_ext = zero(T); Q_env_air = zero(T); Q_surf = zero(T)
    for (i, el) in enumerate(zone.envelope)
        if in_scene[i]
            # already advanced by the scene step, at this air temperature
            f = el.facet
            el.F_outer = f.G
            el.Q_air = el.q_inside + f.Q_interior
            el.fluxes = scene_fluxes[i]
            Q_ext += el.area * f.G
        else
            Q_ext += el isa FacetElement ? _finalize!(el, s, x) : _finalize!(el, s)
        end
        Q_env_air += el.area * el.Q_air
    end
    Q_surf = Q_env_air
    for el in zone.internal_mass
        _finalize!(el, s)
        Q_surf += el.area * el.Q_air
    end
    zone.T_air = x
    W_ve = W_int = zero(T)
    if forcing.q_out !== nothing
        q_new, W_ve, W_int = _moisture_solve(zone, forcing, dt, x, W)
        zone.q_air = q_new
    end
    return ZoneBudget{T}(x, Q_ext, sys.H_w * (sys.T_out - x), sys.H_ve * (sys.T_out - x),
                         T(forcing.Q_internal), T(forcing.Q_solar), T(forcing.Q_hvac) + Φ,
                         Q_surf, Q_env_air, zone.q_air, W_ve, W_int,
                         forcing.q_out === nothing ? zero(T) : T(W))
end

# Air mass (kg) and total outdoor-air exchange rate (s⁻¹) of the vapour balance.
_air_mass(zone::BuildingZone) = zone.ρ_air * zone.volume
_exchange_rate(zone::BuildingZone) = (zone.infiltration_ach + zone.ventilation_ach) / 3600

# Implicit vapour balance: (M/dt + M n) q = M q0/dt + M n q_out + W_int + W.
# Returns the new q and the ventilation and internal vapour sources (kg s⁻¹).
function _moisture_solve(zone::BuildingZone{T}, forcing::ZoneForcing, dt, T_new, W) where {T}
    M, n = _air_mass(zone), _exchange_rate(zone)
    W_int = T(forcing.Q_latent) / latent_heat_vapor(T_new, zone.thermo)
    q_out = T(forcing.q_out)
    q = (M * zone.q_air / dt + M * n * q_out + W_int + W) / (M / dt + M * n)
    return q, M * n * (q_out - q), W_int
end

# =========================================================================
# 3b. HVAC: ideal loads, equipment performance curves and exhaust heat
# =========================================================================

"""
    BiquadraticCurve(c; x_range = (-Inf, Inf), y_range = (-Inf, Inf))

The DOE-2 / EnergyPlus `Curve:Biquadratic` performance curve

```math
f(x, y) = c_1 + c_2 x + c_3 x^2 + c_4 y + c_5 y^2 + c_6 x y,
```

with the inputs clamped to `x_range` and `y_range` (EnergyPlus limits
the independent variables, not the output, to the fitted domain).
Temperatures are in °C, as in the published coefficients.
"""
struct BiquadraticCurve{T<:AbstractFloat}
    c::NTuple{6,T}
    x_min::T
    x_max::T
    y_min::T
    y_max::T
end

function BiquadraticCurve(c; x_range = (-Inf, Inf), y_range = (-Inf, Inf))
    length(c) == 6 || throw(ArgumentError("BiquadraticCurve: six coefficients are needed"))
    (x_range[1] <= x_range[2] && y_range[1] <= y_range[2]) ||
        throw(ArgumentError("BiquadraticCurve: empty input range"))
    return BiquadraticCurve{Float64}(Tuple(Float64.(c)), x_range[1], x_range[2],
                                     y_range[1], y_range[2])
end

function (f::BiquadraticCurve)(x::Real, y::Real)
    xc = clamp(x, f.x_min, f.x_max); yc = clamp(y, f.y_min, f.y_max)
    c = f.c
    return c[1] + c[2] * xc + c[3] * xc^2 + c[4] * yc + c[5] * yc^2 + c[6] * xc * yc
end

"""
    DXCoolingCurves(capacity, eir; name = :custom)

The temperature modifiers of a direct-expansion (DX) cooling unit in the
DOE-2 / EnergyPlus `Coil:Cooling:DX:SingleSpeed` form: two
[`BiquadraticCurve`](@ref)s of the coil entering **wet-bulb** temperature
``x`` (°C, the indoor air) and the condenser entering **dry-bulb**
temperature ``y`` (°C, see [`condenser_inlet_temperature`](@ref)),
normalised to 1 at the rating point (ANSI/AHRI 210/240 "A": 26.7 °C dry
bulb / 19.4 °C wet bulb indoors, 35 °C outdoors):

  * `capacity` — ``CAP_{FT}``: available total cooling capacity over its
    rated value;
  * `eir` — ``EIR_{FT}``: energy input ratio (electric power per unit
    cooling, ``1/COP``) over its rated value, so
    ``COP = COP_{rated} / EIR_{FT}``.

Two published curve sets from the EnergyPlus distribution are provided:
[`DX_CURVES_WINDOW_AC`](@ref) and [`DX_CURVES_PACKAGED_AC`](@ref).
"""
struct DXCoolingCurves{T<:AbstractFloat}
    capacity::BiquadraticCurve{T}
    eir::BiquadraticCurve{T}
    name::Symbol
end
DXCoolingCurves(capacity::BiquadraticCurve{T}, eir::BiquadraticCurve{T};
                name::Symbol = :custom) where {T} = DXCoolingCurves{T}(capacity, eir, name)

"ANSI/AHRI 210/240 rating point used to normalise DX curves: indoor wet bulb, outdoor dry bulb (°C)."
const DX_RATING_POINT = (T_wb_indoor = 19.44444, T_db_outdoor = 35.0)

# Both sets: EnergyPlus test files (NREL/EnergyPlus, testfiles/), fitted
# domain x ∈ [12.78, 23.89] °C, y ∈ [18.0, 46.11] °C.
const _DX_X = (12.77778, 23.88889)
const _DX_Y = (18.0, 46.11111)

"""
    DX_CURVES_WINDOW_AC

Cooling-capacity and EIR temperature modifiers of the room (window) air
conditioner in the EnergyPlus example `WindACAuto.idf` (curves
`WindACCoolCapFT`, `WindACEIRFT`; coefficients copied verbatim):

| | ``c_1`` | ``c_2`` | ``c_3`` | ``c_4`` | ``c_5`` | ``c_6`` |
|:--|--:|--:|--:|--:|--:|--:|
| CAP | 0.942587793 | 0.009543347 | 0.000683770 | −0.011042676 | 0.000005249 | −0.000009720 |
| EIR | 0.342414409 | 0.034885008 | −0.000623700 | 0.004977216 | 0.000437951 | −0.000728028 |
"""
const DX_CURVES_WINDOW_AC = DXCoolingCurves(
    BiquadraticCurve((0.942587793, 0.009543347, 0.000683770, -0.011042676, 0.000005249, -0.000009720);
                     x_range = _DX_X, y_range = _DX_Y),
    BiquadraticCurve((0.342414409, 0.034885008, -0.000623700, 0.004977216, 0.000437951, -0.000728028);
                     x_range = _DX_X, y_range = _DX_Y);
    name = :window_ac)

"""
    DX_CURVES_PACKAGED_AC

Cooling-capacity and EIR temperature modifiers of the packaged
air-to-air heat pump in cooling mode in the EnergyPlus example
`HeatPumpAuto.idf` (curves `HPACCoolCapFT`, `HPACCOOLEIRFT`):

| | ``c_1`` | ``c_2`` | ``c_3`` | ``c_4`` | ``c_5`` | ``c_6`` |
|:--|--:|--:|--:|--:|--:|--:|
| CAP | 0.766956 | 0.0107756 | −0.0000414703 | 0.00134961 | −0.000261144 | 0.000457488 |
| EIR | 0.297145 | 0.0430933 | −0.000748766 | 0.00597727 | 0.000482112 | −0.000956448 |
"""
const DX_CURVES_PACKAGED_AC = DXCoolingCurves(
    BiquadraticCurve((0.766956, 0.0107756, -0.0000414703, 0.00134961, -0.000261144, 0.000457488);
                     x_range = _DX_X, y_range = _DX_Y),
    BiquadraticCurve((0.297145, 0.0430933, -0.000748766, 0.00597727, 0.000482112, -0.000956448);
                     x_range = _DX_X, y_range = _DX_Y);
    name = :packaged_ac)

const _HEATING_KINDS = (:boiler, :electric, :heat_pump)
const _CONDENSERS = (:air_cooled, :evaporative)

"""
    IdealLoadsHVAC(; T_heat = 293.15, T_cool = 299.15,
                   heating_capacity = Inf, cooling_capacity = Inf,
                   heating = :boiler, heating_efficiency = <by kind>,
                   cop_rated = 3.0, curves = DX_CURVES_WINDOW_AC,
                   condenser = :air_cooled, evaporative_effectiveness = 0.9,
                   latent_fraction = 0.85, dehumidify = false, RH_max = 0.6)

An ideal-loads HVAC system (the EnergyPlus `ZoneHVAC:IdealLoadsAirSystem`
idea, as in the TEB-BEM of Bueno et al. 2012): each step it supplies to
the air node exactly the sensible heat that puts the end-of-step air
temperature on the violated setpoint — `T_heat` below, `T_cool` above;
nothing in the dead band — limited by the available capacity. Because the
zone step is implicit and affine in the air temperature, the load is
obtained in closed form from the zone's two-solve assembly and the
setpoint is held **exactly** when the capacity suffices.

**Cooling equipment.** A DX unit with rated `cop_rated` and rated total
(sensible + latent) `cooling_capacity`, modified by the
[`DXCoolingCurves`](@ref) `curves` (`nothing`: constant COP and
capacity) evaluated at the zone air wet bulb and the condenser inlet
temperature at the start of the step. Its electric power is
``P = Q_{cool}/COP``. With `dehumidify = true` the coil also removes the
vapour that would push the indoor relative humidity above `RH_max`; that
latent load shares the capacity (sensible first) and the COP.
Re-heat, part-load degradation and fan power are not modelled.

**Heat rejection — the exhaust heat.** The condenser rejects
``Q_{cool} + P = Q_{cool}(1 + 1/COP)`` to the outdoor air:

  * `condenser = :air_cooled`: all of it as sensible heat;
  * `condenser = :evaporative` (evaporatively cooled condenser or a
    cooling tower): a fraction `latent_fraction` as latent heat, i.e. as
    evaporated water, the rest sensible. The default 0.85 is the
    engineering rule of thumb for wet cooling towers — evaporation of
    0.00085 of the circulating flow per °F of range, against 0.001 if
    all the heat went into evaporation; published values span 60-95 %
    (Perry's *Chemical Engineers' Handbook*, as summarised in Primo,
    *Cooling Towers — Basic Calculations*, PDHonline M374). The curves
    then see the evaporatively cooled condenser inlet temperature.

    The two halves come from different models. The COP uses the
    EnergyPlus pre-cooling-pad inlet temperature. The sensible/latent
    split uses the wet-tower rule. In the EnergyPlus pad model the
    evaporated water is ``\\dot m_{cond}\\,Δw`` across the pad. It depends
    on the condenser air flow, not on the heat rejected, and the coil
    itself rejects sensible heat. `W_evaporated` is therefore an estimate
    within the 60-95 % band, not the EnergyPlus water use. The sum
    ``Q_{exh,s} + Q_{exh,l}`` is exact either way.

**Heating equipment** (`heating`, with `heating_efficiency`):

  * `:boiler` — fuel input ``Q/η`` (default ``η = 0.9``); the flue loss
    ``Q(1/η - 1)`` is sensible exhaust;
  * `:electric` — resistance heating, ``η = 1``, no exhaust;
  * `:heat_pump` — electric input ``Q/COP_h`` (default ``COP_h = 3``);
    the ``Q(1 - 1/COP_h)`` taken from the outdoor air is a *negative*
    sensible exhaust.

`heating_capacity = 0` or `cooling_capacity = 0` removes the heating or
cooling system.

The plant obeys the first law exactly: exhaust (sensible + latent) =
energy input + heat extracted from the zone − heat delivered to it (see
[`HVACBudget`](@ref)).

# References

  * EnergyPlus Engineering Reference, *Coils*: single-speed electric DX
    air cooling coil (capacity and EIR modifiers, condenser options,
    evaporative condenser inlet temperature).
  * Bueno, B., Pigeon, G., Norford, L. K., Zibouche, K. & Marchadier, C.
    (2012). Development and evaluation of a building energy model
    integrated in the TEB scheme. *Geosci. Model Dev.*, 5, 433-448.
  * Salamanca, F., Krpo, A., Martilli, A. & Clappier, A. (2010). A new
    building energy model coupled with an urban canopy parameterization
    for urban climate simulations — part I. *Theor. Appl. Climatol.*,
    99, 331-344 (AC waste heat released to the canyon).
"""
struct IdealLoadsHVAC{T<:AbstractFloat, C}
    T_heat::T
    T_cool::T
    heating_capacity::T
    cooling_capacity::T
    heating::Symbol
    heating_efficiency::T
    cop_rated::T
    curves::C
    condenser::Symbol
    evaporative_effectiveness::T
    latent_fraction::T
    dehumidify::Bool
    RH_max::T
end

function IdealLoadsHVAC(; T_heat::Real = 293.15, T_cool::Real = 299.15,
                        heating_capacity::Real = Inf, cooling_capacity::Real = Inf,
                        heating::Symbol = :boiler, heating_efficiency = nothing,
                        cop_rated::Real = 3.0, curves = DX_CURVES_WINDOW_AC,
                        condenser::Symbol = :air_cooled, evaporative_effectiveness::Real = 0.9,
                        latent_fraction::Real = 0.85, dehumidify::Bool = false,
                        RH_max::Real = 0.6)
    T_heat <= T_cool || throw(ArgumentError("IdealLoadsHVAC: T_heat must not exceed T_cool"))
    (heating_capacity >= 0 && cooling_capacity >= 0) ||
        throw(ArgumentError("IdealLoadsHVAC: capacities must be ≥ 0"))
    heating in _HEATING_KINDS || throw(ArgumentError("IdealLoadsHVAC: heating must be one of $_HEATING_KINDS"))
    condenser in _CONDENSERS || throw(ArgumentError("IdealLoadsHVAC: condenser must be one of $_CONDENSERS"))
    η = heating_efficiency !== nothing ? float(heating_efficiency) :
        heating === :boiler ? 0.9 : heating === :electric ? 1.0 : 3.0
    η > 0 || throw(ArgumentError("IdealLoadsHVAC: heating_efficiency must be positive"))
    heating === :boiler && η > 1 && throw(ArgumentError("IdealLoadsHVAC: a boiler efficiency cannot exceed 1"))
    heating === :electric && η != 1 && throw(ArgumentError("IdealLoadsHVAC: electric resistance heating has η = 1"))
    cop_rated > 0 || throw(ArgumentError("IdealLoadsHVAC: cop_rated must be positive"))
    curves isa Union{Nothing, DXCoolingCurves} ||
        throw(ArgumentError("IdealLoadsHVAC: curves must be DXCoolingCurves or nothing"))
    (0 <= evaporative_effectiveness <= 1 && 0 <= latent_fraction <= 1 && 0 < RH_max <= 1) ||
        throw(ArgumentError("IdealLoadsHVAC: effectiveness and latent_fraction in [0, 1], RH_max in (0, 1]"))
    return IdealLoadsHVAC{Float64, typeof(curves)}(T_heat, T_cool, heating_capacity, cooling_capacity,
                                                   heating, η, cop_rated, curves, condenser,
                                                   evaporative_effectiveness, latent_fraction,
                                                   dehumidify, RH_max)
end

"""
    condenser_inlet_temperature(hvac, T_out, q_out, p = 101325; thermo = DEFAULT_CONSTANTS) -> K

Temperature of the air entering the condenser, the ``y`` variable of the
[`DXCoolingCurves`](@ref): the outdoor dry bulb for an `:air_cooled`
condenser, and for an `:evaporative` one (EnergyPlus Engineering
Reference, DX coil condenser options)

```math
T_{c,i} = T_{wb,o} + (1 - ε)(T_{db,o} - T_{wb,o}),
```

with ``ε`` the `evaporative_effectiveness` and ``T_{wb,o}`` the outdoor
thermodynamic wet bulb (`MoistThermodynamics.wet_bulb_temperature`).
`q_out` may be `nothing` for an air-cooled condenser.
"""
function condenser_inlet_temperature(hvac::IdealLoadsHVAC, T_out::Real, q_out, p::Real = 101325.0;
                                     thermo::ThermodynamicConstants = DEFAULT_CONSTANTS)
    hvac.condenser === :air_cooled && return float(T_out)
    q_out === nothing &&
        throw(ArgumentError("an evaporative condenser needs the outdoor humidity (ZoneForcing q_out)"))
    Twb = wet_bulb_temperature(T_out, p, q_out, thermo)
    return Twb + (1 - hvac.evaporative_effectiveness) * (T_out - Twb)
end

"""
    cooling_cop(hvac, T_wb_indoor, T_condenser) -> COP

Operating coefficient of performance of the cooling equipment,
``COP_{rated} / EIR_{FT}(T_{wb}, T_c)`` with both temperatures in K
(converted to °C for the curve). Constant `cop_rated` when the system
has no curves.
"""
cooling_cop(hvac::IdealLoadsHVAC, T_wb_indoor::Real, T_condenser::Real) =
    hvac.curves === nothing ? hvac.cop_rated :
    hvac.cop_rated / hvac.curves.eir(T_wb_indoor - 273.15, T_condenser - 273.15)

"""
    available_cooling_capacity(hvac, T_wb_indoor, T_condenser) -> W

Total (sensible + latent) cooling capacity at the operating point,
``Q_{rated} \\, CAP_{FT}(T_{wb}, T_c)`` (temperatures in K).
"""
function available_cooling_capacity(hvac::IdealLoadsHVAC, T_wb_indoor::Real, T_condenser::Real)
    (hvac.curves === nothing || !isfinite(hvac.cooling_capacity)) && return hvac.cooling_capacity
    return hvac.cooling_capacity * hvac.curves.capacity(T_wb_indoor - 273.15, T_condenser - 273.15)
end

"""
    HVACBudget

What an [`IdealLoadsHVAC`](@ref) did over the last
`step_zone!(zone, forcing, dt, hvac)`, in W (kg s⁻¹ for water) and
non-negative unless stated:

  * `Q_heating` — sensible heat delivered to the zone air;
  * `Q_cooling_sensible`, `Q_cooling_latent` — sensible heat and latent
    heat (``L_v`` times the condensed vapour `W_condensed`) removed from
    the zone; `Q_cooling` is their sum, the coil load;
  * `COP` — operating cooling COP; `T_condenser` — condenser inlet
    temperature (K); `capacity` — available total cooling capacity;
  * `P_cooling` — electric power of the cooling equipment;
    `P_heating` — energy input of the heating (fuel for a boiler);
  * `Q_exhaust_sensible` (may be negative: heat-pump heating),
    `Q_exhaust_latent` — heat released to the outdoor air; `W_evaporated`
    — water evaporated by an evaporative condenser (kg s⁻¹);
  * `Q_unmet` — sensible load the capacity could not meet (positive:
    heating shortfall, negative: cooling shortfall; 0 when the setpoint
    is held).

First law of the plant, exact to round-off:
``Q_{exh,s} + Q_{exh,l} = P_{heating} + P_{cooling} + Q_{cooling} - Q_{heating}``;
for cooling alone this is ``Q_{cooling}(1 + 1/COP)``.
"""
struct HVACBudget{T<:AbstractFloat}
    Q_heating::T
    Q_cooling_sensible::T
    Q_cooling_latent::T
    Q_cooling::T
    W_condensed::T
    COP::T
    T_condenser::T
    capacity::T
    P_heating::T
    P_cooling::T
    Q_exhaust_sensible::T
    Q_exhaust_latent::T
    W_evaporated::T
    Q_unmet::T
end

function step_zone!(zone::BuildingZone{T}, forcing::ZoneForcing, dt::Real, hvac::IdealLoadsHVAC;
                    kwargs...) where {T}
    op = _hvac_operating_point(zone, forcing, hvac)
    sys = _assemble_zone!(zone, forcing, dt; kwargs...)
    # sensible load that puts the end-of-step air temperature on the setpoint
    x_free = sys.x0 + sys.R / sys.lhs
    Φ_req = x_free < hvac.T_heat ? sys.lhs * (hvac.T_heat - sys.x0) - sys.R :
            x_free > hvac.T_cool ? sys.lhs * (hvac.T_cool - sys.x0) - sys.R : zero(T)
    Φ = clamp(Φ_req, -op.cap_c, hvac.heating_capacity)
    x_new = sys.x0 + (sys.R + Φ) / sys.lhs
    W, L_v = _hvac_dehumidify(zone, forcing, dt, hvac, x_new, Φ, op.cap_c)
    budget = _complete_zone_step!(zone, forcing, dt, sys, Φ, W)
    return budget, _hvac_plant(zone, hvac, op, Φ, Φ_req, W, L_v)
end

# Equipment operating point from the start-of-step state.
function _hvac_operating_point(zone::BuildingZone, forcing::ZoneForcing, hvac::IdealLoadsHVAC)
    thermo, p = zone.thermo, zone.p_ref
    T_wb = wet_bulb_temperature(zone.T_air, p, zone.q_air, thermo)
    T_c = condenser_inlet_temperature(hvac, forcing.T_out, forcing.q_out, p; thermo = thermo)
    cop = cooling_cop(hvac, T_wb, T_c)
    cap_c = available_cooling_capacity(hvac, T_wb, T_c)
    cap_c >= 0 || (cap_c = zero(cap_c))
    return (; cop, T_c, cap_c)
end

# Dehumidification to RH_max at the new air temperature `x_new`: the vapour
# source W (kg s⁻¹, ≤ 0) and the latent heat of vaporisation used.
function _hvac_dehumidify(zone::BuildingZone{T}, forcing::ZoneForcing, dt, hvac::IdealLoadsHVAC,
                          x_new, Φ, cap_c) where {T}
    thermo, p = zone.thermo, zone.p_ref
    W = zero(T)
    L_v = latent_heat_vapor(x_new, thermo)
    if hvac.dehumidify && forcing.q_out !== nothing
        q_free, _, _ = _moisture_solve(zone, forcing, dt, x_new, zero(T))
        q_max = specific_humidity_from_relative_humidity(x_new, p, hvac.RH_max, thermo)
        if q_free > q_max
            M, n = _air_mass(zone), _exchange_rate(zone)
            W_req = (M / T(dt) + M * n) * (q_max - q_free)              # < 0
            avail = max(cap_c - max(-Φ, zero(T)), zero(T))              # sensible first
            W = max(W_req, -avail / L_v)
        end
    end
    return W, L_v
end

# The plant: energy input and heat rejection for a sensible load Φ (> 0
# heating) and a vapour source W.
function _hvac_plant(zone::BuildingZone, hvac::IdealLoadsHVAC, op, Φ::T, Φ_req, W, L_v) where {T}
    Q_h = max(Φ, zero(T)); Q_cs = max(-Φ, zero(T)); Q_cl = -W * L_v
    Q_c = Q_cs + Q_cl
    P_c = Q_c > 0 ? Q_c / op.cop : zero(T)
    P_h = Q_h / hvac.heating_efficiency
    Q_rej = Q_c + P_c
    f_lat = hvac.condenser === :evaporative ? hvac.latent_fraction : zero(T)
    Q_exh_l = f_lat * Q_rej
    Q_exh_s = (Q_rej - Q_exh_l) + (P_h - Q_h)
    W_evap = Q_exh_l / latent_heat_vapor(op.T_c, zone.thermo)
    return HVACBudget{T}(Q_h, Q_cs, Q_cl, Q_c, -W, op.cop, op.T_c, op.cap_c, P_h, P_c,
                         Q_exh_s, Q_exh_l, W_evap, Φ_req - Φ)
end

# =========================================================================
# 3c. Occupancy, lighting and equipment schedules; internal gains
# =========================================================================

"""
    DayProfile(until, values)
    DayProfile(until_hour => value, ...)
    DayProfile(value)

A piecewise-constant daily profile in the form of an EnergyPlus
`Schedule:Compact` day: `values[i]` holds from `until[i-1]` (0 for the
first entry) up to `until[i]` hours of local clock time. The breakpoints
must increase strictly and the last must be 24. `DayProfile(value)` is
constant all day.

Evaluate with `p(hour)` (the hour is taken modulo 24; a breakpoint
belongs to the *next* interval, as in EnergyPlus). The daily integral —
the "equivalent full-load hours" of a fractional schedule — is
[`full_load_hours`](@ref)`(p)`.
"""
struct DayProfile
    until::Vector{Float64}
    values::Vector{Float64}
    function DayProfile(until::AbstractVector{<:Real}, values::AbstractVector{<:Real})
        length(until) == length(values) >= 1 ||
            throw(ArgumentError("DayProfile: need as many breakpoints as values (≥ 1)"))
        u = Float64.(until)
        (u[1] > 0 && all(diff(u) .> 0)) ||
            throw(ArgumentError("DayProfile: breakpoints must be positive and strictly increasing"))
        u[end] == 24 || throw(ArgumentError("DayProfile: the last breakpoint must be 24 h"))
        all(isfinite, values) || throw(ArgumentError("DayProfile: values must be finite"))
        return new(u, Float64.(values))
    end
end
DayProfile(pairs::Pair...) = DayProfile([first(p) for p in pairs], [last(p) for p in pairs])
DayProfile(value::Real) = DayProfile([24.0], [value])

function (p::DayProfile)(hour::Real)
    h = mod(hour, 24)
    @inbounds for i in eachindex(p.until)
        h < p.until[i] && return p.values[i]
    end
    return p.values[end]
end

# ∫ p dh over [a, b] ⊂ [0, 24] (hours × value)
function _profile_integral(p::DayProfile, a::Real, b::Real)
    s = 0.0; lo = 0.0
    for i in eachindex(p.until)
        hi = p.until[i]
        s += p.values[i] * max(0.0, min(b, hi) - max(a, lo))
        lo = hi
    end
    return s
end

"""
    WeeklySchedule(weekday, saturday = weekday, sunday = saturday;
                   holidays = Int[], start_weekday = 1)

A week of [`DayProfile`](@ref)s: `weekday` Monday–Friday, `saturday`,
and `sunday`, which also applies to every day number in `holidays`
(EnergyPlus `Weekdays` / `Saturday` / `Sunday Holidays AllOtherDays`).
A profile may be given as a number (constant all day).

Time is counted in seconds from 00:00 of day 1, whose weekday is
`start_weekday` (1 = Monday, …, 7 = Sunday); `holidays` are day numbers
in the same count. Evaluate with [`schedule_value`](@ref) or exactly
averaged over a step with [`schedule_mean`](@ref).
"""
struct WeeklySchedule
    weekday::DayProfile
    saturday::DayProfile
    sunday::DayProfile
    holidays::Vector{Int}
    start_weekday::Int
end

_profile(p::DayProfile) = p
_profile(v::Real) = DayProfile(v)

function WeeklySchedule(weekday, saturday = weekday, sunday = saturday;
                        holidays::AbstractVector{<:Integer} = Int[], start_weekday::Integer = 1)
    1 <= start_weekday <= 7 || throw(ArgumentError("WeeklySchedule: start_weekday must be 1 … 7"))
    return WeeklySchedule(_profile(weekday), _profile(saturday), _profile(sunday),
                          sort!(Int.(collect(holidays))), Int(start_weekday))
end

"Weekday (1 = Monday … 7 = Sunday) of day number `day` (day 1 = `s.start_weekday`)."
_weekday(s::WeeklySchedule, day::Integer) = mod(s.start_weekday - 1 + day - 1, 7) + 1

function _day_profile(s::WeeklySchedule, day::Integer)
    insorted(day, s.holidays) && return s.sunday
    wd = _weekday(s, day)
    return wd <= 5 ? s.weekday : wd == 6 ? s.saturday : s.sunday
end

"""
    schedule_value(s::WeeklySchedule, t) -> value

The schedule at `t` seconds after 00:00 of day 1.
"""
function schedule_value(s::WeeklySchedule, t::Real)
    day = fld(t, 86400)
    return _day_profile(s, Int(day) + 1)((t - 86400 * day) / 3600)
end

"""
    schedule_mean(s::WeeklySchedule, t0, t1) -> value

The exact mean of the piecewise-constant schedule over ``[t_0, t_1]``
(seconds). Used as the step-mean of a scheduled gain, it makes a model
step of any length deliver exactly the scheduled energy; with
`t0 == t1` it is `schedule_value(s, t0)`.
"""
function schedule_mean(s::WeeklySchedule, t0::Real, t1::Real)
    t1 >= t0 || throw(ArgumentError("schedule_mean: t1 must not precede t0"))
    t1 == t0 && return schedule_value(s, t0)
    acc = 0.0
    d0 = Int(fld(t0, 86400)); d1 = Int(fld(t1, 86400))
    for d in d0:d1
        a = max(t0 - 86400d, 0.0) / 3600
        b = min(t1 - 86400d, 86400.0) / 3600
        b > a && (acc += _profile_integral(_day_profile(s, d + 1), a, b))
    end
    return acc * 3600 / (t1 - t0)
end

"""
    full_load_hours(p::DayProfile) -> h
    full_load_hours(s::WeeklySchedule) -> h per week

The integral of a schedule over a day (or a week without holidays):
for a fractional schedule, the number of hours at full design level
that deliver the same energy.
"""
full_load_hours(p::DayProfile) = _profile_integral(p, 0.0, 24.0)
full_load_hours(s::WeeklySchedule) =
    5 * full_load_hours(s.weekday) + full_load_hours(s.saturday) + full_load_hours(s.sunday)

# ---- occupants ---------------------------------------------------------------

# EnergyPlus InternalHeatGains.cc, `C` (the Engineering Reference prints
# 6.461927 and 0.0000589172 for C₀ and C₅; the source, which EnergyPlus
# runs, has the values below — the two differ by < 0.01 W).
const _EPLUS_PEOPLE_SENSIBLE = (6.4611027, 0.946892, 0.0000255737, 7.139322, -0.0627909,
                                0.0000589271, -0.198550, 0.000940018, -0.00000149532)

"""
    occupant_sensible_heat(activity, T_air) -> W per person

Sensible (radiant + convective) part of an occupant's metabolic heat
`activity` (W per person) at air temperature `T_air` (K); the rest,
`activity - S`, is latent (evaporation and respiration). The EnergyPlus
correlation (Engineering Reference, *Zone internal gains — heat gain
from people*; coefficients from the EnergyPlus source)

```math
S = C_0 + C_1 M + C_2 M^2 + T(C_3 + C_4 M + C_5 M^2) + T^2 (C_6 + C_7 M + C_8 M^2),
```

``T`` in °C, is a fit to Carrier's (1965) sensible gains at average
adjusted metabolic rates of 350–1450 Btu h⁻¹ and 70–82 °F, constrained
to ``S = 0`` at 96 °F and ``S = M`` at 30 °F. The result is limited to
``[0, M]`` as in EnergyPlus.
"""
function occupant_sensible_heat(activity::Real, T_air::Real)
    activity >= 0 || throw(ArgumentError("occupant_sensible_heat: activity must be ≥ 0"))
    C = _EPLUS_PEOPLE_SENSIBLE
    M = float(activity); T = T_air - 273.15
    S = C[1] + M * (C[2] + M * C[3]) + T * ((C[4] + M * (C[5] + M * C[6])) +
                                             T * (C[7] + M * (C[8] + M * C[9])))
    return clamp(S, zero(S), M)
end

# ---- internal gains ----------------------------------------------------------

"""
    InternalGains(; floor_area, floor_area_per_person = Inf, activity = 120,
                  people_radiant_fraction = 0.3,
                  lighting_power_density = 0, lighting_radiant_fraction = 0.4,
                  lighting_visible_fraction = 0.2, lighting_return_air_fraction = 0,
                  equipment_power_density = 0, equipment_radiant_fraction = 0.5,
                  equipment_latent_fraction = 0, equipment_lost_fraction = 0,
                  occupancy = WeeklySchedule(1), lighting = WeeklySchedule(1),
                  equipment = WeeklySchedule(1))

Scheduled internal heat gains of a zone of `floor_area` m², after the
EnergyPlus `People`, `Lights` and `ElectricEquipment` objects:

  * **People**: `floor_area / floor_area_per_person` occupants at design,
    times the `occupancy` schedule, each releasing `activity` W of
    metabolic heat, split by [`occupant_sensible_heat`](@ref) into
    sensible (a fraction `people_radiant_fraction` radiant) and latent.
  * **Lighting**: `lighting_power_density` W m⁻² times the `lighting`
    schedule. Fractions: radiant (long-wave) and visible (short-wave) —
    both go to the interior surfaces here — and return air (carried out
    by the return air stream and lost to the zone); the rest is
    convective.
  * **Equipment**: `equipment_power_density` W m⁻² times the
    `equipment` schedule, split into radiant, latent, lost (vented or
    carried out) and convective.

The electric power of lighting and equipment is the energy demand they
represent; all of it, whatever the split, ends up as heat somewhere.
[`internal_gains`](@ref) evaluates the gains; see
[`office_internal_gains`](@ref) for the DOE medium-office reference
values.
"""
struct InternalGains
    floor_area::Float64
    occupant_density::Float64
    activity::Float64
    people_radiant_fraction::Float64
    lighting_power_density::Float64
    lighting_radiant_fraction::Float64
    lighting_visible_fraction::Float64
    lighting_return_air_fraction::Float64
    equipment_power_density::Float64
    equipment_radiant_fraction::Float64
    equipment_latent_fraction::Float64
    equipment_lost_fraction::Float64
    occupancy::WeeklySchedule
    lighting::WeeklySchedule
    equipment::WeeklySchedule
end

_in01(x) = 0 <= x <= 1

function InternalGains(; floor_area::Real, floor_area_per_person::Real = Inf,
                       activity::Real = 120.0, people_radiant_fraction::Real = 0.3,
                       lighting_power_density::Real = 0.0, lighting_radiant_fraction::Real = 0.4,
                       lighting_visible_fraction::Real = 0.2,
                       lighting_return_air_fraction::Real = 0.0,
                       equipment_power_density::Real = 0.0,
                       equipment_radiant_fraction::Real = 0.5,
                       equipment_latent_fraction::Real = 0.0, equipment_lost_fraction::Real = 0.0,
                       occupancy::WeeklySchedule = WeeklySchedule(1.0),
                       lighting::WeeklySchedule = WeeklySchedule(1.0),
                       equipment::WeeklySchedule = WeeklySchedule(1.0))
    floor_area > 0 || throw(ArgumentError("InternalGains: floor_area must be positive"))
    floor_area_per_person > 0 || throw(ArgumentError("InternalGains: floor_area_per_person must be positive"))
    (activity >= 0 && lighting_power_density >= 0 && equipment_power_density >= 0) ||
        throw(ArgumentError("InternalGains: activity and power densities must be ≥ 0"))
    all(_in01, (people_radiant_fraction, lighting_radiant_fraction, lighting_visible_fraction,
                lighting_return_air_fraction, equipment_radiant_fraction,
                equipment_latent_fraction, equipment_lost_fraction)) ||
        throw(ArgumentError("InternalGains: fractions must lie in [0, 1]"))
    lighting_radiant_fraction + lighting_visible_fraction + lighting_return_air_fraction <= 1 + 1e-12 ||
        throw(ArgumentError("InternalGains: lighting fractions sum to more than 1"))
    equipment_radiant_fraction + equipment_latent_fraction + equipment_lost_fraction <= 1 + 1e-12 ||
        throw(ArgumentError("InternalGains: equipment fractions sum to more than 1"))
    return InternalGains(floor_area, 1 / floor_area_per_person, activity, people_radiant_fraction,
                         lighting_power_density, lighting_radiant_fraction,
                         lighting_visible_fraction, lighting_return_air_fraction,
                         equipment_power_density, equipment_radiant_fraction,
                         equipment_latent_fraction, equipment_lost_fraction,
                         occupancy, lighting, equipment)
end

"""
    InternalGainRates

What [`internal_gains`](@ref) returns, in W (occupants in persons):

  * `occupants`, `people_sensible`, `people_latent`;
  * `lighting`, `equipment` — their electric power (the demand);
  * the split seen by the zone: `convective` (to the air node),
    `radiative` (to the interior surfaces: radiant long-wave plus visible
    light), `latent` (vapour) and `lost` (removed by return or exhaust air
    without entering the zone balance).

Conservation: `people_sensible + people_latent + lighting + equipment ==
convective + radiative + latent + lost` to round-off.
"""
struct InternalGainRates
    occupants::Float64
    people_sensible::Float64
    people_latent::Float64
    lighting::Float64
    equipment::Float64
    convective::Float64
    radiative::Float64
    latent::Float64
    lost::Float64
end

"""
    internal_gains(g::InternalGains, t, dt = 0; T_air = 297.15) -> InternalGainRates

The internal gains at `t` seconds after 00:00 of day 1 — or, with
`dt > 0`, their exact mean over ``[t, t + dt]`` ([`schedule_mean`](@ref)),
which is what a zone step of length `dt` should receive. The occupants'
sensible/latent split uses the indoor air temperature `T_air` (K).
"""
function internal_gains(g::InternalGains, t::Real, dt::Real = 0.0; T_air::Real = 297.15)
    dt >= 0 || throw(ArgumentError("internal_gains: dt must be ≥ 0"))
    f(s) = schedule_mean(s, t, t + dt)
    n = g.occupant_density * g.floor_area * f(g.occupancy)
    S = n * occupant_sensible_heat(g.activity, T_air)
    Lat = n * g.activity - S
    L = g.lighting_power_density * g.floor_area * f(g.lighting)
    E = g.equipment_power_density * g.floor_area * f(g.equipment)
    rad = g.people_radiant_fraction * S +
          (g.lighting_radiant_fraction + g.lighting_visible_fraction) * L +
          g.equipment_radiant_fraction * E
    lat = Lat + g.equipment_latent_fraction * E
    lost = g.lighting_return_air_fraction * L + g.equipment_lost_fraction * E
    conv = (S + Lat + L + E) - rad - lat - lost
    return InternalGainRates(n, S, Lat, L, E, conv, rad, lat, lost)
end

"""
    ZoneForcing(r::InternalGainRates; include_latent = true, kwargs...)

A [`ZoneForcing`](@ref) whose internal gains are `r`: `Q_internal =
r.convective + r.radiative` with the step's own `radiative_fraction`,
and `Q_latent = r.latent` (which needs `q_out` in `kwargs`; pass
`include_latent = false` to leave the vapour out). `kwargs` supply the
rest (`T_out`, `Q_solar`, …).
"""
function ZoneForcing(r::InternalGainRates; include_latent::Bool = true, kwargs...)
    Q = r.convective + r.radiative
    fr = Q > 0 ? r.radiative / Q : 0.0
    return ZoneForcing(; Q_internal = Q, radiative_fraction = fr,
                       Q_latent = include_latent ? r.latent : 0.0, kwargs...)
end

# ---- thermostat schedules ----------------------------------------------------

"""
    ThermostatSchedule(heating, cooling; availability = nothing)

Scheduled setpoints (K) of an [`IdealLoadsHVAC`](@ref): `heating` and
`cooling` are [`WeeklySchedule`](@ref)s (or constants); `availability`,
if given, switches the system off wherever it is 0. Apply with
[`scheduled_hvac`](@ref).
"""
struct ThermostatSchedule
    heating::WeeklySchedule
    cooling::WeeklySchedule
    availability::Union{Nothing, WeeklySchedule}
end
_weekly(s::WeeklySchedule) = s
_weekly(v::Real) = WeeklySchedule(v)
ThermostatSchedule(heating, cooling; availability = nothing) =
    ThermostatSchedule(_weekly(heating), _weekly(cooling),
                       availability === nothing ? nothing : _weekly(availability))

"""
    scheduled_hvac(hvac::IdealLoadsHVAC, ts::ThermostatSchedule, t) -> IdealLoadsHVAC

`hvac` with the setpoints of `ts` at `t` seconds (the value at the start
of the step; setpoint changes are instantaneous, as in EnergyPlus) and
with both capacities set to 0 where the availability schedule is 0.
"""
function scheduled_hvac(hvac::IdealLoadsHVAC{T, C}, ts::ThermostatSchedule, t::Real) where {T, C}
    Th = T(schedule_value(ts.heating, t)); Tc = T(schedule_value(ts.cooling, t))
    Th <= Tc || throw(ArgumentError("scheduled_hvac: heating setpoint above cooling setpoint at t = $t"))
    on = ts.availability === nothing || schedule_value(ts.availability, t) > 0
    return IdealLoadsHVAC{T, C}(Th, Tc, on ? hvac.heating_capacity : zero(T),
                                on ? hvac.cooling_capacity : zero(T), hvac.heating,
                                hvac.heating_efficiency, hvac.cop_rated, hvac.curves,
                                hvac.condenser, hvac.evaporative_effectiveness,
                                hvac.latent_fraction, hvac.dehumidify, hvac.RH_max)
end

# ---- the DOE medium-office reference -------------------------------------------

const _C = 273.15

"""
    DOE_MEDIUM_OFFICE_SCHEDULES

Schedules of the U.S. DOE commercial reference building *Medium office,
new construction (ASHRAE 90.1-2004)* (Deru et al. 2011), transcribed
from the EnergyPlus model `RefBldgMediumOfficeNew2004_Chicago.idf`
(NREL/EnergyPlus `testfiles/`), `Schedule:Compact` objects:

| field | IDF schedule | |
|:--|:--|:--|
| `occupancy` | `BLDG_OCC_SCH` | fraction of design occupants |
| `lighting` | `BLDG_LIGHT_SCH` | fraction of design lighting power |
| `equipment` | `BLDG_EQUIP_SCH` | fraction of design plug load |
| `heating_setpoint` | `HTGSETP_SCH` | K (15.6 / 21.0 °C) |
| `cooling_setpoint` | `CLGSETP_SCH` | K (26.7 / 24.0 °C) |
| `hvac_operation` | `HVACOperationSchd` | 0/1 |

Design-day variants are omitted; `Sunday Holidays AllOtherDays` is the
`sunday` profile.

Reference: Deru, M. et al. (2011). *U.S. Department of Energy Commercial
Reference Building Models of the National Building Stock.* NREL/TP-5500-46861.
"""
const DOE_MEDIUM_OFFICE_SCHEDULES = (
    occupancy = WeeklySchedule(
        DayProfile(6 => 0.0, 7 => 0.1, 8 => 0.2, 12 => 0.95, 13 => 0.5, 17 => 0.95, 18 => 0.7,
                   20 => 0.4, 22 => 0.1, 24 => 0.05),
        DayProfile(6 => 0.0, 8 => 0.1, 14 => 0.5, 17 => 0.1, 24 => 0.0),
        DayProfile(0.0)),
    lighting = WeeklySchedule(
        DayProfile(5 => 0.05, 7 => 0.1, 8 => 0.3, 17 => 0.9, 18 => 0.7, 20 => 0.5, 22 => 0.3,
                   23 => 0.1, 24 => 0.05),
        DayProfile(6 => 0.05, 8 => 0.1, 14 => 0.5, 17 => 0.15, 24 => 0.05),
        DayProfile(0.05)),
    equipment = WeeklySchedule(
        DayProfile(8 => 0.40, 12 => 0.90, 13 => 0.80, 17 => 0.90, 18 => 0.80, 20 => 0.60,
                   22 => 0.50, 24 => 0.40),
        DayProfile(6 => 0.30, 8 => 0.4, 14 => 0.5, 17 => 0.35, 24 => 0.30),
        DayProfile(0.30)),
    heating_setpoint = WeeklySchedule(
        DayProfile(6 => _C + 15.6, 22 => _C + 21.0, 24 => _C + 15.6),
        DayProfile(6 => _C + 15.6, 18 => _C + 21.0, 24 => _C + 15.6),
        DayProfile(_C + 15.6)),
    cooling_setpoint = WeeklySchedule(
        DayProfile(6 => _C + 26.7, 22 => _C + 24.0, 24 => _C + 26.7),
        DayProfile(6 => _C + 26.7, 18 => _C + 24.0, 24 => _C + 26.7),
        DayProfile(_C + 26.7)),
    hvac_operation = WeeklySchedule(
        DayProfile(6 => 0.0, 22 => 1.0, 24 => 0.0),
        DayProfile(6 => 0.0, 18 => 1.0, 24 => 0.0),
        DayProfile(0.0)),
)

"""
    office_internal_gains(floor_area; start_weekday = 1, holidays = Int[], kwargs...) -> InternalGains

The internal gains of the DOE medium-office reference building
(`RefBldgMediumOfficeNew2004`, see [`DOE_MEDIUM_OFFICE_SCHEDULES`](@ref)):
18.58 m² per person at 120 W (`People`, radiant fraction 0.3,
`ACTIVITY_SCH`), lighting 10.76 W m⁻² (radiant 0.4, visible 0.2, return
air 0.4), plug loads 10.76 W m⁻² (radiant 0.5, no latent). `kwargs`
override any [`InternalGains`](@ref) field; note that the reference
building sends 40 % of the lighting heat to a return-air plenum, which
a single zone without a plenum loses — set
`lighting_return_air_fraction = 0` to keep it in the zone.
"""
function office_internal_gains(floor_area::Real; start_weekday::Integer = 1,
                               holidays::AbstractVector{<:Integer} = Int[], kwargs...)
    S = DOE_MEDIUM_OFFICE_SCHEDULES
    cal(s) = WeeklySchedule(s.weekday, s.saturday, s.sunday; holidays, start_weekday)
    return InternalGains(; floor_area, floor_area_per_person = 18.58, activity = 120.0,
                         people_radiant_fraction = 0.3, lighting_power_density = 10.76,
                         lighting_radiant_fraction = 0.4, lighting_visible_fraction = 0.2,
                         lighting_return_air_fraction = 0.4, equipment_power_density = 10.76,
                         equipment_radiant_fraction = 0.5, equipment_latent_fraction = 0.0,
                         occupancy = cal(S.occupancy), lighting = cal(S.lighting),
                         equipment = cal(S.equipment), kwargs...)
end

"""
    office_thermostat(; start_weekday = 1, holidays = Int[], availability = true) -> ThermostatSchedule

The DOE medium-office thermostat: 21 °C heating / 24 °C cooling while
occupied (weekdays 06–22 h, Saturdays 06–18 h), set back to 15.6 / 26.7 °C
otherwise, with the `HVACOperationSchd` availability (the reference
building's night-cycle manager, which restarts the system on the setback
setpoints, is not modelled: `availability = false` keeps the system on
with the setback instead).
"""
function office_thermostat(; start_weekday::Integer = 1,
                           holidays::AbstractVector{<:Integer} = Int[], availability::Bool = true)
    S = DOE_MEDIUM_OFFICE_SCHEDULES
    cal(s) = WeeklySchedule(s.weekday, s.saturday, s.sunday; holidays, start_weekday)
    return ThermostatSchedule(cal(S.heating_setpoint), cal(S.cooling_setpoint);
                              availability = availability ? cal(S.hvac_operation) : nothing)
end

# =========================================================================
# 3d. Anthropogenic heat: what the building releases to the outdoors
# =========================================================================

"""
    BuildingHeatRelease

Where the energy a building uses goes, over one zone step (issue #29).
All terms are in W. The released terms are positive when heat goes **from
the building to the outdoor environment**.

**Energy consumed**

  * `P_electric` — lighting and plug loads, cooling equipment, and
    electric heating (resistance or heat pump);
  * `P_fuel` — boiler fuel input;
  * `Q_metabolic` — occupants' metabolic heat (sensible + latent).

**Released to the outdoor air directly** — what an urban canopy model
receives as the building part of the anthropogenic heat flux ``Q_F``:

  * `Q_exhaust_sensible`, `Q_exhaust_latent` — HVAC plant heat rejection
    (condenser, cooling tower, boiler flue; see [`HVACBudget`](@ref));
  * `Q_ventilation_sensible` — ``H_{ve}(T_{air} - T_{out})``, carried out
    by infiltration and ventilation air;
  * `Q_ventilation_latent` — ``L_v`` times the net vapour carried out by
    the same air (``-L_v W_{ventilation}``);
  * `Q_windows` — conduction through glazing, ``U_w A_w (T_{air} - T_{out})``;
  * `Q_lost` — internal gains removed by return or exhaust air without
    entering the zone (the `lost` part of [`InternalGainRates`](@ref));
  * `Q_latent_vented` — latent internal gains that did not enter the
    zone's vapour balance (all of them when the zone humidity is not
    advanced), assumed vented;
  * `Q_envelope_air` — net heat leaving the outer faces of `:air`
    [`EnvelopeElement`](@ref)s. Their outer face has no surface energy
    balance of its own, so what they give up is a direct source for the
    air (negative in sunshine, when they absorb more than they lose).

**Released through the urban surfaces**

  * `Q_envelope_facets` — heat conducted out through the outer faces of
    [`FacetElement`](@ref)s, ``-Σ A G``. It is *not* a direct source: it is
    already in the facet's surface energy balance, where it raises the
    skin temperature and leaves as the facet's ``H``, ``LE`` and emitted
    long-wave. Adding it to the air as well would count it twice.
  * `Q_envelope_ground` — heat conducted into the ground.

**Other inputs and storage**

  * `Q_solar` — solar gain transmitted through the glazing (not
    anthropogenic);
  * `Q_prescribed` — a prescribed `ZoneForcing.Q_hvac`, an energy input
    from a system the model does not represent;
  * `storage_sensible` — rate of change of the zone's sensible heat
    content, ``ΔE/Δt`` ([`zone_heat_content`](@ref));
  * `storage_latent` — rate of change of the latent heat held by the zone
    air's vapour, ``L_v ρV Δq/Δt``.

# The first law

Summing the zone balance ([`ZoneBudget`](@ref)), the vapour balance, the
plant ([`HVACBudget`](@ref)) and the split of the internal gains gives,
exactly for every step,

```math
Q_{released} + \\frac{ΔE}{Δt} + L_v ρV\\frac{Δq}{Δt}
  = P_{electric} + P_{fuel} + Q_{metabolic} + Q_{solar} + Q_{prescribed},
```

with ``Q_{released}`` the sum of all released terms,
[`total_heat_release`](@ref). Over a period at the end of which the
building is back in its initial state, the heat released equals the
energy consumed plus the transmitted sunshine: that is the premise of
the inventory method of estimating ``Q_F`` (Sailor 2011), which a
building energy model resolves in time, because storage delays the
release (a building heated at night gives the heat back through the
next day).

# References

  * Sailor, D. J. (2011). A review of methods for estimating anthropogenic
    heat and moisture emissions in the urban environment. *Int. J.
    Climatol.*, 31, 189-199.
  * Bueno, B., Pigeon, G., Norford, L. K., Zibouche, K. & Marchadier, C.
    (2012). Development and evaluation of a building energy model
    integrated in the TEB scheme. *Geosci. Model Dev.*, 5, 433-448
    (TEB-BEM: HVAC waste heat and the air exchanged by the building are
    released to the canyon; the envelope exchanges through the wall and
    roof energy balances).
"""
struct BuildingHeatRelease{T<:AbstractFloat}
    P_electric::T
    P_fuel::T
    Q_metabolic::T
    Q_exhaust_sensible::T
    Q_exhaust_latent::T
    Q_ventilation_sensible::T
    Q_ventilation_latent::T
    Q_windows::T
    Q_lost::T
    Q_latent_vented::T
    Q_envelope_air::T
    Q_envelope_facets::T
    Q_envelope_ground::T
    Q_solar::T
    Q_prescribed::T
    storage_sensible::T
    storage_latent::T
end

"""
    building_heat_release(zone, forcing, budget; dt, gains = nothing,
                          hvac = nothing, hvac_budget = nothing,
                          q_before) -> BuildingHeatRelease

The [`BuildingHeatRelease`](@ref) of the step that `zone` has just taken
with `forcing` and returned `budget` ([`ZoneBudget`](@ref)); `hvac` and
`hvac_budget` are the [`IdealLoadsHVAC`](@ref) and its
[`HVACBudget`](@ref) if one was used.

  * `gains` — the [`InternalGainRates`](@ref) the step was forced with.
    They say how much of the internal gains is electric and how much is
    metabolic, and what was lost or vented before reaching the zone.
    Without them, `forcing.Q_internal + forcing.Q_latent` is counted as
    electric power and nothing is lost.
  * `q_before` — the zone's specific humidity before the step, for the
    latent storage term (default: no change).
  * `dt` — the step length (s); only the storage terms need it.

The released terms are read from the budget and from the elements'
last-step fluxes (`F_outer`); the storage terms are the budget's sum of
sources and the vapour balance, so the first law holds to round-off.
"""
function building_heat_release(zone::BuildingZone{T}, forcing::ZoneForcing, budget::ZoneBudget;
                               dt::Real, gains::Union{Nothing, InternalGainRates} = nothing,
                               hvac::Union{Nothing, IdealLoadsHVAC} = nothing,
                               hvac_budget::Union{Nothing, HVACBudget} = nothing,
                               q_before::Real = zone.q_air) where {T}
    (hvac === nothing) == (hvac_budget === nothing) ||
        throw(ArgumentError("building_heat_release: pass both `hvac` and `hvac_budget`, or neither"))
    dt > 0 || throw(ArgumentError("building_heat_release: dt must be positive"))
    # internal gains: electric, metabolic, lost before reaching the zone
    if gains === nothing
        P_int = T(forcing.Q_internal) + T(forcing.Q_latent)
        Q_met = zero(T); lost = zero(T); lat_vented = zero(T)
    else
        P_int = T(gains.lighting + gains.equipment)
        Q_met = T(gains.people_sensible + gains.people_latent)
        lost = T(gains.lost)
        lat_vented = T(gains.latent) - T(forcing.Q_latent)
    end
    # plant
    P_el = P_int; P_fuel = zero(T); Q_xs = zero(T); Q_xl = zero(T)
    if hvac_budget !== nothing
        P_el += hvac_budget.P_cooling
        if hvac.heating === :boiler
            P_fuel += hvac_budget.P_heating
        else
            P_el += hvac_budget.P_heating
        end
        Q_xs, Q_xl = hvac_budget.Q_exhaust_sensible, hvac_budget.Q_exhaust_latent
    end
    Q_prescribed = T(forcing.Q_hvac)
    # envelope, by what is on the other side of each outer face
    env_air = zero(T); env_fac = zero(T); env_gnd = zero(T)
    for el in zone.envelope
        if el isa FacetElement
            env_fac -= el.area * el.F_outer
        elseif el.exterior === :air
            env_air -= el.area * el.F_outer
        elseif el.exterior === :ground
            env_gnd -= el.area * el.F_outer
        end
    end
    L = latent_heat_vapor(budget.T_air, zone.thermo)
    Q_vl = -L * budget.W_ventilation
    # storage, from the zone's own balances
    Q_hvac_air = budget.Q_hvac   # prescribed + ideal load, to the air node
    dE = budget.Q_exterior + budget.Q_windows + budget.Q_ventilation + budget.Q_internal +
         budget.Q_solar + Q_hvac_air
    dL = L * _air_mass(zone) * (budget.q_air - T(q_before)) / T(dt)
    return BuildingHeatRelease{T}(P_el, P_fuel, Q_met, Q_xs, Q_xl, -budget.Q_ventilation, Q_vl,
                                  -budget.Q_windows, lost, lat_vented, env_air, env_fac, env_gnd,
                                  budget.Q_solar, Q_prescribed, dE, dL)
end

"Energy consumed by the building (W): ``P_{electric} + P_{fuel} + Q_{metabolic}``."
energy_consumption(r::BuildingHeatRelease) = r.P_electric + r.P_fuel + r.Q_metabolic

"""
    total_heat_release(r::BuildingHeatRelease) -> W

Everything the building gives to the outdoor environment: the direct
sensible and latent sources, the conduction out through facet outer
faces, and into the ground.
"""
total_heat_release(r::BuildingHeatRelease) =
    anthropogenic_sensible_heat(r) + anthropogenic_latent_heat(r) +
    r.Q_envelope_facets + r.Q_envelope_ground

"""
    anthropogenic_sensible_heat(r::BuildingHeatRelease) -> W

The sensible heat the building puts **directly** into the outdoor air:
HVAC exhaust, air exchange, glazing, lost internal gains and the
exchange of `:air` [`EnvelopeElement`](@ref)s. Conduction through
[`FacetElement`](@ref)s is excluded: it reaches the air through the
facet's own surface energy balance.
"""
anthropogenic_sensible_heat(r::BuildingHeatRelease) =
    r.Q_exhaust_sensible + r.Q_ventilation_sensible + r.Q_windows + r.Q_lost + r.Q_envelope_air

"""
    anthropogenic_latent_heat(r::BuildingHeatRelease) -> W

Latent heat (``L_v`` times the water vapour) the building adds to the
outdoor air: evaporative heat rejection, vapour carried out by the air
exchange, and vented latent gains.
"""
anthropogenic_latent_heat(r::BuildingHeatRelease) =
    r.Q_exhaust_latent + r.Q_ventilation_latent + r.Q_latent_vented

# =========================================================================
# 3d'. The vehicle part of the anthropogenic heat (ONE-e, #454; moved from
#      the urban case's `traffic_heat` coupling)
# =========================================================================

"""
    traffic_heat_flux(Q_urban, n_cells, n_road) -> W m⁻²

The heat flux released over each of `n_road` road cells when the vehicle
part of the anthropogenic heat flux, ``Q_V = `` `Q_urban` W m⁻² of urban
area (`n_cells` columns), is concentrated on the roads:
``Q_V n_\\mathrm{cells} / n_\\mathrm{road}``, so the domain mean is
`Q_urban`. Sailor (2011, Int. J. Climatol. 31, 189) splits
``Q_F = Q_V + Q_H + Q_M`` (vehicles, buildings, metabolism); the building
part is [`building_heat_release`](@ref).
"""
traffic_heat_flux(Q_urban::Real, n_cells::Integer, n_road::Integer) = Q_urban * n_cells / n_road

"""
    add_traffic_heat!(sim, road, n_cells, dt; Q_urban = 8.0, ρcp = 1005.0)

Vehicle heat: a constant `Q_urban` W m⁻² of urban area (`n_cells`
columns), released into the first air layer over the `road` cells
(column indices), ``Δθ = q Δt / (ρc_p Δz)`` with the road flux
``q`` of [`traffic_heat_flux`](@ref). `sim` is a dynamical core (its
`θ` and `mesh`); `ρcp` (J m⁻³ K⁻¹) is the heat that raises θ of that
layer by 1 K, ``ρ̄_1 c_p Π̄_1`` (the base state's on the anelastic core).
The default 8.0 W m⁻² is the traffic heat Bueno et al. (2012, Geosci.
Model Dev. 5, 433, Table 3) prescribe for the Toulouse centre (their
Table 4 uses the same value for Athens); the tables give no source for
it, and it is not a Berlin inventory.
"""
function add_traffic_heat!(sim, road::AbstractVector{CartesianIndex{2}}, n_cells::Integer, dt::Real;
                           Q_urban::Real = 8.0, ρcp::Real = 1005.0)
    isempty(road) && return sim
    q = traffic_heat_flux(Q_urban, n_cells, length(road))
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    for I in road
        i, j = Tuple(I)
        sim.θ.data[Hx+i, Hy+j, Hz+1] += q / ρcp * dt / sim.mesh.dz
    end
    return sim
end

# =========================================================================
# 3f. The city-mean building (ONE-e, #454; moved from the urban case's
#     `buildings_citymean` coupling, which now only wires it in)
# =========================================================================
#
# A surface scheme with no building interior (one temperature per surface
# cell) represents the city's buildings by ONE city-mean `BuildingZone`
# whose volume, footprint and wall area are the scene's totals, run with
# the DOE medium-office schedules (#33). Its walls and roof are conduction
# columns that exchange with the canyon air through the ISO 6946 outside
# coefficient but absorb no sunshine: the surface cells already absorb it,
# and a first version with sol-air walls counted it twice (-8.7 W/m²
# returned to the air in a one-hour run). A version that held the
# envelope's outer face at the slab skin temperature was dropped: that
# stiff interface, exchanged explicitly every 10 min, swung the slab budget
# by -64 then +27 W/m² (domain mean) between two radiation updates.
# The release is mapped onto the grid: HVAC exhaust and the roof envelope
# at the roofs (exhaust weighted by the floors above each roof cell); air
# exchange, glazing, lost gains and the wall envelope into the canyon (the
# wall cells next to buildings). Sunshine transmitted by the windows is
# taken out of the wall cells' surface budget (`S`), which would otherwise
# absorb it too. Totals are conserved: Σ (H + LE) dx dy = the zone's direct
# release, Σ S dx dy = -Q_solar. `UrbanFacetScene` + `step_buildings!`
# resolve each building behind its own facets instead.

"""
    CITY_BUILDING_L_V

The latent heat (J kg⁻¹, 2.45e6, its value near 20 °C) with which
[`apply_building_heat!`](@ref) converts the city-mean building's latent
release into vapour; a ledger of that vapour converts with the same.
"""
const CITY_BUILDING_L_V = 2.45e6

"""
    CityBuildings

State of the city-mean building of the example: the `BuildingZone`, its
`IdealLoadsHVAC`, `ThermostatSchedule` and `InternalGains`, and the maps
`H`, `LE` (W m⁻² of plan area) of the heat it put directly into the air
over the last building step, and `S` (≤ 0), the window-transmitted sun
taken from the wall cells' surface budget. See [`city_buildings`](@ref).
"""
mutable struct CityBuildings
    zone::BuildingZone{Float64}
    hvac::IdealLoadsHVAC
    thermostat::ThermostatSchedule
    gains::InternalGains
    floors::Matrix{Float64}          # floors above each column (0 off buildings)
    kair::Matrix{Int}                # first air layer above each column
    wall_cells::Vector{CartesianIndex{2}}
    H::Matrix{Float64}
    LE::Matrix{Float64}
    S::Matrix{Float64}
    release::Union{Nothing, BuildingHeatRelease{Float64}}
    hvac_budget::Union{Nothing, HVACBudget{Float64}}
end

"""
    city_buildings(heightmap, wall_cells, dx, dy, dz, wall_area; floor_height = 3.96,
                   glazing = 0.33, start_weekday = 4, T_air = 297.15) -> CityBuildings

One city-mean office building standing for every building of a window:
`heightmap` (m, 0 off buildings) on a grid of `dx × dy` columns and
layers `dz`, `wall_cells` the ground columns next to the buildings that
take the canyon part of the release, `wall_area` the total wall area (m²,
e.g. the sum over a radiosity scene's wall facets).

Assumptions, none of them a survey of Alexanderplatz: DOE medium-office
schedules, internal gains and thermostat (`office_internal_gains`,
`office_thermostat`); floor-to-floor height 3.96 m and window-to-wall
ratio 0.33, chosen to be close to the DOE medium-office reference
building (Deru et al. 2011) but not checked against it here; the illustrative
`UrbanSurface` example wall and roof; window U = 2.8 W m⁻² K⁻¹;
infiltration 0.3 h⁻¹; a packaged DX unit (COP 3.0) and a gas boiler.
`start_weekday` (1 = Monday): the weekday of day 1; the coupling passes
`case_weekday(cfg)` (29 June 2019, the case day, is a Saturday).
`T_outdoor` (K): the outdoor air at the start; given, the walls, the roof
and the ground slab start in their steady state between it (the slab:
its ground) and `T_air` ([`steady_envelope!`](@ref), #519), otherwise
uniform at `T_air`.
"""
function city_buildings(heightmap::AbstractMatrix, wall_cells::AbstractVector{CartesianIndex{2}},
                        dx::Real, dy::Real, dz::Real, wall_area::Real;
                        floor_height::Real = 3.96, glazing::Real = 0.33,
                        start_weekday::Integer = 4, T_air::Real = 297.15,
                        T_outdoor::Union{Nothing,Real} = nothing)
    cell = dx * dy
    built = heightmap .> 0
    A_roof = count(built) * cell
    V = sum(heightmap[built]) * cell
    floor_area = V / floor_height
    # outdoor air through 1/R_se, no absorbed sun (the slabs take it)
    walls = EnvelopeElement(example_wall_layers(), (1 - glazing) * wall_area;
                            absorptance = 0.0, T = T_air, name = :walls)
    roof = EnvelopeElement(example_roof_layers(), A_roof; h_inside = 1 / 0.10,
                           absorptance = 0.0, T = T_air, name = :roof)
    slab = EnvelopeElement([WallLayer(:xps_insulation, 0.05), WallLayer(:concrete, 0.20)], A_roof;
                           exterior = :ground, T_ground = 285.15, h_inside = 1 / 0.17,
                           T = T_air, name = :slab)
    # with the outdoor air known, the envelope starts in its steady state
    # (#519) instead of uniform at the zone air
    T_outdoor === nothing || foreach(el -> steady_envelope!(el, T_outdoor, T_air), (walls, roof, slab))
    n_floors = V / A_roof / floor_height
    floors = InternalMass([WallLayer(:concrete, 0.20)], max(n_floors - 1, 0.01) * A_roof;
                          sides = 2, T = T_air, name = :floors)
    zone = BuildingZone(; volume = V, floor_area = floor_area,
                        envelope = AbstractZoneElement[walls, roof, slab],
                        internal_mass = [floors], window_area = glazing * wall_area,
                        window_U = 2.8, infiltration_ach = 0.3, T_air = T_air)
    hvac = IdealLoadsHVAC(cop_rated = 3.0, curves = DX_CURVES_PACKAGED_AC, heating = :boiler)
    fl = [built[I] ? heightmap[I] / floor_height : 0.0 for I in CartesianIndices(heightmap)]
    kair = [clamp(round(Int, heightmap[I] / dz) + 1, 1, typemax(Int)) for I in CartesianIndices(heightmap)]
    wc = collect(wall_cells)
    Nx, Ny = size(heightmap)
    return CityBuildings(zone, hvac, office_thermostat(; start_weekday),
                         office_internal_gains(floor_area; start_weekday), fl, kair, wc,
                         zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny), nothing, nothing)
end

"""
    step_city_buildings!(cb, t, dt, T_out, q_out; sw_wall = 0, shgc = 0.4)

Advance the city-mean building by `dt` seconds starting `t` seconds after
00:00 of day 1, against canyon air at `T_out` (K) and `q_out` (kg/kg),
with `sw_wall` (W m⁻²) the mean irradiance on the walls (for the
window-transmitted sun), and refresh `cb.H`, `cb.LE`, `cb.S`. `shgc` of
the glazing is an assumption.
"""
function step_city_buildings!(cb::CityBuildings, t::Real, dt::Real, T_out::Real, q_out::Real;
                              sw_wall::Real = 0.0, shgc::Real = 0.4)
    z = cb.zone
    # start from the outdoor humidity, so the first hours are not a
    # vapour exchange with an arbitrary initial indoor value
    cb.release === nothing && (z.q_air = q_out)
    r = internal_gains(cb.gains, t, dt; T_air = zone_air_temperature(z))
    f = ZoneForcing(r; T_out = T_out, q_out = q_out, Q_solar = shgc * z.window_area * sw_wall)
    hv = scheduled_hvac(cb.hvac, cb.thermostat, t)
    q0 = zone_specific_humidity(z)
    b, hb = step_zone!(z, f, dt, hv)
    rel = building_heat_release(z, f, b; dt = dt, gains = r, hvac = hv, hvac_budget = hb,
                                q_before = q0)
    cb.release = rel; cb.hvac_budget = hb
    # what the walls and roof give to the air (`Q_envelope_air` split)
    walls, roof = z.envelope[1], z.envelope[2]
    Q_wall = -walls.area * walls.F_outer
    Q_roof = -roof.area * roof.F_outer
    fill!(cb.H, 0.0); fill!(cb.LE, 0.0); fill!(cb.S, 0.0)
    cell_floors = sum(cb.floors)
    nroof = count(>(0), cb.floors)
    Nx, Ny = size(cb.H)
    cell_area = roof.area / nroof
    for I in CartesianIndices(cb.floors)
        cb.floors[I] > 0 || continue
        w = cb.floors[I] / cell_floors
        cb.H[I] += (w * rel.Q_exhaust_sensible + Q_roof / nroof) / cell_area
        cb.LE[I] += w * rel.Q_exhaust_latent / cell_area
    end
    canyon_H = rel.Q_ventilation_sensible + rel.Q_windows + rel.Q_lost + Q_wall
    canyon_LE = rel.Q_ventilation_latent + rel.Q_latent_vented
    nw = length(cb.wall_cells)
    for I in cb.wall_cells
        cb.H[I] += canyon_H / nw / cell_area
        cb.LE[I] += canyon_LE / nw / cell_area
        cb.S[I] -= rel.Q_solar / nw / cell_area
    end
    return cb
end

"""
    apply_building_heat!(sim, qv_field, sensible, cb, dt; ρ = 1.0, cp = 1005.0, π = 1.0)

Put the city-mean building's direct release into the first air layer of
each column (above the roof for building columns): θ gets `H/(ρ_k cp Π_k)`
(`π` the Exner function, scalar or per level; T = θΠ),
the vapour field `LE/(ρ_k L_v)`, per `dz`, over `dt`. `ρ` is a scalar or
a per-level vector (the case passes the reference density `ρ̄`). `S` goes to the slab
surfaces: `sensible` is their loss term, so it is reduced by `S`.
"""
function apply_building_heat!(sim, qv_field, sensible::AbstractMatrix, cb::CityBuildings, dt::Real;
                              ρ = 1.0, cp::Real = 1005.0, π = 1.0)
    sensible .-= cb.S
    Hx, Hy, Hz = sim.mesh.Hx, sim.mesh.Hy, sim.mesh.Hz
    dz = sim.mesh.dz; Nz = sim.mesh.Nz
    qv = interior_view(qv_field)
    @inbounds for I in CartesianIndices(cb.H)
        i, j = Tuple(I)
        k = min(cb.kair[I], Nz)
        ρk = ρ isa AbstractVector ? ρ[k] : ρ
        πk = π isa AbstractVector ? π[k] : π
        sim.θ.data[Hx+i, Hy+j, Hz+k] += cb.H[I] / (ρk * cp * πk) * dt / dz
        qv[i, j, k] += cb.LE[I] / (ρk * CITY_BUILDING_L_V) * dt / dz
    end
    return sim
end

# =========================================================================
# 3e. Buildings in the urban canopy: zones behind the facets of a scene
# =========================================================================

"""
    step_buildings!(u::UrbanFacetScene, zones, forcings, air, dt;
                    hvac = nothing, L_sky, sw = nothing, sun = nothing,
                    dni = 0, dhi = 0, samples = 2, δT = 0.01, kwargs...)
        -> (fluxes, budgets, hvac_budgets)

Advance the facets of an urban scene and the building zones behind them
together, as one implicit system (issue #29).

A zone is *behind* scene facets when its `envelope` holds
[`FacetElement`](@ref)s wrapping the scene's own `UrbanFacet`s — the same
objects, `FacetElement(u.facets[i], area)` with `area` the facet's area
(see [`scene_building_zone`](@ref)). Each such facet then has the zone air
as its interior environment and the rest of the canyon outside, and the
zone sees the facet as part of its envelope. Other envelope elements and
internal mass are stepped by the zone as in [`step_zone!`](@ref).

`zones` and `forcings` are vectors of [`BuildingZone`](@ref) and
[`ZoneForcing`](@ref) (the `surface` of a `ZoneForcing` only matters for
facet elements that are *not* in the scene). `air`, `L_sky` and the
shortwave keywords are those of `UrbanSurface.step_urban_facets!`.
`hvac` is `nothing` or a vector with an [`IdealLoadsHVAC`](@ref) (or
`nothing`) per zone.

# The coupled step

`step_urban_facets!` treats the longwave exchange between facets
implicitly by measuring each facet column's affine response to an extra
imposed surface flux. The same column is also affine in the temperature
of its interior environment, and the zone air balance is affine in the
air temperature, so the facets' skin increments and the zones' air
temperature increments are solved **together** in one dense linear
system. The exchange across the envelope is as implicit as the exchange
between facets and between a zone and its internal mass: a lagged
coupling of the air node (air time constant: minutes) would be unstable
at canopy time steps.

An ideal-loads HVAC is solved by active set: a zone whose free-floating
temperature leaves its dead band is pinned on the violated setpoint and
the load that holds it becomes the unknown; a load beyond capacity is
clamped and the zone freed again. For one zone this is exactly the logic
of `step_zone!(zone, forcing, dt, hvac)`, and the two agree.

Returns the per-facet `SurfaceFluxes` (as `step_urban_facets!`), a
[`ZoneBudget`](@ref) per zone, and an [`HVACBudget`](@ref) (or `nothing`)
per zone. Energy is conserved exactly: every zone satisfies its budget
identity with its facet columns included, and heat crosses the envelope
without loss.
"""
function step_buildings!(u::UrbanFacetScene{T}, zones::AbstractVector{<:BuildingZone},
                         forcings::AbstractVector{<:ZoneForcing}, air, dt::Real;
                         hvac = nothing, L_sky::Real = UrbanSurface._sky_irradiance(air),
                         sw = nothing, sun = nothing, dni::Real = 0, dhi::Real = 0,
                         samples::Int = 2, δT::Real = 0.01, kwargs...) where {T}
    m = length(zones)
    length(forcings) == m ||
        throw(ArgumentError("step_buildings!: $(length(forcings)) forcings for $m zones"))
    hv = hvac === nothing ? fill(nothing, m) : hvac
    length(hv) == m || throw(ArgumentError("step_buildings!: `hvac` needs one entry per zone"))
    node, in_scene = _bind_zones(u, zones)
    # operating points and zone assembly at the start-of-step state; the scene
    # facets get their share of the radiative gains and the zone air as their
    # interior environment
    ops = [h === nothing ? nothing : _hvac_operating_point(zones[k], forcings[k], h)
           for (k, h) in enumerate(hv)]
    syss = map(1:m) do k
        z = zones[k]
        sys = _assemble_zone!(z, forcings[k], dt, in_scene[k]; δT, kwargs...)
        for (j, el) in enumerate(z.envelope)
            in_scene[k][j] || continue
            set_interior_boundary!(el.facet, ColumnBoundary(z.T_air, el.facet.interior.h, sys.q))
        end
        sys
    end
    step = UrbanSurface._prepare_scene_step(u, air, dt; node, L_sky, sw, sun, dni, dhi,
                                             samples, δT, kwargs...)
    lhs = T[s.lhs for s in syss]
    R = T[s.R for s in syss]
    x0 = T[s.x0 for s in syss]
    # active set over the thermostats
    mode = fill(:free, m)
    target = zeros(T, m); Φ = zeros(T, m); Φ_req = zeros(T, m); pin_heat = falses(m)
    local x, y, Φs
    for iter in 1:(3m + 3)
        pinned = mode .=== :pinned
        x, y, Φs = UrbanSurface._solve_scene_step(step; lhs, R, pinned, y = target, Φ)
        changed = false
        for k in 1:m
            h = hv[k]
            h === nothing && continue
            if mode[k] === :free
                xk = x0[k] + y[k]
                Tset = xk < h.T_heat ? h.T_heat : xk > h.T_cool ? h.T_cool : nothing
                Tset === nothing && continue
                mode[k] = :pinned; target[k] = Tset - x0[k]; pin_heat[k] = xk < h.T_heat
                changed = true
            elseif mode[k] === :pinned
                Φ_req[k] = Φs[k]
                # a neighbour's change can leave a pinned zone needing the
                # opposite action: it then floats in its dead band again
                if (pin_heat[k] && Φs[k] < 0) || (!pin_heat[k] && Φs[k] > 0)
                    mode[k] = :free; target[k] = 0; changed = true
                    continue
                end
                Φc = clamp(Φs[k], -ops[k].cap_c, h.heating_capacity)
                if Φc != Φs[k]
                    mode[k] = :clamped; Φ[k] = Φc; changed = true
                end
            end
        end
        changed || break
        iter == 3m + 3 && error("step_buildings!: the thermostat active set did not settle")
    end
    fluxes = UrbanSurface._apply_scene_step!(u, step, x, y)
    budgets = Vector{ZoneBudget{T}}(undef, m)
    hbs = Vector{Union{Nothing, HVACBudget{T}}}(nothing, m)
    for k in 1:m
        z, f, h = zones[k], forcings[k], hv[k]
        Φk = mode[k] === :free ? zero(T) : Φs[k]
        W = zero(T); L_v = zero(T)
        if h !== nothing
            W, L_v = _hvac_dehumidify(z, f, dt, h, x0[k] + y[k], Φk, ops[k].cap_c)
        end
        sf = [in_scene[k][j] ? fluxes[_scene_index(u, el)] : nothing
              for (j, el) in enumerate(z.envelope)]
        budgets[k] = _complete_zone_step!(z, f, dt, syss[k], Φk, W; s = y[k],
                                          in_scene = in_scene[k], scene_fluxes = sf)
        if h !== nothing
            req = mode[k] === :free ? zero(T) : Φ_req[k]
            hbs[k] = _hvac_plant(z, h, ops[k], Φk, req, W, L_v)
        end
    end
    return fluxes, budgets, hbs
end

_scene_index(u::UrbanFacetScene, el::FacetElement) = findfirst(f -> f === el.facet, u.facets)

# Which scene facet belongs to which zone. A facet may bound at most one
# zone, and its element area must be the facet's area (the scene and the
# zone must count the same column).
function _bind_zones(u::UrbanFacetScene{T}, zones) where {T}
    node = zeros(Int, length(u.facets))
    idx = IdDict{Any, Int}(f => i for (i, f) in enumerate(u.facets))
    in_scene = [falses(length(z.envelope)) for z in zones]
    for (k, z) in enumerate(zones)
        for (j, el) in enumerate(z.envelope)
            el isa FacetElement || continue
            i = get(idx, el.facet, 0)
            i == 0 && continue
            node[i] == 0 ||
                throw(ArgumentError("step_buildings!: scene facet $i bounds zones $(node[i]) and $k"))
            isapprox(el.area, u.vf.area[i]; rtol = 1e-12) ||
                throw(ArgumentError("step_buildings!: the element of scene facet $i has area " *
                                    "$(el.area) m², the facet $(u.vf.area[i]) m²"))
            node[i] = k
            in_scene[k][j] = true
        end
    end
    return node, in_scene
end

# ---- building zones from the scene geometry ------------------------------------

"""
    building_facets(u::UrbanFacetScene, footprint) -> Vector{Int}

Indices of the roof and wall facets of the building standing on the grid
columns where `footprint` (an `Nx × Ny` Bool matrix on the scene's
heightmap) is `true`. A roof belongs to its column; a wall belongs to the
column it is the side of (behind its outward normal), not to the open
column it faces and is attributed to for the plan-area aggregation.
"""
function building_facets(u::UrbanFacetScene, footprint::AbstractMatrix{Bool})
    sc = u.scene
    size(footprint) == size(sc.heightmap) ||
        throw(ArgumentError("building_facets: footprint is $(size(footprint)), the scene grid $(size(sc.heightmap))"))
    idx = Int[]
    for (i, rf) in enumerate(sc.facets)
        rf.kind === :ground && continue
        ci, cj = rf.kind === :roof ? rf.col : _owner_column(sc, rf)
        (1 <= ci <= size(footprint, 1) && 1 <= cj <= size(footprint, 2)) || continue
        footprint[ci, cj] && push!(idx, i)
    end
    return idx
end

# The column a wall facet is the side of: a quarter cell behind its centroid.
function _owner_column(sc, rf)
    c, n = rf.centroid, rf.normal
    x = c[1] - n[1] * sc.dx / 4
    y = c[2] - n[2] * sc.dy / 4
    return (floor(Int, (x - sc.x0) / sc.dx) + 1, floor(Int, (y - sc.y0) / sc.dy) + 1)
end

"""
    scene_building_zone(u::UrbanFacetScene, footprint; floor_height = 3.0,
                        floor_layers = [WallLayer(:concrete, 0.20)],
                        slab_layers = [WallLayer(:xps_insulation, 0.05), WallLayer(:concrete, 0.20)],
                        T_ground = 285.15, T_air = <mean interior temperature of its facets>,
                        kwargs...) -> BuildingZone

One well-mixed [`BuildingZone`](@ref) for the building on `footprint`
(see [`building_facets`](@ref)), whose envelope **is** the scene's roof
and wall facets of that building — `FacetElement(u.facets[i], area_i)`,
the same `UrbanFacet` objects — so that [`step_buildings!`](@ref) steps
them together with the zone.

  * volume ``V = Σ h_{ij} Δx Δy`` over the footprint (flat ground at the
    scene's ground level), floor area ``V / `` `floor_height`;
  * a ground-floor slab of the footprint area on `slab_layers`, an
    `EnvelopeElement` with `exterior = :ground` at `T_ground`;
  * the intermediate floors, ``n_{floors} - 1`` times the footprint, as
    two-sided [`InternalMass`](@ref) on `floor_layers`.

`kwargs` go to `BuildingZone` (glazing, air change rates,
`radiative_fraction`, …). The default constructions are illustrative,
not a survey of any building stock. One zone per building is the
TEB-BEM assumption (Bueno et al. 2012): a single indoor air temperature
for the whole building.
"""
function scene_building_zone(u::UrbanFacetScene{T}, footprint::AbstractMatrix{Bool};
                             floor_height::Real = 3.0,
                             floor_layers = [WallLayer(:concrete, 0.20)],
                             slab_layers = [WallLayer(:xps_insulation, 0.05), WallLayer(:concrete, 0.20)],
                             T_ground::Real = 285.15, T_air = nothing, kwargs...) where {T}
    floor_height > 0 || throw(ArgumentError("scene_building_zone: floor_height must be positive"))
    sc = u.scene
    ids = building_facets(u, footprint)
    isempty(ids) && throw(ArgumentError("scene_building_zone: no building on the footprint"))
    cell = sc.dx * sc.dy
    ground = minimum(sc.heightmap)
    A_fp = count(footprint) * cell
    V = sum(max(sc.heightmap[I] - ground, zero(T)) * cell for I in findall(footprint))
    V > 0 || throw(ArgumentError("scene_building_zone: the footprint has no height"))
    n_floors = max(1, round(Int, V / A_fp / floor_height))
    Ti = T_air === nothing ? sum(u.facets[i].interior.T_env for i in ids) / length(ids) : T(T_air)
    env = AbstractZoneElement[FacetElement(u.facets[i], u.vf.area[i]; name = u.facets[i].kind) for i in ids]
    push!(env, EnvelopeElement(slab_layers, A_fp; exterior = :ground, T_ground = T_ground,
                               h_inside = 1 / ISO6946_SURFACE_RESISTANCE.inside_down,
                               T = Ti, name = :ground_slab))
    im = n_floors > 1 ? [InternalMass(floor_layers, (n_floors - 1) * A_fp; sides = 2,
                                      h_inside = 1 / ISO6946_SURFACE_RESISTANCE.inside_horizontal,
                                      T = Ti, name = :floors)] : InternalMass{T}[]
    return BuildingZone(; volume = V, floor_area = V / floor_height, envelope = env,
                        internal_mass = im, T_air = Ti, kwargs...)
end

"""
    anthropogenic_heat_maps(u::UrbanFacetScene, zones, releases) -> (H, LE)

Plan-area maps (W m⁻², on the scene's `Nx × Ny` grid, like
`UrbanSurface.aggregate_facet_fluxes`) of the building anthropogenic
heat that enters the air **directly**, from the
[`BuildingHeatRelease`](@ref)s of zones bound to the scene:

  * HVAC exhaust (sensible into `H`, latent into `LE`) at the building's
    roof facets — rooftop condensers and cooling towers — weighted by
    roof area;
  * air exchange, glazing, lost gains, `:air` envelope elements and
    vented latent gains at its wall facets, i.e. into the canyon they
    face, weighted by wall area (at the roofs if it has no walls).

Each facet's share goes to the column `Facet.col` it is attributed to, so
``Σ (H + LE) Δx Δy`` equals the sum of the direct terms exactly.
`Q_envelope_facets` is not mapped: it is already in the facets' own
fluxes. Add the maps to `aggregate_facet_fluxes` to get the total the
atmosphere receives.
"""
function anthropogenic_heat_maps(u::UrbanFacetScene{T}, zones::AbstractVector{<:BuildingZone},
                                 releases::AbstractVector{<:BuildingHeatRelease}) where {T}
    length(zones) == length(releases) ||
        throw(ArgumentError("anthropogenic_heat_maps: one release per zone"))
    sc = u.scene
    nx, ny = size(sc.heightmap)
    cell = sc.dx * sc.dy
    H = zeros(T, nx, ny); LE = zeros(T, nx, ny)
    node, _ = _bind_zones(u, zones)
    for (k, r) in enumerate(releases)
        roofs = [i for i in eachindex(node) if node[i] == k && sc.facets[i].kind === :roof]
        walls = [i for i in eachindex(node) if node[i] == k && sc.facets[i].kind === :wall]
        isempty(roofs) && isempty(walls) &&
            throw(ArgumentError("anthropogenic_heat_maps: zone $k has no scene facets"))
        roofs_or = isempty(roofs) ? walls : roofs
        walls_or = isempty(walls) ? roofs : walls
        _spread!(H, LE, sc, u.vf.area, roofs_or, r.Q_exhaust_sensible, r.Q_exhaust_latent, cell)
        _spread!(H, LE, sc, u.vf.area, walls_or,
                 r.Q_ventilation_sensible + r.Q_windows + r.Q_lost + r.Q_envelope_air,
                 r.Q_ventilation_latent + r.Q_latent_vented, cell)
    end
    return (H = H, LE = LE)
end

function _spread!(H, LE, sc, area, ids, Qh, Qle, cell)
    A = sum(area[i] for i in ids)
    for i in ids
        ci, cj = sc.facets[i].col
        w = area[i] / A / cell
        H[ci, cj] += w * Qh
        LE[ci, cj] += w * Qle
    end
    return nothing
end

# =========================================================================
# 4. ISO 13790 simple hourly method (5R1C) — the reference model
# =========================================================================

"""
    iso13786_areal_heat_capacity(layers; side = :inner, d_max = 0.10,
                                 k_insulation = 0.1) -> κ (J m⁻² K⁻¹)

Internal areal heat capacity of a construction by the simplified
effective-thickness estimate of ISO 13786 Annex A, as used by ISO 13790
§12.3 for ``C_m``: ``κ = Σ ρ_i c_i d_i`` over the layers from the
exposed surface (`side = :inner` → the last layer, `:outer` → the first)
up to the smallest of

  * half the total thickness of the element,
  * the first thermally insulating layer (``k ≤`` `k_insulation`; the
    threshold 0.1 W m⁻¹ K⁻¹ is this module's choice),
  * `d_max` = 0.10 m.

A symmetric partition exposed on both faces has ``2κ`` per unit face
area.
"""
function iso13786_areal_heat_capacity(layers::AbstractVector{<:WallLayer}; side::Symbol = :inner,
                                      d_max::Real = 0.10, k_insulation::Real = 0.1)
    side in (:inner, :outer) || throw(ArgumentError("side must be :inner or :outer"))
    ls = side === :inner ? reverse(layers) : layers
    d_lim = min(sum(l -> l.thickness, layers) / 2, d_max)
    κ = 0.0; d = 0.0
    for l in ls
        l.material.conductivity <= k_insulation && break
        dd = min(l.thickness, d_lim - d)
        dd <= 0 && break
        κ += volumetric_heat_capacity(l.material) * dd
        d += dd
    end
    return κ
end

"""
    ISO13790Zone(; A_f, C_m, A_m, H_ve, H_tr_w, H_tr_op, Λ_at = 4.5,
                 h_is = 3.45, h_ms = 9.1, θ_m = 293.15)

The ISO 13790:2008 simple hourly model (§7.2.2, Annex C): five
conductances and one capacitance between the outdoor, supply-air, air
(``θ_{air}``), central surface (``θ_s``) and mass (``θ_m``) nodes.

  * ``H_{tr,is} = h_{is} A_t``, ``A_t = Λ_{at} A_f``;
  * ``H_{tr,ms} = h_{ms} A_m``;
  * ``H_{tr,em} = 1/(1/H_{tr,op} - 1/H_{tr,ms})`` — the opaque envelope
    conductance ``H_{tr,op}`` minus the mass-to-surface part;
  * ``H_{tr,w}`` the glazing, ``H_{ve}`` the air exchange;
  * ``C_m`` (J K⁻¹) and the effective mass area ``A_m`` (m²).

Temperatures in K (the scheme is affine, so K and °C are equivalent).
The state is the mass temperature `θ_m`; the air and surface
temperatures of the last step are kept in `θ_air`, `θ_s`.
"""
mutable struct ISO13790Zone{T<:AbstractFloat}
    A_f::T
    A_t::T
    A_m::T
    C_m::T
    H_ve::T
    H_tr_w::T
    H_tr_is::T
    H_tr_ms::T
    H_tr_em::T
    θ_m::T
    θ_air::T
    θ_s::T
end

function ISO13790Zone(; A_f::Real, C_m::Real, A_m::Real, H_ve::Real, H_tr_w::Real,
                      H_tr_op::Real, Λ_at::Real = 4.5, h_is::Real = 3.45,
                      h_ms::Real = 9.1, θ_m::Real = 293.15)
    (A_f > 0 && C_m > 0 && A_m > 0 && H_ve > 0 && H_tr_w >= 0 && H_tr_op > 0) ||
        throw(ArgumentError("ISO13790Zone: A_f, C_m, A_m, H_ve, H_tr_op must be positive, H_tr_w ≥ 0"))
    H_ms = h_ms * A_m
    H_tr_op < H_ms ||
        throw(ArgumentError("ISO13790Zone: H_tr_op must be smaller than H_tr_ms = h_ms A_m"))
    H_em = 1 / (1 / H_tr_op - 1 / H_ms)
    A_t = Λ_at * A_f
    return ISO13790Zone{Float64}(A_f, A_t, A_m, C_m, H_ve, H_tr_w, h_is * A_t, H_ms, H_em,
                                 θ_m, θ_m, θ_m)
end

"""
    step_iso13790!(z::ISO13790Zone, θ_e, dt = 3600; Φ_int = 0, Φ_sol = 0,
                   Φ_HC = 0, θ_sup = θ_e) -> (θ_air, θ_s, θ_m)

One step of the ISO 13790 Annex C scheme (eqs. C.1-C.11), with the
standard gain split

  * ``Φ_{ia} = 0.5 Φ_{int}`` to the air node,
  * ``Φ_m = (A_m/A_t)(0.5 Φ_{int} + Φ_{sol})`` to the mass node,
  * ``Φ_{st} = (1 - A_m/A_t - H_{tr,w}/(9.1 A_t))(0.5 Φ_{int} + Φ_{sol})``
    to the surface node,

and the Crank–Nicolson mass update

```math
θ_{m,t} = \\frac{θ_{m,t-1} (C_m/Δt - (H_{tr,3} + H_{tr,em})/2) + Φ_{m,tot}}
               {C_m/Δt + (H_{tr,3} + H_{tr,em})/2}
```

(the standard fixes ``Δt`` = 3600 s). `Φ_HC` is a heating/cooling input
to the air node. Returns the step's air and surface temperatures and
the step-mean mass temperature.
"""
function step_iso13790!(z::ISO13790Zone, θ_e::Real, dt::Real = 3600.0; Φ_int::Real = 0.0,
                        Φ_sol::Real = 0.0, Φ_HC::Real = 0.0, θ_sup::Real = θ_e)
    Φ_ia = 0.5Φ_int
    Φ_m = z.A_m / z.A_t * (0.5Φ_int + Φ_sol)
    Φ_st = (1 - z.A_m / z.A_t - z.H_tr_w / (9.1 * z.A_t)) * (0.5Φ_int + Φ_sol)
    H1 = 1 / (1 / z.H_ve + 1 / z.H_tr_is)
    H2 = H1 + z.H_tr_w
    H3 = 1 / (1 / H2 + 1 / z.H_tr_ms)
    Φ_mtot = Φ_m + z.H_tr_em * θ_e +
             H3 * (Φ_st + z.H_tr_w * θ_e + H1 * ((Φ_ia + Φ_HC) / z.H_ve + θ_sup)) / H2
    c = z.C_m / dt
    θ_m_old = z.θ_m
    θ_m_new = (θ_m_old * (c - 0.5(H3 + z.H_tr_em)) + Φ_mtot) / (c + 0.5(H3 + z.H_tr_em))
    θ_m = (θ_m_new + θ_m_old) / 2
    θ_s = (z.H_tr_ms * θ_m + Φ_st + z.H_tr_w * θ_e + H1 * (θ_sup + (Φ_ia + Φ_HC) / z.H_ve)) /
          (z.H_tr_ms + z.H_tr_w + H1)
    θ_air = (z.H_tr_is * θ_s + z.H_ve * θ_sup + Φ_ia + Φ_HC) / (z.H_tr_is + z.H_ve)
    z.θ_m, z.θ_s, z.θ_air = θ_m_new, θ_s, θ_air
    return (θ_air, θ_s, θ_m)
end

"""
    iso13790_ideal_load!(z::ISO13790Zone, θ_e, dt = 3600; θ_heat, θ_cool,
                         Φ_int = 0, Φ_sol = 0, heating_capacity = Inf,
                         cooling_capacity = Inf, θ_sup = θ_e) -> (Φ_HC, θ_air, θ_s, θ_m)

The ISO 13790:2008 §C.4.2 procedure for the heating/cooling need of the
5R1C model: step with ``Φ_{HC} = 0``; if ``θ_{air,0}`` lies between the
setpoints that is the step. Otherwise step with ``Φ_{HC,10} = 10 A_f``
and take the unrestricted need
``Φ_{HC,nd,un} = Φ_{HC,10} (θ_{set} - θ_{air,0})/(θ_{air,10} - θ_{air,0})``,
limited to the capacities (cooling as a negative `Φ_HC`), and step with
it. Only the final step changes the state. Returns the load and the
step's temperatures as [`step_iso13790!`](@ref).
"""
function iso13790_ideal_load!(z::ISO13790Zone, θ_e::Real, dt::Real = 3600.0; θ_heat::Real,
                              θ_cool::Real, Φ_int::Real = 0.0, Φ_sol::Real = 0.0,
                              heating_capacity::Real = Inf, cooling_capacity::Real = Inf,
                              θ_sup::Real = θ_e)
    θ_heat <= θ_cool || throw(ArgumentError("iso13790_ideal_load!: θ_heat must not exceed θ_cool"))
    state = (z.θ_m, z.θ_air, z.θ_s)
    trial(Φ) = (r = step_iso13790!(z, θ_e, dt; Φ_int, Φ_sol, Φ_HC = Φ, θ_sup);
                (z.θ_m, z.θ_air, z.θ_s) = state; r[1])
    θ0 = trial(0.0)
    θ_set = θ0 < θ_heat ? θ_heat : θ0 > θ_cool ? θ_cool : θ0
    Φ = 0.0
    if θ_set != θ0
        Φ10 = 10 * z.A_f
        θ10 = trial(Φ10)
        Φ = clamp(Φ10 * (θ_set - θ0) / (θ10 - θ0), -cooling_capacity, heating_capacity)
    end
    θ_air, θ_s, θ_m = step_iso13790!(z, θ_e, dt; Φ_int, Φ_sol, Φ_HC = Φ, θ_sup)
    return (Φ, θ_air, θ_s, θ_m)
end

"""
    iso13790_zone(zone::BuildingZone; Λ_at = 4.5, kwargs...) -> ISO13790Zone

Derive the ISO 13790 5R1C parameters from the same construction as a
[`BuildingZone`](@ref):

  * ``H_{ve}`` = [`ventilation_conductance`](@ref), ``H_{tr,w} = U_w A_w``;
  * ``H_{tr,op} = Σ U_j A_j`` over `:air` envelope and facet elements
    ([`element_transmittance`](@ref)); `:adiabatic` elements add only
    heat capacity; `:ground` elements are rejected (ISO 13790 treats
    them through ISO 13370, which is not implemented);
  * ``C_m = Σ κ_j A_j`` and ``A_m = C_m^2 / Σ A_j κ_j^2`` (ISO 13790
    §12.3.1, detailed method) with ``κ_j`` from
    [`iso13786_areal_heat_capacity`](@ref), counting both faces of a
    two-sided internal mass.

The mass node starts at the zone air temperature. `kwargs` go to
[`ISO13790Zone`](@ref).
"""
function iso13790_zone(zone::BuildingZone; Λ_at::Real = 4.5, kwargs...)
    H_op = 0.0; Cm = 0.0; S2 = 0.0
    add!(A, κ) = (Cm += A * κ; S2 += A * κ^2)
    for el in zone.envelope
        if el isa EnvelopeElement
            el.exterior === :ground &&
                throw(ArgumentError("iso13790_zone: :ground elements are not supported (ISO 13370)"))
            layers = el.column.layers
        else
            layers = el.facet.column.layers
        end
        H_op += element_transmittance(el) * el.area
        add!(el.area, iso13786_areal_heat_capacity(layers; side = :inner))
    end
    for el in zone.internal_mass
        add!(el.area, iso13786_areal_heat_capacity(el.column.layers; side = :inner))
        el.sides == 2 && add!(el.area, iso13786_areal_heat_capacity(el.column.layers; side = :outer))
    end
    return ISO13790Zone(; A_f = zone.floor_area, C_m = Cm, A_m = Cm^2 / S2,
                        H_ve = ventilation_conductance(zone),
                        H_tr_w = zone.window_U * zone.window_area, H_tr_op = H_op,
                        Λ_at = Λ_at, θ_m = zone.T_air, kwargs...)
end

end # module
