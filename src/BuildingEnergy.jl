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
    is closed by the Era V surface energy balance.
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
  * Bueno, B., Norford, L., Pigeon, G. & Britter, R. (2012). Combining a
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
                             PhasePartition, air_density, heat_capacity_moist
using ..UrbanSurface: UrbanSurface, ConductionColumn, ColumnBoundary, WallLayer,
                      UrbanFacet, step_conduction!, face_temperature,
                      column_heat_content, thermal_resistance,
                      fixed_temperature_boundary, heat_flux_boundary,
                      set_interior_boundary!, ISO6946_SURFACE_RESISTANCE,
                      volumetric_heat_capacity

export AbstractZoneElement, EnvelopeElement, FacetElement, InternalMass,
       BuildingZone, ZoneForcing, ZoneBudget, step_zone!, facet_fluxes,
       zone_air_temperature, zone_heat_content, zone_heat_loss_coefficient,
       zone_mean_radiant_temperature, zone_operative_temperature,
       air_heat_capacity, ventilation_conductance, element_transmittance

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
                 p_ref = 101325, thermo = DEFAULT_CONSTANTS,
                 air_capacity = nothing)

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
end

function BuildingZone(; volume::Real, floor_area::Real,
                      envelope::AbstractVector = AbstractZoneElement[],
                      internal_mass::AbstractVector = InternalMass{Float64}[],
                      window_area::Real = 0.0, window_U::Real = 0.0,
                      infiltration_ach::Real = 0.5, ventilation_ach::Real = 0.0,
                      heat_recovery::Real = 0.0, radiative_fraction::Real = 0.5,
                      T_air::Real = 293.15, p_ref::Real = 101325.0,
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
    ρc = air_density(T_air, p_ref, q0, thermo) * heat_capacity_moist(q0, thermo)
    C = air_capacity === nothing ? ρc * volume : float(air_capacity)
    C > 0 || throw(ArgumentError("BuildingZone: air_capacity must be positive"))
    return BuildingZone{FT}(FT(volume), FT(floor_area), AbstractZoneElement[envelope...],
                            InternalMass{FT}[internal_mass...], FT(window_area), FT(window_U),
                            FT(infiltration_ach), FT(ventilation_ach), FT(heat_recovery),
                            FT(radiative_fraction), FT(T_air), FT(C), FT(ρc))
end

_elements(z::BuildingZone) = Iterators.flatten((z.envelope, z.internal_mass))

"Zone air temperature (K)."
zone_air_temperature(z::BuildingZone) = z.T_air

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
speed (ISO 7730 / ISO 13790 Annex C uses the same equal weighting).
"""
zone_operative_temperature(z::BuildingZone) = (z.T_air + zone_mean_radiant_temperature(z)) / 2

# =========================================================================
# 3. Forcing, budget and the step
# =========================================================================

"""
    ZoneForcing(; T_out, Q_internal = 0, Q_solar = 0, Q_hvac = 0,
                irradiance = 0, surface = nothing)

What drives a zone over one step:

  * `T_out` — outdoor air temperature (K), seen by `:air` envelope
    elements, glazing and the air exchange;
  * `Q_internal` — internal gains (W; people, lighting, appliances),
    split by the zone's `radiative_fraction`;
  * `Q_solar` — solar radiation transmitted through glazing (W),
    absorbed by the interior surfaces;
  * `Q_hvac` — sensible heat delivered to the air node by a heating or
    cooling system (W, negative for cooling); the HVAC model itself is
    issue #32;
  * `irradiance` — total irradiance on the outer face of `:air`
    [`EnvelopeElement`](@ref)s (W m⁻², absorbed with the element's
    `absorptance`): a number for all, or a vector indexed like
    `zone.envelope` (entries of other element types are ignored);
  * `surface` — the `SurfaceForcing` of the [`FacetElement`](@ref)s: one
    for all, or a vector indexed like `zone.envelope`. Required when the
    zone has facet elements.
"""
struct ZoneForcing{T<:AbstractFloat, I, S}
    T_out::T
    Q_internal::T
    Q_solar::T
    Q_hvac::T
    irradiance::I
    surface::S
end

function ZoneForcing(; T_out::Real, Q_internal::Real = 0.0, Q_solar::Real = 0.0,
                     Q_hvac::Real = 0.0, irradiance = 0.0, surface = nothing)
    Q_solar >= 0 || throw(ArgumentError("ZoneForcing: Q_solar must be ≥ 0"))
    T = float(promote_type(typeof(T_out), typeof(Q_internal), typeof(Q_solar), typeof(Q_hvac)))
    return ZoneForcing{T, typeof(irradiance), typeof(surface)}(T(T_out), T(Q_internal),
                                                               T(Q_solar), T(Q_hvac),
                                                               irradiance, surface)
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
    calls the envelope load).

Conservation: ``ΔE/Δt = Q_{exterior} + Q_{windows} + Q_{ventilation} +
Q_{internal} + Q_{solar} + Q_{hvac}`` with ``E`` the
[`zone_heat_content`](@ref).
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

`kwargs` are passed to the `step_surface!` of facet elements (`sf`, `c`,
`δT`, …). The step conserves [`zone_heat_content`](@ref) exactly.
"""
function step_zone!(zone::BuildingZone{T}, forcing::ZoneForcing, dt::Real; kwargs...) where {T}
    dt > 0 || throw(ArgumentError("step_zone!: dt must be positive"))
    x0 = zone.T_air
    T_out = T(forcing.T_out)
    Φ_rad = zone.radiative_fraction * T(forcing.Q_internal) + T(forcing.Q_solar)
    A_int = sum((_exposed_area(el) for el in _elements(zone)); init = zero(T))
    q = A_int > 0 ? Φ_rad / A_int : zero(T)
    Φ_air = T(forcing.Q_internal) + T(forcing.Q_solar) - (A_int > 0 ? Φ_rad : zero(T)) +
            T(forcing.Q_hvac)
    a = zero(T); b = zero(T)
    for (i, el) in enumerate(zone.envelope)
        el.q_inside = q
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
    # C/dt (x - x0) = a + b (x - x0) + (H_w + H_ve)(T_out - x) + Φ_air
    lhs = zone.C_air / T(dt) - b + H_w + H_ve
    s = (a + (H_w + H_ve) * (T_out - x0) + Φ_air) / lhs
    x = x0 + s
    Q_ext = zero(T); Q_env_air = zero(T); Q_surf = zero(T)
    for el in zone.envelope
        Q_ext += el isa FacetElement ? _finalize!(el, s, x) : _finalize!(el, s)
        Q_env_air += el.area * el.Q_air
    end
    Q_surf = Q_env_air
    for el in zone.internal_mass
        _finalize!(el, s)
        Q_surf += el.area * el.Q_air
    end
    zone.T_air = x
    return ZoneBudget{T}(x, Q_ext, H_w * (T_out - x), H_ve * (T_out - x),
                         T(forcing.Q_internal), T(forcing.Q_solar), T(forcing.Q_hvac),
                         Q_surf, Q_env_air)
end

end # module
