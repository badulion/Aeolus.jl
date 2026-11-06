"""
    UrbanSurface

Phase 49 — the production urban surface: per-facet (roof / wall /
ground) multi-layer heat conduction through real material layers,
replacing the single heat-capacity slab of the urban example.

Five parts:

  1. **Thermal material table** — [`ThermalMaterial`](@ref) and
     [`thermal_material`](@ref): conductivity, density and specific heat
     of the construction materials, each row cited. Only *thermal*
     properties live here. A material that can form an outer surface
     carries the name of its entry in the `SurfaceOptics` spectral
     library (`optical`), and a facet's albedo and emissivity are derived
     from that entry — the optical numbers are never restated here.
  2. **The conduction core** — [`ConductionColumn`](@ref),
     [`ColumnBoundary`](@ref) and [`step_conduction!`](@ref): a 1-D
     cell-centred finite-volume column through a stack of
     [`WallLayer`](@ref)s, advanced by the θ-scheme (Crank–Nicolson at
     θ = ½, backward Euler at θ = 1) with Robin boundary conditions at
     both faces. It knows nothing about the atmosphere: the interior face
     hook (indoor air temperature and surface coefficient, or a deep-soil
     temperature) is an ordinary [`ColumnBoundary`](@ref), which is what
     lets Phase 50's `BuildingEnergy` reuse the same code for building
     thermal mass.
  3. **Facets** — [`UrbanFacet`](@ref), an implementer of the
     `SurfaceExchange.AbstractSurface` interface: a conduction column
     whose outer face is closed by the surface energy balance
     ``R_n = H + LE + G``, linearised about the current skin temperature
     so the step stays implicit and conserves energy exactly.
  4. **Glazing** (`src/urban_surface/glazing.jl`) — angle-dependent
     shortwave optics of uncoated panes ([`GlassPane`](@ref),
     [`pane_optics`](@ref)) and multi-pane units ([`Glazing`](@ref),
     [`glazing_optics`](@ref)), ISO 15099 gap heat transfer, the steady
     [`glazing_u_value`](@ref) and [`glazing_shgc`](@ref), and
     [`WindowFacet`](@ref), a glazed `AbstractSurface` whose transmitted
     shortwave ([`transmitted_shortwave`](@ref)) is the hook for the
     building energy model.
  5. **Canyon exchange** (`src/urban_surface/canyon_exchange.jl`) — the
     Town Energy Balance closures for the air next to walls and streets:
     the canyon wind [`canyon_wind_speed`](@ref) and the wall convective
     coefficient [`rowley_convection_coefficient`](@ref).

# Sign convention

As in `SurfaceExchange`: `G` is positive *into* the substrate (from the
outer face into the column). Column boundary fluxes returned by
[`step_conduction!`](@ref) are positive *into the column* at both faces;
[`interior_heat_flux`](@ref) is positive from the facet *into the
interior* (the room, or the deep ground).

# References

  * Bergman, T. L., Lavine, A. S., Incropera, F. P. & DeWitt, D. P.
    (2011). *Fundamentals of Heat and Mass Transfer*, 7th ed., Wiley.
    Appendix A, Tables A.1 and A.3.
  * Heisler, G. M. & Brazel, A. J. (2010). The urban physical
    environment: temperature and urban heat islands. In *Urban Ecosystem
    Ecology*, Agronomy Monograph 55, ASA-CSSA-SSSA, 29-56. Table 2-3
    (after Goward, S. N., 1981, Thermal behavior of urban landscapes and
    the urban heat island, *Phys. Geogr.* 2, 19-33).
  * ISO 6946:2017. *Building components and building elements — Thermal
    resistance and thermal transmittance — Calculation methods.* Table 7
    (conventional surface resistances).
  * ISO 13786:2017. *Thermal performance of building components —
    Dynamic thermal characteristics — Calculation methods* (the
    periodic heat-transfer matrix used to validate the solver).
  * Carslaw, H. S. & Jaeger, J. C. (1959). *Conduction of Heat in
    Solids*, 2nd ed., Oxford, §2.6 (periodic surface temperature).
  * Masson, V. (2000). A physically-based scheme for the urban energy
    budget in atmospheric models. *Boundary-Layer Meteorol.*, 94,
    357-397 (the TEB multi-layer roof / wall / road conduction).
  * Resler, J. et al. (2017). PALM-USM v1.0: A new urban surface model
    integrated into the PALM large-eddy simulation model. *Geosci. Model
    Dev.*, 10, 3635-3659.
"""
module UrbanSurface

using ..SurfaceExchange
using ..SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS
import ..SurfaceExchange: surface_temperature, roughness_lengths, step_surface!,
                          column_heat_content
import ..RadiationCore: surface_albedo, surface_emissivity
using ..SurfaceOptics: SurfaceOptics, broadband_albedo, fresnel_polarized_reflectance
using ..RadiationCore: _gauss_legendre, STEFAN_BOLTZMANN
using ..Radiosity: Radiosity, RadiosityScene, ViewFactors, Facet, LongwaveResponse,
                   ShortwaveSolution, longwave_response, solve_longwave,
                   solve_shortwave, set_temperature!
import ..Radiosity: facet_temperatures, nfacets
using LinearAlgebra: I, lu!, mul!, dot, norm
using ..TimeIntegration: NumericalFailure   # a diverged solver (#327 M6)
using ..Backends: Backends, AbstractBackend, backend, foreach_column!

export ThermalMaterial, thermal_material, thermal_material_names,
       volumetric_heat_capacity, thermal_diffusivity, thermal_admittance,
       WallLayer, thermal_resistance, thermal_transmittance,
       ISO6946_SURFACE_RESISTANCE,
       ColumnBoundary, fixed_temperature_boundary, heat_flux_boundary,
       convective_boundary,
       ConductionColumn, step_conduction!, face_temperature,
       column_heat_content, cell_centers,
       UrbanFacet, interior_heat_flux, set_interior_boundary!,
       set_interior_temperature!, facet_layer_temperatures,
       example_roof_layers, example_wall_layers, example_ground_layers,
       # glazing (#27)
       CLEAR_FLOAT_REFRACTIVE_INDEX, CLEAR_FLOAT_EXTINCTION, UNCOATED_GLASS_EMISSIVITY,
       GlassPane, glass_pane_from_normal, pane_optics,
       FillGas, GLAZING_AIR, GLAZING_ARGON, GlazingGap, gap_nusselt, gap_conductance,
       Glazing, single_glazing, double_glazing, triple_glazing,
       glazing_optics, glazing_optics_hemispherical, glazing_areal_heat_capacity, glazing_solar_transmittance,
       glazing_u_value, inward_flowing_fractions, glazing_shgc,
       WindowFacet, set_solar_incidence!, window_optics, transmitted_shortwave,
       pane_temperatures, window_heat_content,
       # facet–radiosity coupling (#28)
       UrbanFacetScene, step_urban_facets!, aggregate_facet_fluxes,
       example_facet_layers,
       # canyon exchange (#30)
       canyon_wind_speed, rowley_convection_coefficient

# =========================================================================
# 1. Thermal material table
# =========================================================================

"""
    ThermalMaterial(name, k, ρ, c; optical = nothing, source = "")

Thermal properties of a homogeneous construction material:

  * `conductivity` `k` — W m⁻¹ K⁻¹
  * `density` `ρ` — kg m⁻³
  * `specific_heat` `c` — J kg⁻¹ K⁻¹
  * `optical` — the name of the matching entry in the `SurfaceOptics`
    spectral library (`SurfaceOptics.material_names()`), or `nothing`
    for a material that is never an exposed surface (insulation,
    plaster, soil). This is the *link* to the optical properties; the
    albedo and emissivity themselves are not stored here.
  * `source` — the citation of the row.

Derived quantities: [`volumetric_heat_capacity`](@ref) ``C = ρc``,
[`thermal_diffusivity`](@ref) ``κ = k/(ρc)`` and
[`thermal_admittance`](@ref) (thermal effusivity)
``μ = \\sqrt{kρc}``, which sets how strongly the surface temperature of
a thick layer responds to a periodic heat flux.
"""
struct ThermalMaterial{T<:AbstractFloat}
    name::Symbol
    conductivity::T
    density::T
    specific_heat::T
    optical::Union{Symbol, Nothing}
    source::String
end

function ThermalMaterial(name::Symbol, k::Real, ρ::Real, c::Real;
                         optical::Union{Symbol, Nothing} = nothing,
                         source::AbstractString = "")
    (k > 0 && ρ > 0 && c > 0) ||
        throw(ArgumentError("ThermalMaterial: k, ρ and c must be positive"))
    optical === nothing || optical in SurfaceOptics.material_names() ||
        throw(ArgumentError("ThermalMaterial: optical material :$optical is not in " *
                            "SurfaceOptics.material_names()"))
    T = float(promote_type(typeof(k), typeof(ρ), typeof(c)))
    return ThermalMaterial{T}(name, T(k), T(ρ), T(c), optical, String(source))
end

"Volumetric heat capacity ``C = ρ c`` (J m⁻³ K⁻¹)."
volumetric_heat_capacity(m::ThermalMaterial) = m.density * m.specific_heat
"Thermal diffusivity ``κ = k/(ρ c)`` (m² s⁻¹)."
thermal_diffusivity(m::ThermalMaterial) = m.conductivity / volumetric_heat_capacity(m)
"Thermal admittance (effusivity) ``μ = \\sqrt{k ρ c}`` (J m⁻² K⁻¹ s⁻¹ᐟ²)."
thermal_admittance(m::ThermalMaterial) = sqrt(m.conductivity * volumetric_heat_capacity(m))

function Base.show(io::IO, m::ThermalMaterial)
    print(io, "ThermalMaterial(:", m.name, ", k = ", m.conductivity,
          " W/m/K, ρ = ", m.density, " kg/m³, c = ", m.specific_heat, " J/kg/K",
          m.optical === nothing ? "" : ", optical :$(m.optical)", ")")
end

const _INCROPERA_A3 = "Bergman et al. (2011), Fundamentals of Heat and Mass Transfer, 7th ed., Table A.3, 300 K"
const _INCROPERA_A1 = "Bergman et al. (2011), Fundamentals of Heat and Mass Transfer, 7th ed., Table A.1, 300 K"
const _GOWARD = "Heisler & Brazel (2010), Agron. Monogr. 55, Table 2-3 (after Goward 1981)"

# The table. Every row is a published value; nothing here is tuned.
#
# Asphalt is the one deliberate choice between sources. Incropera's
# Table A.3 lists asphalt at k = 0.062 W/m/K, an order of magnitude below
# every urban-climate compilation (Goward 1981; Oke 1987 gives 0.75) and
# below dry sand; with it a road would behave like a layer of insulation.
# The Goward/Heisler & Brazel row (k = 0.745, published admittance
# 1204 J m⁻² K⁻¹ s⁻¹ᐟ²) is used instead.
const _THERMAL_TABLE = (
    # --- exposed surfaces (linked to the SurfaceOptics spectral library)
    ThermalMaterial(:asphalt,   0.7454, 2114.0, 920.0; optical = :asphalt,    source = _GOWARD),
    ThermalMaterial(:concrete,  1.4,    2300.0, 880.0; optical = :concrete,   source = _INCROPERA_A3 * " (concrete, stone mix)"),
    ThermalMaterial(:brick,     0.72,   1920.0, 835.0; optical = :brick,      source = _INCROPERA_A3 * " (brick, common)"),
    ThermalMaterial(:glass,     1.4,    2500.0, 750.0; optical = :glass,      source = _INCROPERA_A3 * " (glass, plate, soda lime)"),
    ThermalMaterial(:steel,     60.5,   7854.0, 434.0; optical = :metal_roof, source = _INCROPERA_A1 * " (plain carbon steel)"),
    ThermalMaterial(:granite,   2.79,   2630.0, 775.0;                        source = _INCROPERA_A3 * " (granite, Barre)"),
    # --- renders, plasters, boards
    ThermalMaterial(:cement_mortar,  0.72, 1860.0,  780.0; source = _INCROPERA_A3 * " (cement mortar)"),
    ThermalMaterial(:gypsum_plaster, 0.22, 1680.0, 1085.0; source = _INCROPERA_A3 * " (gypsum plaster, sand aggregate)"),
    ThermalMaterial(:plywood,        0.12,  545.0, 1215.0; source = _INCROPERA_A3 * " (plywood)"),
    ThermalMaterial(:softwood,       0.12,  510.0, 1380.0; source = _INCROPERA_A3 * " (softwoods, fir, pine)"),
    # --- insulation
    ThermalMaterial(:eps_insulation, 0.040,  16.0, 1210.0; source = _INCROPERA_A3 * " (polystyrene, expanded, molded beads)"),
    ThermalMaterial(:xps_insulation, 0.027,  55.0, 1210.0; source = _INCROPERA_A3 * " (polystyrene, extruded R-12)"),
    ThermalMaterial(:glass_fiber_board, 0.036, 105.0, 795.0; source = _INCROPERA_A3 * " (glass fiber, organic bonded)"),
    # --- substrate
    ThermalMaterial(:sand,      0.27,  1515.0,  800.0; source = _INCROPERA_A3 * " (sand)"),
    ThermalMaterial(:soil,      0.52,  2050.0, 1840.0; source = _INCROPERA_A3 * " (soil)"),
    ThermalMaterial(:clay,      1.3,   1460.0,  880.0; source = _INCROPERA_A3 * " (clay)"),
)

const URBAN_THERMAL_MATERIALS = Dict{Symbol, ThermalMaterial{Float64}}(m.name => m for m in _THERMAL_TABLE)

"""
    thermal_material_names() -> Vector{Symbol}

Names in the built-in thermal material table, in table order.
"""
thermal_material_names() = Symbol[m.name for m in _THERMAL_TABLE]

"""
    thermal_material(name::Symbol) -> ThermalMaterial

Look up a construction material in the built-in table:

| name | k (W m⁻¹ K⁻¹) | ρ (kg m⁻³) | c (J kg⁻¹ K⁻¹) | optical | source |
|:-----|------:|------:|------:|:--------|:-------|
| `:asphalt`           | 0.7454 | 2114 | 920  | `:asphalt`    | Goward 1981 via Heisler & Brazel 2010, T2-3 |
| `:concrete`          | 1.4    | 2300 | 880  | `:concrete`   | Incropera 7e, A.3 (stone mix) |
| `:brick`             | 0.72   | 1920 | 835  | `:brick`      | Incropera 7e, A.3 (common) |
| `:glass`             | 1.4    | 2500 | 750  | `:glass`      | Incropera 7e, A.3 (plate, soda lime) |
| `:steel`             | 60.5   | 7854 | 434  | `:metal_roof` | Incropera 7e, A.1 (plain carbon) |
| `:granite`           | 2.79   | 2630 | 775  | —             | Incropera 7e, A.3 (Barre) |
| `:cement_mortar`     | 0.72   | 1860 | 780  | —             | Incropera 7e, A.3 |
| `:gypsum_plaster`    | 0.22   | 1680 | 1085 | —             | Incropera 7e, A.3 (sand aggregate) |
| `:plywood`           | 0.12   | 545  | 1215 | —             | Incropera 7e, A.3 |
| `:softwood`          | 0.12   | 510  | 1380 | —             | Incropera 7e, A.3 (fir, pine) |
| `:eps_insulation`    | 0.040  | 16   | 1210 | —             | Incropera 7e, A.3 (EPS, molded beads) |
| `:xps_insulation`    | 0.027  | 55   | 1210 | —             | Incropera 7e, A.3 (extruded R-12) |
| `:glass_fiber_board` | 0.036  | 105  | 795  | —             | Incropera 7e, A.3 (organic bonded) |
| `:sand`              | 0.27   | 1515 | 800  | —             | Incropera 7e, A.3 |
| `:soil`              | 0.52   | 2050 | 1840 | —             | Incropera 7e, A.3 |
| `:clay`              | 1.3    | 1460 | 880  | —             | Incropera 7e, A.3 |

Values are for dry material near 300 K. Moisture raises the
conductivity of porous masonry and soil substantially; a moist soil
column belongs to `SoilColumn`, not here. The `optical` column names the
`SurfaceOptics` spectral-library entry an exposed layer of this material
takes its albedo and emissivity from.
"""
function thermal_material(name::Symbol)
    haskey(URBAN_THERMAL_MATERIALS, name) ||
        throw(ArgumentError("thermal_material: unknown material :$name; " *
                            "available: $(thermal_material_names())"))
    return URBAN_THERMAL_MATERIALS[name]
end

# =========================================================================
# 2. Layers and steady (ISO 6946) resistance
# =========================================================================

"""
    WallLayer(material, thickness; ncells = nothing, max_cell = 0.02)

One homogeneous layer of a construction: a [`ThermalMaterial`](@ref)
(or the `Symbol` of a table entry) of `thickness` metres, resolved by
`ncells` equal finite-volume cells — by default
`max(1, ceil(thickness / max_cell))`, i.e. cells of at most 2 cm.

Layers are listed **outside first**: the first layer of a wall is the
facade, of a roof the waterproofing, of a road the wearing course.
"""
struct WallLayer{T<:AbstractFloat}
    material::ThermalMaterial{T}
    thickness::T
    ncells::Int
end

function WallLayer(material::ThermalMaterial{T}, thickness::Real;
                   ncells::Union{Nothing, Integer} = nothing,
                   max_cell::Real = 0.02) where {T}
    thickness > 0 || throw(ArgumentError("WallLayer: thickness must be positive"))
    n = ncells === nothing ? max(1, ceil(Int, thickness / max_cell - 1e-9)) : Int(ncells)
    n >= 1 || throw(ArgumentError("WallLayer: ncells must be ≥ 1"))
    return WallLayer{T}(material, T(thickness), n)
end
WallLayer(name::Symbol, thickness::Real; kwargs...) =
    WallLayer(thermal_material(name), thickness; kwargs...)

"Thermal resistance ``d/k`` of one layer (m² K W⁻¹)."
thermal_resistance(l::WallLayer) = l.thickness / l.material.conductivity

"""
    thermal_resistance(layers) -> R

Conductive resistance of a stack of layers, ``R = Σ d_i / k_i``
(m² K W⁻¹), excluding surface resistances.
"""
thermal_resistance(layers::AbstractVector{<:WallLayer}) = sum(thermal_resistance, layers)

"""
    ISO6946_SURFACE_RESISTANCE

Conventional surface resistances of ISO 6946:2017, Table 7 (m² K W⁻¹),
combining convection and long-wave radiation at a plane surface:

| key          | value | meaning |
|:-------------|:------|:--------|
| `inside_up`         | 0.10 | interior, heat flow upward (ceiling / roof underside) |
| `inside_horizontal` | 0.13 | interior, horizontal heat flow (wall) |
| `inside_down`       | 0.17 | interior, heat flow downward (floor) |
| `outside`           | 0.04 | exterior, any direction |

``h = 1/R`` is the default interior surface coefficient of a facet —
e.g. 7.69 W m⁻² K⁻¹ for a wall.
"""
const ISO6946_SURFACE_RESISTANCE = (inside_up = 0.10, inside_horizontal = 0.13,
                                    inside_down = 0.17, outside = 0.04)

"""
    thermal_transmittance(layers; R_si = 0.13, R_se = 0.04) -> U

Steady thermal transmittance (U-value, W m⁻² K⁻¹) of a plane
construction between inside and outside air,
``U = 1 / (R_{si} + Σ d_i/k_i + R_{se})`` (ISO 6946). Defaults are the
ISO 6946 surface resistances for a wall.
"""
thermal_transmittance(layers::AbstractVector{<:WallLayer};
                      R_si::Real = ISO6946_SURFACE_RESISTANCE.inside_horizontal,
                      R_se::Real = ISO6946_SURFACE_RESISTANCE.outside) =
    1 / (R_si + thermal_resistance(layers) + R_se)

# =========================================================================
# 3. Boundary conditions
# =========================================================================

"""
    ColumnBoundary(T_env, h, q)

A Robin boundary condition at one face of a [`ConductionColumn`](@ref).
The heat flux crossing the face *into the column* is

    F = h (T_env - T_face) + q

with `h` (W m⁻² K⁻¹) a surface coefficient to an environment at
`T_env` (K) and `q` (W m⁻²) an imposed flux. Special cases, with their
constructors:

  * [`fixed_temperature_boundary`](@ref)`(T)` — `h = Inf`: Dirichlet,
    `T_face = T` (deep soil, a prescribed surface temperature);
  * [`heat_flux_boundary`](@ref)`(q)` — `h = 0`: Neumann, `F = q`
    (`q = 0` is adiabatic);
  * [`convective_boundary`](@ref)`(T_env, h)` — `q = 0`: an interior
    face exchanging with indoor air at `T_env` through ``h = 1/R_{si}``.

A facet's outer face is also a `ColumnBoundary`: the surface energy
balance, linearised about the skin temperature ``T_s^n``,
``R_n - H - LE ≈ F_0 + F'(T - T_s^n)``, is exactly
`ColumnBoundary(T_s^n, -F', F_0)`.
"""
struct ColumnBoundary{T<:AbstractFloat}
    T_env::T
    h::T
    q::T
    function ColumnBoundary{T}(T_env, h, q) where {T<:AbstractFloat}
        h >= 0 || throw(ArgumentError("ColumnBoundary: h must be ≥ 0"))
        return new{T}(T_env, h, q)
    end
end
function ColumnBoundary(T_env::Real, h::Real, q::Real)
    Tp = float(promote_type(typeof(T_env), typeof(h), typeof(q)))
    return ColumnBoundary{Tp}(Tp(T_env), Tp(h), Tp(q))
end

"Dirichlet face: `T_face = T`."
fixed_temperature_boundary(T::Real) = ColumnBoundary(float(T), Inf, 0.0)
"Neumann face: a flux `q` (W m⁻²) into the column; `q = 0` is adiabatic."
heat_flux_boundary(q::Real) = ColumnBoundary(0.0, 0.0, float(q))
"Robin face exchanging with an environment (indoor air) at `T_env` through coefficient `h`."
convective_boundary(T_env::Real, h::Real) = ColumnBoundary(float(T_env), float(h), 0.0)

# Eliminating the face temperature between F = h (T_env - T_f) + q and the
# half-cell conduction F = K (T_f - T_c), K = 2k/Δz, gives the boundary flux
# as an affine function of the adjacent cell temperature,
#   F = g (T_env - T_c) + s,   g = K h/(K + h),   s = K q/(K + h),
# with the h = Inf (Dirichlet) limit g = K, s = 0.
@inline function _boundary_coefficients(bc::ColumnBoundary{T}, K::Real) where {T}
    isinf(bc.h) && return (K, zero(K))
    return (K * bc.h / (K + bc.h), K * bc.q / (K + bc.h))
end

@inline function _face_temperature(bc::ColumnBoundary, K::Real, Tc::Real)
    isinf(bc.h) && return bc.T_env
    return (K * Tc + bc.h * bc.T_env + bc.q) / (K + bc.h)
end

@inline function _boundary_flux(bc::ColumnBoundary, K::Real, Tc::Real)
    g, s = _boundary_coefficients(bc, K)
    return g * (bc.T_env - Tc) + s
end

# =========================================================================
# 4. The conduction column
# =========================================================================

"""
    ConductionColumn(layers; T = 293.15, θ = 0.5)

A 1-D heat-conduction column through `layers` (outside first), each
[`WallLayer`](@ref) split into equal finite-volume cells. Solves

```math
ρ c\\,\\frac{∂T}{∂t} = \\frac{∂}{∂z}\\Bigl(k \\frac{∂T}{∂z}\\Bigr)
```

with cell-centred temperatures and face conductances
``K_{i+½} = 1/(Δz_i/2k_i + Δz_{i+1}/2k_{i+1})`` — the series resistance
of the two half cells, so a material interface lying on a cell face is
represented exactly and the steady composite-wall flux is exact for any
resolution.

Time stepping is the θ-scheme ([`step_conduction!`](@ref)):
`θ = 0.5` is Crank–Nicolson (second order, A-stable), `θ = 1` backward
Euler (first order, L-stable — damps the stiffest modes, preferable
for time steps much longer than ``Δz²/κ`` of the thinnest cell).
`θ` must lie in `[0.5, 1]`: below ½ the scheme is only conditionally
stable, so it is rejected.

`T` is the initial temperature: a scalar, or a vector of cell values.
Fields: `T` (cell temperatures, K), `dz`, `z` (cell centres measured from
the outer face, m), `k`, `C` (``ρc`` per cell), `layer` (layer index of
each cell), `K` (the `n+1` face conductances, including the half-cell
conductances ``2k/Δz`` of the two boundary cells).

This is the reusable core: it knows nothing about the atmosphere. A
facet ([`UrbanFacet`](@ref)) and the building energy model close it with
[`ColumnBoundary`](@ref)s.
"""
struct ConductionColumn{T<:AbstractFloat}
    layers::Vector{WallLayer{T}}
    T::Vector{T}
    dz::Vector{T}
    z::Vector{T}
    k::Vector{T}
    C::Vector{T}
    layer::Vector{Int}
    K::Vector{T}
    θ::T
    # Thomas-algorithm work arrays
    _a::Vector{T}
    _b::Vector{T}
    _c::Vector{T}
    _d::Vector{T}
end

function ConductionColumn(layers::AbstractVector{<:WallLayer}; T = 293.15, θ::Real = 0.5)
    isempty(layers) && throw(ArgumentError("ConductionColumn: need at least one layer"))
    _check_theta(θ, "ConductionColumn")
    FT = promote_type(map(l -> typeof(l.thickness), layers)...)
    Ls = WallLayer{FT}[WallLayer{FT}(ThermalMaterial{FT}(l.material.name,
                         FT(l.material.conductivity), FT(l.material.density),
                         FT(l.material.specific_heat), l.material.optical,
                         l.material.source), FT(l.thickness), l.ncells) for l in layers]
    dz, k, C, lay = FT[], FT[], FT[], Int[]
    for (j, l) in enumerate(Ls)
        for _ in 1:l.ncells
            push!(dz, l.thickness / l.ncells)
            push!(k, l.material.conductivity)
            push!(C, volumetric_heat_capacity(l.material))
            push!(lay, j)
        end
    end
    n = length(dz)
    z = cumsum(dz) .- dz ./ 2
    K = Vector{FT}(undef, n + 1)
    K[1] = 2k[1] / dz[1]
    for i in 1:n-1
        K[i+1] = 1 / (dz[i] / (2k[i]) + dz[i+1] / (2k[i+1]))
    end
    K[n+1] = 2k[n] / dz[n]
    Tv = T isa Real ? fill(FT(T), n) : collect(FT, T)
    length(Tv) == n || throw(ArgumentError("ConductionColumn: T must have $n entries"))
    return ConductionColumn{FT}(Ls, Tv, dz, z, k, C, lay, K, FT(θ),
                                zeros(FT, n), zeros(FT, n), zeros(FT, n), zeros(FT, n))
end

# θ < ½ (explicit Euler at θ = 0) is only conditionally stable — a facet
# with 2 cm brick cells stepped at 600 s with θ = 0 diverges to 1e8 K within
# 100 steps — and it would also make the linearised skin balance explicit.
# Only the unconditionally stable range is accepted.
function _check_theta(θ::Real, who::AbstractString)
    (0.5 <= θ <= 1) ||
        throw(ArgumentError("$who: θ must lie in [0.5, 1] (Crank–Nicolson to " *
                            "backward Euler); θ < 0.5 is only conditionally stable"))
    return nothing
end

Base.length(col::ConductionColumn) = length(col.T)

"Cell-centre depths (m) of a column, measured from its outer face."
cell_centers(col::ConductionColumn) = col.z

"""
    column_heat_content(col; T_ref = 0) -> J m⁻²

Sensible heat stored in the column per unit face area,
``Σ_i ρ_i c_i Δz_i (T_i - T_{ref})``. Its change over a
[`step_conduction!`](@ref) equals the returned boundary fluxes times
`dt`, to round-off.
"""
column_heat_content(col::ConductionColumn; T_ref::Real = 0) =
    sum(i -> col.C[i] * col.dz[i] * (col.T[i] - T_ref), eachindex(col.T))

"""
    face_temperature(col, bc, side) -> K

Temperature of the `:outer` or `:inner` face of `col` under boundary
condition `bc`, from the adjacent cell and the half-cell conductance.
"""
function face_temperature(col::ConductionColumn, bc::ColumnBoundary, side::Symbol)
    side === :outer && return _face_temperature(bc, col.K[1], col.T[1])
    side === :inner && return _face_temperature(bc, col.K[end], col.T[end])
    throw(ArgumentError("face_temperature: side must be :outer or :inner"))
end

"""
    step_conduction!(col, outer, inner, dt; θ = col.θ) -> (F_outer, F_inner)

Advance `col` by `dt` seconds with [`ColumnBoundary`](@ref) `outer` at
the first layer's face and `inner` at the last. The θ-scheme

```math
ρ_i c_i Δz_i \\frac{T_i^{n+1} - T_i^n}{Δt} = θ R_i(T^{n+1}) + (1-θ) R_i(T^n)
```

(``R_i`` the net conductive and boundary flux into cell ``i``) is a
tridiagonal system solved by the Thomas algorithm, O(n).

Returns the θ-weighted boundary fluxes *into the column* (W m⁻²) that
were applied over the step, so that

    column_heat_content(after) - column_heat_content(before) = (F_outer + F_inner) dt

holds to round-off: the scheme is conservative by construction.
"""
function step_conduction!(col::ConductionColumn{T}, outer::ColumnBoundary,
                          inner::ColumnBoundary, dt::Real; θ::Real = col.θ) where {T}
    _check_theta(θ, "step_conduction!")
    dt > 0 || throw(ArgumentError("step_conduction!: dt must be positive"))
    n = length(col.T)
    a, b, c, d = col._a, col._b, col._c, col._d
    K, Tn = col.K, col.T
    θ = T(θ)
    go, so = _boundary_coefficients(outer, K[1])
    gi, si = _boundary_coefficients(inner, K[n+1])
    Fo_old = go * (T(outer.T_env) - Tn[1]) + so
    Fi_old = gi * (T(inner.T_env) - Tn[n]) + si
    for i in 1:n
        m = col.C[i] * col.dz[i] / T(dt)
        # explicit part R_i(Tⁿ)
        R = zero(T)
        if i > 1
            R += K[i] * (Tn[i-1] - Tn[i])
        end
        if i < n
            R += K[i+1] * (Tn[i+1] - Tn[i])
        end
        # diagonal coupling (conductances leaving the cell)
        diag = zero(T)
        a[i] = zero(T); c[i] = zero(T)
        if i > 1
            diag += K[i]; a[i] = -θ * K[i]
        end
        if i < n
            diag += K[i+1]; c[i] = -θ * K[i+1]
        end
        rhs_bc = zero(T)
        if i == 1
            diag += go
            R += Fo_old
            rhs_bc += θ * (go * T(outer.T_env) + so)
        end
        if i == n
            diag += gi
            R += Fi_old
            rhs_bc += θ * (gi * T(inner.T_env) + si)
        end
        b[i] = m + θ * diag
        # m Tⁿ + (1-θ) R(Tⁿ) + θ (affine boundary constant)
        d[i] = m * Tn[i] + (1 - θ) * R + rhs_bc
    end
    _thomas!(Tn, a, b, c, d)
    Fo_new = go * (T(outer.T_env) - Tn[1]) + so
    Fi_new = gi * (T(inner.T_env) - Tn[n]) + si
    return (θ * Fo_new + (1 - θ) * Fo_old, θ * Fi_new + (1 - θ) * Fi_old)
end

# Thomas algorithm for a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i].
# Overwrites c and d; diagonal dominance (b ≥ |a| + |c| + m) makes it stable.
function _thomas!(x, a, b, c, d)
    n = length(d)
    c[1] = c[1] / b[1]
    d[1] = d[1] / b[1]
    for i in 2:n
        w = b[i] - a[i] * c[i-1]
        c[i] = c[i] / w
        d[i] = (d[i] - a[i] * d[i-1]) / w
    end
    x[n] = d[n]
    for i in n-1:-1:1
        x[i] = d[i] - c[i] * x[i+1]
    end
    return x
end

# =========================================================================
# 5. Urban facets — AbstractSurface implementers
# =========================================================================

"""
    UrbanFacet(kind, layers; T = 293.15, T_interior = T, h_interior = default,
               interior = default, albedo = derived, emissivity = derived,
               z0m = 0.01, z0h = z0m / 10, h_convective = nothing, θ = 0.5)

A roof, wall or ground (road) facet: a [`ConductionColumn`](@ref)
through `layers` (outside first) whose outer face is closed by the
surface energy balance and whose inner face is the interior hook.

`kind ∈ (:roof, :wall, :ground)` sets the default interior condition:

| kind      | default `interior` |
|:----------|:-------------------|
| `:roof`   | [`convective_boundary`](@ref)`(T_interior, 1/0.10)` — ISO 6946 `R_si`, heat flow up |
| `:wall`   | `convective_boundary(T_interior, 1/0.13)` — ISO 6946 `R_si`, horizontal |
| `:ground` | [`fixed_temperature_boundary`](@ref)`(T_interior)` — deep-ground temperature at the base of the column |

Pass `h_interior` to change only the coefficient, or `interior` (any
[`ColumnBoundary`](@ref)) to replace it; [`set_interior_temperature!`](@ref)
and [`set_interior_boundary!`](@ref) change it between steps — this is
the hook through which a building energy model imposes its indoor air
temperature.

**Optics are linked, not stored.** Unless `albedo` / `emissivity` are
given, they are derived from the `SurfaceOptics` spectral-library entry
named by the outer layer's `optical` field: the white-sky broadband
albedo `broadband_albedo(material_albedo(name))` and the broadband
`material_emissivity(name)`. To use a `Radiosity` urban class instead,
pass `albedo = material_albedo(mats, URBAN_WALL)` etc. An outer
material without an optical entry requires both keywords.

**Turbulent exchange.** With `h_convective = nothing` (default) `H` and
`LE` come from `SurfaceExchange.turbulent_fluxes` — Monin–Obukhov
similarity with roughness lengths `z0m`, `z0h` (defaults are generic
placeholders, not published values; set them per site). A number
`h_convective` (W m⁻² K⁻¹) instead gives ``H = h (T_s - T_{air})``,
``LE = 0``, ``τ = 0`` — the usual closure for a vertical wall, for which
MOST has no formal basis. Facets are dry (`moisture_availability = 0`).

**The skin is the outer face**, not an extra prognostic layer: its
temperature is whatever closes the energy balance against the half-cell
conductance ``2k_1/Δz_1`` of the first cell. The constructor starts it
at the first cell's temperature, so the first step moves it onto the
balance (≈ 1 K for a sunlit roof); each step is one Newton iteration on
that balance and the residual falls quadratically.

For a wall, `SurfaceForcing.sw_down` / `lw_down` are the irradiances on
the facet plane. [`UrbanFacetScene`](@ref) supplies them from the
radiosity solves and treats the facet-to-facet longwave implicitly.
"""
mutable struct UrbanFacet{T<:AbstractFloat} <: AbstractSurface
    kind::Symbol
    column::ConductionColumn{T}
    interior::ColumnBoundary{T}
    T_skin::T
    albedo::T
    emissivity::T
    z0m::T
    z0h::T
    h_convective::Union{Nothing, T}
    G::T                  # last step: heat into the outer face, W m⁻²
    Q_interior::T         # last step: heat from the inner face into the interior, W m⁻²
end

const _FACET_KINDS = (:roof, :wall, :ground)

function _default_interior(kind::Symbol, T_int::Real, h_int)
    kind === :ground && return fixed_temperature_boundary(T_int)
    h = h_int !== nothing ? h_int :
        kind === :roof ? 1 / ISO6946_SURFACE_RESISTANCE.inside_up :
                         1 / ISO6946_SURFACE_RESISTANCE.inside_horizontal
    return convective_boundary(T_int, h)
end

function _linked_optics(layers, albedo, emissivity)
    albedo !== nothing && emissivity !== nothing && return (albedo, emissivity)
    opt = first(layers).material.optical
    opt === nothing &&
        throw(ArgumentError("UrbanFacet: outer material :$(first(layers).material.name) " *
                            "has no SurfaceOptics entry; pass `albedo` and `emissivity`"))
    α = albedo !== nothing ? albedo : broadband_albedo(SurfaceOptics.material_albedo(opt))
    ε = emissivity !== nothing ? emissivity : SurfaceOptics.material_emissivity(opt)
    return (α, ε)
end

function UrbanFacet(kind::Symbol, layers::AbstractVector{<:WallLayer};
                    T = 293.15, T_interior::Real = (T isa Real ? T : last(T)),
                    h_interior = nothing,
                    interior::Union{Nothing, ColumnBoundary} = nothing,
                    albedo = nothing, emissivity = nothing,
                    z0m::Real = 0.01, z0h::Real = z0m / 10,
                    h_convective = nothing, θ::Real = 0.5)
    kind in _FACET_KINDS ||
        throw(ArgumentError("UrbanFacet: kind must be one of $_FACET_KINDS"))
    col = ConductionColumn(layers; T = T, θ = θ)
    FT = eltype(col.T)
    ibc = interior !== nothing ? interior : _default_interior(kind, T_interior, h_interior)
    ibc = ColumnBoundary{FT}(FT(ibc.T_env), FT(ibc.h), FT(ibc.q))
    α, ε = _linked_optics(layers, albedo, emissivity)
    (0 <= α < 1) || throw(ArgumentError("UrbanFacet: albedo must lie in [0, 1)"))
    (0 < ε <= 1) || throw(ArgumentError("UrbanFacet: emissivity must lie in (0, 1]"))
    hc = h_convective === nothing ? nothing : FT(h_convective)
    return UrbanFacet{FT}(kind, col, ibc, col.T[1], FT(α), FT(ε), FT(z0m), FT(z0h),
                          hc, zero(FT), zero(FT))
end

surface_temperature(f::UrbanFacet) = f.T_skin
roughness_lengths(f::UrbanFacet) = (z0m = f.z0m, z0h = f.z0h, z0q = f.z0h)
surface_albedo(f::UrbanFacet) = f.albedo
surface_emissivity(f::UrbanFacet) = f.emissivity

"""
    interior_heat_flux(f::UrbanFacet) -> W m⁻²

Heat flux from the facet's inner face into the interior (the room air,
or the deep ground) over the last [`step_surface!`](@ref), positive
into the interior. This is the building-envelope load a building energy
model receives.
"""
interior_heat_flux(f::UrbanFacet) = f.Q_interior

"Cell temperatures (K) of a facet's conduction column, outside first."
facet_layer_temperatures(f::UrbanFacet) = f.column.T

"Replace the interior boundary condition of a facet (the building-energy hook)."
function set_interior_boundary!(f::UrbanFacet{T}, bc::ColumnBoundary) where {T}
    f.interior = ColumnBoundary{T}(T(bc.T_env), T(bc.h), T(bc.q))
    return f
end

"""
    set_interior_temperature!(f, T_in)

Set the indoor air (or deep-ground) temperature seen by the facet's
inner face, keeping its surface coefficient.
"""
set_interior_temperature!(f::UrbanFacet{T}, T_in::Real) where {T} =
    (f.interior = ColumnBoundary{T}(T(T_in), f.interior.h, f.interior.q); f)

# One θ-step of the facet column under the linearised outer balance
# `outer`. Returns the applied fluxes into the column at both faces, the
# new skin (outer face) temperature and the θ-weighted skin increment
# x = θ T_s^{n+1} + (1-θ) T_s^{n,*} - T_s^n on which the linearised
# fluxes are evaluated (T_s^{n,*}: the face temperature of the old cells
# under `outer`).
function _advance_column!(f::UrbanFacet, outer::ColumnBoundary, dt::Real)
    col = f.column
    θ = col.θ
    Tf_old = _face_temperature(outer, col.K[1], col.T[1])
    Fo, Fi = step_conduction!(col, outer, f.interior, dt)
    Tf_new = _face_temperature(outer, col.K[1], col.T[1])
    return Fo, Fi, Tf_new, θ * Tf_new + (1 - θ) * Tf_old - outer.T_env
end

# Fluxes at the current skin temperature.
function _facet_fluxes(f::UrbanFacet, forcing::SurfaceForcing;
                       c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS, kwargs...)
    if f.h_convective === nothing
        return turbulent_fluxes(f, forcing; c = c, kwargs...)
    end
    Rn = net_radiation(f, forcing; c = c)
    H = f.h_convective * (f.T_skin - forcing.T_air)
    z = zero(Rn)
    return SurfaceFluxes(promote(Rn, H, z, Rn - H, z, z)...)
end

"""
    step_surface!(f::UrbanFacet, forcing, dt; δT = 0.01, kwargs...) -> SurfaceFluxes

Advance a facet by `dt` seconds. The surface energy balance
``F(T_s) = R_n - H - LE`` is evaluated at the current skin temperature
``T_s^n`` and its derivative ``F'`` by a one-sided difference of width
`δT` (so the Monin–Obukhov stability dependence of `H` is included),
and the linearisation ``F_0 + F'(T - T_s^n)`` becomes the outer
[`ColumnBoundary`](@ref) — ``h = -F'``, the combined radiative and
turbulent surface coefficient. The column step is then fully implicit
in the skin temperature, and the new skin temperature is the outer face
temperature.

The returned `SurfaceFluxes` are the θ-weighted fluxes actually applied:
``R_n - H - LE = G`` holds exactly, and `G·dt` plus the interior flux
equals the change in column heat content to round-off. `kwargs` go to
`SurfaceExchange.turbulent_fluxes` (`sf`, `c`, `U_min`, `gustiness`).
"""
function step_surface!(f::UrbanFacet{T}, forcing::SurfaceForcing, dt::Real;
                       δT::Real = 0.01, kwargs...) where {T}
    Ts = f.T_skin
    fl0 = _facet_fluxes(f, forcing; kwargs...)
    f.T_skin = Ts + T(δT)
    fl1 = _facet_fluxes(f, forcing; kwargs...)
    f.T_skin = Ts
    dRn = (fl1.Rn - fl0.Rn) / T(δT)
    dH  = (fl1.H - fl0.H) / T(δT)
    dLE = (fl1.LE - fl0.LE) / T(δT)
    F0 = fl0.Rn - fl0.H - fl0.LE
    # F' must be negative (a warmer surface loses more energy). Longwave
    # emission alone guarantees it; guard against a pathological MOST
    # derivative by falling back to the radiative part.
    Fp = dRn - dH - dLE
    Fp < 0 || (Fp = min(dRn, -eps(T)))
    outer = ColumnBoundary{T}(T(Ts), T(-Fp), T(F0))
    Fo, Fi, Tf_new, x = _advance_column!(f, outer, dt)
    Rn = fl0.Rn + dRn * x
    LE = fl0.LE + dLE * x
    H  = Rn - LE - Fo          # = fl0.H + dH x (up to the Fp guard)
    E  = LE / surface_latent_heat(f)
    f.T_skin = Tf_new
    f.G = Fo
    f.Q_interior = -Fi
    return SurfaceFluxes{T}(T(Rn), T(H), T(LE), T(Fo), T(fl0.τ), T(E))
end

# =========================================================================
# 6. Example constructions
# =========================================================================

"""
    example_wall_layers() -> Vector{WallLayer}

An illustrative uninsulated fair-faced (exposed) brick wall, outside
first: 24 cm common brick, 1.5 cm gypsum plaster. Assembled from the
material table to exercise the model; it is not a survey of any
building stock. ``U = 1.750`` W m⁻² K⁻¹ with ISO 6946 surface
resistances.
"""
example_wall_layers() = [WallLayer(:brick, 0.24), WallLayer(:gypsum_plaster, 0.015)]

"""
    example_roof_layers() -> Vector{WallLayer}

An illustrative insulated flat roof, outside first: 1 cm asphalt
waterproofing, 10 cm extruded polystyrene, 20 cm concrete slab, 1.5 cm
gypsum plaster ceiling. Illustrative, not a survey.
"""
example_roof_layers() = [WallLayer(:asphalt, 0.01; ncells = 2), WallLayer(:xps_insulation, 0.10),
                         WallLayer(:concrete, 0.20), WallLayer(:gypsum_plaster, 0.015)]

"""
    example_ground_layers() -> Vector{WallLayer}

An illustrative road, outside first: 5 cm asphalt wearing course, 15 cm
concrete base, 30 cm sand sub-base, 1 m soil. The inner face of a
`:ground` facet is a deep-ground temperature at 1.5 m. Illustrative,
not a survey.
"""
example_ground_layers() = [WallLayer(:asphalt, 0.05; ncells = 5), WallLayer(:concrete, 0.15),
                           WallLayer(:sand, 0.30; max_cell = 0.05),
                           WallLayer(:soil, 1.0; max_cell = 0.1)]

# =========================================================================
# 7-11. Glazing and window facets (#27)
# =========================================================================

include("urban_surface/glazing.jl")

# =========================================================================
# 12. Facet–radiosity coupling (#28)
# =========================================================================

"""
    example_facet_layers(facet_or_kind) -> Vector{WallLayer}

The illustrative construction for a `Radiosity.Facet` (or a kind symbol):
[`example_roof_layers`](@ref), [`example_wall_layers`](@ref) or
[`example_ground_layers`](@ref). The default `layers` of
[`UrbanFacetScene`](@ref).
"""
example_facet_layers(f::Facet) = example_facet_layers(f.kind)
function example_facet_layers(kind::Symbol)
    kind === :roof && return example_roof_layers()
    kind === :wall && return example_wall_layers()
    kind === :ground && return example_ground_layers()
    throw(ArgumentError("example_facet_layers: kind must be one of $_FACET_KINDS"))
end

"""
    UrbanFacetScene(scene, vf; layers = example_facet_layers,
                    options = f -> (;), T = 293.15, θ = 0.5)
    UrbanFacetScene(scene, vf, facets::Vector{<:UrbanFacet})

Every facet of a `Radiosity.RadiosityScene` given its own
[`UrbanFacet`](@ref) — its own conduction column and its own skin
temperature — and coupled to all the others through the longwave
radiosity system on the view factors `vf`.

This removes the one-temperature-per-column approximation of the urban
example, where a whole column's roof, walls and street shared a single
temperature and the radiosity solution was averaged back onto it. Here
the radiosity scene *is* the surface: facet `i` of `scene` is facet `i`
of the energy balance.

  * `layers(f)` — the construction of radiosity facet `f` (outside first);
    the default picks the illustrative roof / wall / road by `f.kind`.
  * `options(f)` — extra [`UrbanFacet`](@ref) keywords per facet, e.g.
    `f -> f.kind === :wall ? (h_convective = 10.0, T_interior = 295.0) : (;)`.
  * `T` — initial temperature, a scalar or one value per facet.

**Optics have one source: the scene.** Each facet's albedo and emissivity
are taken from `scene.materials` for its material class at construction
and re-synchronised at every [`step_urban_facets!`](@ref), so the
radiosity solve and the facet energy balances can never use different
optical properties (an `albedo`/`emissivity` passed through `options`
is overwritten).

Fields of interest after a step (per facet, W m⁻² of facet area):
`sw_absorbed`, `lw_net` (net longwave gain actually applied),
`lw_emission` (the emission `εσT⁴` it corresponds to) and
`skin_increment` (the θ-weighted skin-temperature change the linearised
fluxes were evaluated at). `facet_temperatures(u)` returns the skin
temperatures. `solve_iterations` and `solve_residual` report the last
coupled solve (see [`step_urban_facets!`](@ref)).
"""
mutable struct UrbanFacetScene{T<:AbstractFloat}
    scene::RadiosityScene{T}
    vf::ViewFactors{T}
    facets::Vector{UrbanFacet{T}}
    response::LongwaveResponse{T}
    sw_absorbed::Vector{T}
    lw_net::Vector{T}
    lw_emission::Vector{T}
    skin_increment::Vector{T}
    L_sky::T
    solve_iterations::Int
    solve_residual::T
end

function UrbanFacetScene(scene::RadiosityScene{T}, vf::ViewFactors{T},
                         facets::AbstractVector{<:UrbanFacet}) where {T}
    n = nfacets(scene)
    length(vf.F_sky) == n ||
        throw(ArgumentError("UrbanFacetScene: view factors have $(length(vf.F_sky)) " *
                            "facets, scene has $n"))
    length(facets) == n ||
        throw(ArgumentError("UrbanFacetScene: $(length(facets)) UrbanFacets for $n " *
                            "radiosity facets"))
    fs = UrbanFacet{T}[f for f in facets]
    # every facet owns its column: the facet steps run one work item per
    # facet (KA-m, #301), and a facet listed twice would be stepped twice
    allunique(map(objectid, fs)) ||
        throw(ArgumentError("UrbanFacetScene: the same UrbanFacet object is listed more than once"))
    for i in 1:n
        fs[i].kind === scene.facets[i].kind ||
            throw(ArgumentError("UrbanFacetScene: facet $i is a :$(fs[i].kind) " *
                                "UrbanFacet but a :$(scene.facets[i].kind) radiosity facet"))
    end
    z = zeros(T, n)
    u = UrbanFacetScene{T}(scene, vf, fs, longwave_response(scene, vf),
                           copy(z), copy(z), copy(z), copy(z), zero(T), 0, zero(T))
    _sync_optics!(u)
    _sync_scene_temperatures!(u)
    return u
end

function UrbanFacetScene(scene::RadiosityScene{FT}, vf::ViewFactors{FT};
                         layers = example_facet_layers,
                         options = f -> NamedTuple(),
                         T = 293.15, θ::Real = 0.5) where {FT}
    n = nfacets(scene)
    Tv = T isa Real ? fill(T, n) : collect(T)
    length(Tv) == n ||
        throw(ArgumentError("UrbanFacetScene: T has $(length(Tv)) entries for $n facets"))
    mats = scene.materials
    facets = map(1:n) do i
        rf = scene.facets[i]
        kw = merge(NamedTuple(options(rf)),
                   (albedo = mats.albedo[rf.material], emissivity = mats.emissivity[rf.material]))
        UrbanFacet(rf.kind, layers(rf); T = Tv[i], θ = θ, kw...)
    end
    return UrbanFacetScene(scene, vf, facets)
end

nfacets(u::UrbanFacetScene) = length(u.facets)
"Skin temperatures (K) of every facet of an [`UrbanFacetScene`](@ref)."
facet_temperatures(u::UrbanFacetScene{T}) where {T} = T[f.T_skin for f in u.facets]

function Base.show(io::IO, u::UrbanFacetScene{T}) where {T}
    Ts = facet_temperatures(u)
    print(io, "UrbanFacetScene{$T}(", nfacets(u), " facets, T_skin ",
          round(minimum(Ts); digits = 2), "–", round(maximum(Ts); digits = 2), " K)")
end

# Albedo and emissivity come from the scene's material table.
function _sync_optics!(u::UrbanFacetScene{T}) where {T}
    mats = u.scene.materials
    for (i, f) in enumerate(u.facets)
        m = u.scene.facets[i].material
        f.albedo = mats.albedo[m]
        f.emissivity = mats.emissivity[m]
    end
    ε = T[f.emissivity for f in u.facets]
    ε == u.response.emissivity || (u.response = longwave_response(u.scene, u.vf))
    return u
end

function _sync_scene_temperatures!(u::UrbanFacetScene)
    for (i, f) in enumerate(u.facets)
        set_temperature!(u.scene, i, f.T_skin)
    end
    return u
end

_sky_irradiance(f::SurfaceForcing) = f.lw_down
function _sky_irradiance(fs::AbstractVector{<:SurfaceForcing})
    L = first(fs).lw_down
    all(f -> f.lw_down == L, fs) ||
        throw(ArgumentError("step_urban_facets!: per-facet forcings disagree on " *
                            "lw_down; pass the sky irradiance as `L_sky`"))
    return L
end

_air(f::SurfaceForcing, i) = f
_air(f::AbstractVector{<:SurfaceForcing}, i) = f[i]

# The air state of `air`, with the radiation replaced by the facet's
# incident irradiances from the radiosity solves.
_facet_forcing(air::SurfaceForcing{T}, sw, lw) where {T} =
    SurfaceForcing{T}(T(sw), T(lw), air.T_air, air.q_air, air.U, air.p, air.z_ref,
                      air.rain, air.snow)

function _incident_shortwave(u::UrbanFacetScene{T}, sw, sun, dni, dhi, samples) where {T}
    sw !== nothing && return sw.incident
    if sun === nothing
        dhi > 0 || return zeros(T, nfacets(u))
        return solve_shortwave(u.scene, u.vf, (0.0, 0.0, 1.0); dni = 0, dhi = dhi,
                               samples = samples).incident
    end
    return solve_shortwave(u.scene, u.vf, sun; dni = dni, dhi = dhi,
                           samples = samples).incident
end

"""
    step_urban_facets!(u::UrbanFacetScene, forcing, dt; L_sky, sw = nothing,
                       sun = nothing, dni = 0, dhi = 0, samples = 2,
                       coupling = :implicit, solver = :iterative, tol = 1e-12,
                       maxiter = 200, δT = 0.01, kwargs...) -> Vector{SurfaceFluxes}

Advance every facet of `u` by `dt` seconds with the longwave exchange
between facets treated **implicitly**, and return each facet's
`SurfaceFluxes` (per unit facet area, `SurfaceExchange` sign convention).

`forcing` is the air state — one `SurfaceForcing` for all facets, or one
per facet (canyon air for the street and walls, above-roof air for the
roofs). Its `sw_down` and `lw_down` are **not** used as facet irradiances:
the facets see what the radiosity solves deliver. `L_sky` is the sky's
downwelling longwave on a horizontal surface and defaults to
`forcing.lw_down`. Shortwave is a precomputed `sw::ShortwaveSolution`, or
is solved here from `sun` (anything `Radiosity.solar_unit_vector`
accepts), `dni` and `dhi`; with `sun = nothing` only the diffuse sky
`dhi` is applied. It does not depend on the facet temperatures.

# The implicit longwave exchange

The net longwave a facet absorbs is linear in the emission of every facet,
``N_i = \\sum_j J_{ij} E_j + j^{sky}_i L_{sky}``
(`Radiosity.longwave_response`). Linearising ``E_j = ε_jσT_j^4`` about the
current skin temperatures gives the coupled balance of facet `i`,

```math
R_{n,i} - H_i - LE_i \\approx F_{0,i} + F'_{ii}\\,x_i + \\sum_{j\\ne i} G_{ij}\\,x_j,
\\qquad G_{ij} = J_{ij}\\,4ε_jσT_j^3,
```

where `x` are the θ-weighted skin increments of the step and ``F'_{ii}``
combines the radiative ``G_{ii}`` with the turbulent derivatives. The
diagonal term closes each facet column as in
`step_surface!(::UrbanFacet, …)`. For the off-diagonal term, each column
step is affine in an extra imposed surface flux, ``x_i = a_i + b_i c_i``
(two trial steps give `a` and `b`); the dense system
``(I - \\mathrm{diag}(b)\\,G_{off})\\,x = a`` then gives every facet's
increment at once, and the columns are advanced with
``c_i = \\sum_{j≠i} G_{ij} x_j``.

# Solving the coupled system

`solver = :iterative` (the default) solves it matrix-free with restarted
GMRES: each iteration is one product with the precomputed radiosity
response `J` (``G\\,v = J\\,(∂E/∂T ∘ v)``), `O(n²)` work and `O(n)` extra
memory, and no `n × n` matrix is formed or factorised per step. The
system is strongly diagonally dominant — ``b_i`` is the skin warming per
W m⁻² of extra flux (≲ 1/(4εσT³ + h + conduction)) and
``\\sum_j |G_{ij}|`` is at most the longwave conductance towards the
other facets, so ``b_i\\sum_{j≠i}|G_{ij}| < 1`` — and GMRES converges in
a few iterations; it stops when the residual 2-norm is below
`tol × max(‖a‖₂, 1 K)` and throws if `maxiter` products do not get there.
The longwave at the start of the step is also evaluated from `J`
(`Radiosity.solve_longwave(scene, vf, response)`), so a step costs
`O(n²)`; only a change of emissivities refactorises (`O(n³)`, once).
`solver = :direct` assembles the dense ``(n+m)^2`` matrix and LU-factorises
it (`O(n³)` per step): the reference the test suite compares against, to
`1e-10` K. `u.solve_iterations` and `u.solve_residual` report the last
solve.

What this buys, and why a lagged coupling is not enough: the longwave
that facet `j` emits *extra* during the step, ``G_{\\cdot j}x_j``, is the
longwave the other facets absorb. So the exchange conserves energy
exactly: in an enclosure with no sky, ``\\sum_i A_i N_i = 0`` for the
applied fluxes to round-off, and with a sky, `lw_net` equals the
radiosity solution for the applied emission `lw_emission`. With
`coupling = :lagged` each facet instead steps alone against the incident
longwave frozen at the start of the step (`step_surface!` with
`lw_down = Q_i`) — its own extra emission leaves it but reaches no one —
which is provided only as the reference the test suite measures the
conservation defect of.

`kwargs` go to `SurfaceExchange.turbulent_fluxes`.
"""
function step_urban_facets!(u::UrbanFacetScene{T}, forcing, dt::Real;
                            L_sky::Real = _sky_irradiance(forcing),
                            sw::Union{Nothing, ShortwaveSolution} = nothing,
                            sun = nothing, dni::Real = 0, dhi::Real = 0,
                            samples::Int = 2, coupling::Symbol = :implicit,
                            solver::Symbol = :iterative, tol::Real = 1e-12,
                            maxiter::Int = 200, δT::Real = 0.01, kwargs...) where {T}
    coupling in (:implicit, :lagged) ||
        throw(ArgumentError("step_urban_facets!: coupling must be :implicit or :lagged"))
    if coupling === :lagged
        lw, Qsw = _scene_radiation!(u, forcing, dt; L_sky, sw, sun, dni, dhi, samples)
        return _step_lagged!(u, forcing, lw, Qsw, dt; δT, kwargs...)
    end
    sys = _prepare_scene_step(u, forcing, dt; L_sky, sw, sun, dni, dhi, samples, δT,
                              solver, tol, maxiter, kwargs...)
    x, _, _ = _solve_scene_step(sys)
    return _apply_scene_step!(u, sys, x)
end

# Validate the step and solve the radiation that does not depend on the
# facet temperatures of the new step: the longwave at the current skins and
# the incident shortwave. Sets `sw_absorbed` and `L_sky`.
function _scene_radiation!(u::UrbanFacetScene{T}, forcing, dt; L_sky, sw, sun, dni, dhi,
                           samples) where {T}
    dt > 0 || throw(ArgumentError("step_urban_facets!: dt must be positive"))
    n = nfacets(u)
    forcing isa AbstractVector && length(forcing) != n &&
        throw(ArgumentError("step_urban_facets!: $(length(forcing)) forcings for $n facets"))
    fs = u.facets
    _sync_optics!(u)
    _sync_scene_temperatures!(u)
    # the response is current (`_sync_optics!`): one O(n²) product, no LU
    lw = all(f -> f.emissivity > 0, fs) ? solve_longwave(u.scene, u.vf, u.response; L_sky = L_sky) :
                                          solve_longwave(u.scene, u.vf; L_sky = L_sky)
    Qsw = _incident_shortwave(u, sw, sun, dni, dhi, samples)
    u.L_sky = T(L_sky)
    for i in 1:n
        u.sw_absorbed[i] = (1 - fs[i].albedo) * Qsw[i]
    end
    return lw, Qsw
end

# ---- the implicit step as a linear system ----------------------------------
#
# Every facet column step is affine in (i) an extra flux imposed on its outer
# face, c_i, and (ii) the temperature of the interior environment of its inner
# face. `_prepare_scene_step` measures both responses with trial steps (the
# column state is restored after each): the θ-weighted skin increment
#     x_i = a_i + b_i c_i + d_i y_k
# and the flux into the column at the inner face
#     F_i = F0_i + e_i c_i + g_i y_k,
# where y_k is the change over the step of the temperature of the interior
# node k the facet is bound to (`node[i] = k`, 0 for none: the interior
# environment is then fixed). The longwave exchange closes c = G_off x.
#
# Interior nodes are what a building energy model adds: the zone air behind a
# set of roof and wall facets, whose own balance is linear in y_k,
#     lhs_k y_k = R_k + Φ_k + Σ_{i∈k} A_i (q_i - F_i),
# with q_i the radiative flux imposed on the inner face (in `f.interior.q`),
# lhs_k and R_k whatever else the node exchanges with (the caller's), and Φ_k
# a heat input that is either given (a free node) or the unknown that holds
# y_k at a prescribed value (a node on its thermostat setpoint).
# `_solve_scene_step` solves facets and nodes together, so the exchange
# between a zone and its facets is as implicit as the one between facets.
struct _SceneStep{T, F}
    forcing::F
    dt::T
    lw::Radiosity.LongwaveSolution{T}
    J::Matrix{T}          # longwave response dN/dE (shared with the scene, not copied)
    dEdT::Vector{T}       # G = J diag(dEdT); never formed on the iterative path
    Gii::Vector{T}
    N0::Vector{T}
    outers::Vector{ColumnBoundary{T}}
    fl0s::Vector{SurfaceFluxes{T}}
    dLE::Vector{T}
    a::Vector{T}
    b::Vector{T}
    d::Vector{T}
    F0::Vector{T}
    e::Vector{T}
    g::Vector{T}
    node::Vector{Int}
    area::Vector{T}
    q::Vector{T}
    solver::Symbol
    tol::T
    maxiter::Int
    iterations::Base.RefValue{Int}
    residual::Base.RefValue{T}
end

# out = G_off v = J (dEdT ∘ v) - diag(G) v, with `tmp` as scratch: O(n²)
function _offdiag_exchange!(out, s::_SceneStep, v, tmp)
    tmp .= s.dEdT .* v
    mul!(out, s.J, tmp)
    out .-= s.Gii .* v
    return out
end

function _prepare_scene_step(u::UrbanFacetScene{T}, forcing, dt::Real;
                             node::AbstractVector{<:Integer} = zeros(Int, nfacets(u)),
                             L_sky::Real = _sky_irradiance(forcing),
                             sw::Union{Nothing, ShortwaveSolution} = nothing,
                             sun = nothing, dni::Real = 0, dhi::Real = 0,
                             samples::Int = 2, δT::Real = 0.01,
                             solver::Symbol = :iterative, tol::Real = 1e-12,
                             maxiter::Int = 200, kwargs...) where {T}
    solver in (:iterative, :direct) ||
        throw(ArgumentError("step_urban_facets!: solver must be :iterative or :direct"))
    tol > 0 || throw(ArgumentError("step_urban_facets!: tol must be positive"))
    lw, Qsw = _scene_radiation!(u, forcing, dt; L_sky, sw, sun, dni, dhi, samples)
    n = nfacets(u)
    length(node) == n || throw(ArgumentError("step_urban_facets!: `node` needs one entry per facet"))
    fs = u.facets
    σ = T(STEFAN_BOLTZMANN)
    J = u.response.dN_dE
    dEdT = T[4 * f.emissivity * σ * f.T_skin^3 for f in fs]
    Gii = T[J[i, i] * dEdT[i] for i in 1:n]
    N0 = -lw.net_loss
    outers = Vector{ColumnBoundary{T}}(undef, n)
    fl0s = Vector{SurfaceFluxes{T}}(undef, n)
    dLE = zeros(T, n)
    a = zeros(T, n); b = zeros(T, n); d = zeros(T, n)
    F0 = zeros(T, n); e = zeros(T, n); g = zeros(T, n)
    # the facets' trial steps: one work item per facet (each restores its
    # own column; KA-m, #301), so the same bits at every thread count
    foreach_column!(backend(a), _prepare_facet!, (1:n,), fs, forcing, Qsw, lw, u.sw_absorbed,
                    N0, Gii, outers, fl0s, dLE, a, b, d, F0, e, g, node, T(δT), dt,
                    NamedTuple(kwargs); weight = FACET_WEIGHT)
    return _SceneStep{T, typeof(forcing)}(forcing, T(dt), lw, J, dEdT, Gii, N0, outers, fl0s, dLE,
                                          a, b, d, F0, e, g, collect(Int, node), copy(u.vf.area),
                                          T[f.interior.q for f in fs], solver, T(tol), maxiter,
                                          Ref(0), Ref(zero(T)))
end

# The work of one facet's conduction-column step in grid-point equivalents
# (a tridiagonal θ-step of tens of layers plus the surface fluxes): the
# serial threshold of the facet launches counts it.
const FACET_WEIGHT = 256

# One facet of `_prepare_scene_step`: its linearised outer balance and its
# trial steps (the column state restored after each). Writes entry `i` of
# the step's vectors and nothing of any other facet.
function _prepare_facet!(i, fs, forcing, Qsw, lw, sw_absorbed, N0, Gii, outers, fl0s, dLE,
                         a, b, d, F0, e, g, node, δT::T, dt, kw) where {T}
    f = fs[i]
    Δq = T(100)
    fi = _facet_forcing(_air(forcing, i), Qsw[i], lw.incident[i])
    Ts = f.T_skin
    fl0 = _facet_fluxes(f, fi; kw...)
    f.T_skin = Ts + δT
    fl1 = _facet_fluxes(f, fi; kw...)
    f.T_skin = Ts
    dH = (fl1.H - fl0.H) / δT
    dLE[i] = (fl1.LE - fl0.LE) / δT
    fl0s[i] = fl0
    F0i = sw_absorbed[i] + N0[i] - fl0.H - fl0.LE
    Fp = Gii[i] - dH - dLE[i]
    Fp < 0 || (Fp = min(Gii[i], -eps(T)))
    outers[i] = ColumnBoundary{T}(Ts, -Fp, F0i)
    # trial steps, state restored after each
    save = copy(f.column.T)
    _, Fi_a, _, a[i] = _advance_column!(f, outers[i], dt)
    F0[i] = Fi_a
    copyto!(f.column.T, save)
    o2 = ColumnBoundary{T}(Ts, -Fp, F0i + Δq)
    _, Fi_b, _, xb = _advance_column!(f, o2, dt)
    b[i] = (xb - a[i]) / Δq
    e[i] = (Fi_b - Fi_a) / Δq
    copyto!(f.column.T, save)
    if node[i] != 0
        ib = f.interior
        f.interior = ColumnBoundary{T}(ib.T_env + one(T), ib.h, ib.q)
        _, Fi_d, _, xd = _advance_column!(f, outers[i], dt)
        d[i] = xd - a[i]
        g[i] = Fi_d - Fi_a
        f.interior = ib
        copyto!(f.column.T, save)
    end
    return nothing
end

# Solve the facets (and interior nodes) of a prepared step. With nodes, pass
# per node `lhs`, `R` (see above), `pinned[k]` and either the prescribed
# increment `y[k]` (pinned) or the heat input `Φ[k]` (free). Returns the skin
# increments x, the node increments y and the node heat inputs Φ.
function _solve_scene_step(s::_SceneStep{T}; lhs::AbstractVector = T[], R::AbstractVector = T[],
                           pinned::AbstractVector{Bool} = falses(length(lhs)),
                           y::AbstractVector = zeros(T, length(lhs)),
                           Φ::AbstractVector = zeros(T, length(lhs))) where {T}
    m = length(lhs)
    length(R) == length(pinned) == length(y) == length(Φ) == m ||
        throw(ArgumentError("interior nodes: lhs, R, pinned, y and Φ must have one entry per node"))
    all(k -> 0 <= k <= m, s.node) || throw(ArgumentError("interior nodes: facet bound to a missing node"))
    return s.solver === :direct ? _solve_scene_direct(s, lhs, R, pinned, y, Φ) :
                                  _solve_scene_iterative(s, lhs, R, pinned, y, Φ)
end

# The reference: the dense (n+m)² system, LU-factorised (O((n+m)³)).
function _solve_scene_direct(s::_SceneStep{T}, lhs, R, pinned, y, Φ) where {T}
    n = length(s.a); m = length(lhs)
    M = zeros(T, n + m, n + m)
    rhs = zeros(T, n + m)
    J, dEdT = s.J, s.dEdT
    @inbounds for i in 1:n
        M[i, i] = one(T)
        for j in 1:n
            i == j && continue
            M[i, j] = -s.b[i] * (J[i, j] * dEdT[j])
        end
        rhs[i] = s.a[i]
    end
    for k in 1:m
        rhs[n + k] = R[k]
        if pinned[k]
            M[n + k, n + k] = -one(T)                 # unknown: Φ_k
            rhs[n + k] -= lhs[k] * y[k]
        else
            M[n + k, n + k] = T(lhs[k])               # unknown: y_k
            rhs[n + k] += Φ[k]
        end
    end
    @inbounds for i in 1:n
        k = s.node[i]
        k == 0 && continue
        A = s.area[i]
        # facet: x_i - d_i y_k
        # node:  A_i e_i c_i + A_i g_i y_k, moved to the left; A_i (q_i - F0_i) to R
        rhs[n + k] += A * (s.q[i] - s.F0[i])
        for j in 1:n
            j == i && continue
            M[n + k, j] += A * s.e[i] * (J[i, j] * dEdT[j])
        end
        if pinned[k]
            rhs[i] += s.d[i] * y[k]
            rhs[n + k] -= A * s.g[i] * y[k]
        else
            M[i, n + k] -= s.d[i]
            M[n + k, n + k] += A * s.g[i]
        end
    end
    sol = n + m == 0 ? T[] : Backends.with_dense_blas(() -> lu!(M) \ rhs)
    x = sol[1:n]
    yy = T[pinned[k] ? T(y[k]) : sol[n + k] for k in 1:m]
    ΦΦ = T[pinned[k] ? sol[n + k] : T(Φ[k]) for k in 1:m]
    s.iterations[] = 0
    s.residual[] = zero(T)
    return x, yy, ΦΦ
end

# Matrix-free solve of the same system. Each node row is linear in its own
# unknown given the facet exchange c = G_off x,
#     Σ_{i∈k} w_i c_i + α_k u_k = ρ_k,     w_i = A_i e_i,
# with u_k = y_k, α_k = lhs_k + Σ_{i∈k} A_i g_i for a free node and
# u_k = Φ_k, α_k = -1 for a pinned one (whose y_k is given). Eliminating the
# free nodes leaves a system in the facet increments alone,
#     x_i - b_i c_i + d_i [k free] (Σ_{j∈k} w_j c_j) / α_k
#         = a_i + d_i (pinned ? y_k : ρ_k / α_k),
# with a unit diagonal (c_i excludes x_i). It is solved by GMRES, one
# O(n²) product with J per iteration; the node unknowns follow from x.
function _solve_scene_iterative(s::_SceneStep{T}, lhs, R, pinned, y, Φ) where {T}
    n = length(s.a); m = length(lhs)
    ρn = zeros(T, m); α = zeros(T, m)
    for k in 1:m
        ρn[k] = R[k] + (pinned[k] ? -lhs[k] * y[k] : Φ[k])
        α[k] = pinned[k] ? -one(T) : T(lhs[k])
    end
    w = zeros(T, n)
    for i in 1:n
        k = s.node[i]
        k == 0 && continue
        A = s.area[i]
        w[i] = A * s.e[i]
        ρn[k] += A * (s.q[i] - s.F0[i])
        pinned[k] ? (ρn[k] -= A * s.g[i] * y[k]) : (α[k] += A * s.g[i])
    end
    for k in 1:m
        pinned[k] || α[k] != 0 ||
            throw(ArgumentError("interior nodes: node $k is free with no heat capacity and no " *
                                "facets (singular)"))
    end
    rhs = copy(s.a)
    for i in 1:n
        k = s.node[i]
        k == 0 && continue
        rhs[i] += s.d[i] * (pinned[k] ? T(y[k]) : ρn[k] / α[k])
    end
    c = zeros(T, n); tmp = zeros(T, n); sk = zeros(T, m)
    function op!(out, v)
        _offdiag_exchange!(c, s, v, tmp)
        fill!(sk, zero(T))
        @inbounds for i in 1:n
            k = s.node[i]
            k != 0 && !pinned[k] && (sk[k] += w[i] * c[i])
        end
        @inbounds for i in 1:n
            k = s.node[i]
            out[i] = v[i] - s.b[i] * c[i]
            k != 0 && !pinned[k] && (out[i] += s.d[i] * sk[k] / α[k])
        end
        return out
    end
    x = copy(rhs)                          # the uncoupled columns as first guess
    if n > 0
        its, res = _gmres!(x, op!, rhs; tol = s.tol * max(norm(rhs), one(T)),
                           restart = min(n, 30), maxiter = s.maxiter)
        res <= s.tol * max(norm(rhs), one(T)) ||
            throw(NumericalFailure("step_urban_facets!: GMRES did not converge in " *
                                   "$(s.maxiter) iterations (residual $res K); raise " *
                                   "`maxiter` or use `solver = :direct`"))
        s.iterations[] = its
        s.residual[] = res
    end
    # the node unknowns from the converged exchange
    _offdiag_exchange!(c, s, x, tmp)
    fill!(sk, zero(T))
    for i in 1:n
        k = s.node[i]
        k != 0 && (sk[k] += w[i] * c[i])
    end
    yy = T[pinned[k] ? T(y[k]) : (ρn[k] - sk[k]) / α[k] for k in 1:m]
    ΦΦ = T[pinned[k] ? sk[k] - ρn[k] : T(Φ[k]) for k in 1:m]
    return x, yy, ΦΦ
end

# Restarted GMRES (Saad & Schultz 1986) with modified Gram–Schmidt and
# Givens rotations, for op!(out, v) = A v. Solves A x = b in place from the
# initial `x` until ‖b - A x‖₂ ≤ tol (absolute) or `maxiter` products.
# Returns (products used, true final residual ‖b - A x‖₂).
function _gmres!(x::Vector{T}, op!, b::Vector{T}; tol::T, restart::Int,
                 maxiter::Int) where {T}
    n = length(b)
    V = zeros(T, n, restart + 1)
    H = zeros(T, restart + 1, restart)
    cs = zeros(T, restart); sn = zeros(T, restart); gv = zeros(T, restart + 1)
    r = zeros(T, n); wv = zeros(T, n)
    its = 0
    op!(wv, x); r .= b .- wv
    β = norm(r)
    while β > tol && its < maxiter
        fill!(gv, zero(T)); gv[1] = β
        V[:, 1] .= r ./ β
        k = 0
        for j in 1:restart
            k = j
            op!(wv, view(V, :, j)); its += 1
            for i in 1:j                                  # modified Gram–Schmidt
                H[i, j] = dot(view(V, :, i), wv)
                wv .-= H[i, j] .* view(V, :, i)
            end
            H[j + 1, j] = norm(wv)
            H[j + 1, j] > 0 && (V[:, j + 1] .= wv ./ H[j + 1, j])
            for i in 1:j-1                                # previous rotations
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            end
            ρh = hypot(H[j, j], H[j + 1, j])
            cs[j] = H[j, j] / ρh; sn[j] = H[j + 1, j] / ρh
            H[j, j] = ρh; H[j + 1, j] = zero(T)
            gv[j + 1] = -sn[j] * gv[j]; gv[j] = cs[j] * gv[j]
            (abs(gv[j + 1]) <= tol || its >= maxiter) && break
        end
        # x += V_k z with H[1:k, 1:k] z = g[1:k] (upper triangular)
        z = zeros(T, k)
        for i in k:-1:1
            acc = gv[i]
            for l in i+1:k
                acc -= H[i, l] * z[l]
            end
            z[i] = acc / H[i, i]
        end
        mul!(x, view(V, :, 1:k), z, one(T), one(T))
        op!(wv, x); r .= b .- wv                          # true residual
        β = norm(r)
    end
    return its, β
end

# Advance every facet with the solved skin increments `x` (and node
# increments `y`: the interior environment of a bound facet becomes
# T_env + y_k, and stays there). Returns the facet fluxes; the flux into
# each column at its inner face is `-f.Q_interior`.
function _apply_scene_step!(u::UrbanFacetScene{T}, s::_SceneStep{T}, x::AbstractVector,
                            y::AbstractVector = T[]) where {T}
    fs = u.facets
    n = length(fs)
    tmp = zeros(T, n)
    c = _offdiag_exchange!(zeros(T, n), s, x, tmp)
    out = Vector{SurfaceFluxes{T}}(undef, n)
    Fo = zeros(T, n)
    # the facet column steps: one work item per facet (KA-m, #301); the
    # coupled solve above (one global system) stays serial
    foreach_column!(backend(Fo), _apply_facet!, (1:n,), fs, s.node, y, s.outers, c, s.dt, Fo,
                    u.skin_increment; weight = FACET_WEIGHT)
    xa = u.skin_increment
    # lw_net = N0 + G xa (the full G, diagonal included)
    tmp .= s.dEdT .* xa
    mul!(u.lw_net, s.J, tmp)
    u.lw_net .+= s.N0
    for (i, f) in enumerate(fs)
        u.lw_emission[i] = s.lw.emitted[i] + s.dEdT[i] * xa[i]
        Rn = u.sw_absorbed[i] + u.lw_net[i]
        LE = s.fl0s[i].LE + s.dLE[i] * xa[i]
        H = Rn - LE - Fo[i]
        out[i] = SurfaceFluxes{T}(Rn, H, LE, Fo[i], s.fl0s[i].τ, LE / surface_latent_heat(f))
    end
    u.solve_iterations = s.iterations[]
    u.solve_residual = s.residual[]
    _sync_scene_temperatures!(u)
    return out
end

# one facet's column step of `_apply_scene_step!`
function _apply_facet!(i, fs, node, y, outers, c, dt, Fo, skin_increment)
    f = fs[i]
    T = eltype(Fo)
    k = node[i]
    if k != 0
        ib = f.interior
        f.interior = ColumnBoundary{T}(ib.T_env + T(y[k]), ib.h, ib.q)
    end
    o = outers[i]
    outer = ColumnBoundary{T}(o.T_env, o.h, o.q + c[i])
    Fo[i], Fi, Tf_new, skin_increment[i] = _advance_column!(f, outer, dt)
    f.T_skin = Tf_new
    f.G = Fo[i]
    f.Q_interior = -Fi
    return nothing
end

function _step_lagged!(u::UrbanFacetScene{T}, forcing, lw, Qsw, dt; kwargs...) where {T}
    fs = u.facets
    out = Vector{SurfaceFluxes{T}}(undef, length(fs))
    # one work item per facet (KA-m, #301): each steps its own column
    foreach_column!(backend(u.lw_net), _lagged_facet!, (1:length(fs),), u, fs, forcing, lw, Qsw,
                    dt, out, NamedTuple(kwargs); weight = FACET_WEIGHT)
    _sync_scene_temperatures!(u)
    return out
end

function _lagged_facet!(i, u::UrbanFacetScene{T}, fs, forcing, lw, Qsw, dt, out, kw) where {T}
    f = fs[i]
    Ts = f.T_skin
    fi = _facet_forcing(_air(forcing, i), Qsw[i], lw.incident[i])
    out[i] = step_surface!(f, fi, dt; kw...)
    u.lw_net[i] = out[i].Rn - u.sw_absorbed[i]
    # frozen incident longwave: the applied net gain is ε Qⁿ - E
    u.lw_emission[i] = f.emissivity * lw.incident[i] - u.lw_net[i]
    u.skin_increment[i] = (u.lw_emission[i] - lw.emitted[i]) /
                          (4 * f.emissivity * T(STEFAN_BOLTZMANN) * Ts^3)
    return nothing
end

"""
    aggregate_facet_fluxes(u::UrbanFacetScene, fluxes) -> NamedTuple

Sum facet fluxes onto the `(Nx, Ny)` grid of the scene's heightmap, **per
unit plan area**: each facet contributes `A_i × flux_i / (dx·dy)` to the
column `Facet.col` it is attributed to (the open column a wall faces).
Returns `(Rn, H, LE, G, E, Q_interior, area_density)`; `H`, `LE` and `E`
are what the atmosphere above the column receives. Plan-area totals equal
facet-area totals exactly, so no energy is created or lost in the
aggregation — only the per-facet detail is.
"""
function aggregate_facet_fluxes(u::UrbanFacetScene{T},
                                fluxes::AbstractVector{<:SurfaceFluxes}) where {T}
    nx, ny = size(u.scene.heightmap)
    cell = u.scene.dx * u.scene.dy
    Rn = zeros(T, nx, ny); H = zeros(T, nx, ny); LE = zeros(T, nx, ny)
    G = zeros(T, nx, ny); E = zeros(T, nx, ny); Qi = zeros(T, nx, ny)
    ad = zeros(T, nx, ny)
    for (i, rf) in enumerate(u.scene.facets)
        ci, cj = rf.col
        (1 <= ci <= nx && 1 <= cj <= ny) || continue
        w = rf.area / cell
        fl = fluxes[i]
        Rn[ci, cj] += w * fl.Rn; H[ci, cj] += w * fl.H; LE[ci, cj] += w * fl.LE
        G[ci, cj] += w * fl.G; E[ci, cj] += w * fl.E
        Qi[ci, cj] += w * u.facets[i].Q_interior
        ad[ci, cj] += w
    end
    return (Rn = Rn, H = H, LE = LE, G = G, E = E, Q_interior = Qi, area_density = ad)
end

# =========================================================================
# 13. Street-canyon exchange closures (#30)
# =========================================================================

include("urban_surface/canyon_exchange.jl")

end # module
