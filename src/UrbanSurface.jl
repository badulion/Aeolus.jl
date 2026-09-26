"""
    UrbanSurface

Phase 49 — the production urban surface: per-facet (roof / wall /
ground) multi-layer heat conduction through real material layers,
replacing the single heat-capacity slab of the urban example.

Three parts:

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
  3. **Facets** — [`UrbanFacet`](@ref), an implementer of the Era V
     `SurfaceExchange.AbstractSurface` interface: a conduction column
     whose outer face is closed by the surface energy balance
     ``R_n = H + LE + G``, linearised about the current skin temperature
     so the step stays implicit and conserves energy exactly.

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
    (after Goward, S. N., 1981, *Photogramm. Eng. Remote Sens.* 47,
    1291-1302).
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
import ..SurfaceExchange: surface_temperature, roughness_lengths, step_surface!
import ..RadiationCore: surface_albedo, surface_emissivity
using ..SurfaceOptics: SurfaceOptics, broadband_albedo

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
       example_roof_layers, example_wall_layers, example_ground_layers

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
end
function ColumnBoundary(T_env::Real, h::Real, q::Real)
    h >= 0 || throw(ArgumentError("ColumnBoundary: h must be ≥ 0"))
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
    (0 <= θ <= 1) || throw(ArgumentError("ConductionColumn: θ must lie in [0, 1]"))
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

For a wall, `SurfaceForcing.sw_down` / `lw_down` are the irradiances on
the facet plane (the radiosity coupling that supplies them is Phase 49's
facet–radiosity issue).
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
    col = f.column
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
    θ = col.θ
    Tf_old = _face_temperature(outer, col.K[1], col.T[1])
    Fo, Fi = step_conduction!(col, outer, f.interior, dt)
    Tf_new = _face_temperature(outer, col.K[1], col.T[1])
    x = θ * Tf_new + (1 - θ) * Tf_old - Ts
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

end # module
