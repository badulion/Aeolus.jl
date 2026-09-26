"""
    Snowpack

A layered seasonal snowpack in the tradition of CLM, SNOWPACK and
Crocus: a column of snow layers, each carrying its ice and liquid-water
mass, thickness and temperature, that settles under its own weight,
conducts heat, melts, sublimates and accumulates fresh snow. The pack
is an Era V surface: [`LayeredSnowpack`](@ref) implements
`SurfaceExchange.AbstractSurface`, so its turbulent fluxes, net
radiation and coupling to the atmosphere come from the era contract.

Material laws are dispatch hierarchies so a user (or a later phase)
chooses among published parameterisations without touching the solver:

| law                           | abstract type                     | implementations |
|:----------------------------- |:--------------------------------- |:--------------- |
| effective thermal conductivity | [`AbstractSnowConductivity`](@ref) | `Yen1981Conductivity`, `Sturm1997Conductivity`, `Jordan1991Conductivity`, `Calonne2011Conductivity` |
| fresh-snow density            | [`AbstractFreshSnowDensity`](@ref) | `Anderson1976FreshSnow`, `Vionnet2012FreshSnow` |
| settling / densification      | [`AbstractSnowCompaction`](@ref)  | `Anderson1976Compaction`, `Vionnet2012Compaction` |

Three hooks are deliberately left thin, for the phases that own them:

  * **grain and albedo ageing (#35)** — [`AbstractSnowMetamorphism`](@ref)
    (`snow_metamorphism!`, `viscosity_grain_factor`) and
    [`AbstractSnowAlbedo`](@ref) (`snowpack_albedo`). The defaults age
    the layers but leave the grain size and a constant albedo alone.
  * **liquid water (#36)** — [`AbstractSnowLiquidWater`](@ref)
    (`snow_liquid_water!`). The default drains all liquid water — melt
    and rain — as runoff within the step; percolation, retention and
    refreezing belong to #36.

Constants come from the contracts: ice and water density, ``k_{ice}``,
``k_{air}`` from `SurfaceExchange.SurfaceConstants`; ``c_{ice}``,
``c_w``, ``L_f`` and `g` through its accessors (i.e. from
`MoistThermodynamics`); the snow emissivity from
`SurfaceOptics.material_emissivity(:snow_fresh)`.

# References

  * Anderson, E. A. (1976). *A point energy and mass balance model of a
    snow cover.* NOAA Tech. Rep. NWS 19.
  * Oleson, K. W. et al. (2010). *Technical description of version 4.0
    of the Community Land Model (CLM).* NCAR/TN-478+STR, Sect. 6-7.
  * Vionnet, V. et al. (2012). The detailed snowpack scheme Crocus and
    its implementation in SURFEX v7.2. *Geosci. Model Dev.*, 5, 773-791.
  * Yen, Y.-C. (1981). *Review of thermal properties of snow, ice and
    sea ice.* CRREL Rep. 81-10.
  * Sturm, M., Holmgren, J., König, M. & Morris, K. (1997). The thermal
    conductivity of seasonal snow. *J. Glaciol.*, 43, 26-41.
  * Jordan, R. (1991). *A one-dimensional temperature model for a snow
    cover.* CRREL Spec. Rep. 91-16.
  * Calonne, N. et al. (2011). Numerical and experimental investigations
    of the effective thermal conductivity of snow. *Geophys. Res. Lett.*,
    38, L23501.
  * Sturm, M., Taras, B., Liston, G. E., Derksen, C., Jonas, T. & Lea, J.
    (2010). Estimating snow water equivalent using snow depth data and
    climate classes. *J. Hydrometeorol.*, 11, 1380-1394.
"""
module Snowpack

using LinearAlgebra: Tridiagonal
using ..MoistThermodynamics: latent_heat_sublim, saturation_specific_humidity
using ..SurfaceExchange
using ..SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS,
                         LH_f0, grav, cp_ice, cp_water
import ..SurfaceExchange: surface_temperature, roughness_lengths,
                          surface_albedo, surface_emissivity,
                          moisture_availability, surface_specific_humidity,
                          surface_latent_heat, step_surface!
using ..SurfaceOptics: material_emissivity, SNOW_REFERENCE_RADIUS

export AbstractSnowConductivity, Yen1981Conductivity, Sturm1997Conductivity,
       Jordan1991Conductivity, Calonne2011Conductivity, snow_thermal_conductivity,
       AbstractFreshSnowDensity, Anderson1976FreshSnow, Vionnet2012FreshSnow,
       fresh_snow_density,
       AbstractSnowCompaction, DestructiveMetamorphism, Anderson1976Compaction,
       Vionnet2012Compaction, snow_compaction_rate, snow_viscosity,
       AbstractSnowMetamorphism, NoSnowMetamorphism, snow_metamorphism!,
       viscosity_grain_factor,
       AbstractSnowAlbedo, ConstantSnowAlbedo, snowpack_albedo,
       AbstractSnowLiquidWater, InstantSnowRunoff, snow_liquid_water!,
       AbstractSnowBottom, SnowGroundTemperature, SnowBottomFlux,
       SnowLayering, LayeredSnowpack, SnowBudget,
       snow_nlayers, snow_depth, snow_water_equivalent, snow_bulk_density,
       snow_layer_density, snow_layer_depths, snowpack_enthalpy, snow_load,
       snow_compaction!, snow_heat_conduction!, snow_phase_change!,
       add_snowfall!, add_rainfall!, snow_sublimation!, relayer_snowpack!

_Tfreeze(c::SurfaceConstants) = c.thermo.T_freeze

# =========================================================================
# 1. Effective thermal conductivity
# =========================================================================

"""
    AbstractSnowConductivity

Effective thermal conductivity of snow as a function of bulk density.
Implementers extend `snow_thermal_conductivity(law, ρ; c)` (W m⁻¹ K⁻¹,
ρ in kg m⁻³). The spread between the published laws is real — at
300 kg m⁻³ they disagree by nearly a factor two — and reflects the
dependence of heat transport on the ice microstructure, not just on ρ
(Sturm et al. 1997; Calonne et al. 2011).
"""
abstract type AbstractSnowConductivity end

"""
    Yen1981Conductivity(; exponent = 1.88)

``k = k_{ice} (ρ/ρ_w)^{1.88}`` (Yen 1981), the law used by Crocus
(Vionnet et al. 2012, Eq. 18). ``k_{ice}`` and ``ρ_w`` are the contract's.
"""
Base.@kwdef struct Yen1981Conductivity <: AbstractSnowConductivity
    exponent::Float64 = 1.88
end

"""
    Sturm1997Conductivity()

The regression of Sturm et al. (1997, Eq. 4) through 488 measurements of
seasonal snow, with ρ in g cm⁻³:

    k = 0.138 - 1.01 ρ + 3.233 ρ²     (0.156 ≤ ρ ≤ 0.6)
    k = 0.023 + 0.234 ρ               (ρ < 0.156)

Above 0.6 g cm⁻³ (outside the data) the quadratic is used as is.
"""
struct Sturm1997Conductivity <: AbstractSnowConductivity end

"""
    Jordan1991Conductivity()

``k = k_{air} + (7.75·10^{-5} ρ + 1.105·10^{-6} ρ²)(k_{ice} - k_{air})``
(Jordan 1991), the CLM law (Oleson et al. 2010, Eq. 6.71). It reaches
``k_{ice}`` at the density of ice.
"""
struct Jordan1991Conductivity <: AbstractSnowConductivity end

"""
    Calonne2011Conductivity()

``k = 2.5·10^{-6} ρ² - 1.23·10^{-4} ρ + 0.024`` (Calonne et al. 2011),
fitted to conductivities computed on 3-D tomography images of snow; it
tends to still air (0.024) as ρ → 0.
"""
struct Calonne2011Conductivity <: AbstractSnowConductivity end

"""
    snow_thermal_conductivity(law, ρ; c = DEFAULT_SURFACE_CONSTANTS) -> W m⁻¹ K⁻¹

Effective conductivity of snow of bulk density `ρ` (kg m⁻³) under `law`.
"""
function snow_thermal_conductivity end

snow_thermal_conductivity(law::Yen1981Conductivity, ρ::Real;
                          c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    c.k_ice * (ρ / c.ρ_water)^law.exponent

function snow_thermal_conductivity(::Sturm1997Conductivity, ρ::Real;
                                   c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    ρg = ρ / 1000
    return ρg < 0.156 ? 0.023 + 0.234 * ρg : 0.138 - 1.01 * ρg + 3.233 * ρg^2
end

snow_thermal_conductivity(::Jordan1991Conductivity, ρ::Real;
                          c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    c.k_air + (7.75e-5 * ρ + 1.105e-6 * ρ^2) * (c.k_ice - c.k_air)

snow_thermal_conductivity(::Calonne2011Conductivity, ρ::Real;
                          c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    2.5e-6 * ρ^2 - 1.23e-4 * ρ + 0.024

# =========================================================================
# 2. Fresh-snow density
# =========================================================================

"""
    AbstractFreshSnowDensity

Density (kg m⁻³) of newly fallen snow. Implementers extend
`fresh_snow_density(law, T_air, U; c)`, with air temperature `T_air` (K)
and wind speed `U` (m s⁻¹).
"""
abstract type AbstractFreshSnowDensity end

"""
    Anderson1976FreshSnow()

The temperature-only law of Anderson (1976) as used in CLM4
(Oleson et al. 2010):

    ρ = 50                                  T ≤ T_f - 15
    ρ = 50 + 1.7 (T - T_f + 15)^{3/2}       T_f - 15 < T ≤ T_f + 2
    ρ = 50 + 1.7 · 17^{3/2} ≈ 169           T > T_f + 2
"""
struct Anderson1976FreshSnow <: AbstractFreshSnowDensity end

"""
    Vionnet2012FreshSnow(; a = 109, b = 6, c = 26, ρ_min = 50)

Crocus fresh snow (Vionnet et al. 2012, Eq. 1, after Pahaut 1976 at Col
de Porte): ``ρ = a + b (T - T_f) + c U^{1/2}``, bounded below by
`ρ_min`. Wind packs falling snow: at −5 °C, 0 m s⁻¹ gives 79 kg m⁻³ and
10 m s⁻¹ gives 161 kg m⁻³.
"""
Base.@kwdef struct Vionnet2012FreshSnow <: AbstractFreshSnowDensity
    a::Float64 = 109.0     # kg m⁻³
    b::Float64 = 6.0       # kg m⁻³ K⁻¹
    c::Float64 = 26.0      # kg m⁻⁷ᐟ² s¹ᐟ²
    ρ_min::Float64 = 50.0  # kg m⁻³
end

"""
    fresh_snow_density(law, T_air, U = 0; c = DEFAULT_SURFACE_CONSTANTS) -> kg m⁻³

Density of snow falling at air temperature `T_air` (K) and wind `U` (m s⁻¹).
"""
function fresh_snow_density end

function fresh_snow_density(::Anderson1976FreshSnow, T_air::Real, U::Real = 0.0;
                            c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    ΔT = T_air - _Tfreeze(c)
    ΔT <= -15 && return 50.0
    return 50.0 + 1.7 * (min(ΔT, 2.0) + 15)^1.5
end

fresh_snow_density(law::Vionnet2012FreshSnow, T_air::Real, U::Real = 0.0;
                   c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    max(law.ρ_min, law.a + law.b * (T_air - _Tfreeze(c)) + law.c * sqrt(max(U, 0.0)))

# =========================================================================
# 3. Compaction (settling)
# =========================================================================

"""
    AbstractSnowCompaction

A densification law. Implementers extend
`snow_compaction_rate(law, layer; c)`, returning the fractional rate
``C_R = (1/Δz) ∂Δz/∂t`` (s⁻¹, ≤ 0) of one layer, where `layer` is a
named tuple with fields

| field   | meaning                                               |
|:------- |:----------------------------------------------------- |
| `ρ`     | bulk density, ice + liquid, kg m⁻³                    |
| `ρ_ice` | bulk density of the ice alone, ``w_{ice}/Δz``, kg m⁻³ |
| `ρ_liq` | bulk density of liquid water, ``w_{liq}/Δz``, kg m⁻³  |
| `T`     | layer temperature, K                                  |
| `load`  | overburden, kg m⁻² (mass above + half the layer's own) |
| `f2`    | grain-shape viscosity factor from the metamorphism hook |

Layer mass is conserved by compaction, so ``∂ \\ln ρ/∂t = -C_R``.
"""
abstract type AbstractSnowCompaction end

"""
    DestructiveMetamorphism(; c3 = 2.777e-6, c4 = 0.04, ρ_threshold = 100,
                              decay = 0.046, wet_factor = 2, wet_threshold = 0.01)

Settling by destructive metamorphism of new snow — the breakdown of
dendritic crystals — after Anderson (1976), in the form of CLM4
(Oleson et al. 2010, Eqs. 7.40-7.41):

    C_R1 = -c3 c1 c2 exp[-c4 (T_f - T)]

with ``c_1 = 1`` for ``ρ_{ice} ≤ ρ_{thr}`` and
``\\exp[-0.046 (ρ_{ice} - ρ_{thr})]`` above, and ``c_2 = 2`` if the
liquid bulk density exceeds 0.01 kg m⁻³. At the melting point new snow
halves its thickness in ≈ 2.9 days from this term alone. CLM5 raised
`ρ_threshold` to 175 kg m⁻³ (van Kampenhout et al. 2017).
"""
Base.@kwdef struct DestructiveMetamorphism
    c3::Float64 = 2.777e-6          # s⁻¹
    c4::Float64 = 0.04              # K⁻¹
    ρ_threshold::Float64 = 100.0    # kg m⁻³
    decay::Float64 = 0.046          # m³ kg⁻¹
    wet_factor::Float64 = 2.0
    wet_threshold::Float64 = 0.01   # kg m⁻³
end

function _destructive_rate(d::DestructiveMetamorphism, layer, Tf)
    c1 = layer.ρ_ice <= d.ρ_threshold ? 1.0 : exp(-d.decay * (layer.ρ_ice - d.ρ_threshold))
    c2 = layer.ρ_liq > d.wet_threshold ? d.wet_factor : 1.0
    return -d.c3 * c1 * c2 * exp(-d.c4 * max(Tf - layer.T, 0.0))
end
_destructive_rate(::Nothing, layer, Tf) = 0.0

"""
    Anderson1976Compaction(; destructive = DestructiveMetamorphism(),
                             η0 = 9e5, c5 = 0.08, c6 = 0.023)

CLM4 compaction (Anderson 1976; Oleson et al. 2010, Eqs. 7.40-7.44):
destructive metamorphism plus overburden,

    C_R2 = -P_s / η,     η = η0 exp[c5 (T_f - T) + c6 ρ_ice]

with the load ``P_s`` in kg m⁻² and η in kg s m⁻² (``η_0 = 9·10^5``).
Pass `destructive = nothing` for overburden settling alone.
"""
Base.@kwdef struct Anderson1976Compaction{D<:Union{Nothing,DestructiveMetamorphism}} <: AbstractSnowCompaction
    destructive::D = DestructiveMetamorphism()
    η0::Float64 = 9.0e5     # kg s m⁻²
    c5::Float64 = 0.08      # K⁻¹
    c6::Float64 = 0.023     # m³ kg⁻¹
end

"""
    Vionnet2012Compaction(; destructive = nothing, η0 = 7.62237e6,
                            a_η = 0.1, b_η = 0.023, c_η = 250, liquid_factor = 60)

Crocus viscous settling (Vionnet et al. 2012, Eqs. 5-8):

    (1/D) dD/dt = -σ/η,     σ = g P_s,
    η = f1 f2 η0 (ρ/c_η) exp[a_η (T_f - T) + b_η ρ]

with η0 in Pa s, ``f_1 = 1/(1 + 60 w_{liq}/(ρ_w D))`` for liquid water,
and ``f_2`` the grain-shape factor supplied by the metamorphism hook
([`viscosity_grain_factor`](@ref); Crocus's
``\\min(4, \\exp(\\min(0.4, g_s - 0.2)/0.1))`` needs the grain size of #35,
so the default is 1). Crocus has no separate destructive term; one may
be added with `destructive = DestructiveMetamorphism()`.

Under a constant load at constant temperature this law integrates in
closed form: ``e^{b_η ρ(t)} = e^{b_η ρ_0} + b_η K t`` with
``K = σ c_η / (f_1 f_2 η_0 e^{a_η (T_f - T)})``.
"""
Base.@kwdef struct Vionnet2012Compaction{D<:Union{Nothing,DestructiveMetamorphism}} <: AbstractSnowCompaction
    destructive::D = nothing
    η0::Float64 = 7.62237e6     # Pa s  (kg m⁻¹ s⁻¹)
    a_η::Float64 = 0.1          # K⁻¹
    b_η::Float64 = 0.023        # m³ kg⁻¹
    c_η::Float64 = 250.0        # kg m⁻³
    liquid_factor::Float64 = 60.0
end

"""
    snow_viscosity(law, layer; c = DEFAULT_SURFACE_CONSTANTS)

Compactive viscosity of a layer: kg s m⁻² for `Anderson1976Compaction`
(the load is a mass), Pa s for `Vionnet2012Compaction` (the load is a
stress). See [`AbstractSnowCompaction`](@ref) for `layer`.
"""
function snow_viscosity end

snow_viscosity(law::Anderson1976Compaction, layer; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    law.η0 * exp(law.c5 * max(_Tfreeze(c) - layer.T, 0.0) + law.c6 * layer.ρ_ice)

function snow_viscosity(law::Vionnet2012Compaction, layer; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    θ_w = layer.ρ_liq / c.ρ_water                         # volumetric liquid content
    f1 = 1 / (1 + law.liquid_factor * θ_w)
    return f1 * layer.f2 * law.η0 * (layer.ρ / law.c_η) *
           exp(law.a_η * max(_Tfreeze(c) - layer.T, 0.0) + law.b_η * layer.ρ)
end

"""
    snow_compaction_rate(law, layer; c = DEFAULT_SURFACE_CONSTANTS) -> s⁻¹

Fractional thickness change ``(1/Δz)\\,∂Δz/∂t`` (negative) of one layer.
"""
function snow_compaction_rate end

snow_compaction_rate(law::Anderson1976Compaction, layer; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    _destructive_rate(law.destructive, layer, _Tfreeze(c)) - layer.load / snow_viscosity(law, layer; c = c)

snow_compaction_rate(law::Vionnet2012Compaction, layer; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    _destructive_rate(law.destructive, layer, _Tfreeze(c)) -
    grav(c) * layer.load / snow_viscosity(law, layer; c = c)


# =========================================================================
# 4. Hooks for the phases that extend this module
# =========================================================================

"""
    AbstractSnowMetamorphism

Grain metamorphism — the hook for snow ageing (#35). Implementers extend

  * `snow_metamorphism!(scheme, pack, dt)` — evolve the per-layer
    `grain` (optical radius, μm) and `age` (s) of `pack`;
  * `viscosity_grain_factor(scheme, pack, i)` — the Crocus ``f_2`` factor
    by which angular grains (depth hoar) stiffen layer `i` against
    settling (Vionnet et al. 2012, Eq. 9). Default 1.
"""
abstract type AbstractSnowMetamorphism end

"""
    NoSnowMetamorphism()

Default metamorphism: layers age (`age += dt`), grains keep their size.
"""
struct NoSnowMetamorphism <: AbstractSnowMetamorphism end

"""
    snow_metamorphism!(scheme, pack, dt) -> pack

Advance grain properties of every layer by `dt` seconds.
"""
function snow_metamorphism! end
function snow_metamorphism!(::NoSnowMetamorphism, pack, dt::Real)
    pack.age .+= dt
    return pack
end

"""
    viscosity_grain_factor(scheme, pack, i) -> f₂ ≥ 1

Grain-shape factor on the compactive viscosity of layer `i`; `1` unless
a metamorphism scheme says otherwise.
"""
viscosity_grain_factor(::AbstractSnowMetamorphism, pack, i::Integer) = 1.0

"""
    AbstractSnowAlbedo

Broadband albedo of the pack — the hook for albedo ageing (#35).
Implementers extend `snowpack_albedo(scheme, pack)`. Spectral snow
optics already live in `SurfaceOptics.SnowSurface` (Wiscombe-Warren
grain-size and soot dependence); a grain-aware scheme should delegate to
it rather than restate it.
"""
abstract type AbstractSnowAlbedo end

"""
    ConstantSnowAlbedo(; α = 0.8)

A fixed broadband albedo (0.8: aged dry snow; fresh snow is ≈ 0.85-0.9).
"""
Base.@kwdef struct ConstantSnowAlbedo <: AbstractSnowAlbedo
    α::Float64 = 0.8
end

"""
    snowpack_albedo(scheme, pack) -> α

Broadband shortwave albedo of `pack` under `scheme`.
"""
function snowpack_albedo end
snowpack_albedo(a::ConstantSnowAlbedo, pack) = a.α

"""
    AbstractSnowLiquidWater

Liquid water in the pack — the hook for percolation and refreezing
(#36). Implementers extend `snow_liquid_water!(scheme, pack, dt)`, which
moves, retains or refreezes the per-layer `m_liq` (kg m⁻²) and returns
the runoff leaving the base of the pack in the step (kg m⁻²). The pack
records the enthalpy the runoff carries away, so a scheme only has to
keep its own internal processes enthalpy-conserving.
"""
abstract type AbstractSnowLiquidWater end

"""
    InstantSnowRunoff()

Default: all liquid water (melt and rain) leaves the pack as runoff in
the step it appears. There is no retention and no refreezing.
"""
struct InstantSnowRunoff <: AbstractSnowLiquidWater end

"""
    snow_liquid_water!(scheme, pack, dt) -> runoff (kg m⁻²)

Redistribute liquid water in `pack` over `dt`; return the runoff.
"""
function snow_liquid_water! end
function snow_liquid_water!(::InstantSnowRunoff, pack, dt::Real)
    r = sum(pack.m_liq; init = 0.0)
    fill!(pack.m_liq, 0.0)
    return r
end

# =========================================================================
# 5. Lower boundary, layering, the pack
# =========================================================================

"""
    AbstractSnowBottom

Thermal condition at the base of the snowpack.
"""
abstract type AbstractSnowBottom end

"""
    SnowGroundTemperature(T)

The snow/ground interface is held at temperature `T` (K); the flux into
the ground is conducted across the lower half of the bottom layer.
Mutable, so a soil model can update `T` every step.
"""
mutable struct SnowGroundTemperature <: AbstractSnowBottom
    T::Float64
end

"""
    SnowBottomFlux(F = 0)

Prescribed heat flux `F` (W m⁻²) leaving the base of the snowpack into
the substrate (positive downward, the contract's sign for `G`);
`F = 0` is an insulated base, `F < 0` a geothermal / soil heat supply.
"""
mutable struct SnowBottomFlux <: AbstractSnowBottom
    F::Float64
end
SnowBottomFlux() = SnowBottomFlux(0.0)

# Linear bottom flux F_b = F0 + Kb·(T_n* - T_n) as (F0, Kb).
_bottom_terms(b::SnowGroundTemperature, k, dz, Tn) = (2k / dz * (Tn - b.T), 2k / dz)
_bottom_terms(b::SnowBottomFlux, k, dz, Tn) = (b.F, 0.0)
_bare_temperature(b::SnowGroundTemperature, c) = b.T
_bare_temperature(b::SnowBottomFlux, c) = _Tfreeze(c)

"""
    SnowLayering(; max_layers = 10, dz_top = 0.02, growth = 1.6,
                   min_mass = 0.01, min_total_mass = 1e-6)

Rules for the numerical layers. Layer `i` (1 = top) has a target
thickness ``Δz_i = Δz_{top}\\, r^{i-1}`` (`r = growth`), so the grid is
fine at the surface, where the temperature gradients are, and coarse
below — the Crocus / CLM philosophy. After every step
[`relayer_snowpack!`](@ref)

  1. merges any layer thinner than ``Δz_i/2`` or lighter than `min_mass`
     (kg m⁻²) with its thinner neighbour;
  2. splits any layer thicker than ``2Δz_i`` in two while fewer than
     `max_layers` exist;
  3. merges the thinnest adjacent pair while more than `max_layers` exist.

Merging conserves ice, liquid, thickness and enthalpy exactly; splitting
copies the intensive state. A pack lighter than `min_total_mass` vanishes
(its mass is booked as runoff). `growth` must lie in (1, 2] so a split
layer's halves are never immediately re-merged.
"""
struct SnowLayering
    max_layers::Int
    dz_top::Float64
    growth::Float64
    min_mass::Float64
    min_total_mass::Float64
    function SnowLayering(; max_layers::Integer = 10, dz_top::Real = 0.02,
                          growth::Real = 1.6, min_mass::Real = 0.01,
                          min_total_mass::Real = 1e-6)
        max_layers >= 1 || throw(ArgumentError("SnowLayering: max_layers ≥ 1"))
        dz_top > 0 || throw(ArgumentError("SnowLayering: dz_top > 0"))
        1 < growth <= 2 || throw(ArgumentError("SnowLayering: growth must lie in (1, 2]"))
        new(max_layers, dz_top, growth, min_mass, min_total_mass)
    end
end
_dz_target(l::SnowLayering, i) = l.dz_top * l.growth^(i - 1)

"""
    SnowBudget

Cumulative mass (kg m⁻²) and energy (J m⁻²) exchanged by a pack since
it was built or its budget reset — the bookkeeping behind the
conservation tests:

| field               | meaning                                                   |
|:------------------- |:--------------------------------------------------------- |
| `snowfall`, `rainfall` | precipitation received                                 |
| `sublimation`       | net vapour loss (negative: net deposition)                |
| `runoff`            | liquid water leaving the base                             |
| `melt`              | ice melted                                                |
| `surface_heat`      | ``∫ G\\,dt`` conducted in at the surface                   |
| `bottom_heat`       | heat leaving through the base (incl. melt surplus of a vanished bottom layer) |
| `advected_enthalpy` | enthalpy brought in by precipitation and deposition, minus that carried off by sublimation |
| `runoff_enthalpy`   | enthalpy carried off by runoff                            |

Enthalpy is referred to liquid water at ``T_f``:
``h = w_{ice}[c_{ice}(T - T_f) - L_f] + w_{liq} c_w (T - T_f)``, so

    ΔM = snowfall + rainfall - sublimation - runoff
    ΔH = surface_heat - bottom_heat + advected_enthalpy - runoff_enthalpy
"""
Base.@kwdef mutable struct SnowBudget
    snowfall::Float64 = 0.0
    rainfall::Float64 = 0.0
    sublimation::Float64 = 0.0
    runoff::Float64 = 0.0
    melt::Float64 = 0.0
    surface_heat::Float64 = 0.0
    bottom_heat::Float64 = 0.0
    advected_enthalpy::Float64 = 0.0
    runoff_enthalpy::Float64 = 0.0
end

"""
    LayeredSnowpack <: AbstractSurface

A multi-layer snowpack. Per-layer state, ordered top (1) to bottom:
`dz` (m), `m_ice`, `m_liq` (kg m⁻²), `T` (K), `grain` (optical grain
radius, μm; evolved by #35) and `age` (s). Laws: `compaction`,
`conductivity`, `fresh_density`; hooks: `metamorphism`, `albedo`,
`liquid`; lower boundary `bottom`; `layering`; surface properties
`emissivity`, `z0m`, `z0h`; heat-equation time weight `θ` (1 = backward
Euler, ½ = Crank-Nicolson); constants `consts`; cumulative `budget`.

The skin temperature is the temperature of the top layer (as in CLM),
which is why the default top layer is only 2 cm thick.
"""
mutable struct LayeredSnowpack{C<:AbstractSnowCompaction, K<:AbstractSnowConductivity,
                               F<:AbstractFreshSnowDensity, M<:AbstractSnowMetamorphism,
                               A<:AbstractSnowAlbedo, W<:AbstractSnowLiquidWater,
                               B<:AbstractSnowBottom, SC<:SurfaceConstants} <: AbstractSurface
    dz::Vector{Float64}
    m_ice::Vector{Float64}
    m_liq::Vector{Float64}
    T::Vector{Float64}
    grain::Vector{Float64}
    age::Vector{Float64}
    compaction::C
    conductivity::K
    fresh_density::F
    metamorphism::M
    albedo::A
    liquid::W
    bottom::B
    layering::SnowLayering
    emissivity::Float64
    z0m::Float64
    z0h::Float64
    θ::Float64
    fresh_grain::Float64
    consts::SC
    budget::SnowBudget
end

"""
    LayeredSnowpack(; depth = 0, density = 250, T = 268.15,
                      compaction = Anderson1976Compaction(),
                      conductivity = Jordan1991Conductivity(),
                      fresh_density = Anderson1976FreshSnow(),
                      metamorphism = NoSnowMetamorphism(),
                      albedo = ConstantSnowAlbedo(), liquid = InstantSnowRunoff(),
                      bottom = SnowBottomFlux(0), layering = SnowLayering(),
                      emissivity = material_emissivity(:snow_fresh),
                      z0m = 1e-3, z0h = 1e-4, θ = 1,
                      grain_radius = SurfaceOptics.SNOW_REFERENCE_RADIUS,
                      c = DEFAULT_SURFACE_CONSTANTS)

A pack of uniform `density` (kg m⁻³) and temperature `T` (K), `depth`
metres deep, divided into layers by the `layering` rules. The defaults
are the CLM4 set of laws (Oleson et al. 2010). `depth = 0` is bare
ground waiting for snow.
"""
function LayeredSnowpack(; depth::Real = 0.0, density::Real = 250.0, T::Real = 268.15,
                         compaction::AbstractSnowCompaction = Anderson1976Compaction(),
                         conductivity::AbstractSnowConductivity = Jordan1991Conductivity(),
                         fresh_density::AbstractFreshSnowDensity = Anderson1976FreshSnow(),
                         metamorphism::AbstractSnowMetamorphism = NoSnowMetamorphism(),
                         albedo::AbstractSnowAlbedo = ConstantSnowAlbedo(),
                         liquid::AbstractSnowLiquidWater = InstantSnowRunoff(),
                         bottom::AbstractSnowBottom = SnowBottomFlux(0.0),
                         layering::SnowLayering = SnowLayering(),
                         emissivity::Real = material_emissivity(:snow_fresh),
                         z0m::Real = 1e-3, z0h::Real = 1e-4, θ::Real = 1.0,
                         grain_radius::Real = SNOW_REFERENCE_RADIUS,
                         c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    depth >= 0 || throw(ArgumentError("LayeredSnowpack: depth ≥ 0"))
    0 < density <= c.ρ_ice || throw(ArgumentError("LayeredSnowpack: 0 < density ≤ ρ_ice"))
    T <= _Tfreeze(c) || throw(ArgumentError("LayeredSnowpack: T must not exceed the melting point"))
    0.5 <= θ <= 1 || throw(ArgumentError("LayeredSnowpack: θ must lie in [1/2, 1]"))
    e = Float64[]
    p = LayeredSnowpack(copy(e), copy(e), copy(e), copy(e), copy(e), copy(e),
                        compaction, conductivity, fresh_density, metamorphism,
                        albedo, liquid, bottom, layering, Float64(emissivity),
                        Float64(z0m), Float64(z0h), Float64(θ), Float64(grain_radius),
                        c, SnowBudget())
    if depth > 0
        _push_layer!(p, float(depth), density * depth, 0.0, float(T), p.fresh_grain, 0.0)
        for _ in 1:100                                   # split to the layering rules
            n0 = snow_nlayers(p)
            relayer_snowpack!(p)
            snow_nlayers(p) == n0 && break
        end
    end
    return p
end

function _push_layer!(p, dz, mi, ml, T, g, a; at::Integer = snow_nlayers(p) + 1)
    insert!(p.dz, at, dz); insert!(p.m_ice, at, mi); insert!(p.m_liq, at, ml)
    insert!(p.T, at, T); insert!(p.grain, at, g); insert!(p.age, at, a)
    return p
end
function _delete_layer!(p, i)
    for v in (p.dz, p.m_ice, p.m_liq, p.T, p.grain, p.age)
        deleteat!(v, i)
    end
    return p
end

# ---- diagnostics ---------------------------------------------------------

"Number of snow layers."
snow_nlayers(p::LayeredSnowpack) = length(p.dz)
"Snow depth, m."
snow_depth(p::LayeredSnowpack) = sum(p.dz; init = 0.0)
"Snow water equivalent (ice + liquid), kg m⁻² (= mm of water)."
snow_water_equivalent(p::LayeredSnowpack) = sum(p.m_ice; init = 0.0) + sum(p.m_liq; init = 0.0)
"Bulk density of the whole pack, kg m⁻³ (`NaN` without snow)."
snow_bulk_density(p::LayeredSnowpack) =
    snow_nlayers(p) == 0 ? NaN : snow_water_equivalent(p) / snow_depth(p)
"Bulk density of each layer, kg m⁻³."
snow_layer_density(p::LayeredSnowpack) = (p.m_ice .+ p.m_liq) ./ p.dz
"Depth of each layer midpoint below the snow surface, m."
snow_layer_depths(p::LayeredSnowpack) = cumsum(p.dz) .- p.dz ./ 2

"""
    snow_load(p) -> Vector (kg m⁻²)

Overburden on each layer: the mass of all layers above plus half its own
(Oleson et al. 2010, Eq. 7.44; Vionnet et al. 2012 take half the weight
of the top layer likewise).
"""
function snow_load(p::LayeredSnowpack)
    m = p.m_ice .+ p.m_liq
    return cumsum(m) .- m ./ 2
end

_heat_capacity(p, i) = p.m_ice[i] * cp_ice(p.consts) + p.m_liq[i] * cp_water(p.consts)

"""
    snowpack_enthalpy(p) -> J m⁻²

Enthalpy relative to liquid water at the melting point,
``Σ_i w_{ice,i}[c_{ice}(T_i - T_f) - L_f] + w_{liq,i} c_w (T_i - T_f)``.
Always negative for a pack that contains ice.
"""
function snowpack_enthalpy(p::LayeredSnowpack)
    c = p.consts
    Tf, Lf = _Tfreeze(c), LH_f0(c)
    h = 0.0
    for i in 1:snow_nlayers(p)
        h += _heat_capacity(p, i) * (p.T[i] - Tf) - p.m_ice[i] * Lf
    end
    return h
end
_ice_enthalpy(p, m, T) = m * (cp_ice(p.consts) * (T - _Tfreeze(p.consts)) - LH_f0(p.consts))

# =========================================================================
# 6. Processes
# =========================================================================

"""
    snow_compaction!(p, dt; max_fraction = 0.01) -> p

Settle every layer for `dt` seconds under the pack's compaction law at
fixed mass, temperature and load. ``\\ln Δz`` is integrated with the
explicit midpoint rule in sub-steps short enough that no layer changes
thickness by more than `max_fraction` per sub-step, and no layer is
compacted beyond the density of ice (``Δz ≥ w_{ice}/ρ_{ice} + w_{liq}/ρ_w``).
"""
function snow_compaction!(p::LayeredSnowpack, dt::Real; max_fraction::Real = 0.01)
    n = snow_nlayers(p)
    n == 0 && return p
    c = p.consts
    loads = snow_load(p)
    for i in 1:n
        mi, ml = p.m_ice[i], p.m_liq[i]
        m = mi + ml
        (m > 0 && p.dz[i] > 0) || continue
        f2 = viscosity_grain_factor(p.metamorphism, p, i)
        Ti, Li = p.T[i], loads[i]
        rate(dz) = snow_compaction_rate(p.compaction,
                        (ρ = m / dz, ρ_ice = mi / dz, ρ_liq = ml / dz, T = Ti,
                         load = Li, f2 = f2); c = c)
        lnfloor = log(mi / c.ρ_ice + ml / c.ρ_water)
        lnz = log(p.dz[i])
        t = 0.0
        while t < dt && lnz > lnfloor
            r = rate(exp(lnz))
            r == 0 && break
            h = min(dt - t, max_fraction / abs(r))
            lnz = max(lnz + h * rate(exp(lnz + h * r / 2)), lnfloor)
            t += h
        end
        p.dz[i] = exp(lnz)
    end
    return p
end

"""
    snow_heat_conduction!(p, dt; top_flux = 0, dtop_flux_dT = 0, T_top = nothing)
        -> (F_top, F_bottom)

One θ-weighted step of heat conduction through the layers,

    C_i dT_i/dt = F_{i-1/2} - F_{i+1/2},   F_{i+1/2} = (T_i - T_{i+1}) / (Δz_i/2k_i + Δz_{i+1}/2k_{i+1})

with ``C_i = w_{ice} c_{ice} + w_{liq} c_w`` and ``k_i`` from the pack's
conductivity law. Upper boundary: either a flux into the top layer
linearised about the current top temperature,
``G = G_0 + (∂G/∂T)(T_1^* - T_1^n)`` (`top_flux`, `dtop_flux_dT`), or,
with `T_top`, a fixed surface temperature half a layer above the top
node. Lower boundary: the pack's `bottom`. ``T^* = θT^{n+1} + (1-θ)T^n``,
so θ = 1 is backward Euler and θ = ½ Crank-Nicolson.

Returns the applied flux into the top of the pack and out of its base
(W m⁻²); ``Σ C_i ΔT_i = (F_{top} - F_{bottom})\\,dt`` holds to round-off.
Temperatures may leave the pack above the melting point:
[`snow_phase_change!`](@ref) turns the excess into melt.
"""
function snow_heat_conduction!(p::LayeredSnowpack, dt::Real; top_flux::Real = 0.0,
                               dtop_flux_dT::Real = 0.0, T_top = nothing)
    n = snow_nlayers(p)
    n == 0 && return (0.0, 0.0)
    c, θ = p.consts, p.θ
    T = p.T
    k = [snow_thermal_conductivity(p.conductivity, (p.m_ice[i] + p.m_liq[i]) / p.dz[i]; c = c)
         for i in 1:n]
    K = [1 / (p.dz[i] / (2k[i]) + p.dz[i+1] / (2k[i+1])) for i in 1:n-1]
    d  = [_heat_capacity(p, i) / dt for i in 1:n]
    dl = zeros(n - 1); du = zeros(n - 1); rhs = zeros(n)
    for i in 1:n-1
        F = K[i] * (T[i] - T[i+1])
        rhs[i] -= F;          rhs[i+1] += F
        d[i]   += θ * K[i];   d[i+1]   += θ * K[i]
        du[i]   = -θ * K[i];  dl[i]     = -θ * K[i]
    end
    if T_top === nothing
        rhs[1] += top_flux
        d[1]   -= θ * dtop_flux_dT
        Kt = 0.0
    else
        Kt = 2k[1] / p.dz[1]
        rhs[1] += Kt * (T_top - T[1])
        d[1]   += θ * Kt
    end
    F0b, Kb = _bottom_terms(p.bottom, k[n], p.dz[n], T[n])
    rhs[n] -= F0b
    d[n]   += θ * Kb
    ΔT = Tridiagonal(dl, d, du) \ rhs
    F_top = T_top === nothing ? top_flux + dtop_flux_dT * θ * ΔT[1] :
                                Kt * (T_top - T[1] - θ * ΔT[1])
    F_bot = F0b + Kb * θ * ΔT[n]
    T .+= ΔT
    p.budget.surface_heat += F_top * dt
    p.budget.bottom_heat  += F_bot * dt
    return (F_top, F_bot)
end

"""
    snow_phase_change!(p) -> melted mass (kg m⁻²)

Melt ice in every layer warmer than ``T_f``: the sensible excess
``C_i (T_i - T_f)`` melts ``Δw = \\min(C_i(T_i - T_f)/L_f, w_{ice})`` and
the layer is returned to ``T_f``. The layer thins in proportion to the
ice lost — its density is unchanged, which is CLM's compaction-by-melt
term ``C_{R3}`` applied exactly. Energy left over when a layer melts out
passes to the layer below, and from the bottom layer into the substrate.
Refreezing of retained water is the liquid-water scheme's (#36).
"""
function snow_phase_change!(p::LayeredSnowpack)
    c = p.consts
    Tf, Lf = _Tfreeze(c), LH_f0(c)
    carry, melted = 0.0, 0.0
    for i in 1:snow_nlayers(p)
        C = _heat_capacity(p, i)
        if carry > 0
            p.T[i] += carry / C
            carry = 0.0
        end
        p.T[i] > Tf || continue
        E = C * (p.T[i] - Tf)
        dm = min(E / Lf, p.m_ice[i])
        p.m_ice[i] > 0 && (p.dz[i] *= 1 - dm / p.m_ice[i])
        p.m_ice[i] -= dm
        p.m_liq[i] += dm
        p.T[i] = Tf
        carry = E - dm * Lf
        melted += dm
    end
    carry > 0 && (p.budget.bottom_heat += carry)
    p.budget.melt += melted
    return melted
end

"""
    snow_sublimation!(p, mass) -> mass actually exchanged (kg m⁻²)

Remove `mass` of ice by sublimation from the top of the pack downward
(layers thin at constant density), or, for `mass < 0`, deposit rime/hoar
on the top layer at its temperature and density. The enthalpy of the
exchanged ice is booked in the budget.
"""
function snow_sublimation!(p::LayeredSnowpack, mass::Real)
    n = snow_nlayers(p)
    (n == 0 || mass == 0) && return 0.0
    b = p.budget
    if mass < 0
        m = -mass
        ρ1 = (p.m_ice[1] + p.m_liq[1]) / p.dz[1]
        p.dz[1] += m / ρ1
        p.m_ice[1] += m
        b.advected_enthalpy += _ice_enthalpy(p, m, p.T[1])
        b.sublimation -= m
        return float(mass)
    end
    left = float(mass)
    for i in 1:n
        left <= 0 && break
        take = min(left, p.m_ice[i])
        take > 0 || continue
        p.dz[i] *= 1 - take / p.m_ice[i]
        p.m_ice[i] -= take
        b.advected_enthalpy -= _ice_enthalpy(p, take, p.T[i])
        left -= take
    end
    removed = mass - left
    b.sublimation += removed
    return removed
end

"""
    add_snowfall!(p, mass, T_air, ρ_new) -> p

Add `mass` (kg m⁻²) of snow of density `ρ_new` falling at `T_air`. The
new snow has temperature ``\\min(T_{air}, T_f)`` and the fresh grain
radius; it is mixed into the top layer (enthalpy-conserving) or, on bare
ground, starts the pack. Layering rules later split it off as it grows.
"""
function add_snowfall!(p::LayeredSnowpack, mass::Real, T_air::Real, ρ_new::Real)
    mass > 0 || return p
    Ts = min(float(T_air), _Tfreeze(p.consts))
    b = p.budget
    b.snowfall += mass
    b.advected_enthalpy += _ice_enthalpy(p, mass, Ts)
    if snow_nlayers(p) == 0
        _push_layer!(p, mass / ρ_new, float(mass), 0.0, Ts, p.fresh_grain, 0.0)
        return p
    end
    C0 = _heat_capacity(p, 1)
    Cn = mass * cp_ice(p.consts)
    m0 = p.m_ice[1] + p.m_liq[1]
    p.T[1] = (C0 * p.T[1] + Cn * Ts) / (C0 + Cn)
    p.grain[1] = (m0 * p.grain[1] + mass * p.fresh_grain) / (m0 + mass)
    p.age[1] = m0 * p.age[1] / (m0 + mass)
    p.m_ice[1] += mass
    p.dz[1] += mass / ρ_new
    return p
end

"""
    add_rainfall!(p, mass, T_air) -> p

Add `mass` (kg m⁻²) of rain at ``\\max(T_{air}, T_f)`` to the top layer as
liquid water; its sensible heat is mixed in (enthalpy-conserving). On
bare ground the rain runs straight off.
"""
function add_rainfall!(p::LayeredSnowpack, mass::Real, T_air::Real)
    mass > 0 || return p
    c = p.consts
    Tf = _Tfreeze(c)
    h = mass * cp_water(c) * (max(float(T_air), Tf) - Tf)
    b = p.budget
    b.rainfall += mass
    b.advected_enthalpy += h
    if snow_nlayers(p) == 0
        b.runoff += mass
        b.runoff_enthalpy += h
        return p
    end
    H0 = _heat_capacity(p, 1) * (p.T[1] - Tf)
    p.m_liq[1] += mass
    p.T[1] = Tf + (H0 + h) / _heat_capacity(p, 1)
    return p
end

# The liquid-water hook, with the enthalpy the runoff removes booked.
function _drain!(p::LayeredSnowpack, dt)
    H0 = snowpack_enthalpy(p)
    r = snow_liquid_water!(p.liquid, p, dt)
    p.budget.runoff += r
    p.budget.runoff_enthalpy += H0 - snowpack_enthalpy(p)
    return r
end

function _merge_layers!(p, i, j)          # j into i, then delete j
    Ci, Cj = _heat_capacity(p, i), _heat_capacity(p, j)
    mi, mj = p.m_ice[i] + p.m_liq[i], p.m_ice[j] + p.m_liq[j]
    # (bounded by the inputs, so two layers at T_f never round to above it)
    Ci + Cj > 0 && (p.T[i] = clamp((Ci * p.T[i] + Cj * p.T[j]) / (Ci + Cj),
                                   min(p.T[i], p.T[j]), max(p.T[i], p.T[j])))
    if mi + mj > 0
        p.grain[i] = (mi * p.grain[i] + mj * p.grain[j]) / (mi + mj)
        p.age[i]   = (mi * p.age[i] + mj * p.age[j]) / (mi + mj)
    end
    p.dz[i] += p.dz[j]; p.m_ice[i] += p.m_ice[j]; p.m_liq[i] += p.m_liq[j]
    _delete_layer!(p, j)
    return p
end

"""
    relayer_snowpack!(p) -> p

Apply the [`SnowLayering`](@ref) rules once: merge too-thin or
too-light layers, split too-thick ones, enforce `max_layers`, and remove
a pack lighter than `min_total_mass`. Conserves ice, liquid, depth and
enthalpy exactly.
"""
function relayer_snowpack!(p::LayeredSnowpack)
    l = p.layering
    if snow_nlayers(p) > 0 && snow_water_equivalent(p) < l.min_total_mass
        p.budget.runoff += snow_water_equivalent(p)
        p.budget.runoff_enthalpy += snowpack_enthalpy(p)
        for v in (p.dz, p.m_ice, p.m_liq, p.T, p.grain, p.age)
            empty!(v)
        end
        return p
    end
    # 1. merge thin / light layers with the thinner neighbour
    i = 1
    while i <= snow_nlayers(p) && snow_nlayers(p) > 1
        n = snow_nlayers(p)
        if p.dz[i] < _dz_target(l, i) / 2 || p.m_ice[i] + p.m_liq[i] < l.min_mass
            j = i == 1 ? 2 : i == n ? n - 1 : (p.dz[i-1] <= p.dz[i+1] ? i - 1 : i + 1)
            _merge_layers!(p, min(i, j), max(i, j))
            i = max(min(i, j), 1)
        else
            i += 1
        end
    end
    # 2. split thick layers while room remains
    i = 1
    while i <= snow_nlayers(p) && snow_nlayers(p) < l.max_layers
        if p.dz[i] > 2 * _dz_target(l, i)
            p.dz[i] /= 2; p.m_ice[i] /= 2; p.m_liq[i] /= 2
            _push_layer!(p, p.dz[i], p.m_ice[i], p.m_liq[i], p.T[i], p.grain[i], p.age[i];
                         at = i + 1)
            i += 2
        else
            i += 1
        end
    end
    # 3. too many layers: merge the thinnest adjacent pair
    while snow_nlayers(p) > l.max_layers
        n = snow_nlayers(p)
        k = argmin([p.dz[i] + p.dz[i+1] for i in 1:n-1])
        _merge_layers!(p, k, k + 1)
    end
    return p
end

# =========================================================================
# 7. The AbstractSurface implementation
# =========================================================================

surface_temperature(p::LayeredSnowpack) =
    snow_nlayers(p) > 0 ? p.T[1] : _bare_temperature(p.bottom, p.consts)
roughness_lengths(p::LayeredSnowpack) = (z0m = p.z0m, z0h = p.z0h, z0q = p.z0h)
surface_albedo(p::LayeredSnowpack) = snowpack_albedo(p.albedo, p)
surface_emissivity(p::LayeredSnowpack) = p.emissivity
"A snow surface is saturated: β = 1."
moisture_availability(::LayeredSnowpack) = 1.0

"""
    surface_specific_humidity(p::LayeredSnowpack, forcing; c) -> q_s

Saturation specific humidity over ice at the skin temperature (over
liquid water at the melting point).
"""
function surface_specific_humidity(p::LayeredSnowpack, f::SurfaceForcing;
                                   c::SurfaceConstants = p.consts)
    Ts = surface_temperature(p)
    return saturation_specific_humidity(Ts, f.p, c.thermo;
                                        over = Ts < _Tfreeze(c) ? :ice : :liquid)
end

"Latent heat of sublimation at the skin temperature (`MoistThermodynamics.latent_heat_sublim`)."
surface_latent_heat(p::LayeredSnowpack; c::SurfaceConstants = p.consts) =
    latent_heat_sublim(surface_temperature(p), c.thermo)

"""
    step_surface!(p::LayeredSnowpack, forcing, dt; δT = 0.01, kwargs...) -> SurfaceFluxes

Advance the snowpack by `dt` seconds:

 1. surface fluxes from `SurfaceExchange.turbulent_fluxes` at the current
    skin temperature and at ``T_1 + δT``, giving ``R_n, H, LE, E, τ`` and
    their derivatives with respect to the skin temperature;
 2. heat conduction with the surface energy balance
    ``G = R_n - H - LE`` linearised in the top-layer temperature
    (implicit, so the stiff radiative/turbulent coupling of a 2 cm top
    layer is stable at any step);
 3. the returned fluxes are those actually applied: each is updated
    linearly to the time-weighted skin temperature, so
    ``R_n = H + LE + G`` holds exactly and `G` is what entered the pack;
 4. sublimation/deposition of ``E\\,dt``; rain and snowfall (density from
    `fresh_density`, at `forcing.T_air` and `forcing.U`);
 5. melt, the liquid-water hook, compaction, the metamorphism hook, and
    relayering.

Without snow, the fluxes of a surface at the bare-ground temperature are
returned (not applied) and precipitation is processed.
`kwargs` go to `turbulent_fluxes`.
"""
function step_surface!(p::LayeredSnowpack, f::SurfaceForcing, dt::Real;
                       δT::Real = 0.01, kwargs...)
    c = p.consts
    if snow_nlayers(p) == 0
        fl = turbulent_fluxes(p, f; c = c, kwargs...)
    else
        T1 = p.T[1]
        f0 = turbulent_fluxes(p, f; c = c, kwargs...)
        p.T[1] = T1 + δT
        f1 = turbulent_fluxes(p, f; c = c, kwargs...)
        p.T[1] = T1
        dRn, dH, dLE = (f1.Rn - f0.Rn) / δT, (f1.H - f0.H) / δT, (f1.LE - f0.LE) / δT
        dτ, dE = (f1.τ - f0.τ) / δT, (f1.E - f0.E) / δT
        G0 = f0.Rn - f0.H - f0.LE
        F_top, _ = snow_heat_conduction!(p, dt; top_flux = G0, dtop_flux_dT = dRn - dH - dLE)
        ΔTs = p.θ * (p.T[1] - T1)
        fl = SurfaceFluxes(f0.Rn + dRn * ΔTs, f0.H + dH * ΔTs, f0.LE + dLE * ΔTs,
                           F_top, f0.τ + dτ * ΔTs, f0.E + dE * ΔTs)
        snow_sublimation!(p, fl.E * dt)
    end
    add_rainfall!(p, f.rain * dt, f.T_air)
    add_snowfall!(p, f.snow * dt, f.T_air, fresh_snow_density(p.fresh_density, f.T_air, f.U; c = c))
    snow_phase_change!(p)
    _drain!(p, dt)
    snow_compaction!(p, dt)
    snow_metamorphism!(p.metamorphism, p, dt)
    relayer_snowpack!(p)
    return fl
end

Base.show(io::IO, p::LayeredSnowpack) =
    print(io, "LayeredSnowpack(", snow_nlayers(p), " layers, depth = ",
          round(snow_depth(p); digits = 3), " m, SWE = ",
          round(snow_water_equivalent(p); digits = 1), " kg m⁻², ρ̄ = ",
          round(snow_bulk_density(p); digits = 1), " kg m⁻³, T_s = ",
          round(surface_temperature(p); digits = 2), " K)")

end # module
