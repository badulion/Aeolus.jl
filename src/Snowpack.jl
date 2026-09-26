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

Grain and albedo ageing (#35) plug in through two hooks:

  * [`AbstractSnowMetamorphism`](@ref) (`snow_metamorphism!`,
    `viscosity_grain_factor`, `fresh_grain_radius`,
    `refrozen_grain_radius`) — [`Flanner2006Metamorphism`](@ref) evolves
    the optical grain radius with the CLM/SNICAR dry-ageing fits and
    Brun (1989) wet growth; [`ParabolicGrainGrowth`](@ref) is the
    `SurfaceOptics` ageing law; the default `NoSnowMetamorphism` keeps
    grains fixed.
  * [`AbstractSnowAlbedo`](@ref) (`snowpack_albedo`) —
    [`SnowOpticsAlbedo`](@ref) delegates to the Wiscombe-Warren
    `SurfaceOptics.SnowSurface` at the prognostic grain radius; the
    default is a constant.

One hook is left thin for the phase that owns it:

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
  * Flanner, M. G. & Zender, C. S. (2006). Linking snowpack microphysics
    and albedo evolution. *J. Geophys. Res.*, 111, D12208.
  * Brun, E. (1989). Investigation on wet-snow metamorphism in respect of
    liquid-water content. *Ann. Glaciol.*, 13, 22-26.
  * Lawrence, D. et al. (2018). *Technical description of version 5.0 of
    the Community Land Model (CLM)*, NCAR, Sect. 2.3.2.3.
  * Wiscombe, W. J. & Warren, S. G. (1980). A model for the spectral
    albedo of snow. I: Pure snow. *J. Atmos. Sci.*, 37, 2712-2733.
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
using ..SurfaceOptics: material_emissivity, SNOW_REFERENCE_RADIUS, SnowSurface,
                       snow_grain_radius, broadband_albedo
using ..RadiationCore: SpectralGrid, shortwave_bands
using ..NetCDFIO: nc_open, nc_read

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
       add_snowfall!, add_rainfall!, snow_sublimation!, relayer_snowpack!,
       # metamorphism and albedo ageing (#35)
       fresh_grain_radius, refrozen_grain_radius, mix_refrozen_grain!,
       SnicarAgingTable, snicar_aging_table, snicar_aging_parameters,
       snicar_grain_radius, flanner_grain_radius,
       Flanner2006Metamorphism, ParabolicGrainGrowth,
       snow_dry_grain_growth_rate, snow_wet_grain_growth_rate,
       snow_temperature_gradient, snow_specific_surface_area, snow_optical_radius,
       SnowOpticsAlbedo, snowpack_optics, snowpack_optical_radius

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
    settling (Vionnet et al. 2012, Eq. 9). Default 1;
  * optionally `fresh_grain_radius(scheme, pack, T_air)` and
    `refrozen_grain_radius(scheme)`.

Implementations: [`NoSnowMetamorphism`](@ref),
[`Flanner2006Metamorphism`](@ref), [`ParabolicGrainGrowth`](@ref).
Neither #35 scheme tracks grain shape, so both keep ``f_2 = 1``.
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
    fresh_grain_radius(scheme, pack, T_air) -> μm

Optical grain radius given to snow falling at air temperature `T_air`
(K). The default is the pack's constant `fresh_grain`.
"""
fresh_grain_radius(::AbstractSnowMetamorphism, pack, T_air::Real) = pack.fresh_grain

"""
    AbstractSnowAlbedo

Broadband albedo of the pack — the hook for albedo ageing (#35).
Implementers extend `snowpack_albedo(scheme, pack)`. Spectral snow
optics already live in `SurfaceOptics.SnowSurface` (Wiscombe-Warren
grain-size and soot dependence); a grain-aware scheme should delegate to
it rather than restate it, as [`SnowOpticsAlbedo`](@ref) does.
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
new snow has temperature ``\\min(T_{air}, T_f)`` and the grain radius
[`fresh_grain_radius`](@ref) of the pack's metamorphism scheme; it is
mixed into the top layer (enthalpy-conserving; grain radius and age
mass-weighted) or, on bare ground, starts the pack. Layering rules later split it off as it grows.
"""
function add_snowfall!(p::LayeredSnowpack, mass::Real, T_air::Real, ρ_new::Real)
    mass > 0 || return p
    Ts = min(float(T_air), _Tfreeze(p.consts))
    b = p.budget
    b.snowfall += mass
    b.advected_enthalpy += _ice_enthalpy(p, mass, Ts)
    rg = fresh_grain_radius(p.metamorphism, p, T_air)
    if snow_nlayers(p) == 0
        _push_layer!(p, mass / ρ_new, float(mass), 0.0, Ts, rg, 0.0)
        return p
    end
    C0 = _heat_capacity(p, 1)
    Cn = mass * cp_ice(p.consts)
    m0 = p.m_ice[1] + p.m_liq[1]
    p.T[1] = (C0 * p.T[1] + Cn * Ts) / (C0 + Cn)
    p.grain[1] = (m0 * p.grain[1] + mass * rg) / (m0 + mass)
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
 4. sublimation/deposition of ``E\\,dt`` (if the pack sublimates away
    within the step, `E` and `LE` are reduced to the mass actually
    removed and the unspent latent heat is added to `G` and passed to the
    substrate, so ``R_n = H + LE + G`` still holds); rain and snowfall (density from
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
        E_req = fl.E * dt
        E_got = snow_sublimation!(p, E_req)
        if E_req > 0 && E_got < E_req
            # The pack sublimated away within the step. Only the vapour it
            # actually supplied goes to the atmosphere; the latent heat
            # that was not spent stays in G and, with no ice left to
            # absorb it, passes into the substrate (as a melted-out
            # bottom layer's surplus does in `snow_phase_change!`).
            dE = (E_req - E_got) / dt
            dLE = fl.LE * dE / fl.E
            fl = SurfaceFluxes(fl.Rn, fl.H, fl.LE - dLE, fl.G + dLE, fl.τ, fl.E - dE)
            p.budget.surface_heat += dLE * dt
            p.budget.bottom_heat  += dLE * dt
        end
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

# =========================================================================
# 8. Grain metamorphism and albedo ageing (#35)
# =========================================================================
#
# The optical grain radius r (μm) — the radius of ice spheres with the
# same specific surface area, r = 3/(ρ_ice SSA) — is the one microphysical
# number the snow albedo needs. Fresh snow (SSA ≈ 60 m² kg⁻¹, r ≈ 55 μm)
# coarsens by vapour transport from high- to low-curvature ice (dry
# metamorphism, fastest in warm, low-density snow under a strong
# temperature gradient) and far faster when liquid water is present (wet
# metamorphism). The albedo follows: ice absorbs weakly in the visible and
# strongly beyond ~1 μm, so a larger r lengthens photon paths inside the
# grains and darkens the near-infrared.
#
# The albedo itself is NOT computed here. `SurfaceOptics.SnowSurface` owns
# the Wiscombe-Warren spectral snow albedo (grain size + soot); the pack
# hands it the prognostic grain radius and asks for the broadband value.

"""
    snow_specific_surface_area(r; c = DEFAULT_SURFACE_CONSTANTS) -> m² kg⁻¹

Specific surface area of snow with optical (effective) grain radius `r`
(μm), ``SSA = 3/(ρ_{ice} r)`` (CLM5 Technical Note, Sect. 2.3.2.3;
Carmagnola et al. 2014, Eq. 1 with ``d_{opt} = 2r``). With the contract's
``ρ_{ice} = 917`` kg m⁻³, 54.5 μm is 60 m² kg⁻¹.
"""
snow_specific_surface_area(r::Real; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    3 / (c.ρ_ice * r * 1e-6)

"""
    snow_optical_radius(ssa; c = DEFAULT_SURFACE_CONSTANTS) -> μm

Inverse of [`snow_specific_surface_area`](@ref): ``r = 3/(ρ_{ice} SSA)``.
"""
snow_optical_radius(ssa::Real; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    3e6 / (c.ρ_ice * ssa)

# ---- refreezing hook (used by the liquid-water scheme, #36) -------------

"""
    refrozen_grain_radius(scheme) -> μm or `nothing`

Grain radius of liquid water that refreezes in the pack. Refrozen melt
forms large ice clumps that darken the snow; CLM uses 1000 μm (CLM5
Technical Note, Sect. 2.3.2.3). `nothing` (the default) leaves grains
unchanged.
"""
refrozen_grain_radius(::AbstractSnowMetamorphism) = nothing

"""
    mix_refrozen_grain!(pack, i, m_refrozen) -> pack

Mass-weight the grain radius of layer `i` towards
[`refrozen_grain_radius`](@ref) of the pack's scheme after `m_refrozen`
(kg m⁻²) of its ice has formed by refreezing — CLM's
``r = r_{old} f_{old} + r_{rfz} f_{rfz}`` (CLM5 Technical Note, Eq. 2.3.74).
Call it *after* the refrozen mass has been added to `m_ice[i]`. The
liquid-water scheme (#36) owns refreezing; this is its entry point into
the grain state.
"""
function mix_refrozen_grain!(p::LayeredSnowpack, i::Integer, m_refrozen::Real)
    r_rfz = refrozen_grain_radius(p.metamorphism)
    (r_rfz === nothing || m_refrozen <= 0) && return p
    m = p.m_ice[i] + p.m_liq[i]
    m > 0 || return p
    f = min(m_refrozen / m, 1.0)
    p.grain[i] = (1 - f) * p.grain[i] + f * r_rfz
    return p
end

# ---- temperature gradient -------------------------------------------------

"""
    snow_temperature_gradient(p) -> Vector (K m⁻¹)

Magnitude of the temperature gradient across each layer, from the
interface temperatures (thickness-weighted means of neighbouring layer
temperatures) — CLM5 Technical Note Eq. (2.3.76) (CTSM
`SnowAge_grain`). As in CLM, the upper interface of the top layer is
taken at the top-layer temperature. The lower interface of the bottom
layer is the ground temperature for [`SnowGroundTemperature`](@ref) and,
for [`SnowBottomFlux`](@ref), the temperature that carries the
prescribed flux across the lower half of the layer.
"""
function snow_temperature_gradient(p::LayeredSnowpack)
    n = snow_nlayers(p)
    g = zeros(n)
    n == 0 && return g
    T, dz = p.T, p.dz
    for i in 1:n
        Tu = i == 1 ? T[1] : (T[i-1] * dz[i] + T[i] * dz[i-1]) / (dz[i] + dz[i-1])
        if i < n
            Tl = (T[i+1] * dz[i] + T[i] * dz[i+1]) / (dz[i] + dz[i+1])
        else
            Tl = _bottom_interface_temperature(p.bottom, p, n)
        end
        g[i] = abs(Tu - Tl) / dz[i]
    end
    return g
end

_bottom_interface_temperature(b::SnowGroundTemperature, p, n) = b.T
function _bottom_interface_temperature(b::SnowBottomFlux, p, n)
    k = snow_thermal_conductivity(p.conductivity, (p.m_ice[n] + p.m_liq[n]) / p.dz[n];
                                  c = p.consts)
    return p.T[n] - b.F * p.dz[n] / (2k)
end

# ---- the SNICAR dry-ageing table (Flanner & Zender 2006) ---------------

"""
    SnicarAgingTable

Best-fit parameters of the Flanner & Zender (2006) dry-snow ageing model,
on a grid of temperature `T` (K, 223.15-273.15, 11 points), temperature
gradient `dTdz` (K m⁻¹, 0-300, 31 points) and density `ρ` (kg m⁻³,
50-400, 8 points). Arrays are indexed `[ρ, dTdz, T]`:

| field   | meaning                                   | unit    |
|:------- |:----------------------------------------- |:------- |
| `drdt0` | initial growth rate ``(dr/dt)_0``         | μm h⁻¹  |
| `τ`     | growth-decay length scale ``η``           | μm      |
| `κ`     | growth-decay exponent                     | –       |
| `r0`    | fresh radius the fits start from (54.5)   | μm      |

The data are the file `snicar_drdt_bst_fit_60_c070416.nc` (M. Flanner,
16 Apr 2007; "r_e0 = 54.5 um") distributed with CESM/CLM as
`lnd/clm2/snicardata/` input data, shipped unchanged in `src/data/` and
read with the in-package `NetCDFIO` reader.
"""
struct SnicarAgingTable
    T::Vector{Float64}
    dTdz::Vector{Float64}
    ρ::Vector{Float64}
    drdt0::Array{Float64,3}
    τ::Array{Float64,3}
    κ::Array{Float64,3}
    r0::Float64
    description::String
end

const SNICAR_AGING_FILE = joinpath(@__DIR__, "data", "snicar_drdt_bst_fit_60_c070416.nc")
const _SNICAR_TABLE = Ref{Union{Nothing,SnicarAgingTable}}(nothing)

"""
    snicar_aging_table(path = SNICAR_AGING_FILE) -> SnicarAgingTable

Load the Flanner & Zender (2006) ageing table. The default file is read
once and cached.
"""
function snicar_aging_table(path::AbstractString = SNICAR_AGING_FILE)
    if path == SNICAR_AGING_FILE && _SNICAR_TABLE[] !== nothing
        return _SNICAR_TABLE[]
    end
    f = nc_open(path)
    desc = string(get(f.gattrs, "description", ""))
    tab = SnicarAgingTable(nc_read(f, "T"), nc_read(f, "dTdz"), nc_read(f, "sno_dns"),
                           nc_read(f, "drdsdt0"), nc_read(f, "tau"), nc_read(f, "kappa"),
                           54.5, desc)
    path == SNICAR_AGING_FILE && (_SNICAR_TABLE[] = tab)
    return tab
end

# Bracketing index and weight on a uniform-or-not ascending axis, clamped.
function _bracket(ax::Vector{Float64}, x::Real)
    n = length(ax)
    x <= ax[1] && return (1, 1, 0.0)
    x >= ax[n] && return (n, n, 0.0)
    j = searchsortedlast(ax, x)
    j >= n && return (n, n, 0.0)
    return (j, j + 1, (x - ax[j]) / (ax[j+1] - ax[j]))
end
_nearest(ax::Vector{Float64}, x::Real) = argmin(abs.(ax .- x))

"""
    snicar_aging_parameters(tab, T, dTdz, ρ) -> (drdt0 [μm h⁻¹], τ [μm], κ)

Ageing parameters of the grid point nearest to temperature `T` (K),
temperature-gradient magnitude `dTdz` (K m⁻¹) and density `ρ`
(kg m⁻³), clamped to the table's domain — CLM's look-up (CTSM
`SnowAge_grain`).

The three parameters are fitted jointly, cell by cell, and trade off
against each other (a small `τ` with a large `κ` can describe nearly the
same curve as the reverse), so they must **not** be interpolated
separately: between the 300 and 350 kg m⁻³ nodes at −25 °C, 50 K m⁻¹,
interpolated parameters grow a 88 μm grain to 484 μm in 37 days while
both neighbouring nodes give 181 and 123 μm. Interpolation, where wanted,
is applied to growth outcomes instead ([`snicar_grain_radius`](@ref)).
"""
function snicar_aging_parameters(tab::SnicarAgingTable, T::Real, dTdz::Real, ρ::Real)
    i, j, k = _nearest(tab.ρ, ρ), _nearest(tab.dTdz, abs(dTdz)), _nearest(tab.T, T)
    return (drdt0 = tab.drdt0[i, j, k], τ = tab.τ[i, j, k], κ = tab.κ[i, j, k])
end

# The (up to) eight table nodes around (T, G, ρ) and their trilinear weights.
function _snicar_corners(tab::SnicarAgingTable, T::Real, G::Real, ρ::Real)
    i0, i1, wi = _bracket(tab.ρ, ρ)
    j0, j1, wj = _bracket(tab.dTdz, abs(G))
    k0, k1, wk = _bracket(tab.T, T)
    return ((i0, j0, k0, (1 - wi) * (1 - wj) * (1 - wk)), (i1, j0, k0, wi * (1 - wj) * (1 - wk)),
            (i0, j1, k0, (1 - wi) * wj * (1 - wk)),       (i1, j1, k0, wi * wj * (1 - wk)),
            (i0, j0, k1, (1 - wi) * (1 - wj) * wk),       (i1, j0, k1, wi * (1 - wj) * wk),
            (i0, j1, k1, (1 - wi) * wj * wk),             (i1, j1, k1, wi * wj * wk))
end

function _check_interp(interpolation::Symbol)
    interpolation in (:linear, :nearest) ||
        throw(ArgumentError("interpolation must be :linear or :nearest"))
    return interpolation
end

"""
    flanner_grain_radius(r, t_hours, drdt0, τ, κ; r0 = 54.5) -> μm

Grain radius after `t_hours` of dry ageing at fixed parameters, starting
from `r` — the exact solution of the Flanner & Zender (2006) growth law

```math
\\frac{dr}{dt} = \\left(\\frac{dr}{dt}\\right)_0
                 \\left(\\frac{τ}{(r - r_0) + τ}\\right)^{1/κ},
```

which with ``x = r - r_0 + τ`` integrates to
``x^{1+1/κ} = x_0^{1+1/κ} + (1 + 1/κ)(dr/dt)_0\\,τ^{1/κ}\\,t``.
Composes exactly: ageing for ``t_1`` then ``t_2`` equals ageing for
``t_1 + t_2``, so the result does not depend on the model time step.
"""
function flanner_grain_radius(r::Real, t_hours::Real, drdt0::Real, τ::Real, κ::Real;
                              r0::Real = 54.5)
    t_hours >= 0 || throw(ArgumentError("flanner_grain_radius: t_hours ≥ 0"))
    e = 1 + 1 / κ
    x0 = max(r - r0, 0.0) + τ
    x = (x0^e + e * drdt0 * τ^(1 / κ) * t_hours)^(1 / e)
    return r0 + x - τ
end

"""
    snicar_grain_radius(tab, r, t_hours, T, dTdz, ρ; interpolation = :linear) -> μm

Dry-aged grain radius after `t_hours` at fixed temperature `T` (K),
temperature gradient `dTdz` (K m⁻¹) and density `ρ` (kg m⁻³), starting
from `r` (μm), using [`flanner_grain_radius`](@ref) with the table's
fits. `:nearest` uses the nearest node (CLM). `:linear` evaluates the
exact solution at the eight surrounding nodes and blends the *radii*
trilinearly, so the result always lies between node outcomes and varies
continuously with the state (see [`snicar_aging_parameters`](@ref) for
why the parameters themselves are not interpolated).
"""
function snicar_grain_radius(tab::SnicarAgingTable, r::Real, t_hours::Real, T::Real,
                             dTdz::Real, ρ::Real; interpolation::Symbol = :linear)
    if _check_interp(interpolation) === :nearest
        q = snicar_aging_parameters(tab, T, dTdz, ρ)
        return flanner_grain_radius(r, t_hours, q.drdt0, q.τ, q.κ; r0 = tab.r0)
    end
    acc = 0.0
    for (i, j, k, w) in _snicar_corners(tab, T, dTdz, ρ)
        w == 0 && continue
        acc += w * flanner_grain_radius(r, t_hours, tab.drdt0[i, j, k], tab.τ[i, j, k],
                                        tab.κ[i, j, k]; r0 = tab.r0)
    end
    return acc
end

# ---- metamorphism schemes -------------------------------------------------

"""
    Flanner2006Metamorphism(; table = snicar_aging_table(), interpolation = :linear,
                              C_wet = 4.22e-13, f_liq_max = 0.1,
                              r_min = 54.5, r_max = 1500, r_refrozen = 1000,
                              fresh_radius_max = r_min, scale = 1)

The CLM / SNICAR grain-ageing scheme (Flanner & Zender 2006; CLM5
Technical Note, Sect. 2.3.2.3, Eqs. 2.3.74-2.3.77; CTSM
`SnowAge_grain`). Each step, in each layer:

 1. **dry ageing** — [`flanner_grain_radius`](@ref) with parameters from
    the [`SnicarAgingTable`](@ref) at the layer's temperature,
    temperature gradient ([`snow_temperature_gradient`](@ref)) and
    density, integrated exactly over the step at fixed parameters;
 2. **wet ageing** — Brun (1989), as in CLM:
    ``dr/dt = 10^{18} C_1 f_{liq}^3 / (4π r^2)`` μm s⁻¹ with
    ``C_1 = 4.22·10^{-13}`` m³ s⁻¹ and the liquid mass fraction
    ``f_{liq}`` capped at 0.1, integrated exactly
    (``r^3`` grows linearly in time);
 3. bounded to `[r_min, r_max]` (the range of the SNICAR Mie tables).

`scale` multiplies both rates (CLM's `xdrdt`). Fresh snow falls with
`r_min` (SSA = 60 m² kg⁻¹, the state the table's fits start from);
`fresh_radius_max > r_min` enables CLM5's linear ramp from `r_min` at
−30 °C to `fresh_radius_max` at 0 °C (the CLM5 tech note gives 204.5 μm).
Refrozen water has `r_refrozen` (see [`mix_refrozen_grain!`](@ref)).

`interpolation = :nearest` reproduces CLM's table look-up; the default
`:linear` blends the exact node solutions trilinearly
([`snicar_grain_radius`](@ref)), which removes the jumps in growth rate
as a layer crosses a 5 K, 10 K m⁻¹ or 50 kg m⁻³ cell boundary.

Deviation from CLM: CLM integrates both rates with one forward-Euler
step; here each is integrated exactly over the step at the layer's
current temperature, gradient, density and liquid fraction.
"""
Base.@kwdef struct Flanner2006Metamorphism <: AbstractSnowMetamorphism
    table::SnicarAgingTable = snicar_aging_table()
    interpolation::Symbol = :linear
    C_wet::Float64 = 4.22e-13          # m³ s⁻¹ (Brun 1989, as in CLM)
    f_liq_max::Float64 = 0.1
    r_min::Float64 = 54.5              # μm
    r_max::Float64 = 1500.0            # μm
    r_refrozen::Float64 = 1000.0       # μm
    fresh_radius_max::Float64 = 54.5   # μm; > r_min enables the CLM5 ramp
    scale::Float64 = 1.0
end

"""
    snow_dry_grain_growth_rate(scheme, r, T, dTdz, ρ) -> μm s⁻¹

Instantaneous dry-ageing rate of a grain of radius `r` (μm) at
temperature `T`, temperature gradient `dTdz`, density `ρ`.
"""
function snow_dry_grain_growth_rate(m::Flanner2006Metamorphism, r::Real, T::Real,
                                    dTdz::Real, ρ::Real)
    tab = m.table
    x = max(r - tab.r0, 0.0)
    rate(i, j, k) = tab.drdt0[i, j, k] * (tab.τ[i, j, k] / (x + tab.τ[i, j, k]))^(1 / tab.κ[i, j, k])
    ρc = max(ρ, 50.0)
    if m.interpolation === :nearest
        i, j, k = _nearest(tab.ρ, ρc), _nearest(tab.dTdz, abs(dTdz)), _nearest(tab.T, T)
        return m.scale * rate(i, j, k) / 3600
    end
    acc = 0.0
    for (i, j, k, w) in _snicar_corners(tab, T, dTdz, ρc)
        w == 0 || (acc += w * rate(i, j, k))
    end
    return m.scale * acc / 3600
end

"""
    snow_wet_grain_growth_rate(scheme, r, f_liq) -> μm s⁻¹

Brun (1989) wet-snow growth, ``10^{18} C_1 \\min(f_{liq}, f_{max})^3/(4π r^2)``.
"""
snow_wet_grain_growth_rate(m::Flanner2006Metamorphism, r::Real, f_liq::Real) =
    m.scale * 1e18 * m.C_wet * min(max(f_liq, 0.0), m.f_liq_max)^3 / (4π * r^2)

function fresh_grain_radius(m::Flanner2006Metamorphism, p, T_air::Real)
    m.fresh_radius_max <= m.r_min && return m.r_min
    ΔT = T_air - _Tfreeze(p.consts)
    w = clamp((ΔT + 30) / 30, 0.0, 1.0)
    return m.r_min + w * (m.fresh_radius_max - m.r_min)
end

refrozen_grain_radius(m::Flanner2006Metamorphism) = m.r_refrozen

function snow_metamorphism!(m::Flanner2006Metamorphism, p, dt::Real)
    n = snow_nlayers(p)
    n == 0 && return p
    G = snow_temperature_gradient(p)
    t_h = m.scale * dt / 3600
    for i in 1:n
        mi, ml = p.m_ice[i], p.m_liq[i]
        mt = mi + ml
        r = max(p.grain[i], m.r_min)
        if mt > 0 && p.dz[i] > 0
            ρ = max(mt / p.dz[i], 50.0)
            r = snicar_grain_radius(m.table, r, t_h, p.T[i], G[i], ρ;
                                    interpolation = m.interpolation)
            f = min(ml / mt, m.f_liq_max)
            if f > 0
                # d(r³)/dt = 3·10¹⁸ C₁ f³ / (4π): exact at fixed f
                r = cbrt(r^3 + 3e18 * m.C_wet * f^3 / (4π) * m.scale * dt)
            end
        end
        p.grain[i] = clamp(r, m.r_min, m.r_max)
    end
    p.age .+= dt
    return p
end

"""
    ParabolicGrainGrowth(; growth_rate = SurfaceOptics.SNOW_GROWTH_RATE)

The grain growth that `SurfaceOptics.SnowSurface` applies for its
`age_days` argument, ``r^2 = r_0^2 + G t`` (G in μm² day⁻¹), made
prognostic: each step delegates to `SurfaceOptics.snow_grain_radius`
with the layer's current radius as ``r_0``. Parabolic growth composes
exactly, so a pack aged with this scheme and a `SnowSurface` of the same
age agree to round-off. It has no temperature, gradient or liquid-water
dependence — a reference, not a substitute for
[`Flanner2006Metamorphism`](@ref).
"""
Base.@kwdef struct ParabolicGrainGrowth <: AbstractSnowMetamorphism
    growth_rate::Float64 = 1.0e4      # μm² day⁻¹ (= SurfaceOptics.SNOW_GROWTH_RATE)
end

function snow_metamorphism!(m::ParabolicGrainGrowth, p, dt::Real)
    for i in 1:snow_nlayers(p)
        p.grain[i] = snow_grain_radius(p.grain[i], dt / 86400; growth_rate = m.growth_rate)
    end
    p.age .+= dt
    return p
end

# ---- albedo ---------------------------------------------------------------

"""
    SnowOpticsAlbedo(; spec = shortwave_bands(), soot_ppmw = 0, μ₀ = nothing,
                       sensing_depth = 0.02)

Broadband snowpack albedo from `SurfaceOptics.SnowSurface` — the
Wiscombe & Warren (1980) grain-size and soot dependence already in the
code base — evaluated at the pack's prognostic optical grain radius
([`snowpack_optical_radius`](@ref)), solar-weighted over the shortwave
bands of `spec` by `SurfaceOptics.broadband_albedo`. `μ₀ = nothing` gives
the diffuse (white-sky) albedo, a number the direct-beam albedo at that
cosine solar zenith angle. `soot_ppmw` is the black-carbon mixing ratio
(ppm by mass) of the surface snow. The albedo ages exactly as fast as
the grains grow: this is a *delegating* scheme, not a second snow albedo.

Not represented: the underlying ground showing through thin snow
(patchy cover belongs to the tile framework, #41), impurity deposition
and enrichment by melt, and ponded water.
"""
struct SnowOpticsAlbedo{T<:AbstractFloat} <: AbstractSnowAlbedo
    spec::SpectralGrid{T}
    soot_ppmw::Float64
    μ₀::Union{Nothing,Float64}
    sensing_depth::Float64
end
function SnowOpticsAlbedo(; spec::SpectralGrid = shortwave_bands(), soot_ppmw::Real = 0.0,
                          μ₀::Union{Nothing,Real} = nothing, sensing_depth::Real = 0.02)
    soot_ppmw >= 0 || throw(ArgumentError("SnowOpticsAlbedo: soot_ppmw ≥ 0"))
    sensing_depth > 0 || throw(ArgumentError("SnowOpticsAlbedo: sensing_depth > 0"))
    return SnowOpticsAlbedo(spec, Float64(soot_ppmw),
                            μ₀ === nothing ? nothing : Float64(μ₀), Float64(sensing_depth))
end

"""
    snowpack_optical_radius(p; sensing_depth = 0.02) -> μm

Thickness-weighted mean grain radius of the top `sensing_depth` metres
of snow (the near-infrared, where grain size matters, is absorbed within
a few centimetres). With no snow, the fresh-snow radius.
"""
function snowpack_optical_radius(p::LayeredSnowpack; sensing_depth::Real = 0.02)
    n = snow_nlayers(p)
    n == 0 && return fresh_grain_radius(p.metamorphism, p, _Tfreeze(p.consts))
    acc, z = 0.0, 0.0
    for i in 1:n
        w = min(p.dz[i], sensing_depth - z)
        w <= 0 && break
        acc += w * p.grain[i]
        z += w
    end
    return z > 0 ? acc / z : p.grain[1]
end

"""
    snowpack_optics(scheme::SnowOpticsAlbedo, p) -> SurfaceOptics.SnowSurface

The spectral optics of the pack's surface: a `SnowSurface` with the
pack's optical radius and the scheme's soot loading. Radiation code that
wants band albedos (`surface_albedo(s, band, μ₀)`) uses this directly.
"""
snowpack_optics(a::SnowOpticsAlbedo, p::LayeredSnowpack) =
    SnowSurface(a.spec; grain_radius = snowpack_optical_radius(p; sensing_depth = a.sensing_depth),
                age_days = 0.0, soot_ppmw = a.soot_ppmw)

snowpack_albedo(a::SnowOpticsAlbedo, p::LayeredSnowpack) =
    broadband_albedo(snowpack_optics(a, p), a.μ₀)

end # module
