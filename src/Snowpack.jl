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
       Vionnet2012Compaction, snow_compaction_rate, snow_viscosity

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

end # module
