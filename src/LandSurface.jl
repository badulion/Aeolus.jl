"""
    LandSurface

The vegetated land column of Era V (Phase 47, #21): a two-big-leaf
(sunlit/shaded) canopy over a [`SoilThermalColumn`](@ref), coupled to the
atmosphere through the Monin–Obukhov solver of `SurfaceExchange` and
implementing its `AbstractSurface` interface, so the tile framework can
aggregate it like any other surface.

It owns what a vegetated column needs and no other module has:

  1. **Plant parameters for the column** — the soil-water-stress (ψ_o,
     ψ_c), root-distribution (Zeng 2001), leaf-optics and aerodynamic
     columns of the CLM4.5 PFT tables ([`VegetationParameters`](@ref),
     [`pft_vegetation`](@ref)). The photosynthetic columns belong to
     `Photosynthesis` and are taken from there.
  2. **Soil-water stress** — root fractions per soil layer
     ([`zeng_root_fractions`](@ref)), the CLM plant wilting factor and
     the root-weighted stress factor β_t ([`CLMWaterStress`](@ref),
     [`soil_water_stress`](@ref)), and the root-water-uptake sink that
     removes the transpiration from the soil layers
     ([`root_water_uptake`](@ref)).
  3. **Big-leaf canopy radiation** — sunlit/shaded shortwave absorption
     after Goudriaan (1977) and de Pury & Farquhar (1997)
     ([`bigleaf_shortwave`](@ref)) and the CLM canopy longwave exchange
     ([`bigleaf_longwave`](@ref)).
  4. **The column** — [`VegetatedLandColumn`](@ref): a canopy air space
     joined to the atmosphere (MOST), to the leaves (leaf boundary layer
     in series with the sunlit and shaded stomata of `Photosynthesis`)
     and to the ground (the under-canopy resistance of Zeng et al. 2005),
     with the leaf energy balance solved for the leaf temperature
     (CLM4.5 §5.3; Oleson et al. 2013).

What it delegates: leaf biochemistry, stomatal closures, leaf boundary
layer and the water-stress hook to `Photosynthesis`; soil water, heat,
freeze–thaw and the water table to `SoilColumn`; the surface layer and
material constants to `SurfaceExchange`; saturation, latent heats and air
properties to `MoistThermodynamics`.

# Sign convention

As the contract: `H`, `LE`, `E`, `τ` positive upward, `Rn` positive into
the surface, `G` positive into the soil, so ``R_n = H + LE + G`` (the
canopy stores no heat).

# References

  * Oleson, K. W. et al. (2013). *Technical description of version 4.5 of
    the Community Land Model (CLM)*. NCAR/TN-503+STR. §4.1-4.2 (canopy
    radiation), §5.3 (vegetated surface fluxes, eqs. 5.83-5.128), §7.4
    (root uptake, eqs. 7.104-7.107), §8.4 (soil-water stress, eqs.
    8.26-8.30), §8.2 (canopy scaling, eqs. 8.19-8.23), Tables 3.1, 5.1,
    8.1, 8.3.
  * Zeng, X. (2001). Global vegetation root distribution for land
    modeling. *J. Hydrometeorol.*, 2, 525-530.
  * Zeng, X., Barlage, M., Castro, C. & Fling, K. (2005). Comparison of
    land-precipitation coupling strength using observations and models.
    *J. Hydrometeorol.*, 6, 979-989 (under-canopy turbulent transfer).
  * Sakaguchi, K. & Zeng, X. (2009). Effects of soil wetness, plant
    litter, and under-canopy atmospheric stability on ground evaporation
    in the Community Land Model (CLM3.5). *J. Geophys. Res.*, 114, D01107.
  * Goudriaan, J. (1977). *Crop Micrometeorology: A Simulation Study*.
    Pudoc, Wageningen.
  * de Pury, D. G. G. & Farquhar, G. D. (1997). Simple scaling of
    photosynthesis from leaves to canopies without the errors of big-leaf
    models. *Plant Cell Environ.*, 20, 537-557.
  * Bonan, G. B. et al. (2011). Improving canopy processes in the
    Community Land Model version 4 (CLM4) using global flux fields
    empirically inferred from FLUXNET data. *J. Geophys. Res.*, 116,
    G02014 (K_n = 0.11).
  * Lee, T. J. & Pielke, R. A. (1992). Estimating the soil surface
    specific humidity. *J. Appl. Meteorol.*, 31, 480-484.
"""
module LandSurface

using ..MoistThermodynamics: ThermodynamicConstants, PhasePartition,
                             saturation_vapor_pressure, saturation_specific_humidity,
                             latent_heat_vapor, vapor_pressure, air_density,
                             heat_capacity_moist, dry_adiabatic_lapse_rate, R_v,
                             molmass_ratio
using ..SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS, SurfaceForcing,
                         SurfaceFluxes, AbstractSurface, AbstractStabilityFunctions,
                         BusingerDyerBeljaarsHoltslag, solve_surface_layer, grav
import ..SurfaceExchange: surface_temperature, roughness_lengths, surface_albedo,
                          surface_emissivity, moisture_availability,
                          surface_specific_humidity, surface_latent_heat,
                          net_radiation, step_surface!
using ..SoilColumn: SoilGrid, SoilWaterColumn, SoilThermalColumn, LinearizedSkinBalance,
                    step_soil_heat!, step_soil_water!, soil_porosity, soil_water_storage
import ..SoilColumn
using ..Photosynthesis: AbstractPhotosynthesisParameters, C3Parameters, C4Parameters,
                        AbstractStomatalModel, CLM45_PFT_PHOTOSYNTHESIS,
                        pft_photosynthesis_parameters, pft_stomatal_model,
                        leaf_gas_exchange, leaf_boundary_layer_conductance,
                        molar_conductance, par_to_photon_flux

export VegetationParameters, pft_vegetation,
       CLM45_WATER_STRESS_POTENTIALS, CLM45_ROOT_PARAMETERS, CLM45_LEAF_OPTICS,
       CLM45_AERODYNAMIC_PARAMETERS,
       zeng_root_fractions, AbstractSoilWaterStress, CLMWaterStress,
       wilting_factor, soil_water_stress, root_water_uptake,
       BigLeafShortwave, bigleaf_shortwave, BigLeafLongwave, bigleaf_longwave,
       canopy_roughness, undercanopy_transfer_coefficient,
       VegetatedLandColumn, LandSurfaceDiagnostics, LandSurfaceBudget,
       set_leaf_area!, radiative_temperature

# =========================================================================
# 1. Plant functional types: the columns of the CLM4.5 tables the land
#    column needs (the photosynthetic columns are in Photosynthesis)
# =========================================================================

"""
    CLM45_WATER_STRESS_POTENTIALS :: Dict{Symbol, NTuple{2,Float64}}

Soil matric potential at which the stomata are fully open (ψ_o) and fully
closed (ψ_c), **in metres of water**, per plant functional type — CLM4.5
Technical Note Table 8.1 (Oleson et al. 2013), where they are given in mm
(e.g. −66000 mm → −66 m). Keys are those of `CLM45_PFT_PHOTOSYNTHESIS`.

| PFTs | ψ_o (m) | ψ_c (m) |
| :-- | --: | --: |
| needleleaf trees, broadleaf evergreen trees | −66 | −255 |
| broadleaf deciduous trees | −35 | −224 |
| shrubs | −83 | −428 |
| grasses and crops | −74 | −275 |
"""
const CLM45_WATER_STRESS_POTENTIALS = Dict{Symbol,NTuple{2,Float64}}(
    :net_temperate => (-66.0, -255.0), :net_boreal => (-66.0, -255.0),
    :ndt_boreal => (-66.0, -255.0), :bet_tropical => (-66.0, -255.0),
    :bet_temperate => (-66.0, -255.0),
    :bdt_tropical => (-35.0, -224.0), :bdt_temperate => (-35.0, -224.0),
    :bdt_boreal => (-35.0, -224.0),
    :bes_temperate => (-83.0, -428.0), :bds_temperate => (-83.0, -428.0),
    :bds_boreal => (-83.0, -428.0),
    :c3_arctic_grass => (-74.0, -275.0), :c3_grass => (-74.0, -275.0),
    :c4_grass => (-74.0, -275.0), :crop => (-74.0, -275.0), :corn => (-74.0, -275.0),
    :temperate_cereal => (-74.0, -275.0), :winter_cereal => (-74.0, -275.0),
    :soybean => (-74.0, -275.0),
)

"""
    CLM45_ROOT_PARAMETERS :: Dict{Symbol, NTuple{2,Float64}}

Root-distribution parameters ``(r_a, r_b)`` (m⁻¹) of Zeng (2001) per
plant functional type, CLM4.5 Technical Note Table 8.3; see
[`zeng_root_fractions`](@ref).
"""
const CLM45_ROOT_PARAMETERS = Dict{Symbol,NTuple{2,Float64}}(
    :net_temperate => (7.0, 2.0), :net_boreal => (7.0, 2.0), :ndt_boreal => (7.0, 2.0),
    :bet_tropical => (7.0, 1.0), :bet_temperate => (7.0, 1.0),
    :bdt_tropical => (6.0, 2.0), :bdt_temperate => (6.0, 2.0), :bdt_boreal => (6.0, 2.0),
    :bes_temperate => (7.0, 1.5), :bds_temperate => (7.0, 1.5), :bds_boreal => (7.0, 1.5),
    :c3_arctic_grass => (11.0, 2.0), :c3_grass => (11.0, 2.0), :c4_grass => (11.0, 2.0),
    :crop => (6.0, 3.0), :corn => (6.0, 3.0), :temperate_cereal => (6.0, 3.0),
    :winter_cereal => (6.0, 3.0), :soybean => (6.0, 3.0),
)

"""
    CLM45_LEAF_OPTICS :: Dict{Symbol, NamedTuple}

Leaf reflectance and transmittance in the visible (PAR) and near-infrared
bands, `(ρ_vis, ρ_nir, τ_vis, τ_nir)`, per plant functional type — CLM4.5
Technical Note Table 3.1 (Dorman & Sellers 1989 for trees and shrubs,
Asner et al. 1998 for grasses and crops). The leaf-orientation index χ_L
and the stem optics of that table are not used: the big-leaf radiation
here assumes a spherical leaf-angle distribution and no stems (see
[`bigleaf_shortwave`](@ref)).
"""
const CLM45_LEAF_OPTICS = let
    needle = (ρ_vis = 0.07, ρ_nir = 0.35, τ_vis = 0.05, τ_nir = 0.10)
    broad  = (ρ_vis = 0.10, ρ_nir = 0.45, τ_vis = 0.05, τ_nir = 0.25)
    grass  = (ρ_vis = 0.11, ρ_nir = 0.35, τ_vis = 0.05, τ_nir = 0.34)
    Dict{Symbol,typeof(needle)}(
        :net_temperate => needle, :net_boreal => needle, :ndt_boreal => needle,
        :bet_tropical => broad, :bet_temperate => broad, :bdt_tropical => broad,
        :bdt_temperate => broad, :bdt_boreal => broad,
        :bes_temperate => needle,      # Table 3.1 gives BES the needleleaf values
        :bds_temperate => broad, :bds_boreal => broad,
        :c3_arctic_grass => grass, :c3_grass => grass, :c4_grass => grass,
        :crop => grass, :corn => grass, :temperate_cereal => grass,
        :winter_cereal => grass, :soybean => grass)
end

"""
    CLM45_AERODYNAMIC_PARAMETERS :: Dict{Symbol, NamedTuple}

Ratios of momentum roughness length and displacement height to canopy
height, `(R_z0m, R_d)`, and the leaf dimension `d_leaf` (m) per plant
functional type — CLM4.5 Technical Note Table 5.1.
"""
const CLM45_AERODYNAMIC_PARAMETERS = let
    tree  = (R_z0m = 0.055, R_d = 0.67, d_leaf = 0.04)
    bet   = (R_z0m = 0.075, R_d = 0.67, d_leaf = 0.04)
    short = (R_z0m = 0.120, R_d = 0.68, d_leaf = 0.04)
    Dict{Symbol,typeof(tree)}(
        :net_temperate => tree, :net_boreal => tree, :ndt_boreal => tree,
        :bet_tropical => bet, :bet_temperate => bet,
        :bdt_tropical => tree, :bdt_temperate => tree, :bdt_boreal => tree,
        :bes_temperate => short, :bds_temperate => short, :bds_boreal => short,
        :c3_arctic_grass => short, :c3_grass => short, :c4_grass => short,
        :crop => short, :corn => short, :temperate_cereal => short,
        :winter_cereal => short, :soybean => short)
end

"""
    VegetationParameters(; photosynthesis, stomata, ψ_o, ψ_c, r_a, r_b,
                         ρ_vis, ρ_nir, τ_vis, τ_nir, R_z0m, R_d, d_leaf = 0.04,
                         K_n = 0.11, Ω = 1.0, ε_leaf = 0.97,
                         stress_targets = (:vcmax, :rd, :g0))

Everything the land column needs to know about its plants:

| field | meaning |
| :-- | :-- |
| `photosynthesis` | canopy-top leaf parameters (`C3Parameters` / `C4Parameters`) |
| `stomata` | stomatal closure (`BallBerryStomata` / `MedlynStomata`) |
| `ψ_o`, `ψ_c` | soil potential (m) with stomata fully open / closed |
| `r_a`, `r_b` | Zeng (2001) root-distribution parameters (m⁻¹) |
| `ρ_vis`, `ρ_nir`, `τ_vis`, `τ_nir` | leaf reflectance and transmittance |
| `R_z0m`, `R_d` | roughness length and displacement height per canopy height |
| `d_leaf` | characteristic leaf dimension (m) |
| `K_n` | canopy nitrogen (V_cmax) extinction coefficient; 0.11 is Bonan et al. (2011), CLM4.5 eq. 8.19 |
| `Ω` | foliage clumping index (1 = random) |
| `ε_leaf` | leaf longwave emissivity (CLM: 0.97) |
| `stress_targets` | what β_t multiplies, passed to `leaf_gas_exchange` |

Use [`pft_vegetation`](@ref) for a CLM plant functional type.
"""
Base.@kwdef struct VegetationParameters{P<:AbstractPhotosynthesisParameters,
                                        M<:AbstractStomatalModel}
    photosynthesis::P
    stomata::M
    ψ_o::Float64
    ψ_c::Float64
    r_a::Float64
    r_b::Float64
    ρ_vis::Float64
    ρ_nir::Float64
    τ_vis::Float64
    τ_nir::Float64
    R_z0m::Float64
    R_d::Float64
    d_leaf::Float64 = 0.04
    K_n::Float64 = 0.11
    Ω::Float64 = 1.0
    ε_leaf::Float64 = 0.97
    stress_targets::Tuple = (:vcmax, :rd, :g0)
end

"""
    pft_vegetation(pft::Symbol; model = :medlyn, kwargs...) -> VegetationParameters

The [`VegetationParameters`](@ref) of a CLM plant functional type (a key
of `CLM45_PFT_PHOTOSYNTHESIS`): leaf biochemistry from
`pft_photosynthesis_parameters`, the stomatal closure from
`pft_stomatal_model(pft; model)` (`:medlyn`, CLM5, or `:ball_berry`,
CLM4.5), and the CLM4.5 Tables 3.1, 5.1, 8.1 (ψ_o, ψ_c) and 8.3 columns
held here. Keyword arguments override any field.
"""
function pft_vegetation(pft::Symbol; model::Symbol = :medlyn, kwargs...)
    haskey(CLM45_PFT_PHOTOSYNTHESIS, pft) ||
        throw(ArgumentError("unknown PFT :$pft; see keys(CLM45_PFT_PHOTOSYNTHESIS)"))
    ψo, ψc = CLM45_WATER_STRESS_POTENTIALS[pft]
    ra, rb = CLM45_ROOT_PARAMETERS[pft]
    o = CLM45_LEAF_OPTICS[pft]
    a = CLM45_AERODYNAMIC_PARAMETERS[pft]
    defaults = (photosynthesis = pft_photosynthesis_parameters(pft),
                stomata = pft_stomatal_model(pft; model = model),
                ψ_o = ψo, ψ_c = ψc, r_a = ra, r_b = rb,
                ρ_vis = o.ρ_vis, ρ_nir = o.ρ_nir, τ_vis = o.τ_vis, τ_nir = o.τ_nir,
                R_z0m = a.R_z0m, R_d = a.R_d, d_leaf = a.d_leaf)
    return VegetationParameters(; merge(defaults, (; kwargs...))...)
end

# =========================================================================
# 2. Roots and soil-water stress
# =========================================================================

"""
    zeng_root_fractions(grid::SoilGrid, r_a, r_b) -> Vector

Fraction of roots in each soil layer after Zeng (2001), CLM4.5 eq. 8.30:
the cumulative root fraction above depth z is
``Y(z) = 1 - \\tfrac12\\,(e^{-r_a z} + e^{-r_b z})``, so layer ``i``
(interfaces ``z_{h,i-1}``, ``z_{h,i}``) holds ``Y(z_{h,i}) - Y(z_{h,i-1})``
and the bottom layer also takes the roots below the column. The fractions
sum to one on any grid.
"""
function zeng_root_fractions(grid::SoilGrid, r_a::Real, r_b::Real)
    (r_a > 0 && r_b > 0) || throw(ArgumentError("zeng_root_fractions: r_a, r_b > 0"))
    zh = grid.zh
    N = length(grid)
    tail(z) = (exp(-r_a * z) + exp(-r_b * z)) / 2        # roots below z
    r = [tail(zh[i]) - tail(zh[i+1]) for i in 1:N]
    r[N] = tail(zh[N])
    return r
end

"""
    AbstractSoilWaterStress

How soil water limits the stomata. A subtype implements
`wilting_factor(stress, ψ)`, the stress of one soil layer at matric
potential ψ (m) with no ice; [`soil_water_stress`](@ref) weights it by
the root fractions. Provided: [`CLMWaterStress`](@ref).
"""
abstract type AbstractSoilWaterStress end

"""
    CLMWaterStress(ψ_o, ψ_c; T_limit = 2.0)

The CLM4.5 plant wilting factor (Oleson et al. 2013, eq. 8.27), linear in
the soil matric potential between full opening at ψ_o and closure at ψ_c
(m, ψ_c < ψ_o ≤ 0):

```math
w_i = \\frac{ψ_c - ψ_i}{ψ_c - ψ_o}\\,\\frac{θ_{sat} - θ_{ice,i}}{θ_{sat}} ∈ [0, 1],
```

zero in a layer colder than ``T_f - `` `T_limit` or without liquid water.
"""
struct CLMWaterStress <: AbstractSoilWaterStress
    ψ_o::Float64
    ψ_c::Float64
    T_limit::Float64
    function CLMWaterStress(ψ_o::Real, ψ_c::Real; T_limit::Real = 2.0)
        ψ_c < ψ_o <= 0 || throw(ArgumentError("CLMWaterStress: need ψ_c < ψ_o ≤ 0"))
        new(Float64(ψ_o), Float64(ψ_c), Float64(T_limit))
    end
end
CLMWaterStress(v::VegetationParameters) = CLMWaterStress(v.ψ_o, v.ψ_c)

"""
    wilting_factor(stress, ψ) -> w ∈ [0, 1]

Stress factor of one unfrozen soil layer at matric potential `ψ` (m).
"""
function wilting_factor end
wilting_factor(s::CLMWaterStress, ψ::Real) = clamp((s.ψ_c - ψ) / (s.ψ_c - s.ψ_o), 0.0, 1.0)

"""
    soil_water_stress(stress, col, roots) -> (β_t, w)

The root-zone soil-water stress factor ``β_t = Σ_i r_i w_i`` (CLM4.5 eq.
8.26) of a soil column `col` (`SoilWaterColumn` or `SoilThermalColumn`)
for the root fractions `roots`, together with the per-layer wilting
factors `w`. Each layer's factor is [`wilting_factor`](@ref) at the
layer's matric potential, scaled by the unfrozen pore fraction
``(θ_{sat} - θ_{ice})/θ_{sat}`` and set to zero when the layer is colder
than ``T_f -`` `T_limit` or holds no liquid water (eq. 8.27). β_t is the
factor `Photosynthesis.leaf_gas_exchange` applies to V_cmax, R_d and g_0.
"""
soil_water_stress(s::AbstractSoilWaterStress, col::SoilThermalColumn, roots) =
    soil_water_stress(s, col.water, roots)
function soil_water_stress(s::AbstractSoilWaterStress, col::SoilWaterColumn, roots)
    N = length(col.grid)
    length(roots) == N || throw(ArgumentError("soil_water_stress: one root fraction per layer"))
    Tf = col.constants.thermo.T_freeze
    Tlim = s isa CLMWaterStress ? s.T_limit : 2.0
    w = zeros(N)
    β = 0.0
    for i in 1:N
        (col.T_soil[i] > Tf - Tlim && col.θ_liq[i] > 0) || continue
        θs = soil_porosity(col.hydraulics[i])
        w[i] = wilting_factor(s, col.ψ[i]) * clamp((θs - col.θ_ice[i]) / θs, 0.0, 1.0)
        β += roots[i] * w[i]
    end
    return β, w
end

"""
    root_water_uptake(E_t, roots, w, β_t; ρ_w = 1000.0) -> sink (m s⁻¹ per layer)

Distribute the transpiration `E_t` (kg m⁻² s⁻¹) over the soil layers with
the effective root fraction ``r_{e,i} = r_i w_i / β_t`` (CLM4.5 eqs.
7.104, 7.107): layers that are wetter and hold more roots supply more.
Returns the per-layer sink in m s⁻¹ of liquid water, the `sink` argument
of `SoilColumn.step_soil_water!`; ``Σ_i \\text{sink}_i = E_t/ρ_w`` exactly
when β_t > 0. With β_t = 0 no layer can supply water and the sink is zero
([`VegetatedLandColumn`](@ref) then allows no transpiration).
"""
function root_water_uptake(E_t::Real, roots, w, β_t::Real; ρ_w::Real = 1000.0)
    N = length(roots)
    sink = zeros(N)
    (E_t > 0 && β_t > 0) || return sink
    q = E_t / ρ_w
    for i in 1:N
        sink[i] = q * roots[i] * w[i] / β_t
    end
    return sink
end

# =========================================================================
# 3. Big-leaf canopy radiation
# =========================================================================

# Gauss–Legendre nodes and weights on [0, 1] (16 points), for the
# hemispheric integrals of the isotropic (uniform overcast) sky.
const _GL16 = let
    x = [-0.9894009349916499, -0.9445750230732326, -0.8656312023878318,
         -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
         -0.2816035507792589, -0.0950125098376374, 0.0950125098376374,
         0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
         0.7554044083550030, 0.8656312023878318, 0.9445750230732326,
         0.9894009349916499]
    w = [0.0271524594117541, 0.0622535239386479, 0.0951585116824928,
         0.1246289712555339, 0.1495959888165767, 0.1691565193950025,
         0.1826034150449236, 0.1894506104550685, 0.1894506104550685,
         0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
         0.1246289712555339, 0.0951585116824928, 0.0622535239386479,
         0.0271524594117541]
    ((x .+ 1) ./ 2, w ./ 2)
end

# Spherical leaf-angle distribution: projection G = 1/2 for every direction.
const _G_SPHERICAL = 0.5

# Canopy reflection coefficient for a beam at μ (Goudriaan 1977; de Pury
# & Farquhar 1997 eqs. A19-A20): ρ_h of horizontal leaves corrected for
# the spherical distribution through k_b.
function _beam_canopy_reflectance(σ, kb)
    s = sqrt(1 - σ)
    ρh = (1 - s) / (1 + s)
    return 1 - exp(-2ρh * kb / (1 + kb))
end

# Diffuse (isotropic sky) transmittance of leaf area L, scattered light
# included in the k' = k √(1-σ) approximation, and the diffuse canopy
# reflectance: both hemispheric integrals 2∫ f(μ) μ dμ of the beam values.
function _diffuse_optics(σ, L, Ω)
    xs, ws = _GL16
    τ = zero(float(L)); ρ = zero(float(L))
    for (μ, wt) in zip(xs, ws)
        kb = Ω * _G_SPHERICAL / μ
        τ += 2wt * μ * exp(-kb * sqrt(1 - σ) * L)
        ρ += 2wt * μ * _beam_canopy_reflectance(σ, kb)
    end
    return τ, ρ
end

"""
    BigLeafShortwave

Shortwave absorbed by a two-big-leaf canopy and its ground, one band, from
[`bigleaf_shortwave`](@ref) (W m⁻² of ground): `sunlit`, `shaded` (canopy
absorption of each big leaf), `ground`, `reflected`, and the sunlit and
shaded leaf areas `L_sun`, `L_sha` (m² m⁻²) and the beam extinction
coefficient `k_b`. By construction
`sunlit + shaded + ground + reflected == I_beam + I_diffuse`.
"""
struct BigLeafShortwave
    sunlit::Float64
    shaded::Float64
    ground::Float64
    reflected::Float64
    L_sun::Float64
    L_sha::Float64
    k_b::Float64
end

"""
    bigleaf_shortwave(I_beam, I_diffuse, μ, L; σ, α_ground, Ω = 1) -> BigLeafShortwave

Two-big-leaf (sunlit/shaded) shortwave absorption for one waveband, after
Goudriaan (1977) and de Pury & Farquhar (1997). Inputs: horizontal beam
and diffuse irradiance above the canopy (W m⁻²), cosine of the solar
zenith angle `μ`, leaf area index `L`, leaf scattering coefficient
``σ = ρ_{leaf} + τ_{leaf}``, ground albedo `α_ground` and clumping index
`Ω`. Leaves are spherically distributed (``G = 1/2``), so the beam
extinction coefficient is ``k_b = Ω/(2μ)`` and the sunlit leaf area is
``L_{sun} = (1 - e^{-k_b L})/k_b``.

The beam (and, integrated over an isotropic sky, the diffuse) radiation
decays in the canopy as ``e^{-k' l}`` with ``k' = k\\sqrt{1-σ}``, after a
canopy reflection ``ρ_{cb}`` (Goudriaan) or ``ρ_{cd}``. The diffuse
``k_d'`` is chosen so that ``e^{-k_d' L}`` is the exact hemispheric
transmittance ``2∫_0^1 e^{-k_b'(μ) L} μ\\, dμ``, and ``ρ_{cd}`` is the
hemispheric average of ``ρ_{cb}``. Sunlit leaves absorb the unscattered
beam, the diffuse and the scattered beam in proportion to the sunlit
fraction ``e^{-k_b l}`` (de Pury & Farquhar 1997, eqs. A25-A28, integrated
in closed form); shaded leaves absorb the rest of the canopy absorption.

The ground receives ``(1-ρ_{cb}) I_b e^{-k_b' L} + (1-ρ_{cd}) I_d e^{-k_d' L}``
and reflects `α_ground` of it upward; of that, the fraction
``e^{-k_d' L}`` escapes and the rest is absorbed by the shaded leaves
(one ground reflection, no further canopy–ground reflections). Energy is
conserved exactly: sunlit + shaded + ground + reflected = incoming.
Stems and the leaf-orientation index χ_L of CLM are not represented.
Below the horizon (`μ ≤ 0`) all incident light is treated as diffuse and
there are no sunlit leaves.
"""
function bigleaf_shortwave(Ib::Real, Id::Real, μ::Real, L::Real;
                           σ::Real, α_ground::Real, Ω::Real = 1.0)
    (Ib >= 0 && Id >= 0) || throw(ArgumentError("bigleaf_shortwave: irradiances ≥ 0"))
    (0 <= σ < 1 && 0 <= α_ground <= 1 && L >= 0) ||
        throw(ArgumentError("bigleaf_shortwave: need 0 ≤ σ < 1, 0 ≤ α_ground ≤ 1, L ≥ 0"))
    if μ <= 1e-6
        Id, Ib = Id + Ib, 0.0
    end
    s = sqrt(1 - σ)
    if L <= 0
        down = Ib + Id
        return BigLeafShortwave(0.0, 0.0, (1 - α_ground) * down, α_ground * down,
                                0.0, 0.0, Ib > 0 ? Ω * _G_SPHERICAL / μ : Inf)
    end
    τd, ρcd = _diffuse_optics(σ, L, Ω)
    kdp = -log(τd) / L                           # effective diffuse k_d'
    if Ib > 0
        kb = Ω * _G_SPHERICAL / μ
        kbp = kb * s
        ρcb = _beam_canopy_reflectance(σ, kb)
        Lsun = -expm1(-kb * L) / kb
    else
        kb = kbp = Inf
        ρcb = 0.0
        Lsun = 0.0
    end
    # canopy absorption and transmission of each component
    Ab = Ib > 0 ? (1 - ρcb) * Ib * (-expm1(-kbp * L)) : 0.0
    Ad = (1 - ρcd) * Id * (-expm1(-kdp * L))
    Tb = Ib > 0 ? (1 - ρcb) * Ib * exp(-kbp * L) : 0.0
    Td = (1 - ρcd) * Id * τd
    # sunlit share (de Pury & Farquhar 1997, integrated over the sunlit fraction)
    if Ib > 0
        sun_dir = (1 - σ) * Ib * (-expm1(-kb * L))
        sun_dif = (1 - ρcd) * Id * kdp / (kdp + kb) * (-expm1(-(kdp + kb) * L))
        sun_sca = (1 - ρcb) * Ib * kbp / (kbp + kb) * (-expm1(-(kbp + kb) * L)) -
                  (1 - σ) * Ib * (-expm1(-2kb * L)) / 2
        sun = sun_dir + sun_dif + sun_sca
    else
        sun = 0.0
    end
    ground_in = Tb + Td
    up = α_ground * ground_in
    escape = up * τd
    shade = Ab + Ad - sun + (up - escape)
    return BigLeafShortwave(sun, shade, (1 - α_ground) * ground_in,
                            ρcb * Ib + ρcd * Id + escape, Lsun, L - Lsun, kb)
end

"""
    BigLeafLongwave

Longwave budget of canopy and ground from [`bigleaf_longwave`](@ref)
(W m⁻² of ground): `canopy` and `ground` net absorption (positive = gain),
`upward` emitted to the sky, and the canopy emissivity `ε_v`.
"""
struct BigLeafLongwave
    canopy::Float64
    ground::Float64
    upward::Float64
    ε_v::Float64
end

"""
    bigleaf_longwave(L↓, T_v, T_g, LS; ε_leaf = 0.97, ε_ground, σ_SB) -> BigLeafLongwave

Longwave exchange between sky, canopy and ground (CLM4.5 §4.2, eqs.
4.16-4.20 without the ground-temperature correction). The canopy of leaf
plus stem area `LS` has emissivity ``ε_v = 1 - e^{-LS/\\bar μ}`` with
``\\bar μ = 1`` (as CLM; `ε_leaf` is accepted for API symmetry and not
used). With one reflection at the ground:

    canopy = ε_v[1 + (1-ε_g)(1-ε_v)] L↓ + ε_v ε_g σT_g⁴ - [2 - ε_v(1-ε_g)] ε_v σT_v⁴
    ground = ε_g(1-ε_v) L↓ + ε_g ε_v σT_v⁴ - ε_g σT_g⁴
    upward = (1-ε_g)(1-ε_v)² L↓ + ε_v[1 + (1-ε_g)(1-ε_v)] σT_v⁴ + ε_g(1-ε_v) σT_g⁴

and ``canopy + ground + upward = L↓`` exactly.
"""
function bigleaf_longwave(Ldown::Real, Tv::Real, Tg::Real, LS::Real;
                          ε_leaf::Real = 0.97, ε_ground::Real, σ_SB::Real)
    εv = -expm1(-max(LS, 0.0))
    εg = ε_ground
    Bv = σ_SB * Tv^4
    Bg = σ_SB * Tg^4
    canopy = εv * (1 + (1 - εg) * (1 - εv)) * Ldown + εv * εg * Bg -
             (2 - εv * (1 - εg)) * εv * Bv
    ground = εg * (1 - εv) * Ldown + εg * εv * Bv - εg * Bg
    upward = (1 - εg) * (1 - εv)^2 * Ldown + εv * (1 + (1 - εg) * (1 - εv)) * Bv +
             εg * (1 - εv) * Bg
    return BigLeafLongwave(canopy, ground, upward, εv)
end

# =========================================================================
# 4. Canopy aerodynamics
# =========================================================================

"""
    canopy_roughness(z_top, LS, R_z0m, R_d, z0m_ground) -> (z0m, d)

Momentum roughness length and displacement height of a canopy of height
`z_top` (m) with leaf + stem area `LS` (CLM4.5 eqs. 5.126-5.128): the
full-canopy values ``R_{z0m} z_{top}`` and ``R_d z_{top}`` are reached at
``LS ≥ 2``, and below that the roughness is interpolated in log space
towards the ground's with the weight
``V = [1 - e^{-\\min(LS, 2)}]/(1 - e^{-2})``. CLM sets the heat and vapour
roughness of a vegetated surface equal to `z0m`.
"""
function canopy_roughness(z_top::Real, LS::Real, R_z0m::Real, R_d::Real, z0m_g::Real)
    V = -expm1(-min(max(LS, 0.0), 2.0)) / (-expm1(-2.0))
    z0 = V > 0 ? exp(V * log(z_top * R_z0m) + (1 - V) * log(z0m_g)) : float(z0m_g)
    return (z0m = z0, d = z_top * R_d * V)
end

"""
    undercanopy_transfer_coefficient(u_star, LS, z0m_ground; T_ac, T_g, z_top,
                                     g = 9.80665, κ = 0.4, ν = 1.5e-5) -> C_s

Turbulent transfer coefficient between the ground and the canopy air
(Zeng et al. 2005; Sakaguchi & Zeng 2009; CLM4.5 eqs. 5.115-5.119), so the
ground–canopy-air conductance is ``C_s u_*``:
``C_s = C_{s,bare} W + C_{s,dense}(1 - W)``, ``W = e^{-LS}``, with
``C_{s,bare} = (κ/a)(z_{0m,g} u_*/ν)^{-0.45}``, ``a = 0.13``, and
``C_{s,dense} = 0.004``, reduced to ``0.004/(1 + γ \\min(S, 10))`` with
γ = 0.5 when the canopy air is warmer than the ground (stable under-canopy
layer), ``S = g z_{top}(T_{ac} - T_g)/(T_{ac} u_*²)``.
"""
function undercanopy_transfer_coefficient(u_star::Real, LS::Real, z0m_g::Real;
                                          T_ac::Real, T_g::Real, z_top::Real,
                                          g::Real = 9.80665, κ::Real = 0.4,
                                          ν::Real = 1.5e-5)
    W = exp(-max(LS, 0.0))
    Cbare = κ / 0.13 * (z0m_g * u_star / ν)^(-0.45)
    Cdense = 0.004
    if T_ac > T_g
        S = g * z_top * (T_ac - T_g) / (T_ac * u_star^2)
        Cdense = 0.004 / (1 + 0.5 * min(S, 10.0))
    end
    return Cbare * W + Cdense * (1 - W)
end


# =========================================================================
# 5. The vegetated land column
# =========================================================================

"""
    LandSurfaceDiagnostics

What the last [`step_surface!`](@ref) of a [`VegetatedLandColumn`](@ref)
did. Fluxes in W m⁻² of ground (turbulent positive upward, radiation
positive absorbed), water fluxes in kg m⁻² s⁻¹:

| field | meaning |
| :-- | :-- |
| `Rn_v`, `Rn_g` | net radiation absorbed by the canopy and by the ground |
| `SW_v`, `SW_g`, `SW_up` | shortwave absorbed by canopy and ground, reflected |
| `H_v`, `H_g`, `LE_v`, `LE_g` | sensible and latent heat from canopy and ground |
| `G` | ground heat flux into the soil (step mean) |
| `E_t`, `E_dew`, `E_g` | transpiration, dew on the leaves (≥ 0), ground evaporation |
| `β_t`, `β_soil` | root-zone stress factor, ground evaporation efficiency |
| `GPP`, `A_sun`, `A_sha` | canopy gross and sunlit/shaded net assimilation, µmol m⁻² s⁻¹ (ground) |
| `gs_sun`, `gs_sha` | stomatal conductance of the sunlit and shaded big leaves, mol m⁻² s⁻¹ (leaf) |
| `L_sun`, `L_sha` | sunlit and shaded leaf area |
| `T_v`, `T_g`, `T_ac`, `q_ac` | leaf, ground skin (step mean), canopy-air temperature and humidity |
| `u_star`, `r_ah` | friction velocity, above-canopy resistance to heat (s m⁻¹) |
| `leaf_residual` | ``R_{n,v} - H_v - LE_v`` at the solution, W m⁻² |
| `iterations`, `converged` | canopy iteration count and outcome |
"""
struct LandSurfaceDiagnostics
    Rn_v::Float64; Rn_g::Float64
    SW_v::Float64; SW_g::Float64; SW_up::Float64
    H_v::Float64; H_g::Float64; LE_v::Float64; LE_g::Float64; G::Float64
    E_t::Float64; E_dew::Float64; E_g::Float64
    β_t::Float64; β_soil::Float64
    GPP::Float64; A_sun::Float64; A_sha::Float64
    gs_sun::Float64; gs_sha::Float64
    L_sun::Float64; L_sha::Float64
    T_v::Float64; T_g::Float64; T_ac::Float64; q_ac::Float64
    u_star::Float64; r_ah::Float64
    leaf_residual::Float64
    iterations::Int
    converged::Bool
end

"""
    LandSurfaceBudget

Cumulative budgets of a [`VegetatedLandColumn`](@ref) since construction.
Energy (J m⁻²): `Rn`, `H`, `LE`, `G`, and `leaf_residual`
(∫ (R_{n,v} - H_v - LE_v) dt, the canopy's energy-balance error, which a
converged canopy iteration keeps at round-off). Water (kg m⁻²): `rain`,
`transpiration`, `dew`, `ground_evaporation` (delivered by the soil),
`latent_mismatch` (J m⁻², latent heat charged for evaporation the soil
could not deliver), `snow_ignored` (snowfall, not handled: the column has
no snowpack), and `time` (s).
"""
mutable struct LandSurfaceBudget
    Rn::Float64; H::Float64; LE::Float64; G::Float64; leaf_residual::Float64
    rain::Float64; transpiration::Float64; dew::Float64; ground_evaporation::Float64
    latent_mismatch::Float64; snow_ignored::Float64; time::Float64
end
LandSurfaceBudget() = LandSurfaceBudget(zeros(12)...)

"""
    VegetatedLandColumn(soil::SoilThermalColumn, veg::VegetationParameters;
                        LAI, SAI = 0, z_top, co2 = 400, cosz = 0.5,
                        diffuse_fraction = 0.3, par_fraction = 0.5,
                        stress = CLMWaterStress(veg), roots = zeng_root_fractions(…),
                        T_veg = soil skin temperature, sf = BusingerDyerBeljaarsHoltslag(),
                        U_min = 0.1, maxiter = 40, tol_T = 1e-6)

A vegetated land column — a two-big-leaf canopy (leaf area `LAI`, stem
area `SAI`, height `z_top` m) over the soil column `soil` — implementing
`AbstractSurface`. The canopy follows the CLM4.5 formulation (Oleson et al.
2013, §5.3): leaves, ground and atmosphere exchange heat and vapour
through a canopy air space with no storage, whose temperature ``T_{ac}``
and humidity ``q_{ac}`` follow from flux continuity,

    T_ac = (c_a θ_a + c_g T_g + c_v T_v)/(c_a + c_g + c_v)             (5.90)
    q_ac = (c_a^w q_a + c_g^w q_g + c_v^w q_sat(T_v))/(c_a^w + c_g^w + c_v^w)   (5.104)

with conductances (m s⁻¹): ``c_a = C_H U`` above the canopy from
`SurfaceExchange.solve_surface_layer` between the canopy air (at
``d + z_0``) and the reference height; ``c_v = (L+S)\\, g_b`` for the
leaves, ``g_b`` from `Photosynthesis.leaf_boundary_layer_conductance` at
``U = u_*``; ``c_g = C_s u_*`` below the canopy
([`undercanopy_transfer_coefficient`](@ref)); for vapour
``c_v^w = Σ_{sun,sha} L_x g_b g_{s,x}/(g_b + g_{s,x})`` (leaf boundary
layer and stomata in series, eqs. 5.106, 5.110) and
``c_g^w = β_{soil} c_g`` with the Lee & Pielke β of the soil's top layer.

**Stomata and water stress.** The sunlit and shaded big leaves each call
`Photosynthesis.leaf_gas_exchange` with their absorbed PAR
([`bigleaf_shortwave`](@ref)), the leaf temperature, the canopy-air vapour
pressure, the leaf boundary layer, and the root-zone stress factor
β_t of [`soil_water_stress`](@ref), which scales V_cmax, R_d and g_0
(`veg.stress_targets`). Leaf capacities (V_cmax25, J_max25, T_p25, R_d25;
C4: V_cmax25, k_p25, R_d25) decline through the canopy as ``e^{-K_n x}``
and are integrated over the sunlit and shaded leaf area (CLM4.5 eqs.
8.19-8.23). With β_t = 0 the roots can supply no water and transpiration
is shut off.

**Leaf energy balance.** The leaf temperature ``T_v`` solves
``R_{n,v}(T_v) = H_v(T_v) + λ E_v(T_v)`` (eq. 5.129; the canopy stores no
heat) by Newton's method with the conductances fixed, inside an outer
iteration that updates the surface layer, the stomata and the canopy air
until ``T_v`` and ``T_{ac}`` change by less than `tol_T` (CLM allows 40
iterations). When the air at the leaves is supersaturated (``E_v < 0``)
dew forms through ``(L+S) g_b`` and is passed to the soil surface: the
column has no canopy interception store.

**Ground.** The ground skin is the soil column's: its fluxes
``H_g``, ``E_g``, ``R_{n,g}`` are linearised in ``T_g`` with the leaf
temperature and conductances held (as CLM does after the canopy
iteration) and handed to `SoilColumn.step_soil_heat!` as a
`LinearizedSkinBalance`, which solves the skin and soil temperatures
implicitly; the ground fluxes are then evaluated at the step-mean skin
temperature, so ``R_n - H - LE - G`` is the canopy residual alone.
Finally `SoilColumn.step_soil_water!` takes rain and leaf dew as input,
the ground evaporation as the surface demand, and the transpiration as a
root sink ([`root_water_uptake`](@ref)).

**Radiation.** `cosz` (cosine of the solar zenith angle),
`diffuse_fraction` and `par_fraction` (0.5: the split of CLM's
atmospheric forcing) are state, set before a step or passed to
[`step_surface!`](@ref), because `SurfaceForcing` carries only the global
shortwave — e.g. from `SolarGeometry.solar_position` and
`partition_global_irradiance`. The ground albedo and emissivity are the
soil column's.

Snow is not handled (no snowpack; snowfall is recorded in the budget as
ignored), nor is canopy interception of rain.
"""
mutable struct VegetatedLandColumn{S<:SoilThermalColumn, V<:VegetationParameters,
                                   W<:AbstractSoilWaterStress,
                                   F<:AbstractStabilityFunctions} <: AbstractSurface
    soil::S
    veg::V
    stress::W
    roots::Vector{Float64}
    LAI::Float64
    SAI::Float64
    z_top::Float64
    co2::Float64
    cosz::Float64
    diffuse_fraction::Float64
    par_fraction::Float64
    T_v::Float64
    T_ac::Float64
    q_ac::Float64
    albedo::Float64
    sf::F
    U_min::Float64
    maxiter::Int
    tol_T::Float64
    last::Union{Nothing, LandSurfaceDiagnostics}
    budget::LandSurfaceBudget
end

function VegetatedLandColumn(soil::SoilThermalColumn, veg::VegetationParameters;
                             LAI::Real, SAI::Real = 0.0, z_top::Real,
                             co2::Real = 400.0, cosz::Real = 0.5,
                             diffuse_fraction::Real = 0.3, par_fraction::Real = 0.5,
                             stress::AbstractSoilWaterStress = CLMWaterStress(veg),
                             roots = zeng_root_fractions(soil.water.grid, veg.r_a, veg.r_b),
                             T_veg::Real = soil.T_skin, q_ac::Real = 0.008,
                             sf::AbstractStabilityFunctions = BusingerDyerBeljaarsHoltslag(),
                             U_min::Real = 0.1, maxiter::Integer = 40, tol_T::Real = 1e-6)
    (LAI >= 0 && SAI >= 0 && z_top > 0) ||
        throw(ArgumentError("VegetatedLandColumn: need LAI, SAI ≥ 0 and z_top > 0"))
    length(roots) == length(soil.water.grid) ||
        throw(ArgumentError("VegetatedLandColumn: one root fraction per soil layer"))
    abs(sum(roots) - 1) < 1e-10 ||
        throw(ArgumentError("VegetatedLandColumn: root fractions must sum to 1"))
    s = VegetatedLandColumn(soil, veg, stress, collect(Float64, roots), Float64(LAI),
                            Float64(SAI), Float64(z_top), Float64(co2), Float64(cosz),
                            Float64(diffuse_fraction), Float64(par_fraction),
                            Float64(T_veg), Float64(T_veg), Float64(q_ac), 0.15, sf,
                            Float64(U_min), Int(maxiter), Float64(tol_T), nothing,
                            LandSurfaceBudget())
    return s
end

"""
    set_leaf_area!(s::VegetatedLandColumn, LAI, SAI = s.SAI)

Prescribe the leaf and stem area (phenology is an input of the column).
"""
function set_leaf_area!(s::VegetatedLandColumn, LAI::Real, SAI::Real = s.SAI)
    (LAI >= 0 && SAI >= 0) || throw(ArgumentError("set_leaf_area!: LAI, SAI ≥ 0"))
    s.LAI, s.SAI = Float64(LAI), Float64(SAI)
    return s
end

# ---- the AbstractSurface interface -------------------------------------------

"""
    surface_temperature(s::VegetatedLandColumn) -> T_ac

The *aerodynamic* surface temperature of a vegetated column: the canopy-air
temperature at ``d + z_0``, which is what the surface layer above sees.
[`radiative_temperature`](@ref) is the one that emits.
"""
surface_temperature(s::VegetatedLandColumn) = s.T_ac
"The canopy-air specific humidity ``q_{ac}`` (eq. 5.104): what the surface layer sees."
surface_specific_humidity(s::VegetatedLandColumn, f::SurfaceForcing; kwargs...) = s.q_ac
"Roughness of the canopy ([`canopy_roughness`](@ref)); CLM uses ``z_{0h} = z_{0q} = z_{0m}``."
function roughness_lengths(s::VegetatedLandColumn)
    z0 = canopy_roughness(s.z_top, s.LAI + s.SAI, s.veg.R_z0m, s.veg.R_d, s.soil.z0m).z0m
    return (z0m = z0, z0h = z0, z0q = z0)
end
"Effective shortwave albedo of canopy and ground for the current sun (last step)."
surface_albedo(s::VegetatedLandColumn) = s.albedo
"""
Effective longwave emissivity of canopy over ground:
``1 - (1-ε_g)(1-ε_v)²``, the complement of the sky radiation reflected back.
"""
function surface_emissivity(s::VegetatedLandColumn)
    εv = -expm1(-(s.LAI + s.SAI))
    return 1 - (1 - s.soil.emissivity) * (1 - εv)^2
end
"The root-zone soil-water stress factor β_t ([`soil_water_stress`](@ref))."
moisture_availability(s::VegetatedLandColumn) = soil_water_stress(s.stress, s.soil, s.roots)[1]
"Latent heat of vaporisation at the leaf temperature."
surface_latent_heat(s::VegetatedLandColumn; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    latent_heat_vapor(s.T_v, c.thermo)

"""
    radiative_temperature(s::VegetatedLandColumn; Ldown = 0) -> K

The temperature of a black body emitting the column's upwelling longwave
emission (leaf and ground, excluding reflected sky radiation for
`Ldown = 0`), ``[L↑_{emitted}/(ε σ)]^{1/4}`` with the effective
[`surface_emissivity`](@ref).
"""
function radiative_temperature(s::VegetatedLandColumn; Ldown::Real = 0.0)
    c = s.soil.water.constants
    lw = bigleaf_longwave(Ldown, s.T_v, s.soil.T_skin, s.LAI + s.SAI;
                          ε_ground = s.soil.emissivity, σ_SB = c.σ_SB)
    reflected = (1 - s.soil.emissivity) * (1 - lw.ε_v)^2 * Ldown
    return ((lw.upward - reflected) / (surface_emissivity(s) * c.σ_SB))^(1 / 4)
end

# ---- helpers --------------------------------------------------------------------

# dq_sat/dT from Clausius–Clapeyron with the Kirchhoff L_v of MoistThermodynamics.
function _dqsat_dT(T, p, thermo)
    es = saturation_vapor_pressure(T, thermo)
    ε = molmass_ratio(thermo)
    den = p - (1 - ε) * es
    des = latent_heat_vapor(T, thermo) * es / (R_v(thermo) * T^2)
    return ε * p * des / den^2
end

# Canopy-integrated capacity factors of the sunlit and shaded big leaves,
# per unit leaf area of each, relative to the canopy top (CLM4.5 8.20-8.23).
function _capacity_factors(Kn, kb, L, Lsun)
    L <= 0 && return (1.0, 1.0)
    tot = -expm1(-Kn * L) / Kn
    sun = isfinite(kb) ? -expm1(-(Kn + kb) * L) / (Kn + kb) : 0.0
    Lsha = L - Lsun
    fsun = Lsun > 1e-12 ? sun / Lsun : 1.0
    fsha = Lsha > 1e-12 ? (tot - sun) / Lsha : 1.0
    return (fsun, fsha)
end

_scale_capacity(p::C3Parameters, f) =
    C3Parameters(p.Vcmax25 * f, p.Jmax25 * f, p.Tp25 * f, p.Rd25 * f, p.Kc25, p.Ko25,
                 p.Γstar25, p.o_ref, p.Φ_PSII, p.Θ_PSII, p.Θ_cj, p.Θ_ip, p.temperature)
_scale_capacity(p::C4Parameters, f) =
    C4Parameters(p.Vcmax25 * f, p.kp25 * f, p.Rd25 * f, p.α, p.Θ_cj, p.Θ_ip,
                 p.Vcmax_T, p.kp_T, p.Rd_T)

# Shortwave of both bands for the current sun.
function _canopy_shortwave(s::VegetatedLandColumn, sw)
    v = s.veg
    αg = s.soil.albedo
    fdif = clamp(s.diffuse_fraction, 0.0, 1.0)
    fpar = s.par_fraction
    vis = bigleaf_shortwave(fpar * sw * (1 - fdif), fpar * sw * fdif, s.cosz, s.LAI;
                            σ = v.ρ_vis + v.τ_vis, α_ground = αg, Ω = v.Ω)
    nir = bigleaf_shortwave((1 - fpar) * sw * (1 - fdif), (1 - fpar) * sw * fdif, s.cosz, s.LAI;
                            σ = v.ρ_nir + v.τ_nir, α_ground = αg, Ω = v.Ω)
    return vis, nir
end

# Sunlit/shaded leaf gas exchange at leaf temperature Tv.
function _stomata(s::VegetatedLandColumn, vis, Tv, e_ac, gb_mol, p, β)
    v = s.veg
    fsun, fsha = _capacity_factors(v.K_n, vis.k_b, s.LAI, vis.L_sun)
    Qsun = vis.L_sun > 1e-12 ? par_to_photon_flux(vis.sunlit / vis.L_sun) : 0.0
    Qsha = vis.L_sha > 1e-12 ? par_to_photon_flux(vis.shaded / vis.L_sha) : 0.0
    kw = (ca = s.co2, T_leaf = Tv, e_air = e_ac, gb = gb_mol, p_air = p, β = β,
          stress_targets = v.stress_targets)
    sun = vis.L_sun > 1e-12 ?
          leaf_gas_exchange(_scale_capacity(v.photosynthesis, fsun), v.stomata; Q = Qsun, kw...) :
          nothing
    sha = vis.L_sha > 1e-12 ?
          leaf_gas_exchange(_scale_capacity(v.photosynthesis, fsha), v.stomata; Q = Qsha, kw...) :
          nothing
    return sun, sha
end

"""
    net_radiation(s::VegetatedLandColumn, forcing; c) -> W m⁻²

Net radiation of canopy plus ground for the current state and sun:
absorbed shortwave of both bands ([`bigleaf_shortwave`](@ref)) plus the
net longwave ([`bigleaf_longwave`](@ref)).
"""
function net_radiation(s::VegetatedLandColumn, f::SurfaceForcing;
                       c::SurfaceConstants = s.soil.water.constants)
    vis, nir = _canopy_shortwave(s, f.sw_down)
    lw = bigleaf_longwave(f.lw_down, s.T_v, s.soil.T_skin, s.LAI + s.SAI;
                          ε_ground = s.soil.emissivity, σ_SB = c.σ_SB)
    return vis.sunlit + vis.shaded + vis.ground + nir.sunlit + nir.shaded + nir.ground +
           lw.canopy + lw.ground
end

"""
    step_surface!(s::VegetatedLandColumn, forcing, dt; cosz = s.cosz,
                  diffuse_fraction = s.diffuse_fraction, co2 = s.co2,
                  gustiness = 0) -> SurfaceFluxes

Advance the vegetated column by `dt` seconds: canopy iteration (surface
layer, stomata, leaf energy balance, canopy air), then soil heat with the
ground balance linearised in the ground temperature, then soil water with
rain, dew, ground evaporation and root uptake. Returns the column's
`SurfaceFluxes` (canopy + ground; ``R_n - H - LE - G`` equals the canopy
residual, ≈ 0). Details in [`VegetatedLandColumn`](@ref); the component
fluxes are in `s.last` ([`LandSurfaceDiagnostics`](@ref)).
"""
function step_surface!(s::VegetatedLandColumn, f::SurfaceForcing, dt::Real;
                       cosz::Real = s.cosz, diffuse_fraction::Real = s.diffuse_fraction,
                       co2::Real = s.co2, gustiness::Real = 0.0)
    dt > 0 || throw(ArgumentError("step_surface!: dt > 0"))
    s.cosz, s.diffuse_fraction, s.co2 = Float64(cosz), Float64(diffuse_fraction), Float64(co2)
    soil = s.soil
    c = soil.water.constants
    th = c.thermo
    v = s.veg
    LS = s.LAI + s.SAI
    p = f.p
    qa = PhasePartition(f.q_air)
    ρ = air_density(f.T_air, p, qa, th)
    cp = heat_capacity_moist(qa, th)
    θa = f.T_air + dry_adiabatic_lapse_rate(qa, th) * f.z_ref
    rough = canopy_roughness(s.z_top, LS, v.R_z0m, v.R_d, soil.z0m)
    z0, d = rough.z0m, rough.d
    zm = f.z_ref - d
    zm > z0 || throw(ArgumentError("step_surface!: reference height $(f.z_ref) m must exceed d + z0 = $(d + z0) m"))
    Tg0 = soil.T_skin

    # --- radiation that does not depend on temperatures
    vis, nir = _canopy_shortwave(s, f.sw_down)
    SWv = vis.sunlit + vis.shaded + nir.sunlit + nir.shaded
    SWg = vis.ground + nir.ground
    SWup = vis.reflected + nir.reflected
    s.albedo = f.sw_down > 0 ? SWup / f.sw_down : s.albedo

    # --- soil water stress and ground evaporation efficiency
    βt, wlay = soil_water_stress(s.stress, soil, s.roots)
    βsoil0 = moisture_availability(soil.water)
    qg = saturation_specific_humidity(Tg0, p, th)
    σSB = c.σ_SB
    εg = soil.emissivity

    Tv, Tac, qac = s.T_v, s.T_ac, s.q_ac
    Tac = isfinite(Tac) ? Tac : f.T_air
    ca = caw = cv = cvw = cg = cgw = 0.0
    us = rah = 0.0
    sun = sha = nothing
    gb_mol = 0.0
    λv = latent_heat_vapor(Tv, th)
    iters, conv = 0, false
    for it in 1:s.maxiter
        iters = it
        # surface layer above the canopy (SurfaceExchange)
        sol = solve_surface_layer(s.sf; U = f.U, z_m = zm, θ_a = θa, θ_s = Tac,
                                  q_a = f.q_air, q_s = qac, z0m = z0, z0h = z0, z0q = z0,
                                  c = c, U_min = s.U_min, gustiness = gustiness)
        us = sol.u_star
        ca = sol.C_H * sol.U
        caw = sol.C_E * sol.U
        rah = 1 / ca
        # leaf boundary layer (Photosynthesis) at U = u*, per unit leaf area
        gb_mol = leaf_boundary_layer_conductance(us, f.T_air, p; d_leaf = v.d_leaf, c = th)
        mol_per_m = molar_conductance(1.0, f.T_air, p; c = th)          # mol m⁻³
        gb = gb_mol / mol_per_m                                         # m s⁻¹
        cv = LS * gb
        # below the canopy
        Cs = undercanopy_transfer_coefficient(us, LS, soil.z0m; T_ac = Tac, T_g = Tg0,
                                              z_top = s.z_top, g = grav(c), κ = c.κ)
        cg = Cs * us
        βsoil = qg < qac ? 1.0 : βsoil0                                  # condensation onto soil
        cgw = βsoil * cg
        # stomata of the sunlit and shaded leaves
        eac = vapor_pressure(qac, p, th)
        if s.LAI > 0
            sun, sha = _stomata(s, vis, Tv, eac, gb_mol, p, βt)
        end
        cvw_t = 0.0
        if βt > 0
            sun === nothing || (cvw_t += vis.L_sun * gb * (sun.gs / mol_per_m) / (gb + sun.gs / mol_per_m))
            sha === nothing || (cvw_t += vis.L_sha * gb * (sha.gs / mol_per_m) / (gb + sha.gs / mol_per_m))
        end
        # Newton on the leaf energy balance with the conductances fixed
        Tv_old, Tac_old = Tv, Tac
        Σh = ca + cg + cv
        for _ in 1:(LS > 0 ? 50 : 0)             # no canopy: nothing to balance
            qv = saturation_specific_humidity(Tv, p, th)
            dqv = _dqsat_dT(Tv, p, th)
            cvw = qv > qac ? cvw_t : LS * gb          # dew wets all leaf and stem area
            Σw = caw + cgw + cvw
            λv = latent_heat_vapor(Tv, th)
            lw = bigleaf_longwave(f.lw_down, Tv, Tg0, LS; ε_ground = εg, σ_SB = σSB)
            Hv = -ρ * cp * (ca * θa + cg * Tg0 - (ca + cg) * Tv) * cv / Σh
            Ev = Σw > 0 ? -ρ * (caw * f.q_air + cgw * qg - (caw + cgw) * qv) * cvw / Σw : 0.0
            F = SWv + lw.canopy - Hv - λv * Ev
            εv = lw.ε_v
            dlw = -(2 - εv * (1 - εg)) * εv * 4σSB * Tv^3
            dHv = ρ * cp * (ca + cg) * cv / Σh
            dEv = Σw > 0 ? ρ * (caw + cgw) * cvw / Σw * dqv : 0.0
            dF = dlw - dHv - λv * dEv
            ΔT = -F / dF
            ΔT = clamp(ΔT, -10.0, 10.0)
            Tv += ΔT
            abs(ΔT) < 1e-10 && break
        end
        qv = saturation_specific_humidity(Tv, p, th)
        cvw = qv > qac ? cvw_t : LS * gb
        # Canopy air from flux continuity; under-relaxed after 10 iterations,
        # which breaks the limit cycle the stability switch of the surface
        # layer can otherwise sustain near neutral (H ≈ 0 at dusk).
        ω = it <= 10 ? 1.0 : 0.5
        Σw = caw + cgw + cvw
        Tac += ω * ((ca * θa + cg * Tg0 + cv * Tv) / Σh - Tac)
        qac += ω * ((caw * f.q_air + cgw * qg + cvw * qv) / Σw - qac)
        if it > 1 && abs(Tv - Tv_old) < s.tol_T && abs(Tac - Tac_old) < s.tol_T
            conv = true
            break
        end
    end

    # --- canopy fluxes at the solution (T_g at the start of the step)
    qv = saturation_specific_humidity(Tv, p, th)
    Σh = ca + cg + cv
    Σw = caw + cgw + cvw
    λv = latent_heat_vapor(Tv, th)
    lw = bigleaf_longwave(f.lw_down, Tv, Tg0, LS; ε_ground = εg, σ_SB = σSB)
    Hv = -ρ * cp * (ca * θa + cg * Tg0 - (ca + cg) * Tv) * cv / Σh
    Ev = Σw > 0 ? -ρ * (caw * f.q_air + cgw * qg - (caw + cgw) * qv) * cvw / Σw : 0.0
    Rnv = SWv + lw.canopy
    leaf_res = Rnv - Hv - λv * Ev

    # --- ground: linearised balance handed to the soil heat solver
    λg = latent_heat_vapor(Tg0, th)
    Hg0 = -ρ * cp * (ca * θa + cv * Tv - (ca + cv) * Tg0) * cg / Σh
    dHg = ρ * cp * cg * (ca + cv) / Σh
    Eg0 = Σw > 0 ? -ρ * (caw * f.q_air + cvw * qv - (caw + cvw) * qg) * cgw / Σw : 0.0
    dEg = Σw > 0 ? ρ * cgw * (caw + cvw) / Σw * _dqsat_dT(Tg0, p, th) : 0.0
    Rng0 = SWg + lw.ground
    dRng = -4εg * σSB * Tg0^3
    if soil.hydrology && Eg0 > 0
        # the top layer cannot evaporate more liquid than it holds
        Emax = c.ρ_water * max(soil.water.θ_liq[1] - SoilColumn._liquid_floor(soil, 1), 0.0) *
               soil.water.grid.dz[1] / dt
        if Eg0 > Emax
            φ = Emax / Eg0
            Eg0 *= φ; dEg *= φ
        end
    end
    G0 = Rng0 - Hg0 - λg * Eg0
    D = dRng - dHg - λg * dEg
    st = step_soil_heat!(soil, dt, LinearizedSkinBalance(G0, D, Tg0))
    ΔTg = st.T_skin_mean - Tg0
    Hg = Hg0 + dHg * ΔTg
    Eg = Eg0 + dEg * ΔTg
    Rng = Rng0 + dRng * ΔTg
    LEg = λg * Eg
    G = Rng - Hg - LEg                        # == st.G: the linearised skin balance

    # --- water: rain and leaf dew in, ground evaporation, root uptake
    Et = max(Ev, 0.0)
    Edew = max(-Ev, 0.0)
    ρw = c.ρ_water
    if soil.hydrology
        sink = root_water_uptake(Et, s.roots, wlay, βt; ρ_w = ρw)
        ws = step_soil_water!(soil, dt; water_input = f.rain / ρw + Edew / ρw,
                              evaporation = Eg / ρw, sink = sink)
        s.budget.ground_evaporation += ws.evaporation * ρw
        s.budget.latent_mismatch += (Eg * dt - ws.evaporation * ρw) * λg
    end

    # --- canopy air at the new ground temperature, for the next step
    Tg1 = soil.T_skin
    qg1 = saturation_specific_humidity(Tg1, p, th)
    s.T_v = Tv
    s.T_ac = (ca * θa + cg * Tg1 + cv * Tv) / Σh
    s.q_ac = Σw > 0 ? (caw * f.q_air + cgw * qg1 + cvw * qv) / Σw : f.q_air

    Rn = Rnv + Rng
    H = Hv + Hg
    LE = λv * Ev + LEg
    E = Ev + Eg
    τ = ρ * us^2
    An_sun = sun === nothing ? 0.0 : sun.An
    An_sha = sha === nothing ? 0.0 : sha.An
    GPP = (sun === nothing ? 0.0 : sun.A * vis.L_sun) + (sha === nothing ? 0.0 : sha.A * vis.L_sha)
    s.last = LandSurfaceDiagnostics(Rnv, Rng, SWv, SWg, SWup, Hv, Hg, λv * Ev, LEg, G,
                                    Et, Edew, Eg, βt, βsoil0, GPP,
                                    An_sun * vis.L_sun, An_sha * vis.L_sha,
                                    sun === nothing ? 0.0 : sun.gs, sha === nothing ? 0.0 : sha.gs,
                                    vis.L_sun, vis.L_sha, Tv, st.T_skin_mean, s.T_ac, s.q_ac,
                                    us, rah, leaf_res, iters, conv)
    b = s.budget
    b.Rn += Rn * dt; b.H += H * dt; b.LE += LE * dt; b.G += G * dt
    b.leaf_residual += leaf_res * dt
    b.rain += f.rain * dt; b.transpiration += Et * dt; b.dew += Edew * dt
    b.snow_ignored += f.snow * dt; b.time += dt
    return SurfaceFluxes(promote(Rn, H, LE, G, τ, E)...)
end

end # module
