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
       canopy_roughness, undercanopy_transfer_coefficient

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


end # module
