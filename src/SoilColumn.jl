"""
    SoilColumn

Multi-layer soil column (Phase 46) — CLM-style vertical
discretisation with Richards-equation water transport, heat conduction
and freeze–thaw.

Contents:

  1. **Layer structure** — [`SoilGrid`](@ref) with the CLM5 layer
     thicknesses ([`clm5_soil_grid`](@ref), Lawrence et al. 2019), the
     CLM2–4.5 exponential nodes ([`clm_exponential_soil_grid`](@ref)) and
     a uniform grid for benchmarks.
  2. **Hydraulic functions** — [`AbstractHydraulicModel`](@ref) with
     Clapp–Hornberger (1978), van Genuchten (1980)–Mualem (1976) and
     Gardner (1958) exponential soils; texture tables of Clapp &
     Hornberger (1978) and Carsel & Parrish (1988), and the Cosby et al.
     (1984) pedotransfer functions.
  3. **Temperature and ice** — hydraulic conductivity scales with the
     fluidity of water, ``K ∝ 1/μ(T)``, with ``μ(T)`` from Kestin et al.
     (1978); an [`AbstractIceImpedance`](@ref) hook multiplies ``K`` by
     an ice-impedance factor that the freeze–thaw model (#16) supplies.
  4. **Richards solver** — [`SoilWaterColumn`](@ref) and
     [`step_soil_water!`](@ref): the mixed form of Richards' equation,
     backward Euler in time, finite volumes in depth, linearised by the
     modified Picard iteration of Celia et al. (1990). The water content is
     advanced by the divergence of the converged interface fluxes, so the
     column conserves water to round-off. The top boundary is an
     atmospheric flux that switches to a head condition when the surface
     ponds (runoff) or dries out (soil-limited evaporation); the bottom
     boundary is an [`AbstractSoilBottomBoundary`](@ref) (free drainage by
     default; [`UnconfinedAquifer`](@ref) is the water table of #17).
  5. **Heat and freeze–thaw** (#16) — [`SoilThermalProperties`](@ref)
     with the Johansen (1975) / Farouki (1981) conductivity and a
     constituent heat capacity; [`SoilThermalColumn`](@ref), which wraps a
     water column and solves the heat equation in enthalpy form with
     latent heat of fusion, a sharp or Clapeyron (supercooled-water)
     freezing curve and the CLM power-law ice impedance
     ([`PowerLawIceImpedance`](@ref)); and a skin energy balance that makes
     the column an `AbstractSurface` ([`step_surface!`](@ref)).
  6. **Water table** (#17) — [`UnconfinedAquifer`](@ref), a prognostic
     unconfined aquifer below the column (SIMTOP; Niu et al. 2005, 2007,
     as in CLM4): recharge, water-table dynamics and topographic baseflow
     ``q_{max} e^{-f z_∇}``, conserving soil + aquifer water exactly.

Depth `z` is positive **downward** from the soil surface; matric
potential `ψ` is a pressure head in metres of water (negative when
unsaturated); water fluxes are in m s⁻¹ of liquid water, positive
**downward** inside the soil.

# References

  * Celia, M. A., Bouloutas, E. T. & Zarba, R. L. (1990). A general
    mass-conservative numerical solution for the unsaturated flow
    equation. *Water Resour. Res.*, 26, 1483-1496.
  * Clapp, R. B. & Hornberger, G. M. (1978). Empirical equations for some
    soil hydraulic properties. *Water Resour. Res.*, 14, 601-604.
  * Cosby, B. J., Hornberger, G. M., Clapp, R. B. & Ginn, T. R. (1984). A
    statistical exploration of the relationships of soil moisture
    characteristics to the physical properties of soils. *Water Resour.
    Res.*, 20, 682-690.
  * Carsel, R. F. & Parrish, R. S. (1988). Developing joint probability
    distributions of soil water retention characteristics. *Water Resour.
    Res.*, 24, 755-769.
  * van Genuchten, M. Th. (1980). A closed-form equation for predicting
    the hydraulic conductivity of unsaturated soils. *Soil Sci. Soc. Am.
    J.*, 44, 892-898.
  * Mualem, Y. (1976). A new model for predicting the hydraulic
    conductivity of unsaturated porous media. *Water Resour. Res.*, 12,
    513-522.
  * Gardner, W. R. (1958). Some steady-state solutions of the unsaturated
    moisture flow equation with application to evaporation from a water
    table. *Soil Sci.*, 85, 228-232.
  * Kestin, J., Sokolov, M. & Wakeham, W. A. (1978). Viscosity of liquid
    water in the range −8 °C to 150 °C. *J. Phys. Chem. Ref. Data*, 7,
    941-948.
  * Lawrence, D. M. et al. (2019). The Community Land Model version 5.
    *J. Adv. Model. Earth Syst.*, 11, 4245-4287; CLM5.0 Technical Note,
    Table 2.2.3.
  * Oleson, K. W. et al. (2004). Technical description of the Community
    Land Model (CLM). NCAR/TN-461+STR, §6.1.
  * Lee, T. J. & Pielke, R. A. (1992). Estimating the soil surface
    specific humidity. *J. Appl. Meteorol.*, 31, 480-484.
  * Parlange, J.-Y. (1975). On solving the flow equation in unsaturated
    soils by optimization: horizontal infiltration. *Soil Sci. Soc. Am.
    Proc.*, 39, 415-418.
  * Johansen, O. (1975). *Thermal conductivity of soils.* PhD thesis,
    Trondheim (CRREL draft translation 637, 1977).
  * Farouki, O. T. (1981). *Thermal properties of soils.* CRREL
    Monograph 81-1.
  * de Vries, D. A. (1963). Thermal properties of soils. In *Physics of
    Plant Environment* (ed. van Wijk), 210-235. North-Holland.
  * Peters-Lidard, C. D., Blackburn, E., Liang, X. & Wood, E. F. (1998).
    The effect of soil thermal conductivity parameterization on surface
    energy fluxes and temperatures. *J. Atmos. Sci.*, 55, 1209-1224.
  * Koopmans, R. W. R. & Miller, R. D. (1966). Soil freezing and soil
    water characteristic curves. *Soil Sci. Soc. Am. Proc.*, 30, 680-685.
  * Niu, G.-Y. & Yang, Z.-L. (2006). Effects of frozen soil on snowmelt
    runoff and soil water storage at a continental scale. *J.
    Hydrometeorol.*, 7, 937-952.
  * Dall'Amico, M., Endrizzi, S., Gruber, S. & Rigon, R. (2011). A robust
    and energy-conserving model of freezing variably-saturated soil. *The
    Cryosphere*, 5, 469-484.
  * Swenson, S. C., Lawrence, D. M. & Lee, H. (2012). Improved simulation
    of the terrestrial hydrological cycle in permafrost regions by the
    Community Land Model. *J. Adv. Model. Earth Syst.*, 4, M08002.
  * Carslaw, H. S. & Jaeger, J. C. (1959). *Conduction of Heat in
    Solids*, 2nd ed., §2.6 (periodic surface temperature) and §11.2
    (Neumann's solution). Oxford.
  * Voller, V. R., Swaminathan, C. R. & Thomas, B. G. (1990). Fixed grid
    techniques for phase change problems: a review. *Int. J. Numer.
    Methods Eng.*, 30, 875-898.
  * Nedjar, B. (2002). An enthalpy-based finite element method for
    nonlinear heat problems involving phase change. *Comput. Struct.*,
    80, 9-21.
"""
module SoilColumn

using ..SurfaceExchange: SurfaceExchange, has_packed_state, packed_state_length, pack_state!, unpack_state!,
                         SurfaceConstants, DEFAULT_SURFACE_CONSTANTS,
                         SurfaceForcing, SurfaceFluxes, AbstractSurface, grav,
                         cp_water, cp_ice, LH_f0, turbulent_fluxes, surface_latent_heat
import ..SurfaceExchange: moisture_availability, surface_temperature, roughness_lengths,
                          surface_albedo, surface_emissivity, step_surface!
using ..SurfaceOptics: material_emissivity
using ..TimeIntegration: NumericalFailure   # a diverged solver (#327 M6)

export soil_profile_on_nodes, SOIL_YEAR, soil_damping_depth, annual_soil_temperature,
       SoilGrid, clm5_soil_grid, clm_exponential_soil_grid, uniform_soil_grid,
       AbstractHydraulicModel, ClappHornberger, VanGenuchtenMualem, GardnerExponential,
       CLAPP_HORNBERGER_1978, CARSEL_PARRISH_1988,
       clapp_hornberger_texture, carsel_parrish_texture, cosby_clapp_hornberger,
       soil_water_content, soil_matric_potential, soil_hydraulic_conductivity,
       soil_water_capacity, soil_water_diffusivity, soil_porosity,
       residual_water_content, air_entry_potential, saturated_conductivity,
       parlange_sorptivity,
       water_dynamic_viscosity, viscosity_conductivity_factor,
       AbstractIceImpedance, NoIceImpedance, ice_impedance_factor,
       AbstractSoilTopBoundary, AtmosphericTop, HeadTop,
       AbstractSoilBottomBoundary, FreeDrainage, ZeroFluxBottom, HeadBottom,
       bottom_flux_linearization, finalize_bottom!,
       SoilWaterColumn, SoilWaterBudget, SoilWaterStep, step_soil_water!,
       set_water_content!, set_matric_potential!, soil_water_storage,
       soil_layer_conductivity, soil_field_capacity,
       # heat and freeze–thaw (#16)
       SoilThermalProperties, clm_soil_thermal_properties, johansen_soil_thermal_properties,
       farouki_solids_conductivity, farouki_solids_heat_capacity,
       johansen_solids_conductivity, johansen_dry_conductivity, kersten_number,
       soil_thermal_conductivity, soil_heat_capacity,
       freezing_matric_potential, freezing_onset_temperature, supercooled_liquid_content,
       PowerLawIceImpedance,
       AbstractSoilHeatTop, GroundHeatFluxTop, SkinTemperatureTop, LinearizedSkinBalance,
       AbstractSoilHeatBottom, ZeroHeatFluxBottom, GeothermalHeatFlux, BottomTemperature,
       SoilThermalColumn, SoilHeatBudget, SoilHeatStep, step_soil_heat!,
       set_soil_temperature!, soil_enthalpy, soil_ice_mass, soil_thermal_layer_conductivity,
       # water table (#17)
       UnconfinedAquifer, water_table_depth, set_water_table!, aquifer_baseflow_rate,
       aquifer_impermeable_fraction, gardner_two_point_flux, bottom_water_storage,
       total_water_storage

# =========================================================================
# 1. Layer structure
# =========================================================================

"""
    SoilGrid(zh; z = midpoints)

Vertical structure of a soil column: `N` layers bounded by the `N + 1`
interface depths `zh` (m, positive downward, `zh[1] = 0` the surface),
with node depths `z` (default: layer midpoints) where the prognostic
variables live. Fields `z`, `zh`, `dz` (layer thicknesses).
"""
struct SoilGrid{T<:AbstractFloat}
    z::Vector{T}
    zh::Vector{T}
    dz::Vector{T}
    function SoilGrid{T}(z::Vector{T}, zh::Vector{T}) where {T}
        N = length(z)
        length(zh) == N + 1 || throw(ArgumentError("SoilGrid: need length(zh) == length(z) + 1"))
        zh[1] == 0 || throw(ArgumentError("SoilGrid: the first interface must be the surface, zh[1] = 0"))
        all(i -> zh[i] < z[i] < zh[i+1], 1:N) ||
            throw(ArgumentError("SoilGrid: every node must lie strictly inside its layer"))
        new{T}(z, zh, diff(zh))
    end
end
function SoilGrid(zh::AbstractVector{<:Real}; z = nothing)
    T = float(eltype(zh))
    zhv = Vector{T}(zh)
    zv = z === nothing ? [(zhv[i] + zhv[i+1]) / 2 for i in 1:length(zhv)-1] : Vector{T}(z)
    SoilGrid{T}(zv, zhv)
end
Base.length(g::SoilGrid) = length(g.z)

"""
    clm5_soil_grid(n = 20) -> SoilGrid

The CLM5 soil layers (Lawrence et al. 2019; CLM5.0 Technical Note
§2.2.3, Table 2.2.3): thicknesses

    Δz_j = 0.02 j                 j ≤ 4
         = 0.08 + 0.04 (j - 4)     5 ≤ j ≤ 13
         = 0.44 + 0.10 (j - 13)    j ≥ 14

with nodes at the layer centres. The 20 hydrologically active layers
reach 8.6 m. `n < 20` truncates the column (a shallower depth to
bedrock); `n > 20` continues the last rule.
"""
function clm5_soil_grid(n::Integer = 20)
    n >= 1 || throw(ArgumentError("clm5_soil_grid: n ≥ 1"))
    dz = map(1:n) do j
        j <= 4 ? 0.02j : j <= 13 ? 0.08 + 0.04(j - 4) : 0.44 + 0.10(j - 13)
    end
    SoilGrid(vcat(0.0, cumsum(dz)))
end

"""
    clm_exponential_soil_grid(n = 10; f_s = 0.025) -> SoilGrid

The exponentially thickening layers of CLM2–CLM4.5 (Oleson et al. 2004,
eq. 6.1-6.3): node depths ``z_i = f_s [\\exp(0.5(i - 0.5)) - 1]``,
interfaces midway between nodes, and the last layer as thick as its
node spacing (``Δz_N = z_N - z_{N-1}``). With the defaults the bottom
node is at 2.865 m and the column is 3.433 m deep.
"""
function clm_exponential_soil_grid(n::Integer = 10; f_s::Real = 0.025)
    n >= 2 || throw(ArgumentError("clm_exponential_soil_grid: n ≥ 2"))
    z = [f_s * (exp(0.5 * (i - 0.5)) - 1) for i in 1:n]
    zh = zeros(n + 1)
    for i in 1:n-1
        zh[i+1] = (z[i] + z[i+1]) / 2
    end
    zh[n+1] = z[n] + (z[n] - z[n-1]) / 2
    SoilGrid(zh; z = z)
end

"""
    uniform_soil_grid(depth, n) -> SoilGrid

`n` equal layers down to `depth` (m), nodes at the layer centres — the
grid used for the analytic benchmarks.
"""
uniform_soil_grid(depth::Real, n::Integer) = SoilGrid(collect(range(0.0, float(depth); length = n + 1)))

# =========================================================================
# 2. Hydraulic functions
# =========================================================================

"""
    AbstractHydraulicModel

Water-retention and hydraulic-conductivity functions of one soil. All
potentials are pressure heads in metres of water (negative unsaturated),
conductivities in m s⁻¹, water contents volumetric (m³ m⁻³). A concrete
model implements

    soil_water_content(h, ψ)          # θ(ψ), the retention curve
    soil_matric_potential(h, θ)       # its inverse, ψ(θ)
    soil_hydraulic_conductivity(h, ψ) # K(ψ) at the reference temperature
    soil_water_capacity(h, ψ)         # C = dθ/dψ
    soil_porosity(h), residual_water_content(h),
    air_entry_potential(h), saturated_conductivity(h)

For ``ψ ≥`` `air_entry_potential(h)` the soil is saturated:
``θ = θ_s``, ``K = K_s``, ``C = 0``.
"""
abstract type AbstractHydraulicModel end

"Volumetric water content ``θ(ψ)`` of the retention curve (m³ m⁻³)."
function soil_water_content end
"Matric potential ``ψ(θ)`` (m), the inverse of the retention curve; returns the air-entry potential for ``θ ≥ θ_s``."
function soil_matric_potential end
"Unsaturated hydraulic conductivity ``K(ψ)`` (m s⁻¹) at the reference temperature, without ice impedance."
function soil_hydraulic_conductivity end
"Specific moisture capacity ``C(ψ) = dθ/dψ`` (m⁻¹)."
function soil_water_capacity end
"Saturated water content ``θ_s`` (m³ m⁻³)."
soil_porosity(h::AbstractHydraulicModel) = h.θ_s
"Residual water content ``θ_r`` (m³ m⁻³); zero for Clapp–Hornberger."
residual_water_content(h::AbstractHydraulicModel) = h.θ_r
"Saturated hydraulic conductivity ``K_s`` (m s⁻¹)."
saturated_conductivity(h::AbstractHydraulicModel) = h.K_s
"Matric potential (m) above which the soil is saturated (≤ 0)."
air_entry_potential(::AbstractHydraulicModel) = 0.0

"""
    soil_water_diffusivity(h, ψ) -> D (m² s⁻¹)

Soil-water diffusivity ``D = K/C = K\\, dψ/dθ``; `Inf` where ``C = 0``.
"""
soil_water_diffusivity(h::AbstractHydraulicModel, ψ::Real) =
    soil_hydraulic_conductivity(h, ψ) / soil_water_capacity(h, ψ)

# ---- Clapp–Hornberger -----------------------------------------------------

"""
    ClappHornberger(; θ_s, ψ_s, b, K_s)

Clapp & Hornberger (1978) soil, the Campbell (1974) power laws used by
CLM:

    ψ = ψ_s (θ/θ_s)^{-b},     K = K_s (θ/θ_s)^{2b+3}

`θ_s` porosity, `ψ_s < 0` the saturated (air-entry) matric potential (m),
`b` the pore-size index, `K_s` (m s⁻¹). No residual water content.
"""
struct ClappHornberger{T<:AbstractFloat} <: AbstractHydraulicModel
    θ_s::T
    ψ_s::T
    b::T
    K_s::T
    function ClappHornberger{T}(θ_s, ψ_s, b, K_s) where {T}
        (0 < θ_s < 1 && ψ_s < 0 && b > 0 && K_s > 0) ||
            throw(ArgumentError("ClappHornberger: need 0 < θ_s < 1, ψ_s < 0, b > 0, K_s > 0"))
        new{T}(θ_s, ψ_s, b, K_s)
    end
end
ClappHornberger(θ_s::Real, ψ_s::Real, b::Real, K_s::Real) =
    ClappHornberger{Float64}(θ_s, ψ_s, b, K_s)
ClappHornberger(; θ_s, ψ_s, b, K_s) = ClappHornberger(θ_s, ψ_s, b, K_s)

residual_water_content(::ClappHornberger) = 0.0
air_entry_potential(h::ClappHornberger) = h.ψ_s

soil_water_content(h::ClappHornberger, ψ::Real) =
    ψ >= h.ψ_s ? h.θ_s : h.θ_s * (ψ / h.ψ_s)^(-1 / h.b)
function soil_matric_potential(h::ClappHornberger, θ::Real)
    θ >= h.θ_s && return h.ψ_s
    θ > 0 || return -Inf
    return h.ψ_s * (θ / h.θ_s)^(-h.b)
end
soil_hydraulic_conductivity(h::ClappHornberger, ψ::Real) =
    ψ >= h.ψ_s ? h.K_s : h.K_s * (ψ / h.ψ_s)^(-(2h.b + 3) / h.b)
soil_water_capacity(h::ClappHornberger, ψ::Real) =
    ψ >= h.ψ_s ? 0.0 : -h.θ_s / (h.b * h.ψ_s) * (ψ / h.ψ_s)^(-1 / h.b - 1)

# ---- van Genuchten–Mualem ---------------------------------------------------

"""
    VanGenuchtenMualem(; θ_r, θ_s, α, n, K_s, l = 0.5, ψ_e = 0)

van Genuchten (1980) retention with the Mualem (1976) conductivity:

    S_e = [1 + (α|ψ|)^n]^{-m},  m = 1 - 1/n,   θ = θ_r + (θ_s - θ_r) S_e
    K   = K_s S_e^l [1 - (1 - S_e^{1/m})^m]²

`α` in m⁻¹, `K_s` in m s⁻¹, tortuosity `l` = 0.5 (Mualem).

With an air-entry potential `ψ_e < 0` the curves are those of the
modified model of Vogel et al. (2001) / Ippisch et al. (2006): the soil
is saturated for ``ψ ≥ ψ_e``, and with ``S_c = [1 + (α|ψ_e|)^n]^{-m}``

    S_e = [1 + (α|ψ|)^n]^{-m} / S_c,
    K   = K_s S_e^l {[1 - (1 - (S_e S_c)^{1/m})^m] / [1 - (1 - S_c^{1/m})^m]}²

For ``n < 2`` the unmodified K has an infinite slope at saturation — a
non-physical cusp (Ippisch et al. 2006) that makes Richards solvers
chatter near saturation; ``ψ_e`` of a few centimetres removes it. The
retention curve moves by at most ``(θ_s - θ_r)(1 - S_c)``, reached at
``ψ_e``: for ``ψ_e = -2`` cm that is < 0.6 % of the pore space for loams
and clays but 1.3-2.3 % for the Carsel-Parrish sands, loamy sands, sandy
loams and sandy clay loams. `ψ_e = 0` is the classical model.

  * Vogel, T., van Genuchten, M. Th. & Císlerová, M. (2001). Effect of
    the shape of the soil hydraulic functions near saturation on
    variably-saturated flow predictions. *Adv. Water Resour.*, 24, 133-144.
  * Ippisch, O., Vogel, H.-J. & Bastian, P. (2006). Validity limits for
    the van Genuchten–Mualem model and implications for parameter
    estimation and numerical simulation. *Adv. Water Resour.*, 29,
    1780-1789.
"""
struct VanGenuchtenMualem{T<:AbstractFloat} <: AbstractHydraulicModel
    θ_r::T
    θ_s::T
    α::T
    n::T
    K_s::T
    l::T
    ψ_e::T
    S_c::T     # [1 + (α|ψ_e|)^n]^{-m}, = 1 for the classical model
    function VanGenuchtenMualem{T}(θ_r, θ_s, α, n, K_s, l, ψ_e) where {T}
        (0 <= θ_r < θ_s < 1 && α > 0 && n > 1 && K_s > 0 && ψ_e <= 0) ||
            throw(ArgumentError("VanGenuchtenMualem: need 0 ≤ θ_r < θ_s < 1, α > 0, n > 1, K_s > 0, ψ_e ≤ 0"))
        S_c = (1 + (α * -ψ_e)^n)^(-(1 - 1 / n))
        new{T}(θ_r, θ_s, α, n, K_s, l, ψ_e, S_c)
    end
end
VanGenuchtenMualem(θ_r::Real, θ_s::Real, α::Real, n::Real, K_s::Real, l::Real = 0.5,
                   ψ_e::Real = 0.0) =
    VanGenuchtenMualem{Float64}(θ_r, θ_s, α, n, K_s, l, ψ_e)
VanGenuchtenMualem(; θ_r, θ_s, α, n, K_s, l = 0.5, ψ_e = 0.0) =
    VanGenuchtenMualem(θ_r, θ_s, α, n, K_s, l, ψ_e)

air_entry_potential(h::VanGenuchtenMualem) = h.ψ_e
_vg_m(h::VanGenuchtenMualem) = 1 - 1 / h.n
# the unscaled van Genuchten saturation [1 + (α|ψ|)^n]^{-m} (= S_e S_c)
_vg_Γ(h::VanGenuchtenMualem, ψ::Real) = (1 + (h.α * -ψ)^h.n)^(-_vg_m(h))
_vg_Se(h::VanGenuchtenMualem, ψ::Real) =
    ψ >= h.ψ_e ? one(float(ψ)) : _vg_Γ(h, ψ) / h.S_c

soil_water_content(h::VanGenuchtenMualem, ψ::Real) = h.θ_r + (h.θ_s - h.θ_r) * _vg_Se(h, ψ)
function soil_matric_potential(h::VanGenuchtenMualem, θ::Real)
    θ >= h.θ_s && return h.ψ_e
    θ > h.θ_r || return -Inf
    Γ = (θ - h.θ_r) / (h.θ_s - h.θ_r) * h.S_c
    return -(Γ^(-1 / _vg_m(h)) - 1)^(1 / h.n) / h.α
end
function soil_hydraulic_conductivity(h::VanGenuchtenMualem, ψ::Real)
    ψ >= h.ψ_e && return h.K_s
    m = _vg_m(h)
    Γ = _vg_Γ(h, ψ)
    Se = Γ / h.S_c
    f(x) = 1 - (1 - x^(1 / m))^m
    return h.K_s * Se^h.l * (f(Γ) / f(h.S_c))^2
end
function soil_water_capacity(h::VanGenuchtenMualem, ψ::Real)
    ψ >= h.ψ_e && return 0.0
    m = _vg_m(h)
    x = (h.α * -ψ)^h.n
    return (h.θ_s - h.θ_r) / h.S_c * h.α * m * h.n * (h.α * -ψ)^(h.n - 1) * (1 + x)^(-m - 1)
end

# ---- Gardner exponential ----------------------------------------------------

"""
    GardnerExponential(; θ_r, θ_s, α, K_s)

Gardner (1958) exponential soil with the retention curve of the same
form (Philip 1969; Srivastava & Yeh 1991):

    K = K_s e^{αψ},   θ = θ_r + (θ_s - θ_r) e^{αψ}     (ψ ≤ 0)

Because θ is linear in K, Richards' equation becomes *linear* in K and
has closed-form solutions — which is why this soil is used for the exact
benchmarks. Its diffusivity ``D = K_s/(α(θ_s - θ_r))`` is constant.
"""
struct GardnerExponential{T<:AbstractFloat} <: AbstractHydraulicModel
    θ_r::T
    θ_s::T
    α::T
    K_s::T
    function GardnerExponential{T}(θ_r, θ_s, α, K_s) where {T}
        (0 <= θ_r < θ_s < 1 && α > 0 && K_s > 0) ||
            throw(ArgumentError("GardnerExponential: need 0 ≤ θ_r < θ_s < 1, α > 0, K_s > 0"))
        new{T}(θ_r, θ_s, α, K_s)
    end
end
GardnerExponential(θ_r::Real, θ_s::Real, α::Real, K_s::Real) =
    GardnerExponential{Float64}(θ_r, θ_s, α, K_s)
GardnerExponential(; θ_r, θ_s, α, K_s) = GardnerExponential(θ_r, θ_s, α, K_s)

soil_water_content(h::GardnerExponential, ψ::Real) =
    h.θ_r + (h.θ_s - h.θ_r) * exp(h.α * min(ψ, zero(ψ)))
function soil_matric_potential(h::GardnerExponential, θ::Real)
    θ >= h.θ_s && return 0.0
    θ > h.θ_r || return -Inf
    return log((θ - h.θ_r) / (h.θ_s - h.θ_r)) / h.α
end
soil_hydraulic_conductivity(h::GardnerExponential, ψ::Real) = h.K_s * exp(h.α * min(ψ, zero(ψ)))
soil_water_capacity(h::GardnerExponential, ψ::Real) =
    ψ >= 0 ? 0.0 : (h.θ_s - h.θ_r) * h.α * exp(h.α * ψ)

# ---- Texture tables ---------------------------------------------------------

"""
    CLAPP_HORNBERGER_1978

Clapp & Hornberger (1978), Table 2: mean parameters for the 11 USDA
texture classes — porosity `θ_s`, saturated conductivity `K_s`
(**cm h⁻¹**; the paper gives cm s⁻¹), saturated matric potential `ψ_s`
(cm, as a positive suction) and `b`. Values as reproduced in Tarboton
(2003, *Rainfall-Runoff Processes*, ch. 4, Table 1, `K_s` in cm h⁻¹).
Use [`clapp_hornberger_texture`](@ref) for an SI model.
"""
const CLAPP_HORNBERGER_1978 = (
    sand            = (θ_s = 0.395, K_s = 63.36, ψ_s = 12.1, b = 4.05),
    loamy_sand      = (θ_s = 0.410, K_s = 56.16, ψ_s = 9.0,  b = 4.38),
    sandy_loam      = (θ_s = 0.435, K_s = 12.49, ψ_s = 21.8, b = 4.90),
    silt_loam       = (θ_s = 0.485, K_s = 2.59, ψ_s = 78.6, b = 5.30),
    loam            = (θ_s = 0.451, K_s = 2.50, ψ_s = 47.8, b = 5.39),
    sandy_clay_loam = (θ_s = 0.420, K_s = 2.27, ψ_s = 29.9, b = 7.12),
    silty_clay_loam = (θ_s = 0.477, K_s = 0.612, ψ_s = 35.6, b = 7.75),
    clay_loam       = (θ_s = 0.476, K_s = 0.882, ψ_s = 63.0, b = 8.52),
    sandy_clay      = (θ_s = 0.426, K_s = 0.781, ψ_s = 15.3, b = 10.4),
    silty_clay      = (θ_s = 0.492, K_s = 0.371, ψ_s = 49.0, b = 10.4),
    clay            = (θ_s = 0.482, K_s = 0.461, ψ_s = 40.5, b = 11.4),
)

"""
    CARSEL_PARRISH_1988

Carsel & Parrish (1988), Tables 2-5: mean van Genuchten parameters for
the 12 USDA texture classes, in the paper's units — `θ_r`, `θ_s`,
`α` (cm⁻¹), `n` and `K_s` (cm h⁻¹). Use [`carsel_parrish_texture`](@ref)
for an SI model.
"""
const CARSEL_PARRISH_1988 = (
    sand            = (θ_r = 0.045, θ_s = 0.43, α = 0.145, n = 2.68, K_s = 29.70),
    loamy_sand      = (θ_r = 0.057, θ_s = 0.41, α = 0.124, n = 2.28, K_s = 14.59),
    sandy_loam      = (θ_r = 0.065, θ_s = 0.41, α = 0.075, n = 1.89, K_s = 4.42),
    loam            = (θ_r = 0.078, θ_s = 0.43, α = 0.036, n = 1.56, K_s = 1.04),
    silt            = (θ_r = 0.034, θ_s = 0.46, α = 0.016, n = 1.37, K_s = 0.25),
    silt_loam       = (θ_r = 0.067, θ_s = 0.45, α = 0.020, n = 1.41, K_s = 0.45),
    sandy_clay_loam = (θ_r = 0.100, θ_s = 0.39, α = 0.059, n = 1.48, K_s = 1.31),
    clay_loam       = (θ_r = 0.095, θ_s = 0.41, α = 0.019, n = 1.31, K_s = 0.26),
    silty_clay_loam = (θ_r = 0.089, θ_s = 0.43, α = 0.010, n = 1.23, K_s = 0.07),
    sandy_clay      = (θ_r = 0.100, θ_s = 0.38, α = 0.027, n = 1.23, K_s = 0.12),
    silty_clay      = (θ_r = 0.070, θ_s = 0.36, α = 0.005, n = 1.09, K_s = 0.02),
    clay            = (θ_r = 0.068, θ_s = 0.38, α = 0.008, n = 1.09, K_s = 0.20),
)

function _texture_row(table, texture::Symbol, name)
    haskey(table, texture) ||
        throw(ArgumentError("$name: unknown texture :$texture; one of $(keys(table))"))
    return table[texture]
end

"""
    clapp_hornberger_texture(texture::Symbol) -> ClappHornberger

SI [`ClappHornberger`](@ref) soil for a USDA texture class (`:sand`,
`:loam`, `:clay`, … — the keys of [`CLAPP_HORNBERGER_1978`](@ref)).
"""
function clapp_hornberger_texture(texture::Symbol)
    r = _texture_row(CLAPP_HORNBERGER_1978, texture, "clapp_hornberger_texture")
    ClappHornberger(r.θ_s, -r.ψ_s / 100, r.b, r.K_s / 360000)
end

"""
    carsel_parrish_texture(texture::Symbol; ψ_e = -0.02) -> VanGenuchtenMualem

SI [`VanGenuchtenMualem`](@ref) soil for a USDA texture class (the keys
of [`CARSEL_PARRISH_1988`](@ref), which also includes `:silt`). The
default air-entry potential of −2 cm is the value Vogel et al. (2001)
recommend to remove the near-saturation conductivity cusp; pass
`ψ_e = 0` for the classical published curves.
"""
function carsel_parrish_texture(texture::Symbol; ψ_e::Real = -0.02)
    r = _texture_row(CARSEL_PARRISH_1988, texture, "carsel_parrish_texture")
    VanGenuchtenMualem(r.θ_r, r.θ_s, 100 * r.α, r.n, r.K_s / 360000, 0.5, ψ_e)
end

"""
    cosby_clapp_hornberger(sand, clay) -> ClappHornberger

Clapp–Hornberger parameters from sand and clay percentages by the
univariate regressions of Cosby et al. (1984, Table 4), as adopted by
CLM (Oleson et al. 2013, eqs. 7.85-7.88):

    θ_s = 0.489 - 0.00126 %sand
    ψ_s = -10 × 10^{1.88 - 0.0131 %sand}  mm
    b   = 2.91 + 0.159 %clay
    K_s = 0.0070556 × 10^{-0.884 + 0.0153 %sand}  mm s⁻¹
"""
function cosby_clapp_hornberger(sand::Real, clay::Real)
    (0 <= sand <= 100 && 0 <= clay <= 100 && sand + clay <= 100) ||
        throw(ArgumentError("cosby_clapp_hornberger: percentages in [0, 100] with sand + clay ≤ 100"))
    θ_s = 0.489 - 0.00126 * sand
    ψ_s = -10.0 * 10^(1.88 - 0.0131 * sand) / 1000
    b = 2.91 + 0.159 * clay
    K_s = 0.0070556 * 10^(-0.884 + 0.0153 * sand) / 1000
    ClappHornberger(θ_s, ψ_s, b, K_s)
end

"""
    parlange_sorptivity(h, ψ_i; ψ_0 = 0, n = 4000) -> S (m s^-1/2)

Sorptivity of soil `h` wetted from initial potential `ψ_i` by a
boundary held at `ψ_0`, from Parlange's (1975) approximation

    S² = ∫_{θ_i}^{θ_0} (θ_0 + θ - 2θ_i) D(θ) dθ = ∫_{ψ_i}^{ψ_0} (θ_0 + θ(ψ) - 2θ_i) K(ψ) dψ

(the second form avoids the singular D near saturation, and it keeps
the saturated segment ``ψ_e < ψ < ψ_0`` of a soil with an air-entry
potential, where θ is constant but water still flows: a δ-function in
D(θ) that the first form would miss). It lies midway
between the rigorous bounds ``2∫(θ - θ_i)D\\,dθ ≤ S² ≤ 2(θ_0 - θ_i)∫D\\,dθ``.
It is built for the steeply rising diffusivities of real soils, where it
is within a few per cent of the exact sorptivity; for a *constant* D it
overestimates by 8.5 % (``\\sqrt{3/2}`` against ``2/\\sqrt{π}``). Integrated with the
trapezoid rule on a grid uniform in ``\\log(ψ_0 - ψ + ε)``.
"""
function parlange_sorptivity(h::AbstractHydraulicModel, ψ_i::Real;
                             ψ_0::Real = 0.0, n::Integer = 4000)
    ψ_i < ψ_0 || throw(ArgumentError("parlange_sorptivity: need ψ_i < ψ_0"))
    θi, θ0 = soil_water_content(h, ψ_i), soil_water_content(h, ψ_0)
    # grid clustered near ψ_0, where K is largest
    ε = 1e-6 * (ψ_0 - ψ_i)
    s = range(log(ε), log(ψ_0 - ψ_i + ε); length = n)
    ψs = [ψ_0 + ε - exp(x) for x in s]
    f(ψ) = (θ0 + soil_water_content(h, ψ) - 2θi) * soil_hydraulic_conductivity(h, ψ)
    S2 = (ψ_0 - ψs[1]) * f(ψ_0)        # the sliver [ψs[1], ψ_0]
    for k in 1:n-1
        S2 += (ψs[k] - ψs[k+1]) * (f(ψs[k]) + f(ψs[k+1])) / 2
    end
    return sqrt(S2)
end

# =========================================================================
# 3. Temperature and ice
# =========================================================================

"""
    water_dynamic_viscosity(T; c = DEFAULT_SURFACE_CONSTANTS) -> μ (Pa s)

Dynamic viscosity of liquid water at atmospheric pressure from the
four-term correlation of Kestin, Sokolov & Wakeham (1978) relative to
the 20 °C value, as tabulated with the viscosity of water in the CRC
Handbook of Chemistry and Physics:

    log₁₀(μ/μ₂₀) = (20 - t)/(t + 96) [1.2378 - 1.303×10⁻³ (20 - t)
                   + 3.06×10⁻⁶ (20 - t)² + 2.55×10⁻⁸ (20 - t)³]

with `t` in °C (from `T` K and the contract's freezing point) and
``μ_{20} = 1.0016`` mPa s. It reproduces the CRC values at 0, 10, 20,
25, 40, 60 and 80 °C to better than 0.1 % (largest deviation 0.075 %,
at 10 °C) and extrapolates smoothly into weakly supercooled water.
"""
function water_dynamic_viscosity(T::Real; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    t = T - c.thermo.T_freeze
    Δ = 20 - t
    return 1.0016e-3 * 10^(Δ / (t + 96) * (1.2378 - 1.303e-3 * Δ + 3.06e-6 * Δ^2 + 2.55e-8 * Δ^3))
end

"""
    viscosity_conductivity_factor(T, T_ref = 293.15; c = DEFAULT_SURFACE_CONSTANTS)

Ratio ``K(T)/K(T_{ref}) = μ(T_{ref})/μ(T)``: hydraulic conductivity is
``k ρ g/μ`` with an intrinsic permeability `k` that does not depend on
temperature, and the density change (< 0.5 % over 0–40 °C) is neglected.
Soil hydraulic tables are laboratory values near 20 °C, hence the
default reference. Cold soil conducts less: the factor is 0.56 at 0 °C.
"""
viscosity_conductivity_factor(T::Real, T_ref::Real = 293.15;
                              c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    water_dynamic_viscosity(T_ref; c = c) / water_dynamic_viscosity(T; c = c)

"""
    AbstractIceImpedance

How pore ice reduces the hydraulic conductivity of a layer. A subtype
implements `ice_impedance_factor(imp, θ_ice, θ_s) ∈ (0, 1]`, the factor
multiplying ``K``. The default [`NoIceImpedance`](@ref) returns 1; the
freeze–thaw model (issue #16) adds e.g. the CLM power-law impedance
``10^{-Ω F_{ice}}`` (Swenson et al. 2012).
"""
abstract type AbstractIceImpedance end

"No reduction of conductivity by ice (factor 1) — the default until soil freezing is modelled."
struct NoIceImpedance <: AbstractIceImpedance end

"""
    ice_impedance_factor(imp::AbstractIceImpedance, θ_ice, θ_s) -> factor ∈ (0, 1]

Multiplier on the hydraulic conductivity of a layer with ice content
`θ_ice` and porosity `θ_s`.
"""
ice_impedance_factor(::NoIceImpedance, θ_ice::Real, θ_s::Real) = 1.0

# =========================================================================
# 4. Boundary conditions
# =========================================================================

"""
    AbstractSoilTopBoundary

Upper boundary condition of a [`SoilWaterColumn`](@ref):
[`AtmosphericTop`](@ref) (the land-surface condition) or
[`HeadTop`](@ref) (a prescribed surface potential, for benchmarks and
ponded surfaces).
"""
abstract type AbstractSoilTopBoundary end

"""
    AtmosphericTop(; ψ_pond = 0, ψ_dry = -1e5)

The land-surface boundary: the net water supply ``q = P - E`` (water
input minus evaporation) is imposed as a flux as long as the soil can
take or give it. If the surface would have to exceed `ψ_pond` (m) to
absorb the supply, the condition switches to the head ``ψ = ψ_pond`` and
the excess becomes infiltration-excess (Hortonian) **runoff**; if it would
have to fall below `ψ_dry` to deliver the evaporative demand, it switches
to ``ψ = ψ_dry`` and evaporation becomes **soil-limited** — the
atmospheric boundary condition of e.g. HYDRUS (Šimůnek et al. 2008).
The default `ψ_dry` is CLM's minimum soil potential (−10⁸ mm).
"""
struct AtmosphericTop{T<:AbstractFloat} <: AbstractSoilTopBoundary
    ψ_pond::T
    ψ_dry::T
end
AtmosphericTop(; ψ_pond::Real = 0.0, ψ_dry::Real = -1e5) =
    AtmosphericTop{Float64}(ψ_pond, ψ_dry)

"""
    HeadTop(ψ)

Prescribed matric potential `ψ` (m) at the soil surface (depth 0), e.g.
``ψ = 0`` for a ponded surface with negligible water depth. The water
input and evaporation passed to [`step_soil_water!`](@ref) are ignored;
the infiltration is whatever the soil takes up.
"""
struct HeadTop{T<:AbstractFloat} <: AbstractSoilTopBoundary
    ψ::T
end
HeadTop(ψ::Real) = HeadTop{Float64}(ψ)

"""
    AbstractSoilBottomBoundary

Lower boundary condition of a [`SoilWaterColumn`](@ref). A subtype
implements

    bottom_flux_linearization(bc, col, ψ_N, K_N) -> (a, b)

giving the downward flux out of the column's base as ``q = a + b\\,ψ_N``
in terms of the bottom node potential ``ψ_N`` (``b ≥ 0``; `K_N` is the
conductivity of the bottom layer at the current Picard iterate), and may
implement `finalize_bottom!(bc, col, q, dt)`, called with the converged
flux after every accepted sub-step — the hook through which a water
table / aquifer (#17) updates its own storage. Provided:
[`FreeDrainage`](@ref) (default), [`ZeroFluxBottom`](@ref),
[`HeadBottom`](@ref).
"""
abstract type AbstractSoilBottomBoundary end

"Unit-gradient gravity drainage, ``q = K(ψ_N)`` — CLM's default lower boundary."
struct FreeDrainage <: AbstractSoilBottomBoundary end
"Impermeable base, ``q = 0`` (bedrock)."
struct ZeroFluxBottom <: AbstractSoilBottomBoundary end
"""
    HeadBottom(ψ)

Prescribed potential `ψ` (m) at the base of the column (e.g. ``ψ = 0``: a
fixed water table at the bottom interface).
"""
struct HeadBottom{T<:AbstractFloat} <: AbstractSoilBottomBoundary
    ψ::T
end
HeadBottom(ψ::Real) = HeadBottom{Float64}(ψ)

"""
    bottom_flux_linearization(bc, col, ψ_N, K_N) -> (a, b)
    bottom_flux_linearization(bc, col, ψ_N, K_N, dt) -> (a, b)

Downward flux through the base of the column, ``q = a + b ψ_N`` (m s⁻¹).
The solver calls the five-argument form with the sub-step length `dt`
(s), which a boundary with its own storage (the [`UnconfinedAquifer`](@ref))
needs to couple implicitly; by default it forwards to the four-argument
form. See [`AbstractSoilBottomBoundary`](@ref).
"""
function bottom_flux_linearization end
bottom_flux_linearization(bc::AbstractSoilBottomBoundary, col, ψN, KN, dt) =
    bottom_flux_linearization(bc, col, ψN, KN)
bottom_flux_linearization(::FreeDrainage, col, ψN, KN) = (col.gravity * KN, zero(KN))
bottom_flux_linearization(::ZeroFluxBottom, col, ψN, KN) = (zero(KN), zero(KN))
function bottom_flux_linearization(bc::HeadBottom, col, ψN, KN)
    N = length(col.grid)
    d = col.grid.zh[end] - col.grid.z[N]
    Kb = _interface_K(col, KN, _layer_K(col, N, bc.ψ))
    return (Kb * (col.gravity - bc.ψ / d), Kb / d)
end

"""
    finalize_bottom!(bc, col, q, dt)

Called after each accepted sub-step with the converged basal flux `q`
(m s⁻¹, downward) over `dt` seconds. No-op for the stateless boundaries.
"""
finalize_bottom!(::AbstractSoilBottomBoundary, col, q, dt) = nothing

# =========================================================================
# 5. The column and its Richards solver
# =========================================================================

"""
    SoilWaterBudget

Cumulative water budget of a [`SoilWaterColumn`](@ref) since construction
(metres of liquid water): `infiltration` (net flux through the surface,
downward positive; negative when the soil evaporates), `runoff`
(infiltration-excess), `evaporation` (actually delivered), `drainage`
(through the base, downward positive), `extraction` (layer sinks, e.g.
root uptake), and `time` (s).
"""
mutable struct SoilWaterBudget{T<:AbstractFloat}
    infiltration::T
    runoff::T
    evaporation::T
    drainage::T
    extraction::T
    time::T
end
SoilWaterBudget{T}() where {T} = SoilWaterBudget{T}(0, 0, 0, 0, 0, 0)

"""
    SoilWaterStep

What one call of [`step_soil_water!`](@ref) did (metres of water over
the step): `infiltration`, `runoff`, `evaporation`, `drainage`,
`extraction`, `storage_change`, plus `substeps` and total Picard
`iterations`. Water balance:
``storage\\_change = infiltration - drainage - extraction`` to round-off.
"""
struct SoilWaterStep{T<:AbstractFloat}
    infiltration::T
    runoff::T
    evaporation::T
    drainage::T
    extraction::T
    storage_change::T
    substeps::Int
    iterations::Int
end

struct _SoilWorkspace{T}
    ψm::Vector{T}; ψnew::Vector{T}; θm::Vector{T}; θnew::Vector{T}
    C::Vector{T}; K::Vector{T}; Kf::Vector{T}
    a::Vector{T}; b::Vector{T}; c::Vector{T}; d::Vector{T}
    cp::Vector{T}; dp::Vector{T}
    ψsave::Vector{T}; θsave::Vector{T}
    fK::Vector{T}          # temperature × ice factor on K, per layer
end
_SoilWorkspace{T}(N) where {T} = _SoilWorkspace{T}((zeros(T, N) for _ in 1:16)...)

"""
    SoilWaterColumn(grid, hydraulics; ψ = -1.0, θ = nothing,
                    top = AtmosphericTop(), bottom = FreeDrainage(),
                    ice_impedance = NoIceImpedance(),
                    T_soil = 293.15, T_ref = 293.15, temperature_dependent = true,
                    specific_storage = 1e-5, gravity = true,
                    tol = 1e-7, tol_ψ = 1e-4, maxiter = 40, dt_min = 1e-4,
                    dt_initial = 60, interface_mean = :logarithmic,
                    constants = DEFAULT_SURFACE_CONSTANTS)

A soil column of `length(grid)` layers solving Richards' equation.
`hydraulics` is one [`AbstractHydraulicModel`](@ref) or a vector of one
per layer (layered soil). Initial state from a matric potential `ψ` (m)
or a water content `θ` (scalars or per-layer vectors; `θ` wins).

# State (fields)

| field     | meaning                                                        |
|:--------- |:-------------------------------------------------------------- |
| `ψ`       | matric potential of each layer, m                              |
| `θ_liq`   | liquid water content, m³ m⁻³ — the conserved prognostic        |
| `θ_ice`   | ice content, m³ m⁻³ (set by [`SoilThermalColumn`](@ref), #16)  |
| `T_soil`  | layer temperature, K (prescribed, or by [`SoilThermalColumn`](@ref)) |
| `top`, `bottom` | boundary conditions; a stateful `bottom` (#17's water table) lives here |
| `budget`  | cumulative [`SoilWaterBudget`](@ref)                           |

The layer conductivity is
``K_i = K(ψ_i)\\, μ(T_{ref})/μ(T_i)\\, f_{ice}(θ_{ice,i})``
([`viscosity_conductivity_factor`](@ref), [`ice_impedance_factor`](@ref));
`temperature_dependent = false` drops the viscosity factor. In a layer
with pore ice the retention curve applies to the *total* water
``θ_{liq} + (ρ_{ice}/ρ_w)θ_{ice}`` (after CLM4.5 and later, Swenson et al.
2012, which use the liquid-plus-ice *volume*; the liquid equivalent is
used here so that the 9 % expansion on freezing does not register as
over-saturation): phase change leaves ψ unchanged, and ice blocks flow through the
impedance factor rather than through a suction. Saturated
layers (``ψ ≥ ψ_{ae}``) store ``θ_s + S_s (ψ - ψ_{ae})`` with a small
elastic `specific_storage` ``S_s`` (m⁻¹), which keeps the problem
well-posed when the whole column is saturated. `gravity = false` gives
horizontal flow (absorption benchmarks).

Numerics: see [`step_soil_water!`](@ref). The Picard iteration has
converged when no unsaturated layer's water content changes by more than
`tol` and no saturated layer's potential by more than `tol_ψ` (m) — the
HYDRUS criteria. `interface_mean` sets the conductivity between two
nodes: `:logarithmic` (default), ``(K_2 - K_1)/\\ln(K_2/K_1)``, which is
exact for a conductivity exponential in ψ and gives second-order
convergence on the Gardner benchmark; `:arithmetic` (Celia et al. 1990)
converges only at first order when the boundary gradients are steep;
`:geometric` is also available.
"""
mutable struct SoilWaterColumn{T<:AbstractFloat, H<:AbstractHydraulicModel,
                               TB<:AbstractSoilTopBoundary, BB<:AbstractSoilBottomBoundary,
                               I<:AbstractIceImpedance, SC<:SurfaceConstants}
    grid::SoilGrid{T}
    hydraulics::Vector{H}
    ψ::Vector{T}
    θ_liq::Vector{T}
    θ_ice::Vector{T}
    T_soil::Vector{T}
    top::TB
    bottom::BB
    ice_impedance::I
    T_ref::T
    temperature_dependent::Bool
    specific_storage::T
    gravity::T
    tol::T
    tol_ψ::T
    maxiter::Int
    dt_min::T
    dt_sub::T
    ψ_min::T
    interface_mean::Symbol
    budget::SoilWaterBudget{T}
    constants::SC
    ws::_SoilWorkspace{T}
end

_per_layer(x::AbstractVector, N, name) =
    (length(x) == N || throw(ArgumentError("SoilWaterColumn: `$name` needs $N values")); collect(x))
_per_layer(x, N, name) = fill(x, N)

function SoilWaterColumn(grid::SoilGrid{T}, hydraulics;
                         ψ = -1.0, θ = nothing,
                         top::AbstractSoilTopBoundary = AtmosphericTop(),
                         bottom::AbstractSoilBottomBoundary = FreeDrainage(),
                         ice_impedance::AbstractIceImpedance = NoIceImpedance(),
                         T_soil = 293.15, T_ref::Real = 293.15,
                         temperature_dependent::Bool = true,
                         specific_storage::Real = 1e-5, gravity::Bool = true,
                         tol::Real = 1e-7, tol_ψ::Real = 1e-4, maxiter::Integer = 40,
                         dt_min::Real = 1e-4, dt_initial::Real = 60.0,
                         ψ_min::Real = -1e7, interface_mean::Symbol = :logarithmic,
                         constants::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) where {T}
    N = length(grid)
    hyd = hydraulics isa AbstractHydraulicModel ? fill(hydraulics, N) :
          _per_layer(hydraulics, N, "hydraulics")
    hyd = [h for h in hyd]            # narrow the element type
    specific_storage > 0 || throw(ArgumentError("SoilWaterColumn: specific_storage > 0"))
    interface_mean in (:logarithmic, :arithmetic, :geometric) ||
        throw(ArgumentError("SoilWaterColumn: interface_mean must be :logarithmic, :arithmetic or :geometric"))
    col = SoilWaterColumn(grid, hyd, zeros(T, N), zeros(T, N), zeros(T, N),
                          T.(_per_layer(T_soil, N, "T_soil")), top, bottom, ice_impedance,
                          T(T_ref), temperature_dependent, T(specific_storage),
                          T(gravity ? 1 : 0), T(tol), T(tol_ψ), Int(maxiter), T(dt_min), T(dt_initial),
                          T(ψ_min), interface_mean, SoilWaterBudget{T}(), constants, _SoilWorkspace{T}(N))
    θ === nothing ? set_matric_potential!(col, ψ) : set_water_content!(col, θ)
    return col
end

# ---- constitutive relations of a layer (with storage, temperature, ice) ----

# Liquid water content of layer i at potential ψ. Retention is a function
# of the *total* water, liquid plus the liquid equivalent of the pore ice
# (#16; after CLM4.5, Swenson et al. 2012, which sums ice *volume*), so freezing and thawing move water between
# phases without changing ψ; with no ice this is the retention curve.
function _layer_W(col, i, ψ)
    h = col.hydraulics[i]
    ψae = air_entry_potential(h)
    W = ψ >= ψae ? soil_porosity(h) + col.specific_storage * (ψ - ψae) : soil_water_content(h, ψ)
    return W - _ice_liquid_equivalent(col, i)
end
_ice_liquid_equivalent(col, i) = col.constants.ρ_ice / col.constants.ρ_water * col.θ_ice[i]
function _layer_C(col, i, ψ)
    h = col.hydraulics[i]
    ψ >= air_entry_potential(h) ? col.specific_storage : soil_water_capacity(h, ψ)
end
_layer_K(col, i, ψ) = soil_hydraulic_conductivity(col.hydraulics[i], ψ) * col.ws.fK[i]

function _interface_K(col, K1, K2)
    m = col.interface_mean
    m === :arithmetic && return (K1 + K2) / 2
    m === :geometric && return sqrt(K1 * K2)
    if m === :logarithmic
        (K1 <= 0 || K2 <= 0) && return zero(K1)
        r = K2 / K1
        abs(r - 1) < 1e-6 && return (K1 + K2) / 2
        return (K2 - K1) / log(r)
    end
    throw(ArgumentError("interface_mean must be :arithmetic, :geometric or :logarithmic"))
end

function _refresh_factors!(col)
    c = col.constants
    for i in eachindex(col.ws.fK)
        fT = col.temperature_dependent ?
             viscosity_conductivity_factor(col.T_soil[i], col.T_ref; c = c) : one(col.T_ref)
        fI = ice_impedance_factor(col.ice_impedance, col.θ_ice[i], soil_porosity(col.hydraulics[i]))
        col.ws.fK[i] = fT * fI
    end
    return col
end

"""
    soil_layer_conductivity(col, i) -> K (m s⁻¹)

Current hydraulic conductivity of layer `i`, including the temperature
(viscosity) and ice-impedance factors.
"""
soil_layer_conductivity(col::SoilWaterColumn, i::Integer) =
    (_refresh_factors!(col); _layer_K(col, i, col.ψ[i]))

"""
    set_matric_potential!(col, ψ)

Set the matric potential (scalar or per layer, m) and the consistent
liquid water content.
"""
function set_matric_potential!(col::SoilWaterColumn, ψ)
    ψv = _per_layer(ψ, length(col.grid), "ψ")
    for i in eachindex(col.ψ)
        col.ψ[i] = ψv[i]
        col.θ_liq[i] = _layer_W(col, i, ψv[i])
    end
    return col
end

"""
    set_water_content!(col, θ)

Set the liquid water content (scalar or per layer, m³ m⁻³) and the
consistent matric potential. Contents above porosity are held by the
elastic storage (``ψ = ψ_{ae} + (θ - θ_s)/S_s``). With pore ice, ψ is
that of the total water ``θ + (ρ_{ice}/ρ_w) θ_{ice}``.
"""
function set_water_content!(col::SoilWaterColumn, θ)
    θv = _per_layer(θ, length(col.grid), "θ")
    for i in eachindex(col.ψ)
        h = col.hydraulics[i]
        W = θv[i] + _ice_liquid_equivalent(col, i)       # total water, liquid equivalent
        (θv[i] >= 0 && W > residual_water_content(h)) ||
            throw(ArgumentError("set_water_content!: θ must exceed the residual content of layer $i"))
        col.θ_liq[i] = θv[i]
        col.ψ[i] = W >= soil_porosity(h) ?
                   air_entry_potential(h) + (W - soil_porosity(h)) / col.specific_storage :
                   soil_matric_potential(h, W)
    end
    return col
end

"""
    soil_water_storage(col; c = col.constants) -> m

Total water in the column as a depth of liquid water,
``Σ_i Δz_i (θ_{liq,i} + θ_{ice,i} ρ_{ice}/ρ_w)`` (densities from the
surface shared core).
"""
function soil_water_storage(col::SoilWaterColumn; c::SurfaceConstants = col.constants)
    r = c.ρ_ice / c.ρ_water
    s = zero(eltype(col.ψ))
    for i in eachindex(col.ψ)
        s += col.grid.dz[i] * (col.θ_liq[i] + r * col.θ_ice[i])
    end
    return s
end

"""
    soil_field_capacity(h; c = DEFAULT_SURFACE_CONSTANTS) -> θ_fc

Water content at field capacity, taken as a matric potential of
−33 kPa (−1/3 bar), i.e. ``ψ_{fc} = -33\\,\\mathrm{kPa}/(ρ_w g)`` ≈ −3.37 m
with ``ρ_w`` and ``g`` from the surface shared core (`SurfaceExchange`).
"""
soil_field_capacity(h::AbstractHydraulicModel; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    soil_water_content(h, -33e3 / (c.ρ_water * grav(c)))

"""
    moisture_availability(col::SoilWaterColumn) -> β ∈ [0, 1]

Bare-soil evaporation efficiency of Lee & Pielke (1992) from the top
layer: ``β = \\tfrac14 [1 - \\cos(π θ_1/θ_{fc})]²`` for ``θ_1 < θ_{fc}``,
1 otherwise — the `SurfaceExchange` hook, so a land surface built on the
column gets its evaporative resistance from the soil water.
"""
function moisture_availability(col::SoilWaterColumn)
    θfc = soil_field_capacity(col.hydraulics[1]; c = col.constants)
    θ1 = col.θ_liq[1]
    θ1 >= θfc && return one(θ1)
    return (1 - cos(π * θ1 / θfc))^2 / 4
end

# ---- linear algebra ---------------------------------------------------------

function _thomas!(x, a, b, c, d, cp, dp)
    n = length(d)
    cp[1] = c[1] / b[1]
    dp[1] = d[1] / b[1]
    @inbounds for i in 2:n
        m = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i-1]) / m
    end
    x[n] = dp[n]
    @inbounds for i in n-1:-1:1
        x[i] = dp[i] - cp[i] * x[i+1]
    end
    return x
end

# ---- one backward-Euler step by modified Picard ------------------------------
#
# Mixed form, cell i of thickness Δz_i:
#   Δz_i (θ_i^{n+1} - θ_i^n)/Δt = q_{i-1/2} - q_{i+1/2} - s_i,
#   q_{i+1/2} = K_{i+1/2} [g - (ψ_{i+1} - ψ_i)/(z_{i+1} - z_i)]   (downward),
# linearised as θ^{n+1,m+1} ≈ θ^{n+1,m} + C^m (ψ^{m+1} - ψ^m) with K frozen at
# iterate m (Celia et al. 1990). On convergence θ^{n+1} is formed from the
# flux divergence itself, so storage changes by exactly the boundary fluxes.
# `mode` is :flux (topval = q_top) or :head (topval = ψ_surface).
# Returns (converged, iterations, q_top, q_bottom); results in ws.ψnew/θnew.
function _picard!(col::SoilWaterColumn{T}, dt, mode::Symbol, topval, sink) where {T}
    g, ws, G = col.grid, col.ws, col.gravity
    N = length(g)
    ψm, θn = ws.ψm, col.θ_liq
    copyto!(ψm, col.ψ)
    Kt = at = zero(T)
    for it in 1:col.maxiter
        @inbounds for i in 1:N
            ws.θm[i] = _layer_W(col, i, ψm[i])
            ws.C[i] = _layer_C(col, i, ψm[i])
            ws.K[i] = _layer_K(col, i, ψm[i])
        end
        @inbounds for i in 1:N-1
            ws.Kf[i] = _interface_K(col, ws.K[i], ws.K[i+1])
        end
        if mode === :head
            Kt = _interface_K(col, _layer_K(col, 1, topval), ws.K[1])
            at = Kt / g.z[1]
        end
        αb, βb = bottom_flux_linearization(col.bottom, col, ψm[N], ws.K[N], dt)
        @inbounds for i in 1:N
            r = g.dz[i] / dt
            diag = r * ws.C[i]
            rhs = r * (θn[i] - ws.θm[i] + ws.C[i] * ψm[i])
            sink === nothing || (rhs -= sink[i])
            lo = up = zero(T)
            if i == 1
                if mode === :flux
                    rhs += topval
                else
                    diag += at
                    rhs += Kt * G + at * topval
                end
            else
                a = ws.Kf[i-1] / (g.z[i] - g.z[i-1])
                diag += a; lo = -a; rhs += G * ws.Kf[i-1]
            end
            if i == N
                diag += βb; rhs -= αb
            else
                c = ws.Kf[i] / (g.z[i+1] - g.z[i])
                diag += c; up = -c; rhs -= G * ws.Kf[i]
            end
            ws.a[i], ws.b[i], ws.c[i], ws.d[i] = lo, diag, up, rhs
        end
        _thomas!(ws.ψnew, ws.a, ws.b, ws.c, ws.d, ws.cp, ws.dp)
        # Convergence (as HYDRUS): the water content has stopped changing
        # in unsaturated layers, the potential in saturated ones.
        conv = true
        @inbounds for i in 1:N
            ψn = max(ws.ψnew[i], col.ψ_min)
            ws.ψnew[i] = ψn
            isfinite(ψn) || return (false, it, zero(T), zero(T))
            if ψn >= air_entry_potential(col.hydraulics[i]) &&
               ψm[i] >= air_entry_potential(col.hydraulics[i])
                abs(ψn - ψm[i]) <= col.tol_ψ || (conv = false)
            else
                abs(_layer_W(col, i, ψn) - ws.θm[i]) <= col.tol || (conv = false)
            end
        end
        if conv
            ψ = ws.ψnew
            qtop = mode === :flux ? T(topval) : Kt * (G - (ψ[1] - topval) / g.z[1])
            qprev = qtop
            qbot = zero(T)
            @inbounds for i in 1:N
                qnext = i < N ? ws.Kf[i] * (G - (ψ[i+1] - ψ[i]) / (g.z[i+1] - g.z[i])) :
                                αb + βb * ψ[N]
                s = sink === nothing ? zero(T) : T(sink[i])
                ws.θnew[i] = θn[i] + dt / g.dz[i] * (qprev - qnext - s)
                (ws.θnew[i] >= 0 && ws.θnew[i] + _ice_liquid_equivalent(col, i) >
                                    residual_water_content(col.hydraulics[i])) ||
                    return (false, it, zero(T), zero(T))
                qprev = qnext
            end
            qbot = qprev
            return (true, it, qtop, qbot)
        end
        # Plain Picard can cycle when layers flip across saturation (where
        # the van Genuchten–Mualem K has an infinite slope); after a few
        # iterations the update is under-relaxed.
        ω = it <= 8 ? one(T) : T(1 / 2)
        @inbounds for i in 1:N
            ψm[i] += ω * (ws.ψnew[i] - ψm[i])
        end
    end
    return (false, col.maxiter, zero(T), zero(T))
end

# Implied surface potential under a flux q through the top half-cell.
_surface_potential(col, q) =
    col.ws.ψnew[1] - col.grid.z[1] * (col.gravity - q / col.ws.K[1])

function _substep!(col::SoilWaterColumn{T}, dt, P, E, sink) where {T}
    ws = col.ws
    top = col.top
    infil = runoff = evap = zero(T)
    if top isa HeadTop
        ok, it, qt, qb = _picard!(col, dt, :head, top.ψ, sink)
        ok || return (false, it, infil, runoff, evap, zero(T))
        infil = qt
    else
        qsup = T(P - E)
        ok, it, qt, qb = _picard!(col, dt, :flux, qsup, sink)
        ψswitch = zero(T)
        switch = false
        if ok
            ψ0 = _surface_potential(col, qsup)
            if qsup > 0 && ψ0 > top.ψ_pond
                switch, ψswitch = true, top.ψ_pond
            elseif qsup < 0 && ψ0 < top.ψ_dry
                switch, ψswitch = true, top.ψ_dry
            end
        elseif qsup > 0
            switch, ψswitch = true, top.ψ_pond     # stiff ponding: try the head
        end
        if switch
            copyto!(ws.ψsave, ws.ψnew); copyto!(ws.θsave, ws.θnew)
            ok2, it2, qt2, qb2 = _picard!(col, dt, :head, ψswitch, sink)
            it += it2
            if ok2 && ((qsup > 0 && qt2 < qsup) || (qsup < 0 && qt2 > qsup))
                ok, qt, qb = true, qt2, qb2
            elseif ok                               # the flux solution stands
                copyto!(ws.ψnew, ws.ψsave); copyto!(ws.θnew, ws.θsave)
            else
                return (false, it, infil, runoff, evap, zero(T))
            end
        end
        ok || return (false, it, infil, runoff, evap, zero(T))
        infil = qt
        if qsup >= 0
            runoff = qsup - qt           # zero unless the surface ponded
            evap = T(E)
        else
            evap = T(P) - qt             # ≤ E when soil-limited
        end
    end
    copyto!(col.ψ, ws.ψnew)
    copyto!(col.θ_liq, ws.θnew)
    finalize_bottom!(col.bottom, col, qb, dt)
    return (true, it, infil * dt, runoff * dt, evap * dt, qb * dt)
end

"""
    step_soil_water!(col, dt; water_input = 0, evaporation = 0, sink = nothing)
        -> SoilWaterStep

Advance the soil water of `col` by `dt` seconds. `water_input` (rain,
throughfall, snowmelt; m s⁻¹ ≥ 0) and `evaporation` (m s⁻¹, upward; a
negative value is dew) drive the [`AtmosphericTop`](@ref) boundary;
`sink` is an optional per-layer extraction (m s⁻¹ of water removed from
each layer, e.g. transpiration by roots).

Richards' equation in mixed form,

    ∂θ/∂t = ∂/∂z [K (∂ψ/∂z - 1)] - s        (z depth, positive down)

is discretised by finite volumes on the layer grid (interface
conductivity set by the column's `interface_mean`, by default the
logarithmic mean of its neighbours) and backward Euler
in time, and the nonlinear system is solved by the modified Picard
iteration of Celia et al. (1990) with a tridiagonal solve per iterate.
The new water content is the old one plus the divergence of the
converged fluxes, so

    storage_change = infiltration - drainage - extraction

holds to round-off. The step is split into sub-steps that adapt to the
Picard convergence (halved on failure, grown by half after fast convergence),
which is what keeps the solver robust when the soil nears saturation.
"""
function step_soil_water!(col::SoilWaterColumn{T}, dt::Real;
                          water_input::Real = 0.0, evaporation::Real = 0.0,
                          sink = nothing) where {T}
    dt > 0 || throw(ArgumentError("step_soil_water!: dt > 0"))
    water_input >= 0 || throw(ArgumentError("step_soil_water!: water_input ≥ 0"))
    sink === nothing || length(sink) == length(col.grid) ||
        throw(ArgumentError("step_soil_water!: sink needs one value per layer"))
    _refresh_factors!(col)
    S0 = soil_water_storage(col)
    t = zero(T)
    htry = min(col.dt_sub, T(dt))
    infil = runoff = evap = drain = extr = zero(T)
    nsub = nit = 0
    sinktot = sink === nothing ? zero(T) : T(sum(sink))
    while t < dt
        h = min(htry, T(dt) - t)
        ok, it, i_, r_, e_, d_ = _substep!(col, h, water_input, evaporation, sink)
        nit += it
        if ok
            t = (T(dt) - t - h) <= 16eps(T(dt)) * dt ? T(dt) : t + h
            infil += i_; runoff += r_; evap += e_; drain += d_; extr += sinktot * h
            nsub += 1
            if it <= 6
                htry = min(3htry / 2, T(dt))
            elseif it > col.maxiter ÷ 2
                htry = h * 3 / 4
            end
        else
            htry = h / 2
            htry >= col.dt_min ||
                throw(NumericalFailure("step_soil_water!: Picard iteration failed to converge for sub-steps down to $(col.dt_min) s"))
        end
    end
    col.dt_sub = htry
    b = col.budget
    b.infiltration += infil; b.runoff += runoff; b.evaporation += evap
    b.drainage += drain; b.extraction += extr; b.time += dt
    return SoilWaterStep{T}(infil, runoff, evap, drain, extr,
                            soil_water_storage(col) - S0, nsub, nit)
end

"""
    step_soil_water!(col, forcing::SurfaceForcing, fluxes::SurfaceFluxes, dt;
                     sink = nothing) -> SoilWaterStep

Couple the column to the `SurfaceExchange` surface interface: rain from `forcing`
(kg m⁻² s⁻¹) is the water input and the evaporation mass flux `fluxes.E`
(kg m⁻² s⁻¹, upward; negative = dew) the evaporative demand, both
converted to m s⁻¹ with ``ρ_w`` of the contract. Snow is left to a
snowpack model.
"""
function step_soil_water!(col::SoilWaterColumn, forcing::SurfaceForcing,
                          fluxes::SurfaceFluxes, dt::Real; sink = nothing)
    ρw = col.constants.ρ_water
    return step_soil_water!(col, dt; water_input = forcing.rain / ρw,
                            evaporation = fluxes.E / ρw, sink = sink)
end

# =========================================================================
# 6. Soil heat: thermal properties
# =========================================================================

"""
    SoilThermalProperties(; k_solids, c_solids, ρ_particle = 2700, kersten = :fine)

Thermal constants of the solid phase of one soil, from which
[`soil_thermal_conductivity`](@ref) and [`soil_heat_capacity`](@ref)
build the bulk values for any water and ice content:

| field        | meaning                                                          |
|:------------ |:---------------------------------------------------------------- |
| `k_solids`   | conductivity of the soil solids, W m⁻¹ K⁻¹                       |
| `c_solids`   | volumetric heat capacity of the solids, J m⁻³ K⁻¹                |
| `ρ_particle` | particle density, kg m⁻³ (sets Johansen's dry conductivity)      |
| `kersten`    | `:fine` or `:coarse` — Johansen's (1975) Kersten-number law      |

Use [`clm_soil_thermal_properties`](@ref) (sand/clay, as CLM) or
[`johansen_soil_thermal_properties`](@ref) (quartz content).
"""
struct SoilThermalProperties{T<:AbstractFloat}
    k_solids::T
    c_solids::T
    ρ_particle::T
    kersten::Symbol
    function SoilThermalProperties{T}(k_solids, c_solids, ρ_particle, kersten) where {T}
        (k_solids > 0 && c_solids > 0 && ρ_particle > 0) ||
            throw(ArgumentError("SoilThermalProperties: need k_solids, c_solids, ρ_particle > 0"))
        kersten in (:fine, :coarse) ||
            throw(ArgumentError("SoilThermalProperties: kersten must be :fine or :coarse"))
        new{T}(k_solids, c_solids, ρ_particle, kersten)
    end
end
SoilThermalProperties(; k_solids::Real, c_solids::Real, ρ_particle::Real = 2700.0,
                      kersten::Symbol = :fine) =
    SoilThermalProperties{Float64}(k_solids, c_solids, ρ_particle, kersten)

"""
    farouki_solids_conductivity(sand, clay) -> W m⁻¹ K⁻¹

Conductivity of the soil solids from the sand and clay percentages,
``k_s = (8.80\\,\\%sand + 2.92\\,\\%clay)/(\\%sand + \\%clay)`` (Farouki 1981;
CLM4.5/CLM5 Technical Notes, soil thermal properties).
"""
function farouki_solids_conductivity(sand::Real, clay::Real)
    (sand >= 0 && clay >= 0 && sand + clay > 0) ||
        throw(ArgumentError("farouki_solids_conductivity: sand, clay ≥ 0 with sand + clay > 0"))
    return (8.80 * sand + 2.92 * clay) / (sand + clay)
end

"""
    farouki_solids_heat_capacity(sand, clay) -> J m⁻³ K⁻¹

Volumetric heat capacity of the soil solids,
``c_s = (2.128\\,\\%sand + 2.385\\,\\%clay)/(\\%sand + \\%clay) × 10^6``
(Farouki 1981; CLM4.5/CLM5 Technical Notes).
"""
function farouki_solids_heat_capacity(sand::Real, clay::Real)
    (sand >= 0 && clay >= 0 && sand + clay > 0) ||
        throw(ArgumentError("farouki_solids_heat_capacity: sand, clay ≥ 0 with sand + clay > 0"))
    return (2.128 * sand + 2.385 * clay) / (sand + clay) * 1e6
end

"""
    johansen_solids_conductivity(quartz; k_other = quartz > 0.2 ? 2.0 : 3.0) -> W m⁻¹ K⁻¹

Geometric-mean conductivity of the solids from the quartz fraction `q`
(0–1), ``k_s = k_q^q\\, k_o^{1-q}`` with ``k_q = 7.7`` W m⁻¹ K⁻¹ (Johansen
1975) and ``k_o`` = 2.0 (3.0 for ``q ≤ 0.2``) for the other minerals, as
in Peters-Lidard et al. (1998).
"""
function johansen_solids_conductivity(quartz::Real; k_other::Real = quartz > 0.2 ? 2.0 : 3.0)
    0 <= quartz <= 1 || throw(ArgumentError("johansen_solids_conductivity: quartz fraction in [0, 1]"))
    return 7.7^quartz * k_other^(1 - quartz)
end

"""
    johansen_dry_conductivity(θ_s; ρ_particle = 2700) -> W m⁻¹ K⁻¹

Conductivity of dry natural soil (Johansen 1975),
``k_{dry} = (0.135 ρ_d + 64.7)/(ρ_p - 0.947 ρ_d)`` with bulk density
``ρ_d = ρ_p (1 - θ_s)`` (kg m⁻³).
"""
function johansen_dry_conductivity(θ_s::Real; ρ_particle::Real = 2700.0)
    ρd = ρ_particle * (1 - θ_s)
    return (0.135 * ρd + 64.7) / (ρ_particle - 0.947 * ρd)
end

"""
    clm_soil_thermal_properties(sand, clay) -> SoilThermalProperties

The CLM4.5/CLM5 soil thermal constants from the sand and clay
percentages: [`farouki_solids_conductivity`](@ref),
[`farouki_solids_heat_capacity`](@ref), a 2700 kg m⁻³ particle density and
the fine-soil Kersten number.
"""
clm_soil_thermal_properties(sand::Real, clay::Real) =
    SoilThermalProperties(k_solids = farouki_solids_conductivity(sand, clay),
                          c_solids = farouki_solids_heat_capacity(sand, clay))

"""
    johansen_soil_thermal_properties(quartz; coarse = false, c_solids = 2.0e6,
                                     ρ_particle = 2650) -> SoilThermalProperties

Johansen (1975) / Peters-Lidard et al. (1998) soil from its quartz
fraction ([`johansen_solids_conductivity`](@ref)); `coarse = true` selects
the coarse-soil Kersten number. The default solids heat capacity is the
mineral value of de Vries (1963), 2.0 MJ m⁻³ K⁻¹.
"""
johansen_soil_thermal_properties(quartz::Real; coarse::Bool = false, c_solids::Real = 2.0e6,
                                 ρ_particle::Real = 2650.0) =
    SoilThermalProperties(k_solids = johansen_solids_conductivity(quartz), c_solids = c_solids,
                          ρ_particle = ρ_particle, kersten = coarse ? :coarse : :fine)

"""
    kersten_number(p::SoilThermalProperties, S_r, f_liq = 1) -> Ke ∈ [0, 1]

Johansen's (1975) normalised conductivity ``Ke = (k - k_{dry})/(k_{sat} - k_{dry})``
at degree of saturation ``S_r``. Unfrozen: ``\\log_{10}S_r + 1`` (fine)
or ``0.7\\log_{10}S_r + 1`` (coarse), floored at 0; frozen: ``S_r``. A
partly frozen layer with liquid fraction `f_liq` of its water takes the
liquid-weighted mean — a continuous blend where CLM switches at 0 °C.
"""
function kersten_number(p::SoilThermalProperties, S_r::Real, f_liq::Real = 1.0)
    S = clamp(S_r, 0.0, 1.0)
    S > 0 || return zero(float(S))
    Ku = p.kersten === :fine ? log10(S) + 1 : 0.7 * log10(S) + 1
    return f_liq * max(Ku, zero(Ku)) + (1 - f_liq) * S
end

"""
    soil_thermal_conductivity(p, θ_s, θ_liq, θ_ice; c = DEFAULT_SURFACE_CONSTANTS) -> W m⁻¹ K⁻¹

Bulk conductivity by Johansen's (1975) method as used in CLM (Farouki
1981): ``k = Ke\\,k_{sat} + (1 - Ke)\\,k_{dry}``, with the saturated value the
geometric mean of the constituents,

    k_sat = k_s^{1-θ_s} k_w^{θ_s f_liq} k_ice^{θ_s (1 - f_liq)}

(`f_liq` the liquid fraction of the pore water by volume), ``S_r =
(θ_{liq} + θ_{ice})/θ_s`` and [`kersten_number`](@ref). The conductivities of
water and ice are the surface shared core's `k_water`, `k_ice`.
"""
function soil_thermal_conductivity(p::SoilThermalProperties, θ_s::Real, θ_liq::Real, θ_ice::Real;
                                   c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    kdry = johansen_dry_conductivity(θ_s; ρ_particle = p.ρ_particle)
    tot = θ_liq + θ_ice
    tot > 0 || return kdry
    fl = θ_liq / tot
    ksat = p.k_solids^(1 - θ_s) * c.k_water^(θ_s * fl) * c.k_ice^(θ_s * (1 - fl))
    Ke = kersten_number(p, tot / θ_s, fl)
    return Ke * ksat + (1 - Ke) * kdry
end

"""
    soil_heat_capacity(p, θ_s, θ_liq, θ_ice; c = DEFAULT_SURFACE_CONSTANTS) -> J m⁻³ K⁻¹

Volumetric heat capacity from the constituents (de Vries 1963),
``C = c_s(1-θ_s) + ρ_w c_w θ_{liq} + ρ_{ice} c_{ice} θ_{ice}`` (air
neglected), with the densities from the surface shared core (`SurfaceExchange`) and the specific
heats delegated to `MoistThermodynamics` through it.
"""
soil_heat_capacity(p::SoilThermalProperties, θ_s::Real, θ_liq::Real, θ_ice::Real;
                   c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    p.c_solids * (1 - θ_s) + c.ρ_water * cp_water(c) * θ_liq + c.ρ_ice * cp_ice(c) * θ_ice

# =========================================================================
# 7. Freezing: Clapeyron equilibrium, enthalpy, ice impedance
# =========================================================================

"""
    freezing_matric_potential(T; c = DEFAULT_SURFACE_CONSTANTS) -> ψ (m)

Matric potential of liquid water in equilibrium with pore ice at
temperature `T` < ``T_f``, by the generalised Clapeyron equation
(Koopmans & Miller 1966; CLM5 supercooled water, Niu & Yang 2006):

    ψ = L_f (T - T_f) / (g T)

≈ −125 m (−1.2 MPa) per kelvin of undercooling. Zero at and above
``T_f``. ``L_f``, ``g`` and ``T_f`` come from the surface shared core (`SurfaceExchange`).
"""
function freezing_matric_potential(T::Real; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    Tf = c.thermo.T_freeze
    T >= Tf && return zero(float(T))
    return LH_f0(c) * (T - Tf) / (grav(c) * T)
end

"""
    freezing_onset_temperature(ψ; c = DEFAULT_SURFACE_CONSTANTS) -> K

Temperature at which water held at matric potential `ψ` ≤ 0 (m) starts
to freeze — the inverse of [`freezing_matric_potential`](@ref),
``T^* = L_f T_f / (L_f - g ψ)``.
"""
function freezing_onset_temperature(ψ::Real; c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS)
    L = LH_f0(c)
    return L * c.thermo.T_freeze / (L - grav(c) * min(ψ, zero(ψ)))
end

"""
    supercooled_liquid_content(h, T; c = DEFAULT_SURFACE_CONSTANTS) -> θ_liq,max

Largest liquid water content soil `h` can hold at `T` in the presence of
ice: the retention curve at [`freezing_matric_potential`](@ref) (the
"freezing = drying" analogy; CLM5, Niu & Yang 2006, Dall'Amico et al.
2011). Equals ``θ_s`` at and above ``T_f``.
"""
supercooled_liquid_content(h::AbstractHydraulicModel, T::Real;
                           c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) =
    soil_water_content(h, freezing_matric_potential(T; c = c))

"""
    PowerLawIceImpedance(; Ω = 6)

Ice impedance of CLM4.5/CLM5 (Swenson et al. 2012): hydraulic
conductivity is multiplied by ``10^{-Ω F_{ice}}`` with the ice fraction of
the pore space ``F_{ice} = θ_{ice}/θ_s`` (capped at 1). ``Ω = 6``: a layer
whose pores are half ice conducts 1000 times less.
"""
struct PowerLawIceImpedance{T<:AbstractFloat} <: AbstractIceImpedance
    Ω::T
end
PowerLawIceImpedance(; Ω::Real = 6.0) =
    (Ω >= 0 || throw(ArgumentError("PowerLawIceImpedance: Ω ≥ 0")); PowerLawIceImpedance{Float64}(Ω))
ice_impedance_factor(imp::PowerLawIceImpedance, θ_ice::Real, θ_s::Real) =
    exp10(-imp.Ω * clamp(θ_ice / θ_s, 0.0, 1.0))

# =========================================================================
# 8. Heat boundary conditions
# =========================================================================

"""
    AbstractSoilHeatTop

Upper boundary of the heat equation, given to [`step_soil_heat!`](@ref).
Every subtype is reduced to a ground heat flux linear in the top-node
temperature, ``G = α + β T_1`` (``β ≤ 0``), and a skin temperature
``T_s(T_1)``: [`GroundHeatFluxTop`](@ref), [`SkinTemperatureTop`](@ref),
[`LinearizedSkinBalance`](@ref).
"""
abstract type AbstractSoilHeatTop end

"Prescribed ground heat flux `G` (W m⁻², positive into the soil)."
struct GroundHeatFluxTop{T<:AbstractFloat} <: AbstractSoilHeatTop
    G::T
end
GroundHeatFluxTop(G::Real) = GroundHeatFluxTop{Float64}(G)

"Prescribed skin (soil-surface) temperature `T` (K) at depth 0."
struct SkinTemperatureTop{T<:AbstractFloat} <: AbstractSoilHeatTop
    T::T
end
SkinTemperatureTop(T::Real) = SkinTemperatureTop{Float64}(T)

"""
    LinearizedSkinBalance(G0, dG_dT, T0)

A massless skin whose atmospheric energy supply
``R_n - H - LE`` is linearised about the skin temperature `T0`:
``G_{atm}(T_s) = G_0 + (dG/dT_s)(T_s - T_0)`` with ``dG/dT_s ≤ 0``
(radiative, sensible and latent damping). The skin temperature closes
``G_{atm}(T_s) = (k_1/z_1)(T_s - T_1)`` implicitly with the soil — the
coupling [`step_surface!`](@ref) uses, and the one a canopy model (#21)
passes for the ground below its leaves.
"""
struct LinearizedSkinBalance{T<:AbstractFloat} <: AbstractSoilHeatTop
    G0::T
    dG_dT::T
    T0::T
end
LinearizedSkinBalance(G0::Real, dG_dT::Real, T0::Real) =
    LinearizedSkinBalance{Float64}(promote(float(G0), float(dG_dT), float(T0))...)

# (α, β) with G = α + β T1, for a top half-cell conductance κ = k₁/z₁
_heat_top(bc::GroundHeatFluxTop, κ) = (bc.G, zero(κ))
_heat_top(bc::SkinTemperatureTop, κ) = (κ * bc.T, -κ)
function _heat_top(bc::LinearizedSkinBalance, κ)
    D = bc.dG_dT
    κ - D > 0 || throw(ArgumentError("LinearizedSkinBalance: need dG_dT < k₁/z₁"))
    return (κ * (bc.G0 - D * bc.T0) / (κ - D), κ * D / (κ - D))
end
_skin_temperature(bc::GroundHeatFluxTop, κ, T1) = T1 + bc.G / κ
_skin_temperature(bc::SkinTemperatureTop, κ, T1) = bc.T
_skin_temperature(bc::LinearizedSkinBalance, κ, T1) =
    (bc.G0 - bc.dG_dT * bc.T0 + κ * T1) / (κ - bc.dG_dT)

"""
    AbstractSoilHeatBottom

Lower boundary of the heat equation: the heat flux leaving the column
downward through its base is ``F = α + β T_N`` (``β ≥ 0``).
[`ZeroHeatFluxBottom`](@ref) (CLM's default), [`GeothermalHeatFlux`](@ref),
[`BottomTemperature`](@ref).
"""
abstract type AbstractSoilHeatBottom end
"Insulated base, no heat flux (CLM's lower boundary)."
struct ZeroHeatFluxBottom <: AbstractSoilHeatBottom end
"""
    GeothermalHeatFlux(q)

Heat flux `q` (W m⁻²) entering the column from below (upward positive;
the continental mean is about 0.065 W m⁻²: Pollack, Hurter & Johnson
1993, *Rev. Geophys.*, 31, 267-280).
"""
struct GeothermalHeatFlux{T<:AbstractFloat} <: AbstractSoilHeatBottom
    q::T
end
GeothermalHeatFlux(q::Real) = GeothermalHeatFlux{Float64}(q)
"Prescribed temperature `T` (K) at the base interface of the column."
struct BottomTemperature{T<:AbstractFloat} <: AbstractSoilHeatBottom
    T::T
end
BottomTemperature(T::Real) = BottomTemperature{Float64}(T)

_heat_bottom(::ZeroHeatFluxBottom, κ) = (zero(κ), zero(κ))
_heat_bottom(bc::GeothermalHeatFlux, κ) = (-bc.q + zero(κ), zero(κ))
_heat_bottom(bc::BottomTemperature, κ) = (-κ * bc.T, κ)

# =========================================================================
# 9. The thermal column: a soil water column with heat and freeze–thaw
# =========================================================================

"""
    SoilHeatBudget

Cumulative energy budget of a [`SoilThermalColumn`](@ref) (J m⁻²):
`surface` (ground heat flux into the top, ∫G dt), `bottom` (out of the
base), `water` (enthalpy moved by water redistribution, infiltration and
evaporation, whose water carries the enthalpy of the layer it leaves or
joins at that layer's temperature), `latent_mismatch` (latent heat of
evaporation the surface balance charged but the soil could not deliver;
since #129 [`step_surface!`](@ref) returns the delivered evaporation and
puts this heat into the top layer, so it is also part of `surface`),
and `time` (s). ``Δ(\\text{enthalpy}) = surface - bottom + water``.
"""
mutable struct SoilHeatBudget{T<:AbstractFloat}
    surface::T
    bottom::T
    water::T
    latent_mismatch::T
    time::T
end
SoilHeatBudget{T}() where {T} = SoilHeatBudget{T}(0, 0, 0, 0, 0)

"""
    SoilHeatStep

What one [`step_soil_heat!`](@ref) did: time-mean ground heat flux `G`
and basal flux `bottom_flux` (W m⁻²), the final and time-mean skin
temperature `T_skin`, `T_skin_mean` (K), `enthalpy_change` (J m⁻²),
`freezing` (kg m⁻² of water frozen, negative = melt), `substeps` and
Newton `iterations`.
"""
struct SoilHeatStep{T<:AbstractFloat}
    G::T
    bottom_flux::T
    T_skin::T
    T_skin_mean::T
    enthalpy_change::T
    freezing::T
    substeps::Int
    iterations::Int
end

struct _HeatWorkspace{T}
    W::Vector{T}; Hn::Vector{T}; Hm::Vector{T}; Hnew::Vector{T}
    Tm::Vector{T}; s::Vector{T}; e::Vector{T}; k::Vector{T}; kf::Vector{T}
    a::Vector{T}; b::Vector{T}; c::Vector{T}; d::Vector{T}; cp::Vector{T}; dp::Vector{T}
    Hsave::Vector{T}; θlsave::Vector{T}; θisave::Vector{T}; Tsave::Vector{T}
end
_HeatWorkspace{T}(N) where {T} = _HeatWorkspace{T}((zeros(T, N) for _ in 1:19)...)

"""
    SoilThermalColumn(water::SoilWaterColumn, thermal; T_soil = 283.15, T_skin = T_soil[1],
                      bottom = ZeroHeatFluxBottom(), freezing_point_depression = true,
                      unfrozen_fraction = 1e-3, albedo = 0.2,
                      emissivity = material_emissivity(:soil_dry),
                      z0m = 0.01, z0h = z0m / 10, z0q = z0h, hydrology = true,
                      tol_T = 1e-6, tol_H = 1.0, maxiter = 50, dt_min = 1e-3)

A multi-layer soil with heat conduction, freeze–thaw and (through the
wrapped [`SoilWaterColumn`](@ref)) Richards-equation water transport — an
[`AbstractSurface`](@ref) with a skin energy balance. `thermal` is one
[`SoilThermalProperties`](@ref) or one per layer; `T_soil` the initial layer
temperatures (scalar or vector, K); the initial water of `water` is
partitioned into liquid and ice in equilibrium at `T`.

# Prognostic: enthalpy

Each layer carries its volumetric enthalpy (J m⁻³), relative to liquid
water at the freezing point ``T_f``,

    H = C(θ_liq, θ_ice) (T - T_f) - ρ_ice L_f θ_ice

with [`soil_heat_capacity`](@ref) ``C``. Given `H` and the layer's total
water ``W = θ_{liq} + (ρ_{ice}/ρ_w)θ_{ice}``, temperature and phase
follow from equilibrium:

  * `freezing_point_depression = true` (default): liquid water coexists
    with ice below ``T_f`` up to [`supercooled_liquid_content`](@ref)
    (Clapeyron + retention curve), so freezing starts at the
    [`freezing_onset_temperature`](@ref) of the layer's matric potential
    and proceeds over a temperature range;
  * `false`: a sharp phase change at ``T_f`` — the Stefan problem.

In both cases a fraction `unfrozen_fraction` of the pore space above the
residual content stays liquid, so the water solver always has liquid to
work with.

The state is shared with `water`: `water.T_soil`, `water.θ_liq`,
`water.θ_ice` are this column's temperature and phases, and give the
water solver its viscosity and ice-impedance factors (pass
`ice_impedance = PowerLawIceImpedance()` to the water column).

# Surface

`albedo`, `emissivity` (default the dry-soil value of `SurfaceOptics`)
and roughness lengths `z0m`, `z0h`, `z0q` (m; 0.01 m is CLM4.5's bare-soil
momentum roughness) make it an `AbstractSurface`; `T_skin` is the skin
temperature, a massless surface at depth 0 joined to the first node
through the half layer. `hydrology = false` freezes the water
redistribution in [`step_surface!`](@ref) (heat-only runs).

Numerics: see [`step_soil_heat!`](@ref).
"""
mutable struct SoilThermalColumn{T<:AbstractFloat, WC<:SoilWaterColumn, P<:SoilThermalProperties,
                                 BB<:AbstractSoilHeatBottom} <: AbstractSurface
    water::WC
    thermal::Vector{P}
    H::Vector{T}
    T_skin::T
    bottom::BB
    freezing_point_depression::Bool
    unfrozen_fraction::T
    albedo::T
    emissivity::T
    z0m::T
    z0h::T
    z0q::T
    hydrology::Bool
    tol_T::T
    tol_H::T
    maxiter::Int
    dt_min::T
    dt_sub::T
    budget::SoilHeatBudget{T}
    ws::_HeatWorkspace{T}
end

function SoilThermalColumn(water::SoilWaterColumn{T}, thermal;
                           T_soil = 283.15, T_skin = nothing,
                           bottom::AbstractSoilHeatBottom = ZeroHeatFluxBottom(),
                           freezing_point_depression::Bool = true,
                           unfrozen_fraction::Real = 1e-3,
                           albedo::Real = 0.2, emissivity::Real = material_emissivity(:soil_dry),
                           z0m::Real = 0.01, z0h::Real = z0m / 10, z0q::Real = z0h,
                           hydrology::Bool = true, tol_T::Real = 1e-6, tol_H::Real = 1.0,
                           maxiter::Integer = 50, dt_min::Real = 1e-3) where {T}
    N = length(water.grid)
    P = thermal isa SoilThermalProperties ? fill(thermal, N) : _per_layer(thermal, N, "thermal")
    P = [p for p in P]
    0 < unfrozen_fraction < 1 || throw(ArgumentError("SoilThermalColumn: 0 < unfrozen_fraction < 1"))
    Tv = T.(_per_layer(T_soil, N, "T_soil"))
    col = SoilThermalColumn(water, P, zeros(T, N), T(T_skin === nothing ? Tv[1] : T_skin), bottom,
                            freezing_point_depression, T(unfrozen_fraction), T(albedo),
                            T(emissivity), T(z0m), T(z0h), T(z0q), hydrology, T(tol_T), T(tol_H),
                            Int(maxiter), T(dt_min), T(Inf), SoilHeatBudget{T}(), _HeatWorkspace{T}(N))
    set_soil_temperature!(col, Tv)
    return col
end

# ---- layer thermodynamics -----------------------------------------------------

_ice_ratio(col) = col.water.constants.ρ_ice / col.water.constants.ρ_water
_total_water(col, i) = col.water.θ_liq[i] + _ice_ratio(col) * col.water.θ_ice[i]
function _liquid_floor(col, i)
    h = col.water.hydraulics[i]
    θr = residual_water_content(h)
    return θr + col.unfrozen_fraction * (soil_porosity(h) - θr)
end

_layer_C(col::SoilThermalColumn, i, θl, θi) =
    soil_heat_capacity(col.thermal[i], soil_porosity(col.water.hydraulics[i]), θl, θi;
                       c = col.water.constants)
function _layer_enthalpy(col::SoilThermalColumn, i, T, θl, θi)
    c = col.water.constants
    return _layer_C(col, i, θl, θi) * (T - c.thermo.T_freeze) - c.ρ_ice * LH_f0(c) * θi
end
_layer_k(col::SoilThermalColumn, i, θl, θi) =
    soil_thermal_conductivity(col.thermal[i], soil_porosity(col.water.hydraulics[i]), θl, θi;
                              c = col.water.constants)

# Largest liquid content coexisting with ice at T < T_f, and its T-derivative.
function _liquid_max(col, i, T)
    fl = _liquid_floor(col, i)
    col.freezing_point_depression || return (fl, zero(T))
    c = col.water.constants
    h = col.water.hydraulics[i]
    ψ = freezing_matric_potential(T; c = c)
    θ = soil_water_content(h, ψ)
    θ <= fl && return (fl, zero(T))
    dψdT = LH_f0(c) * c.thermo.T_freeze / (grav(c) * T^2)
    return (θ, soil_water_capacity(h, ψ) * dψdT)
end

# Equilibrium phase state of layer i at temperature T with total water W.
function _equilibrium_at(col, i, T, W)
    r = _ice_ratio(col)
    Tf = col.water.constants.thermo.T_freeze
    fl = _liquid_floor(col, i)
    (T >= Tf || W <= fl) && return (W, zero(W))
    col.freezing_point_depression || return (fl, (W - fl) / r)   # below T_f: all but the floor frozen
    θm, _ = _liquid_max(col, i, T)
    θl = min(W, θm)
    return (θl, (W - θl) / r)
end

"""
Temperature, phases and dT/dH of layer i with enthalpy H and total water W:
(T, θ_liq, θ_ice, s = dT/dH).
"""
function _phase_state(col, i, H, W)
    c = col.water.constants
    Tf = c.thermo.T_freeze
    r = _ice_ratio(col)
    ρL = c.ρ_ice * LH_f0(c)
    fl = _liquid_floor(col, i)
    Cl = _layer_C(col, i, W, zero(W))
    if W <= fl || H >= 0 && !col.freezing_point_depression
        return (Tf + H / Cl, W, zero(W), 1 / Cl)
    end
    if !col.freezing_point_depression
        θimax = (W - fl) / r
        Hs = -ρL * θimax
        if H >= Hs                                         # mushy: at T_f
            θi = -H / ρL
            return (Tf, W - r * θi, θi, zero(H))
        end
        Cf = _layer_C(col, i, fl, θimax)
        return (Tf + (H - Hs) / Cf, fl, θimax, 1 / Cf)
    end
    # Freezing-point depression: freezing starts at T*(ψ(W)).
    h = col.water.hydraulics[i]
    ψW = W >= soil_porosity(h) ? air_entry_potential(h) : soil_matric_potential(h, W)
    Tstar = freezing_onset_temperature(ψW; c = c)
    Hstar = Cl * (Tstar - Tf)
    H >= Hstar && return (Tf + H / Cl, W, zero(W), 1 / Cl)
    # Solve Heq(T) = H on T < T*, Heq increasing: safeguarded Newton.
    Heq(T) = begin
        θl, θi = _equilibrium_at(col, i, T, W)
        _layer_enthalpy(col, i, T, θl, θi)
    end
    dHeq(T) = begin
        θl, θi = _equilibrium_at(col, i, T, W)
        θm, dθ = _liquid_max(col, i, T)
        dθl = θm < W ? dθ : zero(dθ)
        _layer_C(col, i, θl, θi) + dθl * c.ρ_water * ((cp_water(c) - cp_ice(c)) * (T - Tf) + LH_f0(c))
    end
    hi = Tstar
    lo = Tstar - one(Tstar)
    while Heq(lo) > H
        lo = Tstar - 2 * (Tstar - lo)
        lo > 0 || error("SoilThermalColumn: enthalpy $H below any attainable temperature")
    end
    # Converged in enthalpy (not only in T: near T* the ice content moves
    # by ~10 per kelvin, so a T tolerance alone would leave θ_ice loose).
    Htol = 1e-13 * (abs(H) + ρL * W)
    Tk = (lo + hi) / 2
    for _ in 1:200
        f = Heq(Tk) - H
        abs(f) <= Htol && break
        f > 0 ? (hi = Tk) : (lo = Tk)
        Tn = Tk - f / dHeq(Tk)
        (lo < Tn < hi) || (Tn = (lo + hi) / 2)
        Tk = Tn
        hi - lo <= 4eps(Tk) && break
    end
    θl, θi = _equilibrium_at(col, i, Tk, W)
    return (Tk, θl, θi, 1 / dHeq(Tk))
end

"""
    set_soil_temperature!(col::SoilThermalColumn, T)

Set the layer temperatures (scalar or per layer, K), re-partitioning each
layer's total water into liquid and ice in equilibrium at `T` and
recomputing the enthalpy. Water is conserved; the matric potential is
unchanged.
"""
function set_soil_temperature!(col::SoilThermalColumn, T)
    w = col.water
    Tv = _per_layer(T, length(w.grid), "T")
    r = _ice_ratio(col)
    for i in eachindex(col.H)
        W = _total_water(col, i)
        θl, θi = _equilibrium_at(col, i, Tv[i], W)
        w.θ_ice[i] = θi
        w.θ_liq[i] = W - r * θi
        w.T_soil[i] = Tv[i]
        col.H[i] = _layer_enthalpy(col, i, Tv[i], w.θ_liq[i], θi)
    end
    return col
end

# Set the phase state of every layer from the enthalpy in col.H.
function _apply_enthalpy!(col)
    w = col.water
    r = _ice_ratio(col)
    for i in eachindex(col.H)
        W = _total_water(col, i)
        T, _, θi, _ = _phase_state(col, i, col.H[i], W)
        w.T_soil[i] = T
        w.θ_ice[i] = θi
        w.θ_liq[i] = W - r * θi
    end
    return col
end

"""
    soil_enthalpy(col::SoilThermalColumn) -> J m⁻²

Column enthalpy ``Σ_i Δz_i H_i`` relative to liquid water at ``T_f``.
"""
soil_enthalpy(col::SoilThermalColumn) = sum(col.water.grid.dz .* col.H)

"""
    soil_ice_mass(col::SoilThermalColumn) -> kg m⁻²

Ice in the column, ``Σ_i Δz_i ρ_{ice} θ_{ice,i}``.
"""
soil_ice_mass(col::SoilThermalColumn) =
    col.water.constants.ρ_ice * sum(col.water.grid.dz .* col.water.θ_ice)

"""
    soil_thermal_layer_conductivity(col::SoilThermalColumn, i) -> W m⁻¹ K⁻¹

Current bulk thermal conductivity of layer `i`.
"""
soil_thermal_layer_conductivity(col::SoilThermalColumn, i::Integer) =
    _layer_k(col, i, col.water.θ_liq[i], col.water.θ_ice[i])

# ---- one backward-Euler heat step by Newton on the enthalpy -------------------
#
# Cell i:  Δz_i (H_i^{n+1} - H_i^n)/Δt = F_{i-1/2} - F_{i+1/2},
#          F_{i+1/2} = k_{i+1/2} (T_i - T_{i+1})/(z_{i+1} - z_i)   (downward),
# with the series (harmonic, thickness-weighted) interface conductivity.
# T = T(H, W) is linearised as T ≈ T^m + s^m (H - H^m), s = dT/dH, which
# is zero inside a sharp phase change — so the unknown is H, and the
# matrix D + A S is column diagonally dominant. On convergence the new
# enthalpy is the old plus the divergence of the converged fluxes, so the
# column conserves energy to round-off.
function _heat_newton!(col::SoilThermalColumn{T}, dt, top) where {T}
    w, ws, g = col.water, col.ws, col.water.grid
    N = length(g)
    Hm = ws.Hm
    copyto!(Hm, ws.Hn)
    κt = κb = αt = βt = αb = βb = zero(T)
    for it in 1:col.maxiter
        @inbounds for i in 1:N
            Ti, θl, θi, si = _phase_state(col, i, Hm[i], ws.W[i])
            ws.Tm[i], ws.s[i] = Ti, si
            ws.k[i] = _layer_k(col, i, θl, θi)
            ws.e[i] = Ti - si * Hm[i]
        end
        @inbounds for i in 1:N-1
            # conductance (W m⁻² K⁻¹) of the two half layers in series
            ws.kf[i] = 1 / ((g.zh[i+1] - g.z[i]) / ws.k[i] + (g.z[i+1] - g.zh[i+1]) / ws.k[i+1])
        end
        κt = ws.k[1] / g.z[1]
        κb = ws.k[N] / (g.zh[end] - g.z[N])
        αt, βt = _heat_top(top, κt)
        αb, βb = _heat_bottom(col.bottom, κb)
        @inbounds for i in 1:N
            r = g.dz[i] / dt
            diag = r
            rhs = r * ws.Hn[i]
            lo = up = zero(T)
            if i == 1
                diag -= βt * ws.s[1]
                rhs += αt + βt * ws.e[1]
            else
                a = ws.kf[i-1]
                diag += a * ws.s[i]; lo = -a * ws.s[i-1]
                rhs += a * (ws.e[i-1] - ws.e[i])
            end
            if i == N
                diag += βb * ws.s[N]
                rhs -= αb + βb * ws.e[N]
            else
                a = ws.kf[i]
                diag += a * ws.s[i]; up = -a * ws.s[i+1]
                rhs += a * (ws.e[i+1] - ws.e[i])
            end
            ws.a[i], ws.b[i], ws.c[i], ws.d[i] = lo, diag, up, rhs
        end
        _thomas!(ws.Hnew, ws.a, ws.b, ws.c, ws.d, ws.cp, ws.dp)
        conv = true
        @inbounds for i in 1:N
            isfinite(ws.Hnew[i]) || return (false, it, zero(T), zero(T), zero(T))
            Tn = _phase_state(col, i, ws.Hnew[i], ws.W[i])[1]
            (abs(Tn - ws.Tm[i]) <= col.tol_T && abs(ws.Hnew[i] - Hm[i]) <= col.tol_H) || (conv = false)
            ws.Tm[i] = Tn                         # now the temperature of the new iterate
        end
        if conv
            # conservative update from the converged fluxes
            G = αt + βt * ws.Tm[1]
            Fprev = G
            Fb = zero(T)
            @inbounds for i in 1:N
                Fnext = i < N ? ws.kf[i] * (ws.Tm[i] - ws.Tm[i+1]) : αb + βb * ws.Tm[N]
                ws.Hnew[i] = ws.Hn[i] + dt / g.dz[i] * (Fprev - Fnext)
                Fprev = Fnext
            end
            Fb = Fprev
            return (true, it, G, Fb, _skin_temperature(top, κt, ws.Tm[1]))
        end
        ω = it <= 10 ? one(T) : T(1 / 2)
        @inbounds for i in 1:N
            Hm[i] += ω * (ws.Hnew[i] - Hm[i])
        end
    end
    return (false, col.maxiter, zero(T), zero(T), zero(T))
end

"""
    step_soil_heat!(col::SoilThermalColumn, dt, top::AbstractSoilHeatTop) -> SoilHeatStep

Advance the soil temperature and ice of `col` by `dt` seconds with the
water content fixed, under the upper boundary `top`
([`GroundHeatFluxTop`](@ref), [`SkinTemperatureTop`](@ref),
[`LinearizedSkinBalance`](@ref)) and the column's `bottom`.

The heat equation in enthalpy form,

    ∂H/∂t = ∂/∂z (k ∂T/∂z),   H = C (T - T_f) - ρ_ice L_f θ_ice,

is discretised by finite volumes (series interface conductivity) and
backward Euler, and solved by Newton's method on the enthalpy with the
conductivity lagged (the enthalpy method of e.g. Voller et al. 1990,
Nedjar 2002). Because the unknown is `H` and not `T`, a sharp phase
change needs no apparent-heat-capacity smoothing: layers at ``T_f``
simply absorb or release latent heat. The new enthalpy is formed from the
divergence of the converged fluxes, so

    enthalpy_change = (G - bottom_flux) dt

to round-off. Sub-steps adapt to the Newton convergence (halved on
failure). The layer temperatures, `θ_liq` and `θ_ice` of `col.water` are
updated; its matric potential is unchanged (freezing conserves the total
water, on which retention depends).
"""
function step_soil_heat!(col::SoilThermalColumn{T}, dt::Real, top::AbstractSoilHeatTop) where {T}
    dt > 0 || throw(ArgumentError("step_soil_heat!: dt > 0"))
    ws, w = col.ws, col.water
    N = length(col.H)
    for i in 1:N
        ws.W[i] = _total_water(col, i)
    end
    E0 = soil_enthalpy(col)
    ice0 = soil_ice_mass(col)
    t = zero(T)
    htry = min(col.dt_sub, T(dt))
    Gsum = Fsum = Tssum = zero(T)
    Ts = col.T_skin
    nsub = nit = 0
    while t < dt
        h = min(htry, T(dt) - t)
        copyto!(ws.Hn, col.H)
        ok, it, G, Fb, Tsk = _heat_newton!(col, h, top)
        nit += it
        if ok
            copyto!(col.H, ws.Hnew)
            _apply_enthalpy!(col)
            t = (T(dt) - t - h) <= 16eps(T(dt)) * dt ? T(dt) : t + h
            Gsum += G * h; Fsum += Fb * h; Tssum += Tsk * h
            Ts = Tsk
            nsub += 1
            it <= 4 && (htry = min(3htry / 2, T(dt)))
        else
            htry = h / 2
            htry >= col.dt_min ||
                throw(NumericalFailure("step_soil_heat!: Newton iteration failed to converge for sub-steps down to $(col.dt_min) s"))
        end
    end
    col.dt_sub = htry
    col.T_skin = Ts
    b = col.budget
    b.surface += Gsum; b.bottom += Fsum; b.time += dt
    return SoilHeatStep{T}(Gsum / dt, Fsum / dt, Ts, Tssum / dt, soil_enthalpy(col) - E0,
                           soil_ice_mass(col) - ice0, nsub, nit)
end

# ---- water transport inside the thermal column --------------------------------

"""
    step_soil_water!(col::SoilThermalColumn, dt; water_input, evaporation, sink)
    step_soil_water!(col::SoilThermalColumn, forcing, fluxes, dt; sink)

Richards step of the wrapped water column (see the [`SoilWaterColumn`](@ref)
methods), followed by the energy bookkeeping of the moved water: each
layer keeps its temperature while its liquid changes, so water carries
the enthalpy of the layer it enters or leaves (recorded in
`col.budget.water`), and each layer then re-equilibrates adiabatically —
water that infiltrates a frozen layer refreezes and warms it.
"""
function step_soil_water!(col::SoilThermalColumn, dt::Real; kwargs...)
    st = step_soil_water!(col.water, dt; kwargs...)
    _after_water_step!(col)
    return st
end
function step_soil_water!(col::SoilThermalColumn, forcing::SurfaceForcing,
                          fluxes::SurfaceFluxes, dt::Real; sink = nothing)
    st = step_soil_water!(col.water, forcing, fluxes, dt; sink = sink)
    _after_water_step!(col)
    return st
end

function _after_water_step!(col::SoilThermalColumn)
    w = col.water
    dE = zero(eltype(col.H))
    for i in eachindex(col.H)
        Hn = _layer_enthalpy(col, i, w.T_soil[i], w.θ_liq[i], w.θ_ice[i])
        dE += w.grid.dz[i] * (Hn - col.H[i])
        col.H[i] = Hn
    end
    col.budget.water += dE
    _apply_enthalpy!(col)
    return col
end

# ---- the AbstractSurface interface ---------------------------------------------

surface_temperature(col::SoilThermalColumn) = col.T_skin
roughness_lengths(col::SoilThermalColumn) = (z0m = col.z0m, z0h = col.z0h, z0q = col.z0q)
surface_albedo(col::SoilThermalColumn) = col.albedo
surface_emissivity(col::SoilThermalColumn) = col.emissivity
"Lee & Pielke (1992) β of the wrapped water column (liquid water of the top layer)."
moisture_availability(col::SoilThermalColumn) = moisture_availability(col.water)

"""
    step_surface!(col::SoilThermalColumn, forcing, dt; δT = 0.01, Q_extra = 0,
                  kwargs...) -> SurfaceFluxes

One coupled land-surface step: skin energy balance, soil heat and
freeze–thaw, then soil water.

 1. The atmospheric fluxes of the current skin temperature ``T_0`` come
    from `SurfaceExchange.turbulent_fluxes` (Monin–Obukhov, with
    `kwargs` passed on) and are linearised in the skin temperature by a
    finite difference of `δT`: ``R_n``, ``H``, ``LE`` and ``E``.
    Evaporation is capped by the liquid water of the top layer.
 2. [`step_soil_heat!`](@ref) with a [`LinearizedSkinBalance`](@ref):
    the skin temperature and the soil temperatures are solved together,
    implicitly, so the ground heat flux couples the skin to the soil's
    heat store — a sink for a skin warmer than the soil, a source for a
    skin cooling below it at night.
 3. The returned `SurfaceFluxes` are the linearised fluxes at the
    time-mean skin temperature of the step, so ``R_n - H - LE = G`` and
    ``G\\,dt`` is exactly the energy the soil received.
 4. With `hydrology = true`, [`step_soil_water!`](@ref) with the rain of
    `forcing` and the evaporation ``E`` just returned. If the soil
    delivers less (its surface dries to the dry potential), the returned
    `E` and `LE` are what it delivered, and the latent heat charged but
    not spent goes into the top layer and into `G` (#129: before, the
    surface reported the full demand, so an atmosphere coupled to it
    received water the soil never lost).

`Q_extra` (W m⁻², positive *into* the surface, constant over the step) is
energy the skin receives besides radiation and the turbulent fluxes —
the latent heat of dew deposited on it, heat released onto it, or
(negative) sunlight transmitted through glazing. It enters the skin
balance with the atmospheric supply, ``G_{atm} = R_n - H - LE + Q_{extra}``,
so the returned ``G = R_n - H - LE + Q_{extra}`` is still exactly what the
soil received.
"""
function step_surface!(col::SoilThermalColumn{T}, f::SurfaceForcing, dt::Real;
                       δT::Real = 0.01, Q_extra::Real = 0.0, kwargs...) where {T}
    c = col.water.constants
    T0 = col.T_skin
    f0 = turbulent_fluxes(col, f; c = c, kwargs...)
    col.T_skin = T0 + δT
    f1 = turbulent_fluxes(col, f; c = c, kwargs...)
    col.T_skin = T0
    dRn, dH = (f1.Rn - f0.Rn) / δT, (f1.H - f0.H) / δT
    LE0, dLE = f0.LE, (f1.LE - f0.LE) / δT
    E0, dE = f0.E, (f1.E - f0.E) / δT
    if col.hydrology && E0 > 0
        # the top layer cannot evaporate more liquid than it holds
        Emax = c.ρ_water * max(col.water.θ_liq[1] - _liquid_floor(col, 1), zero(T)) *
               col.water.grid.dz[1] / dt
        if E0 > Emax
            φ = Emax / E0
            LE0 *= φ; dLE *= φ; E0 *= φ; dE *= φ
        end
    end
    G0 = f0.Rn - f0.H - LE0 + Q_extra
    D = dRn - dH - dLE
    st = step_soil_heat!(col, dt, LinearizedSkinBalance(G0, D, T0))
    ΔT = st.T_skin_mean - T0
    Rn = f0.Rn + dRn * ΔT
    H = f0.H + dH * ΔT
    LE = LE0 + dLE * ΔT
    E = E0 + dE * ΔT
    fl = SurfaceFluxes{T}(Rn, H, LE, Rn - H - LE + Q_extra, f0.τ, E)
    if col.hydrology
        ws = step_soil_water!(col, f, fl, dt)
        # When the soil limits evaporation (its surface reaches the dry
        # potential), the water step delivers less than the E the skin
        # balance charged. Return what was delivered, so the atmosphere
        # receives exactly the water the soil lost, and keep the latent
        # heat that was charged but not spent in the soil: it goes into the
        # top layer, and G grows by the same amount (energy conserved).
        E_del = T(ws.evaporation * c.ρ_water / dt)
        if E_del != E
            λ = E != 0 ? LE / E : surface_latent_heat(col; c = c)
            ΔLE = (E - E_del) * λ
            col.H[1] += ΔLE * dt / col.water.grid.dz[1]
            _apply_enthalpy!(col)
            col.budget.surface += ΔLE * dt
            col.budget.latent_mismatch += ΔLE * dt
            fl = SurfaceFluxes{T}(Rn, H, LE - ΔLE, fl.G + ΔLE, f0.τ, E_del)
        end
    end
    return fl
end


# =========================================================================
# 10. Water table and unconfined aquifer (#17)
# =========================================================================

"""
    UnconfinedAquifer(; W = 4.8, S_y = 0.2, D = 25.0, q_max = 5.5e-6, f = 2.5,
                      α_ice = 3.0, frozen_restriction = true)

A prognostic water table in an unconfined aquifer below the soil column
— the SIMTOP groundwater model of Niu et al. (2005, 2007) as used in
CLM4.0 (Oleson et al. 2010, §7.5) — used as the column's `bottom`
boundary:

    col = SoilWaterColumn(grid, soil; bottom = UnconfinedAquifer())

**State.** `W` (m of water) is the water stored in an aquifer of
thickness `D` (m) whose top is the base of the soil column ``z_b``. While
the water table ``z_∇`` lies below the column the aquifer drains by
gravity with specific yield ``S_y``,

    z_∇ = z_b + D - W / S_y                      (W ≤ W_max = S_y D)

(Oleson et al. 2010, eq. 7.165; CLM's ``W_a = 5000`` mm, ``z_∇ = z_b`` at
``W_{max}``). When the aquifer is full the water table sits inside the soil
column; the aquifer then stores water only elastically, with the column's
`specific_storage` ``S_s``, and its excess pressure at the base is
``ψ_b = (W - W_{max})/(S_s D)``, so that the hydrostatic water table is
``z_∇ = z_b - ψ_b``. `W` is thus one continuous, monotone store for both
regimes (CLM switches between two separate stores instead).

**Recharge** — the flux across the column's base, downward positive —
is the steady Darcy flux between the bottom node (``ψ_N`` at ``z_N``)
and the water table (``ψ = 0`` at ``z_∇``), or the saturated base
(``ψ_b`` at ``z_b``) when the table is in the column. It is evaluated by
[`gardner_two_point_flux`](@ref), exact for a conductivity exponential
in ``ψ`` between the two points, which keeps capillary rise from a deep
water table realistic where a mean conductivity overstates it by orders
of magnitude. It vanishes exactly for the hydrostatic profile
``ψ_N = -(z_∇ - z_N)``.

**Baseflow** (lateral, topographically controlled drainage; SIMTOP,
Niu et al. 2005, CLM4 eq. 7.159):

    q_drai = (1 - f_imp) q_max exp(-f z_∇)

with ``q_{max} = 5.5×10^{-3}`` kg m⁻² s⁻¹ ``= 5.5×10^{-6}`` m s⁻¹ and
``f = 2.5`` m⁻¹ (CLM4.0 global values, Oleson et al. 2010, p. 155; CLM4.5
replaced ``q_{max}`` by ``10 \\sin β`` kg m⁻² s⁻¹ and ``f_{imp}`` by an ice
impedance factor) and the frozen-soil impermeable
fraction ``f_{imp}`` of [`aquifer_impermeable_fraction`](@ref)
(`frozen_restriction = false` sets it to 0).

**Numerics.** Within a sub-step the recharge is coupled implicitly to the
aquifer (the linearisation handed to the Richards solver includes the
response of the water table, which is stiff when the aquifer is full),
and after convergence `W` is advanced by the *exact* solution of
``dW/dt = q - q_{drai}(W)`` at the converged recharge ``q`` — for a
constant ``q`` the substitution ``u = e^{f z_∇}`` makes it linear in both
regimes. Water is therefore conserved to round-off by
[`total_water_storage`](@ref), and the recession
``q_{drai}(t) = q_0/(1 + t/τ)``, ``τ = S_y/(f q_0)``, is reproduced for
any step length.

Cumulative fields: `recharge` and `baseflow` (m of water since
construction).

# References

  * Niu, G.-Y., Yang, Z.-L., Dickinson, R. E. & Gulden, L. E. (2005). A
    simple TOPMODEL-based runoff parameterization (SIMTOP) for use in
    global climate models. *J. Geophys. Res.*, 110, D21106.
  * Niu, G.-Y., Yang, Z.-L., Dickinson, R. E., Gulden, L. E. & Su, H.
    (2007). Development of a simple groundwater model for use in climate
    models and evaluation with Gravity Recovery and Climate Experiment
    data. *J. Geophys. Res.*, 112, D07103.
  * Oleson, K. W. et al. (2010). Technical description of version 4.0 of
    the Community Land Model (CLM). NCAR/TN-478+STR, §7.4.2.4 and §7.5.
"""
mutable struct UnconfinedAquifer{T<:AbstractFloat} <: AbstractSoilBottomBoundary
    W::T
    S_y::T
    D::T
    q_max::T
    f::T
    α_ice::T
    frozen_restriction::Bool
    recharge::T
    baseflow::T
end
function UnconfinedAquifer(; W::Real = 4.8, S_y::Real = 0.2, D::Real = 25.0,
                           q_max::Real = 5.5e-6, f::Real = 2.5, α_ice::Real = 3.0,
                           frozen_restriction::Bool = true)
    (0 < S_y <= 1 && D > 0 && q_max >= 0 && f > 0 && α_ice > 0) ||
        throw(ArgumentError("UnconfinedAquifer: need 0 < S_y ≤ 1, D > 0, q_max ≥ 0, f > 0, α_ice > 0"))
    W >= 0 || throw(ArgumentError("UnconfinedAquifer: W ≥ 0"))
    UnconfinedAquifer{Float64}(W, S_y, D, q_max, f, α_ice, frozen_restriction, 0.0, 0.0)
end

_aq_Wmax(aq::UnconfinedAquifer) = aq.S_y * aq.D
_aq_Ce(aq::UnconfinedAquifer, col) = col.specific_storage * aq.D
# storage W ↔ water-table depth z (m, positive down), continuous at z = z_b
function _aq_depth(aq::UnconfinedAquifer, col, W)
    zb = col.grid.zh[end]
    W <= _aq_Wmax(aq) ? zb + aq.D - W / aq.S_y : zb - (W - _aq_Wmax(aq)) / _aq_Ce(aq, col)
end
function _aq_storage(aq::UnconfinedAquifer, col, z)
    zb = col.grid.zh[end]
    z >= zb ? aq.S_y * (zb + aq.D - z) : _aq_Wmax(aq) + _aq_Ce(aq, col) * (zb - z)
end
# capacity C = dW/d(-z∇) of the regime W is in
_aq_capacity(aq::UnconfinedAquifer, col, W) = W <= _aq_Wmax(aq) ? aq.S_y : _aq_Ce(aq, col)

"""
    water_table_depth(aq::UnconfinedAquifer, col) -> z_∇ (m)
    water_table_depth(col) -> z_∇ (m) or `nothing`

Depth of the water table below the soil surface. For a column over an
[`UnconfinedAquifer`](@ref) it is the aquifer's (hydrostatic) water
table, which may lie inside the column. For any other lower boundary it
is diagnosed from the column itself: the depth where ``ψ`` crosses zero
above the saturated zone that reaches the base (linear interpolation
between nodes), or `nothing` if the bottom layer is unsaturated.
"""
water_table_depth(aq::UnconfinedAquifer, col) = _aq_depth(aq, col, aq.W)
function water_table_depth(col::SoilWaterColumn)
    col.bottom isa UnconfinedAquifer && return water_table_depth(col.bottom, col)
    return _column_water_table(col)
end
function _column_water_table(col::SoilWaterColumn)
    ψ, z = col.ψ, col.grid.z
    N = length(z)
    ψ[N] >= 0 || return nothing
    for i in N:-1:2
        if ψ[i-1] < 0
            return z[i] - ψ[i] * (z[i] - z[i-1]) / (ψ[i] - ψ[i-1])
        end
    end
    return max(zero(ψ[1]), z[1] - ψ[1])
end

"""
    aquifer_impermeable_fraction(aq, col) -> f_imp ∈ [0, 1]

Fraction of the drainage area made impermeable by ground ice (CLM4
eq. 7.160, after Niu & Yang 2006):

    f_imp = [exp(-α(1 - F)) - exp(-α)] / [1 - exp(-α)],   α = aq.α_ice

where ``F`` is the thickness-weighted ice mass fraction
``w_{ice}/(w_{ice} + w_{liq})`` of the layers from the one holding the
water table (the bottom layer when the table is below the column) down
to the base. Zero for an ice-free column or `frozen_restriction = false`.
"""
function aquifer_impermeable_fraction(aq::UnconfinedAquifer, col)
    aq.frozen_restriction || return zero(aq.W)
    g, c = col.grid, col.constants
    N = length(g)
    zwt = water_table_depth(aq, col)
    j = N
    while j > 1 && g.zh[j] > zwt
        j -= 1
    end
    num = den = zero(aq.W)
    for i in j:N
        mi = c.ρ_ice * col.θ_ice[i]
        ml = c.ρ_water * col.θ_liq[i]
        F = mi + ml > 0 ? mi / (mi + ml) : zero(mi)
        num += F * g.dz[i]
        den += g.dz[i]
    end
    F = num / den
    α = aq.α_ice
    return max(zero(F), (exp(-α * (1 - F)) - exp(-α)) / (1 - exp(-α)))
end

_aq_q0(aq::UnconfinedAquifer, col) = (1 - aquifer_impermeable_fraction(aq, col)) * aq.q_max

"""
    aquifer_baseflow_rate(col) -> q_drai (m s⁻¹)
    aquifer_baseflow_rate(aq, col)

Instantaneous SIMTOP baseflow ``(1 - f_{imp}) q_{max} e^{-f z_∇}`` of the
column's [`UnconfinedAquifer`](@ref) in its current state.
"""
aquifer_baseflow_rate(aq::UnconfinedAquifer, col) =
    _aq_q0(aq, col) * exp(-aq.f * water_table_depth(aq, col))
aquifer_baseflow_rate(col::SoilWaterColumn) = aquifer_baseflow_rate(col.bottom, col)

"""
    gardner_two_point_flux(K_1, K_2, ψ_1, ψ_2, L) -> q (m s⁻¹, downward)

Steady vertical Darcy flux from a point with potential `ψ_1` and
conductivity `K_1` down to a point `L` metres below with `ψ_2`, `K_2`,
assuming the conductivity varies exponentially with potential between
them, ``K ∝ e^{α ψ}`` with ``α = \\ln(K_2/K_1)/(ψ_2 - ψ_1)``. Integrating
``q = K (1 - dψ/dz)`` exactly (Gardner 1958) gives

    q = (K_1 - K_2 e^{-αL}) / (1 - e^{-αL}),

which is exact for a [`GardnerExponential`](@ref) soil, reduces to Darcy's
law ``q = K (1 - (ψ_2 - ψ_1)/L)`` for uniform ``K``, and vanishes for the
hydrostatic pair ``ψ_1 = ψ_2 - L`` whatever the soil. Used for the
recharge between the bottom soil node and the water table.
"""
function gardner_two_point_flux(K1::Real, K2::Real, ψ1::Real, ψ2::Real, L::Real)
    L > 0 || throw(ArgumentError("gardner_two_point_flux: L > 0"))
    K1, K2, ψ1, ψ2, L = promote(float(K1), float(K2), ψ1, ψ2, L)
    tiny = floatmin(K1)
    K1 = max(K1, tiny); K2 = max(K2, tiny)
    Δψ = ψ2 - ψ1
    x = log(K2 / K1)
    # uniform K (or K rising against ψ, which only a layered pair can do):
    # plain Darcy with the logarithmic mean conductivity
    if x == 0 || Δψ == 0 || x / Δψ <= 0
        Km = x == 0 ? K1 : (K2 - K1) / x
        return Km * (1 - Δψ / L)
    end
    y = x / Δψ * L                       # αL > 0
    x - y <= 700 && return K1 * expm1(x - y) / expm1(-y)
    return (K1 - K2 * exp(-y)) / (-expm1(-y))
end

# Steady recharge q(ψ_N, ψ_t) from the bottom node to the target point
# (water table or saturated base) at depth z_t.
_aq_flux(col, ψN, ψt, L) = gardner_two_point_flux(_layer_K(col, length(col.grid), ψN),
                                                  _layer_K(col, length(col.grid), ψt),
                                                  ψN, ψt, L)

# The implicit aquifer coupling. For the current bottom-node potential ψ_N,
# find the end-of-step storage W₁ solving the backward-Euler balance
#   G(W₁) = W₁ - W₀ - Δt [q(ψ_N, ψ_t(W₁)) - q_drai(W₁)] = 0,
# where the target point is frozen at the start-of-step depth z_t = max(z_∇, z_b)
# and ψ_t(W₁) = z_t - z_∇(W₁) is the hydrostatic potential there; then return
# the tangent q ≈ a + b ψ_N including dW₁/dψ_N. G is strictly increasing
# (q falls and q_drai rises with W), so a safeguarded Newton iteration on a
# bracket always converges.
function bottom_flux_linearization(aq::UnconfinedAquifer, col, ψN, KN, dt)
    col.gravity > 0 ||
        throw(ArgumentError("UnconfinedAquifer: needs a column with gravity"))
    T = typeof(aq.W)
    N = length(col.grid)
    zN, zb = col.grid.z[N], col.grid.zh[end]
    W0 = aq.W
    z0 = _aq_depth(aq, col, W0)
    zt = max(z0, zb)
    L = zt - zN
    q0 = _aq_q0(aq, col)
    f = aq.f
    ψt(W) = zt - _aq_depth(aq, col, W)
    qdrai(W) = q0 * exp(-f * _aq_depth(aq, col, W))
    δ(ψ) = 1e-7 * max(one(ψ), abs(ψ))
    function Φ2(ψ2)                                  # ∂q/∂ψ_t
        h = δ(ψ2)
        (_aq_flux(col, ψN, ψ2 + h, L) - _aq_flux(col, ψN, ψ2 - h, L)) / 2h
    end
    G(W) = W - W0 - dt * (_aq_flux(col, ψN, ψt(W), L) - qdrai(W))
    function dG(W)
        C = _aq_capacity(aq, col, W)
        1 + dt * (-Φ2(ψt(W)) + f * qdrai(W)) / C
    end
    # bracket (see the monotonicity argument above)
    g0 = G(W0)
    tolW = 4eps(T) * max(one(T), abs(W0))
    if g0 <= 0
        lo, hi = W0, W0 - g0 + tolW
        k = 0
        while G(hi) < 0 && k < 60; hi += (hi - W0) + tolW; k += 1; end
    else
        lo, hi = W0 - g0 - tolW, W0
        k = 0
        while G(lo) > 0 && k < 60; lo -= (W0 - lo) + tolW; k += 1; end
    end
    W = g0 <= 0 ? lo : hi
    for _ in 1:100
        g = G(W)
        abs(g) <= tolW && break
        g > 0 ? (hi = W) : (lo = W)
        Wn = W - g / dG(W)
        (lo < Wn < hi) || (Wn = (lo + hi) / 2)
        done = abs(Wn - W) <= tolW
        W = Wn
        done && break
    end
    # tangent at the solution
    C = _aq_capacity(aq, col, W)
    ψ2 = ψt(W)
    q = _aq_flux(col, ψN, ψ2, L)
    h = δ(ψN)
    Φ1 = (_aq_flux(col, ψN + h, ψ2, L) - _aq_flux(col, ψN - h, ψ2, L)) / 2h
    b = Φ1 * (1 + dt * f * qdrai(W) / C) / dG(W)
    b = max(b, zero(b))
    return (T(q - b * ψN), T(b))
end

# Exact water-table motion over time t at constant recharge q within one
# regime of capacity C: with u = e^{f z}, du/dt = (f/C)(q₀ - q u). Returns
# z(t) and, when the path would cross z_b, the crossing time instead.
function _aq_regime_evolve(z0, q, q0, f, C, t, zb, deep::Bool)
    Qd0 = q0 * exp(-f * z0)
    λ = f * q / C
    φ(x) = x == 0 ? one(x) : -expm1(-x) / x
    v = exp(-λ * t) + Qd0 * (f * t / C) * φ(λ * t)          # u(t)/u(0)
    z1 = z0 + log(v) / f
    crosses = deep ? z1 < zb : z1 > zb
    crosses || return (z1, t)
    vc = exp(f * (zb - z0))
    tc = if q == 0
        (vc - 1) * C / (f * Qd0)
    else
        v∞ = Qd0 / q
        -log((vc - v∞) / (1 - v∞)) / λ
    end
    tc = isfinite(tc) ? clamp(tc, zero(t), t) : t
    return (zb, tc)
end

function finalize_bottom!(aq::UnconfinedAquifer, col, q, dt)
    q0 = _aq_q0(aq, col)
    zb = col.grid.zh[end]
    W0 = aq.W
    z = _aq_depth(aq, col, W0)
    t = zero(dt)
    for _ in 1:3                               # at most one regime change
        t >= dt && break
        Qd = q0 * exp(-aq.f * z)
        deep = z > zb || (z == zb && q <= Qd)
        C = deep ? aq.S_y : _aq_Ce(aq, col)
        z, tc = _aq_regime_evolve(z, q, q0, aq.f, C, dt - t, zb, deep)
        t = tc >= dt - t ? dt : t + tc
    end
    z <= zb + aq.D ||
        throw(DomainError(z, "UnconfinedAquifer: the aquifer has run dry (water table below its base)"))
    W1 = _aq_storage(aq, col, z)
    aq.W = W1
    aq.recharge += q * dt
    aq.baseflow += q * dt - (W1 - W0)
    return nothing
end

"""
    bottom_water_storage(bc) -> m

Water held by a stateful lower boundary: the aquifer storage `W` of an
[`UnconfinedAquifer`](@ref), zero for the stateless boundaries.
"""
bottom_water_storage(::AbstractSoilBottomBoundary) = 0.0
bottom_water_storage(aq::UnconfinedAquifer) = aq.W

"""
    total_water_storage(col) -> m

Soil water ([`soil_water_storage`](@ref)) plus the water held by the
lower boundary ([`bottom_water_storage`](@ref)). Over an
[`UnconfinedAquifer`](@ref) its change equals infiltration minus
extraction minus baseflow, to round-off.
"""
total_water_storage(col::SoilWaterColumn) = soil_water_storage(col) + bottom_water_storage(col.bottom)

"""
    set_water_table!(col, z_∇)

Put the column in hydrostatic equilibrium with a water table at depth
`z_∇` (m): ``ψ_i = z_i - z_∇`` in every layer (saturated, with elastic
storage, below the table), and — over an [`UnconfinedAquifer`](@ref) —
the aquifer storage consistent with it.
"""
function set_water_table!(col::SoilWaterColumn, zwt::Real)
    set_matric_potential!(col, [z - zwt for z in col.grid.z])
    if col.bottom isa UnconfinedAquifer
        aq = col.bottom
        zwt <= col.grid.zh[end] + aq.D ||
            throw(ArgumentError("set_water_table!: water table below the aquifer base"))
        aq.W = _aq_storage(aq, col, zwt)
    end
    return col
end

# =========================================================================
# 11. Initial state from data (#521)
# =========================================================================

"""
    soil_profile_on_nodes(z_data, v, z) -> Vector or nothing

The profile `v` given at the depths `z_data` (m, positive down, strictly
increasing) on the node depths `z`: linear in depth between the levels
with data (`NaN` levels are skipped: a fill value), the shallowest value
above the first level and the deepest below the last (PALM's and INIFOR's
rule: the deepest level is held, and is the base temperature). `nothing`
when no level has data.
"""
function soil_profile_on_nodes(z_data::AbstractVector, v::AbstractVector, z::AbstractVector)
    length(z_data) == length(v) ||
        throw(ArgumentError("soil_profile_on_nodes: $(length(v)) values for $(length(z_data)) depths"))
    keep = findall(!isnan, v)
    isempty(keep) && return nothing
    zs, vs = Float64.(z_data[keep]), Float64.(v[keep])
    all(>(0), diff(zs)) || throw(ArgumentError("soil_profile_on_nodes: the depths must increase strictly"))
    return map(z) do zz
        zz <= zs[1] && return vs[1]
        zz >= zs[end] && return vs[end]
        n = searchsortedlast(zs, zz)
        w = (zz - zs[n]) / (zs[n+1] - zs[n])
        (1 - w) * vs[n] + w * vs[n+1]
    end
end

"The length of the annual cycle of [`annual_soil_temperature`](@ref) (s): 365.25 d."
const SOIL_YEAR = 365.25 * 86400.0

"""
    soil_damping_depth(k, C; period = SOIL_YEAR) -> d (m)

The damping depth ``d = \\sqrt{2κ/ω} = \\sqrt{k P/(π C)}`` of a periodic
surface temperature wave of `period` ``P`` (s) in a soil of conductivity
`k` (W m⁻¹ K⁻¹) and volumetric heat capacity `C` (J m⁻³ K⁻¹): the depth
over which its amplitude falls by ``e`` (Carslaw & Jaeger 1959, §2.6).
About 2–3 m for the annual wave in moist loam.
"""
soil_damping_depth(k::Real, C::Real; period::Real = SOIL_YEAR) = sqrt(k * period / (π * C))

"""
    annual_soil_temperature(z, t; T_mean, amplitude, damping_depth, t_max, period = SOIL_YEAR) -> T (K)

The periodic solution of the heat equation in a homogeneous half-space
under a surface temperature ``T̄ + A\\cos(ω(t - t_{max}))``:

```math
T(z, t) = T̄ + A\\, e^{-z/d} \\cos\\bigl(ω (t - t_{max}) - z/d\\bigr),\\qquad ω = 2π/P,
```

`z` the depth (m, positive down), `t` and `t_max` (the time of the
surface maximum) in s on the same axis, `d` the [`soil_damping_depth`](@ref)
(Carslaw & Jaeger 1959, §2.6). The initial soil without data (#521): it
starts the slow modes (τ ≈ 30 d for the CLM column's 3.4 m) on the
annual wave instead of at a uniform temperature they then relax from.
"""
annual_soil_temperature(z::Real, t::Real; T_mean::Real, amplitude::Real, damping_depth::Real, t_max::Real,
                        period::Real = SOIL_YEAR) =
    T_mean + amplitude * exp(-z / damping_depth) * cos(2π * (t - t_max) / period - z / damping_depth)

# =============================================================================
# 12. The packed state (#526)
# =============================================================================
#
# What a checkpoint must hold to continue a column exactly: the layer
# profiles (ψ, θ_liq, θ_ice, T_soil; the thermal column's enthalpy H), the
# skin temperature, the adaptive sub-steps and the budgets. Not packed: the
# grid, the hydraulic and thermal properties, the boundary constants (set
# by the configuration) and the solver workspaces `ws` (scratch, written
# before they are read in every step).

# the evolving part of a bottom boundary: the aquifer's storage and its
# last fluxes; the other boundaries are constants
_bottom_state_length(::UnconfinedAquifer) = 3
_bottom_state_length(b) = 0
_bottom_packable(b) = b isa UnconfinedAquifer || !ismutable(b)
function _pack_bottom!(buf, i, b::UnconfinedAquifer)
    buf[i] = b.W; buf[i+1] = b.recharge; buf[i+2] = b.baseflow
    return i + 3
end
_pack_bottom!(buf, i, b) = i
function _unpack_bottom!(b::UnconfinedAquifer, buf, i)
    b.W = buf[i]; b.recharge = buf[i+1]; b.baseflow = buf[i+2]
    return i + 3
end
_unpack_bottom!(b, buf, i) = i

@inline function _pack_vec!(buf, i, v)
    n = length(v)
    @inbounds for l in 1:n
        buf[i+l-1] = v[l]
    end
    return i + n
end
@inline function _unpack_vec!(v, buf, i)
    n = length(v)
    @inbounds for l in 1:n
        v[l] = buf[i+l-1]
    end
    return i + n
end

SurfaceExchange.has_packed_state(c::SoilWaterColumn) = _bottom_packable(c.top) && _bottom_packable(c.bottom)
SurfaceExchange.packed_state_length(c::SoilWaterColumn) = 4 * length(c.ψ) + 1 + 6 + _bottom_state_length(c.bottom)
function SurfaceExchange.pack_state!(buf, i, c::SoilWaterColumn)
    i = _pack_vec!(buf, i, c.ψ); i = _pack_vec!(buf, i, c.θ_liq)
    i = _pack_vec!(buf, i, c.θ_ice); i = _pack_vec!(buf, i, c.T_soil)
    b = c.budget
    buf[i] = c.dt_sub
    buf[i+1] = b.infiltration; buf[i+2] = b.runoff; buf[i+3] = b.evaporation
    buf[i+4] = b.drainage; buf[i+5] = b.extraction; buf[i+6] = b.time
    return _pack_bottom!(buf, i + 7, c.bottom)
end
function SurfaceExchange.unpack_state!(c::SoilWaterColumn, buf, i)
    i = _unpack_vec!(c.ψ, buf, i); i = _unpack_vec!(c.θ_liq, buf, i)
    i = _unpack_vec!(c.θ_ice, buf, i); i = _unpack_vec!(c.T_soil, buf, i)
    b = c.budget
    c.dt_sub = buf[i]
    b.infiltration = buf[i+1]; b.runoff = buf[i+2]; b.evaporation = buf[i+3]
    b.drainage = buf[i+4]; b.extraction = buf[i+5]; b.time = buf[i+6]
    return _unpack_bottom!(c.bottom, buf, i + 7)
end

SurfaceExchange.has_packed_state(c::SoilThermalColumn) =
    has_packed_state(c.water) && _bottom_packable(c.bottom)
SurfaceExchange.packed_state_length(c::SoilThermalColumn) =
    packed_state_length(c.water) + length(c.H) + 2 + 5
function SurfaceExchange.pack_state!(buf, i, c::SoilThermalColumn)
    i = pack_state!(buf, i, c.water)
    i = _pack_vec!(buf, i, c.H)
    b = c.budget
    buf[i] = c.T_skin; buf[i+1] = c.dt_sub
    buf[i+2] = b.surface; buf[i+3] = b.bottom; buf[i+4] = b.water
    buf[i+5] = b.latent_mismatch; buf[i+6] = b.time
    return i + 7
end
function SurfaceExchange.unpack_state!(c::SoilThermalColumn, buf, i)
    i = unpack_state!(c.water, buf, i)
    i = _unpack_vec!(c.H, buf, i)
    b = c.budget
    c.T_skin = buf[i]; c.dt_sub = buf[i+1]
    b.surface = buf[i+2]; b.bottom = buf[i+3]; b.water = buf[i+4]
    b.latent_mismatch = buf[i+5]; b.time = buf[i+6]
    return i + 7
end

end # module
