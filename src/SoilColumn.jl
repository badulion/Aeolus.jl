"""
    SoilColumn

Era V (Phase 46) multi-layer soil column — CLM-style vertical
discretisation with Richards-equation water transport.

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
     default), the extension point for the water table of #17.

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
"""
module SoilColumn

using ..SurfaceExchange: SurfaceConstants, DEFAULT_SURFACE_CONSTANTS,
                         SurfaceForcing, SurfaceFluxes, grav
import ..SurfaceExchange: moisture_availability

export SoilGrid, clm5_soil_grid, clm_exponential_soil_grid, uniform_soil_grid,
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
       soil_layer_conductivity, soil_field_capacity

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
    VanGenuchtenMualem(; θ_r, θ_s, α, n, K_s, l = 0.5)

van Genuchten (1980) retention with the Mualem (1976) conductivity:

    S_e = [1 + (α|ψ|)^n]^{-m},  m = 1 - 1/n,   θ = θ_r + (θ_s - θ_r) S_e
    K   = K_s S_e^l [1 - (1 - S_e^{1/m})^m]²

`α` in m⁻¹, `K_s` in m s⁻¹, tortuosity `l` = 0.5 (Mualem).
"""
struct VanGenuchtenMualem{T<:AbstractFloat} <: AbstractHydraulicModel
    θ_r::T
    θ_s::T
    α::T
    n::T
    K_s::T
    l::T
    function VanGenuchtenMualem{T}(θ_r, θ_s, α, n, K_s, l) where {T}
        (0 <= θ_r < θ_s < 1 && α > 0 && n > 1 && K_s > 0) ||
            throw(ArgumentError("VanGenuchtenMualem: need 0 ≤ θ_r < θ_s < 1, α > 0, n > 1, K_s > 0"))
        new{T}(θ_r, θ_s, α, n, K_s, l)
    end
end
VanGenuchtenMualem(θ_r::Real, θ_s::Real, α::Real, n::Real, K_s::Real, l::Real = 0.5) =
    VanGenuchtenMualem{Float64}(θ_r, θ_s, α, n, K_s, l)
VanGenuchtenMualem(; θ_r, θ_s, α, n, K_s, l = 0.5) = VanGenuchtenMualem(θ_r, θ_s, α, n, K_s, l)

_vg_m(h::VanGenuchtenMualem) = 1 - 1 / h.n
_vg_Se(h::VanGenuchtenMualem, ψ::Real) =
    ψ >= 0 ? one(float(ψ)) : (1 + (h.α * -ψ)^h.n)^(-_vg_m(h))

soil_water_content(h::VanGenuchtenMualem, ψ::Real) = h.θ_r + (h.θ_s - h.θ_r) * _vg_Se(h, ψ)
function soil_matric_potential(h::VanGenuchtenMualem, θ::Real)
    θ >= h.θ_s && return 0.0
    θ > h.θ_r || return -Inf
    Se = (θ - h.θ_r) / (h.θ_s - h.θ_r)
    return -(Se^(-1 / _vg_m(h)) - 1)^(1 / h.n) / h.α
end
function soil_hydraulic_conductivity(h::VanGenuchtenMualem, ψ::Real)
    ψ >= 0 && return h.K_s
    m = _vg_m(h)
    Se = _vg_Se(h, ψ)
    return h.K_s * Se^h.l * (1 - (1 - Se^(1 / m))^m)^2
end
function soil_water_capacity(h::VanGenuchtenMualem, ψ::Real)
    ψ >= 0 && return 0.0
    m = _vg_m(h)
    x = (h.α * -ψ)^h.n
    return (h.θ_s - h.θ_r) * h.α * m * h.n * (h.α * -ψ)^(h.n - 1) * (1 + x)^(-m - 1)
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
texture classes, in the paper's units — porosity `θ_s`, saturated
conductivity `K_s` (cm s⁻¹), saturated matric potential `ψ_s` (cm, as a
positive suction) and `b`. As reproduced in Tarboton (2003, *Rainfall-
Runoff Processes*, Table 4.1, `K_s` in cm h⁻¹ there). Use
[`clapp_hornberger_texture`](@ref) for an SI model.
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
    carsel_parrish_texture(texture::Symbol) -> VanGenuchtenMualem

SI [`VanGenuchtenMualem`](@ref) soil for a USDA texture class (the keys
of [`CARSEL_PARRISH_1988`](@ref), which also includes `:silt`).
"""
function carsel_parrish_texture(texture::Symbol)
    r = _texture_row(CARSEL_PARRISH_1988, texture, "carsel_parrish_texture")
    VanGenuchtenMualem(r.θ_r, r.θ_s, 100 * r.α, r.n, r.K_s / 360000)
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
    parlange_sorptivity(h, ψ_i; ψ_0 = air_entry_potential(h), n = 4000) -> S (m s^-1/2)

Sorptivity of soil `h` wetted from initial potential `ψ_i` by a
boundary held at `ψ_0`, from Parlange's (1975) approximation

    S² = ∫_{θ_i}^{θ_0} (θ_0 + θ - 2θ_i) D(θ) dθ = ∫_{ψ_i}^{ψ_0} (θ_0 + θ(ψ) - 2θ_i) K(ψ) dψ

(the second form avoids the singular D near saturation). It lies midway
between the rigorous bounds ``2∫(θ - θ_i)D\\,dθ ≤ S² ≤ 2(θ_0 - θ_i)∫D\\,dθ``.
It is built for the steeply rising diffusivities of real soils, where it
is within a few per cent of the exact sorptivity; for a *constant* D it
overestimates by 8.5 % (``\\sqrt{3/2}`` against ``2/\\sqrt{π}``). Integrated with the
trapezoid rule on a grid uniform in ``\\log(ψ_0 - ψ + ε)``.
"""
function parlange_sorptivity(h::AbstractHydraulicModel, ψ_i::Real;
                             ψ_0::Real = air_entry_potential(h), n::Integer = 4000)
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
``μ_{20} = 1.0016`` mPa s. It reproduces the CRC table to 0.15 % over
0–80 °C and extrapolates smoothly into weakly supercooled water.
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

end # module
