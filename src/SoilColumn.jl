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

Downward flux through the base of the column, ``q = a + b ψ_N`` (m s⁻¹).
See [`AbstractSoilBottomBoundary`](@ref).
"""
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
| `θ_ice`   | ice content, m³ m⁻³ (zero until the freeze–thaw model, #16)    |
| `T_soil`  | layer temperature, K (prescribed until the heat equation, #16) |
| `top`, `bottom` | boundary conditions; a stateful `bottom` (#17's water table) lives here |
| `budget`  | cumulative [`SoilWaterBudget`](@ref)                           |

The layer conductivity is
``K_i = K(ψ_i)\\, μ(T_{ref})/μ(T_i)\\, f_{ice}(θ_{ice,i})``
([`viscosity_conductivity_factor`](@ref), [`ice_impedance_factor`](@ref));
`temperature_dependent = false` drops the viscosity factor. Saturated
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

function _layer_W(col, i, ψ)
    h = col.hydraulics[i]
    ψae = air_entry_potential(h)
    ψ >= ψae ? soil_porosity(h) + col.specific_storage * (ψ - ψae) : soil_water_content(h, ψ)
end
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
elastic storage (``ψ = ψ_{ae} + (θ - θ_s)/S_s``).
"""
function set_water_content!(col::SoilWaterColumn, θ)
    θv = _per_layer(θ, length(col.grid), "θ")
    for i in eachindex(col.ψ)
        h = col.hydraulics[i]
        θv[i] > residual_water_content(h) ||
            throw(ArgumentError("set_water_content!: θ must exceed the residual content of layer $i"))
        col.θ_liq[i] = θv[i]
        col.ψ[i] = θv[i] >= soil_porosity(h) ?
                   air_entry_potential(h) + (θv[i] - soil_porosity(h)) / col.specific_storage :
                   soil_matric_potential(h, θv[i])
    end
    return col
end

"""
    soil_water_storage(col; c = col.constants) -> m

Total water in the column as a depth of liquid water,
``Σ_i Δz_i (θ_{liq,i} + θ_{ice,i} ρ_{ice}/ρ_w)`` (densities from the
Era V contract).
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
with ``ρ_w`` and ``g`` from the Era V contract.
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
        αb, βb = bottom_flux_linearization(col.bottom, col, ψm[N], ws.K[N])
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
                ws.θnew[i] > residual_water_content(col.hydraulics[i]) ||
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
                error("step_soil_water!: Picard iteration failed to converge for sub-steps down to $(col.dt_min) s")
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

Couple the column to the Era V surface interface: rain from `forcing`
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

end # module
