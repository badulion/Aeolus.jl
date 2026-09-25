"""
    SurfaceExchange

The Era V shared contract: the physics every surface model needs, in
one place, so the soil, canopy, urban-facet, snow, sea-ice, ocean and
tile models of Era V delegate rather than re-derive. It is to Era V what
`RadiationCore` is to Era III and `MoistThermodynamics` to Era IV.

Three parts:

  1. **Material constants** — [`SurfaceConstants`](@ref): densities and
     conductivities of water, ice and air, the Stefan–Boltzmann and von
     Kármán constants. Latent heats, heat capacities of water and ice and
     `g` are *delegated* to `MoistThermodynamics.ThermodynamicConstants`
     through accessors; they are never restated here.
  2. **Monin–Obukhov similarity** — the stability functions
     ``φ_m, φ_h, ψ_m, ψ_h`` as a dispatch hierarchy
     ([`AbstractStabilityFunctions`](@ref)) and a surface-layer solver
     ([`solve_surface_layer`](@ref)) for ``u_*``, ``θ_*``, ``q_*``, the
     Obukhov length and the bulk transfer coefficients.
  3. **The surface interface** — [`AbstractSurface`](@ref) and the
     generics a surface model implements (`surface_temperature`,
     `roughness_lengths`, `surface_albedo`, `surface_emissivity`,
     `step_surface!`), plus a default [`turbulent_fluxes`](@ref) that
     turns any implementer into sensible, latent and momentum fluxes.

# Sign convention

All turbulent and radiative fluxes are **positive upward** (away from
the surface) except net radiation `Rn`, which is positive *into* the
surface, and `G`, positive *into* the substrate — so the surface energy
balance reads ``R_n = H + LE + G``. Scales follow
``\\overline{w'θ'} = -u_* θ_*`` and ``\\overline{w'q'} = -u_* q_*``: a
surface warmer than the air gives ``θ_* < 0``, ``H > 0`` and ``L < 0``
(unstable). ``ζ = z/L`` is positive for stable stratification.

# References

  * Businger, J. A., Wyngaard, J. C., Izumi, Y. & Bradley, E. F. (1971).
    Flux-profile relationships in the atmospheric surface layer.
    *J. Atmos. Sci.*, 28, 181-189.
  * Dyer, A. J. (1974). A review of flux-profile relationships.
    *Boundary-Layer Meteorol.*, 7, 363-372.
  * Paulson, C. A. (1970). The mathematical representation of wind speed
    and temperature profiles in the unstable atmospheric surface layer.
    *J. Appl. Meteorol.*, 9, 857-861.
  * Beljaars, A. C. M. & Holtslag, A. A. M. (1991). Flux parameterization
    over land surfaces for atmospheric models. *J. Appl. Meteorol.*, 30,
    327-341.
  * Fukusako, S. (1990). Thermophysical properties of ice, snow, and sea
    ice. *Int. J. Thermophys.*, 11, 353-372.
"""
module SurfaceExchange

using ..MoistThermodynamics
using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS,
                             PhasePartition, R_d, R_v
import ..MoistThermodynamics: LH_f0
using ..RadiationCore: STEFAN_BOLTZMANN
# The broadband albedo / emissivity of an AbstractSurface are further
# methods of the RadiationCore generics — one function, not two.
import ..RadiationCore: surface_albedo, surface_emissivity

export SurfaceConstants, DEFAULT_SURFACE_CONSTANTS,
       LH_v0, LH_s0, LH_f0, grav, cp_water, cp_ice, von_karman,
       AbstractStabilityFunctions, BusingerDyerBeljaarsHoltslag,
       phi_m, phi_h, psi_m, psi_h

# =========================================================================
# 1. Material constants
# =========================================================================

"""
    SurfaceConstants(; kwargs...)

Material constants of the surface models. Fields (SI units):

| field     | default          | meaning                                          |
|:--------- |:---------------- |:------------------------------------------------ |
| `ρ_water` | 1000             | liquid water density, kg m⁻³                     |
| `ρ_ice`   | 917              | ice density at 0 °C, kg m⁻³                      |
| `k_water` | 0.57             | liquid water conductivity near 0–10 °C, W m⁻¹ K⁻¹ |
| `k_ice`   | 2.22             | ice conductivity at 0 °C, W m⁻¹ K⁻¹ (Fukusako 1990) |
| `k_air`   | 0.024            | still-air conductivity near 0 °C, W m⁻¹ K⁻¹      |
| `σ_SB`    | `STEFAN_BOLTZMANN` | Stefan–Boltzmann, W m⁻² K⁻⁴ (CODATA 2018, from `RadiationCore`) |
| `κ`       | 0.40             | von Kármán constant                              |
| `thermo`  | `DEFAULT_CONSTANTS` | the `MoistThermodynamics` constants            |

Latent heats, the heat capacities of water and ice, and gravity are
**not** fields: they belong to `thermo` and are read through the
accessors [`LH_v0`](@ref), [`LH_s0`](@ref), `LH_f0`, [`grav`](@ref),
[`cp_water`](@ref), [`cp_ice`](@ref), so a surface model can never
disagree with the atmosphere about how much energy evaporation costs.
"""
Base.@kwdef struct SurfaceConstants{T<:AbstractFloat, C<:ThermodynamicConstants}
    ρ_water::T = 1000.0
    ρ_ice::T   = 917.0
    k_water::T = 0.57
    k_ice::T   = 2.22
    k_air::T   = 0.024
    σ_SB::T    = STEFAN_BOLTZMANN
    κ::T       = 0.40
    thermo::C  = DEFAULT_CONSTANTS
end

"The default [`SurfaceConstants`](@ref) (Float64)."
const DEFAULT_SURFACE_CONSTANTS = SurfaceConstants()

"Latent heat of vaporisation at the triple point (delegated to `MoistThermodynamics`)."
LH_v0(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.LH_v0
"Latent heat of sublimation at the triple point (delegated to `MoistThermodynamics`)."
LH_s0(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.LH_s0
LH_f0(c::SurfaceConstants) = LH_f0(c.thermo)
"Gravitational acceleration (delegated to `MoistThermodynamics`)."
grav(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.grav
"Specific heat of liquid water, `thermo.cp_liquid`."
cp_water(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.cp_liquid
"Specific heat of ice, `thermo.cp_ice`."
cp_ice(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.thermo.cp_ice
"The von Kármán constant."
von_karman(c::SurfaceConstants = DEFAULT_SURFACE_CONSTANTS) = c.κ

# =========================================================================
# 2. Monin–Obukhov stability functions
# =========================================================================

"""
    AbstractStabilityFunctions

A family of Monin–Obukhov similarity functions. A concrete subtype
implements the four generics

    phi_m(sf, ζ), phi_h(sf, ζ)   # dimensionless gradients
    psi_m(sf, ζ), psi_h(sf, ζ)   # integrated forms

with ``ζ = z/L`` (positive stable), normalised so ``φ(0) = 1`` and
``ψ(0) = 0``, and related by ``ψ(ζ) = ∫₀^ζ [1 - φ(x)]/x dx``, i.e.
``dψ/dζ = (1 - φ)/ζ``. Profiles then read

    u(z) = (u_*/κ) [ln(z/z₀ₘ) - ψ_m(z/L) + ψ_m(z₀ₘ/L)]

and likewise for θ and q with ``ψ_h``. The surface-layer solver only
calls `psi_m` and `psi_h`, so a new family (e.g. the COARE convective
form of Grachev & Fairall) needs nothing else to be usable.
"""
abstract type AbstractStabilityFunctions end

"""
    BusingerDyerBeljaarsHoltslag(; γ_m = 16, γ_h = 16, a = 1, b = 2/3, c = 5, d = 0.35)

The default stability functions.

*Unstable* (``ζ < 0``): the Businger–Dyer gradients in Dyer's (1974)
form with κ = 0.40,
``φ_m = (1 - γ_m ζ)^{-1/4}``, ``φ_h = (1 - γ_h ζ)^{-1/2}``, and their
exact integrals (Paulson 1970), with ``x = (1 - γ_m ζ)^{1/4}``,
``y = (1 - γ_h ζ)^{1/2}``:

    ψ_m = 2 ln[(1+x)/2] + ln[(1+x²)/2] - 2 atan(x) + π/2
    ψ_h = 2 ln[(1+y)/2]

*Stable* (``ζ ≥ 0``): Beljaars & Holtslag (1991), which stay finite and
well-behaved in very stable conditions where the log-linear
``φ = 1 + 5ζ`` shuts turbulence off:

    ψ_m = -[a ζ + b (ζ - c/d) e^{-dζ} + b c/d]
    ψ_h = -[(1 + 2aζ/3)^{3/2} + b (ζ - c/d) e^{-dζ} + b c/d - 1]

with φ obtained analytically as ``φ = 1 - ζ dψ/dζ``.
"""
Base.@kwdef struct BusingerDyerBeljaarsHoltslag{T<:AbstractFloat} <: AbstractStabilityFunctions
    γ_m::T = 16.0
    γ_h::T = 16.0
    a::T   = 1.0
    b::T   = 2 / 3
    c::T   = 5.0
    d::T   = 0.35
end

"Dimensionless wind shear ``φ_m(ζ) = (κz/u_*) ∂u/∂z``."
function phi_m end
"Dimensionless potential-temperature (and scalar) gradient ``φ_h(ζ)``."
function phi_h end
"Integrated stability correction for momentum, ``ψ_m(ζ)``."
function psi_m end
"Integrated stability correction for heat and scalars, ``ψ_h(ζ)``."
function psi_h end

function phi_m(sf::BusingerDyerBeljaarsHoltslag, ζ::Real)
    ζ < 0 && return (1 - sf.γ_m * ζ)^(-1 / 4)
    return 1 + ζ * (sf.a + sf.b * exp(-sf.d * ζ) * (1 + sf.c - sf.d * ζ))
end

function phi_h(sf::BusingerDyerBeljaarsHoltslag, ζ::Real)
    ζ < 0 && return (1 - sf.γ_h * ζ)^(-1 / 2)
    return 1 + ζ * (sf.a * sqrt(1 + 2 * sf.a * ζ / 3) +
                    sf.b * exp(-sf.d * ζ) * (1 + sf.c - sf.d * ζ))
end

function psi_m(sf::BusingerDyerBeljaarsHoltslag, ζ::Real)
    if ζ < 0
        x = (1 - sf.γ_m * ζ)^(1 / 4)
        return 2 * log((1 + x) / 2) + log((1 + x^2) / 2) - 2 * atan(x) + π / 2
    end
    a, b, c, d = sf.a, sf.b, sf.c, sf.d
    return -(a * ζ + b * (ζ - c / d) * exp(-d * ζ) + b * c / d)
end

function psi_h(sf::BusingerDyerBeljaarsHoltslag, ζ::Real)
    if ζ < 0
        y = sqrt(1 - sf.γ_h * ζ)
        return 2 * log((1 + y) / 2)
    end
    a, b, c, d = sf.a, sf.b, sf.c, sf.d
    return -((1 + 2 * a * ζ / 3)^(3 / 2) + b * (ζ - c / d) * exp(-d * ζ) + b * c / d - 1)
end

end # module
