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
       LH_v0, LH_s0, LH_f0, grav, cp_water, cp_ice, von_karman

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

end # module
