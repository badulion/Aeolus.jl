"""
    P3Microphysics

Predicted Particle Properties (P3) bulk microphysics — Era IV, Phase 40.

P3 (Morrison & Milbrandt 2015) replaces the traditional split of ice into
fixed categories (cloud ice, snow, graupel, hail) with a single ice
category whose *properties* evolve: prognostic rime mass and rime volume
give a continuously varying bulk density, and the mass-dimension relation
switches between four regimes (small spheres, unrimed aggregates, graupel,
partially rimed) with thresholds diagnosed from the rime fraction.

Particle size distributions are complete gamma distributions,

    N(D) = N₀ D^μ exp(-λ D),

so every bulk quantity (mass, reflectivity, sedimentation flux) is a
moment — full or truncated over one mass-dimension regime — expressed
through the gamma and regularised incomplete gamma functions.

LAYOUT

The module is split into part-files under `src/p3/`:

  * `gamma.jl`         — log-gamma (Lanczos), gamma, regularised incomplete
                         gamma P/Q, gamma-PSD moments (full and truncated)
  * `massdim.jl`       — four-regime m(D), A(D), rime density, fall speed
  * `processes.jl`     — microphysical process rates
  * `sedimentation.jl` — sedimentation and radar reflectivity

Thermodynamics (saturation vapour pressures, latent heats, saturation
adjustment) is NOT re-derived here: it comes from the Era IV contract
`MoistThermodynamics`.

REFERENCES

  * Morrison, H. & Milbrandt, J. A. (2015). Parameterization of cloud
    microphysics based on the prediction of bulk ice particle properties.
    Part I. *J. Atmos. Sci.*, 72, 287-311.
  * Milbrandt, J. A. & Morrison, H. (2016). Part III: Introduction of
    multiple free categories. *J. Atmos. Sci.*, 73, 975-995.
  * Press, W. H. et al. (2007). *Numerical Recipes*, 3rd ed., §6.1-6.2.
"""
module P3Microphysics

using ..MoistThermodynamics: air_density, DEFAULT_CONSTANTS
using ..MoistThermodynamics: ThermodynamicConstants, PhasePartition, R_v,
    latent_heat_vapor, latent_heat_sublim, latent_heat_fusion,
    saturation_vapor_pressure, saturation_vapor_pressure_ice,
    saturation_specific_humidity, saturation_adjustment,
    liquid_ice_potential_temperature, heat_capacity_moist
using ..Superdroplet: MicrophysicalConstants, DEFAULT_MICRO,
    vapor_diffusivity, thermal_conductivity, fk_thermal, fd_vapor,
    terminal_velocity
using ..AerosolCloud: autoconversion_rate
using ..RadiationCore: _gauss_legendre

# --- gamma.jl -------------------------------------------------------------
export lgamma_lanczos, gamma_lanczos,
       gamma_inc_lower, gamma_inc_upper, gamma_inc_pq,
       gamma_psd_density, gamma_psd_moment, gamma_psd_partial_moment,
       gamma_psd_intercept, gamma_psd_slope

include("p3/gamma.jl")

# --- massdim.jl -----------------------------------------------------------
export P3IceParameters, P3_DEFAULT_ICE, P3MassDimension,
       p3_rime_state, p3_mass_dimension, p3_regime,
       p3_particle_mass, p3_particle_area,
       p3_mean_mass, p3_ice_mass, p3_bulk_density,
       p3_slope, p3_size_distribution,
       p3_air_viscosity, p3_fall_speed, p3_bulk_fall_speed

include("p3/massdim.jl")

# --- processes.jl ---------------------------------------------------------
export P3ProcessParameters, P3_DEFAULT_PROCESS, P3ParcelState,
       p3_total_water,
       p3_autoconversion, p3_accretion, p3_rain_self_collection,
       p3_cober_list_density, p3_rime_density,
       p3_condensation, p3_ventilation,
       p3_process_rates, p3_tendencies, p3_limit_rates, p3_step

include("p3/processes.jl")

# --- sedimentation.jl -----------------------------------------------------
export p3_rain_fall_speed, p3_rain_size_distribution, p3_rain_limit_number,
       p3_rain_bulk_fall_speed, p3_rain_rate, p3_bulk_fall_speed_z,
       P3_RAIN_D_MEAN_MIN, P3_RAIN_D_MEAN_MAX,
       P3_K2_WATER, P3_K2_ICE, p3_rain_reflectivity, p3_ice_reflectivity, p3_dbz,
       p3_sediment!, p3_rain_sedimentation_velocities, p3_sediment_rain!,
       p3_ice_sedimentation_velocities, p3_sediment_ice!

include("p3/sedimentation.jl")

end # module
