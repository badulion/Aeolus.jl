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

# --- gamma.jl -------------------------------------------------------------
export lgamma_lanczos, gamma_lanczos,
       gamma_inc_lower, gamma_inc_upper, gamma_inc

include("p3/gamma.jl")

end # module
