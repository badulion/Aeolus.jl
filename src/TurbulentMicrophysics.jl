"""
    TurbulentMicrophysics

Turbulence-microphysics interaction — Era IV, Phase 45.

A grid cell of an LES is not uniformly saturated: unresolved turbulent
fluctuations of total water `q_t` and liquid-ice potential temperature
`θ_li` spread the cell over a *distribution* of saturation states. A
scheme that applies the all-or-nothing saturation adjustment to the
cell mean either clouds the whole cell or none of it; the real cell is
partly cloudy, and the part that is cloudy contains more condensate than
the mean-state adjustment would put there.

This module carries that distribution. Its pieces live in part-files
under `src/turbulent_microphysics/`:

  * `sgs_supersaturation.jl` (issue #12) — the probability density of
    the Mellor (1977) saturation deficit `s` within a cell, its variance
    derived from the Smagorinsky-Lilly SGS closure, and the partial cloud
    fraction and mean condensate that follow from it.
  * `sgs_activation.jl` (issue #13) — that distribution fed into Phase 42
    aerosol activation: the activated number is the PDF-mean of the CCN
    spectrum, which differs from the mean-state value by the curvature
    of the spectrum (Jensen).
  * `sgs_growth.jl` (issue #13) — the same fluctuations in Phase 41
    condensational growth: superdroplets sample a stochastic (Ornstein-
    Uhlenbeck) supersaturation, so `dr²/dt ∝ S` spreads the radius
    distribution while the cell's total water is conserved exactly.
  * `spectral_broadening.jl` (issue #14) — the bulk counterpart: a
    moment closure (`⟨r²⟩`, `Var r²`, `Cov(r², S')`) that carries the
    same broadening without droplets, and the adiabatic-parcel case that
    validates it against the superdroplet reference and DNS scaling.

All thermodynamics is delegated to the Phase 39 contract
(`MoistThermodynamics`); the error function is `GasOptics`'; Köhler
theory and the CCN spectrum are `AerosolCloud`'s (Phase 42); droplet
growth and the cell water/heat budget are `Superdroplet`'s (Phase 41).
The module holds no second copy of any of them.

REFERENCES

  * Sommeria, G. & Deardorff, J. W. (1977). Subgrid-scale condensation
    in models of nonprecipitating clouds. *J. Atmos. Sci.*, 34, 344-355.
  * Mellor, G. L. (1977). The Gaussian cloud model relations.
    *J. Atmos. Sci.*, 34, 356-358.
  * Bechtold, P., Cuijpers, J. W. M., Mascart, P. & Trouilhet, P. (1995).
    Modeling of trade wind cumuli with a low-order turbulence model:
    toward a unified description of Cu and Sc clouds in meteorological
    models. *J. Atmos. Sci.*, 52, 455-463.
  * Chaboureau, J.-P. & Bechtold, P. (2002). A simple cloud
    parameterization derived from cloud resolving model data: diagnostic
    and prognostic applications. *J. Atmos. Sci.*, 59, 2362-2372.
  * Larson, V. E., Golaz, J.-C. & Cotton, W. R. (2002). Small-scale and
    mesoscale variability in cloudy boundary layers: joint probability
    density functions. *J. Atmos. Sci.*, 59, 3519-3539.
  * Deardorff, J. W. (1980). Stratocumulus-capped mixed layers derived
    from a three-dimensional model. *Bound.-Layer Meteor.*, 18, 495-527.
  * Ghan, S. J., Leung, L. R., Easter, R. C. & Abdul-Razzak, H. (1997).
    Prediction of cloud droplet number in a general circulation model.
    *J. Geophys. Res.*, 102, 21777-21794.
  * Cooper, W. A. (1989). Effects of variable droplet growth histories on
    droplet size distributions. Part I: Theory. *J. Atmos. Sci.*, 46,
    1301-1311.
  * Paoli, R. & Shariff, K. (2009). Turbulent condensation of droplets:
    direct simulation and a stochastic model. *J. Atmos. Sci.*, 66,
    723-740.
  * Sardina, G., Picano, F., Brandt, L. & Caballero, R. (2015).
    Continuous growth of droplet size variance due to condensation in
    turbulent clouds. *Phys. Rev. Lett.*, 115, 184501.
  * Lanotte, A. S., Seminara, A. & Toschi, F. (2009). Cloud droplet
    growth by condensation in homogeneous isotropic turbulence.
    *J. Atmos. Sci.*, 66, 1685-1697.
  * Grabowski, W. W. & Wang, L.-P. (2013). Growth of cloud droplets in a
    turbulent environment. *Annu. Rev. Fluid Mech.*, 45, 293-324.
"""
module TurbulentMicrophysics

using ..Grids: RegularGrid, interior
using ..Fields: Field
using ..MoistThermodynamics
using ..MoistThermodynamics: R_v, molmass_ratio
using ..GasOptics: erfc_series
# Phase 42: activation and the CCN spectrum (issue #13 delegates to them)
using ..AerosolCloud: ARGMode, ActivationCoefficients, ActivationResult,
                      activation_coefficients, arg_activation, ccn_fraction,
                      mode_critical_supersaturation, KAPPA_MIN, arg_modes
using ..AerosolOptics: AerosolMode
# Phase 41: droplet growth and the cell water/heat budget
using ..Superdroplet: SuperdropletSet, nsuperdroplets, condensation_growth!,
                      EulerianCell, cell_pressure, cell_supersaturation,
                      cell_air_mass, total_water_mass,
                      apply_condensate_tendency!, MicrophysicalConstants,
                      DEFAULT_MICRO, _phase_substeps,
                      fk_thermal, fd_vapor, kelvin_a, vapor_diffusivity,
                      init_monodisperse_superdroplets, cell_from_state
# issue #14 extends the Phase 41 spectral diagnostics to the bulk state
import ..Superdroplet: mean_radius, relative_dispersion
import ..MoistThermodynamics: effective_radius
using Random: AbstractRNG, MersenneTwister, randn

include("turbulent_microphysics/sgs_supersaturation.jl")
include("turbulent_microphysics/sgs_activation.jl")
include("turbulent_microphysics/sgs_growth.jl")
include("turbulent_microphysics/spectral_broadening.jl")

end # module
