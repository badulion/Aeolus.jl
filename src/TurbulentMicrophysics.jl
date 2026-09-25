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

All thermodynamics is delegated to the Phase 39 contract
(`MoistThermodynamics`); the error function is `GasOptics`'. The
module holds no second copy of either.

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
"""
module TurbulentMicrophysics

using ..Grids: RegularGrid, interior
using ..Fields: Field
using ..MoistThermodynamics
using ..MoistThermodynamics: R_v, molmass_ratio
using ..GasOptics: erfc_series

include("turbulent_microphysics/sgs_supersaturation.jl")

end # module
