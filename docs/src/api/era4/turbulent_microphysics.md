# Turbulence-microphysics interaction (Phase 45)

Sub-grid distribution of the saturation deficit, partial cloud fraction
and condensate, and the SGS variance that drives them (issue #12); that
distribution fed into aerosol activation and stochastic condensational
growth (issue #13); and the bulk moment closure of the broadening with its
superdroplet/DNS validation case (issue #14: `SpectralMoments`,
`bulk_condensation_step!`, `parcel_ascent!`, `run_broadening_case`). All
of these live in one module, so the block below covers all three issues.
The bulk state also extends `Superdroplet.mean_radius`,
`Superdroplet.relative_dispersion` and
`MoistThermodynamics.effective_radius`. The CCN spectrum they delegate
to, `ccn_fraction` / `ccn_spectrum`, lives in
`AerosolCloud` (Phase 42).

Theory: [Sub-grid cloud and supersaturation PDF](../../theory/subgrid_cloud.md),
[Sub-grid supersaturation in activation and droplet growth](../../theory/sgs_activation_growth.md)
and [Spectral broadening: bulk closure and validation](../../theory/spectral_broadening.md).

```@autodocs
Modules = [Aeolus.TurbulentMicrophysics]
```
