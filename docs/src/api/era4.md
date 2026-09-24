# Era IV — Clouds and microphysics

This page is the generated API reference for Era IV modules. Theory for
the sub-grid cloud scheme is in
[Sub-grid cloud and supersaturation PDF](../theory/subgrid_cloud.md), and
for its effect on droplets in
[Sub-grid supersaturation in activation and droplet growth](../theory/sgs_activation_growth.md);
the bulk closure of the resulting spectral broadening and its validation
against the superdroplet reference and DNS are in
[Spectral broadening: bulk closure and validation](../theory/spectral_broadening.md).

## Turbulence-microphysics interaction (Phase 45)

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
to, `ccn_fraction` / `ccn_spectrum`, lives in `AerosolCloud` (Phase 42).

```@autodocs
Modules = [Aeolus.TurbulentMicrophysics]
```
