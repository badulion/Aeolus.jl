# Era IV — Clouds and microphysics

This page is the generated API reference for Era IV modules. Theory for
the sub-grid cloud scheme is in
[Sub-grid cloud and supersaturation PDF](../theory/subgrid_cloud.md), and
for its effect on droplets in
[Sub-grid supersaturation in activation and droplet growth](../theory/sgs_activation_growth.md).

## Turbulence-microphysics interaction (Phase 45)

Sub-grid distribution of the saturation deficit, partial cloud fraction
and condensate, and the SGS variance that drives them (issue #12); that
distribution fed into aerosol activation and stochastic condensational
growth (issue #13). All of these live in one module, so the block below
covers both issues. The CCN spectrum they delegate to,
`ccn_fraction` / `ccn_spectrum`, lives in `AerosolCloud` (Phase 42).

```@autodocs
Modules = [Aeolus.TurbulentMicrophysics]
```
