# Era IV — High-fidelity clouds and microphysics

The theory behind these modules is developed in
[P3 microphysics](../theory/p3.md),
[Ice nucleation](../theory/ice_nucleation.md),
[Sub-grid cloud and supersaturation PDF](../theory/subgrid_cloud.md),
[Sub-grid supersaturation in activation and droplet growth](../theory/sgs_activation_growth.md) and
[Spectral broadening: bulk closure and validation](../theory/spectral_broadening.md);
this page is the generated API reference.

## P3 microphysics (Phase 40)

`P3Microphysics` implements the Morrison–Milbrandt predicted-particle-
properties scheme. Its size-distribution integrals rest on the in-package
gamma and regularised incomplete gamma functions documented here; its
sedimentation (one fall speed per moment, Courant sub-stepping) and
radar reflectivity (rain Rayleigh ``Z``, ice equivalent ``Z_e``) are
the observables it is validated with. Secondary ice from Hallett–Mossop
rime splintering is driven by its riming rates.

```@autodocs
Modules = [Aeolus.P3Microphysics]
```

## Ice nucleation (Phase 43)

`IceNucleation` carries ice-nucleating particles (mineral dust and
primary biological aerosol) as prognostic modal tracers with an
available/consumed budget, and provides the log-normal number and
surface-area diagnostics that heterogeneous freezing parameterisations
are written in. On top of that population it implements immersion
freezing (Niemand et al. 2012, Ullrich et al. 2017, DeMott et al. 2015,
Tobo et al. 2013) and deposition nucleation (Ullrich et al. 2017), and
returns ice number and mass source terms with the matching INP, liquid
and vapour sinks and latent heat (`ice_nucleation!`). Those sources are
coupled into the P3 ice category (`p3_ice_nucleation`,
`p3_inp_release!`), and the phase is validated on a 1-D ISDAC-style
mixed-phase Arctic stratus column (`run_arctic_stratus`).

```@autodocs
Modules = [Aeolus.IceNucleation]
```

## Terminal-velocity spectrum and precipitation fraction (Phase 44)

The Kessler scheme's rain now falls at the mass-weighted speed of its
Marshall–Palmer spectrum (`kessler_fall_speed`, in `Aeolus.Microphysics`,
see [Physics](physics.md)), and P3 gains a sub-grid precipitation
fraction carried down the column with maximum, random or maximum-random
overlap (`p3_precipitation_fraction`, `p3_fractional_process_rates`,
`p3_step_column`), documented with the P3 module above. Theory:
[P3 microphysics](../theory/p3.md), last section.

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

## Visibility diagnostic (Phase 44)

Koschmieder visibility from droplet, haze and molecular extinction at
550 nm, plus the Kunkel (1984) and Gultepe et al. (2006) fog
parameterisations. Theory: [Visibility](../theory/visibility.md).

```@autodocs
Modules = [Aeolus.Visibility]
```
