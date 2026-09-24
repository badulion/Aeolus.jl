# Era IV — High-fidelity clouds and microphysics

The theory behind these modules is developed in
[P3 microphysics](../theory/p3.md) and
[Ice nucleation](../theory/ice_nucleation.md); this page is the
generated API reference.

## P3 microphysics (Phase 40)

`P3Microphysics` implements the Morrison–Milbrandt predicted-particle-
properties scheme. Its size-distribution integrals rest on the in-package
gamma and regularised incomplete gamma functions documented here; its
sedimentation (one fall speed per moment, Courant sub-stepping) and
radar reflectivity (rain Rayleigh ``Z``, ice equivalent ``Z_e``) are
the observables it is validated with.

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
and vapour sinks and latent heat (`ice_nucleation!`).

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
