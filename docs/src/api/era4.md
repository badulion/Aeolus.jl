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
are written in.

```@autodocs
Modules = [Aeolus.IceNucleation]
```
