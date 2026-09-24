# Era IV — High-fidelity clouds and microphysics

The theory behind these modules is developed in
[P3 microphysics](../theory/p3.md); this page is the generated API
reference.

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
