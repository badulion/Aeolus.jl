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
