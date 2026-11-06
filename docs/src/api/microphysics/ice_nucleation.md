# Ice nucleation (Phase 43)

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
mixed-phase Arctic stratus column (`run_arctic_stratus`). Theory:
[Ice nucleation](../../theory/ice_nucleation.md).

```@autodocs
Modules = [Aeolus.IceNucleation]
```
