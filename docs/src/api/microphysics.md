# Era IV — High-fidelity clouds and microphysics

The era is summarised, with its measured validation, in
[Clouds and microphysics (Era IV): overview](../theory/clouds_overview.md).
The theory behind these modules is developed in
[P3 microphysics](../theory/p3.md),
[Ice nucleation](../theory/ice_nucleation.md),
[Sub-grid cloud and supersaturation PDF](../theory/subgrid_cloud.md),
[Sub-grid supersaturation in activation and droplet growth](../theory/sgs_activation_growth.md),
[Spectral broadening: bulk closure and validation](../theory/spectral_broadening.md),
[Visibility](../theory/visibility.md),
[Fog and dew](../theory/fog.md) and
[Radiation fog](../theory/radiation_fog.md);
the pages below are the generated API reference, one per module.

- [Shared contract — MoistThermodynamics (Phase 39)](microphysics/moist_thermodynamics.md) — `Aeolus.MoistThermodynamics`
- [P3 microphysics (Phase 40)](microphysics/p3_microphysics.md) — `Aeolus.P3Microphysics`, split by topic:
  [process rates and secondary ice](microphysics/p3_processes.md),
  [sedimentation, reflectivity and precipitation fraction](microphysics/p3_sedimentation.md)
- [Superdroplet microphysics (Phase 41)](microphysics/superdroplet.md) — `Aeolus.Superdroplet`
- [Aerosol-cloud interaction (Phase 42)](microphysics/aerosol_cloud.md) — `Aeolus.AerosolCloud`
- [Ice nucleation (Phase 43)](microphysics/ice_nucleation.md) — `Aeolus.IceNucleation`
- [Terminal-velocity spectrum and precipitation fraction (Phase 44)](microphysics/p3_microphysics.md#Terminal-velocity-spectrum-and-precipitation-fraction-(Phase-44))
  — no module of its own: P3 part in `Aeolus.P3Microphysics`, Kessler part in
  `Aeolus.Microphysics` (documented under [Physics modules](physics.md#Microphysics))
- [Visibility diagnostic (Phase 44)](microphysics/visibility.md) — `Aeolus.Visibility`
- [Fog and dew condensation (Phase 44)](microphysics/radiation_fog.md) — `Aeolus.RadiationFog`
- [Turbulence-microphysics interaction (Phase 45)](microphysics/turbulent_microphysics.md) — `Aeolus.TurbulentMicrophysics`
