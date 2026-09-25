# Era IV — High-fidelity clouds and microphysics

The theory behind these modules is developed in
[P3 microphysics](../theory/p3.md),
[Ice nucleation](../theory/ice_nucleation.md),
[Sub-grid cloud and supersaturation PDF](../theory/subgrid_cloud.md),
[Sub-grid supersaturation in activation and droplet growth](../theory/sgs_activation_growth.md),
[Spectral broadening: bulk closure and validation](../theory/spectral_broadening.md),
and, for fog and dew,
[Fog and dew](../theory/fog.md) and [Radiation fog](../theory/radiation_fog.md);
the pages below are the generated API reference, one per module.

- [Shared contract — MoistThermodynamics (Phase 39)](era4/moist_thermodynamics.md) — `Aeolus.MoistThermodynamics`
- [P3 microphysics (Phase 40)](era4/p3_microphysics.md) — `Aeolus.P3Microphysics`, split by topic:
  [process rates and secondary ice](era4/p3_processes.md),
  [sedimentation, reflectivity and precipitation fraction](era4/p3_sedimentation.md)
- [Superdroplet microphysics (Phase 41)](era4/superdroplet.md) — `Aeolus.Superdroplet`
- [Aerosol-cloud interaction (Phase 42)](era4/aerosol_cloud.md) — `Aeolus.AerosolCloud`
- [Ice nucleation (Phase 43)](era4/ice_nucleation.md) — `Aeolus.IceNucleation`
- [Terminal-velocity spectrum and precipitation fraction (Phase 44)](era4/p3_microphysics.md#Terminal-velocity-spectrum-and-precipitation-fraction-(Phase-44))
  — no module of its own: P3 part in `Aeolus.P3Microphysics`, Kessler part in
  `Aeolus.Microphysics` (documented under [Physics modules](physics.md#Microphysics))
- [Visibility diagnostic (Phase 44)](era4/visibility.md) — `Aeolus.Visibility`
- [Fog and dew condensation (Phase 44)](era4/radiation_fog.md) — `Aeolus.RadiationFog`
- [Turbulence-microphysics interaction (Phase 45)](era4/turbulent_microphysics.md) — `Aeolus.TurbulentMicrophysics`
