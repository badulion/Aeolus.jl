# Era IV — High-fidelity clouds and microphysics

The theory behind these modules is developed in the Era IV theory
pages (for fog and dew: [Fog and dew](../theory/fog.md)); this page is
the generated API reference.

## Fog and dew condensation (Phase 44)

`RadiationFog` couples fog (condensation in the air) and dew
(deposition onto a cold surface) to the vapour field with the exact
moist enthalpy of the `MoistThermodynamics` contract. Both steps are
saturation-limited and unconditionally stable in the time step.

```@autodocs
Modules = [Aeolus.RadiationFog]
```
