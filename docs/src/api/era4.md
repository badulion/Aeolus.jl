# Era IV — High-fidelity clouds and microphysics

The theory behind these modules is developed in the Era IV theory
pages (for fog and dew: [Fog and dew](../theory/fog.md) and
[Radiation fog](../theory/radiation_fog.md)); this page is
the generated API reference.

## Fog and dew condensation (Phase 44)

`RadiationFog` couples fog (condensation in the air) and dew
(deposition onto a cold surface) to the vapour field with the exact
moist enthalpy of the `MoistThermodynamics` contract. Both steps are
saturation-limited and unconditionally stable in the time step.

The same module carries the radiation-fog feedback (#9): the droplet
field (LWC, `N_d`, `r_eff`) is passed to the Era III radiation —
`GasOptics`, `TwoStream.MieCloudOptics`, `RadiationCore.combine_optics!`
and the two-stream solver — through [`fog_atmospheric_column`](@ref) and
[`fog_radiative_transfer!`](@ref), and a 1-D column ([`FogColumn`](@ref),
[`parisfog_2011_column`](@ref)) runs fog through formation, fog-top
cooling and dissipation. See [Radiation fog](../theory/radiation_fog.md).

```@autodocs
Modules = [Aeolus.RadiationFog]
```
