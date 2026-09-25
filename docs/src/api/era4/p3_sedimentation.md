# P3 sedimentation, reflectivity and precipitation fraction (Phases 40, 44)

Part of the [P3 microphysics](p3_microphysics.md) API: sedimentation
with one fall speed per moment and Courant sub-stepping, the radar
reflectivity diagnostics (rain Rayleigh ``Z``, ice equivalent ``Z_e``),
and the Phase 44 sub-grid precipitation fraction carried down the column
with maximum, random or maximum-random overlap
(`p3_precipitation_fraction`, `p3_fractional_process_rates`,
`p3_step_column`). The rain sedimentation here is also what the Kessler
scheme in `Aeolus.Microphysics` delegates its Marshall–Palmer fall speed
to (see [Physics modules](../physics.md#Microphysics)). Theory:
[P3 microphysics](../../theory/p3.md), sections "Sedimentation, size
sorting and radar reflectivity" and "Terminal-velocity spectrum and
sub-grid precipitation fraction".

```@autodocs
Modules = [Aeolus.P3Microphysics]
Pages   = ["p3/sedimentation.jl", "p3/precip_fraction.jl"]
```
