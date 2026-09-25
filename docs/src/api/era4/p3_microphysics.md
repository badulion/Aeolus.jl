# P3 microphysics (Phase 40)

`P3Microphysics` implements the Morrison–Milbrandt predicted-particle-
properties scheme. Its size-distribution integrals rest on the in-package
gamma and regularised incomplete gamma functions documented here; its
sedimentation (one fall speed per moment, Courant sub-stepping) and
radar reflectivity (rain Rayleigh ``Z``, ice equivalent ``Z_e``) are
the observables it is validated with. Secondary ice from Hallett–Mossop
rime splintering is driven by its riming rates. Theory:
[P3 microphysics](../../theory/p3.md).

The module's API is large, so it is split by topic over three pages:

- this page — the module, gamma functions and gamma size-distribution
  moments, and the mass–dimension relation across the rime regimes;
- [P3 process rates and secondary ice](p3_processes.md) — the process
  rates, the time step (`p3_tendencies`, `p3_step`) and Hallett–Mossop
  rime splintering;
- [P3 sedimentation, reflectivity and precipitation fraction](p3_sedimentation.md)
  — per-moment sedimentation, radar reflectivity, and the Phase 44
  sub-grid precipitation fraction.

## Terminal-velocity spectrum and precipitation fraction (Phase 44)

The Kessler scheme's rain now falls at the mass-weighted speed of its
Marshall–Palmer spectrum (`kessler_fall_speed`, in `Aeolus.Microphysics`,
see [Physics modules](../physics.md#Microphysics)), and P3 gains a sub-grid
precipitation fraction carried down the column with maximum, random or
maximum-random overlap (`p3_precipitation_fraction`,
`p3_fractional_process_rates`, `p3_step_column`), documented on
[P3 sedimentation, reflectivity and precipitation fraction](p3_sedimentation.md).
Theory: [P3 microphysics](../../theory/p3.md), last section.

## Gamma functions and the mass–dimension relation

```@autodocs
Modules = [Aeolus.P3Microphysics]
Pages   = ["P3Microphysics.jl", "p3/gamma.jl", "p3/massdim.jl"]
```
