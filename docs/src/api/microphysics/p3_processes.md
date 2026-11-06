# P3 process rates and secondary ice (Phase 40)

Part of the [P3 microphysics](p3_microphysics.md) API: the
warm-rain, ice and mixed-phase process rates, the parcel tendencies and
limited time step built from them (`p3_tendencies`, `p3_step`), and
secondary ice production by Hallett–Mossop rime splintering, which is
driven by the riming rates. Theory: [P3 microphysics](../../theory/p3.md),
sections "Process rates" and "Secondary ice".

```@autodocs
Modules = [Aeolus.P3Microphysics]
Pages   = ["p3/processes.jl", "p3/secondary.jl"]
```
