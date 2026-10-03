# Era VI — High-fidelity turbulence closure

The pages below are the generated API reference of the Era VI modules,
one per module.

- [Stability contract — TurbulenceStability](era6/turbulence_stability.md) — `Aeolus.TurbulenceStability` (Phase 58, #170; [theory](../theory/turbulence_stability.md))
- [Transition diagnostics — Transition](era6/transition.md) — `Aeolus.Transition` (Phase 59, #180; [theory](../theory/transition.md))

## Walls (P59-a, #172)

The no-slip / moving-wall condition
[`NoSlipWall`](@ref Aeolus.BoundaryConditions.NoSlipWall) lives in
`BoundaryConditions` ([API](boundary_conditions.md), together with
`is_wall` and `homogeneous_bcs`). Its plane-mean wall stress is a method
of the cut-cell generic `wall_shear_stress`:

```@docs
Aeolus.CutCell.wall_shear_stress(::Aeolus.Simulations.BoussinesqSimulation)
```
