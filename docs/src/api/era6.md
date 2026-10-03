# Era VI — High-fidelity turbulence closure

The theory is developed in
[Transition and wall-bounded flow](../theory/transition.md).

## Walls (P59-a, #172)

The no-slip / moving-wall condition
[`NoSlipWall`](@ref Aeolus.BoundaryConditions.NoSlipWall) lives in
`BoundaryConditions` ([API](boundary_conditions.md), together with
`is_wall` and `homogeneous_bcs`). Its plane-mean wall stress is a method
of the cut-cell generic `wall_shear_stress`:

```@docs
Aeolus.CutCell.wall_shear_stress(::Aeolus.Simulations.BoussinesqSimulation)
```
