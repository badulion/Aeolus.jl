# Boundary conditions

Per-face boundary conditions, grouped into a
[`FieldBoundaryConditions`](@ref Aeolus.BoundaryConditions.FieldBoundaryConditions)
container. `fill_halos!` applies `Periodic` and `NoFlux` on every face
and the two impermeable z-walls, free-slip `ImpermeableWall` and no-slip /
moving `NoSlipWall` (P59-a, #172; theory in
[Transition and wall-bounded flow](../theory/transition.md#Walls)).
`ValueBC` and `FluxBC` are declared for scalar surface conditions.

## BC types

```@docs
Aeolus.BoundaryConditions.AbstractBC
Aeolus.BoundaryConditions.Periodic
Aeolus.BoundaryConditions.NoFlux
Aeolus.BoundaryConditions.ValueBC
Aeolus.BoundaryConditions.FluxBC
```

## Container

```@docs
Aeolus.BoundaryConditions.FieldBoundaryConditions
```

## Halo fill

```@docs
Aeolus.BoundaryConditions.fill_halos!
```

## Wall conditions

```@docs
Aeolus.BoundaryConditions.ImpermeableWall
Aeolus.BoundaryConditions.NoSlipWall
Aeolus.BoundaryConditions.is_wall
Aeolus.BoundaryConditions.homogeneous_bcs
```
