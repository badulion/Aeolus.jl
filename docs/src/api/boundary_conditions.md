# Boundary conditions

Per-face boundary conditions, grouped into a
[`FieldBoundaryConditions`](@ref Aeolus.BoundaryConditions.FieldBoundaryConditions)
container. Currently only `Periodic` and `NoFlux` are fully applied by
`fill_halos!`; `ValueBC` and `FluxBC` are declared but await
Phase 1b's wall implementation.

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
```
