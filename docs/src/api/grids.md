# Grids

The `Aeolus.Grids` module defines the mesh types and coordinate access.
See [Staggered discretization](../theory/staggered_grid.md) for the
physical meaning of the [`Location`](@ref Aeolus.Grids.Location) types
and [Architecture](../design/architecture.md) for how `AbstractMesh`
fits into the extensibility story.

## Mesh types

```@docs
Aeolus.Grids.AbstractMesh
Aeolus.Grids.RegularGrid
```

## Location tags

```@docs
Aeolus.Grids.Location
Aeolus.Grids.CellCenter
Aeolus.Grids.FaceX
Aeolus.Grids.FaceY
Aeolus.Grids.FaceZ
```

## Coordinate access

```@docs
Aeolus.Grids.xnodes
Aeolus.Grids.interior
Aeolus.Grids.totalsize
```
