# Sub-grid surface tiles — SurfaceTiles

`SurfaceTiles` represents a heterogeneous grid cell as a mosaic of
[`AbstractSurface`](@ref Aeolus.SurfaceExchange.AbstractSurface) tiles
with area fractions. Each tile keeps its own state and solves its own
energy balance through the `SurfaceExchange` contract; the cell returns
the fraction-weighted **fluxes** (flux aggregation), not the flux of a
surface with averaged parameters. Theory, the exact aggregates and the
measured size of the aggregation error are in
[Sub-grid surface tiles](../../theory/surface_tiles.md).

A `SurfaceMosaic` is itself an `AbstractSurface`, so mosaics nest and a
mosaic can be stepped wherever a single surface can. Any implementer of
the contract can be a tile — `SlabLand` today, the soil, canopy, urban,
snow, ice and ocean models of Era V as they land.

```julia
using Aeolus
cell = SurfaceMosaic([SlabLand(T = 300.0, albedo = 0.35, z0m = 0.01),   # dry field
                      SlabLand(T = 292.0, albedo = 0.12, z0m = 1.0)],   # wood
                     [0.7, 0.3])
f  = SurfaceForcing(sw_down = 600.0, T_air = 295.0, U = 3.0)
fl = step_surface!(cell, f, 60.0)        # Σ fᵢ Fᵢ, each tile stepped
last_tile_fluxes(cell)                   # the per-tile fluxes behind it
turbulent_fluxes(parameter_aggregate(cell), f)   # the parameter-averaged alternative
```

The energy mosaic is distinct from `SurfaceOptics.TiledSurface`, the
optical (spectral-albedo) mosaic of Era III.

```@autodocs
Modules = [Aeolus.SurfaceTiles]
```
