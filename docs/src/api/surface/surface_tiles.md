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
snow, ice and ocean models of the surface shared core as they land.

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

## Blending height

A `BlendedMosaic` forces the tiles with the wind, temperature and
humidity *at the blending height* ``l_b`` instead of at the forcing
level, so that the tiles exchange with air already mixed over their
neighbours. The same module provides the blending-height effective
roughness, closures for ``l_b`` and internal-boundary-layer growth laws;
theory and validation are in [Blending height](../../theory/blending_height.md).

```julia
bh   = BouZeidBlendingHeight(3140.0)            # patch length scale (m)
cell = BlendedMosaic([SlabLand(T = 294.0, z0m = 0.01), SlabLand(T = 294.0, z0m = 0.1)], [0.25, 0.75], bh)
blending_height(cell)                             # ≈ 289 m
roughness_lengths(cell).z0m                       # ≈ 0.065 m (LES: 0.063 m)
fl = step_surface!(cell, SurfaceForcing(T_air = 290.0, U = 8.0, z_ref = 400.0), 60.0)
last_blending(cell).U_b                           # blended wind at l_b
ibl_height(DiffusionIBL(), 300.0; z0_up = 0.008, z0_down = 0.032)   # ≈ 26 m
```

The energy mosaic is distinct from `SurfaceOptics.TiledSurface`, the
optical (spectral-albedo) mosaic of the radiation stack.

```@autodocs
Modules = [Aeolus.SurfaceTiles]
```
