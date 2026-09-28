# Snowpack

`Snowpack` (Phase 51, #34) is a multi-layer seasonal snowpack in the
CLM / Crocus tradition: settling under overburden and destructive
metamorphism, heat conduction with a linearised surface energy balance,
melt, sublimation and deposition, fresh-snow accumulation, and layer
splitting and merging that conserve mass and enthalpy exactly.
`LayeredSnowpack` is an `AbstractSurface`: `step_surface!(pack,
forcing, dt)` returns the `SurfaceFluxes` it applied, computed through
`SurfaceExchange.turbulent_fluxes`. The physics is developed in
[Layered snowpack](../../theory/snowpack.md).

Material laws are chosen by dispatch (`Anderson1976Compaction`,
`Vionnet2012Compaction`; `Yen1981Conductivity`, `Sturm1997Conductivity`,
`Jordan1991Conductivity`, `Calonne2011Conductivity`;
`Anderson1976FreshSnow`, `Vionnet2012FreshSnow`). Grain/albedo ageing
(#35) and liquid water (#36) plug in through `AbstractSnowMetamorphism`,
`AbstractSnowAlbedo` and `AbstractSnowLiquidWater`.

```julia
using Aeolus
pack = LayeredSnowpack(depth = 0.8, density = 200.0, T = 265.0,
                       compaction = Vionnet2012Compaction(),
                       conductivity = Sturm1997Conductivity(),
                       bottom = SnowGroundTemperature(273.15))
f = SurfaceForcing(sw_down = 200.0, lw_down = 250.0, T_air = 266.0,
                   q_air = 0.002, U = 4.0, p = 90000.0, z_ref = 10.0,
                   rain = 0.0, snow = 1e-4)
fl = step_surface!(pack, f, 3600.0)      # fl.Rn == fl.H + fl.LE + fl.G
snow_depth(pack), snow_bulk_density(pack), pack.budget.snowfall
```

```@autodocs
Modules = [Aeolus.Snowpack]
```
