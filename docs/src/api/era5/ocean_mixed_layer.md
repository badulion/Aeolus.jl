# Lake and sea surface — OceanMixedLayer

`OceanMixedLayer` (Phase 52, #38) makes the temperature of a water
surface prognostic: a layered column with a Kraus–Turner mixed layer,
shortwave penetration (Jerlov / Soloviev), a Fairall cool skin and a
diurnal warm layer (Fairall 1996b / COARE, or Zeng & Beljaars 2005). The
physics and the measured validation are in
[Water surfaces: mixed layer, cool skin and warm layer](../../theory/ocean_mixed_layer.md).

`WaterColumn` is an `AbstractSurface` of the Era V contract: its fluxes
come from `SurfaceExchange.turbulent_fluxes`, or from any
`flux(col, forcing) -> SurfaceFluxes` passed to `step_surface!` — the
hook for air–sea algorithms (`AirSeaFlux`, #39). Its emissivity default
is `SurfaceOptics.material_emissivity(:water)`; gravity, the von Kármán
constant, the heat capacity and density of water, and the water
conductivity used by the cool skin come from `SurfaceConstants`.

```julia
using Aeolus

# A 20 m lake of 1 m layers (resolves its own warm layer)…
lake = WaterColumn(dz = fill(1.0, 20), T = 290.0, S = 0.0,
                   solar = jerlov_water(:III))
# …or a 10 m coastal slab with a sub-grid warm layer.
sea = WaterColumn(dz = [10.0], T = 298.0, S = 35.0,
                  warm_layer = ZengBeljaarsWarmLayer())

f = SurfaceForcing(sw_down = 800.0, lw_down = 400.0, T_air = 297.0,
                   q_air = 0.016, U = 4.0)
fl = step_surface!(sea, f, 300.0)      # SurfaceFluxes; the column absorbs fl.G
surface_temperature(sea)               # top layer + warm layer - cool skin
```

```@autodocs
Modules = [Aeolus.OceanMixedLayer]
```
