# Lake and sea ice — IceThermodynamics

`IceThermodynamics` is the model of floating ice (Phase 51, issue
#37): the energy-conserving multi-layer thermodynamics of Bitz &
Lipscomb (1999) with brine pockets, a snow layer, surface and basal
melt, basal accretion, sublimation, snow-ice formation, and an ocean or
lake heat flux. The same code does sea ice (a salinity profile, the
water at its liquidus ``T_0 - μS``) and lake ice (salinity zero, water
at 0 °C). Theory, validation and the one discretisation subtlety are on
the [theory page](../../theory/ice_thermodynamics.md).

`IceColumn <: AbstractSurface`, so `step_surface!(col, forcing, dt)`
couples it to the atmosphere through the `SurfaceExchange` contract
(Monin–Obukhov fluxes, saturation over ice, latent heat of
sublimation). For benchmarks and offline use, [`step_ice!`](@ref) also
takes a prescribed surface temperature or prescribed fluxes.

```julia
using Aeolus

# Lake ice under a held -10 °C surface: Stefan / Neumann growth.
lake = lake_ice_column(h_ice = 0.1, T_surface = 263.15)
for _ in 1:30*24
    step_ice!(lake, PrescribedSurfaceTemperature(263.15), 3600.0)
end
ice_thickness(lake)   # 0.616 m; Stefan gives 0.621 m, the ice's heat capacity slows it 1 %

# Multiyear sea ice with snow, coupled to the atmosphere.
sea = sea_ice_column(h_ice = 2.5, h_snow = 0.3)
f = SurfaceForcing(sw_down = 0.0, lw_down = 170.0, T_air = 243.15,
                   q_air = 3e-4, U = 5.0)
fl = step_surface!(sea, f, 1800.0)           # SurfaceFluxes; sea.budget has the details
```

Every step reports an [`IceStepBudget`](@ref); its `energy_residual`
(W m⁻²) is the change of [`ice_column_enthalpy`](@ref) minus every flux
applied, and is zero to round-off.

```@autodocs
Modules = [Aeolus.IceThermodynamics]
```
