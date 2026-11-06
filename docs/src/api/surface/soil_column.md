# Soil column — SoilColumn

Phase 46: a CLM-style soil column with Richards-equation water
transport (#15) and heat conduction with freeze–thaw (#16). Theory,
numerics and validation are on the [soil water](../../theory/soil_water.md)
and [soil heat and freeze–thaw](../../theory/soil_heat.md) pages.

- **Layers**: `clm5_soil_grid` (CLM5, 20 layers to 8.6 m),
  `clm_exponential_soil_grid`, `uniform_soil_grid`.
- **Hydraulics**: `ClappHornberger`, `VanGenuchtenMualem` (with the
  Vogel/Ippisch air-entry option), `GardnerExponential`; texture tables
  `CLAPP_HORNBERGER_1978`, `CARSEL_PARRISH_1988`; `cosby_clapp_hornberger`.
- **Temperature and ice**: `water_dynamic_viscosity`,
  `viscosity_conductivity_factor`, the `AbstractIceImpedance` hook.
- **Solver**: `SoilWaterColumn`, `step_soil_water!` (modified Picard,
  exact water conservation), boundary conditions `AtmosphericTop`,
  `HeadTop`, `FreeDrainage`, `ZeroFluxBottom`, `HeadBottom`.

- **Heat (#16)**: `SoilThermalProperties` (`clm_soil_thermal_properties`,
  `johansen_soil_thermal_properties`), `soil_thermal_conductivity`,
  `soil_heat_capacity`; freezing curve `freezing_matric_potential`,
  `supercooled_liquid_content`; `PowerLawIceImpedance`.
- **Thermal column (#16)**: `SoilThermalColumn` wraps a `SoilWaterColumn`
  and is an `AbstractSurface`; `step_soil_heat!` (enthalpy method, exact
  energy conservation) under `GroundHeatFluxTop`, `SkinTemperatureTop` or
  `LinearizedSkinBalance`, with `ZeroHeatFluxBottom`,
  `GeothermalHeatFlux` or `BottomTemperature`; `step_surface!` couples
  skin energy balance, heat and water.

Boundary types and accessors are not re-exported from `Aeolus`; use
`using Aeolus.SoilColumn` or qualify them.

```julia
using Aeolus, Aeolus.SoilColumn

col = SoilWaterColumn(clm5_soil_grid(), carsel_parrish_texture(:loam); ψ = -2.0)
for hour in 1:24
    st = step_soil_water!(col, 3600.0; water_input = hour < 4 ? 1e-5 : 0.0,
                          evaporation = 5e-8)
end
col.budget          # infiltration, runoff, evaporation, drainage, extraction
```

With heat and freeze–thaw, driven through the surface shared core (`SurfaceExchange`):

```julia
using Aeolus, Aeolus.SoilColumn
using Aeolus.SurfaceExchange: SurfaceForcing

water = SoilWaterColumn(clm5_soil_grid(), carsel_parrish_texture(:loam);
                        ψ = -1.0, ice_impedance = PowerLawIceImpedance())
soil = SoilThermalColumn(water, clm_soil_thermal_properties(40, 20); T_soil = 276.15,
                         bottom = GeothermalHeatFlux(0.065))
night = SurfaceForcing(sw_down = 0.0, lw_down = 220.0, T_air = 263.15, q_air = 0.001, U = 3.0)
fluxes = step_surface!(soil, night, 1800.0)   # Rn = H + LE + G, G conducted into the soil
soil.T_skin, water.T_soil, water.θ_ice, soil.budget
```

## Extending the column

- **Freeze–thaw (#16)**: `SoilThermalColumn` fills `col.θ_ice` and
  `col.T_soil` of the water column it wraps; a new `AbstractIceImpedance`
  subtype implements `ice_impedance_factor`. A canopy (#21) drives the
  ground below it with `step_soil_heat!(soil, dt, LinearizedSkinBalance(G0, dG_dT, T0))`.
- **Water table (#17)** is implemented as `UnconfinedAquifer`, an
  `AbstractSoilBottomBoundary` with its own state (see below). Another
  stateful boundary does the same: it implements
  `bottom_flux_linearization` (the five-argument form receives the
  sub-step `dt`) and `finalize_bottom!`, and optionally
  `bottom_water_storage`.

## Water table (#17)

`UnconfinedAquifer` is the SIMTOP groundwater model (Niu et al. 2005,
2007; CLM4 §7.5). It provides a prognostic aquifer storage and water
table, recharge as the exact exponential-K two-point flux
(`gardner_two_point_flux`), and baseflow ``q_{max} e^{-f z_∇}`` with the
frozen-soil impermeable fraction (`aquifer_impermeable_fraction`). The
diagnostics are `water_table_depth`, `aquifer_baseflow_rate` and
`total_water_storage`, and `set_water_table!` starts the column in
hydrostatic equilibrium. Theory and validation are on the
[water table](../../theory/water_table.md) page.

```julia
col = SoilWaterColumn(clm5_soil_grid(), carsel_parrish_texture(:silt_loam);
                      bottom = UnconfinedAquifer())
set_water_table!(col, 5.0)
step_soil_water!(col, 86400.0; water_input = 5e-7)
water_table_depth(col), col.bottom.baseflow, total_water_storage(col)
```

```@autodocs
Modules = [Aeolus.SoilColumn]
```
