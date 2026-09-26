# Soil column — SoilColumn

Phase 46 (#15): a CLM-style soil column with Richards-equation water
transport. Theory, numerics and validation are on the
[soil water](../../theory/soil_water.md) page.

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

## Extending the column

- **Freeze–thaw (#16)** fills `col.θ_ice` and `col.T_soil`; a new
  `AbstractIceImpedance` subtype implements `ice_impedance_factor`.
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
