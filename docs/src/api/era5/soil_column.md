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
- **Water table (#17)** is an `AbstractSoilBottomBoundary` with its own
  state: it implements `bottom_flux_linearization` and
  `finalize_bottom!`.

```@autodocs
Modules = [Aeolus.SoilColumn]
```
