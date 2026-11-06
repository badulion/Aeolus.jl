# LandSurface — the vegetated land column

`LandSurface` is the vegetated land column (Phase 47, #21): a
two-big-leaf (sunlit/shaded) canopy over a `SoilColumn.SoilThermalColumn`,
coupled to the atmosphere through `SurfaceExchange`'s Monin–Obukhov solver
and implementing `AbstractSurface`, so the tile framework can aggregate it
like any other surface. The theory, the validation against five FLUXNET
towers and what the model gets wrong are on the
[land surface](../../theory/land_surface.md) page.

It owns what only a vegetated column needs:

- the land-column columns of the CLM4.5 PFT tables — soil-water stress
  potentials ψ_o, ψ_c (Table 8.1), Zeng (2001) root parameters (Table
  8.3), leaf optics (Table 3.1), aerodynamic ratios (Table 5.1) —
  gathered with the photosynthetic parameters of `Photosynthesis` by
  `pft_vegetation`;
- root fractions, the CLM wilting factor and the root-zone stress factor
  β_t (`zeng_root_fractions`, `CLMWaterStress`, `soil_water_stress`), and
  the root-uptake sink that takes the transpiration out of the soil
  (`root_water_uptake`);
- the big-leaf canopy radiation (`bigleaf_shortwave`, `bigleaf_longwave`)
  and canopy aerodynamics (`canopy_roughness`,
  `undercanopy_transfer_coefficient`);
- `VegetatedLandColumn`, its `step_surface!`, and the diagnostics and
  budgets of a step.

It delegates the leaf biochemistry, the stomatal closures, the leaf
boundary layer and the water-stress hook to `Photosynthesis`; soil water,
heat, freeze–thaw and the water table to `SoilColumn`; the surface layer
and constants to `SurfaceExchange`; saturation and latent heats to
`MoistThermodynamics`; and, in the FLUXNET tests, the solar position and
the diffuse fraction to `SolarGeometry`.

```julia
using Aeolus
using Aeolus.SoilColumn: SoilThermalColumn
grid = clm5_soil_grid(13)                                     # 2.7 m
water = SoilWaterColumn(grid, cosby_clapp_hornberger(4.0, 40.0); ψ = -3.37,
                        ice_impedance = PowerLawIceImpedance())
soil = SoilThermalColumn(water, clm_soil_thermal_properties(4.0, 40.0); T_soil = 288.0)
col = VegetatedLandColumn(soil, pft_vegetation(:bdt_temperate); LAI = 5.0, z_top = 30.0)

f = SurfaceForcing(sw_down = 700.0, lw_down = 340.0, T_air = 295.0, q_air = 0.010,
                   U = 3.0, z_ref = 45.0)
fl = step_surface!(col, f, 3600.0; cosz = 0.8, diffuse_fraction = 0.2)
fl.Rn - fl.H - fl.LE - fl.G            # ≈ 1e-11 W m⁻²: the column closes its budget
col.last.β_t, col.last.LE_v, col.last.GPP, col.last.T_v
```

`cosz`, `diffuse_fraction` and `co2` are state of the column, set by the
keyword arguments of `step_surface!`, because `SurfaceForcing` carries
only the global shortwave. `set_leaf_area!` prescribes the phenology.

```@autodocs
Modules = [Aeolus.LandSurface]
```
