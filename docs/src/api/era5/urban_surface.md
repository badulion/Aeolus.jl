# Urban facets — UrbanSurface

`UrbanSurface` (Phase 49) replaces the single heat-capacity slab of the
urban example with per-facet multi-layer conduction: every roof, wall and
ground (road) facet is a 1-D column through real material layers, closed
on the outside by the surface energy balance of the Era V contract and on
the inside by an interior boundary condition. The theory and the
validation are on [Urban facet conduction](../../theory/urban_conduction.md).

It has three layers of API:

- **Materials** — `ThermalMaterial`, `thermal_material`,
  `thermal_material_names`: conductivity, density and specific heat,
  every row cited. Thermal properties only: an exposed material names its
  `SurfaceOptics` spectral-library entry (`optical`), and a facet's
  albedo and emissivity are derived from it. `thermal_diffusivity`,
  `thermal_admittance` and `volumetric_heat_capacity` stay qualified
  (`Aeolus.UrbanSurface.thermal_admittance`).
- **The conduction core** — `WallLayer`, `ConductionColumn`,
  `ColumnBoundary` (`fixed_temperature_boundary`, `heat_flux_boundary`,
  `convective_boundary`) and `step_conduction!`. It knows nothing about
  the atmosphere; it is the code `BuildingEnergy` reuses for thermal mass.
- **Facets** — `UrbanFacet <: AbstractSurface`, with `step_surface!`,
  `interior_heat_flux` and the interior hooks `set_interior_temperature!`
  / `set_interior_boundary!`.

## A facet in a diurnal cycle

```julia
using Aeolus

roof = UrbanFacet(:roof, example_roof_layers(); T = 293.0, T_interior = 295.0)
wall = UrbanFacet(:wall, [WallLayer(:brick, 0.24), WallLayer(:gypsum_plaster, 0.015)];
                  T_interior = 295.0, h_convective = 15.0)
thermal_transmittance(example_roof_layers(); R_si = 0.10)   # U-value, W m⁻² K⁻¹

for step in 1:1440
    t = 60.0 * step
    f = SurfaceForcing(sw_down = max(0, 800sin(2π * (t / 86400 - 0.25))),
                       lw_down = 330.0, T_air = 293.0, U = 3.0)
    fl = step_surface!(roof, f, 60.0)     # Rn, H, LE, G (G = Rn - H - LE exactly)
    interior_heat_flux(roof)              # W m⁻² into the room
end
```

## Reusing the conduction core

```julia
col = ConductionColumn([WallLayer(:concrete, 0.20)]; T = 293.0, θ = 0.5)
F_out, F_in = step_conduction!(col, fixed_temperature_boundary(300.0),
                               convective_boundary(295.0, 7.69), 600.0)
# column_heat_content(col) changed by exactly (F_out + F_in) * 600
```

```@autodocs
Modules = [Aeolus.UrbanSurface]
```
