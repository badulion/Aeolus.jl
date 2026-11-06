# Urban facets — UrbanSurface

`UrbanSurface` (Phase 49) replaces the single heat-capacity slab of the
urban example with per-facet multi-layer conduction: every roof, wall and
ground (road) facet is a 1-D column through real material layers, closed
on the outside by the surface energy balance of the surface shared core and on
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
- **Glazing** (#27; theory: [Glazing and window facets](../../theory/urban_glazing.md))
  — `GlassPane` / `glass_pane_from_normal` and `pane_optics` (Fresnel +
  Beer–Lambert, per polarisation), `Glazing` units (`single_glazing`,
  `double_glazing`, `triple_glazing`) with `glazing_optics` and
  `glazing_optics_hemispherical`, gas gaps (`GlazingGap`, `FillGas`,
  `GLAZING_AIR`, `GLAZING_ARGON`, `gap_conductance`), the steady
  `glazing_u_value`, `inward_flowing_fractions` and `glazing_shgc`, and
  `WindowFacet <: AbstractSurface`, whose `transmitted_shortwave` is the
  solar gain handed to a building energy model. The fill-gas property
  fits (`gas_conductivity`, `gas_viscosity`, `gas_heat_capacity`) stay
  qualified.
- **Radiosity coupling** — `UrbanFacetScene`, `step_urban_facets!`,
  `aggregate_facet_fluxes`: one facet column per radiosity facet, with
  the facet-to-facet longwave exchange implicit.
- **Canyon exchange** (#30; theory: [BUBBLE / CAPITOUL validation](../../theory/urban_validation.md))
  — the Town Energy Balance closures for the air next to walls and
  streets: `canyon_wind_speed` (TEB canyon wind, skimming flow) and
  `rowley_convection_coefficient` (`h = 11.8 + 4.2 U`, TEB's wall
  exchange).

## Walls and street in the canyon wind

```julia
using Aeolus

H, W = 20.5, 14.0                                 # CAPITOUL, rue d'Alsace-Lorraine
U_can = canyon_wind_speed(4.0, 29.5, H, W)        # 4 m/s at 29.5 m above the roofs -> 0.74 m/s
wall = UrbanFacet(:wall, [WallLayer(:brick, 0.30)];
                  h_convective = rowley_convection_coefficient(U_can))   # 14.9 W m⁻² K⁻¹
street_air = SurfaceForcing(T_air = 290.0, U = U_can, z_ref = H / 2)    # MOST for the road
```

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

## A street canyon: every facet its own temperature

`UrbanFacetScene` gives each facet of a `Radiosity` scene its own
`UrbanFacet` and couples their longwave exchange implicitly through
`Radiosity.longwave_response`
([theory](../../theory/urban_radiosity_coupling.md)). Albedo and
emissivity come from the scene's `UrbanMaterials`.

```julia
hm = zeros(8, 12); hm[1:3, :] .= 20; hm[6:8, :] .= 20   # H/W = 2 street
scene = build_scene(hm, 5.0, 5.0; wall_subdivisions = 4)
vf = view_factors(scene)
u = UrbanFacetScene(scene, vf; T = 295.0,
                    options = f -> f.kind === :wall ? (h_convective = 10.0,
                                                        T_interior = 295.0) : (;))
air = SurfaceForcing(lw_down = 310.0, T_air = 291.0, U = 2.0, z_ref = 30.0)
fl = step_urban_facets!(u, air, 900.0; sun = (0.3, -0.2, 0.93), dni = 800.0, dhi = 120.0)
facet_temperatures(u)                 # one skin temperature per facet
agg = aggregate_facet_fluxes(u, fl)   # H, LE, … per unit plan area per LES column
```

## Reusing the conduction core

```julia
col = ConductionColumn([WallLayer(:concrete, 0.20)]; T = 293.0, θ = 0.5)
F_out, F_in = step_conduction!(col, fixed_temperature_boundary(300.0),
                               convective_boundary(295.0, 7.69), 600.0)
# column_heat_content(col) changed by exactly (F_out + F_in) * 600
```

## A window in the sun

```julia
using Aeolus

g = double_glazing()                        # 4-16-4 clear, air
glazing_optics(g, deg2rad(60))              # (R, T, A = [A_outer, A_inner])
glazing_shgc(g, 0.0), glazing_u_value(g)    # 0.733, 2.73 W m⁻² K⁻¹ (EN 410 / EN 673 coefficients)

w = WindowFacet(g; T = 293.0, T_interior = 295.0)
set_solar_incidence!(w, cosd(50); diffuse_fraction = 0.3)   # from SolarGeometry / radiosity
fl = step_surface!(w, SurfaceForcing(sw_down = 600.0, lw_down = 330.0, T_air = 293.0), 60.0)
transmitted_shortwave(w)    # W m⁻² of glazing entering the room as sunlight
interior_heat_flux(w)       # W m⁻² conducted / convected from the inner pane
```

```@autodocs
Modules = [Aeolus.UrbanSurface]
```
