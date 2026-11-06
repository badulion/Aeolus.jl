# Land, urban and surface exchange: overview

The land, urban and surface-exchange work (Phases 46–53) made the surface as detailed as the atmosphere
above it: soil, vegetation, a resolved plant canopy, urban facets and
buildings, snow and ice, water surfaces, and the sub-grid mixing of all
of them. This page shows the one interface they share, how the pieces
are composed, and the measured validation. The physics of each piece is
on its own page (list at the end).

## One interface, many surfaces

Every surface model is an `AbstractSurface` of the surface shared core (`SurfaceExchange`),
`SurfaceExchange` ([theory](surface_exchange.md)). An implementer
provides its skin temperature, roughness lengths ``(z_{0m}, z_{0h},
z_{0q})``, broadband albedo and emissivity (the `RadiationCore`
generics), optionally a moisture availability ``β``, and a
`step_surface!(s, forcing, dt)` that advances its own state and returns
`SurfaceFluxes`. The contract supplies the rest once:

* **Monin–Obukhov fluxes** for any implementer through
  `turbulent_fluxes`, from a bracketed solve for ``ζ = z/L``
  (Businger–Dyer / Paulson unstable, Beljaars–Holtslag stable by
  default; COARE's convective form plugs in as another
  `AbstractStabilityFunctions`).
* **One sign convention**: fluxes positive upward, ``R_n`` positive
  into the surface, ``G`` positive into the substrate, so
  ``R_n = H + LE + G``.
* **One set of constants**: densities and conductivities of water, ice
  and air are in `SurfaceConstants`; latent heats, heat capacities and
  ``g`` are read from `MoistThermodynamics`, not restated.

Because the interface is the same, the composites below work with any
mix of surfaces, and an atmospheric model needs to know only the
interface.

## The skin balance as a boundary condition

Most implementers have a massless skin over a conducting column (soil,
facet wall, snow, ice). They share one technique: the surface energy
balance ``F(T_s) = R_n - H - LE`` is linearised about the current skin
temperature,

```math
F(T_s) \approx F(T_s^n) + \frac{\partial F}{\partial T_s}\,(T_s - T_s^n),
```

which is exactly a Robin condition ``-k\,\partial T/\partial z =
h\,(T_{env} - T_s) + q`` with ``h = -\partial F/\partial T_s``
(radiative plus turbulent, stability dependence included). The skin is
then solved implicitly together with the column, and the returned
fluxes close ``R_n - H - LE = G`` to round-off (1e-10 W m⁻² or better in
every implementer's tests). The same trick, applied to the radiosity
longwave of all facets at once, gives the implicit facet–facet coupling
of Phase 49: the longwave exchange is linear in the emission
``εσT^4``, so one linear solve couples every facet's skin, and the
scheme conserves energy to 7e-15 W m⁻² where a lagged coupling leaks
14.7 W m⁻².

## Composition

| layer | built from | phase |
| :-- | :-- | :-- |
| `SoilThermalColumn` | Richards water + enthalpy heat with freeze–thaw + aquifer | 46 |
| `VegetatedLandColumn` | two big leaves (`Photosynthesis`) over a `SoilThermalColumn` | 47 |
| resolved canopy | `PlantCanopy` drag, two-stream and leaf energy in the LES column | 48 |
| `UrbanFacet`, `WindowFacet`, `UrbanFacetScene` | conduction columns + glazing + the radiosity solver | 49 |
| `BuildingZone` | a zone air node implicit with its envelope of facets and columns | 50 |
| `LayeredSnowpack`, `IceColumn` | snow layers; brine-pocket ice | 51 |
| `WaterColumn`, `SeaSurface` | mixed layer with cool skin and warm layer; COARE fluxes | 52 |
| `SurfaceMosaic`, `BlendedMosaic` | any `AbstractSurface` tiles, flux-aggregated or forced at the blending height | 53 |

A `SurfaceMosaic` is itself an `AbstractSurface`, so a grid cell can hold
a soil tile, a snow tile and a water tile, each stepping its own balance,
and return ``\sum_i f_i F_i``. Flux aggregation is the point: averaging
the *parameters* instead (one surface with mean albedo, log-mean
roughness) gets the sign of ``H`` wrong for a stable + unstable pair
(+43.6 against −24.1 W m⁻²) and is 22–37 % low against a resolved
patchy-surface LES.

## Measured validation

Against observations or external reference models (the analytic and
conservation checks of each phase are on its page and in the
[roadmap](../design/roadmap.md)):

| phase | check | result | reference |
| :-- | :-- | :-- | :-- |
| contract | ψ_m(−1), ψ_h(−1) | 1.116232, 1.881227 | 1.1162, 1.8812 (Paulson 1970) |
| 46 | Neumann freezing front at 10 mm | −1.15e-3 relative | analytic |
| 47 | hourly LE, 5 FLUXNET towers, no calibration | r 0.83–0.94 | tower observations |
| 48 | in-canopy wind, wind tunnel | within 0.021 U_h | Brunet et al. (1994) |
| 48 | displacement height, CHATS with / without leaves | 8.36 / 6.64 m | 7.8 / 5.8 m |
| 49 | 3 mm clear pane, SHGC | 0.857 | 0.86 (NFRC) |
| 49 | CAPITOUL road / wall / roof mean temperature | +0.06 / −0.03 / +3.93 K | observed (roof fails, TEB +1.40) |
| 49 | CAPITOUL winter building energy use | 65.7 W m⁻² | 58.6 (inventory) |
| 50 | ASHRAE 140 annual loads, 10 cases | 19 of 20 in range | six reference programs |
| 51 | Summit snow SSA after 37 d | 23.8 m² kg⁻¹ | 23 ± 3 (observed) |
| 51 | MU71 sea ice, first-year mean thickness | 263.79 cm | 264.09 (BL99 code) |
| 52 | cool skin on the COARE ship data | 9.2e-5 K | COARE 3.6 code |
| 52 | peak afternoon SST warming, TOGA COARE | r 0.953, RMSE 0.246 K | ship observations |
| 53 | effective roughness, two-stripe LES, 8 cases | rms ln error 0.074 | Bou-Zeid et al. (2007) |

**Not validated against observations**, and stated as such in each
phase: soil moisture and temperature profiles, GPP, measured demand of
an instrumented building (BESTEST is program-to-program), SnowMIP sites,
buoy SST time series, and field campaigns over patchy terrain.

## In the 3-D flagship

The Berlin case runs the contract in every column: since #129 a
`SoilThermalColumn` with hydrology under streets, building-adjacent cells
and the park, slabs elsewhere, Monin–Obukhov fluxes from the first air
cell and an implicit surface drag ([the Berlin case](../design/berlin_surface_radiation.md#Surface-exchange-(#129))).
The vegetated land column, per-facet walls, building zones, snow, water
surfaces and tiles are wired in by open issues of the Berlin epic
(#139–#149).

## Detailed pages

* [Surface exchange (MOST)](surface_exchange.md)
* [Soil water (Richards)](soil_water.md), [Soil heat and freeze–thaw](soil_heat.md),
  [Water table and aquifer](water_table.md)
* [Leaf photosynthesis](photosynthesis.md), [Stomatal conductance](stomatal_conductance.md),
  [Vegetated land column](land_surface.md)
* [Resolved plant canopy](plant_canopy.md)
* [Urban facet conduction](urban_conduction.md), [Glazing and window facets](urban_glazing.md),
  [Facet–radiosity coupling](urban_radiosity_coupling.md),
  [BUBBLE / CAPITOUL validation](urban_validation.md)
* [Building energy (zone)](building_energy.md), [HVAC and exhaust heat](hvac_exhaust.md),
  [Occupancy and BESTEST](occupancy_validation.md), [Anthropogenic heat (BEM)](anthropogenic_heat.md)
* [Layered snowpack](snowpack.md), [Snow metamorphism and albedo](snow_metamorphism.md),
  [Lake and sea ice](ice_thermodynamics.md)
* [Water surfaces (mixed layer)](ocean_mixed_layer.md), [Air–sea fluxes and waves](air_sea_flux.md),
  [Sea spray and full COARE](sea_spray_coare.md)
* [Sub-grid surface tiles](surface_tiles.md), [Blending height](blending_height.md),
  [Heterogeneous-surface validation](heterogeneous_surface.md)
