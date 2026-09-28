# Era V — High-fidelity land, urban and surface exchange

The theory behind these modules is developed in
[Surface exchange and Monin–Obukhov similarity](../theory/surface_exchange.md);
the pages below are the generated API reference, one per module.

- [Shared contract — SurfaceExchange](era5/surface_exchange.md) — `Aeolus.SurfaceExchange`
- [Soil column — SoilColumn](era5/soil_column.md) — `Aeolus.SoilColumn` (Phase 46, Richards equation; [theory](../theory/soil_water.md))
  and heat conduction with freeze–thaw (#16; [theory](../theory/soil_heat.md))
- [Photosynthesis](era5/photosynthesis.md) — `Aeolus.Photosynthesis` (Farquhar C3 / Collatz C4, Ball-Berry / Medlyn stomatal conductance, Phase 47)
- [Vegetated land column — LandSurface](era5/land_surface.md) — `Aeolus.LandSurface` (Phase 47, #21: two-big-leaf canopy over the soil column, soil-water stress, FLUXNET validation; [theory](../theory/land_surface.md))
- [Plant canopy — PlantCanopy](era5/plant_canopy.md) — `Aeolus.PlantCanopy` (theory: [Resolved plant canopy](../theory/plant_canopy.md))
- [Urban facets — UrbanSurface](era5/urban_surface.md) — `Aeolus.UrbanSurface` (theory: [Urban facet conduction](../theory/urban_conduction.md), [Glazing and window facets](../theory/urban_glazing.md))
