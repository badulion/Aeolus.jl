# Building interior — BuildingEnergy

`BuildingEnergy` (Phase 50) puts a room behind the urban facets: a
prognostic indoor air temperature coupled to the heat stored in the
envelope and in the building's internal structure, with glazing,
infiltration and ventilation. Theory and validation are on
[Building energy: indoor air and thermal mass](../../theory/building_energy.md).

Nothing about conduction is re-implemented here. Every wall, roof, floor
and partition is an `UrbanSurface.ConductionColumn`, stepped by
`UrbanSurface.step_conduction!`, or a whole `UrbanFacet`, stepped by its
own `step_surface!`. The zone only supplies their interior boundary
condition (the air temperature and the radiative gains) and closes the
air-node balance.

The API has three parts:

- **Elements**: `EnvelopeElement` (the exterior is outdoor air with a
  sol-air source, the ground, or adiabatic), `FacetElement` (an
  `UrbanFacet` with the full surface energy balance outside) and
  `InternalMass` (one or two faces exposed to the room).
- **The zone**: `BuildingZone`, `ZoneForcing` and `step_zone!`, which
  returns a `ZoneBudget`. Diagnostics: `zone_air_temperature`,
  `zone_operative_temperature`, `zone_mean_radiant_temperature`,
  `zone_heat_content`, `zone_heat_loss_coefficient`,
  `ventilation_conductance`, `air_heat_capacity` and
  `element_transmittance`.
- **The ISO 13790 reference**: `ISO13790Zone`, `step_iso13790!`,
  `iso13790_zone` (the 5R1C parameters derived from a `BuildingZone`),
  `iso13790_ideal_load!` (the §C.4.2 heating/cooling need) and
  `iso13786_areal_heat_capacity`.
- **HVAC and exhaust heat** (#32; theory:
  [HVAC demand and exhaust heat](../../theory/hvac_exhaust.md)):
  `IdealLoadsHVAC`, passed as `step_zone!(zone, forcing, dt, hvac)`, which
  also returns an `HVACBudget`. Equipment curves: `BiquadraticCurve`,
  `DXCoolingCurves`, `DX_CURVES_WINDOW_AC`, `DX_CURVES_PACKAGED_AC`,
  `DX_RATING_POINT`, `cooling_cop`, `available_cooling_capacity` and
  `condenser_inlet_temperature`. Humidity: `ZoneForcing(q_out, Q_latent)`,
  `zone_specific_humidity`, `zone_relative_humidity`.

## A zone behind two facets

```julia
using Aeolus

roof  = FacetElement(UrbanFacet(:roof, example_roof_layers(); T_interior = 294.0), 48.0)
walls = FacetElement(UrbanFacet(:wall, example_wall_layers(); T_interior = 294.0,
                                h_convective = 15.0), 63.6)
floor = EnvelopeElement([WallLayer(:xps_insulation, 0.1), WallLayer(:concrete, 0.15)], 48.0;
                        exterior = :adiabatic, h_inside = 1 / 0.17)
part  = InternalMass([WallLayer(:brick, 0.12)], 20.0; sides = 2)

zone = BuildingZone(; volume = 129.6, floor_area = 48.0, envelope = [roof, walls, floor],
                    internal_mass = [part], window_area = 12.0, window_U = 2.8,
                    infiltration_ach = 0.5, T_air = 294.0)

for step in 1:144
    t = 600.0 * step
    day = sin(2π * (t / 86400 - 0.25))
    sf = SurfaceForcing(sw_down = max(0, 800day), lw_down = 330.0, T_air = 290 + 5day, U = 3.0)
    b = step_zone!(zone, ZoneForcing(T_out = 290 + 5day, Q_internal = 500.0,
                                     Q_solar = max(0, 600day), surface = sf), 600.0)
    # b.T_air, b.Q_ventilation, … ; facet_fluxes(roof) is what the canopy air receives
end
```

## Air conditioning and its exhaust

```julia
ac = IdealLoadsHVAC(T_heat = 293.15, T_cool = 299.15, cop_rated = 3.0,
                    curves = DX_CURVES_WINDOW_AC, condenser = :air_cooled,
                    dehumidify = true, RH_max = 0.6)
b, h = step_zone!(zone, ZoneForcing(T_out = 306.0, q_out = 0.014, Q_internal = 500.0,
                                    Q_latent = 150.0, surface = sf), 600.0, ac)
h.Q_cooling, h.COP                            # coil load, operating COP
h.Q_exhaust_sensible, h.Q_exhaust_latent      # = Q_cooling (1 + 1/COP) to the outdoor air
```

## The 5R1C reference

```julia
iso = iso13790_zone(zone)                  # same construction, ISO 13790 parameters
θ_air, θ_s, θ_m = step_iso13790!(iso, 288.15; Φ_int = 500.0, Φ_sol = 200.0)
```

```@autodocs
Modules = [Aeolus.BuildingEnergy]
```
