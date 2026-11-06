# Air–sea fluxes — AirSeaFlux

`AirSeaFlux` (Phase 52, issue #39) gives open water a roughness that
responds to the wind and the waves, the matching heat and moisture
roughness, the COARE convective stability functions and gustiness, a
COARE 3.0 / 3.5 bulk algorithm, and `SeaSurface`, an `AbstractSurface`
for the tile framework. Theory and validation:
[Air–sea fluxes and wave-dependent roughness](../../theory/air_sea_flux.md).

Issue #40 adds the Andreas et al. (2015) sea-spray heat fluxes
(`AndreasSpray`, `sea_spray_fluxes`), the rain heat flux, the full COARE
algorithm with cool skin and warm layer (`coare_fluxes`,
`coare_warm_layer_series`), and `coare_flux_hook`, which drives the
prognostic `OceanMixedLayer.WaterColumn` with COARE fluxes. Theory and
validation: [Sea spray and the full COARE algorithm](../../theory/sea_spray_coare.md).

It extends the surface shared core (`SurfaceExchange`) rather than restating it: the
Monin–Obukhov solve is `SurfaceExchange.solve_surface_layer`,
`COAREStability` is a further `AbstractStabilityFunctions` whose Kansas
and stable branches are the contract's `BusingerDyerBeljaarsHoltslag`,
saturation and latent heat come from `MoistThermodynamics`, air
viscosity from `P3Microphysics` (Sutherland), water emissivity from
`SurfaceOptics`.

```julia
using Aeolus

# COARE 3.5, wind-speed Charnock
r = air_sea_fluxes(coare35_bulk(); U = 8.0, z_m = 10.0,
                   θ_a = 299.2, θ_s = 301.0, q_a = 0.016, q_s = 0.0235)
r.surface.u_star, r.C_D10N, r.w_g

# the same with a young sea (4 s peak period, 1 m waves)
rw = air_sea_fluxes(coare35_bulk(roughness = WaveAgeCharnock()); U = 8.0, z_m = 10.0,
                    θ_a = 299.2, θ_s = 301.0, q_a = 0.016, q_s = 0.0235,
                    sea = deep_water_sea_state(4.0, 1.0))

# as a surface
s  = SeaSurface(T_skin = 301.0)
fl = step_surface!(s, SurfaceForcing(U = 8.0, T_air = 299.0, q_air = 0.016), 600.0)
```

```julia
# full COARE: bulk sensor at 5 m, cool skin on, spray at 20 m/s
c = coare_fluxes(coare35_bulk(); U = 20.0, z_u = 10.0, T_air = 299.0, q_air = 0.017,
                 T_sea = 301.0, S = 35.0, lw_down = 400.0, spray = AndreasSpray())
c.H_total, c.LE_total, c.ΔT_cool, c.H_spray

# a prognostic water column driven by COARE fluxes
col = WaterColumn(dz = fill(1.0, 30), T = 301.0, S = 35.0)
step_surface!(col, SurfaceForcing(U = 20.0, T_air = 299.0, q_air = 0.017), 600.0;
              flux = coare_flux_hook(coare35_bulk(); spray = AndreasSpray()))
```

```@autodocs
Modules = [Aeolus.AirSeaFlux]
```
