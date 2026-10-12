# Urban site

`Aeolus.Urban` (EX-a #438; on the model since ONE-e #454; a component of
its domain since ONE-k #474, formerly `Aeolus.UrbanCase`) is the urban
site of the model: a 3-D LES of a city window whose buildings and land use
come from a static input driver, whose initial state and boundary forcing
come from a sounding, a dynamic input driver or an online Aeolus nest
parent, and whose urban physics is a set of components of the domain's
stage pipeline (one file per component under `src/components/urban/`;
[Components](../design/components.md)). A model config with `[urban]
enabled = true` is an urban site, and its sections enable the
components.

The site is a `Model` of one domain like any other: the
[`UrbanSite`](@ref Aeolus.Urban.UrbanSite) is the domain's first
component (it holds the site's shared state: options, geometry, surface
bus, tracers, budgets and reference state), every enabled urban component
follows it, and the site's `:timestepper` component takes the core step.
`step!(model)` advances it; `aeolus run` runs it with the model's run
loop, output writer, checkpoints and checks. There is no case object and
no run loop of its own (ONE-k, #474).

The Berlin (Alexanderplatz) case is configuration only:
`examples/urban/berlin_*.toml`, its static drivers
(`input/*_static.nc`: the OpenStreetMap buildings and the parent's
land-use tiles), its dynamic driver (`input/berlin_20190629_dynamic.nc`:
the Lindenberg sounding) and the source data in `examples/urban/data/`.

## Running it

| config | what runs | output (run folder) |
| :-- | :-- | :-- |
| the model, `[urban] enabled` | the site as a `Model` of one domain ([`build_model`](@ref Aeolus.Urban.build_model)) under the model's run loop: the run monitor, checkpoints (the site is checkpoint component `urban`), `[checks]` | `diagnostics.csv`, the NetCDF output of the variables `[output] variables` selects (`[output] netcdf`), `output/results.toml` ([`run_results`](@ref Aeolus.Urban.run_results) and the checks) |
| `[run] legacy = "urban_parent"` ([`UrbanParentDriver`](@ref Aeolus.Urban.UrbanParentDriver)) | the case's nest parent alone, for `[nesting.parent] hours`, until the parent is a domain of its own | `output/parent.csv`, `parent_theta.csv`, `parent_surface.nc`, `results.toml` ([`parent_results`](@ref Aeolus.Urban.parent_results)) |

Both read the model's sections (`aeolus options`): an `urban_parent`
config is an urban site's file with `[run] legacy = "urban_parent"` (and
`[checkpoint]` off); every file is complete (no `extends`, ONE-b #451).
The site's options are [`site_options`](@ref Aeolus.Urban.site_options)`(cfg)`,
read from the resolved config; they change in the TOML file or with
`--set`. The options are on [Run configuration: the urban
site](config.md#The-urban-site).

```bash
julia --project -m Aeolus run examples/urban/berlin_summer.toml --set time.nsteps=1800
julia --project -m Aeolus run examples/urban/berlin_smoke.toml
julia --project -m Aeolus run examples/urban/berlin_parent.toml
```

From Julia (tests, the bitwise harnesses), `build_case(site_options(cfg))`
builds the site and `Components.run_to!(site.model, t_end)` runs it as the
run loop does, without output, monitor and checkpoints.

### Output

The run's writer (`Aeolus.Output`) writes the variables `[output]
variables` selects from the domain's catalogue: the core's (`u`, `v`, `w`,
`p`, `theta`, their profiles and means) and the site's
([`Components.outputs`](@ref Aeolus.Components.outputs) of the site: the
local `hour` and the `solar_elevation` as time series, every output
tracer, the surface maps `Ts`, `sw_in`, `lw_in`, and every enabled
component's fields), plus the site's static geometry (sky-view factor,
building height, surface class) once. The Berlin configurations list
what the flagship's renderers read.

### The run's own checks: `results.toml`

Every run of an urban site writes `output/results.toml`: a finite state, the
surface-energy budget per class, the water residual of the air and the
air-heat closure, each against
[`BUDGET_TOLERANCES`](@ref Aeolus.Urban.BUDGET_TOLERANCES)
(`[budgets] closed`), every air-heat term in W m⁻² of ground, the heat
carried in through the open or nest faces, the mean θ of the lowest
100 m at the start and the end, and the wall time per simulated hour
(the site's `Components.run_results`), with the checks `[checks]`
enables. A run whose budgets do not close logs a warning; the run itself
is failed only by a required check (`urban_budgets`).

## Input data: AeolusPrep.jl

The case reads its ground, its nest parent's land-use tiles and its
sounding from input drivers (`examples/urban/input/`). They, and
the sources they are made from, are made by the companion package
AeolusPrep.jl (ONE-h, #457; [Input drivers](input_drivers.md)):

| `aeolusprep` command | writes |
| :-- | :-- |
| `static <config.toml> --source <file.osm \| file.geojson>` | the static driver(s) of the case (CFG-f, #361), with the nest parent's tiles from `[urban] landuse` |
| `dynamic --from-sounding <config.toml>` | the sounding driver (CFG-g, #362) |
| `fetch-osm --box S,W,N,E --out file.osm` | OpenStreetMap buildings, parks and water of the box (Overpass API) |
| `fetch-landuse <config.toml>` | the nest parent's land-use table that [`read_parent_landuse`](@ref Aeolus.Urban.read_parent_landuse) reads |
| `fetch-sounding --station ID --launches … --elevation m --out file.csv` | IGRA2 radiosonde launches as the sounding CSV that `read_sounding` reads |
| `osm-sample --lat0 --lon0 --out file.osm` | a synthetic OSM city |

The Berlin data were made with

```bash
# in an AeolusPrep.jl checkout, <aeolus> the Aeolus checkout
julia --project -m AeolusPrep fetch-osm --box 52.518,13.402,52.526,13.418 --out data/berlin/alexanderplatz_real.osm
julia --project -m AeolusPrep fetch-landuse <aeolus>/examples/urban/berlin_summer.toml --min-height 8
julia --project -m AeolusPrep fetch-sounding --station GMM00010393 --elevation 112 \
    --launches 2019-06-29T00,2019-06-29T06,2019-06-29T12,2019-06-29T18,2019-06-30T00,2019-06-30T06 \
    --out <aeolus>/examples/urban/data/lindenberg_10393_20190629.csv
julia --project -m AeolusPrep osm-sample --lat0 52.5220 --lon0 13.4100 --out data/berlin/sample_berlin_mitte.osm
julia --project -m AeolusPrep regenerate <aeolus>      # every committed driver (recipes.toml)
```

The OpenStreetMap extracts stay in AeolusPrep (`data/berlin/`); the
land-use table and the sounding CSV, which a run may read in memory
(`parent_static = "none"`, `dynamic = "none"`), stay in
`examples/urban/data/`. `--source` may also name a GeoJSON file
(`.geojson`, coordinates in metres in the mesh frame), as the synthetic
city of the tests does.

## Analysis of the output

The scripts left in `examples/urban/` read run output only:
`alexanderplatz_animation.jl` and `alexanderplatz_surface_diagnostics.jl`
(the NetCDF of a `berlin_summer.toml` run), `berlin_parent_figures.jl`,
`berlin_parent_lindenberg_check.jl` and `plot_parent_surface.jl` (the
output folder of an `urban_parent` run). `test/urban_run.jl` checks that
no script there builds or runs the model.

## Reference

```@autodocs
Modules = [Aeolus.Urban]
```
