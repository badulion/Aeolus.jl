# Urban run type

`Aeolus.UrbanCase` (EX-a, #438) is the urban LES run type: a 3-D LES of a
city window whose buildings and land use come from a static input driver,
whose initial state and boundary forcing come from a sounding, a dynamic
input driver or an online Aeolus nest parent, and whose urban physics is
wired in through a registry of couplings (one file per coupling under
`src/urban/couplings/`). It was the Berlin example's own code
(`examples/urban/BerlinCase.jl`, `core/`, `couplings/`, `parent/`) and
moved into the package unchanged in its physics: the Berlin smoke run, the
54 runs of the smoke matrix and the resolved configurations of the four
Berlin scenarios are bit for bit those of before the move.

The Berlin (Alexanderplatz) case is now configuration only:
`examples/urban/configs/berlin_*.toml`, its static drivers
(`configs/input/*_static.nc`: the OpenStreetMap buildings and the parent's
land-use tiles), its dynamic driver (`configs/input/berlin_20190629_dynamic.nc`:
the Lindenberg sounding) and the source data in `examples/urban/data/`.

## Run drivers

| `[run] driver` | what runs | output (run folder) |
| :-- | :-- | :-- |
| `urban` ([`UrbanDriver`](@ref Aeolus.UrbanCase.UrbanDriver)) | the case: `build_case` + `run_case!` under the run monitor, checkpointed | `diagnostics.csv`, the NetCDF output (`[output] netcdf`), `output/results.toml` ([`run_results`](@ref Aeolus.UrbanCase.run_results)) |
| `urban_parent` ([`UrbanParentDriver`](@ref Aeolus.UrbanCase.UrbanParentDriver)) | the case's nest parent alone, for `[physics.options.nesting.parent] hours` | `output/parent.csv`, `parent_theta.csv`, `parent_surface.nc`, `results.toml` ([`parent_results`](@ref Aeolus.UrbanCase.parent_results)) |

Both read the same schema (`aeolus options urban`): an `urban_parent`
config may `extends` an urban case and change only `[run] driver` (and
turn `[checkpoint]` off). From Julia, `CaseConfig(path)` loads a config
file into the struct `build_case` consumes.

```bash
julia --project -m Aeolus run examples/urban/configs/berlin_summer.toml --set clock.nsteps=1800
julia --project -m Aeolus run examples/urban/configs/berlin_smoke.toml
julia --project -m Aeolus run examples/urban/configs/berlin_parent.toml
```

### The run's own checks: `results.toml`

Every `urban` run writes `output/results.toml`: a finite state, the
surface-energy budget per class, the water residual of the air and the
air-heat closure, each against
[`LEDGER_TOLERANCES`](@ref Aeolus.UrbanCase.LEDGER_TOLERANCES)
(`[ledgers] closed`), every air-heat term in W m⁻² of ground, the heat
carried in through the open or nest faces, the mean θ of the lowest
100 m at the start and the end, and the wall time per simulated hour. A
run whose ledgers do not close logs a warning; the run itself is not
failed. These are what the by-hand `examples/urban/berlin_checks.jl` and
`berlin_nest_check.jl` used to compute by running the model themselves.

## Input data: AeolusPrep.jl

The case reads its ground, its nest parent's land-use tiles and its
sounding from input drivers (`examples/urban/configs/input/`). They, and
the sources they are made from, are made by the companion package
AeolusPrep.jl (ONE-h, #457; [Input drivers](input_drivers.md)):

| `aeolusprep` command | writes |
| :-- | :-- |
| `static <config.toml> --source <file.osm \| file.geojson>` | the static driver(s) of the case (CFG-f, #361), with the nest parent's tiles from `[geometry] landuse` |
| `dynamic --from-sounding <config.toml>` | the sounding driver (CFG-g, #362) |
| `fetch-osm --box S,W,N,E --out file.osm` | OpenStreetMap buildings, parks and water of the box (Overpass API) |
| `fetch-landuse <config.toml>` | the nest parent's land-use table that [`read_parent_landuse`](@ref Aeolus.UrbanCase.read_parent_landuse) reads |
| `fetch-sounding --station ID --launches … --elevation m --out file.csv` | IGRA2 radiosonde launches as the sounding CSV that `read_sounding` reads |
| `osm-sample --lat0 --lon0 --out file.osm` | a synthetic OSM city |

The Berlin data were made with

```bash
# in an AeolusPrep.jl checkout, <aeolus> the Aeolus checkout
julia --project -m AeolusPrep fetch-osm --box 52.518,13.402,52.526,13.418 --out data/berlin/alexanderplatz_real.osm
julia --project -m AeolusPrep fetch-landuse <aeolus>/examples/urban/configs/berlin_summer.toml --min-height 8
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
Modules = [Aeolus.UrbanCase]
```
