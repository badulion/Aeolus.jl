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

Both are legacy cases until the urban couplings are components (ONE-e
#454, ONE-f #455):

| `[run] legacy` | what runs | output (run folder) |
| :-- | :-- | :-- |
| `urban` ([`UrbanDriver`](@ref Aeolus.UrbanCase.UrbanDriver)) | the case: `build_case` + `run_case!` under the run monitor, checkpointed | `diagnostics.csv`, the NetCDF output (`[output] netcdf`), `output/results.toml` ([`run_results`](@ref Aeolus.UrbanCase.run_results)) |
| `urban_parent` ([`UrbanParentDriver`](@ref Aeolus.UrbanCase.UrbanParentDriver)) | the case's nest parent alone, for `[physics.options.nesting.parent] hours` | `output/parent.csv`, `parent_theta.csv`, `parent_surface.nc`, `results.toml` ([`parent_results`](@ref Aeolus.UrbanCase.parent_results)) |

Both read the same schema (`aeolus options urban`): an `urban_parent`
config is an urban case's file with `[run] legacy = "urban_parent"` (and
`[checkpoint]` off); every file is complete (no `extends`, ONE-b #451). From Julia, `CaseConfig(path)` loads a config
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

## Input data: `aeolus preprocess`

| subcommand | writes | was |
| :-- | :-- | :-- |
| `static <config.toml>` | the static driver(s) of the case (CFG-f, #361) | |
| `dynamic --from-sounding <config.toml>` | the dynamic driver (CFG-g, #362) | |
| `fetch-osm --box S,W,N,E --out file.osm` | OpenStreetMap buildings, parks and water of the box (Overpass API) | `data/fetch_alexanderplatz.jl` |
| `fetch-landuse <config.toml>` | the nest parent's land-use table ([`write_parent_landuse`](@ref Aeolus.UrbanCase.write_parent_landuse)) | `data/fetch_berlin_landuse.jl` |
| `fetch-sounding --station ID --launches … --elevation m --out file.csv` | IGRA2 radiosonde launches as a sounding CSV ([`igra_sounding_csv`](@ref Aeolus.UrbanCase.igra_sounding_csv)) | `data/fetch_lindenberg.jl` |
| `osm-sample --lat0 --lon0 --out file.osm` | a synthetic OSM city ([`synthetic_osm`](@ref Aeolus.UrbanCase.synthetic_osm)) | `data/generate_osm_sample.jl` |

The Berlin data were made with

```bash
julia --project -m Aeolus preprocess fetch-osm --box 52.518,13.402,52.526,13.418 --out examples/urban/data/alexanderplatz_real.osm
julia --project -m Aeolus preprocess fetch-landuse examples/urban/configs/berlin_summer.toml
julia --project -m Aeolus preprocess fetch-sounding --station GMM00010393 --elevation 112 \
    --launches 2019-06-29T00,2019-06-29T06,2019-06-29T12,2019-06-29T18,2019-06-30T00,2019-06-30T06 \
    --out examples/urban/data/lindenberg_10393_20190629.csv
julia --project -m Aeolus preprocess osm-sample --lat0 52.5220 --lon0 13.4100 --out examples/urban/data/sample_berlin_mitte.osm
```

`osm-sample` reproduces the committed `sample_berlin_mitte.osm` byte for
byte (`test/urban_run.jl`). `[preprocess] osm_path` may also name a
GeoJSON file (`.geojson`, coordinates in metres in the mesh frame), as the
synthetic city of the tests does.

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
