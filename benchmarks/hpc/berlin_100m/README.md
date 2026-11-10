# Berlin at 100 m, 24 August 2025 (00–24 h CEST)

One model domain without nesting: 20 km × 20 km × 3 km centred on Berlin
Mitte (52.515 N, 13.40 E), 200 × 200 × 60 cells (100 m × 100 m × 50 m),
2 s fixed step, 43 200 steps.

## Run

Everything in one go (environment, downloads, drivers, the run): `./run_all.sh` (`--fresh` re-downloads and remakes the drivers). Or just the run:

```bash
cd ~/coding/work/Aeolus_runs/berlin_100m_20250824
julia --project=env -t auto -m Aeolus run berlin_100m_20250824.toml
```

- **Cost.** About 1.1–1.3 s per step on 24 threads, so 13–16 h of wall time. That was measured over 10 min of model time (300 steps).
- **Output.** `runs/<date>_berlin_100m_20250824_<id>/`, every 30 min, about 1 GB. A checkpoint is written every 2 h of model time.
- **Resume.** `julia --project=env -t auto -m Aeolus run runs/<folder> --resume`.

`env/` develops two local checkouts:
- the Aeolus worktree `Aeolus.jl/.worktrees/berlin-run` (master at 9053fe04: the soil profiles of #523 and the temperature outputs of #540);
- the AeolusPrep checkout (branch `issue/6-era5-soil-profiles`, PR #7), which writes them.

Once both are merged, point `env` at master (`Pkg.develop` / `Pkg.add`).

## Drivers (`./prepare.sh` re-makes them)

| file | from | how |
| :-- | :-- | :-- |
| `input/berlin_100m_static.nc` | OpenStreetMap (Overpass, 265 MB extract, © OSM contributors, ODbL) | `landuse.jl` → `data/berlin_100m_landuse.csv` → `aeolusprep static --landuse` |
| `input/berlin_100m_20250824_dynamic.nc` | ERA5 (CDS, `~/.cdsapirc`), 22 UTC 23 Aug – 22 UTC 24 Aug 2025, hourly | `fetch_era5.jl` → `data/era5_berlin_20250824_{pl,sfc}.nc` → `dynamic.jl` |

### The surface: land-use tiles

The OSM buildings, parks and water are rasterised at 10 m and coarse-grained to the 100 m cells. Each cell gets:
- the street, roof, park, water and wall fractions;
- λp and λf;
- the mean and the maximum building height.

The buildings are not resolved: they act through roughness and displacement height. Building heights come from:

| source | buildings |
| :-- | --: |
| the `height` tag | 6,624 |
| `building:levels` × 3 m | 113,509 |
| neither, so the 12 m default | 142,085 |

Buildings below 8 m are dropped. The table gives mean λp 0.144, and the tallest cell mean is 142 m.

### The atmosphere: ERA5

- **Initial state.** The profiles at the domain centre at 22 UTC.
- **Lateral faces.** The four faces are open, fed hourly by ERA5's boundary planes.
- **Large-scale forcing.** The geostrophic wind from the driver (`ls_u_g`, `ls_v_g`).
- **Soil.** ERA5's four soil layers (stl1–4, swvl1–4, soil type "medium") start the tiles' soil columns (#521): 283 K at 3.5 cm, 290–292 K below, water 0.17–0.29 m³ m⁻³.
- **Radiation.** Mean-column two-stream, at the sun of the date.

## Known issues found while setting this up

- **The adaptive step does not work with open faces.** Its guard requires max |∇·u| < 1e-6 s⁻¹ after every step. The open faces leave a start-up divergence: 3.2e-3 s⁻¹ after step 1 here, and 1.4e-4 s⁻¹ in the shipped `lindenberg_era5` example. It decays about threefold per step, to round-off by step 30. The adaptive run therefore aborts at step 1, so this run uses a fixed 2 s step.
- **`aeolusprep dynamic --from-era5` refuses a tiled domain alone.** `run_kind` returns `:tiles` and the command accepts only `:model`. `dynamic.jl` makes the same calls without that check.
- **Start-up adjustment.** The smooth ERA5 inflow meets the perturbed interior. max |w| peaks at about 7 m/s about 4 min in, starting at the inflow faces, and is 3.5 m/s by 10 min, when rim and interior are alike (rms w 0.27 and 0.29 m s⁻¹). The first ~30 min are a spin-up and belong outside any analysis.
- **Not tested beyond 10 min of model time.** The full 24 h, through the day's convective boundary layer, has not been run.
