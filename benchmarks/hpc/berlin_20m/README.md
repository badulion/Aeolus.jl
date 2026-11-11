# Berlin at 20 m, 24 August 2025 (scaling test, #566)

The Berlin 100 m case (`../berlin_100m/`) on the same 20 km × 20 km × 3 km
domain at 1000 × 1000 × 150 cells (20 m in every direction), `dt = 0.4 s`
(100 m: 2 s, 50 m: 1 s). Everything else is the 100 m configuration, except
that `[fftw]` has no `threads` key, so FFTW runs on `Threads.nthreads()`
threads.

## Why this size

PALM's published strong-scaling runs (wiki `doc/tec/parallel`, PALM 4.0:
2160³ and 4320³ points on 11 520–43 200 cores) hold about 0.9–4 M grid points
per core. Run 1's cases are far below that at 72 cores (100 m: 33 k, 50 m:
133 k points per core). This case has 150 M points:

| cores | points per core |
| --: | --: |
| 36 (one socket) | 4.2 M |
| 72 (the node) | 2.1 M |

4 M points per core on all 72 cores (288 M points) does not fit one node:
Aeolus holds about 0.7 kB per storage point live and about 1.4 kB at peak
(measured on the 100 m case on the laptop: 1.84 GiB live heap, 4.0 GiB peak
for 204 × 204 × 64 storage points), so 288 M points would need about 210 GB
live and about 400 GB at peak of the node's 256 GB. 150 M points need about
110 GB live and about 200 GB at peak. The land-use raster is 10 m, so the
cell must be a multiple of 10 m; 20 m is the finest cell that fits.

## Drivers (`./prepare.sh`)

Uses the 100 m case's environment and downloaded data (`../berlin_100m/data`):

| file | how |
| :-- | :-- |
| `../berlin_100m/data/berlin_20m_landuse.csv` | `../berlin_100m/landuse.jl --cached --dx 20`: the 10 m raster coarse-grained to 20 m (R = 2) |
| `input/berlin_20m_static.nc` | `aeolusprep static berlin_20m_20250824.toml --landuse …` |
| `input/berlin_20m_20250824_dynamic.nc` | `../berlin_100m/dynamic.jl berlin_20m_20250824.toml`: the same ERA5 data on this grid |

The dynamic driver holds 3-D initial fields of 150 M points; expect the
preparation to take much longer than the 50 m case's and to need tens of GB.
Run it on a compute node, not on the login node.

## The test (`../scaling.sh`)

One exclusive node, bench/all, one process per thread count (72, 36, 16, 8,
4, 2, 1), each building the model, 2 warm-up steps, then 5 steps with
`loop_fusion = "never"` and 5 with `"if_possible"`. See the header of
`scaling.sh` for what changed against run 1.

Expected cost, extrapolated from run 1 and unverified: model build grew from
29 s (100 m) to 154 s (50 m) for 4× the points, so 150 M points may take
about an hour per process; a step at 1 thread about 50–60 s. The whole
ladder may take most of the 24 h limit. Watch the 72-thread process first:
its build time and peak memory decide whether the rest is worth running.
