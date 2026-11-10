# Berlin at 50 m, 24 August 2025 (HPC benchmark #566)

The Berlin 100 m case (`../berlin_100m/`) on the same 20 km × 20 km × 3 km
domain at 400 × 400 × 60 cells (50 m × 50 m × 50 m) with `dt = 1 s` (the
CFL at 50 m), so a run fills a 72-core node and is not served mostly from
L3 cache (HANDOFF.md section 4). Everything else is the 100 m configuration.

## Drivers (`./prepare.sh`)

It uses the 100 m case's environment and downloaded data (`../berlin_100m/data`):

| file | how |
| :-- | :-- |
| `../berlin_100m/data/berlin_50m_landuse.csv` | `../berlin_100m/landuse.jl --cached --dx 50`: the same 10 m raster coarse-grained to 50 m (R = 5) |
| `input/berlin_50m_static.nc` | `aeolusprep static berlin_50m_20250824.toml --landuse …` |
| `input/berlin_50m_20250824_dynamic.nc` | `../berlin_100m/dynamic.jl berlin_50m_20250824.toml`: the same ERA5 data on this grid |
