#!/usr/bin/env bash
# The drivers of the Berlin 100 m run, 24 Aug 2025 (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
J="julia --project=env"
$J fetch_era5.jl                                   # ERA5 → data/era5_berlin_20250824_{pl,sfc}.nc (CDS, ~/.cdsapirc)
$J landuse.jl --cached                             # OSM → data/berlin_100m_landuse.csv
$J -m AeolusPrep static berlin_100m_20250824.toml --landuse data/berlin_100m_landuse.csv
$J dynamic.jl                                     # ERA5 → input/berlin_100m_20250824_dynamic.nc
