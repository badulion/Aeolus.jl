#!/usr/bin/env bash
# The drivers of the Berlin 50 m case (HPC benchmark #566): the scripts and the
# environment of ../berlin_100m, with its downloaded data (ERA5 and the OSM
# extract in ../berlin_100m/data; run ../berlin_100m/prepare.sh first).
set -euo pipefail
cd "$(dirname "$0")"
P=../berlin_100m
J="julia --project=$P/env"
$J $P/landuse.jl --cached --dx 50                 # OSM → $P/data/berlin_50m_landuse.csv (R = 5)
$J -m AeolusPrep static berlin_50m_20250824.toml --landuse $P/data/berlin_50m_landuse.csv
$J $P/dynamic.jl berlin_50m_20250824.toml          # ERA5 → input/berlin_50m_20250824_dynamic.nc
