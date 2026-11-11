#!/usr/bin/env bash
# The drivers of the Berlin 20 m case (HPC benchmark #566): the scripts and the
# environment of ../berlin_100m, with its downloaded data (ERA5 and the OSM
# extract in ../berlin_100m/data; run ../berlin_100m/prepare.sh first).
set -euo pipefail
cd "$(dirname "$0")"
P=../berlin_100m
J="julia --project=$P/env"
$J $P/landuse.jl --cached --dx 20                 # OSM → $P/data/berlin_20m_landuse.csv (R = 2)
$J -m AeolusPrep static berlin_20m_20250824.toml --landuse $P/data/berlin_20m_landuse.csv
$J $P/dynamic.jl berlin_20m_20250824.toml          # ERA5 → input/berlin_20m_20250824_dynamic.nc
