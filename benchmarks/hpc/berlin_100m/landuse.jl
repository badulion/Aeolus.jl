# The land-use table of the Berlin 100 m domain from OpenStreetMap
# (AeolusPrep.jl's fetch-landuse machinery, called directly: `aeolusprep
# fetch-landuse` reads the parent of an urban site, and this domain is a
# tiled domain alone).
#
#   julia --project=env landuse.jl            # fetch + write
#   julia --project=env landuse.jl --cached   # reuse data/raw/berlin_100m.osm
#
# The buildings, parks and water of the 20 km box (+ 100 m) via the
# Overpass API, classified (buildings lower than 8 m dropped; height from
# `height`, else `building:levels` × 3 m, else 12 m) and rasterised on 10 m
# cells, coarse-grained to the 100 m cells: per cell the counts of street,
# roof, park, water and wall 10 m cells, λ_p, λ_f, the mean and the maximum
# building height (data/berlin_100m_landuse.csv). © OpenStreetMap
# contributors, ODbL 1.0.

using AeolusPrep
using Aeolus

const LAT0, LON0 = 52.515, 13.40          # the domain centre ([radiation] latitude, longitude)
const L, DXC, R = 20000.0, 10.0, 10       # the domain (m), the raster cell (m), 10 m → 100 m
const MARGIN = 100.0
const DIR = joinpath(@__DIR__, "data")

csv = joinpath(DIR, "berlin_100m_landuse.csv")
raw = joinpath(DIR, "raw", "berlin_100m.osm")
h = L / 2 + MARGIN
R_E = Aeolus.InputDrivers._EARTH_R
dlat, dlon = rad2deg(h / R_E), rad2deg(h / (R_E * cosd(LAT0)))
box = round.((LAT0 - dlat, LON0 - dlon, LAT0 + dlat, LON0 + dlon); digits = 4)
mkpath(dirname(raw))
if !("--cached" in ARGS && isfile(raw))
    AeolusPrep.fetch_overpass(AeolusPrep.overpass_query(box; relations = true, timeout = 900), raw; max_time = 1800)
end
println("OSM extract: ", raw, " (", round(filesize(raw) / 2^20; digits = 1), " MB)")
N = round(Int, L / DXC)
cc = (lat0 = LAT0, lon0 = LON0, extent = (L, L, 10.0), size = (N, N, 1))
AeolusPrep.write_parent_landuse(csv, raw, cc; extent = L, refinement = R, min_height = 8.0, margin = MARGIN, box = box)
println("wrote ", csv)
