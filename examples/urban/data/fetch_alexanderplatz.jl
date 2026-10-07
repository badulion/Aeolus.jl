# Download real OSM data for the Alexanderplatz area of Berlin.
#
# Uses the public Overpass API (no auth required) via `curl`. Writes a
# plain OSM XML file that `load_city_osm` (Aeolus.InputDrivers, src/preprocess/osm.jl) reads
# directly. A new file makes the committed static drivers stale (their
# source hash no longer matches): re-run
#   julia --project -m Aeolus preprocess static examples/urban/configs/berlin_<s>.toml
# for each shipped config (CFG-f, #361).
#
# Bounding box: 52.518°N–52.526°N, 13.402°E–13.418°E — a roughly
# 1 km × 1 km square centered on Alexanderplatz.

using Printf

const OUT_PATH = joinpath(@__DIR__, "alexanderplatz_real.osm")
const S, W, N, E = 52.518, 13.402, 52.526, 13.418   # south, west, north, east

const QUERY = """
[out:xml][timeout:60];
(
  way["building"]($S,$W,$N,$E);
  way["leisure"="park"]($S,$W,$N,$E);
  way["landuse"~"grass|forest|meadow|recreation_ground"]($S,$W,$N,$E);
  way["natural"="water"]($S,$W,$N,$E);
);
(._;>;);
out body;
"""

const OVERPASS_URL = "https://overpass-api.de/api/interpreter"

@printf("Downloading OSM data for Alexanderplatz box (%.3f,%.3f)–(%.3f,%.3f)…\n",
        S, W, N, E)

# Use curl in GET-style with a url-encoded `data=` form field.
run(pipeline(`curl -sSG --max-time 90
              -A "Aeolus.jl OSM ingest"
              $OVERPASS_URL
              --data-urlencode $("data=" * QUERY)`;
             stdout = OUT_PATH))

sz = filesize(OUT_PATH)
@printf("Wrote %s (%.1f KB)\n", OUT_PATH, sz/1024)

# Sanity — how many buildings did we get?
n_ways = 0
n_buildings = 0
n_parks = 0
n_water = 0
for line in eachline(OUT_PATH)
    occursin("<way ",       line) && (n_ways      += 1)
    occursin("\"building\"", line) && (n_buildings += 1)
    occursin("\"park\"",     line) && (n_parks     += 1)
    occursin("\"water\"",    line) && (n_water     += 1)
end
@printf("XML contains ~%d ways (%d building tags, %d park tags, %d water tags)\n",
        n_ways, n_buildings, n_parks, n_water)
