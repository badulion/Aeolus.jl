# Generator for a plausible OSM XML sample.
#
# This script writes `sample_berlin_mitte.osm` — a synthetic OSM XML
# file with a European-style dense urban layout (perimeter-block housing
# + a few taller offices + a park + a small water body) at real
# geographic coordinates near Alexanderplatz, Berlin (52.52 N, 13.41 E).
#
# The file is real OSM XML — the same parser handles it as it would a
# genuine Overpass API download. We ship the generator (not just the
# output) so users can adjust the layout / anchor to their taste
# without editing OSM XML by hand.

using Printf
using Random

const OUT = joinpath(@__DIR__, "sample_berlin_mitte.osm")

# Anchor and scale
const LAT0 = 52.5220
const LON0 = 13.4100
const R    = 6_371_000.0
const COS_LAT = cos(deg2rad(LAT0))

# Meters → degrees around the anchor
function m_to_lat(dy)::Float64 ; return rad2deg(dy / R); end
function m_to_lon(dx)::Float64 ; return rad2deg(dx / (R * COS_LAT)); end

# Node counter & building counter
node_id = 0
way_id  = 0

# Buffer we'll write out
buf = IOBuffer()

println(buf, """<?xml version="1.0" encoding="UTF-8"?>""")
println(buf, """<osm version="0.6" generator="Aeolus.urban_sample">""")

# Push a node, return its id
function pushnode!(buf, dx_m, dy_m)
    global node_id += 1
    lat = LAT0 + m_to_lat(dy_m)
    lon = LON0 + m_to_lon(dx_m)
    @printf(buf, "  <node id=\"%d\" lat=\"%.7f\" lon=\"%.7f\"/>\n", node_id, lat, lon)
    return node_id
end

# Push a rectangular building at (cx, cy) with (w, h) side lengths and height H
function pushbuilding!(buf, cx, cy, w, h, H; class="yes")
    xs = [cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2]
    ys = [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2]
    ids = [pushnode!(buf, xs[i], ys[i]) for i in 1:5]
    global way_id += 1
    println(buf, "  <way id=\"$(way_id)\">")
    for id in ids
        println(buf, "    <nd ref=\"$id\"/>")
    end
    println(buf, "    <tag k=\"building\" v=\"$class\"/>")
    @printf(buf, "    <tag k=\"height\" v=\"%.1f\"/>\n", H)
    println(buf, "  </way>")
end

# Push a polygon feature (park, water) — no height
function pushpatch!(buf, verts, kind_key, kind_val)
    ids = [pushnode!(buf, v[1], v[2]) for v in verts]
    push!(ids, ids[1])
    global way_id += 1
    println(buf, "  <way id=\"$(way_id)\">")
    for id in ids
        println(buf, "    <nd ref=\"$id\"/>")
    end
    println(buf, "    <tag k=\"$kind_key\" v=\"$kind_val\"/>")
    println(buf, "  </way>")
end

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# Domain in meters relative to (LAT0, LON0):
#   x ∈ [-500, 500], y ∈ [-500, 500]   → 1 km × 1 km area
#
# Perimeter-block housing (Berliner "Blockrandbebauung"):
#   Blocks of ~120 × 120 m with 4 buildings around each interior courtyard,
#   arranged on a rectangular street grid with 30 m streets.

Random.seed!(1)

block_size  = 130.0
street      = 30.0
step        = block_size + street       # 160 m
grid_range  = -400:160.0:400            # 6 blocks per axis
building_w  = 45.0                       # perimeter-building side
building_h  = 45.0

for cy in grid_range, cx in grid_range
    # Skip the central "park" block and one "water" block
    if abs(cx) < step/2 && abs(cy - step) < step/2
        continue
    end
    if cy > 300 && abs(cx - step) < step/2
        continue
    end
    # Four perimeter buildings around a courtyard
    for (dx, dy) in ((-block_size/2 + building_w/2, 0),
                     ( block_size/2 - building_w/2, 0),
                     (0, -block_size/2 + building_h/2),
                     (0,  block_size/2 - building_h/2))
        # Varied heights: taller on the east side, shorter on the west
        H = 18.0 + 6*randn() + 0.02 * (cx + 500)
        H = clamp(H, 10.0, 60.0)
        pushbuilding!(buf, cx + dx, cy + dy, building_w,
                      dy == 0 ? building_h : building_w, H;
                      class = "residential")
    end
end

# A few large office / commercial buildings scattered
office_positions = [(-320.0, -320.0, 60.0, 60.0, 55.0),
                    (   0.0, -320.0, 80.0, 40.0, 75.0),
                    ( 300.0,  100.0, 70.0, 50.0, 90.0)]
for (cx, cy, w, h, H) in office_positions
    pushbuilding!(buf, cx, cy, w, h, H; class = "commercial")
end

# A central park (rectangular)
pushpatch!(buf, [(-60.0, 100.0), (60.0, 100.0), (60.0, 220.0), (-60.0, 220.0)],
           "leisure", "park")

# A small pond in the park
pushpatch!(buf, [(-20.0, 140.0), (20.0, 140.0), (20.0, 180.0), (-20.0, 180.0)],
           "natural", "water")

# Another green space in the south-east corner
pushpatch!(buf, [(220.0, -420.0), (400.0, -420.0), (400.0, -260.0), (220.0, -260.0)],
           "landuse", "grass")

# A larger water feature (river) crossing the northeast corner
pushpatch!(buf, [(240.0, 340.0), (400.0, 340.0), (400.0, 500.0), (240.0, 500.0)],
           "natural", "water")

println(buf, "</osm>")

open(OUT, "w") do io
    write(io, take!(buf))
end

@printf("Wrote %s with %d nodes and %d ways.\n", OUT, node_id, way_id)
