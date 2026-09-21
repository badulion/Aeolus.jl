# OpenStreetMap XML → UrbanCity ingest.
#
# Reads a subset of the OSM XML format: `<node id lat lon />` and
# `<way id>…<nd ref/>…<tag k v/></way>`. Ways with tag
# `building=yes` become UrbanCity buildings; ways with `leisure=park`,
# `landuse=grass|forest|meadow` become park patches;
# `natural=water|waterway=riverbank` become water patches.
#
# Lat/lon coordinates are projected to local metres using a simple
# equirectangular projection anchored at a user-supplied `(lat0, lon0)`.
# Good to sub-metre accuracy over a few km — appropriate for LES
# domains of ~100 m to ~5 km.

include(joinpath(@__DIR__, "urban_geometry.jl"))

export load_city_osm

const _EARTH_R = 6_371_000.0

"""
    latlon_to_meters(lat, lon, lat0, lon0) -> (x, y)

Project geographic (lat, lon) in degrees to local (x, y) meters,
tangent to the surface at (`lat0`, `lon0`). Equirectangular /
plate-carrée: sufficient for ≤ 5 km domains.
"""
function latlon_to_meters(lat::Real, lon::Real, lat0::Real, lon0::Real)
    x = _EARTH_R * deg2rad(lon - lon0) * cos(deg2rad(lat0))
    y = _EARTH_R * deg2rad(lat - lat0)
    return (x, y)
end

"""
    load_city_osm(path; lat0, lon0, x_shift = 0, y_shift = 0,
                        default_height = 12.0)

Parse the OSM XML file at `path` and return an `UrbanCity` whose
polygons are in local meters relative to the anchor `(lat0, lon0)`.
Optional `x_shift`, `y_shift` allow re-centering the extracted city
inside the target mesh.

`default_height` (meters) is used for building ways that have no
`height` or `building:levels` tag.
"""
function load_city_osm(path::AbstractString;
                       lat0::Real, lon0::Real,
                       x_shift::Real = 0, y_shift::Real = 0,
                       default_height::Real = 12.0,
                       min_height::Real = 0.0)
    text = read(path, String)

    # Parse nodes: id → (lat, lon)
    node_lats = Dict{Int, Float64}()
    node_lons = Dict{Int, Float64}()
    for m in eachmatch(r"<node\s+[^/>]*id=\"(\d+)\"[^/>]*/?>", text)
        # nested capture for lat/lon
        line = m.match
        idm = m.captures[1]
        latm = match(r"lat=\"(-?\d+(?:\.\d+)?)\"", line)
        lonm = match(r"lon=\"(-?\d+(?:\.\d+)?)\"", line)
        latm === nothing || lonm === nothing && continue
        node_lats[parse(Int, idm)] = parse(Float64, latm.captures[1])
        node_lons[parse(Int, idm)] = parse(Float64, lonm.captures[1])
    end

    # Parse ways: id → list of node refs + tags
    city = UrbanCity()
    for wm in eachmatch(r"<way\s+id=\"(\d+)\"[^>]*>(.*?)</way>"s, text)
        body = wm.captures[2]
        refs = Int[]
        for nm in eachmatch(r"<nd\s+ref=\"(\d+)\"\s*/>", body)
            push!(refs, parse(Int, nm.captures[1]))
        end
        length(refs) < 3 && continue
        tags = Dict{String, String}()
        for tm in eachmatch(r"<tag\s+k=\"([^\"]+)\"\s+v=\"([^\"]+)\"\s*/>", body)
            tags[tm.captures[1]] = tm.captures[2]
        end
        # Build polygon (skip if any node ref is missing)
        poly = NTuple{2, Float64}[]
        skip = false
        for r in refs
            if !haskey(node_lats, r); skip = true; break; end
            x, y = latlon_to_meters(node_lats[r], node_lons[r], lat0, lon0)
            push!(poly, (x + x_shift, y + y_shift))
        end
        skip && continue
        # Ensure the polygon is closed / has at least 3 unique vertices
        length(poly) < 3 && continue

        # Classify
        if get(tags, "building", "") == "yes" || haskey(tags, "building")
            h = _extract_height(tags, default_height)
            h >= min_height || continue
            push!(city.buildings, Building(poly, h, SURFACE_ROOF))
        elseif get(tags, "leisure", "") == "park" ||
               get(tags, "landuse", "") in ("grass", "forest", "meadow", "recreation_ground")
            push!(city.patches, Building(poly, 0.0, SURFACE_PARK))
        elseif get(tags, "natural", "") == "water" ||
               get(tags, "waterway", "") in ("riverbank",) ||
               get(tags, "water", "") in ("lake", "pond", "river")
            push!(city.patches, Building(poly, 0.0, SURFACE_WATER))
        end
    end
    return city
end

# Height parsing: `height=45.5`, `height="45"`, `building:levels=15` (× 3 m).
function _extract_height(tags::Dict{String,String}, default_height::Real)
    if haskey(tags, "height")
        s = replace(tags["height"], r"[a-zA-Z ]" => "")
        v = tryparse(Float64, s)
        v !== nothing && return v
    end
    if haskey(tags, "building:levels")
        s = replace(tags["building:levels"], r"[a-zA-Z ]" => "")
        v = tryparse(Float64, s)
        v !== nothing && return v * 3.0
    end
    return default_height
end
