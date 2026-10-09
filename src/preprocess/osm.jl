# OpenStreetMap XML → UrbanCity (CFG-f #361: moved from
# examples/urban/core/osm_ingest.jl, the one copy).
#
# Reads a subset of the OSM XML format: `<node id lat lon />` and
# `<way id>…<nd ref/>…<tag k v/></way>`. Ways with tag
# `building=yes` become UrbanCity buildings; ways with `leisure=park`,
# `landuse=grass|forest|meadow` become park patches;
# `natural=water|waterway=riverbank` become water patches.
#
# Lat/lon coordinates are projected to local metres with the drivers' one
# projection, `latlon_to_meters` (equirectangular about `(lat0, lon0)`).
#
# Multipolygon relations (`type=multipolygon`, #249) are read too, with
# the same tag classifier as the ways: their member ways are joined into
# closed rings, and the rings (outer and inner) are stored as ONE polygon,
# each ring closed and followed by a `NaN` vertex. `point_in_polygon`
# never counts an edge that touches a `NaN` vertex, so the even–odd test
# over all rings is exactly the multipolygon with its holes (a courtyard,
# an island in the Spree). The child's OSM file has no relations, so its
# city is unchanged bit for bit; the 6 km parent extract
# (`aeolus preprocess fetch-landuse`) needs them for the Spree, the parks and
# the courtyard blocks.
#
# KNOWN DEFECTS, kept so the refactor stays bit-for-bit (the water /
# Spree issue fixes them):
#   * `latm === nothing || lonm === nothing && continue` parses as
#     `latm === nothing || (lonm === nothing && continue)`;
#   * patches are rasterised after buildings (`rasterize_city!`), so a
#     park polygon can relabel a roof cell.

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

        _add_classified!(city, tags, poly, default_height, min_height)
    end
    occursin("<relation", text) &&
        _add_multipolygons!(city, text, node_lats, node_lons, lat0, lon0, x_shift, y_shift,
                            default_height, min_height)
    return city
end

"""
    osm_class(tags, default_height) -> (class, height) or `nothing`

The one OSM tag classifier of the Berlin case: a `building` (any value)
is a roof of the tagged height (`height`, or `building:levels` × 3 m; `default_height`
when untagged); `leisure=park` and `landuse=grass|forest|meadow|recreation_ground`
are park; `natural=water`, `waterway=riverbank` and
`water=lake|pond|river` are water. Anything else is `nothing` (the
ground class, street). Used for ways and multipolygon relations alike.
"""
function osm_class(tags::Dict{String,String}, default_height::Real)
    if get(tags, "building", "") == "yes" || haskey(tags, "building")
        return (SURFACE_ROOF, _extract_height(tags, default_height))
    elseif get(tags, "leisure", "") == "park" ||
           get(tags, "landuse", "") in ("grass", "forest", "meadow", "recreation_ground")
        return (SURFACE_PARK, 0.0)
    elseif get(tags, "natural", "") == "water" ||
           get(tags, "waterway", "") in ("riverbank",) ||
           get(tags, "water", "") in ("lake", "pond", "river")
        return (SURFACE_WATER, 0.0)
    end
    return nothing
end

function _add_classified!(city, tags, poly, default_height, min_height)
    cls = osm_class(tags, default_height)
    cls === nothing && return city
    c, h = cls
    if c == SURFACE_ROOF
        h >= min_height && push!(city.buildings, Building(poly, h, SURFACE_ROOF; osm_attributes(tags)...))
    else
        push!(city.patches, Building(poly, 0.0, c))
    end
    return city
end

_osm_tags(body) = Dict{String,String}(m.captures[1] => m.captures[2] for m in
    eachmatch(r"<tag\s+k=\"([^\"]+)\"\s+v=\"([^\"]+)\"\s*/>", body))

# Multipolygon relations → one NaN-separated polygon of closed rings each.
function _add_multipolygons!(city, text, node_lats, node_lons, lat0, lon0, x_shift, y_shift,
                             default_height, min_height)
    way_refs = Dict{Int,Vector{Int}}()
    for wm in eachmatch(r"<way\s+id=\"(\d+)\"[^>]*>(.*?)</way>"s, text)
        way_refs[parse(Int, wm.captures[1])] =
            [parse(Int, nm.captures[1]) for nm in eachmatch(r"<nd\s+ref=\"(\d+)\"\s*/>", wm.captures[2])]
    end
    for rm in eachmatch(r"<relation\s+id=\"(\d+)\"[^>]*>(.*?)</relation>"s, text)
        body = rm.captures[2]
        tags = _osm_tags(body)
        get(tags, "type", "") == "multipolygon" || continue
        osm_class(tags, default_height) === nothing && continue
        members = Vector{Int}[]
        for mm in eachmatch(r"<member\s+type=\"way\"\s+ref=\"(\d+)\"\s+role=\"([^\"]*)\"\s*/>", body)
            refs = get(way_refs, parse(Int, mm.captures[1]), nothing)
            refs === nothing || length(refs) < 2 || push!(members, refs)
        end
        poly = NTuple{2,Float64}[]
        for ring in _assemble_rings(members)
            all(r -> haskey(node_lats, r), ring) || continue
            for r in ring
                x, y = latlon_to_meters(node_lats[r], node_lons[r], lat0, lon0)
                push!(poly, (x + x_shift, y + y_shift))
            end
            push!(poly, (NaN, NaN))
        end
        isempty(poly) && continue
        _add_classified!(city, tags, poly, default_height, min_height)
    end
    return city
end

"""
    _assemble_rings(ways) -> Vector{Vector{Int}}

Join the node-id sequences `ways` (the member ways of a multipolygon,
outer and inner alike) end to end into closed rings (first id == last
id). A chain that cannot be closed is dropped (it is cut by the extract).
"""
function _assemble_rings(ways::Vector{Vector{Int}})
    rings = Vector{Int}[]
    pool = [copy(w) for w in ways]
    while !isempty(pool)
        cur = popfirst!(pool)
        while first(cur) != last(cur)
            k = findfirst(w -> first(w) == last(cur) || last(w) == last(cur), pool)
            k === nothing && break
            w = popat!(pool, k)
            append!(cur, first(w) == last(cur) ? w[2:end] : reverse(w)[2:end])
        end
        first(cur) == last(cur) && length(cur) >= 4 && push!(rings, cur)
    end
    return rings
end

"""
    osm_attributes(tags) -> NamedTuple

The [`Building`](@ref) attributes of an OSM building: `levels`
(`building:levels`, `NaN` if untagged or unparsable), `min_height` (m,
`min_height`, 0 if untagged), `material` (`building:material`) and
`roof` (`roof:shape`).
"""
function osm_attributes(tags::Dict{String,String})
    num(k, d) = haskey(tags, k) ? something(tryparse(Float64, replace(tags[k], r"[a-zA-Z ]" => "")), d) : d
    return (levels = num("building:levels", NaN), min_height = num("min_height", 0.0),
            material = get(tags, "building:material", ""), roof = get(tags, "roof:shape", ""))
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
