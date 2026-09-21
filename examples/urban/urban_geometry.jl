# Urban-geometry utilities for the Tier 3 flagship example.
#
# Reads a GeoJSON-style building footprint file (one FeatureCollection
# with Polygon geometries and a `height` property per feature),
# rasterizes each polygon onto a RegularGrid via the SolidMask, and
# tags each ground-cell with a surface class (roof, street, park).
#
# The GeoJSON parser here is deliberately hand-rolled and minimal — we
# read only the subset of GeoJSON we need. Avoids adding a JSON.jl dep
# to Aeolus.
#
# A sample synthetic-city dataset is provided in
# `examples/urban/data/city.geojson`.

using Aeolus
using Aeolus.IBM: SolidMask, mark_solid!

export UrbanCity, load_city_geojson, rasterize_city!, surface_class_map

# ---------------------------------------------------------------------------
# Surface-class enum (integer flags packed into a UInt8 array)
# ---------------------------------------------------------------------------
const SURFACE_STREET = UInt8(1)
const SURFACE_ROOF   = UInt8(2)
const SURFACE_PARK   = UInt8(3)
const SURFACE_WATER  = UInt8(4)
const SURFACE_WALL   = UInt8(5)   # building-adjacent horizontal cell — represents effective wall heating

# ---------------------------------------------------------------------------
# Building footprint record
# ---------------------------------------------------------------------------

"""
    Building(polygon, height, class = SURFACE_ROOF)

A single footprint. `polygon` is a vector of `(x, y)` vertices in the
mesh coordinate system (meters from the mesh origin). `height` in
meters. `class` labels the roof surface (default `SURFACE_ROOF`).
"""
struct Building
    polygon::Vector{NTuple{2, Float64}}
    height::Float64
    class::UInt8
end

"""
    UrbanCity

Container for a set of `Building`s plus optional patch labels for
non-built surfaces (parks, water) as polygons.
"""
struct UrbanCity
    buildings::Vector{Building}
    patches::Vector{Building}    # non-solid surface labels (class = SURFACE_PARK / WATER)
end
UrbanCity() = UrbanCity(Building[], Building[])

# ---------------------------------------------------------------------------
# Minimal GeoJSON parser
# ---------------------------------------------------------------------------
# We recognise:
#   {
#     "type": "FeatureCollection",
#     "features": [
#       { "type": "Feature",
#         "geometry": { "type": "Polygon",
#                       "coordinates": [ [ [x, y], [x, y], ... ] ] },
#         "properties": { "height": 12, "class": "roof" } },
#       ...
#     ]
#   }

"""
    load_city_geojson(path) -> UrbanCity

Load a minimal GeoJSON file into an `UrbanCity`. Building polygons with
a `height` property become solid `Building`s (default class `roof`);
polygons with `class = "park"` or `class = "water"` and no height are
kept as `patches`.
"""
function load_city_geojson(path::AbstractString)
    text = read(path, String)
    features = _extract_features(text)
    city = UrbanCity()
    for feat in features
        poly    = _extract_polygon(feat)
        props   = _extract_properties(feat)
        cls_str = get(props, "class", "roof")
        h       = get(props, "height", nothing)
        cls     = _class_from_str(cls_str)
        if h === nothing
            push!(city.patches, Building(poly, 0.0, cls))
        else
            push!(city.buildings, Building(poly, Float64(h), cls))
        end
    end
    return city
end

_class_from_str(s) = s == "street" ? SURFACE_STREET :
                     s == "park"   ? SURFACE_PARK   :
                     s == "water"  ? SURFACE_WATER  :
                     SURFACE_ROOF

# Naive regex-driven extraction — the GeoJSON files we ship are small
# and well-formed, so this beats pulling in JSON.jl.
function _extract_features(text::AbstractString)
    # Split on `"type": "Feature"` occurrences and capture each block
    # up to the matching close-brace of the object.
    feats = String[]
    i = 1
    N = length(text)
    while true
        k = findnext("\"type\": \"Feature\"", text, i)
        k === nothing && break
        # find enclosing braces of the containing feature object
        # (walk backwards to '{' , forwards to matching '}')
        b0 = findprev('{', text, first(k) - 1)
        b0 === nothing && break
        depth = 0
        b1 = 0
        for j in b0:N
            c = text[j]
            if c == '{'; depth += 1
            elseif c == '}'
                depth -= 1
                if depth == 0; b1 = j; break; end
            end
        end
        b1 > 0 && push!(feats, String(text[b0:b1]))
        i = b1 + 1
    end
    return feats
end

function _extract_polygon(feat::AbstractString)
    # The first array of pairs inside `"coordinates":` is the outer ring.
    k = findfirst("\"coordinates\":", feat)
    k === nothing && error("polygon has no coordinates")
    # Skip to first `[[[` and read pairs until matching `]]`.
    substr = feat[last(k)+1:end]
    # find the outer ring — first triple-open then pairs, close with double
    ring_start = findfirst("[[", substr)
    ring_start === nothing && error("polygon coordinates malformed")
    rs = last(ring_start) + 1     # position just after "[["
    ring_end   = findnext("]]", substr, rs)
    ring_end === nothing && error("polygon coordinates malformed")
    ring_txt = substr[rs:first(ring_end)-1]
    # ring_txt is: [x, y], [x, y], ..., [x, y]
    verts = NTuple{2, Float64}[]
    for m in eachmatch(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", ring_txt)
        push!(verts, (parse(Float64, m.captures[1]), parse(Float64, m.captures[2])))
    end
    return verts
end

function _extract_properties(feat::AbstractString)
    props = Dict{String, Any}()
    k = findfirst("\"properties\":", feat)
    k === nothing && return props
    obj_start = findnext('{', feat, last(k))
    obj_start === nothing && return props
    depth = 0; obj_end = 0
    for j in obj_start:length(feat)
        c = feat[j]
        if c == '{'; depth += 1
        elseif c == '}'
            depth -= 1
            depth == 0 && (obj_end = j; break)
        end
    end
    obj = feat[obj_start:obj_end]
    for m in eachmatch(r"\"([A-Za-z_]+)\"\s*:\s*(?:\"([^\"]*)\"|(-?\d+(?:\.\d+)?))", obj)
        key = m.captures[1]
        props[key] = m.captures[2] !== nothing ? m.captures[2] :
                                                 parse(Float64, m.captures[3])
    end
    return props
end

# ---------------------------------------------------------------------------
# Rasterization: point-in-polygon test into the mesh
# ---------------------------------------------------------------------------

function _point_in_polygon(x::Real, y::Real, poly::Vector{NTuple{2, Float64}})
    n = length(poly)
    inside = false
    j = n
    @inbounds for i in 1:n
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) &&
           (x < (xj - xi) * (y - yi) / ((yj - yi) + eps()) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

"""
    rasterize_city!(sm, sfc, city, mesh)

Populate a `SolidMask sm` (marks cells inside building volumes) and a
surface-class array `sfc::Matrix{UInt8}` (2-D horizontal, ground-level
label per cell). Any cell not inside a building and not covered by a
patch stays at whatever `sfc` was initialised to (typical default:
`SURFACE_STREET`).
"""
function rasterize_city!(sm::SolidMask, sfc::AbstractMatrix{UInt8},
                         city::UrbanCity, mesh)
    xs = Aeolus.Grids.xnodes(mesh, Aeolus.Grids.CellCenter())
    ys = Aeolus.Grids.ynodes(mesh, Aeolus.Grids.CellCenter())
    zs = Aeolus.Grids.znodes(mesh, Aeolus.Grids.CellCenter())
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    # buildings → solid mask + roof label
    for b in city.buildings
        for j in eachindex(ys), i in eachindex(xs)
            if _point_in_polygon(xs[i], ys[j], b.polygon)
                sfc[i, j] = b.class
                for k in eachindex(zs)
                    if zs[k] <= b.height
                        sm.mask[Hx+i, Hy+j, Hz+k] = true
                    end
                end
            end
        end
    end
    # patches → surface label only
    for p in city.patches
        for j in eachindex(ys), i in eachindex(xs)
            if _point_in_polygon(xs[i], ys[j], p.polygon)
                sfc[i, j] = p.class
            end
        end
    end
    return (sm, sfc)
end

"""
    surface_class_map(city, mesh; default = SURFACE_STREET)

Convenience: create a fresh `Matrix{UInt8}` of horizontal surface class
labels initialized to `default`, then populate it by rasterizing `city`
(returning also a fresh solid mask).
"""
function surface_class_map(city::UrbanCity, mesh;
                           default::UInt8 = SURFACE_STREET,
                           tag_walls::Bool = true)
    sfc = fill(default, mesh.Nx, mesh.Ny)
    sm  = SolidMask(mesh)
    rasterize_city!(sm, sfc, city, mesh)
    tag_walls && tag_building_adjacent!(sfc, sm, mesh)
    return sm, sfc
end

"""
    tag_building_adjacent!(sfc, sm, mesh; class = SURFACE_WALL)

Overwrite any `SURFACE_STREET` cell that shares an edge with a building
footprint (in the horizontal plane) with `class`. This lets an
otherwise 2-D surface energy balance carry the effective heat load of
the building's vertical walls, since real walls dominate solar
absorption in dense urban geometry.
"""
function tag_building_adjacent!(sfc::AbstractMatrix{UInt8}, sm::SolidMask, mesh;
                                class::UInt8 = SURFACE_WALL)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    # A footprint cell is any (i, j) that has at least one solid cell in z.
    footprint = falses(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        any_solid = false
        for k in 1:Nz
            if sm.mask[Hx+i, Hy+j, Hz+k]; any_solid = true; break; end
        end
        footprint[i, j] = any_solid
    end
    # Any non-solid horizontal neighbour of a footprint cell → wall class.
    @inbounds for j in 1:Ny, i in 1:Nx
        if footprint[i, j]; continue; end
        if sfc[i, j] != SURFACE_STREET; continue; end
        # 4-neighbour periodic wrap
        neighbours = ((mod1(i-1, Nx), j),
                      (mod1(i+1, Nx), j),
                      (i, mod1(j-1, Ny)),
                      (i, mod1(j+1, Ny)))
        for (ni, nj) in neighbours
            if footprint[ni, nj]
                sfc[i, j] = class
                break
            end
        end
    end
    return sfc
end
