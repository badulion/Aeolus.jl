# A minimal GeoJSON reader → UrbanCity (CFG-f #361: moved from
# examples/urban/core/geometry.jl, the one copy). Hand-rolled: it reads
# only the subset of the format we need (a FeatureCollection of Polygon
# features with flat `properties`), so Aeolus needs no JSON dependency for
# hand-made cities such as examples/urban/data/city.geojson:
#
#   {
#     "type": "FeatureCollection",
#     "features": [
#       { "type": "Feature",
#         "geometry": { "type": "Polygon",
#                       "coordinates": [ [ [x, y], [x, y], ... ], [ hole ], ... ] },
#         "properties": { "height": 12, "class": "roof" } },
#       ...
#     ]
#   }
#
# Coordinates are taken as mesh metres (no projection).

"""
    load_city_geojson(path) -> UrbanCity

Load a minimal GeoJSON file into an `UrbanCity`. A Polygon feature with
a `height` property becomes a solid [`Building`](@ref) (class from
`class`, default `roof`); one without becomes a patch (`class = "park"`,
`"water"` or `"street"`). A polygon of one ring is that ring; a polygon
with holes (a courtyard) is all its rings, each followed by a
`(NaN, NaN)` vertex, the multipolygon convention of
[`point_in_polygon`](@ref).
"""
function load_city_geojson(path::AbstractString)
    text = read(path, String)
    city = UrbanCity()
    for feat in _extract_features(text)
        poly    = _extract_polygon(feat)
        props   = _extract_properties(feat)
        cls     = _class_from_str(get(props, "class", "roof"))
        h       = get(props, "height", nothing)
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

# The text of every `{ "type": "Feature", … }` object.
function _extract_features(text::AbstractString)
    feats = String[]
    i = 1
    N = lastindex(text)
    while true
        k = findnext("\"type\": \"Feature\"", text, i)
        k === nothing && break
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

const _PAIR = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"

# The rings of the `coordinates` array (bracket depth 2 inside it).
function _extract_polygon(feat::AbstractString)
    k = findfirst("\"coordinates\":", feat)
    k === nothing && error("polygon has no coordinates")
    a0 = findnext('[', feat, last(k) + 1)
    a0 === nothing && error("polygon coordinates malformed")
    rings = String[]
    depth = 0; r0 = 0
    for j in a0:lastindex(feat)
        c = feat[j]
        if c == '['
            depth += 1
            depth == 2 && (r0 = j)
        elseif c == ']'
            depth == 2 && push!(rings, feat[r0:j])
            depth -= 1
            depth == 0 && break
        end
    end
    isempty(rings) && error("polygon coordinates malformed")
    ring(t) = NTuple{2,Float64}[(parse(Float64, m.captures[1]), parse(Float64, m.captures[2]))
                                for m in eachmatch(_PAIR, t)]
    length(rings) == 1 && return ring(rings[1])
    verts = NTuple{2,Float64}[]
    for t in rings
        append!(verts, ring(t))
        push!(verts, (NaN, NaN))
    end
    return verts
end

function _extract_properties(feat::AbstractString)
    props = Dict{String,Any}()
    k = findfirst("\"properties\":", feat)
    k === nothing && return props
    obj_start = findnext('{', feat, last(k))
    obj_start === nothing && return props
    depth = 0; obj_end = 0
    for j in obj_start:lastindex(feat)
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
        props[key] = m.captures[2] !== nothing ? m.captures[2] : parse(Float64, m.captures[3])
    end
    return props
end
