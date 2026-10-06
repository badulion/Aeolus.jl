# Geometry of the Berlin case: surface classes, footprint rasterisation,
# and `build_geometry`, which turns a CaseConfig into the solid mask, the
# surface-class map and the radiosity scene with its view factors.
#
# Buildings come from OpenStreetMap (`osm_ingest.jl`); a minimal GeoJSON
# reader is kept for hand-made cities (`data/city.geojson`). Both parsers
# are hand-rolled: they read only the subset of the formats we need and
# keep Aeolus free of a JSON/XML dependency.

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

# The index ranges of the sorted cell centres `xs`, `ys` inside the
# bounding box of `poly`.
function _bbox_cells(poly, xs, ys)
    xmin, xmax, ymin, ymax = polygon_bbox(poly)
    return searchsortedfirst(xs, xmin):searchsortedlast(xs, xmax),
           searchsortedfirst(ys, ymin):searchsortedlast(ys, ymax)
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
    # Only the cells whose centre lies in a polygon's bounding box are
    # tested: a centre outside it is outside the polygon for the even–odd
    # test of `_point_in_polygon` too (no edge spans its y, or every
    # spanning edge lies on one side of it), so this is exact and only
    # makes the 6 km parent raster (#249) affordable.
    # buildings → solid mask + roof label
    for b in city.buildings
        ir, jr = _bbox_cells(b.polygon, xs, ys)
        for j in jr, i in ir
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
        ir, jr = _bbox_cells(p.polygon, xs, ys)
        for j in jr, i in ir
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

# ---------------------------------------------------------------------------
# The Berlin-case geometry
# ---------------------------------------------------------------------------

"""
    Geometry

Static geometry of a case, built once by [`build_geometry`](@ref):

* `city` — the parsed `UrbanCity`;
* `sm` — the IBM `SolidMask` of the buildings;
* `sfc` — the `(Nx, Ny)` surface-class map (`SURFACE_*`; couplings may
  add classes during `init!`, e.g. glass);
* `heightmap` — building top height (m) per column;
* `scene`, `vfac` — the radiosity scene and its view factors;
* `facet_col` — the `(i, j)` column of every facet;
* `svf` — the area-weighted sky-view factor per column;
* `wall_ids` — indices of the wall facets;
* `open_cols` — the columns with no building.
"""
struct Geometry{SM, HM, SC, VF}
    city::UrbanCity
    sm::SM
    sfc::Matrix{UInt8}
    heightmap::HM
    scene::SC
    vfac::VF
    facet_col::Vector{Tuple{Int,Int}}
    svf::Matrix{Float64}
    wall_ids::Vector{Int}
    open_cols::Vector{CartesianIndex{2}}
end

"""
    build_geometry(cfg, mesh; io = stdout) -> Geometry

Load the OSM city of `cfg` (projected about `cfg.lat0/lon0`, shifted so
that the point `cfg.offset` metres from the anchor is the domain centre),
rasterise it on `mesh` with building-adjacent cells tagged
`SURFACE_WALL`, and build the radiosity scene and its view factors (the
geometry is static, so this happens once).
"""
function build_geometry(cfg, mesh; io::IO = stdout)
    isfile(cfg.osm_path) ||
        error("Missing $(cfg.osm_path). Run examples/urban/data/fetch_alexanderplatz.jl first.")
    Lx, Ly = mesh.Lx, mesh.Ly
    city = load_city_osm(cfg.osm_path;
                         lat0 = cfg.lat0, lon0 = cfg.lon0,
                         x_shift = Lx / 2 - cfg.offset[1],
                         y_shift = Ly / 2 - cfg.offset[2],
                         min_height = cfg.min_height)
    sm, sfc = surface_class_map(city, mesh; tag_walls = true)

    # Era III: a radiosity scene replaces the ray-cast sky-view factor.
    # Geometry is static, so facets and view factors are built once. The
    # ground facet of every open column takes the material of its surface
    # class (park, water, street), see `scene_class_map`.
    cfg.verbose && @printf(io, "Computing sky-view factor... ")
    t_svf = time()
    heightmap = heightmap_from_solid_mask(sm)
    scene = build_scene(mesh, heightmap; wall_subdivisions = cfg.wall_subdivisions,
                        class_map = scene_class_map(sfc))
    vfac  = view_factors(scene)
    facet_col = Tuple{Int,Int}[f.col for f in scene.facets]
    svf = aggregate_to_grid(scene, vfac).svf
    cfg.verbose && @printf(io, "radiosity scene: %d facets in %.1fs, recip %.1e, closure %.1e\n",
                           nfacets(scene), time() - t_svf, reciprocity_error(vfac), closure_error(vfac))
    cfg.verbose && @printf(io, "SVF per column: min %.3f mean %.3f max %.3f\n",
                           minimum(svf), mean(svf), maximum(svf))
    wall_ids = [i for (i, f) in enumerate(scene.facets) if f.kind === :wall]
    open_cols = findall(==(0), heightmap)
    return Geometry(city, sm, sfc, heightmap, scene, vfac, facet_col,
                    Matrix{Float64}(svf), wall_ids, open_cols)
end

"""
    scene_class_map(sfc) -> Matrix{Int}

The radiosity ground-material class (`Radiosity.URBAN_*`) of every
column, from the surface-class map `sfc`. The `SURFACE_*` codes 1–4
(street, roof, park, water) are the `URBAN_*` material indices, so they
pass through. `SURFACE_WALL` marks a street cell next to a building so
the 2-D slab can carry the wall heat load; the ground facet in that
column is still street, so it maps to `URBAN_STREET`. `build_facets`
reads the map only for open (unbuilt) columns.
"""
function scene_class_map(sfc::AbstractMatrix{UInt8})
    return [c == SURFACE_PARK  ? URBAN_PARK  :
            c == SURFACE_WATER ? URBAN_WATER :
            c == SURFACE_ROOF  ? URBAN_ROOF  : URBAN_STREET
            for c in sfc]
end

"Building height (m) per column: top of the highest solid cell."
building_height(g::Geometry, mesh) =
    [maximum(k for k in 1:mesh.Nz if g.sm.mask[mesh.Hx+i, mesh.Hy+j, mesh.Hz+k]; init = 0) * mesh.dz
     for i in 1:mesh.Nx, j in 1:mesh.Ny]

"""
    park_lai_map(case) -> (veg_mask, LAI_map)

The park columns (`SURFACE_PARK`) and their leaf-area index
(`cfg.park_lai` on park cells, 0 elsewhere).
"""
function park_lai_map(case)
    sfc = case.geom.sfc
    veg_mask = (sfc .== SURFACE_PARK)
    LAI_map = fill(case.cfg.park_lai, size(sfc))
    LAI_map .*= veg_mask
    return veg_mask, LAI_map
end
