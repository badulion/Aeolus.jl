# The city of the static driver (CFG-f, #361): building and patch
# footprints (`UrbanCity`, the polygons a static driver stores), the
# surface classes, and the rasterisation of the footprints on a run's grid
# (the run rasterises the polygons too, e.g. for the per-building surface
# exchange). The one copy; the readers of OpenStreetMap and GeoJSON that
# make an `UrbanCity` are in AeolusPrep.jl (ONE-h, #457).

# ---------------------------------------------------------------------------
# surface classes
# ---------------------------------------------------------------------------
"Surface class: street (paved ground, the default)."
const SURFACE_STREET = UInt8(1)
"Surface class: roof (the top of a building)."
const SURFACE_ROOF   = UInt8(2)
"Surface class: park (vegetated ground)."
const SURFACE_PARK   = UInt8(3)
"Surface class: water."
const SURFACE_WATER  = UInt8(4)
"Surface class: a street cell next to a building, carrying the wall heat load of the 2-D surface (set by [`tag_building_adjacent!`](@ref))."
const SURFACE_WALL   = UInt8(5)

"""
    SURFACE_CLASSES

The surface-class code table, `code => name`: 1 street, 2 roof, 3 park,
4 water, 5 wall. A static driver stores it as the CF `flag_values` and
`flag_meanings` of `surface_class`.
"""
const SURFACE_CLASSES = (SURFACE_STREET => "street", SURFACE_ROOF => "roof", SURFACE_PARK => "park",
                         SURFACE_WATER => "water", SURFACE_WALL => "wall")

# ---------------------------------------------------------------------------
# footprints
# ---------------------------------------------------------------------------

"""
    Building(polygon, height, class = SURFACE_ROOF; levels = NaN, min_height = 0.0,
             material = "", roof = "")

A single footprint. `polygon` is a vector of `(x, y)` vertices in the
mesh coordinate system (metres from the mesh origin); a polygon with
holes (a multipolygon) is several closed rings, each followed by a
`(NaN, NaN)` vertex. `height` (m) is the top of the building, `class` the
surface class of its roof (default `SURFACE_ROOF`). The attributes
`levels` (`NaN`: untagged), `min_height` (m, the bottom of a part that
does not reach the ground), `material` and `roof` (the roof shape) are
what the source says (OSM `building:levels`, `min_height`,
`building:material`, `roof:shape`); they are kept through a static
driver but do not enter the rasterisation.
"""
struct Building
    polygon::Vector{NTuple{2,Float64}}
    height::Float64
    class::UInt8
    levels::Float64
    min_height::Float64
    material::String
    roof::String
end
Building(polygon, height, class = SURFACE_ROOF; levels = NaN, min_height = 0.0, material = "", roof = "") =
    Building(polygon, Float64(height), UInt8(class), Float64(levels), Float64(min_height),
             String(material), String(roof))

"""
    UrbanCity(buildings = Building[], patches = Building[])

Container for a set of `Building`s plus patch labels for non-built
surfaces (parks, water) as polygons (`height` 0, the class the patch's
surface class).
"""
struct UrbanCity
    buildings::Vector{Building}
    patches::Vector{Building}
end
UrbanCity() = UrbanCity(Building[], Building[])

Base.isequal(a::Building, b::Building) = all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(Building))
Base.isequal(a::UrbanCity, b::UrbanCity) = isequal(a.buildings, b.buildings) && isequal(a.patches, b.patches)
Base.hash(b::Building, h::UInt) = foldr((f, acc) -> hash(getfield(b, f), acc), fieldnames(Building); init = hash(:Building, h))
Base.hash(c::UrbanCity, h::UInt) = hash(c.patches, hash(c.buildings, hash(:UrbanCity, h)))

"""
    polygon_bbox(poly) -> (xmin, xmax, ymin, ymax)

Bounding box of a footprint polygon, skipping the `NaN` ring separators
of a multipolygon.
"""
function polygon_bbox(poly)
    xmin = ymin = Inf; xmax = ymax = -Inf
    for (x, y) in poly
        isnan(x) && continue
        xmin = min(xmin, x); xmax = max(xmax, x)
        ymin = min(ymin, y); ymax = max(ymax, y)
    end
    return xmin, xmax, ymin, ymax
end

"""
    point_in_polygon(x, y, poly) -> Bool

The even–odd (crossing-number) test of the point `(x, y)` against
`poly`. An edge that touches a `NaN` vertex never counts, so for a
multipolygon (closed rings, each followed by `(NaN, NaN)`) the test over
all rings is exactly the polygon with its holes.
"""
function point_in_polygon(x::Real, y::Real, poly::Vector{NTuple{2,Float64}})
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
# bounding box of `poly`. A centre outside the box is outside the polygon
# for the even–odd test too (no edge spans its y, or every spanning edge
# lies on one side of it), so testing only these is exact and makes a
# 6 km raster at building resolution affordable.
function _bbox_cells(poly, xs, ys)
    xmin, xmax, ymin, ymax = polygon_bbox(poly)
    return searchsortedfirst(xs, xmin):searchsortedlast(xs, xmax),
           searchsortedfirst(ys, ymin):searchsortedlast(ys, ymax)
end

# ---------------------------------------------------------------------------
# rasterisation
# ---------------------------------------------------------------------------

"""
    rasterize_city(city, mesh; default = SURFACE_STREET)
        -> (building_id, building_height, surface_class, ground_class)

The footprints of `city` on the horizontal cell centres of `mesh`, cell
by cell (a cell belongs to a polygon when its centre does,
[`point_in_polygon`](@ref)):

* `building_id::Matrix{Int32}` — the last building (index into
  `city.buildings`, list order) whose footprint holds the cell, `-1`
  where none does;
* `building_height` (m) — the largest height of the buildings holding
  the cell, 0 where none does;
* `surface_class::Matrix{UInt8}` — `default`, then each building's class,
  then each patch's class, in list order (a patch drawn over a roof
  relabels it: a documented defect of the Berlin case, kept bit for bit
  until #141);
* `ground_class` — the same without the buildings: what a column shows
  when its building is removed (the building-free frame of an open
  case, [`solid_and_classes`](@ref)).
"""
function rasterize_city(city::UrbanCity, mesh; default::UInt8 = SURFACE_STREET)
    xs = xnodes(mesh, CellCenter()); ys = ynodes(mesh, CellCenter())
    Nx, Ny = length(xs), length(ys)
    bid = fill(Int32(-1), Nx, Ny)
    bh = zeros(Nx, Ny)
    sfc = fill(default, Nx, Ny)
    gnd = fill(default, Nx, Ny)
    for (n, b) in enumerate(city.buildings)
        ir, jr = _bbox_cells(b.polygon, xs, ys)
        for j in jr, i in ir
            if point_in_polygon(xs[i], ys[j], b.polygon)
                sfc[i, j] = b.class
                bid[i, j] = Int32(n)
                bh[i, j] = max(bh[i, j], b.height)
            end
        end
    end
    for p in city.patches
        ir, jr = _bbox_cells(p.polygon, xs, ys)
        for j in jr, i in ir
            if point_in_polygon(xs[i], ys[j], p.polygon)
                sfc[i, j] = p.class
                gnd[i, j] = p.class
            end
        end
    end
    return (building_id = bid, building_height = bh, surface_class = sfc, ground_class = gnd)
end

"""
    solid_and_classes(st, mesh; frame = (0, 0), tag_walls = true,
                      bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux)) -> (sm, sfc)

The IBM `SolidMask` and the `(Nx, Ny)` surface-class map of a run, from
the rasters `st` of [`rasterize_city`](@ref) or a [`StaticInputs`](@ref):
a column with a building is solid in every cell whose centre is at or
below `building_height`, and takes `surface_class`.

`frame = (fx, fy)` (cells) keeps the columns within `fx` cells of the
west and east faces and `fy` cells of the south and north faces free of
buildings (the `boundary_frame` of an open case, N-open-d #262, so the
boundary data is not imposed on a building cut by the box edge): there a
column is fluid and takes `ground_class`. With `tag_walls`, street cells
next to a building become `SURFACE_WALL` ([`tag_building_adjacent!`](@ref),
neighbours read through `bcs`).

Bit for bit what rasterising the city with the frame did before the
static driver (the union over buildings of `z ≤ h_b` is `z ≤ max h_b`).
"""
function solid_and_classes(st, mesh; frame::NTuple{2,Int} = (0, 0), tag_walls::Bool = true,
                           bcs::FieldBoundaryConditions = FieldBoundaryConditions(:periodic, :periodic, :noflux))
    Nx, Ny = size(st.surface_class)
    (Nx, Ny) == (mesh.Nx, mesh.Ny) ||
        error("solid_and_classes: rasters of $Nx × $Ny cells on a $(mesh.Nx) × $(mesh.Ny) mesh")
    zs = znodes(mesh, CellCenter())
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    fx, fy = frame
    sfc = copy(st.surface_class)
    sm = SolidMask(mesh)
    for j in 1:Ny, i in 1:Nx
        if i <= fx || i > Nx - fx || j <= fy || j > Ny - fy
            sfc[i, j] = st.ground_class[i, j]
            continue
        end
        st.building_id[i, j] >= 0 || continue
        h = st.building_height[i, j]
        for k in eachindex(zs)
            zs[k] <= h && (sm.mask[Hx+i, Hy+j, Hz+k] = true)
        end
    end
    tag_walls && tag_building_adjacent!(sfc, sm, mesh; bcs = bcs)
    return sm, sfc
end

"""
    surface_class_map(city, mesh; default = SURFACE_STREET, tag_walls = true,
                      frame = (0, 0), bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))
        -> (sm, sfc)

Rasterise `city` on `mesh` ([`rasterize_city`](@ref)) and return the
solid mask and the surface-class map ([`solid_and_classes`](@ref)).
"""
surface_class_map(city::UrbanCity, mesh; default::UInt8 = SURFACE_STREET, tag_walls::Bool = true,
                  frame::NTuple{2,Int} = (0, 0),
                  bcs::FieldBoundaryConditions = FieldBoundaryConditions(:periodic, :periodic, :noflux)) =
    solid_and_classes(rasterize_city(city, mesh; default = default), mesh;
                      frame = frame, tag_walls = tag_walls, bcs = bcs)

"""
    tag_building_adjacent!(sfc, sm, mesh; class = SURFACE_WALL,
                           bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

Overwrite any `SURFACE_STREET` cell that shares an edge with a building
footprint (in the horizontal plane) with `class`. This lets an
otherwise 2-D surface energy balance carry the effective heat load of
the building's vertical walls, since real walls dominate solar
absorption in dense urban geometry. The four neighbours are read with
`neighbor_index` on `bcs` (#262): across a periodic side they wrap,
across an open one the cell has no neighbour beyond the face.
"""
function tag_building_adjacent!(sfc::AbstractMatrix{UInt8}, sm::SolidMask, mesh;
                                class::UInt8 = SURFACE_WALL,
                                bcs::FieldBoundaryConditions = FieldBoundaryConditions(:periodic, :periodic, :noflux))
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
        neighbours = ((neighbor_index(mesh, bcs, i-1, 1), j),
                      (neighbor_index(mesh, bcs, i+1, 1), j),
                      (i, neighbor_index(mesh, bcs, j-1, 2)),
                      (i, neighbor_index(mesh, bcs, j+1, 2)))
        for (ni, nj) in neighbours
            if footprint[ni, nj]
                sfc[i, j] = class
                break
            end
        end
    end
    return sfc
end
