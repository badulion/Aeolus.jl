# The static driver (CFG-f, #361): what a run reads once, at setup, about
# its ground: terrain, buildings, surface classes and leaf area on the
# run's own horizontal grid (form "resolved"), or the class fractions and
# building morphometry of an unresolved surface on a coarse grid (form
# "tiles", the nest parent).

const _XY = ("x", "y")
const _COORDS = [
    DriverVariable("x", ("x",), Float64; units = "m", standard_name = "projection_x_coordinate",
                   long_name = "x of the cell centres", doc = "the run mesh's coordinates"),
    DriverVariable("y", ("y",), Float64; units = "m", standard_name = "projection_y_coordinate",
                   long_name = "y of the cell centres", doc = "the run mesh's coordinates")]

"""
    STATIC_SCHEMA :: DriverSchema

The static driver of a building-resolving run (`aeolus_driver =
"static"`, `aeolus_form = "resolved"`): rasters on dimensions `x`, `y`
(the cell centres of the run's horizontal grid) and the building and
patch polygons as CF contiguous ragged arrays (a count variable per
polygon over the vertex dimension; rings of a polygon with holes are
`NaN`-separated). The polygon variables are absent when there are no
buildings (no patches).
"""
const STATIC_SCHEMA = DriverSchema("static", "resolved",
    "Terrain, buildings and surface classes on the run's grid, plus the footprint polygons.",
    vcat(_COORDS, [
    DriverVariable("zt", _XY, Float64; units = "m", long_name = "terrain height",
                   range = (-1e4, 1e4), doc = "above the domain bottom"),
    DriverVariable("building_height", _XY, Float64; units = "m", long_name = "building height",
                   range = (0, 1e4), doc = "the tallest building whose footprint holds the cell centre; 0: none"),
    DriverVariable("building_id", _XY, Int32; long_name = "building index",
                   range = (-1, typemax(Int32)), doc = "the last building (1-based, polygon order) holding the cell; -1: none"),
    DriverVariable("surface_class", _XY, UInt8; long_name = "surface class",
                   range = (1, 255), doc = "buildings, then patches drawn over them (`flag_values`, `flag_meanings`: the code table)"),
    DriverVariable("ground_class", _XY, UInt8; long_name = "surface class without buildings",
                   range = (1, 255), doc = "the patches alone: what a column shows when its building is removed"),
    DriverVariable("lai", _XY, Float64; units = "m2 m-2", standard_name = "leaf_area_index",
                   long_name = "leaf area index", range = (0, 100)),
    DriverVariable("building_x", ("building_vertex",), Float64; units = "m", required = false,
                   long_name = "x of the building vertices", doc = "NaN separates the rings of a polygon with holes"),
    DriverVariable("building_y", ("building_vertex",), Float64; units = "m", required = false,
                   long_name = "y of the building vertices"),
    DriverVariable("building_nvertex", ("building",), Int32; required = false, range = (0, typemax(Int32)),
                   long_name = "vertices per building", doc = "CF count variable (`sample_dimension`)"),
    DriverVariable("building_class", ("building",), UInt8; required = false, range = (1, 255),
                   long_name = "roof surface class"),
    DriverVariable("building_polygon_height", ("building",), Float64; units = "m", required = false,
                   range = (0, 1e4), long_name = "building top height"),
    DriverVariable("building_levels", ("building",), Float64; units = "1", required = false,
                   long_name = "building levels", doc = "NaN: untagged"),
    DriverVariable("building_min_height", ("building",), Float64; units = "m", required = false,
                   long_name = "building bottom height"),
    DriverVariable("building_material", ("building",), String; required = false,
                   long_name = "building material"),
    DriverVariable("building_roof", ("building",), String; required = false, long_name = "roof shape"),
    DriverVariable("patch_x", ("patch_vertex",), Float64; units = "m", required = false,
                   long_name = "x of the patch vertices"),
    DriverVariable("patch_y", ("patch_vertex",), Float64; units = "m", required = false,
                   long_name = "y of the patch vertices"),
    DriverVariable("patch_nvertex", ("patch",), Int32; required = false, range = (0, typemax(Int32)),
                   long_name = "vertices per patch"),
    DriverVariable("patch_class", ("patch",), UInt8; required = false, range = (1, 255),
                   long_name = "patch surface class")]))

"""
    STATIC_TILES_SCHEMA :: DriverSchema

The static driver of a run that does not resolve buildings (`aeolus_form
= "tiles"`, the nest parent): per column, the count of fine cells of each
surface class (out of `refinement²`, the variable's attribute; the tile
fractions are `class_count / refinement²`) and the building morphometry.
Global attributes `tile_x0`, `tile_y0`: the south-west corner of the
grid, m from the anchor.
"""
const STATIC_TILES_SCHEMA = DriverSchema("static", "tiles",
    "Surface-class counts and building morphometry per column of a coarse grid.",
    vcat(_COORDS, [
    DriverVariable("class", ("class",), UInt8; long_name = "surface class", range = (1, 255)),
    DriverVariable("class_count", ("x", "y", "class"), Int32; long_name = "fine cells of each surface class",
                   range = (0, typemax(Int32)), doc = "out of `refinement`² per column (attribute)"),
    DriverVariable("lambda_f", _XY, Float64; units = "1", long_name = "frontal area density", range = (0, 100)),
    DriverVariable("h_mean", _XY, Float64; units = "m", long_name = "roof-area-weighted mean building height",
                   range = (0, 1e4)),
    DriverVariable("h_max", _XY, Float64; units = "m", long_name = "maximum building height", range = (0, 1e4))]);
    attributes = ["tile_x0" => "x of the south-west corner of the grid, from the anchor (m)",
                  "tile_y0" => "y of the south-west corner of the grid, from the anchor (m)"])

# ---------------------------------------------------------------------------
# the resolved form
# ---------------------------------------------------------------------------

"""
    StaticInputs

What a building-resolving run reads from its static driver
([`read_static_driver`](@ref)) or makes in memory ([`static_inputs`](@ref)):
the grid (`x`, `y` cell centres, `dx`, `dy`), the anchor (`origin_lat`,
`origin_lon` at `origin_x`, `origin_y` in mesh coordinates), the rasters
`zt`, `building_height`, `building_id`, `surface_class`, `ground_class`,
`lai`, the `city` (its polygons) and the file's global `attributes`
(empty in memory).
"""
struct StaticInputs
    x::Vector{Float64}
    y::Vector{Float64}
    dx::Float64
    dy::Float64
    origin_lat::Float64
    origin_lon::Float64
    origin_x::Float64
    origin_y::Float64
    zt::Matrix{Float64}
    building_height::Matrix{Float64}
    building_id::Matrix{Int32}
    surface_class::Matrix{UInt8}
    ground_class::Matrix{UInt8}
    lai::Matrix{Float64}
    city::UrbanCity
    attributes::Dict{String,Any}
end

"""
    static_inputs(city, mesh; origin_lat, origin_lon, origin_x, origin_y,
                  class_lai = Dict{UInt8,Float64}(), default = SURFACE_STREET) -> StaticInputs

Rasterise `city` on `mesh` ([`rasterize_city`](@ref)): what
[`write_static_driver`](@ref) writes and a run reading no file uses.
Terrain is flat (`zt = 0`). `class_lai` maps a surface class to its leaf
area index: a cell takes that of its `surface_class`, else that of its
`ground_class`, else 0 (so a park under a removed building keeps its
leaves).
"""
function static_inputs(city::UrbanCity, mesh; origin_lat::Real, origin_lon::Real,
                       origin_x::Real, origin_y::Real,
                       class_lai::AbstractDict = Dict{UInt8,Float64}(), default::UInt8 = SURFACE_STREET)
    r = rasterize_city(city, mesh; default = default)
    lai = [Float64(get(class_lai, s, get(class_lai, g, 0.0))) for (s, g) in zip(r.surface_class, r.ground_class)]
    return StaticInputs(collect(Float64, xnodes(mesh, CellCenter())), collect(Float64, ynodes(mesh, CellCenter())),
                        Float64(mesh.dx), Float64(mesh.dy), origin_lat, origin_lon, origin_x, origin_y,
                        zeros(size(r.surface_class)), r.building_height, r.building_id, r.surface_class,
                        r.ground_class, lai, city, Dict{String,Any}())
end

const _FLAGS = Dict{String,Any}("flag_values" => UInt8[first(p) for p in SURFACE_CLASSES],
                                "flag_meanings" => join((last(p) for p in SURFACE_CLASSES), " "))

# CF contiguous ragged arrays of a polygon table
function _polygons(ps::Vector{Building})
    xs = Float64[]; ys = Float64[]; n = Int32[]
    for p in ps
        for (x, y) in p.polygon
            push!(xs, x); push!(ys, y)
        end
        push!(n, length(p.polygon))
    end
    return xs, ys, n
end

"""
    write_static_driver(path, st::StaticInputs; source = "", preprocessor = "") -> path
    write_static_driver(path, city, mesh; origin_lat, origin_lon, origin_x, origin_y,
                        class_lai = Dict(), source = "", preprocessor = …) -> path

Write the static driver ([`STATIC_SCHEMA`](@ref)) of `st` (or of
[`static_inputs`](@ref)`(city, mesh; …)`) to `path` (NetCDF-4, deflated):
the conventions' global attributes ([`DRIVER_ATTRIBUTES`](@ref)), with
`source` (the OSM or GeoJSON file, hashed) and the surface-class code
table on `surface_class` and `ground_class`.
"""
function write_static_driver(path::AbstractString, st::StaticInputs; source::AbstractString = "",
                             preprocessor::AbstractString = "")
    attrs = driver_attributes("static", "resolved"; origin_lat = st.origin_lat, origin_lon = st.origin_lon,
                              origin_x = st.origin_x, origin_y = st.origin_y, dx = st.dx, dy = st.dy,
                              source = source, out = path, preprocessor = preprocessor)
    mkpath(dirname(abspath(path)))
    ds = nc_create(path; attributes = attrs)
    try
        S = STATIC_SCHEMA
        ds.dim["x"] = length(st.x); ds.dim["y"] = length(st.y)
        _write_var!(ds, S["x"], st.x; extra = Dict{String,Any}("axis" => "X"))
        _write_var!(ds, S["y"], st.y; extra = Dict{String,Any}("axis" => "Y"))
        for (name, A) in (("zt", st.zt), ("building_height", st.building_height), ("building_id", st.building_id),
                          ("lai", st.lai))
            _write_var!(ds, S[name], A)
        end
        _write_var!(ds, S["surface_class"], st.surface_class; extra = _FLAGS)
        _write_var!(ds, S["ground_class"], st.ground_class; extra = _FLAGS)
        bs = st.city.buildings
        if !isempty(bs)
            xs, ys, n = _polygons(bs)
            ds.dim["building"] = length(bs); ds.dim["building_vertex"] = length(xs)
            _write_var!(ds, S["building_x"], xs); _write_var!(ds, S["building_y"], ys)
            _write_var!(ds, S["building_nvertex"], n; extra = Dict{String,Any}("sample_dimension" => "building_vertex"))
            _write_var!(ds, S["building_class"], UInt8[b.class for b in bs]; extra = _FLAGS)
            _write_var!(ds, S["building_polygon_height"], Float64[b.height for b in bs])
            _write_var!(ds, S["building_levels"], Float64[b.levels for b in bs])
            _write_var!(ds, S["building_min_height"], Float64[b.min_height for b in bs])
            _write_var!(ds, S["building_material"], String[b.material for b in bs])
            _write_var!(ds, S["building_roof"], String[b.roof for b in bs])
        end
        ps = st.city.patches
        if !isempty(ps)
            xs, ys, n = _polygons(ps)
            ds.dim["patch"] = length(ps); ds.dim["patch_vertex"] = length(xs)
            _write_var!(ds, S["patch_x"], xs); _write_var!(ds, S["patch_y"], ys)
            _write_var!(ds, S["patch_nvertex"], n; extra = Dict{String,Any}("sample_dimension" => "patch_vertex"))
            _write_var!(ds, S["patch_class"], UInt8[p.class for p in ps]; extra = _FLAGS)
        end
    finally
        close(ds)
    end
    return path
end

write_static_driver(path::AbstractString, city::UrbanCity, mesh; origin_lat, origin_lon, origin_x, origin_y,
                    class_lai::AbstractDict = Dict{UInt8,Float64}(), source::AbstractString = "",
                    preprocessor::AbstractString = "") =
    write_static_driver(path, static_inputs(city, mesh; origin_lat = origin_lat, origin_lon = origin_lon,
                                            origin_x = origin_x, origin_y = origin_y, class_lai = class_lai);
                        source = source, preprocessor = preprocessor)

_raw(ds, name) = ds[name].var[:]
_raw2(ds, name) = Array(ds[name].var[:, :])

function _read_polygons(ds, prefix, with_attributes::Bool)
    haskey(ds, prefix * "_nvertex") || return Building[]
    xs = _raw(ds, prefix * "_x"); ys = _raw(ds, prefix * "_y"); n = _raw(ds, prefix * "_nvertex")
    cls = _raw(ds, prefix * "_class")
    if with_attributes
        h = _raw(ds, "building_polygon_height"); lv = _raw(ds, "building_levels")
        mh = _raw(ds, "building_min_height"); mat = _raw(ds, "building_material"); rf = _raw(ds, "building_roof")
    end
    out = Building[]
    k = 0
    for b in eachindex(n)
        poly = NTuple{2,Float64}[(xs[k+v], ys[k+v]) for v in 1:n[b]]
        k += n[b]
        push!(out, with_attributes ?
              Building(poly, h[b], cls[b], lv[b], mh[b], String(mat[b]), String(rf[b])) :
              Building(poly, 0.0, cls[b]))
    end
    k == length(xs) || error("the $(prefix) polygons hold $k vertices, the vertex arrays $(length(xs))")
    return out
end

"Throw one error listing every problem `check_driver` found in `path`."
function _require_valid(path, schema)
    probs = check_driver(path, schema)
    isempty(probs) || error("the input driver $path is not valid:\n  " * join(probs, "\n  "))
    return nothing
end

"""
    read_static_driver(path; mesh, origin_lat, origin_lon, origin_x, origin_y) -> StaticInputs

Read the static driver at `path`. It is first checked against
[`STATIC_SCHEMA`](@ref) ([`check_driver`](@ref): kind, schema version,
attributes, required variables, ranges, a stale source), then against
the run: an error naming both grids unless its `x`, `y`, `dx`, `dy` and
anchor are the run's (`mesh`, and the anchor `origin_lat`/`origin_lon`
expected at `origin_x`/`origin_y` in mesh coordinates). There is no
resampling.
"""
function read_static_driver(path::AbstractString; mesh, origin_lat::Real, origin_lon::Real,
                            origin_x::Real, origin_y::Real)
    _require_valid(path, STATIC_SCHEMA)
    ds = NCDataset(path)
    try
        check_grid(path, ds, mesh; origin_lat = origin_lat, origin_lon = origin_lon,
                   origin_x = origin_x, origin_y = origin_y)
        city = UrbanCity(_read_polygons(ds, "building", true), _read_polygons(ds, "patch", false))
        attrs = Dict{String,Any}(k => v for (k, v) in ds.attrib)
        return StaticInputs(Float64.(_raw(ds, "x")), Float64.(_raw(ds, "y")), attrs["dx"], attrs["dy"],
                            attrs["origin_lat"], attrs["origin_lon"], attrs["origin_x"], attrs["origin_y"],
                            _raw2(ds, "zt"), _raw2(ds, "building_height"), _raw2(ds, "building_id"),
                            _raw2(ds, "surface_class"), _raw2(ds, "ground_class"), _raw2(ds, "lai"), city, attrs)
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# the tiles form
# ---------------------------------------------------------------------------

"""
    StaticTiles

The unresolved surface of a coarse grid ([`STATIC_TILES_SCHEMA`](@ref)):
the grid (`x`, `y`, `dx`, `dy`) and anchor, the south-west corner
`x0`, `y0` (m from the anchor), `class_count[I, J, c]` (fine cells of
class `c` out of `refinement²`), `lambda_f`, `h_mean`, `h_max`, the
source's provenance lines `notes`, and the file's `attributes`.
"""
struct StaticTiles
    x::Vector{Float64}
    y::Vector{Float64}
    dx::Float64
    dy::Float64
    origin_lat::Float64
    origin_lon::Float64
    origin_x::Float64
    origin_y::Float64
    x0::Float64
    y0::Float64
    refinement::Int
    class_count::Array{Int32,3}
    lambda_f::Matrix{Float64}
    h_mean::Matrix{Float64}
    h_max::Matrix{Float64}
    notes::Vector{String}
    attributes::Dict{String,Any}
end

"""
    write_static_tiles(path, t::StaticTiles; source = "", preprocessor = …) -> path

Write the tile form of the static driver ([`STATIC_TILES_SCHEMA`](@ref)).
"""
function write_static_tiles(path::AbstractString, t::StaticTiles; source::AbstractString = "",
                            preprocessor::AbstractString = "")
    attrs = driver_attributes("static", "tiles"; origin_lat = t.origin_lat, origin_lon = t.origin_lon,
                              origin_x = t.origin_x, origin_y = t.origin_y, dx = t.dx, dy = t.dy,
                              source = source, out = path, preprocessor = preprocessor)
    attrs["tile_x0"] = t.x0; attrs["tile_y0"] = t.y0
    attrs["source_notes"] = join(t.notes, "\n")
    mkpath(dirname(abspath(path)))
    ds = nc_create(path; attributes = attrs)
    try
        S = STATIC_TILES_SCHEMA
        nc = size(t.class_count, 3)
        ds.dim["x"] = length(t.x); ds.dim["y"] = length(t.y); ds.dim["class"] = nc
        _write_var!(ds, S["x"], t.x; extra = Dict{String,Any}("axis" => "X"))
        _write_var!(ds, S["y"], t.y; extra = Dict{String,Any}("axis" => "Y"))
        _write_var!(ds, S["class"], UInt8.(1:nc); extra = _FLAGS)
        _write_var!(ds, S["class_count"], t.class_count; extra = Dict{String,Any}("refinement" => Int32(t.refinement)))
        _write_var!(ds, S["lambda_f"], t.lambda_f)
        _write_var!(ds, S["h_mean"], t.h_mean)
        _write_var!(ds, S["h_max"], t.h_max)
    finally
        close(ds)
    end
    return path
end

"""
    read_static_tiles(path; mesh, origin_lat, origin_lon, origin_x, origin_y) -> StaticTiles

Read the tile form of the static driver at `path`, checked as
[`read_static_driver`](@ref) checks the resolved form.
"""
function read_static_tiles(path::AbstractString; mesh, origin_lat::Real, origin_lon::Real,
                           origin_x::Real, origin_y::Real)
    _require_valid(path, STATIC_TILES_SCHEMA)
    ds = NCDataset(path)
    try
        check_grid(path, ds, mesh; origin_lat = origin_lat, origin_lon = origin_lon,
                   origin_x = origin_x, origin_y = origin_y)
        a = Dict{String,Any}(k => v for (k, v) in ds.attrib)
        notes = String.(split(get(a, "source_notes", ""), '\n'; keepempty = false))
        return StaticTiles(Float64.(_raw(ds, "x")), Float64.(_raw(ds, "y")), a["dx"], a["dy"],
                           a["origin_lat"], a["origin_lon"], a["origin_x"], a["origin_y"],
                           a["tile_x0"], a["tile_y0"], Int(ds["class_count"].attrib["refinement"]),
                           Array(ds["class_count"].var[:, :, :]), _raw2(ds, "lambda_f"), _raw2(ds, "h_mean"),
                           _raw2(ds, "h_max"), notes, a)
    finally
        close(ds)
    end
end
