"""
    InputDrivers

Aeolus's own input drivers (CFG-f, #361): NetCDF-4 files that hold what a
run reads from outside the model, so that a run reads its geometry from
one file instead of re-parsing OpenStreetMap at every start.

* **Conventions** (`input_drivers/conventions.jl`, shared with the
  dynamic driver of CFG-g): the global attributes every driver carries
  ([`DRIVER_ATTRIBUTES`](@ref)), the one map projection
  ([`latlon_to_meters`](@ref), equirectangular), and the declarative
  [`DriverSchema`](@ref) a file is checked against
  ([`check_driver`](@ref)) and the docs page is generated from
  ([`schema_markdown`](@ref)).
* **The static driver** (`input_drivers/static.jl`): terrain, buildings
  and surface classes on the run's own horizontal grid plus the building
  and patch polygons ([`STATIC_SCHEMA`](@ref), [`StaticInputs`](@ref),
  [`write_static_driver`](@ref), [`read_static_driver`](@ref)), and the
  tile form of an unresolved surface, class counts and building
  morphometry per column, for a coarse (nest parent) grid
  ([`STATIC_TILES_SCHEMA`](@ref), [`StaticTiles`](@ref)).
* **Preprocessors** (`preprocess/`): the OpenStreetMap and GeoJSON
  readers ([`load_city_osm`](@ref), [`load_city_geojson`](@ref)) that
  make an [`UrbanCity`](@ref), and its rasterisation on a grid
  ([`static_inputs`](@ref), [`solid_and_classes`](@ref)). They are the
  one copy of this code: the Berlin case (`examples/urban/`) uses them.

A file whose grid differs from the run's is an error naming both grids:
there is no resampling inside the model; the preprocessor is re-run for
another grid (`aeolus preprocess static <config.toml>`).

There is no PALM (PIDS) compatibility goal: the variable names are
Aeolus's own, short `snake_case` with CF attributes.
"""
module InputDrivers

using Dates: Dates, now, UTC
using ..Grids: xnodes, ynodes, znodes, CellCenter
using ..IBM: SolidMask
using ..BoundaryConditions: FieldBoundaryConditions, neighbor_index
import NCDatasets
using ..NCIO: NCDataset, nc_create, nc_defvar!
using ..IO: fnv1a64, git_state

export latlon_to_meters, DriverVariable, DriverSchema, check_driver, schema_markdown,
       DRIVER_ATTRIBUTES, DRIVER_SCHEMA_VERSION, file_fnv1a64,
       SURFACE_STREET, SURFACE_ROOF, SURFACE_PARK, SURFACE_WATER, SURFACE_WALL, SURFACE_CLASSES,
       Building, UrbanCity, polygon_bbox, point_in_polygon, load_city_osm, osm_class,
       load_city_geojson, rasterize_city, static_inputs, solid_and_classes,
       surface_class_map, tag_building_adjacent!,
       STATIC_SCHEMA, STATIC_TILES_SCHEMA, StaticInputs, StaticTiles,
       write_static_driver, read_static_driver, write_static_tiles, read_static_tiles

include("input_drivers/conventions.jl")
include("preprocess/city.jl")
include("preprocess/osm.jl")
include("preprocess/geojson.jl")
include("input_drivers/static.jl")

end # module
