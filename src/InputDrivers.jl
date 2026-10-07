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
  ([`check_driver`](@ref)) and the docs pages are generated from
  ([`schema_markdown`](@ref), [`format_reference`](@ref)).
* **The static driver** (`input_drivers/static.jl`): terrain, buildings
  and surface classes on the run's own horizontal grid plus the building
  and patch polygons ([`STATIC_SCHEMA`](@ref), [`StaticInputs`](@ref),
  [`write_static_driver`](@ref), [`read_static_driver`](@ref)), and the
  tile form of an unresolved surface, class counts and building
  morphometry per column, for a coarse (nest parent) grid
  ([`STATIC_TILES_SCHEMA`](@ref), [`StaticTiles`](@ref)).
* **The dynamic driver** (`input_drivers/dynamic.jl`, CFG-g #362): the
  initial state, the boundary planes of the open faces over time and the
  large-scale and surface forcing ([`DYNAMIC_SCHEMA`](@ref),
  [`DynamicInputs`](@ref), [`read_dynamic_driver`](@ref),
  [`write_dynamic_driver`](@ref)); the planes feed the open boundaries as
  [`FileBoundaryData`](@ref), read two records at a time by a
  [`FileBoundaryFeed`](@ref).
* **Preprocessors** (`preprocess/`): the OpenStreetMap and GeoJSON
  readers ([`load_city_osm`](@ref), [`load_city_geojson`](@ref)) that
  make an [`UrbanCity`](@ref), and its rasterisation on a grid
  ([`static_inputs`](@ref), [`solid_and_classes`](@ref)). They are the
  one copy of this code: the Berlin case (`examples/urban/`) uses them.
  The sounding and case-table preprocessor of the dynamic driver
  ([`sounding_dynamic_inputs`](@ref), [`forcing_dynamic_inputs`](@ref))
  and the Aeolus-parent one, an in-run writer
  ([`NestBoundaryRecorder`](@ref)). The ERA5 preprocessor (CFG-h, #363):
  ERA5 on pressure or model levels, GRIB2 or NetCDF ([`read_era5`](@ref)),
  as initial profiles, boundary planes and large-scale and surface forcing
  ([`era5_dynamic_inputs`](@ref), [`write_era5_driver`](@ref)).

A file whose grid differs from the run's is an error naming both grids:
there is no resampling inside the model; the preprocessor is re-run for
another grid (`aeolus preprocess static <config.toml>`).

There is no PALM (PIDS) compatibility goal: the variable names are
Aeolus's own, short `snake_case` with CF attributes.
"""
module InputDrivers

using Dates: Dates, now, UTC
using Printf: @sprintf
using ..Grids: xnodes, ynodes, znodes, CellCenter
using ..IBM: SolidMask
using ..BoundaryConditions: FieldBoundaryConditions, neighbor_index, WindowedBoundaryData
import ..BoundaryConditions: snapshot_values
import NCDatasets
import GRIBDatasets
using ..NCIO: NCDataset, nc_create, nc_defvar!, nc_append_record!
using ..MoistThermodynamics: DEFAULT_CONSTANTS, exner_function, virtual_temperature, PhasePartition,
                             potential_temperature, air_density
using ..LargeScaleForcing: ForcingTable, forcing_value, Sounding, sounding_profiles, AdvectiveTendencies,
                           Subsidence, ProfileNudging
using ..IO: fnv1a64, git_state

export latlon_to_meters, meters_to_latlon, DriverVariable, DriverSchema, check_driver, schema_markdown,
       format_reference, ncdump_header,
       DRIVER_ATTRIBUTES, DRIVER_SCHEMA_VERSION, file_fnv1a64,
       SURFACE_STREET, SURFACE_ROOF, SURFACE_PARK, SURFACE_WATER, SURFACE_WALL, SURFACE_CLASSES,
       Building, UrbanCity, polygon_bbox, point_in_polygon, load_city_osm, osm_class,
       load_city_geojson, rasterize_city, static_inputs, solid_and_classes,
       surface_class_map, tag_building_adjacent!,
       STATIC_SCHEMA, STATIC_TILES_SCHEMA, StaticInputs, StaticTiles,
       write_static_driver, read_static_driver, write_static_tiles, read_static_tiles,
       DYNAMIC_SCHEMA, LS_TABLES, DynamicInputs, read_dynamic_driver, write_dynamic_driver,
       FileSnapshot, SlabArray, FileBoundaryData, FileBoundaryFeed, feed_window!, feed_steady!,
       advance_feed!, feed_coverage, check_coverage, check_planes, shift_table,
       dynamic_forcing, init_time_string, init_day_seconds, surface_pressure_at,
       sounding_dynamic_inputs, forcing_dynamic_inputs, NestBoundaryRecorder, record_boundary!, plane_snapshot,
       ERA5Data, ERA5_OUTPUTS, read_era5, era5_model_level_heights, era5_vertical, era5_subsidence,
       coriolis_parameter, era5_columns, era5_times, era5_dynamic_inputs, era5_planes!, write_era5_driver

include("input_drivers/conventions.jl")
include("preprocess/city.jl")
include("preprocess/osm.jl")
include("preprocess/geojson.jl")
include("input_drivers/static.jl")
include("input_drivers/dynamic.jl")
include("preprocess/sounding.jl")
include("preprocess/parent.jl")
include("preprocess/era5.jl")

end # module
