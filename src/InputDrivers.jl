"""
    InputDrivers

Aeolus's own input drivers (CFG-f, #361): NetCDF-4 files that hold what a
run reads from outside the model. A run reads its geometry, initial state
and forcing from them, never from OpenStreetMap, GIS or reanalysis files.

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
* **The city** (`input_drivers/city.jl`): the footprints a static
  driver stores ([`UrbanCity`](@ref), [`Building`](@ref)), the surface
  classes, and their rasterisation on a grid ([`rasterize_city`](@ref),
  [`static_inputs`](@ref), [`solid_and_classes`](@ref)), which a run uses
  as well as the writer.
* **A sounding or a case's tables as dynamic inputs**
  (`input_drivers/sounding.jl`: [`sounding_dynamic_inputs`](@ref),
  [`forcing_dynamic_inputs`](@ref)), what a run reading its CSV builds in
  memory and what a sounding driver holds, and the in-run writer of a
  parent's child boundaries (`input_drivers/recorder.jl`:
  [`NestBoundaryRecorder`](@ref)).

Everything that **creates** a driver from outside data (OpenStreetMap and
GeoJSON, land-use classification, soundings, ERA5/GRIB, downloads) is the
separate package AeolusPrep.jl (ONE-h, #457), which depends on Aeolus for
these schemas and writers.

A file whose grid differs from the run's is an error naming both grids:
there is no resampling inside the model; the preprocessor is re-run for
another grid (`aeolusprep static <config.toml>`).

There is no PALM (PIDS) compatibility goal: the variable names are
Aeolus's own, short `snake_case` with CF attributes.
"""
module InputDrivers

using Dates: Dates, now, UTC
using ..Grids: xnodes, ynodes, znodes, CellCenter
using ..IBM: SolidMask
using ..BoundaryConditions: FieldBoundaryConditions, neighbor_index, WindowedBoundaryData
import ..BoundaryConditions: snapshot_values, blend_source
import NCDatasets
using ..NCIO: NCDataset, nc_create, nc_defvar!, nc_append_record!
using ..MoistThermodynamics: DEFAULT_CONSTANTS, exner_function, virtual_temperature, PhasePartition
using ..LargeScaleForcing: ForcingTable, forcing_value, Sounding, sounding_profiles, AdvectiveTendencies,
                           Subsidence, ProfileNudging
using ..IO: fnv1a64, git_state

export latlon_to_meters, meters_to_latlon, DriverVariable, DriverSchema, check_driver, schema_markdown,
       format_reference, ncdump_header,
       DRIVER_ATTRIBUTES, DRIVER_SCHEMA_VERSION, file_fnv1a64,
       SURFACE_STREET, SURFACE_ROOF, SURFACE_PARK, SURFACE_WATER, SURFACE_WALL, SURFACE_CLASSES,
       Building, UrbanCity, polygon_bbox, point_in_polygon, rasterize_city, static_inputs, solid_and_classes,
       surface_class_map, tag_building_adjacent!,
       STATIC_SCHEMA, STATIC_TILES_SCHEMA, StaticInputs, StaticTiles,
       write_static_driver, read_static_driver, write_static_tiles, read_static_tiles,
       DYNAMIC_SCHEMA, LS_TABLES, DynamicInputs, read_dynamic_driver, write_dynamic_driver,
       FileSnapshot, SlabArray, FileBoundaryData, FileBoundaryFeed, feed_window!, feed_steady!,
       advance_feed!, feed_coverage, check_coverage, check_planes, shift_table,
       dynamic_forcing, init_time_string, init_day_seconds, surface_pressure_at,
       sounding_dynamic_inputs, forcing_dynamic_inputs, NestBoundaryRecorder, record_boundary!, plane_snapshot

include("input_drivers/conventions.jl")
include("input_drivers/city.jl")
include("input_drivers/static.jl")
include("input_drivers/dynamic.jl")
include("input_drivers/sounding.jl")
include("input_drivers/recorder.jl")

end # module
