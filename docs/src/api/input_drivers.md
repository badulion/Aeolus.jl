# Input drivers

`Aeolus.InputDrivers` (CFG-f, #361) holds Aeolus's own **input drivers**:
NetCDF-4 files with what a run reads from outside the model. The first is
the **static driver**: terrain, buildings, surface classes and leaf area
on the run's own horizontal grid, plus the building and patch polygons,
so a run reads its geometry from one file instead of re-parsing
OpenStreetMap at every start. The OSM and GeoJSON readers are its
preprocessors. There is no PALM (PIDS) compatibility goal: the names are
Aeolus's own, with CF attributes.

```bash
julia --project -m Aeolus preprocess static examples/urban/configs/berlin_summer.toml
julia --project -m Aeolus preprocess static my_case.toml --out input/my_static.nc
```

`aeolus preprocess static` loads the run config and calls its driver's
[`preprocess_static`](@ref Aeolus.Config.preprocess_static), which reads
the grid and the sources from the config and writes the file(s). The
`berlin` driver writes `[geometry] static` (from `[preprocess]
osm_path`, `min_height`, `park_lai`) and, for the nest, `[geometry]
parent_static` (from `[preprocess] landuse`).

## Conventions

Shared with the dynamic driver (CFG-g). Every driver carries the global
attributes of [`DRIVER_ATTRIBUTES`](@ref Aeolus.InputDrivers.DRIVER_ATTRIBUTES):
`aeolus_driver` (`"static"`), `aeolus_form`, `aeolus_schema_version`
(1), the anchor `origin_lat`, `origin_lon` and where it lies in the run's
mesh coordinates (`origin_x`, `origin_y`), `projection =
"equirectangular"` ([`latlon_to_meters`](@ref Aeolus.InputDrivers.latlon_to_meters),
the one implementation; sub-metre over the ≤ 5 km of an LES domain), the
grid spacing `dx`, `dy`, and the provenance: `source` (relative to the
driver's directory), `source_fnv1a64`, `preprocessor`, `aeolus_commit`
and `date_created`. Variables have CF `units`, `long_name` and, where CF
has one, `standard_name`.

**No resampling.** [`read_static_driver`](@ref Aeolus.InputDrivers.read_static_driver)
errors, naming both grids, if the file's `x`, `y`, `dx`, `dy` or anchor
differ from the run's. The preprocessor is re-run for another grid.

**Checks.** A [`DriverSchema`](@ref Aeolus.InputDrivers.DriverSchema) is
a declarative table of variables (name, dims, type, units, required,
valid range). [`check_driver`](@ref Aeolus.InputDrivers.check_driver)
returns every problem of a file against it: wrong kind or form, a schema
version other than 1, a missing attribute or required variable, wrong
dimensions, type or units, values out of range, and a `source_fnv1a64`
that no longer matches the source file (a stale driver). The tables below
are generated from the schemas.

## The static driver, resolved form

Dimensions `x`, `y` are the cell centres of the run's horizontal grid.
The polygons are CF contiguous ragged arrays: `building_nvertex` counts
each building's vertices in `building_x`, `building_y`; the rings of a
polygon with holes (a courtyard, a multipolygon relation) are each
closed and followed by a `NaN` vertex, the convention of
[`point_in_polygon`](@ref Aeolus.InputDrivers.point_in_polygon). The
surface-class code table ([`SURFACE_CLASSES`](@ref Aeolus.InputDrivers.SURFACE_CLASSES))
is stored as the CF `flag_values` and `flag_meanings` of `surface_class`.

```@example
using Aeolus # hide
print(Aeolus.InputDrivers.schema_markdown(Aeolus.InputDrivers.STATIC_SCHEMA)) # hide
```

The rasters do not depend on the run's lateral boundaries: a run applies
its building-free frame (the columns next to an open face take
`ground_class` and no building) and its wall tagging
([`solid_and_classes`](@ref Aeolus.InputDrivers.solid_and_classes)), so
one file serves a periodic and a nested run on the same grid. Terrain is
written as `zt = 0`: no preprocessor reads elevation yet, and the Berlin
case refuses a driver with terrain.

## The static driver, tiles form

For a grid that does not resolve buildings (the Berlin nest parent):

```@example
using Aeolus # hide
print(Aeolus.InputDrivers.schema_markdown(Aeolus.InputDrivers.STATIC_TILES_SCHEMA)) # hide
```

The tile fraction of class `c` is `class_count[:, :, c] / refinement²`
(`refinement`, an attribute of `class_count`, is the number of fine cells
per column per axis); the global attributes `tile_x0`, `tile_y0` are the
grid's south-west corner (m from the anchor) and `source_notes` carries
the source's provenance lines.

## The Berlin case

`examples/urban/configs/berlin_<s>.toml` name their drivers in
`[geometry] static` and `parent_static` (relative to the config, hashed
under `[inputs]` of `run.toml`); the generated files are committed in
`examples/urban/configs/input/` (all of them together 790 KB; each is
under the 2 MB limit for committed drivers) with their source hashes.
A missing file is a configuration error before any run folder is made,
and it prints the exact `aeolus preprocess static <config>` command.
`static = "none"` preprocesses `[preprocess] osm_path` in memory with the
same functions, the way to run a grid no file was made for (Julia callers
that change `size`, `extent` or `offset` pass `static = nothing`, and
`parent_static = nothing` for a different parent box).

Reading the driver is bit for bit the OSM-direct path it replaced
(`test/input_static.jl`, against digests captured before the switch: the
smoke case's solid mask, surface classes, heightmap and view factors, the
summer rasterisation, the nest parents' land use, and the 12-step smoke
state in `test/config_berlin.jl`). The two known defects of the OSM
reader (the `&&`/`||` precedence of the node parser; patches drawn over
roofs) are kept and documented; fixing them is #141's.

## Reference

```@autodocs
Modules = [Aeolus.InputDrivers]
```
