# Input drivers

`Aeolus.InputDrivers` (CFG-f, #361) holds Aeolus's own **input drivers**:
NetCDF-4 files with what a run reads from outside the model. The first is
the **static driver**: terrain, buildings, surface classes and leaf area
on the run's own horizontal grid, plus the building and patch polygons,
so a run reads its geometry from one file instead of re-parsing
OpenStreetMap at every start. The OSM and GeoJSON readers are its
preprocessors. The second is the **dynamic driver** (CFG-g, #362): the
initial state, the boundary planes of the open faces over time, and the
large-scale and surface forcing; its preprocessors are a sounding /
case-table reader and an Aeolus parent run. There is no PALM (PIDS)
compatibility goal: the names are Aeolus's own, with CF attributes.

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

```@eval
using Aeolus, Markdown
Markdown.parse(Aeolus.InputDrivers.schema_markdown(Aeolus.InputDrivers.STATIC_SCHEMA))
```

The rasters do not depend on the run's lateral boundaries: a run applies
its building-free frame (the columns next to an open face take
`ground_class` and no building) and its wall tagging
([`solid_and_classes`](@ref Aeolus.InputDrivers.solid_and_classes)), so
one file serves a periodic and a nested run on the same grid. Terrain is
written as `zt = 0`: no preprocessor reads elevation yet, and the Berlin
case refuses a driver with terrain. There are no `vegetation_type`,
`soil_type` or `water_type` rasters yet: no preprocessor has a source
for them and no run reads them, so they would be written and never
used. Schema version 2 adds them with the first reader that needs them.
`ground_class` (the surface class with the buildings removed) is stored
instead: the building-free frame of a nested run and the leaf area
under a removed building need it.

## The static driver, tiles form

For a grid that does not resolve buildings (the Berlin nest parent):

```@eval
using Aeolus, Markdown
Markdown.parse(Aeolus.InputDrivers.schema_markdown(Aeolus.InputDrivers.STATIC_TILES_SCHEMA))
```

The tile fraction of class `c` is `class_count[:, :, c] / refinement²`
(`refinement`, an attribute of `class_count`, is the number of fine cells
per column per axis); the global attributes `tile_x0`, `tile_y0` are the
grid's south-west corner (m from the anchor) and `source_notes` carries
the source's provenance lines.

## The dynamic driver

```bash
julia --project -m Aeolus preprocess dynamic --from-sounding examples/urban/configs/berlin_summer.toml
julia --project -m Aeolus preprocess dynamic --from-sounding examples/sbl/configs/gabls3.toml --out gabls3_dynamic.nc
julia --project -m Aeolus preprocess dynamic --from-parent examples/urban/configs/berlin_smoke.toml --out child.nc
```

`aeolus preprocess dynamic` calls the driver's
[`preprocess_dynamic`](@ref Aeolus.Config.preprocess_dynamic). One schema,
`aeolus_driver = "dynamic"`, `aeolus_form = "standard"`, covers three
groups; each is optional, and [`read_dynamic_driver`](@ref Aeolus.InputDrivers.read_dynamic_driver)
checks that what a file holds is consistent:

```@eval
using Aeolus, Markdown
Markdown.parse(Aeolus.InputDrivers.schema_markdown(Aeolus.InputDrivers.DYNAMIC_SCHEMA))
```

**Times.** Every time coordinate is in seconds since the global attribute
`init_time` (ISO-8601 UTC, the model start; `""` for a case without a
calendar date, the SBL configs), with CF `units = "seconds since …"`.

**Initial state.** Profiles over `init_z` (piecewise linear in height),
or 3-D fields on the run grid (`init_u3d`, …: the interior cells in
storage order, u at the cell's west face, v at its south face, w at its
bottom face), and `init_p_surface`. A driver without `init_*` profiles
starts from its large-scale profile tables at `t = 0`
([`initial_profile`](@ref Aeolus.InputDrivers.initial_profile)): this is
how a sounding driver gives the bilinear (height, time) interpolation
between the two launches that bracket the start, the same arithmetic as
the CSV path, bit for bit.

**Large-scale forcing.** Each `ls_<name>` table has its own height and
time coordinates (`ls_<name>_z`, `ls_<name>_time`, or the shared
`time_bc` of a parent-made driver; the schema's `*` dimensions). No table
is resampled onto a common grid, so a sounding's or a case's
`ForcingTable` round-trips value for value. [`dynamic_forcing`](@ref Aeolus.InputDrivers.dynamic_forcing)
maps the tables onto `LargeScaleForcing`: `ls_u_g`/`ls_v_g` the
geostrophic wind, `ls_*_adv` [`AdvectiveTendencies`](@ref Aeolus.LargeScaleForcing.AdvectiveTendencies),
`ls_w_subs` [`Subsidence`](@ref Aeolus.LargeScaleForcing.Subsidence)
(global attributes `subsidence_fields`, `subsidence_form`), `ls_nudge_*`
[`ProfileNudging`](@ref Aeolus.LargeScaleForcing.ProfileNudging) (global
attribute `nudging_tau`). `ls_profile_*` are the large-scale mean
profiles: a sounding's tables, or a nest parent's mean over the child
(the sponge and θ-relaxation targets). The surface series are
`sfc_temperature` and `sfc_pressure` over `time_sfc`.

**Boundary planes.** For each open face and each field with data,
`bc_<face>_<var>` over (`…_i`, `…_j`, `…_k`, `time_bc`) holds the
child's **buffer slab**: the storage points of the
[`OpenPlane`](@ref Aeolus.BoundaryConditions.OpenPlane), the face plane
plus the `H` halo layers (attributes `field`, `side`, `i0`, `j0`, `k0`,
`halo_mode`). Slabs, not 3-D fields: a record of the 16³ smoke child
holds the five faces only. [`FileBoundaryData`](@ref Aeolus.InputDrivers.FileBoundaryData)
is the generic two-snapshot [`WindowedBoundaryData`](@ref Aeolus.BoundaryConditions.WindowedBoundaryData)
with a [`FileSnapshot`](@ref Aeolus.InputDrivers.FileSnapshot), one
`time_bc` record whose `snapshot_values` is a slab-backed 3-D view
([`SlabArray`](@ref Aeolus.InputDrivers.SlabArray)): the time blend is
the existing `blend_window!`, the one kernel of the nest, with no second
interpolation. A [`FileBoundaryFeed`](@ref Aeolus.InputDrivers.FileBoundaryFeed)
holds two records and moves the window ([`feed_window!`](@ref Aeolus.InputDrivers.feed_window!),
[`advance_feed!`](@ref Aeolus.InputDrivers.advance_feed!)), reading the one
record it does not hold. Measured (`test/input_dynamic.jl`): over 1000
records `Base.summarysize` of the feed stays at two records plus the time
axes; data linear in time, `u = 1 + 0.01 t` at `time_bc = 0, 60, 120 s`,
is reproduced at t = 17, 60, 95 s to within 1 ulp; data quadratic in time
has a blend error of exactly the bound `Δt²/8 · max|u''|` at the window
midpoints. A face the run has open but the file lacks, a slab at other
storage indices (another grid or halo), a run interval the records do not
cover ([`check_coverage`](@ref Aeolus.InputDrivers.check_coverage), both
intervals printed) and a file on another grid are errors.

### The sounding and case-table preprocessor

[`sounding_dynamic_inputs`](@ref Aeolus.InputDrivers.sounding_dynamic_inputs)
writes a radiosonde sounding's height-time tables (`ls_profile_theta`,
`_qv`, `_u`, `_v`, `_p`, the union of the launches' level heights up to
the table top) on the "seconds since init_time" axis and the ground
pressure reduced from the launch site ([`surface_pressure_at`](@ref Aeolus.InputDrivers.surface_pressure_at)).
[`forcing_dynamic_inputs`](@ref Aeolus.InputDrivers.forcing_dynamic_inputs)
writes the `ForcingTable`s of a case's forcing (GABLS, CASES-99); a
number or a formula stays in the case's config. The `sbl` driver's
`[case] dynamic` replaces the config's geostrophic wind, extra forcings
and surface θ series by those the driver holds (its initial profiles,
which carry formula rules below the lowest table level, stay in
`[initial]`); GABLS3 run 50 steps from its driver is bitwise the CSV path.

### The Aeolus-parent preprocessor

[`NestBoundaryRecorder`](@ref Aeolus.InputDrivers.NestBoundaryRecorder)
is an in-run writer: at every parent step it appends one record, the
values every child plane reads from the prolongated parent state
(exactly the points `blend_window!` reads online), the parent time, the
step the parent took (`time_bc_step`) and the parent's mean profiles over
the child; it also writes the child's 3-D initial wind. Recording every
parent step, rather than extracting from 3-D output at output cadence, is
what makes the offline child bitwise the online one: both blend the same
Float64 snapshots over the same windows.

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

The dynamic driver (CFG-g, #362): `[initial] dynamic` names it.

* **Sounding.** `berlin_summer.toml` reads its initial state from
  `input/berlin_20190629_dynamic.nc` (`aeolus preprocess dynamic
  --from-sounding`, from the Lindenberg CSV, which stays as
  `[initial] sounding`, the driver's hashed source). The tables are the
  ones the CSV path builds, so the initial θ(z), qv(z), ground pressure
  (101841.78 Pa), reference state, the parent's geostrophic wind and z_i
  are bitwise those of master's CSV path (digests captured before the
  switch). With `dynamic = "none"` the CSV is preprocessed in memory by
  the same functions. The driver's `init_time` must be the case start.
  The smoke and winter scenarios keep the idealised profile
  (`sounding = dynamic = "none"`).
* **Offline nest.** An online run with `[output] child_boundaries =
  "<file>"` (or `aeolus preprocess dynamic --from-parent <config>`, which
  runs the config's online nested case for its steps) writes the parent's
  windows, the child's boundary slabs, its initial wind and the parent's
  mean profiles. A child run with `[initial] dynamic = "<file>"` and
  `[physics.options.nesting] parent_driver = "offline"` replays them
  without the parent: the same windows (the child steps land on the
  parent times), the same face data and targets; its own initial θ, qv
  and reference state come from `[initial] sounding` as online. Measured
  on the smoke case: 12 steps, bitwise the online one-way child in every
  child field. The online nest stays the default; `offline` is for a
  parent run once and reused. Two-way and `offline` exclude each other.

## Reference

```@autodocs
Modules = [Aeolus.InputDrivers]
```
