# Input drivers

`Aeolus.InputDrivers` (CFG-f, #361) holds Aeolus's own **input drivers**:
NetCDF-4 files with what a run reads from outside the model. The first is
the **static driver**: terrain, buildings, surface classes and leaf area
on the run's own horizontal grid, plus the building and patch polygons.
The second is the **dynamic driver** (CFG-g, #362): the initial state, the
boundary planes of the open faces over time, and the large-scale and
surface forcing. There is no PALM (PIDS) compatibility goal: the names
are Aeolus's own, with CF attributes.

## Who writes a driver

Aeolus keeps the **readers, the writers and the format schemas** (this
module). Everything that **creates** a driver from outside data is the
separate package [AeolusPrep.jl](https://github.com/badulion/AeolusPrep.jl)
(ONE-h, #457; the single-model design, [One model](../design/single_model.md)): OpenStreetMap
and GeoJSON footprints, the nest parent's land-use classification,
soundings, ERA5 (GRIB2 or NetCDF), and the downloads of
those sources. It depends on Aeolus for the writers, so a file is written
by the code that defines and checks its format, and the model itself
never reads OpenStreetMap, GIS or reanalysis files (and does not depend
on GRIBDatasets).

```bash
# in an AeolusPrep.jl checkout
julia --project -m AeolusPrep static <aeolus>/examples/urban/berlin_summer.toml \
      --source data/berlin/alexanderplatz_real.osm --min-height 8 --park-lai 2.5
julia --project -m AeolusPrep dynamic --from-sounding <aeolus>/examples/urban/berlin_summer.toml
julia --project -m AeolusPrep dynamic --from-era5 data/era5/era5_lindenberg_20190629_*.nc \
      --config <aeolus>/examples/lindenberg_era5/lindenberg_era5.toml --lat 52.2085 --lon 14.118
julia --project -m AeolusPrep regenerate <aeolus> --check     # every committed driver, reproduced?
```

`aeolusprep static` reads the run's grid and anchor from the Aeolus
config and the source and its options (`--source`, `--min-height`,
`--park-lai`) from the command line; for an urban site nested in a parent
`[[domain]]` with a tiled surface it also writes that domain's
`static_driver` from the land-use table `--landuse` (the site's own
classes under its footprint, NEST-p3 #483), and for a tiled domain alone
(the nest parent alone) its tiles from `--landuse`. Every driver
committed to Aeolus (`examples/**/input/`, `test/data/**/input/`) is listed
with its command and options in AeolusPrep's `recipes.toml`, and its tests
regenerate each one: every data variable bit for bit, every global
attribute but `date_created` and `aeolus_commit` equal. The exceptions
are the case-table drivers of the SBL configs (`examples/sbl/input/`),
written once by the former sbl run type's writer and committed here
since ONE-g (#456) removed it; AeolusPrep lists them as not made by it,
and `test/input_dynamic.jl` checks them against the case tables. A
source recorded relative to its driver must exist (`test/input_static.jl`):
[`check_driver`](@ref Aeolus.InputDrivers.check_driver) skips the
staleness check of a source it cannot find, which a moved driver would
otherwise switch off silently (#487). The parent-made driver of an
offline nest is model output, not a preprocessor (below).

**An Aeolus change that AeolusPrep follows runs its tests.** AeolusPrep
calls the config (`load_run_config`, `[run] legacy`, `[urban] enabled`,
the domain's `static_driver` and `dynamic_driver`), this module's writers
and schemas, the urban site's options and writers
(`Urban.site_options`, `write_sounding_driver`, the nest parent's tiles),
the simulation's mesh and open faces, and the layout of
`examples/<case>/`. A PR that changes any of these, or a committed
driver, runs `julia bin/test-against <this checkout>` in an AeolusPrep.jl
checkout and quotes the result ([Contributing](https://github.com/badulion/Aeolus.jl/blob/master/docs/CONTRIBUTING.md#aeolusprep)).

## Conventions

Shared with the dynamic driver (CFG-g). Every driver carries the global
attributes of [`DRIVER_ATTRIBUTES`](@ref Aeolus.InputDrivers.DRIVER_ATTRIBUTES):
`aeolus_driver` (`"static"`), `aeolus_form`, `aeolus_schema_version`
(1), the anchor `origin_lat`, `origin_lon` and where it lies in the run's
mesh coordinates (`origin_x`, `origin_y`), `projection =
"equirectangular"` ([`latlon_to_meters`](@ref Aeolus.InputDrivers.latlon_to_meters),
the one implementation; sub-metre over the ≤ 5 km of an LES domain), the
grid spacing `dx`, `dy`, and the provenance: `source` (relative to the
driver's directory when the source lies in the same package; in another
package `"<Package>:<path in it>"`, e.g.
`"AeolusPrep:data/berlin/alexanderplatz_real.osm"`, which does not depend
on where the packages are checked out), `source_fnv1a64`, `preprocessor`
(e.g. `"aeolusprep static"`), `aeolus_commit` and `date_created`. Variables have CF `units`, `long_name` and, where CF
has one, `standard_name`.

**No resampling.** [`read_static_driver`](@ref Aeolus.InputDrivers.read_static_driver)
errors, naming both grids, if the file's `x`, `y`, `dx`, `dy` or anchor
differ from the run's. The preprocessor is re-run for another grid
(`aeolusprep static`).

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
julia --project -m AeolusPrep dynamic --from-sounding <aeolus>/examples/urban/berlin_summer.toml
julia --project -m Aeolus run examples/urban/berlin_smoke.toml --set 'urban.child_boundaries="child.nc"'
```

One schema,
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

**Initial state.** Profiles over `init_z`,
or 3-D fields on the run grid (`init_u3d`, …: the interior cells in
storage order, u at the cell's west face, v at its south face, w at its
bottom face), and `init_p_surface`. An ERA5 driver (below) fills
the `init_*` profiles and the surface series, and the `simulation`
driver's `[initial] dynamic` reads the profiles; no case reads the
surface series yet. A sounding driver has no `init_*` profiles: the Berlin case reads its
`ls_profile_*` tables and `init_p_surface` and evaluates the tables at
its start time itself (the bilinear (height, time) interpolation between
the two launches that bracket the start), which is bitwise the CSV path
(`test/input_dynamic.jl`).

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

**Initial soil** (#521). `init_soil_t` (K) and `init_soil_m` (m³ m⁻³) are PALM's PIDS variables, on the coordinate `zsoil`. That coordinate is the levels' centres in m below the surface: positive down and strictly increasing.

- **Shape.** Each variable is a profile over `zsoil`, or a volume over (`x`, `y`, `zsoil`) on the run's horizontal grid. The schema's `alt_dims` lets a variable take either shape.
- **Fill values.** `NaN` (the `_FillValue`) marks a level or a column without data.
- **The writer.** AeolusPrep.jl's ERA5 soil writes them from ERA5's four layers (`stl1–4`, `swvl1–4`; centres 0.035, 0.175, 0.64 and 1.945 m), appended to the sounding driver. Columns that are not land are NaN.
- **What reads them.** The urban site's ground starts from them (`[ground] soil_init`, [`GroundSoil`](@ref Aeolus.Urban.GroundSoil)), and so do the nest parent's soil tiles.

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

### Sounding and case-table drivers

[`sounding_dynamic_inputs`](@ref Aeolus.InputDrivers.sounding_dynamic_inputs)
holds a radiosonde sounding's height-time tables (`ls_profile_theta`,
`_qv`, `_u`, `_v`, `_p`, the union of the launches' level heights up to
the table top) on the "seconds since init_time" axis and the ground
pressure reduced from the launch site ([`surface_pressure_at`](@ref Aeolus.InputDrivers.surface_pressure_at)).
[`forcing_dynamic_inputs`](@ref Aeolus.InputDrivers.forcing_dynamic_inputs)
holds the `ForcingTable`s of a case's forcing (GABLS, CASES-99); a number
stays in the case's config. The committed drivers of the SBL
configurations (`examples/sbl/input/*_dynamic.nc`: GABLS2 and
CASES-99 the surface θ series and the subsidence, GABLS3 the geostrophic
wind, the advective tendencies and the 0.25 m θ series, GABLS4 the surface
θ series) were written from the case tables by the former `sbl` run type
(ONE-g, #456). The model reads the large-scale tables generically: the
`[large_scale_forcing]` component (`u_g = "driver"`, `advective_tendencies`,
`subsidence`, `nudging`; [`dynamic_forcing`](@ref Aeolus.InputDrivers.dynamic_forcing))
and the `[surface] model = "most"` `temperature = {kind = "driver"}`, the times read
at the model time minus `[time] start`; a driver with no initial profiles
initialises nothing (the profiles stay in `[initial]`).

### The parent-made driver

[`NestBoundaryRecorder`](@ref Aeolus.InputDrivers.NestBoundaryRecorder)
is an in-run writer (model output: `[output] child_boundaries`): at every parent step it appends one record, the
values every child plane reads from the prolongated parent state
(exactly the points `blend_window!` reads online), the parent time, the
step the parent took (`time_bc_step`) and the parent's mean profiles over
the child; it also writes the child's 3-D initial wind. Recording every
parent step, rather than extracting from 3-D output at output cadence, is
what makes the offline child bitwise the online one: both blend the same
Float64 snapshots over the same windows.

### ERA5 drivers

AeolusPrep's ERA5 preprocessor (CFG-h #363, shared with #57)
reads ERA5 as the CDS delivers it, GRIB2 or NetCDF, on pressure or model
levels, and writes the `init_*` profiles and `init_p_surface`, the
boundary planes of the run's open faces (one `time_bc` record per ERA5
time), the `ls_profile_*` tables, `ls_u_g`/`ls_v_g` (centred differences of
Φ over ±`box`), `ls_w_subs` (from ω), `ls_theta_adv`/`ls_qv_adv` and the
surface series; heights `Φ/g₀` above the ERA5 orography, θ from
`MoistThermodynamics.potential_temperature`, bilinear in lat/lon about
the anchor ([`meters_to_latlon`](@ref Aeolus.InputDrivers.meters_to_latlon),
the inverse of [`latlon_to_meters`](@ref Aeolus.InputDrivers.latlon_to_meters)),
linear in z. Every ERA5 file is listed with its FNV-1a hash in the
global attribute `era5_files`. Its tests (analytic hydrostatic
atmosphere, GRIB2 = NetCDF bit for bit, the Lindenberg sample against the
Lindenberg sounding: θ RMSE 0.787 K, qv RMSE 0.718 g/kg over 200–3000 m)
moved with it.

**The `simulation` driver** takes such a driver as `[initial] dynamic`:
its profiles set the initial u, v, θ (and qv), and lateral faces set to
`boundaries.x`/`y = "open"` take its boundary planes through a
[`FileBoundaryFeed`](@ref Aeolus.InputDrivers.FileBoundaryFeed), blended
linearly in time (boussinesq core, uniform grid, z walls).
`examples/lindenberg_era5/lindenberg_era5.toml` is a 16³ open box over
Lindenberg reading the committed `input/lindenberg_era5_dynamic.nc`
(1.4 MB; ERA5 of 2019-06-29 00–12 UTC at 52.2085 N, 14.118 E); the file is
hashed under `[inputs]` of `run.toml`, and 20 steps round-trip bit for bit
(`test/input_dynamic.jl`).

## The Berlin case

`examples/urban/berlin_<s>.toml` name their drivers in
each domain's `static_driver` (the site's and its parent's, relative to the config, hashed
under `[inputs]` of `run.toml`); the generated files are committed in
`examples/urban/input/` (all of them together 790 KB; each is
under the 2 MB limit for committed drivers) with their source hashes.
A missing file is a configuration error before any run folder is made,
and it prints the `aeolusprep static <config>` command that makes it.
There is no in-memory ground: a grid no file was made for needs its
driver made first (`aeolusprep static <config> --set grid.size=…
--out …`). The Mitte sample at the smoke size of
its `[test]` overrides and the synthetic city of `test/urban_run.jl` have
committed drivers of their own.

Reading the driver is bit for bit the OSM-direct path it replaced
(`test/input_static.jl`, against digests captured before the switch: the
smoke case's solid mask, surface classes, heightmap and view factors, the
summer rasterisation, the nest parents' land use, and the 12-step smoke
state in `test/config_berlin.jl`); the Mitte sample at 32 × 32 × 16 is
its driver's polygons rasterised there. The two known defects of the OSM
reader (the `&&`/`||` precedence of the node parser, now in AeolusPrep;
patches drawn over roofs, the rasterisation here) are kept and
documented; fixing them is #141's.

The dynamic driver (CFG-g, #362): the domain's `dynamic_driver` names it (`[initial] dynamic` of the `urban` run type before ONE-e #454).

* **Sounding.** `berlin_summer.toml` reads its initial state from
  `input/berlin_20190629_dynamic.nc` (`aeolusprep dynamic
  --from-sounding`, from the Lindenberg CSV, which stays as
  `[urban] sounding`, the driver's hashed source). The tables are the
  ones the CSV path builds, so the initial θ(z), qv(z), ground pressure
  (101841.78 Pa), reference state, the parent's geostrophic wind and z_i
  are bitwise those of master's CSV path (digests captured before the
  switch). With `dynamic_driver = "none"` the CSV is preprocessed in memory by
  the same functions. The driver holds the whole sounding series: a case
  starting at another time (`[urban] start_h`) reads the same tables with
  the time axis moved and its ground pressure recomputed at its start,
  bitwise the CSV read in memory.
  The smoke and winter scenarios keep the idealised profile
  (`sounding = dynamic_driver = "none"`).
* **Offline nest.** An online run with `[urban] child_boundaries =
  "<file>"` writes the parent's
  windows, the child's boundary slabs, its initial wind and the parent's
  mean profiles. A child run with `dynamic_driver = "<file>"` and
  `[nesting] parent_driver = "offline"` replays them
  without the parent: the same windows (the child steps land on the
  parent times), the same face data and targets; its own initial θ, qv
  and reference state come from `[urban] sounding` as online. Measured
  on the smoke case: 12 steps, bitwise the online one-way child in every
  child field. The online nest stays the default; `offline` is for a
  parent run once and reused. Two-way and `offline` exclude each other.

## Reference

```@autodocs
Modules = [Aeolus.InputDrivers]
```
