# Run configuration

Every Aeolus run is described by one self-contained TOML file (schema
version 2, ONE-b #451; the target is the single model of
[One model](single_model.md)). The file is checked against the model's
schema, resolved into a typed [`RunConfig`](@ref Aeolus.Config.RunConfig),
and written back in canonical form to the run folder's `run.toml`. A run
folder can therefore always be run again from its own `run.toml`. This page
is the guide. The API is on [Run configuration (API)](../api/config.md),
every option is in the generated [option reference](../reference/options/model.md),
and the NetCDF input files are specified in
[Static input driver](../reference/input_static.md) and
[Dynamic input driver](../reference/input_dynamic.md).

## TOML layout

There is one model and no run types: `aeolus run config.toml` runs it. A
file sets options section by section; a key that is not set takes the
schema default. A key the schema does not know is an error, and so is a
value of the wrong type or outside its range; every problem is reported
together, naming the key and the file it came from.

```toml
schema_version = 2
dynamic_driver = "input/lindenberg_era5_dynamic.nc"   # the domain's input drivers (optional)

[run]
name = "dcbl"
seed = 1

[time]                 # run level: one clock for every domain
scheme = "rk3"
dt = "adaptive"        # or a fixed step in seconds
cfl = 0.5
dt_max = 10.0
duration = 3600.0      # or nsteps, with duration = "none"

[grid]                 # per domain
size = [32, 32, 32]
extent = [3200.0, 3200.0, 1600.0]

[sgs]                  # one section per component; the :sgs slot
closure = "smagorinsky"

[surface_theta_flux]   # a component is enabled by its own section
bottom = 0.24
```

**A bare config is a minimal dry LES**: an empty file runs 32³ cells on
3.2 km × 3.2 km × 1.6 km, periodic in x and y between free-slip walls, the
Boussinesq core with buoyancy on θ₀ = 300 K, Smagorinsky SGS and adaptive
SSP-RK3 steps for one hour. Every other component is off until its section
enables it.

The sections:

| level | sections |
| :-- | :-- |
| run | `[run]`, `[time]`, `[output]`, `[checks]`, `[checkpoint]`, `[fftw]`, `[logging]` |
| per domain | `[grid]`, `[dynamics]` (the core), `[boundaries]`, `[initial]`, and one section per component, generated from its schema: `[advection]`, `[sgs]`, `[coriolis]`, `[sponge]`, `[surface_theta_flux]`; the EX-d couplings `[forcing]`, `[surface]`, `[microphysics]`, `[tracer]`, `[particles]`, `[obstacles]` (one component today, split in ONE-e) |
| the domain itself | `id`, `static_driver`, `dynamic_driver` (top level for one domain, else in each `[[domain]]` block) |

`[checks]` takes one key per check the enabled components publish; none
publishes one yet (ONE-d, #453), so any key is an error. The component
interface and the stage pipeline are on [Components](components.md).

Conventions: keys are ASCII `snake_case`; values are SI and durations are
in seconds; `"none"` stands for nothing (TOML has no null); a path is
relative to the file that sets it and is stored absolute. Environment
variables only fill a few defaults (`AEOLUS_LOG`, `AEOLUS_FFTW_PLANNER`,
`AEOLUS_FFTW_THREADS`, `AEOLUS_ALLOW_UNCOMMITTED`). The resolved value is
always written, so the environment never changes a replay.

## Domains

A config without `[[domain]]` has one domain, `"main"`. Several domains are
`[[domain]]` blocks; the run-level per-domain sections apply to every
domain, and a block's `[domain.<section>]` tables replace keys for its
domain:

```toml
[sgs]
closure = "vreman"         # every domain …

[[domain]]
id = "outer"

[[domain]]
id = "inner"
static_driver = "input/inner_static.nc"
  [domain.sgs]
  closure = "amd"          # … but this one
  [domain.grid]
  extent = [400.0, 400.0, 200.0]
```

The resolved config, and so `run.toml`, lists every domain with every
option of every component (`[[domain]]` then `[domain.grid]`,
`[domain.sgs]`, …), so a run is fully described by its record. A problem
in a block is reported under `domain[i].<section>.<key>`; the cross-field
checks run on each domain. The model runs one domain until nesting by
domains (ONE-f, #455): a config with several validates, and `aeolus run`
refuses it with that reason.

The enabled components of a domain are validated before anything is
built: at most one per exclusive slot (`:advection`, `:sgs`,
`:microphysics`, `:radiation`, `:surface`), an `:advection` provider, and
each component's `requires` and `incompatible`; every problem names the
sections involved.

## Legacy cases

The run types not yet ported to components keep their own schemas until
the epic step that ports them: the urban case (`urban`, `urban_parent`;
ONE-e #454, ONE-f #455), `sbl` and the validation drivers (ONE-g #456), and
the example run types (ONE-i #458). A config selects one with `[run]
legacy = "<case>"`; it has no domains.

| `[run] legacy` | what it runs | options |
| :-- | :-- | :-- |
| `urban` | the urban LES run type (`Aeolus.UrbanCase`): the Berlin (Alexanderplatz) case with every coupling and the nest is its example | [urban](../reference/options/urban.md) |
| `urban_parent` | the nest parent of an urban case alone (same schema) | [urban_parent](../reference/options/urban_parent.md) |
| `sbl` | GABLS1–4, CASES-99 and the stratified channels | [sbl](../reference/options/sbl.md) |
| `dns` | the decaying DNS with its a-priori record | [dns](../reference/options/dns.md) |
| `column` | the single-column validation cases (DYCOMS RF01, ParisFog, parcel, Kessler box) | [column](../reference/options/column.md) |
| `channel`, `cbc_decay`, `patchy_surface`, `dns_decay` | the other validation drivers | [channel](../reference/options/channel.md), [cbc_decay](../reference/options/cbc_decay.md), [patchy_surface](../reference/options/patchy_surface.md), [dns_decay](../reference/options/dns_decay.md) |
| `advection`, `cbl`, `inflow_fetch`, `obstacle_channel`, `shear_layer`, `flat_plate`, `sgs_decay` | the example run types of `Aeolus.RunDrivers` | `aeolus options <case>` |

## No inheritance

There is no `extends`: every file is self-contained (maintainer decision
of 2026-10-09). Variants of a case (Berlin winter, smoke, the Mitte
sample, the parent) are complete files of their own; a change that must
reach several variants is made in each file, and the tests state the
differences between variants explicitly (`test/config_berlin.jl`: winter
differs from summer in exactly 11 keys). A file with `extends`, a file of
`schema_version = 1` and a `[run] driver` key are refused with what changed.

## `--set`

On the command line an override is `--set dotted.key=<TOML value>`.
Strings are quoted, arrays are TOML arrays:

```bash
julia --project -m Aeolus run examples/configs/taylor_green_vortex.toml --set grid.size=[16,16,16]
julia --project -m Aeolus run examples/urban/configs/berlin_smoke.toml --set physics.sgs='"smagorinsky"'
julia --project -m Aeolus check my_case.toml --set time.nsteps=10 --set time.duration='"none"'   # validate only
julia --project -m Aeolus resolve my_case.toml                            # print the canonical TOML
julia --project -m Aeolus options sgs                                     # the model's option reference, one section
julia --project -m Aeolus options urban physics                           # a legacy case's
```

An override of a per-domain section's key applies to every domain, over
the `[[domain]]` blocks. `bin/aeolus` is the same command on any Julia
version (`-m` needs Julia ≥ 1.12). Exit code 0 is success, 1 a failed run, 2 a usage or configuration
error raised before anything is written, 3 a failed required check of
`[checks]` (ONE-d, #453). A file with a `[sweep]` table
expands into one run per member; `--member k` picks one.

## The Julia escape hatch and `replayable`

Scripts can build or change a config in Julia:

```julia
using Aeolus
cfg  = load_run_config("examples/configs/dry_convective_bl.toml"; overrides = ["grid.size=[32,32,32]"])
cfg2 = with(cfg, "run.seed" => 7)          # a new, validated config
res  = run_config(cfg2)                    # what `aeolus run` does
```

`with` accepts any value the schema accepts, and also values TOML cannot
hold, such as a Julia function for a Berlin parent's surface flux. Such a
run is recorded with `replayable = false` and the keys in
`nonreplayable_keys`. `load_run_config` refuses to replay it, because its
`run.toml` does not describe it completely. Everything that has a TOML form
(constants, series, tables, named kinds) should use it.

## Running from `run.toml` versus `--resume`

| command | run ID | starts from |
| :-- | :-- | :-- |
| `aeolus run <folder>/run.toml` | a new run and a new folder | the start, with the old run's `[config]` |
| `aeolus run <folder> --resume` | the same run, a new session in its `run.toml` | the folder's newest checkpoint |

A replay from `run.toml` is bit-identical to the original under the
`ESTIMATE` FFTW planner and the same thread count (#245). `--resume`
refuses `--set`, because a resumed run must be the run it continues. For
the model and the `sbl` and `urban` legacy cases a run interrupted and
resumed is bitwise equal to the uninterrupted run (`test/config_roundtrip.jl`).

## The `[inputs]` hash check

Every input file option (a `Path`: a static or dynamic driver, a case
table) is recorded in `run.toml` under `[inputs]` with its `fnv1a64`
content hash; a domain's under `domain.<id>.<key>`. Replaying or resuming a run whose input files have changed
since is refused, naming each changed file, unless
`allow_changed_inputs = true` is passed to `load_run_config`. A replay
therefore cannot silently read different data.

## The preprocessors

Geometry and forcing come from NetCDF-4 driver files with Aeolus's own
schemas. They are written by the companion package AeolusPrep.jl (ONE-h,
#457), which reads the run's grid from its config; Aeolus keeps the
readers and the schemas:

```bash
# in an AeolusPrep.jl checkout
julia --project -m AeolusPrep static <aeolus>/examples/urban/configs/berlin_summer.toml --source data/berlin/alexanderplatz_real.osm
julia --project -m AeolusPrep dynamic --from-sounding <aeolus>/examples/sbl/configs/gabls3.toml --out gabls3_dynamic.nc
# the parent-made driver of an offline nest is model output
julia --project -m Aeolus run examples/urban/configs/berlin_smoke.toml --set 'output.child_boundaries="child.nc"'
```

The static driver (terrain, buildings, land use, trees, surfaces) is
written from OpenStreetMap or GeoJSON and read from `[geometry] static`.
The dynamic driver (initial profiles, large-scale forcing, surface
series, boundary planes) is written from a sounding, a case table or
ERA5, or by a recording parent run, and is read from the domain's
`dynamic_driver` (the model), `[case] dynamic` (sbl) or `[initial]
dynamic` (urban).
Their variables, dimensions, units and attributes are listed on the
[static](../reference/input_static.md) and
[dynamic](../reference/input_dynamic.md) pages; the API is
[Input drivers](../api/input_drivers.md).

## Where the shipped configs live

| directory | configs |
| :-- | :-- |
| `examples/configs/` | the model: `taylor_green_vortex`, `dry_convective_bl`, `buoyant_convection`, `lindenberg_era5`; `advection`: `scalar_advection_2d` |
| `examples/urban/configs/` | `urban` (in `src/`, `Aeolus.UrbanCase`): `berlin_summer`, `berlin_winter`, `berlin_smoke`, `berlin_mitte_sample` (each complete); `urban_parent`: `berlin_parent`, `berlin_parent_coarse`; `input/` holds their driver files |
| `examples/sbl/configs/` | `sbl` (in `src/`, `Aeolus.StableBoundaryLayer`): `gabls1`–`gabls4_exp3`, `cases99`, the channels, and the sweeps that replace the runner scripts (`*_closures`, `*_sweep`) |
| `examples/validation/configs/` | the validation drivers (`neutral_abl`, `dycoms_rf01`, `dns256`, …) |
| `examples/kernels/configs/` | `column`: `cloud_spinup`; the model: `tgv_energy_decay`, `decaying_hit`, `channel_flow`, `pollution_plume`, `particle_trail` |
| `examples/coupled/configs/` | the model with the EX-d couplings: `dcbl_radiation_lsm`, `moist_convection`, `heated_plume`, `flow_past_cube` (`input/`: the cube's GeoJSON and static driver) |
| `examples/sgs/configs/` | `cbl`: `cbl_tke`; `sgs_decay`: `rotating_stratified_decay`; `obstacle_channel`: `surface_mounted_cube`, `surface_mounted_cube_closures` (`input/`: the cube) |
| `examples/nesting/configs/` | `cbl`: `cbl_inflow_statistics`; `inflow_fetch`: `inflow_fetch` |
| `examples/transition/configs/` | `shear_layer`: `kelvin_helmholtz`; `flat_plate`: `flat_plate_t3a` |

An example is these files only: the `.jl` files next to a `configs/`
directory read a finished run folder and plot or tabulate it; none builds
or steps a model (EX-d, #441).

Each shipped config carries a `[test]` table with its smoke size:

```toml
[test]
overrides = ["grid.size=[16,16,16]", "time.duration=\"none\"", "time.nsteps=10"]
member = "lasd_32x32x32"     # sweeps only: the member replayed (default: the first)
```

The loader ignores `[test]` (it checks only its shape): it is not part of
the resolved config, of `run.toml` or of the config hash. `test/config_roundtrip.jl` finds every
`configs/*.toml` under `examples/` and checks that it validates, that
resolving it is a fixed point, that every key it uses is in the option
reference, and that its smoke run replays bit for bit from its `run.toml`.
A new config without a `[test]` table fails that suite with a message
naming the file.

## Comparison with other models

| model | run configuration | geometry and forcing inputs |
| :-- | :-- | :-- |
| PALM | Fortran namelist (`_p3d`) | NetCDF static and dynamic drivers (PIDS) made by preprocessors |
| ICON | Fortran namelists | grid file, EXTPAR external parameters, initial and boundary files |
| WRF | `namelist.input` | WPS writes `wrfinput` and `wrfbdy` |
| MicroHH | `.ini` file | `input.nc` |
| Oceananigans | a Julia script | built in Julia |
| Aeolus | TOML, schema-checked, with a Julia escape hatch; the resolved file is the run record | NetCDF-4 static and dynamic drivers with Aeolus's own schemas, made by AeolusPrep.jl |

The design follows PALM's split between a namelist and preprocessed driver
files, without aiming at PALM compatibility. What it adds is that the
config *is* the record: the canonical `run.toml` replays the run bit for
bit, and the option reference is generated from the same schema that
validates the file.
