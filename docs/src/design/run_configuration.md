# Run configuration

Every Aeolus run is described by one TOML file. The file is checked against
its driver's schema, resolved into a typed [`RunConfig`](@ref
Aeolus.Config.RunConfig), and written back in canonical form to the run
folder's `run.toml`. A run folder can therefore always be run again from its
own `run.toml`. This page is the guide. The API is on
[Run configuration (API)](../api/config.md), every option of every driver
is in the generated [option reference](../reference/options/common.md), and
the NetCDF input files are specified in
[Static input driver](../reference/input_static.md) and
[Dynamic input driver](../reference/input_dynamic.md).

## TOML layout

A config file names its driver in `[run]` and then sets options section by
section. A key that is not set takes the schema default. A key the schema
does not know is an error, and so is a value of the wrong type or outside
its range; every problem is reported together, naming the key and the file
it came from.

```toml
extends = "berlin_summer.toml"   # optional: start from another file

[run]
driver = "urban"                 # selects the schema: simulation, urban, sbl, dns, column, …
name   = "smoke"
seed   = 0

[grid]
size = [16, 16, 16]

[clock]
nsteps = 12

[fftw]
planner = "estimate"

[checkpoint]
every_steps = 0                  # off
```

Conventions: keys are ASCII `snake_case`; values are SI and durations are
in seconds; `"none"` stands for nothing (TOML has no null); a path is
relative to the file that sets it and is stored absolute. Environment
variables only fill a few defaults (`AEOLUS_LOG`, `AEOLUS_FFTW_PLANNER`,
`AEOLUS_FFTW_THREADS`, `AEOLUS_ALLOW_UNCOMMITTED`). The resolved value is
always written, so the environment never changes a replay.

The common sections (`[run]`, `[fftw]`, `[logging]`, `[checkpoint]`,
`[output]`) are in every driver. The others belong to the driver:

| driver | what it runs | options |
| :-- | :-- | :-- |
| `simulation` | a bare dynamical core on a box (TGV, dry CBL, buoyant convection) | [simulation](../reference/options/simulation.md) |
| `urban` | the urban LES run type (`Aeolus.UrbanCase`): the Berlin (Alexanderplatz) case with every coupling and the nest is its example | [urban](../reference/options/urban.md) |
| `urban_parent` | the nest parent of an urban case alone (same schema) | [urban_parent](../reference/options/urban_parent.md) |
| `sbl` | GABLS1–4, CASES-99 and the stratified channels | [sbl](../reference/options/sbl.md) |
| `dns` | the decaying DNS with its a-priori record | [dns](../reference/options/dns.md) |
| `column` | the single-column validation cases (DYCOMS RF01, ParisFog, parcel, Kessler box) | [column](../reference/options/column.md) |
| `channel`, `cbc_decay`, `patchy_surface`, `dns_decay` | the other validation drivers | [channel](../reference/options/channel.md), [cbc_decay](../reference/options/cbc_decay.md), [patchy_surface](../reference/options/patchy_surface.md), [dns_decay](../reference/options/dns_decay.md) |

## `extends` and overrides

`extends = "base.toml"` starts from another file. The base is looked up
next to the extending file first, then in the driver's config directory.
Tables merge key by key; arrays and scalars replace. Overrides are applied
last, so the precedence is **base < child < override**. The resolved
config has no `extends`: `run.toml` holds every option written out.

```toml
# examples/urban/configs/berlin_winter.toml
extends = "berlin_summer.toml"

[clock]
date = 2024-01-15
```

## `--set`

On the command line an override is `--set dotted.key=<TOML value>`.
Strings are quoted, arrays are TOML arrays:

```bash
julia --project -m Aeolus run examples/configs/taylor_green_vortex.toml --set grid.size=[16,16,16]
julia --project -m Aeolus run examples/urban/configs/berlin_smoke.toml --set physics.sgs='"smagorinsky"'
julia --project -m Aeolus check my_case.toml --set stepper.nsteps=10      # validate only
julia --project -m Aeolus resolve my_case.toml                            # print the canonical TOML
julia --project -m Aeolus options urban physics                           # the option reference
```

`bin/aeolus` is the same command on any Julia version (`-m` needs Julia ≥
1.12). Exit code 0 is success, 1 a failed run, 2 a usage or configuration
error raised before anything is written. A file with a `[sweep]` table
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
the `simulation` and `sbl` drivers a run interrupted and resumed is bitwise
equal to the uninterrupted run (`test/config_roundtrip.jl`). The Berlin case
has no checkpoint/restart yet (#269).

## The `[inputs]` hash check

Every input file option (a `Path`: a static or dynamic driver, a case
table) is recorded in `run.toml` under `[inputs]` with its `fnv1a64`
content hash. Replaying or resuming a run whose input files have changed
since is refused, naming each changed file, unless
`allow_changed_inputs = true` is passed to `load_run_config`. A replay
therefore cannot silently read different data.

## The preprocessors

Geometry and forcing come from NetCDF-4 driver files with Aeolus's own
schemas, written by preprocessors from the run config:

```bash
julia --project -m Aeolus preprocess static  examples/urban/configs/berlin_summer.toml      # OSM/GeoJSON → static driver
julia --project -m Aeolus preprocess dynamic --from-sounding examples/sbl/configs/gabls3.toml --out gabls3_dynamic.nc
julia --project -m Aeolus preprocess dynamic --from-parent  examples/urban/configs/berlin_smoke.toml --out child.nc
```

The static driver (terrain, buildings, land use, trees, surfaces) is
written from `[preprocess]` and read from `[geometry] static`. The dynamic
driver (initial profiles, large-scale forcing, surface series, boundary
planes from a parent run) is written from a sounding, a case table or a
recording parent, and is read from `[case] dynamic` (sbl) or `[initial] dynamic` (urban).
Their variables, dimensions, units and attributes are listed on the
[static](../reference/input_static.md) and
[dynamic](../reference/input_dynamic.md) pages; the API is
[Input drivers](../api/input_drivers.md).

## Where the shipped configs live

| directory | configs |
| :-- | :-- |
| `examples/configs/` | `simulation`: `taylor_green_vortex`, `dry_convective_bl`, `buoyant_convection` |
| `examples/urban/configs/` | `urban` (in `src/`, `Aeolus.UrbanCase`): `berlin_summer` (every option written out), `berlin_winter`, `berlin_smoke`, `berlin_mitte_sample`; `urban_parent`: `berlin_parent`, `berlin_parent_coarse`; `input/` holds their driver files |
| `examples/sbl/configs/` | `sbl` (in `src/`, `Aeolus.StableBoundaryLayer`): `gabls1`–`gabls4_exp3`, `cases99`, the channels, and the sweeps that replace the runner scripts (`*_closures`, `*_sweep`) |
| `examples/validation/configs/` | the validation run types of `Aeolus.ValidationRuns` (`channel`, `column`, `cbc_decay`, `patchy_surface`, `dns`, `plant_canopy`, `dynamic_roughness`): `neutral_abl`, `dycoms_rf01`, `dns256`, `canopy_profiles`, … |
| `examples/kernels/configs/` | `column`: `cloud_spinup` |

Each shipped config carries a `[test]` table with its smoke size:

```toml
[test]
overrides = ["grid.size=[16,16,16]", "stepper.end_time=\"none\"", "stepper.nsteps=10"]
member = "lasd_32x32x32"     # sweeps only: the member replayed (default: the first)
```

The loader ignores `[test]` (it checks only its shape): it is not part of
the resolved config, of `run.toml` or of the config hash, and it is not
inherited through `extends`. `test/config_roundtrip.jl` finds every
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
| Aeolus | TOML, schema-checked, with a Julia escape hatch; the resolved file is the run record | NetCDF-4 static and dynamic drivers with Aeolus's own schemas, made by `aeolus preprocess` |

The design follows PALM's split between a namelist and preprocessed driver
files, without aiming at PALM compatibility. What it adds is that the
config *is* the record: the canonical `run.toml` replays the run bit for
bit, and the option reference is generated from the same schema that
validates the file.
