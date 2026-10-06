# Run configuration

`Aeolus.Config` (CFG-a, #356) is how a run is configured: a TOML file,
checked against a declarative schema, resolved into an immutable typed
[`RunConfig`](@ref Aeolus.Config.RunConfig), and stored in `run.toml` in a
canonical form from which the run can be rebuilt. The loader, the
`aeolus` command line and its `--set` (CFG-b, #357) and the drivers'
option tables all build on it.

```bash
julia --project -m Aeolus run examples/configs/taylor_green_vortex.toml   # Julia ≥ 1.12
bin/aeolus run examples/configs/taylor_green_vortex.toml                  # any Julia
```

## Quick reference

```toml
# case.toml
extends = "base.toml"          # next to this file, else in the driver's config directory

[run]
driver = "common"
name   = "cbl"
seed   = 42

[fftw]
planner = "measure"

[checkpoint]
every_steps = 500
```

```julia
using Aeolus

cfg = load_run_config("case.toml"; overrides = ["checkpoint.keep=3", "run.seed=7"])
cfg.checkpoint.every_steps            # 500
cfg["fftw.planner"]                   # :measure
cfg2 = with(cfg, "run.seed" => 8)     # a new, validated config

write_config("resolved.toml", cfg)    # the canonical resolved form
config_hash(cfg)                      # fnv1a64 of that text

with_run(cfg) do rf                   # run.toml stores [config], its hash and [inputs]
    ck = checkpoint_schedule(cfg, checkpoint_dir(rf))
    # …
end

cfg3 = load_run_config("runs/2026-10-06_cbl_1a2b3c4d")   # == cfg
```

## Files, `extends` and overrides

A configuration file may start from another with `extends = "base.toml"`.
The base is looked up next to the extending file first, then in the
config directory the driver registered ([`register_driver!`](@ref
Aeolus.Config.register_driver!)). Chains are followed up to 8 links deep;
a cycle, a missing file or a deeper chain is an error naming the files.

Merging is key by key for tables; arrays and scalars replace (a child's
`levels = [1.0]` replaces the base's whole array). `overrides` are
`"dotted.key=<TOML value>"` strings applied last, so precedence is
**base < child < override**. Strings are quoted: `run.name="x"`.

[`load_run_config`](@ref Aeolus.Config.load_run_config) also reads a run
folder or its `run.toml`: then the record's `[config]` is used.

## Strict validation

Every problem is collected and thrown as one
[`ConfigError`](@ref Aeolus.Config.ConfigError): unknown keys (with a
"did you mean" suggestion within Damerau–Levenshtein distance 2), wrong
types, values out of range or outside the allowed set, tuples of the
wrong length, failed cross-field checks. Each problem names the dotted
key, the file the value came from, what was expected and what was given:

```
ConfigError: 3 problems in the run configuration
  - clock.nsteps (/work/case.toml): expected an integer in [0, 9223372036854775807], got "ten"
  - clock.scheme (/work/case.toml): expected a string (allowed: "rk3", "ab2", "imex"), got "rk4"
  - grid.sise (/work/base.toml): unknown key in [grid]; did you mean "size"?
```

## Conventions

* Keys are ASCII `snake_case`, transliterated from the Julia field names
  (`ν` → `nu`, `θ₀` → `theta0`, `C_s` → `c_s`); table names are singular
  nouns (`[grid]`, `[clock]`).
* Values are SI; durations and cadences are in seconds. A unit appears in
  a key only where the Julia field carries one (`start_h`).
* `nothing` is written `"none"`. A missing key takes the default, never
  `nothing`.
* A path is resolved against the directory of the file that set it (the
  working directory for an override or `with`) and stored absolute.
* Environment variables do not configure runs. `AEOLUS_LOG`,
  `AEOLUS_FFTW_PLANNER`, `AEOLUS_FFTW_THREADS` and
  `AEOLUS_ALLOW_UNCOMMITTED` only fill the defaults of `[logging]
  console_level`, `[fftw] planner` / `threads` and `[run]
  allow_uncommitted`; the resolved value is always written.

## Option types

| type | TOML | resolved value |
| :-- | :-- | :-- |
| `Float64` | float (an integer is accepted) | `Float64` |
| `Int`, `Bool`, `String` | integer, boolean, string | same |
| `Symbol` | string, checked against `allowed` | `Symbol` |
| [`Path`](@ref Aeolus.Config.Path) | string | absolute `String` |
| `NTuple{N,T}` | array of exactly `N` | `NTuple{N,T}` |
| `Vector{T}` | array | `Vector{T}` |
| `Date`, `Time` | local date, local time | `Dates.Date`, `Dates.Time` |
| [`Optional{T}`](@ref Aeolus.Config.Optional) | `T` or `"none"` | `T` or `nothing` |
| [`OpenTable(validator)`](@ref Aeolus.Config.OpenTable) | table | `Dict{String,Any}` |

`range = (lo, hi)` bounds numbers (closed; `open_lower = true` opens
`lo`), elementwise for tuples and vectors; `allowed` lists the permitted
values. A default may be a zero-argument function, evaluated at load
time.

## The common sections

Every driver's schema starts with these:

| section | options |
| :-- | :-- |
| `[run]` | `driver` (default and only value: the driver's name), `name`, `seed = 0`, `root = "runs"`, `allow_uncommitted` |
| `[fftw]` | `planner` (`estimate`, `measure`, `patient`, `exhaustive`), `threads` (`"none"`: one per Julia thread), `wisdom` |
| `[logging]` | `console_level`, `file_level = "debug"`, `flush_interval = 5.0` s, `diagnostics_cadence = 60.0` s |
| `[checkpoint]` | `every_steps = 0` (off), `every_seconds = inf` (off), `keep = 2`, `prefix = "checkpoint"` |
| `[output]` | `path` (`"none"`: the run folder's `output/`), `cadence = 3600.0` s |

[`checkpoint_schedule`](@ref Aeolus.Config.checkpoint_schedule) and
[`apply_fftw_config!`](@ref Aeolus.Config.apply_fftw_config!) read the
`[checkpoint]` and `[fftw]` sections. Their `Dict` methods (a namelist
from `load_config`) take their missing keys from the same schema
defaults.

## Drivers

A driver registers its sections, cross-field checks and config directory:

```julia
using Aeolus.Config

grid = section("grid"; doc = "the mesh") do s
    option!(s, "size", NTuple{3,Int}; default = (64, 64, 32), range = (1, 4096),
            doc = "cells per direction")
    option!(s, "extent", NTuple{3,Float64}; default = (3200.0, 3200.0, 1600.0),
            range = (0, Inf), open_lower = true, units = "m", doc = "domain size")
end
schema = register_driver!("cbl"; sections = [grid], config_dir = joinpath(@__DIR__, "config"))
check!(schema, cfg -> cfg.grid.size[3] > cfg.grid.size[1] ?
       [ConfigError("grid.size", "more vertical than horizontal cells")] : ConfigError[])
```

`[run] driver = "cbl"` then selects this schema; the driver `"common"`
(the common sections only) is always registered. A driver section named
like a common one adds its options to it (the `simulation` driver's
`[output] fields`); redeclaring a common option is an error.

## Run drivers

A schema says what a config may contain; a **run driver**
([`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver)) also says
how to build and run it. A driver type implements `driver_name`,
`driver_sections`, `driver_checks` (optional), `config_dir` (optional),
`build(d, cfg)`, `run!(d, obj, rf, ck; callback)` and `state_fields(obj)`,
and is registered with `register_driver!(D())`:

```julia
using Aeolus, Aeolus.Config

struct CBLDriver <: AbstractRunDriver end
Config.driver_name(::CBLDriver) = "cbl"
Config.driver_sections(::CBLDriver) = [grid]                 # as above
Config.build(::CBLDriver, cfg) = build_my_simulation(cfg)
function Aeolus.run!(::CBLDriver, sim, rf, ck; callback = nothing)
    d = resume!(ck, sim)                                     # a resumed run continues
    # … step, maybe_checkpoint!(ck, sim; step = n), callback(sim, n) …
end
Config.state_fields(sim) = (u = sim.u.data, θ = sim.θ.data)
register_driver!(CBLDriver())
```

[`run_config`](@ref Aeolus.Config.run_config)`(cfg; root, resume)` is
the Julia API behind `aeolus run`: `apply_fftw_config!`, `start_run`,
`run_checkpoints` from `[checkpoint]` (before the build, so a resumed run
imports the folder's FFTW wisdom before it plans), `build`, `run!`, then
`finish_run!` (`failed` or `interrupted` with the error, which is
rethrown). It returns a [`RunResult`](@ref Aeolus.Config.RunResult)
(`folder`, `obj`).

Drivers that live under `examples/` (the Berlin, SBL and validation
drivers) are not moved into `src/`: [`DRIVER_FILES`](@ref
Aeolus.Config.DRIVER_FILES) maps their names to files relative to the
package directory, and the first config that names one includes the file
into a fresh module. The migrations of #311 add their entries.

## The command line

| command | does |
| :-- | :-- |
| `run <config.toml> [--set k=v]… [--root DIR] [--allow-uncommitted]` | a new run (new run ID and folder) from a config file |
| `run <run-folder>/run.toml …` | a new run with the configuration of an earlier run |
| `run <run-folder> --resume` | continues that run: same run ID, a new session, from its newest checkpoint (`--set` is refused) |
| `check <config> [--set k=v]…` | validates and prints every problem |
| `resolve <config> [--set k=v]…` | prints the canonical resolved TOML |
| `options [driver] [section]` | prints the option reference, generated from the schema |
| `preprocess <kind> …` | a registered preprocessor ([`register_preprocessor!`](@ref Aeolus.Config.register_preprocessor!)) |

Exit codes: **0** success; **1** the run failed (blowup or error; its
session is recorded as `failed`); **2** a usage or configuration error,
raised before anything is written. `--root` and `--allow-uncommitted`
change where and whether the run starts, not its recorded config.
`julia --project -m Aeolus …` needs Julia ≥ 1.12 (the `-m` flag; the
`@main` entry point is marked on ≥ 1.11); `bin/aeolus` works on every
Julia the package supports. In Julia, `Aeolus.main(["run", "case.toml"])`
returns the exit code.

```@docs
Aeolus.main
```

## The `simulation` driver

`[run] driver = "simulation"` is a bare run of one dynamical core on a
box ([`Aeolus.SimulationDriver`](@ref)): what
`examples/taylor_green_vortex.jl`, `dry_convective_bl.jl` and
`buoyant_convection.jl` used to build in code. Their configs are in
`examples/configs/`, and the scripts are now three-line runners.

```toml
[run]
driver = "simulation"
name = "taylor_green_vortex"

[grid]
size = [32, 32, 32]
extent = [6.283185307179586, 6.283185307179586, 6.283185307179586]

[dynamics]
nu = 0.01

[initial]
u = { kind = "taylor_green", amplitude = 1.0, k = 1.0 }
v = { kind = "taylor_green", amplitude = 1.0, k = 1.0 }

[stepper]
dt = 0.07853981633974483        # shortened to end_time / ceil(end_time / dt) = 2/26
end_time = 2.0
```

This config is, bit for bit, the simulation the script built before
(`test/config_simulation_driver.jl`). Notes on the sections:

  * `[stepper]`: `dt` or `cfl` (adaptive, `step_adaptive!`, with a finite
    `dt_max`), and `end_time` or `nsteps`. `euler` is `step!`, `rk3`
    `step_rk3!`.
  * `[sgs] closure = "smagorinsky"` without a stability correction is the
    simulation's built-in closure (`C_s ≠ 0`); every other choice is
    passed to the steps as a closure object.
  * `[dynamics] nu` is also θ's (and qv's) molecular diffusivity: the
    cores have one coefficient (Prandtl number 1), so there is no separate
    `kappa`.
  * Combinations a core cannot run are config errors: the anelastic core
    needs z walls, `theta0 > 0`, a fixed `dt` with `euler`, a uniform grid
    and no checkpoints; a stretched grid takes the `none` or `smagorinsky`
    closure.
  * The run writes `diagnostics.csv` (the run monitor, every
    `[logging] diagnostics_cadence`), checkpoints on `[checkpoint]`, and
    the `[output] fields` every `[output] cadence` as
    `output/fields_<step>.nc`.

The options:

```@example
using Aeolus # hide
print(Aeolus.Config.option_reference("simulation")) # hide
```

## The bit-identical round trip

`test/config_roundtrip_harness.jl` is the check every config migration
uses: `roundtrip_check(path; nsteps, overrides)` runs a config through
`run_config`, re-runs the run from its own `run.toml` (a new run), and
compares every array of `state_fields` bitwise
(`reinterpret(UInt64, a) == reinterpret(UInt64, b)`), `diagnostics.csv`
outside its wall-clock column, and the two `[config]` tables. Both runs
use `[fftw] planner = "estimate"` (the bitwise policy of #245) and the
same thread count.

## The canonical form and `run.toml`

[`write_config`](@ref Aeolus.Config.write_config) writes every option,
defaults included, keys sorted, `schema_version = 1` at the top, no
`extends`, paths absolute and floats in Julia's shortest round-trip
representation (`inf`, `-inf`, `nan` for the specials). Writing a loaded
canonical file again gives the same bytes; `0.1`, `5e-324`, `-0.0`,
`±Inf` and `NaN` keep their bits. [`config_hash`](@ref
Aeolus.Config.config_hash) is the `fnv1a64` of this text.

`start_run(cfg)` (or `start_run(case; config = cfg)`) writes to `run.toml`:

| key | contents |
| :-- | :-- |
| `[config]` | the canonical resolved table |
| `config_schema_version`, `config_hash` | `1`, 16 hex digits |
| `config_sources` | `files` (the extends chain, leaf first), `overrides`, and the keys set by `with` |
| `[inputs]` | every input `Path` option: `path` and its `fnv1a64` content hash |
| `replayable`, `nonreplayable_keys` | `false` and the keys if `with` set a value TOML cannot hold (a closure) |

`load_run_config(folder)` returns a config equal to the one the run
started with. It refuses a run whose input files changed (naming each
file) unless `allow_changed_inputs = true`, and a run that is not
replayable; a restart (`start_run(…; resume = folder)`) checks the inputs
the same way before it writes anything. A `Dict`, `NamedTuple` or other
object passed as `config` is recorded as before (`toml_value`).

## Reference

```@autodocs
Modules = [Aeolus.Config, Aeolus.SimulationDriver]
```
