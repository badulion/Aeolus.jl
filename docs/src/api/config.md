# Run configuration

`Aeolus.Config` (CFG-a, #356; schema version 2, ONE-b #451) is how a run
is configured: one self-contained TOML file, checked against a declarative
schema (the model's, or a legacy case's), resolved into an immutable typed
[`RunConfig`](@ref Aeolus.Config.RunConfig), and stored in `run.toml` in a
canonical form from which the run can be rebuilt. The loader, the
`aeolus` command line and its `--set` (CFG-b, #357) and the option tables
all build on it. The guide is [Run configuration](../design/run_configuration.md).

```bash
julia --project -m Aeolus run examples/configs/taylor_green_vortex.toml   # Julia ≥ 1.12
bin/aeolus run examples/configs/taylor_green_vortex.toml                  # any Julia
```

## Quick reference

```toml
# case.toml: the model (no [run] legacy)
schema_version = 2

[run]
name   = "cbl"
seed   = 42

[surface_theta_flux]
bottom = 0.24

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

## Files, domains and overrides

A configuration file is self-contained: there is no `extends` (refused,
like `schema_version = 1` and `[run] driver`, with what changed). The
schema is the model's ([`MODEL`](@ref Aeolus.Config.MODEL)), or a legacy
case's when `[run] legacy = "<case>"` names one ([`legacy_cases`](@ref
Aeolus.Config.legacy_cases)). The model's per-domain sections
([`is_domain_section`](@ref Aeolus.Config.is_domain_section)) are resolved
for every domain: the run-level values, replaced key by key by a
`[[domain]]` block's `[domain.<section>]` tables
([`config_domains`](@ref Aeolus.Config.config_domains),
[`DomainConfig`](@ref Aeolus.Config.DomainConfig),
[`domain_view`](@ref Aeolus.Config.domain_view)); with one domain,
`cfg.grid` is that domain's section.

`overrides` are `"dotted.key=<TOML value>"` strings applied last
(**file < override**); an override of a per-domain key applies to every
domain, over the blocks. Strings are quoted: `run.name="x"`.

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
| [`OneOf(T1, T2, …)`](@ref Aeolus.Config.OneOf) | a value of any of the types, the first that accepts it | that type's value |

`range = (lo, hi)` bounds numbers (closed; `open_lower = true` opens
`lo`), elementwise for tuples and vectors; `allowed` lists the permitted
values. A default may be a zero-argument function, evaluated at load
time.

## The common sections

Every driver's schema starts with these:

| section | options |
| :-- | :-- |
| `[run]` | `name` (the model: `"aeolus"`), `seed = 0`, `root = "runs"`, `allow_uncommitted`; a legacy case's also `legacy` (default and only value: its name) |
| `[fftw]` | `planner` (`estimate`, `measure`, `patient`, `exhaustive`), `threads` (`"none"`: one per Julia thread), `wisdom` |
| `[logging]` | `console_level`, `file_level = "debug"`, `flush_interval = 5.0` s, `diagnostics_cadence = 60.0` s |
| `[checkpoint]` | `every_steps = 0` (off), `every_seconds = inf` (off), `keep = 2`, `prefix = "checkpoint"` |
| `[output]` | `path` (`"none"`: the run folder's `output/`), `cadence = 3600.0` s |

[`checkpoint_schedule`](@ref Aeolus.Config.checkpoint_schedule) and
[`apply_fftw_config!`](@ref Aeolus.Config.apply_fftw_config!) read the
`[checkpoint]` and `[fftw]` sections. Their `Dict` methods (a namelist
from `load_config`) take their missing keys from the same schema
defaults.

## Schemas: the model and the legacy cases

A schema registers its sections and cross-field checks (the model also
its per-domain sections, `domain_sections`):

```julia
using Aeolus.Config

grid = section("grid"; doc = "the mesh") do s
    option!(s, "size", NTuple{3,Int}; default = (64, 64, 32), range = (1, 4096),
            doc = "cells per direction")
    option!(s, "extent", NTuple{3,Float64}; default = (3200.0, 3200.0, 1600.0),
            range = (0, Inf), open_lower = true, units = "m", doc = "domain size")
end
schema = register_driver!("cbl"; sections = [grid])
check!(schema, cfg -> cfg.grid.size[3] > cfg.grid.size[1] ?
       [ConfigError("grid.size", "more vertical than horizontal cells")] : ConfigError[])
```

`[run] legacy = "cbl"` then selects this schema; `"common"` (the common
sections only) is always registered. The options are in the generated
reference: [the model](../reference/options/model.md), and of the legacy
cases [`urban`](../reference/options/urban.md), [`sbl`](../reference/options/sbl.md), [`dns`](../reference/options/dns.md), [`column`](../reference/options/column.md) and the others; the guide is [Run configuration](../design/run_configuration.md). A section named
like a common one adds its options to it (the model's `[output]
variables`); redeclaring a common option is an error.

## What runs a config

A schema says what a config may contain; an
[`AbstractRunDriver`](@ref Aeolus.Config.AbstractRunDriver) also says how
to build and run it: the model's (`Aeolus.SimulationDriver.SimulationDriverType`)
or a legacy case's. A type implements `driver_name`, `driver_sections`,
`driver_checks` (optional), `domain_sections` (optional, the model's),
`build(d, cfg)`, `run!(d, obj, rf, ck; callback)` and `state_fields(obj)`,
and is registered with `register_driver!(D())`
([`run_driver`](@ref Aeolus.Config.run_driver)`(cfg)` finds it,
[`config_case`](@ref Aeolus.Config.config_case) names it):

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

Legacy cases that live under `examples/` (the validation drivers) are
loaded on first use: [`DRIVER_FILES`](@ref
Aeolus.Config.DRIVER_FILES) maps their names to files relative to the
package directory, and the first config that names one includes the file
into a fresh module. The `sbl` run type moved into `src/` with EX-b (#439,
[`Aeolus.StableBoundaryLayer`](era6/stable_boundary_layer.md)) and is
registered when the package loads. CFG-d (#359) lists `dns` (`examples/validation/dns_apriori_driver.jl`: the filtered-DNS
fixture with its a-priori record, `dns256.toml`); the drivers of CFG-e
(#360) are in the table below. `dns` and `dns_decay` run the same DNS
(the steps of `test/fixtures/dns_decay.jl`); `dns` adds the a-priori
record and its resolution refusal. The Berlin case's driver is no longer
an example file: it is the package's urban run type, `urban` (below; EX-a #438).

### The example legacy cases (CFG-e, #360)

| `[run] legacy` | file | builds and runs | configs |
| :-- | :-- | :-- | :-- |
| `channel` | `examples/validation/channel_driver.jl` | `half_channel_case` / `channel_case`, `run_channel!` (wall-modelled channel LES) | `validation/configs/neutral_abl.toml`, `log_layer_mismatch.toml` |
| `column` | `examples/validation/column_driver.jl` | `[column] case`: `dycoms_rf01` (RF01 column + longwave solve), `parisfog_2011` (the RadiationFog column), `droplet_parcel` (spectral broadening, `run_broadening_case`), `kessler_box` (Kessler microphysics, no dynamics) | `validation/configs/dycoms_rf01.toml`, `parisfog_2011.toml`, `spectral_broadening.toml`, `kernels/configs/cloud_spinup.toml` |
| `cbc_decay` | `examples/validation/cbc_driver.jl` | `cbc_benchmark` (Comte-Bellot & Corrsin decay, spin-up and seeds) | `validation/configs/cbc_decay.toml` |
| `patchy_surface` | `examples/validation/patchy_surface_driver.jl` | `patchy_les` / `patchy_run!` (CBL over hot and cool strips) | `validation/configs/patchy_surface.toml` |
| `dns_decay` | `examples/validation/dns_driver.jl` | `dns_decay` (the decaying DNS the a-priori analysis filters) | `validation/configs/apriori_sgs.toml` |

Each example script is now "run the config(s), then analyse or plot":
`run_configs(path, ARGS)` takes the script's command line (`--set`,
`--member`, `--root`, `--allow-uncommitted`) and returns one
[`RunResult`](@ref Aeolus.Config.RunResult) per run. A driver gets a new
section only where the case has structure of its own; a driver with a
`case` switch keeps one section per case and refuses a key set in the
section of a case that is not selected
([`selected_case_sections`](@ref Aeolus.Config.selected_case_sections)).
A section whose options are the fields of a keyword struct is made with
[`kwdef_section`](@ref Aeolus.Config.kwdef_section) (the struct's
defaults are the schema's: `FogColumnSettings` is `[fog]`,
`BroadeningCase` is `[parcel]`) and read back with
[`kwdef_value`](@ref Aeolus.Config.kwdef_value).

Not runs, so without a config (`EXEMPT` in `test/config_examples.jl`,
each with its reason): `validation/canopy_profiles.jl` (steady 1-D
boundary-value solves), `validation/dynamic_roughness_cubes.jl`
(a-priori roughness of a height map), `validation/dns256_generate.jl`
(the offline fixture generator), the Poisson manufactured-solution
checks, and the library files the drivers include. The same test lists
every example that calls a stepping entry point and requires a config or
an exemption; the scripts of #360's later parts are listed there until
they move.

## Sweeps

A config may carry a `[sweep]`: axes (a product, the first axis
outermost) and/or explicit members. The runner expands it into ordinary
overrides, base < member < `--set`, and each member is its own run
folder whose `run.toml` records the member's overrides and no `[sweep]`,
so it replays as a single run:

```toml
# … the case's sections …

[[sweep.member]]
label = "no_feedback"
fog = { droplet_feedback = false }

[sweep]
axis = [ { key = "sgs.closure", values = ["smagorinsky", "lasd", "lasi"] },
         { key = "channel.size", values = [[32, 32, 32], [48, 48, 48]] } ]
```

Members are labelled by their `label` (or index) joined with the axis
values (`lasd_32x32x32`); the run name gets the label as a suffix.
[`load_run_config`](@ref Aeolus.Config.load_run_config) refuses a sweep
file and says how to run it; [`load_sweep`](@ref
Aeolus.Config.load_sweep) and [`sweep_members`](@ref
Aeolus.Config.sweep_members) expand it; `aeolus run/check/resolve`
handle sweeps, and `run` and `resolve` take `--member k` (index or
label).

## The command line

| command | does |
| :-- | :-- |
| `run <config.toml> [--set k=v]… [--root DIR] [--allow-uncommitted]` | a new run (new run ID and folder) from a config file |
| `run <run-folder>/run.toml …` | a new run with the configuration of an earlier run |
| `run <run-folder> --resume` | continues that run: same run ID, a new session, from its newest checkpoint (`--set` is refused) |
| `check <config> [--set k=v]…` | validates and prints every problem |
| `resolve <config> [--set k=v]…` | prints the canonical resolved TOML |
| `options [legacy-case] [section]` | prints the option reference of the model (or of a legacy case), generated from the schema |
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

## The model

A config without `[run] legacy` runs the model
([`Aeolus.SimulationDriver`](@ref)): a [`Model`](@ref
Aeolus.Components.Model) of the Boussinesq or the anelastic core on a box
with its components. A bare config is a minimal dry LES (32³ on 3.2 km ×
3.2 km × 1.6 km, free-slip z walls, θ₀ = 300 K, Smagorinsky, adaptive
RK3, one hour). The configs of `examples/configs/`, `examples/kernels/`
and `examples/coupled/` replace the scripts that built these runs in code
(EX-d, #441). The run types of the other example cases are legacy cases
in [`Aeolus.RunDrivers`](@ref).

```toml
schema_version = 2

[run]
name = "taylor_green_vortex"

[grid]
size = [32, 32, 32]
extent = [6.283185307179586, 6.283185307179586, 6.283185307179586]

[dynamics]
nu = 0.01
theta0 = 0.0                    # no buoyancy

[boundaries]
z = "periodic"

[sgs]
closure = "none"

[initial]
u = { kind = "taylor_green", amplitude = 1.0, k = 1.0 }
v = { kind = "taylor_green", amplitude = 1.0, k = 1.0 }

[time]
scheme = "euler"
dt = 0.07853981633974483        # shortened to duration / ceil(duration / dt) = 2/26
duration = 2.0
```

This config is, bit for bit, the simulation the script built before
(`test/config_simulation_driver.jl`). Notes on the sections:

  * The run is a [`Model`](@ref Aeolus.Components.Model) of one domain
    stepped through the stage pipeline (ONE-a, #450; see
    [Components and the stage pipeline](@ref)): the core carries the
    dynamics, and its components have one section each, generated from
    their schemas ([`COMPONENTS`](@ref Aeolus.SimulationDriver.COMPONENTS)):
    `[advection]`, `[sgs]`, `[coriolis]` (`f`), `[sponge]` (`alpha_max`,
    `z_start`) and `[surface_theta_flux]` (`bottom`, `top`).
  * `[time]`: `dt` a fixed step or `"adaptive"` (`step_adaptive!` with
    `cfl` and `dt_max`), and `duration` or `nsteps`. `euler` is `step!`,
    `rk3` `step_rk3!`.
  * `[sgs]` selects the closure object of the `:sgs` component
    (`Components.build_closure`); `closure = "none"` enables no SGS
    component.
  * `[dynamics] nu` is also θ's (and qv's) molecular diffusivity: the
    cores have one coefficient (Prandtl number 1), so there is no separate
    `kappa`.
  * Combinations a core cannot run are config errors: the anelastic core
    needs z walls, `theta0 > 0`, a fixed `dt` with `euler`, a uniform grid
    and no checkpoints; a stretched grid takes the `none` or `smagorinsky`
    closure.
  * The run writes `diagnostics.csv` (the run monitor, every
    `[logging] diagnostics_cadence`), checkpoints on `[checkpoint]`, and
    the `[output] variables` every `[output] cadence` as
    `output/fields_<step>.nc`.
  * The domain's `dynamic_driver` names a dynamic driver: its profiles set the
    initial u, v, θ (and qv), replacing the `[initial]` tables. Lateral
    faces set to `"open"` (`boundaries.x`, `y`) take its boundary planes,
    blended linearly in time (boussinesq core, uniform grid, z walls, no
    sponge). `aeolus preprocess dynamic --from-era5 <files…> --config
    <config.toml>` writes one from ERA5 with the site, levels and outputs
    of `[preprocess.era5]` ([Input drivers](input_drivers.md), CFG-h
    #363); `examples/configs/lindenberg_era5.toml` is the example.

Its options, every key with type, default, units, range and doc: [Options: the model](../reference/options/model.md) (generated from the schema; `aeolus options` prints the same).

## The `urban` legacy case

`[run] legacy = "urban"` is the urban LES run type, `Aeolus.UrbanCase`
(`src/urban/core/config_schema.jl`; CFG-c #358 as the `berlin` example
driver, in the package since EX-a #438, see [Urban run type](urban_case.md)).
`urban_parent` runs the nest parent of the same config alone. The Berlin
(Alexanderplatz) case is shipped as files in `examples/urban/configs/`:

| file | what |
| :-- | :-- |
| `berlin_summer.toml` | the flagship, **every option written out** |
| `berlin_winter.toml` | 15 January 2024, idealised cold profile (summer's file with 11 keys changed) |
| `berlin_smoke.toml` | 16³ window at noon, small nest parent, 12 steps, no NetCDF |
| `berlin_mitte_sample.toml` | the synthetic OSM sample |
| `berlin_parent.toml` | `legacy = "urban_parent"`: the 6 km nest parent alone, 24 h |
| `berlin_parent_coarse.toml` | the coarse 24 × 24 × 38 parent (the Lindenberg checks) |

Each file is complete (no `extends`, ONE-b #451).

```bash
julia --project -m Aeolus run examples/urban/configs/berlin_summer.toml --set clock.nsteps=1800
julia --project -m Aeolus run examples/urban/configs/berlin_smoke.toml --set physics.sgs='"smagorinsky"'
```

There is no wrapper script (`alexanderplatz_diurnal.jl` was removed in
#438). The environment variables `AEOLUS_SCENARIO`, `AEOLUS_NSTEPS` and
`AEOLUS_OUTPUT_DT` are gone: use the file and `--set clock.nsteps=…`,
`--set output.cadence=…`. A run writes its own checks (the ledgers'
closure, the timing) to `output/results.toml`. The generic schema points
at no data of the example: `[preprocess] osm_path` and `landuse` and
`[initial] sounding` default to `"none"`.

The sections are `[grid]`, `[geometry]` (the site and the static
drivers `static`, `parent_static` the ground is read from, CFG-f #361,
see [Input drivers](input_drivers.md)), `[preprocess]` (the drivers'
sources: `osm_path`, `min_height`, `park_lai`, `landuse`; `julia
--project -m Aeolus preprocess static <config>.toml` writes the
drivers from them), `[clock]`, `[cadence]`,
`[dynamics]`, `[boundaries]`, `[initial]`, `[ledger]`, `[output]`
(`netcdf`, the file name; `path`, its directory: the summer file writes
`examples/urban/output/alexanderplatz.nc`, where the renderers read it;
`"none"` is the run folder's `output/`) and `[run]` (`scenario`,
`backend = "cpu"`, `serial`). Most of it is generated from the coupling
registry, so a coupling never has to repeat itself in a schema:

  * **`[cadence]`** has a key for every named period some coupling
    declares (the `Symbol` periods of its `stages`, and its
    `cadence_keys`). A period an enabled coupling needs must be set; a
    period set in a file that no enabled coupling uses is an error.
  * **`[physics]`**: `slot = "provider"` for each exclusive slot (the
    allowed values are the registered providers) and `addon = true/false`
    for each additive coupling. The registry's `incompatible`
    declarations are cross-field checks whose message carries the
    coupling's `incompatible_reason`.
  * **`[physics.options.<coupling>]`** is validated against that
    coupling's `default_options`: the key set is closed (an unknown key
    lists the coupling's options), and a value has its default's type
    (a `Symbol` is a string, a tuple an array of the same length, a
    `NamedTuple` a sub-table whose missing keys keep the default, a table
    by surface class has integer keys, a `nothing` default takes `"none"`
    or a number). Keys keep their Julia names (`τ`, `U_g`).
  * **`[physics.options.nesting.parent]`** is the nest parent's own
    schema (`ParentConfig`), its keys the ASCII field names
    (`nudging_tau`, `theta_top`). The function-valued fields have TOML
    forms: `q_surface` / `e_surface` take `{kind = "constant", value}` or
    `{kind = "series", t = […], value = […]}` (linear in time),
    `theta_init` / `qv_init` take `"sounding"` or a table
    `{kind = "table", z = […], values = […]}`, and `u_init` takes
    `"geostrophic"`, a constant or a table. Any other Julia function goes
    through `with` and makes the run `replayable = false`.

**One source of truth for the defaults.** The schema holds them;
`CaseConfig(; kw...)` and `ParentConfig(; kw...)` start from it (the
structs have no `@kwdef` defaults), and `CaseConfig(cfg::RunConfig)`
builds the struct that `build_case` consumes. `CaseConfig(:summer)` (and
the alias `:alexanderplatz`, `:winter`, `:smoke`, `:mitte_sample`) loads
the shipped file. `test/config_berlin.jl` checks that the struct fields
and the schema options map one to one, that the four files give, field
for field and bitwise, the configs the old Julia presets built
(`test/data/config_berlin/fixture.toml`), and that `berlin_smoke.toml`
run 12 steps gives state bitwise equal to the Julia-built smoke case and
passes the round trip with the nest on. The Berlin case has no
checkpoint/restart yet, so `[checkpoint]` must stay off.

Its options, the `[physics.options.<coupling>]` tables included: [Options: legacy case `urban`](../reference/options/urban.md).

## The `sbl` legacy case

`[run] legacy = "sbl"` (CFG-d #359; the module
[`Aeolus.StableBoundaryLayer`](era6/stable_boundary_layer.md) since EX-b
#439) runs the stable-boundary-layer cases of `examples/sbl/configs/`. One
driver covers four kinds, `[case] kind`:

| kind | builds and runs | shipped configs |
| :-- | :-- | :-- |
| `homogeneous` | `sbl_case`, `sbl_run!`: the periodic SBL LES | `gabls1`, `gabls2`, `cases99` (gabls2 + virtual towers), `gabls3`, `gabls4_exp3` |
| `stratified_channel` | a neutral spin-up, then `stratified_channel_run` | `stratified_channel_re180` |
| `relaminarisation_channel` | `relam_spinup`, then `relam_run` | `relaminarisation` |
| `collapse_channel` | the neutral half channel, then `collapse_run` | `collapse_channel` |

The Julia builders (`src/sbl/case.jl`) are the implementation; the
driver (`src/sbl/driver.jl`) only turns the config into their arguments. Each shipped
config gives fields bitwise equal to the builder call it replaced
(`test/config_sbl.jl`). Case constants live in the TOML files; the case
tables stay the CSVs of `examples/sbl/data/`, named by `[case] data` (a
`Path`, hashed under `[inputs]` of `run.toml`) and referenced as
`{table, column[, row]}`. Things TOML cannot hold are named kinds with
parameters, added in code:

  * a forcing quantity (`[forcing] u_g`, `v_g`, the `subsidence` `w`, the
    `advective_tendencies` and `nudging` targets) is a number or
    `{kind = "table", z = […], t = […] | time = {table, column}, levels =
    [{value} | {values} | {column}]}`, a `ForcingTable`;
  * `[surface] temperature` is `constant`, `cooling_rate`, `series` (a
    case-table column; with `p_sfc` and `exponent` the values are T,
    converted to θ) or `gabls2` (the case formula, sampled);
  * `[surface] stability` is a `SurfaceExchange` family with its
    parameters: `{family = "businger_dyer_log_linear", beta_m = 4.8,
    beta_h = 7.8}`;
  * `[initial]` profiles are `constant`, `mixed_layer_lapse` or `table`
    (with a `log` or `linear` profile below the lowest height), the
    perturbation `theta_uniform`, `uniform`, `gaussian` (with a linear
    taper per component) or `none`.

An option that the configured kind does not use, set away from its
default, is an error (a silently ignored key is a bug), as are `f` and
`latitude` together, `end_time` and `nsteps` together, a closure the
channel builders do not take, and a case-table reference that does not
resolve (naming the table, the file and its size). A run writes
`diagnostics.csv`, `output/statistics.toml` (the averaged statistics or
the channel result), `output/series.csv` (every `[averaging]
series_interval`) and, with `[towers] below > 0`, `output/towers.csv` (the
CASES-99 hockey stick bins). The comparisons the old runner scripts made
are sweeps now (`gabls1_closures.toml`, `cases99_closures.toml`,
`gabls3_closures.toml`, `gabls4_closures.toml`,
`stratified_channel_sweep.toml`, `relaminarisation_sweep.toml`,
`collapse_channel_sweep.toml`: one run folder per member, each channel
member with its own spin-up), and the scripts `gabls1.jl` … `gabls4.jl`,
`stratified_channel.jl`, `relaminarisation_channel.jl` and
`collapse_channel.jl` only read the run folders and make the tables and
figures (`julia --project examples/sbl/gabls1.jl runs/gabls1_*`).

Its options, every key with type, default, units, range and doc: [Options: legacy case `sbl`](../reference/options/sbl.md) (generated from the schema; `aeolus options sbl` prints the same).

## The `dns` legacy case

`[run] legacy = "dns"` (`examples/validation/dns_apriori_driver.jl`) is the
decaying-turbulence DNS of the filtered-DNS fixture
(`test/fixtures/dns_decay.jl`) and its a-priori record
(`test/fixtures/dns_apriori.jl`). `examples/validation/configs/dns256.toml`
is the offline 256³ run; `--set grid.n=32` runs the pipeline in seconds.
`[fluid] nu = "scaled"` is the fixture's ν = 0.018 (48/N)^{4/3} times
`nu_margin` (1.035, the generator's margin), so the file holds no
hand-computed viscosity. The run is the fixture's own steps (`dns_start`,
`dns_step!` while `dns_running`), bitwise `dns_decay`'s. After the run the
record goes to `[apriori] out` (default `output/dns_apriori.toml`); with
`require_checks` a run that misses a required DNS check (k_max η ≥ 1.5
over the run, the skewness band, the energy budget, monotone decay, past
the dissipation peak) fails, names the measured k_max η, and leaves its
record at `<out>.failed` only.

Its options, every key with type, default, units, range and doc: [Options: legacy case `dns`](../reference/options/dns.md) (generated from the schema; `aeolus options dns` prints the same).

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
defaults included, keys sorted, `schema_version = 2` at the top, the
model's per-domain sections under every `[[domain]]` (`[domain.grid]`,
`[domain.sgs]`, …), paths absolute and floats in Julia's shortest round-trip
representation (`inf`, `-inf`, `nan` for the specials). Writing a loaded
canonical file again gives the same bytes; `0.1`, `5e-324`, `-0.0`,
`±Inf` and `NaN` keep their bits. [`config_hash`](@ref
Aeolus.Config.config_hash) is the `fnv1a64` of this text.

`start_run(cfg)` (or `start_run(case; config = cfg)`) writes to `run.toml`:

| key | contents |
| :-- | :-- |
| `[config]` | the canonical resolved table |
| `config_schema_version`, `config_hash` | `2`, 16 hex digits |
| `config_sources` | `files` (the config file), `overrides`, and the keys set by `with` |
| `[inputs]` | every input `Path` option: `path` and its `fnv1a64` content hash (a domain's as `domain.<id>.<key>`) |
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
