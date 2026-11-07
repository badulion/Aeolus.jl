# Run configuration

`Aeolus.Config` (CFG-a, #356; schema version 2, ONE-b #451) is how a run
is configured: one self-contained TOML file, checked against a declarative
schema (the model's, or a legacy case's), resolved into an immutable typed
[`RunConfig`](@ref Aeolus.Config.RunConfig), and stored in `run.toml` in a
canonical form from which the run can be rebuilt. The loader, the
`aeolus` command line and its `--set` (CFG-b, #357) and the option tables
all build on it. The guide is [Run configuration](../design/run_configuration.md).

```bash
julia --project -m Aeolus run examples/taylor_green_vortex/taylor_green_vortex.toml   # Julia ≥ 1.12
bin/aeolus run examples/taylor_green_vortex/taylor_green_vortex.toml                  # any Julia
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
reference: [the model](../reference/options/model.md) (no legacy case is
left since NEST-p3, #483); the guide is [Run configuration](../design/run_configuration.md). A section named
like a common one adds its options to it (the model's `[output]
variables`, `averaging`, `series_cadence`, `netcdf`); redeclaring a
common option is an error.

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

No legacy case is left: the last one, `urban_parent` (the nest parent
of an urban site alone), went with NEST-p3 (#483), where the parent became
a `[[domain]]` of the model; nothing is loaded from `examples/`. The
validation run types of the former `Aeolus.RunDrivers` are configurations
of the model since ONE-l (#475): `advection` (`[kinematic_flow]`), `cbl`
(`[sgs] model = "tke"`, `[inflow_statistics]`), `shear_layer` (the
`tanh` and `seed_mode` initial profiles) and `flat_plate` (`[fringe]`);
`inflow_fetch` and `obstacle_channel` (channel cases) and `sgs_decay`
(the filtered-DNS study) were dropped, their harnesses kept as test
fixtures (`test/fixtures/`). The stable-boundary-layer cases (the former `sbl` run
type) are configurations of the model since ONE-g (#456); the validation
run types `dns`, `channel`, `cbc_decay` and `patchy_surface` and the
single-column cases went with their cases (maintainer decisions of
2026-10-09). The two legacy cases that were Julia files under `examples/`
are gone (ONE-i #458): the Kessler box of the kernel demos (`column`) is
the configuration `examples/cloud_spinup/cloud_spinup.toml` of the model,
and the `dns_decay` run type is the shared P56-c fixture
`test/fixtures/dns_decay.jl`.

A legacy case gets a new section only where the case has structure of its
own; a driver with a `case` switch keeps one section per case and refuses
a key set in the section of a case that is not selected
([`selected_case_sections`](@ref Aeolus.Config.selected_case_sections)).
A section whose options are the fields of a keyword struct is made with
[`kwdef_section`](@ref Aeolus.Config.kwdef_section) (the struct's
defaults are the schema's) and read back with
[`kwdef_value`](@ref Aeolus.Config.kwdef_value).
`run_configs`, which ran a config from an example script's command line,
is gone with those scripts (ONE-i #458): `aeolus run` runs a config or
every member of a sweep.

No `.jl` file under `examples/` builds or steps a model:
`test/examples_enforcement.jl` parses each one and fails on a call into
the model's construction or stepping API, with no exemptions (see
[Run configuration](../design/run_configuration.md#Where-the-shipped-configs-live)).

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
axis = [ { key = "sgs.model", values = ["smagorinsky", "vreman", "amd"] },
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

Exit codes: **0** success; **1** the run failed (blowup or error; its
session is recorded as `failed`); **2** a usage or configuration error,
raised before anything is written; **3** the run completed but a required
check of `[checks]` failed (its session is recorded as `failed`;
`output/results.toml` says which, [Checks](checks.md), ONE-d #453). `--root` and `--allow-uncommitted`
change where and whether the run starts, not its recorded config.
The input drivers a run reads are made by the companion package
AeolusPrep.jl (`julia --project -m AeolusPrep …`, ONE-h #457; see
[Input drivers](input_drivers.md)), not by `aeolus`.
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
RK3, one hour). The example cases (`examples/<case>/<case>.toml`) replace
the scripts that built these runs in code (EX-d, #441; ONE-i, #458), and
the validation run types (ONE-l, #475).

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
model = "none"

[initial]
u = { kind = "taylor_green", amplitude = 1.0, k = 1.0 }
v = { kind = "taylor_green", amplitude = 1.0, k = 1.0 }

[timestep]
scheme = "euler"

[time]
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
    `[advection]`, `[sgs]`, `[coriolis]` (`f` or `latitude`), `[sponge]`
    (`alpha_max`, `z_start`, `target`), `[surface_theta_flux]` (`bottom`,
    `top`), and the boundary-layer components of the SBL cases (ONE-g,
    #456): `[large_scale_forcing]`, `[surface] model = "most"` and
    `[towers]` ([The stable-boundary-layer cases](@ref)).
  * **One section per exclusive slot** (ONE-m, #476): `[core] model`
    (`boussinesq`, `anelastic`), `[timestep] scheme` (`rk3`, `euler`),
    `[sgs] model` (`smagorinsky`, `dynamic_smagorinsky`, `vreman`,
    `sigma`, `amd`, `tke`, `none`), `[surface] model` (`none`, `slab`,
    `flux_map`, `most`), `[microphysics] scheme` (`none`, `kessler`) and
    `[advection] scheme` (`centred`). The selector key names the provider;
    the section holds that provider's options and no other's (`[sgs] model
    = "tke"` with `e_min`, `surface_factor`, `wall_cap`, `wall_cap_top`, `surface_value`, `stress`; a
    `c_s` there is an error naming the closures that have one), and the
    resolved config shows the provider with all its options. A selector
    set by `--set`, `with` or a `[domain.<section>]` block that names
    another provider replaces it: the options set below it are dropped
    and the new provider starts from its defaults (`--set
    sgs.model="vreman"` on a file with `[sgs] model = "tke"`). Each provider's options are a block of the
    option reference (`[sgs] model = "tke"`).
  * `[initial]` sets the velocity, θ and qv from vertical profiles
    (`constant`, `mixed_layer_lapse`, `table` with an optional `log` or
    `linear` tail below the lowest height, `exponential`), the Taylor–Green
    field, or a named analytic field: `field = {kind =
    "isotropic_turbulence", spectrum = "passot_pouquet", u_rms, k_peak,
    seed}`, decaying isotropic turbulence defined in spectral space, so that
    a coarse grid on the same box starts from exactly the large scales of a
    fine one (`examples/dns_vs_les/`). The θ noise (`perturbation_amplitude`,
    `perturbation_depth`, `perturbation_kind` uniform or normal) and the
    velocity noise (`velocity_perturbation`: kind, amplitude, `depth`,
    `components`, a `linear` taper and `project`) are drawn from the run's
    seed. `[time] start` is the model time at the start (the clock on which
    the dynamic driver's tables are read).
  * `[time]`: `dt` a fixed step or `"adaptive"` (`step_adaptive!` with
    `cfl` and `dt_max`), and `duration` or `nsteps`. `[timestep] scheme`:
    `euler` is `step!`, `rk3` `step_rk3!`.
  * `[sgs]` selects the closure object of the `:sgs` component
    (`Components.build_closure`), each closure's coefficient `c_s` its
    own default (smagorinsky and vreman 0.17, sigma 1.35, amd 0.3);
    `model = "none"` enables no SGS component.
  * `[dynamics] nu` is also θ's (and qv's) molecular diffusivity: the
    cores have one coefficient (Prandtl number 1), so there is no separate
    `kappa`.
  * Combinations a core cannot run are config errors: the anelastic core
    needs z walls, `theta0 > 0`, no moisture and a uniform grid (it takes
    `euler` and `rk3`, a fixed or adaptive `dt` and checkpoints like the
    Boussinesq core since ONE-j, #460); a stretched grid takes the `none`
    or `smagorinsky` closure.
  * The run writes `diagnostics.csv` (the run monitor, every
    `[logging] diagnostics_cadence`), checkpoints on `[checkpoint]`, and
    the `[output] variables` with the one writer ([Output](output.md),
    ONE-c #452) into `output/output.nc`. The 3-D fields, maps and profiles
    are written every `[output] cadence`, and the time series (domain
    means, budgets) every `series_cadence` (0: every step), either
    `instant` or as the `mean` over the interval (`averaging`). The
    variables come from the catalogues of the core and the enabled
    components. The default set is `u`, `v`, `w`, `theta` and the domain
    means of u, v, w, θ. A `budget_*` variable (`budget_energy`,
    `budget_water`, `budget_momentum`, `budget_tracer`) switches on that
    budget of the domain ([Budgets](budgets.md)) and writes its terms. The
    core also offers `kinetic_energy`, `dissipation` (molecular + SGS),
    `dissipation_sgs` and, on a triply periodic regular grid, the energy
    spectrum `energy_spectrum` (a record over the shell wavenumbers `k`).
  * `[checks]` enables the checks the core and the components publish,
    each `name = {required, tolerance}`: the sanity check `finite`, the
    budget closures `energy_closure`, `water_closure`,
    `momentum_closure` and `tracer_closure` (which turn their budgets on),
    and the reference comparisons `[checks.reference.<quantity>]` against
    `reference_file`. The run writes `output/results.toml` and ends with
    exit code 3 when a required check failed ([Checks](checks.md), ONE-d
    #453). Every shipped model config enables at least `finite`.
  * `[budgets]` (#539): `enabled = true` (default) measures the budgets
    the `budget_*` outputs and the closure checks ask for; `enabled =
    false` builds none, in model domains and urban sites alike, for
    production runs. The requested budget outputs and budget checks
    (`*_closure`, `urban_budgets`) are then dropped with one warning that
    lists them, so the same file runs with `--set budgets.enabled=false`,
    and `results.toml` records `[budgets] enabled = false`.
    `verify_writes = true` measures every stage entry and stops at a
    component that changed a budget its declared writes
    (`Components.budget_writes`) leave out (tests, debugging).
  * The domain's `dynamic_driver` names a dynamic driver: its profiles set the
    initial u, v, θ (and qv), replacing the `[initial]` tables. Lateral
    faces set to `"open"` (`boundaries.x`, `y`) take its boundary planes,
    blended linearly in time (boussinesq core, uniform grid, z walls, no
    sponge). AeolusPrep.jl writes one from ERA5 (`aeolusprep dynamic
    --from-era5 <files…> --config <config.toml> --lat … --lon …`;
    [Input drivers](input_drivers.md), CFG-h #363);
    `examples/lindenberg_era5/lindenberg_era5.toml` is the example, with its
    committed driver.

Its options, every key with type, default, units, range and doc: [Options: the model](../reference/options/model.md) (generated from the schema; `aeolus options` prints the same).

## The urban site

A model config with `[urban] enabled = true` is an urban site: a 3-D LES
of a city window (`Aeolus.Urban`, `src/components/urban/site/config_schema.jl`;
CFG-c #358 as the `berlin` example driver, the `urban` run type of EX-a
#438, on the model since ONE-e #454, see [Urban site](urban.md)).
In a nest it is the child of a domain of the model's core (NEST-p3,
#483: its parent's `[[domain]]` block configures the parent completely).
The Berlin (Alexanderplatz) case is shipped as files in `examples/urban/`:

| file | what |
| :-- | :-- |
| `berlin_summer.toml` | the flagship: the 6 km parent `[[domain]]` and the site nested in it, **every option of the site written out** |
| `berlin_winter.toml` | 15 January 2024, idealised cold profile (summer's file with the site's 11 keys and the parent's initial state, sun and forcing changed) |
| `berlin_smoke.toml` | 16³ window at noon in a small parent domain, 12 steps, no NetCDF |
| `berlin_smoke_site.toml` | the smoke site alone, a periodic box (the smoke runs without a parent start from it) |
| `berlin_mitte_sample.toml` | the synthetic OSM sample in the 6 km parent |
| `berlin_parent.toml` | the 6 km nest parent alone, 24 h (a domain of the model's core) |
| `berlin_parent_coarse.toml` | the coarse 24 × 24 × 38 parent (the Lindenberg checks) |

Each file is complete (no `extends`, ONE-b #451).

```bash
julia --project -m Aeolus run examples/urban/berlin_summer.toml --set time.nsteps=1800
julia --project -m Aeolus run examples/urban/berlin_smoke.toml --set domain.site.sgs.model=\"smagorinsky\"
```

A run writes its own checks (the budgets' closure, the timing) to
`output/results.toml`. The model's sections it takes are `[grid]` (a
uniform grid), `[dynamics]` `nu`, `theta0`, `g`, `[time]` (a fixed
`dt` and `nsteps` with `duration = "none"`: with `[timestep] scheme =
"adaptive"` `nsteps × dt` is the simulated time; in a nest `[time] cfl`
and `dt_max` are the root's adaptive step), `[output]` (`path`,
`cadence`) and the domain's `static_driver` (the ground, CFG-f #361, see
[Input drivers](input_drivers.md), made by AeolusPrep.jl) and
`dynamic_driver` (a sounding driver, or the parent-made driver of an
offline nest). The model's own components (`[coriolis]`, `[sponge]`,
`[surface_theta_flux]`, the EX-d couplings) are not part of an urban
site: the site's core carries its Coriolis force and sponge. The slot
sections the site shares with the model (`[core]`, `[timestep]`, `[sgs]`,
`[microphysics]`) offer the site's providers on it (ONE-m, #476).

  * **`[urban]`**: the site — the anchor
    (`lat0`, `lon0`, `offset`, `elevation`), the clock (`date`,
    `utc_offset`, `start_h`), the periods of the radiation, the soil and
    the surfaces, the core's `f_coriolis`, `c_s`, `pr_t`,
    `sgs_stability`, sponge and θ top, the lateral boundaries without the
    nest, the initial state (`sounding`, or the idealised profile),
    `canopy_top` of the air-heat budget, the NetCDF file name (`netcdf`,
    written by the one writer of [Output](output.md) into `[output]
    path`: the summer file writes `examples/urban/output/alexanderplatz.nc`,
    where the renderers read it) and the child's boundary driver
    (`child_boundaries`). The generic site points at no data of the
    example: `sounding` defaults to `"none"`.
  * **One section per urban slot** (ONE-m, #476; `URBAN_SLOT_SECTIONS`):
    `[core] model`, `[timestep] scheme`, `[pressure] solver`,
    `[momentum_advection] scheme`, `[scalar_advection] model`, `[sgs]
    model`, `[surface_exchange] model`, `[ground] model`,
    `[surface_radiation] model`, `[column_radiation] model`, `[buildings]
    model`, `[glazing] model`, `[immersed_boundary] method`, `[chemistry]
    scheme` (`aq_chem` or `none`) and `[microphysics] scheme`: the
    selector names the provider (default: the one of `URBAN_DEFAULTS`)
    and the section holds its options (`urban_slot_section`). **One
    section per additive component**, `[<name>]`: `enabled` (default
    `false`) and its options. A component's options are its
    `default_options`, generated from them (`coupling_options!`), so a
    component never has to repeat itself in a schema: the key set is closed, a key is the option's name in ASCII
    snake case (`U_g` → `u_g`, `τ` → `tau`), and a value has its default's
    form (a `Symbol` is a string, a tuple an array of the same length, a
    `NamedTuple` a sub-table whose missing keys keep the default, a table
    by surface class has integer keys, a `nothing` default takes `"none"`
    or a number). The components' `incompatible` declarations are
    cross-field checks whose message names the sections (`[core] model =
    "anelastic" is incompatible with [pressure] solver = "cosine_fft"`)
    and carries the `incompatible_reason`.
  * **`[nesting]`**: the site's nest (`parent_driver`: `"aeolus"`, its
    parent `[[domain]]`; `"profile"` or `"offline"` for a site alone; the
    two-way options). Whether it feeds back is its `[[domain]]` block's
    `feedback`. The parent is a domain of the model's core, configured in
    its own block (NEST-p3, #483).

**One source of truth for the defaults.** The schema holds them;
`site_options(default_run_config())` starts from it, and
`site_options(cfg::RunConfig)` reads the site's options that
`build_case` consumes (ONE-k, #474: there is no `CaseConfig` struct of
its own). `test/config_berlin.jl` checks that the site's fields and the
model's options map one to one,
and that the shipped Berlin files load, round-trip and report their
errors. Running them end to end is a validation case
(`validation/urban_checks.jl`).
`[checkpoint]` needs the Boussinesq core.

Its options, the component sections included: [Options: the model](../reference/options/model.md).

## The stable-boundary-layer cases

GABLS1-4 and CASES-99 (`examples/sbl/`) are configurations of the
model since ONE-g (#456): the geostrophic forcing and the dynamic driver's
large-scale tables are `[large_scale_forcing]`
([`LargeScale`](@ref Aeolus.Components.LargeScale)), the Monin-Obukhov
surface layer over the prescribed surface temperature `[surface] model = "most"`
([`SurfaceLayer`](@ref Aeolus.Components.SurfaceLayer)), the sponge
retargeted at the geostrophic wind or the plane mean `[sponge] target`,
the CASES-99 towers `[towers]` ([`Towers`](@ref Aeolus.Components.Towers)).
Their case tables are dynamic drivers (`examples/sbl/input/`), their
initial profiles TOML tables (with a `log` or `linear` tail below the
lowest height), their comparisons with the intercomparisons
`[checks.reference]` (`results.toml`), and the scripts of `examples/sbl/`
read `output/output.nc`. The channel cases of the former `sbl` run type
(stratified, relaminarisation, collapse) are gone (maintainer decision of
2026-10-09: no pressure-gradient forcing component).

## The round trip

`test/config_roundtrip_harness.jl` is the check every config migration
uses: `roundtrip_check(path; nsteps, overrides)` runs a config through
`run_config`, re-runs the run from its own `run.toml` (a new run), and
compares every array of `state_fields` and every numeric column of
`diagnostics.csv` (outside its machine columns) within `ROUNDOFF_RTOL`
(`test/support/agreement.jl`, #495), and the two `[config]` tables
exactly (they are parsed and written, not computed).

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
