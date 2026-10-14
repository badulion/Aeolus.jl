# One model: core code, configuration and input drivers

Status: **design, agreed with the maintainer on 2026-10-09**. The refactor
that implements it is tracked in the single-model epic. This page is the
target; where the code disagrees with it, the code is what changes.

## Principles

1. **There is one model.** The repository contains the model's core code and
   reusable modules. There is no per-case code anywhere: no case
   frameworks, no case-specific run types, and no Julia scripts that build or
   run a simulation.
2. **A run is a configuration plus input.** The TOML file says which modules
   run and how they are set up. The input comes from the **static driver**
   (geometry and surface description) and the **dynamic driver** (initial
   state, boundary conditions and forcing), both NetCDF files.
3. **Everything else is a reusable module.** Each module is written once and
   switched on or off by the configuration. A module never knows which city or
   case it runs in.

Berlin is therefore a configuration plus two NetCDF files, and so are GABLS1,
a Taylor-Green vortex and the flow past a cube. The same model runs each of
them.

## Decisions

| Topic | Decision |
| :-- | :-- |
| Run types | **Exactly one model.** `aeolus run config.toml` always runs it. There are no run types and no `[run] driver` key. Ensembles, parameter sweeps and spin-up chains are repeated `aeolus run` calls made by the user, not model features. |
| Module switching | **Explicit only.** Every module has its own TOML section and is enabled there. There are no presets and no mode switches. An exclusive slot is **one section with a selector key** (`[sgs] model = "tke"`) that holds the chosen provider's options; an additive module is on or off (`enabled`). |
| Bare-config default | **Minimal dry LES**: dynamics, SGS, pressure solve and dry θ. Every other module is off until enabled. |
| Coupling in a step | **A fixed, documented stage pipeline.** Modules implement one component interface and attach to named stages. |
| Nesting | **One TOML with several `[[domain]]` blocks.** Each domain has its own static and dynamic driver, and one `aeolus run` steps all of them. |
| Per-domain modules | **Every domain carries its complete module configuration**, the parent included, inside its own `[[domain]]` block. Run-level sections hold only what must be common to the run (`[run]`, `[time]`, output defaults, `[checks]`). There is no sharing or overriding of module sections between domains (decided 2026-10-12). |
| Nest time stepping | **Different step sizes per domain.** The parent picks its step adaptively; each child takes *n* equal substeps per parent step, *n* chosen every parent step as the smallest whole number that keeps the child stable. Parent and child meet at the end of every parent step, where two-way feedback is applied. |
| Failed steps in a nest | **Retry the window.** A failing child substep makes the child redo its window with a larger *n*. A failing parent step redoes the whole window (parent plus all children) with a smaller parent step, from the state saved at the window start. |
| Budgets across a nest | **Selectable, default `exchange_term`.** Each domain closes its own budgets, and what crosses its nest boundary or arrives through two-way feedback is booked as one "nest exchange" term per quantity. `off` is also available. A third mode, `attributed` (the exchange split into the physical processes that caused it), is a later issue on the same interface. |
| Concurrency in a nest | **The parent steps ahead (prefetch) only for one-way children.** With two-way feedback the parent waits for the feedback. Sibling children step in parallel. |
| Exchanged fields | **PALM rule plus a driver exception.** Every field exchanged through the nest (moisture, cloud and rain variables, tracers, chemistry species) must exist with the same definition in parent and child. A child may carry an extra field only if its own dynamic driver provides that field's boundary values. Nothing is switched on automatically; the config check names both domains. |
| Subgrid model per domain | **May differ** (for example TKE in the parent, dynamic Smagorinsky in the child). The nest passes resolved fields; a TKE field present on one side only has a documented boundary treatment. The config check warns when the parent is much smoother than the child and no inflow turbulence is switched on. |
| Mixed cores in a nest | **Fully supported**, two-way included. A Boussinesq child can nest in an anelastic parent and the other way round, with a conservative density conversion in the boundary data and the feedback. |
| Idealised set-ups | Allowed in TOML: vertical profiles, a named set of analytic initial fields, and simple box obstacles. Anything spatially detailed comes from the static or dynamic driver. |
| Driver format | **Aeolus's own schema** (CFG-f/g, `Aeolus.InputDrivers`). No PALM PIDS compatibility. |
| Parameters | Small parameter tables (materials, land-use classes, emission factors) may be written in TOML. Per-cell and per-object values belong in the static driver. |
| Output | **The configuration lists the variables.** Each module publishes a catalogue; `[output]` selects variables, cadence and averaging. |
| Validation | **Checks run inside the model.** `[checks]` turns on checks that modules publish. Results go to `results.toml` with pass/fail, and `aeolus run` exits non-zero if a `required` check failed. |
| Preprocessing | **A separate package in its own repository**, AeolusPrep.jl. It covers OSM/GIS to static driver, ERA5/sounding/parent output to dynamic driver, and data download. Aeolus keeps only the readers and the format schemas. |
| Single-column cases | **Dropped.** Only 3-D runs. |
| Channel cases | **Dropped** (the stratified, relaminarisation and collapse channels, and the wall-modelled validation channels). No pressure-gradient forcing component is built for them. |
| DNS a-priori study | **Dropped** (the filtered-DNS record and its run type). LES validation is to be done differently, possibly externally (#461); the model ships an a-posteriori DNS-against-LES example as configuration plus an output-only comparison script. |
| Config inheritance | **None.** Every TOML is self-contained; `extends` is removed. Variants of a case (Berlin winter, smoke) are complete files of their own; `--set` overrides on the command line remain for one-off changes. |
| Dynamical cores | **Every component supports both cores**, Boussinesq and anelastic, checkpointing included. There is no core-specific component and no "Boussinesq only" declaration. |
| Self-nesting feedback | **Two-way by default** for an Aeolus parent driving an Aeolus child (as in PALM). One-way stays available per child. Driving from reanalysis or a mesoscale model (offline, through the dynamic driver) is one-way by nature. |
| Compatibility | **Clean break.** All shipped configs are converted in the same PRs; old names, old run folders and old checkpoints are not supported. |

## What exists today, and why it changes

Measured on master `9024c0eb`.

- **Three ways of assembling a model.**
  - The core `simulation` run type (`src/config/simulation_driver.jl`, 630 lines). It maps a fixed set of sections onto `BoussinesqSimulation`/`Anelastic` and passes only `sgs=` to `step!`. It cannot attach forcing, wall models, a surface scheme or any extra physics.
  - The urban case framework (`src/urban/`, about 16,600 lines). It has its own `CaseConfig`, a coupling registry with stages, slots and providers, its own ledgers, diagnostics and output, and a separately hard-wired `ParentCase`.
  - The SBL run type (`src/sbl/`, 2,277 lines). It builds its own `force!`/`wall!` closures for large-scale forcing and MOST wall stress, and has its own momentum budget and writers.
- **Concepts that exist two or three times:**
  - the config layer;
  - SGS selection (an if-chain, an urban slot and an SBL function);
  - ledgers and budgets;
  - diagnostics;
  - output writers;
  - nesting assembly;
  - dynamic-driver consumption: only SBL reads the large-scale forcing tables.
- **Physics living inside glue code.** Several urban couplings define physics of their own instead of calling a module: `moist_buoyancy.jl` (moist buoyancy perturbation), `traffic_heat.jl`, `veg_drag.jl`, `snow_legacy.jl`, and the traffic profile and deposition velocities in `aq_chem.jl`. That breaks the one-source-of-truth rule.

The urban coupling contract is the most complete plug-in mechanism the code
has (build, init, stages, budgets, output, checks, dt limits). The design keeps
its ideas, moves them into the core, and makes it the only mechanism.

## Architecture

```
TOML ──► Config (schema, validation, resolved config) ──► Model
                                                          ├─ Domain 1: grid, state, core, components, drivers
                                                          ├─ Domain 2 … (nesting)
                                                          └─ Nest couplings (one-/two-way)
static.nc / dynamic.nc ──► InputDrivers (readers) ──► components that need them
Model ──► Output (catalogue-selected) + results.toml (checks) + checkpoints
```

### The model and its domains

`Model` holds one or more `Domain`s and the nest couplings between them.

A `Domain` holds:
- its grid;
- its prognostic state;
- its **core**: the dynamical core, an exclusive choice of `boussinesq` or `anelastic`. Every component works with either; a component's contract tests run on both cores;
- its tuple of active **components**;
- its static and dynamic driver;
- its clock, which nesting keeps in sync.

A single-domain run is a `Model` with one domain; there is no special case
for it.

### The component interface

A component is a reusable module instance configured by one TOML section. The
interface (`Aeolus.Components`, in the core) is:

| Method | Purpose |
| :-- | :-- |
| `section(::Type{C})`, `schema(::Type{C})` | Its TOML section name and the option schema (types, units, defaults, docs). |
| `slot(::Type{C})` | `nothing` for an additive component; otherwise an exclusive slot such as `:sgs`, `:advection`, `:microphysics`, `:radiation` or `:surface`. Only one component per slot per domain. |
| `requires`, `incompatible` | Other components it needs or excludes. The config is validated against these. |
| `build(C, opts, domain)` | Creates the instance and allocates its fields and work arrays. Reads drivers through `domain`. |
| `stages(c)` | The pipeline stages it runs in, each with an optional cadence. |
| `run_stage!(c, domain, Val(stage), dt)` | The work. It must be type-stable. |
| `limit_dt(c, domain)` | An optional stability limit. |
| `budget!(c, domain, budgets)` | Its terms in the shared budgets (energy, water, mass, momentum, tracers). |
| `outputs(c)` | Its output catalogue: names, units, dimensions, how each is computed. |
| `checks(c)` | The checks it offers: name, tolerance default, evaluation. |
| `checkpoint_state(c)` | Everything it needs for a bit-identical restart. |

Physics lives in the physics modules (`SurfaceExchange`, `Radiation`,
`Microphysics`, `Chemistry`, …). A component is a thin adapter from the
module to the pipeline. The rule for review: **if a component file contains
an equation, that equation belongs in a physics module.**

### The fixed stage pipeline

One step of one domain, in order. The core's numbered operations are written
in *italics*.

1. `clock`, then `nest_sync`: the parent window and the boundary planes for children.
2. `radiation`, then `sw_modifiers`.
3. `surface_flux`: land, urban, sea and building surfaces, plus the MOST/wall laws.
4. `sources`: anthropogenic heat, emissions, traffic.
5. `chemistry`, `microphysics`, `tracers`.
6. *Halo fill, advection, SGS, buoyancy, Coriolis.*
7. `body_force`: large-scale forcing, canopy and vegetation drag, sponge, IBM forcing.
8. *Predictor*, then `post_predictor`: wall stress, surface momentum.
9. *Open boundaries*: inflow turbulence, nest boundary planes.
10. *Projection*, then the *scalar advance*.
11. `surface_update`, `column_tendency`.
12. `nest_feedback`: two-way anterpolation and flux correction.
13. `budgets`, `checks`, then `output`.

Within a stage, components run in a fixed documented order: by slot first,
then by component name. Their order is therefore never a configuration
question.

Two performance requirements:
- The core keeps its hook-free fast path (SC-d/SC-e) when no active component attaches to stages 6–10.
- The component tuple is concretely typed, so dispatch is static.

### Configuration

```toml
schema_version = 2

[run]
name = "berlin_summer"
seed = 0

[time]
start = "2019-06-29T06:00:00+02:00"
duration = "24h"
dt = "adaptive"

[sgs]                        # exclusive slot; default Smagorinsky if omitted
model = "tke"

[moisture]
enabled = true

[radiation]
scheme = "two_stream"
cadence = "60s"

[urban_surface]
enabled = true
materials = { concrete = { conductivity = 1.51, heat_capacity = 2.0e6 } }   # TOML table allowed

[chemistry]                  # exclusive slot: the selector names the provider
scheme = "aq_chem"

[output]
variables = ["u", "v", "w", "theta", "qv", "t_surface", "budget_energy"]
cadence = "600s"

[checks]
energy_closure = { required = true, tolerance = 1e-10 }

[[domain]]
id = "parent"
grid = { nx = 192, ny = 192, nz = 53, dx = 31.25, stretch = "berlin_parent" }
static_driver = "input/berlin_parent_static.nc"
dynamic_driver = "input/berlin_20190629_dynamic.nc"
  [domain.urban_surface]
  enabled = false            # per-domain override: the parent uses tiles
  [domain.land_surface]
  scheme = "tiles"

[[domain]]
id = "child"
parent = "parent"
origin = [2500.0, 2500.0]
refinement = [2, 2, 4]
feedback = "two_way"         # the self-nesting default; "one_way" per child is allowed
static_driver = "input/berlin_mitte_static.nc"
```

Rules:
- **Single domain.** A run with one domain may write `[grid]` and the driver paths at top level instead of a `[[domain]]` block.
- **Complete per domain.** Each `[[domain]]` block contains all of that domain's module sections as `[domain.<section>]` tables, the parent included. Nothing is inherited from run level or from another domain. (The example above still shows module sections at run level; under this rule they move into each domain block.)
- **Resolved config.** It lists, for every domain, every component with every option (resolved), so a run is fully described by its `run.toml`.
- **Validation.** Unknown keys are errors. Slot conflicts, missing requirements and incompatibilities are reported with the section names involved.
- **Exclusive slots** (ONE-m, #476). Every exclusive slot is one section with a selector key: `[core] model`, `[timestep] scheme`, `[sgs] model`, `[surface] model`, `[microphysics] scheme`, `[advection] scheme`, and the urban site's `[pressure] solver`, `[momentum_advection] scheme`, `[chemistry] scheme = "none" | "aq_chem"` and the rest (`Urban.URBAN_SLOT_SECTIONS`). The section holds the chosen provider's options and no other's: an option of another provider is an error naming the providers that have it, and the resolved config shows the provider with all its options. A selector set at a higher level (`--set`, a `[domain.<section>]` block, `with`) that names another provider replaces it: the lower level's options are dropped and the new provider starts from its defaults. A section the model and the urban site share offers each domain the providers of its kind. There are no `enabled` pairs for a choice: `enabled` is for additive components.
- **Self-contained.** There is no `extends` and no inheritance. Every shipped TOML contains every option it relies on, apart from the model defaults. A change that must reach several variants is made in each file, and the round-trip tests show the diffs between variants explicitly.

### Input

- **TOML** holds:
  - scalars and switches;
  - vertical profiles;
  - a named, documented set of analytic initial fields (`taylor_green`, `random_perturbation`, `thermal_bubble`, …);
  - box obstacles;
  - small parameter tables.
- **The static driver** holds per-cell and per-object geometry and parameters: terrain, buildings (2-D and polygons), surface and ground classes, vegetation, and per-object parameters such as materials. Today's `STATIC_SCHEMA` and `STATIC_TILES_SCHEMA` are the starting point.
- **The dynamic driver** holds initial profiles and 3-D fields, boundary planes, surface time series, and the large-scale forcing tables (`ls_*`). These are read **generically**: the `large_scale_forcing` component consumes the `ls_*` tables for every case, not only SBL.

The reader and the format schema stay in Aeolus. Everything that **creates**
driver files moves to AeolusPrep.jl: OSM and GeoJSON, land-use classification,
soundings, ERA5/GRIB, parent output to boundary planes, and downloads. Its
GIS and downloader dependencies leave the model package.

### Output

- Each component's `outputs` adds to the domain's catalogue; the core adds the prognostic and standard derived fields.
- `[output]` selects variables per output kind (3-D fields, profiles, time series, surface maps, budgets) with cadence and averaging. Selecting a variable whose component is off is a config error.
- Default set: `u`, `v`, `w`, `theta` and the time series of domain means. Nothing else unless listed.
- Today's three writers (`simulation`, urban, SBL) become the one writer.

### Budgets and checks

- **One budget system in the core,** replacing the urban `Ledger`, the `AirHeatLedger` and the SBL momentum budget. Budgets cover energy (surface classes and air), water, mass, momentum and tracers. Components book terms through `budget!`; the stage-context measurement of the air-heat ledger is kept as the core mechanism for air heat.
- **Checks are named and published by components:**
  - budget closure;
  - nest mass and flux conservation;
  - a blow-up/sanity check;
  - reference comparisons. A reference comparison takes a reference data file (CSV/NetCDF) named in the config, so GABLS-style comparisons are configuration plus data, not code.
- **Configuring checks.** `[checks]` enables them and sets `required` and the tolerances.
- **Reporting.** Results go to `results.toml`. The process exits non-zero if any required check failed; advisory checks only warn. A run that blows up records the failed sanity check and exits non-zero.

### Nesting

- **A parent is just a domain** with its own components and drivers. `ParentCase`, `ParentConfig` and the `urban_parent` run type disappear. The parent's physics is whatever its own `[[domain]]` block enables. Its coarse surface (today's tiled mosaic) becomes data in the parent's own static driver, read by a surface component.
- **One nest engine.** The model's generic nest (`ModelNest`, built from the `[[domain]]` graph: one level, several levels or siblings, #328) becomes the only one. It gains what the Berlin nest needs today and the generic engine refuses:
  - adaptive parent steps with adaptive integer child substeps, and window retries;
  - per-domain budgets with the nest-exchange term (`[nesting] budget_exchange = "exchange_term"` by default, or `"off"`);
  - parent prefetch for one-way children, siblings in parallel;
  - synthetic inflow with parent-measured statistics (#268) and the two-way generator fix (#419);
  - mixed cores with a conservative density conversion;
  - validation of exchanged fields (the PALM rule plus the driver exception) and the subgrid-model handover.

  After that, `Nesting.NestedRun`'s separate Berlin path is deleted.

  *Status (NEST-p2, #482):* the engine has all of the above, tested on
  idealised nests ([Components](components.md), [Nested time
  coupling](../theory/nesting_time.md), [Two-way
  nesting](../theory/nesting_twoway.md)). Not yet: the parent prefetch at
  the root (the engine prefetches a one-way parent below the root; the
  root's next step is the model's next window, which the run loop reads
  first), a child's extra field from its own dynamic driver (a nested
  domain has no dynamic driver yet), and the Berlin switch: the Berlin
  nest moves onto `ModelNest`, and `NestedRun`'s Berlin path is deleted,
  in NEST-p3, once the parent's physics are components (NEST-p1, #481).
- **Feedback default.** Two-way for an Aeolus parent driving an Aeolus child; one-way per child is allowed. Driving through the dynamic driver from outside data is one-way.

How PALM does it, for comparison (Hellsten et al. 2021): every domain is the same PALM program with its own namelist and static driver; local modules (surface, buildings, radiation, canopy) are free per domain; exchanged fields must match in all domains; all domains share one time step (the minimum); domains run concurrently on separate MPI processes. Aeolus keeps per-domain step sizes instead of the common step, and closes budgets across the nest, which PALM does not.

### Tests

- **Physics modules** keep their module tests: analytic limits, conservation, benchmarks.
- **Components** get contract tests: build, a single step, `budget!` closure on a tiny domain, output catalogue consistency and checkpoint round-trip.
- **End-to-end configs:** a small set of tiny configs in `test/` (dry LES, moist convection, a synthetic city with every urban component, a two-domain nest) run in the fast tier with `[checks]` required.
- **Examples:** an enforcement test fails if any `.jl` in `examples/` builds or steps a model. Analysis scripts may only read output files.

### Repository layout after the refactor

```
src/          model core (Grids, Fields, dynamics, pressure, time stepping, Components, Model, Config,
              InputDrivers readers, Output, Budgets, Checks, Nesting) and physics modules
src/components/   one thin adapter per physics module
examples/     <case>/<case>.toml + input/*.nc (+ reference data) + optional output-analysis scripts
test/         module tests, component contract tests, end-to-end tiny configs
```

These disappear: `src/urban/` (the `UrbanCase` framework; gone since ONE-k,
#474: the urban components are `src/components/urban/`), `src/sbl/` (the
run type; its case physics becomes components), the run-driver registry, and
`src/urban/preprocess.jl` and `src/preprocess/` (moved to AeolusPrep.jl).

## Migration

The order of the epic:
1. Components and the pipeline in the core.
2. The schema.
3. Budgets, output and checks.
4. The urban couplings ported to components.
5. Nesting by domains.
6. The SBL and validation cases as configs.
7. The preprocessing split.
8. Examples, docs and the enforcement test.

Every step that is not a physics change must reproduce today's results bit for
bit, checked on the config digests, the Berlin smoke and the thread-invariance
harness. Re-captures are allowed only with a stated reason, and assertions are
never weakened.

**The urban site after ONE-k (#474).** The site is a `Model` of one domain
stepped by `step!(::Model)` and run by the model's run loop: the
`UrbanSite` is the domain's first component and its `:timestepper`
component takes the core step (`Components.takes_core_step`); its output
is the run's writer's (the site publishes its catalogue), its restart the
run's checkpoints, its options `site_options(cfg)` of the model's config.
`UrbanCase`, `CaseConfig`, the case run loop and the site's own output
and restart are gone; `Aeolus.Urban` lives in `src/components/urban/`.
Still open for the urban site: its nest parent (`ParentCase`, run by the
`:nesting` component and alone by the `urban_parent` run type) is not a
`[[domain]]` yet; and the site still builds its own core (Coriolis,
Smagorinsky and sponge in the core's constructor) instead of using the
model's components.

**The parent's physics as components (NEST-p1, #481).** The parent is a
`Domain` of components (`Urban.parent_domain`): the tiled mosaic surface
(`TiledUrbanSurface`, the `:surface` slot, on the tile form of the parent's
static driver, `STATIC_TILES_SCHEMA`), the prescribed surface flux of the
idealised parents, the column radiation of the plane-mean column
(`MeanColumnRadiation`, the `:radiation` slot), the large-scale forcing
(`LargeScale`: the geostrophic pressure gradient, the nudging with the
boundary-layer-dependent weight of a `BoundaryLayerTracker`, subsidence
and damping) and `[sgs] model = "tke"` with the wall distance of the
stretched grid and the tiles' u*. Each works on both cores. `ParentCase`
keeps the parent's ledgers around the components' entries for the nest
and the `urban_parent` run; the domain alone through the pipeline is that
run bit for bit. The parent's core keeps its built-in Coriolis term and
sponge, and its forcing the core's RK3 substage clock: moving either to
the configuration's form changes the summation order or the forcing
window, a re-capture for the issue that makes the parent a `[[domain]]`.

**One key per exclusive slot (ONE-m, #476).** The enable/disable pairs
that #472 and #473 left (`[pressure_fft] enabled = false` with
`[pressure_multigrid] enabled = true`, `[aq_chem]` with `[chem_off]`,
`[surface_layer] enabled` beside `[surface] model`) are gone: each slot is
one section with a selector (`Config.slot_section`), the model's and the
urban site's providers of a slot pooled in one section and offered by
domain kind (`[sgs] model = "tke"`: the model's TKE closure on a model
domain, the site's on an urban site). `[dynamics] core` became `[core]
model`, `[time] scheme` the run-level `[timestep] scheme`, `[sgs]
closure` `[sgs] model` (with a coefficient default per closure in place
of `c_s = "none"`), `[surface_layer] enabled` `[surface] model = "most"`.
Every shipped TOML is converted; the resolved configs mean what they
meant (the simulation digests, the Berlin fixture and the smoke-matrix
hashes are unchanged).

## Resolved questions (2026-10-09)

1. **`extends`:** removed. Every TOML is self-contained.
2. **Channel cases and the DNS a-priori study:** both dropped (2026-10-09). LES validation is to be done differently, possibly externally (#461). In place of the a-priori study the model ships an **a-posteriori** example, `examples/dns_vs_les/`: decaying isotropic turbulence from the named analytic initial field `isotropic_turbulence` (defined in spectral space, so the 64³ LES starts from exactly the large scales of the 256³ DNS), a DNS config and one self-contained LES config per SGS model, the resolved kinetic energy, the dissipation (molecular + SGS) and the energy spectrum E(k) as output catalogue entries, and a script that only reads the outputs and compares the decay, the dissipation and the spectra.
3. **Anelastic parity:** every component supports both cores, checkpointing included.
4. **Self-nesting default:** two-way. The Berlin two-way nest is still validated in #331, but the default does not wait for it. Two-way with synthetic inflow has been stable on idealised nests since #419.
