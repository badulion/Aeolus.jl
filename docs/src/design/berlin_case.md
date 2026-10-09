# The Berlin case

The Berlin (Alexanderplatz) run is the flagship example of Aeolus. Until
2026-10-06 every physics module in `src/` had to be exercised by it (the
rule of #123); the maintainer dropped that rule, and the example has no
test suite of its own (#320): the physics it couples is tested in the
module tests, and its end-to-end runs are `examples/urban/berlin_checks.jl`,
run by hand. Since #126 it is not a script but a small
framework, the `BerlinCase` module in `examples/urban/`, built so that
many couplings can be added independently and at the same time.

```julia
include("examples/urban/BerlinCase.jl")
using .BerlinCase

cfg  = CaseConfig(:alexanderplatz; nsteps = 1800)   # 1 h at dt = 2 s
case = build_case(cfg)
print(describe(case))                               # couplings and schedule
run_case!(case)                                     # NetCDF + log events (frames, timing, ledgers)
```

What the case prints is the installed logger's choice (#347; there is no
`verbose` setting and no `io` keyword): every line is a structured log
record with `subsystem = :berlin`, an `event` key and its values as
key-values (#338; the code is `examples/urban/core/diagnostics.jl`):

| event | level | key-values |
| :-- | :-- | :-- |
| `:built`, `:radiosity_scene`, `:init`, `:parent_built` | `Debug` | the setup summary: grid, buildings, solid / wall / glass cells, couplings; facets, view-factor reciprocity and closure, sky-view factor; the facade walls; the city-mean building |
| `:run_start` | `Info` | `nsteps`, `t_end`, `dt`, `variable_dt`, `records` |
| `:frame` | `Info` | each output record: `frame`, `wall` and the frame diagnostics (`hour`, `solar_elevation`, `Ts_min`/`Ts_max`, `sw_min`/`sw_max`, `theta1_min`/`theta1_max`, `qv1_min`, `dew_max`, `ql1_max`) |
| `:coupling` | `Info` | with each frame, one per coupling that declares diagnostics: `coupling` and its values |
| `:run_end` | `Info` | `steps`, `wall`, `records`, `output`, `size_mb`, `peak_uhi`, `max_div` |
| `:timing` | `Info` | one per stage entry, largest first: `stage`, `entry`, `seconds`, `ms_per_step` |
| `:ledger` | `Info` | `ledger = :surface` per class (`Rn`, `H`, `LE`, `G`, `dS`, `residual`, the extra terms, `closure`, W m⁻²); `:water` (`storage`, the sources, `residual`, kg m⁻²); `:heat` per layer (`storage`, `closure`, W m⁻²); `:heat_term` per attributed term (W m⁻² per layer) |
| `:blowup` | `Error` | `step` |

`with_logger(NullLogger()) do … end` silences a run. Under `aeolus run`
(below) the run folder's log takes everything from `[logging]
file_level` (`debug`) up and the console `[logging] console_level`
(`info`; `debug` adds the setup summary).

### Diagnostics

A coupling reports through the library's diagnostics registry
([Logging](../api/logging.md)): it declares its scalars with
`diagnostics(c, case)` and its vertical profiles with
`profile_diagnostics(c, case)`, as it declares `output_fields`. The same
declarations are the coupling's `:coupling` log event and its columns of
the run folder's `diagnostics.csv` and `diagnostics.nc`, so there is one
source of truth for what a coupling reports. `case_diagnostics(case)`
collects every scalar of the run:

* the dynamical core's (`ke_mean`, `max_div`, `theta_min`, …);
* the frame quantities (`Ts_min`, `sw_max`, …), the closure the steps
  run under `sgs.` (`sgs.nu_h_max`, `sgs.e_mean`, …);
* the ledgers under `ledger.`: `ledger.surface.<class>.<term>` (`Rn`,
  `H`, `LE`, `G`, `dS`, `closure`, W m⁻², mean over the run, for every
  class present and `domain`), `ledger.water.storage` /
  `ledger.water.residual` (kg m⁻²), `ledger.heat.storage` /
  `ledger.heat.closure` (W m⁻², whole column);
* the stage timing, `wall.<stage>.<entry>` (s, cumulative);
* each coupling's, prefixed with its name (`pressure_multigrid.cycles`,
  `timestep_adaptive.dt_mean`, `nesting.boundary_heat_top`, …).

Every wall-clock quantity is named `wall.…`: a coupling declares one as
`wall.<x>` and it is filed as `wall.<coupling>.<x>`, so a comparison of
two runs that must agree bit for bit (the configuration round trip of
#358) drops them all by the prefix, with the CSV's own `wall` column.
The rows carry the step's `dt`, what limited it (`dt_limiter`: the
binding CFL limit, `dt_min`, `dt_max`, `growth`, or the coupling whose
`limit_dt` shortened it) and its `retries`. The quantities that cost a
reduction or more are on the costly cadence (`[logging]
diagnostics_cadence`, 60 s simulated); counts and solver residuals
every step. The three that cost more than a step (the SGS a-priori
metrics, the Leighton ratio, the transition summary and its profiles)
are evaluated once per output record (`per_record`), as their report
lines were, and repeated in the rows in between. Measured on the
64 × 64 × 64 summer case (#338): a step 122 ms; every diagnostic and
profile evaluated afresh 193 ms, of which those three 168 ms; with them
once per record a costly row costs 16 ms, 0.4 % of the run at a 2 s
step and the 60 s cadence.

The flagship is configured in TOML (CFG-c, #358): the scenarios are the
shipped files `examples/urban/configs/berlin_<scenario>.toml`, run by the
`berlin` driver,

```bash
julia --project -m Aeolus run examples/urban/configs/berlin_summer.toml --set clock.nsteps=1800 --set output.cadence=600
```

and `examples/urban/alexanderplatz_diurnal.jl` is a three-line wrapper of
that command. The run folder (`runs/<date>_<name>_<id>/`) holds `run.toml`,
the log and `diagnostics.csv`; the NetCDF output still goes to
`examples/urban/output/alexanderplatz.nc` (`[output] path`). See
[the `berlin` driver](../api/config.md#The-berlin-driver) for the schema;
`CaseConfig(:summer)` in Julia loads the same file.

### Configuring with TOML

A variant of a scenario is a file that `extends` it and sets only what
differs, or a `--set` on the command line:

```toml
# my_berlin.toml
extends = "berlin_summer.toml"     # found next to this file, else in examples/urban/configs/

[run]
name = "summer_no_traffic_nox"

[physics]
sgs = "smagorinsky"                # a slot: one of its registered providers

[physics.options.aq_chem]
nox_street = 0.0                   # a coupling option, checked against its default_options
```

`julia --project -m Aeolus check my_berlin.toml` validates it,
`resolve` prints every option it resolves to, and `run` starts it. In
Julia, `CaseConfig(load_run_config("my_berlin.toml"))` gives the
`CaseConfig` that `build_case` takes. Every key, with its type, default,
units and range, is in the generated
[option reference of the `berlin` driver](../reference/options/berlin.md);
the guide to files, overrides, replays and `--resume` is
[Run configuration](run_configuration.md). The run's `run.toml` holds the
resolved config, so `aeolus run <folder>/run.toml` replays the run bit for
bit.

**A coupling's `default_options` is its option table.** The schema of
`[physics.options.<name>]` is generated from the coupling's
`default_options` `NamedTuple`: the keys are closed (an unknown key is an
error that lists the coupling's options), each value must have its
default's type (a `Symbol` is a string, a tuple an array of its length,
a `NamedTuple` a sub-table). The option reference documents the table
(`berlin_options_reference`), one row per key with its TOML form and its
default, generated from the same `NamedTuple`; the meaning of each key is
documented on the coupling's docstring. Adding a key to `default_options`
is all it takes to make it configurable from TOML and listed in the
reference; there is no separate schema to edit. `[physics]` gets one key per
slot (allowed values: the registered providers) and one per additive
coupling, and `[cadence]` one per named period a coupling declares.

## Status of the epic (#125)

The work is tracked in the milestone "Berlin — full-physics flagship"
under the epic #125. On master `915bd49` (after PRs #161 and #166) 10 of
its 29 issues are closed and 19 are open:

* **Wave 0, done:** the refactor (#126) and the smoke / coverage /
  ledger tests (#127).
* **Wave 1, 7 of 10 merged:** shade / class map (#128), surface exchange
  and soil (#129), column radiation and cloud shadow (#130), reference
  state and moist buoyancy (#131), numerics slots (#132), time
  integration and core options (#133, PR #161; the adaptive stepper
  became the default in PR #166), chemistry and deposition (#135).
  Open: Era I add-ons (#134), the Monte-Carlo RT diagnostic (#136),
  visibility and fog feedback (#137).
* **Wave 2:** the GMRES facet solve (#138) is merged; the per-facet
  walls (#139), park land column (#140), water surfaces (#141), aerosol
  optics (#142) and P3 slot (#143) are open.
* **Waves 3–4 and the solver issues** (#144–#153) are open.

None of this is a roadmap phase: the Era VI and Era VII groundwork it
delivered (the dynamic Smagorinsky in `Operators`, the reduced
chemistry in `src/Chemistry.jl`) is recorded under those eras in the
long-range roadmap, without marking a phase shipped.

## Layout

```
examples/urban/
  BerlinCase.jl        module: includes core/*, then every couplings/*.jl in sorted order
  core/
    config.jl          CaseConfig — grid, window, clock, cadences (s), physics/options
    scenarios.jl       CaseConfig(:summer) etc.: loads configs/berlin_<scenario>.toml
    config_schema.jl   the `berlin` run driver: TOML schema (defaults), RunConfig → CaseConfig
    registry.jl        AbstractCoupling, the contract, Cadence, register!, selection
    geometry.jl        build_geometry: the static driver (or the OSM file in memory) → mask,
                       classes, radiosity scene, view factors; the readers, classes and
                       rasterisation are Aeolus.InputDrivers (src/preprocess/, CFG-f #361)
    bus.jl             SurfaceBus (incl. the per-column surfaces), SurfaceParams, surface factories
    tracers.jl         tracer registry (ensure_tracer!)
    ledger.jl          energy/water budget ledger (surface energy per class, water in the air)
    reference_state.jl hydrostatic reference p̄, ρ̄, T̄, Π̄ (CaseReference)
    open_boundary.jl   cfg.lateral (:periodic / :open), the open faces' data and heat (#262)
    output.jl          NetCDF output driven by the registry
    case.jl            Case, build_case
    stages.jl          the stage loop: step_case!, run_case!
    diagnostics.jl     the case's diagnostics and log events (case_diagnostics, log_frame, log_run_end)
  couplings/           one file per coupling, each ending in register!(Type)
  configs/             berlin_summer.toml (every option), _winter, _smoke, _mitte_sample
  alexanderplatz_diurnal.jl              the run (a wrapper of `aeolus run configs/berlin_summer.toml`)
  alexanderplatz_animation.jl            renderer (reads the NetCDF only)
  alexanderplatz_surface_diagnostics.jl  per-class diurnal summary (NetCDF only)
```

Adding a coupling means adding one file under `couplings/`. No shared
file is edited: the file is picked up by the directory scan, registers
itself, and the registry decides from the config whether it runs.

## The coupling contract

A coupling is a struct `<: AbstractCoupling` plus methods of the
contract below for its type. Only `coupling_name` and the `run_stage!`
methods of the declared stages are required; everything else has a
default.

| method | meaning |
| :-- | :-- |
| `coupling_name(::Type{C})` | unique `Symbol`; key in `cfg.physics` / `cfg.options` |
| `slot(::Type{C})` | `:addon` (additive, on/off) or an exclusive slot such as `:microphysics` |
| `provider(::Type{C})` | the value of `cfg.physics[slot]` that selects `C` |
| `is_default(::Type{C})` | default provider of its slot; for an addon, on by default |
| `modules_used(::Type{C})` | Aeolus submodules it runs |
| `default_options(::Type{C})` | `NamedTuple`, merged with `cfg.options[name]` and passed to `build`; also the TOML schema of `[physics.options.<name>]` (closed keys, each value of its default's type) |
| `incompatible(::Type{C})` | `slot => provider` pairs or coupling names it cannot run with |
| `stages(::Type{C})` | `((stage, order, Cadence), …)`, at most one entry per stage; a `Symbol` period is a `[cadence]` key |
| `cadence_keys(::Type{C})` | named cadences it reads beyond its stages' (`[cadence]` keys; default `()`) |
| `build(::Type{C}, case, opts)` | construct the instance (default `C()`) |
| `init!(c, case)` | after every coupling is built, in registry order |
| `run_stage!(c, case, ::Val{S}, dt)` | the work of stage `S` |
| `output_fields(c, case)` | `OutputField`s added to the NetCDF file |
| `budget!(c, case, ledger)` | this step's surface-energy and water terms (see [the ledger](berlin_ledgers.md#The-budget-ledger)) |
| `diagnostics(c, case)` | `Vector{Diagnostic}`: the scalars it reports, in `diagnostics.csv` / `.nc` and its `:coupling` log event with each output record (see [Diagnostics](#Diagnostics)) |
| `profile_diagnostics(c, case)` | `Vector{ProfileDiagnostic}`: its vertical profiles, in `diagnostics.nc` |
| `smoke_variants(::Type{C})` | extra smoke-run configurations, a tuple of `NamedTuple`s (see [the smoke runs](berlin_ledgers.md#Smoke-test)) |
| `smoke_check(c, case)` | `Vector{Pair{String,Bool}}` of physics checks on the state after a smoke run (`examples/urban/berlin_checks.jl`) |
| `sim_options(::Type{C}, cfg, mesh, opts)` | keywords of the dynamical core's constructor (the `solver` of `BoussinesqSimulation`; `tol`, `maxit` of a sound-proof core), before the core is built (see [numerics slots](berlin_numerics.md#Numerics-slots)) |
| `step_options(c, case)` | keywords added to every dynamics step, `step!` or `step_imex!` (e.g. `momentum_advection`, `sgs`) |
| `incompatible_reason(::Type{C})` | why the `incompatible` pairs fail, appended to the selection error |

Two slots drive the dynamics itself and have four more methods
(#133, see [Dynamical core and time integration](berlin_numerics.md#Dynamical-core-and-time-integration)):

| method | meaning |
| :-- | :-- |
| `build_core(::Type{C}, cfg, mesh, ref, simkw)` | slot `:core`: construct the dynamical-core simulation (`case.sim`) on the case's reference state `ref` with the merged `sim_options` keywords `simkw`; a keyword it does not take is an error |
| `next_dt(c, case)` | slot `:timestepper`: the length of the next step (default `cfg.dt`) |
| `advance_dynamics!(c, case, dt)` | slot `:timestepper`: advance `case.sim` by `dt` with the composed `:body_force` / `:post_predictor` hooks and the numerics slots' `case.step_kwargs` (default `dynamics_step!`: one `step!`) |
| `variable_dt(::Type{C})` | slot `:timestepper`: `true` if `next_dt` varies; `run_case!` then runs to the simulated time `nsteps · cfg.dt` |
| `stable_dt(c, case)` | slot `:timestepper`, `variable_dt` providers: the longest step the provider's stability limits (CFL, diffusion, buoyancy, Coriolis) and `dt_max` allow from the current flow, without the growth limit and without side effects (default `Inf`). `step_case!(case, dt)` throws an `ArgumentError` for a `dt` above it, before any stage runs (#392) |

A keyword from `sim_options` or `step_options` is never dropped: every
time stepper passes `case.step_kwargs` to the step it takes, and a core
or stepper that cannot take a provider's keyword declares that provider
`incompatible` (with `incompatible_reason`), so the selection fails
with the reason instead.

`selected_couplings(cfg)` checks the configuration: an unknown key, an
unknown provider, an additive switch used on a slot coupling, a slot
without exactly one default, and a declared incompatibility are all
errors rather than silently ignored.

**Cadences are in seconds.** `Cadence(600.0)` fires on the step that
crosses a multiple of 600 s, `floor(t_new/P) > floor(t_old/P)`;
`Cadence(:radiation; first = true)` names a period in `cfg.cadence` and
also fires on step 1. With `dt = 2 s` this is exactly the old
`n % 300 == 0 || n == 1` over a whole day. With an adaptive `dt` the entry still fires
once per period crossed, and the adaptive stepper caps `dt` at the
shortest `cfg.cadence` period so that no firing is skipped. The surface
exchange steps its surfaces (the soil columns of `ground_soil`
included) forward by its period `P` and gives the held fluxes to the
air; a step that runs past the end of a surface interval delivers them
only up to that end and carries the rest of the step into the next
interval (`air_delivery_time!`), so the air receives every interval's
fluxes for exactly `P` (smoke check, 1e-9 P; with a fixed `dt` dividing
`P` that is every step's `dt`, bit for bit). `buildings_citymean`
advances its zone over the *next* radiation interval, which it cannot
know under an adaptive `dt`; it uses the nominal `P`, so the zone and
the air it heats disagree by less than one step per interval (≤ 10 s of
600 s) and the difference does not accumulate. Every `budget!` books
with `ledger.dt`, the current step's length, never `cfg.dt`.

## The stage loop

One step is a fixed sequence of stages; within a stage the entries run
in the integer `order` their couplings declare.

| stage | entries (order) in the default case |
| :-- | :-- |
| `:clock` | local hour (core) |
| `:nest_parent` | `nesting` (10): opens a parent window when the child has reached the last one — the parent takes its step, the child's nest faces point at the new window (see [Nesting](berlin_nesting.md#Nesting-(#265))) |
| `:radiation` (600 s, and step 1) | `column_radiation` (5), `radiosity_sun` (10), `cloud_shadow` (15), `buildings_citymean` (20) |
| `:sw_modifiers` | `sw_modifiers` (10, tree shade) |
| `:surface_flux` | `surface_exchange` (10): the held H, E into the first air cell (step 1: the first surface step) |
| `:anthropogenic` | `buildings_citymean` (10), `traffic_heat` (20), `glass` (30) |
| `:chemistry` | `aq_chem` (10), `aq_aerosol` (20) |
| `:transport` | `scalar_advection` (10) |
| `:microphysics` | `micro_kessler` (10) |
| `:tracers` | `pollutant` (10) |
| `:fog_dew` | `fog_dew` (10) |
| dynamics: `advance_dynamics!` of the `:timestepper` provider (default one `step!`) with `:body_force` / `:post_predictor` | `nesting` (5: the two-way θ booking, a no-op one-way, #267), `geostrophic` (10), `moist_buoyancy` (20), `wall_model_facades` stencil-flux removal (40) / `ibm` (10), `wall_model_facades` facade stress (12), `surface_exchange` drag (15), `veg_drag` (20); the `step_options` of the numerics slots |
| `:surface_update` | `surface_exchange` (10: collects `bus.sensible`; every surface period steps all surfaces, soil included), `snow_legacy` (30), `aq_chem` r_a / v_d refresh (90, every 30 s) |
| `:column_tendency` | `column_radiation` (10) |
| `:nest_feedback` | `nesting` (10): closes the window when the child reaches its end (one-way: clock and statistics; `mode = :two_way`, #267: also the feedback onto the parent, see [Nesting](berlin_nesting.md#Two-way-option-(#267))) |
| `:sanity` | the core finiteness check |
| (ledger) | every coupling's `budget!`, and the change of water in the air (core) |
| `:output` (1800 s) | NetCDF record and log events (core), then coupling entries: `turbulence_diag` (10) |

The order was the order of the pre-refactor loop, floating-point order
included (#126). The surface exchange (#129) replaced the slab, whose
latent flux sat at the end of `:anthropogenic` and whose Ts clamp ran in
`:sanity` only to keep that order; see [Surface exchange](berlin_surface_radiation.md#Surface-exchange-(#129)).

`run_case!` logs the wall time of every entry (the `:timing` events and
the `wall.*` diagnostics). At
#126, in the 1 h flagship run (1800 steps, first-call compilation
included), the largest entry was the WENO-Z5 scalar transport, about
17 ms/step, ahead of `step!` itself (about 12.5 ms/step), then `fog_dew`
(6.1), `radiosity_sun` (5.5 averaged over all steps) and `soil_urban`
(2.8); everything else was below 1 ms/step (table in the #126 PR).
Since then `soil_urban` has been replaced by the surface exchange
(#129; its surface step costs 13.6–16.3 ms per step of the adaptive
stepper, remeasured on master `915bd49`) and the chemistry (#135) adds five advected species; see the
cost paragraphs of those sections.

The serial baseline before threading (#295, KA-c) removed the runtime
integer power from the WENO-Z / TENO weights, the plane copies from every
halo fill and the per-level slice copies from the dew, with no change to
any bit of the smoke matrix or of 12 flagship steps. Flagship, 1 thread,
ms/step over 40 steps after 4 warm-up steps (minimum of 3 runs, 1-min
load 10–20 on 24 cores; an earlier pair at load ≈ 1.5 in brackets):

| entry | before | after |
| :-- | --: | --: |
| `scalar_advection` | 84.9 (68.6) | 34.6 (25.7) |
| `step!` | 30.7 (23.5) | 28.6 (21.9) |
| `fog_dew` | 10.1 (8.5) | 6.3 (5.9) |
| whole step | 155.4 (128.3) | 99.7 (82.7) |

A step of the flagship now allocates 1.0 MB (30.4 MB before), the
scalar-advection stage 7 KB (13.3 MB). What remains of `fog_dew` is the
fog saturation adjustment (5.7 ms on its own: one exact saturation
humidity per cell), not the dew.

## Shared state

Couplings never keep a private copy of a quantity another coupling
needs.

* **`SurfaceBus`** (`case.bus`, per column): `Ts`, `sw`, `lw`, the
  extra surface loss `sensible` (energy other couplings take from or give
  to the surfaces this step — window sun, glazing transmission, dew heat;
  not the turbulent sensible flux), the dew reservoir `dew`, the clear-sky
  beam `sw_beam`, the per-class `params`, the last radiosity solutions
  `swsol`/`lwsol`, the sky longwave `L_sky` and the direct/diffuse
  shortwave `dni`/`dhi` the `:column_radiation` slot hands the radiosity
  solve (#130; `NaN` shortwave means "Bird"), and (#129) the per-column `surface`
  (`AbstractSurface`), the roof-aware air level `k_air`, the forcing each
  surface last saw (`T_air`, `q_air`, `U`, `z_ref`), the fluxes it
  returned (`Rn`, `H`, `LE`, `G`, `E`, `u_star`) and the `dew_mask`.
* **Surface factories.** A coupling that provides the surface of a
  class registers `register_surface_factory!(case, class, name, (case, site) -> surface)`
  in its `build`, where `site::SurfaceSite` carries the class, the
  initial skin temperature, the child column (`nothing` for a tile of a
  parent column) and an optional roughness override (take the roughness
  from `class_roughness(case, site)`). `class_surface(case, site)` is the
  one class-to-surface factory: the surface exchange builds every child
  column's surface with it in its `init!` (a `SlabLand` for a class
  nobody claims), and the parent's tiled surface (#249, below) every
  tile. This is how the park (#140), water (#141) and facet (#139)
  couplings replace a class's surface — in the child and the parent at
  once — without touching the surface exchange.
* **Tracers** (`case.tracers`): `ensure_tracer!(case, :qi; advect = true,
  positive = true, units = …, long_name = …)` creates or returns a
  cell-centred field; flags are OR-ed, so two couplings share one `ql`.
  The `:transport` stage advects every tracer with `advect = true` in
  registration order, and every tracer with `output = true` is written to
  the NetCDF file. The core registers `qv`, `ql` (advected) and `qr`.
  A tracer diffuses with the closure's `κt` unless it names its own
  diffusivity (`ensure_tracer!(…; diffusivity = case -> field)`): the SGS
  TKE `e` of the `:tke` provider diffuses with `2ν_t` (#182). Passive
  tracers (qv, ql, the chemistry species, the pollutant) never name one:
  they share θ's κ_t under every `:sgs` provider, AMD included (#234, see
  below).
* **Geometry** (`case.geom`): mask, surface classes, scene, view factors.
  Couplings may add classes at `init!` (the glass coupling does). The
  scene is built with `class_map = scene_class_map(sfc)`, so the ground
  facet of every open column has the material of its class (park, water,
  street; wall-tagged cells are street ground).
* **Reference state** (`case.ref`, see [below](#Reference-state-and-moist-buoyancy)):
  the hydrostatic `p̄`, `ρ̄`, `T̄`, `Π̄` per level. Every coupling that
  needs a pressure, converts a flux into a mixing-ratio or θ increment,
  or turns θ into T takes it from here (`ref_pressure`, `ref_density`,
  `ref_exner`, `surface_pressure`, `ref_ρcp`), never from a constant of
  its own.

**Modifying `bus.sw`.** The radiation provider writes `bus.sw` only at
the radiation cadence; the `:sw_modifiers` stage runs every step. A
modifier must therefore act on the unmodified field, not on its own
previous output. `sw_modifiers` keeps that field (`sw_base`), takes a
new one whenever `bus.sw` differs from what it wrote on its previous
step, and rebuilds `bus.sw = cloud(shade(sw_base))` each step. A second
modifier added later belongs in the same pattern. Before #128 the tree
shade and the cloud shadow multiplied `bus.sw` in place, so they
compounded between radiation updates.

The case is checkpointed whole (#269): `checkpoint_state(case)` is the
structural state of the built case (every coupling's state included, so
a new coupling needs no checkpoint method of its own), restored into a
fresh build by `aeolus run <folder> --resume`; see
[Checkpoint / restart](checkpoint_restart.md#The-Berlin-case).

## Initial state: the Lindenberg sounding (#247)

Until #247 the case started from an analytic profile, θ = 293 K up to
100 m and +3 K/km above, with ``q_v = 10\,{\rm g\,kg^{-1}}\,e^{-z/800\,{\rm m}}``
and 1013.25 hPa at the ground. #238 showed that this profile cannot cap
a convective boundary layer, and the nesting parent (epic #246) must
start from the same observed state as the child. The summer case now
starts from the radiosondes of the Lindenberg observatory (WMO 10393,
60 km south-east of Alexanderplatz).

**Case day.** The epic first named 20 June 2024. That day was overcast
(DWD hourly total cloud cover 7–8 oktas most hours at Lindenberg and
Berlin-Tempelhof; the 12 UTC sounding is saturated at 907 hPa), so by the
maintainer decision of 2026-10-05 the case moved to **29 June 2019**, a
clear day of the late-June 2019 heat wave: 0–2 oktas at Lindenberg and
Berlin-Brandenburg from 00 UTC on the 29th to 16 UTC on the 30th, and a
12 UTC dew-point depression of at least 6 K below 700 hPa. The day is a
Saturday; `case_weekday(cfg)` now feeds the weekday of `cfg.date` to the
building schedules (they assumed the Thursday of 20 June 2024 before).

**Data.** `examples/urban/data/fetch_lindenberg.jl` downloads the IGRA2
record of GMM00010393 from NOAA NCEI (public domain; Durre et al. 2018)
and writes `lindenberg_10393_20190629.csv`: the 00/06/12/18 UTC launches
of 29 June and the 00/06 UTC launches of 30 June, with pressure, reported
geopotential height, T, T_d, wind and the IGRA QC flags. Its header
records the URL, the download date, the station elevation (launch site
112 m, not the 97.7 m of the IGRA station list: with 97.7 m every launch
misses the reported heights by 11–19 m) and the case-day check. The
profiles are the GTS standard and significant levels, not the
full-resolution BUFR ascent.

**Profiles** (`src/large_scale_forcing/sounding.jl`). `read_sounding`
reads the CSV; `sounding_profiles(s; z_top)` returns θ, ``q_v``, u, v
and p as `ForcingTable`s on heights above the launch site and the
release times. Heights are hypsometric (virtual temperature from
`MoistThermodynamics`), θ uses its Exner function and ``q_v`` its
``e_s(T_d)``; no thermodynamics is defined in the reader. The reported
geopotential heights are used only as a check: below 3 km the computed
heights match them to RMS 0.8–4.3 m per launch, within the ±4.7 m that
the whole-hPa surface pressure of the GTS message allows.

**The case** (`core/reference_state.jl`). `initial_θ(cfg)` remains the
one definition of the initial θ: `forcing_value` of the θ table at the
case start, 06:00 CEST = 04:00 UTC, which lies between the 22:45 and
04:45 UTC releases (weight 0.875 on the later one). `initial_qv(cfg)`
does the same for ``q_v`` and replaces the inline exponential of
`build_case`. `case_surface_pressure(cfg)` reduces the launch-site
surface pressure from 112 m to the case's 34 m with the surface virtual
temperature. `build_reference` integrates from it, and so does every
consumer of the reference state, the nesting parent included. The
heights of the sounding are used as heights above the case's ground:
the profile is taken as horizontally homogeneous over 60 km and as
terrain-following over the 78 m elevation difference.

| z (m) | θ sounding (K) | θ former (K) | q_v sounding (g/kg) | q_v former (g/kg) |
| --: | --: | --: | --: | --: |
| 0 | 288.60 | 293.00 | 7.70 | 10.00 |
| 50 | 290.02 | 293.00 | 7.04 | 9.39 |
| 100 | 291.21 | 293.00 | 6.08 | 8.82 |
| 200 | 291.68 | 293.30 | 8.02 | 7.79 |
| 400 | 292.42 | 293.90 | 8.02 | 6.07 |
| 1000 | 294.07 | 295.70 | 7.12 | 2.87 |
| 1500 | 300.16 | 297.20 | 1.13 | 1.53 |

The ground pressure is 1018.4 hPa (former 1013.25 hPa). At 06 CEST the
profile has a surface-based inversion, so the parcel z_i
(`initial_inversion_height(cfg)`) is 0. The same profile interpolated to
14 CEST has a mixed layer of 1.18 km, capped by the inversion at
884–879 hPa. The smoke checks of the turbulence diagnostics and of the
AMD/Vreman/σ switch-off take the capping inversion from
`initial_inversion_height`, not from `cfg.θ_inversion_z`.

**Effect on the standalone run** (flagship 64 × 64 × 32, 200 m deep,
06:00 to 18:00 CEST on 29 June 2019, default couplings, adaptive step;
the only difference between the two runs is `sounding` against
`sounding = nothing`). Plane-mean θ of the fluid cells below 30 m:

| | 06:00 | 12:00 | 18:00 |
| :-- | --: | --: | --: |
| sounding | 289.06 K | 303.81 K | 313.89 K |
| former analytic profile | 293.00 K | 305.35 K | 315.22 K |
| difference | −3.94 K | −1.54 K | −1.33 K |

The colder and drier observed morning column stays 1.3 K cooler at
18:00. In both runs the 200 m box warms by about 25 K, and by 10:00 it
is mixed to the lid. The observed mixed layer is 1.2 km deep, so the
box cannot hold it: entrainment of free-tropospheric air through a
capping inversion is missing. Supplying that air is the job of the
nesting parent (#246). Each run took 1.7–1.8 h of wall time with 2
threads on a shared machine.

**Idealised profile.** `sounding = nothing` selects the former analytic
profile (`θ_surface`, `θ_inversion_z`, `θ_lapse`, `qv0`, `qv_scale`) and
101 325 Pa. `:winter` uses it, explicitly labelled as idealised, until a
winter sounding is added. `:smoke` uses it as well, because its checks
need a stable layer above an inversion inside a 100 m box. The smoke
matrix also runs "summer sounding, everything on", the smoke box started
from the sounding at noon, so the observed path is exercised by every
suite run.

**The boundary-layer height.** z_i comes from
`TurbulenceStability.boundary_layer_height` everywhere: soundings, the
CBL harness (`tke_cbl` of the `cbl` run type, formerly `test/tke_cbl_driver.jl`) and the nesting diagnostics. It is
the only z_i diagnostic in the package; see [Stability functions and the
gradient Richardson number](../theory/turbulence_stability.md).

## Reference state and moist buoyancy

The Boussinesq core carries neither pressure nor density, and before
#131 each coupling assumed its own: fog and dew an isothermal 8 km
scale height from 1013.25 hPa, the surface, building and traffic
fluxes and the ledger `ρ = 1` (`ρcp = 1005`). Since #131
`core/reference_state.jl` builds one hydrostatic profile per case,
`Aeolus.Anelastic.reference_from_profile` of the initial θ(z) (see the
initial-state section above) from
`case_surface_pressure(cfg)` (1018.4 hPa at the ground on the summer
case day since #247; 101 325 Pa for the idealised profile), with the dry-air constants
of `MoistThermodynamics` (so `T = θ Π̄` is the temperature the fog
adjustment and the saturation functions assume) and the `g` of the
dynamics. It is exactly (1e-12 Pa m⁻¹) in the discrete balance of the
anelastic core and 5e-8 from an RK4 integration of the continuous
hydrostatic equation. The isothermal profile it replaces was 0.17 % off
at the flagship's top level; the density is 1.20 kg m⁻³ at the ground
in summer, 1.30 in winter.

| consumer | uses |
| :-- | :-- |
| `fog_dew` | fog adjusted at `p̄_k` (`T = θ Π̄_k`); dew at the `p̄_k`, `ρ̄_k` of each column's first air cell `k_air` (#129; one pass over the columns since #295) |
| `surface_exchange` | forcing `T_air = θ Π̄_k`, `p̄_k`, `ρ̄_k` at the first air cell `k_air`; `ρ̄_k c_pd`, `ρ̄_k` in the conversion of H and E to θ, qv |
| `aq_chem` | `T = θ̄ Π̄_k` and `p̄_k` for the rate constants; its r_a solve uses `θ_s = Ts/Π̄_k` at the first air cell |
| `column_radiation` | surface pressure `surface_pressure(case)` (`case.ref.p_surface`) |
| `cloud_shadow` | `ρ̄_k` for the cloud water path when no density is given |
| `buildings_citymean`, `traffic_heat` | `ρ̄_k c_pd`, `ρ̄_k` for the release into level k |
| the sound-proof cores (#133) | `case.ref.state` itself as the base state: the projection, buoyancy and advection weights are this ρ̄, θ̄ (`build_case` builds the reference before the core) |
| the ledger | `ledger.ρ = ρ̄`: the water in the air is integrated with the density the couplings use, so the water ledger still closes |
| `moist_buoyancy` | `Π̄_1`, `p̄_1` for the RH and dew point it reports |

**Moist buoyancy** (`couplings/moist_buoyancy.jl`, on by default). The
core adds only the dry buoyancy `g (θ − θ₀)/θ₀` (`sim.moisture = false`:
the water lives in the tracer registry). In `:body_force` the coupling
adds the rest of the density-potential-temperature buoyancy,

    g (θ_ρ − θ)′ / θ₀,     θ_ρ = θ (1 + q_v/ε − q_t) ≈ θ (1 + 0.608 q_v − q_c),

from `MoistThermodynamics.density_temperature`, at every interior w face
(the two adjacent centres averaged, as the core does for θ). The prime
is the deviation from the mean over the fluid cells of the level: a
horizontally uniform force is a pure pressure gradient in the
incompressible core, so subtracting it only changes the diagnosed
pressure. The condensate `q_c` is every tracer the ledger counts as
water other than `qv` (`ql` and `qr` today), so a microphysics provider
that calls `water_tracer!` has its species loaded without touching this
coupling. It writes `b_moist` (m s⁻²) to the NetCDF file and declares
diagnostics (rms and max of b′, max RH and min dew-point depression of
the lowest layer).

In the flagship configuration (1 h from 06:00, no fog) it costs 0.86
ms/step. At the end of the hour the rms moist buoyancy is 8.8e-4 m s⁻²
(max 1.8e-2, the buoyancy of 0.5 K), a quarter of the rms dry buoyancy
perturbation (3.6e-3). In the same hour run without the coupling, the
moist buoyancy the state would have is twice as large (rms 1.8e-3,
max 3.4e-2): with the term on, the light, humid air the buildings and
surfaces release rises and mixes instead of accumulating as anomalies.
The two runs also differ in the minimum level-1 humidity (4.8 g/kg with,
3.3 without) and in dew deposition (1.6e-3 against 2.9e-3 kg m⁻²), but
after an hour of turbulence part of that is the divergence of two
chaotic runs, not the term alone.

## Adding a coupling

This is the whole procedure. It touches one new file and the PR body; no shared file of the case is edited.

**1. Put the physics in `src/`.** A coupling only wires a src
implementation into the 3-D run. If the physics is missing, add it to
the right src module with its own tests first. Never copy it into the
example.

**2. Write `examples/urban/couplings/<name>.jl`.** One struct, the
contract methods it needs, and `register!` as the last line:

```julia
# P3 microphysics (Aeolus.P3Microphysics) in the :microphysics slot.

struct MicroP3{W} <: AbstractCoupling
    work::W
end

coupling_name(::Type{<:MicroP3}) = :micro_p3             # unique; key in cfg.options
slot(::Type{<:MicroP3})          = :microphysics         # or :addon
provider(::Type{<:MicroP3})      = :p3                   # cfg.physics[:microphysics] = :p3
is_default(::Type{<:MicroP3})    = false                 # an alternative: off by default
modules_used(::Type{<:MicroP3})  = (:P3Microphysics,)    # the Aeolus modules it runs
default_options(::Type{<:MicroP3}) = (skip_dry = true,)
stages(::Type{<:MicroP3}) = ((:microphysics, 10, EVERY_STEP),)

function build(::Type{<:MicroP3}, case, o)
    ensure_tracer!(case, :qi; advect = true, positive = true, units = "kg kg-1",
                   long_name = "ice mass")               # transport picks it up
    return MicroP3(...)
end
init!(c::MicroP3, case) = water_tracer!(case.ledger, :qi) # count it as water

function run_stage!(c::MicroP3, case, ::Val{:microphysics}, dt)
    ...                                                  # calls into Aeolus.P3Microphysics
end

budget!(c::MicroP3, case, l) = add_water!(l, :precipitation, -c.surface_precip_step)

smoke_check(c::MicroP3, case) =
    ["ice ≥ 0" => minimum(interior_view(tracer(case, :qi))) >= 0,
     "precipitation ≥ 0" => c.surface_precip_step >= 0]

register!(MicroP3)
```

The file is picked up by the directory scan; include order is the
sorted file name, which is also the registry order (and the tie-break
between entries of the same stage and order).

* **On by default.** An additive coupling is on unless `is_default`
  says otherwise; a slot provider is the slot's default only if it is
  the one with `is_default = true`. Coupling new physics into the
  example is welcome where it makes sense, not required (the rule of
  #123 was dropped on 2026-10-06); quote the cost per step from
  the `:timing` events in the PR.
* **Cadence.** An expensive stage can run every `P` seconds:
  `Cadence(600.0)`, or `Cadence(:mykey)` with a default in
  `cfg.cadence` (add the key through the scenario or your option; a
  missing key is an error). Stages inside `step!` (`:body_force`,
  `:post_predictor`) must run every step.
* **Shared state** goes through the bus, the tracer registry and the
  geometry, never a private copy (see above). If you need a bus field
  that does not exist, add it to `core/bus.jl` additively and say so
  in the PR.
* **Replacing a coupling.** Delete the old file (the issue says which)
  and move its `budget!` terms and checks over. The ledger closure of
  the smoke runs fails if a term that changes a surface temperature or
  the water in the air is lost.

**3. Report to the ledger.** If the coupling changes a surface
temperature, adds or removes water from the air, or clips anything,
book it in `budget!` (next section). The smoke runs of
`examples/urban/berlin_checks.jl` assert the closure, so a missing term
fails them.

**4. Give it smoke coverage.** The smoke runs already run every
additive coupling and one configuration per non-default provider, so
most couplings need no `smoke_variants`. Add one for a configuration
that is not reached that way, for example an option that changes the
code path:

```julia
smoke_variants(::Type{<:MicroP3}) =
    ((label = "sgs cloud on", physics = Dict(:microphysics => :p3),
      options = Dict(:micro_p3 => (sgs_cloud = true,))),)
```

Each variant is a `NamedTuple`: `label`, then `physics` and `options`
(merged into the everything-on configuration), then any `CaseConfig`
field (`nsteps = 30`, `size = …`). Write `smoke_check`s that assert
physics: a conservation or a budget, a bound that follows from the
equations (a maximum principle, positivity, a range set by the
thermostat), a sign that follows from the physics. "It ran" is already
asserted for every coupling.

**5. Run and quote.** `julia --project=. examples/urban/berlin_checks.jl smoke`
locally (about 3.5 min); the physics itself is tested in its module's
tests, which the fast tier runs.
In the PR body quote one Berlin diagnostic — a ledger term is often the
natural one — the configuration it was measured in, and the cost from
the `:timing` events.

## Validation of the refactor (#126)

The refactor had to be bit-for-bit. On a 1 h run (1800 steps,
two records) every NetCDF variable of the refactored case is identical
(`==`) to the pre-refactor script, and the printed frame lines match
apart from the wall-clock column.

**The pre-refactor script is not reproducible run to run.** Two 1 h runs
of the *same* old script, one with a freshly measured FFTW plan and one
with an imported plan, differed by up to 7.5e-9 m/s in `w`, 4.7e-10 m/s
in `v` and 6.0e-8 in the pollutant (Float32 output; θ, moisture and the
surface fields happened to agree). The cause is `FFTW.MEASURE` in the cosine-FFT
pressure solver: the planner times candidate algorithms and may pick a
different one in each process, with different round-off, which the flow
then amplifies. With the same FFTW wisdom imported into both processes
(`FFTW.import_wisdom`) the planner is deterministic, and old and new
cases then agree exactly in every variable. Bit-for-bit comparisons of
the Berlin case have to fix the FFTW plan this way.

## Known defects kept by the refactor

These are deliberately unchanged so the refactor could be checked
bit-for-bit; each is labelled `KNOWN` in its file and belongs to a
later issue.

| defect | file | fixed by |
| :-- | :-- | :-- |
| tree shade and cloud shadow multiply `bus.sw` in place every step, so they compound between radiation updates | `sw_modifiers.jl` | #128 (fixed) |
| no `class_map` in the radiosity scene (park/water facets get street material) | `core/geometry.jl` | #128 (fixed) |
| slab: constant C_h and U, flux clamps, roof fluxes into solid k = 1, no surface drag | `surface_slab.jl`, `fog_dew.jl` | #129 (fixed: `surface_exchange.jl`, `ground_soil.jl`) |
| `qr` sedimented but not advected; qv clamp | `micro_kessler.jl` | #143 |
| glass is a synthetic roof mask | `glass.jl` | #144 |
| toy aerosol: not advected or output, zero with no source; its PM deposition table | `aq_aerosol.jl` (table in `aq_chem.jl`) | #147 |
| OSM `&&`/`||` precedence; patches overwrite roofs | `src/preprocess/osm.jl`, `src/preprocess/city.jl` (moved by CFG-f #361, kept bit for bit) | #141 |

**Found during #126, fixed in #132: stale velocity halos in the scalar
transport.**
`step!` corrects the interior velocity after the pressure solve but
leaves the halos at the predictor values. The Berlin case advects its
scalars (qv, ql with WENO-Z5, the pollutant with centred differences)
at the start of the next step, on those halos, so the velocity it sees
is not divergence-free across the periodic seam and the flux form stops
conserving: in the `:smoke` window after four steps the transport stage
changes the domain total of qv by −4.4e-5 and of ql by −3.5e-5 *per
step*. With the halos refreshed (`fill_halos!` on u, v, w) the same
stage conserves both to 1.5e-15 and 3.8e-15. The fix changes the
answer, so it was not in #126; the Berlin tests recorded the leak as
`@test_broken` and asserted the conservation with fresh halos. The
scalar-advection issue (#132) owned it.

The ledger (#127) measures the same leak in water. On **step 1** it is
much larger: the velocity halos have never been filled when the
transport first runs (`set!` fills only the interior), and in the
`:smoke` window the first transport step *adds* 7.5e-3 kg m⁻² of water,
0.8 % of the water in the air (9.4e-1 kg m⁻²); every later step removes
about 3.5e-5 kg m⁻². With the halos refreshed the same steps change the
water by at most 1.3e-15 kg m⁻². Over the 1 h flagship run the
transport term is −2.70e-4 kg m⁻². **Fixed in #132**: the
`scalar_advection` coupling refreshes the u, v, w halos before it
advects (`refresh_halos = true`), which also gives the pollutant, which
advects later in the step, fresh halos. The water change of the
transport stage is now ~1e-19 kg m⁻² per run (smoke) and +6.9e-19 kg m⁻²
over the flagship hour; both `@test_broken` became `@test`, and the
Berlin tests kept the mechanism tested with `refresh_halos = false`
until they were removed (#320; the smoke runs of
`examples/urban/berlin_checks.jl` still assert the water closure of the
transport).
