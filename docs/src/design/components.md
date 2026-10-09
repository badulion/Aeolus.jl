# Components and the stage pipeline

Status: **built in ONE-a (#450)**, the first step of the single-model
epic, with the configuration schema of ONE-b (#451), on both dynamical
cores since ONE-j (#460). The target is [One model](single_model.md); this
page says what the code does today, and where it still differs from the
target.

## What a step does

`step!(model, dt)` steps every domain of the [`Model`](@ref
Aeolus.Components.Model) (one today; nesting by domains is ONE-f, #455). For one
[`Domain`](@ref Aeolus.Components.Domain):

1. the domain's drivers advance to the step (the open faces' boundary
   window, `advance_feed!`);
2. the components of the stages before the core run, in
   [`PIPELINE`](@ref Aeolus.Components.PIPELINE) order: `:clock`,
   `:nest_sync`, `:radiation`, `:sw_modifiers`, `:surface_flux`,
   `:sources`, `:chemistry`, `:microphysics`, `:tracers`;
3. the core steps — `step!` (Euler) or `step_rk3!` (SSP-RK3), on either
   core — and calls the
   components of stages 6–8 from inside every (sub)step:

   | stage | where the core calls it |
   | :-- | :-- |
   | 6 `:tendencies` | after its own tendencies: advection and diffusion, the `:sgs` closure, buoyancy |
   | 7 `:body_force` | right after stage 6, before the predictor |
   | 8 `:post_predictor` | after the predictor `u* = u + Δt F`, before the projection |

   Stages 6 and 7 run in the core's `body_force!` hook, stage 8 in its
   `post_predictor!` hook (callable structs, so the calls are static).
   When no component attaches to them the hooks are `nothing`, and a
   Boussinesq Euler step without SGS keeps the cache-blocked DNS core of
   SC-e (`test/components_pipeline.jl` checks that its sweep buffers are
   used). Stages 9 (`:open_boundaries`) and 10 (`:projection`) are the
   core's own operations; no component may attach to them yet;
4. the components of the stages after the core run: `:surface_update`,
   `:column_tendency`, `:nest_feedback`, `:budgets`, `:checks`, `:output`.

A stage outside the core may have a cadence (`Stage(:radiation; every =
60)`): it runs in the step that reaches or crosses a multiple of
`every`. Within a stage, components run by slot first (`:advection`,
`:sgs`, `:microphysics`, `:radiation`, `:surface`), then by
`component_name`; the domain stores them in that order, as a concretely
typed tuple, and the pipeline walks the tuple recursively, so every
`run_stage!` call is statically dispatched.

An adaptive step (`step_adaptive!(model)`, a domain with `cfl`) picks
`dt` from the core's CFL limits and every component's
[`limit_dt`](@ref Aeolus.Components.limit_dt) (compared right after the
core's own Coriolis limit, so a Coriolis component reproduces the core's
pick and its `:coriolis` limiter), runs the stages before the core with
it, and lets `TimeIntegration.step_adaptive!` take it, retrying with half
the step if the guard fails. The stages before the core are not repeated
on a retry.

## The core's components

| component | slot | section | stage | delegates to |
| :-- | :-- | :-- | :-- | :-- |
| [`CentredAdvection`](@ref Aeolus.Components.CentredAdvection) | `:advection` | `[advection]` (no options) | called by the core in stage 6 | `Operators.momentum_advect!` (Boussinesq, fused with diffusion), `weighted_momentum_advect!` (anelastic) |
| [`SGS`](@ref Aeolus.Components.SGS) | `:sgs` | `[sgs]` `closure` (default `smagorinsky`), `c_s`, `pr_t`, `stability` | called by the core in stage 6 | the closure of `Components.build_closure` |
| [`Coriolis`](@ref Aeolus.Components.Coriolis) | — | `[coriolis]` `f` | `:tendencies` | `Operators.add_coriolis!` |
| [`SurfaceThetaFlux`](@ref Aeolus.Components.SurfaceThetaFlux) | — | `[surface_theta_flux]` `bottom`, `top` | `:tendencies` | `Operators.add_surface_flux!` |
| [`Sponge`](@ref Aeolus.Components.Sponge) | — | `[sponge]` `alpha_max`, `z_start` | `:body_force` | `Simulations.add_velocity_sponge!` → `Operators.add_sponge!` |

Every domain has an `:advection` provider; `[sgs] closure = "none"`
enables no `:sgs` component. Each component has its own per-domain
section of the model's configuration, generated from its
[`schema`](@ref Aeolus.Components.schema)
([`COMPONENTS`](@ref Aeolus.SimulationDriver.COMPONENTS), ONE-b #451), and
is switched on by it ([`enabled`](@ref Aeolus.Components.enabled)`(C,
opts)`, from the options alone). Before anything is built, the enabled
component types of every domain are checked
([`component_problems`](@ref Aeolus.Components.component_problems)): at
most one per slot, an `:advection` provider, and each component's
`requires` and `incompatible`, every problem keyed by the section it
concerns; [`validate_components`](@ref Aeolus.Components.validate_components)
adds the stage rules for the built instances. The schema, the domains and
the bare-config dry LES are on [Run configuration](run_configuration.md).

The model builds its core with the dynamics only
(viscosity, buoyancy, moisture, boundary conditions) and hands SGS,
advection, Coriolis, the sponge and the θ fluxes to these components. The
plain Smagorinsky closure is therefore the `:sgs` component's
`StaticSmagorinsky`, no longer the core's built-in `C_s`; its runs now
report the `sgs.nu_h_max`, `sgs.nu_v_max` and `sgs.kappa_h_max`
diagnostics every other closure already reported.

The couplings EX-d (#441) put around the model's step —
`[forcing]` radiative cooling, the `[surface]` slab or flux map,
`[microphysics]` Kessler, `[tracer]`, `[particles]` and `[obstacles]` —
are one component of the domain for now,
[`SimulationCouplings`](@ref Aeolus.SimulationDriver.SimulationCouplings),
which runs each piece where it acted before the pipeline: the slab's
flux of the step and the flux map in `:surface_flux`, the slab's floor
flux in `:tendencies` (it used to set the core's `θ_flux_bottom`), the
obstacles' IBM penalty in `:body_force`, and cooling, the slab's energy
balance, Kessler, the tracer and the particles right after the core, in
`:surface_update`. Splitting them into components of their own puts
Kessler and the tracer in stage 5, before the core, as the design has
them: a change of the splitting order, so of the results, which is left
to the step that ports them.

## Both cores (ONE-j, #460)

Every component supports both dynamical cores, and so does the
infrastructure around them:

| | Boussinesq | anelastic (and pseudo-incompressible) |
| :-- | :-- | :-- |
| schemes (`Domain` `scheme`) | `:euler`, `:rk3` | `:euler`, `:rk3` (SSP-RK3 with the weighted projection per substage) |
| adaptive step (`cfl`) | `TimeIntegration.step_adaptive!` | the same driver: the core's `adaptive_dt` with the components' limits, its snapshot, and a guard on `∇·(ρ̄u)` |
| checkpoint / restart | `IO.checkpoint!`, `IO.restart!` | the same, with `u, v, w, p, θ, νt, κt` and the base state (a restart onto another is refused) |
| blowup guard | crash checkpoint of the last good state | the same |
| `state_fields` of a `simulation` run | the checkpoint fields | the checkpoint fields |

The components are thin adapters, so their port is a question of the
physics module they call. On the anelastic core the SGS fluxes (`SGS`)
are in the `ρ̄`-weighted flux form of its advection
(`Operators.flux_weights`; see [Sound-proof cores](../theory/anelastic.md)),
and so is the prescribed surface flux (`SurfaceThetaFlux`) as it was (the
base state's wall faces carry the adjacent cell's weight): `Σ ρ̄ θ ΔV`
changes only by the surface fluxes, to round-off, and the SGS terms
conserve `Σ ρ̄ uᵢ ΔV`.
`CentredAdvection` is the core's own advection (`weighted_momentum_advect!`
on the anelastic core); `Coriolis` and `Sponge` act on the velocity and are
the same on both cores.

`test/components_both_cores.jl` runs the contracts of every component on
both cores and both schemes, the checkpoint round trip on both cores with
Euler, RK3 and the adaptive step, and a dry CBL on both cores. Its numbers
(16³, the DCBL config): the θ budget closes to below `1e-11` on both cores
with every scheme; after 30 min the anelastic mixed layer has warmed more
than the Boussinesq one by the column-mass ratio `ρ̄_bot h / ∫₀ʰ ρ̄ dz`
within a factor of 1.5. The anelastic digests of the `simulation` variants
(`simulation_digests_anelastic.toml`) were re-captured on ONE-j: the
weighted SGS fluxes changed those with a closure, which
`test/components_pipeline.jl` asserts; the variant without one kept
master's digests bit for bit.

## Bit identity

Every model-config variant of `test/support/simulation_digests.jl`
— the four shipped configs plus overrides switching on each SGS closure,
Coriolis (also as the adaptive step's binding limit), the sponge, the top
θ flux, moisture, a stretched grid, the anelastic core, RK3 and the
adaptive stepper: 28 runs at 16³ — reproduces the state digests captured
on master `ead7b986` before the pipeline, bit for bit, and its
`diagnostics.csv` byte for byte apart from those three new columns
(`test/components_pipeline.jl`). So do the seven EX-d simulation configs
with their couplings, plus RK3 runs of the obstacles and the slab, against
digests captured on master `a22b57ba`. The same file runs the components
against the core's built-in terms, bit for bit, on both cores and with
Euler and RK3. The schema of ONE-b (#451) renamed the keys of every
config (`[time]`, the component sections) and changed defaults; the
shipped configs state the old defaults they relied on, and the digests
are reproduced unchanged.

## Budgets: the stage-context measurement

A domain measures its budgets only when they are switched on
(`Components.with_budgets`, which a run does when `[output] variables`
selects a `budget_*` variable). With the budgets off the pipeline is the
plain loop above: nothing is measured, and the hook-free fast path holds.
With them on, every stage entry is an attribution context of every
measured quantity. This is the mechanism of the urban case's air-heat
ledger (#253), which is now `Aeolus.BudgetSystem.StageBudget` (ONE-c,
#452):

| quantity | measured | weight per level |
| :-- | :-- | :-- |
| `:energy` | θ: the heat of the air ``Σ ρ̄ c_p Π̄ θ ΔV / A`` (J m⁻²) | ``ρ̄ c_p Π̄ Δz / (N_x N_y)`` |
| `:water` | qv and the water the components carry (ql, qr) | ``ρ̄ Δz / (N_x N_y)`` |
| `:momentum_x`, `:momentum_y` | u, v | ``ρ̄ Δz / (N_x N_y)`` |
| `:tracer` | the components' tracer | ``ρ̄ Δz / (N_x N_y)`` |

ρ̄, Π̄ and c_p are the anelastic base state. On the Boussinesq core they
are its constant density ``ρ_0 = p_{ref} / (R_d θ_0)`` with ``Π̄ = 1``,
because the Boussinesq transport conserves ``Σ θ``. Solid cells (the
obstacles) are not air.

- **Outside the core** (stages 1–5 and 11–13), the budget measures the
  quantity when a component's entry opens and when it closes, and books
  the difference under `(stage, component)`.
- **Inside the core** (stages 6–8), a component adds to the core's
  *tendencies*, and the predictor applies them. The entry is therefore
  measured on the tendency array and scaled by the substage's weight
  times Δt (1 for Euler; 1/6, 1/6, 2/3 for the SSP-RK3 substages). A
  `:post_predictor` entry changes the predicted state, so it is measured
  on the state and scaled by the weight.
- **The core step** is one context (`:dynamics`). Whatever the core does
  beyond the components inside it is split into `(:dynamics, :transport)`
  (what the transport conserves; over the whole column it is the
  numerics' residual) and `(:dynamics, :redistribution)` (what moved
  between levels of different weight). Exchange with solid cells is
  booked as `(:dynamics, :solid_exchange)`.
- **Retries.** An adaptive step can reject an attempt and retry it with
  half the step. What the in-core entries booked is held by the dynamics'
  context until the step is taken. The domain notices a retry when the
  first in-core hook runs again at the step's start time, and drops that
  booking.
- **Closure.** Every change is attributed, so each budget closes by
  construction: `storage − Σ terms` is round-off. A change outside every
  context is booked as `(:core, :unattributed)`.

After the stages of 11–12 and `:budgets` (13) have run, each component's
`budget!(c, domain, budgets)` books the terms it *reports*: surface energy
per class, water from outside the air, free-form terms. The step then
closes and `:checks` and `:output` run.

Measured (`test/budgets_output.jl`, 6 × 5 × 8 box):

- The prescribed θ fluxes are attributed to `(:tendencies,
  :surface_theta_flux)` as ``ρ_0 c_p (F_{bottom} − F_{top}) t`` to rtol
  1e-12. This holds with Euler, RK3 and the adaptive RK3 step.
- Coriolis is the whole change of a uniform flow's momentum (rtol 1e-12).
- The sponge's momentum term is zero below `z_start`.
- An adaptive step whose first attempt throws books its in-core heating
  once.
- Every closure is below 1e-12 of the largest term.
- The state is bit for bit the same with the budgets on or off.

## Output: catalogues and the one writer

Every component publishes an output catalogue (`outputs(c)`: a list of
`Aeolus.Output.OutputVariable`s, named as `output_names(C, opts)` declares
for its type). The core publishes `Components.CORE_OUTPUTS`:

- u, v, w, p, θ, qv as 3-D fields;
- the horizontal-mean profiles;
- the domain means as time series;
- the `budget_*` groups.

`[output] variables` selects from the catalogues of the core and the
enabled components. The configuration is checked against the declared
names, so selecting a variable whose component is off is an error. The
default is `u`, `v`, `w`, `theta` and the time series of the domain
means.

One writer, `Aeolus.Output.OutputWriter`, writes the selection into one
NetCDF-4 file. Records (3-D, maps, profiles) go out every `cadence`, and
the time series every `series_cadence`. Each value is either the value at
the record or its mean over the record's interval (`averaging`). The
urban case and the SBL run type write through the same writer: the urban
couplings' `output_fields` are catalogues of `OutputVariable`s, and the
SBL time series are the time-series stream of `output/output.nc`.

## Checks

Every component publishes its checks (`checks(c)`: `Aeolus.Checks.Check`s,
named as `check_names(C)` declares for its type), and the core publishes
`Components.CORE_CHECKS`: the sanity check `finite` and the closure check
of each measured budget. The model's `[checks]` section has one entry per
published name, `name = {required, tolerance}` (an empty table is off),
and the reference comparisons `[checks.reference.<quantity>]` against
`reference_file`. The configuration is validated against the declared
names and against the requirements of a closure's budget. A closure check
turns its budget on.

The checks run in stage 13, after the budgets close and before the
output: each step, a check that accumulates (a reference comparison over
a time window) samples the step; at the end of the run every enabled
check is evaluated and `output/results.toml` is written. A failed
required check ends the run with `Checks.ChecksFailed`, and `aeolus run`
exits with 3. A run that stops with an error records the failed sanity
check before the error propagates. See [Checks](../api/checks.md).

## Where the code still differs from the target

* **ONE-e (#454), first part: the physics out of the urban glue.** The
  equations the design names in the urban couplings, and the other
  self-contained physics found there, now live in their physics modules,
  and the couplings call them, bit for bit:

  | from `src/urban/couplings/` | now |
  | :-- | :-- |
  | `moist_buoyancy.jl` | `Operators.moist_buoyancy_perturbation!`, `add_moist_buoyancy!` |
  | `traffic_heat.jl` | `BuildingEnergy.traffic_heat_flux`, `add_traffic_heat!` |
  | `buildings_citymean.jl` | `BuildingEnergy.CityBuildings`, `city_buildings`, `step_city_buildings!`, `apply_building_heat!` |
  | `veg_drag.jl` | `PlantCanopy.vegetation_canopy_drag` (applied with `apply_canopy_drag!`) |
  | `sw_modifiers.jl` | `PlantCanopy.apply_tree_shade!` |
  | `aq_chem.jl` | `Chemistry.traffic_profile`, `SurfaceExchange.aerodynamic_resistance` |
  | `glass.jl` | `UrbanSurface.glazing_areal_heat_capacity`, `glazing_solar_transmittance` |
  | `fog_dew.jl` | `RadiationFog.dew_exchange_columns!` |
  | `cloud_shadow.jl` | `Radiosity.direct_absorbed_share!`, `SolarGeometry.shade_direct_beam!` |
  | `micro_kessler.jl` | `Microphysics.clamp_booked!` |

  Two couplings measurably did nothing and were deleted rather than
  moved: `snow_legacy.jl` (a snow flag that nothing read, and that could
  never be set, as nothing passed it precipitation) and `aq_aerosol.jl`
  (a toy aerosol that started at zero, had no source and was neither
  advected nor written, so it stayed zero), with their `:snow` and
  `:aerosol` slots. `test/glue_physics.jl` tests each moved function at
  module level. Still in glue code: the surface exchange's delivery of
  the surface fluxes to the first air cell (`surface_exchange.jl`), the
  column radiation's stitching of the LES column onto a standard
  atmosphere (`column_radiation.jl`) and the case's hydrostatic reference
  state (`core/reference_state.jl`). The couplings themselves,
  `UrbanCase`, `CaseConfig`, the registry and the urban stages, ledgers
  and output are still there: their port to components is the rest of
  ONE-e.
* The low-level cores keep their keyword interface — `f_coriolis`,
  `C_s`, the sponge and the θ fluxes in the constructors, and
  `body_force!`, `post_predictor!`, `sgs`, `momentum_advection` on the
  steps — because the urban and SBL run types, the nesting drivers and
  many module tests still assemble the core by hand. A `Model` passes
  these keywords; the run types that still use them directly move to
  components in the later steps of the epic (urban couplings ONE-e #454,
  nesting ONE-f #455, SBL ONE-g #456), after which the keywords go.
* The urban run type and the SBL run type still select their SGS closures
  themselves; their `[sgs]`/`:sgs` paths become the `:sgs` component when
  they are ported.
* The urban couplings' `smoke_check`s, the urban run type's own
  `results.toml` closure verdict (`run_results`) and the SBL reference
  comparisons (the plotting scripts of `examples/sbl/`) are not checks
  yet: they move to `checks(c)` and `[checks.reference]` when ONE-e (#454)
  and ONE-g (#456) port those run types to components.
* The urban couplings and the SBL run type report into the one
  `Budgets` and write through the one writer, but they are still run
  types of their own. Their budgets keep their own step hooks (the urban
  `budgets_begin_step!`/`budgets_end_step!`, the SBL substage evaluation
  of the momentum terms) until ONE-e (#454) and ONE-g (#456) port them to
  components. The particle tracks of the model's couplings
  (`particles.csv`) are Lagrangian rows, not a gridded catalogue variable,
  and stay a file of their own.
* The urban and SBL run types still refuse the anelastic core's
  checkpoints and adaptive step in their own config checks; they become
  components in ONE-e (#454) and ONE-g (#456), which deliver every
  component they port on both cores.
