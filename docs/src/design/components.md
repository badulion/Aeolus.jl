# Components and the stage pipeline

Status: **built in ONE-a (#450)**, the first step of the single-model
epic, with the configuration schema of ONE-b (#451), on both dynamical
cores since ONE-j (#460). The target is [One model](single_model.md); this
page says what the code does today, and where it still differs from the
target.

## What a step does

`step!(model, dt)` steps every domain of the [`Model`](@ref
Aeolus.Components.Model): one alone, or several as a nest (ONE-f, #455;
see "Nesting by domains" below). For one
[`Domain`](@ref Aeolus.Components.Domain):

1. the domain's drivers advance to the step (the open faces' boundary
   window, `advance_feed!`);
2. the components of the stages before the core run, in
   [`PIPELINE`](@ref Aeolus.Components.PIPELINE) order: `:clock`,
   `:nest_sync`, `:radiation`, `:sw_modifiers`, `:surface_flux`,
   `:sources`, `:chemistry`, `:transport`, `:microphysics`, `:tracers`;
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
60)`): it runs in the step that crosses a multiple of `every`
(`floor(t₁/every) > floor(t₀/every)`), and with `first = true` also in
the step that starts at model time 0. Within a stage, components run by
slot first (`:advection`, `:sgs`, `:microphysics`, `:radiation`,
`:surface`), then by `component_name`, except where a component declares
its place in a stage with `Stage(…; order)`: a lower `order` runs first.
The orders are the components' own (never configuration); they exist for
work that depends on another component's in the same stage — the column
radiation's fluxes before the radiosity that distributes them, before
the buildings that absorb them (ONE-e, #454). The domain stores its
components in the one order that meets every stage's declared orders and
is otherwise by slot and name (`order_components`, a topological sort;
orders that contradict each other across stages are an error), as a
concretely typed tuple, and the pipeline walks the tuple recursively, so
every `run_stage!` call is statically dispatched.

`:transport` (stage 5, between `:chemistry` and `:microphysics`) is the
advection of the tracers components carry with the velocity at the step's
start; the urban case's scalar transport runs there, as it ran in the
urban stage loop. (The design's scalar advance after the projection,
stage 10, would change the urban results; moving it there is a physics
change left to the port of the tracers.)

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
| [`CentredAdvection`](@ref Aeolus.Components.CentredAdvection) | `:advection` | `[advection]` `scheme = "centred"` (the one provider, no options) | called by the core in stage 6 | `Operators.momentum_advect!` (Boussinesq, fused with diffusion), `weighted_momentum_advect!` (anelastic) |
| [`SGS`](@ref Aeolus.Components.SGS) | `:sgs` | `[sgs]` `model` (default `smagorinsky`; `tke`, ONE-l #475); per closure its options (ONE-m #476): `c_s` (smagorinsky, vreman, sigma, amd; each its default), `pr_t`, `stability` (not amd), `stress`; `tke`: `e_min`, `surface_factor`, `wall_cap`, `wall_cap_top`, `surface_value` (NEST-p1 #481), `stress` | called by the core in stage 6; `tke` also `:surface_flux` (after the others: the wall fluxes of [`wall_theta_flux`](@ref Aeolus.Components.wall_theta_flux), and with `surface_value` the u* of the `:surface` component, [`surface_friction_velocity`](@ref Aeolus.Components.surface_friction_velocity)) | the closure of `Components.build_closure`; `PrognosticSGS.TKEClosure`, `initialize_tke!` |
| [`Coriolis`](@ref Aeolus.Components.Coriolis) | — | `[coriolis]` `f` or `latitude` | `:tendencies` | `Operators.add_coriolis!` |
| [`SurfaceThetaFlux`](@ref Aeolus.Components.SurfaceThetaFlux) | — | `[surface_theta_flux]` `bottom`, `top` | `:tendencies` | `Operators.add_surface_flux!` |
| [`Sponge`](@ref Aeolus.Components.Sponge) | — | `[sponge]` `alpha_max`, `z_start`, `target` (`rest`, `geostrophic`, `mean`) | `:body_force` | `Simulations.add_velocity_sponge!` → `Operators.add_sponge!` |

The boundary-layer components of the SBL cases (ONE-g, #456), on both cores:

| component | slot | section | stage | delegates to |
| :-- | :-- | :-- | :-- | :-- |
| [`LargeScale`](@ref Aeolus.Components.LargeScale) | — | `[large_scale_forcing]` `u_g`, `v_g` (numbers or `"driver"`), `advective_tendencies`, `subsidence`, `nudging` (the dynamic driver's `ls_*` tables); built in code also with `clock` and a [`BoundaryLayerTracker`](@ref Aeolus.Components.BoundaryLayerTracker) (NEST-p1 #481) | `:clock` (the step start), `:body_force`, with a tracker `:column_tendency` | `LargeScaleForcing.apply_forcing!` (`GeostrophicPressureGradient` with the `Coriolis` component's ``f``, else the core's) |
| [`SurfaceLayer`](@ref Aeolus.Components.SurfaceLayer) | `:surface` | `[surface]` `model = "most"` (ONE-m #476; the slot's other providers are the couplings `none`, `slab`, `flux_map`), `z0m`, `z0h`, `temperature`, `stability`, `rho_cp` | `:surface_flux` (the solve), `:tendencies` (heat flux), `:post_predictor` (wall stress) | `SurfaceExchange.solve_surface_layer`, `WallModels.MOSTWallLaw`, `apply_wall_stress!` |
| [`Towers`](@ref Aeolus.Components.Towers) | — | `[towers]` `below`, `window`, `hours` | `:output` | `Transition.TowerSampler`, `hockey_stick_threshold` |

The urban nest parent's physics (NEST-p1, #481), components of the
parent's domain ([`parent_domain`](@ref Aeolus.Urban.parent_domain)), on
both cores, checkpointed:

| component | slot | section | stage | delegates to |
| :-- | :-- | :-- | :-- | :-- |
| [`TiledUrbanSurface`](@ref Aeolus.Urban.TiledUrbanSurface) | `:surface` | `[surface]` `model = "tiles"`: `period`, `u_min`, `sw_down`, `lw_down`; the land use is the tile form of the domain's static driver (`drivers.static_tiles`), the tiles those of an urban site's class-to-surface factory | `:surface_flux` (the held canopy sources, a tile step every `period`), `:post_predictor` (the implicit drag) | `SurfaceTiles` (`step_parent_column!`), [`build_parent_surface`](@ref Aeolus.Urban.build_parent_surface) |
| [`PrescribedSurfaceFlux`](@ref Aeolus.Urban.PrescribedSurfaceFlux) | `:surface` | `[surface]` `model = "prescribed"` (the functions of time `q_surface`, `e_surface` are build options) | `:surface_flux` | `LargeScaleForcing.forcing_mean` |
| [`MeanColumnRadiation`](@ref Aeolus.Urban.MeanColumnRadiation) | `:radiation` | `[radiation]` `scheme = "mean_column"`: `period`, `humid`; the sun of the site's clock ([`SiteSun`](@ref Aeolus.Urban.SiteSun)) | `:surface_flux`, order 1 (after the surface: the call every `period`, the heating every step) | `ColumnRadiation.column_radiation!`, `stitched_column` |

The parent's large-scale forcing is the `LargeScale` component on the
core's substage clock (`clock = :core`: the windows the parent's
forcing has used since #259; the step mean of the configuration is a
physics change for a later issue) with a `BoundaryLayerTracker` for its
nudging weight, and its closure `[sgs] model = "tke"` with the exact wall
distance of the stretched grid (`wall_cap_top = false`) and the tiles'
surface value. These three sections are not yet in the model's
configuration schema: the parent is built from `[nesting.parent]`
(`ParentConfig`) until it is a `[[domain]]` of its own (NEST-p2/p3), and
the tiles need an urban site's class tables.

The components of the validation cases (ONE-l, #475; the former run
types of `Aeolus.RunDrivers`), on both cores, checkpointed:

| component | slot | section | stage | delegates to |
| :-- | :-- | :-- | :-- | :-- |
| [`KinematicFlow`](@ref Aeolus.Components.KinematicFlow) | — | `[kinematic_flow]` `kind` (`solid_body_rotation`, `uniform`), `omega`, `center`, `velocity` | `:clock`, `:post_predictor`, `:surface_update` (the velocity prescribed; stateless) | — (the core's scalar advection carries θ) |
| [`ParentStatistics`](@ref Aeolus.Components.ParentStatistics) | — | `[inflow_statistics]` `enabled`, `start`, `tau_avg`, `refine`, `depth` | `:output` | `Transition.ParentResolvedStatistics`, `update_inflow_statistics!`, `inflow_stress_parts` |
| [`Fringe`](@ref Aeolus.Components.Fringe) | — | `[fringe]` `profile` (`blasius`), `inflow_reynolds`, `velocity`, `length`, `rise`, `fall`, `lambda_max`, `top`, `tu`, … | `:body_force`; the free-stream top before the core is built ([`core_boundaries`](@ref Aeolus.Components.core_boundaries)) | `Transition.FringeRegion`, `blasius_target`, `blasius_freestream`, `FreeStreamDisturbance`, `disturbance_fringe_update`, `set_blasius!` |

The validation run types they replace: `advection` is
`examples/scalar_advection_2d/` (θ carried by `[kinematic_flow]`: the
core's centred flux form on the staggered grid instead of the run type's
upwind kernel on cell-centred velocities, a stated change); `cbl` is
`examples/cbl_tke/` and `examples/cbl_inflow_statistics/`, bit for bit;
`shear_layer` is `examples/kelvin_helmholtz/` (the `[initial]` profiles
`tanh` and `seed_mode`, the outputs `w_rms`, `turbulent_fraction_profile`,
`shear_production_profile`, `buoyancy_production_profile`,
`vorticity_y`), bit for bit; `flat_plate` is `examples/flat_plate_t3a/`,
bit for bit. The numbers the run types computed at the end of a run are
now computed from the output by the cases' scripts. `inflow_fetch` and
`obstacle_channel` ran on pressure-driven channels and `sgs_decay` is the
filtered-DNS study; all three are dropped (the design drops the channel
cases and the a-priori study), their harnesses kept as test fixtures
(`test/fixtures/inflow_fetch.jl`, `sgs_validation.jl`), and so is the
channel harness of `WallModels` (`test/fixtures/channel_cases.jl`).

Every domain has an `:advection` provider; `[sgs] model = "none"`
enables no `:sgs` component. Each component has its own per-domain
section of the model's configuration, generated from its
[`schema`](@ref Aeolus.Components.schema)
([`COMPONENTS`](@ref Aeolus.SimulationDriver.COMPONENTS), ONE-b #451), and
is switched on by it. The component of an exclusive slot is a provider of
the slot's section instead (ONE-m, #476): its `schema` is a
[`slot_section`](@ref Aeolus.Config.slot_section) with its provider(s),
the model's schema merges the parts of one slot
([`merge_sections`](@ref Aeolus.Config.merge_sections)), and `enabled`
compares the selector with its provider name (`opts.model === :most`);
the section holds the chosen provider's options only ([`enabled`](@ref Aeolus.Components.enabled)`(C,
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
  An entry that books what it does itself inside the dynamics' context
  (`add_heat!`) opts out with `Components.attributed(c, Val(stage)) =
  false`, so it is not counted twice: the urban site's open faces, whose
  θ relaxation is `(:dynamics, :relaxation)` (ONE-e, #454).
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
- the horizontal-mean profiles, and (ONE-l #475) the total heat-flux,
  w-variance and resolved-TKE profiles and the shear-production,
  buoyancy-flux and turbulent-fraction profiles of the transition
  diagnostics;
- the domain means and `w_rms` as time series; `u_squared` and
  `vorticity_y` as 3-D fields;
- the `budget_*` groups.

`[output] variables` selects from the catalogues of the core and the
enabled components. The configuration is checked against the declared
names, so selecting a variable whose component is off is an error. The
default is `u`, `v`, `w`, `theta` and the time series of the domain
means.

One writer, `Aeolus.Output.OutputWriter`, writes the selection into one
NetCDF-4 file. Records (3-D, maps, profiles) go out every `cadence`, and
the time series every `series_cadence`. Each value is either the value at
the record or its mean over the record's interval (`averaging`). A
spectrum (dimension `k`, the shell wavenumbers) is a record too: the
core's `energy_spectrum` (ONE-g #456, with `kinetic_energy`, `dissipation`
and `dissipation_sgs`). The urban case writes through the same writer:
the urban couplings' `output_fields` are catalogues of `OutputVariable`s.

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

## Nesting by domains (ONE-f, #455)

A model of several domains is built by
[`nested_model`](@ref Aeolus.Components.nested_model) from one
[`NestLink`](@ref Aeolus.Components.NestLink) per child; the
configuration builds it from its `[[domain]]` blocks
([Run configuration](run_configuration.md)). The coupling is
`Nesting.NestHierarchy` (#328), unchanged: its operators, boundary data,
subcycling, flux registers and two-way feedback. What is new is that each
domain's advance is its own pipeline:
[`PipelineCore`](@ref Aeolus.Components.PipelineCore) runs a domain's
drivers, its stages before the core, the core step with stages 6–8 (the
nest's register hook chained before the domain's body-force hook, a nest
sponge after its post-predictor hook) and its stages after the core. A
child therefore runs every stage once per substep, and its in-core stages
once per stage of its scheme. The design's `:nest_sync` (1) and
`:nest_feedback` (12) are the two halves of the coupling the model runs
around the domains' pipelines: the boundary data prolongated at both ends
of the parent's step, then the feedback after the children reach its end.

Measured (`test/model_nesting.jl`; 16 × 16 × 8 parent at Δ = 100 m, its
16 × 16 × 8 child at Δ = 50 m, RK3, dt = 4 s):

- one-way, the parent's state after 6 steps is the parent run alone, bit
  for bit;
- two-way, the composite θ of the nest is conserved to round-off: a
  relative change of 2.0e-16 over 6 steps (the test bound is 1e-13), and
  4.0e-16 over 3 steps of a three-level two-way nest; one-way, where
  nothing is refluxed, it changes by 8.4e-7. The parent differs from the
  parent alone under the child;
- a child's pipeline stages run twice per parent step (2 substeps), its
  in-core stage 3 × 2 times with RK3;
- an interrupted nested run resumes from its checkpoint bit for bit, both
  domains;
- a one-way child that goes non-finite stops the run as a blow-up (the
  guard inspects every domain, not only the root it steps), and the crash
  checkpoint holds the whole nest at its last good state.

A root step of this nest costs 1.27 ms one-way and 1.71 ms two-way,
against 0.19 ms for the parent alone (the child has as many cells as the
parent and takes two substeps). The single-domain step is unchanged: a
64³ RK3 step of the bare config takes 33.6 ms on the branch and 33.4 ms
on its base (minimum of 15, two alternating runs each).

### The one nest engine (NEST-p2, #482)

The nest is stepped by the model's own engine
([`nest_window!`](@ref Aeolus.Components.nest_window!), on the links of
the hierarchy): a domain's pipeline is split around the nest — stages
1–11, then its children's windows and the two-way feedback (stage 12,
`:nest_feedback`), then stage 13 — so a parent's budgets close after its
children's feedback, which they book as its nest exchange. With fixed
steps the prognostic state is that of `Nesting.step_nested!`, bit for
bit; on a 64 × 64 × 32 root with a 64 × 64 × 32 child at `r = 2` (RK3,
one thread, minimum of 4 × 5 windows) a window takes 0.0501 s two-way
and 0.0393 s one-way, against 0.0511 s and 0.0397 s on the hierarchy's
own path (4 threads: 0.0366 / 0.0299 s against 0.0385 / 0.0298 s). The
engine adds adaptive steps with the smallest stable whole number of
child substeps and window retries, per-domain budgets with a nest-exchange
term, siblings in parallel and a one-way parent below the root stepping
ahead of its children, synthetic inflow turbulence with the parent's
statistics, mixed Boussinesq/anelastic cores, the PALM rule for the
exchanged fields and the subgrid handover; the measured numbers are on
[Nested time coupling](../theory/nesting_time.md) and
[Two-way nesting](../theory/nesting_twoway.md), the configuration on
[Run configuration](run_configuration.md).

Measured on the nest of this section (`test/model_nesting.jl`): every
domain's energy and momentum budgets close to `1e-12` of their content
(one-/two-way, Euler/RK3, fixed and adaptive), and the parent's exchange
and the child's add up to the heat the covered parent cells gained, the
composite conservation; an adaptive two-way nest with two retried windows
changes its composite θ by 1.8e-15 over 6 windows; with inflow turbulence
the added mass flux is at most 1e-12 of the mean-flow flux and the
two-way composite θ changes by 2.2e-15.

## Where the code still differs from the target

* **ONE-e (#454), first part: the physics out of the urban glue.** The
  equations the design names in the urban couplings, and the other
  self-contained physics found there, now live in their physics modules,
  and the couplings call them, bit for bit:

  | from `src/components/urban/` | now |
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
  | `column_radiation.jl` | the module `ColumnRadiation`: `ColumnRadiationModel`, `column_radiation_solver`, `stitched_column`, `column_radiation!`, `column_heating_kday` (the nest parent uses it too) |
  | `surface_exchange.jl` | `SurfaceExchange.deliver_surface_fluxes!` (the held H and E into the first air cell) |
  | `radiosity_sun.jl` | `Radiosity.surface_irradiance!` (plan-area fluxes to per-surface irradiance) |
  | `site/reference_state.jl` | `Anelastic.hydrostatic_pressure` (the continuous hydrostatic integration), `MoistThermodynamics.θ_per_heat` (the heat-to-θ conversion ``1/(ρ c_p Π)``) |

  Two couplings measurably did nothing and were deleted rather than
  moved: `snow_legacy.jl` (a snow flag that nothing read, and that could
  never be set, as nothing passed it precipitation) and `aq_aerosol.jl`
  (a toy aerosol that started at zero, had no source and was neither
  advected nor written, so it stayed zero), with their `:snow` and
  `:aerosol` slots. `test/glue_physics.jl` tests each moved function at
  module level. The couplings now gather the case's fields and hand the
  results on (the case's reference state itself is
  `Anelastic.reference_from_profile` of its initial profile); small
  formulas that remain are the radiative temperature of the surface
  ``(Σε T_s^4/Σε)^{1/4}`` (the column radiation and, summed differently,
  the nest parent), the friction velocity ``\sqrt{τ/ρ}`` of a surface step,
  and the independent recomputations inside the `smoke_check`s.
* **ONE-e (#454), second part: the urban case on the pipeline and on the
  model's configuration.**
  - **The stage loop is gone.** A built case is a `Domain` of the
    pipeline (`case.domain`; `Urban.build_domain!`): the case's own
    component `UrbanCore` (the local hour and the budgets' snapshot first
    in `:clock`; the heat captures of the fixed-gradient top and the open
    faces, with the open faces' θ relaxation, in `:tendencies`) and one
    component `UrbanEntry` per enabled coupling, carrying the pipeline
    stages, cadences and `order`s the coupling declares. A step is the
    pipeline's stages around the `:timestepper` provider's core step,
    whose hooks are the pipeline's (`core_hooks`). The urban stages became
    the pipeline's: `:nest_parent` is `:nest_sync`, `:anthropogenic` is
    `:sources`, `:fog_dew` the second entry of `:microphysics`,
    `:transport` a new stage of the pipeline, and the integer orders
    within a stage are `Stage(…; order)`, which `order_components` meets.
    The case's ledger is the domain's budgets (`DomainBudgets` around
    `case.budgets`, binned by the local hour). Measured: the 54 runs of
    the smoke matrix (12 steps, `seed_fog!`, 1245 state digests of the
    child, every tracer, Ts and the nest parent) are bit for bit those of
    master `37053292`.
  - **Berlin on the model.** `examples/urban/berlin_*.toml` are model
    configs: `[urban] enabled` makes the domain an urban site, and every
    urban component has a section of the model's schema, generated from
    its `default_options` (`enabled`, default `false`, and its options).
    `aeolus run` builds and runs the site (then `UrbanCase.run_urban!`); the
    `urban` run type, its schema and `[physics]` are gone. The configs of
    the four scenarios resolve, field for field and bitwise, to the
    `CaseConfig`s of the fixture (`test/config_berlin.jl`).
  - **Its checks.** The site publishes checks of the one checks system
    (ONE-d): `urban_budgets` (its budgets' closure against
    `BUDGET_TOLERANCES`) and `<component>_checks` per enabled component
    (its `smoke_check`s that fail), besides the core's `finite` and
    `energy_closure` of its domain; `[checks]` enables them, the run
    evaluates them into `results.toml`, and a failed required one ends it
    with exit code 3.

  - **The coupling registry is gone.** The couplings are urban
    components, `UrbanComponent <: AbstractComponent`, with the
    component interface's `component_name` (their section), `slot`
    (`nothing`, or an urban slot of `URBAN_SLOTS`) and `incompatible`
    (component types, not `slot => provider` pairs). The site's
    components are the fixed list `URBAN_COMPONENTS`; a config enables
    one by its section (`enabled`), and `CaseConfig.components` holds
    the enabled ones with all their options (the registry's `REGISTRY`,
    `register!`, `selected_couplings`, `is_default`, `default_provider`
    and `CaseConfig.physics`/`options` are gone; `URBAN_DEFAULTS` is the
    reference selection of `CaseConfig()` and the smoke runs). The
    smoke matrix's 54 runs hash bit for bit as before (3809 hashes of
    the thread-invariance worker's `smoke` section).

* **ONE-k (#474): the urban site on the model.**
  - **A component takes the core step.** A component may take its
    domain's core step itself: `takes_core_step(c)`, `advance_core!(c, d,
    dt)` (with the pipeline's hooks), `next_dt(c, d)` (its step) and
    `check_dt(c, d, dt)` (an explicit step it refuses). `step!(d, dt)`
    runs it in place of the core's `step!`/`step_rk3!`; `pick_dt(model)`
    and `step!(model)` take the step it picks. The stages after the core
    see `t0 + dt` (an adaptive step's retried pieces sum to it only to
    round-off). The urban site is that component: the case object became
    `UrbanSite <: AbstractComponent`, the first component of its domain,
    with the site's shared state (its options, geometry, surface bus,
    tracers, budgets and reference state); its `:timestepper` component
    takes the step (fixed, adaptive with retried pieces, IMEX). Its
    sanity check is its `:budgets` entry (a blowup throws, and the run
    monitor diagnoses it), its frame log its first `:output` entry.
    `step_case!` and the case's own step are gone.
  - **The run loop is the model's.** The run loop of `aeolus run`
    (`SimulationDriver`) asks the component that takes the core step for
    the run's horizon and steps (`run_begin!`, `run_continues`,
    `run_end!`; the urban nest prefetches its parent's next window within
    it, SC-c), the diagnostics row (`step_report`, `run_diagnostics`),
    the numbers it adds to `results.toml` (`run_results`) and its state
    arrays for the bitwise comparisons (`state_arrays`).
    `Components.run_to!(model, t_end)` is the same loop without output,
    monitor and checkpoints. Gone: `run_case!`, `run_urban!`,
    `UrbanSiteRun`, the site's NetCDF writer and its restart: the run's
    writer writes the variables `[output] variables` selects from the
    domain's catalogue (the site publishes its frame series, tracers,
    surface maps and the components' fields, and `static_outputs` its
    geometry), and the site is checkpoint component `urban` of the run's
    `resume!`.
  - **The configuration is the model's.** `CaseConfig` (a mutable struct
    with Julia-side defaults, a keyword and a path constructor) is gone:
    the site's options are `site_options(cfg)`, a `NamedTuple` read from
    the resolved model config, changed in TOML or with `--set`.
  - **The module and its files.** `Aeolus.UrbanCase` is `Aeolus.Urban`,
    its files under `src/components/urban/` (the components),
    `src/components/urban/site/` (the site) and
    `src/components/urban/parent/` (the nest parent).

  Measured: the 54 runs of the smoke matrix (`seed_fog!`, 24 s each)
  are bit for bit those of the branch base in every state digest (the
  child, every tracer, Ts, the nest parent), step count, heat closure,
  water residual and air-heat term, after every commit of ONE-k.

  What is still the site's own: the urban components run through the
  `UrbanEntry` adapter (one concrete type, so a selection does not
  compile the domain anew; their methods take the site, `run_stage!(c,
  case, Val(S), dt)`, since they read its shared state); the site builds
  its own core (the Coriolis force, the Smagorinsky coefficient and the
  sponge in the core's constructor, not the model's `Coriolis`, `SGS` and
  `Sponge` components); and its nest parent (`ParentCase`,
  `ParentConfig`, run by the `:nesting` component, and alone by the
  `urban_parent` run type) is not yet a `[[domain]]`: that needs the
  Berlin nest (one-way with the prefetched parent window, two-way, the
  synthetic inflow) as the model's `ModelNest`. Its physics are
  components of a core domain since NEST-p1 (#481, below).
* **NEST-p1 (#481): the parent's physics as components.** The tiled
  surface, the prescribed surface flux, the column radiation, the
  large-scale forcing with its boundary-layer-dependent nudging weight and
  the TKE closure on the stretched grid are components of the parent's
  `Domain` ([`parent_domain`](@ref Aeolus.Urban.parent_domain)); the
  hand-wired code of `ParentCase` that they replace is gone. `ParentCase`
  is the shell the nest and the `urban_parent` run type step: its ledgers
  (air heat, water) around the components' stage entries, in the
  pipeline's order. The domain alone through the pipeline
  (`step_adaptive!(domain)`) is that run bit for bit. Measured (FFTW
  `:estimate`): five parents of the branch base (the 24 × 24 × 38 tiled
  smoke parent with and without subsidence, 12 steps; the encroachment,
  Wangara-like and Smagorinsky/Euler idealised parents, 20 steps) give
  the same bits in every field, through `ParentCase` and through the
  pipeline; the step of the 96 × 96 × 38 tiled parent took 0.0742 s
  before, 0.0757 s through `ParentCase` and 0.0711 s through the pipeline
  (median of 12). Two findings: the model's `[sgs] model = "tke"` could
  not be built on a `StretchedGrid` (fixed: the exact wall distance), and
  the parent's forcing on the core's RK3 clock leads the step mean by
  3Δt/2 (kept, `clock = :core`; tested).
* **ONE-m (#476): one key per exclusive slot.** After #472/#473 a
  non-default provider took an enable/disable pair of sections
  (`[pressure_fft] enabled = false` with `[pressure_multigrid] enabled =
  true`, `[aq_chem]` with `[chem_off]`; `[surface_layer] enabled` beside
  `[surface] model`). Each exclusive slot is now one section with a
  selector key, holding the chosen provider's options only
  (`Config.slot_section`, `provider!`; validated and resolved by the
  loader, `with` and the option reference): the model's `[core] model`,
  `[timestep] scheme` (run level), `[sgs] model`, `[surface] model`
  (`none`, `slab`, `flux_map`, `most`: the `SurfaceLayer` component),
  `[microphysics] scheme`, `[advection] scheme`, and the urban site's
  fifteen slots (`Urban.URBAN_SLOT_SECTIONS`; `[chemistry] scheme =
  "aq_chem" | "none"`). A slot the model and the site share pools both
  sets of providers, each offered on its domain kind
  (`SimulationDriver.domain_kind`). Every shipped TOML is converted.
  Measured: the state digests of every shipped config at its `[test]`
  size, of the simulation-digest runs and of the smoke matrix are those
  of the branch base; the four Berlin scenarios resolve to the fixture
  field for field.
* The low-level cores keep their keyword interface — `f_coriolis`,
  `C_s`, the sponge and the θ fluxes in the constructors, and
  `body_force!`, `post_predictor!`, `sgs`, `momentum_advection` on the
  steps — because the urban site, the nesting drivers and many module
  tests still assemble the core by hand. A `Model` passes these keywords;
  the urban site moves to components in the rest of ONE-e (#454), after
  which the keywords go. (The SBL run type is gone: its cases are
  configurations of the model, ONE-g #456.)
* The urban site still selects its SGS closures itself; its `:sgs` path
  becomes the `:sgs` component when it is ported.
* The urban site's own `results.toml` closure verdict (`run_results`) is
  written beside the checks it publishes. (The SBL reference comparisons
  are `[checks.reference]` of the SBL configurations since ONE-g, #456.)
* The urban couplings report into the one `Budgets` and write through the
  one writer; the urban case's budgets are its domain's (ONE-e part 2).
  The particle tracks of the model's couplings (`particles.csv`) are
  Lagrangian rows, not a gridded catalogue variable, and stay a file of
  their own; so are the CASES-99 tower bins, of which the model writes the
  fitted threshold profile only (`tower_threshold`).
* The urban site still refuses the anelastic core's checkpoints and
  adaptive step in its own config checks.
