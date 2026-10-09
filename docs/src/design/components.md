# Components and the stage pipeline

Status: **built in ONE-a (#450)**, the first step of the single-model
epic, with the configuration schema of ONE-b (#451). The target is
[One model](single_model.md); this page says what the code does today,
and where it still differs from the target.

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
3. the core steps — `step!` (Euler) or `step_rk3!` (SSP-RK3) of the
   Boussinesq core, `step!` of the anelastic core — and calls the
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

## Where the code still differs from the target

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
* `budget!`, `outputs` and `checks` are part of the interface with empty
  defaults: the shared budgets and the catalogue-selected output are
  ONE-c (#452), the `[checks]`/`results.toml` machinery ONE-d (#453).
* The anelastic core has no checkpoints yet (`IO.checkpoint!` takes a
  Boussinesq core): the components support both cores, and their
  checkpoint state is core-independent, but an anelastic restart is
  ONE-j (#460).
