# Components and the stage pipeline

Status: **built in ONE-a (#450)**, the first step of the single-model
epic; on both dynamical cores since ONE-j (#460). The target is [One model](single_model.md); this page says what the
code does today, and where it still differs from the target.

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

| component | slot | section (today) | stage | delegates to |
| :-- | :-- | :-- | :-- | :-- |
| [`CentredAdvection`](@ref Aeolus.Components.CentredAdvection) | `:advection` | — | called by the core in stage 6 | `Operators.momentum_advect!` (Boussinesq, fused with diffusion), `weighted_momentum_advect!` (anelastic) |
| [`SGS`](@ref Aeolus.Components.SGS) | `:sgs` | `[sgs]` | called by the core in stage 6 | the closure of `Components.build_closure` |
| [`Coriolis`](@ref Aeolus.Components.Coriolis) | — | `[dynamics] f_coriolis` | `:tendencies` | `Operators.add_coriolis!` |
| [`SurfaceThetaFlux`](@ref Aeolus.Components.SurfaceThetaFlux) | — | `[boundaries] theta_flux_*` | `:tendencies` | `Operators.add_surface_flux!` |
| [`Sponge`](@ref Aeolus.Components.Sponge) | — | `[dynamics] sponge_*` | `:body_force` | `Simulations.add_velocity_sponge!` → `Operators.add_sponge!` |

Every domain has an `:advection` provider; `[sgs] closure = "none"`
enables no `:sgs` component. The sections are those of the `simulation`
run type: each component's [`schema`](@ref Aeolus.Components.schema) is
the option list the run type's section is assembled from
([`add_options!`](@ref Aeolus.Components.add_options!)), so the option
reference pages are unchanged. The schema step of the epic (ONE-b, #451) gives
each component its own section.

The `simulation` run type builds its core with the dynamics only
(viscosity, buoyancy, moisture, boundary conditions) and hands SGS,
advection, Coriolis, the sponge and the θ fluxes to these components. The
plain Smagorinsky closure is therefore the `:sgs` component's
`StaticSmagorinsky`, no longer the core's built-in `C_s`; its runs now
report the `sgs.nu_h_max`, `sgs.nu_v_max` and `sgs.kappa_h_max`
diagnostics every other closure already reported.

The couplings EX-d (#441) put around the `simulation` run type's step —
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
and the prescribed surface fluxes (`SurfaceThetaFlux`) are in the
`ρ̄`-weighted flux form of its advection (`Operators.flux_weights`; see
[Sound-proof cores](../theory/anelastic.md)): `Σ ρ̄ θ ΔV` changes only by
the surface fluxes, to round-off, and the SGS terms conserve `Σ ρ̄ uᵢ ΔV`.
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
weighted fluxes changed them, which `test/components_pipeline.jl` asserts.

## Bit identity

Every `simulation`-config variant of `test/support/simulation_digests.jl`
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
Euler and RK3.

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
* The urban and SBL run types still refuse the anelastic core's
  checkpoints and adaptive step in their own config checks; they become
  components in ONE-e (#454) and ONE-g (#456), which deliver every
  component they port on both cores.
