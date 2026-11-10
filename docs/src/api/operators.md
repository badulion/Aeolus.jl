# Operators

The `Aeolus.Operators` module is a thin dispatch façade. Each function
here forwards to a location-and-mesh-specialized kernel in the
`FiniteVolume` submodule (see [Multiple dispatch](../design/dispatch.md)).

## First derivatives

```@docs
Aeolus.Operators.∂x
Aeolus.Operators.∂y
Aeolus.Operators.∂z
```

## Second-order operators

```@docs
Aeolus.Operators.laplacian
Aeolus.Operators.divergence
```

## Advection

```@docs
Aeolus.Operators.advect_centered!
Aeolus.Operators.advect_upwind!
Aeolus.Operators.momentum_advect!
```

## Interpolation between staggered locations

```@docs
Aeolus.Operators.interpolate_to_face!
Aeolus.Operators.interpolate_to_center!
```

## Physics forcings

```@docs
Aeolus.Operators.add_buoyancy!
Aeolus.Operators.add_coriolis!
```

## Subgrid-scale closure

```@docs
Aeolus.Operators.compute_smagorinsky_νt!
Aeolus.Operators.compute_dynamic_smagorinsky_νt!
Aeolus.Operators.DynamicSmagorinskyWorkspace
Aeolus.Operators.add_eddy_viscosity_u!
Aeolus.Operators.add_eddy_viscosity_v!
Aeolus.Operators.add_eddy_viscosity_w!
Aeolus.Operators.add_eddy_diffusion_scalar!
```

## Stratification (P58-a)

Pointwise fields of the [`TurbulenceStability`](turbulence/turbulence_stability.md)
contract.

```@docs
Aeolus.Operators.buoyancy_frequency_squared!
Aeolus.Operators.strain_rate_squared!
Aeolus.Operators.gradient_richardson!
```

## Advection schemes

```@docs
Aeolus.Operators.advect_weno5!
```

## Forcing terms

```@docs
Aeolus.Operators.add_surface_flux!
Aeolus.Operators.add_sponge!
```

## Damping profiles

The one implementation of relaxation rates, shared by the Rayleigh sponge
(every mesh type, and the `:cos2` nest sponge), the fringe region of
`Transition` (P59-d, #185) and `LargeScaleForcing.ReferenceDamping`
(N-parent-b, #254). The arithmetic is one kernel,
`FiniteVolume._add_damping!`, on a `RegularGrid` and a `StretchedGrid`;
its target is a number, a level vector, a `Field` or a 3-D array.

```@docs
Aeolus.Operators.AbstractRamp
Aeolus.Operators.SinSquaredRamp
Aeolus.Operators.NordstromRamp
Aeolus.Operators.smooth_step
Aeolus.Operators.DampingProfile
Aeolus.Operators.sponge_profile
Aeolus.Operators.damping_rate
Aeolus.Operators.damping_coordinates
Aeolus.Operators.FiniteVolume.in_damping
Aeolus.Operators.FiniteVolume._add_damping!
```

## The fused sweep: tendencies and predictor in one pass (SC-e, #533)

A substage of the Boussinesq core computes its tendencies and its
predictor in one sweep per k-slab, row by row in cache, when every term
between the base tendencies and the predictor is a **sweep term** (see
[the KA port](../design/ka_port.md), "SC-e", and the roadmap section of
#533). The tendency of a row is a compile-time tuple of terms: the base
tendencies (advection and molecular diffusion) initialise it, each
further term adds its point kernel. A term declares its stencil radius
(`footprint`); the sweep's ring holds `1 + max ry` rows and its slabs keep
`max rz` edge planes. Anything global (a horizontal mean, a precomputed
eddy viscosity) is computed before the sweep, in `prepare!`.

A component or built-in term without a sweep term keeps the substage on
separate loops, one per term (the reference implementation); the run logs which term
caused it, once, and writes it to `results.toml` (`[loop_fusion.<domain>]`:
`setting`, `substages_fused` and `not_fusable`). The input key
`[timestep] loop_fusion` (`Simulations.set_loop_fusion!`) chooses
`"if_possible"` (the default: fused where every term has a fused form),
`"always"` (fused, or the run stops at its first step naming the terms
without one; PALM's `loop_optimization = 'cache'` stops the same way) or
`"never"` (one loop per term). The
documented reasons are `Simulations.SWEEP_FALLBACKS`, the allowlist of
the completeness guard `test/fused_sweep.jl`.

Every step core has the fused form (#533 plan step 8): `step!` and
`step_rk3!` of the Boussinesq core; the sound-proof cores
(`step_anelastic!`, `step_pseudo_incompressible!` and their `step_rk3!`),
whose base term is the `W`-weighted advection
(`Anelastic.WeightedBaseTendencies`, with `Anelastic.AnelasticBuoyancy`);
and the explicit part of `step_imex!`, as one fused tendency launch
without the predictor (`term_tendencies!`; the stage combinations read the
tendencies). On a CPU backend the sweep runs with its ring buffer; on any
other backend the same term tuple runs in the per-point form
(`pointwise_predictor!`, one KernelAbstractions work item per point,
`Backends.launch_points!`), tested on `KernelAbstractions.CPU()` as the
stand-in for a GPU.

**Post-predictor operators** (#533 step 6) are `PostTerm`s: an implicit
drag, a penalty or a prescribed flow that corrects the predicted
velocity before the projection (the core's `post_predictor!` stage). A
post term with footprint 0 runs on a row right after its predictor; one
with y-radius `r` computes its factors `r` rows and applies them `2r`
rows behind its input, so every factor reads values its own operator
has not changed yet (Jacobi, as the whole-field operators). Points whose
stencil reaches a halo, a slab's edge planes or a point an earlier
operator deferred run after the sweep, in the same two phases, with the
halos filled (`post_plan`). An operator that needs the whole predicted
field is an `AfterSweep`: its whole-field method runs after the sweep,
and so does every operator after it. Post terms do not read θ or qv,
which the sweep has already advanced (the whole-field order runs the
post-predictor stage before the scalars' update). A domain puts a
component's post term on the rows only when it is pointwise (the
prescribed flow, the IBM correction) and runs one with a stencil (canopy
drag, facade and ground wall stress, the tiles' drag) as an `AfterSweep`
of its own `run_stage!`: measured, their row forms cost 1.4–2.2× their
whole-field passes (`benchmarks/bench_post_sweep.jl`; the roadmap's #533
step 6).

With the budgets on, the sweep books the in-core components' contributions
itself (plan step 7): a `BookedTerm` sums the booked fields of each row
before and after its term into `LevelSums`, and the term's
`finish_sweep!` (the domain's budget term) books them after the sweep.

How the steps combine (the integration of steps 3, 6, 7 and 8): the
per-point form runs the post terms after its predictor on every point
(the schedule `post_plan` gives a point outside the sweep's widths);
with the budgets on, a post-predictor component the budgets attribute
runs in its budget context as an `AfterSweep`, and a term tuple that
`books` stays in the ring-buffer sweep (the per-point form has no
per-level sums: `builtin.budgets_per_point`); the sound-proof cores and
the fused IMEX launch call `finish_sweep!` as the Boussinesq core does.

The slab and ring machinery is one driver, `blocked_update!`, generic
over the update it runs (a `BlockedUpdate`, #554): the substage predictor
(`blocked_predictor!`) and the update of a `TKEClosure`'s SGS energy
(`advance_sgs!`, `PrognosticSGS`), and a scalar's explicit eddy-diffusion
step (`blocked_eddy_diffusion!`, #559: the SGS diffusion of the urban
tracers, `Urban.TracerSGSDiffusion`).

```@docs
Aeolus.Operators.FiniteVolume.SweepTerm
Aeolus.Operators.FiniteVolume.sweep_term
Aeolus.Operators.FiniteVolume.footprint
Aeolus.Operators.FiniteVolume.sweep_inputs
Aeolus.Operators.FiniteVolume.prepare!
Aeolus.Operators.FiniteVolume.finish_sweep!
Aeolus.Operators.FiniteVolume.add_tendency!
Aeolus.Operators.FiniteVolume.add_row!
Aeolus.Operators.FiniteVolume.SweepRow
Aeolus.Operators.FiniteVolume.SWEEP_U
Aeolus.Operators.FiniteVolume.BaseTerm
Aeolus.Operators.FiniteVolume.init_point!
Aeolus.Operators.FiniteVolume.BaseTendencies
Aeolus.Operators.FiniteVolume.TendencyFields
Aeolus.Operators.FiniteVolume.term_tendencies!
Aeolus.Operators.FiniteVolume.pointwise_predictor!
Aeolus.Simulations.fused_predictor!
Aeolus.Operators.FiniteVolume.sweep_lags
Aeolus.Operators.FiniteVolume.PostTerm
Aeolus.Operators.FiniteVolume.post_sweep_term
Aeolus.Operators.FiniteVolume.post_row!
Aeolus.Operators.FiniteVolume.finish!
Aeolus.Operators.FiniteVolume.AfterSweep
Aeolus.Operators.FiniteVolume.post_plan
Aeolus.Operators.FiniteVolume.post_delay
Aeolus.Operators.FiniteVolume.row_index
Aeolus.Operators.FiniteVolume.row_entries
Aeolus.Operators.FiniteVolume.rows_sorted
Aeolus.Operators.FiniteVolume.blocked_predictor!
Aeolus.Operators.FiniteVolume.BlockedUpdate
Aeolus.Operators.FiniteVolume.blocked_update!
Aeolus.Operators.FiniteVolume.update_lags
Aeolus.Operators.FiniteVolume.begin_slab!
Aeolus.Operators.FiniteVolume.compute_rows!
Aeolus.Operators.FiniteVolume.write_row!
Aeolus.Operators.FiniteVolume.SweepBuffers
Aeolus.Operators.FiniteVolume.sweep_buffers
Aeolus.Operators.FiniteVolume.sweep_slabs
Aeolus.Operators.FiniteVolume.LevelSums
Aeolus.Operators.FiniteVolume.BookedTerm
Aeolus.Operators.FiniteVolume.SWEEP_BOOKED
Aeolus.Operators.FiniteVolume.books
Aeolus.Operators.blocked_eddy_diffusion!
Aeolus.Operators.EddyDiffusionScratch
Aeolus.Simulations.SWEEP_FALLBACKS
Aeolus.Simulations.SweepReport
Aeolus.Simulations.sweep_report
Aeolus.Simulations.sweep_reason
Aeolus.Simulations.sweep_key
Aeolus.Simulations.hook_terms
Aeolus.Simulations.hook_blockers
Aeolus.Simulations.post_hook_blockers
Aeolus.Simulations.post_hook_terms
Aeolus.Components.run_after_sweep
Aeolus.Components.sweep_type
Aeolus.Components.sweep_report
```

### Pointwise terms of the sweep (#533 plan step 2)

The core's buoyancy, Coriolis, surface flux and sponge, and the
components `Coriolis`, `SurfaceThetaFlux`, `Sponge`, `Fringe`, the urban
site's geostrophic forcing (pressure-gradient mode) and moist buoyancy and
the slab's floor flux, as sweep terms. Each evaluates the expression of its
whole-field kernel, which stays the reference. The built-in ones are
`OptionalTerm`s on the core's switches, so a Boussinesq configuration has
one compiled sweep whatever is switched on.

```@docs
Aeolus.Operators.FiniteVolume.BuoyancyTerm
Aeolus.Operators.FiniteVolume.CoriolisTerm
Aeolus.Operators.FiniteVolume.SurfaceFluxTerm
Aeolus.Operators.FiniteVolume.DampingTerm
Aeolus.Operators.FiniteVolume.DampingTargets
Aeolus.Operators.FiniteVolume.sponge_term
Aeolus.Operators.FiniteVolume.damping_levels
Aeolus.Operators.FiniteVolume.OptionalTerm
Aeolus.Simulations.SpongeTargets
```

### The SGS closure inside the sweep (#533 step 4)

The closure's coefficients are computed before the sweep
(`compute_sgs!`, in the term's `prepare!`), its momentum, θ and qv
tendencies inside it, by the point bodies of the separate loops'
operators. Which body each tendency uses comes from the closure contract:
`sgs_momentum_sweep` (momentum), `sgs_scalar_face_diffusivity` or
`sgs_scalar_tensor` (θ, and qv unless `sgs_tracer_own_diffusivity`). A
closure that adds its own `add_sgs_momentum!`, `add_sgs_scalar!` or
`sgs_tracer_diffusivity!` adds the matching generic (guard:
`test/fused_sweep_sgs.jl`); one without a sweep form, or whose own
tendency is more specific than its sweep form (`sgs_sweep_form_matches`,
checked at run time), runs one loop per term and is reported by its
type.

```@docs
Aeolus.Operators.SGSTerm
Aeolus.Operators.sgs_sweep_term
Aeolus.Operators.sgs_momentum_sweep
Aeolus.Operators.sgs_tracer_own_diffusivity
Aeolus.Operators.sgs_sweep_form_matches
Aeolus.Operators.FiniteVolume.EddyMomentum
Aeolus.Operators.FiniteVolume.StressMomentum
Aeolus.Operators.FiniteVolume.EddyScalar
Aeolus.Operators.FiniteVolume.TensorScalar
Aeolus.Simulations.SGSChoice
```

## LES analysis toolbox (Phase 54)

Explicit filters, the isotropic energy spectrum and the isotropic
random-field initialiser (see [LES filtering & closure](../theory/les_filtering.md#Explicit-filters-and-spectra)).
The box filter is the test filter of the dynamic Smagorinsky model; the
Fourier tools treat the interior as triply periodic.

```@docs
Aeolus.Operators.box_filter!
Aeolus.Operators.staggered_solid_mask
Aeolus.Operators.gaussian_filter!
Aeolus.Operators.spectral_cutoff_filter!
Aeolus.Operators.isotropic_energy_spectrum
Aeolus.Operators.isotropic_random_field!
Aeolus.Operators.rescale_to_spectrum!
```

## A-priori SGS-stress analysis (Phase 56)

The exact subfilter stress of a resolved field under an explicit filter,
model stresses for comparison, the SGS dissipation and the a-priori
metrics (see [LES filtering & closure](../theory/les_filtering.md#A-priori-tests:-the-exact-subfilter-stress-(Phase-56))).

```@docs
Aeolus.Operators.SymmetricTensorField
Aeolus.Operators.components
Aeolus.Operators.subfilter_stress!
Aeolus.Operators.bardina_stress!
Aeolus.Operators.filtered_strain
Aeolus.Operators.filtered_strain!
Aeolus.Operators.eddy_viscosity_stress!
Aeolus.Operators.smagorinsky_stress!
Aeolus.Operators.sgs_dissipation
Aeolus.Operators.stress_eigvals
Aeolus.Operators.lumley_invariants
Aeolus.Operators.apriori_metrics
Aeolus.Operators.dissipation_matched_cs
```

## FFT planning policy

The one place the FFTW planner flag and thread count of every FFT plan
(the pressure solvers, the diagnostics) are chosen (#245). Default
`:measure`, `:patient` opt-in; the test suite uses `:estimate`. Why it matters for restarts:
[Checkpoint / restart and FFT planning](../design/checkpoint_restart.md).

```@docs
Aeolus.Operators.set_fftw_policy!
Aeolus.Operators.fftw_policy
Aeolus.Operators.fftw_planner_flags
Aeolus.Operators.save_fftw_wisdom
Aeolus.Operators.load_fftw_wisdom!
```
