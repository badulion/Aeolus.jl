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

The block and ring machinery is one driver, `blocked_update!`, generic
over the update it runs (a `BlockedUpdate`, #554); it cuts the interior
into slabs of planes times ranges of j-rows, one block per thread
(`sweep_blocks`, #558). It runs the substage predictor
(`blocked_predictor!`) and the update of a `TKEClosure`'s SGS energy
(`advance_sgs!`, `PrognosticSGS`).

```@docs
Aeolus.Operators.FiniteVolume.SweepTerm
Aeolus.Operators.FiniteVolume.sweep_term
Aeolus.Operators.FiniteVolume.footprint
Aeolus.Operators.FiniteVolume.sweep_inputs
Aeolus.Operators.FiniteVolume.prepare!
Aeolus.Operators.FiniteVolume.add_tendency!
Aeolus.Operators.FiniteVolume.add_row!
Aeolus.Operators.FiniteVolume.SweepRow
Aeolus.Operators.FiniteVolume.SWEEP_U
Aeolus.Operators.FiniteVolume.BaseTendencies
Aeolus.Operators.FiniteVolume.sweep_lags
Aeolus.Operators.FiniteVolume.blocked_predictor!
Aeolus.Operators.FiniteVolume.BlockedUpdate
Aeolus.Operators.FiniteVolume.blocked_update!
Aeolus.Operators.FiniteVolume.SweepBlock
Aeolus.Operators.FiniteVolume.update_lags
Aeolus.Operators.FiniteVolume.edge_rows
Aeolus.Operators.FiniteVolume.begin_block!
Aeolus.Operators.FiniteVolume.compute_rows!
Aeolus.Operators.FiniteVolume.block_slot
Aeolus.Operators.FiniteVolume.write_row!
Aeolus.Operators.FiniteVolume.SweepBuffers
Aeolus.Operators.FiniteVolume.sweep_buffers
Aeolus.Operators.FiniteVolume.sweep_blocks
Aeolus.Simulations.SWEEP_FALLBACKS
Aeolus.Simulations.SweepReport
Aeolus.Simulations.sweep_report
Aeolus.Simulations.sweep_reason
Aeolus.Simulations.sweep_key
Aeolus.Simulations.hook_terms
Aeolus.Simulations.hook_blockers
Aeolus.Simulations.post_hook_blockers
Aeolus.Components.sweep_type
Aeolus.Components.sweep_report
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
