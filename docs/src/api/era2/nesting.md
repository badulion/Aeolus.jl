# Parent ↔ child grid nesting (Phase 30)

`Aeolus.Nesting` is the single home of every parent-child operator:
the [`NestedGrid`](@ref) geometry (any integer refinement per direction),
the conservative staggered prolongation, conservative restriction,
Berger–Colella refluxing, the [`NestSponge`](@ref), the field-set
[`NestInterpState`](@ref) (velocities, θ, `qv`, extra scalars), and the
time coupling: [`NestedRun`](@ref), the per-domain
[`domain_advance`](@ref) closures, [`boundary_time`](@ref) and
[`step_nested!`](@ref) — or, for a caller that advances the child itself
(the Berlin case, #265), [`begin_parent_step!`](@ref),
[`subcycle_dt`](@ref) and [`end_parent_step!`](@ref) — and the child's boundary conditions:
[`nest_bcs`](@ref) (`OpenBoundary` nest faces) fed by
[`NestBoundaryData`](@ref) (the nest parent as a two-snapshot
[`WindowedBoundaryData`](@ref Aeolus.BoundaryConditions.WindowedBoundaryData)),
refreshed by [`fill_nest_halos!`](@ref), and the two-way feedback:
[`TwoWayFeedback`](@ref) (anterpolation with buffer zone, canopy
restriction, inflow exclusion and relaxation, scalar refluxing from
[`NestFluxRegisters`](@ref), re-projection), [`anterpolate!`](@ref),
[`feedback_mask`](@ref) and [`two_way_feedback!`](@ref); a tracer moved
by a separate transport books its interface flux with
[`book_nest_flux!`](@ref), and a user core declares its stage weights with
[`nest_stage_weights`](@ref). The parent may be z-stretched
(`StretchedGrid`), the operators may run in the reference-mass measure of
a sound-proof pair ([`NestMeasure`](@ref), [`nest_measure`](@ref)), and
restriction may be solid-aware ([`nest_fluid_mask`](@ref)). The block-structured AMR of
[`Aeolus.AdaptiveMesh`](adaptive_mesh.md) builds on these operators.
The theory is on the [nesting theory page](../../theory/nesting.md), the
[nested time-coupling page](../../theory/nesting_time.md), the
[nested boundary-conditions page](../../theory/nesting_boundaries.md) and the
[two-way nesting page](../../theory/nesting_twoway.md).

```@autodocs
Modules = [Aeolus.Nesting]
```
