# Parent ↔ child grid nesting (Phase 30)

`Aeolus.Nesting` is the single home of every parent-child operator:
the [`NestedGrid`](@ref) geometry (any integer refinement per direction),
the conservative staggered prolongation, conservative restriction,
Berger–Colella refluxing, the [`NestSponge`](@ref), the field-set
[`NestInterpState`](@ref) (velocities, θ, `qv`, extra scalars), and the
time coupling: [`NestedRun`](@ref), the per-domain
[`domain_advance`](@ref) closures, [`boundary_time`](@ref) and
[`step_nested!`](@ref). The block-structured AMR of
[`Aeolus.AdaptiveMesh`](adaptive_mesh.md) builds on these operators.
The theory is on the [nesting theory page](../../theory/nesting.md) and the
[nested time-coupling page](../../theory/nesting_time.md).

```@autodocs
Modules = [Aeolus.Nesting]
```
