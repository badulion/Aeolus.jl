# Era VI — High-fidelity turbulence closure

Era VI replaces the single Smagorinsky closure by a family of
subgrid-scale (SGS) models: Lagrangian-dynamic and scale-dependent
coefficients, anisotropic and minimum-dissipation closures, prognostic
TKE and Reynolds-stress transport, stability-aware mixing. They all plug
into the dynamical cores through one interface, the **SGS closure
contract** (P54-a, #167), defined once in `Aeolus.Operators`
(`src/operators/sgs_closures.jl`). The design and the numbers it was
tested at are in
[Smagorinsky-Lilly SGS → Closure contract](../theory/sgs.md#Closure-contract).

- Shared contract — SGS closures (P54-a, #167; below) — `Aeolus.Operators`
- Stress-form momentum divergence (P55-b, #175; below) — `Aeolus.Operators`
- [Stability contract — TurbulenceStability](era6/turbulence_stability.md) — `Aeolus.TurbulenceStability` (Phase 58, #170; [theory](../theory/turbulence_stability.md))
- [Wall models — WallModels](era6/wall_models.md) — `Aeolus.WallModels` (Phase 57, #169: equilibrium wall laws and the implicit wall-stress operator; [theory](../theory/wall_models.md))
- [Large-scale forcing — LargeScaleForcing](era6/large_scale_forcing.md) — `Aeolus.LargeScaleForcing` (Phase 58, #171: geostrophic pressure gradient, nudging, advective tendencies, surface-temperature series; [theory](../theory/large_scale_forcing.md))
- Walls — `NoSlipWall` and the resolved wall stress (P59-a, #172; below; [theory](../theory/transition.md))

## Shared contract — SGS closures (P54-a)

Every core (`step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`, and the
anelastic / pseudo-incompressible `step!`) takes the keyword
`sgs = closure`. A closure fills the eddy viscosity and diffusivity
(`sim.νt`, `sim.κt`) with the pure `sgs_coefficients!`, adds its
tendencies with `add_sgs_momentum!` / `add_sgs_scalar!`, and commits its
memory with `advance_sgs!` exactly once per accepted step;
`step_adaptive!` rolls the memory back on a rejected step through
`sgs_state` / `restore_sgs_state!`.

```julia
using Aeolus
sim = BoussinesqSimulation(RegularGrid(; size = (32, 32, 32), extent = (1.0, 1.0, 1.0)))
step!(sim, 1e-3; sgs = StaticSmagorinsky(0.17))           # == BoussinesqSimulation(…; C_s = 0.17)
step!(sim, 1e-3; sgs = DynamicSmagorinsky(sim.mesh))       # Germano–Lilly
step_rk3!(sim, 1e-3; sgs = (νt, u, v, w) -> compute_smagorinsky_νt!(νt, u, v, w, 0.1))  # a plain function
```

```@docs
Aeolus.Operators.AbstractSGSClosure
Aeolus.Operators.SGSFields
Aeolus.Operators.sgs_fields
Aeolus.Operators.sgs_coefficients!
Aeolus.Operators.advance_sgs!
Aeolus.Operators.sgs_state
Aeolus.Operators.restore_sgs_state!
Aeolus.Operators.add_sgs_momentum!
Aeolus.Operators.add_sgs_scalar!
Aeolus.Operators.compute_sgs!
Aeolus.Operators.sgs_closure
Aeolus.Operators.AbstractEddyViscosityClosure
Aeolus.Operators.sgs_viscosity!
Aeolus.Operators.turbulent_prandtl
Aeolus.Operators.StaticSmagorinsky
Aeolus.Operators.DynamicSmagorinsky
Aeolus.Operators.FunctionSGS
Aeolus.Operators._velocity_gradient_at
Aeolus.Operators._strain_at
```

## Stress-form momentum divergence (P55-b)

The SGS momentum tendency in the stress form `∂ⱼ(2νᵢⱼSᵢⱼ)`, with
horizontal and vertical eddy viscosities, and the generic operator for
any symmetric stress τ on the C-grid. Every core selects the form with
`sgs_stress = :laplacian` (default) or `:symmetric`; under `step_imex!`
the diagonal part is implicit and the cross terms explicit. With
`sgs_stress = :symmetric`, `cfl_diagnostics`, `adaptive_dt`,
`step_adaptive!` and `imex_stability_limit` take the stress form's own
explicit diffusive limit,
[`stress_form_cfl_viscosity`](@ref Aeolus.TimeIntegration.stress_form_cfl_viscosity)
(`2νt` for `νh = νv`). Theory:
[Smagorinsky-Lilly SGS → Stress form vs Laplacian form](../theory/sgs.md#Stress-form-vs-Laplacian-form).

```julia
step!(sim, 0.5; sgs = StaticSmagorinsky(0.17), sgs_stress = :symmetric)
τ = sgs_stress_tensor!(StressTensor(sim.mesh), sim.u, sim.v, sim.w, νh, νv)
add_stress_divergence!(sim.tu, sim.tv, sim.tw, τ)   # == add_sgs_stress_divergence!(…, νh, νv)
```

```@docs
Aeolus.Operators.StressTensor
Aeolus.Operators.add_stress_divergence!
Aeolus.Operators.sgs_stress_tensor!
Aeolus.Operators.add_sgs_stress_divergence!
Aeolus.Operators.sgs_stress_dissipation
```

## Walls (P59-a, #172)

The theory is developed in
[Transition and wall-bounded flow](../theory/transition.md).
The no-slip / moving-wall condition
[`NoSlipWall`](@ref Aeolus.BoundaryConditions.NoSlipWall) lives in
`BoundaryConditions` ([API](boundary_conditions.md), together with
`is_wall` and `homogeneous_bcs`). Its plane-mean wall stress is a method
of the cut-cell generic `wall_shear_stress`:

```@docs
Aeolus.CutCell.wall_shear_stress(::Aeolus.Simulations.BoussinesqSimulation)
```
