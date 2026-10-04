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
- Wall-bounded flow diagnostics (P54-d, #174; below) — `Aeolus.Operators`
- Lagrangian dynamic Smagorinsky (P54-b, #173; below) — `Aeolus.Operators`
- Stress-form momentum divergence (P55-b, #175; below) — `Aeolus.Operators`
- [Stability contract — TurbulenceStability](era6/turbulence_stability.md) — `Aeolus.TurbulenceStability` (Phase 58, #170; [theory](../theory/turbulence_stability.md))
- [Transition diagnostics — Transition](era6/transition.md) — `Aeolus.Transition` (Phase 59, #180; [theory](../theory/transition.md#Laminar–turbulent-transition))
- [Wall models — WallModels](era6/wall_models.md) — `Aeolus.WallModels` (Phase 57, #169: equilibrium wall laws and the implicit wall-stress operator; #179: wall model on immersed and cut-cell walls; [theory](../theory/wall_models.md))
- [Large-scale forcing — LargeScaleForcing](era6/large_scale_forcing.md) — `Aeolus.LargeScaleForcing` (Phase 58, #171: geostrophic pressure gradient, nudging, advective tendencies, surface-temperature series; [theory](../theory/large_scale_forcing.md))
- Walls — `NoSlipWall` and the resolved wall stress (P59-a, #172; below; [theory](../theory/transition.md))
- AMD and Vreman closures (P55-d, #176; below) — `Aeolus.Operators`

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
Aeolus.Operators.sgs_tracer_diffusivity!
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
Aeolus.Operators._vorticity_at
```

## Lagrangian dynamic Smagorinsky (P54-b)

The Germano identity averaged along pathlines (Meneveau, Lund & Cabot
1996), a stateful closure on the contract above. It takes an optional
solid mask, and the plane / local [`DynamicSmagorinsky`](@ref
Aeolus.Operators.DynamicSmagorinsky) take the same one. Theory, tests and
numbers:
[Smagorinsky-Lilly SGS → Lagrangian dynamic averaging](../theory/sgs.md#Lagrangian-dynamic-averaging-(Meneveau,-Lund-and-Cabot-1996)).

```julia
using Aeolus
g = RegularGrid(; size = (32, 32, 16), extent = (32.0, 32.0, 16.0), halo = (3, 3, 3))
sm = SolidMask(g); mark_solid!(sm, (x, y, z) -> 12 < x < 20 && 12 < y < 20 && z < 6)
sim = BoussinesqSimulation(g; bcs = FieldBoundaryConditions(:periodic, :periodic, :wall))
lag = LagrangianDynamicSmagorinsky(g; mask = sm.mask)
step!(sim, 0.05; sgs = lag)          # memory committed once per accepted step
lag.workspace.cs2                    # the coefficient C_s² of the last call
```

```@docs
Aeolus.Operators.LagrangianDynamicSmagorinsky
Aeolus.Operators.advance_sgs!(::Aeolus.Operators.LagrangianDynamicSmagorinsky, ::Any, ::Any)
Aeolus.Operators._germano_identity!
Aeolus.Operators._lagrangian_relax!
Aeolus.Operators._upstream
Aeolus.Operators._SGSMask
Aeolus.Operators._masked_velocities!
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

## Wall-bounded flow diagnostics (P54-d)

Plane-averaged diagnostics of a horizontally periodic, wall-bounded run,
shared by every Era VI benchmark. The cases that produce the flows —
`half_channel_case` (the neutral ABL half channel of Porté-Agel et al.
2000) and `channel_case` (the two-wall channel) — are in
[WallModels](era6/wall_models.md); theory and measured numbers in
[Wall models → Wall-bounded benchmark cases](../theory/wall_models.md#Wall-bounded-benchmark-cases).

```julia
using Aeolus
hc = half_channel_case(; N = 24, closure = StaticSmagorinsky(0.1))
run_channel!(hc, 8.0)                                   # spin-up, H/u* = 1
st = run_channel!(hc, 20.0; average_from = 8.0)         # st.τ_wall, st.phi_m, st.stress
E = streamwise_spectrum(hc.sim.u, [3, 6, 12])           # E11(k1) at three levels
```

```@docs
Aeolus.Operators.phi_m_profile
Aeolus.Operators.total_stress_profile
Aeolus.Operators.streamwise_spectrum
```

## Anisotropic minimum-dissipation and Vreman closures (P55-d)

Two closures on the contract that weight each direction with its own grid
spacing and vanish in laminar flow (`src/operators/sgs_amd_vreman.jl`,
#176). The theory, the constants and the measured numbers are in
[Smagorinsky-Lilly SGS → AMD and Vreman](../theory/sgs.md#Anisotropic-minimum-dissipation-and-Vreman-closures).

```julia
step!(sim, dt; sgs = AMDClosure())            # C = 0.3, own κ_t, buoyancy term with g/θ₀
step!(sim, dt; sgs = VremanClosure(0.17))     # c = 2.5 C_s², κ_t = ν_t/Pr_t
```

```@docs
Aeolus.Operators.AMDClosure
Aeolus.Operators.VremanClosure
Aeolus.Operators.compute_amd_coefficients!
Aeolus.Operators.compute_amd_scalar_diffusivity!
Aeolus.Operators.compute_vreman_νt!
Aeolus.Operators.sgs_buoyancy_parameter
```
