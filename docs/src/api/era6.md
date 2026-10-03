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
