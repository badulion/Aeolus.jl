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
- [Wall models — WallModels](era6/wall_models.md) — `Aeolus.WallModels` (Phase 57, #169: equilibrium wall laws and the implicit wall-stress operator; #183: matching height and input filtering against log-layer mismatch; [theory](../theory/wall_models.md))

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
