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
- Lagrangian dynamic Smagorinsky (P54-b, #173) and its scale-dependent form (LASD, P54-e, #186; below) — `Aeolus.Operators`
- Stress-form momentum divergence (P55-b, #175; below) — `Aeolus.Operators`
- Tensorial scalar eddy diffusivity (P55-c, #181; below) — `Aeolus.Operators`
- Stability-corrected anisotropic eddy viscosity (P55-e, #187; below) — `Aeolus.Operators`
- [Stability contract — TurbulenceStability](era6/turbulence_stability.md) — `Aeolus.TurbulenceStability` (Phase 58, #170; [theory](../theory/turbulence_stability.md))
- [Transition diagnostics — Transition](era6/transition.md) — `Aeolus.Transition` (Phase 59, #180; flat-plate boundary layer, P59-d, #185; [theory](../theory/transition.md#Laminar–turbulent-transition), [flat plate](../theory/transition.md#Flat-plate-transition))
- [Wall models — WallModels](era6/wall_models.md) — `Aeolus.WallModels` (Phase 57, #169: equilibrium wall laws and the implicit wall-stress operator; #179: wall model on immersed and cut-cell walls; #183: matching height and input filtering against log-layer mismatch; #189: the non-equilibrium (TBLE) wall model with dynamic κ; #190: dynamic roughness of partially resolved relief; [theory](../theory/wall_models.md))
- [Large-scale forcing — LargeScaleForcing](era6/large_scale_forcing.md) — `Aeolus.LargeScaleForcing` (Phase 58, #171: geostrophic pressure gradient, nudging, advective tendencies, surface-temperature series; [theory](../theory/large_scale_forcing.md))
- Walls — `NoSlipWall` and the resolved wall stress (P59-a, #172; below; [theory](../theory/transition.md))
- AMD and Vreman closures (P55-d, #176; below) — `Aeolus.Operators`
- [Prognostic SGS TKE — PrognosticSGS](era6/prognostic_sgs.md) — `Aeolus.PrognosticSGS`, the Deardorff/Moeng `TKEClosure` (Phase 56, P56-b, #182) and the differential `ReynoldsStressClosure` (P56-d, #188; [theory](../theory/sgs_tke.md))
- σ-model SGS closure (P59-b, #184; below) — `Aeolus.Operators`

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
Aeolus.Operators.sgs_explicit_stress
Aeolus.Operators.sgs_tracer_diffusivity!
Aeolus.Operators.sgs_scalar_face_diffusivity
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

The scale-dependent model (Bou-Zeid, Meneveau & Parlange 2005) is the
same type with `scale_dependent = true`: a second test filter (5-cell
box) and the power law between the two test-filter scales; `beta` holds
β, the coefficient ratio per doubling of the filter width. Theory and
numbers:
[Smagorinsky-Lilly SGS → Scale-dependent Lagrangian dynamic model](../theory/sgs.md#Scale-dependent-Lagrangian-dynamic-model-(Bou-Zeid,-Meneveau-and-Parlange-2005)).

```julia
lasd = LagrangianDynamicSmagorinsky(g; mask = sm.mask, scale_dependent = true, cadence = 1)
step!(sim, 0.05; sgs = lasd)
lasd.beta                            # β = 2^φ ∈ [1/8, 8] of the last call
```

```@docs
Aeolus.Operators.LagrangianDynamicSmagorinsky
Aeolus.Operators.advance_sgs!(::Aeolus.Operators.LagrangianDynamicSmagorinsky, ::Any, ::Any)
Aeolus.Operators._germano_identity!
Aeolus.Operators._lagrangian_relax!
Aeolus.Operators._upstream
Aeolus.Operators._scale_dependent_cs2!
Aeolus.Operators._second_test_filter!
Aeolus.Operators._SecondTestFilter
Aeolus.Operators._SGSMask
Aeolus.Operators._masked_velocities!
```

`laminar_consistent(closure)` (P59-e, #195) is the contract's trait for
closures whose ``ν_t`` vanishes identically in laminar flow: `true` for
the σ-model and for `StabilityCorrected` / `IsentropicDiffusion` around
it, `false` for Smagorinsky. The Berlin `turbulence_diag` asks it to
decide which closures must give ``ν_t = 0`` in a laminar flow on its
mesh; the relaminarisation study is in
[Transition → Relaminarisation](../theory/transition.md#Relaminarisation-in-strongly-stable-stratification).

```@docs
Aeolus.Operators.laminar_consistent
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

## Tensorial scalar eddy diffusivity (P55-c)

The scalar SGS flux `∂ᵢ(Kᵢⱼ ∂ⱼϕ)`. The diagonal `(κh, κv)` is the
contract's default. A general symmetric `K` at cell centres uses the triad
discretisation of Griffies et al. (1998), which is conservative, symmetric
and dissipates variance for positive semi-definite K. The rotated isentropic
(Redi) tensor has a slope taper and an isotropic fallback where N² ≤ 0.
Under `step_imex!` the diagonal is implicit and the cross terms explicit.
Theory: [Smagorinsky-Lilly SGS → Tensorial scalar diffusivity](../theory/sgs.md#Tensorial-scalar-diffusivity).

```julia
add_eddy_diffusion_scalar!(tθ, θ, κh, κv)                     # diag(κh, κh, κv); docs in Operators
K = isentropic_tensor!(DiffusivityTensor(mesh), θ, κh, 0.0; bcs = bcs)
add_tensor_diffusion_scalar!(tϕ, ϕ, K; bounded = bounded_directions(bcs))
step!(sim, 0.5; sgs = IsentropicDiffusion(StaticSmagorinsky(0.17), sim.mesh; diapycnal = 0.1))
```

```@docs
Aeolus.Operators.DiffusivityTensor
Aeolus.Operators.diagonal_tensor!
Aeolus.Operators.bounded_directions
Aeolus.Operators.add_tensor_diffusion_scalar!
Aeolus.Operators.isentropic_tensor!
Aeolus.Operators.sgs_scalar_tensor
Aeolus.Operators.IsentropicDiffusion
```

## Stability-corrected anisotropic eddy viscosity (P55-e)

`StabilityCorrected` wraps an eddy-viscosity closure: `νh = ν ℓh`,
`νv = ν ℓv f_m(Ri)`, `κh = ν ℓh/Prₜ`, `κv = ν ℓv f_h(Ri)/Prₜ`, with Ri_g
from `gradient_richardson!`, `f_m, f_h` from the TurbulenceStability
contract (Brown et al. 1994 by default) and the anisotropic length scales
`Δh = √(dx dy)`, `Δv = dz`. Neutral flow (`N² = 0`, `ℓ = 1`) is the base
closure bit for bit. Theory:
[Smagorinsky-Lilly SGS → Stability correction](../theory/sgs.md#Stability-correction).

```julia
c = StabilityCorrected(StaticSmagorinsky(0.17), sim.mesh)          # Brown 1994, :anisotropic
step!(sim, 0.5; sgs = c, sgs_stress = :symmetric)                   # νv reaches the stress form
c.Ri, c.νv, c.κv                                                    # the last Ri_g, νv, κv
StabilityCorrected(DynamicSmagorinsky(sim.mesh), sim.mesh; scheme = LillyCutoff())
IsentropicDiffusion(StabilityCorrected(StaticSmagorinsky(0.17), sim.mesh), sim.mesh)
```

```@docs
Aeolus.Operators.StabilityCorrected
Aeolus.Operators.dynamic_sgs_coefficient
Aeolus.Operators.anisotropic_length_factors
Aeolus.Operators.sgs_stratification!
```

## σ-model SGS closure (P59-b, #184)

`SigmaModel(C_σ = 1.35)` is an `AbstractEddyViscosityClosure` of the
contract above, with
``ν_t = (C_σΔ)^2 σ_3(σ_1 − σ_2)(σ_2 − σ_3)/σ_1^2`` built from the singular
values of `_velocity_gradient_at` (Nicoud et al. 2011). It is zero in
pure shear, solid-body rotation, any 2-D flow, isotropic expansion and
axisymmetric strain, and it scales as ``z^3`` at a no-slip wall. Theory and
measured numbers:
[Smagorinsky-Lilly SGS → σ-model](../theory/sgs.md#σ-model). The Berlin
case selects it with `physics[:sgs] = :sigma`.

```julia
step!(sim, dt; sgs = SigmaModel())             # C_σ = 1.35, κ_t = ν_t/Pr_t
```

```@docs
Aeolus.Operators.SigmaModel
Aeolus.Operators.compute_sigma_νt!
Aeolus.Operators.sigma_operator
Aeolus.Operators.sigma_singular_values
Aeolus.Operators._sigma_operator_raw
Aeolus.Operators._sym3_eigvals
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
Aeolus.Operators.resolved_tke_profile
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
