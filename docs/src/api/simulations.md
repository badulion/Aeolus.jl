# Simulations

The high-level model driver.
[`BoussinesqSimulation`](@ref Aeolus.Simulations.BoussinesqSimulation)
owns the mesh, boundary conditions, pressure solver, all prognostic
fields, and all scratch buffers. One `step!(sim, dt)` executes the
Euler-projection fractional step described in
[Fractional-step projection](../theory/fractional_step.md).

```@docs
Aeolus.Simulations.BoussinesqSimulation
Aeolus.Simulations.run!
Aeolus.Simulations.kinetic_energy
Aeolus.Simulations.max_divergence
Aeolus.Simulations.max_νt
Aeolus.Simulations.mean_θ_profile
```

## Swapping operators in `step!`

Two keywords of `step!(sim, dt; …)` replace a numerical operator without
changing the algorithm (both default to the built-in one):

* `momentum_advection = f(tu, tv, tw, u, v, w)` — the momentum-advection
  operator, e.g. `(tu, tv, tw, u, v, w) -> momentum_advect_weno_z!(tu, tv, tw, u, v, w)`
  from `AdvectionSchemes`;
* `sgs = closure` — the subgrid-scale closure, any
  [`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) of the
  Era VI contract ([`StaticSmagorinsky`](@ref Aeolus.Operators.StaticSmagorinsky),
  [`DynamicSmagorinsky`](@ref Aeolus.Operators.DynamicSmagorinsky), an
  `ML.MLSGS`, …) or a plain function `f(νt, u, v, w)` that fills `νt`
  (wrapped in a [`FunctionSGS`](@ref Aeolus.Operators.FunctionSGS),
  `κt = νt/Prₜ`). It switches the SGS terms on even with `C_s = 0`, and
  they are added where the built-in closure adds them, so
  `StaticSmagorinsky(C_s)` — or an `MLSGS` wrapping the Smagorinsky
  closure — reproduces the built-in run bit for bit. The pre-contract
  keyword name `eddy_viscosity! = f(νt, u, v, w)` is still accepted.
  `step_rk3!`, `step_imex!`, `step_adaptive!` and the sound-proof cores
  take the same keyword; see [Era VI](era6.md).

The pressure solver is chosen at construction (`solver = …`, any
`AbstractPressureSolver`, e.g. `MultigridPoissonSolver` or
`MGPreconditionedCG`).

## Turbulence diagnostics

```@docs
Aeolus.Simulations.horizontal_mean
Aeolus.Simulations.horizontal_variance
Aeolus.Simulations.resolved_flux_wθ
```

## Low-level stepping

```@docs
Aeolus.Simulations.step_rk3!
```
