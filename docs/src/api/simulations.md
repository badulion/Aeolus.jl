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
* `eddy_viscosity! = f(νt, u, v, w)` — the SGS closure that fills `νt`,
  e.g. [`compute_dynamic_smagorinsky_νt!`](@ref Aeolus.Operators.compute_dynamic_smagorinsky_νt!)
  or an `ML.MLSGS` via `apply_mlsgs!`. It switches the SGS terms on even
  with `C_s = 0`, and they are added where the built-in closure adds
  them, so an `MLSGS` wrapping the Smagorinsky closure reproduces the
  built-in run bit for bit.

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
