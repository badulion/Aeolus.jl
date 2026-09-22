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
