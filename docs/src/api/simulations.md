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
Aeolus.Simulations.fill_state_halos!
Aeolus.Simulations.max_νt
Aeolus.Simulations.mean_θ_profile
```

## The sponge target (N-open-d, #262)

The Rayleigh sponge of every core relaxes `u, v, w` toward
`sim.sponge_target`: zero below a lid (the Phase-7 default), the
boundary data's state below an open top (required there). Theory:
[Near the open faces](../theory/fractional_step.md#Near-the-open-faces-(N-open-d)).

```julia
data = ProfileBoundaryData(u = 5.0, v = -1.5, θ = 300.0)
sim  = BoussinesqSimulation(mesh; bcs = open_boundary_conditions(mesh, data),
                            sponge_α_max = 0.05, sponge_z_start = 300.0,
                            sponge_target = sponge_target(data, mesh))
```

```@docs
Aeolus.Simulations.sponge_target
Aeolus.Simulations.add_velocity_sponge!
```

## Swapping operators in `step!`

Two keywords of `step!(sim, dt; …)` replace a numerical operator without
changing the algorithm (both default to the built-in one):

* `momentum_advection = f(tu, tv, tw, u, v, w)` — the momentum-advection
  operator, e.g. `(tu, tv, tw, u, v, w) -> momentum_advect_weno_z!(tu, tv, tw, u, v, w)`
  from `AdvectionSchemes`;
* `sgs = closure` — the subgrid-scale closure, any
  [`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) of the
  SGS closure contract ([`StaticSmagorinsky`](@ref Aeolus.Operators.StaticSmagorinsky),
  [`DynamicSmagorinsky`](@ref Aeolus.Operators.DynamicSmagorinsky), an
  `ML.MLSGS`, …) or a plain function `f(νt, u, v, w)` that fills `νt`
  (wrapped in a [`FunctionSGS`](@ref Aeolus.Operators.FunctionSGS),
  `κt = νt/Prₜ`). It switches the SGS terms on even with `C_s = 0`, and
  they are added where the built-in closure adds them, so
  `StaticSmagorinsky(C_s)` — or an `MLSGS` wrapping the Smagorinsky
  closure — reproduces the built-in run bit for bit. The pre-contract
  keyword name `eddy_viscosity! = f(νt, u, v, w)` is still accepted.
  `step_rk3!`, `step_imex!`, `step_adaptive!` and the sound-proof cores
  take the same keyword; see [Turbulence closure](turbulence.md).

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
Aeolus.Simulations.RK3Workspace
Aeolus.Simulations.rk3_workspace
Aeolus.Simulations.StepWorkspaces
```

### The step record and the diagnostics hook (#336)

Every step core fills the simulation's
[`StepRecord`](@ref Aeolus.Simulations.StepRecord) and calls its hook at
the end of each step; the run monitor attaches to it. See
[Per-step diagnostics](logging.md#Per-step-diagnostics-in-the-step-cores).

```@docs
Aeolus.Simulations.StepRecord
Aeolus.Simulations.step_record
Aeolus.Simulations.set_step_hook!
Aeolus.Simulations.step_hook
Aeolus.Simulations.reset_interval!
Aeolus.Simulations.step_begun
Aeolus.Simulations.guard_covered
```

### Field algebra of the step cores (KA-j, #306)

The drivers update their fields only through pointwise kernels, which
give the bits of the broadcasts they replaced at every thread count
([the kernel port](../design/ka_port.md)).

```@docs
Aeolus.Simulations.axpy!
Aeolus.Simulations.pressure_from_potential!
Aeolus.Simulations.copy_storage!
Aeolus.Simulations.zero_storage!
```
