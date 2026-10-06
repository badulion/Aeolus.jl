# PressureSolvers

Pluggable Poisson-solver interface used by the fractional-step driver.
See [Discrete Poisson & FFT](../theory/poisson.md) for the mathematics
and [Pluggable pressure solvers](../design/pressure_solvers.md) for the
design.

```@docs
Aeolus.PressureSolvers.AbstractPressureSolver
Aeolus.PressureSolvers.FFTPoissonSolver
Aeolus.PressureSolvers.CosineFFTPoissonSolver
Aeolus.PressureSolvers.solve!
Aeolus.PressureSolvers.poisson_axis_bcs
```

## Iterative solver

```@docs
Aeolus.PressureSolvers.CGPoissonSolver
```

## Kernels of the spectral solvers

The FFT, CosineFFT and stretched solvers run their pointwise steps as
[`launch!`](@ref Aeolus.Backends.launch!) kernels and their transforms
through [`apply_transform!`](@ref Aeolus.Backends.apply_transform!) from
the host (KA-h, #298); see [The kernel port](../design/ka_port.md).

```@docs
Aeolus.PressureSolvers.copy_interior_in!
Aeolus.PressureSolvers.copy_interior_out!
Aeolus.PressureSolvers.scale_by!
```
