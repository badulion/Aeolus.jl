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

## Column-separable direct solver

The FFT/DCT + tridiagonal solver behind `StretchedPoissonSolver` and the
anelastic `DirectWeightedPoissonSolver` (#309).

```@docs
Aeolus.PressureSolvers.ColumnTridiagonalSolver
```

## Iterative solver

```@docs
Aeolus.PressureSolvers.CGPoissonSolver
Aeolus.PressureSolvers.fill_unit_halo!
Aeolus.PressureSolvers.copy_unit_halo_in!
```

`CGPoissonSolver`, `MGPreconditionedCG`, `DensityWeightedPoissonSolver`
and `solve_implicit_diffusion!` all iterate with the one CG driver
[`pcg!`](@ref Aeolus.Backends.pcg!) (KA-g, #300).

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
