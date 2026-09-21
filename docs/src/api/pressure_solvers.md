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
```
