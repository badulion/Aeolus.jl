# Adaptive time stepping and IMEX integration (Phases 23, 25)

Explicit diffusion imposes a ``\Delta t \sim \Delta z^2 / \nu`` limit
that becomes crippling in the stretched near-surface layers. Treating
diffusion implicitly while keeping advection explicit removes that
limit without paying for a fully implicit solve.

```@autodocs
Modules = [Aeolus.TimeIntegration]
```
