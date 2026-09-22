# Era II — Numerical foundations

These modules deliver the numerical fidelity needed for real-Reynolds
urban LES: geometry that resolves walls rather than staircasing them,
time integration that is not hostage to the stiffest term, advection
that does not manufacture scalar variance, a sound treatment of
compressibility, a pressure solver that scales, and mesh refinement.
See the [long-range roadmap](../design/longterm_roadmap.md) for the
validation results behind each.

## Cut-cell IBM and ghost-cell walls (Phases 21–22)

Buildings intersect grid cells at arbitrary angles. A cut-cell
treatment carries the *fluid* volume and wetted area of each partial
cell, so the discrete divergence remains conservative on a body-fitted
boundary; ghost-cell reconstruction then imposes the wall condition to
second order rather than by first-order masking.

```@autodocs
Modules = [Aeolus.CutCell]
```

## Stretched and terrain-following grids (Phase 24)

```@autodocs
Modules = [Aeolus.StretchedGrids]
```

## Adaptive time stepping and IMEX integration (Phases 23, 25)

Explicit diffusion imposes a ``\Delta t \sim \Delta z^2 / \nu`` limit
that becomes crippling in the stretched near-surface layers. Treating
diffusion implicitly while keeping advection explicit removes that
limit without paying for a fully implicit solve.

```@autodocs
Modules = [Aeolus.TimeIntegration]
```

## WENO-Z / TENO advection and conservative operators (Phases 26, 28)

```@autodocs
Modules = [Aeolus.AdvectionSchemes]
```

## Anelastic and pseudo-incompressible dynamics (Phase 27)

```@autodocs
Modules = [Aeolus.Anelastic]
```

## Geometric multigrid pressure solver (Phase 29)

```@autodocs
Modules = [Aeolus.Multigrid]
```

## Production nesting and block-structured AMR (Phases 30–31)

```@autodocs
Modules = [Aeolus.AdaptiveMesh]
```
