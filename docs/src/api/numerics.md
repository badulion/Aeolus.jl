# Numerical foundations

These modules deliver the numerical fidelity needed for real-Reynolds
urban LES: geometry that resolves walls rather than staircasing them,
time integration that is not hostage to the stiffest term, advection
that does not manufacture scalar variance, a sound treatment of
compressibility, a pressure solver that scales, and mesh refinement.
See the [roadmap](../design/roadmap.md) for the
validation results behind each.

- [Cut-cell IBM and ghost-cell walls (Phases 21–22)](numerics/cutcell.md) — `Aeolus.CutCell`
- [Stretched and terrain-following grids (Phase 24)](numerics/stretched_grids.md) — `Aeolus.StretchedGrids`
- [Adaptive time stepping and IMEX integration (Phases 23, 25)](numerics/time_integration.md) — `Aeolus.TimeIntegration`
- [WENO-Z / TENO advection and conservative operators (Phases 26, 28)](numerics/advection.md) — `Aeolus.AdvectionSchemes`
- [Anelastic and pseudo-incompressible dynamics (Phase 27)](numerics/anelastic.md) — `Aeolus.Anelastic`
- [Geometric multigrid pressure solver (Phase 29)](numerics/multigrid.md) — `Aeolus.Multigrid`
- [Parent ↔ child grid nesting (Phase 30)](numerics/nesting.md) — `Aeolus.Nesting`
- [Block-structured AMR (Phase 31)](numerics/adaptive_mesh.md) — `Aeolus.AdaptiveMesh`
