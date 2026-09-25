# Era II — Numerical foundations

These modules deliver the numerical fidelity needed for real-Reynolds
urban LES: geometry that resolves walls rather than staircasing them,
time integration that is not hostage to the stiffest term, advection
that does not manufacture scalar variance, a sound treatment of
compressibility, a pressure solver that scales, and mesh refinement.
See the [long-range roadmap](../design/longterm_roadmap.md) for the
validation results behind each.

- [Cut-cell IBM and ghost-cell walls (Phases 21–22)](era2/cutcell.md) — `Aeolus.CutCell`
- [Stretched and terrain-following grids (Phase 24)](era2/stretched_grids.md) — `Aeolus.StretchedGrids`
- [Adaptive time stepping and IMEX integration (Phases 23, 25)](era2/time_integration.md) — `Aeolus.TimeIntegration`
- [WENO-Z / TENO advection and conservative operators (Phases 26, 28)](era2/advection.md) — `Aeolus.AdvectionSchemes`
- [Anelastic and pseudo-incompressible dynamics (Phase 27)](era2/anelastic.md) — `Aeolus.Anelastic`
- [Geometric multigrid pressure solver (Phase 29)](era2/multigrid.md) — `Aeolus.Multigrid`
- [Production nesting and block-structured AMR (Phases 30–31)](era2/adaptive_mesh.md) — `Aeolus.AdaptiveMesh`
