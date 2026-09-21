# Aeolus.jl

*A Julia-native large-eddy simulation framework for atmospheric, oceanic,
and engineering boundary-layer flows.*

Aeolus is a fresh, from-scratch Julia implementation of a general-purpose
LES/DNS solver. It is inspired by PALM (the German atmospheric LES) and by
Oceananigans.jl (Julia LES for the ocean), but is a new codebase that
targets a broader scope — atmospheric ABL, urban microscale, oceanic
mixed-layer, and engineering CFD in a single framework.

## Scope and status

Aeolus is under active development. The public API is not yet stable.

* **Phase 0 (foundations)** — done. Staggered Arakawa-C mesh, halo-fill,
  FV operators, SSP-RK3 stepper, scalar-advection smoke test.
* **Phase 1a (dry Boussinesq core)** — done. Face-centered velocity fields,
  staggered momentum advection, FFT Poisson solver, fractional-step
  projection, Smagorinsky-Lilly SGS, buoyancy, Coriolis.
* **Phase 1b (DCBL validation)** — next.
* Phases 2–5 in [`design/roadmap.md`](design/roadmap.md).

## Design philosophy

Two ideas drive the codebase:

1. **The spatial discretization is behind an interface.** All user-facing
   code is written against [`AbstractMesh`](@ref Aeolus.Grids.AbstractMesh)
   and [`AbstractPressureSolver`](@ref Aeolus.PressureSolvers.AbstractPressureSolver).
   Adding a spectral-element backend later is a new subtype + new methods —
   not a rewrite. See [Design → Architecture](design/architecture.md).

2. **Multiple dispatch selects discrete stencils.** A `Field` carries its
   staggered [`Location`](@ref Aeolus.Grids.Location) as a type parameter.
   Calls like `∂x(dpdx, p)` resolve to the pressure-gradient stencil when
   `dpdx isa Field{FaceX}` and `p isa Field{CellCenter}`, and to the
   divergence stencil when the roles are reversed. See
   [Design → Dispatch](design/dispatch.md).

## Where to start

  * [Installation](installation.md)
  * [Quick start](quickstart.md) — a first Taylor-Green vortex run.
  * [Theory → Overview](theory/overview.md) — mathematical background.
  * [API reference](api/grids.md) — per-module docstrings.

## Validation snapshots

Phase 0/1a shipping tests (`Pkg.test()`):

| Test suite                                        | Result           |
| :------------------------------------------------ | :--------------- |
| Manufactured Poisson (32³, sinusoidal mode)       | ‖err‖∞ ≈ 10⁻¹⁵   |
| 3D Taylor-Green (32³, ν = 10⁻², t = 2)            | KE/KE₀ = 0.881 (analytic 0.887) |
| Fractional-step: random field → projection        | max ∇·u ≈ 10⁻¹¹  |
| Buoyant convection (32³, coupled full system)    | max ∇·u ≈ 10⁻¹⁶ |

37 unit tests pass on Julia 1.12.
