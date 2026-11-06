# Aeolus.jl

*A Julia-native large-eddy simulation framework for atmospheric, oceanic,
and engineering boundary-layer flows.*

Aeolus is a fresh, from-scratch Julia implementation of a general-purpose
LES/DNS solver. It is inspired by PALM (the German atmospheric LES) and by
Oceananigans.jl (Julia LES for the ocean), but is a new codebase that
targets a broader scope — atmospheric ABL, urban microscale, oceanic
mixed-layer, and engineering CFD in a single framework.

## Scope and status

Aeolus is under active development; the public API is not yet stable.
Milestones, ✅ shipped · ☐ open (every phase, and what each shipped one
measured, is in the [Roadmap](design/roadmap.md)):

* ✅ [LES core and physics scaffolds](design/roadmap.md#LES-core-and-physics-scaffolds) (Phases 1–20)
* ✅ [Numerical foundations](design/roadmap.md#Numerical-foundations) (Phases 21–31)
* ✅ [Radiation](design/roadmap.md#Radiation) (Phases 32–38)
* ✅ [Clouds and microphysics](design/roadmap.md#Clouds-and-microphysics) (Phases 39–45)
* ✅ [Land, urban and surface exchange](design/roadmap.md#Land,-urban-and-surface-exchange) (Phases 46–53)
* ✅ [Turbulence closure](design/roadmap.md#Turbulence-closure) (Phases 54–59)
* ✅ [Model infrastructure](design/roadmap.md#Model-infrastructure)
* ☐ [Chemistry and aerosol](design/roadmap.md#Chemistry-and-aerosol) (Phases 60–66)
* ☐ [Real-world data, forcing and validation](design/roadmap.md#Real-world-data,-forcing-and-validation) (Phases 67–73)
* ☐ [Data assimilation and inverse problems](design/roadmap.md#Data-assimilation-and-inverse-problems) (Phases 74–79)
* ☐ [Performance, scale and portability](design/roadmap.md#Performance,-scale-and-portability) (Phases 80–87)
* ☐ [Machine learning, applications, ecosystem](design/roadmap.md#Machine-learning,-applications,-ecosystem) (Phases 88–100)
* ☐ [Kilometre-scale regional and climate modelling](design/roadmap.md#Kilometre-scale-regional-and-climate-modelling) (Phases 101–112)

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
