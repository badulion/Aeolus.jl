# Architecture

Aeolus is organized as a tree of small submodules, each responsible for
one concept. This page describes the module layout, the two central
abstract interfaces, and how the pieces compose.

## Module layout

```text
Aeolus                                   top-level module, re-exports the public API
├── Grids                                mesh definitions, coordinate access
│   • AbstractMesh (abstract type)
│   • RegularGrid (concrete FV mesh)
│   • CellCenter, FaceX, FaceY, FaceZ (Location types)
├── Fields                               scalar field on a mesh
│   • Field{L,T,A,M}
│   • set!, interior_view, location, mesh
├── BoundaryConditions                   per-face BCs + halo fill
│   • Periodic, NoFlux, ValueBC, FluxBC
│   • FieldBoundaryConditions
│   • fill_halos!
├── Operators                            public API for finite-difference operators
│   • ∂x, ∂y, ∂z, laplacian, divergence
│   • advect_centered!, advect_upwind!, momentum_advect!
│   • add_buoyancy!, add_coriolis!
│   • compute_smagorinsky_νt!, add_eddy_viscosity_*!, add_eddy_diffusion_scalar!
│   • interpolate_to_face!, interpolate_to_center!
│   └── FiniteVolume                     kernel implementations dispatched from Operators
├── TimeSteppers                         generic time-integration schemes
│   • RK3 (Shu-Osher SSP)
│   • step! (generic function; also extended by Simulations)
├── PressureSolvers                      Poisson-solver interface + implementations
│   • AbstractPressureSolver
│   • FFTPoissonSolver
│   • solve!
└── Simulations                          high-level model driver
    • BoussinesqSimulation
    • step! (extends TimeSteppers.step!)
    • run!, kinetic_energy, max_divergence, max_νt, mean_θ_profile
```

Roughly 1500 lines of Julia. Each module is under ~300 lines except
`FiniteVolume.jl` (~600) which contains the actual FD kernels.

## Two abstract interfaces

Two abstract types define the extension points of the codebase:

### `AbstractMesh` — the spatial discretization

Every mesh is a subtype of [`AbstractMesh{T}`](@ref
Aeolus.Grids.AbstractMesh) and must implement:

  * `totalsize(mesh)` — tuple of storage-array dimensions.
  * `interior(mesh)` — tuple of interior index ranges.
  * `xnodes(mesh, ::Location)`, `ynodes(mesh, ::Location)`,
    `znodes(mesh, ::Location)` — physical coordinates.

`RegularGrid` is the only concrete mesh so far. A future
`SpectralElementMesh` (Phase 5) will implement the same interface with
element-local GLL nodes instead of Cartesian cells. Every user-facing
call — `Field`, `set!`, `∂x`, `divergence`, `step!` — is written
against `AbstractMesh`, so a new mesh subtype requires *no* changes at
the call sites: it just needs to provide operator kernels that
dispatch on its concrete type.

### `AbstractPressureSolver` — the Poisson-solver

Every solver implements one method:

```julia
solve!(p::Field{CellCenter}, source::Field{CellCenter}, solver)
```

The Aeolus fractional-step driver calls this from `Simulations.step!`
without knowing which concrete solver is in use. Currently
`FFTPoissonSolver` (triply-periodic). Planned:

  * `CosineFFTPoissonSolver` — FFT in x, y and cosine transform in z
    for channel-like domains (Phase 1b).
  * `CGPoissonSolver` — matrix-free preconditioned conjugate gradient
    for arbitrary BCs and immersed boundaries (Phase 3).
  * `SEMPoissonSolver` — element-local mass matrix + matrix-free CG,
    for the spectral-element backend (Phase 5).

## Composition of a `BoussinesqSimulation`

At construction time
[`BoussinesqSimulation(mesh; …)`](@ref Aeolus.Simulations.BoussinesqSimulation)
allocates:

```text
mesh, bcs, θ_bcs, solver         # external references
parameters                        # ν, θ₀, g, f, C_s, Pr_t
flags                             # has_viscosity, has_buoyancy, ...

prognostic  →  u@FaceX, v@FaceY, w@FaceZ, p@CellCenter, θ@CellCenter
tendency    →  tu, tv, tw, tθ
scratch     →  div_buf, φ, dφdx, dφdy, dφdz, lap_u, lap_v, lap_w, lap_θ, νt
```

All arrays exist for the entire simulation lifetime; `step!` writes
into pre-existing buffers only. No heap allocation on the hot path.

## Directory layout

```text
Aeolus.jl/
├── Project.toml              declares the package and its deps
├── README.md                 short summary + roadmap
├── src/                      module sources (see above)
├── test/runtests.jl          37 unit tests
├── examples/                 runnable examples
│   ├── scalar_advection_2d.jl
│   ├── poisson_manufactured.jl
│   ├── taylor_green_vortex.jl
│   └── buoyant_convection.jl
├── benchmarks/               BenchmarkTools.jl scripts
│   ├── bench_operators.jl
│   ├── bench_poisson.jl
│   └── bench_step.jl
└── docs/                     Documenter.jl source
    ├── Project.toml
    ├── make.jl
    └── src/                  markdown pages (this site)
```

## Design principles

1. **No premature abstraction.** Every abstract type has at least one
   concrete implementation and a clear plan for a second. We do not
   add abstractions for hypothetical needs.
2. **Multiple dispatch selects stencils.** See
   [Multiple dispatch on Location](dispatch.md).
3. **Pre-allocate; the hot path does not touch the heap.** The
   `Simulations.step!` function contains no allocations after warm-up.
4. **Public API is thin.** `Operators` re-exports one name per
   physically meaningful operation; the underscore-prefixed backend
   kernels are internal.
5. **A single generic `step!`.** Both the SSP-RK3 stepper and the
   Boussinesq fractional-step driver add methods to the *same* generic
   function. Callers do not care which one runs.
