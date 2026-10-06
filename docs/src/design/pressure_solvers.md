# Pluggable pressure solvers

The pressure-Poisson solve is the single most consequential part of an
incompressible LES code:

  * It runs once per time step (more, with a projected RK3 scheme).
  * Its cost dominates the run.
  * It is the piece most likely to change when boundary conditions or
    grid geometry change.

Aeolus wraps it in an `AbstractPressureSolver` interface so different
solvers can be swapped in without touching the fractional-step driver.

## The interface

```julia
abstract type AbstractPressureSolver end

solve!(p::Field{CellCenter}, source::Field{CellCenter}, solver::AbstractPressureSolver)
```

That's the entire contract. A `solver` is constructed once — typically
at simulation setup — and reused every time step. `solve!` populates
the interior of `p` from the interior of `source`; halos are left
untouched.

## Current: `FFTPoissonSolver`

The only shipped implementation is [`FFTPoissonSolver`](@ref
Aeolus.PressureSolvers.FFTPoissonSolver): direct solve via real FFT on
a triply-periodic `RegularGrid`. See [Discrete Poisson &
FFT](../theory/poisson.md) for the derivation.

At construction:

  * Preallocate a real buffer `(Nx, Ny, Nz)` and a complex buffer
    `(Nx÷2+1, Ny, Nz)`.
  * Plan a real forward FFT (`plan_rfft`) and a real inverse FFT
    (`plan_irfft`).
  * Precompute the discrete-Laplacian eigenvalues on the complex-buffer
    grid, storing their reciprocals. The `(0,0,0)` mode's reciprocal is
    stored as `0` — this projects out the zero-mean component.

At each `solve!` call:

  * Copy interior of `source` into the real buffer.
  * Forward FFT.
  * Multiply element-wise by the precomputed reciprocal eigenvalues.
  * Inverse FFT.
  * Copy into the interior of `p`.

No heap allocation. The forward+inverse FFT pair dominates the cost:
`O(Nx Ny Nz log(Nx Ny Nz))`.

**On threads (KA-h, #298).** The copies, the eigenvalue multiply and the
normalisation are pointwise [`launch!`](@ref Aeolus.Backends.launch!)
kernels, bit-identical to the broadcasts they replaced. The transforms
are executed from the host between launches through
[`apply_transform!`](@ref Aeolus.Backends.apply_transform!), never inside
a kernel, so a GPU backend can substitute its own FFT library. FFTW's
plans are threaded (`init_threads!` sets FFTW to `Threads.nthreads()`)
and give the same bits at every thread count under `ESTIMATE` or
imported wisdom. The stretched solver's tridiagonal solve in z is one
launch over the ky rows, serial in z within each work item. Measured
speed-ups: [The kernel port](ka_port.md#Measured-at-KA-h).

## Planned solvers

### `CosineFFTPoissonSolver` *(Phase 1b — shipped)*

For channel-like domains — periodic in `x, y`, walls in `z`. Uses a
Type-II discrete cosine transform in `z` (matches the pressure-Neumann
BC at the walls) combined with a real FFT in `x, y`. Cost is still
`O(N log N)`. Passes the manufactured-Poisson test to machine precision;
see [Discrete Poisson & FFT](../theory/poisson.md#Extension-to-channel-BCs:-the-cosine-transform).

Selecting it is a constructor-time decision:

```julia
solver = CosineFFTPoissonSolver(mesh)
sim    = BoussinesqSimulation(mesh; solver = solver, θ_bcs = θ_bcs, bcs = bcs)
```

The rest of the driver is unchanged. This unlocks the Dry Convective
Boundary Layer benchmark and everything downstream in Phase 1b.

#### Per-axis boundary kinds (#251)

`CosineFFTPoissonSolver(mesh; bc_x = :periodic, bc_y = :periodic, bc_z = :neumann)`:
every axis is `:periodic` (rfft) or `:neumann` (DCT-II). Neumann sides are
an option of this one type, not a new solver — one source of truth for
the transform solve. The default keeps the channel plans, eigenvalues and
normalisation bit for bit. `poisson_axis_bcs(solver)` reports the axes of
any solver (FFT, cosine, CG, multigrid, MG-CG, density-weighted CG).

#### Choosing the axes from the boundary conditions

The pressure condition is not chosen separately: it follows from the
velocity faces. `BoundaryConditions.axis_poisson_bc(face)` returns
`:periodic` for `Periodic` and `:neumann` for every face on which the
normal velocity is prescribed (`ImpermeableWall`, `NoSlipWall`, `NoFlux`,
and every `AbstractOpenBoundary`: `FreeStreamBoundary` and the
data-driven `OpenBoundary` of N-open-b). A
face type with no method throws, so a solver is never silently built for
the wrong problem. `PressureSolvers.poisson_axis_bcs(bcs)` pairs the two
faces of each axis and rejects periodic-against-Neumann.

`Simulations._default_solver(mesh::RegularGrid, bcs)` uses it: triply
periodic gives `FFTPoissonSolver`, anything else
`CosineFFTPoissonSolver(mesh; poisson_axis_bcs(bcs)...)` — the channel as
before, the all-Neumann box for a nested child. The sound-proof cores
pass the same axes to their `DensityWeightedPoissonSolver`, and the Berlin
`:pressure_solver` providers (`examples/urban/couplings/pressure_solver.jl`)
build every solver from the case's `bcs` the same way, with a smoke check
that the axes match.

### `CGPoissonSolver` (Phase 3)

Matrix-free conjugate gradient on the same 7-point operator, periodic or
Neumann per axis (`bc_x`, `bc_y`, `bc_bottom`, `bc_top`). Until #251 it
stored `bc_xy` but always wrapped `x` and `y` periodically — a code bug that
silently returned the periodic solution for `bc_xy = :neumann`. Intended
for immersed boundaries, cut cells — anything where a direct
transform-based solver fails. Cost per solve is
`O(N · √cond) ≈ O(N^{4/3})` for a Laplacian with mild
preconditioning. The preconditioner will start as a simple diagonal
scaling and can be upgraded to geometric multigrid.

A solve allocates nothing (#295): the source, the iterate and the
scratch of the dot products live in the solver, as they do in
`MultigridPoissonSolver` (source buffer) and `MGPreconditionedCG`. The
dot products are still `sum(a .* b)` — the product is written into the
scratch array first — so the pairwise summation, the bits and the
iteration counts are those of the allocating version. On the 64 × 64 ×
32 flagship grid a plain CG solve went from 11.0 to 7.3 ms, multigrid
from 60.9 to 55.3 ms and MG-CG from 41.9 to 39.0 ms (1 thread, minimum
of 2 × 5 calls on a loaded 24-core machine; most of the CG gain is
garbage collection that no longer runs).

### `SEMPoissonSolver` (Phase 5)

For the spectral-element backend. Element-local mass is not diagonal
(GLL weights per element), so a matrix-free CG or a block-Jacobi
multigrid is needed. This is what NekRS-family codes use.

## What `Simulations.step!` sees

The driver holds only `sim.solver::AbstractPressureSolver`. Its call
site is a single line:

```julia
solve!(sim.φ, sim.div_buf, sim.solver)
```

No conditional; no interface knowledge. Swapping solvers is a
constructor-time decision.

```julia
# Default: FFT, triply periodic
sim1 = BoussinesqSimulation(mesh; ν=1e-2)

# Custom solver — e.g. a future channel solver
solver = CosineFFTPoissonSolver(mesh)
sim2   = BoussinesqSimulation(mesh; ν=1e-2, solver=solver)
```

`step!` is bit-identical between the two.

## Testing new solvers

The unit-test recipe for any new solver is the same as for FFT:

1. **Manufactured solution.** Prescribe an analytic `φ`, apply the
   discrete Laplacian kernel to get `source = L_h φ`, call `solve!` and
   check `‖φ_num − φ‖∞ < 10⁻¹⁰`.
2. **Divergence-removal.** Fill `u, v, w` with random values, call the
   full projection step, check `‖∇·u‖∞ < 10⁻¹⁰`.

Both patterns are in `test/runtests.jl` for the FFT solver.

## When *not* to build a new solver

If you only need to change boundary conditions on `u, v, w` (e.g. add
a slip wall) but keep the domain topology, extending `BoundaryConditions`
is enough — the pressure BC follows automatically from the discrete
Laplacian used in projection. New pressure solvers only earn their
keep when the domain topology, BC type on pressure itself, or grid
geometry actually changes.
