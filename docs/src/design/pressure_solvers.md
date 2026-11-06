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
launch over slabs of ky rows, serial in z within each work item. Measured
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
pass the same axes to their `DirectWeightedPoissonSolver` (or
`DensityWeightedPoissonSolver`), and the Berlin
`:pressure_solver` providers (`src/components/urban/pressure_solver.jl`)
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

A solve allocates nothing (#295): the source, the iterate and the CG
work arrays live in the solver, as they do in `MultigridPoissonSolver`
(source buffer) and `MGPreconditionedCG`.

**One CG loop (KA-g, #300).** `CGPoissonSolver`, `MGPreconditionedCG`,
`DensityWeightedPoissonSolver` and the implicit diffusion of the IMEX
steppers all run [`pcg!`](@ref Aeolus.Backends.pcg!) in `Backends`. A
solver supplies only its operator (a point stencil, `PointOperator`, or
a function such as the multigrid fine-grid operator), its preconditioner
(none, a pointwise Jacobi step, or a V-cycle), whether the constants are
projected out, and its pinned rows. The CG and density-weighted solvers
keep their iterates in arrays with one halo layer that
`fill_unit_halo!` fills (periodic wrap or Neumann mirror) before each
operator application, so the stencil carries no wrap-around index
arithmetic. The dot products are canonical plane sums, so a solve and
its iteration count are identical at every thread count. Iteration
counts, timings and the one-time last-bit change are in
[The kernel port](ka_port.md) (section KA-g).

**The V-cycle on the kernel layer (KA-i, #303).** The multigrid
smoother, residual, restriction and prolongation are `launch!`es of
pointwise bodies with the bits of the old loops. The red-black smoother
makes one launch per colour over that colour's cells only. The means
and norms of the null-space projection and of the convergence test are
canonical plane sums, so `MultigridPoissonSolver` and the V-cycle
preconditioner of `MGPreconditionedCG` give the same solution and cycle
count at every thread count. Levels below the launch threshold
(32 768 points), including the coarsest solve, run serially. Timings,
the per-level split and the last-bit change are in
[The kernel port](ka_port.md) (section KA-i).

### Column-separable operators: `ColumnTridiagonalSolver` (#309)

Two operators of the code have uniform horizontal spacing and
coefficients that depend on height only: the conservative stretched-z
Laplacian of the Boussinesq core on a `StretchedGrid` (#151) and the
density-weighted projection operator ∇·(w(z)∇φ) of the anelastic and
pseudo-incompressible cores (w = ρ̄ or ρ̄θ̄). Both are

```math
h_k\,(δ_x^2 + δ_y^2)φ_k + a_k(φ_{k-1} - φ_k) + c_k(φ_{k+1} - φ_k) = b_k ,
```

which the horizontal real FFT (periodic side) or DCT-II (Neumann side)
reduces to one tridiagonal system per wavenumber (Schumann & Sweet
1988). [`ColumnTridiagonalSolver`](@ref Aeolus.PressureSolvers.ColumnTridiagonalSolver)
is the one implementation: transform, Thomas sweep with the
factorisation stored at construction (a launch over slabs of ky rows, see
above), the singular mean mode integrated as a column flux, inverse
transform. `StretchedGrids.StretchedPoissonSolver` (`h = 1`) and
`Anelastic.DirectWeightedPoissonSolver` (`h = w_c`, `a_k = w_f[k]/Δz²`)
only build its coefficients; the latter is the default projection of
the sound-proof cores (`pressure_solver = :direct`), with
`DensityWeightedPoissonSolver` (CG, `pressure_solver = :cg`) kept for
a periodic column and as the independent check.

Measured (#309, one thread, isothermal 300 K base state over 3 km,
random source; CG at the cores' tolerances):

| grid | CG tol 1e-10 | CG tol 1e-12 | direct | speed-up |
| :-- | --: | --: | --: | --: |
| 32 × 32 × 32 | 12.4 ms (186 it.) | 14.5 ms (206 it.) | 0.10 ms | 124–145× |
| 64 × 64 × 64 | 176 ms (367 it.) | 175 ms (409 it.) | 1.05 ms | 166× |
| 128 × 128 × 64 | 1868 ms (612 it.) | 1960 ms (696 it.) | 8.1 ms | 231–242× |

A 64³ anelastic step (warm bubble, fixed 1 s) costs 6.1 ms with the
direct solve against 173 ms with the CG (tol 1e-10). The direct solution
satisfies the discrete equation to a relative residual of ~5e-16 and
agrees with a tol-1e-14 CG to ~2e-13 (`test/anelastic.jl`); the
weighted constraint after a step is at round-off (~2e-16 s⁻¹ against
~8e-15 s⁻¹ for the CG at tol 1e-12). A solve allocates nothing.

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
