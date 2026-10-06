# Discrete Poisson & the FFT solver

The [pressure projection](fractional_step.md) requires solving a Poisson
equation once per time step:

```math
\nabla^2 \phi = f, \qquad f = \nabla\!\cdot\! u^*.
```

For triply-periodic boundary conditions on a uniform Cartesian grid,
this can be solved *directly* in `O(N log N)` operations by an FFT.
This page derives the solver used by Aeolus and explains why it is
exact on the discrete operator.

## The discrete Laplacian in Fourier space

On a periodic grid of length `L` with `N` cells and spacing `Δ = L/N`,
the 3-point centered second-difference operator `Δ²/Δx² φ`, applied to
a Fourier mode

```math
\phi_i = e^{i\, 2\pi m i / N},
```

produces an eigenvalue:

```math
\frac{\phi_{i+1} - 2\phi_i + \phi_{i-1}}{\Delta x^2}
= \frac{e^{i 2\pi m/N} - 2 + e^{-i 2\pi m/N}}{\Delta x^2}\,\phi_i
= \frac{2\cos(2\pi m/N) - 2}{\Delta x^2}\,\phi_i
= -\frac{4}{\Delta x^2}\sin^2\!\left(\frac{\pi m}{N}\right)\phi_i.
```

So each Fourier mode `m` is an eigenvector, and the eigenvalue is

```math
\lambda_m^{(x)} = -\frac{4}{\Delta x^2}\sin^2\!\left(\frac{\pi m}{N_x}\right).
```

These are the **modified wavenumbers** — sometimes written
`k_m^\ast = (2/\Delta x)\sin(\pi m/N)`. They match the *continuous*
Fourier eigenvalues `-(2\pi m/L)^2 = -k_m^2` only in the long-wave limit
`m \ll N`; the difference grows toward the Nyquist frequency.

The full 3-D discrete Laplacian has eigenvalues

```math
\lambda_{m_x m_y m_z} = \lambda^{(x)}_{m_x} + \lambda^{(y)}_{m_y} + \lambda^{(z)}_{m_z}.
```

## Solving in Fourier space

Because the discrete Laplacian is diagonal in the discrete Fourier
basis, solving `∇²φ = f` becomes:

1. Forward FFT: `f̂ = 𝓕(f)`.
2. Divide: `φ̂ = f̂ / λ`.
3. Inverse FFT: `φ = 𝓕⁻¹(φ̂)`.

The zero mode `(0, 0, 0)` has `λ = 0`. For periodic BCs the pressure is
only defined up to an additive constant; we set `φ̂[0,0,0] = 0`, which
also enforces the compatibility condition `⟨f⟩ = 0` (the source must
have zero mean for a solution to exist).

## Real FFTs

The source and solution are real. Aeolus uses `FFTW.plan_rfft` which
maps a real array of size `(Nx, Ny, Nz)` to a complex array of size
`(Nx÷2 + 1, Ny, Nz)`, halving both memory and cost. The eigenvalue
array is precomputed at solver construction time.

## Why the recovered `φ` is *exact*

If we solved the *continuous* Poisson equation with spectral
eigenvalues `-k_m²`, applying our *discrete* Laplacian to the result
would give an answer off by the wavenumber-modification error. By
using the discrete eigenvalues `λ_m` instead, we recover the exact
right-inverse of the discrete operator:

```math
L_h\!\cdot\! L_h^{-1}\,f = f, \quad \text{for } f \text{ with zero mean}.
```

This is what the manufactured-solution test verifies: given `f = L_h φ`
with `φ` prescribed and `L_h` our discrete Laplacian kernel,
`solve!(φ_num, f, solver)` returns `φ_num` matching `φ` to `10⁻¹⁵`.

## Cost and scaling

Per time step: one forward `rfft`, one element-wise divide, one inverse
`rfft`. Time complexity `O(N log N)` in the total number of grid points.
For `N = 32³ = 32768` the solve is a few milliseconds on a laptop; for
`N = 128³ ≈ 2×10⁶` it is a fraction of a second.

Memory: one real buffer `(Nx, Ny, Nz)` and one complex buffer
`(Nx/2+1, Ny, Nz)`, plus the eigenvalue array — pre-allocated inside
[`FFTPoissonSolver`](@ref Aeolus.PressureSolvers.FFTPoissonSolver).

## Extension to channel BCs: the cosine transform

For a channel-like domain — periodic in `x, y` but walls in `z` — the
appropriate transform in `z` is a **discrete cosine transform**
(DCT-II) rather than a full FFT. This is the workhorse of Phase 1b's
[`CosineFFTPoissonSolver`](@ref Aeolus.PressureSolvers.CosineFFTPoissonSolver).

### Neumann BC on cell-centered pressure

An impermeable wall (`w = 0`) combined with the w-momentum equation
implies `∂p/∂z = 0` at the wall. Discretely, for cell-centered pressure
with the wall between cell 0 (halo) and cell 1 (first interior), the
condition is enforced by mirror reflection:

```math
p_{i,j,0} \;=\; p_{i,j,1},
\qquad
p_{i,j,N_z+1} \;=\; p_{i,j,N_z}.
```

Applying our 3-point centered second-difference stencil at cell 1 gives

```math
\frac{p_{i,j,2} - 2p_{i,j,1} + p_{i,j,0}}{\Delta z^2}
\;=\;
\frac{p_{i,j,2} - p_{i,j,1}}{\Delta z^2},
```

a one-sided form that is second-order accurate at the wall.

### DCT-II eigenvalues

Substituting a candidate eigenfunction `p_i = cos(π (i - ½) l / N_z)`
into the reflected stencil above gives eigenvalues

```math
\lambda_l^{(z)} \;=\; -\frac{4}{\Delta z^2}\,\sin^2\!\left(\frac{\pi\, l}{2 N_z}\right),
\qquad l = 0, 1, \ldots, N_z-1.
```

Compare with the periodic case's `-4/Δz² sin²(π l/Nz)`. The extra
factor of 2 in the denominator comes from the *doubled period* of the
even-reflected signal — the same reason DCT-II is the appropriate
transform.

### Implementation

Aeolus does the mixed-BC solve in the natural order:

1. DCT-II along `z` (FFTW's `REDFT10`), real → real.
2. `rfft` along `(x, y)`, real → complex.
3. Element-wise multiply by `1/λ` (with `λ[0,0,0] = 0` projected out).
4. `irfft` along `(x, y)`, complex → real.
5. DCT-III along `z` (FFTW's `REDFT01`), the inverse of DCT-II up to a
   factor `2 N_z`.
6. Multiply the result by `1/(2 N_z)` — the only manual normalization.

Two real scratch buffers of shape `(Nx, Ny, Nz)` and one complex buffer
of shape `(Nx/2+1, Ny, Nz)` are held by the solver. No allocation per
solve.

### Neumann on any axis: the closed box of a nested child

Nothing above is specific to `z`. Each axis of the 7-point operator is
diagonalised independently (Schumann & Sweet 1988): a periodic axis by the
FFT with ``-4/Δ^2\sin^2(πm/N)``, a Neumann axis by the DCT-II with
``-4/Δ^2\sin^2(πm/2N)``, and the 3-D eigenvalue is their sum.
`CosineFFTPoissonSolver(mesh; bc_x, bc_y, bc_z)` takes the kind of every
axis (#251). The case that needs it is a **nested or open-boundary
child**: its normal velocity is prescribed on every face (from the parent
or from boundary data), and since

```math
u^{n+1}\cdot n = u^*\cdot n - Δt\,\frac{∂φ}{∂n},
```

the projection keeps those face values exactly when ``∂φ/∂n = 0``: φ is
homogeneous Neumann on all six faces, whatever the velocity condition is
called. The solve is then DCT-II in all three directions (no complex
buffer at all), and it costs no more than the channel solve: measured per
solve on this machine (minimum of 5),

| grid | periodic `x, y` (FFT+DCT) | all Neumann (3-D DCT) | multigrid, all Neumann, tol 1e-10 |
| :--- | ---: | ---: | ---: |
| 64 × 64 × 32 (Berlin child) | 1.83 ms | 1.48 ms (× 0.81) | 81 ms (24 cycles) |
| 64 × 64 × 64 (deepened child) | 3.90 ms | 2.96 ms (× 0.76) | 144 ms (22 cycles) |
| 160 × 160 × 64 | 37.9 ms | 25.9 ms (× 0.68) | 583 ms (12 cycles) |

The default solver of `BoussinesqSimulation` (and the density-weighted CG
of the sound-proof cores) derives the axis kinds from the six velocity
faces through `axis_poisson_bc(face)` (`:periodic` for `Periodic`,
`:neumann` for every face with a prescribed normal velocity); an axis
whose two faces disagree is an error.

### The discrete compatibility condition

With no Dirichlet face the operator is singular: constants are its null
space, and summing ``L_h φ`` over the domain telescopes to the net flux
of ``∇φ`` through the boundary, which is zero for periodic and Neumann
faces. So ``L_h φ = s`` is solvable only if ``\sum_{ijk} s_{ijk} = 0``.
For the projection ``s = ∇·u^*/Δt``, and ``\sum ∇·u^*\,ΔV`` is the net
volume flux through the boundary: periodic faces cancel, walls carry none,
but **prescribed inflow and outflow on open faces need not balance**. The
solvers project the mean out: an incompatible source with mean
``\bar s = Q/V`` is solved as ``L_h φ = s − \bar s``, which leaves a
uniform residual ``L_h φ − s = −\bar s`` (the test suite checks this to
1e-14) — after the projection ``∇·u = −\bar s Δt`` in *every* cell, a
uniform spurious source or sink. The open-boundary mass-flux correction
(N-open-b, `correct_boundary_mass_flux!`) therefore makes the net boundary
volume flux vanish *before* the projection, not after; see
[Prescribed-flux faces](fractional_step.md#Prescribed-flux-faces-(open-boundaries)).

### When *this* solver stops applying

The remaining cases:

  * **Dirichlet pressure faces, immersed boundaries, cut cells,
    variable coefficients**: need an iterative solver — matrix-free
    conjugate gradient (`CGPoissonSolver`, which takes the same per-axis
    periodic / Neumann kinds), the density-weighted CG of the sound-proof
    cores, or geometric multigrid.
  * **Non-uniform grids**: the horizontal FFTs still work if `dx, dy`
    are uniform; for non-uniform `dz(k)`, the DCT is replaced by a
    tridiagonal solve of the same one-dimensional operator.

All three variants slot into the same [`AbstractPressureSolver`](@ref
Aeolus.PressureSolvers.AbstractPressureSolver) interface —
`Simulations.step!` will not need to know which one it's calling.

## References

  * Schumann, U. & Sweet, R. A. (1988), *Fast Fourier transforms for
    direct solution of Poisson's equation with staggered boundary
    conditions*, JCP 75, 123–137.

  * Vanka, S. P. (1986), *Block-implicit multigrid solution of
    Navier-Stokes equations in primitive variables*, JCP 65, 138.
  * Ferziger, J. H. & Perić, M. (2020), *Computational Methods for
    Fluid Dynamics*, ch. 4 (Poisson solvers).
  * Trefethen, L. N. (2000), *Spectral Methods in MATLAB*, ch. 3
    (modified wavenumbers).
