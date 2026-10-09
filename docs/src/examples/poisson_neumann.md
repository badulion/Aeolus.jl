# Manufactured-solution Poisson (Neumann in z)

Same test as [Manufactured-solution Poisson (periodic)](poisson.md),
but for channel-type BCs — periodic in `x, y`, Neumann in `z` — using
the [`CosineFFTPoissonSolver`](@ref
Aeolus.PressureSolvers.CosineFFTPoissonSolver).

Location: the testset "CosineFFTPoissonSolver on a Neumann-BC
manufactured mode" of `test/runtests.jl` (64³, `n = 3`; formerly
`examples/poisson_neumann_manufactured.jl` — a solver check, not an
example, so it moved to the tests in EX-d, #441).

```bash
$ julia --project=. test/run.jl --only runtests.jl
```

## Setup

```julia
φ_exact(x, y, z) = cos(2π x/Lx) * cos(2π y/Ly) * cos(π n z/Lz)
```

The `cos(π n z/Lz)` factor satisfies `∂φ/∂z = 0` at `z = 0` and
`z = Lz` for any integer `n`. `64³` grid over the unit cube, `n = 3`.

## Expected output

The test asserts an error below 1e-10; measured (2026-10-09, the same
code the script ran):

```text
N = 64³   z-mode n = 3
||p_num - φ_exact||∞ = 6.661e-16     (expected ≲ 1e-10)
```

## What to look for

Just as with the periodic solver, the error sits at **floating-point
roundoff**. This is the property that lets the walls-in-z fractional-
step projection produce `∇·u ≈ 10⁻¹⁵`.

The recipe is identical to the periodic case:

1. Choose an analytic `φ` compatible with the BCs.
2. Reflect the halo in `z` (`NoFlux`) — this is the discrete Neumann BC.
3. Apply our stencil `laplacian!` to `φ` to get the source.
4. Call `solve!(p_num, source, solver)`.
5. Compare `p_num` to `φ`.
