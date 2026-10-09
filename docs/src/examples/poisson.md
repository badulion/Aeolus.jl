# Manufactured-solution Poisson

Prescribes an analytic `φ` satisfying triply-periodic BCs, forms
`source = L_h φ` using our discrete Laplacian, then solves
`L_h φ_num = source` with [`FFTPoissonSolver`](@ref
Aeolus.PressureSolvers.FFTPoissonSolver) and compares to `φ`.

Because the solver uses the *exact* eigenvalues of the same discrete
operator, the error sits at floating-point precision.

Location: the testset "FFTPoissonSolver on a manufactured mode" of
`test/runtests.jl` (32³ and 64³; formerly `examples/poisson_manufactured.jl`
— a solver check, not an example, so it moved to the tests in EX-d, #441).

```bash
$ julia --project=. test/run.jl --only runtests.jl
```

## Setup

```julia
φ_exact(x, y, z) = sin(2π x) * cos(2π y) * sin(2π z)
```

on a `64³` grid over `[0, 1]³`, triply-periodic.

## Expected output

The test asserts an error below 1e-10; measured at 64³ (2026-10-09):

```text
N = 64³   ||p_num - p_exact||∞ = 1.221e-15
expected: ≈ machine precision (< 1e-10)
```

## What to look for

The error `10⁻¹⁵` is **floating-point roundoff** — not truncation
error. This is the property that lets the fractional-step projection
produce `∇·u` at the same level, run after run. See
[Discrete Poisson](../theory/poisson.md).
