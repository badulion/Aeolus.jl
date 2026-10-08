# Benchmarks

The `benchmarks/` directory contains three BenchmarkTools.jl scripts:

  * `bench_operators.jl` — individual finite-volume kernels at 64³.
  * `bench_poisson.jl` — FFT Poisson solver scaling with grid size.
  * `bench_step.jl` — end-to-end `BoussinesqSimulation.step!` at
    several grid sizes and physics combinations.

Set up the environment once:

```bash
$ julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

then run any of the scripts:

```bash
$ julia --project=benchmarks benchmarks/bench_operators.jl
$ julia --project=benchmarks benchmarks/bench_poisson.jl
$ julia --project=benchmarks benchmarks/bench_step.jl
```

## Multithreading scaling

`bench_ka_scaling.jl` (KA-o, #308) times the KernelAbstractions port at
a given thread count: the Berlin flagship with and without its nest
parent (`timing_report` per entry, `step_dt` separately), the 256³ DNS
per step, idealised memory- and compute-bound kernels at 64 × 64 × 32 and
256³, and the launch overhead. It runs in the main project, one process
per thread count:

```bash
$ julia -t 12 --project=. benchmarks/bench_ka_scaling.jl flagship 2 9
$ AEOLUS_FFTW_PLANNER=estimate julia -t 12 --project=. benchmarks/bench_ka_scaling.jl kernels
```

The measured tables at 1/4/12/24 threads and their reading are in
[the kernel port design page](design/ka_port.md), section "KA-o: the scaling report".

## Reference numbers

Absolute numbers below are from the development machine, single-core,
Julia 1.12, no threading. *Relative* scaling and per-point costs should
be portable.

### Individual kernels (grid 64³ = 262 144 points)

```text
Aeolus operator benchmarks — grid 64³ = 262 144 points
─────────────────────────────────────────────────────────────
  ∂x   (CellCenter → FaceX)         0.046 ms   0.2 ns/point
  ∂x   (CellCenter → CellCenter)    0.043 ms   0.2 ns/point
  divergence (FaceX,Y,Z → CC)       0.112 ms   0.4 ns/point
  laplacian  (CellCenter)           0.120 ms   0.5 ns/point
  laplacian  (FaceX)                0.120 ms   0.5 ns/point
  momentum_advect!                  0.865 ms   3.3 ns/point
  advect_centered! (staggered θ)    0.265 ms   1.0 ns/point
  compute_smagorinsky_νt!           1.276 ms   4.9 ns/point
  add_eddy_viscosity_u!             0.312 ms   1.2 ns/point
  add_buoyancy!                     0.062 ms   0.2 ns/point
  add_coriolis!                     0.198 ms   0.8 ns/point
  fill_halos! (CellCenter)          0.046 ms   0.2 ns/point
─────────────────────────────────────────────────────────────
```

Takeaways:

  * The heavy kernels are Smagorinsky (`compute_smagorinsky_νt!`) at
    ~5 ns/point and staggered momentum advection at ~3 ns/point. Both
    do ~30–50 arithmetic ops per cell.
  * The compact stencils (`∂x`, `add_buoyancy!`) sit at ~0.2 ns/point,
    consistent with a memory-bandwidth-limited kernel touching 3
    doubles per cell.
  * Eddy-viscosity flux divergences (per component) are ~1 ns/point.

### FFT Poisson solver

```text
FFTPoissonSolver — solve! scaling (triply periodic)
──────────────────────────────────────────────────
      N      cells   median (ms)   ns/cell
──────────────────────────────────────────────────
     16       4 096         0.02      4.8
     32      32 768         0.16      4.9
     64     262 144         1.50      5.7
     96     884 736         6.63      7.5
    128   2 097 152        18.21      8.7
──────────────────────────────────────────────────
```

The per-cell cost grows *logarithmically* with `N` (from 4.8 to 8.7
ns/cell over an 8× resolution range), matching the theoretical
`O(N³ log N)` FFT complexity.

### End-to-end `step!`

```text
BoussinesqSimulation step! — scaling with grid size
────────────────────────────────────────────────────────────────
   N     cells      physics                     median (ms)   ns/cell
────────────────────────────────────────────────────────────────
  16     4 096   advect + project only              0.13    31.7
  16     4 096   + SGS                              0.18    43.2
  16     4 096   + SGS + buoyancy                   0.18    44.0
  32    32 768   advect + project only              0.65    19.9
  32    32 768   + SGS                              0.98    30.0
  32    32 768   + SGS + buoyancy                   0.99    30.3
  64   262 144   advect + project only              7.39    28.2
  64   262 144   + SGS                              9.87    37.7
  64   262 144   + SGS + buoyancy                  10.04    38.3
  96   884 736   advect + project only             26.39    29.8
  96   884 736   + SGS                             35.59    40.2
  96   884 736   + SGS + buoyancy                  36.02    40.7
────────────────────────────────────────────────────────────────
```

Takeaways:

  * **Per-point cost is roughly constant** at 30–40 ns/cell — the
    hallmark of a memory-bandwidth-limited stencil code that scales
    linearly.
  * **SGS costs ~30–35 %** over advect+project-only. Consistent with
    an extra strain-rate pass (`compute_smagorinsky_νt!`) plus three
    flux-form eddy-viscosity divergences plus one for the scalar.
  * **Buoyancy is essentially free** — a single pass touching a
    cell-center and a face-face pair.

## Reading the numbers

**Time-per-grid-point** is the most portable metric. For a well-tuned
FV code on modern CPUs it should sit in the 20–50 ns/point range for a
full LES time step. Kernels that fit in L2 (up to ~4 MB per array at
double precision, so ~64³) can be faster.

If you see the per-point cost *grow* with `N`, cache is thrashing —
consider tiling. If you see it *decrease*, the kernel is not warm yet —
increase `samples` or `evals` in the `@benchmark` macro.

## Threads / SIMD

None of the current kernels use `@threads` or explicit SIMD. Adding
`Threads.@threads` to the `k`-loop in the `RegularGrid` kernels is a
straightforward next step and will give near-linear scaling to the
core count. GPU kernels via `KernelAbstractions.jl` are Phase 5.
