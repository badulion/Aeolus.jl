# The kernel port (KernelAbstractions)

Aeolus runs single-node multithreaded through
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
(KA), with one kernel path that a GPU backend will reuse in Phase 81
(port plan #291, maintainer decision 2026-10-06). This page is the
contract every port PR follows. KA-a (#294) laid the foundation: the
dependency, `src/Backends.jl`, `set!` / `interior_map!` as the first
kernels, the deterministic reductions and the thread-invariance harness.
The API is under [Compute backends](../api/physics.md#Compute-backends).

## Rules of the port

* **All KA code lives in `src/Backends.jl`.** Every `@kernel` is defined
  there. The rest of the code base writes plain Julia *bodies* and calls
  [`launch!`](@ref Aeolus.Backends.launch!).
* **Bit-for-bit means N threads = 1 thread** on the new code. A
  *pointwise* kernel also gives the same bits as the loop it replaces. A
  call site that moves to a canonical reduction changes its bits once
  against today's master. That is allowed only where the port plan names
  it (canonical reductions, CG/MG iteration counts, the WENO power fix,
  the TKE / IBM wall-stress / radiosity gather rewrites), and the PR
  reports the change with numbers.
* **Gate per port PR: 1-thread time ≤ 1.10× the loops it replaces.**
  Quote the 4/12/24-thread speed-ups too.
* **Every port PR runs the thread-invariance harness and quotes it**
  (below).

## Backends

```julia
CPUBackend()                       # KA.CPU() on Julia's threads (julia -t N)
CPUBackend(serial = true)          # every launch on the calling task
CPUBackend(threshold = 0)          # thread even the smallest launch (tests)
GPUBackend(:cuda)                  # stub until Phase 81: errors on use
```

`default_backend()` is the backend new fields are allocated on:
`Field(loc, mesh; backend = default_backend())` allocates through
`allocate(backend, T, dims)`. `backend(field)` gives a field's backend.
`init_threads!(backend)` makes `backend` the default and sets the
library thread counts (below). `with_backend(f, backend)` runs `f` under
a temporary default; it is process-global, not task-local. The Berlin
case takes its backend from `CaseConfig.backend` and calls
`init_threads!` first thing in `build_case`.

## The launch contract

```julia
@inline function axpy_body!(i, j, k, y, a, x)
    @inbounds y[i, j, k] = y[i, j, k] + a * x[i, j, k]
    return nothing
end
launch!(backend, axpy_body!, interior(mesh), y.data, a, x.data)
```

* **Ranges are storage indices** (halos included), as a tuple of
  `UnitRange{Int}`s. Any region is one launch: the interior
  `(Hx+1:Hx+Nx, …)`; a face range with one more point
  (`Hx+1:Hx+Nx+1`); an extended range into the halo (`Hx:i2`); a halo
  shell (`1:Hx, …`); a 2-D surface range `(i, j)` with
  `body!(i, j, args...)`; or a 1-D range of ids (surface elements,
  particles) with `body!(n, args...)`. The body gets the storage index
  and does its own offset arithmetic (`xs[i - Hx]`).
* **The body writes only point `(i, j, k)`** of its outputs. It may read
  its inputs anywhere, but never something another point writes in the
  same launch. An in-place stencil therefore needs a scratch array (see
  scratch rules).
* **Everything the body reads is an argument.** Do not close over
  globals. Closures over local values are fine (`(y, x) -> y + a*x`).
* **The body carries its own `@inbounds`.** `launch!` does not check
  bounds. The public wrappers (`set!`, `interior_map!`, `range_map!` and
  the reductions) check the whole range once per call.
* **The result must not depend on the order points run in.** Accumulate
  only through the reductions below, never through a shared variable.

**Two performance traps** that the 1-thread gate caught in KA-a. Both
gave correct results and were only slow.

* *Specialisation.* Julia does not specialise a pass-through `args...`,
  or a `::Function` argument that is only handed on. A wrapper written
  as `foo(f::Function, args...) = launch!(b, body!, r, f, args...)` is
  compiled for `Function` / `Any`, and the loop then dispatches at every
  point: 375× the loop for an axpy, 17× for `set!(f, g)`. Write wrappers
  as `foo(f::F, args::Vararg{Any,M}) where {F, M}`. `launch!` itself
  forwards its arguments as one type parameter.
* *Aliasing.* If the same array reaches the body through two arguments
  (`interior_map!(f, y, y, x)`: `out` and an input), the compiler cannot
  prove the loop free of dependences and keeps it scalar (1.8×).
  `range_map!` therefore reads an input that *is* `out` through `out`.
  Hand-written bodies should take an in-place array once.

**How a launch runs.** The loops run on the calling task, in the
historical order `for k, for j, for i`, when the backend is
`serial = true`, when Julia has one thread, or when the launch covers
fewer than `threshold` points (default 32 768, a 32³ block: a task start
costs a few µs per thread). This is also why a `-t 1` run executes
exactly the loops it executed before the port. Otherwise the launch is a
KA kernel whose **work item is one i-row**, in **workgroups of four rows**
(the `(n_i, min(n_j, 4), 1)` block of the plan). KA's CPU backend splits
the workgroups into `nthreads()` contiguous chunks, one task each.

Why rows and not points: on KA's CPU backend a point-per-work-item
kernel computes a Cartesian index per point and does not vectorise. For
an axpy at 64×64×32 at one thread it measured **5.9× slower** than the
plain loop (119 vs 20 µs), and 2.0× at 256³. The row kernel keeps the
inner `i` loop a plain loop the compiler vectorises. A GPU backend will
launch the same body one point per work item; the body is the portable
unit, not the iteration strategy.

## The reduction contract

Sums over a region use one **canonical order** that does not depend on
the thread count:

1. **Within a plane** (fixed `k`), rows are taken in ascending `j`. Each
   row is reduced in **eight lanes**. Point `i` of the first `8⌊n/8⌋`
   goes to lane `(i − i₁) mod 8`, and the rest of the row goes to lane 1
   in ascending `i`. The lanes are combined as
   `((l₁+l₂)+(l₃+l₄)) + ((l₅+l₆)+(l₇+l₈))`.
2. **Planes** are combined in ascending `k` by
   [`ordered_sum`](@ref Aeolus.Backends.ordered_sum), a strict left fold.

Each plane is reduced by one work item, so the plane partials, and
everything built from them, are identical serially and at any thread
count. The eight independent lanes let the compiler use SIMD and hide
the add latency without reassociating anything. `test/backends.jl`
checks the implementation bit for bit against an independent transcript
of this rule.

| primitive | result |
| :-------- | :----- |
| `plane_partials!(p, [b,] f, ranges, arrays...)` | `p[c]` = canonical plane sum of `f(a[i,j,k], …)` at `k = ranges[3][c]` |
| `interior_sum([b,] A, ranges)`, `interior_sum(field)` | `ordered_sum` of the plane partials |
| `interior_dot([b,] A, B, ranges)`, `interior_dot(f, g)` | the same with `f = *` |
| `plane_mean!(profile, [b,] A, ranges)`, `plane_mean!(profile, field)` | plane partial / (n_i n_j): the horizontal-mean profile |
| `column_partials([b,] A, ranges)` + `ordered_sum` | one sum over ascending `k` per column `(i, j)`, then the columns in column-major order: a *different* canonical total |
| `interior_max([b,] A, ranges)` | NaN-propagating maximum (any NaN gives NaN); `-Inf` for an empty region |
| `interior_all_finite([b,] A, ranges)` | no NaN, no ±Inf |
| `plane_reduce!(p, [b,] body, ranges, args...)` | `p[c]` = canonical plane sum of the *indexed* `body(i, j, k, args...)`; a tuple-valued body sums each component into `p::NTuple` of vectors (KA-b) |
| `range_max([b,] body, ranges, args...)` | NaN-propagating maximum of an indexed body (KA-b) |

`plane_reduce!` is the general form. Its body follows the launch
contract: it may read a stencil (a strain rate, a centred `w`), may
write its own point (an update fused with the sum of what it changed: a
chemistry step returning its production), and takes per-level
coefficients as arguments indexed by `k`. A masked mean is a tuple body
returning `(value, 1.0)` or `(0.0, 0.0)`: adding an exact `0.0` leaves
a lane unchanged, so the masked sum is the canonical sum of the masked
values. Ranges may be storage indices of halo-padded arrays or `axes`
of interior-sized views.

**Implementation.** For `Float64` sums the eight lanes are the eight
elements of one `<8 x double>` vector (an `llvmcall` `fadd`): point
`i₁ + 8b + r` goes to element `r`, so a plane is one vector add per eight
points. The vector add is eight independent IEEE additions, so the bits
are those of the scalar lanes (`test/backends.jl` checks the transcript);
the compiler does not vectorise eight scalar accumulators by itself, and
the vector form measured 2.4× faster (`interior_sum` 23.3 → 9.6 µs at
64×64×32, one thread). Other element types use the scalar lanes.
Maxima need no canonical order (`max` is exact and commutative), so
`interior_max` and `range_max` scan a plane in a plain loop: their bits
equal those of any serial maximum.

**Surface (2-D) sums** over the `(i, j)` columns stay serial: a loop
over the columns folds their contributions in column-major order
(`for j, for i`), or Base `sum` reduces a finished 2-D array. Both run
on one task, so they are thread-invariant; a threaded kernel that
produces the per-column values keeps the serial sum after it (a few µs
at 64×64).

**Per-level loops that already reduce one plane at a time** in a plain
`for j, for i` loop (the dynamic-Smagorinsky plane averages, the
#288 damping `levels`, the large-scale forcing's nudging and subsidence
plane means) are thread-invariant as they stand, because a plane is one
work item, and are left unchanged and documented as canonical.

The canonical order is not Base's. `sum(view(A, I...))` over a
non-contiguous view is a plain sequential sum, so it is not just
different, it is less accurate. For `Σ θ²` of a 37×29×23 linear profile
the eight-lane order is within 3.1e-16 of the exact (BigFloat) value,
and Base's `sum(abs2, view)` within 2.9e-13. Moving a call site to the
canonical order is therefore a named one-time bit change, and the PR
quotes its size.

## Scratch rules

* A kernel that reads a neighbour of what it writes uses a separate
  output array (a tendency, a `lap_*` field, a workspace). Allocate it
  once with the simulation, not per launch.
* Per-work-item temporaries are locals (registers / stack), never a
  shared buffer indexed by thread id: KA gives no stable thread id, and
  the work-item → task map changes with the thread count.
* Partials arrays for reductions are sized by the outer range (`n_k`
  planes, `n_i × n_j` columns), not by the thread count.

## FFTW and BLAS

* **FFTW** plans come from the #245 policy (`set_fftw_policy!`): MEASURE
  by default, PATIENT opt-in, ESTIMATE in the tests (set by
  `test/runtests.jl` and inherited by every subprocess, including the
  harness). ESTIMATE plans are deterministic. `init_threads!(b;
  fftw_threads = n)` sets the FFTW thread count *through that policy*, so
  `fftw_policy()` and the checkpoint record report it. The default
  (`nothing`) keeps `AEOLUS_FFTW_THREADS`, which defaults to 1. Thread
  invariance of threaded FFTW plans is the FFT port's (KA-h) to
  establish.
* **BLAS** is pinned to one thread whenever Julia runs threaded (the
  accepted default on #294). BLAS threads would oversubscribe the cores
  next to the kernel tasks, and a threaded BLAS may split a reduction
  differently at different thread counts. Under `-t 1` the BLAS setting
  is left alone.

## The thread-invariance harness

`test/thread_invariance.jl` has two levels:

1. **In process (always):** the reductions and a 3-step 32³ Boussinesq
   run with `CPUBackend(serial = true)` against the threaded backend.
   This is meaningful when the suite runs threaded
   (`test/run.jl --threads 4`).
2. **Subprocesses (opt-in, `AEOLUS_THREAD_TESTS=1`):**
   `test/support/thread_invariance_worker.jl` runs at `julia -t 1/4/24`
   (`AEOLUS_THREAD_COUNTS`). It hashes the raw bits of: the primitives on
   seeded fields (NaN / Inf included); every field of a 64³ Boussinesq
   DNS after 2 steps; and every run of the Berlin smoke matrix (12 steps)
   plus one step of the 64×64×32 flagship. For each Berlin run it hashes
   every simulation field, tracer and surface-bus array, the surface /
   water / air-heat ledgers, the dt sequence and the pressure-solver
   iteration counts. Every hash must equal the 1-thread hash, and the
   differing keys are listed otherwise.

```bash
AEOLUS_THREAD_TESTS=1 julia --project=. test/run.jl --only thread_invariance.jl
```

Later port PRs extend the worker with their own sections, not the
driver.

## Measured at KA-a

Machine: AMD Ryzen 9 7900 (12 cores / 24 threads, 2 × 32 MB L3),
Julia 1.13.1, KernelAbstractions 0.9.43. Other agents shared the machine,
so these are minima over repeats.

Times in µs, the minimum of two runs at load average 6-9. "1 t / old"
is the gate (≤ 1.10). Speed-up = old loop at 1 thread / new code at N
threads.

| kernel | grid | old loop, 1 t | new, 1 t | 1 t / old | 4 t | 12 t | 24 t |
| :----- | :--- | ------------: | -------: | --------: | --: | ---: | ---: |
| `interior_map!` axpy (in place) | 64×64×32 | 22.1 | 22.1 | 1.00 | 1.36× | 1.56× | 1.44× |
| `interior_map!` axpy (in place) | 256³ | 11 033 | 11 296 | 1.02 | 1.11× | 1.30× | 1.32× |
| `set!(f, value)` | 64×64×32 | 11.0 | 11.1 | 1.01 | 1.28× | 0.94× | 0.80× |
| `set!(f, value)` | 256³ | 6 334 | 5 814 | 0.92 | 1.16× | 1.29× | 1.22× |
| `set!(f, (x,y,z) -> sin(3x)·exp(−y) + z²/7)` | 64×64×32 | 740 | 741 | 1.00 | 2.12× | 3.15× | 3.48× |
| `set!(f, (x,y,z) -> …)` | 256³ | 97 459 | 96 542 | 0.99 | 3.22× | 6.18× | 6.97× |

At one thread every launch takes the serial path, so the gate measures
the loops themselves, and the results are bit-identical to the old loops
(`test/backends.jl`). The KA row kernel forced at one thread costs 1.33×
the loop at 64×64×32 (29.3 vs 22.1 µs) and 0.98-1.00× at 256³, which is
why `-t 1` does not use it.

Reductions: `interior_sum` takes 24.7 µs at 64×64×32 and 4.3 ms at 256³
on one thread, against 113 µs and 15.0 ms for Base's `sum(view(A, I...))`.
On 4/12/24 threads it takes 8.7/14.1/16.2 µs and 2.5/2.2/2.4 ms.

The axpy and `set!(f, value)` are memory-bound. `set!` with a
transcendental function is compute-bound, and it scales: 7× at 24
threads on 256³.

A 256³ axpy reads two arrays and writes one, 144 MB each with halos. At
one thread that is 3 × 144 MB in 11.0 ms, or 39 GB/s, already most of
what the dual-channel DDR5 delivers. Threads add only 1.3× there (8.3 ms
at 24 threads, 52 GB/s). The plan expected about 2-4×, which assumed
more DRAM bandwidth per core than this desktop has.

At 64×64×32 each array is 1.5 MB, so the working set fits in L3, and
threads help until task start-up dominates. A launch therefore gets at
least 16 384 points per task. Without that cap the 64×64×32 axpy took
67.6 µs at 24 threads, 2.7× the serial loop. A pure store
(`set!(f, value)`, 11 µs) is still slower threaded at 12 and 24 threads.

## KA-b: the call sites (#296)

KA-b moved every order-sensitive volume or plane reduction on the step
path, and the budget diagnostics, onto the primitives (still serial at
one thread; threaded through `launch!`'s rules at more). It fixes the
canonical order once, so later threaded kernels match it bit for bit.

| where | what | primitive |
| :---- | :--- | :-------- |
| `AdvectionSchemes` | `_dot_interior` (KE and scalar-variance budgets) | `interior_dot` |
| `FiniteVolume` | `_sgs_stress_dissipation` | `plane_reduce!` (strain body) |
| `Simulations` | `kinetic_energy` (regular, stretched), `horizontal_mean` / `_variance`, `mean_θ_profile`, `resolved_flux_wθ` | `plane_partials!`, `plane_reduce!` |
| `Simulations`, `Anelastic` | `max_divergence` | `range_max` (exact) |
| `Anelastic` | `kinetic_energy` (density-weighted), `unweighted_kinetic_energy` | `plane_reduce!`, `plane_partials!` |
| `TimeIntegration` | `_max_advective_rate`, `_max_interior`, `_max_speed` | `range_max`, `interior_max` (exact) |
| `TimeIntegration` | `_all_finite` | `interior_all_finite` (exact) |
| `TimeIntegration` | `_brunt_vaisala` plane means | `plane_partials!` |
| `Nesting` | `NestedRun` speed cap, bound-field finiteness, `_covered_mean` | `range_max`, `interior_all_finite`, `plane_partials!` |
| `abl_diagnostics` | `_plane_means`, `resolved_tke_profile` | `plane_partials!`, `plane_reduce!` |
| `Chemistry` | `apply_photochemistry!` P / L / H, `apply_wet_deposition!` | `plane_reduce!` (fused update) |
| Berlin `core/ledger.jl` | `ledger_water`, `ledger_water_split` (`level_sums`), `add_surface!` | `plane_partials!`; per-class ordered fold |
| Berlin `core/air_heat.jl` | `air_heat_levels!` (fluid and solid sums) | `plane_reduce!` (tuple) |
| Berlin `parent/case.jl` | `_level_sums!` (the parent's budget levels) | `plane_partials!` |
| `scalar_advection` | `sgs_diffuse!` Σ\|q\|, Σ tendency, its water split | `plane_reduce!` (tuple) |
| `aq_chem` | species totals; level air temperature | `interior_sum`; `plane_reduce!` (masked) |
| `moist_buoyancy` | the level means of ``(θ_ρ − θ)``, fused with the cell values | `plane_reduce!` (masked, fused) |
| `column_radiation` | fluid-mean profiles | `plane_reduce!` (masked) |
| `geostrophic` | interior means of `u`, `v`, `tu`, `tv` | `interior_sum` |

Left as they are, with the reason:

* **2-D surface sums** (`surface_exchange` `evap` / `nsolid`, `aq_chem`
  emitted and dry deposition, `column_radiation`'s emissivity-weighted
  `T_rad`, the `buildings_citymean` and `fog_dew` ledger terms): serial
  2-D sums (above).
* **Maxima and minima** of `output.jl` and the reports: exact and
  order-independent.
* **Plane-at-a-time loops** (dynamic Smagorinsky, #288 damping levels,
  `LargeScaleForcing` nudging and subsidence): canonical as they stand.
* **CG / MG dot products**: KA-g (#300).
* **Reports, smoke checks and one-off diagnostics** (`bubble_top_height`,
  `θ_variance`, the SGS reports, `laminar_reference_νt`): not on the step
  path; serial and thread-invariant.
* `examples/urban/couplings/nesting.jl`'s `_covered_profile!`: left for
  the in-flight Berlin nesting work (#330 / #267), to avoid a conflict.

**One-time bit changes against master (70d9f83).** Every ported sum
changes its last bits once; maxima, minima and finiteness checks do not
(`test/era2_timeint.jl` compares them with transcripts of the old loops,
NaN and ±Inf included), and neither does any pointwise update (the
chemistry cells, the moist-buoyancy cells: identical). Measured on a
seeded 32³ LES after three steps (before → after):

| quantity | master | KA-b |
| :------- | -----: | ---: |
| KE budget `residual` | −3.6e-15 | 0 |
| KE budget `advection` (≈ 0 for an energy-conserving scheme) | 7.4e-18 | −1.4e-18 |
| KE budget `pressure` (≈ 0) | 8.5e-18 | 2.2e-19 |
| scalar-variance budget `residual` | 1.1e-11 | 1.4e-12 |
| scalar-variance budget `advection` (≈ 0) | 7.5e-13 | 2.7e-13 |
| `kinetic_energy` | 0.079382106926299545 | 0.079382106926300086 |
| `sgs_stress_dissipation` | 0.74437976056020061 | 0.74437976056019994 |
| `_brunt_vaisala` | 1.1608508246122187 | 1.1608508246121418 |
| chemistry `P_Ox` (24×20×12) | 87.93448478692747 | 87.934484786927598 |
| `_max_advective_rate`, `_max_speed`, `max_divergence` | identical | identical |

The Berlin smoke matrix (49 runs, 12 steps) keeps every physical term to
the printed digits; the ledger closures move at round-off (largest over
the runs, master → KA-b): surface closure 1.0e-12 → 2.6e-13 W m⁻², water
residual 7.5e-16 → 6.5e-16 kg m⁻², air-heat closure 1.0e-12 → 1.2e-12
W m⁻². The tolerances (1e-6 W m⁻², 1e-12 kg m⁻², 1e-10 W m⁻²) are
unchanged.

**Measured at KA-b.** Same machine; minima of three runs at load 2-4, on
the 64×64×64 flagship state after two steps. "1 t / old" is the gate
(≤ 1.10); speed-up = old loop at 1 thread / new code at N threads.

| reduction | old, 1 t (µs) | new, 1 t (µs) | 1 t / old | 4 t | 12 t | 24 t |
| :-------- | ------------: | ------------: | --------: | --: | ---: | ---: |
| `ledger_water` (qv) | 145.3 | 19.1 | 0.13 | 12.6× | 11.2× | 6.8× |
| `ledger_water_split` (qv) | 145.4 | 19.2 | 0.13 | 12.9× | 12.4× | 5.9× |
| `air_heat_levels!` | 35.4 | 34.4 | 0.97 | 2.0× | 2.1× | 1.4× |
| `moist_buoyancy_perturbation!` | 1496 | 1355 | 0.91 | 3.0× | 3.5× | 3.5× |
| `fluid_mean_profile` (θ) | 174.5 | 186.6 | 1.07 | 2.9× | 3.7× | 3.6× |
| aq_chem `_aq_temperature!` | 143.4 | 130.4 | 0.91 | 3.4× | 4.3× | 3.8× |
| `kinetic_energy` | 147.0 | 57.8 | 0.39 | 7.3× | 8.8× | 5.0× |
| `horizontal_mean` (θ) | 25.9 | 19.3 | 0.75 | 2.7× | 2.2× | 1.1× |
| `_max_advective_rate` | 148.6 | 155.7 | 1.05 | 3.3× | 4.4× | 4.1× |
| `_brunt_vaisala` | 142.8 | 19.2 | 0.13 | 14.9× | 11.7× | 6.0× |
| `_max_speed` | 2440 | 253.9 | 0.10 | 29.7× | 42.1× | 27.3× |
| `_all_finite` | 990.2 | 574.5 | 0.58 | 5.7× | 7.9× | 5.4× |
| `sgs_stress_dissipation` | 1030 | 905.0 | 0.88 | 4.2× | 6.3× | 7.2× |
| `apply_photochemistry!` | 14456 | 11666 | 0.81 | 4.6× | 12.3× | 11.7× |
| `apply_wet_deposition!` (no rain) | 58.4 | 49.2 | 0.84 | 1.2× | 1.2× | 0.6× |
| `apply_wet_deposition!` (10 % rain) | 329.5 | 449.9 | 1.37 | 5.3× | 8.3× | 6.1× |

`apply_wet_deposition!` with rain in 10 % of the cells misses the gate
(1.37×, 450 vs 330 µs): the old loop skipped a dry cell entirely, the
lane order adds its exact `0.0`. Without rain it returns early and is
faster than before. It runs once per step (0.2 % of a flagship step).

The flagship (`CaseConfig()`, 64×64×64), minimum of 3 × 10 steps at
load 2: **238.7 ms/step on master, 213.5 ms/step with KA-b at one
thread (0.89×)**, mostly from the ledgers' water integrals (145 → 19 µs
per call). At 4/12/24 threads it takes 208.8/213.6/216.0 ms/step: the
reductions thread, but the step is still dominated by the loops later
port PRs move.

The thread-invariance harness (`AEOLUS_THREAD_TESTS=1`, sections
`primitives,dns,reductions,smoke,flagship`) gave 2665 hashes at
`-t 1`, `-t 4` and `-t 24`, all identical.

## Dependency tree

KernelAbstractions 0.9.43 → Adapt, Atomix 1.6.0, UnsafeAtomics 0.4.0
(→ LLVM.jl 10.1.0 → CEnum, the binary `LLVMExtra_jll` 0.0.51), MacroTools,
PrecompileTools (→ Preferences), Requires, StaticArrays
(→ StaticArraysCore), and the stdlibs InteractiveUtils and UUIDs.

Until 2026-10-06 `Project.toml` held `UnsafeAtomics` at 0.3 through a
weak-dependency compat bound, the last series where LLVM.jl is optional,
to keep the tree free of JLLs. The maintainer lifted it with the
dependency-policy change ("Allow LLVMExtra_jll"). Measured cost, `using
Aeolus` in a fresh process, minimum of three: 215 ms with the pin, 252 ms
without (whole process 341 ms → 380 ms), about +37 ms.
