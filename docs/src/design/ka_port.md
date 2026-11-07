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
`init_threads!` first thing in `build_case`; every other run gets it
from `Config.build` (#524: until then a model domain's FFTs ran on one
thread).

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

**The arguments reach the loop as arguments** (KA-d, #299). A large body
with branches (WENO-5, the tensor-diffusion face fluxes) inlined into a
loop that reads its arguments out of the `args` tuple ran 3-5× slower
than the same body in a plain loop (WENO-5 at 64×64×32, one thread:
6.3 ms against 2.1 ms): LLVM did not hoist the array loads behind the
branches. Arrays that arrive as function arguments are known
dereferenceable, and then it does. So the serial path calls
`_nest!(body!, ir, jr, kr, a1, …, aN)`, the whole `k/j/i` nest in one
function with the arguments separate, and the threaded row kernel calls
`_row!(body!, ir, j, k, a1, …, aN)` per work item. Each argument has its
own type parameter (a function among them is still specialised; methods
are generated for up to 24 arguments, plain loops beyond). Same
operations in the same order, so no bit changes. Small kernels are
unchanged at one thread (axpy 22-24 µs, `set!` 11 µs, the Laplacian
46-47 µs at 64×64×32); WENO-5 went from 6.3 to 2.0 ms at one thread and
from 0.83 to 0.39 ms at 12. A call per *row* on the serial path instead
of one per nest made the Laplacian 1.3× the old loop, hence `_nest!`.
A 2-D launch on the serial path calls `_nest2!(body!, ir, jr, a1, …,
aN)` the same way (KA-e, #302): `boundary_scalar_flux`'s face bodies (up
to 19 arguments) ran 2-2.5× the old loop when they read the tuple.

**Column kernels: `launch_weighted!`** (KA-h, #298). A body that loops
over a whole column itself — a tridiagonal solve in z, serial within the
work item — does `Nz` points of work per index, so a launch over
`(i, j)` or over rows `j` would look small to the threshold and to the
points-per-task rule (a 64-row launch is 64 "points").
[`launch_weighted!`](@ref Aeolus.Backends.launch_weighted!)`(b, weight,
body!, ranges, args...)` is `launch!` with those two rules counting
`weight × length(ranges)` points. Same contract, same serial order. It is
not a per-kernel cost hint: the weight is the number of points an index
covers.

A column per work item can be slow on the CPU: the column walks z with
a stride of `Nx·Ny·8` bytes, a power of two on the usual grids, so all
its levels fall in the same L1 set. `StretchedPoissonSolver`'s Thomas
solve with one column per item took 4.3 ms at 64×64×53 against 0.94 ms
for the whole old solve. One ky row per item (kx innermost, vectorised)
fixed most of it, but the Thomas part alone still cost 1.2× the old loop
at 64×64×53 and 1.6× at 192×192×53 on one thread: `Nx` contiguous values
per level are too short a stream for the prefetcher. The work item is
therefore a contiguous slab of ky rows, walked in the old loop order
(level by level, kx innermost), with one slab per thread and a single
slab, the old loop exactly, when the launch runs serially. The slab
count may follow the thread count because every column is still solved
by one work item with the same operations, so the bits do not depend on
it. A GPU backend will launch one column per item.

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
| `range_max([b,] body, ranges, args...)` | NaN-propagating maximum of an indexed body, which must only read (KA-b); a tuple-valued body gives the component-wise maxima, several in one sweep (KA-j) |
| `plane_map!(p, [b,] plane, ranges, args...)` | `p[c]` = `plane(k, ranges[1], ranges[2], args...)`: a whole-plane function in its own fixed loop order, one work item per plane (KA-b) |

`plane_reduce!` is the general form. Its body follows the launch
contract: it may read a stencil (a strain rate, a centred `w`), may
write its own point (an update fused with the sum of what it changed: a
chemistry step returning its production), and takes per-level
coefficients as arguments indexed by `k`. A masked mean is a tuple body
returning `(value, 1.0)` or `(0.0, 0.0)`: adding an exact `0.0` leaves
a lane unchanged, so the masked sum is the canonical sum of the masked
values. Ranges may be storage indices of halo-padded arrays or `axes`
of interior-sized views. A `range_max` body only reads: its result type
is inferred (`Base.promote_op`), never found by evaluating the body an
extra time.

`plane_map!` is the escape hatch for a reduction where the eight lanes
cost more than they gain: the plane function runs its own loop (one
running sum in cell order, `i` fastest) and each plane is still one work
item, so the partials are thread-invariant. Its only user is
`apply_wet_deposition!` (below): `exp` is called in the raining cells
only, and keeping eight lanes live across that call made the fused
update 1.3× slower than the old loop at one thread.

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
* Random numbers: never one generator shared by the work items of a
  launch (the draws would depend on which task got there first). Each
  item makes its own stream from what identifies it,
  [`counter_rng`](@ref Aeolus.Backends.counter_rng)`(key, stream, id)`
  for a few draws or [`counter_xoshiro`](@ref Aeolus.Backends.counter_xoshiro)
  for hundreds, keyed by a seed or one draw from the caller's `rng`
  (KA-n). Tallies go into per-item buffers merged in item order.
* A column work item ([`foreach_column!`](@ref Aeolus.Backends.foreach_column!),
  KA-m) owns its scratch: a workspace that belongs to the column's own
  object (a soil column's, a facet's conduction column), or vectors the
  column allocates itself (a raining Kessler column's `col` / `Nr`).
  Never one buffer captured by a closure and shared by every column, and
  never a global scratch vector (the Mie log-derivative buffer was one,
  and raced, until KA-m made it task-local).
* Lazily filled global caches are filled before any launch can reach
  them, or guarded by a lock (KA-m): the Gauss-Legendre nodes are a table
  computed when `RadiationCore` loads, the aerosol optics memo and the
  SNICAR table are locked.

## FFTW and BLAS

* **FFTW** plans come from the #245 policy (`set_fftw_policy!`): MEASURE
  by default, PATIENT opt-in, ESTIMATE in the tests (set by
  `test/runtests.jl` and inherited by every subprocess, including the
  harness). ESTIMATE plans are deterministic. Since KA-h (#298)
  `init_threads!` sets the FFTW thread count to `Threads.nthreads()`,
  through that policy, so `fftw_policy()` and the checkpoint record
  report it. A count chosen explicitly wins: `init_threads!(b;
  fftw_threads = n)`, `AEOLUS_FFTW_THREADS`, `set_fftw_policy!(threads =
  n)` or a namelist's `[fftw] threads`. Call `init_threads!` before the
  simulation is built: a plan keeps the thread count it was made with.
* **Threaded FFTW is thread-invariant under ESTIMATE.** FFTW's threaded
  plans hand whole 1-D transforms (and contiguous slabs of them) to the
  threads and leave each transform's arithmetic unchanged. Measured at
  KA-h: every pressure solver (FFT, CosineFFT with each of the 8 axis-BC
  combinations, Stretched with each of the 4 lateral pairs) gives
  bit-identical solves with plans made at 1, 2, 4, 12 and 24 threads, at
  64×64×32, 128²×64, 192²×53, 256²×128, 96×80×40 and 37×29×23. Under
  MEASURE the planner times candidates per thread count, and the same
  check found differing bits in 3 of the 13 solvers at 64×64×32 (for 2,
  12 and 24 threads). MEASURE runs are not bitwise reproducible anyway
  (#245), and imported wisdom made at the same thread count restores
  reproducibility. The pencil-parallel fallback the plan held in reserve
  (serial 1-D plans launched over pencils) was not needed.
* **Transforms run from the host.** Plans are executed through
  [`apply_transform!`](@ref Aeolus.Backends.apply_transform!)`(backend,
  out, plan, in)` between kernel launches, never inside a kernel, so a GPU
  backend can put cuFFT behind the same call (Phase 81).
* **BLAS** is pinned to one thread whenever Julia runs threaded (the
  accepted default on #294): BLAS threads would oversubscribe the cores
  next to the kernel tasks. The dense solves (the radiosity LU and its
  solves, the direct facet solve), which run while the kernels wait, use
  `Backends.dense_blas_threads()` for their duration (#496; OpenBLAS's
  default, its physical-core count, unless `blas_threads` or
  `OPENBLAS_NUM_THREADS` says otherwise). Under `-t 1` the BLAS setting is
  left alone. (Bit-identity across BLAS thread counts is not a goal since
  epic #493.)

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
driver. KA-m added `columns`: the column kernels on seeded inputs, every
launch threaded (`threshold = 0`), BLAS on one thread in that section
(see "Measured at KA-m").

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
| `AdvectionSchemes` | `enstrophy` (edge vorticities) | `plane_reduce!` (stencil body) |
| `FiniteVolume` | `_sgs_stress_dissipation` | `plane_reduce!` (strain body) |
| `Simulations` | `kinetic_energy` (regular, stretched), `horizontal_mean` / `_variance`, `mean_θ_profile`, `resolved_flux_wθ` | `plane_partials!`, `plane_reduce!` |
| `Simulations`, `Anelastic` | `max_divergence` | `range_max` (exact) |
| `Anelastic` | `kinetic_energy` (density-weighted), `unweighted_kinetic_energy` | `plane_reduce!`, `plane_partials!` |
| `TimeIntegration` | `_max_advective_rate`, `_max_interior`, `_max_speed` | `range_max`, `interior_max` (exact) |
| `TimeIntegration` | `_all_finite` | `interior_all_finite` (exact) |
| `TimeIntegration` | `_brunt_vaisala` plane means | `plane_partials!` |
| `Nesting` | `NestedRun` speed cap, bound-field finiteness, `_covered_mean` | `range_max`, `interior_all_finite`, `plane_partials!` |
| `abl_diagnostics` | `_plane_means`, `resolved_tke_profile` | `plane_partials!`, `plane_reduce!` |
| `Chemistry` | `apply_photochemistry!` P / L / H | `plane_reduce!` (fused update) |
| `Chemistry` | `apply_wet_deposition!` | `plane_map!` (fused update, cell order per plane) |
| Berlin `src/budgets/budgets.jl` | `ledger_water`, `ledger_water_split` (`level_sums`), `add_surface!` | `plane_partials!`; per-class ordered fold |
| Berlin `src/budgets/meter.jl` | `air_heat_levels!` (fluid and solid sums) | `plane_reduce!` (tuple) |
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
* **CG / MG dot products**: ported by KA-g (#300), below.
* **Reports, smoke checks and one-off diagnostics** (`bubble_top_height`,
  `θ_variance`, the SGS reports, `laminar_reference_νt`): not on the step
  path; serial and thread-invariant.
* `src/urban/couplings/nesting.jl`'s `_covered_profile!`: left for
  the in-flight Berlin nesting work (#330 / #267), to avoid a conflict.

**One-time bit changes against master (70d9f83).** Every ported sum
changes its last bits once; maxima, minima and finiteness checks do not
(`test/timeint.jl` compares them with transcripts of the old loops,
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
| `apply_wet_deposition!` total (64³, 10 % rain; cells identical) | 6.518950163789894e-9 | 6.518950163789874e-9 |
| `enstrophy` (seeded 64³ field) | 24548.03365630357 | 24548.033656303025 |
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
| `apply_wet_deposition!` (no rain) | 109.4 | 54.5 | 0.50 | 1.2× | 1.3× | 1.2× |
| `apply_wet_deposition!` (10 % rain) | 342.5 | 364.5 | 1.06 | 2.6× | 2.6× | 1.8× |
| `apply_wet_deposition!` (all rain) | 840.6 | 860.0 | 1.02 | 1.7× | 3.2× | 2.5× |
| `apply_wet_deposition!`, `Field` views (10 % rain) | 544.8 | 402.9 | 0.74 | | | |
| `enstrophy` | 397.6 | 132.9 | 0.33 | 7.8× | 4.7× | 6.7× |

`apply_wet_deposition!` first went through `plane_reduce!` and missed
the gate (1.31–1.39× at 10 % random rain, 64³). The dry cells were not
the cause: skipping blocks of eight dry cells, or adding only the
nonzero terms to the lanes, both stayed at 1.2× or more, and the lane
version was 1.3× even with rain everywhere. What cost the time was
keeping eight lanes live across the `exp` call of each raining cell,
together with 3-index stores that make the compiler reload the array's
size at every cell (a plain `k, j, i` loop with `c[i, j, k]` is 1.5×
the linear loop). It now runs through `plane_map!`: one running sum
per plane in cell order over linear indices, planes in parallel, then
`ordered_sum` over `k` — thread-invariant, the cells bit-identical to
master, the total a named one-time change. Plain arrays: 1.00–1.07×
the old loop at 2, 10, 50 and 100 % rain (table: minima of 200 runs,
three repeats, the reviewer's protocol, load 13 from other jobs); the
`Field` views the Berlin `aq_chem` coupling passes are 0.54–0.82×,
because the old loop indexed the views Cartesian. The no-rain pre-scan
is kept for plain arrays only (on a view it cost more than the sweep).
The thread columns of these two rows were measured at load 13–15.

The flagship (`CaseConfig()`, 64×64×64), minimum of 3 × 10 steps at
load 2: **238.7 ms/step on master, 213.5 ms/step with KA-b at one
thread (0.89×)**, mostly from the ledgers' water integrals (145 → 19 µs
per call). At 4/12/24 threads it takes 208.8/213.6/216.0 ms/step: the
reductions thread, but the step is still dominated by the loops later
port PRs move.

The thread-invariance harness (`AEOLUS_THREAD_TESTS=1`, sections
`primitives,dns,reductions,smoke,flagship`) gave 2668 hashes at
`-t 1`, `-t 4` and `-t 24`, all identical.

## KA-f: halo fills, open boundaries and nesting transfer (#297)

**Halo fills.** Every phase of `fill_halos!` — periodic, mirror, the
free-slip and no-slip walls, the free-stream top, the fixed-gradient top,
and the open faces in `:value`, `:odd`, `:mirror` and `:inflow` mode — is
one `launch!` of a point body over the face slab
(`src/boundary_conditions/halo_kernels.jl`). A slab is `layers × the full
extent of the two tangential axes`, so edges and corners are filled too.
The phases run in the old order: x, then y, then z, and within an axis
the faces in their old order. An edge or corner therefore keeps the value
of the last axis that reaches it, and `test/ka_halos.jl` checks this
("corner order").

A phase on the pencil normal to a face reads only that pencil, and the
old loops ran its layers in ascending `h`. Two cases follow:

* A one-sided phase reads the interior side of its face, plus the face
  slot written by an earlier launch, and writes only beyond the face. It
  is always independent.
* A two-sided phase (periodic, or mirror on both faces) can read its own
  writes when the interior is shallower than the halo (`N < H`, e.g. a
  2-D run with `Ny = 1`). Such a launch runs on the calling task in
  `launch!`'s serial order, which keeps ascending `h` per pencil. The
  result is then the old one at every thread count.

Along x the layers are the contiguous direction, so a work item is one
x-pencil whose layer count is a compile-time constant (1–4). The first
version used a 3-D launch with the layer as the innermost loop, and it
made the multigrid halo (`H = 1`) 2.7× the old plane copies.

The multigrid level halo (`Multigrid._fill_halo!`) is the same periodic
and mirror kernel with `H = 1`: one source with `fill_halos!`. The open
faces no longer box their plane when they look it up, so an open fill
allocates 0 B (it was 0.8–3.4 kB per call).

**Radiation outflow.** The phase speeds, the stage mark and the upstream
update of a convective or Orlanski face are launches over the face
points. In the radiation update a work item owns its buffer pencil.

**Nesting.** Owner computes for every write:

* **Prolongation:** each child point interpolates from its parent
  stencil, one launch per box (and per slab of a halo-only shell).
* **Limiters:** per box, each parent cell gathers the extrema of its
  children. This was a scatter into the parent arrays. `max` and `min`
  return one of their operands, so the order of the children does not
  change the bits. Then the factor is computed per parent cell, and the
  scaling and the face clip per child. A plan now carries the inverse of
  its `cell` map, the contiguous children of each parent cell.
* **Restriction:** each covered parent cell or face gathers its
  children. This covers the volume and mass measures, the solid-aware
  case, cells and faces.
* **Per parent face or cell:** the flux mismatch, the face flux
  correction and the refluxing. Refluxing finds the parent cell beyond a
  face with `neighbor_index` instead of `mod1`. The same holds for the
  two-way anterpolation, interface correction, register mismatch and
  bookings.
* **Pointwise:** the sponge and the boundary-data time blend
  (`blend_window!`).

The interface-imbalance L1 sum stays a serial loop. It is a diagnostic,
and moving it to a canonical reduction would change its bits. The
synthetic-inflow RNG is not touched (seeded per face, serial).

**Bit for bit.** Every BC type, combination and location matches the
pre-port fill on random fields with random halos, at 12×10×8,
at 9×7×6 with halo (2, 4, 3) and at 5×1×2 with `H = 3`. This holds on the
serial, threshold-0 threaded and default backends. The reference is a
transcript of the old plane copies (`test/support/legacy_halos.jl`).
Likewise 124 transfer-operator calls match the pre-port operators
(`test/support/legacy_nesting.jl`). The covered cases are orders 1, 3
and 5, the three limiters, halo / interface / whole-child regions,
solid-aware and mass-measure restriction, and the flux mismatch and
correction. Outside the suite, 411 hashes (halos, multigrid, nesting
operators, open-boundary runs with Dirichlet / convective / Orlanski
faces, one-way / two-way / sponge / Orlanski / anelastic nested runs)
agree with master at 70d9f83 at `-t 1` and `-t 24` (threshold 0). The
worker's `nested` section hashes four nested runs across thread counts.

**Measured** (same machine; minimum of two runs at load 3–4; µs; old =
master at 70d9f83 at one thread; speed-up = old 1 t / new N t):

| kernel | old, 1 t | new, 1 t | 1 t / old | 4 t | 12 t | 24 t |
| :----- | -------: | -------: | --------: | --: | ---: | ---: |
| `fill_halos!` periodic θ, 64×64×32 | 18.6 | 9.6 | 0.52 | 1.94× | 1.96× | 1.95× |
| `fill_halos!` walls, w, 64×64×32 | 18.8 | 10.2 | 0.54 | 1.86× | 1.83× | 1.82× |
| `fill_halos!` open box, u, 64×64×32 | 27.5 | 11.0 | 0.40 | 2.56× | 2.54× | 2.53× |
| multigrid `_fill_halo!` periodic, 66×66×34 | 4.1 | 3.3 | 0.81 | 1.25× | 1.23× | 1.25× |
| `fill_halos!` periodic θ, 256³ | 917 | 501 | 0.55 | 6.95× | 3.87× | 2.46× |
| `fill_halos!` walls, w, 256³ | 901 | 536 | 0.59 | 6.15× | 3.22× | 2.96× |
| `fill_halos!` open box, u, 256³ | 890 | 546 | 0.61 | 5.56× | 3.14× | 3.30× |
| multigrid `_fill_halo!` periodic, 258³ | 200 | 135 | 0.67 | 3.51× | 1.98× | 1.81× |
| prolongation, halos, θ, order 3 (64³ child) | 1279 | 1204 | 0.94 | 1.05× | 1.08× | 0.82× |
| … with the `:positive` limiter | 1490 | 1591 | 1.07 | 0.92× | 0.91× | 0.74× |
| prolongation, whole child, θ, order 3 | 5296 | 4989 | 0.94 | 3.86× | 5.71× | 7.12× |
| … with the `:monotone` limiter | 10039 | 10458 | 1.04 | 2.37× | 2.64× | 2.72× |
| restriction θ (32³ parent cells) | 225 | 198 | 0.88 | 2.03× | 1.97× | 1.99× |
| restriction θ, solid-aware | 387 | 319 | 0.82 | 2.22× | 2.01× | 2.01× |
| restriction u | 152 | 127 | 0.84 | 1.18× | 1.18× | 1.19× |
| flux mismatch u | 12.9 | 12.0 | 0.93 | 1.01× | 1.00× | 0.99× |
| face flux correction u | 1.9 | 2.0 | 1.03 | 0.96× | 0.96× | 0.96× |
| refluxing θ | 2.0 | 2.0 | 1.00 | 0.99× | 0.99× | 0.97× |

Radiation outflow at 64×64×32, one thread: the stage mark takes 92 µs
(was 2752; the field array used to be dispatched on at every point), the
phase speeds 413 µs (500) and the update 308 µs (374).

Reading the table:

* **The 1-thread gate holds everywhere** (largest 1.07, the `:positive`
  limiter). At one thread every halo fill is faster than the plane copies
  it replaces, 0.40–0.81×.
* **Small launches stay serial.** At 64×64×32 every halo slab is below
  the 32 768-point threshold, so the fill runs serially at any thread
  count and does not slow down at 24 threads, as the issue requires. The
  speed-ups there are the 1-thread gain. The halo prolongation of a 64³
  child is serial for the same reason: each slab has about 15 000
  points. Its 24-thread figures (0.74–0.82×) are machine noise, because
  a direct per-box measurement gives the same 1.20 ms at 1, 4 and 24
  threads.
* **At 256³ the slabs thread.** A halo fill reaches 5.6–7.0× the old
  loop at 4 threads, and only 2.5–3.9× at 12 and 24 threads. The fill is
  memory-bound; the likely cause is that tasks beyond what the DRAM
  channels can feed only add start-up cost (not profiled further). The issue expected 1.5–3× at 12 threads. Against the new
  1-thread time it is 1.6–2.1× at 12 threads.
* **The whole-child prolongation is compute-bound** (a 27-point stencil
  per point) and scales to 7.1× at 24 threads.

## KA-g: one CG driver (#300)

**One loop.** `Backends.pcg!` is the only preconditioned conjugate
gradient in Aeolus. `CGPoissonSolver`, `MGPreconditionedCG`,
`DensityWeightedPoissonSolver` and `solve_implicit_diffusion!` (the IMEX
steppers' implicit diffusion) call it. Their own loops are deleted:
`_dot!`, `_project_mean_flat!`, Anelastic's `_dot`, `_remove_mean!` and
`_jacobi!`, TimeIntegration's `_apply_implicit!` and `_idot`. A solver
supplies four things:

* the operator: a `PointOperator(f, args...; prepare)` (the stencil at
  one point; `prepare` is the halo fill of the array it reads) or a
  function `A!(Ap, p)` (the multigrid fine-grid operator);
* the preconditioner: `IdentityPreconditioner()`, a pointwise
  `PointPreconditioner` (Jacobi), or a function `M!(z, r)` (V-cycles);
* `project_mean` for a singular operator (all axes periodic or Neumann);
* `constraint`: the pinned planes of a staggered field with walls.

Per iteration the driver makes three sweeps and two canonical
reductions:

1. `Ap = A p`, with `p·Ap` reduced over each plane while it is in cache;
2. `x += αp`, `r -= αAp`, `z = M⁻¹r` (a pointwise `M`), with
   `(r·r, r·z, Σz, Σr)` as one 4-tuple reduction;
3. `p = (z − m) + βp`, where `m = Σz / n` with the mean projected out.

The mean projection costs no extra sweep: `r·z` of the deflated `z` is
`Σ rz − m Σ r`. The density-weighted solver went from nine passes over
the arrays per iteration (five sweeps and four separate reductions) to
three sweeps carrying two reductions. A function preconditioner adds one
reduction after `M!`. Every reduction is a `plane_reduce!`-style plane
sum, so a solve, its iteration count and its residual are the same at
every thread count.

**Unit halos.** `CGPoissonSolver` and `DensityWeightedPoissonSolver`
now keep `b`, `x`, `r`, `z`, `p` and `Ap` as `(N+2)`-arrays with one halo
layer. `fill_unit_halo!` (the KA-f halo kernels, also the multigrid
level halo) fills the periodic wrap or the Neumann mirror before each
operator application. The stencil then has no wrap-around index
arithmetic and vectorises: one point of `_laplacian_at` and
`_neg_wlap_at` gives the values of the old wrap-around loops (a mirrored
neighbour makes the face difference exactly zero). The implicit
diffusion iterates on the field's own storage as before, with
`fill_halos!` as `prepare`. `_diffusion_kernel!` is a `launch!` of
`_diff_at`.

**Pinned rows.** A per-point pinned test in the sweeps stopped the row
loops from vectorising: 4× slower in a first version. The sweeps
therefore compute every point and then zero the pinned plane (`Ap`, `z`
and `p` there are 0, `x` keeps its value, `b` there does not count in
`‖b‖`), as `_apply_constraint!` did.

**Status.** `pcg!` returns `(iterations, residual, bnorm, status,
curvature)`. `status` is `:converged`, `:maxit`, `:breakdown` or
`:zero_rhs`, and each solver logs it as before. One change:
`CGPoissonSolver` now stops at a breakdown and warns. Before, it divided
by a vanishing `p·Ap` and went on. `definite = :negative` is for the
two Poisson solvers that iterate on `∇²` itself (negative
semi-definite). For those only a vanishing curvature is a breakdown.

**Workspace.** `PCGWorkspace` holds `r`, `z`, `p`, `Ap` and the
per-plane partial sums. Every solver preallocates one, so a solve
allocates nothing on one thread. Measured with `@allocated`, the
Poisson solvers allocate 0 B. `solve_implicit_diffusion!` allocates
48 B (352 B with a moving wall), the same as before; the 48 B are its
returned NamedTuple at a dynamic call site. A first version allocated
80 B more with a ν array: the `Union` of the two preconditioners was
boxed. It now has one `pcg!` call per preconditioner. Above the serial threshold a threaded launch allocates
KernelAbstractions' tasks, about 9 kB per iteration at 4 threads.

**One-time change (expectation).** The dot products were
`sum(w .= a .* b)` (pairwise) in CG and MG-CG and serial `@simd` loops
in the other two. They are now canonical plane sums. The iteration
counts did not change: identical in all 28 cases measured (the 14
below at 32×32×16 and 64×64×32; the issue allowed ±1). The solutions
change in the last bits: `sum(|x|)` and `max(|x|)` changed by at most
2.2e-15 and 1.8e-15 relative (deep density-weighted case). In 2 of the
28 cases both are bit-identical. No regression reference of the suite
moved. One assertion was replaced: `multigrid.jl` asserted that
MG-CG's dot equals `sum(x .* y)`. It now asserts that the driver's dot
is `interior_dot` (canonical).

**Measured** (64×64×32; ms per solve, minimum of 5; load 2-4 for 1 and
24 threads and 2-6 for 4 and 12; Julia 1.13.1; old = master b4f6f82 at
one thread; speed-up = old 1 t / new N t, in brackets new 1 t / new N
t):

| solver | iterations old → new | old 1 t | new 1 t | new / old | 4 t | 12 t | 24 t |
| :----- | :------------------- | ------: | ------: | --------: | --: | ---: | ---: |
| CG, periodic xy, Neumann z | 288 → 288 | 123.4 | 51.0 | 0.41 | 5.3× (2.18×) | 3.8× (1.56×) | 3.8× (1.55×) |
| CG, closed box | 508 → 508 | 221.2 | 89.6 | 0.41 | 5.9× (2.40×) | 2.9× (1.17×) | 3.7× (1.50×) |
| CG, triply periodic | 259 → 259 | 108.9 | 46.7 | 0.43 | 6.0× (2.56×) | 3.0× (1.30×) | 3.7× (1.57×) |
| MG-CG, periodic xy, Neumann z | 12 → 12 | 40.3 | 39.1 | 0.97 | 1.0× | 1.0× | 1.0× |
| MG-CG, triply periodic | 12 → 12 | 40.4 | 39.0 | 0.97 | 1.0× | 0.9× | 1.0× |
| MG-CG, stretched (×1.04) | 47 → 47 | 226.7 | 222.4 | 0.98 | 1.0× | 0.9× | 1.0× |
| density-weighted, periodic xy, Neumann z | 330 → 330 | 124.7 | 62.0 | 0.50 | 4.8× (2.41×) | 2.4× (1.18×) | 3.3× (1.62×) |
| density-weighted, triply periodic | 292 → 292 | 93.7 | 54.0 | 0.58 | 4.1× (2.37×) | 2.4× (1.40×) | 2.8× (1.62×) |
| density-weighted, closed box | 569 → 569 | 224.1 | 105.3 | 0.47 | 5.5× (2.61×) | 2.8× (1.32×) | 3.3× (1.56×) |
| density-weighted, 3 km deep | 519 → 519 | 223.9 | 96.1 | 0.43 | 6.0× (2.57×) | 3.2× (1.38×) | 3.6× (1.56×) |
| implicit diffusion θ, scalar ν | 27 → 27 | 11.0 | 5.6 | 0.51 | 4.0× (2.07×) | 2.4× (1.23×) | 2.8× (1.44×) |
| implicit diffusion θ, ν(x, z), walls | 34 → 34 | 46.6 | 28.6 | 0.61 | 5.1× (3.12×) | 4.3× (2.67×) | 3.2× (1.99×) |
| implicit diffusion w, pinned walls | 27 → 27 | 11.0 | 5.6 | 0.51 | 3.9× (2.01×) | 2.1× (1.07×) | 2.9× (1.47×) |
| implicit diffusion u, moving no-slip wall | 28 → 28 | 11.5 | 5.9 | 0.51 | 4.5× (2.28×) | 2.5× (1.28×) | 2.8× (1.41×) |

The Berlin flagship step at 64³ (ms per step, minimum of 8 after 2
warm-up steps, the smoke matrix's `core = anelastic` and `timestepper =
imex` rows without nesting; the `:dynamics` timers in brackets; 419
density-weighted iterations per solve; old = master b4f6f82):

| run | old 1 t | new 1 t | new 4 t | new 12 t | new 24 t | old 24 t |
| :-- | ------: | ------: | ------: | -------: | -------: | -------: |
| anelastic core | 362.0 (276.0) | 291.0 (205.2) | 169.2 (91.0) | 196.8 (120.0) | 243.6 (164.0) | 353.2 (276.8) |
| IMEX ARK2 | 234.2 (151.3) | 216.1 (132.2) | 141.2 (69.7) | 144.2 (70.8) | 164.6 (85.5) | 220.9 (148.2) |

Reading the tables:

* **The 1-thread gate holds everywhere** (largest new / old 0.98,
  MG-CG stretched). CG, density-weighted CG and implicit diffusion are
  0.41-0.61× the old loops on one thread. The gain comes from fewer
  sweeps (the fused reductions, the mean projection folded into `r·z`)
  and from stencils without wrap-around indices.
* **MG-CG does not change.** Its time is the V-cycle (smoothing,
  restriction, prolongation). That cycle is still serial and outside
  this issue.
* **Threads.** The issue expected 2-3× at 4 threads and 3-5× at 12.
  Against the new 1-thread time we get 2.0-3.1× at 4 threads and
  1.1-2.7× at 12. At 12 threads it is worse than at 4, and 24 threads
  is between the two. The likely cause (not profiled) is that 32 planes
  of 64×64 points are too little work per task for the start-up of
  three launches per iteration. Against the old loops at one
  thread this is 3.9-6.0× at 4 threads.
* **Berlin.** At 4 threads the anelastic dynamics take 91 ms instead of
  276 ms (3.0×) and the IMEX dynamics 70 ms instead of 151 ms (2.2×).
  The issue quoted 139.6 ms and 124.7-163.9 ms, measured on another
  setup; on this one master takes 276 ms and 151 ms. The replacement of the
  density-weighted CG by a direct solve is #309.

**Thread invariance.** The worker's `pcg` section hashes the solution
and the iteration count of every solver above at 64×64×32 (the
singular Neumann and periodic cases, the stretched multigrid and the
ν(x, z) / pinned / moving-wall implicit cases): 28 hashes. With
`AEOLUS_THREAD_TESTS=1` at 1, 4 and 24 threads, all 2894 hashes of the
harness (every section, `pcg` included) are identical to 1 thread.
`test/ka_pcg.jl` checks the serial backend against the default one,
solution and iteration count, for every solver.

## Measured at KA-h

Same machine, Julia 1.13.1, FFTW 1.10.0 (FFTW 3.3.10), ESTIMATE plans,
FFTW threads = Julia threads, BLAS pinned to 1. Minimum over repeats at
load average 4-9 (other agents on the machine). Times in ms for one
`solve!`. "old" is master before KA-h at one thread; speed-up = old at
1 thread / new at N threads. CosineFFT is the default channel
`(:periodic, :periodic, :neumann)`; "closed box" is all-Neumann;
Stretched uses the parent's 53 levels (25 m to 500 m, ×1.08 to 100 m,
3 km).

| solver | grid | old 1 t | new 1 t | new / old | 4 t | 12 t | 24 t |
| :----- | :--- | ------: | ------: | --------: | --: | ---: | ---: |
| FFT | 64×64×32 | 0.699 | 0.509 | 0.73 | 3.8× | 2.4× | 2.2× |
| FFT | 128²×64 | 10.50 | 9.74 | 0.93 | 3.9× | 4.3× | 4.0× |
| FFT | 256²×128 | 119.2 | 114.6 | 0.96 | 2.7× | 3.5× | 3.5× |
| CosineFFT | 64×64×32 | 1.950 | 1.671 | 0.86 | 4.1× | 4.8× | 4.7× |
| CosineFFT | 128²×64 | 19.82 | 18.35 | 0.93 | 4.1× | 5.3× | 5.3× |
| CosineFFT | 256²×128 | 246.4 | 238.9 | 0.97 | 3.2× | 5.3× | 5.8× |
| CosineFFT closed box | 64×64×32 | 2.890 | 2.734 | 0.95 | 3.9× | 5.7× | 5.7× |
| CosineFFT closed box | 128²×64 | 32.76 | 31.51 | 0.96 | 3.9× | 7.9× | 7.4× |
| CosineFFT closed box | 256²×128 | 321.7 | 319.9 | 0.99 | 3.6× | 6.8× | 8.6× |
| Stretched, periodic | 64×64×53 | 0.958 | 0.819 | 0.85 | 3.9× | 3.1× | 3.0× |
| Stretched, periodic | 192²×53 (parent) | 34.51 | 33.74 | 0.98 | 3.2× | 5.7× | 6.3× |
| Stretched, Neumann | 64×64×53 | 2.030 | 1.877 | 0.92 | 3.8× | 4.8× | 4.3× |
| Stretched, Neumann | 192²×53 (parent) | 29.61 | 29.14 | 0.98 | 3.2× | 5.5× | 5.9× |

The Thomas solve alone (the old level-by-level loop against the slab
kernel): 0.074 / 0.074 ms at 64×64×53 and 1.099 / 1.127 ms at 192²×53
on one thread (1.00 and 1.03), then 0.029 / 0.028 / 0.033 ms and
0.327 / 0.160 / 0.171 ms at 4/12/24 threads (6.9× at 12 threads at the
parent size).

The pointwise kernels at one thread against the broadcasts they
replaced (64×64×32 / 256²×128, µs): copy-in 18 vs 127 / 4 254 vs 8 333,
the eigenvalue multiply 64 vs 100 / 4 404 vs 6 324, normalise and copy
out 18 vs 84 / 4 346 vs 5 500. These are memory-bound. At 256²×128 they
take 2.7 / 1.8 / 2.8 ms at 4 threads and no less at 12 or 24.

Threaded FFTW alone (the old code with FFTW on N threads) gives
CosineFFT 256²×128 in 90.7 / 60.2 / 56.2 ms at 4/12/24 threads; the
kernels take that to 76.9 / 46.3 / 42.7 ms. At 64×64×32, where the issue
measured 1.2× at 12 threads, the old code with threaded FFTW reaches
2.7× (0.711 ms), the new 4.8× (0.408 ms). Beyond 12 threads the solves
are bandwidth-bound and gain little, and the 64×64×32 FFT and stretched
solves get slower again: task start-up then costs more than the work.

**ESTIMATE against MEASURE** (new code, ms, 1 / 12 / 24 threads):
CosineFFT 256²×128 238.9 / 46.3 / 42.7 vs 236.1 / 44.2 / 39.4; closed
box 128²×64 31.5 / 4.14 / 4.46 vs 13.3 / 2.57 / 2.70; FFT 256²×128
114.6 / 33.7 / 33.6 vs 82.3 / 30.6 / 29.3; Stretched periodic 192²×53
33.7 / 6.09 / 5.49 vs 16.8 / 3.83 / 4.02. So ESTIMATE costs up to 2.4×
on one thread (mostly the r2r DCTs and the stretched solver's
half-complex transforms) and 1.0-1.6× at 12 threads. Production keeps
MEASURE (with wisdom for reproducible restarts); the tests use ESTIMATE.

**Thread invariance.** The harness (`AEOLUS_THREAD_TESTS=1`,
`AEOLUS_THREAD_COUNTS=1,4,12,24`), with FFTW now threaded in every
section: all 2656 hashes (primitives, 64³ DNS, the Berlin smoke matrix,
the flagship step and the new fft section) identical at 4, 12 and 24
threads to 1 thread. Two separate processes at `-t 1` and two at `-t 24`
gave the same fft and dns hashes. At `-t 1` every solve is bit-identical
to master: no one-time change.

## KA-i: the multigrid V-cycle (#303)

**Kernels.** Every loop nest of `MultigridPoissonSolver` and of the
V-cycle preconditioner of `MGPreconditionedCG` is a `launch!` of a
pointwise body, and every sum a canonical plane reduction:

* `_smooth_rbgs!`: per colour, the halo fill and then **one launch over
  that colour's cells only**. The launch covers half of each i-row; the
  index `ih` maps to `i = 2ih + ((j + k + colour) & 1)`, so
  `(i + j + k) & 1 == colour`. It is still red-black Gauss-Seidel. The
  cells of a colour read only the other colour, so the order they run in
  does not matter and the sweep has the bits of the serial loop. For an
  even `nx` every row has `nx/2` cells of a colour and the body has no
  bounds test. The strided loop then vectorises: one colour at
  128×128×64 takes 0.65 ms instead of 1.19 ms with the test, and 1.49 ms
  for the old "skip by parity" loop over all cells. An odd `nx` keeps the test.
* `_compute_residual!` (constant and variable coefficient: one body,
  the operator passed as `_ConstOp` / `_VarOp`), `_restrict!` (each
  coarse cell writes itself), `_prolong_add!` (each fine cell writes
  itself; the eight weight products `wi·wj·wk` are hoisted out of the
  body and added in the old loop order: 1.59 ms instead of 2.20 ms at
  128×128×64), and the fine operator of MG-CG (`_apply_fine!`, which
  writes `out` directly).
* `_l2` and `_project_mean!` use `plane_partials!` and `ordered_sum`,
  with the partials in a per-level buffer (`MGLevel.part`).

`test/ka_multigrid.jl` compares the smoother, residual, fine operator,
restriction and prolongation with verbatim copies of the old loops, bit
for bit. It covers even and odd `nx`, constant and cut-cell coefficients
and three closures, on the default, the serial and a `threshold = 1`
backend (the last threads every launch).

**Serial coarse levels.** No multigrid-specific threshold: the coarse
levels go through the global `launch!` threshold (32 768 points). A
colour sweep covers half a level, so at 64×64×32 only the finest level
is threaded (65 536 points per colour), and at 128×128×64 the two
finest. 32×32×16 and below, and the 50 sweeps of the coarsest solve,
run serially on the calling task. `_restrict_coeff` is setup only and
stays a serial loop.

**Projections.** The operator is always singular (every axis periodic
or Neumann), so the mean is projected out up to six times per level per
cycle: after each smoothing, after the prolongation, from the residual
and from the coarse right-hand side. Where it saves a sweep the
projection is fused into a neighbouring kernel:

* the residual's mean is subtracted inside the restriction, which reads
  `r − m`. The result has the bits of a projection followed by the
  restriction, and the sweep that wrote the projected `r` is gone;
* the right-hand side's mean is reduced while it is copied in (fused
  `plane_reduce!`);
* the final projection of the solution is fused with the copy-out;
* the projection of the convergence check's residual is fused with its
  sum of squares.

The sums were *not* fused into the stencil launches. A `plane_reduce!`
whose body evaluates the residual and writes it took 1.49 ms at
128×128×64 on one thread. The residual launch plus a separate sum took
0.90 ms: the eight-lane reduction keeps the stencil from vectorising.
The restriction showed the same (0.24 against 0.10 ms).

**One-time change (expectation).** The means and norms were serial
`for k, j, i` sums. They are now canonical plane sums. This changes the
solution in the last bits and could have moved a cycle count by ±1 (the
issue allowed that). Measured over 22 cases (MG with three closures,
W-cycle, FMG, stretched and cut-cell; MG-CG with two closures, stretched
and cut-cell; each at 32×32×16 and 64×64×32), every cycle and iteration
count is identical to the old loops. The largest change of the solution
is `max|Δx| / max|x|` = 2.2e-15 (MG on the stretched grid at 64×64×32);
the others are 3.1e-16 to 1.6e-15. No regression reference of the suite
moved, and no assertion was changed. A solve allocates 0 B on one thread
(`@allocated`, for every case above).

**Measured.** ms per `solve!`, minimum of 7 (64×64×32) or 3 (128×128×64,
stretched), load average 7-8, Julia 1.13.1. Old = the merge base
(master + KA-g) at one thread. Speed-up = old 1 t / new N t; in brackets
new 1 t / new N t. Source: seeded random, tol 1e-10, default closure
(periodic x, y; Neumann z) unless stated. Extent 1000 × 1000 × 200 m.

| solver | grid | cycles | old 1 t | new 1 t | new / old | 4 t | 12 t | 24 t |
| :----- | :--- | -----: | ------: | ------: | --------: | --: | ---: | ---: |
| MG | 64×64×32 | 21 | 66.0 | 29.1 | 0.44 | 2.3× (1.03×) | 2.5× (1.12×) | 2.0× (0.88×) |
| MG | 128²×64 | 21 | 580.5 | 284.8 | 0.49 | 3.4× (1.66×) | 4.2× (2.06×) | 3.0× (1.47×) |
| MG, all Neumann | 64×64×32 | 24 | 76.0 | 33.2 | 0.44 | 2.5× (1.07×) | 2.7× (1.17×) | 2.4× (1.04×) |
| MG-CG | 64×64×32 | 12 | 41.1 | 19.2 | 0.47 | 2.3× (1.06×) | 2.2× (1.01×) | 1.6× (0.76×) |
| MG-CG | 128²×64 | 12 | 382.6 | 211.2 | 0.55 | 2.9× (1.57×) | 3.0× (1.68×) | 2.5× (1.36×) |
| MG-CG, stretched (×1.04) | 64×64×32 | 43 | 212.2 | 126.1 | 0.59 | 2.6× (1.53×) | 2.2× (1.29×) | 1.8× (1.07×) |

Per-level split of one V-cycle at 128×128×64 (ms; a level's own time
is the cycle from that level minus the cycle from the next coarser one):

| level | size | old 1 t | new 1 t | new 4 t | new 12 t | new 24 t |
| :---- | :--- | ------: | ------: | ------: | -------: | -------: |
| 1 | 128×128×64 | 21.44 | 9.45 | 5.10 | 5.01 | 5.44 |
| 2 | 64×64×32 | 2.47 | 1.09 | 0.92 | 0.93 | 1.00 |
| 3 | 32×32×16 | 0.32 | 0.13 | 0.14 | 0.14 | 0.14 |
| 4 | 16×16×8 | 0.043 | 0.021 | 0.022 | 0.021 | 0.025 |
| 5 (coarsest, 50 sweeps) | 8×8×4 | 0.067 | 0.049 | 0.048 | 0.048 | 0.048 |

Reading the tables:

* **The 1-thread gate holds**, by a wide margin: 0.44-0.59× of the old
  loops. Most of the gain comes from the smoother (half the cells per
  launch instead of a parity test on all of them, and vectorised); the
  rest from the faster prolongation and the dropped projection sweep.
* **Threads.** The issue asked for at least 2.5× at 4 threads and 4× at
  12 at 128²×64. Against the old loops it is 3.4× and 4.2×; against the
  new code on one thread only 1.7× and 2.1×. The finest level stops
  scaling at 4 threads (5.1 ms, and 5.0 ms at 12). It is memory-bound
  once vectorised: one colour sweep reads φ and b (17 MB) in 0.65 ms on one
  thread. Level 2 gains 1.2× (launch start-up against 65 536 points),
  and the serial levels are flat by design. 24 threads are slower than
  12 everywhere. The uniform 64×64×32 solves gain 1.0-1.2× from
  threads (the stretched one 1.5× at 4). As in KA-g, the task start-up
  of the many small launches per cycle costs about what the work does.

**Thread invariance.** The worker's new `multigrid` section hashes the
solution, the cycle count and the residual history of MG and MG-CG at
64×64×32 and 128×128×64. It covers three closures, the W-cycle, FMG,
stretched z and cut-cell coefficients around a solid block: 58 hashes,
identical at 1, 4 and 24 threads. The whole harness is quoted in the PR.
One of its cases does not converge: plain MG on the stretched grid
(×1.04, extent 2000 m in z) at 128×128×64 stops at its cap of 200
cycles with a relative residual of 1.04e-6 and logs the
`maxcycles reached` warning. The old loops do the same (200 cycles,
1.0370739097032905e-6 against 1.0370739097035124e-6 now): point
smoothing stalls on the anisotropic cells. MG-CG solves the same case
in 61 iterations. It is kept as a hash, not a convergence test.

**Re-checked after the master merge** (KA-d's `_row!` / `_nest!` launch
change, load 4-8). Every solution hash and cycle count is unchanged. ms
per solve, old 1 t / new 1 t / 4 t / 12 t / 24 t: MG 128²×64 581.4 /
276.4 / 173.6 / 150.4 / 179.9 (new / old 0.48; 3.3×, 3.9×, 3.2× against
the old loops). MG-CG 128²×64 390.3 / 210.2 / 96.6 / 127.7 / 155.9
(0.54; 4.0×, 3.1×, 2.5×). MG 64×64×32 66.2 / 28.7 / 25.8 / 25.6 / 29.7
(0.43). MG-CG stretched 64×64×32 215.3 / 110.6 / 83.4 / 105.7 / 111.2
(0.51). The finest level of a V-cycle at 128²×64 takes 10.0 ms at 1 t
and 5.6 / 5.2 / 7.1 ms at 4 / 12 / 24 t. The threaded times vary by
±20 % between runs with other agents on the machine; the 1-thread gate
(≤ 1.10×) holds at 0.43-0.54×.

## KA-d: grid metrics and the finite-volume kernels (#299)

**One kernel per operator for both grids.** Every kernel of
`Operators.FiniteVolume` is a point body run by `launch!`, and the body
reads the grid only through its *metrics*. `grid_metrics(g)` returns a
`GridMetrics`: uniform `1/Δx`, `1/Δy` (and `1/Δx²`, `1/(2Δx)`, …) and, in
z, a `UniformZ` on a `RegularGrid` (scalars, built per call, no
allocation) or a `LevelZ` on a `StretchedGrid` (vectors over the storage
levels, built once with the grid by `StretchedGrids.level_metrics` and
cached in its `zm` field). A body asks for a metric through an `@inline`
accessor:

| accessor | `RegularGrid` (`UniformZ`) | `StretchedGrid` (`LevelZ`) |
| :------- | :------------------------- | :------------------------- |
| `_idx(m, i)`, `_idy(m, j)` | `1/Δx`, `1/Δy` | same |
| `_idz(m, k)` | `1/Δz` | `1/Δz_c[k]`: across cell `k` (face → centre; the volume of a centre-located field) |
| `_idzf(m, k)` | `1/Δz` | `1/Δz_f[k]`: centres `k−1`, `k` (centre → face; the volume of `w`) |
| `_idz_off(m, k)` | `1/Δz` | `2/(Δz_f[k] + Δz_f[k+1])`: ∂u/∂z at a centre |
| `_zface(m, k, lo, hi)` | `0.5*(lo + hi)` | `wlo[k]*lo + whi[k]*hi`, the geometric weights |
| `_zflux(m, k, lo, hi)` | `0.5*(lo + hi)` | `alo[k]*lo + ahi[k]*hi`, the side-face volume flux of a `w` cell |
| `_zcentre(m, k, lo, hi)` | `0.5*(lo + hi)` | `clo[k]*lo + chi[k]*hi` |
| `_lapz(m, loc, k, fm, c, fp)` | `(fp − 2c + fm)/Δz²` | the conservative flux difference, `w`'s metrics for `loc = FaceZ` |
| `_addz3(m, k, acc, fm, fc, fp)` | `acc + (fp − fm)/(2Δz)` | `acc + d3m[k]*fm + d3c[k]*fc + d3p[k]*fp` |
| `_filter_width(m, k)` | `(Δx Δy Δz)^{1/3}` | `(Δx Δy Δz_c[k])^{1/3}` |

**The contract:** every accessor evaluates the *same floating-point
expression* that the pre-port kernel of that grid evaluated (on a
uniform grid `0.5*(a + b)`, never `0.5a + 0.5b`; on a stretched grid the
per-level quantity the old loop computed at the top of its `k` loop,
precomputed with the same expression). A body written against the
accessors is therefore the old RegularGrid kernel on a `RegularGrid`
and the old StretchedGrid kernel on a `StretchedGrid`, bit for bit. A
new kernel that needs a metric not in the table adds an accessor with a
method for both types; a mesh without `grid_metrics` (the
`TerrainFollowingGrid`, whose metrics are 3-D) keeps its own methods.

**What was folded.** The `StretchedGrid` had about twenty kernels that
were copies of the RegularGrid ones with the vertical metric swapped in:
the Phase-24 derivatives, Laplacian, divergence, interpolation, surface
flux, sponge and buoyancy of `StretchedGrids.jl`, and the #151 / #276
momentum and scalar advection, `w` Laplacian, moist buoyancy,
Smagorinsky, eddy viscosity `u/v/w` and eddy diffusion of
`stretched/boussinesq.jl`. They are gone; the FiniteVolume kernels take
`g::AbstractMesh`. `stretched/boussinesq.jl` keeps only what has no
RegularGrid twin there (the core's `_dzc/_dzf/_dz_min` and the TKE
closure's N²).

**Ported to `launch!`** (all in FiniteVolume unless noted):

* derivatives (collocated and staggered), Laplacian, divergence (both),
  interpolation, the surface flux (a 2-D launch);
* momentum advection (three launches), dry and moist buoyancy, Coriolis
  (one launch writing `tu` and `tv`), Smagorinsky νt, eddy viscosity
  `u/v/w`, the SGS stress divergence (one launch), the stored-stress
  divergence, `_sgs_stress_tensor!` (one launch per component range);
* scalar eddy diffusion (every variant, including the masked and weighted
  `_add_eddy_diffusion_scalar_masked!`), the tensor diffusion (one launch
  per face direction over its extended range, e.g. `i ∈ Hx:i2`, then the
  divergence; the face gradients `gW … gT` are the named functions
  `_tg_W … _tg_T`), `_isentropic_tensor!`;
* WENO-5, centred (staggered and collocated) and upwind scalar advection
  (all three are public API, `advect_weno5!`, `advect_centered!`,
  `advect_upwind!`: ported, not deleted);
* `StretchedGrids`: `_set_physical!` (one launch over the storage; `fun`
  must not mutate shared state), and the `TerrainFollowingGrid` kernels
  (derivatives, divergence, Laplacian, surface flux). The staggered
  divergence used a column buffer for the contravariant flux Ω; each work
  item now evaluates the Ω of its two faces with the same arithmetic
  (a face is computed twice). The Laplacian is three flux launches and a
  divergence launch.

**`_add_damping!` with `levels`** (the #288 budget path) is launched over
the levels only (`launch_weighted!` over `k` with weight `Nx·Ny`): one
work item owns plane `k` and runs its `(j, i)` loop serially, so the sum
into `levels[kl]` is the old one at every thread count. A 3-D launch
would have several tasks adding into the same `levels[kl]`. Without
`levels` the damping is a 3-D pointwise launch.

**A launch change for large bodies.** WENO-5 and the tensor-diffusion
fluxes first ran 3.1× and 5.2× the old loops at one thread. The bodies
were not the problem (the same body in a plain loop matched the old
kernel); the `args` tuple was: see "The arguments reach the loop as
arguments" above (`_nest!` / `_row!`, a `Backends.jl` change that leaves
every bit unchanged). The tensor fluxes also took their flags and bounds
in a struct, which cost 2.3× on its own; they take scalars now.

**Tests.** `test/ka_fv.jl` compares every ported kernel bit for bit with
a transcript of the pre-port loops (`test/support/legacy_fv.jl`: the old
FiniteVolume, StretchedGrids and `stretched/boussinesq.jl` methods) on a
RegularGrid, a geometric and a `:uniform` StretchedGrid and a
TerrainFollowingGrid over a hill, on random fields with random halos,
on the serial, threshold-0 and default backends (600 checks; the folded
stretched kernels against the copies they replace); the damping with
`levels` on every backend; the metric accessors against the old per-level
expressions; zero allocation on the serial path. The harness has an `fv`
section (below). `test/stretched.jl` asked *which module* defined
the StretchedGrid stencils; with one kernel per operator that
expectation was obsolete, and it now checks that a StretchedGrid hands
the kernels its per-level metrics (its behavioural check is unchanged).

No bit changes: KA-d names no one-time change, and none occurred (every
kernel bit-identical to the old loop, serially and threaded).

**Measured** (same machine, a Ryzen 9 7900: 12 cores / 24 threads, 2 ×
32 MB L3, dual-channel DDR5; minimum of 200 runs at 64×64×32, 8 runs at
256³; load 6–9 from other agents; µs; old = the pre-port loop at one
thread; speed-up = old at 1 thread / new at N threads; R = RegularGrid,
S = geometric StretchedGrid):

| kernel | old, 1 t (µs) | new, 1 t (µs) | 1 t / old | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: | --: | --: |
| R ∂x c→f | 21.2 | 21.1 | 0.99 | 1.3× | 1.5× | 1.1× |
| R ∂z c→f | 28.3 | 28.1 | 1.00 | 1.6× | 1.6× | 1.5× |
| R ∂z cc | 27.9 | 27.9 | 1.00 | 1.8× | 2.1× | 1.4× |
| R interp c→fz | 28.2 | 28.1 | 1.00 | 1.2× | 1.8× | 1.5× |
| R laplacian cc | 46.5 | 46.9 | 1.01 | 2.2× | 2.1× | 1.9× |
| R laplacian w | 45.8 | 45.8 | 1.00 | 1.8× | 2.1× | 1.9× |
| R divergence | 48.9 | 48.4 | 0.99 | 1.3× | 2.0× | 1.7× |
| R surface flux | 0.7 | 0.7 | 1.03 | 0.9× | 0.9× | 1.0× |
| R momentum advection | 302.9 | 303.1 | 1.00 | 2.4× | 2.3× | 2.0× |
| R scalar advection | 92.7 | 93.6 | 1.01 | 2.0× | 2.3× | 2.1× |
| R buoyancy | 30.4 | 30.4 | 1.00 | 1.8× | 1.7× | 1.5× |
| R moist buoyancy | 48.6 | 48.2 | 0.99 | 1.8× | 2.1× | 1.8× |
| R coriolis | 66.0 | 66.1 | 1.00 | 2.0× | 2.4× | 2.2× |
| R smagorinsky | 791.8 | 747.4 | 0.94 | 3.9× | 4.8× | 5.5× |
| R eddy viscosity u | 115.7 | 116.0 | 1.00 | 1.6× | 2.6× | 2.8× |
| R eddy viscosity w | 112.8 | 112.1 | 0.99 | 1.8× | 2.7× | 2.5× |
| R eddy diffusion scalar | 103.1 | 104.6 | 1.01 | 2.2× | 3.0× | 2.4× |
| R sponge | 317.4 | 292.7 | 0.92 | 2.0× | 2.6× | 3.6× |
| R damping with levels | 345.6 | 299.8 | 0.87 | 2.1× | 1.9× | 2.0× |
| R masked eddy diffusion | 1124.0 | 959.8 | 0.85 | 4.5× | 5.0× | 6.0× |
| R SGS stress divergence | 522.7 | 520.7 | 1.00 | 3.4× | 3.1× | 3.4× |
| R SGS stress tensor | 287.0 | 288.1 | 1.00 | 1.6× | 1.2× | 1.3× |
| R stored-stress divergence | 142.1 | 142.6 | 1.00 | 2.1× | 2.7× | 2.4× |
| R tensor diffusion | 631.0 | 621.5 | 0.98 | 2.7× | 2.6× | 2.5× |
| R isentropic tensor | 196.7 | 197.3 | 1.00 | 2.4× | 1.1× | 2.8× |
| R WENO-5 | 1957.4 | 1975.6 | 1.01 | 3.9× | 4.1× | 6.7× |
| S ∂x c→f | 20.9 | 21.2 | 1.01 | 1.4× | 1.3× | 1.1× |
| S ∂z c→f | 28.4 | 28.5 | 1.00 | 1.4× | 1.8× | 1.4× |
| S ∂z cc | 37.6 | 38.5 | 1.02 | 1.6× | 1.9× | 1.7× |
| S interp c→fz | 28.8 | 28.6 | 0.99 | 1.3× | 1.6× | 1.4× |
| S laplacian cc | 45.9 | 47.4 | 1.03 | 1.5× | 1.9× | 1.9× |
| S laplacian w | 47.0 | 49.0 | 1.04 | 1.9× | 2.0× | 1.8× |
| S divergence | 49.7 | 48.9 | 0.98 | 1.8× | 2.0× | 1.9× |
| S surface flux | 0.7 | 0.7 | 1.00 | 0.9× | 1.0× | 1.0× |
| S momentum advection | 315.0 | 318.9 | 1.01 | 2.5× | 2.9× | 2.2× |
| S scalar advection | 95.2 | 95.1 | 1.00 | 2.1× | 2.5× | 2.2× |
| S buoyancy | 30.3 | 30.8 | 1.02 | 1.2× | 1.7× | 1.4× |
| S moist buoyancy | 48.4 | 47.9 | 0.99 | 1.2× | 2.2× | 1.9× |
| S coriolis | 66.5 | 65.9 | 0.99 | 2.5× | 2.4× | 2.2× |
| S smagorinsky | 746.1 | 789.4 | 1.06 | 3.6× | 4.0× | 4.6× |
| S eddy viscosity u | 109.8 | 112.3 | 1.02 | 2.4× | 3.3× | 2.5× |
| S eddy viscosity w | 115.4 | 115.8 | 1.00 | 2.6× | 2.7× | 2.6× |
| S eddy diffusion scalar | 104.7 | 106.2 | 1.01 | 2.6× | 2.5× | 2.5× |
| S sponge | 283.3 | 257.0 | 0.91 | 1.9× | 2.8× | 3.1× |
| S damping with levels | 303.4 | 265.7 | 0.88 | 1.9× | 1.6× | 1.9× |

At 256³ (DRAM-bound):

| kernel | old, 1 t (µs) | new, 1 t (µs) | 1 t / old | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: | --: | --: |
| R laplacian cc | 15456.5 | 15633.4 | 1.01 | 1.7× | 1.7× | 1.6× |
| R divergence | 17178.2 | 17197.4 | 1.00 | 1.2× | 1.2× | 1.1× |
| R momentum advection | 87552.2 | 87564.9 | 1.00 | 1.9× | 1.6× | 1.4× |
| R scalar advection | 30407.0 | 31573.6 | 1.04 | 1.5× | 1.5× | 1.4× |
| R smagorinsky | 114957.5 | 108128.6 | 0.94 | 3.8× | 4.3× | 4.3× |
| R eddy diffusion scalar | 25217.2 | 25761.6 | 1.02 | 1.9× | 1.5× | 1.4× |
| R masked eddy diffusion | 144628.6 | 124607.9 | 0.86 | 4.0× | 6.3× | 6.9× |
| R tensor diffusion | 117952.6 | 117062.8 | 0.99 | 1.6× | 1.5× | 1.4× |
| S laplacian cc | 15610.6 | 15377.3 | 0.99 | 1.7× | 1.6× | 1.6× |
| S divergence | 17998.0 | 17714.2 | 0.98 | 1.2× | 1.2× | 1.2× |
| S momentum advection | 82376.1 | 82006.6 | 1.00 | 1.7× | 1.4× | 1.3× |
| S scalar advection | 29441.5 | 29893.1 | 1.02 | 1.6× | 1.4× | 1.3× |
| S smagorinsky | 114068.0 | 115879.0 | 1.02 | 3.5× | 4.6× | 4.5× |
| S eddy diffusion scalar | 25350.4 | 26754.3 | 1.06 | 1.9× | 1.8× | 1.4× |

Reading the tables:

* **The 1-thread gate holds for every kernel**: largest 1.06 (the
  stretched Smagorinsky; its RegularGrid twin is 0.94). Re-measured
  after merging master `b4f6f82` (load 6-10, so absolute times 15-50 %
  higher, the ratios noisier): every 64×64×32 kernel 0.92-1.08 (largest
  R `∂z c→f` 1.08, R SGS stress divergence 1.07). Several kernels
  are faster (not profiled): the damping kernel 0.87–0.92, the masked
  eddy diffusion 0.85, the RegularGrid Smagorinsky 0.94.
* **At 64×64×32 the light stencils reach 1.5–3×**, not the 4–6× the
  issue expected. Each takes 20–120 µs at one thread; a launch costs
  about 10–16 µs of task start and join (measured at KA-a), and the
  points-per-task rule (16 384) gives a 131 072-point launch at most 8
  tasks. The heavy kernels reach 4.6–6.7× at 24 threads (Smagorinsky,
  masked diffusion, WENO-5). Fusing launches (fewer, longer kernels per step) is the lever
  for the light ones; that is not in KA-d's scope.
* **At 256³ the stencils are DRAM-bound**: 1.2–1.9× at 4–24 threads.
  A plain threaded triad (`a = b + 2c` over 256³, `Threads.@threads`)
  moves 25.6 GB/s on one thread and 35.7 / 34.4 / 36.7 GB/s at 4 / 12 /
  24 threads: at most 1.4×, so the stencils sit at the memory ceiling of
  this dual-channel machine (they beat the triad's ratio a little through
  cache reuse of the stencil neighbours). The compute-heavy kernels scale
  further (Smagorinsky 4.3–4.6×, masked diffusion 6.9× at 24 threads).
* The 64×64×32 24-thread column was re-measured at load 5 (the first
  run overlapped other agents' jobs at load 9 and gave e.g. S `∂x` 0.5×);
  the columns still carry a few tenths of noise (the R isentropic tensor's
  1.1× at 12 threads against 2.4× and 2.8× at 4 and 24).

**Flagship** (`CaseConfig()`, the 64×64×64 Alexanderplatz run, minimum
of 3 × 10 steps after 2 warm-up steps, load 5–11): **237.1 ms/step before
KA-d, 238.0 ms/step with KA-d at one thread (1.004×)**; 212.9 / 180.0 /
198.2 ms/step at 4 / 12 / 24 threads. The tracer-transport line of
`timing_report` (`transport/scalar_advection`: the tracers' advection and
their SGS diffusion through `_add_eddy_diffusion_scalar_masked!`) is
62.3 ms/step before and 64.4 with KA-d at one thread (1.03×, inside the
rep-to-rep spread of 3 %), 59.6 / 47.6 / 55.9 at 4 / 12 / 24 threads; the
`dynamics/step!` line 50.1 → 49.6 (44.7 / 38.4 / 41.7). The step is still
dominated by loops later port PRs move (nesting 84 ms, the tracers'
WENO-Z advection, chemistry, fog).

**The race the `levels` path avoids, measured.** A 3-D launch of the same
damping body with `levels` (128×128×32, `threshold = 0`, `-t 24`, where
the row workgroups split each plane between tasks) gave level sums
different from the serial loop in 320 of 20 × 32 level values; the
plane-per-work-item launch differed in none.

**Thread invariance.** The harness (`AEOLUS_THREAD_TESTS=1`,
`AEOLUS_THREAD_COUNTS=1,4,24`) with the new `fv` section: every FV kernel
on seeded random fields with filled halos on a 64×64×32 RegularGrid, a
geometric StretchedGrid and a TerrainFollowingGrid, the damping kernel
with `levels` (tendency and per-level sums, also at 128×128×32 where row
workgroups split a plane), and 3 steps of a Smagorinsky LES on the
stretched grid (87 hashes). All 2953 hashes (primitives, 64³ DNS,
reductions, the Berlin smoke matrix, the flagship step, nested, fft, fv)
are identical at 4 and 24 threads to 1 thread. At `-t 1` every kernel is
the old loop bit for bit (`test/ka_fv.jl`): no one-time change.

## KA-e: scalar transport, positivity limiter, momentum advection (#302)

**Ported to `launch!`** (all in AdvectionSchemes): the flux-form sweep
`_advect_reconstructed!` (one launch over the interior; every cell still
evaluates both of its faces, so neighbouring cells evaluate a shared face
twice exactly as before: sharing face fluxes would change bits and is not
done here), `_reconstructed_fluxes!` and `_upwind_fluxes!` (three launches
each, over the face ranges `last + 1`), `_add_flux_divergence!`, the
WENO-Z and skew-symmetric momentum advection (one launch per component),
the variance-conserving advection, and `boundary_scalar_flux` (#284). The
reconstruction point functions stay `@inline` scalar functions on
`NTuple`s.

**The limiter in three phases.** `_positivity_limit!` is pass 1 (the
ratio `R` of every cell, one launch), the halo of `R` (`_wrap_one_layer!`
as three 2-D launches over the face planes, or `fill_halos!(R, bcs)`),
and pass 2 (each face flux rescaled with its donor's `R`, three launches).
Pass 2 and the upwind fluxes choose with `ifelse` instead of a branch on
the sign of the flux or the velocity: the same value (both operands are
computed, one is chosen), and at one thread the limiter ran 240 µs
instead of 1800 µs on random-sign fluxes, where the branch mispredicted
on half the faces.

**Scratch.** `apply_positivity_limiter!(…; ratio = nothing)` and
`advect_positive!(…; workspace = nothing)` throw an `ArgumentError` when
the backend runs threaded: a time loop allocates the scratch once. The
Berlin coupling (`couplings/scalar_advection.jl`), the TKE closure and the
Reynolds-stress closure pass their `FluxLimiterWorkspace`; the coupling
builds the reconstruction descriptor once. With a workspace,
`advect_positive!` and `boundary_scalar_flux` allocate nothing on a serial
backend (`@allocated`, `test/ka_advection.jl`).

**`boundary_scalar_flux`.** Its totals, per-level sums and mass-flux
corrections are ordered sums, so they stay serial in the old loop order.
When a face plane's 2-D launch runs on several tasks the face terms are
computed by it into a scratch vector (one module-level buffer, grown once,
behind a lock) and the sums then run over the scratch; on a serial launch
the sums compute the terms on the fly, the old fused loop (a separate
pass over the scratch was 1.6-2.2× the fused loop at one thread). The
two-point schemes (`:centered`, `:upwind`) always take the fused loop:
their face work is smaller than the extra pass (256×256×128 box,
`:centered`: 306 µs fused at one thread, 455 / 487 / 730 µs threaded at
4 / 12 / 24). A 2-D serial launch passes the body's arguments separately
(`_nest2!`, as `_nest!` for 3-D): through the tuple the face bodies ran
2-2.5× the old loop.

**Bit for bit.** `test/ka_advection.jl` keeps a transcript of the pre-port
loops (`test/support/legacy_advection.jl`) and compares, on seeded random
fields and halos included: the sweep of every reconstruction (WENO-Z5/7,
TENO5/8, WENO-JS5, Centered2, upwind), the reconstructed and upwind face
fluxes, the limiter (with and without low-order fluxes, periodic wrap and
BC halos, positivity active), `advect_positive!` on a RegularGrid and a
StretchedGrid, both momentum advections, the variance-conserving
advection and `boundary_scalar_flux` (every flux-form scheme, levels,
corrections, diffusion, weights, `face_fluxes`). Each runs serially,
threaded with every launch on the KA path (`threshold = 0`) and on the
default backend: all identical to the old loops. No one-time change: at
one thread the flagship's state after 4 steps hashes the same as master
(`b20cfbd49603e21f`).

**Measured** (Ryzen 9 7900, 12 cores / 24 threads; µs, minimum of 200
calls, 64×64×32 RegularGrid, halo 3; load 5-7 from other jobs; old = the
transcript at one thread):

| kernel | old, 1 t | new, 1 t | new/old | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: | --: | --: |
| sweep WENO-Z5 | 2235 | 2263 | 1.01 | 583 (3.9×) | 430 (5.3×) | 435 (5.2×) |
| sweep TENO5 | 2657 | 2663 | 1.00 | 684 (3.9×) | 526 (5.1×) | 470 (5.7×) |
| sweep Centered2 | 103 | 103 | 1.00 | 36 (2.8×) | 38 (2.7×) | 40 (2.6×) |
| face fluxes WENO-Z5 | 1042 | 1037 | 0.99 | 295 (3.5×) | 261 (4.0×) | 261 (4.0×) |
| upwind fluxes | 275 | 103 | 0.38 | 65 (1.6×) | 66 (1.6×) | 78 (1.3×) |
| limiter (3 phases, walls) | 1800 | 240 | 0.13 | 188 (1.3×) | 152 (1.6×) | 179 (1.3×) |
| flux divergence | 52 | 52 | 1.01 | 28 (1.9×) | 24 (2.2×) | 26 (2.0×) |
| advect_positive! WENO-Z5 | 3425 | 1451 | 0.42 | 571 (2.5×) | 568 (2.6×) | 609 (2.4×) |
| momentum WENO-Z5 | 6658 | 6664 | 1.00 | 1718 (3.9×) | 1452 (4.6×) | 1387 (4.8×) |
| momentum skew | 301 | 305 | 1.01 | 125 (2.4×) | 118 (2.6×) | 132 (2.3×) |
| variance conserving | 87 | 87 | 1.00 | 35 (2.5×) | 31 (2.8×) | 39 (2.2×) |
| bsf `:centered`, 64×64×32 | 16 | 16 | 1.03 | 16 | 16 | 16 |
| bsf `:weno_z5`, 64×64×32 | 647 | 129 | 0.20 | 129 | 130 | 130 |
| bsf `:centered`, 256×256×128 | 283 | 290 | 1.02 | 286 | 358 | 287 |
| bsf `:weno_z5`, 256×256×128 | 14245 | 2444 | 0.17 | 1544 (1.6×) | 1667 (1.5×) | 1497 (1.6×) |

Speed-ups are against the new code at one thread. Every one-thread time is
within the 1.10× gate (worst: `boundary_scalar_flux :centered` 1.03×).
`boundary_scalar_flux` at 64×64×32 stays serial (its planes are below the
launch threshold); with WENO-Z5 it runs 5× faster at one thread than
master (651 → 129 µs, master's own function measured too). The arithmetic
is the same; the old loop body called a local closure `addl!` and the
new one top-level functions with every array an argument. Which of the
two made the difference was not isolated.

**Flagship** (`CaseConfig()`, 64×64×64; ms/step, minimum of 3 × 8 steps
after 4 warm-up steps; load 5-8 from other jobs; master = 8cb0b572):

| threads | step, master | step, KA-e | `scalar_advection`, master | `scalar_advection`, KA-e |
| --: | --: | --: | --: | --: |
| 1 | 254.9 | 246.3 | 57.3 | 50.0 (0.87×) |
| 4 | 238.6 | 217.5 | 57.4 | 34.3 (1.46×) |
| 12 | 241.8 | 203.1 | 59.3 | 29.1 (1.72×) |
| 24 | 246.1 | 210.2 | 58.2 | 31.6 (1.58×) |

(KA-e speed-ups against KA-e at one thread.) The stage is short of the
issue's targets (≥ 2.5× at 4, ≥ 5× at 12). Per tracer at 12 threads,
`advect_positive!` takes 1.14 ms (3.26 at one thread, 2.9×) and the SGS
diffusion `sgs_diffuse!` 0.63-0.81 ms (1.45, 1.8-2.3×); the limiter and
the upwind fluxes are memory-bound and gain 1.3-1.6×. The 12- and 24-thread
runs shared the machine with other jobs (load 5-8 on 12 cores), so they
are lower bounds.

**Launch overhead.** One `scalar_advection` stage (8 advected tracers)
makes 259 launches, 96 of them threaded (the others are below the
threshold: halo planes, the 2-D wraps). An empty-body launch over the 64³
interior costs 6.3 / 13.3 / 21.6 µs at 4 / 12 / 24 threads, so about
0.6 / 1.3 / 2.1 ms of the stage is launch overhead.

**Thread invariance.** The harness (`AEOLUS_THREAD_TESTS=1`,
`AEOLUS_THREAD_COUNTS=1,4,24`) with the new `advection` section (64
hashes: the sweep, face fluxes and `advect_positive!` with its ratio
field, periodic wrap and wall halos, of every reconstruction; the bare
limiter; both momentum advections; the variance-conserving advection;
`advect_positive!` on a stretched grid; `boundary_scalar_flux` on a
256×256×128 open box, where its face planes launch on several tasks).
All 3071 hashes (primitives, dns, reductions, smoke, flagship, nested,
fft, fv, pcg, radiation, advection) are identical at 4 and 24 threads to
1 thread.

## KA-n: radiation and stochastic modules (#305)

**Radiosity.** Three changes.

* *LU cache.* [`RadiosityFactorCache`](@ref Aeolus.Radiosity.RadiosityFactorCache)
  keeps the dense LU of ``M = I - \mathrm{diag}(\rho) F`` between calls
  and refactorises only when ``F`` (by identity) or ``\rho`` (by value)
  changes. A hit runs the same LAPACK `getrs` on the same factors, so the
  bits equal a refactorisation at the same BLAS thread count. The
  `radiosity_sun` coupling keeps one cache per band. The longwave
  ``\rho = 1 - \varepsilon`` is fixed, so its LU is computed once per run.
  The shortwave albedos follow μ₀ (`default_urban_materials(; μ₀)`; all
  five classes change, from 0.140 / 0.268 / 0.293 / 0.359 / 0.272 at
  μ₀ = 0.1 to 0.121 / 0.204 / 0.186 / 0.031 / 0.217 at 0.9), so the
  shortwave operator is still refactorised on every daytime call.
* *BLAS.* Pinned to one thread when Julia runs threaded (KA-a); the
  LU and its solves run on `dense_blas_threads()` since #496 (1.773 →
  0.358 s for the 5639 facets with the threaded assembly, 12 threads).
  LAPACK rounds differently per BLAS thread count, within round-off.
* *View factors and aggregation.* `view_factors` is one
  `launch_weighted!` over pairs of upper-triangle rows `(p, n − p)`, each
  writing `F[i, j]` and `F[j, i]` of its own rows. The Gauss nodes are
  cached before the threads read them, and the occlusion quadrature no
  longer allocates. `F` and `F_sky` are bit-identical to the serial loop.
  `aggregate_to_grid` gathers through a CSR column → facets index
  (`facet_column_index`) in ascending facet order, which gives the same
  sums as the old facet scatter (bit-identical, no expectation change).
  An exact tie between dominant classes goes to the lowest material
  index. `set_temperatures!(scene, Ts, facet_col)` is one launch over the
  facets.

The iterative (Gauss-Seidel) radiosity solver stays serial.

**Gas optics and two-stream.** `compute_optics!` is one
`launch_weighted!` over (column, layer): a work item fills one layer's
g-points in the serial loop order, so the optics are bit-identical at
every thread count. The k-distribution bisections are 33 of the 37 ms of a
Berlin radiation call. The multi-column forms take optics `τ[k, g, c]`
(`allocate_column_optics`). `solve_sw!` / `solve_lw!` launch over
(g-point, column) work items, with the adding recurrence sequential in
``k`` inside each item. Per-(g, column) scratch lives in a reusable
[`TwoStreamWorkspace`](@ref Aeolus.TwoStream.TwoStreamWorkspace), so a
call does not allocate. A second launch sums the g-points in ascending
order. A single-column solve is the multi-column solve on one column (one
code path), so column ``c`` gets the single-column bits.
`column_radiation!`, the entry the Berlin child and the nesting parent
share (#259), runs through these forms.

**Cloud shadow.** The optical-depth field is a pointwise launch, and the
slant rays are one `launch_weighted!` over the columns, one ray per work
item. Both are identical to the old loops.

**Random streams.** No generator is shared between work items.
[`counter_rng`](@ref Aeolus.Backends.counter_rng)`(seed, stream, id)` is
a Philox4x32-10 stream (Salmon et al. 2011, checked against the
Random123 known-answer vectors) and a pure function of the triple.
[`counter_xoshiro`](@ref Aeolus.Backends.counter_xoshiro) is a Xoshiro256++
seeded from the first four numbers of that stream, for long streams.

| module | work item | stream |
| :-- | :-- | :-- |
| `MonteCarloRT.mc_solve` | a batch of photons, tallied into its own buffer and merged in batch order | photon ``p``: `(key, g-point, p)` |
| `init_exponential_superdroplets`, `broadening_initial_population` | a droplet | `(key, 0, i)` |
| `collision_coalescence!` | a cell | `counter_xoshiro(key, 0, ci)` |
| `SupersaturationFluctuations` | a droplet | the ``n``-th Ornstein-Uhlenbeck draw of droplet ``i``: `(key, n, i)` |

The key is the caller's `seed`, or one `UInt64` drawn from the caller's
`rng`, so a seeded `MersenneTwister` caller stays reproducible.
`mc_solve` without either is an error, and the unseeded global `rand()`
default is gone. A Philox block costs about 4 ns per two `UInt64`s: ten
dependent rounds of multiplication. That is fine for a few draws per
item, and `randn(counter_rng(…))` is not heap-allocated. A cell's
collisions draw a few hundred numbers, which took 1.26× the old loop on
Philox, so they use `counter_xoshiro` (under 1 ns per draw). A Philox that
computes four blocks per refill (2.2 ns per draw) was tried and dropped.
Its larger refill stops the generator from being inlined, so a one-draw
stream is heap-allocated, and the superdroplet initialisation took 1.6×
as long.

**Named expectation changes** (one time, against master):

* *Monte-Carlo photons.* Every realisation changes, but the statistics
  do not: the standard-error calibration over 60 seeds gives sd/err 0.97
  on the new streams against 0.99 on the old. Four `montecarlo`
  assertions held only for the old realisation. Their tolerances are
  unchanged. The conservative closure and the delta-Eddington comparison
  now use 96k photons instead of 24k: at 24k they failed for 1 or 2 of
  20 seeds with the old draws and for 2 or 4 with the new. The 3-D cloud
  "> 5σ" test now uses 128k / 64k photons instead of 32k / 16k: the old
  draws failed it for 5 of 10 seeds. The shadow test's
  `argmin(profile) == i_expect` was a wrong expectation: the τ = 20
  shadow is a plateau of six cells equal within the noise. It now
  asserts that both lie in the geometric shadow.
* *Superdroplets and sub-grid fluctuations.* The realisations change. The
  era4 superdroplet, broadening, sgs-growth and turbulent-microphysics
  tests pass with their tolerances unchanged. The spectral-broadening
  page's seed-1 table and seed lists were re-measured. At
  ``\sigma_S = 0.2\,\%`` the bulk ``\varepsilon`` deficit is −2.9 / −5.1 /
  −4.3 % over seeds 1-3, against −4.1 / −4.1 / −3.5 % before (asserted
  2-6 %). The stationary exponent is 0.540 / 0.529 / 0.526, against
  0.529 / 0.514 / 0.518 before.
* Everything else in this PR is bit-identical to master at `-t 1`: the
  radiosity (cached or not), the view factors, the aggregation, the gas
  and cloud optics, the two-stream solves and the cloud shadow.

**Thread invariance.** The harness's `radiation` section hashes the
following:

* the view factors of a 16×14 city;
* its shortwave and longwave solves, the second longwave solve through
  the LU cache;
* `aggregate_to_grid`;
* the gas and cloud optics and the multi-column solves of 6 columns;
* the cloud optical depth and the shadow rays;
* a Monte-Carlo solve;
* 4 collision steps of 2000 superdroplets in 16 cells, and the sub-grid
  fluctuations.

## Measured at KA-n

Same machine, Julia 1.13.1, BLAS on one thread in every run. Each figure
is the minimum over repeats, with other agents on the machine. The
Berlin radiosity scene is the flagship's (5639 facets). "Day call" is one
`radiosity_sun` `:radiation` stage: materials for a new μ₀, temperatures,
shortwave and longwave solves and aggregation. "Night" is the same
without the sun. The parent columns are 96 columns of the 53 parent
levels under the standard atmosphere (113 layers), LW + SW. The old code
is master `b4f6f82`. Speed-up is old at 1 thread divided by new at N
threads.

| kernel | old 1 t (ms) | new 1 t (ms) | new / old | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: | --: | --: |
| radiosity day call | 3783 | 1800 | 0.48 | 2.1× | 2.1× | 2.1× |
| radiosity night call | 1892 | 17.4 | 0.009 | 110× | 111× | 109× |
| `view_factors` (5639 facets) | 937 | 833 | 0.89 | 2.9× | 4.8× | 5.0× |
| `aggregate_to_grid` | 0.556 | 0.120 | 0.22 | 4.7× | 4.7× | 4.8× |
| `column_radiation!` (child, 91 layers) | 45.7 | 45.4 | 0.99 | 3.6× | 7.4× | 12.1× |
| gas optics LW (1 column) | 20.2 | 20.2 | 1.00 | 3.8× | 7.4× | 12.5× |
| 96 parent columns, optics + LW + SW | 3828 | 3879 | 1.01 | 3.8× | 8.4× | 10.1× |
| `cloud_optical_depth_column` 64×64×32 | 0.0229 | 0.0235 | 1.03 | 1.8× | 1.6× | 1.3× |
| `cloud_shadow_transmittance!` 64×64×32 | 0.798 | 0.805 | 1.01 | 3.8× | 4.2× | 3.6× |
| `mc_solve` 16³ cloud, 64k photons, 24 batches | 95.5 | 98.6 | 1.03 | 3.7× | 9.8× | 9.6× |
| `init_exponential_superdroplets` 64k | 3.37 | 2.64 | 0.78 | 1.8× | 1.9× | 1.9× |
| `collision_coalescence!` 64k in 512 cells, 5 steps | 3.99 | 3.67 | 0.92 | 1.8× | 1.7× | 1.8× |

A single-column `solve_lw!` / `solve_sw!` (0.13 / 0.10 ms) runs serially:
its 1 × 16 or 1 × 14 work items are below the launch threshold. It takes
1.00 / 1.02× the old time.

* **The issue's radiosity target (≥ 10× per call) is met at night, not
  by day.** At night only the longwave solve runs, on the cached LU:
  17 ms against 1.9 s. By day the shortwave operator changes with μ₀, so
  it is refactorised (1.8 s, 2.1× faster than before, when both bands
  were). Reusing the LU across μ₀ needs a non-bitwise scheme, for
  example iterative refinement on a stale LU or albedos held across a
  μ₀ tolerance. That is a maintainer question (PR body).
* **Superdroplet collisions scale poorly:** 1.7-1.8× at 4-24 threads. The bucketing,
  `compact!` and the launch over 512 small cells are serial or too fine.
  This path is not on the Berlin default.
* The multi-column radiation reaches 8.4× at 12 threads at the parent
  size (the issue expected 6-10×). The Berlin child's one domain-mean
  column gains through the threaded gas optics (layers × g-points): 7.4×
  at 12 threads.

**Harness.** `AEOLUS_THREAD_TESTS=1`, `AEOLUS_THREAD_COUNTS=1,4,12,24`,
every section (primitives, dns, reductions, smoke, flagship, nested, fft,
radiation): all 2900 hashes, the 34 of the radiation section included,
are identical at 4, 12 and 24 threads to 1 thread.

## KA-j: the step cores (#306)

**What the drivers do now.** `step!` (`_euler_projection!`), `step_rk3!`,
`step_adaptive!`, `step_imex!` and the sound-proof cores of `Anelastic`
keep their serial operator order. Every operator they call is a kernel:
the finite-volume operators of KA-d, the solvers of KA-g / KA-h, and the
field algebra below.

* **One axpy.** The three copies of `_axpy_interior!` (Simulations,
  TimeIntegration, Anelastic) are gone. `Simulations.axpy!(y, a, x)` is
  one `interior_map!`. `_combine2/3/4!` (the IMEX stage sums), `p = φ/Δt`
  (`pressure_from_potential!`) and the `ν + K` freezes of `step_imex!`
  are `interior_map!` / `range_map!` kernels with the expression of the
  broadcast they replace. The tendencies are zeroed with `fill!` and
  copies use `copyto!` (`zero_storage!`, `copy_storage!`): a `memset` /
  `memcpy` beat the pointwise kernel (see *A reset is a memset* below).
* **RK3.** `step_rk3!` copies `uⁿ` into its workspace (`copy_storage!`)
  and forms the stage combinations `¾uⁿ + ¼u` and `⅓uⁿ + ⅔u` as kernels
  over the whole storage (`combine_storage!`; the in-place array is the
  first argument, so the loop vectorises: 0.56-0.68× the broadcast).
  Since SC-d (#404) the combination is written by the projection's
  correction pass and the scalar update, and on the halos by one
  halo-shell launch; `combine_storage!` is gone (see "SC-d" below).
* **Fused CFL sweep.** `cfl_diagnostics` made five sweeps: the advective
  rate and one maximum each of `νh`, `νv`, `κh`, `κv`. It now makes one:
  a `range_max` whose body returns the 5-tuple, each component
  NaN-propagating (`_cfl_maxima`). N² comes from `plane_mean!`.
* **Fused step guard.** After a step, `step_adaptive!` checked
  finiteness (one sweep per field), ran the divergence kernel, took its
  maximum, and took one maximum per velocity component. `step_guard(sim)`
  does all of it in one sweep: per point `(Σ x·0, |∇·u|, max(|u|,|v|,|w|))`
  over `u, v, w, θ, p` (and `qv`). `x·0` is `0` for a finite `x` and NaN
  for NaN or ±Inf, so the first component is 0 exactly when every value
  is finite. `|∇·u|` uses the divergence stencil `_div_at` of
  `FiniteVolume`, the one `max_divergence` uses. The verdict is the old
  one, in the old order: non-finite, then divergence, then the speed cap.
  The guard no longer writes the post-step divergence into `div_buf`.
* **Rollback.** `save_state!` / `restore_state!` copy with
  `copy_storage!`, and `restore_state!` synchronizes the backend (`Backends.synchronize`, a
  no-op on the CPU) before the retry reads the state. The snapshot now
  holds `νt` and `κt` too (#346 item 1): before, a rejected try left the
  NaN coefficients of the blown-up state in `sim.νt` / `sim.κt`, where
  the diagnostics and a crash checkpoint read them.
* **`_all_finite`** is `IO.allfinite` on the interiors, the one
  branch-free finiteness test (#346 item 2).
* **Per-simulation workspaces.** The process-global `WeakKeyDict`s
  `_SNAPSHOTS`, `_IMEX_WORKSPACES` and `_RK3_WORKSPACES` are replaced by
  a `StepWorkspaces` field of each simulation (`BoussinesqSimulation`,
  `MassWeightedSimulation`). Each slot is allocated on first use with
  `Backends.allocate`. A lookup in a shared dictionary raced when two
  domains were stepped from different tasks; a field of the simulation
  does not. `test/ka_step.jl` steps two domains concurrently and compares
  them with the sequential bits.
* **Anelastic.** `weighted_divergence!`, `weighted_momentum_advect!`,
  `weighted_scalar_advect!`, `add_anelastic_buoyancy!` and
  `set_reference_state!` are `launch!` bodies with the z-dependent
  weights as arguments indexed by `k`. `max_*_divergence` and the maxima
  of `adaptive_dt` use `range_max` / `interior_max`.
* **Open boundaries.** The combination and restore of the open-boundary
  buffers in `step_rk3!` and the rollback are kernels.

**Backends.** `range_max` takes a tuple-valued body (several maxima in
one sweep). `range_map!` substituted every argument that is `out` with a
map returning a tuple of `Union`s, which made every `interior_map!` a
dynamic dispatch allocating 128-256 B. It now substitutes only the first
argument, with two concretely typed branches: 0 B per launch, same bits.
One `step!` of the Boussinesq core allocates 0 B on the serial backend
and one `step_rk3!` 16 B (the workspace lookup).

**Bit for bit against the old code.** `test/support/step_cases.jl` runs
one to four steps of every core from seeded states: Euler, RK3,
adaptive (dt sequence, CFL diagnostics, and a forced retry), IMEX ARK2
and Rosenbrock2, anelastic, pseudo-incompressible, a stretched grid,
and every core with open boundaries. It uses only APIs that predate
KA-j, so the same file hashes the old checkout. At one thread, 402
hashes: every prognostic field, `νt`, `κt`, `p`, the dt sequences and
the retry counts are identical to the pre-KA-j code. The 4 that differ
are the scratch buffer `div_buf` after `step_adaptive!` (adaptive,
adaptive retry, stretched adaptive, open adaptive), which the fused
guard no longer writes. `test/ka_step.jl` also checks the ported
kernels against `test/support/legacy_step.jl`, a transcript of the old
loops, on the serial, threshold-0 and default backends.

**One changed expectation.** `logging_blowup.jl` expected a failed
`step_adaptive!` to leave `[:v, :νt, :κt]` non-finite: the NaN injected
in `v`, plus the eddy coefficients of the failed tries, which the
rollback did not restore. The rollback now restores them, so only `v`
is non-finite and the test expects `[:v]`. The code changed on purpose
(#346 item 1); the old expectation described the missing restore.

**Measured: kernels** (`benchmarks/bench_ka_step.jl kernels`, µs,
minimum of 200 runs at 64×64×32 and 30 at 128×128×64; old = the pre-port
loop of `test/support/legacy_step.jl` in the same process; the 1-thread
columns are the minimum of four runs; in brackets old 1 t / new N t;
machine load 4-7 from other jobs; Julia 1.13.1):

| kernel, 64×64×32 | old 1 t | new 1 t | new / old | new 4 t | new 12 t | new 24 t |
| :--- | ------: | ------: | --------: | ------: | -------: | -------: |
| weighted divergence | 48.3 | 48.6 | 1.01 | 24.6 (2.0×) | 27.5 (1.8×) | 27.7 (1.7×) |
| weighted momentum advection | 306.8 | 309.6 | 1.01 | 131.9 (2.3×) | 123.8 (2.5×) | 141.4 (2.2×) |
| weighted scalar advection | 91.6 | 94.5 | 1.03 | 37.9 (2.4×) | 34.6 (2.6×) | 42.9 (2.1×) |
| anelastic buoyancy | 30.3 | 30.5 | 1.01 | 15.4 (2.0×) | 14.8 (2.0×) | 20.9 (1.4×) |
| axpy (interior) | 36.1 | 21.5 | 0.60 | 18.8 (1.9×) | 16.4 (2.2×) | 15.8 (2.3×) |
| combine4 (interior) | 57.0 | 52.7 | 0.92 | 27.9 (2.0×) | 22.9 (2.5×) | 27.2 (2.1×) |
| RK3 combination (storage) | 47.1 | 26.4 | 0.56 | 32.2 (1.5×) | 27.9 (1.7×) | 30.0 (1.6×) |
| CFL maxima (5 sweeps → 1) | 259.1 | 143.0 | 0.55 | 49.5 (5.2×) | 41.2 (6.3×) | 44.0 (5.9×) |
| step guard (11 sweeps → 1) | 631.3 | 157.8 | 0.25 | 148.8 (4.2×) | 109.5 (5.8×) | 143.1 (4.4×) |

| kernel, 128×128×64 | old 1 t | new 1 t | new / old | new 4 t | new 12 t | new 24 t |
| :--- | ------: | ------: | --------: | ------: | -------: | -------: |
| weighted divergence | 850.7 | 813.6 | 0.96 | 194.5 (4.4×) | 188.7 (4.5×) | 194.5 (4.4×) |
| weighted momentum advection | 4284.1 | 4237.1 | 0.99 | 1199.1 (3.6×) | 1355.7 (3.2×) | 1202.3 (3.6×) |
| weighted scalar advection | 1579.8 | 1557.0 | 0.99 | 489.9 (3.2×) | 225.9 (7.0×) | 451.9 (3.5×) |
| anelastic buoyancy | 238.6 | 244.0 | 1.02 | 100.2 (2.4×) | 138.3 (1.7×) | 192.1 (1.2×) |
| axpy (interior) | 256.0 | 159.8 | 0.62 | 74.7 (3.4×) | 137.1 (1.9×) | 108.6 (2.4×) |
| combine4 (interior) | 810.2 | 775.9 | 0.96 | 247.8 (3.3×) | 141.5 (5.7×) | 320.4 (2.5×) |
| RK3 combination (storage) | 305.0 | 180.7 | 0.59 | 171.4 (1.8×) | 277.2 (1.1×) | 235.8 (1.3×) |
| CFL maxima (5 sweeps → 1) | 1854.0 | 1289.5 | 0.70 | 409.3 (4.5×) | 251.4 (7.4×) | 144.1 (12.9×) |
| step guard (11 sweeps → 1) | 9919.3 | 1719.7 | 0.17 | 1063.3 (9.3×) | 1238.7 (8.0×) | 1419.0 (7.0×) |

**Measured: whole steps** (`benchmarks/bench_ka_step.jl steps`, ms per
step, minimum of 5 after 2 warm-up steps, 20 for `adaptive_dt`; old =
the merge base before the KA-j commits, the same script; at one thread
the minimum of three interleaved old/new runs; at 4-24 threads the
minimum of two runs; in brackets new 1 t / new N t). The LES is
64×64×32, Smagorinsky, moist, walls in z. The anelastic case's
density-weighted CG stops at `maxit = 500` (tol 1e-12) in both
versions:

| core | old 1 t | new 1 t | new / old | new 4 t | new 12 t | new 24 t |
| :--- | ------: | ------: | --------: | ------: | -------: | -------: |
| 64³ DNS `step!` | 7.40 | 7.26 | 0.98 | 4.86 (1.49×) | 4.42 (1.64×) | 7.90 (0.92×) |
| LES `step!` | 5.96 | 5.82 | 0.98 | 3.61 (1.61×) | 3.51 (1.66×) | 4.82 (1.21×) |
| LES `step_rk3!` | 18.58 | 18.05 | 0.97 | 12.42 (1.45×) | 12.17 (1.48×) | 17.28 (1.04×) |
| LES `step_adaptive!` | 8.01 | 7.47 | 0.93 | 5.02 (1.49×) | 5.03 (1.49×) | 7.26 (1.03×) |
| LES `adaptive_dt` | 1.23 | 1.11 | 0.90 | 0.60 (1.85×) | 0.61 (1.82×) | 0.59 (1.88×) |
| LES `step_imex!` ARK2 | 50.11 | 49.86 | 1.00 | 23.87 (2.09×) | 22.79 (2.19×) | 27.30 (1.83×) |
| anelastic `step!` | 111.74 | 109.48 | 0.98 | 66.48 (1.65×) | 74.47 (1.47×) | 78.11 (1.40×) |

An earlier, quieter run (load 2-6) of the same cores gave 4.30 ms for
the 64³ DNS and 2.85 ms for the LES `step!` at 12 threads: at 24
threads and under load the numbers move by up to 2×, the 1-thread
ratios by about 5 %.

**Measured: the Berlin flagship** (`bench_berlin.jl`: default
`CaseConfig`, nest off, 64×64×64; ms, minimum of 8 steps after 3
warm-up steps; `step_dt` timed around the call, `dynamics/step!` from
`case.timing`; minimum of two runs, the second one alone for old 12 t):

| | old 1 t | new 1 t | new 4 t | new 12 t | new 24 t | old 12 t |
| :-- | ------: | ------: | ------: | -------: | -------: | -------: |
| `step_dt` | 4.55 | 4.55 | 4.49 | 4.31 | 4.43 | 4.39 |
| `dynamics/step!` | 44.29 | 43.53 | 38.08 | 33.91 | 36.15 | 39.44 |
| `step_case!` | 128.1 | 129.4 | 111.9 | 102.2 | 106.8 | 118.1 |

Reading the tables:

* **The 1-thread gate (≤ 1.10×) holds.** Every core is 0.90-1.00× the
  old code (the largest, 1.00, is ARK2; in the Berlin case
  `dynamics/step!` is 0.98× and `step_case!` 1.01×, run-to-run spread
  about 3 %). Of the kernels, the weighted operators are 0.96-1.03×
  (their bodies are the old loop bodies), the fused CFL sweep
  0.55-0.70× and the fused guard 0.17-0.25×. The largest kernel ratio
  is 1.03 (weighted scalar advection at 64×64×32).
* **Threads.** The cores scale 1.5-2.1× at 4 threads and gain little
  past 4. The issue expected 2-3× at 4 and 3.5-5× at 12 *with KA-k in*.
  Most of a step is still outside this issue. In the Berlin case,
  `step_dt` is 4.3-4.6 ms at every thread count: `compute_sgs!` of
  the `TKEClosure` takes 3.5-3.7 ms of it (KA-l, #304). The fused CFL
  sweep KA-j owns takes 0.30 ms at 1 thread and 0.06 ms at 12 (the old
  five sweeps: 0.25 ms in the LES at 1 t). The flagship dynamics hooks
  are KA-k (#307), and the scalar transport KA-e (#302).
* **A reset is a memset.** The tendency reset and the state copies
  stayed `fill!` / `copyto!`. As pointwise kernels (`range_map!` over
  the storage) they measured, against the serial `fill!` / `copyto!`:
  zero 1.42-1.47× at one thread and 64×64×32 (15.8 against 11.2 µs),
  1.16-1.20× at 128×128×64, and 2.6× at 4 threads (28.4 µs); copy
  1.07-1.15× at one thread, 1.09-1.65× at 64×64×32 on 4-24 threads and
  0.82-1.58× at 128×128×64. A 1-D launch over the linear indices
  measured no better (9.4 / 13.1 / 19.9 µs at 4 / 12 / 24 threads for
  the zero). A `memset` / `memcpy` is memory-bound and already vectorised;
  a GPU array implements both, so nothing is lost for Phase 81.

**Thread invariance.** The worker's `steps` section hashes the same
cases as above (402 hashes). `AEOLUS_THREAD_TESTS=1
AEOLUS_THREAD_COUNTS=1,4,12,24 julia --project=. test/run.jl --only
thread_invariance.jl` with every section (primitives, dns, reductions,
smoke, flagship, nested, fft, fv, pcg, steps, radiation): all 3417
hashes are identical at 4, 12 and 24 threads to 1 thread (21 of 21
tests pass, 17.9 min). The harness's `reductions` section called
`TimeIntegration._max_speed`, which the fused guard replaced; it now
records `step_guard(sim).speed`, the same bits.

## KA-k: the dynamics hooks and the immersed walls (#307)

The `:body_force` and `:post_predictor` hooks of the Berlin case run
inside `step!`. KA-k ports the physics behind them: the facade wall
model (`WallModels`, `wall_models/immersed.jl`), the IBM direct forcing
(`IBM`), the resolved canopy drag (`PlantCanopy`), the Berlin moist
buoyancy (`couplings/moist_buoyancy.jl`) and every large-scale forcing
(`LargeScaleForcing`, which the geostrophic coupling calls). The ground's
wall stress (`post_predictor/surface_exchange`) is not part of it.

**The immersed-wall stress as a gather.** `apply_immersed_wall_stress!`
used to scatter `A[c][I,J,K] += a[n]` over the (face, wall) pairs into
three full work arrays, because a face between two walls (a corner, a
gap one cell wide) has two pairs. `ImmersedWalls` now carries a face
index built once by `immersed_walls`: compressed sparse rows
`face_ptr`/`face_entry`, the faces in the order of their first pair, the
pairs of a face in pair order. Each face sums its own pairs and updates
its own velocity (owner computes), and the per-pair and per-face work
vectors replace the work arrays and the per-call allocation of `a`.
`remove_stencil_wall_flux!` subtracts each face's pair fluxes the same
way. The diagnostics (`F`, `τA`, `area`, `τ_min`, `F_stencil`) are folded
afterwards on the calling task, in the order of the former loops.
`fill_halos!` stays outside the kernels.

**No one-time change.** The port plan named this gather as a one-time
bit change. It is not one: a face's pairs are summed in pair order,
which is the order the scatter added them in, so the gather gives the
former bits. `test/ka_hooks.jl` compares every velocity, tendency and
diagnostic with a transcript of the scatter (`test/support/legacy_hooks.jl`),
also for a shuffled pair order. The harness's `smoke` and `flagship`
sections (2 591 hashes: every field, tracer, bus array, ledger, dt and
solver count of the 12-step smoke matrix and the flagship step) are
identical between master `b94e9484` and this branch at `-t 1`, so the
largest state difference after 12 smoke steps is 0 and the facade
ledger values (`τA`, `F`, `F_stencil`) are unchanged.

**The kernels.**

* *Immersed walls.* The drag of every pair is a 1-D launch over the
  pairs. The face update is two 1-D launches over the faces: the faces
  with one pair (7 634 of 9 717 in the flagship) and those with
  several. The flux removal is one launch whose work item is a block of
  64 consecutive faces walked as one flat loop (a face never splits
  between work items; the block size does not depend on the thread
  count). One work item per face walked through the face index cost
  1.35× the former loop at one thread.
* *IBM direct forcing.* `implicit_ibm_correction!` and `add_ibm_penalty!`
  looped over the solid cells and scaled (or penalised) the six faces of
  each, so a face between two solid cells was written twice from two
  iterations. Now one work item per row `(j, k)` owns the row's `u` faces
  and the `v` and `w` faces on its cells' lower sides. It applies the
  operation of the solid cells in rows `j − 1` and `k − 1` to those
  faces first, then walks its own cells, one mask read per cell: per
  face the same operations in the same order as before. Rows with no
  solid cell in them or below them are skipped after word-wise scans of
  the `BitArray`. Walking each row face by face instead (up to six mask
  reads per face) measured 1.00× in an isolated call but 1.14× inside
  the flagship step (244 against 219 µs per call). The cell-wise walk
  measures 0.76× there.
* *Canopy drag.* `ResolvedCanopyDrag` lists its faces with leaves once
  (`act`, `u` then `v` then `w`, each in `k, j, i` order). The factors
  and the update are two 1-D launches over that list instead of two
  scans of the whole grid. The three full scratch arrays are gone. The
  force is folded per component in the former order.
* *Moist buoyancy.* The level sums were already a canonical
  `plane_reduce!` (KA-b). The deviation is one work item per level (a
  pointwise body that reloaded the level mean at every point cost 3.8×),
  and the `w` forcing is pointwise.
* *Large-scale forcing.* The geostrophic pressure gradient, profile
  nudging, both subsidence forms, the advective tendencies and the
  `:mean` target of `ReferenceDamping` run as `launch_weighted!` over
  the levels (weight `Nx·Ny`). A work item evaluates its level's
  coefficient once and walks the level in the former `for j, for i`
  order. The volume integrals are folded over the levels afterwards, in
  ascending `k`. `ProfileNudging` has a level buffer, so it no longer
  allocates per call.

Every kernel is bit-identical to the loop it replaces, at every thread
count (`test/ka_hooks.jl` on the serial, threshold-0 and default
backends, `--threads 4`, with a `BitArray` and a dense `Array{Bool}`
mask for the IBM).

**Measured.** Same machine as above, Julia 1.13.1, other agents' test
runs on the machine (load 3-7). Isolated calls on the flagship geometry
(`CaseConfig()`, 64³ after 2 steps: 11 984 facade pairs, 6 016 faces
with leaves, 2 870 solid cells) and on synthetic forcing grids. Each
value is the minimum over 200 calls, in µs. Speed-up = old at 1 thread /
new at N threads.

| kernel | old, 1 t | new, 1 t | 1 t / old | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: | --: | --: |
| `remove_stencil_wall_flux!` | 81.4 | 75.9 | 0.93 | 1.08× | 1.09× | 1.07× |
| `apply_immersed_wall_stress!` | 300.5 | 290.8 | 0.97 | 1.06× | 1.04× | 1.01× |
| `apply_canopy_drag!` | 1 008.9 | 98.0 | 0.10 | 10.5× | 10.3× | 10.1× |
| `implicit_ibm_correction!` (τ = 0) | 195.9 | 103.1 | 0.53 | 2.4× | 3.0× | 3.3× |
| moist buoyancy (deviation + `w` forcing) | 1 134.9 | 1 135.3 | 1.00 | 3.7× | 3.7× | 3.8× |
| geostrophic pressure gradient, 64³ | 53.3 | 51.2 | 0.96 | 2.5× | 2.7× | 1.9× |
| profile nudging θ, u, 64³ | 348.0 | 360.7 | 1.04 | 3.3× | 5.5× | 3.7× |
| subsidence (`:local`) θ, 64³ | 15 071 | 154.6 | 0.01 | 328× | 589× | 317× |
| geostrophic pressure gradient, 192²×64 | 911.9 | 929.4 | 1.02 | 2.2× | 4.6× | 2.0× |
| profile nudging θ, u, 192²×64 | 3 662.6 | 3 792.0 | 1.04 | 2.7× | 4.8× | 2.3× |
| subsidence (`:local`) θ, 192²×64 | 165 608 | 1 513.1 | 0.01 | 336× | 931× | 741× |

* **The 1-thread gate holds for every kernel**; the largest ratio is
  1.04 (profile nudging). Three are much faster at one thread. The
  canopy drag works on its 6 016 listed faces instead of 3 × 262 144.
  The IBM skips the rows without solid cells. The `:local` subsidence
  went from 15.1 ms to 0.15 ms per call at 64³ with the same bits; its
  former loop ran about 57 ns per point (not profiled). The parent of
  the Berlin nest uses this form; it is the biggest single gain of the
  PR in the flagship (see below).
* **The immersed walls do not thread at the flagship size.** Their
  launches cover 9 717 faces or 11 984 pairs, fewer than the 32 768
  points of the serial threshold, and a forced launch (threshold 0)
  still gets one task under the 16 384-points-per-task rule (KA-a).
  Forced through the KA path that way, `apply_immersed_wall_stress!`
  costs 1.19-1.22× the serial path and the flux removal 1.01-1.03×,
  for no parallelism. The canopy drag (6 016 faces) is in the same
  position (1.5× when forced). The `launch_weighted!` weight counts
  the points an index covers, not its cost, so it cannot be used to
  thread them. A cost-weighted launch for sparse lists is an open
  question for the maintainer.
* The moist buoyancy is flat beyond 4 threads. Its cost is the
  saturation evaluation inside the `plane_reduce!` of KA-b, not the two
  launches added here.

**Flagship, per hook entry** (`timing_report` lines, ms/step, the
64×64×64 Alexanderplatz run, minimum over 3 × 8 steps after 4 warm-up
steps; old = master `b94e9484`, 3 runs; new at 1 thread, 2 runs):

| entry | old, 1 t | new, 1 t | 1 t / old | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: | --: | --: |
| `body_force/moist_buoyancy` | 1.282 | 1.274 | 0.99 | 0.556 | 0.546 | 0.558 |
| `body_force/geostrophic` | 0.327 | 0.266 | 0.81 | 0.272 | 0.395 | 0.421 |
| `body_force/wall_model_facades` | 0.308 | 0.276 | 0.90 | 0.303 | 0.334 | 0.297 |
| `post_predictor/wall_model_facades` | 0.637 | 0.478 | 0.75 | 0.671 | 0.614 | 0.658 |
| `post_predictor/veg_drag` | 0.614 | 0.161 | 0.26 | 0.234 | 0.220 | 0.256 |
| `post_predictor/ibm` | 0.222 | 0.162 | 0.73 | 0.184 | 0.157 | 0.153 |
| **KA-k entries** | **3.390** | **2.617** | **0.77** | **2.220** | **2.266** | **2.343** |
| `post_predictor/surface_exchange` (not ported here) | 0.379 | 0.352 | | 0.445 | 0.409 | 0.478 |
| `dynamics/step!` | 52.3 | 51.8 | 0.99 | 46.1 | 44.5 | 43.1 |
| step | 249.1 | 217.3 | 0.87 | 191.5 | 188.1 | 188.3 |

* The KA-k entries drop **1.30× at one thread** and **1.50× at 12
  threads** against the former loops. The issue expected at least 3× at
  12 threads, from a profile where the hooks were about 23 % of
  `step!`. On master all hook entries together are 3.8 of 52.3 ms (7 %).
  Of the 2.6 ms the KA-k entries take now at one thread, 0.75 ms are
  the two facade entries, which stay serial (above). Those
  serial entries get slower once Julia runs threaded (0.48 → 0.61-0.67
  ms): the velocities they read were last written by other cores.
  The `geostrophic` entry gets slower at 12 and 24 threads. Most of its
  time is the coupling's own six interior means, for its smoke checks,
  and `add_sponge!`, not the forcing.
* The whole step at one thread drops from 249 to 217 ms because of the
  nest parent (`nest_parent/nesting`: 100 → 73 ms). The parent runs
  `Subsidence(form = :local)` (`parent/case.jl`), the 15 ms → 0.15 ms
  row of the table at its own size.

**Thread invariance.** The harness's workers, run directly in four
groups so each fits one 4.5 GB memory slot (primitives, dns, reductions,
fft, fv, pcg, hooks; smoke; flagship, nested; radiation), at 1, 4, 12
and 24 threads: all 3 055 hashes, the 40 of the `hooks` section
included, are identical at 4, 12 and 24 threads to 1 thread.

That needed one change to the worker. `init_threads!` pins BLAS to one
thread only when Julia runs threaded, so at `-t 1` OpenBLAS kept its own
default thread count. The Berlin radiosity solves through LAPACK, whose
bits depend on that count, so the surface bus (`sw`, `lw`, `Rn`, `H`,
`LE`, `G`, `E`, `Ts`) and the surface and water ledgers of the flagship
step, and from there 2 216 of the 2 539 smoke hashes, differed between
`-t 1` and every other count. It is not new in this branch: the
flagship section gives the same two sets of hashes on master `b94e9484`,
`6412fb9a` and `d5ee77da`, and the same hashes as this branch at 1 and
at 4 threads. With `OPENBLAS_NUM_THREADS=1` the `-t 1` flagship matches
the threaded runs. The worker now calls
`init_threads!(BACKEND; blas_threads = 1)`.

## KA-l: the SGS closures (#304)

Every subgrid closure's coefficient and memory update runs on `launch!`:
the Smagorinsky family (static via KA-d; the dynamic procedure with plane,
local and masked averaging; the Lagrangian scale-invariant and
scale-dependent models), AMD (velocity and scalar), Vreman, the σ-model,
the `StabilityCorrected` split with its N² / Ri_g / S² stencils, and the
prognostic TKE and Reynolds-stress closures of `PrognosticSGS`. The point
stencil `_velocity_gradient_at` / `_strain_at` stays the one shared
`@inline` function every body calls.

**What runs how.**

* *Pointwise bodies* (bit-identical to the old loops): the Germano
  passes (pass 1 over the interior and one shell, `|S|`, pass 2 plain and
  masked), the local coefficient, `ν_t`, the masked velocities (one launch
  over the storage), the mask halo fill (one launch per direction and
  side), the box filter, the Lagrangian coefficient, power law, solid
  zeroing, cell velocity and pathline relaxation, AMD / Vreman / σ, the
  stability correction, N² / Ri / S², the TKE and RSM coefficients, the
  six RSM edge-stress placements, the TKE / RSM initialisations (a
  bisection per cell) and the storage-wide `κ = ν/Prₜ`, `K = 2ν`,
  `κ_v = d κ_h` broadcasts (`range_map!`).
* *Closures captured in kernels are gone.* The `_box_stencil_sum`
  summands are named types (`_GermanoAt`, `_GermanoFluidAt`,
  `_FluidCount`, `_ValAt`, `_MaskedValAt`) summed in the fixed `(dk, dj,
  di)` order; the AMD buoyancy parameter of the sound-proof cores is the
  callable struct `_LevelBuoyancy` (the arithmetic of the closure it
  replaces). The point functions keep their small internal closures over
  isbits locals (`_upstream_n`'s `own()`, the `ntuple` lambdas).
* *The dynamic plane average* runs one level per work item
  (`launch_weighted!` over `k`), its `(j, i)` sums in the old order: bit
  for bit, as the reduction contract foresaw.
* *The solid masks of `_SGSMask` are `Array{Bool,3}`*: the halo fill writes
  neighbouring cells from different work items, which would race on the
  64-bit chunks of a packed `BitArray`.
* *The Lagrangian relaxation is double-buffered*: each point writes its own
  entry of the new memories and reads the old memories upstream; the
  caller copies the new over the old after the launch. With one memory
  pair its six arrays are separate launch arguments (as a tuple argument
  the body ran 1.22× the loop; the KA-d argument rule).
* *The fused updates* (TKE `e`, RSM `τ`) run one plane per work item
  (`plane_map!`): the plane's cells in their loop order, each writing its
  own point and adding its budget contributions to the plane's running
  sums; the planes are combined in ascending `k`. The eight-lane
  `plane_reduce!` of the same body took 1.09× the old TKE loop (eight
  tuple components in eight vector lanes) against 1.02× for `plane_map!`.
* *The RSM's shared 6 × 6 scratch* (`L6`, `A6`, `b6`) is gone: the
  operator, the system matrix and the right-hand side of the per-cell
  backward-Euler solve are `MMatrix` / `MVector` locals, and the Jacobi
  projection uses a 3 × 3 `MMatrix` instead of two heap matrices. Same
  arithmetic, same bits. StaticArrays becomes a direct dependency; it was
  already in the tree through KernelAbstractions, so nothing new is
  installed or loaded (`using Aeolus`, minimum of three fresh processes:
  606.8 ms before, 604.5 ms after).
* *The TKE / RSM advection* passes the reconstruction built once in the
  constructor instead of dispatching on the scheme's `Symbol` every call
  (maintainer decision on #291).

**The second test filter** (the 5 × 5 × 5 box of the scale-dependent
model) reused one-plane scratch buffers across the planes of a serial
loop. It now runs the x and y passes per shell plane into the full 3-D
buffer `C` and then the z pass and the identity per interior plane. A first
version with a second full 3-D buffer for the x pass (one plane per work
item, 26.5 MB at 64×64×32) cost 1.22× the old loop at one thread, almost
all of it writing the fresh 26 MB every call. The planes now go to at most
12 work items (a constant, not the thread count); each work item owns one
slot of the scratch `B` and reuses it for its planes, so `B` stays in cache,
and a stencil entirely inside the domain is one fused left-to-right sum
(the same additions as assign-and-add, one store). The split never changes
a bit (tested with one and several planes per work item, and with rows too
short for the fused sum). At 64×64×32, one thread: 10.8 ms against
11.6 ms for the old loop; 12 work items also beat one plane per work item
at 4, 12 and 24 threads (2.2-3.3 ms against 3.2-3.8 ms). Memory: `B`
9.3 MB (12 slots) + `C` 24.5 MB = 33.9 MB, against 26.7 MB before. The
per-call stencil vectors are gone (`_box5_src`).

**The dual strain** of the TKE closure (the exact discrete dual of the
Laplacian SGS momentum operator) was a scatter: each velocity point added
its flux shares to up to four cells, then the first halo layer was folded
onto the interior. It is now an owner-computes gather: each interior cell
sums the 18 shares per velocity component destined for it (the inverse of
the scatter's target lists; every share is the scatter's expression, bit
for bit) and those destined for the halo cells it is the image of. Bulk
cells skip the range checks. One body serves the RegularGrid and the
StretchedGrid. The scatter's own halo fold stays for the wall-model
production, a scatter over the list of wall faces.

**Named one-time changes** (against the pre-port loops):

* *The dual strain*: the same shares in another order. On random fields
  the gather differs from the scatter by at most 6.2e-16 of max |D|
  (64×64×32), and the exact duality `Σ ν D V = −Σ u·T(ν) V` still holds to
  1e-12. After 12 steps of a 64×64×32 TKE LES the largest change of `e` is
  4.4e-15 m² s⁻² (1.7e-15 relative). After 12 steps of the Berlin flagship
  (`CaseConfig()`, 64³, default `:tke` closure) the largest change of `e` is
  4.6e-10 m² s⁻² (2.1e-12 relative to max e = 220 m² s⁻²), of `u` 8.2e-12
  m s⁻¹ (3.4e-13 relative), of θ 1.8e-11 K; the domain means agree to all
  printed digits (e 6.6662e-2 m² s⁻²). The `:smoke` scenario (12 steps,
  the state digests of `test/data/config_berlin/fixture.toml`): 13 of 22
  digests change (u, v, w, p, θ, qv, e, Ts and the chemistry tracers; the
  nest parent's do not), by at most 6.8e-14 relative (e: max |Δe| 2.0e-15
  m² s⁻² at max e 2.9e-2) and 3.9e-16 (θ); with the old scatter swapped
  back in, all 22 equal the fixture (as re-recorded by KA-m), so the
  gather is the only change. The 13 digests are re-captured.
* *The TKE and RSM budget statistics* (`tke_statistics`,
  `reynolds_stress_statistics`): per-plane running sums combined in
  ascending `k` instead of one running sum over the domain. Equal to
  1e-12 relative in `test/ka_sgs.jl`; the counters are exact.

Nothing else changes: compared across versions (the harness's `sgs`
section run on the old and the new code at one thread), every hash of the
static, dynamic (plane, local, masked), AMD (and its tracer diffusivity),
Vreman, σ, stability-corrected and Lagrangian (scale-invariant,
scale-dependent, masked: coefficients and memories) closures, the masked box
filter and the 3-step dynamic LES is identical; the 34 hashes that differ
are the dual strain and what is built from it (the TKE and RSM
initialisations and states, the TKE LES) and the two closures' statistics.

**Tests.** `test/ka_sgs.jl` compares every ported kernel with a
transcript of the pre-port loops (`test/support/legacy_sgs.jl`) on random
fields with random halos, masked and not, periodic and bounded, on the
serial, threshold-0 and default backends (418 checks): bit for bit, except
the two named changes above (the fused updates' fields bit for bit and
their sums to 1e-12; the gather to 64 ε of max |D| plus its exact
duality). The second test filter allocates nothing per call. The harness
has an `sgs` section (109 hashes): every closure's coefficients on a seeded
64×64×32 state, 3 commits of every stateful closure with its statistics,
the dual strain, the masked box filter, the initialisations, and 3 LES
steps with the dynamic and with the TKE closure. `AEOLUS_THREAD_TESTS=1`,
`AEOLUS_THREAD_COUNTS=1,4,12,24`, every section (primitives, dns,
reductions, smoke, flagship, nested, fft, fv, pcg, multigrid, steps,
radiation, hooks, columns, advection, sgs; after merging KA-e, KA-i and
KA-m): all 3692 hashes are identical at 4, 12 and 24 threads to 1 thread
(21 of 21 checks pass).

**Measurements** (`compute_sgs!` / `advance_sgs!` per closure on a seeded
64×64×32 LES state, minimum of up to 40 calls, the best of four processes
per entry; the old loops are today's master, the speed-ups are against the
old loop at one thread; the machine carried other jobs at load 2-11, so the
4-24-thread columns are lower bounds). Gate: every kernel at one thread is
at most 1.04× the old loop (`advance_sgs!` of the Lagrangian model, its
relaxation's double buffer); the scale-dependent coefficient over (i, j, k)
cost 1.20× (2.82 ms against 2.35 ms) and now runs over one linear index
(1.00×). The RSM coefficients take 0.13× (the old Jacobi projection's two heap
matrices are gone), the dual-strain gather 0.60×.

| kernel (64×64×32) | old loop, 1 t (µs) | KA, 1 t (µs) | 1 t / old | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: | --: | --: |
| `compute_sgs! static smagorinsky` | 888.0 | 861.8 | 0.97 | 2.9× | 3.0× | 2.8× |
| `compute_sgs! dynamic plane` | 9680.3 | 9292.2 | 0.96 | 3.8× | 5.3× | 4.9× |
| `compute_sgs! dynamic local` | 9574.7 | 9299.9 | 0.97 | 3.9× | 5.3× | 4.8× |
| `compute_sgs! dynamic plane masked` | 13865.3 | 11693.5 | 0.84 | 4.2× | 5.8× | 5.3× |
| `compute_sgs! lagrangian` | 1052.9 | 891.2 | 0.85 | 3.2× | 2.8× | 2.9× |
| `advance_sgs! lagrangian` | 18299.5 | 19015.5 | 1.04 | 3.5× | 5.1× | 4.9× |
| `advance_sgs! lagrangian masked` | 25629.4 | 22965.6 | 0.90 | 4.0× | 6.0× | 6.1× |
| `compute_sgs! LASD` | 2999.6 | 2986.4 | 1.00 | 3.5× | 4.5× | 3.4× |
| `advance_sgs! LASD` | 37097.4 | 34825.0 | 0.94 | 3.3× | 4.5× | 4.7× |
| `compute_sgs! AMD` | 395.1 | 385.1 | 0.97 | 2.4× | 2.6× | 2.5× |
| `compute_sgs! Vreman` | 1463.0 | 1472.5 | 1.01 | 3.3× | 3.8× | 3.5× |
| `compute_sgs! sigma` | 6623.0 | 6472.9 | 0.98 | 3.8× | 5.9× | 5.6× |
| `compute_sgs! stability-corrected smag` | 3484.1 | 3279.2 | 0.94 | 3.5× | 3.5× | 3.1× |
| `compute_sgs! stability-corrected dynamic` | 12124.1 | 11895.6 | 0.98 | 3.6× | 4.9× | 4.1× |
| `compute_sgs! TKE` | 235.0 | 238.1 | 1.01 | 1.6× | 1.3× | 1.2× |
| `advance_sgs! TKE` | 11840.0 | 9848.5 | 0.83 | 1.8× | 1.9× | 1.9× |
| `initialize_tke!` | 149073.4 | 147300.0 | 0.99 | 4.0× | 7.8× | 7.8× |
| `dual_strain!` | 4392.4 | 2627.9 | 0.60 | 5.6× | 7.4× | 7.3× |
| `compute_sgs! RSM` | 16373.0 | 2193.3 | 0.13 | 14.8× | 14.1× | 11.4× |
| `advance_sgs! RSM` | 62091.0 | 56467.1 | 0.91 | 1.9× | 2.1× | 2.1× |

The thread speed-ups are those of a 64×64×32 domain: the Germano and
Lagrangian filters, memory-bound, reach 4.5-6.0× at 12 threads (expected
3-5×); the σ-model 5.9× and the TKE initialisation 7.8× (compute-bound,
expected 6-9×); the small pointwise closures (AMD 0.39 ms, `compute_sgs!`
of TKE 0.24 ms) are dominated by their halo fills. The fused TKE and RSM
updates run one plane per work item over 32 planes and stop at about 2×:
the advection and the six-component solve they call are not part of this
issue. Berlin flagship (`CaseConfig()`, 64³, `:tke`, 2 warm-up steps,
best of three 3-step runs, s/step): old 13.60 / 12.43 / 12.15 / 12.04, new
14.14 / 11.07 / 10.42 / 10.07 at 1 / 4 / 12 / 24 threads (one process each,
on the loaded machine: 1.04× at one thread, not separated from the noise
by a single sample; the SGS kernels themselves measure 0.13-1.04× above).
A 32-step run (2 warm-up + 3 × 10) of `CaseConfig()` fails on both the old
and the new code (`step_adaptive!: step failed (velocity) after 3
retries`), so the flagship is timed over 11 steps.

## KA-m: column physics, one work item per column (#301)

**Primitives** (`src/Backends.jl`). [`foreach_column!`](@ref Aeolus.Backends.foreach_column!)
runs a column body `body!(i, j, args...)` (or `body!(n, args...)` over a
list of ids: the columns of one surface type, the facets of a scene) as
one work item. The body loops over its own levels in its own order and
writes only its own column. What the old loops accumulated across
columns, the body *returns*; the launch stores it in row `c` of a
`partials` matrix (column-major `c`), and
[`column_sums`](@ref Aeolus.Backends.column_sums) folds each term in
row order from zero. [`column_sum`](@ref Aeolus.Backends.column_sum) does
both, and on the calling task it adds the terms as they come (no partials
array). Both paths add the same numbers in the same order, so a total is
identical at every thread count and equal to the old `s += x` loop. A
maximum is `foldl(max, …)` over the partials (exact in any order).
[`foreach_cell!`](@ref Aeolus.Backends.foreach_cell!) is `launch!` with a
weight, or `plane_reduce!` with per-plane partials. `weight` is the work
of one column in grid points (its levels, or its root solves), so that
the serial threshold counts the real work.

**What runs per column.**

* Surface exchange (`step_all_surfaces!`): the columns are grouped by
  concrete surface type at `init!` and each group is one launch, so every
  launch calls `step_surface!` on one type. The worst storage error, the
  worst MSHF ratios and the MSHF time are per-column partials; the
  delivery loop (`:surface_flux`) returns evaporation and the solid-write
  flag.
* The parent's tiled surface (`step_parent_surface!`): one work item per
  parent column, with the area sums of the tile fluxes as partials.
* `aq_chem`: the deposition velocities (one surface-layer solve per
  column) and the emissions; `Chemistry.apply_dry_deposition!` returns
  what each column removed. `apply_photochemistry!` was already a
  per-cell `plane_reduce!` (KA-b).
* `RadiationFog.fog_adjust!`: one plane per work item, with an early exit
  for a cell with `ql ≤ 0` and `qv < (1 − 10⁻⁸) q_s`, where the
  adjustment returns exactly zero. `dew_exchange!` and the Berlin dew
  pass: one column each.
* Kessler: conversion per cell; the spectrum sedimentation one column
  per work item, a raining column allocating its own `col` / `Nr` scratch
  (the old loop shared one pair through a captured closure); the legacy
  upwind one column per work item in ascending `k`. This is also the only
  3-D use of `p3_sediment!`; the Berlin case has no 3-D P3 coupling.
* `WallModels` dynamic roughness: the masked mean wind, the drag parts
  and the residual at every α of the scan and the root solve.
* `UrbanSurface` facet scenes: the trial steps of the implicit step, the
  final column advance and the lagged step, one work item per facet. A
  scene that lists one facet object twice is refused (it would race).

**What stays serial, and why.** The facet scene's coupled longwave /
node solve between the facet launches and the building energy balance
are global linear systems: every unknown depends on every other, so
there is no column to give a work item. The LU solve also depends on the
BLAS thread count, which is why the harness section pins BLAS to one
thread. Per-tracer parallelism is a follow-up.

**Global caches.** `RadiationCore._gauss_legendre` is the one source of
Gauss-Legendre nodes; `SurfaceOptics` and `Radiosity` delegate to it
(their copies were the same arithmetic and gave the same bits). Orders
1-64 are a table computed when the module loads; higher orders are
memoised under a lock. The aerosol bulk-optics memo and the SNICAR table
are locked (the SNICAR file is read once). The Mie log-derivative buffer
was one global vector, and two tasks calling `mie_sphere` at once
corrupted each other's recursion. The new first-use-under-threads test
found this; the buffer is task-local now (2000 calls: 4.39 → 4.02 ms).

**Two performance traps** found while measuring, both now fixed and
tested:

* A closure that captures a variable the loop reassigns is boxed. The
  first `column_sums` row update did this, and every row then went
  through dynamic dispatch (72 bytes per row). `update_dynamic_roughness!`
  at 128² took 101 / 98 / 97 ms at 4 / 12 / 24 threads against 8.3 ms
  for the old loop at one thread.
* A type passed among the launch arguments (`::Type{T}`) is stored as a
  `DataType` in the argument tuple, so the body is dispatched per
  column. Pass `zero(T)` instead.

**One-time changes against master** (named in the maintainer's
decisions on #291):

* The saturation vapour pressure is one fused exponential,
  `e_s = p_tr exp(a ln(T/T0) + b)`, instead of `p_tr (T/T0)^a exp(b)`.
  This changes the last bits of `e_s` and `q_s` everywhere: at most
  8.3 × 10⁻¹⁶ relative over 230-320 K (6 ulps) and 2.4 × 10⁻¹⁵ over
  150-400 K. The P3 precipitation-fraction golden values move by
  9.5 × 10⁻¹⁶ in `q_c` of the cold parcel; they were re-recorded because
  that expectation was a bit pattern.
  After 12 flagship steps at one thread against master: θ differs by at
  most 3.4 × 10⁻¹³ K (1.2 × 10⁻¹⁵ relative), qv by 1.4 × 10⁻¹⁷
  (1.7 × 10⁻¹⁵), u by 6.2 × 10⁻¹⁵ m s⁻¹, the surface temperature by
  2.3 × 10⁻¹³ K; ql is identical. In the ledgers the park class moves in
  the 14th digit, and the domain evaporation goes from
  1.4646778140248087 × 10⁻⁴ to 1.4646778140248762 × 10⁻⁴.
  The CFG fixture's smoke digests (`test/data/config_berlin/fixture.toml`,
  `berlin_smoke.toml`, 12 steps) were re-recorded for this change: the 13
  child fields move (θ by at most 1.1 × 10⁻¹³ K, 3.9 × 10⁻¹⁶ relative; qv
  2.8 × 10⁻¹⁷; u 6.1 × 10⁻¹⁵ m s⁻¹; Ts 1.1 × 10⁻¹³ K; O₃ 1.5 × 10⁻¹⁴
  relative), the nest parent and the zero tracers do not. With the old
  `e_s` restored the branch reproduces the old digests exactly, so the
  `e_s` change is the only difference.
* The total that `fog_adjust!` returns is the plane sums in ascending
  `k`, so its last bits change. The Berlin coupling ignores it, and the
  fields are bit-identical.

Apart from these, every ported kernel is bit-identical to master at one
thread. This was checked by hashing in two worktrees: facet scenes
(implicit, direct, lagged), dry deposition, dynamic roughness, the
smoke matrix and the flagship, 2591 hashes. The `dew_exchange!` hash
equals the hash of master with the `e_s` change alone.

**Measured** (AMD Ryzen 9 7900, Julia 1.13.1; minimum of 20 calls,
5-10 for the slow ones; old = master at b4f6f82, the minimum of two runs
at one thread; speed-up = old 1 t / new N t; µs).

| kernel | old, 1 t | new, 1 t | 1 t / old | 4 t | 12 t | 24 t |
| :----- | -------: | -------: | --------: | --: | ---: | ---: |
| `fog_adjust!`, flagship state, 64×64×32 | 11 350 | 4 968 | 0.44 | 9.04× | 15.93× | 17.97× |
| `fog_adjust!`, foggy state, 64×64×32 | 49 276 | 43 523 | 0.88 | 4.61× | 11.48× | 14.52× |
| `fog_dew` dew pass, 64×64 | 33.5 | 32.2 | 0.96 | 2.01× | 1.32× | 1.25× |
| `apply_kessler_step!` (spectrum), flagship state | 677 | 644 | 0.95 | 3.34× | 4.00× | 3.55× |
| `apply_kessler_step!` (spectrum), 20 % raining cells | 3 242 882 | 3 342 466 | 1.03 | 3.77× | 6.97× | 10.56× |
| `step_all_surfaces!`, child 64×64 | 146 241 | 140 142 | 0.96 | 4.04× | 5.42× | 11.51× |
| `step_parent_surface!`, parent 192×192 (tiled) | 1 987 205 | 1 993 987 | 1.00 | 3.38× | 5.28× | 6.24× |
| `aq_chem` deposition velocities, 64×64 | 24 336 | 21 086 | 0.87 | 4.37× | 7.45× | 7.98× |
| `aq_chem` emissions, 64×64 | 5.11 | 5.32 | 1.04 | 0.96× | 0.86× | 0.85× |
| `apply_dry_deposition!` (O₃), 64×64 | 30.9 | 23.7 | 0.77 | 1.29× | 1.29× | 1.28× |
| surface delivery (`:surface_flux`), 64×64 | 184 | 22.9 | 0.12 | 7.92× | 7.95× | 7.88× |
| `step_urban_facets!` implicit, 564 facets | 9 587 | 9 646 | 1.01 | 3.47× | 3.73× | 4.21× |
| `step_urban_facets!` lagged, 564 facets | 9 586 | 9 854 | 1.03 | 3.67× | 4.21× | 4.66× |
| `update_dynamic_roughness!`, 128×128 | 8 336 | 8 708 | 1.04 | 2.20× | 2.53× | 2.46× |

Re-checked at one thread after merging KA-d (#299) and the CFG changes
(single run each): every ratio is between 0.12 and 1.09. The largest are
the lagged facet step (9 581 → 10 468 µs, 1.09; 1.03 above, so it is
within run-to-run noise but close to the gate), the emissions (1.06) and
the dynamic roughness (1.04).
Re-checked again after merging KA-g (#300) and KA-n (#305), single run
against the same old numbers: 0.12-1.06 for every kernel but the lagged
facet step, which read 1.11 (5 calls). Measured on its own, old and new
alternating, three processes each, the minimum of 60 calls: old
9 662-9 855 µs, new 9 550-9 882 µs (0.97-1.02). The 1.11 was the 5-call
minimum under a machine load of 5-7 from other jobs.

Reading the table:

* **The 1-thread gate (≤ 1.10) holds for every kernel.** The `aq_chem`
  emissions first missed it at 1.38-1.50× (5.0 → 7.5 µs): through
  `column_sum` every column adds a 3-tuple, zeros included, where the
  old loop added only the one or two terms a street or park column
  emits (3265 of 4096 columns emit). The site now runs its own serial
  loop whenever the launch would run on the calling task, adding only
  the emitted terms, and launches `column_sum` otherwise; both classify
  a column through one helper, `_aq_emit_source`. Adding a +0.0 term to
  a sum that starts at +0.0 changes nothing, so the two paths give the
  same bits (the harness record `columns/aq_emit` runs the serial loop
  at one thread and the threaded launch at more). Measured on the
  flagship state, 3 repeats of the minimum of 20 000 calls: 5.11-5.24 →
  5.32-5.39 µs, 1.02-1.05×. The emission row's 4-24 thread columns are
  serial too (4096 points are below the threshold), and the spread
  there is machine load.
* **`fog_adjust!` gains at one thread** (0.44×) from the early exit,
  because most flagship cells are far from saturation. A foggy state
  (30 % of cells supersaturated) gains 0.88×. Both reach 15-18× at 24
  threads.
* **The surface delivery** is 0.12× at one thread. The old loop ran
  inside the stage method, which reads the case's fields, and some of
  them are not concretely typed. The column body sits behind
  `column_sum`, which acts as a function barrier, so it is compiled for
  concrete types. That is the likely reason; it was not profiled.
* **Compute-heavy columns scale.** The child surface step reaches
  11.5× at 24 threads, the parent's 192² tiled surface 6.2× (5.3× at
  12), the deposition velocities 8.0×, and Kessler with 20 % rain 10.6×.
  The issue expected 6-10× at 12 threads for the surface exchange; the
  child gets 5.4× at 12 and the parent 5.3×, limited by load imbalance
  between surface types (one launch per type).
* **Small 2-D sums stay serial.** The emissions and the dry deposition
  are 4096 columns × 1 point, below the threshold, so they run on the
  calling task at any thread count. The dew pass has weight 16 (a
  saturation humidity per column) and does thread. It reaches 2.0× at 4 threads, but only 1.3× at
  12 and 24, where task start-up costs about as much as the 33 µs of
  work.
* **The facet scene** (564 facets) reaches 4.2-4.7× at 24 threads. The
  serial longwave solve between the facet launches bounds it.
* **The dynamic roughness** gains 2.2-2.5×: the filtered velocities and
  the scan's bookkeeping between the residual launches are serial.

**The flagship** (64×64×32 child with its parent, default cadence;
`case.timing` per step in ms, mean over 8 steps, minimum of 3 repeats;
old = master at one thread):

| stage | old, 1 t | new, 1 t | 4 t | 12 t | 24 t |
| :---- | -------: | -------: | --: | ---: | ---: |
| whole step | 227.9 | 220.1 | 196.2 | 197.4 | 211.6 |
| `chemistry/aq_chem` | 12.78 | 12.88 | 4.24 | 2.68 | 2.45 |
| `fog_dew/fog_dew` | 11.43 | 4.83 | 1.64 | 0.99 | 0.86 |
| `microphysics/micro_kessler` | 1.12 | 1.05 | 0.62 | 0.59 | 0.78 |
| `surface_flux/surface_exchange` | 0.207 | 0.043 | 0.059 | 0.076 | 0.070 |
| `surface_update/surface_exchange` | 0.106 | 0.107 | 0.115 | 0.173 | 0.139 |

The issue's baseline was aq_chem 15-16.5 ms and fog_dew 9.4 ms. These
stages now take 2.5 and 0.9 ms at 24 threads. The surface step runs at
its cadence, so its mean per step is small; per call it is the
`step_all_surfaces!` row above. The step as a whole is still bound by
stages that other port issues own: the parent's nesting step (92 ms),
scalar advection (49 ms) and the dynamics (48 ms).

**Thread invariance.** The harness (`AEOLUS_THREAD_TESTS=1`,
`AEOLUS_THREAD_COUNTS=1,4,12,24`) gives 3027 hashes per process
(primitives, 64³ DNS, reductions, the Berlin smoke matrix, the flagship
step, nested runs, fft, fv, pcg, radiation and the new `columns` section;
measured after merging KA-g and KA-n), and all of them at 4, 12 and 24
threads equal the 1-thread hashes. `test/ka_columns.jl` at
`--threads 24` includes the first use of every cache from hundreds of
tasks at once, and passes.

## KA-o: the scaling report (#308)

The whole port (KA-a to KA-n) measured end to end with
`benchmarks/bench_ka_scaling.jl`: one process per thread count, 1, 4, 12
and 24 threads (12 cores, 24 with SMT). Same machine as above, Julia
1.13.1, BLAS on one thread. Each run started only when the 1-minute load
was below 4 (below 6 for the 1- and 4-thread runs); other agents' jobs
still ran on the machine, so every figure is the minimum over two to four
processes. Speed-up is the new code at 1 thread over the new code at N
threads (the 1-thread gate against the pre-port loops was each port PR's
job, above).

**What was run, and what was not.** The issue asked for 300 warm-up and
300 timed flagship steps. The default case fails near step 12 (#392) and
long Berlin runs are paused, so the flagship is timed over steps 3-11
(2 warm-up steps, 9 timed). In those 9 steps the nest opens one parent
window, so the per-step mean carries one parent window per nine child
steps (child steps of 0.34 s on average; in a long run the ratio follows
the two steps). The flagship is `CaseConfig(:summer)`
as shipped: a 64 × 64 × 64 child (not the 64 × 64 × 32 of the issue; the
default was deepened for the nest, #246) nested in its 192 × 192 × 53
parent, MEASURE FFTW plans. The 256³ DNS is timed per step (1 warm-up,
3 timed) and scaled to the 1264 steps of the measured run
(`test/fixtures/dns256_apriori.toml`). The kernels and the DNS use
ESTIMATE plans: MEASURE planning of a threaded 256³ c2r transform
segfaults intermittently at 10-14 threads, in FFTW itself (#397).

### Results against the targets

| | target | 1 t | 4 t | 12 t | 24 t | met |
| :-- | :-- | --: | --: | --: | --: | :-- |
| flagship `step_case!`, ms/step (mean of 9) | ≥ 2.5× at 4, ≥ 4× at 12 | 155.2 | 90.7 (1.71×) | 87.5 (1.77×) | 97.0 (1.60×) | no |
| flagship, 24 t against 12 t | within ±15 % | | | | +10.8 % | yes |
| child alone (nest off), ms/step | | 100.3 | 46.8 (2.14×) | 46.5 (2.16×) | 56.4 (1.78×) | |
| 256³ DNS `dns_step!`, s/step | ≥ 2.5× at 4, ≥ 4× at 12 | 3.28 | 2.30 (1.43×) | 2.29 (1.43×) | 2.25 (1.46×) | no |
| 256³ DNS `step_rk3!` alone, s/step | | 2.21 | 1.24 (1.78×) | 1.18 (1.87×) | 1.19 (1.85×) | |
| 256³ DNS, 1264 steps | 2.05 h → ≈ 30 min | 1.15 h | 48 min | 48 min | 47 min | no |

The DNS took 2.05 h single-core when the issue was written and 5671 s
(1.58 h) in the run of 2026-10-05 (#235). On today's code it is 1.15 h at
one thread and 48 min at 12 threads: 2.6× against the issue's figure, of
which 1.8× is one-thread gains since the issue was written (the port's
serial fixes among them; the timings here use ESTIMATE plans, which are
not faster) and 1.4× is threads.

Neither target is met. The issue estimated 85-90 % of the flagship step
to be parallel. Fitting Amdahl's law to the measured step gives a serial
fraction of 45 % at 4 threads and 53 % at 12. Most of that is not serial
code: the parts below show it is parallel code that stops at 1.5-2×,
because it is bandwidth-bound or its launches are too small.

### Where the flagship step goes

`case.timing` per entry, ms per step (mean of the 9 timed steps, minimum
of four processes). `nest_parent` is the parent's steps (one window of
469 ms at 1 t, 316 ms at 12 t, spread over 9 child steps). The
`:body_force` and `:post_predictor` entries run inside `dynamics/step!`.

| entry | 1 t | 4 t | 12 t | 24 t | 12 t speed-up |
| :-- | --: | --: | --: | --: | --: |
| **whole step** (`step_case!`) | 155.2 | 90.7 | 87.5 | 97.0 | 1.77× |
| `step_dt` (timed around the call) | 4.28 | 2.45 | 2.60 | 2.83 | 1.65× |
| `nest_parent/nesting` (the parent) | 52.2 | 35.5 | 35.2 | 34.9 | 1.48× |
| `transport/scalar_advection` | 41.0 | 20.9 | 19.7 | 25.4 | 2.08× |
| `dynamics/step!` | 35.5 | 21.6 | 20.1 | 24.2 | 1.76× |
| `chemistry/aq_chem` | 12.96 | 4.14 | 2.60 | 2.57 | 4.98× |
| `fog_dew/fog_dew` | 4.81 | 1.31 | 0.93 | 0.74 | 5.19× |
| `tracers/pollutant` | 2.21 | 1.45 | 1.53 | 1.79 | 1.45× |
| `nest_feedback/nesting` | 1.93 | 1.99 | 2.03 | 1.94 | 0.95× |
| `ledger/air_heat` + `ledger/ledger` | 2.33 | 1.69 | 1.90 | 2.66 | 1.23× |
| `body_force/moist_buoyancy` | 1.20 | 0.42 | 0.41 | 0.49 | 2.93× |
| `microphysics/micro_kessler` | 1.03 | 0.55 | 0.53 | 0.69 | 1.95× |
| `chemistry/aq_aerosol` | 1.03 | 1.08 | 1.08 | 1.18 | 0.95× |
| facade wall model (both hooks) | 0.66 | 0.74 | 0.80 | 0.89 | 0.83× |
| thirteen more entries under 0.35 ms each | 1.32 | 1.42 | 1.77 | 1.97 | 0.75× |

* **Three entries hold 83 % of the one-thread step** (the parent, scalar
  transport, the child's dynamics: 129 of 155 ms) and reach 1.5-2.1×.
  They are made of stencils at 64³ (child) and 192² × 53 (parent), the
  sizes where the idealised stencils below stop at 2-3× (memory-bound),
  with fill-halo and boundary work between them that does not thread at
  all (`fill_halos!` 1.0× at 64 × 64 × 32). The parent's steps are its
  dynamics (457 of 462 ms per window at 1 t; its surface takes 1.4 ms and
  its radiation 0.9 ms). Per-tracer parallelism of the transport was
  deferred to a follow-up on #291.
* **The column physics scales** (aq_chem 5.0×, fog_dew 5.2-6.5×, moist
  buoyancy 2.9×): one work item per column, compute-bound. It is 12 % of
  the one-thread step.
* **The serial remainder is small.** The entries that do not speed up at
  all (nest feedback, aq_aerosol, the wall models, IBM, the surface
  exchange hooks, glass, sw_modifiers, the radiation heating) take 5.7 ms
  at 12 threads, 6.5 % of the step; several get slower with threads (task
  start-up on launches of a few thousand points). The issue's candidates:
  * *stage driver*: `step_case!` minus its entries is −0.2 to −0.6 ms,
    i.e. zero within the timer overlap (the air-heat ledger of the hooks
    inside `step!` is booked twice);
  * *iterative radiosity*: every 600 s of model time (300-1800 child
    steps), 2.1× at 4-24 threads by day and 110× at night against the old
    code (KA-n, above); not in the per-step mean;
  * *BuildingEnergy facet solves* (`buildings_citymean`): 0.02 ms per step,
    0.2-0.3 ms per radiation call, flat;
  * *single-column radiation*: its heating (`column_tendency`) is 0.2-0.3
    ms per step; the column solve every 600 s takes 45.4 / 12.5 / 7.3 /
    5.0 ms per call at 1 / 4 / 12 / 24 threads (6.2× at 12, threaded gas
    optics).
* **KA launch overhead.** 263 launches per flagship step run threaded and
  569 run serially (below the 32 768-point threshold; counted at
  `runs_serially`, `launches` mode). An empty threaded launch costs
  1.37-1.42 µs with one task and 2.6-9.9 µs with sixteen, at any thread
  count; so launch overhead is 0.4-2.6 ms of the 87.5 ms step at 12
  threads, under 3 %.

### Memory-bound and compute-bound kernels

The idealised kernels on seeded fields (`kernels` mode; µs, minimum over
repeats; the bandwidth counts 8 B per point for every array read or
written, so it ignores halos and caches):

| kernel | 64×64×32, 1 t | 4 t | 12 t | 24 t | 256³, 1 t | 4 t | 12 t | 24 t | 256³ GB/s at 12 t | bound |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- |
| axpy | 22.3 | 1.69× | 1.76× | 1.40× | 8 468 | 1.14× | 1.06× | 1.06× | 50.6 | memory |
| `set!` value | 10.4 | 0.91× | 0.88× | 0.66× | 4 673 | 1.11× | 1.00× | 0.98× | 28.8 | memory |
| `interior_sum` | 9.4 | 1.22× | 1.20× | 0.69× | 2 593 | 1.31× | 1.44× | 1.13× | 74.3 | memory |
| `laplacian` | 44.4 | 2.03× | 2.10× | 1.78× | 13 892 | 1.78× | 1.65× | 1.49× | 31.7 | memory |
| `advect_centered!` | 88.2 | 2.49× | 3.17× | 2.20× | 29 226 | 1.75× | 1.62× | 1.33× | 44.7 | memory |
| `momentum_advect!` | 295.8 | 2.48× | 2.98× | 2.21× | 84 891 | 1.95× | 1.80× | 1.48× | 25.7 | memory |
| `fill_halos!` (4 fields) | 41.9 | 1.00× | 1.01× | 0.99× | 4 637 | 1.67× | 1.89× | 1.71× | | serial / memory |
| Smagorinsky νₜ | 725.5 | 3.81× | 6.57× | 4.74× | 103 492 | 3.78× | 5.16× | 4.26× | 24.6 | mixed |
| FFT pressure `solve!` | 499.3 | 2.71× | 1.58× | 1.39× | 334 755 | 2.77× | 3.99× | 4.43× | | mixed |
| WENO-Z5 `advect_weno_z!` | 2 188 | 3.93× | 5.64× | 5.64× | 325 747 | 4.58× | 8.57× | 9.50× | 20.9 | compute |
| `set!` sin·exp | 1 056 | 4.19× | 7.25× | 5.30× | 123 104 | 4.25× | 8.07× | 10.08× | 8.8 | compute |

* **Memory-bound**: the streaming kernels reach 50-75 GB/s at 256³ at any
  thread count, the DRAM bandwidth of this dual-channel DDR5 desktop
  (the axpy alone streams 47.6 GB/s on one thread). They gain 1.1-1.9× from threads
  at 256³ and 1.7-3.2× at 64 × 64 × 32, where the arrays sit in L3. The
  plan's Amdahl estimate assumed more bandwidth per core.
* **Compute-bound**: WENO-Z5 and transcendental functions reach 8-10× at
  256³ (5-7× at 64 × 64 × 32, where task start-up costs more). The
  Smagorinsky viscosity (strain-rate stencil and a square root) and the
  pressure solve are in between.
* The 256³ DNS step (`step_rk3!`, centred advection, no SGS) is made of
  the memory-bound rows plus the FFT solve, hence 1.8-1.9×. Its
  `dns_step!` adds a per-step history in the fixture code
  (`test/fixtures/dns_decay.jl`) that is not ported: the velocity
  derivative skewness (`circshift` and broadcasts, 0.66-0.68 s at every
  thread count) and ε (0.20-0.24 s). That is 0.88 s of the 2.29 s step at
  12 threads, and the reason the DNS run ends at 48 min. With both on the
  reduction primitives the 12-thread step would be about 1.25 s
  (≈ 26 min for the run); that is a follow-up, not part of this report.

### SMT (24 threads on 12 cores)

Against 12 threads, 24 threads gain where the work is compute-bound
(WENO-Z5 at 256³ +11 %, `set!` sin·exp +25 %, the FFT solve +11 %,
fog_dew +25 %) and lose where it is memory-bound (`advect_centered!` and
`momentum_advect!` at 256³ −18 %, `laplacian` −10 %). The flagship step
is 10.8 % slower at 24 threads than at 12 (inside the ±15 % target); the
child alone is 21 % slower. A threaded in-place update of a 2 MB array
(4096 rows of 64, `launch` mode) takes 29.7 µs at 24 threads against 14.9
at 12 and 15.0 serially. The likely cause, not profiled, is that the
tasks are not pinned, so the array moves between the cores' L2 caches
and between the two CCDs from one launch to the next. **12 threads is the
better setting on this machine.**

### Thread invariance

The harness with every section (`AEOLUS_THREAD_TESTS=1`,
`AEOLUS_THREAD_COUNTS=1,4,12,24`; primitives, 64³ DNS, reductions, the
Berlin smoke matrix, the flagship step, nested, fft, fv, pcg, multigrid,
steps, radiation, hooks, columns, advection, sgs) on this branch, i.e.
every port issue merged with master of 2026-10-07: 3746 hashes per
process, all identical at 4, 12 and 24 threads to 1 thread; 21 of 21
checks pass (24.7 min; the workers took 340 / 353 / 366 / 391 s).
The worker plans FFTs under ESTIMATE, as the whole suite does, so #397
does not reach it.

### Reproducing

```bash
# one process per thread count and mode; JL_MEM-capped wrapper or plain julia
julia -t 12 --project=. benchmarks/bench_ka_scaling.jl flagship 2 9
julia -t 12 --project=. benchmarks/bench_ka_scaling.jl child 2 9
AEOLUS_FFTW_PLANNER=estimate julia -t 12 --project=. benchmarks/bench_ka_scaling.jl dns 256 1 3
AEOLUS_FFTW_PLANNER=estimate julia -t 12 --project=. benchmarks/bench_ka_scaling.jl kernels 64,64,32 256,256,256
julia -t 12 --project=. benchmarks/bench_ka_scaling.jl launch
julia -t 4  --project=. benchmarks/bench_ka_scaling.jl launches 2 9
```

Each mode prints readable lines and `ROW <mode> <threads> <name> <value>
<unit>` lines for collecting the table. The DNS needs about 8 GB.

## SC-a: roofline profile and thread pinning (#401)

Where the time of the flagship step and of the 256³ DNS step goes, kernel
by kernel, against the machine's memory bandwidth, and whether pinning
the threads helps. Script: `benchmarks/bench_roofline.jl`; the same
machine as KA-o (Ryzen 9 7900: 12 cores in two CCDs of 6, 32 MB of L3
per CCD, 24 hardware threads; dual-channel DDR5), Julia 1.13.1, master of
2026-10-08. Most runs were taken on the branch point before #400 and
#392 merged; neither changed `src/`, and the Berlin example changed only
in the `dt` check of #392, which `step_case!(case)` does not take. The
`dns_step!` row, whose history #400 rewrote, was re-measured after the
merge (its `step_rk3!` came out within 3 % of the earlier runs). Every
run started with the 1-minute load below 4 and at least 12 GB free;
every figure is the minimum over three or four processes.

**How it is measured.** The script replaces Backends' three launch paths
(`_launch!`, `_plane_partials!`, `_serial_column_sum`) by copies that time
each call and book it under the kernel body's name, so every launch of
the step is seen, with no change to `src/`. Per kernel it records calls
per step, whether they ran serially (below the backend's threshold), µs
per call, and the compulsory bytes: every distinct numeric array the body
reaches, counted once, capped at the launch's points. That is a lower
bound on the traffic (an axpy counts 2 arrays where STREAM's triad counts
3). In the same process it measures STREAM copy and triad on 256 MB
arrays (DRAM) and on 8 MB arrays (L3), and the empty threaded launch.
Launches over 10⁶ points (the nest parent's 192 × 192 × 53 fields, the
DNS) are booked apart, as `[≥1M]`. The profiled step runs 0-15 % above
the plain step even with the profiler's own time taken out (it evicts
cache), so the step times below are those of plain runs
(`ROOFLINE_PROF=0`) and the in-launch times those of profiled runs.

**Classes.** *Serial*: ran below the threshold. *Bandwidth*: its bytes at
the DRAM triad bandwidth take at least half its time, so only moving
fewer bytes helps. *Compute*: its one-thread time over 12 threads takes
at least half. *Overhead*: neither, so launch, synchronisation or
imbalance dominates. "Excess" is the time per call that neither bound
explains. The roof is DRAM for the 64³ child too: its fields would fit in
L3, but between two uses of a field the step touches more than L3 holds,
and its threaded launches move their compulsory bytes at 53 GB/s in all
(1.32 GB in 24.8 ms per step), 1.35× the DRAM triad and 9 % of the L3
triad (573-587 GB/s).

### Machine

| at 1 / 4 / 12 / 24 threads | 1 t | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: |
| STREAM triad, DRAM (GB/s) | 34.9 | 41.3 | 39.7 | 39.0 |
| STREAM triad, L3 (GB/s) | 114 | 444 | 587 | 179 |
| empty threaded launch (µs) | (serial) | 1.33 | 1.34 | 1.33 |

One thread gets 88 % of the DRAM bandwidth that twelve get, and four
threads saturate it. A kernel that streams from DRAM can therefore gain
little more than 1.2× from threads, however it is launched. (The L3
triad falls at 24 threads, where two threads share each core.)

### Where the step goes

| | 1 t | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: |
| flagship `step_case!`, ms/step (mean of 9) | 158.8 | 92.8 (1.71×) | 87.5 (1.81×) | 95.5 (1.66×) |
| in launches | 143.5 | 67.5 | 60.4 | 66.6 |
| outside launches (FFTW, serial code, allocation) | 15.3 | 25.3 | 27.1 | 28.9 |
| 256³ DNS `step_rk3!`, ms/step | 2105 | 1211 (1.74×) | 1145 (1.84×) | 1181 (1.78×) |
| in launches | 1114 | 811 | 858 | 928 |
| outside launches (FFTW: three Poisson solves) | 992 | 401 | 287 | 252 |
| 256³ DNS `dns_step!` (step + #400's history), ms/step | 2163 | 1300 (1.66×) | 1238 (1.75×) | 1267 (1.71×) |

In launches at 12 threads, by class (ms/step and share of the in-launch
time):

| class | flagship | 256³ DNS |
| :-- | --: | --: |
| bandwidth | 40.5 (68 %, 93 kernels) | 837.7 (98 %, 15 kernels) |
| serial | 14.5 (24 %, 20 kernels) | — |
| compute | 2.9 (5 %, 3 kernels) | — |
| overhead | 1.8 (3 %, 4 kernels) | 20.0 (2 %, 2 kernels) |

By size, at 12 threads: the flagship's launches over 10⁶ points (the
parent) move 0.77 GB/step at 37.6 GB/s, 95 % of the DRAM triad; the
DNS's move 28.3 GB/step at 33.8 GB/s, 86 %. At one thread the same
launches run at 62 % (flagship parent) and 78 % (DNS) of the 12-thread
DRAM triad, which is why threads give them 1.0-1.8×.

### The roofline table: 256³ DNS, 12 threads

All 17 kernels. MB/call is the compulsory model; a 256³ field is 134 MB.

| kernel | calls/step | MB/call | µs/call 1 t | µs/call 12 t | speed-up | % DRAM triad | excess µs | class | ms/step 12 t |
| :-- | --: | --: | --: | --: | --: | --: | --: | :-- | --: |
| `map:_Axpy[≥1M]` | 33 | 268.44 | 8473.6 | 7919.0 | 1.07× | 86 % | 1118 | bandwidth | 261.328 |
| `_b_laplacian![≥1M]` | 12 | 268.44 | 13753.2 | 8426.1 | 1.63× | 81 % | 1625 | bandwidth | 101.113 |
| `map:_Lin2r[≥1M]` | 8 | 287.76 | 8569.3 | 8495.0 | 1.01× | 86 % | 1204 | bandwidth | 67.960 |
| `_b_advect_centered![≥1M]` | 3 | 671.09 | 27677.4 | 18551.3 | 1.49× | 92 % | 1548 | bandwidth | 55.654 |
| `_b_momentum_w![≥1M]` | 3 | 536.87 | 27474.3 | 16881.2 | 1.63× | 81 % | 3279 | bandwidth | 50.644 |
| `_b_momentum_u![≥1M]` | 3 | 536.87 | 25984.1 | 16167.2 | 1.61× | 84 % | 2565 | bandwidth | 48.502 |
| `_b_momentum_v![≥1M]` | 3 | 536.87 | 27289.0 | 15391.3 | 1.77× | 88 % | 1789 | bandwidth | 46.174 |
| `_b_divergence![≥1M]` | 3 | 536.87 | 16326.9 | 14297.2 | 1.14× | 95 % | 695 | bandwidth | 42.892 |
| `_b_∂z_cf![≥1M]` | 3 | 268.44 | 10080.2 | 8657.7 | 1.16× | 79 % | 1856 | bandwidth | 25.973 |
| `_copy_out_body![≥1M]` | 3 | 268.44 | 9875.5 | 8147.2 | 1.21× | 83 % | 1346 | bandwidth | 24.442 |
| `map:_DivBy[≥1M]` | 3 | 268.44 | 8701.6 | 7992.8 | 1.09× | 85 % | 1192 | bandwidth | 23.979 |
| `_b_∂y_cf![≥1M]` | 3 | 268.44 | 9480.1 | 7961.7 | 1.19× | 85 % | 1160 | bandwidth | 23.885 |
| `_b_∂x_cf![≥1M]` | 3 | 268.44 | 8614.0 | 7956.3 | 1.08× | 85 % | 1155 | bandwidth | 23.869 |
| `_copy_in_body![≥1M]` | 3 | 268.44 | 8993.5 | 7413.6 | 1.21× | 92 % | 612 | bandwidth | 22.241 |
| `_scale_by_body![≥1M]` | 3 | 202.90 | 8773.3 | 6331.4 | 1.39× | 81 % | 1191 | bandwidth | 18.994 |
| `_xpencil!` | 24 | 0.55 | 710.5 | 423.9 | 1.68× | 3 % | 365 | overhead | 10.174 |
| `_periodic_body!` | 48 | 1.65 | 249.0 | 204.8 | 1.22× | 20 % | 163 | overhead | 9.831 |

The DNS step is a sequence of full-field passes at 80-95 % of the DRAM
bandwidth. 28.3 GB/step in launches at 39.7 GB/s is 713 ms: the
in-launch time cannot fall below that, at any thread count, without
moving fewer bytes. The two overhead rows are the periodic halo fills.

### The roofline table: flagship, 12 threads

The 30 kernels with the most time per step (120 in all;
`bench_roofline.jl merge flagship 12 <logs>` prints every one). `[≥1M]`
rows are the nest parent's (one parent window in the nine timed child
steps, so its kernels show fractional calls per step).

| kernel | calls/step | serial | MB/call | µs/call 1 t | µs/call 12 t | speed-up | % DRAM triad | excess µs | class | ms/step 12 t |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | --: |
| `_xpencil!` | 152 | 100 % | 0.07 | 24.5 | 29.1 | 0.84× | 6 % | – | serial | 4.428 |
| `_blend_body!` | 72.2 | 100 % | 0.36 | 43.4 | 41.6 | 1.04× | 22 % | – | serial | 3.001 |
| `map:_Axpy[≥1M]` | 2.67 | 0 % | 31.26 | 1007.4 | 925.1 | 1.09× | 85 % | 137 | bandwidth | 2.467 |
| `_interp_body!` | 8.67 | 100 % | 0.23 | 245.8 | 241.8 | 1.02× | 2 % | – | serial | 2.096 |
| `_b_eddy_scalar_masked!` | 9 | 0 % | 6.55 | 1213.9 | 217.1 | 5.59× | 76 % | 52 | bandwidth | 1.954 |
| `_mirror_body!` | 149 | 95 % | 0.14 | 10.1 | 12.6 | 0.80× | 29 % | – | serial | 1.881 |
| `reduce:_photochemistry_at` | 1 | 0 % | 10.49 | 12036.6 | 1566.2 | 7.69× | 17 % | 563 | compute | 1.566 |
| `reduce:_air_heat_at` | 35 | 0 % | 4.19 | 62.0 | 41.8 | 1.48× | 253 % | 0 | bandwidth | 1.463 |
| `_coefficients_body![≥1M]` | 0.667 | 0 % | 78.15 | 2726.9 | 2044.4 | 1.33× | 96 % | 74 | bandwidth | 1.363 |
| `reduce:identity` | 27 | 0 % | 2.10 | 53.7 | 45.6 | 1.18× | 116 % | 0 | bandwidth | 1.231 |
| `_b_advect_centered![≥1M]` | 0.667 | 0 % | 78.16 | 3092.7 | 1809.6 | 1.71× | 109 % | 0 | bandwidth | 1.206 |
| `map:_Lin2r[≥1M]` | 1.11 | 0 % | 35.04 | 1041.8 | 1004.5 | 1.04× | 88 % | 121 | bandwidth | 1.116 |
| `_b_rflux_x!` | 8 | 0 % | 6.39 | 679.1 | 138.1 | 4.92× | 117 % | 0 | bandwidth | 1.104 |
| `_b_eddy_scalar![≥1M]` | 0.778 | 0 % | 46.90 | 2773.5 | 1391.5 | 1.99× | 85 % | 209 | bandwidth | 1.082 |
| `_b_rflux_y!` | 8 | 0 % | 6.39 | 679.9 | 129.8 | 5.24× | 124 % | 0 | bandwidth | 1.039 |
| `_b_rflux_z!` | 8 | 0 % | 6.39 | 714.7 | 121.9 | 5.86× | 132 % | 0 | bandwidth | 0.975 |
| `_buffer_body!` | 116 | 100 % | 0.24 | 8.1 | 8.3 | 0.97× | 73 % | – | serial | 0.968 |
| `_b_damping![≥1M]` | 1 | 0 % | 31.26 | 1561.3 | 945.4 | 1.65× | 83 % | 157 | bandwidth | 0.945 |
| `_b_limit_ratio!` | 8 | 0 % | 16.78 | 512.5 | 117.6 | 4.36× | 360 % | 0 | bandwidth | 0.941 |
| `map:_Axpy` | 11 | 0 % | 4.19 | 140.2 | 83.2 | 1.68× | 127 % | 0 | bandwidth | 0.916 |
| `_n2_body!` | 3 | 0 % | 8.39 | 1797.8 | 299.8 | 6.00× | 71 % | 88 | bandwidth | 0.900 |
| `_b_limit_x!` | 8 | 0 % | 6.39 | 118.8 | 104.1 | 1.14× | 155 % | 0 | bandwidth | 0.833 |
| `reduce:_fog_plane!` | 1 | 0 % | 6.29 | 4837.9 | 808.2 | 5.99× | 20 % | 405 | overhead | 0.808 |
| `_coefficients_body!` | 3 | 0 % | 10.75 | 479.3 | 256.2 | 1.87× | 106 % | 0 | bandwidth | 0.769 |
| `_b_flux_div!` | 8 | 0 % | 8.39 | 207.5 | 96.0 | 2.16× | 220 % | 0 | bandwidth | 0.768 |
| `_b_momentum_u![≥1M]` | 0.333 | 0 % | 62.53 | 3259.6 | 2251.6 | 1.45× | 70 % | 675 | bandwidth | 0.750 |
| `_dual_strain_body!` | 1 | 0 % | 8.39 | 4852.2 | 727.6 | 6.67× | 29 % | 323 | compute | 0.728 |
| `_b_damping_plane![≥1M]` | 0.333 | 0 % | 31.26 | 1666.2 | 1880.1 | 0.89× | 42 % | 1092 | overhead | 0.627 |
| `_dual_strain_body![≥1M]` | 0.111 | 0 % | 62.52 | 39786.1 | 5637.3 | 7.06× | 28 % | 2322 | compute | 0.626 |
| `reduce:_tke_update_plane` | 1 | 0 % | 15.21 | 3876.1 | 606.8 | 6.39× | 63 % | 223 | bandwidth | 0.607 |

* **Bandwidth** holds 68 % of the in-launch time: the parent's stencils
  at 66-157 % of the DRAM triad (above 100 %: part of the data came from
  L3), and the child's at 54-360 %. These are where KA-o's 1.5-2× comes from.
* **Serial** launches take 14.5 ms/step at 12 threads, 17 % of the step
  and as much as at one thread: halo fills (`_xpencil!`, `_mirror_body!`,
  `_buffer_body!`, `_inflow_body!`, `_periodic_body!`, 7.9 ms together),
  the nest's open-boundary blend (`_blend_body!`, 3.0 ms) and parent-to-
  child interpolation (`_interp_body!`, 2.1 ms), and the immersed walls
  (0.7 ms). Each runs below the 32 768-point threshold: about 570 calls
  per step, most of 8-40 µs (the interpolation 240 µs).
* **Launch overhead is not the limit.** The empty threaded launch costs
  1.35 µs at 12 threads; the overhead class is 1.8 ms/step (3 %), and is
  most likely imbalance (`_fog_plane!` over 64 planes, `_b_damping_plane!` over the
  sponge's few planes), not launch cost. Coarser launches alone gain
  little; fusion pays because it moves fewer bytes, not because it
  launches less.
* **Compute** (photochemistry, the dual-grid strain) scales 6.7-7.7×
  already.

### Thread pinning

`JULIA_EXCLUSIVE=1` (no dependency) pins Julia thread *i* to CPU *i − 1*;
on this machine CPUs 0-11 are the 12 cores' first hardware threads and
12-23 their SMT siblings, CPUs 0-5 and 12-17 are CCD 0 (from `lscpu`).
So `-t 12,0` with it is one thread per physical core across both CCDs,
`-t 24,0` is SMT on, and `-t 6,0` is one CCD's six cores; `taskset -c
0-5,12-17 julia -t 12` keeps 12 threads in one CCD (one L3) with SMT.
Each run logs every thread's CPU (`sched_getcpu`) and affinity mask, and
the logs confirm the layouts. ms/step, minimum of three processes (the
spread between processes was up to 40 % for the flagship and 10 % for the
DNS):

| setting | flagship `step_case!` | 256³ `step_rk3!` |
| :-- | --: | --: |
| `-t 12` (default: 12 + 1 interactive, unpinned) | 87.5 | 1145 |
| `-t 12,0` unpinned | 85.5 | 1160 |
| `-t 12,0`, `JULIA_EXCLUSIVE=1` (one per core, both CCDs) | 83.0 | 1144 |
| `-t 24` unpinned (SMT) | 95.5 | 1181 |
| `-t 24,0`, `JULIA_EXCLUSIVE=1` (SMT, pinned) | 91.2 | 1176 |
| `-t 4` unpinned | 92.8 | 1211 |
| `-t 4,0`, `JULIA_EXCLUSIVE=1` | 109.1 | 1225 |
| `-t 6,0`, `JULIA_EXCLUSIVE=1` (one CCD, 6 cores) | 91.4 | 1180 |
| `taskset -c 0-5,12-17`, `-t 12` (one CCD, SMT) | 89.6 | 1210 |

**Pinning does not help.** At 12 threads it is within 5 % of unpinned,
inside the spread between processes; at 4 threads it is 18 % slower for
the flagship. Pinned SMT is still slower than 12 threads, so KA-o's
guess that unpinned tasks cause the SMT loss is not borne out: the loss
is the bandwidth-bound kernels sharing a core's load/store bandwidth.
One CCD's six cores reach 96 % of the flagship's and 97 % of the DNS's
12-core speed, the signature of a bandwidth limit: the second CCD adds
compute, not memory bandwidth. **Recommended launch setting: unchanged,
`julia -t 12` with no pinning.** ThreadPinning.jl was not tried:
`JULIA_EXCLUSIVE` and `taskset` produce the three layouts the issue asks
about without a dependency, and none of them helps.

### Ranked targets for SC-d and SC-e

Savings are estimated from the table: a pass over a field saved is its
bytes at the measured 33-38 GB/s. DNS figures are per 12-thread step of
1145 ms; flagship figures per 87.5 ms step (the parent's per child step).

1. **SC-d: fold the diffusion into the tendency kernels** (DNS ≈ 190 ms,
   17 %). Each substep runs `laplacian(lap, f)` then `axpy!(t, ν, lap)`
   for u, v, w, θ: 12 Laplacians (101 ms) and 12 of the 33 axpys (≈ 95
   ms). Adding `ν∇²f` inside the momentum and scalar advection kernels,
   which already read `f` and write `t`, removes both passes. The parent
   runs the same core.
2. **SC-d: fuse the predictor** `axpy!(u, Δt, tu)` (12 per step, ≈ 95 ms)
   into the last kernel that writes the tendency.
3. **SC-d: fuse the projection's correction**: `∂x_cf/∂y_cf/∂z_cf(φ)`
   then an axpy, 9 + 9 passes (≈ 145 ms), into one gradient-and-correct
   kernel per component (≈ 70 ms saved).
4. **SC-d: fold the RK3 combinations** (`_Lin2r`, 8 per step, 68 ms) into
   the predictor of the substep before them, `u = a u₀ + b (u + Δt t)`:
   one extra read of `u₀` in place of a pass, ≈ 34 ms.
5. **SC-d: the Poisson right-hand side and result**: `divergence` then
   `_copy_in_body!` into the FFT buffer, and `_copy_out_body!` then
   `_DivBy` (≈ 46 ms of 113): write the divergence into the FFT buffer and
   scale on the way out.
6. **SC-b: thread the halo fills and the nest transfer** (flagship 14.5
   ms, 17 % of the step, serial; DNS 20 ms). Not a traffic problem: ~150
   launches per step below the threshold. Batch the faces and fields of
   one fill into one launch so it crosses the threshold, and give the
   nest's blend and interpolation one launch per boundary.
7. **SC-e: re-scope.** The DNS and parent stencils already move their
   compulsory bytes at 80-95 % of the DRAM triad, so their real traffic is
   at most 1.05-1.25× the compulsory one: spatial blocking of a single
   kernel has at most that to gain. What blocking can buy is across
   kernels: running items 1-4 for one block of k-planes while it is in L3
   (temporal blocking), which is the same work as SC-d done per block.
8. **Not targets**: launch overhead (≤ 3 %), pinning (no gain), compute
   kernels (already 6-8×). Outside launches the flagship spends 27 ms per
   step at 12 threads (FFTW and serial code, more than at one thread) and
   the DNS 287 ms (the three FFT solves, 3.5× from one thread): the FFTs
   are the next profile, not this one's.

Items 1-5 together remove about 15 GB of the DNS step's 28.3 GB: at the
measured 33.8 GB/s about 435 ms of 1145, which would bring the 12-thread
`step_rk3!` to about 0.71 s (3.0× over one thread today, against 1.84×).
The flagship's child and parent run the same step core, so items 1-4
apply there too (its axpy, `_Lin2r`, Laplacian and gradient rows).

### Reproducing

```bash
# one process per thread count / setting; JL_MEM-capped wrapper or plain julia
julia -t 12 --project=. benchmarks/bench_roofline.jl flagship 2 9 > f12.log
AEOLUS_FFTW_PLANNER=estimate julia -t 12 --project=. benchmarks/bench_roofline.jl dns 256 1 3 > d12.log
ROOFLINE_PROF=0 JULIA_EXCLUSIVE=1 julia -t 12,0 --project=. benchmarks/bench_roofline.jl flagship 2 9
ROOFLINE_PROF=0 taskset -c 0-5,12-17 julia -t 12 --project=. benchmarks/bench_roofline.jl flagship 2 9
julia --project=. benchmarks/bench_roofline.jl merge flagship 12 f1.log f12.log ...
```

## SC-b: threaded halo fills (#402)

KA-o measured `fill_halos!` at 1.0× on 12 threads at 64 × 64 × 32: every
per-axis phase of a fill (`halo_kernels.jl`) is a separate launch, and at
the flagship's sizes each face slab is below the backend's 32 768-point
threshold, so the fills ran serially between threaded stencils.

**Decomposition.** The phases cannot run at once (an edge or corner holds
the value of the last axis that reaches it), but every x and y phase reads
and writes one k-plane only (the open-boundary plane buffers are read at
the same (j, k)) and every z phase one (i, j) column only. A fill is
therefore two launches: (1) each work item takes a contiguous chunk of
k-planes of every field and runs that field's x then y phases restricted
to them; (2) each work item takes a chunk of j-rows and runs the z phases.
The restriction is carried by the array (`_Clip`, `halo_sets.jl`): the
fill methods of every BC type run unchanged on it and its `_halo_launch!`
narrows the phase to the clip's planes or rows, serially in `launch!`'s
order. Every point is computed from the same operands in the same order as
the unrestricted fill, so the bits are the same at every thread count and
every chunking, including the N < H phases whose layers depend on each
other.

**Sets.** `fill_halos!(fields::Tuple, bcs)` (one `bcs` or one per field,
plus the per-field open-boundary keys) and
`fill_halos!(fields::AbstractVector, bcs)` fill a whole set in the same two
launches, weighted by the halo points of the set. The call sites that fill
several fields together now do it as one set: the prognostic state in the
step cores (`fill_state_halos!`), the velocities before the projection,
the CFL diagnostics and the step guard, the anelastic step, the
advected tracers, the turbulence diagnostics, the nest's interpolated
state and the DiffusivityTensor components. A single-field
`fill_halos!(f, bcs)` is a set of one. Below the threshold and at one
thread the set is the historical field-by-field, phase-by-phase fill.

**Tests.** `test/ka_halos.jl` checks the set fill (θ, u, v, w as one
tuple, and a vector of three tracers sharing one set of conditions)
against the pre-port loops for every closed and open BC case of the KA-f
tests (23 on the two larger meshes, 10 on the N < H mesh), at 1, 2, 5 and
more-chunks-than-planes work items and through the public method on the
serial, default and zero-threshold backends; the
thread-invariance worker hashes set fills (periodic, walls, open) at
every thread count. The full harness (every section, 1, 4, 12 and 24
threads) on this branch merged with master of 2026-10-08: 3785 hashes per
process, all identical to 1 thread; 21 of 21 checks pass (27 min).

**Measured** (same machine and method as KA-o: `bench_ka_scaling.jl`,
one process per thread count, 1-minute load below 4 at the start of every
process, minimum over 3 processes; master `c2a579a5` against this branch,
alternating):

| | 1 t | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: |
| flagship `step_case!`, ms/step (mean of 9), master | 166.3 | 94.1 (1.77×) | 92.1 (1.81×) | 97.3 (1.71×) |
| flagship `step_case!`, SC-b | 162.0 | 91.3 (1.77×) | 83.4 (1.94×) | 97.4 (1.66×) |
| flagship `transport/scalar_advection`, master → SC-b | 45.2 → 43.5 | 21.4 → 20.0 | 20.2 → 18.6 | 25.6 → 25.4 |
| flagship `nest_parent/nesting`, master → SC-b | 54.9 → 54.2 | 37.2 → 36.2 | 35.1 → 33.8 | 34.9 → 35.4 |
| flagship `step_dt`, master → SC-b | 4.56 → 4.48 | 2.57 → 2.45 | 2.65 → 2.39 | 2.80 → 2.69 |
| 256³ DNS `step_rk3!`, s/step, master → SC-b | 2.13 → 2.18 | 1.24 → 1.24 | 1.16 → 1.16 | 1.19 → 1.19 |
| 256³ DNS `dns_step!`, s/step, master → SC-b | 3.31 → 3.19 | 2.29 → 2.29 | 2.25 → 2.21 | 2.24 → 2.24 |
| `fill_halos!` ×4 at 256³ (kernels mode), µs | 4638 → 4554 | 2805 → 2463 | 2182 → 1707 | 2473 → 2094 |

The flagship step is 9.5 % faster at 12 threads (1.94× against its own
one-thread step, from 1.81×) and 2.6 % faster at one thread; 24 threads
are unchanged. The DNS is unchanged within noise: its four 256³ fields
were already above the threshold phase by phase, and the halo is a small
part of a step dominated by memory-bound stencils and the FFT. Halo
filling is 3.9 % of the one-thread flagship step (6.8 ms of 173 ms,
every `fill_halos!` timed: 68 calls per step on SC-b against 89 on
master), which bounds what this issue can gain. In isolation a set of
the four flagship fields (u, v, w, θ) at 192 × 192 × 53 fills in 156 µs
at 12 threads against 458 µs per-field-per-phase (2.9×); at 64 × 64 × 32
the set is slower threaded than serial (43.7-46.9 µs at one thread,
48.2-58.8 µs at 12, 52.4-70.0 µs at 24; periodic, walls, open): task
start-up there costs more than the fill saves. At 64 × 64 × 64, the
flagship child, it is 58-61 µs at 4 threads against 66-71 µs.
A halo point is counted as one point of launch work (`launch_weighted!`):
an intermediate version counted two halo points as one (a fill of a
cached field is cheaper than a stencil point) and gained nothing in the
flagship step at 12 threads (88.3 ms against 89.6 on master, minimum of
3, an earlier session than the table), presumably because the stencils
between two fills have evicted the field; at one unit per point it is
the 83.4 against 92.1 of the table.

## SC-d: kernel fusion in the step cores (#404)

SC-a (#401) found the 256³ DNS step 98 % bandwidth-bound in its launches
and the step cores built from single-purpose passes, each streaming its
fields from DRAM. SC-d fuses them, following SC-a's ranked targets 1-5,
so each field streams once per pass group. Every fused value is computed
from the same operands in the same order as the separate passes, so the
results are the old bits (`test/ka_fusion.jl` checks every field of the
simulation, halos included, against the pre-fusion cores kept in
`test/support/legacy_fusion.jl`).

**What is fused** (`FiniteVolume`, "Fused step-core kernels";
`Simulations._euler_projection!`, `_project_velocity!`):

| before (per substep) | after |
| :-- | :-- |
| `zero_storage!` ×4-5 (`memset`), `momentum_advect!` (3 launches, each reading u, v, w), `advect_centered!` θ (and qv), 4-5 `laplacian` into scratch, 4-5 `axpy!(t, ν, ∇²f)` | write-only launches `t = (0 − A) + ν∇²f`: u, v, w in one (`_b_tendency_momentum!`, reading u, v, w once), θ and qv one each (`_b_tendency_scalar!`); the tendencies' halos zeroed by one halo-shell launch |
| `divergence` into `div_buf`, then the solver's copy into its buffer | `_divergence_into!` writes ∇·u into the spectral solver's source buffer (`PressureSolvers.source_buffer`, `solve_source_buffer!`) |
| `∂x/∂y/∂z(φ)` into `dφdx…` + three `axpy!(u, −1, dφdx)`; `p = φ/Δt` | one launch `_b_correct!`: `u − (φᵢ − φᵢ₋₁)/Δx` … and `p = φ/Δt` (`u + (−1)·d` is `u − d` exactly); an open face's gradient slab only, for its check |
| RK3: after substeps 2 and 3, `combine_storage!` ×4-5 over the storage | the correction writes `c(u − ∇φ, uⁿ)`; θ (qv) `c(θ + Δt tθ, θⁿ)` in its predictor pass; the halos `c(u, uⁿ)` in one halo-shell launch |

**How much to put in one launch.** A launch that writes all five
tendencies reads u, v and w once, but with five stores and some ten
arrays per point the loop stops vectorising. Measured, ms per call of the
five tendencies (minimum of 8; 1 / 12 threads):

| layout | viscous dry 256³ | moist inviscid stretched 192 × 192 × 53 | moist inviscid regular 192 × 192 × 53 |
| :-- | --: | --: | --: |
| before SC-d (`memset` + 5 read-modify-write launches) | — | 18.3 / 10.8 | 18.9 / 10.8 |
| all five in one launch | 82.1 / 48.3 | 33.2 / 6.5 | 39.0 / 6.1 |
| one launch per tendency | 108.8 / 65.6 | 14.6 / 6.7 | 14.7 / 7.0 |
| **u, v, w in one launch, one per scalar** (kept) | 86.0 / 53.1 | 12.0 / 6.1 | 11.6 / 6.0 |

The first version of SC-d used the single launch, which made the
flagship's parent (the moist stretched case) 10 % slower at one thread
(`parent_steps` 51.1 → 56.1 ms, minimum of 3); the kept layout is the
fastest or within 10 % of it everywhere. The correction pass (u, v, w and
p in one launch) does not have the problem: fused 2.9 ms against 3.2 ms
split at one thread, 1.5 against 1.9 at 12 (regular 192 × 192 × 53).

The *halo shell* (`Simulations._shell_map!`) covers what the old passes did
outside the interior (the `memset` zeroed the tendencies' halos; the RK3
combination ran over the storage): one launch over the storage's (j, k)
rows, each row only its halo points, for a tuple of arrays. The fused
tendency kernel applies when the core's momentum advection is the
built-in `momentum_advect!` (the default; a custom `momentum_advection`
keeps the separate passes). The divergence goes into the buffer for the
FFT, cosine-FFT and column (stretched-z, `ColumnTridiagonalSolver`)
solvers: the DNS, the child and the nest parent; the iterative solvers
keep `div_buf`. `div_buf`, `dφdx/dφdy/dφdz` and the
`lap_*` fields are no longer written by `step!` / `step_rk3!` (no reader
used them after a step).

**Serial launches.** The nest's prolongation (`_interp_box!`) sums 8-27
parent values per child point; it now counts that stencil as launch work
(`launch_weighted!`), so the 64³ child's halo boxes (~14 000 points)
run threaded: one halo prolongation of a field at `-t 12` took 2070-2089
µs and takes 867-886 µs (64³ child, ratio 3, order 3; 1 thread 2490-2534
µs, unchanged). The open boundary's time blend (`_blend_body!`, 72 calls
per flagship step, one per plane) stays serial: on a warm 3 × 70 × 70
plane it takes 14.2 µs serially and 28.5 µs threaded (y and z planes 3.0
against 5.0 µs), so threading or merging the planes does not pay.

**Not fused.** The predictor `u ← u + Δt tu` (9 passes per RK3 step) stays
a separate pass: the tendency kernel cannot write `u` in place (its
stencil reads the neighbours), and the divergence cannot form `u*` on
the fly without the halo fill, the open-boundary update and the
`post_predictor!` hook that run between them. The SGS, buoyancy, Coriolis,
sponge and hook terms still add to the tendencies in their own passes.

### Bytes and launches per step

From `bench_roofline.jl` (12 threads, the compulsory-bytes model of SC-a;
the halo-shell launch counted at its halo points, which the model would
count as the whole storage: 4.03 GB/step of the DNS row, 0.27 GB real):

| | before | after |
| :-- | --: | --: |
| 256³ DNS `step_rk3!`, GB/step in launches | 29.14 | 14.90 (−49 %) |
| 256³ DNS, tendency `memset`s (outside launches) | 1.73 GB | none |
| 256³ DNS, launches/step (profiler, nested included): threaded / serial | 107 / 864 | 55 / 864 |
| flagship `step_case!`, GB/step in launches | 2.313 | 2.194 (−5 %) |
| flagship, of which the nest parent's launches over 10⁶ points | 0.766 | 0.666 (−13 %) |
| flagship, launches/step (KA-o's `launches` mode): threaded / serial | 416 / 6476 | 405 / 6469 |

SC-a estimated 15 GB for targets 1-5; the fusion removes 14.2 GB of the
DNS's launches and the 1.7 GB of `memset`s (keeping u, v, w apart from
the scalars' tendency launches costs 1.2 GB of it, see above). The
flagship gains far less: its step cores are a small part of its step (the child's SGS, scalar
transport, surface and radiation couplings are not step-core passes). The
launch counts of the flagship include, since SC-b, the serial per-item
launches inside its threaded set fills, so they are not comparable with
KA-o's 263 / 569.

### Measured speed-ups

`benchmarks/bench_ka_scaling.jl` (KA-o), one process per thread count,
base (the branch point: SC-a + SC-b + master) and SC-d run alternately;
every process started with the 1-minute load below 4 and at least 12 GB
free; each figure is the minimum over 3 processes. Flagship: 2 warm-up and
9 timed steps, the mean `step_case!` of the 9 (long Berlin runs are
paused). DNS: `AEOLUS_FFTW_PLANNER=estimate`, 1 warm-up and 3 timed steps.
Speed-ups in brackets are over the same code at one thread.

| | 1 t | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: |
| 256³ DNS `step_rk3!`, base (s) | 2.090 | 1.210 (1.73×) | 1.141 (1.83×) | 1.173 (1.78×) |
| 256³ DNS `step_rk3!`, SC-d (s) | **1.627** (−22 %) | **0.793** (2.05×) | **0.693** (2.35×) | **0.693** (2.35×) |
| SC-d over base at one thread | 1.28× | 2.63× | 3.02× | 3.02× |
| 256³ DNS `dns_step!` (with the history), base → SC-d (s) | 2.172 → 1.726 | 1.276 → 0.856 | 1.209 → 0.762 | 1.245 → 0.765 |
| flagship `step_case!`, base (ms) | 157.2 | 88.7 (1.77×) | 83.5 (1.88×) | 94.1 (1.67×) |
| flagship `step_case!`, SC-d (ms) | **152.6** (−3 %) | **83.3** (1.83×) | **76.6** (1.99×) | **87.6** (1.74×) |
| flagship, the nest parent (`parent_steps`), base → SC-d (ms) | 51.1 → 47.6 | 35.4 → 30.3 | 32.8 → 27.8 | 33.9 → 29.2 |
| flagship, the child's core (`dynamics/step!`), base → SC-d (ms) | 35.0 → 33.4 | 20.3 → 19.5 | 19.3 → 18.1 | 22.4 → 20.1 |

The 12-thread DNS step is 0.693 s against SC-a's estimate of 0.71 s for
targets 1-5. Single-thread time is better, not worse: −22 % for the DNS
and −3 % for the flagship. The flagship's spread between processes is up
to 18 % (SC-a saw 40 %); its minima move by 3-8 %, consistent with the 5 %
fewer bytes, while its child-core and parent entries are steadier
(−4 to −15 %).

Thread invariance: the harness (`AEOLUS_THREAD_TESTS=1`,
`AEOLUS_THREAD_COUNTS=1,4,12,24`) passes, 21 of 21 checks, 3785 hashes
identical at 1, 4, 12 and 24 threads.

### Reproducing

```bash
# base and SC-d, one process per thread count, alternating, load < 4
julia -t 12 --project=. benchmarks/bench_ka_scaling.jl flagship 2 9
AEOLUS_FFTW_PLANNER=estimate julia -t 12 --project=. benchmarks/bench_ka_scaling.jl dns 256 1 3
julia -t 12 --project=. benchmarks/bench_roofline.jl flagship 2 9    # bytes and launches
julia -t 12 --project=. benchmarks/bench_ka_scaling.jl launches 2 9
```

## SC-e: temporal blocking of the hook-free step core (#405)

The issue asked for spatial cache blocking of the bandwidth-bound
stencils at 256³ that SC-d's fusion leaves. Re-measured after SC-d
(`bench_roofline.jl dns 256 1 3`, 12 threads, profiled), every remaining
full-field kernel already moves its compulsory bytes, plus the
write-allocate of its outputs, at 88-120 % of the DRAM triad (39.1 GB/s
in the same process):

| kernel (256³, 12 t, after SC-d) | calls/step | MB/call (compulsory + write-allocate) | ms/call | GB/s | % DRAM triad |
| :-- | --: | --: | --: | --: | --: |
| `_b_tendency_momentum!` | 3 | 805 + 403 | 35.1 | 34.4 | 88 % |
| `_b_correct!` (u, v, w, p, RK3 combination) | 3 | 940 + 134 | 30.5 | 35.2 | 90 % |
| `_b_tendency_scalar!` (θ) | 3 | 671 + 134 | 18.8 | 42.9 | 110 % |
| `_b_divergence_into!` | 3 | 537 + 134 | 14.3 | 47.0 | 120 % |
| `map:_Axpy` (the u, v, w predictors) | 10 | 268 + 0 | 7.8 | 34.2 | 88 % |
| `map:_AxpyStage` (θ predictor + RK3 combination) | 2 | 403 + 0 | 10.8 | 37.2 | 95 % |

So the stencils' neighbour planes are already reused from cache, and no
tiling of a single kernel can move fewer bytes than these: **spatial
blocking has nothing left to win**. What is left is the traffic *between*
kernels. Of SC-d's "not fused" list only one pair has nothing global
between its two halves: the tendencies and the predictor `u ← u + Δt·tu`
(the divergence needs the halo fill of u*, the correction the FFT
solve). SC-d could not fuse them pointwise (the tendency stencil reads
the neighbours of the point the predictor would overwrite); blocking
can, when nothing else touches the tendencies in between.

**The sweep** (`FiniteVolume.blocked_predictor!`). The k-range is cut
into one slab per thread. A slab is swept j-row by j-row: the tendencies
of row j (every k of the slab) go into a ring of two rows (`SweepBuffers`,
2 × 5 × 22 × 262 values per slab at 256³, in L2), then row j − 1 gets its
predictor in place, since no later tendency of the slab reads it (the
stencils reach j ± 1, k ± 1). A slab's first and last planes are read by
the neighbouring slabs, so their tendencies are kept (`edge`) and updated
in a second launch once every slab is done. Every value is the
expression of the separate passes (`_tend_u` … then the predictor
functor), so the bits are the same for any slab count and thread count,
and the same as the unblocked core: `test/ka_blocking.jl` checks 1-5
slabs, regular and stretched grids, dry and moist, `step!` and
`step_rk3!` against the separate passes and against the pre-fusion cores
of `test/support/legacy_fusion.jl`. The tendency fields are neither read
nor written on this path; the halo zeroing of the tendencies goes too.

It applies when **nothing adds to the tendencies or reads the state
between the base tendencies and the predictor** (`_blocked_core`): the
built-in momentum advection and no SGS closure, buoyancy, Coriolis,
surface flux, sponge, `body_force!`/`post_predictor!` hook or open
boundary, on the CPU. That is the DNS core. θ (and qv) are updated in the
same sweep, before the projection, which does not read them. The flagship
child and parent have SGS, buoyancy and hooks, so they keep the SC-d path:
SC-e does not change the flagship.

**Since #533** the condition is a contract instead of a fixed list: the
sweep composes a compile-time tuple of terms (`FiniteVolume.SweepTerm`:
`footprint`, `add_tendency!`, `prepare!`, `sweep_inputs`), the base
tendencies first, and a substage takes it when every term of the
substage has a `sweep_term`. The ring holds `1 + max ry` rows and a slab
keeps `max rz` edge planes per end, so a term with a wider stencil than
the base's needs no change of the driver. A substage with a term that
has none runs the passes above (the reference) and the run records the
term and its documented reason (`Simulations.SWEEP_FALLBACKS`) in its
`results.toml`. Plan step 1 moved only the base tendencies onto the
interface; the terms follow in the later steps of #533.

Three layouts were tried first (a prototype outside `src/`, momentum
only, 256³, ms per substage, minimum of 3-6 processes at load < 4; its
row loops were inlined, see below):

| | 1 t | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: |
| SC-d: tendency launch + 3 predictor launches | 82.6 | 55.7 | 56.5 | 62.2 |
| plane-lagged per slab, tendencies still in `tu, tv, tw` | 84.2 | 46.7 | 54.1 | 60.9 |
| plane-lagged, a ring of two k-planes | 63.4 | 29.1 | 40.1 | 53.1 |
| **j-row-lagged, a ring of two j-rows** (kept) | 78.1 | 27.9 | 27.6 | 37.3 |

Writing the tendencies to memory keeps almost all the traffic (the
second row). A ring of whole planes holds 7.5 MB per thread, more than
the 32 MB L3 of a CCD holds for six threads, so it gains little at 12 and
24 threads; a ring of j-rows fits in L2. **The row loops must be their own
`@noinline` functions with the arrays as arguments** (the rule of
`Backends._row!`): inlined into the slab body the first version did not
vectorise and ran at 0.51× of the separate passes at one thread (253.6
against 130.1 ms per substage, momentum + θ); with `@noinline` and
`@simd ivdep` it is 98.8 against 119.5 ms at one thread and 26.1 against
83.0 ms at 12 (3.2×). One slab per thread is the fastest count at 12
threads (4: 31.3, 8: 27.7, 12: 26.1, 16: 29.3, 24: 30.8 ms); at one
thread 4-12 slabs are 5 % faster than one, which is kept for simplicity.

### Measured speed-ups

`benchmarks/bench_ka_scaling.jl` (KA-o), one process per thread count,
base (the branch point: SC-a + SC-b + SC-c + SC-d + master) and SC-e
alternately; every process started with the 1-minute load below 4 and at
least 12 GB free; each figure is the minimum over 3 processes. DNS:
`AEOLUS_FFTW_PLANNER=estimate`, 1 warm-up and 3 timed steps. Brackets:
speed-up over the same code at one thread.

| | 1 t | 4 t | 12 t | 24 t |
| :-- | --: | --: | --: | --: |
| 256³ DNS `step_rk3!`, base (s) | 1.628 | 0.798 (2.04×) | 0.692 (2.35×) | 0.690 (2.36×) |
| 256³ DNS `step_rk3!`, SC-e (s) | **1.545** (−5 %) | **0.650** (2.38×) | **0.509** (3.03×) | **0.495** (3.12×) |
| SC-e over base at the same thread count | 1.05× | 1.23× | 1.36× | 1.39× |
| 256³ DNS `dns_step!` (with the history), base → SC-e (s) | 1.718 → 1.643 | 0.866 → 0.711 | 0.762 → 0.573 | 0.763 → 0.569 |
| flagship `step_case!`, base (ms) | 161.4 | 90.4 (1.79×) | 80.5 (2.01×) | 89.7 (1.80×) |
| flagship `step_case!`, SC-e (ms) | 163.6 | 86.1 (1.90×) | 81.1 (2.02×) | 91.4 (1.79×) |
| flagship, the nest parent (`nest_parent`), base → SC-e (ms) | 52.9 → 53.1 | 32.3 → 32.2 | 29.3 → 29.2 | 30.0 → 30.1 |
| flagship, the child's core (`dynamics/step!`), base → SC-e (ms) | 37.5 → 37.3 | 20.8 → 20.8 | 19.2 → 19.4 | 22.0 → 20.4 |

The flagship does not take the blocked path (its cores have SGS,
buoyancy and hooks), and its times agree within the spread between
processes (up to 12 % here: 161-173 ms at one thread).

The 12-thread DNS step went from 0.692 to 0.509 s (−26 %), 3.0× over one
thread of the same code and 4.1× over the 2.090 s of the pre-SC-d base at
one thread. Single-thread time is better, not worse (−5 %).
From a profiled run at 12 threads (`bench_roofline.jl dns`, the
compulsory-bytes model of SC-a), the kernels the sweep replaces, per
step:

| | before (SC-d) | after (SC-e) |
| :-- | --: | --: |
| launches | `_b_tendency_momentum!` ×3, `_b_tendency_scalar!` ×3, `map:_Axpy` ×10, `map:_AxpyStage` ×2, halo shells ×7 | `_b_sweep_slab!` ×3, `_b_sweep_edge!` ×3, halo shells ×4 (the tendencies' are gone) |
| compulsory GB/step (shells not counted) | 7.92 | 2.26 |
| ms/step at 12 t (shells included) | 275.3 | 96.2 |

One sweep call reads and writes u, v, w and θ once (1.07 GB of real
traffic with the write-back) plus the edge planes, in 25.9 ms: 42 GB/s,
at the DRAM triad (39.1 GB/s in the same process). That is the minimum
for an in-place update of four fields; the sweep is itself at the roof.
The top-level kernels of the profiled step sum to 517 ms before and 341
ms after.
Of the 0.509 s at 12 threads, the FFTs of the three Poisson solves
(outside launches) are now the larger part: SC-a measured 287 ms of them,
and no kernel change touches them. They, not blocking, are the next
target.

Thread invariance: the harness (`AEOLUS_THREAD_TESTS=1`,
`AEOLUS_THREAD_COUNTS=1,4,12,24`) passes, 21 of 21 checks, 4797 hashes
identical at 1, 4, 12 and 24 threads; its DNS fixture section takes the
blocked path with 1, 4, 12 and 16 slabs (threshold 0, 32³).

### Reproducing

```bash
# base and SC-e, one process per thread count, alternating, load < 4
AEOLUS_FFTW_PLANNER=estimate julia -t 12 --project=. benchmarks/bench_ka_scaling.jl dns 256 1 3
julia -t 12 --project=. benchmarks/bench_ka_scaling.jl flagship 2 9
AEOLUS_FFTW_PLANNER=estimate julia -t 12 --project=. benchmarks/bench_roofline.jl dns 256 1 3
```

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
