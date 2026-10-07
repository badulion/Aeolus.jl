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
| `range_max([b,] body, ranges, args...)` | NaN-propagating maximum of an indexed body, which must only read (KA-b) |
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
* **CG / MG dot products**: ported by KA-g (#300), below.
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
moved. One assertion was replaced: `era2_multigrid.jl` asserted that
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
section (below). `test/era2_stretched.jl` asked *which module* defined
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
* *BLAS.* Pinned to one thread when Julia runs threaded (KA-a). LAPACK's
  bits depend on the BLAS thread count, so the harness section pins it
  too. Under `-t 1` the BLAS setting is left alone, and a cached solve
  equals an uncached one at whatever count is set.
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
  on the new streams against 0.99 on the old. Four `era3_montecarlo`
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
