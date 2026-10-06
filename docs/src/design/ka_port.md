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
