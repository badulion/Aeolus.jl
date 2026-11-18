# #566 handoff: run 1 done; the Berlin 20 m scaling test failed (2026-11-12)

For the agent that picks this up locally (with `gh`). The original plan is in
`HANDOFF.md` (maintainer, 2026-11-10); `CLUSTER.md` has the facts about Fritz.
This file records what happened since, where everything is, and what is still
open.

**Your first task: file issue 1 (slow model setup) from §5.1, as it stands, on
`badulion/Aeolus.jl`.** Issue 2 (§5.2) is a draft: file it only if the
maintainer agrees. Fritz has no `gh` and no GitHub token, so neither could be
filed from there.

---

## 1. State of the branches

| What | Where |
| :-- | :-- |
| scripts, run 1 report, this handoff | `issue/566-hpc-benchmark` (pushed with this file) |
| the 13 benchmark builds, the hook commits included | `bench/master`, `bench/all`, `bench/only-{A,B,C,D,E,543}`, `bench/all-minus-{A,B,C,D,E}` (all pushed) |
| the kernel-fix variants (node 17) | `bench/all-revert-E4`, `-E5`, `-E6` (pushed) |
| the launch-trace hook | `bench/hook` (f4e22fc5), `bench/hook-dyn` (efb5ec4c + test fix 5444aa27) |

- **bench/all is at 21196f5e.** That's what run 1 and the 20 m attempt both ran.
- **Two trial branches stay local on Fritz:** `bench/only-C-trial` and `bench/only-D-trial`, superseded merges kept for reference.
- **No PR is open for #566.** Nothing has been merged; agents don't merge.

## 2. Run 1 (2026-11-10): done

The report is `results/run1/report.md`, next to `summary.md`, the figures and the per-process TOMLs. The headline:
- **Speed-up:** bench/all is 1.35–1.56× faster per step than master at 1–72 threads, more on 50 m and at higher thread counts.
- **Scaling stops at about 16 threads:** 16 → 72 threads gains only 1.13–1.17×.
- **The cause isn't memory bandwidth:** memory traffic is 15–22% of the STREAM triad, and threads sit idle about 80% of the time inside kernel launches (load imbalance or launches too small).
- **C (#550, loop fusion) is the only group that still pays off inside all** (1.06–1.17×).
- **E6:** its update call is 5× faster at 72 threads, and the allocation per call drops from 857 MB to about 0.1 MB.

**Two caveats** (the report has both):
- **2 threads ran slower than 1.** The maintainer's analysis blames the main task running on an unpinned interactive thread; `-t N,0` in `scaling.sh` should fix that. Not yet verified, because the 20 m run never got to 2 threads.
- **The launch trace inflates step times about 1.7×,** so the breakdown figures show shares only.

## 3. The Berlin 20 m scaling test (2026-11-11): failed in its first step

**The plan** (maintainer, commit 55d9659e): `berlin_20m/` is the 100 m domain at 1000 × 1000 × 150 cells of 20 m, 150 M points, `dt = 0.4 s`. One job, `scaling.sh`, runs bench/all at 72, 36, 16, 8, 4, 2 and 1 threads, one process each.

### 3.1 Driver preparation: fine

- **Job:** `prepare.sh` ran on a compute node (job 4461823): 4 min 36 s wall time, peak 1.9 GB (largest single process).
  The README expected "much longer than 50 m, tens of GB". Record: `results/scaling_20m_attempt1/prepare.{log,time.txt}`.
- **Output:** `berlin_20m/input/berlin_20m_static.nc` (3.2 MB, 1000 × 1000) and `berlin_20m_20250824_dynamic.nc` (746 MB). The land-use table is `berlin_100m/data/berlin_20m_landuse.csv` (43 MB). These sit under the gitignored `input/` and `data/`.
- **The land-use check** (the 50 m check at R = 2):

  | | 100 m | 50 m | 20 m |
  | :-- | --: | --: | --: |
  | cells | 40 000 | 160 000 | 1 000 000 |
  | counts sum to R² | yes (100) | yes (25) | yes (4) |
  | λp = roof count / R² | yes | yes | yes |
  | mean λp | 0.1439 | 0.1439 | 0.1439 |
  | mean λf | 0.0958 | 0.0958 | 0.0958 |
  | max λf | 1.27 | 2.72 | **11.50** |
  | max h_mean | 142 m | 164.6 m | **368 m** (Fernsehturm) |
  | mean h_mean | 10.44 m | 8.16 m | 4.09 m |
  | cells with buildings | 75% | 59% | 29% |

### 3.2 Two script fixes (commit bdc807d8, pushed with this file)

- **No `/usr/bin/time` on the compute nodes** (the login node has it). The first driver job failed at once with exit 127. Every `scaling.sh` run would have done the same, because it wraps each process in `/usr/bin/time -v`.
  - **Fix:** `timev.py` (stdlib Python) writes the same report lines (`Maximum resident set size (kbytes)`, `Elapsed (wall clock) time`, exit status). `scaling.sh` uses it when `/usr/bin/time` is missing, loading `python/3.13-miniforge` if no `python3` is on the PATH.
  - **Its peak memory** is that of the largest single process among the children (`getrusage(RUSAGE_CHILDREN)`), as GNU time reports it.
- **`BENCH_HPC` must be set when submitting `scaling.sh`.** sbatch runs a copy of the script from Slurm's spool directory, so it falls back to `$BENCH_HPC` to find `bench_run.jl`. Under `set -u` an unset variable stops the job at once. The command in the instructions lacked it. The working submit command:

  ```
  W=<repo>/.worktrees
  BENCH_ENV=$WORK/aeolus-bench/env.sh BENCH_ROOT=$W BENCH_HPC=$W/issue-566/benchmarks/hpc \
    sbatch --constraint=hwperf -o $WORK/aeolus-bench/scaling20/scaling_%j.out \
    $W/issue-566/benchmarks/hpc/scaling.sh $WORK/aeolus-bench/results/scaling_20m
  ```

### 3.3 The 72-thread run (job 4461824, node f0352)

| | |
| :-- | :-- |
| start | 12:34:27 |
| setup | about 43 min, almost all on one core (see §4) |
| memory during setup | 36–39 GB resident, flat from minute 2 |
| first warm-up step | **failed**: `ArgumentError: solve_surface_layer: need z_m > z0m > 0 and z_h > z0h, z0q > 0` |
| wall time to the failure | 44 min 53 s |
| peak memory | **69 GB** (README estimate: ~200 GB) |
| CPU | 12 626 s user + 3 721 s system |

- **I cancelled the job** when the 36-thread process had just started. Setup and the first step are identical at every thread count, so each remaining rung would have spent about 45 min and then failed the same way.
- **The ladder produced no step timings,** so `summary.md` was never written and nothing went under `results/scaling_20m/`.
- **The failure record is in `results/scaling_20m_attempt1/`:**

  | File | Content |
  | :-- | :-- |
  | `machine.txt` | the node |
  | `scaling_20m_t72_all.time.txt` | time and memory report |
  | `failure_t72.log` | the error with its full stack trace |
  | `stack_t72_minute33.txt` | the main thread during setup |
  | `watch_t72.log` | RSS and CPU% every 2 min (`srun --overlap`) |
  | `prepare.*` | driver preparation |

  The full job log (116 KB, mostly the `USR1` all-thread stack dump) stays in `$WORK/aeolus-bench/results/scaling_20m/scaling.log`.

**The failing call path** (`failure_t72.log`):

```
_tile_column! (urban/parent/components.jl:580) → step_parent_column! (urban/parent/surface.jl:520)
 → SurfaceTiles.step_surface! → _step(SlabLand) → Radiation.step_surface! → turbulent_fluxes
 → surface_layer_solution (SurfaceExchange.jl:1047) → solve_surface_layer (SurfaceExchange.jl:517): ArgumentError
```

## 4. Why setup took about 43 minutes: partly understood

**Known:**
- **Not I/O.** The drivers had been read by about minute 2 (38 GB resident). `/proc/<pid>/io` showed 650 MB read, and nothing more over 10 s at minute 31; the process had no files open.
- **Mostly one busy core.** `top` showed ~100% at almost every 2-minute sample, with bursts of 2980% at minute 4 and 3567% at minute 10.
- **But the CPU accounting is ~6 cores on average:** 16 347 CPU-s over 2 693 s wall, 3 721 s of it system time. That fits frequent garbage collections that each wake the 71 idle GC threads, and the 2-minute samples miss them.
- **One stack sample** at minute 33 (`kill -USR1`; `stack_t72_minute33.txt`):

  ```
  Config.build → _build_domain → build_components (components/domain.jl:62)
   → SGS build (components/sgs.jl:184) → _tke_sgs (sgs.jl:196) → _wall_distance (sgs.jl:209)
    → IBM.wall_distance, seed loop (IBM.jl:472)
     → wrap (IBM.jl:453) → ijl_gc_small_alloc → _jl_gc_collect → gc_wake_all_for_marking
  ```

  The profile report Julia announces after `USR1` never printed: it waits for a yield point, and the main thread didn't yield. So there is a stack, not a profile.
- **`IBM.wall_distance` (src/IBM.jl:435-514) is serial.** It loops over all cells × 26 neighbours × 3 neighbour indices in the seed pass, and again in every forward/backward sweep round until nothing changes (`maxsweeps = 50`, at least 2 rounds). Its neighbour lookup allocates on every call:

  ```julia
  nb = bcs === nothing ?
      FieldBoundaryConditions(ntuple(a -> periodic[a] ? :periodic : :noflux, 3)...) : bcs
  @inline wrap(i, a) = neighbor_index(m, nb, i, a; outside = :none)
  ```

  My guess, unverified: `nb` is type-unstable, because its boundary types come from runtime symbols, so every call is a dynamic dispatch with a keyword argument. At 150 M cells that's on the order of 10¹⁰ allocating calls.
- **Setup cost is about linear in grid size:**

  | Case | Points | Build | Per M points |
  | :-- | --: | --: | --: |
  | 100 m | 2.7 M | 29 s | 11 s |
  | 50 m | 9.6 M | 157 s | 16 s |
  | 20 m | 150 M | ~2580 s | 17 s |

  Nothing superlinear shows up; setup is just slow and serial per point. The 100 m and 50 m numbers are from run 1, which used system images; the 20 m run had none, so it compiled during setup too, about 1 min of the 43.

**Not known:**
- **How the 43 minutes split** between `wall_distance`, GC overhead and the rest of setup (urban and land-surface columns, `initialize_tke!`, driver interpolation, FFTW planning).
- **Whether `wall_distance` dominates at all.** A rough cost estimate of its seed loop alone exceeds the ~10 minutes left after the minute-33 sample, so either the estimate or the premise is off.

**Proposed measurement:** the 50 m case has the same cost per point and builds in 2.5 min, so profile that.
- One node, about 15 min.
- Build under `Profile`, once at `-t 72,0` and once at `-t 1,0`.
- Report per-function counts and the `@timed` GC share.

That is issue 1.

## 5. Issues

### 5.1 Issue 1: file this as it stands

**Title:** Model setup is slow and serial at large grid sizes (Berlin 20 m: ~43 min before the first step)

**Body:**

> **Observed.** The #566 scaling test (Berlin 20 m, `benchmarks/hpc/berlin_20m/`:
> 1000 × 1000 × 150 cells, 150 M points, bench/all at 21196f5e) spent about
> 43 min in `Config.build` before its first step, on one Fritz node (2 × 36 Ice
> Lake cores, 256 GB) with `-t 72,0`. (The run then failed in that first step for
> an unrelated reason: see the surface-layer issue.)
>
> | case | points | build (s) | s per M points |
> | :-- | --: | --: | --: |
> | Berlin 100 m | 2.7 M | 29 | ~11 |
> | Berlin 50 m | 9.6 M | 157 | 16 |
> | Berlin 20 m | 150 M | ~2580 | 17 |
>
> 100 m and 50 m from run 1 (`benchmarks/hpc/results/run1/`); 20 m from the wall time to the failure (44:53) less the first step.
> The build grows about linearly with the number of points: setup is slow and mostly serial per point, not superlinear.
> At 20 m each process of a scaling ladder pays it again (7 thread counts ≈ 5 h of setup).
>
> **What is known** (record: `benchmarks/hpc/results/scaling_20m_attempt1/`, analysis: `benchmarks/hpc/HANDOFF_scaling20m.md` §4).
> - **Not I/O.** The drivers had been read by about minute 2 (38 GB resident); after that, no further reads.
> - **Mostly one busy core** (~100% CPU in 2-min samples, bursts of 3000–3600%).
>   But the process used 12 626 s user + 3 721 s system over 2 693 s wall, about 6 cores on average.
>   The high system time and the bursts point at garbage collections that wake all 72 GC threads.
> - **One stack sample** at minute 33:
>   ```
>   Config.build → build_components → SGS build → _tke_sgs (components/sgs.jl:196)
>     → IBM.wall_distance, seed loop (IBM.jl:472)
>       → wrap (IBM.jl:453) → ijl_gc_small_alloc → _jl_gc_collect → gc_wake_all_for_marking
>   ```
>   `wall_distance` loops serially over all cells × 26 neighbours × 3 axes, in the seed pass and in each sweep round.
>   Its neighbour lookup `wrap(i, a) = neighbor_index(m, nb, i, a; outside = :none)` allocates on every call.
>   Possibly `nb` is type-unstable (built from runtime symbols), so each call is a dynamic dispatch with a keyword argument; not checked.
> - **Peak memory** at 72 threads: 69 GB (38 GB during the build).
>
> **What is not known.**
> - How the time splits between `wall_distance`, the GC overhead and the rest of setup (urban/land-surface columns, `initialize_tke!`, driver interpolation, FFTW planning).
>   One sample can't say.
> - Whether `wall_distance` dominates at all: a rough cost estimate of its seed loop alone exceeds the time left after the sample.
>
> **Proposed investigation.**
> 1. **Profile the build on the 50 m case** (same cost per point, 2.5 min instead of 45).
>    Use `Profile` with `Profile.print(format = :flat, sortedby = :count)` plus `@timed` for the GC share.
>    Run at `-t 72,0` and `-t 1,0` (the GC thread count the only difference), on a compute node, not a login node.
> 2. **Report** per setup phase: wall time, allocations, GC time.
> 3. **Fix what dominates.** If it's `wall_distance`: a type-stable neighbour lookup or a function barrier, checked against the current result (`test/tke.jl` has the analytic cube case); then decide whether its loops should be threaded.
> 4. **Re-measure** the 100 m, 50 m and (once the surface-layer failure is fixed) 20 m build times, before and after.
>
> Related: #566.

### 5.2 Issue 2: draft, file only if the maintainer agrees

**Title:** Urban parent surface: forcing height at or below z0m on fine grids (Berlin 20 m fails in its first step)

**Body:**

> The Berlin 20 m case (#566, `benchmarks/hpc/berlin_20m/`) builds, then fails in its first step:
> `ArgumentError: solve_surface_layer: need z_m > z0m > 0 and z_h > z0h, z0q > 0`
> (`SurfaceExchange.jl:517`, via `turbulent_fluxes(::SlabLand)` ← `step_parent_column!` ← `_tile_column!`;
> trace in `benchmarks/hpc/results/scaling_20m_attempt1/failure_t72.log`).
> 100 m and 50 m on the same domain and land-use source run fine.
>
> **How the code gets there.**
> - Each built tile is forced at `z_ref = z_k − d` (`TileForcings`, `urban/parent/surface.jl`).
>   `k = parent_forcing_level(z_centres, h)` is the lowest level centre at or above the mean building height `h`.
> - Its roughness is `z0m = max(z0_Macdonald, class z0m)` (`build_parent_surface`).
> - `parent_forcing_level`'s docstring argues that above `h`, `z − d ≥ h − d > z_0` for every λ (Macdonald et al. 1998).
>
> **Hypothesis (unverified): two ways that argument fails on a fine grid.**
> 1. **d → h as λp → 1.** Macdonald's `d/h = 1 + A^(−λp)(λp − 1)` reaches 1 at full plan cover.
>    At 20 m (R = 2, four 10 m raster cells per model cell) fully roofed cells are common.
>    So `h − d` → 0, and `z_k − d` is tiny when a level centre sits just above `h`.
> 2. **The class z0m floor.** `max(z0_Macdonald, class z0m)` isn't covered by Macdonald's inequality.
>    Where `z0_Macdonald` → 0 (also at λp → 1), the class value sets z0m regardless of `z_k − d`.
>
> The 20 m land use also has extremes the coarser grids smooth away: max λf 11.5 (1.27 at 100 m), a 368 m cell (the Fernsehturm), mean h 4.1 m.
> Whether `parent_morphometry`'s smoothing window (`window = 8`) softens λp enough is part of the question.
>
> **To check first:** evaluate `macdonald_roughness` and `parent_forcing_level` on the 20 m table (`berlin_100m/data/berlin_20m_landuse.csv`, cheap, no model).
> List the cells with `z_k − d ≤ z0m`, with their λp, λf, h, d, z0.
>
> **Then decide (maintainer):** a physical fix in the forcing level (e.g. the lowest level with `z_k − d` above a multiple of z0m), a bound on d/h, the treatment of the class z0m floor for built tiles, or a statement that 20 m is outside the parametrisation's range (and a different case for the scaling test).
>
> Related: #566.

### 5.3 Issue 3: filed as #574 (allocation per kernel row and per step)

**Title:** Stepping allocates ~1.2 GiB per step at Berlin 20 m: boxed arguments per kernel row (`Backends.jl:356`) and a fresh open-boundary snapshot every step

**Body:**

> **Observed** (#566, Fritz, bench/all 51154519 = master 4f61b8a6 + the bench PRs, unpinned; `benchmarks/hpc/results/memory_20m/`, `bench_run.jl memory` mode):
>
> | case, threads | s/step | allocated per step | live heap after a full GC |
> | :-- | --: | --: | :-- |
> | Berlin 20 m (1000 × 1000 × 150), 72 | 5.9–6.2 (7.7 with a GC) | 1.22 GiB | back to the baseline (73.61 GiB) |
> | Berlin 50 m (400 × 400 × 60), 72 | 1.0 | 0.15 GiB | back to the baseline |
> | Berlin 50 m, 8 | 1.4 | 0.15 GiB | back to the baseline |
>
> - Not a leak: the live heap grows by the allocated bytes and a full GC returns it exactly. But at a 74 GiB live heap Julia collects rarely, so RSS grows by ~10 GB over 10 steps, and a collection costs 1.7 s at 72 threads (~25 % of a step).
> - The allocation grows with the grid (the number of rows), not with the thread count.
>
> **Where** (`Profile.Allocs`, 1 % sampling, one step at 20 m, 72 threads):
>
> 1. **~1150 MiB per step at `src/Backends.jl:356`**, `_row!(body!, i1:i2, j, k, args...)` in `_rows3_kernel!`: ~48 M allocations per step, a few per kernel row. Types: `Tuple{Int64,Int64,Int64}` 422 MiB, `UnitRange{Int64}` 275 MiB, `Float64` 183 MiB, `BoundaryConditions._StageCombine{Float64,Float64}` 94 MiB, `FiniteVolume.GridMetrics{…UniformZ…}` 41 MiB, `PrognosticSGS._FaceFlux{…}` 9 MiB.
>    Boxed index tuples, ranges and floats per row mean the call to `@noinline _row!` is dynamically dispatched for some kernels: the "specialisation trap" the comment above it describes. Candidates: kernels with more than 24 arguments (they fall to the `Vararg{Any,N}` method at `Backends.jl:405`), or an `args` tuple whose type isn't concrete. The parameter types point at the open-boundary stage combine, the grid-metrics bodies and the SGS face fluxes.
>    The stack inside a thread ends at the kernel, so the profile does not name the launch. To find them: a small grid with the launch trace (`Backends.set_launch_trace!`) and `@allocated` per launch name, or `@code_warntype`/JET on `_rows3_kernel!` for those bodies.
> 2. **~236 MiB per step in `open_boundary_snapshot`** (`src/boundary_conditions/open_boundary.jl:1198`): `[copy(p.data) for …]` allocates every plane each step. `step_rk3!` could keep one snapshot buffer and `copyto!` into it.
>
> **Expected gain:** no GC pauses while stepping at 20 m, and less time per row in the affected kernels. To be measured.
>
> **Check:** `@allocated Aeolus.step!(model, dt)` after warm-up on a small open-boundary case, as a test with a near-zero bound, plus the 20 m step time on Fritz.
>
> Related: #566.

### 5.4 Issue 4: filed as #575 (2026-11-14; the fix chosen is option 4, serial FFTW plans on Julia's threads): pinning with `JULIA_EXCLUSIVE=1` puts FFTW's threads and thread 1 on one core

**Title:** `JULIA_EXCLUSIVE=1` serialises the step: helper threads inherit thread 1's pin to core 0 (28.4 s/step pinned against ~6 s unpinned at Berlin 20 m, 72 threads)

**Body:**

> **Observed** (#566, Fritz, Berlin 20 m, 1000 × 1000 × 150, `-t 72,0`, FFTW `estimate` on 72 threads; job 4471418, `benchmarks/hpc/results/scaling_20m_rerun2/scaling_20m_t72_all.toml`, `[affinity]` and `[schedstat]`, 295 s of stepping measured):
>
> | threads of the process | count | pinned to | on a CPU | waiting in the run queue |
> | :-- | --: | :-- | --: | --: |
> | Julia thread 1 (main) | 1 | core 0 | 52 s | **194 s** |
> | Julia threads 2–72 | 71 | cores 1–71, one each | ~29–32 s each | < 0.3 s each |
> | helper threads present at startup (GC threads, presumably) | 71 | core 0 | 0 s | 0 s |
> | threads created after startup (FFTW's own pool, #397, and other lazily started pools) | 99 | not recorded; inherited from their creator | 227 s in total | **15 124 s in total** |
>
> - With `JULIA_EXCLUSIVE=1`, Julia pins thread *i* to CPU *i − 1*. A thread created later by native code inherits the affinity of the thread that creates it (Linux `pthread_create` semantics).
>   - FFTW's own pool (`_fftw_own_threads!`, #397) is created on the first threaded transform, from thread 1, so all its workers share core 0 with thread 1.
>   - The startup helpers are pinned to core 0 too, so a parallel GC would run on one core.
> - Because every kernel launch waits for thread 1, the whole step slows down, not only the FFTs:
>   - 28.4 s/step pinned at 72 threads, flat from about 4 threads on (38.7 s at 4, 30.8 s at 16, 55.1 s at 1);
>   - **5.9–6.2 s/step unpinned** (job 4471714).
> - The same effect is visible on the 12-core development machine (`docs/src/design/ka_port.md`, "Thread pinning"): `-t 4,0` with `JULIA_EXCLUSIVE=1` was 18 % slower than unpinned for the Berlin example. That page concludes "pinning does not help", but doesn't name this mechanism, and it presents `JULIA_EXCLUSIVE=1` as a valid layout.
> - Run 1 of #566 (`node.sh`) set `JULIA_EXCLUSIVE=1` too, so its scaling curve carries the same caveat.
>
> **Code.** Aeolus never sets affinities:
> - `Backends.init_threads!` sets the thread counts of BLAS and FFTW only;
> - `_fftw_own_threads!` removes FFTW.jl's callback, so FFTW creates its own pthreads, which inherit the caller's mask.
>
> FFTW.jl's callback (Julia tasks, so the pinned Julia threads) would avoid the inheritance, but it was removed for the #397 crashes and is 2× slower at 20 m (#572 report).
>
> **Options:**
> 1. **Warn** in `Backends.init_threads!` (Linux) when thread 1's affinity mask is a single CPU while `Threads.nthreads() > 1` or FFTW threads > 1 (`sched_getaffinity` on thread 1, or `/proc/self/task/<tid>/status`, `Cpus_allowed_list`). The warning would say that helper threads (FFTW, GC) inherit that CPU, and recommend external binding (`srun --cpu-bind=cores` on the process) or no pinning.
>    Cost: ~20 lines and a test with a narrowed mask. No behaviour change.
> 2. **Document** in `docs/src/design/ka_port.md` ("Thread pinning") and `docs/src/benchmarks.md` that `JULIA_EXCLUSIVE=1` must not be combined with FFTW threading (or GC threads), with the numbers above.
> 3. **Fix the masks:** in `init_threads!`, before FFTW's pool exists, widen the calling thread's mask to the union of the Julia threads' masks (`sched_setaffinity` via `ccall`, Linux only), run one small threaded transform so FFTW creates its workers with that mask, then restore thread 1's pin.
>    - The same could widen the masks of helper threads already present, by tid from `/proc/self/task`.
>    - Cost: ~40 lines of OS-specific code, a Linux-only test, and a Fritz check. It fixes FFTW but not every library that starts threads later.
>
> 4. **Don't let FFTW start threads: PALM's method.**
>    - Plan every transform single-threaded (`FFTW.set_num_threads(1)` while planning) and run the plans on Julia's own threads, each thread taking a block of levels (or of lines) through `launch!`.
>    - This is what PALM does. It never calls `fftw_init_threads` or `fftw_plan_with_nthreads`; each OpenMP thread runs a serial 1-D plan on its own lines (`fft_xy_mod.f90:820-897`, PALM 25.10.1), and it parallelises mainly with MPI processes, one per core, pinned by the launcher.
>    - **Effect:** there's no FFTW pool, so nothing inherits thread 1's pin, and every FFT runs on the threads the user placed. It also removes the reason for `_fftw_own_threads!`: there's no FFTW.jl callback either, so the #397 crash can't occur. And FFTW's planner can no longer decide how to thread a transform, the failure behind #572.
>    - **Speed, measured** (the forward and inverse 2-D transform pair at 1000 × 1000 × 150, `solver_compare.jl`):
>      - Fritz, 72 threads (`solver_compare_t72.txt`): **0.326 s** with serial plans on Julia threads over levels, against 0.353 s with FFTW's own threads.
>      - Development machine, 24 threads: 0.60 s against 0.62 s.
>    - **Where:**
>      - the x-y plans of `ColumnTridiagonalSolver`, which serves `CosineFFTPoissonSolver` with walls in z, `StretchedPoissonSolver` and `DirectWeightedPoissonSolver`: one serial 2-D plan applied per level;
>      - `_CosineTransformSolver` (periodic z) and the triply periodic `FFTPoissonSolver` also need their z transform split into blocks of columns, each with a serial 1-D plan;
>      - the FFT diagnostics in `Operators`.
>
>      A plan made for one level can be reused on other levels only if they are aligned the same way. Levels of an Nx × Ny slab of Float64 are 16-byte aligned when Nx·Ny is even; otherwise plan with `FFTW.UNALIGNED`, or keep one plan per alignment class.
>    - **Cost:** a moderate change in `PressureSolvers` and the FFTW policy; the FFTW thread count would lose its meaning (keep it as 1, or remove `[fftw] threads`); results change at round-off; tests for agreement with the current solves.
>    - It doesn't touch the GC threads. They were pinned to core 0 too, but used no CPU during stepping here. Options 1 and 2 cover them.
>
> **Proposal:** 1 and 2 now; 4 as the fix, rather than 3: it removes the hidden pool instead of re-pinning it, and it was not slower in either measurement. 3 only if FFTW's own threads are kept.
>
> **Check:** the 20 m step at `-t 72,0` with `JULIA_EXCLUSIVE=1` should equal the unpinned ~6 s (after 3), or the warning should appear (after 1).
>
> Related: #566, #397, #572.

### 5.5 Where the step loses threads at 72 threads: filed as #581 (2026-11-15)

The full measurements and every candidate (21 items: serial code outside launches, launches on one thread or a few, partly idle launches, allocation and dispatch, placement) are in **`REPORT_threads_20m.md`**. Filed as **#581** (milestone Speedup), each item checked against master's code; corrections to the report are marked ⚠ there (among them: the blend's cause is 3-point work items along i, not the serial threshold; Berlin's fused sweep has lz = 1, so master makes 72 slabs of 2–3 planes, not 37). Related PRs: #577 (fixes #574, merged), #580 (fixes #575), #579 (the fused sweep split in j).

## 6. Open items

1. **Issue 1:** filed as #567 (§5.1); fixed by #571, confirmed on Fritz, closed 2026-11-14.
2. **Issue 2:** filed as #568 (§5.2, maintainer agreed 2026-11-12); fixed by #569 and #570, confirmed on Fritz, closed 2026-11-14.
2. **Issue 3:** filed as #574 (§5.3, 2026-11-14): ~1.2 GiB allocated per step at 20 m. Fixed by #577 (merged 2026-11-15; 50 m: 151 → 1–3 MiB per step); confirmed on Fritz 2026-11-15 (20 m: 0.0086 GiB per step, no GC; `REPORT_after_581.md`); closed 2026-11-15.
2. **Issue 4:** filed as #575 (§5.4, 2026-11-14): `JULIA_EXCLUSIVE=1` puts FFTW's threads and thread 1 on core 0. Chosen fix: serial FFTW plans run on Julia's threads (option 4), PR #580; confirmed on Fritz 2026-11-15 (pinned 6.12 s/step at 72 threads, unpinned 6.08); closed 2026-11-15.
2. **#572** (the pressure-solve stall): fixed by #573, confirmed on Fritz, closed 2026-11-14.
2. **Thread utilisation at 72 threads:** filed as **#581** (§5.5, 2026-11-15) from `REPORT_threads_20m.md`. To measure on Fritz after #577, #580, #579 and the #581 PR: `bench_run.jl profile` and the unpinned ladder at 20 m and 50 m.
0. **#584 candidates for #581 A2 measured on Fritz (2026-11-16, job 4474794): `REPORT_after_584.md`, `results/after_584/`.** Builds `bench/master-581` (f6e8b10c, master 22f644f7 + trace), `bench/584-swap` (53059213), `bench/584-copy` (0fbd1140); Berlin 20 m unpinned. s/step at 72 threads: master 5.65, swap 4.70 (−16.7 %), copy 4.85 (−14.2 %). At 36 threads: master 6.65, swap 6.02, copy 5.95. Swap wins at 72 by 0.14 s; copy wins at 36 by 0.07 s. The main-thread copy (~1.1 s per step) is gone with swap and ~0.07 s with copy. The out-of-place sweep shows no cost in the trace. Allocation is unchanged at 9 MiB per step. Swap's `loop_fusion = never` path is 26 % slower (9.65 against 7.67 s at 36 threads). The system images were not built (`SYSIMG_CFG` unset in the job; fixed in `results/after_584/a2.sh`). Issue comment (local agent, 2026-11-16): https://github.com/badulion/Aeolus.jl/issues/584#issuecomment-6915150256. The unfused regression was not reproduced locally: with fusion off the swap build makes the same 5 copies per step as master; Berlin 100 m at 12 threads 0.252 against 0.255 s/step, and moist 1006×1006×156 at 24 threads 13.75 against 13.53 s/step. After 21 Berlin 100 m steps swap and copy match master exactly. A Fritz profile with `never` at 36 threads (swap against master, plus `numastat -p`) would decide it. #581's A2 item points to the comment. Job 4477000 (trace, f0828, 36 threads, `never`): swap 8.26 against master 7.72 s/step (+7 %). Serial time and launch counts are unchanged, and the extra time is spread over all the memory-bound launches. Reading: memory placement, not the code path (the size varies by run, 7–26 %; unconfirmed). Fused is unaffected (swap −0.76 s). No change to `issue/584-rk3-swap`; both jobs ran `never` after 2 fused warm-up steps, on the original arrays. Follow-up comment: https://github.com/badulion/Aeolus.jl/issues/584#issuecomment-6921757694. Placement check, job 4481036 (f0313, 4 NUMA nodes, `numa_balancing = 1`; swap 27b12c34 with e918e33e): with fusion off, swap against master is −4 % unpinned, −12 % pinned (one socket) and +1.6 % interleaved. This node shows no gap, so the earlier +7–26 % is run-to-run page-placement variation, not a cost of the swap. Fused: swap faster in every variant (−0.79 s unpinned, −0.84 s interleaved). `REPORT_after_584.md` § Placement check, `results/after_584_placement/`. Follow-up comment: https://github.com/badulion/Aeolus.jl/issues/584#issuecomment-6976750556. Since job 4477000, `issue/584-rk3-swap` has e918e33e, which fixes nested tracers' open-boundary bindings and was found by the full test suite. It is not a timing change.
0. **Master after #576–#582 measured on Fritz (2026-11-15, jobs 4472316–4472319, `bench/master-581` b188b88a): `REPORT_after_581.md`, `results/after_581/`.** 20 m unpinned: 6.08 s/step at 72 threads (bench/all 6.10, which had group D), 7.35 at 36, so 36 → 72 is 1.21× (was 1.05×). Pinned at 72: 6.12 s (28.37 before), run-queue wait 0.9 s (15 124 s before): #575 confirmed. Allocation 0.0086 GiB per step (1.22 before), no GC: #574 confirmed. Fused sweep 4.73 → 2.25 s per 2 traced steps, busy 37.6 → 49.9: #558/#579 confirmed. Outside launches 25 % of the traced step (31 % before). Remaining serial work: `copy_storage!` ~1.0 s per step, whole-field `fill!` in `advance_sgs!` ~0.55 s (new), the serial part of the pressure solve ~0.4 s. Issue comments posted 2026-11-15:
   - #574: https://github.com/badulion/Aeolus.jl/issues/574#issuecomment-6892495379
   - #575: https://github.com/badulion/Aeolus.jl/issues/575#issuecomment-6892496516
   - #558: https://github.com/badulion/Aeolus.jl/issues/558#issuecomment-6892497677 (#574, #575 and #558 closed by the maintainer's decision, 2026-11-15)
   - #581: https://github.com/badulion/Aeolus.jl/issues/581#issuecomment-6892498789

   #581's checklist is updated (B1, B2, B3, A3 and A4 ticked; A2 is the largest item; new A7 and C5; a ranked list). The comment corrects the report on two points, both checked in master's code:
   - `fill!(c.Pw)` at `PrognosticSGS.jl:867` is not needed every step without immersed walls (Berlin: no samples in the wall path). `fill!(c.tend)` at `:964` is needed, since the advection and the diffusion add into it.
   - `_dual_strain_row!` is at 23 busy because of load imbalance: KA's static split of 150 000 one-row items, with all-checked rows on the first and last k-plane, costing 4.7× a bulk row (10.4× for j-boundary rows, measured locally); the cost model gives ~27.6 of 72. It is neither the launch's shape nor `launch_weighted!`.
0. **20 m rerun after #572 (2026-11-14): the ladder ran; setup and the pressure solve are fixed, the steps don't scale.**
   - **Builds:** bench/all 51154519 and this branch, both with master 4f61b8a6 (#567, #568, #572) merged in. Job 4471418 (`results/scaling_20m_rerun2/`, `ladder.sh`), `scaling.sh` 72 36 16 8 4 2 1, `-t N,0`, `JULIA_EXCLUSIVE=1`, FFTW `estimate` on N threads. The earlier test job (4468996, 3-D DCT stall) was cancelled at 01:11:36 by SIGTERM with no TOML.
   - **Results** (s/step is the median of the `if_possible` steps; "busy cores" and "queued" are from `/proc/self/task/*/schedstat` over the measured steps, summed over the process's OS threads and divided by wall time):

     | threads | build s | s/step | speed-up | efficiency | peak RSS GB | busy cores | queued threads |
     | --: | --: | --: | --: | --: | --: | --: | --: |
     | 1 | 159 | 55.1 | 1.00 | 100 % | 71.0 | 1.00 | 0.0 |
     | 2 | 159 | 51.6 | 1.07 | 53 % | 70.7 | 1.44 | |
     | 4 | 146 | 38.7 | 1.42 | 36 % | 71.0 | 2.11 | |
     | 8 | 136 | 33.9 | 1.62 | 20 % | 72.0 | 2.93 | |
     | 16 | 128 | 30.8 | 1.79 | 11 % | 73.9 | 4.34 | 18.5 |
     | 36 | 132 | 28.7 | 1.92 | 5 % | 86.6 | 6.10 | |
     | 72 | 138 | 28.4 | 1.94 | 3 % | 89.2 | 8.15 | 52.0 |

   - **#567 confirmed on Fritz:** build 128–159 s at every thread count, against ~2580 s in attempt 1. Build GC share 9–20 %, 86.6 GB allocated.
   - **#568 confirmed:** each process raised exactly 1 tile (column (30, 615), h = d = 9.96 m, z0m = 0.05, 0.039 → 20.04 m), no `solve_surface_layer` error.
   - **#572 confirmed:** during stepping at 72 threads, 1–2 % kernel time and ~700 context switches/s (`procwatch.log`), against 82 % and 656 k/s before. The pressure solve alone at 1000 × 1000 × 150 on 72 threads: 0.44 s, agreement with the 3-D DCT 5e-16 (`solver_compare_t72.txt`). Three solves per step ≈ 1.3 s of 28.4 s, ~5 %.
   - **Why the steps don't scale: the benchmark's pinning, not the model (confirmed 2026-11-14).** With `JULIA_EXCLUSIVE=1` the threads created after startup inherit thread 1's affinity, core 0. At 72 threads the TOML's `affinity` shows 72 helper threads (probably GC) already pinned to core 0 at startup; the 99 threads created later (FFTW's own pthreads, #397) spent 227 s on a CPU and **15 124 s waiting in the run queue** over the 295 s measured: the FFTW pool ran on one core. At 16 threads: 5 748 s queued. Unpinned (memory job 4471714), the same 20 m step at 72 threads takes **5.9–6.2 s against 28.4 s** pinned; 50 m 1.0 s. Run 1's `node.sh` sets `JULIA_EXCLUSIVE=1` too, so its scaling curve (plateau at ~16 threads) is affected the same way and should be read with this caveat.
   - **The growing RSS is garbage, not a leak** (job 4471714, `results/memory_20m/`): 1.22 GiB allocated per step at 20 m; the live heap returns exactly to its baseline after a full GC; a collection at 72 threads costs 1.7 s. ~1.15 GiB of it is boxed arguments per kernel row at `Backends.jl:356`, ~0.24 GiB a fresh `open_boundary_snapshot` per step. Filed as #574 (§5.3).
   - **Profile job 4471709 failed:** `Profile.print` hits the Fritz ENODEV bug (§7), and the fallback had a soft-scope bug (fixed in 559013ad). Raw samples kept in `$WORK/aeolus-bench/results/profile_20m_t72/*.jls`; no step times from it.
   - **Unpinned ladder** (job 4471876, `results/scaling_20m_unpinned/`, `BENCH_PIN=0`), stopped after 8 threads for the serial finder: 72: 6.10 s/step, 36: 6.40, 16: 9.06, 8: 16.5 (build 119–138 s, peak RSS 71–82 GB). 4, 2, 1 threads not run.
   - **Serial finder** (job 4471898, `results/serial_t72/`, `bench_run.jl profile`, 72 threads unpinned): at 20 m 31 % of the step is outside launches (FFTW transforms, `copy_storage!`, radiative cooling, open-boundary bookkeeping) and the open-boundary blend runs on 0.5 threads (16 %). Every place the step loses threads: `REPORT_threads_20m.md` (§5.5), for the local agent to decide on.
   - Not done: the bench/all system image (only `node.sh` uses it) and the pressure-solver tests on Fritz.
3. **Before rerunning the 20 m ladder:** a fix for issue 2, and ideally issue 1 too, since the ladder spends about 5 h on setup.
   - `scaling.sh` builds the model in every process (7 builds) and uses no system image.
   - The maintainer's stop rule (setup over ~90 min or peak memory over ~230 GB at 72 threads) held this time: setup took 43 min and memory peaked at 69 GB.
4. **The 2-thread anomaly** (run 1) is still unverified; `-t N,0` is the proposed fix.
   This can be checked quickly on 100 m with `scaling.sh` pointed at the 100 m config, or with `node.sh` after changing its `julia -t` to `-t N,0`.
5. **`state_compare.jl`'s "10 × spread" criterion is unusable:** master is bit-identical across 17 and 18 threads, so the spread is 0.
   Run 1 reported the sanity check as "B, D, E bit-identical to master; every #543 build differs from master by the same amount" instead.
   Rewrite it as agreement against only-543 within a stated tolerance.
6. **`kernel_fixes/node17.sh` doesn't use the system images** (it fit its time limit without them).
7. **No PR for #566 yet.** When the maintainer wants one: base master, from `issue/566-hpc-benchmark`.

## 7. Working on Fritz: what bit us (also in memory notes)

- **Nothing computational on the login node.** Use `sbatch`/`srun -p singlenode --exclusive`. Post-processing in Python is fine when the user says so.
- **Every partition is `OverSubscribe=EXCLUSIVE`:** any job gets, and is billed for, a whole 72-core node.
  Pack several small tasks into one job, don't spend a node on one test suite, and ask before submitting more than a couple of node jobs.
- **`--constraint=hwperf`** is required for LIKWID. Use `likwid-perfctr -c` (measure only): `-C` pins the process and breaks `JULIA_EXCLUSIVE` at 72 threads.
  `likwid-perfctr -t` writes no column header into the timeline file; `summary.py` reads the rows by position (LIKWID 5.5 ICX group files).
- **`module load likwid` must not run in a pipe or subshell:** the PATH change is lost.
- **Compute nodes have no `/usr/bin/time`:** use `timev.py`.
- **Watch a running job** with `srun --jobid=<id> --overlap -N1 -n1 <cmd>`. `pgrep -f <script>` also matches the `bash -c` wrapper itself; use `pgrep -n -x julia`.
- **`kill -USR1 <julia pid>`** prints all thread stacks at once, but its profile report only at the main thread's next yield point.
- **Depot:** `$WORK/aeolus-bench/depot` with `JULIA_MAX_NUM_PRECOMPILE_FILES=100` (the default 10 thrashes across the 18 worktrees).
  System images are in `$WORK/aeolus-bench/sysimages/<build>.so` with a `.commit` sidecar; `node.sh` uses one only if it matches the worktree HEAD.
- **`du` on `/home/hpc` counts allocated blocks:** a 19 KB TOML showed as 32 MB. Use `du --apparent-size`.
