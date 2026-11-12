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

## 6. Open items

1. **Issue 1:** file it (§5.1).
2. **Issue 2:** ask the maintainer whether to file it (§5.2). Until the surface-layer failure is fixed, or the case changes, the 20 m scaling test can't run.
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
