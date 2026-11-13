# The pressure solve's FFT is pathologically slow on Berlin 20 m at 72 threads (#566 rerun, 2026-11-13)

Short report for the agent researching this. Nothing has been changed for it yet.

## Setup

- **Build and case:** bench/all 0b498671 (origin/master 366d67d3 merged in: #567, #568). Berlin 20 m, 1000 × 1000 × 150 cells, open lateral faces (`[boundaries] x = y = "open"`).
- **Run:** `scaling.sh` rung at `-t 72,0`, Fritz node f0314 (2 × 36 Ice Lake cores), job 4468996.
- **FFTW:** planner `estimate` (`AEOLUS_FFTW_PLANNER`), no `[fftw] threads` key, so FFTW gets `Threads.nthreads()` = 72 threads (`Backends.init_threads!`). The transforms run on FFTW's own pthreads, because Aeolus removes FFTW.jl's task callback (`_fftw_own_threads!`, #397).

## Observed

- **Setup is fixed.** The build finished in about a minute, against ~43 min in attempt 1. The #568 warning (1 raised tile) came early, and the steps started without the `solve_surface_layer` error.
- **The steps don't finish.** 12 steps (2 warm-up + 2 × 5) were expected to take about 3 min. They hadn't finished after 55 min.
- **Stack sample** (`kill -USR1`) during stepping:
  ```
  steps! → step! → _step_rk3! → _rk3_substep! → _euler_projection! → _try_fused!
   → _blocked_projection! (fused_sweep.jl:427) → _project_velocity! (Simulations.jl:706)
    → solve_source_buffer!(…, ::CosineFFTPoissonSolver) (PressureSolvers.jl:435)
     → apply_transform! → FFTW mul!            ← the forward DCT
  ```
  With open faces in x and y and Neumann in z, the solver is `CosineFFTPoissonSolver` with all three axes Neumann:
  1. one 3-D `REDFT10` (DCT-II) over the whole 1000 × 1000 × 150 array;
  2. scale by the eigenvalues;
  3. one 3-D `REDFT01` back.

  No FFT plus tridiagonal solve in z.
- **The process during stepping** (`/proc` deltas over 10 s, via `srun --overlap`):

  | | |
  | :-- | :-- |
  | busy cores | 1.19 in total, across 51 threads |
  | kernel share | 82% (980 of 1190 ticks), user 18% |
  | context switches | ~656 000 per second (voluntary, summed over threads) |
  | page faults | none (minor and major) |
  | memory | 82–84 GB resident, stable |

  The threads spend their time sleeping and waking, not computing, which looks like FFTW's thread pool synchronising per sub-transform. Unverified.
- **For comparison:** run 1 (100 m and 50 m, 72 threads) had normal step times (0.26 s and 0.86 s for bench/all). Its configs' `[fftw] threads = 12` was never applied (`bench_run.jl` never called `apply_fftw_config!`; fixed in 0ba54820), so FFTW probably ran on 72 threads there too, without this collapse. Run 1's TOMLs don't record the FFTW thread count; new runs record `fftw_planner` and `fftw_threads`.

## Questions to research

1. **Is it the FFTW thread count?** Time the 20 m `solve_source_buffer!` alone at FFTW threads 1, 8, 18, 36 and 72, with 72 Julia threads, on a compute node.
2. **Is it FFTW's own pthreads pool** (callback removed, #397) against FFTW.jl's Julia-task callback? And `estimate` against `measure` for this shape. Print the chosen plan (`fftw_sprint_plan`) for the 3-D `REDFT10` at 72 threads.
3. **Is a single 3-D r2r over 150 M points a sensible decomposition?** Compare it with batched lower-dimensional transforms:
   - 2-D DCTs per level plus a tridiagonal solve in z (`ColumnTridiagonalSolver`, which already exists);
   - 1-D transforms along x and y, threaded over lines.
4. **Why does 50 m (9.6 M points) not show it,** and at what size does it start?

## Compare with PALM

Check PALM's source before relying on these points; they are from memory.
- **The FFT solver** (`psolver = 'poisfft'`):
  - FFTs along x and y, then a tridiagonal solve in z (`tridia_solver_mod`);
  - transposes between the decompositions (`transpose_mod`);
  - 1-D transforms over lines (`fft_xy_mod`: FFTW, Temperton or Singleton), with OpenMP threading over the other index, so each thread does whole lines;
  - parallelism across processes through MPI transposes, not through one threaded multi-dimensional transform.
- **Non-cyclic lateral boundaries.** As far as I recall, PALM's FFT solver needs cyclic lateral boundaries, and non-cyclic ones (e.g. open faces from a mesoscale nest) use the multigrid solver (`psolver = 'multigrid'`). If so, PALM never runs a DCT pressure solve on an open domain.
- **What to compare with Aeolus's choices:**
  - for the open-face case: a 3-D DCT (Aeolus) against multigrid (PALM);
  - the thread decomposition of the transforms: FFTW-internal threads against threads over lines;
  - the cost per point at 150 M points.

  Files to read: `poisfft_mod.f90`, `fft_xy_mod.f90`, `tridia_solver_mod.f90`, `transpose_mod.f90`, `poismg_mod.f90` / `multigrid` in the PALM source tree.

## Records

- **This run's log** (with the stack sample): `$WORK/aeolus-bench/results/scaling_20m_rerun_test/scaling.log`.
- **The job log:** `$WORK/aeolus-bench/scaling20/test72_4468996.out`.
- **No TOML yet:** `bench_run.jl` writes only at the end of a run.
- **`kill -USR1` on Fritz** prints the thread stacks, but the profile report then fails in Julia's `Profile` stdlib (`ispath("/proc/sys/fs/binfmt_misc/WSLInterop")` → ENODEV). This only kills the report task; the run continues.

## Findings (2026-11-13, measured on the development machine)

**Setup.**
- **Machine:** AMD Ryzen AI 9 HX 370, 12 cores / 24 threads, one NUMA node, 125 GB. Julia 1.13.3, FFTW 3.3.11 (FFTW_jll), `-t 24,0`.
- **Code:** this branch at ff7e166d (master 366d67d3 merged).
- **Scripts and raw output:** `benchmarks/hpc/fft_20m/`.
  - `dct_transform.jl` times the solver's `plan_r2r(…, REDFT10, dims)` on its own. It records the time per transform, cores busy, kernel share, context switches per second and OS threads (from `/proc`), and writes the plan (`fftw_sprint_plan`) to `plans/`.
  - `solver_compare.jl` times the two solvers on a `RegularGrid`.
  - `results/*.txt` holds every run.
- **Size:** the full 20 m array (1000 × 1000 × 150, 1.2 GB per buffer) fits here, so nothing had to be scaled down. What this machine can't do is 72 cores; the 72-thread `estimate` plan was printed without being executed (see "Cause").
- **Caution with `dct_transform.jl`:** calling `fftw_threads_set_callback(C_NULL)` with a bare `ccall` before any other FFTW call did **not** take effect: timings and the OS thread count were identical to FFTW.jl's callback. Going through Aeolus's policy (`set_fftw_policy!` → `_fftw_own_threads!`) does take effect: +43 OS threads at 24 FFTW threads. All "own pool" numbers below use Aeolus's path.

### Measured

**1. FFTW thread count, planner, and thread mechanism.** 3-D `REDFT10` at 200 × 200 × 150 (6 M points, the same pathology as 20 m at 1/25 of the cost), 24 Julia threads. Time per transform:

| FFTW threads | estimate, own pool (Aeolus) | measure, own pool | estimate, Julia tasks (FFTW.jl callback) | measure, Julia tasks |
| --: | --: | --: | --: | --: |
| 1 | 0.050 s | 0.043 s | 0.055 s | 0.044 s |
| 2 | 0.049 s | 0.027 s | 0.032 s | 0.026 s |
| 4 | **0.624 s** | 0.015 s | **3.92 s** | 0.017 s |
| 12 | **2.18 s** | 0.012 s | **5.83 s** | 0.011 s |
| 24 | **4.64 s** | 0.011 s | **9.97 s** | 0.008 s |

- With `estimate`, FFTW's own pool at 24 threads uses 5.0 cores, with 71% kernel time and 1.18 M context switches per second. That is the Fritz signature (1.19 cores, 82% kernel, 656 k/s).
- FFTW.jl's Julia-task callback uses 23 cores, almost all in user time: Julia's scheduler spins. It is 2× slower still.
- `measure` never collapses. It plans in 2.3–5.6 s.

**2. Grid size: it is Nz, not the number of points.** `estimate`, own pool. Nanoseconds per point:

| Nx = Ny | Nz = 60, 1 thread | Nz = 60, 24 threads | Nz = 150, 1 thread | Nz = 150, 24 threads |
| --: | --: | --: | --: | --: |
| 100 | 6.3 | 2.6 | 7.1 | **780** |
| 200 | 7.8 | 1.7 | 9.0 | **775** |
| 400 | 10.3 | 2.4 | 13.2 | **783** |
| 600 | 9.4 | 2.5 | 10.4 | **799** |
| 800 | 17.8 | 3.4 | 18.6 | **788** |
| 1000 | 10.8 | 2.8 | 11.5 | **788** |

- At Nz = 150 the cost per point is the same ~780 ns at every horizontal size from 100² (1.5 M points) up. So there is no size threshold.
- At Nz = 60, every size scales normally.
- **Question 4:** the 100 m and 50 m cases are 200 × 200 × 60 and 400 × 400 × 60, so they never hit this. The 20 m case is the first with Nz = 150.

**3. The 20 m transform itself (1000 × 1000 × 150).**

| | 1 thread | 24 threads |
| :-- | --: | --: |
| `estimate` | 1.73 s | **118 s** |
| `measure` | 1.38 s (plan 8.2 s) | 0.47 s (plan 15.2 s) |
| `estimate`, Julia-task callback | | over 20 min for 1 warm-up + 1 timed transform (stopped by the timeout) |

**4. The periodic channel is affected too.** Its only DCT is along z (`DIMS=3`): 200 × 200 × 150, `estimate`, 24 threads.

| Nz | 1 thread | 24 threads |
| --: | --: | --: |
| 150 | 0.024 s | **4.75 s** (791 ns/point) |
| 60 | 0.007 s | 0.002 s |

With `measure`, the z-only DCT at 20 m takes 0.21 s on 24 threads.

**5. A full step at 20 m.** One step does three pressure solves (one per RK3 substep, `Simulations.jl:856-870`), and each solve does two transforms.
- At 24 threads, the 3-D DCT solve took 243 s with `estimate` (table below). That is about 12 min per step, against about 3.5 s with `measure`.
- The 12-step test on Fritz therefore needs more than 2 h at this rate, which fits "not finished after 55 min".

### Cause

**The plan.**
- With `FFTW_ESTIMATE` and 4 or more threads, FFTW plans the z-axis part of the DCT (`redft10e-r2hc-150-x1000000`) as a **serial loop over the 1 000 000 columns**. Each column's 150-point transform is threaded on its own (`rdft-thr-ct-dit-x16/5` plus threaded sub-loops; see `fft_20m/plans/1000x1000x150_ft24_estimate_aeolus.txt`, line 1696 on).
- So every column of every transform hands work to the threads and waits for it: millions of hand-offs per transform.
  - On FFTW's own pool, each hand-off wakes the workers through semaphores, which gives the kernel time and the context-switch rate.
  - On Julia tasks, each hand-off spawns and syncs tasks, and the scheduler spins.
- At Nz = 60, or with 2–3 threads, the same planner splits the columns between the threads instead (`rdft-thr-vrank>=1-x24` around `redft10e-r2hc-60-x41667`; at 2 threads `rdft-thr-vrank>=1-x2` around `redft10e-r2hc-150-x500000`).
- `measure` times the candidates and picks the column split for Nz = 150 too (`200x200x150_ft24_measure_aeolus.txt`, line 904 on).

**The 72-thread plan of the Fritz run.** `estimate` makes the same plan on any machine, so it could be printed here without executing it (`*_planonly.txt`):
- at 72 and at 36 FFTW threads, Nz = 150 gives the same pathological structure, `redft10e-r2hc-150-x1000000` → `rdft-thr-ct-dit-x16/5`;
- at 72 threads, Nz = 60 gives `rdft-thr-vrank>=1-x72` (fine).

**Ranked conclusion.**
1. *Confirmed:* FFTW's `estimate` planner threads inside each 150-point z transform instead of across the columns. This happens whenever FFTW has 4 or more threads and Nz = 150, under both thread mechanisms.
2. The thread mechanism only changes how the waste shows: kernel/semaphore time with Aeolus's own pool (Fritz), spinning user time with Julia tasks. Neither is the cause. FFTW.jl's callback is 2× worse here, and it was removed for crashes (#397).
3. The thread count matters only through the plan: 1–3 FFTW threads are safe, 4–72 collapse.
4. Why `estimate` prefers that plan for 150 and not for 60 is FFTW's heuristic. Not investigated further.

### 3-D DCT against 2-D DCT per level plus a tridiagonal solve in z

`solver_compare.jl`:
- `CosineFFTPoissonSolver` (Neumann on all three axes) against `ColumnTridiagonalSolver` (`REDFT10` in x and y, the Thomas algorithm in z, with uniform-z coefficients as in `DirectWeightedPoissonSolver` with unit weights);
- the same random zero-mean source (`Xoshiro(566)`);
- times are the minimum over the repetitions (1 at 1000², 3 otherwise), 24 Julia threads.

| Grid, FFTW threads, planner | 3-D DCT solve | column solve | build: 3-D / column | max rel. difference |
| :-- | --: | --: | --: | --: |
| 200² × 150, 1, estimate | 0.120 s | 0.088 s | 0.70 / 0.36 s | 1.8e-13 |
| 200² × 150, 24, estimate | **9.61 s** | 0.023 s | 0.76 / 0.38 s | 1.8e-13 |
| 200² × 150, 24, measure | 0.036 s | 0.036 s | 8.8 / 0.38 s | 1.8e-13 |
| 1000² × 150, 1, estimate | 3.68 s | 2.72 s | 0.96 / 1.24 s | 1.9e-12 |
| 1000² × 150, 24, estimate | **243 s** | **0.90 s** | 0.95 / 1.24 s | 1.9e-12 |
| 1000² × 150, 1, measure | 3.04 s | 2.27 s | 17.4 / 1.26 s | 1.9e-12 |
| 1000² × 150, 24, measure | 1.15 s | **0.79 s** | 30.7 / 1.24 s | 1.9e-12 |
| 400² × 60, 24, estimate (50 m) | 0.071 s | 0.060 s | 0.75 / 0.45 s | 1.3e-12 |

- **Agreement.** The two solvers discretise the same 7-point Laplacian with the same staggered Neumann condition, and they agree to round-off: 1.9e-12 relative at 150 M points. The difference grows with N as round-off does.
- **Speed.** The column solver is faster under both planners, and it is insensitive to the planner: its x-y transform is threaded over levels (`thr-vrank`) by `estimate` too.
  - At 20 m, 24 threads: 0.79–0.90 s, against 1.15 s for the 3-D DCT even with `measure`.
  - It also builds in 1.2 s, against 31 s of `measure` planning for the 3-D DCT.
- **PALM-style threading.** One serial 2-D plan per level, with Julia threads over levels (`pair2d_juliathreads_s` in the raw output), takes 0.60 s for the forward and inverse pair at 20 m. FFTW's threaded 2-D plan takes 0.62 s. So once z is out of the FFT, FFTW's threading of the 2-D transform is as good as threading by hand.
- **Memory.** The column solver keeps four Nx·Ny·Nz arrays (`buf`, `buf2`, `cp`, `inv_den`); the 3-D DCT solver keeps three (two buffers and 1/λ). That is +1.2 GB at 20 m, on about 83 GB resident.

### PALM (checked in the source, PALM 25.10.1, `packages/palm/model/src`)

The report's PALM section was from memory. Point by point against the source:

- **Decomposition and threading. Confirmed.** `poisfft` (`poisfft_mod.f90:197-303`) does:
  - transpose z→x (`:199`), `fft_x` (`:209`), transpose x→y (`:219`), `fft_y` (`:235`), transpose y→z (`:246`);
  - the tridiagonal solve in z (`tridia_substi`, `:252`);
  - the same steps in reverse back to z (`:301`).

  The transforms are 1-D:
  - With FFTW (`fft_xy_mod.f90:269-305`), one serial 1-D plan per axis is made with `FFTW_ESTIMATE`.
  - It is executed line by line inside `!$OMP PARALLEL` / `!$OMP DO` over (k, j), on thread-private buffers (`fft_xy_mod.f90:116`, `:820-897`). FFTW's own threading is never used.
  - `tridia_substi` threads each level's (j, i) loop with `!$OMP DO`, serial in k (`tridia_solver_mod.f90:297-363`).
  - On one process (`npex = 1`) the transposes are local reorderings, OpenMP-threaded. Across processes they are `MPI_ALLTOALL` (`transpose_mod.f90:313, 546-644`).
- **Solver with non-cyclic lateral boundaries. Corrected: the report's recollection is wrong for this version.**
  - With non-cyclic boundaries, `check_parameters.f90:886-899` allows any `psolver` (not only multigrid), as long as `fft_method` is `fftw`, `temperton-algorithm` or `singleton-algorithm`.
  - `:913-919` requires x and y to be both cyclic or both non-cyclic for `poisfft`.
  - The default is `psolver = 'poisfft'` with `fft_method = 'temperton-algorithm'` (`modules.f90:563, 580`).
  - Neither `nesting_offl_mod.f90` nor `pmc_interface_mod.f90` mentions `psolver`.
  - Multigrid (`poismg_mod.f90`, OpenMP-threaded, e.g. `:480`) is an option, not a requirement.
- **DCT on an open domain. Yes.**
  - For a non-cyclic axis, `fft_init` plans `FFTW_REDFT00` (DCT-I on nx+1 points, with the symmetry about the outermost grid points) instead of the r2c FFT (`fft_xy_mod.f90:286-291, 301-305`). Temperton and Singleton use a 2(n+1)−2-point transform for the same purpose (`:193-195, 245-262`).
  - The tridiagonal solver uses `compute_ll`'s cosine-mode eigenvalues (`tridia_solver_mod.f90:194-201, 231-262`) and normalises by `4·nx·ny` (`:384`).
  - PALM has no DCT in z: z is always the tridiagonal solve.
- **Comparison.**

  | | Aeolus today | PALM |
  | :-- | :-- | :-- |
  | open box | one 3-D DCT-II, z included | 1-D DCT-I in x and y + tridiagonal in z |
  | threading of the transforms | FFTW's planner decides how to thread one big multidimensional transform | the code threads over whole lines, each line on a serial plan |
  | its cost here | with `estimate`, the planner threads inside short z columns | no planner decision is left that could do that |

  Aeolus's `ColumnTridiagonalSolver` already has PALM's structure, apart from DCT-II (cell-centred, Aeolus's staggering) instead of DCT-I.

### Recommended fix and its cost

**Recommended.** When z is Neumann, solve the pressure with transforms in x and y and a tridiagonal solve in z. That means `ColumnTridiagonalSolver` with the uniform-z coefficients, chosen by `Simulations._default_solver` (`Simulations.jl:1051`). It covers the open box and the periodic channel. Only triply periodic domains keep the 3-D FFT; they have no z DCT.
- **Why:**
  - It removes the z axis from FFTW entirely, so no planner choice can serialise the columns.
  - It is faster than the 3-D DCT under both planners: 0.79–0.90 s against 1.15 s (`measure`) or 243 s (`estimate`) at 20 m on 24 threads.
  - Its setup cost doesn't depend on the planner (1.2 s against 31 s).
  - It is PALM's method, and the same class Aeolus already uses for stretched grids (`StretchedPoissonSolver`) and the anelastic core (`DirectWeightedPoissonSolver`).
- **Expected cost:**
  - **Code:** a small change in `_default_solver`, plus the constructor call with uniform coefficients. One option is a convenience constructor `ColumnTridiagonalSolver(mesh::RegularGrid; bc_x, bc_y)`; better, the uniform case can delegate to the existing code, keeping one source of truth.
  - **Results:** they change at round-off (≤ 2e-12 relative measured). Statements and tests that call the channel solver "bit-for-bit the solver chosen before #251" need a tolerance; this is allowed by the speed-over-bit-identity rule.
  - **Memory:** +1 Nx·Ny·Nz array (+1.2 GB at 20 m). This could be avoided by storing the Thomas factors per (kx, ky) only where they vary, which for uniform z is the eigenvalue alone. Not needed now.
  - **Validation:** the tests (`test/run.jl`); `validation/run.jl` for the channel cases and `urban_checks.jl`; and a 72-thread step-time check on Fritz.

**Stopgap for the 20 m rerun, without a code change.** Use `measure`: drop the `fftw.planner="estimate"` override in `bench_run.jl` for this case, or set `AEOLUS_FFTW_PLANNER=measure`.
- Measured: 0.47 s per transform and 1.15 s per solve at 24 threads, with 15 s of planning per plan (31 s for the solver).
- It loses bitwise reproducibility across processes, which the benchmark doesn't need.
- The alternative, `[fftw] threads = 2` with `estimate`, is safe but leaves the transforms on 2 threads.

**Not recommended:**
- Going back to FFTW.jl's callback: 2× worse here, and the #397 crashes.
- Raising the thread count: it makes the collapse worse.

## Draft issue (not filed: the maintainer decides)

**Title:** Pressure solve: tridiagonal in z instead of a DCT along z (FFTW `estimate` serialises the z transform at Nz = 150)

**Body:**

> **Problem.**
>
> - `CosineFFTPoissonSolver` does the z part of the pressure solve with FFTW (`REDFT10` along z, alone in the channel case or inside the 3-D transform in the open box).
> - Under `FFTW_ESTIMATE` with 4 or more FFTW threads, FFTW plans that part at Nz = 150 as a serial loop over all Nx·Ny columns, threading each 150-point transform internally. Every column then costs a thread hand-off.
> - Measured on 24 threads at Berlin 20 m (1000 × 1000 × 150): 118 s per transform and 243 s per pressure solve, against 1.73 s per transform on one thread. That is ~12 min per step.
> - This stalled the #566 20 m benchmark on Fritz (72 threads, 1.2 cores busy, 82% kernel time, 656 k context switches per second).
> - The periodic channel's z-only DCT collapses the same way (790 ns per point against 4 ns).
> - Nz = 60 (the 100 m and 50 m cases) is not affected.
> - `measure` picks a sane plan (1.15 s per solve), but leaves the solver's speed to a planner heuristic, costs 31 s of planning at 20 m, and isn't what the benchmark and test settings use.
> - Details, plans and raw numbers: `benchmarks/hpc/REPORT_fft_20m.md` § Findings, and `benchmarks/hpc/fft_20m/` on `issue/566-hpc-benchmark`.
>
> **Proposal.** When z is Neumann, use `ColumnTridiagonalSolver`: transforms in x and y, the Thomas algorithm in z, uniform-z coefficients. Choose it in `Simulations._default_solver`.
>
> - This is PALM's method: `poisfft` does 1-D transforms in x and y, threaded over lines, then `tridia_substi` in z, also on non-cyclic domains with a DCT-I (`fft_xy_mod.f90:286-305`, `check_parameters.f90:886-899`).
> - Measured at 20 m on 24 threads:
>   - solve: 0.90 s (`estimate`) / 0.79 s (`measure`);
>   - build: 1.2 s;
>   - agreement with the 3-D DCT: 1.9e-12 relative.
> - Triply periodic domains keep `FFTPoissonSolver`.
>
> **Acceptance:**
>
> - `_default_solver` returns the column solver for every Neumann-z case. The 3-D DCT stays available but is no longer the default for those cases (or is removed if nothing else uses it).
> - A test that the new default agrees with `CosineFFTPoissonSolver` within a stated round-off tolerance on a random source, for the open box and for the channel.
> - Tests that assumed bit-identity with the former channel solver restated with a tolerance.
> - Solve time at 1000 × 1000 × 150 quoted for 1 and N threads under `estimate` and `measure`.
> - `validation/run.jl` for the channel cases and `urban_checks.jl` quoted.
> - The #566 20 m single-node test on Fritz at `-t 72,0` reports its step time.
> - Docs: `docs/src/design` (the pressure-solver page) and the roadmap section of this issue.
>
> **Cost:** +1 Nx·Ny·Nz array (+1.2 GB at 20 m). Results change at round-off. A small code change.
>
> Refs #566.
