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
