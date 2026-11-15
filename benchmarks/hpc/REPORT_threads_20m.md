# Where the step loses threads at 72 threads (Berlin 20 m, #566, 2026-11-14)

> **Filed as #581** (2026-11-15, milestone Speedup). The issue checks each item against master's code and corrects this report where it was wrong (marked ⚠ there). Read the issue for the current state.

Report for the local agent. It lists every place the measurements show the step not using the node, with the evidence and the options seen so far. Nothing here has been changed in `src/`. The agent decides what to do, what to file and in which order.

## Setup

- **Build:** bench/all 51154519 = master 4f61b8a6 (with #567, #568, #572) + the bench PRs.
- **Machine:** Fritz, one node, 2 × 36 Ice Lake cores, 4 NUMA domains, 256 GB.
- **Runs:** `-t N,0`, **unpinned** (`JULIA_EXCLUSIVE` unset; pinned runs are distorted by #575), FFTW `estimate` on N threads.
- **Cases:** Berlin 20 m (1000 × 1000 × 150, 150 M points, dt 0.4 s), and 50 m (400 × 400 × 60, dt 1 s) for comparison.

## Measurements

### Scaling, 20 m, unpinned (job 4471876, `results/scaling_20m_unpinned/`)

The ladder was stopped after 8 threads to run the serial finder; 4, 2 and 1 threads were not run.

| threads | build s | s/step `if_possible` | s/step `never` | peak RSS GB |
| --: | --: | --: | --: | --: |
| 8 | 138 | 16.5 | 19.9 | 71.2 |
| 16 | 123 | 9.06 | 10.9 | 73.0 |
| 36 | 119 | 6.40 | 9.25 | 77.8 |
| 72 | 119 | 6.10 | 8.02 | 81.8 |

- 8 → 16 threads: 1.82×. 16 → 36: 1.42×. 36 → 72: 1.05×.
- For comparison, pinned (`JULIA_EXCLUSIVE=1`, job 4471418, `results/scaling_20m_rerun2/`): 28.4 s/step at 72 threads, 55.1 s at 1 thread.

### Threads during stepping at 72 threads (`/proc/self/task/*/schedstat`, from the same TOMLs)

| | on a CPU, share of wall time |
| :-- | --: |
| main thread (thread 1) | 78 % |
| each of the other 71 Julia threads | 37 %, very even |
| ~100 more OS threads (FFTW's pool, GC) | 5–20 % each, ~7.6 cores together |

- 34 busy cores on average. At 36 threads: main 80 %, workers 46 %.
- A thread stalled on memory still counts as on a CPU, so the idle time is real sleeping: the workers have no work ~60 % of the step.
- Run-queue waits are near zero (1.5 threads at 72), so unpinned there is no CPU contention.

### Launch trace and profile (job 4471898, `results/serial_t72/`, `bench_run.jl profile`)

At 72 threads, unpinned: 2 warm-up steps, 2 steps with the launch trace (`Backends.set_launch_trace!`), then 3 steps under `Profile`, split by thread and sleep state from the samples' metadata.

| | 20 m | 50 m |
| :-- | --: | --: |
| traced steps (s) | 6.81, 8.59 | 1.36, 1.57 |
| profiled steps, untraced (s) | 6.52, 6.75, 6.97 | 2.26, 1.14, 1.13 |
| top-level launches / nested launches (2 steps) | 482 / 44 064 | 482 / 40 352 |
| outside any launch (`serial_between_s`) | 4.77 s, **31 %** | 0.72 s, 25 % |
| idle thread time inside launches | 402 of 766 thread-s, 52 % | 115 of 159 thread-s, 72 % |

The trace slows the step by about 15 %. "Busy" below is the trace's busy time inside work items divided by the launch's wall time (threads out of 72); a kernel limited by memory bandwidth also counts as busy.

**Per launch name, 20 m** (2 steps; "lost" = wall × (1 − busy/72), thread time idle inside the launch, in wall-seconds):

| launch | n | wall s | share | busy | lost s |
| :-- | --: | --: | --: | --: | --: |
| `_b_sweep_slab!` | 8 | 3.242 | 21.0 % | 37.6 | 1.551 |
| `_blend_body!` | 120 | 2.409 | 15.6 % | **0.5** | 2.391 |
| `_b_sweep_edge!` | 8 | 1.484 | 9.6 % | 59.5 | 0.259 |
| `_b_correct!` | 6 | 0.541 | 3.5 % | 57.9 | 0.106 |
| `_coefficients_body!` | 18 | 0.497 | 3.2 % | 45.0 | 0.186 |
| `_n2_body!` | 18 | 0.430 | 2.8 % | 48.7 | 0.139 |
| `_thomas_slab!` | 6 | 0.350 | 2.3 % | 64.5 | 0.036 |
| `_xy_item!` | 66 | 0.248 | 1.6 % | 46.8 | 0.087 |
| `plane_reduce/_air_at` | 26 | 0.239 | 1.6 % | 49.5 | 0.075 |
| `_b_divergence_into!` | 6 | 0.232 | 1.5 % | 61.0 | 0.036 |
| `_map_body!` | 80 | 0.203 | 1.3 % | **3.7** | 0.192 |
| `_b_shell!` | 12 | 0.193 | 1.3 % | **4.1** | 0.182 |
| `_b_limit_ratio!` | 12 | 0.155 | 1.0 % | **2.4** | 0.149 |
| `_z_item!` | 66 | 0.152 | 1.0 % | 37.6 | 0.072 |
| `_copy_out_scaled_body!` | 6 | 0.117 | 0.8 % | 52.6 | 0.031 |
| `_b_∂x_cf!` | 6 | 0.059 | 0.4 % | **2.2** | 0.057 |
| `_drag_denominators!` | 6 | 0.052 | 0.3 % | 48.2 | 0.017 |
| `_drag_apply!` | 6 | 0.031 | 0.2 % | 53.9 | 0.008 |
| `_b_∂y_cf!` | 6 | 0.002 | 0.0 % | 1.3 | 0.002 |

**Per launch name, 50 m** (2 steps, 2.94 s), where it differs from 20 m: `_z_item!` 0.825 s (28 %, busy 24.3), `_xy_item!` 0.128 s (4.3 %, busy 11.9), `_b_sweep_slab!` 0.415 s (14 %, busy 18.5), `_blend_body!` 0.247 s (8.4 %, busy 0.8), `_map_body!` 0.155 s (5.3 %, busy 0.8), `_drag_apply!` busy 1.0, `_drag_denominators!` busy 8.1. Full table in the TOML.

**Main thread, 20 m** (`serial_20m_t72_all.profile_threads.txt`: 354 samples awake, 33 asleep; inclusive counts):

| frame | share of main-thread samples |
| :-- | --: |
| inside launches (its own work items, `__thread_run`) | 48 % |
| `solve_source_buffer!` → `apply_transform!` → FFTW `mul!` (`PressureSolvers.jl:377, 669, 683`) | 19.5 % |
| `copy_storage!` (`Simulations.jl:1140`, from `_step_rk3!` lines 853–854) | 15.5 % |
| `_attributed!` (`budgets.jl:250`) / `_run_stages!` / `_run_each` (component pipeline, inclusive) | 4.5 % |
| `apply_radiative_cooling!` (`Radiation.jl:65-72`) via `MeanColumnRadiation` `:surface_flux` (`components/urban/parent/components.jl:1092`) | 3.7 % |
| `correct_boundary_mass_flux!` / `update_open_boundaries!` / `_fold_faces` (`open_boundary.jl:953, 1033-1043, 1132`) | 3.1 % |
| other lines of `_step_rk3!` (`Simulations.jl:856, 858, 866, 873, 879`, including their launches) | 2–3 % each |

At 50 m the main thread got only 85 samples, too few to rank.

## Every place the step loses threads

Grouped by kind. For each: the evidence, what the code does, the options seen. Estimates are for an untraced 6.5 s step at 20 m and are upper bounds until measured.

### A. Serial code outside launches (31 % of the traced 20 m step)

1. **FFTW transforms of the pressure solve.** 19.5 % of the main thread. Three solves per step at ~0.44 s (`results/scaling_20m_rerun2/solver_compare_t72.txt`). They run on FFTW's own thread pool, outside the launch machinery, so the trace counts them as serial. Up to ~1.3 s per step. **Already #575** (option 4 chosen there: serial FFTW plans run on Julia's threads, PALM's method).
2. **`copy_storage!` of the RK3 start state.** 15.5 % of the main thread. `step_rk3!` saves u, v, w, θ (and qv) every step with `copyto!`: a single-threaded memcpy of ~4 × 1.2 GB. The docstring reports no reliable gain from a copy kernel, but measured at 64×64×32 and 128×128×64 only; at 150 M points one core copies ~10 GB/s. Options: a threaded copy above a size threshold; or no copy (swap buffers, or let the stage combination read uⁿ where it already is). ~0.4 s.
3. **`apply_radiative_cooling!`.** 3.7 %. A serial loop `@views tθ.data[:, :, k] .+= r` over all levels: one pass over θ (1.2 GB) on one thread, every step; plus a small allocation in the caller (`c.model.heat .* dt ./ c.π̄`). Options: one launch; or add the profile where the θ tendency is already written. ~0.2 s.
4. **Open-boundary mass-flux correction.** 3.1 %. Serial loops over every boundary face with a closure per face, in two or three passes (flux, area, correction). Options: threaded reductions like `plane_reduce` (exists for `_air_at`), or one fused pass. ~0.1–0.2 s.
5. **Component pipeline and budgets.** 4.5 % inclusive, mostly items 3 and 4 below them. Worth checking whether the budget attribution (`_attributed!`) does serial whole-field work per stage (differences, copies).
6. **The rest of `_step_rk3!`.** Lines 856, 858, 866, 873, 879 at 2–3 % each, launches included. Possible serial pieces between their launches (stage bookkeeping, `combine_open_boundaries!`).

### B. Launches that run on one thread or a few (busy ≤ 4 of 72)

7. **`_blend_body!`, the open-boundary time blend** (`blend_window!`, `open_boundary.jl:291`). 120 launches per 2 steps, 2.41 s, **0.5 threads busy**: 16 % of the traced step, the largest single loss. One launch per field × side × snapshot, each over one thin plane (1 × Ny × Nz or Nx × 1 × Nz). Its rows have length 1, or its point count falls under the backend's threading threshold (`Backends.jl:108`, `npoints < b.threshold`) or `_MIN_POINTS_PER_TASK = 16_384` (`Backends.jl:507`), so it runs serially. Options: one batched launch over all planes of a substep, items (plane, k) or (plane, j-block); or threading along the long axis of each plane. ~1 s.
8. **`_map_body!`.** 80 launches, 3.7 busy (0.8 at 50 m). From `range_map!` / `interior_map!`; the callers are not identified, probably the same thin-plane pattern (the open-boundary stage combine `_StageCombine`, snapshot copies). Same options as 7. ~0.2 s.
9. **`_b_shell!`** (`Simulations.jl:1207`). 12 launches, 4.1 busy: `launch_weighted!` over `(1:nj, 1:nk)` of a block's shell; too few or too thin items. ~0.18 s.
10. **`_b_limit_ratio!`** (`AdvectionSchemes.jl:1069`, `prognostic_sgs/tke_blocked.jl:483`). 12 launches, 2.4 busy. Why it runs on ~2 threads (box shape, threshold) is not checked. ~0.15 s.
11. **`_b_∂x_cf!`, `_b_∂y_cf!`.** 6 launches each, 1–2 busy. ~0.06 s.
12. **`_drag_apply!`, `_drag_denominators!`.** Fine at 20 m (48–54 busy), 1 and 8 busy at 50 m: the threshold on smaller grids.

### C. Launches with part of the threads idle (busy 35–65)

13. **The fused sweep, `_b_sweep_slab!` and `_b_sweep_edge!`** (`fused_sweep.jl:771, 837`). 3.24 s at 37.6 busy (21 %) and 1.48 s at 59.5. `launch_weighted!` over `ns` slabs along k with a lag between neighbours (a wavefront), so fewer than 72 slabs are active at once, and the edges follow the slabs. Options: more slabs (split in j too), a second level of tiling, checking the slab/edge balance. ~0.5 s.
14. **`_coefficients_body!`, `_n2_body!`** (PrognosticSGS, stratification). 45 and 49 busy, 18 launches each. Decomposition (rows per item) or bandwidth; fusing them into the SGS sweep would also save a pass over memory each.
15. **`plane_reduce/_air_at`** (49.5), **`_b_correct!`** (58), **`_b_divergence_into!`** (61), **`_thomas_slab!`** (64.5), **`_copy_out_scaled_body!`** (53). ≤ 0.1 s each; possibly partial last chunks or bandwidth.
16. **Halo fills, `_xy_item!` / `_z_item!`** (`halo_sets.jl:140-144`). 1–3 % at 20 m (38–47 busy), but **32 % of the step at 50 m** (12–24 busy). Items are `n` parts of each field along one axis (k for the x/y halos, j for the z halos), weighted with `_MIN_POINTS_PER_TASK`, so smaller grids get few items; x-halo writes are strided (one cache line per row). Options: items over (field, j-block, k-block); fewer, merged fill launches (33 per step); longer term deeper halos or boundary conditions folded into the stencils. Matters for the 100 m and 50 m cases and for nested children.

### D. Allocation, GC and dispatch

17. **#574 (filed).** 1.22 GiB per step at 20 m: boxed arguments per kernel row at `Backends.jl:356` (a dynamic dispatch per row) and a fresh `open_boundary_snapshot` per step. A collection costs 1.7 s at 72 threads, and the dispatch costs time in every affected row.
18. **22 000 nested launches per step.** Nested launches run inline in their parent's task; worth checking that each does not pay launch setup (closure, range tuple, threshold check) that a plain loop would not.

### E. Placement and threads (not measured yet)

19. **NUMA placement unpinned.** `Backends` places pages by first touch from the thread that later uses the slab (`Backends.jl:173`). Run 1 found no difference against `--interleave=all`, but it was pinned and at 15–22 % of STREAM. Unpinned, the scheduler can move a thread to the other socket after its first touch. With 36 → 72 threads gaining 1.05×, worth checking: `numastat -p` during stepping, and Julia threads pinned with helper threads free (possible once #575 removes FFTW's own pool).
20. **GC threads.** 72 at startup, all pinned to core 0 under `JULIA_EXCLUSIVE=1` (#575). If pinning comes back, they need a full CPU mask or explicit `--gcthreads` placement.
21. **The main thread's double role.** It runs its share of every launch and all serial code between them: 78 % busy against 37 % for the workers, the critical path. After A, it may pay to give thread 1 a smaller share of the launches.

## How to measure a change

`bench_run.jl profile` on Fritz (one node, unpinned), 20 m and 50 m at 72 threads, as in `results/serial_t72/serial.sh`: the TOML's `trace_if_possible.by_name` gives each launch's wall and busy time, `serial_between_s` the serial time, and `*.profile_threads.txt` the main thread's frames. `bench_run.jl time` (or `scaling.sh` with `BENCH_PIN=0`) gives the step times.

## Records

- `results/serial_t72/`: TOMLs and `*.profile_threads.txt` of both cases, `serial.sh`. The raw `Profile` samples (`*.profile.jls`, 1.4 and 5.6 MB) are in `$WORK/aeolus-bench/results/serial_t72/` only. `Profile.print` fails on Fritz (ENODEV, HANDOFF §7), so `profile_main/all/tree.txt` are empty; `profile_fallback.txt` has inclusive counts over all threads.
- `results/scaling_20m_unpinned/`: the partial unpinned ladder, `ladder_unpinned.sh`, `procwatch.log`.
- `results/memory_20m/`: the allocation diagnosis behind #574.
- `results/scaling_20m_rerun2/`: the pinned ladder.

Related: #566, #574, #575.
