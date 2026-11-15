# Berlin 20 m on Fritz after #576–#582 (#566)

Measured 2026-11-15 on `bench/master-581` (b188b88a): master 92702802 (#576–#582 merged) plus the launch trace of `bench/hook-dyn` (c14e92f0, efb5ec4c, 5444aa27). The trace commit conflicted with #574's `(body!, args)` tuple and #581's `_planes3_kernel!`. I kept master's kernels; the traced variants take the same tuple, and `_planes3_kernel!` has a traced twin. A new case in `test/launch_trace.jl` checks a launch only 3 points wide along i; the trace test passed on the node (44 of 44).

**Compared against** the `bench/all` build (51154519 = master 4f61b8a6 + the bench PRs, **including group D**, the once-per-step cache-blocked passes, which master does not have):
- 4471876, the unpinned ladder;
- 4471418, the pinned ladder;
- 4471898, the profile ("serial finder");
- 4471714, the memory run.

So "before → after" mixes the #576–#582 gains with the loss of group D.

**Jobs** (one node each, `-t N,0`, FFTW estimate; results in `results/after_581/`):

| job | what | where |
| :-- | :-- | :-- |
| 4472316 | trace test; `bench_run.jl profile` at 50 m, then 20 m, 72 threads, unpinned | `profile/` |
| 4472317 | `scaling.sh` 72 36 16 8, unpinned (`BENCH_PIN=0`) | `scaling_unpinned/` |
| 4472318 | `scaling.sh` 72 36, pinned (`JULIA_EXCLUSIVE=1`), for #575 | `scaling_pinned/` |
| 4472319 | `bench_run.jl memory` at 20 m, 72 threads, unpinned | `memory/` |

Setup took 130–156 s per process; peak RSS was 71–78 GB. The stop rule did not trigger.

## Step time and scaling (20 m, median s/step, untraced)

| threads | unpinned before | unpinned after | pinned before | pinned after |
| --: | --: | --: | --: | --: |
| 72 | 6.10 | **6.08** | 28.37 | **6.12** |
| 36 | 6.40 | 7.35 | 28.69 | 7.72 |
| 16 | 9.06 | 10.98 | 30.77 | – |
| 8 | 16.47 | 17.52 | 33.89 | – |

- **36 → 72 threads:** 1.05× before, **1.21×** after unpinned (1.26× pinned).
- **At 72 threads the step time is unchanged** (6.08 against 6.10 s). At 8–36 threads master is 6–21 % slower than `bench/all`, which fits the missing group D: its passes do the same work in fewer passes over memory, and their gain is largest where bandwidth per thread is highest. At 72 threads the #576–#582 gains roughly cancel that loss. A `bench/all`-style build on top of master is needed to separate the two.

## Busy threads (`/proc/self/task/*/schedstat`, untraced 72-thread run)

| | main thread | each other Julia thread | cores in use |
| :-- | --: | --: | --: |
| before, unpinned | 78 % | 36 % | 33.9 |
| after, unpinned | 83 % | 55 % | 39.6 |
| after, pinned | 83 % | 60 % | 43.5 |

The workers are busier but the step at 72 threads is not faster. Part of the busy time is probably spinning before sleep, and part is memory bandwidth; the trace's per-launch busy counts work items only (below).

## Launch trace (2 traced steps, 72 threads, unpinned)

| | 20 m before | 20 m after | 50 m before | 50 m after |
| :-- | --: | --: | --: | --: |
| traced steps, s | 6.81, 8.59 | 5.90, 5.98 | 1.36, 1.57 | 0.65, 1.57¹ |
| profiled steps, s | 6.52–6.97 | 5.89–6.23 | 1.13–2.26 | 0.63–0.65 |
| outside launches | 4.77 s, **31 %** | 2.93 s, **25 %** | 0.72 s, 25 % | 0.23 s, 10 % |
| threads idle inside launches | 52 % | 27 % | 72 % | 61 % |
| launches / nested | 482 / 44 064 | 626 / 44 064 | 482 / 40 352 | 626 / 40 352 |

¹ The second 50 m traced step has 0.53 s more in `_xy_item!` than the first; the three profiled steps after it are 0.63–0.65 s, so this is probably a one-off rather than a cost per step.

**The launches named in #581** (per 2 steps: count, wall s, busy threads of 72):

| launch | 20 m before | 20 m after | 50 m before | 50 m after |
| :-- | :-- | :-- | :-- | :-- |
| `_blend_body!` (B1) | 120, 2.409, 0.5 | 120, **0.034**, 6.8 | 120, 0.247, 0.8 | 120, 0.018, 1.8 |
| `_map_body!` (B2) | 80, 0.203, 3.7 | 80, **0.021**, 5.6 | 80, 0.155, 0.8 | 80, 0.014, 1.3 |
| `_b_shell!` (B3) | 12, 0.193, 4.1 | 12, **0.014**, 23.6 | 12, 0.036, 3.0 | 12, 0.007, 6.7 |
| fused sweep (C1, #558/#579) | slab 3.242 at 37.6 + edge 1.484 at 59.5 | block **1.514** at 49.9 + edge **0.731** at 63.5 | slab 0.415 at 18.5 + edge 0.133 at 47.0 | block 0.164 at 28.4 + edge 0.112 at 55.8 |
| halo fills `_xy_item!` | 66, 0.248, 46.8 | 66, 0.250, 48.2 | 66, 0.128, 11.9 | 66, 1.071¹, 22.7 |
| halo fills `_z_item!` | 66, 0.152, 37.6 | 66, 0.137, 38.4 | 66, 0.825, 24.3 | 66, 0.100, 10.0 |
| `_b_limit_ratio!` | 12, 0.155, 2.4 | 2, 0.145, 67.6 | 12, 0.041, 1.0 | 2, 0.010, 60.7 |
| `_b_∂x_cf!` (B5) | 6, 0.059, 2.2 | 6, 0.041, 4.5 | 6, 0.004, 1.0 | 6, 0.005, 1.0 |

`_block_transform_body!` (12 per 2 steps, 2.30 s at 49.3 busy at 20 m) is new: the pressure solve's FFTW transforms, now running as launches on Julia's threads (#575). Before, they ran on FFTW's own pool and counted as time outside launches.

**The main thread outside launches at 20 m** (Profile, thread 1 awake: 264 samples; 305 including asleep, over 3 steps of 5.9–6.2 s; so one sample is about 0.06 s per step):
- `copy_storage!` (`Simulations.jl:1152`, the `copyto!` of u, v, w, θ, qv at the start of the step): 53 samples, **~1.0 s per step** (before: 55 samples of 387, 15.5 %);
- `fill!` in `advance_sgs!`: `PrognosticSGS.jl:867` (`c.Pw` in `_wall_production!`) and `:964` (`c.tend`), whole-field memsets: 28 samples, **~0.55 s per step**. This was not in `REPORT_threads_20m.md`, where it was probably hidden in group D's TKE pass;
- the pressure solve outside its transform launches (`solve_source_buffer!` lines 388, 715, 729): 20 samples, ~0.4 s per step (before: the FFTW calls, 19.5 %);
- `apply_radiative_cooling!` and `correct_boundary_mass_flux!` (A3, A4): no longer among the main thread's frames.

## Allocation and GC (20 m, 72 threads)

| | before | after |
| :-- | --: | --: |
| allocated per step | 1.22 GiB | **0.0086 GiB** (8.8 MiB) |
| GC time in 4 steps | 1.66 s (one collection) | 0 |
| live heap over 4 steps | +1.22 GiB per step | 72.52 → 72.55 GiB |
| s/step | 5.9–6.2 (7.7 with a GC) | 6.12–6.30 |

The remaining 8.8 MiB per step are KernelAbstractions' task objects in `_threaded!` (`Backends.jl:604`, 591, 568) and `_threaded_planes!` (`:993`), as #577 described.

## For the issues

- **#574:** allocation 1.22 GiB → 0.0086 GiB per step, no collection in 4 steps (or in the 5-step ladder runs); the live heap is flat. Confirmed on Fritz.
- **#575:** pinned at 72 threads, 6.12 s against 6.08 unpinned (28.37 before). The run-queue wait of all 144 tasks is 0.9 s over 69 s of stepping (15 124 s before), and FFTW runs on 1 thread of its own. Confirmed on Fritz.
- **#558 (fused sweep split in j, #579):** at 20 m the sweep went from 4.73 s (slab + edge) to 2.25 s (block + edge) per 2 steps, and busy from 37.6 to 49.9 threads. Confirmed; the block launch still leaves about 22 threads idle.
- **#581:** B1, B2, B3, A3 and A4 are confirmed. 25 % of the step is still outside launches at 20 m, the step at 72 threads is not faster than `bench/all`, and the 36 → 72 gain is 1.21×. What remains is below.

**#581's remaining items at 20 m, 72 threads, by time per step** ("lost" = launch wall × (1 − busy/72) per step; serial items are their full time):

| # | item | cost per step |
| --: | :-- | --: |
| 1 | A2: `copy_storage!`, a serial memcpy of 5 fields | ~1.0 s serial |
| 2 | new: whole-field `fill!` in `advance_sgs!` (`PrognosticSGS.jl:867, 964`) | ~0.55 s serial |
| 3 | serial part of the pressure solve around the transforms (`solve_source_buffer!`) | ~0.4 s serial |
| 4 | the transform launches `_block_transform_body!` (49 of 72 busy) | 0.36 s lost |
| 5 | C1 sweep blocks `_b_sweep_block!` (50 busy) + edge (64 busy) | 0.23 + 0.04 s lost |
| 6 | new: `_dual_strain_row!` (23 busy) | 0.19 s lost |
| 7 | halo fills `_xy_item!` + `_z_item!` (20 m) | 0.07 s lost (50 m: 0.41, with the one-off¹) |
| 8 | `_coefficients_body!`, `plane_reduce/_air_at`, `_b_correct!`, `_n2_body!` | 0.02–0.035 s lost each |
| 9 | `_b_∂x_cf!` (B5) still at 4.5 busy | 0.02 s lost |
| 10 | D2: 44 064 nested launches per 2 steps (unchanged) | not separable in the trace |

Launches together lose 1.2 thread-equivalent seconds per step at 20 m and 0.6 at 50 m. Items 1–3 are serial and add up to about 2 s of a 6 s step.
