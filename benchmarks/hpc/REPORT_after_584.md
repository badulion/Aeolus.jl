# #584 (#581 A2, the step-start copy) on Fritz (#566)

This measures two ways to remove the serial copy at the start of each step, against master. Job **4474794** ran on 2026-11-16 on node f0316: Berlin 20 m (1000 × 1000 × 150, moist), unpinned, FFTW estimate. The three builds ran one after another on the same node:

| build | commit | what |
| :-- | :-- | :-- |
| `bench/master-581` | f6e8b10c | master 22f644f7 (#583) + the launch trace |
| `bench/584-swap` | 53059213 | + `issue/584-rk3-swap`: stage 1 swaps the state arrays with the workspace, and the fused sweep writes out of place |
| `bench/584-copy` | 0fbd1140 | + `issue/584-chunked-copy`: `copy_storage!` copies in chunks of ≥ 1 M points, one `copyto!` per task |

**Runs per build:**
- `bench_run.jl profile` at `-t 72,0`: 2 warm-up steps, 2 traced steps, then **10 profiled steps**;
- `bench_run.jl time` at `-t 36,0`: 2 warm-up steps, then **10 steps** with `loop_fusion = if_possible` and 10 with `never`.

Model setup took 132–133 s per process, there was no GC, and the stop rule did not trigger.

**The system images were not built.** The job script did not set `SYSIMG_CFG`, which the image workload reads, so all three builds stopped with a `KeyError` (`results/after_584/sysimage_logs/`). The benchmark processes don't use the images, so the measurements are unaffected. `a2.sh` in the results now sets the variable; rebuilding the images needs another job.

## Results (median s/step)

| build | 72 threads (10 profiled steps) | vs master | 36 threads (10 steps) | vs master |
| :-- | --: | --: | --: | --: |
| master | **5.65** (5.17–5.75) | | **6.65** (6.57–6.71) | |
| swap | **4.70** (4.58–4.83) | −0.94 s, −16.7 % | **6.02** (5.92–6.10) | −0.63 s, −9.5 % |
| copy | **4.85** (4.79–4.96) | −0.80 s, −14.2 % | **5.95** (5.88–6.05) | −0.70 s, −10.6 % |

- **At 72 threads swap wins**: 0.14 s (3 %) faster than copy.
- **At 36 threads copy wins**: 0.07 s (1.2 %) faster than swap.

Each is one run per build. The 72-thread ranges do not overlap; the 36-thread ranges overlap.

- **Master at 72 threads is 5.65 s**, against 6.08 s before #583; the A7 and C5 changes of #583 sit in between.
- **36 → 72 threads:** master 1.18×, swap 1.28×, copy 1.23×.

## Main thread: the copy (Profile, thread 1, 10 steps)

Thread 1 has 941 samples (master) over 56.5 s of stepping, so one sample is about 0.06 s, or 0.006 s per step averaged over the 10 steps.

| build | `copy_storage!` on thread 1 | per step |
| :-- | :-- | --: |
| master | 183 of 852 awake samples (21.5 %), all in `memmove` | **~1.1 s** |
| swap | no `copy_storage!` frame at all | **0** |
| copy | 12 awake + 1 asleep in `_chunked_copy!`; the other 71 threads together 899 samples in `_chunked_copy!`, about 12.7 each | **~0.07 s** |

**Launch trace** (2 traced steps):

| build | outside launches per step | share of step | idle inside launches |
| :-- | --: | --: | --: |
| master | 0.99 s | 17 % | 27 % |
| swap | 0.19 s | 4 % | 29 % |
| copy | 0.20 s | 4 % | 25 % |

## The fused sweep (launch trace, per step; busy threads of 72)

| build | `_b_sweep_block!` | `_b_sweep_edge!` | together |
| :-- | :-- | :-- | --: |
| master | 0.805 s, 46.9 | 0.640 s, 51.9 | 1.445 s |
| swap | 0.803 s, 47.3 | 0.531 s, 55.0 | 1.334 s |
| copy | 0.801 s, 47.7 | 0.495 s, 63.3 | 1.296 s |

**The out-of-place sweep shows no cost at this resolution.** The desktop measured +25 % of a stage-1 sweep for it. Here the block launch is 0.803 s against 0.805 s, and the edge launch varies by 0.15 s between builds on the same code path. With 2 traced steps, the difference is below the noise.

## Allocation

All three builds: **9.0–9.1 MiB per step** at 72 threads (profiled steps), 9.8 MiB at 36 threads, and no GC. Neither candidate allocates more.

## Also seen: swap's unfused path is slower

At 36 threads with `loop_fusion = never`:

| build | s/step |
| :-- | --: |
| master | 7.67 |
| copy | 6.95 |
| **swap** | **9.65** (+26 % against master) |

The default (`if_possible`) is unaffected. Runs with fusion off (or any domain the fused sweep does not cover) would pay this.

### Diagnosis (job 4477000)

The desktop could not reproduce it (the same 5 copies per step as master; Berlin 100 m 0.252 against 0.255 s). Job **4477000** (node f0828, 2026-11-16) ran `bench_run.jl trace` at `-t 36,0`, Berlin 20 m, unpinned: master-581, then 584-swap, each with 2 warm-up steps and then, for each setting, 2 discarded and 4 traced steps (`results/after_584_diag/`).

| per step | master `never` | swap `never` | master `if_possible` | swap `if_possible` |
| :-- | --: | --: | --: | --: |
| step | 7.72 s | **8.26 s** (+0.54, +7 %) | 6.85 s | 6.09 s |
| inside launches | 6.66 s | 7.25 s | 5.84 s | 5.99 s |
| between launches (serial) | 1.06 s | 1.01 s | 1.01 s | 0.10 s |
| idle thread-seconds in launches | 39.6 | 48.8 | 38.0 | 38.8 |

- **The regression is smaller this time:** +7 % against +26 % in job 4474794, on a different node with the same builds. So its size varies from run to run.
- **It is not extra serial work and not an extra pass.** The serial time between launches is the same (1.01 against 1.06 s, the 5 copies), and every launch runs the same number of times.
- **It is spread over all the memory-bound launches:** each is 5–35 % longer, with slightly fewer busy threads. The largest are `_b_eddy_scalar!` (+0.09 s), `_coefficients_body!` (+0.08), `_map_body!` (+0.07), `_b_tendency_scalar!` (+0.05) and `_n2_body!` (+0.04).
- **Reading:** the same work, with slower memory access throughout. That points to where the arrays' pages lie (NUMA, or page layout) rather than to the code path. A slowdown that varies by run fits this too. It is not explained yet:
  - every array is first-touched in parallel (`Backends.allocate`);
  - the 2 fused warm-up steps swap the arrays twice, so the state is back in its original arrays when the `never` steps start.

  Whether Fritz migrates pages (`kernel.numa_balancing`), and how the fused sweep's writes to the workspace interact with that, is unchecked.
- **The fused path is unaffected:** swap is again −0.76 s per step against master at 36 threads.

### Placement check (job 4481036)

Job **4481036** ran on node f0313 on 2026-11-18. It used `bench_run.jl trace` at `-t 36,0`, Berlin 20 m, with `nwarm=2 nsteps=4`. The builds were `bench/master-581` (f6e8b10c) and `bench/584-swap` (27b12c34, which includes the open-boundary fix e918e33e). Each build ran in three variants, one process each; `numastat -p` was sampled every 20 s. The raw data is in `results/after_584_placement/`.

**The node:**
- `kernel.numa_balancing = 1`.
- 4 NUMA nodes of about 64 GB each (sub-NUMA clustering, 2 per socket): cores 0–17, 18–35, 36–53 and 54–71.
- Distances: 10 within a node, 11 to the other node of the same socket, 20 to the other socket.
- **Pinned** (`JULIA_EXCLUSIVE=1`): the 36 threads sit on cores 0–35, which is one socket (NUMA nodes 0 and 1).

**Per step (4 traced steps per setting):** step / inside launches / between launches, in s, then idle thread-seconds inside launches.

| variant | build | `never` | `if_possible` | memory during `never`, GB on NUMA nodes 0/1/2/3 |
| :-- | :-- | :-- | :-- | :-- |
| unpinned | master | **7.90** / 6.93 / 0.97, 44.9 | 6.82 / 5.81 / 1.01, 36.7 | 18.2 / 24.7 / 16.2 / 15.9 |
| unpinned | swap | **7.60** / 6.53 / 1.06, 37.9 | 6.03 / 5.92 / 0.11, 40.6 | 17.8 / 14.3 / 23.5 / 20.7 |
| pinned | master | **9.05** / 8.13 / 0.92, 31.9 | 8.21 / 7.28 / 0.93, 31.9 | 49.0 / 29.2 / 0.0 / 0.0 |
| pinned | swap | **7.99** / 7.06 / 0.92, 31.0 | 5.80 / 5.71 / 0.09, 28.7 | 44.0 / 35.0 / 0.0 / 0.0 |
| interleaved | master | **8.30** / 7.35 / 0.94, 31.3 | 6.94 / 5.99 / 0.96, 31.8 | 19.1 / 19.1 / 19.1 / 18.9 |
| interleaved | swap | **8.43** / 7.47 / 0.96, 33.7 | 6.10 / 5.99 / 0.11, 32.3 | 19.3 / 18.9 / 19.0 / 19.0 |

The memory column is the last `numastat` snapshot inside the `never` block; the totals are 75–79 GB.

**Swap against master with fusion off:**

| variant | difference |
| :-- | :-- |
| unpinned | −0.30 s (−4 %) |
| pinned | −1.06 s (−12 %) |
| interleaved | +0.13 s (+1.6 %) |

**Reading.**
- **This node shows no gap unpinned.** Swap's unfused path is 4 % faster than master's here, against +26 % (job 4474794) and +7 % (job 4477000) before. Neither pinning nor interleaving has a gap to close, so the check can't tell thread placement from page placement.
- **The differences between two processes running the same work follow where the pages land, not the build.**
  - Unpinned, each process spreads its 76 GB unevenly over the 4 NUMA nodes, and differently each time: master 18/25/16/16 GB, swap 18/14/24/21 GB.
  - Pinned, one NUMA node holds 44–49 GB of its 64 GB.
  - Pinned, master is the slower one (by 1.06 s with fusion off). Its launches are slower although it runs the same unfused passes as swap (8.13 against 7.06 s inside launches); it also has the more lopsided split (49/29 GB against 44/35 GB).
  - Interleaved, the pages are even (19 GB per node), and the two builds are within 1.6 %.
- **Conclusion:** the earlier "regression" is run-to-run variation from page placement. It can go either way, and it is not a cost of the swap. Read single unpinned unfused comparisons at 20 m with ±10–25 % in mind.
- **The fused default:** swap is faster than master in every variant: −0.79 s unpinned, −0.84 s interleaved, and −2.41 s pinned (where master's run was slow throughout).

## Against the desktop reference

The desktop reference is 1006 × 1006 × 156 moist, sweep plus copy per step:

| threads | master | copy | swap |
| --: | --: | --: | --: |
| 24 | 731 ms | 654 ms (−77) | 572 ms (−159) |
| 1 | 1856 ms | 1855 ms | 1746 ms |

On Fritz at 72 threads, master's copy is ~1.1 s per step on one core. Swap removes it (−0.94 s per step); chunked copy leaves ~0.07 s (−0.80 s). The desktop's 2:1 advantage of swap over copy shrinks to 0.14 s at 72 threads and reverses by 0.07 s at 36 threads.
