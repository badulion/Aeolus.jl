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

The default (`if_possible`) is unaffected. Runs with fusion off (or any domain the fused sweep does not cover) would pay this. The cause is not checked; perhaps the unfused stage 1 with swapped arrays adds a copy or a pass.

## Against the desktop reference

The desktop reference is 1006 × 1006 × 156 moist, sweep plus copy per step:

| threads | master | copy | swap |
| --: | --: | --: | --: |
| 24 | 731 ms | 654 ms (−77) | 572 ms (−159) |
| 1 | 1856 ms | 1855 ms | 1746 ms |

On Fritz at 72 threads, master's copy is ~1.1 s per step on one core. Swap removes it (−0.94 s per step); chunked copy leaves ~0.07 s (−0.80 s). The desktop's 2:1 advantage of swap over copy shrinks to 0.14 s at 72 threads and reverses by 0.07 s at 36 threads.
