# The cluster: Fritz (NHR@FAU), as found on 2026-11-10 (#566, P1)

## Nodes and scheduler

- Scheduler: Slurm. Partition `singlenode` (816 nodes, `--exclusive`, one node per job).
  The accounting daemon (`slurmdbd`, fadm2:6819) refused connections on
  2026-11-10, so `sacct`/`sacctmgr` fail; `squeue`/`sbatch` work. Job metrics are in
  ClusterCockpit (monitoring.nhr.fau.de; the API needs a JWT).
- Node: 2 × Intel Xeon Platinum 8360Y (Ice Lake SP), 36 cores each, **72 cores**,
  **no SMT** (1 thread per core), 256 GB in **4 NUMA domains** of 18 cores
  (sub-NUMA clustering: cores 0-17, 18-35, 36-53, 54-71; 64 GB each).
  AVX-512, 8 programmable counters per core.
- Clock: governor `powersave`, turbo off (`intel_pstate/no_turbo = 1`), at most
  2.4 GHz. LIKWID's frequency tools are not available; Slurm features set the
  frequency. The runs keep the default, so no turbo noise between nodes.
- So the handoff's 64 threads are **72** here (the full node) and its 32 are **36**
  (one socket; with `JULIA_EXCLUSIVE=1`, thread i on core i, 36 threads fill
  socket 0). The thread ladder is 1, 2, 4, 8, 16, 36, 72. There is no 128-thread
  (SMT) case.
- `/scratch` on a compute node is node-local: job files and results go under
  `$WORK` (`/home/atuin/b337bd/b337bd10`).
- The login node (fritz4) is shared: nothing is computed or timed there.

## Julia

- Julia **1.13.3**, the official tarball, at `$WORK/aeolus-bench/julia-1.13.3`
  (no juliaup). Depot `$WORK/aeolus-bench/depot`. `$WORK/aeolus-bench/env.sh` sets
  `JULIA`, `PATH`, `JULIA_DEPOT_PATH` and `AEOLUS_FFTW_PLANNER=estimate`.
- Each build is a worktree `.worktrees/bench-<build>` with its own Manifest. A job
  runs `Pkg.instantiate()` first, so precompilation happens on the compute node.

## Tools

| Tool | Status |
| :-- | :-- |
| LIKWID 5.5.1 (`module load likwid`) | works **only with `--constraint=hwperf`** (every singlenode node has the feature). Groups on ICX: MEM, MEM_DP, FLOPS_DP, FLOPS_AVX, L2, L3, L2CACHE, L3CACHE, CYCLE_STALLS, CYCLE_ACTIVITY, TMA, UPI, ENERGY, CLOCK, … **No NUMA group**: placement shows in MEM (per socket) and UPI (cross-socket traffic). Timeline mode with several groups multiplexed works (`likwid_wrap.sh`). |
| likwid-powermeter / RAPL | works (package and DRAM energy) |
| perf | `/usr/bin/perf`; `perf_event_paranoid = 2`, so user-space events of one's own processes only (`task-clock`, `context-switches`, `cpu-migrations`). The `uncore_imc` CAS events are listed but need LIKWID's access. |
| numactl | present (`--interleave=all` for the NUMA comparison) |
| `/proc/self/task/*/schedstat` | readable: on-CPU and run-queue wait per thread |
| `gh` | not installed |

## Data

- ERA5 and OSM are not fetched on the cluster (no `~/.cdsapirc`). The maintainer
  copied the laptop's `data/` to `$WORK/aeolus-bench/berlin_100m_data/`; the 100 m
  and 50 m drivers were made from it (`prepare.sh`, P4).

## Measured on the way (one node each, not yet the benchmark)

- Berlin smoke runs (job 4456879, `bench/master` era code, 72 threads): setup
  187 s (100 m) and 296 s (50 m), driver preparation 118 s, so the handoff's
  "about 2 min" of setup per process is 1.5-3× low here. Stepping 0.66 s (100 m)
  and 2.03 s (50 m) per step, unverified (few steps, possibly warm-up).
