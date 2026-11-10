# HPC benchmark of the speedup changes: handoff (#566)

This document is for the Claude instance on the HPC cluster. Its job is to prepare and run a benchmark of every speedup change in the Speedup milestone, each one alone and all combined, then report back. The maintainer agreed the plan on 2026-11-10. Where a step needs a decision this document does not make, ask the maintainer and do not guess.

Read the repository's `CLAUDE.md` first; its rules apply here too. In short:
- one issue (#566), one branch (`issue/566-hpc-benchmark`), several commits, one PR;
- never merge, never push to `master`, never close issues;
- quote measured numbers, not adjectives;
- no coined jargon: use established terms (loop fusion, cache blocking, vectorisation) or plain descriptions. Follow `test/naming_conventions.jl` (the reanalysis name is reserved for ERA5).

Every commit message ends with:

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Wty6GpQRQqrEGvu68Ch3H6
```

## 1. Goal

For each change, measure what it gives on its own, what it still gives with everything else in, and what all of them give together. Measure on a 64-CPU node, in production-like runs.

Beyond time per step, measure where the time goes:
- thread idle and wait time;
- the serial share between kernels;
- memory bandwidth, both measured and from the byte model;
- IPC, vectorisation, cache misses and NUMA traffic.

Background reading, all on GitHub:
- **#531:** the step is memory-bandwidth bound.
- **#533:** loop fusion and its plan.
- **#535:** the laptop's power limit. This is why every comparison below is interleaved within one process.
- **#554:** the TKE update as one cache-blocked pass.
- **#556–#560:** follow-ups.

The PR bodies and the roadmap sections (`docs/src/design/roadmap.md` on each branch) hold the laptop measurements to compare against.

## 2. The changes and their groups

All open PRs, with their branches. "→" names the base branch.

| PR | Branch | Content |
| :-- | :-- | :-- |
| #541 | `issue/539-budget-cost` → master | Budgets: 50 → 13 passes per step |
| #543 | `issue/538-canopy-drag` → master | Canopy drag of the tiled surface, threaded |
| #544 | `issue/537-tile-update` → master | Tile surface update (2.4× at 1 thread) |
| #545 | `issue/533-sweep-interface` → master | Loop fusion step 1: the interface, the cache-blocked driver, the `loop_fusion` key. No behaviour change. |
| #546–#549 | steps 7, 3, 8, 6 | Loop fusion steps, each → #545 |
| #550 (draft) | `issue/533-integration` → #545 | Steps 3, 6, 7, 8 merged, **with five fixes for conflicts between the steps in its merge commits** (see the PR body) |
| #551, #552, #553 | steps 2, 5, 4 | Loop fusion steps, each → #545 |
| #555 | `issue/554-tke-blocked` → #545 | TKE update: faster kernels (part 1) and one cache-blocked pass (part 2) |
| #561 | `issue/556-tke-n2-strain` → #555 | N² and the strain rate inside the TKE pass |
| #564 | `issue/557-branchfree-flux` → #561 | Branch-free face values; the moist-N² vectorisation fix |
| #562 | `issue/559-tracer-diffusion-blocked` → #555 | Tracer SGS diffusion as one pass, plus the masked-diffusion vectorisation fix |
| #563 | `issue/558-blocked-j-split` → #555 | The cache-blocked driver split in j as well |
| #565 | `issue/560-rsc-blocked` → #555 | Reynolds-stress update as one pass, plus its per-cell allocation fix |

#542 is documentation only and is not benchmarked.

The groups. Infrastructure (#545) is in every build except master: with `loop_fusion = "never"` it changes nothing.

| Group | What | Comes from |
| :-- | :-- | :-- |
| **A** urban surface | tile update, canopy drag | #544, #543 |
| **B** budgets | fewer budget passes | #541 |
| **C** predictor loop fusion | steps 2–8 | #550 (+ #551, #552, #553) |
| **D** once-per-step cache-blocked passes | TKE pass, N²/strain inside it, tracer pass, Reynolds-stress pass | #555 part 2, #561, #562 (the pass), #565 (the pass) |
| **E** kernel fixes | the five fixes listed below | commits inside #555, #564, #562, #565 |
| **F** j split | `blocked_update!` split in j | #563 (needs D) |

The kernel fixes of group E, by commit:

| Fix | Commit(s) | Branch |
| :-- | :-- | :-- |
| E1: update of e, row-parallel, branch-free, vectorised | `16ad00c9` | `issue/554-tke-blocked` |
| E2: dual strain rate, bulk loop without checks | `cf704ba3` | `issue/554-tke-blocked` |
| E3: branch-free face values | `7bb2cd6a` | `issue/557-branchfree-flux` |
| E4: moist N², `@propagate_inbounds` | `0ff95557` | `issue/557-branchfree-flux` |
| E5: masked diffusion vectorises | `33b3ed00` (+ review `811a9b8e`) | `issue/559-tracer-diffusion-blocked` |
| E6: Reynolds-stress source update, no heap allocation per cell | `ea8c3fcb` | `issue/560-rsc-blocked` |

Check each SHA with `git show --stat <sha>` before use. If a branch has been rewritten, find the commit by its message.

## 3. The builds (14 worktrees, plus kernel-fix variants)

Each build is a worktree of its own branch, prefixed `bench/`. Push these branches; they are benchmark builds, not PRs.

1. **`bench/master`**: `origin/master`.
2. **`bench/all`**: everything, in this merge order (interface changes first):
   1. `#545`;
   2. `#550`, which already contains #548, #549, #546, #547 and its conflict fixes;
   3. `#551`, `#552`, `#553`;
   4. `#555`, then `#561`, then `#564`;
   5. `#562`, `#563`, `#565`;
   6. `#541`, `#543`, `#544`.

   Resolve conflicts the way #550's body describes: textual conflicts as unions, and semantic conflicts found by tests, fixed in the merge commit with a test. **Then run the full suite: `julia --project=. test/run.jl` must be green.** Also run `validation/run.jl --only gabls1.jl,urban_checks.jl`.
3. **Add-one builds `bench/only-<G>`**: master + #545 + group G:
   - `only-A`: + #544, #543.
   - `only-B`: + #541.
   - `only-C`: + #550, #551, #552, #553.
   - `only-D`: + #555, #561, #562, #565, with the group E commits reverted (`git revert` E1, E2, E5, E6).
   - `only-E`: + the E commits cherry-picked onto #545.
   - `only-F`: = `only-D` + #563. **F needs D.** Its effect is `only-F` against `only-D`, not against master.
4. **Leave-one-out builds `bench/all-minus-<G>`**: `bench/all` with group G taken out:
   - for A, B and C: build without those merges;
   - for D: revert its pass commits (keep E);
   - for E: revert E1–E6;
   - for F: revert #563.

   If a revert cannot be made to work without rewriting other code, record that and skip that build. Do not force it.
5. **Every build:** full test suite green before the slot (the suite takes about 8–9 min on a laptop). As a sanity check, after 20 steps of Berlin 100 m each build's u, v, w, θ and e must agree with `bench/master` within a stated tolerance. That tolerance is a reason, not a hash: chaotic divergence after 20 steps is small but not zero, so compare relative norms against `bench/master`'s run-to-run spread.

**The idle/wait hook must be in every build, master included.** It is one small commit (section 5), cherry-picked onto every `bench/*` branch. When its environment variable is unset it must cost nothing; measure that it doesn't.

## 4. Cases

1. **Berlin 100 m**: `benchmarks/hpc/berlin_100m/berlin_100m_20250824.toml`.
   - 200×200×60, 100 m cells, open lateral faces, TKE closure, the urban tile surface, budgets, `dt = 2 s`.
   - It's the maintainer's production run.
   - The drivers are not in git; make them with `prepare.sh` (step P4).
2. **Berlin 50 m**: the same domain at 400×400×60, so the run fills a 64-core node and is not mostly served from L3 cache.
   - Make a copy of the config with `size = [400, 400, 60]`, `dt = 1.0` (the CFL at 50 m), and its own driver names.
   - `landuse.jl` rasterises at 10 m and coarse-grains to 100 m (`R = 10`). For 50 m use `R = 5`, with output names `*_50m*`.
   - Remake the static and dynamic drivers.

Both cases run with their configured physics; do not switch physics off. `[checks]` and output can be off for timing. Read the config's `[output]` and `[checks]` sections and disable them through `--set` or a config copy. Budgets stay on, because group B is about them.

## 5. Metrics and how to take them

| Metric | How |
| :-- | :-- |
| Time per step, min and median | Own driver script. Warm up for 5 steps, then alternate `loop_fusion = "never"` and `"if_possible"` within one process, swapping their order every round, 4 rounds of 10 steps each. On builds without #545 (master) the key doesn't exist: there, time 40 steps. |
| Time per kernel, launch count, bytes per point (model) | `benchmarks/hpc/roofline_cfg.jl`: a copy of `bench_roofline.jl`'s profiler with config mode and a lock fix for nested launches. Separate runs: the profiler adds overhead. |
| **Thread idle and wait** | A new hook in `Backends.launch!` (and `launch_weighted!`, `plane_map!`, `row_map!`), active when `AEOLUS_LAUNCH_TRACE=1`. It records, per launch, its name, the wall time, and per task its thread, start and end. From that: busy time per thread, idle inside launches (wall × threads − busy), and serial time between launches (step wall − Σ launch walls). |
| OS-level wait | `/proc/self/task/*/schedstat`, read before and after: on-CPU time and time waiting on the run queue, per thread. Also `perf stat -e task-clock,context-switches,cpu-migrations` if `perf` exists. |
| GC and allocation | `@timed` around the measured steps |
| Memory bandwidth, measured | LIKWID `likwid-perfctr -g MEM` if installed; else `perf` memory-controller events (AMD `amd_df`, Intel `uncore_imc` CAS counts); else none, and say so |
| Memory bandwidth, peak | STREAM triad (the profiler has one) at each thread count |
| IPC, vectorisation, caches, NUMA | LIKWID groups `FLOPS_DP` (vector share), `L2`/`L3`, `NUMA` (remote accesses), `CYCLE_STALLS`/`TMA` if the CPU has them. One run per group. |
| Energy | `likwid-powermeter` or RAPL in `perf`, if permitted |
| The machine | `lscpu`, `numactl -H`, `likwid-topology`, Julia `versioninfo()`, CPU frequency governor and turbo state, SMT on/off |

Spinning: Julia's idle workers spin briefly before they sleep, so OS CPU time counts spinning as busy. Use the hook's numbers for idle time. Use the OS numbers only for run-queue wait and migrations.

The script detects which tools exist. It records which metrics it could take, and why it skipped others.

## 6. Node layout: 17 nodes in parallel, about 1 h each

Every node starts with a **reference block** of about 4 min:
- STREAM triad at the node's thread counts;
- 20 steps of `bench/master` on Berlin 100 m at 64 threads.

Results are reported raw and normalised to the node's own reference, so the nodes do not have to be identical.

| Node | Build | Matrix |
| :-- | :-- | :-- |
| 1 | master | both cases, threads 1, 2, 4, 8 |
| 2 | master | both cases, threads 16, 32, 64 (128 if SMT); counter groups at 64 |
| 3 | all | both cases, threads 1, 2, 4, 8 |
| 4 | all | both cases, threads 16, 32, 64 (128); counter groups at 64; NUMA test: 64 threads with and without `numactl --interleave=all` |
| 5–10 | `only-A` … `only-F` | both cases, threads 16 and 64; counter groups at 64 |
| 11–16 | `all-minus-A` … `all-minus-F` | the same |
| 17 | all + kernel-fix variants | the per-fix kernel benchmarks (section 7) |

Pinning: `JULIA_EXCLUSIVE=1`, or `likwid-pin`, so that thread i sits on core i. If the node has SMT, 128 threads must use the hardware threads; check it in the topology.

**The estimates are from the laptop, so check them in the dry run.** Each process spends about 2 min compiling and setting up Berlin before measuring. The busiest nodes (2 and 4) come to about 55 min.

If the dry run shows a node over 50 min, either:
- split its thread counts over one more node (up to 16 + node 17 are available, and node 17 can take more), or
- build system images with PackageCompiler in `benchmarks/Project.toml`. Ask the maintainer first: it is a new dependency of the benchmark environment.

Submit all 17 at once, as an array or as separate jobs, with exclusive nodes and a walltime of 1 h 15 min.

## 7. Node 17: the kernel fixes one by one

Each fix is measured as its kernel alone, old against new, at threads 1, 16 and 64 on the Berlin 100 m state after 3 steps. The kernel's share of the step comes from the profile of `bench/all` (node 4). The fix's effect on the step is then about that share times its kernel speed-up.

Old against new:
- **E1, E2:** `kernel_fixes/tke_update_strain_old.jl` holds copies of the old kernels; `tke_update_strain_bench.jl` interleaves them in one process.
- **E3:** `test/support/branching_faces.jl` on `issue/557-branchfree-flux` gives `Branching(r)`, the old branching face values as a test descriptor. `face_values_bench.jl` uses it.
- **E4, E5, E6:** no old copy exists. Build a variant worktree of `bench/all` with that one commit reverted. Precompile only; no full suite is needed for a kernel benchmark, but check that the kernel's results agree with `bench/all` within round-off. Then run the kernel in alternating processes: all, variant, all, variant.
  - E4: `moist_n2_bench.jl`
  - E5: `masked_diffusion_bench.jl`
  - E6: `rsc_update_bench.jl`

The scripts in `kernel_fixes/` and `roofline_cfg.jl` are laptop starting material: their paths point at the laptop. Adapt them, then commit them as tools.

## 8. Preparation checklist (on the cluster, before the slot)

- **P1. The cluster.** Find out the scheduler, the Julia setup (the project uses Julia 1.13.3; use exactly that version, e.g. via juliaup), the node type, SMT, and which of LIKWID and perf exist and whether they are allowed. Write it down in `benchmarks/hpc/CLUSTER.md`.
- **P2. Branches.** Clone, fetch all branches above, build `bench/*` (section 3), and run the full suite on each. Use compute nodes for the suites if the login node is too small.
- **P3. The hook.** Write the idle/wait hook (section 5) with a test that it records what it should and costs nothing when off. Commit it on `issue/566-hpc-benchmark` and cherry-pick it onto every `bench/*`.
- **P4. Drivers.**
  - Berlin 100 m: `benchmarks/hpc/berlin_100m/prepare.sh`. It needs ERA5 from the Copernicus CDS (`~/.cdsapirc`), OpenStreetMap via the Overpass API (a 265 MB extract), and AeolusPrep (`github.com/badulion/AeolusPrep.jl`, branch `issue/6-era5-soil-profiles`).
  - Berlin 50 m: as in section 4.
  - If the cluster cannot reach CDS or Overpass, ask the maintainer to copy the laptop's `data/` folder (270 MB, ERA5 and the OSM extract already downloaded) or the finished drivers.
- **P5. The scripts:**
  - the run driver (section 5), writing one TOML per run with build, commit, case, threads, setting, every metric, the node's reference, and the tool availability;
  - the job script per node, from section 6;
  - the submit script;
  - the summary script, which writes tables and figures.
- **P6. Dry run.** On one node, 15 min: every metric path once, 2 builds, threads 2 and 64, Berlin 100 m, 5 steps. Check that the TOMLs parse and the summary runs. Use it to correct the time estimates in section 6.
- **P7. Submit** the 17 jobs.

## 9. What to report back

1. **The summary:**
   - time per step, both cases, every build and thread count;
   - the speed-up of each group alone (add-one against master; F against only-D) and within everything (all-minus against all);
   - the interaction: the product of the single speed-ups against the speed-up of all;
   - scaling curves of master and all;
   - stacked bars per build at 64 threads: kernels, idle inside kernels, serial between kernels, GC;
   - measured bandwidth against STREAM and against the byte model;
   - the per-fix kernel speed-ups and their estimated step shares;
   - the NUMA and SMT results.
2. **The figures:** open and look at every figure before reporting (CLAUDE.md: "Look at what you produce").
3. **Where:**
   - results under `benchmarks/hpc/results/<date>/`: raw TOMLs plus the summary;
   - a section `#566` in `docs/src/design/roadmap.md`;
   - a PR from `issue/566-hpc-benchmark` to master with the tools, the results and the roadmap section, saying "Closes #566" only if everything planned was measured.
4. **Honesty:**
   - say which metrics could not be taken and why;
   - say which builds failed or were skipped;
   - say where nodes differed (the reference block);
   - say where the laptop numbers and the HPC numbers disagree.

## 10. Things that went wrong on the laptop (avoid them)

- **Background shells die when an agent's turn ends.** Never leave a run pending when you return; wait for it.
- **Outside load spoils timings.** On a shared login node, never time anything. Time only on exclusive compute nodes.
- **Stale scratch fields.** #555 found `validation/anelastic_cbl.jl` reading a field the blocked pass no longer writes. If a sanity check disagrees, look there first.
- **Most setup is serial.** Setting up Berlin is single-threaded (compiling, reading drivers, initialising the surface), so CPU load looks low during it. That's expected. Measure only the stepping.
- **First-touch NUMA.** Aeolus allocates its arrays from the main thread. On a multi-socket node all memory may then sit on one NUMA domain. The `NUMA` counter group and the `--interleave=all` comparison on node 4 test this; report it, even if it is the largest effect found.
