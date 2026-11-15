# Built but not merged

A catalogue of work that was implemented, tested and in most cases measured,
but deliberately left out of `master`. Each entry says where the code is,
what it does, what was measured, why it was left out, and what reviving it
would take. The branches stay on `origin`. **Do not delete them**: they are
the only copy.

Add an entry whenever a finished PR is closed without merging or only part
of it is taken. Remove an entry when its code reaches `master`, with a note
saying how.

## How to read the numbers

Unless an entry says otherwise, "run 1" is the HPC benchmark of #566 on
Fritz (NHR@FAU), 2026-11-10: Berlin 100 m (200 × 200 × 60) and 50 m (400 ×
400 × 60), median s/step with `loop_fusion = "if_possible"`, one process per
configuration. The results are in `benchmarks/hpc/results/run1/` on
`issue/566-hpc-benchmark` (`report.md`, `summary.md`).

- **"Alone"** is the add-one build against master.
- **"Within all"** is all against the build without that group.
- Differences under about 3 % are within the run-to-run spread.
- Run 1 pinned the threads (`JULIA_EXCLUSIVE=1`). That slows high thread
  counts (#575) but affects every build in the same way. At 1 thread pinning
  plays no part.

## The speedup integration of 2026-11-14: what was left out

The integration PR (branch `issue/566-speedup-integration`) brought these
groups to master:

- **C:** loop fusion, #533 (#545, #550, #551–#553).
- **B:** the budget passes, #541.
- **A:** the urban surface, #543 and #544.
- **E:** the kernel fixes, cherry-picked from the D branches.
- **#542:** documentation.

This is the build `bench/all-minus-D` that run 1 measured, plus today's
master. Measured against master: 1.42× (100 m, 16 threads), 1.42× (100 m,
72), 1.39× (50 m, 16), 1.53× (50 m, 72).

Group D and group F were left out (maintainer decision, 2026-11-14).

### D1. The TKE update as one cache-blocked pass (#554, PR #555 part 2)

- **Where:** `issue/554-tke-blocked`, head `0dc066a0`, PR #555 (base
  `issue/533-sweep-interface`).
- **What:**
  - **The generic blocked driver.** The #533 slab-and-ring driver becomes
    `FiniteVolume.blocked_update!` with a `BlockedUpdate` interface, and the
    predictor is one such update.
  - **The TKE pass.** The prognostic TKE update `advance_sgs!` runs as one
    pass over cache-sized slabs (`src/prognostic_sgs/tke_blocked.jl`): ν,
    face fluxes, the limiter ratio, divergence, diffusion and the update,
    row by row, two rows ahead. Every expression comes from the separate
    loops' point bodies through adapters.
  - **Selection.** `loop_fusion` chooses it, and `results.toml` reports
    `tke_update_blocked`.
- **Measured, run 1:** group D (D1–D4 together):
  - alone 1.07× / 1.00× (100 m, 16 / 72 threads), 1.11× / 1.09× (50 m);
  - within all 1.01× / 0.98× (100 m), 1.10× / 1.02× (50 m);
  - all against all-minus-D, s/step: 100 m 16 threads 0.289 against 0.291,
    72 threads 0.256 against 0.249; 50 m 16 threads 1.036 against 1.132,
    72 threads 0.859 against 0.874.
- **Why left out:** with the rest of the speedups in, D gains nothing at
  100 m and up to 1.10× at 50 m and 16 threads. In return it adds a second
  code path for each closure: blocked and separate, with their own tests.
- **Taken from it:** part 1, the two kernel fixes:
  - E1, the update of e row-parallel and branch-free: `16ad00c9`,
    cherry-picked as `4d247af2`. Measured 2.63× at 1 thread, 1.30× at 72.
  - E2, the dual strain rate without bounds checks in the bulk: `cf704ba3`,
    cherry-picked as `9f7c68e1`. Measured 2.40× at 1 thread, 2.08× at 72.
- **Not taken:**
  - the roadmap section of #554, part 1 (`6c61a9da`, `71329614`, `0dc066a0`);
  - the review commits of the pass (`7c638734`, `949ab673`, `ef8f8337`,
    `411aefa1`, `c7892591`, `96b2567e`). The `_sqrt_nonneg` note and the
    `anelastic_cbl` validation fix (`411aefa1`: it read κt from the
    closure's scratch) may matter without the pass. Check them before
    discarding.
- **Reviving it:** rebase onto master and re-target the PR to master. The
  #533 driver on master is the pre-generic one, so `3c89e281` (the generic
  driver) is the first commit to port. Measure at 20 m (1000 × 1000 × 150)
  first: D was never measured there.

### D2. N² and the strain rate inside the TKE pass (#556, PR #561)

- **Where:** `issue/556-tke-n2-strain`, head `3d6ef342`, PR #561 (base
  `issue/554-tke-blocked`). Needs D1.
- **What:**
  - Without immersed walls, the TKE pass computes N² and the dual strain
    row by row instead of as whole-field loops ahead of the pass.
  - It also refactors N² into one point body for both grids
    (`Operators.n2_args`, `_dz_level`, `_n2_dz` for the `StretchedGrid`).
    That removes the `StretchedGrid`'s own N² loop and chooses the moisture
    variant in one place (`PrognosticSGS._with_n2_args`).
- **Measured:** part of group D (see D1).
- **Why left out:** it builds on D1.
- **Worth keeping without the pass:** the N² refactor is a
  one-source-of-truth improvement: two N² loops become one body. It could
  be extracted without the pass.

### D3. Urban tracer SGS diffusion as one cache-blocked pass (#559, PR #562)

- **Where:** `issue/559-tracer-diffusion-blocked`, head `11db7226`, PR #562
  (base `issue/554-tke-blocked`). Needs D1's generic driver.
- **What:** `Operators.blocked_eddy_diffusion!` makes the masked and
  weighted scalar diffusion of the urban tracers one blocked update.
  `sgs_diffuse!(::TracerSGSDiffusion)` uses it unless
  `loop_fusion = "never"`. The scratch is built with the struct, so a call
  allocates 0 B.
- **Measured:** part of group D (see D1).
- **Why left out:** with D.
- **Taken from it:** E5, the masked diffusion vectorises. `_open` and
  `_vertical_divergence` did not inherit `@inbounds`, so every mask read was
  bounds-checked.
  - Commits: `33b3ed00` and the review's `811a9b8e`, cherry-picked as
    `22505bde` and `fdbd6948`.
  - Measured 1.78× at 1 thread and 1.21× at 72. On the development machine
    (200 × 200 × 60): 11.3 → 4.8 ms at 1 thread.
- **Not taken:** the roadmap section of #559 (`5ba96cfc`, `d6044d38`,
  `11db7226`).

### D4. The Reynolds-stress closure's update as one cache-blocked pass (#560, PR #565)

- **Where:** `issue/560-rsc-blocked`, head `232829b9`, PR #565 (base
  `issue/556-tke-n2-strain`). Needs D1 and D2.
- **What:** with transport, `advance_sgs!(::ReynoldsStressClosure)` as one
  blocked pass, built from the TKE pass's pieces generalised to several
  scalars.
- **Measured:** the PR's own finding is that the pass does not pay as
  planned. The update is limited by arithmetic: the 6 × 6 backward-Euler
  solve costs about 350 ns per cell. `if_possible` takes it only with
  centred advection on several threads. The PR asked for a maintainer
  decision on that rule.
- **Why left out:** with D, and it does not pay on its own.
- **Taken from it:** E6, the Reynolds-stress source update allocated on the
  heap in every cell.
  - Commit: `ea8c3fcb`, cherry-picked as `ad83640d`.
  - Measured: the update is 1.20× faster at 1 thread and 5.00× at 72.
    Allocation per call drops from 857 MB to 0 B at 1 thread and to 0.1 MB
    at 72 (run 1, node 17).

### E3 and E4 (#557, PR #564): code taken, documentation not

- **Where:** `issue/557-branchfree-flux`, head `f34e2fea`, PR #564 (base
  `issue/556-tke-n2-strain`).
- **Taken:**
  - E3, branch-free face values (`7bb2cd6a` → `536cbc4e`), and its review
    test (`f7ed09ad` → `1239aa52`). Measured 0.97–1.00×: no gain at any
    thread count.
  - E4, the moist N² row loop vectorises (`0ff95557` → `ef6a98a2`).
    Measured 1.34× at 1 thread and 0.98× at 72.
- **Not taken:** the roadmap and `ka_port.md` text of #557 (`cec6b35b`,
  `f34e2fea`).

### F. The cache-blocked loop split in j as well (#558, PR #563)

- **Where:** `issue/558-blocked-j-split`, head `ee965c24`, PR #563 (base
  `issue/554-tke-blocked`). Needs D1's generic driver.
- **What:** the blocked driver cuts the interior into slabs in k **and**
  row ranges in j (`SweepBlock`, `sweep_blocks`, `edge_rows`), so the
  useful thread count is no longer capped at `nk ÷ 2lz`. Both the predictor
  sweep and the TKE pass use it.
- **Measured:** never on Fritz. Group F was dropped from run 1
  (maintainer, 2026-11-10). The PR has the development machine's numbers.
- **Why it may matter later:**
  - At 20 m and 72 threads (#566, `REPORT_threads_20m.md`, item 13), the
    fused sweep's `_b_sweep_slab!` takes 21 % of the step with only 37.6 of
    72 threads busy. The proposed fix there, splitting in j as well, is what
    this PR does.
  - Reviving it means porting it onto master's #533 driver, or reviving D1
    first.
- **Ported for the predictor sweep** (#566, branch `issue/566-sweep-jsplit`):
  master's fused sweep (`blocked_predictor!`) now splits in j with the
  same rule (`sweep_blocks`, edge rows, a third launch), without D1's
  generic driver; see the roadmap section "Fused sweep split in j (#566
  thread utilisation, port of #563)". What remains here is the TKE pass
  part, which needs D1.

## Thread utilisation (#566 follow-up): what was left out

### A2. The step-start copy as a threaded launch

- **Where:** commit `158a139b` on `issue/thread-utilisation`, reverted
  in `4326dbf6` before the PR. `copy_storage!` launched `_copy_point!`
  over the array above 4 194 304 points (`_COPY_THREADED_MIN`), with a test
  in `test/thread_utilisation.jl`.
- **Why it was built:** at Berlin 20 m on 72 threads, the copy of the
  state at the start of each step ran on one thread.
- **Measured** on a 24-thread desktop, 1006 × 1006 × 156, after #577:
  - copy: 61.5 ms threaded against 61.5 ms for `copyto!` at 4 threads, and
    72.7 against 60.6 ms at 24;
  - zeroing: 41.9 ms threaded against 17.5 ms for `fill!`.

  One thread already saturates that machine's memory bandwidth.
- **What is missing:** the same comparison on a two-socket Fritz node.
  There a single thread reaches a small part of the node's bandwidth, and
  the pages are spread over both sockets by the first touch. Reapply the
  commit if the launch wins there.

## Tools that are not in master

### The launch trace (#566)

- **Where:** `bench/hook` (`f4e22fc5`) and `bench/hook-dyn` (`efb5ec4c` and
  the test fix `5444aa27`). The same commits are on `bench/all`
  (`8a360b01`, `6da4b9e4`, `21196f5e`).
- **What:** `Backends.set_launch_trace!`, off unless `AEOLUS_LAUNCH_TRACE=1`.
  It measures wall and busy time per launch name, idle thread time inside
  launches, and the serial time between them.
- **Status:** `benchmarks/hpc/bench_run.jl` (`trace` and `profile` modes)
  needs it. Every thread-utilisation finding of #566 comes from it. No PR
  has been opened. It belongs with the #566 PR, or in its own, so that
  later speedups can be measured the same way on master.

### The HPC benchmark itself (#566)

- **Where:** `issue/566-hpc-benchmark`. It has `benchmarks/hpc/`: the
  drivers, the scripts, the results of run 1 and of the 20 m reruns, the
  reports and the handoffs.
- **Status:** no PR (HANDOFF_scaling20m.md §6, item 7).

## Measurement builds (not for merging)

`bench/master`, `bench/all`, `bench/only-{A,B,C,D,E,543}`,
`bench/all-minus-{A,B,C,D,E}` and `bench/all-revert-E{4,5,6}` are the run 1
builds. They are kept so that run 1 can be reproduced
(`benchmarks/hpc/HANDOFF.md` §3).
