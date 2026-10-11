# Running the tests

The suite has two tiers and two ways to run them. The rule (#316) is:
**a PR needs the fast tier green; the merge train runs the full tier
before merging.** An agent that changes physics validated by a full-only
file runs that file as well and quotes it.

```bash
julia --project=. test/run.jl                          # fast tier, parallel (the PR gate)
julia --project=. test/run.jl --only era6_gabls3.jl    # one or more files, any tier
AEOLUS_TEST_TIER=full julia --project=. test/run.jl    # full tier, parallel (merge train)
julia --project=. test/runtests.jl                     # serial; honours AEOLUS_TEST_TIER
AEOLUS_TEST_TIER=full julia --project=. test/runtests.jl
```

`Pkg.test()` runs `test/runtests.jl` and therefore also honours
`AEOLUS_TEST_TIER`.

## Tiers

| Tier | Selected by | What runs |
| :-- | :-- | :-- |
| `fast` (default) | `AEOLUS_TEST_TIER` unset or `fast` | every file except those tagged `tier = "full"` |
| `full` | `AEOLUS_TEST_TIER=full` | every file of the include list |

The full-only files are validation mini-simulations: the GABLS1-3 LES
(configurations of the model), the log-layer channels, the flat plate, the
Kelvin–Helmholtz growth rates, the scale-dependent
Lagrangian model's channel runs, the Comte-Bellot–Corrsin benchmark, the
SGS closures against filtered DNS, the BUBBLE / CAPITOUL campaigns and
the patchy-surface LES. Each has a one-line reason in
`test/test_tags.toml` naming the fast file that keeps its module
covered. Every `src` module keeps fast coverage (checked by searching
each module's name and exports in the fast files). The Berlin example
has no tests of its own (#320); the urban run type it configures
(`Aeolus.UrbanCase`, #438) is tested in `urban_run.jl` (fast: a
synthetic city) and `urban_checks.jl` (full: the end-to-end runs on the
Berlin configs, formerly the by-hand `examples/urban/berlin_checks.jl`).

The granularity is the file. A few full-only files also hold closed-form
checks next to their simulations (the Blasius solution in
`era6_flat_plate.jl`, the maximum sustainable heat flux in
`era6_collapse.jl`); those checks run in the full tier only until the
files are split.

## The single include list

`test/runtests.jl`'s `include` lines are **the** list of test files.
Nothing else lists them: the serial run, the tiers and the parallel
runner all read that file. A new test file needs its include line and
nothing more. Untagged files are fast, light and not exclusive.

Tag a file in `test/test_tags.toml` only when

- it is a validation mini-simulation that belongs in the full tier
  (`tier = "full"` plus a one-line `reason`; the runner refuses a
  full-only entry without one), or
- it asserts wall-clock or CI-budget times (`exclusive = true`), or
- it is long (`weight = "heavy"`, `seconds = ...`), so the runner starts
  it first.

```toml
["era6_gabls3.jl"]
tier = "full"
weight = "heavy"
seconds = 113
reason = "GABLS3 4 h LES on the model (gabls3.toml); LargeScaleForcing and wall models stay fast in ..."
```

## The parallel runner

`test/run.jl` (standard library only) runs every selected file in its
own Julia process, plus one job, `runtests.jl`, for the inline
`@testset`s written directly in `test/runtests.jl`.

- **Order.** Heavy files first, longest first (by the last measured
  times in `.testruns/timings.toml`, else the tag's `seconds`).
- **Jobs.** By default the number of free cores: CPU threads minus the
  rounded 1-minute load, at least 1, divided by the threads per job and
  capped by available memory (`--mem-per-job`, default 1 GB; the
  heaviest file peaks at 2.2 GB, the median at 0.5 GB). So two agents
  running suites at once do not oversubscribe the machine.
- **Exclusive files** (`era6_apriori_sgs.jl`, `era6_cbc_decay.jl`,
  `thread_invariance.jl`) run one at a time after the parallel batch,
  because they assert their own wall time.
- **Logs.** One log and one result file per job in the run folder,
  `.testruns/<yyyymmdd-HHMMSS>-<tier>/` (gitignored), plus
  `summary.txt` and `summary.toml`. The path of a failing file's log is
  printed as soon as it finishes and again in the summary.
- **Summary.** Per file: time and pass / fail / error / broken counts.
  Per top-level `@testset` block ("Era IV", "Berlin case", ...): the same
  counts as the serial run prints.
- **Exit code.** Non-zero on any failure, error, error outside a
  `@testset`, or crashed job.

| Option | Environment | Default |
| :-- | :-- | :-- |
| `--tier fast\|full` | `AEOLUS_TEST_TIER` | `fast` |
| `--jobs N` | `JOBS` | free cores, capped by memory |
| `--threads T` | `AEOLUS_TEST_THREADS` | 1 (Julia threads per job; BLAS threads follow) |
| `--mem-per-job GB` | `AEOLUS_TEST_MEM_GB` | 1 |
| `--only a.jl,b.jl` | | all files of the tier |
| `--log LEVEL` | `AEOLUS_LOG` | `warn`: the jobs' log level (`debug`, `info`, `warn`, `error`, `off`) |
| `--outdir DIR` | | `.testruns/<time>-<tier>/` |
| `--list` | | print the plan and exit |

`--threads` exists for the KernelAbstractions port (#291), after which
heavy files can use more than one thread.

## Logging in the suite

The header of `runtests.jl` installs `Aeolus.test_logger()`: library
events at warning level and up are printed (with a short timestamp and
their `subsystem[domain]` tag), info and debug are not. Set
`AEOLUS_LOG=debug` (or pass `--log debug` to `test/run.jl`) to see
everything. A test that checks an event captures it with its own logger
(`@test_logs`, `Test.TestLogger`) instead of reading stderr; see
[Logging](api/logging.md) (#333).

## How the selection works

`test/runner/selection.jl` parses `test/runtests.jl` and evaluates a
pruned copy of it with `Base.include(mapexpr, Main, path)`: the selected
`include` lines survive inside the `@testset` blocks that enclose them,
with the original line numbers and include paths, so a file sees exactly
the environment it sees in the serial run. A worker
(`test/runner/worker.jl`) evaluates one file this way inside an outer
test set, reads the counts of each top-level block, and writes them to
its result file.

The serial run uses the same code. A five-line header at the top of
`runtests.jl` does nothing in the full tier (the file then runs as it
always did); in the fast tier it runs the pruned copy and exits. In an
interactive session the header ignores the tier and runs everything,
because exiting would close the REPL.

## Determinism

Processes share no state, so a file's results do not depend on which
files ran before it. On the fast tier the runner and the serial run give
identical pass / fail / error / broken counts per block (measured below).

One known source of run-to-run variation is independent of the runner:
the pressure solvers plan their FFTs with `FFTW.MEASURE`, which picks
algorithms by timing, so round-off and with it chaotic LES statistics can
differ between runs. `era6_gabls3.jl` has a `@test_broken` on the jet
height that sits within one grid spacing of its envelope; sometimes it
passes, which `@test_broken` reports as an error ("Unexpected Pass").
That is the expectation's margin, not the runner: run alone
(`--only era6_gabls3.jl --jobs 1`, load 1.6-5.2) it passed unexpectedly
in 1 of 3 runs, and in parallel in 1 of 2 full-tier runs. Until the
assertion is settled, a red full tier with only this error is this
flakiness.

## Thread invariance (KernelAbstractions port)

`test/thread_invariance.jl` checks that the code gives identical bits at
N threads and at 1 thread (KA-a, #294; [the kernel port](design/ka_port.md)).
In the normal run it compares `CPUBackend(serial = true)` with the
threaded backend in process, which is meaningful under `--threads 4`.
Its subprocess level is opt-in. Every port PR runs it and quotes the
result:

```bash
AEOLUS_THREAD_TESTS=1 julia --project=. test/run.jl --only thread_invariance.jl
```

It starts `julia -t 1/4/24` (`AEOLUS_THREAD_COUNTS`) and runs the
sections in `AEOLUS_THREAD_SECTIONS` (default
`primitives,dns,reductions,smoke,flagship`). Every hash must match the 1-thread
run. At KA-a it produced 2261 hashes, all identical, in about 2 min per
process.

## Measured times

24-core machine, 2026-10-06, other agents' suites running alongside
(the 1-minute load is quoted).

| Run | Wall | Jobs × threads | 1-min load at start |
| :-- | --: | :-- | --: |
| fast tier, serial `runtests.jl` | 18.7 min | 1 × 1 | 4.6 (10.3 at the end) |
| fast tier, `test/run.jl` | 5.5 min | 16 × 1 (free cores, capped by memory) | 2.7 |
| full tier, `test/run.jl` | 7.9 min | 18 × 1 | 1.8 |
| full tier, `test/run.jl`, earlier run | 11.1 min | 8 × 1 | 5.2 |
| full tier, serial `runtests.jl` | about 48 min | 1 × 1 | not re-measured; figure from #316 (#259 branch) |

Counts per top-level block, fast tier (serial and parallel identical):

| Block | Pass | Broken |
| :-- | --: | --: |
| Aeolus.jl | 182 | 0 |
| Chemistry: urban NOx–O3–VOC mechanism (#135) | 3718 | 0 |
| Era II | 2148 | 0 |
| Era III | 4912 | 0 |
| Era IV | 47392 | 0 |
| Era V | 19269 | 3 |
| Era VI | 3191 | 3 |
| Berlin case | 3378 | 0 |
| **Total** | **84190** | **6** |

The full tier adds 825 assertions (85021 in all: 84997 pass, 24 broken;
in the earlier run one of them was the GABLS3 unexpected pass above, counted as an error).
The fast tier is bounded by its longest files (`berlin_smoke.jl`, run
alone, 125 s; `berlin_dynamics.jl`, 115 s) and the exclusive files run
after the batch (`era6_apriori_sgs.jl`, 59 s alone), so more jobs do not
shorten it much.

### The Berlin suite removed (#320)

The 13 `test/berlin_*.jl` files (the "Berlin case" block, 4162
assertions) were removed on 2026-10-07; their checks of `src/` physics
are in module tests, the example's end-to-end runs in
`examples/urban/berlin_checks.jl` (run by hand; `test/urban_checks.jl`
since #438). Fast tier, `test/run.jl
--jobs 4`, one thread per job, other agents' suites alongside:

| | Wall | Jobs | Assertions (pass + broken) | 1-min load at start |
| :-- | --: | --: | --: | --: |
| before (master + the KA work of #303, 2026-10-07 14:15) | 21.0 min | 134 | 100 062 | 2.8 |
| after | 12.6 min | 124 | 97 185 | 6.2 |

Before, the Berlin jobs took 1260 s of job time (`berlin_parent.jl`
234 s, `berlin_smoke.jl` 222 s exclusive, the others 28–97 s each) and
were 7 of the 10 slowest jobs. After, the slowest are `config_berlin.jl`
(108 s), `era6_large_scale_forcing.jl` (83 s) and `era6_apriori_sgs.jl`
(73 s, exclusive). Assertions per file, before → after, of the files that
took over Berlin checks: `era2_anelastic.jl` 316 → 1483 (the reference
state checked level by level on three profiles), `era2_anelastic_cbl.jl`
new, 3 (71 s), `era5_surface_tiles.jl` 64 → 79, `input_static.jl`
114 → 121, `era6_relaminarisation.jl` 66 → 73, `era5_soil_thermal.jl`
2499 → 2504, `era2_multigrid.jl` 35 → 39, `era6_open_near_boundary.jl`
71 → 74, `era6_sigma_sgs.jl` 56 → 58, `config_examples.jl` +1;
`era6_nudging_subsidence.jl` (68, full) split into 58 fast and
`era6_gabls2.jl` (10, full, 92 s). The full tier was not re-run for #320.

### Redundant mini-simulations pruned (#317)

An audit of the test files for duplicated or overlapping
mini-simulations, after the Berlin suite (the smoke variants that
exercised the same code paths) was removed in #320:

- `era6_large_scale_forcing.jl`: the bit-for-bit check of
  `GeostrophicNudging` against the former `add_geostrophic_wind!` ran its
  oracle inside 100 Berlin smoke steps, most of the file's time. The forcing is
  a body-force hook either way; it now runs as the hook of 100 SSP-RK3
  and 100 forward-Euler steps of a rotating, sheared, stratified,
  turbulent box with Smagorinsky SGS, with ``τ`` = 60 s so that the
  forcing moves the mean wind by O(1) (``⟨u⟩`` 3.90 → 1.10 m/s against
  ``U_g = 1`` over the 200 s). A deliberately reordered oracle
  fails both runs. The Berlin version also asserted every tracer bit for
  bit; that assertion was dropped as redundant, not replaced: both
  forcings write only ``u`` and ``v``, never a tracer, so a tracer can
  differ between the two runs only through the state that advects,
  diffuses and sources it (``u, v, w, θ, p`` and the eddy diffusivity
  computed from them), and that state is still compared with `==`; the
  box carries no tracers. The file went from 78.3 s to 12.6 s
  (fast-tier runs named below) and lost its heavy tag.
- `test/support/kolmogorov.jl`: one builder, `kolmogorov_field!`, for the
  ``k^{-5/3}`` initial field that `era2_sgs_numerics.jl`,
  `era6_lagrangian_sgs.jl`, `era6_lasd.jl` (named copies),
  `era6_amd_vreman.jl` and `era6_sigma_sgs.jl` (inline copies) each
  built. The fields are bit-identical to the copies; no run time changes.

Audited and kept, because the runs assert different physics or different
averaging windows: the 24³ half channel of `era6_neutral_abl.jl`
(wall stress over 8–20 ``H/u_*``, Φ_m of ``C_s = 0.17``) and of
`era6_wall_models_channel.jl` (first cell against ``h_{wm}`` and
filtering over 6–12 ``H/u_*``), whose tolerances were set from seed
studies over those windows; the laminar Poiseuille channel of
`era6_relaminarisation.jl` (laminar consistency of the σ closures,
``Ri_τ = 50``; the former `era6_stratified_channel.jl`, Nu = 1 at
``Ri_τ = 5``, was removed by ONE-g, #456); the 96³ (`era6_apriori_sgs.jl`) and 48³
(`era6_sgs_validation.jl`) DNS fixtures; the config tests' runs of the
legacy in-code builds, which are the bit-identity oracles of the config
migration. The slowest fast file, `config_berlin.jl` (111.9 s in the
fast-tier run below, 105.0 s alone with `--only`), spends its
time building the case and starting a fresh process, not stepping (12
Berlin steps take 0.5 s).

Fast tier after #317 (`test/run.jl --jobs 4`, run folder
`.testruns/i317-fast3`, 1-min load 2.7 at start, other agents' suites
alongside): 12.6 min, 128 jobs, 100 981 assertions (100 973 pass,
8 broken). It is not comparable with the 12.6 min of #320 above: master
gained four KA jobs in between. The job time saved is the 66 s of
`era6_large_scale_forcing.jl`: 78.3 s in the fast tier of master at
7da2b073, the last master this branch merged before the change (run
folder `.testruns/20261007-174014-fast` of the steward worktree, 1-min
load 2.7), against 12.6 s in `i317-fast3`. That is one run each, both
with other agents' suites alongside; the file's time in a fast tier
depends on the load (83 s in #320's run above). The full
tier was not re-measured for #317. The only full-only file #317 touches
is `era6_lasd.jl`, and only to swap in the shared `kolmogorov_field!`
fixture (bit-identical field, unchanged assertion counts; 90.1 s with
`--only` after the swap, no before time was recorded). The
expected full-tier saving is therefore the same 66 s of
`era6_large_scale_forcing.jl`; that is an estimate, not a measurement.
