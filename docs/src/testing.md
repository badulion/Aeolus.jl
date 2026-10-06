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

The full-only files are validation mini-simulations: the GABLS1 and
GABLS3 LES, the stratified and log-layer channels, the flat plate, the
Kelvin–Helmholtz growth rates, turbulence collapse, the scale-dependent
Lagrangian model's channel runs, the Comte-Bellot–Corrsin benchmark, the
SGS closures against filtered DNS, the BUBBLE / CAPITOUL campaigns and
the patchy-surface LES. Each has a one-line reason in
`test/test_tags.toml` naming the fast file that keeps its module
covered. Every `src` module keeps fast coverage (checked by searching
each module's name and exports in the fast files), and the fast tier
includes the Berlin smoke test.

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
reason = "GABLS3 4 h LES on the SBL driver; LargeScaleForcing and wall models stay fast in ..."
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
- **Exclusive files** (`berlin_smoke.jl`, `era6_apriori_sgs.jl`,
  `era6_cbc_decay.jl`) run one at a time after the parallel batch,
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
