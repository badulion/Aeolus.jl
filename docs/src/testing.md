# Running the tests

Aeolus has two suites, run by the same parallel runner:

- **The tests** (`test/`): checks of code. Analytic and closed-form values,
  identities, conservation over a few steps, interfaces, configuration and
  I/O, short runs that exercise a code path, regression fingerprints,
  thread invariance. **A PR needs them green.** `Pkg.test()` runs them.
- **The validation runs** (`validation/`): simulated cases that confirm the
  physics works and stays stable (GABLS, channels and log laws, decaying
  turbulence, growth rates, long nest runs, observed campaigns, published
  benchmarks). They are **not a gate**; see [Validation runs](validation.md).

```bash
julia --project=. test/run.jl                          # the tests, in parallel (the PR gate)
julia --project=. test/run.jl --only tke.jl       # one or more test files
julia --project=. test/runtests.jl                     # the tests, serially
julia --project=. validation/run.jl                    # the validation runs, in parallel
julia --project=. validation/run.jl --only gabls1.jl
```

The fast/full tiers (#316) were replaced by the two suites: what the full
tier added, and every validation run that the fast tier carried, is now in
`validation/`. `AEOLUS_TEST_TIER` is ignored (the runner warns), and a
`tier` key in a tag table is an error that points here.

## Test or validation?

A file, or a `@testset` inside one, is **validation** when its point is that
a *simulated case* behaves physically or stays stable: it integrates a model
for many steps or a long simulated time (or solves a case iteratively) and
asserts emergent results — statistics, profiles, log laws, growth rates,
agreement with DNS, observations, a published benchmark or a reference run,
no blow-up over N steps. Everything else is a **test**: it checks a unit of
code or a code path. Cost is a hint, not the criterion: a long laminar run
against an exact solution is a test (it verifies the discretisation); a
single evaluation of a closure against observed numbers, with nothing
simulated, is a test too.

Mixed files are split at `@testset` granularity: the test part stays in
`test/<file>.jl`, the validation part goes to `validation/<file>.jl` (same
name). A helper both parts need lives once, in `test/support/` or
`test/fixtures/`. A module whose only checks were in a validation run gets a
small test of its own (an analytic limit, a budget over a few steps), so
every `src` module keeps a test.

## The single include list

Each suite's entry file is **the** list of its files:
`test/runtests.jl` and `validation/runvalidation.jl`. Nothing else lists
them: the serial run and the parallel runner both read the entry file. A
new file needs its include line and nothing more; untagged files are light
and not exclusive. Both entry files first include
`test/support/environment.jl` (packages, FFTW `ESTIMATE` planning, the
[tolerance helpers](#Comparing-results:-tolerances,-not-bits), the test
logger), so a file sees the same environment in either suite.

Tag a file in its suite's table (`test/test_tags.toml`,
`validation/validation_tags.toml`) only when

- it asserts wall-clock or CI-budget times (`exclusive = true`), or
- it is long (`weight = "heavy"`, `seconds = ...`), so the runner starts
  it first.

```toml
["apriori_sgs.jl"]
weight = "heavy"
exclusive = true
seconds = 87
reason = "asserts the filtered-DNS fixture wall time, so it runs alone"
```

`reason` is an optional line of text; in the validation table it says what
the case validates and which test file keeps its module tested.

## The parallel runner

`test/runner/runner.jl` (standard library only; entry points `test/run.jl`
and `validation/run.jl`) runs every file of the suite in its own Julia
process, plus one job for the inline `@testset`s written directly in the
entry file (`runtests.jl`; the validation entry file has none).

- **Order.** Heavy files first, longest first (by the last measured
  times in `.testruns/timings.toml`, `.testruns/timings_validation.toml`,
  else the tag's `seconds`).
- **Jobs.** By default the number of free cores: CPU threads minus the
  rounded 1-minute load, at least 1, divided by the threads per job and
  capped by available memory (`--mem-per-job`, default 1 GB; the
  heaviest file peaks at 2.2 GB, the median at 0.5 GB). So two agents
  running suites at once do not oversubscribe the machine.
- **Exclusive files** run one at a time after the parallel batch,
  because they assert their own wall time.
- **Logs.** One log and one result file per job in the run folder,
  `.testruns/<yyyymmdd-HHMMSS>-<suite>/` (gitignored), plus
  `summary.txt` and `summary.toml`. The path of a failing file's log is
  printed as soon as it finishes and again in the summary.
- **Summary.** Per file: time and pass / fail / error / broken counts.
  Per top-level `@testset` block ("Clouds and microphysics", "Turbulence", ...): the same
  counts as the serial run prints.
- **Exit code.** Non-zero on any failure, error, error outside a
  `@testset`, or crashed job.
- **`--only`** takes files of this suite; naming a file of the other suite
  is an error that says which runner to use.

| Option | Environment | Default |
| :-- | :-- | :-- |
| `--jobs N` | `JOBS` | free cores, capped by memory |
| `--threads T` | `AEOLUS_TEST_THREADS` | 1 (Julia threads per job; BLAS threads follow) |
| `--mem-per-job GB` | `AEOLUS_TEST_MEM_GB` | 1 |
| `--only a.jl,b.jl` | | all files of the suite |
| `--log LEVEL` | `AEOLUS_LOG` | `warn`: the jobs' log level (`debug`, `info`, `warn`, `error`, `off`) |
| `--outdir DIR` | | `.testruns/<time>-<suite>/` |
| `--list` | | print the plan and exit |

`--threads` exists for the KernelAbstractions port (#291), after which
heavy files can use more than one thread.

## Logging in the suite

`test/support/environment.jl` installs `Aeolus.test_logger()`: library
events at warning level and up are printed (with a short timestamp and
their `subsystem[domain]` tag), info and debug are not. Set
`AEOLUS_LOG=debug` (or pass `--log debug` to the runner) to see
everything. A test that checks an event captures it with its own logger
(`@test_logs`, `Test.TestLogger`) instead of reading stderr; see
[Logging](api/logging.md) (#333).

## How the selection works

`test/runner/selection.jl` parses the suite's entry file and evaluates a
pruned copy of it with `Base.include(mapexpr, Main, path)`: the selected
`include("file.jl")` lines survive inside the `@testset` blocks that
enclose them, with the original line numbers and include paths, so a file
sees exactly the environment it sees in the serial run. Setup includes
written with `joinpath` (the environment) are statements, not list
entries, and always survive. A worker (`test/runner/worker.jl`) evaluates
one file this way inside an outer test set, reads the counts of each
top-level block, and writes them to its result file.

## Determinism

Processes share no state, so a file's results do not depend on which
files ran before it; the runner and the serial run give identical
pass / fail / error / broken counts per block.

The suites plan FFTs with `FFTW.ESTIMATE` (`test/support/environment.jl`),
so plans, and with them round-off, do not depend on timing. Chaotic
statistics can still differ across machines and thread counts; the
validation runs assert them within tolerances that say so (see
[Validation runs](validation.md)).

## Comparing results: tolerances, not bits

Results need not be bit-identical across thread counts, machines, restarts
or refactors (maintainer decision 2026-10-16, epic #493). The tests compare
simulated results with `test/support/agreement.jl` (#495), which
`test/support/environment.jl` includes for every file of both suites:

- `agree(a, b)`: two computed fields (or numbers) are the same computation
  up to round-off, `max |a − b| ≤ ROUNDOFF_RTOL · max |b|`, NaN in the same
  places. `ROUNDOFF_RTOL = 1e-10` was measured: the nested Berlin smoke at
  1 and at 12 BLAS threads differs by at most 7.1e-14 of a field's size,
  and FFTW's MEASURE plans against ESTIMATE move the 40 simulation-config
  variants by at most 4.9e-16.
- `values_agree(a, b)`: the same, recursively through dictionaries,
  tuples, named tuples and snapshot structs; integers, symbols and strings
  exactly.
- `fingerprint(a)` / `fingerprints_differing(got, ref)`: a regression
  fixture in place of a digest of the bits: length, non-finite count,
  mean, mean |·|, RMS, min, max and 16 sampled values, compared within
  `ROUNDOFF_RTOL` of the field's size. No committed test fixture uses it
  any more (the simulation and Berlin fingerprint fixtures went with the
  test reduction of the validation split); `test/agreement.jl` tests it.

Exact comparison stays where data is copied, not computed: a checkpoint read
back, a halo filled by copies, a state restored from a snapshot, a parsed
configuration, a driver file regenerated from the same input, and integer
and class data (masks, counts, step numbers).

## Thread invariance (KernelAbstractions port)

`test/thread_invariance.jl` checks that the code gives the same results at
N threads and at 1 thread, within `ROUNDOFF_RTOL` (KA-a, #294;
[the kernel port](design/ka_port.md)): a race between threads moves a
result by far more than round-off.
In the normal run it compares `CPUBackend(serial = true)` with the
threaded backend in process, which is meaningful under `--threads 4`.
Its subprocess level is opt-in. Every port PR runs it and quotes the
result:

```bash
AEOLUS_THREAD_TESTS=1 julia --project=. test/run.jl --only thread_invariance.jl
```

It starts `julia -t 1/4/24` (`AEOLUS_THREAD_COUNTS`) and runs the
sections in `AEOLUS_THREAD_SECTIONS` (default
`primitives,dns,reductions,smoke,flagship`). Every result must agree with
the 1-thread run (fields as fingerprints within `ROUNDOFF_RTOL`, integer data
exactly). At KA-a it produced 2261 hashes, all identical, in about 2 min
per process.

## Measured times

### The reduced suite (2026-11-05)

After the validation runs moved to `validation/`, the test suite was cut
to the checks that matter (maintainer request: fewer than 10 000
executed assertions). Kept: analytic values, published reference values,
identities, conservation and budgets over a few steps, interfaces and
refusals, one short code-path run per configuration. Cut: parameter
sweeps, resolution studies, long runs, repeated configurations,
bit-identity and allocation checks, and the ported-kernel comparisons of
the KA work (`test/ka_*.jl`). Per-element `@test` loops became one
aggregate assertion each; a failure then names the group, not the
element.

Serial `julia --project=. test/runtests.jl`, 24-core machine, 1-minute
load 6.4 at the start, 110 files, all pass:

| Block | Executed assertions | Time |
| :-- | --: | --: |
| Aeolus.jl | 1929 | 6 min 53 s |
| Chemistry: urban NOx–O3–VOC mechanism (#135) | 31 | 0.6 s |
| Numerics | 376 | 2 min 09 s |
| Radiation | 316 | 17 s |
| Clouds and microphysics | 506 | 17 s |
| Land, urban and surface | 782 | 57 s |
| Turbulence | 159 | 14 s |
| **Total** | **4099** | **652 s wall** (with `using Aeolus`) |

Before the reduction the same suite executed 123 087 assertions in 158
jobs (17.5 min wall with `test/run.jl` at 16 jobs). Most of what is left
is compilation: the first model built from a configuration costs 65–90 s,
the first urban-site run 137 s, while a second run of either takes
seconds. That is why the `Aeolus.jl` block (configuration, components,
nesting and the urban site) dominates.

### Before the reduction

The figures below were measured under the fast/full tiers, before the
validation runs became a suite of their own; "full tier" is the test files
plus what is now `validation/`. The split itself is measured in
[Validation runs](validation.md#Measured).

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
| Numerics | 2148 | 0 |
| Radiation | 4912 | 0 |
| Clouds and microphysics | 47392 | 0 |
| Land, urban and surface | 19269 | 3 |
| Turbulence | 3191 | 3 |
| Berlin case | 3378 | 0 |
| **Total** | **84190** | **6** |

The full tier adds 825 assertions (85021 in all: 84997 pass, 24 broken;
in the earlier run one of them was the GABLS3 unexpected pass above, counted as an error).
The fast tier is bounded by its longest files (`berlin_smoke.jl`, run
alone, 125 s; `berlin_dynamics.jl`, 115 s) and the exclusive files run
after the batch (`apriori_sgs.jl`, 59 s alone), so more jobs do not
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
(108 s), `large_scale_forcing.jl` (83 s) and `apriori_sgs.jl`
(73 s, exclusive). Assertions per file, before → after, of the files that
took over Berlin checks: `anelastic.jl` 316 → 1483 (the reference
state checked level by level on three profiles), `anelastic_cbl.jl`
new, 3 (71 s), `surface_tiles.jl` 64 → 79, `input_static.jl`
114 → 121, `relaminarisation.jl` 66 → 73, `soil_thermal.jl`
2499 → 2504, `multigrid.jl` 35 → 39, `open_near_boundary.jl`
71 → 74, `sigma_sgs.jl` 56 → 58, `config_examples.jl` +1;
`nudging_subsidence.jl` (68, full) split into 58 fast and
`gabls2.jl` (10, full, 92 s). The full tier was not re-run for #320.

### Redundant mini-simulations pruned (#317)

An audit of the test files for duplicated or overlapping
mini-simulations, after the Berlin suite (the smoke variants that
exercised the same code paths) was removed in #320:

- `large_scale_forcing.jl`: the bit-for-bit check of
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
  ``k^{-5/3}`` initial field that `sgs_numerics.jl`,
  `lagrangian_sgs.jl`, `lasd.jl` (named copies),
  `amd_vreman.jl` and `sigma_sgs.jl` (inline copies) each
  built. The fields are bit-identical to the copies; no run time changes.

Audited and kept, because the runs assert different physics or different
averaging windows: the 24³ half channel of `neutral_abl.jl`
(wall stress over 8–20 ``H/u_*``, Φ_m of ``C_s = 0.17``) and of
`wall_models_channel.jl` (first cell against ``h_{wm}`` and
filtering over 6–12 ``H/u_*``), whose tolerances were set from seed
studies over those windows; the laminar Poiseuille channel of
`relaminarisation.jl` (laminar consistency of the σ closures,
``Ri_τ = 50``; the former `stratified_channel.jl`, Nu = 1 at
``Ri_τ = 5``, was removed by ONE-g, #456); the 96³ (`apriori_sgs.jl`) and 48³
(`sgs_validation.jl`) DNS fixtures; the config tests' runs of the
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
`large_scale_forcing.jl`: 78.3 s in the fast tier of master at
7da2b073, the last master this branch merged before the change (run
folder `.testruns/20261007-174014-fast` of the steward worktree, 1-min
load 2.7), against 12.6 s in `i317-fast3`. That is one run each, both
with other agents' suites alongside; the file's time in a fast tier
depends on the load (83 s in #320's run above). The full
tier was not re-measured for #317. The only full-only file #317 touches
is `lasd.jl`, and only to swap in the shared `kolmogorov_field!`
fixture (bit-identical field, unchanged assertion counts; 90.1 s with
`--only` after the swap, no before time was recorded). The
expected full-tier saving is therefore the same 66 s of
`large_scale_forcing.jl`; that is an estimate, not a measurement.
