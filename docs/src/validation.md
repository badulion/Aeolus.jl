# Validation runs

The validation runs are simulated cases that confirm the physics works and
stays stable: the GABLS intercomparisons, channels and log laws, decaying
isotropic turbulence, Kelvin–Helmholtz growth rates, long nest runs,
observed campaigns (BUBBLE, CAPITOUL, FLUXNET, ParisFog, ISDAC, TOGA COARE)
and published benchmarks (Comte-Bellot–Corrsin, ASHRAE 140, MU71). They are
a suite of their own in `validation/`, separate from
[the tests](testing.md) (maintainer decision, 2026-11-05).

```bash
julia --project=. validation/run.jl                          # all, in parallel
julia --project=. validation/run.jl --only gabls1.jl,gabls3.jl
julia --project=. validation/run.jl --list                   # the plan
julia --project=. validation/runvalidation.jl                # all, serially
```

The runner is the tests' runner (`test/runner/runner.jl`), with the same
options, logs (`.testruns/<time>-validation/`) and summary.

## Role: not a gate

- A PR needs **the tests** green, not the validation runs.
- A PR that changes physics a validation case covers runs that case
  (`validation/run.jl --only <file>`) and quotes its result in the PR body:
  pass / fail / broken counts and the numbers the case reports. A failure is
  reported, with whether the code or the expectation is wrong; it does not
  by itself block the merge.
- CI runs the whole suite weekly (Mondays 03:17 UTC) and on demand
  (`workflow_dispatch`), not on PRs or pushes. Its logs are kept 30 days as
  the `validationlogs-*` artifact.

## What is a validation run

A file or `@testset` belongs here when its point is that a *simulated case*
behaves physically or stays stable: it integrates a model for many steps or
a long simulated time (or solves a case iteratively) and asserts emergent
results against theory, DNS, observations, a published benchmark or a
reference run, or the absence of blow-up over many steps. A unit or a code
path, however long it runs, is a test; see
[Test or validation?](testing.md#Test-or-validation?).

Each validation file has a test counterpart that keeps its module tested:
the case's analytic limits, budgets over a few steps and interfaces are in
`test/`, usually in a file of the same name. The `reason` of a file's entry
in `validation/validation_tags.toml` names it.

## Layout

```
validation/
  runvalidation.jl          the include list (single list of validation files)
  validation_tags.toml      weight, exclusive, seconds, reason
  run.jl                    parallel entry point
  <file>.jl                 one file per case group, named like its test file
  support/  fixtures/  data/   validation-only drivers, fixtures and reference data
```

Fixtures and data that tests also use stay in `test/` (`test/support`,
`test/fixtures`, `test/data`) and are included from there with
`joinpath(@__DIR__, "..", "test", ...)`: one copy, never two.

## Adding a validation case

1. Put the case's unit checks (analytic limits, conservation over a few
   steps, the interface) in the module's test file.
2. Put the run in `validation/<file>.jl`, with a header saying what it
   validates, the reference, and which test file covers the module.
3. Add its include line to `validation/runvalidation.jl`, in its topic's
   block; tag it `heavy` with its measured `seconds` if it takes 45 s or
   more.
4. Quote its measured numbers in the PR body.

## Known behaviour

`gabls3.jl` has a `@test_broken` on the jet height that sits within
one grid spacing of its envelope; sometimes it passes, which `@test_broken`
reports as an error ("Unexpected Pass"). Under the former full tier it
passed unexpectedly in 1 of 3 runs alone and 1 of 2 parallel runs. Until
the assertion is settled, a validation run with only this error is this
flakiness, not a regression.

## Measured

The split (2026-11-05) left 50 validation files and 158 test jobs
(157 files plus the inline tests of `runtests.jl`). Before it, the fast
tier had 149 jobs and the full tier added 16 files.

- 16 files were validation-only (the former full tier) and moved whole;
  nine of them held analytic or code-path checks that no fast test had
  (the LASD kernels, the Transition diagnostics, the flat-plate fringe and
  Blasius solution, `development_length`, the TEB canyon closures, ...).
  Those came back to `test/` as tests.
- 34 test files held validation runs; those parts moved to
  `validation/<same name>.jl`.
- Where a function's only check was inside a moved run, a test was added
  (an analytic limit, or a budget over a few steps).

The suite has not yet been timed as a whole. The `seconds` in
`validation/validation_tags.toml` are the full-tier times of the whole
files (2026-10-06), which are upper bounds for the split files. The first
CI run records the actual times (`.testruns/timings_validation.toml`).
