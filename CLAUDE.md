# Aeolus.jl — working agreement

Operational contract for agents working in this repository. Rationale,
worked examples and recovery procedures are in
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md); this file is the short
version that must be followed.

## The loop

**One issue → one worktree → one branch → several commits → one PR.**

```bash
# 1. Claim an issue and create an isolated worktree for it
git worktree add .worktrees/issue-42 -b issue/42-p3-microphysics master
cd .worktrees/issue-42

# 2. Work. Commit after every self-contained subproblem (see below).
# 3. Run the tests before proposing anything (parallel), plus the
#    validation runs of any physics you touched.
julia --project=. test/run.jl
julia --project=. validation/run.jl --only gabls1.jl,gabls3.jl

# 4. Push the branch and open a PR. Then stop.
git push -u origin issue/42-p3-microphysics
gh pr create --fill --base master
```

Worktrees live in `.worktrees/` (gitignored). They exist so that several
agents can work at once without touching each other's checkout. **Never
work directly on `master`.**

> One hazard worth knowing: `git add -A` run from the *parent* checkout
> will see nested worktrees. `.worktrees/` is gitignored, which handles
> it — do not remove that entry.

## Commit cadence

Commit after each self-contained subproblem, not once at the end. A good
commit is one a reviewer can evaluate on its own: a helper function plus
its tests, one bug fixed, one module wired in.

For a typical physics phase that means something like:

```
P3: incomplete gamma function + validation against known values
P3: mass-dimension relation across the four rime regimes
P3: sedimentation with per-moment fall speeds
P3: radar reflectivity diagnostic
P3: wire into Aeolus.jl, add to runtests, fix export collision
```

PRs are merged with a **merge commit**, so these survive on `master`.
That is the point of asking for them — do not squash them yourself.

## Tests and validation runs are separate suites

**Tests** (`test/`) check code: analytic values, identities, conservation
over a few steps, interfaces, short code-path runs. **Validation runs**
(`validation/`) are simulated cases that confirm the physics works and
stays stable: GABLS, channels, decaying turbulence, campaigns, benchmarks
(maintainer decision, 2026-11-05; the fast/full tiers are gone).

- **A PR needs the tests green**: `julia --project=. test/run.jl`
  (parallel) or `julia --project=. test/runtests.jl` (serial).
- **Validation is not a gate.** When you touch physics a validation case
  covers, run it (`validation/run.jl --only <file>`) and quote its result
  in the PR body; report a failure honestly, it does not block by itself.
  CI runs the whole validation suite weekly.
- Never put a validation run in `test/`. A new case goes in
  `validation/<file>.jl` (its unit checks in the module's test file), with
  its include line in `validation/runvalidation.jl`.
- Each entry file's include list (`test/runtests.jl`,
  `validation/runvalidation.jl`) is the single list of its files; a new
  file needs only its include line. Tag it (`test/test_tags.toml`,
  `validation/validation_tags.toml`) only if it is heavy or
  timing-sensitive (`exclusive`).

Details: [`docs/src/testing.md`](docs/src/testing.md),
[`docs/src/validation.md`](docs/src/validation.md).

## Before opening a PR

- [ ] Tests green: `julia --project=. test/run.jl`, plus the validation
      runs of the physics you touched (`validation/run.jl --only`), quoted
- [ ] New code has tests that assert **physics**, not just that it runs
- [ ] Validation numbers measured and quoted in the PR body
- [ ] No new export collisions (see below)
- [ ] Documentation updated — see below
- [ ] New physics is tested on its own (module tests, synthetic setups) — see below

## Physics is tested on its own

Every piece of physics is tested **in its own module's tests**, with small
synthetic setups (analytic limits, conservation, published benchmarks) —
not through the Berlin case. The Berlin (Alexanderplatz) run,
`julia --project -m Aeolus run examples/urban/berlin_summer.toml`,
is an **example** of the urban site of the model (`Aeolus.Urban`,
`[urban] enabled`, #438, #454): couple new physics into it when that makes sense, but
nothing requires it, and it has no test suite of its own (maintainer
decision, 2026-10-06; the `test/berlin_*.jl` files and the coverage lists
are gone, #320).

- Do not add Berlin tests; put new physics tests in module tests.
- There is one model (`docs/src/design/single_model.md`): a run is a
  TOML configuration plus its static and dynamic input drivers. New
  physics goes in its module, with a thin component adapter in
  `src/components/` and its own TOML section; never a run type or a
  per-case script.
- Examples are configuration: a case directory `examples/<case>/` with
  `<case>.toml` (and its variants), `input/*.nc`, `data/` (reference
  data) and optional scripts that only read run output (maintainer rule,
  2026-10-09, #438). `test/examples_enforcement.jl` fails on any `.jl`
  under `examples/` that builds or steps a model, with no exemptions.
- After changing the urban site (`Aeolus.Urban`), run its tests and its
  end-to-end validation runs:
  `julia --project=. test/run.jl --only urban_run.jl,urban_checks.jl` and
  `julia --project=. validation/run.jl --only urban_checks.jl`
  (every coupling on, ledgers closed; the nest parent's smoke run and
  validations). A run's own checks are in its `output/results.toml`.

## Documentation is part of done

Docs are not a follow-up task. But **status updates happen after the PR
is merged, not inside it** — see the split below.

**In every PR** (the code and its documentation land together)

- Docstrings on new public functions, with references for any non-obvious
  physics.
- A new module gets an `@autodocs` entry in `docs/src/api/` and a line in
  `docs/make.jl`.
- If behaviour or an interface changed, update the affected theory or
  design page under `docs/src/`.

**After the PR is merged** — a separate *closeout* change, never part of
the feature PR:

- When a phase's last issue is merged: tick the phase (✅) in the Status
  checklist of `docs/src/design/roadmap.md` and write in its section
  **what was actually built and the numbers it was measured at**; record
  any bug the tests caught and any test *expectation* that proved wrong.
- When a milestone's last issue is merged: close the milestone, tick it in
  the roadmap's Status checklist and in the README roadmap, capability summary, validation table and test
  count, add or extend the topic's theory page, and rebuild the docs
  (`julia --project=docs docs/make.jl`) checking for missing-docstring
  warnings.

Two reasons status updates are kept out of the feature PR. A phase cannot
honestly be marked SHIPPED while the PR delivering it is still open. And
`roadmap.md` and `README.md` are shared files — if every
phase-completing branch edited them, concurrent agents would collide on
exactly the two files most likely to be touched at once.

Closeout is done by the maintainer on `master`, or by a small dedicated
closeout PR. Agents do not push to `master`.

## What an agent may and may not do

| May | Must not |
| :-- | :-- |
| Create worktrees and branches | Merge any PR |
| Commit, push its own branch | Close issues |
| Open PRs, comment on issues | Push or force-push to `master` |
| Run tests, benchmarks, examples | Delete branches or worktrees it did not create |
| Edit its own PR | Rewrite published history |

Review and merge are the maintainer's. Stop after opening the PR.

## Conventions that are not negotiable

**One source of truth.** If two modules need the same physics, it lives
in one place and the others delegate. This has been violated three times
and caught three times: urban material tables, band-wavelength helpers,
and Köhler theory — where three copies had drifted far enough to
disagree about whether a particle activates. Shared quantities go in the
shared core (`RadiationCore`, `MoistThermodynamics`) as a generic that
implementers extend.

**Minimal dependencies.** Use a well-maintained Julia package rather
than reimplementing it; binary (JLL) packages are allowed. Keep the
dependency list short: every new dependency is justified in its PR —
what it replaces, why it is worth it, and its load-time cost (`using
Aeolus`, min of 3 fresh processes, before and after). Do not pin versions
only to avoid a JLL. Today's non-stdlib dependencies (`Project.toml`):
FFTW, KernelAbstractions, StaticArrays and NCDatasets (GRIBDatasets went
with the ERA5 reader to AeolusPrep.jl, #457).
The existing in-package implementations (PNG,
Mie, error and Bessel functions) stay as they are, but may
be replaced by a package in a dedicated PR. (This replaced the old
"no binary dependencies" rule on 2026-10-06.) Details:
`docs/CONTRIBUTING.md`.

**Report honestly.** If a test fails, say whether the *code* or the
*expectation* was wrong — both happen, and the distinction matters. Never
weaken an assertion to get green without stating that the original
expectation was physically incorrect. Quote measured numbers, not
adjectives.

**Speed over bit-identity.** Results need not be bit-identical — not
across thread counts (Julia, BLAS, FFTW), machines, restarts or
refactors (maintainer decision, 2026-10-16, epic #493). Never trade speed
for bitwise reproducibility. Tests assert physics: a tolerance with its
reason (round-off growth, chaotic divergence, sampling error),
conservation, statistics, analytic limits — no hashes of simulated
fields, no `==` on simulated floats. Exact equality stays only where
data is copied, not computed (a checkpoint read back, a config parsed,
a driver file regenerated from the same input). A refactor is checked by
agreement within a stated tolerance.

**Look at what you produce.** Figures, animations and output files get
opened and inspected before they are committed. A missing-glyph bug
silently rendered "surface temperature" as "sur a e te erature" for
weeks because nobody viewed the image at size.

## Layout

```
src/            70 modules. Shared cores: RadiationCore, MoistThermodynamics, SurfaceExchange
test/           runtests.jl + one test file per topic or module (the PR gate);
                run.jl = parallel runner, test_tags.toml = weights
validation/     runvalidation.jl + simulated validation cases (not a gate);
                run.jl = the same runner
examples/       one directory per case: <case>.toml + input/*.nc + data/ + output
                analysis scripts; examples/urban/ (Berlin) is the flagship
docs/src/       theory, design, API reference
.worktrees/     agent worktrees (gitignored)
```

Roadmap: [`docs/src/design/roadmap.md`](docs/src/design/roadmap.md).
Issues are grouped into milestones named by content (clouds and
microphysics, turbulence closure, …), never into numbered periods.
"Eras" are not used: the word is reserved for the ERA5 reanalysis
(maintainer decision, 2026-11-06; `test/naming_conventions.jl`).
