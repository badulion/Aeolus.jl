# Contributing to Aeolus.jl

This document explains the development workflow and, more importantly,
*why* it is shaped the way it is. The short operational version — the
part an agent must follow every session — is in
[`CLAUDE.md`](../CLAUDE.md) at the repository root.

---

## Table of contents

- [The shape of the work](#the-shape-of-the-work)
- [Issues and milestones](#issues-and-milestones)
- [The worktree workflow](#the-worktree-workflow)
- [A worked example](#a-worked-example)
- [Commit granularity](#commit-granularity)
- [Pull requests](#pull-requests)
- [Testing standards](#testing-standards)
- [Continuous integration](#continuous-integration)
- [Documentation is part of done](#documentation-is-part-of-done)
- [Conventions with a history](#conventions-with-a-history)
- [Recovery procedures](#recovery-procedures)

---

## The shape of the work

Development is organised into **eras** — coherent blocks of capability,
each ending at a validation target — subdivided into **phases** of two
to six weeks. The full plan is
[`docs/src/design/longterm_roadmap.md`](src/design/longterm_roadmap.md):
eleven eras, one hundred phases.

A phase is too large to be a unit of work. Phases are therefore broken
into **issues**, each one a self-contained piece that can be implemented,
tested and reviewed on its own. How finely a phase is decomposed depends
on how clearly it can be seen:

| Horizon | Decomposition |
| :-- | :-- |
| Current and next era | 3–5 issues per phase |
| Two to four eras out | One issue per phase |
| Beyond that | One tracking issue per era |

Decomposing work eight years out is fiction. Far-horizon eras carry a
single epic issue and are broken down when they come into view.

---

## Issues and milestones

**Milestones map one-to-one onto eras.** Completed eras have closed
milestones so the project board shows the real arc of the work and
progress figures mean something.

There are no retrospective issues for work that shipped before this
workflow existed. Inventing an issue history that never happened would
make the tracker lie.

**Labels** carry the era, the phase and the subsystem, so the board can
be sliced either way:

```
era-iv  phase-40  microphysics  radiation  numerics  urban
infrastructure  tech-debt  validation  docs
```

### Claiming an issue

Comment on it before starting, so two agents do not pick up the same
work. If an issue turns out to be bigger than one branch, say so on the
issue and split it rather than opening a sprawling PR.

---

## The worktree workflow

Each issue is developed in its own **git worktree** on its own branch.

A worktree is a second working directory backed by the same repository.
Several can exist at once, each on a different branch, each with its own
files. That is the whole point: two agents working simultaneously do not
see each other's edits, cannot leave each other half-finished files, and
cannot trip over each other's `Pkg` state.

```bash
git worktree add .worktrees/issue-<N> -b issue/<N>-<slug> master
```

- **Location** — `.worktrees/`, inside the repository and gitignored.
- **Branch name** — `issue/<number>-<short-slug>`, e.g.
  `issue/42-p3-microphysics`. The number makes the mapping back to the
  issue unambiguous.
- **Base** — always `master`.

### The one hazard

`.worktrees/` lives inside the repository, so a `git add -A` from the
parent checkout would otherwise try to stage nested checkouts. It is
gitignored; **do not remove that entry**.

Everything else is safe: `test/runtests.jl` and `docs/make.jl` use
explicit file and module lists rather than recursive globs, so neither
will wander into a worktree. This was checked rather than assumed. The
parallel runner reads the same include list, and writes its logs to
`.testruns/`, which is gitignored too.

### Cleaning up

After a PR is merged, the maintainer removes the worktree:

```bash
git worktree remove .worktrees/issue-42
git branch -d issue/42-p3-microphysics
```

Agents do not delete worktrees or branches they did not create.

---

## A worked example

Implementing *Phase 40 — P3 microphysics*, issue #42.

```bash
# Isolated checkout
git worktree add .worktrees/issue-42 -b issue/42-p3-microphysics master
cd .worktrees/issue-42
```

Work proceeds bottom-up, committing at each self-contained step:

```bash
# P3 needs moments of a gamma distribution, and SpecialFunctions does not
# load here — so the special functions come first, with their own tests.
git commit -m "P3: incomplete gamma function + validation against known values"

# The mass-dimension relation is the heart of the scheme.
git commit -m "P3: mass-dimension relation across the four rime regimes"

git commit -m "P3: sedimentation with per-moment fall speeds"
git commit -m "P3: radar reflectivity diagnostic"
git commit -m "P3: wire into Aeolus.jl, add to runtests, resolve export collision"
```

Then the gate, and only then the PR:

```bash
julia --project=. test/run.jl                           # fast tier: must be green
julia --project=. test/run.jl --only era4_icenucleation.jl  # full-tier files of the physics you touched
git push -u origin issue/42-p3-microphysics
gh pr create --base master --title "Phase 40 — P3 microphysics (#42)" --body "..."
```

And stop. Review and merge belong to the maintainer.

---

## Commit granularity

> Commit after each self-contained subproblem, not once at the end.

A good commit is one a reviewer can evaluate in isolation. A helper
function together with its tests. One bug, with the diagnosis in the
message. One module wired into the package.

This is not bookkeeping. When something breaks six months later,
`git bisect` over five meaningful commits finds the cause; over one
four-thousand-line commit it finds only that the PR was bad.

Because granular history is the goal, **PRs land as merge commits**. Do
not squash — squashing would discard exactly what the cadence was for.

### Commit messages

State what changed and, where it is not obvious, why. When a commit
fixes a bug, record what the bug *was* — the repository history is the
only place that knowledge survives. Real examples from this codebase:

```
Fix step_soil! energy bug: the ground-heat flux F_skin warmed soil
layer 1 but was never subtracted from Ts, so the column created heat
from nothing AND the surface lost its dominant nocturnal sink.
```

```
The backward-Euler residual is U-shaped, not monotonic: the Kelvin term
sends f to +inf as r -> 0 and the r^2 term does the same as r -> inf.
Choosing the bracket by testing f at a tiny lower bound silently
collapsed a growing droplet onto the floor.
```

---

## Pull requests

One PR per issue, closing it with `Closes #N`.

The PR body should contain:

1. **What was implemented** — a short paragraph, not a diff summary.
2. **Validation numbers**, measured, with what they are compared against.
3. **Bugs found**, and how they were diagnosed.
4. **Anything deliberately left out**, and why.

If a test expectation turned out to be wrong rather than the code, say
so explicitly. Both happen; the distinction is the most useful sentence
in the whole PR.

### Agent boundaries

| May | Must not |
| :-- | :-- |
| Create worktrees and branches | Merge any PR |
| Commit, push its own branch | Close issues |
| Open PRs, comment | Push or force-push to `master` |
| Run tests and examples | Delete others' branches or worktrees |
| Edit its own PR | Rewrite published history |

---

## Testing standards

The gate is that the agent runs the suite and reports the result
honestly in the PR; GitHub Actions re-runs it (see
[Continuous integration](#continuous-integration)). **Fast tier for PRs,
full tier at merge:**

```bash
julia --project=. test/run.jl                          # fast tier, parallel: the PR gate
julia --project=. test/run.jl --only <file>.jl         # a full-tier file you affected
AEOLUS_TEST_TIER=full julia --project=. test/run.jl    # full tier: the merge train
julia --project=. test/runtests.jl                     # serial, same tier selection
```

- The **fast tier** (default) is every test file except the validation
  mini-simulations tagged `tier = "full"` in `test/test_tags.toml`. It
  keeps every module covered by conservation, analytic and unit checks,
  and runs in about 5 minutes on the 24-core machine. A PR needs it
  green.
- The **full tier** adds the validation runs (GABLS, channels, flat
  plate, CBC decay, urban and heterogeneous-surface campaigns). The merge
  train runs it before merging. If your change touches physics that a
  full-only file validates, run that file yourself and quote it: the
  merge train is not the place to find out.
- `test/runtests.jl`'s include list stays the single list of test files.
  A new file needs its include line and nothing else; tag it only if it
  is a full-only validation run (with a one-line reason saying which fast
  file keeps the module covered) or timing-sensitive (`exclusive`).

The runner, its options and the tag table are described in
[`docs/src/testing.md`](src/testing.md).

**Test physics, not execution.** A test asserting a function returns
without error is nearly worthless. Assert conservation laws, analytic
limits, published benchmarks, convergence rates, symmetry and
reciprocity. The suite's strongest tests are the ones checking against
something the code does not contain:

| Check | Result |
| :-- | :-- |
| MODIS Ross-thick white-sky albedo | `0.189186` vs published `0.189184` |
| Mie extinction (Wiscombe) | `2.881999` vs `2.8820` |
| View factors vs closed form | `9.2e-15` |
| Golovin analytic coalescence | `L¹ = 0.037` at 1200 s |
| Clausius-Clapeyron self-consistency | `5.2e-12` |

**Print measured values with `@info`** so they can be quoted in the PR
and tracked over time.

**When a test fails, diagnose before fixing.** Decide whether the
implementation or the expectation is wrong. Weakening an assertion to
reach green, without stating that the original expectation was
physically incorrect, is the one thing that will get a PR rejected
outright.

---

## Continuous integration

Four workflows live in `.github/workflows/`:

| Workflow | Trigger | What it does |
| :-- | :-- | :-- |
| `CI.yml` | every PR, push to `master` | fast tier: `julia --project=. test/run.jl` |
| `CI.yml` | push to `master`, Mondays 03:17 UTC, manual | full tier: `AEOLUS_TEST_TIER=full` |
| `Documenter.yml` | push to `master`, tags, PRs | builds `docs/make.jl`; deploys from `master` and tags only |
| `CompatHelper.yml` | daily | opens PRs that bump `[compat]` entries |
| `TagBot.yml` | registry release | tags and releases a registered version |

The test jobs run Julia 1.13 on `ubuntu-latest` (4 cores, 16 GB) with
`JOBS=4`, one thread per job and `AEOLUS_FFTW_PLANNER=estimate`. Each job
uploads its run folder (`.testruns/ci-*`: per-file logs, `summary.txt`)
as an artifact, so a failure can be read without rerunning it. Timeouts:
90 min for the fast tier, 240 min for the full tier, 60 min for the docs.

CI is a second check, not a replacement for running the suite yourself:
the wall-clock assertions (`era6_apriori_sgs.jl`, `era6_cbc_decay.jl`) were
calibrated on the 24-core machine, and a slow runner failing one of them
is reported, not papered over by loosening the limit.

### Documentation deployment

`docs/make.jl` ends in a `deploydocs` call guarded by
`GITHUB_ACTIONS == "true"`, so a local build never tries to push. On
GitHub it pushes the built site to the `gh-pages` branch with the
workflow's `GITHUB_TOKEN`: `master` becomes `dev/`, a tag `vX.Y.Z` becomes
`vX.Y.Z/` and `stable/`. A `DOCUMENTER_KEY` secret is optional (only
needed if the token is not allowed to push, or for TagBot to trigger the
docs build on tags).

Serving the branch is a one-time repository setting that only the
maintainer changes (agents must not):

1. Let the `Documentation` workflow run once on `master`, so the
   `gh-pages` branch exists.
2. **Settings → Pages → Build and deployment → Source: Deploy from a
   branch**, branch **`gh-pages`**, folder **`/ (root)`**, Save.
3. If the deploy step fails with a permission error: **Settings →
   Actions → General → Workflow permissions → Read and write
   permissions**.

The site is then at <https://badulion.github.io/Aeolus.jl/dev/>
(`stable/` appears with the first tagged release).

---

## Documentation is part of done

Documentation is not a follow-up task and not a separate issue. A phase
that ships with the roadmap still describing it in the future tense has
left the repository lying about itself, and the next person — quite
possibly you, months later — will trust the wrong document.

There are three levels, because three different things go stale.

### Every pull request

- **Docstrings** on new public functions. Include a literature reference
  for anything non-obvious, and state the limits of validity. The
  existing modules set the standard: see `MoistThermodynamics` on why
  its saturation adjustment is bisection rather than a fixed point.
- **A new module** needs an `@autodocs` block in `docs/src/api/` and a
  line in the `pages` list in `docs/make.jl`. Without both, its
  docstrings exist but no page reaches them, and Documenter will say so.
- **Changed behaviour or interface** means updating the affected theory
  or design page under `docs/src/`.

### After the PR is merged — closeout

Status updates are **not** part of the feature PR. They happen once it is
merged, as a separate change on `master` made by the maintainer, or as a
small dedicated closeout PR.

Two reasons. A phase cannot honestly be marked ✅ SHIPPED while the PR
delivering it is still open and might yet be rejected. And
`longterm_roadmap.md` and `README.md` are shared files — if every
phase-completing branch edited them, concurrent agents would conflict on
precisely the two files most likely to be touched simultaneously. Keeping
them off feature branches removes that class of conflict entirely.

**When a phase's last issue is merged**, update
`docs/src/design/longterm_roadmap.md`:

- Mark the phase **✅ SHIPPED**.
- Replace the planned description with **what was actually built**, and
  the numbers it was measured at. A shipped phase's entry is a record,
  not a forecast.
- **If the implementation deviated from the plan, say so and say why.**
  Phase 32 was planned as "bind RRTMGP.jl" and shipped as a Malkmus band
  model inverted through Domoto, because RRTMGP needs a binary stack that
  does not build here. That deviation is recorded with its reasoning, and
  it is more useful than the original plan was.
- **Record what the tests caught.** Bugs found, and — separately — test
  *expectations* that proved wrong rather than the code. Both happen. The
  distinction is the single most useful thing in the file, because it
  tells the next reader which of their own assumptions to distrust.
- Tick the phase in the README roadmap checkboxes.

A worked example of the tone, from the Phase 32 entry:

> **Accuracy bug found later by the DYCOMS-II RF01 case, and fixed.**
> The version that shipped first had *two compensating errors* in the
> 8–12 μm window: line strengths roughly 50× too large and a continuum
> roughly 15× too weak. Those cancel for one particular column, which is
> precisely why a single broadband number is not an adequate test of a
> band model.

**When an era's last phase is merged**, additionally:

- **Close the era milestone.**
- **Update the README** — era table, capability summary, validation table
  and test count. The README carried "39 unit tests" long after the suite
  passed seven thousand; that is how a README rots.
- **Add or extend the theory page** for the era's physics. Era III added
  `docs/src/theory/radiation.md`, explaining the transfer equation, why
  band-averaging is biased, and where plane-parallel breaks down — the
  reasoning behind the code, not a restatement of it.
- **Rebuild the docs and read the warnings:**

  ```bash
  julia --project=docs docs/make.jl
  ```

  Missing-docstring warnings mean public API that no page documents. That
  check once went from hundreds of undocumented names to two, both
  private helpers that correctly stay out of the manual.

### What not to do

Do not open a separate "update the docs" issue at the end of an era.
By then nobody remembers which numbers were measured under which
configuration, and the temptation is to quote figures from a run that no
longer exists. Quote what you measured, when you measured it.

---

## Conventions with a history

These are not style preferences. Each was learned by something breaking.

### One source of truth

If two modules need the same physics, it lives in one place and the
others delegate. Shared quantities belong in the era contract —
`RadiationCore` for radiation, `MoistThermodynamics` for
thermodynamics — declared as a generic with no method, which
implementers extend by dispatching on their own types.

This has been violated and caught three times:

- **Urban material tables** existed in three places; the albedo of
  concrete was defined independently in each.
- **Band-wavelength helpers** had four implementations.
- **Köhler theory** had three, and they had *drifted*: one used a
  surface tension of 0.0728 N/m at 298 K where the measured value is
  0.0720. That is 0.8 % in the Kelvin coefficient, but critical
  supersaturation goes as `A^{3/2}`, so 1.2 % — enough for two modules
  to disagree about whether a given particle activates.

Before adding a function, check whether it already exists under another
name. A collision scan across all modules takes a minute and has caught
a real conflict every time it has been run.

### Physics is tested on its own

Physics is tested in its own module's tests, with small synthetic setups
(analytic limits, conservation, published benchmarks). The Berlin run is
an **example**, not a test harness: coupling new physics into it is
welcome where it makes sense, but not required, and the example has no
test suite of its own.

*History.* From the end of Era V until 2026-10-06 the rule was the
opposite: every physics PR had to wire its module into the Berlin run,
enforced by `test/berlin_coverage.jl` (frozen coverage lists, deleted
on 2026-10-06) and `test/berlin_smoke.jl`. It found real cross-module defects, but the
Berlin tests grew to about 15 minutes of suite time spent mostly
rebuilding the Berlin geometry, and they tested `src/` physics
indirectly. The maintainer dropped the rule, and #320 moved every check
of `src/` physics from `test/berlin_*.jl` into the owning module's tests
(synthetic setups, same quantitative targets) and deleted those files.
The end-to-end runs of the example (every coupling on, ledgers closed;
the nest parent's smoke run and its idealised validations) are now
`examples/urban/berlin_checks.jl`, run by hand after changing the
example.

### Minimal dependencies

The maintainer replaced the former "no binary dependencies" rule on
2026-10-06: "Dependencies are a good thing, no need to reinvent the
wheel. However, keep them to a minimum."

- **Use a package rather than reimplementing it.** If a well-maintained
  Julia package does the job, depend on it. Binary (JLL) packages are
  allowed.
- **Keep the list short.** Every new dependency is justified in the PR
  that adds it: what it replaces, why it is worth it, and its load-time
  cost — `using Aeolus` timed in fresh processes, minimum of three, before
  and after. A package that pulls a large tree for one function is
  usually not worth it.
- **Do not pin versions only to avoid a JLL.** Take the version the
  resolver picks unless there is a real incompatibility.
- **Existing in-package implementations stay.** PNG/APNG,
  Mie scattering, the error function and Bessel functions were written
  under the old rule. They are tested and validated and are not to be
  churned in passing; replacing one with a package is fine, in a
  dedicated PR that shows the validation still holds.

Current direct dependencies: FFTW, NCDatasets and KernelAbstractions
(admitted
2026-10-06, #291: "single-node multithreading now, with one kernel path
for the GPU in Phase 81"). All KernelAbstractions code lives in
`src/Backends.jl` (`docs/src/design/ka_port.md`). Its tree resolves
`UnsafeAtomics` 0.4, which brings `LLVM.jl` and the binary
`LLVMExtra_jll`; the earlier `UnsafeAtomics = "0.3"` pin that kept them
out was lifted on 2026-10-06 (maintainer: "Allow LLVMExtra_jll"). It
costs about 37 ms of `using Aeolus` (215 → 252 ms).

NCDatasets (admitted 2026-10-06, #310) replaced the in-package NetCDF-3
reader and writer (`NetCDFIO`, 566 lines). NetCDF-4/HDF5 is the format of
ERA5, ICON, COSMO and PALM and of the tools that read them; it gives the
output lossless compression, chunking and no 4 GiB limit per record, none
of which a hand-written writer could reasonably provide. Its tree brings
the binary `NetCDF_jll` and `HDF5_jll`. It costs about 270 ms of
`using Aeolus` (449 → 717 ms, minimum of three fresh processes,
2026-10-06). On the 24 h Berlin output of 2026-10-04 (48 records,
64×64×32) it reduces the file from 316 MB (NetCDF-3) to 185 MB, bit for
bit identical on read-back. All of Aeolus' output policy (NetCDF-4, deflate level 4
with shuffle, chunks of one level and one record) is in `src/NCIO.jl`;
reading is plain NCDatasets. GRIBDatasets, which #310 also admitted, is
added with the first code that reads GRIB (the ERA5 preprocessors, #311).

### Look at what you produce

Open the figures. Read the output file back. A missing-glyph bug
rendered "surface temperature" as "sur a e te erature" in every figure
the package produced, and survived because the images were generated but
never viewed at size. `add_label!` substituted a space for unknown
characters, so it failed silently — it now renders `?`.

The same applies to data: a writer whose output cannot be read back is
not a working format, which is why `test/ncio.jl` reads every output
variable back and checks it bit for bit.

---

## Recovery procedures

**A worktree is in a broken state.** Worktrees are disposable. Commit or
stash anything worth keeping, then:

```bash
git worktree remove --force .worktrees/issue-42
git worktree add .worktrees/issue-42 issue/42-p3-microphysics
```

**`master` moved while you were working.** Rebase onto it in your
worktree; never merge `master` into a feature branch.

```bash
git fetch origin && git rebase origin/master
```

**A worktree was deleted by hand** and git still lists it:

```bash
git worktree prune
```

**An issue is bigger than expected.** Say so on the issue, split it, and
open a PR for the part that is finished. A large PR that touches six
subsystems cannot be reviewed meaningfully.

**A full-tier file fails at merge but the fast tier was green.** The
log path is printed in the runner's summary. Rerun that file alone
(`test/run.jl --only <file>`) to rule out load: files tagged
`exclusive` assert wall-clock budgets and must not share the machine.

**The suite fails on `master`.** Stop and report it. Do not build on a
red baseline — you will not be able to tell your failures from the
existing ones.
