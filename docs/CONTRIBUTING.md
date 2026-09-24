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
will wander into a worktree. This was checked rather than assumed.

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
julia --project=. test/runtests.jl      # must be green
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

There is no CI yet. The gate is that the agent runs the full suite and
reports the result honestly in the PR.

```bash
julia --project=. test/runtests.jl
```

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

### No binary dependencies

FFTW is the only non-stdlib dependency. NetCDF, PNG/APNG, Mie
scattering, the error function and Bessel functions are all implemented
in-package, because the usual binary stack does not build on every
target platform and a package that fails to load is worse than one that
is slightly less convenient.

Do not add a dependency to avoid writing 150 lines. If you genuinely
need one, raise it on the issue first.

### Look at what you produce

Open the figures. Read the output file back. A missing-glyph bug
rendered "surface temperature" as "sur a e te erature" in every figure
the package produced, and survived because the images were generated but
never viewed at size. `add_label!` substituted a space for unknown
characters, so it failed silently — it now renders `?`.

The same applies to data: a writer whose output cannot be read back is
not a working format, which is why `NetCDFIO` ships a reader that
re-derives offsets independently of the writer.

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

**The suite fails on `master`.** Stop and report it. Do not build on a
red baseline — you will not be able to tell your failures from the
existing ones.
