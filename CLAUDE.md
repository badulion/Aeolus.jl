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
# 3. Run the FULL suite before proposing anything.
julia --project=. test/runtests.jl

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

## Before opening a PR

- [ ] Full suite green: `julia --project=. test/runtests.jl`
- [ ] New code has tests that assert **physics**, not just that it runs
- [ ] Validation numbers measured and quoted in the PR body
- [ ] No new export collisions (see below)
- [ ] Docs updated if behaviour or API changed

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
era contract (`RadiationCore`, `MoistThermodynamics`) as a generic that
implementers extend.

**No binary dependencies.** FFTW is the only non-stdlib dependency and it
should stay that way. NetCDF, PNG, Mie scattering, error and Bessel
functions are all implemented in-package because the usual binary stack
does not build on every target. Do not add a dependency to avoid writing
150 lines.

**Report honestly.** If a test fails, say whether the *code* or the
*expectation* was wrong — both happen, and the distinction matters. Never
weaken an assertion to get green without stating that the original
expectation was physically incorrect. Quote measured numbers, not
adjectives.

**Look at what you produce.** Figures, animations and output files get
opened and inspected before they are committed. A missing-glyph bug
silently rendered "surface temperature" as "sur a e te erature" for
weeks because nobody viewed the image at size.

## Layout

```
src/            40 modules. Era contracts: RadiationCore, MoistThermodynamics
test/           runtests.jl + era*_*.jl, 7,407 tests
examples/       worked cases; examples/urban/ is the flagship
docs/src/       theory, design, API reference
.worktrees/     agent worktrees (gitignored)
```

Roadmap: [`docs/src/design/longterm_roadmap.md`](docs/src/design/longterm_roadmap.md).
Issues are grouped into milestones, one per era.
