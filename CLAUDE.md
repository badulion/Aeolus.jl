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
# 3. Run the FAST tier before proposing anything (parallel, ~5 min),
#    plus the full-tier files of any physics you touched.
julia --project=. test/run.jl
julia --project=. test/run.jl --only era6_gabls1.jl,era6_gabls3.jl

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

## Tests: fast tier for PRs, full tier at merge

The suite has two tiers (`AEOLUS_TEST_TIER=fast|full`, default `fast`).
The full tier adds the validation mini-simulations that
`test/test_tags.toml` marks `tier = "full"`, each with its reason.

- **A PR needs the fast tier green**: `julia --project=. test/run.jl`
  (parallel) or `julia --project=. test/runtests.jl` (serial).
- **When you touch physics that a full-only file validates, run that
  file too** (`test/run.jl --only <file>`) and quote its result.
- **The merge train runs the full tier before merging**:
  `AEOLUS_TEST_TIER=full julia --project=. test/run.jl`.
- `test/runtests.jl`'s include list stays the single list of test
  files; a new file needs only its include line. Tag it in
  `test/test_tags.toml` only if it is a full-only validation run
  (with a one-line reason) or timing-sensitive (`exclusive`).

Details: [`docs/src/testing.md`](docs/src/testing.md).

## Before opening a PR

- [ ] Fast tier green: `julia --project=. test/run.jl`, plus the
      full-tier files of the physics you touched (`--only`)
- [ ] New code has tests that assert **physics**, not just that it runs
- [ ] Validation numbers measured and quoted in the PR body
- [ ] No new export collisions (see below)
- [ ] Documentation updated — see below
- [ ] New physics is tested on its own (module tests, synthetic setups) — see below

## Physics is tested on its own

Every piece of physics is tested **in its own module's tests**, with small
synthetic setups (analytic limits, conservation, published benchmarks) —
not through the Berlin case. The Berlin (Alexanderplatz) run,
`examples/urban/alexanderplatz_diurnal.jl`, is an **example**: couple new
physics into it when that makes sense, but nothing requires it, and it has
no test suite of its own (maintainer decision, 2026-10-06; the
`test/berlin_*.jl` files and the coverage lists are gone, #320).

- Do not add Berlin tests; put new physics tests in module tests.
- After changing the example, run its end-to-end checks by hand:
  `julia --project=. examples/urban/berlin_checks.jl` (every coupling
  on, ledgers closed; the nest parent's smoke run and validations).

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

- When a phase's last issue is merged: mark the phase ✅ SHIPPED in
  `docs/src/design/longterm_roadmap.md` with **what was actually built
  and the numbers it was measured at**, record any bug the tests caught
  and any test *expectation* that proved wrong, and tick the phase in the
  README.
- When an era's last phase is merged: close the era milestone, update the
  README era table, capability summary, validation table and test count,
  add or extend the era's theory page, and rebuild the docs
  (`julia --project=docs docs/make.jl`) checking for missing-docstring
  warnings.

Two reasons status updates are kept out of the feature PR. A phase cannot
honestly be marked SHIPPED while the PR delivering it is still open. And
`longterm_roadmap.md` and `README.md` are shared files — if every
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
era contract (`RadiationCore`, `MoistThermodynamics`) as a generic that
implementers extend.

**Minimal dependencies.** Use a well-maintained Julia package rather
than reimplementing it; binary (JLL) packages are allowed. Keep the
dependency list short: every new dependency is justified in its PR —
what it replaces, why it is worth it, and its load-time cost (`using
Aeolus`, min of 3 fresh processes, before and after). Do not pin versions
only to avoid a JLL. The existing in-package implementations (PNG,
Mie, error and Bessel functions) stay as they are, but may
be replaced by a package in a dedicated PR. Details:
`docs/CONTRIBUTING.md`.

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
src/            55 modules. Era contracts: RadiationCore, MoistThermodynamics, SurfaceExchange
test/           runtests.jl + era*_*.jl and other module tests;
                run.jl = parallel runner, test_tags.toml = tiers
examples/       worked cases; examples/urban/ is the flagship
docs/src/       theory, design, API reference
.worktrees/     agent worktrees (gitignored)
```

Roadmap: [`docs/src/design/longterm_roadmap.md`](docs/src/design/longterm_roadmap.md).
Issues are grouped into milestones, one per era.
