# Checkpoint / restart and FFT planning

A restart is only useful if it is *exact*: a run checkpointed at step N
and continued must be the run that never stopped, bit for bit. Anything
less makes a restarted validation run a different experiment, and makes
"did my change alter the answer?" unanswerable for long jobs. This page
records how Aeolus gets there (#245) and the decisions behind it. The API
is on [I/O, checkpoints and restart](../api/io.md).

## What a step reads

A checkpoint must hold everything the next step reads, not only the
prognostic fields. For the Boussinesq cores that is (the sound-proof
cores, since ONE-j #460, the same without `qv`, plus their base state):

| state | where it lives | how it is checkpointed |
| :---- | :------------- | :--------------------- |
| `u, v, w, p, θ, qv` **with halos**, the clock | the simulation | `checkpoint_fields(sim)`, full storage arrays |
| the base state `θ̄`, `ρ̄` (sound-proof cores) | `sim.ref` | `checkpoint_reference(sim)`: recorded and compared on restart like the run parameters, not restored (a restart onto another base state is refused) |
| last eddy coefficients `νt, κt` | the simulation | stored too: a driver's dt choice (`channel_dt`, the model's adaptive step) reads them before the closure recomputes them |
| SGS closure memory: TKE ``e``, Lagrangian / LASD ``I_{LM}, I_{MM}, I_{QN}, I_{NN}`` and the pathline velocities, the six Reynolds stresses, booked statistics, commit cadence | the closure | the closure contract's `sgs_state` / `restore_sgs_state!` (#167), whose snapshots are now required to be self-contained |
| wall-model memory: the input filter state (P57-b), the TBLE columns and dynamic κ (P57-c), the per-column laws, the dynamic roughness α and z0 | `MatchingInput`, `TBLEWallModel`, `WallModel`, `DynamicRoughness` | `checkpoint_state` methods in `src/wall_models/checkpoint.jl` |
| case counters and ledgers (steps, wall stresses, heat applied, budgets) | `WallBoundedCase`, the model's components (the surface layer's applied stress, the towers' segments; ONE-g #456) | `checkpoint_state` methods |
| the adaptive controller's input `dt_prev`, step count, dt history, retries | `StepperState` | its fields |
| random numbers drawn after the start | an `AbstractRNG` | `copy(rng)` |
| a driver's running averages and series | its run loop | `checkpoint_components` (any `Vector`, `Dict`, `Ref`, struct with a method) |

Tendencies and scratch buffers are not stored: every core writes them
before reading them. The bitwise tests prove this rather than assume it —
the restarted simulation's scratch buffers are filled with `NaN` before
the restart, so a single stale read would make the result non-finite.

## One generic, many owners

Each owner of state implements the pair

```julia
checkpoint_state(x)              -> self-contained serialisable copy
restore_checkpoint_state!(x, s)  -> x
```

next to its own code (the closure contract for closures; one file in
WallModels; the drivers for their structs). `checkpoint!` collects them by
name; `restart!` checks the names and types and restores them. This keeps
one source of truth for each piece of memory: the rollback of a rejected
adaptive step and the checkpoint use the same `sgs_state` snapshot. Two
closures failed the self-contained requirement before #245 — the TKE and
Reynolds-stress `restore_sgs_state!` read their booked statistics back from
a buffer inside the closure, correct for a rollback in one object, wrong
for a restore into a freshly built closure — and were fixed.

## Format

A checkpoint is a `Serialization` dictionary with `:format =>
"Aeolus.checkpoint"` and `:version => 1`, the Julia version, the mesh, the
run's scalar parameters, the fields,
closure, components, stepper, RNG, free-form `:extra`, and the FFTW
provenance. It is written to `path.tmp` and renamed, so a job killed while
writing never leaves a truncated checkpoint. Snapshots without a version —
`write_snapshot` output and pre-#245 `checkpoint!` files, which hold only
the interior of `u, v, w, p, θ, qv` and `t` — are **refused** by
`restart!` with an explanation (maintainer decision on #245): restarting
from them would silently start the closure memory cold.

**Run parameters** (maintainer decision on #318): `restart!` compares the
simulation's scalar parameters (`ν`, `θ₀`, `g_accel`, `f_coriolis`,
`C_s`, `Pr_t`, surface fluxes, sponge, enable flags) with the
checkpoint's and throws an `ArgumentError` listing every difference
(`f_coriolis = 0.0002 (checkpoint: 0.0001), …`): a restart with other
parameters is a different experiment, not a continuation. A deliberate
change is made with `restart!(…; allow_mismatch = true)` (also a keyword
of `resume!`), which restores the run and records the differences in the
log with `@warn`. The mesh, the domain, the simulation type, the closure
type and the component names are always checked.

## Periodic checkpoints and resume

`CheckpointSchedule(dir; every_steps, every_seconds, keep)` writes
`dir/checkpoint_<step>.jls` every M steps or S seconds of wall clock,
keeps the newest K, and `resume!` restores the newest one at start. The
shared run loop `run_channel!` takes it as a keyword and also writes a
final checkpoint, so re-running a finished phase is instant; the model's
run takes the one of `[checkpoint]`. (The `sbl_run!` loop, the
relaminarisation sweep and `dns256_generate.jl` that used it went with
ONE-g, #456; the measurements below were made with them.)

Measured by killing the scripts mid-run and starting them again: GABLS1
at 8³ to 12 h, killed after step 5000 of 7758 — profiles, u* and the u*(t)
series identical to an uninterrupted run; `dns256_generate.jl` at 32³,
resumed at step 60 of 70 — the TOML record identical except its timestamp
and wall time. At 256³ one checkpoint is ≈ 1.15 GB (eight fields with
halos).

## FFT planning: why the planner matters

FFTW's `MEASURE`, `PATIENT` and `EXHAUSTIVE` planners time candidate
algorithms and keep the fastest, so two processes can choose different
plans, which round differently in the last bit (FFTW FAQ 3.8). Before
#245 every pressure-solver plan used `MEASURE`; same-process bitwise tests
passed only because FFTW reuses its in-process plans. Measured on a
32×32×16 wall-bounded grid with the TKE closure: after 4 steps the
`PATIENT` run differs from the `ESTIMATE` run in 24611 of 31768 values of
`u` (max 9.1e-15). (On 16×12×10 the two planners happened to pick the
same plans.)

The policy (maintainer decisions on #245), one helper,
`Operators.fftw_planner_flags()`, used by every `plan_*` call:

* **Configurable**: planner `ESTIMATE / MEASURE / PATIENT / EXHAUSTIVE`
  and the FFTW thread count, through `set_fftw_policy!`, the environment
  (`AEOLUS_FFTW_PLANNER`, `AEOLUS_FFTW_THREADS`) or a namelist `[fftw]`
  table. Without an explicit count, `init_threads!` sets FFTW to
  `Threads.nthreads()` (KA-h, #298); under `ESTIMATE` the solves are
  bit-identical at every thread count, and wisdom covers the thread
  count it was made at.
* **Default `MEASURE`, `PATIENT` opt-in** (maintainer decision on #318):
  `PATIENT` finds somewhat faster plans, but its planning cost grows
  steeply — measured at 256³: 494 s of planning with `PATIENT` against
  2.4 s with `MEASURE`, for transforms of 153 vs 182 ms (`PATIENT`
  planning was also measured at 0.22 s for the 32×32×16 rfft solver and
  4.7 s at 96×96×48, against < 0.01 s with `ESTIMATE`). `PATIENT` pays
  off only for long runs; select
  it with `set_fftw_policy!(; planner = :patient)`,
  `AEOLUS_FFTW_PLANNER=patient` or the `[fftw]` table.
* **Saved plans** (the MicroHH practice): the FFTW wisdom is written to
  the run folder at run start and loaded by a restarted process before
  any plan is made, so the restart makes the same plans. The cross-process
  test checks it (with `PATIENT`, the strongest timing planner; `MEASURE`
  wisdom works the same way): a `PATIENT` run checkpointed at step 2 in one process
  and resumed in a new one with the saved wisdom equals the continuous run
  bit for bit at step 4.
* **Tests force `ESTIMATE`** (deterministic, and planning is free):
  `runtests.jl` sets `AEOLUS_FFTW_PLANNER=estimate`, which subprocesses
  inherit; two separate `ESTIMATE` processes agree bit for bit.
* **Provenance**: `fftw_policy()` — planner, threads, wisdom loaded,
  bitwise reproducible (`ESTIMATE`, or wisdom loaded) — is stored in every
  checkpoint and logged when a schedule starts or resumes.

The one-shot FFTs of the diagnostics and filters (`rfft(x)`) are planned
by FFTW.jl with `ESTIMATE` and are deterministic already.

Not covered: bitwise identity across machines or FFTW builds (wisdom is
machine-specific; `ESTIMATE` is not guaranteed to pick the same codelets
on a different CPU), and across thread counts (a threaded plan is a
different problem).

## Composite runs: the structural state (#269)

A nested run or the Berlin case holds state in dozens of objects: two
simulations with their open-boundary buffers and closures, the parent
field set prolongated at both ends of the current parent step, the window
and the step-size memory, flux registers, an inflow generator with its
RNG, soil and building columns, ledgers, caches. A `checkpoint_state`
method per object would have to be kept complete by hand, and a missing
one cold-starts that piece silently. Such runs are instead checkpointed
with **`structural_state(x)`**: a walk of the whole object graph of `x`
that records every array (copied), every field of every struct and every
`Dict` / `IdDict` entry, with the aliasing between them.
`restore_structural_state!(x, s)` walks a freshly built graph of the same
construction alongside the record:

* arrays and mutable structs are refilled **in place**, so everything that
  shares them (fields, hooks, views, bound open-boundary planes) still
  does;
* a place whose fresh value differs in type or identity takes a rebuilt
  object (a buffer allocated lazily during the run, an inflow plane added
  later) or the object the record maps there (members swapped during the
  run: the nest's two parent time levels and its spare state rotate);
* `IdDict` entries are restored last, so their keys are the objects at
  their own places (the open boundaries bind field arrays by identity);
* **opaque** objects are not state and keep the fresh build's: pointers
  and anything holding one (FFTW plans), tasks, locks, conditions, `IO`
  and NetCDF handles, and what `checkpoint_opaque` declares (the Berlin
  case's output file);
* a `const` field that differs, or an opaque object that would have to be
  rebuilt, is an `ArgumentError` naming its path.

Four hooks (`structural_save_begin` / `_end`, `structural_restore_begin`
/ `_end`) adapt the walk for a type whose live form is not the form to
record. The nested run uses them for a parent step running ahead on
another task (SC-c, #403): it is waited for, recorded as plain values and
detached, and rebuilt on restore with an already finished task, so the
restored run takes it as the sequential one does.

The record holds every reachable array, static ones too (meshes, masks,
metric terms, scratch), so it is larger than the prognostic fields. It is
used where completeness matters more than size: `nest_state`
(`Aeolus.Nesting`) and the Berlin case.

A bug the nested test caught: arrays whose element type is stored inline
were first copied whole. An immutable struct with array fields (an
`OpenPlane`) is stored inline too, so the copy held references to the
run's own plane arrays, and the restore put those into the fresh build,
breaking the identity between the open boundary's plane dictionary and its
state. The child's halos then read stale planes and the nested child
diverged in its first step after the restart. Only `isbits` element types
are copied now; the rest are walked.

## Nested runs

`nest_state(run; window)` / `restore_nest_state!(run, s)` (#269) are the
structural state of a `NestedRun` with a header (format, version, feedback
mode, sync, mesh sizes), checked on restore: a snapshot without a nest
state, of another feedback mode or other meshes is refused with an
`ArgumentError`. `window` is the open parent-step window of a caller that
drives the child itself, so a run can be checkpointed between two child
steps of a parent step. As a `checkpoint!` component
(`components = (nest = run,)`) the run is checkpointed at a window
boundary.

`test/nest_restart.jl` drives a parent (16 × 16 × 8, TKE closure) and a
child (refinement 3, Lagrangian-dynamic closure, inflow generator with
parent-resolved statistics) the way the Berlin case does, adaptive in
both domains, 20 child steps (4 parent windows) against 10 + 10 with the
checkpoint written to disk mid-window and restored into a freshly built
run: every field of both domains with halos, both closures' memory, both
parent time levels, the child's open-boundary buffers, the generator's
planes and the clocks are **bit-identical**, one-way, one-way with the
parent step running ahead, and two-way, at 1 and at 12 threads. Restoring
only the two simulations (IO's `checkpoint!` / `restart!` with their
closures) is shown to differ. The record is 9.0 MiB for the 16 × 16 × 8
parent and the 18 × 18 × 12 child.

## The Berlin case

An urban site (the `urban` run type until ONE-e #454, formerly `berlin`, #438) checkpoints the whole case (#269):
`checkpoint_state(case)` is its structural state, a component of the
checkpoint `[checkpoint]` schedules in the run folder. It covers the
child core and its open-boundary buffers, every coupling's state (soil
and building columns, the surface bus, tracers, ledgers, radiation and
chemistry), the clock and, with the online nest, the parent case and the
nested run. `aeolus run <folder> --resume` builds the case from the
recorded config, restores the newest checkpoint into it and runs to the
original end time; the NetCDF output is reopened and continues at the
checkpoint's record (a record the interrupted session wrote after the
checkpoint is rewritten). Refused, with the reason: a checkpoint without
the case state (a nested case would cold-start its parent and the
couplings), checkpoints with a sound-proof core (the urban site's own config
check; `checkpoint!` takes the sound-proof cores since ONE-j #460, and the
urban couplings are to become components on both cores in the rest of ONE-e #454), and resuming a run that
records the child's boundary driver (`[urban] child_boundaries`).
Run folders of the old `berlin` example driver (before #438) cannot be
resumed: their `run.toml` names a driver that no longer exists, and their
checkpoints hold the case's types under the example's module
(`Main.AeolusDriver_berlin`), which Serialization cannot map to
`Aeolus.UrbanCase`; start such a run again from its config (today a
model config of an urban site, ONE-e #454). Run folders of the `urban`
run type (`[run] legacy = "urban"`, before ONE-e) cannot be resumed
either: the type is gone (a clean break); start them again from the
converted config.
`test/config_roundtrip.jl` resumes `berlin_smoke.toml` (nested, every
coupling on) interrupted after its middle checkpoint and compares it
with the uninterrupted run bit for bit, its NetCDF output included.

Measured on the flagship (`berlin_summer.toml`: the 64 × 64 × 64 child in
the 6 km × 6 km × 3 km parent, every coupling on), after 20 child steps:

| checkpoint | size | write |
| :-- | --: | --: |
| the child core alone (`checkpoint!(case.sim, path)`) | 20.9 MiB | 0.02 s |
| the whole case (`components = (case = case,)`) | 2754.5 MiB | 9.5 s |

The structural walk itself takes 4.0 s (2.88 million objects, 2628 MiB of
arrays); the rest is serialisation. The record holds every array the case
reaches, static ones included, and the nest parent's arrays dominate
(the parent alone takes ≈ 1.5 GB in memory). Writing it raises the peak
memory from 4.7 to 8.2 GB (the record is built in memory before it is
written). A fresh process built the case (39 s), restored the checkpoint
(30.6 s, compilation included) and continued 30 child steps (to t = 6 s)
bitwise equal to the uninterrupted run in all 22 arrays of `case_state`
(child, tracers, Ts, parent). At the default cadence of one checkpoint per
30 min of wall clock that is 0.5 % of the run's wall time. Leaving the
static arrays out of the record (masks, metric terms, view factors,
scratch) would make it smaller; it is not done here because which arrays
are static is a property of each coupling, which the walk is meant not to
need.
The hook is the generic above: one `checkpoint_state` /
`restore_checkpoint_state!` method per coupling state, collected as the
`components` of `checkpoint!` (the dynamical core, closure and wall
models of the Berlin run are covered already).
