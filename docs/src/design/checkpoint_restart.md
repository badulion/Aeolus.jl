# Checkpoint / restart and FFT planning

A restart is only useful if it is *exact*: a run checkpointed at step N
and continued must be the run that never stopped, bit for bit. Anything
less makes a restarted validation run a different experiment, and makes
"did my change alter the answer?" unanswerable for long jobs. This page
records how Aeolus gets there (#245) and the decisions behind it. The API
is on [I/O, checkpoints and restart](../api/io.md).

## What a step reads

A checkpoint must hold everything the next step reads, not only the
prognostic fields. For the Boussinesq cores that is:

| state | where it lives | how it is checkpointed |
| :---- | :------------- | :--------------------- |
| `u, v, w, p, θ, qv` **with halos**, the clock | the simulation | `checkpoint_fields(sim)`, full storage arrays |
| last eddy coefficients `νt, κt` | the simulation | stored too: a driver's dt choice (`sbl_dt`, `channel_dt`) reads them before the closure recomputes them |
| SGS closure memory: TKE ``e``, Lagrangian / LASD ``I_{LM}, I_{MM}, I_{QN}, I_{NN}`` and the pathline velocities, the six Reynolds stresses, booked statistics, commit cadence | the closure | the closure contract's `sgs_state` / `restore_sgs_state!` (#167), whose snapshots are now required to be self-contained |
| wall-model memory: the input filter state (P57-b), the TBLE columns and dynamic κ (P57-c), the per-column laws, the dynamic roughness α and z0 | `MatchingInput`, `TBLEWallModel`, `WallModel`, `DynamicRoughness` | `checkpoint_state` methods in `src/wall_models/checkpoint.jl` |
| case counters and ledgers (steps, wall stresses, heat applied, budgets) | `WallBoundedCase`, the SBL driver's `SBLCase` | `checkpoint_state` methods |
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
two shared run loops, `run_channel!` and `sbl_run!`, take it as a keyword
and also write a final checkpoint, so re-running a finished phase is
instant and a multi-phase script (GABLS3) uses one prefix per phase. The
long drivers (`dns256_generate.jl`, the relaminarisation sweep,
`gabls1`–`gabls4.jl`) create one per run by default and delete it once the
result is cached.

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
  table.
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

## The Berlin case

The Berlin case's own state — soil columns, building zones, tracers,
the surface bus, radiation caches, the stage clock — is not checkpointed
yet; that is #269 / #134, and `IO` stays in `BERLIN_PENDING` until then.
The hook is the generic above: one `checkpoint_state` /
`restore_checkpoint_state!` method per coupling state, collected as the
`components` of `checkpoint!` (the dynamical core, closure and wall
models of the Berlin run are covered already).
