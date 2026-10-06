# Logging

`Aeolus.RunLog` is the logging core of the library (LOG-a, #333; the
full design is epic #293). The rule is simple: **the library emits
structured events and never prints; the caller decides what is shown and
where by installing a logger.** Events go through the standard library's
`Logging` (`@debug`, `@info`, `@warn`, `@error`), so any `AbstractLogger`
works, and there is no dependency beyond the standard library.

## Running with a run log

```julia
using Aeolus

with_run_log("runs/my_case/run.log") do       # info → console, debug → run.log
    for n in 1:nsteps
        step_adaptive!(sim, cfl)
        maybe_checkpoint!(ck, sim; step = n)
    end
end
```

The console shows info and up with `HH:MM:SS.sss` timestamps; `run.log`
gets debug and up with full timestamps and the source location of every
record. `AEOLUS_LOG=debug` (or `info`, `warn`, `error`, `off`) overrides
the console level. The file is appended to, so a restart continues the
same log; an error that ends the run is logged at `Error` level, with its
backtrace, before it propagates. Without `with_run_log`, open the logger
yourself and close it at the end:

```julia
lg = open_run_log("run.log"; console = stderr, console_level = Logging.Info,
                  file_level = Logging.Debug)
with_logger(lg) do … end
close(lg)
```

### Flush policy

`run.log` is not flushed after every record (maintainer decision of
2026-10-06, PR #346):

| record | on disk |
| :-- | :-- |
| `Warn`, `Error` | at once (and everything buffered before it with it) |
| `Debug`, `Info` | at the next record more than `flush_interval` seconds (default 5) after the last flush |
| anything still buffered | at `close`: the end of [`with_run_log`](@ref Aeolus.RunLog.with_run_log) or [`with_run`](@ref Aeolus.IO.with_run), also when the run throws; and by an `atexit` hook if the process dies of an uncaught error with the log still open |

So only a hard kill (`SIGKILL`, the OOM killer, power loss) can lose the
last few seconds of debug and info lines; a warning or error never waits
in a buffer. There is no timer task: the clock is read when a record
arrives, which keeps logging deterministic and thread-free, so a debug
line written just before a long quiet spell stays buffered until the
next record or the close. To follow a run with `tail -f run.log` at
record granularity, or when debugging a hang, use `flush_interval = 0`
(flush every record, the LOG-a behaviour):

```julia
with_run_log("run.log"; flush_interval = 0) do … end
lg = open_run_log("run.log"; flush_interval = 1.0)   # or any interval
flush(lg)                                            # force one now
```

The keyword is [`AeolusLogger`](@ref Aeolus.RunLog.AeolusLogger)'s
(`flush_interval = 5.0`); [`open_run_log`](@ref Aeolus.RunLog.open_run_log)
passes it to the file. The console logger and the test logger flush
every record.

A run log line:

```
2026-10-06T17:26:07.029 WARN  pressure[child] MultigridPoissonSolver: maxcycles=50 reached, rel. residual 2.1e-5  cycles=50 relative_residual=2.1e-5 tol=1.0e-8  @ Multigrid.jl:687
```

## Event conventions

Every event of the library carries two tags:

| key | meaning |
| :-- | :------ |
| `subsystem` | the emitting part of the code, a `Symbol` |
| `domain` | the domain of a multi-domain run, [`current_domain`](@ref Aeolus.RunLog.current_domain)`()`: `:main` unless a driver wraps the step in [`with_domain`](@ref Aeolus.RunLog.with_domain) |

and its numbers as further key-values, so a test or a diagnostics sink
reads them without parsing text. A new event follows the same pattern:

```julia
@warn("MySolver did not converge", subsystem = :pressure, domain = current_domain(),
      iterations = it, relative_residual = rel, tol = s.tol)
```

Key-values are evaluated only when some logger wants the record, so a
disabled `@debug` costs a level check (under 1 µs, asserted in
`test/logging_core.jl`).

The events the core emits today:

| subsystem | level | event | key-values |
| :-- | :-- | :-- | :-- |
| `:pressure` | Warn | `CGPoissonSolver` hit `maxit` | `solver = :cg`, `iterations`, `residual`, `relative_residual`, `tol` |
| `:pressure` | Warn | `MultigridPoissonSolver` hit `maxcycles` | `solver = :multigrid`, `cycles`, `relative_residual`, `tol` |
| `:pressure` | Warn | `DensityWeightedPoissonSolver` breakdown / no convergence | `solver = :density_weighted_cg`, `reason = :breakdown \| :maxit`, `iterations`, `curvature` or `residual`, `relative_residual`, `tol` |
| `:diffusion` | Warn | `solve_implicit_diffusion!` breakdown / `maxit` (when `warn = true`) | `reason`, `iterations`, `curvature` or `relative_residual`, `tol` |
| `:timestepping` | Debug | `step_adaptive!`: the stepper threw (recovered by a retry) | `dt`, `retry`, `exception` |
| `:timestepping` | Debug | `step_adaptive!`: retry at a smaller step | `reason`, `from`, `to`, `retry`, `max_retries` |
| `:timestepping` | Warn | `step_adaptive!`: a fast process left non-finite values | `dt`, `processes` |
| `:checkpoint` | Warn | `restart!` across Julia versions / with `allow_mismatch` | `path`, `written_by`, `running` / `parameters` |
| `:checkpoint` | Debug | `resume!`: fresh start / resumed | `event = :fresh_start \| :resume`, `dir` / `path`, `t`, `step` |
| `:checkpoint` | Debug | `maybe_checkpoint!` wrote a checkpoint | `event = :write`, `path`, `step`, `t`, `removed` |

So a healthy run emits nothing at `Info` or above: the default console
stays quiet, and `run.log` records the retries and checkpoints. Under a
`NullLogger` nothing reaches stdout or stderr at all.

## One switch: the logger level

There are no `verbose` keywords (removed outright by #347, maintainer
decision of 2026-10-06, without a deprecation): what is shown is decided
only by the logger the caller installs and its level. The library, the
examples and the test fixtures follow one convention:

| level | for | examples |
| :-- | :-- | :-- |
| `Debug` | chatter: what a run log should keep but a console need not show | adaptive-step retries, checkpoint fresh start / resume / write, a costly diagnostics row |
| `Info` | milestones | the run header, the progress line, the Berlin setup summary, output frames and end-of-run report |
| `Warn` | problems the run survives | a solver that did not converge, a non-finite fast process, a restart across Julia versions |
| `Error` | the run stops | a blowup, an error that ends `with_run` |

To see the retries on the console, raise the console level instead of
passing a keyword:

```julia
with_run_log("run.log"; console_level = Logging.Debug) do … end   # or AEOLUS_LOG=debug
with_logger(NullLogger()) do … end                                # silence everything
```

## Run folder and provenance

A production run writes one folder (#334), `runs/<yyyy-mm-dd>_<case>_<runID>/`:

| file | contents |
| :-- | :------ |
| `run.log` | the run log: debug and up, full timestamps, source locations; appended to by every session |
| `run.toml` | provenance: `run_id`, `case`, `created`, the full `config` and `seeds`, one `[[session]]` per start or restart |
| `diagnostics.csv` | the diagnostics time series ([`diagnostics_path`](@ref Aeolus.IO.diagnostics_path); written by the [run monitor](#Run-monitor), `diagnostics.nc` after #310) |
| `crash/` | crash checkpoints of a blown-up run ([`crash_dir`](@ref Aeolus.IO.crash_dir); see [Blowup](#Blowup:-diagnose,-dump,-stop)) |
| `output/` | the run's output files ([`output_dir`](@ref Aeolus.IO.output_dir)) |
| `checkpoints/` | the [`CheckpointSchedule`](@ref Aeolus.IO.CheckpointSchedule) and FFTW wisdom ([`run_checkpoints`](@ref Aeolus.IO.run_checkpoints)) |
| `uncommitted-<n>.diff`, `untracked-<n>.txt` | the uncommitted code of session `n`, only with `allow_uncommitted = true` |

```julia
using Aeolus

with_run("gabls1"; config = cfg, seeds = (; noise = 42)) do rf
    ck  = run_checkpoints(rf; every_steps = 500)      # before the simulation: loads the wisdom
    sim = build_simulation(cfg)
    resume!(ck, sim)                                   # continues from the folder's newest checkpoint
    for n in ck.last_step+1:nsteps
        step_adaptive!(sim, cfl)
        maybe_checkpoint!(ck, sim; step = n)
    end
end
```

[`with_run`](@ref Aeolus.IO.with_run) calls [`start_run`](@ref Aeolus.IO.start_run),
installs the run log as the current logger while `f(rf)` runs, and
[`finish_run!`](@ref Aeolus.IO.finish_run!) at the end. If `f` throws, the
error and its backtrace go to `run.log`, the session is recorded as
`failed` (or `interrupted`) with the message, and the error propagates.

**Restart.** `with_run(case; resume = folder)` (or `resume = :latest`, the
newest folder of `case` under `root`) reuses the folder: the run ID stays,
`run.log` is appended to, and `run.toml` gets a new session of kind
`"restart"`. A restart whose config, seeds or git commit differ from the
first session logs a warning and records the new values in its session;
a previous session that never finished (no end time) is marked `aborted`.

**Uncommitted code is blocked.** Before anything is written, `start_run`
checks the git work tree of the Aeolus sources (`repo`): staged, unstaged
or untracked changes throw [`UncommittedCodeError`](@ref
Aeolus.IO.UncommittedCodeError), listing the files. `allow_uncommitted =
true` (or `AEOLUS_ALLOW_UNCOMMITTED=1`) runs anyway, saves `git diff HEAD`
and the untracked file list in the folder, and logs a warning. Ignored
files and the run folders themselves do not count (`runs/` is in
`.gitignore`). Code that is not in a git work tree (an installed release)
starts, with `git.available = false` in the record. Only `start_run`
looks at git: tests and in-memory runs are unaffected.

**What a session records** ([`provenance`](@ref Aeolus.IO.provenance)):

| table | keys |
| :-- | :-- |
| (session) | `number`, `kind`, `started`, `ended`, `status`, `wall_seconds`, `allow_uncommitted`, `error` |
| `git` | `available`, `repo`, `commit`, `branch`, `dirty`, `changes`, `diff_file`, `untracked_file` |
| `julia` | `version`, `commit` |
| `aeolus` | `version`, `source` |
| `manifest` | `path`, `fnv1a64` (FNV-1a 64-bit hash of the Manifest bytes) |
| `packages` | every package in the Manifest with its version |
| `host` | `hostname`, `os`, `arch`, `cpu`, `cpu_threads`, `julia_threads`, `blas_threads` |
| `command` | `julia` (the Julia command), `program`, `args`, `cwd` |
| `fftw` | `planner`, `threads`, `wisdom_loaded`, `wisdom_file`, `bitwise_reproducible` (refreshed by `run_checkpoints` and at the end) |
| `env` | every `AEOLUS_*` environment variable |

The log header carries the same: one `Info` record (`subsystem = :run`,
`event = :start` or `:restart`) with the run ID, folder, commit, versions,
host, threads and FFTW policy, shown on the console, and one `Debug`
record with the whole session record and the config, kept in `run.log`
only:

```
18:02:11.374 INFO  run[main] run started  event=start run_id=e072ceb3 case=gabls1 folder=…/runs/2026-10-06_gabls1_e072ceb3 session=1 commit=d08c8d30a5c0 branch=master julia=1.13.1 aeolus=0.0.1 manifest=8595c875cf129a87 host=geralt threads=1 fftw_planner=measure fftw_threads=1 wisdom_loaded=false bitwise_reproducible=false
```

The run ID comes from the operating system's entropy source, so starting
a run does not advance the global random number generator.

**Cost.** `start_run` + `finish_run!` take 11.6 ms per run (git calls,
provenance, `run.toml`, the header; 64×64×32 benchmark below). Nothing
is done per step: on the 64×64×32 Boussinesq case (`step_adaptive!`, one
thread, `benchmarks/bench_run_folder.jl`) a step took 13.37 ms with a
`NullLogger`, 13.17 ms inside `with_run` and 13.20 ms inside `with_run`
with one six-key `@debug` record per step, medians of 9 interleaved
repetitions on a loaded machine whose repetition spread (11.8 to 14.7 ms)
is far larger than the differences.

## Diagnostics registry

One generic, [`diagnostics`](@ref Aeolus.RunLog.diagnostics)`(component, sim)`,
declares the scalar diagnostics of anything that runs in a simulation
(#335). Each is a [`Diagnostic`](@ref Aeolus.RunLog.Diagnostic): a name, its
units, a **cadence class** (`:step`, cheap, every record; `:costly`, on
the writer's cadence, default every 60 s simulated) and a **reduction**
(`:max`, `:min`, `:maxabs`, `:sum`, `:mean`, `:rms`, or `:value` when the
source already returns the scalar), with a source function. The
reduction is declared rather than done by the source, so the threaded,
deterministic reductions of the KernelAbstractions port replace
[`reduce_diagnostic`](@ref Aeolus.RunLog.reduce_diagnostic) without
touching a component (#336).

```julia
# a component of your own
Aeolus.RunLog.diagnostics(c::MyTracer, sim) = [
    Diagnostic("mass", "kg", () -> tracer_mass(c, sim)),                        # :value, :costly
    Diagnostic("c_max", "kg m-3", () -> interior_view(c.field); reduction = :max),
    Diagnostic("source_rate", "kg s-1", () -> c.rate; cadence = :step),
]

ds = collect_diagnostics(sim, (; sgs = closure, wall = wm, tracer = c))
# ke_mean, max_abs_u, …, sgs.nu_h_max, …, wall.ustar_mean, …, tracer.mass, …
```

[`collect_diagnostics`](@ref Aeolus.RunLog.collect_diagnostics) takes the
simulation's own and prefixes each component's with its key; a name
declared twice is an error. A source that throws gives `NaN` and one
warning, so a broken diagnostic never stops a run. What the library
declares (each source calls the component's own function, one source of
truth):

| component | diagnostics (`:costly` unless marked) |
| :-- | :-- |
| `BoussinesqSimulation` | `ke_mean` (`kinetic_energy` / domain volume, m² s⁻²), `max_abs_u`, `max_abs_v`, `max_abs_w`, `theta_min`, `theta_max`, `max_div` (`max_divergence`), `nu_t_max`, `qv_max` (with moisture) |
| `MassWeightedSimulation` | `ke` (density-weighted, J), `max_abs_u/v/w`, `theta_min/max`, `max_div` (the core's own constraint: anelastic or pseudo-incompressible residual, s⁻¹) |
| any `AbstractSGSClosure` | `nu_h_max`, `nu_v_max`, `kappa_h_max` (via `sgs_fields`); `TKEClosure` adds `e_mean`, `e_max` |
| `WallModel` | `ustar_mean`, `ustar_max` of the bottom wall (the law at the first-cell speed, from `apply_wall_stress!` with a zero step) |
| `TBLEWallModel` | per wall used: `<wall>.utau_mean` (its active columns), `<wall>.kappa_wm` |
| `WallBoundedCase` | `tau_wall_bottom`, `tau_wall_top`, with a scalar `q_wall_bottom`, `q_wall_top` (all `:step`); a TBLE wall model's under `wall.` |
| `NestedRun` | `parent_steps`, `child_substeps`, `parent_retries`, `child_retries`, `substeps_last`, `max_child_cfl`, `max_clock_error` (`:step`); the stage wall times; every diagnostic of both domains under `parent.` and `child.` |

The Berlin couplings' `report` strings move onto the registry with #338.

## diagnostics.csv

A [`DiagnosticsWriter`](@ref Aeolus.RunLog.DiagnosticsWriter) appends one
row per [`record_diagnostics!`](@ref Aeolus.RunLog.record_diagnostics!):

```
step [1],t [s],wall [s],dt [s],dt_limiter,retries [1],ke_mean [m2 s-2],max_abs_u [m s-1],…
1,5.0,6.879,5.0,advective,0,0.5352798106352713,1.8665482844936574,…
2,10.0,7.458,5.0,advective,0,,,…
```

The costly cells are empty between their evaluations. Numbers are
Julia's shortest round-trip form, so a cell read back is the `Float64`
that was computed (tested bit for bit); a non-finite value is `NaN`,
`Inf` or `-Inf`. A restart appends below the same header; a file whose
columns differ is left alone and the session writes `diagnostics-2.csv`
with a warning. Each costly row is also a `Debug` record (`subsystem =
:diagnostics`) in `run.log`. `diagnostics.nc` (profiles, unlimited time
dimension) follows the NetCDF-4 switch (#310, #337).

## Progress line

A [`ProgressLine`](@ref Aeolus.RunLog.ProgressLine)`(t_end)` emits, at most
every `every` wall seconds (default 10), one `Info` record (`subsystem =
:progress`), so it reaches the console and `run.log`:

```
18:37:00.311 INFO  progress[main] t = 25.1 s (12.5 %)  719.9 steps/s  ETA 0:00:52  dt = 0.823 s (advective)  last warning (0:00:12 ago): MultigridPoissonSolver: maxcycles=50 reached …
```

with the key-values `t`, `percent`, `step`, `steps_per_s` (over the last
interval), `eta_s` (the simulated time per wall second since the start,
extrapolated), `dt`, `limiter` and `last_warning`. The last warning is
noted by a [`WarningTap`](@ref Aeolus.RunLog.WarningTap) around the current
logger ([`with_progress`](@ref Aeolus.RunLog.with_progress), or the run
monitor). The binding limit of dt comes from the caller until the step
cores report it (#336).

## Blowup: diagnose, dump, stop

A [`BlowupGuard`](@ref Aeolus.IO.BlowupGuard) checks the state for `NaN`
and `Inf` every `every_steps` steps (default 10) and keeps the last state
that passed as an in-memory [`checkpoint_payload`](@ref
Aeolus.IO.checkpoint_payload) (fields with halos, closure memory,
components, stepper, RNG). When the run blows up, a non-finite value at
a check ([`check_blowup!`](@ref Aeolus.IO.check_blowup!)) or a step that
throws ([`guarded_step!`](@ref Aeolus.IO.guarded_step!), e.g.
`step_adaptive!` out of retries), it

1. **diagnoses**: one `Error` record (`subsystem = :blowup`, `event =
   :blowup`) with the step, the time, the cause, and, per non-finite field
   ([`nonfinite_fields`](@ref Aeolus.IO.nonfinite_fields)), the count and
   the index and coordinates of the first bad value, then the last value
   of every diagnostic:

   ```
   18:37:00.649 ERROR blowup[main] blowup at step 26, t = 33.587 s: non-finite values in θ; crash checkpoint of step 21 written  …
       │ details =
       │   θ: 1 of 4096 non-finite (1 NaN, 0 Inf), first at (7, 7, 7) = (325.0, 325.0, 162.5) m
       │   last ke_mean = 5.670237360521602 m2 s-2  (step 14, t = 25.09735475430639 s)
       │   …
   ```
2. **dumps** the crash checkpoint `crash/crash_<step>.jls` of the last good
   state: the current state when it is still finite (a throwing
   `step_adaptive!` restores the start of its step), else the state of the
   last passing check. It is an ordinary checkpoint: `restart!` continues
   from it (tested bit for bit);
3. **stops** with a [`BlowupError`](@ref Aeolus.IO.BlowupError) carrying all
   of it. Inside `with_run`, the session is recorded as `failed`.

Only `BoussinesqSimulation` has a checkpoint format (#245) today; for the
other cores the diagnosis is logged and says that no checkpoint was
written.

## Run monitor

[`RunMonitor`](@ref Aeolus.IO.RunMonitor) puts the three behind one call
per step, [`monitor!`](@ref Aeolus.IO.monitor!) (or
[`monitored_step!`](@ref Aeolus.IO.monitored_step!), which also runs the
step under the guard and takes dt and the retries from
`step_adaptive!`'s return value). Given a run folder it writes the
folder's `diagnostics.csv` and `crash/`:

```julia
with_run("cbl"; config = cfg) do rf
    with_monitor(sim; run = rf, t_end = 4 * 3600.0, components = (; sgs = closure, wall = wm),
                 sgs = closure, stepper = st) do m
        dt = nothing
        for n in 1:nsteps
            _, dt, _ = monitored_step!(m, sim; step = n) do
                step_adaptive!(sim, cfl; dt_prev = dt, sgs = closure)
            end
        end
    end
end
```

The per-step hook in the step cores (#336, after the KernelAbstractions
step cores) will make this call itself, with dt's binding limit, the CFL
numbers and the solver statistics as `:step` diagnostics.

**Cost** (`benchmarks/bench_run_monitor.jl`, 64×64×32 Boussinesq,
`step_adaptive!`, one thread, 15 interleaved repetitions of 40 steps,
mean dt 0.14 s): 8.56 ms per step without a monitor, 8.70 ms with
`monitor!` after every step at the defaults (costly row every 60 s
simulated, progress line every 10 s wall, check and snapshot every 10
steps): **+1.6 %** (median of the paired ratios), within the 2 % budget
of epic #293, but the repetitions of the bare step spread over 8.31 to
9.93 ms. Timed alone, the parts are a cheap row 0.6 µs, a costly row
(the 8 core diagnostics) 1.46 ms, a blowup check 0.12 ms (a branch-free
vectorised finiteness test) and a snapshot 0.26 ms (the field copies
reuse the previous snapshot's buffers), which at the default cadence
add up to 0.042 ms per step, 0.49 %; the benchmark fails if that exceeds
2 %. The worst case, costly diagnostics, check and snapshot after every
step, costs +30 %.

## Cost of a record

Measured on a 64×64×32 Boussinesq case, `step_adaptive!`, one thread
(#333): 9.14 ms per step with a `NullLogger`, 9.11 ms with the run log
installed and no records (the difference is within the ±1.7 % run-to-run
spread), 9.16 ms with one six-key `@debug` record per step written to
`run.log`. One record costs 10.6 µs (0.12 % of the step), about half of
which is the standard library's macro.

The [flush policy](#Flush-policy) (PR #346) measured on one machine, one
thread, the same six-key `@debug` record to `run.log`, best of 7 × 20 000
records, three runs each: flushing every record (the old code, and
`flush_interval = 0`) costs 8.82–8.98 µs a record, the default
`flush_interval = 5` 7.89–7.98 µs, so the policy saves about 1 µs a
record (11 %). Timed alone, `format_record` is 2.5 µs, `write` + `flush`
of one line 0.68 µs against 0.05 µs for `write` alone, and the `time()`
call the policy adds 23 ns. (The #333 figure of 10.6 µs was measured on
a different day; the comparison here is like for like.)

## Tests

The test suite installs [`test_logger`](@ref Aeolus.RunLog.test_logger)
(warnings and up) in the header of `test/runtests.jl`, so every job of
`test/run.jl` has it too. `AEOLUS_LOG=debug` or `test/run.jl --log debug`
shows more. Tests that assert on events capture them with their own
logger (`Test.@test_logs`, `Test.TestLogger`), which the suite's logger
does not affect.

## Not here yet

Per-step diagnostics in the step cores and the threaded reductions (#336,
after the KernelAbstractions port), `diagnostics.nc` and the NetCDF
global attributes from the provenance (#337, after #310), and the Berlin
console text and coupling reports through the registry (#338).

## Reference

```@autodocs
Modules = [Aeolus.RunLog]
```

The registry's methods for the library's components:

```@autodocs
Modules = [Aeolus]
Pages = ["logging/component_diagnostics.jl"]
```
