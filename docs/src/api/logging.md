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
| `:pressure` | Warn | `CGPoissonSolver` hit `maxit` / breakdown | `solver = :cg`, `iterations`, `residual`, `relative_residual`, `tol`; for a breakdown `reason = :breakdown`, `curvature` |
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
| `Debug` | chatter: what a run log should keep but a console need not show | adaptive-step retries, checkpoint fresh start / resume / write, a costly diagnostics row, the Berlin setup summary |
| `Info` | milestones | the run header, the progress line, the Berlin output frames (with each coupling's diagnostics) and end-of-run events (stage timing, ledgers) |
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
| `run.toml` | provenance: `run_id`, `case`, `created`, the full `config` and `seeds`, one `[[session]]` per start or restart; for a [`RunConfig`](config.md) the canonical `[config]`, `config_hash`, `config_sources`, `[inputs]` hashes and `replayable` |
| `diagnostics.csv` | the scalar diagnostics time series ([`diagnostics_path`](@ref Aeolus.IO.diagnostics_path); written by the [run monitor](#Run-monitor)) |
| `diagnostics.nc` | the same rows plus the vertical profiles, NetCDF-4, with the provenance as global attributes ([`diagnostics_nc_path`](@ref Aeolus.IO.diagnostics_nc_path); see [diagnostics.nc](#diagnostics.nc)) |
| `crash/` | crash checkpoints of a blown-up run ([`crash_dir`](@ref Aeolus.IO.crash_dir); see [Blowup](#Blowup:-diagnose,-dump,-stop)) |
| `output/` | the run's output files ([`output_dir`](@ref Aeolus.IO.output_dir)) |
| `checkpoints/` | the [`CheckpointSchedule`](@ref Aeolus.IO.CheckpointSchedule) and FFTW wisdom ([`run_checkpoints`](@ref Aeolus.IO.run_checkpoints)) |
| `uncommitted-<n>.diff`, `untracked-<n>.txt` | the uncommitted code of session `n`, only with `allow_uncommitted = true` |

**`run.toml` `[config]` is the configuration of record.** For a run
started from a [`RunConfig`](config.md) (every `aeolus run`, every shipped
config), `[config]` is the canonical resolved TOML: every option written
out (every domain's sections under `[[domain]]`), paths absolute, floats bit-exact. It is what
`load_run_config(folder)` reads, what `aeolus run <folder>/run.toml`
replays bit for bit and what `aeolus run <folder> --resume` continues; the
free-form `config` of a run started from a `Dict` or a struct is kept for
provenance only and cannot be replayed. See
[Run configuration](../design/run_configuration.md).

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

The Berlin case (an example, not the library) implements the same
generic for each of its couplings, `diagnostics(c, case)`, and adds its
frame quantities, ledgers and stage timing; the `berlin` run driver
hands them to the run monitor (#338, [the Berlin case](../design/berlin_case.md#Diagnostics)).
Its wall-clock quantities are all named `wall.…`.

### Profiles

Vertical profiles are declared the same way (#337), through
[`profile_diagnostics`](@ref Aeolus.RunLog.profile_diagnostics)`(component, sim)`,
each a [`ProfileDiagnostic`](@ref Aeolus.RunLog.ProfileDiagnostic): a name,
units, the name and heights of its vertical axis, and a source that
returns the profile (the source does the horizontal reduction, through
the port's deterministic `plane_mean!` or the core's own profile
functions). [`collect_profiles`](@ref Aeolus.RunLog.collect_profiles)
prefixes them as `collect_diagnostics` does, the axis included (`parent.z`,
`child.z`). They are evaluated on the costly cadence and go to
`diagnostics.nc` only.

| component | profiles |
| :-- | :-- |
| `BoussinesqSimulation` | on `z` (cell centres): `u`, `v`, `theta`, `qv` (with moisture): horizontal means; `w_theta`, the resolved `⟨w'θ'⟩` (`resolved_flux_wθ`); on `zf` (w faces): `w_var` (`horizontal_variance`) |
| `MassWeightedSimulation` | `u`, `v`, `theta` on `z`, `w_var` on `zf` |
| any `AbstractSGSClosure` | `nu_v`, the mean vertical eddy viscosity (via `sgs_fields`); `TKEClosure` adds `e` |
| `NestedRun` | both domains' profiles under `parent.` and `child.`, on their own axes |

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
:diagnostics`) in `run.log`.

## diagnostics.nc

A [`DiagnosticsNCWriter`](@ref Aeolus.IO.DiagnosticsNCWriter) writes the
same rows as NetCDF-4 (#337, through [`NCIO`](physics.md): deflate,
shuffle, CF attributes), plus the profiles. The file has two unlimited
dimensions, so no costly quantity is padded with fill values between its
evaluations:

```
dimensions:
    time = UNLIMITED ;          // one record per row of diagnostics.csv
    time_costly = UNLIMITED ;   // one record per costly row
    z = 32 ; zf = 32 ;          // profile axes; equal levels share one
variables:
    double time(time) ; int64 step(time) ; double wall(time) ; double dt(time) ;
    string dt_limiter(time) ; int retries(time) ;      // + every :step scalar
    double time_costly(time_costly) ; int64 step_costly(time_costly) ;
    double ke_mean(time_costly) ; double max_abs_w(time_costly) ; …  // every :costly scalar
    double theta(time_costly, z) ; double w_var(time_costly, zf) ; …  // every profile
```

(`ncdump` order; in Julia `ds["theta"]` is `(z, time_costly)`.) Every
variable has `units` and `long_name`, a reduced scalar its CF
`cell_methods` (`area: maximum`, …). Time is in seconds of simulated
time; `time_origin = DateTime(…)` gives it CF units `seconds since …`, so
xarray and NCDatasets decode dates.

**Provenance.** In a run folder the global attributes are
[`nc_provenance_attributes`](@ref Aeolus.IO.nc_provenance_attributes) of
the session: `run_id`, `case`, `session`, `git_commit`, `git_branch`,
`git_dirty`, `julia_version`, `aeolus_version`, `manifest_fnv1a64`,
`hostname`, `julia_threads`, `blas_threads`, `fftw_planner`,
`fftw_threads`, `fftw_wisdom_loaded`, `bitwise_reproducible`, `command`,
`config` and `seeds` (the session's configuration and seeds as TOML, with
`config_hash` for a `RunConfig`), `created` (the run), `session_started`
and `session_finished` (set when the writer closes; empty while the
session runs or if it died), plus `title`, `source` and a `history` line per session.
Flags are `Int8` 0/1 (NetCDF has no boolean). `run.toml` stays the full
record (every session, every package version).

**Restart.** A restart appends to the file when its layout (variables,
units, cadence classes, profile levels; stored in the attribute
`aeolus_diagnostics_layout`) is the same, refreshes the attributes to the
new session and adds a `history` line; otherwise the session writes
`diagnostics-2.nc`, as the CSV does. The costly cadence restarts with the
session (its first row is costly).

**Flushing.** The rows of `time` are buffered and written as one block at
the first row 5 s wall after the last write (`flush_interval`, the
`run.log` policy), after 1000 rows, at every costly row, when a blowup is
thrown out of the monitor, and on `close`; each write ends with an HDF5
`sync`, so after a crash the file holds every written row. A sync costs
about 2 ms, which is why the rows are not written one by one.

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

The check covers the state the checkpoint components hold outside the
simulation too ([`component_nonfinite_fields`](@ref
Aeolus.IO.component_nonfinite_fields)): in a nested model the guard steps
the root's core, and the nest reports its child domains, each field named
`<domain>.<field>` (`inner.θ`). A one-way child feeds nothing back, so
without this a NaN in it never reached the root and the run ended
`completed` (`test/model_nesting.jl`).

Only `BoussinesqSimulation` has a checkpoint format (#245) today; for the
other cores the diagnosis is logged and says that no checkpoint was
written.

## Run monitor

[`RunMonitor`](@ref Aeolus.IO.RunMonitor) puts the three behind one call
per step. Given a run folder it writes the folder's `diagnostics.csv`,
`diagnostics.nc` and `crash/`; each scalar is evaluated once per row for
both files. [`with_monitor`](@ref Aeolus.IO.with_monitor) **attaches** it
to the simulation by default (#336): it becomes the simulation's
diagnostics hook, so the step cores report every step themselves and the
driver only steps:

```julia
with_run("cbl"; config = cfg) do rf
    with_monitor(sim; run = rf, t_end = 4 * 3600.0, components = (; sgs = closure, wall = wm),
                 sgs = closure, stepper = st) do m
        dt = nothing
        for n in 1:nsteps
            _, dt, _ = monitored_step!(m, sim; step = n) do   # or just step_adaptive!(…)
                step_adaptive!(sim, cfl; dt_prev = dt, sgs = closure)
            end
        end
    end
end
```

[`monitored_step!`](@ref Aeolus.IO.monitored_step!) is then only needed
to turn a step that throws into a diagnosed blowup; the row came from the
hook. Rows are numbered `step0 + 1, step0 + 2, …` (`step0` keyword: the
step a resumed run continues from). A driver whose step is several core
steps, or which numbers its own steps (a run that resumes inside the
driver), passes `attach = false` and calls
[`monitor!`](@ref Aeolus.IO.monitor!) after its step: the row then covers
every core step since the previous row, with `dt` the time they covered,
the last limiter and the summed retries (the SBL and Berlin drivers do;
the bare-simulation and DNS drivers are attached).

## Per-step diagnostics in the step cores

Every step core (`step!`, `step_rk3!`, `step_imex!`, `step_adaptive!`,
the sound-proof cores, and per domain inside `step_nested!`) records its
step in the simulation's
[`StepRecord`](@ref Aeolus.Simulations.StepRecord) and then calls the
simulation's hook ([`set_step_hook!`](@ref Aeolus.Simulations.set_step_hook!);
an attached monitor's [`MonitorHook`](@ref Aeolus.IO.MonitorHook)). A
core called inside another is part of the outer step (`step!` inside
`step_adaptive!`), a rejected attempt is not a step, a step that throws
records nothing. What each step costs to record is a few `time_ns()`
calls and additions: nothing is swept for it.

| quantity | diagnostic (every row) | how it is known |
| :------- | :--------------------- | :-------------- |
| dt | `dt` column | the step taken |
| its binding limit | `dt_limiter` column | the limiter of the dt pick made from the step's start state ([`cfl_diagnostics`](@ref Aeolus.TimeIntegration.cfl_diagnostics), [`adaptive_dt`](@ref Aeolus.TimeIntegration.adaptive_dt), also when the driver called them); `growth`, `dt_max`, `dt_min` when a clamp bound; `retry` (shortened by rejected attempts); `prescribed` (a pick was made but another dt taken); `fixed` (no pick) |
| retries | `retries` column | `step_adaptive!`'s, a nested domain's |
| CFL advective, diffusive | `cfl_adv`, `cfl_diff` | `dt ×` the rates of that pick: `max_cell Σ|uᵢ|/Δxᵢ` and `max(ν + νₜ, κₜ)(Δx⁻² + Δy⁻² + Δz_min⁻²)`; `NaN` without a pick |
| solver iterations, residual | `solver_iters`, `solver_residual` | [`solver_stats`](@ref Aeolus.PressureSolvers.solver_stats) after every pressure solve: summed, largest; a direct solver gives 0 and `NaN` |
| implicit diffusion | `implicit_iters` | `step_imex!`'s CG iterations |

On the costly cadence (default 60 s simulated): the core's KE, `max_div`,
θ and w extremes (the Field reductions below), the cumulative wall time
per stage `wall_step`, `wall_pressure`, `wall_sgs`, `wall_cfl` (the CFL
sweeps of the dt picks), `wall_guard` (`step_adaptive!`'s snapshot,
guard and rollbacks), `cfl_adv_state` and `cfl_diff_state` (the CFL of
the current state at the last dt, one fused sweep: the CFL of a
fixed-step run, which makes no pick), and the process's memory
(`proc.mem_live_mb`, `proc.mem_maxrss_mb`, `proc.gc_time`,
[`process_diagnostics`](@ref Aeolus.RunLog.process_diagnostics)). A
nested run reports per parent step (its own record, completed in
`end_parent_step!`): `cfl_adv` the largest child-substep CFL of the
window, `wall_step`, and for a two-way run with a ledger the relative
conservation errors `ledger_<quantity>`; each domain's record counts its
accepted steps. The ledger residuals of the Berlin couplings come with
their diagnostics (#338).

**Deterministic reductions.** A [`Diagnostic`](@ref Aeolus.RunLog.Diagnostic)
source that returns a `Field` is reduced over the interior with the
KernelAbstractions port's reductions: `range_max` for `:max`, `:min`,
`:maxabs` (exact, the bits of a serial maximum), the canonical plane
sums for `:sum`, `:mean`, `:rms` (the same bits at any thread count). The
core and closure diagnostics return their Fields.

**Why not a CFL sweep every step.** On the 64×64×32 case below a fixed
`step!` takes 5.3 ms; the advective-rate sweep 0.085 ms (1.6 %) and the
fused sweep with the SGS maxima 0.15 ms (2.8 %): either alone would
exceed the 2 % budget. So the every-row CFL numbers are those of the
pick the step already made, and a fixed-step run gets its CFL on the
costly cadence (`cfl_*_state`).

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

With `diagnostics.nc` (#337; same benchmark, a later day, 7.18 ms per bare
step, mean dt 0.10 s): the CSV-only monitor **+1.40 %**, the monitor
with `diagnostics.nc` **+1.55 %** (paired medians; the bare-step
repetitions spread over 6.99 to 7.38 ms). Its parts: a buffered row
0.2 µs, a block write plus sync 2.13 ms (once per 5 s wall), a costly row
6.5 ms (0.26 ms of it the five profiles, the rest the write and the
sync; once per 60 s simulated). At the default cadence all parts add up
to 0.060 ms per step, 0.84 %. The worst case (everything every step) is
+137 %, dominated by the per-step HDF5 sync of the costly row. A first
version synced every 20 rows: 146 µs per step amortised, 3.1 % from the
parts, over budget; hence the wall-time flush.

With the step record and the attached monitor (#336; same benchmark, 32
steps × 15 repetitions so that the case stays below the ~2930 steps after
which it blows up, on master too; a shared machine at 1-minute load 8 to
9, 5.38 ms per bare step, mean dt 0.10 s): the record's bracket around a
step with a no-op hook costs **45 ns** (0.001 %); a cheap row with the
step-record columns 2.0 µs (0.6 µs before), a costly row 0.46 ms (the
core, step-record and process diagnostics, with the port's Field
reductions), a blowup check 0.13 ms, a snapshot 0.36 ms; diagnostics.nc's
block write and sync 5.5 ms and its costly row 15 ms (both slower than
the #337 figures, I/O under that load). At the default cadence the parts
add up to **0.084 ms per step, 1.56 %** of a step, within the asserted
2 %. The whole-step comparison is within the run-to-run noise at that
load: the attached monitor +3.05 % and the driver-called one +5.63 %
(paired medians) in one run, +4.03 % and +1.00 % in the run before (the
bare-step repetitions spread over 5.26 to 5.61 ms).

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

The Berlin output file
(`src/components/urban/site/output.jl`) does not carry the provenance
attributes yet: it is not written inside a run folder.

## Reference

```@autodocs
Modules = [Aeolus.RunLog]
```

The step record of the step cores (#336) is documented with the
[Simulations](simulations.md#The-step-record-and-the-diagnostics-hook-(#336)) API.

The registry's methods for the library's components:

```@autodocs
Modules = [Aeolus]
Pages = ["logging/component_diagnostics.jl"]
```
