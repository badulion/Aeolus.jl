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
        step_adaptive!(sim, cfl; verbose = false)
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
| `:timestepping` | Warn if `verbose`, else Debug | `step_adaptive!`: the stepper threw | `dt`, `retry`, `exception` |
| `:timestepping` | Info if `verbose`, else Debug | `step_adaptive!`: retry at a smaller step | `reason`, `from`, `to`, `retry`, `max_retries` |
| `:timestepping` | Warn | `step_adaptive!`: a fast process left non-finite values | `dt`, `processes` |
| `:checkpoint` | Warn | `restart!` across Julia versions / with `allow_mismatch` | `path`, `written_by`, `running` / `parameters` |
| `:checkpoint` | Info if `verbose`, else Debug | `resume!`: fresh start / resumed | `event = :fresh_start \| :resume`, `dir` / `path`, `t`, `step` |
| `:checkpoint` | Debug | `maybe_checkpoint!` wrote a checkpoint | `event = :write`, `path`, `step`, `t`, `removed` |

So a healthy run emits nothing at `Info` or above: the default console
stays quiet, and `run.log` records the retries and checkpoints. Under a
`NullLogger` nothing reaches stdout or stderr at all.

## Run folder and provenance

A production run writes one folder (#334), `runs/<yyyy-mm-dd>_<case>_<runID>/`:

| file | contents |
| :-- | :------ |
| `run.log` | the run log: debug and up, full timestamps, source locations; appended to by every session |
| `run.toml` | provenance: `run_id`, `case`, `created`, the full `config` and `seeds`, one `[[session]]` per start or restart |
| `diagnostics.csv` | the diagnostics time series ([`diagnostics_path`](@ref Aeolus.IO.diagnostics_path); written from #335 on, `diagnostics.nc` after #310) |
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

## Cost of a record

Measured on a 64×64×32 Boussinesq case, `step_adaptive!`, one thread
(#333): 9.14 ms per step with a `NullLogger`, 9.11 ms with the run log
installed and no records (the difference is within the ±1.7 % run-to-run
spread), 9.16 ms with one six-key `@debug` record per step written to
`run.log`. One record costs 10.6 µs (0.12 % of the step), about half of
which is the standard library's macro; the file is flushed after every
record so a crash loses nothing.

## Tests

The test suite installs [`test_logger`](@ref Aeolus.RunLog.test_logger)
(warnings and up) in the header of `test/runtests.jl`, so every job of
`test/run.jl` has it too. `AEOLUS_LOG=debug` or `test/run.jl --log debug`
shows more. Tests that assert on events capture them with their own
logger (`Test.@test_logs`, `Test.TestLogger`), which the suite's logger
does not affect.

## Not here yet

Per-step diagnostics in the step cores (after the KernelAbstractions
port), the diagnostics registry, CSV and `diagnostics.nc` time series,
the progress line and blowup handling (#335), NetCDF global
attributes from the provenance (#337), and the Berlin console text
(#338).

## Reference

```@autodocs
Modules = [Aeolus.RunLog]
```
