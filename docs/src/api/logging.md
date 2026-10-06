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

## Cost

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
the progress line and blowup handling (#335), the run folder and
provenance (#334), and the Berlin console text (#338).

## Reference

```@autodocs
Modules = [Aeolus.RunLog]
```
