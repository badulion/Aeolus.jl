# I/O, checkpoints and restart

`Aeolus.IO` holds the run's persistence: field snapshots for output,
**full-state, versioned checkpoints** from which a run continues bit for
bit (#245), periodic checkpoints with automatic resume for long drivers,
and TOML namelists. The design (what a checkpoint must contain, why the
FFT planner matters, what is not covered yet) is on the design page
[Checkpoint / restart and FFT planning](../design/checkpoint_restart.md).

## Quick reference

```julia
using Aeolus, Random

sim = BoussinesqSimulation(mesh; θ₀ = 300.0, bcs = bcs)
c   = TKEClosure(mesh); initialize_tke!(c, sim)
st  = StepperState(:adaptive)
cfl = CFLConstraints(; dt_max = 5.0)

for _ in 1:100
    _, dt, nr = step_adaptive!(sim, cfl; dt_prev = previous_dt(st), sgs = c)
    record_step!(st, dt; retries = nr)
end
checkpoint!(sim, "run/ckpt.jls"; sgs = c, stepper = st, rng = Random.default_rng(),
            extra = Dict(:note => "after 100 steps"))

# … in another process, with the run built the same way:
restart!(sim2, "run/ckpt.jls"; sgs = c2, stepper = st2, rng = Random.default_rng())
read_checkpoint("run/ckpt.jls")[:extra][:note]
```

A wall-modelled channel is checkpointed through its case harness, one
component that carries the closure memory, the wall model (TBLE columns,
input filters), the dynamic roughness and the counters:

```julia
checkpoint!(case.sim, path; components = (case = case,))
restart!(case2.sim, path; components = (case = case2,))
```

A long driver hands its run loop a [`CheckpointSchedule`](@ref): built
**before** the simulation (it loads the folder's FFTW wisdom), resumed
after it, checked after every step:

```julia
ck = CheckpointSchedule("run"; every_steps = 500, every_seconds = 1800, keep = 2)
sim = …; series = Float64[]
resume!(ck, sim; sgs = c, stepper = st, components = (series = series,))
while st.step < nsteps
    step_rk3!(sim, dt; sgs = c); record_step!(st, dt); push!(series, kinetic_energy(sim))
    maybe_checkpoint!(ck, sim; step = st.step, sgs = c, stepper = st,
                      components = (series = series,))
end
```

`run_channel!` (WallModels) and the SBL driver's `sbl_run!` take the
schedule as their `checkpoint` keyword; `dns256_generate.jl`, the
relaminarisation driver and `gabls1`–`gabls4.jl` use it by default
([`driver_checkpoints`](@ref): every 30 min of wall clock,
`AEOLUS_CHECKPOINT=0` / `_EVERY` / `_SECONDS` / `_KEEP` override).

A production run keeps its log, provenance, output and checkpoints in a
**run folder**, `runs/<date>_<case>_<runID>/` ([`start_run`](@ref),
[`with_run`](@ref), [`run_checkpoints`](@ref); #334). It refuses to start
on uncommitted code unless `allow_uncommitted = true`; a restart reuses
the folder and keeps the run ID. See [Run folder and
provenance](logging.md#Run-folder-and-provenance) on the Logging page.

[`checkpoint!`](@ref) is [`write_checkpoint`](@ref) of
[`checkpoint_payload`](@ref), the in-memory checkpoint. A
[`BlowupGuard`](@ref) keeps the last good state of a run as such a payload
and, when the run blows up, writes it as the crash checkpoint
`crash/crash_<step>.jls`, which [`restart!`](@ref) continues from like any
other (#335; see [Blowup](logging.md#Blowup:-diagnose,-dump,-stop) on the
Logging page).

## FFT planning policy

The planner of every FFT plan is set in one place,
`Aeolus.Operators.set_fftw_policy!` (`planner = :estimate | :measure |
:patient | :exhaustive`, `threads`), default `:measure` (`:patient`
opt-in), or from the
environment (`AEOLUS_FFTW_PLANNER`, `AEOLUS_FFTW_THREADS`) or a namelist
`[fftw]` table ([`apply_fftw_config!`](@ref)). `fftw_policy()` reports
the planner, the thread count, whether wisdom was loaded and whether the
run is bitwise reproducible; every checkpoint records it, and
`save_fftw_wisdom` / `load_fftw_wisdom!` keep a restart on the same plans.
The suite runs with `AEOLUS_FFTW_PLANNER=estimate`. These functions are
documented with [Operators](operators.md#FFT-planning-policy).

## Namelist tables

```toml
[fftw]
planner = "measure"        # estimate | measure (default) | patient | exhaustive
threads = 4
wisdom  = "run/fftw_wisdom"

[checkpoint]
dir           = "run"
every_steps   = 500
every_seconds = 1800.0
keep          = 2
```

`checkpoint_schedule(load_config("run.toml"))` applies the `[fftw]` table
and returns the schedule of the `[checkpoint]` table.

## Reference

```@autodocs
Modules = [Aeolus.IO]
```
