# The run monitor (LOG-c, #335): the diagnostics CSV, the progress line and
# the blowup guard behind one call per step, `monitor!`. Included into
# `module IO`, after blowup.jl. Attached to a simulation (the default of
# `with_monitor`), it is the diagnostics hook of the step cores (#336):
# every step core reports its step to it, so a driver only installs it.

using ..RunLog: ProgressLine, progress!, WarningTap, collect_diagnostics, record_diagnostics!,
                Diagnostic, ProfileDiagnostic, collect_profiles, prefixed, process_diagnostics
using ..Simulations: Simulations, StepRecord, step_record, set_step_hook!, step_hook, reset_interval!

"`path/crash`, the folder of a run's crash checkpoints (created)."
crash_dir(rf::RunFolder) = mkpath(joinpath(rf.path, "crash"))

"""
    RunMonitor(sim; run = nothing, components = (;), diagnostics = nothing,
               t_end = nothing, t_start = sim.t, cadence = 60.0,
               progress_every = 10.0, csv = …, nc = …, profiles = nothing,
               nc_attributes = …, time_origin = nothing, check_every = 10,
               crash = …, sgs = nothing, checkpoint_components = (;),
               stepper = nothing, rng = nothing, attach = false, step0 = 0,
               process = true)

Everything a run reports, behind one call per step, [`monitor!`](@ref)
(or none, attached to the step cores):

* the **diagnostics** `diagnostics` (default:
  [`collect_diagnostics`](@ref Aeolus.RunLog.collect_diagnostics)`(sim,
  components)` and, with `process = true`, the
  [`process_diagnostics`](@ref Aeolus.RunLog.process_diagnostics) under
  `proc.`) written to the CSV `csv` (default: the run folder's
  [`diagnostics_path`](@ref) when `run` is a [`RunFolder`](@ref), else
  no file), the costly ones every `cadence` simulated seconds;
* the same rows, plus the **profiles** `profiles` (default:
  [`collect_profiles`](@ref Aeolus.RunLog.collect_profiles)`(sim,
  components)`) on the costly rows, written to the NetCDF-4 file `nc`
  ([`DiagnosticsNCWriter`](@ref); default: the run folder's
  [`diagnostics_nc_path`](@ref), else no file) with the global
  attributes `nc_attributes` (default: the run folder's
  [`nc_provenance_attributes`](@ref)); each scalar is evaluated once per
  row for both files;
* the **progress line** towards `t_end` (none when `t_end = nothing`), at
  most every `progress_every` wall seconds;
* the **blowup guard** ([`BlowupGuard`](@ref)): a check every
  `check_every` steps, crash checkpoints in `crash` (default: the run
  folder's `crash/`, else none) of the state with `sgs`,
  `checkpoint_components`, `stepper` and `rng`, as [`checkpoint!`](@ref)
  takes them.

**Attached** (`attach = true`, the default of [`with_monitor`](@ref);
#336) the monitor is the diagnostics hook of `sim`
([`set_step_hook!`](@ref Aeolus.Simulations.set_step_hook!)): every step
core reports each step of `sim` to it as it ends (a `NestedRun`: each
parent step), with the step's dt, its limiter and its retries from the
simulation's [`StepRecord`](@ref Aeolus.Simulations.StepRecord), and the
monitor runs the blowup check, writes the row and the progress line.
The rows are numbered `step0 + 1, step0 + 2, …` (pass the step a resumed
run continues from). A driver then calls neither `monitor!` nor
[`monitored_step!`](@ref) for the reporting (`monitored_step!` still
turns a step that throws into a diagnosed blowup). Without `attach`, the
driver reports with `monitor!`; the row then covers every step core call
since the previous row (the record's interval), so a driver step made of
several core steps is one row.

`close(m)` flushes and closes the CSV and the NetCDF file and removes the
hook. [`with_monitor`](@ref) does that and feeds the warnings to the
progress line.
"""
mutable struct RunMonitor
    writer::Union{Nothing,DiagnosticsWriter}
    nc::Union{Nothing,DiagnosticsNCWriter}
    progress::Union{Nothing,ProgressLine}
    guard::BlowupGuard
    steps::Int
    step::Int
    dt::Float64
    limiter::String
    sim::Any
    attached::Bool
end

"""
    MonitorHook(monitor)

The diagnostics hook an attached [`RunMonitor`](@ref) installs on its
simulation ([`attach_monitor!`](@ref)): called by the step cores as
`hook(sim, record)` at the end of every step, it runs the monitor's
blowup check, row and progress line for that step (#336).
"""
struct MonitorHook
    monitor::RunMonitor
end
(h::MonitorHook)(sim, rec::StepRecord) = _hook_report!(h.monitor, sim, rec)

# The start of a step of an attached monitor's simulation (#514): before a
# due check, take what a crash checkpoint of this step's start state needs
# besides the fields (the step's rollback snapshot holds those; see
# `check_blowup!(guard, sim, record)`).
function Simulations.step_begun(h::MonitorHook, sim, rec::StepRecord)
    m = h.monitor; g = m.guard
    g.pending = nothing
    step = m.step + 1
    (sim === m.sim && step - g.last_check >= g.every_steps && can_checkpoint(sim)) || return nothing
    g.pending = (step = step, payload = _payload(g, sim, step - 1; with_fields = false))
    return nothing
end

function RunMonitor(sim; run::Union{Nothing,RunFolder} = nothing, components = (;),
                    diagnostics::Union{Nothing,AbstractVector{Diagnostic}} = nothing,
                    t_end::Union{Nothing,Real} = nothing,
                    t_start::Real = hasproperty(sim, :t) ? sim.t : 0.0,
                    cadence::Real = 60.0, progress_every::Real = 10.0,
                    csv::Union{Nothing,AbstractString} = run === nothing ? nothing : diagnostics_path(run),
                    nc::Union{Nothing,AbstractString} = run === nothing ? nothing : diagnostics_nc_path(run),
                    profiles::Union{Nothing,AbstractVector{ProfileDiagnostic}} = nothing,
                    nc_attributes::AbstractDict = run === nothing ? Dict{String,Any}() :
                                                  nc_provenance_attributes(run),
                    time_origin = nothing,
                    check_every::Integer = 10,
                    crash::Union{Nothing,AbstractString} = run === nothing ? nothing : crash_dir(run),
                    sgs = nothing, checkpoint_components = (;), stepper = nothing, rng = nothing,
                    attach::Bool = false, step0::Integer = 0, process::Bool = true)
    diags = diagnostics !== nothing ? diagnostics :
            process ? vcat(collect_diagnostics(sim, components), prefixed("proc", process_diagnostics())) :
                      collect_diagnostics(sim, components)
    writer = csv === nothing ? nothing : DiagnosticsWriter(csv, diags; cadence = cadence)
    ncw = nc === nothing ? nothing :
          DiagnosticsNCWriter(nc, diags, profiles === nothing ? collect_profiles(sim, components) : profiles;
                              cadence = cadence, attributes = nc_attributes,
                              time_origin = time_origin)
    prog = t_end === nothing ? nothing :
           ProgressLine(t_end; t_start = t_start, every = progress_every)
    guard = BlowupGuard(sim; every_steps = check_every, dir = crash, sgs = sgs,
                        components = checkpoint_components, stepper = stepper, rng = rng,
                        diagnostics = writer)
    m = RunMonitor(writer, ncw, prog, guard, 0, Int(step0), NaN, "", sim, false)
    attach && attach_monitor!(m)
    return m
end

"""
    attach_monitor!(m::RunMonitor) -> m

Make `m` the diagnostics hook of its simulation (see [`RunMonitor`](@ref),
`attach`): every step core then reports each step to it. An
`ArgumentError` if the simulation already has a hook, or has no
[`StepRecord`](@ref Aeolus.Simulations.StepRecord).
"""
function attach_monitor!(m::RunMonitor)
    m.attached && return m
    sim = m.sim
    step_record(sim) === nothing &&
        throw(ArgumentError("RunMonitor: a $(typeof(sim)) has no step record to attach to; report with monitor!"))
    step_hook(sim) === nothing ||
        throw(ArgumentError("RunMonitor: the simulation already has a diagnostics hook (another monitor?)"))
    reset_interval!(step_record(sim))
    set_step_hook!(sim, MonitorHook(m))
    m.attached = true
    return m
end

"""
    detach_monitor!(m::RunMonitor) -> m

Remove `m` as the diagnostics hook of its simulation (`close` does it).
"""
function detach_monitor!(m::RunMonitor)
    m.attached || return m
    h = step_hook(m.sim)
    h isa MonitorHook && h.monitor === m && set_step_hook!(m.sim, nothing)
    m.attached = false
    return m
end

# An attached monitor: the end of a step of `sim` (#336)
function _hook_report!(m::RunMonitor, sim, rec::StepRecord)
    step = m.step + 1
    _flushing_nc(m) do
        check_blowup!(m.guard, sim, rec; step)
    end
    _report!(m, sim; step, dt = rec.dt, limiter = rec.limiter, retries = rec.retries_sum)
    return nothing
end

# After a row: the next row covers the steps from here (the run's own
# record and, for a nested run, those of its domains).
_reset_records!(sim) = (r = step_record(sim); r === nothing || reset_interval!(r); nothing)

"""
    monitor!(m::RunMonitor, sim; step = m.step + 1, dt = nothing, limiter = nothing,
             retries = nothing) -> m

After step `step` (which took `dt`, limited by `limiter`, with `retries`
rejected attempts): the blowup check (on its cadence; a blowup throws a
[`BlowupError`](@ref) after the diagnosis and the crash checkpoint), a
diagnostics row, and the progress line when due. What is not given is
taken from the simulation's [`StepRecord`](@ref
Aeolus.Simulations.StepRecord) (#336), over the step core calls since
the previous row: `dt` the simulated time they covered, `limiter` that of
the last one, `retries` their sum (`NaN`, `""`, `0` without a record or
a step). An attached monitor
reports by itself; a call of `monitor!` adds a row.
"""
function monitor!(m::RunMonitor, sim; step::Integer = m.step + 1, dt = nothing, limiter = nothing,
                  retries = nothing)
    _flushing_nc(m) do
        check_blowup!(m.guard, sim; step)
    end
    r = step_record(sim)
    stepped = r !== nothing && r.n > 0
    dt === nothing && (dt = stepped ? r.dt_sum : NaN)
    limiter === nothing && (limiter = stepped ? r.limiter : "")
    retries === nothing && (retries = r === nothing ? 0 : r.retries_sum)
    _report!(m, sim; step, dt, limiter, retries)
    return m
end

# the row and the progress line of a step
function _report!(m::RunMonitor, sim; step, dt, limiter, retries, force = false)
    t = hasproperty(sim, :t) ? Float64(sim.t) : NaN
    lim = limiter === nothing ? "" : string(limiter)
    costly = nothing
    m.writer === nothing ||
        (costly = record_diagnostics!(m.writer, t; step, dt, limiter = lim, retries))
    if m.nc !== nothing
        if costly === nothing
            record_diagnostics!(m.nc, t; step, dt, limiter = lim, retries)
        else   # the CSV writer has just evaluated every scalar of this row
            record_diagnostics!(m.nc, t; step, dt, limiter = lim, retries, costly,
                                values = m.writer.last)
        end
    end
    m.progress === nothing ||
        progress!(m.progress, t; step, dt, limiter = isempty(lim) ? nothing : lim, force)
    m.steps += 1; m.step = step; m.dt = dt; m.limiter = lim
    _reset_records!(sim)
    return m
end

"""
    monitored_step!(f, m::RunMonitor, sim; step, limiter = "") -> f()

One step `f()` under the monitor: run by [`guarded_step!`](@ref) (a
step that throws is a blowup), then [`monitor!`](@ref). When `f` returns
`step_adaptive!`'s `(sim, dt, retries)`, the dt and the retries go into
the row and the progress line. An attached monitor has already reported
the step cores' steps of `f` itself; then only a throw is handled here
(and a row written if `f` took no core step of `sim`).

```julia
with_run("cbl"; config = cfg) do rf
    with_monitor(sim; run = rf, t_end = 3600.0, components = (; sgs = closure)) do m
        dt = nothing
        for n in 1:nsteps
            _, dt, _ = monitored_step!(m, sim; step = n) do
                step_adaptive!(sim, cfl; dt_prev = dt, sgs = closure)
            end
        end
    end
end
```
"""
function monitored_step!(f, m::RunMonitor, sim; step::Integer, limiter = "")
    m.attached && sim === m.sim && return _attached_step!(f, m, sim, step, limiter)
    result = _flushing_nc(m) do
        guarded_step!(f, m.guard, sim; step)
    end
    dt, retries = NaN, 0
    r = step_record(sim)
    stepped = r !== nothing && r.n > 0                   # the step cores' record (#336)
    if result isa Tuple && length(result) == 3 && result[2] isa Real && result[3] isa Integer
        dt, retries = Float64(result[2]), Int(result[3])
    elseif stepped
        dt, retries = r.dt_sum, r.retries_sum
    end
    stepped && isempty(string(limiter)) && (limiter = r.limiter)
    # guarded_step! has already run the cadence check of this step
    _report!(m, sim; step, dt, limiter, retries)
    return result
end

function _attached_step!(f, m::RunMonitor, sim, step, limiter)
    rec = step_record(sim)
    s0 = rec.steps
    result = _flushing_nc(m) do
        try
            f()
        catch err
            err isa InterruptException && rethrow()
            err isa BlowupError && rethrow()
            blowup!(m.guard, sim; step = m.step + 1, cause = err, backtrace = catch_backtrace())
        end
    end
    if rec.steps == s0              # no step core reported: a row of its own
        _flushing_nc(m) do
            check_blowup!(m.guard, sim; step)
        end
        _report!(m, sim; step, dt = NaN, limiter, retries = 0)
    end
    return result
end

# a blowup throws out of the monitor: the rows buffered for diagnostics.nc
# are written first, as the guard flushes the CSV
function _flushing_nc(f, m::RunMonitor)
    try
        return f()
    catch
        m.nc === nothing || flush(m.nc)
        rethrow()
    end
end

function Base.close(m::RunMonitor)
    detach_monitor!(m)
    m.writer === nothing || close(m.writer)
    m.nc === nothing || close(m.nc)
    return nothing
end

"""
    with_monitor(f, sim; attach = true, kwargs...) -> f(m)

Make a [`RunMonitor`](@ref)`(sim; attach, kwargs...)`, run `f(m)` with the
monitor's progress line noting every warning (a
[`WarningTap`](@ref Aeolus.RunLog.WarningTap) around the current
logger), emit a final progress line, and close the CSV and
`diagnostics.nc` (and remove the hook), also when `f` throws.

By default the monitor is **attached** to `sim` (#336): the step cores
report every step to it, so `f` only steps:

```julia
with_monitor(sim; run = rf, t_end = 3600.0, components = (; sgs = closure)) do m
    while sim.t < 3600.0
        step_adaptive!(sim, cfl; sgs = closure)
    end
end
```

A simulation without a [`StepRecord`](@ref Aeolus.Simulations.StepRecord)
is not attached (the driver reports with [`monitor!`](@ref)); pass
`attach = false` for a driver whose step is several core steps and which
reports once per its own step.
"""
function with_monitor(f, sim; attach::Bool = true, kwargs...)
    m = RunMonitor(sim; attach = attach && step_record(sim) !== nothing, kwargs...)
    try
        if m.progress === nothing
            return f(m)
        end
        return Logging.with_logger(WarningTap(Logging.current_logger(), m.progress)) do
            r = f(m)
            t = hasproperty(sim, :t) ? Float64(sim.t) : NaN
            m.steps > 0 && progress!(m.progress, t; step = m.step, dt = m.dt,
                                     limiter = isempty(m.limiter) ? nothing : m.limiter,
                                     force = true)
            r
        end
    finally
        close(m)
    end
end

# A step record's hook is the run session's monitor (its log, its writer,
# its open files), not state of the model: a structural checkpoint
# (`structural_state`, #269) of a model whose cores carry it records no
# hook, and a restore keeps the fresh run's own (ONE-f, #455: the nest
# of a monitored model run is checkpointed structurally).
const _SESSION_HOOKS = IdDict{StepRecord,Any}()
const _SESSION_HOOKS_LOCK = ReentrantLock()
_hold_hook!(r::StepRecord) = (lock(() -> (_SESSION_HOOKS[r] = r.hook), _SESSION_HOOKS_LOCK); r.hook = nothing)
_give_hook!(r::StepRecord) = (r.hook = lock(() -> pop!(_SESSION_HOOKS, r), _SESSION_HOOKS_LOCK); nothing)
structural_save_begin(r::StepRecord) = (_hold_hook!(r); nothing)
structural_save_end(r::StepRecord, _) = _give_hook!(r)
structural_restore_begin(r::StepRecord, _) = (_hold_hook!(r); nothing)
structural_restore_end(r::StepRecord, _) = _give_hook!(r)
