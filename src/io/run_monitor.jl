# The run monitor (LOG-c, #335): the diagnostics CSV, the progress line and
# the blowup guard behind one call per step, `monitor!`. Included into
# `module IO`, after blowup.jl. The per-step hook in the step cores
# (#336, after the KernelAbstractions step cores) will call `monitor!`;
# until then a driver calls it after each step.

using ..RunLog: ProgressLine, progress!, WarningTap, collect_diagnostics, record_diagnostics!,
                Diagnostic

"`path/crash`, the folder of a run's crash checkpoints (created)."
crash_dir(rf::RunFolder) = mkpath(joinpath(rf.path, "crash"))

"""
    RunMonitor(sim; run = nothing, components = (;), diagnostics = nothing,
               t_end = nothing, t_start = sim.t, cadence = 60.0,
               progress_every = 10.0, csv = …, check_every = 10, crash = …,
               sgs = nothing, checkpoint_components = (;), stepper = nothing,
               rng = nothing)

Everything a run reports, behind one call per step, [`monitor!`](@ref):

* the **diagnostics** `diagnostics` (default:
  [`collect_diagnostics`](@ref Aeolus.RunLog.collect_diagnostics)`(sim,
  components)`) written to the CSV `csv` (default: the run folder's
  [`diagnostics_path`](@ref) when `run` is a [`RunFolder`](@ref), else
  no file), the costly ones every `cadence` simulated seconds;
* the **progress line** towards `t_end` (none when `t_end = nothing`), at
  most every `progress_every` wall seconds;
* the **blowup guard** ([`BlowupGuard`](@ref)): a check every
  `check_every` steps, crash checkpoints in `crash` (default: the run
  folder's `crash/`, else none) of the state with `sgs`,
  `checkpoint_components`, `stepper` and `rng`, as [`checkpoint!`](@ref)
  takes them.

`close(m)` flushes and closes the CSV. [`with_monitor`](@ref) does that
and feeds the warnings to the progress line.
"""
mutable struct RunMonitor
    writer::Union{Nothing,DiagnosticsWriter}
    progress::Union{Nothing,ProgressLine}
    guard::BlowupGuard
    steps::Int
    step::Int
    dt::Float64
    limiter::String
end

function RunMonitor(sim; run::Union{Nothing,RunFolder} = nothing, components = (;),
                    diagnostics::Union{Nothing,AbstractVector{Diagnostic}} = nothing,
                    t_end::Union{Nothing,Real} = nothing,
                    t_start::Real = hasproperty(sim, :t) ? sim.t : 0.0,
                    cadence::Real = 60.0, progress_every::Real = 10.0,
                    csv::Union{Nothing,AbstractString} = run === nothing ? nothing : diagnostics_path(run),
                    check_every::Integer = 10,
                    crash::Union{Nothing,AbstractString} = run === nothing ? nothing : crash_dir(run),
                    sgs = nothing, checkpoint_components = (;), stepper = nothing, rng = nothing)
    diags = diagnostics === nothing ? collect_diagnostics(sim, components) : diagnostics
    writer = csv === nothing ? nothing : DiagnosticsWriter(csv, diags; cadence = cadence)
    prog = t_end === nothing ? nothing :
           ProgressLine(t_end; t_start = t_start, every = progress_every)
    guard = BlowupGuard(sim; every_steps = check_every, dir = crash, sgs = sgs,
                        components = checkpoint_components, stepper = stepper, rng = rng,
                        diagnostics = writer)
    return RunMonitor(writer, prog, guard, 0, 0, NaN, "")
end

"""
    monitor!(m::RunMonitor, sim; step, dt = NaN, limiter = "", retries = 0) -> m

After step `step` (which took `dt`, limited by `limiter`, with `retries`
rejected attempts): the blowup check (on its cadence; a blowup throws a
[`BlowupError`](@ref) after the diagnosis and the crash checkpoint), a
diagnostics row, and the progress line when due.
"""
function monitor!(m::RunMonitor, sim; step::Integer, dt::Real = NaN, limiter = "",
                  retries::Integer = 0)
    check_blowup!(m.guard, sim; step)
    _report!(m, sim; step, dt, limiter, retries)
    return m
end

# the row and the progress line of a step
function _report!(m::RunMonitor, sim; step, dt, limiter, retries, force = false)
    t = hasproperty(sim, :t) ? Float64(sim.t) : NaN
    lim = limiter === nothing ? "" : string(limiter)
    m.writer === nothing || record_diagnostics!(m.writer, t; step, dt, limiter = lim, retries)
    m.progress === nothing ||
        progress!(m.progress, t; step, dt, limiter = isempty(lim) ? nothing : lim, force)
    m.steps += 1; m.step = step; m.dt = dt; m.limiter = lim
    return m
end

"""
    monitored_step!(f, m::RunMonitor, sim; step, limiter = "") -> f()

One step `f()` under the monitor: run by [`guarded_step!`](@ref) (a
step that throws is a blowup), then [`monitor!`](@ref). When `f` returns
`step_adaptive!`'s `(sim, dt, retries)`, the dt and the retries go into
the row and the progress line.

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
    result = guarded_step!(f, m.guard, sim; step)
    dt, retries = NaN, 0
    if result isa Tuple && length(result) == 3 && result[2] isa Real && result[3] isa Integer
        dt, retries = Float64(result[2]), Int(result[3])
    end
    # guarded_step! has already run the cadence check of this step
    _report!(m, sim; step, dt, limiter, retries)
    return result
end

Base.close(m::RunMonitor) = (m.writer === nothing || close(m.writer); nothing)

"""
    with_monitor(f, sim; kwargs...) -> f(m)

Make a [`RunMonitor`](@ref)`(sim; kwargs...)`, run `f(m)` with the
monitor's progress line noting every warning (a
[`WarningTap`](@ref Aeolus.RunLog.WarningTap) around the current
logger), emit a final progress line, and close the CSV, also when `f`
throws.
"""
function with_monitor(f, sim; kwargs...)
    m = RunMonitor(sim; kwargs...)
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
