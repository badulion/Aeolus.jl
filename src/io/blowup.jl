# Blowup: diagnose, dump, stop (LOG-c, #335; design in epic #293).
# Included into `module IO`, after checkpoint.jl and run_folder.jl.
#
# A `BlowupGuard` checks a run for non-finite values on a step cadence and
# keeps the last state that passed the check as an in-memory checkpoint
# payload. When the run blows up (a non-finite value, or a step that
# throws), it logs which fields went non-finite, where and when, with the
# last diagnostics; writes the crash checkpoint of the last good state;
# and stops the run with a `BlowupError`.
#
# The cost (#514): one fused, threaded sweep over every inspected field
# instead of one serial sweep per field; between due checks one value of
# the pressure (a NaN anywhere reaches all of it in one solve); and under
# an attached monitor nothing the adaptive step has already done itself
# (its post-step sweep, its rollback copy) is done again.

using ..RunLog: DiagnosticsWriter, last_diagnostics
using ..Grids: Grids
using ..Backends: Backends, range_max
using ..Simulations: Simulations, StepRecord, guard_covered

# ---------------------------------------------------------------------------
# the fused finiteness sweep (#514)
# ---------------------------------------------------------------------------

# `x * 0` is 0 for a finite x and NaN otherwise (`Inf * 0` is NaN); summed
# over the arrays at a point, NaN-propagating in `range_max`
@inline _nf_at(i, j, k, a) = @inbounds a[i, j, k] * zero(eltype(a))
@inline _nf_at(i, j, k, a, b, rest...) = _nf_at(i, j, k, a) + _nf_at(i, j, k, b, rest...)

"""
    interior_allfinite(arrays::Tuple, ranges) -> Bool

Whether every value of the same-sized storage `arrays` over the index
`ranges` (a field's interior) is finite: **one** sweep over all of them,
threaded and on the arrays' backend (`Backends.range_max`, a
NaN-propagating maximum of `Σ x·0`), instead of one serial
[`allfinite`](@ref) per array (#514; on 8 fields of 70×70×38 storage
0.079 ms serially against 0.135 ms, 0.05 ms on 4-8 threads).
"""
function interior_allfinite(arrays::Tuple, ranges)
    isempty(arrays) && return true
    r = map(UnitRange{Int}, Tuple(ranges))
    any(isempty, r) && return true
    return range_max(Backends.backend(first(arrays)), _nf_at, r, arrays...) == 0
end

# Whether every field of `fs` (a NamedTuple of Fields) is finite in its
# interior: one fused sweep when they share a mesh, else one per mesh.
function _fields_finite(fs::NamedTuple)
    isempty(fs) && return true
    m = first(fs).mesh
    all(f -> f.mesh === m, fs) && return interior_allfinite(map(f -> f.data, Tuple(fs)), interior(m))
    return all(f -> interior_allfinite((f.data,), interior(f.mesh)), fs)
end

# ---------------------------------------------------------------------------
# where a state went non-finite
# ---------------------------------------------------------------------------

"""
    NonFiniteField

One field of a blown-up state ([`nonfinite_fields`](@ref)): `name`,
`count` (non-finite interior values), `nan` and `inf` (how many of each),
`index` (the interior `(i, j, k)` of the first one in memory order),
`position` (its coordinates in m at the field's own staggered location;
`NaN` where the mesh does not say) and `size` (the interior size).
"""
struct NonFiniteField
    name::Symbol
    count::Int
    nan::Int
    inf::Int
    index::NTuple{3,Int}
    position::NTuple{3,Float64}
    size::NTuple{3,Int}
end

Base.show(io::Base.IO, f::NonFiniteField) =
    print(io, f.name, ": ", f.count, " of ", prod(f.size), " non-finite (", f.nan, " NaN, ",
          f.inf, " Inf), first at ", f.index, " = ",
          map(x -> round(x; sigdigits = 5), f.position), " m")

"""
    blowup_fields(sim) -> NamedTuple of Fields

The fields the blowup check inspects: for a checkpointable core
([`can_checkpoint`](@ref): the Boussinesq and the sound-proof cores) its
[`checkpoint_fields`](@ref) (`u, v, w, p, θ`, `qv` on the Boussinesq core,
`νt, κt`); for any other simulation those of `u, v, w, p, θ, qv` it has.
"""
function blowup_fields(sim)
    can_checkpoint(sim) && return checkpoint_fields(sim)
    names = Tuple(n for n in (:u, :v, :w, :p, :θ, :qv) if hasproperty(sim, n) &&
                  getproperty(sim, n) isa Field)
    return NamedTuple{names}(map(n -> getproperty(sim, n), names))
end

function _node(nodes, i)
    try
        return Float64(nodes[i])
    catch
        return NaN
    end
end
function _nodes(mesh, loc, axis)
    try
        return axis == 1 ? Grids.xnodes(mesh, loc) : axis == 2 ? Grids.ynodes(mesh, loc) :
               Grids.znodes(mesh, loc)
    catch
        return Float64[]
    end
end

"""
    nonfinite_fields(sim) -> Vector{NonFiniteField}

The fields of [`blowup_fields`](@ref)`(sim)` that hold a `NaN` or `Inf`
in their interior, each with the count and the location of the first
one. Empty for a healthy state.
"""
function nonfinite_fields(sim)
    out = NonFiniteField[]
    fs = blowup_fields(sim)
    _fields_finite(fs) && return out          # the healthy case: one fused sweep (#514)
    for (name, f) in pairs(fs)
        a = interior_view(f)
        allfinite(a) && continue
        n = 0; nn = 0; first_I = nothing
        @inbounds for I in CartesianIndices(a)
            x = a[I]
            isfinite(x) && continue
            n += 1; isnan(x) && (nn += 1)
            first_I === nothing && (first_I = I)
        end
        i, j, k = Tuple(first_I)
        loc = location(f)
        pos = (_node(_nodes(f.mesh, loc, 1), i), _node(_nodes(f.mesh, loc, 2), j),
               _node(_nodes(f.mesh, loc, 3), k))
        push!(out, NonFiniteField(name, n, nn, n - nn, (i, j, k), pos, size(a)))
    end
    return out
end

"""
    component_nonfinite_fields(c) -> Vector{NonFiniteField}

The non-finite fields of state that a checkpoint component `c` holds
outside the simulation itself: the child domains of a nest, whose cores
are not `sim` (`Components.ModelNest` extends it, naming each field
`<domain>.<field>`). [`check_blowup!`](@ref) inspects every component the
guard checkpoints, so a run cannot end `completed` with a non-finite
child. Empty by default.
"""
component_nonfinite_fields(c) = NonFiniteField[]

# the simulation's non-finite fields and those of the guard's components
function _guard_nonfinite(g, sim)
    bad = nonfinite_fields(sim)
    for c in values(g.components)
        append!(bad, component_nonfinite_fields(c))
    end
    return bad
end

"""
    allfinite(a) -> Bool

Whether every value of `a` is finite. Branch-free (`x * 0` is `0` for a
finite `x` and `NaN` otherwise, summed with `@simd` along the contiguous
first dimension), so the healthy case, the one the blowup check pays for
every time, vectorises; about 10× faster than `all(isfinite, a)` on an
interior view.
"""
function allfinite(a::AbstractArray{T,3}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds for k in axes(a, 3), j in axes(a, 2)
        @simd for i in axes(a, 1)
            s += a[i, j, k] * zero(T)
        end
    end
    return s == 0
end
allfinite(a) = all(isfinite, a)

# ---------------------------------------------------------------------------
# the error
# ---------------------------------------------------------------------------

"""
    BlowupError <: Exception

The error that stops a blown-up run ([`check_blowup!`](@ref),
[`guarded_step!`](@ref)): `step` and `t` of the failure, `fields` (the
[`NonFiniteField`](@ref)s; empty when the state stayed finite and a step
threw), `cause` (the exception a step threw, or `nothing`), `checkpoint`
(the crash checkpoint's path, or `""` when none was written),
`checkpoint_step` and `checkpoint_t` (the state it holds), and
`diagnostics` (the last diagnostics, see
[`last_diagnostics`](@ref Aeolus.RunLog.last_diagnostics)).
"""
struct BlowupError <: Exception
    step::Int
    t::Float64
    fields::Vector{NonFiniteField}
    cause::Any
    checkpoint::String
    checkpoint_step::Int
    checkpoint_t::Float64
    diagnostics::Vector{Any}
end

function Base.showerror(io::Base.IO, e::BlowupError)
    print(io, "BlowupError: the run blew up at step ", e.step, ", t = ", e.t, " s")
    if !isempty(e.fields)
        print(io, "; non-finite: ", join(("$(f.name) ($(f.count))" for f in e.fields), ", "))
        for f in e.fields
            print(io, "\n  ", f)
        end
    end
    e.cause === nothing || print(io, "\n  cause: ", first(sprint(showerror, e.cause), 500))
    if isempty(e.checkpoint)
        print(io, "\n  no crash checkpoint was written")
    else
        print(io, "\n  crash checkpoint (step ", e.checkpoint_step, ", t = ", e.checkpoint_t,
              " s): ", e.checkpoint, "\n  continue from it with restart!(sim, path; …)")
    end
end

# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

"""
    BlowupGuard(sim; every_steps = 10, dir = nothing, sgs = nothing, components = (;),
                stepper = nothing, rng = nothing, diagnostics = nothing,
                prefix = "crash")

Blowup handling of a run (epic #293: diagnose, dump, stop).

* **Check.** [`check_blowup!`](@ref)`(guard, sim; step)` inspects
  [`blowup_fields`](@ref) for non-finite values every `every_steps`
  steps (and whenever forced). A state that passes is kept as the **last
  good state**: an in-memory [`checkpoint_payload`](@ref) with the
  closure memory `sgs`, the `components`, the `stepper` and the `rng`
  (the same keywords as [`checkpoint!`](@ref)).
* **Diagnose.** On a non-finite value, or when the step run by
  [`guarded_step!`](@ref) throws, it logs an `Error` record (`subsystem
  = :blowup`) naming each non-finite field with its count and the
  position of its first bad value, the step and time, the cause, and the
  last diagnostics of `diagnostics` (a
  [`DiagnosticsWriter`](@ref Aeolus.RunLog.DiagnosticsWriter)).
* **Dump.** It writes the crash checkpoint `dir/<prefix>_<step>.jls` of
  the last good state: the current state if it is still finite (a step
  that threw after `step_adaptive!` restored its start), else the last
  state that passed a check. `restart!` continues from it like from any
  checkpoint. With `dir = nothing` (or for a simulation `checkpoint!`
  does not cover) nothing is written, and the diagnosis says so.
* **Stop.** It throws a [`BlowupError`](@ref) with all of the above.

**Between due checks** (#514) the guard reads one interior value of the
pressure (on a host array): the pressure solve spreads a non-finite
value anywhere in its right-hand side over the whole field, so a blowup
of `u, v, w` (and of `θ` one step later, through the buoyancy) is caught
at the step it reaches the solve, with a full check then. A device array
is not read between checks (one value would be a transfer).

**Cost.** A due check is one fused, threaded sweep over the inspected
fields ([`interior_allfinite`](@ref)) and a copy of them into the
previous snapshot's buffers. **Under an attached run monitor** whose
simulation steps with `step_adaptive!`, a due check does neither (#514):

* the step has verified its end state itself (its post-step
  `step_guard`, [`StepRecord`](@ref Aeolus.Simulations.StepRecord)
  `exit_verified`): the guard sweeps only the inspected fields it does not
  cover ([`guard_covered`](@ref Aeolus.Simulations.guard_covered): `νt,
  κt`, and `qv` of a dry Boussinesq core) — the same fields, at the same
  points and the same step, as a full check;
* the step has saved its start state into its rollback snapshot: the
  guard keeps that snapshot (it hands the simulation the buffers of an
  older one for the next step) as the last good state, with the closure's
  memory, the components, the stepper and the RNG taken at the start of
  that step ([`step_begun`](@ref Aeolus.Simulations.step_begun)). So the
  last good state is that of the step **before** the check (the old
  guard copied the state of the checked step), and it is verified finite
  only if a blowup needs it: the newest of the kept states that is
  finite becomes the crash checkpoint. The guard holds two such
  snapshots.

On 64×64×32 (`benchmarks/bench_run_monitor.jl`): see that file's header
for the measured cost.
"""
mutable struct BlowupGuard
    every_steps::Int
    dir::Union{Nothing,String}
    prefix::String
    sgs::Any
    components::Any
    stepper::Any
    rng::Any
    diagnostics::Union{Nothing,DiagnosticsWriter}
    good::Union{Nothing,Dict{Symbol,Any}}
    good_step::Int
    last_check::Int
    checks::Int
    # an attached monitor's kept rollback snapshots, newest first (#514):
    # (step, payload, snap, verified::Ref{Union{Nothing,Bool}})
    claims::Vector{Any}
    # the payload without its fields, taken at the start of a due step
    pending::Any
end

function BlowupGuard(sim; every_steps::Integer = 10, dir = nothing, sgs = nothing,
                     components = (;), stepper = nothing, rng = nothing,
                     diagnostics::Union{Nothing,DiagnosticsWriter} = nothing,
                     prefix::AbstractString = "crash")
    every_steps >= 1 || throw(ArgumentError("BlowupGuard: every_steps must be ≥ 1, got $every_steps"))
    return BlowupGuard(Int(every_steps), dir === nothing ? nothing : String(dir), String(prefix),
                       sgs, components, stepper, rng, diagnostics, nothing, -1,
                       typemin(Int) ÷ 2, 0, Any[], nothing)
end

"`dir/<prefix>_<step, 9 digits>.jls`: the crash checkpoint of a blowup at `step`."
crash_checkpoint_path(g::BlowupGuard, step::Integer) =
    joinpath(g.dir, string(g.prefix, "_", lpad(step, 9, '0'), ".jls"))

function _payload(g::BlowupGuard, sim, step; with_fields::Bool = true)
    can_checkpoint(sim) || return nothing
    return checkpoint_payload(sim; sgs = g.sgs, components = g.components,
                              stepper = g.stepper, rng = g.rng,
                              extra = Dict{Symbol,Any}(:step => step), with_fields)
end

"""
    snapshot_good!(guard, sim; step) -> guard

Keep the current state of `sim` as the last good state (without
checking it; [`check_blowup!`](@ref) calls this after a passing check).
"""
function snapshot_good!(g::BlowupGuard, sim; step::Integer)
    can_checkpoint(sim) || return g
    # the field copies reuse the buffers of the previous snapshot
    g.good = checkpoint_payload(sim; sgs = g.sgs, components = g.components,
                                stepper = g.stepper, rng = g.rng,
                                extra = Dict{Symbol,Any}(:step => step),
                                buffers = g.good === nothing ? nothing : g.good[:fields])
    g.good_step = step
    return g
end

"""
    check_blowup!(guard, sim; step, force = false) -> Bool

If `every_steps` steps have passed since the last check (or `force`),
inspect `sim` and the state its checkpoint components hold outside it
([`component_nonfinite_fields`](@ref): a nest's child domains): a healthy state becomes the last good state and `false` is
returned; a non-finite one is diagnosed, dumped and stopped (see
[`BlowupGuard`](@ref)): this throws a [`BlowupError`](@ref). Returns
`false` without looking when no check is due.
"""
function check_blowup!(g::BlowupGuard, sim; step::Integer, force::Bool = false)
    (force || step - g.last_check >= g.every_steps || _nonfinite_hint(sim)) || return false
    g.last_check = step; g.checks += 1
    bad = _guard_nonfinite(g, sim)
    isempty(bad) || blowup!(g, sim; step, fields = bad)
    snapshot_good!(g, sim; step)
    return false
end

# One interior value of the pressure, on a host array (#514): the pressure
# solve (FFT, multigrid or CG) spreads a non-finite value anywhere in its
# right-hand side over the whole field, so this is a blowup of the
# velocities at the step it reaches the solve. A device array is not read.
function _nonfinite_hint(sim)
    hasproperty(sim, :p) || return false
    p = getproperty(sim, :p)
    p isa Field || return false
    a = p.data
    a isa Array || return false
    ir, jr, kr = interior(p.mesh)
    (isempty(ir) || isempty(jr) || isempty(kr)) && return false
    @inbounds return !isfinite(a[(first(ir) + last(ir)) ÷ 2, (first(jr) + last(jr)) ÷ 2,
                                 (first(kr) + last(kr)) ÷ 2])
end

"""
    check_blowup!(guard, sim, record::StepRecord; step) -> Bool

The check of an **attached** run monitor at the end of a step of `sim`
(#514): [`check_blowup!`](@ref)`(guard, sim; step)` with what the step
has done itself (see [`BlowupGuard`](@ref)). Between due checks: nothing
when the step verified its end state, else the pressure value. At a due
check after a verified `step_adaptive!` step (with its start taken by
[`step_begun`](@ref Aeolus.Simulations.step_begun)): the sweep of the
fields the step's guard does not cover and the components' fields, and
the step's rollback snapshot kept as the last good state; otherwise the
full check.
"""
function check_blowup!(g::BlowupGuard, sim, rec::StepRecord; step::Integer)
    if step - g.last_check < g.every_steps
        rec.exit_verified && return false
        _nonfinite_hint(sim) || return false
        return check_blowup!(g, sim; step, force = true)
    end
    _can_claim(g, sim, rec, step) || (g.pending = nothing; return check_blowup!(g, sim; step, force = true))
    g.last_check = step; g.checks += 1
    fs = blowup_fields(sim)
    cov = guard_covered(sim)
    rest = NamedTuple{Tuple(k for k in keys(fs) if !(k in cov))}(Tuple(f for (k, f) in pairs(fs) if !(k in cov)))
    bad = _fields_finite(rest) ? NonFiniteField[] : nonfinite_fields(sim)
    for c in values(g.components)
        append!(bad, component_nonfinite_fields(c))
    end
    isempty(bad) || (g.pending = nothing; blowup!(g, sim; step, fields = bad))
    _claim!(g, sim, rec, step)
    return false
end

# A due check may keep the step's rollback snapshot: the step verified its
# end state, it is a checkpointable core stepping with the snapshot of its
# workspace, and the rest of the payload was taken at the step's start.
_can_claim(g::BlowupGuard, sim, rec::StepRecord, step) =
    rec.exit_verified && rec.entry_snap !== nothing && can_checkpoint(sim) &&
    g.pending !== nothing && g.pending.step == step &&
    hasproperty(sim, :workspaces) && getproperty(sim.workspaces, :snapshot) === rec.entry_snap

# Keep the step's start state: the snapshot's buffers become the payload's
# fields; the simulation steps on with the buffers of the oldest kept
# snapshot (none for the first two: it allocates one).
function _claim!(g::BlowupGuard, sim, rec::StepRecord, step)
    snap = rec.entry_snap
    payload = g.pending.payload
    payload[:t] = snap.t
    payload[:fields] = Dict{Symbol,Any}(k => getfield(snap, k) for k in keys(checkpoint_fields(sim)))
    pushfirst!(g.claims, (step = step - 1, payload = payload, snap = snap,
                          verified = Ref{Union{Nothing,Bool}}(nothing)))
    sim.workspaces.snapshot = length(g.claims) > 2 ? pop!(g.claims).snap : nothing
    rec.entry_snap = nothing
    g.pending = nothing
    return g
end

# The last good state for a crash checkpoint: the newest of the copied one
# (`good`) and the kept rollback snapshots that is finite (a kept snapshot
# is swept the first time it is asked). `(payload, step)` or `nothing`.
function _last_good(g::BlowupGuard, sim)
    cands = Any[]
    g.good === nothing || push!(cands, (step = g.good_step, payload = g.good, verified = Ref{Union{Nothing,Bool}}(true)))
    append!(cands, g.claims)
    sort!(cands; by = c -> c.step, rev = true)
    for c in cands
        if c.verified[] === nothing
            c.verified[] = interior_allfinite(Tuple(values(c.payload[:fields])), interior(sim.mesh))
        end
        c.verified[] && return (c.payload, c.step)
    end
    return nothing
end

"""
    guarded_step!(f, guard, sim; step) -> f()

Run one step `f()` under the guard: if it throws (anything but an
`InterruptException`), the run is diagnosed, dumped and stopped with a
[`BlowupError`](@ref) whose `cause` is the exception. Otherwise the
regular [`check_blowup!`](@ref) follows, on its cadence: a non-finite
value is caught at the next due check (`step_adaptive!` already rejects
a non-finite step itself, and throws when its retries run out).

```julia
guard = BlowupGuard(sim; dir = "crash", sgs = closure, stepper = st)
for n in 1:nsteps
    guarded_step!(guard, sim; step = n) do
        step_adaptive!(sim, cfl; sgs = closure)
    end
end
```
"""
function guarded_step!(f, g::BlowupGuard, sim; step::Integer)
    result = try
        f()
    catch err
        err isa InterruptException && rethrow()
        err isa BlowupError && rethrow()
        blowup!(g, sim; step, cause = err, backtrace = catch_backtrace())
    end
    check_blowup!(g, sim; step)
    return result
end

"""
    blowup!(guard, sim; step, fields = <the non-finite fields of sim and its components>, cause = nothing)

Diagnose, dump and stop: log the diagnosis, write the crash checkpoint,
throw the [`BlowupError`](@ref). [`check_blowup!`](@ref) and
[`guarded_step!`](@ref) call it; a driver with its own failure detection
may too.
"""
function blowup!(g::BlowupGuard, sim; step::Integer, fields = _guard_nonfinite(g, sim),
                 cause = nothing, backtrace = nothing)
    t = hasproperty(sim, :t) ? Float64(sim.t) : NaN
    dom = current_domain()
    # 1. the crash checkpoint: the current state if it is still finite, else the last good one
    payload = nothing; cstep = -1
    if isempty(fields) && can_checkpoint(sim)
        payload = _payload(g, sim, step); cstep = step
    elseif (lg = _last_good(g, sim)) !== nothing
        payload, cstep = lg
    end
    path = ""; ct = NaN
    if payload !== nothing && g.dir !== nothing
        payload[:extra][:crash] = true
        payload[:extra][:blowup_step] = step
        payload[:extra][:blowup_t] = t
        payload[:extra][:nonfinite] = [string(f) for f in fields]
        cause === nothing || (payload[:extra][:cause] = first(sprint(showerror, cause), 2000))
        path = crash_checkpoint_path(g, step)
        try
            write_checkpoint(path, payload)
            ct = Float64(payload[:t])
        catch err
            @error("blowup: the crash checkpoint could not be written", subsystem = :blowup,
                   domain = dom, path, exception = err)
            path = ""
        end
    end
    # 2. the diagnosis
    diags = g.diagnostics === nothing ? Any[] : Any[d for d in last_diagnostics(g.diagnostics)]
    g.diagnostics === nothing || flush(g.diagnostics)
    lines = String[]
    for f in fields
        push!(lines, string(f))
    end
    for d in diags
        push!(lines, string("last ", d.name, " = ", d.value, " ", d.units, "  (step ", d.step,
                            ", t = ", d.t, " s)"))
    end
    why = !isempty(fields) ? "non-finite values in " * join((string(f.name) for f in fields), ", ") :
          cause === nothing ? "blowup declared" : "the step threw " * string(nameof(typeof(cause)))
    ckmsg = !isempty(path) ? "crash checkpoint of step $cstep written" :
            g.dir === nothing ? "no crash checkpoint (no directory given)" :
            !can_checkpoint(sim) ? "no crash checkpoint (checkpoint! does not cover a $(nameof(typeof(sim))))" :
            "no crash checkpoint (no good state was recorded)"
    msg = "blowup at step $step, t = $t s: $why; $ckmsg"
    # the cause without its backtrace: with_run logs the BlowupError with one
    exc = cause
    names = Tuple(f.name for f in fields); details = join(lines, '\n')
    if exc === nothing
        @error(msg, subsystem = :blowup, domain = dom, event = :blowup, step, t, reason = why,
               fields = names, checkpoint = path, checkpoint_step = cstep, details)
    else
        @error(msg, subsystem = :blowup, domain = dom, event = :blowup, step, t, reason = why,
               fields = names, checkpoint = path, checkpoint_step = cstep, details,
               exception = exc)
    end
    throw(BlowupError(step, t, collect(NonFiniteField, fields), cause, path, cstep, ct, diags))
end
