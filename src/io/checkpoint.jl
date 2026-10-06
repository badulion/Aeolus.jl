# =========================================================================
# Full-state checkpoint / restart (#245). Included into `module IO`.
# =========================================================================
#
# A checkpoint is a versioned, serialised dictionary holding everything the
# next step of a run reads:
#
#   * the simulation's prognostic fields with their halos (u, v, w, p, θ,
#     qv) and the last eddy coefficients (νt, κt), plus the clock;
#   * the SGS closure's memory, through the closure contract's
#     `sgs_state` / `restore_sgs_state!` (#167), whose snapshots are
#     self-contained and so serialisable;
#   * any other stateful component of the run — wall models with input
#     filters or TBLE columns, a dynamic roughness, a case harness, a
#     driver's accumulators — through `checkpoint_state` /
#     `restore_checkpoint_state!`, the generic pair below;
#   * the stepper state (`StepperState`: step count, last dt, dt history,
#     retries) and RNG state;
#   * provenance: the format version, Julia version, the FFTW planning
#     policy (planner, threads, wisdom loaded, bitwise reproducible).
#
# A restart from step N then reproduces the continuous run bit for bit
# (test/io_checkpoint.jl). Unversioned snapshots (`write_snapshot`, and
# `checkpoint!` before this format) are refused by `restart!`.

using Random: AbstractRNG, TaskLocalRNG
using ..Operators: AbstractSGSClosure, sgs_state, restore_sgs_state!, sgs_closure,
                   fftw_policy, set_fftw_policy!, load_fftw_wisdom!, save_fftw_wisdom

"Format tag stored under `:format` in every checkpoint."
const CHECKPOINT_FORMAT = "Aeolus.checkpoint"

"""
    CHECKPOINT_VERSION

The checkpoint format version this build writes (and the newest it reads).
Version 1 (#245): full state — fields with halos, closure memory, named
components, stepper, RNG, FFTW provenance. Files without a version (the
pre-#245 `checkpoint!` / `write_snapshot` output) are refused by
[`restart!`](@ref).
"""
const CHECKPOINT_VERSION = 1

# -------------------------------------------------------------------------
# The component generic
# -------------------------------------------------------------------------

"""
    checkpoint_state(x) -> state
    restore_checkpoint_state!(x, state) -> x

The generic pair through which a stateful component of a run is
checkpointed: `checkpoint_state` returns a **self-contained, serialisable**
copy of everything of `x` that the run's next step reads (plain arrays,
numbers, tuples, named tuples, dicts of them — nothing aliased to `x`), and
`restore_checkpoint_state!` copies such a state back into an `x` built the
same way, in this or another process.

Methods shipped:

| component | state |
| :-------- | :---- |
| `nothing` | `nothing` |
| any [`AbstractSGSClosure`](@ref Aeolus.Operators.AbstractSGSClosure) | a deep copy of [`sgs_state`](@ref Aeolus.Operators.sgs_state) (the closure contract, #167) |
| `AbstractRNG` (including the task-local default) | `copy(rng)` |
| [`StepperState`](@ref) | its fields |
| `Field`, `Array` | a copy of the data |
| `Vector`, `Dict`, `Base.RefValue` (a driver's accumulators) | a deep copy of the contents |
| `NamedTuple`, `Tuple` | element-wise |
| `WallModels`: `MatchingInput`, `WallModel`, `TBLEWallModel`, `DynamicRoughness`, `WallBoundedCase` | see `src/wall_models/checkpoint.jl` |

Anything else throws a `MethodError`-like `ArgumentError` naming the type:
add the two methods for it (for a mutable struct,
[`fieldwise_checkpoint_state`](@ref) does the work). The **Berlin case**'s
own state (soil, buildings, tracers, surface bus, radiation) is not covered
yet (#269 / #134): its hook is exactly this pair, one method per coupling
state, collected as the `components` of [`checkpoint!`](@ref).
"""
checkpoint_state(x) =
    throw(ArgumentError("checkpoint_state: no method for $(typeof(x)); add " *
                        "`Aeolus.IO.checkpoint_state(::$(nameof(typeof(x))))` and " *
                        "`restore_checkpoint_state!` (see the IO API page)"))

"See [`checkpoint_state`](@ref)."
restore_checkpoint_state!(x, s) =
    throw(ArgumentError("restore_checkpoint_state!: no method for $(typeof(x))"))

checkpoint_state(::Nothing) = nothing
restore_checkpoint_state!(::Nothing, s) = nothing

checkpoint_state(c::AbstractSGSClosure) = deepcopy(sgs_state(c))
restore_checkpoint_state!(c::AbstractSGSClosure, s) = restore_sgs_state!(c, s)

checkpoint_state(rng::AbstractRNG) = copy(rng)
function restore_checkpoint_state!(rng::AbstractRNG, s)
    copy!(rng, s)
    return rng
end

checkpoint_state(f::Field) = copy(f.data)
restore_checkpoint_state!(f::Field, s) = (_copy_same!(f.data, s, "Field"); f)

checkpoint_state(a::Array{<:Number}) = copy(a)
restore_checkpoint_state!(a::Array{<:Number}, s) = (_copy_same!(a, s, "Array"); a)

checkpoint_state(v::Vector) = deepcopy(v)
function restore_checkpoint_state!(v::Vector, s)
    empty!(v)
    append!(v, deepcopy(s))
    return v
end
# a vector of numbers may have grown (a driver's time series): resized
checkpoint_state(v::Vector{<:Number}) = copy(v)
function restore_checkpoint_state!(v::Vector{<:Number}, s)
    resize!(v, length(s))
    copyto!(v, s)
    return v
end

checkpoint_state(d::AbstractDict) = deepcopy(d)
function restore_checkpoint_state!(d::AbstractDict, s)
    empty!(d)
    merge!(d, deepcopy(s))
    return d
end

checkpoint_state(r::Base.RefValue) = deepcopy(r[])
restore_checkpoint_state!(r::Base.RefValue, s) = (r[] = deepcopy(s); r)

checkpoint_state(t::Union{Tuple, NamedTuple}) = map(checkpoint_state, t)
function restore_checkpoint_state!(t::Union{Tuple, NamedTuple}, s)
    length(t) == length(s) ||
        throw(ArgumentError("restore_checkpoint_state!: $(length(t)) components, state has $(length(s))"))
    t isa NamedTuple && keys(t) != keys(s) &&
        throw(ArgumentError("restore_checkpoint_state!: components $(keys(t)), state has $(keys(s))"))
    foreach(restore_checkpoint_state!, t, s)
    return t
end

function _copy_same!(dst::AbstractArray, src, what)
    size(dst) == size(src) ||
        throw(DimensionMismatch("restore_checkpoint_state!: $what of size $(size(dst)), checkpoint has $(size(src))"))
    copyto!(dst, src)
    return dst
end

"""
    fieldwise_checkpoint_state(x, names) -> NamedTuple
    restore_fieldwise!(x, state) -> x

Helpers for a mutable struct's [`checkpoint_state`](@ref) /
[`restore_checkpoint_state!`](@ref): the state is a deep copy of the fields
`names`; restoring copies arrays in place (`copyto!`, sizes must match) and
sets every other field (`setfield!`).

```julia
Aeolus.IO.checkpoint_state(c::MyCase) = fieldwise_checkpoint_state(c, (:nsteps, :flux, :acc))
Aeolus.IO.restore_checkpoint_state!(c::MyCase, s) = restore_fieldwise!(c, s)
```
"""
fieldwise_checkpoint_state(x, names) =
    NamedTuple{Tuple(names)}(map(n -> deepcopy(getfield(x, n)), Tuple(names)))

"See [`fieldwise_checkpoint_state`](@ref)."
function restore_fieldwise!(x, s::NamedTuple)
    for n in keys(s)
        v = getfield(x, n)
        if v isa Array && s[n] isa Array && isbitstype(eltype(v))
            _copy_same!(v, s[n], "field $n")
        else
            setfield!(x, n, deepcopy(s[n]))
        end
    end
    return x
end

# -------------------------------------------------------------------------
# Stepper state
# -------------------------------------------------------------------------

"""
    StepperState(scheme = :fixed; keep = 16)

The time stepper's own state across steps, checkpointed with the run:
`scheme` (a label: `:euler`, `:rk3`, `:imex`, `:adaptive`, …), `step`
(accepted steps so far), `dt_prev` (the last accepted dt, `NaN` before the
first step), `retries` (total rejected tries of [`step_adaptive!`](@ref
Aeolus.TimeIntegration.step_adaptive!)) and `dt_history` (the last `keep`
accepted dts). The adaptive controller's input is `dt_prev`: pass
[`previous_dt`](@ref)`(st)` as `step_adaptive!`'s `dt_prev` and record each
step with [`record_step!`](@ref), so a restarted run limits its first dt's
growth exactly as the continuous run did.

The fixed-dt cores (`step!`, `step_rk3!`, `step_imex!`) keep no state
between steps beyond the simulation and the closure: for them the stepper
state is only the step count.
"""
mutable struct StepperState
    scheme::Symbol
    step::Int
    dt_prev::Float64
    retries::Int
    dt_history::Vector{Float64}
    keep::Int
end

StepperState(scheme::Symbol = :fixed; keep::Integer = 16) =
    StepperState(scheme, 0, NaN, 0, Float64[], Int(keep))

"""
    record_step!(st::StepperState, dt; retries = 0) -> st

Book one accepted step of length `dt` (and the rejected tries before it).
"""
function record_step!(st::StepperState, dt::Real; retries::Integer = 0)
    st.step += 1
    st.dt_prev = Float64(dt)
    st.retries += retries
    push!(st.dt_history, Float64(dt))
    length(st.dt_history) > st.keep && popfirst!(st.dt_history)
    return st
end

"""
    previous_dt(st::StepperState) -> Union{Nothing, Float64}

The last accepted dt (`nothing` before the first step): the `dt_prev`
keyword of [`step_adaptive!`](@ref Aeolus.TimeIntegration.step_adaptive!)
and [`adaptive_dt`](@ref Aeolus.TimeIntegration.adaptive_dt).
"""
previous_dt(st::StepperState) = isnan(st.dt_prev) ? nothing : st.dt_prev

checkpoint_state(st::StepperState) =
    (scheme = st.scheme, step = st.step, dt_prev = st.dt_prev, retries = st.retries,
     dt_history = copy(st.dt_history), keep = st.keep)
function restore_checkpoint_state!(st::StepperState, s)
    st.scheme = s.scheme; st.step = s.step; st.dt_prev = s.dt_prev
    st.retries = s.retries; st.dt_history = copy(s.dt_history); st.keep = s.keep
    return st
end

# -------------------------------------------------------------------------
# The simulation's part
# -------------------------------------------------------------------------

"""
    checkpoint_fields(sim) -> NamedTuple of Fields

The fields of `sim` a checkpoint stores **with their halos**: the
prognostic `u, v, w, p, θ, qv` and the eddy coefficients of the last
evaluation `νt, κt` (a step's CFL estimate or a driver's dt choice may read
them before the closure recomputes them). Tendencies and scratch buffers
are rewritten before they are read in every step and are not stored.
"""
checkpoint_fields(sim::BoussinesqSimulation) =
    (u = sim.u, v = sim.v, w = sim.w, p = sim.p, θ = sim.θ, qv = sim.qv,
     νt = sim.νt, κt = sim.κt)

# dz: the uniform spacing, or a StretchedGrid's cell thicknesses (it has no
# `dz` field; a stretched run's checkpoint failed on it before CFG-b, #357)
_mesh_record(m) = (Nx = m.Nx, Ny = m.Ny, Nz = m.Nz, Hx = m.Hx, Hy = m.Hy, Hz = m.Hz,
                   Lx = m.Lx, Ly = m.Ly, Lz = m.Lz, dx = m.dx, dy = m.dy,
                   dz = hasfield(typeof(m), :dz) ? m.dz : copy(m.dzc))

# the run's scalar parameters (numbers and flags of the simulation struct)
_parameters(sim) = Dict{Symbol,Any}(n => getfield(sim, n) for n in fieldnames(typeof(sim))
                                     if n !== :t && isbits(getfield(sim, n)))

# -------------------------------------------------------------------------
# checkpoint! / restart!
# -------------------------------------------------------------------------

_component_dict(::Nothing) = Dict{Symbol,Any}()
_component_dict(nt::NamedTuple) = Dict{Symbol,Any}(pairs(nt))
_component_dict(d::AbstractDict) = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d)

"""
    checkpoint!(sim, path; sgs = nothing, components = (;), stepper = nothing,
                rng = nothing, extra = Dict()) -> path

Write a **full-state, versioned** checkpoint of a run to `path` (format
[`CHECKPOINT_VERSION`](@ref)), from which [`restart!`](@ref) continues the
run bit for bit:

* the fields of [`checkpoint_fields`](@ref) with their halos, and `sim.t`;
* `sgs`: the closure the run steps with (an `AbstractSGSClosure`, or a
  legacy `f(νt, u, v, w)` function, stateless) — its memory via
  [`checkpoint_state`](@ref), i.e. the closure contract's `sgs_state`;
* `components`: a `NamedTuple` (or `Dict`) of any other stateful objects —
  wall models, input filters, a `WallBoundedCase`, a driver's accumulators —
  each through [`checkpoint_state`](@ref); restored by name;
* `stepper`: a [`StepperState`](@ref); `rng`: an `AbstractRNG` the run
  draws from (e.g. `Random.default_rng()`);
* `extra`: free-form serialisable metadata (a driver's series, tags);
* provenance: format and version, Julia version, the mesh, the run's
  scalar parameters, and the FFTW policy ([`fftw_policy`](@ref
  Aeolus.Operators.fftw_policy): planner, threads, wisdom loaded,
  bitwise reproducible).

The file is written to `path * ".tmp"` and renamed, so an interrupted
write never leaves a truncated checkpoint under `path`. The format is
Julia's `Serialization`: portable across machines with the same Julia
version (recorded; a mismatch warns on restart).
"""
function checkpoint!(sim::BoussinesqSimulation, path::AbstractString;
                     sgs = nothing, components = (;), stepper = nothing,
                     rng = nothing, extra::AbstractDict = Dict{Symbol,Any}())
    return write_checkpoint(path, checkpoint_payload(sim; sgs, components, stepper, rng, extra))
end

"""
    checkpoint_payload(sim; sgs = nothing, components = (;), stepper = nothing,
                       rng = nothing, extra = Dict(), buffers = nothing) -> Dict{Symbol,Any}

The in-memory checkpoint that [`checkpoint!`](@ref) writes: everything
it lists, self-contained (the field data and every component state are
copies, nothing is aliased to the run). [`write_checkpoint`](@ref) puts
it on disk. The blowup guard (#335) holds the last good state of a run
this way and writes it as the crash checkpoint. `buffers`, the `:fields`
of an earlier payload of the same simulation, are reused for the field
copies (no allocation); they then belong to the new payload.
"""
function checkpoint_payload(sim::BoussinesqSimulation;
                            sgs = nothing, components = (;), stepper = nothing,
                            rng = nothing, extra::AbstractDict = Dict{Symbol,Any}(),
                            buffers::Union{Nothing,AbstractDict} = nothing)
    closure = sgs === nothing ? nothing : sgs_closure(sgs)
    comps = _component_dict(components)
    fields = Dict{Symbol,Any}()
    for (k, f) in pairs(checkpoint_fields(sim))
        b = buffers === nothing ? nothing : get(buffers, k, nothing)
        fields[k] = b isa Array && size(b) == size(f.data) && eltype(b) == eltype(f.data) ?
                    copyto!(b, f.data) : copy(f.data)
    end
    return Dict{Symbol,Any}(
        :format         => CHECKPOINT_FORMAT,
        :version        => CHECKPOINT_VERSION,
        :julia_version  => string(VERSION),
        :created        => time(),
        :sim_type       => string(nameof(typeof(sim))),
        :mesh           => _mesh_record(sim.mesh),
        :t              => sim.t,
        :fields         => fields,
        :parameters     => _parameters(sim),
        :closure_type   => closure === nothing ? "" : string(nameof(typeof(closure))),
        :closure        => checkpoint_state(closure),
        :components     => Dict{Symbol,Any}(k => (type = string(nameof(typeof(v))),
                                                  state = checkpoint_state(v)) for (k, v) in comps),
        :stepper        => stepper === nothing ? nothing : checkpoint_state(stepper),
        :rng            => rng === nothing ? nothing : checkpoint_state(rng),
        :fftw           => fftw_policy(),
        :extra          => Dict{Symbol,Any}(Symbol(k) => v for (k, v) in extra),
    )
end

"""
    write_checkpoint(path, payload) -> path

Write a [`checkpoint_payload`](@ref) to `path`, through `path * ".tmp"`
and a rename, so an interrupted write never leaves a truncated
checkpoint under `path`.
"""
function write_checkpoint(path::AbstractString, payload::AbstractDict)
    mkpath(dirname(abspath(path)))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, payload)
    end
    mv(tmp, path; force = true)
    return path
end

"""
    read_checkpoint(path) -> Dict{Symbol,Any}

Read and validate a checkpoint written by [`checkpoint!`](@ref): returns
its payload (keys `:version`, `:t`, `:fields`, `:closure`, `:components`,
`:stepper`, `:rng`, `:fftw`, `:extra`, …). Throws an `ArgumentError` for
an **unversioned** file — a [`write_snapshot`](@ref) snapshot or a
`checkpoint!` file from before format version 1 — which holds no closure
memory or stepper state and so cannot restart a run, and for a file of a
newer format than this build reads.
"""
function read_checkpoint(path::AbstractString)
    d = open(deserialize, path, "r")
    d isa AbstractDict && get(d, :format, nothing) == CHECKPOINT_FORMAT ||
        throw(ArgumentError(
            "restart!: $path is an unversioned snapshot (written by write_snapshot, or by " *
            "checkpoint! before checkpoint format version 1). It holds only u, v, w, p, θ, qv " *
            "and t, without the SGS closure memory, wall-model state or stepper state, so a " *
            "run cannot be restarted from it. Read its fields with read_snapshot, or rerun " *
            "to write a versioned checkpoint."))
    v = d[:version]
    v <= CHECKPOINT_VERSION ||
        throw(ArgumentError("restart!: $path has checkpoint format version $v; this build " *
                            "reads up to version $CHECKPOINT_VERSION"))
    return d
end

"""
    restart!(sim, path; sgs = nothing, components = (;), stepper = nothing,
             rng = nothing, allow_mismatch = false) -> sim

Restore a run from a checkpoint written by [`checkpoint!`](@ref): `sim`
must be built exactly as the checkpointed one (same mesh — checked —,
boundary conditions, physics), and `sgs`, `components`, `stepper`, `rng` must be the
same kinds of objects the checkpoint was written with (the closure's type
and the component names are checked). Copies the fields with their halos,
the clock, the closure memory and every component's state back; the next
step then equals the continuous run's bit for bit (given the same FFT
plans: see [`fftw_policy`](@ref Aeolus.Operators.fftw_policy)).

**Run parameters.** The scalar parameters of `sim` (`ν`, `θ₀`, `g_accel`,
`f_coriolis`, `C_s`, `Pr_t`, the surface fluxes, the sponge, the enable
flags) are compared with the checkpoint's. A difference throws an
`ArgumentError` listing every differing parameter with both values: a
restart with other parameters is a different experiment, not a
continuation. Pass `allow_mismatch = true` to restart anyway (a deliberate
change, e.g. a new sponge); the differences are then logged with `@warn`.

Refuses unversioned snapshots (see [`read_checkpoint`](@ref)). To read the
checkpoint's `extra` or provenance, use [`read_checkpoint`](@ref).
"""
function restart!(sim::BoussinesqSimulation, path::AbstractString;
                  sgs = nothing, components = (;), stepper = nothing, rng = nothing,
                  allow_mismatch::Bool = false)
    restore_checkpoint!(sim, read_checkpoint(path); sgs, components, stepper, rng, path,
                        allow_mismatch)
    return sim
end

# The restoring half of restart!, on an already read payload.
function restore_checkpoint!(sim::BoussinesqSimulation, d::AbstractDict;
                             sgs = nothing, components = (;), stepper = nothing,
                             rng = nothing, path = "checkpoint", allow_mismatch::Bool = false)
    d[:julia_version] == string(VERSION) ||
        @warn("restart!: $path was written by Julia $(d[:julia_version]), this is $(VERSION)",
              subsystem = :checkpoint, domain = current_domain(), path,
              written_by = d[:julia_version], running = string(VERSION))
    m = _mesh_record(sim.mesh); fm = d[:mesh]
    (fm.Nx, fm.Ny, fm.Nz, fm.Hx, fm.Hy, fm.Hz) == (m.Nx, m.Ny, m.Nz, m.Hx, m.Hy, m.Hz) ||
        throw(ArgumentError("restart!: mesh mismatch: file $(fm.Nx)×$(fm.Ny)×$(fm.Nz) (halo " *
                            "$((fm.Hx, fm.Hy, fm.Hz))) vs. sim $(m.Nx)×$(m.Ny)×$(m.Nz) (halo $((m.Hx, m.Hy, m.Hz)))"))
    (fm.Lx, fm.Ly, fm.Lz) == (m.Lx, m.Ly, m.Lz) ||
        throw(ArgumentError("restart!: domain mismatch: file $((fm.Lx, fm.Ly, fm.Lz)) vs. sim $((m.Lx, m.Ly, m.Lz))"))
    d[:sim_type] == string(nameof(typeof(sim))) ||
        throw(ArgumentError("restart!: checkpoint of a $(d[:sim_type]), restoring into a $(nameof(typeof(sim)))"))
    p = _parameters(sim)
    diff = sort!([k for (k, v) in d[:parameters] if haskey(p, k) && !isequal(p[k], v)])
    if !isempty(diff)
        list = join(("$k = $(p[k]) (checkpoint: $(d[:parameters][k]))" for k in diff), ", ")
        allow_mismatch ||
            throw(ArgumentError("restart!: run parameters differ from the checkpointed run in " *
                                "$path: $list. Build the simulation with the checkpoint's " *
                                "parameters, or pass `allow_mismatch = true` to restart anyway"))
        @warn("restart!: restarting with allow_mismatch = true; run parameters differ from " *
              "the checkpointed run in $path: $list",
              subsystem = :checkpoint, domain = current_domain(), path,
              parameters = Tuple(diff))
    end

    closure = sgs === nothing ? nothing : sgs_closure(sgs)
    ctype = closure === nothing ? "" : string(nameof(typeof(closure)))
    ctype == d[:closure_type] ||
        throw(ArgumentError("restart!: the checkpoint's closure is $(repr(d[:closure_type])), " *
                            "restoring with $(repr(ctype)) (pass the same `sgs`)"))
    comps = _component_dict(components)
    fc = d[:components]
    Set(keys(comps)) == Set(keys(fc)) ||
        throw(ArgumentError("restart!: the checkpoint has components $(sort!(collect(keys(fc)))), " *
                            "restoring $(sort!(collect(keys(comps))))"))
    for (k, v) in comps
        string(nameof(typeof(v))) == fc[k].type ||
            throw(ArgumentError("restart!: component $k is a $(fc[k].type) in the checkpoint, " *
                                "a $(nameof(typeof(v))) here"))
    end
    (stepper === nothing) == (d[:stepper] === nothing) ||
        throw(ArgumentError("restart!: pass `stepper` exactly when the checkpoint has one " *
                            "(it has $(d[:stepper] === nothing ? "none" : "one"))"))
    (rng === nothing) == (d[:rng] === nothing) ||
        throw(ArgumentError("restart!: pass `rng` exactly when the checkpoint has one"))

    for (k, f) in pairs(checkpoint_fields(sim))
        _copy_same!(f.data, d[:fields][k], "field $k")
    end
    sim.t = d[:t]
    closure === nothing || restore_checkpoint_state!(closure, d[:closure])
    for (k, v) in comps
        restore_checkpoint_state!(v, fc[k].state)
    end
    stepper === nothing || restore_checkpoint_state!(stepper, d[:stepper])
    rng === nothing || restore_checkpoint_state!(rng, d[:rng])
    return d
end

# -------------------------------------------------------------------------
# Periodic checkpoints with automatic resume (the long drivers' helper)
# -------------------------------------------------------------------------

"""
    CheckpointSchedule(dir; every_steps = 0, every_seconds = Inf, keep = 2,
                       prefix = "checkpoint", wisdom = true, verbose = true)

Periodic checkpoints of a long run in the folder `dir`, with automatic
resume — the one helper the long drivers (`dns256_generate.jl`, the
relaminarisation and SBL drivers) hook into:

1. **Construct it first**, before the simulation: it creates `dir` and, with
   `wisdom = true`, imports the FFTW wisdom `dir/fftw_wisdom` if a previous
   (interrupted) run left one, so the new process makes the same FFT plans
   ([`load_fftw_wisdom!`](@ref Aeolus.Operators.load_fftw_wisdom!)).
2. Build the run, then call [`resume!`](@ref)`(ck, sim; …)`: it saves the
   FFTW wisdom at run start (if `dir` has none yet) and, if `dir` holds a
   checkpoint, restores the newest one ([`restart!`](@ref)) and returns
   its payload (`nothing` on a fresh start).
3. After every step, [`maybe_checkpoint!`](@ref)`(ck, sim; step, …)`
   writes `dir/<prefix>_<step>.jls` when `every_steps` steps or
   `every_seconds` of wall clock have passed since the last one, and
   deletes all but the newest `keep`.

The policy and the FFTW provenance are logged when the run starts or
resumes: at `Info` with `verbose = true`, else at `Debug` (only a run log
shows it); each checkpoint written is a `Debug` event (`subsystem =
:checkpoint`, see the Logging page). The `[checkpoint]` table of a TOML namelist builds one
with [`checkpoint_schedule`](@ref).
"""
mutable struct CheckpointSchedule
    dir::String
    prefix::String
    every_steps::Int
    every_seconds::Float64
    keep::Int
    wisdom::Bool
    verbose::Bool
    last_step::Int
    last_wall::Float64
    resumed_from::String
end

function CheckpointSchedule(dir::AbstractString; every_steps::Integer = 0,
                            every_seconds::Real = Inf, keep::Integer = 2,
                            prefix::AbstractString = "checkpoint", wisdom::Bool = true,
                            verbose::Bool = true)
    every_steps >= 0 || throw(ArgumentError("every_steps must be ≥ 0 (0: off), got $every_steps"))
    every_seconds > 0 || throw(ArgumentError("every_seconds must be > 0 (Inf: off), got $every_seconds"))
    keep >= 1 || throw(ArgumentError("keep must be ≥ 1, got $keep"))
    mkpath(dir)
    ck = CheckpointSchedule(abspath(dir), String(prefix), Int(every_steps), Float64(every_seconds),
                            Int(keep), wisdom, verbose, 0, time(), "")
    wisdom && load_fftw_wisdom!(wisdom_path(ck))
    return ck
end

# checkpoint_schedule(cfg) builds one from the [checkpoint] section of a
# run configuration: src/config/common.jl (CFG-a, #356).

"`dir/fftw_wisdom`: where a [`CheckpointSchedule`](@ref) keeps the run's FFTW wisdom."
wisdom_path(ck::CheckpointSchedule) = joinpath(ck.dir, "fftw_wisdom")

"`dir/<prefix>_<step, 9 digits>.jls`: the checkpoint of step `step`."
checkpoint_path(ck::CheckpointSchedule, step::Integer) =
    joinpath(ck.dir, string(ck.prefix, "_", lpad(step, 9, '0'), ".jls"))

function _checkpoint_files(ck::CheckpointSchedule)
    isdir(ck.dir) || return Pair{Int,String}[]
    re = Regex("^" * replace(ck.prefix, r"([\\^$.|?*+()\[\]{}])" => s"\\\1") * "_(\\d+)\\.jls\$")
    out = Pair{Int,String}[]
    for f in readdir(ck.dir)
        m = match(re, f)
        m === nothing || push!(out, parse(Int, m[1]) => joinpath(ck.dir, f))
    end
    return sort!(out; by = first)
end

"""
    latest_checkpoint(ck::CheckpointSchedule) -> Union{Nothing, String}

The newest checkpoint of the schedule's folder (by step number), or
`nothing`.
"""
function latest_checkpoint(ck::CheckpointSchedule)
    fs = _checkpoint_files(ck)
    return isempty(fs) ? nothing : last(fs[end])
end

_fftw_line() = (p = fftw_policy();
                "FFTW planner = :$(p.planner), threads = $(p.threads), wisdom loaded = $(p.wisdom_loaded), " *
                "bitwise reproducible = $(p.bitwise_reproducible)")

"""
    resume!(ck::CheckpointSchedule, sim; sgs, components, stepper, rng,
            allow_mismatch = false) -> payload or nothing

Run start for a [`CheckpointSchedule`](@ref): write the FFTW wisdom to
`dir/fftw_wisdom` if there is none yet (the plans of `sim` exist now), then
restore the newest checkpoint of `dir` if there is one ([`restart!`](@ref),
same keywords) and return its payload (with `:extra` and `:path`), or
return `nothing` for a fresh start. The schedule's step counter continues
from the checkpoint's `stepper.step` (or its `:step` extra).
"""
function resume!(ck::CheckpointSchedule, sim; sgs = nothing, components = (;),
                 stepper = nothing, rng = nothing, allow_mismatch::Bool = false)
    ck.wisdom && !isfile(wisdom_path(ck)) && save_fftw_wisdom(wisdom_path(ck))
    path = latest_checkpoint(ck)
    if path === nothing
        @logmsg((ck.verbose ? Info : Debug), "checkpoints: fresh start in $(ck.dir); $(_fftw_line())",
                subsystem = :checkpoint, domain = current_domain(), event = :fresh_start,
                dir = ck.dir)
        ck.last_wall = time()
        return nothing
    end
    d = read_checkpoint(path)
    restore_checkpoint!(sim, d; sgs, components, stepper, rng, path, allow_mismatch)
    ck.last_step = stepper === nothing ? get(d[:extra], :step, 0) : stepper.step
    ck.last_wall = time()
    ck.resumed_from = path
    d[:path] = path
    @logmsg((ck.verbose ? Info : Debug),
            "checkpoints: resumed from $(basename(path)) (t = $(d[:t]), step $(ck.last_step)); " *
            "$(_fftw_line()); checkpoint written with planner = :$(d[:fftw].planner), " *
            "bitwise reproducible = $(d[:fftw].bitwise_reproducible)",
            subsystem = :checkpoint, domain = current_domain(), event = :resume, path,
            t = d[:t], step = ck.last_step)
    return d
end

"""
    checkpoint_due(ck::CheckpointSchedule, step) -> Bool

Whether [`maybe_checkpoint!`](@ref) would write at `step`: `every_steps`
steps or `every_seconds` of wall clock since the last checkpoint (or the
run start).
"""
checkpoint_due(ck::CheckpointSchedule, step::Integer) =
    (ck.every_steps > 0 && step - ck.last_step >= ck.every_steps) ||
    (time() - ck.last_wall >= ck.every_seconds)

"""
    maybe_checkpoint!(ck::CheckpointSchedule, sim; step, force = false, sgs, components,
                      stepper, rng, extra) -> Union{Nothing, String}

Write a checkpoint ([`checkpoint!`](@ref), same keywords; `step` is also
stored in `extra`) when [`checkpoint_due`](@ref) (or `force`), keep the
newest `keep` checkpoints of the folder and delete the older ones. Returns
the path written, or `nothing`.
"""
function maybe_checkpoint!(ck::CheckpointSchedule, sim; step::Integer, force::Bool = false,
                           sgs = nothing, components = (;), stepper = nothing, rng = nothing,
                           extra::AbstractDict = Dict{Symbol,Any}())
    (force || checkpoint_due(ck, step)) || return nothing
    ex = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in extra)
    ex[:step] = step
    path = checkpoint!(sim, checkpoint_path(ck, step); sgs, components, stepper, rng, extra = ex)
    ck.last_step = step
    ck.last_wall = time()
    fs = _checkpoint_files(ck)
    removed = String[]
    for (_, f) in fs[1:end-min(ck.keep, length(fs))]
        rm(f; force = true)
        push!(removed, basename(f))
    end
    @debug("checkpoint written", subsystem = :checkpoint, domain = current_domain(),
           event = :write, path, step, t = hasproperty(sim, :t) ? sim.t : nothing,
           removed = Tuple(removed))
    return path
end

"""
    driver_checkpoints(dir; every_steps = 0, every_seconds = 1800, keep = 2,
                       prefix = "checkpoint") -> Union{Nothing, CheckpointSchedule}

The [`CheckpointSchedule`](@ref) of a long example driver: on by default
(every 30 min of wall clock, the newest 2 kept), overridden by the
environment so every driver is controlled the same way:

| variable | effect |
| :------- | :----- |
| `AEOLUS_CHECKPOINT=0` | no checkpoints (returns `nothing`) |
| `AEOLUS_CHECKPOINT_EVERY=n` | every `n` steps |
| `AEOLUS_CHECKPOINT_SECONDS=s` | every `s` s of wall clock (`Inf`: off) |
| `AEOLUS_CHECKPOINT_KEEP=k` | keep the newest `k` |

Call it before building the run (it imports the run folder's FFTW wisdom),
then hand it to the driver's run loop; [`finish_checkpoints!`](@ref)
removes the folder once the driver has stored its result.
"""
function driver_checkpoints(dir::AbstractString; every_steps::Integer = 0,
                            every_seconds::Real = 1800, keep::Integer = 2,
                            prefix::AbstractString = "checkpoint")
    get(ENV, "AEOLUS_CHECKPOINT", "1") == "0" && return nothing
    n = parse(Int, get(ENV, "AEOLUS_CHECKPOINT_EVERY", string(every_steps)))
    sec = parse(Float64, get(ENV, "AEOLUS_CHECKPOINT_SECONDS", string(every_seconds)))
    k = parse(Int, get(ENV, "AEOLUS_CHECKPOINT_KEEP", string(keep)))
    return CheckpointSchedule(dir; every_steps = n, every_seconds = sec, keep = k, prefix = prefix)
end

"""
    finish_checkpoints!(ck) -> nothing

Delete the checkpoints (every prefix) and the FFTW wisdom of the folder of
`ck`, and the folder if it is then empty: call it once the driver has
stored the run's result, so a later run starts fresh. `nothing` is
accepted (checkpointing off).
"""
function finish_checkpoints!(ck::Union{Nothing, CheckpointSchedule})
    ck === nothing && return nothing
    isdir(ck.dir) || return nothing
    for f in readdir(ck.dir)
        (endswith(f, ".jls") || f == "fftw_wisdom") && rm(joinpath(ck.dir, f); force = true)
    end
    isempty(readdir(ck.dir)) && rm(ck.dir)
    return nothing
end

# apply_fftw_config!(cfg), the [fftw] section of a run configuration, is in
# src/config/common.jl (CFG-a, #356).
