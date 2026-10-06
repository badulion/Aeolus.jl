# The diagnostics registry, the CSV writer and the progress line
# (LOG-c, #335; design in epic #293). Included into `module RunLog`, so it
# depends on nothing but the standard library; the methods for the
# library's components (simulations, closures, wall models, nesting) are in
# src/logging/component_diagnostics.jl, the blowup guard and the run
# monitor in src/io/blowup.jl and src/io/run_monitor.jl.

# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------

"The cadence classes of a [`Diagnostic`](@ref)."
const DIAGNOSTIC_CADENCES = (:step, :costly)

"The reductions of a [`Diagnostic`](@ref)."
const DIAGNOSTIC_REDUCTIONS = (:value, :max, :min, :maxabs, :sum, :mean, :rms)

"""
    Diagnostic(name, units, source; cadence = :costly, reduction = :value,
               description = "")

One scalar diagnostic a component declares through [`diagnostics`](@ref):

* `name`: letters, digits, `_` and `.` (it is a CSV column and, after
  #310, a NetCDF variable); [`collect_diagnostics`](@ref) prefixes it
  with the component's key (`"sgs.nu_h_max"`);
* `units`: UDUNITS-style text (`"m s-1"`, `"K"`, `"1"` for a count or a
  ratio);
* `source`: a function of no arguments that returns either a number
  (`reduction = :value`, the source did the reduction) or an array that
  the registry reduces;
* `cadence`: `:step` for the cheap quantities, evaluated at every record,
  or `:costly` for those evaluated on the writer's cadence (default every
  60 s simulated; epic #293);
* `reduction`: how an array is reduced to the scalar: `:max`, `:min`,
  `:maxabs`, `:sum`, `:mean` or `:rms` (root mean square), or `:value`.
  The reduction is declared rather than done by the source, so the
  threaded, deterministic reductions of the KernelAbstractions port can
  replace [`reduce_diagnostic`](@ref) without touching any component
  (#336).

```julia
Diagnostic("max_abs_w", "m s-1", () -> interior_view(sim.w); reduction = :maxabs)
```
"""
struct Diagnostic
    name::String
    units::String
    source::Any
    cadence::Symbol
    reduction::Symbol
    description::String
    function Diagnostic(name::AbstractString, units::AbstractString, source;
                        cadence::Symbol = :costly, reduction::Symbol = :value,
                        description::AbstractString = "")
        occursin(r"^[A-Za-z0-9_.]+$", name) ||
            throw(ArgumentError("Diagnostic: name \"$name\" may hold only letters, digits, '_' and '.'"))
        cadence in DIAGNOSTIC_CADENCES ||
            throw(ArgumentError("Diagnostic $name: cadence must be one of $DIAGNOSTIC_CADENCES, got :$cadence"))
        reduction in DIAGNOSTIC_REDUCTIONS ||
            throw(ArgumentError("Diagnostic $name: reduction must be one of $DIAGNOSTIC_REDUCTIONS, got :$reduction"))
        return new(String(name), String(units), source, cadence, reduction, String(description))
    end
end

Base.show(io::Base.IO, d::Diagnostic) =
    print(io, "Diagnostic(\"", d.name, "\" [", d.units, "], ", d.cadence, ", ", d.reduction, ")")

"""
    diagnostics(component, sim) -> Vector{Diagnostic}
    diagnostics(sim) = diagnostics(sim, sim)

The registry's one generic (epic #293): the scalar diagnostics
`component` declares when it runs in `sim`, each a [`Diagnostic`](@ref)
(name, units, cadence class, reduction and a source). Closures, wall
models, nesting, simulations and the Berlin couplings implement it, as
they implement `output_fields`; the default is no diagnostics. The
sources close over `component` and `sim`, so a declaration is made once
and evaluated at every record.

Methods shipped (see the Logging API page for the full table):
`BoussinesqSimulation`, `MassWeightedSimulation`, every
`AbstractSGSClosure` (and `TKEClosure` adds its energy),
`WallModel`, `TBLEWallModel`, `WallBoundedCase`, `NestedRun`.
"""
diagnostics(component, sim) = Diagnostic[]
diagnostics(sim) = diagnostics(sim, sim)

"""
    prefixed(prefix, ds) -> Vector{Diagnostic}

The diagnostics `ds` with their names prefixed by `"\$prefix."` (nothing
when `prefix` is empty). A component made of others (a nested run, a
case harness) declares its parts' diagnostics this way.
"""
function prefixed(prefix, ds)
    p = string(prefix)
    isempty(p) && return collect(Diagnostic, ds)
    return Diagnostic[Diagnostic(p * "." * d.name, d.units, d.source; cadence = d.cadence,
                                 reduction = d.reduction, description = d.description)
                      for d in ds]
end

"""
    collect_diagnostics(sim, components = (;); include_sim = true) -> Vector{Diagnostic}

Every diagnostic of a run: those of `sim` itself (`diagnostics(sim, sim)`,
unprefixed, unless `include_sim = false`) and of each component, given as
a `NamedTuple` (or `Dict`, or pairs) `key => component`, prefixed with
its key: `collect_diagnostics(sim, (; sgs = closure, wall = wm))` has
`ke_mean`, `sgs.nu_h_max`, `wall.ustar_mean`, … A name that occurs twice
is an error.
"""
function collect_diagnostics(sim, components = (;); include_sim::Bool = true)
    out = Diagnostic[]
    include_sim && append!(out, diagnostics(sim, sim))
    for (k, c) in pairs(components)
        append!(out, prefixed(k, diagnostics(c, sim)))
    end
    seen = Set{String}()
    for d in out
        d.name in seen && throw(ArgumentError("collect_diagnostics: diagnostic \"$(d.name)\" is declared twice"))
        push!(seen, d.name)
    end
    return out
end

# ---------------------------------------------------------------------------
# profiles (LOG-e, #337): vertical profiles for diagnostics.nc
# ---------------------------------------------------------------------------

"""
    ProfileDiagnostic(name, units, axis, levels, source; description = "")

One vertical profile a component declares through
[`profile_diagnostics`](@ref), written to `diagnostics.nc` on the costly
cadence (profiles are not written to the CSV):

* `name`: as for a [`Diagnostic`](@ref); scalars and profiles share the
  NetCDF namespace, so a profile's name must differ from every scalar's;
* `units`: UDUNITS-style text;
* `axis`, `levels`: the name and the heights (m) of its vertical
  coordinate, e.g. `"z"` with the cell-centre heights or `"zf"` with
  the `w`-face heights. Profiles with the same levels share one
  NetCDF dimension;
* `source`: a function of no arguments returning the profile, a vector of
  `length(levels)` (the source does the horizontal reduction, typically
  the deterministic `plane_mean!` of the port).

```julia
ProfileDiagnostic("theta", "K", "z", znodes(mesh, CellCenter()),
                  () -> plane_mean!(zeros(Nz), sim.θ))
```
"""
struct ProfileDiagnostic
    name::String
    units::String
    axis::String
    levels::Vector{Float64}
    source::Any
    description::String
    function ProfileDiagnostic(name::AbstractString, units::AbstractString, axis::AbstractString,
                               levels::AbstractVector{<:Real}, source;
                               description::AbstractString = "")
        for (what, n) in (("name", name), ("axis", axis))
            occursin(r"^[A-Za-z0-9_.]+$", n) ||
                throw(ArgumentError("ProfileDiagnostic: $what \"$n\" may hold only letters, digits, '_' and '.'"))
        end
        isempty(levels) && throw(ArgumentError("ProfileDiagnostic $name: no levels"))
        return new(String(name), String(units), String(axis), collect(Float64, levels), source,
                   String(description))
    end
end

Base.show(io::Base.IO, p::ProfileDiagnostic) =
    print(io, "ProfileDiagnostic(\"", p.name, "\" [", p.units, "] on ", p.axis, " (",
          length(p.levels), " levels))")

"""
    profile_diagnostics(component, sim) -> Vector{ProfileDiagnostic}
    profile_diagnostics(sim) = profile_diagnostics(sim, sim)

The vertical profiles `component` declares when it runs in `sim`, each a
[`ProfileDiagnostic`](@ref); the profile counterpart of
[`diagnostics`](@ref), written to `diagnostics.nc` (#337). The default is
none. Methods shipped: `BoussinesqSimulation` (horizontal means of `u`,
`v`, `θ`, `qv`, the variance of `w` and the resolved flux `w'θ'`),
`MassWeightedSimulation`, every `AbstractSGSClosure` (the mean vertical
eddy viscosity; `TKEClosure` adds its energy) and `NestedRun` (both
domains, prefixed).
"""
profile_diagnostics(component, sim) = ProfileDiagnostic[]
profile_diagnostics(sim) = profile_diagnostics(sim, sim)

"""
    prefixed(prefix, ps::AbstractVector{ProfileDiagnostic})

Profiles with their names **and axes** prefixed by `"\$prefix."`: the two
domains of a nested run have different levels, so `parent.z` and
`child.z` are different axes. (`diagnostics.nc` merges axes with equal
levels, so a closure's `sgs.z` is the simulation's `z`.)
"""
function prefixed(prefix, ps::AbstractVector{ProfileDiagnostic})
    p = string(prefix)
    isempty(p) && return collect(ProfileDiagnostic, ps)
    return ProfileDiagnostic[ProfileDiagnostic(p * "." * q.name, q.units, p * "." * q.axis,
                                               q.levels, q.source; description = q.description)
                             for q in ps]
end

"""
    collect_profiles(sim, components = (;); include_sim = true) -> Vector{ProfileDiagnostic}

Every profile of a run, collected as [`collect_diagnostics`](@ref)
collects the scalars: `sim`'s own unprefixed, each component's prefixed
with its key. A name that occurs twice is an error.
"""
function collect_profiles(sim, components = (;); include_sim::Bool = true)
    out = ProfileDiagnostic[]
    include_sim && append!(out, profile_diagnostics(sim, sim))
    for (k, c) in pairs(components)
        append!(out, prefixed(k, profile_diagnostics(c, sim)))
    end
    seen = Set{String}()
    for q in out
        q.name in seen && throw(ArgumentError("collect_profiles: profile \"$(q.name)\" is declared twice"))
        push!(seen, q.name)
    end
    return out
end

"""
    evaluate_profile(p::ProfileDiagnostic) -> Vector{Float64}

The profile `p` now. A source that throws, or returns the wrong number
of levels, gives a profile of `NaN` and a warning (once per profile), so
one broken profile never stops a run.
"""
function evaluate_profile(p::ProfileDiagnostic)
    n = length(p.levels)
    v = try
        p.source()
    catch err
        err isa InterruptException && rethrow()
        @warn("profile $(p.name) could not be evaluated", subsystem = :diagnostics,
              domain = current_domain(), diagnostic = p.name, exception = err,
              maxlog = 1, _id = Symbol("prof_fail_", p.name))
        return fill(NaN, n)
    end
    if length(v) != n
        @warn("profile $(p.name) has $(length(v)) values for $n levels", subsystem = :diagnostics,
              domain = current_domain(), diagnostic = p.name, maxlog = 1,
              _id = Symbol("prof_len_", p.name))
        return fill(NaN, n)
    end
    return collect(Float64, v)
end

# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

"""
    reduce_diagnostic(reduction, A) -> Float64

`A` reduced as declared: `:max`, `:min`, `:maxabs`, `:sum`, `:mean`,
`:rms`. Serial and deterministic (a fixed summation order); a non-finite
element propagates (`NaN` wins over `max`), so a blowup shows in the
diagnostics. The threaded deterministic reductions of the
KernelAbstractions port replace this one function (#336). An empty array
gives `NaN`.
"""
function reduce_diagnostic(r::Symbol, A::AbstractArray)
    isempty(A) && return NaN
    r === :max && return Float64(_nanmax(A))
    r === :min && return Float64(_nanmin(A))
    r === :maxabs && return Float64(_nanmaxabs(A))
    r === :sum && return Float64(sum(A))
    r === :mean && return Float64(sum(A)) / length(A)
    r === :rms && return sqrt(Float64(sum(abs2, A)) / length(A))
    throw(ArgumentError("reduce_diagnostic: unknown reduction :$r"))
end
reduce_diagnostic(r::Symbol, x::Number) = Float64(x)

# max / min that propagate NaN (Base.maximum already does for floats; this
# keeps the semantics explicit and independent of the element type).
function _nanmax(A)
    m = -Inf
    @inbounds for x in A
        isnan(x) && return NaN
        x > m && (m = x)
    end
    return m
end
function _nanmin(A)
    m = Inf
    @inbounds for x in A
        isnan(x) && return NaN
        x < m && (m = x)
    end
    return m
end
function _nanmaxabs(A)
    m = 0.0
    @inbounds for x in A
        isnan(x) && return NaN
        a = abs(x)
        a > m && (m = a)
    end
    return m
end

"""
    evaluate_diagnostic(d::Diagnostic) -> Float64

The value of `d` now: its source, reduced as declared. A source that
throws gives `NaN` and a warning (once per diagnostic name), so one
broken diagnostic never stops a run.
"""
function evaluate_diagnostic(d::Diagnostic)
    v = try
        d.source()
    catch err
        err isa InterruptException && rethrow()
        @warn("diagnostic $(d.name) could not be evaluated", subsystem = :diagnostics,
              domain = current_domain(), diagnostic = d.name, exception = err,
              maxlog = 1, _id = Symbol("diag_fail_", d.name))
        return NaN
    end
    if d.reduction === :value
        v isa Number ||
            throw(ArgumentError("Diagnostic $(d.name): reduction = :value needs a number, the source returned a $(typeof(v))"))
        return Float64(v)
    end
    return reduce_diagnostic(d.reduction, v)
end

# ---------------------------------------------------------------------------
# the CSV writer
# ---------------------------------------------------------------------------

"The fixed leading columns of `diagnostics.csv`: name and units."
const DIAGNOSTICS_CSV_FIXED = (("step", "1"), ("t", "s"), ("wall", "s"), ("dt", "s"),
                               ("dt_limiter", ""), ("retries", "1"))

_csv_header(diags) =
    join(Iterators.flatten(((isempty(u) ? n : "$n [$u]" for (n, u) in DIAGNOSTICS_CSV_FIXED),
                            ("$(d.name) [$(d.units)]" for d in diags))), ',')

"""
    DiagnosticsWriter(path, diags; cadence = 60.0, append = true, flush_every = 20)

The diagnostics time series of a run as CSV (`diagnostics.csv` of the
run folder; `diagnostics.nc` follows with #310). One row per
[`record_diagnostics!`](@ref): the columns `step`, `t [s]`, `wall [s]`
(wall time since the writer opened), `dt [s]`, `dt_limiter`, `retries`,
then one column `name [units]` per [`Diagnostic`](@ref) of `diags`.

The `:step` diagnostics are evaluated in every row; the `:costly` ones
when `cadence` simulated seconds have passed since their last evaluation
(and in the first row), and their cells are empty in between. Numbers are
written in Julia's shortest round-trip form, so a value read back equals
the `Float64` that was computed; a non-finite value is `NaN`, `Inf` or
`-Inf`.

`append = true` continues an existing file with the same header (a
restart of the run); a file with a different header is left alone and
the rows go to `<name>-2.csv` (`-3`, …) instead, with a warning. The
file is flushed every `flush_every` rows, after every costly row, and on
`close`. `w.last` holds the latest value of every diagnostic, `w.last_t`
and `w.last_step` when it was taken.
"""
mutable struct DiagnosticsWriter
    path::String
    io::Base.IO
    diags::Vector{Diagnostic}
    cadence::Float64
    flush_every::Int
    last_costly::Float64
    rows::Int
    unflushed::Int
    wall0::Float64
    last::Dict{String,Float64}
    last_t::Dict{String,Float64}
    last_step::Dict{String,Int}
    buf::IOBuffer
end

function DiagnosticsWriter(path::AbstractString, diags::AbstractVector{Diagnostic};
                           cadence::Real = 60.0, append::Bool = true, flush_every::Integer = 20)
    cadence >= 0 || throw(ArgumentError("DiagnosticsWriter: cadence must be ≥ 0, got $cadence"))
    flush_every >= 1 || throw(ArgumentError("DiagnosticsWriter: flush_every must be ≥ 1"))
    names = Set{String}()
    for d in diags
        d.name in names && throw(ArgumentError("DiagnosticsWriter: diagnostic \"$(d.name)\" twice"))
        d.name in first.(DIAGNOSTICS_CSV_FIXED) &&
            throw(ArgumentError("DiagnosticsWriter: \"$(d.name)\" is a fixed column"))
        push!(names, d.name)
    end
    header = _csv_header(diags)
    p = String(path)
    mkpath(dirname(abspath(p)))
    write_header = true
    if append && isfile(p) && filesize(p) > 0
        old = readline(p)
        if old == header
            write_header = false
        else
            base, ext = splitext(p)
            n = 2
            while isfile("$base-$n$ext") && (filesize("$base-$n$ext") == 0 ||
                                             readline("$base-$n$ext") != header)
                n += 1
            end
            newp = "$base-$n$ext"
            @warn("diagnostics file $p has other columns; this session writes $newp",
                  subsystem = :diagnostics, domain = current_domain(), path = p, new_path = newp)
            p = newp
            write_header = !(isfile(p) && filesize(p) > 0)
        end
    end
    io = open(p, append ? "a" : "w")
    if write_header
        println(io, header)
        flush(io)
    end
    w = DiagnosticsWriter(p, io, collect(diags), Float64(cadence), Int(flush_every), -Inf, 0, 0,
                          time(), Dict{String,Float64}(), Dict{String,Float64}(),
                          Dict{String,Int}(), IOBuffer())
    @debug("diagnostics file opened", subsystem = :diagnostics, domain = current_domain(),
           event = :open, path = p, columns = length(diags), cadence = w.cadence,
           appended = !write_header)
    return w
end

_csv_number(io, x::Real) = isfinite(x) ? print(io, Float64(x)) :
                           print(io, isnan(x) ? "NaN" : x > 0 ? "Inf" : "-Inf")

"""
    costly_due(w::DiagnosticsWriter, t) -> Bool

Whether the next row at simulated time `t` evaluates the `:costly`
diagnostics: the first row, or `cadence` seconds after the last costly
row (with a relative tolerance of 1e-9, so a cadence that is a multiple
of the step is not missed by roundoff).
"""
costly_due(w::DiagnosticsWriter, t::Real) =
    w.last_costly == -Inf || t - w.last_costly >= w.cadence * (1 - 1e-9)

"""
    record_diagnostics!(w::DiagnosticsWriter, t; step, dt = NaN, limiter = "",
                        retries = 0, costly = costly_due(w, t)) -> Bool

Evaluate the diagnostics of `w` at simulated time `t` and append a row
(see [`DiagnosticsWriter`](@ref)). Returns whether the costly diagnostics
were evaluated. A costly row is also a `Debug` record (`subsystem =
:diagnostics`, `event = :record`) with every value, so `run.log` keeps
the series too.
"""
function record_diagnostics!(w::DiagnosticsWriter, t::Real; step::Integer, dt::Real = NaN,
                             limiter = "", retries::Integer = 0,
                             costly::Bool = costly_due(w, t))
    b = w.buf
    print(b, step, ','); _csv_number(b, t); print(b, ',')
    _csv_number(b, round(time() - w.wall0; digits = 3)); print(b, ',')
    _csv_number(b, dt); print(b, ',', limiter === nothing ? "" : limiter, ',', retries)
    tt = Float64(t)
    for d in w.diags
        print(b, ',')
        if d.cadence === :step || costly
            v = evaluate_diagnostic(d)
            _csv_number(b, v)
            w.last[d.name] = v; w.last_t[d.name] = tt; w.last_step[d.name] = step
        end
    end
    println(b)
    write(w.io, take!(b))
    w.rows += 1; w.unflushed += 1
    costly && (w.last_costly = tt)
    if costly || w.unflushed >= w.flush_every
        flush(w.io); w.unflushed = 0
    end
    if costly
        @debug("diagnostics", subsystem = :diagnostics, domain = current_domain(),
               event = :record, step, t = tt, dt,
               values = join(("$(d.name)=$(get(w.last, d.name, NaN))" for d in w.diags), ' '))
    end
    return costly
end

"""
    last_diagnostics(w::DiagnosticsWriter) -> Vector{NamedTuple}

The latest value of every diagnostic of `w`, in column order, as
`(name, units, value, t, step)` (`value = NaN`, `t = NaN`, `step = -1`
for one not evaluated yet). The blowup diagnosis logs it.
"""
last_diagnostics(w::DiagnosticsWriter) =
    [(name = d.name, units = d.units, value = get(w.last, d.name, NaN),
      t = get(w.last_t, d.name, NaN), step = get(w.last_step, d.name, -1)) for d in w.diags]

"Flush and close the file (the writer's stream belongs to it)."
function Base.close(w::DiagnosticsWriter)
    isopen(w.io) && (flush(w.io); close(w.io))
    return nothing
end
Base.flush(w::DiagnosticsWriter) = (isopen(w.io) && flush(w.io); w.unflushed = 0; nothing)

# ---------------------------------------------------------------------------
# the progress line
# ---------------------------------------------------------------------------

"""
    ProgressLine(t_end; t_start = 0.0, every = 10.0, step0 = 0, level = Info)

The progress of a run (epic #293): simulated time, % done, steps/s, ETA,
dt and its binding limit, and the last warning, as one `Info` record
(`subsystem = :progress`) at most every `every` wall-clock seconds, so it
reaches both the console and `run.log`:

    14:03:12.481 INFO  progress[main] t = 2.50 h (41.7 %)  18.3 steps/s  ETA 0:41:07  dt = 0.84 s (advective)  step=10712 …

The rate is that of the last interval; the ETA extrapolates the
simulated time per wall second since the start. `last warning` is the
message of the most recent `Warn` or `Error` record seen by a
[`WarningTap`](@ref) feeding this line (installed by
[`with_progress`](@ref) and by the run monitor), with its age.
"""
mutable struct ProgressLine
    t_start::Float64
    t_end::Float64
    every::Float64
    level::LogLevel
    wall0::Float64
    last_wall::Float64
    last_step::Int
    step0::Int
    lines::Int
    last_warning::String
    last_warning_wall::Float64
end

function ProgressLine(t_end::Real; t_start::Real = 0.0, every::Real = 10.0, step0::Integer = 0,
                      level = Info)
    t_end > t_start || throw(ArgumentError("ProgressLine: t_end = $t_end must exceed t_start = $t_start"))
    every >= 0 || throw(ArgumentError("ProgressLine: every must be ≥ 0"))
    w = time()
    return ProgressLine(Float64(t_start), Float64(t_end), Float64(every), parse_log_level(level),
                        w, w, Int(step0), Int(step0), 0, "", NaN)
end

"`h:mm:ss` of a duration in seconds (`--:--:--` when not finite)."
function format_duration(s::Real)
    (isfinite(s) && s >= 0) || return "--:--:--"
    n = round(Int, s)
    h, r = divrem(n, 3600); m, sec = divrem(r, 60)
    return string(h, ':', lpad(m, 2, '0'), ':', lpad(sec, 2, '0'))
end

"A simulated time as text, in s, min, h or d."
function format_simtime(t::Real)
    a = abs(t)
    a < 120 && return string(round(t; digits = 2), " s")
    a < 7200 && return string(round(t / 60; digits = 2), " min")
    a < 172800 && return string(round(t / 3600; digits = 2), " h")
    return string(round(t / 86400; digits = 2), " d")
end

"""
    note_warning!(p::ProgressLine, message)

Remember `message` as the last warning of the run, for the next progress
line. A [`WarningTap`](@ref) calls it.
"""
function note_warning!(p::ProgressLine, message)
    p.last_warning = first(replace(string(message), '\n' => ' '), 120)
    p.last_warning_wall = time()
    return p
end

"""
    progress!(p::ProgressLine, t; step, dt = NaN, limiter = nothing, force = false) -> Bool

Emit the progress line for simulated time `t` after step `step` if
`every` wall seconds have passed since the last one (or `force`); returns
whether it did. Key-values: `t`, `percent`, `step`, `steps_per_s`,
`eta_s`, `dt`, `limiter`, `last_warning`.
"""
function progress!(p::ProgressLine, t::Real; step::Integer, dt::Real = NaN, limiter = nothing,
                   force::Bool = false)
    now = time()
    (force || now - p.last_wall >= p.every) || return false
    frac = (t - p.t_start) / (p.t_end - p.t_start)
    elapsed = now - p.wall0
    interval = now - p.last_wall
    rate = interval > 0 ? (step - p.last_step) / interval : NaN
    eta = (frac > 0 && elapsed > 0) ? elapsed * (1 - frac) / frac : NaN
    lim = limiter === nothing || limiter == "" ? "" : " ($limiter)"
    msg = string("t = ", format_simtime(t), " (", round(100 * frac; digits = 1), " %)  ",
                 isfinite(rate) ? round(rate; digits = 1) : "--", " steps/s  ETA ",
                 format_duration(eta), "  dt = ", isfinite(dt) ? round(dt; sigdigits = 3) : "--",
                 " s", lim)
    lw = ""
    if !isempty(p.last_warning)
        lw = p.last_warning
        msg *= string("  last warning (", format_duration(now - p.last_warning_wall), " ago): ", lw)
    end
    @logmsg(p.level, msg, subsystem = :progress, domain = current_domain(), t = Float64(t),
            percent = round(100 * frac; digits = 2), step, steps_per_s = rate, eta_s = eta,
            dt = Float64(dt), limiter = limiter === nothing ? "" : string(limiter),
            last_warning = lw)
    p.last_wall = now; p.last_step = step; p.lines += 1
    return true
end

"""
    WarningTap(logger, progress::ProgressLine)

A logger that passes every record on to `logger` unchanged and notes the
message of each `Warn` and `Error` record in `progress`, for its "last
warning". Progress records themselves are not noted.
"""
struct WarningTap{L<:AbstractLogger} <: AbstractLogger
    logger::L
    progress::ProgressLine
end

min_enabled_level(t::WarningTap) = min(min_enabled_level(t.logger), Warn)
shouldlog(t::WarningTap, level, _module, group, id) =
    level >= Warn || shouldlog(t.logger, level, _module, group, id)
catch_exceptions(t::WarningTap) = catch_exceptions(t.logger)

function handle_message(t::WarningTap, level::LogLevel, message, _module, group, id,
                        file, line; kwargs...)
    if level >= Warn && get(kwargs, :subsystem, nothing) !== :progress
        note_warning!(t.progress, message)
    end
    if level >= min_enabled_level(t.logger) && shouldlog(t.logger, level, _module, group, id)
        handle_message(t.logger, level, message, _module, group, id, file, line; kwargs...)
    end
    return nothing
end
Base.close(t::WarningTap) = applicable(close, t.logger) ? close(t.logger) : nothing

"""
    with_progress(f, p::ProgressLine)

Run `f()` with a [`WarningTap`](@ref) around the current logger, so the
warnings `f` emits show as the "last warning" of `p`'s lines.
"""
with_progress(f, p::ProgressLine) = Logging.with_logger(f, WarningTap(Logging.current_logger(), p))
