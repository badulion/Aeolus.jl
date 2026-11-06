# diagnostics.nc (LOG-e, #337; design in epic #293): the diagnostics time
# series of a run as NetCDF-4, with an unlimited time dimension and the
# vertical profiles, beside the CSV of the scalars (src/logging/
# diagnostics.jl). Included into `module IO`, after run_folder.jl; written
# through the package's NetCDF policy, src/NCIO.jl.

using Dates: DateTime
using ..RunLog: Diagnostic, ProfileDiagnostic, evaluate_diagnostic, evaluate_profile,
                DIAGNOSTICS_CSV_FIXED
import ..RunLog: record_diagnostics!, costly_due
using ..NCIO: NCDataset, nc_create, nc_defvar!, nc_append_record!, cf_time_units

"""
    diagnostics_nc_path(rf::RunFolder)

`path/diagnostics.nc`, the run's diagnostics as NetCDF-4 (scalars and
profiles; #337), beside [`diagnostics_path`](@ref) (`diagnostics.csv`).
"""
diagnostics_nc_path(rf::RunFolder) = joinpath(rf.path, "diagnostics.nc")

_nc_flag(b) = Int8(b === true)

"""
    nc_provenance_attributes(rf::RunFolder) -> Dict{String,Any}

The provenance of the current session of `rf` as NetCDF global
attributes (epic #293: the provenance goes into the log header,
`run.toml` and the NetCDF global attributes). Flat names, values that
NetCDF stores (text, `Int64`, `Float64`; flags as `Int8` 0/1):

`run_id`, `case`, `run_folder`, `session`, `created`, `git_commit`,
`git_branch`, `git_dirty`, `julia_version`, `aeolus_version`,
`manifest_fnv1a64`, `hostname`, `julia_threads`, `blas_threads`,
`fftw_planner`, `fftw_threads`, `fftw_wisdom_loaded`,
`bitwise_reproducible`, `command`, `config` and `seeds`, the
configuration and the seeds of the session as TOML text (a restart's
override, else the run's; with `config_hash` and `config_schema_version`
when it is a `RunConfig`), `session_started`, and `session_finished`,
empty while the session runs (or if it died) and set by `close` of the
[`DiagnosticsNCWriter`](@ref). `run.toml` stays the complete record
(every session, every package version).
"""
function nc_provenance_attributes(rf::RunFolder)
    r = rf.record
    s = _current_session(rf)
    g = s["git"]; h = s["host"]; f = s["fftw"]; c = s["command"]
    cmd = join(vcat(String[string(c["program"])], String[string(a) for a in c["args"]]), ' ')
    toml_text(k) = (buf = IOBuffer();     # the session's override, else the run's
                    TOML.print(buf, Dict{String,Any}(k => get(s, k, get(r, k, Dict{String,Any}())));
                               sorted = true);
                    String(take!(buf)))
    a = Dict{String,Any}(
        "run_id" => rf.run_id, "case" => rf.case, "run_folder" => rf.path,
        "session" => Int64(rf.session), "created" => string(get(r, "created", "")),
        "git_commit" => g["available"] ? string(g["commit"]) : "unavailable",
        "git_branch" => string(get(g, "branch", "")), "git_dirty" => _nc_flag(g["dirty"]),
        "julia_version" => string(s["julia"]["version"]),
        "aeolus_version" => string(s["aeolus"]["version"]),
        "manifest_fnv1a64" => string(s["manifest"]["fnv1a64"]),
        "hostname" => string(h["hostname"]), "julia_threads" => Int64(h["julia_threads"]),
        "blas_threads" => Int64(h["blas_threads"]),
        "fftw_planner" => string(f["planner"]), "fftw_threads" => Int64(f["threads"]),
        "fftw_wisdom_loaded" => _nc_flag(f["wisdom_loaded"]),
        "bitwise_reproducible" => _nc_flag(f["bitwise_reproducible"]),
        "command" => cmd, "config" => toml_text("config"), "seeds" => toml_text("seeds"),
        "session_started" => string(get(s, "started", "")), "session_finished" => "")
    haskey(r, "config_hash") && (a["config_hash"] = string(r["config_hash"]))
    haskey(r, "config_schema_version") && (a["config_schema_version"] = r["config_schema_version"])
    return a
end

"""
    DiagnosticsNCWriter(path, diags, profiles = ProfileDiagnostic[];
                        cadence = 60.0, append = true, flush_interval = 5.0,
                        flush_every = 1000, attributes = Dict(), time_origin = nothing)

The diagnostics of a run as NetCDF-4 (`diagnostics.nc` of the run
folder, #337), written by [`record_diagnostics!`](@ref) at the same rows
and on the same costly cadence as the CSV
([`DiagnosticsWriter`](@ref Aeolus.RunLog.DiagnosticsWriter)). The file
has two unlimited dimensions:

* `time`, one record per row: `time` (simulated, s), `step`, `wall`
  (wall seconds since the writer opened), `dt`, `dt_limiter` (text),
  `retries`, and every `:step` [`Diagnostic`](@ref) of `diags`;
* `time_costly`, one record per costly row: `time_costly`, `step_costly`,
  every `:costly` diagnostic and every [`ProfileDiagnostic`](@ref) of
  `profiles`, as `(level, time_costly)`.

So a costly quantity is never padded with fill values between its
evaluations, and a row of the CSV is a record of `time`. Each profile
axis is a coordinate variable (heights in m, `axis = "Z"`); profiles
whose levels are equal share it, whatever their prefix. Every variable
has `units` and `long_name` (the diagnostic's description, or its name),
reductions their CF `cell_methods`. `time_origin = DateTime(…)` gives the
time coordinates CF units `seconds since …` (UTC); otherwise their units
are `s` from the start of the simulation.

`attributes` become global attributes (the run monitor passes
[`nc_provenance_attributes`](@ref) of its run folder), with `title`,
`source` and a `history` line per session.

`append = true` continues an existing file of the same layout (same
variables, units and profile levels): a restart of the run appends its
records, refreshes the global attributes to the new session and adds a
`history` line. A file of another layout is left alone, and the session
writes `<name>-2.nc` (`-3`, …) instead, with a warning. The rows of
`time` are buffered and written as one block at the first row
`flush_interval` wall seconds after the last write, after `flush_every`
rows, when a blowup is thrown out of the run monitor and on `close`;
every write ends with a `sync`, so the file on disk holds every written
row even if the run dies. The rows of `time_costly` (the costly scalars
and the profiles, evaluated when the row is due) are buffered too and
written in the same block (#514). The policy is the one of `run.log`
(epic #293, "flush every few s"): a block write plus `sync` of the HDF5
file measured 2.13 ms, and syncing every 20 rows measured 146 µs per
step amortised, 3.1 % of a 64×64×32 step from the parts, over the 2 %
budget (see the Logging page); writing each costly row at once measured
13.5 ms a row, of which 0.085 ms are the five profiles (#514).

`close` writes the remaining rows and, if the file has a
`session_finished` attribute (the provenance of
[`nc_provenance_attributes`](@ref)), sets it to the current time.
"""
mutable struct DiagnosticsNCWriter
    path::String
    ds::NCDataset
    step_diags::Vector{Diagnostic}
    costly_diags::Vector{Diagnostic}
    profiles::Vector{ProfileDiagnostic}
    cadence::Float64
    flush_interval::Float64
    flush_every::Int
    last_flush::Float64
    last_costly::Float64
    wall0::Float64
    rows::Int
    costly_rows::Int
    b_step::Vector{Int64}
    b_t::Vector{Float64}
    b_wall::Vector{Float64}
    b_dt::Vector{Float64}
    b_lim::Vector{String}
    b_retries::Vector{Int32}
    b_vals::Vector{Vector{Float64}}
    b_costly::Vector{Dict{String,Any}}     # the buffered rows of `time_costly` (#514)
end

"The variables of the `time_costly` record that are not diagnostics."
const DIAGNOSTICS_NC_COSTLY_FIXED = ("time_costly", "step_costly")

# the layout a file must have to be appended to
function _nc_layout(diags, profiles, axes)
    io = IOBuffer()
    for d in diags
        print(io, d.name, '|', d.units, '|', d.cadence, '|', d.reduction, ';')
    end
    for (p, ax) in zip(profiles, axes)
        print(io, p.name, '|', p.units, '|', ax, ';')
    end
    return String(take!(io))
end

# the dimension of each profile: equal levels share one; a name that is
# taken with other levels gets a suffix
function _profile_axes(profiles)
    names = String[]; levels = Vector{Float64}[]
    axes = String[]
    for p in profiles
        k = findfirst(l -> l == p.levels, levels)
        if k === nothing
            n = p.axis; i = 2
            while n in names
                n = "$(p.axis)_$i"; i += 1
            end
            push!(names, n); push!(levels, p.levels)
            push!(axes, n)
        else
            push!(axes, names[k])
        end
    end
    return axes, names, levels
end

const _CELL_METHODS = Dict(:max => "maximum", :min => "minimum", :maxabs => "maximum_absolute_value",
                           :sum => "sum", :mean => "mean", :rms => "root_mean_square")

function _nc_attrs(units, long_name; reduction = :value)
    a = Dict{String,Any}("units" => units, "long_name" => long_name)
    reduction === :value || (a["cell_methods"] = "area: " * _CELL_METHODS[reduction])
    return a
end

function _define_diagnostics_nc!(ds, step_diags, costly_diags, profiles, axes, axis_names,
                                 axis_levels, time_origin)
    ds.dim["time"] = Inf
    ds.dim["time_costly"] = Inf
    for (n, l) in zip(axis_names, axis_levels)
        ds.dim[n] = length(l)
    end
    tunits = time_origin === nothing ? "s" : cf_time_units(time_origin)
    tattr(long) = (a = Dict{String,Any}("units" => tunits, "long_name" => long, "axis" => "T");
                   time_origin === nothing || (a["standard_name"] = "time"; a["calendar"] = "standard");
                   a)
    nc_defvar!(ds, "time", Float64, ("time",); attributes = tattr("simulated time"))
    nc_defvar!(ds, "step", Int64, ("time",); attributes = _nc_attrs("1", "step number"))
    nc_defvar!(ds, "wall", Float64, ("time",);
               attributes = _nc_attrs("s", "wall-clock time since the writer opened"))
    nc_defvar!(ds, "dt", Float64, ("time",); attributes = _nc_attrs("s", "time step"))
    nc_defvar!(ds, "dt_limiter", String, ("time",); deflatelevel = 0,
               attributes = Dict{String,Any}("long_name" => "the constraint that set dt"))
    nc_defvar!(ds, "retries", Int32, ("time",); attributes = _nc_attrs("1", "rejected step attempts"))
    nc_defvar!(ds, "time_costly", Float64, ("time_costly",);
               attributes = tattr("simulated time of the costly diagnostics and profiles"))
    nc_defvar!(ds, "step_costly", Int64, ("time_costly",);
               attributes = _nc_attrs("1", "step number of the costly diagnostics and profiles"))
    for (n, l) in zip(axis_names, axis_levels)
        v = nc_defvar!(ds, n, Float64, (n,);
                       attributes = Dict{String,Any}("units" => "m", "axis" => "Z", "positive" => "up",
                                                     "long_name" => "height of the profile levels"))
        v[:] = l
    end
    for (d, dim) in Iterators.flatten((((d, "time") for d in step_diags),
                                       ((d, "time_costly") for d in costly_diags)))
        nc_defvar!(ds, d.name, Float64, (dim,);
                   attributes = _nc_attrs(d.units, isempty(d.description) ? d.name : d.description;
                                          reduction = d.reduction))
    end
    for (p, ax) in zip(profiles, axes)
        nc_defvar!(ds, p.name, Float64, (ax, "time_costly");
                   attributes = _nc_attrs(p.units, isempty(p.description) ? p.name : p.description))
    end
    return ds
end

function _nc_check_names(diags, profiles, axis_names)
    taken = Set{String}([first.(DIAGNOSTICS_CSV_FIXED)..., DIAGNOSTICS_NC_COSTLY_FIXED...])
    for n in Iterators.flatten(((d.name for d in diags), (p.name for p in profiles), axis_names))
        n in taken && throw(ArgumentError("DiagnosticsNCWriter: the name \"$n\" is used twice " *
                                          "(diagnostic, profile, axis or fixed variable)"))
        push!(taken, n)
    end
end

function DiagnosticsNCWriter(path::AbstractString, diags::AbstractVector{Diagnostic},
                             profiles::AbstractVector{ProfileDiagnostic} = ProfileDiagnostic[];
                             cadence::Real = 60.0, append::Bool = true,
                             flush_interval::Real = 5.0, flush_every::Integer = 1000,
                             attributes::AbstractDict = Dict{String,Any}(),
                             time_origin::Union{Nothing,DateTime} = nothing)
    cadence >= 0 || throw(ArgumentError("DiagnosticsNCWriter: cadence must be ≥ 0, got $cadence"))
    flush_every >= 1 || throw(ArgumentError("DiagnosticsNCWriter: flush_every must be ≥ 1"))
    flush_interval >= 0 || throw(ArgumentError("DiagnosticsNCWriter: flush_interval must be ≥ 0"))
    axes, axis_names, axis_levels = _profile_axes(profiles)
    _nc_check_names(diags, profiles, axis_names)
    step_diags = Diagnostic[d for d in diags if d.cadence === :step]
    costly_diags = Diagnostic[d for d in diags if d.cadence === :costly]
    layout = _nc_layout(diags, profiles, axes) *
             join((string(n, '=', string(fnv1a64(reinterpret(UInt8, l)); base = 16, pad = 16))
                   for (n, l) in zip(axis_names, axis_levels)), ';')
    p = String(path)
    mkpath(dirname(abspath(p)))
    now = _iso_time()
    session = get(attributes, "session", 1)
    function compatible(q)
        isfile(q) || return false
        try
            return NCDataset(ds -> get(ds.attrib, "aeolus_diagnostics_layout", "") == layout, q)
        catch
            return false
        end
    end
    if append && isfile(p) && !compatible(p)
        base, ext = splitext(p)
        n = 2
        while isfile("$base-$n$ext") && !compatible("$base-$n$ext")
            n += 1
        end
        newp = "$base-$n$ext"
        @warn("diagnostics file $p has another layout; this session writes $newp",
              subsystem = :diagnostics, domain = current_domain(), path = p, new_path = newp)
        p = newp
    end
    appended = append && isfile(p)
    if appended
        ds = NCDataset(p, "a")
        for (k, v) in attributes
            ds.attrib[String(k)] = v
        end
        ds.attrib["history"] = string(get(ds.attrib, "history", ""), "\n", now,
                                      " session ", session, ": appended by Aeolus")
    else
        attrs = Dict{String,Any}("title" => "Aeolus run diagnostics",
                                 "source" => "Aeolus.jl")
        for (k, v) in attributes
            attrs[String(k)] = v
        end
        attrs["history"] = string(now, " session ", session, ": created by Aeolus")
        attrs["aeolus_diagnostics_layout"] = layout
        ds = nc_create(p; attributes = attrs)
        _define_diagnostics_nc!(ds, step_diags, costly_diags, collect(profiles), axes,
                                axis_names, axis_levels, time_origin)
        NCDatasets.sync(ds)
    end
    w = DiagnosticsNCWriter(p, ds, step_diags, costly_diags, collect(profiles), Float64(cadence),
                            Float64(flush_interval), Int(flush_every), time(), -Inf, time(),
                            Int(ds.dim["time"]),
                            Int(ds.dim["time_costly"]), Int64[], Float64[], Float64[], Float64[],
                            String[], Int32[], [Float64[] for _ in step_diags],
                            Dict{String,Any}[])
    @debug("diagnostics.nc opened", subsystem = :diagnostics, domain = current_domain(),
           event = :open, path = p, scalars = length(diags), profiles = length(profiles),
           cadence = w.cadence, appended)
    return w
end

"""
    costly_due(w::DiagnosticsNCWriter, t) -> Bool

Whether a row at simulated time `t` is a costly row, by the rule of the
CSV writer.
"""
costly_due(w::DiagnosticsNCWriter, t::Real) =
    w.last_costly == -Inf || t - w.last_costly >= w.cadence * (1 - 1e-9)

"""
    record_diagnostics!(w::DiagnosticsNCWriter, t; step, dt = NaN, limiter = "",
                        retries = 0, costly = costly_due(w, t), values = nothing) -> Bool

Add a row at simulated time `t` to `diagnostics.nc` (see
[`DiagnosticsNCWriter`](@ref)): a buffered record of `time` and, when
`costly`, a record of `time_costly` with the costly scalars and the
profiles (written at once, after the buffered rows). `values`, a
dictionary `name => value`, supplies scalars already evaluated for this
row (the run monitor passes those of its CSV writer, so nothing is
evaluated twice); the others are evaluated here. Returns `costly`.
"""
function record_diagnostics!(w::DiagnosticsNCWriter, t::Real; step::Integer, dt::Real = NaN,
                             limiter = "", retries::Integer = 0,
                             costly::Bool = costly_due(w, t),
                             values::Union{Nothing,AbstractDict} = nothing)
    val(d) = values === nothing ? evaluate_diagnostic(d) :
             (v = get(values, d.name, nothing); v === nothing ? evaluate_diagnostic(d) : Float64(v))
    push!(w.b_step, step); push!(w.b_t, t); push!(w.b_wall, round(time() - w.wall0; digits = 3))
    push!(w.b_dt, dt); push!(w.b_lim, limiter === nothing ? "" : string(limiter))
    push!(w.b_retries, retries)
    for (k, d) in enumerate(w.step_diags)
        push!(w.b_vals[k], val(d))
    end
    w.rows += 1
    if costly
        rec = Dict{String,Any}("time_costly" => [Float64(t)], "step_costly" => [Int64(step)])
        for d in w.costly_diags
            rec[d.name] = [val(d)]
        end
        for p in w.profiles
            rec[p.name] = evaluate_profile(p)
        end
        push!(w.b_costly, rec)          # written with the next block (#514)
        w.costly_rows += 1
        w.last_costly = Float64(t)
    end
    (length(w.b_step) >= w.flush_every || time() - w.last_flush >= w.flush_interval) && _flush_rows!(w)
    return costly
end

# write the buffered rows of `time` as one block, then those of
# `time_costly` (#514), with one sync at the end
function _flush_rows!(w::DiagnosticsNCWriter; sync::Bool = true)
    n = length(w.b_step)
    w.last_flush = time()
    n == 0 && (_flush_costly!(w; sync); return w)
    rec = Dict{String,Any}("time" => w.b_t, "step" => w.b_step, "wall" => w.b_wall,
                           "dt" => w.b_dt, "dt_limiter" => w.b_lim, "retries" => w.b_retries)
    for (k, d) in enumerate(w.step_diags)
        rec[d.name] = w.b_vals[k]
    end
    nc_append_record!(w.ds, rec; dim = "time", count = n, sync = sync && isempty(w.b_costly))
    for b in (w.b_step, w.b_t, w.b_wall, w.b_dt, w.b_lim, w.b_retries)
        empty!(b)
    end
    foreach(empty!, w.b_vals)
    _flush_costly!(w; sync)
    return w
end

# the buffered rows of `time_costly` as one block: every scalar a vector of
# the rows, every profile a (levels × rows) matrix (a vector for one row)
function _flush_costly!(w::DiagnosticsNCWriter; sync::Bool = true)
    rows = w.b_costly
    nr = length(rows)
    nr == 0 && return w
    profs = Set(p.name for p in w.profiles)
    rec = Dict{String,Any}()
    for k in keys(first(rows))
        rec[k] = k in profs ? (nr == 1 ? first(rows)[k] : reduce(hcat, (r[k] for r in rows))) :
                 [only(r[k]) for r in rows]
    end
    nc_append_record!(w.ds, rec; dim = "time_costly", count = nr, sync = sync)
    empty!(rows)
    return w
end

"Write the buffered rows and sync the file."
Base.flush(w::DiagnosticsNCWriter) = (isopen(w.ds) && _flush_rows!(w); nothing)

"Write the buffered rows and close the file."
function Base.close(w::DiagnosticsNCWriter)
    if isopen(w.ds)
        _flush_rows!(w)
        haskey(w.ds.attrib, "session_finished") && (w.ds.attrib["session_finished"] = _iso_time())
        close(w.ds)
    end
    return nothing
end
