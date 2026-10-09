"""
    Aeolus.Output

The one output writer of the model (ONE-c, #452; the design is
`docs/src/design/single_model.md`, "Output"). It replaces the three
writers the run types had (the `simulation` run type's
`fields_<step>.nc`, the urban case's NetCDF and the SBL run type's
`series.csv`).

Every component publishes an output **catalogue**: a list of
[`OutputVariable`](@ref)s (name, units, long name, dimensions and how the
value is computed). The configuration selects variables from the
catalogues of the enabled components (`[output] variables`); the writer
([`OutputWriter`](@ref)) writes them into one NetCDF-4 file (through
`Aeolus.NCIO`) as two record streams:

* the **records** (unlimited dimension `time`): 3-D fields (`:xyz`),
  surface maps (`:xy`), vertical profiles (`:z`) and per-record scalars,
  every `[output] cadence`;
* the **time series** (unlimited dimension `time_series`): scalars
  (`:scalar`, e.g. the domain means), every `series_cadence`.

With `averaging = :mean` every written value is the time mean over the
interval since the last record of its stream (the writer accumulates
every step, weighted by the step length; the variables carry
`cell_methods = "time: mean"`); with `:instant` it is the value at the
record's time.
"""
module Output

using ..NCIO: NCDataset, nc_create, nc_defvar!, nc_append_record!

export OutputVariable, OUTPUT_DIMS, check_catalogue, catalogue_names, select_variables,
       cell_centres, OutputWriter, accumulate_output!, write_record!, write_series!, close_output!,
       output_path

"""
    OUTPUT_DIMS

The dimensions an output variable can have: `:xyz` (a 3-D field, per
record), `:xy` (a surface map, per record), `:z` (a vertical profile, per
record), `:k` (a spectrum over the writer's wavenumbers `k`, per record)
and `:scalar` (one number: a time series, or a per-record value).
"""
const OUTPUT_DIMS = (:xyz, :xy, :z, :k, :scalar)

"""
    OutputVariable(name, units, long_name, dims, getter; eltype = Float32 (Float64 for :scalar),
                   group = false)

One variable of an output catalogue. `dims` is one of [`OUTPUT_DIMS`](@ref);
`getter(ctx)` returns its value for the writer's context `ctx` (a model
domain, an urban case, …): an interior-sized `(Nx, Ny, Nz)` array for
`:xyz`, `(Nx, Ny)` for `:xy`, a length-`Nz` vector for `:z`, a vector of
the length of the writer's wavenumbers `k` for `:k`, a number for
`:scalar`. With `group = true` (a `:scalar` variable) `getter` returns a
`Dict{String,Float64}`: a **group** of scalars, written as
`<name>__<key>` and defined as they appear (the budget terms, which
appear as the run books them).
"""
struct OutputVariable
    name::String
    units::String
    long_name::String
    dims::Symbol
    getter::Function
    eltype::DataType
    group::Bool
    function OutputVariable(name::AbstractString, units::AbstractString, long_name::AbstractString,
                            dims::Symbol, getter::Function; eltype::DataType = dims === :scalar ? Float64 : Float32,
                            group::Bool = false)
        dims in OUTPUT_DIMS || throw(ArgumentError("OutputVariable $name: dims must be one of $(OUTPUT_DIMS), got :$dims"))
        group && dims !== :scalar && throw(ArgumentError("OutputVariable $name: a group is :scalar"))
        return new(String(name), String(units), String(long_name), dims, getter, eltype, group)
    end
end

"The names of a catalogue, in order."
catalogue_names(vars) = String[v.name for v in vars]

"""
    check_catalogue(vars) -> vars

Throw an `ArgumentError` if two variables of the catalogue `vars` share a
name.
"""
function check_catalogue(vars)
    names = catalogue_names(vars)
    dup = unique(n for n in names if count(==(n), names) > 1)
    isempty(dup) || throw(ArgumentError("duplicate output variable names: $(join(dup, ", "))"))
    return vars
end

"""
    select_variables(catalogue, names) -> Vector{OutputVariable}

The variables of `catalogue` named in `names`, in the order of `names`;
an `ArgumentError` names any that the catalogue does not have.
"""
function select_variables(catalogue, names)
    by = Dict(v.name => v for v in check_catalogue(catalogue))
    missing_ = [n for n in names if !haskey(by, n)]
    isempty(missing_) ||
        throw(ArgumentError("output variables not in the catalogue: $(join(missing_, ", ")) " *
                            "(it has: $(join(sort!(collect(keys(by))), ", ")))"))
    return OutputVariable[by[n] for n in names]
end

"""
    cell_centres(mesh) -> (x, y, z)

The cell-centre coordinates of `mesh` (m from its origin): uniform in `x`
and `y`, `zc` of a stretched grid in `z`.
"""
function cell_centres(mesh)
    x = [(i - 0.5) * mesh.dx for i in 1:mesh.Nx]
    y = [(j - 0.5) * mesh.dy for j in 1:mesh.Ny]
    z = hasproperty(mesh, :zc) ? Float64[mesh.zc[mesh.Hz+k] - mesh.z0 for k in 1:mesh.Nz] :
                                 [(k - 0.5) * mesh.dz for k in 1:mesh.Nz]
    return x, y, z
end

# one record stream: its variables, the averaging accumulators and its count
mutable struct _Stream
    dim::String
    vars::Vector{OutputVariable}
    acc::Vector{Any}              # :mean: Σ value·dt per variable (nothing for a group)
    acc_dt::Float64
    groups::Dict{String,Set{String}}   # a group variable's members defined so far
    n::Int
end

"""
    OutputWriter(path; x, y, z, k = nothing, records = [], series = [], statics = [], ctx = nothing,
                 time_units = "s", attributes = Dict(), averaging = :instant, resume_at = nothing,
                 resume_time = nothing)

The writer of one run's output file `path` (NetCDF-4, `Aeolus.NCIO`):
coordinates `x`, `y`, `z` (cell centres, m) and, for spectra, `k` (rad
m⁻¹), the record variables
`records` (any of [`OUTPUT_DIMS`](@ref); unlimited dimension `time`), the
time-series variables `series` (`:scalar`; unlimited dimension
`time_series`), and `statics`, written once from `ctx` (`:xy` or `:xyz`
maps without a time dimension: the static geometry). `time_units` is the
`units` of the two time coordinates (CF: `"seconds since …"`).

`averaging = :mean` writes the mean over each record's interval (call
[`accumulate_output!`](@ref) after every step); `:instant` the value at
the record. `resume_at = n` reopens an existing file and continues its
records at record `n` (a run resumed from a checkpoint, #269);
`resume_time = t` reopens it and continues both streams after their last
sample at or before `t` (the samples after it, written by an interrupted
session, are overwritten). With no variable at all no file is written
([`output_path`](@ref) is `nothing`).
"""
mutable struct OutputWriter
    ds::Union{Nothing,NCDataset}
    path::String
    records::_Stream
    series::_Stream
    averaging::Symbol
end

function OutputWriter(path::AbstractString; x, y, z, k = nothing, records = OutputVariable[], series = OutputVariable[],
                      statics = OutputVariable[], ctx = nothing, time_units::AbstractString = "s",
                      attributes::AbstractDict = Dict{String,Any}(), averaging::Symbol = :instant,
                      resume_at::Union{Nothing,Integer} = nothing, resume_time::Union{Nothing,Real} = nothing)
    averaging in (:instant, :mean) || throw(ArgumentError("OutputWriter: averaging is :instant or :mean, got :$averaging"))
    records = collect(OutputVariable, records); series = collect(OutputVariable, series)
    statics = collect(OutputVariable, statics)
    check_catalogue(vcat(records, series, statics))
    all(v -> v.dims === :scalar, series) || throw(ArgumentError("OutputWriter: the time series are :scalar variables"))
    all(v -> v.dims in (:xy, :xyz), statics) || throw(ArgumentError("OutputWriter: the static variables are :xy or :xyz maps"))
    any(v -> v.dims === :k, records) && k === nothing &&
        throw(ArgumentError("OutputWriter: a spectrum (:k) needs the wavenumbers k"))
    rs = _Stream("time", records, Any[nothing for _ in records], 0.0, Dict{String,Set{String}}(), 0)
    ss = _Stream("time_series", series, Any[nothing for _ in series], 0.0, Dict{String,Set{String}}(), 0)
    w = OutputWriter(nothing, String(path), rs, ss, averaging)
    isempty(records) && isempty(series) && isempty(statics) && return w
    if (resume_at !== nothing || resume_time !== nothing) && isfile(path)
        w.ds = NCDataset(path, "a")
        upto(d) = count(<=(resume_time * (1 + 1e-12) + 1e-12), w.ds[d].var[:])
        rs.n = resume_at !== nothing ? Int(resume_at) - 1 : upto("time")
        ss.n = resume_time !== nothing ? upto("time_series") : Int(w.ds.dim["time_series"])
        for st in (rs, ss), v in st.vars
            v.group && (st.groups[v.name] = Set(k[length(v.name)+3:end] for k in keys(w.ds)
                                                           if startswith(k, v.name * "__")))
        end
        return w
    end
    mkpath(dirname(abspath(path)))
    ds = nc_create(path; attributes = attributes)
    w.ds = ds
    ds.dim["x"] = length(x); ds.dim["y"] = length(y); ds.dim["z"] = length(z)
    ds.dim["time"] = Inf
    ds.dim["time_series"] = Inf
    for (d, c) in (("x", x), ("y", y), ("z", z))
        nc_defvar!(ds, d, Float32, (d,); attributes = Dict("units" => "m", "axis" => uppercase(d),
                                                            "long_name" => "$d cell centre"))
        ds[d][:] = Float32.(c)
    end
    if k !== nothing
        ds.dim["k"] = length(k)
        nc_defvar!(ds, "k", Float64, ("k",); attributes = Dict("units" => "rad m-1",
                                                                "long_name" => "wavenumber of the spectral shells"))
        ds["k"][:] = Float64.(k)
    end
    for (d, ln) in (("time", "simulation time of the records"), ("time_series", "simulation time of the time series"))
        nc_defvar!(ds, d, Float64, (d,); attributes = Dict("units" => time_units, "axis" => "T", "long_name" => ln))
    end
    for v in statics
        nc_defvar!(ds, v.name, v.eltype, v.dims === :xy ? ("x", "y") : ("x", "y", "z");
                   attributes = Dict("units" => v.units, "long_name" => v.long_name,
                                     "coordinates" => v.dims === :xy ? "x y" : "x y z"))
        ds[v.name][ntuple(_ -> Colon(), v.dims === :xy ? 2 : 3)...] = v.eltype.(v.getter(ctx))
    end
    for st in (rs, ss), v in st.vars
        v.group ? (st.groups[v.name] = Set{String}()) : _defvar!(w, st, v, v.name, v.units, v.long_name)
    end
    return w
end

function _defvar!(w::OutputWriter, st::_Stream, v::OutputVariable, name, units, long_name)
    dims = v.dims === :xyz ? ("x", "y", "z", st.dim) : v.dims === :xy ? ("x", "y", st.dim) :
           v.dims === :z ? ("z", st.dim) : v.dims === :k ? ("k", st.dim) : (st.dim,)
    attrs = Dict{String,Any}("units" => units, "long_name" => long_name)
    v.dims === :scalar ||
        (attrs["coordinates"] = v.dims === :xyz ? "x y z" : v.dims === :xy ? "x y" : v.dims === :k ? "k" : "z")
    w.averaging === :mean && (attrs["cell_methods"] = "$(st.dim): mean")
    nc_defvar!(w.ds, name, v.eltype, dims; attributes = attrs)
    return nothing
end

"The path of the writer's file, or `nothing` when it writes none."
output_path(w::OutputWriter) = w.ds === nothing ? nothing : w.path

_value(v::OutputVariable, ctx) = v.getter(ctx)

"""
    accumulate_output!(w, ctx, dt) -> w

With `averaging = :mean`, add the current values of every variable times
the step length `dt` to the record and time-series means (call it after
every step). A no-op with `:instant`.
"""
function accumulate_output!(w::OutputWriter, ctx, dt::Real)
    (w.averaging === :mean && w.ds !== nothing) || return w
    for st in (w.records, w.series)
        isempty(st.vars) && continue
        for (i, v) in enumerate(st.vars)
            x = _value(v, ctx)
            if v.group
                a = st.acc[i] === nothing ? (st.acc[i] = Dict{String,Float64}()) : st.acc[i]
                for (k, y) in x
                    a[k] = get(a, k, 0.0) + Float64(y) * dt
                end
            elseif st.acc[i] === nothing
                st.acc[i] = x isa Number ? Float64(x) * dt : Float64.(x) .* dt
            elseif x isa Number
                st.acc[i] += Float64(x) * dt
            else
                st.acc[i] .+= x .* dt
            end
        end
        st.acc_dt += dt
    end
    return w
end

function _write!(w::OutputWriter, st::_Stream, ctx, t::Real, at)
    w.ds === nothing && return w
    isempty(st.vars) && return w
    mean = w.averaging === :mean
    mean && st.acc_dt == 0 && return w          # an empty interval: nothing to average
    rec = Dict{String,Any}(st.dim => [Float64(t)])
    for (i, v) in enumerate(st.vars)
        x = mean ? st.acc[i] : _value(v, ctx)
        if v.group
            members = st.groups[v.name]
            for (k, y) in x
                name = string(v.name, "__", k)
                if !(k in members)
                    _defvar!(w, st, v, name, v.units, "$(v.long_name): $k")
                    push!(members, k)
                end
                rec[name] = [v.eltype(mean ? y / st.acc_dt : y)]
            end
            for k in members                  # a member this record does not have
                haskey(x, k) || (rec[string(v.name, "__", k)] = [v.eltype(NaN)])
            end
        elseif v.dims === :scalar
            rec[v.name] = [v.eltype(mean ? x / st.acc_dt : x)]
        else
            rec[v.name] = v.eltype.(mean ? x ./ st.acc_dt : x)
        end
    end
    st.n = at === nothing ? st.n + 1 : Int(at)       # after a resume: over the interrupted session's samples
    nc_append_record!(w.ds, rec; dim = st.dim, at = st.n)
    if mean
        fill!(st.acc, nothing); st.acc_dt = 0.0
    end
    return w
end

"""
    write_record!(w, ctx, t; at = nothing) -> w

Write one record of the record variables at time `t` (s): their values
now (`:instant`) or their means since the last record (`:mean`; nothing is
written for an empty interval). `at` writes it as record `at` (a resumed
run rewrites the records after its checkpoint).
"""
write_record!(w::OutputWriter, ctx, t::Real; at = nothing) = _write!(w, w.records, ctx, t, at)

"""
    write_series!(w, ctx, t) -> w

Append one sample of the time-series variables at time `t` (s), like
[`write_record!`](@ref).
"""
write_series!(w::OutputWriter, ctx, t::Real) = _write!(w, w.series, ctx, t, nothing)

"Close the writer's file (if it has one)."
function close_output!(w::OutputWriter)
    w.ds === nothing && return w
    close(w.ds)
    w.ds = nothing
    return w
end

end # module
