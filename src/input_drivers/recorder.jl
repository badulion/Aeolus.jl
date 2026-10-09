# The dynamic driver an Aeolus parent writes for its child (CFG-g, #362):
# an in-run writer, model output (`[output] child_boundaries`). A parent run
# (in the Berlin case, the nest of the child run) records the prolongated
# boundary slabs of the child at every parent step, so a child run alone, fed from the file, sees the same Float64
# snapshots and the same time blend as the online nest: offline equals
# online bit for bit. One-way offline nesting.

# ---------------------------------------------------------------------------

"""
    NestBoundaryRecorder(path, planes, mesh; init_time = "", fields = Dict(), profiles = (;),
                         attributes = Dict(), inputs = nothing, source = "", preprocessor, anchor…)

The in-run writer of a dynamic driver's boundary planes (an Aeolus parent
writing its child's driver, CFG-g #362; AeolusPrep.jl uses it for the
steady planes of a sounding driver and the ERA5 planes): it defines a `bc_<face>_<var>` variable for
every non-mirror plane of `planes` (the `OpenPlane`s of the child's open
boundary set) over the unlimited `time_bc`, writes the 3-D initial
`fields` (`:u`, `:v`, `:w`, …: interior arrays), and on each
[`record_boundary!`](@ref) appends one record: every plane's values read
from a snapshot (`snapshot_values(s, name)` at the plane's storage
points, exactly what `blend_window!` reads online), the record time, the
window length, and the mean `profiles` (`name => vector` over the run's
level centres: `ls_profile_<name>` over (`ls_profile_<name>_z`,
`time_bc`)). `inputs` (a [`DynamicInputs`](@ref)) writes its profiles,
tables, series and attributes into the same file (a sounding driver with
the steady planes of an open case). The file is opened for each record,
so a run that stops leaves a readable driver.
"""
mutable struct NestBoundaryRecorder
    path::String
    layout::Vector{_SlabLayout}
    profiles::Vector{Symbol}
    records::Int
end

function NestBoundaryRecorder(path::AbstractString, planes, mesh; init_time::AbstractString = "",
                              fields::AbstractDict = Dict{Symbol,Array{Float64,3}}(),
                              profiles = Symbol[], attributes::AbstractDict = Dict{String,Any}(),
                              inputs::Union{Nothing,DynamicInputs} = nothing,
                              source::AbstractString = "", preprocessor::AbstractString = "aeolus preprocess dynamic",
                              origin_lat = NaN, origin_lon = NaN, origin_x = NaN, origin_y = NaN)
    inputs === nothing || (init_time = inputs.init_time)
    extra = inputs === nothing ? Dict{String,Any}() : copy(inputs.attributes)
    merge!(extra, attributes); extra["dynamic_source"] = get(extra, "dynamic_source", "parent")
    ds = _create_dynamic(path, init_time, extra; mesh, source, preprocessor, origin_lat, origin_lon,
                         origin_x, origin_y, dx = NaN, dy = NaN)
    layout = _SlabLayout[]
    try
        _write_static_part!(ds, inputs === nothing ? DynamicInputs(; init_time, fields) : inputs)
        ds.dim["time_bc"] = Inf
        tu = _tunits(init_time)
        NCDatasets.defVar(ds, "time_bc", Float64, ("time_bc",);
                          attrib = ["units" => tu, "long_name" => DYNAMIC_SCHEMA["time_bc"].long_name])
        NCDatasets.defVar(ds, "time_bc_step", Float64, ("time_bc",);
                          attrib = ["units" => "s", "long_name" => DYNAMIC_SCHEMA["time_bc_step"].long_name])
        for p in planes
            p.mode === :mirror && continue
            name = "bc_$(p.side)_$(_ascii_name(p.name))"
            haskey(ds, name) && error("NestBoundaryRecorder: two planes for :$(p.name) on the $(p.side) face")
            r = (p.irange, p.jrange, p.krange)
            dn = [name * s for s in ("_i", "_j", "_k")]
            for (d, rr) in zip(dn, r)
                ds.dim[d] = length(rr)
            end
            nc_defvar!(ds, name, Float64, (dn..., "time_bc");
                       attributes = Dict{String,Any}("field" => String(p.name), "side" => String(p.side),
                                                     "i0" => first(r[1]), "j0" => first(r[2]), "k0" => first(r[3]),
                                                     "halo_mode" => String(p.mode),
                                                     "long_name" => "boundary slab of $(p.name), $(p.side) face"),
                       chunksizes = [length.(r)..., 1])
            push!(layout, _SlabLayout(name, p.name, p.side, r, p.mode))
        end
        zc = collect(Float64, znodes(mesh, CellCenter()))
        for n in profiles
            nm = Symbol(:profile_, n)
            v = DYNAMIC_SCHEMA["ls_$nm"]
            zd = "ls_$(nm)_z"
            ds.dim[zd] = length(zc)
            _defvar1!(ds, zd, zc, Dict{String,Any}("units" => "m", "long_name" => "height of ls_$nm", "axis" => "Z"))
            nc_defvar!(ds, v.name, Float64, (zd, "time_bc");
                       attributes = Dict{String,Any}("units" => v.units, "long_name" => v.long_name),
                       chunksizes = [length(zc), 1])
        end
    finally
        close(ds)
    end
    return NestBoundaryRecorder(String(path), layout, Symbol[profiles...], 0)
end

"""
    record_boundary!(rec, t, step, snapshot; profiles = (;)) -> rec

Append one `time_bc` record to the driver of `rec`: time `t`, window
length `step` (`NaN` for the first record), the values of every plane
from `snapshot` (`snapshot_values(snapshot, name)` at the plane's storage
ranges) and the mean `profiles` (`name => vector`, the names given to the
recorder).
"""
function record_boundary!(rec::NestBoundaryRecorder, t::Real, step::Real, snap; profiles = (;))
    data = Dict{String,Any}("time_bc" => [Float64(t)], "time_bc_step" => [Float64(step)])
    for l in rec.layout
        A = snapshot_values(snap, l.field)
        data[l.var] = Array{Float64}(A[l.ranges...])
    end
    for n in rec.profiles
        haskey(profiles, n) || error("record_boundary!: no mean profile :$n")
        data["ls_profile_$n"] = Float64.(collect(profiles[n]))
    end
    NCDataset(rec.path, "a") do ds
        nc_append_record!(ds, data; dim = "time_bc")
    end
    rec.records += 1
    return rec
end

"""
    plane_snapshot(planes, mesh) -> FileSnapshot

The current buffers of `planes` (the non-mirror `OpenPlane`s of an open
boundary set) as a boundary snapshot: what [`record_boundary!`](@ref)
writes for boundary data evaluated on the planes themselves (the steady
planes of an open case, `boundary_values!` of its analytic data).
"""
function plane_snapshot(planes, mesh)
    slabs = Dict{Symbol,Vector{_Slab}}()
    for p in planes
        p.mode === :mirror && continue
        push!(get!(slabs, p.name, _Slab[]), _Slab((p.irange, p.jrange, p.krange), copy(p.data)))
    end
    return FileSnapshot(slabs, (mesh.Nx + 2mesh.Hx, mesh.Ny + 2mesh.Hy, mesh.Nz + 2mesh.Hz))
end
