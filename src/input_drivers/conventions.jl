# Conventions shared by every Aeolus input driver (CFG-f #361; the dynamic
# driver of CFG-g uses the same): global attributes, the projection, and
# the declarative schema a file is checked against.

"The version of the driver conventions and schemas below; a file of another version is refused."
const DRIVER_SCHEMA_VERSION = 1

"""
    DRIVER_ATTRIBUTES

The global attributes every Aeolus input driver carries, with what each
holds:

* `aeolus_driver` — `"static"` or `"dynamic"`;
* `aeolus_form` — the layout within the kind (`"resolved"`, `"tiles"`);
* `aeolus_schema_version` — [`DRIVER_SCHEMA_VERSION`](@ref);
* `origin_lat`, `origin_lon` (degrees) — the anchor of the projection;
* `projection` — `"equirectangular"` ([`latlon_to_meters`](@ref));
* `origin_x`, `origin_y` (m) — where the anchor lies in the run's mesh
  coordinates (the `x`, `y` of the file);
* `dx`, `dy` (m) — the horizontal spacing of the run's grid;
* `source` — the file the driver was made from, relative to the driver's
  own directory when it lies under it, else absolute;
* `source_fnv1a64` — the FNV-1a hash of that file (16 hex digits): a
  driver whose source has since changed is reported as stale;
* `preprocessor` — what wrote it (e.g. `"aeolus preprocess static"`);
* `aeolus_commit` — the git commit of the writing Aeolus (`""` outside a
  checkout).

A file also carries the CF `Conventions` attribute and `date_created`.
"""
const DRIVER_ATTRIBUTES = ("aeolus_driver", "aeolus_form", "aeolus_schema_version", "origin_lat", "origin_lon",
                           "projection", "origin_x", "origin_y", "dx", "dy", "source",
                           "source_fnv1a64", "preprocessor", "aeolus_commit")

const _EARTH_R = 6_371_000.0

"""
    latlon_to_meters(lat, lon, lat0, lon0) -> (x, y)

Project geographic (`lat`, `lon`) in degrees to local (x, y) metres
about the anchor (`lat0`, `lon0`): the equirectangular (plate carrée)
projection with the anchor's parallel as the standard parallel,
``x = R\\,Δλ\\cos φ_0``, ``y = R\\,Δφ``, ``R = 6371`` km. Sub-metre
accurate over the ≤ 5 km of an LES domain. The one projection of the
input drivers (`projection = "equirectangular"`).
"""
function latlon_to_meters(lat::Real, lon::Real, lat0::Real, lon0::Real)
    x = _EARTH_R * deg2rad(lon - lon0) * cos(deg2rad(lat0))
    y = _EARTH_R * deg2rad(lat - lat0)
    return (x, y)
end

"`fnv1a64` of the bytes of the file `path`, as 16 hex digits."
file_fnv1a64(path::AbstractString) = string(fnv1a64(read(path)); base = 16, pad = 16)

# ---------------------------------------------------------------------------
# the declarative schema
# ---------------------------------------------------------------------------

"""
    DriverVariable(name, dims, type; units = "", long_name, standard_name = "",
                   required = true, range = nothing, doc = "")

One variable of a [`DriverSchema`](@ref): its `name`, its dimensions
`dims` (Julia index order: `("x", "y")` is an `(Nx, Ny)` array), its
element `type`, its CF `units`, `long_name` and `standard_name` (`""`
where CF has none), whether a file must have it, and the closed `range`
`(lo, hi)` its values must lie in (`nothing`: unchecked; `NaN` values,
the ring separators of the polygons, are skipped).
"""
struct DriverVariable
    name::String
    dims::Tuple{Vararg{String}}
    type::DataType
    units::String
    long_name::String
    standard_name::String
    required::Bool
    range::Union{Nothing,Tuple{Real,Real}}
    doc::String
end
DriverVariable(name, dims, type; units = "", long_name = name, standard_name = "", required = true,
               range = nothing, doc = "") =
    DriverVariable(String(name), Tuple(String.(dims)), type, String(units), String(long_name),
                   String(standard_name), required, range, String(doc))

"""
    DriverSchema(kind, form, doc, variables)

A declarative table of the variables of one kind of input driver: `kind`
is the `aeolus_driver` attribute (`"static"`), `form` the `aeolus_form`
attribute that tells two layouts of one kind apart (`"resolved"`,
`"tiles"`). [`check_driver`](@ref) checks a file against it and
[`schema_markdown`](@ref) renders it for the docs.
"""
struct DriverSchema
    kind::String
    form::String
    doc::String
    variables::Vector{DriverVariable}
end

function Base.getindex(s::DriverSchema, name::AbstractString)
    i = findfirst(v -> v.name == name, s.variables)
    i === nothing && error("DriverSchema $(s.kind)/$(s.form): no variable $name")
    return s.variables[i]
end

_attr(ds, k, default = nothing) = haskey(ds.attrib, k) ? ds.attrib[k] : default

"""
    source_path(path, ds) -> String or nothing

The source file a driver at `path` names in its `source` attribute
(resolved against the driver's directory), or `nothing` if it names none.
"""
function source_path(path::AbstractString, ds)
    s = _attr(ds, "source", "")
    isempty(s) && return nothing
    return isabspath(s) ? s : normpath(joinpath(dirname(abspath(path)), s))
end

"""
    check_driver(path, schema; source = true) -> Vector{String}

Every problem of the driver file `path` against `schema`, as messages
(empty: the file is good):

* not a NetCDF file, another `aeolus_driver` kind or `aeolus_form`;
* an `aeolus_schema_version` other than [`DRIVER_SCHEMA_VERSION`](@ref);
* a missing global attribute of [`DRIVER_ATTRIBUTES`](@ref);
* a missing required variable, a variable over the wrong dimensions, of
  the wrong element type or units, or with values outside its range;
* with `source = true`, a `source_fnv1a64` that does not match the
  source file (when that file is present: a driver can be used without
  its source).
"""
function check_driver(path::AbstractString, schema::DriverSchema; source::Bool = true)
    probs = String[]
    isfile(path) || return ["$path: no such file"]
    ds = try
        NCDataset(path)
    catch e
        return ["$path: not a NetCDF file ($(sprint(showerror, e)))"]
    end
    try
        kind = _attr(ds, "aeolus_driver")
        kind == schema.kind || push!(probs, "$path: aeolus_driver = $(repr(kind)), expected $(repr(schema.kind))")
        form = _attr(ds, "aeolus_form")
        form == schema.form || push!(probs, "$path: aeolus_form = $(repr(form)), expected $(repr(schema.form))")
        ver = _attr(ds, "aeolus_schema_version")
        ver == DRIVER_SCHEMA_VERSION ||
            push!(probs, "$path: aeolus_schema_version = $(repr(ver)), this Aeolus reads version $DRIVER_SCHEMA_VERSION")
        for a in DRIVER_ATTRIBUTES
            haskey(ds.attrib, a) || push!(probs, "$path: missing global attribute $a")
        end
        for v in schema.variables
            if !haskey(ds, v.name)
                v.required && push!(probs, "$path: missing required variable $(v.name) ($(v.long_name))")
                continue
            end
            var = ds[v.name]
            dn = Tuple(String.(NCDatasets.dimnames(var)))
            dn == v.dims || push!(probs, "$path: $(v.name) is over $(dn), expected $(v.dims)")
            T = eltype(var.var)
            T == v.type || push!(probs, "$path: $(v.name) has element type $T, expected $(v.type)")
            u = _attr(var, "units", "")
            isempty(v.units) || u == v.units || push!(probs, "$path: $(v.name) has units $(repr(u)), expected $(repr(v.units))")
            if v.range !== nothing && v.type <: Real && dn == v.dims
                a = var.var[:]
                lo, hi = v.range
                bad = count(x -> !(x isa AbstractFloat && isnan(x)) && !(lo <= x <= hi), a)
                bad == 0 || push!(probs, "$path: $(v.name) has $bad value(s) outside [$lo, $hi]")
            end
        end
        if source
            sp = source_path(path, ds)
            want = _attr(ds, "source_fnv1a64", "")
            if sp !== nothing && isfile(sp) && !isempty(want)
                got = file_fnv1a64(sp)
                got == want || push!(probs, "$path: stale: its source $sp has changed since it was written " *
                                            "(source_fnv1a64 $want, the file now $got)")
            end
        end
    finally
        close(ds)
    end
    return probs
end

"""
    schema_markdown(schema) -> String

A Markdown table of `schema`'s variables (name, dimensions, type, units,
required, range, meaning), for the docs page.
"""
function schema_markdown(s::DriverSchema)
    io = IOBuffer()
    println(io, "| variable | dims | type | units | required | range | meaning |")
    println(io, "| :-- | :-- | :-- | :-- | :-- | :-- | :-- |")
    for v in s.variables
        r = v.range === nothing ? "" : "[$(v.range[1]), $(v.range[2])]"
        sn = isempty(v.standard_name) ? "" : " (CF `$(v.standard_name)`)"
        println(io, "| `$(v.name)` | ", isempty(v.dims) ? "scalar" : join(v.dims, ", "), " | `", v.type, "` | ",
                isempty(v.units) ? "" : "`$(v.units)`", " | ", v.required ? "yes" : "no", " | ", r, " | ",
                v.long_name, sn, isempty(v.doc) ? "" : ": " * v.doc, " |")
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# writing the conventions
# ---------------------------------------------------------------------------

"""
    driver_attributes(kind, form; origin_lat, origin_lon, origin_x, origin_y, dx, dy,
                      source = "", out = "", preprocessor) -> Dict{String,Any}

The global attributes of a driver ([`DRIVER_ATTRIBUTES`](@ref)) about to
be written to `out`: `source` is recorded relative to `out`'s directory
when it lies under the same tree (else absolute) with its hash, the
commit is the writing checkout's.
"""
function driver_attributes(kind, form; origin_lat, origin_lon, origin_x, origin_y, dx, dy,
                           source::AbstractString = "", out::AbstractString = "",
                           preprocessor::AbstractString)
    src = ""; h = ""
    if !isempty(source)
        isfile(source) || error("driver_attributes: the source $source does not exist")
        h = file_fnv1a64(source)
        src = _source_ref(source, out)
    end
    g = git_state()
    return Dict{String,Any}(
        "aeolus_driver" => String(kind), "aeolus_form" => String(form),
        "aeolus_schema_version" => DRIVER_SCHEMA_VERSION,
        "origin_lat" => Float64(origin_lat), "origin_lon" => Float64(origin_lon),
        "projection" => "equirectangular",
        "origin_x" => Float64(origin_x), "origin_y" => Float64(origin_y),
        "dx" => Float64(dx), "dy" => Float64(dy),
        "source" => src, "source_fnv1a64" => h, "preprocessor" => String(preprocessor),
        "aeolus_commit" => g.commit,
        "date_created" => Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))
end

# `source` relative to the directory of `out` when the two share a
# directory below the top level (a source in the same checkout), else absolute
function _source_ref(source, out)
    s, o = abspath(source), dirname(abspath(out))
    isempty(out) && return s
    a, b = splitpath(s), splitpath(o)
    n = 0
    while n < min(length(a), length(b)) && a[n+1] == b[n+1]
        n += 1
    end
    return n >= 3 ? relpath(s, o) : s
end

"The attributes that change on every write (a reproduced file differs only in these)."
const VOLATILE_ATTRIBUTES = ("date_created", "aeolus_commit")

"Define and write variable `v` of `schema` with data `A` (CF attributes from the schema, `extra` on top)."
function _write_var!(ds, v::DriverVariable, A; extra = Dict{String,Any}())
    attrs = Dict{String,Any}("long_name" => v.long_name)
    isempty(v.units) || (attrs["units"] = v.units)
    isempty(v.standard_name) || (attrs["standard_name"] = v.standard_name)
    v.range === nothing || v.type === String || (attrs["valid_range"] = v.type[v.range[1], v.range[2]])
    merge!(attrs, extra)
    if v.type === String
        var = NCDatasets.defVar(ds, v.name, String, collect(v.dims); attrib = collect(attrs))
    else
        var = nc_defvar!(ds, v.name, v.type, v.dims; attributes = attrs,
                         chunksizes = [Int(ds.dim[d]) for d in v.dims])
    end
    var[ntuple(_ -> Colon(), length(v.dims))...] = A
    return var
end

"""
    check_grid(path, ds, mesh; origin_lat, origin_lon, origin_x, origin_y)

Error, naming both grids, unless the driver `ds` (at `path`) is on the
horizontal grid of `mesh` (its `x`, `y` cell centres, `dx`, `dy`) with
the anchor at the same place.
"""
function check_grid(path, ds, mesh; origin_lat, origin_lon, origin_x, origin_y)
    xs = collect(Float64, xnodes(mesh, CellCenter())); ys = collect(Float64, ynodes(mesh, CellCenter()))
    fx = Float64.(ds["x"].var[:]); fy = Float64.(ds["y"].var[:])
    file = (Nx = length(fx), Ny = length(fy), dx = _attr(ds, "dx"), dy = _attr(ds, "dy"),
            x = isempty(fx) ? NaN : first(fx), y = isempty(fy) ? NaN : first(fy),
            origin_x = _attr(ds, "origin_x"), origin_y = _attr(ds, "origin_y"),
            origin_lat = _attr(ds, "origin_lat"), origin_lon = _attr(ds, "origin_lon"))
    run = (Nx = length(xs), Ny = length(ys), dx = Float64(mesh.dx), dy = Float64(mesh.dy),
           x = first(xs), y = first(ys), origin_x = Float64(origin_x), origin_y = Float64(origin_y),
           origin_lat = Float64(origin_lat), origin_lon = Float64(origin_lon))
    same = fx == xs && fy == ys && all(k -> isequal(getfield(file, k), getfield(run, k)),
                                       (:dx, :dy, :origin_x, :origin_y, :origin_lat, :origin_lon))
    same && return nothing
    show_(g) = "$(g.Nx) × $(g.Ny) cells of $(g.dx) × $(g.dy) m, first centre ($(g.x), $(g.y)) m, " *
               "anchor ($(g.origin_lat)°, $(g.origin_lon)°) at ($(g.origin_x), $(g.origin_y)) m"
    error("the input driver $path is not on the run's grid (no resampling: re-run the preprocessor " *
          "for this grid, `aeolus preprocess static <config.toml>`):\n  file: $(show_(file))\n  run:  $(show_(run))")
end
