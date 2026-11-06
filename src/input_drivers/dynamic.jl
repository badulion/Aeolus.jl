# The dynamic driver (CFG-g, #362): what a run reads over time from
# outside the model. Three groups, each optional, checked for consistency:
# the initial state (profiles, or 3-D fields on the run grid), the
# boundary planes of the open faces (the child's buffer slabs over a
# `time_bc` dimension) and the large-scale and surface forcing (height-time
# tables over their own coordinates, surface series). The boundary planes
# plug into the open-boundary contract as `FileBoundaryData`, the generic
# two-snapshot `WindowedBoundaryData` with a snapshot read from the file,
# so the time blend is the one `blend_window!` of the nest.

# the large-scale height-time tables, `ls_<name>`, and their LargeScaleForcing role
const LS_TABLES = (
    (:profile_u, "m s-1", "large-scale mean u", "the nest parent's mean over the child, or the sounding"),
    (:profile_v, "m s-1", "large-scale mean v", ""),
    (:profile_theta, "K", "large-scale mean potential temperature", ""),
    (:profile_qv, "kg kg-1", "large-scale mean specific humidity", ""),
    (:profile_p, "Pa", "large-scale pressure", "a sounding's"),
    (:u_g, "m s-1", "geostrophic wind u", "GeostrophicPressureGradient"),
    (:v_g, "m s-1", "geostrophic wind v", ""),
    (:w_subs, "m s-1", "subsidence velocity", "Subsidence (global attributes subsidence_fields, subsidence_form)"),
    (:u_adv, "m s-2", "advective tendency of u", "AdvectiveTendencies"),
    (:v_adv, "m s-2", "advective tendency of v", ""),
    (:theta_adv, "K s-1", "advective tendency of potential temperature", ""),
    (:qv_adv, "kg kg-1 s-1", "advective tendency of specific humidity", ""),
    (:nudge_u, "m s-1", "nudging target of u", "ProfileNudging (global attribute nudging_tau, s)"),
    (:nudge_v, "m s-1", "nudging target of v", ""),
    (:nudge_theta, "K", "nudging target of potential temperature", ""),
    (:nudge_qv, "kg kg-1", "nudging target of specific humidity", ""))

const _LS_NAMES = Tuple(first(t) for t in LS_TABLES)

# 3-D initial fields: the variable suffix and the field
const INIT_FIELDS = ((:u, "init_u3d", "m s-1", "west face of the cell"), (:v, "init_v3d", "m s-1", "south face"),
                     (:w, "init_w3d", "m s-1", "bottom face"), (:θ, "init_theta3d", "K", "cell centre"),
                     (:qv, "init_qv3d", "kg kg-1", "cell centre"))

const _XYZ = ("x", "y", "z")

"""
    DYNAMIC_SCHEMA :: DriverSchema

The dynamic driver (`aeolus_driver = "dynamic"`, `aeolus_form =
"standard"`). Every group is optional; [`read_dynamic_driver`](@ref)
checks that what is present is consistent:

* **initial state** — profiles over `init_z` (`init_theta`, `init_qv`,
  `init_u`, `init_v`; schema only: written and read back, but no
  preprocessor fills them and no case reads them yet), or 3-D fields on
  the run grid (`init_u3d`, …, over the run's `x`, `y`, `z`, interior
  cells in storage order: u at the cell's west face, v at its south face,
  w at its bottom face), and `init_p_surface`. A sounding driver has no
  profiles: its consumer evaluates the large-scale profile tables
  (`ls_profile_*`) at its start. Global attribute `init_time`: the model
  start, ISO-8601 UTC (`""`: a case without a calendar date); every time
  coordinate is in seconds since it.
* **boundary planes** — for each open face (`west`, `east`, `south`,
  `north`, `top`) and each field with data, `bc_<face>_<var>` over
  (`bc_<face>_<var>_i`, `_j`, `_k`, `time_bc`): the child's buffer slab
  (the face plane plus the `H` halo layers, the storage points of the
  [`OpenPlane`](@ref Aeolus.BoundaryConditions.OpenPlane)), with
  attributes `field` (the field's name), `side`, `i0`, `j0`, `k0` (the
  slab's first storage index) and `halo_mode`. `time_bc_step` is the
  window length the writer asked for ending at each record (`NaN`: none).
* **large-scale forcing** — `ls_<name>` tables (see [`LS_TABLES`](@ref)),
  each over its own height and time coordinate (`ls_<name>_z`,
  `ls_<name>_time`, or a shared `time_bc`), so a table round-trips without
  resampling; and the surface series `sfc_temperature`, `sfc_pressure`
  over `time_sfc` (schema only, like the `init_*` profiles).
* **initial soil** (#521, PALM's PIDS `init_soil_t`, `init_soil_m`) —
  `init_soil_t` (K) and `init_soil_m` (m³ m⁻³) on the coordinate `zsoil`
  (m below the surface, positive down, strictly increasing: the levels'
  centres), each a profile over `zsoil` or a volume over (`x`, `y`,
  `zsoil`) on the run's horizontal grid; `NaN` (the `_FillValue`) marks a
  level or a column without data (AeolusPrep.jl's ERA5 soil marks the
  columns that are not land). What the urban site's soil starts from
  (`[ground] soil_init`).

Global attribute `dynamic_source`: `"sounding"`, `"case_table"` or
`"parent"` (an Aeolus nest parent, [`NestBoundaryRecorder`](@ref)).
"""
const DYNAMIC_SCHEMA = DriverSchema("dynamic", "standard",
    "Initial state, boundary planes and large-scale and surface forcing over time. Every group is " *
    "optional. Besides the variables below, the boundary planes are one variable `bc_<face>_<var>` per " *
    "open face (`west`, `east`, `south`, `north`, `top`) and field with data, over (`bc_<face>_<var>_i`, " *
    "`_j`, `_k`, `time_bc`): the child's buffer slab (the face plane plus the halo layers), with " *
    "attributes `field`, `side`, `i0`, `j0`, `k0` (the slab's first storage index) and `halo_mode`.",
    vcat([DriverVariable("x", ("x",), Float64; units = "m", required = false, long_name = "x of the cell centres",
                         doc = "the run mesh's (with 3-D fields or boundary planes)"),
          DriverVariable("y", ("y",), Float64; units = "m", required = false, long_name = "y of the cell centres"),
          DriverVariable("z", ("z",), Float64; units = "m", required = false, long_name = "z of the cell centres"),
          DriverVariable("init_z", ("init_z",), Float64; units = "m", required = false,
                         long_name = "height of the initial profiles", doc = "above the ground"),
          DriverVariable("init_theta", ("init_z",), Float64; units = "K", required = false,
                         long_name = "initial potential temperature", range = (100, 1000)),
          DriverVariable("init_qv", ("init_z",), Float64; units = "kg kg-1", required = false,
                         long_name = "initial specific humidity", range = (0, 1)),
          DriverVariable("init_u", ("init_z",), Float64; units = "m s-1", required = false, long_name = "initial u"),
          DriverVariable("init_v", ("init_z",), Float64; units = "m s-1", required = false, long_name = "initial v"),
          DriverVariable("init_p_surface", (), Float64; units = "Pa", required = false,
                         long_name = "initial pressure at the ground", range = (1e3, 2e5))],
         [DriverVariable(n, _XYZ, Float64; units = u, required = false, long_name = "initial $(f) (3-D)",
                         doc = "at the $d") for (f, n, u, d) in INIT_FIELDS],
         [DriverVariable("time_bc", ("time_bc",), Float64; required = false,
                         long_name = "time of the boundary records",
                         doc = "s since init_time (CF units `seconds since <init_time>`); strictly increasing"),
          DriverVariable("time_bc_step", ("time_bc",), Float64; units = "s", required = false,
                         long_name = "window length ending at the record", doc = "the step the writer took; NaN: none")],
         [DriverVariable("ls_$n", ("*", "*"), Float64; units = u, required = false, long_name = l,
                         doc = "over (its height, its time) coordinate" * (isempty(r) ? "" : "; " * r))
          for (n, u, l, r) in LS_TABLES],
         [DriverVariable("time_sfc", ("time_sfc",), Float64; required = false,
                         long_name = "time of the surface series", doc = "s since init_time"),
          DriverVariable("sfc_temperature", ("time_sfc",), Float64; units = "K", required = false,
                         long_name = "surface (skin or screen-level) potential temperature", range = (100, 1000)),
          DriverVariable("sfc_pressure", ("time_sfc",), Float64; units = "Pa", required = false,
                         long_name = "surface pressure", range = (1e3, 2e5))],
         [DriverVariable("zsoil", ("zsoil",), Float64; units = "m", required = false,
                         long_name = "depth of the initial soil levels",
                         doc = "below the surface, positive down, strictly increasing (the levels' centres)",
                         range = (0, 100)),
          DriverVariable("init_soil_t", ("zsoil",), Float64; units = "K", required = false,
                         long_name = "initial soil temperature", range = (150, 400),
                         alt_dims = (("x", "y", "zsoil"),), doc = "a profile or a volume; NaN: no data"),
          DriverVariable("init_soil_m", ("zsoil",), Float64; units = "m3 m-3", required = false,
                         long_name = "initial volumetric soil water content", range = (0, 1),
                         alt_dims = (("x", "y", "zsoil"),), doc = "a profile or a volume; NaN: no data")]);
    attributes = [
        "init_time" => "the model start, ISO-8601 UTC (`\"\"`: a case without a calendar date); every time coordinate is in s since it",
        "dynamic_source" => "`\"sounding\"`, `\"case_table\"` or `\"parent\"` (an Aeolus nest parent)",
        "sounding_station, sounding_elevation, sounding_launches" =>
            "a sounding driver's station name, station elevation (m) and number of launches",
        "subsidence_fields, subsidence_form" =>
            "with `ls_w_subs`: the fields subsidence acts on (space-separated) and its form (`Subsidence`)",
        "nudging_tau" => "with `ls_nudge_*`: the nudging time scale (s, `ProfileNudging`)",
        "soil_source, soil_time, soil_layers" =>
            "with `init_soil_*`: where the soil profiles come from (e.g. AeolusPrep.jl's ERA5 soil), " *
            "their valid time and the source's layers (informative)"])

# ---------------------------------------------------------------------------
# boundary snapshots read from the file
# ---------------------------------------------------------------------------

# One buffer slab: the field, the face and its storage-index ranges.
struct _SlabLayout
    var::String
    field::Symbol
    side::Symbol
    ranges::NTuple{3,UnitRange{Int}}
    mode::Symbol
end

struct _Slab
    ranges::NTuple{3,UnitRange{Int}}
    data::Array{Float64,3}
end

"""
    FileSnapshot

One `time_bc` record of a dynamic driver's boundary planes: per field, the
buffer slabs of its open faces at their storage-index ranges. The snapshot
type of [`FileBoundaryData`](@ref): `snapshot_values(s, name)` is a
[`SlabArray`](@ref), a storage-indexed 3-D view backed by the slabs, so the
time blend is `blend_window!` reading the plane points, as for the nest.
"""
struct FileSnapshot
    slabs::Dict{Symbol,Vector{_Slab}}
    dims::NTuple{3,Int}
end

"""
    SlabArray <: AbstractArray{Float64,3}

The storage-indexed values of one field held by a [`FileSnapshot`](@ref):
`A[i, j, k]` is the value of the slab whose storage ranges hold
`(i, j, k)`; a point no slab holds is a `BoundsError` (only the open-face
buffers exist). Where two faces' slabs overlap (the corners) they hold
the same value, written from one field.
"""
struct SlabArray <: AbstractArray{Float64,3}
    slabs::Vector{_Slab}
    dims::NTuple{3,Int}
end
Base.size(a::SlabArray) = a.dims
Base.IndexStyle(::Type{SlabArray}) = IndexCartesian()
@inline function Base.getindex(a::SlabArray, i::Int, j::Int, k::Int)
    for s in a.slabs
        r = s.ranges
        if i in r[1] && j in r[2] && k in r[3]
            return @inbounds s.data[i - first(r[1]) + 1, j - first(r[2]) + 1, k - first(r[3]) + 1]
        end
    end
    throw(BoundsError(a, (i, j, k)))
end

function snapshot_values(s::FileSnapshot, name::Symbol)
    v = get(s.slabs, name, nothing)
    v === nothing && throw(ArgumentError("FileBoundaryData: the run has an open-boundary plane for :$name, " *
                                         "but the dynamic driver has no boundary data for it " *
                                         "(it holds $(sort!(collect(keys(s.slabs)))))"))
    return SlabArray(v, s.dims)
end

"""
    FileBoundaryData = WindowedBoundaryData{FileSnapshot}

The boundary planes of a dynamic driver as open-boundary data: two
`time_bc` records (`a` at `t0`, `b` at `t0 + Δt`) blended linearly in time
by the generic [`WindowedBoundaryData`](@ref
Aeolus.BoundaryConditions.WindowedBoundaryData) (`blend_window!`). Held
and advanced by a [`FileBoundaryFeed`](@ref).
"""
const FileBoundaryData = WindowedBoundaryData{FileSnapshot}

_empty_snapshot(layout, dims) =
    FileSnapshot(Dict{Symbol,Vector{_Slab}}(f => [_Slab(l.ranges, zeros(Float64, length.(l.ranges)...))
                                                  for l in layout if l.field === f]
                                            for f in unique(l.field for l in layout)), dims)

"""
    FileBoundaryFeed

The boundary planes of a dynamic driver, read lazily: the record times
`time` (s since `init_time`), the window lengths `step` the writer took
(`time_bc_step`), the slab `layout`, and the [`FileBoundaryData`](@ref)
`data` holding **two** records (`data.a` = record `k`, `data.b` = record
`k + 1`). [`feed_window!`](@ref) moves the window (reading only the
record it does not hold yet) and [`advance_feed!`](@ref) moves it to the
window holding a time. The file is opened for each read, so the feed
holds no file handle. `reads` counts the records read.
"""
mutable struct FileBoundaryFeed
    path::String
    time::Vector{Float64}
    step::Vector{Float64}
    layout::Vector{_SlabLayout}
    dims::NTuple{3,Int}
    k::Int
    data::FileBoundaryData
    reads::Int
end

function _read_record!(snap::FileSnapshot, ds, layout, k::Int)
    cnt = Dict{Symbol,Int}()
    for l in layout
        n = cnt[l.field] = get(cnt, l.field, 0) + 1
        s = snap.slabs[l.field][n]
        copyto!(s.data, ds[l.var].var[:, :, :, k])
    end
    return snap
end

"""
    feed_window!(feed, k) -> feed

Point the feed's data at the window from record `k` to record `k + 1`
(`t0 = time[k]`, `Δt = time[k+1] − time[k]`); for the last record (or a
one-record, steady file) `b = a` and `Δt = 0`. Moving to the next window
reads one record into the buffers of the one dropped, so at most two
records are resident.
"""
function feed_window!(f::FileBoundaryFeed, k::Int)
    n = length(f.time)
    1 <= k <= n || throw(ArgumentError("feed_window!: record $k of 1:$n"))
    k == f.k && return f
    d = f.data
    NCDataset(f.path) do ds
        if f.k >= 1 && k == f.k + 1 && d.b !== d.a  # record k is held: read k + 1 only
            d.a, d.b = d.b, d.a
            if k < n
                _read_record!(d.b, ds, f.layout, k + 1); f.reads += 1
            end
        else
            d.a === d.b && (d.b = _empty_snapshot(f.layout, f.dims))
            _read_record!(d.a, ds, f.layout, k); f.reads += 1
            if k < n
                _read_record!(d.b, ds, f.layout, k + 1); f.reads += 1
            end
        end
    end
    if k == n
        d.b = d.a
        d.t0 = f.time[k]; d.Δt = 0.0
    else
        d.t0 = f.time[k]; d.Δt = f.time[k+1] - f.time[k]
    end
    f.k = k
    return f
end

"""
    feed_steady!(feed, k) -> feed

Hold record `k` as steady data (`b = a`, `Δt = 0`, `t0 = time[k]`): the
data a nest child is started from before its first window.
"""
function feed_steady!(f::FileBoundaryFeed, k::Int)
    feed_window!(f, k)
    f.data.b = f.data.a; f.data.Δt = 0.0
    f.k = -k                                      # the next feed_window! reads afresh
    return f
end

"""
    feed_coverage(feed) -> (t_first, t_last)

The time interval the boundary records cover (s since `init_time`). A
one-record file is steady and covers every time (`(-Inf, Inf)`).
"""
feed_coverage(f::FileBoundaryFeed) = length(f.time) == 1 ? (-Inf, Inf) : (first(f.time), last(f.time))

"""
    check_coverage(feed, t0, t1; path = feed.path)

Error, printing both intervals, unless the boundary records cover the run
interval `[t0, t1]` (to 1e-9 of its length).
"""
function check_coverage(f::FileBoundaryFeed, t0::Real, t1::Real)
    a, b = feed_coverage(f)
    tol = 1e-9 * max(abs(t1 - t0), 1.0)
    (a <= t0 + tol && t1 <= b + tol) && return nothing
    error("the dynamic driver $(f.path) does not cover the run: its time_bc spans [$a, $b] s, " *
          "the run needs [$(Float64(t0)), $(Float64(t1))] s (seconds since init_time)")
end

"""
    advance_feed!(feed, t) -> feed

Move the window to the one holding time `t` (`time[k] ≤ t ≤ time[k+1]`);
an error if `t` lies outside the records.
"""
function advance_feed!(f::FileBoundaryFeed, t::Real)
    n = length(f.time)
    n == 1 && return feed_window!(f, 1)
    check_coverage(f, t, t)
    k = clamp(searchsortedlast(f.time, Float64(t)), 1, n - 1)
    return feed_window!(f, k)
end

"""
    check_planes(feed, planes) -> nothing

Error unless the driver has a slab for every non-mirror open-boundary
plane of the run (`OpenPlane`s, e.g. `st.planes` of the run's open
boundary state) at that plane's storage ranges: a face missing for a
configured open side, or a slab of another size or place (another grid),
names the plane.
"""
function check_planes(f::FileBoundaryFeed, planes)
    for p in planes
        p.mode === :mirror && continue
        r = (p.irange, p.jrange, p.krange)
        i = findfirst(l -> l.field === p.name && l.side === p.side, f.layout)
        i === nothing && error("the dynamic driver $(f.path) has no boundary data for :$(p.name) on the " *
                               "$(p.side) face, which the run has open (it holds " *
                               join(sort!(unique(["$(l.field)@$(l.side)" for l in f.layout])), ", ") * ")")
        f.layout[i].ranges == r ||
            error("the dynamic driver $(f.path): the $(p.side) slab of :$(p.name) covers storage indices " *
                  "$(f.layout[i].ranges), the run's plane $r (another grid or halo; re-run the preprocessor)")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# what a run reads
# ---------------------------------------------------------------------------

"""
    DynamicInputs

What a run reads from a dynamic driver ([`read_dynamic_driver`](@ref)) or
a preprocessor makes in memory ([`sounding_dynamic_inputs`](@ref)):

* `init_time` — the model start (ISO-8601 UTC, `""`: none); every time
  below is in seconds since it;
* `profiles` — the initial profiles `(z, theta, qv, u, v)` (each a vector,
  or `nothing`), or `nothing`;
* `p_surface` — the initial pressure at the ground (Pa), or `nothing`;
* `fields` — 3-D initial fields by name (`:u`, `:v`, `:w`, `:θ`, `:qv`),
  interior cells of the run grid;
* `tables` — the large-scale `ForcingTable`s by name (`:profile_theta`,
  `:u_g`, `:theta_adv`, …, see [`LS_TABLES`](@ref));
* `surface` — `(t, temperature, pressure)` series (a vector or `nothing`
  each), or `nothing`;
* `soil` — the initial soil `(z, t, m)`: the depths `zsoil` (m, positive
  down) and the temperature (K) and water content (m³ m⁻³), each a vector
  over `z` (a profile), an `(Nx, Ny, Nz)` array (a volume) or `nothing`;
  or `nothing`;
* `boundary` — the [`FileBoundaryFeed`](@ref) of the boundary planes, or
  `nothing`;
* `attributes` — the file's global attributes (in memory: what a writer
  records, e.g. `dynamic_source`, `nudging_tau`).
"""
struct DynamicInputs
    init_time::String
    profiles::Union{Nothing,NamedTuple}
    p_surface::Union{Nothing,Float64}
    fields::Dict{Symbol,Array{Float64,3}}
    tables::Dict{Symbol,ForcingTable{Float64}}
    surface::Union{Nothing,NamedTuple}
    soil::Union{Nothing,NamedTuple}
    boundary::Union{Nothing,FileBoundaryFeed}
    attributes::Dict{String,Any}
end

DynamicInputs(; init_time::AbstractString = "", profiles = nothing, p_surface = nothing,
              fields = Dict{Symbol,Array{Float64,3}}(), tables = Dict{Symbol,ForcingTable{Float64}}(),
              surface = nothing, soil = nothing, boundary = nothing, attributes = Dict{String,Any}()) =
    DynamicInputs(String(init_time), profiles, p_surface === nothing ? nothing : Float64(p_surface),
                  Dict{Symbol,Array{Float64,3}}(fields), Dict{Symbol,ForcingTable{Float64}}(tables),
                  surface, _soil_checked(soil), boundary, Dict{String,Any}(attributes))

# the initial soil (z, t, m), checked: z positive and strictly increasing,
# each of t and m a profile over z or a volume over (x, y, z)
function _soil_checked(soil)
    soil === nothing && return nothing
    z = Float64.(collect(soil.z))
    (!isempty(z) && z[1] > 0 && all(>(0), diff(z))) ||
        throw(ArgumentError("dynamic driver: zsoil must be positive (m below the surface) and strictly increasing, got $z"))
    chk(A, n) = A === nothing ? nothing :
        (ndims(A) == 1 || ndims(A) == 3) && size(A, ndims(A)) == length(z) ? Array{Float64}(A) :
        throw(ArgumentError("dynamic driver: $n is $(size(A)), expected ($(length(z)),) or (Nx, Ny, $(length(z)))"))
    t, m = chk(soil.t, "init_soil_t"), chk(soil.m, "init_soil_m")
    (t === nothing && m === nothing) && throw(ArgumentError("dynamic driver: zsoil without init_soil_t or init_soil_m"))
    return (z = z, t = t, m = m)
end

"""
    shift_table(tab, Δt) -> ForcingTable

`tab` with its time coordinate moved by `Δt` (s): a table on the
"seconds since init_time" axis of a driver onto another time axis (a
sounding's epoch, a case's model time). Exact for times and shifts that
are whole seconds below 2⁵³.
"""
shift_table(tab::ForcingTable, Δt::Real) = ForcingTable(tab.z, tab.t .+ Float64(Δt), tab.values)

"""
    dynamic_forcing(inputs; t0 = 0) -> NamedTuple

The large-scale forcing of the driver as `LargeScaleForcing` objects
(times: seconds since `init_time`, plus `t0`, [`shift_table`](@ref): the
run's model time at the start): `U_g`, `V_g` (`ForcingTable`s or
`nothing`) and `extra`, a tuple of [`AdvectiveTendencies`](@ref
Aeolus.LargeScaleForcing.AdvectiveTendencies) (from `ls_*_adv`),
[`Subsidence`](@ref Aeolus.LargeScaleForcing.Subsidence) (`ls_w_subs`,
global attributes `subsidence_fields`, `subsidence_form`) and
[`ProfileNudging`](@ref Aeolus.LargeScaleForcing.ProfileNudging)
(`ls_nudge_*`, global attribute `nudging_tau`), in that order, each only
when the driver has it.
"""
function dynamic_forcing(d::DynamicInputs; t0::Real = 0.0)
    T = Dict{Symbol,ForcingTable{Float64}}(k => (t0 == 0 ? v : shift_table(v, t0)) for (k, v) in d.tables)
    g(k) = get(T, k, nothing)
    extra = Any[]
    if any(k -> haskey(T, k), (:u_adv, :v_adv, :theta_adv, :qv_adv))
        push!(extra, AdvectiveTendencies(; θ = g(:theta_adv), q = g(:qv_adv), u = g(:u_adv), v = g(:v_adv)))
    end
    if haskey(T, :w_subs)
        a = d.attributes
        fs = haskey(a, "subsidence_fields") ? Tuple(Symbol.(split(a["subsidence_fields"]))) : nothing
        push!(extra, Subsidence(T[:w_subs]; fields = fs, form = Symbol(get(a, "subsidence_form", "mean_gradient"))))
    end
    ks = [(f, k) for (f, k) in ((:u, :nudge_u), (:v, :nudge_v), (:θ, :nudge_theta), (:qv, :nudge_qv)) if haskey(T, k)]
    if !isempty(ks)
        haskey(d.attributes, "nudging_tau") ||
            error("dynamic driver: nudging targets without the global attribute nudging_tau")
        push!(extra, ProfileNudging(NamedTuple{Tuple(first.(ks))}(Tuple(T[k] for (_, k) in ks)),
                                    Float64(d.attributes["nudging_tau"])))
    end
    return (U_g = g(:u_g), V_g = g(:v_g), extra = Tuple(extra))
end

# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

_tunits(init_time) = isempty(init_time) ? "s" : "seconds since " * init_time
_ascii_name(n::Symbol) = n === :θ ? "theta" : String(n)

function _defvar1!(ds, name, A, attrs)
    v = NCDatasets.defVar(ds, name, Float64, (name,); attrib = collect(attrs))
    v[:] = A
    return v
end

function _write_table!(ds, name::Symbol, tab::ForcingTable, init_time; tdim = nothing)
    v = DYNAMIC_SCHEMA["ls_$name"]
    zd = "ls_$(name)_z"
    td = tdim === nothing ? "ls_$(name)_time" : tdim
    ds.dim[zd] = length(tab.z)
    _defvar1!(ds, zd, tab.z, Dict{String,Any}("units" => "m", "long_name" => "height of ls_$name", "axis" => "Z"))
    if tdim === nothing
        ds.dim[td] = length(tab.t)
        _defvar1!(ds, td, tab.t, Dict{String,Any}("units" => _tunits(init_time), "long_name" => "time of ls_$name"))
    end
    var = nc_defvar!(ds, v.name, Float64, (zd, td); attributes = Dict{String,Any}("units" => v.units,
                     "long_name" => v.long_name), chunksizes = [length(tab.z), tdim === nothing ? length(tab.t) : 1])
    tdim === nothing && (var[:, :] = tab.values)
    return var
end

"""
    write_dynamic_driver(path, inputs::DynamicInputs; mesh = nothing, source = "",
                         preprocessor, origin_lat = NaN, origin_lon = NaN,
                         origin_x = NaN, origin_y = NaN, dx = NaN, dy = NaN) -> path

Write `inputs` (its profiles, `p_surface`, 3-D fields, tables and surface
series; not its boundary feed, see [`NestBoundaryRecorder`](@ref) for
streamed planes) as a dynamic driver ([`DYNAMIC_SCHEMA`](@ref)) with the
conventions' global attributes (`source` hashed) plus `inputs.attributes`
and `init_time`. `mesh` (needed with 3-D fields) gives the `x`, `y`, `z`
coordinates and `dx`, `dy`.
"""
function write_dynamic_driver(path::AbstractString, d::DynamicInputs; mesh = nothing, source::AbstractString = "",
                              preprocessor::AbstractString = "",
                              origin_lat = NaN, origin_lon = NaN, origin_x = NaN, origin_y = NaN,
                              dx = NaN, dy = NaN)
    ds = _create_dynamic(path, d.init_time, d.attributes; mesh, source, preprocessor, origin_lat, origin_lon,
                         origin_x, origin_y, dx, dy)
    try
        _write_static_part!(ds, d)
    finally
        close(ds)
    end
    return path
end

function _create_dynamic(path, init_time, extra; mesh, source, preprocessor, origin_lat, origin_lon,
                         origin_x, origin_y, dx, dy)
    mesh === nothing || (dx = mesh.dx; dy = mesh.dy)
    attrs = driver_attributes("dynamic", "standard"; origin_lat, origin_lon, origin_x, origin_y, dx, dy,
                              source, out = path, preprocessor)
    attrs["init_time"] = String(init_time)
    for (k, v) in extra
        attrs[k] = v
    end
    mkpath(dirname(abspath(path)))
    ds = nc_create(path; attributes = attrs)
    if mesh !== nothing
        for (n, c, ax) in (("x", xnodes(mesh, CellCenter()), "X"), ("y", ynodes(mesh, CellCenter()), "Y"),
                           ("z", znodes(mesh, CellCenter()), "Z"))
            ds.dim[n] = length(c)
            _defvar1!(ds, n, collect(Float64, c), Dict{String,Any}("units" => "m", "axis" => ax,
                                                                  "long_name" => "$n of the cell centres"))
        end
    end
    return ds
end

function _write_static_part!(ds, d::DynamicInputs)
    S = DYNAMIC_SCHEMA
    p = d.profiles
    if p !== nothing
        ds.dim["init_z"] = length(p.z)
        _defvar1!(ds, "init_z", p.z, Dict{String,Any}("units" => "m", "long_name" => S["init_z"].long_name))
        for n in (:theta, :qv, :u, :v)
            a = getproperty(p, n)
            a === nothing && continue
            v = S["init_$n"]
            NCDatasets.defVar(ds, v.name, Float64, ("init_z",);
                              attrib = ["units" => v.units, "long_name" => v.long_name])[:] = a
        end
    end
    if d.p_surface !== nothing
        v = S["init_p_surface"]
        NCDatasets.defVar(ds, v.name, Float64, (); attrib = ["units" => v.units, "long_name" => v.long_name])[] =
            d.p_surface
    end
    for (f, n, u, doc) in INIT_FIELDS
        A = get(d.fields, f, nothing)
        A === nothing && continue
        haskey(ds.dim, "x") || error("write_dynamic_driver: 3-D initial fields need the mesh")
        size(A) == (Int(ds.dim["x"]), Int(ds.dim["y"]), Int(ds.dim["z"])) ||
            error("write_dynamic_driver: $n is $(size(A)), the mesh $((ds.dim["x"], ds.dim["y"], ds.dim["z"]))")
        var = nc_defvar!(ds, n, Float64, _XYZ; attributes = Dict{String,Any}("units" => u,
                         "long_name" => "initial $f", "comment" => "at the $doc"))
        var[:, :, :] = A
    end
    for name in _LS_NAMES
        haskey(d.tables, name) && _write_table!(ds, name, d.tables[name], d.init_time)
    end
    s = d.surface
    if s !== nothing
        ds.dim["time_sfc"] = length(s.t)
        _defvar1!(ds, "time_sfc", s.t, Dict{String,Any}("units" => _tunits(d.init_time),
                                                       "long_name" => S["time_sfc"].long_name))
        for (n, a) in (("sfc_temperature", s.temperature), ("sfc_pressure", s.pressure))
            a === nothing && continue
            v = S[n]
            NCDatasets.defVar(ds, n, Float64, ("time_sfc",); attrib = ["units" => v.units,
                                                                       "long_name" => v.long_name])[:] = a
        end
    end
    so = d.soil
    if so !== nothing
        ds.dim["zsoil"] = length(so.z)
        _defvar1!(ds, "zsoil", so.z, Dict{String,Any}("units" => "m", "positive" => "down",
                                                     "long_name" => S["zsoil"].long_name))
        for (n, A) in (("init_soil_t", so.t), ("init_soil_m", so.m))
            A === nothing && continue
            v = S[n]
            if ndims(A) == 3
                haskey(ds.dim, "x") || error("write_dynamic_driver: a soil volume ($n) needs the mesh")
                size(A)[1:2] == (Int(ds.dim["x"]), Int(ds.dim["y"])) ||
                    error("write_dynamic_driver: $n is $(size(A)), the mesh $((ds.dim["x"], ds.dim["y"]))")
            end
            dims = ndims(A) == 3 ? ("x", "y", "zsoil") : ("zsoil",)
            var = NCDatasets.defVar(ds, n, Float64, dims; fillvalue = NaN,
                                    attrib = ["units" => v.units, "long_name" => v.long_name])
            var.var[ntuple(_ -> Colon(), ndims(A))...] = A
        end
    end
    return ds
end

# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

_f64(v) = Float64.(collect(v))

function _read_table(ds, name)
    var = ds[name]
    zd, td = String.(NCDatasets.dimnames(var))
    (haskey(ds, zd) && haskey(ds, td)) ||
        error("dynamic driver: $name is over ($zd, $td), which need coordinate variables of those names")
    return ForcingTable(_f64(ds[zd].var[:]), _f64(ds[td].var[:]), Array{Float64}(var.var[:, :]))
end

# The boundary-slab layout of a file: every bc_* variable.
function _layout(ds)
    out = _SlabLayout[]
    for name in keys(ds)
        startswith(name, "bc_") || continue
        v = ds[name]; a = v.attrib
        dn = String.(NCDatasets.dimnames(v))
        (length(dn) == 4 && dn[4] == "time_bc") ||
            error("dynamic driver: $name is over $(Tuple(dn)), expected (i, j, k, time_bc)")
        for key in ("field", "side", "i0", "j0", "k0", "halo_mode")
            haskey(a, key) || error("dynamic driver: $name has no attribute $key")
        end
        sz = size(v)[1:3]
        r = ntuple(d -> Int(a[("i0", "j0", "k0")[d]]):Int(a[("i0", "j0", "k0")[d]]) + sz[d] - 1, 3)
        push!(out, _SlabLayout(name, Symbol(a["field"]), Symbol(a["side"]), r, Symbol(a["halo_mode"])))
    end
    return out
end

"""
    read_dynamic_driver(path; mesh = nothing, origin_lat, origin_lon, origin_x, origin_y,
                        check_source = true) -> DynamicInputs

Read the dynamic driver at `path`. It is checked against
[`DYNAMIC_SCHEMA`](@ref) ([`check_driver`](@ref): kind, schema version,
attributes, units, ranges, a stale source) and for consistency: an
initial profile needs `init_z`; 3-D fields and boundary planes need the
run `mesh` and the `x`, `y`, `z` of the file must be the run's (an error
naming both grids; with the anchor keywords, the anchor too); `time_bc`
must increase strictly. The boundary planes become a
[`FileBoundaryFeed`](@ref) positioned on its first window (two records
read). `planes = false` reads only the profiles, tables and series (no
mesh needed): what a consumer of a sounding uses.
"""
function read_dynamic_driver(path::AbstractString; mesh = nothing, origin_lat = nothing, origin_lon = nothing,
                             origin_x = nothing, origin_y = nothing, planes::Bool = true)
    _require_valid(path, DYNAMIC_SCHEMA)
    ds = NCDataset(path)
    try
        a = Dict{String,Any}(k => v for (k, v) in ds.attrib)
        init_time = String(get(a, "init_time", ""))
        has(n) = haskey(ds, n)
        profiles = nothing
        if any(n -> has("init_$n"), ("theta", "qv", "u", "v"))
            has("init_z") || error("the dynamic driver $path has initial profiles but no init_z")
            pv(n) = has("init_$n") ? _f64(ds["init_$n"].var[:]) : nothing
            profiles = (z = _f64(ds["init_z"].var[:]), theta = pv("theta"), qv = pv("qv"), u = pv("u"), v = pv("v"))
        end
        ps = has("init_p_surface") ? Float64(ds["init_p_surface"].var[]) : nothing
        layout = planes ? _layout(ds) : _SlabLayout[]
        fields = Dict{Symbol,Array{Float64,3}}()
        need_grid = planes && (!isempty(layout) || any(t -> has(t[2]), INIT_FIELDS))
        if need_grid
            mesh === nothing && error("the dynamic driver $path holds 3-D fields or boundary planes: " *
                                      "read it with the run's mesh (mesh = …)")
            all(has, _XYZ) || error("the dynamic driver $path holds 3-D fields or boundary planes but no x, y, z")
            _check_dynamic_grid(path, ds, mesh; origin_lat, origin_lon, origin_x, origin_y)
        end
        for (f, n, _, _) in INIT_FIELDS
            planes && has(n) && (fields[f] = Array{Float64}(ds[n].var[:, :, :]))
        end
        tables = Dict{Symbol,ForcingTable{Float64}}(n => _read_table(ds, "ls_$n") for n in _LS_NAMES if has("ls_$n"))
        surface = nothing
        if has("sfc_temperature") || has("sfc_pressure")
            has("time_sfc") || error("the dynamic driver $path has surface series but no time_sfc")
            sv(n) = has(n) ? _f64(ds[n].var[:]) : nothing
            surface = (t = _f64(ds["time_sfc"].var[:]), temperature = sv("sfc_temperature"), pressure = sv("sfc_pressure"))
        end
        soil = nothing
        if has("init_soil_t") || has("init_soil_m")
            has("zsoil") || error("the dynamic driver $path has initial soil but no zsoil")
            soil = _read_soil(path, ds, mesh)
        end
        feed = nothing
        if !isempty(layout)
            has("time_bc") || error("the dynamic driver $path has boundary planes but no time_bc")
            t = _f64(ds["time_bc"].var[:])
            isempty(t) && error("the dynamic driver $path has boundary planes but no records (time_bc is empty)")
            all(>(0), diff(t)) || error("the dynamic driver $path: time_bc is not strictly increasing")
            st = has("time_bc_step") ? _f64(ds["time_bc_step"].var[:]) : fill(NaN, length(t))
            dims = (mesh.Nx + 2mesh.Hx, mesh.Ny + 2mesh.Hy, mesh.Nz + 2mesh.Hz)
            for l in layout
                all(d -> first(l.ranges[d]) >= 1 && last(l.ranges[d]) <= dims[d], 1:3) ||
                    error("the dynamic driver $path: the slab $(l.var) covers storage indices $(l.ranges), " *
                          "outside the run's storage $dims (another grid or halo)")
            end
            e = _empty_snapshot(layout, dims)
            feed = FileBoundaryFeed(String(path), t, st, layout, dims, 0,
                                    WindowedBoundaryData(e, _empty_snapshot(layout, dims), 0.0, 0.0), 0)
        end
        d = DynamicInputs(init_time, profiles, ps, fields, tables, surface, soil, feed, a)
        feed === nothing || feed_window!(feed, 1)
        return d
    finally
        close(ds)
    end
end

# The initial soil of a file: zsoil and init_soil_t / init_soil_m, a profile
# or a volume (whose x, y must be the run's: read with the mesh). The fill
# values are NaN in memory.
function _read_soil(path, ds, mesh)
    z = _f64(ds["zsoil"].var[:])
    function rd(n)
        haskey(ds, n) || return nothing
        v = ds[n]
        A = Array{Float64}(v.var[ntuple(_ -> Colon(), ndims(v))...])
        fv = get(v.attrib, "_FillValue", nothing)
        fv === nothing || isnan(fv) || replace!(A, Float64(fv) => NaN)
        if ndims(A) == 3
            mesh === nothing && error("the dynamic driver $path holds a soil volume ($n): read it with the run's mesh (mesh = …)")
            (haskey(ds, "x") && haskey(ds, "y")) || error("the dynamic driver $path holds a soil volume ($n) but no x, y")
            xs = collect(Float64, xnodes(mesh, CellCenter())); ys = collect(Float64, ynodes(mesh, CellCenter()))
            (_f64(ds["x"].var[:]) == xs && _f64(ds["y"].var[:]) == ys) ||
                error("the dynamic driver $path: the soil volume $n is not on the run's horizontal grid " *
                      "($(size(A, 1)) × $(size(A, 2)) cells, the run $(length(xs)) × $(length(ys))): re-run the preprocessor for this grid")
        end
        return A
    end
    try
        return _soil_checked((z = z, t = rd("init_soil_t"), m = rd("init_soil_m")))
    catch e
        e isa ArgumentError && error("the dynamic driver $path: $(e.msg)")
        rethrow()
    end
end

function _check_dynamic_grid(path, ds, mesh; origin_lat, origin_lon, origin_x, origin_y)
    if origin_lat === nothing
        xs = collect(Float64, xnodes(mesh, CellCenter())); ys = collect(Float64, ynodes(mesh, CellCenter()))
        fx = _f64(ds["x"].var[:]); fy = _f64(ds["y"].var[:])
        (fx == xs && fy == ys) ||
            error("the input driver $path is not on the run's grid (no resampling: re-run the preprocessor " *
                  "for this grid):\n  file: $(length(fx)) × $(length(fy)) cells, x from $(first(fx)) m, " *
                  "y from $(first(fy)) m\n  run:  $(length(xs)) × $(length(ys)) cells, x from $(first(xs)) m, " *
                  "y from $(first(ys)) m")
    else
        check_grid(path, ds, mesh; origin_lat, origin_lon, origin_x, origin_y)
    end
    zs = collect(Float64, znodes(mesh, CellCenter())); fz = _f64(ds["z"].var[:])
    fz == zs || error("the input driver $path is not on the run's grid: its $(length(fz)) levels " *
                      "(first centre $(first(fz)) m) are not the run's $(length(zs)) (first centre $(first(zs)) m)")
    return nothing
end

