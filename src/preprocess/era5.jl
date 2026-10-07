# The ERA5 preprocessor of the dynamic driver (CFG-h, #363; shared with
# Era VIII #57): ERA5 on pressure or model levels, GRIB2 (GRIBDatasets) or
# NetCDF (NCDatasets), as the CFG-g dynamic driver — initial profiles,
# boundary planes of the open faces, large-scale profiles, geostrophic
# wind, subsidence, advective tendencies and the surface series.
#
#   e = read_era5(["era5_pl.nc", "era5_sfc.nc"])
#   d = era5_dynamic_inputs(e; lat = 52.21, lon = 14.12, z = zc)     # in memory
#   write_era5_driver("dynamic.nc", e; lat, lon, mesh, planes)       # the file
#
# Every step is a plain, exact-for-linear-data operation (bilinear in
# lat/lon, linear in z, centred differences over a ±box), so the tests can
# assert it against analytic fields to rounding.

const _G0 = DEFAULT_CONSTANTS.grav                  # 9.80665 m s⁻², the WMO standard gravity of Φ = g₀ z
const _EARTH_OMEGA = 7.292115e-5                    # s⁻¹, Earth's rotation rate (IERS)
const ERA5_OUTPUTS = (:profiles, :planes, :large_scale, :geostrophic, :subsidence, :advection, :surface)

"""
    ERA5Data

ERA5 read by [`read_era5`](@ref), on one regular lat/lon grid and one time
axis. Arrays are `(lon, lat, level, time)` (3-D fields) or `(lon, lat,
time)` (surface fields), longitudes and latitudes increasing, levels from
the bottom (highest pressure) up:

* `lon`, `lat` (degrees), `time` (`DateTime`, UTC valid times), `levels`
  (`:pressure` or `:model`);
* `p` (Pa) and `z` (m above mean sea level, the geopotential height
  `Φ/g₀`) of every level point;
* `T` (K), `q` (kg kg⁻¹, specific humidity), `u`, `v` (m s⁻¹, east and
  north) and `omega` (Pa s⁻¹, ERA5's `w`); a field the files do not hold
  is `nothing`;
* `sp` (Pa), `skt`, `t2m` (K) or `nothing`, and `zs` (m, the surface
  geopotential height of the first time) or `nothing`;
* `sources` — the files read.
"""
struct ERA5Data
    lon::Vector{Float64}
    lat::Vector{Float64}
    time::Vector{Dates.DateTime}
    levels::Symbol
    p::Array{Float64,4}
    z::Array{Float64,4}
    T::Array{Float64,4}
    q::Union{Nothing,Array{Float64,4}}
    u::Union{Nothing,Array{Float64,4}}
    v::Union{Nothing,Array{Float64,4}}
    omega::Union{Nothing,Array{Float64,4}}
    sp::Union{Nothing,Array{Float64,3}}
    skt::Union{Nothing,Array{Float64,3}}
    t2m::Union{Nothing,Array{Float64,3}}
    zs::Union{Nothing,Matrix{Float64}}
    sources::Vector{String}
end

# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

const _CDM = NCDatasets.CommonDataModel
const _LON_DIMS = ("longitude", "lon")
const _LAT_DIMS = ("latitude", "lat")
const _TIME_DIMS = ("valid_time", "time")
const _PLEV_DIMS = ("pressure_level", "isobaricInhPa", "level", "plev")
const _MLEV_DIMS = ("model_level", "hybrid", "level")
const _LEVEL_VARS = ("t", "q", "u", "v", "w", "z")
const _SURFACE_VARS = ("sp", "skt", "t2m", "z", "lnsp")

_isgrib(path) = open(io -> (b = read(io, 4); length(b) == 4 && String(b) == "GRIB"), path)

function _open_era5(f, path::AbstractString)
    isfile(path) || error("read_era5: no file $path")
    if _isgrib(path)
        ds = GRIBDatasets.GRIBDataset(path)
        try
            return f(ds)
        finally
            close(ds)
        end
    end
    return NCDataset(f, path)
end

_findname(names, cands) = (i = findfirst(in(names), cands); i === nothing ? nothing : cands[i])
_nan(A) = Float64[ismissing(x) ? NaN : Float64(x) for x in A]

# The level coordinate in Pa (pressure levels) or as model-level numbers.
function _level_values(ds, name, kind)
    v = ds[name]
    x = _nan(collect(v[:]))
    kind === :model && return x
    u = lowercase(String(get(v.attrib, "units", "")))
    u in ("pa",) && return x
    (u in ("hpa", "millibars", "mbar", "mb") || maximum(x) <= 1100) && return 100 .* x
    error("read_era5: the pressure levels $name have units \"$u\" (expected hPa or Pa)")
end

function _times(ds, name)
    t = collect(ds[name][:])
    all(x -> x isa Dates.DateTime, t) ||
        error("read_era5: the time coordinate $name was not decoded as dates (got $(eltype(t)))")
    return Dates.DateTime[x for x in t]
end

# One variable as (lon, lat, level, time) (`level = false`: (lon, lat, time)),
# with its coordinates.
function _read_var(ds, name, kind)
    v = ds[name]
    dn = String.(collect(_CDM.dimnames(v)))
    lo = _findname(dn, _LON_DIMS); la = _findname(dn, _LAT_DIMS)
    (lo === nothing || la === nothing) &&
        error("read_era5: $name is over $(Tuple(dn)): expected a regular lat/lon grid (longitude, latitude)")
    lv = _findname(dn, kind === :model ? _MLEV_DIMS : _PLEV_DIMS)
    tm = _findname(dn, _TIME_DIMS)
    other = setdiff(dn, filter(!isnothing, [lo, la, lv, tm]))
    A = _nan(Array(v))
    # drop singleton dimensions that are not ours (a GRIB `step`, `number`, …)
    for o in other
        size(A, findfirst(==(o), dn)) == 1 ||
            error("read_era5: $name has the dimension $o of length $(size(A, findfirst(==(o), dn))) " *
                  "(an ensemble or forecast steps); give one member and analysis times")
    end
    keep = [i for (i, d) in enumerate(dn) if !(d in other)]
    A = reshape(A, size(A)[keep]...); dn = dn[keep]
    order = Int[findfirst(==(lo), dn), findfirst(==(la), dn)]
    lv === nothing || push!(order, findfirst(==(lv), dn))
    tm === nothing || push!(order, findfirst(==(tm), dn))
    A = permutedims(A, order)
    lv === nothing && (A = reshape(A, size(A, 1), size(A, 2), 1, :))
    tm === nothing && (A = reshape(A, size(A)..., 1))
    lon = _nan(collect(ds[lo][:])); lat = _nan(collect(ds[la][:]))
    lev = lv === nothing ? nothing : _level_values(ds, lv, kind)
    time = tm === nothing ? nothing : _times(ds, tm)
    return (data = A, lon = lon, lat = lat, lev = lev, time = time, haslevel = lv !== nothing)
end

"""
    read_era5(paths; levels = :pressure, hybrid = nothing) -> ERA5Data

Read ERA5 from the files `paths` (one or several; GRIB2 through
GRIBDatasets, NetCDF through NCDatasets, told apart by their first bytes):
`t`, `q`, `u`, `v`, `w` (ERA5's vertical velocity, the pressure velocity
ω in Pa s⁻¹) and `z` (geopotential) on levels, and the surface fields
`sp`, `skt`, `t2m` and `z` (the surface geopotential, the orography). The
variables may be spread over files (pressure levels and single levels,
as the CDS delivers them), but share one lat/lon grid and one time axis.

* `levels = :pressure` — ERA5 on pressure levels: `p` is the level, the
  height is the geopotential height `z = Φ/g₀` (g₀ = 9.80665 m s⁻²,
  [`MoistThermodynamics`](@ref Aeolus.MoistThermodynamics) `grav`).
* `levels = :model` — ERA5 on IFS model levels (L137): the half-level
  pressures `p_{k±1/2} = a_{k±1/2} + b_{k±1/2} p_s` from the hybrid
  coefficients (`hybrid = (a, b)`, the `N + 1` half-level values from the
  model top down, Pa and 1; or file variables `a_half`/`b_half` or
  `hyai`/`hybi`), `p_s` from `sp` or `lnsp`, full-level pressure the
  half-level mean, and the heights by hydrostatic integration from the
  orography ([`era5_model_level_heights`](@ref)). The levels read must be
  the bottom ones, contiguous up to level `N`.
"""
function read_era5(paths; levels::Symbol = :pressure, hybrid = nothing)
    levels in (:pressure, :model) || throw(ArgumentError("read_era5: levels = :$levels; use :pressure or :model"))
    files = paths isa AbstractString ? [String(paths)] : String.(collect(paths))
    isempty(files) && throw(ArgumentError("read_era5: no files"))
    got = Dict{String,Any}()
    hyb = hybrid
    for f in files
        _open_era5(f) do ds
            names = Set(String.(collect(keys(ds))))
            for n in names
                sfc_only = n in ("sp", "skt", "t2m", "lnsp")
                (n in _LEVEL_VARS || sfc_only) || continue
                r = _read_var(ds, n, levels)
                # the surface geopotential: `z` without levels (model levels: `z` is always the orography)
                key = n == "z" && (!r.haslevel || levels === :model) ? "zs" : n
                if key in ("zs", "sp", "skt", "t2m", "lnsp")
                    r = merge(r, (data = r.data[:, :, 1:1, :], haslevel = false))
                end
                haskey(got, key) && error("read_era5: $key is in more than one of the files $files")
                got[key] = merge(r, (file = f,))
            end
            if levels === :model && hyb === nothing
                for (a, b) in (("a_half", "b_half"), ("hyai", "hybi"))
                    (a in names && b in names) && (hyb = (_nan(ds[a][:]), _nan(ds[b][:])))
                end
            end
        end
    end
    haskey(got, "t") || error("read_era5: no temperature t in $files")
    ref = got["t"]
    ref.haslevel || error("read_era5: t has no level dimension in $files")
    ref.time === nothing && error("read_era5: t has no time coordinate")
    # one grid and one time axis
    for (k, r) in got
        (r.lon == ref.lon && r.lat == ref.lat) ||
            error("read_era5: $k ($(r.file)) is on another lat/lon grid than t ($(ref.file))")
        k == "zs" && continue                         # the orography: its first time is used
        r.time == ref.time ||
            error("read_era5: $k ($(r.file)) has other times than t: $(r.time) vs $(ref.time)")
        r.haslevel && r.lev != ref.lev && error("read_era5: $k has other levels than t: $(r.lev) vs $(ref.lev)")
    end
    # increasing lon and lat, time; levels bottom (high p) up
    io = sortperm(ref.lon); ia = sortperm(ref.lat); it = sortperm(ref.time)
    lev = ref.lev
    il = sortperm(lev; rev = true)                   # bottom first: the highest pressure, model level N
    allunique(ref.time) || error("read_era5: repeated valid times")
    ord4(A) = A[io, ia, il, it]
    ord3(A) = A[io, ia, 1, it]
    g4(k) = haskey(got, k) ? ord4(got[k].data) : nothing
    g3(k) = haskey(got, k) ? ord3(got[k].data) : nothing
    T = g4("t"); q = g4("q")
    sp = haskey(got, "sp") ? g3("sp") : haskey(got, "lnsp") ? exp.(g3("lnsp")) : nothing
    zs = haskey(got, "zs") ? got["zs"].data[io, ia, 1, 1] ./ _G0 : nothing
    nx, ny, nl, nt = size(T)
    if levels === :pressure
        p = repeat(reshape(lev[il], 1, 1, nl, 1), nx, ny, 1, nt)
        haskey(got, "z") || error("read_era5: no geopotential z on the pressure levels in $files")
        z = g4("z") ./ _G0
    else
        hyb === nothing && error("read_era5: levels = :model needs the hybrid coefficients (hybrid = (a, b), " *
                                 "or a_half/b_half in a file)")
        sp === nothing && error("read_era5: levels = :model needs the surface pressure (sp or lnsp)")
        zs === nothing && error("read_era5: levels = :model needs the surface geopotential z")
        a, b = Float64.(hyb[1]), Float64.(hyb[2])
        N = length(a) - 1
        lv = Int.(round.(lev[il]))                     # N, N-1, … (bottom up)
        lv == collect(N:-1:N-nl+1) ||
            error("read_era5: model levels $(lv) are not the bottom ones contiguous up from level $N")
        p = zeros(nx, ny, nl, nt); z = zeros(nx, ny, nl, nt)
        for t in 1:nt, j in 1:ny, i in 1:nx
            qc = q === nothing ? zeros(nl) : q[i, j, :, t]
            pf, zf = era5_model_level_heights(a, b, sp[i, j, t], zs[i, j], T[i, j, :, t], qc)
            p[i, j, :, t] = pf; z[i, j, :, t] = zf
        end
    end
    return ERA5Data(ref.lon[io], ref.lat[ia], ref.time[it], levels, p, z, T, q, g4("u"), g4("v"), g4("w"),
                    sp, g3("skt"), g3("t2m"), zs, files)
end

"""
    era5_model_level_heights(a, b, ps, zs, T, q; c = DEFAULT_CONSTANTS) -> (p, z)

Full-level pressure (Pa) and height (m above mean sea level) of the lowest
`length(T)` IFS model levels of a column, ordered from the bottom up (`T`,
`q` likewise), from the half-level hybrid coefficients `a`, `b` (the `N +
1` values from the model top down), the surface pressure `ps` and the
surface height `zs`: `p_{k±1/2} = a + b p_s`, `p_k` their mean, and
hydrostatic integration with the virtual temperature
([`virtual_temperature`](@ref Aeolus.MoistThermodynamics.virtual_temperature)),

``Φ_{k+1/2} = Φ_s + \\sum_{j>k} R_d T_{v,j} \\ln(p_{j+1/2}/p_{j-1/2})``,
``Φ_k = Φ_{k+1/2} + α_k R_d T_{v,k}``,
``α_k = 1 − \\frac{p_{k-1/2}}{Δp_k} \\ln\\frac{p_{k+1/2}}{p_{k-1/2}}`` (``α_1 = \\ln 2``).

Reference: ECMWF (2016), *IFS Documentation Cy41r2, Part III: Dynamics and
numerical procedures*, §2.2.1, eqs. 2.20–2.23.
"""
function era5_model_level_heights(a::AbstractVector, b::AbstractVector, ps::Real, zs::Real,
                                  T::AbstractVector, q::AbstractVector; c = DEFAULT_CONSTANTS)
    N = length(a) - 1
    n = length(T)
    Rd = c.gas_constant / c.molmass_dry
    ph = [a[k] + b[k] * ps for k in 1:N+1]          # half levels 1/2 … N+1/2 (top down)
    p = zeros(n); z = zeros(n)
    Φh = c.grav * zs                                 # Φ at the lower half level of model level N
    for m in 1:n                                     # m = 1: model level N (the bottom)
        k = N - m + 1
        Tv = virtual_temperature(T[m], PhasePartition(q[m]), c)
        lo, hi = ph[k+1], ph[k]                      # p_{k+1/2} (below), p_{k-1/2} (above)
        α = k == 1 ? log(2.0) : 1 - hi / (lo - hi) * log(lo / hi)
        p[m] = (lo + hi) / 2
        z[m] = (Φh + α * Rd * Tv) / c.grav
        k == 1 || (Φh += Rd * Tv * log(lo / hi))
    end
    return p, z
end

# ---------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------

# The bilinear weights of (lat, lon) on the ERA5 grid; an error (with the
# numbers) when the point lies outside it.
function _bilinear(e::ERA5Data, lat::Real, lon::Real)
    x = Float64(lon); y = Float64(lat)
    lo, hi = e.lon[1], e.lon[end]
    if !(lo <= x <= hi)                              # another longitude convention (0…360 vs −180…180)
        for s in (-360.0, 360.0)
            lo <= x + s <= hi && (x += s; break)
        end
    end
    tol = 1e-9
    (e.lon[1] - tol <= x <= e.lon[end] + tol && e.lat[1] - tol <= y <= e.lat[end] + tol) ||
        error(@sprintf("ERA5: the point (%.4f N, %.4f E) lies outside the file's lat/lon box [%.4f, %.4f] N × [%.4f, %.4f] E (%s)",
                       lat, lon, e.lat[1], e.lat[end], e.lon[1],
                       e.lon[end], join(basename.(e.sources), ", ")))
    i = clamp(searchsortedlast(e.lon, x), 1, max(length(e.lon) - 1, 1))
    j = clamp(searchsortedlast(e.lat, y), 1, max(length(e.lat) - 1, 1))
    wx = length(e.lon) == 1 ? 0.0 : clamp((x - e.lon[i]) / (e.lon[i+1] - e.lon[i]), 0.0, 1.0)
    wy = length(e.lat) == 1 ? 0.0 : clamp((y - e.lat[j]) / (e.lat[j+1] - e.lat[j]), 0.0, 1.0)
    i1 = min(i + 1, length(e.lon)); j1 = min(j + 1, length(e.lat))
    return ((i, j, (1 - wx) * (1 - wy)), (i1, j, wx * (1 - wy)), (i, j1, (1 - wx) * wy), (i1, j1, wx * wy))
end

# The bilinear value at (lat, lon): of a (lon, lat, level, time) array a
# (level, time) matrix; of a (lon, lat, time) array a vector; of a
# (lon, lat) matrix a number.
_at(A::Array{Float64,4}, W) = sum(w .* A[i, j, :, :] for (i, j, w) in W)
_at(A::Array{Float64,3}, W) = sum(w .* A[i, j, :] for (i, j, w) in W)
_at(A::Matrix{Float64}, W) = sum(w * A[i, j] for (i, j, w) in W)

"""
    era5_vertical(zl, vl, z; name = "", logarithmic = false) -> Vector

Values at the heights `z` of a column given at the level heights `zl`
(any order; m above the ground, levels below the ground, `zl < −1 µm`,
dropped): linear in z (`logarithmic`: in `ln v`, for pressure), the
lowest level's value below it, and an error naming the heights when a
`z` lies above the highest level.
"""
function era5_vertical(zl::AbstractVector, vl::AbstractVector, z::AbstractVector; name = "", logarithmic::Bool = false)
    keep = findall(>=(-1e-6), zl)                    # a level at the ground (to rounding) is kept
    isempty(keep) && error("ERA5: every level of $name lies below the ground")
    o = keep[sortperm(zl[keep])]
    zs = Float64.(zl[o]); vs = Float64.(vl[o])
    logarithmic && (vs = log.(vs))
    out = similar(z, Float64)
    for (n, zz) in enumerate(z)
        zz <= zs[end] * (1 + 1e-12) ||
            error(@sprintf("ERA5: the run's z = %.1f m lies above the ERA5 top (%s): its highest level is at %.1f m above the ground; fetch higher levels or lower the domain top",
                           zz, name, zs[end]))
        if zz <= zs[1]
            out[n] = vs[1]
        else
            i = min(searchsortedlast(zs, zz), length(zs) - 1)
            w = (zz - zs[i]) / (zs[i+1] - zs[i])
            out[n] = (1 - w) * vs[i] + w * vs[i+1]
        end
    end
    return logarithmic ? exp.(out) : out
end

# ---------------------------------------------------------------------------
# derived quantities on the levels
# ---------------------------------------------------------------------------

_theta(T, p) = potential_temperature(T, p)
_qv(e, i...) = e.q === nothing ? 0.0 : e.q[i...]

"""
    era5_subsidence(omega, T, p, q) -> w

The vertical velocity of a pressure velocity ω (Pa s⁻¹),
`w = −ω/(ρ g₀)`, with the moist-air density
([`air_density`](@ref Aeolus.MoistThermodynamics.air_density)) of `T`,
`p`, `q`: hydrostatic, exact to the ERA5 ω's own approximation.
"""
era5_subsidence(ω::Real, T::Real, p::Real, q::Real) = -ω / (air_density(T, p, PhasePartition(q)) * _G0)

"""
    coriolis_parameter(lat) -> s⁻¹

`f = 2Ω sin φ`, Ω = 7.292115 × 10⁻⁵ s⁻¹.
"""
coriolis_parameter(lat::Real) = 2 * _EARTH_OMEGA * sind(lat)

# The four points of the centred differences over the ±box (degrees) and
# their spacings (m): east-west Δx, north-south Δy, in the projection of
# `latlon_to_meters` about the centre.
function _box_points(e, lat, lon, box)
    W = (N = _bilinear(e, lat + box, lon), S = _bilinear(e, lat - box, lon),
         E = _bilinear(e, lat, lon + box), Wst = _bilinear(e, lat, lon - box))
    Δx = latlon_to_meters(lat, lon + box, lat, lon)[1] - latlon_to_meters(lat, lon - box, lat, lon)[1]
    Δy = latlon_to_meters(lat + box, lon, lat, lon)[2] - latlon_to_meters(lat - box, lon, lat, lon)[2]
    return W, Δx, Δy
end

"""
    era5_columns(e::ERA5Data; lat, lon, box = 0.5) -> NamedTuple

The ERA5 columns at the domain centre (`lat`, `lon`), as `(level, time)`
matrices: height above the ground `z` (m; the ground is the ERA5
orography there), `p`, `θ` ([`potential_temperature`](@ref
Aeolus.MoistThermodynamics.potential_temperature) of T and p), `qv`,
`u`, `v`, the subsidence `w` ([`era5_subsidence`](@ref)), and, on
pressure levels, from centred differences over the ±`box` (degrees)
points `(lat ± box, lon)`, `(lat, lon ± box)`:

* the geostrophic wind `u_g = −(1/f) ∂Φ/∂y`, `v_g = (1/f) ∂Φ/∂x`
  ([`coriolis_parameter`](@ref) at `lat`);
* the advective tendencies `θ_adv = −(u ∂θ/∂x + v ∂θ/∂y)`, `qv_adv`
  likewise, with the centre's wind.

`ground` is the ground height (m above sea level) and `sp`, `skt` the
surface series (or `nothing`). Exact for fields linear in lat and lon.
"""
function era5_columns(e::ERA5Data; lat::Real, lon::Real, box::Real = 0.5)
    W = _bilinear(e, lat, lon)
    e.zs === nothing && error("ERA5: no surface geopotential z (the orography): add it to the single-level request")
    ground = _at(e.zs, W)
    z = _at(e.z, W) .- ground
    p = _at(e.p, W)
    θA = _theta.(e.T, e.p)
    qA = e.q === nothing ? zeros(size(e.T)) : e.q
    θ = _at(θA, W); qv = _at(qA, W)
    u = e.u === nothing ? nothing : _at(e.u, W)
    v = e.v === nothing ? nothing : _at(e.v, W)
    wA = e.omega === nothing ? nothing : era5_subsidence.(e.omega, e.T, e.p, qA)
    w = wA === nothing ? nothing : _at(wA, W)
    ug = vg = θadv = qadv = nothing
    if e.levels === :pressure && box > 0
        P, Δx, Δy = _box_points(e, lat, lon, box)
        f = coriolis_parameter(lat)
        Φ(pt) = _G0 .* _at(e.z, pt)
        ug = -(Φ(P.N) .- Φ(P.S)) ./ (f * Δy)
        vg = (Φ(P.E) .- Φ(P.Wst)) ./ (f * Δx)
        if u !== nothing && v !== nothing
            grad(A) = ((_at(A, P.E) .- _at(A, P.Wst)) ./ Δx, (_at(A, P.N) .- _at(A, P.S)) ./ Δy)
            gx, gy = grad(θA); θadv = -(u .* gx .+ v .* gy)
            gx, gy = grad(qA); qadv = -(u .* gx .+ v .* gy)
        end
    end
    sp = e.sp === nothing ? nothing : _at(e.sp, W)
    skt = e.skt === nothing ? nothing : _at(e.skt, W)
    return (; z, p, θ, qv, u, v, w, u_g = ug, v_g = vg, θ_adv = θadv, qv_adv = qadv, ground, sp, skt)
end

"""
    era5_times(e::ERA5Data; init_time = first(e.time)) -> (init_time::String, t::Vector{Float64})

The time axis of a driver from ERA5: `init_time` (ISO-8601 UTC) and the
valid times in seconds since it. Hourly data stay hourly: the model
interpolates in time (`ForcingTable`, `blend_window!`).
"""
function era5_times(e::ERA5Data; init_time::Dates.DateTime = first(e.time))
    t = [Dates.value(x - init_time) / 1000 for x in e.time]
    return Dates.format(init_time, "yyyy-mm-ddTHH:MM:SS") * "Z", t
end

# a (level, time) matrix of a column on the run's heights `z`: (nz, nt)
function _on_z(C, M, z, name; logarithmic = false)
    return reduce(hcat, [era5_vertical(view(C.z, :, k), view(M, :, k), z; name = "$name at record $k",
                                       logarithmic) for k in axes(M, 2)])
end

"""
    era5_dynamic_inputs(e::ERA5Data; lat, lon, z, box = 0.5, outputs = ERA5_OUTPUTS,
                        init_time = first(e.time)) -> DynamicInputs

The dynamic-driver form of ERA5 at the domain centre (`lat`, `lon`) on
the run's level centres `z` (m above the ground) — everything but the
boundary planes ([`write_era5_driver`](@ref) adds those):

* `:profiles` — the initial profiles `init_theta`, `init_qv`, `init_u`,
  `init_v` at the first record, and `init_p_surface` (`sp`);
* `:large_scale` — `ls_profile_theta`, `_qv`, `_u`, `_v`, `_p` over the
  records (pressure interpolated in `ln p`);
* `:geostrophic` — `ls_u_g`, `ls_v_g` (pressure levels);
* `:subsidence` — `ls_w_subs` from ω ([`era5_subsidence`](@ref));
* `:advection` — `ls_theta_adv`, `ls_qv_adv` (pressure levels);
* `:surface` — `sfc_pressure` (`sp`) and `sfc_temperature` (the skin
  temperature as a potential temperature, `skt/Π(sp)`).

Steps ([`era5_columns`](@ref)): heights `Φ/g₀` above the ERA5 orography,
θ from T and p, bilinear in lat/lon, linear in z
([`era5_vertical`](@ref): constant below the lowest level, an error above
the ERA5 top), centred differences over the ±`box`. Times: the ERA5
valid times in seconds since `init_time` ([`era5_times`](@ref)).
"""
function era5_dynamic_inputs(e::ERA5Data; lat::Real, lon::Real, z::AbstractVector, box::Real = 0.5,
                             outputs = ERA5_OUTPUTS, init_time::Dates.DateTime = first(e.time))
    for o in outputs
        o in ERA5_OUTPUTS || throw(ArgumentError("era5_dynamic_inputs: unknown output :$o (one of $(ERA5_OUTPUTS))"))
    end
    C = era5_columns(e; lat, lon, box = (:geostrophic in outputs || :advection in outputs) ? box : 0.0)
    it, t = era5_times(e; init_time)
    zr = Float64.(collect(z))
    need(x, what, o) = x === nothing ? error("ERA5: the output :$o needs $what, which the files do not hold") : x
    tabs = Dict{Symbol,ForcingTable{Float64}}()
    tab(name, M) = (tabs[name] = ForcingTable(zr, t, M))
    profiles = nothing; ps = nothing
    if :profiles in outputs
        prof(M, n) = era5_vertical(view(C.z, :, 1), view(M, :, 1), zr; name = n)
        profiles = (z = zr, theta = prof(C.θ, "θ"), qv = prof(C.qv, "qv"),
                    u = C.u === nothing ? nothing : prof(C.u, "u"), v = C.v === nothing ? nothing : prof(C.v, "v"))
        ps = C.sp === nothing ? nothing : C.sp[1]
    end
    if :large_scale in outputs
        tab(:profile_theta, _on_z(C, C.θ, zr, "θ")); tab(:profile_qv, _on_z(C, C.qv, zr, "qv"))
        C.u === nothing || tab(:profile_u, _on_z(C, C.u, zr, "u"))
        C.v === nothing || tab(:profile_v, _on_z(C, C.v, zr, "v"))
        tab(:profile_p, _on_z(C, C.p, zr, "p"; logarithmic = true))
    end
    if :geostrophic in outputs
        e.levels === :pressure || error("ERA5: the geostrophic wind needs pressure levels (levels = :pressure)")
        tab(:u_g, _on_z(C, C.u_g, zr, "u_g")); tab(:v_g, _on_z(C, C.v_g, zr, "v_g"))
    end
    :subsidence in outputs && tab(:w_subs, _on_z(C, need(C.w, "the vertical velocity w (ω)", :subsidence), zr, "w"))
    if :advection in outputs
        e.levels === :pressure || error("ERA5: the advective tendencies need pressure levels (levels = :pressure)")
        tab(:theta_adv, _on_z(C, need(C.θ_adv, "u and v", :advection), zr, "θ_adv"))
        tab(:qv_adv, _on_z(C, C.qv_adv, zr, "qv_adv"))
    end
    sfc = nothing
    if :surface in outputs && C.sp !== nothing
        sfc = (t = t, temperature = C.skt === nothing ? nothing : C.skt ./ exner_function.(C.sp), pressure = C.sp)
    end
    attrs = Dict{String,Any}("dynamic_source" => "era5", "era5_levels" => String(e.levels),
                             "era5_box_deg" => Float64(box), "era5_site_lat" => Float64(lat),
                             "era5_site_lon" => Float64(lon), "era5_ground_height" => C.ground,
                             "era5_outputs" => join(String.(collect(outputs)), " "),
                             "era5_files" => join(["$(basename(f)) $(file_fnv1a64(f))" for f in e.sources], "; "))
    return DynamicInputs(; init_time = it, profiles, p_surface = ps, tables = tabs, surface = sfc, attributes = attrs)
end

"""
    era5_planes!(planes, e::ERA5Data, k; lat, lon, origin_x, origin_y) -> planes

Fill the buffers of the non-mirror `planes` (`OpenPlane`s of a run's open
boundary set) with ERA5 record `k`: every plane point `(x, y, z)` is
mapped to lat/lon about the anchor (`lat`, `lon` at the mesh point
`origin_x`, `origin_y`; [`meters_to_latlon`](@ref)), the columns there
are interpolated bilinearly and then linearly to `z − z0` (`z0` the
mesh's ground) above the local ERA5 ground. `u`, `v` are the east and north wind, `w` the subsidence of ω,
`θ` the potential temperature and `qv` the specific humidity.
"""
function era5_planes!(planes, e::ERA5Data, k::Integer; lat::Real, lon::Real, origin_x::Real, origin_y::Real,
                      z0::Real = 0.0)
    e.zs === nothing && error("ERA5: no surface geopotential z (the orography): add it to the single-level request")
    qA = e.q === nothing ? zeros(size(e.T)) : e.q
    src = Dict{Symbol,Any}(:θ => _theta.(view(e.T, :, :, :, k), view(e.p, :, :, :, k)), :qv => qA[:, :, :, k],
                           :u => e.u === nothing ? nothing : e.u[:, :, :, k],
                           :v => e.v === nothing ? nothing : e.v[:, :, :, k],
                           :w => e.omega === nothing ? nothing :
                                 era5_subsidence.(view(e.omega, :, :, :, k), view(e.T, :, :, :, k),
                                                  view(e.p, :, :, :, k), view(qA, :, :, :, k)))
    zk = e.z[:, :, :, k]
    _at3(A, W) = sum(w .* A[i, j, :] for (i, j, w) in W)
    for p in planes
        p.mode === :mirror && continue
        A = get(src, p.name, nothing)
        A === nothing && error("ERA5: the run has an open plane for :$(p.name), which the ERA5 files do not hold")
        for (b, y) in enumerate(p.y), (a, x) in enumerate(p.x)
            la, lo = meters_to_latlon(x - origin_x, y - origin_y, lat, lon)
            W = _bilinear(e, la, lo)
            zl = _at3(zk, W) .- _at(e.zs, W)
            p.data[a, b, :] = era5_vertical(zl, _at3(A, W), p.z .- z0; name = "the $(p.side) plane of $(p.name)")
        end
    end
    return planes
end

"""
    write_era5_driver(path, e::ERA5Data; lat, lon, mesh, planes = nothing, box = 0.5,
                      outputs = ERA5_OUTPUTS, init_time = first(e.time)) -> path

Write the dynamic driver of ERA5 `e` for a run on `mesh` whose centre
lies at (`lat`, `lon`): [`era5_dynamic_inputs`](@ref) on the mesh's level
centres and, with `:planes` in `outputs` and the run's open-boundary
`planes`, one `time_bc` record per ERA5 time ([`era5_planes!`](@ref)),
which the run blends linearly in time (`FileBoundaryData`). The anchor is
the domain centre (`origin_x`, `origin_y` attributes); the first ERA5
file is the hashed `source`, every file is listed with its hash in the
attribute `era5_files`.
"""
function write_era5_driver(path::AbstractString, e::ERA5Data; lat::Real, lon::Real, mesh, planes = nothing,
                           box::Real = 0.5, outputs = ERA5_OUTPUTS, init_time::Dates.DateTime = first(e.time),
                           preprocessor::AbstractString = "aeolus preprocess dynamic --from-era5")
    zc = collect(Float64, znodes(mesh, CellCenter())) .- mesh.z0
    d = era5_dynamic_inputs(e; lat, lon, z = zc, box, outputs = Tuple(o for o in outputs if o !== :planes), init_time)
    ox = mesh.x0 + mesh.Lx / 2; oy = mesh.y0 + mesh.Ly / 2
    kw = (; source = first(e.sources), preprocessor, origin_lat = Float64(lat), origin_lon = Float64(lon),
          origin_x = ox, origin_y = oy)
    if :planes in outputs && planes !== nothing && any(p -> p.mode !== :mirror, planes)
        _, t = era5_times(e; init_time)
        rec = NestBoundaryRecorder(path, planes, mesh; inputs = d, kw...)
        for k in eachindex(t)
            era5_planes!(planes, e, k; lat, lon, origin_x = ox, origin_y = oy, z0 = mesh.z0)
            record_boundary!(rec, t[k], k == 1 ? NaN : t[k] - t[k-1], plane_snapshot(planes, mesh))
        end
    else
        write_dynamic_driver(path, d; dx = mesh.dx, dy = mesh.dy, kw...)
    end
    return path
end
