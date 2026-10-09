# Radiosonde soundings → height-time profiles (N-parent-a, #247).
#
# Included from LargeScaleForcing. A `Sounding` is a set of launches from
# one station; `sounding_profiles` turns it into θ, q_v, u, v (and p)
# `ForcingTable`s on heights above the station and times from an epoch,
# the form every initial state, nudging target and boundary-data reader of
# the nesting epic (#246) consumes. No thermodynamics is defined here:
# θ is T over the MoistThermodynamics Exner function, q_v comes from its
# saturation vapour pressure at the dew point, heights from hypsometric
# integration of its virtual temperature.

"""
    SoundingLaunch(t, p, T, Td; pw = p, u = nothing, v = nothing, gph = nothing,
                   tag = "", c = MoistThermodynamics.DEFAULT_CONSTANTS)

One radiosonde ascent. Thermodynamic levels: pressure `p` (Pa, strictly
decreasing, the surface first), temperature `T` and dew point `Td` (K;
`NaN` where the dew point is missing). Wind levels: pressures `pw` with
components `u`, `v` (m/s; omit for none). `gph` is the reported
geopotential height above sea level (m, `NaN` where not reported), kept
only as a check. `t` is the time stamp of the profile (s from the
sounding's epoch; the release time).

The heights above the surface level are computed, not read: hypsometric
integration of the virtual temperature,

```math
z_{k+1} = z_k + \\frac{R_d}{g_0}\\,\\frac{T_{v,k} + T_{v,k+1}}{2}\\,
          \\ln\\frac{p_k}{p_{k+1}},
```

exact for ``T_v`` linear in ``\\ln p`` between levels, with
``T_v = `` `virtual_temperature(T, PhasePartition(q_v))` and
``q_v = `` `specific_humidity_from_relative_humidity(Td, p, 1)` (the
specific humidity whose vapour pressure is ``e_s(T_d)``; ``q_v = 0``
where `Td` is missing). ``g_0 = 9.80665`` m s⁻² is the standard gravity
that defines the geopotential metre, so ``z`` + station elevation is
directly comparable with `gph`. Wind-level heights are interpolated
linearly in ``\\ln p`` between the thermodynamic levels (levels outside
their pressure range are dropped).

Reference: Wallace & Hobbs (2006), *Atmospheric Science*, 2nd ed.,
§3.2.2 (the hypsometric equation, eq. 3.29).
"""
struct SoundingLaunch
    tag::String
    t::Float64
    p::Vector{Float64}
    T::Vector{Float64}
    Td::Vector{Float64}
    qv::Vector{Float64}
    z::Vector{Float64}
    gph::Vector{Float64}
    zw::Vector{Float64}
    u::Vector{Float64}
    v::Vector{Float64}
end

function SoundingLaunch(t::Real, p::AbstractVector, T::AbstractVector, Td::AbstractVector;
                        pw::AbstractVector = p, u = nothing, v = nothing, gph = nothing,
                        tag::AbstractString = "",
                        c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    n = length(p)
    (length(T) == n && length(Td) == n) ||
        throw(ArgumentError("SoundingLaunch: p, T, Td must have equal lengths"))
    n >= 2 || throw(ArgumentError("SoundingLaunch: need at least two thermodynamic levels"))
    all(<(0), diff(p)) || throw(ArgumentError("SoundingLaunch: p must be strictly decreasing (surface first)"))
    all(isfinite, T) || throw(ArgumentError("SoundingLaunch: T must be finite on the thermodynamic levels"))
    qv = [isnan(Td[k]) ? 0.0 : specific_humidity_from_relative_humidity(Td[k], p[k], 1.0, c)
          for k in 1:n]
    Tv = [virtual_temperature(T[k], PhasePartition(qv[k]), c) for k in 1:n]
    Rd = c.gas_constant / c.molmass_dry
    z = zeros(n)
    for k in 1:n-1
        z[k+1] = z[k] + Rd / c.grav * (Tv[k] + Tv[k+1]) / 2 * log(p[k] / p[k+1])
    end
    g = gph === nothing ? fill(NaN, n) : Float64.(collect(gph))
    length(g) == n || throw(ArgumentError("SoundingLaunch: gph must have the length of p"))
    # wind levels: z(ln p) from the thermodynamic levels
    zw = Float64[]; uw = Float64[]; vw = Float64[]
    if u !== nothing
        (v !== nothing && length(u) == length(pw) == length(v)) ||
            throw(ArgumentError("SoundingLaunch: u, v and pw must have equal lengths"))
        lp = reverse(log.(p)); zr = reverse(z)          # increasing ln p
        for j in eachindex(pw)
            (isfinite(u[j]) && isfinite(v[j])) || continue
            x = log(pw[j])
            (lp[1] <= x <= lp[end]) || continue
            i, w = _bracket(lp, x)
            push!(zw, _lerp(zr, i, w)); push!(uw, u[j]); push!(vw, v[j])
        end
        perm = sortperm(zw)
        zw, uw, vw = zw[perm], uw[perm], vw[perm]
        keep = [k == 1 || zw[k] > zw[k-1] for k in eachindex(zw)]   # one value per height
        zw, uw, vw = zw[keep], uw[keep], vw[keep]
    end
    return SoundingLaunch(String(tag), Float64(t), Float64.(collect(p)), Float64.(collect(T)),
                          Float64.(collect(Td)), qv, z, g, zw, uw, vw)
end

"""
    Sounding(launches; station = "", elevation = NaN, epoch = (1970, 1, 1))

A set of [`SoundingLaunch`](@ref)es from one station (`station` name,
launch-site `elevation` in m above sea level), ordered by time. Times
are seconds from 00 UTC of the date `epoch = (y, m, d)`
([`sounding_time`](@ref) converts a calendar time). Read one from a CSV
with [`read_sounding`](@ref); turn it into profiles with
[`sounding_profiles`](@ref).
"""
struct Sounding
    station::String
    elevation::Float64
    epoch::NTuple{3,Int}
    launches::Vector{SoundingLaunch}
    function Sounding(launches::AbstractVector{SoundingLaunch}; station::AbstractString = "",
                      elevation::Real = NaN, epoch::NTuple{3,Int} = (1970, 1, 1))
        isempty(launches) && throw(ArgumentError("Sounding: no launches"))
        ls = sort(collect(launches); by = l -> l.t)
        allunique(l.t for l in ls) || throw(ArgumentError("Sounding: two launches share a time stamp"))
        return new(String(station), Float64(elevation), epoch, ls)
    end
end

# days since 1970-01-01 of a Gregorian date (Hinnant's days_from_civil)
function _days_from_civil(y::Integer, m::Integer, d::Integer)
    y -= m <= 2
    era = fld(y, 400); yoe = y - era * 400
    doy = fld(153 * (m + (m > 2 ? -3 : 9)) + 2, 5) + d - 1
    doe = yoe * 365 + fld(yoe, 4) - fld(yoe, 100) + doy
    return era * 146097 + doe - 719468
end

"""
    sounding_time(s::Sounding, y, m, d, hour_utc = 0) -> t (s)

Seconds from the epoch of `s` (00 UTC of `s.epoch`) to the UTC time
`hour_utc` (fractional hours) of the date `(y, m, d)`: the `t` at which
to read the tables of [`sounding_profiles`](@ref).
"""
sounding_time(s::Sounding, y::Integer, m::Integer, d::Integer, hour_utc::Real = 0) =
    86400.0 * (_days_from_civil(y, m, d) - _days_from_civil(s.epoch...)) + 3600.0 * hour_utc

"""
    read_sounding(path; c = MoistThermodynamics.DEFAULT_CONSTANTS) -> Sounding

Read a sounding CSV in the format written by
`aeolus preprocess fetch-sounding`: comment lines start with
`#`; lines `#! key = value` carry `epoch` (`yyyy-mm-ddT00:00Z`) and
`station_elevation_m`; then a header and one row per level with (at
least) the columns `launch`, `t_s`, `p_Pa`, `gph_m`, `T_K`, `Td_K`,
`wdir_deg`, `wspd_ms` (`NaN` = missing). Rows with a temperature are
thermodynamic levels, rows with a wind are wind levels (a row can be
both). Wind direction is meteorological (the direction the wind blows
from): ``u = -U\\sin φ``, ``v = -U\\cos φ``.
"""
function read_sounding(path::AbstractString; c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    meta = Dict{String,String}()
    header = String[]
    rows = Vector{Vector{String}}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "#!")
            kv = split(s[3:end], "="; limit = 2)
            length(kv) == 2 && (meta[strip(kv[1])] = strip(kv[2]))
        elseif startswith(s, "#")
            continue
        elseif isempty(header)
            header = strip.(split(s, ","))
        else
            push!(rows, strip.(split(s, ",")))
        end
    end
    col(name) = (i = findfirst(==(name), header);
                 i === nothing ? throw(ArgumentError("read_sounding: column $name missing in $path")) : i)
    iL, it, ip, iz, iT, iTd, id, iw = col.(("launch", "t_s", "p_Pa", "gph_m", "T_K", "Td_K",
                                             "wdir_deg", "wspd_ms"))
    num(x) = parse(Float64, x)
    epoch = haskey(meta, "epoch") ? Tuple(parse.(Int, split(first(split(meta["epoch"], "T")), "-"))) :
            (1970, 1, 1)
    elevation = haskey(meta, "station_elevation_m") ? num(meta["station_elevation_m"]) : NaN
    tags = unique(r[iL] for r in rows)
    launches = SoundingLaunch[]
    for tag in tags
        R = [r for r in rows if r[iL] == tag]
        t = num(R[1][it])
        th = [r for r in R if isfinite(num(r[iT]))]
        wd = [r for r in R if isfinite(num(r[iw])) && isfinite(num(r[id]))]
        U = [num(r[iw]) for r in wd]; φ = [deg2rad(num(r[id])) for r in wd]
        push!(launches, SoundingLaunch(t, [num(r[ip]) for r in th], [num(r[iT]) for r in th],
                                       [num(r[iTd]) for r in th];
                                       pw = [num(r[ip]) for r in wd],
                                       u = -U .* sin.(φ), v = -U .* cos.(φ),
                                       gph = [num(r[iz]) for r in th], tag = tag, c = c))
    end
    station = get(meta, "station", basename(path))
    return Sounding(launches; station = station, elevation = elevation, epoch = epoch)
end

"""
    sounding_profiles(s::Sounding; z_top = 3000.0, z = nothing,
                      c = MoistThermodynamics.DEFAULT_CONSTANTS)
        -> (; θ, qv, u, v, p, z)

The launches of `s` as height-time [`ForcingTable`](@ref)s: potential
temperature `θ` (K), specific humidity `qv` (kg kg⁻¹), wind components
`u`, `v` (m/s) and pressure `p` (Pa), each at heights above the launch
site (m) and the launch times (s from the epoch,
[`sounding_time`](@ref)); `z` is the common height grid.

* ``θ = T/Π(p)`` with the dry Exner function ``Π = (p/p_{ref})^{R_d/c_{pd}}``
  of MoistThermodynamics (`exner_function(p)`), ``q_v`` from the dew point
  (see [`SoundingLaunch`](@ref)), heights hypsometric.
* Between the levels of one launch every quantity is linear in ``z``;
  the table holds that interpolant exactly: the default grid `z` is the
  union of all launches' level heights in ``[0, z_{top}]`` (plus 0 and
  `z_top`), so each launch's piecewise-linear profile is reproduced
  without loss at every height. Pass `z` for another grid.
* Between launches [`forcing_value`](@ref) interpolates linearly in
  time; before the first and after the last launch, and above a launch's
  highest level, the table holds its end values (the `ForcingTable`
  convention).
"""
function sounding_profiles(s::Sounding; z_top::Real = 3000.0, z = nothing,
                           c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    L = s.launches
    θs = [[T / exner_function(p, PhasePartition(0.0), c) for (T, p) in zip(l.T, l.p)] for l in L]
    hasq = [findall(!isnan, l.Td) for l in L]
    if z === nothing
        zz = Float64[0.0, Float64(z_top)]
        for l in L
            append!(zz, filter(h -> 0 <= h <= z_top, l.z))
            append!(zz, filter(h -> 0 <= h <= z_top, l.zw))
        end
        zg = sort!(unique!(zz))
    else
        zg = Float64.(collect(z))
    end
    nz, nt = length(zg), length(L)
    tab(f) = [f(n, h) for h in zg, n in 1:nt]
    prof(x, y, h) = isempty(x) ? NaN : (length(x) == 1 ? y[1] : (iw = _bracket(x, h); _lerp(y, iw...)))
    Θ = tab((n, h) -> prof(L[n].z, θs[n], h))
    Q = tab((n, h) -> prof(L[n].z[hasq[n]], L[n].qv[hasq[n]], h))
    P = tab((n, h) -> prof(L[n].z, L[n].p, h))
    U = tab((n, h) -> prof(L[n].zw, L[n].u, h))
    V = tab((n, h) -> prof(L[n].zw, L[n].v, h))
    t = [l.t for l in L]
    return (θ = ForcingTable(zg, t, Θ), qv = ForcingTable(zg, t, Q), u = ForcingTable(zg, t, U),
            v = ForcingTable(zg, t, V), p = ForcingTable(zg, t, P), z = zg)
end
