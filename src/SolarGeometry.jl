module SolarGeometry

# Phase 38 — Solar position, atmospheric refraction, direct/diffuse
# partitioning, and true 3-D cloud shadows.
#
# What this replaces: the Era-I urban runs used
# `solar_declination(day) = 23.44 sin(2π(284+d)/365)` with a mean-solar-time
# hour angle. That is a ~1° approximation with no equation of time, no
# refraction, no nutation, and a fixed 365-day year — errors of order 15
# minutes in sunrise time and 1° in elevation, which is enough to
# misplace a shadow by several metres at low sun.
#
# Accuracy statement (important, and deliberately explicit):
#   The solar position here follows the standard astronomical series of
#   Meeus, *Astronomical Algorithms* 2nd ed., ch. 25 (solar coordinates),
#   ch. 12 (sidereal time) and ch. 22 (nutation/obliquity), with the
#   Bennett refraction formula. Achieved accuracy is ~0.01° in elevation
#   and azimuth over 1950-2050.
#
#   The full NREL SPA of Reda & Andreas (2004) attains ±0.0003° by
#   evaluating the complete VSOP87 periodic-term tables (≈ 350 published
#   coefficients). Those tables are not reproduced here: transcribing
#   them from memory would risk silent, hard-to-detect errors, which is
#   worse than a documented 0.01°. `AbstractSolarPosition` is the
#   extension point — a `SPAPosition` carrying the real tables is a
#   drop-in refinement.
#
#   For radiative transfer 0.01° is not the limiting error: the solar
#   disc subtends 0.53°, so the position uncertainty is ~50× smaller than
#   the source itself. It *would* matter for a heliostat or a sundial.

using Printf
using ..Grids
using ..Grids: RegularGrid, CellCenter, interior
using ..Fields
using ..Fields: Field, interior_view
using ..RadiationCore
using ..RadiationCore: SOLAR_CONSTANT, liquid_mass_extinction, ice_mass_extinction

export AbstractSolarPosition, MeeusPosition,
       SolarTime, solar_time, julian_day, julian_century,
       SolarPosition, solar_position,
       equation_of_time, solar_declination_accurate,
       sun_direction_enu, air_mass,
       refraction_bennett, apply_refraction,
       earth_sun_distance_factor, extraterrestrial_irradiance,
       ClearSkyModel, BirdClearSky, clear_sky_irradiance,
       clearness_index, diffuse_fraction_erbs, diffuse_fraction_reindl,
       partition_global_irradiance,
       cloud_optical_depth_column, cloud_shadow_transmittance!,
       cloud_shadow_map, combined_shadow_transmittance!

# =========================================================================
# Time
# =========================================================================

"""
    SolarTime

An instant specified in UTC plus the observer's geographic position, which
is everything the position algorithm needs.

  * `year`, `month`, `day` — calendar date (Gregorian)
  * `hour`, `minute`, `second` — UTC
  * `latitude`, `longitude` — degrees, north and **east** positive
  * `elevation` — metres above sea level
  * `pressure` — station pressure in hPa (for refraction)
  * `temperature` — station air temperature in °C (for refraction)
  * `delta_t` — TT − UT1 in seconds (≈ 69 s in the 2020s); affects the
    position by < 0.001° so the default is fine for radiation

Longitude sign convention is east-positive, the ISO convention. Note the
original NREL SPA paper uses east-positive too, but many meteorological
datasets use west-positive — a silent sign flip here moves the sun by
hours, so it is worth checking against a known sunrise.
"""
struct SolarTime{T<:AbstractFloat}
    year::Int
    month::Int
    day::Int
    hour::Int
    minute::Int
    second::T
    latitude::T
    longitude::T
    elevation::T
    pressure::T
    temperature::T
    delta_t::T
end

function solar_time(year, month, day, hour, minute, second;
                    latitude, longitude, elevation = 0.0,
                    pressure = 1013.25, temperature = 15.0,
                    delta_t = 69.0)
    T = promote_type(typeof(float(second)), typeof(float(latitude)))
    -90 <= latitude <= 90 || error("solar_time: latitude must be in [-90, 90]")
    -180 <= longitude <= 360 ||
        error("solar_time: longitude must be in [-180, 360] (east positive)")
    return SolarTime{T}(Int(year), Int(month), Int(day), Int(hour), Int(minute),
                        T(second), T(latitude), T(longitude), T(elevation),
                        T(pressure), T(temperature), T(delta_t))
end

"""
    julian_day(st::SolarTime) -> Float64

Julian Day number for the instant, using the standard Gregorian-calendar
algorithm (Meeus ch. 7). Valid for all dates after the Gregorian reform.
"""
function julian_day(st::SolarTime)
    y = st.year; m = st.month
    d = st.day + (st.hour + (st.minute + st.second / 60) / 60) / 24
    if m <= 2
        y -= 1
        m += 12
    end
    A = fld(y, 100)
    B = 2 - A + fld(A, 4)          # Gregorian correction
    return floor(365.25 * (y + 4716)) + floor(30.6001 * (m + 1)) + d + B - 1524.5
end

"Julian centuries from J2000.0 (Terrestrial Time)."
function julian_century(st::SolarTime)
    jd = julian_day(st)
    jde = jd + st.delta_t / 86400      # UT → TT
    return (jde - 2451545.0) / 36525
end

# =========================================================================
# Solar position
# =========================================================================

"""
    AbstractSolarPosition

Solar-position algorithm selector. Concrete types:

  * [`MeeusPosition`](@ref) — the standard astronomical series, ~0.01°.

A `SPAPosition` carrying the full NREL/VSOP87 periodic tables (±0.0003°)
is the documented refinement; see the module header for why it is not
included here.
"""
abstract type AbstractSolarPosition end

"""
    MeeusPosition()

Solar position from the standard astronomical series (Meeus,
*Astronomical Algorithms* 2nd ed.). Achieved accuracy ~0.01° in elevation
and azimuth for 1950-2050.
"""
struct MeeusPosition <: AbstractSolarPosition end

"""
    SolarPosition{T}

Computed solar geometry:

  * `elevation_true` — geometric elevation above the horizon, degrees
  * `elevation`      — *apparent* elevation, i.e. refraction-corrected
  * `zenith`         — apparent zenith angle, degrees (`90 - elevation`)
  * `azimuth`        — degrees clockwise from **north** (N=0, E=90, S=180, W=270)
  * `μ₀`             — `cos(zenith)` of the apparent position, clamped at 0
  * `declination`, `right_ascension`, `hour_angle` — degrees
  * `eot_minutes`    — equation of time in minutes
  * `distance_au`    — Earth-Sun distance in AU
  * `air_mass`       — relative optical air mass (Kasten-Young), `Inf` below
                       the horizon

The azimuth convention (clockwise from north) matches the meteorological
standard and `sun_direction_enu` below.
"""
struct SolarPosition{T<:AbstractFloat}
    elevation_true::T
    elevation::T
    zenith::T
    azimuth::T
    μ₀::T
    declination::T
    right_ascension::T
    hour_angle::T
    eot_minutes::T
    distance_au::T
    air_mass::T
end

function Base.show(io::IO, p::SolarPosition)
    @printf(io, "SolarPosition(elev %.4f° [true %.4f°], az %.4f°, zen %.4f°, μ₀ %.6f, δ %.4f°, R %.6f AU)",
            p.elevation, p.elevation_true, p.azimuth, p.zenith, p.μ₀,
            p.declination, p.distance_au)
end

@inline _deg(x) = rad2deg(x)
@inline _rad(x) = deg2rad(x)
# wrap to [0, 360)
@inline _wrap360(x) = mod(x, 360)
# wrap to (-180, 180]
@inline function _wrap180(x)
    y = mod(x + 180, 360) - 180
    return y == -180 ? 180.0 : y
end

"""
    solar_position(st::SolarTime, alg = MeeusPosition()) -> SolarPosition

Compute the apparent solar position. Includes the equation of time,
nutation in longitude, apparent obliquity, aberration, the topocentric
parallax correction, and Bennett atmospheric refraction scaled by the
station pressure and temperature.
"""
function solar_position(st::SolarTime{T}, ::MeeusPosition = MeeusPosition()) where {T}
    jd = julian_day(st)
    Tc = julian_century(st)                     # Julian centuries (TT)

    # --- Sun's geometric mean longitude and mean anomaly (Meeus 25.2-25.3)
    L0 = _wrap360(280.46646 + 36000.76983Tc + 0.0003032Tc^2)
    M  = _wrap360(357.52911 + 35999.05029Tc - 0.0001537Tc^2)
    e  = 0.016708634 - 0.000042037Tc - 0.0000001267Tc^2

    # --- equation of the centre (Meeus 25.4)
    Mr = _rad(M)
    C = (1.914602 - 0.004817Tc - 0.000014Tc^2) * sin(Mr) +
        (0.019993 - 0.000101Tc) * sin(2Mr) +
         0.000289 * sin(3Mr)

    Θ = L0 + C                      # true geometric longitude
    ν = M + C                       # true anomaly
    R = 1.000001018 * (1 - e^2) / (1 + e * cos(_rad(ν)))   # AU (Meeus 25.5)

    # --- apparent longitude: nutation + aberration (Meeus p. 164)
    Ω = 125.04 - 1934.136Tc
    λ = Θ - 0.00569 - 0.00478 * sin(_rad(Ω))

    # --- obliquity of the ecliptic (Meeus 22.2), apparent
    ε0 = 23 + (26 + (21.448 - 46.8150Tc - 0.00059Tc^2 + 0.001813Tc^3) / 60) / 60
    ε  = ε0 + 0.00256 * cos(_rad(Ω))

    λr = _rad(λ); εr = _rad(ε)
    # --- geocentric right ascension and declination (Meeus 25.6-25.7)
    α = _wrap360(_deg(atan(cos(εr) * sin(λr), cos(λr))))
    δ = _deg(asin(sin(εr) * sin(λr)))

    # --- Greenwich apparent sidereal time (Meeus 12.4) + nutation in RA
    d = jd - 2451545.0
    θ_gmst = _wrap360(280.46061837 + 360.98564736629d +
                      0.000387933Tc^2 - Tc^3 / 38710000)
    # nutation in longitude (dominant term) projected onto the equator
    Δψ = -17.20 / 3600 * sin(_rad(Ω))
    θ_gast = θ_gmst + Δψ * cos(εr)

    # --- local hour angle
    H = _wrap180(θ_gast + st.longitude - α)

    φr = _rad(st.latitude); δr = _rad(δ); Hr = _rad(H)

    # --- topocentric parallax (Meeus ch. 40): the observer is on the
    # surface, not at the geocentre. ~8.8" at most, but free to include.
    ξ = 8.794 / 3600 / R                         # equatorial parallax, degrees
    u = atan(0.99664719 * tan(φr))
    ρsinφ′ = 0.99664719 * sin(u) + st.elevation / 6378140 * sin(φr)
    ρcosφ′ = cos(u) + st.elevation / 6378140 * cos(φr)
    Δα = atan(-ρcosφ′ * sin(_rad(ξ)) * sin(Hr),
               cos(δr) - ρcosφ′ * sin(_rad(ξ)) * cos(Hr))
    δ′ = atan((sin(δr) - ρsinφ′ * sin(_rad(ξ))) * cos(Δα),
               cos(δr) - ρcosφ′ * sin(_rad(ξ)) * cos(Hr))
    H′ = Hr + Δα
    δ′d = _deg(δ′)

    # --- elevation and azimuth
    sin_h = sin(φr) * sin(δ′) + cos(φr) * cos(δ′) * cos(H′)
    h_true = _deg(asin(clamp(sin_h, -1, 1)))

    # azimuth measured clockwise from north
    az = _wrap360(180 + _deg(atan(sin(H′),
                    cos(H′) * sin(φr) - tan(δ′) * cos(φr))))

    # --- refraction
    Δh = refraction_bennett(h_true; pressure = st.pressure,
                                     temperature = st.temperature)
    h_app = h_true + Δh
    zen = 90 - h_app
    μ₀ = max(zero(T), T(cosd(zen)))

    # --- equation of time (minutes). EoT = apparent solar time − mean solar
    # time; the standard series form (Meeus 28.3) expressed via L0 and α.
    eot = 4 * _wrap180(L0 - 0.0057183 - α + Δψ * cos(εr))

    am = h_app > 0 ? air_mass(h_app; pressure = st.pressure) : T(Inf)

    return SolarPosition{T}(T(h_true), T(h_app), T(zen), T(az), μ₀,
                            T(δ′d), T(α), T(H), T(eot), T(R), T(am))
end

"""
    equation_of_time(st) -> minutes

Difference between apparent and mean solar time. Ranges ±16 minutes over
the year; this is the term the Era-I sinusoidal approximation omitted
entirely, and it is why the analemma exists.
"""
equation_of_time(st::SolarTime) = solar_position(st).eot_minutes

"""
    solar_declination_accurate(st) -> degrees

Apparent solar declination. Compare against the Era-I approximation
`23.44 sin(2π(284+doy)/365)`, which is in error by up to ~0.5°.
"""
solar_declination_accurate(st::SolarTime) = solar_position(st).declination

"""
    sun_direction_enu(p::SolarPosition) -> (east, north, up)

Unit vector pointing from the observer **toward** the sun, in a local
east-north-up frame — the convention the shadow ray-marchers use. `up > 0`
when the sun is above the (apparent) horizon.
"""
function sun_direction_enu(p::SolarPosition)
    hr = _rad(p.elevation)
    ar = _rad(p.azimuth)
    ch = cos(hr)
    return (ch * sin(ar),     # east
            ch * cos(ar),     # north
            sin(hr))          # up
end

# =========================================================================
# Refraction and air mass
# =========================================================================

"""
    refraction_bennett(h_true; pressure = 1013.25, temperature = 15.0) -> degrees

Atmospheric refraction correction to add to the true elevation, from
Bennett (1982):

```math
R = \\frac{1.02}{\\tan\\left(h + \\frac{10.3}{h + 5.11}\\right)} \\text{ arcmin}
```

with `h` in degrees, scaled by `(P/1010)(283/(273+T))` for non-standard
conditions. At the horizon this is ~34 arcmin — slightly more than the
solar diameter, which is why the sun appears to rise before it
geometrically does.

Returns 0 for elevations below −1° (the sun is well set and refraction is
meaningless there).
"""
function refraction_bennett(h_true::Real; pressure::Real = 1013.25,
                            temperature::Real = 15.0)
    h_true < -1 && return zero(float(h_true))
    R_arcmin = 1.02 / tand(h_true + 10.3 / (h_true + 5.11))
    scale = (pressure / 1010) * (283 / (273 + temperature))
    return R_arcmin * scale / 60          # arcmin → degrees
end

"""
    apply_refraction(h_true; kwargs...) -> apparent elevation (degrees)
"""
apply_refraction(h_true::Real; kwargs...) =
    h_true + refraction_bennett(h_true; kwargs...)

"""
    air_mass(h_apparent; pressure = 1013.25) -> relative air mass

Kasten & Young (1989) relative optical air mass:

```math
m = \\frac{1}{\\cos z + 0.50572 (96.07995 - z)^{-1.6364}}
```

with `z` the apparent zenith angle in degrees, scaled by `P/1013.25` to
give the *absolute* (pressure-corrected) air mass. Equals 1 at the zenith
and ≈ 38 at the horizon — the plane-parallel `1/μ₀` would diverge there,
which is why this formula exists.
"""
function air_mass(h_apparent::Real; pressure::Real = 1013.25)
    # The Kasten-Young fit is valid down to and including the horizon
    # (h = 0 gives m ≈ 38); only genuinely negative elevations have no
    # air mass. Guarding at `<= 0` instead of `< 0` would incorrectly
    # return Inf exactly at sunrise/sunset.
    h_apparent < 0 && return Inf
    z = 90 - h_apparent
    m = 1 / (cosd(z) + 0.50572 * (96.07995 - z)^(-1.6364))
    return m * pressure / 1013.25
end

"""
    earth_sun_distance_factor(p::SolarPosition) -> (R₀/R)²

Inverse-square correction to the solar constant for the Earth-Sun
distance. Varies ±3.4% over the year (perihelion early January).
"""
earth_sun_distance_factor(p::SolarPosition) = 1 / p.distance_au^2

"""
    extraterrestrial_irradiance(p::SolarPosition; S₀ = SOLAR_CONSTANT) -> W/m²

Solar irradiance on a plane normal to the beam at the top of the
atmosphere, corrected for the Earth-Sun distance.
"""
extraterrestrial_irradiance(p::SolarPosition; S₀::Real = SOLAR_CONSTANT) =
    S₀ * earth_sun_distance_factor(p)

# =========================================================================
# Clear-sky model and direct/diffuse partitioning
# =========================================================================

"""
    ClearSkyModel

Clear-sky irradiance model selector. Concrete: [`BirdClearSky`](@ref).
"""
abstract type ClearSkyModel end

"""
    BirdClearSky(; aod380, aod500, precipitable_water, ozone, albedo, ba)

Bird & Hulstrom (1981) clear-sky broadband model. Returns direct-normal,
diffuse-horizontal and global-horizontal irradiance from the solar
position plus a small number of atmospheric parameters. It is the
standard compact clear-sky reference: accurate to a few percent against
rigorous spectral calculations, with no spectral integration required.

Parameters:
  * `aod380`, `aod500` — aerosol optical depth at 380 and 500 nm
  * `precipitable_water` — cm
  * `ozone` — atm-cm (0.3 is typical)
  * `albedo` — ground albedo (feeds the multiple-reflection term)
  * `ba` — forward-scattering fraction, 0.85 in the original paper
"""
struct BirdClearSky{T<:AbstractFloat} <: ClearSkyModel
    aod380::T
    aod500::T
    precipitable_water::T
    ozone::T
    albedo::T
    ba::T
end

BirdClearSky(; aod380 = 0.15, aod500 = 0.10, precipitable_water = 1.5,
               ozone = 0.3, albedo = 0.2, ba = 0.85) =
    BirdClearSky{Float64}(aod380, aod500, precipitable_water, ozone,
                          albedo, ba)

"""
    clear_sky_irradiance(p::SolarPosition, m::BirdClearSky; S₀ = SOLAR_CONSTANT)
        -> (; dni, dhi, ghi)

Bird clear-sky irradiance: direct-normal (`dni`), diffuse-horizontal
(`dhi`) and global-horizontal (`ghi`), all W/m². Returns all zeros when
the sun is below the horizon.
"""
function clear_sky_irradiance(p::SolarPosition, m::BirdClearSky;
                              S₀::Real = SOLAR_CONSTANT)
    if p.elevation <= 0
        return (dni = 0.0, dhi = 0.0, ghi = 0.0)
    end
    μ = p.μ₀
    am = p.air_mass                       # pressure-corrected air mass
    amp = am                              # Bird uses the pressure-corrected one
    etr = extraterrestrial_irradiance(p; S₀ = S₀)

    # transmittances (Bird & Hulstrom 1981)
    T_r  = exp(-0.0903 * amp^0.84 * (1 + amp - amp^1.01))                # Rayleigh
    T_o  = 1 - 0.1611 * (m.ozone * am) * (1 + 139.48 * (m.ozone * am))^(-0.3034) -
               0.002715 * (m.ozone * am) /
               (1 + 0.044 * (m.ozone * am) + 0.0003 * (m.ozone * am)^2)  # ozone
    T_um = exp(-0.0127 * amp^0.26)                                        # uniform gases
    wm   = m.precipitable_water * am
    T_w  = 1 - 2.4959 * wm / ((1 + 79.034 * wm)^0.6828 + 6.385 * wm)      # water vapour
    # aerosol: broadband equivalent optical depth from the two AODs
    τa   = 0.2758 * m.aod380 + 0.35 * m.aod500
    T_a  = exp(-(τa^0.873) * (1 + τa - τa^0.7088) * am^0.9108)            # aerosol

    dni = 0.9662 * etr * T_r * T_o * T_um * T_w * T_a

    # aerosol absorptance / scattering split
    T_aa = 1 - 0.1 * (1 - am + am^1.06) * (1 - T_a)
    T_as = T_a / T_aa

    # diffuse
    I_as = etr * μ * 0.79 * T_o * T_um * T_w * T_aa *
           (0.5 * (1 - T_r) + m.ba * (1 - T_as)) /
           (1 - am + am^1.02)
    rs = 0.0685 + (1 - m.ba) * (1 - T_as)          # sky reflectance
    denom = 1 - m.albedo * rs
    ghi = (dni * μ + I_as) / denom
    dhi = max(0.0, ghi - dni * μ)
    return (dni = dni, dhi = dhi, ghi = ghi)
end

"""
    clearness_index(ghi, p::SolarPosition; S₀ = SOLAR_CONSTANT) -> kt

Clearness index `kt = GHI / (E₀ μ₀)` — the fraction of the
extraterrestrial horizontal irradiance that reaches the ground. Near 0.75
for a clear sky, ~0.2 for heavy overcast. The standard predictor for the
diffuse fraction. Returns 0 when the sun is below the horizon.
"""
function clearness_index(ghi::Real, p::SolarPosition; S₀::Real = SOLAR_CONSTANT)
    p.μ₀ <= 0 && return 0.0
    e_h = extraterrestrial_irradiance(p; S₀ = S₀) * p.μ₀
    e_h <= 0 && return 0.0
    return clamp(ghi / e_h, 0.0, 1.2)
end

"""
    diffuse_fraction_erbs(kt) -> DHI/GHI

Erbs, Klein & Duffie (1982) diffuse-fraction correlation — the most
widely used decomposition of a measured global irradiance:

```
kt ≤ 0.22 :  1 − 0.09 kt
0.22 < kt ≤ 0.80 : 0.9511 − 0.1604kt + 4.388kt² − 16.638kt³ + 12.336kt⁴
kt > 0.80 : 0.165
```
"""
function diffuse_fraction_erbs(kt::Real)
    kt = clamp(kt, 0.0, 1.0)
    if kt <= 0.22
        return 1 - 0.09kt
    elseif kt <= 0.80
        return 0.9511 - 0.1604kt + 4.388kt^2 - 16.638kt^3 + 12.336kt^4
    else
        return 0.165
    end
end

"""
    diffuse_fraction_reindl(kt, μ₀) -> DHI/GHI

Reindl, Beckman & Duffie (1990) correlation, which adds a solar-elevation
dependence to Erbs and performs better at low sun — the regime that
matters for urban canyons.
"""
function diffuse_fraction_reindl(kt::Real, μ₀::Real)
    kt = clamp(kt, 0.0, 1.0)
    s = clamp(μ₀, 0.0, 1.0)
    f = if kt <= 0.3
        1.020 - 0.254kt + 0.0123s
    elseif kt < 0.78
        1.400 - 1.749kt + 0.177s
    else
        0.486kt - 0.182s
    end
    return clamp(f, 0.0, 1.0)
end

"""
    partition_global_irradiance(ghi, p; method = :reindl, S₀ = SOLAR_CONSTANT)
        -> (; dni, dhi, ghi, kt, diffuse_fraction)

Split a measured or modelled global horizontal irradiance into its direct
and diffuse components. `method` is `:erbs` or `:reindl`.

This is the routine a real driven simulation needs: observations and
reanalysis give GHI, but a radiative-transfer solver needs the direct
beam separately because only the direct beam casts shadows.
"""
function partition_global_irradiance(ghi::Real, p::SolarPosition;
                                     method::Symbol = :reindl,
                                     S₀::Real = SOLAR_CONSTANT)
    kt = clearness_index(ghi, p; S₀ = S₀)
    fd = method === :erbs   ? diffuse_fraction_erbs(kt) :
         method === :reindl ? diffuse_fraction_reindl(kt, p.μ₀) :
         error("partition_global_irradiance: method must be :erbs or :reindl")
    dhi = fd * ghi
    dni = p.μ₀ > 1e-6 ? (ghi - dhi) / p.μ₀ : 0.0
    return (dni = max(0.0, dni), dhi = dhi, ghi = ghi,
            kt = kt, diffuse_fraction = fd)
end

# =========================================================================
# 3-D cloud shadows
# =========================================================================

"""
    cloud_optical_depth_column(ql, qi, ρ, dz; k_liq = liquid_mass_extinction(10),
                               k_ice = ice_mass_extinction(30))

Per-cell cloud optical depth from the condensate field, `τ = k · WC · dz`
with the mass extinction `k` in m²/g and the water content `WC` in g/m³.
The defaults are the `RadiationCore` coefficients for effective radii of
10 μm (liquid, 0.15 m²/g) and 30 μm (ice, 0.033 m²/g), the same ones
`TwoStream.MieCloudOptics` uses, so a column scheme and the 3-D shadow
see the same cloud. (Before #130 the defaults were 0.06 and 0.02 m²/g,
which the docstring called "~10 μm" but correspond to 25 μm liquid.)
This lightweight form exists so shadow geometry can be computed without
instantiating the whole optics pipeline.

`ql`, `qi` are mass mixing ratios (kg/kg), `ρ` is air density (kg/m³),
a scalar, a per-level vector or a 3-D array, `dz` the layer thickness
(m). Returns a 3-D array of optical depths.
"""
function cloud_optical_depth_column(ql::AbstractArray{T,3},
                                    qi::AbstractArray{T,3},
                                    ρ, dz;
                                    k_liq::Real = liquid_mass_extinction(10.0),
                                    k_ice::Real = ice_mass_extinction(30.0)) where {T}
    τ = similar(ql)
    nx, ny, nz = size(ql)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ρk = ρ isa Real ? ρ : (ρ isa AbstractVector ? ρ[k] : ρ[i, j, k])
        hk = dz isa Real ? dz : dz[k]
        lwc = ql[i, j, k] * ρk * 1000      # kg/kg → g/m³
        iwc = qi[i, j, k] * ρk * 1000
        τ[i, j, k] = (k_liq * lwc + k_ice * iwc) * hk
    end
    return τ
end

"""
    cloud_shadow_transmittance!(trans, τ_cloud, mesh, sun_dir; k_start = nothing)

Fill `trans[i,j] ∈ (0,1]` with the **direct-beam transmittance** reaching
each surface cell, by tracing the ray from the centre of the cell's
bottom face (z = 0) toward the sun through the 3-D cloud optical-depth
field `τ_cloud` and accumulating the slant optical depth:

```math
T_{i,j} = \\exp\\!\\left(-\\sum_\\text{cells c on the ray} \\frac{τ_c}{Δz}\\,ℓ_c\\right)
```

`τ_c` is the vertical optical depth of cell `c` (its extinction
coefficient is `τ_c/Δz`) and `ℓ_c` the exact length of the ray inside
it, found by a grid traversal (Amanatides & Woo 1987). For a horizontally
uniform layer of vertical optical depth `τ` this is exactly
`exp(-τ/μ₀)`, with `μ₀` the `up` component of `sun_dir`: the slant
amplification comes from the path lengths, and is not applied again.

`sun_dir = (east, north, up)` is the unit vector *toward* the sun (from
[`sun_direction_enu`](@ref)). Lateral boundaries wrap periodically, matching
the LES. If the sun is at or below the horizon, `trans` is filled with 0.

`k_start[i, j]` (optional) starts the ray of column `(i, j)` at height
`k_start·Δz` instead of the ground: on a roof, pass the number of solid
cells under it, so the roof sees only the cloud above it. Nothing blocks
the ray (for opaque buildings see
[`combined_shadow_transmittance!`](@ref)).

This is genuinely 3-D: a cloud displaced horizontally from a surface cell
shadows it when the sun is low, which no column-by-column (independent
column) calculation can represent.
"""
function cloud_shadow_transmittance!(trans::AbstractMatrix{T},
                                     τ_cloud::AbstractArray{T,3},
                                     mesh::RegularGrid,
                                     sun_dir::NTuple{3,<:Real};
                                     k_start::Union{Nothing,AbstractMatrix{<:Integer}} = nothing) where {T}
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    size(trans) == (Nx, Ny) ||
        error("cloud_shadow_transmittance!: trans is $(size(trans)), expected ($Nx, $Ny)")
    size(τ_cloud) == (Nx, Ny, Nz) ||
        error("cloud_shadow_transmittance!: τ_cloud is $(size(τ_cloud)), expected ($Nx, $Ny, $Nz)")
    k_start === nothing &&
        return combined_shadow_transmittance!(trans, τ_cloud, nothing, mesh, sun_dir)
    size(k_start) == (Nx, Ny) ||
        error("cloud_shadow_transmittance!: k_start is $(size(k_start)), expected ($Nx, $Ny)")
    sx, sy, sz = Float64.(sun_dir)
    if sz <= 1e-6
        fill!(trans, zero(T))
        return trans
    end
    dx, dy, dz = Float64(mesh.dx), Float64(mesh.dy), Float64(mesh.dz)
    @inbounds for j in 1:Ny, i in 1:Nx
        τ_slant, _ = _shadow_ray(τ_cloud, nothing, i, j, Int(k_start[i, j]), sx, sy, sz,
                                 dx, dy, dz, Nx, Ny, Nz, mesh.Hx, mesh.Hy, mesh.Hz)
        trans[i, j] = T(exp(-τ_slant))
    end
    return trans
end

"""
    cloud_shadow_map(τ_cloud, mesh, sun_dir) -> Matrix

Allocating form of [`cloud_shadow_transmittance!`](@ref).
"""
function cloud_shadow_map(τ_cloud::AbstractArray{T,3}, mesh::RegularGrid,
                          sun_dir::NTuple{3,<:Real}) where {T}
    trans = Matrix{T}(undef, mesh.Nx, mesh.Ny)
    return cloud_shadow_transmittance!(trans, τ_cloud, mesh, sun_dir)
end

"""
    combined_shadow_transmittance!(trans, τ_cloud, solid_mask, mesh, sun_dir)

Direct-beam transmittance including **both** cloud attenuation and opaque
building blocking, in a single ray traversal. The ray starts at the top
of the column's solid stack (the roof, or the ground at z = 0), at the
centre of the cell, and runs toward the sun until it leaves the top of
the domain. A cell is fully shadowed (`trans = 0`) if the ray enters a
solid cell; otherwise the cloud slant optical depth attenuates it, with
the exact path length in every cell as in
[`cloud_shadow_transmittance!`](@ref).

`solid_mask` is a `Bool` array with the halo layout of the mesh (i.e. the
`.mask` field of an `IBM.SolidMask`); pass `nothing` for no buildings,
and `τ_cloud = nothing` for no cloud. Doing the two effects in one march
is not just an optimisation — it is also the only way to get the
ordering right when a cloud shadow and a building shadow overlap.
"""
function combined_shadow_transmittance!(trans::AbstractMatrix{T},
                                        τ_cloud::Union{Nothing,AbstractArray{T,3}},
                                        solid_mask::Union{Nothing,AbstractArray{Bool,3}},
                                        mesh::RegularGrid,
                                        sun_dir::NTuple{3,<:Real}) where {T}
    sx, sy, sz = Float64.(sun_dir)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
    if sz <= 1e-6
        fill!(trans, zero(T))
        return trans
    end
    dx, dy, dz = Float64(mesh.dx), Float64(mesh.dy), Float64(mesh.dz)
    @inbounds for j in 1:Ny, i in 1:Nx
        # start on top of the local solid stack so a rooftop is not
        # shadowed by its own building
        k_top = 0
        if solid_mask !== nothing
            for k in Nz:-1:1
                if solid_mask[Hx + i, Hy + j, Hz + k]; k_top = k; break; end
            end
        end
        τ_slant, blocked = _shadow_ray(τ_cloud, solid_mask, i, j, k_top,
                                       sx, sy, sz, dx, dy, dz, Nx, Ny, Nz, Hx, Hy, Hz)
        trans[i, j] = blocked ? zero(T) : T(exp(-τ_slant))
    end
    return trans
end

# Exact traversal of the ray from the centre of cell (i, j)'s face at
# height k0·dz toward the sun (Amanatides & Woo 1987), with periodic
# lateral wrap. Returns the slant optical depth Σ (τ_c/dz)·ℓ_c and
# whether the ray entered a solid cell.
function _shadow_ray(τ_cloud, solid_mask, i::Int, j::Int, k0::Int,
                     sx::Float64, sy::Float64, sz::Float64,
                     dx::Float64, dy::Float64, dz::Float64,
                     Nx::Int, Ny::Int, Nz::Int, Hx::Int, Hy::Int, Hz::Int)
    # unwrapped 0-based cell indices and the ray parameter (path length)
    # at which the ray crosses the next x, y and z face
    ix, iy, iz = i - 1, j - 1, k0
    stx = sx > 0 ? 1 : -1
    sty = sy > 0 ? 1 : -1
    tdx = sx != 0 ? dx / abs(sx) : Inf
    tdy = sy != 0 ? dy / abs(sy) : Inf
    tdz = dz / sz
    tmx = sx != 0 ? 0.5 * tdx : Inf          # start at the cell centre in x, y
    tmy = sy != 0 ? 0.5 * tdy : Inf
    tmz = tdz                                # and on the bottom face in z
    t = 0.0
    τ = 0.0
    @inbounds while iz < Nz
        ci = mod(ix, Nx) + 1
        cj = mod(iy, Ny) + 1
        ck = iz + 1
        if solid_mask !== nothing && solid_mask[Hx + ci, Hy + cj, Hz + ck]
            return τ, true
        end
        tn = min(tmx, tmy, tmz)
        τ_cloud === nothing || (τ += τ_cloud[ci, cj, ck] / dz * (tn - t))
        t = tn
        if tmx == tn
            ix += stx; tmx += tdx
        elseif tmy == tn
            iy += sty; tmy += tdy
        else
            iz += 1; tmz += tdz
        end
    end
    return τ, false
end

end # module
