# P3 part-file: per-moment sedimentation velocities, column sedimentation
# with Courant sub-stepping, and radar reflectivity (issue #4).
#
# Two ideas carry this file.
#
#  1. Each prognostic moment falls at its *own* bulk speed. Number N
#     falls at the number-weighted V_N, mass q (and rime mass q_rim, rime
#     volume B_rim) at the mass-weighted V_M, reflectivity at V_Z. Because
#     big particles fall faster, V_Z ≥ V_M ≥ V_N for any PSD, with
#     equality only when all particles are the same size. The difference
#     V_M - V_N *is* size sorting: mass outruns number, so the leading edge
#     of a precipitation shaft carries few, large particles (Milbrandt &
#     Yau 2005; Morrison & Milbrandt 2015, §2d).
#
#  2. Radar reflectivity is the observable P3 is validated against. For
#     rain it is the Rayleigh sixth moment; for ice the "equivalent"
#     reflectivity of a sphere of solid ice of the same mass, scaled by
#     the dielectric factor |K_i|²/|K_w|² (Smith 1984).
#
# Column convention: index k = 1 is the lowest (surface) cell, k = nz the
# top, as in `Microphysics.apply_kessler_step!`. Velocities are fall
# speeds, positive downward.

using ..Superdroplet: terminal_velocity, DEFAULT_MICRO

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

"""
    P3_K2_WATER = 0.93

Dielectric factor `|K_w|² = |(ε-1)/(ε+2)|²` of liquid water at
centimetre (S/C-band) wavelengths, the value radar reflectivity is
conventionally normalised by (Battan 1973; Smith 1984).
"""
const P3_K2_WATER = 0.93

"""
    P3_K2_ICE = 0.176

Dielectric factor `|K_i|²` of solid ice (density 917 kg m⁻³) at
centimetre wavelengths (Smith 1984). The ratio `|K_i|²/|K_w|² = 0.189`
is the factor that appears in the P3 reference code's ice reflectivity.
"""
const P3_K2_ICE = 0.176

# Reference air density of Superdroplet.terminal_velocity (sea level).
_p3_rain_ρ_ref() = DEFAULT_MICRO.ρ_air

# Rain velocity has kinks where Superdroplet's drag regimes meet.
const _P3_RAIN_BREAKS = (19.5e-6, 1.07e-3, 1.4e-3)

# -------------------------------------------------------------------------
# Rain: fall speed and size distribution
# -------------------------------------------------------------------------

"""
    p3_rain_fall_speed(D; ρ_air = 1.225) -> m s⁻¹

Terminal fall speed of a raindrop of diameter `D` (m) in air of density
`ρ_air` (kg m⁻³). The sea-level speed is `Superdroplet.terminal_velocity`
(Stokes / Beard 1976 / Atlas et al. 1973 — one source of truth for drop
fall speeds in the package), corrected for air density with the
Foote & du Toit (1969) factor used in P3,

    V(D, ρ_a) = V(D, ρ₀) (ρ₀/ρ_a)^0.54,   ρ₀ = 1.225 kg m⁻³.

A 1 mm drop falls at ≈ 4.0 m s⁻¹ and large drops level off near
9.2-9.6 m s⁻¹ at sea level.

References: Beard (1976, JAS 33, 851-864); Atlas, Srivastava & Sekhon
(1973, Rev. Geophys. 11, 1-35); Foote & du Toit (1969, JAM 8, 249-253).
"""
function p3_rain_fall_speed(D::Real; ρ_air::Real = _p3_rain_ρ_ref())
    Df = float(D)
    Df > 0 || return zero(Df)
    v0 = terminal_velocity(Df / 2)
    return oftype(Df, v0 * (_p3_rain_ρ_ref() / ρ_air)^0.54)
end

"""
    p3_rain_size_distribution(q_r, N_r; μ_r = 0, D_mean_min = 1e-5, D_mean_max = 8e-4) -> (N0, μ, λ)

Gamma size distribution `N(D) = N₀ D^μ e^{-λD}` of rain with mass `q_r`
and number `N_r` (both per kg of air, or both per m³), drops being
spheres of liquid water (`ρ_w = Superdroplet.DEFAULT_MICRO.ρ_water`):

    λ = [ (π/6) ρ_w N_r Γ(μ+4) / (q_r Γ(μ+1)) ]^{1/3}

([`gamma_psd_slope`](@ref)), `N₀` from [`gamma_psd_intercept`](@ref).
The mean diameter `(μ+1)/λ` is limited to `[D_mean_min, D_mean_max]`,
defaults 10 μm and 0.8 mm, i.e. `λ ∈ (μ+1)·[1250, 10⁵] m⁻¹`, the rain
slope limits of the P3 reference code. The upper size limit matters in
sedimentation: number falls slower than mass, so the leading edge of a
rain shaft is left with mass but almost no number, and without a limit
its diagnosed drops grow without bound (the "excessive size sorting" of
two-moment schemes; Wacker & Seifert 2001, Milbrandt & McTaggart-Cowan
2010). When the limit is active `N₀` is set from `N_r`, so the returned
PSD carries the number but not exactly the mass; to restore consistency
the caller adjusts `N_r` with [`p3_rain_limit_number`](@ref), as P3 does.

Returns `nothing` if `q_r ≤ 0` or `N_r ≤ 0`.

Reference: Morrison & Milbrandt (2015), §2c; Seifert & Beheng (2006).
"""
function p3_rain_size_distribution(q_r::Real, N_r::Real; μ_r::Real = 0,
                                   D_mean_min::Real = 1e-5, D_mean_max::Real = 8e-4)
    (q_r > 0 && N_r > 0) || return nothing
    q, N = promote(float(q_r), float(N_r))
    T = typeof(q)
    μ = T(μ_r)
    λ = gamma_psd_slope(N, 0, q / _p3_cw(T), 3, μ)
    λ = clamp(λ, (μ + 1) / T(D_mean_max), (μ + 1) / T(D_mean_min))
    return (gamma_psd_intercept(N, μ, λ), μ, λ)
end

_p3_cw(::Type{T}) where {T} = T(π) / 6 * T(DEFAULT_MICRO.ρ_water)

"""
    p3_rain_limit_number(q_r, N_r; μ_r = 0, D_mean_min = 1e-5, D_mean_max = 8e-4) -> N_r′

Rain number made consistent with the mean-size limits of
[`p3_rain_size_distribution`](@ref): if `q_r/N_r` implies a mean
diameter `(μ+1)/λ` outside `[D_mean_min, D_mean_max]`, return the number
that puts it exactly on the limit, keeping the mass,

    N_r′ = q_r Γ(μ+1) λ_lim³ / ((π/6) ρ_w Γ(μ+4));

otherwise `N_r` unchanged. Rain with mass but no number (`N_r ≤ 0`) is
given the number of the largest allowed drops. Mass is never changed;
number is not conserved by this step — the same trade the P3 reference
code makes when it limits `λ_r`.
"""
function p3_rain_limit_number(q_r::Real, N_r::Real; μ_r::Real = 0,
                              D_mean_min::Real = 1e-5, D_mean_max::Real = 8e-4)
    q, N = promote(float(q_r), float(N_r))
    T = typeof(q)
    q > 0 || return N
    μ = T(μ_r)
    lg = lgamma_lanczos(μ + 1) - lgamma_lanczos(μ + 4)
    Nlim(λ) = q / _p3_cw(T) * exp(lg + 3 * log(λ))
    Nmin = Nlim((μ + 1) / T(D_mean_max))
    Nmax = Nlim((μ + 1) / T(D_mean_min))
    return clamp(N, Nmin, Nmax)
end

"""
    p3_rain_bulk_fall_speed(μ, λ; ρ_air = 1.225) -> (V_N, V_M, V_Z)

Number-, mass- and reflectivity-weighted fall speeds (m s⁻¹) of a gamma
rain size distribution `N(D) ∝ D^μ e^{-λD}`:

    V_N = ∫ V N dD / ∫ N dD,
    V_M = ∫ V D³ N dD / ∫ D³ N dD,
    V_Z = ∫ V D⁶ N dD / ∫ D⁶ N dD,

with `V(D)` from [`p3_rain_fall_speed`](@ref). `V_N` moves `N_r`, `V_M`
moves `q_r`, `V_Z` a third (reflectivity) moment. Since `V(D)` increases
with `D`, `V_N ≤ V_M ≤ V_Z`, equal only in the monodisperse limit.
Integrated by composite Gauss-Legendre quadrature split at the kinks of
`V(D)`, as for ice ([`p3_bulk_fall_speed`](@ref)).

Reference: Morrison & Milbrandt (2015), §2d; Milbrandt & Yau (2005, JAS
62, 3051-3064) for multi-moment sedimentation.
"""
function p3_rain_bulk_fall_speed(μ::Real, λ::Real; ρ_air::Real = _p3_rain_ρ_ref())
    μT, λT = promote(float(μ), float(λ))
    T = typeof(μT)
    ρa = T(ρ_air)
    VN, V3, M3, V6, M6 = _p3_psd_expectation(μT, λT, _P3_RAIN_BREAKS; kmax = 6) do D
        v = p3_rain_fall_speed(D; ρ_air = ρa)
        D3 = D^3
        D6 = D3 * D3
        (v, v * D3, D3, v * D6, D6)
    end
    return (VN, V3 / M3, V6 / M6)
end

"""
    p3_rain_rate(N0, μ, λ; ρ_air = 1.225) -> mm h⁻¹

Surface-equivalent rain rate of a gamma drop size distribution whose
intercept `N₀` is per m³ of air (m⁻⁴⁻μ):

    R = ∫ (π/6) D³ V(D) N(D) dD   (m³ of water m⁻² s⁻¹) × 3.6e6.

This is `ρ_air q_r V_M / ρ_w` converted to mm h⁻¹.
"""
function p3_rain_rate(N0::Real, μ::Real, λ::Real; ρ_air::Real = _p3_rain_ρ_ref())
    _, VM, _ = p3_rain_bulk_fall_speed(μ, λ; ρ_air = ρ_air)
    vol = π / 6 * gamma_psd_moment(N0, μ, λ, 3)
    return vol * VM * 3.6e6
end

# -------------------------------------------------------------------------
# Ice: reflectivity-weighted fall speed
# -------------------------------------------------------------------------

"""
    p3_bulk_fall_speed_z(μ, λ, md::P3MassDimension; ρ_air, η_air) -> V_Z

Reflectivity-weighted fall speed (m s⁻¹) of P3 ice, weighting each
particle by `m(D)²` (the quantity equivalent reflectivity is made of, see
[`p3_ice_reflectivity`](@ref)):

    V_Z = ∫ V(D) m(D)² N(D) dD / ∫ m(D)² N(D) dD.

This is the sedimentation speed of the sixth moment in three-moment P3
(Milbrandt et al. 2021, JAS 78, 439-458, introduce it; two-moment P3 needs only `V_N`,
`V_M` from [`p3_bulk_fall_speed`](@ref)). `V_Z ≥ V_M ≥ V_N`. Air state
defaults as in [`p3_fall_speed`](@ref).
"""
function p3_bulk_fall_speed_z(μ::Real, λ::Real, md::P3MassDimension{T};
                              ρ_air::Real = _p3_ref_air(md.params)[1],
                              η_air::Real = _p3_ref_air(md.params)[2]) where {T}
    ρa, η = T(ρ_air), T(η_air)
    p = md.params
    VZ, Z = _p3_psd_expectation(T(μ), T(λ), md; kmax = 2 * max(3, p.β_va)) do D
        m = p3_particle_mass(D, md)
        v = _p3_vt(D, m, p3_particle_area(D, md), ρa, η, p)
        (v * m * m, m * m)
    end
    return VZ / Z
end

# -------------------------------------------------------------------------
# Radar reflectivity
# -------------------------------------------------------------------------

"""
    p3_rain_reflectivity(N0, μ, λ) -> Z in mm⁶ m⁻³

Rayleigh radar reflectivity factor of rain with gamma drop size
distribution `N(D) = N₀ D^μ e^{-λD}`, `N₀` per m³ of air:

    Z = ∫ D⁶ N(D) dD = N₀ Γ(μ+7) / λ^{μ+7}   (m⁶ m⁻³),

returned in the radar unit mm⁶ m⁻³ (× 10¹⁸). Valid where drops are small
compared with the wavelength (Rayleigh regime; S-band and, to within
~1 dB at moderate rain rates, C-band). Convert with [`p3_dbz`](@ref).

Reference: Battan (1973), *Radar Observation of the Atmosphere*;
Marshall & Palmer (1948, J. Meteor. 5, 165-166).
"""
p3_rain_reflectivity(N0::Real, μ::Real, λ::Real) =
    1e18 * gamma_psd_moment(N0, μ, λ, 6)

"""
    p3_ice_reflectivity(N0, μ, λ, md::P3MassDimension; K2_ratio = P3_K2_ICE / P3_K2_WATER) -> Z_e in mm⁶ m⁻³

Equivalent radar reflectivity factor of P3 ice with gamma size
distribution `N(D) = N₀ D^μ e^{-λD}` (`N₀` per m³ of air). Each ice
particle of mass `m` scatters (Rayleigh) like a sphere of solid ice of
the same mass, i.e. of melted-equivalent diameter
`D_eq = (6m/(π ρ_ice))^{1/3}`, so

    Z_e = (|K_i|²/|K_w|²) ∫ D_eq⁶ N dD
        = (|K_i|²/|K_w|²) (6/(π ρ_ice))² ∫ m(D)² N(D) dD.

Equivalently a particle of bulk density `ρ_p` contributes
`(|K_i|²/|K_w|²)(ρ_p/ρ_ice)² D⁶`: low-density snow is a weak target for
its size. `|K_i|²/|K_w|² = 0.176/0.93 = 0.189` (Smith 1984), the factor
in the P3 reference code. Because `m(D)` is a piecewise power law, `m²`
is too and `Z_e` is a sum of truncated gamma moments
([`gamma_psd_partial_moment`](@ref)) — exact, no quadrature.

References: Smith (1984, JCAM 23, 1258-1260); Morrison & Milbrandt
(2015), §2; Heymsfield et al. (2005) for density scaling.
"""
function p3_ice_reflectivity(N0::Real, μ::Real, λ::Real, md::P3MassDimension{T};
                             K2_ratio::Real = P3_K2_ICE / P3_K2_WATER) where {T}
    N0T, μT, λT = T(N0), T(μ), T(λ)
    s = zero(T)
    for (c, b, D1, D2) in _p3_mass_segments(md)
        c == 0 && continue
        s += c^2 * gamma_psd_partial_moment(N0T, μT, λT, 2b, D1, D2)
    end
    return T(1e18) * T(K2_ratio) * (6 / (T(π) * md.params.ρ_ice))^2 * s
end

"""
    p3_dbz(Z; Z_min = 1e-10) -> dBZ

Logarithmic reflectivity `10 log₁₀(Z / 1 mm⁶ m⁻³)`. `Z` below `Z_min`
(including zero) is floored, giving -100 dBZ by default, so an empty
cell has a finite, clearly non-meteorological value.
"""
p3_dbz(Z::Real; Z_min::Real = 1e-10) = 10 * log10(max(float(Z), Z_min))
