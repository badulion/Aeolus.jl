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

using ..Superdroplet: terminal_velocity, DEFAULT_MICRO, _VTERM_REGIME_D

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

# Rain velocity has kinks where Superdroplet's drag regimes meet; the
# diameters are Superdroplet's own, so the two cannot drift apart.
const _P3_RAIN_BREAKS = _VTERM_REGIME_D

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

# -------------------------------------------------------------------------
# Column sedimentation
# -------------------------------------------------------------------------

"""
    p3_sediment!(fields, velocities, ρ, dz, dt; courant_max = 1) -> (surface, nsub)

Sediment the columns in the tuple `fields` (vectors of specific
quantities per kg of air, e.g. `q_r`, `N_r`) over time `dt` (s) with
first-order upwind, flux-form finite volumes, sub-stepping so that the
Courant number never exceeds `courant_max ≤ 1`.

`velocities(fields)` must return a tuple, one vector per field, of fall
speeds (m s⁻¹, ≥ 0, positive downward) at cell centres; it is called
again at every sub-step, so the speeds follow the evolving PSD (as in
the P3 reference code). `ρ` is air density (kg m⁻³) and `dz` layer
thickness (m, scalar or vector); cell `k = 1` is at the surface.

For each field `ψ` the update over a sub-step `h` is

    ρ_k ψ_k Δz_k ← ρ_k ψ_k Δz_k − h F_k + h F_{k+1},   F_k = ρ_k ψ_k V_k,

with `F_{nz+1} = 0` (nothing enters through the model top) and `h F_1`
leaving through the surface. Properties, all exact in floating point up
to round-off:

  * **conservative** — `Σ ρ ψ Δz` changes only by the surface flux;
  * **positive** — `h` is chosen so `C_k = V_k h / Δz_k ≤ courant_max`,
    whence the outflow never exceeds the cell content; the outflow is
    also capped at the content so rounding cannot drive `ψ` negative.

The sub-step is `h = min(remaining, courant_max · min_k Δz_k / V_k)`,
recomputed each sub-step. Returns the time-integrated surface flux of
each field (`Σ h F_1`, e.g. kg m⁻² of precipitation for `q`) and the
number of sub-steps taken.

Reference: Morrison & Milbrandt (2015), §2d; the P3 reference code
sub-steps its sedimentation in the same way.
"""
function p3_sediment!(fields::Tuple, velocities, ρ::AbstractVector,
                      dz, dt::Real; courant_max::Real = 1)
    nf = length(fields)
    nz = length(ρ)
    all(length(f) == nz for f in fields) ||
        throw(DimensionMismatch("fields and ρ must have the same length"))
    dz isa AbstractVector && length(dz) != nz &&
        throw(DimensionMismatch("dz must be a scalar or have length(ρ)"))
    0 < courant_max <= 1 ||
        throw(ArgumentError("courant_max must lie in (0, 1] for positivity"))
    T = promote_type(eltype(ρ), map(eltype, fields)...)
    Δz(k) = dz isa AbstractVector ? T(dz[k]) : T(dz)
    surface = zeros(T, nf)
    remaining = T(dt)
    nsub = 0
    flux = zeros(T, nz + 1)
    while remaining > 0
        V = velocities(fields)
        length(V) == nf ||
            throw(DimensionMismatch("velocities must return one vector per field"))
        # Largest stable sub-step over every field and cell.
        rate = zero(T)
        for i in 1:nf, k in 1:nz
            v = T(V[i][k])
            v >= 0 || throw(DomainError(v, "fall speeds must be ≥ 0"))
            rate = max(rate, v / Δz(k))
        end
        h = rate > 0 ? min(remaining, T(courant_max) / rate) : remaining
        last_step = h >= remaining
        for i in 1:nf
            ψ = fields[i]
            v = V[i]
            # Outflow through each cell's lower face: at most the content.
            for k in 1:nz
                content = ρ[k] * max(ψ[k], zero(T)) * Δz(k)
                flux[k] = min(h * ρ[k] * max(ψ[k], zero(T)) * T(v[k]), content)
            end
            flux[nz + 1] = zero(T)
            surface[i] += flux[1]
            for k in 1:nz
                mass = ρ[k] * ψ[k] * Δz(k) - flux[k] + flux[k + 1]
                ψ[k] = mass / (ρ[k] * Δz(k))
            end
        end
        nsub += 1
        remaining = last_step ? zero(T) : remaining - h
    end
    return (surface, nsub)
end

# --- rain -----------------------------------------------------------------

"""
    p3_rain_sedimentation_velocities(q_r, N_r, ρ_air; μ_r = 0, q_small = 1e-14) -> (V_N, V_M)

Per-cell number- and mass-weighted rain fall speeds (m s⁻¹) for columns
of `q_r` (kg kg⁻¹) and `N_r` (kg⁻¹), from
[`p3_rain_size_distribution`](@ref) and [`p3_rain_bulk_fall_speed`](@ref)
with the local air-density correction. Cells with `q_r < q_small` or no
number get zero speed.
"""
function p3_rain_sedimentation_velocities(q_r::AbstractVector, N_r::AbstractVector,
                                          ρ_air::AbstractVector;
                                          μ_r::Real = 0, q_small::Real = 1e-14)
    T = promote_type(eltype(q_r), eltype(N_r), eltype(ρ_air))
    VN = zeros(T, length(q_r))
    VM = zeros(T, length(q_r))
    for k in eachindex(q_r)
        q_r[k] >= q_small || continue
        psd = p3_rain_size_distribution(q_r[k], N_r[k]; μ_r = μ_r)
        psd === nothing && continue
        _, μ, λ = psd
        vn, vm, _ = p3_rain_bulk_fall_speed(μ, λ; ρ_air = ρ_air[k])
        VN[k], VM[k] = vn, vm
    end
    return (VN, VM)
end

"""
    p3_sediment_rain!(q_r, N_r, ρ, dz, dt; μ_r = 0, courant_max = 1, limit_number = true) -> (P_q, P_N, nsub)

Sediment two-moment rain in a column: `q_r` (kg kg⁻¹) falls at the
mass-weighted and `N_r` (kg⁻¹) at the number-weighted fall speed
([`p3_rain_sedimentation_velocities`](@ref)), both recomputed every
sub-step by [`p3_sediment!`](@ref). Returns the accumulated surface
precipitation `P_q` (kg m⁻², i.e. mm of water), the number of drops
reaching the ground per m² `P_N`, and the sub-step count.

With `limit_number = true` (default, as in P3) `N_r` is passed through
[`p3_rain_limit_number`](@ref) before every sub-step and at the end, so
that the size-sorted leading edge of a shaft cannot hold drops with a
mean size above `D_mean_max`; mass is conserved exactly either way,
number only with `limit_number = false`.
"""
function p3_sediment_rain!(q_r::AbstractVector, N_r::AbstractVector,
                           ρ::AbstractVector, dz, dt::Real;
                           μ_r::Real = 0, courant_max::Real = 1,
                           limit_number::Bool = true)
    limit!() = limit_number &&
        (N_r .= p3_rain_limit_number.(q_r, N_r; μ_r = μ_r))
    function vel(f)
        limit!()
        return reverse(p3_rain_sedimentation_velocities(f[1], f[2], ρ; μ_r = μ_r))
    end
    surf, nsub = p3_sediment!((q_r, N_r), vel, ρ, dz, dt; courant_max = courant_max)
    limit!()
    return (surf[1], surf[2], nsub)
end

# --- ice ------------------------------------------------------------------

"""
    p3_ice_sedimentation_velocities(q_i, q_rim, B_rim, N_i, ρ_air; T_air = nothing, μ = 0, p = P3_DEFAULT_ICE, q_small = 1e-14) -> (V_N, V_M)

Per-cell number- and mass-weighted P3 ice fall speeds. In each cell the
mass-dimension state is diagnosed from the prognostic rime variables
([`p3_mass_dimension`](@ref)), the slope from `q_i/N_i` with P3's mean
size limiter `(μ+1)/λ ∈ [2 μm, 5 mm + 20 mm·F_rim²]`
([`p3_size_distribution`](@ref)), and `V_N, V_M` from
[`p3_bulk_fall_speed`](@ref) at the local air density and — when the
temperature column `T_air` (K) is given — viscosity
([`p3_air_viscosity`](@ref); otherwise at `p.T_ref`).
"""
function p3_ice_sedimentation_velocities(q_i::AbstractVector, q_rim::AbstractVector,
                                         B_rim::AbstractVector, N_i::AbstractVector,
                                         ρ_air::AbstractVector;
                                         T_air = nothing, μ::Real = 0,
                                         p::P3IceParameters = P3_DEFAULT_ICE,
                                         q_small::Real = 1e-14)
    Tf = promote_type(eltype(q_i), eltype(N_i), eltype(ρ_air))
    VN = zeros(Tf, length(q_i))
    VM = zeros(Tf, length(q_i))
    for k in eachindex(q_i)
        (q_i[k] >= q_small && N_i[k] > 0) || continue
        md = p3_mass_dimension(q_i[k], max(q_rim[k], 0), max(B_rim[k], 0), p)
        λmin, λmax = _p3_ice_λ_limits(μ, md.F_rim)
        _, μk, λ = p3_size_distribution(q_i[k], N_i[k], md; μ = μ,
                                        λ_min = λmin, λ_max = λmax)
        η = p3_air_viscosity(T_air === nothing ? p.T_ref : T_air[k])
        vn, vm = p3_bulk_fall_speed(μk, λ, md; ρ_air = ρ_air[k], η_air = η)
        VN[k], VM[k] = vn, vm
    end
    return (VN, VM)
end

"""
    p3_sediment_ice!(q_i, q_rim, B_rim, N_i, ρ, dz, dt; T_air = nothing, μ = 0, p = P3_DEFAULT_ICE, courant_max = 1) -> (P_q, P_rim, P_B, P_N, nsub)

Sediment the single P3 ice category: `q_i`, `q_rim` and `B_rim` fall
together at the mass-weighted speed — so the rime fraction and rime
density of the falling mass are carried unchanged — and `N_i` at the
number-weighted speed ([`p3_ice_sedimentation_velocities`](@ref)),
recomputed every sub-step by [`p3_sediment!`](@ref). Returns the
surface accumulations of each field and the sub-step count.

Unlike rain, `N_i` is not adjusted here: the ice slope limiter bounds
the diagnosed fall speeds, and the matching number adjustment belongs
with the process rates.

Reference: Morrison & Milbrandt (2015), §2d.
"""
function p3_sediment_ice!(q_i::AbstractVector, q_rim::AbstractVector,
                          B_rim::AbstractVector, N_i::AbstractVector,
                          ρ::AbstractVector, dz, dt::Real;
                          T_air = nothing, μ::Real = 0,
                          p::P3IceParameters = P3_DEFAULT_ICE,
                          courant_max::Real = 1)
    function vel(f)
        VN, VM = p3_ice_sedimentation_velocities(f[1], f[2], f[3], f[4], ρ;
                                                 T_air = T_air, μ = μ, p = p)
        return (VM, VM, VM, VN)
    end
    surf, nsub = p3_sediment!((q_i, q_rim, B_rim, N_i), vel, ρ, dz, dt;
                              courant_max = courant_max)
    return (surf[1], surf[2], surf[3], surf[4], nsub)
end
