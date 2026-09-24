# P3 part-file: secondary ice production by rime splintering (issue #7).
#
# Hallett & Mossop (1974) found that ice riming supercooled drops between
# about -3 and -8 °C throws off small ice splinters, ~350 per milligram of
# rime at the optimum near -5 °C, and none outside that window (Mossop
# 1976). It is the one secondary-ice mechanism every bulk scheme carries,
# P3 included (MM15 §2c, `microphy_p3.f90`: `nimul`, `qimul`).
#
# What is NOT recomputed here: the riming rates. The splinter source is a
# function of the P3 riming rates `rime_c`, `rime_r` of
# `p3_process_rates` (processes.jl) — the same numbers that feed the rime
# mass and volume — so the two cannot disagree about how much rime there is.

"""
    P3_NUCLEATED_ICE_MASS = 4π/3 × 900 kg m⁻³ × (1 μm)³ ≈ 3.77e-15 kg

Mass of a newly formed ice crystal in P3 — a sphere of 1 μm radius at
900 kg m⁻³ (`mi0` of the P3 reference code, Morrison & Milbrandt 2015).
It is the mass of a Hallett–Mossop splinter here and of a crystal
nucleated by deposition in `IceNucleation` (`NUCLEATED_ICE_MASS` is
defined as this constant, so the two schemes cannot drift apart).
"""
const P3_NUCLEATED_ICE_MASS = 4π / 3 * 900.0 * (1e-6)^3

"""
    P3HallettMossop{T}(; kwargs...)

Parameters of Hallett–Mossop rime splintering in P3.

| field | default | meaning |
| :-- | :-- | :-- |
| `splinters_per_kg` | 3.5e8 | splinters per kg of rime at the optimum temperature (350 per mg) |
| `T_lo`   | 265.15 | cold edge of the window, K (-8 °C) |
| `T_peak` | 268.15 | optimum, K (-5 °C) |
| `T_hi`   | 270.15 | warm edge of the window, K (-3 °C) |
| `m_splinter` | [`P3_NUCLEATED_ICE_MASS`](@ref) | mass of one splinter, kg |
| `D_drop_min` | 0 | only cloud droplets larger than this diameter (m) produce splinters; 0 = all |

The window and the 350 mg⁻¹ yield are those of Hallett & Mossop (1974)
and Mossop (1976), in the triangular form of the P3 reference code.
Mossop (1976) and Choularton et al. (1980) found splintering needs drops
larger than about 24 μm diameter; the P3 code does not apply that
restriction (`D_drop_min = 0`), setting `D_drop_min = 25e-6` applies it
to the cloud-droplet rime through the droplet gamma size distribution
(see [`p3_hm_large_drop_fraction`](@ref)). Rime collected from rain is
counted in full: rain drops are all larger than 25 μm.
"""
Base.@kwdef struct P3HallettMossop{T<:AbstractFloat}
    splinters_per_kg::T = 3.5e8
    T_lo::T = 265.15
    T_peak::T = 268.15
    T_hi::T = 270.15
    m_splinter::T = P3_NUCLEATED_ICE_MASS
    D_drop_min::T = 0.0
end

"Default Hallett–Mossop parameters (Float64), see [`P3HallettMossop`](@ref)."
const P3_DEFAULT_HM = P3HallettMossop{Float64}()

"""
    p3_hm_temperature_factor(T, hm = P3_DEFAULT_HM) -> f ∈ [0, 1]

Temperature dependence of rime splintering: a triangle that is zero at
or outside `[T_lo, T_hi]` = [-8, -3] °C and rises linearly to 1 at
`T_peak` = -5 °C from either side,

    f = (T - T_lo)/(T_peak - T_lo)     T_lo < T ≤ T_peak
    f = (T_hi - T)/(T_hi - T_peak)     T_peak < T < T_hi.

This is the shape used by the P3 reference code (`dum` in the `nimul`
block of `microphy_p3.f90`), a piecewise-linear fit to the laboratory
yield curve of Hallett & Mossop (1974, Nature 249, 26), their Fig. 2.
"""
function p3_hm_temperature_factor(T::Real, hm::P3HallettMossop = P3_DEFAULT_HM)
    t = float(T)
    (t <= hm.T_lo || t >= hm.T_hi) && return zero(t)
    t <= hm.T_peak && return (t - hm.T_lo) / (hm.T_peak - hm.T_lo)
    return (hm.T_hi - t) / (hm.T_hi - hm.T_peak)
end

# Shape parameter of the cloud-droplet gamma distribution as a function of
# droplet concentration (Martin et al. 1994, in the form of the P3 code):
# μ_c = 1/(0.0005714 N_c[cm⁻³] + 0.2714)² - 1, bounded to [2, 15].
_p3_cloud_μ(N_c_m3) = clamp(1 / (0.0005714 * N_c_m3 * 1e-6 + 0.2714)^2 - 1, 2.0, 15.0)

"""
    p3_hm_large_drop_fraction(q_c, N_c, ρ_air, D_min, pp = P3_DEFAULT_PROCESS) -> f ∈ [0, 1]

Fraction of the cloud-water mass carried by droplets larger than
`D_min` (m), for droplets distributed as a gamma PSD whose shape `μ_c`
follows Martin et al. (1994) as in the P3 code and whose slope matches
`q_c` and `N_c` (per kg) as spheres of liquid water:

    f = Q(μ_c + 4, λ_c D_min),      λ_c from gamma_psd_slope(N_c, 0, q_c/(π/6 ρ_w), 3, μ_c),

`Q` the regularised upper incomplete gamma function
([`gamma_inc_upper`](@ref)). With the geometric sweep-out kernel of P3
riming (droplet fall speed neglected) each droplet is collected in
proportion to the volume it occupies, so `f` is also the fraction of
cloud rime that comes from droplets larger than `D_min`. Returns 1 for
`D_min ≤ 0` and 0 without cloud.
"""
function p3_hm_large_drop_fraction(q_c::Real, N_c::Real, ρ_air::Real, D_min::Real,
                                   pp::P3ProcessParameters = P3_DEFAULT_PROCESS)
    D_min <= 0 && return 1.0
    (q_c > 0 && N_c > 0) || return 0.0
    μ = _p3_cloud_μ(N_c * ρ_air)
    λ = gamma_psd_slope(float(N_c), 0, q_c / (π / 6 * pp.micro.ρ_water), 3, μ)
    return gamma_inc_upper(μ + 4, λ * D_min)
end

"""
    p3_hallett_mossop(s::P3ParcelState, r, hm = P3_DEFAULT_HM, pp = P3_DEFAULT_PROCESS)
        -> (n_hm, q_hm)

Rime-splintering source of ice number `n_hm` (kg⁻¹ s⁻¹) and the mass
`q_hm = n_hm m_splinter` (kg kg⁻¹ s⁻¹) it carries, from the P3 riming
rates in `r` (the NamedTuple of [`p3_process_rates`](@ref), or its
limited form from [`p3_limit_rates`](@ref)):

    n_hm = splinters_per_kg · f(T) · (φ_c rime_c + rime_r),

`f` = [`p3_hm_temperature_factor`](@ref), `φ_c` =
[`p3_hm_large_drop_fraction`](@ref) (1 unless `hm.D_drop_min > 0`). At
-5 °C this is exactly 350 splinters per mg of rime. Zero outside
-8…-3 °C and whenever there is no riming. The riming rates are used as
given — they are not recomputed here.

How the splinters are booked is [`p3_hm_tendencies`](@ref).

References: Hallett & Mossop (1974, Nature 249, 26); Mossop (1976,
QJRMS 102, 45); Morrison & Milbrandt (2015, JAS 72, 287), §2c.
"""
function p3_hallett_mossop(s::P3ParcelState{FT}, r::NamedTuple,
                           hm::P3HallettMossop = P3_DEFAULT_HM,
                           pp::P3ProcessParameters = P3_DEFAULT_PROCESS) where {FT}
    f = p3_hm_temperature_factor(s.T, hm)
    z = zero(FT)
    (f > 0 && r.rime_c + r.rime_r > 0) || return (z, z)
    φc = one(FT)
    if hm.D_drop_min > 0 && r.rime_c > 0
        ρa = air_density(s.T, s.p, _p3_partition(s), pp.thermo)
        φc = FT(p3_hm_large_drop_fraction(s.q_c, s.N_c, ρa, hm.D_drop_min, pp))
    end
    n = FT(hm.splinters_per_kg) * FT(f) * (φc * r.rime_c + r.rime_r)
    return (n, n * FT(hm.m_splinter))
end

"""
    p3_hm_tendencies(s::P3ParcelState, r, hm = P3_DEFAULT_HM, pp = P3_DEFAULT_PROCESS)
        -> P3ParcelState

Tendencies (a `P3ParcelState` of `d/dt`, like [`p3_tendencies`](@ref))
of rime splintering, to be **added** to those of the other processes:

    dN_i   = +n_hm
    dq_rim = -q_hm
    dB_rim = -q_hm / ρ_new,     1/ρ_new = (rime_c/ρ_rime_c + rime_r/ρ_rim_rain)/(rime_c + rime_r)
    dq_i = dq_v = dq_c = dq_r = dT = 0.

A splinter is a fragment of the freezing rime, so its mass is ice that
already exists: total ice mass and every other species are unchanged,
and so is the temperature (no phase change). What changes is the split
of that ice: the splinters are small unrimed particles in P3's single
ice category, so their mass leaves the *rime* mass, and rime volume
leaves with it at the density of the rime being formed this step, which
leaves the density of the new rime — and hence the rime-volume
bookkeeping — unchanged. The mass is tiny (`350 mg⁻¹ × 3.8e-15 kg`
= 1.3×10⁻⁶ of the rime), so the rime cannot be driven negative by it.
"""
function p3_hm_tendencies(s::P3ParcelState{FT}, r::NamedTuple,
                          hm::P3HallettMossop = P3_DEFAULT_HM,
                          pp::P3ProcessParameters = P3_DEFAULT_PROCESS) where {FT}
    n, q = p3_hallett_mossop(s, r, hm, pp)
    z = zero(FT)
    dB = z
    if q > 0
        vol = r.rime_c / r.ρ_rime_c + r.rime_r / pp.ρ_rim_rain
        dB = -q * vol / (r.rime_c + r.rime_r)
    end
    return P3ParcelState{FT}(z, z, z, z, z, z, z, z, n, -q, dB)
end
