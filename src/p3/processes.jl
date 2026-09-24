# P3 part-file: microphysical process rates (issue #3).
#
# Every process moves mass between exactly two of the five water species
# (vapour q_v, cloud q_c, rain q_r, ice q_i — with rime q_rim ⊂ q_i) and
# is written as ONE non-negative rate. The tendencies of all prognostic
# variables are then linear combinations of those rates
# (`p3_tendencies`), so water conservation holds by construction and the
# latent heating is fixed by which phases a rate connects:
#
#     c_pm dT/dt = -L_v dq_v/dt + L_f dq_i/dt.
#
# Thermodynamics is not re-derived here. Saturation vapour pressures,
# latent heats, heat capacity and the saturation adjustment are those of
# the Era IV contract `MoistThermodynamics`; vapour diffusivity, thermal
# conductivity, the liquid growth resistances F_k, F_d and the drop fall
# speed are those of `Superdroplet`; the Khairoutdinov-Kogan autoconversion
# is `AerosolCloud.autoconversion_rate`. What lives here is only what P3
# adds: the integrals of those per-particle laws over the P3 ice and rain
# size distributions, and the P3 bookkeeping of number and rime.

# -------------------------------------------------------------------------
# Parameters and state
# -------------------------------------------------------------------------

"""
    P3ProcessParameters{T}(; kwargs...)

Parameters of the P3 process rates (SI units). Defaults follow the P3
reference code (`microphy_p3.f90`, Morrison & Milbrandt 2015, hereafter
MM15) unless noted.

| field | default | meaning |
| :-- | :-- | :-- |
| `q_small` | 1e-14 | mixing ratio below which a species is treated as absent, kg kg⁻¹ |
| `q_c_auto_min` | 1e-8 | cloud water below which autoconversion is off (P3 code) |
| `kk_A`, `kk_a`, `kk_b` | 1350, 2.47, 1.79 | KK2000 autoconversion `A q_c^a N_c^{-b}` (N_c in cm⁻³) |
| `kk_acc`, `kk_acc_exp` | 67, 1.15 | KK2000 accretion `67 (q_c q_r)^1.15` |
| `r_auto` | 25e-6 | radius of new drizzle drops from autoconversion, m (KK2000) |
| `k_rr` | 5.78 | rain self-collection coefficient, m³ kg⁻¹ s⁻¹ (Seifert & Beheng 2001) |
| `D_break`, `k_break` | 280e-6, 2300 | breakup onset diameter (m) and rate (m⁻¹) |
| `E_ci`, `E_ri` | 0.5, 1.0 | collection efficiencies of cloud and rain by ice |
| `ρ_rim_rain` | 900 | density of rime from collected rain, kg m⁻³ |
| `D_shed` | 1e-3 | diameter of drops shed from melting ice, m |
| `N_c_prescribed` | 200e6 | droplet number given to newly formed cloud, m⁻³ |
| `μ_r`, `μ_i` | 0, 0 | gamma shape parameters of rain and ice |
| `D_r_min`, `D_r_max` | 20e-6, 5e-3 | bounds on the rain mean-volume diameter, m |
| `cap_agg` | 0.48 | capacitance of nonspherical ice relative to a sphere, `C = 0.48 D/2` |
| `ice` | `P3_DEFAULT_ICE` | [`P3IceParameters`](@ref) |
| `thermo` | `DEFAULT_CONSTANTS` | `MoistThermodynamics.ThermodynamicConstants` |
| `micro` | `DEFAULT_MICRO` | `Superdroplet.MicrophysicalConstants` (ρ_w, D_v, K_a) |

References: Khairoutdinov & Kogan (2000, MWR 128, 229-243); Seifert &
Beheng (2001, Atmos. Res. 59-60, 265-281); Seifert (2008, JAS 65,
3608-3619); Westbrook et al. (2008, QJRMS 134, 1235) for the aggregate
capacitance; MM15.
"""
Base.@kwdef struct P3ProcessParameters{T<:AbstractFloat}
    q_small::T = 1e-14
    q_c_auto_min::T = 1e-8
    kk_A::T = 1350.0
    kk_a::T = 2.47
    kk_b::T = 1.79
    kk_acc::T = 67.0
    kk_acc_exp::T = 1.15
    r_auto::T = 25e-6
    k_rr::T = 5.78
    D_break::T = 280e-6
    k_break::T = 2300.0
    E_ci::T = 0.5
    E_ri::T = 1.0
    ρ_rim_rain::T = 900.0
    D_shed::T = 1e-3
    N_c_prescribed::T = 200e6
    μ_r::T = 0.0
    μ_i::T = 0.0
    D_r_min::T = 20e-6
    D_r_max::T = 5e-3
    cap_agg::T = 0.48
    ice::P3IceParameters{T} = P3_DEFAULT_ICE
    thermo::ThermodynamicConstants{T} = DEFAULT_CONSTANTS
    micro::MicrophysicalConstants{T} = DEFAULT_MICRO
end

"Default P3 process parameters (Float64), see [`P3ProcessParameters`](@ref)."
const P3_DEFAULT_PROCESS = P3ProcessParameters{Float64}()

"""
    P3ParcelState(; T, p, q_v = 0, q_c = 0, N_c = 0, q_r = 0, N_r = 0,
                    q_i = 0, N_i = 0, q_rim = 0, B_rim = 0)

Thermodynamic and microphysical state of one grid cell for the P3
scheme: temperature `T` (K), pressure `p` (Pa), and the prognostic
variables of MM15 — mixing ratios `q_*` (kg kg⁻¹), number mixing ratios
`N_*` (kg⁻¹), rime mass `q_rim` (kg kg⁻¹, part of `q_i`) and rime volume
`B_rim` (m³ kg⁻¹). All per kg of moist air. Cloud liquid, rain and ice
are two-moment; the rime pair gives the predicted rime density
`q_rim/B_rim` ([`p3_rime_state`](@ref)).

[`p3_tendencies`](@ref) returns its time derivatives in the same type.
"""
struct P3ParcelState{FT<:AbstractFloat}
    T::FT
    p::FT
    q_v::FT
    q_c::FT
    N_c::FT
    q_r::FT
    N_r::FT
    q_i::FT
    N_i::FT
    q_rim::FT
    B_rim::FT
end

function P3ParcelState(; T, p, q_v = 0.0, q_c = 0.0, N_c = 0.0, q_r = 0.0,
                       N_r = 0.0, q_i = 0.0, N_i = 0.0, q_rim = 0.0, B_rim = 0.0)
    v = promote(float(T), float(p), float(q_v), float(q_c), float(N_c),
                float(q_r), float(N_r), float(q_i), float(N_i),
                float(q_rim), float(B_rim))
    return P3ParcelState{typeof(v[1])}(v...)
end

_p3_fields(s::P3ParcelState) = ntuple(i -> getfield(s, i), fieldcount(P3ParcelState))
_p3_axpy(s::P3ParcelState{FT}, dt, ds::P3ParcelState) where {FT} =
    P3ParcelState{FT}((_p3_fields(s) .+ FT(dt) .* _p3_fields(ds))...)
_p3_with(s::P3ParcelState{FT}; kw...) where {FT} =
    P3ParcelState{FT}((haskey(kw, f) ? FT(kw[f]) : getfield(s, f)
                       for f in fieldnames(P3ParcelState))...)

"Total water `q_v + q_c + q_r + q_i` of a [`P3ParcelState`](@ref), kg kg⁻¹."
p3_total_water(s::P3ParcelState) = s.q_v + s.q_c + s.q_r + s.q_i

# Phase partition of the full state, for MoistThermodynamics (c_pm, ρ_air).
_p3_partition(s::P3ParcelState) =
    PhasePartition(p3_total_water(s), s.q_c + s.q_r, s.q_i)

# -------------------------------------------------------------------------
# Warm rain: Khairoutdinov-Kogan autoconversion and accretion
# -------------------------------------------------------------------------

"""
    p3_autoconversion(q_c, N_c, ρ_air, pp = P3_DEFAULT_PROCESS) -> (dq_r, dN_r, dN_c)

Cloud-to-rain autoconversion of Khairoutdinov & Kogan (2000, KK2000),
the formulation used by P3 (MM15 §2c):

    ∂q_r/∂t = 1350 q_c^2.47 (N_c ρ_air · 10⁻⁶)^{-1.79}       kg kg⁻¹ s⁻¹

with `q_c` in kg kg⁻¹ and the droplet number converted from kg⁻¹ to
cm⁻³. The rate itself is **delegated** to
`AerosolCloud.autoconversion_rate` (one source of truth for KK2000).
New drops are drizzle drops of radius `r_auto = 25 μm` (KK2000), so

    ∂N_r/∂t = (∂q_r/∂t) / ((4/3) π ρ_w r_auto³),
    ∂N_c/∂t = -(∂q_r/∂t) N_c / q_c

(droplets are removed at the mean droplet mass). Returns the three
tendencies as non-negative magnitudes of the rain source, rain number
source and cloud number sink.

The KK2000 regression was fitted to LES of stratocumulus with
`q_c ≲ 1 g kg⁻¹`; it has no hard threshold of its own, so, as in the P3
code, it is switched off below `q_c_auto_min = 10⁻⁸ kg kg⁻¹` (and for
`N_c ≤ 0`). The P3 code draws the new drops with *diameter* 25 μm; this
implementation follows the KK2000 paper (radius 25 μm), so it creates 8×
fewer rain drops per unit mass than the P3 code.
"""
function p3_autoconversion(q_c::Real, N_c::Real, ρ_air::Real,
                           pp::P3ProcessParameters = P3_DEFAULT_PROCESS)
    qc, Nc, ρa = float(q_c), float(N_c), float(ρ_air)
    z = zero(qc)
    (qc >= pp.q_c_auto_min && Nc > 0) || return (z, z, z)
    auto = autoconversion_rate(qc, Nc * ρa * 1e-6;
                               A = pp.kk_A, a = pp.kk_a, b = pp.kk_b)
    m0 = 4 / 3 * π * pp.micro.ρ_water * pp.r_auto^3
    return (auto, auto / m0, auto * Nc / qc)
end

"""
    p3_accretion(q_c, N_c, q_r, pp = P3_DEFAULT_PROCESS) -> (dq_r, dN_c)

Accretion of cloud water by rain, Khairoutdinov & Kogan (2000):

    ∂q_r/∂t = 67 (q_c q_r)^1.15,       ∂N_c/∂t = -(∂q_r/∂t) N_c / q_c,

mixing ratios in kg kg⁻¹. Rain number is unchanged (a drop that collects
droplets stays one drop). Zero unless both `q_c` and `q_r` exceed
`q_small`. Returns the non-negative rain source and cloud-number sink.
"""
function p3_accretion(q_c::Real, N_c::Real, q_r::Real,
                      pp::P3ProcessParameters = P3_DEFAULT_PROCESS)
    qc, Nc, qr = float(q_c), float(N_c), float(q_r)
    z = zero(qc)
    (qc >= pp.q_small && qr >= pp.q_small) || return (z, z)
    acc = pp.kk_acc * (qc * qr)^pp.kk_acc_exp
    return (acc, Nc > 0 ? acc * Nc / qc : z)
end

"""
    p3_rain_self_collection(q_r, N_r, λ_r, ρ_air, pp = P3_DEFAULT_PROCESS) -> dN_r

Net rate at which rain drop number is **lost** (kg⁻¹ s⁻¹) to
self-collection, reduced by collisional breakup, following Seifert &
Beheng (2001) with the breakup efficiency of Seifert (2008) in the form
used by the P3 code:

    ∂N_r/∂t = -φ(D_m) k_rr ρ_air q_r N_r,          k_rr = 5.78 m³ kg⁻¹ s⁻¹
    φ = 1                                   for D_m < D_break = 280 μm
    φ = 2 - exp(k_break (D_m - D_break))    otherwise,  k_break = 2300 m⁻¹

with `D_m = 1/λ_r`. Mass is unchanged. Breakup balances self-collection
(`φ = 0`) at `D_m = D_break + ln 2 / k_break = 581.4 μm`; for larger
drops the result is negative, i.e. breakup is a net number *source*,
which is what holds rain at an equilibrium mean size.
"""
function p3_rain_self_collection(q_r::Real, N_r::Real, λ_r::Real, ρ_air::Real,
                                 pp::P3ProcessParameters = P3_DEFAULT_PROCESS)
    qr, Nr = float(q_r), float(N_r)
    (qr >= pp.q_small && Nr > 0 && λ_r > 0) || return zero(qr)
    Dm = 1 / float(λ_r)
    φ = Dm < pp.D_break ? one(qr) : 2 - exp(pp.k_break * (Dm - pp.D_break))
    return φ * pp.k_rr * float(ρ_air) * qr * Nr
end

# -------------------------------------------------------------------------
# Cober & List (1993) rime density
# -------------------------------------------------------------------------

"""
    p3_cober_list_density(R_i) -> kg m⁻³

Density of rime accreted from cloud droplets as a function of the
Macklin impact parameter `R_i`, after Cober & List (1993, JAS 50,
1591-1609) in the form used by the P3 code (MM15 §2c):

    ρ_rim = (0.051 + 0.114 R_i - 0.0055 R_i²) · 10³      1 ≤ R_i ≤ 8
    ρ_rim = 611 + 72.25 (R_i - 8)                        8 < R_i ≤ 12

with `R_i` clamped to `[1, 12]`. Continuous at `R_i = 8` (611 kg m⁻³),
increasing, and bounded by 159.5 kg m⁻³ (`R_i = 1`) and 900 kg m⁻³
(`R_i = 12`), inside the P3 limits `[ρ_rim_min, ρ_rim_max] = [50, 900]`.
"""
function p3_cober_list_density(R_i::Real)
    Ri = clamp(float(R_i), 1, 12)
    return Ri <= 8 ? (0.051 + 0.114 * Ri - 0.0055 * Ri^2) * 1000 :
                     611 + 72.25 * (Ri - 8)
end

"""
    p3_rime_density(D_c, V_impact, T, pp = P3_DEFAULT_PROCESS) -> kg m⁻³

Density of rime formed by ice collecting cloud droplets of diameter
`D_c` (m) at impact speed `V_impact` (m s⁻¹) and air temperature `T`
(K). The Macklin (1962) parameter is

    R_i = -(r_c [μm]) V_impact / T_s[°C],

(droplet radius in μm, surface temperature taken as the air temperature,
bounded away from 0 °C at -0.001 °C as in the P3 code), and the density
is [`p3_cober_list_density`](@ref)`(R_i)`. Faster impacts, larger drops
and warmer surfaces give denser rime: the droplets spread before they
freeze.
"""
function p3_rime_density(D_c::Real, V_impact::Real, T::Real,
                         pp::P3ProcessParameters = P3_DEFAULT_PROCESS)
    Tc = min(-0.001, float(T) - pp.thermo.T_freeze)
    Ri = -(0.5e6 * float(D_c)) * abs(float(V_impact)) / Tc
    return p3_cober_list_density(Ri)
end

# -------------------------------------------------------------------------
# Condensation / evaporation of cloud water: saturation adjustment
# -------------------------------------------------------------------------

# A copy of the thermodynamic constants whose liquid/ice ramp is disabled
# (`liquid_fraction ≡ 1`): in P3 cloud water is liquid at every
# temperature — supercooled below 0 °C — and ice exists only as its own
# prognostic category, created and grown by explicit processes. Nothing
# else in `ThermodynamicConstants` depends on T_freeze / T_icenuc.
function _p3_liquid_only(c::ThermodynamicConstants{T}) where {T}
    vals = (f === :T_freeze || f === :T_icenuc ? zero(T) : getfield(c, f)
            for f in fieldnames(ThermodynamicConstants))
    return ThermodynamicConstants{T}(vals...)
end

"""
    p3_condensation(s::P3ParcelState, pp = P3_DEFAULT_PROCESS) -> P3ParcelState

Condensation of vapour to cloud water, or evaporation of cloud water,
by **saturation adjustment** with respect to liquid:
`MoistThermodynamics.saturation_adjustment` is applied to the vapour +
cloud subsystem (`q_tot = q_v + q_c`) at constant liquid-ice potential
temperature and pressure, and returns the temperature and cloud water
that leave the parcel exactly saturated over liquid (or subsaturated
with no cloud). Nothing thermodynamic is re-derived here.

P3 cloud water stays liquid below 0 °C (it can only become ice by
riming or freezing), so the adjustment is called with a copy of the
constants whose liquid-fraction ramp is disabled. Rain and ice do not
take part (their heat capacity is left out of the adjusted `c_pm`, a
≲ 0.1 % effect for 5 g kg⁻¹ of hydrometeors).

`q_v + q_c` is conserved exactly. Energy is conserved in the sense of
the contract's invariant, θ_li, which is *not* exactly isobaric
enthalpy: `c_pm ΔT / (L_v Δq_c)` measures 0.998 at 1000 hPa, 1.012 at
900 hPa and 1.064 at 600 hPa (tests), the deviation coming from the
moisture dependence of the Exner exponent `R_m/c_pm` inside θ_li. It is
a property of `MoistThermodynamics.saturation_adjustment` and is
inherited here rather than patched, so that every scheme shares one
adjustment.

Cloud number: newly formed cloud
(`N_c = 0`) receives the prescribed `N_c_prescribed / ρ_air`; complete
evaporation removes all droplets, partial evaporation keeps `N_c` (the
P3 convention — droplets shrink before they disappear).

The explicit supersaturation-relaxation condensation of the P3 code
(Morrison & Grabowski 2008) is replaced by this adjustment; the two agree
whenever the phase relaxation time (~1-10 s) is short compared with the
time step.
"""
function p3_condensation(s::P3ParcelState, pp::P3ProcessParameters = P3_DEFAULT_PROCESS)
    cl = _p3_liquid_only(pp.thermo)
    qt = s.q_v + s.q_c
    qt > 0 || return s
    θli = liquid_ice_potential_temperature(s.T, s.p, PhasePartition(qt, s.q_c, 0.0), cl)
    Tn, part = saturation_adjustment(θli, s.p, qt, cl)
    qc = part.liq
    Nc = s.N_c
    if qc < pp.q_small
        # evaporate the residue too, so no cloud "dust" is left behind
        cpm = heat_capacity_moist(PhasePartition(qt, qc, 0.0), pp.thermo)
        Tn -= latent_heat_vapor(Tn, pp.thermo) * qc / cpm
        qc, Nc = zero(qc), zero(qc)
    elseif Nc <= 0
        new = _p3_with(s; T = Tn, q_v = qt - qc, q_c = qc)
        Nc = pp.N_c_prescribed / air_density(Tn, s.p, _p3_partition(new), pp.thermo)
    end
    return _p3_with(s; T = Tn, q_v = qt - qc, q_c = qc, N_c = Nc)
end
