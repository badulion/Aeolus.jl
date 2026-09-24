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
3608-3619); Westbrook et al. (2008, JAS 65, 206-219) for the aggregate
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
take part: their heat capacity is left out of the adjusted `c_pm`
(and their mass is counted as dry air), so the condensational `ΔT` is
too large by the fraction of `c_pm` they carry — measured 1.5 % for
5 g kg⁻¹ of rain and 0.5 % for 5 g kg⁻¹ of ice. The contract's
`saturation_adjustment` takes only `q_tot` and diagnoses the whole
condensate, so fixed rain and ice cannot be passed to it.

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

# -------------------------------------------------------------------------
# Ventilation and growth resistances
# -------------------------------------------------------------------------

"""
    p3_ventilation(X, kind::Symbol) -> f_v

Ventilation coefficient `f_v` of diffusional growth as a function of
`X = Sc^{1/3} Re^{1/2}`. For ice (`kind = :ice`), Hall & Pruppacher
(1976):

    f_v = 1 + 0.14 X²        X < 1,       f_v = 0.86 + 0.28 X        X ≥ 1;

for drops (`kind = :rain`), Beard & Pruppacher (1971):

    f_v = 1 + 0.108 X²       X < 1.4,     f_v = 0.78 + 0.308 X       X ≥ 1.4.

Both are ≥ 1 and continuous (to 5e-4 at the rain junction). The P3
reference code applies only the linear large-`X` branches (0.86, 0.28
for ice and 0.78, 0.32 for rain), which give `f_v < 1` for the smallest
particles; the piecewise forms keep the stagnant-air limit `f_v → 1`.
"""
function p3_ventilation(X::Real, kind::Symbol)
    x = float(X)
    if kind === :ice
        return x < 1 ? 1 + 0.14x^2 : 0.86 + 0.28x
    elseif kind === :rain
        return x < 1.4 ? 1 + 0.108x^2 : 0.78 + 0.308x
    end
    throw(ArgumentError("kind must be :ice or :rain"))
end

# Diffusional growth resistance A + B (s m kg⁻¹) such that one particle of
# capacitance C grows as  dm/dt = 4π C f_v S / (A + B),  S = e/e_s - 1.
# Liquid: the Superdroplet resistances F_k + F_d (which carry ρ_w for the
# r dr/dt form), divided by ρ_w. Ice: the same two terms with L_s and e_si.
_p3_resistance_liquid(T, p, pp::P3ProcessParameters) =
    (fk_thermal(T, pp.micro, pp.thermo) + fd_vapor(T, p, pp.micro, pp.thermo)) /
    pp.micro.ρ_water

function _p3_resistance_ice(T, p, pp::P3ProcessParameters)
    c = pp.thermo
    Rv = R_v(c)
    Ls = latent_heat_sublim(T, c)
    Ka = thermal_conductivity(T, pp.micro)
    Dv = vapor_diffusivity(T, p, pp.micro)
    return (Ls / (Rv * T) - 1) * Ls / (Ka * T) +
           Rv * T / (Dv * saturation_vapor_pressure_ice(T, c))
end

# -------------------------------------------------------------------------
# Size-distribution integrals
# -------------------------------------------------------------------------

# A mass-dimension state with no finite thresholds, so that
# `_p3_psd_expectation` (massdim.jl) integrates an arbitrary gamma PSD
# without panel edges from the ice regimes — used for rain.
_p3_no_thresholds(p::P3IceParameters{T}) where {T} =
    P3MassDimension{T}(zero(T), p.ρ_rim_default, zero(T), zero(T),
                       T(Inf), T(Inf), T(Inf), p)

# Rain PSD slope from (q_r, N_r) for spheres of density ρ_w, with N_r
# bounded so that the mean-volume diameter lies in [D_r_min, D_r_max].
function _p3_rain_slope(q_r, N_r, pp::P3ProcessParameters)
    μ = pp.μ_r
    mD(D) = π / 6 * pp.micro.ρ_water * D^3
    Nr = clamp(N_r, q_r / mD(pp.D_r_max), q_r / mD(pp.D_r_min))
    λ = gamma_psd_slope(Nr, 0, q_r / (π / 6 * pp.micro.ρ_water), 3, μ)
    return λ, Nr
end

# Rain fall speed: the Superdroplet drop terminal velocity (Beard 1976 /
# Atlas et al. 1973, at the reference air density) with the density
# correction (ρ_ref/ρ_a)^0.54 of Heymsfield & Bansemer used in P3.
_p3_rain_vt(D, ρa, pp::P3ProcessParameters) =
    terminal_velocity(D / 2, pp.micro) * (pp.micro.ρ_air / ρa)^0.54

# Rain integrals: E_N[D f_v], V_N, V_M.
function _p3_rain_integrals(λ, ρa, η, Sc13, pp::P3ProcessParameters{T}) where {T}
    md0 = _p3_no_thresholds(pp.ice)
    Df, V, VD3, D3 = _p3_psd_expectation(T(pp.μ_r), T(λ), md0) do D
        v = _p3_rain_vt(D, ρa, pp)
        f = p3_ventilation(Sc13 * sqrt(v * D * ρa / η), :rain)
        (D * f, v, v * D^3, D^3)
    end
    return (Df = Df, V_N = V, V_M = VD3 / D3)
end

# Ice integrals over the P3 PSD (number-weighted expectations).
function _p3_ice_integrals(μ, λ, md::P3MassDimension{T}, ρa, η, Sc13,
                           VrN, VrM, pp::P3ProcessParameters) where {T}
    p = md.params
    r = _p3_psd_expectation(T(μ), T(λ), md) do D
        m = p3_particle_mass(D, md)
        A = p3_particle_area(D, md)
        v = _p3_vt(D, m, A, T(ρa), T(η), p)
        reg = p3_regime(D, md)
        C = (reg == 1 || reg == 3 ? one(T) : T(pp.cap_agg)) * D / 2
        f = p3_ventilation(Sc13 * sqrt(v * D * ρa / η), :ice)
        (C * f, C, A * v, A * abs(v - VrM), A * abs(v - VrN), v * m, m)
    end
    Cf, C, AV, AVrM, AVrN, vm, m = r
    return (Cf = Cf, C = C, AV = AV, AVrM = AVrM, AVrN = AVrN, V_M = vm / m)
end

# P3's physical limiter on the ice mean size (μ+1)/λ ∈ [2 μm, 5 mm + 20 mm F_rim²]
# (create_p3_lookupTable_1.f90), as a λ bracket.
_p3_ice_lambda_bounds(μ, F_rim) = ((μ + 1) / (5e-3 + 2e-2 * F_rim^2), (μ + 1) / 2e-6)

# -------------------------------------------------------------------------
# All process rates
# -------------------------------------------------------------------------

"""
    p3_process_rates(s::P3ParcelState, pp = P3_DEFAULT_PROCESS) -> NamedTuple

Instantaneous rates of every P3 process for the state `s`, each a
**non-negative magnitude** of the transfer it names (kg kg⁻¹ s⁻¹ for
mass, kg⁻¹ s⁻¹ for number), except `n_self_r`, a net number sink that is
negative when breakup dominates:

| field | transfer | formulation |
| :-- | :-- | :-- |
| `evap_r`, `n_evap_r` | rain → vapour | ventilated diffusion, `S_w < 0` only |
| `dep` | vapour → ice | ventilated diffusion, `S_i > 0`, `T < T₀` |
| `sub`, `n_sub` | ice → vapour | ventilated diffusion, `S_i < 0`, `T < T₀` |
| `auto`, `n_auto_r`, `n_auto_c` | cloud → rain | KK2000 ([`p3_autoconversion`](@ref)) |
| `accr`, `n_accr_c` | cloud → rain | KK2000 ([`p3_accretion`](@ref)) |
| `n_self_r` | rain number | SB2001 + breakup ([`p3_rain_self_collection`](@ref)) |
| `rime_c`, `n_rime_c`, `ρ_rime_c` | cloud → rime | continuous collection, `T < T₀`, Cober & List density |
| `rime_r`, `n_rime_r` | rain → rime | continuous collection, `T < T₀`, density `ρ_rim_rain` |
| `shed`, `n_shed_c`, `n_shed_r` | cloud → rain | cloud collected by ice at `T ≥ T₀`, shed as 1 mm drops |
| `melt`, `n_melt` | ice → rain | heat balance, `T > T₀` |

**Diffusional growth.** One particle of capacitance `C` and ventilation
coefficient `f_v` ([`p3_ventilation`](@ref)) grows as
`dm/dt = 4π C f_v S / (A + B)` with `S` the super- or subsaturation over
water (rain) or ice, `A` the heat-conduction and `B` the vapour-diffusion
resistance (Pruppacher & Klett 1997, §13.2). `C = D/2` for spheres and
graupel, `0.48 D/2` for unrimed and partially rimed nonspherical ice.
The rate of a population is `N · 4π E_N[C f_v] S/(A+B)`, the expectation
over the PSD taken by quadrature; `Re` uses [`p3_fall_speed`](@ref) for
ice and the Superdroplet drop terminal velocity for rain. Sublimation and
evaporation remove number in proportion to mass.

**Riming.** Continuous collection with the geometric kernel of the ice
particle, `∂q_c/∂t = -E_ci ρ_a q_c N_i E_N[A V]` (cloud droplets'
own fall speed neglected), and for rain the kernel
`A |V_i - V_r|` with the rain mass- (number-) weighted fall speed for
mass (number). The rime density of cloud rime is
[`p3_rime_density`](@ref) at the droplet mean-volume diameter and the
impact speed `|V_M,i - V_c|`. Below 0 °C collected water becomes rime
(`q_rim`, `B_rim += dq/ρ_rime`); at or above 0 °C collected cloud water
is shed as rain (MM15 §2c) and collected rain stays rain but carries
its sensible heat into melting.

**Melting.** For `T > T₀` (Mason 1971; Rutledge & Hobbs 1983; MM15),

    melt = [N_i 4π E_N[C f_v] (K_a (T-T₀) + L_v D_v (ρ_v - ρ_vs(T₀)))
            + c_w (T-T₀) (coll_c + coll_r)] / L_f,

floored at zero (dry air can hold a particle below 0 °C at its surface
by evaporative cooling). Each melted particle becomes one drop, and the
rime melts in proportion.

Ice PSD: gamma with shape `μ_i` and λ from [`p3_slope`](@ref) with the
P3 mean-size limiter (2 μm to 5 mm + 20 mm F_rim²). Rain PSD: gamma with
shape `μ_r`, mean-volume diameter bounded to `[D_r_min, D_r_max]`.
"""
function p3_process_rates(s::P3ParcelState{FT},
                          pp::P3ProcessParameters = P3_DEFAULT_PROCESS) where {FT}
    c = pp.thermo
    T, p = s.T, s.p
    T0 = c.T_freeze
    z = zero(FT)
    ρa = air_density(T, p, _p3_partition(s), c)
    η = p3_air_viscosity(T)
    Dv = vapor_diffusivity(T, p, pp.micro)
    Ka = thermal_conductivity(T, pp.micro)
    Sc13 = cbrt(η / (ρa * Dv))
    qs = saturation_specific_humidity(T, p, c)
    qsi = saturation_specific_humidity(T, p, c; over = :ice)
    Sw = s.q_v / qs - 1
    Si = s.q_v / qsi - 1

    have_c = s.q_c >= pp.q_small
    have_r = s.q_r >= pp.q_small && s.N_r > 0
    have_i = s.q_i >= pp.q_small && s.N_i > 0

    # --- warm rain -------------------------------------------------------
    auto, n_auto_r, n_auto_c = p3_autoconversion(s.q_c, s.N_c, ρa, pp)
    accr, n_accr_c = have_r ? p3_accretion(s.q_c, s.N_c, s.q_r, pp) : (z, z)

    evap_r = n_evap_r = n_self_r = z
    VrN = VrM = z
    λr = Nr = z
    if have_r
        λr, Nr = _p3_rain_slope(s.q_r, s.N_r, pp)
        ri = _p3_rain_integrals(λr, ρa, η, Sc13, pp)
        VrN, VrM = ri.V_N, ri.V_M
        if Sw < 0
            evap_r = -Nr * 2π * ri.Df * Sw / _p3_resistance_liquid(T, p, pp)
            n_evap_r = evap_r * s.N_r / s.q_r
        end
        n_self_r = p3_rain_self_collection(s.q_r, s.N_r, λr, ρa, pp)
    end

    # --- ice ---------------------------------------------------------------
    dep = sub = n_sub = z
    rime_c = n_rime_c = rime_r = n_rime_r = z
    ρ_rime_c = FT(pp.ice.ρ_rim_default)
    shed = n_shed_c = n_shed_r = melt = n_melt = z
    if have_i
        md = p3_mass_dimension(s.q_i, s.q_rim, s.B_rim, pp.ice)
        λmin, λmax = _p3_ice_lambda_bounds(pp.μ_i, md.F_rim)
        λi = p3_slope(s.q_i, s.N_i, md; μ = pp.μ_i, λ_min = λmin, λ_max = λmax)
        ii = _p3_ice_integrals(pp.μ_i, λi, md, ρa, η, Sc13, VrN, VrM, pp)

        coll_c = have_c ? pp.E_ci * ρa * s.q_c * s.N_i * ii.AV : z
        coll_r = have_r ? pp.E_ri * ρa * s.q_r * s.N_i * ii.AVrM : z
        ncoll_r = have_r ? pp.E_ri * ρa * s.N_r * s.N_i * ii.AVrN : z

        if T < T0
            g = s.N_i * 4π * ii.Cf * Si / _p3_resistance_ice(T, p, pp)
            dep, sub = max(g, z), max(-g, z)
            n_sub = sub * s.N_i / s.q_i
            rime_c, rime_r, n_rime_r = coll_c, coll_r, ncoll_r
            if have_c && s.N_c > 0
                n_rime_c = rime_c * s.N_c / s.q_c
                Dc = cbrt(6 * s.q_c / (s.N_c * π * pp.micro.ρ_water))
                Vc = terminal_velocity(Dc / 2, pp.micro)
                ρ_rime_c = FT(p3_rime_density(Dc, ii.V_M - Vc, T, pp))
            end
        else
            shed = coll_c
            n_shed_c = have_c && s.N_c > 0 ? shed * s.N_c / s.q_c : z
            n_shed_r = shed / (π / 6 * pp.micro.ρ_water * pp.D_shed^3)
            if T > T0
                Lf0 = latent_heat_fusion(T0, c)
                Lv0 = latent_heat_vapor(T0, c)
                ρvs0 = saturation_vapor_pressure(T0, c) / (R_v(c) * T0)
                heat = Ka * (T - T0) + Lv0 * Dv * (ρa * s.q_v - ρvs0)
                melt = (s.N_i * 4π * ii.Cf * heat +
                        c.cp_liquid * (T - T0) * (coll_c + coll_r)) / Lf0
                melt = max(melt, z)
                n_melt = melt * s.N_i / s.q_i
            end
        end
    end

    return (evap_r = evap_r, n_evap_r = n_evap_r,
            dep = dep, sub = sub, n_sub = n_sub,
            auto = auto, n_auto_r = n_auto_r, n_auto_c = n_auto_c,
            accr = accr, n_accr_c = n_accr_c,
            n_self_r = n_self_r,
            rime_c = rime_c, n_rime_c = n_rime_c, ρ_rime_c = ρ_rime_c,
            rime_r = rime_r, n_rime_r = n_rime_r,
            shed = shed, n_shed_c = n_shed_c, n_shed_r = n_shed_r,
            melt = melt, n_melt = n_melt)
end

"""
    p3_tendencies(s::P3ParcelState, r = p3_process_rates(s, pp), pp = P3_DEFAULT_PROCESS)
        -> P3ParcelState

Time derivatives of every prognostic variable implied by the process
rates `r` ([`p3_process_rates`](@ref)), returned as a `P3ParcelState`
whose fields are `d/dt` of the corresponding fields (`p` is held fixed,
`dp/dt = 0`). The mapping is linear in the rates:

    dq_v  =  evap_r - dep + sub
    dq_c  = -auto - accr - rime_c - shed
    dq_r  = -evap_r + auto + accr - rime_r + shed + melt
    dq_i  =  dep - sub + rime_c + rime_r - melt
    dq_rim = rime_c + rime_r - (sub + melt) q_rim/q_i
    dB_rim = rime_c/ρ_rime_c + rime_r/ρ_rim_rain - (sub + melt) B_rim/q_i
    c_pm dT = -L_v(T) dq_v + L_f(T) dq_i

so `dq_v + dq_c + dq_r + dq_i = 0` identically, sublimation and melting
leave the rime density `q_rim/B_rim` unchanged, and the heating is
exactly the latent heat of the net phase change (`L_s = L_v + L_f` for
deposition, `L_f` for riming, `-L_f` for melting). Latent heats and
`c_pm` are those of `MoistThermodynamics` at the current state.
"""
function p3_tendencies(s::P3ParcelState{FT}, r::NamedTuple,
                       pp::P3ProcessParameters = P3_DEFAULT_PROCESS) where {FT}
    c = pp.thermo
    dqv = r.evap_r - r.dep + r.sub
    dqc = -r.auto - r.accr - r.rime_c - r.shed
    dqr = -r.evap_r + r.auto + r.accr - r.rime_r + r.shed + r.melt
    dqi = r.dep - r.sub + r.rime_c + r.rime_r - r.melt
    dNc = -r.n_auto_c - r.n_accr_c - r.n_rime_c - r.n_shed_c
    dNr = -r.n_evap_r + r.n_auto_r - r.n_self_r - r.n_rime_r + r.n_shed_r + r.n_melt
    dNi = -r.n_sub - r.n_melt
    loss = r.sub + r.melt
    fq = s.q_i > 0 ? s.q_rim / s.q_i : zero(FT)
    fB = s.q_i > 0 ? s.B_rim / s.q_i : zero(FT)
    dqrim = r.rime_c + r.rime_r - loss * fq
    dBrim = r.rime_c / r.ρ_rime_c + r.rime_r / pp.ρ_rim_rain - loss * fB
    cpm = heat_capacity_moist(_p3_partition(s), c)
    dT = (-latent_heat_vapor(s.T, c) * dqv + latent_heat_fusion(s.T, c) * dqi) / cpm
    return P3ParcelState{FT}(dT, zero(FT), dqv, dqc, dNc, dqr, dNr, dqi, dNi,
                             dqrim, dBrim)
end

p3_tendencies(s::P3ParcelState, pp::P3ProcessParameters = P3_DEFAULT_PROCESS) =
    p3_tendencies(s, p3_process_rates(s, pp), pp)

# -------------------------------------------------------------------------
# Full source-term step
# -------------------------------------------------------------------------

_p3_frac(avail, need) =
    (need > 0 && need > avail) ? max(avail, zero(avail)) / need : one(avail)

"""
    p3_limit_rates(s::P3ParcelState, r, dt, pp = P3_DEFAULT_PROCESS) -> NamedTuple

Scale the process rates `r` so that one forward-Euler step of length
`dt` can neither drive a species negative nor overshoot saturation, in
the manner of the P3 code's conservation checks:

  * the sinks of each of `q_c`, `q_r`, `q_i` are scaled by a common
    factor so their total over `dt` does not exceed what is there;
  * deposition may not remove more vapour than brings the parcel to ice
    saturation, and sublimation / rain evaporation no more than brings it
    back to ice / liquid saturation, each reduced by the psychrometric
    factor `Γ = 1 + (L/c_pm) dq_s/dT` that accounts for its own latent
    heating shifting `q_s` during the step; the saturation values are
    taken at the temperature after the step's riming and melting;
    sublimation may fill only the part of the ice-saturation deficit
    that the (limited) rain evaporation leaves, so the two vapour
    sources together cannot push the parcel past ice saturation;
  * number rates are scaled with the mass rate they belong to, and the
    rain self-collection and ice-collection number sinks so `N_r ≥ 0`.

Each process drains exactly one species, so scaling it by that species'
factor keeps every species non-negative while the linear tendency map
keeps water conserved exactly.
"""
function p3_limit_rates(s::P3ParcelState{FT}, r::NamedTuple, dt::Real,
                        pp::P3ProcessParameters = P3_DEFAULT_PROCESS) where {FT}
    c = pp.thermo
    dt = FT(dt)
    T, p = s.T, s.p
    cpm = heat_capacity_moist(_p3_partition(s), c)
    Rv = R_v(c)
    Lv, Ls = latent_heat_vapor(T, c), latent_heat_sublim(T, c)

    fc = _p3_frac(s.q_c, dt * (r.auto + r.accr + r.rime_c + r.shed))
    fr = _p3_frac(s.q_r, dt * (r.evap_r + r.rime_r))
    fi = _p3_frac(s.q_i, dt * (r.sub + r.melt))
    # The saturation targets are taken at the temperature the step's
    # freezing (riming) and melting will produce, so their latent heat
    # cannot push the vapour processes past saturation.
    Tf = T + latent_heat_fusion(T, c) * dt *
             (fc * r.rime_c + fr * r.rime_r - fi * r.melt) / cpm
    qs = saturation_specific_humidity(Tf, p, c)
    qsi = saturation_specific_humidity(Tf, p, c; over = :ice)
    Γw = 1 + Lv / cpm * qs * Lv / (Rv * Tf^2)
    Γi = 1 + Ls / cpm * qsi * Ls / (Rv * Tf^2)
    fev = min(fr, _p3_frac((qs - s.q_v) / Γw, dt * r.evap_r))
    # Sublimation and rain evaporation both feed vapour. Once q_v reaches
    # ice saturation the ice stops sublimating (and starts to grow) while
    # the rain keeps evaporating toward liquid saturation, so sublimation
    # may only fill the part of the ice deficit that evaporation leaves;
    # Γwi accounts for the evaporative cooling lowering q_si.
    Γwi = 1 + Lv / cpm * qsi * Ls / (Rv * Tf^2)
    Eev = dt * fev * r.evap_r
    fsub = min(fi, _p3_frac((qsi - s.q_v - Γwi * Eev) / Γi, dt * r.sub))
    fdep = min(_p3_frac((s.q_v - qsi) / Γi, dt * r.dep), _p3_frac(s.q_v, dt * r.dep))

    evap_r, n_evap_r = fev * r.evap_r, fev * r.n_evap_r
    # number is cut with the mass it belongs to, then by what N_r can supply
    rime_r, n_rime_r = fr * r.rime_r, fr * r.n_rime_r
    # rain number: sinks other than evaporation limited by what is left
    nsink = dt * (max(r.n_self_r, zero(FT)) + n_rime_r)
    fN = _p3_frac(s.N_r - dt * n_evap_r, nsink)
    n_self_r = r.n_self_r > 0 ? fN * r.n_self_r : r.n_self_r
    n_rime_r = fN * n_rime_r

    return (evap_r = evap_r, n_evap_r = n_evap_r,
            dep = fdep * r.dep, sub = fsub * r.sub, n_sub = fsub * r.n_sub,
            auto = fc * r.auto, n_auto_r = fc * r.n_auto_r, n_auto_c = fc * r.n_auto_c,
            accr = fc * r.accr, n_accr_c = fc * r.n_accr_c,
            n_self_r = n_self_r,
            rime_c = fc * r.rime_c, n_rime_c = fc * r.n_rime_c, ρ_rime_c = r.ρ_rime_c,
            rime_r = rime_r, n_rime_r = n_rime_r,
            shed = fc * r.shed, n_shed_c = fc * r.n_shed_c, n_shed_r = fc * r.n_shed_r,
            melt = fi * r.melt, n_melt = fi * r.n_melt)
end

# Remove species below q_small (evaporating / sublimating the residue, so
# water and energy are conserved), clamp roundoff negatives the same way,
# and keep number, rime and PSD consistent with the mass.
function _p3_cleanup(s::P3ParcelState{FT}, pp::P3ProcessParameters) where {FT}
    c = pp.thermo
    T, qv = s.T, s.q_v
    qc, Nc, qr, Nr = s.q_c, s.N_c, s.q_r, s.N_r
    qi, Ni, qrim, Brim = s.q_i, s.N_i, s.q_rim, s.B_rim
    cpm = heat_capacity_moist(_p3_partition(s), c)
    z = zero(FT)
    if qc < pp.q_small
        qv += qc; T -= latent_heat_vapor(T, c) * qc / cpm
        qc, Nc = z, z
    end
    if qr < pp.q_small
        qv += qr; T -= latent_heat_vapor(T, c) * qr / cpm
        qr, Nr = z, z
    end
    if qi < pp.q_small
        qv += qi; T -= latent_heat_sublim(T, c) * qi / cpm
        qi, Ni, qrim, Brim = z, z, z, z
    end
    Nc = qc > 0 ? max(Nc, z) : z
    if qc > 0 && Nc == 0
        Nc = pp.N_c_prescribed / air_density(T, s.p, _p3_partition(s), c)
    end
    if qr > 0
        mD(D) = π / 6 * pp.micro.ρ_water * D^3
        Nr = clamp(Nr, qr / mD(pp.D_r_max), qr / mD(pp.D_r_min))
    end
    if qi > 0
        qrim = clamp(qrim, z, qi)
        Brim = max(Brim, z)
        _, _, Brim = p3_rime_state(qi, qrim, Brim, pp.ice)
        md = p3_mass_dimension(qi, qrim, Brim, pp.ice)
        λmin, λmax = _p3_ice_lambda_bounds(pp.μ_i, md.F_rim)
        m_lo = p3_mean_mass(pp.μ_i, λmax, md)
        m_hi = p3_mean_mass(pp.μ_i, λmin, md)
        # P3 adjusts N_i when the mean size leaves its limiter (MM15 §2b)
        Ni = qi / clamp(Ni > 0 ? qi / Ni : m_hi, m_lo, m_hi)
    end
    return P3ParcelState{FT}(T, s.p, qv, qc, Nc, qr, Nr, qi, Ni, qrim, Brim)
end

"""
    p3_step(s::P3ParcelState, dt, pp = P3_DEFAULT_PROCESS; adjust = true) -> P3ParcelState

Advance the P3 microphysical source terms (no transport, no
sedimentation) of one cell by `dt` seconds:

 1. [`p3_process_rates`](@ref) at the start-of-step state;
 2. [`p3_limit_rates`](@ref) for positivity and no saturation overshoot;
 3. a forward-Euler update with [`p3_tendencies`](@ref);
 4. clean-up: species below `q_small` are evaporated/sublimated (with
    their latent heat), rime is clamped to `0 ≤ q_rim ≤ q_i` and
    `ρ_rim ∈ [50, 900]` via [`p3_rime_state`](@ref), `N_i` is adjusted
    when the ice mean size leaves P3's limiter and `N_r` when the rain
    mean-volume diameter leaves `[D_r_min, D_r_max]`;
 5. if `adjust`, cloud condensation/evaporation by
    [`p3_condensation`](@ref) (`MoistThermodynamics.saturation_adjustment`).

Total water `q_v + q_c + q_r + q_i` is conserved to rounding in every
stage. Steps 1-4 conserve the enthalpy form `c_pm ΔT = -L_v Δq_v +
L_f Δq_i` exactly for a step without clean-up; step 5 conserves the
contract's θ_li (see [`p3_condensation`](@ref)).

Sub-grid fractions (Phase 44, off by default): passing `cloud_fraction`
and/or `precip_fraction` (a missing one is taken as 1) evaluates step 1
with [`p3_fractional_process_rates`](@ref) — accretion, riming,
autoconversion with in-cloud / in-precipitation mixing ratios weighted
by area, rain evaporation only in the clear part. With both `nothing`
(the default) the step is the grid-mean one, bit for bit. Step 5 is
still the grid-mean all-or-nothing adjustment.
"""
function p3_step(s::P3ParcelState{FT}, dt::Real,
                 pp::P3ProcessParameters = P3_DEFAULT_PROCESS;
                 adjust::Bool = true, cloud_fraction = nothing,
                 precip_fraction = nothing, f_min::Real = 0.01) where {FT}
    r0 = cloud_fraction === nothing && precip_fraction === nothing ?
         p3_process_rates(s, pp) :
         p3_fractional_process_rates(s, something(cloud_fraction, 1),
                                     something(precip_fraction, 1), pp; f_min = f_min)
    r = p3_limit_rates(s, r0, dt, pp)
    s1 = _p3_cleanup(_p3_axpy(s, dt, p3_tendencies(s, r, pp)), pp)
    return adjust ? p3_condensation(s1, pp) : s1
end
