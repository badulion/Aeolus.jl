# Phase 43, issue #7 — coupling of the nucleation source terms into the
# single P3 ice category.
#
# freezing.jl returns increments (IceNucleationTendency) and never
# touches a hydrometeor; this file is the host side of that contract for
# P3: it converts the per-m³ increments of one call into changes of a
# `P3ParcelState` (per kg of air), and books INP back to the available
# population when the P3 ice that carries them sublimates or melts.
#
# Rime bookkeeping follows the P3 reference code (MM15; `qcheti`,
# `qinuc` in microphy_p3.f90): a frozen droplet is a small graupel-like
# particle and enters the rime mass and volume at the maximum rime
# density; a crystal nucleated from vapour is unrimed ice.

"""
    p3_ice_nucleation(s::P3ParcelState, inp::INPPopulation, Δt;
                      pp = P3_DEFAULT_PROCESS,
                      schemes = default_freezing_schemes(inp),
                      immersed_fraction = nothing)
        -> (P3ParcelState, IceNucleationTendency)

Heterogeneous (and, below `T_icenuc`, homogeneous) freezing of one P3
cell over `Δt`: calls [`ice_nucleation!`](@ref) with the cell's
temperature, ice saturation ratio (`MoistThermodynamics.relative_humidity_ice`),
pressure and cloud droplets (`N_liq = N_c ρ`, `M_liq = q_c ρ`, with
`ρ` = `MoistThermodynamics.air_density` of the cell), consuming INP in
`inp`, and applies the returned increments to the P3 variables:

| P3 variable | change |
| :-- | :-- |
| `q_c`, `N_c` | `- M_liq/ρ`, `- N_liq/ρ` (droplets frozen) |
| `q_v`        | `- M_vap/ρ` (vapour deposited on new crystals) |
| `q_i`        | `+ M_ice/ρ` (= `(M_liq + M_vap)/ρ` exactly) |
| `N_i`        | `+ (N_ice_imm + N_ice_dep + N_ice_hom)/ρ` |
| `q_rim`, `B_rim` | `+ M_liq/ρ`, `+ M_liq/(ρ ρ_rim_max)` (frozen drops are rime at 900 kg m⁻³) |
| `T`          | `+ heat/(ρ c_pm)`, `c_pm` = `heat_capacity_moist` of the cell |

so total water `q_v + q_c + q_r + q_i` is unchanged to rounding, every
crystal formed is one P3 ice particle, and the heating is the
`c_pm dT = -L_v dq_v + L_f dq_i` of [`p3_tendencies`](@ref). Crystals
nucleated from vapour have mass `NUCLEATED_ICE_MASS`, which *is*
`P3_NUCLEATED_ICE_MASS`.

Only cloud droplets freeze: P3 rain has no immersion-freezing process
here (the Bigg rain freezing of the P3 code is not implemented), and
rain is untouched even at `T ≤ T_icenuc`.

Note that the next [`p3_step`](@ref) applies P3's mean-size limiter
(mean diameter ≥ 2 μm): a population made only of deposition-nucleated
1-μm crystals is below it and has its number reduced there, as in the
P3 code — this function itself conserves number exactly.
"""
function p3_ice_nucleation(s::P3ParcelState{FT}, inp::INPPopulation, Δt::Real;
                           pp::P3ProcessParameters = P3_DEFAULT_PROCESS,
                           schemes::AbstractVector = default_freezing_schemes(inp),
                           immersed_fraction = nothing) where {FT}
    c = pp.thermo
    part = PhasePartition(p3_total_water(s), s.q_c + s.q_r, s.q_i)
    ρ = air_density(s.T, s.p, part, c)
    S_i = relative_humidity_ice(s.T, s.p, s.q_v, c)
    have_c = s.q_c >= pp.q_small && s.N_c > 0
    N_liq = have_c ? s.N_c * ρ : 0.0
    M_liq = have_c ? s.q_c * ρ : 0.0
    kw = immersed_fraction === nothing ? (;) : (; immersed_fraction)
    t = ice_nucleation!(inp, s.T, S_i, Δt; p = s.p, N_liq, M_liq, schemes,
                        m_ice0 = NUCLEATED_ICE_MASS, c, kw...)
    ice_number_formed(t) > 0 || t.M_ice > 0 || return (s, t)

    all_frozen = have_c && t.N_liq >= N_liq
    # (q_c ρ)/ρ can exceed q_c by an ulp: clamp, and book dqc from the
    # clamped value so water stays exact
    qc = all_frozen ? zero(FT) : max(s.q_c - FT(t.M_liq / ρ), zero(FT))
    Nc = all_frozen ? zero(FT) : s.N_c - FT(t.N_liq / ρ)
    dqc = s.q_c - qc                     # exactly what leaves the cloud
    dqv = FT(t.M_vap / ρ)
    cpm = heat_capacity_moist(part, c)
    snew = P3ParcelState{FT}(s.T + FT(t.heat / ρ) / cpm, s.p,
                             s.q_v - dqv, qc, max(Nc, zero(FT)),
                             s.q_r, s.N_r,
                             s.q_i + (dqc + dqv), s.N_i + FT(ice_number_formed(t) / ρ),
                             s.q_rim + dqc, s.B_rim + dqc / FT(pp.ice.ρ_rim_max))
    return (snew, t)
end

"""
    p3_inp_release!(inp::INPPopulation, s::P3ParcelState, r, Δt) -> ΔN

Return to the available population the consumed INP of the P3 ice that
sublimated or melted during a step: the limited rates `r` (from
[`p3_step_with_rates`](@ref)) remove `Δt (n_sub + n_melt)` of the `N_i`
crystals of the start-of-step state `s`, and the same fraction
`f = min(1, Δt (n_sub + n_melt) / N_i)` of each mode's consumed INP is
released ([`inp_release!`](@ref)) — each crystal carries its INP, and a
crystal that has gone leaves its INP behind (Solomon et al. 2015 call
this INP recycling). Returns the number released (m⁻³).
"""
function p3_inp_release!(inp::INPPopulation, s::P3ParcelState, r::NamedTuple, Δt::Real)
    s.N_i > 0 || return 0.0
    f = clamp(Δt * (r.n_sub + r.n_melt) / s.N_i, 0.0, 1.0)
    f > 0 || return 0.0
    return inp_release!(inp, f)
end
