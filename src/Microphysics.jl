module Microphysics

# Phase 12 — Kessler warm-rain microphysics.
#
# The Kessler (1969) parameterization is the simplest cloud-microphysics
# scheme: three water mixing-ratio categories (qv, ql, qr) exchange mass
# via
#
#   * autoconversion  : ql above a threshold converts to qr
#   * accretion       : qr grows by collecting ql
#   * rain fallout    : qr sediments downward. By default (Phase 44) at
#                       the mass-weighted fall speed of a Marshall-Palmer
#                       drop spectrum, V_M(q_r), delegated to the P3 rain
#                       sedimentation (`P3Microphysics.p3_sediment!`); a
#                       single constant speed is kept as an option.
#
# Evaporation of falling rain and a full saturation adjustment (which
# couples to θ and pressure) are deferred to a later refinement.
#
# The kernel here is fully local — it operates cell-by-cell on the
# prognostic fields.

using ..Grids: RegularGrid, interior
using ..Fields: Field, interior_view
using ..Superdroplet: DEFAULT_MICRO
using ..P3Microphysics: p3_sediment!, p3_rain_sedimentation_velocities
using ..Backends: AbstractBackend, backend, launch!, foreach_column!

export apply_kessler_step!, KESSLER_N0, kessler_rain_number, kessler_fall_speed

"""
    KESSLER_N0 = 8.0e6

Intercept `N₀` (m⁻⁴) of the Marshall & Palmer (1948) exponential drop
size distribution `N(D) = N₀ e^{-λD}` that Kessler (1969) assumes for
rain. It closes the single-moment scheme: given `q_r` the whole drop
spectrum, hence its fall-speed spectrum, is fixed.
"""
const KESSLER_N0 = 8.0e6

"""
    kessler_rain_number(q_r, ρ_air; N0 = KESSLER_N0) -> N_r  (kg⁻¹)

Rain drop number per kg of air implied by the Marshall-Palmer spectrum
`N(D) = N₀ e^{-λD}` holding rain mass `q_r` (kg kg⁻¹) in air of density
`ρ_air` (kg m⁻³):

    ρ_air q_r = (π/6) ρ_w N₀ Γ(4) / λ⁴  ⇒  λ = (π ρ_w N₀ / (ρ_air q_r))^{1/4},
    N_r = N₀ / (λ ρ_air).

`ρ_w` is `Superdroplet.DEFAULT_MICRO.ρ_water`. Zero for `q_r ≤ 0`. This
turns single-moment Kessler rain into the `(q_r, N_r)` pair the P3 rain
functions take, so the fall-speed spectrum is P3's, not a new law.

Reference: Marshall & Palmer (1948, J. Meteor. 5, 165-166); Kessler
(1969, Meteor. Monogr. 10).
"""
function kessler_rain_number(q_r::Real, ρ_air::Real; N0::Real = KESSLER_N0)
    q = float(q_r)
    q > 0 || return zero(q)
    λ = (π * DEFAULT_MICRO.ρ_water * N0 / (ρ_air * q))^(1 / 4)
    return N0 / (λ * ρ_air)
end

"""
    kessler_fall_speed(q_r, ρ_air = 1.225; N0 = KESSLER_N0) -> V_M  (m s⁻¹)

Mass-weighted terminal fall speed of Kessler rain: the Marshall-Palmer
spectrum of [`kessler_rain_number`](@ref) integrated against the drop
terminal velocity by `P3Microphysics.p3_rain_sedimentation_velocities`
(Beard 1976 / Atlas et al. 1973 drop speeds from
`Superdroplet.terminal_velocity`, Foote & du Toit density correction).
It rises with `q_r` because heavier rain has larger drops: 3.77 m s⁻¹ at
0.1 g m⁻³, 5.66 m s⁻¹ at 1 g m⁻³ and 7.48 m s⁻¹ at 10 g m⁻³ of rain at
sea level, within 10 % of (and at 1-3 g m⁻³ within 0.2 % of) the
Klemp & Wilhelmson (1978) fit `V = 36.34 (ρ q_r [g cm⁻³])^0.1346`
derived from the same Marshall-Palmer spectrum (see the tests).
"""
function kessler_fall_speed(q_r::Real, ρ_air::Real = DEFAULT_MICRO.ρ_air;
                            N0::Real = KESSLER_N0)
    q, ρ = promote(float(q_r), float(ρ_air))
    _, VM = p3_rain_sedimentation_velocities([q], [kessler_rain_number(q, ρ; N0 = N0)], [ρ])
    return VM[1]
end

"""
    apply_kessler_step!(qv, ql, qr, dt; k_auto=1e-3, ql_c=1e-3, k_acc=2.2,
                        Vt=:spectrum, ρ_air=1.225, N0=KESSLER_N0, mesh=qv.mesh)

Advance one microphysics substep in-place. Handles autoconversion,
accretion, and 1st-order upwind sedimentation of `qr`.

  * `k_auto` [1/s]  — autoconversion rate.
  * `ql_c` [kg/kg]  — cloud-water threshold above which autoconversion
    activates.
  * `k_acc` [1/s]   — accretion rate.
  * `Vt`            — rain fall speed. `:spectrum` (default, Phase 44):
    each cell falls at the mass-weighted speed of its own Marshall-Palmer
    drop spectrum ([`kessler_fall_speed`](@ref)), recomputed every
    sub-step, with the conservative, Courant-sub-stepped column scheme
    `P3Microphysics.p3_sediment!`; nothing leaves except through the
    bottom face. A number (m/s, ≥ 0) instead selects the legacy single
    fall speed, bit-identical to the pre-Phase-44 kernel (which caps the
    Courant number at 0.5 and so slows rain whose `Vt Δt/Δz` exceeds it).
  * `ρ_air` [kg/m³] — air density, a scalar or one value per model level
    (bottom first); used only by `Vt = :spectrum`.
  * `N0` [m⁻⁴]      — Marshall-Palmer intercept, see [`KESSLER_N0`](@ref).

`qv` is untouched here — evaporation of falling rain is a follow-up.
"""
function apply_kessler_step!(qv::Field, ql::Field, qr::Field, dt::Real;
                             k_auto::Real=1e-3, ql_c::Real=1e-3,
                             k_acc::Real=2.2, Vt::Union{Real,Symbol}=:spectrum,
                             ρ_air = DEFAULT_MICRO.ρ_air, N0::Real = KESSLER_N0,
                             mesh::RegularGrid = qv.mesh,
                             backend::AbstractBackend = backend(ql.data))
    Vt isa Symbol && Vt !== :spectrum &&
        throw(ArgumentError("Vt must be a fall speed (m/s) or :spectrum"))
    dz  = mesh.dz
    Δt  = dt
    ii, jj, kk = interior(mesh)

    qld = ql.data; qrd = qr.data

    # 1. Autoconversion + accretion (source/sink between ql and qr): per
    #    cell, one work item each (KA-m, #301; pointwise, so the same bits
    #    at every thread count).
    launch!(backend, _kessler_convert!, (ii, jj, kk), qld, qrd, float(k_auto), float(ql_c),
            float(k_acc), float(Δt))

    # 2a. Rain fallout with the size-dependent fall-speed spectrum.
    if Vt === :spectrum
        _kessler_sediment_spectrum!(qrd, ii, jj, kk, dz, Δt, ρ_air, N0; backend = backend)
        return (qv, ql, qr)
    end

    # 2b. Legacy single fall speed — 1st-order upwind (rain falls downward).
    # ∂qr/∂t += ∂(Vt qr)/∂z (positive Vt means fallout, so qr gained
    # from above by 1 cell and lost downward). A column per work item,
    # its levels in ascending k as in the old k-outer loop (each level
    # reads the not-yet-updated level above it).
    if Vt > 0
        α = Vt * Δt / dz
        α = min(α, 0.5)   # keep CFL stable
        foreach_column!(backend, _kessler_upwind_column!, (ii, jj), qrd, kk, float(α);
                        weight = length(kk))
    end

    return (qv, ql, qr)
end

@inline function _kessler_convert!(i, j, k, qld, qrd, k_auto, ql_c, k_acc, Δt)
    @inbounds begin
        qle = qld[i,j,k]
        qre = qrd[i,j,k]
        auto = k_auto * max(qle - ql_c, 0.0)
        accr = k_acc * qle * (max(qre, 0.0) ^ (7/8))
        dqr = (auto + accr) * Δt
        # cap conversion to available ql
        dqr = min(dqr, max(qle, 0.0))
        qld[i,j,k] -= dqr
        qrd[i,j,k] += dqr
    end
    return nothing
end

@inline function _kessler_upwind_column!(i, j, qrd, kk, α)
    @inbounds for k in kk
        in_from_above = k + 1 <= last(kk) ? qrd[i,j,k+1] : 0.0
        out_to_below  = qrd[i,j,k]
        qrd[i,j,k] += α * (in_from_above - out_to_below)
    end
    return nothing
end

# Column-by-column sedimentation of Kessler rain at the Marshall-Palmer
# mass-weighted fall speed, delegated to the P3 flux-form, sub-stepped
# column scheme (which is exactly conservative and positive). One work
# item per column (KA-m, #301): a raining column owns its scratch (`col`,
# `Nr`, allocated by the column) and its fall-speed functor, so no two
# columns share a buffer; the arithmetic is the old loop's.
function _kessler_sediment_spectrum!(qrd, ii, jj, kk, dz, Δt, ρ_air, N0;
                                     backend::AbstractBackend = backend(qrd))
    nz = length(kk)
    ρ = ρ_air isa AbstractVector ? collect(float.(ρ_air)) : fill(float(ρ_air), nz)
    length(ρ) == nz || throw(DimensionMismatch("ρ_air must be a scalar or have one value per level"))
    foreach_column!(backend, _kessler_sediment_column!, (ii, jj), qrd, kk, ρ, dz, Δt, float(N0);
                    weight = nz)
    return nothing
end

# The fall speeds of one column's rain, recomputed at every sub-step from
# the column's own number scratch `Nr` (a functor, not a closure over a
# shared buffer).
struct _KesslerRainSpeed{V<:AbstractVector,R<:AbstractVector}
    Nr::V
    ρ::R
    N0::Float64
end
function (v::_KesslerRainSpeed)(f)
    v.Nr .= kessler_rain_number.(f[1], v.ρ; N0 = v.N0)
    return (p3_rain_sedimentation_velocities(f[1], v.Nr, v.ρ)[2],)
end

function _kessler_sediment_column!(i, j, qrd, kk, ρ, dz, Δt, N0)
    rains = false
    @inbounds for k in kk
        qrd[i, j, k] > 0 && (rains = true; break)
    end
    rains || return nothing
    col = zeros(eltype(qrd), length(kk))
    @inbounds for (n, k) in enumerate(kk); col[n] = qrd[i, j, k]; end
    p3_sediment!((col,), _KesslerRainSpeed(similar(col), ρ, N0), ρ, dz, Δt)
    @inbounds for (n, k) in enumerate(kk); qrd[i, j, k] = col[n]; end
    return nothing
end

end # module
