# Batch D — moisture / cloud microphysics add-ons for the Aeolus urban stack.
#
# The core `apply_kessler_step!` in `src/Microphysics.jl` only tracks
# (qv, ql, qr) with a Kessler-style rate law and does not attempt to
# saturate the vapour field. This file layers on four physically
# richer pieces without touching the framework core:
#
#   M1  saturation_adjustment!    — coupled qv ↔ ql with Bolton (1980)
#                                    latent-heat release into θ.
#   M2  TwoMomentState + apply_two_moment_micro!
#                                 — Morrison–Milbrandt-lite two-moment
#                                    warm-rain scaffold (Nc, Nr prognostic).
#   M3  apply_ice_processes!      — homogeneous freezing / melting with
#                                    L_f accounting.
#   M4  apply_fog_dew!            — dew onto cold surfaces + fog in the
#                                    air, delegating to Aeolus.RadiationFog
#                                    (Phase 44); latent heat of dew goes to
#                                    the surface budget.
#
# All functions accept either raw 3-D `AbstractArray`s (interior extent)
# or Aeolus `Field{CellCenter}` objects via multiple dispatch. The Field
# methods just strip the halo through `interior_view` and forward.
#
# Classical references:
#   Bolton, 1980, MWR 108, 1046-1053         — saturation adjustment.
#   Kessler, 1969, Met. Monogr. 10           — autoconv / accretion base.
#   Morrison & Milbrandt, 2015, JAS 72       — two-moment framework.

using Aeolus
using Aeolus.Fields: Field, interior_view
using Aeolus.Grids: CellCenter, RegularGrid, AbstractMesh, interior

export saturation_adjustment!,
       TwoMomentState,
       apply_two_moment_micro!,
       apply_ice_processes!,
       apply_fog_dew!

# ---------------------------------------------------------------------------
# Constants — locally scoped, re-declared so the file is loadable on its
# own even when `urban_physics.jl` (which owns `_L_v`, `_R_d`, `_R_v`)
# has not been included yet.
# ---------------------------------------------------------------------------

const _CP_D   = 1005.0        # J kg⁻¹ K⁻¹ (dry-air heat capacity)
const _RD_D   = 287.05        # J kg⁻¹ K⁻¹
const _RV_D   = 461.5         # J kg⁻¹ K⁻¹
const _LV_D   = 2.45e6        # J kg⁻¹ latent heat of vaporization
const _LF_D   = 3.34e5        # J kg⁻¹ latent heat of fusion
const _LS_D   = _LV_D + _LF_D # J kg⁻¹ latent heat of sublimation
const _PREF_D = 1.0e5         # Pa reference pressure for Exner
const _EPS_D  = _RD_D / _RV_D
const _T_ICE  = 273.15        # K
const _T_HOM  = 235.0         # K — homogeneous freezing threshold
const _N_C_INIT   = 100.0e6   # m⁻³  — 100 cm⁻³ (typical maritime CCN)
const _RHO_LIQ    = 1000.0    # kg m⁻³ liquid-water density
const _D_R_NEW    = 50.0e-6   # m — mean diameter of a newborn raindrop

# ---------------------------------------------------------------------------
# Local Tetens saturation helpers
# ---------------------------------------------------------------------------

@inline function _qsat_tet(T::Real, p::Real)
    Tc = T - 273.15
    es = 611.2 * exp(17.67 * Tc / (Tc + 243.5))          # Pa
    qs = _EPS_D * es / max(p - (1 - _EPS_D) * es, 1e-6)
    return qs, es
end

# dq_sat/dT via Clausius–Clapeyron applied to the Tetens form.
@inline function _dqsat_dT(T::Real, qs::Real)
    return _LV_D * qs / (_RV_D * T * T)
end

# ===========================================================================
# M1 — Full saturation adjustment (Bolton 1980)
# ===========================================================================

"""
    saturation_adjustment!(θ, qv, ql, ρ, p_col, dt)

One-step coupled qv ⇄ ql saturation adjustment with latent-heat feedback
into potential temperature `θ`.

Accepts either Aeolus `Field{CellCenter}` objects or bare 3-D arrays
sized `(Nx, Ny, Nz)` (interior extent). `ρ` may be a scalar or a
column vector — it is not used in the adjustment itself but is retained
in the signature so callers can pass the same profile they use for
buoyancy. `p_col` is a length-`Nz` vector of layer pressures (Pa) and
`dt` is passed through for future use in a time-implicit variant.

Algorithm: for each cell compute `T = θ · (p/1e5)^(R_d/c_p)`, evaluate
`q_sat(T, p)` via Tetens, then

  * if `qv > q_sat`, condense `Δq = (qv - q_sat) / (1 + L·dq_sat/dT/c_p)`
    (Bolton 1980 eq. 4), release `+L·Δq/c_p` into `T`, i.e.
    `Δθ = +L·Δq / (c_p·Π)` with `Π = (p/1e5)^(R_d/c_p)`.
  * if `qv < q_sat && ql > 0`, evaporate `min(ql, q_sat - qv)` and
    cool by the same `-L·Δq/(c_p·Π)`.

Mass is conserved between `qv` and `ql`; only the two are touched.
"""
function saturation_adjustment!(θ::AbstractArray{<:Real,3},
                                qv::AbstractArray{<:Real,3},
                                ql::AbstractArray{<:Real,3},
                                ρ, p_col::AbstractVector, dt::Real)
    Nx, Ny, Nz = size(θ)
    @assert size(qv) == size(ql) == size(θ) "θ, qv, ql must share shape"
    @assert length(p_col) >= Nz "p_col must cover all Nz layers"
    @inbounds for k in 1:Nz
        p     = float(p_col[k])
        exner = (p / _PREF_D)^(_RD_D / _CP_D)
        for j in 1:Ny, i in 1:Nx
            θe = θ[i, j, k]
            T  = θe * exner
            qs, _ = _qsat_tet(T, p)
            qve   = qv[i, j, k]
            qle   = ql[i, j, k]
            if qve > qs
                dqsdT = _dqsat_dT(T, qs)
                Δq    = (qve - qs) / (1.0 + _LV_D * dqsdT / _CP_D)
                Δq    = min(max(Δq, 0.0), qve)
                qv[i, j, k] = qve - Δq
                ql[i, j, k] = qle + Δq
                θ[i, j, k]  = θe  + _LV_D * Δq / (_CP_D * exner)
            elseif qve < qs && qle > 0
                Δq = min(qle, qs - qve)
                qv[i, j, k] = qve + Δq
                ql[i, j, k] = qle - Δq
                θ[i, j, k]  = θe  - _LV_D * Δq / (_CP_D * exner)
            end
        end
    end
    return (θ, qv, ql)
end

function saturation_adjustment!(θ::Field, qv::Field, ql::Field,
                                ρ, p_col::AbstractVector, dt::Real)
    return saturation_adjustment!(interior_view(θ), interior_view(qv),
                                  interior_view(ql), ρ, p_col, dt)
end

# ===========================================================================
# M2 — Two-moment microphysics scaffold
# ===========================================================================

"""
    TwoMomentState(q_v, q_l, q_r, q_i, N_c, N_r)

Container of the six prognostic microphysical fields (all
`Field{CellCenter}` on the same mesh) used by
`apply_two_moment_micro!` and `apply_ice_processes!`.

Access by name (`state.q_v`, …) just like a `NamedTuple`. Constructed
via `TwoMomentState(mesh)`, which initializes `N_c` to a maritime CCN
value of 100 cm⁻³ (=`1.0e8 m⁻³`) and everything else to zero.
"""
struct TwoMomentState{F<:Field}
    q_v::F
    q_l::F
    q_r::F
    q_i::F
    N_c::F
    N_r::F
end

"""
    TwoMomentState(mesh) -> TwoMomentState

Allocate the six prognostic fields on `mesh`, zero all mass mixing
ratios and `N_r`, and seed `N_c = 1.0e8 m⁻³` (100 cm⁻³, maritime CCN).
"""
function TwoMomentState(mesh::AbstractMesh)
    q_v = Field(CellCenter(), mesh)
    q_l = Field(CellCenter(), mesh)
    q_r = Field(CellCenter(), mesh)
    q_i = Field(CellCenter(), mesh)
    N_c = Field(CellCenter(), mesh)
    N_r = Field(CellCenter(), mesh)
    fill!(N_c.data, _N_C_INIT)
    fill!(N_r.data, 0.0)
    return TwoMomentState(q_v, q_l, q_r, q_i, N_c, N_r)
end

function Base.show(io::IO, s::TwoMomentState)
    m = s.q_v.mesh
    print(io, "TwoMomentState(q_v, q_l, q_r, q_i, N_c, N_r on ", m, ")")
end

Base.propertynames(::TwoMomentState) = (:q_v, :q_l, :q_r, :q_i, :N_c, :N_r)

"""
    apply_two_moment_micro!(state, dt; k_auto=1e-3, ql_c=1e-3, k_acc=2.2,
                            ρ_a=1.0, mesh=state.q_v.mesh, Vt_floor=4.0,
                            Vt_cap=10.0)

Advance the two-moment warm-rain state by `dt`. Steps:

  1. Autoconversion  ql → qr : `dq_r/dt = k_auto · max(q_l - q_l_crit, 0)`;
     the mass carved off populates `N_r` assuming a newborn drop diameter
     `D_R = 50 µm`, i.e. `ΔN_r = ρ_a · Δq_r / m₀` with
     `m₀ = π/6 · ρ_l · D_R³`.
  2. Accretion       ql → qr : `dq_r/dt = k_acc · q_l · q_r^{7/8}` (Kessler
     form); the fractional loss `Δq_l / q_l` is applied to `N_c` too, so
     droplet number tracks the mass being scavenged.
  3. Rain sedimentation      : cell-wise terminal velocity
     `V_t = 6.5 · (q_r · ρ_a / N_r)^{1/6}` (from the mean-mass equivalent
     of a Marshall–Palmer distribution), falling back to `Vt_floor = 4 m/s`
     when `N_r` is not populated, then a 1st-order upwind step on both
     `q_r` and `N_r` with a CFL cap at 0.5.

References: Morrison & Milbrandt (2015) two-moment framework; Kessler
(1969) rates for autoconv/accretion; Marshall & Palmer (1948) sizing.
Returns the mutated `state`.
"""
function apply_two_moment_micro!(state::TwoMomentState, dt::Real;
                                 k_auto::Real=1e-3, ql_c::Real=1e-3,
                                 k_acc::Real=2.2, ρ_a::Real=1.0,
                                 mesh::AbstractMesh=state.q_v.mesh,
                                 Vt_floor::Real=4.0, Vt_cap::Real=10.0)
    ii, jj, kk = interior(mesh)
    dz = mesh.dz
    Δt = float(dt)

    qld = state.q_l.data
    qrd = state.q_r.data
    ncd = state.N_c.data
    nrd = state.N_r.data

    # Mass of a newborn raindrop assumed by the autoconv → N_r mapping.
    m0 = (π / 6) * _RHO_LIQ * _D_R_NEW^3   # kg per new drop

    # ---- 1 & 2 : autoconversion + accretion (local rate updates) --------
    @inbounds for k in kk, j in jj, i in ii
        qle  = qld[i, j, k]
        qre  = qrd[i, j, k]
        auto = k_auto * max(qle - ql_c, 0.0)
        accr = k_acc  * max(qle, 0.0) * (max(qre, 0.0) ^ 0.875)
        dqa  = auto * Δt
        dqc  = accr * Δt
        dq_tot = dqa + dqc
        # Cap conversion to available liquid water.
        if dq_tot > max(qle, 0.0)
            scale  = max(qle, 0.0) / max(dq_tot, 1e-30)
            dqa   *= scale
            dqc   *= scale
            dq_tot = dqa + dqc
        end
        qld[i, j, k] = qle - dq_tot
        qrd[i, j, k] = qre + dq_tot
        # Populate N_r from autoconverted mass.
        if dqa > 0
            nrd[i, j, k] += (ρ_a * dqa) / m0
        end
        # Reduce N_c proportionally when accretion scavenges cloud water.
        if dqc > 0 && qle > 1e-12
            frac = dqc / qle
            ncd[i, j, k] = max(0.0, ncd[i, j, k] * (1.0 - frac))
        end
    end

    # ---- 3 : rain sedimentation (1st-order upwind, cell-wise Vt) --------
    kend = last(kk)
    @inbounds for k in kk, j in jj, i in ii
        qre = max(qrd[i, j, k], 0.0)
        nre = nrd[i, j, k]
        Vt  = (qre > 1e-12 && nre > 1.0) ?
                 6.5 * (qre * ρ_a / nre)^(1/6) : Vt_floor
        Vt  = clamp(Vt, 0.0, Vt_cap)
        α   = min(Vt * Δt / dz, 0.5)
        k_up = k + 1
        q_in  = k_up <= kend ? qrd[i, j, k_up] : 0.0
        q_out = qrd[i, j, k]
        qrd[i, j, k] = q_out + α * (q_in - q_out)
        n_in  = k_up <= kend ? nrd[i, j, k_up] : 0.0
        n_out = nrd[i, j, k]
        nrd[i, j, k] = n_out + α * (n_in - n_out)
    end

    return state
end

# ===========================================================================
# M3 — Ice / mixed-phase processes
# ===========================================================================

# Small helpers so the same kernel handles T given as a full 3-D array
# (interior-sized) or as a 1-D column vector of layer temperatures.
@inline _tget(T::AbstractArray{<:Real,3}, i, j, k, i0, j0, k0) = T[i - i0, j - j0, k - k0]
@inline _tget(T::AbstractVector{<:Real},  i, j, k, i0, j0, k0) = T[k - k0]
@inline function _tset!(T::AbstractArray{<:Real,3}, v, i, j, k, i0, j0, k0)
    T[i - i0, j - j0, k - k0] = v
end
# Vector T is treated as read-only (per-column mean) — no write-back to avoid
# corrupting a shared profile with per-cell perturbations.
@inline _tset!(::AbstractVector{<:Real}, v, i, j, k, i0, j0, k0) = v

"""
    apply_ice_processes!(state, T_col, dt) -> state

Homogeneous freezing / melting between `q_l` and `q_i`:

  * `T < 235 K`  : all cloud liquid freezes, `q_l → q_i`, releasing
    `L_f·Δq` — added to `T_col` as `+L_f·Δq/c_p` (when `T_col` is a
    3-D array; skipped when it is a per-column vector).
  * `T > 273.15 K` and `q_i > 0` : ice melts back to liquid, absorbing
    `L_f·Δq`, i.e. `T_col -= L_f·Δq/c_p`.

`T_col` may be an `(Nx, Ny, Nz)` interior-sized array or a length-`Nz`
column vector. `dt` is accepted for API symmetry but the process is
treated as instantaneous (typical for homogeneous freezing in cheap
schemes).
"""
function apply_ice_processes!(state::TwoMomentState, T_col::AbstractArray,
                              dt::Real)
    mesh = state.q_l.mesh
    ii, jj, kk = interior(mesh)
    qld = state.q_l.data
    qid = state.q_i.data
    # offsets so `_tget/_tset!` can index a halo-free array with (i,j,k)
    # that live in the halo'd coordinate system.
    i0 = first(ii) - 1
    j0 = first(jj) - 1
    k0 = first(kk) - 1
    @inbounds for k in kk, j in jj, i in ii
        T = _tget(T_col, i, j, k, i0, j0, k0)
        qle = qld[i, j, k]; qie = qid[i, j, k]
        if T < _T_HOM && qle > 0.0
            Δq = qle
            qld[i, j, k] = 0.0
            qid[i, j, k] = qie + Δq
            _tset!(T_col, T + _LF_D * Δq / _CP_D, i, j, k, i0, j0, k0)
        elseif T > _T_ICE && qie > 0.0
            Δq = qie
            qid[i, j, k] = 0.0
            qld[i, j, k] = qle + Δq
            _tset!(T_col, T - _LF_D * Δq / _CP_D, i, j, k, i0, j0, k0)
        end
    end
    return state
end

# ===========================================================================
# M4 — Fog / dew condensation (delegates to Aeolus.RadiationFog)
# ===========================================================================
#
# This used to be a local scheme that shipped DISABLED because "its
# warming term amplifies". Two defects, both measured (Phase 44, #11):
#
#   * whenever the SURFACE was below the dew point it removed vapour from
#     the AIR and put the latent heat into the AIR, even when the air was
#     subsaturated. Nothing tied the heating to air-side saturation, so
#     with vapour resupplied by evaporation it heated a 288 K cell to
#     363 K in 24 h while its RH fell to 0.016;
#   * in supersaturated air the explicit `rate·dt·(q_v - q_s)` step lacked
#     the latent-heat feedback and overshot below saturation once
#     rate·dt ≳ 0.4.
#
# The physics now lives in `src/RadiationFog.jl` (one source of truth,
# with MoistThermodynamics saturation and latent heats): air-side fog is
# an enthalpy-conserving saturation adjustment into `ql`, and dew is an
# implicit bulk exchange whose latent heat goes to the SURFACE. This
# wrapper only adapts the urban example's Fields to it.

"""
    apply_fog_dew!(θ, qv, ql, q_dew, Ts, dt; p = 1.01325e5, dz, ρ = 1.0,
                   C_e = 1e-3, U = 3.0, τ_fog = 0.0,
                   fog_levels = axes(θ, 3))
        -> H_dew (W m⁻², 2-D)

Dew onto the surface, then fog in the air, for the urban example.

  * **Dew** — `Aeolus.RadiationFog.dew_exchange!` on the lowest layer of
    `qv`: vapour is deposited into the 2-D reservoir `q_dew` (kg m⁻²)
    while `q_v > q_s(T_s)`, and re-evaporated from it when the surface
    warms. The latent heat is **returned** as `H_dew` (positive = heating
    the surface); the caller adds it to the surface energy budget. The
    air temperature is not touched by dew.
  * **Fog** — `Aeolus.RadiationFog.fog_adjust!` on `θ`, `qv`, `ql` in
    `fog_levels`: enthalpy-conserving saturation adjustment (relaxed over
    `τ_fog` if > 0), condensing into and evaporating from `ql`.

`p` is the surface pressure used for dew and either a scalar or a
per-level vector for fog. Both steps are unconditionally stable in `dt`
and cannot overshoot saturation.
"""
function apply_fog_dew!(θ::AbstractArray{<:Real,3}, qv::AbstractArray{<:Real,3},
                        ql::AbstractArray{<:Real,3}, q_dew::AbstractMatrix,
                        Ts::AbstractMatrix, dt::Real;
                        p = 1.01325e5, dz::Real, ρ::Real = 1.0,
                        C_e::Real = 1e-3, U::Real = 3.0,
                        τ_fog::Real = 0.0, fog_levels = axes(θ, 3))
    p_sfc = p isa AbstractVector ? first(p) : p
    H = Aeolus.RadiationFog.dew_exchange!(qv, q_dew, Ts, p_sfc, dt;
                                          dz = dz, ρ = ρ, C_e = C_e, U = U)
    Aeolus.RadiationFog.fog_adjust!(θ, qv, ql, p, dt; τ = τ_fog,
                                    levels = fog_levels)
    return H
end

function apply_fog_dew!(θ::Field, qv::Field, ql::Field, q_dew::AbstractMatrix,
                        Ts::AbstractMatrix, dt::Real; dz::Real = θ.mesh.dz,
                        kwargs...)
    return apply_fog_dew!(interior_view(θ), interior_view(qv),
                          interior_view(ql), q_dew, Ts, dt; dz = dz, kwargs...)
end
