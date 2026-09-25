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
#   M4  apply_fog_dew!            — cell-local dew / radiation-fog
#                                    condensation when T < T_dew.
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
                            ρ_a=1.0, mesh=state.q_v.mesh)

Advance the two-moment warm-rain state by `dt`. Steps:

  1. Autoconversion  ql → qr : `dq_r/dt = k_auto · max(q_l - q_l_crit, 0)`;
     the mass carved off populates `N_r` assuming a newborn drop diameter
     `D_R = 50 µm`, i.e. `ΔN_r = ρ_a · Δq_r / m₀` with
     `m₀ = π/6 · ρ_l · D_R³`.
  2. Accretion       ql → qr : `dq_r/dt = k_acc · q_l · q_r^{7/8}` (Kessler
     form); the fractional loss `Δq_l / q_l` is applied to `N_c` too, so
     droplet number tracks the mass being scavenged.
  3. Rain sedimentation (Phase 44): delegated column by column to
     `Aeolus.p3_sediment_rain!` — `q_r` falls at the mass-weighted and
     `N_r` at the number-weighted speed of the gamma drop spectrum given
     by `(q_r, N_r)` (drop terminal velocities of
     `Superdroplet.terminal_velocity`), flux-form upwind with Courant
     sub-stepping, so rain mass is conserved exactly and size sorting is
     represented. This replaces a single cell-wise speed
     `6.5 (q_r ρ_a / N_r)^{1/6}` with a 4 m/s floor whose upwind update
     took the inflow from above at the *receiving* cell's Courant number
     and so created or destroyed mass wherever the speed varied (+0.068 %
     in 20 s for a 9-level shaft of 0-1 g kg⁻¹).

`N_r` is carried per m³ here and converted to per kg (÷ ρ_a) for the P3
call. References: Morrison & Milbrandt (2015) two-moment framework;
Kessler (1969) rates for autoconv/accretion.
Returns the mutated `state`.
"""
function apply_two_moment_micro!(state::TwoMomentState, dt::Real;
                                 k_auto::Real=1e-3, ql_c::Real=1e-3,
                                 k_acc::Real=2.2, ρ_a::Real=1.0,
                                 mesh::AbstractMesh=state.q_v.mesh)
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

    # ---- 3 : rain sedimentation: size-dependent fall speeds (P3) --------
    nz = length(kk)
    ρcol = fill(float(ρ_a), nz)
    qcol = zeros(nz); ncol = zeros(nz)
    @inbounds for j in jj, i in ii
        for (n, k) in enumerate(kk)
            qcol[n] = max(qrd[i, j, k], 0.0)
            ncol[n] = max(nrd[i, j, k], 0.0) / ρ_a      # m⁻³ → kg⁻¹
        end
        any(>(0), qcol) || continue
        p3_sediment_rain!(qcol, ncol, ρcol, dz, Δt)
        for (n, k) in enumerate(kk)
            qrd[i, j, k] = qcol[n]
            nrd[i, j, k] = ncol[n] * ρ_a
        end
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
# M4 — Fog / dew condensation
# ===========================================================================

# Invert Tetens to recover the dew-point temperature from a vapour
# specific humidity qv at pressure p.  qv = ε es / (p - (1-ε) es)
# ⇒ es = qv p / (ε + (1-ε) qv). Then Tetens is inverted analytically.
@inline function _dew_point(qv::Real, p::Real)
    es = qv * p / (_EPS_D + (1 - _EPS_D) * qv)
    r  = log(max(es, 1e-6) / 611.2)
    return 273.15 + 243.5 * r / max(17.67 - r, 1e-6)
end

"""
    apply_fog_dew!(θ, qv, q_dew, Ts, dt; ρ=1.0, dz, p=1.0e5, rate=0.1)

Cell-local fog / dew condensation for the first atmospheric layer.
For each `(i, j)`, temperature is diagnosed from `θ[i,j,1]` via the
Exner factor `Π = (p/1e5)^(R_d/c_p)`, `T_dew` is inverted analytically
from `qv[i,j,1]`, and — when either the air (`T < T_dew`) or the ground
(`Ts[i,j] < T_dew`) is below the dew point — a small fraction of the
saturation excess is condensed:

    Δq = rate · dt · max(qv - q_sat(T, p), 0)

`Δq` is removed from `qv[i,j,1]`, deposited into the 2-D `q_dew[i,j]`
(taken by reference, units `kg m⁻²` — the areal water column now on
the surface as dew/fog droplets = `ρ · Δq · dz`), and the corresponding
latent-heat release `+L_v·Δq/(c_p·Π)` is added to `θ[i,j,1]`.

`θ` and `qv` can be `Field` objects or interior-sized 3-D arrays;
`q_dew` is any 2-D `(Nx, Ny)` array; `Ts` is the 2-D surface-temp
matrix. `dz` is the vertical cell height (used to convert mixing
ratio into an areal water content). `rate` (1/s) controls how fast
the sub-saturation column relaxes toward `q_sat`.
"""
function apply_fog_dew!(θ::AbstractArray{<:Real,3}, qv::AbstractArray{<:Real,3},
                        q_dew::AbstractMatrix, Ts::AbstractMatrix,
                        dt::Real; ρ::Real = 1.0, dz::Real = 1.0,
                        p::Real = _PREF_D, rate::Real = 0.1)
    Nx, Ny, _ = size(θ)
    exner = (p / _PREF_D)^(_RD_D / _CP_D)
    invΠ  = 1.0 / exner
    @inbounds for j in 1:Ny, i in 1:Nx
        θe = θ[i, j, 1]
        qve = qv[i, j, 1]
        T   = θe * exner
        T_d = _dew_point(qve, p)
        if T < T_d || Ts[i, j] < T_d
            qs, _ = _qsat_tet(T, p)
            excess = max(qve - qs, 0.0)
            # For radiation-fog (Ts cold, T just above dew) we still want
            # some condensation, so use the (T_d - min(T, Ts))/T_d as a
            # secondary driver whenever the vapour is not yet super-saturated.
            if excess == 0.0
                ΔT = max(T_d - min(T, Ts[i, j]), 0.0)
                excess = qve * min(ΔT / 20.0, 1.0)   # cap fraction at 100 %
            end
            Δq = rate * dt * excess
            if Δq > 0
                qv[i, j, 1]   = qve - Δq
                q_dew[i, j]  += ρ * Δq * dz
                θ[i, j, 1]    = θe + _LV_D * Δq * invΠ / _CP_D
            end
        end
    end
    return (θ, qv, q_dew)
end

function apply_fog_dew!(θ::Field, qv::Field, q_dew::AbstractMatrix,
                        Ts::AbstractMatrix, dt::Real; ρ::Real = 1.0,
                        dz::Real = θ.mesh.dz, p::Real = _PREF_D,
                        rate::Real = 0.1)
    return apply_fog_dew!(interior_view(θ), interior_view(qv), q_dew, Ts,
                          dt; ρ = ρ, dz = dz, p = p, rate = rate)
end
