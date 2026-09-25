"""
    RadiationFog

Fog and dew condensation coupled consistently to the atmospheric vapour
field — Era IV, Phase 44.

This module owns the *thermodynamic* half of radiation fog: how much
vapour condenses in the air (fog) or onto a cold surface (dew), where the
latent heat goes, and why the step cannot overshoot saturation for any
time step. The droplet-radiative feedback that actually drives radiation
fog (fog cooling itself by emitting longwave from droplet tops) is the
subject of the follow-up issue and is meant to be added here, on top of
these functions.

WHY THIS EXISTS — THE BUG IT REPLACES

The urban example shipped `apply_fog_dew!` disabled because "its warming
term amplifies". The diagnosis, measured in a one-cell box model with the
example's own settings (`rate = 1e-4`, `dt = 2 s`, surface at 280 K,
air at 288 K and `qv = 0.010`, i.e. RH = 0.95 and vapour resupplied by
evaporation/advection as in the full run):

  1. **Condensation in subsaturated air.** Whenever the *surface* was
     below the dew point the old scheme removed vapour from the *air* and
     released its latent heat into the *air* — even though the air was
     subsaturated. Nothing tied the heating to air-side saturation, so
     there was no thermodynamic brake: the air warmed from 288 K to
     363 K in 24 h while its RH fell to 0.016 and it kept "condensing".
     Physically, dew forms at the surface and its latent heat goes into
     the surface energy budget; the air loses vapour but not enthalpy per
     unit of dry air beyond what that vapour carried.
  2. **Explicit-rate overshoot.** In supersaturated air the step removed
     `rate·dt·(q_v − q_s)` without the `1 + (L/c_p) dq_s/dT` latent-heat
     feedback, so for `rate·dt ≳ 1/(1+γ)` it overshot to *below*
     saturation (`q_v − q_s` = −6.3e-4 kg/kg at `rate·dt = 1`, −2.9e-3 at
     `rate·dt = 3`).

THE FIX

  * **Fog** ([`fog_condensation`](@ref)) conserves the isobaric moist
    enthalpy of the contract (`moist_enthalpy`) exactly, with `L_v(T)` and
    `q_s(T, p)` from `MoistThermodynamics` (one source of truth). The
    target is the saturation-adjusted state, found by a bracketed solve of
    `q_v − Δ = q_s(T(Δ), p)`; a finite timescale `τ` relaxes toward it by
    the exact exponential factor `1 − exp(−dt/τ)` ∈ [0, 1). Because the
    residual is monotone in the condensed amount, a partial step can
    never cross saturation — for any `dt`.
  * **Dew** ([`dew_deposition`](@ref)) draws vapour from the lowest air
    layer only while `q_v > q_s(T_s)`, with an implicit (exponential)
    bulk-exchange law, so the air can at most be dried to the surface's
    saturation humidity. Its latent heat `L_v(T_s)·m` is *returned* for
    the surface energy budget and never added to the air.

References

  * Monteith, J. L. (1957). Dew. *Q. J. R. Meteorol. Soc.*, 83, 322-341.
  * Garratt, J. R. & Segal, M. (1988). On the contribution of atmospheric
    moisture to dew formation. *Bound.-Layer Meteorol.*, 45, 209-236.
  * Bott, A., Sievers, U. & Zdunkowski, W. (1990). A radiation fog model
    with a detailed treatment of the interaction between radiative
    transfer and fog microphysics. *J. Atmos. Sci.*, 47, 2153-2166.
  * Emanuel, K. A. (1994). *Atmospheric Convection*, Ch. 4 (moist
    enthalpy).
"""
module RadiationFog

using ..MoistThermodynamics
using ..MoistThermodynamics: ThermodynamicConstants, DEFAULT_CONSTANTS,
                             PhasePartition, moist_enthalpy,
                             heat_capacity_moist, latent_heat_vapor,
                             saturation_specific_humidity, exner_function

export relaxation_fraction, fog_equilibrium, fog_condensation, fog_adjust!,
       dew_deposition, dew_exchange!

# =========================================================================
# Helpers
# =========================================================================

"""
    relaxation_fraction(dt, τ) -> f ∈ [0, 1]

Fraction of the distance to equilibrium covered in one step of length
`dt` by linear relaxation with timescale `τ`: `f = 1 − exp(−dt/τ)`.
`τ ≤ 0` means instantaneous adjustment (`f = 1`).

This is the *exact* solution of `dx/dt = −x/τ` over the step, not the
explicit-Euler `dt/τ`, which exceeds 1 — i.e. overshoots — as soon as
`dt > τ`.
"""
relaxation_fraction(dt::Real, τ::Real) =
    τ <= 0 ? one(float(dt)) : -expm1(-float(dt) / float(τ))

# Isobaric moist enthalpy per unit mass — the contract's `moist_enthalpy`,
# the exact invariant of reversible isobaric phase change. Liquid only
# (fog is taken as supercooled liquid below 0 °C; ice fog is out of scope).
@inline _enthalpy(T, qt, ql, c) =
    moist_enthalpy(T, PhasePartition(qt, ql, 0.0), c)

# Temperature of the state (qt, ql1) that has the same moist enthalpy h0.
# h is exactly linear in T with slope c_pm(q1) (Kirchhoff), so one linear
# solve is exact; equivalently c_pm(q1) ΔT = L_v(T0) Δq.
@inline function _isenthalpic_T(h0, T0, qt, ql1, c)
    q1 = PhasePartition(qt, ql1, 0.0)
    return T0 + (h0 - _enthalpy(T0, qt, ql1, c)) / heat_capacity_moist(q1, c)
end

# =========================================================================
# Fog: saturation-limited condensation in the air
# =========================================================================

"""
    fog_equilibrium(T, p, qv, ql, c = DEFAULT_CONSTANTS; tol = 1e-14)
        -> (Δq, T_eq)

Amount of vapour `Δq` (kg/kg; negative = fog evaporates) that must
condense, at constant pressure `p` and constant moist enthalpy, to bring
an air parcel with temperature `T`, vapour `qv` and fog liquid `ql`
exactly to saturation over liquid water — or to evaporate all of `ql`
if even that leaves the air subsaturated. `T_eq` is the temperature of
the adjusted state.

The residual `g(Δ) = (q_v − Δ) − q_s(T(Δ), p)` is strictly decreasing in
`Δ` (condensing both removes vapour and, by warming, raises `q_s`), and
`Δ ∈ [−ql, qv]`, so a safeguarded Newton iteration inside that bracket
cannot fail. Subsaturated air with no fog returns `(0, T)` after one
saturation evaluation.
"""
function fog_equilibrium(T::Real, p::Real, qv::Real, ql::Real,
                         c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                         tol::Real = 1e-14, maxiter::Int = 100)
    T0 = float(T); pp = float(p)
    qv0 = max(float(qv), 0.0); ql0 = max(float(ql), 0.0)
    qt = qv0 + ql0
    h0 = _enthalpy(T0, qt, ql0, c)
    Tof(Δ) = _isenthalpic_T(h0, T0, qt, ql0 + Δ, c)
    g(Δ)   = (qv0 - Δ) - saturation_specific_humidity(Tof(Δ), pp, c)

    g0 = g(0.0)
    g0 == 0 && return (0.0, T0)
    if g0 < 0                      # subsaturated: can only evaporate fog
        ql0 == 0 && return (0.0, T0)
        lo, hi = -ql0, 0.0
        g(lo) <= 0 && return (lo, Tof(lo))   # all fog evaporates
    else                           # supersaturated: condense
        lo, hi = 0.0, qv0
    end
    # Safeguarded Newton on the bracket [lo, hi] with g(lo) > 0 > g(hi).
    Δ = 0.5 * (lo + hi)
    for _ in 1:maxiter
        gΔ = g(Δ)
        # Converged: keep THIS Δ. (Testing after the Newton update instead
        # returned the bracket midpoint whenever g hit exactly 0 — the
        # update set hi = Δ, the Newton iterate Δ then failed the strict
        # `lo < Δn < hi` safeguard and was replaced by (lo+hi)/2.)
        abs(gΔ) < tol && break
        gΔ > 0 ? (lo = Δ) : (hi = Δ)
        (hi - lo) < tol && break
        Tn  = Tof(Δ)
        qs  = qv0 - Δ - gΔ
        # dq_s/dT from Clausius-Clapeyron, dT/dΔ ≈ L_v / c_pm.
        Lv  = latent_heat_vapor(Tn, c)
        Rv  = c.gas_constant / c.molmass_water
        dg  = -1 - qs * Lv / (Rv * Tn^2) *
                   Lv / heat_capacity_moist(PhasePartition(qt, ql0 + Δ, 0.0), c)
        Δn  = Δ - gΔ / dg
        Δ   = (lo < Δn < hi) ? Δn : 0.5 * (lo + hi)
    end
    return (Δ, Tof(Δ))
end

"""
    fog_condensation(T, p, qv, ql, dt; τ = 0, c = DEFAULT_CONSTANTS)
        -> (; T, qv, ql, Δq, latent_heat)

One fog condensation/evaporation step for a single air parcel at
pressure `p`. The parcel relaxes toward the saturation-adjusted state of
[`fog_equilibrium`](@ref) by the fraction
[`relaxation_fraction`](@ref)`(dt, τ)`; `τ = 0` (default) is full
saturation adjustment, the standard LES choice since the phase-relaxation
time of fog droplets (a few seconds at 100 cm⁻³) is shorter than any
dynamical step.

Properties, all tested:

  * total water `qv + ql` is conserved exactly;
  * moist enthalpy (`MoistThermodynamics.moist_enthalpy`) is conserved
    to round-off, which makes the sensible warming exactly
    `c_pm ΔT = L_v(T)·Δq` (the released heat, returned as `latent_heat`,
    J/kg);
  * no condensation occurs in subsaturated air, and the step never
    crosses saturation for any `dt` and any `τ ≥ 0`.
"""
function fog_condensation(T::Real, p::Real, qv::Real, ql::Real, dt::Real;
                          τ::Real = 0.0,
                          c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Δeq, _ = fog_equilibrium(T, p, qv, ql, c)
    f  = relaxation_fraction(dt, τ)
    Δ  = f * Δeq
    qv0 = max(float(qv), 0.0); ql0 = max(float(ql), 0.0)
    qt = qv0 + ql0
    h0 = _enthalpy(float(T), qt, ql0, c)
    T1 = Δ == 0 ? float(T) : _isenthalpic_T(h0, float(T), qt, ql0 + Δ, c)
    return (T = T1, qv = qv0 - Δ, ql = ql0 + Δ, Δq = Δ,
            latent_heat = latent_heat_vapor(float(T), c) * Δ)
end

"""
    fog_adjust!(θ, qv, ql, p, dt; τ = 0, levels = axes(θ, 3),
                c = DEFAULT_CONSTANTS) -> total condensed (kg/kg, summed)

Apply [`fog_condensation`](@ref) to every cell of 3-D arrays `θ`
(potential temperature, K), `qv` and `ql` (kg/kg) in the vertical
`levels`. `p` is a scalar pressure or a vector of layer pressures indexed
by `k`. Temperature is `T = θ Π(p)` with the dry Exner function of the
thermodynamic contract, and the enthalpy-conserving warming is written
back as `Δθ = ΔT / Π`.

Returns the domain sum of the condensed amount (negative when fog
evaporated), which callers can use as a budget check.
"""
function fog_adjust!(θ::AbstractArray{<:Real,3}, qv::AbstractArray{<:Real,3},
                     ql::AbstractArray{<:Real,3}, p, dt::Real;
                     τ::Real = 0.0, levels = axes(θ, 3),
                     c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    size(θ) == size(qv) == size(ql) ||
        throw(DimensionMismatch("fog_adjust!: θ, qv, ql must share shape"))
    Nx, Ny, _ = size(θ)
    total = 0.0
    @inbounds for k in levels
        pk = p isa AbstractVector ? float(p[k]) : float(p)
        Π  = exner_function(pk, PhasePartition(0.0), c)
        for j in 1:Ny, i in 1:Nx
            T = θ[i, j, k] * Π
            r = fog_condensation(T, pk, qv[i, j, k], ql[i, j, k], dt; τ = τ, c = c)
            r.Δq == 0 && continue
            θ[i, j, k]  = r.T / Π
            qv[i, j, k] = r.qv
            ql[i, j, k] = r.ql
            total += r.Δq
        end
    end
    return total
end

# =========================================================================
# Dew: vapour deposition onto a surface below the dew point
# =========================================================================

"""
    dew_deposition(qv, Ts, p, dt; dz, ρ = 1.2, C_e = 1e-3, U = 3.0,
                   dew = Inf, c = DEFAULT_CONSTANTS)
        -> (; Δqv, mass, surface_heat)

Vapour exchange between the lowest air layer (thickness `dz`, density
`ρ`, specific humidity `qv`) and a surface at temperature `Ts` over one
step `dt`, by the bulk law

    ρ dz dq_v/dt = −ρ g_a (q_v − q_s(T_s, p)),    g_a = C_e U.

Solved exactly over the step with `T_s` held fixed, so
`Δq_v = −(q_v − q_s(T_s))·(1 − exp(−g_a dt/dz))`: the air can at most be
dried to the surface saturation humidity, whatever `dt` is.

  * `q_v > q_s(T_s)` — the surface is below the dew point and dew forms:
    `mass = −ρ dz Δq_v > 0` kg m⁻².
  * `q_v < q_s(T_s)` — existing dew (`dew`, kg m⁻², the amount available)
    evaporates, limited by what is there; with `dew = 0` nothing happens.

Saturation is taken over **liquid** water. Below 273.15 K deposition
is really frost (hoar), which would saturate over ice and release
`L_s`; that regime is out of scope here, and dew on a sub-freezing
surface is treated as supercooled liquid.

`surface_heat = L_v(T_s)·mass` (J m⁻²) is the latent heat released **at
the surface** (negative when dew evaporates). It belongs in the surface
energy budget; the air temperature is untouched, because the vapour that
left carried its own sensible enthalpy with it. This is the coupling the
original urban `apply_fog_dew!` got wrong.
"""
function dew_deposition(qv::Real, Ts::Real, p::Real, dt::Real;
                        dz::Real, ρ::Real = 1.2, C_e::Real = 1e-3,
                        U::Real = 3.0, dew::Real = Inf,
                        c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    qs = saturation_specific_humidity(Ts, p, c)
    f  = relaxation_fraction(dt, dz / (C_e * U))
    Δqv = -(float(qv) - qs) * f
    if Δqv > 0                                  # evaporation of existing dew
        Δqv = min(Δqv, max(float(dew), 0.0) / (ρ * dz))
    end
    mass = -ρ * dz * Δqv
    return (Δqv = Δqv, mass = mass,
            surface_heat = latent_heat_vapor(Ts, c) * mass)
end

"""
    dew_exchange!(qv, q_dew, Ts, p, dt; dz, ρ = 1.2, C_e = 1e-3, U = 3.0,
                  c = DEFAULT_CONSTANTS) -> H (W m⁻²)

Apply [`dew_deposition`](@ref) column by column: `qv` is the 3-D vapour
array (only level 1 is touched) or a 2-D array of lowest-layer values,
`q_dew` the 2-D surface dew reservoir (kg m⁻², updated in place), `Ts`
the 2-D surface temperature. Returns the 2-D latent heating of the
surface, W m⁻², positive when dew forms — add it to the surface energy
balance, not to the air.
"""
function dew_exchange!(qv::AbstractArray{<:Real}, q_dew::AbstractMatrix,
                       Ts::AbstractMatrix, p::Real, dt::Real;
                       dz::Real, ρ::Real = 1.2, C_e::Real = 1e-3,
                       U::Real = 3.0,
                       c::ThermodynamicConstants = DEFAULT_CONSTANTS)
    Nx, Ny = size(Ts)
    H = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        q = ndims(qv) == 3 ? qv[i, j, 1] : qv[i, j]
        r = dew_deposition(q, Ts[i, j], p, dt; dz = dz, ρ = ρ, C_e = C_e,
                           U = U, dew = q_dew[i, j], c = c)
        r.mass == 0 && continue
        if ndims(qv) == 3
            qv[i, j, 1] = q + r.Δqv
        else
            qv[i, j] = q + r.Δqv
        end
        q_dew[i, j] = max(q_dew[i, j] + r.mass, 0.0)
        H[i, j] = r.surface_heat / dt
    end
    return H
end

end # module
