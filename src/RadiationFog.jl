"""
    RadiationFog

Fog and dew condensation coupled consistently to the atmospheric vapour
field — Era IV, Phase 44.

This module owns radiation fog. Its *thermodynamic* half (#11): how much
vapour condenses in the air (fog) or onto a cold surface (dew), where the
latent heat goes, and why the step cannot overshoot saturation for any
time step. Its *radiative* half (#9, `radiation_fog/coupling.jl`): the
droplet field handed to the Era III radiation, so that fog cools itself
by emitting longwave from its top — the feedback that thickens it. And a
1-D radiation-fog column (`radiation_fog/column.jl`) that couples the
two with turbulence, a surface energy balance and droplet settling,
validated on the ParisFog case of 14–15 Nov 2011.

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
                             temperature_from_moist_enthalpy,
                             isobaric_saturation_adjustment,
                             heat_capacity_moist, latent_heat_vapor,
                             saturation_specific_humidity, exner_function
using ..Backends: AbstractBackend, plane_map!, ordered_sum, backend, foreach_column!

# Era III radiation and Era IV droplet physics the fog column delegates to
# (issue #9): no optics or transfer is computed in this module.
using ..RadiationCore: SpectralGrid, OpticalProperties, RadiativeFluxes,
                       AtmosphericColumn, longwave_bands, shortwave_bands,
                       allocate_optics, allocate_fluxes, zero_fluxes!,
                       combine_optics!, flux_divergence!, compute_optics!,
                       STEFAN_BOLTZMANN
using ..GasOptics: correlated_k_optics, us_standard_column
using ..TwoStream: mie_cloud_optics, two_stream_solver, solve_lw!, solve_sw!
using ..AerosolCloud: droplet_effective_radius
using ..Superdroplet: terminal_velocity
using ..SurfaceExchange: DEFAULT_SURFACE_CONSTANTS   # von Kármán constant
using ..TurbulenceStability: ShortTail, stability_factor_m, gradient_richardson
using ..SolarGeometry: solar_time, solar_position, earth_sun_distance_factor
import ..TimeSteppers: step!      # the column extends the generic step!/run!
import ..Simulations: run!
using ..MoistThermodynamics: saturation_vapor_pressure,
                             density_potential_temperature,
                             virtual_potential_temperature, air_density

export relaxation_fraction, fog_equilibrium, fog_condensation, fog_adjust!,
       dew_deposition, dew_exchange!, dew_exchange_columns!
# Phase 44 (#9) — droplet-radiative feedback and the 1-D radiation-fog column
export fog_effective_radius, FogRadiation, fog_radiation,
       fog_atmospheric_column, fog_radiative_transfer!,
       fog_radiative_heating!, fog_top_flux_jump,
       radiative_condensation_rate,
       FogColumnSettings, FogBudget, FogColumn, stretched_levels,
       parisfog_2011_column, column_energy, column_water,
       fog_liquid_water_path, update_radiation!, moist_stability,
       eddy_diffusivity, surface_exchange, settling_velocity,
       fog_diagnostics, fog_life_cycle

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

This is the contract's process adjustment,
`MoistThermodynamics.isobaric_saturation_adjustment(T, p, qv, ql, c)`,
with no rain or ice: the solver (a safeguarded Newton iteration on the
monotone residual `(q_v − Δ) − q_s(T(Δ), p)` over `Δ ∈ [−ql, qv]`) lives
there, one copy for every scheme (issue #82). It was written here first
(#85) and moved to the contract unchanged, so fog results are
bit-for-bit those of #85.
"""
fog_equilibrium(T::Real, p::Real, qv::Real, ql::Real,
                c::ThermodynamicConstants = DEFAULT_CONSTANTS;
                tol::Real = 1e-14, maxiter::Int = 100) =
    Tuple(isobaric_saturation_adjustment(T, p, qv, ql, c; tol = tol, maxiter = maxiter))

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
    T1 = Δ == 0 ? float(T) :
         temperature_from_moist_enthalpy(h0, PhasePartition(qt, ql0 + Δ, 0.0), c;
                                         T_ref = float(T))
    return (T = T1, qv = qv0 - Δ, ql = ql0 + Δ, Δq = Δ,
            latent_heat = latent_heat_vapor(float(T), c) * Δ)
end

"""
    fog_adjust!(θ, qv, ql, p, dt; τ = 0, levels = axes(θ, 3),
                c = DEFAULT_CONSTANTS, backend = backend(θ)) -> total condensed (kg/kg, summed)

Apply [`fog_condensation`](@ref) to every cell of 3-D arrays `θ`
(potential temperature, K), `qv` and `ql` (kg/kg) in the vertical
`levels`. `p` is a scalar pressure or a vector of layer pressures indexed
by `k`. Temperature is `T = θ Π(p)` with the dry Exner function of the
thermodynamic contract, and the enthalpy-conserving warming is written
back as `Δθ = ΔT / Π`.

A cell without fog water (`ql ≤ 0`) whose vapour is below saturation,
``q_v < (1 − 10^{-8})\\,q_s(T, p)``, is left as it is without the
adjustment (it cannot condense, and the adjustment returns `Δq = 0` for
it exactly, so the skip changes no bits; maintainer decision on #291).

The cells are independent: the levels run in parallel on `backend`
(one plane per work item, KA-m #301), each plane in cell order, so the
fields are identical at every thread count and to a serial sweep.

Returns the domain sum of the condensed amount (negative when fog
evaporated), which callers can use as a budget check: each plane's sum in
cell order, the planes added in ascending `k` (thread-invariant; before
KA-m one running sum over all cells, so the last bits of the total moved
once).
"""
function fog_adjust!(θ::AbstractArray{<:Real,3}, qv::AbstractArray{<:Real,3},
                     ql::AbstractArray{<:Real,3}, p, dt::Real;
                     τ::Real = 0.0, levels = axes(θ, 3),
                     c::ThermodynamicConstants = DEFAULT_CONSTANTS,
                     backend::AbstractBackend = backend(θ))
    size(θ) == size(qv) == size(ql) ||
        throw(DimensionMismatch("fog_adjust!: θ, qv, ql must share shape"))
    Nx, Ny, _ = size(θ)
    lv = UnitRange{Int}(levels)
    isempty(lv) && return 0.0
    checkbounds(θ, 1:Nx, 1:Ny, lv)
    p isa AbstractVector && checkbounds(p, lv)
    parts = Vector{Float64}(undef, length(lv))
    plane_map!(parts, backend, _fog_plane!, (1:Nx, 1:Ny, lv), θ, qv, ql, p,
               float(dt), float(τ), c)
    return ordered_sum(parts)
end

# One level of fog_adjust!: the cells in (j, i) order, the plane's sum of
# the condensed amount in that order.
function _fog_plane!(k, ir, jr, θ, qv, ql, p, dt, τ, c)
    pk = p isa AbstractVector ? float(@inbounds p[k]) : float(p)
    Π  = exner_function(pk, PhasePartition(0.0), c)
    total = 0.0
    @inbounds for j in jr, i in ir
        T = θ[i, j, k] * Π
        qvi, qli = qv[i, j, k], ql[i, j, k]
        # no fog water and subsaturated: Δq = 0 exactly (the adjustment
        # would find T(Δ = 0) = T and q_v − q_s(T) < 0); skip the solve
        qli <= 0 && qvi < (1 - 1e-8) * saturation_specific_humidity(T, pk, c) && continue
        r = fog_condensation(T, pk, qvi, qli, dt; τ = τ, c = c)
        r.Δq == 0 && continue
        θ[i, j, k]  = r.T / Π
        qv[i, j, k] = r.qv
        ql[i, j, k] = r.ql
        total += r.Δq
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
                  c = DEFAULT_CONSTANTS, out = nothing) -> H (W m⁻²)

Apply [`dew_deposition`](@ref) column by column: `qv` is the 3-D vapour
array (only level 1 is touched) or a 2-D array of lowest-layer values,
`q_dew` the 2-D surface dew reservoir (kg m⁻², updated in place), `Ts`
the 2-D surface temperature. Returns the 2-D latent heating of the
surface, W m⁻², positive when dew forms — add it to the surface energy
balance, not to the air. `H` is a new array, or `out` (overwritten) when
given, so a time loop can avoid the allocation.
"""
function dew_exchange!(qv::AbstractArray{<:Real}, q_dew::AbstractMatrix,
                       Ts::AbstractMatrix, p::Real, dt::Real;
                       dz::Real, ρ::Real = 1.2, C_e::Real = 1e-3,
                       U::Real = 3.0,
                       c::ThermodynamicConstants = DEFAULT_CONSTANTS,
                       out::Union{Nothing,AbstractMatrix} = nothing)
    Nx, Ny = size(Ts)
    H = out === nothing ? zeros(Float64, Nx, Ny) : fill!(out, 0)
    # one work item per column (KA-m, #301): a column touches only its own
    # air cell, reservoir and H
    foreach_column!(backend(Ts), _dew_exchange_column!, (1:Nx, 1:Ny), qv, q_dew, Ts, H,
                    p, dt, dz, ρ, C_e, U, c; weight = 16)
    return H
end

@inline function _dew_exchange_column!(i, j, qv, q_dew, Ts, H, p, dt, dz, ρ, C_e, U, c)
    @inbounds begin
        q = ndims(qv) == 3 ? qv[i, j, 1] : qv[i, j]
        r = dew_deposition(q, Ts[i, j], p, dt; dz = dz, ρ = ρ, C_e = C_e,
                           U = U, dew = q_dew[i, j], c = c)
        r.mass == 0 && return nothing
        if ndims(qv) == 3
            qv[i, j, 1] = q + r.Δqv
        else
            qv[i, j] = q + r.Δqv
        end
        q_dew[i, j] = max(q_dew[i, j] + r.mass, 0.0)
        H[i, j] = r.surface_heat / dt
    end
    return nothing
end

"""
    dew_exchange_columns!(H, qv, q_dew, Ts, mask, k_air, p, ρ, dz, dt) -> H

[`dew_deposition`](@ref) between the air cell `k_air[i, j]` of each
column with `mask[i, j]` (e.g. the first air cell above a roof, and
only over surfaces without a vapour exchange of their own) and the dew
reservoir `q_dew` (kg m⁻², updated in place) at the surface temperature
`Ts`, at that cell's pressure `p[k]` and density `ρ[k]` (per-level
vectors). `qv` is the 3-D vapour array; its increment is applied as
`qv += (q + Δqv) − q`. `H` (W m⁻², overwritten; 0 where nothing forms or
evaporates) is the latent heat released at the surface, positive when
dew forms: it belongs in the surface energy budget, not the air's. The
columns are independent work items, so the result is the same at any
thread count. Moved from the urban case's `fog_dew` coupling (ONE-e,
#454); [`dew_exchange!`](@ref) is the form at level 1 with one pressure.
"""
function dew_exchange_columns!(H::AbstractMatrix, qv::AbstractArray{<:Real,3}, q_dew::AbstractMatrix,
                               Ts::AbstractMatrix, mask::AbstractMatrix{Bool}, k_air::AbstractMatrix{<:Integer},
                               p::AbstractVector, ρ::AbstractVector, dz::Real, dt::Real)
    Nx, Ny = size(H)
    foreach_column!(backend(H), _dew_column!, (1:Nx, 1:Ny), H, qv,
                    mask, k_air, Ts, q_dew, p, ρ, dz, dt; weight = DEW_WEIGHT)
    return H
end

# the work of one dew column in grid-point equivalents (one saturation
# humidity and a few divisions: about the cost of 16 pointwise updates)
const DEW_WEIGHT = 16

@inline function _dew_column!(i, j, H_dew, qv, dew_mask, k_air, Ts, dew, p̄, ρ̄, dz, dt)
    @inbounds begin
        H_dew[i, j] = 0.0
        dew_mask[i, j] || return nothing
        k = k_air[i, j]
        q = qv[i, j, k]
        r = dew_deposition(q, Ts[i, j], p̄[k], dt; dz = dz, ρ = ρ̄[k], dew = dew[i, j])
        r.mass == 0 && return nothing
        qv[i, j, k] += (q + r.Δqv) - q
        dew[i, j] = max(dew[i, j] + r.mass, 0.0)
        H_dew[i, j] = r.surface_heat / dt
    end
    return nothing
end

include("radiation_fog/coupling.jl")
include("radiation_fog/column.jl")

end # module
