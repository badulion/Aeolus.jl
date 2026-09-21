module Radiation

# Phase 19 — Simple radiation forcings and a single-layer land-surface
# model. Both are lightweight physics that couple to θ.
#
# Shipped:
#   * `apply_radiative_cooling!(tθ, cooling_rate)` — adds a prescribed
#     cooling-rate profile (K/s) to the potential-temperature tendency
#     at every cell. `cooling_rate` may be a scalar (uniform cooling)
#     or a length-Nz vector (profile).
#   * `SlabLand(T_soil, Cₛ, albedo)` — one-layer soil / surface model.
#     `step_land!(land, sw_in, lw_in, sensible_flux, dt)` advances its
#     temperature reservoir.
#
# Deferred: `RRTMGP.jl` two-stream radiation (Phase 19 refinement) and
# vegetation / canopy models (Phase 19 refinement).

using ..Grids: RegularGrid, interior
using ..Fields: Field

export apply_radiative_cooling!, SlabLand, step_land!, surface_temperature

# -------------------------------------------------------------------------
# Prescribed radiative cooling / heating
# -------------------------------------------------------------------------

"""
    apply_radiative_cooling!(tθ, rate)

Add a radiative-cooling tendency to `tθ` at every interior cell.

  * `rate::Real`               — uniform tendency (K/s) applied everywhere.
  * `rate::AbstractVector`     — length-`Nz` vertical profile (K/s at each
                                 interior z-level, from bottom up).

Signs: positive `rate` = warming, negative = cooling. The typical
free-atmosphere clear-sky longwave cooling in ABL LES is `≈ -2 K/day
= -2.3e-5 K/s`.
"""
function apply_radiative_cooling!(tθ::Field, rate::Real)
    _, _, kk = interior(tθ.mesh)
    r = eltype(tθ.data)(rate)
    for k in kk
        @views tθ.data[:, :, k] .+= r
    end
    return tθ
end

function apply_radiative_cooling!(tθ::Field, rate::AbstractVector)
    ii, jj, kk = interior(tθ.mesh)
    length(rate) == length(kk) ||
        error("apply_radiative_cooling!: rate profile length $(length(rate)) ≠ Nz $(length(kk))")
    for (idx, k) in enumerate(kk)
        r = eltype(tθ.data)(rate[idx])
        @views tθ.data[:, :, k] .+= r
    end
    return tθ
end

# -------------------------------------------------------------------------
# Single-layer (slab) land-surface model
# -------------------------------------------------------------------------

"""
    SlabLand(; T = 290.0, C_s = 2e5, albedo = 0.2)

A single-layer soil-temperature reservoir governed by

    C_s dT/dt = (1 - α) SW↓ + LW↓ - σT⁴ - H

where `α` is the albedo, `σ = 5.67e-8` the Stefan-Boltzmann constant,
`H` is the upward sensible heat flux to the atmosphere (typically
`ρ cₚ w'θ'|₀`), and `C_s` [J/m²/K] is the effective heat capacity.

This is deliberately the simplest useful LSM — vegetation, soil
moisture, and multi-layer conduction are future refinements.
"""
mutable struct SlabLand{T<:AbstractFloat}
    T::T           # surface (skin) temperature [K]
    C_s::T         # areal heat capacity        [J/m²/K]
    albedo::T
end

SlabLand(; T::Real = 290.0, C_s::Real = 2e5, albedo::Real = 0.2) =
    SlabLand{Float64}(Float64(T), Float64(C_s), Float64(albedo))

surface_temperature(l::SlabLand) = l.T

const _STEFAN_BOLTZMANN = 5.67e-8

"""
    step_land!(land, sw_in, lw_in, sensible_flux, dt)

Advance `land.T` by `dt` seconds under the balance above. Positive
`sensible_flux` means heat flows *from* the surface *to* the air
(cooling the ground).
"""
function step_land!(l::SlabLand, sw_in::Real, lw_in::Real,
                    sensible_flux::Real, dt::Real)
    net = (1 - l.albedo) * sw_in + lw_in -
          _STEFAN_BOLTZMANN * l.T^4 - sensible_flux
    l.T += net * dt / l.C_s
    return l
end

end # module
