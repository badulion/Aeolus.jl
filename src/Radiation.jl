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
using ..RadiationCore: STEFAN_BOLTZMANN
using ..SurfaceExchange: AbstractSurface, SurfaceForcing, SurfaceFluxes,
                         turbulent_fluxes
# `surface_temperature` is the SurfaceExchange generic (one function for
# every surface model); SlabLand adds a method and this module re-exports it.
import ..SurfaceExchange: surface_temperature, roughness_lengths,
                          surface_albedo, surface_emissivity, step_surface!

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
    SlabLand(; T = 290.0, C_s = 2e5, albedo = 0.2, emissivity = 1.0,
               z0m = 0.1, z0h = z0m / 10)

A single-layer soil-temperature reservoir governed by

    C_s dT/dt = (1 - α) SW↓ + ε LW↓ - ε σT⁴ - H - LE

where `α` is the albedo, `ε` the emissivity, `σ` the Stefan-Boltzmann
constant (`RadiationCore.STEFAN_BOLTZMANN`), `H` the upward sensible
heat flux to the atmosphere (typically `ρ cₚ w'θ'|₀`), `LE` the latent
heat flux, and `C_s` [J/m²/K] the effective heat capacity.

`SlabLand` is the first implementer of the Era V
`SurfaceExchange.AbstractSurface` interface: it is dry
(`moisture_availability = 0`, so `LE = 0`) and has roughness lengths
`z0m`, `z0h` (= `z0q`), so `step_surface!(land, forcing, dt)` computes
its own Monin–Obukhov fluxes. `step_land!` remains for callers that
supply `H` themselves.

This is deliberately the simplest useful LSM — vegetation, soil
moisture, and multi-layer conduction are future refinements.
"""
mutable struct SlabLand{T<:AbstractFloat} <: AbstractSurface
    T::T           # surface (skin) temperature [K]
    C_s::T         # areal heat capacity        [J/m²/K]
    albedo::T
    emissivity::T
    z0m::T         # momentum roughness length  [m]
    z0h::T         # heat / moisture roughness length [m]
end

SlabLand(; T::Real = 290.0, C_s::Real = 2e5, albedo::Real = 0.2,
         emissivity::Real = 1.0, z0m::Real = 0.1, z0h::Real = z0m / 10) =
    SlabLand{Float64}(Float64(T), Float64(C_s), Float64(albedo),
                      Float64(emissivity), Float64(z0m), Float64(z0h))

surface_temperature(l::SlabLand) = l.T
roughness_lengths(l::SlabLand) = (z0m = l.z0m, z0h = l.z0h, z0q = l.z0h)
surface_albedo(l::SlabLand) = l.albedo
surface_emissivity(l::SlabLand) = l.emissivity

"""
    step_land!(land, sw_in, lw_in, sensible_flux, dt)

Advance `land.T` by `dt` seconds under the balance above with a
prescribed `sensible_flux` and no latent flux. Positive `sensible_flux`
means heat flows *from* the surface *to* the air (cooling the ground).
"""
function step_land!(l::SlabLand, sw_in::Real, lw_in::Real,
                    sensible_flux::Real, dt::Real)
    net = (1 - l.albedo) * sw_in + l.emissivity * lw_in -
          l.emissivity * STEFAN_BOLTZMANN * l.T^4 - sensible_flux
    l.T += net * dt / l.C_s
    return l
end

"""
    step_surface!(land::SlabLand, forcing, dt; kwargs...) -> SurfaceFluxes

Forward-Euler step of the slab under `SurfaceExchange.turbulent_fluxes`
(Monin–Obukhov H, LE from the current skin temperature):
`C_s ΔT = G dt` with `G = Rn - H - LE`. `kwargs` go to `turbulent_fluxes`.
"""
function step_surface!(l::SlabLand, f::SurfaceForcing, dt::Real; kwargs...)
    fl = turbulent_fluxes(l, f; kwargs...)
    l.T += fl.G * dt / l.C_s
    return fl
end

end # module
