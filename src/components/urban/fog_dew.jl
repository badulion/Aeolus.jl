# Dew on the surfaces and fog in the air (Aeolus.RadiationFog, Phase 44).
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
# The physics lives in `src/RadiationFog.jl` (one source of truth, with
# MoistThermodynamics saturation and latent heats): air-side fog is an
# enthalpy-conserving saturation adjustment into `ql`, and dew is an
# implicit bulk exchange whose latent heat goes to the SURFACE
# (`dew_exchange_columns!`, moved there from this file in ONE-e #454).
# This file only adapts the case's fields to it.
#
# Pressure and density come from the case's hydrostatic reference state
# (site/reference_state.jl, #131): fog is adjusted at p̄_k (T = θ Π̄_k),
# and dew exchanges at the p̄_k and ρ̄_k of its air cell, the density the
# ledger integrates the water with. Before #131 fog and dew used an
# isothermal 8 km scale height from 1013.25 hPa (0.17 % off the
# hydrostatic p̄ at the flagship's top level) and ρ = 1.
#
# Since #129 the dew exchanges with the first AIR cell of each column
# (`bus.k_air`: above the roof on building columns, not inside the
# building, at that cell's pressure), and only over surfaces with no vapour exchange of their own
# (`bus.dew_mask`: moisture availability 0, e.g. roofs and glass). Over a
# soil column or water the surface exchange's Monin–Obukhov evaporation
# already turns negative when the air is above saturation at the skin,
# and that dew goes into the soil water; a second dew path there would
# count the same deposition twice.

"""
    FogDew

Additive coupling: every step in `:microphysics` (after the microphysics provider), dew (`RadiationFog.dew_exchange!`)
between the first air cell of each `bus.dew_mask` column (`bus.k_air`)
and the dew reservoir `bus.dew` at the surface temperature `bus.Ts`, then
fog (`RadiationFog.fog_adjust!`) on θ, qv, ql. The dew's latent heat warms the surface
(`bus.sensible -= H_dew`, delivered by the surface exchange). Fog is
adjusted at the reference pressure `p̄_k`, dew at the `p̄_k` and `ρ̄_k` of
its air cell (`case.ref`).
"""
mutable struct FogDew <: UrbanComponent
    H_dew::Matrix{Float64}    # this step's dew latent heat into the surface (W/m²)
    dew_start::Matrix{Float64}  # dew reservoir before this step's exchange (kg/m²)
end

component_name(::Type{FogDew}) = :fog_dew
modules_used(::Type{FogDew}) = (:RadiationFog, :MoistThermodynamics)
stages(::Type{FogDew}) = ((:microphysics, 20, EVERY_STEP),)

build(::Type{FogDew}, case, o) =
    FogDew(zeros(case.mesh.Nx, case.mesh.Ny), zeros(case.mesh.Nx, case.mesh.Ny))

function run_stage!(c::FogDew, case, ::Val{:microphysics}, dt)
    bus = case.bus
    copyto!(c.dew_start, bus.dew)
    qv = interior_view(tracer(case, :qv))
    # dew: on the first air cell of the columns in dew_mask, at the pressure
    # and density of that cell (roof columns exchange tens of metres up).
    # One pass over the columns, each with RadiationFog.dew_deposition at
    # its own k_air (#295). Before, every distinct air level copied two 2-D
    # slices and ran a full-surface dew_exchange! on them (about 2.4 of the
    # stage's 8.5 ms/step on the flagship; the rest is the fog adjustment);
    # the arithmetic per column is the same, bit for bit:
    # the vapour increment is still applied as qv += (q + Δqv) − q.
    Aeolus.RadiationFog.dew_exchange_columns!(c.H_dew, qv, bus.dew, bus.Ts, bus.dew_mask, bus.k_air,
                                              case.ref.p̄, case.ref.ρ̄, case.mesh.dz, dt)
    # fog: saturation adjustment in the air
    Aeolus.RadiationFog.fog_adjust!(interior_view(case.sim.θ), qv,
                                    interior_view(tracer(case, :ql)), ref_pressure(case), dt)
    @. bus.sensible -= c.H_dew
    return nothing
end

# Ledger: the dew's latent heat is an extra flux into the surface; the
# vapour deposited into the dew reservoir leaves the air (dew = the
# reservoir change, negative = deposition, positive = re-evaporation).
function budget!(c::FogDew, case, l)
    add_surface!(l, :dew_heat, c.H_dew, case.geom.sfc, l.dt)
    add_water!(l, :dew, -(sum(case.bus.dew) - sum(c.dew_start)) / length(c.dew_start))
    return nothing
end

function smoke_check(c::FogDew, case)
    Δdew = case.bus.dew .- c.dew_start
    ql = interior_view(tracer(case, :ql))
    return ["dew reservoir ≥ 0" => all(>=(0), case.bus.dew),
            "cloud water ≥ 0 (to round-off)" => minimum(ql) >= -1e-15,
            # deposition releases latent heat into the surface,
            # re-evaporation takes it: H_dew and Δdew have the same sign
            "dew latent heat has the sign of the deposition" =>
                all(I -> c.H_dew[I] * Δdew[I] >= 0, eachindex(Δdew)),
            "H_dew = L_v(Ts) · deposition rate (rel. 1e-9)" =>
                all(I -> abs(c.H_dew[I] - Aeolus.MoistThermodynamics.latent_heat_vapor(case.budgets.Ts_start[I]) *
                                          Δdew[I] / case.budgets.dt) <=
                         1e-9 * abs(c.H_dew[I]) + 1e-12, eachindex(Δdew))]
end

