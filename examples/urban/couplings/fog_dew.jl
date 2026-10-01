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
# implicit bulk exchange whose latent heat goes to the SURFACE. This file
# only adapts the case's fields to it.
#
# Pressure and density come from the case's hydrostatic reference state
# (core/reference_state.jl, #131): fog is adjusted at p̄_k (T = θ Π̄_k),
# dew exchanges with the surface at the ground pressure and the level-1
# density ρ̄_1, the density the ledger integrates the water with. Before
# #131 fog and dew used an isothermal 8 km scale height from 1013.25 hPa
# (0.17 % off the hydrostatic p̄ at the flagship's top level) and ρ = 1.
#
# KNOWN: dew goes into k = 1 of every column, inside the building on
# roof columns (#129).

"""
    apply_fog_dew!(θ, qv, ql, q_dew, Ts, dt; p = 1.01325e5, p_sfc, dz, ρ = 1.0,
                   C_e = 1e-3, U = 3.0, τ_fog = 0.0,
                   fog_levels = axes(θ, 3))
        -> H_dew (W m⁻², 2-D)

Dew onto the surface, then fog in the air.

  * **Dew** — `Aeolus.RadiationFog.dew_exchange!` on the lowest layer of
    `qv`: vapour is deposited into the 2-D reservoir `q_dew` (kg m⁻²)
    while `q_v > q_s(T_s)`, and re-evaporated from it when the surface
    warms. The latent heat is **returned** as `H_dew` (positive = heating
    the surface); the caller adds it to the surface energy budget. The
    air temperature is not touched by dew.
  * **Fog** — `Aeolus.RadiationFog.fog_adjust!` on `θ`, `qv`, `ql` in
    `fog_levels`: enthalpy-conserving saturation adjustment (relaxed over
    `τ_fog` if > 0), condensing into and evaporating from `ql`.

`p` is the pressure for fog, a scalar or a per-level vector; `p_sfc`
(default: `p`, or its first entry) the surface pressure for dew, and `ρ`
the density of the lowest air layer. Both steps are unconditionally stable in `dt`
and cannot overshoot saturation.
"""
function apply_fog_dew!(θ::AbstractArray{<:Real,3}, qv::AbstractArray{<:Real,3},
                        ql::AbstractArray{<:Real,3}, q_dew::AbstractMatrix,
                        Ts::AbstractMatrix, dt::Real;
                        p = 1.01325e5, p_sfc = (p isa AbstractVector ? first(p) : p),
                        dz::Real, ρ::Real = 1.0,
                        C_e::Real = 1e-3, U::Real = 3.0,
                        τ_fog::Real = 0.0, fog_levels = axes(θ, 3))
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

"""
    FogDew

Additive coupling: every step in `:fog_dew`, [`apply_fog_dew!`](@ref) on
θ, qv, ql with the dew reservoir `bus.dew` and the surface temperature
`bus.Ts`; the dew's latent heat warms the surface (`bus.sensible -= H_dew`).
Fog is adjusted at the reference pressure `p̄_k`, dew at the ground
pressure with the reference density `ρ̄_1` (`case.ref`).
"""
mutable struct FogDew <: AbstractCoupling
    H_dew::Matrix{Float64}    # this step's dew latent heat into the surface (W/m²)
    dew_start::Matrix{Float64}  # dew reservoir before this step's exchange (kg/m²)
end

coupling_name(::Type{FogDew}) = :fog_dew
modules_used(::Type{FogDew}) = (:RadiationFog, :MoistThermodynamics)
stages(::Type{FogDew}) = ((:fog_dew, 10, EVERY_STEP),)

build(::Type{FogDew}, case, o) =
    FogDew(zeros(case.mesh.Nx, case.mesh.Ny), zeros(case.mesh.Nx, case.mesh.Ny))

function run_stage!(c::FogDew, case, ::Val{:fog_dew}, dt)
    bus = case.bus
    copyto!(c.dew_start, bus.dew)
    H_dew = apply_fog_dew!(case.sim.θ, tracer(case, :qv), tracer(case, :ql), bus.dew, bus.Ts, dt;
                           p = ref_pressure(case), p_sfc = surface_pressure(case),
                           dz = case.mesh.dz, ρ = ref_density(case, 1))
    @. bus.sensible -= H_dew
    c.H_dew = H_dew
    return nothing
end

# Ledger: the dew's latent heat is an extra flux into the surface; the
# vapour deposited into the dew reservoir leaves the air (dew = the
# reservoir change, negative = deposition, positive = re-evaporation).
function budget!(c::FogDew, case, l)
    add_surface!(l, :dew_heat, c.H_dew, case.geom.sfc, case.cfg.dt)
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
                all(I -> abs(c.H_dew[I] - Aeolus.MoistThermodynamics.latent_heat_vapor(case.ledger.Ts_start[I]) *
                                          Δdew[I] / case.cfg.dt) <=
                         1e-9 * abs(c.H_dew[I]) + 1e-12, eachindex(Δdew))]
end

register!(FogDew)
