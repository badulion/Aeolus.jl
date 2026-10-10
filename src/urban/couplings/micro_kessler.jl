# Warm-rain microphysics: Kessler autoconversion / accretion /
# sedimentation of Aeolus.Microphysics on (qv, ql, qr).
#
# KNOWN (the microphysics-slot issue): qv is clamped to [0, 0.03] after
# the step, and qr is sedimented but not advected.

"""
    MicroKessler

Slot `:microphysics`, provider `:kessler` (default): every step
`apply_kessler_step!(qv, ql, qr, dt; Vt = :spectrum)`, then the qv clamp.
"""
mutable struct MicroKessler <: AbstractCoupling
    qv_max::Float64
    precip::Float64   # this step: water that fell out of the bottom (kg/m² of domain, ≥ 0)
    clip::Float64     # this step: vapour removed by the qv clamp (kg/m² of domain, ≥ 0)
end

coupling_name(::Type{MicroKessler}) = :micro_kessler
slot(::Type{MicroKessler}) = :microphysics
provider(::Type{MicroKessler}) = :kessler
modules_used(::Type{MicroKessler}) = (:Microphysics,)
default_options(::Type{MicroKessler}) = (qv_max = 0.03,)
stages(::Type{MicroKessler}) = ((:microphysics, 10, EVERY_STEP),)
build(::Type{MicroKessler}, case, o) = MicroKessler(o.qv_max, 0.0, 0.0)

function run_stage!(c::MicroKessler, case, ::Val{:microphysics}, dt)
    qv, ql, qr = tracer(case, :qv), tracer(case, :ql), tracer(case, :qr)
    # Kessler conserves ql + qr except for the rain that sediments out of
    # the bottom, so the loss of ql + qr over the step is the surface
    # precipitation (ρ = 1, the density the scheme is called with).
    Wc0 = water_content(case.budgets, case.mesh, interior_view(ql)) + water_content(case.budgets, case.mesh, interior_view(qr))
    apply_kessler_step!(qv, ql, qr, dt; Vt = :spectrum)
    c.precip = Wc0 - water_content(case.budgets, case.mesh, interior_view(ql)) - water_content(case.budgets, case.mesh, interior_view(qr))
    # the clamp, booking what it removes (same values as clamp!)
    c.clip = Aeolus.Microphysics.clamp_booked!(interior_view(qv), 0.0, c.qv_max, case.budgets.ρ) *
             case.mesh.dz / (case.mesh.Nx * case.mesh.Ny)
    return nothing
end

function budget!(c::MicroKessler, case, l)
    add_water!(l, :precipitation, -c.precip)
    add_water!(l, :qv_clamp, -c.clip)
    return nothing
end

smoke_check(c::MicroKessler, case) =
    ["rain ≥ 0" => minimum(interior_view(tracer(case, :qr))) >= 0,
     "qv ≤ qv_max = $(c.qv_max)" => maximum(interior_view(tracer(case, :qv))) <= c.qv_max,
     # sedimentation only moves rain down and out of the bottom: the
     # column water cannot grow (round-off of ~1e-16 relative allowed)
     "precipitation ≥ 0" => c.precip >= -1e-15]

register!(MicroKessler)
