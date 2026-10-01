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
struct MicroKessler <: AbstractCoupling
    qv_max::Float64
end

coupling_name(::Type{MicroKessler}) = :micro_kessler
slot(::Type{MicroKessler}) = :microphysics
provider(::Type{MicroKessler}) = :kessler
modules_used(::Type{MicroKessler}) = (:Microphysics,)
default_options(::Type{MicroKessler}) = (qv_max = 0.03,)
stages(::Type{MicroKessler}) = ((:microphysics, 10, EVERY_STEP),)
build(::Type{MicroKessler}, case, o) = MicroKessler(o.qv_max)

function run_stage!(c::MicroKessler, case, ::Val{:microphysics}, dt)
    qv = tracer(case, :qv)
    apply_kessler_step!(qv, tracer(case, :ql), tracer(case, :qr), dt; Vt = :spectrum)
    clamp!(interior_view(qv), 0.0, c.qv_max)
    return nothing
end

register!(MicroKessler)
