# Buildings as immersed boundaries: the implicit direct-forcing IBM
# correction of Aeolus.IBM on the OSM solid mask, after the predictor.

"""
    IBMCoupling

The buildings as an immersed boundary: `Aeolus.IBM.implicit_ibm_correction!`
with `τ = 0` (velocity forced to zero inside the solid) in the
`:post_predictor` hook of `step!`. Slot `:immersed_boundary`, provider
`:direct_forcing` (a conservative cut-cell provider is the solver issue S2).
"""
struct IBMCoupling{SM} <: AbstractCoupling
    sm::SM
end

coupling_name(::Type{<:IBMCoupling}) = :ibm
slot(::Type{<:IBMCoupling}) = :immersed_boundary
provider(::Type{<:IBMCoupling}) = :direct_forcing
modules_used(::Type{<:IBMCoupling}) = (:IBM,)
stages(::Type{<:IBMCoupling}) = ((:post_predictor, 10, EVERY_STEP),)
build(::Type{<:IBMCoupling}, case, opts) = IBMCoupling(case.geom.sm)

run_stage!(c::IBMCoupling, case, ::Val{:post_predictor}, dt) =
    implicit_ibm_correction!(case.sim, c.sm; τ = 0.0, dt = dt)

register!(IBMCoupling)
