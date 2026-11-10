# Buildings as immersed boundaries: the implicit direct-forcing IBM
# correction of Aeolus.IBM on the OSM solid mask, after the predictor.

"""
    IBMCoupling

The buildings as an immersed boundary: `Aeolus.IBM.implicit_ibm_correction!`
with `τ = 0` (velocity forced to zero inside the solid) in the
`:post_predictor` hook of `step!`. Slot `:immersed_boundary`, provider
`:direct_forcing` (a conservative cut-cell provider is the solver issue S2).
"""
struct IBMCoupling{SM} <: UrbanComponent
    sm::SM
end

component_name(::Type{<:IBMCoupling}) = :ibm
slot(::Type{<:IBMCoupling}) = :immersed_boundary
provider(::Type{<:IBMCoupling}) = :direct_forcing
modules_used(::Type{<:IBMCoupling}) = (:IBM,)
stages(::Type{<:IBMCoupling}) = ((:post_predictor, 10, EVERY_STEP),)
Components.budget_writes(::IBMCoupling, ::Val{:post_predictor}) = (:momentum_x, :momentum_y)   # the velocity (#539)
build(::Type{<:IBMCoupling}, case, opts) = IBMCoupling(case.geom.sm)

run_stage!(c::IBMCoupling, case, ::Val{:post_predictor}, dt) =
    implicit_ibm_correction!(case.sim, c.sm; τ = 0.0, dt = dt)

# The direct forcing zeroes the predicted velocity inside the solid; the
# projection then adds only the pressure-gradient correction there, which
# must stay small against the flow outside.
function smoke_check(c::IBMCoupling, case)
    m = case.mesh
    mask = view(c.sm.mask, m.Hx+1:m.Hx+m.Nx, m.Hy+1:m.Hy+m.Ny, m.Hz+1:m.Hz+m.Nz)
    any(mask) || return ["solid cells in the window" => false]
    U = sqrt.(interior_view(case.sim.u) .^ 2 .+ interior_view(case.sim.v) .^ 2)
    u_in, u_out = maximum(U[mask]), maximum(U[.!mask])
    return ["solid cells in the window" => true,
            "|u_h| inside the buildings < 0.3 × max outside" => u_in < 0.3 * u_out]
end

