# The momentum-advection operator of `step!`, in the `:momentum_advection`
# slot, through the `momentum_advection` keyword of `step!`
# (`step_options`).
#
#   :centered (default) Operators.momentum_advect! — 2nd-order centred
#             flux form, energy-conserving in space.
#   :weno_z   AdvectionSchemes.momentum_advect_weno_z! — WENO-Z
#             reconstruction of the transported component (order 5, halo
#             3; order 7 needs halo 4): upwind-biased, dissipative, robust
#             at the sharp shear layers of the roof edges.

"""
    MomentumCentered

Slot `:momentum_advection`, provider `:centered` (default): the built-in
`momentum_advect!`.
"""
struct MomentumCentered <: AbstractCoupling end
coupling_name(::Type{MomentumCentered}) = :momentum_centered
slot(::Type{MomentumCentered}) = :momentum_advection
provider(::Type{MomentumCentered}) = :centered
modules_used(::Type{MomentumCentered}) = (:Operators,)
step_options(c::MomentumCentered, case) = (momentum_advection = momentum_advect!,)

"""
    MomentumWENOZ

Slot `:momentum_advection`, provider `:weno_z`:
`momentum_advect_weno_z!(…; order)` (option `order = 5`).
"""
struct MomentumWENOZ <: AbstractCoupling
    order::Int
end
coupling_name(::Type{MomentumWENOZ}) = :momentum_weno_z
slot(::Type{MomentumWENOZ}) = :momentum_advection
provider(::Type{MomentumWENOZ}) = :weno_z
is_default(::Type{MomentumWENOZ}) = false
modules_used(::Type{MomentumWENOZ}) = (:AdvectionSchemes,)
default_options(::Type{MomentumWENOZ}) = (order = 5,)
function build(::Type{MomentumWENOZ}, case, o)
    m = case.mesh
    H = o.order == 5 ? 3 : o.order == 7 ? 4 : error("momentum_weno_z: order must be 5 or 7")
    min(m.Hx, m.Hy, m.Hz) >= H || error("momentum_weno_z: order $(o.order) needs halo ≥ $H")
    return MomentumWENOZ(o.order)
end
function step_options(c::MomentumWENOZ, case)
    order = c.order
    return (momentum_advection = (tu, tv, tw, u, v, w) ->
                Aeolus.AdvectionSchemes.momentum_advect_weno_z!(tu, tv, tw, u, v, w; order = order),)
end
# The flux form telescopes: on the periodic x, y directions and with w = 0
# at the walls, the domain sums of the u and v tendencies vanish (measured
# ~1e-18 relative in the :smoke window, the centred operator the same).
# w is not conserved by the operator alone (the walls take momentum).
function smoke_check(c::MomentumWENOZ, case)
    sim, m = case.sim, case.mesh
    for f in (sim.u, sim.v, sim.w)
        fill_halos!(f, sim.bcs)
    end
    tu, tv, tw = Field(FaceX(), m), Field(FaceY(), m), Field(FaceZ(), m)
    Aeolus.AdvectionSchemes.momentum_advect_weno_z!(tu, tv, tw, sim.u, sim.v, sim.w; order = c.order)
    rel(t) = (x = interior_view(t); abs(sum(x)) / max(sum(abs, x), eps()))
    return ["projected velocity divergence-free (max|∇·u| < 1e-10 s⁻¹)" => max_divergence(sim) < 1e-10,
            "WENO-Z flux form conserves horizontal momentum (|Σtu|, |Σtv| < 1e-12 Σ|·|)" =>
                rel(tu) < 1e-12 && rel(tv) < 1e-12]
end

register!(MomentumCentered)
register!(MomentumWENOZ)
