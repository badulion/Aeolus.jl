# The `:advection` slot (ONE-a, #450): the core's momentum advection.

"""
    CentredAdvection()

The provider of the `:advection` slot: the second-order centred flux form
of the momentum and scalar advection `−∂ⱼ(uⱼϕ)` on the staggered grid
(`Operators.momentum_advect!` and `advect_centered!` on the Boussinesq
core, run fused with the molecular diffusion, SC-d #404; the
`ρ̄`-weighted flux form `weighted_momentum_advect!` on the anelastic
core). The core calls it inside stage 6 of the pipeline; it attaches to no
stage itself. Every domain has an `:advection` provider: `[advection]
scheme = "centred"` (the default and only one).
"""
struct CentredAdvection <: AbstractComponent end

component_name(::Type{CentredAdvection}) = :advection
section(::Type{CentredAdvection}) = "advection"
schema(::Type{CentredAdvection}) =
    Config.slot_section("advection"; key = "scheme", doc = "The momentum and scalar advection.") do s
        Config.provider!(s, "centred"; default = [:model], doc = "The second-order centred flux form.")
    end
enabled(::Type{CentredAdvection}, opts) = opts.scheme === :centred
slot(::Type{CentredAdvection}) = :advection
build(::Type{CentredAdvection}, opts, domain) = CentredAdvection()

"""
    momentum_advection(c) -> Function

The momentum-advection operator `f(tu, tv, tw, u, v, w)` the Boussinesq
Euler step runs for the `:advection` component `c` (`step!`'s
`momentum_advection`; the default operator keeps the fused tendencies and
the hook-free fast path).
"""
momentum_advection(::CentredAdvection) = momentum_advect!
