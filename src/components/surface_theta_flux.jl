# A prescribed kinematic θ flux through the bottom and the top (ONE-a, #450).

"""
    SurfaceThetaFlux(bottom, top)

Prescribed kinematic θ fluxes `w'θ'` (K m s⁻¹) through the bottom and the
top of the box (`[surface_theta_flux] bottom`, `top`; both `0` enable no
component): `Operators.add_surface_flux!` adds `F/Δz` to
the θ tendency of the first and last level, in stage 6 (`:tendencies`).
In flux form the domain-integrated θ changes by `(F_bottom − F_top) A` per
unit time, exactly. On a sound-proof core the same tendency is the
`W`-weighted flux form (the wall faces of its base state carry the weight
of the adjacent cell): `Σ W θ ΔV` changes by `(W_bot F_bottom − W_top
F_top) A`, `W = ρ̄` on the anelastic core (ONE-j, #460).
"""
struct SurfaceThetaFlux <: AbstractComponent
    bottom::Float64
    top::Float64
end

component_name(::Type{SurfaceThetaFlux}) = :surface_theta_flux
section(::Type{SurfaceThetaFlux}) = "surface_theta_flux"
schema(::Type{SurfaceThetaFlux}) =
    Config.section("surface_theta_flux"; doc = "Prescribed kinematic θ fluxes through the bottom and the top (z walls).") do s
        Config.option!(s, "bottom", Float64; default = 0.0, units = "K m s^-1",
                       doc = "prescribed kinematic θ flux w'θ' through the bottom; 0: none")
        Config.option!(s, "top", Float64; default = 0.0, units = "K m s^-1",
                       doc = "prescribed kinematic θ flux through the top; 0: none")
    end
enabled(::Type{SurfaceThetaFlux}, opts) = opts.bottom != 0 || opts.top != 0
build(::Type{SurfaceThetaFlux}, opts, domain) = SurfaceThetaFlux(opts.bottom, opts.top)
stages(::SurfaceThetaFlux) = (Stage(:tendencies),)

function run_stage!(c::SurfaceThetaFlux, d, ::Val{:tendencies}, dt)
    add_surface_flux!(d.state.tθ, c.bottom, c.top)
    return nothing
end

wall_theta_flux(c::SurfaceThetaFlux, d) = (c.bottom, c.top)

# the fused sweep (#533): the flux of the first and last level, pointwise
FiniteVolume.sweep_term(c::SurfaceThetaFlux, d) = FiniteVolume.SurfaceFluxTerm(c.bottom, c.top, d.state.mesh)
