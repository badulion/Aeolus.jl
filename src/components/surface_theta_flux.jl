# A prescribed kinematic θ flux through the bottom and the top (ONE-a, #450).

"""
    SurfaceThetaFlux(bottom, top)

Prescribed kinematic θ fluxes `w'θ'` (K m s⁻¹) through the bottom and the
top of the box (`[boundaries] theta_flux_bottom`, `theta_flux_top`; both
`0` enable no component): `Operators.add_surface_flux!` adds `F/Δz` to
the θ tendency of the first and last level, in stage 6 (`:tendencies`).
In flux form the domain-integrated θ changes by `(F_bottom − F_top) A` per
unit time, exactly. On a sound-proof core the flux carries the mass of the
wall face (`Operators.flux_weights`, ONE-j #460): `Σ W θ ΔV` changes by
`(W_f,bot F_bottom − W_f,top F_top) A`, `W = ρ̄` on the anelastic core.
"""
struct SurfaceThetaFlux <: AbstractComponent
    bottom::Float64
    top::Float64
end

component_name(::Type{SurfaceThetaFlux}) = :surface_theta_flux
section(::Type{SurfaceThetaFlux}) = "boundaries"
schema(::Type{SurfaceThetaFlux}) =
    Config.section("boundaries") do s
        Config.option!(s, "theta_flux_bottom", Float64; default = 0.0, units = "K m s^-1",
                       doc = "prescribed kinematic θ flux w'θ' through the bottom (z walls)")
        Config.option!(s, "theta_flux_top", Float64; default = 0.0, units = "K m s^-1",
                       doc = "prescribed kinematic θ flux through the top (z walls)")
    end
enabled(::Type{SurfaceThetaFlux}, opts, domain) = opts.theta_flux_bottom != 0 || opts.theta_flux_top != 0
build(::Type{SurfaceThetaFlux}, opts, domain) = SurfaceThetaFlux(opts.theta_flux_bottom, opts.theta_flux_top)
stages(::SurfaceThetaFlux) = (Stage(:tendencies),)

function run_stage!(c::SurfaceThetaFlux, d, ::Val{:tendencies}, dt)
    add_surface_flux!(d.state.tθ, c.bottom, c.top; weights = flux_weights(d.state))
    return nothing
end
