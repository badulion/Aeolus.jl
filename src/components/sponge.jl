# The Rayleigh sponge below the top (ONE-a, #450).

"""
    Sponge(α_max, z_start, target)

Rayleigh damping `−α(z)(u − u_target)` of `u, v, w` above `z_start`, with
`α(z) = α_max sin²(π/2 (z − z_start)/(z_top − z_start))`
(`Simulations.add_velocity_sponge!` → `Operators.add_sponge!`; Klemp &
Lilly 1978), in stage 7 (`:body_force`). `[dynamics] sponge_alpha_max`
(s⁻¹) and `sponge_z_start` (m) configure it; it is enabled when
`α_max ≠ 0` and `z_start` lies below the top. The target of a closed top
is rest (`Simulations.sponge_target`).
"""
struct Sponge{G<:NamedTuple} <: AbstractComponent
    α_max::Float64
    z_start::Float64
    target::G
end

component_name(::Type{<:Sponge}) = :sponge
section(::Type{<:Sponge}) = "dynamics"
schema(::Type{<:Sponge}) =
    Config.section("dynamics") do s
        Config.option!(s, "sponge_alpha_max", Float64; default = 0.0, range = (0, Inf), units = "s^-1",
                       doc = "Rayleigh-damping rate at the top; 0: no sponge")
        Config.option!(s, "sponge_z_start", Config.Optional{Float64}; units = "m",
                       doc = "height where the sponge starts; \"none\": the top (no sponge)")
    end
_z_top(domain) = grid(domain).z0 + grid(domain).Lz
enabled(::Type{<:Sponge}, opts, domain) =
    opts.sponge_alpha_max != 0 && opts.sponge_z_start !== nothing && opts.sponge_z_start < _z_top(domain)
function build(::Type{<:Sponge}, opts, domain)
    s = domain.state
    return Sponge(opts.sponge_alpha_max, Float64(opts.sponge_z_start),
                  Simulations._sponge_target(nothing, s.mesh, s.bcs, true))
end
stages(::Sponge) = (Stage(:body_force),)

function run_stage!(c::Sponge, d, ::Val{:body_force}, dt)
    add_velocity_sponge!(d.state, c.α_max, c.z_start, c.target)
    return nothing
end
