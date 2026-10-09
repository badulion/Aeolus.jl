# The Rayleigh sponge below the top (ONE-a, #450).

"""
    Sponge(α_max, z_start, target)

Rayleigh damping `−α(z)(u − u_target)` of `u, v, w` above `z_start`, with
`α(z) = α_max sin²(π/2 (z − z_start)/(z_top − z_start))`
(`Simulations.add_velocity_sponge!` → `Operators.add_sponge!`; Klemp &
Lilly 1978), in stage 7 (`:body_force`). `[sponge] alpha_max` (s⁻¹) and
`z_start` (m) configure it; it is enabled when `α_max ≠ 0` and `z_start`
is set (the model's configuration requires it below the top). The target
of a closed top is rest (`Simulations.sponge_target`).
"""
struct Sponge{G<:NamedTuple} <: AbstractComponent
    α_max::Float64
    z_start::Float64
    target::G
end

component_name(::Type{<:Sponge}) = :sponge
section(::Type{<:Sponge}) = "sponge"
schema(::Type{<:Sponge}) =
    Config.section("sponge"; doc = "Rayleigh damping of the velocity below the top (Klemp & Lilly 1978).") do s
        Config.option!(s, "alpha_max", Float64; default = 0.0, range = (0, Inf), units = "s^-1",
                       doc = "Rayleigh-damping rate at the top; 0: no sponge")
        Config.option!(s, "z_start", Config.Optional{Float64}; units = "m",
                       doc = "height where the sponge starts, below the top; \"none\": no sponge")
    end
enabled(::Type{<:Sponge}, opts) = opts.alpha_max != 0 && opts.z_start !== nothing
function build(::Type{<:Sponge}, opts, domain)
    s = domain.state
    z_top = grid(domain).z0 + grid(domain).Lz
    opts.z_start < z_top ||
        throw(ArgumentError("Sponge: z_start = $(opts.z_start) m is not below the top ($z_top m)"))
    return Sponge(opts.alpha_max, Float64(opts.z_start),
                  Simulations._sponge_target(nothing, s.mesh, s.bcs, true))
end
stages(::Sponge) = (Stage(:body_force),)

function run_stage!(c::Sponge, d, ::Val{:body_force}, dt)
    add_velocity_sponge!(d.state, c.α_max, c.z_start, c.target)
    return nothing
end
