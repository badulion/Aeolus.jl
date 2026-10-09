# The Coriolis term (ONE-a, #450): f-plane rotation of the horizontal momentum.

"""
    Coriolis(f)

The f-plane Coriolis term `(+f v, −f u)` on the horizontal momentum
tendencies (`Operators.add_coriolis!`, the core's operator), in stage 6
(`:tendencies`) of every (sub)step. `f` is `[coriolis] f` (s⁻¹), or
`2Ω sin φ` of `[coriolis] latitude` (degrees, `Operators.coriolis_parameter`);
`f = 0` without a latitude enables no component. On an adaptive step it limits the step
to `C_coriolis/|f|` ([`limit_dt`](@ref), label `:coriolis`). The term does
no work: it rotates `(u, v)` at the inertial frequency `|f|` and keeps
the horizontal kinetic energy.
"""
struct Coriolis <: AbstractComponent
    f::Float64
end

component_name(::Type{Coriolis}) = :coriolis
section(::Type{Coriolis}) = "coriolis"
schema(::Type{Coriolis}) =
    Config.section("coriolis"; doc = "f-plane rotation of the horizontal momentum.") do s
        Config.option!(s, "f", Float64; default = 0.0, units = "s^-1",
                       doc = "Coriolis parameter; 0 (and no latitude): off")
        Config.option!(s, "latitude", Config.Optional{Float64}; range = (-90, 90), units = "degree",
                       doc = "the latitude φ instead of f: f = 2Ω sin φ (Ω = 7.2921e-5 s⁻¹); \"none\": f")
    end
"The Coriolis parameter of resolved `[coriolis]` options: `f`, or `2Ω sin φ` of `latitude`."
coriolis_f(opts) = opts.latitude === nothing ? Float64(opts.f) : coriolis_parameter(opts.latitude)
enabled(::Type{Coriolis}, opts) = coriolis_f(opts) != 0
build(::Type{Coriolis}, opts, domain) = Coriolis(coriolis_f(opts))
stages(::Coriolis) = (Stage(:tendencies),)

function run_stage!(c::Coriolis, d, ::Val{:tendencies}, dt)
    s = d.state
    add_coriolis!(s.tu, s.tv, s.u, s.v, c.f)
    return nothing
end

limit_dt(c::Coriolis, d) = d.cfl === nothing ? nothing : (d.cfl.C_coriolis / abs(c.f), :coriolis)
