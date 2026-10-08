# The tracer registry.
#
# Every 3-D scalar the case carries besides the dynamical core's own
# fields (u, v, w, θ, p) is a registered tracer. A coupling that needs one
# calls `ensure_tracer!` in its `build`/`init!`; the `:transport` stage
# advects every tracer flagged `advect = true` with the scheme in the
# `scalar_advection` slot, so a new microphysics can add its species
# without touching transport code.

"""
    Tracer

A registered cell-centred scalar: `name`, `field`, and the flags
`advect` (moved by the `:transport` stage), `positive` (kept ≥ 0 by the
transport scheme) and `output` (written to the NetCDF file with `units`
and `long_name`). `diffusivity` is the eddy diffusivity the `:transport`
stage diffuses it with: `nothing` (default) for the closure's `sim.κt`,
or a function `case -> Field` (the SGS TKE diffuses with `2ν_t`, #182).
"""
mutable struct Tracer{F}
    name::Symbol
    field::F
    advect::Bool
    positive::Bool
    output::Bool
    units::String
    long_name::String
    diffusivity::Any
end

"The eddy diffusivity the `:transport` stage diffuses tracer `t` with."
tracer_diffusivity(t::Tracer, case) = t.diffusivity === nothing ? case.sim.κt : t.diffusivity(case)

"""
    tracer_vertical_diffusivity(t::Tracer, case) -> Field

The diffusivity of tracer `t` on the z faces: θ's vertical κv when the
`:sgs` closure is stability-corrected (P55-e: κv ≠ κh, so a tracer gets the
same diagonal diffusivity θ gets), else `tracer_diffusivity(t, case)`. A
tracer with its own `diffusivity` keeps it in every direction.
"""
tracer_vertical_diffusivity(t::Tracer, case) =
    t.diffusivity === nothing ? sgs_vertical_diffusivity(case) : tracer_diffusivity(t, case)

"θ's SGS diffusivity on the z faces: the stability-corrected closure's κv, else `sim.κt`."
function sgs_vertical_diffusivity(case)
    sc = stability_closure(get(case.step_kwargs, :sgs, nothing))
    return sc === nothing ? case.sim.κt : sc.κv
end

"Ordered set of tracers; iteration order is registration order."
struct TracerRegistry
    list::Vector{Tracer}
    index::Dict{Symbol,Int}
end
TracerRegistry() = TracerRegistry(Tracer[], Dict{Symbol,Int}())

Base.iterate(r::TracerRegistry, s...) = iterate(r.list, s...)
Base.length(r::TracerRegistry) = length(r.list)
Base.haskey(r::TracerRegistry, name::Symbol) = haskey(r.index, name)

"""
    ensure_tracer!(case, name; advect = false, positive = false, output = true,
                   units = "", long_name = "", diffusivity = nothing) -> Field

Return the tracer `name`, creating a zero `Field(CellCenter(), mesh)` if
it does not exist yet. Calling it again is safe: the flags are OR-ed
(`advect`, `positive`, `output`) and empty metadata is filled in, so two
couplings that both need `ql` share one field. `diffusivity` (see
[`Tracer`](@ref)) is set by the first caller that gives one.
"""
function ensure_tracer!(case, name::Symbol; advect::Bool = false, positive::Bool = false,
                        output::Bool = true, units::AbstractString = "",
                        long_name::AbstractString = "", diffusivity = nothing)
    reg = case.tracers
    if haskey(reg.index, name)
        t = reg.list[reg.index[name]]
        t.advect   |= advect
        t.positive |= positive
        t.output   |= output
        isempty(t.units) && (t.units = units)
        isempty(t.long_name) && (t.long_name = long_name)
        t.diffusivity === nothing && (t.diffusivity = diffusivity)
        return t.field
    end
    f = Field(CellCenter(), case.mesh)
    push!(reg.list, Tracer(name, f, advect, positive, output, String(units), String(long_name),
                           diffusivity))
    reg.index[name] = length(reg.list)
    return f
end

"The field of registered tracer `name` (error if absent)."
function tracer(case, name::Symbol)
    haskey(case.tracers.index, name) || error("no tracer :$name registered")
    return case.tracers.list[case.tracers.index[name]].field
end

"The registered tracers with `advect = true`, in registration order."
advected_tracers(case) = [t for t in case.tracers.list if t.advect]

"The fields of the advected tracers, in registration order (one halo set fill)."
advected_fields(case) = [t.field for t in case.tracers.list if t.advect]
