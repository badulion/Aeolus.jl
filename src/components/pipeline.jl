# The fixed stage pipeline (ONE-a, #450): the stage names in execution
# order, the slots, the order of components within a stage, the
# validation of a domain's components, and the loop over them.

"""
    PIPELINE

The stages of one step of one domain, in execution order, as
`(number, name)`; the numbers are those of the design
(`docs/src/design/single_model.md`, "The fixed stage pipeline"):

| # | stages | |
| :-- | :-- | :-- |
| 1 | `:clock`, `:nest_sync` | after the domain's drivers advance to the step |
| 2 | `:radiation`, `:sw_modifiers` | |
| 3 | `:surface_flux` | |
| 4 | `:sources` | |
| 5 | `:chemistry`, `:microphysics`, `:tracers` | |
| 6 | `:tendencies` | inside the core, after its halo fill, advection, SGS and buoyancy |
| 7 | `:body_force` | inside the core, before the predictor |
| 8 | `:post_predictor` | inside the core, after the predictor |
| 9 | `:open_boundaries` | the core's open-boundary update (no components yet) |
| 10 | `:projection` | the core's projection and scalar advance (no components) |
| 11 | `:surface_update`, `:column_tendency` | |
| 12 | `:nest_feedback` | |
| 13 | `:budgets`, `:checks`, `:output` | |

Stages 6–8 run inside every substage of the core (three times per RK3
step) and take no cadence. When no component attaches to them the core
runs its hook-free fast path (SC-d/SC-e). Within a stage, components run
in the fixed order of [`order_components`](@ref).
"""
const PIPELINE = ((1, :clock), (1, :nest_sync),
                  (2, :radiation), (2, :sw_modifiers),
                  (3, :surface_flux),
                  (4, :sources),
                  (5, :chemistry), (5, :microphysics), (5, :tracers),
                  (6, :tendencies),
                  (7, :body_force),
                  (8, :post_predictor),
                  (9, :open_boundaries),
                  (10, :projection),
                  (11, :surface_update), (11, :column_tendency),
                  (12, :nest_feedback),
                  (13, :budgets), (13, :checks), (13, :output))

const _STAGE_NAMES = map(last, PIPELINE)
"The stages that run before the core step, in order."
const PRE_CORE = (:clock, :nest_sync, :radiation, :sw_modifiers, :surface_flux, :sources,
                  :chemistry, :microphysics, :tracers)
"The stages the core runs inside each (sub)step, in order."
const IN_CORE = (:tendencies, :body_force, :post_predictor)
"The stages that run after the core step, in order."
const POST_CORE = (:surface_update, :column_tendency, :nest_feedback, :budgets, :checks, :output)
# :open_boundaries and :projection are the core's own operations: no component attaches yet

"""
    SLOTS

The exclusive slots, in the order their components run within a stage
(before every additive component): `:advection`, `:sgs`, `:microphysics`,
`:radiation`, `:surface`.
"""
const SLOTS = (:advection, :sgs, :microphysics, :radiation, :surface)

"""
    Stage(name; every = 0.0)

A pipeline stage a component runs in ([`stages`](@ref)), with a cadence:
`every = 0` runs every step; `every > 0` (seconds of model time) runs in
the step that reaches or crosses a multiple of `every`. The stages inside
the core (6–8) take no cadence.
"""
struct Stage
    name::Symbol
    every::Float64
    function Stage(name::Symbol; every::Real = 0.0)
        name in _STAGE_NAMES || throw(ArgumentError("Stage: unknown stage :$name (the stages: $(join(_STAGE_NAMES, ", ")))"))
        every >= 0 || throw(ArgumentError("Stage: every must be ≥ 0, got $every"))
        return new(name, Float64(every))
    end
end

"Whether a stage with cadence `every` runs in the step from `t0` to `t1`."
_due(every::Float64, t0, t1) = every == 0 || fld(t1, every) > fld(t0, every)

"""
    ComponentError(problems)

A [`Config.ConfigError`](@ref Aeolus.Config.ConfigError) about a domain's
components: each problem is keyed by the section of the component it
concerns.
"""
ComponentError(problems::Vector{ConfigProblem}) = ConfigError(problems)

_slot_rank(s::Nothing) = length(SLOTS) + 1
_slot_rank(s::Symbol) = something(findfirst(==(s), SLOTS), length(SLOTS) + 2)

"""
    order_components(components) -> Tuple

The components in the fixed order they run within every stage: by slot
first (in the order of [`SLOTS`](@ref), additive components last), then
by [`component_name`](@ref). Their order is never a configuration
question.
"""
order_components(cs) =
    Tuple(sort!(collect(Any, cs); by = c -> (_slot_rank(slot(c)), string(component_name(c)))))

"""
    validate_components(components)

Throw a [`ComponentError`](@ref) listing every problem of a domain's
components: two in one slot, an unknown slot, a duplicate name, a missing
requirement, an incompatible pair, a stage no component may attach to
(`:open_boundaries`, `:projection`), a cadence on a stage inside the core,
or a domain without a provider of the `:advection` slot.
"""
function validate_components(cs)
    probs = ConfigProblem[]
    p(c, msg) = push!(probs, ConfigProblem(section(c), "", msg))
    types = map(typeof, collect(Any, cs))
    names = Dict{Symbol,Any}(); slots = Dict{Symbol,Any}()
    for c in cs
        n = component_name(c)
        haskey(names, n) && p(c, "two components named :$n")
        names[n] = c
        s = slot(c)
        if s !== nothing
            s in SLOTS || p(c, "unknown slot :$s (the slots: $(join(SLOTS, ", ")))")
            haskey(slots, s) &&
                p(c, "the :$s slot holds one component; [$(section(slots[s]))] ($(component_name(slots[s]))) " *
                     "and [$(section(c))] ($n) are both enabled")
            slots[s] = c
        end
        for R in requires(typeof(c))
            any(T -> T <: R, types) || p(c, "$n requires $(nameof(R)) ([$(section(R))]), which is not enabled")
        end
        for X in incompatible(typeof(c)), d in cs
            typeof(d) <: X &&
                p(c, "$n is incompatible with $(component_name(d)) ([$(section(d))]); enable one of them")
        end
        for st in stages(c)
            st.name in (:open_boundaries, :projection) &&
                p(c, "$n attaches to :$(st.name), a core operation no component attaches to")
            st.name in IN_CORE && st.every != 0 &&
                p(c, "$n: stage :$(st.name) runs inside the core step and takes no cadence (every = $(st.every))")
        end
    end
    haskey(slots, :advection) ||
        push!(probs, ConfigProblem("advection", "", "a domain needs a provider of the :advection slot"))
    isempty(probs) || throw(ComponentError(probs))
    return nothing
end

"""
    attaches(c, stage::Symbol) -> Bool

Whether component `c` runs in `stage`.
"""
attaches(c, stage::Symbol) = any(s -> s.name === stage, stages(c))

# the cadence of `c` in `stage` (attached)
function _every(c, stage::Symbol)
    for s in stages(c)
        s.name === stage && return s.every
    end
    return -1.0
end

# Run the components of `cs` attached to stage `S`, in tuple order (the
# domain's tuple is ordered): recursion over the tuple, so each call is
# statically dispatched.
@inline _run_each(::Tuple{}, d, ::Val, dt, t0, t1) = nothing
@inline function _run_each(cs::Tuple, d, v::Val{S}, dt, t0, t1) where {S}
    c = first(cs)
    e = _every(c, S)
    e >= 0 && _due(e, t0, t1) && run_stage!(c, d, v, dt)
    return _run_each(Base.tail(cs), d, v, dt, t0, t1)
end
# inside the core: every attached component, every substage
@inline _run_in_core(::Tuple{}, d, ::Val, dt) = nothing
@inline function _run_in_core(cs::Tuple, d, v::Val{S}, dt) where {S}
    c = first(cs)
    attaches(c, S) && run_stage!(c, d, v, dt)
    return _run_in_core(Base.tail(cs), d, v, dt)
end
