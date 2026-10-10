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
| 5 | `:chemistry`, `:transport`, `:microphysics`, `:tracers` | `:transport`: the carried tracers' advection with the step's starting velocity |
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
                  (5, :chemistry), (5, :transport), (5, :microphysics), (5, :tracers),
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
                  :chemistry, :transport, :microphysics, :tracers)
"The stages the core runs inside each (sub)step, in order."
const IN_CORE = (:tendencies, :body_force, :post_predictor)
"The stages that run after the core step, in order (the budgets close after `:budgets`)."
const POST_CORE = (:surface_update, :column_tendency, :nest_feedback, :budgets, :checks, :output)
const _POST_CORE_BUDGETS = (:surface_update, :column_tendency, :nest_feedback, :budgets)
const _POST_BUDGETS = (:checks, :output)
# :open_boundaries and :projection are the core's own operations: no component attaches yet

"""
    SLOTS

The exclusive slots, in the order their components run within a stage
(before every additive component): `:advection`, `:sgs`, `:microphysics`,
`:radiation`, `:surface`.
"""
const SLOTS = (:advection, :sgs, :microphysics, :radiation, :surface)

"""
    Stage(name; every = 0.0, first = false, order = 0)

A pipeline stage a component runs in ([`stages`](@ref)), with a cadence:
`every = 0` runs every step; `every > 0` (seconds of model time) runs in
the step that crosses a multiple of `every`, `floor(t₁/every) >
floor(t₀/every)`, and with `first = true` also in the step that starts at
model time 0. The stages inside the core (6–8) take no cadence.

`order` places the component within the stage: lower first, ties by slot,
then by [`component_name`](@ref) ([`order_components`](@ref)). It is
declared by the component's code, never by the configuration, and is for
components whose work in a stage depends on another's (the column
radiation before the radiosity that distributes its fluxes, before the
buildings that absorb them).
"""
struct Stage
    name::Symbol
    every::Float64
    first::Bool
    order::Int
    function Stage(name::Symbol; every::Real = 0.0, first::Bool = false, order::Integer = 0)
        name in _STAGE_NAMES || throw(ArgumentError("Stage: unknown stage :$name (the stages: $(join(_STAGE_NAMES, ", ")))"))
        every >= 0 || throw(ArgumentError("Stage: every must be ≥ 0, got $every"))
        return new(name, Float64(every), first, Int(order))
    end
end

"Whether stage entry `st` runs in the step from `t0` to `t1`."
_due(st::Stage, t0, t1) = st.every == 0 || (st.first && t0 == 0) || floor(t1 / st.every) > floor(t0 / st.every)

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
by [`component_name`](@ref), except where a stage's [`Stage`](@ref)
`order`s say otherwise: a component with a lower `order` in some stage
runs before one with a higher `order` in that stage. The result is the
order by slot and name that meets every such precedence (a topological
order; components that declare no `order` keep the slot-and-name order).
Their order is never a configuration question.
"""
function order_components(cs)
    key(c) = (_slot_rank(slot(c)), string(component_name(c)))
    v = sort!(collect(Any, cs); by = key)
    n = length(v)
    # precedence: within a stage, a lower `order` first
    before = [Int[] for _ in 1:n]          # before[j]: the components that must precede j
    for i in 1:n, j in 1:n
        i == j && continue
        for a in stages(v[i]), b in stages(v[j])
            a.name === b.name && a.order < b.order && (push!(before[j], i); break)
        end
    end
    done = falses(n); out = Any[]
    for _ in 1:n
        k = findfirst(j -> !done[j] && all(i -> done[i], before[j]), 1:n)
        k === nothing && throw(ArgumentError("order_components: the stage orders of $(join(map(c -> component_name(c), v[.!done]), ", ")) contradict each other"))
        done[k] = true; push!(out, v[k])
    end
    return Tuple(out)
end

"""
    component_problems(types) -> Vector{ConfigProblem}

The problems of a set of enabled component types (one domain's), keyed
by the section of the component each concerns: two in one slot, an
unknown slot, a duplicate name, a missing requirement ([`requires`](@ref)),
an incompatible pair ([`incompatible`](@ref)), or no provider of the
`:advection` slot. Type-level, so the configuration is checked against it
before anything is built (ONE-b, #451); [`validate_components`](@ref)
adds the instances' stage rules.
"""
function component_problems(types)
    probs = ConfigProblem[]
    p(C, msg) = push!(probs, ConfigProblem(section(C), "", msg))
    names = Dict{Symbol,Any}(); slots = Dict{Symbol,Any}()
    for C in types
        n = component_name(C)
        haskey(names, n) && p(C, "two components named :$n")
        names[n] = C
        s = slot(C)
        if s !== nothing
            s in SLOTS || p(C, "unknown slot :$s (the slots: $(join(SLOTS, ", ")))")
            haskey(slots, s) &&
                p(C, "the :$s slot holds one component; [$(section(slots[s]))] ($(component_name(slots[s]))) " *
                     "and [$(section(C))] ($n) are both enabled")
            slots[s] = C
        end
        for R in requires(C)
            any(T -> T <: R, types) || p(C, "$n requires $(nameof(R)) ([$(section(R))]), which is not enabled")
        end
        for X in incompatible(C), D in types
            D <: X && p(C, "$n is incompatible with $(component_name(D)) ([$(section(D))]); enable one of them")
        end
    end
    haskey(slots, :advection) ||
        push!(probs, ConfigProblem("advection", "", "a domain needs a provider of the :advection slot"))
    return probs
end

"""
    validate_components(components)

Throw a [`ComponentError`](@ref) listing every problem of a domain's
components: those of their types ([`component_problems`](@ref): slots,
names, requirements, incompatibilities, the `:advection` provider), a
stage no component may attach to (`:open_boundaries`, `:projection`), or
a cadence on a stage inside the core.
"""
function validate_components(cs)
    probs = component_problems(Any[typeof(c) for c in cs])
    for c in cs, st in stages(c)
        n = component_name(c)
        st.name in (:open_boundaries, :projection) &&
            push!(probs, ConfigProblem(section(c), "", "$n attaches to :$(st.name), a core operation no component attaches to"))
        st.name in IN_CORE && st.every != 0 &&
            push!(probs, ConfigProblem(section(c), "", "$n: stage :$(st.name) runs inside the core step and takes no cadence (every = $(st.every))"))
    end
    isempty(probs) || throw(ComponentError(probs))
    return nothing
end

"""
    attaches(c, stage::Symbol) -> Bool

Whether component `c` runs in `stage`.
"""
attaches(c, stage::Symbol) = any(s -> s.name === stage, stages(c))

# the entry of `c` in `stage`, or nothing
function _entry(c, stage::Symbol)
    for s in stages(c)
        s.name === stage && return s
    end
    return nothing
end

# Run the components of `cs` attached to stage `S`, in tuple order (the
# domain's tuple is ordered): recursion over the tuple, so each call is
# statically dispatched.
# With the domain's budgets on, every entry is an attribution context of
# the measured budgets (components/budgets.jl); off, it is the plain call.
@inline _run_each(::Tuple{}, d, ::Val, dt, t0, t1) = nothing
@inline function _run_each(cs::Tuple, d, v::Val{S}, dt, t0, t1) where {S}
    c = first(cs)
    e = _entry(c, S)
    e === nothing || _due(e, t0, t1) && _attributed!(d.budgets, c, d, v, dt)
    return _run_each(Base.tail(cs), d, v, dt, t0, t1)
end
# inside the core: every attached component, every substage
@inline _run_in_core(::Tuple{}, d, ::Val, dt) = nothing
@inline function _run_in_core(cs::Tuple, d, v::Val{S}, dt) where {S}
    c = first(cs)
    attaches(c, S) && _attributed_in_core!(d.budgets, c, d, v, dt)
    return _run_in_core(Base.tail(cs), d, v, dt)
end
