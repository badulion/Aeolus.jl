# Nesting by domains (ONE-f, #455): the nest coupling of a Model with
# several domains. The parent is just a domain; the coupling is
# Nesting.NestHierarchy (#328: one level, several levels, siblings; one-way
# or two-way per child), with every domain stepped through its own stage
# pipeline. The operators, the boundary data, the subcycling and the
# feedback are those of the Nesting module, unchanged.

"""
    PipelineCore(domain)

The dynamical core of a nested domain as `Aeolus.Nesting` sees it: calling it, `core(sim, dt; body_force!, post_predictor!)`, takes
one step of `domain` through its stage pipeline (the drivers, the stages
before the core, the core step with stages 6–8 and the nest's hooks
chained to the domain's, the stages after the core). The nest's
body-force hook (the two-way flux registers) runs before the domain's,
its post-predictor hook (a nest sponge) after it, as
`Nesting.DomainAdvance` chains them. Its stage times and stage weights
are those of the domain's scheme (`:euler`: one projection step;
`:rk3`: SSP-RK3).
"""
struct PipelineCore{D<:Domain}
    d::D
end

function (c::PipelineCore)(sim, dt; body_force! = nothing, post_predictor! = nothing, sgs = nothing)
    d = c.d
    sim === d.state || throw(ArgumentError("PipelineCore: the simulation is not its domain's core"))
    sync_drivers!(d)
    t0 = clock(d)
    _run_stages!(d, PRE_CORE, dt, t0, t0 + dt)
    _core_step!(sim, d, dt, body_force!, post_predictor!)
    _run_stages!(d, POST_CORE, dt, t0, clock(d))
    return sim
end

Nesting.nest_stage_weights(c::PipelineCore, args...) =
    c.d.scheme === :rk3 ? (1.0 / 6, 1.0 / 6, 2.0 / 3) : (1.0,)
Nesting._stage_times(c::PipelineCore, args) = c.d.scheme === :rk3 ? (0.0, 1.0, 0.5) : (0.0,)

"""
    pipeline_advance(d::Domain) -> Nesting.DomainAdvance

The per-domain advance of `d` in a nest: its [`PipelineCore`](@ref), with
its SGS closure as the `sgs` keyword (what the nest's flux registers read
for the SGS scalar flux at the interface).
"""
pipeline_advance(d::Domain) = domain_advance(PipelineCore(d); sgs = sgs_closure(d))

"""
    NestLink(child, parent, nest; feedback = :two_way)

One parent–child pair of a nested model: the domain numbers `child` and
`parent` (the parent listed first), their `Nesting.NestedGrid` `nest`
(whose child mesh is the child domain's grid, built with
`Nesting.nest_bcs`) and the `feedback`, `:two_way` (the self-nesting
default, the design's decision of 2026-10-09) or `:one_way`.
"""
struct NestLink{N}
    child::Int
    parent::Int
    nest::N
    feedback::Symbol
    function NestLink(child::Integer, parent::Integer, nest; feedback::Symbol = :two_way)
        feedback in (:one_way, :two_way) ||
            throw(ArgumentError("NestLink: feedback is :one_way or :two_way, got :$feedback"))
        parent < child || throw(ArgumentError("NestLink: a parent is listed before its child " *
                                              "(parent $parent, child $child)"))
        return new{typeof(nest)}(Int(child), Int(parent), nest, feedback)
    end
end

"""
    ModelNest

The nest coupling of a [`Model`](@ref) with several domains:
`hierarchy`, the `Nesting.NestHierarchy` whose domains are the model's
cores (domain `i` of the model is domain `i` of the hierarchy, with the
model's id), and `links`, the [`NestLink`](@ref)s it was built from.

One `step!(model, dt)` is one root step of the hierarchy (Berger & Oliger
1984): the root steps through its pipeline, the boundary data of each
child are prolongated at both ends of that step, every child subcycles
through its own pipeline to land on the step's end (`max(r)` substeps for
a refinement `r`, recursively for its own children), and the two-way
children feed back (anterpolation, interface flux correction,
Berger–Colella refluxing, re-projection of the parent). The design's
stages `:nest_sync` (1) and `:nest_feedback` (12) are these two halves of
the coupling, run by the model around the domains' pipelines.
"""
struct ModelNest{H}
    hierarchy::H
    links::Vector{NestLink}
end

"""
    nested_model(domains::Tuple, links) -> Model

The model of `domains` (the root first) coupled by `links` (a
[`NestLink`](@ref) per child, in domain order): a [`ModelNest`](@ref)
whose hierarchy steps every domain through its own pipeline
([`pipeline_advance`](@ref)), with fixed steps (the root's `dt`; each
child `max(r)` substeps per step of its parent) and no step rejection (a
blow-up is the run's sanity check: the blow-up guard inspects every
domain, [`component_nonfinite_fields`](@ref Aeolus.IO.component_nonfinite_fields),
a one-way child too). Refused with an `ArgumentError` that
names the domains: a domain that measures budgets (not attributed across
the nest's feedback yet), an adaptive domain, a child without a link or
with two, a child that is not properly nested in its parent, overlapping
siblings, or a two-way pair without a conservation box or of mixed cores
(those of `Nesting.add_nest!`).
"""
function nested_model(domains::Tuple, links)
    length(domains) >= 2 || throw(ArgumentError("nested_model: a nest needs two domains or more"))
    n = length(domains)
    for d in domains
        d.budgets === nothing ||
            throw(ArgumentError("nested_model: domain $(d.id) measures budgets; budgets across a nest " *
                                "(the two-way feedback is outside every stage) are not attributed yet"))
        d.cfl === nothing ||
            throw(ArgumentError("nested_model: domain $(d.id) steps adaptively; a nested model takes fixed steps"))
    end
    ls = NestLink[links...]
    sort(Int[l.child for l in ls]) == collect(2:n) ||
        throw(ArgumentError("nested_model: every domain but the first has exactly one parent link " *
                            "(links to $(sort(Int[l.child for l in ls])) for domains 2:$n)"))
    sort!(ls; by = l -> l.child)
    ids = [Symbol(d.id) for d in domains]
    root = domains[1]
    h = NestHierarchy(root.state; id = ids[1], advance = pipeline_advance(root), cfl = nothing,
                      validate = false)
    for l in ls
        c = domains[l.child]
        try
            add_nest!(h, ids[l.parent] => c.state, l.nest; id = ids[l.child], advance = pipeline_advance(c),
                      cfl = nothing, feedback = l.feedback)
        catch e
            e isa ArgumentError || rethrow()
            throw(ArgumentError("nested_model: domain $(c.id) in $(domains[l.parent].id): " * e.msg))
        end
    end
    return Model(domains, ModelNest(h, ls))
end

step!(m::Model{<:Tuple,<:ModelNest}, dt::Real) = (step_nested!(m.nest.hierarchy; dt = Float64(dt)); m)

"""
    Nesting.nest_link(m::Model, id::AbstractString) -> Nesting.NestedRun

The coupling of the model's domain `id` (its id string) to its parent:
the link of the nest hierarchy, with its statistics and its two-way
conservation ledger (`Nesting.nest_conservation_error`).
"""
Nesting.nest_link(m::Model{<:Tuple,<:ModelNest}, id::AbstractString) = Nesting.nest_link(m.nest.hierarchy, Symbol(id))

# A checkpoint of a nested model holds the whole hierarchy: every domain's
# core and components, both parent time levels of every link, the flux
# registers and statistics (the structural walk of #269).
checkpoint_state(n::ModelNest) = structural_state(n.hierarchy)
restore_checkpoint_state!(n::ModelNest, s) = (restore_structural_state!(n.hierarchy, s); n)

# The blow-up guard inspects the run's root core; the child domains are
# the nest's, so it reports theirs (a one-way child feeds nothing back,
# and a NaN in it would otherwise never reach the root), each field
# named `<domain>.<field>`.
function component_nonfinite_fields(n::ModelNest)
    out = NonFiniteField[]
    for d in @view n.hierarchy.domains[2:end]
        for f in nonfinite_fields(d.sim)
            push!(out, NonFiniteField(Symbol(d.id, ".", f.name), f.count, f.nan, f.inf,
                                      f.index, f.position, f.size))
        end
    end
    return out
end
