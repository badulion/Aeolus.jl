# The component interface (ONE-a, #450): the methods every component type
# implements or inherits. docs/src/design/single_model.md, "The component
# interface", is the specification; docs/src/design/components.md describes
# what is built.

"""
    AbstractComponent

The supertype of every component: a reusable module instance configured by
one TOML section and attached to named stages of the pipeline
([`PIPELINE`](@ref)). A component is a **thin adapter**: if its file
contains an equation, that equation belongs in a physics module. Its
interface:

| method | default | purpose |
| :-- | :-- | :-- |
| [`component_name`](@ref)`(C)` | required | its name, the key of its checkpoint state and its order within a stage |
| [`section`](@ref)`(C)`, [`schema`](@ref)`(C)` | required | its TOML section and option schema |
| [`enabled`](@ref)`(C, opts)` | `true` | whether the resolved section switches it on |
| [`slot`](@ref)`(C)` | `nothing` | an exclusive slot (`SLOTS`), or `nothing` for an additive component |
| [`requires`](@ref)`(C)`, [`incompatible`](@ref)`(C)` | `()` | component types it needs or excludes |
| [`build`](@ref)`(C, opts, domain)` | required | the instance, with its fields and work arrays |
| [`stages`](@ref)`(c)` | `()` | the [`Stage`](@ref)s it runs in |
| [`run_stage!`](@ref)`(c, domain, Val(stage), dt)` | required per stage | the work; type-stable |
| [`limit_dt`](@ref)`(c, domain)` | `nothing` | an explicit stability limit `(dt, label)` |
| [`budget!`](@ref)`(c, domain, budgets)` | no terms | the terms it reports in the shared budgets |
| [`budget_fields`](@ref)`(c, q)`, [`solid_mask`](@ref)`(c)` | `()`, `nothing` | the fields of a measured quantity it carries, its obstacles |
| [`outputs`](@ref)`(c)`, [`output_names`](@ref)`(C, opts)` | empty | its output catalogue |
| [`checks`](@ref)`(c)` | `()` | the checks it publishes |
| `checkpoint_state(c)`, `restore_checkpoint_state!(c, s)` | `nothing` (stateless) | everything a bit-identical restart needs ([`checkpoint_state`](@ref Aeolus.IO.checkpoint_state)) |

Every component works with both dynamical cores (`BoussinesqSimulation`
and `AnelasticSimulation`).
"""
abstract type AbstractComponent end

"""
    component_name(::Type{C}) -> Symbol

The name of component type `C` (`:sgs`, `:coriolis`, …): the key of its
state in a checkpoint and, after the slot, its place within a stage
([`order_components`](@ref)). Unique within a domain.
"""
function component_name end
component_name(c::AbstractComponent) = component_name(typeof(c))

"""
    section(::Type{C}) -> String

The TOML section that configures `C`; [`ComponentError`](@ref)s name it.
"""
function section end
section(c::AbstractComponent) = section(typeof(c))

"""
    schema(::Type{C}) -> Config.Section

The option schema of `C`: a [`Config.Section`](@ref Aeolus.Config.Section)
named [`section`](@ref)`(C)` with the component's own options (types,
units, defaults, docs). The model's configuration schema has one
per-domain section per component, generated from these (ONE-b, #451).
"""
function schema end

"""
    enabled(::Type{C}, opts) -> Bool

Whether the resolved options `opts` of `C`'s section switch `C` on
(default `true`). It depends on the options only, so the configuration is
validated ([`component_problems`](@ref)) before anything is built.
"""
enabled(::Type{<:AbstractComponent}, opts) = true

"""
    slot(::Type{C}) -> Union{Nothing,Symbol}

`nothing` for an additive component; otherwise its exclusive slot, one of
[`SLOTS`](@ref). A domain holds at most one component per slot.
"""
slot(::Type{<:AbstractComponent}) = nothing
slot(c::AbstractComponent) = slot(typeof(c))

"""
    requires(::Type{C}) -> Tuple
    incompatible(::Type{C}) -> Tuple

The component types `C` needs in the same domain, and those it excludes.
[`validate_components`](@ref) reports a violation with the section names
involved.
"""
requires(::Type{<:AbstractComponent}) = ()
"See [`requires`](@ref)."
incompatible(::Type{<:AbstractComponent}) = ()

"""
    build(::Type{C}, opts, domain::Domain) -> C

Create the component from the resolved options `opts` of its section,
allocating its fields and work arrays on `domain`'s grid (the domain's
core and drivers are in place; its components are not).
"""
function build end

"""
    stages(c) -> Tuple{Vararg{Stage}}

The pipeline stages `c` runs in, each a [`Stage`](@ref) with an optional
cadence. Default: none (a slot component the core calls itself, such as
[`SGS`](@ref) and [`CentredAdvection`](@ref)).
"""
stages(::AbstractComponent) = ()

"""
    run_stage!(c, domain::Domain, ::Val{stage}, dt)

The work of `c` in `stage` over the (sub)step `dt`. It must be
type-stable and allocate nothing per call. Inside the core (stages
`:tendencies`, `:body_force`, `:post_predictor`) the state is the core's
substage state and `dt` the substage length.
"""
function run_stage! end

"""
    limit_dt(c, domain::Domain) -> Union{Nothing,Tuple{Float64,Symbol}}

An explicit stability limit of `c` on an adaptive step, `(dt, label)`,
or `nothing` (the default). The adaptive step takes the smallest limit
of the core and every component (`TimeIntegration.cfl_diagnostics`,
`limits`); `label` becomes the step's limiter.
"""
limit_dt(::AbstractComponent, domain) = nothing

"""
    budget!(c, domain::Domain, budgets::Budgets) -> nothing

Book the terms `c` reports in the domain's shared budgets
(`Aeolus.BudgetSystem`, ONE-c #452): its surface energy terms
(`add_surface!`), the water it adds to the air from outside it
(`add_water!`) and free-form terms (`add_term!`). The model calls it in
stage 13 (`:budgets`) of every step with the domain's budgets on, after
the step (`budgets.dt` is the step taken). What `c` does to the air in
its stage entries needs no booking: the budgets measure every entry
([`with_budgets`](@ref)). The default books nothing.
"""
budget!(::AbstractComponent, domain, budgets) = nothing

"""
    outputs(c) -> Vector{OutputVariable}

`c`'s output catalogue: the `Aeolus.Output.OutputVariable`s it can write
(name, units, dimensions and how each is computed), named exactly as
[`output_names`](@ref) gives them for its type and options. `[output]
variables` selects from the catalogues of the enabled components and the
core ([`domain_catalogue`](@ref)). The default is empty.
"""
outputs(::AbstractComponent) = OutputVariable[]

"""
    checks(c) -> Tuple

The checks `c` publishes (name, default tolerance, evaluation), which
`[checks]` enables (ONE-d, #453). The default is empty.
"""
checks(::AbstractComponent) = ()

# A stateless component: nothing to checkpoint (the IO component contract).
checkpoint_state(::AbstractComponent) = nothing
restore_checkpoint_state!(c::AbstractComponent, ::Nothing) = c
