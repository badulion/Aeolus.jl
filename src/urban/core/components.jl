# The urban components (ONE-e, #454; formerly the coupling registry).
#
# Every urban component of the site is one file under `couplings/`: a
# struct `<: UrbanComponent` (an `Aeolus.Components.AbstractComponent`)
# with the methods below for its type. The site's components are the
# fixed list [`URBAN_COMPONENTS`](@ref) (`UrbanCase.jl`); each has a
# section of the model's schema (`[<name>]`, `enabled` off by default,
# its options from `default_options`), and a config enables a component
# by its section. `build_case` builds the enabled ones in list order, and
# the case's `Domain` runs them through the core's stage pipeline
# (`stages.jl`).
#
# Interface (every method has a default except `component_name` and the
# `run_stage!` methods for the stages a component declares):
#
#   component_name(::Type{C})   -> Symbol, unique: its section `[name]` (Components)
#   slot(::Type{C})             -> nothing (additive) or an urban slot (Components;
#                                  the urban slots are [`URBAN_SLOTS`](@ref)): a site
#                                  enables exactly one component per urban slot
#   provider(::Type{C})         -> Symbol, the name of the component within its slot
#   incompatible(::Type{C})     -> component types it excludes (Components)
#   incompatible_reason(::Type{C}) -> String, why (appended to the config error)
#   modules_used(::Type{C})     -> tuple of Aeolus submodule names it runs
#   default_options(::Type{C})  -> NamedTuple: its options and their defaults (the schema)
#   stages(::Type{C})           -> tuple of (stage::Symbol, order::Int, Cadence), one per stage
#   build(::Type{C}, case, opts)-> instance (default `C()`)
#   init!(c, case)              -> called once, after every component is built
#   init_priority(::Type{C})    -> Int, init! order (lower first, default 0; ties: list order)
#   run_stage!(c, case, ::Val{S}, dt)
#   output_fields(c, case)      -> Vector of OutputVariable, registered when the file is created
#   budget!(c, case, ledger)    -> energy / water terms for the ledger
#   diagnostics(c, case)        -> Vector{Diagnostic} (Aeolus.RunLog): the scalars c reports,
#                                  in diagnostics.csv / .nc and in its log event of each output
#                                  record (core/diagnostics.jl); a wall-clock one is `wall.<x>`
#   profile_diagnostics(c, case)-> Vector{ProfileDiagnostic}: its vertical profiles, diagnostics.nc
#   smoke_variants(::Type{C})   -> tuple of configs worth a smoke run
#   smoke_check(c, case)        -> Vector{Pair{String,Bool}} (its `<name>_checks`)
#   sim_options(::Type{C}, cfg, mesh, opts) -> NamedTuple of keywords of the core's constructor
#   step_options(c, case)       -> NamedTuple of keywords of the core's step! (and of the
#                                  :timestepper provider's stepper)
#   limit_dt(c, case, dt)       -> the next step of a variable-step stepper, shortened
#                                  if `c` needs it (the nest: land on the parent window, #265)
#   end_run!(c, case)           -> called by run_case! after its last step (the nest waits
#                                  for a parent step it started ahead, SC-c #403)
#
# Dynamics interface (#133), used only by the components of the two
# dynamics slots:
#
#   slot :core         build_core(::Type{C}, cfg, mesh, ref, simkw) -> the dynamical-core
#                      simulation, on the case reference state `ref`, with the merged
#                      `sim_options` keywords `simkw` (a keyword it cannot honour is an error)
#   slot :timestepper  next_dt(c, case)                 -> dt (s) of the next step
#                      advance_dynamics!(c, case, dt)   -> advance case.sim by dt, with the
#                                                          hooks and case.step_kwargs
#                      variable_dt(::Type{C})           -> Bool: run_case! runs to a time
#                      stable_dt(c, case)               -> the longest step the provider's
#                                                          limits allow now (variable_dt only;
#                                                          an explicit dt of step_case! is checked against it)
#
# Keywords are never dropped: a combination of a core / time stepper with
# a numerics component whose keywords it does not take is declared with
# `incompatible` (and `incompatible_reason`), so enabling both is a config
# error.

"Every stage name, in execution order: the core pipeline's (`Components.PIPELINE`)."
const ALL_STAGES = Components._STAGE_NAMES

"""
    UrbanComponent <: Components.AbstractComponent

Supertype of every urban component of the site (ONE-e, #454; formerly
`AbstractCoupling` of the coupling registry). See `core/components.jl`
for its interface and `docs/src/design/berlin_case.md` for how to add
one. Its section is `[`[`component_name`](@ref Aeolus.Components.component_name)`]`;
the case runs it through the pipeline in an [`UrbanEntry`](@ref).
"""
abstract type UrbanComponent <: AbstractComponent end

"""
    Cadence(period; first = false)
    Cadence(key::Symbol; first = false)

When a stage entry runs. `period` is in **seconds**: the entry fires on
the step that crosses a multiple of it, `floor(t_new/P) > floor(t_old/P)`,
and on step 1 as well when `first = true`. `period ≤ 0` means every step.
The `Symbol` form names a period in `cfg.cadence` (e.g. `:radiation`),
resolved when the case is built.

With a fixed `dt` that divides `P` this is exactly `n % (P/dt) == 0`, and
it keeps working when `dt` changes from step to step.
"""
struct Cadence{P}
    period::P
    first::Bool
end
Cadence(period; first::Bool = false) = Cadence(period, first)

"Every step."
const EVERY_STEP = Cadence(0.0)

resolve(c::Cadence{<:Real}, cfg) = Cadence(Float64(c.period), c.first)
resolve(c::Cadence{Symbol}, cfg) = Cadence(cadence_period(cfg, c.period), c.first)

"""
    fires(c::Cadence, n, t_old, t_new) -> Bool

Whether the entry with cadence `c` runs on step `n`, which advances the
simulation time from `t_old` to `t_new` (seconds).
"""
function fires(c::Cadence{Float64}, n::Integer, t_old::Real, t_new::Real)
    c.first && n == 1 && return true
    c.period <= 0 && return true
    return floor(t_new / c.period) > floor(t_old / c.period)
end

# --- contract defaults -------------------------------------------------------

Components.section(C::Type{<:UrbanComponent}) = String(component_name(C))
Components.enabled(::Type{<:UrbanComponent}, opts) = opts.enabled
provider(C::Type{<:UrbanComponent}) = component_name(C)
modules_used(::Type{<:UrbanComponent}) = ()
default_options(::Type{<:UrbanComponent}) = (;)
incompatible_reason(::Type{<:UrbanComponent}) = ""
stages(::Type{<:UrbanComponent}) = ()
build(C::Type{<:UrbanComponent}, case, opts) = C()
init!(c::UrbanComponent, case) = nothing
"""
    init_priority(::Type{C}) -> Int

The order of the `init!` calls: lower first (default 0), ties in the
order of [`URBAN_COMPONENTS`](@ref). The `:nesting` coupling initialises first (−1, #265): the nest
faces' data and the child's initial wind come from the parent, and the
other couplings initialise from them (e.g. the deposition's r_a from the
first air cell's wind, which reads the face velocity at an edge column).
"""
init_priority(::Type{<:UrbanComponent}) = 0
output_fields(c::UrbanComponent, case) = OutputVariable[]
budget!(c::UrbanComponent, case, ledger) = nothing
smoke_variants(::Type{<:UrbanComponent}) = ()
smoke_check(c::UrbanComponent, case) = Pair{String,Bool}[]
"""
    sim_options(::Type{C}, cfg, mesh, opts) -> NamedTuple

Keywords `C` adds to the constructor of the dynamical core (e.g. the
`solver` of `BoussinesqSimulation`, or the `tol` / `maxit` of the
sound-proof cores' density-weighted CG), called before the core is built
and passed to `build_core` of the `:core` provider. Two couplings may not
set the same keyword, and a core rejects a keyword it does not take.
"""
sim_options(::Type{<:UrbanComponent}, cfg, mesh, opts) = (;)
"""
    step_options(c, case) -> NamedTuple

Keywords `c` adds to every dynamics step (e.g. `momentum_advection`,
`sgs`), collected into `case.step_kwargs` when the schedule
is built and passed by the `:timestepper` provider to `step!` (or to
`step_imex!`, which takes the same keywords). Two couplings may not set
the same keyword.
"""
step_options(c::UrbanComponent, case) = (;)
"""
    limit_dt(c, case, dt) -> dt

The length of the next step under a variable-step `:timestepper`
provider (`variable_dt`), given the stepper's own `dt`: a coupling that
needs the steps to land on its own times shortens it (the `:nesting`
coupling makes the child's steps land on the end of the parent's step,
#265). Default: `dt`. [`step_dt`](@ref) applies every enabled coupling's.
"""
limit_dt(c::UrbanComponent, case, dt) = dt
"""
    end_run!(c, case)

Called by [`run_case!`](@ref) after its last step, before the output is
closed: a coupling that runs work ahead of the child on its own task
finishes it here (the `:nesting` coupling waits for a prefetched parent
step, SC-c #403). Default: nothing.
"""
end_run!(c::UrbanComponent, case) = nothing
function build_core end
variable_dt(::Type{<:UrbanComponent}) = false
next_dt(c::UrbanComponent, case) = case.cfg.dt
"""
    stable_dt(c, case) -> dt

The longest step (s) a variable-step `:timestepper` provider
(`variable_dt`) allows from the current state: its stability limits
(CFL, diffusion, buoyancy, ...) and `dt_max`, without the growth limit of
`next_dt` and without side effects. [`step_case!`](@ref) rejects
an explicit `dt` above it (#392). Default: `Inf` (a fixed-step provider
takes the `dt` it is given).
"""
stable_dt(c::UrbanComponent, case) = Inf
advance_dynamics!(c::UrbanComponent, case, dt) = dynamics_step!(case, dt)
run_stage!(c::UrbanComponent, case, ::Val{S}, dt) where {S} =
    error("coupling :$(component_name(typeof(c))) declares stage :$S but has no run_stage! method for it")


"Merge the keyword `NamedTuple`s `parts` (`name => nt`); a keyword set twice is an error."
function merge_keywords(what::AbstractString, parts)
    kw = (;)
    for (name, nt) in parts
        for k in keys(nt)
            haskey(kw, k) && error("$what keyword `$k` is set by two couplings (the second is :$name)")
        end
        kw = merge(kw, nt)
    end
    return kw
end

# --- the components of the site -----------------------------------------------

"""
    URBAN_SLOTS

The exclusive slots of the urban site, each filled by exactly one enabled
component: the dynamics (`:core`, `:timestepper`, `:pressure_solver`,
`:momentum_advection`, `:scalar_advection`, `:sgs`), the surfaces and
radiation (`:surface_exchange`, `:ground`, `:surface_radiation`,
`:column_radiation`, `:buildings`, `:glazing`, `:immersed_boundary`) and
the air's chemistry and water (`:chemistry`, `:microphysics`).
"""
const URBAN_SLOTS = (:core, :timestepper, :pressure_solver, :momentum_advection, :scalar_advection, :sgs,
                     :surface_exchange, :ground, :surface_radiation, :column_radiation, :buildings, :glazing,
                     :immersed_boundary, :chemistry, :microphysics)

"""
    check_urban_component(C)

What every component of [`URBAN_COMPONENTS`](@ref) must satisfy: an
urban slot or none, stage entries `(stage, order, Cadence)` of pipeline
stages, at most one per stage (`run_stage!` dispatches on the stage).
"""
function check_urban_component(C::Type{<:UrbanComponent})
    s = slot(C)
    s === nothing || s in URBAN_SLOTS || error("$C: unknown urban slot :$s (the slots: $(join(URBAN_SLOTS, ", ")))")
    for st in stages(C)
        length(st) == 3 || error("$C: stage entries are (stage, order, Cadence), got $st")
        st[1] in ALL_STAGES || error("$C: unknown stage :$(st[1]); stages are $(ALL_STAGES)")
    end
    ss = [st[1] for st in stages(C)]
    allunique(ss) || error("$C: at most one entry per stage (run_stage! dispatches on the stage), got $ss")
    return C
end

"The urban component type called `name`."
function component_type(name::Symbol)
    for C in URBAN_COMPONENTS
        component_name(C) === name && return C
    end
    error("no urban component named :$name")
end

"""
    selected_components(cfg::CaseConfig) -> Vector{Type}

The urban component types `cfg` enables (`cfg.components`), in the order
of [`URBAN_COMPONENTS`](@ref).
"""
selected_components(cfg) = Type[C for C in URBAN_COMPONENTS if haskey(cfg.components, component_name(C))]

"""
    component_options(C, cfg::CaseConfig) -> NamedTuple

The options of urban component type `C` under `cfg`: those of its section
when `cfg` enables it, else its `default_options`.
"""
component_options(C::Type{<:UrbanComponent}, cfg) = get(cfg.components, component_name(C), default_options(C))
