# Domain and Model, and one step of the pipeline (ONE-a, #450).

"""
    Domain(state; id = "main", scheme = :euler, cfl = nothing, drivers = (;),
           components = ())

One domain of a [`Model`](@ref):

  * `state` — its **core**, the dynamical core with its prognostic state
    and its grid (`state.mesh`): a `BoussinesqSimulation` or an
    `AnelasticSimulation` ([`core`](@ref)); its clock is `state.t`
    ([`clock`](@ref));
  * `scheme` — the core's time scheme, `:euler` (`step!`) or `:rk3`
    (`step_rk3!`, SSP-RK3), on either core (ONE-j, #460);
  * `cfl` — `nothing` for a fixed step, or the `CFLConstraints` of an
    adaptive one (`step_adaptive!(domain)`), on either core;
  * `drivers` — its input drivers: `boundary`, the
    `FileBoundaryFeed` of open lateral faces, advanced at the start of
    every step, and `dynamic`, the `DynamicInputs` of the domain's
    dynamic driver (its large-scale forcing tables and surface series,
    which components read when they are built: [`LargeScale`](@ref),
    [`SurfaceLayer`](@ref));
  * `components` — its components, validated ([`validate_components`](@ref))
    and stored as a concretely typed tuple in the fixed order of
    [`order_components`](@ref), so dispatch on them is static;
  * `budgets` — `nothing` (the default: no measurement), or the
    [`DomainBudgets`](@ref) of [`with_budgets`](@ref).

`Domain(d::Domain; components)` is `d` with other components (how a run
type builds them: [`build_components`](@ref) needs the domain), and its
budgets off.
"""
struct Domain{S, C<:Tuple, D<:NamedTuple, F, B}
    id::String
    state::S
    scheme::Symbol
    cfl::F
    drivers::D
    components::C
    budgets::B
end

function Domain(state; id::AbstractString = "main", scheme::Symbol = :euler, cfl = nothing,
                drivers::NamedTuple = (;), components = (), validate::Bool = true)
    state isa Union{BoussinesqSimulation,MassWeightedSimulation} ||
        throw(ArgumentError("Domain: the core is a BoussinesqSimulation or an AnelasticSimulation, got $(typeof(state))"))
    scheme in (:euler, :rk3) || throw(ArgumentError("Domain: scheme is :euler or :rk3, got :$scheme"))
    cfl === nothing || cfl isa CFLConstraints || throw(ArgumentError("Domain: cfl is nothing or a CFLConstraints"))
    validate && validate_components(components)
    return Domain(String(id), state, scheme, cfl, drivers, order_components(components), nothing)
end
Domain(d::Domain; components) =
    Domain(d.state; id = d.id, scheme = d.scheme, cfl = d.cfl, drivers = d.drivers, components = components)

"""
    build_components(specs, domain::Domain) -> Tuple

Build the enabled components of `specs`, pairs `C => opts` of a component
type and the resolved options of its section, on `domain`
([`enabled`](@ref), [`build`](@ref)).
"""
build_components(specs, domain::Domain) =
    Tuple(build(C, opts, domain) for (C, opts) in specs if enabled(C, opts))

"The dynamical core of a domain: `:boussinesq` or `:anelastic`."
core(d::Domain) = d.state isa BoussinesqSimulation ? :boussinesq : :anelastic
"The grid of a domain."
grid(d::Domain) = d.state.mesh
"The model time of a domain (or of a model: its first domain's)."
clock(d::Domain) = d.state.t

"The component of `d` in `slot`, or `nothing`."
slot_component(d::Domain, s::Symbol) = _in_slot(d.components, s)
_in_slot(::Tuple{}, s) = nothing
_in_slot(cs::Tuple, s) = slot(first(cs)) === s ? first(cs) : _in_slot(Base.tail(cs), s)

"""
    sgs_closure(d::Domain) -> Union{Nothing,AbstractSGSClosure}

The SGS closure the core runs: that of the domain's `:sgs` component, or
`nothing` (no SGS terms).
"""
sgs_closure(d::Domain) = (c = slot_component(d, :sgs); c === nothing ? nothing : c.closure)

"""
    sgs_stress(d::Domain) -> Symbol

The form of the SGS momentum divergence the core runs: the `:sgs`
component's `stress` (`:laplacian` or `:symmetric`), `:laplacian` without
one.
"""
sgs_stress(d::Domain) = (c = slot_component(d, :sgs); c === nothing ? :laplacian : c.stress)

"""
    Model(domain)
    Model(domains::Tuple, nest)

The model: its domains and the nest coupling between them. One domain
needs no coupling (`nest = nothing`); several are a nest tree (ONE-f,
#455), built from the `[[domain]]` graph by [`nested_model`](@ref), whose
`nest` is a [`ModelNest`](@ref). `step!(model, dt)` advances every domain
by `dt` through the stage pipeline: one domain alone, a nest tree with
the children subcycling inside each step of their parent
([`ModelNest`](@ref)).
"""
struct Model{D<:Tuple, N}
    domains::D
    nest::N
    function Model(domains::Tuple, nest = nothing)
        isempty(domains) && throw(ArgumentError("Model: no domain"))
        all(d -> d isa Domain, domains) || throw(ArgumentError("Model: the domains are Domains"))
        length(domains) == 1 || nest !== nothing ||
            throw(ArgumentError("Model: $(length(domains)) domains need their nest coupling (nested_model)"))
        return new{typeof(domains), typeof(nest)}(domains, nest)
    end
end
Model(d::Domain) = Model((d,))

"The domains of a model."
domains(m::Model) = m.domains
clock(m::Model) = clock(first(m.domains))

# ---------------------------------------------------------------------------
# the hooks of the core: stages 6-8 inside every (sub)step
# ---------------------------------------------------------------------------

# Stages 6 and 7 (:tendencies, then :body_force) run in the core's
# body-force hook, after its own tendencies (advection, diffusion, SGS,
# buoyancy) and before the predictor; stage 8 (:post_predictor) in its
# post-predictor hook. A callable struct, so the core calls it statically.
struct _ForceHook{D} <: Function
    d::D
end
@inline (h::_ForceHook)(sim, Δt) =
    (cs = h.d.components; _count_substage!(h.d.budgets, sim);
     _run_in_core(cs, h.d, Val(:tendencies), Δt);
     _run_in_core(cs, h.d, Val(:body_force), Δt); nothing)
struct _PostHook{D} <: Function
    d::D
end
@inline (h::_PostHook)(sim, Δt) = (_run_in_core(h.d.components, h.d, Val(:post_predictor), Δt); nothing)

_any_attached(cs, stages) = any(c -> any(s -> attaches(c, s), stages), cs)

# --- the hooks in the fused sweep (#533) ------------------------------------
# The body-force hook is a tuple of sweep terms when every component of
# stages 6 and 7 has one (`FiniteVolume.sweep_term(c, d)`), in the
# pipeline's order (:tendencies, then :body_force, each in the domain's
# component order). With the budgets on, the tuple starts with the
# domain's `SweepBudget` (it counts the substages and books the in-core
# components), and the term of each attributed component is a
# `FiniteVolume.BookedTerm` (plan step 7). Otherwise `nothing`, and the
# core runs the hook.
function Simulations.hook_terms(h::_ForceHook, sim)
    d = h.d
    bk = d.budgets === nothing ? nothing : _sweep_budget(d.budgets, d, sim)
    a = _stage_terms(d.components, d, Val(:tendencies), 1, bk)
    a === nothing && return nothing
    b = _stage_terms(d.components, d, Val(:body_force), 2, bk)
    b === nothing && return nothing
    return bk === nothing ? (a..., b...) : (bk, a..., b...)
end
_stage_terms(::Tuple{}, d, ::Val, si, bk) = ()
function _stage_terms(cs::Tuple, d, v::Val{S}, si, bk) where {S}
    rest = _stage_terms(Base.tail(cs), d, v, si, bk)
    c = first(cs)
    attaches(c, S) || return rest
    t = FiniteVolume.sweep_term(c, d)
    (t === nothing || rest === nothing) && return nothing
    nc = length(d.components)
    return (_booked_term(t, bk, c, v, _sweep_slot(si, nc - length(cs) + 1, nc)), rest...)
end
function Simulations.hook_blockers(h::_ForceHook, sim)
    out = Pair{String,String}[]
    for c in h.d.components
        (attaches(c, :tendencies) || attaches(c, :body_force)) && FiniteVolume.sweep_term(c, h.d) === nothing &&
            push!(out, Simulations._blocker(sweep_type(c)))
    end
    return out
end
# the post-predictor stage: no operator runs in the sweep before plan step 6;
# its components are reported
function Simulations.post_hook_blockers(h::_PostHook, sim)
    out = [Simulations._blocker("builtin.post_predictor")]
    for c in h.d.components
        attaches(c, :post_predictor) && push!(out, Simulations._blocker(sweep_type(c)))
    end
    return out
end
"""
    sweep_type(c) -> Type

The type whose [`sweep_term`](@ref Aeolus.Operators.FiniteVolume.sweep_term)
and documented reason (`Simulations.SWEEP_FALLBACKS`) stand for component
`c` in the fused sweep (#533): its own type, or the wrapped coupling's
for a wrapper (the urban site's `UrbanEntry`).
"""
sweep_type(c) = typeof(c)

"""
    sweep_report(d::Domain) -> Dict{String,Any}

The `results.toml` table `[loop_fusion.<id>]` of the step core of `d`:
its setting and whether its substages ran fused (#533,
`Simulations.sweep_report`).
"""
sweep_report(d::Domain) = Simulations.sweep_report(d.state)

"""
    core_hooks(d::Domain) -> NamedTuple

The keywords of the core step that run stages 6–8: `body_force!` and
`post_predictor!` (`nothing` where no component attaches, which keeps the
core's hook-free fused sweep; with the budgets on the body-force hook is
always there, it counts the substages), `sgs` (the `:sgs` component's
closure) and `sgs_stress` (its momentum divergence, [`sgs_stress`](@ref)).
The Boussinesq Euler step also takes `momentum_advection` (the
`:advection` component's operator).
"""
function core_hooks(d::Domain)
    cs = d.components
    bf = d.budgets !== nothing || _any_attached(cs, (:tendencies, :body_force)) ? _ForceHook(d) : nothing
    pp = _any_attached(cs, (:post_predictor,)) ? _PostHook(d) : nothing
    return (; body_force! = bf, post_predictor! = pp, sgs = sgs_closure(d), sgs_stress = sgs_stress(d))
end

# ---------------------------------------------------------------------------
# one step
# ---------------------------------------------------------------------------

# Stage 1's drivers: the open faces' boundary window around the step start
# (the blend extrapolates linearly over a step that crosses a record).
function sync_drivers!(d::Domain)
    f = get(d.drivers, :boundary, nothing)
    f === nothing || advance_feed!(f, clock(d))
    return d
end

function _run_stages!(d::Domain, names, dt, t0, t1)
    for S in names
        _run_each(d.components, d, Val(S), dt, t0, t1)
    end
    return d
end

# Two hooks of the core called in turn (a nest's hook and the domain's).
struct _Chained{A,B} <: Function
    a::A
    b::B
end
@inline (c::_Chained)(sim, Δt) = (c.a(sim, Δt); c.b(sim, Δt); nothing)
_chain(::Nothing, ::Nothing) = nothing
_chain(a, ::Nothing) = a
_chain(::Nothing, b) = b
_chain(a, b) = _Chained(a, b)
# a nest's hook chained with the domain's: both
Simulations.hook_terms(h::_Chained, sim) =
    (a = Simulations.hook_terms(h.a, sim); b = Simulations.hook_terms(h.b, sim);
     a === nothing || b === nothing ? nothing : (a..., b...))
Simulations.hook_blockers(h::_Chained, sim) =
    vcat(Simulations.hook_blockers(h.a, sim), Simulations.hook_blockers(h.b, sim))
Simulations.post_hook_blockers(h::_Chained, sim) =
    unique!(vcat(Simulations.post_hook_blockers(h.a, sim), Simulations.post_hook_blockers(h.b, sim)))

# The pipeline's hooks with a nest's (ModelNest): the nest's body-force
# hook (the flux registers) before the domain's, its post-predictor hook
# (the nest sponge) after the domain's, as `Nesting.DomainAdvance` chains
# them. Without them, exactly `core_hooks(d)`.
function _hooks(d::Domain, bf, pp)
    h = core_hooks(d)
    bf === nothing && pp === nothing && return h
    return (; body_force! = _chain(bf, h.body_force!), post_predictor! = _chain(h.post_predictor!, pp), sgs = h.sgs)
end

# the core step of stages 6-10 with the pipeline's hooks (and a nest's)
function _core_step!(sim::BoussinesqSimulation, d::Domain, dt, bf = nothing, pp = nothing)
    h = _hooks(d, bf, pp)
    if d.scheme === :rk3
        step_rk3!(sim, dt; h...)
    else
        step!(sim, dt; momentum_advection = momentum_advection(slot_component(d, :advection)), h...)
    end
    return sim
end
function _core_step!(sim::MassWeightedSimulation, d::Domain, dt, bf = nothing, pp = nothing)
    d.scheme === :rk3 ? step_rk3!(sim, dt; _hooks(d, bf, pp)...) : step!(sim, dt; _hooks(d, bf, pp)...)
    return sim
end

# ---------------------------------------------------------------------------
# a component that takes the core step (ONE-k, #474)
# ---------------------------------------------------------------------------

"""
    takes_core_step(c) -> Bool

Whether component `c` takes its domain's core step itself
([`advance_core!`](@ref)) and picks its length ([`next_dt`](@ref)),
instead of the domain's `scheme` and `cfl`: the urban site's time
stepper (fixed, adaptive with retried pieces, or IMEX; ONE-k, #474).
Default `false`. A domain holds at most one such component; the first
in its order is used.
"""
takes_core_step(::AbstractComponent) = false

"""
    advance_core!(c, d::Domain, dt)

The core step of `d` over `dt` (stages 6–10) taken by `c`, a component
that [`takes_core_step`](@ref); it calls the core's step with the
pipeline's hooks of [`core_hooks`](@ref) (or the equivalent).
"""
function advance_core! end

"""
    next_dt(c, d::Domain) -> dt

The length of the next step of `d` that `c`, the component that
[`takes_core_step`](@ref), picks: a fixed `dt` or its own stability
limit.
"""
function next_dt end

"""
    check_dt(c, d::Domain, dt)

Throw an `ArgumentError` when `c` cannot take an explicit step of `dt`
on `d` (`step!(d, dt)`, before any stage runs); default: nothing. The
urban site's adaptive stepper refuses a `dt` above its stability limit
(#392).
"""
check_dt(::AbstractComponent, d, dt) = nothing

# the component that takes the core step, or nothing (statically: recursion over the tuple)
_core_taker(::Tuple{}) = nothing
_core_taker(cs::Tuple) = takes_core_step(first(cs)) ? first(cs) : _core_taker(Base.tail(cs))
"The component of `d` that [`takes_core_step`](@ref), or `nothing`."
core_taker(d::Domain) = _core_taker(d.components)

"""
    last_step_dt(dt, remaining) -> dt

The step to take when `remaining` seconds are left: `dt`, or all of
`remaining` when it is no longer than `dt`, or half of it when it is
shorter than `2dt` (so a run does not end on a sliver of a step).
"""
last_step_dt(dt, remaining) = remaining <= dt ? remaining : remaining < 2dt ? remaining / 2 : dt

"""
    step!(d::Domain, dt; checked = true) -> d
    step!(m::Model, dt; checked = true) -> m

One fixed step of `dt` through the pipeline ([`PIPELINE`](@ref)): the
drivers advance (stage 1), the stages before the core run, the core
steps (`step!` or `step_rk3!`, or [`advance_core!`](@ref) of the
component that [`takes_core_step`](@ref)) with the components of stages
6–8 called from inside it, then the stages after the core. With
`checked`, every component's [`check_dt`](@ref) is asked first.
"""
function step!(d::Domain, dt::Real; checked::Bool = true)
    d.cfl === nothing || throw(ArgumentError("step!: the domain $(d.id) steps adaptively (step_adaptive!)"))
    checked && foreach(c -> check_dt(c, d, dt), d.components)
    sync_drivers!(d)
    t0 = clock(d)
    _budgets_begin!(d.budgets, d)
    _run_stages!(d, PRE_CORE, dt, t0, t0 + dt)
    _dynamics_open!(d.budgets, d)
    st = _core_taker(d.components)
    if st === nothing
        _core_step!(d.state, d, dt)
        _dynamics_close!(d.budgets, d)
        _post_core!(d, dt, t0, clock(d))
    else
        advance_core!(st, d, dt)
        _dynamics_close!(d.budgets, d)
        # the stages after the core see the step the component was given
        # (its pieces may sum to it only to round-off)
        _post_core!(d, dt, t0, t0 + dt)
    end
    return d
end

# stages 11-13: the budgets close after :budgets, before :checks and :output
function _post_core!(d::Domain, dt, t0, t1)
    if d.budgets === nothing
        _run_stages!(d, POST_CORE, dt, t0, t1)
    else
        _run_stages!(d, _POST_CORE_BUDGETS, dt, t0, t1)
        _budgets_end!(d.budgets, d, dt)
        _run_stages!(d, _POST_BUDGETS, dt, t0, t1)
    end
    return d
end
step!(m::Model{<:Tuple,Nothing}, dt::Real; checked::Bool = true) =
    (foreach(d -> step!(d, dt; checked), m.domains); m)

"""
    pick_dt(m::Model; t_end = nothing) -> dt

The length of the next step of a single-domain model whose domain has a
component that [`takes_core_step`](@ref): its [`next_dt`](@ref),
shortened by [`last_step_dt`](@ref) to land on `t_end`.
"""
function pick_dt(m::Model{<:Tuple,Nothing}; t_end = nothing)
    d = only(m.domains)
    st = _core_taker(d.components)
    st === nothing && throw(ArgumentError("pick_dt: no component of the domain $(d.id) takes the core step " *
                                          "(step!(model, dt) or step_adaptive!(model))"))
    dt = next_dt(st, d)
    return t_end === nothing ? dt : last_step_dt(dt, t_end - clock(d))
end

"""
    step!(m::Model; t_end = nothing) -> m

One step of the length the model's domain picks ([`pick_dt`](@ref)),
landing on `t_end`.
"""
step!(m::Model{<:Tuple,Nothing}; t_end = nothing) = step!(m, pick_dt(m; t_end); checked = false)

"""
    run_to!(m::Model, t_end; callback = nothing) -> m

Run a single-domain model whose step a component takes
([`takes_core_step`](@ref)) to the model time `t_end`, as the run loop of
`aeolus run` does but without its output, monitor and checkpoints: every
component's [`run_begin!`](@ref), `step!(m)` while the stepping component
[`run_continues`](@ref) (`callback(m, n)` after the `n`-th step), then
every [`run_end!`](@ref), also when a step throws.
"""
function run_to!(m::Model{<:Tuple,Nothing}, t_end::Real; callback = nothing)
    d = only(m.domains)
    st = _core_taker(d.components)
    st === nothing && throw(ArgumentError("run_to!: no component of the domain $(d.id) takes the core step"))
    foreach(c -> run_begin!(c, d, Float64(t_end)), d.components)
    n = 0
    try
        while run_continues(st, d)
            step!(m)
            n += 1
            callback === nothing || callback(m, n)
        end
    finally
        foreach(c -> run_end!(c, d), d.components)
    end
    return m
end

"""
    component_limits(d::Domain) -> Tuple

The explicit stability limits of `d`'s components ([`limit_dt`](@ref)),
one entry per component (`nothing`: none).
"""
component_limits(d::Domain) = map(c -> limit_dt(c, d), d.components)

"""
    step_adaptive!(d::Domain; dt_prev = nothing, t_end = nothing) -> (state, dt, retries)
    step_adaptive!(m::Model; …)

One adaptive step: the drivers advance, the step is picked from the
core's CFL limits and the components' ([`component_limits`](@ref),
`TimeIntegration.adaptive_dt`), shortened to land on `t_end`, then the
stages before the core run with it, the core takes it with
`TimeIntegration.step_adaptive!` (which may retry with half the step; the
stages before the core are not repeated), and the stages after the core
run with the step taken.
"""
function step_adaptive!(d::Domain; dt_prev = nothing, t_end = nothing)
    c = d.cfl
    c === nothing && throw(ArgumentError("step_adaptive!: the domain $(d.id) takes a fixed step (cfl = nothing)"))
    sync_drivers!(d)
    sim = d.state
    closure = sgs_closure(d); stress = sgs_stress(d)
    dt = adaptive_dt(sim, c; dt_prev = dt_prev, sgs = closure, sgs_stress = stress, limits = component_limits(d))
    t_end === nothing || (dt = min(dt, t_end - sim.t))
    t0 = clock(d)
    _budgets_begin!(d.budgets, d)
    _run_stages!(d, PRE_CORE, dt, t0, t0 + dt)
    h = core_hooks(d)
    stepper = d.scheme === :rk3 ? ((s, Δ; kw...) -> step_rk3!(s, Δ; kw..., sgs = closure, sgs_stress = stress)) : nothing
    _dynamics_open!(d.budgets, d)
    res = step_adaptive!(sim, c; dt = dt, sgs = closure, sgs_stress = stress, stepper = stepper,
                                         body_force! = h.body_force!, post_predictor! = h.post_predictor!)
    _dynamics_close!(d.budgets, d)
    _post_core!(d, res[2], t0, clock(d))
    return res
end
step_adaptive!(m::Model{<:Tuple,Nothing}; kw...) = step_adaptive!(only(m.domains); kw...)
step_adaptive!(m::Model; kw...) =
    throw(ArgumentError("step_adaptive!: a nested model takes fixed steps (step!(model, dt)); its children subcycle"))

# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

"""
    checkpoint_components(d::Domain) -> NamedTuple

`component_name => component` of every component of `d` (and `budgets`,
the [`DomainBudgets`](@ref), when they are on): what a checkpoint stores
through `checkpoint_state` (IO's component contract) and a restart
restores.
"""
function checkpoint_components(d::Domain)
    cs = NamedTuple{map(component_name, d.components)}(d.components)
    return d.budgets === nothing ? cs : merge(cs, (; budgets = d.budgets))
end
