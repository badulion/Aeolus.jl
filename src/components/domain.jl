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
    (`step_rk3!`, Boussinesq);
  * `cfl` — `nothing` for a fixed step, or the `CFLConstraints` of an
    adaptive one (`step_adaptive!(domain)`);
  * `drivers` — its input drivers: `boundary`, the
    `FileBoundaryFeed` of open lateral faces, advanced at the start of
    every step;
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
    scheme === :rk3 && state isa MassWeightedSimulation &&
        throw(ArgumentError("Domain: the anelastic core steps with :euler"))
    cfl === nothing || cfl isa CFLConstraints || throw(ArgumentError("Domain: cfl is nothing or a CFLConstraints"))
    cfl !== nothing && state isa MassWeightedSimulation &&
        throw(ArgumentError("Domain: the anelastic core takes a fixed dt (cfl = nothing)"))
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
    Model(domain)

The model: its domains (today one; several, with their nest couplings,
arrive with ONE-f, #455). `step!(model, dt)` advances every domain by `dt`
through the stage pipeline.
"""
struct Model{D<:Tuple}
    domains::D
    function Model(domains::Tuple)
        length(domains) == 1 ||
            throw(ArgumentError("Model: one domain (several domains, with their nest couplings, arrive with ONE-f, #455)"))
        all(d -> d isa Domain, domains) || throw(ArgumentError("Model: the domains are Domains"))
        return new{typeof(domains)}(domains)
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

"""
    core_hooks(d::Domain) -> NamedTuple

The keywords of the core step that run stages 6–8: `body_force!` and
`post_predictor!` (`nothing` where no component attaches, which keeps the
core's hook-free fast path; with the budgets on the body-force hook is
always there, it counts the substages), `sgs` (the `:sgs` component's
closure) and, for the Boussinesq Euler step, `momentum_advection` (the
`:advection` component's operator).
"""
function core_hooks(d::Domain)
    cs = d.components
    bf = d.budgets !== nothing || _any_attached(cs, (:tendencies, :body_force)) ? _ForceHook(d) : nothing
    pp = _any_attached(cs, (:post_predictor,)) ? _PostHook(d) : nothing
    return (; body_force! = bf, post_predictor! = pp, sgs = sgs_closure(d))
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

# the core step of stages 6-10 with the pipeline's hooks
function _core_step!(sim::BoussinesqSimulation, d::Domain, dt)
    h = core_hooks(d)
    if d.scheme === :rk3
        step_rk3!(sim, dt; h...)
    else
        step!(sim, dt; momentum_advection = momentum_advection(slot_component(d, :advection)), h...)
    end
    return sim
end
_core_step!(sim::MassWeightedSimulation, d::Domain, dt) = (step!(sim, dt; core_hooks(d)...); sim)

"""
    step!(d::Domain, dt) -> d
    step!(m::Model, dt) -> m

One fixed step of `dt` through the pipeline ([`PIPELINE`](@ref)): the
drivers advance (stage 1), the stages before the core run, the core steps
(`step!` or `step_rk3!`) with the components of stages 6–8 called from
inside it, then the stages after the core.
"""
function step!(d::Domain, dt::Real)
    d.cfl === nothing || throw(ArgumentError("step!: the domain $(d.id) steps adaptively (step_adaptive!)"))
    sync_drivers!(d)
    t0 = clock(d)
    _budgets_begin!(d.budgets, d)
    _run_stages!(d, PRE_CORE, dt, t0, t0 + dt)
    _dynamics_open!(d.budgets, d)
    _core_step!(d.state, d, dt)
    _dynamics_close!(d.budgets, d)
    _post_core!(d, dt, t0)
    return d
end

# stages 11-13: the budgets close after :budgets, before :checks and :output
function _post_core!(d::Domain, dt, t0)
    if d.budgets === nothing
        _run_stages!(d, POST_CORE, dt, t0, clock(d))
    else
        _run_stages!(d, _POST_CORE_BUDGETS, dt, t0, clock(d))
        _budgets_end!(d.budgets, d, dt)
        _run_stages!(d, _POST_BUDGETS, dt, t0, clock(d))
    end
    return d
end
step!(m::Model, dt::Real) = (foreach(d -> step!(d, dt), m.domains); m)

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
    closure = sgs_closure(d)
    dt = adaptive_dt(sim, c; dt_prev = dt_prev, sgs = closure, limits = component_limits(d))
    t_end === nothing || (dt = min(dt, t_end - sim.t))
    t0 = clock(d)
    _budgets_begin!(d.budgets, d)
    _run_stages!(d, PRE_CORE, dt, t0, t0 + dt)
    h = core_hooks(d)
    stepper = d.scheme === :rk3 ? ((s, Δ; kw...) -> step_rk3!(s, Δ; kw..., sgs = closure)) : nothing
    _dynamics_open!(d.budgets, d)
    res = step_adaptive!(sim, c; dt = dt, sgs = closure, stepper = stepper,
                                         body_force! = h.body_force!, post_predictor! = h.post_predictor!)
    _dynamics_close!(d.budgets, d)
    _post_core!(d, res[2], t0)
    return res
end
step_adaptive!(m::Model; kw...) = step_adaptive!(only(m.domains); kw...)

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
