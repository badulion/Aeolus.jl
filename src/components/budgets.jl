# The budgets of a domain (ONE-c, #452): the stage-context measurement of
# `Aeolus.BudgetSystem` wired into the pipeline. With budgets on, every
# component's stage entry is an attribution context of every measured
# quantity, and the core step is one (the dynamics), so each budget closes
# by construction and names what each component did. With budgets off
# (`d.budgets === nothing`, the default) the pipeline is unchanged: no
# measurement, and the core keeps its hook-free fast path.

"The quantities a domain's budgets can measure."
const BUDGET_QUANTITIES = (:energy, :water, :momentum_x, :momentum_y, :tracer)

"""
    DomainBudgets

The measured budgets of a [`Domain`](@ref): a
[`Budgets`](@ref Aeolus.BudgetSystem.Budgets) with one
[`StageBudget`](@ref Aeolus.BudgetSystem.StageBudget) per measured
quantity, the fields each is measured on (`state`; a tuple for the water
of the air: vapour and what the components carry) and the core tendency
that the components inside the core add to (`tendency`, or `nothing`),
and the count of the core's in-core hook calls of the current step (to
weight the RK3 substages and to notice a retried step), and `hour`, the
function of the model time that picks a step's hour-of-day bin
(`nothing`: one bin). Built by [`with_budgets`](@ref), or around a
`Budgets` of its own by `DomainBudgets(budgets, quantities, state,
tendency; nsub, hour)`.
"""
mutable struct DomainBudgets
    budgets::Budgets
    quantities::Vector{Symbol}
    state::Dict{Symbol,Any}
    tendency::Dict{Symbol,Any}
    calls::Int
    t0::Float64
    nsub::Int
    hour::Any
    exchange::Union{Nothing,Tuple{Symbol,Symbol}}
    sweep::Any       # the budget term of the fused sweep (`SweepBudget`, #533), built at its first use
end
DomainBudgets(b::Budgets, quantities, state, tendency; nsub::Integer = 1, hour = nothing, exchange = nothing) =
    DomainBudgets(b, collect(Symbol, quantities), Dict{Symbol,Any}(state), Dict{Symbol,Any}(tendency),
                  0, 0.0, Int(nsub), hour, exchange)
DomainBudgets(b, quantities, state, tendency, calls, t0, nsub, hour, exchange) =
    DomainBudgets(b, quantities, state, tendency, calls, t0, nsub, hour, exchange, nothing)

"""
    NEST_EXCHANGE

The key `(:nest_feedback, :nest_exchange)` of the nest-exchange term of a
nested domain's budgets (NEST-p2, #482; `[nesting] budget_exchange =
"exchange_term"`): one term per measured quantity holding everything the
nest did to the domain — what crossed its nest faces in its dynamics (the
column total of the transport, for the conserved scalars: energy, water,
tracers), what the window opening did between its steps (the two-way
child's re-projection onto the new boundary data) and, in a parent, the
two-way feedback of its children (stage 12). Each domain's budget closes
with it; [`nest_exchange`](@ref) reads it.
"""
const NEST_EXCHANGE = (:nest_feedback, :nest_exchange)

# the quantities whose dynamics' column total is the nest exchange (the
# transport conserves them; for momentum the column total also holds the
# pressure force and the walls' friction)
const _EXCHANGE_SPLIT = (:energy, :water, :tracer)

"""
    budget_fields(c, q::Symbol) -> Tuple

The fields of the measured quantity `q` (`:water`, `:tracer`) that
component `c` carries: its cloud water and rain for the water of the air,
its tracer for `:tracer`. Default: none.
"""
budget_fields(::AbstractComponent, q::Symbol) = ()

"""
    solid_mask(c) -> Union{Nothing,AbstractArray{Bool,3}}

The halo-padded solid mask of the obstacles `c` carries (the budgets leave
the solid cells out of the air), or `nothing` (the default).
"""
solid_mask(::AbstractComponent) = nothing

# the reference profiles the budgets weight with: the anelastic base state,
# or the Boussinesq core's constant density ρ₀ = p_ref / (R_d θ₀) with Π̄ = 1
function _reference_profiles(sim::MassWeightedSimulation)
    r = sim.ref
    return (ρ̄ = Float64.(r.ρ̄), π̄ = Float64.(r.π̄), cp = Float64(r.cp))
end
function _reference_profiles(sim::BoussinesqSimulation)
    c = DEFAULT_CONSTANTS
    θ₀ = Float64(sim.θ₀)
    θ₀ > 0 || throw(ArgumentError("budgets: the Boussinesq core needs θ₀ > 0 for its reference density (dynamics.theta0)"))
    Nz = sim.mesh.Nz
    return (ρ̄ = fill(c.p_ref / (R_d(c) * θ₀), Nz), π̄ = ones(Nz), cp = c.cp_dry)
end

_first_solid(::Tuple{}) = nothing
_first_solid(cs::Tuple) = (m = solid_mask(first(cs)); m === nothing ? _first_solid(Base.tail(cs)) : m)

"""
    with_budgets(d::Domain, quantities) -> Domain

`d` with its budgets on for `quantities` (a subset of
[`BUDGET_QUANTITIES`](@ref)), started from the current state:

| quantity | measured | weight | core tendency |
| :-- | :-- | :-- | :-- |
| `:energy` | θ: the heat of the air ``Σ ρ̄ c_p Π̄ θ ΔV / A`` (J m⁻²) | ρ̄ c_p Π̄ | `tθ` |
| `:water` | qv and the components' water (kg m⁻²) | ρ̄ | `tqv` |
| `:momentum_x`, `:momentum_y` | u, v (kg m⁻¹ s⁻¹) | ρ̄ | `tu`, `tv` |
| `:tracer` | the components' tracer (kg m⁻² per unit mixing ratio) | ρ̄ | — |

ρ̄, Π̄ and c_p are the anelastic base state, or on the Boussinesq core its
constant ``ρ_0 = p_{ref} / (R_d θ_0)`` with ``Π̄ = 1`` (the Boussinesq
transport conserves ``Σ θ``, so the energy budget has no redistribution
term). Solid cells of the components' obstacles are left out of the air.
"""
function with_budgets(d::Domain, quantities)
    qs = collect(Symbol, quantities)
    isempty(qs) && return d
    all(q -> q in BUDGET_QUANTITIES, qs) ||
        throw(ArgumentError("with_budgets: the quantities are $(BUDGET_QUANTITIES), got $(qs)"))
    sim = d.state; mesh = sim.mesh
    ref = _reference_profiles(sim)
    W = transport_conserved_weight(sim)
    solid = _first_solid(d.components)
    Nz = mesh.Nz; one_ = ones(Nz)
    b = Budgets(; bins = 1, order = _STAGE_NAMES)
    empty!(b.measured)
    b.ρ = copy(ref.ρ̄)
    state = Dict{Symbol,Any}(); tend = Dict{Symbol,Any}()
    comp_fields(q) = Tuple(f for c in d.components for f in budget_fields(c, q))
    for q in qs
        if q === :energy
            m = LevelMeter(mesh, ref.ρ̄, ref.π̄, ref.cp, solid; W = W)
            state[q] = sim.θ; tend[q] = sim.tθ
        elseif q === :water
            hasproperty(sim, :has_moisture) && sim.has_moisture ||
                throw(ArgumentError("with_budgets: the water budget needs moisture (dynamics.moisture)"))
            m = LevelMeter(mesh, ref.ρ̄, one_, 1.0, solid; W = W)
            state[q] = (sim.qv, comp_fields(:water)...); tend[q] = sim.tqv
        elseif q === :tracer
            fs = comp_fields(:tracer)
            isempty(fs) && throw(ArgumentError("with_budgets: no component carries a tracer"))
            m = LevelMeter(mesh, ref.ρ̄, one_, 1.0, solid; W = W)
            state[q] = length(fs) == 1 ? only(fs) : fs; tend[q] = nothing
        else
            m = LevelMeter(mesh, ref.ρ̄, one_, 1.0, nothing; W = W)
            state[q] = q === :momentum_x ? sim.u : sim.v
            tend[q] = q === :momentum_x ? sim.tu : sim.tv
        end
        sb = StageBudget(; bins = 1, order = _STAGE_NAMES)
        budget_start!(sb, m)
        b.measured[q] = sb
    end
    db = DomainBudgets(b, qs, state, tend, 0, 0.0, d.scheme === :rk3 ? 3 : 1, nothing, nothing)
    return Domain(d.id, d.state, d.scheme, d.cfl, d.drivers, d.components, db)
end

"The budgets of a domain (a `Budgets`), or `nothing` when they are off."
budgets(d::Domain) = d.budgets === nothing ? nothing : d.budgets.budgets

# --- the step: begin, contexts, end -------------------------------------

_budgets_begin!(::Nothing, d) = nothing
function _budgets_begin!(b::DomainBudgets, d)
    h = b.hour === nothing ? 0.0 : Float64(b.hour(clock(d)))
    gap = b.exchange === nothing ? (:output, :between_steps) : b.exchange
    for q in b.quantities
        budget_begin_step!(b.budgets.measured[q], b.state[q], h; gap = gap)
    end
    return nothing
end

_budgets_end!(::Nothing, d, dt) = nothing
function _budgets_end!(b::DomainBudgets, d, dt)
    bs = b.budgets
    bs.dt = dt
    foreach(c -> budget!(c, d, bs), d.components)
    for q in b.quantities
        budget_end_step!(bs.measured[q], b.state[q], dt)
    end
    bs.elapsed += dt
    bs.nsteps += 1
    return nothing
end

"""
    attributed(c, ::Val{stage}) -> Bool

Whether the budgets measure `c`'s entry in `stage` as a context of its
own (default `true`). `false` for an entry that books what it does itself
inside the dynamics' context (`add_heat!`), so that it is not counted
twice: the urban site's open faces, whose θ relaxation is the dynamics'
term `(:dynamics, :relaxation)`; and for an entry that only reads the
state (the urban site's record and sanity check), which books no term.
"""
attributed(c, ::Val) = true

# a component's stage entry outside the core: a context on the state
@inline _attributed!(::Nothing, c, d, v::Val, dt) = run_stage!(c, d, v, dt)
function _attributed!(b::DomainBudgets, c, d, v::Val{S}, dt) where {S}
    attributed(c, v) || return run_stage!(c, d, v, dt)
    n = component_name(c)
    for q in b.quantities
        budget_open!(b.budgets.measured[q], S, n, b.state[q])
    end
    run_stage!(c, d, v, dt)
    for q in Iterators.reverse(b.quantities)
        budget_close!(b.budgets.measured[q], b.state[q])
    end
    return nothing
end

# the weight of the current substage's effect on the step's end state
_substage_weight(b::DomainBudgets) = b.nsub == 3 ? (1 / 6, 1 / 6, 2 / 3)[mod1(b.calls, 3)] : 1.0

# a component's stage entry inside the core: :tendencies and :body_force add
# to the core's tendencies, which the predictor applies over Δt (measured on
# the tendency, scaled by the substage's weight × Δt); :post_predictor
# changes the predicted state (measured on it, scaled by the weight)
@inline _attributed_in_core!(::Nothing, c, d, v::Val, Δt) = run_stage!(c, d, v, Δt)
function _attributed_in_core!(b::DomainBudgets, c, d, v::Val{S}, Δt) where {S}
    attributed(c, v) || return run_stage!(c, d, v, Δt)
    n = component_name(c)
    w = _substage_weight(b)
    ontend = S !== :post_predictor
    for q in b.quantities
        x = ontend ? b.tendency[q] : b.state[q]
        x === nothing && continue
        budget_open!(b.budgets.measured[q], S, n, x; scale = ontend ? w * Δt : w)
    end
    run_stage!(c, d, v, Δt)
    for q in Iterators.reverse(b.quantities)
        x = ontend ? b.tendency[q] : b.state[q]
        x === nothing && continue
        budget_close!(b.budgets.measured[q], x)
    end
    return nothing
end

# the first in-core hook of every (sub)step: count the substages. Only the
# first substage of an attempt starts at the step's start time (the RK3
# substages 2 and 3 at t + Δt and t + 2Δt), so a later call at that time is
# the core retrying a step it rejected (or that threw): what the attempt's
# in-core stages booked is dropped
_count_substage!(::Nothing, sim) = nothing
function _count_substage!(b::DomainBudgets, sim)
    if b.calls > 0 && sim.t == b.t0
        foreach(q -> budget_discard!(b.budgets.measured[q]), b.quantities)
        b.calls = 0
    end
    b.calls += 1
    return nothing
end

# --- the in-core components inside the fused sweep (#533 plan step 7) -------
# With the budgets on and every in-core term fusable, the core runs no
# body-force hook: the sweep's term tuple starts with the domain's
# `SweepBudget`, which counts the substage before the sweep (the hook's
# `_count_substage!`) and books after it what each attributed component's
# term (a `FiniteVolume.BookedTerm`) added to the tendencies, per level,
# summed in the sweep: the same bookings as `_attributed_in_core!`'s
# contexts on the whole tendency fields, without their two passes per
# component and quantity.

"""
    SweepBudget

The budget term of a domain's fused sweep (#533 plan step 7): its
`prepare!` counts the substage (the body-force hook's work with the
budgets on), its `finish_sweep!` books what each attributed in-core
component added to the core tendencies during the sweep, summed per level
by its `FiniteVolume.BookedTerm` into `sums`, as the attribution context
`(stage, component)` of every measured quantity with a core tendency
([`book_context!`](@ref Aeolus.BudgetSystem.book_context!), scaled by the
substage's weight × Δt as `_attributed_in_core!` scales its contexts).
Adds nothing to the tendencies. Built once per domain, at the first
substage that asks for the sweep's terms (`slots`: the `(stage, name)` of
each slot, `active` the slots of every attributed in-core entry, whether
or not its component has a sweep term yet: a slot whose component added
nothing sums to 0 and books nothing; `quantity[f]` the budget of booked
field `f`).
"""
struct SweepBudget{L<:FiniteVolume.LevelSums,C} <: FiniteVolume.SweepTerm
    b::DomainBudgets
    sums::L
    components::C
    slots::Vector{Tuple{Symbol,Symbol}}
    active::Vector{Int}
    quantity::NTuple{4,Symbol}
    rest::Vector{Float64}
end
FiniteVolume.footprint(::SweepBudget) = (0, 0, 0)
@inline FiniteVolume.add_row!(::SweepBudget, buf, a, b, s, j, k, ii, st) = nothing
function FiniteVolume.prepare!(t::SweepBudget, sim, Δt)
    _count_substage!(t.b, sim)
    fill!(t.sums.acc, 0.0)
    return nothing
end
function FiniteVolume.finish_sweep!(t::SweepBudget, sim, Δt)
    scale = _substage_weight(t.b) * Δt
    acc = t.sums.acc; rest = t.rest
    for slot in t.active, f in 1:4
        t.sums.on[f] || continue
        sb = t.b.budgets.measured[t.quantity[f]]
        w = sb.meter.w
        @inbounds for k in eachindex(rest)
            rest[k] = scale * w[k] * acc[f, slot, k]
        end
        stage, name = t.slots[slot]
        book_context!(sb, stage, name, rest)
    end
    return nothing
end

# the slot of the term of component number `pos` in stage `si` (1:
# :tendencies, 2: :body_force) of `nc` components
_sweep_slot(si, pos, nc) = (si - 1) * nc + pos

# the domain's budget term, built at the first fused substage (and again
# if the domain's components changed); a concrete type for the hook terms
function _sweep_budget(b::DomainBudgets, d::Domain, sim)
    S = SweepBudget{FiniteVolume.LevelSums{fieldtype(LevelMeter, :fluid)},typeof(d.components)}
    c = b.sweep
    c isa S && c.components === d.components && return c
    return _build_sweep_budget(b, d, sim)::S
end
@noinline function _build_sweep_budget(b::DomainBudgets, d::Domain, sim)
    cs = d.components; nc = length(cs)
    tends = (sim.tu, sim.tv, sim.tθ, sim.has_moisture ? sim.tqv : nothing)
    on = fill(false, 4); masked = fill(false, 4); quantity = fill(:none, 4)
    fluid = nothing
    for q in b.quantities
        x = b.tendency[q]
        x === nothing && continue
        f = findfirst(t -> t === x, tends)
        f === nothing && throw(ArgumentError("budgets: the core tendency of $q is not one the fused sweep books"))
        m = b.budgets.measured[q].meter
        on[f] = true; quantity[f] = q
        masked[f] = any(iszero, m.fluid)
        if masked[f]
            fluid === nothing || fluid == m.fluid ||
                throw(ArgumentError("budgets: the fused sweep books with one solid mask; $q's differs"))
            fluid = m.fluid
        end
    end
    fluid === nothing && (fluid = first(values(b.budgets.measured)).meter.fluid)
    slots = fill((:none, :none), 2nc); active = Int[]
    for (si, S) in enumerate((:tendencies, :body_force)), (pos, c) in enumerate(cs)
        # the slot set is the one `_booked_term` wraps (not decided by
        # `sweep_term`, which may return a term only later)
        attaches(c, S) && attributed(c, Val(S)) || continue
        slot = _sweep_slot(si, pos, nc)
        slots[slot] = (S, component_name(c)); push!(active, slot)
    end
    g = sim.mesh
    sums = FiniteVolume.LevelSums(2nc, g.Nz, Tuple(on), Tuple(masked), fluid, g.Hx, g.Hy, g.Hz)
    t = SweepBudget(b, sums, cs, slots, active, Tuple(quantity), zeros(g.Nz))
    b.sweep = t
    return t
end

# the term of an in-core component inside the sweep: booked when the budgets
# attribute its entry
@inline _booked_term(t, ::Nothing, c, v::Val, slot) = t
@inline _booked_term(t, bk::SweepBudget, c, v::Val, slot) =
    attributed(c, v) ? FiniteVolume.BookedTerm(t, bk.sums, slot) : t

# the core step: one deferring context per quantity (the dynamics)
_dynamics_open!(::Nothing, d) = nothing
function _dynamics_open!(b::DomainBudgets, d)
    b.calls = 0
    b.t0 = clock(d)
    for q in b.quantities
        budget_open!(b.budgets.measured[q], :dynamics, :core, b.state[q]; split = true, defer = true)
    end
    return nothing
end
_dynamics_close!(::Nothing, d) = nothing
function _dynamics_close!(b::DomainBudgets, d)
    for q in Iterators.reverse(b.quantities)
        budget_close!(b.budgets.measured[q], b.state[q];
                      exchange = q in _EXCHANGE_SPLIT ? b.exchange : nothing)
    end
    return nothing
end

# checkpoints: the accumulated budgets restart with the run (#269)
checkpoint_state(b::DomainBudgets) = (budgets = b.budgets,)
function restore_checkpoint_state!(b::DomainBudgets, s)
    for q in b.quantities
        src = s.budgets.measured[q]; dst = b.budgets.measured[q]
        for f in fieldnames(StageBudget)
            f === :meter || setfield!(dst, f, deepcopy(getfield(src, f)))
        end
    end
    for f in fieldnames(Budgets)
        f === :measured || setfield!(b.budgets, f, deepcopy(getfield(s.budgets, f)))
    end
    return b
end

"""
    nest_exchange(d::Domain, q::Symbol) -> Float64

The nest-exchange term of quantity `q` of the nested domain `d`'s
budgets over the run so far ([`NEST_EXCHANGE`](@ref); per unit ground
area of the domain, J m⁻² for the energy), `0.0` when nothing was booked.
"""
function nest_exchange(d::Domain, q::Symbol)
    d.budgets === nothing && throw(ArgumentError("nest_exchange: the domain $(d.id) measures no budgets"))
    return get(stage_budget(d.budgets.budgets.measured[q]).terms, NEST_EXCHANGE, 0.0)
end
