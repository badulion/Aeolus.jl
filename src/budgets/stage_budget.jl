# The stage budget: where a quantity of the air came from, step by step.
#
# The mechanism of the urban case's air-heat ledger (#253), moved into the
# core as the one attribution mechanism of every measured budget (ONE-c,
# #452): the heat of the air (the energy budget), the water, the momentum
# and the tracers of a domain. A `StageBudget` closes one quantity
# Q = Σ_k M_k (a `LevelMeter`, budgets/meter.jl) and attributes every
# change of it automatically, without the components reporting anything:
#
#   * every stage entry the step runs and the dynamics (the core step) is
#     an attribution *context*. The budget measures Q (per level) when the
#     context opens and when it closes, and books the difference under
#     (stage, component) — e.g. (:surface_flux, :surface_exchange),
#     (:column_tendency, :column_radiation);
#   * contexts nest: the stages inside the core run inside the dynamics'
#     context, and what they do is taken out of the dynamics' share. A
#     component that adds a *tendency* inside the core is measured on the
#     tendency array, scaled by the length of the step the tendency acts
#     over (`budget_open!(…; scale)`, the model's in-core stages);
#   * the change a context makes can be split into named terms with
#     `book!(budget, term, J)` from inside it (the top heat flux, the
#     open-boundary heat of the nesting epic, #246). The rest is booked
#     under the context itself. For the dynamics the rest is split by
#     `split_transport` into (:dynamics, :transport) — what the transport
#     conserves; over the whole column it is the numerics residual — and
#     (:dynamics, :redistribution). What the dynamics moves into or out of
#     solid (building) cells, which are not air, is booked apart as
#     (:dynamics, :solid_exchange): the core's transport and diffusion do
#     not stop at the immersed boundary, so the solid cells' θ evolves and
#     exchanges heat with the air through the numerics alone;
#   * between two contexts at the top level nothing else touches the
#     quantity, so the budget measures only when a context closes (the
#     value at the open is the previous close): one pass over the interior
#     per entry. A change outside any context still cannot hide: the
#     step's end value is compared with the last close (booked as
#     (:core, :unattributed)), and the start of the next step with the end
#     of this one (booked as (:output, :between_steps), the stage that
#     runs after the budget closes a step);
#   * a context whose component cannot change the quantity is skipped
#     (`budget_skip!`, #539; the component's declared writes,
#     `Components.budget_writes`): no pass. At the top level the budget
#     then no longer knows the state between contexts, so the next context
#     that measures, or the end of the step, measures afresh and books any
#     change since the last measurement as (:core, :unattributed): a
#     skipped component that did write cannot hide, and is not attributed
#     to its neighbour. Inside a context (the stages inside the core) a
#     skip is free: what an undeclared write did stays in the enclosing
#     context's own share, the dynamics'. The verify mode of the domain's
#     budgets measures every context anyway and fails on such a write;
#   * a context opened with `defer = true` (the model's core step, which
#     an adaptive step may retry) keeps what its nested contexts book until
#     it closes; `budget_discard!` drops it when the core restarts a step.
#
# Every term is kept per level (so any layer — the canopy, 0–100 m — is
# a weighted sum of the same numbers) and per time bin of the step's start:
# the urban case bins by local hour (`stage_budget(b; window = (20, 5))`
# returns the night integrals of #238), a model domain has one bin.
#
# Nothing here refers to a case: the functions take the budget, the meter
# and the field, so the urban case, its nest parent (#259) and every model
# domain keep their budgets with the same code.

"One open attribution context (a stage entry or the dynamics)."
mutable struct BudgetContext
    stage::Symbol
    name::Symbol
    H0::Vector{Float64}       # per-level content when the context opened
    S0::Vector{Float64}       # per-level content of the solid cells when it opened
    booked::Vector{Float64}   # per level + 1 (unlocated): book! inside it, and nested contexts
    split::Bool               # split the rest into :transport / :redistribution
    scale::Float64            # the measured change is scaled by this (a tendency: the step length)
    defer::Bool               # keep the nested contexts' bookings in `pending` until it closes
    pending::Dict{Tuple{Symbol,Symbol},Vector{Float64}}
end

"""
    StageBudget(; bins = 24, order = ())

The budget of one quantity of the air, attributed to the stage entries of
the steps (see `budgets/stage_budget.jl`). Fields:

* `meter` — the [`LevelMeter`](@ref) (set by [`budget_start!`](@ref));
* `acc[(stage, term)]` — accumulated change of the quantity (per unit
  ground area of the domain: J m⁻² for the heat of the air), as an
  `(Nz + 1) × bins` matrix: row `k ≤ Nz` is level `k`, row `Nz + 1` the
  part booked without a level (`book!` without `k`); column `b` the steps
  of bin `b` (the urban case: the steps that started in local hour
  `b − 1`);
* `storage` — the measured change of the quantity, same layout;
* `elapsed` — seconds per bin;
* `canopy_top` — top (m) of the `:canopy` layer of [`stage_budget`](@ref)
  (the urban case sets it to the tallest roof);
* `order` — the stages in the order the reports list them;
* `active` — true while a step is being attributed;
* `nmeasure` — the passes over the field made since the start (one per
  measurement; the cost of the budget, #539).
"""
mutable struct StageBudget
    meter::Union{Nothing,LevelMeter}
    acc::Dict{Tuple{Symbol,Symbol},Matrix{Float64}}
    storage::Matrix{Float64}
    elapsed::Vector{Float64}
    bin::Int
    H_now::Vector{Float64}
    Hs_now::Vector{Float64}
    sbuf::Vector{Float64}
    H_end::Vector{Float64}
    have_end::Bool
    buf::Vector{Float64}
    active::Bool
    stack::Vector{BudgetContext}
    pool::Vector{BudgetContext}
    canopy_top::Float64
    nmeasure::Int
    nbins::Int
    order::Vector{Symbol}
    stale::Bool               # a top-level context was skipped since the last measurement (#539)
    changed::Bool             # the last closed context measured a change (the verify mode)
end
StageBudget(; bins::Integer = 24, order = ()) =
    StageBudget(nothing, Dict{Tuple{Symbol,Symbol},Matrix{Float64}}(), zeros(0, bins),
                zeros(bins), 1, Float64[], Float64[], Float64[], Float64[], false, Float64[], false,
                BudgetContext[], BudgetContext[], NaN, 0, Int(bins), collect(Symbol, order), false, false)

"""
    budget_start!(b, meter; canopy_top = NaN) -> b

(Re)start the stage budget `b` with `meter`: every accumulated term is
zeroed.
"""
function budget_start!(hl::StageBudget, m::LevelMeter; canopy_top::Real = hl.canopy_top)
    Nz = m.Nz
    hl.meter = m
    empty!(hl.acc)
    hl.storage = zeros(Nz + 1, hl.nbins)
    hl.elapsed = zeros(hl.nbins)
    hl.H_now = zeros(Nz); hl.H_end = zeros(Nz); hl.buf = zeros(Nz)
    hl.Hs_now = zeros(Nz); hl.sbuf = zeros(Nz)
    hl.have_end = false
    hl.active = false
    empty!(hl.stack)
    hl.canopy_top = canopy_top
    hl.nmeasure = 0
    hl.stale = false
    return hl
end

# measure the air's content per level into `out`, and the solid cells' into hl.sbuf
function _measure!(out, hl::StageBudget, θ)
    hl.nmeasure += 1
    measure_levels!(out, hl.meter, θ)
    solid_levels!(hl.sbuf, hl.meter)
    return out
end

function _book!(hl::StageBudget, key::Tuple{Symbol,Symbol}, v::AbstractVector, extra::Real = 0.0)
    # inside a deferring context (the model's core step) the booking waits
    # for that context to close: a retried step drops it (budget_discard!)
    for i in length(hl.stack):-1:1
        ctx = hl.stack[i]
        ctx.defer || continue
        p = get!(() -> zeros(hl.meter.Nz + 1), ctx.pending, key)
        @inbounds for k in eachindex(v)
            p[k] += v[k]
        end
        p[end] += extra
        return hl
    end
    a = get!(() -> zeros(hl.meter.Nz + 1, hl.nbins), hl.acc, key)
    b = hl.bin
    @inbounds for k in eachindex(v)
        a[k, b] += v[k]
    end
    a[end, b] += extra
    return hl
end

"""
    budget_begin_step!(b, x, hour = 0; gap = (:output, :between_steps)) -> b

Start attributing one step: measure the quantity of the field `x` (any
change since the end of the previous step is booked under `gap`,
`(:output, :between_steps)` by default; a nested domain names it its nest
exchange, NEST-p2 #482). With 24 bins the step is binned by its local
`hour`; with one bin `hour` is ignored.
"""
function budget_begin_step!(hl::StageBudget, θ, hour::Real = 0.0;
                            gap::Tuple{Symbol,Symbol} = (:output, :between_steps))
    hl.meter === nothing && error("budget_begin_step!: the stage budget has no meter (budget_start!)")
    hl.bin = hl.nbins == 24 ? mod(floor(Int, hour), 24) + 1 : 1
    _measure!(hl.H_now, hl, θ)
    copyto!(hl.Hs_now, hl.sbuf)
    if hl.have_end
        Δ = hl.H_now .- hl.H_end
        any(!iszero, Δ) && _book!(hl, gap, Δ)
    else
        copyto!(hl.H_end, hl.H_now)       # storage counts from here
        hl.have_end = true
    end
    empty!(hl.stack)
    hl.stale = false
    hl.active = true
    return hl
end

"""
    budget_end_step!(b, x, dt) -> b

Finish the step of length `dt` (s): measure the quantity, book anything
that changed it outside every context as `(:core, :unattributed)`, and
add the step's change to `storage`.
"""
function budget_end_step!(hl::StageBudget, θ, dt::Real)
    hl.active || return hl
    isempty(hl.stack) || error("budget_end_step!: context ($(hl.stack[end].stage), $(hl.stack[end].name)) still open")
    _measure!(hl.buf, hl, θ)
    gap = hl.buf .- hl.H_now
    any(!iszero, gap) && _book!(hl, (:core, :unattributed), gap)
    b = hl.bin
    @inbounds for k in eachindex(hl.buf)
        hl.storage[k, b] += hl.buf[k] - hl.H_end[k]
    end
    copyto!(hl.H_end, hl.buf)
    copyto!(hl.H_now, hl.buf)
    copyto!(hl.Hs_now, hl.sbuf)
    hl.elapsed[b] += dt
    hl.stale = false
    hl.active = false
    return hl
end

"""
    budget_open!(b, stage, name, x; split = false, scale = 1, defer = false) -> b

Open the attribution context `(stage, name)` on the field `x`. At the top
level its starting content is the last measured value (nothing touches
the quantity between two top-level contexts); nested, it is measured. With
`split = true` (the dynamics) what the solid cells gained is booked as
`(stage, :solid_exchange)` (negated: what the air lost to them), and the
rest of the change not booked through [`book!`](@ref) is split by
[`split_transport`](@ref) into `(stage, :transport)` and
`(stage, :redistribution)`, applied to air and solid cells together (the
transport conserves their sum, not the air's alone).

`x` may be a tendency instead of the state (a component inside the core
adds to the tendency, which the predictor applies over the step): then
`scale` is the length (s) the tendency acts over, and the context books
`scale ×` the change of the measured tendency. `defer = true` keeps what
nested contexts book until this one closes (see
[`budget_discard!`](@ref)). A no-op unless a step is being attributed.
"""
function budget_open!(hl::StageBudget, stage::Symbol, name::Symbol, θ; split::Bool = false,
                      scale::Real = 1.0, defer::Bool = false)
    hl.active || return hl
    Nz = hl.meter.Nz
    ctx = isempty(hl.pool) ?
          BudgetContext(stage, name, zeros(Nz), zeros(Nz), zeros(Nz + 1), split, 1.0, false,
                        Dict{Tuple{Symbol,Symbol},Vector{Float64}}()) :
          pop!(hl.pool)
    ctx.stage, ctx.name, ctx.split = stage, name, split
    ctx.scale, ctx.defer = Float64(scale), defer
    empty!(ctx.pending)
    fill!(ctx.booked, 0.0)
    if isempty(hl.stack) && scale == 1
        hl.stale && _refresh!(hl, θ)
        copyto!(ctx.H0, hl.H_now); copyto!(ctx.S0, hl.Hs_now)
    else
        _measure!(ctx.H0, hl, θ); copyto!(ctx.S0, hl.sbuf)
    end
    push!(hl.stack, ctx)
    return hl
end

"""
    budget_close!(b, x; exchange = nothing) -> b

Close the innermost context: measure the quantity, and book the change
minus what was booked inside it under `(stage, name)` (or split, see
[`budget_open!`](@ref)). A nested context's change counts as booked for
its parent. A deferring context commits what its nested contexts booked.

`exchange = (stage, term)` (a split context only: the dynamics of a
domain with nest faces, NEST-p2 #482): the column total of what the
transport would conserve, `Σ_k conserved_k` of [`split_transport`](@ref)
— for a closed domain the numerics residual, for a domain with open faces
what crossed them — is booked under that key as a column amount (no
level), and taken out of `(stage, :transport)`, whose column total is
then the residual alone.
"""
function budget_close!(hl::StageBudget, θ; exchange::Union{Nothing,Tuple{Symbol,Symbol}} = nothing)
    hl.active || return hl
    ctx = pop!(hl.stack)
    Nz = hl.meter.Nz
    H1 = hl.buf
    _measure!(H1, hl, θ)
    rest = H1 .- ctx.H0                       # measured change per level
    hl.changed = any(!iszero, rest)
    if ctx.scale != 1
        rest .*= ctx.scale                    # a tendency: its effect over the step
    end
    if !isempty(hl.stack)
        parent = hl.stack[end]
        @inbounds for k in 1:Nz
            parent.booked[k] += rest[k]
        end
    elseif ctx.scale == 1
        copyto!(hl.H_now, H1); copyto!(hl.Hs_now, hl.sbuf)
        hl.stale = false
    end
    @inbounds for k in 1:Nz
        rest[k] -= ctx.booked[k]
    end
    unlocated = -ctx.booked[Nz+1]
    if ctx.defer
        for (key, p) in ctx.pending       # committed to the context's parent (or to acc)
            _book!(hl, key, view(p, 1:Nz), p[end])
        end
        empty!(ctx.pending)
    end
    if ctx.split
        # the solid cells' gain is what the air lost to them through the
        # dynamics; the transport conserves air + solid together
        Δs = hl.sbuf .- ctx.S0
        any(!iszero, Δs) && _book!(hl, (ctx.stage, :solid_exchange), -Δs)
        cons, red = split_transport(hl.meter, rest .+ Δs)
        if exchange === nothing
            _book!(hl, (ctx.stage, :transport), cons, unlocated)
        else
            X = sum(cons)
            _book!(hl, exchange, zeros(Nz), X)
            _book!(hl, (ctx.stage, :transport), cons, unlocated - X)
        end
        any(!iszero, red) && _book!(hl, (ctx.stage, :redistribution), red)
    elseif any(!iszero, rest) || unlocated != 0
        _book!(hl, (ctx.stage, ctx.name), rest, unlocated)
    end
    push!(hl.pool, ctx)
    return hl
end

# a top-level context after skipped ones: measure the state afresh; what
# changed since the last measurement was done outside every measured
# context (by a component whose declared writes left the quantity out)
function _refresh!(hl::StageBudget, θ)
    _measure!(hl.buf, hl, θ)
    gap = hl.buf .- hl.H_now
    any(!iszero, gap) && _book!(hl, (:core, :unattributed), gap)
    copyto!(hl.H_now, hl.buf); copyto!(hl.Hs_now, hl.sbuf)
    hl.stale = false
    return hl
end

"""
    budget_skip!(b) -> b

Skip a context: the component that would run in it cannot change the
quantity (its declared writes, `Components.budget_writes`, #539), so it is
not measured. At the top level the next context that measures (or the end
of the step) measures afresh, and books a change since the last
measurement as `(:core, :unattributed)`: a skipped component that did
write is not attributed to the next context. Nested (inside the core
step) it does nothing: an undeclared write stays in the enclosing
context's own share. A no-op unless a step is being attributed.
"""
function budget_skip!(hl::StageBudget)
    hl.active && isempty(hl.stack) && (hl.stale = true)
    return hl
end

"""
    budget_discard!(b) -> b

Drop what the contexts nested in the innermost deferring context booked
(the model calls it when the core restarts a step it rejected: the stages
inside the core run again with the shorter step). Contexts still open
above it (a stage entry the rejected attempt left by throwing) are closed
without booking.
"""
function budget_discard!(hl::StageBudget)
    hl.active || return hl
    for i in length(hl.stack):-1:1
        ctx = hl.stack[i]
        ctx.defer || continue
        while length(hl.stack) > i
            push!(hl.pool, pop!(hl.stack))
        end
        empty!(ctx.pending)
        fill!(ctx.booked, 0.0)
        return hl
    end
    return hl
end

"""
    book!(b, term, J; k = nothing) -> b
    book!(b, term, J::AbstractVector) -> b

Book `J` (per unit ground area of the domain: J m⁻² for the heat of the
air) added to the air by `term` (negative: removed) during the current
context — the stage entry or dynamics step that is running. It is stored
under `(stage, term)` and taken out of what the context books for itself,
so the context's own entry (for the dynamics: `:transport`) becomes the
residual of the terms named inside it. `k` names the level the amount went
into; without it the term is known only for the whole column, and layer
budgets report it as unlocated. The vector form gives one value per
level.

Must be called while a context is open (from a stage entry, or from code
that runs inside the core step); called after the step it is an error:
that amount would have no place in the step's balance.

Terms other issues book (#246): `:top_flux` (N-parent-b),
`:boundary_heat`, `:relaxation`, `:mass_correction` (N-open-d),
`:feedback_heat` (N-berlin-twoway, #267, in the parent's budgets).
"""
function book!(hl::StageBudget, term::Symbol, J::Real; k::Union{Nothing,Integer} = nothing)
    hl.active && !isempty(hl.stack) ||
        error("book!(:$term): no stage entry or dynamics step is running; book from inside the step")
    ctx = hl.stack[end]
    Nz = hl.meter.Nz
    idx = k === nothing ? Nz + 1 : Int(k)
    1 <= idx <= Nz + 1 || throw(BoundsError(ctx.booked, idx))
    ctx.booked[idx] += J
    a = get!(() -> zeros(Nz + 1, hl.nbins), hl.acc, (ctx.stage, term))
    a[idx, hl.bin] += J
    return hl
end
function book!(hl::StageBudget, term::Symbol, J::AbstractVector)
    hl.active && !isempty(hl.stack) ||
        error("book!(:$term): no stage entry or dynamics step is running; book from inside the step")
    Nz = hl.meter.Nz
    length(J) == Nz || throw(DimensionMismatch("book!: $(length(J)) levels, the budget has $Nz"))
    ctx = hl.stack[end]
    ctx.booked[1:Nz] .+= J
    _book!(hl, (ctx.stage, term), J)
    return hl
end

"Bins (1-based columns) of the local-hour `window = (h0, h1)` (`nothing`: all)."
function _hour_bins(hl::StageBudget, window)
    window === nothing && return collect(1:hl.nbins)
    hl.nbins == 24 || throw(ArgumentError("stage_budget: a window needs the 24 local-hour bins"))
    h0, h1 = Int(window[1]), Int(window[2])
    hs = h0 < h1 ? collect(h0:h1-1) : vcat(collect(h0:23), collect(0:h1-1))
    return [mod(h, 24) + 1 for h in hs]
end

"Level weights of a `layer` (`:column`, `:canopy` or `(z0, z1)` in m) and whether it is the whole column."
function _layer_weights(hl::StageBudget, layer)
    m = hl.meter
    layer === :column && return ones(m.Nz), true
    if layer === :canopy
        isfinite(hl.canopy_top) || error("stage_budget: canopy_top is not set")
        z = (0.0, hl.canopy_top)
    else
        z = (Float64(layer[1]), Float64(layer[2]))
    end
    f = layer_fraction(m, z[1], z[2])
    return f, all(==(1.0), f)
end

"""
    stage_budget(b; window = nothing, layer = :column) -> NamedTuple

The budget over the steps that started in the local-hour `window = (h0,
h1)` (integer hours, 24 bins only; `(20, 5)` is 20:00–05:00, crossing
midnight; `nothing` = every step; several days add up), for the `layer`:
`:column` (the whole domain depth), `:canopy` (0 to `canopy_top`, the
tallest roof in the urban case) or `(z0, z1)` in m above the ground (e.g.
`(0, 100)`). Returns

`(storage, terms, closure, elapsed, layer, unlocated)`

* `storage` — the measured change of the quantity in the layer (J m⁻² of
  domain for the heat of the air);
* `terms::Dict{Tuple{Symbol,Symbol},Float64}` — the attributed changes,
  keyed `(stage, component)` or `(stage, term)`;
* `closure = storage − Σ terms` — round-off if nothing escaped;
* `elapsed` — seconds in the window (`storage / elapsed` is in W m⁻²);
* `unlocated` — for a sub-layer, the column-only terms (`book!` without a
  level) that it cannot include.

For a sub-layer, `(:dynamics, :transport)` is what the resolved and SGS
transport carry across its top (for the whole column it is the numerics
residual of the transport).
"""
function stage_budget(hl::StageBudget; window = nothing, layer = :column)
    m = hl.meter
    m === nothing && return (storage = 0.0, terms = Dict{Tuple{Symbol,Symbol},Float64}(),
                             closure = 0.0, elapsed = 0.0, layer = layer, unlocated = 0.0)
    bins = _hour_bins(hl, window)
    f, whole = _layer_weights(hl, layer)
    Nz = m.Nz
    tot(a) = sum(f[k] * a[k, b] for k in 1:Nz, b in bins; init = 0.0) +
             (whole ? sum(a[Nz+1, b] for b in bins) : 0.0)
    terms = Dict{Tuple{Symbol,Symbol},Float64}()
    unloc = 0.0
    for (key, a) in hl.acc
        terms[key] = tot(a)
        whole || (unloc += sum(a[Nz+1, b] for b in bins))
    end
    st = tot(hl.storage)
    return (storage = st, terms = terms, closure = st - sum(values(terms); init = 0.0),
            elapsed = sum(hl.elapsed[b] for b in bins), layer = layer, unlocated = unloc)
end

"""
    report_order(b, key) -> Tuple

The sort key of the term `key = (stage, name)` in the reports of `b`: its
stage in `b.order`, then the dynamics and the core, the other stages after
them; then the name.
"""
function report_order(hl::StageBudget, key)
    order = (hl.order..., :dynamics, :core)
    i = findfirst(==(key[1]), order)
    return (i === nothing ? length(order) + 1 : i, string(key[2]))
end

"""
    stage_report(b; window = nothing, layers = (:column, :canopy, (0, 100)), title = "Air-heat budget",
                 units = ("J/m²", "W/m²")) -> String

The budget as a table: one row per attributed term, one column per layer,
in `units[1]` and as the window mean in `units[2]` (per second).
"""
function stage_report(hl::StageBudget; window = nothing,
                      layers = (:column, :canopy, (0.0, 100.0)), title = "Air-heat budget",
                      units = ("J/m²", "W/m²"))
    hl.meter === nothing && return ""
    io = IOBuffer()
    ls = [l for l in layers if !(l === :canopy && !isfinite(hl.canopy_top))]
    bs = [stage_budget(hl; window = window, layer = l) for l in ls]
    el = bs[1].elapsed
    el > 0 || return ""
    lname(l) = l === :column ? @sprintf("column 0-%.0f m", hl.meter.zf[end]) :
               l === :canopy ? @sprintf("canopy 0-%.0f m", hl.canopy_top) :
               @sprintf("%.0f-%.0f m", l[1], l[2])
    wname = window === nothing ? "the whole run" : @sprintf("local %02d-%02d h", window[1], window[2])
    @printf(io, "%s over %s (%.0f s): %s [%s]\n", title, wname, el, units[1], units[2])
    @printf(io, "  %-34s", "term")
    foreach(l -> @printf(io, " %26s", lname(l)), ls)
    println(io)
    keys_ = sort!(unique(vcat([collect(keys(b.terms)) for b in bs]...)); by = k -> report_order(hl, k))
    row(label, vals) = (@printf(io, "  %-34s", label);
                        foreach(v -> @printf(io, " %15.4e [%8.3f]", v, v / el), vals); println(io))
    row("storage", [b.storage for b in bs])
    for key in keys_
        row("$(key[1]) / $(key[2])", [get(b.terms, key, 0.0) for b in bs])
    end
    @printf(io, "  %-34s", "closure ($(units[2]))")
    foreach(b -> @printf(io, " %26.2e", b.closure / el), bs)
    println(io)
    return String(take!(io))
end
