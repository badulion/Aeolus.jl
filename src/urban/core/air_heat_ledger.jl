# The air-heat ledger: where the heat of the air came from, step by step.
#
# The surface ledger closes the energy balance of every surface class and
# the water ledger the water of the air. This one closes the heat of the
# air, H_air = Σ ρ̄ c_p Π̄ θ ΔV / A (core/air_heat.jl), and attributes
# every change of it automatically, without the couplings reporting
# anything:
#
#   * every stage entry the step runs (`timed_run!`) and the dynamics
#     (`step!`, through the `:timestepper` provider) is an attribution
#     *context*. The ledger measures H_air (per level) when the context
#     opens and when it closes, and books the difference under
#     (stage, coupling) — e.g. (:surface_flux, :surface_exchange),
#     (:fog_dew, :fog_dew), (:column_tendency, :column_radiation);
#   * contexts nest: the `:body_force` / `:post_predictor` entries run
#     inside `step!`, and what they do is taken out of the dynamics'
#     share;
#   * the change a context makes can be split into named terms with
#     `add_heat!(ledger, term, J)` from inside it (the top heat flux, the
#     open-boundary heat of the nesting epic, #246). The rest is booked
#     under the context itself. For the dynamics the rest is split by
#     `split_transport` into (:dynamics, :transport) — what the transport
#     conserves; over the whole column it is the numerics residual — and
#     (:dynamics, :redistribution). The heat the dynamics moves into or
#     out of solid (building) cells, which are not air, is booked apart as
#     (:dynamics, :solid_exchange): the core's θ transport and diffusion
#     do not stop at the immersed boundary, so the solid cells' θ evolves
#     and exchanges heat with the air through the numerics alone;
#   * between two contexts at the top level nothing else touches θ, so
#     the ledger measures only when a context closes (the value at the
#     open is the previous close): one pass over the interior per entry.
#     A change outside any context still cannot hide: the step's end
#     value is compared with the last close (booked as
#     (:core, :unattributed)), and the start of the next step with the
#     end of this one (booked as (:output, :between_steps), the stage
#     that runs after the ledger closes a step).
#
# Every term is kept per level (so any layer — the canopy, 0–100 m — is
# a weighted sum of the same numbers) and per local hour of the step's
# start (so `heat_budget(ledger; window = (20, 5))` returns the night
# integrals of #238).
#
# Nothing here refers to the Berlin `Case`: the functions take the
# ledger, the meter and the θ field, so the parent run (#259) keeps its
# own `AirHeatLedger` with the same code.

"One open attribution context (a stage entry or the dynamics)."
mutable struct HeatContext
    stage::Symbol
    name::Symbol
    H0::Vector{Float64}       # per-level heat when the context opened
    S0::Vector{Float64}       # per-level heat of the solid cells when it opened
    booked::Vector{Float64}   # per level + 1 (unlocated): add_heat! inside it, and nested contexts
    split::Bool               # split the rest into :transport / :redistribution
end

"""
    AirHeatLedger()

The air-heat ledger of a run (see `core/air_heat_ledger.jl`). Fields:

* `meter` — the [`AirHeatMeter`](@ref) (set when the run starts);
* `acc[(stage, term)]` — accumulated change of H_air, J m⁻² of domain,
  as an `(Nz + 1) × 24` matrix: row `k ≤ Nz` is level `k`, row `Nz + 1`
  the part booked without a level (`add_heat!` without `k`); column `h`
  the steps that started in local hour `h − 1`;
* `storage` — the measured change of H_air, same layout;
* `elapsed` — seconds per local-hour bin;
* `canopy_top` — top (m) of the `:canopy` layer of [`heat_budget`](@ref)
  (the Berlin case sets it to the tallest roof);
* `active` — true while a step is being attributed.
"""
mutable struct AirHeatLedger
    meter::Union{Nothing,AirHeatMeter}
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
    stack::Vector{HeatContext}
    pool::Vector{HeatContext}
    canopy_top::Float64
    nmeasure::Int
end
AirHeatLedger() = AirHeatLedger(nothing, Dict{Tuple{Symbol,Symbol},Matrix{Float64}}(), zeros(0, 24),
                                zeros(24), 1, Float64[], Float64[], Float64[], Float64[], false, Float64[], false,
                                HeatContext[], HeatContext[], NaN, 0)

"""
    heat_start!(hl, meter; canopy_top = NaN) -> hl

(Re)start the air-heat ledger `hl` with `meter`: every accumulated term
is zeroed.
"""
function heat_start!(hl::AirHeatLedger, m::AirHeatMeter; canopy_top::Real = hl.canopy_top)
    Nz = m.Nz
    hl.meter = m
    empty!(hl.acc)
    hl.storage = zeros(Nz + 1, 24)
    hl.elapsed = zeros(24)
    hl.H_now = zeros(Nz); hl.H_end = zeros(Nz); hl.buf = zeros(Nz)
    hl.Hs_now = zeros(Nz); hl.sbuf = zeros(Nz)
    hl.have_end = false
    hl.active = false
    empty!(hl.stack)
    hl.canopy_top = canopy_top
    hl.nmeasure = 0
    return hl
end

# measure the air's heat per level into `out`, and the solid cells' into hl.sbuf
function _measure!(out, hl::AirHeatLedger, θ)
    hl.nmeasure += 1
    air_heat_levels!(out, hl.meter, θ)
    solid_heat_levels!(hl.sbuf, hl.meter)
    return out
end

function _book!(hl::AirHeatLedger, key::Tuple{Symbol,Symbol}, v::AbstractVector, extra::Real = 0.0)
    a = get!(() -> zeros(hl.meter.Nz + 1, 24), hl.acc, key)
    b = hl.bin
    @inbounds for k in eachindex(v)
        a[k, b] += v[k]
    end
    a[end, b] += extra
    return hl
end

"""
    heat_begin_step!(hl, θ, hour) -> hl

Start attributing one step that begins at local `hour`: measure H_air
(any change since the end of the previous step is booked as
`(:output, :between_steps)`).
"""
function heat_begin_step!(hl::AirHeatLedger, θ, hour::Real)
    hl.meter === nothing && error("heat_begin_step!: the air-heat ledger has no meter (heat_start!)")
    hl.bin = mod(floor(Int, hour), 24) + 1
    _measure!(hl.H_now, hl, θ)
    copyto!(hl.Hs_now, hl.sbuf)
    if hl.have_end
        gap = hl.H_now .- hl.H_end
        any(!iszero, gap) && _book!(hl, (:output, :between_steps), gap)
    else
        copyto!(hl.H_end, hl.H_now)       # storage counts from here
        hl.have_end = true
    end
    empty!(hl.stack)
    hl.active = true
    return hl
end

"""
    heat_end_step!(hl, θ, dt) -> hl

Finish the step of length `dt` (s): measure H_air, book anything that
changed θ outside every context as `(:core, :unattributed)`, and add the
step's change to `storage`.
"""
function heat_end_step!(hl::AirHeatLedger, θ, dt::Real)
    hl.active || return hl
    isempty(hl.stack) || error("heat_end_step!: context ($(hl.stack[end].stage), $(hl.stack[end].name)) still open")
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
    hl.active = false
    return hl
end

"""
    heat_open!(hl, stage, name, θ; split = false) -> hl

Open the attribution context `(stage, name)`. At the top level its
starting heat is the last measured value (nothing touches θ between
two top-level contexts); nested, it is measured. With `split = true`
(the dynamics) the heat the solid cells gained is booked as
`(stage, :solid_exchange)` (negated: heat the air lost to them), and the
rest of the change not booked through `add_heat!` is split by
[`split_transport`](@ref) into `(stage, :transport)` and
`(stage, :redistribution)`, applied to air and solid cells together (the
transport conserves their sum, not the air's alone). A no-op unless a step is being attributed.
"""
function heat_open!(hl::AirHeatLedger, stage::Symbol, name::Symbol, θ; split::Bool = false)
    hl.active || return hl
    Nz = hl.meter.Nz
    ctx = isempty(hl.pool) ? HeatContext(stage, name, zeros(Nz), zeros(Nz), zeros(Nz + 1), split) :
                             pop!(hl.pool)
    ctx.stage, ctx.name, ctx.split = stage, name, split
    fill!(ctx.booked, 0.0)
    if isempty(hl.stack)
        copyto!(ctx.H0, hl.H_now); copyto!(ctx.S0, hl.Hs_now)
    else
        _measure!(ctx.H0, hl, θ); copyto!(ctx.S0, hl.sbuf)
    end
    push!(hl.stack, ctx)
    return hl
end

"""
    heat_close!(hl, θ) -> hl

Close the innermost context: measure H_air, and book the change minus
what was booked inside it under `(stage, name)` (or split, see
[`heat_open!`](@ref)). A nested context's change counts as booked for
its parent.
"""
function heat_close!(hl::AirHeatLedger, θ)
    hl.active || return hl
    ctx = pop!(hl.stack)
    Nz = hl.meter.Nz
    H1 = hl.buf
    _measure!(H1, hl, θ)
    rest = H1 .- ctx.H0                       # measured change per level
    if !isempty(hl.stack)
        parent = hl.stack[end]
        @inbounds for k in 1:Nz
            parent.booked[k] += rest[k]
        end
    else
        copyto!(hl.H_now, H1); copyto!(hl.Hs_now, hl.sbuf)
    end
    @inbounds for k in 1:Nz
        rest[k] -= ctx.booked[k]
    end
    unlocated = -ctx.booked[Nz+1]
    if ctx.split
        # the solid cells' gain is heat the air lost to them through the
        # dynamics; the transport conserves air + solid together
        Δs = hl.sbuf .- ctx.S0
        any(!iszero, Δs) && _book!(hl, (ctx.stage, :solid_exchange), -Δs)
        cons, red = split_transport(hl.meter, rest .+ Δs)
        _book!(hl, (ctx.stage, :transport), cons, unlocated)
        any(!iszero, red) && _book!(hl, (ctx.stage, :redistribution), red)
    elseif any(!iszero, rest) || unlocated != 0
        _book!(hl, (ctx.stage, ctx.name), rest, unlocated)
    end
    push!(hl.pool, ctx)
    return hl
end

"""
    add_heat!(hl_or_ledger, term, J; k = nothing) -> ledger
    add_heat!(hl_or_ledger, term, J::AbstractVector) -> ledger

Book `J` J m⁻² (per unit ground area of the domain) of heat added to the
air by `term` (negative: removed) during the current context — the
stage entry or dynamics step that is running. It is stored under
`(stage, term)` and taken out of what the context books for itself, so
the context's own entry (for the dynamics: `:transport`) becomes the
residual of the terms named inside it. `k` names the level the heat went
into; without it the term is known only for the whole column, and layer
budgets report it as unlocated. The vector form gives one value per
level.

Must be called while a context is open (from a `run_stage!`, or from
code that runs inside `step!`); called from `budget!`, after the step,
it is an error: that heat would have no place in the step's balance.

Terms other issues book (#246): `:top_flux` (N-parent-b),
`:boundary_heat`, `:relaxation`, `:mass_correction` (N-open-d),
`:feedback_heat` (N-berlin-twoway, #267, in the parent's ledgers).
"""
function add_heat!(hl::AirHeatLedger, term::Symbol, J::Real; k::Union{Nothing,Integer} = nothing)
    hl.active && !isempty(hl.stack) ||
        error("add_heat!(:$term): no stage entry or dynamics step is running; book heat from inside the step")
    ctx = hl.stack[end]
    Nz = hl.meter.Nz
    idx = k === nothing ? Nz + 1 : Int(k)
    1 <= idx <= Nz + 1 || throw(BoundsError(ctx.booked, idx))
    ctx.booked[idx] += J
    a = get!(() -> zeros(Nz + 1, 24), hl.acc, (ctx.stage, term))
    a[idx, hl.bin] += J
    return hl
end
function add_heat!(hl::AirHeatLedger, term::Symbol, J::AbstractVector)
    hl.active && !isempty(hl.stack) ||
        error("add_heat!(:$term): no stage entry or dynamics step is running; book heat from inside the step")
    Nz = hl.meter.Nz
    length(J) == Nz || throw(DimensionMismatch("add_heat!: $(length(J)) levels, the ledger has $Nz"))
    ctx = hl.stack[end]
    ctx.booked[1:Nz] .+= J
    _book!(hl, (ctx.stage, term), J)
    return hl
end

"Hour bins (1-based columns) of the local-hour `window = (h0, h1)` (`nothing`: all)."
function _hour_bins(window)
    window === nothing && return collect(1:24)
    h0, h1 = Int(window[1]), Int(window[2])
    hs = h0 < h1 ? collect(h0:h1-1) : vcat(collect(h0:23), collect(0:h1-1))
    return [mod(h, 24) + 1 for h in hs]
end

"Level weights of a `layer` (`:column`, `:canopy` or `(z0, z1)` in m) and whether it is the whole column."
function _layer_weights(hl::AirHeatLedger, layer)
    m = hl.meter
    layer === :column && return ones(m.Nz), true
    if layer === :canopy
        isfinite(hl.canopy_top) || error("heat_budget: canopy_top is not set")
        z = (0.0, hl.canopy_top)
    else
        z = (Float64(layer[1]), Float64(layer[2]))
    end
    f = layer_fraction(m, z[1], z[2])
    return f, all(==(1.0), f)
end

"""
    heat_budget(ledger_or_hl; window = nothing, layer = :column) -> NamedTuple

The air-heat budget over the steps that started in the local-hour
`window = (h0, h1)` (integer hours; `(20, 5)` is 20:00–05:00, crossing
midnight; `nothing` = every step; several days add up), for the
`layer`: `:column` (the whole domain depth), `:canopy` (0 to
`canopy_top`, the tallest roof in the Berlin case) or `(z0, z1)` in m
above the ground (e.g. `(0, 100)`). Returns

`(storage, terms, closure, elapsed, layer, unlocated)`

* `storage` — the measured change of the heat of the air in the layer,
  J m⁻² of domain;
* `terms::Dict{Tuple{Symbol,Symbol},Float64}` — the attributed changes,
  J m⁻², keyed `(stage, coupling)` or `(stage, term)`;
* `closure = storage − Σ terms` — round-off if nothing escaped;
* `elapsed` — seconds in the window (`storage / elapsed` is in W m⁻²);
* `unlocated` — for a sub-layer, the column-only terms (`add_heat!`
  without a level) that it cannot include.

For a sub-layer, `(:dynamics, :transport)` is the heat the resolved and
SGS transport carry across its top (for the whole column it is the
numerics residual of the transport).
"""
function heat_budget(hl::AirHeatLedger; window = nothing, layer = :column)
    m = hl.meter
    m === nothing && return (storage = 0.0, terms = Dict{Tuple{Symbol,Symbol},Float64}(),
                             closure = 0.0, elapsed = 0.0, layer = layer, unlocated = 0.0)
    bins = _hour_bins(window)
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

"The stage order of a key in the reports (core stages, then the rest alphabetically)."
function _heat_key_order(key)
    order = (ALL_STAGES..., :dynamics, :core)
    i = findfirst(==(key[1]), order)
    return (i === nothing ? length(order) + 1 : i, string(key[2]))
end

"""
    heat_report(hl; window = nothing, layers = (:column, :canopy, (0, 100))) -> String

The air-heat budget as a table: one row per attributed term, one column
per layer, in J m⁻² and as the window mean in W m⁻².
"""
function heat_report(hl::AirHeatLedger; window = nothing,
                     layers = (:column, :canopy, (0.0, 100.0)))
    hl.meter === nothing && return ""
    io = IOBuffer()
    ls = [l for l in layers if !(l === :canopy && !isfinite(hl.canopy_top))]
    bs = [heat_budget(hl; window = window, layer = l) for l in ls]
    el = bs[1].elapsed
    el > 0 || return ""
    lname(l) = l === :column ? @sprintf("column 0-%.0f m", hl.meter.zf[end]) :
               l === :canopy ? @sprintf("canopy 0-%.0f m", hl.canopy_top) :
               @sprintf("%.0f-%.0f m", l[1], l[2])
    wname = window === nothing ? "the whole run" : @sprintf("local %02d-%02d h", window[1], window[2])
    @printf(io, "Air-heat ledger over %s (%.0f s): J/m² [W/m²]\n", wname, el)
    @printf(io, "  %-34s", "term")
    foreach(l -> @printf(io, " %26s", lname(l)), ls)
    println(io)
    keys_ = sort!(unique(vcat([collect(keys(b.terms)) for b in bs]...)); by = _heat_key_order)
    row(label, vals) = (@printf(io, "  %-34s", label);
                        foreach(v -> @printf(io, " %15.4e [%8.3f]", v, v / el), vals); println(io))
    row("storage ΔH_air", [b.storage for b in bs])
    for key in keys_
        row("$(key[1]) / $(key[2])", [get(b.terms, key, 0.0) for b in bs])
    end
    @printf(io, "  %-34s", "closure (W/m²)")
    foreach(b -> @printf(io, " %26.2e", b.closure / el), bs)
    println(io)
    return String(take!(io))
end
