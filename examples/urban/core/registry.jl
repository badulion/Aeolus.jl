# The coupling registry.
#
# Every physics coupling of the Berlin case is one file under
# `couplings/`. The file defines a struct `<: AbstractCoupling`, extends
# the contract below for its type, and ends with `register!(ItsType)`.
# Adding a coupling therefore means adding one file: no shared file is
# edited. `build_case` asks the registry which couplings the config
# enables, builds them, and the stage loop (`stages.jl`) calls them.
#
# Contract (every method has a default except `coupling_name` and the
# `run_stage!` methods for the stages a coupling declares):
#
#   coupling_name(::Type{C})    -> Symbol, unique; key in cfg.physics / cfg.options
#   slot(::Type{C})             -> :addon (additive, on/off) or an exclusive slot
#   provider(::Type{C})         -> Symbol, the value of cfg.physics[slot] that selects C
#   is_default(::Type{C})       -> Bool, default provider of its slot / default-on addon
#   modules_used(::Type{C})     -> tuple of Aeolus submodule names (coverage test)
#   default_options(::Type{C})  -> NamedTuple
#   incompatible(::Type{C})     -> tuple of `slot => provider` or coupling names
#   incompatible_reason(::Type{C}) -> String, why (appended to the selection error)
#   stages(::Type{C})           -> tuple of (stage::Symbol, order::Int, Cadence), one per stage
#   build(::Type{C}, case, opts)-> instance (default `C()`)
#   init!(c, case)              -> called once, after every coupling is built
#   run_stage!(c, case, ::Val{S}, dt)
#   output_fields(c, case)      -> Vector of OutputField, registered before nc_endef!
#   budget!(c, case, ledger)    -> energy / water terms for the ledger
#   report(c, case)             -> Union{Nothing,String}, printed with each output record
#   smoke_variants(::Type{C})   -> tuple of configs worth a smoke run
#   smoke_check(c, case)        -> Vector{Pair{String,Bool}}
#   sim_options(::Type{C}, cfg, mesh, opts) -> NamedTuple of BoussinesqSimulation keywords
#   step_options(c, case)       -> NamedTuple of step! keywords
#
# Dynamics contract (#133), used only by the providers of the two
# dynamics slots:
#
#   slot :core         build_core(::Type{C}, cfg, mesh, simkw) -> the dynamical-core simulation
#   slot :timestepper  next_dt(c, case)                 -> dt (s) of the next step
#                      advance_dynamics!(c, case, dt)   -> advance case.sim by dt
#                      variable_dt(::Type{C})           -> Bool: run_case! runs to a time

"""
    AbstractCoupling

Supertype of every Berlin-case coupling. See `core/registry.jl` for the
contract and `docs/src/design/berlin_case.md` for how to add one.
"""
abstract type AbstractCoupling end

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

function coupling_name end
slot(::Type{<:AbstractCoupling}) = :addon
provider(C::Type{<:AbstractCoupling}) = coupling_name(C)
is_default(::Type{<:AbstractCoupling}) = true
modules_used(::Type{<:AbstractCoupling}) = ()
default_options(::Type{<:AbstractCoupling}) = (;)
incompatible(::Type{<:AbstractCoupling}) = ()
incompatible_reason(::Type{<:AbstractCoupling}) = ""
stages(::Type{<:AbstractCoupling}) = ()
build(C::Type{<:AbstractCoupling}, case, opts) = C()
init!(c::AbstractCoupling, case) = nothing
output_fields(c::AbstractCoupling, case) = OutputField[]
budget!(c::AbstractCoupling, case, ledger) = nothing
report(c::AbstractCoupling, case) = nothing
smoke_variants(::Type{<:AbstractCoupling}) = ()
smoke_check(c::AbstractCoupling, case) = Pair{String,Bool}[]
"""
    sim_options(::Type{C}, cfg, mesh, opts) -> NamedTuple

Keywords `C` adds to the `BoussinesqSimulation` constructor (e.g. a
`solver`), called before the dynamical core is built. Two couplings may
not set the same keyword.
"""
sim_options(::Type{<:AbstractCoupling}, cfg, mesh, opts) = (;)
"""
    step_options(c, case) -> NamedTuple

Keywords `c` adds to every `step!` call (e.g. `momentum_advection`,
`eddy_viscosity!`), collected when the schedule is built. Two couplings
may not set the same keyword.
"""
step_options(c::AbstractCoupling, case) = (;)
function build_core end
variable_dt(::Type{<:AbstractCoupling}) = false
next_dt(c::AbstractCoupling, case) = case.cfg.dt
advance_dynamics!(c::AbstractCoupling, case, dt) =
    (step!(case.sim, dt; body_force! = case.body_force, post_predictor! = case.post_predictor); nothing)
run_stage!(c::AbstractCoupling, case, ::Val{S}, dt) where {S} =
    error("coupling :$(coupling_name(typeof(c))) declares stage :$S but has no run_stage! method for it")

coupling_name(c::AbstractCoupling) = coupling_name(typeof(c))

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

# --- the registry ------------------------------------------------------------

"All registered coupling types, in registration (= file-name) order."
const REGISTRY = Type[]

"""
    register!(C) -> C

Add coupling type `C` to the registry. The last line of every coupling
file. Checks that the declared stages exist and that the name is unique.
"""
function register!(C::Type{<:AbstractCoupling})
    C in REGISTRY && return C
    name = coupling_name(C)
    for D in REGISTRY
        coupling_name(D) === name && error("coupling name :$name registered twice ($D and $C)")
    end
    for st in stages(C)
        length(st) == 3 || error("$C: stage entries are (stage, order, Cadence), got $st")
        st[1] in ALL_STAGES || error("$C: unknown stage :$(st[1]); stages are $(ALL_STAGES)")
    end
    ss = [st[1] for st in stages(C)]
    allunique(ss) || error("$C: at most one entry per stage (run_stage! dispatches on the stage), got $ss")
    push!(REGISTRY, C)
    return C
end

"The registered coupling type called `name`."
function coupling_type(name::Symbol)
    for C in REGISTRY
        coupling_name(C) === name && return C
    end
    error("no coupling named :$name")
end

"All exclusive slots and their registered providers, `slot => [providers]`."
function slot_providers()
    d = Dict{Symbol,Vector{Symbol}}()
    for C in REGISTRY
        s = slot(C)
        s === :addon && continue
        push!(get!(d, s, Symbol[]), provider(C))
    end
    return d
end

"The default provider of exclusive slot `s` (the one with `is_default`)."
function default_provider(s::Symbol)
    ds = [provider(C) for C in REGISTRY if slot(C) === s && is_default(C)]
    length(ds) == 1 || error("slot :$s must have exactly one default provider, has $(ds)")
    return ds[1]
end

"""
    selected_couplings(cfg) -> Vector{Type}

The coupling types `cfg` enables, in registry order. An additive coupling
runs when `cfg.physics[name]` (default `is_default(C)`) is true; an
exclusive slot runs the provider `cfg.physics[slot]` (default: the slot's
default provider). Unknown keys, unknown providers and declared
incompatibilities are errors.
"""
function selected_couplings(cfg)
    slots = slot_providers()
    names = Set(coupling_name(C) for C in REGISTRY)
    for (k, v) in cfg.physics
        if haskey(slots, k)
            v in slots[k] || error("slot :$k has no provider :$v (providers: $(join(slots[k], ", ")))")
        elseif k in names
            v isa Bool || error("cfg.physics[:$k] must be true/false for an additive coupling")
            slot(coupling_type(k)) === :addon ||
                error("coupling :$k fills slot :$(slot(coupling_type(k))); select it with physics[:$(slot(coupling_type(k)))] = :$(provider(coupling_type(k)))")
        else
            error("cfg.physics has unknown key :$k (slots: $(join(sort!(collect(keys(slots))), ", ")); couplings: $(join(sort!(collect(names)), ", ")))")
        end
    end
    for k in keys(cfg.options)
        k in names || error("cfg.options has unknown coupling :$k")
    end
    sel = Type[]
    for C in REGISTRY
        s = slot(C)
        on = s === :addon ? get(cfg.physics, coupling_name(C), is_default(C)) :
                            provider(C) === get(cfg.physics, s, default_provider(s))
        on && push!(sel, C)
    end
    for C in sel, x in incompatible(C)
        for D in sel
            hit = x isa Pair ? (slot(D) === x.first && provider(D) === x.second) :
                               coupling_name(D) === x
            why = incompatible_reason(C)
            hit && error("coupling :$(coupling_name(C)) is incompatible with :$(coupling_name(D))" *
                         (isempty(why) ? "" : ": " * why))
        end
    end
    return sel
end

"Options of coupling type `C` under `cfg`: its defaults merged with `cfg.options[name]`."
function coupling_options(C::Type{<:AbstractCoupling}, cfg)
    user = get(cfg.options, coupling_name(C), (;))
    return merge(default_options(C), user)
end
