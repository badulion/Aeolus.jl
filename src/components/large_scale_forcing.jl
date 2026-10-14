# The large-scale forcing (ONE-g, #456): the geostrophic pressure gradient,
# the advective tendencies, subsidence and nudging of `LargeScaleForcing`,
# from numbers or from the dynamic driver's `ls_*` tables, read generically.
# (The `force!` closure of the former `sbl` run type, P55-f #191 / P58-d #194.)

"""
    LargeScale(geostrophic, extra, U_g, V_g, t0)

The large-scale forcing of a domain, `[large_scale_forcing]`: the
`LargeScaleForcing` objects it applies in stage 7 (`:body_force`) of every
(sub)step, each with its exact mean over the step `[t₀, t₀ + Δt]`
(`apply_forcing!`; `t₀` is the step start, recorded in stage 1, `:clock`):

  * `geostrophic` — the `GeostrophicPressureGradient` of the geostrophic
    wind `(U_g, V_g)`: `u_g`, `v_g` are numbers (m s⁻¹), or `"driver"`,
    the dynamic driver's `ls_u_g`, `ls_v_g` height-time tables. It takes
    the Coriolis parameter of the domain's [`Coriolis`](@ref) component;
  * `extra` — from the dynamic driver, each when its switch is on:
    `advective_tendencies` (`AdvectiveTendencies` of `ls_u_adv`,
    `ls_v_adv`, `ls_theta_adv`, `ls_qv_adv`), `subsidence` (`Subsidence`
    of `ls_w_subs`, its fields and form from the driver's attributes) and
    `nudging` (`ProfileNudging` toward `ls_nudge_*`, the time scale
    `nudging_tau`), in that order (`InputDrivers.dynamic_forcing`).

The driver's times (s since its `init_time`) are read at the model time
minus the run's start (`[time] start`). The equations are those of
`Aeolus.LargeScaleForcing`; this component only selects and applies them.
The geostrophic wind is also the target of a sponge retargeted at it
([`Sponge`](@ref), `target = "geostrophic"`).

Built in code (`LargeScale(geostrophic, extra; U_g, V_g, clock, zi)`, the
urban nest parent's domain, NEST-p1 #481), the component also takes:

  * `geostrophic = nothing` — no pressure gradient, only the `extra`
    forcings;
  * `clock` — the time window of each (sub)step: `:step` (the default and
    the configuration's) is the step `[t₀, t₀ + Δt]` in every substage, so
    the SSP-RK3 combination applies exactly the step mean; `:core` is the
    core's own clock `sim.t` at each substage, which SSP-RK3 advances by
    Δt per substage (windows ``[t₀ + (s-1)Δt, t₀ + sΔt]``, s = 1, 2, 3):
    the urban nest parent's forcing since #259, kept bit for bit;
  * `zi` — a [`BoundaryLayerTracker`](@ref): the running maximum of the
    boundary-layer height that a `ProfileNudging`'s `NudgingWeight` reads
    (the nudging acts above a multiple of it), updated in stage 11
    (`:column_tendency`).
"""
struct LargeScale{G, E<:Tuple, U, V, Z} <: AbstractComponent
    geostrophic::G
    extra::E
    U_g::U
    V_g::V
    t0::Base.RefValue{Float64}
    clock::Symbol
    zi::Z
end
LargeScale(geostrophic, extra::Tuple, U_g, V_g, t0::Base.RefValue{Float64}) =
    LargeScale(geostrophic, extra, U_g, V_g, t0, :step, nothing)
function LargeScale(geostrophic, extra::Tuple; U_g = nothing, V_g = nothing, clock::Symbol = :step,
                    zi = nothing, t0::Real = 0.0)
    clock in (:step, :core) || throw(ArgumentError("LargeScale: clock is :step or :core, got :$clock"))
    zi === nothing || zi isa BoundaryLayerTracker ||
        throw(ArgumentError("LargeScale: zi is nothing or a BoundaryLayerTracker"))
    return LargeScale(geostrophic, extra, U_g, V_g, Ref(Float64(t0)), clock, zi)
end

"""
    BoundaryLayerTracker(; floor = 0.0, every = 10)

The running maximum of the boundary-layer height of a domain, the height
a boundary-layer-dependent `NudgingWeight` reads ([`nudging_height`](@ref)):
every `every` steps the parcel height of the plane-mean virtual potential
temperature ([`parcel_zi`](@ref)) raises it, and it never falls below
`floor` (m; the urban nest parent: the largest parcel height of its
sounding's launches). Part of the [`LargeScale`](@ref) component.
"""
struct BoundaryLayerTracker
    zi::Base.RefValue{Float64}
    floor::Float64
    every::Int
    n::Base.RefValue{Int}
end
function BoundaryLayerTracker(; floor::Real = 0.0, every::Integer = 10)
    every >= 1 || throw(ArgumentError("BoundaryLayerTracker: every must be ≥ 1, got $every"))
    return BoundaryLayerTracker(Ref(0.0), Float64(floor), Int(every), Ref(0))
end

"The height (m) a boundary-layer-dependent nudging weight reads: the tracked maximum, at least the floor."
nudging_height(z::BoundaryLayerTracker) = max(z.floor, z.zi[])

"""
    plane_mean(f) -> Vector{Float64}

The plane means of the interior of a cell-centred field, per level (the
sum of each level in column-major order over the number of columns).
"""
function plane_mean(f)
    a = interior_view(f)
    return [sum(view(a, :, :, k)) / (size(a, 1) * size(a, 2)) for k in axes(a, 3)]
end

"The heights (m) of the level centres of a grid (`LargeScaleForcing`'s level coordinate)."
level_heights(m) = Float64[_zc(m, k) for k in 1:m.Nz]

"""
    parcel_zi(sim) -> m

The boundary-layer height of the state of `sim`: the parcel height
(`TurbulenceStability.boundary_layer_height`, `:parcel`) of the plane-mean
virtual potential temperature at the level centres (``θ_v`` of the
plane-mean θ and, on a moist core, ``q_v``).
"""
function parcel_zi(sim)
    θ = plane_mean(sim.θ)
    moist = hasproperty(sim, :has_moisture) && sim.has_moisture
    q = moist ? plane_mean(sim.qv) : zeros(length(θ))
    θv = [virtual_temperature(θ[k], PhasePartition(max(q[k], 0.0))) for k in eachindex(θ)]
    return boundary_layer_height(level_heights(sim.mesh), θv; method = :parcel)
end

# one step of the tracker: the parcel height every `every` steps
function track!(z::BoundaryLayerTracker, sim)
    z.n[] += 1
    z.n[] % z.every == 0 && (z.zi[] = max(z.zi[], parcel_zi(sim)))
    return z
end
checkpoint_state(z::BoundaryLayerTracker) = (zi = z.zi[], n = z.n[])
restore_checkpoint_state!(z::BoundaryLayerTracker, s) = (z.zi[] = s.zi; z.n[] = s.n; z)

component_name(::Type{<:LargeScale}) = :large_scale_forcing
section(::Type{<:LargeScale}) = "large_scale_forcing"

schema(::Type{<:LargeScale}) =
    Config.section("large_scale_forcing";
                   doc = "Large-scale forcing (LargeScaleForcing): the geostrophic pressure gradient, and the domain " *
                         "dynamic driver's advective tendencies, subsidence and nudging (its ls_* tables).") do s
        for (k, c) in (("u_g", "u"), ("v_g", "v"))
            Config.option!(s, k, Config.OneOf(Float64, Symbol); default = 0.0, allowed = [:driver], units = "m s^-1",
                           doc = "the geostrophic wind $c: a number, or \"driver\": the dynamic driver's ls_$k table (height, time)")
        end
        Config.option!(s, "advective_tendencies", Bool; default = false,
                       doc = "the dynamic driver's advective tendencies ls_u_adv, ls_v_adv, ls_theta_adv, ls_qv_adv (AdvectiveTendencies)")
        Config.option!(s, "subsidence", Bool; default = false,
                       doc = "the dynamic driver's subsidence velocity ls_w_subs (Subsidence; its fields and form from the " *
                             "driver's attributes subsidence_fields, subsidence_form)")
        Config.option!(s, "nudging", Bool; default = false,
                       doc = "nudging toward the dynamic driver's ls_nudge_* targets (ProfileNudging, time scale nudging_tau)")
    end

_lsf_geostrophic(opts) = opts.u_g != 0 || opts.v_g != 0
"Whether the `[large_scale_forcing]` options read the domain's dynamic driver."
lsf_reads_driver(opts) = opts.u_g === :driver || opts.v_g === :driver || opts.advective_tendencies ||
                         opts.subsidence || opts.nudging
enabled(::Type{<:LargeScale}, opts) = _lsf_geostrophic(opts) || lsf_reads_driver(opts)

function build(::Type{<:LargeScale}, opts, domain)
    dyn = get(domain.drivers, :dynamic, nothing)
    lsf_reads_driver(opts) && dyn === nothing &&
        throw(ConfigError("large_scale_forcing", "reads the domain's dynamic driver, which the domain has not (set dynamic_driver)"))
    F = dyn === nothing ? nothing : dynamic_forcing(dyn; t0 = clock(domain))
    function quantity(k)
        v = getproperty(opts, Symbol(k))
        v === :driver || return Float64(v)
        q = getproperty(F, Symbol(uppercase(k[1:1]) * k[2:end]))
        q === nothing && throw(ConfigError("large_scale_forcing.$k", "the dynamic driver has no ls_$k table"))
        return q
    end
    U = quantity("u_g"); V = quantity("v_g")
    extra = Any[]
    for (on, T, what) in ((opts.advective_tendencies, AdvectiveTendencies, "advective tendencies (ls_*_adv)"),
                          (opts.subsidence, Subsidence, "subsidence (ls_w_subs)"),
                          (opts.nudging, ProfileNudging, "nudging targets (ls_nudge_*)"))
        on || continue
        x = findfirst(f -> f isa T, F.extra)
        x === nothing && throw(ConfigError("large_scale_forcing", "the dynamic driver has no $what"))
        push!(extra, F.extra[x])
    end
    return LargeScale(GeostrophicPressureGradient(U, V), Tuple(extra), U, V, Ref(clock(domain)))
end

stages(c::LargeScale) = c.zi === nothing ? (Stage(:clock), Stage(:body_force)) :
                                            (Stage(:clock), Stage(:body_force), Stage(:column_tendency))

run_stage!(c::LargeScale, d, ::Val{:clock}, dt) = (c.t0[] = clock(d); nothing)

# The Coriolis parameter of the domain's Coriolis component; without one,
# the core's own (0 on a model domain, whose rotation is the component's).
_coriolis_f(::Tuple{}, s) = Float64(s.f_coriolis)
_coriolis_f(cs::Tuple, s) = first(cs) isa Coriolis ? first(cs).f : _coriolis_f(Base.tail(cs), s)

"The start of the time window of the (sub)step of `c` on `d` ([`LargeScale`](@ref)'s `clock`)."
forcing_time(c::LargeScale, d) = c.clock === :core ? Float64(d.state.t) : c.t0[]

function run_stage!(c::LargeScale, d, ::Val{:body_force}, dt)
    s = d.state; t0 = forcing_time(c, d)
    c.geostrophic === nothing || apply_forcing!(s, c.geostrophic, t0, dt; f = _coriolis_f(d.components, s))
    apply_forcing!(s, c.extra, t0, dt)
    return nothing
end

run_stage!(c::LargeScale, d, ::Val{:column_tendency}, dt) = (track!(c.zi, d.state); nothing)

checkpoint_state(c::LargeScale) = c.zi === nothing ? nothing : (zi = checkpoint_state(c.zi),)
restore_checkpoint_state!(c::LargeScale, s::NamedTuple) = (restore_checkpoint_state!(c.zi, s.zi); c)
