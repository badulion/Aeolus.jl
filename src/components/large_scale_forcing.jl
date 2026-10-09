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
"""
struct LargeScale{G, E<:Tuple, U, V} <: AbstractComponent
    geostrophic::G
    extra::E
    U_g::U
    V_g::V
    t0::Base.RefValue{Float64}
end

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
        v = opts[k]
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

stages(::LargeScale) = (Stage(:clock), Stage(:body_force))

run_stage!(c::LargeScale, d, ::Val{:clock}, dt) = (c.t0[] = clock(d); nothing)

# The Coriolis parameter of the domain's Coriolis component (0 without one).
_coriolis_f(::Tuple{}) = 0.0
_coriolis_f(cs::Tuple) = first(cs) isa Coriolis ? first(cs).f : _coriolis_f(Base.tail(cs))

function run_stage!(c::LargeScale, d, ::Val{:body_force}, dt)
    s = d.state; t0 = c.t0[]
    apply_forcing!(s, c.geostrophic, t0, dt; f = _coriolis_f(d.components))
    apply_forcing!(s, c.extra, t0, dt)
    return nothing
end
