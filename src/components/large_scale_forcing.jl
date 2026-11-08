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
    sweep::ColumnForcing          # the coefficients of the fused sweep's term (#533), scratch
end
LargeScale(geostrophic, extra::Tuple, U_g, V_g, t0::Base.RefValue{Float64}, clock::Symbol = :step, zi = nothing) =
    LargeScale(geostrophic, extra, U_g, V_g, t0, clock, zi, ColumnForcing())
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
                   doc = "Large-scale forcing (LargeScaleForcing): the geostrophic pressure gradient, the domain " *
                         "dynamic driver's advective tendencies, subsidence and nudging (its ls_* tables), and the " *
                         "forcing of a deep boundary-layer domain from a sounding driver (NEST-p3, #483: the " *
                         "urban nest parent's of #259): nudging toward the sounding with a boundary-layer-dependent " *
                         "weight, subsidence of a large-scale divergence, θ damping over the top.") do s
        for (k, c) in (("u_g", "u"), ("v_g", "v"))
            Config.option!(s, k, Config.OneOf(Float64, Symbol); default = 0.0, allowed = [:driver, :sounding],
                           units = "m s^-1",
                           doc = "the geostrophic wind $c: a number; \"driver\": the dynamic driver's ls_$k table " *
                                 "(height, time); \"sounding\": the sounding driver's wind (ls_profile_$c) at max(z, z_g), " *
                                 "the wind above the boundary layer taken as geostrophic")
        end
        Config.option!(s, "z_g", Float64; default = NaN, units = "m",
                       doc = "with \"sounding\": the height of the geostrophic wind; nan: 1.2 × the largest parcel " *
                             "boundary-layer height of the sounding's launches")
        Config.option!(s, "advective_tendencies", Bool; default = false,
                       doc = "the dynamic driver's advective tendencies ls_u_adv, ls_v_adv, ls_theta_adv, ls_qv_adv (AdvectiveTendencies)")
        Config.option!(s, "subsidence", Bool; default = false,
                       doc = "the dynamic driver's subsidence velocity ls_w_subs (Subsidence; its fields and form from the " *
                             "driver's attributes subsidence_fields, subsidence_form)")
        Config.option!(s, "nudging", Bool; default = false,
                       doc = "nudging toward the dynamic driver's ls_nudge_* targets (ProfileNudging, time scale nudging_tau)")
        Config.option!(s, "sounding_nudging_tau", Float64; default = Inf, range = (0, Inf), open_lower = true, units = "s",
                       doc = "nudging of θ (and qv) toward the sounding driver's profiles (ls_profile_theta, ls_profile_qv, " *
                             "linear in time between the launches) with this time scale, above nudging_factor × the " *
                             "running maximum of the domain's boundary-layer height (at least the sounding's); inf: off")
        Config.option!(s, "nudging_factor", Float64; default = 1.2, range = (0, Inf), open_lower = true, units = "1",
                       doc = "the sounding nudging acts above nudging_factor × the tracked boundary-layer height")
        Config.option!(s, "nudging_depth", Float64; default = 200.0, range = (0, Inf), units = "m",
                       doc = "depth of the sounding nudging's ramp (sin²) above that height")
        Config.option!(s, "divergence", Float64; default = 0.0, units = "s^-1",
                       doc = "large-scale divergence D of the subsidence w_s = -D z (local form, on θ and qv); 0: off")
        Config.option!(s, "damping_rate", Float64; default = 0.0, range = (0, Inf), units = "s^-1",
                       doc = "θ reference damping (ReferenceDamping) at this largest rate over the top damping_frac of the domain; 0: off")
        Config.option!(s, "damping_frac", Float64; default = 0.25, range = (0, 1), open_lower = true, units = "1",
                       doc = "the damping acts over the top damping_frac of the domain (the sponge profile)")
        Config.option!(s, "damping_target", Symbol; default = :initial, allowed = [:initial, :sounding],
                       doc = "the θ the damping relaxes toward: the domain's initial profile, or the sounding driver's (ls_profile_theta)")
    end

_lsf_geostrophic(opts) = opts.u_g != 0 || opts.v_g != 0
_lsf_sounding(opts) = opts.u_g === :sounding || opts.v_g === :sounding || isfinite(opts.sounding_nudging_tau) ||
                      (opts.damping_rate > 0 && opts.damping_target === :sounding)
"Whether the `[large_scale_forcing]` options read the domain's dynamic driver."
lsf_reads_driver(opts) = opts.u_g === :driver || opts.v_g === :driver || opts.advective_tendencies ||
                         opts.subsidence || opts.nudging || _lsf_sounding(opts)
enabled(::Type{<:LargeScale}, opts) = _lsf_geostrophic(opts) || lsf_reads_driver(opts) || opts.divergence != 0 ||
                                      opts.damping_rate > 0

# a sounding table of the dynamic driver (on the model time: shifted by the run's start t0)
function _sounding_table(dyn, k::Symbol, t0)
    tab = get(dyn.tables, k, nothing)
    tab === nothing && throw(ConfigError("large_scale_forcing", "the dynamic driver has no ls_$k: a sounding driver " *
                                                                "(AeolusPrep.jl: aeolusprep dynamic --from-sounding)"))
    return t0 == 0 ? tab : shift_table(tab, t0)
end

"""
    sounding_zi_max(θ, qv) -> m

The largest parcel boundary-layer height (`boundary_layer_height`,
`:parcel`, on θ_v) of the launches of a sounding (the time levels of its
θ and qv tables, `ForcingTable`s): 1356 m over the six Lindenberg
launches of 29/30 June 2019 in `berlin_20190629_dynamic.nc` (#247; the
10:45 UTC release alone gives 1171 m).
"""
function sounding_zi_max(θ, qv)
    zi = 0.0
    for t in θ.t
        θv = [virtual_temperature(forcing_value(θ, z, t), PhasePartition(forcing_value(qv, z, t))) for z in θ.z]
        zi = max(zi, boundary_layer_height(θ.z, θv; method = :parcel))
    end
    return zi
end

"""
    sounding_geostrophic(dyn, z_g; t0 = 0) -> (U_g, V_g)

The geostrophic wind of a sounding driver `dyn`, functions `(z, t)` of
the model time (the driver's times less `t0`, the run's start): its wind
(`ls_profile_u`, `ls_profile_v`, linear in time between the launches) at
``\\max(z, z_g)`` — the wind above the boundary layer is taken as
geostrophic and held constant with height below `z_g` (no thermal wind
inside the boundary layer); `z_g = NaN` is 1.2 × [`sounding_zi_max`](@ref).
The one definition of a domain's sounding geostrophic wind (the nest
parent of #259 and its urban child's `[geostrophic]`).
"""
function sounding_geostrophic(dyn, z_g::Real; t0::Real = 0.0)
    Tu, Tv = _sounding_table(dyn, :profile_u, t0), _sounding_table(dyn, :profile_v, t0)
    isnan(z_g) && (z_g = 1.2 * sounding_zi_max(_sounding_table(dyn, :profile_theta, t0), _sounding_table(dyn, :profile_qv, t0)))
    zg = Float64(z_g)
    return ((z, t) -> forcing_value(Tu, max(z, zg), t), (z, t) -> forcing_value(Tv, max(z, zg), t))
end

# the domain height (m) above its ground
_domain_height(m) = hasproperty(m, :zf) ? m.zf[m.Hz+m.Nz+1] - m.zf[m.Hz+1] : m.Lz

function build(::Type{<:LargeScale}, opts, domain)
    dyn = get(domain.drivers, :dynamic, nothing)
    lsf_reads_driver(opts) && dyn === nothing &&
        throw(ConfigError("large_scale_forcing", "reads the domain's dynamic driver, which the domain has not (set dynamic_driver)"))
    t0 = clock(domain)
    F = dyn === nothing ? nothing : dynamic_forcing(dyn; t0 = t0)
    sg = (opts.u_g === :sounding || opts.v_g === :sounding) ? sounding_geostrophic(dyn, opts.z_g; t0 = t0) : nothing
    function quantity(k)
        v = getproperty(opts, Symbol(k))
        v === :sounding && return k == "u_g" ? sg[1] : sg[2]
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
    # the forcing of a deep boundary-layer domain (the urban nest parent of
    # #259): sounding nudging above the tracked boundary layer, subsidence of
    # a large-scale divergence, θ damping over the top
    sim = domain.state
    moist = hasproperty(sim, :has_moisture) && sim.has_moisture
    zi = nothing
    if isfinite(opts.sounding_nudging_tau)
        Tθ, Tq = _sounding_table(dyn, :profile_theta, t0), _sounding_table(dyn, :profile_qv, t0)
        zi = BoundaryLayerTracker(; floor = sounding_zi_max(Tθ, Tq), every = 10)
        targets = moist ? (θ = Tθ, qv = Tq) : (θ = Tθ,)
        push!(extra, ProfileNudging(targets, opts.sounding_nudging_tau;
                                    weight = NudgingWeight(_ -> nudging_height(zi); factor = opts.nudging_factor,
                                                           depth = opts.nudging_depth)))
    end
    D = opts.divergence
    D != 0 && push!(extra, Subsidence((z, t) -> -D * z; form = :local))
    if opts.damping_rate > 0
        H = _domain_height(grid(domain))
        target = if opts.damping_target === :sounding
            _sounding_table(dyn, :profile_theta, t0)
        else
            θf = get(domain.drivers, :theta_profile, nothing)
            θf === nothing && throw(ConfigError("large_scale_forcing.damping_target",
                                                "the domain has no initial θ profile to damp toward"))
            (z, t) -> θf(z)
        end
        push!(extra, ReferenceDamping(:θ, sponge_profile(opts.damping_rate, (1 - opts.damping_frac) * H, H), target))
    end
    # (a zero geostrophic wind with the driver's ls_* forcing keeps its no-op term, as before #483)
    geo = _lsf_geostrophic(opts) || opts.advective_tendencies || opts.subsidence || opts.nudging ?
          GeostrophicPressureGradient(U, V) : nothing
    return LargeScale(geo, Tuple(extra); U_g = U, V_g = V, zi = zi, t0 = t0)
end

stages(c::LargeScale) = c.zi === nothing ? (Stage(:clock), Stage(:body_force)) :
                                            (Stage(:clock), Stage(:body_force), Stage(:column_tendency))

run_stage!(c::LargeScale, d, ::Val{:clock}, dt) = (c.t0[] = clock(d); nothing)

# The Coriolis parameter of the domain's Coriolis component; without one,
# the core's own (0 on a model domain, whose rotation is the component's).
_coriolis_f(::Tuple{}, s) = Float64(s.f_coriolis)
_coriolis_f(cs::Tuple, s) = first(cs) isa Coriolis ? first(cs).f : _coriolis_f(Base.tail(cs), s)

"The start of the time window of the (sub)step of `c` on `d` ([`LargeScale`](@ref)'s `clock`)."
forcing_time(c::LargeScale, d) = _forcing_time(c, d.state)
_forcing_time(c::LargeScale, sim) = c.clock === :core ? Float64(sim.t) : c.t0[]

function run_stage!(c::LargeScale, d, ::Val{:body_force}, dt)
    s = d.state; t0 = forcing_time(c, d)
    c.geostrophic === nothing || apply_forcing!(s, c.geostrophic, t0, dt; f = _coriolis_f(d.components, s))
    apply_forcing!(s, c.extra, t0, dt)
    return nothing
end

run_stage!(c::LargeScale, d, ::Val{:column_tendency}, dt) = (track!(c.zi, d.state); nothing)

# --- in the fused sweep (#533 plan step 3) -----------------------------------
# Every forcing is, per level, affine in its field's own column; the
# reductions it needs (the plane means of nudging, mean-gradient
# subsidence, a damping toward the slab mean) are of the stage-start state,
# which the sweep only reads until the predictor overwrites a row. So
# `prepare!` computes the per-level coefficients before the sweep
# (`LargeScaleForcing.column_forcing!`, with the time window and Coriolis
# parameter of `run_stage!`) and the term adds them row by row. The
# whole-field `run_stage!` above stays the reference.

"""
    LargeScaleTerm

The fused-sweep term of a [`LargeScale`](@ref) component (#533): its
forcings' column coefficients (`LargeScaleForcing.ColumnForcing`, held by
the component) computed in `prepare!` from the stage-start state, added
by the sweep's row kernel; `f` is the Coriolis parameter the geostrophic
pressure gradient takes. A component whose forcings all have a column
form (`LargeScaleForcing.column_forcing_supported`: every forcing of
`LargeScaleForcing` but a damping toward a 3-D target) has one.
"""
struct LargeScaleTerm{C<:LargeScale} <: FiniteVolume.SweepTerm
    c::C
    f::Float64
end
FiniteVolume.sweep_term(c::LargeScale, d) =
    column_forcing_supported(c.geostrophic) && column_forcing_supported(c.extra) ?
        LargeScaleTerm(c, _coriolis_f(d.components, d.state)) : nothing
FiniteVolume.footprint(::LargeScaleTerm) = (0, 0, 1)        # the local subsidence's upwind level
function FiniteVolume.prepare!(t::LargeScaleTerm, sim, Δt)
    c = t.c; col = c.sweep
    t0 = _forcing_time(c, sim)
    reset_column_forcing!(col, sim)
    c.geostrophic === nothing || column_forcing!(col, sim, c.geostrophic, t0, Δt; f = t.f)
    column_forcing!(col, sim, c.extra, t0, Δt)
    finish_column_forcing!(col)
    return nothing
end
FiniteVolume.add_row!(t::LargeScaleTerm, buf, a, b, s, j, k, ii, st) =
    add_column_rows!(t.c.sweep, buf, a, b, s, j, k, ii, st)

checkpoint_state(c::LargeScale) = c.zi === nothing ? nothing : (zi = checkpoint_state(c.zi),)
restore_checkpoint_state!(c::LargeScale, s::NamedTuple) = (restore_checkpoint_state!(c.zi, s.zi); c)
