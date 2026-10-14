# Nesting: two-way feedback (N-nest-e, #263).
#
#   * TwoWayFeedback — the configuration of the child → parent feedback:
#     anterpolation (Clark & Farley 1984) with an optional relaxation, the
#     buffer zone, the canopy restriction, the inflow exclusion band,
#     scalar refluxing and the re-projection of the parent;
#   * NestScalarFlux — the transport whose face flux a register books;
#   * NestFluxRegisters — Berger–Colella flux registers on the interface
#     planes, accumulated at every stage of every parent step and child
#     substep through a body-force hook (exactly the fluxes the cores used),
#     and by separate transports through book_nest_flux! (#327 M3);
#   * anterpolate!, two_way_feedback!, feedback_mask — the feedback itself.
#
# All operators come from N-nest-a (#250) / N-nest-b (#256): the restriction
# is `restrict_to_parent!` (volume- or mass-weighted, solid-aware), the
# refluxing `apply_flux_correction!`, the face flux of a scalar
# `AdvectionSchemes.scalar_face_flux` (the kernel of `boundary_scalar_flux`
# too, #327 M2) and the closures' `sgs_scalar_face_diffusivity`.

const _SIDE_NAMES = (:west, :east, :south, :north, :bottom, :top)
_side_axis(s::Int) = (s + 1) ÷ 2
_side_high(s::Int) = iseven(s)

# =========================================================================
# Configuration
# =========================================================================

"""
    NestScalarFlux(; scheme = :centered, molecular = false, sgs = :none, applied_by = :core,
                   child_applied_by = applied_by)

The transport of a cell-centred scalar whose interface flux a two-way
nest books in its flux registers ([`TwoWayFeedback`](@ref)).

`applied_by = :core` (default): the scalar is transported **inside** the
core's stages (the core's own `θ`/`qv` transport, or an owner that updates
it in the `body_force!` hook), and the run's register hook books, at every
stage with the core's stage weight, the advective face flux of `scheme` (a
name of [`boundary_scalar_flux`](@ref Aeolus.AdvectionSchemes.boundary_scalar_flux)
or a reconstruction descriptor; the Boussinesq and sound-proof cores
advect `θ` and `qv` with `:centered`), plus the molecular diffusion `ν∇ϕ`
of the simulation (`molecular = true`, when the simulation has viscosity)
and the SGS flux of the domain's closure: `sgs = :theta` with θ's
diffusivity
([`sgs_scalar_face_diffusivity`](@ref Aeolus.Operators.sgs_scalar_face_diffusivity)),
`sgs = :tracer` with the diffusivity the core gives a tracer
([`sgs_tracer_diffusivity!`](@ref Aeolus.Operators.sgs_tracer_diffusivity!),
as for `qv`), `:none` without.

`applied_by = :transport`: the scalar is transported by a separate
transport (a WENO tracer slot after the dynamics, as in the Berlin case),
whose state, velocity and time weight the core's stages do not see. The
hook books nothing for it; the transport books **exactly the flux it
applies**, at the state it applies it to, by calling
[`book_nest_flux!`](@ref) (#327, M3). `scheme`, `molecular` and `sgs` are
then unused.

`applied_by` is the parent's transport and `child_applied_by` (default:
the same) the child's: a scalar that the parent's core transports and the
child moves by a separate transport (the urban site's θ and qv in a model
parent, NEST-p3 #483) is booked by the run's hook in the parent's stages
and by `book_nest_flux!` in the child.

`θ` and a core-transported `qv` get their models from the simulations; a
tracer of the field set (also `qv` when it is a tracer, as in a moist
sound-proof nest) is refluxed only when its owner declares how it is
transported, `TwoWayFeedback(tracer_fluxes = (:co2 => NestScalarFlux(),))`.
"""
struct NestScalarFlux{R}
    rec::R
    molecular::Bool
    sgs::Symbol
    applied_by::Symbol
    child_applied_by::Symbol
end

function NestScalarFlux(; scheme = :centered, molecular::Bool = false, sgs::Symbol = :none,
                        applied_by::Symbol = :core, child_applied_by::Symbol = applied_by)
    sgs in (:none, :theta, :tracer) ||
        throw(ArgumentError("NestScalarFlux: sgs must be :none, :theta or :tracer, got :$sgs"))
    for a in (applied_by, child_applied_by)
        a in (:core, :transport) ||
            throw(ArgumentError("NestScalarFlux: applied_by and child_applied_by are :core or :transport, got :$a"))
    end
    return NestScalarFlux(_flux_reconstruction(scheme), molecular, sgs, applied_by, child_applied_by)
end

# whether the core's hook books the scalar on side `which` (:parent or :child), or on either side
_by_core(m::NestScalarFlux, which::Symbol) = (which === :parent ? m.applied_by : m.child_applied_by) === :core
_by_core(m::NestScalarFlux) = _by_core(m, :parent) || _by_core(m, :child)

"""
    TwoWayFeedback(; buffer = 2, z_anterp = -Inf, τ = nothing, exclude_width = 0,
                   exclude_sides = :inflow, reflux = true, reproject = true,
                   tracer_fluxes = (), theta_flux = nothing)

The child → parent feedback of a two-way [`NestedRun`](@ref)
(`feedback = TwoWayFeedback(…)`, or `feedback = :two_way` for the
defaults), applied once per parent step after the child substeps:

 1. **Anterpolation** (Clark & Farley 1984; Hellsten et al. 2021, §3.5,
    Eq. 15). A covered parent cell takes the conservative child mean
    ``⟨φ_C⟩`` of `θ`, `qv`, the tracers and the velocities: the N-nest-a
    / N-nest-b restriction ([`restrict_to_parent!`](@ref), volume- or
    reference-mass-weighted, solid-aware). With a relaxation time `τ` it
    is relaxed instead,

        φ_P ← φ_P + (Δt_p/τ)(⟨φ_C⟩ − φ_P),

    so a covered-parent deviation decays as ``(1 − Δt_p/τ)^n``.
    `τ = nothing` (default) is direct replacement, as in PALM, and so is
    any `τ ≤ Δt_p`. The relaxation is our extension, not PALM's.
 2. **Buffer zone.** No anterpolation in the outermost `buffer` parent
    cells next to every open (nest) face of the child (PALM's default is
    2): there the child's solution is still dominated by its boundary
    data.
 3. **Canopy restriction.** Only parent cells whose lower face is at or
    above `z_anterp` are anterpolated (PALM leaves the canopy layer out,
    where the coarse parent cannot represent what the child resolves).
 4. **Inflow exclusion.** A band of `exclude_width` parent cells at the
    child's inflow faces is left out too, so synthetic inflow turbulence
    (N-inflow-c) is never fed back. `exclude_sides = :inflow` picks the
    open faces whose mean normal velocity points into the child at the
    time of the feedback; a tuple of side names (`(:west,)`) or an
    `NTuple{6,Bool}` (west, east, south, north, bottom, top) fixes them.
 5. **Scalar refluxing** (Berger & Colella 1989): every parent step and
    child substep books the flux of `θ`, `qv` and the declared tracers
    through the interface planes (advective, molecular and SGS — exactly
    the face fluxes the cores used, at every stage with the stage weights
    of the core) into flux registers; the coarse-minus-fine mismatch is
    put into the parent cells just outside the nest
    ([`apply_flux_correction!`](@ref)). The composite integral
    ``∫φ dV`` (``∫Wφ dV`` for a sound-proof pair) is then conserved to
    round-off. `tracer_fluxes` declares the transport of the tracers
    (`name => NestScalarFlux(…)`; `:qv` too when it is a tracer of the
    field set rather than a core prognostic); undeclared tracers are not
    refluxed. A tracer moved by a separate transport is declared with
    `applied_by = :transport` and booked by that transport
    ([`book_nest_flux!`](@ref)). `theta_flux` replaces the model of `θ`
    (default: the cores' centred advection, molecular and SGS diffusion)
    for a user core that transports `θ` otherwise.
 6. **Re-projection** of the parent after the feedback, so ``∇·u = 0``
    holds in the parent again (anterpolated faces next to the buffer and
    the interface flux correction leave the parent divergent).

The anterpolated velocity faces are those strictly inside the covered
region whose two parent cells are both anterpolated; the interface faces
themselves get the child's flux from the interface flux correction
(`flux_correction` of the run, on the open faces only), less the part of
the child's normal velocity its boundary data declare synthetic
([`synthetic_normal_velocity`](@ref Aeolus.BoundaryConditions.synthetic_normal_velocity):
inflow turbulence, which would otherwise return to the child through the
next window's boundary data and accumulate, #418).

Refluxing needs a core whose stage structure is known
([`nest_stage_weights`](@ref)): `step!` (Boussinesq or sound-proof; one
stage), `step_rk3!` (weights 1/6, 1/6, 2/3), or a user core that declares
its weights; the IMEX cores (`step_imex!`, implicit θ diffusion) are
refused with `reflux = true`. A run whose scalars are all booked by their
transports (`theta_flux` and every `tracer_fluxes` entry with
`applied_by = :transport`) needs no stage weights: any core runs, as the
Berlin case's stage loop does (#267). `qv` can be declared in
`tracer_fluxes` wherever it is a tracer of at least one domain (a moist
Boussinesq parent around a child that carries `qv` as a tracer). The SGS closure needs a two-point face
diffusivity. Exact conservation also needs `sponge = nothing` (a sponge is
a source).

References: Clark, T. L. & Farley, R. D. (1984), *J. Atmos. Sci.* 41,
329–350; Berger, M. J. & Colella, P. (1989), *J. Comput. Phys.* 82,
64–84; Hellsten, A. et al. (2021), *Geosci. Model Dev.* 14, 3185–3214,
§3.5 (anterpolation, buffer, canopy) and §4.1 (the spurious secondary
circulation of two-way nests in a convective boundary layer).
"""
struct TwoWayFeedback
    buffer::Int
    z_anterp::Float64
    τ::Union{Nothing,Float64}
    exclude_width::Int
    exclude_sides::Union{Symbol,NTuple{6,Bool}}
    reflux::Bool
    reproject::Bool
    tracer_fluxes::Vector{Pair{Symbol,NestScalarFlux}}
    theta_flux::Union{Nothing,NestScalarFlux}
end

function TwoWayFeedback(; buffer::Integer = 2, z_anterp::Real = -Inf,
                        τ::Union{Nothing,Real} = nothing, exclude_width::Integer = 0,
                        exclude_sides = :inflow, reflux::Bool = true, reproject::Bool = true,
                        tracer_fluxes = (), theta_flux::Union{Nothing,NestScalarFlux} = nothing)
    buffer >= 0 || throw(ArgumentError("TwoWayFeedback: buffer must be ≥ 0, got $buffer"))
    exclude_width >= 0 ||
        throw(ArgumentError("TwoWayFeedback: exclude_width must be ≥ 0, got $exclude_width"))
    τ === nothing || τ > 0 || throw(ArgumentError("TwoWayFeedback: τ must be > 0, got $τ"))
    ex = _exclude_sides(exclude_sides)
    tf = Pair{Symbol,NestScalarFlux}[]
    for pr in tracer_fluxes
        first(pr) isa Symbol && last(pr) isa NestScalarFlux ||
            throw(ArgumentError("TwoWayFeedback: tracer_fluxes are `name => NestScalarFlux(…)`"))
        first(pr) in (:θ, :theta) &&
            throw(ArgumentError("TwoWayFeedback: θ is not a tracer; declare its transport " *
                                "with theta_flux = NestScalarFlux(…)"))
        push!(tf, first(pr) => last(pr))
    end
    return TwoWayFeedback(Int(buffer), Float64(z_anterp),
                          τ === nothing ? nothing : Float64(τ), Int(exclude_width), ex,
                          reflux, reproject, tf, theta_flux)
end

_exclude_sides(s::Symbol) = s === :inflow ? s :
    throw(ArgumentError("TwoWayFeedback: exclude_sides must be :inflow, side names or " *
                        "an NTuple{6,Bool}, got :$s"))
_exclude_sides(t::NTuple{6,Bool}) = t
function _exclude_sides(t::Tuple{Vararg{Symbol}})
    for s in t
        s in _SIDE_NAMES || throw(ArgumentError("TwoWayFeedback: unknown side :$s " *
                                                "(use $(_SIDE_NAMES))"))
    end
    return ntuple(i -> _SIDE_NAMES[i] in t, 6)
end

Base.show(io::IO, f::TwoWayFeedback) =
    print(io, "TwoWayFeedback(buffer=", f.buffer, ", z_anterp=", f.z_anterp, ", τ=",
          f.τ === nothing ? "Δt_p" : f.τ, ", exclude_width=", f.exclude_width,
          ", reflux=", f.reflux, ", reproject=", f.reproject, ")")

# =========================================================================
# Flux registers
# =========================================================================

"""
    NestFluxRegisters

The Berger–Colella flux registers of a two-way [`NestedRun`](@ref): for
every refluxed scalar (`θ`, `qv`, declared tracers) and every interface
plane (the child's open faces), the time integral `∫ F dt` of the face
flux density (`∫ W F dt` for a sound-proof pair) through every parent
face of the plane and every child face of the plane, and the volume (mass)
flux `∫ W u_n dt` that carried it (for the volume-consistent refluxing of
[`two_way_feedback!`](@ref)). Scalars transported inside the cores
(`applied_by = :core`) are booked by a hook the run chains in front of
each domain's `body_force!`, at every stage of every attempted step, with
the stage weight of the core; scalars moved by a separate transport
(`applied_by = :transport`) are booked by that transport through
[`book_nest_flux!`](@ref). A rejected step's bookings are discarded with
it; the registers are consumed by the refluxing of
[`two_way_feedback!`](@ref). Only the planes are stored, not whole face
fields.
"""
mutable struct NestFluxRegisters{WP<:Tuple, WC<:Tuple, KP, KC, FP<:Field, FC<:Field,
                                 SP<:Field, SC<:Field, MP, MC}
    names::Vector{Symbol}
    models::Vector{NestScalarFlux}
    pfield::Vector{FP}
    cfield::Vector{FC}
    sides::NTuple{6,Bool}
    pacc::Vector{Vector{Matrix{Float64}}}     # committed, per scalar, per side
    cacc::Vector{Vector{Matrix{Float64}}}
    ppend::Vector{Vector{Matrix{Float64}}}    # the step being attempted
    cpend::Vector{Vector{Matrix{Float64}}}
    pvacc::Vector{Vector{Matrix{Float64}}}    # the volume flux of each scalar's transport
    cvacc::Vector{Vector{Matrix{Float64}}}
    pvpend::Vector{Vector{Matrix{Float64}}}
    cvpend::Vector{Vector{Matrix{Float64}}}
    pvstage::Vector{Matrix{Float64}}           # scratch: one stage's volume flux
    cvstage::Vector{Matrix{Float64}}
    pweights::WP
    cweights::WC
    pstage::Int
    cstage::Int
    popen::Bool                                # an attempt of the domain is under way
    copen::Bool
    psgs::KP                                   # the domains' (sgs, eddy_viscosity!) keywords
    csgs::KC
    pscratch::SP
    cscratch::SC
    mismatch::NestFluxMismatch{Float64}        # scratch of the refluxing (parent planes)
    pW::MP                                     # the (cell, face) mass weights of each side's
    cW::MC                                     # fluxes (nothing: the Boussinesq volume flux)
end

"""
    nest_stage_weights(core, args...) -> NTuple or nothing

The weights with which the stages of a dynamical core combine into its
step, as the flux registers of a two-way [`NestedRun`](@ref) need them:
`(1.0,)` for `step!` (Boussinesq or sound-proof), `(1/6, 1/6, 2/3)` for
`step_rk3!` (the Shu–Osher combination of its three Euler stages), and
`nothing` — refluxing refused — for any other core. A user core
`f(sim, dt, args...; body_force!, kw...)` that calls `body_force!` once
per stage (for instance one that takes a `step!` and then moves its
tracers) declares its weights by adding a method,

```julia
Aeolus.Nesting.nest_stage_weights(::typeof(my_core), args...) = (1.0,)
```

The IMEX cores (`step_imex!`) are refused: their `θ` diffusion is
implicit, so the diffusive interface flux is not the one the explicit
hook would see.
"""
nest_stage_weights(core, args...) = nothing
nest_stage_weights(::typeof(step!), args...) = (1.0,)
nest_stage_weights(::typeof(step_rk3!), args...) = (1.0 / 6, 1.0 / 6, 2.0 / 3)

# Stage weights of a core; `nothing` = unknown.
_stage_weights(a::DomainAdvance) = nest_stage_weights(a.core, a.args...)

_sgs_keywords(a::DomainAdvance) = (get(a.kwargs, :sgs, nothing),
                                   get(a.kwargs, :eddy_viscosity!, nothing))

_domain_closure(sim, kw) = _active_sgs(sim, kw[1], kw[2])

# (parent, child) plane sizes of a register on side `s`: the conservation
# box's parent cells and child cells along the two tangential axes (#328;
# the whole child, nn·r, on an aligned nest).
function _plane_sizes(nest::NestedGrid, s::Int)
    ax = _side_axis(s)
    nn = (nest.nx, nest.ny, nest.nz)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    a = nest.axes
    return (nn[t1], nn[t2]), (length(a[t1].cbox), length(a[t2].cbox))
end

function _new_planes(nest, sides, which::Int)
    return [sides[s] ? zeros(_plane_sizes(nest, s)[which]) : zeros(0, 0) for s in 1:6]
end

function _check_register_closure(sim, kw, what)
    c = _domain_closure(sim, kw)
    c === nothing && return nothing
    f = sgs_fields(c, sim)
    sgs_scalar_face_diffusivity(c, f, sim) === nothing &&
        throw(ArgumentError("TwoWayFeedback: the $what's SGS closure $(nameof(typeof(c))) " *
                            "has a tensor scalar flux, which the flux registers cannot " *
                            "book face by face; use TwoWayFeedback(reflux = false)"))
    return nothing
end

function _core_name(a::DomainAdvance)
    n = a.core isa Function ? nameof(a.core) : nameof(typeof(a.core))
    return isempty(a.args) ? string(n) : string(n, "(", join(map(x -> nameof(typeof(x)), a.args), ", "), ")")
end

function NestFluxRegisters(nest::NestedGrid, parent, child, s::NestInterpState,
                           fb::TwoWayFeedback, ap::DomainAdvance, ac::DomainAdvance,
                           conv = nothing)
    names = Symbol[:θ]
    models = NestScalarFlux[fb.theta_flux === nothing ?
                            NestScalarFlux(; molecular = true, sgs = :theta) : fb.theta_flux]
    pf = Field[parent.θ]; cf = Field[child.θ]       # narrowed to concrete types below
    declared = Dict(fb.tracer_fluxes)
    core_qv = _moist(parent)                  # qv is a core prognostic (moist Boussinesq)
    # qv may be declared where it is a tracer of at least one domain (a moist
    # parent around a child that carries qv as a tracer, as in Berlin #267)
    core_qv && _moist(child) && haskey(declared, :qv) &&
        throw(ArgumentError("TwoWayFeedback: qv is transported by the core here (a moist " *
                            "BoussinesqSimulation pair) and booked by the run; declare it in " *
                            "tracer_fluxes only where it is a tracer of the field set"))
    for i in eachindex(s.names)
        n = s.names[i]
        model = haskey(declared, n) ? declared[n] :
                n === :qv && core_qv ? NestScalarFlux(; molecular = true, sgs = :tracer) : nothing
        model === nothing && continue
        push!(names, n); push!(models, model)
        push!(pf, s.parent[i]); push!(cf, s.child[i])
    end
    # the stage weights are needed only when the cores' hooks book a scalar:
    # with every scalar booked by its transport (book_nest_flux!) any core runs
    wp = _stage_weights(ap); wc = _stage_weights(ac)
    if any(_by_core, models)
        bp = any(m -> _by_core(m, :parent), models); bc = any(m -> _by_core(m, :child), models)
        ((bp && wp === nothing) || (bc && wc === nothing)) &&
            throw(ArgumentError("TwoWayFeedback: refluxing needs the stage weights of each " *
                                "domain's core (step!, step_rk3!, or a user core declaring " *
                                "nest_stage_weights); got parent $(_core_name(ap)), child " *
                                "$(_core_name(ac)). The IMEX cores are not supported (implicit " *
                                "θ diffusion). Use TwoWayFeedback(reflux = false), or book " *
                                "every scalar with book_nest_flux! (applied_by = :transport)"))
        bp ? _check_register_closure(parent, _sgs_keywords(ap), "parent") : wp === nothing && (wp = ())
        bc ? _check_register_closure(child, _sgs_keywords(ac), "child") : wc === nothing && (wc = ())
    else
        wp === nothing && (wp = ()); wc === nothing && (wc = ())
    end
    for k in keys(declared)
        k in s.names || throw(ArgumentError("TwoWayFeedback: tracer :$k is not a scalar of " *
                                            "the run's field set $(s.names)"))
    end
    sides = nest_open_sides(child)
    _warn_internal_walls(nest, sides)
    mk(which) = [_new_planes(nest, sides, which) for _ in 1:length(names)]
    # the parent planes of every axis that has an open side (the refluxing
    # mismatch: zero on a non-open side of such an axis)
    mm = NestFluxMismatch{Float64}(ntuple(s -> (sides[2_side_axis(s)-1] || sides[2_side_axis(s)]) ?
                                               zeros(_plane_sizes(nest, s)[1]) : zeros(0, 0), 6)...)
    FP = mapreduce(typeof, typejoin, pf); FC = mapreduce(typeof, typejoin, cf)
    return NestFluxRegisters(names, models, convert(Vector{FP}, pf), convert(Vector{FC}, cf),
                             sides, mk(1), mk(2), mk(1), mk(2),
                             mk(1), mk(2), mk(1), mk(2),
                             _new_planes(nest, sides, 1), _new_planes(nest, sides, 2),
                             wp, wc, 0, 0, false, false, _sgs_keywords(ap), _sgs_keywords(ac),
                             Field(CellCenter(), nest.parent_mesh),
                             Field(CellCenter(), nest.child_mesh), mm,
                             _register_weights(parent, conv), _register_weights(child, conv))
end

# A child face that is not open but lies inside the parent (a lid below the
# parent top, a side wall inside a wider parent) blocks the child but not
# the parent: the parent's flux through that plane has no fine counterpart
# and no register, so the composite integral is not conserved across it.
function _warn_internal_walls(nest::NestedGrid, sides::NTuple{6,Bool})
    pm = nest.parent_mesh
    inner = (nest.ix > 1, nest.ix + nest.nx - 1 < pm.Nx,
             nest.iy > 1, nest.iy + nest.ny - 1 < pm.Ny,
             nest.iz > 1, nest.iz + nest.nz - 1 < pm.Nz)
    for s in 1:6
        inner[s] && !sides[s] &&
            @warn "TwoWayFeedback: the child's $(_SIDE_NAMES[s]) face is not open (nest) " *
                  "but lies inside the parent; the parent's flux through that plane has " *
                  "no fine counterpart, so refluxing cannot conserve the composite " *
                  "integral across it. Make it a :nest face." maxlog = 1
    end
    return nothing
end

# --- attempts: begin / commit -----------------------------------------------

function _begin_attempt!(r::NestFluxRegisters, which::Symbol)
    for planes in (which === :parent ? (r.ppend, r.pvpend) : (r.cpend, r.cvpend)),
        P in planes, M in P
        fill!(M, 0.0)
    end
    if which === :parent
        r.pstage = 0; r.popen = true
    else
        r.cstage = 0; r.copen = true
    end
    return r
end

function _commit_attempt!(r::NestFluxRegisters, which::Symbol)
    par = which === :parent
    for (acc, pend) in (par ? ((r.pacc, r.ppend), (r.pvacc, r.pvpend)) :
                              ((r.cacc, r.cpend), (r.cvacc, r.cvpend)))
        for n in eachindex(acc), s in 1:6
            acc[n][s] .+= pend[n][s]
        end
    end
    par ? (r.popen = false) : (r.copen = false)
    return r
end

function _clear_registers!(r::NestFluxRegisters)
    for P in (r.pacc, r.cacc, r.ppend, r.cpend, r.pvacc, r.cvacc, r.pvpend, r.cvpend),
        Q in P, M in Q
        fill!(M, 0.0)
    end
    return r
end

# The hook chained in front of a domain's body_force!: it runs inside every
# stage, after the tendencies and before the predictor, when the halos of
# the state, the velocities and the SGS coefficients are those the stage
# used. It books the scalars the core transports (applied_by = :core).
struct _RegisterHook{R<:NestFluxRegisters, N<:NestedGrid} <: Function
    regs::R
    nest::N
    which::Symbol
end

function (h::_RegisterHook)(sim, Δt)
    r = h.regs
    par = h.which === :parent
    if par
        r.pstage += 1; st = r.pstage; ws = r.pweights; planes = r.ppend; vplanes = r.pvpend
        fields = r.pfield; kw = r.psgs; scratch = r.pscratch; vst = r.pvstage
    else
        r.cstage += 1; st = r.cstage; ws = r.cweights; planes = r.cpend; vplanes = r.cvpend
        fields = r.cfield; kw = r.csgs; scratch = r.cscratch; vst = r.cvstage
    end
    any(m -> _by_core(m, h.which), r.models) || return nothing
    st <= length(ws) || error("NestFluxRegisters: stage $st of a core with " *
                              "$(length(ws)) stages")
    w = ws[st] * Float64(Δt)
    closure = _domain_closure(sim, kw)
    f = closure === nothing ? nothing : sgs_fields(closure, sim)
    W = par ? r.pW : r.cW
    # the stage's volume (mass) flux, the same for every core-transported scalar
    for s in 1:6
        r.sides[s] || continue
        fill!(vst[s], 0.0)
        _book_plane!(vst[s], sim, nothing, nothing, nothing, 0.0, W, h.nest, s, par, w,
                     sim.u, sim.v, sim.w)
    end
    for n in eachindex(r.names)
        _by_core(r.models[n], h.which) || continue
        ϕ = fields[n]
        n > 1 && !(r.names[n] === :qv && _moist(sim)) && fill_halos!(ϕ, sim.θ_bcs)   # tracers
        K = _face_diffusivity(r.models[n], closure, f, sim, ϕ, scratch)
        ν = r.models[n].molecular && sim.has_viscosity ? Float64(sim.ν) : 0.0
        for s in 1:6
            r.sides[s] || continue
            _book_plane!(planes[n][s], sim, ϕ, r.models[n].rec, K, ν, W, h.nest, s, par, w,
                         sim.u, sim.v, sim.w)
            vplanes[n][s] .+= vst[s]
        end
    end
    return nothing
end

_register_weights(::BoussinesqSimulation) = nothing
_register_weights(sim::MassWeightedSimulation) = constraint_weights(sim)

function _face_diffusivity(m::NestScalarFlux, closure, f, sim, ϕ, scratch)
    (m.sgs === :none || closure === nothing) && return nothing
    m.sgs === :theta && return sgs_scalar_face_diffusivity(closure, f, sim)
    κ = sgs_tracer_diffusivity!(scratch, closure, f, sim, ϕ)
    return κ === f.κh ? sgs_scalar_face_diffusivity(closure, f, sim) : (κ, κ, 1.0)
end

# The total flux density (advective + molecular + SGS, times W) through the
# face of storage index (i, j, k) normal to `ax`: the one per-face kernel
# `AdvectionSchemes.scalar_face_flux` (#327 M2: the same numbers as
# `boundary_scalar_flux`), with the face diffusivity ν + s·½(K_L + K_R) of
# the closure's `sgs_scalar_face_diffusivity`.
@inline function _face_total_flux(rec, ϕd, veld, ax::Int, i::Int, j::Int, k::Int, g,
                                  ν::Float64, K, W)
    D = K === nothing ? face_diffusivity(ν, nothing, 1.0, ax, i, j, k) :
                        face_diffusivity(ν, ax == 3 ? K[2] : K[1], K[3], ax, i, j, k)
    Fa, Fd, _ = scalar_face_flux(rec, ϕd, veld, ax, i, j, k, D, face_inv_spacing(g, ax, k))
    F = Fa + Fd
    W === nothing && return F
    @inbounds return (ax == 3 ? W[2][k] : W[1][k]) * F
end

# ϕ === nothing books the volume (mass) flux W u_n instead of a scalar flux.
# (u, v, w) is the velocity the transport used (the simulation's, unless a
# separate transport says otherwise).
function _book_plane!(M::Matrix{Float64}, sim, ϕ, rec, K, ν, W, nest::NestedGrid,
                      s::Int, parent::Bool, w::Float64, u = sim.u, v = sim.v, wv = sim.w)
    ax = _side_axis(s); hi = _side_high(s)
    g = sim.mesh
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    H = (g.Hx, g.Hy, g.Hz)
    veld = ax == 1 ? u.data : ax == 2 ? v.data : wv.data
    if parent
        oo = (nest.ix, nest.iy, nest.iz); nn = (nest.nx, nest.ny, nest.nz)
        base = H[ax] + oo[ax] + (hi ? nn[ax] : 0)
        o1 = H[t1] + oo[t1] - 1; o2 = H[t2] + oo[t2] - 1
    else
        # the child faces on the conservation box planes (the child's
        # boundary faces on an aligned nest, interior faces on a shrunk box)
        cb = (nest.axes[1].cbox, nest.axes[2].cbox, nest.axes[3].cbox)
        base = H[ax] + first(cb[ax]) + (hi ? length(cb[ax]) : 0)
        o1 = H[t1] + first(cb[t1]) - 1; o2 = H[t2] + first(cb[t2]) - 1
    end
    r = (1:size(M, 1), 1:size(M, 2))
    if ϕ === nothing
        launch!(Backends.backend(M), _book_volume_body!, r, M, veld, ax, t1, base, o1, o2, W, w)
        return M
    end
    launch!(Backends.backend(M), _book_flux_body!, r, M, ϕ.data, veld, rec, ax, t1, base, o1, o2,
            g, ν, K, W, w)
    return M
end

# The booked volume flux w·W·u_n, and the booked scalar flux, of one face.
@inline function _book_volume_body!(A, B, M, veld, ax, t1, base, o1, o2, W, w)
    idx = _pos(ax, t1, base, o1 + A, o2 + B)
    k = idx[3]
    Wk = W === nothing ? 1.0 : (ax == 3 ? W[2][k] : W[1][k])
    @inbounds M[A, B] += w * Wk * veld[idx[1], idx[2], k]
    return nothing
end
@inline function _book_flux_body!(A, B, M, ϕd, veld, rec, ax, t1, base, o1, o2, g, ν, K, W, w)
    idx = _pos(ax, t1, base, o1 + A, o2 + B)
    @inbounds M[A, B] += w * _face_total_flux(rec, ϕd, veld, ax, idx[1], idx[2], idx[3], g,
                                              ν, K, W)
    return nothing
end

# coarse − Σ fine on every interface plane, in the units of the registers ×
# the parent face area (a NestFluxMismatch for apply_flux_correction!), for
# the flux of ϕ − ϕ_ref: the booked scalar flux minus ϕ_ref times the booked
# volume flux.
function _register_mismatch(r::NestFluxRegisters, n::Int, nest::NestedGrid,
                            ϕref::Float64 = 0.0)
    rr = (nest.rx, nest.ry, nest.rz)
    out = r.mismatch                      # preallocated planes, refilled in place (#329)
    planes = (out.west, out.east, out.south, out.north, out.bottom, out.top)
    for s in 1:6
        ax = _side_axis(s)
        M = planes[s]
        isempty(M) && continue
        fill!(M, 0.0)
        r.sides[s] || continue
        P = r.pacc[n][s]; C = r.cacc[n][s]
        Pv = r.pvacc[n][s]; Cv = r.cvacc[n][s]
        t1, t2 = _tang(ax)
        if nest.aligned
            launch!(Backends.backend(M), _register_body!, (1:size(M, 1), 1:size(M, 2)), M, P, C,
                    Pv, Cv, ϕref, rr[t1], rr[t2], nest, ax)
        else
            a1 = nest.axes[t1]; a2 = nest.axes[t2]
            launch!(Backends.backend(M), _register_ov_body!, (1:size(M, 1), 1:size(M, 2)), M, P,
                    C, Pv, Cv, ϕref, a1, a2, a1.Hc + first(a1.cbox) - 1,
                    a2.Hc + first(a2.cbox) - 1, nest, ax)
        end
    end
    return out
end

# The general fine sum (#328): the overlap-area mean of the child register
# faces under the parent face (A, B) (the registers carry flux densities).
@inline function _register_ov_body!(A, B, M, P, C, Pv, Cv, ϕref, a1, a2, o1, o2, nest, ax)
    fine = 0.0; den = 0.0
    @inbounds for q2 in a2.ptr[B]:a2.ptr[B+1]-1, q1 in a1.ptr[A]:a1.ptr[A+1]-1
        w = a1.len[q1] * a2.len[q2]
        ia = a1.cidx[q1] - o1; ib = a2.cidx[q2] - o2
        fine += w * (C[ia, ib] - ϕref * Cv[ia, ib]); den += w
    end
    @inbounds M[A, B] = ((P[A, B] - ϕref * Pv[A, B]) - fine / den) * _face_area(nest, nothing, ax, B)
    return nothing
end

@inline function _register_body!(A, B, M, P, C, Pv, Cv, ϕref, r1, r2, nest, ax)
    inv = 1 / (r1 * r2)
    fine = 0.0
    @inbounds for b in 1:r2, a in 1:r1
        ia = (A - 1) * r1 + a; ib = (B - 1) * r2 + b
        fine += C[ia, ib] - ϕref * Cv[ia, ib]
    end
    @inbounds M[A, B] = ((P[A, B] - ϕref * Pv[A, B]) - fine * inv) * _face_area(nest, nothing, ax, B)
    return nothing
end

# =========================================================================
# Bookings of a separate transport (#327, M3)
# =========================================================================

# simulation => [(registers, :parent | :child, nest)] of the two-way runs it
# belongs to (weak keys: a discarded simulation drops out). Set by NestedRun.
# A domain of a nest hierarchy (#328) is the child of one link and the
# parent of others, hence a list; a domain re-registered on the same side
# of the same nest (a run rebuilt on the same simulations) replaces its entry.
const _NEST_BOOKERS = WeakKeyDict{Any,Vector{Tuple{NestFluxRegisters,Symbol,NestedGrid}}}()

function _register_booker!(sim, r::NestFluxRegisters, which::Symbol, nest::NestedGrid)
    v = get!(_NEST_BOOKERS, sim) do
        Tuple{NestFluxRegisters,Symbol,NestedGrid}[]
    end
    filter!(e -> !(e[2] === which && e[3] === nest), v)
    push!(v, (r, which, nest))
    return nothing
end

"""
    book_nest_flux!(sim, name, ϕ, dt; scheme = nothing, diffusivity = nothing,
                    velocity = (sim.u, sim.v, sim.w), weights = :sim, sgs = nothing) -> Bool

Book into the flux registers of the two-way [`NestedRun`](@ref) that `sim`
belongs to (parent or child) the interface flux that a **separate
transport** of the tracer `name` (field `ϕ`, declared
`TwoWayFeedback(tracer_fluxes = (name => NestScalarFlux(applied_by = :transport),))`)
applies over the time `dt` (#327, M3): a transport that moves its tracers
outside the core's stages — after the dynamics, with its own scheme,
velocity and time step, as the Berlin case's WENO slot does — calls it
with **the state it is about to apply the flux to**, before updating `ϕ`:

* `scheme` (a name of `boundary_scalar_flux` or a reconstruction; `nothing`
  for no advection) books the advective flux `u_n ϕ_f` with the
  transport's `velocity` (halos as the transport filled them), and the
  volume flux that carried it (volume-consistent refluxing);
* `diffusivity = (ν = …, κh = …, κv = …, scale = 1)` books the diffusive
  flux `−(ν + s·½(κ_L + κ_R)) ∂ϕ/∂n` (κh on the lateral, κv on the
  vertical planes: cell-centred fields, halos filled), the face form of
  `add_eddy_diffusion_scalar!`. A Lie-split transport that diffuses after
  advecting calls it a second time with the advected `ϕ`;
* `diffusivity = :theta` or `:tracer` books the diffusive flux **the
  core** gives the scalar inside its stages — molecular `ν` (when the
  simulation has viscosity) plus the domain closure's θ diffusivity
  (`:theta`) or tracer diffusivity (`:tracer`, as for `qv`), the closure
  being `sgs` or the simulation's own — exactly what the run's hook books
  for a core-transported scalar. A caller that drives a core itself and
  books its `θ` and `qv` explicitly (`applied_by = :transport`, the Berlin
  case, #267) calls it from the core's `body_force!` at every stage with
  `dt` times the stage weight, with `scheme = :centered`.

Both are formed by the one face kernel
[`scalar_face_flux`](@ref Aeolus.AdvectionSchemes.scalar_face_flux) and
weighted by the simulation's constraint weights `W` (`weights = :sim`;
`nothing` for none; a tuple `(Wc, Wf)` of storage-indexed vectors to
override) — the `∫ W ϕ` the composite conserves. A multi-stage transport
calls it once per stage with `dt` times the stage weight.

Bookings land in the step being attempted and are discarded with it if the
step is rejected. Returns `false` (and books nothing) when `sim` is in no
refluxing two-way run, `name` is not declared with `applied_by = :transport`,
or no step of `sim`'s domain is under way — so a transport can call it
unconditionally. A domain of a [`NestHierarchy`](@ref) that belongs to
several two-way links (the child of one, the parent of others) books into
every one of them.
"""
function book_nest_flux!(sim, name::Symbol, ϕ::Field, dt::Real; scheme = nothing,
                         diffusivity = nothing, velocity = (sim.u, sim.v, sim.w),
                         weights = :sim, sgs = nothing)
    entries = get(_NEST_BOOKERS, sim, nothing)
    entries === nothing && return false
    booked = false
    for (r, which, nest) in entries            # every link the domain belongs to (#328)
        booked |= _book_entry!(r, which, nest, sim, name, ϕ, dt; scheme, diffusivity,
                               velocity, weights, sgs)
    end
    return booked
end

function _book_entry!(r::NestFluxRegisters, which::Symbol, nest::NestedGrid, sim, name::Symbol,
                      ϕ::Field, dt::Real; scheme, diffusivity, velocity, weights, sgs)
    n = findfirst(==(name), r.names)
    (n === nothing || _by_core(r.models[n], which)) && return false
    par = which === :parent
    (par ? r.popen : r.copen) || return false
    W = weights === :sim ? (par ? r.pW : r.cW) : weights
    w = Float64(dt)
    planes = par ? r.ppend[n] : r.cpend[n]
    vplanes = par ? r.pvpend[n] : r.cvpend[n]
    u, v, wv = velocity
    if scheme !== nothing
        rec = _flux_reconstruction(scheme)
        for s in 1:6
            r.sides[s] || continue
            _book_plane!(planes[s], sim, ϕ, rec, nothing, 0.0, W, nest, s, par, w, u, v, wv)
            _book_plane!(vplanes[s], sim, nothing, nothing, nothing, 0.0, W, nest, s, par, w,
                         u, v, wv)
        end
    end
    if diffusivity isa Symbol
        diffusivity in (:theta, :tracer) ||
            throw(ArgumentError("book_nest_flux!: diffusivity = :$diffusivity; use :theta, " *
                                ":tracer or (ν = …, κh = …, κv = …, scale = …)"))
        closure = _active_sgs(sim, sgs, nothing)
        f = closure === nothing ? nothing : sgs_fields(closure, sim)
        K = _face_diffusivity(NestScalarFlux(; molecular = true, sgs = diffusivity), closure, f,
                              sim, ϕ, par ? r.pscratch : r.cscratch)
        ν = sim.has_viscosity ? Float64(sim.ν) : 0.0
        for s in 1:6
            r.sides[s] || continue
            _book_plane!(planes[s], sim, ϕ, nothing, K, ν, W, nest, s, par, w, u, v, wv)
        end
    elseif diffusivity !== nothing
        ν = Float64(get(diffusivity, :ν, 0.0))
        κh = get(diffusivity, :κh, nothing); κv = get(diffusivity, :κv, κh)
        (κh === nothing) == (κv === nothing) ||
            throw(ArgumentError("book_nest_flux!: give both κh and κv, or neither"))
        K = κh === nothing ? nothing : (κh, κv, Float64(get(diffusivity, :scale, 1.0)))
        for s in 1:6
            r.sides[s] || continue
            _book_plane!(planes[s], sim, ϕ, nothing, K, ν, W, nest, s, par, w, u, v, wv)
        end
    end
    return true
end

# =========================================================================
# Masks
# =========================================================================

# The open faces of the child whose mean normal velocity points inward.
function _inflow_sides(csim, open::NTuple{6,Bool})
    m = csim.mesh
    ii, jj, kk = interior(m)
    flows = (_mean_plane(csim.u.data, first(ii), jj, kk, 1), -_mean_plane(csim.u.data, last(ii) + 1, jj, kk, 1),
             _mean_plane(csim.v.data, first(jj), ii, kk, 2), -_mean_plane(csim.v.data, last(jj) + 1, ii, kk, 2),
             _mean_plane(csim.w.data, first(kk), ii, jj, 3), -_mean_plane(csim.w.data, last(kk) + 1, ii, jj, 3))
    return ntuple(s -> open[s] && flows[s] > 0, 6)
end

function _mean_plane(d, q, ra, rb, ax)
    s = 0.0
    @inbounds for b in rb, a in ra
        s += ax == 1 ? d[q, a, b] : ax == 2 ? d[a, q, b] : d[a, b, q]
    end
    return s / (length(ra) * length(rb))
end

"""
    feedback_mask(nest, fb::TwoWayFeedback, child_sim) -> BitArray{3}
    feedback_mask(run::NestedRun) -> BitArray{3}

The covered parent cells (`nx × ny × nz`, nest-local indices) that the
two-way feedback anterpolates: all covered cells except the `fb.buffer`
outermost cells next to each open (nest) face of the child, the cells
whose lower face lies below `fb.z_anterp`, and the `fb.exclude_width`
cells next to the excluded (inflow) faces. See [`TwoWayFeedback`](@ref).
"""
feedback_mask(nest::NestedGrid, fb::TwoWayFeedback, csim) =
    _feedback_mask!(BitArray(undef, nest.nx, nest.ny, nest.nz), nest, fb, csim)

function _feedback_mask!(act::BitArray{3}, nest::NestedGrid, fb::TwoWayFeedback, csim)
    open = nest_open_sides(csim)
    excl = fb.exclude_sides === :inflow ? _inflow_sides(csim, open) :
           ntuple(s -> open[s] && fb.exclude_sides[s], 6)
    nn = (nest.nx, nest.ny, nest.nz)
    fill!(act, true)
    for s in 1:6
        open[s] || continue
        ax = _side_axis(s); hi = _side_high(s)
        wd = excl[s] ? max(fb.buffer, fb.exclude_width) : fb.buffer
        for q in 1:min(wd, nn[ax])
            fill!(selectdim(act, ax, hi ? nn[ax] - q + 1 : q), false)
        end
    end
    if isfinite(fb.z_anterp)
        pm = nest.parent_mesh
        tol = 1e-9 * max(1.0, abs(fb.z_anterp))
        for K in 1:nest.nz
            _zface_storage(pm, pm.Hz + nest.iz + K - 1) >= fb.z_anterp - tol ||
                fill!(view(act, :, :, K), false)
        end
    end
    return act
end

_zface_storage(m::RegularGrid, P::Integer) = m.z0 + (P - m.Hz - 1) * m.dz
_zface_storage(m::StretchedGrid, P::Integer) = Float64(zfaces(m)[P])

# =========================================================================
# Anterpolation
# =========================================================================

# Whether the parent storage position `I` (1-based within the covered box of
# a field at location `ax`: 0 cells, 1..3 faces) is anterpolated: a cell of
# the mask, or a face strictly inside the covered region between two cells
# of the mask.
@inline function _active(act::BitArray{3}, ax::Int, I::NTuple{3,Int}, nn)
    ax == 0 && return @inbounds act[I[1], I[2], I[3]]
    q = I[ax] - 1                                    # face index 0..n along ax
    (1 <= q <= nn[ax] - 1) || return false
    a = Base.setindex(I, q, ax); b = Base.setindex(I, q + 1, ax)
    return @inbounds act[a[1], a[2], a[3]] && act[b[1], b[2], b[3]]
end

_loc_axis(::Field{CellCenter}) = 0
_loc_axis(::Field{FaceX}) = 1
_loc_axis(::Field{FaceY}) = 2
_loc_axis(::Field{FaceZ}) = 3

# The covered parent box of a field at staggered axis `ax` (0: cells).
function _covered_box(nest::NestedGrid, ax::Int)
    pm = nest.parent_mesh
    nn = (nest.nx, nest.ny, nest.nz)
    lo = (pm.Hx + nest.ix, pm.Hy + nest.iy, pm.Hz + nest.iz)
    return ntuple(d -> lo[d]:lo[d] + nn[d] - (d == ax ? 0 : 1), 3)
end

# Scratch of the anterpolation: the covered parent box of a cell-centred
# field and of each face field, to keep the cells outside the mask.
_anterp_scratch(nest::NestedGrid) =
    ntuple(a -> zeros(length.(_covered_box(nest, a - 1))...), 4)

function _anterpolate_field!(pf::Field, cf::Field, nest::NestedGrid, act::BitArray{3},
                             α::Float64, meas, χ, scratch)
    ax = _loc_axis(pf)
    nn = (nest.nx, nest.ny, nest.nz)
    rng = _covered_box(nest, ax)
    old = scratch[ax + 1]
    copyto!(old, view(pf.data, rng...))
    restrict_to_parent!(pf, cf, nest; measure = meas, solid = χ)
    o = (first(rng[1]) - 1, first(rng[2]) - 1, first(rng[3]) - 1)
    launch!(Backends.backend(pf.data), _anterp_body!, rng, pf.data, old, act, ax, nn, o, α)
    return pf
end

# Outside the mask: the value before the restriction; inside: relaxed by α.
@inline function _anterp_body!(i, j, k, pd, old, act, ax, nn, o, α)
    I = (i - o[1], j - o[2], k - o[3])
    @inbounds if !_active(act, ax, I, nn)
        pd[i, j, k] = old[I[1], I[2], I[3]]
    elseif α != 1.0
        pd[i, j, k] = old[I[1], I[2], I[3]] + α * (pd[i, j, k] - old[I[1], I[2], I[3]])
    end
    return nothing
end

"""
    anterpolate!(parent_sim, child_sim, nest, state; mask, α = 1, solid = nothing)

Anterpolation of the whole field set (`θ`, the scalars of `state`,
`u, v, w`) from the child onto the covered parent cells of `mask`
([`feedback_mask`](@ref)): the conservative child mean of
[`restrict_to_parent!`](@ref) in the state's measure, solid-aware with
`solid`, relaxed with `α = Δt_p/τ` (`φ_P ← φ_P + α(⟨φ_C⟩ − φ_P)`; `α = 1`
replaces exactly). Parent cells and faces outside the mask are left
bit for bit as they were. Velocity faces are anterpolated when strictly
inside the covered region and both their parent cells are in the mask;
the interface planes are the flux correction's.
"""
function anterpolate!(psim::_NestSim, csim::_NestSim, nest::NestedGrid, s::NestInterpState;
                      mask::BitArray{3}, α::Real = 1.0, solid = nothing,
                      scratch = _anterp_scratch(nest))
    size(mask) == (nest.nx, nest.ny, nest.nz) ||
        throw(DimensionMismatch("anterpolate!: mask has size $(size(mask)), expected " *
                                "$((nest.nx, nest.ny, nest.nz))"))
    0 < α <= 1 || throw(ArgumentError("anterpolate!: α = $α must be in (0, 1]"))
    _check_two_way_pair(psim, csim, "anterpolate!", s)
    a = Float64(α)
    meas = s.measure
    χ = _fluid_mask(nest, solid)
    _anterpolate_field!(psim.θ, csim.θ, nest, mask, a, meas, χ, scratch)
    for i in eachindex(s.names)
        _anterpolate_field!(s.parent[i], s.child[i], nest, mask, a, meas, χ, scratch)
    end
    _anterpolate_field!(psim.u, csim.u, nest, mask, a, meas, χ, scratch)
    _anterpolate_field!(psim.v, csim.v, nest, mask, a, meas, χ, scratch)
    _anterpolate_field!(psim.w, csim.w, nest, mask, a, meas, χ, scratch)
    return psim
end

# =========================================================================
# Re-projection
# =========================================================================

_reproject!(sim::BoussinesqSimulation) = _project_velocity!(sim)
_reproject!(sim::MassWeightedSimulation) = _project_mass_weighted!(sim)

# The interface flux correction of the open sides, in place: on every open
# interface plane the parent's face velocity takes the child's summed flux
# (the arithmetic of compute_nest_flux_mismatch + apply_flux_correction!,
# without the mismatch matrices, #329).
function _correct_interface_flux!(pf::Field, cf::Field, nest::NestedGrid, ax::Int, meas,
                                  sides::NTuple{6,Bool})
    nn = (nest.nx, nest.ny, nest.nz)
    for (q, s) in ((0, 2ax - 1), (nn[ax], 2ax))
        sides[s] || continue
        _plane_correct!(pf, cf, nest, ax, q, meas)
    end
    return pf
end

function _plane_correct!(pf::Field, cf::Field, nest::NestedGrid, ax::Int, q::Int, measure)
    nn = (nest.nx, nest.ny, nest.nz)
    t1, t2 = _tang(ax)
    ωz = ax == 3 ? nothing : _zweights(measure)
    g = !nest.aligned ? _plane_geom_ov(nest, ax, q, measure) :
        ωz === nothing ? _plane_geom(nest, ax, q, measure, nothing) :
                         _plane_geom(nest, ax, q, measure, ωz)
    launch!(Backends.backend(pf.data), _plane_correct_body!, (1:nn[t1], 1:nn[t2]), pf.data,
            cf.data, g)
    return pf
end

# One parent interface face: the mismatch kernel, then u_p -= δΦ / A_p.
@inline function _plane_correct_body!(A, B, pd, cd, g)
    M, pidx = _mismatch_at(pd, cd, g, A, B)
    @inbounds pd[pidx[1], pidx[2], pidx[3]] -= M / _face_area(g.nest, g.measure, g.ax, B)
    return nothing
end
