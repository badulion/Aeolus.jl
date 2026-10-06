# Nesting: NestSponge and the parent state on the child mesh (moved verbatim
# from AdaptiveMesh.jl, N-nest-a #250).

# =========================================================================
# Nest sponge (reflection control)
# =========================================================================

"""
    NestSponge(; width=4, α_max=1.0, profile=:cos2, sides=ntuple(_->true, 6))

Relaxation zone `width` child cells deep inside the child, adjacent to
its nest boundaries. [`apply_nest_sponge!`](@ref) blends the child
solution toward the interpolated parent solution with rate `α_max`
(1/time), ramped by `profile` (`:cos2` — smooth `sin²` taper — or
`:linear`) from 0 at the inner edge of the zone to `α_max` at the
boundary. `sides` is `(west, east, south, north, bottom, top)`.

The smooth taper is the point: a hard relaxation edge is itself a
reflector. Without a sponge, the halo/interior mismatch at a nest
boundary reflects outgoing waves back into the child.

In a [`NestedRun`](@ref) the velocity relaxation acts on the predicted
velocity before the pressure solve, so the child stays divergence-free
(#329); see [`apply_nest_sponge!`](@ref).
"""
struct NestSponge{T}
    width::Int
    α_max::T
    profile::Symbol
    sides::NTuple{6,Bool}
end

function NestSponge(; width::Int=4, α_max::Real=1.0, profile::Symbol=:cos2,
                    sides::NTuple{6,Bool}=(true, true, true, true, true, true))
    width >= 1 || error("NestSponge: width must be >= 1")
    profile in (:cos2, :linear) || error("NestSponge: profile must be :cos2 or :linear")
    return NestSponge{Float64}(width, Float64(α_max), profile, sides)
end

# The :cos2 weight is the shared sin² damping ramp (Operators.FiniteVolume),
# evaluated on s ∈ (0, 1]; bit-identical to the former sin(0.5π s)^2.
const _NEST_SIN2 = DampingProfile(SinSquaredRamp(), 1.0, 0.0, 1.0, Inf, 0)

_sponge_prof(sp::NestSponge, s::Float64) =
    sp.profile === :cos2 ? damping_rate(_NEST_SIN2, s) : s

function _sponge_ramp(sp::NestSponge, d::Int)
    d < 1 && return 1.0                 # halo side of the boundary
    d > sp.width && return 0.0
    return _sponge_prof(sp, (sp.width + 1 - d) / sp.width)
end

function _sponge_weight(sp::NestSponge, cm::RegularGrid, i::Int, j::Int, k::Int)
    n1 = i - cm.Hx; n2 = j - cm.Hy; n3 = k - cm.Hz
    w = 0.0
    sp.sides[1] && (w = max(w, _sponge_ramp(sp, n1)))
    sp.sides[2] && (w = max(w, _sponge_ramp(sp, cm.Nx + 1 - n1)))
    sp.sides[3] && (w = max(w, _sponge_ramp(sp, n2)))
    sp.sides[4] && (w = max(w, _sponge_ramp(sp, cm.Ny + 1 - n2)))
    sp.sides[5] && (w = max(w, _sponge_ramp(sp, n3)))
    sp.sides[6] && (w = max(w, _sponge_ramp(sp, cm.Nz + 1 - n3)))
    return w
end

# =========================================================================
# Parent state interpolated onto the child mesh: a generic field set
# =========================================================================

const _CCField{T,M} = Field{CellCenter, T, Array{T,3}, M}

"""
    NestInterpState

The parent solution interpolated onto the child mesh — boundary data for
the child and the relaxation target for [`apply_nest_sponge!`](@ref).
It is a state over a *named field set*:

  * the velocities `u, v, w` (fields `s.u`, `s.v`, `s.w`);
  * `θ` (field `s.θ`);
  * `qv`, when built from moist simulations with
    `nest_interp_state(nest, parent_sim, child_sim)`;
  * any extra cell-centred scalars, passed as
    `scalars = (:co2 => (child_field, parent_field), …)`.

Every cell-centred scalar of the set other than `θ` is bound to its
child and parent fields; read the interpolated buffer of any member
with `s[name]` (e.g. `s[:qv]`, `s[:co2]`, `s[:u]`) and list the scalar
names with [`nest_scalar_names`](@ref). Prolongation
([`interpolate_parent_state!`](@ref)), the halo write
([`apply_nest_boundary!`](@ref)), the sponge, restriction
([`restrict_state_to_parent!`](@ref)), [`composite_integral`](@ref) and
refluxing ([`reflux_scalars!`](@ref)) all iterate over the whole set,
through the same operator per member — so a tracer initialised equal to
`θ` stays bit-identical to `θ` through all of them.

`s.measure` is the vertical [`NestMeasure`](@ref) of the set's
operators: `nothing` (volume) for Boussinesq sims, the reference-mass
measure for a [`MassWeightedSimulation`](@ref) pair, so prolongation,
restriction, refluxing and the composite integrals conserve `∫ W φ dV`.
"""
struct NestInterpState{T, M<:RegularGrid{T}}
    u::Field{FaceX,      T, Array{T,3}, M}
    v::Field{FaceY,      T, Array{T,3}, M}
    w::Field{FaceZ,      T, Array{T,3}, M}
    θ::Field{CellCenter, T, Array{T,3}, M}
    names::Vector{Symbol}               # extra scalars (qv first, when moist)
    bufs::Vector{_CCField{T,M}}         # parent value interpolated to the child mesh
    child::Vector{Field}                # bound child field per extra scalar
    parent::Vector{Field}               # bound parent field per extra scalar
    measure::Union{Nothing,NestMeasure} # vertical measure of the operators
end

const _RESERVED_NEST_NAMES = (:u, :v, :w, :θ, :theta)

function _check_scalar_pair(nest::NestedGrid, name::Symbol, cf, pf)
    name in _RESERVED_NEST_NAMES &&
        error("nest_interp_state: scalar name :$name is reserved (u, v, w and θ are " *
              "always part of the set)")
    (cf isa Field && pf isa Field) ||
        error("nest_interp_state: scalar :$name must be given as " *
              ":$name => (child_field, parent_field)")
    (location(cf) isa CellCenter && location(pf) isa CellCenter) ||
        error("nest_interp_state: scalar :$name must be cell-centred on both meshes")
    size(cf.data) == totalsize(nest.child_mesh) ||
        error("nest_interp_state: the child field of :$name is not on the nest's child mesh")
    size(pf.data) == totalsize(nest.parent_mesh) ||
        error("nest_interp_state: the parent field of :$name is not on the nest's parent mesh")
    return nothing
end

"""
    nest_interp_state(nest; scalars=(), measure=nothing)
    nest_interp_state(nest, parent_sim, child_sim; scalars=())

Allocate a [`NestInterpState`](@ref) on `nest`'s child mesh. The field
set always holds `u, v, w, θ`. The second form adds `qv` (bound to
`child_sim.qv` / `parent_sim.qv`) when both Boussinesq simulations carry
moisture, and errors when only one does. `scalars` adds extra
cell-centred scalars, `name => (child_field, parent_field)`, in the
order given.

For two [`MassWeightedSimulation`](@ref)s the state carries the mass
measure of their shared `ReferenceState` ([`nest_measure`](@ref), which
asserts that the profile is shared). The sound-proof core has no `qv`
field, so a moist anelastic nest carries `qv` as a tracer:
`scalars = (:qv => (child_qv, parent_qv),)`.
"""
function nest_interp_state(nest::NestedGrid{T}; scalars=(), measure=nothing) where {T}
    cm = nest.child_mesh
    names = Symbol[]; bufs = _CCField{T,typeof(cm)}[]
    child = Field[]; parent = Field[]
    for pr in scalars
        name, fs = first(pr), last(pr)
        name isa Symbol || error("nest_interp_state: scalar names must be Symbols")
        length(fs) == 2 || error("nest_interp_state: scalar :$name must be given as " *
                                 ":$name => (child_field, parent_field)")
        cf, pf = fs
        _check_scalar_pair(nest, name, cf, pf)
        name in names && error("nest_interp_state: scalar :$name given twice")
        push!(names, name); push!(bufs, Field(CellCenter(), cm))
        push!(child, cf); push!(parent, pf)
    end
    return NestInterpState(Field(FaceX(), cm), Field(FaceY(), cm),
                           Field(FaceZ(), cm), Field(CellCenter(), cm),
                           names, bufs, child, parent, measure)
end

function nest_interp_state(nest::NestedGrid, psim::MassWeightedSimulation,
                           csim::MassWeightedSimulation; scalars=())
    return nest_interp_state(nest; scalars, measure=nest_measure(nest, psim, csim))
end

# The measure of a pair of simulations: volume for Boussinesq, the
# (cached, profile-checked) mass measure for sound-proof sims.
_sim_measure(nest, ::BoussinesqSimulation, ::BoussinesqSimulation) = nothing
_sim_measure(nest, p::MassWeightedSimulation, c::MassWeightedSimulation) =
    nest_measure(nest, p, c)
_sim_measure(nest, p, c) = nothing          # a mixed pair: the volume measure

# Same dynamical-core family (the pairs the restriction / feedback and the
# mass measure are defined for).
_same_core(::BoussinesqSimulation, ::BoussinesqSimulation) = true
_same_core(::MassWeightedSimulation, ::MassWeightedSimulation) = true
_same_core(p, c) = false

# The guard of every child → parent transfer (#327, M1). A mixed pair (a
# Boussinesq parent with a sound-proof child, or the reverse) has no
# consistent measure: the restriction would average with the volume where
# the sound-proof grid needs its reference mass, and the flux registers
# would mix a volume flux with a mass flux. A sound-proof pair must carry
# the reference-mass measure (a state built with `nest_interp_state(nest)`
# has the volume measure). One-way nesting of a mixed pair is fine: the
# parent only provides boundary data.
function _check_two_way_pair(p, c, what::AbstractString, s = nothing)
    _same_core(p, c) ||
        throw(ArgumentError("$what: two-way feedback needs parent and child of the same " *
                            "core family (two BoussinesqSimulations, or two " *
                            "MassWeightedSimulations sharing one reference profile); got " *
                            "$(nameof(typeof(p))) and $(nameof(typeof(c))). A mixed pair " *
                            "can only be nested one-way (feedback = :one_way)"))
    s !== nothing && p isa MassWeightedSimulation && s.measure === nothing &&
        throw(ArgumentError("$what: a sound-proof pair needs the reference-mass measure; " *
                            "build the state with nest_interp_state(nest, parent, child)"))
    return nothing
end

# A Boussinesq pair, or a mixed pair (e.g. a Boussinesq parent providing
# one-way boundary data to a sound-proof child, see NestedRun): volume
# measure. A sound-proof pair is the more specific method above.
function nest_interp_state(nest::NestedGrid, psim::_NestSim,
                           csim::_NestSim; scalars=())
    _moist(psim) == _moist(csim) ||
        error("nest_interp_state: parent has_moisture = $(_moist(psim)) but child " *
              "has_moisture = $(_moist(csim)); a nest must carry qv on both grids or " *
              "on neither")
    moist = _moist(psim) ? (:qv => (csim.qv, psim.qv),) : ()
    return nest_interp_state(nest; scalars=(moist..., scalars...))
end


"""
    nest_scalar_names(state) -> Vector{Symbol}

Names of the cell-centred scalars in the [`NestInterpState`](@ref)
field set: `:θ` first, then `:qv` (if present) and the extra scalars.
"""
nest_scalar_names(s::NestInterpState) = Symbol[:θ; s.names]

function Base.getindex(s::NestInterpState, name::Symbol)
    name === :u && return s.u
    name === :v && return s.v
    name === :w && return s.w
    (name === :θ || name === :theta) && return s.θ
    i = findfirst(==(name), s.names)
    i === nothing && throw(KeyError(name))
    return s.bufs[i]
end

# (name, interpolated buffer, child field, parent field) for every
# cell-centred scalar of the set; θ is bound to the simulations.
function _scalar_entries(s::NestInterpState, psim, csim)
    θc = csim === nothing ? nothing : csim.θ
    θp = psim === nothing ? nothing : psim.θ
    out = Any[(:θ, s.θ, θc, θp)]
    for i in eachindex(s.names)
        push!(out, (s.names[i], s.bufs[i], s.child[i], s.parent[i]))
    end
    return out
end

"""
    interpolate_parent_state!(state, parent_sim, nest; order=3, region=:all)

Interpolate every member of `state`'s field set from the parent onto the
child mesh: `parent_sim`'s `u, v, w, θ`, plus each bound scalar's parent
field (`qv`, tracers). `region=:all` fills the whole child field (needed
for the sponge); `region=:halo` fills only halos. Parent halos are filled
first (scalars with `parent_sim.θ_bcs`, as `step!` does for `qv`). The
operators use the state's vertical measure (`s.measure`): the parent's
actual face heights on a stretched parent, the reference mass for a
sound-proof pair.
"""
function interpolate_parent_state!(s::NestInterpState, psim::_NestSim,
                                   nest::NestedGrid; order::Int=3,
                                   region::Symbol=:all)
    fill_halos!(psim.u, psim.bcs); fill_halos!(psim.v, psim.bcs)
    fill_halos!(psim.w, psim.bcs)
    meas = s.measure
    _interp_child!(s.u, psim.u, nest; order, region, measure=meas)
    _interp_child!(s.v, psim.v, nest; order, region, measure=meas)
    _interp_child!(s.w, psim.w, nest; order, region, measure=meas)
    for (_, buf, _, pf) in _scalar_entries(s, psim, nothing)
        fill_halos!(pf, psim.θ_bcs)
        _interp_child!(buf, pf, nest; order, region, measure=meas)
    end
    return s
end

# The whole field set blended in time, member by member, with the one
# two-snapshot kernel of the open boundaries (blend_window!, #329 item 8).
function _blend_state!(dst::NestInterpState, a::NestInterpState,
                       b::NestInterpState, τ::Real)
    blend_window!(dst.u.data, a.u.data, b.u.data, τ)
    blend_window!(dst.v.data, a.v.data, b.v.data, τ)
    blend_window!(dst.w.data, a.w.data, b.w.data, τ)
    blend_window!(dst.θ.data, a.θ.data, b.θ.data, τ)
    for i in eachindex(dst.bufs)
        blend_window!(dst.bufs[i].data, a.bufs[i].data, b.bufs[i].data, τ)
    end
    return dst
end

function _sponge_field!(cf::Field, gf::Field, sp::NestSponge, cm, αdt)
    ii, jj, kk = interior(cm)
    cd = cf.data; gd = gf.data
    @inbounds for k in kk, j in jj, i in ii
        wgt = _sponge_weight(sp, cm, i, j, k)
        wgt == 0.0 && continue
        β = clamp(αdt * wgt, 0.0, 1.0)
        cd[i, j, k] = (1 - β) * cd[i, j, k] + β * gd[i, j, k]
    end
    return cf
end

"""
    apply_nest_sponge!(child_sim, parent_interp, sponge; dt=1.0, velocity=true, scalars=true)

Relax the child solution toward `parent_interp` inside the sponge zone:
`f ← (1-β) f + β f_parent` with `β = clamp(α_max·dt·ramp, 0, 1)`.
Applied to every member of the field set: the velocities `u, v, w`
(`velocity = true`) and the scalars `θ`, `qv`, tracers (`scalars = true`).

The relaxation of the velocity is a source the projection must see: a
velocity relaxed *after* a step is no longer divergence-free (the
prolongated parent velocity is flux-consistent on parent cells, not on
child cells). A [`NestedRun`](@ref) therefore relaxes the velocities on
the predicted velocity, inside every stage's `post_predictor!` hook and
before the pressure solve, and only the scalars after the step (#329).
Call it with `velocity = true` after a step only where `∇·u = 0` does
not matter, or from a `post_predictor!` hook with `scalars = false`.
"""
function apply_nest_sponge!(csim::_NestSim, s::NestInterpState,
                            sp::NestSponge; dt::Real=1.0, velocity::Bool=true,
                            scalars::Bool=true)
    cm = csim.mesh
    αdt = sp.α_max * dt
    if velocity
        _sponge_field!(csim.u, s.u, sp, cm, αdt)
        _sponge_field!(csim.v, s.v, sp, cm, αdt)
        _sponge_field!(csim.w, s.w, sp, cm, αdt)
    end
    if scalars
        for (_, buf, cf, _) in _scalar_entries(s, nothing, csim)
            _sponge_field!(cf, buf, sp, cm, αdt)
        end
    end
    return csim
end
