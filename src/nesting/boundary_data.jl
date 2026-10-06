# Nesting: the parent as the boundary data of the child's open faces
# (N-nest-d, #261). The child's nest faces are N-open-b `OpenBoundary`
# faces; `NestBoundaryData` is the Aeolus-parent implementer of their
# `AbstractBoundaryData` contract (#58 and #57 implement it for mesoscale and
# reanalysis parents). It owns no physics: prolongation is the N-nest-a
# operator set (`interpolate_parent_state!`), the time blend is N-nest-c's
# linear interpolation over the parent step, and the mass-flux correction
# is N-open-b's single `correct_boundary_mass_flux!`.

"""
    NestBoundaryData <: AbstractBoundaryData

The parent of a nest as the boundary data of the child's
[`OpenBoundary`](@ref Aeolus.BoundaryConditions.OpenBoundary) faces
(built by [`nest_bcs`](@ref)). It holds two
[`NestInterpState`](@ref)s — the parent field set prolongated onto the
child mesh at the two ends of the current parent step, `a = B(t₀)` and
`b = B(t₀ + Δt)` — and

    boundary_values!(planes, data, t)

writes `(1 − τ) a + τ b`, `τ = (t − t₀)/Δt`, into every open-boundary
buffer, at the buffer's own storage points: the normal-velocity face
planes (the boundary face and the `H` layers beyond it) and the
tangential-velocity and scalar halos, for every member of the field set
(`u, v, w, θ`, `qv`, extra scalars). `Δt = 0` is steady data (`a`).
The time blend is the one of [`boundary_time`](@ref): exact for data
linear in time, second order otherwise (N-nest-c, #257).

Faces inside solids are not this type's business (#327, M4): a
[`NestedRun`](@ref) built with `solid = mask` (and
[`apply_nest_boundary!`](@ref)`(…; solid)`) marks them with the one
open-boundary path,
[`mark_open_boundary_solids!`](@ref Aeolus.BoundaryConditions.mark_open_boundary_solids!)
(N-open-d #262): every velocity buffer — normal **and** tangential — is
zero at the solid points, and the mass-flux correction neither counts nor
adjusts them.

The prolongated parent states are filled by a [`NestedRun`](@ref) (or,
steady, by [`apply_nest_boundary!`](@ref)); see
[`nest_boundary_data`](@ref).

References: Hellsten, A. et al. (2021), A nested multi-scale system
implemented in the large-eddy simulation model PALM model system 6.0,
*Geosci. Model Dev.* 14, 3185–3214, §3.2–3.3 (Dirichlet child boundary from
the interpolated parent, mass-flux correction); Kadasch, E. et al. (2021),
*Geosci. Model Dev.* 14, 5435–5465, §2.2–2.3 (the same interface for a
mesoscale parent).
"""
mutable struct NestBoundaryData{S<:NestInterpState} <: AbstractBoundaryData
    a::S
    b::S
    t0::Float64
    Δt::Float64
end

_solid_array(::Nothing) = nothing
_solid_array(x::AbstractArray{Bool,3}) = x
_solid_array(x) = hasproperty(x, :mask) ? _solid_array(getproperty(x, :mask)) :
    throw(ArgumentError("nest: `solid` must be a SolidMask or a Bool array " *
                        "on the child's storage, got $(typeof(x))"))

"""
    nest_boundary_data(a, b = a; t0 = 0, Δt = 0) -> NestBoundaryData

A [`NestBoundaryData`](@ref) blending the prolongated parent states `a`
(at `t0`) and `b` (at `t0 + Δt`); with `b = a` (or `Δt = 0`) it is
steady. Solid faces are marked on the child's boundary conditions, not
here (`NestedRun(…; solid)`, [`apply_nest_boundary!`](@ref)).
"""
function nest_boundary_data(a::NestInterpState, b::NestInterpState = a;
                            t0::Real = 0.0, Δt::Real = 0.0)
    return NestBoundaryData(a, b, Float64(t0), Float64(Δt))
end

# The child's solid mask on its open faces: the single N-open-d path
# (`mark_open_boundary_solids!`: zero normal and tangential velocity at the
# solid points, excluded from the mass-flux correction). #327 M4 retired the
# nest's own `_solid_faces!`, which zeroed only the normal velocity.
function _mark_nest_solids!(csim, solid)
    sa = _solid_array(solid)
    sa === nothing && return csim
    size(sa) == size(csim.θ.data) ||
        throw(DimensionMismatch("nest: solid mask of size $(size(sa)) is not on the " *
                                "child's storage $(size(csim.θ.data))"))
    has_open_boundaries(csim) && mark_open_boundary_solids!(csim.bcs, sa)
    return csim
end

function _nest_member(s::NestInterpState, name::Symbol)
    try
        return s[name]
    catch err
        err isa KeyError || rethrow()
        throw(ArgumentError("NestBoundaryData: the child has an open-boundary plane for " *
                            ":$name but the nest's field set is $(nest_scalar_names(s)) " *
                            "plus u, v, w"))
    end
end

function boundary_values!(planes::AbstractVector{<:OpenPlane}, d::NestBoundaryData, t)
    τ = d.Δt > 0 ? (Float64(t) - d.t0) / d.Δt : 0.0
    for p in planes
        p.mode === :mirror && continue
        _blend_plane!(p, _nest_member(d.a, p.name).data, _nest_member(d.b, p.name).data, τ)
    end
    return planes
end

# p.data ← (1-τ)·A + τ·B at the plane's storage points (the blend of
# `_blend_state!` / `boundary_time`, written out per point).
function _blend_plane!(p::OpenPlane{T}, A::AbstractArray, B::AbstractArray, τ) where {T}
    τ1 = one(T) - T(τ); τT = T(τ)
    pd = p.data
    @inbounds for (c, k) in enumerate(p.krange), (b, j) in enumerate(p.jrange),
                  (a, i) in enumerate(p.irange)
        pd[a, b, c] = τ1 * A[i, j, k] + τT * B[i, j, k]
    end
    return p
end

"""
    nest_open_sides(sim_or_bcs) -> NTuple{6,Bool}

Which faces `(west, east, south, north, bottom, top)` of a child are open
(nest) faces. A [`NestSponge`](@ref) in a [`NestedRun`](@ref) acts only on
these: a sponge against a wall or a periodic face would relax the child
toward the parent where there is no nest boundary.
"""
nest_open_sides(b::FieldBoundaryConditions) =
    (b.west isa OpenBoundary, b.east isa OpenBoundary, b.south isa OpenBoundary,
     b.north isa OpenBoundary, b.bottom isa OpenBoundary, b.top isa OpenBoundary)
nest_open_sides(sim) = nest_open_sides(sim.bcs)

# Attach `data` to every open boundary set of the child and bind the
# field set's extra scalars (tracers that are not simulation properties)
# to their planes, so `fill_halos!(tracer, θ_bcs)` finds them. The planes
# and the field set must name the same fields: a set member without a plane
# would get a zero-gradient halo (its inflow lost), a plane without a
# member has no data.
function _attach_nest_data!(csim, data::NestBoundaryData)
    has_open_boundaries(csim) || return csim
    s = data.a
    set = Set{Symbol}([:u, :v, :w, :θ]); foreach(n -> push!(set, n), s.names)
    for st in open_boundary_states(csim)
        have = Set{Symbol}(p.name for p in st.planes)
        have == set || throw(ArgumentError(
            "the child's open faces carry $(sort!(collect(have))) but the nest's field set " *
            "is $(sort!(collect(set))); build the child's bcs with " *
            "nest_bcs(mesh, …; moisture, scalars) matching the run's field set"))
        st.data = data
        st.initialized = false
        for i in eachindex(s.names)
            st.bound[s.child[i].data] = s.names[i]
        end
    end
    return csim
end

# Refresh the child's open-boundary buffers at the child's time and write
# them into the halos of every member of the field set.
function _fill_nest_halos!(csim, s::NestInterpState)
    has_open_boundaries(csim) || return csim
    update_open_boundaries!(csim, csim.t, zero(csim.t))
    fill_halos!(csim.u, csim.bcs); fill_halos!(csim.v, csim.bcs); fill_halos!(csim.w, csim.bcs)
    for (_, _, cf, _) in _scalar_entries(s, nothing, csim)
        fill_halos!(cf, csim.θ_bcs)
    end
    return csim
end

"""
    apply_nest_boundary!(child_sim, state; solid = nothing) -> child_sim

Make the interpolated parent `state` (a [`NestInterpState`](@ref)) the
**steady** boundary data of the child's open (nest) faces
([`nest_boundary_data`](@ref)), refresh the open-boundary buffers with
the mass-flux correction, and write them into the child: the boundary
faces of the normal velocity, the tangential-velocity halos and the
halos of every scalar of the set (`θ`, `qv`, tracers). Faces that are
not nest faces (periodic, walls) are filled by their own condition.
`solid` (a `SolidMask` or a `Bool` array) marks the open faces inside
solids ([`mark_open_boundary_solids!`](@ref
Aeolus.BoundaryConditions.mark_open_boundary_solids!)).

A [`NestedRun`](@ref) does this itself, with the time-blended parent;
calling this on a run's child replaces the run's boundary data.
"""
function apply_nest_boundary!(csim::_NestSim, s::NestInterpState; solid = nothing)
    has_open_boundaries(csim) || return csim
    _mark_nest_solids!(csim, solid)
    _attach_nest_data!(csim, nest_boundary_data(s; t0 = csim.t))
    return _fill_nest_halos!(csim, s)
end
