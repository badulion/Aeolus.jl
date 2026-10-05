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
# Parent state interpolated onto the child mesh
# =========================================================================

"""
    NestInterpState

The parent solution (`u, v, w, θ`) interpolated onto the child mesh —
boundary data for the child and the relaxation target for
[`apply_nest_sponge!`](@ref). Build with [`nest_interp_state`](@ref),
fill with [`interpolate_parent_state!`](@ref).
"""
struct NestInterpState{T, M<:RegularGrid{T}}
    u::Field{FaceX,      T, Array{T,3}, M}
    v::Field{FaceY,      T, Array{T,3}, M}
    w::Field{FaceZ,      T, Array{T,3}, M}
    θ::Field{CellCenter, T, Array{T,3}, M}
end

function nest_interp_state(nest::NestedGrid{T}) where {T}
    cm = nest.child_mesh
    return NestInterpState(Field(FaceX(), cm), Field(FaceY(), cm),
                           Field(FaceZ(), cm), Field(CellCenter(), cm))
end

"""
    interpolate_parent_state!(state, parent_sim, nest; order=3, region=:all)

Interpolate `parent_sim`'s `u, v, w, θ` onto the child mesh into `state`.
`region=:all` fills the whole child field (needed for the sponge);
`region=:halo` fills only halos. Parent halos are filled first.
"""
function interpolate_parent_state!(s::NestInterpState, psim::BoussinesqSimulation,
                                   nest::NestedGrid; order::Int=3,
                                   region::Symbol=:all)
    fill_halos!(psim.u, psim.bcs); fill_halos!(psim.v, psim.bcs)
    fill_halos!(psim.w, psim.bcs); fill_halos!(psim.θ, psim.θ_bcs)
    _interp_child!(s.u, psim.u, nest; order, region)
    _interp_child!(s.v, psim.v, nest; order, region)
    _interp_child!(s.w, psim.w, nest; order, region)
    _interp_child!(s.θ, psim.θ, nest; order, region)
    return s
end

function _blend_state!(dst::NestInterpState, a::NestInterpState,
                       b::NestInterpState, τ::Real)
    for f in (:u, :v, :w, :θ)
        getfield(dst, f).data .= (1 - τ) .* getfield(a, f).data .+ τ .* getfield(b, f).data
    end
    return dst
end

function _copy_halo!(dst::Field, src::Field; interface::Bool=false)
    m = dst.mesh
    ii, jj, kk = interior(m)
    loc = location(dst)
    dd = dst.data; sd = src.data
    @inbounds for k in axes(dd, 3), j in axes(dd, 2), i in axes(dd, 1)
        if (i in ii) && (j in jj) && (k in kk)
            (interface && _interface_plane(loc, m, i, j, k)) || continue
        end
        dd[i, j, k] = sd[i, j, k]
    end
    return dst
end

"""
    apply_nest_boundary!(child_sim, state; impose_interface=false)

Write the interpolated parent `state` into `child_sim`'s halo cells (and,
optionally, the boundary-normal interior face planes).
"""
function apply_nest_boundary!(csim::BoussinesqSimulation, s::NestInterpState;
                              impose_interface::Bool=false)
    _copy_halo!(csim.u, s.u; interface=impose_interface)
    _copy_halo!(csim.v, s.v; interface=impose_interface)
    _copy_halo!(csim.w, s.w; interface=impose_interface)
    _copy_halo!(csim.θ, s.θ)
    return csim
end

"""
    apply_nest_sponge!(child_sim, parent_interp, sponge; dt=1.0)

Relax the child solution toward `parent_interp` inside the sponge zone:
`f ← (1-β) f + β f_parent` with `β = clamp(α_max·dt·ramp, 0, 1)`.
Applied to `u, v, w, θ`. Call once per child substep, after `step!`.
"""
function apply_nest_sponge!(csim::BoussinesqSimulation, s::NestInterpState,
                            sp::NestSponge; dt::Real=1.0)
    cm = csim.mesh
    ii, jj, kk = interior(cm)
    αdt = sp.α_max * dt
    for (cf, gf) in ((csim.u, s.u), (csim.v, s.v), (csim.w, s.w), (csim.θ, s.θ))
        cd = cf.data; gd = gf.data
        @inbounds for k in kk, j in jj, i in ii
            wgt = _sponge_weight(sp, cm, i, j, k)
            wgt == 0.0 && continue
            β = clamp(αdt * wgt, 0.0, 1.0)
            cd[i, j, k] = (1 - β) * cd[i, j, k] + β * gd[i, j, k]
        end
    end
    return csim
end
