# =========================================================================
# Pointwise terms of the fused sweep (#533 plan step 2). Included by
# FiniteVolume after `fused_sweep.jl`.
# =========================================================================
#
# The local forms of the core's pointwise operators: buoyancy, f-plane
# Coriolis, a prescribed flux through the first and last level and the
# relaxation `−λ(ξ)(φ − φ_target)` of the sponge and the fringe. Each
# evaluates the expression of its whole-field kernel (`_b_buoyancy!`,
# `_b_buoyancy_moist!`, `_b_coriolis!`, `_b_surface_flux!`, `_b_damping!`)
# in the same order, so a sweep with them agrees with the separate passes
# within round-off. The whole-field kernels stay: they are the reference
# and the form every other caller uses.

"""
    BuoyancyTerm(θ₀, coef)

The Boussinesq buoyancy `g(θ − θ₀)/θ₀` on the w tendency as a term of the
fused sweep, with `θ` brought to the w face by the grid's geometric
interpolation (`_add_buoyancy!`). When the sweep carries `qv` (a moist
core) it is the moist form `g(θᵥ − θ₀)/θ₀`, `θᵥ = θ(1 + 0.61 qv)`
(`_add_buoyancy_moist!`), as the core's step adds it. `coef = g_accel/θ₀`,
as the whole-field kernels take it. Footprint `(0, 0, 1)`.
"""
struct BuoyancyTerm{T} <: SweepTerm
    θ₀::T
    coef::T
end
footprint(::BuoyancyTerm) = (0, 0, 1)
@inline add_tendency!(t::BuoyancyTerm, tb, i, j, k, st) = _buoy_point!(t, tb, i, j, k, st.θ, st.qv, st.m)
@inline function _buoy_point!(t, tb, i, j, k, θ, ::Nothing, m)
    @inbounds θface = _zface(m, k, θ[i, j, k-1], θ[i, j, k])
    tb[i, SWEEP_W] += t.coef * (θface - t.θ₀)
    return nothing
end
@inline function _buoy_point!(t, tb, i, j, k, θ, qv::AbstractArray, m)
    @inbounds begin
        θface  = _zface(m, k, θ[i, j, k-1], θ[i, j, k])
        qvface = _zface(m, k, qv[i, j, k-1], qv[i, j, k])
    end
    θv = θface * (1 + 0.61 * qvface)
    tb[i, SWEEP_W] += t.coef * (θv - t.θ₀)
    return nothing
end

"""
    CoriolisTerm(f)

The f-plane Coriolis term `(+f v, −f u)` on the u and v tendencies as a
term of the fused sweep, with the four-point averages of `_add_coriolis!`.
Footprint `(1, 1, 0)`.
"""
struct CoriolisTerm{T} <: SweepTerm
    f::T
end
footprint(::CoriolisTerm) = (1, 1, 0)
@inline function add_tendency!(t::CoriolisTerm, tb, i, j, k, st)
    u, v = st.u, st.v
    @inbounds begin
        v_at_u = 0.25 * (v[i-1, j, k] + v[i, j, k] + v[i-1, j+1, k] + v[i, j+1, k])
        u_at_v = 0.25 * (u[i, j-1, k] + u[i+1, j-1, k] + u[i, j, k] + u[i+1, j, k])
    end
    tb[i, SWEEP_U] += t.f * v_at_u
    tb[i, SWEEP_V] -= t.f * u_at_v
    return nothing
end

"""
    SurfaceFluxTerm(F_bot, F_top, g)

Prescribed kinematic fluxes of θ through the bottom and the top as a term
of the fused sweep: `+F_bot/Δz_c` on the first interior level, `−F_top/Δz_c`
on the last (`_add_surface_flux!`, the local cell thickness). Only the
rows of those two levels do work. Footprint `(0, 0, 0)`.
"""
struct SurfaceFluxTerm{T} <: SweepTerm
    fb::T
    ft::T
    kb::Int
    kt::Int
end
function SurfaceFluxTerm(F_bot::Real, F_top::Real, g::AbstractMesh)
    m = grid_metrics(g)
    kk = interior(g)[3]
    fb, ft = F_bot * _idz(m, first(kk)), F_top * _idz(m, last(kk))
    T = promote_type(typeof(fb), typeof(ft))
    return SurfaceFluxTerm{T}(fb, ft, first(kk), last(kk))
end
footprint(::SurfaceFluxTerm) = (0, 0, 0)
@inline function add_row!(t::SurfaceFluxTerm, buf, a, b, s, j, k, ii, st)
    k == t.kb && _row_add!(buf, a, b, SWEEP_Θ, s, ii, t.fb)
    k == t.kt && _row_add!(buf, a, b, SWEEP_Θ, s, ii, -t.ft)
    return nothing
end
# a constant added along a row (−x is exact, so `+= −ft` is `−= ft`)
@noinline function _row_add!(buf, a, b, f, s, ii, x)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, f, s] += x
    end
    return nothing
end

"""
    DampingTerm(profile, axis, coords, targets, g)

The relaxation `−λ(ξ)(φ − φ_target)` of `u, v, w` as a term of the fused
sweep: the profile `profile` ([`DampingProfile`](@ref)) sampled at the
coordinate `ξ` of each point along `axis` (`3`: z, the Rayleigh sponge;
`1`: x, the fringe region), with `coords = (cu, cv, cw)` the coordinates
of the interior indices of each component along that axis and
`targets = (tu, tv, tw)` its targets (each a number, a level vector, a
`Field` or a halo-padded array, as `_add_damping!`). The arithmetic is that
of `_add_damping!`, the one damping kernel; along z the rate is taken
once per row. `targets` may also be a NamedTuple `(u, v, w)` or, along z,
a [`DampingTargets`](@ref).
Footprint `(0, 0, 0)`.
"""
struct DampingTerm{A, P<:DampingProfile, C<:Tuple, G} <: SweepTerm
    p::P
    coords::C
    targets::G
    i0::Int
    j0::Int
    k0::Int
end
function DampingTerm(p::DampingProfile, axis::Int, coords::Tuple, targets, g::AbstractMesh)
    axis in (1, 3) || throw(ArgumentError("DampingTerm: axis 1 (x) or 3 (z), got $axis"))
    ii, jj, kk = interior(g)
    return DampingTerm{axis, typeof(p), typeof(coords), typeof(targets)}(p, coords, targets, first(ii) - 1,
                                                                          first(jj) - 1, first(kk) - 1)
end
footprint(::DampingTerm) = (0, 0, 0)

"""
    DampingTargets(tu, tv, tw)

Targets of a [`DampingTerm`](@ref) whose kinds are known only at run time
(the core's sponge target, a NamedTuple of abstract fields): each a
number, a level vector or a 3-D array of eltype `T`, kept in fields of
concrete type (`kind` 0, 1, 2 per component; the unused slots hold empty
arrays), so the sweep branches on the kind once per row and every call
stays concretely typed. Build it once and keep it: it allocates.
"""
struct DampingTargets{T}
    kind::NTuple{3,Int}
    c::NTuple{3,T}
    v::NTuple{3,Vector{T}}
    a::NTuple{3,Array{T,3}}
end
function DampingTargets{T}(tu, tv, tw) where {T}
    kind = (_tkind(tu)::Int, _tkind(tv)::Int, _tkind(tw)::Int)
    c = (_tconst(T, tu)::T, _tconst(T, tv)::T, _tconst(T, tw)::T)
    v = (_tlevels(T, tu)::Vector{T}, _tlevels(T, tv)::Vector{T}, _tlevels(T, tw)::Vector{T})
    a = (_tarray(T, tu)::Array{T,3}, _tarray(T, tv)::Array{T,3}, _tarray(T, tw)::Array{T,3})
    return DampingTargets{T}(kind, c, v, a)
end
# one method per kind (the arguments may be of abstract type: no closures, no unions)
_tkind(::Real) = 0
_tkind(::AbstractVector) = 1
_tkind(::Any) = 2
_tconst(::Type{T}, x::Real) where {T} = T(x)
_tconst(::Type{T}, ::Any) where {T} = zero(T)
_tlevels(::Type{T}, x::Vector{T}) where {T} = x
_tlevels(::Type{T}, x::AbstractVector) where {T} = Vector{T}(x)
_tlevels(::Type{T}, ::Any) where {T} = T[]
_tarray(::Type{T}, ::Union{Real,AbstractVector}) where {T} = Array{T,3}(undef, 0, 0, 0)
_tarray(::Type{T}, x::Field) where {T} = _tarray(T, x.data)
_tarray(::Type{T}, x::Array{T,3}) where {T} = x
_tarray(::Type{T}, x::AbstractArray{<:Real,3}) where {T} = Array{T,3}(x)

# component n's target of a tuple or a NamedTuple (u, v, w)
@inline _dtarget(g::Union{Tuple,NamedTuple}, n) = g[n]
# the 3-D targets are inputs (a number or a level vector is not checked)
_input3(x::Field) = (x.data,)
_input3(x::AbstractArray{<:Any,3}) = (x,)
_input3(x) = ()
sweep_inputs(t::DampingTerm) = _damping_inputs(t.targets)
_damping_inputs(g::Union{Tuple,NamedTuple}) = (_input3(g[1])..., _input3(g[2])..., _input3(g[3])...)
_damping_inputs(g::DampingTargets) = g.a

@inline function add_row!(t::DampingTerm{3}, buf, a, b, s, j, k, ii, st)
    kl = k - t.k0
    @inbounds begin
        _damp_z!(buf, a, b, SWEEP_U, s, j, k, ii, st.u, t.p, t.coords[1][kl], t.targets, 1, kl)
        _damp_z!(buf, a, b, SWEEP_V, s, j, k, ii, st.v, t.p, t.coords[2][kl], t.targets, 2, kl)
        _damp_z!(buf, a, b, SWEEP_W, s, j, k, ii, st.w, t.p, t.coords[3][kl], t.targets, 3, kl)
    end
    return nothing
end
@inline _damp_z!(buf, a, b, f, s, j, k, ii, fd, p, ξ, g, n, kl) =
    _damp_row_z!(buf, a, b, f, s, j, k, ii, fd, p, ξ, _dtarget(g, n), kl)
@inline function _damp_z!(buf, a, b, f, s, j, k, ii, fd, p, ξ, g::DampingTargets, n, kl)
    kd = g.kind[n]
    kd == 0 ? _damp_row_z!(buf, a, b, f, s, j, k, ii, fd, p, ξ, g.c[n], kl) :
    kd == 1 ? _damp_row_z!(buf, a, b, f, s, j, k, ii, fd, p, ξ, g.v[n], kl) :
              _damp_row_z!(buf, a, b, f, s, j, k, ii, fd, p, ξ, g.a[n], kl)
    return nothing
end
@noinline function _damp_row_z!(buf, a, b, f, s, j, k, ii, fd, p, ξ, target, kl)
    in_damping(p, ξ) || return nothing
    λ = damping_rate(p, ξ)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, f, s] -= λ * (fd[i, j, k] - _damping_target(target, i, j, k, kl))
    end
    return nothing
end

@inline function add_row!(t::DampingTerm{1}, buf, a, b, s, j, k, ii, st)
    kl = k - t.k0
    g = t.targets
    _damp_row_x!(buf, a, b, SWEEP_U, s, j, k, ii, st.u, t.p, t.coords[1], t.i0, _dtarget(g, 1), kl)
    _damp_row_x!(buf, a, b, SWEEP_V, s, j, k, ii, st.v, t.p, t.coords[2], t.i0, _dtarget(g, 2), kl)
    _damp_row_x!(buf, a, b, SWEEP_W, s, j, k, ii, st.w, t.p, t.coords[3], t.i0, _dtarget(g, 3), kl)
    return nothing
end
@noinline function _damp_row_x!(buf, a, b, f, s, j, k, ii, fd, p, c, i0, target, kl)
    @inbounds for i in ii
        ξ = c[i - i0]
        in_damping(p, ξ) || continue
        λ = damping_rate(p, ξ)
        buf[i, a, b, f, s] -= λ * (fd[i, j, k] - _damping_target(target, i, j, k, kl))
    end
    return nothing
end

"""
    damping_levels(g, loc) -> AbstractVector

The heights of the `Nz` interior levels of a field at `loc` on `g`, as
[`damping_coordinates`](@ref) but without a copy (a range on a
`RegularGrid`, a view of the level heights on a `StretchedGrid`): the
coordinates a [`DampingTerm`](@ref) along z is built with every substage.
"""
damping_levels(g::RegularGrid, loc::Location) = Grids.znodes(g, loc)

"""
    sponge_term(α_max, z_start, targets, g) -> DampingTerm

The Rayleigh sponge of `add_sponge!` on `u, v, w` (the profile
[`sponge_profile`](@ref)`(α_max, z_start, z_top)` sampled at the level
heights) toward `targets` (fields `u, v, w`, or a tuple) as a term of the
fused sweep.
"""
sponge_term(α_max, z_start, targets, g::AbstractMesh) =
    DampingTerm(sponge_profile(α_max, z_start, g.z0 + g.Lz), 3,
                (damping_levels(g, FaceX()), damping_levels(g, FaceY()), damping_levels(g, FaceZ())), targets, g)

"""
    OptionalTerm(on, term)

`term` when `on`, nothing when not: a built-in term of the core whose
switch is a run-time flag (`sim.has_buoyancy`, …) keeps one term tuple
type per configuration, so a substage's sweep is compiled once whatever
the switches. The test costs one branch per row. Its footprint is the
term's, on or off.
"""
struct OptionalTerm{X<:SweepTerm} <: SweepTerm
    on::Bool
    term::X
end
footprint(t::OptionalTerm) = footprint(t.term)
sweep_inputs(t::OptionalTerm) = t.on ? sweep_inputs(t.term) : ()
prepare!(t::OptionalTerm, sim, Δt) = (t.on && prepare!(t.term, sim, Δt); nothing)
@inline add_row!(t::OptionalTerm, buf, a, b, s, j, k, ii, st) =
    (t.on && add_row!(t.term, buf, a, b, s, j, k, ii, st); nothing)
