# KA-f (#297): the halo fills as they were before the port (master at
# 70d9f83): the plane copies on views, in the same order, written against
# the same BC types. The reference of test/ka_halos.jl and of the KA-f
# timings (old loops vs launches).
# ---------------------------------------------------------------------------
module LegacyHalos
import Aeolus
using Aeolus.Grids: FaceX, FaceY, FaceZ, CellCenter, Location
using Aeolus.BoundaryConditions: Periodic, NoFlux, ImpermeableWall, NoSlipWall, FreeStreamBoundary,
                                 FluxBC, OpenBoundary, _ZWall, _wall_velocity, _side_axis,
                                 _face_index, _face_local, _center_gap

sl(d, ax, r) = ax == 1 ? view(d, r, :, :) : ax == 2 ? view(d, :, r, :) : view(d, :, :, r)

function legacy_fill!(f, bcs; key = f.data)
    g = f.mesh; d = f.data; loc = Aeolus.Fields.location(f)
    axis!(d, bcs.west, bcs.east, 1, g.Hx, g.Nx, loc, key)
    axis!(d, bcs.south, bcs.north, 2, g.Hy, g.Ny, loc, key)
    zaxis!(d, bcs.bottom, bcs.top, g.Hz, g.Nz, loc, g, key)
    return f
end

function periodic!(d, ax, H, N)
    for h in 1:H
        sl(d, ax, h) .= sl(d, ax, N + h)
        sl(d, ax, H + N + h) .= sl(d, ax, H + h)
    end
end
function mirror!(d, ax, H, N)
    for h in 1:H
        sl(d, ax, H + 1 - h) .= sl(d, ax, H + h)
        sl(d, ax, H + N + h) .= sl(d, ax, H + N + 1 - h)
    end
end
function mirror_side!(d, ax, high, H, N)
    for h in 1:H
        dst, src = high ? (H + N + h, H + N + 1 - h) : (H + 1 - h, H + h)
        sl(d, ax, dst) .= sl(d, ax, src)
    end
end
function odd_side!(d, ax, high, H, N)
    f = _face_index(high, H, N)
    f <= size(d, ax) || return
    for h in 1:H
        dst, src = high ? (f + h, f - h) : (f - h, f + h)
        1 <= dst <= size(d, ax) || continue
        sl(d, ax, dst) .= 2 .* sl(d, ax, f) .- sl(d, ax, src)
    end
end

axis!(d, ::Periodic, ::Periodic, ax, H, N, loc, key) = periodic!(d, ax, H, N)
axis!(d, ::NoFlux, ::NoFlux, ax, H, N, loc, key) = mirror!(d, ax, H, N)
axis!(d, a::OpenBoundary, b::OpenBoundary, ax, H, N, loc, key) =
    (open_face!(d, a, H, N, loc, key); open_face!(d, b, H, N, loc, key))

zaxis!(d, ::Periodic, ::Periodic, H, N, loc, g, key) = periodic!(d, 3, H, N)
zaxis!(d, ::NoFlux, ::NoFlux, H, N, loc, g, key) = mirror!(d, 3, H, N)
zaxis!(d, ::ImpermeableWall, ::ImpermeableWall, H, N, ::Union{CellCenter,FaceX,FaceY}, g, key) =
    mirror!(d, 3, H, N)
function zaxis!(d::AbstractArray{T}, ::ImpermeableWall, ::ImpermeableWall, H, N, ::FaceZ, g, key) where {T}
    sl(d, 3, H + 1) .= zero(T)
    H + N + 1 <= size(d, 3) && (sl(d, 3, H + N + 1) .= zero(T))
    for h in 1:H
        sl(d, 3, H + 1 - h) .= .-sl(d, 3, H + 1 + h)
        dst = H + N + 1 + h
        dst <= size(d, 3) && (sl(d, 3, dst) .= .-sl(d, 3, H + N + 1 - h))
    end
end
zaxis!(d, b::_ZWall, t::_ZWall, H, N, loc, g, key) = (wall_bottom!(d, b, H, N, loc); wall_top!(d, t, H, N, loc))
zaxis!(d, b::_ZWall, t::FreeStreamBoundary, H, N, loc, g, key) = (wall_bottom!(d, b, H, N, loc); fs_top!(d, t, H, N, loc))
zaxis!(d, b::_ZWall, t::OpenBoundary, H, N, loc, g, key) = (wall_bottom!(d, b, H, N, loc); open_face!(d, t, H, N, loc, key))
zaxis!(d, b::OpenBoundary, t::OpenBoundary, H, N, loc, g, key) =
    (open_face!(d, b, H, N, loc, key); open_face!(d, t, H, N, loc, key))
function zaxis!(d::AbstractArray{T}, b, t::FluxBC, H, N, loc::CellCenter, g, key) where {T}
    zaxis!(d, b, b, H, N, loc, g, key)
    Γ = T(t.f); kN = H + N
    for h in 1:H
        src, dst = kN + 1 - h, kN + h
        sl(d, 3, dst) .= sl(d, 3, src) .+ Γ * T(_center_gap(g, src, dst))
    end
end

function wall_bottom!(d::AbstractArray{T}, bc::NoSlipWall, H, N, loc::Union{FaceX,FaceY}) where {T}
    U2 = 2 * _wall_velocity(bc, loc, T)
    for h in 1:H; sl(d, 3, H + 1 - h) .= U2 .- sl(d, 3, H + h); end
end
function wall_top!(d::AbstractArray{T}, bc::NoSlipWall, H, N, loc::Union{FaceX,FaceY}) where {T}
    U2 = 2 * _wall_velocity(bc, loc, T)
    for h in 1:H; sl(d, 3, H + N + h) .= U2 .- sl(d, 3, H + N + 1 - h); end
end
wall_bottom!(d, bc::_ZWall, H, N, loc::Union{CellCenter,FaceX,FaceY}) =
    for h in 1:H; sl(d, 3, H + 1 - h) .= sl(d, 3, H + h); end
wall_top!(d, bc::_ZWall, H, N, loc::Union{CellCenter,FaceX,FaceY}) =
    for h in 1:H; sl(d, 3, H + N + h) .= sl(d, 3, H + N + 1 - h); end
function wall_bottom!(d::AbstractArray{T}, ::_ZWall, H, N, ::FaceZ) where {T}
    sl(d, 3, H + 1) .= zero(T)
    for h in 1:H; sl(d, 3, H + 1 - h) .= .-sl(d, 3, H + 1 + h); end
end
function wall_top!(d::AbstractArray{T}, ::_ZWall, H, N, ::FaceZ) where {T}
    kw = H + N + 1
    kw <= size(d, 3) && (sl(d, 3, kw) .= zero(T))
    for h in 1:H
        dst = kw + h
        dst <= size(d, 3) && (sl(d, 3, dst) .= .-sl(d, 3, kw - h))
    end
end

fs_top!(d, ::FreeStreamBoundary, H, N, ::Union{CellCenter,FaceX,FaceY}) = mirror_side!(d, 3, true, H, N)
function fs_top!(d::AbstractArray{T}, bc::FreeStreamBoundary, H, N, ::FaceZ) where {T}
    kw = _face_index(true, H, N)
    if kw <= size(d, 3)
        for j in axes(d, 2), i in axes(d, 1); d[i, j, kw] = T(bc.w_top[i]); end
    end
    odd_side!(d, 3, true, H, N)
end

function plane_of(bc::OpenBoundary, loc, key)
    loc isa FaceX && return get(bc.planes, :u, nothing)
    loc isa FaceY && return get(bc.planes, :v, nothing)
    loc isa FaceZ && return get(bc.planes, :w, nothing)
    name = get(bc.state.bound, key, nothing)
    return name === nothing ? nothing : get(bc.planes, name, nothing)
end

function open_face!(d::AbstractArray{T}, bc::OpenBoundary, H, N, loc, key) where {T}
    ax, high = _side_axis(bc.side)
    p = plane_of(bc, loc, key)
    (p === nothing || p.mode === :mirror) && return mirror_side!(d, ax, high, H, N)
    p.mode === :inflow && return inflow_face!(d, bc, p, ax, high, H, N)
    if p.normal && p.mode === :odd
        f = _face_index(high, H, N)
        sl(d, ax, f) .= bc.homogeneous ? zero(T) : sl(p.data, ax, _face_local(p))
        return odd_side!(d, ax, high, H, N)
    end
    bc.homogeneous ? Base.fill!(sl(d, ax, p.range), zero(T)) : (sl(d, ax, p.range) .= p.data)
end

function inflow_face!(d::AbstractArray{T}, bc, p, ax, high, H, N) where {T}
    pn = bc.planes[ax == 1 ? :u : ax == 2 ? :v : :w]
    fl = _face_local(pn)
    s = high ? one(T) : -one(T)
    t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    at(A, l, a, b) = ax == 1 ? A[l, a, b] : ax == 2 ? A[a, l, b] : A[a, b, l]
    for b in axes(d, t2), a in axes(d, t1)
        un = s * at(pn.data, fl, a, b)
        for h in 1:H
            dst = high ? H + N + h : H + 1 - h
            v = un < 0 ? (bc.homogeneous ? zero(T) : at(p.data, high ? h : H + 1 - h, a, b)) :
                         at(d, high ? H + N + 1 - h : H + h, a, b)
            ax == 1 ? (d[dst, a, b] = v) : ax == 2 ? (d[a, dst, b] = v) : (d[a, b, dst] = v)
        end
    end
end
end # module LegacyHalos
