# KA-f (#297) — the per-axis halo kernels: every halo fill of every BC type
# (and the multigrid level halos) is a `launch!` of one of the point bodies
# below over a face slab.
#
# Included inside `module BoundaryConditions`, before the fill methods.
#
# A fill phase of the pre-port code — "for h in 1:H, copy plane src(h) to
# plane dst(h)" on one or both faces of an axis — is one launch over the
# slab `layers × (full extent of the two tangential axes)`, where the layer
# index is `h` (or the storage index itself, for a face slot or a copy
# from a plane buffer). The body writes the storage points of layer `h`
# only, so a body is a point function on the face index. Phases run in the
# pre-port order (x, then y, then z; within an axis the faces in their old
# order), because edges and corners take the values of the previous axis.
#
# Bit for bit with the pre-port fill: every point gets the same arithmetic
# from the same operands. A phase on the pencil normal to the face (fixed
# tangential indices) reads only that pencil, so pencils are independent;
# within a pencil the old code ran the layers in ascending `h`. When the
# interior is at least as deep as the halo (`N > H`) no layer of a phase
# reads what another layer writes, and the launch may run threaded. When
# it is not (a 2-D run with `Ny = 1`, say), the layers of a pencil depend
# on each other in the old order, and the launch runs on the calling task
# in `launch!`'s serial loop order, which keeps ascending `h` per pencil:
# the old result, at every thread count. Only a phase that fills both faces
# of an axis at once can read its own writes; a one-sided phase reads the
# interior side of its face (and the face slot, set by an earlier launch)
# and writes only beyond it, so it is always independent.

# The axis a halo phase works on, as a type (the bodies specialise on it).
struct _Ax{A} end

# The layer coordinate of point (i, j, k), and the storage index of layer
# `l` on the pencil through (i, j, k).
@inline _layer(::_Ax{1}, i, j, k) = i
@inline _layer(::_Ax{2}, i, j, k) = j
@inline _layer(::_Ax{3}, i, j, k) = k
@inline _ix(::_Ax{1}, l, i, j, k) = (l, j, k)
@inline _ix(::_Ax{2}, l, i, j, k) = (i, l, k)
@inline _ix(::_Ax{3}, l, i, j, k) = (i, j, l)

# The serial reference for a launch whose layers depend on each other.
const _SERIAL_HALO = Backends.CPUBackend(serial = true)

# Launch `body!(i, j, k, d, ax, args...)` over the layers `lr` along `ax`
# (all of the two tangential axes). `independent = false` runs it on the
# calling task, in launch!'s serial order (ascending layer per pencil).
#
# Along x the layers are the contiguous direction, so a work item is one
# x-pencil (j, k) running its layers in ascending order (`_xpencil!`): a
# 3-D launch would make the layer loop launch!'s innermost loop, of 1-4
# iterations, whose set-up cost made the multigrid halo (H = 1) 2.7x the
# old plane copies; with the layer count a compile-time constant (1-4) the
# loop unrolls. A pencil owns all its layers, so this is exact even when
# the layers depend on each other.
function _halo_launch!(body!::F, d::AbstractArray{<:Any,3}, ax::_Ax{A}, lr::UnitRange{Int},
                       independent::Bool, args::Vararg{Any,M}) where {F, A, M}
    isempty(lr) && return d
    checkbounds(d, A == 1 ? lr : 1, A == 2 ? lr : 1, A == 3 ? lr : 1)
    if A == 1
        b = Backends.backend(d)
        r = (1:size(d, 2), 1:size(d, 3))
        o = first(lr) - 1
        L = length(lr)
        L == 1 ? launch!(b, _xpencil!, r, Val(1), o, body!, d, ax, args) :
        L == 2 ? launch!(b, _xpencil!, r, Val(2), o, body!, d, ax, args) :
        L == 3 ? launch!(b, _xpencil!, r, Val(3), o, body!, d, ax, args) :
        L == 4 ? launch!(b, _xpencil!, r, Val(4), o, body!, d, ax, args) :
                 launch!(b, _xpencil!, r, L, o, body!, d, ax, args)
        return d
    end
    r = A == 2 ? (1:size(d, 1), lr, 1:size(d, 3)) : (1:size(d, 1), 1:size(d, 2), lr)
    b = independent ? Backends.backend(d) : _SERIAL_HALO
    launch!(b, body!, r, d, ax, args...)
    return d
end

# The axis as a number: one static branch per axis (indexing a tuple of the
# three axis types would give a Union and a dynamic call that allocates).
@inline function _halo_launch!(body!::F, d::AbstractArray{<:Any,3}, ax::Int, lr::UnitRange{Int},
                               independent::Bool, args::Vararg{Any,M}) where {F, M}
    ax == 1 ? _halo_launch!(body!, d, _Ax{1}(), lr, independent, args...) :
    ax == 2 ? _halo_launch!(body!, d, _Ax{2}(), lr, independent, args...) :
              _halo_launch!(body!, d, _Ax{3}(), lr, independent, args...)
end

# One x-pencil (j, k): the layers o+1 … o+L in ascending order.
@inline _nlayers(::Val{L}) where {L} = L
@inline _nlayers(L::Int) = L
@inline function _xpencil!(j, k, L, o, body!::F, d, ax, args) where {F}
    for t in 1:_nlayers(L)
        body!(o + t, j, k, d, ax, args...)
    end
    return nothing
end

# ---- bodies (the layer index is h = 1…H unless stated) ----------------------

# Periodic: copy from the opposite side, low face first.
@inline function _periodic_body!(i, j, k, d, ax, H, N)
    h = _layer(ax, i, j, k)
    @inbounds d[_ix(ax, h, i, j, k)...] = d[_ix(ax, N + h, i, j, k)...]
    @inbounds d[_ix(ax, H + N + h, i, j, k)...] = d[_ix(ax, H + h, i, j, k)...]
    return nothing
end

# Mirror (zero gradient) on the low and/or high face.
@inline function _mirror_body!(i, j, k, d, ax, H, N, lo::Bool, hi::Bool)
    h = _layer(ax, i, j, k)
    if lo
        @inbounds d[_ix(ax, H + 1 - h, i, j, k)...] = d[_ix(ax, H + h, i, j, k)...]
    end
    if hi
        @inbounds d[_ix(ax, H + N + h, i, j, k)...] = d[_ix(ax, H + N + 1 - h, i, j, k)...]
    end
    return nothing
end

# Antisymmetric about a wall value: ghost = U2 − interior (U2 = 2 u_wall).
@inline function _reflect_body!(i, j, k, d, ax, H, N, lo::Bool, hi::Bool, U2)
    h = _layer(ax, i, j, k)
    if lo
        @inbounds d[_ix(ax, H + 1 - h, i, j, k)...] = U2 - d[_ix(ax, H + h, i, j, k)...]
    end
    if hi
        @inbounds d[_ix(ax, H + N + h, i, j, k)...] = U2 - d[_ix(ax, H + N + 1 - h, i, j, k)...]
    end
    return nothing
end

# Antisymmetric about a zero wall face (w at a wall): ghost = −interior, the
# low face at H+1, the high face at H+N+1; layers past the array are skipped.
@inline function _negate_body!(i, j, k, d, ax, H, N, lo::Bool, hi::Bool)
    h = _layer(ax, i, j, k)
    if lo
        @inbounds d[_ix(ax, H + 1 - h, i, j, k)...] = -d[_ix(ax, H + 1 + h, i, j, k)...]
    end
    if hi
        dst = H + N + 1 + h
        if dst <= size(d, _axis(ax))
            @inbounds d[_ix(ax, dst, i, j, k)...] = -d[_ix(ax, H + N + 1 - h, i, j, k)...]
        end
    end
    return nothing
end

# Antisymmetric about the (already set) face value at storage index f:
# d[f ± h] = 2 d[f] − d[f ∓ h], on the high (+) or low (−) side.
@inline function _odd_body!(i, j, k, d, ax, f, high::Bool)
    h = _layer(ax, i, j, k)
    dst, src = high ? (f + h, f - h) : (f - h, f + h)
    if 1 <= dst <= size(d, _axis(ax))
        @inbounds d[_ix(ax, dst, i, j, k)...] = 2 * d[_ix(ax, f, i, j, k)...] - d[_ix(ax, src, i, j, k)...]
    end
    return nothing
end

# Fixed-gradient top (FluxBC, #254): ghost = interior + Γ·(z_c(dst) − z_c(src)).
@inline function _gradient_body!(i, j, k, d::AbstractArray{T}, ax, kN, Γ, g) where {T}
    h = _layer(ax, i, j, k)
    src, dst = kN + 1 - h, kN + h
    δ = Γ * T(_center_gap(g, src, dst))
    @inbounds d[_ix(ax, dst, i, j, k)...] = d[_ix(ax, src, i, j, k)...] + δ
    return nothing
end

# The layer index is the storage index: a constant (a wall face, a
# homogeneous open slab), a profile along x (the free-stream w_top), or the
# matching point of a plane buffer whose layer `l` is storage index l + o.
@inline function _value_body!(i, j, k, d, ax, v)
    @inbounds d[i, j, k] = v
    return nothing
end
@inline function _xprofile_body!(i, j, k, d::AbstractArray{T}, ax, w) where {T}
    @inbounds d[i, j, k] = T(w[i])
    return nothing
end
@inline function _buffer_body!(i, j, k, d, ax, P, o)
    l = _layer(ax, i, j, k)
    @inbounds d[i, j, k] = P[_ix(ax, l - o, i, j, k)...]
    return nothing
end

# Open scalar in :inflow mode: the plane data (zero when homogeneous) where
# the face's normal velocity — layer `fl` of the normal plane's buffer Pn —
# points into the domain, the mirror of the interior where it points out.
@inline function _inflow_body!(i, j, k, d::AbstractArray{T}, ax, H, N, high::Bool, P, Pn, fl,
                               homogeneous::Bool) where {T}
    h = _layer(ax, i, j, k)
    s = high ? one(T) : -one(T)                  # outward sign
    @inbounds un = s * Pn[_ix(ax, fl, i, j, k)...]
    dst = high ? H + N + h : H + 1 - h
    @inbounds v = if un < 0                      # inflow: the data
        homogeneous ? zero(T) : P[_ix(ax, high ? h : H + 1 - h, i, j, k)...]
    else                                         # outflow or calm: zero gradient
        d[_ix(ax, high ? H + N + 1 - h : H + h, i, j, k)...]
    end
    @inbounds d[_ix(ax, dst, i, j, k)...] = v
    return nothing
end

@inline _axis(::_Ax{A}) where {A} = A

# ---- the per-axis fills ------------------------------------------------------
# `ax` is 1, 2 or 3 (x, y, z); `H`, `N` the halo width and the interior
# size along it.

"Periodic halo of axis `ax`: both faces, the low one first (per layer)."
_halo_periodic!(d, ax::Int, H::Int, N::Int) =
    _halo_launch!(_periodic_body!, d, ax, 1:H, N >= H, H, N)

"Mirror (zero-gradient) halo of axis `ax` on the low and/or high face."
_halo_mirror!(d, ax::Int, H::Int, N::Int, lo::Bool = true, hi::Bool = true) =
    _halo_launch!(_mirror_body!, d, ax, 1:H, !(lo && hi) || N >= H, H, N, lo, hi)

"Halo antisymmetric about the wall value `U2/2` on the low and/or high face."
_halo_reflect!(d, ax::Int, H::Int, N::Int, lo::Bool, hi::Bool, U2) =
    _halo_launch!(_reflect_body!, d, ax, 1:H, !(lo && hi) || N >= H, H, N, lo, hi, U2)

"Halo antisymmetric about a zero face (`H+1` low, `H+N+1` high), faces not written."
_halo_negate!(d, ax::Int, H::Int, N::Int, lo::Bool, hi::Bool) =
    _halo_launch!(_negate_body!, d, ax, 1:H, !(lo && hi) || N > H, H, N, lo, hi)

"Halo antisymmetric about the face value at storage index `f` (`high`: beyond it)."
_halo_odd!(d, ax::Int, f::Int, H::Int, N::Int, high::Bool) =
    _halo_launch!(_odd_body!, d, ax, 1:H, true, f, high)

"Set the storage layers `l` (an index: a face slot, or a range) of axis `ax` to the value `v`."
_halo_set!(d, ax::Int, l::Int, v) = _halo_set!(d, ax, l:l, v)
_halo_set!(d, ax::Int, r::UnitRange{Int}, v) = _halo_launch!(_value_body!, d, ax, r, true, v)

"Fixed-gradient top of a scalar: the `H` layers above storage index `kN = H + N`."
_halo_gradient!(d, H::Int, N::Int, Γ, g) =
    _halo_launch!(_gradient_body!, d, 3, 1:H, true, H + N, Γ, g)

"Open scalar in `:inflow` mode on one face (see `_inflow_body!`)."
_halo_inflow!(d, ax::Int, H::Int, N::Int, high::Bool, P, Pn, fl::Int, homogeneous::Bool) =
    _halo_launch!(_inflow_body!, d, ax, 1:H, true, H, N, high, P, Pn, fl, homogeneous)

"Copy layers `r .- o` of the plane buffer `P` into the storage layers `r` of axis `ax`."
_halo_copy!(d, ax::Int, r::UnitRange{Int}, P::AbstractArray, o::Int) =
    _halo_launch!(_buffer_body!, d, ax, r, true, P, o)
