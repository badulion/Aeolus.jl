# Nesting: halo-only storage of the parent state on the child mesh (#329,
# item 4). A one-way or two-way NestedRun without a sponge reads the
# prolongated parent only at the child's open-boundary points — the halos
# and the boundary-normal face plane — so its three field sets (the two
# window ends and the blend) need not be child-sized.

# The six non-overlapping halo slabs of a storage box of size `n`, `lo`
# layers deep on the low side and `hi` on the high side of each axis: the
# two x slabs span the whole y–z extent, the y slabs the inner x extent,
# the z slabs the inner x–y extent.
function _shell_boxes(n::NTuple{3,Int}, lo::NTuple{3,Int}, hi::NTuple{3,Int})
    ix = lo[1]+1:n[1]-hi[1]; iy = lo[2]+1:n[2]-hi[2]
    return ((1:lo[1], 1:n[2], 1:n[3]), (n[1]-hi[1]+1:n[1], 1:n[2], 1:n[3]),
            (ix, 1:lo[2], 1:n[3]),     (ix, n[2]-hi[2]+1:n[2], 1:n[3]),
            (ix, iy, 1:lo[3]),         (ix, iy, n[3]-hi[3]+1:n[3]))
end

"""
    _ShellArray{T} <: AbstractArray{T,3}

A storage-indexed child array (size `n`, halos included) that stores only
its halo shell: `lo[d]` layers on the low side and `hi[d]` on the high side
of axis `d`, as six dense slabs (see `_shell_boxes`). Reading or writing an
interior point is an error. For a `FaceX`/`FaceY`/`FaceZ` member the low
side of its own axis holds one more layer, the boundary-normal face plane
of the west/south/bottom face, which the open boundaries read too.

For a child of `N³` cells with halo `H` the shell holds
`(N+2H)³ − N³` points instead of `(N+2H)³` (40 % for `N = 32`, `H = 3`;
24 % for `N = 64`).
"""
struct _ShellArray{T} <: AbstractArray{T,3}
    n::NTuple{3,Int}
    lo::NTuple{3,Int}
    hi::NTuple{3,Int}
    xl::Array{T,3}; xh::Array{T,3}
    yl::Array{T,3}; yh::Array{T,3}
    zl::Array{T,3}; zh::Array{T,3}
end

function _ShellArray{T}(n::NTuple{3,Int}, lo::NTuple{3,Int}, hi::NTuple{3,Int}) where {T}
    all(lo .+ hi .<= n) || throw(ArgumentError("_ShellArray: the shell is deeper than the array"))
    b = _shell_boxes(n, lo, hi)
    z(r) = zeros(T, length.(r)...)
    return _ShellArray{T}(n, lo, hi, z(b[1]), z(b[2]), z(b[3]), z(b[4]), z(b[5]), z(b[6]))
end

"A halo-only array for a child field at `loc` on mesh `cm`."
function _shell_array(::Type{T}, cm::NestChildMesh, loc::Location) where {T}
    H = (cm.Hx, cm.Hy, cm.Hz)
    ax = _stagger_axis(loc)
    lo = ntuple(d -> H[d] + (d == ax), 3)
    return _ShellArray{T}(totalsize(cm), lo, H)
end

Base.size(a::_ShellArray) = a.n
Base.IndexStyle(::Type{<:_ShellArray}) = IndexCartesian()

@noinline _shell_interior(i, j, k) =
    throw(ArgumentError("a halo-only (storage = :shell) nest state does not store the " *
                        "child interior point ($i, $j, $k); use storage = :full"))

@inline function Base.getindex(a::_ShellArray, i::Int, j::Int, k::Int)
    @boundscheck checkbounds(a, i, j, k)
    n1, n2, n3 = a.n; l1, l2, l3 = a.lo; h1, h2, h3 = a.hi
    @inbounds begin
        i <= l1      && return a.xl[i, j, k]
        i > n1 - h1  && return a.xh[i - (n1 - h1), j, k]
        ii = i - l1
        j <= l2      && return a.yl[ii, j, k]
        j > n2 - h2  && return a.yh[ii, j - (n2 - h2), k]
        jj = j - l2
        k <= l3      && return a.zl[ii, jj, k]
        k > n3 - h3  && return a.zh[ii, jj, k - (n3 - h3)]
    end
    _shell_interior(i, j, k)
end

@inline function Base.setindex!(a::_ShellArray, v, i::Int, j::Int, k::Int)
    @boundscheck checkbounds(a, i, j, k)
    n1, n2, n3 = a.n; l1, l2, l3 = a.lo; h1, h2, h3 = a.hi
    @inbounds begin
        i <= l1      && return (a.xl[i, j, k] = v)
        i > n1 - h1  && return (a.xh[i - (n1 - h1), j, k] = v)
        ii = i - l1
        j <= l2      && return (a.yl[ii, j, k] = v)
        j > n2 - h2  && return (a.yh[ii, j - (n2 - h2), k] = v)
        jj = j - l2
        k <= l3      && return (a.zl[ii, jj, k] = v)
        k > n3 - h3  && return (a.zh[ii, jj, k - (n3 - h3)] = v)
    end
    _shell_interior(i, j, k)
end

_slabs(a::_ShellArray) = (a.xl, a.xh, a.yl, a.yh, a.zl, a.zh)

# (slab, offset of its storage indices, storage box) for the six slabs.
function _shell_slabs(a::_ShellArray)
    b = _shell_boxes(a.n, a.lo, a.hi)
    s = _slabs(a)
    off(r) = (first(r[1]) - 1, first(r[2]) - 1, first(r[3]) - 1)
    return ntuple(q -> (s[q], off(b[q]), b[q]), 6)
end

Base.fill!(a::_ShellArray, x) = (foreach(s -> fill!(s, x), _slabs(a)); a)
Base.copyto!(a::_ShellArray, b::_ShellArray) =
    (a.n == b.n && a.lo == b.lo && a.hi == b.hi ||
         throw(DimensionMismatch("_ShellArray: different shells"));
     foreach(copyto!, _slabs(a), _slabs(b)); a)

# The time blend of two halo-only snapshots, slab by slab (the one kernel).
function blend_window!(dst::_ShellArray, A::_ShellArray, B::_ShellArray, τ)
    (dst.n == A.n == B.n && dst.lo == A.lo == B.lo && dst.hi == A.hi == B.hi) ||
        throw(DimensionMismatch("blend_window!: different shells"))
    for (d, a, b) in zip(_slabs(dst), _slabs(A), _slabs(B))
        blend_window!(d, a, b, τ)
    end
    return dst
end

"Bytes of field data a nest state holds (halo slabs only for a shell)."
_data_bytes(a::Array) = sizeof(a)
_data_bytes(a::_ShellArray) = sum(sizeof, _slabs(a))
