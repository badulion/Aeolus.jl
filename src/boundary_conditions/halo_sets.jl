# SC-b (#402) — threaded halo fills: one launch over all the faces of a
# field, and one launch over a whole set of fields filled together.
#
# Included inside `module BoundaryConditions`, after the fill methods.
#
# The per-phase launches of `fill_halos!` (halo_kernels.jl) are too small to
# thread at the sizes the flagship runs (every slab of a 64 × 64 × 32 field
# is below the backend's 32 768-point threshold), so a fill ran serially
# between threaded stencils: `fill_halos!` 1.0× at 12 threads (KA-o, #308).
# The phases cannot simply run at once — an edge or corner takes the value of
# the last axis that reaches it — but they decompose:
#
#   * every x and y phase reads and writes one k-plane only (its bodies move
#     along x or y and nothing else; the plane buffers of an open face are
#     read at the same (j, k)), so "x then y" can run plane by plane, the
#     planes independent;
#   * every z phase reads and writes one (i, j) column only, so the z phases
#     can run row by row (a j-row of columns), the rows independent — once
#     every plane has had its x and y fills.
#
# A fill of a set of fields is then two launches: (1) every work item takes
# a contiguous chunk of k-planes of every field and runs that field's x and
# y phases restricted to those planes; (2) every work item takes a chunk of
# j-rows of every field and runs its z phases restricted to those rows.
# Each restricted phase runs on the calling task in `launch!`'s serial order
# (the restriction of the whole phase's order), so a chunk computes every
# point from the same operands in the same order as the unrestricted fill:
# the same bits at every thread count and every chunking, including the
# phases whose layers depend on each other (N < H).
#
# The restriction is carried by the array: the fill methods (every BC type,
# halo_kernels.jl and the files included above) run unchanged on a
# `_Clip` of the storage array, and the clip's `_halo_launch!` method
# narrows the launch to the clip's planes or rows.

"""
    _Clip(parent, axis, r)

The storage array `parent` seen by a halo fill restricted to the indices `r`
along `axis`: `3` (k-planes, for the x and y phases) or `2` (j-rows, for the
z phases). Size and element access are the parent's; a halo phase launched
on it covers only the pencils inside the clip, serially (`_halo_launch!`).
"""
struct _Clip{T, A<:AbstractArray{T,3}} <: AbstractArray{T,3}
    parent::A
    axis::Int
    r::UnitRange{Int}
end
Base.size(c::_Clip) = size(c.parent)
Base.@propagate_inbounds Base.getindex(c::_Clip, i::Int, j::Int, k::Int) = c.parent[i, j, k]
Base.@propagate_inbounds Base.setindex!(c::_Clip, v, i::Int, j::Int, k::Int) =
    (c.parent[i, j, k] = v; c)

# The halo launch on a clip: the phase's own ranges, the clipped axis
# narrowed to the clip, run on the calling task in launch!'s serial order.
function _halo_launch!(body!::F, c::_Clip, ax::_Ax{A}, lr::UnitRange{Int},
                       independent::Bool, args::Vararg{Any,M}) where {F, A, M}
    isempty(lr) && return c
    d = c.parent
    c.axis == (A == 3 ? 2 : 3) ||
        throw(ArgumentError("halo phase along axis $A cannot run clipped along axis $(c.axis)"))
    checkbounds(d, A == 1 ? lr : 1, A == 2 ? lr : 1, A == 3 ? lr : 1)
    if A == 1
        r = (1:size(d, 2), c.r)
        o = first(lr) - 1
        L = length(lr)
        L == 1 ? launch!(_SERIAL_HALO, _xpencil!, r, Val(1), o, body!, d, ax, args) :
        L == 2 ? launch!(_SERIAL_HALO, _xpencil!, r, Val(2), o, body!, d, ax, args) :
        L == 3 ? launch!(_SERIAL_HALO, _xpencil!, r, Val(3), o, body!, d, ax, args) :
        L == 4 ? launch!(_SERIAL_HALO, _xpencil!, r, Val(4), o, body!, d, ax, args) :
                 launch!(_SERIAL_HALO, _xpencil!, r, L, o, body!, d, ax, args)
        return c
    end
    r = A == 2 ? (1:size(d, 1), lr, c.r) : (1:size(d, 1), c.r, lr)
    launch!(_SERIAL_HALO, body!, r, d, ax, args...)
    return c
end

# Part `c` of `n` of the range 1:N (contiguous, sizes differing by at most 1).
@inline _part(N::Int, c::Int, n::Int) = (div(N * (c - 1), n) + 1):div(N * c, n)

# Points written by the x and y phases, and by the z phases, of a field.
_xy_points(f::Field) = (s = size(f.data); g = f.mesh; 2g.Hx * s[2] * s[3] + 2g.Hy * s[1] * s[3])
_z_points(f::Field) = (s = size(f.data); g = f.mesh; 2g.Hz * s[1] * s[2])

# Work item `c` of `n`: planes (x, y phases) or rows (z phases) part `c` of
# every field of the set — a tuple unrolled (fields of different locations
# and boundary conditions are different types), or a vector of fields that
# share one set of boundary conditions (`keys === nothing`: each field is
# its own key).
@inline _xy_item!(c, fs, bcs, keys, n) = _xy_each!(c, n, fs, bcs, keys)
@inline _z_item!(c, fs, bcs, keys, n) = _z_each!(c, n, fs, bcs, keys)

@inline function _xy_one!(c, n, f::Field, bc, key)
    d = f.data; g = f.mesh; loc = location(f)
    r = _part(size(d, 3), c, n)
    if !isempty(r)
        cd = _Clip(d, 3, r)
        _fill_x!(cd, bc.west,  bc.east,  g.Hx, g.Nx, loc, key)
        _fill_y!(cd, bc.south, bc.north, g.Hy, g.Ny, loc, key)
    end
    return nothing
end
@inline function _z_one!(c, n, f::Field, bc, key)
    d = f.data; g = f.mesh
    r = _part(size(d, 2), c, n)
    isempty(r) || _fill_z!(_Clip(d, 2, r), bc.bottom, bc.top, g.Hz, g.Nz, location(f), g, key)
    return nothing
end

@inline _xy_each!(c, n, ::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function _xy_each!(c, n, fs::Tuple, bcs::Tuple, keys::Tuple)
    _xy_one!(c, n, first(fs), first(bcs), first(keys))
    return _xy_each!(c, n, Base.tail(fs), Base.tail(bcs), Base.tail(keys))
end
@inline _z_each!(c, n, ::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function _z_each!(c, n, fs::Tuple, bcs::Tuple, keys::Tuple)
    _z_one!(c, n, first(fs), first(bcs), first(keys))
    return _z_each!(c, n, Base.tail(fs), Base.tail(bcs), Base.tail(keys))
end
function _xy_each!(c, n, fs::AbstractVector, bc::FieldBoundaryConditions, ::Nothing)
    for f in fs
        _xy_one!(c, n, f, bc, f.data)
    end
    return nothing
end
function _z_each!(c, n, fs::AbstractVector, bc::FieldBoundaryConditions, ::Nothing)
    for f in fs
        _z_one!(c, n, f, bc, f.data)
    end
    return nothing
end

# The two launches. One work item per thread and chunk: `n` parts of every
# field, each launch weighted by the halo points it writes, so that the
# backend's threshold and points-per-task rule see the real work. (Counting
# a halo point as less than a stencil point, which it is when the field is
# in cache, measured slower in the regime of a run, where the stencils
# between two fills have evicted the field: SC-b in ka_port.md.)
function _fill_set_threaded!(b, fs, bcs, keys, n::Int)
    wxy = cld(sum(_xy_points, fs), n)
    wz  = cld(sum(_z_points, fs), n)
    launch_weighted!(b, wxy, _xy_item!, (1:n,), fs, bcs, keys, n)
    launch_weighted!(b, wz,  _z_item!,  (1:n,), fs, bcs, keys, n)
    return fs
end

# The serial fill of a set: field by field, phase by phase (the historical
# order, and the one-thread path).
_fill_set_serial!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
function _fill_set_serial!(fs::Tuple, bcs::Tuple, keys::Tuple)
    _fill_one!(first(fs), first(bcs), first(keys))
    return _fill_set_serial!(Base.tail(fs), Base.tail(bcs), Base.tail(keys))
end

_bcs_tuple(bcs::Tuple, fs) = bcs
_bcs_tuple(bcs, fs) = map(_ -> bcs, fs)

"""
    fill_halos!(fields::Tuple, bcs; keys = data of each field) -> fields

Fill the halos of a set of fields together: `bcs` is one
[`FieldBoundaryConditions`](@ref) for all of them or a tuple with one per
field (`keys`, the per-field key of [`OpenBoundary`](@ref) data, likewise).
The result is that of `fill_halos!(f, bcs)` on each field in turn, bit for
bit; the fields must not share storage.

Threaded (SC-b, #402), the whole set is two launches: the x and y phases
of every field, a chunk of k-planes per work item, then the z phases, a
chunk of j-rows per work item (every x/y phase stays inside a plane and
every z phase inside a column, so the axis order x, y, z of the edges and
corners is kept). Below the backend's threshold (counted over the halo
points of the whole set) and at one thread the fields are filled one
after the other with the per-phase launches.
"""
function fill_halos!(fs::Tuple{Vararg{Field}}, bcs; keys::Tuple = map(f -> f.data, fs))
    n = length(fs)
    n == 0 && return fs
    bt = _bcs_tuple(bcs, fs)
    length(bt) == n && length(keys) == n ||
        throw(ArgumentError("fill_halos!: $n fields, $(length(bt)) boundary conditions, $(length(keys)) keys"))
    b = Backends.backend(first(fs).data)
    np = sum(_xy_points, fs) + sum(_z_points, fs)
    if b isa Backends.CPUBackend && !runs_serially(b, np)
        _fill_set_threaded!(b, fs, bt, keys, Threads.nthreads())
    else
        _fill_set_serial!(fs, bt, keys)
    end
    return fs
end

"""
    fill_halos!(fields::AbstractVector{<:Field}, bcs::FieldBoundaryConditions) -> fields

The set fill for a vector of fields with one set of boundary conditions
(the tracers of a run): each field is its own key, the result is that of
`fill_halos!(f, bcs)` on each field in turn, bit for bit, and threaded
the whole vector is the same two launches as a tuple.
"""
function fill_halos!(fs::AbstractVector{<:Field}, bcs::FieldBoundaryConditions)
    isempty(fs) && return fs
    b = Backends.backend(first(fs).data)
    np = sum(_xy_points, fs) + sum(_z_points, fs)
    if b isa Backends.CPUBackend && !runs_serially(b, np)
        _fill_set_threaded!(b, fs, bcs, nothing, Threads.nthreads())
    else
        for f in fs
            _fill_one!(f, bcs, f.data)
        end
    end
    return fs
end
