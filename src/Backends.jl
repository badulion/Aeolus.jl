module Backends

# Compute backends and the kernel layer (KA-a, #294; port plan #291).
#
# Every KernelAbstractions (KA) kernel of Aeolus lives in this file
# (maintainer decision, 2026-10-06). The rest of the code base writes a
# plain Julia *body* — `body!(i, j, k, args...)` updating one point — and
# hands it to [`launch!`](@ref) with the storage-index ranges to cover.
# The backend decides how the ranges are walked: serially in the
# historical `for k, for j, for i` order, or as a KA kernel on the CPU's
# threads (later: on a GPU, Phase 81). Pointwise bodies therefore give
# identical bits on every path. Reductions use one canonical order that
# does not depend on the thread count ([`plane_partials!`](@ref)).
#
# Design page: docs/src/design/ka_port.md.

using KernelAbstractions
const KA = KernelAbstractions
using LinearAlgebra: BLAS

export AbstractBackend, CPUBackend, GPUBackend, allocate, to_host, backend_name,
       backend, default_backend, set_default_backend!, with_backend, init_threads!,
       runs_serially, launch!, range_map!,
       plane_partials!, ordered_sum, interior_sum, interior_dot, plane_mean!,
       column_partials, interior_max, interior_all_finite

# =========================================================================
# Backend types
# =========================================================================

"""
    AbstractBackend

Base type for the compute-backend selector. Subtypes:

  * [`CPUBackend`](@ref) — default; `Base.Array` storage, kernels run by
    KernelAbstractions on the CPU's Julia threads (or serially).
  * [`GPUBackend`](@ref) — stub until Phase 81.
"""
abstract type AbstractBackend end

"""
Default serial threshold of [`CPUBackend`](@ref): a launch covering fewer
points runs serially even when Julia has several threads, because the
task start-up (a few µs per thread) costs more than the work. 32 768
points is a 32×32×32 block; measured in ka_port.md.
"""
const SERIAL_THRESHOLD = 32_768

"""
    CPUBackend(; serial = false, threshold = SERIAL_THRESHOLD)

The CPU backend: `Base.Array` storage, kernels launched through
`KernelAbstractions.CPU()` on Julia's threads (`julia -t N`).

  * `serial = true` runs every launch on the calling task, in the
    historical `for k, for j, for i` loop order. Pointwise kernels give the
    same bits as the threaded path (each point is computed by the same
    body); reductions use the same canonical order on both paths, so
    `serial = true` is the in-process reference of the thread-invariance
    tests.
  * `threshold`: launches over fewer points run serially (task start-up
    dominates below it).

With one Julia thread every launch runs serially as well, so a `-t 1` run
executes exactly the loops it executed before the port.
"""
struct CPUBackend <: AbstractBackend
    ka::KA.CPU
    serial::Bool
    threshold::Int
end
CPUBackend(; serial::Bool = false, threshold::Integer = SERIAL_THRESHOLD) =
    CPUBackend(KA.CPU(), serial, Int(threshold))

"""
    GPUBackend(kind = :cuda)

GPU backend stub (Phase 81). `kind` is one of `:cuda`, `:rocm`, `:oneapi`,
`:metal`. Using it trips a helpful error rather than silently falling back
to the CPU.
"""
struct GPUBackend <: AbstractBackend
    kind::Symbol
end
GPUBackend() = GPUBackend(:cuda)

backend_name(::CPUBackend) = :cpu
backend_name(b::GPUBackend) = b.kind

"""
    runs_serially(backend, npoints = typemax(Int)) -> Bool

Whether a launch over `npoints` points runs on the calling task: always
for `serial = true`, for one Julia thread, and below the backend's
threshold.
"""
runs_serially(b::CPUBackend, npoints::Integer = typemax(Int)) =
    b.serial || Threads.nthreads() == 1 || npoints < b.threshold

_gpu_error(b::GPUBackend, what) =
    error("$what: GPUBackend($(b.kind)) is a stub until Phase 81 (it needs an extension " *
          "package: CUDA.jl / AMDGPU.jl / oneAPI.jl / Metal.jl).")

# ---- process default -------------------------------------------------------

const _DEFAULT_BACKEND = Ref{AbstractBackend}(CPUBackend())

"""
    default_backend() -> AbstractBackend

The backend new fields are allocated on and plain arrays are launched on:
`CPUBackend()` unless [`set_default_backend!`](@ref) (or
[`init_threads!`](@ref), or [`with_backend`](@ref)) changed it.
"""
default_backend() = _DEFAULT_BACKEND[]

"""
    set_default_backend!(backend) -> backend

Make `backend` the process default ([`default_backend`](@ref)). Global
state: call it at run start, not from inside a threaded region.
"""
set_default_backend!(b::AbstractBackend) = (_DEFAULT_BACKEND[] = b)

"""
    with_backend(f, backend)

Run `f()` with `backend` as the [`default_backend`](@ref), restoring the
previous default afterwards (also on error). Not task-local: do not use it
concurrently from several tasks.
"""
function with_backend(f, b::AbstractBackend)
    old = _DEFAULT_BACKEND[]
    _DEFAULT_BACKEND[] = b
    try
        return f()
    finally
        _DEFAULT_BACKEND[] = old
    end
end

"""
    backend(x) -> AbstractBackend

The backend that owns array `x` (for a `Field`, its storage): for a host
`Array`, the default backend if that is a CPU backend, else `CPUBackend()`.
"""
backend(::Array) = (b = default_backend(); b isa CPUBackend ? b : CPUBackend())
backend(x::SubArray) = backend(parent(x))

# =========================================================================
# Allocation
# =========================================================================

"""
    allocate(backend, T, dims) -> array

A zero-filled array of element type `T` and size `dims` on `backend`
(`Array{T}` for the CPU).
"""
allocate(::CPUBackend, ::Type{T}, dims::NTuple{N,Int}) where {T, N} =
    zeros(T, dims...)
allocate(b::GPUBackend, ::Type{T}, dims::NTuple{N,Int}) where {T, N} =
    _gpu_error(b, "allocate")

"""
    to_host(x) -> Array

Copy `x` to the host. No-op if it already lives on the host.
"""
to_host(x::AbstractArray) = Array(x)

# =========================================================================
# Thread policy
# =========================================================================

# Filled by Operators (the owner of the FFTW policy, #245), so that the
# thread count is set through `set_fftw_policy!` and reported by
# `fftw_policy()`; one source of truth for FFTW settings.
function _set_fftw_threads! end

"""
    init_threads!(backend = default_backend(); fftw_threads = nothing,
                  blas_threads = nothing) -> NamedTuple

Prepare the process to run on `backend`: makes it the
[`default_backend`](@ref) and sets the thread counts of the two libraries
that thread on their own.

  * **BLAS** is pinned to one thread when Julia runs threaded
    (`Threads.nthreads() > 1`; maintainer decision on #294): BLAS threads
    would compete with the kernel tasks, and a threaded BLAS may split a
    reduction differently at different thread counts. `blas_threads`
    overrides.
  * **FFTW**: `fftw_threads = nothing` keeps the FFTW policy
    (`AEOLUS_FFTW_THREADS`, default 1; `Aeolus.Operators.set_fftw_policy!`);
    an integer sets it there. Thread invariance of threaded FFTW plans
    (under `ESTIMATE`) is established by the FFT port (KA-h), not here.

Returns `(backend, julia_threads, blas_threads, fftw_threads)`.
"""
function init_threads!(b::AbstractBackend = default_backend();
                       fftw_threads::Union{Nothing,Integer} = nothing,
                       blas_threads::Union{Nothing,Integer} = nothing)
    set_default_backend!(b)
    nt = Threads.nthreads()
    if blas_threads !== nothing
        BLAS.set_num_threads(Int(blas_threads))
    elseif nt > 1
        BLAS.set_num_threads(1)
    end
    fftw_threads === nothing || _set_fftw_threads!(Int(fftw_threads))
    return (backend = b, julia_threads = nt, blas_threads = BLAS.get_num_threads(),
            fftw_threads = _fftw_threads())
end
# The FFTW thread count as the policy reports it (Operators adds the method).
function _fftw_threads end

# =========================================================================
# launch!: the one way a loop nest over storage indices is run
# =========================================================================

# ---- KA kernels (all of Aeolus' kernels live here) --------------------------
#
# On the CPU a work item is one i-row (3-D) or one i-row of a plane (2-D):
# the inner loop over i is a plain loop the compiler vectorises, which a
# point-per-item kernel on KA's CPU backend does not (measured 5.9x slower
# at 1 thread, ka_port.md). A workgroup of 4 rows is the issue's
# (n_i, min(n_j, 4), 1) block. A GPU backend will launch the same body
# point per item.

@kernel inbounds = true function _rows3_kernel!(body!, i1, i2, oj, ok, args)
    b, c = @index(Global, NTuple)
    j = b + oj
    k = c + ok
    for i in i1:i2
        body!(i, j, k, args...)
    end
end

@kernel inbounds = true function _rows2_kernel!(body!, i1, i2, oj, args)
    b = @index(Global, Linear)
    j = b + oj
    for i in i1:i2
        body!(i, j, args...)
    end
end

@kernel inbounds = true function _chunks1_kernel!(body!, n1, n2, chunk, args)
    c = @index(Global, Linear)
    lo = n1 + (c - 1) * chunk
    hi = min(lo + chunk - 1, n2)
    for n in lo:hi
        body!(n, args...)
    end
end

# A plane (3-D: one k) or a row (2-D: one j) per work item: reductions.
@kernel inbounds = true function _planes_kernel!(out, plane!, ok, args)
    c = @index(Global, Linear)
    out[c] = plane!(c + ok, args...)
end

const _ROWS_PER_GROUP = 4
const _CHUNK_1D = 1024

_npoints(ranges) = prod(length, ranges; init = 1)

# ---- serial reference loops (the historical loop order) ---------------------

@noinline function _serial!(body!::F, (ir, jr, kr)::NTuple{3,UnitRange{Int}}, args) where {F}
    for k in kr, j in jr, i in ir
        body!(i, j, k, args...)
    end
    return nothing
end
@noinline function _serial!(body!::F, (ir, jr)::NTuple{2,UnitRange{Int}}, args) where {F}
    for j in jr, i in ir
        body!(i, j, args...)
    end
    return nothing
end
@noinline function _serial!(body!::F, (nr,)::NTuple{1,UnitRange{Int}}, args) where {F}
    for n in nr
        body!(n, args...)
    end
    return nothing
end

# ---- threaded CPU launches ------------------------------------------------

function _threaded!(b::CPUBackend, body!::F, (ir, jr, kr)::NTuple{3,UnitRange{Int}}, args) where {F}
    nj, nk = length(jr), length(kr)
    kern = _rows3_kernel!(b.ka, (min(nj, _ROWS_PER_GROUP), 1))
    kern(body!, first(ir), last(ir), first(jr) - 1, first(kr) - 1, args; ndrange = (nj, nk))
    return nothing
end
function _threaded!(b::CPUBackend, body!::F, (ir, jr)::NTuple{2,UnitRange{Int}}, args) where {F}
    nj = length(jr)
    kern = _rows2_kernel!(b.ka, min(nj, _ROWS_PER_GROUP))
    kern(body!, first(ir), last(ir), first(jr) - 1, args; ndrange = nj)
    return nothing
end
function _threaded!(b::CPUBackend, body!::F, (nr,)::NTuple{1,UnitRange{Int}}, args) where {F}
    nc = cld(length(nr), _CHUNK_1D)
    kern = _chunks1_kernel!(b.ka, 1)
    kern(body!, first(nr), last(nr), _CHUNK_1D, args; ndrange = nc)
    return nothing
end

"""
    launch!(backend, body!, ranges::NTuple{N,UnitRange{Int}}, args...)

Run `body!(i, j, k, args...)` (N = 3), `body!(i, j, args...)` (N = 2) or
`body!(n, args...)` (N = 1) once for every index of `ranges`, which are
**storage** indices (halos included), so any region is a launch: the
interior `(Hx+1:Hx+Nx, …)`, a face range with one extra point
(`Hx+1:Hx+Nx+1`), an extended range into the halo (`Hx:i2`), a halo
shell, a 2-D `(i, j)` surface range, or a 1-D range of ids (surface
elements, particles).

The contract (docs/src/design/ka_port.md):

  * the body writes only to point `(i, j, k)` of its outputs and reads its
    inputs anywhere — no two points write the same location, no point reads
    what another point writes in the same launch (use a scratch array);
  * the body carries its own `@inbounds` (or is `@propagate_inbounds`);
  * results must not depend on the order points run in. Pointwise bodies
    then give the same bits serially and threaded; reductions go through
    [`plane_partials!`](@ref) and friends, never through a shared
    accumulator.

Serial (`serial = true`, one Julia thread, or fewer points than the
backend's threshold): the loops run on the calling task in the historical
order `for k, for j, for i`. Threaded: a KernelAbstractions kernel whose
work item is one i-row, in workgroups of four rows.
"""
# Specialisation: with a plain `args...`, or with a function among the
# arguments, Julia compiles a pass-through method for widened argument types
# (`Any`, `Function`), and the inlined serial loop then dispatches at every
# point (measured 375x slower than the loop at 64x64x32). `Vararg{Any,M}`
# and the type parameter `A` of `_launch!` prevent that.
launch!(b::CPUBackend, body!::F, ranges::NTuple{N,UnitRange{Int}}, args::Vararg{Any,M}) where {F, N, M} =
    _launch!(b, body!, ranges, args)

# The argument tuple as one parameter `A`, so it is never widened (a function
# among `args` would otherwise be compiled as `::Function` and called by
# dynamic dispatch at every point).
function _launch!(b::CPUBackend, body!::F, ranges::NTuple{N,UnitRange{Int}}, args::A) where {F, N, A<:Tuple}
    np = _npoints(ranges)
    np == 0 && return nothing
    if runs_serially(b, np)
        _serial!(body!, ranges, args)
    else
        _threaded!(b, body!, ranges, args)
    end
    return nothing
end
launch!(b::GPUBackend, body!, ranges::Tuple, args...) = _gpu_error(b, "launch!")

# ---- pointwise map ----------------------------------------------------------

# An argument that is `out` itself (an in-place update, `y = f(y, x)`) is
# replaced by `_OUT` and read through `out`: one pointer for the load and
# the store, which the compiler then vectorises. With two pointers to the
# same array it cannot prove the loop free of dependences and stays scalar
# (measured 1.8x slower than the old in-place loop at 64x64x32).
struct _Out end
const _OUT = _Out()

@inline _at(a::AbstractArray, out, i, j, k) = @inbounds a[i, j, k]
@inline _at(::_Out, out, i, j, k) = @inbounds out[i, j, k]
@inline _at(x, out, i, j, k) = x
@inline _at(a::AbstractArray, out, i, j) = @inbounds a[i, j]
@inline _at(::_Out, out, i, j) = @inbounds out[i, j]
@inline _at(x, out, i, j) = x

@inline function _map_body!(i, j, k, f::F, out, args) where {F}
    @inbounds out[i, j, k] = f(map(a -> _at(a, out, i, j, k), args)...)
    return nothing
end
@inline function _map_body!(i, j, f::F, out, args) where {F}
    @inbounds out[i, j] = f(map(a -> _at(a, out, i, j), args)...)
    return nothing
end

"""
    range_map!(backend, f, out, ranges, args...) -> out

`out[I] = f(args[1][I], args[2][I], …)` for every storage index `I` of
`ranges` (3-D or 2-D); an argument that is not an array (a number) is
passed as is. A pointwise kernel: bit-identical serially and threaded.
`Fields.interior_map!` is the field-level form.
"""
function range_map!(b::AbstractBackend, f::F, out::AbstractArray, ranges::Tuple,
                    args::Vararg{Any,M}) where {F, M}
    _npoints(ranges) == 0 || _check((out, args...), ranges)
    launch!(b, _map_body!, ranges, f, out, map(a -> a === out ? _OUT : a, args))
    return out
end

# =========================================================================
# Deterministic reductions
# =========================================================================
#
# The canonical order (docs/src/design/ka_port.md, "Reduction contract"):
# a plane (fixed k) is reduced row by row (j ascending), each row in eight
# lanes — the point i goes to lane (i - i1) mod 8 for the first 8⌊n/8⌋
# points, the rest of the row to lane 1 in ascending i — and the lanes are
# combined as ((l1 ∘ l2) ∘ (l3 ∘ l4)) ∘ ((l5 ∘ l6) ∘ (l7 ∘ l8)). The plane
# results are then combined in ascending k by `ordered_sum`. Every plane is
# reduced by one work item, so the result is the same at any thread count
# and serially. The eight independent lanes let the compiler use SIMD and
# hide the add latency without reassociating anything.

# The values of every array at one point (inbounds: the caller checked the ranges).
@inline _ld(arrs::Tuple, i, j, k) = map(a -> (@inbounds a[i, j, k]), arrs)

@inline function _plane_reduce(op::O, init::T, f::F, ir::UnitRange{Int}, jr::UnitRange{Int},
                               k::Int, arrs::Tuple) where {O, T, F}
    l1 = l2 = l3 = l4 = l5 = l6 = l7 = l8 = init
    i1 = first(ir)
    n = length(ir)
    nb = n >> 3
    @inbounds for j in jr
        i = i1
        for _ in 1:nb
            l1 = op(l1, T(f(_ld(arrs, i, j, k)...)))
            l2 = op(l2, T(f(_ld(arrs, i + 1, j, k)...)))
            l3 = op(l3, T(f(_ld(arrs, i + 2, j, k)...)))
            l4 = op(l4, T(f(_ld(arrs, i + 3, j, k)...)))
            l5 = op(l5, T(f(_ld(arrs, i + 4, j, k)...)))
            l6 = op(l6, T(f(_ld(arrs, i + 5, j, k)...)))
            l7 = op(l7, T(f(_ld(arrs, i + 6, j, k)...)))
            l8 = op(l8, T(f(_ld(arrs, i + 7, j, k)...)))
            i += 8
        end
        for ii in i:last(ir)
            l1 = op(l1, T(f(_ld(arrs, ii, j, k)...)))
        end
    end
    return op(op(op(l1, l2), op(l3, l4)), op(op(l5, l6), op(l7, l8)))
end

# The per-plane function a work item (or the serial loop) evaluates.
struct _PlaneReduce{O, T, F, R}
    op::O
    init::T
    f::F
    ir::R
    jr::R
end
@inline (p::_PlaneReduce)(k, arrs) = _plane_reduce(p.op, p.init, p.f, p.ir, p.jr, k, arrs)

# Bounds are checked once per launch; the kernels then index @inbounds.
function _check(arrs::Tuple, ranges::Tuple)
    for a in arrs
        a isa AbstractArray && checkbounds(a, ranges...)
    end
    return nothing
end

function _plane_partials!(b::CPUBackend, out::AbstractVector, p::_PlaneReduce,
                          kr::UnitRange{Int}, arrs::Tuple)
    isempty(p.ir) || isempty(p.jr) || isempty(kr) || _check(arrs, (p.ir, p.jr, kr))
    length(out) == length(kr) ||
        throw(DimensionMismatch("partials has length $(length(out)), the k range $(length(kr))"))
    isempty(kr) && return out
    if runs_serially(b, length(p.ir) * length(p.jr) * length(kr)) || length(kr) == 1
        for (c, k) in enumerate(kr)
            out[c] = p(k, arrs)
        end
    else
        kern = _planes_kernel!(b.ka, 1)
        kern(out, p, first(kr) - 1, (arrs,); ndrange = length(kr))
    end
    return out
end
_plane_partials!(b::GPUBackend, args...) = _gpu_error(b, "plane_partials!")

_acc_type(arrs) = float(promote_type(map(eltype, arrs)...))

"""
    plane_partials!(partials, [backend,] f, ranges::NTuple{3,UnitRange{Int}}, arrays...)
        -> partials

`partials[c]` = the canonical sum, over the plane `k = ranges[3][c]` and the
points `(i, j)` of `ranges[1] × ranges[2]`, of `f(arrays[1][i,j,k], …)`.
The order of the additions is fixed (eight lanes per row, rows in order;
see ka_port.md), and each plane is summed by one work item, so `partials`
is identical at every thread count and with `serial = true`. Combine with
[`ordered_sum`](@ref) for a total, or divide by the plane size for a
horizontal-mean profile ([`plane_mean!`](@ref)).
"""
function plane_partials!(partials::AbstractVector, b::AbstractBackend, f::F,
                         ranges::NTuple{3,UnitRange{Int}}, arrays::Vararg{AbstractArray,M}) where {F, M}
    T = eltype(partials)
    _plane_partials!(b, partials, _PlaneReduce(+, zero(T), f, ranges[1], ranges[2]), ranges[3], arrays)
end
plane_partials!(partials::AbstractVector, f::F, ranges::NTuple{3,UnitRange{Int}},
                arrays::Vararg{AbstractArray,M}) where {F, M} =
    plane_partials!(partials, backend(first(arrays)), f, ranges, arrays...)

"""
    ordered_sum(v) -> sum

`v[1] + v[2] + … + v[end]`, strictly left to right (`foldl`). The fixed
combination of the partial sums from [`plane_partials!`](@ref) or
[`column_partials`](@ref); `zero(eltype(v))` for an empty `v`.
"""
function ordered_sum(v::AbstractArray{T}) where {T}
    s = zero(T)
    @inbounds for x in v
        s += x
    end
    return s
end

"""
    interior_sum([backend,] A, ranges::NTuple{3,UnitRange{Int}}) -> sum

Thread-invariant sum of `A` over the storage ranges: the
[`ordered_sum`](@ref) of the [`plane_partials!`](@ref). Its bits differ
from `sum(view(A, ranges...))` (pairwise) — the canonical order is a named
one-time change for the call sites the port moves to it.
"""
interior_sum(b::AbstractBackend, A::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) =
    ordered_sum(plane_partials!(Vector{_acc_type((A,))}(undef, length(ranges[3])), b,
                                identity, ranges, A))
interior_sum(A::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) = interior_sum(backend(A), A, ranges)

"""
    interior_dot([backend,] A, B, ranges::NTuple{3,UnitRange{Int}}) -> Σ A·B

Thread-invariant inner product over the storage ranges, in the canonical
order of [`plane_partials!`](@ref).
"""
interior_dot(b::AbstractBackend, A::AbstractArray, B::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) =
    ordered_sum(plane_partials!(Vector{_acc_type((A, B))}(undef, length(ranges[3])), b,
                                *, ranges, A, B))
interior_dot(A::AbstractArray, B::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) =
    interior_dot(backend(A), A, B, ranges)

"""
    plane_mean!(profile, [backend,] A, ranges::NTuple{3,UnitRange{Int}}) -> profile

Horizontal-mean profile: `profile[c]` = the canonical plane sum of level
`k = ranges[3][c]` ([`plane_partials!`](@ref)) divided by the number of
points of the plane. Thread-invariant.
"""
function plane_mean!(profile::AbstractVector, b::AbstractBackend, A::AbstractArray,
                     ranges::NTuple{3,UnitRange{Int}})
    plane_partials!(profile, b, identity, ranges, A)
    n = length(ranges[1]) * length(ranges[2])
    profile ./= n
    return profile
end
plane_mean!(profile::AbstractVector, A::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) =
    plane_mean!(profile, backend(A), A, ranges)

# ---- column partials: one sum over k per (i, j) ----------------------------

@inline function _column_body!(i, j, out, A, kr, oi, oj)
    s = zero(eltype(out))
    @inbounds for k in kr
        s += A[i, j, k]
    end
    @inbounds out[i - oi, j - oj] = s
    return nothing
end

"""
    column_partials([backend,] A, ranges::NTuple{3,UnitRange{Int}}) -> Matrix

`P[a, b]` = `A[i, j, k1] + A[i, j, k1+1] + …` (ascending k) for the column
`i = ranges[1][a]`, `j = ranges[2][b]`: one partial per column, computed by
one work item, so thread-invariant. [`ordered_sum`](@ref)`(P)` is then a
canonical total in column order (a different order, hence different bits,
from [`interior_sum`](@ref)).
"""
function column_partials(b::AbstractBackend, A::AbstractArray, (ir, jr, kr)::NTuple{3,UnitRange{Int}})
    out = Matrix{_acc_type((A,))}(undef, length(ir), length(jr))
    isempty(ir) || isempty(jr) || isempty(kr) || checkbounds(A, ir, jr, kr)
    launch!(b, _column_body!, (ir, jr), out, A, kr, first(ir) - 1, first(jr) - 1)
    return out
end
column_partials(A::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) = column_partials(backend(A), A, ranges)

# ---- max / all-finite ------------------------------------------------------

"""
    interior_max([backend,] A, ranges::NTuple{3,UnitRange{Int}}) -> maximum

Maximum of `A` over the storage ranges, **NaN-propagating**: any NaN in the
region gives NaN (as `Base.maximum`), so a blown-up field is never hidden
by a finite maximum. Per plane in the canonical lane order, then combined
in ascending k: thread-invariant. `-Inf` for an empty region.
"""
function interior_max(b::AbstractBackend, A::AbstractArray, (ir, jr, kr)::NTuple{3,UnitRange{Int}})
    T = _acc_type((A,))
    parts = _plane_partials!(b, Vector{T}(undef, length(kr)),
                             _PlaneReduce(max, T(-Inf), identity, ir, jr), kr, (A,))
    m = T(-Inf)
    for x in parts
        m = max(m, x)
    end
    return m
end
interior_max(A::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) = interior_max(backend(A), A, ranges)

"""
    interior_all_finite([backend,] A, ranges::NTuple{3,UnitRange{Int}}) -> Bool

`true` iff every value of `A` over the storage ranges is finite (no NaN,
no ±Inf). Thread-invariant.
"""
function interior_all_finite(b::AbstractBackend, A::AbstractArray, (ir, jr, kr)::NTuple{3,UnitRange{Int}})
    parts = _plane_partials!(b, Vector{Bool}(undef, length(kr)),
                             _PlaneReduce(&, true, isfinite, ir, jr), kr, (A,))
    return all(parts)
end
interior_all_finite(A::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) =
    interior_all_finite(backend(A), A, ranges)

end # module
