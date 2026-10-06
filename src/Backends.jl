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
using LinearAlgebra: BLAS, mul!

export AbstractBackend, CPUBackend, GPUBackend, allocate, to_host, backend_name,
       backend, default_backend, set_default_backend!, with_backend, init_threads!,
       runs_serially, launch!, launch_weighted!, range_map!,
       plane_partials!, ordered_sum, interior_sum, interior_dot, plane_mean!,
       column_partials, interior_max, interior_all_finite, plane_reduce!, range_max,
       plane_map!, apply_transform!

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

"`backend_name(backend)`: `:cpu`, or the GPU kind (`:cuda`, `:rocm`, …)."
function backend_name end
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
backend(::Array) = _host_backend()
_host_backend() = (b = default_backend(); b isa CPUBackend ? b : CPUBackend())
backend(x::SubArray) = backend(parent(x))
backend(::AbstractArray) = _host_backend()     # other host arrays (reshapes, BitArrays, …)

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
# `_set_fftw_threads!(n; default = true)` sets `n` only if the count was
# not chosen explicitly (AEOLUS_FFTW_THREADS, `set_fftw_policy!(threads =)`,
# an `[fftw]` namelist table).
function _set_fftw_threads! end

"""
    init_threads!(backend = default_backend(); fftw_threads = nothing,
                  blas_threads = nothing) -> NamedTuple

Prepare the process to run on `backend`: makes it the
[`default_backend`](@ref) and sets the thread counts of the two libraries
that thread on their own. Call it before the simulation (and so its FFT
plans) is built: an FFTW plan keeps the thread count it was made with.

  * **BLAS** is pinned to one thread when Julia runs threaded
    (`Threads.nthreads() > 1`; maintainer decision on #294): BLAS threads
    would compete with the kernel tasks, and a threaded BLAS may split a
    reduction differently at different thread counts. `blas_threads`
    overrides.
  * **FFTW** runs on `Threads.nthreads()` threads (KA-h, #298; maintainer
    decision on #291): FFTW's threaded plans split the independent 1-D
    transforms between threads without changing any of them, so a solve
    gives the same bits at every thread count under a deterministic
    planner (`ESTIMATE`, or imported wisdom); measured in
    docs/src/design/ka_port.md. A count chosen explicitly wins:
    `fftw_threads = n` here, or before the call `AEOLUS_FFTW_THREADS`,
    `Aeolus.Operators.set_fftw_policy!(threads = n)` or a namelist's
    `[fftw] threads`. The count is set through that FFTW policy, so
    `fftw_policy()` and the checkpoint record report it.

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
    if fftw_threads === nothing
        _set_fftw_threads!(nt; default = true)
    else
        _set_fftw_threads!(Int(fftw_threads))
    end
    return (backend = b, julia_threads = nt, blas_threads = BLAS.get_num_threads(),
            fftw_threads = _fftw_threads())
end
# The FFTW thread count as the policy reports it (Operators adds the method).
function _fftw_threads end

# =========================================================================
# Transforms (KA-h, #298)
# =========================================================================

"""
    apply_transform!(backend, out, plan, in) -> out

Execute the precomputed transform `plan` (an FFTW plan on the CPU) from
`in` into `out`. The spectral pressure solvers call every transform
through this method, from host orchestration between kernel launches and
never inside a kernel, so that Phase 81 can give the GPU backend its own
transform library (cuFFT) behind the same call.

On the [`CPUBackend`](@ref) it is `mul!(out, plan, in)`. The plan carries
its own thread count (FFTW's, set at planning time through the FFTW policy;
[`init_threads!`](@ref) makes it `Threads.nthreads()`). FFTW's threaded
plans split the independent 1-D transforms of a multidimensional transform
between the threads without changing any of them, so the result is
identical at every thread count under a deterministic planner (measured
in docs/src/design/ka_port.md).
"""
apply_transform!(::CPUBackend, out, plan, in) = (mul!(out, plan, in); out)
apply_transform!(b::GPUBackend, out, plan, in) = _gpu_error(b, "apply_transform!")

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
    _row!(body!, i1:i2, j, k, args...)
end

# One i-row of a 3-D launch, as its own function that receives the body's
# arguments as separate arguments (KA-d, #299). Inlined into a loop that
# reads them from the `args` tuple, a large body with branches (WENO-5, the
# tensor diffusion fluxes) ran 3-5x slower than the same body in a plain
# loop: LLVM did not hoist the array loads behind the branches, which it
# does for arrays that arrive as function arguments (known dereferenceable).
# One call per row costs nothing measurable (ka_port.md, KA-d).
#
# Every argument has its own type parameter: a `Vararg{Any,N}` would not be
# specialised on a function among the arguments (Julia's heuristic for
# arguments that are only passed on), and the body would then be called by
# dynamic dispatch at every point (the specialisation trap of ka_port.md).
#
# The serial path is the same idea for the whole loop nest (`_nest!`): one
# call per launch, the arguments separate, the loops `for k, for j, for i`
# in one function, so a small stencil vectorises exactly as the old loop
# did (a call per row made the Laplacian 1.3x the old loop at 64x64x32).
for N in 0:24
    as = [Symbol(:a, n) for n in 1:N]
    Ts = [Symbol(:A, n) for n in 1:N]
    sig = [:($(as[n])::$(Ts[n])) for n in 1:N]
    @eval @noinline function _row!(body!::F, ir::UnitRange{Int}, j::Int, k::Int,
                                   $(sig...)) where {F, $(Ts...)}
        for i in ir
            body!(i, j, k, $(as...))
        end
        return nothing
    end
    @eval @noinline function _nest!(body!::F, ir::UnitRange{Int}, jr::UnitRange{Int},
                                    kr::UnitRange{Int}, $(sig...)) where {F, $(Ts...)}
        for k in kr, j in jr, i in ir
            body!(i, j, k, $(as...))
        end
        return nothing
    end
end
# more than 24 arguments: the plain loops
@noinline function _row!(body!::F, ir::UnitRange{Int}, j::Int, k::Int, args::Vararg{Any,N}) where {F, N}
    for i in ir
        body!(i, j, k, args...)
    end
    return nothing
end
@noinline function _nest!(body!::F, ir::UnitRange{Int}, jr::UnitRange{Int}, kr::UnitRange{Int},
                          args::Vararg{Any,N}) where {F, N}
    for k in kr, j in jr, i in ir
        body!(i, j, k, args...)
    end
    return nothing
end

const _ROWS_PER_GROUP = 4
const _CHUNK_1D = 1024

# KA's CPU backend starts min(nthreads, nblocks) tasks. A launch too small
# to keep every thread busy gets bigger workgroups, so fewer blocks and so
# fewer tasks: at least `_MIN_POINTS_PER_TASK` points per task (a task
# start costs a few µs; measured in ka_port.md). Pointwise results do not
# depend on the split.
const _MIN_POINTS_PER_TASK = 16_384
_ntasks(np) = clamp(np ÷ _MIN_POINTS_PER_TASK, 1, Threads.nthreads())

_npoints(ranges) = prod(length, ranges; init = 1)

# ---- serial reference loops (the historical loop order) ---------------------

@noinline function _serial!(body!::F, (ir, jr, kr)::NTuple{3,UnitRange{Int}}, args) where {F}
    _nest!(body!, ir, jr, kr, args...)    # for k in kr, j in jr, i in ir: body!(i, j, k, args...)
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

function _threaded!(b::CPUBackend, body!::F, (ir, jr, kr)::NTuple{3,UnitRange{Int}}, args,
                    weight::Int = 1) where {F}
    nj, nk = length(jr), length(kr)
    nt = _ntasks(length(ir) * nj * nk * weight)
    wg = nt >= Threads.nthreads() ? (min(nj, _ROWS_PER_GROUP), 1) : (nj, cld(nk, nt))
    kern = _rows3_kernel!(b.ka, wg)
    kern(body!, first(ir), last(ir), first(jr) - 1, first(kr) - 1, args; ndrange = (nj, nk))
    return nothing
end
function _threaded!(b::CPUBackend, body!::F, (ir, jr)::NTuple{2,UnitRange{Int}}, args,
                    weight::Int = 1) where {F}
    nj = length(jr)
    nt = _ntasks(length(ir) * nj * weight)
    kern = _rows2_kernel!(b.ka, nt >= Threads.nthreads() ? min(nj, _ROWS_PER_GROUP) : cld(nj, nt))
    kern(body!, first(ir), last(ir), first(jr) - 1, args; ndrange = nj)
    return nothing
end
function _threaded!(b::CPUBackend, body!::F, (nr,)::NTuple{1,UnitRange{Int}}, args,
                    weight::Int = 1) where {F}
    nt = _ntasks(length(nr) * weight)
    chunk = nt >= Threads.nthreads() ? max(1, _CHUNK_1D ÷ weight) : cld(length(nr), nt)
    kern = _chunks1_kernel!(b.ka, 1)
    kern(body!, first(nr), last(nr), chunk, args; ndrange = cld(length(nr), chunk))
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
work item is one i-row, in workgroups of four rows; a launch too small
for every thread gets fewer, larger workgroups (at least 16 384 points
per task).
"""
function launch! end

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
function _launch!(b::CPUBackend, body!::F, ranges::NTuple{N,UnitRange{Int}}, args::A,
                  weight::Int = 1) where {F, N, A<:Tuple}
    np = _npoints(ranges)
    np == 0 && return nothing
    if runs_serially(b, np * weight)
        _serial!(body!, ranges, args)
    else
        _threaded!(b, body!, ranges, args, weight)
    end
    return nothing
end

"""
    launch_weighted!(backend, weight, body!, ranges, args...)

[`launch!`](@ref) for a body that covers `weight` points per index: a
column kernel launched over `(i, j)` (or over rows `j`) that loops over
`weight` levels itself, such as the tridiagonal solve of
`StretchedPoissonSolver` (KA-h, #298). The backend's serial threshold and
its points-per-task rule then count `weight × length(ranges)` points, the
work actually done, instead of the number of indices. Same contract and
same serial order as `launch!`; `weight = 1` is `launch!`.
"""
launch_weighted!(b::CPUBackend, weight::Integer, body!::F, ranges::NTuple{N,UnitRange{Int}},
                 args::Vararg{Any,M}) where {F, N, M} =
    _launch!(b, body!, ranges, args, max(1, Int(weight)))
launch_weighted!(b::GPUBackend, weight, body!, ranges::Tuple, args...) =
    _gpu_error(b, "launch_weighted!")
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

# Point evaluators: what a reduction adds up at the point (i, j, k).
#   _Vals: f of the values of the arrays there (plane_partials!, interior_*)
#   _Ix:   an indexed body f(i, j, k, args...) (plane_reduce!, range_max), which
#          may read a stencil and write point (i, j, k) (the launch contract).
struct _Vals{F, A}
    f::F
    arrs::A
end
@inline (v::_Vals)(i, j, k) = v.f(_ld(v.arrs, i, j, k)...)
struct _Ix{F, A}
    f::F
    args::A
end
@inline (x::_Ix)(i, j, k) = x.f(i, j, k, x.args...)

# The sum of the reductions: `+`, and component-wise for the tuple-valued
# bodies of `plane_reduce!` (each component in the same canonical order).
@inline _add(a, b) = a + b
@inline _add(a::Tuple, b::Tuple) = map(+, a, b)

@inline function _plane_reduce(op::O, init::T, g::G, ir::UnitRange{Int}, jr::UnitRange{Int},
                               k::Int) where {O, T, G}
    l1 = l2 = l3 = l4 = l5 = l6 = l7 = l8 = init
    i1 = first(ir)
    n = length(ir)
    nb = n >> 3
    @inbounds for j in jr
        i = i1
        for _ in 1:nb
            l1 = op(l1, T(g(i, j, k)))
            l2 = op(l2, T(g(i + 1, j, k)))
            l3 = op(l3, T(g(i + 2, j, k)))
            l4 = op(l4, T(g(i + 3, j, k)))
            l5 = op(l5, T(g(i + 4, j, k)))
            l6 = op(l6, T(g(i + 5, j, k)))
            l7 = op(l7, T(g(i + 6, j, k)))
            l8 = op(l8, T(g(i + 7, j, k)))
            i += 8
        end
        for ii in i:last(ir)
            l1 = op(l1, T(g(ii, j, k)))
        end
    end
    return op(op(op(l1, l2), op(l3, l4)), op(op(l5, l6), op(l7, l8)))
end

# ---- the same order, explicitly vectorised (Float64 sums) -------------------
#
# The eight lanes of the canonical order are the eight elements of one
# `<8 x double>` vector: point `i1 + 8b + r` of a row goes to element `r`
# (`r = 0…7`), which is lane `(i − i1) mod 8 + 1`. The vector add is eight
# independent IEEE additions, so the bits are those of the scalar lanes
# above (`test/backends.jl` checks both against a transcript), but the
# compiler issues one vector instruction where it does not vectorise eight
# scalar accumulators on its own (measured 2.4x faster at 64³, below).
# A tuple-valued body (`plane_reduce!`) gets one vector per component.

const _V8 = NTuple{8,VecElement{Float64}}
@inline _vadd(a::_V8, b::_V8) =
    Base.llvmcall("%3 = fadd <8 x double> %0, %1\nret <8 x double> %3", _V8, Tuple{_V8,_V8}, a, b)
@inline _v8(x1, x2, x3, x4, x5, x6, x7, x8) =
    (VecElement(x1), VecElement(x2), VecElement(x3), VecElement(x4),
     VecElement(x5), VecElement(x6), VecElement(x7), VecElement(x8))
@inline _vfill(x::Float64) = _v8(x, x, x, x, x, x, x, x)
@inline function _vfold(a::_V8)
    l1, l2, l3, l4, l5, l6, l7, l8 = map(x -> x.value, a)
    return ((l1 + l2) + (l3 + l4)) + ((l5 + l6) + (l7 + l8))
end
@inline _vlane1add(a::_V8, x::Float64) = Base.setindex(a, VecElement(a[1].value + x), 1)

# component n of eight tuples, as one vector; and the step of N accumulators
@inline _vcomp(n, t1, t2, t3, t4, t5, t6, t7, t8) =
    _v8(t1[n], t2[n], t3[n], t4[n], t5[n], t6[n], t7[n], t8[n])
@inline _vstep(acc::NTuple{N,_V8}, t1, t2, t3, t4, t5, t6, t7, t8) where {N} =
    ntuple(n -> _vadd(acc[n], _vcomp(n, t1, t2, t3, t4, t5, t6, t7, t8)), Val(N))
@inline _vtail(acc::NTuple{N,_V8}, t) where {N} = ntuple(n -> _vlane1add(acc[n], t[n]), Val(N))

@inline function _plane_reduce(::typeof(_add), init::Float64, g::G, ir::UnitRange{Int},
                               jr::UnitRange{Int}, k::Int) where {G}
    acc = _vfill(init)                  # every lane starts at init, as the scalar lanes
    i1 = first(ir)
    nb = length(ir) >> 3
    @inbounds for j in jr
        i = i1
        for _ in 1:nb
            acc = _vadd(acc, _v8(Float64(g(i, j, k)), Float64(g(i + 1, j, k)),
                                 Float64(g(i + 2, j, k)), Float64(g(i + 3, j, k)),
                                 Float64(g(i + 4, j, k)), Float64(g(i + 5, j, k)),
                                 Float64(g(i + 6, j, k)), Float64(g(i + 7, j, k))))
            i += 8
        end
        if i <= last(ir)
            l1 = acc[1].value
            for ii in i:last(ir)
                l1 += Float64(g(ii, j, k))
            end
            acc = Base.setindex(acc, VecElement(l1), 1)
        end
    end
    return _vfold(acc)
end

@inline function _plane_reduce(::typeof(_add), init::T, g::G, ir::UnitRange{Int},
                               jr::UnitRange{Int}, k::Int) where {N, T<:NTuple{N,Float64}, G}
    acc = ntuple(n -> _vfill(init[n]), Val(N))
    i1 = first(ir)
    nb = length(ir) >> 3
    @inbounds for j in jr
        i = i1
        for _ in 1:nb
            acc = _vstep(acc, T(g(i, j, k)), T(g(i + 1, j, k)), T(g(i + 2, j, k)), T(g(i + 3, j, k)),
                         T(g(i + 4, j, k)), T(g(i + 5, j, k)), T(g(i + 6, j, k)), T(g(i + 7, j, k)))
            i += 8
        end
        for ii in i:last(ir)
            acc = _vtail(acc, T(g(ii, j, k)))
        end
    end
    return map(_vfold, acc)
end

# max is exact and commutative: any order gives the same bits, so a plane is
# scanned in a plain loop (NaN-propagating `max`), as the loops it replaces
struct _PlaneMax{T, R}
    init::T
    ir::R
    jr::R
end
@inline function (p::_PlaneMax{T})(k, g) where {T}
    m = p.init
    @inbounds for j in p.jr, i in p.ir
        m = max(m, T(g(i, j, k)))
    end
    return m
end

# The per-plane function a work item (or the serial loop) evaluates; the
# point evaluator `g` is its argument.
struct _PlaneReduce{O, T, R}
    op::O
    init::T
    ir::R
    jr::R
end
@inline (p::_PlaneReduce)(k, g) = _plane_reduce(p.op, p.init, g, p.ir, p.jr, k)

# Bounds are checked once per launch; the kernels then index @inbounds.
function _check(arrs::Tuple, ranges::Tuple)
    for a in arrs
        a isa AbstractArray && checkbounds(a, ranges...)
    end
    return nothing
end
_checkr(arrs::Tuple, (ir, jr, kr)::NTuple{3,UnitRange{Int}}) =
    isempty(ir) || isempty(jr) || isempty(kr) || _check(arrs, (ir, jr, kr))

# A whole-plane function `f(k, ir, jr, args...)` (plane_map!): the plane's
# own loop, in its own fixed order.
struct _PlaneFn{F, R}
    f::F
    ir::R
    jr::R
end
@inline (p::_PlaneFn)(k, args) = p.f(k, p.ir, p.jr, args...)

function _plane_partials!(b::CPUBackend, out::AbstractVector, p::Union{_PlaneReduce,_PlaneMax,_PlaneFn},
                          kr::UnitRange{Int}, g)
    length(out) == length(kr) ||
        throw(DimensionMismatch("partials has length $(length(out)), the k range $(length(kr))"))
    isempty(kr) && return out
    if runs_serially(b, length(p.ir) * length(p.jr) * length(kr)) || length(kr) == 1
        for (c, k) in enumerate(kr)
            out[c] = p(k, g)
        end
    else
        nk = length(kr)
        nt = _ntasks(length(p.ir) * length(p.jr) * nk)
        kern = _planes_kernel!(b.ka, nt >= Threads.nthreads() ? 1 : cld(nk, nt))
        kern(out, p, first(kr) - 1, (g,); ndrange = length(kr))
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
    _checkr(arrays, ranges)
    _plane_partials!(b, partials, _PlaneReduce(_add, zero(T), ranges[1], ranges[2]), ranges[3],
                     _Vals(f, arrays))
end
plane_partials!(partials::AbstractVector, f::F, ranges::NTuple{3,UnitRange{Int}},
                arrays::Vararg{AbstractArray,M}) where {F, M} =
    plane_partials!(partials, backend(first(arrays)), f, ranges, arrays...)

plane_partials!(partials::AbstractVector, b::AbstractBackend, f::F,
                ranges::NTuple{3,AbstractUnitRange{<:Integer}}, arrays::Vararg{AbstractArray,M}) where {F, M} =
    plane_partials!(partials, b, f, map(UnitRange{Int}, ranges), arrays...)
plane_partials!(partials::AbstractVector, f::F, ranges::NTuple{3,AbstractUnitRange{<:Integer}},
                arrays::Vararg{AbstractArray,M}) where {F, M} =
    plane_partials!(partials, backend(first(arrays)), f, map(UnitRange{Int}, ranges), arrays...)

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

# ---- indexed reductions: a body per point, as launch! ----------------------

"""
    plane_reduce!(partials, [backend,] body, ranges::NTuple{3,UnitRange{Int}}, args...)
        -> partials

The canonical plane sums of an *indexed* body: `partials[c]` = the sum,
over the plane `k = ranges[3][c]` and the points `(i, j)` of
`ranges[1] × ranges[2]`, of `body(i, j, k, args...)`, added in the order
of [`plane_partials!`](@ref) (eight lanes per row, rows in order, one work
item per plane). Thread-invariant, and the same bits as `plane_partials!`
for a body that returns `f(A[i,j,k], …)`.

The body follows the [`launch!`](@ref) contract: it may read its inputs
anywhere (a stencil: a strain rate, a centred `w`), may **write point
`(i, j, k)`** of its outputs (an update fused with the sum of what it
changed: a chemistry step returning its production), carries its own
`@inbounds`, and takes everything it reads as an argument. A per-level
coefficient is an argument indexed by `k`.

A body that returns an `NTuple` of `N` numbers sums every component in
the same canonical order: pass `partials` as an `NTuple` of `N` vectors
(`plane_reduce!((S, n), b, body, r, …)` for a masked sum and its count).

`ranges` are whatever indices the body takes — storage indices of
halo-padded arrays, or `axes(view)` of interior-sized arrays.
"""
function plane_reduce!(partials::AbstractVector, b::AbstractBackend, body::F,
                       ranges::NTuple{3,UnitRange{Int}}, args::Vararg{Any,M}) where {F, M}
    T = eltype(partials)
    _plane_partials!(b, partials, _PlaneReduce(_add, zero(T), ranges[1], ranges[2]), ranges[3],
                     _Ix(body, args))
end
function plane_reduce!(partials::NTuple{N,AbstractVector}, b::AbstractBackend, body::F,
                       ranges::NTuple{3,UnitRange{Int}}, args::Vararg{Any,M}) where {N, F, M}
    T = Tuple{map(eltype, partials)...}
    tmp = Vector{T}(undef, length(ranges[3]))
    _plane_partials!(b, tmp, _PlaneReduce(_add, map(zero, fieldtypes(T)), ranges[1], ranges[2]),
                     ranges[3], _Ix(body, args))
    for (c, t) in enumerate(tmp), n in 1:N
        partials[n][c] = t[n]
    end
    return partials
end
plane_reduce!(partials::Union{AbstractVector,Tuple}, body::F, ranges::NTuple{3,AbstractUnitRange{<:Integer}},
              args::Vararg{Any,M}) where {F, M} =
    plane_reduce!(partials, _host_backend(), body, ranges, args...)

# `axes(view)` gives `Base.OneTo`s: any unit ranges are accepted
plane_reduce!(partials::Union{AbstractVector,Tuple}, b::AbstractBackend, body::F,
              ranges::NTuple{3,AbstractUnitRange{<:Integer}}, args::Vararg{Any,M}) where {F, M} =
    plane_reduce!(partials, b, body, map(UnitRange{Int}, ranges), args...)

"""
    range_max([backend,] body, ranges::NTuple{3,UnitRange{Int}}, args...) -> maximum

The maximum of the indexed `body(i, j, k, args...)` (indexed as in
[`plane_reduce!`](@ref)) over `ranges`, **NaN-propagating** like
[`interior_max`](@ref): any NaN gives NaN. `max` is exact and
commutative, so the result equals any serial maximum of the same values
(a CFL rate, `max |∇·u|`, a speed cap).

Unlike a `plane_reduce!` body, a `range_max` body must **only read**: it
is evaluated once per point, but a fused update belongs in
`plane_reduce!`. The result type is the inferred (floating-point) return
type of `body`, `Float64` when inference gives no concrete real type;
an empty region gives `-Inf`.
"""
function range_max(b::AbstractBackend, body::F, (ir, jr, kr)::NTuple{3,UnitRange{Int}},
                   args::Vararg{Any,M}) where {F, M}
    (isempty(ir) || isempty(jr) || isempty(kr)) && return -Inf
    T = _max_type(Base.promote_op(body, Int, Int, Int, map(typeof, args)...))
    parts = _plane_partials!(b, Vector{T}(undef, length(kr)),
                             _PlaneMax(T(-Inf), ir, jr), kr, _Ix(body, args))
    m = T(-Inf)
    for x in parts
        m = max(m, x)
    end
    return m
end
# the body's type without evaluating it (it is never called twice at a point)
_max_type(::Type{R}) where {R<:Real} = isconcretetype(R) ? float(R) : Float64
_max_type(::Type) = Float64

range_max(b::AbstractBackend, body::F, ranges::NTuple{3,AbstractUnitRange{<:Integer}},
          args::Vararg{Any,M}) where {F, M} =
    range_max(b, body, map(UnitRange{Int}, ranges), args...)
range_max(body::F, ranges::NTuple{3,AbstractUnitRange{<:Integer}}, args::Vararg{Any,M}) where {F, M} =
    range_max(_host_backend(), body, ranges, args...)

"""
    plane_map!(partials, [backend,] plane, ranges::NTuple{3,UnitRange{Int}}, args...)
        -> partials

`partials[c] = plane(k, ranges[1], ranges[2], args...)` for
`k = ranges[3][c]`: a function of a whole plane, one work item per plane
(the plane parallelism of [`plane_reduce!`](@ref)), so every partial is
computed by one task in the plane function's own fixed loop order and
`partials` is identical at every thread count and with `serial = true`.
Combine with [`ordered_sum`](@ref).

For the reductions where the eight-lane order of `plane_reduce!` costs
more than it gains: a fused update that touches few points and calls an
expensive function there (`apply_wet_deposition!`: `exp` in the raining
cells only), where keeping eight lanes live across the call is slower
than one running sum in cell order (measured in ka_port.md). The plane
function carries its own `@inbounds` and may write any point of its own
plane `k`, and only of it.
"""
function plane_map!(partials::AbstractVector, b::AbstractBackend, plane::F,
                    ranges::NTuple{3,UnitRange{Int}}, args::Vararg{Any,M}) where {F, M}
    _plane_partials!(b, partials, _PlaneFn(plane, ranges[1], ranges[2]), ranges[3], args)
end
plane_map!(partials::AbstractVector, b::AbstractBackend, plane::F,
           ranges::NTuple{3,AbstractUnitRange{<:Integer}}, args::Vararg{Any,M}) where {F, M} =
    plane_map!(partials, b, plane, map(UnitRange{Int}, ranges), args...)
plane_map!(partials::AbstractVector, plane::F, ranges::NTuple{3,AbstractUnitRange{<:Integer}},
           args::Vararg{Any,M}) where {F, M} =
    plane_map!(partials, _host_backend(), plane, map(UnitRange{Int}, ranges), args...)

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
    _checkr((A,), (ir, jr, kr))
    parts = _plane_partials!(b, Vector{T}(undef, length(kr)),
                             _PlaneMax(T(-Inf), ir, jr), kr, _Vals(identity, (A,)))
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
    _checkr((A,), (ir, jr, kr))
    parts = _plane_partials!(b, Vector{Bool}(undef, length(kr)),
                             _PlaneReduce(&, true, ir, jr), kr, _Vals(isfinite, (A,)))
    return all(parts)
end
interior_all_finite(A::AbstractArray, ranges::NTuple{3,UnitRange{Int}}) =
    interior_all_finite(backend(A), A, ranges)

end # module
