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
import Random
using Random: AbstractRNG

export AbstractBackend, CPUBackend, GPUBackend, allocate, to_host, backend_name,
       backend, default_backend, set_default_backend!, with_backend, init_threads!,
       dense_blas_threads, with_dense_blas,
       runs_serially, launch!, launch_weighted!, range_map!,
       plane_partials!, ordered_sum, interior_sum, interior_dot, plane_mean!,
       column_partials, interior_max, interior_all_finite, plane_reduce!, range_max,
       plane_map!, apply_transform!,
       pcg!, PCGWorkspace, PointOperator, PointPreconditioner, IdentityPreconditioner,
       CounterRNG, counter_rng, counter_xoshiro, set_stream!,
       foreach_column!, foreach_cell!, column_sums, column_sum

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
(`Array{T}` for the CPU). A 3-D array of more points than the backend
runs serially is zeroed by a [`launch!`](@ref) over its indices, so each
thread first touches the pages of about the slab the kernels later give it:
the memory is placed by the threads that use it (first-touch NUMA placement
on a multi-socket node) and zeroing a large field is not one thread's
work (#567).
"""
function allocate(b::CPUBackend, ::Type{T}, dims::NTuple{N,Int}) where {T, N}
    (N == 3 && isbitstype(T) && !runs_serially(b, prod(dims))) || return zeros(T, dims...)
    a = Array{T}(undef, dims)
    launch!(b, _zero_body!, (1:dims[1], 1:dims[2], 1:dims[3]), a)
    return a
end
@inline function _zero_body!(i, j, k, a)
    @inbounds a[i, j, k] = zero(eltype(a))
    return nothing
end
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
    (`Threads.nthreads() > 1`; #294): idle BLAS threads would compete with
    the kernel tasks. The dense solves, which run while the kernels wait
    (the radiosity LU), take [`dense_blas_threads`](@ref) for their duration
    ([`with_dense_blas`](@ref)). `blas_threads` sets both.
  * **FFTW** runs on `Threads.nthreads()` threads (KA-h, #298; maintainer
    decision on #291): FFTW's threaded plans usually split the
    independent 1-D transforms between threads (the planner may also
    thread inside one transform, #572), and a solve gave the same bits at
    every thread count under a deterministic planner (`ESTIMATE`, or
    imported wisdom) where measured, docs/src/design/ka_port.md. A count chosen explicitly wins:
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
    _blas_default!()
    if blas_threads !== nothing
        BLAS.set_num_threads(Int(blas_threads))
        _DENSE_BLAS[] = Int(blas_threads)
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
# The BLAS thread count OpenBLAS starts with (OPENBLAS_NUM_THREADS, else its
# physical-core count), read once before anything changes it.
const _BLAS_DEFAULT = Ref(0)
const _DENSE_BLAS = Ref(0)
_blas_default!() = (_BLAS_DEFAULT[] == 0 && (_BLAS_DEFAULT[] = BLAS.get_num_threads()); _BLAS_DEFAULT[])
__init__() = (_blas_default!(); nothing)

"""
    dense_blas_threads() -> Int

The BLAS thread count of the dense LAPACK solves (the radiosity LU and its
solves, `with_dense_blas`): `blas_threads` of [`init_threads!`](@ref) if it
was given, else the count OpenBLAS starts with (`OPENBLAS_NUM_THREADS`, else
its physical-core count). They run while the kernel threads wait, so they
use the cores (#496). Measured, the LU of the Berlin scene's 5639 facets:
1.670 s on 1 thread, 0.453 s on 4, 0.204 s on 12 (OpenBLAS's default on the
12-core, 24-thread development machine) and 0.986 s on 24 (hyperthreads).
"""
dense_blas_threads() = _DENSE_BLAS[] > 0 ? _DENSE_BLAS[] : _blas_default!()

"""
    with_dense_blas(f) -> f()

Run `f` (a dense LAPACK solve) with BLAS on [`dense_blas_threads`](@ref)
threads, then restore the count it had.
"""
function with_dense_blas(f)
    n0 = BLAS.get_num_threads()
    n = dense_blas_threads()
    n == n0 && return f()
    BLAS.set_num_threads(n)
    try
        return f()
    finally
        BLAS.set_num_threads(n0)
    end
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
plans usually split the independent 1-D transforms of a multidimensional
transform between the threads; the planner may also thread inside one
transform, which made the estimate plan of a 150-point z DCT serialise
over 10⁶ columns (#572). Under a deterministic planner the result was
identical at every thread count where measured (docs/src/design/ka_port.md).
"""
apply_transform!(::CPUBackend, out, plan, in) = (mul!(out, plan, in); out)

"""
    synchronize(backend)

Wait until every launch issued on `backend` has finished. A launch on the
[`CPUBackend`](@ref) returns only when its kernel is done, so this is a
no-op there; a GPU backend (Phase 81) queues its kernels, and the host must
synchronize before it reads what they wrote — e.g. `step_adaptive!` before
a retry reads the state that `restore_state!` copied back (KA-j, #306).
"""
synchronize(b::CPUBackend) = KA.synchronize(b.ka)
synchronize(b::GPUBackend) = _gpu_error(b, "synchronize")
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
    # the 2-D serial nest (KA-e, #302: boundary_scalar_flux's face bodies ran
    # 2-2.5x the old loop when the 2-D path read their arguments from the tuple)
    @eval @noinline function _nest2!(body!::F, ir::UnitRange{Int}, jr::UnitRange{Int},
                                     $(sig...)) where {F, $(Ts...)}
        for j in jr, i in ir
            body!(i, j, $(as...))
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
@noinline function _nest2!(body!::F, ir::UnitRange{Int}, jr::UnitRange{Int},
                           args::Vararg{Any,N}) where {F, N}
    for j in jr, i in ir
        body!(i, j, args...)
    end
    return nothing
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
    _nest2!(body!, ir, jr, args...)       # for j in jr, i in ir: body!(i, j, args...)
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
#
# Only the *first* argument is substituted, by a branch on `args[1] === out`
# whose two arms both have concrete argument types (KA-j, #306). Substituting
# any position by `map(a -> a === out ? _OUT : a, args)` gave a tuple of
# `Union`s, so every launch was a dynamic dispatch that allocated 128-256 B
# (a step of the Boussinesq core allocated 1.8 kB). An alias in a later
# position is read through its own pointer: the same values (the body reads
# point (i, j, k) before writing it), only not vectorised.
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
`Fields.interior_map!` is the field-level form. For an in-place update
pass `out` as the *first* argument (`range_map!(b, f, y, r, y, x)`): it is
then read through `out`, which keeps the loop vectorisable.
"""
function range_map!(b::AbstractBackend, f::F, out::AbstractArray, ranges::Tuple,
                    args::Vararg{Any,M}) where {F, M}
    _npoints(ranges) == 0 || _check((out, args...), ranges)
    if M >= 1 && args[1] === out
        launch!(b, _map_body!, ranges, f, out, (_OUT, Base.tail(args)...))
    else
        launch!(b, _map_body!, ranges, f, out, args)
    end
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
# scanned in a plain loop (NaN-propagating `max`), as the loops it replaces.
# A tuple-valued body (`range_max` of several maxima in one sweep, KA-j #306)
# takes the maximum of every component separately.
@inline _max(a, b) = max(a, b)
@inline _max(a::Tuple, b::Tuple) = map(max, a, b)

struct _PlaneMax{T, R}
    init::T
    ir::R
    jr::R
end
@inline function (p::_PlaneMax{T})(k, g) where {T}
    m = p.init
    @inbounds for j in p.jr, i in p.ir
        m = _max(m, convert(T, g(i, j, k)))
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

A body that returns an `NTuple` of real numbers gives the tuple of the
component-wise maxima, each NaN-propagating on its own: several maxima in
one sweep over the region (the fused CFL and step-guard kernels of KA-j,
#306). A component that is `x * 0` turns the maximum into a finiteness
test: `0` if every value was finite, `NaN` otherwise.

Unlike a `plane_reduce!` body, a `range_max` body must **only read**: it
is evaluated once per point, but a fused update belongs in
`plane_reduce!`. The result type is the inferred (floating-point) return
type of `body`, `Float64` when inference gives no concrete real type;
an empty region gives `-Inf` (in every component).
"""
function range_max(b::AbstractBackend, body::F, (ir, jr, kr)::NTuple{3,UnitRange{Int}},
                   args::Vararg{Any,M}) where {F, M}
    T = _max_type(Base.promote_op(body, Int, Int, Int, map(typeof, args)...))
    (isempty(ir) || isempty(jr) || isempty(kr)) && return _neg_inf(T)
    parts = _plane_partials!(b, Vector{T}(undef, length(kr)),
                             _PlaneMax(_neg_inf(T), ir, jr), kr, _Ix(body, args))
    m = _neg_inf(T)
    for x in parts
        m = _max(m, x)
    end
    return m
end
# the body's type without evaluating it (it is never called twice at a point)
_max_type(::Type{R}) where {R<:Real} = isconcretetype(R) ? float(R) : Float64
_max_type(::Type{R}) where {N, R<:NTuple{N,Real}} =
    isconcretetype(R) ? Tuple{map(float, fieldtypes(R))...} : NTuple{N,Float64}
_max_type(::Type) = Float64
_neg_inf(::Type{T}) where {T<:Real} = T(-Inf)
_neg_inf(::Type{T}) where {T<:Tuple} = map(S -> S(-Inf), fieldtypes(T))

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

# =========================================================================
# pcg!: the one preconditioned conjugate-gradient driver (KA-g, #300)
# =========================================================================
#
# Every CG solver of Aeolus (PressureSolvers.CGPoissonSolver,
# Multigrid.MGPreconditionedCG, Anelastic.DensityWeightedPoissonSolver and
# the implicit diffusion of TimeIntegration) runs this loop; they differ
# only in the operator, the preconditioner, the null-space projection and
# the pinned rows. Per iteration it makes three sweeps over the region and
# two canonical reductions:
#
#   1. Ap = A p, fused with p·Ap                    (operator sweep + reduction)
#   2. x += αp, r -= αAp, z = M⁻¹r, fused with
#      (r·r, r·z, Σz, Σr)                           (update sweep + reduction)
#   3. p = (z − m) + βp                             (pointwise)
#
# where m is the mean of z when the null space is projected out (else 0).
# Every reduction goes through the plane partials of `plane_reduce!`, so a
# solve gives the same bits at every thread count, iteration count included.

"""
    PointOperator(f, args...; prepare = nothing)

A matrix-free operator for [`pcg!`](@ref) given point by point:
`(A p)[i, j, k] = f(i, j, k, p, args...)`. `f` reads `p` anywhere (a
stencil) and carries its own `@inbounds`. `prepare(p)`, if given, runs
before each application (a halo fill of the array the operator reads).
`pcg!` evaluates it inside the reduction of `p·Ap`, so the operator and
the dot product are one sweep.
"""
struct PointOperator{F, A, P}
    f::F
    args::A
    prepare::P
    PointOperator(f::F, args...; prepare::P = nothing) where {F, P} =
        new{F, typeof(args), P}(f, args, prepare)
end

"""
    PointPreconditioner(f, args...)

A pointwise preconditioner for [`pcg!`](@ref): `z[i, j, k] =
f(r[i, j, k], i, j, k, args...)` (Jacobi: `r / diag`). Fused into the
residual update of every iteration.
"""
struct PointPreconditioner{F, A}
    f::F
    args::A
    PointPreconditioner(f::F, args...) where {F} = new{F, typeof(args)}(f, args)
end

"`IdentityPreconditioner()`: `z = r` (unpreconditioned CG) for [`pcg!`](@ref)."
struct IdentityPreconditioner end

@inline _pc(::IdentityPreconditioner, rv, i, j, k) = rv
@inline _pc(M::PointPreconditioner, rv, i, j, k) = M.f(rv, i, j, k, M.args...)

# Pinned rows (rows removed from the system): `nothing`, or the storage
# planes `(ic, jc, kc)` (0 = none on that axis), as TimeIntegration's
# constraint planes. The sweeps compute every point and then zero the
# pinned points of their plane: a test per point would stop the row loops
# from vectorising (measured 4x slower, ka_port.md).
@inline _pinned(::Nothing, i, j, k) = false
@inline _pinned(c::NTuple{3,Int}, i, j, k) = i == c[1] || j == c[2] || k == c[3]

@inline _zero_pinned!(a, ::Nothing, ir, jr, k) = nothing
@inline function _zero_pinned!(a, c::NTuple{3,Int}, ir, jr, k)
    T = eltype(a)
    @inbounds if k == c[3]
        for j in jr, i in ir
            a[i, j, k] = zero(T)
        end
    else
        if c[2] in jr
            for i in ir
                a[i, c[2], k] = zero(T)
            end
        end
        if c[1] in ir
            for j in jr
                a[c[1], j, k] = zero(T)
            end
        end
    end
    return nothing
end

"""
    PCGWorkspace(x; ranges = axes(x))
    PCGWorkspace(r, z, p, Ap; ranges)

Preallocated scratch of [`pcg!`](@ref): the residual `r`, the
preconditioned residual `z`, the search direction `p`, the operator image
`Ap` (arrays of the shape of the unknown) and the per-plane partial sums
of the fused reductions. One per solver, so a solve allocates nothing
(on one thread; a threaded launch allocates KernelAbstractions' tasks).
"""
struct PCGWorkspace{T, A<:AbstractArray{T,3}}
    r::A
    z::A
    p::A
    Ap::A
    s1::Vector{T}
    s4::Vector{NTuple{4,T}}
    s5::Vector{NTuple{5,T}}
end
function PCGWorkspace(r::A, z::A, p::A, Ap::A; ranges) where {T, A<:AbstractArray{T,3}}
    nk = length(ranges[3])
    return PCGWorkspace{T,A}(r, z, p, Ap, zeros(T, nk), fill(ntuple(_ -> zero(T), 4), nk),
                             fill(ntuple(_ -> zero(T), 5), nk))
end
PCGWorkspace(x::AbstractArray{T,3}; ranges = axes(x)) where {T} =
    PCGWorkspace(zero(x), zero(x), zero(x), zero(x); ranges)

# canonical sums of the plane partials of a body, into the workspace buffer
@inline _zero_of(::Type{T}) where {T<:Number} = zero(T)
@inline _zero_of(::Type{T}) where {T<:Tuple} = map(zero, fieldtypes(T))
function _pcg_reduce!(buf::Vector{T}, b::CPUBackend, ranges, body::F, args::Vararg{Any,M}) where {T, F, M}
    _plane_partials!(b, buf, _PlaneReduce(_add, _zero_of(T), ranges[1], ranges[2]), ranges[3],
                     _Ix(body, args))
    s = _zero_of(T)
    @inbounds for x in buf
        s = _add(s, x)
    end
    return s
end

# ---- bodies (each writes only its own point) -------------------------------

# Ap = A p over the plane k in a plain row loop (the operator's index
# arithmetic is then hoisted out of the i loop), followed by the canonical
# lanes of p·Ap over the plane while it is in cache: the same bits as a
# fused `plane_reduce!` body, 1.1x faster at 64x64x32 (ka_port.md).
@inline function _pcg_op_plane(k, ir, jr, Ap, p, f::F, args, c) where {F}
    @inbounds for j in jr, i in ir
        Ap[i, j, k] = f(i, j, k, p, args...)
    end
    _zero_pinned!(Ap, c, ir, jr, k)
    return _plane_reduce(_add, zero(eltype(Ap)), _Vals(*, (p, Ap)), ir, jr, k)
end
@inline _pcg_dot_body(i, j, k, a, b) = @inbounds a[i, j, k] * b[i, j, k]
@inline _pcg_val_body(i, j, k, a) = @inbounds a[i, j, k]

# the initial residual r = (b − mb) − Ap (or r = b − mb for a zero guess),
# z = M⁻¹r; returns ((b − mb)², r², r·z, z, r)
@inline function _pcg_init_body(i, j, k, r, z, b, Ap, mb, M, c, zero_guess::Bool)
    @inbounds begin
        T = eltype(r)
        if _pinned(c, i, j, k)
            r[i, j, k] = zero(T); z[i, j, k] = zero(T)
            return (zero(T), zero(T), zero(T), zero(T), zero(T))
        end
        bv = b[i, j, k] - mb
        rv = zero_guess ? bv : bv - Ap[i, j, k]
        zv = M isa Union{IdentityPreconditioner,PointPreconditioner} ? _pc(M, rv, i, j, k) : zero(T)
        r[i, j, k] = rv
        z[i, j, k] = zv
        return (bv * bv, rv * rv, rv * zv, zv, rv)
    end
end

# x += αp; r -= αAp; z = M⁻¹r (pointwise M) over the plane k, then the
# canonical lanes of (r², r·z, z, r) over the plane (for a general M,
# whose z is formed after this, only r² is used)
@inline _rz4(rv, zv) = (rv * rv, rv * zv, zv, rv)
@inline function _pcg_update_plane(k, ir, jr, x, r, z, p, Ap, α, M, c)
    T = eltype(r)
    @inbounds for j in jr, i in ir
        x[i, j, k] = x[i, j, k] + α * p[i, j, k]
        rv = r[i, j, k] - α * Ap[i, j, k]
        r[i, j, k] = rv
        if M isa Union{IdentityPreconditioner,PointPreconditioner}
            z[i, j, k] = _pc(M, rv, i, j, k)
        end
    end
    _pointwise(M) && _zero_pinned!(z, c, ir, jr, k)
    return _plane_reduce(_add, ntuple(_ -> zero(T), 4), _Vals(_rz4, (r, z)), ir, jr, k)
end

# (r², r·z, z, r) for a general preconditioner, after M!(z, r)
@inline _pcg_rz_body(i, j, k, r, z) = @inbounds _rz4(r[i, j, k], z[i, j, k])

@inline function _pcg_dir_body!(i, j, k, p, z, m, β, first::Bool)
    @inbounds p[i, j, k] = first ? z[i, j, k] - m : (z[i, j, k] - m) + β * p[i, j, k]
    return nothing
end
# p = (z − m) + βp, then p = 0 on the pinned planes
function _pcg_dir!(bk, rg, p, z, m, β, c, first::Bool)
    launch!(bk, _pcg_dir_body!, rg, p, z, m, β, first)
    c === nothing && return nothing
    ir, jr, kr = rg
    c[1] in ir && launch!(bk, _pcg_fill_body!, (c[1]:c[1], jr, kr), p, zero(eltype(p)))
    c[2] in jr && launch!(bk, _pcg_fill_body!, (ir, c[2]:c[2], kr), p, zero(eltype(p)))
    c[3] in kr && launch!(bk, _pcg_fill_body!, (ir, jr, c[3]:c[3]), p, zero(eltype(p)))
    return nothing
end

@inline function _pcg_fill_body!(i, j, k, x, v)
    @inbounds x[i, j, k] = v
    return nothing
end
@inline function _pcg_shift_body!(i, j, k, x, m)
    @inbounds x[i, j, k] = x[i, j, k] - m
    return nothing
end

# ---- operator application: Ap = A p, returns p·Ap --------------------------

# the canonical sum of per-plane partials from a plane function
function _pcg_planes!(buf::Vector{T}, b::CPUBackend, (ir, jr, kr), f::F, args::Vararg{Any,M}) where {T, F, M}
    _plane_partials!(b, buf, _PlaneFn(f, ir, jr), kr, args)
    s = _zero_of(T)
    @inbounds for x in buf
        s = _add(s, x)
    end
    return s
end

function _pcg_apply!(b::CPUBackend, A::PointOperator, Ap, p, ranges, ws, c = nothing)
    A.prepare === nothing || A.prepare(p)
    return _pcg_planes!(ws.s1, b, ranges, _pcg_op_plane, Ap, p, A.f, A.args, c)
end
function _pcg_apply!(b::CPUBackend, A!::F, Ap, p, ranges, ws, c = nothing) where {F}
    A!(Ap, p)
    return _pcg_reduce!(ws.s1, b, ranges, _pcg_dot_body, p, Ap)
end

_pointwise(M) = M isa Union{IdentityPreconditioner,PointPreconditioner}

"""
    pcg!(A, M, x, b, ws::PCGWorkspace; ranges = axes(x), tol, maxit,
         project_mean = false, constraint = nothing, zero_guess = true,
         backend = backend(x))
        -> (; iterations, residual, bnorm, status, curvature)

Solve `A x = b` by preconditioned conjugate gradients over the storage
`ranges` of `x` and `b`, for a symmetric positive (semi-)definite `A`.
The one CG loop of Aeolus (KA-g, #300): `CGPoissonSolver`,
`MGPreconditionedCG`, `DensityWeightedPoissonSolver` and
`solve_implicit_diffusion!` all call it.

  * `A` is a [`PointOperator`](@ref) (fused with the reduction `p·Ap`:
    one sweep) or a function `A!(Ap, p)` that writes `A p` over `ranges`.
  * `M` is an [`IdentityPreconditioner`](@ref), a
    [`PointPreconditioner`](@ref) (fused with the residual update and its
    reductions) or a function `M!(z, r)` (a multigrid V-cycle).
  * `project_mean = true`: `A` is singular with the constants as its null
    space. The mean is removed from `b` (without modifying `b`), from
    every preconditioned residual `z` (a deflated preconditioner, which
    keeps every iterate in the zero-mean subspace) and from the solution.
  * `constraint`: pinned rows, `(ic, jc, kc)` storage planes (0 = none):
    there `Ap` (from a `PointOperator`; a function `A!` writes 0 there
    itself), `r`, `z` and `p` are zero and
    `x` keeps its value; they do not count in `‖b‖`.
  * `zero_guess = true` starts from `x = 0` (written over `ranges`);
    `false` uses `x` as the initial guess (`b === x` is allowed: `A x` is
    formed before `r`).

  * `definite = :positive` (default): `A` and `M` are symmetric positive
    (semi-)definite, and `p·Ap ≤ 0` is a breakdown. `:negative`: both are
    negative definite (the Laplacian `∇²` itself with a multigrid cycle
    of it); then only vanishing curvature is one.

Convergence is `‖r‖₂ ≤ tol·‖b‖₂`, checked before the first iteration
and after every update. `status` is `:converged`, `:maxit`, `:breakdown`
(`|p·Ap| ≤ eps·|r·z|`, vanishing curvature, or `p·Ap ≤ 0` for
`definite = :positive`; the iteration stops before the update) or
`:zero_rhs` (`‖b‖ = 0`: `x = 0`). `curvature` is the last `p·Ap`
formed (the offending one after a breakdown; `NaN` if none was). The
caller reports a breakdown or `:maxit` as it sees fit.

**Determinism.** The reductions are canonical plane sums
([`plane_reduce!`](@ref)), the other sweeps pointwise launches, so a
solve, its iteration count and its residual are identical at every
thread count and with `serial = true` (provided a function `A!` or `M!`
is itself thread-invariant). With the mean projected, `r·z` of the
deflated `z` is formed as `Σ r z − m Σ r`, so the projection costs no
extra sweep.
"""
function pcg!(A, M, x::AbstractArray{T,3}, b::AbstractArray{T,3}, ws::PCGWorkspace{T};
              ranges::NTuple{3,AbstractUnitRange{<:Integer}} = axes(x), tol::Real,
              maxit::Integer, project_mean::Bool = false, constraint = nothing,
              zero_guess::Bool = true, definite::Symbol = :positive,
              backend::AbstractBackend = _backend_of(x)) where {T}
    rg = map(UnitRange{Int}, ranges)
    _check((x, b, ws.r, ws.z, ws.p, ws.Ap), rg)
    project_mean && constraint !== nothing &&
        throw(ArgumentError("pcg!: project_mean with pinned rows is not supported"))
    definite in (:positive, :negative) ||
        throw(ArgumentError("pcg!: definite must be :positive or :negative, got :$definite"))
    return _pcg!(backend, A, M, x, b, ws, rg, T(tol), Int(maxit), project_mean,
                 constraint, zero_guess, definite === :positive)
end

function _pcg!(bk::CPUBackend, A, M, x::AbstractArray{T,3}, b, ws::PCGWorkspace{T}, rg,
               tol::T, maxit::Int, project_mean::Bool, c, zero_guess::Bool,
               positive::Bool) where {T}
    r, z, p, Ap = ws.r, ws.z, ws.p, ws.Ap
    n = T(_npoints(rg))
    mb = project_mean ? _pcg_reduce!(ws.s1, bk, rg, _pcg_val_body, b) / n : zero(T)
    if zero_guess
        launch!(bk, _pcg_fill_body!, rg, x, zero(T))
    else
        _pcg_apply!(bk, A, Ap, x, rg, ws, c)
    end
    b2, r2, rz, Σz, Σr = _pcg_reduce!(ws.s5, bk, rg, _pcg_init_body, r, z, b, Ap, mb, M, c,
                                      zero_guess)
    bnorm = sqrt(b2)
    if bnorm == 0
        launch!(bk, _pcg_fill_body!, rg, x, zero(T))
        return (iterations = 0, residual = zero(T), bnorm = bnorm, status = :zero_rhs,
                curvature = T(NaN))
    end
    if !_pointwise(M)
        M(z, r)
        r2, rz, Σz, Σr = _pcg_reduce!(ws.s4, bk, rg, _pcg_rz_body, r, z)
    end
    m = project_mean ? Σz / n : zero(T)
    project_mean && (rz -= m * Σr)
    _pcg_dir!(bk, rg, p, z, m, zero(T), c, true)

    rnorm = sqrt(r2)
    iters = 0
    pAp = T(NaN)
    status = rnorm <= tol * bnorm ? :converged : :maxit
    while status !== :converged && iters < maxit
        pAp = _pcg_apply!(bk, A, Ap, p, rg, ws, c)
        # vanishing curvature, or curvature of the wrong sign for an SPD
        # system (any NaN fails both tests)
        if !(abs(pAp) > eps(T) * abs(rz)) || (positive && !(pAp > 0))
            status = :breakdown
            break
        end
        α = rz / pAp
        iters += 1
        r2, rzn, Σz, Σr = _pcg_planes!(ws.s4, bk, rg, _pcg_update_plane, x, r, z, p, Ap, α, M, c)
        rnorm = sqrt(r2)
        if rnorm <= tol * bnorm
            status = :converged
            break
        end
        if !_pointwise(M)
            M(z, r)
            _, rzn, Σz, Σr = _pcg_reduce!(ws.s4, bk, rg, _pcg_rz_body, r, z)
        end
        m = project_mean ? Σz / n : zero(T)
        project_mean && (rzn -= m * Σr)
        β = rzn / rz
        _pcg_dir!(bk, rg, p, z, m, β, c, false)
        rz = rzn
    end
    if project_mean
        mx = _pcg_reduce!(ws.s1, bk, rg, _pcg_val_body, x) / n
        launch!(bk, _pcg_shift_body!, rg, x, mx)
    end
    return (iterations = iters, residual = rnorm, bnorm = bnorm, status = status,
            curvature = pAp)
end
_pcg!(b::GPUBackend, args...) = _gpu_error(b, "pcg!")
_backend_of(x) = backend(x)

# =========================================================================
# Counter-based random streams (KA-n, #305)
# =========================================================================
#
# A threaded Monte-Carlo loop cannot share one generator: the draws would
# depend on which task reached it first. Instead every independent unit of
# work (a photon, a grid cell) gets its own stream, a pure function of
# (seed, stream, id): Philox4x32-10 (Salmon, Moraes, Dror & Shaw 2011,
# "Parallel random numbers: as easy as 1, 2, 3", SC'11), the generator of
# Random123, cuRAND and JAX. Each output block is a bijection of a 128-bit
# counter under a 64-bit key, so streams with different counters never
# overlap and need no state beyond the counter. The draws of a unit are
# then the same whichever thread runs it and however the units are split.

const _PHILOX_M0 = 0xD2511F53
const _PHILOX_M1 = 0xCD9E8D57
const _PHILOX_W0 = 0x9E3779B9
const _PHILOX_W1 = 0xBB67AE85

@inline function _mulhilo(a::UInt32, b::UInt32)
    p = UInt64(a) * UInt64(b)
    return (p >> 32) % UInt32, p % UInt32
end

"""
    philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32}) -> NTuple{4,UInt32}

The Philox4x32-10 block function: ten rounds of the Salmon et al. (2011)
round function, with the Weyl key schedule. Checked against the Random123
known-answer vectors in `test/backends.jl`.
"""
@inline function philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in 1:10
        hi0, lo0 = _mulhilo(_PHILOX_M0, c0)
        hi1, lo1 = _mulhilo(_PHILOX_M1, c2)
        c0, c1, c2, c3 = hi1 ⊻ c1 ⊻ k0, lo1, hi0 ⊻ c3 ⊻ k1, lo0
        if r < 10
            k0 += _PHILOX_W0
            k1 += _PHILOX_W1
        end
    end
    return (c0, c1, c2, c3)
end

"""
    CounterRNG <: AbstractRNG

A counter-based random stream (Philox4x32-10), made by
[`counter_rng`](@ref)`(seed, stream, id)`. The key is the 64-bit `seed`;
the 128-bit counter holds the 64-bit `id` (a photon, a cell, a particle),
the 32-bit `stream` (a step, a g-point, a purpose) and a 32-bit block
index that advances as numbers are drawn (2³² blocks of two `UInt64`s per
stream). Two `CounterRNG`s with the same `(seed, stream, id)` produce the
same numbers; different triples give independent streams.

Use one per work item, never one shared between threads; re-point it with
[`set_stream!`](@ref) instead of allocating a new one for every unit. It
implements the `Random` API (`rand`, `randn`, `shuffle!`, …) through
`rand(rng, UInt64)`. The numbers differ from those of `MersenneTwister` or
`Xoshiro` with the same seed.
"""
mutable struct CounterRNG <: AbstractRNG
    key::NTuple{2,UInt32}
    stream::UInt32
    id::UInt64
    block::UInt32
    buf::UInt64          # second half of the last block
    has_buf::Bool
end

"""
    counter_rng(seed, stream, id) -> CounterRNG

The random stream of unit `id` (≥ 0, < 2⁶⁴) in stream `stream`
(0 ≤ stream < 2³²) under `seed` (any integer, taken modulo 2⁶⁴). See
[`CounterRNG`](@ref).
"""
function counter_rng(seed::Integer, stream::Integer, id::Integer)
    s = seed % UInt64
    return set_stream!(CounterRNG(((s >> 32) % UInt32, s % UInt32), 0, 0, 0, 0, false),
                       stream, id)
end

"""
    counter_xoshiro(seed, stream, id) -> Random.Xoshiro

A `Xoshiro256++` generator whose state is the first four `UInt64`s of the
counter stream `(seed, stream, id)` ([`counter_rng`](@ref)), so it is, like
that stream, a pure function of the triple and the same on every thread.
For long streams: one Philox block costs about 4 ns (the latency of ten
dependent rounds) per two `UInt64`s, a Xoshiro draw under 1 ns, so a work
item that draws hundreds of numbers (a cell's shuffle and pair draws in
`collision_coalescence!`) takes this, while one that draws a few
(`randn(counter_rng(k, n, i))` per droplet) keeps the `CounterRNG`, which
needs no allocation. The draws follow Julia's `Xoshiro` (xoshiro256++,
Blackman & Vigna 2021, ACM TOMS 47(4) 36), seeded with Philox output as
Julia seeds its task-local generators from a parent's.
"""
function counter_xoshiro(seed::Integer, stream::Integer, id::Integer)
    r = counter_rng(seed, stream, id)
    s0, s1, s2, s3 = rand(r, UInt64), rand(r, UInt64), rand(r, UInt64), rand(r, UInt64)
    # the all-zero state is the one xoshiro cannot leave (probability 2⁻²⁵⁶)
    (s0 | s1 | s2 | s3) == 0 && (s0 = one(UInt64))
    return Random.Xoshiro(s0, s1, s2, s3)
end

"""
    set_stream!(rng::CounterRNG, stream, id) -> rng
    set_stream!(rng::CounterRNG, seed, stream, id) -> rng

Point `rng` at the start of stream `(seed, stream, id)` (the seed kept if
not given), without allocating: equivalent to a new
`counter_rng(seed, stream, id)`.
"""
function set_stream!(r::CounterRNG, stream::Integer, id::Integer)
    0 <= stream <= typemax(UInt32) ||
        throw(ArgumentError("set_stream!: stream must lie in 0:$(typemax(UInt32)), got $stream"))
    id >= 0 || throw(ArgumentError("set_stream!: id must be ≥ 0, got $id"))
    r.stream = stream % UInt32
    r.id = id % UInt64
    r.block = 0
    r.has_buf = false
    return r
end
function set_stream!(r::CounterRNG, seed::Integer, stream::Integer, id::Integer)
    s = seed % UInt64
    r.key = ((s >> 32) % UInt32, s % UInt32)
    return set_stream!(r, stream, id)
end

@inline function _next_u64!(r::CounterRNG)
    if r.has_buf
        r.has_buf = false
        return r.buf
    end
    r.block == typemax(UInt32) &&
        error("CounterRNG: stream exhausted (2³² blocks); use more streams")
    c = (r.block, r.stream, r.id % UInt32, (r.id >> 32) % UInt32)
    x0, x1, x2, x3 = philox4x32(c, r.key)
    r.block += one(UInt32)
    r.buf = (UInt64(x3) << 32) | x2
    r.has_buf = true
    return (UInt64(x1) << 32) | x0
end

Random.rng_native_52(::CounterRNG) = UInt64
Random.rand(r::CounterRNG, ::Random.SamplerType{UInt64}) = _next_u64!(r)
Random.rand(r::CounterRNG, ::Random.SamplerType{UInt32}) = _next_u64!(r) % UInt32
Base.copy(r::CounterRNG) = CounterRNG(r.key, r.stream, r.id, r.block, r.buf, r.has_buf)
function Base.show(io::IO, r::CounterRNG)
    print(io, "CounterRNG(stream ", r.stream, ", id ", r.id, ", block ", r.block, ")")
end

# =========================================================================
# Column physics: one work item per column (KA-m, #301)
# =========================================================================
#
# A column body runs the k-sequential physics of one column (a surface
# energy balance, a sedimentation sweep, a dew exchange) and writes only
# its own column. What the old loops accumulated across columns (an
# evaporation total, a worst ratio) the body *returns*; the launch stores
# it in row `c` of `partials` (c = the linear index of the column in
# column-major order), and the caller folds each term over the columns in
# that order ([`column_sums`](@ref), or `foldl(max, …)` for a maximum).
# The partials are written by one work item each, so they, and every
# fold of them, are identical at every thread count — and equal, bit for
# bit, to the old serial loop that accumulated `s += x` column by column
# from zero (a skipped column stores 0, and `s + 0.0 == s`).

@inline function _store_terms!(P, c, t::Tuple)
    _store_from!(P, c, 1, t...)
    return nothing
end
@inline _store_from!(P, c, n) = nothing
@inline function _store_from!(P, c, n, x, rest...)
    @inbounds P[c, n] = x
    return _store_from!(P, c, n + 1, rest...)
end
@inline function _store_terms!(P, c, x)
    @inbounds P[c, 1] = x
    return nothing
end

@inline function _column_store_body!(i, j, body!::F, P, oi, ni, oj, args) where {F}
    _store_terms!(P, (i - oi) + ni * (j - oj - 1), body!(i, j, args...))
    return nothing
end
@inline function _column_store_body!(n, body!::F, P, on, args) where {F}
    _store_terms!(P, n - on, body!(n, args...))
    return nothing
end

"""
    foreach_column!(backend, body!, cols, args...; partials = nothing, weight = 1)
        -> partials

Run the column body `body!(i, j, args...)` (`cols = (ir, jr)`, the
`(i, j)` columns of a surface) or `body!(n, args...)` (`cols = (nr,)`, a
list of column ids: the columns of one surface type, the facets of a
building) once per column, as one work item each. The body loops over
its column's levels itself, in its own (k-sequential) order, and writes
only to its own column — the [`launch!`](@ref) contract with "point"
read as "column": no two columns write the same location, and a column
reads nothing another column writes in the same launch. Each column owns
its scratch (a workspace per column, never one shared buffer).

`partials` (optional): an `AbstractMatrix` with one row per column, in
column-major order (`c = (i - first(ir) + 1) + length(ir) (j - first(jr))`,
or `c = n - first(nr) + 1`) and one column per term. The body then
returns its column's terms — a number, or an `NTuple` with one entry per
term (zeros for a column that contributes nothing) — and the launch
stores them in row `c`. No column body touches a shared accumulator.
Fold them in column order with [`column_sums`](@ref) (a sum) or `foldl`
(a maximum: `max` is exact); both are thread-invariant and equal, bit for
bit, to the serial loop `s = 0.0; for j, for i: s += x` they replace.

`weight` is the work of one column in grid points (its number of levels,
or the cost of its root solves in point-equivalents): the backend's
serial threshold and its points-per-task rule count `weight × ncolumns`
points (see [`launch_weighted!`](@ref)). Serially the columns run in the
historical `for j, for i` (or ascending `n`) order.
"""
function foreach_column!(b::AbstractBackend, body!::F, cols::NTuple{N,UnitRange{Int}},
                         args::Vararg{Any,M}; partials = nothing,
                         weight::Integer = 1) where {F, N, M}
    N in (1, 2) || throw(ArgumentError("foreach_column!: cols must be (ir, jr) or (nr,)"))
    if partials === nothing
        launch_weighted!(b, weight, body!, cols, args...)
        return nothing
    end
    nc = _npoints(cols)
    size(partials, 1) == nc ||
        throw(DimensionMismatch("partials has $(size(partials, 1)) rows, the launch $nc columns"))
    if N == 2
        ir, jr = cols
        launch_weighted!(b, weight, _column_store_body!, cols, body!, partials,
                         first(ir) - 1, length(ir), first(jr) - 1, args)
    else
        launch_weighted!(b, weight, _column_store_body!, cols, body!, partials,
                         first(cols[1]) - 1, args)
    end
    return partials
end
foreach_column!(b::AbstractBackend, body!::F, cols::NTuple{N,AbstractUnitRange{<:Integer}},
                args::Vararg{Any,M}; kw...) where {F, N, M} =
    foreach_column!(b, body!, map(UnitRange{Int}, cols), args...; kw...)

"""
    foreach_cell!(backend, body!, ranges::NTuple{3,UnitRange{Int}}, args...;
                  partials = nothing, weight = 1) -> partials

Per-cell physics (a saturation adjustment, a conversion step): run
`body!(i, j, k, args...)` once per cell of `ranges` — [`launch!`](@ref)
with a `weight` (points per cell, for a body with a root solve).

With `partials` (a vector, or an `NTuple` of vectors, with one entry per
plane `k`) the body returns what each cell adds to the totals and the
launch is [`plane_reduce!`](@ref): the canonical, thread-invariant plane
sums (eight lanes per row), to be combined with [`ordered_sum`](@ref).
For a fused update whose terms must be added in cell order (most cells
contribute nothing, and the cheap ones are skipped) use
[`plane_map!`](@ref) instead.
"""
function foreach_cell!(b::AbstractBackend, body!::F, ranges::NTuple{3,UnitRange{Int}},
                       args::Vararg{Any,M}; partials = nothing,
                       weight::Integer = 1) where {F, M}
    if partials === nothing
        launch_weighted!(b, weight, body!, ranges, args...)
        return nothing
    end
    return plane_reduce!(partials, b, body!, ranges, args...)
end
foreach_cell!(b::AbstractBackend, body!::F, ranges::NTuple{3,AbstractUnitRange{<:Integer}},
              args::Vararg{Any,M}; kw...) where {F, M} =
    foreach_cell!(b, body!, map(UnitRange{Int}, ranges), args...; kw...)

"""
    column_sum(backend, body, cols, args...; weight = 1) -> sum or NTuple of sums

The totals over the columns of what a column body returns — a number, or
an `NTuple` of numbers — in column-major order from zero: the
[`foreach_column!`](@ref) launch with partials followed by
[`column_sums`](@ref), without the partials when the launch runs on the
calling task. Serially (one thread, `serial = true`, or fewer than the
backend's threshold of `weight × ncolumns` points) the body's terms are
added as they come, column after column — the old loops' running sums,
with no partials array to fill and fold afterwards; threaded, every
column stores its terms and the columns are folded in the same order. The
two paths add the same numbers in the same order: the totals are
identical, bit for bit, at every thread count.
"""
function column_sum(b::AbstractBackend, body::F, cols::NTuple{N,AbstractUnitRange{<:Integer}},
                    args::Vararg{Any,M}; weight::Integer = 1) where {F, N, M}
    cr = map(UnitRange{Int}, cols)
    nc = _npoints(cr)
    R = N == 2 ? Base.promote_op(body, Int, Int, map(typeof, args)...) :
                 Base.promote_op(body, Int, map(typeof, args)...)
    if runs_serially(b, nc * max(1, Int(weight))) || !_summable(R)
        return _serial_column_sum(body, cr, args, _zero_terms(R))
    end
    P = Matrix{_term_eltype(R)}(undef, nc, _nterms(R))
    foreach_column!(b, body, cr, args...; partials = P, weight = weight)
    S = column_sums(P)
    return R <: Tuple ? Tuple(S) : S[1]
end

_summable(::Type{R}) where {R<:Real} = isconcretetype(R)
_summable(::Type{R}) where {R<:Tuple} = isconcretetype(R) && all(T -> T <: Real, fieldtypes(R))
_summable(::Type) = false
_nterms(::Type{R}) where {R<:Tuple} = fieldcount(R)
_nterms(::Type) = 1
_term_eltype(::Type{R}) where {R<:Tuple} = promote_type(fieldtypes(R)...)
_term_eltype(::Type{R}) where {R} = R
# the zero the running sums start from (`nothing`: not inferred, taken from
# the first column's terms)
_zero_terms(::Type{R}) where {R<:Real} = isconcretetype(R) ? zero(R) : nothing
_zero_terms(::Type{R}) where {R<:Tuple} = _summable(R) ? map(zero, fieldtypes(R)) : nothing
_zero_terms(::Type) = nothing

@inline _add_terms(s::Tuple, x::Tuple) = map(+, s, x)
@inline _add_terms(s, x) = s + x
@inline _add_terms(::Nothing, x::Tuple) = map((z, y) -> z + y, map(zero, x), x)
@inline _add_terms(::Nothing, x) = zero(x) + x

@noinline function _serial_column_sum(body::F, (ir, jr)::NTuple{2,UnitRange{Int}}, args, s) where {F}
    for j in jr, i in ir
        s = _add_terms(s, body(i, j, args...))
    end
    return s
end
@noinline function _serial_column_sum(body::F, (nr,)::NTuple{1,UnitRange{Int}}, args, s) where {F}
    for n in nr
        s = _add_terms(s, body(n, args...))
    end
    return s
end

"""
    column_sums(partials) -> Vector

The sum of every term (column of `partials`) over the columns (rows), in
row order from zero ([`ordered_sum`](@ref)): the canonical total of a
[`foreach_column!`](@ref) launch, thread-invariant.
"""
function column_sums(P::AbstractMatrix{T}) where {T}
    nt = size(P, 2)
    nt <= 8 && return collect(_column_sums(P, Val(nt)))
    return [ordered_sum(view(P, :, t)) for t in axes(P, 2)]
end
# One pass over the rows, every term its own running sum held in a register
# (independent chains, so one's add latency hides behind the others'),
# each in row order from zero: the ordered_sum of each column of P.
#
# The row update is a separate function: a closure `t -> s[t] + P[c, t]`
# inside the loop would capture `s`, which the loop reassigns, so Julia
# boxes it and every row runs through dynamic dispatch (measured 5.5 ms
# instead of a few µs for 128² rows).
function _column_sums(P::AbstractMatrix{T}, ::Val{N}) where {T, N}
    s = ntuple(_ -> zero(T), Val(N))
    for c in axes(P, 1)
        s = _add_row(s, P, c)
    end
    return s
end
@inline _add_row(s::NTuple{N,T}, P, c) where {N, T} =
    ntuple(t -> (@inbounds s[t] + P[c, t]), Val(N))

end # module
