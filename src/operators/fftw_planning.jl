# =========================================================================
# The FFTW planning policy (#245): one source of truth for the planner
# flag of every FFTW plan in the code base, and the one way a transform is
# threaded (#575).
# =========================================================================
#
# Included into `module Operators` (the lowest module that uses FFTW), so
# the pressure solvers (PressureSolvers) and the diagnostics here call the
# same helper. Every `plan_*` call passes `flags = fftw_planner_flags()`.
#
# Why it matters: FFTW's MEASURE / PATIENT / EXHAUSTIVE planners time
# candidate algorithms and keep the fastest, so two processes may pick
# different plans, which round differently in the last bit (FFTW FAQ 3.8).
# ESTIMATE is deterministic, and so is any planner when the plan comes from
# imported wisdom. Hence: MEASURE by default (maintainer decision on #318:
# PATIENT plans 256³ in 494 s against 2.4 s for MEASURE, for transforms of
# 153 vs 182 ms), PATIENT opt-in, wisdom saved at run start and reloaded
# on restart, ESTIMATE in the test suite.
#
# Threads (#575). Every FFTW plan is single-threaded; a large transform is
# cut into blocks (levels, or rows of columns) that Julia's threads
# transform with serial plans ([`BlockTransform`](@ref)), as PALM does
# (`fft_xy_mod.f90`: serial 1-D plans inside `!$OMP DO`). FFTW never starts
# a thread pool of its own. That pool was created from Julia's thread 1 and
# so inherited its CPU affinity: with `JULIA_EXCLUSIVE=1` every FFTW thread
# ran on core 0 next to thread 1, and the Berlin 20 m step took 28.4 s
# pinned against ~6 s unpinned at 72 threads (#566, #575). Its threaded
# planner could also thread *inside* one transform: the estimate plan of
# the 150-point z DCT at 1000 × 1000 × 150 took 118 s on 24 threads
# against 1.7 s on one (#572).

using FFTW: FFTW
using LinearAlgebra: mul!

const _FFTW_PLANNERS = (:estimate, :measure, :patient, :exhaustive)

_fftw_flag(p::Symbol) = p === :estimate ? FFTW.ESTIMATE :
                        p === :measure ? FFTW.MEASURE :
                        p === :patient ? FFTW.PATIENT :
                        p === :exhaustive ? FFTW.EXHAUSTIVE :
                        throw(ArgumentError("FFTW planner must be one of $(_FFTW_PLANNERS), got :$p"))

mutable struct _FFTWPolicyState
    resolved::Bool
    planner::Symbol
    threads::Int             # always 1 (#575); kept for the checkpoint record
    wisdom_loaded::Bool
    wisdom_file::String
end

const _FFTW_POLICY = _FFTWPolicyState(false, :measure, 1, false, "")

# Default planner: :measure (#245, maintainer decision on #318: near-PATIENT
# plans at a fraction of the planning cost), unless ENV["AEOLUS_FFTW_PLANNER"]
# says otherwise. :patient is opt-in.
const FFTW_DEFAULT_PLANNER = :measure

function _parse_planner(s)
    p = Symbol(lowercase(strip(String(s))))
    p in _FFTW_PLANNERS ||
        throw(ArgumentError("FFTW planner must be one of $(join(_FFTW_PLANNERS, ", ")), got \"$s\""))
    return p
end

# The policy is resolved lazily (no module __init__): on first use, from the
# environment variable AEOLUS_FFTW_PLANNER, which is how the test suite (and
# every subprocess it starts) is put on ESTIMATE. AEOLUS_FFTW_THREADS is
# deprecated (#575): a value other than 1 is reported and ignored.
function _fftw_policy!()
    st = _FFTW_POLICY
    if !st.resolved
        st.planner = _parse_planner(get(ENV, "AEOLUS_FFTW_PLANNER", String(FFTW_DEFAULT_PLANNER)))
        t = get(ENV, "AEOLUS_FFTW_THREADS", "")
        isempty(t) || _fftw_threads_deprecated(something(tryparse(Int, t), t), "AEOLUS_FFTW_THREADS")
        st.threads = 1
        FFTW.set_num_threads(1)
        st.resolved = true
    end
    return st
end

# A requested FFTW thread count other than 1 (#575): FFTW plans are
# single-threaded and the transforms run on Julia's threads, so the count
# has no meaning any more. Reported once per source, then ignored.
const _FFTW_THREADS_WARNED = Set{String}()
function _fftw_threads_deprecated(n, source::AbstractString)
    n isa Integer && n < 1 && throw(ArgumentError("FFTW threads must be ≥ 1, got $n"))
    n == 1 && return nothing
    if !(source in _FFTW_THREADS_WARNED)
        push!(_FFTW_THREADS_WARNED, String(source))
        @warn "$source = $n is deprecated and ignored: FFTW plans are single-threaded and " *
              "Aeolus runs its transforms on Julia's threads (#575). Set the thread count " *
              "with `julia -t`."
    end
    return nothing
end

"""
    set_fftw_policy!(; planner = nothing, threads = nothing) -> NamedTuple

Set the FFTW planning policy every FFT plan of Aeolus is made with (the
pressure solvers' and the diagnostics'): `planner` is `:estimate`,
`:measure` (the default), `:patient` or `:exhaustive`. `:patient` finds somewhat
faster plans at a much higher planning cost — measured at 256³: 494 s of
planning with `:patient` against 2.4 s with `:measure`, for transforms of
153 vs 182 ms — so it is worth opting into only for long runs. Plans made before the call keep
their flags; set the policy before constructing a simulation.

The default can also be set from the environment, before the first plan:
`AEOLUS_FFTW_PLANNER=estimate`. The test suite sets
`AEOLUS_FFTW_PLANNER=estimate`, so its plans are deterministic and planning
stays fast. A TOML namelist sets it with an `[fftw]` table
([`apply_fftw_config!`](@ref Aeolus.Config.apply_fftw_config!)).

**Threads (#575).** `threads` is deprecated: every FFTW plan is
single-threaded, and a large transform is cut into blocks (levels, or rows
of columns) that Julia's threads transform ([`BlockTransform`](@ref)), as
PALM does. A value other than 1 here, in `AEOLUS_FFTW_THREADS`, in a
namelist's `[fftw] threads` or in `init_threads!(; fftw_threads)` is
reported once and ignored. FFTW's own thread pool, used before, inherited
the CPU affinity of Julia's thread 1, so with `JULIA_EXCLUSIVE=1` all its
threads shared core 0 (28.4 s per step pinned against ~6 s unpinned at
Berlin 20 m, 72 threads, #566); its planner could also thread inside one
transform (118 s per transform on 24 threads against 1.7 s on one, #572).

**Reproducibility.** `:measure`, `:patient` and `:exhaustive` time
candidate algorithms, so two processes may choose different plans and
differ in the last bit (FFTW FAQ 3.8); `:estimate` is deterministic, and
so is any planner whose plans come from imported wisdom
([`load_fftw_wisdom!`](@ref)). [`fftw_policy`](@ref) reports which case
holds. The blocks of a transform follow the Julia thread count, so the
plans (and their last bits) may differ between thread counts; results
agree to round-off. Returns the new [`fftw_policy`](@ref).

# References

  * Frigo, M. & Johnson, S. G. (2005). The design and implementation of
    FFTW3. *Proc. IEEE*, 93, 216-231.
  * FFTW FAQ, question 3.8: "FFTW gives different results between runs".
"""
function set_fftw_policy!(; planner = nothing, threads = nothing)
    st = _fftw_policy!()
    if planner !== nothing
        st.planner = planner isa Symbol ? (planner in _FFTW_PLANNERS ? planner :
                     throw(ArgumentError("FFTW planner must be one of $(_FFTW_PLANNERS), got :$planner"))) :
                     _parse_planner(planner)
    end
    threads === nothing || _fftw_threads_deprecated(threads, "FFTW threads")
    return fftw_policy()
end

"""
    fftw_planner_flags() -> UInt32

The FFTW planner flag of the current policy ([`set_fftw_policy!`](@ref)).
**Every** `plan_*` call in Aeolus passes `flags = fftw_planner_flags()`:
it is the one place the planner is chosen.
"""
fftw_planner_flags() = _fftw_flag(_fftw_policy!().planner)

"""
    fftw_policy() -> NamedTuple

The current FFTW policy and its provenance: `planner` (Symbol), `threads`,
`wisdom_loaded` (wisdom was imported with [`load_fftw_wisdom!`](@ref) in
this process), `wisdom_file` (`""` if none) and `bitwise_reproducible`:
`true` when the planner is `:estimate` or wisdom was loaded, i.e. when a
rerun or restart in another process makes the same plans (for wisdom: on
the same machine and FFTW build, for the plan sizes the wisdom contains,
which [`save_fftw_wisdom`](@ref) at run start guarantees). Recorded in
every checkpoint ([`checkpoint!`](@ref Aeolus.IO.checkpoint!)).
"""
function fftw_policy()
    st = _fftw_policy!()
    return (planner = st.planner, threads = st.threads, wisdom_loaded = st.wisdom_loaded,
            wisdom_file = st.wisdom_file,
            bitwise_reproducible = st.planner === :estimate || st.wisdom_loaded)
end

"""
    save_fftw_wisdom(path) -> path

Write FFTW's accumulated wisdom (every plan made so far in this process,
with the planner's choices) to `path` (`FFTW.export_wisdom`). Called at run
start, after the simulation (and so its plans) exists, next to the run's
checkpoints, so that a restart can [`load_fftw_wisdom!`](@ref) it and make
the same plans bit for bit (the MicroHH practice).
"""
function save_fftw_wisdom(path::AbstractString)
    _fftw_policy!()
    mkpath(dirname(abspath(path)))
    FFTW.export_wisdom(path)
    return path
end

"""
    load_fftw_wisdom!(path) -> Bool

Import FFTW wisdom from `path` (`FFTW.import_wisdom`) **before** any plan
of the run is made; returns `false` (and changes nothing) if the file does
not exist. Afterwards [`fftw_policy`](@ref) reports `wisdom_loaded = true`
and `bitwise_reproducible = true`: plans whose problem is in the wisdom are
taken from it instead of being re-timed.
"""
function load_fftw_wisdom!(path::AbstractString)
    st = _fftw_policy!()
    isfile(path) || return false
    FFTW.import_wisdom(path)
    st.wisdom_loaded = true
    st.wisdom_file = abspath(path)
    return true
end

# `Backends.init_threads!` (KA-a, #294) reports the FFTW thread count
# through this policy, so there is one place FFTW is configured. Since
# #575 the count is always 1: no count is set, an explicit one other than
# 1 is reported and ignored.
using ..Backends: Backends, CPUBackend, GPUBackend, runs_serially
function Backends._set_fftw_threads!(n::Int; default::Bool = false)
    _fftw_policy!()
    default || _fftw_threads_deprecated(n, "init_threads!(; fftw_threads)")
    return nothing
end
Backends._fftw_threads() = _fftw_policy!().threads

# =========================================================================
# BlockTransform (#575): serial FFTW plans run on Julia's threads
# =========================================================================

"""
    BlockTransform(make_plan, out, in, split; backend, nblocks)

A transform of the 3-D array `in` into `out` (which may be `in` itself),
done as independent pieces: both arrays are cut along axis `split` into
`nblocks` contiguous blocks (default: one per Julia thread, at most one per
index of that axis; one when `backend` runs serially), each block is
transformed by a single-threaded FFTW plan, and Julia's threads take the
blocks ([`launch_weighted!`](@ref Aeolus.Backends.launch_weighted!)).
`split` must not be a transformed axis: the 2-D transforms in `x` and `y`
split along `z` (blocks of levels), a transform along `z` splits along `y`
(blocks of rows).

`make_plan(out_block, in_block)` makes the plan of one block (a serial
plan: FFTW's thread count is always 1, see [`set_fftw_policy!`](@ref));
it is called once per distinct block size and memory alignment (at most
four plans; a level of an `Nx × Ny` Float64 array is 16-byte aligned only
when the offset `(k-1)·Nx·Ny` is even, and FFTW plans depend on the
alignment mod 16). Applied with
[`apply_transform!`](@ref Aeolus.Backends.apply_transform!)`(backend, out,
t, in)`, to `out` and `in` of the sizes and alignment it was made with
(FFTW.jl checks both); arrays allocated on their own (`Array`, Julia's
64-byte alignment) qualify.

This is PALM's method (PALM 25.10.1 `fft_xy_mod.f90`, serial 1-D plans
inside `!\$OMP DO`). FFTW starts no threads of its own, so every transform
runs on the threads the user placed (#575).
"""
struct BlockTransform{S, P}
    plans::Vector{P}
    plan_of::Vector{Int}                 # the plan of each block
    ranges::Vector{UnitRange{Int}}       # the indices of axis S of each block
end

@inline _block(a::AbstractArray{<:Any,3}, ::Val{3}, r) = view(a, :, :, r)
@inline _block(a::AbstractArray{<:Any,3}, ::Val{2}, r) = view(a, :, r, :)
@inline _block(a::AbstractArray{<:Any,3}, ::Val{1}, r) = view(a, r, :, :)

function BlockTransform(make_plan, out::AbstractArray{<:Any,3}, in::AbstractArray{<:Any,3},
                        split::Int; backend = Backends._host_backend(),
                        nblocks::Integer = _default_nblocks(backend, in, split))
    1 <= split <= 3 || throw(ArgumentError("BlockTransform: split must be 1, 2 or 3, got $split"))
    n = size(in, split)
    size(out, split) == n ||
        throw(DimensionMismatch("BlockTransform: in and out differ along the split axis $split"))
    nb = clamp(Int(nblocks), 1, max(n, 1))
    ranges = [((b - 1) * n ÷ nb + 1):(b * n ÷ nb) for b in 1:nb]
    S = Val(split)
    keys = Tuple{Int,Int32,Int32}[]
    plans = Any[]
    plan_of = Int[]
    for r in ranges
        bi, bo = _block(in, S, r), _block(out, S, r)
        key = (length(r), FFTW.alignment_of(bi), FFTW.alignment_of(bo))
        idx = findfirst(==(key), keys)
        if idx === nothing
            push!(plans, make_plan(bo, bi))
            push!(keys, key)
            idx = length(plans)
        end
        push!(plan_of, idx)
    end
    P = typeof(plans[1])
    all(p -> p isa P, plans) ||
        throw(ArgumentError("BlockTransform: make_plan gave plans of different types"))
    return BlockTransform{split, P}(P[p for p in plans], plan_of, ranges)
end

_default_nblocks(b::CPUBackend, a, split) =
    runs_serially(b, length(a)) ? 1 : min(size(a, split), Threads.nthreads())
_default_nblocks(b, a, split) = 1

"""
    nblocks(t::BlockTransform) -> Int

The number of blocks (independent serial transforms) of `t`.
"""
nblocks(t::BlockTransform) = length(t.ranges)

@inline function _block_transform_body!(b, plans, plan_of, ranges, out, inp, s::Val)
    @inbounds r = ranges[b]
    @inbounds p = plans[plan_of[b]]
    mul!(_block(out, s, r), p, _block(inp, s, r))
    return nothing
end

function Backends.apply_transform!(be::CPUBackend, out, t::BlockTransform{S}, in) where {S}
    nb = length(t.ranges)
    # weight: the points of one block, so the serial threshold counts the work
    launch_weighted!(be, cld(length(in), nb), _block_transform_body!, (1:nb,),
                     t.plans, t.plan_of, t.ranges, out, in, Val(S))
    return out
end
Backends.apply_transform!(be::GPUBackend, out, t::BlockTransform, in) =
    Backends._gpu_error(be, "apply_transform!")

# ---- serial plans of one block, made on the block's own output ---------------
#
# FFTW.jl's `plan_*` make an out-of-place plan for a fresh contiguous
# output, whose strides differ from those of a block of a larger array (a
# block of rows `view(a, :, r, :)`); these make the plan for the given
# output block (FFTW.jl's guru constructors), with the policy's planner
# flag. FFTW's thread count is 1 (the policy), so each plan is serial.

"""
    serial_r2r_plan(out, in, kinds, region)

A single-threaded FFTW real-to-real plan from `in` into `out` (the arrays
the plan is applied to, or blocks of the same shape, strides and
alignment) along the axes `region` with transform `kinds` (`FFTW.REDFT10`,
…), made with [`fftw_planner_flags`](@ref).
"""
serial_r2r_plan(out::StridedArray{T,N}, in::StridedArray{T,N}, kinds, region) where {T,N} =
    FFTW.r2rFFTWPlan{T,Any,out === in,N}(in, out, region, kinds, fftw_planner_flags(), FFTW.NO_TIMELIMIT)

"""
    serial_rfft_plan(out, in, region)

A single-threaded real-to-complex FFTW plan (unnormalised forward DFT,
the first axis of `region` halved to `n ÷ 2 + 1`) from `in` into `out`.
"""
serial_rfft_plan(out::StridedArray{Complex{T},N}, in::StridedArray{T,N}, region) where {T,N} =
    FFTW.rFFTWPlan{T,FFTW.FORWARD,false,N}(in, out, region, fftw_planner_flags(), FFTW.NO_TIMELIMIT)

"""
    serial_brfft_plan(out, in, region)

A single-threaded complex-to-real FFTW plan (the unnormalised inverse of
[`serial_rfft_plan`](@ref): a forward and backward pair multiplies by the
product of the lengths of `region`) from `in` into `out`. Like every
multidimensional c2r transform it overwrites `in`.
"""
serial_brfft_plan(out::StridedArray{T,N}, in::StridedArray{Complex{T},N}, region) where {T,N} =
    FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,N}(in, out, region, fftw_planner_flags(),
                                                    FFTW.NO_TIMELIMIT)

"""
    serial_fft_plan(out, in, region, direction)

A single-threaded complex FFTW plan (`direction` `FFTW.FORWARD` or
`FFTW.BACKWARD`, unnormalised) from `in` into `out`, in place when they are
the same array.
"""
serial_fft_plan(out::StridedArray{Complex{T},N}, in::StridedArray{Complex{T},N}, region,
                direction::Integer) where {T,N} =
    direction == FFTW.FORWARD ?
        FFTW.cFFTWPlan{Complex{T},FFTW.FORWARD,out === in,N}(in, out, region, fftw_planner_flags(), FFTW.NO_TIMELIMIT) :
        FFTW.cFFTWPlan{Complex{T},FFTW.BACKWARD,out === in,N}(in, out, region, fftw_planner_flags(), FFTW.NO_TIMELIMIT)
