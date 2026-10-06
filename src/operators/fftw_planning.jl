# =========================================================================
# The FFTW planning policy (#245): one source of truth for the planner
# flag and the thread count of every FFTW plan in the code base.
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

using FFTW: FFTW

const _FFTW_PLANNERS = (:estimate, :measure, :patient, :exhaustive)

_fftw_flag(p::Symbol) = p === :estimate ? FFTW.ESTIMATE :
                        p === :measure ? FFTW.MEASURE :
                        p === :patient ? FFTW.PATIENT :
                        p === :exhaustive ? FFTW.EXHAUSTIVE :
                        throw(ArgumentError("FFTW planner must be one of $(_FFTW_PLANNERS), got :$p"))

mutable struct _FFTWPolicyState
    resolved::Bool
    planner::Symbol
    threads::Int
    threads_explicit::Bool   # chosen by AEOLUS_FFTW_THREADS / set_fftw_policy!(threads =)
    wisdom_loaded::Bool
    wisdom_file::String
end

const _FFTW_POLICY = _FFTWPolicyState(false, :measure, 1, false, false, "")

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
# environment variables AEOLUS_FFTW_PLANNER / AEOLUS_FFTW_THREADS, which is
# how the test suite (and every subprocess it starts) is put on ESTIMATE.
function _fftw_policy!()
    st = _FFTW_POLICY
    if !st.resolved
        st.planner = _parse_planner(get(ENV, "AEOLUS_FFTW_PLANNER", String(FFTW_DEFAULT_PLANNER)))
        st.threads_explicit = haskey(ENV, "AEOLUS_FFTW_THREADS")
        st.threads = parse(Int, get(ENV, "AEOLUS_FFTW_THREADS", "1"))
        st.threads >= 1 || throw(ArgumentError("AEOLUS_FFTW_THREADS must be ≥ 1"))
        FFTW.set_num_threads(st.threads)
        st.resolved = true
    end
    return st
end

"""
    set_fftw_policy!(; planner = nothing, threads = nothing) -> NamedTuple

Set the FFTW planning policy every FFT plan of Aeolus is made with (the
pressure solvers' and the diagnostics'): `planner` is `:estimate`,
`:measure` (the default), `:patient` or `:exhaustive`, and `threads` the
FFTW thread count (`FFTW.set_num_threads`). `:patient` finds somewhat
faster plans at a much higher planning cost — measured at 256³: 494 s of
planning with `:patient` against 2.4 s with `:measure`, for transforms of
153 vs 182 ms — so it is worth opting into only for long runs. Plans made before the call keep
their flags; set the policy before constructing a simulation.

The default can also be set from the environment, before the first plan:
`AEOLUS_FFTW_PLANNER=estimate`, `AEOLUS_FFTW_THREADS=4`. The test suite
sets `AEOLUS_FFTW_PLANNER=estimate`, so its bitwise comparisons hold across
processes and planning stays fast. A TOML namelist sets it with an
`[fftw]` table ([`apply_fftw_config!`](@ref Aeolus.Config.apply_fftw_config!)).

**Threads.** Without an explicit count the policy starts at one thread
and [`init_threads!`](@ref Aeolus.Backends.init_threads!) raises it to
`Threads.nthreads()` (KA-h, #298). A count set here, through
`AEOLUS_FFTW_THREADS` or by a namelist is explicit, and `init_threads!`
keeps it. FFTW's threaded plans hand whole 1-D transforms to the
threads, so under `:estimate` a solve gives identical bits at every FFTW
thread count (measured for every pressure solver and axis-BC
combination, 1-24 threads, docs/src/design/ka_port.md). Under the timing
planners the planner may pick a different algorithm for a different
thread count (as it may for another process), so there it holds only
with imported wisdom made at the same thread count.

**Reproducibility.** `:measure`, `:patient` and `:exhaustive` time
candidate algorithms, so two processes may choose different plans and
differ in the last bit (FFTW FAQ 3.8); `:estimate` is deterministic, and
so is any planner whose plans come from imported wisdom
([`load_fftw_wisdom!`](@ref)). [`fftw_policy`](@ref) reports which case
holds. Returns the new [`fftw_policy`](@ref).

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
    if threads !== nothing
        threads >= 1 || throw(ArgumentError("FFTW threads must be ≥ 1, got $threads"))
        st.threads = Int(threads)
        st.threads_explicit = true
        FFTW.set_num_threads(st.threads)
    end
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

# `Backends.init_threads!` (KA-a, #294) sets and reports the FFTW thread
# count through this policy, so there is one place FFTW is configured.
using ..Backends: Backends
function Backends._set_fftw_threads!(n::Int; default::Bool = false)
    st = _fftw_policy!()
    default && st.threads_explicit && return nothing   # an explicit count wins
    n >= 1 || throw(ArgumentError("FFTW threads must be ≥ 1, got $n"))
    st.threads = n
    default || (st.threads_explicit = true)
    FFTW.set_num_threads(n)
    return nothing
end
Backends._fftw_threads() = _fftw_policy!().threads
