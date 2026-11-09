# =========================================================================
# The fused sweep: the local-term interface and its driver
# (SC-e #405, #533). Included by FiniteVolume.
# =========================================================================
#
# After SC-d a substage still streams u, v, w (and θ, qv) twice: the
# tendency launch reads them and writes the tendencies, then the predictor
# `u ← u + Δt·tu` reads both back and writes u. The fused sweep runs both
# per block, while the block is in cache:
#
#   * the k-range is cut into `ns` slabs and the j-range into `nt` row
#     ranges (#558), one task per block (slab × row range);
#   * a block is swept j-row by j-row: the tendencies of row j (every k of
#     the slab) go into a ring of `lag + 1` rows, then row j − lag gets its
#     predictor. `lag` is the largest y-radius of the terms' footprints (1
#     for the base tendencies' j ± 1 stencils): no later tendency of the
#     block reads row j − lag, so the update can be in place;
#   * the slab's first and last `lz` planes (`lz` the largest z-radius) are
#     read by the neighbouring slabs' stencils, so their tendencies are kept
#     (`edge`) and their predictor runs in a second launch, after every
#     block is done; likewise the first and last `ly` rows of a row range
#     (kept in extra rows of the ring, written by a third launch).
#
# What a row's tendency is, is a compile-time tuple of *terms* (#533):
# the first initialises the row (the base tendencies of SC-d: advection
# and molecular diffusion), each further term adds its contribution in
# the fixed order of the tuple. A term is an immutable object with a
# point kernel (`add_tendency!`) or a row kernel (`add_row!`); everything
# dispatches statically, nothing per point.
#
# The slab and ring machinery is one driver, `blocked_update!`, generic
# over the update it runs (a `BlockedUpdate`, #554): the predictor here,
# and the update of the SGS TKE (`PrognosticSGS`, src/prognostic_sgs/
# tke_blocked.jl), whose row pipeline (eddy viscosity, limited fluxes,
# diffusion, the update) keeps its own intermediate rows.
#
# Every value is the expression of the separate passes (`_tend_*`, the
# terms' whole-field reference, then the predictor functor `pu` / `ps`),
# so the result does not depend on the number of slabs, of threads, or on
# whether the sweep is used, beyond round-off. The tendency fields are
# neither read nor written. Measured in docs/src/design/ka_port.md,
# "SC-e", and docs/src/design/roadmap.md (#533).

# ---------------------------------------------------------------------------
# the local-term interface
# ---------------------------------------------------------------------------

"""
    SweepTerm

The supertype of a term of the fused sweep (#533): an immutable object
that adds its tendency at one point while the sweep holds the row in
cache. A term implements

  * [`footprint`](@ref)`(t) -> (rx, ry, rz)`, the stencil radius over the
    state it reads (required);
  * [`add_tendency!`](@ref)`(t, tb, i, j, k, st)`, its point kernel, or
    [`add_row!`](@ref) for a whole row (required, one of the two);
  * [`sweep_inputs`](@ref)`(t)`, the extra read-only arrays it reads
    (default none);
  * [`prepare!`](@ref)`(t, sim, Δt)`, its work before the sweep:
    reductions of the stage-start state, precomputed fields (default
    none).

A component or a built-in term of the core is made one by
[`sweep_term`](@ref). The term's whole-field method (its `run_stage!` or
the core's `add_*!` operator) stays the reference the tests compare the
sweep against.
"""
abstract type SweepTerm end

"""
    sweep_term(x, ctx) -> Union{SweepTerm,Nothing}

The term of the fused sweep that does what `x` (a component, with `ctx`
its domain; or a built-in operator of the core, with `ctx` the
simulation) adds to the tendencies, or `nothing` (the default) when `x`
cannot be expressed as a local term yet. A substage with one `nothing`
runs one loop per term, and the run reports which term caused it (#533).
"""
sweep_term(x, ctx) = nothing

"""
    footprint(t::SweepTerm) -> NTuple{3,Int}

The stencil radius `(rx, ry, rz)` over the state term `t` reads at a
point: `(0, 0, 0)` for a pointwise term, `(1, 1, 1)` for the base
tendencies' second-order stencils. The sweep's ring holds `1 + max ry`
rows and its slabs keep `max rz` edge planes at each end.
"""
footprint(t::SweepTerm) = throw(ArgumentError("footprint: the sweep term $(typeof(t)) declares no footprint"))

"""
    sweep_inputs(t::SweepTerm) -> Tuple

The extra read-only arrays term `t` reads besides the state (masks,
profiles, eddy viscosities …), on the state's backend. Default none.
"""
sweep_inputs(::SweepTerm) = ()

"""
    prepare!(t::SweepTerm, sim, Δt)

The work of term `t` before the sweep of a substage of length `Δt` on
`sim`: a reduction of the stage-start state (a horizontal mean, a domain
integral), a precomputed field its kernel reads. The state is read-only
during the sweep until the predictor overwrites a row, so anything
global goes here. Default nothing.
"""
prepare!(::SweepTerm, sim, Δt) = nothing

"""
    SweepRow

The view of the sweep's tendency buffer on one row that a term's
[`add_tendency!`](@ref) adds into: `tb[i, f]` is the tendency of field
`f` at `i` ([`SWEEP_U`](@ref), `SWEEP_V`, `SWEEP_W`, `SWEEP_Θ`,
`SWEEP_QV`), `tb[i, f] += x` adds to it.
"""
struct SweepRow{B<:AbstractArray}
    buf::B
    a::Int
    b::Int
    s::Int
end
@inline Base.getindex(r::SweepRow, i::Integer, f::Integer) = @inbounds r.buf[i, r.a, r.b, f, r.s]
@inline Base.setindex!(r::SweepRow, x, i::Integer, f::Integer) = (@inbounds r.buf[i, r.a, r.b, f, r.s] = x; r)

"The fields of the sweep's tendency buffer: u, v, w, θ, qv (`SweepRow`)."
const SWEEP_U = 1
@doc (@doc SWEEP_U) const SWEEP_V = 2
@doc (@doc SWEEP_U) const SWEEP_W = 3
@doc (@doc SWEEP_U) const SWEEP_Θ = 4
@doc (@doc SWEEP_U) const SWEEP_QV = 5

"""
    add_tendency!(t::SweepTerm, tb::SweepRow, i, j, k, st)

Add term `t`'s tendency at storage point `(i, j, k)` into `tb`
(`tb[i, SWEEP_W] += …`). `st` is the sweep's state, a NamedTuple of the
storage arrays `u, v, w, θ, qv` (`θ`, `qv` may be `nothing`) and the grid
metrics `m`; nothing in it may be written. `@inline`, no allocation: it
runs inside the row loop of [`add_row!`](@ref).
"""
function add_tendency! end

"""
    add_row!(t::SweepTerm, buf, a, b, s, j, k, ii, st)

Add term `t`'s tendencies on row `(ii, j, k)` into the tendency buffer
`buf[:, a, b, :, s]`: by default a vectorised loop over `ii` of
[`add_tendency!`](@ref). A term that needs a row form of its own (a
recurrence along x, a loop that vectorises better split) overrides it;
keep it `@noinline` with the arrays as arguments (the rule of
`Backends._row!`, docs/src/design/ka_port.md "SC-e").
"""
@noinline function add_row!(t::SweepTerm, buf, a, b, s, j, k, ii, st)
    tb = SweepRow(buf, a, b, s)
    @inbounds @simd ivdep for i in ii
        add_tendency!(t, tb, i, j, k, st)
    end
    return nothing
end

"""
    BaseTendencies(ν)

The base tendencies of the Boussinesq core as the first term of the
sweep, the one that initialises a row: `−∂(uⱼuᵢ)/∂xⱼ + ν∇²uᵢ` for u, v,
w and `−∂(uⱼϕ)/∂xⱼ + ν∇²ϕ` for θ and qv (SC-d's fused expressions,
`_fused_tendencies!`); `ν = nothing` drops the diffusion.
"""
struct BaseTendencies{V} <: SweepTerm
    ν::V
end
footprint(::BaseTendencies) = (1, 1, 1)

# The first term of a sweep writes the row (`=`), the others add to it.
# Only the base tendencies initialise.
@inline init_row!(t::BaseTendencies, buf, a, b, s, j, k, ii, st) =
    _trows!(buf, a, b, s, j, k, ii, st.u, st.v, st.w, st.θ, st.qv, st.m, t.ν)
init_row!(t::SweepTerm, args...) =
    throw(ArgumentError("init_row!: the first term of a sweep initialises the row; $(typeof(t)) only adds to it"))

"""
    sweep_lags(terms::Tuple) -> (lag, lz)

The predictor's lag in j-rows (the ring holds `lag + 1` rows) and the
number of edge planes at each end of a slab of the sweep over `terms`:
the largest y- and z-radii of their [`footprint`](@ref)s, at least 1.
"""
function sweep_lags(terms::Tuple)
    fps = map(footprint, terms)
    return max(1, maximum(f -> f[2], fps)), max(1, maximum(f -> f[3], fps))
end

# ---------------------------------------------------------------------------
# buffers and slabs
# ---------------------------------------------------------------------------

"""
    SweepBuffers

The scratch of [`blocked_update!`](@ref) for a sweep cut into `ns` slabs
in k and `nt` row ranges in j, `nb = ns·nt` blocks (block `n = (t − 1)·ns
+ s`), `nf` fields:

  * `ring[i, kl, r, f, n]`: the new values of block `n`'s inner planes
    (`kl = k − k0 + 1`) on the `lag + 1` ring rows `r ≤ lag + 1`, and on
    its `2ly` edge rows in j (`r = lag + 1 + d`, `d ≤ ly` the first and
    `d > ly` the last `ly` rows of its row range; `ly = 0` when `nt = 1`);
  * `edge[i, j, e, f, s]`: those of slab `s`'s first (`e ≤ lz`) and last
    (`e > lz`) `lz` planes, every row.
"""
mutable struct SweepBuffers{A<:AbstractArray}    # mutable: kept in an `Any` slot without a box per step
    ring::A
    edge::A
    ns::Int
    nf::Int
    lag::Int
    lz::Int
    nt::Int
    ly::Int
end

"""
    sweep_buffers(old, data, nf, ns, nk; lag = 1, lz = 1, nt = 1, ly = 1) -> SweepBuffers

Buffers for `ns` slabs of a k-range of length `nk` and `nt` row ranges in
j, `nf` fields shaped like the storage array `data`, a ring of `lag + 1`
rows, `lz` edge planes per slab end and, when `nt > 1`, `ly` edge rows
per row-range end ([`sweep_lags`](@ref), [`edge_rows`](@ref)); `old` (a
`SweepBuffers` or `nothing`) is returned when it already fits.
"""
function sweep_buffers(old, data::AbstractArray{T,3}, nf::Int, ns::Int, nk::Int;
                       lag::Int = 1, lz::Int = 1, nt::Int = 1, ly::Int = 1) where {T}
    # (the split keeps a cached `old::Any` from a dynamic call: no allocation)
    return old isa SweepBuffers{_buf_type(data)} ? _sweep_buffers(old, data, nf, ns, nk, lag, lz, nt, ly) :
                                                   _sweep_buffers(nothing, data, nf, ns, nk, lag, lz, nt, ly)
end
function _sweep_buffers(old, data::AbstractArray{T,3}, nf::Int, ns::Int, nk::Int, lag::Int, lz::Int,
                        nt::Int, ly::Int) where {T}
    (lag >= 1 && lz >= 1 && ly >= 1) ||
        throw(ArgumentError("sweep_buffers: lag, lz and ly are ≥ 1, got $lag, $lz, $ly"))
    (ns >= 1 && nt >= 1) || throw(ArgumentError("sweep_buffers: ns and nt are ≥ 1, got $ns, $nt"))
    lye = nt > 1 ? ly : 0
    old isa SweepBuffers && old.ns == ns && old.nt == nt && old.nf == nf && old.lag == lag &&
        old.lz == lz && old.ly == lye &&
        size(old.edge, 1) == size(data, 1) && size(old.edge, 2) == size(data, 2) &&
        size(old.ring, 2) >= cld(nk, ns) && return old
    ring = similar(data, size(data, 1), cld(nk, ns), lag + 1 + 2lye, nf, ns * nt)
    edge = similar(data, size(data, 1), size(data, 2), 2lz, nf, ns)
    return SweepBuffers(ring, edge, ns, nf, lag, lz, nt, lye)
end

# the buffers' array type (a concrete type, so `sweep_buffers` infers)
_buf_type(::Array{T,3}) where {T} = Array{T,5}
_buf_type(a::AbstractArray) = typeof(similar(a, 0, 0, 0, 0, 0))

"""
    sweep_blocks(backend, npoints, nk, nj; lz = 1, ly = 1) -> (ns, nt)

The blocks of the blocked sweep (#558): `ns` slabs in k (at least `2lz`
planes each) times `nt` row ranges in j (at least `2ly` rows each), one
work item each, when a launch of `npoints` runs threaded; else `(1, 1)`.

A slab keeps `lz` edge planes at each end and a row range `ly` edge rows:
their new values wait in a buffer until every block is done, then a
second launch writes them. In k alone (#533, #554) the slabs capped the
threads that work at `nk ÷ 2lz`: with 60 levels 30 for the predictor
(`lz = 1`), 15 for the centred TKE update (`lz = 2`) and 7 for WENO-Z5
(`lz = 4`), where 8 of every 10 planes of a slab were edge planes. The
row ranges lift the cap to `(nk ÷ 2lz)(nj ÷ 2ly)`.

The rule, from measurements on 200 × 200 × 60 at 6, 12 and 24 threads
(docs/src/design/roadmap.md, #558): slabs no taller than 20 planes
(`_SLAB_PLANES`) where the thread count allows (taller slabs keep too
many rows of every field in cache: whole 60-level columns ran 10-110 %
slower), then the least estimated time `cld(ns·nt, T) / (ns·nt) · (1 +
c·f_edge)` for `T` threads, a fraction `f_edge` of edge points and `c =
0.5` (`_C_EDGE`): the most blocks up to one per thread, and among those
the fewest edge points.
"""
function sweep_blocks(b, npoints::Integer, nk::Integer, nj::Integer; lz::Integer = 1, ly::Integer = 1)
    Backends.runs_serially(b, npoints) && return (1, 1)
    return _split_blocks(Threads.nthreads(), nk, nj, lz, ly)
end

# the tallest slab the split prefers, and the weight of an edge point
# against an inner one (measured, #558)
const _SLAB_PLANES = 20
const _C_EDGE = 0.5

function _split_blocks(T::Integer, nk::Integer, nj::Integer, lz::Integer, ly::Integer)
    nsmax = max(1, min(T, nk ÷ (2lz)))
    nsmin = min(nsmax, cld(nk, _SLAB_PLANES))
    best, bc = (nsmax, 1), Inf
    for ns in nsmin:nsmax, nt in 1:max(1, min(T ÷ ns, nj ÷ (2ly)))
        fe = 1 - (1 - (ns > 1 ? min(1, 2lz * ns / nk) : 0.0)) * (1 - (nt > 1 ? min(1, 2ly * nt / nj) : 0.0))
        c = cld(ns * nt, T) / (ns * nt) * (1 + _C_EDGE * fe)
        c < bc * (1 - 1e-9) && ((best, bc) = ((ns, nt), c))
    end
    return best
end

# the part s of ns of the range kk (slabs in k, row ranges in j)
@inline function _slab(kk, ns, s)
    nk = length(kk)
    return first(kk) + ((s - 1) * nk) ÷ ns, first(kk) + (s * nk) ÷ ns - 1
end

# the edge slot of plane k of the slab k0:k1 (0: an inner plane): the
# first lz planes are slots 1:lz, the last lz planes slots lz+1:2lz (a
# slab shorter than 2lz, the one slab of a short column: front first).
# The same for the edge rows of a row range (ly, j0:j1).
@inline _edge_slot(k, k0, k1, lz) = k - k0 < lz ? k - k0 + 1 : k1 - k < lz ? 2lz - (k1 - k) : 0

# ---------------------------------------------------------------------------
# the row kernels
# ---------------------------------------------------------------------------

# the predictor of one value: `f(y, x)`, or `f(y, x, snap)` with a
# step-start array (the RK3 stage combination of a scalar)
@inline _pred(f, y, x, ::Nothing, i, j, k) = f(y, x)
@inline _pred(f, y, x, snap::AbstractArray, i, j, k) = @inbounds f(y, x, snap[i, j, k])

# One row (j, k) of base tendencies into buf[:, a, b, f, s], and the
# predictor of one row from it. Each loop is its own @noinline function
# with the arrays as separate arguments (as `Backends._row!`: LLVM then
# hoists the array loads and vectorises); `ivdep`: within a row no store
# feeds a load (the tendencies only read the state, the predictor reads and
# writes one point).
@noinline function _trow_mom!(buf, a, b, s, j, k, ii, u, v, w, m, ν)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, 1, s] = _tend_u(i, j, k, u, v, w, m, ν)
        buf[i, a, b, 2, s] = _tend_v(i, j, k, u, v, w, m, ν)
        buf[i, a, b, 3, s] = _tend_w(i, j, k, u, v, w, m, ν)
    end
    return nothing
end
@noinline function _trow_ϕ!(buf, a, b, f, s, j, k, ii, u, v, w, ϕ, m, ν)
    @inbounds @simd ivdep for i in ii
        buf[i, a, b, f, s] = _tend_ϕ(i, j, k, u, v, w, ϕ, m, ν)
    end
    return nothing
end
@noinline function _prow_mom!(u, v, w, buf, a, b, s, j, k, ii, pu)
    @inbounds @simd ivdep for i in ii
        u[i, j, k] = pu(u[i, j, k], buf[i, a, b, 1, s])
        v[i, j, k] = pu(v[i, j, k], buf[i, a, b, 2, s])
        w[i, j, k] = pu(w[i, j, k], buf[i, a, b, 3, s])
    end
    return nothing
end
@noinline function _prow_ϕ!(ϕ, buf, a, b, f, s, j, k, ii, ps, snap)
    @inbounds @simd ivdep for i in ii
        ϕ[i, j, k] = _pred(ps, ϕ[i, j, k], buf[i, a, b, f, s], snap, i, j, k)
    end
    return nothing
end
@inline _trow_sc!(buf, a, b, f, s, j, k, ii, u, v, w, ::Nothing, m, ν) = nothing
@inline _trow_sc!(buf, a, b, f, s, j, k, ii, u, v, w, ϕ::AbstractArray, m, ν) =
    _trow_ϕ!(buf, a, b, f, s, j, k, ii, u, v, w, ϕ, m, ν)
@inline _prow_sc!(buf, a, b, f, s, j, k, ii, ::Nothing, p) = nothing
@inline _prow_sc!(buf, a, b, f, s, j, k, ii, ϕ::AbstractArray, p::Tuple) =
    _prow_ϕ!(ϕ, buf, a, b, f, s, j, k, ii, p[1], p[2])

@inline function _trows!(buf, a, b, s, j, k, ii, u, v, w, θ, qv, m, ν)
    _trow_mom!(buf, a, b, s, j, k, ii, u, v, w, m, ν)
    _trow_sc!(buf, a, b, 4, s, j, k, ii, u, v, w, θ, m, ν)
    _trow_sc!(buf, a, b, 5, s, j, k, ii, u, v, w, qv, m, ν)
end

# the composed row: the first term initialises, the others add, in order
# (recursion over the tuple: every call static)
@inline function _term_rows!(buf, a, b, s, j, k, ii, terms::Tuple, st)
    init_row!(first(terms), buf, a, b, s, j, k, ii, st)
    _add_rows!(Base.tail(terms), buf, a, b, s, j, k, ii, st)
    return nothing
end
@inline _add_rows!(::Tuple{}, buf, a, b, s, j, k, ii, st) = nothing
@inline function _add_rows!(ts::Tuple, buf, a, b, s, j, k, ii, st)
    add_row!(first(ts), buf, a, b, s, j, k, ii, st)
    return _add_rows!(Base.tail(ts), buf, a, b, s, j, k, ii, st)
end

# the predictor of one row: u, v, w by `pu`, each scalar by its `(ps, snap)`
@inline function _prows!(buf, a, b, s, j, k, ii, st, pu, p1, p2)
    _prow_mom!(st.u, st.v, st.w, buf, a, b, s, j, k, ii, pu)
    _prow_sc!(buf, a, b, 4, s, j, k, ii, st.θ, p1)
    _prow_sc!(buf, a, b, 5, s, j, k, ii, st.qv, p2)
end

# ---------------------------------------------------------------------------
# the driver, generic over the update it runs (#554)
# ---------------------------------------------------------------------------

"""
    BlockedUpdate

The supertype of an update that [`blocked_update!`](@ref) runs over the
interior, block by block (a slab of planes times a range of j-rows, a
[`SweepBlock`](@ref)) and j-row by j-row: the new values of a row are
computed while its neighbourhood is in cache, kept in the driver's ring
(or, on a block's edge planes and edge rows, its edge buffers) while
later rows still read the old values, and written back `lag` rows later
(#533, #554, #558). Two updates use it: the predictor of a Boussinesq
substage ([`blocked_predictor!`](@ref)) and the update of the SGS TKE
(`PrognosticSGS`, `advance_sgs!` of a `TKEClosure`). An update implements

  * `update_lags(op) -> (lag, lz)`: rows between computing a row and
    writing it back (no row computed later reads the old values of row
    `j − lag`), and the edge planes at each end of a slab (a slab reads
    the old values of at most `lz` planes of its neighbours), both ≥ 1;
  * `edge_rows(op) -> ly`: the rows of its neighbours a row range reads
    the old values of (default `lag`);
  * `compute_rows!(op, ring, edge, blk, j, ii)`: the new values of row `j`
    of block `blk`, plane `k` into the slot
    [`block_slot`](@ref)`(ring, edge, blk, j, k)`;
  * `write_row!(op, buf, a, b, n, j, k, ii)`: row `(j, k)` from
    `buf[:, a, b, :, n]` into the fields;
  * `write_point!(op, edge, i, j, k, e, s)`: one point of an edge plane;
  * `begin_block!(op, blk, ii)`: work at the start of a block (default
    none), e.g. the rows a pipelined update needs before the first.
"""
abstract type BlockedUpdate end

"""
    SweepBlock

One block of [`blocked_update!`](@ref): number `n` (its ring), slab `s`
(its edge planes), planes `k0:k1`, rows `j0:j1`, the ring lag `lag`, `lz`
edge planes per slab end and `ly` edge rows per row-range end (0 when the
sweep is not split in j).
"""
struct SweepBlock
    n::Int
    s::Int
    k0::Int
    k1::Int
    j0::Int
    j1::Int
    lag::Int
    lz::Int
    ly::Int
end

"""
    update_lags(op::BlockedUpdate) -> (lag, lz)

The ring lag (in j-rows) and the edge planes per slab end of the update
`op` ([`BlockedUpdate`](@ref)).
"""
function update_lags end

"""
    edge_rows(op::BlockedUpdate) -> Int

The edge rows per row-range end of the update `op` when the sweep is
split in j: the j-radius of the old values a new value reads. The
default, the ring lag, holds for an update that writes a row back as
soon as no later row reads it ([`BlockedUpdate`](@ref)).
"""
edge_rows(op::BlockedUpdate) = update_lags(op)[1]

"""
    begin_block!(op::BlockedUpdate, blk::SweepBlock, ii)

The work of update `op` at the start of block `blk` before its first row
([`BlockedUpdate`](@ref)). Default none.
"""
begin_block!(::BlockedUpdate, blk::SweepBlock, ii) = nothing

"""
    compute_rows!(op::BlockedUpdate, ring, edge, blk::SweepBlock, j, ii)

The new values of row `j` (planes `blk.k0:blk.k1`) of block `blk` into
the driver's buffers, each plane into its [`block_slot`](@ref)
([`BlockedUpdate`](@ref)).
"""
function compute_rows! end

"""
    write_row!(op::BlockedUpdate, buf, a, b, n, j, k, ii)
    write_point!(op::BlockedUpdate, edge, i, j, k, e, s)

Write the new values of row `(j, k)` from `buf[:, a, b, :, n]` (the ring
or an edge row), or of the edge-plane point `(i, j, k)` from
`edge[i, j, e, :, s]`, into the fields ([`BlockedUpdate`](@ref)).
"""
function write_row! end
@doc (@doc write_row!) function write_point! end

"""
    block_slot(ring, edge, blk::SweepBlock, j, k) -> (buf, a, b, n)

Where the new values of row `(j, k)` of block `blk` go,
`buf[:, a, b, :, n]`: on an edge plane the slab's edge buffer
(`edge[:, j, e, :, s]`), on an edge row the block's edge rows in the ring
(`ring[:, k − k0 + 1, lag + 1 + d, :, n]`), else the ring row
`mod(j, lag + 1) + 1`.
"""
@inline function block_slot(ring, edge, blk::SweepBlock, j, k)
    e = _edge_slot(k, blk.k0, blk.k1, blk.lz)
    e > 0 && return (edge, j, e, blk.s)
    d = blk.ly > 0 ? _edge_slot(j, blk.j0, blk.j1, blk.ly) : 0
    return (ring, k - blk.k0 + 1, d > 0 ? blk.lag + 1 + d : mod(j, blk.lag + 1) + 1, blk.n)
end

@inline function _block(n, kk, jj, ns, nt, lag, lz, ly)
    s = (n - 1) % ns + 1
    k0, k1 = _slab(kk, ns, s)
    j0, j1 = _slab(jj, nt, (n - 1) ÷ ns + 1)
    return SweepBlock(n, s, k0, k1, j0, j1, lag, lz, ly)
end

# block n: the j-row sweep (one launch item per block; blocks write
# disjoint points and read the others' only on their edge planes and edge
# rows, not yet updated)
function _b_sweep_block!(n, op, ring, edge, ii, jj, kk, ns, nt, lag, lz, ly)
    blk = _block(n, kk, jj, ns, nt, lag, lz, ly)
    begin_block!(op, blk, ii)
    for j in blk.j0:blk.j1
        compute_rows!(op, ring, edge, blk, j, ii)
        jp = j - lag
        jp >= blk.j0 && _write_rows!(op, ring, blk, jp, ii)
    end
    for jp in max(blk.j0, blk.j1 - lag + 1):blk.j1
        _write_rows!(op, ring, blk, jp, ii)
    end
    return nothing
end
@inline function _write_rows!(op, ring, blk, jp, ii)
    blk.ly > 0 && _edge_slot(jp, blk.j0, blk.j1, blk.ly) > 0 && return nothing    # an edge row
    rp = mod(jp, blk.lag + 1) + 1
    for k in blk.k0+blk.lz:blk.k1-blk.lz
        write_row!(op, ring, k - blk.k0 + 1, rp, blk.n, jp, k, ii)
    end
    return nothing
end

# the edge planes, after every block: (i, j, m), m = (s − 1)·2lz + e for
# edge slot e of slab s
@inline function _b_sweep_edge!(i, j, m, op, edge, kk, ns, lz)
    s = (m - 1) ÷ (2lz) + 1
    e = (m - 1) % (2lz) + 1
    k0, k1 = _slab(kk, ns, s)
    if e <= lz
        k = k0 + e - 1
        k > k1 && return nothing
    else
        k = k1 - (2lz - e)
        (k < k0 || k - k0 < lz) && return nothing        # a front slot already
    end
    write_point!(op, edge, i, j, k, e, s)
    return nothing
end

# the edge rows, after every block: (kl, d, n), plane k0 + kl − 1 of edge
# row d of block n (the inner planes; an edge plane went to `edge`)
@inline function _b_sweep_yedge!(kl, d, n, op, ring, ii, jj, kk, ns, nt, lag, lz, ly)
    blk = _block(n, kk, jj, ns, nt, lag, lz, ly)
    k = blk.k0 + kl - 1
    (k < blk.k0 + lz || k > blk.k1 - lz) && return nothing
    if d <= ly
        j = blk.j0 + d - 1
        j > blk.j1 && return nothing
    else
        j = blk.j1 - (2ly - d)
        (j < blk.j0 || j - blk.j0 < ly) && return nothing  # a front slot already
    end
    write_row!(op, ring, kl, lag + 1 + d, n, j, k, ii)
    return nothing
end

"""
    blocked_update!(bufs::SweepBuffers, op::BlockedUpdate, b, g; reverse_slabs = false)

Run the update `op` ([`BlockedUpdate`](@ref)) over the interior of the
mesh `g` on backend `b`: the k-range cut into `bufs.ns` slabs and the
j-range into `bufs.nt` row ranges, one launch item per block
([`SweepBlock`](@ref)), each block swept j-row by j-row with its new rows
held in the ring of `bufs` until `update_lags(op)[1]` rows later; then
the slabs' edge planes and the row ranges' edge rows in a second and
third launch. `bufs` ([`sweep_buffers`](@ref)) must have the lags and
edge rows of `op` and as many fields as `op` writes. `reverse_slabs =
true` runs the blocks one after another in reverse order, serially: with
the forward order of one thread, the two orders bound what concurrent
blocks can see (a test hook, #533 review).
"""
function blocked_update!(bufs::SweepBuffers, op::BlockedUpdate, b, g::AbstractMesh;
                         reverse_slabs::Bool = false)
    ii, jj, kk = interior(g)
    ns, nt = bufs.ns, bufs.nt
    lag, lz = update_lags(op)
    ly = nt > 1 ? edge_rows(op) : 0
    bufs.lag == lag && bufs.lz == lz && bufs.ly == ly && size(bufs.ring, 2) >= cld(length(kk), ns) &&
        (ns == 1 || 2lz * ns <= length(kk)) && nt <= length(jj) ||
        throw(DimensionMismatch("blocked_update!: buffers do not fit ($ns slabs over $(length(kk)) " *
                                "planes, $nt row ranges over $(length(jj)) rows, lag $lag, $lz edge " *
                                "planes, $ly edge rows)"))
    nb = ns * nt
    if reverse_slabs
        for n in nb:-1:1
            _b_sweep_block!(n, op, bufs.ring, bufs.edge, ii, jj, kk, ns, nt, lag, lz, ly)
        end
        for m in 1:2lz*ns, j in jj, i in ii
            _b_sweep_edge!(i, j, m, op, bufs.edge, kk, ns, lz)
        end
        for n in 1:nb, d in 1:2ly, kl in 1:size(bufs.ring, 2)
            _b_sweep_yedge!(kl, d, n, op, bufs.ring, ii, jj, kk, ns, nt, lag, lz, ly)
        end
    else
        launch_weighted!(b, cld(length(ii) * length(jj) * length(kk), nb), _b_sweep_block!, (1:nb,),
                         op, bufs.ring, bufs.edge, ii, jj, kk, ns, nt, lag, lz, ly)
        launch!(b, _b_sweep_edge!, (ii, jj, 1:2lz*ns), op, bufs.edge, kk, ns, lz)
        ly > 0 && launch_weighted!(b, length(ii), _b_sweep_yedge!, (1:size(bufs.ring, 2), 1:2ly, 1:nb),
                                   op, bufs.ring, ii, jj, kk, ns, nt, lag, lz, ly)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# the predictor as a blocked update
# ---------------------------------------------------------------------------

# The tendencies of a substage (the terms) and the predictor of u, v, w and
# the scalars, as an update of the blocked driver (SC-e #405, #533).
struct _PredictorUpdate{TT<:Tuple, ST, PU, P1, P2} <: BlockedUpdate
    terms::TT
    st::ST
    pu::PU
    p1::P1
    p2::P2
end
update_lags(op::_PredictorUpdate) = sweep_lags(op.terms)
@inline function compute_rows!(op::_PredictorUpdate, ring, edge, blk::SweepBlock, j, ii)
    for k in blk.k0:blk.k1
        buf, a, b, n = block_slot(ring, edge, blk, j, k)
        _term_rows!(buf, a, b, n, j, k, ii, op.terms, op.st)
    end
    return nothing
end
@inline write_row!(op::_PredictorUpdate, ring, a, b, s, j, k, ii) =
    _prows!(ring, a, b, s, j, k, ii, op.st, op.pu, op.p1, op.p2)
@inline function write_point!(op::_PredictorUpdate, edge, i, j, k, e, s)
    st, pu = op.st, op.pu
    u, v, w = st.u, st.v, st.w
    @inbounds begin
        u[i, j, k] = pu(u[i, j, k], edge[i, j, e, 1, s])
        v[i, j, k] = pu(v[i, j, k], edge[i, j, e, 2, s])
        w[i, j, k] = pu(w[i, j, k], edge[i, j, e, 3, s])
    end
    _edge_sc!(i, j, k, e, 4, s, st.θ, op.p1, edge)
    _edge_sc!(i, j, k, e, 5, s, st.qv, op.p2, edge)
    return nothing
end
@inline _edge_sc!(i, j, k, e, f, s, ::Nothing, p, edge) = nothing
@inline function _edge_sc!(i, j, k, e, f, s, ϕ, (ps, snap), edge)
    @inbounds ϕ[i, j, k] = _pred(ps, ϕ[i, j, k], edge[i, j, e, f, s], snap, i, j, k)
    return nothing
end

_sweep_field(::Nothing) = nothing
_sweep_field((ϕ, ps, snap)::Tuple{Field{CellCenter}, Any, Any}) = ϕ.data
_sweep_pred(::Nothing) = nothing
_sweep_pred((ϕ, ps, snap)::Tuple{Field{CellCenter}, Any, Any}) = (ps, snap)

# every input of every term on the state's backend
_check_inputs(b, ::Tuple{}) = nothing
function _check_inputs(b, terms::Tuple)
    for x in sweep_inputs(first(terms))
        x isa AbstractArray && typeof(Backends.backend(x)) !== typeof(b) &&
            throw(ArgumentError("blocked_predictor!: an input of $(typeof(first(terms))) is on " *
                                "$(typeof(Backends.backend(x))), the state on $(typeof(b))"))
    end
    return _check_inputs(b, Base.tail(terms))
end

"""
    blocked_predictor!(bufs, terms, u, v, w, pu, s1, s2, g; reverse_slabs = false)

The tendencies of a substage and its predictor in one sweep (SC-e #405,
#533): the tendency of each field is the sum of the `terms` (a tuple; the
first, [`BaseTendencies`](@ref), initialises it, every further
[`SweepTerm`](@ref) adds its [`add_tendency!`](@ref) in tuple order), then
`u ← pu(u, tu)` (likewise v, w) and, for each scalar `s = (ϕ, ps, snap)`
(or `nothing`), `ϕ ← ps(ϕ, tϕ)` (`ps(ϕ, tϕ, snap[i, j, k])` when `snap` is
an array), over the interior of `g`. Within round-off the whole-field
passes of the same terms followed by the predictor passes, but the
tendencies stay in `bufs` ([`sweep_buffers`](@ref) with the
[`sweep_lags`](@ref) of `terms`, `ly = lag` when it splits in j, `bufs.nf`
≥ 3 + the number of scalars),
blocked so each field is streamed once. Every halo a term reads must be
filled; nothing else may read or write the fields meanwhile. The sweep is
the [`blocked_update!`](@ref) driver running the predictor
(`reverse_slabs`: its test hook).
"""
function blocked_predictor!(bufs::SweepBuffers, terms::Tuple{BaseTendencies,Vararg{SweepTerm}},
                            u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                            pu, s1, s2, g::AbstractMesh; reverse_slabs::Bool = false)
    ii, jj, kk = interior(g)
    ns = bufs.ns
    lag, lz = sweep_lags(terms)
    nf = 3 + (s1 === nothing ? 0 : 1) + (s2 === nothing ? 0 : 1)
    (s1 === nothing && s2 !== nothing) && throw(ArgumentError("blocked_predictor!: s2 without s1"))
    bufs.nf >= nf && bufs.lag == lag && bufs.lz == lz && size(bufs.ring, 2) >= cld(length(kk), ns) &&
        (ns == 1 || 2lz * ns <= length(kk)) ||
        throw(DimensionMismatch("blocked_predictor!: buffers do not fit ($ns slabs over $(length(kk)) " *
                                "planes, lag $lag, $lz edge planes)"))
    # (the edge rows of a split in j are checked by blocked_update!)
    b = Backends.backend(u.data)
    _check_inputs(b, terms)
    st = (u = u.data, v = v.data, w = w.data, θ = _sweep_field(s1), qv = _sweep_field(s2), m = grid_metrics(g))
    op = _PredictorUpdate(terms, st, pu, _sweep_pred(s1), _sweep_pred(s2))
    blocked_update!(bufs, op, b, g; reverse_slabs = reverse_slabs)
    return nothing
end
