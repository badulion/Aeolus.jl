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
#   * the k-range is cut into `ns` slabs, one task each;
#   * a slab is swept j-row by j-row: the tendencies of row j (every k of
#     the slab) go into a ring of `lag + 1` rows, then row j − lag gets its
#     predictor. `lag` is the largest y-radius of the terms' footprints (1
#     for the base tendencies' j ± 1 stencils): no later tendency of the
#     slab reads row j − lag, so the update can be in place;
#   * the slab's first and last `lz` planes (`lz` the largest z-radius) are
#     read by the neighbouring slabs' stencils, so their tendencies are kept
#     (`edge`) and their predictor runs in a second launch, after every slab
#     is done.
#
# What a row's tendency is, is a compile-time tuple of *terms* (#533):
# the first initialises the row (the base tendencies of SC-d: advection
# and molecular diffusion), each further term adds its contribution in
# the fixed order of the tuple. A term is an immutable object with a
# point kernel (`add_tendency!`) or a row kernel (`add_row!`); everything
# dispatches statically, nothing per point.
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
# post-predictor operators in the sweep (#533 plan step 6)
# ---------------------------------------------------------------------------

"""
    PostTerm

The supertype of a post-predictor operator of the fused sweep (#533): an
immutable object that corrects the predicted velocity ``u^*`` (an implicit
drag, a penalty, a prescribed flow) on a row while the sweep holds it in
cache, as the core's `post_predictor!` hook does on the whole field
before the projection. A post term implements

  * [`footprint`](@ref)`(t) -> (rx, ry, rz)`, the stencil radius over the
    predicted velocity it reads (required);
  * [`post_row!`](@ref)`(t, Val(2), j, k, i1, i2, st)`, the correction of
    the faces `(i1:i2, j, k)` (required), and, for an operator whose
    corrections are computed from the values *before* any is applied
    (Jacobi: every factor from the predicted velocity, then every face
    updated), `post_row!(t, Val(1), …)`, which computes the row's factors
    into the term's own storage without changing the velocity;
  * [`prepare!`](@ref)`(t, sim, Δt)` and [`finish!`](@ref)`(t, sim, Δt)`,
    its work before the sweep and after it (the ordered sums of its
    diagnostics; default none).

The driver runs a term with footprint `(0, 0, 0)` on a row right after
the row's predictor; a term with y-radius `r` computes its factors `r`
rows behind the front of its input and applies them `2r` rows behind, so
every factor reads values its own operator has not changed yet (the
delayed-row scheme). The points whose stencil reaches a halo or a plane
the sweep updates only after all slabs (the slab edges), or a point an
earlier operator had to defer, run after the sweep, in the same two
phases, with the halos of the predicted velocity filled
([`post_plan`](@ref)). An operator that needs the whole predicted field
is an [`AfterSweep`](@ref). Several post terms run in tuple order (the
pipeline's), each on the output of the one before.
"""
abstract type PostTerm end

"""
    post_sweep_term(x, ctx, Δt) -> Union{PostTerm,Nothing}

The post term of the fused sweep that does what `x` (a component of the
`:post_predictor` stage, with `ctx` its domain) does to the predicted
velocity in a substage of length `Δt`: a [`PostTerm`](@ref) that runs on
the sweep's rows, an [`AfterSweep`](@ref) (its whole-field form after the
sweep, when it needs the whole predicted field), or `nothing` (the
default) when it has no fused form yet; the substage then runs one loop
per term and the run reports why (#533).
"""
post_sweep_term(x, ctx, Δt) = nothing

"""
    post_row!(t::PostTerm, phase::Val, j, k, i1, i2, st)

Phase `Val(1)` of post term `t` on the faces `(i1:i2, j, k)` (storage
indices): compute its factors from the velocity in `st` (the sweep's
state, as for [`add_tendency!`](@ref)) into the term's storage, without
writing the velocity; the default does nothing. Phase `Val(2)`: correct
the velocity at those faces (from the factors of phase 1, or pointwise).
A face may only be written by the call of its own row `(j, k)`.
"""
post_row!(::PostTerm, ::Val{1}, j, k, i1, i2, st) = nothing

"""
    finish!(t::PostTerm, sim, Δt)

The work of post term `t` after the sweep of a substage of length `Δt`
on `sim`, when every face has been corrected: the ordered sums of its
diagnostics (forces, stress statistics) from the per-face values its
[`post_row!`](@ref) stored. Default nothing.
"""
finish!(::PostTerm, sim, Δt) = nothing
prepare!(::PostTerm, sim, Δt) = nothing
footprint(t::PostTerm) = throw(ArgumentError("footprint: the post term $(typeof(t)) declares no footprint"))

"""
    row_range(key, r::UnitRange, (k, j)) -> UnitRange

The entries `n ∈ r` of a sparse list sorted by `(k, j)` (then by `i`)
whose `key(n) == (k, j)`: the faces of one row of a post term's face list,
found by bisection (two searches, no index stored).
"""
@inline function row_range(key::F, r::UnitRange{Int}, kj::Tuple{Int,Int}) where {F}
    lo, hi = first(r), last(r) + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        key(mid) < kj ? (lo = mid + 1) : (hi = mid)
    end
    a = lo
    hi = last(r) + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        key(mid) <= kj ? (lo = mid + 1) : (hi = mid)
    end
    return a:lo-1
end

"""
    AfterSweep(f)

A post-predictor operator that needs the whole predicted field (a plane
mean of ``u^*``, a solve across the domain): `f()` runs after the sweep,
when every row has its predictor, before the projection. The substage
stays fused; every post term after an `AfterSweep` runs after it too, on
the whole field (pipeline order is kept).
"""
struct AfterSweep{F} <: PostTerm
    f::F
end

# a width no index distance reaches: the operator runs after the sweep only
const _AFTER = typemax(Int) >> 4

"""
    post_plan(post::Tuple, lz) -> NTuple{N,NTuple{8,Int}}

The schedule of the post terms `post` in a sweep whose slabs keep `lz`
edge planes: per term `(fx, fy, fz, ax, ay, az, df, da)`. A point runs
phase 1 (its factors) inside the sweep when its index distance to the
x- and y-boundaries exceeds `fx`, `fy` and its distance to its slab's
first and last plane exceeds `fz` (phase 2: `ax, ay, az`), else after
the sweep. Phase 1 runs `df` rows behind the predictor, phase 2 `da`.
The widths grow by the term's radius over those of the previous term's
phase 2 (the halos: 0; the slab edges: `lz`), so a stencil inside the
sweep reads only values that are final for the previous term and not yet
changed by its own; the delays grow by `r` and `2r` rows.
"""
post_plan(post::Tuple, lz::Integer) = _post_plan(post, 0, 0, Int(lz), 0)
_post_plan(::Tuple{}, wx, wy, wz, d) = ()
function _post_plan(post::Tuple, wx, wy, wz, d)
    e = _plan_entry(first(post), wx, wy, wz, d)
    return (e, _post_plan(Base.tail(post), e[4], e[5], e[6], e[8])...)
end
_plan_entry(::AfterSweep, wx, wy, wz, d) = (_AFTER, _AFTER, _AFTER, _AFTER, _AFTER, _AFTER, d, d)
function _plan_entry(t::PostTerm, wx, wy, wz, d)
    wx >= _AFTER && return (_AFTER, _AFTER, _AFTER, _AFTER, _AFTER, _AFTER, d, d)
    rx, ry, rz = footprint(t)
    fx, fy, fz = wx + rx, wy + ry, wz + rz
    return (fx, fy, fz, fx + rx, fy + ry, fz + rz, d + ry, d + 2ry)
end

"The rows the last post term of `plan` runs behind the predictor (0 without post terms)."
post_delay(::Tuple{}) = 0
post_delay(plan::Tuple) = last(plan)[8]

# the post terms on their rows behind the predictor row F of a slab k0:k1
@inline _post_rows!(::Tuple{}, ::Tuple{}, F, k0, k1, ii, jj, st) = nothing
@inline function _post_rows!(post::Tuple, plan::Tuple, F, k0, k1, ii, jj, st)
    t = first(post)
    fx, fy, fz, ax, ay, az, df, da = first(plan)
    _post_phase!(t, Val(1), F - df, fx, fy, fz, k0, k1, ii, jj, st)
    _post_phase!(t, Val(2), F - da, ax, ay, az, k0, k1, ii, jj, st)
    return _post_rows!(Base.tail(post), Base.tail(plan), F, k0, k1, ii, jj, st)
end
@inline _post_phase!(::AfterSweep, ph, j, wx, wy, wz, k0, k1, ii, jj, st) = nothing
@inline function _post_phase!(t, ph, j, wx, wy, wz, k0, k1, ii, jj, st)
    (j - first(jj) >= wy && last(jj) - j >= wy) || return nothing
    i1, i2 = first(ii) + wx, last(ii) - wx
    i1 <= i2 || return nothing
    for k in k0+wz:k1-wz
        post_row!(t, ph, j, k, i1, i2, st)
    end
    return nothing
end

# the slab of plane k
@inline function _slab_of(k, kk, ns)
    s = clamp(((k - first(kk)) * ns) ÷ length(kk) + 1, 1, ns)
    while k < first(_slab(kk, ns, s)); s -= 1; end
    while k > last(_slab(kk, ns, s)); s += 1; end
    return s
end

# After the sweep, phase `ph` of a post term on the points the sweep left
# to it (outside the widths of `post_plan`): row (j, k) whole, or its x-ends.
@inline function _post_deferred_row!(j, k, t, ph, wx, wy, wz, st, ii, jj, kk, ns)
    k0, k1 = _slab(kk, ns, _slab_of(k, kk, ns))
    inner = j - first(jj) >= wy && last(jj) - j >= wy && k - k0 >= wz && k1 - k >= wz
    i1, i2 = first(ii) + wx, last(ii) - wx
    if !inner || i1 > i2
        post_row!(t, ph, j, k, first(ii), last(ii), st)
    elseif wx > 0
        post_row!(t, ph, j, k, first(ii), i1 - 1, st)
        post_row!(t, ph, j, k, i2 + 1, last(ii), st)
    end
    return nothing
end

# every post term's points that ran outside the sweep, in order: the halos
# of the predicted velocity first (`halo!()`) for a term that reads beyond
# its own face, then phase 1 and phase 2 over its deferred points; an
# AfterSweep runs its whole-field operator
_post_deferred!(b, ::Tuple{}, ::Tuple{}, halo!, st, ii, jj, kk, ns) = nothing
function _post_deferred!(b, post::Tuple, plan::Tuple, halo!::H, st, ii, jj, kk, ns) where {H}   # (H: specialise on a callable struct)
    _post_deferred1!(b, first(post), first(plan), halo!, st, ii, jj, kk, ns)
    return _post_deferred!(b, Base.tail(post), Base.tail(plan), halo!, st, ii, jj, kk, ns)
end
_post_deferred1!(b, t::AfterSweep, e, halo!, st, ii, jj, kk, ns) = (t.f(); nothing)
function _post_deferred1!(b, t::PostTerm, e, halo!::H, st, ii, jj, kk, ns) where {H}
    fx, fy, fz, ax, ay, az = e
    footprint(t) == (0, 0, 0) || halo!()
    launch!(b, _post_deferred_row!, (jj, kk), t, Val(1), fx, fy, fz, st, ii, jj, kk, ns)
    launch!(b, _post_deferred_row!, (jj, kk), t, Val(2), ax, ay, az, st, ii, jj, kk, ns)
    return nothing
end

# ---------------------------------------------------------------------------
# buffers and slabs
# ---------------------------------------------------------------------------

"""
    SweepBuffers

The scratch of [`blocked_predictor!`](@ref): `ring[i, kl, r, f, s]`, the
tendencies of field `f` on `lag + 1` j-rows (`r`) of slab `s`'s inner
planes (`kl = k − k0 + 1`), and `edge[i, j, e, f, s]`, those of its first
(`e ≤ lz`) and last (`e > lz`) `lz` planes. `ns` slabs, `nf` fields.
"""
mutable struct SweepBuffers{A<:AbstractArray}    # mutable: kept in an `Any` slot without a box per step
    ring::A
    edge::A
    ns::Int
    nf::Int
    lag::Int
    lz::Int
end

"""
    sweep_buffers(old, data, nf, ns, nk; lag = 1, lz = 1) -> SweepBuffers

Buffers for `ns` slabs of a k-range of length `nk`, `nf` fields shaped like
the storage array `data`, a ring of `lag + 1` rows and `lz` edge planes
per slab end ([`sweep_lags`](@ref)); `old` (a `SweepBuffers` or
`nothing`) is returned when it already fits.
"""
function sweep_buffers(old, data::AbstractArray{T,3}, nf::Int, ns::Int, nk::Int;
                       lag::Int = 1, lz::Int = 1) where {T}
    # (the split keeps a cached `old::Any` from a dynamic call: no allocation)
    return old isa SweepBuffers{_buf_type(data)} ? _sweep_buffers(old, data, nf, ns, nk, lag, lz) :
                                                   _sweep_buffers(nothing, data, nf, ns, nk, lag, lz)
end
function _sweep_buffers(old, data::AbstractArray{T,3}, nf::Int, ns::Int, nk::Int, lag::Int, lz::Int) where {T}
    (lag >= 1 && lz >= 1) || throw(ArgumentError("sweep_buffers: lag and lz are ≥ 1, got $lag, $lz"))
    old isa SweepBuffers && old.ns == ns && old.nf == nf && old.lag == lag && old.lz == lz &&
        size(old.edge, 1) == size(data, 1) && size(old.edge, 2) == size(data, 2) &&
        size(old.ring, 2) >= cld(nk, ns) && return old
    ring = similar(data, size(data, 1), cld(nk, ns), lag + 1, nf, ns)
    edge = similar(data, size(data, 1), size(data, 2), 2lz, nf, ns)
    return SweepBuffers(ring, edge, ns, nf, lag, lz)
end

# the buffers' array type (a concrete type, so `sweep_buffers` infers)
_buf_type(::Array{T,3}) where {T} = Array{T,5}
_buf_type(a::AbstractArray) = typeof(similar(a, 0, 0, 0, 0, 0))

"""
    sweep_slabs(backend, npoints, nk; lz = 1) -> Int

The number of slabs of the blocked sweep: one per thread (at least `2lz`
planes each) when a launch of `npoints` runs threaded, else one.
"""
sweep_slabs(b, npoints::Integer, nk::Integer; lz::Integer = 1) =
    Backends.runs_serially(b, npoints) ? 1 : clamp(Threads.nthreads(), 1, max(1, nk ÷ (2lz)))

@inline function _slab(kk, ns, s)
    nk = length(kk)
    return first(kk) + ((s - 1) * nk) ÷ ns, first(kk) + (s * nk) ÷ ns - 1
end

# the edge slot of plane k of the slab k0:k1 (0: an inner plane): the
# first lz planes are slots 1:lz, the last lz planes slots lz+1:2lz (a
# slab shorter than 2lz, the one slab of a short column: front first)
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
# the sweep
# ---------------------------------------------------------------------------

# slab s: the j-row sweep (one launch item per slab; slabs write disjoint
# planes and read the others' only on their edge planes, not yet updated).
# Front t: the tendencies of row t, the predictor of row t − lag, then the
# post-predictor operators `post` on their rows behind it (`plan`,
# [`post_plan`](@ref)); the front runs `lag` + the operators' delay past
# the last row, so every row gets its predictor and its operators.
_b_sweep_slab!(s, terms, st, pu, p1, p2, ring, edge, ii, jj, kk, ns, lag, lz) =
    _b_sweep_slab!(s, terms, (), (), st, pu, p1, p2, ring, edge, ii, jj, kk, ns, lag, lz)
function _b_sweep_slab!(s, terms, post, plan, st, pu, p1, p2, ring, edge, ii, jj, kk, ns, lag, lz)
    k0, k1 = _slab(kk, ns, s)
    nr = lag + 1
    for t in first(jj):last(jj)+lag+post_delay(plan)
        if t <= last(jj)
            r = mod(t, nr) + 1
            for k in k0:k1
                e = _edge_slot(k, k0, k1, lz)
                if e > 0
                    _term_rows!(edge, t, e, s, t, k, ii, terms, st)
                else
                    _term_rows!(ring, k - k0 + 1, r, s, t, k, ii, terms, st)
                end
            end
        end
        jp = t - lag
        if first(jj) <= jp <= last(jj)
            rp = mod(jp, nr) + 1
            for k in k0+lz:k1-lz
                _prows!(ring, k - k0 + 1, rp, s, jp, k, ii, st, pu, p1, p2)
            end
        end
        _post_rows!(post, plan, jp, k0, k1, ii, jj, st)
    end
    return nothing
end

# the edge planes, after every slab: (i, j, n), n = (s − 1)·2lz + e for
# edge slot e of slab s
@inline function _b_sweep_edge!(i, j, n, st, pu, p1, p2, edge, kk, ns, lz)
    s = (n - 1) ÷ (2lz) + 1
    e = (n - 1) % (2lz) + 1
    k0, k1 = _slab(kk, ns, s)
    if e <= lz
        k = k0 + e - 1
        k > k1 && return nothing
    else
        k = k1 - (2lz - e)
        (k < k0 || k - k0 < lz) && return nothing        # a front slot already
    end
    u, v, w = st.u, st.v, st.w
    @inbounds begin
        u[i, j, k] = pu(u[i, j, k], edge[i, j, e, 1, s])
        v[i, j, k] = pu(v[i, j, k], edge[i, j, e, 2, s])
        w[i, j, k] = pu(w[i, j, k], edge[i, j, e, 3, s])
    end
    _edge_sc!(i, j, k, e, 4, s, st.θ, p1, edge)
    _edge_sc!(i, j, k, e, 5, s, st.qv, p2, edge)
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
    blocked_predictor!(bufs, terms, [post, halo!,] u, v, w, pu, s1, s2, g)

The tendencies of a substage and its predictor in one sweep (SC-e #405,
#533): the tendency of each field is the sum of the `terms` (a tuple; the
first, [`BaseTendencies`](@ref), initialises it, every further
[`SweepTerm`](@ref) adds its [`add_tendency!`](@ref) in tuple order), then
`u ← pu(u, tu)` (likewise v, w) and, for each scalar `s = (ϕ, ps, snap)`
(or `nothing`), `ϕ ← ps(ϕ, tϕ)` (`ps(ϕ, tϕ, snap[i, j, k])` when `snap` is
an array), over the interior of `g`. Within round-off the whole-field
passes of the same terms followed by the predictor passes, but the
tendencies stay in `bufs` ([`sweep_buffers`](@ref) with the
[`sweep_lags`](@ref) of `terms`, `bufs.nf` ≥ 3 + the number of scalars),
blocked so each field is streamed once. Every halo a term reads must be
filled; nothing else may read or write the fields meanwhile.

`post` (default none) is a tuple of [`PostTerm`](@ref)s, the core's
post-predictor operators: inside the sweep on the rows behind the
predictor, the rest after it ([`post_plan`](@ref)); `halo!()` fills the
halos of the predicted u, v, w for the points run after the sweep. Their
[`prepare!`](@ref) and [`finish!`](@ref) are the caller's.
"""
blocked_predictor!(bufs::SweepBuffers, terms::Tuple{BaseTendencies,Vararg{SweepTerm}},
                   u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, pu, s1, s2, g::AbstractMesh) =
    blocked_predictor!(bufs, terms, (), nothing, u, v, w, pu, s1, s2, g)
function blocked_predictor!(bufs::SweepBuffers, terms::Tuple{BaseTendencies,Vararg{SweepTerm}},
                            post::Tuple{Vararg{PostTerm}}, halo!::H,
                            u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                            pu, s1, s2, g::AbstractMesh) where {H}
    ii, jj, kk = interior(g)
    ns = bufs.ns
    lag, lz = sweep_lags(terms)
    nf = 3 + (s1 === nothing ? 0 : 1) + (s2 === nothing ? 0 : 1)
    (s1 === nothing && s2 !== nothing) && throw(ArgumentError("blocked_predictor!: s2 without s1"))
    bufs.nf >= nf && bufs.lag == lag && bufs.lz == lz && size(bufs.ring, 2) >= cld(length(kk), ns) &&
        (ns == 1 || 2lz * ns <= length(kk)) ||
        throw(DimensionMismatch("blocked_predictor!: buffers do not fit ($ns slabs over $(length(kk)) " *
                                "planes, lag $lag, $lz edge planes)"))
    b = Backends.backend(u.data)
    _check_inputs(b, terms)
    st = (u = u.data, v = v.data, w = w.data, θ = _sweep_field(s1), qv = _sweep_field(s2), m = grid_metrics(g))
    p1, p2 = _sweep_pred(s1), _sweep_pred(s2)
    plan = post_plan(post, lz)
    launch_weighted!(b, cld(length(ii) * length(jj) * length(kk), ns), _b_sweep_slab!, (1:ns,),
                     terms, post, plan, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz)
    launch!(b, _b_sweep_edge!, (ii, jj, 1:2lz*ns), st, pu, p1, p2, bufs.edge, kk, ns, lz)
    _post_deferred!(b, post, plan, halo!, st, ii, jj, kk, ns)
    return nothing
end
