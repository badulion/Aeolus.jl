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
#     ranges, one task per block (a slab times a row range; #566, the split
#     in j of #558);
#   * a block is swept j-row by j-row: the tendencies of row j (every k of
#     the slab) go into a ring of `lag + 1` rows, then row j − lag gets its
#     predictor. `lag` is the largest y-radius of the terms' footprints (1
#     for the base tendencies' j ± 1 stencils): no later tendency of the
#     block reads row j − lag, so the update can be in place;
#   * the slab's first and last `lz` planes (`lz` the largest z-radius) are
#     read by the neighbouring slabs' stencils, so their tendencies are kept
#     (`edge`) and their predictor runs in a second launch, after every
#     block is done; likewise the first and last `ly = lag` rows of a row
#     range, read by the neighbouring row ranges (kept in extra rows of the
#     ring, their predictor in a third launch).
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
# so the result does not depend on the number of blocks, of threads, or on
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
  * [`add_tendency!`](@ref)`(t, tb, i, j, k, st)`, its point kernel
    (required: the per-point form of the sweep for a GPU,
    [`pointwise_predictor!`](@ref), calls it), and optionally
    [`add_row!`](@ref), a row form of its own for the CPU's ring-buffer
    sweep;
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
For a component it is everything the component adds in the in-core
stages `:tendencies` and `:body_force` together (one attached to both
gets it switched on once); it may be asked at a stage the component does
not attach to, so it must not assume one.
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
    finish_sweep!(t::SweepTerm, sim, Δt)

The work of term `t` after the sweep of a substage of length `Δt` on
`sim`, before the projection: the reduction of what its kernel
accumulated during the sweep (the budgets' per-level sums, #533 plan
step 7). Default nothing.
"""
finish_sweep!(::SweepTerm, sim, Δt) = nothing

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
    BaseTerm <: SweepTerm

A term that starts a row of the fused sweep: the base tendencies of a step
core (the advection and molecular diffusion), the first term of every
term tuple. It writes the row (`=`) where every other term adds to it, by
`init_row!(t, buf, a, b, s, j, k, ii, st)` (the ring-buffer sweep) and
`init_point!(t, tb, i, j, k, st)` (the per-point form). The Boussinesq
core's is [`BaseTendencies`](@ref); the sound-proof cores' is
`Anelastic.WeightedBaseTendencies` (the `W`-weighted advection, #533
plan step 8).
"""
abstract type BaseTerm <: SweepTerm end

"""
    init_point!(t::BaseTerm, tb, i, j, k, st)

Write the base term `t`'s tendencies at storage point `(i, j, k)` into
`tb` (`tb[i, SWEEP_U] = …`): the point form of `init_row!`, used by the
per-point (GPU) form of the sweep, [`pointwise_predictor!`](@ref).
"""
function init_point! end

"""
    BaseTendencies(ν)

The base tendencies of the Boussinesq core as the first term of the
sweep, the one that initialises a row: `−∂(uⱼuᵢ)/∂xⱼ + ν∇²uᵢ` for u, v,
w and `−∂(uⱼϕ)/∂xⱼ + ν∇²ϕ` for θ and qv (SC-d's fused expressions,
`_fused_tendencies!`); `ν = nothing` drops the diffusion.
"""
struct BaseTendencies{V} <: BaseTerm
    ν::V
end
footprint(::BaseTendencies) = (1, 1, 1)

# The first term of a sweep writes the row (`=`), the others add to it.
# Only the base tendencies initialise.
@inline init_row!(t::BaseTendencies, buf, a, b, s, j, k, ii, st) =
    _trows!(buf, a, b, s, j, k, ii, st.u, st.v, st.w, st.θ, st.qv, st.m, t.ν)
init_row!(t::SweepTerm, args...) =
    throw(ArgumentError("init_row!: the first term of a sweep initialises the row; $(typeof(t)) only adds to it"))
@inline function init_point!(t::BaseTendencies, tb, i, j, k, st)
    u, v, w, m, ν = st.u, st.v, st.w, st.m, t.ν
    tb[i, SWEEP_U] = _tend_u(i, j, k, u, v, w, m, ν)
    tb[i, SWEEP_V] = _tend_v(i, j, k, u, v, w, m, ν)
    tb[i, SWEEP_W] = _tend_w(i, j, k, u, v, w, m, ν)
    _init_point_sc!(tb, SWEEP_Θ, i, j, k, u, v, w, st.θ, m, ν)
    _init_point_sc!(tb, SWEEP_QV, i, j, k, u, v, w, st.qv, m, ν)
    return nothing
end
@inline _init_point_sc!(tb, f, i, j, k, u, v, w, ::Nothing, m, ν) = nothing
@inline _init_point_sc!(tb, f, i, j, k, u, v, w, ϕ::AbstractArray, m, ν) =
    (tb[i, f] = _tend_ϕ(i, j, k, u, v, w, ϕ, m, ν); nothing)

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
delayed-row scheme). The points whose stencil reaches a halo, a plane or
a row the sweep updates only after all blocks (the slab edges and the
row-range edges), or a point an earlier operator had to defer, run after
the sweep, in the same two phases, with the halos of the predicted
velocity filled
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
    row_index(rowid, n, nrows) -> Union{Vector{Int32},Nothing}

The row pointer of a sparse list of `n` entries sorted by row (`rowid(e)`
in `1:nrows`, non-decreasing in `e`): the entries of row `r` are
[`row_entries`](@ref)`(ptr, r)` (compressed sparse rows), so a post term
finds the faces of one row of the sweep in O(1). `nothing` when the
entries are not sorted by row.
"""
function row_index(rowid::F, n::Integer, nrows::Integer) where {F}
    ptr = zeros(Int32, nrows + 1)
    prev = 1
    for e in 1:n
        r = rowid(e)
        (prev <= r <= nrows) || return nothing
        ptr[r+1] += 1
        prev = r
    end
    ptr[1] = 1
    for r in 1:nrows
        ptr[r+1] += ptr[r]
    end
    return ptr
end

"The entries of row `r` of a sparse list with the row pointer `ptr` ([`row_index`](@ref))."
@inline row_entries(ptr::AbstractVector{Int32}, r::Integer) = @inbounds Int(ptr[r]):Int(ptr[r+1])-1

"""
    row_entries(ptr, r, x, I1, I2) -> UnitRange{Int}

The entries `e` of row `r` with `I1 ≤ x(e) ≤ I2`, found by bisection, for
a list whose entries are non-decreasing in `x` within each row
([`rows_sorted`](@ref)). The sweep calls a post term on parts of a row
(the inner points, and after the sweep the x-ends), so a scan of the
whole row with a test per entry would visit each entry three times.
"""
@inline function row_entries(ptr::AbstractVector{Int32}, r::Integer, x::F, I1, I2) where {F}
    e = row_entries(ptr, r)
    lo, hi = first(e), last(e) + 1
    while lo < hi
        m = (lo + hi) >>> 1
        x(m) < I1 ? (lo = m + 1) : (hi = m)
    end
    s, hi = lo, last(e) + 1
    while lo < hi
        m = (lo + hi) >>> 1
        x(m) <= I2 ? (lo = m + 1) : (hi = m)
    end
    return s:lo-1
end

"Whether the entries of every row of `ptr` are non-decreasing in `x(e)` (the condition of the bisecting [`row_entries`](@ref))."
function rows_sorted(ptr::AbstractVector{Int32}, x::F) where {F}
    for r in 1:length(ptr)-1, e in Int(ptr[r]):Int(ptr[r+1])-2
        x(e) <= x(e + 1) || return false
    end
    return true
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
    post_plan(post::Tuple, lz, ly = 0) -> NTuple{N,NTuple{8,Int}}

The schedule of the post terms `post` in a sweep whose slabs keep `lz`
edge planes and whose row ranges keep `ly` edge rows (0: not split in j,
[`sweep_blocks`](@ref)): per term `(fx, fy, fz, ax, ay, az, df, da)`. A
point runs phase 1 (its factors) inside the sweep when its index distance
to the x-boundaries exceeds `fx`, to its row range's first and last row
`fy` and to its slab's first and last plane `fz` (phase 2: `ax, ay,
az`), else after the sweep. Phase 1 runs `df` rows behind the predictor,
phase 2 `da`. The widths grow by the term's radius over those of the
previous term's phase 2 (the halos: 0; the slab edges: `lz`; the
row-range edges: `ly`), so a stencil inside the sweep reads only values
of its own block that are final for the previous term and not yet
changed by its own; the delays grow by `r` and `2r` rows.
"""
post_plan(post::Tuple, lz::Integer, ly::Integer = 0) = _post_plan(post, 0, Int(ly), Int(lz), 0)
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

# the post terms on their rows behind the predictor row F of a block
# (planes k0:k1, rows jj: the block's row range)
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

# the slab of plane k (likewise the row range of row j)
@inline function _slab_of(k, kk, ns)
    s = clamp(((k - first(kk)) * ns) ÷ length(kk) + 1, 1, ns)
    while k < first(_slab(kk, ns, s)); s -= 1; end
    while k > last(_slab(kk, ns, s)); s += 1; end
    return s
end

# After the sweep, phase `ph` of a post term on the points the sweep left
# to it (outside the widths of `post_plan`): row (j, k) whole, or its x-ends.
@inline function _post_deferred_row!(j, k, t, ph, wx, wy, wz, st, ii, jj, kk, ns, nt)
    k0, k1 = _slab(kk, ns, _slab_of(k, kk, ns))
    j0, j1 = _slab(jj, nt, _slab_of(j, jj, nt))
    inner = j - j0 >= wy && j1 - j >= wy && k - k0 >= wz && k1 - k >= wz
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
# AfterSweep runs its whole-field operator. `ns` slabs, `nt` row ranges.
_post_deferred!(b, ::Tuple{}, ::Tuple{}, halo!, st, ii, jj, kk, ns, nt = 1) = nothing
function _post_deferred!(b, post::Tuple, plan::Tuple, halo!::H, st, ii, jj, kk, ns, nt = 1) where {H}   # (H: specialise on a callable struct)
    _post_deferred1!(b, first(post), first(plan), halo!, st, ii, jj, kk, ns, nt)
    return _post_deferred!(b, Base.tail(post), Base.tail(plan), halo!, st, ii, jj, kk, ns, nt)
end
_post_deferred1!(b, t::AfterSweep, e, halo!, st, ii, jj, kk, ns, nt) = (t.f(); nothing)
function _post_deferred1!(b, t::PostTerm, e, halo!::H, st, ii, jj, kk, ns, nt) where {H}
    fx, fy, fz, ax, ay, az = e
    footprint(t) == (0, 0, 0) || halo!()
    launch!(b, _post_deferred_row!, (jj, kk), t, Val(1), fx, fy, fz, st, ii, jj, kk, ns, nt)
    launch!(b, _post_deferred_row!, (jj, kk), t, Val(2), ax, ay, az, st, ii, jj, kk, ns, nt)
    return nothing
end

# ---------------------------------------------------------------------------
# buffers and blocks
# ---------------------------------------------------------------------------

"""
    SweepBuffers

The scratch of [`blocked_predictor!`](@ref) for a sweep cut into `ns`
slabs in k and `nt` row ranges in j, `nb = ns·nt` blocks (block `n = (t −
1)·ns + s` is slab `s` of row range `t`), `nf` fields:

  * `ring[i, kl, r, f, n]`: the tendencies of field `f` on block `n`'s
    inner planes (`kl = k − k0 + 1`), on the `lag + 1` ring rows `r ≤ lag
    + 1` and on its `2ly` edge rows (`r = lag + 1 + d`, `d ≤ ly` the first
    and `d > ly` the last `ly` rows of its row range; `ly = 0` when `nt =
    1`);
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
    sweep_buffers(old, data, nf, ns, nk; lag = 1, lz = 1, nt = 1, ly = lag) -> SweepBuffers

Buffers for `ns` slabs of a k-range of length `nk` times `nt` row ranges
in j, `nf` fields shaped like the storage array `data`, a ring of `lag +
1` rows, `lz` edge planes per slab end and, when `nt > 1`, `ly` edge rows
per row-range end ([`sweep_lags`](@ref); the predictor needs `ly = lag`);
`old` (a `SweepBuffers` or `nothing`) is returned when it already fits.
"""
function sweep_buffers(old, data::AbstractArray{T,3}, nf::Int, ns::Int, nk::Int;
                       lag::Int = 1, lz::Int = 1, nt::Int = 1, ly::Int = lag) where {T}
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

The blocks of the blocked sweep (#566, the split in j of #558): `ns`
slabs in k (at least `2lz` planes each) times `nt` row ranges in j (at
least `2ly` rows each), one work item each, when a launch of `npoints`
runs threaded; else `(1, 1)`.

A slab keeps `lz` edge planes at each end and a row range `ly` edge rows:
their predictor waits until every block is done and runs in a second and
a third launch. Slabs alone (#405, #533) capped the threads that work at
`nk ÷ 2lz`, 37 for 150 levels at `lz = 2` whatever the thread count
(measured on 72 threads: 37.6 busy in the sweep, #566); the row ranges
lift the cap to `(nk ÷ 2lz)(nj ÷ 2ly)`.

The rule (measured on 200 × 200 × 60 at 6, 12 and 24 threads, #558):
slabs no taller than 20 planes (`_SLAB_PLANES`) where the thread count
allows (taller slabs keep too many rows of every field in cache: whole
60-level columns ran 10-110 % slower), then the least estimated time
`cld(ns·nt, T) / (ns·nt) · (1 + c·f_edge)` for `T` threads, a fraction
`f_edge` of edge points and `c = 0.5` (`_C_EDGE`): the most blocks up to
one per thread, and among those the fewest edge points.
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
# the sweep
# ---------------------------------------------------------------------------

# block n (slab s = (n − 1) % ns + 1, row range (n − 1) ÷ ns + 1): the
# j-row sweep (one launch item per block; blocks write disjoint points and
# read the others' only on their edge planes and edge rows, not yet
# updated). Front t: the tendencies of row t, the predictor of row t − lag,
# then the post-predictor operators `post` on their rows behind it
# (`plan`, [`post_plan`](@ref)); the front runs `lag` + the operators'
# delay past the last row, so every row gets its predictor and its
# operators. `nt` row ranges with `ly` edge rows each (`nt = 1`: `ly = 0`,
# the whole j-range, slabs alone). The short form (no post terms, one row
# range) is the tests'.
_b_sweep_block!(n, terms, st, pu, p1, p2, ring, edge, ii, jj, kk, ns, lag, lz) =
    _b_sweep_block!(n, terms, (), (), st, pu, p1, p2, ring, edge, ii, jj, kk, ns, lag, lz)
function _b_sweep_block!(n, terms, post, plan, st, pu, p1, p2, ring, edge, ii, jj, kk, ns, lag, lz,
                         nt = 1, ly = 0)
    s = (n - 1) % ns + 1
    k0, k1 = _slab(kk, ns, s)
    j0, j1 = _slab(jj, nt, (n - 1) ÷ ns + 1)
    nr = lag + 1
    for t in j0:j1+lag+post_delay(plan)
        if t <= j1
            d = ly > 0 ? _edge_slot(t, j0, j1, ly) : 0
            r = d > 0 ? nr + d : mod(t, nr) + 1          # an edge row: its own slot, kept to the third launch
            for k in k0:k1
                e = _edge_slot(k, k0, k1, lz)
                if e > 0
                    _term_rows!(edge, t, e, s, t, k, ii, terms, st)
                else
                    _term_rows!(ring, k - k0 + 1, r, n, t, k, ii, terms, st)
                end
            end
        end
        jp = t - lag
        if j0 <= jp <= j1 && (ly == 0 || _edge_slot(jp, j0, j1, ly) == 0)
            rp = mod(jp, nr) + 1
            for k in k0+lz:k1-lz
                _prows!(ring, k - k0 + 1, rp, n, jp, k, ii, st, pu, p1, p2)
            end
        end
        _post_rows!(post, plan, jp, k0, k1, ii, j0:j1, st)
    end
    return nothing
end

# the edge rows, after every block: (kl, d, n), plane k0 + kl − 1 of edge
# row d of block n (its inner planes; an edge plane went to `edge`)
@inline function _b_sweep_yedge!(kl, d, n, st, pu, p1, p2, ring, ii, jj, kk, ns, nt, lag, lz, ly)
    k0, k1 = _slab(kk, ns, (n - 1) % ns + 1)
    j0, j1 = _slab(jj, nt, (n - 1) ÷ ns + 1)
    k = k0 + kl - 1
    (k < k0 + lz || k > k1 - lz) && return nothing
    j = d <= ly ? j0 + d - 1 : j1 - (2ly - d)
    _prows!(ring, kl, lag + 1 + d, n, j, k, ii, st, pu, p1, p2)
    return nothing
end

# the edge planes, after every block: (i, j, n), n = (s − 1)·2lz + e for
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
first, a [`BaseTerm`](@ref) such as [`BaseTendencies`](@ref), initialises it, every further
[`SweepTerm`](@ref) adds its [`add_tendency!`](@ref) in tuple order), then
`u ← pu(u, tu)` (likewise v, w) and, for each scalar `s = (ϕ, ps, snap)`
(or `nothing`), `ϕ ← ps(ϕ, tϕ)` (`ps(ϕ, tϕ, snap[i, j, k])` when `snap` is
an array), over the interior of `g`. Within round-off the whole-field
passes of the same terms followed by the predictor passes, but the
tendencies stay in `bufs` ([`sweep_buffers`](@ref) with the
[`sweep_lags`](@ref) of `terms`, `bufs.nf` ≥ 3 + the number of scalars),
blocked so each field is streamed once. Every halo a term reads must be
filled; nothing else may read or write the fields meanwhile.

The interior is cut into `bufs.ns` slabs in k times `bufs.nt` row ranges
in j ([`sweep_blocks`](@ref)), one launch item per block; the slabs' edge
planes and the row ranges' edge rows (`bufs.ly = lag` when `nt > 1`) get
their predictor in a second and a third launch.

`post` (default none) is a tuple of [`PostTerm`](@ref)s, the core's
post-predictor operators: inside the sweep on the rows behind the
predictor, the rest after it ([`post_plan`](@ref)); `halo!()` fills the
halos of the predicted u, v, w for the points run after the sweep. Their
[`prepare!`](@ref) and [`finish!`](@ref) are the caller's.
"""
blocked_predictor!(bufs::SweepBuffers, terms::Tuple{BaseTerm,Vararg{SweepTerm}},
                   u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, pu, s1, s2, g::AbstractMesh) =
    blocked_predictor!(bufs, terms, (), nothing, u, v, w, pu, s1, s2, g)
function blocked_predictor!(bufs::SweepBuffers, terms::Tuple{BaseTerm,Vararg{SweepTerm}},
                            post::Tuple{Vararg{PostTerm}}, halo!::H,
                            u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                            pu, s1, s2, g::AbstractMesh) where {H}
    ii, jj, kk = interior(g)
    ns, nt = bufs.ns, bufs.nt
    lag, lz = sweep_lags(terms)
    ly = nt > 1 ? lag : 0
    nf = 3 + (s1 === nothing ? 0 : 1) + (s2 === nothing ? 0 : 1)
    (s1 === nothing && s2 !== nothing) && throw(ArgumentError("blocked_predictor!: s2 without s1"))
    bufs.nf >= nf && bufs.lag == lag && bufs.lz == lz && bufs.ly == ly &&
        size(bufs.ring, 2) >= cld(length(kk), ns) && (ns == 1 || 2lz * ns <= length(kk)) &&
        (nt == 1 || 2ly * nt <= length(jj)) ||
        throw(DimensionMismatch("blocked_predictor!: buffers do not fit ($ns slabs over $(length(kk)) " *
                                "planes, $nt row ranges over $(length(jj)) rows, lag $lag, $lz edge " *
                                "planes, $(bufs.ly) edge rows)"))
    b = Backends.backend(u.data)
    _check_inputs(b, terms)
    st = (u = u.data, v = v.data, w = w.data, θ = _sweep_field(s1), qv = _sweep_field(s2), m = grid_metrics(g))
    p1, p2 = _sweep_pred(s1), _sweep_pred(s2)
    plan = post_plan(post, lz, ly)
    nb = ns * nt
    launch_weighted!(b, cld(length(ii) * length(jj) * length(kk), nb), _b_sweep_block!, (1:nb,),
                     terms, post, plan, st, pu, p1, p2, bufs.ring, bufs.edge, ii, jj, kk, ns, lag, lz, nt, ly)
    launch!(b, _b_sweep_edge!, (ii, jj, 1:2lz*ns), st, pu, p1, p2, bufs.edge, kk, ns, lz)
    ly > 0 && launch_weighted!(b, length(ii), _b_sweep_yedge!, (1:size(bufs.ring, 2), 1:2ly, 1:nb),
                               st, pu, p1, p2, bufs.ring, ii, jj, kk, ns, nt, lag, lz, ly)
    _post_deferred!(b, post, plan, halo!, st, ii, jj, kk, ns, nt)
    return nothing
end

# ---------------------------------------------------------------------------
# the tendencies written to fields: the per-point (GPU) form and the
# fused tendency launch (#533 plan step 8)
# ---------------------------------------------------------------------------

"""
    TendencyFields(tu, tv, tw, tθ, tqv)

The tendency fields' storage arrays seen as the five-index buffer of the
sweep: `tf[i, j, k, f, 1]` is `tu[i, j, k]` for `f = SWEEP_U`, … `tqv` for
`SWEEP_QV` (`tθ`, `tqv` may be `nothing` when the core has no such
scalar). A [`SweepRow`](@ref) over it (`a = j`, `b = k`, `s = 1`) is the
row a term's [`add_tendency!`](@ref) writes into when the tendencies go to
the fields instead of the ring buffer: in the per-point form of the sweep
([`pointwise_predictor!`](@ref)) and in [`term_tendencies!`](@ref). `f` is
a constant in every term kernel, so the branch on it folds away.
"""
struct TendencyFields{T,A<:AbstractArray{T,3},S,Q} <: AbstractArray{T,5}
    u::A
    v::A
    w::A
    θ::S
    qv::Q
end
Base.size(t::TendencyFields) = (size(t.u)..., 5, 1)
@inline function Base.getindex(t::TendencyFields, i::Int, j::Int, k::Int, f::Int, s::Int)
    @inbounds f == 1 ? t.u[i, j, k] : f == 2 ? t.v[i, j, k] : f == 3 ? t.w[i, j, k] :
                       f == 4 ? t.θ[i, j, k] : t.qv[i, j, k]
end
@inline function Base.setindex!(t::TendencyFields, x, i::Int, j::Int, k::Int, f::Int, s::Int)
    @inbounds if f == 1
        t.u[i, j, k] = x
    elseif f == 2
        t.v[i, j, k] = x
    elseif f == 3
        t.w[i, j, k] = x
    elseif f == 4
        t.θ[i, j, k] = x
    else
        t.qv[i, j, k] = x
    end
    return t
end

# the composed point: the base term writes, the others add, in order
@inline function _term_point!(terms::Tuple, tb, i, j, k, st)
    init_point!(first(terms), tb, i, j, k, st)
    _add_points!(Base.tail(terms), tb, i, j, k, st)
    return nothing
end
@inline _add_points!(::Tuple{}, tb, i, j, k, st) = nothing
@inline function _add_points!(ts::Tuple, tb, i, j, k, st)
    add_tendency!(first(ts), tb, i, j, k, st)
    return _add_points!(Base.tail(ts), tb, i, j, k, st)
end

# one point of the per-point form: every term's tendency into the fields
@inline function _b_term_point!(i, j, k, terms, st, tf)
    _term_point!(terms, SweepRow(tf, j, k, 1), i, j, k, st)
    return nothing
end
# one row (j, k) of the fused tendency launch on the CPU (the ring sweep's
# row kernels, writing the fields)
@inline function _b_term_row!(j, k, terms, st, tf, ii)
    _term_rows!(tf, j, k, 1, j, k, ii, terms, st)
    return nothing
end
# the predictor of one point from the tendency fields
@inline function _b_point_predictor!(i, j, k, st, tf, pu, p1, p2)
    u, v, w = st.u, st.v, st.w
    @inbounds begin
        u[i, j, k] = pu(u[i, j, k], tf.u[i, j, k])
        v[i, j, k] = pu(v[i, j, k], tf.v[i, j, k])
        w[i, j, k] = pu(w[i, j, k], tf.w[i, j, k])
    end
    _point_sc!(i, j, k, st.θ, tf.θ, p1)
    _point_sc!(i, j, k, st.qv, tf.qv, p2)
    return nothing
end
@inline _point_sc!(i, j, k, ::Nothing, t, p) = nothing
@inline function _point_sc!(i, j, k, ϕ, t, (ps, snap))
    @inbounds ϕ[i, j, k] = _pred(ps, ϕ[i, j, k], t[i, j, k], snap, i, j, k)
    return nothing
end

_sweep_state(u, v, w, θ, qv, g) = (u = u.data, v = v.data, w = w.data, θ = _data(θ), qv = _data(qv),
                                   m = grid_metrics(g))
_data(::Nothing) = nothing
_data(f::Field) = f.data

"""
    term_tendencies!(tf::TendencyFields, terms, u, v, w, θ, qv, g; ka = nothing)

Write the sum of the `terms` (a [`BaseTerm`](@ref) first, then any
[`SweepTerm`](@ref)s) into the interior of the tendency fields `tf` in one
launch: the loop fusion of the separate passes of the same terms, without
the predictor (the explicit part of `step_imex!`, whose tendencies are
kept for the stage combinations, #533 plan step 8). `θ`, `qv` are the
scalar fields or `nothing`. On the CPU (`ka = nothing`) one i-row per work
item, the row kernels of the ring-buffer sweep; with a KernelAbstractions
backend `ka`, the per-point form (one work item per point, the GPU's).
Halos of the state must be filled; the tendencies' halos are not written.
"""
function term_tendencies!(tf::TendencyFields, terms::Tuple{BaseTerm,Vararg{SweepTerm}},
                          u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}, θ, qv, g::AbstractMesh;
                          ka = nothing)
    ii, jj, kk = interior(g)
    st = _sweep_state(u, v, w, θ, qv, g)
    b = Backends.backend(u.data)
    _check_inputs(b, terms)
    if ka === nothing
        launch_weighted!(b, length(ii), _b_term_row!, (jj, kk), terms, st, tf, ii)
    else
        Backends.launch_points!(ka, _b_term_point!, (ii, jj, kk), terms, st, tf)
    end
    return nothing
end

"""
    pointwise_predictor!(ka, tf::TendencyFields, terms, u, v, w, pu, s1, s2, g)

The per-point form of [`blocked_predictor!`](@ref), for a GPU (#533 plan
step 8; SC-d's form): one KernelAbstractions kernel on backend `ka` with
one work item per point evaluates every term of `terms` there (the
[`BaseTerm`](@ref)'s `init_point!`, then each [`add_tendency!`](@ref) in
order) and writes the sum into the tendency fields `tf`; a second kernel
applies the predictor (`u ← pu(u, tu)`, and each scalar `s = (ϕ, ps,
snap)` as in `blocked_predictor!`). No ring buffer: the predictor cannot
run in the first kernel, since a neighbour's stencil still reads the
point. The values are those of the ring-buffer sweep within round-off
(the same expressions). Tested on `KernelAbstractions.CPU()` as the
stand-in for a GPU.
"""
function pointwise_predictor!(ka, tf::TendencyFields, terms::Tuple{BaseTerm,Vararg{SweepTerm}},
                              u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                              pu, s1, s2, g::AbstractMesh)
    (s1 === nothing && s2 !== nothing) && throw(ArgumentError("pointwise_predictor!: s2 without s1"))
    (s1 === nothing || tf.θ !== nothing) && (s2 === nothing || tf.qv !== nothing) ||
        throw(ArgumentError("pointwise_predictor!: a scalar without its tendency field"))
    I3 = interior(g)
    b = Backends.backend(u.data)
    _check_inputs(b, terms)
    st = (u = u.data, v = v.data, w = w.data, θ = _sweep_field(s1), qv = _sweep_field(s2), m = grid_metrics(g))
    Backends.launch_points!(ka, _b_term_point!, I3, terms, st, tf)
    Backends.launch_points!(ka, _b_point_predictor!, I3, st, tf, pu, _sweep_pred(s1), _sweep_pred(s2))
    return nothing
end

"""
    pointwise_predictor!(ka, tf, terms, post, halo!, u, v, w, pu, s1, s2, g)

[`pointwise_predictor!`](@ref) followed by the post-predictor operators
`post` (a tuple of [`PostTerm`](@ref)s, as in [`blocked_predictor!`](@ref)):
the per-point form has no delayed rows, so every post term runs after the
predictor on every point, phase 1 then phase 2 (an [`AfterSweep`](@ref)
runs its whole-field operator), in tuple order, with the halos of the
predicted velocity filled by `halo!()` before a term with a stencil. This
is the schedule [`post_plan`](@ref) gives a point outside the sweep's
widths, so the values are those of the ring-buffer sweep within round-off.
Their [`prepare!`](@ref) and [`finish!`](@ref) are the caller's.
"""
function pointwise_predictor!(ka, tf::TendencyFields, terms::Tuple{BaseTerm,Vararg{SweepTerm}},
                              post::Tuple{Vararg{PostTerm}}, halo!::H,
                              u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                              pu, s1, s2, g::AbstractMesh) where {H}
    pointwise_predictor!(ka, tf, terms, u, v, w, pu, s1, s2, g)
    post === () && return nothing
    ii, jj, kk = interior(g)
    st = (u = u.data, v = v.data, w = w.data, θ = _sweep_field(s1), qv = _sweep_field(s2), m = grid_metrics(g))
    _post_deferred!(Backends.backend(u.data), post, map(_ -> _ALL_AFTER, post), halo!, st, ii, jj, kk, 1)
    return nothing
end
# a plan entry that leaves every point to the pass after the predictor
const _ALL_AFTER = (_AFTER, _AFTER, _AFTER, _AFTER, _AFTER, _AFTER, 0, 0)
