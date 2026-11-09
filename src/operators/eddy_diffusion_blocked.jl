# =========================================================================
# A scalar's explicit eddy-diffusion step as one cache-blocked pass (#559).
# Included by Operators after FiniteVolume.
# =========================================================================
#
# The step `ϕ ← ϕ + Δt · D(ϕ)`, with D the masked and/or weighted operator
# of `add_eddy_diffusion_scalar!(…; mask, weights)`, and the per-level sums
# of |ϕ| and of D(ϕ) that the urban tracer diffusion books (its
# conservation residual and the water ledger, `TracerSGSDiffusion`). As
# separate loops that is: zero a tendency field, the diffusion into it, a
# plane reduction of |ϕ| and the tendency, the update. Here it runs on the
# blocked driver of #533 / #554 (`FiniteVolume.blocked_update!`): per slab
# and j-row, the tendency of each row into a one-row buffer (`_Line`) by
# the point body of the separate loop (`_b_eddy_scalar_masked!`: one source
# of truth), then the new ϕ of the row into the driver's ring (or edge
# buffer) and the row's two sums. D(ϕ) reads ϕ one cell away, so the new
# rows are written back one row later (lag 1) and each slab keeps one edge
# plane per end (lz = 1). ϕ is the separate loops' bit for bit (the same
# expressions); the per-level sums are the row sums added in ascending j,
# which differs from the plane reduction's order only by round-off and
# does not depend on the thread or slab count.

using .FiniteVolume: BlockedUpdate, blocked_update!, sweep_buffers, sweep_slabs, SweepBuffers,
                     _row_slot, _Line, _zero_line!, _write_ring_row!, _b_eddy_scalar_masked!
import .FiniteVolume: update_lags, compute_rows!, write_row!, write_point!

"""
    EddyDiffusionScratch

The scratch of [`blocked_eddy_diffusion!`](@ref): the driver's buffers
(`FiniteVolume.SweepBuffers`, one field, lag 1, one edge plane), one
tendency row per slab and the two sums (Σ|ϕ|, Σ tendency) of every
interior i-row. Made by the first call and returned for the next.
"""
mutable struct EddyDiffusionScratch{B<:SweepBuffers, L<:AbstractMatrix, T}  # mutable: kept in an `Any` slot without a box
    const bufs::B
    const td::L
    const rowsums::Matrix{NTuple{2,T}}
end

function _eddy_diffusion_scratch(old, data::AbstractArray{T,3}, ns::Int, ny::Int, nz::Int) where {T}
    B = SweepBuffers{FiniteVolume._buf_type(data)}
    L = typeof(similar(data, 0, 0))
    o = old isa EddyDiffusionScratch{B, L, T} ? old : nothing
    bufs = sweep_buffers(o === nothing ? nothing : o.bufs, data, 1, ns, nz; lag = 1, lz = 1)
    o !== nothing && o.bufs === bufs && size(o.td, 2) == ns && size(o.td, 1) == size(data, 1) &&
        size(o.rowsums) == (ny, nz) && return o
    return EddyDiffusionScratch(bufs, similar(data, size(data, 1), ns), Matrix{NTuple{2,T}}(undef, ny, nz))
end

# The update of the blocked driver: every field typed, the grid spacing and
# Δt as numbers, the first interior row and plane (J1, K1).
struct _EddyDiffusionUpdate{Q, KH, KV, M, WC, WF, T, L} <: BlockedUpdate
    q::Q
    kh::KH
    kv::KV
    mask::M
    wc::WC
    wf::WF
    idx::T
    idy::T
    idz::T
    dt::T
    td::L
    rowsums::Matrix{NTuple{2,T}}
    J1::Int
    K1::Int
end

update_lags(::_EddyDiffusionUpdate) = (1, 1)

function compute_rows!(op::_EddyDiffusionUpdate, ring, edge, r, s, j, k0, k1, ii, lz)
    tl = _Line(op.td, s)
    for k in k0:k1
        buf, a, b = _row_slot(ring, edge, r, j, k, k0, k1, lz)
        _zero_line!(tl, ii)
        _diffusion_row!(tl, ii, j, k, op.q, op.kh, op.kv, op.mask, op.wc, op.wf, op.idx, op.idy, op.idz)
        sums = _diffusion_step_row!(buf, a, b, s, op.q, op.td, op.dt, j, k, ii)
        @inbounds op.rowsums[j - op.J1 + 1, k - op.K1 + 1] = sums
    end
    return nothing
end

# the tendency of row (j, k) by the point body of the separate loop, in a
# row loop of its own (`ivdep`: the body writes only its own point of the
# line, which nothing else in the row reads)
@noinline function _diffusion_row!(tl, ii, j, k, q, kh, kv, mask, wc, wf, idx, idy, idz)
    @inbounds @simd ivdep for i in ii
        _b_eddy_scalar_masked!(i, j, k, tl, q, kh, kv, mask, wc, wf, idx, idy, idz)
    end
    return nothing
end

# the new ϕ of row (j, k), `ϕ + Δt · t` (the expression of the separate
# update `ϕ .+= Δt .* t`), and the row's Σ|ϕ| (old ϕ) and Σ t
@noinline function _diffusion_step_row!(buf, a, b, s, q, td, dt, j, k, ii)
    T = eltype(q)
    sq = st = zero(T)
    @inbounds @simd ivdep for i in ii
        x = q[i, j, k]; t = td[i, s]
        buf[i, a, b, 1, s] = x + dt * t
        sq += abs(x); st += t
    end
    return (sq, st)
end

write_row!(op::_EddyDiffusionUpdate, ring, a, b, s, j, k, ii) = _write_ring_row!(op.q, ring, a, b, s, j, k, ii)
@inline function write_point!(op::_EddyDiffusionUpdate, edge, i, j, k, e, s)
    @inbounds op.q[i, j, k] = edge[i, j, e, 1, s]
    return nothing
end

"""
    blocked_eddy_diffusion!(scratch, Sq, St, ϕ, κh, κv, Δt; mask = nothing, weights = nothing,
                            ns = nothing, reverse_slabs = false) -> scratch

One explicit eddy-diffusion step of the cell-centred scalar `ϕ` as one
cache-blocked pass (#559): `ϕ ← ϕ + Δt · D(ϕ)` over the interior, with
`D` the operator of [`add_eddy_diffusion_scalar!`](@ref)`(tϕ, ϕ, κh, κv;
mask, weights)` (`κh` on the x and y faces, `κv` on the z faces; pass the
same field twice for the isotropic operator), and per interior level `k`
`Sq[k] = Σ_ij |ϕ|` (before the step) and `St[k] = Σ_ij D(ϕ)`.

The separate loops it replaces (zero a tendency, the diffusion, a plane
reduction, the update; the reference, e.g. `loop_fusion = "never"` in the
urban tracer diffusion) give the same `ϕ` bit for bit, and sums that agree
to round-off (here: the row sums in ascending j, independent of the thread
and slab count). It runs on `FiniteVolume.blocked_update!`: ring lag 1,
one edge plane per slab end, `ns` slabs (default one per thread,
`FiniteVolume.sweep_slabs`); `reverse_slabs` is the driver's test hook.

The halos of `ϕ`, `κh`, `κv` and `mask` must be filled. A `RegularGrid`
and the CPU backend only (the masked kernel's grid; the driver's
threading). `scratch` is an [`EddyDiffusionScratch`](@ref) from an earlier
call or `nothing`; the one used is returned.
"""
function blocked_eddy_diffusion!(scratch, Sq::AbstractVector, St::AbstractVector,
                                 ϕ::Field{CellCenter}, κh::Field{CellCenter}, κv::Field{CellCenter},
                                 Δt::Real; mask = nothing, weights = nothing, ns = nothing,
                                 reverse_slabs::Bool = false)
    g = mesh(ϕ)
    g isa RegularGrid ||
        throw(ArgumentError("blocked_eddy_diffusion!: a RegularGrid only (the masked diffusion kernel's grid)"))
    b = Backends.backend(ϕ.data)
    b isa Backends.CPUBackend ||
        throw(ArgumentError("blocked_eddy_diffusion!: the cache-blocked pass runs on the CPU backend only"))
    wc, wf = _check_mask_weights(ϕ, mask, weights)
    ii, jj, kk = interior(g)
    (length(Sq) >= length(kk) && length(St) >= length(kk)) ||
        throw(DimensionMismatch("blocked_eddy_diffusion!: Sq and St need $(length(kk)) levels"))
    T = eltype(ϕ.data)
    nsl = ns === nothing ? sweep_slabs(b, length(ii) * length(jj) * length(kk), length(kk); lz = 1) : ns
    sc = _eddy_diffusion_scratch(scratch, ϕ.data, nsl, length(jj), length(kk))
    op = _EddyDiffusionUpdate(ϕ.data, κh.data, κv.data, mask, wc, wf, T(1 / g.dx), T(1 / g.dy), T(1 / g.dz),
                              T(Δt), sc.td, sc.rowsums, first(jj), first(kk))
    blocked_update!(sc.bufs, op, b, g; reverse_slabs = reverse_slabs)
    _level_sums!(Sq, St, sc.rowsums)
    return sc
end

# the per-level sums from the row sums, in ascending j
function _level_sums!(Sq, St, rowsums::Matrix{NTuple{2,T}}) where {T}
    @inbounds for k in axes(rowsums, 2)
        a = c = zero(T)
        for j in axes(rowsums, 1)
            x = rowsums[j, k]
            a += x[1]; c += x[2]
        end
        Sq[k] = a; St[k] = c
    end
    return nothing
end
