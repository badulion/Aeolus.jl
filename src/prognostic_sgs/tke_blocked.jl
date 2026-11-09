# =========================================================================
# The update of the SGS TKE as one cache-blocked pass (#554, part 2).
# Included by PrognosticSGS after advance_sgs!.
# =========================================================================
#
# `advance_sgs!` of a TKEClosure runs, as separate loops (the reference,
# `_advance_separate!`): N², ν and κt, the halo fill of ν, the dual strain
# D, the wall production, the flux-limited advection of e (reconstructed
# and upwind face fluxes, the limiter's ratio, its halo, the limited
# fluxes, their divergence), K = 2ν, the masked diffusion, the update. Here
# the part that reads e runs row by row on the blocked driver of #533
# (`FiniteVolume.blocked_update!`, the slabs and the ring of the predictor
# sweep); N², D and the wall production, which do not read e, stay
# separate loops before it.
#
# Per slab (planes k0:k1) and j-row, a pipeline of intermediate rows, each
# in a small ring of its own (`_TKERings`, planes k0−1:k1+1):
#
#   step j:  ν of row j+1 (the coefficient body); the reconstructed and
#            upwind x and z face fluxes of row j+1 and the y faces between
#            rows j+1 and j+2; the limiter's ratio of row j+1; the y faces
#            between rows j and j+1 limited (ratios of rows j and j+1);
#   row j:   its x and z faces limited, the flux divergence, the diffusion
#            with 2ν (rows j−1:j+1 of ν), the update of e with its budget
#            sums, into the driver's ring (or edge buffer).
#
# Every value is computed by the point body of the separate loop
# (`_coefficients_body!`, `AdvectionSchemes._b_rflux_*!`, `_b_uflux_*!`,
# `_b_limit_ratio!`, `_b_limit_*!`, `_b_flux_div!`, `FiniteVolume.
# _b_eddy_scalar!` / `_b_eddy_scalar_masked!`, `_tke_update_row`), reading
# and writing the rings through array adapters (`_Ring`, `_Line`, …): one
# source of truth for every expression, and the values are those of the
# separate loops.
#
# Footprint. A face flux of a reconstruction of halo `hr` (3 for WENO-Z5,
# 1 centred) reads e up to `hr` cells either side of the face; the ratio
# of a cell reads its six faces; a limited face the ratios of its two
# cells. So the new e of a cell reads the old e up to `hr + 1` cells away:
# the slabs keep `lz = hr + 1` edge planes per end. In j the pipeline runs
# two rows ahead of the row it finishes and its rings carry the rest, so
# the old e of row j − lag is read by no later step for `lag = max(1, hr −
# 2)`: the y faces of step j read rows ≥ j + 2 − hr, the diffusion of row
# j rows ≥ j − 1.
#
# Halos. The separate loops fill the halos of ν (with the velocity
# conditions) and of the ratio (with e's) from the interior. The pass
# computes ν and the ratio of the interior cells next to the boundary (the
# shell) before the sweep, fills those two halos as the separate loops do,
# and copies the halo cells into its rings; every boundary condition whose
# first halo layer is a function of the first interior layer (periodic,
# mirror, an open face's mirror for fields without boundary data) is then
# reproduced.

using ..AdvectionSchemes: _b_rflux_x!, _b_rflux_y!, _b_rflux_z!, _b_uflux_x!, _b_uflux_y!,
    _b_uflux_z!, _b_limit_ratio!, _b_limit_x!, _b_limit_y!, _b_limit_z!, _b_flux_div!,
    _dz_metric, _xfaces, _yfaces, _zfaces, required_halo, _check_halo, _check_stretched_recon
using ..Operators.FiniteVolume: BlockedUpdate, blocked_update!, sweep_buffers, sweep_slabs,
    SweepBuffers, _row_slot, grid_metrics, _b_eddy_scalar!, _b_eddy_scalar_masked!
import ..Operators.FiniteVolume: update_lags, begin_slab!, compute_rows!, write_row!, write_point!
using ..Simulations: StepWorkspaces

# -------------------------------------------------------------------------
# array adapters: a whole-field body reads and writes the rings
# -------------------------------------------------------------------------

# a ring of NR rows of slab s: (I, J, K) ↦ buf[I, K − koff, mod(J, NR) + 1, s]
struct _Ring{T, NR, A<:AbstractArray{T,4}} <: AbstractArray{T,3}
    buf::A
    koff::Int
    s::Int
end
_Ring{NR}(buf::AbstractArray{T,4}, koff, s) where {T, NR} = _Ring{T, NR, typeof(buf)}(buf, koff, s)
Base.size(r::_Ring{T,NR}) where {T, NR} = (size(r.buf, 1), size(r.buf, 2), NR)
@inline Base.getindex(r::_Ring{T,NR}, I::Integer, J::Integer, K::Integer) where {T, NR} =
    @inbounds r.buf[I, K - r.koff, mod(J, NR) + 1, r.s]
@inline Base.setindex!(r::_Ring{T,NR}, x, I::Integer, J::Integer, K::Integer) where {T, NR} =
    (@inbounds r.buf[I, K - r.koff, mod(J, NR) + 1, r.s] = x; r)

# one row of slab s, whatever (J, K): (I, J, K) ↦ buf[I, s]
struct _Line{T, A<:AbstractMatrix{T}} <: AbstractArray{T,3}
    buf::A
    s::Int
end
Base.size(r::_Line) = (size(r.buf, 1), 1, 1)
@inline Base.getindex(r::_Line, I::Integer, J::Integer, K::Integer) = @inbounds r.buf[I, r.s]
@inline Base.setindex!(r::_Line, x, I::Integer, J::Integer, K::Integer) = (@inbounds r.buf[I, r.s] = x; r)

# the driver's slot of one row of field f: (I, J, K) ↦ buf[I, a, b, f, s]
struct _Slot{T, A<:AbstractArray{T,5}} <: AbstractArray{T,3}
    buf::A
    a::Int
    b::Int
    f::Int
    s::Int
end
Base.size(r::_Slot) = (size(r.buf, 1), 1, 1)
@inline Base.getindex(r::_Slot, I::Integer, J::Integer, K::Integer) = @inbounds r.buf[I, r.a, r.b, r.f, r.s]
@inline Base.setindex!(r::_Slot, x, I::Integer, J::Integer, K::Integer) =
    (@inbounds r.buf[I, r.a, r.b, r.f, r.s] = x; r)

# the old e read from the field, the new e written to the slot
struct _ReadWrite{T, A<:AbstractArray{T,3}, B<:AbstractArray{T,3}} <: AbstractArray{T,3}
    src::A
    dst::B
end
Base.size(r::_ReadWrite) = size(r.src)
@inline Base.getindex(r::_ReadWrite, I::Integer, J::Integer, K::Integer) = @inbounds r.src[I, J, K]
@inline Base.setindex!(r::_ReadWrite, x, I::Integer, J::Integer, K::Integer) =
    (@inbounds r.dst[I, J, K] = x; r)

# 2ν read from ν (`2 * ν`: the expression of `_double!`)
struct _Doubled{T, A<:AbstractArray{T,3}} <: AbstractArray{T,3}
    a::A
end
Base.size(r::_Doubled) = size(r.a)
@inline Base.getindex(r::_Doubled, I::Integer, J::Integer, K::Integer) = @inbounds 2 * r.a[I, J, K]

# a face flux evaluated where it is read, by the flux body itself writing
# into a one-value sink (the shell's ratio: no flux array, one launch)
struct _Sink{T} <: AbstractArray{T,3}
    r::Base.RefValue{T}
end
Base.size(::_Sink) = (1, 1, 1)
@inline Base.setindex!(s::_Sink, x, I::Integer, J::Integer, K::Integer) = (s.r[] = x; s)
struct _FaceFlux{T, F, A<:Tuple} <: AbstractArray{T,3}
    body!::F
    args::A
end
_FaceFlux{T}(body!::F, args::A) where {T, F, A} = _FaceFlux{T, F, A}(body!, args)
Base.size(::_FaceFlux) = (0, 0, 0)
@inline function Base.getindex(f::_FaceFlux{T}, I::Integer, J::Integer, K::Integer) where {T}
    s = _Sink{T}(Ref(zero(T)))
    f.body!(I, J, K, s, f.args...)
    return s.r[]
end

# a field of zeros (the wall production without walls) and a sink (κt,
# which the update recomputes)
struct _Zeros{T} <: AbstractArray{T,3} end
Base.size(::_Zeros) = (0, 0, 0)
@inline Base.getindex(::_Zeros{T}, I::Integer, J::Integer, K::Integer) where {T} = zero(T)
struct _Discard{T} <: AbstractArray{T,3} end
Base.size(::_Discard) = (0, 0, 0)
@inline Base.setindex!(d::_Discard, x, I::Integer, J::Integer, K::Integer) = d

# -------------------------------------------------------------------------
# the rings of the pipeline
# -------------------------------------------------------------------------

# Per slab s, planes k0−1:k1+1 (`kl = k − k0 + 2`; the z faces k0−1:k1+2):
# ν of three rows; for each flux-limited scalar (`_FluxRings`) the ratio of
# three rows, the x and z face fluxes (reconstructed H, upwind L) of two
# rows and the y face fluxes of three face rows; and one row of the
# tendency of e.
struct _FluxRings{A<:AbstractArray}
    R::A
    FxH::A; FxL::A; FzH::A; FzL::A
    FyH::A; FyL::A
end
_flux_rings(r) = _FluxRings(r(3), r(2), r(2), r(2, 1), r(2, 1), r(3), r(3))
# (mutable, as `SweepBuffers`: kept in the closure's `Any` slot without a
# box per step; the fields are `const`)
mutable struct _TKERings{A<:AbstractArray, M<:AbstractMatrix}
    const ν::A
    const f::_FluxRings{A}
    const td::M
end

# (the split, as in `sweep_buffers`, keeps a cached `old::Any` from the
# closure's `RefValue{Any}` from making the rings, and with them the update
# and its launch, abstract: a concrete ring type costs no allocation)
function _tke_rings(old, data::AbstractArray{T,3}, nkl::Int, ns::Int) where {T}
    R = _ring_type(data)
    return old isa _TKERings{R, _row_type(data)} ? _tke_rings_(old, data, nkl, ns) :
                                                    _tke_rings_(nothing, data, nkl, ns)
end
_ring_type(::Array{T,3}) where {T} = Array{T,4}
_ring_type(a::AbstractArray) = typeof(similar(a, 0, 0, 0, 0))
_row_type(::Array{T,3}) where {T} = Matrix{T}
_row_type(a::AbstractArray) = typeof(similar(a, 0, 0))
function _tke_rings_(old, data::AbstractArray{T,3}, nkl::Int, ns::Int) where {T}
    sx = size(data, 1)
    old isa _TKERings && size(old.ν, 1) == sx && size(old.ν, 2) >= nkl && size(old.ν, 4) == ns &&
        return old
    r(nr, extra = 0) = similar(data, sx, nkl + extra, nr, ns)
    return _TKERings(r(3), _flux_rings(r), similar(data, sx, ns))
end

# -------------------------------------------------------------------------
# the update
# -------------------------------------------------------------------------

# Everything the pipeline reads (a NamedTuple, so every field is typed):
# fields `e`, `νf` (ν with its halos filled), `u, v, w`, `n2`, `dd`, `pw`;
# the transported scalar `a = (ϕ = e, Rf = the ratio with its halos
# filled)`; the body of ν and its arguments after the output (`νbody`,
# `νargs`); the reconstruction `r` (nothing: no transport); the closure's
# arrays and coefficients; the rings `rg`; the budget sums of every row
# `rowsums`; the interior `J1:J2`, `Kb:Kt` (storage).
struct _TKEUpdate{P<:NamedTuple, INPLACE} <: BlockedUpdate
    p::P
end
_TKEUpdate(p::NamedTuple, inplace::Bool) = _TKEUpdate{typeof(p), inplace}(p)

update_lags(op::_TKEUpdate) = (op.p.lag, op.p.lz)

# the rings of slab s as 3-D arrays
@inline _flux_views(f::_FluxRings, s, ko) =
    (R = _Ring{3}(f.R, ko, s),
     FxH = _Ring{2}(f.FxH, ko, s), FxL = _Ring{2}(f.FxL, ko, s),
     FzH = _Ring{2}(f.FzH, ko, s), FzL = _Ring{2}(f.FzL, ko, s),
     FyH = _Ring{3}(f.FyH, ko, s), FyL = _Ring{3}(f.FyL, ko, s))
@inline function _rings(p, s, k0)
    rg = p.rg; ko = k0 - 2
    return (ν = _Ring{3}(rg.ν, ko, s), f = _flux_views(rg.f, s, ko), td = _Line(rg.td, s))
end

# the interior planes among k0−1:k1+1
@inline _inner_planes(p, k0, k1) = max(k0 - 1, p.Kb):min(k1 + 1, p.Kt)

# a halo row (J, K) (or the two halo ends of an interior row) of a field
# into a ring
@noinline function _copy_row!(dst, src, ir, J, K)
    @inbounds for I in ir
        dst[I, J, K] = src[I, J, K]
    end
    return nothing
end
@inline function _copy_ends!(dst, src, ii, J, K)
    @inbounds dst[first(ii) - 1, J, K] = src[first(ii) - 1, J, K]
    @inbounds dst[last(ii) + 1, J, K] = src[last(ii) + 1, J, K]
    return nothing
end

# ν of row J, planes ka:kb into the ring ν (interior: the closure's body
# `p.νbody`; halo: copied)
function _ν_row!(p, ν, J, ka, kb, ii)
    iw = first(ii)-1:last(ii)+1
    for K in ka:kb
        if p.J1 <= J <= p.J2 && p.Kb <= K <= p.Kt
            Backends._row!(p.νbody, ii, J, K, ν, p.νargs...)
            _copy_ends!(ν, p.νf, ii, J, K)
        else
            _copy_row!(ν, p.νf, iw, J, K)
        end
    end
    return nothing
end

# The flux-limited advection of one scalar `a = (ϕ, Rf)` through its rings
# `f` (`_flux_views`), the steps of `advect_positive!`: the reconstructed
# and upwind face fluxes, the limiter's ratio, the limited fluxes.

# the y faces J (between rows J−1 and J), planes k0−1:k1+1 of the interior
function _yfaces_row!(p, a, f, J, k0, k1, ii)
    for K in _inner_planes(p, k0, k1)
        Backends._row!(_b_rflux_y!, ii, J, K, f.FyH, p.v, a.ϕ, p.r)
        Backends._row!(_b_uflux_y!, ii, J, K, f.FyL, p.v, a.ϕ)
    end
    return nothing
end

# The front of a scalar's pipeline at row Jn: the x and z face fluxes and
# the ratio of row Jn, the y faces between Jn and Jn+1 (halo rows: the
# ratio copied), then the y faces between Jn−1 and Jn limited.
function _flux_front!(p, a, f, Jn, k0, k1, ii)
    iw = first(ii)-1:last(ii)+1
    kin = _inner_planes(p, k0, k1)
    if Jn <= p.J2
        xf = first(ii):last(ii)+1
        for K in kin
            Backends._row!(_b_rflux_x!, xf, Jn, K, f.FxH, p.u, a.ϕ, p.r)
            Backends._row!(_b_uflux_x!, xf, Jn, K, f.FxL, p.u, a.ϕ)
        end
        for K in max(k0 - 1, p.Kb):min(k1 + 2, p.Kt + 1)
            Backends._row!(_b_rflux_z!, ii, Jn, K, f.FzH, p.w, a.ϕ, p.r)
            Backends._row!(_b_uflux_z!, ii, Jn, K, f.FzL, p.w, a.ϕ)
        end
        _yfaces_row!(p, a, f, Jn + 1, k0, k1, ii)
        for K in k0-1:k1+1
            if K in kin
                Backends._row!(_b_limit_ratio!, ii, Jn, K, f.R, f.FxH, f.FyH, f.FzH, a.ϕ,
                               f.FxL, f.FyL, f.FzL, p.Δt, p.idx, p.idy, p.dzm)
                _copy_ends!(f.R, a.Rf, ii, Jn, K)
            else
                _copy_row!(f.R, a.Rf, iw, Jn, K)
            end
        end
    else
        for K in k0-1:k1+1
            _copy_row!(f.R, a.Rf, iw, Jn, K)
        end
    end
    z = zero(eltype(a.ϕ))
    for K in k0:k1
        Backends._row!(_b_limit_y!, ii, Jn, K, f.FyH, f.R, f.FyL, z)
    end
    return nothing
end

# the start of a scalar's pipeline in a slab: the ratio of the halo row
# before the first, the y faces of the first row, the front at the first row
function _begin_flux!(p, a, f, J1, k0, k1, ii)
    iw = first(ii)-1:last(ii)+1
    for K in k0-1:k1+1
        _copy_row!(f.R, a.Rf, iw, J1 - 1, K)
    end
    _yfaces_row!(p, a, f, J1, k0, k1, ii)
    _flux_front!(p, a, f, J1, k0, k1, ii)
    return nothing
end

# the x and z faces of row J limited, each once (planes k0:k1, the z faces
# k0:k1+1)
function _limit_xz!(f, J, k0, k1, ii, z)
    for K in k0:k1+1
        Backends._row!(_b_limit_z!, ii, J, K, f.FzH, f.R, f.FzL, z)
    end
    xf = first(ii):last(ii)+1
    for K in k0:k1
        Backends._row!(_b_limit_x!, xf, J, K, f.FxH, f.R, f.FxL, z)
    end
    return nothing
end

# the halo row of ν before the first row of a slab, and ν of the first row
function _begin_ν!(p, ν, J1, k0, k1, ii)
    iw = first(ii)-1:last(ii)+1
    for K in k0-1:k1+1
        _copy_row!(ν, p.νf, iw, J1 - 1, K)
    end
    _ν_row!(p, ν, J1, k0 - 1, k1 + 1, ii)
    return nothing
end

function begin_slab!(op::_TKEUpdate, s, k0, k1, ii, jj)
    p = op.p
    p.r === nothing && return nothing             # no transport: nothing ahead
    rs = _rings(p, s, k0)
    _begin_ν!(p, rs.ν, first(jj), k0, k1, ii)
    _begin_flux!(p, p.a, rs.f, first(jj), k0, k1, ii)
    return nothing
end

@noinline function _zero_line!(td, ii)
    @inbounds for I in ii
        td[I, 1, 1] = zero(eltype(td))
    end
    return nothing
end

# the tendency of a flux-limited scalar ϕ on row (J, K) into the line td:
# the divergence of its limited fluxes `f` and the diffusion with 2ν, in
# the order of the separate loops (0, −div, + diffusion)
@inline function _tendency_row!(p, td, ϕ, f, ν, J, K, ii)
    _zero_line!(td, ii)
    Backends._row!(_b_flux_div!, ii, J, K, td, f.FxH, f.FyH, f.FzH, p.idx, p.idy, p.dzm)
    _diffusion_row!(p, td, ϕ, ν, J, K, ii)
    return nothing
end
@inline function _diffusion_row!(p, td, ϕ, ν, J, K, ii)
    K2 = _Doubled(ν)
    if p.mask === nothing
        Backends._row!(_b_eddy_scalar!, ii, J, K, td, ϕ, K2, K2, one(eltype(ϕ)), p.m)
    else
        Backends._row!(_b_eddy_scalar_masked!, ii, J, K, td, ϕ, K2, K2, p.mask, nothing, nothing,
                       p.idx, p.idy, p.idz)
    end
    return nothing
end

function compute_rows!(op::_TKEUpdate{P,INPLACE}, ring, edge, r, s, J, k0, k1, ii, lz) where {P, INPLACE}
    p = op.p
    rs = _rings(p, s, k0)
    if p.r === nothing                            # no transport: ν and the update, in place
        _ν_row!(p, rs.ν, J, k0, k1, ii)            # (planes k0:k1: no neighbour read)
        for K in k0:k1
            out = INPLACE ? p.e : _rowslot(ring, edge, r, J, K, k0, k1, lz, s)
            _update_row!(p, rs, J, K, ii, _ReadWrite(p.e, out), p.ztd)
        end
        return nothing
    end
    _ν_row!(p, rs.ν, J + 1, k0 - 1, k1 + 1, ii)    # the front at row J + 1
    _flux_front!(p, p.a, rs.f, J + 1, k0, k1, ii)
    _limit_xz!(rs.f, J, k0, k1, ii, zero(eltype(p.e)))
    for K in k0:k1
        _tendency_row!(p, rs.td, p.e, rs.f, rs.ν, J, K, ii)
        _update_row!(p, rs, J, K, ii, _ReadWrite(p.e, _rowslot(ring, edge, r, J, K, k0, k1, lz, s)), rs.td)
    end
    return nothing
end

# field f of the driver's slot of row (J, K)
@inline function _rowslot(ring, edge, r, J, K, k0, k1, lz, s, f = 1)
    buf, a, b = _row_slot(ring, edge, r, J, K, k0, k1, lz)
    return _Slot(buf, a, b, f, s)
end

# the update of row (J, K) and its budget sums (the row loop of `_update!`)
@inline function _update_row!(p, rs, J, K, ii, ed, td)
    sums = _tke_update_row(J, K, ii, ed, rs.ν, p.n2, p.dd, td, p.pw, p.Δk, p.vrel, p.l_wall,
                           p.first_level, p.mask, p.par, p.exact)
    @inbounds p.rowsums[J - p.J1 + 1, K - p.Kb + 1] = sums
    return nothing
end

# the new e from the driver's buffers (nothing to write when updated in place)
write_row!(::_TKEUpdate{P,true}, ring, a, b, s, j, k, ii) where {P} = nothing
write_point!(::_TKEUpdate{P,true}, edge, i, j, k, e, s) where {P} = nothing
write_row!(op::_TKEUpdate{P,false}, ring, a, b, s, j, k, ii) where {P} =
    _write_field_row!(op.p.e, ring, a, b, 1, s, j, k, ii)
# row (j, k) of a field from field f of the driver's ring
@noinline function _write_field_row!(ed, ring, a, b, f, s, j, k, ii)
    @inbounds @simd ivdep for i in ii
        ed[i, j, k] = ring[i, a, b, f, s]
    end
    return nothing
end
@inline function write_point!(op::_TKEUpdate{P,false}, edge, i, j, k, e, s) where {P}
    @inbounds op.p.e[i, j, k] = edge[i, j, e, 1, s]
    return nothing
end

# -------------------------------------------------------------------------
# the shell: ν and the ratio of the interior cells next to the boundary
# -------------------------------------------------------------------------

# the six one-cell boxes of the interior next to its faces
function _shell_boxes((ii, jj, kk))
    return ((first(ii):first(ii), jj, kk), (last(ii):last(ii), jj, kk),
            (ii, first(jj):first(jj), kk), (ii, last(jj):last(jj), kk),
            (ii, jj, first(kk):first(kk)), (ii, jj, last(kk):last(kk)))
end

# ν (and κt) of the shell cells into c.ν, c.κt, then ν's halos filled as
# the separate loops fill them
function _shell_coefficients!(c::TKEClosure, bcs)
    g = mesh(c.e)
    b = Backends.backend(c.e.data)
    for box in _shell_boxes(interior(g))
        launch!(b, _coefficients_body!, box, c.ν.data, c.κt.data, c.e.data, c.N2.data, c.Δk, c.mask,
                c.l_wall, g.Hx, g.Hy, g.Hz, c.c_k, c.c_l)
    end
    fill_halos!(c.ν, bcs)
    return nothing
end

# the limiter's ratio of the shell cells of a scalar ϕ (reconstruction r)
# into `ratio`, each face flux evaluated by its body where the ratio reads
# it (`_FaceFlux`), then its halos filled with ϕ's conditions `ebcs`, as
# `advect_positive!` fills them
_shell_ratio!(c::TKEClosure{T}, sim, Δt::T, ebcs) where {T} =
    _shell_ratio!(c.ws.ratio, c.e, c.recon, sim, Δt, ebcs)
function _shell_ratio!(ratio::Field, ϕ::Field{CellCenter,T}, r, sim, Δt::T, ebcs) where {T}
    g = mesh(ϕ)
    b = Backends.backend(ϕ.data)
    ed = ϕ.data; ud, vd, wd = sim.u.data, sim.v.data, sim.w.data
    idx = 1 / g.dx; idy = 1 / g.dy; dzm = _dz_metric(g)
    fx = _FaceFlux{T}(_b_rflux_x!, (ud, ed, r)); lx = _FaceFlux{T}(_b_uflux_x!, (ud, ed))
    fy = _FaceFlux{T}(_b_rflux_y!, (vd, ed, r)); ly = _FaceFlux{T}(_b_uflux_y!, (vd, ed))
    fz = _FaceFlux{T}(_b_rflux_z!, (wd, ed, r)); lz = _FaceFlux{T}(_b_uflux_z!, (wd, ed))
    for box in _shell_boxes(interior(g))
        launch!(b, _b_limit_ratio!, box, ratio.data, fx, fy, fz, ed, lx, ly, lz, Δt, idx, idy, dzm)
    end
    fill_halos!(ratio, ebcs)
    return nothing
end

# -------------------------------------------------------------------------
# the pass
# -------------------------------------------------------------------------

# The reason the blocked pass cannot run for (c, sim), or nothing. Open
# faces need nothing of their own: ν and the ratio carry no boundary data,
# so an open face fills their halos as the mirror of the first interior
# layer, which the boundary shell reproduces (tested).
function _tke_blocked_blocker(c::TKEClosure, sim)
    Backends.backend(c.e.data) isa Backends.CPUBackend ||
        return "the cache-blocked TKE update runs on the CPU backend only"
    return nothing
end

# The scratch of the blocked pass (driver buffers, rings), kept in the closure.
struct _TKEBlockedScratch{B<:SweepBuffers, R<:_TKERings}
    bufs::B
    rings::R
end

"""
    tke_blocked_lags(c::TKEClosure) -> (lag, lz)

The ring lag (j-rows) and the edge planes per slab end of the
cache-blocked TKE update of `c` (#554): with transport and a
reconstruction of halo `hr`, the new e reads the old e `hr + 1` cells
away, so `lz = hr + 1` and `lag = max(1, hr − 2)` (the pipeline runs two
rows ahead); without transport the update is pointwise, `(1, 1)`.
"""
function tke_blocked_lags(c::TKEClosure)
    c.transport || return (1, 1)
    hr = required_halo(c.recon)
    return (max(1, hr - 2), hr + 1)
end

# The e update of `advance_sgs!` as one cache-blocked pass (see the top of
# this file); N² and the dual strain (and the wall production) as separate
# loops before it. `ns`: the slab count (default: one per thread);
# `reverse_slabs`: the driver's test hook.
function _advance_blocked!(c::TKEClosure{T}, sim, Δt::T; ns = nothing, reverse_slabs::Bool = false) where {T}
    g = mesh(c.e)
    bcs = sim.bcs
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    _stratification!(c, sim)
    dual_strain!(c.D, sim.u, sim.v, sim.w, bcs)
    walls = c.walls[] !== nothing
    if walls                                      # the stencil share reads the stored ν
        _coefficients!(c.ν, c.κt, c, c.N2)
        fill_halos!(c.ν, bcs)
        _wall_production!(c, sim, bcs)
    end
    ebcs = passive_scalar_bcs(sim.θ_bcs)
    if c.transport
        _check_stretched_recon(c.recon, g)
        _check_halo(c.recon, g)
        fill_halos!(c.e, ebcs)
        walls || _shell_coefficients!(c, bcs)
        _shell_ratio!(c, sim, Δt, ebcs)
    end
    ii, jj, kk = interior(g)
    lag, lz = tke_blocked_lags(c)
    b = Backends.backend(c.e.data)
    nsl = ns === nothing ? sweep_slabs(b, length(ii) * length(jj) * length(kk), length(kk); lz = lz) : ns
    old = c.blocked[]
    bufs = sweep_buffers(old isa _TKEBlockedScratch ? old.bufs : nothing, c.e.data, 1, nsl, length(kk);
                         lag = lag, lz = lz)
    rings = _tke_rings(old isa _TKEBlockedScratch ? old.rings : nothing, c.e.data, cld(length(kk), nsl) + 2, nsl)
    (old isa _TKEBlockedScratch && old.bufs === bufs && old.rings === rings) ||
        (c.blocked[] = _TKEBlockedScratch(bufs, rings))
    par = _update_params(c, Δt, _surface_buoyancy(c, sim))
    # the wall production and the dissipation form as types, split here so
    # the update and its launch are concrete (no dynamic call, no allocation)
    exact = c.dissipation === :exact
    if walls
        exact ? _tke_pass!(c, sim, Δt, bufs, rings, par, b, g, c.Pw.data, Val(true), reverse_slabs) :
                _tke_pass!(c, sim, Δt, bufs, rings, par, b, g, c.Pw.data, Val(false), reverse_slabs)
    else
        exact ? _tke_pass!(c, sim, Δt, bufs, rings, par, b, g, _Zeros{T}(), Val(true), reverse_slabs) :
                _tke_pass!(c, sim, Δt, bufs, rings, par, b, g, _Zeros{T}(), Val(false), reverse_slabs)
    end
    _book_sums!(c, par.V0)
    c.surface === nothing || _surface!(c, Δt)
    return c
end

function _tke_pass!(c::TKEClosure{T}, sim, Δt::T, bufs, rings, par, b, g, pw, exact::Val,
                    reverse_slabs::Bool) where {T}
    ii, jj, kk = interior(g)
    lag, lz = tke_blocked_lags(c)
    # without transport (no reconstruction: `c.recon === nothing`, known
    # from the closure's type) the update is pointwise and writes e in place
    inplace = c.recon === nothing
    p = (e = c.e.data, νf = c.ν.data, a = (ϕ = c.e.data, Rf = c.transport ? c.ws.ratio.data : c.ν.data),
         νbody = _coefficients_body!,
         νargs = (_Discard{T}(), c.e.data, c.N2.data, c.Δk, c.mask, c.l_wall, g.Hx, g.Hy, g.Hz, c.c_k, c.c_l),
         u = sim.u.data, v = sim.v.data, w = sim.w.data, n2 = c.N2.data, dd = c.D.data, pw = pw,
         r = c.recon, ztd = _Zeros{T}(),
         Δk = c.Δk, vrel = c.vrel, l_wall = c.l_wall, first_level = c.first_level, mask = c.mask,
         Hx = g.Hx, Hy = g.Hy, Hz = g.Hz, ck = c.c_k, cl = c.c_l, Δt = Δt,
         idx = 1 / g.dx, idy = 1 / g.dy, idz = _idz_mask(g), dzm = _dz_metric(g), m = grid_metrics(g),
         par = par, exact = exact, rg = rings, rowsums = c.rowsums,
         J1 = first(jj), J2 = last(jj), Kb = first(kk), Kt = last(kk), lag = lag, lz = lz)
    blocked_update!(bufs, _TKEUpdate(p, inplace), b, g; reverse_slabs = reverse_slabs)
    return nothing
end

# 1/Δz of the masked diffusion (a RegularGrid only; never read otherwise)
_idz_mask(g::RegularGrid) = 1 / g.dz
_idz_mask(g::StretchedGrid) = nothing

# --- which form ran: the loop-fusion setting of the step core (#533) and
# the record in its `SweepReport` (`Simulations.sweep_report`)
_fusion_setting(sim) = hasproperty(sim, :workspaces) && sim.workspaces isa StepWorkspaces ?
    sim.workspaces.fused.loop_fusion : :if_possible
function _note_tke!(sim, blocked::Bool, reason)
    (hasproperty(sim, :workspaces) && sim.workspaces isa StepWorkspaces) || return nothing
    r = sim.workspaces.fused
    if blocked
        r.tke_blocked += 1
    else
        r.tke_separate += 1
        reason === nothing || (r.tke_reason = reason)
    end
    return nothing
end
