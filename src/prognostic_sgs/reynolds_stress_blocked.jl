# =========================================================================
# The update of the Reynolds-stress closure as one cache-blocked pass
# (#560). Included by PrognosticSGS after the closure and after the TKE
# pass (prognostic_sgs/tke_blocked.jl), whose pipeline it runs.
# =========================================================================
#
# `advance_sgs!` of a ReynoldsStressClosure with `transport` runs, as
# separate loops (the reference, `_rsm_advance_separate!`): N², the halo
# fill of the six stresses, ν, its halo fill, K = 2ν, then for each stress
# its tendency (the flux-limited advection of the diagonals,
# `advect_positive!`; the plain advection of the off-diagonals; the
# diffusion with 2ν), and the source update of all six (`_rsm_update!`).
# Here everything after the halo fill of the stresses runs row by row on
# the blocked driver (`FiniteVolume.blocked_update!`) with the pipeline of
# the TKE pass (#554):
#
#   step j:  ν of row j+1; for each diagonal τ₁₁, τ₂₂, τ₃₃ the front of
#            its flux pipeline (`_flux_front!`: its face fluxes and ratio
#            of row j+1, its y faces between rows j and j+1 limited) and
#            its x and z faces of row j limited;
#   row j:   the six tendencies (the diagonals: the divergence of their
#            limited fluxes; the off-diagonals: the plain advection; all:
#            the diffusion with 2ν), then the source update of the row
#            (`_rsm_update_row`, the row of the separate loops), its new
#            stresses into the driver's ring (six fields) or edge buffer.
#
# Every value is computed by the point body of the separate loop, reached
# through the TKE pass's adapters on the rings: the same values as the
# separate loops, and the budget sums are the same row sums in the same
# order.
#
# Footprint. The limited fluxes need `lz = hr + 1` edge planes and a lag of
# `max(1, hr − 2)` rows (see tke_blocked.jl). The plain advection of the
# off-diagonals reads the old stresses of row j − hr directly (its y faces
# are not pipelined), so the ring holds `lag = max(1, hr − 2, hr)` = hr
# rows behind (3 with WENO-Z5, 1 centred): `rsm_blocked_lags`.
#
# Halos. As the TKE pass: ν and the three ratios of the interior cells next
# to the boundary (the shell) are computed before the sweep and their halos
# filled as the separate loops fill them, then copied into the rings. The
# three ratios need a field each (the separate loops reuse one workspace):
# the first is the closure's `ws.ratio`, two more are kept in the pass's
# scratch.

using ..AdvectionSchemes: _b_recon!
using ..Operators.FiniteVolume: _b_advect_centered!

# -------------------------------------------------------------------------
# the plain advection of the off-diagonals, as a row
# -------------------------------------------------------------------------

# The row form of `advection_scheme(name)` (the closure's `c.adv`): the
# same body as its whole-field launch. `:weno5` (`advect_weno5!`) has none.
struct _ReconAdv{R, T}
    r::R
    idx::T; idy::T; idz::T
end
struct _CentredAdv{M}
    m::M
end
@inline _plain_adv_row!(a::_ReconAdv, td, ii, J, K, u, v, w, ϕ) =
    Backends._row!(_b_recon!, ii, J, K, td, u, v, w, ϕ, a.r, a.idx, a.idy, a.idz)
@inline _plain_adv_row!(a::_CentredAdv, td, ii, J, K, u, v, w, ϕ) =
    Backends._row!(_b_advect_centered!, ii, J, K, td, u, v, w, ϕ, a.m)

# (`advection_scheme`: :centered → `advect_centered!`, the reconstructed
# schemes → `_advect_reconstructed!` with `_reconstruction(name)` — the
# closure's `recon` — and 1/Δx, 1/Δy, 1/Δz)
function _plain_advection(c::ReynoldsStressClosure)
    g = mesh(c.τ[1])
    c.advection === :centered && return _CentredAdv(grid_metrics(g))
    c.advection in (:weno_js5, :weno_z5, :weno_z7, :teno5, :teno8) &&
        return _ReconAdv(c.recon, 1 / g.dx, 1 / g.dy, 1 / g.dz)
    return nothing
end

# -------------------------------------------------------------------------
# the rings and the update
# -------------------------------------------------------------------------

# Per slab: ν (three rows), the flux rings of the three diagonals, one row
# of each of the six tendencies (see `_TKERings`).
mutable struct _RSCRings{A<:AbstractArray, M<:AbstractMatrix}
    const ν::A
    const f::NTuple{3,_FluxRings{A}}
    const td::NTuple{6,M}
end
function _rsc_rings(old, data::AbstractArray{T,3}, nkl::Int, ns::Int) where {T}
    return old isa _RSCRings{_ring_type(data), _row_type(data)} ? _rsc_rings_(old, data, nkl, ns) :
                                                                   _rsc_rings_(nothing, data, nkl, ns)
end
function _rsc_rings_(old, data::AbstractArray{T,3}, nkl::Int, ns::Int) where {T}
    sx = size(data, 1)
    old isa _RSCRings && size(old.ν, 1) == sx && size(old.ν, 2) >= nkl && size(old.ν, 4) == ns &&
        return old
    r(nr, extra = 0) = similar(data, sx, nkl + extra, nr, ns)
    l() = similar(data, sx, ns)
    return _RSCRings(r(3), (_flux_rings(r), _flux_rings(r), _flux_rings(r)), (l(), l(), l(), l(), l(), l()))
end

# Everything the pipeline reads (a NamedTuple): the stresses `τ` (six
# arrays) and the three diagonals as transported scalars `a` (ϕ and its
# ratio field), `νf` (ν with its halos filled), the row of ν `νrow!` and its
# arguments, `u, v, w`, `θ`, `q`, `n2`, the edges `E`, the closure `c` and
# the update's parameters `par`; the reconstruction `r`, the plain
# advection `padv`; the rings `rg`, the row sums `rowsums`; the interior.
struct _RSCUpdate{P<:NamedTuple} <: BlockedUpdate
    p::P
end

update_lags(op::_RSCUpdate) = (op.p.lag, op.p.lz)

@inline function _rsc_views(p, s, k0)
    rg = p.rg; ko = k0 - 2
    return (ν = _Ring{3}(rg.ν, ko, s), f = map(f -> _flux_views(f, s, ko), rg.f),
            td = map(t -> _Line(t, s), rg.td))
end

# ν of the interior row (J, K) into its ring: the body of the separate
# loop. (N² is read from c.N2, computed before the pass by the whole-field
# loop: the in-pass N² of the TKE pass (#556) would save about 1.5 ms of a
# 110–270 ms update limited by its 6×6 solve.)
@inline _rsm_ν_row!(p, rs, J, K, ii) = Backends._row!(_rsm_viscosity_body!, ii, J, K, rs.ν, p.νargs...)

function begin_slab!(op::_RSCUpdate, s, k0, k1, ii, jj)
    p = op.p
    rs = _rsc_views(p, s, k0)
    _begin_ν!(p, rs, first(jj), k0, k1, ii)
    for m in 1:3
        _begin_flux!(p, p.a[m], rs.f[m], first(jj), k0, k1, ii)
    end
    return nothing
end

function compute_rows!(op::_RSCUpdate, ring, edge, r, s, J, k0, k1, ii, lz)
    p = op.p
    rs = _rsc_views(p, s, k0)
    _ν_row!(p, rs, J + 1, k0 - 1, k1 + 1, ii)     # the front at row J + 1
    z = zero(eltype(p.u))
    for m in 1:3
        _flux_front!(p, p.a[m], rs.f[m], J + 1, k0, k1, ii)
        _limit_xz!(rs.f[m], J, k0, k1, ii, z)
    end
    for K in k0:k1
        for m in 1:3                                  # the separate loops' order per stress
            _tendency_row!(p, rs.td[m], p.τ[m], rs.f[m], rs.ν, J, K, ii)
        end
        for m in 4:6
            _zero_line!(rs.td[m], ii)
            _plain_adv_row!(p.padv, rs.td[m], ii, J, K, p.u, p.v, p.w, p.τ[m])
            _diffusion_row!(p, rs.td[m], p.τ[m], rs.ν, J, K, ii)
        end
        buf, a, b = _row_slot(ring, edge, r, J, K, k0, k1, lz)
        τrw = ntuple(m -> _ReadWrite(p.τ[m], _Slot(buf, a, b, m, s)), Val(6))
        sums = _rsm_update_row(J, K, ii, τrw, rs.td, p.u, p.v, p.w, p.θ, p.q, p.n2, p.E, p.c, p.par)
        @inbounds p.rowsums[J - p.J1 + 1, K - p.Kb + 1] = sums
    end
    return nothing
end

# the new stresses from the driver's buffers (field m = stress m)
function write_row!(op::_RSCUpdate, ring, a, b, s, j, k, ii)
    τ = op.p.τ
    for m in 1:6
        _write_field_row!(τ[m], ring, a, b, m, s, j, k, ii)
    end
    return nothing
end
@inline function write_point!(op::_RSCUpdate, edge, i, j, k, e, s)
    τ = op.p.τ
    @inbounds for m in 1:6
        τ[m][i, j, k] = edge[i, j, e, m, s]
    end
    return nothing
end

# -------------------------------------------------------------------------
# the pass
# -------------------------------------------------------------------------

# The reason the blocked pass cannot run for (c, sim), or nothing (with
# transport; without, the update is one pointwise pass already).
function _rsm_blocked_blocker(c::ReynoldsStressClosure, sim)
    Backends.backend(c.τ[1].data) isa Backends.CPUBackend ||
        return "the cache-blocked Reynolds-stress update runs on the CPU backend only"
    _plain_advection(c) === nothing &&
        return "the off-diagonal advection scheme :$(c.advection) has no row form"
    return nothing
end

# Whether `loop_fusion = "if_possible"` takes the blocked pass (nothing) or
# keeps the separate loops (the reason). Measured on 200×200×60 (#560, see
# the roadmap): the update is limited by its arithmetic (the 6×6 solve), so
# one pass saves little traffic; it was faster only with centred advection
# on several threads (12 threads 110 against 155 ms), level at 6 threads,
# and slower at one thread (+6 % centred, +10–24 % WENO-Z5) and with
# WENO-Z5 at 12 threads (+25 %, +58 % with a mask: 60 levels give
# nk ÷ 2lz = 7 slabs, so 5 threads idle). `"always"` still runs it.
function _rsm_blocked_slower(c::ReynoldsStressClosure, sim)
    g = mesh(c.τ[1])
    ii, jj, kk = interior(g)
    serial = Backends.runs_serially(Backends.backend(c.τ[1].data), length(ii) * length(jj) * length(kk))
    c.advection === :centered && !serial && return nothing
    return "the separate loops measured as fast or faster here (#560: the blocked " *
           "Reynolds-stress update pays only with centred advection on several threads); " *
           "loop_fusion = \"always\" runs it"
end

# The scratch of the blocked pass (driver buffers, rings, the second and
# third ratio fields), kept in the closure.
struct _RSCBlockedScratch{B<:SweepBuffers, R<:_RSCRings, F<:Field}
    bufs::B
    rings::R
    ratio::NTuple{2,F}
end

"""
    rsm_blocked_lags(c::ReynoldsStressClosure) -> (lag, lz)

The ring lag (j-rows) and the edge planes per slab end of the
cache-blocked Reynolds-stress update of `c` (#560), for a reconstruction
of halo `hr`: the limited fluxes of the diagonals need `lz = hr + 1`
(as [`tke_blocked_lags`](@ref)); the plain advection of the off-diagonals
reads the stresses `hr` rows back, so `lag = max(1, hr)`.
"""
function rsm_blocked_lags(c::ReynoldsStressClosure)
    hr = required_halo(c.recon)
    return (max(1, hr - 2, hr), hr + 1)
end

# The update of `advance_sgs!` with transport as one cache-blocked pass (see
# the top of this file); N² and the halos before it. `ns`: the slab count
# (default: one per thread); `reverse_slabs`: the driver's test hook;
# `lags`: a test hook (a ring shorter than the footprint must disagree).
function _rsm_advance_blocked!(c::ReynoldsStressClosure{T}, sim, Δt::T; ns = nothing,
                               reverse_slabs::Bool = false,
                               lags::NTuple{2,Int} = rsm_blocked_lags(c)) where {T}
    g = mesh(c.τ[1])
    bcs = sim.bcs
    sbcs = _scalar_bcs(sim)
    fill_halos!(sim.u, bcs); fill_halos!(sim.v, bcs); fill_halos!(sim.w, bcs)
    _stratification!(c, sim)
    _fill_stress_halos!(c, sbcs)
    _check_halo(c.recon, g)
    ii, jj, kk = interior(g)
    lag, lz = lags
    b = Backends.backend(c.τ[1].data)
    nsl = ns === nothing ? sweep_slabs(b, length(ii) * length(jj) * length(kk), length(kk); lz = lz) : ns
    old = c.blocked[]
    sc = old isa _RSCBlockedScratch ? old : nothing
    bufs = sweep_buffers(sc === nothing ? nothing : sc.bufs, c.τ[1].data, 6, nsl, length(kk); lag = lag, lz = lz)
    rings = _rsc_rings(sc === nothing ? nothing : sc.rings, c.τ[1].data, cld(length(kk), nsl) + 2, nsl)
    ratio = _rsc_ratios(sc, c, typeof(c.ws.ratio))
    (sc !== nothing && sc.bufs === bufs && sc.rings === rings) ||
        (c.blocked[] = _RSCBlockedScratch(bufs, rings, ratio))
    # the shell: ν (halos with the velocity conditions) and the ratios
    # (halos with the stresses' conditions), as the separate loops fill them
    for box in _shell_boxes(interior(g))
        launch!(b, _rsm_viscosity_body!, box, c.ν.data, map(x -> x.data, c.τ), c.N2.data, c,
                g.Hx, g.Hy, g.Hz)
    end
    fill_halos!(c.ν, bcs)
    ratios = (c.ws.ratio, ratio...)
    for m in 1:3
        _shell_ratio!(ratios[m], c.τ[m], c.recon, sim, Δt, sbcs)
    end
    # (q and the plain advection are each one of two types: split here, so
    # the pass is concrete)
    q = _qv(c, sim)
    _rsm_pass!(c, sim, Δt, bufs, rings, ratios, _plain_advection(c), q === nothing ? nothing : q.data,
               b, g, lag, lz, reverse_slabs)
    _rsm_book!(c, g.dx * g.dy * g.dz)
    return c
end

# the second and third ratio fields (kept; of the type `F` of the
# closure's `ws.ratio`, so the scratch is concrete)
_rsc_ratios(sc, c, ::Type{F}) where {F} =
    sc isa _RSCBlockedScratch ? sc.ratio::NTuple{2,F} :
        (Field(CellCenter(), mesh(c.τ[1])), Field(CellCenter(), mesh(c.τ[1])))::NTuple{2,F}

function _rsm_pass!(c::ReynoldsStressClosure{T}, sim, Δt::T, bufs, rings, ratios, padv, qd, b, g,
                    lag::Int, lz::Int, reverse_slabs::Bool) where {T}
    ii, jj, kk = interior(g)
    τd = map(x -> x.data, c.τ)
    p = (τ = τd, a = ntuple(m -> (ϕ = τd[m], Rf = ratios[m].data), Val(3)), νf = c.ν.data,
         νrow! = _rsm_ν_row!, νargs = (τd, c.N2.data, c, g.Hx, g.Hy, g.Hz),
         u = sim.u.data, v = sim.v.data, w = sim.w.data, θ = sim.θ.data,
         q = qd, n2 = c.N2.data,
         E = _Edges(c.mask, bounded_directions(sim.bcs), g), c = c, par = _rsm_update_params(c, sim, Δt),
         r = c.recon, padv = padv, mask = c.mask, Δt = Δt,
         idx = 1 / g.dx, idy = 1 / g.dy, idz = 1 / g.dz, dzm = _dz_metric(g), m = grid_metrics(g),
         rg = rings, rowsums = c.rowsums,
         J1 = first(jj), J2 = last(jj), Kb = first(kk), Kt = last(kk), lag = lag, lz = lz)
    blocked_update!(bufs, _RSCUpdate(p), b, g; reverse_slabs = reverse_slabs)
    return nothing
end

# which form ran, in the step core's `SweepReport` (`Simulations.sweep_report`)
function _note_rsm!(sim, blocked::Bool, reason)
    (hasproperty(sim, :workspaces) && sim.workspaces isa StepWorkspaces) || return nothing
    r = sim.workspaces.fused
    if blocked
        r.rsm_blocked += 1
    else
        r.rsm_separate += 1
        reason === nothing || (r.rsm_reason = reason)
    end
    return nothing
end
