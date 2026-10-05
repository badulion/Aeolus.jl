# N-inflow-b (#258) — the statistics that drive the synthetic inflow.
#
# Included inside `module Transition`, after synthetic_inflow.jl.
#
# The generator of N-inflow-a (`DigitalFilterGenerator` + `apply_lund!`)
# needs, at every child face, a Reynolds-stress tensor R_ij(z), integral
# lengths L_s(z), L_z(z) and an integral time T(z). An `InflowStatistics`
# source supplies them through two generics, whatever the parent is:
#
#   ParentResolvedStatistics   an Aeolus LES parent: running mean of its
#                              resolved stresses + the SGS stress of its
#                              closure; scale-gap mode (default) keeps only
#                              what the child resolves and the parent
#                              does not deliver through the interpolation.
#   ParametrizedStatistics     a parent with no resolved turbulence (a
#                              mesoscale model, #58): similarity profiles
#                              in u*, w*, z_i, L (Panofsky & Dutton 1984;
#                              Lenschow et al. 1980; Rotach et al. 1996).
#
# Realizability goes through `lund_coefficients` (N-inflow-a); a tensor it
# rejects is projected onto the realizable set and the projection is
# booked in the source's log.

# =========================================================================
# The contract
# =========================================================================

"""
    InflowStatistics

Supertype of the sources of synthetic-inflow statistics (N-inflow-b,
#258). An implementer provides, at heights `z` (m, a number or a vector)
and time `t` (s),

| method | returns |
| :-- | :-- |
| [`inflow_stresses`](@ref)`(src, z, t)` | ``R_{ij}`` as 6-tuples ``(R_{11}, R_{22}, R_{33}, R_{12}, R_{13}, R_{23})`` (m² s⁻²), realizable |
| [`inflow_lengths`](@ref)`(src, z, t)` | `(L_s, L_z, T)`: integral lengths along the face and in the vertical (m), integral time (s), per component |
| [`update_inflow_statistics!`](@ref)`(src, Δt)` | advance a running statistic by a parent step (default: no-op) |

Components are in the **face frame** of the generator: (normal, along the
face, vertical) — for a west or east face ``(u, v, w)``, for a south or
north face ``(v, u, w)``. The along-face component is unsigned (the other
horizontal velocity), so ``R_{12}`` is ``\\overline{u'v'}`` on every face.
The output feeds [`DigitalFilterGenerator`](@ref) (`L_s`, `L_z`, `T` as
3 × nz matrices for a vector `z`) and [`apply_lund!`](@ref) through
[`inflow_lund_coefficients`](@ref).

Providers: [`ParentResolvedStatistics`](@ref), [`ParametrizedStatistics`](@ref).
"""
abstract type InflowStatistics end

"""
    inflow_stresses(src::InflowStatistics, z, t) -> R

The Reynolds-stress tensor the generator must impose at height `z` (m)
and time `t` (s): a 6-tuple ``(R_{11}, R_{22}, R_{33}, R_{12}, R_{13},
R_{23})`` in the face frame for a number `z`, a vector of them for a
vector. Every tensor returned has passed [`lund_coefficients`](@ref); one
that did not was projected by [`realizable_stress`](@ref) and the
projection booked in [`inflow_projection_log`](@ref)`(src)`.
"""
function inflow_stresses end

"""
    inflow_lengths(src::InflowStatistics, z, t) -> (L_s, L_z, T)

Integral length along the face `L_s` (m), vertical integral length `L_z`
(m) and integral time `T` (s) per component (normal, along, vertical):
3-tuples for a number `z`, 3 × `length(z)` matrices (the
[`DigitalFilterGenerator`](@ref) keyword format) for a vector.
"""
function inflow_lengths end

"""
    update_inflow_statistics!(src::InflowStatistics, Δt) -> src

Fold the parent's current state into a running statistic after a parent
step of `Δt` seconds. A source without memory (a parametrization) ignores
it.
"""
update_inflow_statistics!(src::InflowStatistics, Δt) = src

"""
    inflow_projection_log(src::InflowStatistics) -> Vector

The realizability projections booked by [`inflow_stresses`](@ref): one
`(t, z, change)` entry per projected tensor, `change` the Frobenius norm
of the correction (m² s⁻²). Empty when every tensor was realizable.
"""
inflow_projection_log(src::InflowStatistics) = src.log

const _ProjectionLog = Vector{NamedTuple{(:t, :z, :change), NTuple{3,Float64}}}

"""
    realizable_stress(R; rtol = 1e-12) -> (R6, change)

`R` (a 6-tuple ``(R_{11}, R_{22}, R_{33}, R_{12}, R_{13}, R_{23})`` or a
symmetric 3 × 3 matrix) when [`lund_coefficients`](@ref) accepts it
(`change = 0`); otherwise its projection onto the realizable
(positive-semidefinite) tensors: the eigenvalues are clipped at
``10^{-9}\\operatorname{tr}R`` (a vanishing component, not a negative
one), which is the nearest realizable tensor in the Frobenius norm up to
that floor, and `change` is the Frobenius norm of the correction. The
trace is kept when no eigenvalue is negative, so it changes only by what
the clipping removes. The realizability test itself is
[`lund_coefficients`](@ref) (Schumann 1977 through the Cholesky pivots);
nothing here re-implements it.
"""
function realizable_stress(R; rtol::Real = 1e-12)
    R6 = _stress6(R)
    try
        lund_coefficients(R6; rtol = rtol)
        return R6, 0.0
    catch err
        err isa ArgumentError || rethrow()
    end
    M = [R6[1] R6[4] R6[5]; R6[4] R6[2] R6[6]; R6[5] R6[6] R6[3]]
    F = eigen(Symmetric(M))
    floor_ = 1e-9 * max(sum(max.(F.values, 0.0)), floatmin(Float64))
    λ = max.(F.values, floor_)
    P = F.vectors * Diagonal(λ) * F.vectors'
    P = (P + P') / 2
    Rp = (P[1, 1], P[2, 2], P[3, 3], P[1, 2], P[1, 3], P[2, 3])
    lund_coefficients(Rp; rtol = rtol)          # the projection must pass the check
    return Rp, sqrt(sum(abs2, M .- P))
end

function _book!(log, t, z, R)
    Rp, ch = realizable_stress(R)
    ch > 0 && push!(log, (t = Float64(t), z = Float64(z), change = ch))
    return Rp
end

"""
    inflow_lund_coefficients(src::InflowStatistics, z::AbstractVector, t) -> a

The Lund et al. (1998) amplitude matrices, one per height, of the
(realizable) tensors of [`inflow_stresses`](@ref): pass them to
[`apply_lund!`](@ref).
"""
inflow_lund_coefficients(src::InflowStatistics, z::AbstractVector, t) =
    [lund_coefficients(R) for R in inflow_stresses(src, z, t)]

# linear interpolation in z of a per-level profile (constant beyond the ends)
function _interp(zl::AbstractVector, f::AbstractVector, z::Real)
    n = length(zl)
    z <= zl[1] && return f[1]
    z >= zl[n] && return f[n]
    k = searchsortedlast(zl, z)
    a = (z - zl[k]) / (zl[k+1] - zl[k])
    return (1 - a) * f[k] + a * f[k+1]
end

# grid (x, y, z) components → face frame (normal, along, vertical)
_face_perm(face::Symbol) = face in (:west, :east) ? (1, 2, 3) :
    face in (:south, :north) ? (2, 1, 3) :
    throw(ArgumentError("face must be :west, :east, :south or :north, got :$face"))

# stress 6-tuple in grid order → face order
function _to_face(R::NTuple{6,Float64}, face::Symbol)
    face in (:west, :east) && return R
    # (x, y) swapped: 11 ↔ 22, 13 ↔ 23
    return (R[2], R[1], R[3], R[4], R[6], R[5])
end

# integral of a correlation (ρ_0 = 1) to its first zero crossing, spacing Δ
function _integral_scale(ρ::AbstractVector, Δ::Real)
    s = 0.0
    for r in 1:length(ρ)-1
        a, b = ρ[r], ρ[r+1]
        if b > 0
            s += (a + b) / 2
        else
            s += a^2 / (2 * (a - b))          # linear to the zero crossing
            return Δ * s
        end
    end
    return Δ * s
end

# =========================================================================
# Parent-resolved statistics
# =========================================================================

"""
    ParentResolvedStatistics(parent; closure = nothing, face = :west, τ_avg = 600,
                             region = :plane, footprint = nothing, ring_width = 2,
                             mode = :scale_gap, refine = 2, c_gap = 4, c_len = c_gap/2,
                             U_min = 0.1, levels = 1:Nz, lag_z = 16, c_k = 0.1)

The inflow statistics of an Aeolus LES parent (any simulation with
`u, v, w, νt, bcs` on a `RegularGrid`, e.g. a `BoussinesqSimulation`),
implementing [`InflowStatistics`](@ref). Call
[`update_inflow_statistics!`](@ref)`(src, Δt)` after every parent step
(or at a cadence, with the elapsed `Δt`).

**Running mean.** Each sample ``x`` enters the exponential average with the
exact kernel weight of its interval, ``S ← ρS + w\\,x``, ``W ← ρW + w``,
``ρ = e^{-Δt/τ_{\\rm avg}}``, ``w = τ_{\\rm avg}(1 - ρ)``; the statistic is
``S/W``. The memory e-folds in exactly ``τ_{\\rm avg}`` for any sequence of
(adaptive) steps and the first sample is not biased towards zero.
`τ_avg = Inf` is the uniform time average (``w = Δt``).

**Resolved part.** At each level the velocities are interpolated to the
cell centres (as `test/tke_cbl_driver.jl` and the Transition diagnostics
do); ``R^{\\rm res}_{ij} = ⟨u_i'u_j'⟩`` with ``⟨·⟩`` the mean over the
`region` and ``'`` the deviation from it: `region = :plane`, the whole
horizontal plane (the plane means of `Operators`); `region = :ring`, the
parent columns within `ring_width` cells of the edge of the child
`footprint = (i_range, j_range)` (parent cell indices), on both sides
of it (periodic wrap).

**SGS part.** ``R^{\\rm SGS}_{ij} = \\tfrac23 e\\,δ_{ij} - 2ν_t\\bar S_{ij}``
with ``e`` from [`sgs_tke!`](@ref)`(e, closure, parent)` (the TKE closure's
``e``, or the closure's own estimate), ``ν_t`` the closure's viscosity
(`νh` for the horizontal pairs, `νv` for those with ``z``), and
``\\bar S_{ij}`` the deviatoric part of the cell-centred strain rate
(`Operators._strain_at`), so ``\\operatorname{tr}R^{\\rm SGS} = 2e``
exactly.

**Scale gap** (`mode = :scale_gap`, the default). The child receives the
parent's large eddies through the interpolated boundary values; the
generator must add only what is missing,

```math
R^{\\rm gen} = R^{\\rm SGS} + R^{\\rm res}(k_c < |k_h| ≤ r\\,k_c),
\\qquad k_c = \\frac{2π}{c_{\\rm gap}Δ_p},
```

the resolved stress of the horizontal wavenumbers between the parent's
effective resolution (wavelength ``c_{\\rm gap}Δ_p``; `c_gap = 4`, within
the 2–4 ``Δ_p`` of the issue, ``Δ_p = \\max(Δx, Δy)``) and the child's
(``c_{\\rm gap}Δ_p/r``, `refine = r`, the horizontal refinement), obtained
by a spectral band-pass of each level (FFT, the parent laterally
periodic). For ``r = 1`` the band is empty and ``R^{\\rm gen} = R^{\\rm SGS}``.
`mode = :full` gives ``R^{\\rm SGS} + R^{\\rm res}`` (a child whose
boundary values carry no parent turbulence).

**Lengths and time.** The horizontal autocovariances of every level come
from the same spectra (Wiener–Khinchin, whole plane, running-averaged);
the vertical ones from the plane covariance of a level with the
`lag_z` levels above it. Each integral length is the integral of the
correlation coefficient to its first zero crossing; ``L_s`` is along the
face, ``L_z`` vertical and ``T = L_n/\\max(|U_n|, U_{\\min})`` with ``L_n``
the length along the face normal and ``U_n`` the region-mean normal
velocity (Taylor; `U_min = 0.1` m s⁻¹ keeps a calm parent finite). In
scale-gap mode the lengths are those of the band-passed field (of the
full field where the band is empty) and are capped at ``c_{\\rm len}Δ_p``
(`c_len = c_gap/2`: half the cutoff wavelength).

`levels` restricts the statistics to the parent levels the child needs;
the work arrays hold ``2 × 3 × (\\text{lag\\_z} + 1)`` horizontal planes.

[`inflow_stress_parts`](@ref) returns the resolved, band, SGS parts
separately.

References: Lund, Wu & Squires (1998), *J. Comput. Phys.* 140, 233.
Kadasch, E. et al. (2021), *Geosci. Model Dev.* 14, 5435 (the PALM
mesoscale nesting with synthetic turbulence at the boundaries). Deardorff
(1980), *Boundary-Layer Meteorol.* 18, 495 (``e`` and ``ν_t``).
"""
mutable struct ParentResolvedStatistics{P, C, F, PL, IPL} <: InflowStatistics
    parent::P
    closure::C
    face::Symbol
    τ_avg::Float64
    region::Symbol
    columns::BitMatrix
    mode::Symbol
    refine::Int
    c_gap::Float64
    c_len::Float64
    U_min::Float64
    c_k::Float64
    levels::UnitRange{Int}
    lag_z::Int
    z::Vector{Float64}
    Δh::Float64
    band::BitMatrix                  # (Nx÷2+1) × Ny spectral band mask
    # running sums (grid components x, y, z; stresses 11 22 33 12 13 23)
    W::Float64
    t::Float64
    nsamples::Int
    Um::Matrix{Float64}              # 3 × nl
    Rres::Matrix{Float64}            # 6 × nl
    Rband::Matrix{Float64}
    Rsgs::Matrix{Float64}
    esgs::Vector{Float64}
    Cx::Array{Float64,3}             # 3 × nl × (Nx÷2+1), autocovariance along x
    Cy::Array{Float64,3}             # 3 × nl × (Ny÷2+1)
    Cz::Array{Float64,3}             # 3 × nl × (lag_z+1)
    Vz::Matrix{Float64}              # 3 × (nl + lag_z): plane variances for Cz's normalisation
    Bx::Array{Float64,3}             # band versions
    By::Array{Float64,3}
    Bz::Array{Float64,3}
    BVz::Matrix{Float64}
    # scratch
    e::F
    buf::Array{Float64,4}            # Nx × Ny × 3 × (lag_z+1) plane fluctuations
    bbuf::Array{Float64,4}           # band-passed
    plan::PL
    iplan::IPL
    spec::Matrix{ComplexF64}
    tmp::Matrix{Float64}
    cosx::Matrix{Float64}            # w_kx cos(2π kx r/Nx), nkx × (Nx÷2+1)
    cosy::Matrix{Float64}            # cos(2π ky r/Ny), Ny × (Ny÷2+1)
    log::_ProjectionLog
end

function ParentResolvedStatistics(parent; closure = nothing, face::Symbol = :west,
                                  τ_avg::Real = 600.0, region::Symbol = :plane,
                                  footprint = nothing, ring_width::Integer = 2,
                                  mode::Symbol = :scale_gap, refine = 2, c_gap::Real = 4.0,
                                  c_len::Real = c_gap / 2, U_min::Real = 0.1,
                                  levels = nothing, lag_z::Integer = 16, c_k::Real = 0.1)
    g = mesh(parent.u)
    g isa RegularGrid || throw(ArgumentError("ParentResolvedStatistics: the parent must be on a RegularGrid"))
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    _face_perm(face)
    τ_avg > 0 || throw(ArgumentError("ParentResolvedStatistics: τ_avg must be > 0"))
    mode in (:scale_gap, :full) ||
        throw(ArgumentError("ParentResolvedStatistics: mode must be :scale_gap or :full, got :$mode"))
    r = refine isa Integer ? Int(refine) : Int(min(refine[1], refine[2]))
    r >= 1 || throw(ArgumentError("ParentResolvedStatistics: refine must be ≥ 1"))
    c_gap >= 2 || throw(ArgumentError("ParentResolvedStatistics: c_gap must be ≥ 2 (the 2Δ wave is the shortest)"))
    lev = levels === nothing ? (1:Nz) : UnitRange{Int}(levels)
    (first(lev) >= 1 && last(lev) <= Nz && !isempty(lev)) ||
        throw(ArgumentError("ParentResolvedStatistics: levels must lie in 1:$Nz"))
    cols = falses(Nx, Ny)
    if region === :plane
        cols .= true
    elseif region === :ring
        footprint === nothing &&
            throw(ArgumentError("ParentResolvedStatistics: region = :ring needs footprint = (i_range, j_range)"))
        is, js = footprint
        w = Int(ring_width)
        for j in 1:Ny, i in 1:Nx
            # periodic distance of (i, j) to the footprint box, signed inside
            outer = _in_box(i, first(is) - w, last(is) + w, Nx) && _in_box(j, first(js) - w, last(js) + w, Ny)
            inner = _in_box(i, first(is) + w, last(is) - w, Nx) && _in_box(j, first(js) + w, last(js) - w, Ny)
            cols[i, j] = outer && !inner
        end
        any(cols) || throw(ArgumentError("ParentResolvedStatistics: the ring is empty"))
    else
        throw(ArgumentError("ParentResolvedStatistics: region must be :plane or :ring, got :$region"))
    end
    nl = length(lev); L = Int(lag_z)
    L >= 1 || throw(ArgumentError("ParentResolvedStatistics: lag_z must be ≥ 1"))
    Δh = max(g.dx, g.dy)
    kc = 2π / (c_gap * Δh)
    nkx = Nx ÷ 2 + 1
    band = falses(nkx, Ny)
    for j in 1:Ny, i in 1:nkx
        ky = j - 1 <= Ny ÷ 2 ? j - 1 : j - 1 - Ny
        k = 2π * sqrt(((i - 1) / g.Lx)^2 + (ky / g.Ly)^2)
        band[i, j] = kc < k <= r * kc * (1 + 1e-12)
    end
    r == 1 && (band .= false)
    z = [g.z0 + (k - 0.5) * g.dz for k in lev]
    tmp = zeros(Nx, Ny)
    plan = plan_rfft(tmp)
    iplan = plan_irfft(zeros(ComplexF64, Nx ÷ 2 + 1, Ny), Nx)
    return ParentResolvedStatistics(parent, closure, face, Float64(τ_avg), region, cols, mode, r,
                                    Float64(c_gap), Float64(c_len), Float64(U_min), Float64(c_k),
                                    lev, L, z, Δh, band, 0.0, 0.0, 0,
                                    zeros(3, nl), zeros(6, nl), zeros(6, nl), zeros(6, nl), zeros(nl),
                                    zeros(3, nl, nkx), zeros(3, nl, Ny ÷ 2 + 1), zeros(3, nl, L + 1),
                                    zeros(3, nl + L),
                                    zeros(3, nl, nkx), zeros(3, nl, Ny ÷ 2 + 1), zeros(3, nl, L + 1),
                                    zeros(3, nl + L),
                                    Field(CellCenter(), g), zeros(Nx, Ny, 3, L + 1), zeros(Nx, Ny, 3, L + 1),
                                    plan, iplan, zeros(ComplexF64, nkx, Ny), tmp,
                                    [_wx(i, Nx) * cos(2π * (i - 1) * r / Nx) for i in 1:nkx, r in 0:nkx-1],
                                    [cos(2π * (j - 1) * r / Ny) for j in 1:Ny, r in 0:Ny÷2],
                                    _ProjectionLog())
end

# weight of the half-spectrum column i (kx = i - 1): 1 for kx = 0 and the Nyquist column
_wx(i, Nx) = (i == 1 || (iseven(Nx) && i == Nx ÷ 2 + 1)) ? 1.0 : 2.0

# i ∈ [a, b] on a periodic axis of length N (a ≤ b, b - a < N)
function _in_box(i, a, b, N)
    b < a && return false
    b - a + 1 >= N && return true
    return mod(i - a, N) <= b - a
end

Base.show(io::IO, s::ParentResolvedStatistics) =
    print(io, "ParentResolvedStatistics(:$(s.mode), face = :$(s.face), τ_avg = $(s.τ_avg) s, ",
          "region = :$(s.region), refine = $(s.refine), c_gap = $(s.c_gap), samples = $(s.nsamples))")

function update_inflow_statistics!(s::ParentResolvedStatistics, Δt::Real)
    Δt > 0 || throw(ArgumentError("update_inflow_statistics!: Δt must be > 0"))
    p = s.parent
    g = mesh(p.u)
    Nx, Ny, Nz, Hx, Hy, Hz = g.Nx, g.Ny, g.Nz, g.Hx, g.Hy, g.Hz
    fill_halos!(p.u, p.bcs); fill_halos!(p.v, p.bcs); fill_halos!(p.w, p.bcs)
    ρ, w = isinf(s.τ_avg) ? (1.0, Float64(Δt)) : (exp(-Δt / s.τ_avg), s.τ_avg * -expm1(-Δt / s.τ_avg))
    s.W = ρ * s.W + w
    for A in (s.Um, s.Rres, s.Rband, s.Rsgs, s.esgs, s.Cx, s.Cy, s.Cz, s.Vz, s.Bx, s.By, s.Bz, s.BVz)
        A .*= ρ
    end
    s.t = hasproperty(p, :t) ? Float64(p.t) : s.t + Δt
    s.nsamples += 1

    # SGS energy and viscosities of the closure
    sgs_tke!(s.e, s.closure, p; c_k = s.c_k)
    νh, νv = if s.closure === nothing
        p.νt, p.νt
    else
        f = sgs_fields(s.closure, p); f.νh, f.νv
    end
    ud, vd, wd = p.u.data, p.v.data, p.w.data
    idx, idy, idz = 1 / g.dx, 1 / g.dy, 1 / g.dz
    cols = s.columns
    ncol = count(cols)
    lev = s.levels; k1 = first(lev); L = s.lag_z
    kend = min(last(lev) + L, Nz)
    M = L + 1
    N2 = Float64(Nx * Ny)^2
    nkx = Nx ÷ 2 + 1
    wx = [_wx(i, Nx) for i in 1:nkx]
    scale_gap = s.mode === :scale_gap
    pm = zeros(3)
    for kk in k1:kend
        slot = mod(kk - k1, M) + 1
        K = Hz + kk
        # cell-centred velocities minus their plane means
        @inbounds for j in 1:Ny, i in 1:Nx
            I, J = Hx + i, Hy + j
            s.buf[i, j, 1, slot] = (ud[I, J, K] + ud[I+1, J, K]) / 2
            s.buf[i, j, 2, slot] = (vd[I, J, K] + vd[I, J+1, K]) / 2
            s.buf[i, j, 3, slot] = (wd[I, J, K] + wd[I, J, K+1]) / 2
        end
        for c in 1:3
            pm[c] = sum(@view s.buf[:, :, c, slot]) / (Nx * Ny)
            @views s.buf[:, :, c, slot] .-= pm[c]
        end
        l = kk - k1 + 1                     # index in the Vz/BVz columns
        inlev = kk in lev
        for c in 1:3
            f = @view s.buf[:, :, c, slot]
            copyto!(s.tmp, f)
            mul!(s.spec, s.plan, s.tmp)
            s.Vz[c, l] += w * sum(abs2, f) / (Nx * Ny)
            if inlev
                _acc_autocov!(s.Cx, s.Cy, c, l, s.spec, nothing, wx, w / N2, s.cosx, s.cosy)
            end
            # band-passed field
            if scale_gap && any(s.band)
                inlev && _acc_autocov!(s.Bx, s.By, c, l, s.spec, s.band, wx, w / N2, s.cosx, s.cosy)
                @inbounds for j in 1:Ny, i in 1:nkx
                    s.band[i, j] || (s.spec[i, j] = 0)
                end
                mul!(s.tmp, s.iplan, s.spec)
                @views s.bbuf[:, :, c, slot] .= s.tmp
                s.BVz[c, l] += w * sum(abs2, @view s.bbuf[:, :, c, slot]) / (Nx * Ny)
            else
                @views s.bbuf[:, :, c, slot] .= 0
            end
        end
        if inlev
            # region means, resolved and band covariances, SGS stress
            reg = zeros(3); breg = zeros(3)
            Rr = zeros(6); Rb = zeros(6); Rs = zeros(6); es = 0.0
            @inbounds for j in 1:Ny, i in 1:Nx
                cols[i, j] || continue
                a1, a2, a3 = s.buf[i, j, 1, slot], s.buf[i, j, 2, slot], s.buf[i, j, 3, slot]
                b1, b2, b3 = s.bbuf[i, j, 1, slot], s.bbuf[i, j, 2, slot], s.bbuf[i, j, 3, slot]
                reg[1] += a1; reg[2] += a2; reg[3] += a3
                breg[1] += b1; breg[2] += b2; breg[3] += b3
                Rr[1] += a1 * a1; Rr[2] += a2 * a2; Rr[3] += a3 * a3
                Rr[4] += a1 * a2; Rr[5] += a1 * a3; Rr[6] += a2 * a3
                Rb[1] += b1 * b1; Rb[2] += b2 * b2; Rb[3] += b3 * b3
                Rb[4] += b1 * b2; Rb[5] += b1 * b3; Rb[6] += b2 * b3
                I, J = Hx + i, Hy + j
                S11, S22, S33, S12, S13, S23, _ = _strain_at(ud, vd, wd, I, J, K, idx, idy, idz)
                e = s.e.data[I, J, K]
                nh = νh.data[I, J, K]; nv = νv.data[I, J, K]
                T11 = -2nh * S11; T22 = -2nh * S22; T33 = -2nv * S33
                tr3 = (T11 + T22 + T33) / 3
                Rs[1] += 2e / 3 + T11 - tr3; Rs[2] += 2e / 3 + T22 - tr3; Rs[3] += 2e / 3 + T33 - tr3
                Rs[4] += -2nh * S12; Rs[5] += -2nv * S13; Rs[6] += -2nv * S23
                es += e
            end
            reg ./= ncol; breg ./= ncol
            pairs = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
            for (q, (a, b)) in enumerate(pairs)
                s.Rres[q, l] += w * (Rr[q] / ncol - reg[a] * reg[b])
                s.Rband[q, l] += w * (Rb[q] / ncol - breg[a] * breg[b])
                s.Rsgs[q, l] += w * Rs[q] / ncol
            end
            s.esgs[l] += w * es / ncol
            for c in 1:3
                s.Um[c, l] += w * (pm[c] + reg[c])
            end
        end
        # vertical lags: this level against the ≤ L levels below it
        for m in 0:L
            k0 = kk - m
            (k0 >= k1 && k0 in lev) || continue
            s0 = mod(k0 - k1, M) + 1
            l0 = k0 - k1 + 1
            for c in 1:3
                acc = dot(view(s.buf, :, :, c, s0), view(s.buf, :, :, c, slot))
                s.Cz[c, l0, m+1] += w * acc / (Nx * Ny)
                if scale_gap
                    bacc = dot(view(s.bbuf, :, :, c, s0), view(s.bbuf, :, :, c, slot))
                    s.Bz[c, l0, m+1] += w * bacc / (Nx * Ny)
                end
            end
        end
    end
    return s
end

# Accumulate the x and y autocovariances of one level from its half
# spectrum F (rfft along x): C_x(r) = Σ_kx w_kx p_x(kx) cos(2πkx r/Nx)/N²,
# p_x = Σ_ky |F|², and C_y(r) = Σ_ky p_y(ky) cos(2πky r/Ny)/N²,
# p_y = Σ_kx w_kx |F|² (Wiener–Khinchin with the Hermitian half).
function _acc_autocov!(Cx, Cy, c, l, F, band, wx, fac, cosx, cosy)
    nkx, Ny = size(F)
    px = zeros(nkx); py = zeros(Ny)
    @inbounds for j in 1:Ny, i in 1:nkx
        (band === nothing || band[i, j]) || continue
        a = abs2(F[i, j])
        px[i] += a
        py[j] += wx[i] * a
    end
    @views Cx[c, l, :] .+= fac .* (cosx' * px)
    @views Cy[c, l, :] .+= fac .* (cosy' * py)
    return nothing
end

_check_samples(s::ParentResolvedStatistics) = s.nsamples > 0 ||
    throw(ArgumentError("ParentResolvedStatistics: no samples yet; call update_inflow_statistics! first"))

"""
    inflow_stress_parts(src::ParentResolvedStatistics) -> NamedTuple

The running-mean profiles at the parent levels `z` (m), in the face frame:
`resolved` (``R^{\\rm res}``, 6 × nl), `band` (the scale-gap part
``R^{\\rm res}(k_c < |k_h| ≤ rk_c)``), `sgs` (``R^{\\rm SGS}``), `generated`
(what [`inflow_stresses`](@ref) returns before the realizability check:
`sgs + band`, or `sgs + resolved` for `mode = :full`: the part the
child's boundary values miss), `e_sgs` (nl), `U` (3 × nl region-mean velocity, face
frame), the lengths of [`inflow_lengths`](@ref) at the levels (`L_s`,
`L_z`, `L_n`, `T`; 3 × nl) and those of the full resolved field, uncapped
(`L_s_full`, `L_z_full`, `L_n_full`).
"""
function inflow_stress_parts(s::ParentResolvedStatistics)
    _check_samples(s)
    perm = _face_perm(s.face)
    tf(A) = reduce(hcat, [collect(_to_face(Tuple(A[:, l] ./ s.W), s.face)) for l in axes(A, 2)])
    res = tf(s.Rres); bnd = tf(s.Rband); sgs = tf(s.Rsgs)
    gen = s.mode === :scale_gap ? sgs .+ bnd : sgs .+ res
    U = s.Um[collect(perm), :] ./ s.W
    Ls = _lengths(s)
    Lf = _lengths(s; band = false)
    return (z = copy(s.z), resolved = res, band = bnd, sgs = sgs, generated = gen,
            e_sgs = s.esgs ./ s.W, U = U, L_s = Ls.L_s, L_z = Ls.L_z, L_n = Ls.L_n, T = Ls.T,
            L_s_full = Lf.L_s, L_z_full = Lf.L_z, L_n_full = Lf.L_n)
end

# per-level lengths in the face frame, 3 × nl
function _lengths(s::ParentResolvedStatistics; band::Bool = s.mode === :scale_gap)
    g = mesh(s.parent.u)
    perm = _face_perm(s.face)
    nl = length(s.levels)
    Ls = zeros(3, nl); Lz = zeros(3, nl); Ln = zeros(3, nl); T = zeros(3, nl)
    along_x = s.face in (:south, :north)        # the along-face direction
    useband = band && any(s.band)
    cap = band ? s.c_len * s.Δh : Inf
    Nz = g.Nz
    for l in 1:nl, (fc, c) in enumerate(perm)
        band_ok = useband && s.Bx[c, l, 1] > 0
        Cx = band_ok ? s.Bx : s.Cx; Cy = band_ok ? s.By : s.Cy
        Cz = band_ok ? s.Bz : s.Cz; V = band_ok ? s.BVz : s.Vz
        lx = _corr_length(@view(Cx[c, l, :]), g.dx)
        ly = _corr_length(@view(Cy[c, l, :]), g.dy)
        # vertical: correlation coefficient with the levels above
        kk = s.levels[l]
        nm = min(s.lag_z, Nz - kk)
        ρ = zeros(nm + 1)
        for m in 0:nm
            v0 = V[c, l]; vm = V[c, l+m]
            ρ[m+1] = v0 > 0 && vm > 0 ? Cz[c, l, m+1] / sqrt(v0 * vm) : (m == 0 ? 1.0 : 0.0)
        end
        lz = nm == 0 ? g.dz / 2 : _integral_scale(ρ, g.dz)
        ls, ln = along_x ? (lx, ly) : (ly, lx)
        Ls[fc, l] = min(ls, cap); Lz[fc, l] = min(lz, cap); Ln[fc, l] = min(ln, cap)
    end
    for l in 1:nl
        Un = abs(s.Um[perm[1], l] / s.W)
        for fc in 1:3
            T[fc, l] = Ln[fc, l] / max(Un, s.U_min)
        end
    end
    return (L_s = Ls, L_z = Lz, L_n = Ln, T = T)
end

function _corr_length(C::AbstractVector, Δ)
    C[1] > 0 || return Δ / 2
    return _integral_scale(C ./ C[1], Δ)
end

function inflow_stresses(s::ParentResolvedStatistics, z::Real, t::Real)
    P = _stress_profile(s)
    R = ntuple(q -> _interp(s.z, @view(P[q, :]), z), 6)
    return _book!(s.log, t, z, R)
end
function inflow_stresses(s::ParentResolvedStatistics, z::AbstractVector, t::Real)
    P = _stress_profile(s)
    return [_book!(s.log, t, zz, ntuple(q -> _interp(s.z, @view(P[q, :]), zz), 6)) for zz in z]
end
function _stress_profile(s::ParentResolvedStatistics)
    _check_samples(s)
    A = s.mode === :scale_gap ? s.Rsgs .+ s.Rband : s.Rsgs .+ s.Rres
    return reduce(hcat, [collect(_to_face(Tuple(A[:, l] ./ s.W), s.face)) for l in axes(A, 2)])
end

function inflow_lengths(s::ParentResolvedStatistics, z::Real, t::Real)
    _check_samples(s)
    L = _lengths(s)
    f(M) = ntuple(c -> _interp(s.z, @view(M[c, :]), z), 3)
    return (L_s = f(L.L_s), L_z = f(L.L_z), T = f(L.T))
end
function inflow_lengths(s::ParentResolvedStatistics, z::AbstractVector, t::Real)
    _check_samples(s)
    L = _lengths(s)
    f(M) = [_interp(s.z, @view(M[c, :]), zz) for c in 1:3, zz in z]
    return (L_s = f(L.L_s), L_z = f(L.L_z), T = f(L.T))
end

# =========================================================================
# Parametrized statistics (a parent without resolved turbulence)
# =========================================================================

"""
    ParametrizedStatistics(u_star, w_star, z_i, L_Ob; length_scale, U_n = 0,
                           wind_angle = 0, U_min = 0.1)
    ParametrizedStatistics(; u_star, z_i, L_Ob = Inf, w_star = nothing, kwargs...)
    ParametrizedStatistics(; u_star, z_i, θv_star, θv_ref, kwargs...)  # L_Ob from the fluxes

Similarity profiles for a parent that resolves no turbulence (a
mesoscale model, #58), implementing [`InflowStatistics`](@ref). The
scales may be numbers or functions of time `t`. `w_star = nothing`
derives the convective velocity from the Obukhov length,
``w_*^3 = -z_iu_*^3/(κL)`` for ``L < 0`` and ``0`` otherwise (κ =
`SurfaceExchange.von_karman()`); the flux form (``θ_{v*}`` and the
reference ``θ_v``, both K) takes ``L`` from
`SurfaceExchange.obukhov_length(u_star, θv_star, θv_ref)`. No Obukhov
length is computed here.

With ``x = \\min(z/z_i, 1)``, in the frame of the mean wind (along,
cross, vertical), the shear-generated and the convective variances are
added (the blend of Rotach, Gryning & Tassone 1996, as PALM's
synthetic-turbulence generator applies it):

```math
\\begin{aligned}
σ_u^2 &= (2.39u_*)^2(1 - 0.8x) + 0.35w_*^2, &
σ_v^2 &= (1.92u_*)^2(1 - 0.8x) + 0.35w_*^2,\\\\
σ_w^2 &= (1.25u_*)^2\\left(1 - \\frac{x}{1.7}\\right) + 1.8\\,x^{2/3}(1 - 0.8x)^2w_*^2, &
\\overline{u'w'} &= -u_*^2(1 - x),
\\end{aligned}
```

``\\overline{u'v'} = \\overline{v'w'} = 0``. The surface values of the shear
part are the neutral surface-layer ratios of Panofsky & Dutton (1984,
§7.4: ``σ_{u,v,w}/u_* = 2.39, 1.92, 1.25``); their decay with height is
Rotach et al.'s (``u_*^2(5 - 4x)`` and ``u_*^2(1.7 - x)``, normalised to
the surface value); the convective horizontal variance ``0.35w_*^2`` is
Rotach et al.'s; the convective ``σ_w^2`` is the mixed-layer fit of
Lenschow, Wyngaard & Pennell (1980, *J. Atmos. Sci.* 37, 1313),
maximal ``σ_w/w_* = 0.683`` at ``z/z_i = 0.3125``; ``\\overline{u'w'}``
is linear to zero at ``z_i`` (Rotach et al.; PALM). Above ``z_i`` every
component decays by 10 per `length_scale` (PALM's blend). The tensor is
rotated by `wind_angle` (rad, the mean wind's direction from the face
normal towards the along-face axis) into the face frame. It is realizable
for every stability (``σ_u^2σ_w^2 ≥ 8.9u_*^4(1-x)^2``).

Lengths, as in PALM (Kadasch et al. 2021): one `length_scale` (m) for
every component and direction — a numerical choice (perturbations
shorter than ~8Δ of the receiving grid are dissipated), not a
similarity law — with ``L_z ≤ z_i``, and ``T = ℓ/\\max(|U_n|, U_{\\min})``,
`U_n` the normal velocity (a number or a function of `z`).

Stable stratification (``L > 0``) has ``w_* = 0`` and keeps the neutral
shear profile scaled by ``u_*`` and ``z_i``: the blend has no stable
correction (Rotach et al. is a daytime model).

References: Panofsky, H. A. & Dutton, J. A. (1984). *Atmospheric
Turbulence*. Wiley. Lenschow, D. H., Wyngaard, J. C. & Pennell, W. T.
(1980). Mean-field and second-moment budgets in a baroclinic, convective
boundary layer. *J. Atmos. Sci.*, 37, 1313-1326. Rotach, M. W.,
Gryning, S.-E. & Tassone, C. (1996). A two-dimensional Lagrangian
stochastic dispersion model for daytime conditions. *Q. J. R. Meteorol.
Soc.*, 122, 367-389. Kadasch, E. et al. (2021). *Geosci. Model Dev.*,
14, 5435-5465.
"""
struct ParametrizedStatistics{A, B, C, D, U} <: InflowStatistics
    u_star::A
    w_star::B          # nothing: from L_Ob
    z_i::C
    L_Ob::D
    length_scale::Float64
    U_n::U
    wind_angle::Float64
    U_min::Float64
    κ::Float64
    log::_ProjectionLog
end

function ParametrizedStatistics(u_star, w_star, z_i, L_Ob; length_scale::Real, U_n = 0.0,
                                wind_angle::Real = 0.0, U_min::Real = 0.1,
                                surface::SurfaceConstants = SurfaceConstants())
    length_scale > 0 || throw(ArgumentError("ParametrizedStatistics: length_scale must be > 0"))
    return ParametrizedStatistics(u_star, w_star, z_i, L_Ob, Float64(length_scale), U_n,
                                  Float64(wind_angle), Float64(U_min),
                                  Float64(von_karman(surface)), _ProjectionLog())
end
function ParametrizedStatistics(; u_star, z_i, L_Ob = nothing, w_star = nothing,
                                θv_star = nothing, θv_ref = nothing,
                                surface::SurfaceConstants = SurfaceConstants(), kwargs...)
    L = if L_Ob !== nothing
        θv_star === nothing || throw(ArgumentError("ParametrizedStatistics: give L_Ob or θv_star, not both"))
        L_Ob
    elseif θv_star !== nothing
        θv_ref === nothing && throw(ArgumentError("ParametrizedStatistics: θv_star needs θv_ref"))
        θv_star == 0 ? Inf : obukhov_length(u_star, θv_star, θv_ref; c = surface)
    else
        Inf
    end
    return ParametrizedStatistics(u_star, w_star, z_i, L; surface = surface, kwargs...)
end

_at(x, t) = x isa Function ? x(t) : x

"""
    parametrized_scales(src::ParametrizedStatistics, t) -> (u_star, w_star, z_i, L_Ob)

The scales at time `t`, with ``w_*`` derived from ``L`` when not given.
"""
function parametrized_scales(s::ParametrizedStatistics, t::Real)
    us = Float64(_at(s.u_star, t)); zi = Float64(_at(s.z_i, t)); L = Float64(_at(s.L_Ob, t))
    zi > 0 || throw(ArgumentError("ParametrizedStatistics: z_i must be > 0"))
    ws = s.w_star === nothing ? (L < 0 ? cbrt(-zi * us^3 / (s.κ * L)) : 0.0) :
         Float64(_at(s.w_star, t))
    return (u_star = us, w_star = ws, z_i = zi, L_Ob = L)
end

# the wind-frame tensor (11 22 33 12 13 23) at z
function _parametrized_tensor(s::ParametrizedStatistics, z::Real, t::Real)
    p = parametrized_scales(s, t)
    us, ws, zi = p.u_star, p.w_star, p.z_i
    x = clamp(z / zi, 0.0, 1.0)
    b = z > zi ? 10.0^(-(z - zi) / s.length_scale) : 1.0
    su2 = ((2.39us)^2 * (1 - 0.8x) + 0.35ws^2) * b
    sv2 = ((1.92us)^2 * (1 - 0.8x) + 0.35ws^2) * b
    sw2 = ((1.25us)^2 * (1 - x / 1.7) + 1.8 * x^(2 / 3) * (1 - 0.8x)^2 * ws^2) * b
    uw = -us^2 * (1 - x) * b
    return (su2, sv2, sw2, 0.0, uw, 0.0)
end

# rotate the wind frame by φ about z into the face frame: R' = Q R Qᵀ
function _rotate(R::NTuple{6,Float64}, φ::Float64)
    φ == 0 && return R
    c, s = cos(φ), sin(φ)
    M = [R[1] R[4] R[5]; R[4] R[2] R[6]; R[5] R[6] R[3]]
    Q = [c -s 0.0; s c 0.0; 0.0 0.0 1.0]
    P = Q * M * Q'
    return (P[1, 1], P[2, 2], P[3, 3], (P[1, 2] + P[2, 1]) / 2, (P[1, 3] + P[3, 1]) / 2,
            (P[2, 3] + P[3, 2]) / 2)
end

inflow_stresses(s::ParametrizedStatistics, z::Real, t::Real) =
    _book!(s.log, t, z, _rotate(_parametrized_tensor(s, z, t), s.wind_angle))
inflow_stresses(s::ParametrizedStatistics, z::AbstractVector, t::Real) =
    [inflow_stresses(s, zz, t) for zz in z]

function _param_lengths(s::ParametrizedStatistics, z::Real, t::Real)
    zi = parametrized_scales(s, t).z_i
    ℓ = s.length_scale
    Un = Float64(s.U_n isa Function ? s.U_n(z) : s.U_n)
    T = ℓ / max(abs(Un), s.U_min)
    return ℓ, min(ℓ, zi), T
end
function inflow_lengths(s::ParametrizedStatistics, z::Real, t::Real)
    ls, lz, T = _param_lengths(s, z, t)
    return (L_s = (ls, ls, ls), L_z = (lz, lz, lz), T = (T, T, T))
end
function inflow_lengths(s::ParametrizedStatistics, z::AbstractVector, t::Real)
    v = [_param_lengths(s, zz, t) for zz in z]
    f(q) = [v[k][q] for c in 1:3, k in eachindex(z)]
    return (L_s = f(1), L_z = f(2), T = f(3))
end
