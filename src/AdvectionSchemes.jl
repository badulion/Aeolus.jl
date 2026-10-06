module AdvectionSchemes

# =========================================================================
# Era II — Phase 26 (WENO-Z / TENO advection) and Phase 28 (energy- and
# enstrophy-conserving discrete operators) for the finite-volume backend.
# =========================================================================
#
# Two families of advection operators live here, aimed at opposite ends of
# the accuracy/robustness spectrum:
#
#   Phase 26 — shock/front-capturing high-order upwind reconstructions
#     * WENO-Z 5th/7th order   (Borges, Carmona, Costa & Don 2008;
#                               Castro, Costa & Don 2011)
#     * TENO 5th/8th order     (Fu, Hu & Adams 2016)
#     * a positivity-preserving flux limiter for moisture / chemistry
#     * WENO-Z momentum advection on the Arakawa-C grid
#
#   Phase 28 — discretely conservative (skew-symmetric) operators
#     * Morinishi (1998)-style fully conservative momentum advection,
#       exactly kinetic-energy conserving
#     * variance-conserving scalar advection
#     * kinetic-energy / scalar-variance budgets and enstrophy diagnostics
#
# Conventions follow `src/FiniteVolume.jl` exactly: `u @ FaceX` is the west
# face of a cell, `v @ FaceY` the south face, `w @ FaceZ` the bottom face,
# scalars at `CellCenter`. All kernels assume the halos of their inputs are
# filled, and *add* their contribution to the tendency argument.
#
# Reconstruction index convention
# -------------------------------
# All reconstructions produce the value of a field at a cell interface.
# Following the WENO literature the interface is written `i+1/2`, and the
# stencil offsets below are relative to cell `i` — the cell *immediately
# upwind-left* of the interface. In the code's face numbering, face index
# `f` is the west/south/bottom face of cell `f`, so the interface `f` has
# left cell `f-1`.
#
#   left-biased  (use when the face velocity is ≥ 0)
#   right-biased (use when the face velocity is  < 0) = mirror image
#
# NOTE ON A JULIA PITFALL: `3f0` is a *Float32 literal* (3.0f0), not
# `3 * f0`.  Every coefficient below is written with an explicit `*`.
# (`src/FiniteVolume.jl`'s `_weno5_left` smoothness indicators are hit by
# exactly this parsing trap — see the Phase 26 notes in test/era2_advection.jl.)

using ..Grids
using ..Grids: AbstractMesh, RegularGrid, interior, CellCenter, FaceX, FaceY, FaceZ, Location
using ..Fields
using ..Fields: Field, interior_view, mesh
using ..BoundaryConditions
using ..BoundaryConditions: FieldBoundaryConditions, fill_halos!, AbstractOpenBoundary, Periodic
using ..Operators
using ..Operators: advect_centered!, momentum_advect!, laplacian, ∂x, ∂y, ∂z,
                   add_buoyancy!, add_coriolis!, compute_smagorinsky_νt!,
                   add_eddy_viscosity_u!, add_eddy_viscosity_v!, add_eddy_viscosity_w!,
                   add_eddy_diffusion_scalar!, add_surface_flux!, advect_weno5!
using ..Operators: compute_sgs!, add_sgs_momentum!, sgs_closure
using ..Simulations
using ..Simulations: BoussinesqSimulation

export # --- Phase 26: reconstructions -------------------------------------
       weno_z5_left, weno_z5_right, weno_z7_left, weno_z7_right,
       teno5_left, teno5_right, teno8_left, teno8_right,
       weno_js5_left, weno_js5_right,
       # --- Phase 26: advection operators ---------------------------------
       advect_weno_z!, advect_teno!, momentum_advect_weno_z!,
       apply_positivity_limiter!, advect_positive!, FluxLimiterWorkspace,
       # --- scheme selection ----------------------------------------------
       AdvectionScheme, advection_scheme, required_halo,
       # --- Phase 28: conservative operators ------------------------------
       momentum_advect_skew!, advect_variance_conserving!,
       # --- Phase 28: conservation diagnostics ----------------------------
       kinetic_energy_budget, scalar_variance_budget, enstrophy,
       # --- N-open-c (#260): advective flux through the open faces --------
       boundary_scalar_flux,
       # --- N-nest-e (#263): the flux through one face ---------------------
       advective_face_flux

# =========================================================================
# 1. Smoothness indicators
# =========================================================================
# The Jiang & Shu (1996) smoothness indicator of a candidate stencil is
#
#     β = Σ_{l=1}^{n-1} ∫_{I_i} h^{2l-1} (dˡp/dxˡ)² dx ,
#
# where `p` is the degree-(n-1) polynomial whose cell averages reproduce
# the `n` stencil values and `I_i` is the cell to the left of the
# interface. Writing `p(ξ) = Σ_m a_m ξ^m` with `ξ = (x - x_i)/h` makes the
# integrals independent of `h`:
#
#     n = 3:  β = a₁² + 13/3 a₂²
#     n = 4:  β = a₁² + ½a₁a₃ + 13/3 a₂² + 3129/80 a₃²
#     n = 5:  β = a₁² + ½a₁a₃ + 13/3 a₂² + 21/5 a₂a₄ + 3129/80 a₃²
#                  + 87617/140 a₄²
#
# (The n = 3 form reproduces the textbook
# `13/12 (v₁-2v₂+v₃)² + 1/4 (v₁-4v₂+3v₃)²` identically.)
# Using the `a` basis rather than the expanded quadratic form in the raw
# stencil values is deliberate: the `a_m` are difference operators, so the
# indicators keep full relative accuracy as h → 0. That accuracy is what
# makes the τ-based Z/TENO weights reach their design order.

@inline _beta3(a1::T, a2::T) where {T} = a1*a1 + T(13/3)*(a2*a2)

@inline function _beta4(a1::T, a2::T, a3::T) where {T}
    return a1*a1 + T(1/2)*(a1*a3) + T(13/3)*(a2*a2) + T(3129/80)*(a3*a3)
end

@inline function _beta5(a1::T, a2::T, a3::T, a4::T) where {T}
    return a1*a1 + T(1/2)*(a1*a3) + T(13/3)*(a2*a2) + T(21/5)*(a2*a4) +
           T(3129/80)*(a3*a3) + T(87617/140)*(a4*a4)
end

# --- 3-cell stencils, by offset of their first cell from cell i ----------
@inline _b3_m2(v1, v2, v3) = _beta3((v1 - 4*v2 + 3*v3)/2, (v1 - 2*v2 + v3)/2)   # {i-2,i-1,i}
@inline _b3_m1(v1, v2, v3) = _beta3((v3 - v1)/2,           (v1 - 2*v2 + v3)/2)   # {i-1,i,i+1}
@inline _b3_p0(v1, v2, v3) = _beta3((-3*v1 + 4*v2 - v3)/2, (v1 - 2*v2 + v3)/2)   # {i,i+1,i+2}

# --- 4-cell stencils -----------------------------------------------------
@inline _b4_m3(v1, v2, v3, v4) =                                                  # {i-3,…,i}
    _beta4((-7*v1 + 33*v2 - 69*v3 + 43*v4)/24,
           (-v1 + 4*v2 - 5*v3 + 2*v4)/2,
           (-v1 + 3*v2 - 3*v3 + v4)/6)

@inline _b4_p0(v1, v2, v3, v4) =                                                  # {i,…,i+3}
    _beta4((-43*v1 + 69*v2 - 33*v3 + 7*v4)/24,
           (2*v1 - 5*v2 + 4*v3 - v4)/2,
           (-v1 + 3*v2 - 3*v3 + v4)/6)

@inline _b4_m2(v1, v2, v3, v4) =                                                  # {i-2,…,i+1}
    _beta4((5*v1 - 27*v2 + 15*v3 + 7*v4)/24,
           (v2 - 2*v3 + v4)/2,
           (-v1 + 3*v2 - 3*v3 + v4)/6)

@inline _b4_m1(v1, v2, v3, v4) =                                                  # {i-1,…,i+2}
    _beta4((-7*v1 - 15*v2 + 27*v3 - 5*v4)/24,
           (v1 - 2*v2 + v3)/2,
           (-v1 + 3*v2 - 3*v3 + v4)/6)

# --- 5-cell stencil {i-4,…,i} -------------------------------------------
@inline _b5_m4(v1, v2, v3, v4, v5) =
    _beta5((9*v1 - 50*v2 + 120*v3 - 174*v4 + 95*v5)/48,
           (7*v1 - 36*v2 + 74*v3 - 68*v4 + 23*v5)/16,
           (3*v1 - 14*v2 + 24*v3 - 18*v4 + 5*v5)/12,
           (v1 - 4*v2 + 6*v3 - 4*v4 + v5)/24)

# Machine-zero regulariser. Unlike WENO-JS's ε = 1e-6 (which acts as an
# unintended smoothness cut-off and is what destroys the formal order of
# the classic scheme at small h) this is only there to keep 0/0 finite.
@inline _zeps(::Type{T}) where {T} = eps(T)^2
@inline _zeps(x) = _zeps(typeof(x))

# =========================================================================
# 2. WENO-Z  (Borges, Carmona, Costa & Don, JCP 227 (2008) 3191)
# =========================================================================
# Z-type nonlinear weights replace WENO-JS's `α_k = d_k/(β_k+ε)^2` by
#
#     α_k = d_k ( 1 + (τ / (β_k + ε))^p ),      ω_k = α_k / Σ α
#
# with τ a *global* smoothness measure built from the candidate β's.
# Because τ vanishes to high order for smooth data (τ = O(h⁵) for the
# 5-point stencil) the weights approach the optimal linear weights fast
# enough to retain the design order — including at first-order critical
# points, where WENO-JS degenerates to 3rd order.
#
#   * 5th order: τ₅ = |β₀ − β₂|
#   * 7th order: τ₇ = |β₀ − β₁ − β₂ + β₃|  (the highest-order undivided
#     difference of the indicators, Castro, Costa & Don, JCP 230 (2011) 1766)
#
# `p = 2` is the default: it sharpens the discrimination near
# discontinuities relative to the p = 1 of the original paper while
# keeping the formal order.

"""
    weno_z5_left(fm2, fm1, f0, fp1, fp2; p = 2)

5th-order WENO-Z reconstruction of the interface value at `i+1/2` from the
five cell averages `(f_{i-2}, …, f_{i+2})` — the *left-biased* (upwind for
positive advecting velocity) stencil.

Z-type nonlinear weights `α_k = d_k (1 + (τ₅/(β_k+ε))^p)` with
`τ₅ = |β₀ − β₂|`, ideal weights `d = (1/10, 6/10, 3/10)`.

Reference: Borges, Carmona, Costa & Don, *An improved weighted essentially
non-oscillatory scheme for hyperbolic conservation laws*, JCP 227 (2008).
"""
@inline weno_z5_left(fm2, fm1, f0, fp1, fp2; p::Integer = 2) =
    _weno_z5_left(promote(fm2, fm1, f0, fp1, fp2)..., p)

"""
    weno_z5_right(fm1, f0, fp1, fp2, fp3; p = 2)

Right-biased 5th-order WENO-Z reconstruction at `i+1/2` (upwind for a
*negative* advecting velocity), i.e. [`weno_z5_left`](@ref) applied to the
mirrored stencil `(f_{i-1}, …, f_{i+3})`.
"""
@inline weno_z5_right(fm1, f0, fp1, fp2, fp3; p::Integer = 2) =
    weno_z5_left(fp3, fp2, fp1, f0, fm1; p = p)

@inline function _weno_z5_left(v1::T, v2::T, v3::T, v4::T, v5::T, p::Integer) where {T}
    q0 = (2*v1 - 7*v2 + 11*v3)/6
    q1 = (-v2 + 5*v3 + 2*v4)/6
    q2 = (2*v3 + 5*v4 - v5)/6

    b0 = _b3_m2(v1, v2, v3)
    b1 = _b3_m1(v2, v3, v4)
    b2 = _b3_p0(v3, v4, v5)

    ε = _zeps(T)
    τ = abs(b0 - b2)
    a0 = T(1/10)*(one(T) + (τ/(b0 + ε))^p)
    a1 = T(6/10)*(one(T) + (τ/(b1 + ε))^p)
    a2 = T(3/10)*(one(T) + (τ/(b2 + ε))^p)
    return (a0*q0 + a1*q1 + a2*q2)/(a0 + a1 + a2)
end

"""
    weno_z7_left(fm3, fm2, fm1, f0, fp1, fp2, fp3; p = 2)

7th-order WENO-Z reconstruction of the interface value at `i+1/2` from the
seven cell averages `(f_{i-3}, …, f_{i+3})` — left-biased stencil.

Four 4-point candidate stencils with ideal weights
`d = (1/35, 12/35, 18/35, 4/35)` and the 7th-order global smoothness
measure `τ₇ = |β₀ − β₁ − β₂ + β₃|`.

Reference: Castro, Costa & Don, *High order weighted essentially
non-oscillatory WENO-Z schemes for hyperbolic conservation laws*,
JCP 230 (2011) 1766 (extending Borges et al. 2008 to arbitrary order;
candidate polynomials/indicators after Balsara & Shu, JCP 160 (2000) 405).
"""
@inline weno_z7_left(fm3, fm2, fm1, f0, fp1, fp2, fp3; p::Integer = 2) =
    _weno_z7_left(promote(fm3, fm2, fm1, f0, fp1, fp2, fp3)..., p)

"""
    weno_z7_right(fm2, fm1, f0, fp1, fp2, fp3, fp4; p = 2)

Right-biased 7th-order WENO-Z reconstruction at `i+1/2`: [`weno_z7_left`](@ref)
on the mirrored stencil `(f_{i-2}, …, f_{i+4})`.
"""
@inline weno_z7_right(fm2, fm1, f0, fp1, fp2, fp3, fp4; p::Integer = 2) =
    weno_z7_left(fp4, fp3, fp2, fp1, f0, fm1, fm2; p = p)

@inline function _weno_z7_left(v1::T, v2::T, v3::T, v4::T, v5::T, v6::T, v7::T,
                               p::Integer) where {T}
    q0 = (-3*v1 + 13*v2 - 23*v3 + 25*v4)/12
    q1 = (v2 - 5*v3 + 13*v4 + 3*v5)/12
    q2 = (-v3 + 7*v4 + 7*v5 - v6)/12
    q3 = (3*v4 + 13*v5 - 5*v6 + v7)/12

    b0 = _b4_m3(v1, v2, v3, v4)
    b1 = _b4_m2(v2, v3, v4, v5)
    b2 = _b4_m1(v3, v4, v5, v6)
    b3 = _b4_p0(v4, v5, v6, v7)

    ε = _zeps(T)
    τ = abs(b0 - b1 - b2 + b3)
    a0 = T(1/35) *(one(T) + (τ/(b0 + ε))^p)
    a1 = T(12/35)*(one(T) + (τ/(b1 + ε))^p)
    a2 = T(18/35)*(one(T) + (τ/(b2 + ε))^p)
    a3 = T(4/35) *(one(T) + (τ/(b3 + ε))^p)
    return (a0*q0 + a1*q1 + a2*q2 + a3*q3)/(a0 + a1 + a2 + a3)
end

# --- classic WENO-JS 5, written correctly --------------------------------
# Provided as the *reference* against which WENO-Z is judged. Identical in
# intent to `FiniteVolume._weno5_left`, but with the `2*f0` / `3*f0`
# products spelled out (see the Julia-literal note at the top of this
# file) so that the indicators are the true Jiang & Shu ones.

"""
    weno_js5_left(fm2, fm1, f0, fp1, fp2; ε = 1e-6)

Classic 5th-order WENO-JS (Jiang & Shu, JCP 126 (1996) 202) left-biased
reconstruction, `α_k = d_k/(β_k+ε)²`. Kept here as the reference scheme
for accuracy comparisons against [`weno_z5_left`](@ref); production runs
should prefer the Z weights, which do not lose order at critical points.
"""
@inline function weno_js5_left(fm2, fm1, f0, fp1, fp2; ε::Real = 1e-6)
    v1, v2, v3, v4, v5 = promote(fm2, fm1, f0, fp1, fp2)
    T = typeof(v1)
    q0 = (2*v1 - 7*v2 + 11*v3)/6
    q1 = (-v2 + 5*v3 + 2*v4)/6
    q2 = (2*v3 + 5*v4 - v5)/6
    b0 = _b3_m2(v1, v2, v3)
    b1 = _b3_m1(v2, v3, v4)
    b2 = _b3_p0(v3, v4, v5)
    e  = T(ε)
    a0 = T(1/10)/(b0 + e)^2
    a1 = T(6/10)/(b1 + e)^2
    a2 = T(3/10)/(b2 + e)^2
    return (a0*q0 + a1*q1 + a2*q2)/(a0 + a1 + a2)
end

"""
    weno_js5_right(fm1, f0, fp1, fp2, fp3; ε = 1e-6)

Right-biased classic WENO-JS 5 reconstruction — [`weno_js5_left`](@ref) on
the mirrored stencil.
"""
@inline weno_js5_right(fm1, f0, fp1, fp2, fp3; ε::Real = 1e-6) =
    weno_js5_left(fp3, fp2, fp1, f0, fm1; ε = ε)

# =========================================================================
# 3. TENO  (Fu, Hu & Adams, JCP 305 (2016) 333)
# =========================================================================
# TENO replaces WENO's continuous weighting by a *discrete* ENO-like
# selection followed by optimal linear weighting of the survivors:
#
#   1. smoothness indicators β_k of K candidate stencils of increasing
#      width (3,3,3 | 4,4 | 5 …);
#   2. Z-like measures  γ_k = (1 + τ_K/(β_k+ε))^q ,  χ_k = γ_k/Σγ  (q = 6);
#   3. *cut-off*: δ_k = 0 if χ_k < C_T else 1   (C_T ≈ 1e-5);
#   4. ω_k = δ_k d_k / Σ δ_m d_m with the optimal linear weights d_k.
#
# In smooth regions every δ_k = 1 and the scheme collapses *exactly* onto
# the linear upwind scheme of maximal order (5th for K = 3, 8th for
# K = 6) — no residual weight error at all, unlike any WENO variant. At a
# discontinuity the contaminated stencils are switched off completely,
# giving sharp ENO behaviour. Since `max_k χ_k ≥ 1/K > C_T`, at least one
# candidate always survives and the scheme is always well defined.
#
# Candidate stencils (offsets relative to cell `i`, interface `i+1/2`):
#
#   TENO5 (K = 3, full stencil {i-2,…,i+2}, 5th order)
#     S₀ {i-2,i-1,i}   S₁ {i-1,i,i+1}   S₂ {i,i+1,i+2}
#     d = (1/10, 6/10, 3/10)
#
#   TENO8 (K = 6, full stencil {i-4,…,i+3}, 8th order)
#     S₀,S₁,S₂ as above,  S₃ {i-3,…,i},  S₄ {i,…,i+3},  S₅ {i-4,…,i}
#     d = (3/28, 15/28, 3/14, 3/56, 1/14, 1/56)
#
# All candidate coefficients and the ideal weights above were derived in
# exact rational arithmetic (the 8-cell union admits a unique convex
# combination reproducing the 8th-order upwind reconstruction).
#
# Halo note: the TENO8 union spans five cells upwind and three downwind of
# the interface, so the flux-form sweep touches `i±5` and needs halo ≥ 5
# (WENO-Z7 and TENO5 need 4 and 3).

"""
    teno5_left(fm2, fm1, f0, fp1, fp2; C_T = 1e-5, q = 6)

5th-order TENO reconstruction at `i+1/2`, left-biased. Three 3-point
candidate stencils, ENO-like discrete selection with cut-off `C_T`, then
the optimal linear weights `d = (1/10, 6/10, 3/10)` over the survivors.

Reference: Fu, Hu & Adams, *A family of high-order targeted ENO schemes
for compressible-flow simulations*, JCP 305 (2016) 333.
"""
@inline teno5_left(fm2, fm1, f0, fp1, fp2; C_T::Real = 1e-5, q::Integer = 6) =
    _teno5_left(promote(fm2, fm1, f0, fp1, fp2)..., C_T, q)

"""
    teno5_right(fm1, f0, fp1, fp2, fp3; C_T = 1e-5, q = 6)

Right-biased 5th-order TENO reconstruction — [`teno5_left`](@ref) on the
mirrored stencil.
"""
@inline teno5_right(fm1, f0, fp1, fp2, fp3; C_T::Real = 1e-5, q::Integer = 6) =
    teno5_left(fp3, fp2, fp1, f0, fm1; C_T = C_T, q = q)

@inline function _teno5_left(v1::T, v2::T, v3::T, v4::T, v5::T,
                             C_T::Real, q::Integer) where {T}
    q0 = (2*v1 - 7*v2 + 11*v3)/6
    q1 = (-v2 + 5*v3 + 2*v4)/6
    q2 = (2*v3 + 5*v4 - v5)/6

    b0 = _b3_m2(v1, v2, v3)
    b1 = _b3_m1(v2, v3, v4)
    b2 = _b3_p0(v3, v4, v5)

    ε = _zeps(T)
    τ = abs(b0 - b2)
    g0 = (one(T) + τ/(b0 + ε))^q
    g1 = (one(T) + τ/(b1 + ε))^q
    g2 = (one(T) + τ/(b2 + ε))^q
    gs = g0 + g1 + g2
    c  = T(C_T)

    d0 = (g0/gs < c) ? zero(T) : T(1/10)
    d1 = (g1/gs < c) ? zero(T) : T(6/10)
    d2 = (g2/gs < c) ? zero(T) : T(3/10)
    return (d0*q0 + d1*q1 + d2*q2)/(d0 + d1 + d2)
end

"""
    teno8_left(fm4, fm3, fm2, fm1, f0, fp1, fp2, fp3; C_T = 1e-5, q = 6)

8th-order TENO reconstruction at `i+1/2`, left-biased, from the eight cell
averages `(f_{i-4}, …, f_{i+3})`. Six candidate stencils of widths
`3,3,3,4,4,5`; ENO cut-off `C_T`; optimal linear weights
`d = (3/28, 15/28, 3/14, 3/56, 1/14, 1/56)` reproducing the 8th-order
upwind-biased reconstruction when all candidates survive.

The global smoothness measure is
`τ = |β₅ − (β₀ + 4β₁ + β₂)/6|`, the TENO-style comparison of the widest
candidate against the Simpson-weighted narrow ones.

Reference: Fu, Hu & Adams, JCP 305 (2016) 333.
"""
@inline teno8_left(fm4, fm3, fm2, fm1, f0, fp1, fp2, fp3;
                   C_T::Real = 1e-5, q::Integer = 6) =
    _teno8_left(promote(fm4, fm3, fm2, fm1, f0, fp1, fp2, fp3)..., C_T, q)

"""
    teno8_right(fm2, fm1, f0, fp1, fp2, fp3, fp4, fp5; C_T = 1e-5, q = 6)

Right-biased 8th-order TENO reconstruction at `i+1/2` — [`teno8_left`](@ref)
on the mirrored stencil `(f_{i-2}, …, f_{i+5})`.
"""
@inline teno8_right(fm2, fm1, f0, fp1, fp2, fp3, fp4, fp5;
                    C_T::Real = 1e-5, q::Integer = 6) =
    teno8_left(fp5, fp4, fp3, fp2, fp1, f0, fm1, fm2; C_T = C_T, q = q)

@inline function _teno8_left(v1::T, v2::T, v3::T, v4::T, v5::T, v6::T, v7::T, v8::T,
                             C_T::Real, q::Integer) where {T}
    #    v1    v2    v3    v4    v5   v6    v7    v8
    #  i-4   i-3   i-2   i-1    i   i+1   i+2   i+3
    q0 = (2*v3 - 7*v4 + 11*v5)/6                       # {i-2,i-1,i}
    q1 = (-v4 + 5*v5 + 2*v6)/6                         # {i-1,i,i+1}
    q2 = (2*v5 + 5*v6 - v7)/6                          # {i,i+1,i+2}
    q3 = (-3*v2 + 13*v3 - 23*v4 + 25*v5)/12            # {i-3,…,i}
    q4 = (3*v5 + 13*v6 - 5*v7 + v8)/12                 # {i,…,i+3}
    q5 = (12*v1 - 63*v2 + 137*v3 - 163*v4 + 137*v5)/60 # {i-4,…,i}

    b0 = _b3_m2(v3, v4, v5)
    b1 = _b3_m1(v4, v5, v6)
    b2 = _b3_p0(v5, v6, v7)
    b3 = _b4_m3(v2, v3, v4, v5)
    b4 = _b4_p0(v5, v6, v7, v8)
    b5 = _b5_m4(v1, v2, v3, v4, v5)

    ε = _zeps(T)
    τ = abs(b5 - (b0 + 4*b1 + b2)/6)
    g0 = (one(T) + τ/(b0 + ε))^q
    g1 = (one(T) + τ/(b1 + ε))^q
    g2 = (one(T) + τ/(b2 + ε))^q
    g3 = (one(T) + τ/(b3 + ε))^q
    g4 = (one(T) + τ/(b4 + ε))^q
    g5 = (one(T) + τ/(b5 + ε))^q
    gs = g0 + g1 + g2 + g3 + g4 + g5
    c  = T(C_T)

    d0 = (g0/gs < c) ? zero(T) : T(3/28)
    d1 = (g1/gs < c) ? zero(T) : T(15/28)
    d2 = (g2/gs < c) ? zero(T) : T(3/14)
    d3 = (g3/gs < c) ? zero(T) : T(3/56)
    d4 = (g4/gs < c) ? zero(T) : T(1/14)
    d5 = (g5/gs < c) ? zero(T) : T(1/56)
    return (d0*q0 + d1*q1 + d2*q2 + d3*q3 + d4*q4 + d5*q5)/
           (d0 + d1 + d2 + d3 + d4 + d5)
end

# =========================================================================
# 4. Reconstruction descriptors and the generic flux-form sweep
# =========================================================================
# A descriptor bundles "which reconstruction, with which parameters" so
# that one kernel can serve every scheme without losing type stability.
# `stencil_left/right` return `Val`-wrapped (offset, width) pairs; the
# gather helpers therefore unroll to plain, bounds-check-free loads.

"""
    AbstractReconstruction

Descriptor for a face reconstruction used by the flux-form advection
kernels: [`WENOZ`](@ref), [`TENO`](@ref), `WENOJS`, `Centered2`.
"""
abstract type AbstractReconstruction end

"WENO-Z descriptor of order `N ∈ (5,7)` with τ-exponent `P`."
struct WENOZ{N,P} <: AbstractReconstruction end

"TENO descriptor of order `N ∈ (5,8)` with ENO cut-off `C_T` and exponent `q`."
struct TENO{N,Q,T<:Real} <: AbstractReconstruction
    C_T::T
end

"Classic WENO-JS descriptor (order 5)."
struct WENOJS{N,T<:Real} <: AbstractReconstruction
    ε::T
end

"2nd-order centered (linear average) descriptor — the baseline scheme."
struct Centered2 <: AbstractReconstruction end

function WENOZ(N::Integer = 5, p::Integer = 2)
    (N == 5 || N == 7) || throw(ArgumentError("WENO-Z order must be 5 or 7 (got $N)"))
    (p == 1 || p == 2) || throw(ArgumentError("WENO-Z τ-exponent p must be 1 or 2 (got $p)"))
    return WENOZ{Int(N),Int(p)}()
end

function TENO(N::Integer = 5; C_T::Real = 1e-5, q::Integer = 6)
    (N == 5 || N == 8) || throw(ArgumentError("TENO order must be 5 or 8 (got $N)"))
    C_T > 0 || throw(ArgumentError("TENO cut-off C_T must be positive (got $C_T)"))
    return TENO{Int(N),Int(q),typeof(C_T)}(C_T)
end

WENOJS(N::Integer = 5; ε::Real = 1e-6) = WENOJS{Int(N),typeof(ε)}(ε)

# stencil geometry: (offset of first cell relative to cell i, width)
@inline stencil_left(::WENOZ{5})  = (Val(-2), Val(5))
@inline stencil_right(::WENOZ{5}) = (Val(-1), Val(5))
@inline stencil_left(::WENOZ{7})  = (Val(-3), Val(7))
@inline stencil_right(::WENOZ{7}) = (Val(-2), Val(7))
@inline stencil_left(::TENO{5})   = (Val(-2), Val(5))
@inline stencil_right(::TENO{5})  = (Val(-1), Val(5))
@inline stencil_left(::TENO{8})   = (Val(-4), Val(8))
@inline stencil_right(::TENO{8})  = (Val(-2), Val(8))
@inline stencil_left(::WENOJS{5}) = (Val(-2), Val(5))
@inline stencil_right(::WENOJS{5})= (Val(-1), Val(5))
@inline stencil_left(::Centered2) = (Val(0), Val(2))
@inline stencil_right(::Centered2)= (Val(0), Val(2))

"""
    required_halo(r) -> Int

Minimum halo width (in every direction) needed by reconstruction
descriptor `r` inside the flux-form advection sweep: 1 for `Centered2`,
3 for WENO-Z5 / TENO5 / WENO-JS5, 4 for WENO-Z7, 5 for TENO8 (whose
8-cell union reaches five cells upwind of the interface).
"""
required_halo(::Centered2) = 1
required_halo(::WENOZ{5})  = 3
required_halo(::WENOZ{7})  = 4
required_halo(::TENO{5})   = 3
required_halo(::TENO{8})   = 5
required_halo(::WENOJS{5}) = 3

scheme_name(::WENOZ{5}) = :weno_z5
scheme_name(::WENOZ{7}) = :weno_z7
scheme_name(::TENO{5})  = :teno5
scheme_name(::TENO{8})  = :teno8
scheme_name(::WENOJS{5})= :weno_js5
scheme_name(::Centered2)= :centered

# reconstruction entry points ---------------------------------------------
@inline reconstruct_left(::WENOZ{5,P}, t::NTuple{5}) where {P} = _weno_z5_left(t..., P)
@inline reconstruct_right(::WENOZ{5,P}, t::NTuple{5}) where {P} =
    _weno_z5_left(reverse(t)..., P)
@inline reconstruct_left(::WENOZ{7,P}, t::NTuple{7}) where {P} = _weno_z7_left(t..., P)
@inline reconstruct_right(::WENOZ{7,P}, t::NTuple{7}) where {P} =
    _weno_z7_left(reverse(t)..., P)

@inline reconstruct_left(r::TENO{5,Q}, t::NTuple{5}) where {Q} = _teno5_left(t..., r.C_T, Q)
@inline reconstruct_right(r::TENO{5,Q}, t::NTuple{5}) where {Q} =
    _teno5_left(reverse(t)..., r.C_T, Q)
@inline reconstruct_left(r::TENO{8,Q}, t::NTuple{8}) where {Q} = _teno8_left(t..., r.C_T, Q)
@inline reconstruct_right(r::TENO{8,Q}, t::NTuple{8}) where {Q} =
    _teno8_left(reverse(t)..., r.C_T, Q)

@inline reconstruct_left(r::WENOJS{5}, t::NTuple{5}) = weno_js5_left(t...; ε = r.ε)
@inline reconstruct_right(r::WENOJS{5}, t::NTuple{5}) = weno_js5_left(reverse(t)...; ε = r.ε)

# `Centered2` gathers cells (i, i+1) and averages — the linear baseline.
@inline reconstruct_left(::Centered2, t::NTuple{2})  = (t[1] + t[2])/2
@inline reconstruct_right(::Centered2, t::NTuple{2}) = (t[1] + t[2])/2

# gather helpers ----------------------------------------------------------
@inline function _gather_x(d, c::Int, j::Int, k::Int, ::Val{lo}, ::Val{n}) where {lo,n}
    return ntuple(m -> @inbounds(d[c + lo + m - 1, j, k]), Val(n))
end
@inline function _gather_y(d, i::Int, c::Int, k::Int, ::Val{lo}, ::Val{n}) where {lo,n}
    return ntuple(m -> @inbounds(d[i, c + lo + m - 1, k]), Val(n))
end
@inline function _gather_z(d, i::Int, j::Int, c::Int, ::Val{lo}, ::Val{n}) where {lo,n}
    return ntuple(m -> @inbounds(d[i, j, c + lo + m - 1]), Val(n))
end

# Upwinded face value of `d` along each axis. `c` is the cell to the left
# of the interface (i.e. interface = face index c+1) and `vel` the
# transporting velocity at that interface.
@inline function _face_x(r, d, c::Int, j::Int, k::Int, vel)
    if vel >= 0
        lo, n = stencil_left(r);  return reconstruct_left(r, _gather_x(d, c, j, k, lo, n))
    else
        lo, n = stencil_right(r); return reconstruct_right(r, _gather_x(d, c, j, k, lo, n))
    end
end
@inline function _face_y(r, d, i::Int, c::Int, k::Int, vel)
    if vel >= 0
        lo, n = stencil_left(r);  return reconstruct_left(r, _gather_y(d, i, c, k, lo, n))
    else
        lo, n = stencil_right(r); return reconstruct_right(r, _gather_y(d, i, c, k, lo, n))
    end
end
@inline function _face_z(r, d, i::Int, j::Int, c::Int, vel)
    if vel >= 0
        lo, n = stencil_left(r);  return reconstruct_left(r, _gather_z(d, i, j, c, lo, n))
    else
        lo, n = stencil_right(r); return reconstruct_right(r, _gather_z(d, i, j, c, lo, n))
    end
end

function _check_halo(r::AbstractReconstruction, g::RegularGrid)
    H = required_halo(r)
    (g.Hx >= H && g.Hy >= H && g.Hz >= H) ||
        error("$(scheme_name(r)) advection requires halo ≥ $H in every direction " *
              "(got ($(g.Hx),$(g.Hy),$(g.Hz)))")
    return nothing
end

# --- the generic staggered flux-form scalar sweep -------------------------
# Adds -∂(u_j ϕ)/∂x_j to `tend`, with ϕ reconstructed to each face by `r`
# and upwinded on the sign of the face velocity. Structure (and the
# redundant recomputation of each interior face from both neighbours) is
# deliberately identical to `FiniteVolume._advect_weno5!`: both sides
# evaluate the *same* expression, so the flux telescopes bit-for-bit and
# Σϕ is conserved to round-off.

function _advect_reconstructed!(tend::Field{CellCenter},
                                u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                ϕ::Field{CellCenter},
                                r::AbstractReconstruction, g::RegularGrid)
    _check_halo(r, g)
    idx = 1/g.dx; idy = 1/g.dy; idz = 1/g.dz
    ii, jj, kk = interior(g)
    ϕd = ϕ.data; ud = u.data; vd = v.data; wd = w.data; td = tend.data
    @inbounds for k in kk, j in jj, i in ii
        uw = ud[i,   j, k]; ue = ud[i+1, j, k]
        Fx = (ue*_face_x(r, ϕd, i, j, k, ue) - uw*_face_x(r, ϕd, i-1, j, k, uw))*idx

        vs = vd[i, j,   k]; vn = vd[i, j+1, k]
        Fy = (vn*_face_y(r, ϕd, i, j, k, vn) - vs*_face_y(r, ϕd, i, j-1, k, vs))*idy

        wb = wd[i, j, k];   wt = wd[i, j, k+1]
        Fz = (wt*_face_z(r, ϕd, i, j, k, wt) - wb*_face_z(r, ϕd, i, j, k-1, wb))*idz

        td[i, j, k] -= Fx + Fy + Fz
    end
    return tend
end

# =========================================================================
# 5. Public high-order scalar advection
# =========================================================================

"""
    advect_weno_z!(tend, u, v, w, ϕ; order = 5, p = 2)

Add `-∂(uⱼϕ)/∂xⱼ` to `tend` using WENO-Z upwind reconstruction of `ϕ` at
each cell face, upwinded on the sign of the face velocity. Staggered
Arakawa-C only: `u @ FaceX`, `v @ FaceY`, `w @ FaceZ`, `ϕ, tend @ CellCenter`.

`order = 5` (halo ≥ 3) or `order = 7` (halo ≥ 4). `p` is the exponent of
the τ-ratio in the Z weights (1 = Borges et al. 2008 original, 2 = default,
sharper near discontinuities).

Drop-in replacement for `advect_weno5!` with the accuracy loss at critical
points removed. Conserves `Σϕ` to round-off; *not* positivity preserving —
wrap with [`advect_positive!`](@ref) for moisture and chemical species.
"""
function advect_weno_z!(tend::Field{CellCenter},
                        u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                        ϕ::Field{CellCenter}; order::Integer = 5, p::Integer = 2)
    return _advect_reconstructed!(tend, u, v, w, ϕ, WENOZ(order, p), mesh(ϕ))
end

"""
    advect_teno!(tend, u, v, w, ϕ; order = 5, C_T = 1e-5, q = 6)

Add `-∂(uⱼϕ)/∂xⱼ` to `tend` using TENO (targeted ENO) reconstruction.
`order = 5` (halo ≥ 3) or `order = 8` (halo ≥ 5).

`C_T` is the ENO cut-off on the normalised stencil measures: candidate
stencils with `χ_k < C_T` are discarded outright, the survivors are
combined with their optimal linear weights. Smaller `C_T` → closer to the
linear scheme (less dissipative, less robust); larger → more ENO-like.
In smooth regions the scheme *is* the linear upwind scheme of design
order, which makes TENO markedly less dissipative than WENO at the same
stencil width.
"""
function advect_teno!(tend::Field{CellCenter},
                      u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                      ϕ::Field{CellCenter};
                      order::Integer = 5, C_T::Real = 1e-5, q::Integer = 6)
    return _advect_reconstructed!(tend, u, v, w, ϕ, TENO(order; C_T = C_T, q = q), mesh(ϕ))
end

# =========================================================================
# 6. Positivity-preserving flux limiting (Zhang & Shu / Zalesak FCT style)
# =========================================================================
# Moisture, cloud water and chemical species must stay non-negative: a
# single negative `q` poisons microphysics and reaction rates. High-order
# upwind reconstructions are non-monotone and *will* undershoot at a steep
# front, so the fluxes are corrected before differencing.
#
# Split each face flux into a positivity-safe part and an antidiffusive
# remainder,  F = F^L + A,  A = F^H − F^L, and scale `A` per face by a
# factor `C ∈ [0,1]` taken from the *donor* cell:
#
#   ϕ^td_c = ϕ_c − Δt ∇·F^L                  (safe predictor)
#   P⁻_c   = Δt Σ_faces (antidiffusive *outflow* of cell c)
#   R⁻_c   = min(1, max(ϕ^td_c,0) / P⁻_c)     (1 if P⁻ = 0)
#   C_f    = R⁻ of whichever cell `A_f` flows out of
#
# Then for every cell the scaled outflow is ≤ ϕ^td_c while every inflow
# stays non-negative, hence
#
#   ϕ^{n+1}_c = ϕ^td_c − Δt ∇·(C A)|_c ≥ 0     — provably, not heuristically.
#
# Two fallbacks:
#   * `flux_lo = nothing` (default): F^L ≡ 0. Then ϕ^td = ϕ and positivity
#     is guaranteed *unconditionally* in Δt (the flux is simply clipped).
#   * `flux_lo = (Lx,Ly,Lz)`: first-order upwind fluxes. Accuracy in the
#     limited cells is far better; the guarantee then inherits the
#     first-order scheme's condition Σ_d |u_d| Δt/Δx_d ≤ 1.
#
# Conservation is untouched: a single limited flux is shared by the two
# cells that straddle the face.

@inline _atlo(::Nothing, i, j, k) = false          # additive zero, no promotion
@inline _atlo(a::AbstractArray, i, j, k) = @inbounds a[i, j, k]

_data_or_nothing(::Nothing) = nothing
_data_or_nothing(f::Field) = f.data

"""
    FluxLimiterWorkspace(mesh)

Pre-allocated scratch for [`advect_positive!`](@ref): high-order face
fluxes, first-order-upwind face fluxes, and the cell-centred limiter
ratio. Pass it as `workspace = ws` to keep the positive-definite advection
path allocation-free inside a time loop.
"""
struct FluxLimiterWorkspace{T, M<:RegularGrid{T}}
    Fx::Field{FaceX, T, Array{T,3}, M}
    Fy::Field{FaceY, T, Array{T,3}, M}
    Fz::Field{FaceZ, T, Array{T,3}, M}
    Lx::Field{FaceX, T, Array{T,3}, M}
    Ly::Field{FaceY, T, Array{T,3}, M}
    Lz::Field{FaceZ, T, Array{T,3}, M}
    ratio::Field{CellCenter, T, Array{T,3}, M}
end

function FluxLimiterWorkspace(m::RegularGrid)
    return FluxLimiterWorkspace(Field(FaceX(), m), Field(FaceY(), m), Field(FaceZ(), m),
                                Field(FaceX(), m), Field(FaceY(), m), Field(FaceZ(), m),
                                Field(CellCenter(), m))
end

# --- face-flux assembly ---------------------------------------------------
# Fluxes are stored at their own face index over the interior *plus* the
# one extra face that closes the last cell (`last+1`).

function _reconstructed_fluxes!(Fx::Field{FaceX}, Fy::Field{FaceY}, Fz::Field{FaceZ},
                                u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                ϕ::Field{CellCenter}, r::AbstractReconstruction,
                                g::RegularGrid)
    _check_halo(r, g)
    ii, jj, kk = interior(g)
    ϕd = ϕ.data
    @inbounds for k in kk, j in jj, i in first(ii):last(ii)+1
        vel = u.data[i, j, k]
        Fx.data[i, j, k] = vel * _face_x(r, ϕd, i-1, j, k, vel)
    end
    @inbounds for k in kk, j in first(jj):last(jj)+1, i in ii
        vel = v.data[i, j, k]
        Fy.data[i, j, k] = vel * _face_y(r, ϕd, i, j-1, k, vel)
    end
    @inbounds for k in first(kk):last(kk)+1, j in jj, i in ii
        vel = w.data[i, j, k]
        Fz.data[i, j, k] = vel * _face_z(r, ϕd, i, j, k-1, vel)
    end
    return (Fx, Fy, Fz)
end

function _upwind_fluxes!(Fx::Field{FaceX}, Fy::Field{FaceY}, Fz::Field{FaceZ},
                         u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                         ϕ::Field{CellCenter}, g::RegularGrid)
    ii, jj, kk = interior(g)
    ϕd = ϕ.data
    @inbounds for k in kk, j in jj, i in first(ii):last(ii)+1
        vel = u.data[i, j, k]
        Fx.data[i, j, k] = vel * (vel >= 0 ? ϕd[i-1, j, k] : ϕd[i, j, k])
    end
    @inbounds for k in kk, j in first(jj):last(jj)+1, i in ii
        vel = v.data[i, j, k]
        Fy.data[i, j, k] = vel * (vel >= 0 ? ϕd[i, j-1, k] : ϕd[i, j, k])
    end
    @inbounds for k in first(kk):last(kk)+1, j in jj, i in ii
        vel = w.data[i, j, k]
        Fz.data[i, j, k] = vel * (vel >= 0 ? ϕd[i, j, k-1] : ϕd[i, j, k])
    end
    return (Fx, Fy, Fz)
end

function _add_flux_divergence!(tend::Field{CellCenter},
                               Fx::Field{FaceX}, Fy::Field{FaceY}, Fz::Field{FaceZ},
                               g::RegularGrid)
    idx = 1/g.dx; idy = 1/g.dy; idz = 1/g.dz
    ii, jj, kk = interior(g)
    fx = Fx.data; fy = Fy.data; fz = Fz.data; td = tend.data
    @inbounds for k in kk, j in jj, i in ii
        td[i,j,k] -= (fx[i+1,j,k] - fx[i,j,k])*idx +
                     (fy[i,j+1,k] - fy[i,j,k])*idy +
                     (fz[i,j,k+1] - fz[i,j,k])*idz
    end
    return tend
end

# one-layer face-normal periodic wrap of a cell-centred field
function _wrap_one_layer!(f::Field{CellCenter}, g::RegularGrid)
    ii, jj, kk = interior(g)
    d = f.data
    i1, i2 = first(ii), last(ii); j1, j2 = first(jj), last(jj); k1, k2 = first(kk), last(kk)
    @inbounds for k in kk, j in jj
        d[i1-1, j, k] = d[i2, j, k]; d[i2+1, j, k] = d[i1, j, k]
    end
    @inbounds for k in kk, i in ii
        d[i, j1-1, k] = d[i, j2, k]; d[i, j2+1, k] = d[i, j1, k]
    end
    @inbounds for j in jj, i in ii
        d[i, j, k1-1] = d[i, j, k2]; d[i, j, k2+1] = d[i, j, k1]
    end
    return f
end

"""
    apply_positivity_limiter!(flux_x, flux_y, flux_z, ϕ, dt, mesh;
                              flux_lo = nothing, ratio = nothing, bcs = nothing)

Limit the face fluxes `(flux_x, flux_y, flux_z)` *in place* so that the
forward-Euler update

    ϕ^{n+1} = ϕ - dt * ∇·flux

cannot make a non-negative `ϕ` negative. Flux-corrected-transport /
Zhang–Shu style: the antidiffusive part of each flux (relative to
`flux_lo`, or to zero) is scaled by the smallest factor the donor cell can
afford. Conservative — the two cells sharing a face see the same limited
flux.

  * `flux_lo` — optional `(Field{FaceX}, Field{FaceY}, Field{FaceZ})` of
    positivity-safe low-order (first-order upwind) fluxes to fall back to.
    With `nothing` the fallback is zero flux, which makes the guarantee
    unconditional in `dt`; with upwind fluxes the guarantee holds under the
    first-order CFL condition `Σ_d |u_d| dt/Δx_d ≤ 1`.
  * `ratio` — optional `Field{CellCenter}` scratch for the limiter factors.
  * `bcs` — if given, the limiter field's halo is filled with
    `fill_halos!(ratio, bcs)`; otherwise a one-layer periodic wrap is used
    (harmless at impermeable walls, where the wall-face flux is zero
    anyway).

Returns `(flux_x, flux_y, flux_z)`.
"""
function apply_positivity_limiter!(flux_x::Field{FaceX,T}, flux_y::Field{FaceY,T},
                                   flux_z::Field{FaceZ,T}, ϕ::Field{CellCenter,T},
                                   dt::Real, m::RegularGrid{T};
                                   flux_lo = nothing,
                                   ratio::Union{Nothing,Field{CellCenter}} = nothing,
                                   bcs::Union{Nothing,FieldBoundaryConditions} = nothing) where {T}
    Δt = T(dt)
    R  = ratio === nothing ? Field(CellCenter(), m) : ratio
    Lx = flux_lo === nothing ? nothing : _data_or_nothing(flux_lo[1])
    Ly = flux_lo === nothing ? nothing : _data_or_nothing(flux_lo[2])
    Lz = flux_lo === nothing ? nothing : _data_or_nothing(flux_lo[3])
    return _positivity_limit!(flux_x, flux_y, flux_z, ϕ, Δt, m, R, Lx, Ly, Lz, bcs)
end

function _positivity_limit!(flux_x, flux_y, flux_z, ϕ, Δt::T, m::RegularGrid{T},
                            R, Lx, Ly, Lz, bcs) where {T}
    idx = 1/m.dx; idy = 1/m.dy; idz = 1/m.dz
    ii, jj, kk = interior(m)
    fx = flux_x.data; fy = flux_y.data; fz = flux_z.data
    ϕd = ϕ.data; Rd = R.data
    z = zero(T)

    # --- pass 1: antidiffusive outflow budget per cell --------------------
    @inbounds for k in kk, j in jj, i in ii
        lxe = _atlo(Lx, i+1, j, k); lxw = _atlo(Lx, i,   j, k)
        lyn = _atlo(Ly, i, j+1, k); lys = _atlo(Ly, i,   j, k)
        lzt = _atlo(Lz, i, j, k+1); lzb = _atlo(Lz, i, j, k  )

        axe = fx[i+1,j,k] - lxe; axw = fx[i,j,k] - lxw
        ayn = fy[i,j+1,k] - lyn; ays = fy[i,j,k] - lys
        azt = fz[i,j,k+1] - lzt; azb = fz[i,j,k] - lzb

        Pm = Δt * ((max(axe, z) + max(-axw, z))*idx +
                   (max(ayn, z) + max(-ays, z))*idy +
                   (max(azt, z) + max(-azb, z))*idz)

        div_lo = (lxe - lxw)*idx + (lyn - lys)*idy + (lzt - lzb)*idz
        Q = max(ϕd[i,j,k] - Δt*div_lo, z)

        Rd[i,j,k] = Pm > z ? min(one(T), Q/Pm) : one(T)
    end

    # --- limiter halo (needed by the closing faces) -----------------------
    if bcs === nothing
        _wrap_one_layer!(R, m)
    else
        fill_halos!(R, bcs)
    end

    # --- pass 2: scale each antidiffusive flux by the donor's factor ------
    @inbounds for k in kk, j in jj, i in first(ii):last(ii)+1
        lo = _atlo(Lx, i, j, k)
        a  = fx[i,j,k] - lo
        c  = a >= z ? Rd[i-1,j,k] : Rd[i,j,k]
        fx[i,j,k] = lo + c*a
    end
    @inbounds for k in kk, j in first(jj):last(jj)+1, i in ii
        lo = _atlo(Ly, i, j, k)
        a  = fy[i,j,k] - lo
        c  = a >= z ? Rd[i,j-1,k] : Rd[i,j,k]
        fy[i,j,k] = lo + c*a
    end
    @inbounds for k in first(kk):last(kk)+1, j in jj, i in ii
        lo = _atlo(Lz, i, j, k)
        a  = fz[i,j,k] - lo
        c  = a >= z ? Rd[i,j,k-1] : Rd[i,j,k]
        fz[i,j,k] = lo + c*a
    end
    return (flux_x, flux_y, flux_z)
end

"""
    advect_positive!(tend, u, v, w, ϕ, dt; scheme = :weno_z5, C_T = 1e-5,
                     p = 2, workspace = nothing, bcs = nothing)

Positivity-preserving high-order scalar advection: reconstruct, limit,
difference. Adds `-∂(uⱼϕ)/∂xⱼ` to `tend` such that a forward-Euler update
of `ϕ` with the *same* `dt` keeps `ϕ ≥ 0` exactly, provided

  * `ϕ ≥ 0` on entry (including its halos), and
  * the first-order CFL condition `Σ_d |u_d| dt/Δx_d ≤ 1` holds.

`scheme` selects the high-order reconstruction (`:weno_z5`, `:weno_z7`,
`:teno5`, `:teno8`, `:weno5`, `:centered`). The flux correction blends
towards first-order upwind only in the cells that would otherwise go
negative, so smooth regions keep the full high-order accuracy.

`tend` must hold only this advection contribution for the guarantee to be
meaningful; other tendencies (diffusion, sources) are limited separately or
must be sign-safe themselves. Pass a [`FluxLimiterWorkspace`](@ref) as
`workspace` to avoid the internal allocation.
"""
function advect_positive!(tend::Field{CellCenter,T}, u::Field{FaceX,T},
                          v::Field{FaceY,T}, w::Field{FaceZ,T},
                          ϕ::Field{CellCenter,T}, dt::Real;
                          scheme::Symbol = :weno_z5, C_T::Real = 1e-5, p::Integer = 2,
                          workspace::Union{Nothing,FluxLimiterWorkspace} = nothing,
                          bcs::Union{Nothing,FieldBoundaryConditions} = nothing) where {T}
    m  = mesh(ϕ)
    r  = _reconstruction(scheme; C_T = C_T, p = p)
    ws = workspace === nothing ? FluxLimiterWorkspace(m) : workspace

    _reconstructed_fluxes!(ws.Fx, ws.Fy, ws.Fz, u, v, w, ϕ, r, m)
    _upwind_fluxes!(ws.Lx, ws.Ly, ws.Lz, u, v, w, ϕ, m)
    apply_positivity_limiter!(ws.Fx, ws.Fy, ws.Fz, ϕ, dt, m;
                              flux_lo = (ws.Lx, ws.Ly, ws.Lz), ratio = ws.ratio, bcs = bcs)
    _add_flux_divergence!(tend, ws.Fx, ws.Fy, ws.Fz, m)
    return tend
end

# =========================================================================
# 7. High-order momentum advection (Arakawa-C, WENO-Z)
# =========================================================================
# Same control volumes and interpolation pattern as
# `FiniteVolume._momentum_advect!`: the *transporting* velocity is linearly
# interpolated to the flux face (that is what keeps the discrete continuity
# constraint intact) while the *transported* momentum component is
# reconstructed to the face with WENO-Z and upwinded on the sign of the
# transporting velocity.
#
# Flux faces per component:
#   u @ FaceX : uu at cell centres (x), vu at x-y edges, wu at x-z edges
#   v @ FaceY : uv at x-y edges,    vv at cell centres (y), wv at y-z edges
#   w @ FaceZ : uw at x-z edges,    vw at y-z edges,  ww at cell centres (z)
#
# Reducing the reconstruction to a 2-point average recovers
# `_momentum_advect!` exactly.

"""
    momentum_advect_weno_z!(tu, tv, tw, u, v, w; order = 5, p = 2)

Add `-∂(uⱼuᵢ)/∂xⱼ` to the face-centred momentum tendencies using WENO-Z
reconstruction of the transported momentum component, with the transporting
velocity linearly interpolated to the flux face (Arakawa-C conventions of
[`momentum_advect!`](@ref) preserved exactly).

`order = 5` needs halo ≥ 3, `order = 7` halo ≥ 4. Upwind-biased and
therefore *dissipative*: momentum is conserved (flux form) but discrete
kinetic energy is not — use [`momentum_advect_skew!`](@ref) when energy
conservation matters, or this one when robustness at sharp shear layers
does.
"""
function momentum_advect_weno_z!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                 u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ};
                                 order::Integer = 5, p::Integer = 2)
    return _momentum_advect_recon!(tu, tv, tw, u, v, w, WENOZ(order, p), mesh(u))
end

function _momentum_advect_recon!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                 u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                 r::AbstractReconstruction, g::RegularGrid)
    _check_halo(r, g)
    idx = 1/g.dx; idy = 1/g.dy; idz = 1/g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data

    # --- u-momentum (west face of cell i,j,k) ---------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i,   j, k] + ud[i+1, j, k])/2      # at cell centre i
        Uw = (ud[i-1, j, k] + ud[i,   j, k])/2      # at cell centre i-1
        d_uu_dx = (Ue*_face_x(r, ud, i, j, k, Ue) - Uw*_face_x(r, ud, i-1, j, k, Uw))*idx

        Vn = (vd[i-1, j+1, k] + vd[i, j+1, k])/2    # at x-y edge (i,j+1)
        Vs = (vd[i-1, j,   k] + vd[i, j,   k])/2    # at x-y edge (i,j)
        d_vu_dy = (Vn*_face_y(r, ud, i, j, k, Vn) - Vs*_face_y(r, ud, i, j-1, k, Vs))*idy

        Wt = (wd[i-1, j, k+1] + wd[i, j, k+1])/2    # at x-z edge (i,k+1)
        Wb = (wd[i-1, j, k  ] + wd[i, j, k  ])/2    # at x-z edge (i,k)
        d_wu_dz = (Wt*_face_z(r, ud, i, j, k, Wt) - Wb*_face_z(r, ud, i, j, k-1, Wb))*idz

        tu.data[i, j, k] -= d_uu_dx + d_vu_dy + d_wu_dz
    end

    # --- v-momentum (south face of cell i,j,k) --------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i+1, j-1, k] + ud[i+1, j, k])/2
        Uw = (ud[i,   j-1, k] + ud[i,   j, k])/2
        d_uv_dx = (Ue*_face_x(r, vd, i, j, k, Ue) - Uw*_face_x(r, vd, i-1, j, k, Uw))*idx

        Vn = (vd[i, j,   k] + vd[i, j+1, k])/2
        Vs = (vd[i, j-1, k] + vd[i, j,   k])/2
        d_vv_dy = (Vn*_face_y(r, vd, i, j, k, Vn) - Vs*_face_y(r, vd, i, j-1, k, Vs))*idy

        Wt = (wd[i, j-1, k+1] + wd[i, j, k+1])/2
        Wb = (wd[i, j-1, k  ] + wd[i, j, k  ])/2
        d_wv_dz = (Wt*_face_z(r, vd, i, j, k, Wt) - Wb*_face_z(r, vd, i, j, k-1, Wb))*idz

        tv.data[i, j, k] -= d_uv_dx + d_vv_dy + d_wv_dz
    end

    # --- w-momentum (bottom face of cell i,j,k) -------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i+1, j, k-1] + ud[i+1, j, k])/2
        Uw = (ud[i,   j, k-1] + ud[i,   j, k])/2
        d_uw_dx = (Ue*_face_x(r, wd, i, j, k, Ue) - Uw*_face_x(r, wd, i-1, j, k, Uw))*idx

        Vn = (vd[i, j+1, k-1] + vd[i, j+1, k])/2
        Vs = (vd[i, j,   k-1] + vd[i, j,   k])/2
        d_vw_dy = (Vn*_face_y(r, wd, i, j, k, Vn) - Vs*_face_y(r, wd, i, j-1, k, Vs))*idy

        Wt = (wd[i, j, k  ] + wd[i, j, k+1])/2
        Wb = (wd[i, j, k-1] + wd[i, j, k  ])/2
        d_ww_dz = (Wt*_face_z(r, wd, i, j, k, Wt) - Wb*_face_z(r, wd, i, j, k-1, Wb))*idz

        tw.data[i, j, k] -= d_uw_dx + d_vw_dy + d_ww_dz
    end

    return (tu, tv, tw)
end

# =========================================================================
# 8. Phase 28 — energy- and variance-conserving (skew-symmetric) operators
# =========================================================================
# Write a one-dimensional advective contribution at node `n` with the
# transporting flux `U` living on the faces `n±1/2` of the node's control
# volume and the transported quantity `φ` at the nodes. The two classical
# discrete forms are
#
#   divergence form   C^div_n = [U₊·½(φₙ+φ₊) − U₋·½(φ₋+φₙ)] / h
#   advective form    C^adv_n = [U₊·(φ₊−φₙ)  + U₋·(φₙ−φ₋)]  / 2h
#
# and their discrete energy sums are *exactly* opposite:
#
#   Σₙ φₙ C^div_n = +Σₙ ½φₙ² (δU)ₙ ,   Σₙ φₙ C^adv_n = −Σₙ ½φₙ² (δU)ₙ .
#
# Hence the Morinishi (1998) skew-symmetric form
#
#   C^skew = ½(C^div + C^adv) = [U₊φ₊ − U₋φ₋] / 2h
#
# satisfies Σₙ φₙ C^skew_n ≡ 0 identically — *without* requiring the
# discrete divergence of `U` to vanish. The compact right-hand expression
# is what is implemented below (it is algebraically identical to the average
# of the two forms and half the arithmetic); it is 2nd-order accurate and
# consistent, reducing to the standard centred difference for uniform `U`.
#
# What this buys, precisely:
#
#   * conserved to machine precision: Σ ½(u²+v²+w²) over a periodic domain,
#     for *any* velocity field, divergence-free or not. (The cancellation is
#     exact algebraically; floating point leaves an O(ε_mach) relative
#     residue because the two adjacent nodes multiply the same three numbers
#     in a different order.)
#   * conserved when — and only when — the discrete divergence vanishes:
#     momentum Σuᵢ (the divergence form conserves it unconditionally).
#   * NOT provided: monotonicity, boundedness, any dissipation at all. The
#     operator is centred; the SGS model (or explicit filtering) must supply
#     the dissipation. For sharp fronts/shocks use the Phase 26 WENO-Z/TENO
#     operators instead.
#   * The *existing* 2nd-order flux form `momentum_advect!` conserves the
#     same energy norm too, but only for a discretely divergence-free
#     velocity: its energy production is exactly Σᵢ ½uᵢ²·(∇·u)|ᵢ. Inside a
#     fractional-step method the advection is evaluated on the projected
#     (div-free to round-off) field, so the two agree there; they part
#     company whenever ∇·u ≠ 0 — sub-iterations, nested/forced boundaries,
#     IBM masks, or a loosened Poisson tolerance.
#
# Reference: Morinishi, Lund, Vasilyev & Moin, *Fully conservative higher
# order finite difference schemes for incompressible flow*, JCP 143 (1998) 90.

"""
    momentum_advect_skew!(tu, tv, tw, u, v, w)

Add the Morinishi (1998) fully-conservative ("skew-symmetric") momentum
advection `-∂(uⱼuᵢ)/∂xⱼ` to the face-centred tendencies on the Arakawa-C
grid.

Conservation properties (periodic domain, interior sum):

  * **discrete kinetic energy `Σ½(u²+v²+w²)` is conserved exactly** — to
    round-off, and for an arbitrary velocity field, not only a
    divergence-free one. This is the defining property of the form.
  * momentum `Σuᵢ` is conserved only for discretely divergence-free
    velocity (unlike the flux form [`momentum_advect!`](@ref), which
    conserves momentum unconditionally and energy only for `∇·u = 0`).
  * 2nd-order accurate, centred, **zero numerical dissipation**: it needs an
    SGS model or explicit filtering to remove enstrophy at the grid scale,
    and it will ring at underresolved fronts.

Halos of `u, v, w` must be filled (with `Periodic` BCs for the exact
conservation statement to apply).
"""
momentum_advect_skew!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                      u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ}) =
    _momentum_advect_skew!(tu, tv, tw, u, v, w, mesh(u))

function _momentum_advect_skew!(tu::Field{FaceX}, tv::Field{FaceY}, tw::Field{FaceZ},
                                u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                g::RegularGrid)
    h2x = 1/(2g.dx); h2y = 1/(2g.dy); h2z = 1/(2g.dz)
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data

    # --- u-momentum ------------------------------------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i,   j, k] + ud[i+1, j, k])/2
        Uw = (ud[i-1, j, k] + ud[i,   j, k])/2
        Vn = (vd[i-1, j+1, k] + vd[i, j+1, k])/2
        Vs = (vd[i-1, j,   k] + vd[i, j,   k])/2
        Wt = (wd[i-1, j, k+1] + wd[i, j, k+1])/2
        Wb = (wd[i-1, j, k  ] + wd[i, j, k  ])/2

        tu.data[i,j,k] -= (Ue*ud[i+1,j,k] - Uw*ud[i-1,j,k])*h2x +
                          (Vn*ud[i,j+1,k] - Vs*ud[i,j-1,k])*h2y +
                          (Wt*ud[i,j,k+1] - Wb*ud[i,j,k-1])*h2z
    end

    # --- v-momentum ------------------------------------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i+1, j-1, k] + ud[i+1, j, k])/2
        Uw = (ud[i,   j-1, k] + ud[i,   j, k])/2
        Vn = (vd[i, j,   k] + vd[i, j+1, k])/2
        Vs = (vd[i, j-1, k] + vd[i, j,   k])/2
        Wt = (wd[i, j-1, k+1] + wd[i, j, k+1])/2
        Wb = (wd[i, j-1, k  ] + wd[i, j, k  ])/2

        tv.data[i,j,k] -= (Ue*vd[i+1,j,k] - Uw*vd[i-1,j,k])*h2x +
                          (Vn*vd[i,j+1,k] - Vs*vd[i,j-1,k])*h2y +
                          (Wt*vd[i,j,k+1] - Wb*vd[i,j,k-1])*h2z
    end

    # --- w-momentum ------------------------------------------------------
    @inbounds for k in kk, j in jj, i in ii
        Ue = (ud[i+1, j, k-1] + ud[i+1, j, k])/2
        Uw = (ud[i,   j, k-1] + ud[i,   j, k])/2
        Vn = (vd[i, j+1, k-1] + vd[i, j+1, k])/2
        Vs = (vd[i, j,   k-1] + vd[i, j,   k])/2
        Wt = (wd[i, j, k  ] + wd[i, j, k+1])/2
        Wb = (wd[i, j, k-1] + wd[i, j, k  ])/2

        tw.data[i,j,k] -= (Ue*wd[i+1,j,k] - Uw*wd[i-1,j,k])*h2x +
                          (Vn*wd[i,j+1,k] - Vs*wd[i,j-1,k])*h2y +
                          (Wt*wd[i,j,k+1] - Wb*wd[i,j,k-1])*h2z
    end

    return (tu, tv, tw)
end

"""
    advect_variance_conserving!(tend, u, v, w, ϕ)

Skew-symmetric scalar advection: add
`-½[∂(uⱼϕ)/∂xⱼ + uⱼ∂ϕ/∂xⱼ]` to `tend` for a cell-centred scalar with
face-centred transporting velocities.

Conservation properties (periodic domain, interior sum):

  * **`Σϕ²` is conserved exactly** (round-off), for any velocity field —
    the discrete analogue of `d/dt ∫ϕ²/2 = -∫ϕ²(∇·u)/2 = 0`;
  * `Σϕ` is conserved when the discrete velocity divergence vanishes;
  * 2nd-order accurate, centred, non-dissipative — it does not bound `ϕ`
    and will overshoot at sharp fronts. Use [`advect_weno_z!`](@ref) /
    [`advect_teno!`](@ref) there, or [`advect_positive!`](@ref) for
    positive-definite species.

This is the scalar counterpart of [`momentum_advect_skew!`](@ref) and is
the right choice when the physics of interest is the scalar-variance
cascade (spectra, structure functions, SGS-model calibration), where the
centred flux form's spurious variance production contaminates the budget.
"""
advect_variance_conserving!(tend::Field{CellCenter},
                            u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                            ϕ::Field{CellCenter}) =
    _advect_variance_conserving!(tend, u, v, w, ϕ, mesh(ϕ))

function _advect_variance_conserving!(tend::Field{CellCenter},
                                     u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ},
                                     ϕ::Field{CellCenter}, g::RegularGrid)
    h2x = 1/(2g.dx); h2y = 1/(2g.dy); h2z = 1/(2g.dz)
    ii, jj, kk = interior(g)
    ϕd = ϕ.data; ud = u.data; vd = v.data; wd = w.data
    @inbounds for k in kk, j in jj, i in ii
        tend.data[i,j,k] -= (ud[i+1,j,k]*ϕd[i+1,j,k] - ud[i,j,k]*ϕd[i-1,j,k])*h2x +
                            (vd[i,j+1,k]*ϕd[i,j+1,k] - vd[i,j,k]*ϕd[i,j-1,k])*h2y +
                            (wd[i,j,k+1]*ϕd[i,j,k+1] - wd[i,j,k]*ϕd[i,j,k-1])*h2z
    end
    return tend
end

# =========================================================================
# 9. Conservation diagnostics
# =========================================================================

_zero!(fs::Field...) = (for f in fs; fill!(f.data, 0); end; nothing)

# Σ_interior a·b dV at a common location
function _dot_interior(a::Field, b::Field, g::RegularGrid)
    ii, jj, kk = interior(g)
    s = zero(eltype(a))
    ad = a.data; bd = b.data
    @inbounds for k in kk, j in jj, i in ii
        s += ad[i,j,k]*bd[i,j,k]
    end
    return s * (g.dx*g.dy*g.dz)
end

function _uvw_dot(u, v, w, tu, tv, tw, g::RegularGrid)
    return _dot_interior(u, tu, g) + _dot_interior(v, tv, g) + _dot_interior(w, tw, g)
end

function _accumulate!(dest::Field, a::Real, src::Field, g::RegularGrid)
    ii, jj, kk = interior(g)
    @inbounds @views dest.data[ii,jj,kk] .+= a .* src.data[ii,jj,kk]
    return dest
end

"""
    kinetic_energy_budget(sim; momentum_advection! = momentum_advect!, sgs = nothing) -> NamedTuple

Term-by-term budget of the resolved kinetic energy
`E = ∫½(u²+v²+w²) dV` for the current state of `sim`, evaluated with the
simulation's own physics settings. Returned fields (all in energy per unit
time, integrated over the interior volume):

| field         | meaning / expected value |
| :------------ | :----------------------- |
| `kinetic_energy` | `E` itself |
| `advection`   | `Σu·C(u)` — **≈ 0 iff the advection operator conserves energy on this state**; this is the instrument that validates Phase 28 |
| `buoyancy`    | `Σw·b` — buoyancy production/destruction (0 if buoyancy off) |
| `coriolis`    | exactly 0 — the C-grid 4-point averaging in [`add_coriolis!`](@ref) is its own adjoint, so `f×u` does no work even discretely |
| `pressure`    | `-Σu·∇p` — 0 up to round-off for a discretely divergence-free field |
| `viscous`     | `νΣu·∇²u` ≤ 0 — molecular dissipation (0 if `ν = 0`) |
| `sgs`         | SGS dissipation (0 if `C_s = 0`); ≤ 0 for a positive eddy viscosity |
| `total`       | `Σu·F` with `F` the *fully assembled* tendency = `dE/dt` |
| `residual`    | `total − Σ(terms)` — closure of the decomposition, round-off sized |

Pass `momentum_advection! = momentum_advect_skew!` (or
`momentum_advect_weno_z!`) to budget an alternative operator: the
`advection` entry then measures that operator's energy conservation.

`sgs`: an SGS closure of the Era VI contract (`Operators.AbstractSGSClosure`,
or a function `f(νt, u, v, w)`) to budget instead of the built-in
Smagorinsky: its coefficients are computed with `compute_sgs!` (pure: no
closure memory changes; `sim.νt`, `sim.κt` are overwritten) and its
momentum tendency `add_sgs_momentum!` gives the `sgs` entry. This is the
resolved-KE sink the prognostic TKE closure's shear production must equal
(P56-b, #182).

Diagnostic routine — it allocates its own scratch fields and does not
touch `sim`'s prognostic state (halos are refilled as a side effect).
"""
function kinetic_energy_budget(sim::BoussinesqSimulation{T};
                               momentum_advection! = momentum_advect!,
                               sgs = nothing) where {T}
    m = sim.mesh
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    fill_halos!(sim.θ, sim.θ_bcs)
    sim.has_moisture && fill_halos!(sim.qv, sim.θ_bcs)

    tu = Field(FaceX(), m); tv = Field(FaceY(), m); tw = Field(FaceZ(), m)
    au = Field(FaceX(), m); av = Field(FaceY(), m); aw = Field(FaceZ(), m)
    u, v, w = sim.u, sim.v, sim.w

    # --- advection -------------------------------------------------------
    _zero!(tu, tv, tw)
    momentum_advection!(tu, tv, tw, u, v, w)
    adv = _uvw_dot(u, v, w, tu, tv, tw, m)
    _accumulate!(au, one(T), tu, m); _accumulate!(av, one(T), tv, m); _accumulate!(aw, one(T), tw, m)

    # --- buoyancy --------------------------------------------------------
    buoy = zero(T)
    if sim.has_buoyancy
        _zero!(tw)
        if sim.has_moisture
            add_buoyancy!(tw, sim.θ, sim.qv, sim.θ₀, sim.g_accel)
        else
            add_buoyancy!(tw, sim.θ, sim.θ₀, sim.g_accel)
        end
        buoy = _dot_interior(w, tw, m)
        _accumulate!(aw, one(T), tw, m)
    end

    # --- Coriolis --------------------------------------------------------
    cor = zero(T)
    if sim.has_coriolis
        _zero!(tu, tv)
        add_coriolis!(tu, tv, u, v, sim.f_coriolis)
        cor = _dot_interior(u, tu, m) + _dot_interior(v, tv, m)
        _accumulate!(au, one(T), tu, m); _accumulate!(av, one(T), tv, m)
    end

    # --- pressure work ---------------------------------------------------
    fill_halos!(sim.p, sim.bcs)
    ∂x(tu, sim.p); ∂y(tv, sim.p); ∂z(tw, sim.p)
    pres = -(_dot_interior(u, tu, m) + _dot_interior(v, tv, m) + _dot_interior(w, tw, m))
    _accumulate!(au, -one(T), tu, m); _accumulate!(av, -one(T), tv, m); _accumulate!(aw, -one(T), tw, m)

    # --- molecular viscosity --------------------------------------------
    visc = zero(T)
    if sim.has_viscosity
        laplacian(tu, u); laplacian(tv, v); laplacian(tw, w)
        visc = sim.ν * (_dot_interior(u, tu, m) + _dot_interior(v, tv, m) + _dot_interior(w, tw, m))
        _accumulate!(au, sim.ν, tu, m); _accumulate!(av, sim.ν, tv, m); _accumulate!(aw, sim.ν, tw, m)
    end

    # --- SGS -------------------------------------------------------------
    closure = sgs_closure(sgs)
    sgs = zero(T)
    if closure !== nothing
        f = compute_sgs!(closure, sim)
        _zero!(tu, tv, tw)
        add_sgs_momentum!(tu, tv, tw, closure, f, sim)
        sgs = _uvw_dot(u, v, w, tu, tv, tw, m)
        _accumulate!(au, one(T), tu, m); _accumulate!(av, one(T), tv, m); _accumulate!(aw, one(T), tw, m)
    elseif sim.has_sgs
        compute_smagorinsky_νt!(sim.νt, u, v, w, sim.C_s)
        fill_halos!(sim.νt, sim.bcs)
        _zero!(tu, tv, tw)
        add_eddy_viscosity_u!(tu, u, sim.νt)
        add_eddy_viscosity_v!(tv, v, sim.νt)
        add_eddy_viscosity_w!(tw, w, sim.νt)
        sgs = _uvw_dot(u, v, w, tu, tv, tw, m)
        _accumulate!(au, one(T), tu, m); _accumulate!(av, one(T), tv, m); _accumulate!(aw, one(T), tw, m)
    end

    total = _uvw_dot(u, v, w, au, av, aw, m)
    return (kinetic_energy = Simulations.kinetic_energy(sim),
            advection = adv, buoyancy = buoy, coriolis = cor, pressure = pres,
            viscous = visc, sgs = sgs, total = total,
            residual = total - (adv + buoy + cor + pres + visc + sgs))
end

"""
    scalar_variance_budget(sim, ϕ; advection! = advect_centered!) -> NamedTuple

Budget of the scalar "energy" `½∫ϕ² dV` for a cell-centred field `ϕ`
advected by `sim`'s velocity, the direct analogue of
[`kinetic_energy_budget`](@ref):

| field | meaning / expected value |
| :---- | :----------------------- |
| `variance` | `½∫ϕ² dV` |
| `advection` | `Σϕ·C(ϕ)` — **≈ 0 iff the scalar operator conserves variance**; identically `Σ½ϕ²(∇·u)` for the centred flux form |
| `diffusion` | `νΣϕ∇²ϕ` ≤ 0 (0 if `ν = 0`) |
| `sgs` | SGS scalar dissipation (0 if `C_s = 0`) |
| `surface_flux` | production by the prescribed wall fluxes (0 if off) |
| `total` | `Σϕ·F` = `d/dt ½∫ϕ²` |
| `residual` | `total − Σ(terms)`, round-off sized |

Pass `advection! = advect_variance_conserving!` to budget the
skew-symmetric operator, or any of the Phase 26 schemes.
"""
function scalar_variance_budget(sim::BoussinesqSimulation{T}, ϕ::Field{CellCenter};
                                advection! = advect_centered!) where {T}
    m = sim.mesh
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    fill_halos!(ϕ, sim.θ_bcs)
    tϕ = Field(CellCenter(), m); aϕ = Field(CellCenter(), m)
    u, v, w = sim.u, sim.v, sim.w

    advection!(tϕ, u, v, w, ϕ)
    adv = _dot_interior(ϕ, tϕ, m)
    _accumulate!(aϕ, one(T), tϕ, m)

    diff = zero(T)
    if sim.has_viscosity
        laplacian(tϕ, ϕ)
        diff = sim.ν * _dot_interior(ϕ, tϕ, m)
        _accumulate!(aϕ, sim.ν, tϕ, m)
    end

    sgs = zero(T)
    if sim.has_sgs
        compute_smagorinsky_νt!(sim.νt, u, v, w, sim.C_s)
        fill_halos!(sim.νt, sim.bcs)
        _zero!(tϕ)
        add_eddy_diffusion_scalar!(tϕ, ϕ, sim.νt, sim.Pr_t)
        sgs = _dot_interior(ϕ, tϕ, m)
        _accumulate!(aϕ, one(T), tϕ, m)
    end

    sflux = zero(T)
    if sim.has_surface_flux
        _zero!(tϕ)
        add_surface_flux!(tϕ, sim.θ_flux_bottom, sim.θ_flux_top)
        sflux = _dot_interior(ϕ, tϕ, m)
        _accumulate!(aϕ, one(T), tϕ, m)
    end

    total = _dot_interior(ϕ, aϕ, m)
    var = 0.5 * _dot_interior(ϕ, ϕ, m)
    return (variance = var, advection = adv, diffusion = diff, sgs = sgs,
            surface_flux = sflux, total = total,
            residual = total - (adv + diff + sgs + sflux))
end

"""
    enstrophy(sim) -> Float64
    enstrophy(u, v, w) -> Float64

Total enstrophy `½∫|ω|² dV` with `ω = ∇×u` evaluated on the staggered grid.

On the Arakawa-C grid the vorticity components live on cell *edges*, each
built from the two differences that need no interpolation at all:

  * `ω_x = ∂w/∂y − ∂v/∂z` on the x-edge `(centre_x, face_y, face_z)`
  * `ω_y = ∂u/∂z − ∂w/∂x` on the y-edge `(face_x, centre_y, face_z)`
  * `ω_z = ∂v/∂x − ∂u/∂y` on the z-edge `(face_x, face_y, centre_z)`

The sum runs over one edge per interior cell, which tiles the periodic
domain exactly. `enstrophy(sim)` refills the velocity halos first.
"""
function enstrophy(sim::BoussinesqSimulation)
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    return enstrophy(sim.u, sim.v, sim.w)
end

function enstrophy(u::Field{FaceX}, v::Field{FaceY}, w::Field{FaceZ})
    g = mesh(u)
    idx = 1/g.dx; idy = 1/g.dy; idz = 1/g.dz
    ii, jj, kk = interior(g)
    ud = u.data; vd = v.data; wd = w.data
    s = zero(eltype(u))
    @inbounds for k in kk, j in jj, i in ii
        ωx = (wd[i,j,k] - wd[i,j-1,k])*idy - (vd[i,j,k] - vd[i,j,k-1])*idz
        ωy = (ud[i,j,k] - ud[i,j,k-1])*idz - (wd[i,j,k] - wd[i-1,j,k])*idx
        ωz = (vd[i,j,k] - vd[i-1,j,k])*idx - (ud[i,j,k] - ud[i,j-1,k])*idy
        s += ωx*ωx + ωy*ωy + ωz*ωz
    end
    return 0.5 * s * (g.dx*g.dy*g.dz)
end

# =========================================================================
# 10. Uniform scheme selection
# =========================================================================

function _reconstruction(name::Symbol; C_T::Real = 1e-5, p::Integer = 2)
    name === :centered  && return Centered2()
    name === :weno5     && return WENOJS(5)
    name === :weno_js5  && return WENOJS(5)
    name === :weno_z5   && return WENOZ(5, p)
    name === :weno_z7   && return WENOZ(7, p)
    name === :teno5     && return TENO(5; C_T = C_T)
    name === :teno8     && return TENO(8; C_T = C_T)
    throw(ArgumentError("unknown reconstruction $(repr(name)); expected one of " *
                        ":centered, :weno5, :weno_js5, :weno_z5, :weno_z7, :teno5, :teno8"))
end

"""
    AdvectionScheme

Callable wrapper returned by [`advection_scheme`](@ref): invoke it exactly
like the underlying operator, `scheme(tend, u, v, w, ϕ)`. Carries the
scheme's `name` and its minimum halo width (`required_halo(scheme)`).
"""
struct AdvectionScheme{F}
    name::Symbol
    halo::Int
    f::F
end

(s::AdvectionScheme)(tend, u, v, w, ϕ) = s.f(tend, u, v, w, ϕ)
required_halo(s::AdvectionScheme) = s.halo
Base.show(io::IO, s::AdvectionScheme) =
    print(io, "AdvectionScheme(:", s.name, ", halo ≥ ", s.halo, ")")

"""
    advection_scheme(name::Symbol; C_T = 1e-5, p = 2) -> AdvectionScheme

Uniform selector for the scalar advection operators, so that a simulation
can switch scheme from configuration:

| `name` | operator | halo |
| :----- | :------- | :--- |
| `:centered` | [`advect_centered!`](@ref) — 2nd-order flux form | 1 |
| `:upwind`   | 1st-order upwind flux form (monotone, diffusive) | 1 |
| `:weno5`    | `advect_weno5!` — the classic WENO-5 already in `Operators` | 3 |
| `:weno_js5` | corrected classic WENO-JS 5 reconstruction (see the note on `FiniteVolume._weno5_left`) | 3 |
| `:weno_z5`  | [`advect_weno_z!`](@ref) order 5 | 3 |
| `:weno_z7`  | [`advect_weno_z!`](@ref) order 7 | 4 |
| `:teno5`    | [`advect_teno!`](@ref) order 5 | 3 |
| `:teno8`    | [`advect_teno!`](@ref) order 8 | 5 |
| `:variance_conserving` | [`advect_variance_conserving!`](@ref) | 1 |

```julia
advect! = advection_scheme(:weno_z7)
advect!(sim.tθ, sim.u, sim.v, sim.w, sim.θ)
```

Check `required_halo(scheme)` against the mesh halo before use — the
kernels also error out clearly on their own.
"""
function advection_scheme(name::Symbol; C_T::Real = 1e-5, p::Integer = 2)
    if name === :centered
        return AdvectionScheme(name, 1, advect_centered!)
    elseif name === :upwind
        return AdvectionScheme(name, 1,
            (t, u, v, w, ϕ) -> _advect_reconstructed!(t, u, v, w, ϕ, _UpwindFirst(), mesh(ϕ)))
    elseif name === :weno5
        return AdvectionScheme(name, 3, advect_weno5!)
    elseif name === :weno_js5
        r = WENOJS(5)
        return AdvectionScheme(name, required_halo(r),
            (t, u, v, w, ϕ) -> _advect_reconstructed!(t, u, v, w, ϕ, r, mesh(ϕ)))
    elseif name === :weno_z5 || name === :weno_z7
        r = WENOZ(name === :weno_z5 ? 5 : 7, p)
        return AdvectionScheme(name, required_halo(r),
            (t, u, v, w, ϕ) -> _advect_reconstructed!(t, u, v, w, ϕ, r, mesh(ϕ)))
    elseif name === :teno5 || name === :teno8
        r = TENO(name === :teno5 ? 5 : 8; C_T = C_T)
        return AdvectionScheme(name, required_halo(r),
            (t, u, v, w, ϕ) -> _advect_reconstructed!(t, u, v, w, ϕ, r, mesh(ϕ)))
    elseif name === :variance_conserving
        return AdvectionScheme(name, 1, advect_variance_conserving!)
    else
        throw(ArgumentError("unknown advection scheme $(repr(name)); expected one of " *
                            ":centered, :upwind, :weno5, :weno_js5, :weno_z5, :weno_z7, " *
                            ":teno5, :teno8, :variance_conserving"))
    end
end

# 1st-order upwind as a reconstruction descriptor (completes the family;
# `stencil_*` gathers the two cells straddling the interface).
struct _UpwindFirst <: AbstractReconstruction end
@inline stencil_left(::_UpwindFirst)  = (Val(0), Val(1))
@inline stencil_right(::_UpwindFirst) = (Val(1), Val(1))
@inline reconstruct_left(::_UpwindFirst, t::NTuple{1})  = t[1]
@inline reconstruct_right(::_UpwindFirst, t::NTuple{1}) = t[1]
required_halo(::_UpwindFirst) = 1
scheme_name(::_UpwindFirst) = :upwind

# =========================================================================
# N-open-c (#260) — the advective flux of a scalar through the boundary
# =========================================================================

# Reconstruction descriptor of a flux-form scheme name.
_flux_reconstruction(r::AbstractReconstruction; kwargs...) = r
function _flux_reconstruction(name::Symbol; C_T::Real = 1e-5, p::Integer = 2)
    name === :centered && return Centered2()
    name === :upwind   && return _UpwindFirst()
    name === :weno_js5 && return WENOJS(5)
    name === :weno_z5  && return WENOZ(5, p)
    name === :weno_z7  && return WENOZ(7, p)
    name === :teno5    && return TENO(5; C_T = C_T)
    name === :teno8    && return TENO(8; C_T = C_T)
    throw(ArgumentError("boundary_scalar_flux: no flux-form face reconstruction for " *
                        "scheme $(repr(name)); expected one of :centered, :upwind, " *
                        ":weno_js5, :weno_z5, :weno_z7, :teno5, :teno8"))
end

@inline _wflux(::Nothing, k) = 1
@inline _wflux(W::AbstractVector, k) = @inbounds W[k]

"""
    advective_face_flux(r, ϕd, veld, ax, i, j, k) -> F

The advective flux `u_n ϕ_f` of a cell-centred scalar through **one**
face: the face of storage index `(i, j, k)` normal to axis `ax` (`1, 2,
3` for x, y, z), i.e. the face between the cells `(i-1, j, k)` and
`(i, j, k)` along `ax`. `veld` is the storage array of the face-normal
velocity (`u.data`, `v.data` or `w.data`), `ϕd` that of the scalar
(halos filled), `r` a reconstruction descriptor (`Centered2()` is the
Boussinesq core's `advect_centered!`; see [`boundary_scalar_flux`](@ref)
for the scheme names), upwinded on the sign of `u_n`.

This is the face flux every flux-form operator of this module uses, and
the one source of [`boundary_scalar_flux`](@ref) (its boundary faces)
and of the two-way nest's flux registers (its interface faces, N-nest-e
#263).
"""
@inline function advective_face_flux(r, ϕd::AbstractArray, veld::AbstractArray,
                                     ax::Int, i::Int, j::Int, k::Int)
    @inbounds vel = veld[i, j, k]
    ax == 1 && return vel * _face_x(r, ϕd, i - 1, j, k, vel)
    ax == 2 && return vel * _face_y(r, ϕd, i, j - 1, k, vel)
    return vel * _face_z(r, ϕd, i, j, k - 1, vel)
end

"""
    boundary_scalar_flux(sim, ϕ; scheme = :centered, bcs = sim.θ_bcs,
                         weights = nothing, fill = true)
        -> (west, east, south, north, bottom, top, total)

The advective flux of the cell-centred scalar `ϕ` **out of** the domain
through each boundary face, `∮ ϕ u·n dA` (units `[ϕ] m³ s⁻¹`; with
`weights = (Wc, Wf)` the weighted flux `∮ W ϕ u·n dA` of a sound-proof
core, `Wc[k]` on the lateral faces, `Wf[k]` on the bottom and top).

The flux is formed from **the same face fluxes the advection scheme
uses** — the face velocity of `sim.u, sim.v, sim.w` times the face value
of `ϕ` reconstructed by `scheme` (`:centered` = `advect_centered!`, the
Boussinesq core's θ and qv advection; `:upwind`, `:weno_js5`,
`:weno_z5`, `:weno_z7`, `:teno5`, `:teno8` = the reconstructions of
[`advect_weno_z!`](@ref), [`advect_teno!`](@ref) and
[`advection_scheme`](@ref), or any reconstruction descriptor), upwinded on
the sign of the face velocity. Because the scheme is in flux form, the
interior fluxes telescope and

    ∫ϕ dV (after a forward-Euler step) − ∫ϕ dV (before) = −Δt · total

to round-off, for every scheme: this is the term that closes a scalar or
energy budget of an open (nested) domain. With `fill = true` (default)
the halos of `u, v, w` (with `sim.bcs`) and of `ϕ` (with `bcs`) are
filled first, from the open-boundary buffers as the step itself fills
them, so the result is the flux the next step uses. Faces of a periodic
axis carry no flux (`0`); wall faces carry `0` because their normal
velocity is zero.

A mesh whose halo along an axis with an [`AbstractOpenBoundary`](@ref)
face is shallower than `required_halo(scheme)` throws an `ArgumentError`.

This is the only boundary-flux function of the code base: the
nesting budgets (N-nest-d #261, N-nest-e #263), the open-boundary Berlin
option (N-open-d #262) and the nested Berlin ledger (N-berlin-nest #265)
use it.
"""
function boundary_scalar_flux(sim, ϕ::Field{CellCenter};
                              scheme = :centered, bcs::FieldBoundaryConditions = sim.θ_bcs,
                              weights = nothing, fill::Bool = true)
    r = _flux_reconstruction(scheme)
    g = mesh(ϕ)
    vb = sim.bcs
    H = required_halo(r)
    for (side, face, Hs) in ((:west, vb.west, g.Hx), (:east, vb.east, g.Hx),
                             (:south, vb.south, g.Hy), (:north, vb.north, g.Hy),
                             (:top, vb.top, g.Hz))
        face isa AbstractOpenBoundary && Hs < H &&
            throw(ArgumentError("boundary_scalar_flux: the open face :$side has a halo of " *
                                "$Hs layers, shallower than required_halo = $H of " *
                                "$(scheme_name(r))"))
    end
    _check_halo(r, g)
    if fill
        fill_halos!(sim.u, vb); fill_halos!(sim.v, vb); fill_halos!(sim.w, vb)
        fill_halos!(ϕ, bcs)
    end
    Wc, Wf = weights === nothing ? (nothing, nothing) : weights
    T = eltype(ϕ.data)
    ϕd = ϕ.data; ud = sim.u.data; vd = sim.v.data; wd = sim.w.data
    ii, jj, kk = interior(g)
    Ax = g.dy * g.dz; Ay = g.dx * g.dz; Az = g.dx * g.dy
    fW = fE = fS = fN = fB = fT = zero(T)
    if !(vb.west isa Periodic)
        iw = first(ii); ie = last(ii) + 1
        @inbounds for k in kk, j in jj
            w = _wflux(Wc, k) * Ax
            fW -= w * advective_face_flux(r, ϕd, ud, 1, iw, j, k)
            fE += w * advective_face_flux(r, ϕd, ud, 1, ie, j, k)
        end
    end
    if !(vb.south isa Periodic)
        js = first(jj); jn = last(jj) + 1
        @inbounds for k in kk, i in ii
            w = _wflux(Wc, k) * Ay
            fS -= w * advective_face_flux(r, ϕd, vd, 2, i, js, k)
            fN += w * advective_face_flux(r, ϕd, vd, 2, i, jn, k)
        end
    end
    if !(vb.bottom isa Periodic)
        kb = first(kk); kt = last(kk) + 1
        wb_ = _wflux(Wf, kb) * Az; wt_ = _wflux(Wf, kt) * Az
        @inbounds for j in jj, i in ii
            fB -= wb_ * advective_face_flux(r, ϕd, wd, 3, i, j, kb)
            fT += wt_ * advective_face_flux(r, ϕd, wd, 3, i, j, kt)
        end
    end
    return (west = fW, east = fE, south = fS, north = fN, bottom = fB, top = fT,
            total = fW + fE + fS + fN + fB + fT)
end

end # module
