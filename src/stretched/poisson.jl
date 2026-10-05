# =========================================================================
# #151 (N-grid) — the pressure solve on a StretchedGrid.
# =========================================================================
#
# Included into `module StretchedGrids`. The projection of the Boussinesq
# core solves  D G φ = ∇·u*  with the grid's own divergence D (metric dzc)
# and gradient G (metric dzf), i.e. the conservative stretched Laplacian
# `_laplacian!(::Field{CellCenter}, …, ::StretchedGrid)`:
#
#   (φ[i+1]-2φ+φ[i-1])/Δx² + (φ[j+1]-2φ+φ[j-1])/Δy²
#     + (1/Δz_c[k]) [ (φ[k+1]-φ[k])/Δz_f[k+1] - (φ[k]-φ[k-1])/Δz_f[k] ].
#
# x and y are uniform, so the operator is diagonalised horizontally by the
# discrete Fourier transform (periodic axis) or the DCT-II (Neumann axis:
# a wall or any non-periodic side), with the exact eigenvalues of the
# 3-point stencil, and what is left is one tridiagonal system in z per
# horizontal wavenumber, solved by the Thomas algorithm (Schumann & Sweet
# 1988, J. Comput. Phys. 75, 123, the FFT–tridiagonal method; Press et al.
# 2007, Numerical Recipes 3rd ed. §20.4). The solve is direct and exact to
# round-off, so the projected velocity is solenoidal to ~1e-15 relative on
# any face distribution.
#
# Why not the geometric multigrid with `coeff_z` (which also represents
# this operator, see `Multigrid.MultigridPoissonSolver(::StretchedGrid)`)?
# Its hierarchy only coarsens when all three dimensions halve, and a
# parent grid that is uniform to 500 m and then stretched has an odd or
# 2·odd level count (53 levels for 25 m to 500 m, ×1.08 to 100 m, 3 km):
# one or two levels, i.e. a smoother, not a multigrid. Measured on that
# grid (#151, one thread, random compatible source, tol 1e-12): the bare
# V-cycle stalls (relative residual 1.9e-6 after 500 cycles, 10 s, at
# 32×32×53; 1.2e-4 at 64×64×53); multigrid-preconditioned CG converges in
# 34 iterations (0.72 s) at 32×32×53 and 64 (5.5 s) at 64×64×53; this
# direct solve takes 0.4 ms and 1.2 ms. It is the default; the multigrid route stays
# available for a coefficient field (cut cells) the transform cannot
# diagonalise.

using FFTW: FFTW, plan_r2r
using LinearAlgebra: mul!
using ..PressureSolvers: AbstractPressureSolver
import ..PressureSolvers: solve!
using ..BoundaryConditions: FieldBoundaryConditions, Periodic
import ..Simulations: _default_solver

"""
    StretchedPoissonSolver(g::StretchedGrid; bc_x = :periodic, bc_y = :periodic)

Direct solver of the conservative stretched-z Poisson equation (the
divergence of the gradient of the grid's own `∂x, ∂y, ∂z` stencils),
cell-centred, with homogeneous-Neumann conditions at the bottom and top
(walls) and `bc_x`, `bc_y ∈ (:periodic, :neumann)` on the sides.

Horizontally the operator is diagonalised by the real-to-real FFT
(`:periodic`, half-complex DFT) or the DCT-II (`:neumann`), with the exact
3-point eigenvalues `-4/Δx² sin²(πm/Nx)` and `-4/Δx² sin²(πm/2Nx)`; in z it
leaves one tridiagonal system per horizontal wavenumber,

```math
\\frac{1}{\\Delta z_c[k]}\\left[\\frac{\\hat φ_{k+1}-\\hat φ_k}{\\Delta z_f[k+1]}
 - \\frac{\\hat φ_k-\\hat φ_{k-1}}{\\Delta z_f[k]}\\right] - λ_h \\hat φ_k = \\hat b_k ,
```

solved by the Thomas algorithm, whose factorisation is computed once at
construction (Schumann & Sweet 1988, *J. Comput. Phys.* 75, 123). The mean
mode (`λ_h = 0`) is singular with Neumann ends: the volume-weighted mean
of the source is removed (the compatibility condition of a closed domain)
and the column is integrated directly. The returned `φ` has zero
volume-weighted mean. Metrics come from [`cell_thickness`](@ref) and
[`center_spacing`](@ref); exact to round-off on any face distribution.

The default solver of `BoussinesqSimulation(g::StretchedGrid)` for a
non-uniform grid; periodic-in-z is not supported (a stretched vertical
coordinate has no periodic meaning).
"""
struct StretchedPoissonSolver{T, PF, PI} <: AbstractPressureSolver
    plan_fwd::PF
    plan_inv::PI
    buf::Array{T,3}          # transform buffer, interior size
    buf2::Array{T,3}
    cp::Array{T,3}           # Thomas: modified upper diagonal c'_k
    inv_den::Array{T,3}      # Thomas: 1 / (b_k - a_k c'_{k-1})
    lower::Vector{T}         # a_k: coefficient of φ_{k-1} (0 at k = 1)
    dzc::Vector{T}           # interior cell thickness
    dzf::Vector{T}           # interior face spacing, length Nz+1 (face k = bottom of cell k)
    col::Vector{T}           # the mean-mode column (scratch)
    norm::T                  # 1 / (forward × inverse transform scale)
    bc_x::Symbol
    bc_y::Symbol
end

_r2r_kinds(bc::Symbol) = bc === :periodic ? (FFTW.R2HC, FFTW.HC2R, 1) :
                         bc === :neumann  ? (FFTW.REDFT10, FFTW.REDFT01, 2) :
                         throw(ArgumentError("StretchedPoissonSolver: lateral BC must be " *
                                             ":periodic or :neumann, got :$bc"))

# Eigenvalue magnitudes 4/Δ² sin²(·) of the 3-point second difference in the
# r2r basis: half-complex index p ↔ frequency min(p, N-p) for the DFT, p for
# the DCT-II (period 2N).
function _r2r_eigen(::Type{T}, N::Int, Δ::T, bc::Symbol) where {T}
    if bc === :periodic
        return T[4 / Δ^2 * sin(π * min(p, N - p) / N)^2 for p in 0:N-1]
    else
        return T[4 / Δ^2 * sin(π * p / (2N))^2 for p in 0:N-1]
    end
end

function StretchedPoissonSolver(g::StretchedGrid{T};
                                bc_x::Symbol = :periodic,
                                bc_y::Symbol = :periodic) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    kx_f, kx_i, sx = _r2r_kinds(bc_x)
    ky_f, ky_i, sy = _r2r_kinds(bc_y)
    buf  = zeros(T, Nx, Ny, Nz)
    buf2 = zeros(T, Nx, Ny, Nz)
    plan_fwd = plan_r2r(buf, (kx_f, ky_f), (1, 2); flags = FFTW.MEASURE)
    plan_inv = plan_r2r(buf, (kx_i, ky_i), (1, 2); flags = FFTW.MEASURE)
    norm = one(T) / T(sx * Nx * sy * Ny)

    kk = g.Hz+1 : g.Hz+Nz
    dzc = T[cell_thickness(g)[k] for k in kk]
    dzf = T[center_spacing(g)[k] for k in g.Hz+1 : g.Hz+Nz+1]

    # z-operator: a_k φ_{k-1} + (-(a_k + c_k)) φ_k + c_k φ_{k+1}; Neumann ends
    # (the wall flux is zero) drop a_1 and c_Nz.
    lower = zeros(T, Nz); upper = zeros(T, Nz)
    for k in 1:Nz
        k > 1  && (lower[k] = 1 / (dzc[k] * dzf[k]))
        k < Nz && (upper[k] = 1 / (dzc[k] * dzf[k+1]))
    end

    λx = _r2r_eigen(T, Nx, T(g.dx), bc_x)
    λy = _r2r_eigen(T, Ny, T(g.dy), bc_y)
    cp = zeros(T, Nx, Ny, Nz); inv_den = zeros(T, Nx, Ny, Nz)
    @inbounds for q in 1:Ny, p in 1:Nx
        λ = λx[p] + λy[q]
        (p == 1 && q == 1) && continue          # the mean mode: integrated directly
        cprev = zero(T)
        for k in 1:Nz
            b   = -(lower[k] + upper[k]) - λ
            den = b - lower[k] * cprev
            inv_den[p, q, k] = 1 / den
            cprev = upper[k] / den
            cp[p, q, k] = cprev
        end
    end
    return StretchedPoissonSolver{T, typeof(plan_fwd), typeof(plan_inv)}(
        plan_fwd, plan_inv, buf, buf2, cp, inv_den, lower, dzc, dzf, zeros(T, Nz), norm,
        bc_x, bc_y)
end

function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                s::StretchedPoissonSolver{T}) where {T}
    m = source.mesh
    ii, jj, kk = interior(m)
    Nx, Ny, Nz = size(s.buf)
    @views s.buf .= source.data[ii, jj, kk]
    mul!(s.buf2, s.plan_fwd, s.buf)
    b = s.buf2
    cp, inv_den, lower = s.cp, s.inv_den, s.lower
    @views s.col .= b[1, 1, :]          # the mean mode, solved separately

    # forward elimination (level by level, wavenumbers inner; the mean
    # mode has inv_den = cp = 0 and is overwritten below)
    @inbounds for k in 1:Nz
        a = lower[k]
        if k == 1
            for q in 1:Ny, pp in 1:Nx
                b[pp, q, 1] *= inv_den[pp, q, 1]
            end
        else
            for q in 1:Ny, pp in 1:Nx
                b[pp, q, k] = (b[pp, q, k] - a * b[pp, q, k-1]) * inv_den[pp, q, k]
            end
        end
    end
    # back substitution
    @inbounds for k in Nz-1:-1:1, q in 1:Ny, pp in 1:Nx
        b[pp, q, k] -= cp[pp, q, k] * b[pp, q, k+1]
    end
    _mean_mode!(s)
    @views b[1, 1, :] .= s.col

    mul!(s.buf, s.plan_inv, s.buf2)
    @views p.data[ii, jj, kk] .= s.buf .* s.norm
    return p
end

# Column of the λ_h = 0 mode, in transform units (the operator is linear,
# so the horizontal-sum scaling of the transform carries through): remove
# the volume-weighted mean of the source (the compatibility condition of
# the Neumann ends), integrate the flux up from the zero bottom flux,
#   F_{k} = F_{k-1} + b_{k-1} Δz_c[k-1],   φ_k = φ_{k-1} + F_k Δz_f[k],
# then remove the volume-weighted mean of φ.
function _mean_mode!(s::StretchedPoissonSolver{T}) where {T}
    col, dzc, dzf = s.col, s.dzc, s.dzf
    Nz = length(col)
    H  = sum(dzc)
    mb = zero(T)
    @inbounds for k in 1:Nz
        mb += col[k] * dzc[k]
    end
    mb /= H
    F = zero(T); φk = zero(T)
    @inbounds begin
        prev = col[1] - mb
        col[1] = zero(T)
        for k in 2:Nz
            F  += prev * dzc[k-1]
            φk += F * dzf[k]
            prev = col[k] - mb
            col[k] = φk
        end
    end
    mφ = zero(T)
    @inbounds for k in 1:Nz
        mφ += col[k] * dzc[k]
    end
    col .-= mφ / H
    return s
end

# =========================================================================
# Default solver of the Boussinesq core on a StretchedGrid
# =========================================================================

"""
    _regular_equivalent(g::StretchedGrid) -> RegularGrid

The `RegularGrid` with the same size, extent, halo and origin; a
`:uniform` StretchedGrid *is* this grid.
"""
_regular_equivalent(g::StretchedGrid{T}) where {T} =
    RegularGrid(T; size = (g.Nx, g.Ny, g.Nz), extent = (g.Lx, g.Ly, g.Lz),
                halo = (g.Hx, g.Hy, g.Hz), origin = (g.x0, g.y0, g.z0))

_lateral_bc(a, b) = (a isa Periodic && b isa Periodic) ? :periodic : :neumann

"""
    _default_solver(g::StretchedGrid, bcs) -> AbstractPressureSolver

The pressure solver `BoussinesqSimulation(g::StretchedGrid)` uses when none
is passed. A `:uniform` grid takes the `RegularGrid` default of the same
grid (so it reproduces the RegularGrid run); a stretched grid takes the
direct [`StretchedPoissonSolver`](@ref), periodic on each side with
`Periodic` velocity BCs and Neumann otherwise. The vertical BCs must not
be periodic.
"""
function _default_solver(g::StretchedGrid, bcs::FieldBoundaryConditions)
    g.stretching === :uniform && return _default_solver(_regular_equivalent(g), bcs)
    (bcs.bottom isa Periodic || bcs.top isa Periodic) &&
        throw(ArgumentError("BoussinesqSimulation on a stretched grid needs non-periodic " *
                            "bottom and top boundaries (walls), got $(typeof(bcs.bottom)), " *
                            "$(typeof(bcs.top))"))
    return StretchedPoissonSolver(g; bc_x = _lateral_bc(bcs.west, bcs.east),
                                     bc_y = _lateral_bc(bcs.south, bcs.north))
end
