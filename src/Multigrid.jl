module Multigrid

# Phase 29 — Geometric multigrid pressure solver.
#
# Why this exists: `CGPoissonSolver` iteration counts scale like O(N) for
# the Poisson operator because the condition number grows as the grid is
# refined. Multigrid is the standard cure — its iteration count is
# *independent of grid size*, which is the property this module
# validates.
#
# Design:
#   * Cell-centred geometric multigrid with a V-cycle (and optional
#     W-cycle and full-multigrid start).
#   * Red-black Gauss-Seidel smoother — better smoothing factor than
#     damped Jacobi and still trivially parallel per colour.
#   * Full-weighting (8-point average) restriction, trilinear
#     prolongation. These are the correct transfer-operator pair for a
#     cell-centred discretisation: their product is the identity on
#     constants and they are adjoint up to the volume scaling, which is
#     what makes the coarse-grid correction consistent.
#   * Variable-coefficient support (`coeff` array) so cut-cell and
#     density-weighted (anelastic) operators can reuse the same machinery.
#   * Optional use as a *preconditioner* for CG — the most robust
#     configuration when the operator is not quite an M-matrix.
#
# Singular-system handling: with all-periodic or all-Neumann boundaries
# the Poisson operator has the constants in its null space. The solver
# projects the mean out of the right-hand side, out of the solution at
# every level, and out of the residual before restriction. Without all
# three the coarse-grid solve drifts and the V-cycle stalls.
#
# References:
#   Brandt (1977) Math. Comp. 31, 333 — multi-level adaptive solutions.
#   Briggs, Henson & McCormick (2000) — A Multigrid Tutorial, 2nd ed.
#   Trottenberg, Oosterlee & Schüller (2001) — Multigrid.

using ..Grids
using ..Grids: RegularGrid, interior, CellCenter
using ..Fields
using ..Fields: Field, interior_view
using ..PressureSolvers: AbstractPressureSolver, solve!
using ..RunLog: current_domain   # structured events (LOG-a, #333)
import ..PressureSolvers: solve!, poisson_axis_bcs, solver_stats
using ..StretchedGrids: StretchedGrid, cell_thickness, center_spacing
using ..Backends: pcg!, PCGWorkspace   # the one CG driver (KA-g, #300)
using ..Backends: backend, launch!, plane_partials!, plane_reduce!, ordered_sum   # KA-i (#303)
using ..PressureSolvers: fill_unit_halo!

export MultigridPoissonSolver, MGPreconditionedCG,
       mg_vcycle!, mg_solve!, mg_residual_norm,
       mg_level_count, mg_iteration_history

# =========================================================================
# BC handling
# =========================================================================

# Per-axis boundary kind. `:periodic` wraps; `:neumann` mirrors (the
# homogeneous-Neumann condition appropriate for a pressure-projection
# solve against impermeable walls).
const AxisBC = Symbol

@inline function _valid_bc(bc::Symbol)
    bc === :periodic || bc === :neumann ||
        error("Multigrid: unsupported BC $bc (use :periodic or :neumann)")
    return bc
end

"""
    _fill_halo!(φ, bcx, bcy, bcz)

Fill the single-cell halo of a level array according to the per-axis
boundary conditions. `φ` has size `(nx+2, ny+2, nz+2)`; the interior is
`2:nx+1` in each direction. The per-axis halo kernels of
`BoundaryConditions` (KA-f, #297; one source with `fill_halos!`):
`:periodic` wraps, `:neumann` mirrors, x then y then z.
"""
_fill_halo!(φ::Array{T,3}, bcx::Symbol, bcy::Symbol, bcz::Symbol) where {T} =
    fill_unit_halo!(φ, bcx, bcy, bcz)   # one source with the CG solvers' halo (KA-g, #300)

"True when the operator is singular, i.e. the constants are in its null space."
_is_singular(bcx, bcy, bcz) =
    (bcx === :periodic || bcx === :neumann) &&
    (bcy === :periodic || bcy === :neumann) &&
    (bcz === :periodic || bcz === :neumann)

# =========================================================================
# Level container
# =========================================================================

"""
    MGLevel{T}

One level of the multigrid hierarchy. Arrays carry a single-cell halo so
the 7-point stencil can be applied without branching at the boundary.

Fields `cx`, `cy`, `cz` hold the *face* coefficients of a
variable-coefficient operator `∇·(c ∇φ)`; for the constant-coefficient
Laplacian they are all one and the fast path is taken. `part` holds the
per-plane partial sums of the level's canonical reductions (one entry per
level `k`; KA-i, #303), so a solve allocates nothing.
"""
struct MGLevel{T}
    nx::Int; ny::Int; nz::Int
    dx::T;  dy::T;  dz::T
    φ::Array{T,3}
    b::Array{T,3}
    r::Array{T,3}
    # variable-coefficient face arrays (size (nx+1, ny, nz) etc. padded to interior+1)
    cx::Union{Nothing, Array{T,3}}
    cy::Union{Nothing, Array{T,3}}
    cz::Union{Nothing, Array{T,3}}
    part::Vector{T}
end

function MGLevel(::Type{T}, nx, ny, nz, dx, dy, dz;
                 cx = nothing, cy = nothing, cz = nothing) where {T}
    z3(a, b, c) = zeros(T, a, b, c)
    return MGLevel{T}(nx, ny, nz, T(dx), T(dy), T(dz),
                      z3(nx+2, ny+2, nz+2),
                      z3(nx+2, ny+2, nz+2),
                      z3(nx+2, ny+2, nz+2),
                      cx, cy, cz, zeros(T, nz))
end

_has_coeff(L::MGLevel) = L.cx !== nothing

# Storage ranges of a level's interior, and the ranges of an interior-sized array.
_interior(L::MGLevel) = (2:L.nx+1, 2:L.ny+1, 2:L.nz+1)
_compact(L::MGLevel) = (1:L.nx, 1:L.ny, 1:L.nz)
_ncells(L::MGLevel) = L.nx * L.ny * L.nz

# =========================================================================
# Kernels (KA-i, #303)
# =========================================================================
#
# Every loop nest of a solve is a `Backends.launch!` of a pointwise body
# and every sum a canonical plane reduction (`plane_partials!`,
# `plane_reduce!`; KA-a, #294), so a solve, its cycle count and its
# residual history are the same at every thread count. A launch over fewer
# points than the backend's serial threshold (`Backends.SERIAL_THRESHOLD`,
# 32 768 points) runs serially on the calling task: the coarse levels
# (below 32×32×32, e.g. 32×32×16 and coarser; the 50-sweep coarsest
# solve) never pay a task start-up. A red-black colour sweep covers half
# a level, so it is threaded from 64×64×32 up.
#
# The operator reaches the bodies as a small struct: `_ConstOp` (the
# constant-coefficient fast path) or `_VarOp` (`∇·(c∇φ)` with the face
# coefficient arrays). The arithmetic of every pointwise body is that of
# the serial loop it replaced, operation for operation (test/ka_multigrid.jl (removed)
# compares them bit for bit).

struct _ConstOp{T}
    ix2::T; iy2::T; iz2::T
end
struct _VarOp{T}
    ix2::T; iy2::T; iz2::T
    cx::Array{T,3}; cy::Array{T,3}; cz::Array{T,3}
end

# `f(op, args...)` with the level's operator, concretely typed.
@inline function _with_op(f::F, L::MGLevel{T}, args::Vararg{Any,M}) where {F, T, M}
    ix2 = one(T) / L.dx^2
    iy2 = one(T) / L.dy^2
    iz2 = one(T) / L.dz^2
    if _has_coeff(L)
        return f(_VarOp{T}(ix2, iy2, iz2, L.cx::Array{T,3}, L.cy::Array{T,3}, L.cz::Array{T,3}),
                 args...)
    else
        return f(_ConstOp{T}(ix2, iy2, iz2), args...)
    end
end

# (Aφ) at the storage point (i, j, k) of a halo-filled level array.
@inline function _Aφ(op::_ConstOp, φ, i, j, k)
    @inbounds begin
        c = φ[i, j, k]
        return (φ[i+1, j, k] - 2c + φ[i-1, j, k]) * op.ix2 +
               (φ[i, j+1, k] - 2c + φ[i, j-1, k]) * op.iy2 +
               (φ[i, j, k+1] - 2c + φ[i, j, k-1]) * op.iz2
    end
end
@inline function _Aφ(op::_VarOp, φ, i, j, k)
    ii = i - 1; jj = j - 1; kk = k - 1     # interior-local indices
    @inbounds begin
        fxp = op.cx[ii+1, jj, kk]; fxm = op.cx[ii, jj, kk]
        fyp = op.cy[ii, jj+1, kk]; fym = op.cy[ii, jj, kk]
        fzp = op.cz[ii, jj, kk+1]; fzm = op.cz[ii, jj, kk]
        return (fxp * (φ[i+1, j, k] - φ[i, j, k]) - fxm * (φ[i, j, k] - φ[i-1, j, k])) * op.ix2 +
               (fyp * (φ[i, j+1, k] - φ[i, j, k]) - fym * (φ[i, j, k] - φ[i, j-1, k])) * op.iy2 +
               (fzp * (φ[i, j, k+1] - φ[i, j, k]) - fzm * (φ[i, j, k] - φ[i, j, k-1])) * op.iz2
    end
end

# ---- small pointwise bodies --------------------------------------------------

@inline function _shift_body!(i, j, k, a, m)
    @inbounds a[i, j, k] -= m
    return nothing
end
@inline function _zero_body!(i, j, k, a)
    @inbounds a[i, j, k] = zero(eltype(a))
    return nothing
end
# the interior-sized `src` into the interior of the level array `dst`;
# `_copyin_val` also returns the value (fused with its sum)
@inline function _copyin_val(i, j, k, dst, src)
    @inbounds v = src[i, j, k]
    @inbounds dst[i+1, j+1, k+1] = v
    return v
end
@inline function _copyin_body!(i, j, k, dst, src)
    _copyin_val(i, j, k, dst, src)
    return nothing
end
# the interior of the level array `φ` minus `m` into `out` and back into
# `φ`: the final null-space projection fused with the copy (`m = 0`: a
# plain copy, `x − 0 = x` exactly)
@inline function _copyout_body!(i, j, k, out, φ, m)
    @inbounds v = φ[i+1, j+1, k+1] - m
    @inbounds φ[i+1, j+1, k+1] = v
    @inbounds out[i, j, k] = v
    return nothing
end
# a ← a − m at one point, returning the square of the new value (a
# projection fused with the squared norm)
@inline function _shift_sq(i, j, k, a, m)
    @inbounds v = a[i, j, k] - m
    @inbounds a[i, j, k] = v
    return v * v
end

# ---- level helpers -----------------------------------------------------------

# Canonical interior sum of a level array (the plane partials go to L.part).
function _isum(L::MGLevel{T}, a::Array{T,3}) where {T}
    plane_partials!(L.part, backend(a), identity, _interior(L), a)
    return ordered_sum(L.part)
end

_mean(L::MGLevel, a::Array) = _isum(L, a) / _ncells(L)

# a ← a − m on the interior of level L
_shift!(L::MGLevel, a::Array, m) = launch!(backend(a), _shift_body!, _interior(L), a, m)

# a ← 0 everywhere, halo included
_zero!(a::Array{T,3}) where {T} =
    launch!(backend(a), _zero_body!, (1:size(a, 1), 1:size(a, 2), 1:size(a, 3)), a)

"""
    _project_mean!(L, a) -> a

Subtract the interior mean from the level array `a` of level `L` in place
(null-space projection). The sum is the canonical plane sum of
`Backends.plane_partials!` (KA-a, #294), the same at every thread count.
"""
function _project_mean!(L::MGLevel{T}, a::Array{T,3}) where {T}
    _shift!(L, a, _mean(L, a))
    return a
end

# =========================================================================
# Operator application and residual
# =========================================================================

@inline function _res_body!(i, j, k, r, φ, b, op)
    @inbounds r[i, j, k] = b[i, j, k] - _Aφ(op, φ, i, j, k)
    return nothing
end
@inline function _apply_off_body!(i, j, k, out, φ, op)
    @inbounds out[i-1, j-1, k-1] = _Aφ(op, φ, i, j, k)
    return nothing
end
_residual_launch!(op, L::MGLevel) = launch!(backend(L.r), _res_body!, _interior(L), L.r, L.φ, L.b, op)

"""
    _compute_residual!(L, bcx, bcy, bcz)

`r = b − Aφ` on the interior of level `L`: the halo fill, then one launch
(constant or variable coefficient). Returns `L.r`.
"""
function _compute_residual!(L::MGLevel{T}, bcx, bcy, bcz) where {T}
    _fill_halo!(L.φ, bcx, bcy, bcz)
    _with_op(_residual_launch!, L, L)
    return L.r
end

"L2 norm of the interior of a level array (canonical sum of squares)."
function _l2(L::MGLevel{T}, a::Array{T,3}) where {T}
    plane_partials!(L.part, backend(a), abs2, _interior(L), a)
    return sqrt(ordered_sum(L.part))
end

# =========================================================================
# Smoother — red-black Gauss-Seidel
# =========================================================================

# One colour of a red-black sweep. The launch covers half of each i-row:
# the index `ih` maps to the `ih`-th cell of the colour in row (j, k),
# i = 2ih + ((j + k + colour) & 1), so (i + j + k) & 1 == colour. For an
# even nx every row has nx/2 cells of each colour (`imax = nothing`: no
# bounds test, and the strided loop vectorises: 1.8x faster than with the
# test at 128×128×64); for an odd nx half the rows have one cell fewer
# and their last index is skipped (`imax = nx + 1`).
@inline _in_row(i, ::Nothing) = true
@inline _in_row(i, imax::Int) = i <= imax

@inline function _rbgs_body!(ih, j, k, φ, b, op::_ConstOp, diag, colour, imax)
    i = 2ih + ((j + k + colour) & 1)
    _in_row(i, imax) || return nothing
    @inbounds begin
        off = (φ[i+1, j, k] + φ[i-1, j, k]) * op.ix2 +
              (φ[i, j+1, k] + φ[i, j-1, k]) * op.iy2 +
              (φ[i, j, k+1] + φ[i, j, k-1]) * op.iz2
        φ[i, j, k] = (b[i, j, k] - off) / diag
    end
    return nothing
end
@inline function _rbgs_body!(ih, j, k, φ, b, op::_VarOp{T}, ::Nothing, colour, imax) where {T}
    i = 2ih + ((j + k + colour) & 1)
    _in_row(i, imax) || return nothing
    ii = i - 1; jj = j - 1; kk = k - 1
    @inbounds begin
        fxp = op.cx[ii+1, jj, kk]; fxm = op.cx[ii, jj, kk]
        fyp = op.cy[ii, jj+1, kk]; fym = op.cy[ii, jj, kk]
        fzp = op.cz[ii, jj, kk+1]; fzm = op.cz[ii, jj, kk]
        diag = -((fxp + fxm) * op.ix2 + (fyp + fym) * op.iy2 + (fzp + fzm) * op.iz2)
    end
    abs(diag) < eps(T) && return nothing          # fully blocked cell
    @inbounds begin
        off = (fxp * φ[i+1, j, k] + fxm * φ[i-1, j, k]) * op.ix2 +
              (fyp * φ[i, j+1, k] + fym * φ[i, j-1, k]) * op.iy2 +
              (fzp * φ[i, j, k+1] + fzm * φ[i, j, k-1]) * op.iz2
        φ[i, j, k] = (b[i, j, k] - off) / diag
    end
    return nothing
end

_rbgs_diag(op::_ConstOp) = -2 * (op.ix2 + op.iy2 + op.iz2)
_rbgs_diag(::_VarOp) = nothing

function _rbgs_sweeps!(op, L::MGLevel, bcx, bcy, bcz, nsweeps::Int)
    iseven(L.nx) ? _rbgs_sweeps!(op, L, bcx, bcy, bcz, nsweeps, nothing) :
                   _rbgs_sweeps!(op, L, bcx, bcy, bcz, nsweeps, L.nx + 1)
end
function _rbgs_sweeps!(op, L::MGLevel, bcx, bcy, bcz, nsweeps::Int, imax)
    φ, b = L.φ, L.b
    bk = backend(φ)
    diag = _rbgs_diag(op)
    half = (1:cld(L.nx, 2), 2:L.ny+1, 2:L.nz+1)
    for _ in 1:nsweeps, colour in 0:1
        _fill_halo!(φ, bcx, bcy, bcz)
        launch!(bk, _rbgs_body!, half, φ, b, op, diag, colour, imax)
    end
    return nothing
end

"""
    _smooth_rbgs!(L, bcx, bcy, bcz, nsweeps)

Red-black Gauss-Seidel relaxation. Each sweep updates the even-parity
cells then the odd-parity cells, so within a colour the updates are
independent while still getting Gauss-Seidel's superior smoothing factor
over Jacobi. Per colour: the halo fill, then one launch over that
colour's cells only, half the level (KA-i, #303). The updates of a colour
read only the other colour, so the result does not depend on the order
the cells run in and equals the serial sweep bit for bit.
"""
function _smooth_rbgs!(L::MGLevel{T}, bcx, bcy, bcz, nsweeps::Int) where {T}
    _with_op(_rbgs_sweeps!, L, L, bcx, bcy, bcz, nsweeps)
    return L
end

# =========================================================================
# Grid transfer
# =========================================================================

# Coarse storage point (I, J, K): the mean of its eight fine cells of
# `r − m` (m = 0: the plain restriction, as `x − 0 = x` exactly).
@inline function _restrict_body!(I, J, K, b, r, m, eighth)
    i = 2I - 2; j = 2J - 2; k = 2K - 2      # fine halo-offset base
    @inbounds begin
        s = (r[i,     j,     k    ] - m) + (r[i + 1, j,     k    ] - m) +
            (r[i,     j + 1, k    ] - m) + (r[i + 1, j + 1, k    ] - m) +
            (r[i,     j,     k + 1] - m) + (r[i + 1, j,     k + 1] - m) +
            (r[i,     j + 1, k + 1] - m) + (r[i + 1, j + 1, k + 1] - m)
        b[I, J, K] = s * eighth
    end
    return nothing
end

"""
    _restrict!(coarse, fine, m = 0)

Full-weighting restriction for a cell-centred grid: each coarse cell
takes the mean of the eight fine cells it contains. Exact on constants.
Operates on `fine.r − m` → `coarse.b`, one launch in which each coarse
cell writes itself. `m` is the mean of `fine.r` when the null space is
projected out: subtracting it inside the restriction gives the bits of a
projection of `fine.r` followed by the restriction, without the sweep
that writes the projected `fine.r` (KA-i, #303).
"""
function _restrict!(coarse::MGLevel{T}, fine::MGLevel{T}, m::T = zero(T)) where {T}
    launch!(backend(coarse.b), _restrict_body!, _interior(coarse), coarse.b, fine.r, m, one(T) / 8)
    return coarse
end

# The trilinear correction at the fine storage point (is, js, ks). The
# eight products w = wi·wj·wk are the same at every point (`_prolong_w`);
# they are added in the order of the loop this replaced (dk, then dj,
# then di; near before far), so the bits are those of that loop.
@inline function _prolong_body!(is, js, ks, φ, Φ, w::NTuple{8,T}) where {T}
    i = is - 1; j = js - 1; k = ks - 1
    # halo-offset coarse index containing this fine cell (cld(i, 2) + 1)
    # and the far neighbour (one coarse cell towards the fine cell)
    Ih = (i + 1) >> 1 + 1; Ix = Ih + ifelse(isodd(i), -1, 1)
    Jh = (j + 1) >> 1 + 1; Jy = Jh + ifelse(isodd(j), -1, 1)
    Kh = (k + 1) >> 1 + 1; Kz = Kh + ifelse(isodd(k), -1, 1)
    @inbounds begin
        acc = zero(T)
        acc += w[1] * Φ[Ih, Jh, Kh]
        acc += w[2] * Φ[Ix, Jh, Kh]
        acc += w[3] * Φ[Ih, Jy, Kh]
        acc += w[4] * Φ[Ix, Jy, Kh]
        acc += w[5] * Φ[Ih, Jh, Kz]
        acc += w[6] * Φ[Ix, Jh, Kz]
        acc += w[7] * Φ[Ih, Jy, Kz]
        acc += w[8] * Φ[Ix, Jy, Kz]
        φ[is, js, ks] += acc
    end
    return nothing
end

# wi * wj * wk for (dk, dj, di) in the loop order, near (3/4) before far (1/4)
function _prolong_w(::Type{T}) where {T}
    n = T(3) / 4
    f = T(1) / 4
    return (n * n * n, f * n * n, n * f * n, f * f * n,
            n * n * f, f * n * f, n * f * f, f * f * f)
end

"""
    _prolong_add!(fine, coarse, bcx, bcy, bcz)

Trilinear prolongation of the coarse correction, added onto the fine
solution. For a cell-centred grid the fine cell centre sits a quarter of
a coarse cell from the coarse centre, giving the separable (3/4, 1/4)
weights in each direction. The coarse halo fill, then one launch in which
each fine cell writes itself.
"""
function _prolong_add!(fine::MGLevel{T}, coarse::MGLevel{T}, bcx, bcy, bcz) where {T}
    _fill_halo!(coarse.φ, bcx, bcy, bcz)
    launch!(backend(fine.φ), _prolong_body!, _interior(fine), fine.φ, coarse.φ, _prolong_w(T))
    return fine
end

"""
    _restrict_coeff(cx, cy, cz, cnx, cny, cnz)

Coarsen variable face coefficients by averaging the four fine faces that
tile each coarse face. Keeps the coarse operator consistent with the fine
one (a "Galerkin-lite" coarsening) — important for cut-cell and
density-weighted problems where a naive re-evaluation on the coarse grid
would lose the fine-scale blocking information.
"""
function _restrict_coeff(cx::Array{T,3}, cy::Array{T,3}, cz::Array{T,3},
                         cnx::Int, cny::Int, cnz::Int) where {T}
    Cx = zeros(T, cnx + 1, cny, cnz)
    Cy = zeros(T, cnx, cny + 1, cnz)
    Cz = zeros(T, cnx, cny, cnz + 1)
    quarter = one(T) / 4
    @inbounds for K in 1:cnz, J in 1:cny, I in 1:cnx+1
        i = 2I - 1
        s = zero(T)
        for dk in 0:1, dj in 0:1
            s += cx[min(i, size(cx,1)), min(2J-1+dj, size(cx,2)), min(2K-1+dk, size(cx,3))]
        end
        Cx[I, J, K] = s * quarter
    end
    @inbounds for K in 1:cnz, J in 1:cny+1, I in 1:cnx
        j = 2J - 1
        s = zero(T)
        for dk in 0:1, di in 0:1
            s += cy[min(2I-1+di, size(cy,1)), min(j, size(cy,2)), min(2K-1+dk, size(cy,3))]
        end
        Cy[I, J, K] = s * quarter
    end
    @inbounds for K in 1:cnz+1, J in 1:cny, I in 1:cnx
        k = 2K - 1
        s = zero(T)
        for dj in 0:1, di in 0:1
            s += cz[min(2I-1+di, size(cz,1)), min(2J-1+dj, size(cz,2)), min(k, size(cz,3))]
        end
        Cz[I, J, K] = s * quarter
    end
    return Cx, Cy, Cz
end

# =========================================================================
# Solver
# =========================================================================

"""
    MultigridPoissonSolver(mesh; kwargs...)

Geometric-multigrid Poisson solver. Iteration count is independent of
grid size, which is the reason to prefer it over `CGPoissonSolver` on
large grids, stretched grids, and cut-cell geometry.

Keyword arguments:

| kwarg | default | meaning |
| :---- | :------ | :------ |
| `bc_x`, `bc_y`, `bc_z` | `:periodic`, `:periodic`, `:neumann` | per-axis BC |
| `tol` | `1e-10` | relative residual tolerance |
| `maxcycles` | `50` | cycle cap |
| `nu1` | `2` | pre-smoothing sweeps |
| `nu2` | `2` | post-smoothing sweeps |
| `cycle` | `:V` | `:V` or `:W` |
| `fmg` | `false` | start from a full-multigrid (nested-iteration) initial guess |
| `min_size` | `4` | stop coarsening when any dimension reaches this |
| `coeff_x/y/z` | `nothing` | face coefficients for `∇·(c∇φ)` |
| `rhs_scale` | `nothing` | per-level factor `s_k` applied to the source (`(s_k b)`): solves `s_k (Lφ)_k = s_k b_k` |

`MultigridPoissonSolver(g::StretchedGrid; kwargs...)` builds the
stretched-z operator from these (see below).

The mesh dimensions do not need to be powers of two — coarsening stops
on any axis that becomes odd, so the hierarchy degrades gracefully
(semi-coarsening) rather than failing.
"""
mutable struct MultigridPoissonSolver{T} <: AbstractPressureSolver
    levels::Vector{MGLevel{T}}
    bc_x::Symbol
    bc_y::Symbol
    bc_z::Symbol
    tol::T
    maxcycles::Int
    nu1::Int
    nu2::Int
    cycle::Symbol
    fmg::Bool
    singular::Bool
    rhs_scale::Union{Nothing, Vector{T}}   # per-level source factor (stretched z)
    # diagnostics
    last_cycles::Base.RefValue{Int}
    last_residual::Base.RefValue{T}
    history::Vector{T}
    rhs_buf::Array{T,3}           # interior-sized scaled source of solve! (no allocation per solve, #295)
end

function MultigridPoissonSolver(mesh::RegularGrid{T};
                                bc_x::Symbol = :periodic,
                                bc_y::Symbol = :periodic,
                                bc_z::Symbol = :neumann,
                                tol::Real = 1e-10,
                                maxcycles::Int = 50,
                                nu1::Int = 2,
                                nu2::Int = 2,
                                cycle::Symbol = :V,
                                fmg::Bool = false,
                                min_size::Int = 4,
                                coeff_x = nothing,
                                coeff_y = nothing,
                                coeff_z = nothing,
                                rhs_scale = nothing) where {T}
    _valid_bc(bc_x); _valid_bc(bc_y); _valid_bc(bc_z)
    rhs_scale === nothing || length(rhs_scale) == mesh.Nz ||
        throw(DimensionMismatch("rhs_scale needs one entry per level ($(mesh.Nz)), got $(length(rhs_scale))"))
    cycle === :V || cycle === :W ||
        error("Multigrid: cycle must be :V or :W")

    levels = MGLevel{T}[]
    nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    cx, cy, cz = coeff_x, coeff_y, coeff_z
    push!(levels, MGLevel(T, nx, ny, nz, dx, dy, dz; cx = cx, cy = cy, cz = cz))

    while true
        # coarsen only the axes that stay even and above min_size
        can_x = iseven(nx) && nx ÷ 2 >= min_size
        can_y = iseven(ny) && ny ÷ 2 >= min_size
        can_z = iseven(nz) && nz ÷ 2 >= min_size
        (can_x && can_y && can_z) || break      # require full coarsening for the
                                                # 8-point transfer operators
        nx ÷= 2; ny ÷= 2; nz ÷= 2
        dx *= 2; dy *= 2; dz *= 2
        if cx !== nothing
            cx, cy, cz = _restrict_coeff(cx, cy, cz, nx, ny, nz)
        end
        push!(levels, MGLevel(T, nx, ny, nz, dx, dy, dz; cx = cx, cy = cy, cz = cz))
    end

    singular = _is_singular(bc_x, bc_y, bc_z)
    return MultigridPoissonSolver{T}(levels, bc_x, bc_y, bc_z,
                                     T(tol), maxcycles, nu1, nu2, cycle, fmg,
                                     singular,
                                     rhs_scale === nothing ? nothing : collect(T, rhs_scale),
                                     Ref(0), Ref(zero(T)), T[],
                                     zeros(T, mesh.Nx, mesh.Ny, mesh.Nz))
end

poisson_axis_bcs(s::MultigridPoissonSolver) = (bc_x = s.bc_x, bc_y = s.bc_y, bc_z = s.bc_z)

"""
    MultigridPoissonSolver(g::StretchedGrid; kwargs...)
    MGPreconditionedCG(g::StretchedGrid; kwargs...)

The multigrid solvers for the conservative stretched-z Poisson operator of
a [`StretchedGrid`](@ref Aeolus.StretchedGrids.StretchedGrid) (the
divergence of the gradient of its `∂z` stencils, #151). Multiplying row
`k` by `s_k = Δz_c[k]/Δ` (any reference `Δ`; here the thinnest cell) turns
it into the symmetric variable-coefficient form this module solves,

```math
\\frac{s_k}{Δx^2}δ_x^2φ + \\frac{s_k}{Δy^2}δ_y^2φ
 + \\frac{1}{Δ^2}\\Big[c_{k+1}(φ_{k+1}-φ_k) - c_k(φ_k-φ_{k-1})\\Big] = s_k b_k,
\\qquad c_k = \\frac{Δ}{Δz_f[k]},
```

i.e. `coeff_x = coeff_y = s_k`, `coeff_z = c_k` and `rhs_scale = s_k` on a
uniform grid of spacing `Δ`. Metrics from `cell_thickness`/`center_spacing`.
Keyword arguments are those of the `RegularGrid` constructors (BCs,
tolerance, cycles).

Convergence needs a hierarchy: coarsening requires all three dimensions
to halve, so a level count with few factors of two (a parent grid uniform
to 500 m, then stretched: 53 levels) leaves one level and the "multigrid"
is a Gauss-Seidel smoother. The default solver of the Boussinesq core is
therefore the direct [`StretchedPoissonSolver`](@ref
Aeolus.StretchedGrids.StretchedPoissonSolver); this route is for a
variable coefficient field the transform cannot diagonalise.
"""
function MultigridPoissonSolver(g::StretchedGrid{T}; kwargs...) where {T}
    rg, cx, cy, cz, sc = _stretched_operator(g)
    return MultigridPoissonSolver(rg; coeff_x = cx, coeff_y = cy, coeff_z = cz,
                                  rhs_scale = sc, kwargs...)
end

# The equivalent uniform grid (spacing Δ = thinnest cell in z) and the
# face coefficients / row scaling of the stretched operator on it.
function _stretched_operator(g::StretchedGrid{T}) where {T}
    Nx, Ny, Nz, Hz = g.Nx, g.Ny, g.Nz, g.Hz
    dzc = T[cell_thickness(g)[Hz + k] for k in 1:Nz]
    dzf = T[center_spacing(g)[Hz + k] for k in 1:Nz+1]
    Δ = minimum(dzc)
    rg = RegularGrid(T; size = (Nx, Ny, Nz), extent = (g.Lx, g.Ly, Nz * Δ),
                     halo = (g.Hx, g.Hy, g.Hz), origin = (g.x0, g.y0, g.z0))
    sc = dzc ./ Δ
    cx = zeros(T, Nx + 1, Ny, Nz); cy = zeros(T, Nx, Ny + 1, Nz)
    cz = zeros(T, Nx, Ny, Nz + 1)
    for k in 1:Nz
        cx[:, :, k] .= sc[k]; cy[:, :, k] .= sc[k]
    end
    for k in 1:Nz+1
        cz[:, :, k] .= Δ / dzf[k]
    end
    return rg, cx, cy, cz, sc
end

# s_k b_k in place on an interior-sized array (no-op without a scale)
_scale_rhs!(b, ::Nothing) = b
function _scale_rhs!(b::AbstractArray{T,3}, sc::Vector{T}) where {T}
    @inbounds for k in axes(b, 3)
        @views b[:, :, k] .*= sc[k]
    end
    return b
end

"Number of levels in the hierarchy (1 means no coarsening was possible)."
mg_level_count(s::MultigridPoissonSolver) = length(s.levels)

"Residual norm history of the last `solve!` call, one entry per cycle."
mg_iteration_history(s::MultigridPoissonSolver) = copy(s.history)

"Relative residual after the last `solve!`."
mg_residual_norm(s::MultigridPoissonSolver) = s.last_residual[]
solver_stats(s::MultigridPoissonSolver) = (s.last_cycles[], Float64(s.last_residual[]))

# --- coarse-grid solve ---------------------------------------------------

# On the coarsest level just smooth hard: it is tiny, so many sweeps are
# cheap and this is more robust than a direct solve when the operator is
# singular or has blocked cells.
# Serial through the launch threshold: the coarsest level is far below it.
function _solve_coarsest!(s::MultigridPoissonSolver{T}, L::MGLevel{T}) where {T}
    _smooth_rbgs!(L, s.bc_x, s.bc_y, s.bc_z, 50)
    s.singular && _project_mean!(L, L.φ)
    return L
end

# --- one cycle -----------------------------------------------------------

"""
    mg_vcycle!(solver, lev)

One multigrid cycle starting at level index `lev` (1 = finest). Recursive;
does one recursive call for a V-cycle and two for a W-cycle.

For a singular operator the mean is projected out of the solution after
each smoothing and after the prolongation, out of the residual and out of
the coarse right-hand side (canonical sums: the same at every thread
count). The residual's mean is subtracted inside the restriction
instead of in a sweep over the residual (KA-i, #303; fusing the sums
into the stencil launches themselves measured slower, ka_port.md).
"""
function mg_vcycle!(s::MultigridPoissonSolver{T}, lev::Int = 1) where {T}
    L = s.levels[lev]
    if lev == length(s.levels)
        _solve_coarsest!(s, L)
        return s
    end
    C = s.levels[lev + 1]
    sing = s.singular

    # pre-smooth
    _smooth_rbgs!(L, s.bc_x, s.bc_y, s.bc_z, s.nu1)
    sing && _project_mean!(L, L.φ)

    # residual → coarse rhs (of the zero-mean residual)
    _compute_residual!(L, s.bc_x, s.bc_y, s.bc_z)
    _restrict!(C, L, sing ? _mean(L, L.r) : zero(T))
    sing && _project_mean!(C, C.b)

    # coarse solve for the correction (zero initial guess)
    _zero!(C.φ)
    nrec = s.cycle === :W ? 2 : 1
    for _ in 1:nrec
        mg_vcycle!(s, lev + 1)
    end

    # prolong and correct
    _prolong_add!(L, C, s.bc_x, s.bc_y, s.bc_z)
    sing && _project_mean!(L, L.φ)

    # post-smooth
    _smooth_rbgs!(L, s.bc_x, s.bc_y, s.bc_z, s.nu2)
    sing && _project_mean!(L, L.φ)
    return s
end

# --- full multigrid (nested iteration) ----------------------------------

function _fmg_start!(s::MultigridPoissonSolver{T}) where {T}
    nlev = length(s.levels)
    # restrict the fine rhs all the way down
    for lev in 1:nlev-1
        copyto!(s.levels[lev].r, s.levels[lev].b)
        _restrict!(s.levels[lev + 1], s.levels[lev])
        s.singular && _project_mean!(s.levels[lev + 1], s.levels[lev + 1].b)
    end
    # solve coarsest, then interpolate + one V-cycle at each finer level
    _zero!(s.levels[nlev].φ)
    _solve_coarsest!(s, s.levels[nlev])
    for lev in nlev-1:-1:1
        _zero!(s.levels[lev].φ)
        _prolong_add!(s.levels[lev], s.levels[lev + 1], s.bc_x, s.bc_y, s.bc_z)
        mg_vcycle!(s, lev)
    end
    return s
end

# --- driver --------------------------------------------------------------

# The interior-sized `rhs` into L.b; for a singular operator minus its
# mean (the sum is reduced while copying).
function _load_rhs!(L::MGLevel{T}, rhs::AbstractArray{T,3}, singular::Bool) where {T}
    bk = backend(L.b)
    if singular
        plane_reduce!(L.part, bk, _copyin_val, _compact(L), L.b, rhs)
        _shift!(L, L.b, ordered_sum(L.part) / _ncells(L))
    else
        launch!(bk, _copyin_body!, _compact(L), L.b, rhs)
    end
    return L.b
end

# ‖b − Aφ‖ on the finest level, the residual projected to zero mean for a
# singular operator: the residual sweep reduces its sum, the projection
# sweep its sum of squares.
function _residual_norm!(L::MGLevel{T}, s) where {T}
    _compute_residual!(L, s.bc_x, s.bc_y, s.bc_z)
    s.singular || return _l2(L, L.r)
    plane_reduce!(L.part, backend(L.r), _shift_sq, _interior(L), L.r, _mean(L, L.r))
    return sqrt(ordered_sum(L.part))
end

# The interior of L.φ into the interior-sized `out`; for a singular
# operator minus its mean, removed from L.φ too.
function _store_solution!(out::AbstractArray{T,3}, L::MGLevel{T}, singular::Bool) where {T}
    m = singular ? _mean(L, L.φ) : zero(T)
    launch!(backend(L.φ), _copyout_body!, _compact(L), out, L.φ, m)
    return out
end

"""
    mg_solve!(solver, φ_out, rhs) -> NamedTuple

Solve `Aφ = rhs` on the finest level. `φ_out` and `rhs` are interior-sized
`(Nx, Ny, Nz)` arrays. Returns `(; cycles, residual, converged)`.
"""
function mg_solve!(s::MultigridPoissonSolver{T},
                   φ_out::AbstractArray{T,3},
                   rhs::AbstractArray{T,3}) where {T}
    L = s.levels[1]
    nx, ny, nz = L.nx, L.ny, L.nz
    size(rhs) == (nx, ny, nz) ||
        error("mg_solve!: rhs size $(size(rhs)) ≠ ($nx, $ny, $nz)")

    _load_rhs!(L, rhs, s.singular)
    _zero!(L.φ)
    empty!(s.history)

    b_norm = _l2(L, L.b)
    if b_norm == 0
        fill!(φ_out, zero(T))
        s.last_cycles[] = 0
        s.last_residual[] = zero(T)
        return (; cycles = 0, residual = zero(T), converged = true)
    end

    s.fmg && _fmg_start!(s)

    converged = false
    rel = one(T)
    ncyc = 0
    for cyc in 1:s.maxcycles
        mg_vcycle!(s, 1)
        rel = _residual_norm!(L, s) / b_norm
        push!(s.history, rel)
        ncyc = cyc
        if rel < s.tol
            converged = true
            break
        end
    end

    _store_solution!(φ_out, L, s.singular)
    s.last_cycles[] = ncyc
    s.last_residual[] = rel
    converged ||
        @warn("MultigridPoissonSolver: maxcycles=$(s.maxcycles) reached, rel. residual $rel",
              subsystem = :pressure, domain = current_domain(), solver = :multigrid,
              cycles = ncyc, relative_residual = rel, tol = s.tol)
    return (; cycles = ncyc, residual = rel, converged = converged)
end

# --- AbstractPressureSolver interface -----------------------------------

"""
    solve!(p, source, solver::MultigridPoissonSolver)

`AbstractPressureSolver` interface: solve `∇²p = source` in place.
Drop-in replacement for the FFT and CG solvers.
"""
function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                solver::MultigridPoissonSolver{T}) where {T}
    src = copyto!(solver.rhs_buf, interior_view(source))
    _scale_rhs!(src, solver.rhs_scale)
    mg_solve!(solver, interior_view(p), src)     # writes every interior cell of p
    return p
end

# =========================================================================
# Multigrid-preconditioned CG
# =========================================================================

"""
    MGPreconditionedCG(mesh; mg_kwargs..., tol, maxit)

Conjugate gradient with one multigrid V-cycle as the preconditioner. This
is the most robust configuration: CG supplies the optimal Krylov
acceleration and guards against a mildly non-symmetric or
poorly-coarsened multigrid operator, while the V-cycle keeps the
iteration count grid-independent.

Use this instead of a bare `MultigridPoissonSolver` when the coefficient
field is strongly varying (cut cells, large density contrast).
"""
mutable struct MGPreconditionedCG{T} <: AbstractPressureSolver
    mg::MultigridPoissonSolver{T}
    tol::T
    maxit::Int
    # CG work arrays, interior-sized
    x::Array{T,3}
    b::Array{T,3}                 # the (scaled) source
    ws::PCGWorkspace{T,Array{T,3}}   # r, z, p, Ap and the reduction partials
    last_iters::Base.RefValue{Int}
    last_residual::Base.RefValue{T}
end

poisson_axis_bcs(s::MGPreconditionedCG) = poisson_axis_bcs(s.mg)

function MGPreconditionedCG(mesh::RegularGrid{T};
                            tol::Real = 1e-10, maxit::Int = 200,
                            mg_cycles::Int = 1, kwargs...) where {T}
    mg = MultigridPoissonSolver(mesh; tol = 0.0, maxcycles = mg_cycles,
                                kwargs...)
    x = zeros(T, mesh.Nx, mesh.Ny, mesh.Nz)
    return MGPreconditionedCG{T}(mg, T(tol), maxit, x, zero(x), PCGWorkspace(x),
                                 Ref(0), Ref(zero(T)))
end

function MGPreconditionedCG(g::StretchedGrid{T}; kwargs...) where {T}
    rg, cx, cy, cz, sc = _stretched_operator(g)
    return MGPreconditionedCG(rg; coeff_x = cx, coeff_y = cy, coeff_z = cz,
                              rhs_scale = sc, kwargs...)
end

# Apply the fine-level operator to an interior array.
# (copy in, halo, one operator launch writing `out` directly; KA-i, #303)
function _apply_fine!(out::Array{T,3}, x::Array{T,3},
                      s::MultigridPoissonSolver{T}) where {T}
    L = s.levels[1]
    launch!(backend(L.φ), _copyin_body!, _compact(L), L.φ, x)
    _fill_halo!(L.φ, s.bc_x, s.bc_y, s.bc_z)
    _with_op(_apply_launch!, L, out, L)
    return out
end
_apply_launch!(op, out, L) = launch!(backend(out), _apply_off_body!, _interior(L), out, L.φ, op)

# One V-cycle as a preconditioner: z ← M⁻¹ r.
function _mg_precond!(z::Array{T,3}, r::Array{T,3},
                      s::MultigridPoissonSolver{T}) where {T}
    L = s.levels[1]
    _load_rhs!(L, r, s.singular)
    _zero!(L.φ)
    for _ in 1:max(s.maxcycles, 1)
        mg_vcycle!(s, 1)
    end
    _store_solution!(z, L, s.singular)
    return z
end

# The operator and the preconditioner as the functions `pcg!` calls.
struct _FineOp{T}
    s::MultigridPoissonSolver{T}
end
(A::_FineOp)(Ap, p) = _apply_fine!(Ap, p, A.s)
struct _VCyclePC{T}
    s::MultigridPoissonSolver{T}
end
(M::_VCyclePC)(z, r) = _mg_precond!(z, r, M.s)

"""
    solve!(p, source, solver::MGPreconditionedCG)

CG preconditioned by `mg_cycles` V-cycles, through the one CG driver
[`Aeolus.Backends.pcg!`](@ref) (KA-g, #300): canonical reductions, so a
solve and its iteration count are identical at every thread count. For a
singular operator (periodic / Neumann on every axis) the mean is removed
from the source, from every preconditioned residual and from the
solution. A breakdown (`p·Ap ≤ eps·|r·z|`) stops the iteration silently,
as before.
"""
function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                solver::MGPreconditionedCG{T}) where {T}
    s  = solver.mg
    b  = copyto!(solver.b, interior_view(source))
    _scale_rhs!(b, s.rhs_scale)
    st = pcg!(_FineOp(s), _VCyclePC(s), solver.x, b, solver.ws;
              tol = solver.tol, maxit = solver.maxit, project_mean = s.singular,
              definite = :negative)
    interior_view(p) .= solver.x
    solver.last_iters[] = st.iterations
    solver.last_residual[] = st.status === :zero_rhs ? zero(T) : st.residual / st.bnorm
    return p
end

"Iterations used by the last `solve!` of an `MGPreconditionedCG`."
mg_iteration_history(s::MGPreconditionedCG) = s.last_iters[]
mg_residual_norm(s::MGPreconditionedCG) = s.last_residual[]
solver_stats(s::MGPreconditionedCG) = (s.last_iters[], Float64(s.last_residual[]))

end # module
