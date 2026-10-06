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
import ..PressureSolvers: solve!, poisson_axis_bcs
using ..StretchedGrids: StretchedGrid, cell_thickness, center_spacing

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
`2:nx+1` in each direction.
"""
function _fill_halo!(φ::Array{T,3}, bcx::Symbol, bcy::Symbol, bcz::Symbol) where {T}
    nx, ny, nz = size(φ) .- 2
    @inbounds @views begin      # plane copies on views: no copy of the source plane (#295)
        if bcx === :periodic
            φ[1,      :, :] .= φ[nx + 1, :, :]
            φ[nx + 2, :, :] .= φ[2,      :, :]
        else  # :neumann
            φ[1,      :, :] .= φ[2,      :, :]
            φ[nx + 2, :, :] .= φ[nx + 1, :, :]
        end
        if bcy === :periodic
            φ[:, 1,      :] .= φ[:, ny + 1, :]
            φ[:, ny + 2, :] .= φ[:, 2,      :]
        else
            φ[:, 1,      :] .= φ[:, 2,      :]
            φ[:, ny + 2, :] .= φ[:, ny + 1, :]
        end
        if bcz === :periodic
            φ[:, :, 1     ] .= φ[:, :, nz + 1]
            φ[:, :, nz + 2] .= φ[:, :, 2     ]
        else
            φ[:, :, 1     ] .= φ[:, :, 2     ]
            φ[:, :, nz + 2] .= φ[:, :, nz + 1]
        end
    end
    return φ
end

"True when the operator is singular, i.e. the constants are in its null space."
_is_singular(bcx, bcy, bcz) =
    (bcx === :periodic || bcx === :neumann) &&
    (bcy === :periodic || bcy === :neumann) &&
    (bcz === :periodic || bcz === :neumann)

"Subtract the interior mean from `a` in place (null-space projection)."
function _project_mean!(a::Array{T,3}) where {T}
    nx, ny, nz = size(a) .- 2
    s = zero(T)
    @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        s += a[i, j, k]
    end
    m = s / (nx * ny * nz)
    @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        a[i, j, k] -= m
    end
    return a
end

# =========================================================================
# Level container
# =========================================================================

"""
    MGLevel{T}

One level of the multigrid hierarchy. Arrays carry a single-cell halo so
the 7-point stencil can be applied without branching at the boundary.

Fields `cx`, `cy`, `cz` hold the *face* coefficients of a
variable-coefficient operator `∇·(c ∇φ)`; for the constant-coefficient
Laplacian they are all one and the fast path is taken.
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
end

function MGLevel(::Type{T}, nx, ny, nz, dx, dy, dz;
                 cx = nothing, cy = nothing, cz = nothing) where {T}
    z3(a, b, c) = zeros(T, a, b, c)
    return MGLevel{T}(nx, ny, nz, T(dx), T(dy), T(dz),
                      z3(nx+2, ny+2, nz+2),
                      z3(nx+2, ny+2, nz+2),
                      z3(nx+2, ny+2, nz+2),
                      cx, cy, cz)
end

_has_coeff(L::MGLevel) = L.cx !== nothing

# =========================================================================
# Operator application and residual
# =========================================================================

"""
    _apply_laplacian!(out, L, bcx, bcy, bcz)

Apply the 7-point discrete operator to `L.φ`, writing into `out`
(interior only). Constant-coefficient fast path plus a
variable-coefficient branch for `∇·(c∇φ)`.
"""
function _apply_laplacian!(out::Array{T,3}, L::MGLevel{T},
                           bcx::Symbol, bcy::Symbol, bcz::Symbol) where {T}
    _fill_halo!(L.φ, bcx, bcy, bcz)
    φ = L.φ
    nx, ny, nz = L.nx, L.ny, L.nz
    ix2 = one(T) / L.dx^2
    iy2 = one(T) / L.dy^2
    iz2 = one(T) / L.dz^2
    if !_has_coeff(L)
        @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
            c = φ[i, j, k]
            out[i, j, k] = (φ[i+1, j, k] - 2c + φ[i-1, j, k]) * ix2 +
                           (φ[i, j+1, k] - 2c + φ[i, j-1, k]) * iy2 +
                           (φ[i, j, k+1] - 2c + φ[i, j, k-1]) * iz2
        end
    else
        cx, cy, cz = L.cx, L.cy, L.cz
        @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
            ii = i - 1; jj = j - 1; kk = k - 1     # interior-local indices
            fxp = cx[ii+1, jj, kk]; fxm = cx[ii, jj, kk]
            fyp = cy[ii, jj+1, kk]; fym = cy[ii, jj, kk]
            fzp = cz[ii, jj, kk+1]; fzm = cz[ii, jj, kk]
            out[i, j, k] =
                (fxp * (φ[i+1, j, k] - φ[i, j, k]) - fxm * (φ[i, j, k] - φ[i-1, j, k])) * ix2 +
                (fyp * (φ[i, j+1, k] - φ[i, j, k]) - fym * (φ[i, j, k] - φ[i, j-1, k])) * iy2 +
                (fzp * (φ[i, j, k+1] - φ[i, j, k]) - fzm * (φ[i, j, k] - φ[i, j, k-1])) * iz2
        end
    end
    return out
end

"""
    _compute_residual!(L, bcx, bcy, bcz)

`r = b − Aφ` on the interior. Returns `L.r`.
"""
function _compute_residual!(L::MGLevel{T}, bcx, bcy, bcz) where {T}
    _apply_laplacian!(L.r, L, bcx, bcy, bcz)
    nx, ny, nz = L.nx, L.ny, L.nz
    @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        L.r[i, j, k] = L.b[i, j, k] - L.r[i, j, k]
    end
    return L.r
end

"L2 norm of the interior of an array."
function _l2(a::Array{T,3}) where {T}
    nx, ny, nz = size(a) .- 2
    s = zero(T)
    @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        s += a[i, j, k]^2
    end
    return sqrt(s)
end

# =========================================================================
# Smoother — red-black Gauss-Seidel
# =========================================================================

"""
    _smooth_rbgs!(L, bcx, bcy, bcz, nsweeps)

Red-black Gauss-Seidel relaxation. Each sweep updates the even-parity
cells then the odd-parity cells, so within a colour the updates are
independent (and hence trivially threadable) while still getting
Gauss-Seidel's superior smoothing factor over Jacobi.
"""
function _smooth_rbgs!(L::MGLevel{T}, bcx, bcy, bcz, nsweeps::Int) where {T}
    nx, ny, nz = L.nx, L.ny, L.nz
    ix2 = one(T) / L.dx^2
    iy2 = one(T) / L.dy^2
    iz2 = one(T) / L.dz^2
    φ, b = L.φ, L.b
    for _ in 1:nsweeps
        for colour in 0:1
            _fill_halo!(φ, bcx, bcy, bcz)
            if !_has_coeff(L)
                diag = -2 * (ix2 + iy2 + iz2)
                @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
                    ((i + j + k) & 1) == colour || continue
                    off = (φ[i+1, j, k] + φ[i-1, j, k]) * ix2 +
                          (φ[i, j+1, k] + φ[i, j-1, k]) * iy2 +
                          (φ[i, j, k+1] + φ[i, j, k-1]) * iz2
                    φ[i, j, k] = (b[i, j, k] - off) / diag
                end
            else
                cx, cy, cz = L.cx, L.cy, L.cz
                @inbounds for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
                    ((i + j + k) & 1) == colour || continue
                    ii = i - 1; jj = j - 1; kk = k - 1
                    fxp = cx[ii+1, jj, kk]; fxm = cx[ii, jj, kk]
                    fyp = cy[ii, jj+1, kk]; fym = cy[ii, jj, kk]
                    fzp = cz[ii, jj, kk+1]; fzm = cz[ii, jj, kk]
                    diag = -((fxp + fxm) * ix2 + (fyp + fym) * iy2 + (fzp + fzm) * iz2)
                    abs(diag) < eps(T) && continue          # fully blocked cell
                    off = (fxp * φ[i+1, j, k] + fxm * φ[i-1, j, k]) * ix2 +
                          (fyp * φ[i, j+1, k] + fym * φ[i, j-1, k]) * iy2 +
                          (fzp * φ[i, j, k+1] + fzm * φ[i, j, k-1]) * iz2
                    φ[i, j, k] = (b[i, j, k] - off) / diag
                end
            end
        end
    end
    return L
end

# =========================================================================
# Grid transfer
# =========================================================================

"""
    _restrict!(coarse, fine)

Full-weighting restriction for a cell-centred grid: each coarse cell
takes the mean of the eight fine cells it contains. Exact on constants.
Operates on `fine.r` → `coarse.b`.
"""
function _restrict!(coarse::MGLevel{T}, fine::MGLevel{T}) where {T}
    cnx, cny, cnz = coarse.nx, coarse.ny, coarse.nz
    r, b = fine.r, coarse.b
    eighth = one(T) / 8
    @inbounds for K in 1:cnz, J in 1:cny, I in 1:cnx
        i = 2I; j = 2J; k = 2K            # fine halo-offset base (2I-1 → index 2I)
        s = r[i,     j,     k    ] + r[i + 1, j,     k    ] +
            r[i,     j + 1, k    ] + r[i + 1, j + 1, k    ] +
            r[i,     j,     k + 1] + r[i + 1, j,     k + 1] +
            r[i,     j + 1, k + 1] + r[i + 1, j + 1, k + 1]
        b[I + 1, J + 1, K + 1] = s * eighth
    end
    return coarse
end

"""
    _prolong_add!(fine, coarse, bcx, bcy, bcz)

Trilinear prolongation of the coarse correction, added onto the fine
solution. For a cell-centred grid the fine cell centre sits a quarter of
a coarse cell from the coarse centre, giving the separable (3/4, 1/4)
weights in each direction.
"""
function _prolong_add!(fine::MGLevel{T}, coarse::MGLevel{T},
                       bcx, bcy, bcz) where {T}
    _fill_halo!(coarse.φ, bcx, bcy, bcz)
    fnx, fny, fnz = fine.nx, fine.ny, fine.nz
    Φ = coarse.φ
    φ = fine.φ
    w_near = T(3) / 4
    w_far  = T(1) / 4
    @inbounds for k in 1:fnz, j in 1:fny, i in 1:fnx
        # coarse index containing this fine cell, and the far-neighbour offset
        I = cld(i, 2); sx = isodd(i) ? -1 : 1
        J = cld(j, 2); sy = isodd(j) ? -1 : 1
        K = cld(k, 2); sz = isodd(k) ? -1 : 1
        Ih = I + 1; Jh = J + 1; Kh = K + 1        # halo-offset coarse indices
        acc = zero(T)
        for (dk, wk) in ((0, w_near), (sz, w_far))
            for (dj, wj) in ((0, w_near), (sy, w_far))
                for (di, wi) in ((0, w_near), (sx, w_far))
                    acc += wi * wj * wk * Φ[Ih + di, Jh + dj, Kh + dk]
                end
            end
        end
        φ[i + 1, j + 1, k + 1] += acc
    end
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

# --- coarse-grid solve ---------------------------------------------------

# On the coarsest level just smooth hard: it is tiny, so many sweeps are
# cheap and this is more robust than a direct solve when the operator is
# singular or has blocked cells.
function _solve_coarsest!(s::MultigridPoissonSolver{T}, L::MGLevel{T}) where {T}
    _smooth_rbgs!(L, s.bc_x, s.bc_y, s.bc_z, 50)
    s.singular && _project_mean!(L.φ)
    return L
end

# --- one cycle -----------------------------------------------------------

"""
    mg_vcycle!(solver, lev)

One multigrid cycle starting at level index `lev` (1 = finest). Recursive;
does one recursive call for a V-cycle and two for a W-cycle.
"""
function mg_vcycle!(s::MultigridPoissonSolver{T}, lev::Int = 1) where {T}
    L = s.levels[lev]
    if lev == length(s.levels)
        _solve_coarsest!(s, L)
        return s
    end
    C = s.levels[lev + 1]

    # pre-smooth
    _smooth_rbgs!(L, s.bc_x, s.bc_y, s.bc_z, s.nu1)
    s.singular && _project_mean!(L.φ)

    # residual → coarse rhs
    _compute_residual!(L, s.bc_x, s.bc_y, s.bc_z)
    s.singular && _project_mean!(L.r)
    _restrict!(C, L)
    s.singular && _project_mean!(C.b)

    # coarse solve for the correction (zero initial guess)
    fill!(C.φ, zero(T))
    nrec = s.cycle === :W ? 2 : 1
    for _ in 1:nrec
        mg_vcycle!(s, lev + 1)
    end

    # prolong and correct
    _prolong_add!(L, C, s.bc_x, s.bc_y, s.bc_z)
    s.singular && _project_mean!(L.φ)

    # post-smooth
    _smooth_rbgs!(L, s.bc_x, s.bc_y, s.bc_z, s.nu2)
    s.singular && _project_mean!(L.φ)
    return s
end

# --- full multigrid (nested iteration) ----------------------------------

function _fmg_start!(s::MultigridPoissonSolver{T}) where {T}
    nlev = length(s.levels)
    # restrict the fine rhs all the way down
    for lev in 1:nlev-1
        s.levels[lev].r .= s.levels[lev].b
        _restrict!(s.levels[lev + 1], s.levels[lev])
        s.singular && _project_mean!(s.levels[lev + 1].b)
    end
    # solve coarsest, then interpolate + one V-cycle at each finer level
    fill!(s.levels[nlev].φ, zero(T))
    _solve_coarsest!(s, s.levels[nlev])
    for lev in nlev-1:-1:1
        fill!(s.levels[lev].φ, zero(T))
        _prolong_add!(s.levels[lev], s.levels[lev + 1], s.bc_x, s.bc_y, s.bc_z)
        mg_vcycle!(s, lev)
    end
    return s
end

# --- driver --------------------------------------------------------------

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

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        L.b[i+1, j+1, k+1] = rhs[i, j, k]
    end
    s.singular && _project_mean!(L.b)
    fill!(L.φ, zero(T))
    empty!(s.history)

    b_norm = _l2(L.b)
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
        _compute_residual!(L, s.bc_x, s.bc_y, s.bc_z)
        s.singular && _project_mean!(L.r)
        rel = _l2(L.r) / b_norm
        push!(s.history, rel)
        ncyc = cyc
        if rel < s.tol
            converged = true
            break
        end
    end

    s.singular && _project_mean!(L.φ)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        φ_out[i, j, k] = L.φ[i+1, j+1, k+1]
    end
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
    r::Array{T,3}
    z::Array{T,3}
    pdir::Array{T,3}
    Ap::Array{T,3}
    x::Array{T,3}
    b::Array{T,3}                 # the (scaled, zero-mean) source
    w::Array{T,3}                 # scratch of the dot products
    last_iters::Base.RefValue{Int}
    last_residual::Base.RefValue{T}
end

poisson_axis_bcs(s::MGPreconditionedCG) = poisson_axis_bcs(s.mg)

function MGPreconditionedCG(mesh::RegularGrid{T};
                            tol::Real = 1e-10, maxit::Int = 200,
                            mg_cycles::Int = 1, kwargs...) where {T}
    mg = MultigridPoissonSolver(mesh; tol = 0.0, maxcycles = mg_cycles,
                                kwargs...)
    nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
    z3() = zeros(T, nx, ny, nz)
    return MGPreconditionedCG{T}(mg, T(tol), maxit,
                                 z3(), z3(), z3(), z3(), z3(), z3(), z3(),
                                 Ref(0), Ref(zero(T)))
end

function MGPreconditionedCG(g::StretchedGrid{T}; kwargs...) where {T}
    rg, cx, cy, cz, sc = _stretched_operator(g)
    return MGPreconditionedCG(rg; coeff_x = cx, coeff_y = cy, coeff_z = cz,
                              rhs_scale = sc, kwargs...)
end

# Apply the fine-level operator to an interior array.
function _apply_fine!(out::Array{T,3}, x::Array{T,3},
                      s::MultigridPoissonSolver{T}) where {T}
    L = s.levels[1]
    nx, ny, nz = L.nx, L.ny, L.nz
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        L.φ[i+1, j+1, k+1] = x[i, j, k]
    end
    _apply_laplacian!(L.r, L, s.bc_x, s.bc_y, s.bc_z)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        out[i, j, k] = L.r[i+1, j+1, k+1]
    end
    return out
end

# One V-cycle as a preconditioner: z ← M⁻¹ r.
function _mg_precond!(z::Array{T,3}, r::Array{T,3},
                      s::MultigridPoissonSolver{T}) where {T}
    L = s.levels[1]
    nx, ny, nz = L.nx, L.ny, L.nz
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        L.b[i+1, j+1, k+1] = r[i, j, k]
    end
    s.singular && _project_mean!(L.b)
    fill!(L.φ, zero(T))
    for _ in 1:max(s.maxcycles, 1)
        mg_vcycle!(s, 1)
    end
    s.singular && _project_mean!(L.φ)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        z[i, j, k] = L.φ[i+1, j+1, k+1]
    end
    return z
end

# a·b as `sum(a .* b)` — the same pairwise sum, bit for bit — with the
# product in the scratch `w` instead of a fresh array (#295)
_dot!(w::Array{T,3}, a::Array{T,3}, b::Array{T,3}) where {T} = sum(w .= a .* b)

function _project_mean_flat!(a::Array{T,3}) where {T}
    a .-= sum(a) / length(a)
    return a
end

function solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T},
                solver::MGPreconditionedCG{T}) where {T}
    s  = solver.mg
    b  = copyto!(solver.b, interior_view(source))
    _scale_rhs!(b, s.rhs_scale)
    s.singular && _project_mean_flat!(b)
    bn = sqrt(sum(abs2, b))
    if bn == 0
        interior_view(p) .= 0
        solver.last_iters[] = 0
        solver.last_residual[] = zero(T)
        return p
    end

    x, r, z, pd, Ap = solver.x, solver.r, solver.z, solver.pdir, solver.Ap
    fill!(x, zero(T))
    r .= b
    _mg_precond!(z, r, s)
    s.singular && _project_mean_flat!(z)
    pd .= z
    rz_old = _dot!(solver.w, r, z)
    iters = 0
    rel = one(T)

    for it in 1:solver.maxit
        _apply_fine!(Ap, pd, s)
        pAp = _dot!(solver.w, pd, Ap)
        # breakdown test relative to r·z (pAp ~ r·z in magnitude), so the
        # iteration is invariant to the scale of the right-hand side
        abs(pAp) <= eps(T) * abs(rz_old) && break
        α = rz_old / pAp
        @. x = x + α * pd
        @. r = r - α * Ap
        rel = sqrt(sum(abs2, r)) / bn
        iters = it
        rel < solver.tol && break
        _mg_precond!(z, r, s)
        s.singular && _project_mean_flat!(z)
        rz_new = _dot!(solver.w, r, z)
        β = rz_new / rz_old
        @. pd = z + β * pd
        rz_old = rz_new
    end

    s.singular && _project_mean_flat!(x)
    interior_view(p) .= x
    solver.last_iters[] = iters
    solver.last_residual[] = rel
    return p
end

"Iterations used by the last `solve!` of an `MGPreconditionedCG`."
mg_iteration_history(s::MGPreconditionedCG) = s.last_iters[]
mg_residual_norm(s::MGPreconditionedCG) = s.last_residual[]

end # module
