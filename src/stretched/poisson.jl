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

using ..PressureSolvers: AbstractPressureSolver, ColumnTridiagonalSolver
import ..PressureSolvers: solve!, poisson_axis_bcs
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

The transform, the Thomas sweep and the mean-mode integration are those of
[`ColumnTridiagonalSolver`](@ref Aeolus.PressureSolvers.ColumnTridiagonalSolver)
(field `core`), which also solves the density-weighted projection of the
sound-proof cores (#309): `h = 1`, `a_k = 1/(Δz_c[k] Δz_f[k])`,
`c_k = 1/(Δz_c[k] Δz_f[k+1])`, measure `Δz_c`, gauge `Δz_c`.

The default solver of `BoussinesqSimulation(g::StretchedGrid)` for a
non-uniform grid; periodic-in-z is not supported (a stretched vertical
coordinate has no periodic meaning).
"""
struct StretchedPoissonSolver{T, C<:ColumnTridiagonalSolver{T}} <: AbstractPressureSolver
    core::C
end

function StretchedPoissonSolver(g::StretchedGrid{T};
                                bc_x::Symbol = :periodic,
                                bc_y::Symbol = :periodic) where {T}
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
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
    core = ColumnTridiagonalSolver(T, (Nx, Ny, Nz), (T(g.dx), T(g.dy));
                                   lower = lower, upper = upper, hweight = ones(T, Nz),
                                   measure = dzc, inv_coupling = dzf, gauge = dzc,
                                   bc_x = bc_x, bc_y = bc_y)
    return StretchedPoissonSolver{T, typeof(core)}(core)
end

solve!(p::Field{CellCenter,T}, source::Field{CellCenter,T}, s::StretchedPoissonSolver{T}) where {T} =
    solve!(p, source, s.core)
poisson_axis_bcs(s::StretchedPoissonSolver) = poisson_axis_bcs(s.core)

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
