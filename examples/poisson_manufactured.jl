# Manufactured-solution validation for the FFT Poisson solver.
#
# We prescribe an analytic p(x,y,z) that satisfies triply-periodic BCs,
# form the discrete Laplacian source = L_h p using our operators, then
# solve L_h p̃ = source and compare p̃ to p.
#
# Because the solver's eigenvalues are the exact eigenvalues of the same
# discrete Laplacian, the error should sit at O(1e-12) — this is a
# machine-precision consistency check, not a convergence study.

using Aeolus
using Aeolus.Grids: interior
using Aeolus.Fields: interior_view, set!
using Printf

Lx = Ly = Lz = 1.0
Nx = Ny = Nz = 64
mesh = RegularGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), halo=(1, 1, 1))
bcs  = FieldBoundaryConditions(:periodic)

p_exact = Field(CellCenter(), mesh)
source  = Field(CellCenter(), mesh)
p_num   = Field(CellCenter(), mesh)

set!(p_exact, (x, y, z) -> sin(2π * x/Lx) * cos(2π * y/Ly) * sin(2π * z/Lz))
fill_halos!(p_exact, bcs)

laplacian(source, p_exact)          # discrete RHS from the same stencil

solver = FFTPoissonSolver(mesh)
solve!(p_num, source, solver)

err = maximum(abs.(interior_view(p_num) .- interior_view(p_exact)))
@printf("N = %d³   ||p_num - p_exact||∞ = %.3e\n", Nx, err)
@printf("expected: ≈ machine precision (< 1e-10)\n")
