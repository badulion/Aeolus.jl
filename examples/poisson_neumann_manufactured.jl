# Manufactured-solution validation for the CosineFFTPoissonSolver.
#
# Domain: [0, 1]³, periodic in x, y and homogeneous-Neumann in z.
# We prescribe an analytic φ whose z-part is cos(π n z / Lz) — this
# satisfies ∂φ/∂z = 0 at both walls for any integer n. The discrete
# Laplacian of φ (with reflected halos in z) gives the source, and
# the solver recovers φ to machine precision.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Printf

Lx = Ly = Lz = 1.0
Nx = Ny = Nz = 64
mesh = RegularGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), halo=(1, 1, 1))
bcs  = FieldBoundaryConditions(:periodic, :periodic, :noflux)

φ_exact = Field(CellCenter(), mesh)
source  = Field(CellCenter(), mesh)
p_num   = Field(CellCenter(), mesh)

n = 3
set!(φ_exact, (x, y, z) -> cos(2π * x/Lx) * cos(2π * y/Ly) * cos(π * n * z/Lz))
fill_halos!(φ_exact, bcs)
laplacian(source, φ_exact)

solver = CosineFFTPoissonSolver(mesh)
solve!(p_num, source, solver)

err = maximum(abs.(interior_view(p_num) .- interior_view(φ_exact)))
@printf("N = %d³   z-mode n = %d\n", Nx, n)
@printf("||p_num - φ_exact||∞ = %.3e     (expected ≲ 1e-10)\n", err)
