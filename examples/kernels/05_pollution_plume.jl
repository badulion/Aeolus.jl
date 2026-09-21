# Kernel demo 05 — Pollutant plume from a point source.
#
# Passive tracer with linear chemistry (first-order decay). Injected at
# a fixed point every step; advected by a prescribed uniform wind. The
# steady-state Gaussian-plume-like structure develops. Plot horizontal
# slice at emission height.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators: advect_centered!
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

Nx = 64; Ny = 48; Nz = 8; L = 100.0
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(2L, 1.5L, L), halo=(2,2,2))

sim = BoussinesqSimulation(mesh; ν = 1e-3, C_s = 0.15)   # a little SGS for plume spread
set!(sim.u, (x,y,z) -> 5.0)   # 5 m/s constant westerly
set!(sim.v, (x,y,z) -> 0.0)
set!(sim.w, (x,y,z) -> 0.0)
fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)

# Passive tracer c (kg/kg), transported alongside θ but with our own tendency
c  = Field(CellCenter(), mesh)
tc = Field(CellCenter(), mesh)
set!(c, (x,y,z) -> 0.0)

# Point source at (x0, y0, z0)
i_src = round(Int, 20.0 / mesh.dx)              # 20 m from west edge
j_src = round(Int, (1.5L / 2) / mesh.dy)        # centered in y
k_src = 4                                        # near-surface
S_val = 1.0   # kg / m³ · s injection rate

dt = 0.3 * mesh.dx / 5.0
T_end = 30.0
nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps

Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
for n in 1:nsteps
    # advect c
    fill_halos!(c, FieldBoundaryConditions(:periodic))
    fill!(tc.data, 0)
    advect_centered!(tc, sim.u, sim.v, sim.w, c)
    apply_linear_chemistry!(tc, c, 60.0)   # 60 s decay
    # inject at source
    tc.data[Hx + i_src, Hy + j_src, Hz + k_src] += S_val
    # advance c
    interior_view(c) .+= dt .* view(tc.data, Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz)
    # also advance velocity (small effect)
    step!(sim, dt)
end

# Horizontal slice at source height
slice = view(interior_view(c), :, :, k_src)          # shape (Nx, Ny)
Z     = Array(slice')                                # (Ny, Nx) for image row=y
save_heatmap(joinpath(OUT, "05_plume.ppm"), Z; upscale=6)

@printf("Plume max concentration: %.2e; source cell = (%d, %d, %d)\n",
        maximum(interior_view(c)), i_src, j_src, k_src)
@printf("output → %s\n", joinpath(OUT, "05_plume.ppm"))
