# Coupled example 03 — heated hot-patch plume (dry).
#
# A localized hot square drives a rising plume. Visualizes horizontal
# slice of θ' at mid-height showing the plume's warm signature and
# ring of colder subsiding air around it.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Printf
using Random

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

Lx = Ly = 2000.0; Lz = 1500.0
Nx = Ny = 32; Nz = 32
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))

sim = BoussinesqSimulation(mesh;
    θ₀ = 300.0, g_accel = 9.81,
    C_s = 0.15, Pr_t = 1/3,
    sponge_α_max = 0.02, sponge_z_start = 0.8Lz,
    bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
    θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

Random.seed!(2)
set!(sim.θ, (x,y,z) -> 300.0 + max(0, (z - 1000.0)) * 5e-3 + 0.02*(rand()-0.5))

# Hot patch flux at surface — inject via per-cell θ update at k=1
Hx, Hy, Hz = mesh.Hx, mesh.Hy, mesh.Hz
xs = collect(Aeolus.Grids.xnodes(mesh, Aeolus.Grids.CellCenter()))
ys = collect(Aeolus.Grids.ynodes(mesh, Aeolus.Grids.CellCenter()))

hot_flux(x, y) = (abs(x - Lx/2) < Lx/8 && abs(y - Ly/2) < Ly/8) ? 0.3 : 0.02

dt = 2.0
T_end = 15 * 60.0
nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps

for n in 1:nsteps
    for j in 1:Ny, i in 1:Nx
        sim.θ.data[Hx+i, Hy+j, Hz+1] += hot_flux(xs[i], ys[j]) * dt / mesh.dz
    end
    step!(sim, dt)
end

# Horizontal θ' slice at 400 m
k_slice = round(Int, 400.0 / mesh.dz)
θ_ref = 300.0
slice = view(interior_view(sim.θ), :, :, k_slice) .- θ_ref
save_heatmap(joinpath(OUT, "coupled_03_theta_400m.ppm"),
             Array(slice'); upscale=8)

# Vertical center-plane slice: y = Ly/2
j_cen = round(Int, (Ly/2) / mesh.dy)
θ_vslice = view(interior_view(sim.θ), :, j_cen, :) .- θ_ref
save_heatmap(joinpath(OUT, "coupled_03_theta_vslice.ppm"),
             reverse(Array(θ_vslice'); dims=1); upscale=8)

@printf("θ' range at z=400 m: [%+.2f, %+.2f] K, max|div| = %.2e\n",
        minimum(slice), maximum(slice), max_divergence(sim))
