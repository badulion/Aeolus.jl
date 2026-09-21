# Kernel demo 03 — Neutral channel flow.
#
# Periodic in x, y; impermeable walls in z. A body-force analog is
# supplied by giving u a mean initial value — the profile relaxes.
# Prints and plots the horizontally-averaged ⟨u⟩(z) profile.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Random
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

Nx = Ny = 16; Nz = 32; L = 1.0; ν = 5e-4
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(L,L,L), halo=(2,2,2))
wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
sim  = BoussinesqSimulation(mesh; ν=ν, C_s=0.17, Pr_t=1/3, bcs=wall)

# Uniform mean flow with random perturbations
Random.seed!(1)
set!(sim.u, (x,y,z) -> 1.0 + 0.05*(rand()-0.5))
set!(sim.v, (x,y,z) -> 0.05*(rand()-0.5))
set!(sim.w, (x,y,z) -> 0.05*(rand()-0.5))

dx  = mesh.dx; dz = mesh.dz
dt  = min(0.3 * dx / 1.2, 0.4 * dz^2 / (2ν))
T_end = 2.0
nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps

for n in 1:nsteps
    step!(sim, dt)
end

# ⟨u⟩(z)
u_int = interior_view(sim.u)
u_mean = [sum(view(u_int, :, :, k)) / (size(u_int,1)*size(u_int,2)) for k in 1:Nz]
zs = collect(Aeolus.Grids.znodes(mesh, Aeolus.Grids.CellCenter()))

save_lineplot(joinpath(OUT, "03_channel_umean.svg"),
              u_mean, zs;
              xlabel="⟨u⟩(z)", ylabel="z",
              title="Channel-flow mean profile (Nz=$Nz)")

@printf("Channel: ⟨u⟩ range = [%.3f, %.3f], max|div| = %.2e\n",
        minimum(u_mean), maximum(u_mean), max_divergence(sim))
@printf("output → %s\n", joinpath(OUT, "03_channel_umean.svg"))
