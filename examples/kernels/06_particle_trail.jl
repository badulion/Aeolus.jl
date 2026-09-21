# Kernel demo 06 — Lagrangian particle trails in a Taylor-Green vortex.
#
# Seed particles at random positions; advance them in a periodic TGV
# flow; record their trajectories and render a 2-D plot of the trails.

using Aeolus
using Aeolus.Fields: set!
using Random
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

L = 2π; N = 32
mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
sim  = BoussinesqSimulation(mesh; ν=1e-2)
set!(sim.u, (x,y,z) ->  sin(x)*cos(y)*cos(z))
set!(sim.v, (x,y,z) -> -cos(x)*sin(y)*cos(z))
fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)

ps = ParticleSet(mesh)
Random.seed!(7)
add_particles!(ps, [(L*rand(), L*rand(), L/2) for _ in 1:24])

dt = 0.05
nsteps = 100

# Record trajectories (only x, y — we're in a z-slice)
trails_x = [Float64[] for _ in eachindex(positions(ps))]
trails_y = [Float64[] for _ in eachindex(positions(ps))]

for n in 1:nsteps
    step!(sim, dt)
    fill_halos!(sim.u, sim.bcs); fill_halos!(sim.v, sim.bcs); fill_halos!(sim.w, sim.bcs)
    advance_particles!(ps, sim, dt)
    for (i, p) in enumerate(positions(ps))
        push!(trails_x[i], p[1])
        push!(trails_y[i], p[2])
    end
end

# Render each particle's y(x) trajectory (24 lines)
labels = ["" for _ in 1:24]
save_multiline(joinpath(OUT, "06_particle_trails.svg"),
               trails_x[1], trails_y, labels;
               xlabel="x", ylabel="y",
               title="Particle trails in 3-D TGV (z = L/2)")

@printf("%d particles integrated for %d steps.\n", length(positions(ps)), nsteps)
@printf("output → %s\n", joinpath(OUT, "06_particle_trails.svg"))
