# Kernel demo 01 — Taylor-Green vortex energy decay.
#
# Runs the 3-D TGV, records KE(t), plots decay against analytic
# `KE₀ · exp(-6νt)` reference.

using Aeolus
using Aeolus.Fields: set!
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output")   # shared output dir
mkpath(OUT)

N   = 32; L = 2π; ν = 1e-2
mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
sim  = BoussinesqSimulation(mesh; ν = ν)
set!(sim.u, (x,y,z) ->  sin(x)*cos(y)*cos(z))
set!(sim.v, (x,y,z) -> -cos(x)*sin(y)*cos(z))

dx  = mesh.dx
dt  = min(0.4dx / 1.0, 0.4dx^2 / (2ν))
T_end = 4.0
nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps

ke0 = kinetic_energy(sim)
ts  = Float64[0.0]
kes = Float64[ke0]

for n in 1:nsteps
    step!(sim, dt)
    push!(ts, sim.t); push!(kes, kinetic_energy(sim))
end

analytic = ke0 .* exp.(-6ν .* ts)
save_multiline(joinpath(OUT, "01_tgv_ke.svg"),
               ts, [kes ./ ke0, analytic ./ ke0],
               ["Aeolus", "e^(-6νt) (linear)"];
               xlabel="t", ylabel="KE / KE₀",
               title="TGV kinetic-energy decay (N=$N, ν=$ν)")

@printf("TGV %d³ ν=%.1e  final KE/KE₀ = %.4f (linear = %.4f)\n",
        N, ν, kes[end]/ke0, analytic[end]/ke0)
@printf("output → %s\n", joinpath(OUT, "01_tgv_ke.svg"))
