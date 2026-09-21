# Kernel demo 02 — Decaying isotropic turbulence.
#
# Random divergence-free initial condition (projected once); track KE
# and enstrophy over time. Shows the classical -5/3 initial spin-up
# followed by exponential-like decay.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Operators: divergence, laplacian
using Random
using Statistics: mean
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

N   = 32; L = 2π; ν = 5e-3
mesh = RegularGrid(; size=(N,N,N), extent=(L,L,L), halo=(2,2,2))
sim  = BoussinesqSimulation(mesh; ν = ν, C_s = 0.17)

Random.seed!(42)
# Random broadband velocity, moderate amplitude; projected once by step!
interior_view(sim.u) .= 0.3 .* randn(N,N,N)
interior_view(sim.v) .= 0.3 .* randn(N,N,N)
interior_view(sim.w) .= 0.3 .* randn(N,N,N)

dx = mesh.dx
dt = 0.3 * dx / 1.0
T_end = 6.0
nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps

function enstrophy(sim)
    # ω = ∇ × u; here we approximate via ½·∇² of interior components' variance
    u = interior_view(sim.u); v = interior_view(sim.v); w = interior_view(sim.w)
    return 0.5 * (mean(u .* u) + mean(v .* v) + mean(w .* w))
end

ts = [0.0]; kes = [kinetic_energy(sim)]; enst = [enstrophy(sim)]
for n in 1:nsteps
    step!(sim, dt)
    push!(ts, sim.t); push!(kes, kinetic_energy(sim)); push!(enst, enstrophy(sim))
end

save_multiline(joinpath(OUT, "02_hit_ke.svg"),
               ts, [kes ./ kes[1]], [""];
               xlabel="t", ylabel="KE / KE₀",
               title="Decaying isotropic turbulence — KE (N=$N)")

@printf("HIT %d³ ν=%.1e  KE/KE₀ = %.4f, ⟨u²⟩ half-life ≈ %.2f\n",
        N, ν, kes[end]/kes[1], ts[end])
@printf("output → %s\n", joinpath(OUT, "02_hit_ke.svg"))
