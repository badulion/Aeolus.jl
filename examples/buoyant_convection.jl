# Rayleigh–Bénard-flavoured convection: unstably stratified θ profile,
# small random perturbations, triply periodic (we're not yet ready for
# real walls — Phase 1b). This exercises the coupled system end-to-end:
# advection, buoyancy, Smagorinsky SGS, pressure projection.
#
# What to look for:
#   * KE grows from ~0 as thermals develop and then plateaus.
#   * ν_t stays bounded (does not blow up).
#   * ∇·u stays at O(1e-12) throughout.

using Aeolus
using Aeolus.Fields: set!
using Random
using Printf

Random.seed!(1)

L  = 1.0                     # cubic box
N  = 32
θ₀ = 300.0
Δθ = 1.0                     # top-to-bottom potential-temp difference
mesh = RegularGrid(; size=(N, N, N), extent=(L, L, L), halo=(2, 2, 2))

sim = BoussinesqSimulation(mesh;
                           θ₀       = θ₀,
                           g_accel  = 9.81,
                           C_s      = 0.17,
                           Pr_t     = 1/3)

# Unstable stratification: θ decreases with z (hot bottom, cold top),
# plus small random perturbations to break symmetry.
set!(sim.θ, (x, y, z) -> θ₀ + Δθ * (0.5 - z/L) + 1e-3 * Δθ * (rand() - 0.5))

# CFL — buoyancy sets the free-fall velocity scale w* ~ sqrt(g Δθ/θ₀ L)
w_star = sqrt(sim.g_accel * Δθ / θ₀ * L)
dx     = mesh.dx
dt     = 0.3 * dx / w_star
T_end  = 5.0
nsteps = ceil(Int, T_end / dt)
dt     = T_end / nsteps

@printf("Convection: N = %d³   w* = %.3f m/s   dt = %.3e   nsteps = %d\n",
        N, w_star, dt, nsteps)
@printf("t = %.3f   KE = %.3e   max νₜ = %.3e   max|div| = %.2e\n",
        0.0, kinetic_energy(sim), max_νt(sim), max_divergence(sim))

for n in 1:nsteps
    step!(sim, dt)
    if n % max(1, nsteps ÷ 10) == 0
        @printf("t = %.3f   KE = %.3e   max νₜ = %.3e   max|div| = %.2e\n",
                sim.t, kinetic_energy(sim), max_νt(sim), max_divergence(sim))
    end
end
