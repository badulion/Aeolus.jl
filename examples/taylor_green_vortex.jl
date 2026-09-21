# 3D Taylor-Green vortex — Phase 1a step-2 end-to-end validation.
#
# Initial condition (triply periodic, domain [0, 2π]³):
#     u =  sin(x) cos(y) cos(z)
#     v = -cos(x) sin(y) cos(z)
#     w =  0
#
# This field is analytically incompressible; after discretization there
# is a small residual divergence that the projection step removes on the
# first step. The vortex then evolves into a turbulent state and decays
# under viscous action.
#
# What we check here (this script prints — not an automated test):
#   * ∇·u stays at O(1e-12) throughout (correctness of the projection).
#   * Kinetic energy monotonically decreases for ν > 0 (no blow-up).

using Aeolus
using Aeolus.Fields: set!
using Printf

L  = 2π
N  = 32
ν  = 1e-2                 # Re ≈ 100 based on L, U₀=1

mesh = RegularGrid(; size=(N, N, N), extent=(L, L, L), halo=(2, 2, 2))
sim  = BoussinesqSimulation(mesh; ν = ν)

set!(sim.u, (x, y, z) ->  sin(x) * cos(y) * cos(z))
set!(sim.v, (x, y, z) -> -cos(x) * sin(y) * cos(z))
set!(sim.w, (x, y, z) ->  0.0)

# CFL: umax ≈ 1, dx = 2π/N.  Include a viscous safety factor (ν · dt / dx²).
dx = mesh.dx
dt_conv = 0.4 * dx / 1.0
dt_visc = 0.4 * dx^2 / (2ν)
dt      = min(dt_conv, dt_visc)

T_final = 2.0
nsteps  = ceil(Int, T_final / dt)
dt      = T_final / nsteps

ke0    = kinetic_energy(sim)
divmag = max_divergence(sim)
@printf("TGV N = %d³   ν = %.2e   dt = %.3e   nsteps = %d\n", N, ν, dt, nsteps)
@printf("t = %.3f   KE = %.5e   max|div| = %.2e\n", 0.0, ke0, divmag)

for n in 1:nsteps
    step!(sim, dt)
    if n % max(1, nsteps ÷ 10) == 0
        ke = kinetic_energy(sim)
        dv = max_divergence(sim)
        @printf("t = %.3f   KE = %.5e   KE/KE₀ = %.4f   max|div| = %.2e\n",
                sim.t, ke, ke / ke0, dv)
    end
end
