# Dry Convective Boundary Layer (DCBL) — Phase 8.
#
# Setup roughly following Sullivan & Patton (2011, JAS), simplified:
#   * Domain L × L × H = 3200 × 3200 × 1600 m
#   * Grid  32 × 32 × 32  (very coarse — this is a smoke test, not a
#     reference-quality run)
#   * BC: periodic in x, y; impermeable free-slip walls in z
#   * Prescribed surface heat flux w'θ'|₀ = 0.24 K m/s
#   * Initial θ = θ₀ + Γ(z - z_i) capping inversion at z_i, with small
#     noise near the surface to trigger turbulence
#   * Rayleigh sponge in the upper 25 % of the domain
#   * SGS: Smagorinsky-Lilly with Cₛ = 0.17, Prₜ = 1/3
#
# Diagnostics printed each 10 % of the run: total KE, divergence, max νₜ,
# top-of-BL height (based on θ maximum gradient), mixed-layer mean θ,
# resolved w'θ' at the surface layer.

using Aeolus
using Aeolus.Fields: set!
using Random
using Statistics: mean, argmax
using Printf

Random.seed!(1)

# Physical parameters
Lx = Ly = 3200.0          # horizontal domain [m]
Lz    = 1600.0            # vertical extent   [m]
θ₀    = 300.0             # reference potential temperature [K]
g_ac  = 9.81              # gravity [m/s²]
z_i0  = 800.0             # initial inversion height [m]
Γ_free = 3e-3             # free-atmosphere lapse rate [K/m]
Q_sfc  = 0.24             # kinematic surface heat flux [K m/s]

# Numerics
Nx = Ny = 32
Nz     = 32
mesh   = RegularGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), halo=(2, 2, 2))

sim = BoussinesqSimulation(mesh;
    θ₀              = θ₀,
    g_accel         = g_ac,
    C_s             = 0.17,
    Pr_t            = 1/3,
    θ_flux_bottom   = Q_sfc,
    θ_flux_top      = 0.0,
    sponge_α_max    = 0.01,
    sponge_z_start  = 0.75 * Lz,
    bcs             = FieldBoundaryConditions(:periodic, :periodic, :wall),
    θ_bcs           = FieldBoundaryConditions(:periodic, :periodic, :noflux),
)

# Initial θ: mixed-layer capping inversion + small perturbations near the ground
Δz_i = 100.0    # capping-inversion width
function θ_init(x, y, z)
    if z < z_i0
        base = θ₀
    else
        base = θ₀ + Γ_free * (z - z_i0)
    end
    perturb = z < 100.0 ? 0.1 * (rand() - 0.5) : 0.0
    return base + perturb
end
set!(sim.θ, θ_init)
set!(sim.u, (x,y,z) -> 0.0)
set!(sim.v, (x,y,z) -> 0.0)
set!(sim.w, (x,y,z) -> 0.0)

# Convective velocity scale
w_star = (g_ac / θ₀ * Q_sfc * z_i0) ^ (1/3)

# CFL — dominated by advective time scale ~ z_i / w*
dx  = mesh.dx
dy  = mesh.dy
dz  = mesh.dz
dt  = min(0.3 * dx / max(w_star, 1e-6),
          0.5 * dz^2 / (0.17^2 * (dx*dy*dz)^(1/3) * w_star + 1e-12),
          5.0)

T_end  = 300.0            # 5 minutes of simulated time
nsteps = ceil(Int, T_end / dt)
dt     = T_end / nsteps

@printf("DCBL:   N = %d × %d × %d    w* = %.2f m/s    dt = %.2f s    nsteps = %d\n",
        Nx, Ny, Nz, w_star, dt, nsteps)

function report(sim, tag)
    m_θ = mean_θ_profile(sim)
    dθdz = diff(m_θ) ./ mesh.dz
    z_i_est = argmax(dθdz) * mesh.dz + mesh.dz/2   # crude z_i estimate
    flux0 = resolved_flux_wθ(sim)[max(1, div(Nz, 8))]  # near the surface
    @printf("[%s]  KE = %.3e   max|div| = %.2e   max νₜ = %.3e   z_i ≈ %.0f m   ⟨w'θ'⟩|z=z_i/8 = %+.3e\n",
            tag, kinetic_energy(sim), max_divergence(sim), max_νt(sim), z_i_est, flux0)
end

report(sim, @sprintf("t=%.0fs", 0.0))

for n in 1:nsteps
    step!(sim, dt)
    if n % max(1, nsteps ÷ 5) == 0
        report(sim, @sprintf("t=%.0fs", sim.t))
    end
end
