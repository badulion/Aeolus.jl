# Coupled example 01 — DCBL + radiative cooling + slab LSM.
#
# Standard convective boundary layer with:
#   * Prescribed atmospheric radiative cooling (2 K/day)
#   * `SlabLand` surface reservoir warmed by SW + LW, cooled by
#     upward sensible flux
#   * Surface flux fed back into θ tendency each step
#
# Runs a compressed "day": 3 h simulated in a few minutes wall-time.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Aeolus.Radiation: apply_radiative_cooling!, SlabLand, step_land!, surface_temperature
using Random
using Statistics: mean
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

# --- Setup -----------------------------------------------------------------

Lx = Ly = 3200.0; Lz = 1600.0
Nx = Ny = 32; Nz = 32
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))

wall = FieldBoundaryConditions(:periodic, :periodic, :wall)
θ_bc = FieldBoundaryConditions(:periodic, :periodic, :noflux)

sim = BoussinesqSimulation(mesh;
    θ₀ = 300.0, g_accel = 9.81,
    C_s = 0.17, Pr_t = 1/3,
    sponge_α_max = 0.01, sponge_z_start = 0.75*Lz,
    bcs = wall, θ_bcs = θ_bc)

Random.seed!(1)
set!(sim.θ, (x,y,z) -> 300.0 + max(0, (z - 800.0)) * 3e-3 + 0.05*(rand()-0.5))

# Land: coarse "prairie" surface. C_s per m² of ground.
land = SlabLand(; T = 297.0, C_s = 1e5, albedo = 0.25)

# Radiative cooling profile: -2 K/day = -2.315e-5 K/s
cool_profile = fill(-2.315e-5, Nz)

# --- Time loop -------------------------------------------------------------

T_end = 3.0 * 3600.0    # 3 h
dt    = 5.0
nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps

ρcp = 1.0 * 1005.0      # air ρcp

ts     = Float64[0.0]
T_soil = Float64[surface_temperature(land)]
z_i    = Float64[800.0]
KE     = Float64[kinetic_energy(sim)]

@printf("DCBL + radiation + LSM: %d steps × %.1f s = %.1f h\n",
        nsteps, dt, T_end / 3600)

for n in 1:nsteps
    # SW forcing — sinusoidal "afternoon" (0 → 800 W/m² across the 3 h)
    sw_in = 800.0 * sin(π * sim.t / T_end)
    lw_in = 340.0                                  # atmospheric downward LW

    # 1. Sensible flux from LSM (upward, warming the air)
    Q_sfc = (surface_temperature(land) - 300.0) * 1e-2   # very simple K-flux law
    # 2. Set the surface θ-flux via a mutable field on sim (dry Boussinesq)
    sim.θ_flux_bottom = Q_sfc
    # 3. Advance atmosphere
    fill!(sim.tθ.data, 0)   # (step! zeros this anyway; belt & braces)
    step!(sim, dt)
    # 4. Apply radiative cooling profile
    apply_radiative_cooling!(sim.θ, cool_profile .* dt) # already tendency but we integrate directly here
    # 5. Advance LSM (upward sensible flux cools the ground)
    step_land!(land, sw_in, lw_in, ρcp * Q_sfc, dt)

    if n % max(1, nsteps ÷ 30) == 0
        m_θ = mean_θ_profile(sim)
        dθdz = diff(m_θ) ./ mesh.dz
        push!(ts, sim.t / 3600)
        push!(T_soil, surface_temperature(land))
        push!(z_i, argmax(dθdz) * mesh.dz)
        push!(KE,  kinetic_energy(sim))
    end
end

save_multiline(joinpath(OUT, "coupled_01_Tsoil.svg"),
               ts, [T_soil], ["T_soil (K)"];
               xlabel="time (h)", ylabel="T (K)",
               title="DCBL: surface temperature over 3 h")

save_multiline(joinpath(OUT, "coupled_01_zi.svg"),
               ts, [z_i], ["z_i (m)"];
               xlabel="time (h)", ylabel="height (m)",
               title="DCBL: mixed-layer height")

save_multiline(joinpath(OUT, "coupled_01_KE.svg"),
               ts, [KE], ["KE"];
               xlabel="time (h)", ylabel="KE",
               title="DCBL: turbulence kinetic energy")

@printf("Final:  T_soil = %.1f K,  z_i ≈ %.0f m,  KE = %.2e,  max|div| = %.2e\n",
        T_soil[end], z_i[end], KE[end], max_divergence(sim))
