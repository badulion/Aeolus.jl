# Coupled example 02 — moist LES with an initial cloud layer.
#
# Runs the LES with prognostic moisture (qv). A cloud puff of ql is
# injected at mid-height and processed by Kessler microphysics — some
# rains out, some remains as cloud. Tracks column mass over time.

using Aeolus
using Aeolus.Fields: set!, interior_view
using Random
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

Lx = Ly = 3200.0; Lz = 2400.0
Nx = Ny = 24; Nz = 32
mesh = RegularGrid(; size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz), halo=(2,2,2))

sim = BoussinesqSimulation(mesh;
    θ₀ = 300.0, g_accel = 9.81,
    C_s = 0.15, Pr_t = 1/3,
    moisture = true,
    θ_flux_bottom = 0.05,
    sponge_α_max = 0.02, sponge_z_start = 0.8*Lz,
    bcs   = FieldBoundaryConditions(:periodic, :periodic, :wall),
    θ_bcs = FieldBoundaryConditions(:periodic, :periodic, :noflux))

Random.seed!(3)
set!(sim.θ, (x,y,z) -> 300.0 + max(0, (z - 1500)) * 4e-3 + 0.03*(rand()-0.5))
set!(sim.qv, (x,y,z) -> 0.012 * exp(-z / 1500))

# Prescribed cloud puff between 1.0 and 1.4 km (a "cloud layer" in some
# horizontal patches only). Rain grows from Kessler; there's no
# saturation-adjustment feedback so behavior stays bounded.
ql = Field(CellCenter(), mesh)
qr = Field(CellCenter(), mesh)
set!(ql, (x,y,z) -> begin
    xc, yc = Lx/2, Ly/2
    if abs(x - xc) < Lx/4 && abs(y - yc) < Ly/4 && 1000 < z < 1400
        return 2e-3
    else
        return 0.0
    end
end)

dt = 5.0
T_end = 20 * 60.0
nsteps = ceil(Int, T_end / dt); dt = T_end / nsteps

# History
ts       = Float64[0.0]
ql_total = Float64[sum(interior_view(ql))]
qr_total = Float64[sum(interior_view(qr))]

@printf("Moist convection with cloud puff: %d steps × %.1f s = %.1f min\n",
        nsteps, dt, T_end / 60)

for n in 1:nsteps
    step!(sim, dt)
    apply_kessler_step!(sim.qv, ql, qr, dt; Vt=:spectrum)
    if n % 12 == 0
        push!(ts, sim.t / 60)
        push!(ql_total, sum(interior_view(ql)))
        push!(qr_total, sum(interior_view(qr)))
    end
end

save_multiline(joinpath(OUT, "coupled_02_ql_qr_history.svg"),
               ts, [ql_total, qr_total], ["Σ ql", "Σ qr"];
               xlabel="time (min)", ylabel="total mixing ratio",
               title="Kessler processing of an initial cloud puff")

# Vertical profiles at end
zs = collect(Aeolus.Grids.znodes(mesh, Aeolus.Grids.CellCenter()))
hprof(f) = [sum(view(interior_view(f), :, :, k)) / (Nx*Ny) for k in 1:Nz]
save_multiline(joinpath(OUT, "coupled_02_final_profiles.svg"),
               zs, [hprof(ql) .* 1e5, hprof(qr) .* 1e5], ["ql × 1e5", "qr × 1e5"];
               xlabel="z (m)", ylabel="mixing ratio × 1e5",
               title="Moisture profiles at t=$(round(Int, T_end/60)) min")

@printf("Final: max ql = %.2e, max qr = %.2e, max|div| = %.2e\n",
        maximum(interior_view(ql)), maximum(interior_view(qr)),
        max_divergence(sim))
