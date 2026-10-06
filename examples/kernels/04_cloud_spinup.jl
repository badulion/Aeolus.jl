# Kernel demo 04 — Kessler cloud spinup.
#
# Cell-column model: initialize ql above the "cloud threshold" and let
# Kessler microphysics convert it to qr, then let rain fall out. Plot
# vertical profiles at multiple times.

# The box is examples/kernels/configs/cloud_spinup.toml (driver "column",
# case "kessler_box"); this script runs it and plots the profiles.
#
#   julia --project=. examples/kernels/04_cloud_spinup.jl [--set key=value]

using Aeolus
using Printf

include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))

const OUT = joinpath(@__DIR__, "..", "urban", "output"); mkpath(OUT)

box = only(run_configs(joinpath(@__DIR__, "configs", "cloud_spinup.toml"), ARGS)).obj
zs = collect(Aeolus.Grids.znodes(box.state.mesh, Aeolus.Grids.CellCenter()))
ql_snap = [s.ql for s in box.snapshots]
qr_snap = [s.qr for s in box.snapshots]
labels  = ["t=$(round(Int, s.t))s" for s in box.snapshots]

save_multiline(joinpath(OUT, "04_cloud_ql_profile.svg"),
               zs, ql_snap, labels;
               xlabel="z (m)", ylabel="ql (kg/kg)",
               title="Cloud water — Kessler spinup")

save_multiline(joinpath(OUT, "04_cloud_qr_profile.svg"),
               zs, qr_snap, labels;
               xlabel="z (m)", ylabel="qr (kg/kg)",
               title="Rain water — Kessler spinup")

@printf("Cloud spinup complete. Final ql max = %.2e, qr max = %.2e\n",
        maximum(ql_snap[end]), maximum(qr_snap[end]))
@printf("output → %s , %s\n",
        joinpath(OUT, "04_cloud_ql_profile.svg"),
        joinpath(OUT, "04_cloud_qr_profile.svg"))
