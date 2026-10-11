# Kernel demo 04, the plot: the horizontal-mean ql and qr profiles after
# each Kessler step of a run of configs/cloud_spinup.toml. Reads the
# micro.ql and micro.qr profiles of the run's diagnostics.nc only.
#
#   julia --project -m Aeolus run examples/kernels/configs/cloud_spinup.toml
#   julia --project=. examples/kernels/plot_cloud_spinup.jl runs/<date>_cloud_spinup_<id>
#
# Writes output/cloud_ql_profile.svg and cloud_qr_profile.svg in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
z, QL, t = read_profile(folder, "micro.ql")
_, QR, _ = read_profile(folder, "micro.qr")
labels = ["t=$(round(Int, x))s" for x in t]
a = save_multiline(plot_path(folder, "cloud_ql_profile.svg"), z, [QL[:, j] for j in axes(QL, 2)], labels;
                   xlabel = "z (m)", ylabel = "ql (kg/kg)", title = "Cloud water — Kessler spinup")
b = save_multiline(plot_path(folder, "cloud_qr_profile.svg"), z, [QR[:, j] for j in axes(QR, 2)], labels;
                   xlabel = "z (m)", ylabel = "qr (kg/kg)", title = "Rain water — Kessler spinup")
@printf("final ql max = %.2e, qr max = %.2e\n%s\n%s\n", maximum(QL[:, end]), maximum(QR[:, end]), a, b)
