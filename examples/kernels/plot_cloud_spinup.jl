# Kernel demo 04, the plot: the horizontal-mean ql and qr profiles after
# each Kessler step of a run of configs/cloud_spinup.toml (the column
# driver's kessler_box). Reads output/results.toml only.
#
#   julia --project -m Aeolus run examples/kernels/configs/cloud_spinup.toml
#   julia --project=. examples/kernels/plot_cloud_spinup.jl runs/<date>_cloud_spinup_<id>
#
# Writes output/cloud_ql_profile.svg and cloud_qr_profile.svg in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
r = read_results(folder)
kb = recorded_config(folder)["kessler_box"]
nz = kb["size"][3]
z = [(k - 0.5) * kb["height"] / nz for k in 1:nz]
labels = ["t=$(round(Int, t))s" for t in r["t"]]
a = save_multiline(plot_path(folder, "cloud_ql_profile.svg"), z, r["ql_profiles"], labels;
                   xlabel = "z (m)", ylabel = "ql (kg/kg)", title = "Cloud water — Kessler spinup")
b = save_multiline(plot_path(folder, "cloud_qr_profile.svg"), z, r["qr_profiles"], labels;
                   xlabel = "z (m)", ylabel = "qr (kg/kg)", title = "Rain water — Kessler spinup")
@printf("final ql max = %.2e, qr max = %.2e\n%s\n%s\n", r["ql_max"][end], r["qr_max"][end], a, b)
