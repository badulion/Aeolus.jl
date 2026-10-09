# Coupled example 02, the plots: the summed cloud and rain water over time
# and their horizontal-mean profiles at the end of a run of
# configs/moist_convection.toml. Reads diagnostics.csv and diagnostics.nc only.
#
#   julia --project -m Aeolus run examples/coupled/configs/moist_convection.toml
#   julia --project=. examples/coupled/plot_moist_convection.jl runs/<date>_moist_convection_<id>
#
# Writes output/moist_ql_qr_history.svg and moist_final_profiles.svg in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
d = read_diagnostics_csv(folder)
t, qls = diagnostic_series(d, "micro.ql_sum"); _, qrs = diagnostic_series(d, "micro.qr_sum")
_, qlm = diagnostic_series(d, "micro.ql_max"); _, qrm = diagnostic_series(d, "micro.qr_max")
_, div = diagnostic_series(d, "max_div")
a = save_multiline(plot_path(folder, "moist_ql_qr_history.svg"), t ./ 60, [qls, qrs], ["Σ ql", "Σ qr"];
                   xlabel = "time (min)", ylabel = "total mixing ratio", title = "Kessler processing of a cloud puff")
z, QL, t = read_profile(folder, "micro.ql")
_, QR, _ = read_profile(folder, "micro.qr")
b = save_multiline(plot_path(folder, "moist_final_profiles.svg"), z, [QL[:, end] .* 1e5, QR[:, end] .* 1e5],
                   ["ql × 1e5", "qr × 1e5"]; xlabel = "z (m)", ylabel = "mixing ratio × 1e5",
                   title = @sprintf("Cloud and rain water at t = %.0f min", t[end] / 60))
@printf("t = %.0f min: max ql = %.2e, max qr = %.2e, max|div| = %.1e\n%s\n%s\n", t[end] / 60,
        qlm[end], qrm[end], div[end], a, b)
