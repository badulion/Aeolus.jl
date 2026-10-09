# Kernel demo 02, the plot: KE(t)/KE₀ of a finished run of
# configs/decaying_hit.toml. Reads diagnostics.csv only.
#
#   julia --project -m Aeolus run examples/kernels/configs/decaying_hit.toml
#   julia --project=. examples/kernels/plot_decaying_hit.jl runs/<date>_decaying_hit_<id>
#
# Writes output/hit_ke.svg in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
d = read_diagnostics_csv(folder)
t, ke = diagnostic_series(d, "ke_mean")
# the rows start after the first step (the first projection of the white noise)
out = save_multiline(plot_path(folder, "hit_ke.svg"), t, [ke ./ ke[1]], [""];
                     xlabel = "t", ylabel = "KE / KE(t₁)", title = "Decaying isotropic turbulence")
@printf("KE(t = %.2f)/KE(t = %.3f) = %.4f\n%s\n", t[end], t[1], ke[end] / ke[1], out)
