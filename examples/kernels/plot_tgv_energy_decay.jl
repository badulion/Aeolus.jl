# Kernel demo 01, the plot: KE(t)/KE(t₁) of a finished run of
# configs/tgv_energy_decay.toml against the linear decay e^(−6νt).
# Reads diagnostics.csv and run.toml only.
#
#   julia --project -m Aeolus run examples/kernels/configs/tgv_energy_decay.toml
#   julia --project=. examples/kernels/plot_tgv_energy_decay.jl runs/<date>_tgv_energy_decay_<id>
#
# Writes output/tgv_ke.svg in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
ν = run_config_table(folder)["dynamics"]["nu"]
d = read_diagnostics_csv(folder)
t, ke = diagnostic_series(d, "ke_mean")          # the rows start after the first step
linear = exp.(-6ν .* (t .- t[1]))
out = save_multiline(plot_path(folder, "tgv_ke.svg"), t, [ke ./ ke[1], linear],
                     ["Aeolus", "e^(-6νt) (linear)"]; xlabel = "t", ylabel = "KE / KE(t₁)",
                     title = "TGV kinetic-energy decay (ν = $ν)")
@printf("KE(%.2f)/KE(%.3f) = %.4f (linear %.4f)\n%s\n", t[end], t[1], ke[end] / ke[1], linear[end], out)
