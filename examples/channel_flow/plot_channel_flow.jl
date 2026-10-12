# Kernel demo 03, the plot: the plane-mean ⟨u⟩(z) at the end of a run of
# channel_flow.toml. Reads the u profile of diagnostics.nc only.
#
#   julia --project -m Aeolus run examples/channel_flow/channel_flow.toml
#   julia --project=. examples/channel_flow/plot_channel_flow.jl runs/<date>_channel_flow_<id>
#
# Writes output/channel_umean.svg in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
z, U, t = read_profile(folder, "u")
u = U[:, end]
out = save_lineplot(plot_path(folder, "channel_umean.svg"), u, z;
                    xlabel = "⟨u⟩(z)", ylabel = "z", title = @sprintf("Channel-flow mean profile, t = %.2f", t[end]))
@printf("⟨u⟩ range = [%.3f, %.3f] at t = %.2f\n%s\n", extrema(u)..., t[end], out)
