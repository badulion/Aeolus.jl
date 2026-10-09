# Coupled example 04, the plots: u and the tracer in the vertical centre
# plane y = L_y/2 at the end of a run of configs/flow_past_cube.toml.
# Reads the last output/fields_*.nc and run.toml only.
#
#   julia --project -m Aeolus run examples/coupled/configs/flow_past_cube.toml
#   julia --project=. examples/coupled/plot_flow_past_cube.jl runs/<date>_flow_past_cube_<id>
#
# Writes output/cube_u_slice.ppm and cube_tracer.ppm in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
cfg = run_config_table(folder)
F, t = read_fields(last_fields_file(folder))
Ny = size(F["u"], 2); Ly = cfg["grid"]["extent"][2]
j = round(Int, (Ly / 2) / (Ly / Ny))
u = Float64.(F["u"][:, j, :]); c = Float64.(F["tracer"][:, j, :])
a = save_heatmap(plot_path(folder, "cube_u_slice.ppm"), reverse(Array(u'); dims = 1); upscale = 6)
b = save_heatmap(plot_path(folder, "cube_tracer.ppm"), reverse(Array(c'); dims = 1); upscale = 6)
@printf("t = %.0f s: u in [%.2f, %.2f] m/s, tracer max %.2e\n%s\n%s\n", t, extrema(u)..., maximum(c), a, b)
