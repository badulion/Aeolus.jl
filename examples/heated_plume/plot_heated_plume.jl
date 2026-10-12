# Coupled example 03, the plots: θ − 300 K in the horizontal plane nearest
# 400 m and in the vertical centre plane at the end of a run of
# heated_plume.toml. Reads the last record of output/output.nc and run.toml only.
#
#   julia --project -m Aeolus run examples/heated_plume/heated_plume.toml
#   julia --project=. examples/heated_plume/plot_heated_plume.jl runs/<date>_heated_plume_<id>
#
# Writes output/plume_theta_400m.ppm and plume_theta_vslice.ppm in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
dom = recorded_domain(folder)
F, t = read_last_record(output_file(folder))
θ = Float64.(F["theta"]) .- 300.0
Nx, Ny, Nz = size(θ); _, Ly, Lz = dom["grid"]["extent"]
k = round(Int, 400.0 / (Lz / Nz))
j = round(Int, (Ly / 2) / (Ly / Ny))
a = save_heatmap(plot_path(folder, "plume_theta_400m.ppm"), Array(θ[:, :, k]'); upscale = 8)
b = save_heatmap(plot_path(folder, "plume_theta_vslice.ppm"), reverse(Array(θ[:, j, :]'); dims = 1); upscale = 8)
@printf("t = %.0f s: θ' at z = %.0f m in [%+.2f, %+.2f] K\n%s\n%s\n", t, (k - 0.5) * Lz / Nz,
        extrema(θ[:, :, k])..., a, b)
