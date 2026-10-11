# Kernel demo 05, the plot: the tracer at the source height at the end of a
# run of pollution_plume.toml. Reads the last record of output/output.nc
# and run.toml only.
#
#   julia --project -m Aeolus run examples/pollution_plume/pollution_plume.toml
#   julia --project=. examples/pollution_plume/plot_pollution_plume.jl runs/<date>_pollution_plume_<id>
#
# Writes output/plume.ppm in the run folder.

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
dom = recorded_domain(folder)
F, t = read_last_record(output_file(folder))
c = F["tracer"]
nz = size(c, 3); Lz = dom["grid"]["extent"][3]
zs = only(values(dom["tracer"]["sources"]))["z"]
k = clamp(round(Int, (zs[1] + zs[2]) / 2 / (Lz / nz) + 0.5), 1, nz)      # the source level
out = save_heatmap(plot_path(folder, "plume.ppm"), Array(c[:, :, k]'); upscale = 6)
@printf("t = %.1f s: tracer max %.2e, at the source level %d: %.2e\n%s\n", t, maximum(c), k,
        maximum(c[:, :, k]), out)
