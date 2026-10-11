# Kernel demo 06, the plot: the particle trajectories (x, y) of a run of
# particle_trail.toml. Reads output/particles.csv and run.toml only.
#
#   julia --project -m Aeolus run examples/particle_trail/particle_trail.toml
#   julia --project=. examples/particle_trail/plot_particle_trail.jl runs/<date>_particle_trail_<id>
#
# Writes output/particle_trails.svg in the run folder (paths unwrapped across
# the periodic faces).

using Printf
include(joinpath(@__DIR__, "..", "viz", "plot_utils.jl"))
include(joinpath(@__DIR__, "..", "viz", "run_output.jl"))

folder = run_folder_arg()
Lx, Ly, _ = recorded_domain(folder)["grid"]["extent"]
rows = [parse.(Float64, split(l, ',')) for l in readlines(joinpath(folder, "output", "particles.csv"))[2:end]]
ids = sort(unique(r[3] for r in rows))
unwrap(v, L) = (u = copy(v); for i in 2:length(u); u[i] = u[i-1] + (v[i] - v[i-1]) - L * round((v[i] - v[i-1]) / L); end; u)
xs = [unwrap([r[4] for r in rows if r[3] == id], Lx) for id in ids]
ys = [unwrap([r[5] for r in rows if r[3] == id], Ly) for id in ids]
out = save_paths(plot_path(folder, "particle_trails.svg"), xs, ys; xlabel = "x", ylabel = "y",
                 title = "Particle trails in a Taylor–Green vortex (z = π)")
@printf("%d particles, %d positions each\n%s\n", length(ids), length(xs[1]), out)
