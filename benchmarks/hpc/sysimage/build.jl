# Build the system image of one build (#566): Aeolus of a bench-<build>
# worktree, with the Berlin workload (workload.jl) compiled in.
#
#   JULIA_LOAD_PATH="@:<this dir>:@stdlib" julia --project=<this dir> build.jl <worktree> <out.so> <envdir>
#
# A package's own project cannot list itself, so <envdir> gets a wrapper
# environment with Aeolus from the worktree ([sources]) and the worktree's
# Manifest. The image holds that commit's code: `<out.so>.commit` records
# it, and node.sh uses the image only while the worktree is at that commit.
using PackageCompiler, Pkg
wt, out, env = abspath(ARGS[1]), abspath(ARGS[2]), abspath(ARGS[3])
mkpath(env)
write(joinpath(env, "Project.toml"), """
[deps]
Aeolus = "a1e01005-1e51-4a5e-8d1e-4a5e8d1e4a5e"

[sources]
Aeolus = {path = "$wt"}
""")
cp(joinpath(wt, "Manifest.toml"), joinpath(env, "Manifest.toml"); force = true)
Pkg.activate(env); Pkg.resolve(); Pkg.instantiate()      # resolve: the copied Manifest lacks Aeolus itself
commit = strip(read(`git -C $wt rev-parse HEAD`, String))
t = @elapsed create_sysimage(["Aeolus"]; sysimage_path = out, project = env,
                             precompile_execution_file = joinpath(@__DIR__, "workload.jl"))
write(out * ".commit", commit)
println("sysimage $out for $commit in $(round(t / 60; digits = 1)) min")
