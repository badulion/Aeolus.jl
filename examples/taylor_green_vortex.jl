# 3-D Taylor-Green vortex: the run is examples/configs/taylor_green_vortex.toml (aeolus run; extra arguments are passed on).
using Aeolus
exit(Aeolus.main(["run", joinpath(@__DIR__, "configs", "taylor_green_vortex.toml"), ARGS...]))
