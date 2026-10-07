# Dry convective boundary layer: the run is examples/configs/dry_convective_bl.toml (aeolus run; extra arguments are passed on).
using Aeolus
exit(Aeolus.main(["run", joinpath(@__DIR__, "configs", "dry_convective_bl.toml"), ARGS...]))
