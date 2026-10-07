# Buoyant (Rayleigh-Benard-like) convection: the run is examples/configs/buoyant_convection.toml (aeolus run; extra arguments are passed on).
using Aeolus
exit(Aeolus.main(["run", joinpath(@__DIR__, "configs", "buoyant_convection.toml"), ARGS...]))
