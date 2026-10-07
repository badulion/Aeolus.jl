# Alexanderplatz, Berlin — THE simulation. The run is the shipped config
# examples/urban/configs/berlin_summer.toml (the `berlin` run driver, CFG-c
# #358), so this script is the same as
#
#   julia --project -m Aeolus run examples/urban/configs/berlin_summer.toml [--set clock.nsteps=1800] [--set output.cadence=600]
#
# and passes its arguments on. The NetCDF output goes to
# examples/urban/output/alexanderplatz.nc (read by alexanderplatz_animation.jl
# and alexanderplatz_surface_diagnostics.jl); run.toml, the log and
# diagnostics.csv go to the run folder runs/<date>_summer_<id>/.
using Aeolus
exit(Aeolus.main(["run", joinpath(@__DIR__, "configs", "berlin_summer.toml"), ARGS...]))
