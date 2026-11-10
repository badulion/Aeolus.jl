# The ERA5 dynamic driver of the Berlin 100 m run: what `aeolusprep dynamic
# --from-era5` does (AeolusPrep.era5_dynamic), whose run-kind check refuses a
# tiled domain alone (run_kind = :tiles, though it is a model domain that is
# not an urban site): the same calls without that check.
#
#   julia --project=env dynamic.jl                    # this directory's config
#   julia --project=env dynamic.jl ../berlin_50m/berlin_50m_20250824.toml   # another grid
#                                                     # (the same ERA5 data/; HPC benchmark #566)
using AeolusPrep, Aeolus, Aeolus.Config
using Aeolus.BoundaryConditions: open_boundary_states
const AP = AeolusPrep
cfg = load_run_config(isempty(ARGS) ? joinpath(@__DIR__, "berlin_100m_20250824.toml") : abspath(ARGS[1]))
SD = Aeolus.SimulationDriver
mesh = SD._mesh(cfg.grid)
planes = only(open_boundary_states(SD._open_bcs(cfg, mesh))).planes
e = AP.read_era5([joinpath(@__DIR__, "data", "era5_berlin_20250824_$s.nc") for s in ("pl", "sfc")]; levels = :pressure)
path = cfg.domain.dynamic_driver
AP.write_era5_driver(path, e; lat = 52.515, lon = 13.40, mesh, planes, box = 0.5, outputs = AP.ERA5_OUTPUTS, soil = :auto)
println("wrote ", path)
