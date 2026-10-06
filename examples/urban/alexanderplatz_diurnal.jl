# Alexanderplatz, Berlin — THE simulation.
#
# Real OSM geometry, full Era III radiation, and every physics coupling of
# the Berlin case (examples/urban/BerlinCase.jl, one file per coupling
# under couplings/). This is the single run; nothing else in
# examples/urban integrates the equations. It writes the full 3-D state to
# output/alexanderplatz.nc; `alexanderplatz_animation.jl` and
# `alexanderplatz_surface_diagnostics.jl` only read that file.
#
#   julia --project=. examples/urban/alexanderplatz_diurnal.jl
#
# Environment:
#   AEOLUS_NSTEPS     number of steps (default: 24 h at dt = 2 s = 43200)
#   AEOLUS_OUTPUT_DT  seconds between NetCDF records (default 1800)
#   AEOLUS_SCENARIO   scenario preset (default alexanderplatz; see SCENARIOS)
#   AEOLUS_LOG        console log level (default info; debug adds the setup
#                     summary). output/alexanderplatz.log keeps everything
#                     at debug and up.

include(joinpath(@__DIR__, "BerlinCase.jl"))
using .BerlinCase
using Aeolus: with_run_log

cfg = CaseConfig(Symbol(get(ENV, "AEOLUS_SCENARIO", "alexanderplatz")))
haskey(ENV, "AEOLUS_NSTEPS") && (cfg.nsteps = parse(Int, ENV["AEOLUS_NSTEPS"]))
haskey(ENV, "AEOLUS_OUTPUT_DT") && (cfg.cadence[:output] = parse(Float64, ENV["AEOLUS_OUTPUT_DT"]))

with_run_log(joinpath(@__DIR__, "output", "alexanderplatz.log")) do
    case = build_case(cfg)
    run_case!(case)
end
println("Render with:  julia --project=. examples/urban/alexanderplatz_animation.jl")
