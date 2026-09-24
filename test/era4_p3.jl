# Era IV, Phase 40 — Morrison–Milbrandt P3 microphysics.
#
# One test file per part-file of src/p3/, included here so parallel
# issues only add an include line.

using Test, Printf
using Aeolus

@testset "P3 microphysics (Phase 40)" begin
    include("p3/gamma.jl")          # gamma, incomplete gamma, PSD moments
    include("p3/massdim.jl")        # mass-dimension regimes, rime density, fall speed
    include("p3/processes.jl")      # process rates: condensation, warm rain, riming, melting
    include("p3/sedimentation.jl")  # per-moment sedimentation, size sorting, radar reflectivity
    include("p3/terminal_velocity.jl")  # Phase 44: Kessler fall-speed spectrum
    include("p3/precip_fraction.jl")    # Phase 44: sub-grid precipitation fraction
end
