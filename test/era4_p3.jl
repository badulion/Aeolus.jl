# Era IV, Phase 40 — Morrison–Milbrandt P3 microphysics.
#
# One test file per part-file of src/p3/, included here so parallel
# issues only add an include line.

using Test, Printf
using Aeolus

@testset "P3 microphysics (Phase 40)" begin
    include("p3/gamma.jl")          # gamma, incomplete gamma, PSD moments
end
