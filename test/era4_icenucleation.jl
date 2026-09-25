# Era IV, Phase 43 — ice nucleation.
#
# One file per part of the phase, under test/icenuc/:
#   inp.jl       issue #5  prognostic INP species, budget, diagnostics
#   freezing.jl  issue #6  immersion / deposition freezing (added by #6)

using Test, Printf, Random
using Aeolus

@testset "Ice nucleation (Phase 43)" begin
    include("icenuc/inp.jl")
end
