# Era IV, Phase 43 — ice nucleation.
#
# One file per part of the phase, under test/icenuc/:
#   inp.jl       issue #5  prognostic INP species, budget, diagnostics
#   freezing.jl  issue #6  immersion / deposition freezing, source terms
#   p3_coupling.jl issue #7 nucleation sources into the P3 ice category
#   stratus_column.jl issue #7 the ISDAC-style Arctic stratus column: initial profile,
#                short-run budgets (the 2-8 h persistence runs: validation/icenucleation.jl)

using Test, Printf, Random
using Aeolus

@testset "Ice nucleation (Phase 43)" begin
    include("icenuc/inp.jl")
    include("icenuc/freezing.jl")
    include("icenuc/p3_coupling.jl")
    include("icenuc/stratus_column.jl")
end
