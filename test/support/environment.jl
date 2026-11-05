# The environment every file of both suites runs in: test/runtests.jl and
# validation/runvalidation.jl include this first (the parallel runner's
# workers keep it, since it is not a literal include of the list).

using Test
using Random
# #245: the suites plan every FFT with FFTW.ESTIMATE: planning takes no time
# (MEASURE times candidate plans for every new size). Set before `using
# Aeolus`; subprocesses the files start inherit it. Results are compared
# within tolerances, not bits (#495, test/support/agreement.jl).
ENV["AEOLUS_FFTW_PLANNER"] = "estimate"
include(joinpath(@__DIR__, "agreement.jl"))   # agree, fingerprints (#495): every file has them
using Statistics: mean
using Aeolus
using Aeolus.Grids: xnodes, ynodes, znodes, interior
# #333: the suites show warnings and errors only (AEOLUS_LOG=debug or info
# opts in); files that assert on events capture them with their own logger.
using Logging: global_logger
global_logger(test_logger())
using Aeolus.Fields: set!, interior_view
