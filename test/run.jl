# Parallel runner for the tests of Aeolus.jl (the PR gate). Stdlib only.
#
#   julia --project=. test/run.jl                       # every test file, in parallel
#   julia --project=. test/run.jl --only io_checkpoint.jl
#   julia --project=. test/run.jl --list                # the plan
#
# Validation runs are a separate suite: julia --project=. validation/run.jl
# Options: see test/runner/runner.jl. Docs: docs/src/testing.md.

include(joinpath(@__DIR__, "runner", "runner.jl"))
exit(main(ARGS, "test"))
