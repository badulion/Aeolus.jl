# Parallel runner for the validation runs of Aeolus.jl: simulated cases that
# confirm the physics works and stays stable. Not a merge gate. Stdlib only.
#
#   julia --project=. validation/run.jl                 # every validation run
#   julia --project=. validation/run.jl --only gabls1.jl,gabls3.jl
#   julia --project=. validation/run.jl --list          # the plan
#
# Options: see test/runner/runner.jl (the same runner as test/run.jl).
# Docs: docs/src/validation.md.

include(joinpath(@__DIR__, "..", "test", "runner", "runner.jl"))
exit(main(ARGS, "validation"))
