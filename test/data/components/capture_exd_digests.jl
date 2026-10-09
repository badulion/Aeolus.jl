# Captures test/data/components/simulation_digests_exd.toml (ONE-a, #450):
# the simulation configs of EX-d (#441) with their couplings, run ONCE on
# master a22b57ba (the merge of #446), before they went through the
# pipeline. From a checkout of that commit, with this branch's support file:
#
#   julia --project=. <branch>/test/data/components/capture_exd_digests.jl <checkout>/examples
#
# Kept for the record; test/components_pipeline.jl compares with it.

using TOML
include(joinpath(@__DIR__, "..", "..", "support", "simulation_digests.jl"))

examples = ARGS[1]
d = exd_digests(; examples = examples)
out = Dict{String,Any}("commit" => readchomp(`git -C $examples rev-parse HEAD`),
                       "threads" => Threads.nthreads(), "digests" => d)
open(joinpath(@__DIR__, "simulation_digests_exd.toml"), "w") do io
    println(io, "# fnv1a64 digests of state_fields, the clock and diagnostics.csv of every variant")
    println(io, "# of SD_EXD_VARIANTS (test/support/simulation_digests.jl; capture_exd_digests.jl).")
    TOML.print(io, out; sorted = true)
end
println("captured $(length(d)) variants")
