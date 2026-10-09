# Captures test/data/components/simulation_digests.toml (ONE-a, #450). Run
# ONCE on master ead7b986, before the core's step went through
# `Aeolus.Components`:
#
#   julia --project=. test/data/components/capture_simulation_digests.jl
#
# Kept for the record: test/components_pipeline.jl compares the
# pipeline's runs with these digests (test/support/simulation_digests.jl
# defines the variants).

using TOML
include(joinpath(@__DIR__, "..", "..", "support", "simulation_digests.jl"))

d = simulation_digests(; inputs = sd_made_inputs())
out = Dict{String,Any}("commit" => readchomp(`git -C $(@__DIR__) rev-parse HEAD`),
                       "threads" => Threads.nthreads(), "digests" => d)
open(joinpath(@__DIR__, "simulation_digests.toml"), "w") do io
    println(io, "# fnv1a64 digests of state_fields and the clock of every variant of")
    println(io, "# test/support/simulation_digests.jl at 16³ (capture_simulation_digests.jl).")
    TOML.print(io, out; sorted = true)
end
println("captured $(length(d)) variants")
