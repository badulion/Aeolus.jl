# Captures test/data/components/simulation_fingerprints.toml (#495): the
# fingerprints of every variant of test/support/simulation_fingerprints.jl.
# Re-run after a change that moves the variants' results beyond round-off,
# with the reason in the commit:
#
#   julia --project=. test/data/components/capture_simulation_fingerprints.jl
#
# First captured on master 68c5724c (2026-10-16), whose runs reproduced the
# bitwise digests these replace (ead7b986, ONE-j, a22b57ba).

using TOML
ENV["AEOLUS_FFTW_PLANNER"] = "estimate"
include(joinpath(@__DIR__, "..", "..", "support", "simulation_fingerprints.jl"))

d = Dict{String,Any}(first(v) => variant_fingerprints(v) for v in SD_ALL)
out = Dict{String,Any}("commit" => readchomp(`git -C $(@__DIR__) rev-parse HEAD`), "variants" => d)
open(joinpath(@__DIR__, "simulation_fingerprints.toml"), "w") do io
    println(io, "# Fingerprints (test/support/agreement.jl) of every variant of")
    println(io, "# test/support/simulation_fingerprints.jl (capture_simulation_fingerprints.jl).")
    TOML.print(io, out; sorted = true)
end
println("captured $(length(d)) variants")
