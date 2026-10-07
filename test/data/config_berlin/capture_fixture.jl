# Captures test/data/config_berlin/fixture.toml (CFG-c, #358). Run ONCE on
# the commit before the Berlin scenarios moved to TOML (SCENARIOS still in
# examples/urban/core/scenarios.jl), so the fixture holds what the
# Julia-built cases were:
#
#   julia --project=. test/data/config_berlin/capture_fixture.jl
#
# It records, per scenario, toml_value(CaseConfig(s)) and
# toml_value(nest_parent_config(CaseConfig(s))) (the parent the case runs),
# and the fnv1a64 digests of the :smoke case's state after 12 steps
# (FFTW planner "estimate"). Normalisation (shared with
# test/config_berlin.jl): the package directory in a path is written
# "<pkg>"; a function-valued field is written as its values at
# t = 0, 3600, 43200 s. Kept for the record; it no longer runs after the
# migration (SCENARIOS is gone).

using Aeolus, TOML
using Aeolus.IO: toml_value
isdefined(Main, :BerlinCase) || include(joinpath(@__DIR__, "..", "..", "..", "examples", "urban", "BerlinCase.jl"))
const BC = Main.BerlinCase
include(joinpath(@__DIR__, "..", "..", "support", "berlin_fixture.jl"))

out = Dict{String,Any}("commit" => readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
for s in (:summer, :winter, :smoke, :mitte_sample)
    cfg = BC.CaseConfig(s)
    out[string(s)] = Dict{String,Any}("case" => normalise(cfg), "parent" => normalise(BC.nest_parent_config(cfg)))
end
Aeolus.Operators.set_fftw_policy!(; planner = :estimate)
case = BC.build_case(BC.CaseConfig(:smoke))
BC.run_case!(case)
out["smoke_digests"] = Dict{String,Any}(string(k) => digest(v) for (k, v) in pairs(berlin_state(case)))
out["smoke_digests_threads"] = Threads.nthreads()
open(joinpath(@__DIR__, "fixture.toml"), "w") do io
    TOML.print(io, out; sorted = true)
end
println("wrote ", length(out["smoke_digests"]), " digests at ", Threads.nthreads(), " threads")
