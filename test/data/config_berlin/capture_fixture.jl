# Captures test/data/config_berlin/fixture.toml (CFG-c, #358). Run ONCE on
# the commit before the Berlin scenarios moved to TOML (SCENARIOS still in
# examples/urban/core/scenarios.jl), so the fixture holds what the
# Julia-built cases were:
#
#   julia --project=. test/data/config_berlin/capture_fixture.jl
#
# It records, per scenario, toml_value(site_options(s)) and
# toml_value(nest_parent_config(site_options(s))) (the parent the case runs),
# and the fnv1a64 digests of the :smoke case's state after 12 steps
# (FFTW planner "estimate"). Normalisation (shared with
# test/config_berlin.jl): the package directory in a path is written
# "<pkg>"; a function-valued field is written as its values at
# t = 0, 3600, 43200 s. Kept for the record; it no longer runs after the
# migration (SCENARIOS is gone). Its smoke_digests are replaced by the
# [smoke_fingerprints] of capture_smoke_fingerprints.jl (#495).

using Aeolus, TOML
using Aeolus.IO: toml_value
const BC = Aeolus.Urban
include(joinpath(@__DIR__, "..", "..", "support", "berlin_configs.jl"))
include(joinpath(@__DIR__, "..", "..", "support", "berlin_fixture.jl"))

out = Dict{String,Any}("commit" => readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
for s in (:summer, :winter, :smoke, :mitte_sample)
    cfg = site_config(berlin_toml(s))
    out[string(s)] = Dict{String,Any}("case" => normalise(cfg), "parent" => normalise(BC.nest_parent_config(cfg)))
end
Aeolus.Operators.set_fftw_policy!(; planner = :estimate)
case = BC.build_case(site_config(berlin_toml("smoke")))
BC.run_case!(case)
out["smoke_digests"] = Dict{String,Any}(string(k) => digest(v) for (k, v) in pairs(berlin_state(case)))
out["smoke_digests_threads"] = Threads.nthreads()
open(joinpath(@__DIR__, "fixture.toml"), "w") do io
    TOML.print(io, out; sorted = true)
end
println("wrote ", length(out["smoke_digests"]), " digests at ", Threads.nthreads(), " threads")
