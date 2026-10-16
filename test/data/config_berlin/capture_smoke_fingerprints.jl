# Captures the [smoke_fingerprints] tables of test/data/config_berlin/fixture.toml
# (#495, in place of the bitwise [smoke_digests]): the fingerprints
# (test/support/agreement.jl) of the nested Berlin smoke's state after its
# 12 steps, as test/config_berlin.jl runs it. Re-run after a change that
# moves the smoke run beyond round-off, with the reason in the commit:
#
#   julia --project=. test/data/config_berlin/capture_smoke_fingerprints.jl
#
# First captured on master 68c5724c (2026-10-16), whose run reproduced the
# smoke digests of NEST-p3 (#483) and #489 these replace.

using Aeolus, TOML, Printf, Logging, Test
using Aeolus.Config
using Aeolus.IO: read_run_toml, diagnostics_path
using Aeolus.Fields: interior_view
ENV["AEOLUS_FFTW_PLANNER"] = "estimate"
const BCC = Aeolus.Urban
const T = joinpath(@__DIR__, "..", "..")
include(joinpath(T, "support", "berlin_configs.jl"))
include(joinpath(T, "support", "berlin_fixture.jl"))
include(joinpath(T, "support", "agreement.jl"))
include(joinpath(T, "config_roundtrip_harness.jl"))

r = roundtrip_check(berlin_toml("smoke"); nsteps = 12, nsteps_overrides = n -> ["time.nsteps=$n"],
                    overrides = ["logging.diagnostics_cadence=1e-3"])
m = r.runs[1].obj.model
st = berlin_state(urban_site(m); parent = m.domains[1])
fps = Dict{String,Any}(string(k) => fingerprint(v) for (k, v) in st)

path = joinpath(@__DIR__, "fixture.toml")
text = read(path, String)
i = findfirst("\n[smoke_fingerprints", text)
text = i === nothing ? text : text[1:first(i)]
buf = IOBuffer()
println(buf, "\n# The nested smoke's state after 12 steps (capture_smoke_fingerprints.jl, #495).")
TOML.print(buf, Dict("smoke_fingerprints" => fps); sorted = true)
write(path, rstrip(text) * "\n" * String(take!(buf)))
println("wrote ", length(fps), " fingerprints")
