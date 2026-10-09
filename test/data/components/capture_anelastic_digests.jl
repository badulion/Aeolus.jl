# Captures test/data/components/simulation_digests_anelastic.toml (ONE-j,
# #460): the digests of the anelastic `simulation` variants after the
# anelastic core's SGS and surface fluxes moved to the ρ̄-weighted flux form
# (and of the new anelastic RK3 / adaptive variants). Run ONCE on the ONE-j
# commit that made that change:
#
#   julia --project=. test/data/components/capture_anelastic_digests.jl
#
# test/components_pipeline.jl compares the pipeline's anelastic runs with
# these, and asserts they differ from master's (the weighting is new physics).

using TOML
include(joinpath(@__DIR__, "..", "..", "support", "simulation_digests.jl"))

vs = vcat([v for v in SD_VARIANTS if sd_anelastic(first(v))], SD_ANELASTIC_VARIANTS)
d = simulation_digests(; variants = vs)
out = Dict{String,Any}("commit" => readchomp(`git -C $(@__DIR__) rev-parse HEAD`),
                       "threads" => Threads.nthreads(), "digests" => d)
open(joinpath(@__DIR__, "simulation_digests_anelastic.toml"), "w") do io
    println(io, "# fnv1a64 digests of state_fields and the clock of the anelastic variants of")
    println(io, "# test/support/simulation_digests.jl at 16³ (capture_anelastic_digests.jl, ONE-j #460).")
    TOML.print(io, out; sorted = true)
end
println("captured $(length(d)) variants")
