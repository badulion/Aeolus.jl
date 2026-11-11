# The sanity check of the builds (#566, HANDOFF.md section 3 item 5): u, v,
# w, θ, e after 20 steps of Berlin 100 m (bench_run.jl `state`) of every
# build against bench/master. The tolerance is a reason, not a hash: the
# relative L2 difference of master against itself at another thread count
# (round-off grown over 20 steps; summation order changes with the thread
# count) is the spread; a build agrees when its difference is within
# `factor` (10) times that spread. Stdlib only.
#
#   julia state_compare.jl <dir> [factor]     (dir: *.state.jls from bench_run.jl)
#     master run 1: master_a.state.jls, master run 2: master_b.state.jls
using Serialization, Printf, TOML

dir = ARGS[1]; factor = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 10.0
load(n) = deserialize(joinpath(dir, n * ".state.jls"))
reldiff(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b))) / max(sqrt(sum(abs2, Float64.(b))), eps())
ref = load("master_a")
spread = Dict(k => reldiff(v, ref[k]) for (k, v) in load("master_b"))
fields = sort!(collect(keys(ref)))
out = Dict{String,Any}("spread" => spread, "factor" => factor, "builds" => Dict{String,Any}())
@printf("%-14s %s\n", "build", join((@sprintf("%-22s", f) for f in fields), ""))
@printf("%-14s %s\n", "spread", join((@sprintf("%-22.3e", spread[f]) for f in fields), ""))
for f in sort!(filter(n -> endswith(n, ".state.jls") && !startswith(n, "master_"), readdir(dir)))
    b = replace(f, ".state.jls" => ""); s = load(b)
    d = Dict(k => reldiff(s[k], ref[k]) for k in fields if haskey(s, k))
    ok = all(k -> haskey(d, k) && (d[k] <= factor * max(spread[k], 1e-12)), fields)
    out["builds"][b] = Dict("reldiff" => d, "agrees" => ok)
    @printf("%-14s %s %s\n", b, join((@sprintf("%-22s", haskey(d, k) ? @sprintf("%.3e (%.1f×)", d[k], d[k] / max(spread[k], 1e-300)) : "missing") for k in fields), ""), ok ? "ok" : "DIFFERS")
end
open(io -> TOML.print(io, out; sorted = true), joinpath(dir, "state_compare.toml"), "w")
