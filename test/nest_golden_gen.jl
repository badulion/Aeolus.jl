# Records the NestedRun golden checksums (#328) into test/data/nest_golden.toml.
#
#   julia --project=. test/nest_golden_gen.jl
#
# Run on the code before a change that must keep NestedRun the same
# (it was run on the commit before the nest hierarchy refactor); the test
# (era2_nest_hierarchy.jl, "golden NestedRun") compares against the file.

using TOML
include(joinpath(@__DIR__, "nest_golden_cases.jl"))

out = Dict{String,Any}()
for (name, f) in nest_golden_cases()
    v = f()
    out[name] = v
    println(rpad(name, 40), " ", length(v), " checksums")
end
path = joinpath(@__DIR__, "data", "nest_golden.toml")
open(path, "w") do io
    println(io, "# NestedRun golden checksums (#328), written by test/nest_golden_gen.jl")
    TOML.print(io, out)
end
println("wrote ", path)
