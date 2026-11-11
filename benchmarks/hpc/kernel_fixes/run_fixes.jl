# Several kernel fixes in one process, on one Berlin build (#566, HANDOFF.md
# section 7; the keys in common.jl):
#
#   julia -t N --project=<build> run_fixes.jl <out.toml> E1E2 E3 E4 E5 E6 [key=value ...]
#
# Only the requested fixes run; the Berlin state is built once if any needs it.
include(joinpath(@__DIR__, "common.jl"))
for f in ("tke_update_strain_bench.jl", "face_values_bench.jl", "moist_n2_bench.jl",
          "masked_diffusion_bench.jl", "rsc_update_bench.jl")
    include(joinpath(@__DIR__, f))
end
isempty(REQUESTED) && error("no fix named (of $(join(sort!(collect(keys(FIXES))), ", ")))")
run_fixes(REQUESTED)
