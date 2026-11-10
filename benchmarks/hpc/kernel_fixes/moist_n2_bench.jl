# Starting material from the laptop measurements (#554-#560), see benchmarks/hpc/HANDOFF.md.
# Paths below point at the laptop's scratch and worktrees: adapt them before use.
# #557 N² commit: Berlin advance_sgs! (pass, separate) and the moist N²
# loop alone; one checkout per process (base = the face commit, new = +N²)
using Aeolus, Statistics
using Aeolus.Operators: advance_sgs!
const SIM = Aeolus.Simulations
const TKB = Aeolus.PrognosticSGS
which = ARGS[1]
S = "/tmp/claude-1000/-home-andi-coding-work-Aeolus-jl/3e8f9011-18a5-4b37-8ab9-483314fc7365/scratchpad"
cfg = load_run_config("$S/i533int/berlin_100m_20250824.toml"; overrides = ["fftw.planner=\"estimate\""])
model = Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model
for _ in 1:3; step!(model, 2.0); end
d = model.domains[1]; sim = d.state; c = Aeolus.Components.sgs_closure(d)
e0 = copy(c.e.data)
runb(set) = (SIM.set_loop_fusion!(sim, set); copyto!(c.e.data, e0); advance_sgs!(c, sim, 2.0))
runb(:if_possible); write("$S/i557/e_$(which)_t$(Threads.nthreads()).bin", c.e.data)
fs = ("pass" => () -> runb(:if_possible), "separate" => () -> runb(:never),
      "n2_loop" => () -> TKB._stratification!(c, sim), "copy_e" => () -> copyto!(c.e.data, e0))
for (k, f) in fs; f(); f(); end
res = Dict(k => Float64[] for (k, _) in fs)
for rep in 1:15, (k, f) in (isodd(rep) ? fs : reverse(fs))
    for _ in 1:3; t0 = time_ns(); f(); push!(res[k], (time_ns() - t0) / 1e6); end
end
print(which, " t", Threads.nthreads())
for (k, _) in fs
    print("  ", k, " ", round(minimum(res[k]), digits = 2), " / ", round(median(res[k]), digits = 2))
end
println()
