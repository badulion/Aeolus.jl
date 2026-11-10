# #533: ms per step of a configured case with loop_fusion = "never" (one
# loop per term) and "if_possible" (fused), alternating in blocks of steps
# within one process, so the boost/sustained drift of a power-capped CPU
# (#535) hits both alike. Prints the minimum ms/step of each setting.
#
#   julia -t N --project=. benchmarks/bench_loop_fusion.jl [case.toml] [nblocks=4] [steps_per_block=3]
#
# The default case (bench_loop_fusion_cbl.toml) is the dry CBL without SGS,
# with Coriolis: only the pointwise terms of #533 step 2 besides advection
# and diffusion. Run it under the benchmark lock of a shared machine.
using Aeolus, Printf
using Aeolus.Config: load_run_config
const SD = Aeolus.SimulationDriver
const SIMs = Aeolus.Simulations
cfg = load_run_config(get(ARGS, 1, joinpath(@__DIR__, "bench_loop_fusion_cbl.toml")))
r = Aeolus.Config.build(SD.SimulationDriverType(), cfg)
d = SD._domain(r)
sim = d.state
dt = 2.0
nblk = parse(Int, get(ARGS, 2, "4")); nper = parse(Int, get(ARGS, 3, "3"))
for m in (:never, :if_possible)        # compile both
    SIMs.set_loop_fusion!(sim, m); step!(d, dt); step!(d, dt)
end
best = Dict(:never => Inf, :if_possible => Inf)
for b in 1:nblk, m in (isodd(b) ? (:never, :if_possible) : (:if_possible, :never))
    SIMs.set_loop_fusion!(sim, m)
    for _ in 1:nper
        t = @elapsed step!(d, dt)
        best[m] = min(best[m], t)
    end
end
rep = SIMs.sweep_report(sim)
@printf("ROW threads=%d never=%.2f fused=%.2f ms/step speedup=%.3f  (%s; report %s %s)\n", Threads.nthreads(),
        1e3best[:never], 1e3best[:if_possible], best[:never] / best[:if_possible],
        string(size(Aeolus.Fields.interior_view(sim.u))), rep["substages_fused"], join(rep["not_fusable"], ";"))
