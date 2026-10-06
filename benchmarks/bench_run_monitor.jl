# Cost of the run monitor (#335) on a 64×64×32 Boussinesq case
# (step_adaptive!, the thread count Julia was started with): the
# diagnostics CSV, the progress line and the blowup guard, per step.
#
# Run: julia --project=. benchmarks/bench_run_monitor.jl
#
# Configurations, interleaved (NREP repetitions) so drift hits them equally:
#   null     no monitor
#   default  monitor! after every step with the defaults: costly diagnostics
#            every 60 s simulated, a progress line every 10 s wall, a blowup
#            check + last-good snapshot every 10 steps
#   every    the worst case: costly diagnostics, check and snapshot every step
# and, timed in isolation (median of 200 calls), the parts: a cheap row, a
# costly row (all 8 core diagnostics), a blowup check, a snapshot.

using Aeolus
using Aeolus.Fields: set!
using Aeolus.TimeIntegration: step_adaptive!, CFLConstraints
using Logging
using Printf
using Random

function build()
    m = RegularGrid(; size = (64, 64, 32), extent = (3200.0, 3200.0, 1600.0), halo = (3, 3, 3))
    sim = BoussinesqSimulation(m; θ₀ = 300.0,
                               bcs = Aeolus.BoundaryConditions.FieldBoundaryConditions(:periodic, :periodic, :wall))
    rng = Xoshiro(1)
    set!(sim.u, (x, y, z) -> 2.0 + 0.3 * randn(rng))
    set!(sim.θ, (x, y, z) -> 300.0 + 0.003 * z + 0.1 * randn(rng))
    return sim
end

const CFL = CFLConstraints(; dt_max = 5.0)

function steps!(sim, n, mon, step0)
    t = @elapsed for i in 1:n
        _, dt, nr = step_adaptive!(sim, CFL)
        mon === nothing || monitor!(mon, sim; step = step0 + i, dt, retries = nr)
    end
    return t / n
end

med(v) = sort(v)[cld(length(v), 2)]

sim = build()
dir = mktempdir()
mk(cadence, check) = RunMonitor(sim; csv = joinpath(dir, "d_$(cadence)_$check.csv"),
                                crash = joinpath(dir, "crash"), t_end = 1e9,
                                cadence = cadence, check_every = check)
mon_default = mk(60.0, 10)
mon_every = mk(0.0, 1)

with_logger(NullLogger()) do
    steps!(sim, 3, nothing, 0)                       # compile
    steps!(sim, 3, mon_default, 0); steps!(sim, 3, mon_every, 0)
end

n = 40
const NREP = 15
res = Dict(:null => Float64[], :default => Float64[], :every => Float64[])
step0 = Dict(:default => 3, :every => 3)
with_logger(NullLogger()) do
    for rep in 1:NREP
        push!(res[:null], steps!(sim, n, nothing, 0))
        push!(res[:default], steps!(sim, n, mon_default, step0[:default])); step0[:default] += n
        push!(res[:every], steps!(sim, n, mon_every, step0[:every])); step0[:every] += n
    end
end
dt_mean = sim.t / (9 + NREP * 3 * n)        # 9 compile steps + the timed ones

# the parts in isolation
w = mon_default.writer
g = BlowupGuard(sim; every_steps = 1)
tcheap = med([@elapsed(record_diagnostics!(w, sim.t; step = 0, costly = false)) for _ in 1:200])
tcostly = med([@elapsed(record_diagnostics!(w, sim.t; step = 0, costly = true)) for _ in 1:200])
tcheck = med([@elapsed(isempty(nonfinite_fields(sim))) for _ in 1:200])
tsnap = med([@elapsed(snapshot_good!(g, sim; step = 0)) for _ in 1:200])
close(mon_default); close(mon_every)

base = med(res[:null])
@printf("64×64×32 step_adaptive!, %d thread(s), %d steps × %d repetitions (median of repetitions); mean dt %.2f s\n",
        Threads.nthreads(), n, NREP, dt_mean)
for k in (:null, :default, :every)
    v = res[k]
    @printf("  %-8s %8.3f ms/step   (%+.2f %% vs null; spread of repetitions %.2f–%.2f ms)\n",
            k, 1e3 * med(v), 100 * (med(v) / base - 1), 1e3 * minimum(v), 1e3 * maximum(v))
end
for k in (:default, :every)
    @printf("  %-8s paired: median over repetitions of (config / null − 1) = %+.2f %%\n", k,
            100 * med(res[k] ./ res[:null] .- 1))
end
@printf("  parts: cheap row %.1f µs, costly row %.3f ms, blowup check %.3f ms, snapshot %.3f ms\n",
        1e6 * tcheap, 1e3 * tcostly, 1e3 * tcheck, 1e3 * tsnap)
per = tcheap + (tcheck + tsnap) / 10 + tcostly * dt_mean / 60
@printf("  default cadence from the parts: %.3f ms/step = %.2f %% of a null step\n",
        1e3 * per, 100 * per / base)
# the budget of epic #293: at most about 2 % of a step at the default cadence
# (asserted on the parts, which are free of the step-time noise)
per / base <= 0.02 || error("run monitor over budget: $(round(100 * per / base; digits = 2)) % of a step > 2 %")
