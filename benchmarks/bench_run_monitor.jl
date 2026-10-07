# Cost of the run monitor (#335, #337, #336) on a 64×64×32 Boussinesq case
# (step_adaptive!, the thread count Julia was started with): the
# diagnostics CSV, diagnostics.nc (scalars and profiles), the progress
# line and the blowup guard, per step; and the step cores' record and
# hook (#336) that an attached monitor reports through.
#
# Run: julia --project=. benchmarks/bench_run_monitor.jl
#
# Configurations, interleaved (NREP repetitions) so drift hits them equally:
#   null     no monitor
#   csv      monitor! after every step with the defaults: costly diagnostics
#            every 60 s simulated, a progress line every 10 s wall, a blowup
#            check + last-good snapshot every 10 steps; diagnostics.csv only
#   default  the same plus diagnostics.nc (what a run folder writes)
#   every    the worst case: costly diagnostics, check and snapshot every step
#   attached the default configuration attached to the step cores (#336):
#            the hook reports every step, the driver calls nothing
# and, timed in isolation (median of 200 calls), the parts: a cheap row, a
# costly row (all core, step-record and process diagnostics), a blowup
# check, a snapshot, and for diagnostics.nc a buffered row, the block write
# + sync (every 5 s wall) and a costly row (the 5 profiles plus their write
# and sync); and the step record's bracket with a no-op hook (#336).

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
mk(cadence, check; nc = true) = RunMonitor(sim; csv = joinpath(dir, "d_$(cadence)_$(check)_$nc.csv"),
                                nc = nc ? joinpath(dir, "d_$(cadence)_$check.nc") : nothing,
                                crash = joinpath(dir, "crash"), t_end = 1e9,
                                cadence = cadence, check_every = check)
mon_default = mk(60.0, 10)
mon_csv = mk(60.0, 10; nc = false)
mon_every = mk(0.0, 1)
mon_att = mk(60.0, 10)          # attached around its own repetitions (one hook per sim)

with_logger(NullLogger()) do
    steps!(sim, 3, nothing, 0)                       # compile
    steps!(sim, 3, mon_default, 0); steps!(sim, 3, mon_every, 0); steps!(sim, 3, mon_csv, 0)
    attach_monitor!(mon_att); steps!(sim, 3, nothing, 0); detach_monitor!(mon_att)
end

n = 32          # 5 configurations × 15 × 32 steps: the case blows up after ~2930 steps (master too)
const NREP = 15
res = Dict(:null => Float64[], :csv => Float64[], :default => Float64[], :every => Float64[],
           :attached => Float64[])
step0 = Dict(:csv => 3, :default => 3, :every => 3)
with_logger(NullLogger()) do
    for rep in 1:NREP
        push!(res[:null], steps!(sim, n, nothing, 0))
        push!(res[:csv], steps!(sim, n, mon_csv, step0[:csv])); step0[:csv] += n
        push!(res[:default], steps!(sim, n, mon_default, step0[:default])); step0[:default] += n
        # attached before every: the worst case's HDF5 syncs slow whatever follows
        attach_monitor!(mon_att)
        push!(res[:attached], steps!(sim, n, nothing, 0))
        detach_monitor!(mon_att)
        push!(res[:every], steps!(sim, n, mon_every, step0[:every])); step0[:every] += n
    end
end
dt_mean = sim.t / (15 + NREP * 5 * n)       # 15 compile steps + the timed ones

# the parts in isolation
w = mon_default.writer
g = BlowupGuard(sim; every_steps = 1)
tcheap = med([@elapsed(record_diagnostics!(w, sim.t; step = 0, costly = false)) for _ in 1:200])
tcostly = med([@elapsed(record_diagnostics!(w, sim.t; step = 0, costly = true)) for _ in 1:200])
tcheck = med([@elapsed(isempty(nonfinite_fields(sim))) for _ in 1:200])
tsnap = med([@elapsed(snapshot_good!(g, sim; step = 0)) for _ in 1:200])
# the step record's bracket around a step, with a no-op hook (#336): what
# every step core adds per step (a few time_ns() calls and additions);
# timed over 10^4 brackets, as one is ~100 ns
rec = step_record(sim)
set_step_hook!(sim, (s, r) -> nothing)
const SIM = Aeolus.Simulations
function brackets(rec, sim, k)
    for _ in 1:k
        o = SIM._step_begin!(rec, sim)
        SIM._step_end!(rec, sim, 1.0, o)
    end
end
brackets(rec, sim, 10)
trec = med([@elapsed(brackets(rec, sim, 10_000)) for _ in 1:20]) / 10_000
set_step_hook!(sim, nothing)
nw = mon_default.nc
nw.flush_interval = Inf; nw.flush_every = typemax(Int)       # the buffered row alone
tnc_cheap = med([@elapsed(record_diagnostics!(nw, sim.t; step = 0, costly = false, values = w.last))
                 for _ in 1:200])
tnc_flush = med([(record_diagnostics!(nw, sim.t; step = 0, costly = false, values = w.last);
                  @elapsed(flush(nw))) for _ in 1:50])      # block write + sync
nw.flush_interval = 5.0
tnc_costly = med([@elapsed(record_diagnostics!(nw, sim.t; step = 0, costly = true, values = w.last))
                  for _ in 1:100])
close(mon_default); close(mon_every); close(mon_csv)

base = med(res[:null])
@printf("64×64×32 step_adaptive!, %d thread(s), %d steps × %d repetitions (median of repetitions); mean dt %.2f s\n",
        Threads.nthreads(), n, NREP, dt_mean)
for k in (:null, :csv, :default, :every, :attached)
    v = res[k]
    @printf("  %-8s %8.3f ms/step   (%+.2f %% vs null; spread of repetitions %.2f–%.2f ms)\n",
            k, 1e3 * med(v), 100 * (med(v) / base - 1), 1e3 * minimum(v), 1e3 * maximum(v))
end
for k in (:csv, :default, :every, :attached)
    @printf("  %-8s paired: median over repetitions of (config / null − 1) = %+.2f %%\n", k,
            100 * med(res[k] ./ res[:null] .- 1))
end
@printf("  parts: cheap row %.1f µs, costly row %.3f ms, blowup check %.3f ms, snapshot %.3f ms\n",
        1e6 * tcheap, 1e3 * tcostly, 1e3 * tcheck, 1e3 * tsnap)
@printf("  step record bracket + hook call: %.0f ns per step\n", 1e9 * trec)
@printf("  diagnostics.nc: buffered row %.1f µs, block write + sync %.3f ms (every 5 s wall), costly row %.3f ms\n",
        1e6 * tnc_cheap, 1e3 * tnc_flush, 1e3 * tnc_costly)
per = trec + tcheap + tnc_cheap + (tcheck + tsnap) / 10 + (tcostly + tnc_costly) * dt_mean / 60 +
      tnc_flush * base / 5.0
@printf("  default cadence from the parts: %.3f ms/step = %.2f %% of a null step\n",
        1e3 * per, 100 * per / base)
# the budget of epic #293: at most about 2 % of a step at the default cadence
# (asserted on the parts, which are free of the step-time noise)
per / base <= 0.02 || error("run monitor over budget: $(round(100 * per / base; digits = 2)) % of a step > 2 %")
