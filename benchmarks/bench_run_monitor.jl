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
# + sync (every 5 s wall, with a buffered costly row since #514) and a
# costly row (the 5 profiles, buffered since #514); the step record's
# bracket with a no-op hook (#336); and the attached guard (#514): its
# start-of-step hook and end-of-step check on a step that is not due and on
# a due one after a verified `step_adaptive!` step (the sweep of `νt, κt,
# qv` and the kept rollback snapshot, instead of a full check and a copy).
#
# The budget (epic #293, at most about 2 % of a step) is asserted on the
# parts of the default configuration: the attached monitor of
# `with_monitor` and `aeolus run`. The parts of a monitor called with
# `monitor!` (the full check and copy every 10 steps) are printed too.
# Run it at several thread counts (`julia -t 4 ...`): the step is threaded,
# so are the guard's sweeps (#514).
#
# Measured (#514, 2026-11-06, this machine; master 89719c2b before it):
#   default (attached) from the parts: 0.57-0.58 % of a step on 1 thread,
#     0.46-0.55 % on 4, 0.47 % on 8 (master: 2.27 % on 1 thread, over budget);
#   a due check 0.068 ms (0.035-0.044 ms on 4-8 threads) against the full
#     check + copy of 0.44 ms;
#   per step, trimmed means (what a run pays): +1.1 to +1.5 % on 1 thread
#     (master +2.8 to +3.1 %); on 4 threads -1.2 to +4.6 % over 5 runs
#     (master +2.9 to +6.4 %): at 4-8 threads this measurement scatters by
#     2-3 points here, the parts do not.

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

# what a run pays, per step (#514): every step timed on its own, the null
# step and the attached default interleaved in blocks of 50; a trimmed mean
# (the top 5 % dropped: GC and the OS) is the mean a run pays. The parts
# below miss what a monitor costs the step around it (a due sweep reads
# fields the step has pushed out of the cache): this is about 2.5x them.
# A fresh simulation (the first one blows up after ~2930 steps).
sim2 = build()
mon2 = RunMonitor(sim2; csv = joinpath(dir, "p.csv"), nc = joinpath(dir, "p.nc"),
                  crash = joinpath(dir, "crash2"), t_end = 1e9)
function per_step!(v, s, k)
    for _ in 1:k
        t = time_ns(); step_adaptive!(s, CFL); push!(v, (time_ns() - t) / 1e9)
    end
end
tn_steps = Float64[]; ta_steps = Float64[]
with_logger(NullLogger()) do
    per_step!(Float64[], sim2, 5)
    for _ in 1:12
        per_step!(tn_steps, sim2, 50)
        attach_monitor!(mon2); per_step!(ta_steps, sim2, 50); detach_monitor!(mon2)
    end
end
close(mon2)
trim(v) = (w = sort(v); sum(w[1:floor(Int, 0.95 * length(w))]) / floor(Int, 0.95 * length(w)))

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
# a block write with one buffered costly row in it (#514)
tnc_flush_costly = med([(record_diagnostics!(nw, sim.t; step = 0, costly = true, values = w.last);
                         @elapsed(flush(nw))) for _ in 1:50])

# the attached guard (#514), in isolation: the start-of-step hook and the
# end-of-step check, the record set as a verified step_adaptive! step leaves it
const TI = Aeolus.TimeIntegration
ga = mon_att.guard
hk = Aeolus.IO.MonitorHook(mon_att)
function guard_call!(ga, hk, sim, rec, due)
    st = mon_att.step + 1
    ga.last_check = due ? st - ga.every_steps : st - 1
    snap = TI.state_snapshot(sim)                    # allocated after a claim (the first two only)
    t = @elapsed begin
        SIM.step_begun(hk, sim, rec)
        rec.entry_snap = snap; rec.exit_verified = true
        check_blowup!(ga, sim, rec; step = st)
    end
    return t
end
for _ in 1:4; guard_call!(ga, hk, sim, rec, true); end
tguard_due = med([guard_call!(ga, hk, sim, rec, true) for _ in 1:200])
tguard_idle = med([guard_call!(ga, hk, sim, rec, false) for _ in 1:200])
close(mon_default); close(mon_every); close(mon_csv); close(mon_att)

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
@printf("  per step, trimmed means: null %.3f ms, attached default %.3f ms (%+.2f %%)\n",
        1e3 * trim(tn_steps), 1e3 * trim(ta_steps), 100 * (trim(ta_steps) / trim(tn_steps) - 1))
@printf("  attached guard (#514): idle step %.2f µs, due check %.3f ms (full check + copy: %.3f ms)\n",
        1e6 * tguard_idle, 1e3 * tguard_due, 1e3 * (tcheck + tsnap))
@printf("  diagnostics.nc block write + sync with a costly row: %.3f ms\n", 1e3 * tnc_flush_costly)
rows = trec + tcheap + tnc_cheap + (tcostly + tnc_costly) * dt_mean / 60 + tnc_flush_costly * base / 5.0
per_manual = rows + (tcheck + tsnap) / 10
per = rows + tguard_idle + (tguard_due - tguard_idle) / 10
@printf("  monitor! cadence from the parts: %.3f ms/step = %.2f %% of a null step\n",
        1e3 * per_manual, 100 * per_manual / base)
@printf("  default (attached) cadence from the parts: %.3f ms/step = %.2f %% of a null step\n",
        1e3 * per, 100 * per / base)
# the budget of epic #293: at most about 2 % of a step at the default cadence
# (asserted on the parts, which are free of the step-time noise)
per / base <= 0.02 || error("run monitor over budget: $(round(100 * per / base; digits = 2)) % of a step > 2 %")
