# Cost of a run folder (#334): start_run / finish_run! once per run, and the
# per-step cost of running inside it, on a 64×64×32 Boussinesq case
# (step_adaptive!, the thread count Julia was started with).
#
# Run: julia --project=. benchmarks/bench_run_folder.jl
#
# Configurations, interleaved so drift hits them equally:
#   null    NullLogger, no run folder
#   folder  inside with_run (run.log installed, no records per step)
#   record  inside with_run, one six-key @debug record per step to run.log

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

function steps!(sim, n; record = false)
    t = @elapsed for i in 1:n
        _, dt, _ = step_adaptive!(sim, CFL)
        record && @debug("step", subsystem = :timestepping, domain = current_domain(),
                         step = i, dt = dt, t = sim.t, cfl = 0.5)
    end
    return t / n
end

repo = mktempdir()      # a clean repository, so the run is not blocked by this checkout's state
G = `git -c user.name=b -c user.email=b@b -c commit.gpgsign=false`
run(`git -C $repo init -q`); write(joinpath(repo, "f"), "1"); run(`$G -C $repo add f`)
run(`$G -C $repo commit -q -m x`)
root = mktempdir()

sim = build()
steps!(sim, 3)                                           # compile
with_run("warm"; root, repo, console = false) do rf
    steps!(sim, 2; record = true)
end

# start + finish of a run folder (git calls, provenance, run.toml, header)
tstart = minimum(1:5) do _
    @elapsed finish_run!(start_run("bench"; root, repo, console = false,
                                   config = (; N = (64, 64, 32)), seeds = 1))
end

n = 30
res = Dict(:null => Float64[], :folder => Float64[], :record => Float64[])
for rep in 1:9
    push!(res[:null], with_logger(() -> steps!(sim, n), NullLogger()))
    with_run("bench"; root, repo, console = false) do rf
        push!(res[:folder], steps!(sim, n))
        push!(res[:record], steps!(sim, n; record = true))
    end
end

med(v) = sort(v)[cld(length(v), 2)]
base = med(res[:null])
@printf("64×64×32 step_adaptive!, %d thread(s), %d steps × 9 repetitions (median of repetitions)\n",
        Threads.nthreads(), n)
for k in (:null, :folder, :record)
    v = res[k]
    @printf("  %-7s %8.3f ms/step   (%+.2f %% vs null; spread of repetitions %.2f–%.2f ms)\n",
            k, 1e3 * med(v), 100 * (med(v) / base - 1), 1e3 * minimum(v), 1e3 * maximum(v))
end
@printf("  start_run + finish_run!: %.1f ms per run (min of 5)\n", 1e3 * tstart)
