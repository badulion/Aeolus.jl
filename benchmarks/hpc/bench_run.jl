# The run driver of the HPC benchmark (#566, benchmarks/hpc/HANDOFF.md
# section 5). One process, one build (the active project), one case, one
# thread count; it writes one TOML.
#
#   julia -t N --project=<build> bench_run.jl <mode> <config.toml> <dt> <out.toml> [key=value ...]
#
# modes
#   time   warm up `nwarm` (5) steps, then `rounds` (4) rounds of `nsteps` (10)
#          steps with loop_fusion = "never" and "if_possible" alternating, the
#          order swapped every round. A build without the loop_fusion setting
#          (master) times rounds × 2 × nsteps (80) steps instead. Per step its
#          wall time; around the measured steps @timed (GC, allocation) and
#          /proc/self/task/*/schedstat (on-CPU and run-queue wait per thread).
#   trace  the same warm-up, then `nsteps` steps per setting with the launch
#          trace on (Backends.set_launch_trace!, the #566 hook): busy per
#          thread, idle inside launches, serial time between launches. The
#          traced path runs copies of the kernels, so 2 steps per setting
#          with the trace on are run and discarded first (their compilation).
#          `time` with key trace=1 runs this after its timed steps in the same
#          process (one model build less; the trace's cost stays outside the
#          timed blocks).
#   state  `nsteps` (20) steps with `setting` (if_possible), then u, v, w, θ, e
#          written next to <out.toml> (Serialization, interior only), for the
#          sanity check against bench/master (HANDOFF section 3, item 5).
#
# keys: nwarm, nsteps, rounds, setting, build, case, label (copied into the TOML),
# budget (seconds of measured stepping, default 600): at low thread counts a
# step can take tens of seconds, so after 2 warm-up steps the warm-up stops
# when a step takes over 5 s, and nsteps shrinks (to at least 2) so that
# rounds × 2 × nsteps steps fit the budget. The TOML records what ran.
# Timestamps (`ts_*`, Unix seconds) bracket the measured steps, so a counter
# timeline taken around the process (likwid-perfctr -t) can be cut to them.

using TOML, Dates, Serialization, Statistics, InteractiveUtils
const T_START = time()

mode, cfgpath, dtstr, outpath = ARGS[1], ARGS[2], ARGS[3], ARGS[4]
kv = Dict{String,String}(split(a, "="; limit = 2)[1] => split(a, "="; limit = 2)[2] for a in ARGS[5:end])
getint(k, d) = parse(Int, get(kv, k, string(d)))
dt = parse(Float64, dtstr)

using Aeolus
const T_LOADED = time()

# ---- the machine and the process -------------------------------------------

function schedstat()
    d = Dict{Int,NTuple{3,Int}}()
    for t in readdir("/proc/self/task")
        f = "/proc/self/task/$t/schedstat"
        isfile(f) || continue
        v = parse.(Int, split(read(f, String)))
        d[parse(Int, t)] = (v[1], v[2], v[3])          # on-CPU ns, run-queue wait ns, timeslices
    end
    return d
end
function schedstat_delta(a, b)
    tids = sort!(collect(keys(b)))
    on = Int[]; wait = Int[]; slices = Int[]
    for t in tids
        x = get(a, t, (0, 0, 0)); y = b[t]
        push!(on, y[1] - x[1]); push!(wait, y[2] - x[2]); push!(slices, y[3] - x[3])
    end
    return Dict("tid" => tids, "on_cpu_s" => on .* 1e-9, "runqueue_wait_s" => wait .* 1e-9,
                "timeslices" => slices)
end
function affinity()
    out = Dict{String,String}()
    for t in readdir("/proc/self/task")
        for l in eachline("/proc/self/task/$t/status")
            startswith(l, "Cpus_allowed_list:") && (out[t] = strip(split(l, ':')[2]))
        end
    end
    return out
end
# the CPU each Julia thread runs on, sampled from every thread
function thread_cpus()
    cpus = zeros(Int, Threads.nthreads())
    Threads.@threads :static for i in 1:Threads.nthreads()
        cpus[i] = ccall(:sched_getcpu, Cint, ())
    end
    return cpus
end

# ---- the model ----------------------------------------------------------------

const SIMS = Aeolus.Simulations
const HAS_FUSION = isdefined(SIMS, :set_loop_fusion!)
const HAS_TRACE = isdefined(Aeolus.Backends, :set_launch_trace!)

cfg = Aeolus.load_run_config(cfgpath; overrides = ["fftw.planner=\"estimate\""])
# Config.build does not apply [fftw] (the CLI run does, Config.apply_fftw_config!):
# without this the override above and the config's [fftw] threads were ignored,
# and the planner came from AEOLUS_FFTW_PLANNER (env.sh: estimate) — so run 1
# and 20 m attempt 1 planned with estimate, but with the default FFTW thread
# count, not their configs' `threads = 12`. The policy used goes into the TOML.
Aeolus.Config.apply_fftw_config!(cfg)
const BUILD = @timed Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model
model = BUILD.value
const T_BUILT = time()
const FFTW_USED = let p = Aeolus.Operators.fftw_policy(); (planner = string(p.planner), threads = p.threads) end

set_fusion!(s) = HAS_FUSION && foreach(d -> SIMS.set_loop_fusion!(d.state, s), model.domains)
function steps!(n)
    ts = Float64[]
    for _ in 1:n
        push!(ts, @elapsed(Aeolus.step!(model, dt)))
    end
    return ts
end

res = Dict{String,Any}(
    "mode" => mode, "config" => abspath(cfgpath), "dt" => dt,
    "build" => get(kv, "build", ""), "case" => get(kv, "case", ""), "label" => get(kv, "label", ""),
    "commit" => strip(read(`git -C $(pkgdir(Aeolus)) rev-parse HEAD`, String)),
    "aeolus_dir" => pkgdir(Aeolus),
    "threads" => Threads.nthreads(), "julia" => string(VERSION),
    "host" => gethostname(), "slurm_job" => get(ENV, "SLURM_JOB_ID", ""),
    "julia_exclusive" => get(ENV, "JULIA_EXCLUSIVE", ""), "numactl" => get(ENV, "BENCH_NUMACTL", ""),
    "has_loop_fusion" => HAS_FUSION, "has_launch_trace" => HAS_TRACE,
    "started" => string(now()),
    "t_load_s" => T_LOADED - T_START, "t_build_s" => T_BUILT - T_LOADED,
    "build_gc_s" => BUILD.gctime, "build_alloc_bytes" => BUILD.bytes,
    "affinity" => affinity(), "thread_cpus" => thread_cpus(),
    "fftw_planner" => FFTW_USED.planner, "fftw_threads" => FFTW_USED.threads,
)

nwarm = getint("nwarm", 5)
budget = parse(Float64, get(kv, "budget", "600"))
if mode in ("time", "trace")
    set_fusion!(:if_possible)
    w = steps!(min(2, nwarm))
    last(w) <= 5 && append!(w, steps!(max(0, nwarm - 2)))
    res["warm_s"] = w
    tstep = last(w)
    fit(n, per) = clamp(floor(Int, budget / (per * tstep)), min(2, n), n)
    res["budget_s"] = budget
end

if mode == "time"
    rounds = getint("rounds", 4)
    nsteps = fit(getint("nsteps", 10), 2rounds)
    res["rounds"] = rounds; res["nsteps"] = nsteps
    times = Dict{String,Vector{Float64}}("never" => Float64[], "if_possible" => Float64[], "none" => Float64[])
    blocks = Dict{String,Any}[]          # each block of steps: setting and Unix start/end (counter timelines)
    block!(s, n) = (t0 = time(); append!(times[s], steps!(n)); push!(blocks, Dict("setting" => s, "ts_begin" => t0, "ts_end" => time())))
    GC.gc()
    s0 = schedstat(); res["ts_begin"] = time()
    tm = @timed begin
        if HAS_FUSION
            for r in 1:rounds
                order = isodd(r) ? (:never, :if_possible) : (:if_possible, :never)
                for s in order
                    set_fusion!(s)
                    block!(string(s), nsteps)
                end
            end
        else
            block!("none", rounds * 2 * nsteps)
        end
    end
    res["ts_end"] = time(); s1 = schedstat()
    for (k, v) in times
        isempty(v) && continue
        res["steps_$k"] = v
        res["step_min_$k"] = minimum(v)
        res["step_median_$k"] = median(v)
    end
    res["blocks"] = blocks
    res["measured_wall_s"] = tm.time
    res["gc_s"] = tm.gctime
    res["alloc_bytes"] = tm.bytes
    res["gc_count"] = tm.gcstats.pause
    res["schedstat"] = schedstat_delta(s0, s1)
end
if mode == "trace" || (mode == "time" && get(kv, "trace", "0") == "1")
    HAS_TRACE || error("trace: this build has no launch trace (#566 hook)")
    B = Aeolus.Backends
    ntrace = fit(getint(mode == "trace" ? "nsteps" : "ntrace", 10), HAS_FUSION ? 2 : 1)
    res["ntrace"] = ntrace
    for s in (HAS_FUSION ? (:never, :if_possible) : (:none,))
        set_fusion!(s)
        B.set_launch_trace!(true); steps!(2); B.set_launch_trace!(false)   # the traced copies compiled
        B.reset_launch_trace!(); B.set_launch_trace!(true)
        t0 = time()
        ts = steps!(ntrace)
        B.set_launch_trace!(false)
        rs = B.launch_trace()
        sm = B.launch_trace_summary(rs)
        wall = sum(ts)
        res["trace_$s"] = Dict{String,Any}(
            "ts_begin" => t0, "steps" => ts, "step_wall_s" => wall,
            "launch_wall_s" => sm.launch_wall, "serial_between_s" => wall - sm.launch_wall,
            "idle_in_launches_s" => sm.idle_in_launches, "busy_per_thread_s" => collect(sm.busy),
            "launches" => sm.launches, "nested" => sm.nested,
            # per launch name: count, wall, busy summed over threads (top level only)
            "by_name" => let d = Dict{String,Any}()
                for r in rs
                    r.depth == 0 || continue
                    e = get!(() -> Dict{String,Any}("count" => 0, "wall_s" => 0.0, "busy_s" => 0.0), d, string(r.name))
                    e["count"] += 1; e["wall_s"] += (r.t1 - r.t0) * 1e-9; e["busy_s"] += sum(r.busy; init = 0) * 1e-9
                end
                d
            end)
    end
end
if mode == "state"
    s = Symbol(get(kv, "setting", "if_possible"))
    s === :never || s === :if_possible || error("state: setting must be never or if_possible")
    set_fusion!(s)
    nsteps = getint("nsteps", 20)
    res["steps"] = steps!(nsteps)
    d = first(model.domains)
    st = d.state
    iv(f) = Array(Aeolus.Fields.interior_view(f))
    fields = Dict{String,Array}("u" => iv(st.u), "v" => iv(st.v), "w" => iv(st.w), "theta" => iv(st.θ))
    c = Aeolus.Components.sgs_closure(d)
    c !== nothing && hasproperty(c, :e) && (fields["e"] = iv(c.e))
    statefile = replace(outpath, r"\.toml$" => ".state.jls")
    serialize(statefile, fields)
    res["state_file"] = statefile
    res["state_fields"] = sort!(collect(keys(fields)))
end
mode in ("time", "trace", "state") || error("unknown mode $mode")

res["finished"] = string(now())
res["t_total_s"] = time() - T_START
mkpath(dirname(abspath(outpath)))
open(io -> TOML.print(io, res; sorted = true), outpath, "w")
println("bench_run: wrote $outpath")
