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
#   profile  (20 m rerun, the step's serial part) the warm-up, then the
#          trace above (`ntrace` steps, if the build has the hook), then
#          `nprof` (2) steps under Profile (sampling every `delay_ms`, 10).
#          Writes <out>.profile_main.txt (flat, thread 1 only: where the main
#          task, i.e. all serial code, spends its time), .profile_all.txt
#          (flat, all threads) and .profile_tree.txt (thread 1, tree), and
#          the raw samples (.profile.jls) for analysis elsewhere, and
#          .profile_threads.txt (inclusive counts for thread 1 awake / asleep
#          and the other threads awake, from the samples' metadata). If
#          Profile.print fails (the ENODEV of Fritz's /proc/sys/fs/binfmt_misc
#          seen with USR1), a fallback counts frames per sample itself.
#   memory (the 20 m rerun's growing RSS) the warm-up, a full GC and a
#          baseline (live heap, RSS, smaps_rollup), then `nmem` (4) steps with
#          RSS, live heap and allocated bytes after each, then a full GC (how
#          much of the growth was garbage); then one step under
#          Profile.Allocs (`sample_rate`, 0.01), summed by the innermost Aeolus
#          frame, by every Aeolus frame (inclusive) and by type, scaled by
#          1/sample_rate: <out>.memory.txt.
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

using TOML, Dates, Serialization, Statistics, InteractiveUtils, Profile
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
if mode in ("time", "trace", "profile", "memory")
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
if mode == "trace" || (mode == "profile" && HAS_TRACE) || (mode == "time" && get(kv, "trace", "0") == "1")
    HAS_TRACE || error("trace: this build has no launch trace (#566 hook)")
    B = Aeolus.Backends
    ntrace = fit(getint(mode == "trace" ? "nsteps" : "ntrace", mode == "profile" ? 2 : 10), HAS_FUSION ? 2 : 1)
    res["ntrace"] = ntrace
    for s in (HAS_FUSION ? (mode == "profile" ? (:if_possible,) : (:never, :if_possible)) : (:none,))
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
if mode == "profile"
    set_fusion!(:if_possible)
    nprof = getint("nprof", 2)
    Profile.init(n = 400_000_000, delay = parse(Float64, get(kv, "delay_ms", "10")) * 1e-3)
    Profile.clear()
    t0 = time()
    ts = Profile.@profile steps!(nprof)
    res["profile"] = Dict{String,Any}("ts_begin" => t0, "steps" => ts, "nprof" => nprof)
    base = replace(outpath, r"\.toml$" => "")
    data, lidict = Profile.retrieve()
    serialize(base * ".profile.jls", (data, lidict))
    wide = (:displaysize => (100_000, 300),)
    printed = String[]
    for (suffix, kw) in ((".profile_main.txt", (format = :flat, sortedby = :count, threads = 1, mincount = 10)),
                         (".profile_all.txt", (format = :flat, sortedby = :count, mincount = 50)),
                         (".profile_tree.txt", (format = :tree, threads = 1, mincount = 20, maxdepth = 60)))
        try
            open(base * suffix, "w") do io
                Profile.print(IOContext(io, wide...), data, lidict; C = false, kw...)
            end
            push!(printed, suffix)
        catch e
            @warn "Profile.print failed for $suffix" exception = (e, catch_backtrace())
        end
    end
    # fallback: inclusive counts per frame, all threads (a function, so the
    # counters are locals, not soft-scope globals)
    function profile_fallback(file)
        d0 = Profile.fetch(include_meta = false); ld = Profile.getdict(d0)
        cnt = Dict{String,Int}(); seen = Set{String}(); nsamp = 0
        for ip in d0
            if ip == 0
                for k in seen; cnt[k] = get(cnt, k, 0) + 1; end
                empty!(seen); nsamp += 1; continue
            end
            for fr in get(ld, ip, Base.StackTraces.StackFrame[])
                fr.from_c && continue
                push!(seen, "$(fr.func) $(fr.file):$(fr.line)")
            end
        end
        open(file, "w") do io
            println(io, "# samples (all threads) ", nsamp, "; inclusive count per frame")
            for (k, v) in sort!(collect(cnt); by = last, rev = true)[1:min(end, 400)]
                println(io, v, "\t", k)
            end
        end
    end
    # per thread and sleep state, from Profile's per-sample metadata (thread
    # id, sleep state; internals, so guarded): inclusive counts per frame for
    # thread 1 awake (the serial code and the main thread's share of
    # launches), thread 1 asleep (what it waits in), and the other threads
    # awake. Written always; Profile.print fails on Fritz anyway.
    function profile_by_thread(file)
        P = Profile
        data = Profile.fetch(); ld = Profile.getdict(data)
        groups = Dict("thread 1 awake" => Dict{String,Int}(), "thread 1 asleep" => Dict{String,Int}(),
                      "other threads awake" => Dict{String,Int}())
        nsamp = Dict(k => 0 for k in keys(groups))
        start = 1
        for i in eachindex(data)
            P.is_block_end(data, i) || continue
            tid = data[i - P.META_OFFSET_THREADID]
            asleep = data[i - P.META_OFFSET_SLEEPSTATE] == 2
            g = tid == 1 ? (asleep ? "thread 1 asleep" : "thread 1 awake") : (asleep ? nothing : "other threads awake")
            if g !== nothing
                nsamp[g] += 1
                seen = Set{String}()
                for ip in @view data[start:(i - P.nmeta - 2)]
                    for fr in get(ld, ip, Base.StackTraces.StackFrame[])
                        fr.from_c && continue
                        push!(seen, "$(fr.func) $(replace(string(fr.file), r".*/src/" => "src/")):$(fr.line)")
                    end
                end
                for k in seen; groups[g][k] = get(groups[g], k, 0) + 1; end
            end
            start = i + 1
        end
        open(file, "w") do io
            println(io, "# samples: ", join(["$k $(nsamp[k])" for k in sort!(collect(keys(nsamp)))], ", "))
            for g in ("thread 1 awake", "thread 1 asleep", "other threads awake")
                println(io, "\n## ", g, " (", nsamp[g], " samples): inclusive count per frame")
                for (k, v) in sort!(collect(groups[g]); by = last, rev = true)[1:min(end, 250)]
                    println(io, v, "\t", round(100v / max(nsamp[g], 1), digits = 1), "%\t", k)
                end
            end
        end
        return nsamp
    end
    try
        res["profile"]["samples"] = Dict(k => v for (k, v) in profile_by_thread(base * ".profile_threads.txt"))
    catch e
        @warn "profile by thread failed" exception = (e, catch_backtrace())
    end
    length(printed) < 3 && try
        profile_fallback(base * ".profile_fallback.txt")
    catch e
        @warn "profile fallback failed" exception = (e, catch_backtrace())
    end
    res["profile"]["printed"] = printed
end
if mode == "memory"
    set_fusion!(:if_possible)
    rss_gb() = parse(Int, split(only(filter(l -> startswith(l, "VmRSS:"), readlines("/proc/self/status"))))[2]) / 2^20
    function smaps()
        d = Dict{String,Float64}()
        for l in readlines("/proc/self/smaps_rollup")
            m = match(r"^(\w+):\s+(\d+) kB", l); m === nothing || (d[m[1]] = parse(Int, m[2]) / 2^20)
        end
        d
    end
    snap(tag) = Dict{String,Any}("tag" => tag, "rss_gb" => rss_gb(), "live_gb" => Base.gc_live_bytes() / 2^30,
                                 "allocd_total_gb" => Base.gc_bytes() / 2^30, "smaps_gb" => smaps(), "t" => time())
    GC.gc(true); GC.gc(true)
    snaps = [snap("after warm-up + full GC")]
    nmem = getint("nmem", 4)
    stepinfo = Dict{String,Any}[]
    for i in 1:nmem
        tm = @timed Aeolus.step!(model, dt)
        push!(stepinfo, Dict{String,Any}("s" => tm.time, "alloc_gb" => tm.bytes / 2^30, "gc_s" => tm.gctime))
        push!(snaps, snap("after step $i"))
    end
    GC.gc(true); GC.gc(true)
    push!(snaps, snap("after full GC"))
    res["memory"] = Dict{String,Any}("snapshots" => snaps, "steps" => stepinfo)
    rate = parse(Float64, get(kv, "sample_rate", "0.01"))
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=rate Aeolus.step!(model, dt)
    pr = Profile.Allocs.fetch()
    isaeolus(fr) = occursin("Aeolus", string(fr.file)) && !occursin("benchmarks/hpc", string(fr.file))
    inner = Dict{String,Float64}(); incl = Dict{String,Float64}(); bytype = Dict{String,Float64}(); cnt = Dict{String,Int}()
    for a in pr.allocs
        sz = a.size / rate
        bytype[string(a.type)] = get(bytype, string(a.type), 0.0) + sz
        frs = [fr for fr in a.stacktrace if isaeolus(fr)]
        key(fr) = "$(fr.func) $(replace(string(fr.file), r".*/src/" => "src/")):$(fr.line)"
        k = isempty(frs) ? "(no Aeolus frame)" : key(first(frs))
        inner[k] = get(inner, k, 0.0) + sz; cnt[k] = get(cnt, k, 0) + 1
        for k2 in unique(key.(frs)); incl[k2] = get(incl, k2, 0.0) + sz; end
    end
    total = sum(values(inner); init = 0.0)
    res["memory"]["allocs_sampled"] = length(pr.allocs)
    res["memory"]["allocs_est_gb"] = total / 2^30
    open(replace(outpath, r"\.toml$" => ".memory.txt"), "w") do io
        println(io, "# threads $(Threads.nthreads()), case $(get(kv, "case", "")), sample_rate $rate, sampled allocations $(length(pr.allocs))")
        println(io, "# estimated bytes allocated in one step: $(round(total / 2^30, digits = 3)) GiB")
        println(io, "\n## snapshots (GiB)")
        for sn in snaps
            sm = sn["smaps_gb"]
            println(io, rpad(sn["tag"], 26), " rss ", round(sn["rss_gb"], digits = 2), "  live ", round(sn["live_gb"], digits = 2),
                    "  allocated so far ", round(sn["allocd_total_gb"], digits = 2),
                    "  anon ", round(get(sm, "Anonymous", NaN), digits = 2), "  file ", round(get(sm, "Rss", NaN) - get(sm, "Anonymous", 0.0), digits = 2))
        end
        println(io, "\n## steps"); for (i, x) in enumerate(stepinfo); println(io, i, "  ", round(x["s"], digits = 2), " s  ", round(x["alloc_gb"], digits = 3), " GiB allocated  gc ", round(x["gc_s"], digits = 2), " s"); end
        for (title, d) in (("innermost Aeolus frame", inner), ("every Aeolus frame (inclusive)", incl), ("type", bytype))
            println(io, "\n## by $title (MiB per step, est.)")
            for (k, v) in sort!(collect(d); by = last, rev = true)[1:min(end, 60)]
                println(io, lpad(round(v / 2^20, digits = 1), 10), "  ", title == "innermost Aeolus frame" ? "n=$(cnt[k]) " : "", k)
            end
        end
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
mode in ("time", "trace", "profile", "memory", "state") || error("unknown mode $mode")

res["finished"] = string(now())
res["t_total_s"] = time() - T_START
mkpath(dirname(abspath(outpath)))
open(io -> TOML.print(io, res; sorted = true), outpath, "w")
println("bench_run: wrote $outpath")
