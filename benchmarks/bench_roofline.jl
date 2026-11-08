# SC-a (#401): a roofline profile of the step, kernel by kernel, and the
# thread-pinning experiment. One process per thread count (and per
# pinning setting); docs/src/design/ka_port.md, "SC-a", has the tables.
#
#   julia -t N --project=. benchmarks/bench_roofline.jl stream
#   julia -t N --project=. benchmarks/bench_roofline.jl flagship [nwarm=2] [nsteps=9]
#   julia -t N --project=. benchmarks/bench_roofline.jl dns      [N=256] [nwarm=1] [nsteps=3]
#   julia -t N --project=. benchmarks/bench_roofline.jl config   <toml> [nwarm=2] [nsteps=6] [--set k=v ...]
#   julia --project=. benchmarks/bench_roofline.jl merge  case ref_threads log1 log2 ...
#
# stream: STREAM copy and triad (Threads.@threads :static, first touch by
#   the same threads) on arrays of 256 MB (DRAM) and of 8 MB (three of
#   them, 24 MB, stay in the 2 × 32 MB of L3 but not in the 12 MB of L2),
#   plus the cost of an empty threaded launch (Backends.launch! with
#   threshold 0, 8 rows of 64). The flagship and dns modes run the same measurements first, in the same process,
#   so every kernel is compared with the bandwidth of its own thread count.
# flagship: the Berlin case `examples/urban/berlin_summer.toml` (the 64³ child nested in
#   its 192 × 192 × 53 parent, the #265 default), `nwarm` steps, then
#   `nsteps` timed steps (at most 11 steps in all: #392, and long Berlin
#   runs are paused).
# config: any configured run (a TOML file, e.g. the dry CBL, GABLS1, the LES
#   of examples/dns_vs_les), with `--set` overrides (the grid size,
#   `timestep.loop_fusion="never"` for the reference of #533): `nwarm` steps,
#   then `nsteps` timed steps (`step_adaptive!(model)`, or `step!(model, dt)` of a
#   fixed-step run) (#533 step 4).
#   The case name in the ROW lines is the file's base name.
# dns: `step_rk3!` of the 256³ decaying-turbulence DNS
#   (test/fixtures/dns_decay.jl, as in KA-o's `dns` mode). The per-step
#   history of `dns_step!` (on the reduction primitives since #400) is
#   not profiled; with ROOFLINE_PROF=0 its time is printed for reference. Needs about 8 GB at 256³; run with
#   AEOLUS_FFTW_PLANNER=estimate (#397).
#
# The profile. Every launch goes through one of three functions of
# Backends: `_launch!` (launch!, launch_weighted!, range_map!,
# foreach_column!, ...), `_plane_partials!` (every reduction: plane_*,
# range_max, interior_*, the PCG dots) and `_serial_column_sum` (the serial
# path of column_sum). With ROOFLINE_PROF unset or 1, this script replaces
# them, before anything is compiled, by copies that time each call
# (time_ns) and record it under the name of the kernel body. Per kernel:
# calls per step, the fraction of calls that ran serially (below the
# backend's threshold), µs per call, the points per call and the bytes per
# call. The bytes are the compulsory traffic: every distinct numeric array
# reachable from the body and its arguments (through tuples and structs, 3
# levels deep) counted once, as min(its size, points × weight × element
# size) (a launch over the interior streams the interior, not the halo).
# An array counts once whether it is read, written or both (axpy's `y`
# counts once, so an axpy is 2 arrays where STREAM's triad is 3): this is
# the compulsory footprint, a lower bound on the traffic. It is a model: a stencil that misses cache reads more, a kernel that
# reads only part of an array (a boundary row of a 3-D field) less.
# Time outside the three functions (FFTW, BLAS, the remaining serial code,
# allocation) is reported as "outside launches". A body launched over
# 10⁶ points or more (the nest parent's 192 × 192 × 53 fields, the DNS) is
# recorded apart, with the suffix `[≥1M]`: its arrays do not fit in L3.
# ROOFLINE_PROF=0 times the plain step only (the pinning experiment).
# The profiler's own time (naming the kernel, walking its arguments; most
# of it compiling in the first recorded step) is measured and taken out of
# the profiled step: of the mean over the steps (flagship) or of each step
# before the minimum is taken (dns). The profiled step still runs 0-15 %
# above the plain one (the profiler's work evicts cache), so the tables
# quote the plain step of a ROOFLINE_PROF=0 run and the in-launch times of
# a profiled one.
#
# merge: reads the KROW/ROW lines of several runs of one case, profiled
# and plain (any number of processes per thread count: per kernel the call
# with the smallest µs/call is kept, the step and launch times are minima,
# the bandwidths maxima) and prints the roofline table of ka_port.md,
# ranked by ms per step at `ref_threads`, with the class of each kernel
# (`kclass`): serial (ran below the backend's threshold), bandwidth (its
# bytes at the DRAM triad bandwidth take at least half its time), compute
# (its one-thread time over the threads takes at least half), else
# overhead (launch, synchronisation, imbalance). It also sums the launches
# below and above 10⁶ points, serial and threaded.
#
# Every run prints its threads' CPU affinity, so a pinning setting
# (JULIA_EXCLUSIVE=1, taskset) can be checked in the log.

const MODE = get(ARGS, 1, "stream")
const PROF = MODE in ("flagship", "dns", "config") && get(ENV, "ROOFLINE_PROF", "1") != "0"

using Printf

# ---------------------------------------------------------------------------
# merge (no Aeolus needed)
# ---------------------------------------------------------------------------

# the class of a kernel at `ref` threads (see the header). The roof is the
# DRAM triad bandwidth for every kernel: the 64³ child's fields would fit
# in L3, but between two uses of a field the step touches more than L3
# holds, and its threaded launches stream at about DRAM speed in all
# (`merge` prints the sums). `excess` is the time per call that neither
# the compulsory bytes at the DRAM roof nor the one-thread time over `ref`
# threads explains (launch, synchronisation, imbalance, cache misses
# beyond the compulsory traffic).
function kclass(serial, us, us1, bytes, ref, bw)
    t_bw = bytes / bw / 1e3              # µs at the DRAM roof
    excess = us - max(t_bw, us1 / ref)
    serial > 0.5 && return ("serial", excess)
    t_bw >= 0.5us && return ("bandwidth", excess)
    us1 / ref >= 0.5us && return ("compute", excess)
    return ("overhead", excess)
end

function merge_logs(case, ref, files)
    k = Dict{Tuple{String,Int},NTuple{5,Float64}}()   # (name, nt) => calls, serial, us, bytes, points
    r = Dict{Tuple{String,Int},Float64}()
    for f in files
        lines = readlines(f)
        prof = any(l -> startswith(l, "KROW"), lines)
        for l in lines
            w = split(l)
            isempty(w) && continue
            if w[1] == "KROW" && w[2] == case
                key, v = (w[4], parse(Int, w[3])), Tuple(parse.(Float64, w[5:9]))
                (!haskey(k, key) || v[3] < k[key][3]) && (k[key] = v)
            elseif w[1] == "ROW" && w[2] == case
                # a profiled run's step (profiler's time taken out) is kept
                # apart from the plain step of a ROOFLINE_PROF=0 run
                name = prof && w[4] == "step" ? "step_prof" : String(w[4])
                key, v = (name, parse(Int, w[3])), parse(Float64, w[5])
                r[key] = !haskey(r, key) ? v : startswith(name, "stream") ? max(r[key], v) : min(r[key], v)
            end
        end
    end
    nts = sort!(unique(last.(keys(r))))
    bw = r[("stream_triad_dram", ref)]
    bwl3 = r[("stream_triad_l3", ref)]
    tl = r[("launch_empty", ref)]
    println("case $case, bandwidth at -t $ref: DRAM triad $(round(bw; digits = 1)) GB/s, ",
            "L3 triad $(round(bwl3; digits = 1)) GB/s, empty threaded launch $(round(tl; digits = 2)) µs")
    for nt in nts
        st = get(r, ("step", nt), NaN)
        @printf("  -t %-2d step %9.2f ms (profiled %9.2f), in launches %9.2f ms, outside launches %9.2f ms\n", nt,
                st, get(r, ("step_prof", nt), NaN), get(r, ("in_launches", nt), NaN),
                st - get(r, ("in_launches", nt), NaN))
    end
    names = unique(first.(keys(k)))
    ms(n, nt) = haskey(k, (n, nt)) ? k[(n, nt)][1] * k[(n, nt)][3] / 1e3 : 0.0
    sort!(names; by = n -> -ms(n, ref))
    println()
    println("| kernel | calls/step | serial | MB/call | µs/call 1 t | µs/call $ref t | speed-up | GB/s $ref t | % DRAM | % L3 | excess µs | class | ms/step $ref t |")
    println("| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | --: |")
    tot = sum(n -> ms(n, ref), names)
    cls = Dict{String,Int}()
    clsms = Dict{String,Float64}()
    grp = Dict{String,NTuple{2,Float64}}()       # launches below / above 10⁶ points, threaded / serial: ms, GB per step
    for n in names
        haskey(k, (n, ref)) || continue
        c, s, us, b, p = k[(n, ref)]
        us1 = haskey(k, (n, 1)) ? k[(n, 1)][3] : NaN
        gbs = b / us / 1e3
        class, ex = kclass(s, us, us1, b, ref, bw)
        big = (p >= 1_000_000 ? "≥1M points" : "<1M points") * (s > 0.5 ? ", serial" : ", threaded")
        grp[big] = get(grp, big, (0.0, 0.0)) .+ (ms(n, ref), c * b / 1e9)
        cls[class] = get(cls, class, 0) + 1
        clsms[class] = get(clsms, class, 0.0) + ms(n, ref)
        @printf("| `%s` | %.3g | %.0f %% | %.2f | %.1f | %.1f | %.2f× | %.1f | %.0f %% | %.0f %% | %s | %s | %.3f |\n",
                n, c, 100s, b / 1e6, us1, us, us1 / us, gbs, 100gbs / bw, 100gbs / bwl3,
                class == "serial" ? "–" : @sprintf("%.0f", max(ex, 0.0)), class, ms(n, ref))
    end
    println()
    @printf("in launches at -t %d: %.2f ms/step over %d kernels\n", ref, tot, length(names))
    for (g, (t, gb)) in sort!(collect(grp); by = first)
        @printf("launches of %s: %.2f ms/step, %.3f GB/step (model): %.1f GB/s, %.0f %% of the DRAM triad\n",
                g, t, gb, gb / t * 1e3, 100gb / t * 1e3 / bw)
    end
    for (c, n) in sort!(collect(cls); by = first)
        @printf("  %-10s %3d kernels  %8.2f ms/step (%.0f %%)\n", c, n, clsms[c], 100clsms[c] / tot)
    end
end

if MODE == "merge"
    merge_logs(ARGS[2], parse(Int, ARGS[3]), ARGS[4:end])
    exit()
end

# ---------------------------------------------------------------------------
# the instrumented launch paths
# ---------------------------------------------------------------------------

using Aeolus
using Aeolus.Backends: CPUBackend, launch!, init_threads!
using Random

const NT = Threads.nthreads()

struct KStat
    calls::Int
    serial::Int
    ns::Int
    bytes::Float64
    points::Float64
end
const STATS = Dict{String,KStat}()
const STATS_LOCK = ReentrantLock()
const RECORDING = Ref(false)

const NAMES = Dict{Any,String}()
_fname(f) = get!(NAMES, typeof(f)) do
    f isa Function || return string(nameof(typeof(f)))
    s = string(nameof(f))
    m = first(methods(f))
    startswith(s, '#') ? "$(m.name)@$(basename(string(m.file))):$(m.line)" : s
end
# range_map! launches one generic body: name it by the mapped function
_kname(body!, args) = body! === Aeolus.Backends._map_body! ? "map:" * _fname(args[1]) : _fname(body!)

# compulsory bytes of the distinct numeric arrays reachable from x
function _arrays!(seen, @nospecialize(x), depth)
    if x isa AbstractArray
        if isbitstype(eltype(x)) && eltype(x) <: Union{Number,Tuple}
            id = x isa StridedArray ? UInt(pointer(x)) : objectid(x)
            haskey(seen, id) || (seen[id] = (length(x), sizeof(eltype(x))))
        end
        return
    end
    depth <= 0 && return
    (x isa Module || x isa Type || x isa Function && isempty(fieldnames(typeof(x)))) && return
    isbits(x) && return
    for i in 1:nfields(x)
        isdefined(x, i) && _arrays!(seen, getfield(x, i), depth - 1)
    end
end
# cached per (arguments, points): walking the arguments costs ~100 µs
const BYTES = Dict{UInt,Int}()
_bytes(points, @nospecialize(objs...)) = get!(() -> _walk_bytes(points, objs...), BYTES, hash(objectid(objs), UInt(points)))
function _walk_bytes(points, @nospecialize(objs...))
    seen = Dict{UInt,Tuple{Int,Int}}()
    for o in objs
        _arrays!(seen, o, 3)
    end
    return sum(((n, s),) -> min(n, points) * s, values(seen); init = 0)
end

const GC_NS = Ref(0)           # GC time in the timed steps
const OVERHEAD_NS = Ref(0)     # time spent in record! itself (excluded from the step's split)
function record!(@nospecialize(prefix), @nospecialize(f), @nospecialize(args), serial::Bool, ns, points,
                 @nospecialize(objs...))
    RECORDING[] || return nothing
    t0 = time_ns(); g0 = Base.gc_time_ns()
    name = prefix * _kname(f, args) * (points >= 1_000_000 ? "[≥1M]" : "")
    b = _bytes(points, objs...)
    lock(STATS_LOCK) do
        s = get(STATS, name, KStat(0, 0, 0, 0.0, 0.0))
        STATS[name] = KStat(s.calls + 1, s.serial + serial, s.ns + ns, s.bytes + b, s.points + points)
        OVERHEAD_NS[] += (time_ns() - t0) - (Base.gc_time_ns() - g0)   # a GC pause is the step's
    end
    return nothing
end

if PROF
    @eval Aeolus.Backends begin
        function _launch!(b::CPUBackend, body!::F, ranges::NTuple{N,UnitRange{Int}}, args::A,
                          weight::Int = 1) where {F, N, A<:Tuple}
            np = _npoints(ranges)
            np == 0 && return nothing
            t0 = time_ns()
            s = runs_serially(b, np * weight)
            if s
                _serial!(body!, ranges, args)
            else
                _threaded!(b, body!, ranges, args, weight)
            end
            t1 = time_ns()
            Main.record!("", body!, args, s, Int(t1 - t0), np * weight, body!, args)
            return nothing
        end
        function _plane_partials!(b::CPUBackend, out::AbstractVector,
                                  p::Union{_PlaneReduce,_PlaneMax,_PlaneFn}, kr::UnitRange{Int}, g)
            length(out) == length(kr) ||
                throw(DimensionMismatch("partials has length $(length(out)), the k range $(length(kr))"))
            isempty(kr) && return out
            t0 = time_ns()
            np = length(p.ir) * length(p.jr) * length(kr)
            s = runs_serially(b, np) || length(kr) == 1
            if s
                for (c, k) in enumerate(kr)
                    out[c] = p(k, g)
                end
            else
                nk = length(kr)
                nt = _ntasks(np)
                kern = _planes_kernel!(b.ka, nt >= Threads.nthreads() ? 1 : cld(nk, nt))
                kern(out, p, first(kr) - 1, (g,); ndrange = length(kr))
            end
            t1 = time_ns()
            f = p isa _PlaneFn ? p.f : g isa _Ix ? g.f : g isa _Vals ? g.f : p
            Main.record!("reduce:", f, (), s, Int(t1 - t0), np, p, g)
            return out
        end
        @noinline function _serial_column_sum(body::F, (ir, jr)::NTuple{2,UnitRange{Int}}, args, s) where {F}
            t0 = time_ns()
            for j in jr, i in ir
                s = _add_terms(s, body(i, j, args...))
            end
            Main.record!("colsum:", body, (), true, Int(time_ns() - t0),
                         length(ir) * length(jr), body, args)
            return s
        end
        @noinline function _serial_column_sum(body::F, (nr,)::NTuple{1,UnitRange{Int}}, args, s) where {F}
            t0 = time_ns()
            for n in nr
                s = _add_terms(s, body(n, args...))
            end
            Main.record!("colsum:", body, (), true, Int(time_ns() - t0), length(nr), body, args)
            return s
        end
    end
end

# ---------------------------------------------------------------------------
# machine: affinity, STREAM, launch cost
# ---------------------------------------------------------------------------

row(case, name, v) = @printf("ROW %s %d %s %.6g\n", case, NT, name, v)

function affinity()
    sets = String[]
    for t in readdir("/proc/self/task")
        st = read("/proc/self/task/$t/status", String)
        m = match(r"Cpus_allowed_list:\s*(\S+)", st)
        m === nothing || push!(sets, m.captures[1])
    end
    # the CPU each Julia thread runs on now (pinned threads stay put)
    cpus = zeros(Int, Threads.nthreads())
    Threads.@threads :static for i in 1:Threads.nthreads()
        cpus[Threads.threadid() - Threads.nthreads(:interactive)] = ccall(:sched_getcpu, Cint, ())
    end
    @printf("threads: %d default + %d interactive, JULIA_EXCLUSIVE=%s; OS-thread affinities: %s; default threads on CPUs %s\n",
            Threads.nthreads(), Threads.nthreads(:interactive), get(ENV, "JULIA_EXCLUSIVE", "0"),
            join(sort!(unique(sets)), " | "), join(cpus, ","))
end

function stream(case)
    for (tag, n) in (("dram", 2^25), ("l3", 2^20))
        a = Vector{Float64}(undef, n); b = similar(a); c = similar(a)
        Threads.@threads :static for i in 1:n
            @inbounds a[i] = 1.0; b[i] = 2.0; c[i] = 0.0
        end
        copy!(a, b, c) = Threads.@threads :static for i in eachindex(a)
            @inbounds a[i] = b[i]
        end
        triad!(a, b, c) = Threads.@threads :static for i in eachindex(a)
            @inbounds a[i] = b[i] + 3.0 * c[i]
        end
        nrep = n > 10^6 ? 10 : 2000
        for (nm, f!, narr) in (("copy", copy!, 2), ("triad", triad!, 3))
            f!(a, b, c)
            t = Inf
            for _ in 1:nrep
                t = min(t, @elapsed f!(a, b, c))
            end
            gbs = narr * 8 * n / t / 1e9
            @printf("STREAM %-5s %-4s %8.1f GB/s\n", nm, tag, gbs)
            row(case, "stream_$(nm)_$(tag)", gbs)
        end
    end
    bt = CPUBackend(threshold = 0)
    z = zeros(64, 8, 1)
    noop(i, j, k, a) = nothing
    launch!(bt, noop, (1:64, 1:8, 1:1), z)
    t = Inf
    for _ in 1:2000
        t = min(t, @elapsed launch!(bt, noop, (1:64, 1:8, 1:1), z))
    end
    @printf("empty threaded launch (8 rows of 64): %.2f µs\n", 1e6t)
    row(case, "launch_empty", 1e6t)
end

function dump_stats(case, nsteps, tstep; step_overhead_ms = nothing)
    tin = 0.0
    for (n, s) in sort!(collect(STATS); by = x -> -x[2].ns)
        @printf("KROW %s %d %s %.4g %.4g %.6g %.6g %.6g\n", case, NT, n, s.calls / nsteps,
                s.serial / s.calls, s.ns / s.calls / 1e3, s.bytes / s.calls, s.points / s.calls)
        tin += s.ns / 1e6 / nsteps
    end
    # the profiler's own cost (naming, walking the arguments; most of it
    # compiling in the first recorded step) is taken out of the step: `step`
    # is the step without it. The DNS takes it out step by step already.
    ov = something(step_overhead_ms, OVERHEAD_NS[] / 1e6 / nsteps)
    step_overhead_ms === nothing && (tstep -= ov)
    PROF && row(case, "in_launches", tin)
    row(case, "gc", GC_NS[] / 1e6 / nsteps)
    PROF && row(case, "profiler_overhead", ov)
    row(case, "step", tstep)
    PROF && @printf("step %.2f ms (profiler's own %.2f ms removed), in launches %.2f ms, outside %.2f ms (%d kernels)\n",
                    tstep, ov, tin, tstep - tin, length(STATS))
end

# ---------------------------------------------------------------------------

MODE == "dns" && include(joinpath(@__DIR__, "..", "test", "fixtures", "dns_decay.jl"))

function bench_flagship(; nwarm = 2, nsteps = 9)
    BC = Aeolus.Urban
    case = BC.build_case(BC.site_options(load_run_config(joinpath(pkgdir(Aeolus), "examples", "urban", "berlin_summer.toml"))))
    for _ in 1:nwarm
        step!(case.model)
    end
    t = 0.0
    RECORDING[] = true
    g0 = Base.gc_time_ns()
    for _ in 1:nsteps
        t += @elapsed(step!(case.model))
    end
    GC_NS[] = Base.gc_time_ns() - g0
    RECORDING[] = false
    dump_stats("flagship", nsteps, 1e3t / nsteps)
end

function bench_dns(N; nwarm = 1, nsteps = 3)
    ν = 1.035 * 0.018 * (48 / N)^(4 / 3)
    st = Main.dns_start(; N, seed = 1, ν, spectrum = q -> q^4 * exp(-2 * (q / 1.5)^2))
    for _ in 1:nwarm
        Main.dns_step!(st; cfl = 0.5, t_end = 2.0, t_max = 6.0)
    end
    sim = st.sim
    dt = sim.t / (length(st.hist.t) - 1)
    step_rk3!(sim, dt)
    t = Inf
    RECORDING[] = true
    g0 = Base.gc_time_ns()
    ov = 0                     # the profiler's own time, taken out step by step
    for _ in 1:nsteps
        o0 = OVERHEAD_NS[]
        ts = @elapsed step_rk3!(sim, dt)
        if ts - (OVERHEAD_NS[] - o0) / 1e9 < t
            t, ov = ts - (OVERHEAD_NS[] - o0) / 1e9, OVERHEAD_NS[] - o0
        end
    end
    GC_NS[] = Base.gc_time_ns() - g0
    RECORDING[] = false
    # minimum per step for the step (the profiler's time of that step taken
    # out); the kernel stats are means over all steps
    dump_stats("dns$N", nsteps, 1e3t; step_overhead_ms = ov / 1e6)
    if !PROF
        tf = minimum(@elapsed(Main.dns_step!(st; cfl = 0.5, t_end = 2.0, t_max = 6.0)) for _ in 1:nsteps)
        @printf("dns_step! (step_rk3! + history) min %.3f s\n", tf)
        row("dns$N", "dns_step", 1e3tf)
    end
end

function bench_config(path, sets; nwarm = 2, nsteps = 6)
    CP = Aeolus.Components
    cfg = load_run_config(path; overrides = vcat(["output.variables=[]"], sets))
    model = Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model
    d = only(model.domains)
    dt = cfg.time.dt
    # an adaptive run: its CFL step; a fixed one: its dt
    stepper() = d.cfl === nothing ? step!(model, Float64(dt); checked = false) : CP.step_adaptive!(model)
    for _ in 1:nwarm
        stepper()
    end
    name = splitext(basename(path))[1]
    t = 0.0
    RECORDING[] = true
    g0 = Base.gc_time_ns()
    o0 = OVERHEAD_NS[]
    for _ in 1:nsteps
        t += @elapsed stepper()
    end
    GC_NS[] = Base.gc_time_ns() - g0
    RECORDING[] = false
    ov = (OVERHEAD_NS[] - o0) / 1e6 / nsteps          # the profiler's own time, per step
    t -= ov * nsteps / 1e3
    for d in model.domains
        rep = CP.sweep_report(d)
        println("loop fusion ($(name)): ", rep["setting"], ", substages fused: ", rep["substages_fused"],
                isempty(rep["not_fusable"]) ? "" : " (not fusable: " * join(first.(split.(rep["not_fusable"], ":")), ", ") * ")")
    end
    dump_stats(name, nsteps, 1e3t / nsteps; step_overhead_ms = ov)
end

argint(i, d) = length(ARGS) >= i ? parse(Int, ARGS[i]) : d
init_threads!()
affinity()
if MODE == "stream"
    stream("machine")
elseif MODE == "flagship"
    stream("flagship")
    bench_flagship(; nwarm = argint(2, 2), nsteps = argint(3, 9))
elseif MODE == "dns"
    N = argint(2, 256)
    stream("dns$N")
    bench_dns(N; nwarm = argint(3, 1), nsteps = argint(4, 3))
elseif MODE == "config"
    pos = String[]; sets = String[]
    let i = 2
        while i <= length(ARGS)
            ARGS[i] == "--set" ? (push!(sets, ARGS[i+1]); i += 2) : (push!(pos, ARGS[i]); i += 1)
        end
    end
    stream(splitext(basename(pos[1]))[1])
    bench_config(pos[1], sets; nwarm = length(pos) >= 2 ? parse(Int, pos[2]) : 2,
                 nsteps = length(pos) >= 3 ? parse(Int, pos[3]) : 6)
else
    error("unknown mode $MODE (stream, flagship, dns, config, merge)")
end
