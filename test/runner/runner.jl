# Parallel runner for the Aeolus.jl suites (#316). Stdlib only.
#
# Entry points (each runs one suite):
#
#   julia --project=. test/run.jl                 # the tests (the PR gate)
#   julia --project=. validation/run.jl           # the validation runs (not a gate)
#   JOBS=8 julia --project=. test/run.jl --threads 2
#   julia --project=. test/run.jl --only tke.jl,io_checkpoint.jl
#   julia --project=. validation/run.jl --only gabls1.jl
#   julia --project=. test/run.jl --list          # show the plan, run nothing
#
# Every file of the suite's include list (test/runtests.jl or
# validation/runvalidation.jl), plus the inline tests of that entry file
# (job "runtests.jl" / "runvalidation.jl"), runs in its own Julia process,
# longest first. Files tagged `exclusive` in the suite's tag table run alone
# after the parallel batch. Each job writes a log and a result file into a
# run folder (default .testruns/<time>-<suite>/); the runner prints a
# summary per file and per top-level @testset block and exits non-zero on
# any failure, error or crashed job.
#
# Options (environment variable in brackets):
#   --jobs N           [JOBS]                 default: free cores = CPU threads
#                                             minus the 1-min load, at least 1,
#                                             divided by --threads, capped by memory
#   --threads T        [AEOLUS_TEST_THREADS]  Julia threads per job, default 1
#   --mem-per-job GB   [AEOLUS_TEST_MEM_GB]   memory reserved per job, default 1
#   --only a.jl,b.jl                          run only these jobs
#   --log LEVEL        [AEOLUS_LOG]           log level of the jobs' test logger:
#                                             warn (default), info, debug (#333)
#   --outdir DIR                              run folder
#   --list                                    print the plan and exit
#
# See docs/src/testing.md and docs/src/validation.md.

using Dates
using Printf
using TOML

Base.include(@__MODULE__, joinpath(@__DIR__, "selection.jl"))
using .AeolusTestSelection: ROOT, Suite, Tag, core, has_inline, read_tags, listed_files, tag

const WORKER = joinpath(@__DIR__, "worker.jl")
"Measured wall times per job, one file per suite (the tests keep the old name)."
timings_file(s::Suite) = joinpath(ROOT, ".testruns",
                                  s.name == "test" ? "timings.toml" : "timings_$(s.name).toml")

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--list", "--help", "-h")
            opts[lstrip(a, '-')] = "true"
        elseif startswith(a, "--")
            k, v = occursin('=', a) ? split(a[3:end], '='; limit = 2) :
                   (a[3:end], (i += 1; i <= length(args) ? args[i] : error("$a needs a value")))
            k == "tier" && error("--tier is gone: the full tier's validation runs are a separate " *
                                 "suite, `julia --project=. validation/run.jl` (docs/src/validation.md)")
            k in ("jobs", "threads", "mem-per-job", "only", "outdir", "log") ||
                error("unknown option --$k (see the header of test/runner/runner.jl)")
            opts[k] = v
        else
            error("unexpected argument $a")
        end
        i += 1
    end
    return opts
end

"1-minute load average, from /proc/loadavg where available."
function load1()
    try
        return parse(Float64, split(read("/proc/loadavg", String))[1])
    catch
        return Sys.loadavg()[1]
    end
end

"Available memory in GB (MemAvailable on Linux)."
function mem_available_gb()
    try
        for l in eachline("/proc/meminfo")
            startswith(l, "MemAvailable:") && return parse(Float64, split(l)[2]) / 2^20
        end
    catch
    end
    return Sys.free_memory() / 2^30
end

"""
Default job count: free cores (CPU threads minus the rounded 1-min load,
at least 1) divided by the threads per job, capped by available memory.
"""
function default_jobs(threads, mem_per_job)
    free = max(1, Sys.CPU_THREADS - round(Int, load1()))
    bymem = max(1, floor(Int, mem_available_gb() / mem_per_job))
    return max(1, min(free ÷ threads, bymem))
end

# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

struct Job
    name::String
    tag::Tag
    estimate::Float64     # seconds, for longest-first ordering
end

function load_timings(s::Suite)
    f = timings_file(s)
    isfile(f) || return Dict{String,Float64}()
    try
        return Dict{String,Float64}(k => Float64(v) for (k, v) in TOML.parsefile(f))
    catch
        return Dict{String,Float64}()
    end
end

"Seconds estimate: last measured run, else the tag's `seconds`, else by weight."
function estimate(name, t::Tag, timings)
    haskey(timings, name) && return timings[name]
    t.seconds > 0 && return t.seconds
    return t.weight == "heavy" ? 300.0 : 30.0
end

function plan(s::Suite, only, tags)
    files = listed_files(s)
    names = has_inline(s) ? vcat(core(s), files) : copy(files)
    entry = basename(s.entry)
    dup = [f for f in unique(files) if count(==(f), files) > 1]
    isempty(dup) || error("$entry includes these files more than once: $(join(dup, ", "))")
    stale = setdiff(keys(tags), names)
    isempty(stale) || @warn "$(basename(s.tagfile)) has entries for files $entry does not include: $(join(sort(collect(stale)), ", "))"
    if only !== nothing
        bad = setdiff(only, names)
        if !isempty(bad)
            other = s.name == "test" ? "validation" : "test"
            elsewhere = intersect(bad, listed_files(AeolusTestSelection.suite(other)))
            hint = isempty(elsewhere) ? "" :
                   " ($(join(elsewhere, ", ")) $(length(elsewhere) == 1 ? "is" : "are") in the $other suite: " *
                   "$(other == "test" ? "test/run.jl" : "validation/run.jl"))"
            error("--only: not in $entry's include list: $(join(bad, ", "))$hint")
        end
        names = filter(in(only), names)
    end
    timings = load_timings(s)
    jobs = [Job(n, tag(tags, n), estimate(n, tag(tags, n), timings)) for n in names]
    par = sort(filter(j -> !j.tag.exclusive, jobs); by = j -> (j.tag.weight != "heavy", -j.estimate))
    exc = sort(filter(j -> j.tag.exclusive, jobs); by = j -> -j.estimate)
    return par, exc
end

# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

mutable struct Running
    job::Job
    proc::Base.Process
    t0::Float64
    log::String
    result::String
end

function start(s::Suite, job::Job, outdir, threads)
    stem = replace(job.name, r"\.jl$" => "")
    log = joinpath(outdir, stem * ".log")
    result = joinpath(outdir, stem * ".result.toml")
    cmd = `$(Base.julia_cmd()) --project=$ROOT --startup-file=no --threads=$threads $WORKER $(s.name) $(job.name) $result`
    # BLAS threads follow the job's thread count, so N jobs do not each
    # start a BLAS pool sized for the whole machine
    env = copy(ENV)
    env["OPENBLAS_NUM_THREADS"] = string(threads)
    env["AEOLUS_TEST_WORKER"] = "1"
    proc = run(pipeline(setenv(cmd, env); stdout = log, stderr = log); wait = false)
    return Running(job, proc, time(), log, result)
end

struct Outcome
    job::Job
    seconds::Float64
    exitcode::Int
    blocks::Vector{Any}
    loose::Dict{String,Any}
    loaderror::String
    log::String
    crashed::Bool
end

function collect_outcome(r::Running)
    wall = time() - r.t0
    if isfile(r.result)
        d = TOML.parsefile(r.result)
        return Outcome(r.job, wall, r.proc.exitcode, get(d, "blocks", Any[]),
                       get(d, "loose", Dict{String,Any}()), get(d, "loaderror", ""), r.log, false)
    end
    return Outcome(r.job, wall, r.proc.exitcode, Any[], Dict{String,Any}(), "", r.log, true)
end

total(o::Outcome, k) = sum((b[k] for b in o.blocks); init = 0)
failed(o::Outcome) = o.crashed || !isempty(o.loaderror) || total(o, "fail") > 0 ||
                     total(o, "error") > 0 || get(o.loose, "fail", 0) > 0 || get(o.loose, "error", 0) > 0

function report_one(o::Outcome, k, n)
    status = o.crashed ? "CRASHED (exit $(o.exitcode))" :
             !isempty(o.loaderror) ? "ERROR outside @testset" :
             failed(o) ? "FAILED" : "ok"
    @printf("[%3d/%-3d] %-38s %8.1f s  pass %6d  fail %3d  error %3d  broken %3d  %s\n",
            k, n, o.job.name, o.seconds, total(o, "pass"), total(o, "fail"),
            total(o, "error"), total(o, "broken"), status)
    failed(o) && println("          log: ", o.log)
    flush(stdout)
end

"Run `jobs` with at most `njobs` processes at a time, longest first."
function run_batch!(s::Suite, outcomes, jobs, njobs, outdir, threads, counter, n)
    queue = copy(jobs)
    running = Running[]
    while !isempty(queue) || !isempty(running)
        while !isempty(queue) && length(running) < njobs
            push!(running, start(s, popfirst!(queue), outdir, threads))
        end
        sleep(0.2)
        for r in filter(r -> process_exited(r.proc), running)
            o = collect_outcome(r)
            push!(outcomes, o)
            counter[] += 1
            report_one(o, counter[], n)
        end
        filter!(r -> !process_exited(r.proc), running)
    end
end

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

function block_table(s::Suite, outcomes)
    order = String[]
    agg = Dict{String,Vector{Int}}()
    # blocks in include-list order
    names = vcat(core(s), listed_files(s))
    for name in names, o in outcomes
        o.job.name == name || continue
        for b in o.blocks
            bn = b["name"]
            haskey(agg, bn) || (push!(order, bn); agg[bn] = zeros(Int, 4))
            agg[bn] .+= (b["pass"], b["fail"], b["error"], b["broken"])
        end
    end
    return order, agg
end

function summarize(io, s::Suite, outcomes, njobs, threads, wall, load0, outdir)
    order, agg = block_table(s, outcomes)
    println(io, "\nAeolus $(s.name) run: $njobs jobs × $threads threads, ",
            @sprintf("wall %.1f s (%.1f min), 1-min load at start %.2f", wall, wall / 60, load0))
    println(io, "Run folder: ", outdir)
    println(io, "\nPer top-level @testset block:")
    @printf(io, "  %-44s %8s %6s %6s %7s %8s\n", "block", "pass", "fail", "error", "broken", "total")
    tot = zeros(Int, 4)
    for bn in order
        c = agg[bn]
        tot .+= c
        @printf(io, "  %-44s %8d %6d %6d %7d %8d\n", first(bn, 44), c..., sum(c))
    end
    @printf(io, "  %-44s %8d %6d %6d %7d %8d\n", "TOTAL", tot..., sum(tot))
    loose = sum((get(o.loose, "pass", 0) for o in outcomes); init = 0)
    loose > 0 && println(io, "  ($loose passing @test outside any @testset, not counted, as in the serial run)")
    bad = filter(failed, outcomes)
    println(io, "\nSlowest jobs:")
    for o in first(sort(outcomes; by = o -> -o.seconds), 10)
        @printf(io, "  %-40s %8.1f s%s\n", o.job.name, o.seconds, o.job.tag.exclusive ? "  (exclusive)" : "")
    end
    if isempty(bad)
        println(io, "\nAll $(length(outcomes)) jobs passed.")
    else
        println(io, "\n$(length(bad)) of $(length(outcomes)) jobs FAILED:")
        for o in bad
            println(io, "  ", o.job.name, "  ", o.crashed ? "crashed, exit $(o.exitcode)" : "", "  log: ", o.log)
        end
    end
    return tot
end

function save_timings(s::Suite, outcomes)
    t = load_timings(s)
    for o in outcomes
        o.crashed || (t[o.job.name] = round(o.seconds; digits = 1))
    end
    f = timings_file(s)
    mkpath(dirname(f))
    open(io -> TOML.print(io, t; sorted = true), f, "w")
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

"""
    main(args, suite) -> exit code

Run one suite (`"test"` or `"validation"`) with the command-line `args`.
"""
function main(args, suitename::AbstractString)
    s = AeolusTestSelection.suite(suitename)
    opts = parse_args(args)
    if haskey(opts, "help") || haskey(opts, "h")
        foreach(l -> println(lstrip(l, ['#'])), filter(startswith("#"), readlines(@__FILE__)))
        return 0
    end
    if haskey(ENV, "AEOLUS_TEST_TIER")
        @warn "AEOLUS_TEST_TIER is ignored: the fast/full tiers were replaced by the test and " *
              "validation suites (test/run.jl, validation/run.jl; docs/src/validation.md)"
    end
    # every job installs Aeolus.test_logger() (runtests.jl header), which
    # reads AEOLUS_LOG: warnings and up unless asked for more (#333)
    haskey(opts, "log") && (ENV["AEOLUS_LOG"] = opts["log"])
    threads = parse(Int, get(opts, "threads", get(ENV, "AEOLUS_TEST_THREADS", "1")))
    memjob = parse(Float64, get(opts, "mem-per-job", get(ENV, "AEOLUS_TEST_MEM_GB", "1")))
    load0 = load1()
    jobs_s = get(opts, "jobs", get(ENV, "JOBS", ""))
    njobs = isempty(jobs_s) ? default_jobs(threads, memjob) : parse(Int, jobs_s)
    njobs >= 1 || error("jobs must be ≥ 1")
    only = haskey(opts, "only") ? String.(split(opts["only"], ',')) : nothing
    tags = read_tags(s)
    par, exc = plan(s, only, tags)

    if haskey(opts, "list")
        println("$(s.name) suite: $(length(par)) parallel + $(length(exc)) exclusive jobs, ",
                "$njobs jobs × $threads threads (1-min load $(load0), $(Sys.CPU_THREADS) CPU threads)")
        for j in par
            @printf("  %-40s %-6s ~%6.0f s\n", j.name, j.tag.weight, j.estimate)
        end
        for j in exc
            @printf("  %-40s %-6s ~%6.0f s  exclusive\n", j.name, j.tag.weight, j.estimate)
        end
        return 0
    end

    outdir = get(opts, "outdir", joinpath(ROOT, ".testruns",
                 Dates.format(now(), "yyyymmdd-HHMMSS") * "-" * s.name))
    mkpath(outdir)
    n = length(par) + length(exc)
    println("Aeolus $(s.name) suite: $(length(par)) parallel jobs then $(length(exc)) exclusive, ",
            "$njobs jobs × $threads threads (1-min load $(load0), $(Sys.CPU_THREADS) CPU threads)")
    println("Logs: ", outdir)
    flush(stdout)

    # precompile once, so the workers do not queue on the precompile lock
    t0 = time()
    tp = @elapsed run(`$(Base.julia_cmd()) --project=$ROOT --startup-file=no -e "using Aeolus"`)
    @printf("Precompiled / loaded Aeolus in %.1f s\n", tp)
    flush(stdout)

    outcomes = Outcome[]
    counter = Ref(0)
    run_batch!(s, outcomes, par, njobs, outdir, threads, counter, n)
    if !isempty(exc)
        println("-- exclusive jobs, one at a time --")
        run_batch!(s, outcomes, exc, 1, outdir, threads, counter, n)
    end
    wall = time() - t0

    summary = sprint(io -> summarize(io, s, outcomes, njobs, threads, wall, load0, outdir))
    print(summary)
    write(joinpath(outdir, "summary.txt"), summary)
    open(joinpath(outdir, "summary.toml"), "w") do io
        order, agg = block_table(s, outcomes)
        TOML.print(io, Dict(
            "suite" => s.name, "jobs" => njobs, "threads" => threads, "wall" => wall, "load" => load0,
            "blocks" => [Dict("name" => b, "pass" => agg[b][1], "fail" => agg[b][2],
                              "error" => agg[b][3], "broken" => agg[b][4]) for b in order],
            "files" => [Dict("name" => o.job.name, "seconds" => o.seconds, "failed" => failed(o),
                             "exclusive" => o.job.tag.exclusive, "log" => o.log) for o in outcomes]))
    end
    save_timings(s, outcomes)
    return any(failed, outcomes) ? 1 : 0
end

