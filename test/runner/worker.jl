# One job of the parallel runner (#316): run a single file of a suite's
# include list (test/runtests.jl or validation/runvalidation.jl), or that
# entry file's inline tests (job "runtests.jl" / "runvalidation.jl"), in
# this process and write its counts to a TOML result file.
#
#   julia --project=. test/runner/worker.jl <suite> <job> <result.toml>
#
# The job is evaluated by AeolusTestSelection.run_selected inside an outer
# DefaultTestSet, so the top-level @testset blocks it meets (e.g. "Turbulence closure")
# become its children: their counts are read per block, and a failure does
# not stop the process before the counts are written.

using Test
using TOML

Base.include(Main, joinpath(@__DIR__, "selection.jl"))
using .AeolusTestSelection: core, run_selected, suite

const SUITE = suite(ARGS[1])
const JOB = ARGS[2]
const OUT = ARGS[3]

"(pass, fail, error, broken) of a finished testset, children included."
function counts(ts)
    c = Test.get_test_counts(ts)
    if c isa Tuple   # Julia ≤ 1.10: (p, f, e, b, cp, cf, ce, cb, ...)
        return (c[1] + c[5], c[2] + c[6], c[3] + c[7], c[4] + c[8])
    end
    return (c.passes + c.cumulative_passes, c.fails + c.cumulative_fails,
            c.errors + c.cumulative_errors, c.broken + c.cumulative_broken)
end

keep, inline = JOB == core(SUITE) ? ((f -> false), true) : ((f -> f == JOB), false)

outer = Test.DefaultTestSet("job $JOB")
loaderror = ""
t0 = time()
"Run `f()` with `ts` as the current testset (Julia ≥ 1.11: scoped values)."
function with_testset(f, ts)
    if isdefined(Test, Symbol("@with_testset"))
        Core.eval(@__MODULE__, :(Test.@with_testset $ts $f()))
    else
        Test.push_testset(ts)
        try
            f()
        finally
            Test.pop_testset()
        end
    end
end

try
    with_testset(() -> run_selected(keep, inline, SUITE.entry), outer)
catch err
    # an error outside every @testset: the serial run would stop here
    global loaderror = sprint(showerror, err, catch_backtrace())
    println(stderr, "ERROR outside any @testset in job $JOB:\n", loaderror)
end
seconds = time() - t0

blocks = Dict{String,Any}[]
loose = Dict("pass" => 0, "fail" => 0, "error" => 0, "broken" => 0)
for r in outer.results
    if r isa Test.AbstractTestSet
        p, f, e, b = counts(r)
        push!(blocks, Dict("name" => r.description, "pass" => p, "fail" => f,
                           "error" => e, "broken" => b))
    elseif r isa Test.Fail
        loose["fail"] += 1
    elseif r isa Test.Error
        loose["error"] += 1
    elseif r isa Test.Broken
        loose["broken"] += 1
    end
end
# passing @tests outside any @testset: the serial run does not count them
n = outer.n_passed
loose["pass"] = n isa Integer ? n : n[]

try
    Test.print_test_results(outer)
catch
end

open(OUT, "w") do io
    TOML.print(io, Dict("job" => JOB, "seconds" => seconds, "threads" => Threads.nthreads(),
                        "maxrss_mb" => round(Sys.maxrss() / 2^20; digits = 1),
                        "blocks" => blocks, "loose" => loose, "loaderror" => loaderror))
end
