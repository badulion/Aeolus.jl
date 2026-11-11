# The shared part of the kernel-fix benchmarks (#566, HANDOFF.md section 7):
# the arguments, the Berlin 100 m state, the timing loop, the TOML.
#
#   julia -t N --project=<build> run_fixes.jl <out.toml> <fix>... [key=value ...]
#   julia -t N --project=<build> <fix script>.jl <out.toml> [key=value ...]
#
# keys: config (the Berlin 100 m TOML; default berlin_100m/ next to this
# directory), budget (seconds of timing per kernel set, default 20), ref (the
# TOML of a bench/all process: the alternating fixes E4-E6 read its dumps,
# `<ref>.<name>.jls`, as input or reference), build, label (copied).
#
# Every fix registers `FIXES[name] = () -> Dict` (its timings and agreement
# check); a process builds the Berlin state at most once (`berlin()`, about
# 3 min) and writes one TOML with a table per fix it ran. Included once: the
# fix scripts include it unless FIXES exists.

using TOML, Dates, Serialization, Statistics
const T_START = time()
using Aeolus
using Aeolus.Operators: advance_sgs!
const T_LOADED = time()
const SIM = Aeolus.Simulations
const PS = Aeolus.PrognosticSGS
const SUPPORT = joinpath(pkgdir(Aeolus), "test", "support")   # the build's own test cases

const OUT = abspath(ARGS[1])
const KV = Dict{String,String}(split(a, "="; limit = 2)[1] => split(a, "="; limit = 2)[2]
                               for a in ARGS[2:end] if occursin('=', a))
const REQUESTED = [a for a in ARGS[2:end] if !occursin('=', a)]
const BUDGET = parse(Float64, get(KV, "budget", "20"))
const REF = get(KV, "ref", "")
const FIXES = Dict{String,Function}()
const STATE = Ref{Any}(nothing)
const T_BUILD = Ref(0.0)

# the Berlin 100 m model after 3 steps of dt = 2 s, built once per process
function berlin()
    STATE[] === nothing || return STATE[]
    t0 = time()
    cfgpath = get(KV, "config", joinpath(@__DIR__, "..", "berlin_100m", "berlin_100m_20250824.toml"))
    cfg = Aeolus.load_run_config(cfgpath; overrides = ["fftw.planner=\"estimate\""])
    model = Aeolus.Config.build(Aeolus.SimulationDriver.SimulationDriverType(), cfg).model
    for _ in 1:3; Aeolus.step!(model, 2.0); end
    d = first(model.domains)
    STATE[] = (model = model, d = d, sim = d.state, c = Aeolus.Components.sgs_closure(d))
    T_BUILD[] = time() - t0
    return STATE[]
end

# min and median (ms) of every kernel `name => f`: two warm-up calls each,
# then rounds of `inner` calls per kernel (1 when the slowest warm-up call
# took over 0.3 s), the order reversed every round, until `budget` seconds
# have passed (at least `minrounds` rounds)
function measure(fs; inner = 3, minrounds = 3, budget = BUDGET)
    tw = maximum(fs) do (_, f)
        f(); @elapsed f()
    end
    tw > 0.3 && (inner = 1)
    res = Dict(k => Float64[] for (k, _) in fs)
    t0 = time(); rep = 0
    while rep < minrounds || time() - t0 < budget
        rep += 1
        for (k, f) in (isodd(rep) ? fs : reverse(fs)), _ in 1:inner
            t = time_ns(); f(); push!(res[k], (time_ns() - t) / 1e6)
        end
    end
    out = Dict{String,Any}()
    for (k, _) in fs
        out[k] = Dict("min_ms" => minimum(res[k]), "median_ms" => median(res[k]), "n" => length(res[k]))
        println("  ", rpad(k, 22), " min ", lpad(round(minimum(res[k]), digits = 3), 9), " ms  median ",
                lpad(round(median(res[k]), digits = 3), 9), " ms  (", length(res[k]), ")")
    end
    return out
end

relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs, b), floatmin(Float64))

# the dumps of the alternating fixes: written next to the TOML, read from ref
dump_path(name) = OUT * "." * name * ".jls"
ref_path(name) = isempty(REF) ? "" : abspath(REF) * "." * name * ".jls"
function read_ref(name)
    p = ref_path(name)
    isfile(p) || return nothing
    t = @elapsed x = deserialize(p)
    println("  read ", p, " (", round(t, digits = 1), " s)")
    return x
end
function write_dump(name, x)
    t = @elapsed serialize(dump_path(name), x)
    println("  wrote ", dump_path(name), " (", round(t, digits = 1), " s)")
end

function run_fixes(names)
    res = Dict{String,Any}(
        "build" => get(KV, "build", ""), "label" => get(KV, "label", ""), "fixes" => names,
        "commit" => String(strip(read(`git -C $(pkgdir(Aeolus)) rev-parse HEAD`, String))),
        "aeolus_dir" => pkgdir(Aeolus), "threads" => Threads.nthreads(), "julia" => string(VERSION),
        "host" => gethostname(), "slurm_job" => get(ENV, "SLURM_JOB_ID", ""), "ref" => REF,
        "budget_s" => BUDGET, "started" => string(now()), "t_load_s" => T_LOADED - T_START)
    for n in names
        println("== $n, threads $(Threads.nthreads()), $(res["commit"][1:8])")
        t0 = time()
        res[n] = try
            FIXES[n]()
        catch err
            showerror(stdout, err, catch_backtrace()); println()
            Dict{String,Any}("error" => sprint(showerror, err))
        end
        res[n]["t_s"] = time() - t0
    end
    res["t_build_s"] = T_BUILD[]
    res["t_total_s"] = time() - T_START
    mkpath(dirname(OUT))
    open(io -> TOML.print(io, res), OUT, "w")
    println("wrote $OUT ($(round(res["t_total_s"], digits = 1)) s)")
    return res
end

# a fix script run on its own: its fix, into ARGS[1]
standalone(name, file) = abspath(PROGRAM_FILE) == abspath(file) && run_fixes([name])
