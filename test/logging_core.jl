# Logging core (LOG-a, #333; design in epic #293): src/logging/RunLog.jl and
# the structured events of the core (solver non-convergence, adaptive
# retries, checkpoints).
#
# The acceptance tests of the issue:
#   * the library is silent with no logger: under a NullLogger nothing at
#     all reaches stdout/stderr, even when every solver fails to converge;
#     and a healthy run emits nothing at Info or above, so the default
#     console shows nothing;
#   * events have the right structure under a capturing logger: level,
#     subsystem, domain and key-values equal to the solver's own state;
#   * the levels filter correctly: per logger, per tee child, before the
#     message is built.

using Test
using Logging
using Aeolus
using Aeolus.Fields: interior_view, set!
using Aeolus.BoundaryConditions: FieldBoundaryConditions
using Aeolus.TimeIntegration: step_adaptive!, CFLConstraints, solve_implicit_diffusion!
using Random

function lg_sim()
    m = RegularGrid(; size = (8, 8, 8), extent = (800.0, 800.0, 400.0), halo = (3, 3, 3))
    bcs = FieldBoundaryConditions(:periodic, :periodic, :wall)
    sim = BoussinesqSimulation(m; θ₀ = 300.0, bcs = bcs)
    rng = Xoshiro(7)
    set!(sim.u, (x, y, z) -> 1.0 + 0.3 * randn(rng))
    set!(sim.θ, (x, y, z) -> 300.0 + 0.003 * z + 0.1 * randn(rng))
    Aeolus.TimeIntegration._project!(sim, 1.0)
    return sim
end

"A rough right-hand side on an N³ periodic grid (zero mean)."
function lg_rhs(g)
    src = Field(CellCenter(), g)
    rng = Xoshiro(11)
    set!(src, (x, y, z) -> randn(rng))
    v = interior_view(src); v .-= sum(v) / length(v)
    return src
end

"Every way the core can fail to converge or retry, once each."
function lg_failures(; verbose = true)
    g = RegularGrid(; size = (16, 16, 16), extent = (1.0, 1.0, 1.0), halo = (1, 1, 1))
    src = lg_rhs(g); out = Field(CellCenter(), g)
    mg = MultigridPoissonSolver(g; bc_x = :periodic, bc_y = :periodic, bc_z = :periodic,
                                tol = 1e-14, maxcycles = 1)
    solve!(out, src, mg)
    cg = CGPoissonSolver(g; bc_top = :periodic, bc_bottom = :periodic, tol = 1e-14, maxit = 2)
    solve!(out, src, cg)
    sol = Field(CellCenter(), g)
    solve_implicit_diffusion!(sol, src, 0.7, 0.3, g, FieldBoundaryConditions(:periodic);
                              tol = 1e-14, maxit = 1)
    sim = lg_sim()
    calls = Ref(0)
    flaky = (s, h; kw...) -> (calls[] += 1; calls[] == 1 && error("injected failure");
                              step!(s, h; kw...))
    step_adaptive!(sim, CFLConstraints(; dt_max = 2.0); stepper = flaky, verbose)
    return (; mg, cg, sim)
end

"stdout and stderr of `f()`, captured through a file."
function lg_capture_output(f)
    path = tempname()
    open(path, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                f()
            end
        end
    end
    s = read(path, String); rm(path)
    return s
end

kw(r) = Dict{Symbol,Any}(r.kwargs)

@testset "Logging core (LOG-a, #333)" begin

    @testset "silent with no logger: nothing reaches stdout/stderr under a NullLogger" begin
        out = lg_capture_output() do
            with_logger(NullLogger()) do
                lg_failures(; verbose = true)
                dir = mktempdir()
                sim = lg_sim()
                ck = CheckpointSchedule(dir; every_steps = 1, wisdom = false, verbose = true)
                resume!(ck, sim)
                step!(sim, 1.0)
                maybe_checkpoint!(ck, sim; step = 1)
            end
        end
        @test out == ""
    end

    @testset "a healthy run emits nothing at Info or above" begin
        tl = Test.TestLogger(; min_level = Logging.Debug)
        with_logger(tl) do
            sim = lg_sim()
            dir = mktempdir()
            ck = CheckpointSchedule(dir; every_steps = 2, wisdom = false, verbose = false)
            resume!(ck, sim)
            dt = nothing
            for n in 1:4
                _, dt, _ = step_adaptive!(sim, CFLConstraints(; dt_max = 2.0); dt_prev = dt)
                maybe_checkpoint!(ck, sim; step = n)
            end
        end
        @test all(r -> r.level < Logging.Info, tl.logs)
        # ... but run.log (debug) sees the run's events: the fresh start and
        # the two checkpoint writes (steps 2 and 4), with their tags
        events = [kw(r)[:event] for r in tl.logs if get(kw(r), :subsystem, nothing) === :checkpoint]
        @test events == [:fresh_start, :write, :write]
        w = [r for r in tl.logs if get(kw(r), :event, nothing) === :write]
        @test [kw(r)[:step] for r in w] == [2, 4]
        @test all(r -> isfile(kw(r)[:path]) && kw(r)[:removed] == (), w)   # keep = 2 (default)
        @test kw(w[end])[:t] > 0
    end

    @testset "events carry subsystem, domain and the solver's own numbers" begin
        tl = Test.TestLogger(; min_level = Logging.Debug)
        res = with_logger(tl) do
            with_domain(:child) do
                lg_failures(; verbose = false)
            end
        end
        @test all(r -> haskey(kw(r), :subsystem) && kw(r)[:domain] === :child, tl.logs)

        mgr = only(r for r in tl.logs if get(kw(r), :solver, nothing) === :multigrid)
        @test mgr.level == Logging.Warn
        @test kw(mgr)[:subsystem] === :pressure
        @test kw(mgr)[:cycles] == 1 == res.mg.last_cycles[]
        @test kw(mgr)[:relative_residual] == res.mg.last_residual[]
        @test kw(mgr)[:relative_residual] > kw(mgr)[:tol] == 1e-14
        @test occursin("maxcycles=1 reached", mgr.message)

        cgr = only(r for r in tl.logs if get(kw(r), :solver, nothing) === :cg)
        @test cgr.level == Logging.Warn && kw(cgr)[:subsystem] === :pressure
        @test kw(cgr)[:iterations] == 2 == res.cg.last_iter[]
        @test kw(cgr)[:residual] == res.cg.last_residual[]
        @test kw(cgr)[:relative_residual] > kw(cgr)[:tol]

        dr = only(r for r in tl.logs if get(kw(r), :subsystem, nothing) === :diffusion)
        @test dr.level == Logging.Warn
        @test kw(dr)[:reason] === :maxit && kw(dr)[:iterations] == 1
        @test kw(dr)[:relative_residual] > kw(dr)[:tol] == 1e-14

        # the adaptive driver: one throw, one retry at half the step, both
        # at Debug when not verbose (run.log only)
        ts = [r for r in tl.logs if get(kw(r), :subsystem, nothing) === :timestepping]
        @test length(ts) == 2
        thr, rty = ts
        @test thr.level == Logging.Debug && rty.level == Logging.Debug
        @test kw(thr)[:exception] isa ErrorException
        @test kw(thr)[:exception].msg == "injected failure"
        @test kw(rty)[:reason] === :exception
        @test kw(rty)[:to] == kw(rty)[:from] / 2 && kw(rty)[:from] == kw(thr)[:dt]
        @test kw(rty)[:retry] == 1 && kw(rty)[:max_retries] == 3

        # verbose = true raises the same two events to Warn and Info
        tl2 = Test.TestLogger(; min_level = Logging.Debug)
        with_logger(tl2) do
            sim = lg_sim(); calls = Ref(0)
            step_adaptive!(sim, CFLConstraints(; dt_max = 2.0); verbose = true,
                           stepper = (s, h; kw...) -> (calls[] += 1; calls[] == 1 && error("x");
                                                       step!(s, h; kw...)))
        end
        @test [r.level for r in tl2.logs] == [Logging.Warn, Logging.Info]
        @test all(r -> kw(r)[:domain] === :main, tl2.logs)    # outside with_domain
    end

    @testset "with_domain nests and restores; tasks start at :main" begin
        @test current_domain() === :main
        inner = with_domain(:parent) do
            a = current_domain()
            b = with_domain(:child) do
                (current_domain(), fetch(Threads.@spawn current_domain()))
            end
            (a, b..., current_domain())
        end
        @test inner == (:parent, :child, :main, :parent)
        @test current_domain() === :main
        @test_throws ErrorException with_domain(() -> error("x"), :d)
        @test current_domain() === :main
    end

    @testset "AeolusLogger: format, levels, laziness, maxlog" begin
        buf = IOBuffer()
        lg = AeolusLogger(buf; min_level = Logging.Warn)
        evaluated = Ref(0)
        with_logger(lg) do
            @debug "d" subsystem = :t x = (evaluated[] += 1)
            @info "i" subsystem = :t x = (evaluated[] += 1)
            @warn "w" subsystem = :pressure domain = :child iterations = 7 residual = 0.5
            @error "e"
        end
        lines = split(String(take!(buf)), '\n'; keepempty = false)
        @test evaluated[] == 0          # filtered records are never built
        @test length(lines) == 2
        @test occursin(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3} WARN  pressure\[child\] w  iterations=7 residual=0\.5  @ logging_core\.jl:\d+$", lines[1])
        @test occursin(r"^\S+ ERROR e  @ logging_core\.jl:\d+$", lines[2])

        # multi-line messages, exceptions and maxlog
        lg = AeolusLogger(buf; min_level = Logging.Debug, timestamps = :none)
        with_logger(lg) do
            @info "first\nsecond" subsystem = :t
            @warn "oops" exception = ArgumentError("bad value")
            for _ in 1:5
                @info "once" maxlog = 2
            end
        end
        s = String(take!(buf))
        @test startswith(s, "INFO  t[main] first\n    │ second\n")
        @test occursin("WARN  oops", s) && occursin("    │ exception =\n    │   ArgumentError: bad value", s)
        @test count("INFO  once", s) == 2
        @test_throws ArgumentError AeolusLogger(buf; timestamps = :bogus)
    end

    @testset "parse_log_level and the test logger" begin
        @test parse_log_level("debug") == Logging.Debug
        @test parse_log_level(" Info ") == Logging.Info
        @test parse_log_level("warning") == parse_log_level(:warn) == Logging.Warn
        @test parse_log_level("error") == Logging.Error
        @test parse_log_level("off") == Logging.AboveMaxLevel
        @test parse_log_level("-500") == Logging.LogLevel(-500)
        @test_throws ArgumentError parse_log_level("verbose")
        withenv("AEOLUS_LOG" => nothing) do
            @test Logging.min_enabled_level(test_logger()) == Logging.Warn
        end
        withenv("AEOLUS_LOG" => "debug") do
            @test Logging.min_enabled_level(test_logger()) == Logging.Debug
        end
        withenv("AEOLUS_LOG" => "") do
            @test Logging.min_enabled_level(test_logger()) == Logging.Warn
        end
        # the test logger shows warnings, not info, on the current stderr
        out = lg_capture_output() do
            with_logger(test_logger(; level = Logging.Warn)) do
                @info "hidden" subsystem = :t
                @warn "shown" subsystem = :t
            end
        end
        @test !occursin("hidden", out) && occursin(r"^\d\d:\d\d:\d\d\.\d{3} WARN  t\[main\] shown", out)
    end

    @testset "the run log: info to the console, debug to run.log, append, close" begin
        dir = mktempdir()
        path = joinpath(dir, "sub", "run.log")
        con = IOBuffer()
        withenv("AEOLUS_LOG" => nothing) do
            with_run_log(path; console = con) do
                @debug "dbg" subsystem = :t k = 1
                @info "inf" subsystem = :t
                @warn "wrn" subsystem = :t
            end
        end
        file = read(path, String)
        c = String(take!(con))
        @test count('\n', file) == 3 && count('\n', c) == 2
        @test occursin(r"^\d{4}-\d\d-\d\dT[\d:.]{12} DEBUG t\[main\] dbg  k=1  @ logging_core\.jl:\d+\n", file)
        @test occursin(r"^\d\d:\d\d:\d\d\.\d{3} INFO  t\[main\] inf\n", c)
        @test !occursin("dbg", c)
        @test !occursin("@ ", split(c, '\n')[1])      # console: no source for info

        # a restart appends; an error is logged at Error level and rethrown,
        # and the log is closed either way
        @test_throws ErrorException with_run_log(path; console = false) do
            @info "second run" subsystem = :t
            error("diverged")
        end
        file = read(path, String)
        @test count('\n', split(file, "second run")[1]) == 3          # first run kept
        @test occursin(r"ERROR runlog\[main\] run stopped by an error", file)
        @test occursin("diverged", file)

        # a tee sends each record only to the children that want it
        a = IOBuffer(); b = IOBuffer()
        tee = TeeLogger(AeolusLogger(a; min_level = Logging.Warn), AeolusLogger(b; min_level = Logging.Debug))
        @test Logging.min_enabled_level(tee) == Logging.Debug
        with_logger(tee) do
            @debug "x"; @warn "y"
        end
        @test count('\n', String(take!(a))) == 1 && count('\n', String(take!(b))) == 2
        lg = open_run_log(joinpath(dir, "r2.log"); console = false, append = false)
        close(lg)
        @test !isopen(lg.loggers[1].stream)
        @test isopen(con)                        # the caller's console stays open
    end

    @testset "logging overhead is negligible next to a step" begin
        # One structured debug record written to run.log (formatted, flushed)
        # against one step of an 8³ case, the smallest the suite runs; the
        # 64×64×32 figure is measured in the PR (#333). The epic's budget is
        # 2 % of run time at the default cadence with a few records a step.
        sim = lg_sim()
        tstep = Inf
        for _ in 1:6
            tstep = min(tstep, @elapsed step!(sim, 1.0))
        end
        path = tempname()
        lg = open_run_log(path; console = false)
        records(n) = with_logger(lg) do
            @elapsed for i in 1:n
                @debug "step" subsystem = :timestepping domain = current_domain() step = i dt = 0.5 cfl = 0.31
            end
        end
        records(10)                               # compile
        trec = minimum(records(200) for _ in 1:3) / 200
        close(lg)
        @test countlines(path) == 10 + 3 * 200
        rm(path)
        # one record (≈10 µs, about half of it Base's logging macro) is
        # cheaper than one step of the smallest case; a regression such as a
        # recompilation or a blocking write per record would break this
        @test trec < tstep
        # a disabled @debug (no logger at Debug) costs nanoseconds
        quiet(n) = with_logger(AeolusLogger(devnull; min_level = Logging.Info)) do
            @elapsed for i in 1:n
                @debug "x" subsystem = :t i
            end
        end
        quiet(10)
        @test quiet(10_000) / 10_000 < 1e-6
    end
end
