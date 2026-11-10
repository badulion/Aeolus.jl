# The launch trace of Backends (#566): thread busy, idle and wait time per
# launch, off unless AEOLUS_LAUNCH_TRACE=1. In this process (one thread, the
# trace off): the off path records nothing and costs nothing measurable, the
# serial path when switched on, and the summary. The threaded paths, nesting
# and the idle accounting run in a `julia -t 4` subprocess with the variable
# set (test/support/launch_trace_worker.jl).

using Test
using Aeolus
const BT = Aeolus.Backends

@testset "launch trace (#566)" begin
    trbody!(i, j, k, a) = (@inbounds a[i, j, k] += 1.0; nothing)
    trplane(k, ir, jr, a) = (s = 0.0; @inbounds for j in jr, i in ir; s += a[i, j, k]; end; s)
    a = zeros(8, 8, 8)
    r = (1:8, 1:8, 1:8)
    b = BT.CPUBackend(serial = true)

    @testset "off: nothing recorded" begin
        @test !BT.launch_trace_enabled()      # AEOLUS_LAUNCH_TRACE is not set for the tests
        BT.reset_launch_trace!()
        BT.launch!(b, trbody!, r, a)
        BT.plane_partials!(zeros(8), b, identity, r, a)
        @test isempty(BT.launch_trace())
        @test all(==(1.0), a)
    end

    @testset "on: serial launches, names, summary" begin
        old = BT.set_launch_trace!(true)
        @test old == false && BT.launch_trace_enabled()
        try
            BT.reset_launch_trace!()
            BT.launch!(b, trbody!, r, a)
            BT.plane_partials!(zeros(8), b, abs, r, a)
            BT.range_map!(b, abs, a, r, a)
            BT.plane_map!(zeros(8), b, trplane, r, a)
            rs = BT.launch_trace()
            @test length(rs) == 4
            @test [x.name for x in rs] == [:trbody!, Symbol("plane_reduce/abs"), Symbol("_map_body!/abs"), :trplane]
            @test all(x -> x.serial && x.depth == 0 && x.tid == [Threads.threadid()], rs)
            # a serial launch: its one thread is busy for (almost) the whole launch
            @test all(x -> x.t0 <= only(x.start) <= only(x.stop) <= x.t1, rs)
            @test all(x -> only(x.busy) <= x.t1 - x.t0, rs)
            s = BT.launch_trace_summary(rs; nthreads = 1)
            @test s.launches == 4 && s.serial_launches == 4 && s.nested == 0
            @test s.launch_wall ≈ sum(x -> (x.t1 - x.t0) * 1e-9, rs)       # (sums of the same numbers)
            @test 0 <= s.idle_in_launches <= s.launch_wall                  # busy ≤ wall at 1 thread
            @test Set(first.(s.by_name)) == Set(x.name for x in rs)
            BT.reset_launch_trace!()
            @test isempty(BT.launch_trace())
        finally
            BT.set_launch_trace!(old)
        end
    end

    @testset "summary of synthetic records" begin
        # two top-level launches on 2 threads and one nested launch:
        # launch 1, 10 ms, threads 1 and 2 busy 10 and 4 ms; launch 2 serial, 5 ms
        ms = 1_000_000
        rs = [BT.LaunchRecord(:k1, 1, 0, 3ms, false, [2], [0], [3ms], [3ms]),
              BT.LaunchRecord(:k1, 0, 0, 10ms, false, [1, 2], [0, 0], [10ms, 4ms], [10ms, 4ms]),
              BT.LaunchRecord(:k2, 0, 20ms, 25ms, true, [1], [20ms], [25ms], [5ms])]
        s = BT.launch_trace_summary(rs; nthreads = 2)
        @test s.launches == 2 && s.nested == 1 && s.serial_launches == 1
        @test s.launch_wall ≈ 0.015 && s.serial_launch_wall ≈ 0.005 && s.nested_wall ≈ 0.003
        @test s.busy ≈ [0.015, 0.004]
        @test s.idle_in_launches ≈ 2 * 0.015 - 0.019                        # 6 ms + 5 ms
        @test first(s.by_name[1]) == :k1 && s.by_name[1].second.idle ≈ 0.006
        @test s.by_name[2].second.idle ≈ 0.005
    end

    @testset "off costs nothing measurable" begin
        # The off path adds one load and one predictable branch to a launch.
        # Measured: a serial 8³ launch against the same loop called directly
        # (`_serial!`, what the launch runs), minimum of 20 batches of 2000
        # calls each (the minimum: other jobs share the node). The bound,
        # 25 ns per launch, is the launch's own bookkeeping (point count,
        # threshold) plus the branch; the body costs ~150 ns.
        z = zeros(8, 8, 8)
        f(n) = (t0 = time_ns(); for _ in 1:n; BT.launch!(b, trbody!, r, z); end; (time_ns() - t0) / n)
        g(n) = (t0 = time_ns(); for _ in 1:n; BT._serial!(trbody!, r, (z,)); end; (time_ns() - t0) / n)
        f(100); g(100)
        tf = minimum(f(2000) for _ in 1:20)
        tg = minimum(g(2000) for _ in 1:20)
        @info "launch trace off: serial 8³ launch $(round(tf; digits = 1)) ns, the loop alone $(round(tg; digits = 1)) ns"
        @test tf - tg < 25
    end

    @testset "threaded paths, nesting, idle (julia -t 4, AEOLUS_LAUNCH_TRACE=1)" begin
        worker = joinpath(@__DIR__, "support", "launch_trace_worker.jl")
        out = tempname()
        project = dirname(Base.active_project())
        cmd = addenv(`$(Base.julia_cmd()) -t 4 --startup-file=no --project=$project $worker $out`,
                     "AEOLUS_LAUNCH_TRACE" => "1", "OPENBLAS_NUM_THREADS" => "1")
        @test success(pipeline(cmd; stdout = devnull, stderr = stderr))
        kv = Dict(first(split(l, ' '; limit = 2)) => last(split(l, ' '; limit = 2)) for l in eachline(out))
        @test kv["nthreads"] == "4" && kv["enabled"] == "true"    # the variable switches it on at load
        # 1. one record per launch, in order, each threaded over 4 tasks but the serial one
        @test kv["basic_count"] == "6"
        @test kv["basic_names"] == "body3!,body2!,plane_reduce/sq,plane!,_map_body!/sq,body3!"
        @test kv["basic_depths"] == "0,0,0,0,0,0"
        @test kv["basic_serial"] == "false,false,false,false,false,true"
        @test kv["basic_records_ok"] == "true"
        @test all(n -> 1 <= n <= 4, parse.(Int, split(kv["basic_nthreads_used"], ",")))
        # 2. one of four work items spins 50 ms: one thread busy 50 ms, the
        # three others idle for the whole launch (≥ 3 × 50 ms; more if the
        # node is loaded, which only lengthens the launch)
        @test parse(Float64, kv["spin_busy_max"]) >= 0.05
        @test parse(Float64, kv["spin_busy_rest"]) < 0.005
        @test parse(Float64, kv["spin_idle"]) >= 3 * 0.05 * 0.95
        @test parse(Float64, kv["spin_wall"]) >= 0.05
        # (1 to 4 threads: only work item 1 has work, so the scheduler may run
        # the three empty ones on any thread, also all on one)
        @test 1 <= parse(Int, kv["spin_threads_used"]) <= 4 && kv["spin_record_ok"] == "true"
        # 3. nesting: four inner launches (depth ≥ 1) end before the outer (depth 0)
        @test kv["nest_count"] == "5"
        @test count(==("0"), split(kv["nest_depths"], ",")) == 1 && endswith(kv["nest_depths"], ",0")
        @test kv["nest_last_name"] == "nested!"
        @test kv["nest_summary"] == "1,4"
        @test kv["nest_records_ok"] == "true" && kv["nest_values_ok"] == "true"
        # 3b. the dynamic column schedule (#537) is traced as one launch
        @test kv["dyn_count"] == "1" && kv["dyn_name"] == "dynamic_columns"
        @test kv["dyn_record_ok"] == "true" && parse(Float64, kv["dyn_busy_max"]) >= 0.03
        off, on = parse(Float64, kv["cost_off"]), parse(Float64, kv["cost_on"])
        @info "launch trace, 64×64×16 launch on 4 threads: off $(round(off * 1e6; digits = 1)) µs, on $(round(on * 1e6; digits = 1)) µs"
        rm(out; force = true)
    end
end
