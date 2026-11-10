# Worker of test/launch_trace.jl (#566): started as
#   AEOLUS_LAUNCH_TRACE=1 julia -t 4 --project=. launch_trace_worker.jl <out>
# It runs the threaded launch paths with the trace on and writes what it
# measured as `key value` lines to <out>; the test asserts on them.
using Aeolus
const B = Aeolus.Backends

out = String[]
put(k, v) = push!(out, string(k, " ", v))
nt = Threads.nthreads()
put("nthreads", nt)
put("enabled", B.launch_trace_enabled())
b = B.CPUBackend(threshold = 0)

# every record: its spans inside the launch, busy within the span, one entry per
# thread. (`busy`: not for a launch whose work items wait for nested launches:
# while one waits, its thread may run another of the launch's work items, and
# both count as busy, so a thread's busy time can exceed its span.)
function check_record(r::B.LaunchRecord; busy = true)
    ok = r.t0 <= r.t1 && allunique(r.tid) && length(r.tid) <= nt &&
         length(r.tid) == length(r.start) == length(r.stop) == length(r.busy)
    for n in eachindex(r.tid)
        ok &= r.t0 <= r.start[n] <= r.stop[n] <= r.t1
        ok &= !busy || r.busy[n] <= r.stop[n] - r.start[n]
    end
    return ok
end

body3!(i, j, k, a) = (@inbounds a[i, j, k] += 1.0; nothing)
body2!(i, j, a) = (@inbounds a[i, j] += 1.0; nothing)
plane!(k, ir, jr, a) = (s = 0.0; @inbounds for j in jr, i in ir; s += a[i, j, k]; end; s)
sq(x) = x * x
function spin!(n, t)               # work item 1 spins t seconds, the others do nothing
    n == 1 || return nothing
    t0 = time_ns()
    while time_ns() - t0 < t * 1e9
    end
    return nothing
end
function nested!(n, arrs)          # each work item launches over its own array
    B.launch!(B.CPUBackend(threshold = 0), body3!, (1:64, 1:64, 1:16), arrs[n])
    return nothing
end

a3 = zeros(64, 64, 16); a2 = zeros(256, 256); pp = zeros(16)
arrs = ntuple(_ -> zeros(64, 64, 16), 4)
r3 = (1:64, 1:64, 1:16)
# compile every path once, then start from an empty trace
for _ in 1:2
    B.launch!(b, body3!, r3, a3); B.launch!(b, body2!, (1:256, 1:256), a2)
    B.plane_partials!(pp, b, sq, r3, a3); B.plane_map!(pp, b, plane!, r3, a3)
    B.range_map!(b, sq, a3, r3, a3)
    B.launch_weighted!(b, 16384, spin!, (1:4,), 0.0)
    B.launch_weighted!(b, 16384, nested!, (1:4,), arrs)
    B.launch!(B.CPUBackend(serial = true), body3!, r3, a3)
end

# 1. one record per launch, named, top level, threaded over 4 tasks
B.reset_launch_trace!()
B.launch!(b, body3!, r3, a3)
B.launch!(b, body2!, (1:256, 1:256), a2)
B.plane_partials!(pp, b, sq, r3, a3)
B.plane_map!(pp, b, plane!, r3, a3)
B.range_map!(b, sq, a3, r3, a3)
B.launch!(B.CPUBackend(serial = true), body3!, r3, a3)
rs = B.launch_trace()
put("basic_count", length(rs))
put("basic_names", join(string.(getfield.(rs, :name)), ","))
put("basic_depths", join(getfield.(rs, :depth), ","))
put("basic_serial", join(getfield.(rs, :serial), ","))
put("basic_nthreads_used", join(length.(getfield.(rs, :tid)), ","))
put("basic_records_ok", all(check_record, rs))

# 2. idle accounting: one work item of four spins 50 ms
B.reset_launch_trace!()
B.launch_weighted!(b, 16384, spin!, (1:4,), 0.05)
s = B.launch_trace_summary()
r = only(B.launch_trace())
put("spin_wall", s.launch_wall)
put("spin_busy_max", maximum(s.busy))
put("spin_busy_rest", sum(s.busy) - maximum(s.busy))
put("spin_idle", s.idle_in_launches)
put("spin_threads_used", length(r.tid))
put("spin_record_ok", check_record(r))

# 3. nesting: four work items, each with a launch of its own
B.reset_launch_trace!()
B.launch_weighted!(b, 16384, nested!, (1:4,), arrs)
rs = B.launch_trace()
s = B.launch_trace_summary(rs)
put("nest_count", length(rs))
put("nest_depths", join(getfield.(rs, :depth), ","))
put("nest_last_name", last(rs).name)
put("nest_summary", "$(s.launches),$(s.nested)")
put("nest_records_ok", all(check_record, rs[1:end-1]) && check_record(rs[end]; busy = false))
put("nest_values_ok", all(a -> all(==(3.0), a), arrs))   # 2 warm-up + 1 traced launch per array

# 4. the cost of the trace on a small threaded launch (4 tasks, 65 536 points)
function per_launch(b, a, n)
    t0 = time_ns()
    for _ in 1:n
        B.launch!(b, body3!, (1:64, 1:64, 1:16), a)
    end
    return (time_ns() - t0) / n * 1e-9
end
B.set_launch_trace!(false); per_launch(b, a3, 50)
off = minimum(per_launch(b, a3, 200) for _ in 1:5)
B.set_launch_trace!(true); per_launch(b, a3, 50)
on = minimum(per_launch(b, a3, 200) for _ in 1:5)
put("cost_off", off)
put("cost_on", on)

write(ARGS[1], join(out, "\n") * "\n")
