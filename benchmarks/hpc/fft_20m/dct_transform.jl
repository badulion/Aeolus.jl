# The pressure solver's DCT (CosineFFTPoissonSolver) on its own, #566 REPORT_fft_20m.md.
# usage: [DIMS=1,2,3] julia -t J,0 dct_transform.jl Nx Ny Nz fftw_threads planner(estimate|measure) callback(aeolus|julia) [reps]
#   callback aeolus: Aeolus's FFTW policy, so FFTW's own pthreads (_fftw_own_threads!, #397)
#   callback julia:  FFTW.jl's default, FFTW's jobs as Julia tasks (spawnloop)
#   DIMS=3: the z-only DCT of the periodic channel; default 1,2,3 the open box.
# Same plan call as src/PressureSolvers.jl:363: plan_r2r(buf, REDFT10, dims; flags).
# Prints time per transform, cores busy, kernel share, context switches per second
# and the OS thread count; writes the plan (fftw_sprint_plan) to plans/.
using FFTW, LinearAlgebra

Nx, Ny, Nz, nt = parse.(Int, ARGS[1:4])
planner = ARGS[5]; cb = ARGS[6]
reps = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 3
flags = planner == "estimate" ? FFTW.ESTIMATE : FFTW.MEASURE

# A bare ccall of fftw_threads_set_callback(C_NULL) here, before any other FFTW
# call, did NOT take effect (same timings and OS thread count as "julia"); going
# through Aeolus's policy does.
if cb == "aeolus"   # the real path: Aeolus's policy (calls _fftw_own_threads!)
    @eval using Aeolus
    Aeolus.Operators.set_fftw_policy!(threads = nt)
end
FFTW.set_num_threads(nt)

function taskstats()
    cs = 0
    for t in readdir("/proc/self/task")
        for l in eachline("/proc/self/task/$t/status")
            if startswith(l, "voluntary_ctxt_switches") || startswith(l, "nonvoluntary_ctxt_switches")
                cs += parse(Int, split(l)[2])
            end
        end
    end
    f = split(read("/proc/self/stat", String))
    return cs, parse(Int, f[14]), parse(Int, f[15])   # ctx switches, utime, stime (ticks)
end
const TICK = 100

a = Array{Float64}(undef, Nx, Ny, Nz); b = similar(a)
Threads.@threads for k in 1:Nz; @inbounds for j in 1:Ny, i in 1:Nx; a[i, j, k] = sin(0.1i + 0.2j + 0.3k); end; end
fill!(b, 0)
tp = @elapsed p = FFTW.plan_r2r(a, FFTW.REDFT10, Tuple(parse.(Int, split(get(ENV, "DIMS", "1,2,3"), ","))); flags = flags)
sp = ccall((:fftw_sprint_plan, FFTW.libfftw3), Ptr{Cchar}, (Ptr{Cvoid},), p.plan); s = unsafe_string(sp); Libc.free(sp)
mul!(b, p, a)   # warm
ts = Float64[]
c0, u0, s0 = taskstats(); w0 = time()
for r in 1:reps
    push!(ts, @elapsed mul!(b, p, a))
end
c1, u1, s1 = taskstats(); w1 = time()
wall = w1 - w0
npts = Nx * Ny * Nz
println("RESULT dims=$(get(ENV, "DIMS", "1,2,3")) Nx=$Nx Ny=$Ny Nz=$Nz pts=$(npts) jt=$(Threads.nthreads()) ft=$nt planner=$planner cb=$cb ",
        "plan_s=$(round(tp, digits=2)) min_s=$(round(minimum(ts), digits=4)) ",
        "ns_per_pt=$(round(1e9 * minimum(ts) / npts, digits=2)) ",
        "cores=$(round((u1 - u0 + s1 - s0) / TICK / wall, digits=2)) ",
        "kernel=$(round(100 * (s1 - s0) / max(1, u1 - u0 + s1 - s0), digits=0))% ",
        "cs_per_s=$(round(Int, (c1 - c0) / wall)) os_threads=$(length(readdir("/proc/self/task")))")
mkpath("plans"); write("plans/dct3_d$(replace(get(ENV, "DIMS", "123"), "," => ""))_$(Nx)x$(Ny)x$(Nz)_ft$(nt)_$(planner)_$(cb).txt", s)
println("PLANHEAD ", join(first(split(s, "\n"), 8), " | "))
